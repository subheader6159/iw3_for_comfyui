"""Decode a video latent chunk by chunk and encode each finished frame straight to a video file.

Only one chunk of decoded frames is held in memory at a time, so RAM use does not grow with video length.
The temporal chunking and blending mirror ComfyUI's VAE Decode (Tiled) (comfy.utils.tiled_scale_multidim),
so the frames match what that node produces with the same temporal settings, except that each chunk is clamped
to 0-1 before blending rather than after.
"""

import json
import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import av
import numpy as np
import torch
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

BT709_NCL = 1
ProgressFn = Callable[[int, int], None]


def _noop_progress(done: int, total: int) -> None:
    pass


def _scale_fn(value: Any) -> Callable[[int], int]:
    if callable(value):
        return lambda x: int(round(value(x)))
    return lambda x: int(round(value * x))


def temporal_tiles(length: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """(start, length) of each temporal tile in latent frames, same positions as tiled_scale_multidim."""
    if length <= tile:
        return [(0, length)]
    tiles = []
    for it in range(0, length - overlap, tile - overlap):
        pos = max(0, min(length - overlap, it))
        tiles.append((pos, min(tile, length - pos)))
    return tiles


def _feather_weights(n: int, feather: int) -> torch.Tensor:
    w = torch.ones(n)
    if feather < n:
        for t in range(feather):
            a = (t + 1) / feather
            w[t] *= a
            w[n - 1 - t] *= a
    return w


@torch.inference_mode()
def stream_temporal_decode(
    latent: torch.Tensor,
    decode: Callable[[torch.Tensor], torch.Tensor],
    write: Callable[[torch.Tensor], None],
    tile_t: int,
    overlap_t: int,
    upscale_t: Any,
    index_t: Any,
    progress: ProgressFn = _noop_progress,
) -> int:
    """Decode a (B, C, T, H, W) latent in overlapping temporal tiles and write finished frames in order.

    `decode` maps a latent slice to frames (N, H, W, C) in 0-1. `upscale_t` / `index_t` are the VAE's
    temporal upscale ratio and index formula (number or callable). Returns the number of frames written.
    """
    get_scale = _scale_fn(upscale_t)
    get_pos = _scale_fn(index_t)
    T = latent.shape[2]
    tiles = temporal_tiles(T, tile_t, overlap_t)
    total = get_scale(T)
    feather = get_scale(overlap_t)
    written = 0
    done = 0
    for b in range(latent.shape[0]):
        acc: torch.Tensor | None = None  # pending frames [base, base + len)
        wsum: torch.Tensor | None = None
        base = 0
        for i, (pos, length) in enumerate(tiles):
            frames = decode(latent[b : b + 1, :, pos : pos + length]).float().cpu()
            start = get_pos(pos)
            n = min(frames.shape[0], total - start)
            frames = frames[:n]
            weights = _feather_weights(frames.shape[0], feather) if len(tiles) > 1 else torch.ones(n)
            end = start + n
            if acc is None:
                acc = torch.zeros((end - base, *frames.shape[1:]), dtype=torch.float32)
                wsum = torch.zeros(end - base)
            elif end - base > acc.shape[0]:
                grow = end - base - acc.shape[0]
                acc = torch.cat([acc, acc.new_zeros((grow, *acc.shape[1:]))])
                assert wsum is not None
                wsum = torch.cat([wsum, wsum.new_zeros(grow)])
            assert wsum is not None
            acc[start - base : end - base] += frames * weights.view(-1, 1, 1, 1)
            wsum[start - base : end - base] += weights
            del frames

            # Frames before the next tile's start receive no more contributions.
            final = get_pos(tiles[i + 1][0]) if i + 1 < len(tiles) else end
            ready = max(0, final - base)
            if ready:
                write(acc[:ready] / wsum[:ready].view(-1, 1, 1, 1))
                written += ready
                acc = acc[ready:].clone()
                wsum = wsum[ready:].clone()
                base += ready
            done += 1
            progress(done, len(tiles) * latent.shape[0])
    return written


class VideoFileWriter:
    """Incremental H.264 MP4 writer for 0-1 RGB frames, with optional AUDIO muxed at the end."""

    def __init__(
        self,
        path: str,
        fps: float,
        crf: int = 18,
        pix_fmt: str = "yuv420p",
        metadata: dict[str, Any] | None = None,
        audio: dict[str, Any] | None = None,
    ):
        self.path = path
        self.rate = Fraction(round(fps * 1000), 1000)
        self.crf = crf
        self.pix_fmt = pix_fmt
        self.container = av.open(path, mode="w", options={"movflags": "use_metadata_tags+faststart"})
        for key, value in (metadata or {}).items():
            self.container.metadata[key] = json.dumps(value)
        self.stream: Any = None
        self.audio = audio if audio is not None and audio.get("waveform") is not None else None
        self.audio_stream: Any = None
        self.frames = 0

    def _open_stream(self, height: int, width: int) -> None:
        stream = self.container.add_stream("libx264", rate=self.rate)
        stream.width = width - width % 2
        stream.height = height - height % 2
        stream.pix_fmt = self.pix_fmt
        stream.options = {"crf": str(self.crf)}
        stream.codec_context.color_primaries = ColorPrimaries.BT709
        stream.codec_context.color_trc = ColorTrc.BT709
        stream.codec_context.colorspace = BT709_NCL
        stream.codec_context.color_range = ColorRange.MPEG
        self.stream = stream
        if self.audio is not None:
            # All streams must exist before the first packet is written.
            channels = self.audio["waveform"].shape[1]
            self.audio_layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
            self.audio_stream = self.container.add_stream(
                "aac", rate=int(self.audio["sample_rate"]), layout=self.audio_layout
            )

    def write(self, frames: torch.Tensor) -> None:
        """frames: (N, H, W, C) float in 0-1."""
        if self.stream is None:
            self._open_stream(frames.shape[1], frames.shape[2])
        h, w = self.stream.height, self.stream.width
        data = (frames[:, :h, :w, :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        for img in data:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            frame = frame.reformat(format=self.pix_fmt, dst_colorspace=BT709_NCL, dst_color_range=ColorRange.MPEG)
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.frames += 1

    def _write_audio(self) -> None:
        assert self.audio is not None
        stream = self.audio_stream
        layout = self.audio_layout
        sample_rate = int(self.audio["sample_rate"])
        waveform = self.audio["waveform"][0]
        waveform = waveform[:, : math.ceil(sample_rate / self.rate * self.frames)].float().cpu().numpy()
        if layout == "stereo" and waveform.shape[0] != 2:
            waveform = np.repeat(waveform[:1], 2, axis=0)
        chunk = 1024
        for start in range(0, waveform.shape[1], chunk):
            block = np.ascontiguousarray(waveform[:, start : start + chunk])
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in stream.encode(frame):
                self.container.mux(packet)
        for packet in stream.encode():
            self.container.mux(packet)

    def close(self) -> None:
        if self.stream is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            if self.audio_stream is not None:
                self._write_audio()
        self.container.close()

    def abort(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass


class FrameSink:
    """Stand-in for the `output_buffer` tensor of VAEs with chunked IO (`comfy_has_chunked_io`).

    Those VAEs decode with their own temporal chunking and write finished frames in order with
    `output_buffer[:, :, start:end].copy_(pixels)`. The sink forwards each write to `write` as (N, H, W, C)
    frames instead of keeping them, so the decoded video never has to fit in memory.
    """

    def __init__(self, shape: tuple[int, ...], write: Callable[[torch.Tensor], None]):
        if len(shape) != 5 or shape[0] != 1:
            raise ValueError(f"FrameSink supports a single video (1, C, T, H, W), got {tuple(shape)}")
        self.shape = torch.Size(shape)
        self.ndim = 5
        self.write = write
        self.next_frame = 0

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def __getitem__(self, index: Any) -> "_FrameSinkView":
        return _FrameSinkView(self, 0, self.shape[2])[index]

    def copy_(self, src: torch.Tensor) -> "FrameSink":
        _FrameSinkView(self, 0, self.shape[2]).copy_(src)
        return self

    def receive(self, start: int, frames: torch.Tensor) -> None:
        if start != self.next_frame:
            raise RuntimeError(
                f"The VAE wrote frames out of order (frame {start}, expected {self.next_frame}); "
                "it cannot be streamed to a file."
            )
        # (1, C, F, H, W) -> (F, H, W, C)
        self.write(frames[0].movedim(0, -1).float().cpu())
        self.next_frame += frames.shape[2]


class _FrameSinkView:
    def __init__(self, sink: FrameSink, start: int, stop: int):
        self.sink = sink
        self.start = start
        self.stop = stop
        self.shape = torch.Size((1, sink.shape[1], stop - start, *sink.shape[3:]))
        self.ndim = 5

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def __getitem__(self, index: Any) -> "_FrameSinkView":
        if not isinstance(index, tuple):
            index = (index,)
        index = index + (slice(None),) * (5 - len(index))
        for d in (0, 1, 3, 4):
            s = index[d]
            if not (isinstance(s, slice) and s.step in (None, 1) and (s.start or 0) == 0):
                raise RuntimeError(f"Unsupported output_buffer index {index}; only frame ranges can be streamed.")
            if s.stop is not None and s.stop < self.shape[d]:
                raise RuntimeError(f"Unsupported output_buffer index {index}; only frame ranges can be streamed.")
        f = index[2]
        if not (isinstance(f, slice) and f.step in (None, 1)):
            raise RuntimeError(f"Unsupported output_buffer frame index {f}.")
        length = self.stop - self.start
        lo, hi, _ = f.indices(length)
        return _FrameSinkView(self.sink, self.start + lo, self.start + max(lo, hi))

    def copy_(self, src: torch.Tensor) -> "_FrameSinkView":
        if src.shape[2] != self.stop - self.start:
            frames = self.stop - self.start
            raise RuntimeError(f"output_buffer write of {src.shape[2]} frames into a {frames} frame range")
        self.sink.receive(self.start, src)
        return self
