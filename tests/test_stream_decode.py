import os
import urllib.request

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scene import probe_video, read_frame, render_clip

WAN_VAE_URL = (
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
)


def run(node_cls, **kwargs):
    return node_cls.execute(**kwargs)


@pytest.fixture(scope="module")
def stream(nodes):
    import sys

    return sys.modules[nodes.__name__.rsplit(".", 1)[0] + ".stream_decode"]


def _fake_causal_decode(x: torch.Tensor) -> torch.Tensor:
    """Stand-in for a causal video VAE: (1, C, t, h, w) -> (1, 3, 4t-3, 8h, 8w)."""
    t = x.shape[2]
    idx = torch.tensor([(f + 3) // 4 for f in range(4 * t - 3)])
    out = x[:, :3, idx]
    # Make each frame depend on its position inside the slice, so tile placement errors show up.
    ramp = torch.linspace(0, 0.1, out.shape[2]).view(1, 1, -1, 1, 1)
    out = (out + ramp).clamp(0, 1)
    return F.interpolate(out, scale_factor=(1, 8, 8), mode="nearest")


@pytest.mark.parametrize(
    "T,tile,overlap",
    [(5, 16, 1), (17, 4, 1), (17, 4, 2), (23, 6, 3), (40, 16, 4), (9, 2, 1)],
)
def test_stream_matches_comfy_tiled_decode(stream, T, tile, overlap):
    import comfy.utils

    upscale = (lambda a: max(0, a * 4 - 3), 8, 8)
    index = (4, 8, 8)
    latent = torch.rand(2, 4, T, 3, 5)
    expected = comfy.utils.tiled_scale_multidim(
        latent,
        _fake_causal_decode,
        tile=(tile, 999, 999),
        overlap=(overlap, 1, 1),
        upscale_amount=upscale,
        out_channels=3,
        index_formulas=index,
    )
    expected = expected.movedim(1, -1).reshape(-1, 24, 40, 3)

    chunks = []
    written = stream.stream_temporal_decode(
        latent,
        lambda x: _fake_causal_decode(x).movedim(1, -1).reshape(-1, 24, 40, 3),
        chunks.append,
        tile_t=tile,
        overlap_t=overlap,
        upscale_t=upscale[0],
        index_t=index[0],
    )
    got = torch.cat(chunks)
    assert written == got.shape[0] == expected.shape[0] == 2 * (4 * T - 3)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)
    # Never more than one tile of frames is pending at a time.
    assert max(c.shape[0] for c in chunks) <= max(4 * tile, 4 * T - 3)


class _FakeImageVAE:
    """Image VAE stand-in: (N, C, h, w) -> (N, 8h, 8w, 3), no temporal compression."""

    def spacial_compression_decode(self):
        return 8

    def temporal_compression_decode(self):
        return None

    def decode(self, x):
        return F.interpolate(x[:, :3], scale_factor=8, mode="nearest").movedim(1, -1)


def test_image_vae_batch(nodes, tmp_path):
    latent = torch.rand(10, 4, 8, 12)
    out = run(
        nodes.IW3StreamDecodeVideo,
        samples={"samples": latent},
        vae=_FakeImageVAE(),
        fps=12.0,
        filename_prefix="iw3_test/stream_image",
        temporal_size=8,
    )
    (video,) = out.args
    info = probe_video(video.get_stream_source())
    assert (info["width"], info["height"], info["frames"]) == (96, 64, 10)
    os.unlink(video.get_stream_source())


@pytest.fixture(scope="module")
def wan_vae(require_gpu):
    import comfy.sd
    import comfy.utils
    import folder_paths

    path = os.path.join(folder_paths.models_dir, "iw3", "test_cache", "wan_2.1_vae.safetensors")
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(WAN_VAE_URL, path + ".part")
        os.replace(path + ".part", path)
    return comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))


@pytest.fixture(scope="module")
def wan_clip_latent(wan_vae):
    clip = render_clip(37, 256, 448)  # 4 * 10 - 3 frames
    with torch.inference_mode():
        latent = wan_vae.encode(clip)
    assert latent.shape[2] == 10
    return clip, latent


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-10)))


def test_wan_vae_stream_decode(nodes, stream, wan_vae, wan_clip_latent, artifacts_dir):
    clip, latent = wan_clip_latent
    sr = 48000
    audio = {"waveform": torch.zeros(1, 2, sr * 2), "sample_rate": sr}
    out = run(
        nodes.IW3StreamDecodeVideo,
        samples={"samples": latent},
        vae=wan_vae,
        fps=24.0,
        filename_prefix="iw3_test/stream_wan",
        audio=audio,
        temporal_size=16,
        temporal_overlap=8,
    )
    (video,) = out.args
    path = video.get_stream_source()
    info = probe_video(path)
    assert (info["width"], info["height"], info["frames"]) == (448, 256, 37)
    assert info["has_audio"]

    # Compare with ComfyUI's own decoders.
    with torch.inference_mode():
        tiled = wan_vae.decode_tiled(latent, tile_x=999, tile_y=999, overlap=8, tile_t=4, overlap_t=2)
    tiled = tiled.reshape(-1, *tiled.shape[-3:])
    for i in (0, 18, 36):
        frame = read_frame(path, i)
        to8 = lambda t: (t[i].float().clamp(0, 1) * 255).round().byte().cpu().numpy()  # noqa: E731
        assert _psnr(frame, to8(tiled)) > 33, i  # H.264 (crf 18) loss only
        assert _psnr(frame, (clip[i].numpy() * 255).round().astype(np.uint8)) > 24, i

    # Before encoding, the streamed frames match ComfyUI's VAE Decode (Tiled) with the same temporal settings.
    # They are not bit-identical: each chunk is clamped to 0-1 before blending instead of after.
    chunks = []
    with torch.inference_mode():
        stream.stream_temporal_decode(
            latent,
            lambda x: wan_vae.decode(x).reshape(-1, 256, 448, 3),
            chunks.append,
            tile_t=4,
            overlap_t=2,
            upscale_t=wan_vae.upscale_ratio[0],
            index_t=wan_vae.upscale_index_formula[0],
        )
    diff = (torch.cat(chunks) - tiled.float().cpu()).abs()
    assert diff.mean() < 1e-3 and (diff > 2e-3).float().mean() < 0.02
    from PIL import Image

    Image.fromarray(read_frame(path, 18)).save(os.path.join(artifacts_dir, "stream_decode_frame.png"))


def test_stream_decode_into_convert_video(nodes, wan_vae, wan_clip_latent):
    _, latent = wan_clip_latent
    (video,) = run(
        nodes.IW3StreamDecodeVideo,
        samples={"samples": latent},
        vae=wan_vae,
        fps=24.0,
        filename_prefix="iw3_test/stream_to_iw3",
    ).args
    (depth_model,) = nodes.IW3LoadDepthModel.execute(depth_model="Any_V2_S").args
    (sbs,) = nodes.IW3ConvertVideo.execute(
        video=video,
        depth_model=depth_model,
        method="mlbw_l2s",
        divergence=2.0,
        convergence=0.25,
        foreground_scale=0.0,
        stereo_format="Full SBS",
        flicker_reduction=True,
    ).args
    info = probe_video(sbs.get_stream_source())
    assert (info["width"], info["height"], info["frames"]) == (896, 256, 37)


class _FakeRTXVideoSuperResolution:
    """Mirrors the execute() signature and output sizing of comfyui_nvidia_rtx_nodes' RTXVideoSuperResolution,
    which needs nvidia-vfx and a newer RTX GPU than the test machine has."""

    calls: list[int] = []

    @classmethod
    def execute(cls, images, resize_type, quality, strength=1.0, image_encoding="8-bit RGB"):
        from comfy_api.latest import io

        assert resize_type["resize_type"] == "scale by multiplier"
        assert quality in ("ULTRA", "HIGH") and 0 <= strength <= 1 and image_encoding == "8-bit RGB"
        cls.calls.append(images.shape[0])
        b, h, w, _ = images.shape
        out_h = max(8, round(int(h * resize_type["scale"]) / 8) * 8)
        out_w = max(8, round(int(w * resize_type["scale"]) / 8) * 8)
        up = F.interpolate(images.movedim(-1, 1), size=(out_h, out_w), mode="bilinear", align_corners=False)
        return io.NodeOutput(up.movedim(1, -1))


def test_rtx_stream_decode(nodes, wan_vae, wan_clip_latent, monkeypatch, artifacts_dir):
    import nodes as comfy_nodes

    _, latent = wan_clip_latent
    _FakeRTXVideoSuperResolution.calls = []
    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "RTXVideoSuperResolution", _FakeRTXVideoSuperResolution)
    (video,) = run(
        nodes.IW3StreamDecodeVideoRTX,
        samples={"samples": latent},
        vae=wan_vae,
        fps=24.0,
        filename_prefix="iw3_test/stream_rtx",
        rtx_scale=1.5,
        rtx_quality="ULTRA",
        temporal_size=16,
        temporal_overlap=8,
    ).args
    info = probe_video(video.get_stream_source())
    assert (info["width"], info["height"], info["frames"]) == (672, 384, 37)
    # Upscaled chunk by chunk, never the whole clip at once.
    assert len(_FakeRTXVideoSuperResolution.calls) > 1
    assert sum(_FakeRTXVideoSuperResolution.calls) == 37
    assert max(_FakeRTXVideoSuperResolution.calls) < 37


def test_rtx_stream_decode_without_rtx_nodes(nodes, monkeypatch):
    import nodes as comfy_nodes

    monkeypatch.delitem(comfy_nodes.NODE_CLASS_MAPPINGS, "RTXVideoSuperResolution", raising=False)
    with pytest.raises(RuntimeError, match="comfyui_nvidia_rtx_nodes"):
        run(
            nodes.IW3StreamDecodeVideoRTX,
            samples={"samples": torch.rand(1, 16, 2, 4, 4)},
            vae=None,
            fps=24.0,
            filename_prefix="iw3_test/stream_rtx_missing",
            rtx_scale=2.0,
            rtx_quality="ULTRA",
        )
