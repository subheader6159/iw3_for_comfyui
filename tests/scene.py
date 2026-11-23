"""Procedural test footage: a ray-traced checkerboard floor with shaded spheres and a camera pan."""

import math
from fractions import Fraction

import av
import numpy as np
import torch

SPHERES = [
    # center (x, y, z), radius, rgb
    ((-1.3, 0.7, -4.0), 0.7, (0.9, 0.25, 0.2)),
    ((0.9, 0.5, -3.0), 0.5, (0.2, 0.6, 0.9)),
    ((0.3, 1.2, -8.0), 1.2, (0.95, 0.85, 0.3)),
]


def render_frame(
    camera_x: float,
    height: int = 256,
    width: int = 448,
    sky: tuple[float, float, float] = (0.55, 0.75, 1.0),
    floor: tuple[float, float, float] = (0.85, 0.85, 0.8),
) -> torch.Tensor:
    """Returns an (H, W, 3) float32 image in 0-1."""
    fov = math.radians(60)
    ys = torch.linspace(1, -1, height).view(-1, 1).expand(height, width)
    xs = torch.linspace(-1, 1, width).view(1, -1).expand(height, width)
    tan = math.tan(fov / 2)
    d = torch.stack([xs * tan * width / height, ys * tan, -torch.ones_like(xs)], dim=-1)
    d = d / d.norm(dim=-1, keepdim=True)
    origin = torch.tensor([camera_x, 1.0, 0.0])
    light = torch.tensor([-0.5, 1.0, 0.6])
    light = light / light.norm()

    t_hit = torch.full((height, width), float("inf"))
    color = torch.zeros(height, width, 3)

    # Floor (y = 0) with a checkerboard that fades with distance.
    t_floor = torch.where(d[..., 1] < -1e-4, -origin[1] / d[..., 1], torch.full_like(t_hit, float("inf")))
    p = origin + d * t_floor.unsqueeze(-1).clamp(max=1e4)
    checker = ((torch.floor(p[..., 0]) + torch.floor(p[..., 2])) % 2).unsqueeze(-1)
    fade = torch.exp(-t_floor.clamp(max=1e4) / 25).unsqueeze(-1)
    floor_color = torch.tensor(floor) * (0.35 + 0.65 * checker) * fade + torch.tensor(sky) * (1 - fade)
    mask = t_floor < t_hit
    t_hit = torch.where(mask, t_floor, t_hit)
    color = torch.where(mask.unsqueeze(-1), floor_color, color)

    for center, radius, rgb in SPHERES:
        oc = origin - torch.tensor(center)
        b = (d * oc).sum(-1)
        c = (oc * oc).sum() - radius * radius
        disc = b * b - c
        t = torch.where(disc > 0, -b - torch.sqrt(disc.clamp(min=0)), torch.full_like(t_hit, float("inf")))
        t = torch.where(t > 0, t, torch.full_like(t, float("inf")))
        mask = t < t_hit
        hit = origin + d * t.clamp(max=1e4).unsqueeze(-1)
        normal = (hit - torch.tensor(center)) / radius
        shade = (normal * light).sum(-1).clamp(min=0).unsqueeze(-1)
        sphere_color = torch.tensor(rgb) * (0.25 + 0.75 * shade)
        t_hit = torch.where(mask, t, t_hit)
        color = torch.where(mask.unsqueeze(-1), sphere_color, color)

    sky_color = torch.tensor(sky) * (0.8 + 0.2 * ys.unsqueeze(-1))
    color = torch.where(torch.isinf(t_hit).unsqueeze(-1), sky_color, color)
    return color.clamp(0, 1).float()


def render_clip(num_frames: int, height: int = 256, width: int = 448, scene_cut_at: int | None = None) -> torch.Tensor:
    """Returns a ComfyUI IMAGE batch (B, H, W, 3). Frames after `scene_cut_at` use a different look."""
    frames = []
    for i in range(num_frames):
        cut = scene_cut_at is not None and i > scene_cut_at
        camera_x = -0.5 + i * 0.03 + (6.0 if cut else 0.0)
        if cut:
            frames.append(render_frame(camera_x, height, width, sky=(1.0, 0.55, 0.35), floor=(0.3, 0.8, 0.35)))
        else:
            frames.append(render_frame(camera_x, height, width))
    return torch.stack(frames, dim=0)


def write_video(path: str, frames: torch.Tensor, fps: int = 24, with_audio: bool = True) -> None:
    """Encode an IMAGE batch to H.264 MP4, optionally with a sine-wave AAC track."""
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16"}
        audio = None
        if with_audio:
            audio = container.add_stream("aac", rate=48000)
            audio.layout = "stereo"

        for frame in frames:
            data = (frame.numpy() * 255).round().astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(data, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

        if audio is not None:
            duration = frames.shape[0] / fps
            samples = int(48000 * duration)
            t = np.arange(samples) / 48000
            wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            chunk = 1024
            for start in range(0, samples, chunk):
                block = np.stack([wave[start : start + chunk]] * 2)
                aframe = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
                aframe.sample_rate = 48000
                aframe.pts = start
                aframe.time_base = Fraction(1, 48000)
                for packet in audio.encode(aframe):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)


def probe_video(path: str) -> dict:
    with av.open(path) as container:
        video = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video=0))
        return {
            "width": video.codec_context.width,
            "height": video.codec_context.height,
            "frames": frames,
            "has_audio": len(container.streams.audio) > 0,
        }


def read_frame(path: str, index: int = 0) -> np.ndarray:
    with av.open(path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(index)
