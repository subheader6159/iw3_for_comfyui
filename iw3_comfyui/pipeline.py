import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from iw3.base_depth_model import BaseDepthModel
from iw3.hub_dir import HUB_MODEL_DIR
from iw3.utils import apply_stereo, postprocess_image
from nunif.utils.transnetv2 import TransNetV2
from nunif.utils.ui import TorchHubDir

ProgressFn = Callable[[int, int], None]


@dataclass
class IW3Depth:
    """Normalized depth for a batch of frames, as produced by iw3 (near = 1.0)."""

    depth: torch.Tensor  # (B, 1, h, w), float16, CPU
    is_metric: bool
    is_video: bool
    scene_boundaries: set[int] = field(default_factory=set)

    def __len__(self) -> int:
        return int(self.depth.shape[0])


def _noop_progress(done: int, total: int) -> None:
    pass


def images_to_bchw(images: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE (B, H, W, C) -> RGB (B, 3, H, W)."""
    if images.ndim != 4:
        raise ValueError(f"Expected an IMAGE batch with shape (B, H, W, C), got {tuple(images.shape)}")
    return images[..., :3].permute(0, 3, 1, 2)


def depth_to_image(depth: IW3Depth, size: tuple[int, int] | None = None, batch_size: int = 16) -> torch.Tensor:
    """IW3Depth -> grayscale ComfyUI IMAGE (B, H, W, 3), optionally resized to `size` (H, W)."""
    B = len(depth)
    h, w = size if size is not None else depth.depth.shape[-2:]
    out = torch.empty((B, h, w, 3), dtype=torch.float32)
    for start in range(0, B, batch_size):
        d = depth.depth[start : start + batch_size].float()
        if d.shape[-2:] != (h, w):
            d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        out[start : start + batch_size] = d.clamp(0, 1).permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
    return out


def image_to_depth(
    images: torch.Tensor,
    is_video: bool,
    invert: bool = False,
    normalize: str = "per_frame",
) -> IW3Depth:
    """Grayscale depth IMAGE (near = bright) -> IW3Depth."""
    depth = images_to_bchw(images).float().mean(dim=1, keepdim=True)
    if invert:
        depth = 1.0 - depth
    if normalize == "per_frame":
        min_v = depth.amin(dim=(1, 2, 3), keepdim=True)
        max_v = depth.amax(dim=(1, 2, 3), keepdim=True)
        depth = (depth - min_v) / (max_v - min_v).clamp(min=1e-6)
    elif normalize == "whole_batch":
        min_v, max_v = depth.amin(), depth.amax()
        depth = (depth - min_v) / (max_v - min_v).clamp(min=1e-6)
    elif normalize != "none":
        raise ValueError(f"Unknown normalize mode: {normalize}")
    return IW3Depth(depth=depth.clamp(0, 1).to(torch.float16), is_metric=False, is_video=is_video)


@torch.inference_mode()
def detect_scene_boundaries(
    images: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    window_size: int = 100,
    padding_size: int = 25,
) -> set[int]:
    """TransNetV2 shot boundary detection on an in-memory frame batch.

    Mirrors nunif.utils.shot_boundary_detection.detect_boundary(). Returned indices are the last frame of a scene.
    """
    B = images.shape[0]
    if B < 2:
        return set()

    with TorchHubDir(HUB_MODEL_DIR):
        model = TransNetV2().load().eval().to(device)

    small = []
    for start in range(0, B, 64):
        x = images_to_bchw(images[start : start + 64]).to(device, torch.float32)
        small.append(F.interpolate(x, size=(27, 48), mode="area"))
    frames = torch.cat(small, dim=0)

    step = window_size - padding_size * 2
    num_windows = math.ceil(B / step)
    tail = num_windows * step + padding_size - B
    padded = torch.cat(
        [
            frames[:1].expand(padding_size, -1, -1, -1),
            frames,
            frames[-1:].expand(tail, -1, -1, -1),
        ],
        dim=0,
    )
    preds = []
    for i in range(num_windows):
        window = padded[i * step : i * step + window_size]
        single_frame_pred, _ = model(window)
        single_frame_pred = torch.sigmoid(single_frame_pred).flatten()
        preds.append(single_frame_pred[padding_size:-padding_size].cpu())
    frame_preds = torch.cat(preds, dim=0)[:B]
    del model

    # The last frame cannot start a new scene.
    return {i for i in torch.nonzero(frame_preds > threshold).flatten().tolist() if i < B - 1}


@torch.inference_mode()
def estimate_depth(
    images: torch.Tensor,
    depth_model: BaseDepthModel,
    args: argparse.Namespace,
    is_video: bool,
    scene_boundaries: set[int] | None = None,
    progress: ProgressFn = _noop_progress,
) -> torch.Tensor:
    """Run iw3 depth estimation + normalization on a frame batch. Returns (B, 1, h, w) float16 on CPU.

    Video mode follows iw3's batch video callback (EMA normalization, streaming models, scene resets).
    Image mode follows iw3's process_image() (each frame independent).
    """
    B = images.shape[0]
    device = depth_model.device
    scene_boundaries = scene_boundaries or set()
    infer_kwargs: dict[str, Any] = dict(
        tta=args.tta,
        low_vram=args.low_vram,
        enable_amp=not args.disable_amp,
        edge_dilation=args.edge_dilation,
        depth_aa=args.depth_aa,
    )
    flush_kwargs: dict[str, Any] = dict(
        enable_amp=not args.disable_amp,
        edge_dilation=args.edge_dilation,
        depth_aa=args.depth_aa,
    )
    results: list[torch.Tensor] = []

    def collect(depths: list[torch.Tensor]) -> None:
        for d in depths:
            results.append(d.to("cpu", torch.float16))

    def strip_mask(out: Any) -> torch.Tensor:
        # Matting models return (depth, mask). The mask is not used here.
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    depth_model.reset()
    if is_video and args.ema_normalize:
        depth_model.enable_ema(decay=args.ema_decay, buffer_size=args.ema_buffer)
    else:
        depth_model.disable_ema()

    if not is_video:
        for i in range(B):
            x = images_to_bchw(images[i : i + 1])[0].to(device, torch.float32)
            depth = strip_mask(depth_model.infer(x, **infer_kwargs))
            collect([depth_model.minmax_normalize_chw(depth)])
            progress(i + 1, B)
        return torch.stack(results, dim=0)

    batch_size = max(args.batch_size // 2, 1) if args.tta else args.batch_size
    streaming = depth_model.is_streaming()
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        x = images_to_bchw(images[start:end]).to(device, torch.float32)
        if streaming:
            for j in range(x.shape[0]):
                # Streaming models return None while they buffer frames.
                streamed = depth_model.infer(x[j : j + 1], streaming=True, **infer_kwargs)
                collect(depth_model.minmax_normalize(streamed))
                if start + j in scene_boundaries:
                    collect(depth_model.minmax_normalize(depth_model.flush(**flush_kwargs)))
                    collect(depth_model.flush_minmax_normalize())
                    depth_model.reset()
        else:
            depth = strip_mask(depth_model.infer(x, **infer_kwargs))
            reset_ema = [i in scene_boundaries for i in range(start, end)]
            collect(depth_model.minmax_normalize(depth, reset_ema=reset_ema))
        progress(end, B)

    if streaming:
        collect(depth_model.minmax_normalize(depth_model.flush(**flush_kwargs)))
    collect(depth_model.flush_minmax_normalize())
    depth_model.reset()

    if len(results) != B:
        raise RuntimeError(f"Depth estimation returned {len(results)} frames for {B} input frames")
    return torch.stack(results, dim=0)


@torch.inference_mode()
def generate_stereo(
    images: torch.Tensor,
    depth: IW3Depth,
    side_model: Any,
    args: argparse.Namespace,
    progress: ProgressFn = _noop_progress,
) -> torch.Tensor:
    """Warp frames into stereo pairs with iw3 and pack them into the requested format. Returns IMAGE on CPU."""
    B = images.shape[0]
    if len(depth) != B:
        raise ValueError(f"Frame count mismatch: {B} images but {len(depth)} depth maps")
    device = args.state["device"]
    is_video = depth.is_video and B > 1

    if side_model is not None and hasattr(side_model, "set_mode"):
        side_model.set_mode("video" if is_video else "image")
        side_model.reset()
    if args.state["convergence_model"] is not None:
        args.state["convergence_model"].reset(enable_ema=is_video)

    output: torch.Tensor | None = None
    count = 0

    def emit(left_eyes: torch.Tensor | None, right_eyes: torch.Tensor | None) -> None:
        nonlocal output, count
        if left_eyes is None or right_eyes is None:
            return
        for i in range(left_eyes.shape[0]):
            frame = postprocess_image(left_eyes[i], right_eyes[i], args)
            if output is None:
                output = torch.empty((B, frame.shape[1], frame.shape[2], frame.shape[0]), dtype=torch.float32)
            if count >= B:
                raise RuntimeError("Stereo generation produced more frames than its input")
            output[count] = frame.permute(1, 2, 0).float().cpu()
            count += 1

    for start in range(0, B, args.batch_size):
        end = min(start + args.batch_size, B)
        x = images_to_bchw(images[start:end]).to(device, torch.float32)
        d = depth.depth[start:end].to(device, torch.float32)
        reset_pts = [i in depth.scene_boundaries for i in range(start, end)] if is_video else None
        left_eyes, right_eyes, _, _ = apply_stereo(x, d, None, args, side_model, reset_pts=reset_pts)
        emit(left_eyes, right_eyes)
        progress(end, B)

    if is_video and hasattr(side_model, "flush"):
        ret = side_model.flush(enable_amp=not args.disable_amp)
        emit(ret[0], ret[1])

    if output is None or count != B:
        raise RuntimeError(f"Stereo generation returned {count} frames for {B} input frames")
    return output
