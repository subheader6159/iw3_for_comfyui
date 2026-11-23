import argparse
from dataclasses import dataclass, field
from typing import Any

import torch
from iw3.mapper import resolve_mapper_name
from iw3.utils import calc_auto_warp_steps, create_parser, set_state_args

STEREO_METHODS = [
    "mlbw_l2s",
    "mlbw_l2",
    "mlbw_l4s",
    "mlbw_l4",
    "mlbw_l2_inpaint",
    "row_flow_v3",
    "row_flow_v3_sym",
    "row_flow_v2",
    "forward_fill",
    "forward_inpaint",
    "monobw",
    "monobw_inpaint",
    "backward",
    "forward",
]
INPAINT_METHODS = {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}

STEREO_FORMATS: dict[str, dict[str, Any]] = {
    "Full SBS": {},
    "Half SBS": {"half_sbs": True},
    "Full TB": {"tb": True},
    "Half TB": {"half_tb": True},
    "VR90": {"vr180": True},
    "Cross Eyed": {"cross_eyed": True},
    "RGB-D": {"rgbd": True},
    "Half RGB-D": {"half_rgbd": True},
    "Anaglyph": {},  # anaglyph method is set separately
}
ANAGLYPH_METHODS = ["dubois", "dubois2", "color", "gray", "half-color", "wimmer", "wimmer2"]
PAD_MODES = ["tblr", "tb", "lr", "16:9", "1:1", "top"]
CONVERGENCE_MODES = ["constant", "sod_v1"]
SYNTHETIC_VIEWS = ["both", "right", "left"]


@dataclass
class StereoOptions:
    method: str = "mlbw_l2s"
    divergence: float = 2.0
    convergence: float = 0.25
    foreground_scale: float = 0.0
    stereo_format: str = "Full SBS"
    anaglyph_method: str = "dubois"
    convergence_mode: str = "constant"
    synthetic_view: str = "both"
    ipd_offset: float = 0.0
    preserve_screen_border: bool = False
    warp_steps: int = 0
    stereo_width: int = 0
    inpaint_model: str = "light_inpaint_v1"
    inpaint_overlap_frames: int = 3
    mask_inner_dilation: int = 0
    mask_outer_dilation: int = 0
    inpaint_max_width: int = 0
    pad: float = 0.0
    pad_mode: str = "tblr"
    max_output_width: int = 0
    max_output_height: int = 0
    keep_aspect_ratio: bool = False


@dataclass
class DepthOptions:
    batch_size: int = 4
    flicker_reduction: bool = True
    ema_decay: float = 0.9
    ema_buffer: int = 30
    edge_dilation: int = 2
    edge_dilation_y: int = 1
    depth_aa: bool = False
    tta: bool = False
    low_vram: bool = False


@dataclass
class VideoOptions:
    max_fps: float = 30.0
    scene_detect: bool = False
    video_codec: str = "auto"
    crf: int = 20
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    extra: dict[str, Any] = field(default_factory=dict)


def device_to_gpu_id(device: torch.device) -> int:
    if device.type == "cpu":
        return -1
    return device.index or 0


def uses_stereo_model(stereo: StereoOptions) -> bool:
    return stereo.stereo_format not in {"RGB-D", "Half RGB-D"}


def effective_divergence(stereo: StereoOptions) -> float:
    # A single synthetic view has to cover the whole disparity.
    return stereo.divergence * (2.0 if stereo.synthetic_view in {"right", "left"} else 1.0)


def build_args(
    device: torch.device,
    stereo: StereoOptions,
    depth: DepthOptions | None = None,
    video: VideoOptions | None = None,
    output: str = "",
) -> argparse.Namespace:
    """Build an iw3 argparse namespace from node options, starting from iw3's CLI defaults."""
    if stereo.stereo_format not in STEREO_FORMATS:
        raise ValueError(f"Unknown stereo format: {stereo.stereo_format}")
    if stereo.method not in STEREO_METHODS:
        raise ValueError(f"Unknown stereo method: {stereo.method}")

    args = create_parser(required_true=False).parse_args([])
    values: dict[str, Any] = dict(
        gpu=[device_to_gpu_id(device)],
        output=output,
        yes=True,
        method=stereo.method,
        divergence=float(stereo.divergence),
        convergence=float(stereo.convergence),
        foreground_scale=float(stereo.foreground_scale),
        convergence_mode=stereo.convergence_mode,
        synthetic_view=stereo.synthetic_view,
        ipd_offset=float(stereo.ipd_offset),
        preserve_screen_border=bool(stereo.preserve_screen_border),
        warp_steps=stereo.warp_steps if stereo.warp_steps > 0 else None,
        stereo_width=stereo.stereo_width if stereo.stereo_width > 0 else None,
        mask_inner_dilation=int(stereo.mask_inner_dilation),
        mask_outer_dilation=int(stereo.mask_outer_dilation),
        inpaint_max_width=stereo.inpaint_max_width if stereo.inpaint_max_width > 0 else None,
        pad=stereo.pad if stereo.pad > 0 else None,
        pad_mode=stereo.pad_mode,
        max_output_width=stereo.max_output_width if stereo.max_output_width > 0 else None,
        max_output_height=stereo.max_output_height if stereo.max_output_height > 0 else None,
        keep_aspect_ratio=bool(stereo.keep_aspect_ratio),
    )
    if stereo.method in INPAINT_METHODS:
        values["inpaint_model"] = stereo.inpaint_model
        values["inpaint_overlap_frames"] = [int(stereo.inpaint_overlap_frames)]
    values.update(STEREO_FORMATS[stereo.stereo_format])
    if stereo.stereo_format == "Anaglyph":
        values["anaglyph"] = stereo.anaglyph_method

    if depth is not None:
        values.update(
            batch_size=int(depth.batch_size),
            ema_normalize=bool(depth.flicker_reduction),
            ema_decay=float(depth.ema_decay),
            ema_buffer=int(depth.ema_buffer),
            edge_dilation=[int(depth.edge_dilation), int(depth.edge_dilation_y)],
            depth_aa=bool(depth.depth_aa),
            tta=bool(depth.tta),
            low_vram=bool(depth.low_vram),
        )
    if video is not None:
        values.update(
            max_fps=float(video.max_fps),
            scene_detect=bool(video.scene_detect),
            video_codec=None if video.video_codec == "auto" else video.video_codec,
            crf=int(video.crf),
            preset=video.preset,
            pix_fmt=video.pix_fmt,
        )
        values.update(video.extra)

    for key, value in values.items():
        if not hasattr(args, key):
            raise AttributeError(f"iw3 has no option `{key}`")
        setattr(args, key, value)
    return args


def finalize_args(
    args: argparse.Namespace,
    depth_model: Any,
    is_metric: bool,
    stop_event: Any = None,
    tqdm_fn: Any = None,
) -> argparse.Namespace:
    """Apply the parts of iw3_main()/set_state_args() that resolve derived options."""
    if depth_model is None:
        from iw3.depth_model_factory import create_depth_model

        args.depth_model = "NULL"
        depth_model = create_depth_model("NULL")
    else:
        args.depth_model = depth_model.model_type
    set_state_args(args, stop_event=stop_event, tqdm_fn=tqdm_fn, depth_model=depth_model)
    if args.warp_steps is None:
        args.warp_steps = calc_auto_warp_steps(
            method=args.method, divergence=args.divergence, synthetic_view=args.synthetic_view
        )
    args.mapper = resolve_mapper_name(
        mapper=None,
        foreground_scale=args.foreground_scale,
        metric_depth=is_metric,
        mapper_type=args.mapper_type,
    )
    return args
