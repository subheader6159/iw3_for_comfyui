import os
import threading
import uuid
from typing import Any

import comfy.model_management as mm
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import ComfyExtension, InputImpl, io, ui
from tqdm import tqdm
from typing_extensions import override

from .env import setup_nunif

folder_paths.add_model_folder_path("iw3", os.path.join(folder_paths.models_dir, "iw3"))
setup_nunif(folder_paths.models_dir)

from iw3 import models as _iw3_models  # noqa: E402, F401  (registers iw3 model classes)
from iw3.inpaint_utils import INPAINT_MODELS  # noqa: E402

from .models import (  # noqa: E402
    DEFAULT_DEPTH_MODEL,
    DEPTH_MODEL_TYPES,
    DepthModelHandle,
    prepare_vram,
    release_vram,
    stereo_models,
)
from .options import (  # noqa: E402
    ANAGLYPH_METHODS,
    CONVERGENCE_MODES,
    PAD_MODES,
    STEREO_FORMATS,
    STEREO_METHODS,
    SYNTHETIC_VIEWS,
    DepthOptions,
    StereoOptions,
    VideoOptions,
    build_args,
    effective_divergence,
    finalize_args,
    uses_stereo_model,
)
from .pipeline import (  # noqa: E402
    IW3Depth,
    depth_to_image,
    detect_scene_boundaries,
    estimate_depth,
    generate_stereo,
    image_to_depth,
)
from .video import convert_video_file, resolve_video_source  # noqa: E402

CATEGORY = "iw3"
# Advanced inputs are optional so API prompts and older workflows may omit them; omitted values use the
# defaults of the option dataclasses / execute() signatures.
ADVANCED: dict[str, Any] = {"advanced": True, "optional": True}
DepthModelIO = io.Custom("IW3_DEPTH_MODEL")
DepthIO = io.Custom("IW3_DEPTH")

BATCH_MODES = ["video", "images"]
VIDEO_CODECS = ["auto", "libx264", "libx265", "h264_nvenc", "hevc_nvenc", "libopenh264"]
VIDEO_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
PIX_FMTS = ["yuv420p", "yuv444p", "yuv420p10le"]
DEPTH_NORMALIZE_MODES = ["per_frame", "whole_batch", "none"]


def _progress_callback(total: int):
    pbar = comfy.utils.ProgressBar(total)

    def progress(done: int, total: int) -> None:
        pbar.update_absolute(done, total)
        mm.throw_exception_if_processing_interrupted()

    return progress


class _ComfyTqdm(tqdm):
    """tqdm that mirrors progress to the ComfyUI progress bar and stops iw3 when the user cancels."""

    stop_event: threading.Event | None = None

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._comfy_pbar = comfy.utils.ProgressBar(int(self.total or 0))

    def update(self, n: float | None = 1) -> bool | None:
        ret = super().update(n)
        if self.total:
            self._comfy_pbar.update_absolute(min(int(self.n), int(self.total)), int(self.total))
        if mm.processing_interrupted() and self.stop_event is not None:
            self.stop_event.set()
        return ret


def _make_tqdm_fn(stop_event: threading.Event) -> type[tqdm]:
    return type("ComfyTqdm", (_ComfyTqdm,), {"stop_event": stop_event})


def _stereo_inputs() -> list[io.Input]:
    return [
        io.Combo.Input(
            "method",
            options=STEREO_METHODS,
            default="mlbw_l2s",
            tooltip=(
                "Stereo generation method. mlbw_* are the recommended learned warping models; "
                "*_inpaint methods fill disoccluded areas with an inpainting model; "
                "row_flow_* are older learned models; forward_fill/backward are classic warps."
            ),
        ),
        io.Float.Input(
            "divergence",
            display_name="3D strength",
            default=2.0,
            min=0.0,
            max=10.0,
            step=0.1,
            tooltip="Strength of the 3D effect (iw3 divergence). 1-5 is a reasonable range.",
        ),
        io.Float.Input(
            "convergence",
            default=0.25,
            min=0.0,
            max=1.0,
            step=0.05,
            tooltip=(
                "Position of the screen plane. 0 puts everything behind the screen, "
                "1 makes everything pop out of the screen."
            ),
        ),
        io.Float.Input(
            "foreground_scale",
            default=0.0,
            min=-3.0,
            max=3.0,
            step=0.1,
            tooltip="Scales the depth of foreground objects. Positive values exaggerate the foreground.",
        ),
        io.Combo.Input(
            "stereo_format",
            options=list(STEREO_FORMATS),
            default="Full SBS",
            tooltip=(
                "Output layout. VR90 is an equirectangular VR180 frame (image spans 90 degrees). "
                "Most VR players detect the layout from the filename, e.g. `_LRF_Full_SBS` or `_180x180_LR`."
            ),
        ),
        io.Combo.Input("anaglyph_method", options=ANAGLYPH_METHODS, default="dubois", **ADVANCED),
        io.Combo.Input(
            "convergence_mode",
            options=CONVERGENCE_MODES,
            default="constant",
            **ADVANCED,
            tooltip="sod_v1 estimates the convergence plane per frame from the most salient object.",
        ),
        io.Combo.Input(
            "synthetic_view",
            options=SYNTHETIC_VIEWS,
            default="both",
            **ADVANCED,
            tooltip="Which eye is synthesized. `right`/`left` keep the original frame as the other eye.",
        ),
        io.Float.Input(
            "ipd_offset",
            default=0.0,
            min=-10.0,
            max=10.0,
            step=0.1,
            **ADVANCED,
            tooltip="IPD offset in percent of the frame width (iw3 --ipd-offset). Changes the perceived scale.",
        ),
        io.Boolean.Input(
            "preserve_screen_border",
            default=False,
            **ADVANCED,
            tooltip="Force zero parallax at the frame border.",
        ),
        io.Int.Input("warp_steps", default=0, min=0, max=16, **ADVANCED, tooltip="row_flow warp steps. 0 = auto."),
        io.Int.Input(
            "stereo_width",
            default=0,
            min=0,
            max=8192,
            **ADVANCED,
            tooltip="Processing width for row_flow/mlbw models. 0 = source width.",
        ),
        io.Combo.Input(
            "inpaint_model",
            options=list(INPAINT_MODELS),
            default="light_inpaint_v1",
            **ADVANCED,
            tooltip="Inpainting model for *_inpaint methods.",
        ),
        io.Int.Input(
            "inpaint_overlap_frames",
            default=3,
            min=0,
            max=16,
            **ADVANCED,
            tooltip="Frames of temporal context for video inpainting.",
        ),
        io.Int.Input("mask_inner_dilation", default=0, min=0, max=8, **ADVANCED),
        io.Int.Input("mask_outer_dilation", default=0, min=0, max=8, **ADVANCED),
        io.Int.Input("inpaint_max_width", default=0, min=0, max=8192, **ADVANCED, tooltip="0 = no limit."),
        io.Float.Input(
            "pad",
            default=0.0,
            min=0.0,
            max=2.0,
            step=0.01,
            **ADVANCED,
            tooltip="Padding around each eye, as a fraction of its size. 0 = none.",
        ),
        io.Combo.Input("pad_mode", options=PAD_MODES, default="tblr", **ADVANCED),
        io.Int.Input("max_output_width", default=0, min=0, max=16384, **ADVANCED, tooltip="0 = no limit."),
        io.Int.Input("max_output_height", default=0, min=0, max=16384, **ADVANCED, tooltip="0 = no limit."),
        io.Boolean.Input("keep_aspect_ratio", default=False, **ADVANCED),
        io.Boolean.Input(
            "keep_stereo_model_in_vram",
            default=False,
            **ADVANCED,
            tooltip="Keep the stereo model on the GPU between runs. Faster, but ComfyUI cannot reclaim that VRAM.",
        ),
    ]


STEREO_OPTION_KEYS = [
    "method",
    "divergence",
    "convergence",
    "foreground_scale",
    "stereo_format",
    "anaglyph_method",
    "convergence_mode",
    "synthetic_view",
    "ipd_offset",
    "preserve_screen_border",
    "warp_steps",
    "stereo_width",
    "inpaint_model",
    "inpaint_overlap_frames",
    "mask_inner_dilation",
    "mask_outer_dilation",
    "inpaint_max_width",
    "pad",
    "pad_mode",
    "max_output_width",
    "max_output_height",
    "keep_aspect_ratio",
]


def _depth_inputs() -> list[io.Input]:
    return [
        io.Boolean.Input(
            "flicker_reduction",
            default=True,
            tooltip="Smooth depth normalization over time (iw3 --ema-normalize). Video only.",
        ),
        io.Float.Input("ema_decay", default=0.9, min=0.0, max=0.999, step=0.01, **ADVANCED),
        io.Int.Input("ema_buffer", default=30, min=1, max=1800, **ADVANCED, tooltip="Look-ahead frames."),
        io.Int.Input("batch_size", default=4, min=1, max=64, **ADVANCED),
        io.Int.Input("edge_dilation", default=2, min=0, max=8, **ADVANCED, tooltip="iw3 Edge Fix (x)."),
        io.Int.Input("edge_dilation_y", default=1, min=0, max=8, **ADVANCED, tooltip="iw3 Edge Fix (y)."),
        io.Boolean.Input("depth_aa", default=False, **ADVANCED, tooltip="Depth anti-aliasing (some models)."),
        io.Boolean.Input("tta", default=False, **ADVANCED, tooltip="Flip test-time augmentation for depth."),
        io.Boolean.Input(
            "low_vram",
            default=False,
            **ADVANCED,
            tooltip=(
                "iw3 --low-vram. Depth Pro 2 uses a shorter clip window (8 instead of 16 frames); "
                "Convert Video processes one frame at a time."
            ),
        ),
    ]


DEPTH_OPTION_KEYS = [
    "batch_size",
    "flicker_reduction",
    "ema_decay",
    "ema_buffer",
    "edge_dilation",
    "edge_dilation_y",
    "depth_aa",
    "tta",
    "low_vram",
]


def _pick(kwargs: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: kwargs[k] for k in keys if k in kwargs}


def _check_depth_model_mode(handle: DepthModelHandle, is_video: bool) -> None:
    model = handle.model
    if is_video and not model.is_video_supported():
        raise ValueError(f"`{handle.model_type}` does not support video. Set batch_mode to `images`.")
    if not is_video and not model.is_image_supported():
        raise ValueError(f"`{handle.model_type}` only supports video. Set batch_mode to `video`.")


def _run_stereo(
    images: torch.Tensor, depth: IW3Depth, stereo: StereoOptions, batch_size: int, keep_in_vram: bool
) -> torch.Tensor:
    device = mm.get_torch_device()
    args = build_args(device, stereo, DepthOptions(batch_size=batch_size))
    finalize_args(args, depth_model=None, is_metric=depth.is_metric)
    progress = _progress_callback(images.shape[0])
    try:
        if uses_stereo_model(stereo):
            with stereo_models.use(
                stereo.method,
                divergence=effective_divergence(stereo),
                device=device,
                inpaint_model=args.inpaint_model,
                overlap_frames=args.inpaint_overlap_frames,
                keep_in_vram=keep_in_vram,
            ) as side_model:
                return generate_stereo(images, depth, side_model, args, progress=progress)
        prepare_vram(device, 0)
        return generate_stereo(images, depth, None, args, progress=progress)
    finally:
        args.state = None
        release_vram()


class IW3LoadDepthModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IW3LoadDepthModel",
            display_name="iw3 Load Depth Model",
            category=CATEGORY,
            description="Loads an iw3 depth estimation model. Weights are downloaded to models/iw3 on first use.",
            inputs=[
                io.Combo.Input(
                    "depth_model",
                    options=DEPTH_MODEL_TYPES,
                    default=DEFAULT_DEPTH_MODEL,
                    tooltip=(
                        "Any_V3_Mono / Any_V2_S / Distill_Any_S are good general choices. "
                        "VDA_* (Video Depth Anything) and DepthPro2_* (Apple Depth Pro 2) are temporally stable; "
                        "VDA_* and DepthPro2_Stream_* are video only. "
                        "Some models have no automatic download and must be placed manually."
                    ),
                ),
                io.Int.Input(
                    "resolution",
                    default=0,
                    min=0,
                    max=8190,
                    **ADVANCED,
                    tooltip="Depth model input resolution (short side). 0 = model default.",
                ),
                io.Boolean.Input(
                    "limit_resolution",
                    default=False,
                    **ADVANCED,
                    tooltip="Do not process at a higher resolution than the source.",
                ),
                io.Boolean.Input(
                    "keep_in_vram",
                    default=False,
                    **ADVANCED,
                    tooltip="Keep the model on the GPU between runs. Faster, but ComfyUI cannot reclaim that VRAM.",
                ),
            ],
            outputs=[DepthModelIO.Output("depth_model")],
        )

    @classmethod
    def execute(
        cls, depth_model: str, resolution: int = 0, limit_resolution: bool = False, keep_in_vram: bool = False
    ) -> io.NodeOutput:
        handle = DepthModelHandle(
            depth_model,
            device=mm.get_torch_device(),
            resolution=resolution if resolution > 0 else None,
            limit_resolution=limit_resolution,
            keep_in_vram=keep_in_vram,
        )
        return io.NodeOutput(handle)


class IW3EstimateDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IW3EstimateDepth",
            display_name="iw3 Estimate Depth",
            category=CATEGORY,
            description="Estimates normalized depth for an image batch or video frames with iw3.",
            inputs=[
                io.Image.Input("images"),
                DepthModelIO.Input("depth_model"),
                io.Combo.Input(
                    "batch_mode",
                    options=BATCH_MODES,
                    default="video",
                    tooltip="`video`: the batch is consecutive frames. `images`: each image is processed on its own.",
                ),
                *_depth_inputs(),
                io.Boolean.Input(
                    "scene_detect",
                    default=False,
                    **ADVANCED,
                    tooltip="Detect scene cuts and reset temporal smoothing at each cut. Video only.",
                ),
            ],
            outputs=[
                DepthIO.Output("depth"),
                io.Image.Output("depth_image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        depth_model: DepthModelHandle,
        batch_mode: str,
        scene_detect: bool = False,
        **kwargs: Any,
    ) -> io.NodeOutput:
        is_video = batch_mode == "video"
        _check_depth_model_mode(depth_model, is_video)
        device = depth_model.device
        args = build_args(device, StereoOptions(), DepthOptions(**_pick(kwargs, DEPTH_OPTION_KEYS)))

        scene_boundaries: set[int] = set()
        if is_video and scene_detect:
            prepare_vram(device, 0)
            scene_boundaries = detect_scene_boundaries(images, device)
            release_vram()

        with depth_model.use() as model:
            depths = estimate_depth(
                images,
                model,
                args,
                is_video=is_video,
                scene_boundaries=scene_boundaries,
                progress=_progress_callback(images.shape[0]),
            )
        depth = IW3Depth(
            depth=depths,
            is_metric=depth_model.is_metric(),
            is_video=is_video,
            scene_boundaries=scene_boundaries,
        )
        depth_image = depth_to_image(depth, size=(images.shape[1], images.shape[2]))
        return io.NodeOutput(depth, depth_image)


class IW3DepthFromImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IW3DepthFromImage",
            display_name="iw3 Depth From Image",
            category=CATEGORY,
            description="Uses a depth map from another node (near = bright) as iw3 depth.",
            inputs=[
                io.Image.Input("depth_image"),
                io.Combo.Input("batch_mode", options=BATCH_MODES, default="video"),
                io.Boolean.Input("invert", default=False, tooltip="Enable if near objects are dark."),
                io.Combo.Input("normalize", options=DEPTH_NORMALIZE_MODES, default="per_frame"),
            ],
            outputs=[DepthIO.Output("depth")],
        )

    @classmethod
    def execute(cls, depth_image: torch.Tensor, batch_mode: str, invert: bool, normalize: str) -> io.NodeOutput:
        return io.NodeOutput(
            image_to_depth(depth_image, is_video=batch_mode == "video", invert=invert, normalize=normalize)
        )


class IW3Stereo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IW3Stereo",
            display_name="iw3 Make Stereo",
            category=CATEGORY,
            description="Turns images/frames plus iw3 depth into stereo 3D (SBS, TB, VR90, anaglyph, ...).",
            inputs=[
                io.Image.Input("images"),
                DepthIO.Input("depth"),
                *_stereo_inputs(),
                io.Int.Input("batch_size", default=4, min=1, max=64, **ADVANCED),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        depth: IW3Depth,
        batch_size: int = 4,
        keep_stereo_model_in_vram: bool = False,
        **kwargs: Any,
    ) -> io.NodeOutput:
        stereo = StereoOptions(**_pick(kwargs, STEREO_OPTION_KEYS))
        output = _run_stereo(images, depth, stereo, batch_size, keep_stereo_model_in_vram)
        return io.NodeOutput(output.to(mm.intermediate_device()))


class IW3ConvertVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IW3ConvertVideo",
            display_name="iw3 Convert Video",
            category=CATEGORY,
            description=(
                "Converts a video file to stereo 3D frame by frame with iw3's streaming pipeline. "
                "Memory use does not grow with video length. Audio is kept."
            ),
            inputs=[
                io.Video.Input("video"),
                DepthModelIO.Input("depth_model"),
                *_stereo_inputs(),
                *_depth_inputs(),
                io.Boolean.Input(
                    "scene_detect",
                    default=False,
                    **ADVANCED,
                    tooltip="Detect scene cuts and reset temporal smoothing at each cut.",
                ),
                io.Float.Input(
                    "max_fps",
                    default=30.0,
                    min=1.0,
                    max=240.0,
                    step=1.0,
                    **ADVANCED,
                    tooltip="Output frame rate is min(source fps, max_fps).",
                ),
                io.Combo.Input("video_codec", options=VIDEO_CODECS, default="auto", **ADVANCED),
                io.Int.Input("crf", default=20, min=0, max=51, **ADVANCED, tooltip="Lower is higher quality."),
                io.Combo.Input("preset", options=VIDEO_PRESETS, default="medium", **ADVANCED),
                io.Combo.Input("pix_fmt", options=PIX_FMTS, default="yuv420p", **ADVANCED),
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        video: Any,
        depth_model: DepthModelHandle,
        keep_stereo_model_in_vram: bool = False,
        scene_detect: bool = False,
        max_fps: float = 30.0,
        video_codec: str = "auto",
        crf: int = 20,
        preset: str = "medium",
        pix_fmt: str = "yuv420p",
        **kwargs: Any,
    ) -> io.NodeOutput:
        _check_depth_model_mode(depth_model, is_video=True)
        device = depth_model.device
        temp_dir = os.path.join(folder_paths.get_temp_directory(), "iw3")
        source, source_is_temporary = resolve_video_source(video, temp_dir)
        filename = f"iw3_{uuid.uuid4().hex[:12]}.mp4"
        output_path = os.path.join(temp_dir, filename)

        stereo = StereoOptions(**_pick(kwargs, STEREO_OPTION_KEYS))
        video_options = VideoOptions(
            max_fps=max_fps,
            scene_detect=scene_detect,
            video_codec=video_codec,
            crf=crf,
            preset=preset,
            pix_fmt=pix_fmt,
            extra={"disable_scene_cache": True, "video_format": "mp4"},
        )
        args = build_args(
            device,
            stereo,
            DepthOptions(**_pick(kwargs, DEPTH_OPTION_KEYS)),
            video_options,
            output=output_path,
        )
        stop_event = threading.Event()
        try:
            with depth_model.use() as model:
                finalize_args(
                    args,
                    depth_model=model,
                    is_metric=depth_model.is_metric(),
                    stop_event=stop_event,
                    tqdm_fn=_make_tqdm_fn(stop_event),
                )
                if uses_stereo_model(stereo):
                    with stereo_models.use(
                        stereo.method,
                        divergence=effective_divergence(stereo),
                        device=device,
                        inpaint_model=args.inpaint_model,
                        overlap_frames=args.inpaint_overlap_frames,
                        keep_in_vram=keep_stereo_model_in_vram,
                    ) as side_model:
                        convert_video_file(source, output_path, args, model, side_model)
                else:
                    convert_video_file(source, output_path, args, model, None)
        finally:
            args.state = None
            release_vram()
            if source_is_temporary and os.path.isfile(source):
                os.unlink(source)
        mm.throw_exception_if_processing_interrupted()

        result = InputImpl.VideoFromFile(output_path)
        preview = ui.PreviewVideo([ui.SavedResult(filename, "iw3", io.FolderType.temp)])
        return io.NodeOutput(result, ui=preview)


class IW3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [IW3LoadDepthModel, IW3EstimateDepth, IW3DepthFromImage, IW3Stereo, IW3ConvertVideo]


async def comfy_entrypoint() -> IW3Extension:
    return IW3Extension()
