import logging
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
from .stream_decode import FrameSink, VideoFileWriter, stream_temporal_decode  # noqa: E402
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


def _stream_decode_inputs() -> list[io.Input]:
    return [
        io.Latent.Input("samples"),
        io.Vae.Input("vae"),
        io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=1.0),
        io.String.Input("filename_prefix", default="video/ComfyUI"),
        io.Audio.Input("audio", optional=True, tooltip="Muxed into the file, e.g. from VAE Decode Audio."),
        io.Int.Input("crf", default=18, min=0, max=51, **ADVANCED, tooltip="Lower is higher quality."),
        io.Int.Input(
            "temporal_size",
            default=64,
            min=8,
            max=4096,
            step=4,
            **ADVANCED,
            tooltip="Frames decoded at a time. Lower uses less RAM.",
        ),
        io.Int.Input(
            "temporal_overlap",
            default=8,
            min=4,
            max=4096,
            step=4,
            **ADVANCED,
            tooltip="Frames shared between chunks, blended to hide the joins.",
        ),
        io.Int.Input(
            "tile_size",
            default=0,
            min=0,
            max=4096,
            step=32,
            **ADVANCED,
            tooltip="Spatial tile size for VRAM savings, as in VAE Decode (Tiled). 0 = whole frame.",
        ),
        io.Int.Input("overlap", default=64, min=0, max=4096, step=32, **ADVANCED),
    ]


def _decode_chunked_io(vae: Any, samples: torch.Tensor, sink: Any) -> None:
    """vae.decode() for chunked-IO VAEs, writing into `sink` instead of a preallocated tensor."""
    vae.throw_exception_if_invalid()
    memory_used = vae.memory_used_decode(samples.shape, vae.vae_dtype)
    mm.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)
    with torch.inference_mode(), mm.cuda_device_context(vae.device):
        vae.first_stage_model.decode(samples.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=sink)


def _stream_decode_to_file(
    hidden: Any,
    samples: dict[str, Any],
    vae: Any,
    fps: float,
    filename_prefix: str,
    audio: dict[str, Any] | None = None,
    crf: int = 18,
    temporal_size: int = 64,
    temporal_overlap: int = 8,
    tile_size: int = 0,
    overlap: int = 64,
    post: Any = None,
    output_fps: float | None = None,
) -> io.NodeOutput:
    """Shared body of the streaming decode nodes. `post` optionally transforms each chunk of frames."""
    latent = samples["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]

    # Same parameter handling as ComfyUI's VAE Decode (Tiled).
    if tile_size and tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    compression = vae.spacial_compression_decode()

    def decode(x: torch.Tensor) -> torch.Tensor:
        if tile_size > 0:
            t = x.shape[2] if x.ndim == 5 else None
            images = vae.decode_tiled(
                x,
                tile_x=tile_size // compression,
                tile_y=tile_size // compression,
                overlap=overlap // compression,
                tile_t=t,
                overlap_t=1 if t is not None else None,
            )
        else:
            images = vae.decode(x)
        return images.reshape(-1, *images.shape[-3:])

    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory()
    )
    file = f"{filename}_{counter:05}_.mp4"
    path = os.path.join(full_output_folder, file)
    metadata: dict[str, Any] = {}
    if hidden is not None and hidden.extra_pnginfo is not None:
        metadata.update(hidden.extra_pnginfo)
    if hidden is not None and hidden.prompt is not None:
        metadata["prompt"] = hidden.prompt

    writer = VideoFileWriter(path, fps=output_fps or fps, crf=crf, metadata=metadata or None, audio=audio)

    def write(frames: torch.Tensor) -> None:
        writer.write(post(frames) if post is not None else frames)

    try:
        temporal_compression = vae.temporal_compression_decode()
        first_stage: Any = getattr(vae, "first_stage_model", None)
        if latent.ndim == 5 and getattr(first_stage, "comfy_has_chunked_io", False):
            # VAEs like MiniMax H3 / LTX chunk the video themselves and write finished frames in order.
            # Stream those writes to the file; slicing their latents ourselves would break their chunking.
            progress = _progress_callback(1)
            for b in range(latent.shape[0]):
                samples_b = latent[b : b + 1]
                shape = tuple(first_stage.decode_output_shape(samples_b.shape))
                total = shape[2]

                def sink_write(frames: torch.Tensor, total: int = total, b: int = b) -> None:
                    write(frames)
                    progress(b * total + sink.next_frame + frames.shape[0], total * latent.shape[0])

                sink = FrameSink(shape, sink_write)
                _decode_chunked_io(vae, samples_b, sink)
                if sink.next_frame != total:
                    raise RuntimeError(f"The VAE wrote {sink.next_frame} of {total} frames")
        elif latent.ndim == 5 and getattr(vae, "handles_tiling", False):
            # Model-owned tiling without chunked IO: its temporal layout is unknown, so decode it in one go.
            logging.warning("iw3: this VAE cannot be decoded in chunks; decoding the whole video at once.")
            write(decode(latent).float().cpu())
        elif latent.ndim == 5 and temporal_compression is not None:
            tile_t = max(2, temporal_size // temporal_compression)
            overlap_t = max(1, min(tile_t // 2, temporal_overlap // temporal_compression))
            upscale_t = vae.upscale_ratio[0]
            index_formula = getattr(vae, "upscale_index_formula", None)
            index_t = index_formula[0] if index_formula is not None else upscale_t
            stream_temporal_decode(
                latent,
                decode,
                write,
                tile_t=tile_t,
                overlap_t=overlap_t,
                upscale_t=upscale_t,
                index_t=index_t,
                progress=_progress_callback(1),
            )
        else:
            # Image VAE: the batch is the frame sequence, decode it in independent chunks.
            progress = _progress_callback(latent.shape[0])
            for start in range(0, latent.shape[0], temporal_size):
                write(decode(latent[start : start + temporal_size]).float().cpu())
                progress(min(start + temporal_size, latent.shape[0]), latent.shape[0])
        writer.close()
    except BaseException:
        writer.abort()
        if os.path.isfile(path):
            os.unlink(path)
        raise

    result = InputImpl.VideoFromFile(path)
    preview = ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)])
    return io.NodeOutput(result, ui=preview)


class IW3StreamDecodeVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IW3StreamDecodeVideo",
            display_name="VAE Decode to Video File (streaming)",
            category=CATEGORY,
            search_aliases=["stream decode", "low ram decode", "decode to file", "vae decode video"],
            description=(
                "Decodes a video latent a chunk of frames at a time and writes each chunk straight to an MP4 "
                "in the output folder, so RAM use does not grow with video length. Replaces VAE Decode + "
                "Create Video + Save Video. The VIDEO output can feed iw3 Convert Video directly."
            ),
            inputs=_stream_decode_inputs(),
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> io.NodeOutput:
        # cls.hidden is None when called outside of a ComfyUI prompt.
        return _stream_decode_to_file(cls.hidden, **kwargs)


# From nvidia-vfx 0.2 (nvvfx.QualityLevel), as offered by the RTX Video Super Resolution node.
RTX_QUALITY_OPTIONS = [
    "BICUBIC",
    "LOW",
    "MEDIUM",
    "HIGH",
    "ULTRA",
    "DENOISE_LOW",
    "DENOISE_MEDIUM",
    "DENOISE_HIGH",
    "DENOISE_ULTRA",
    "DEBLUR_LOW",
    "DEBLUR_MEDIUM",
    "DEBLUR_HIGH",
    "DEBLUR_ULTRA",
    "HIGHBITRATE_LOW",
    "HIGHBITRATE_MEDIUM",
    "HIGHBITRATE_HIGH",
    "HIGHBITRATE_ULTRA",
    "STREAMING_MEDIUM",
    "STREAMING_ULTRA",
]
RTX_ENCODINGS = ["8-bit RGB", "10-bit RGB (RGB10A2)"]
RTX_NODE_ID = "RTXVideoSuperResolution"
RTX_FRAME_GENERATION_NODE_ID = "RTXVideoFrameGeneration"
RTX_FRAME_GENERATION_MODES = ["LOW", "MEDIUM", "HIGH"]


def _rtx_node(node_id: str, display_name: str) -> Any:
    import nodes as comfy_nodes

    node = comfy_nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if node is None:
        raise RuntimeError(
            f"This needs the `{display_name}` node from the comfyui_nvidia_rtx_nodes pack. "
            "Install it with ComfyUI Manager, or use `VAE Decode to Video File (streaming)` instead."
        )
    return node


def _node_images(out: Any) -> torch.Tensor:
    images = out.args[0] if hasattr(out, "args") else out[0]
    return images.float().cpu()


def _rtx_frame_generator(multiplier: int, mode: str, shot_detection: bool, encoding: str) -> Any:
    """Per-chunk RTX frame interpolation. Each chunk starts with the previous chunk's last frame,
    so frames are also generated across chunk joins. N input frames become (N - 1) * multiplier + 1."""
    rtx_node = _rtx_node(RTX_FRAME_GENERATION_NODE_ID, "RTX Video Frame Generation")
    last: list[torch.Tensor] = []

    def interpolate(frames: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([last[0], frames]) if last else frames
        out = _node_images(
            rtx_node.execute(
                images=joined,
                generation_type={"generation_type": "frame rate multiplier", "multiplier": multiplier},
                mode=mode,
                automatic_shot_change_detection=shot_detection,
                shot_change=False,
                image_encoding=encoding,
            )
        )
        result = out[1:] if last else out  # the carried frame was already written
        last[:] = [frames[-1:].clone()]
        return result

    return interpolate


def _rtx_upscaler(scale: float, quality: str, strength: float, encoding: str) -> Any:
    """Per-chunk RTX upscale using the installed RTX Video Super Resolution node (comfyui_nvidia_rtx_nodes).

    RTX VSR processes every frame independently, so upscaling chunk by chunk equals upscaling the whole video.
    """
    rtx_node = _rtx_node(RTX_NODE_ID, "RTX Video Super Resolution")

    def upscale(frames: torch.Tensor) -> torch.Tensor:
        out = rtx_node.execute(
            images=frames,
            resize_type={"resize_type": "scale by multiplier", "scale": scale},
            quality=quality,
            strength=strength,
            image_encoding=encoding,
        )
        return _node_images(out)

    return upscale


class IW3StreamDecodeVideoRTX(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IW3StreamDecodeVideoRTX",
            display_name="VAE Decode to Video File + RTX Upscale (streaming)",
            category=CATEGORY,
            search_aliases=["stream decode rtx", "rtx upscale video", "low ram decode upscale", "vsr"],
            description=(
                "Like VAE Decode to Video File (streaming), but each decoded chunk is optionally interpolated "
                "(RTX Video Frame Generation) and upscaled (RTX Video Super Resolution) before it is written, "
                "so RAM stays flat. Needs the comfyui_nvidia_rtx_nodes pack and an RTX GPU it supports."
            ),
            inputs=[
                *_stream_decode_inputs(),
                io.Float.Input("rtx_scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Upscale factor."),
                io.Combo.Input(
                    "rtx_quality",
                    options=RTX_QUALITY_OPTIONS,
                    default="ULTRA",
                    tooltip="RTX model. DENOISE_* and DEBLUR_* keep the original size.",
                ),
                io.Float.Input("rtx_strength", default=1.0, min=0.0, max=1.0, step=0.01, **ADVANCED),
                io.Combo.Input("rtx_image_encoding", options=RTX_ENCODINGS, default="8-bit RGB", **ADVANCED),
                io.Int.Input(
                    "rtx_frame_multiplier",
                    default=1,
                    min=1,
                    max=16,
                    tooltip=(
                        "RTX frame generation (interpolation). 1 = off, 2 = double the frame rate (24 -> 48 fps). "
                        "Set fps to the source frame rate; the output frame rate is fps x multiplier."
                    ),
                ),
                io.Combo.Input(
                    "rtx_frame_generation_mode",
                    options=RTX_FRAME_GENERATION_MODES,
                    default="MEDIUM",
                    **ADVANCED,
                ),
                io.Boolean.Input(
                    "rtx_shot_change_detection",
                    default=True,
                    **ADVANCED,
                    tooltip="Do not interpolate across scene cuts.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        rtx_scale: float,
        rtx_quality: str,
        rtx_strength: float = 1.0,
        rtx_image_encoding: str = "8-bit RGB",
        rtx_frame_multiplier: int = 1,
        rtx_frame_generation_mode: str = "MEDIUM",
        rtx_shot_change_detection: bool = True,
        **kwargs: Any,
    ) -> io.NodeOutput:
        steps = []
        # Interpolate before upscaling: frame generation runs on the smaller frames.
        if rtx_frame_multiplier > 1:
            steps.append(
                _rtx_frame_generator(
                    rtx_frame_multiplier, rtx_frame_generation_mode, rtx_shot_change_detection, rtx_image_encoding
                )
            )
        if rtx_scale != 1.0 or rtx_quality.startswith(("DENOISE_", "DEBLUR_")):
            steps.append(_rtx_upscaler(rtx_scale, rtx_quality, rtx_strength, rtx_image_encoding))

        def post(frames: torch.Tensor) -> torch.Tensor:
            for step in steps:
                frames = step(frames)
            return frames

        return _stream_decode_to_file(
            cls.hidden, post=post, output_fps=kwargs.get("fps", 24.0) * rtx_frame_multiplier, **kwargs
        )


class IW3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            IW3LoadDepthModel,
            IW3EstimateDepth,
            IW3DepthFromImage,
            IW3Stereo,
            IW3ConvertVideo,
            IW3StreamDecodeVideo,
            IW3StreamDecodeVideoRTX,
        ]


async def comfy_entrypoint() -> IW3Extension:
    return IW3Extension()
