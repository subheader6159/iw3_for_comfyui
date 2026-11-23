import argparse
import io
import os
import uuid
from typing import Any

from comfy_api.latest import InputImpl, Types
from iw3.utils import process_video

# Quality of the intermediate file written when a VIDEO has to be encoded before conversion.
INTERMEDIATE_CRF = 12


def _temp_video_path(temp_dir: str) -> str:
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"iw3_input_{uuid.uuid4().hex}.mp4")


def resolve_video_source(video: Any, temp_dir: str) -> tuple[str, bool]:
    """Get a file path for a ComfyUI VIDEO, avoiding decoding and re-encoding where possible.

    Returns (path, is_temporary_copy).
    """
    start, duration = video.get_active_trim_window()
    # Cropping was added to ComfyUI's VIDEO type after v0.46.0.
    get_crop = getattr(video, "get_active_crop_rect", None)
    cropped = get_crop is not None and get_crop() is not None
    trimmed_or_cropped = start > 0 or duration > 0 or cropped
    if isinstance(video, InputImpl.VideoFromFile) and not trimmed_or_cropped:
        src = video.get_stream_source()
        if isinstance(src, (str, os.PathLike)):
            return os.fspath(src), False
        if isinstance(src, io.BytesIO):
            path = _temp_video_path(temp_dir)
            with open(path, "wb") as f:
                f.write(src.getbuffer())
            return path, True
        raise TypeError(f"Unsupported video source: {type(src).__name__}")

    # Frames from other nodes (Create Video, ...) must be encoded first. ComfyUI's default (crf 18, 4:2:0)
    # loses color detail, so a near-lossless 4:4:4 file is written instead.
    # Trimmed/cropped files are also cut here, because iw3's own --start-time/--end-time are keyframe-granular
    # and can drop frames at the end.
    path = _temp_video_path(temp_dir)
    video.save_to(
        path,
        format=Types.VideoContainer.MP4,
        codec=Types.VideoCodec.H264,
        crf=INTERMEDIATE_CRF,
        chroma_subsampling="444",
    )
    return path, True


def convert_video_file(
    input_path: str,
    output_path: str,
    args: argparse.Namespace,
    depth_model: Any,
    side_model: Any,
) -> None:
    """Convert a video file with iw3's streaming pipeline (decode -> depth -> stereo -> encode)."""
    args.input = input_path
    args.output = output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_video(input_path, output_path, args, depth_model, side_model)
    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
        return
    if not os.path.isfile(output_path):
        raise RuntimeError(f"iw3 did not produce an output file: {output_path}")
