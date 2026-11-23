import asyncio
import os

import numpy as np
import pytest
import torch
from PIL import Image
from scene import probe_video, read_frame, render_clip, write_video

H, W = 256, 448


def run(node_cls, **kwargs):
    out = node_cls.execute(**kwargs)
    return out.args


@pytest.fixture(scope="module")
def clip():
    return render_clip(12, H, W)


@pytest.fixture(scope="module")
def depth_model(nodes, require_gpu):
    (handle,) = run(nodes.IW3LoadDepthModel, depth_model="Any_V2_S")
    return handle


@pytest.fixture(scope="module")
def video_depth(nodes, depth_model, clip):
    depth, depth_image = run(
        nodes.IW3EstimateDepth, images=clip, depth_model=depth_model, batch_mode="video", flicker_reduction=True
    )
    return depth, depth_image


def test_extension_nodes(custom_node, nodes):
    extension = asyncio.run(custom_node.comfy_entrypoint())
    node_list = asyncio.run(extension.get_node_list())
    ids = [cls.define_schema().node_id for cls in node_list]
    assert ids == [
        "IW3LoadDepthModel",
        "IW3EstimateDepth",
        "IW3DepthFromImage",
        "IW3Stereo",
        "IW3ConvertVideo",
        "IW3StreamDecodeVideo",
    ]
    for cls in node_list:
        schema = cls.define_schema()
        schema.validate()
        for item in schema.inputs:
            if getattr(item, "advanced", False):
                assert item.optional, f"{schema.node_id}.{item.id} must be optional"


def test_weights_go_to_comfyui_models_dir(nodes):
    import folder_paths
    from iw3.hub_dir import HUB_MODEL_DIR

    assert HUB_MODEL_DIR.startswith(os.path.join(folder_paths.models_dir, "iw3"))


def test_estimate_depth_video(nodes, clip, video_depth, artifacts_dir):
    depth, depth_image = video_depth
    assert len(depth) == clip.shape[0]
    assert depth.is_video and not depth.is_metric
    assert depth.depth.dtype == torch.float16
    d = depth.depth.float()
    assert 0.0 <= d.min() and d.max() <= 1.0
    # The floor at the bottom of the frame is nearer than the sky at the top.
    h = d.shape[-2]
    assert d[..., -h // 8 :, :].mean() > d[..., : h // 8, :].mean() + 0.3
    assert depth_image.shape == (clip.shape[0], H, W, 3)
    Image.fromarray((depth_image[0].numpy() * 255).astype(np.uint8)).save(os.path.join(artifacts_dir, "depth.png"))
    Image.fromarray((clip[0].numpy() * 255).astype(np.uint8)).save(os.path.join(artifacts_dir, "input.png"))


def test_estimate_depth_images(nodes, depth_model, clip):
    depth, _ = run(
        nodes.IW3EstimateDepth, images=clip[:3], depth_model=depth_model, batch_mode="images", flicker_reduction=True
    )
    assert len(depth) == 3 and not depth.is_video
    # Each frame is normalized on its own.
    for i in range(3):
        assert depth.depth[i].float().max() > 0.99


def _expected_shape(stereo_format: str, b: int) -> tuple[int, ...]:
    from iw3.equirectangular import resolve_equirectangular_size

    if stereo_format in {"Full SBS", "Cross Eyed", "RGB-D"}:
        return (b, H, W * 2, 3)
    if stereo_format == "Full TB":
        return (b, H * 2, W, 3)
    if stereo_format == "VR90":
        eye_h, eye_w = resolve_equirectangular_size(H, W)
        return (b, eye_h, eye_w * 2, 3)
    return (b, H, W, 3)


@pytest.mark.parametrize(
    "stereo_format",
    ["Full SBS", "Half SBS", "Full TB", "Half TB", "VR90", "Cross Eyed", "RGB-D", "Half RGB-D", "Anaglyph"],
)
def test_stereo_formats(nodes, clip, video_depth, stereo_format):
    depth, _ = video_depth
    (out,) = run(
        nodes.IW3Stereo,
        images=clip,
        depth=depth,
        method="mlbw_l2s",
        divergence=2.0,
        convergence=0.25,
        foreground_scale=0.0,
        stereo_format=stereo_format,
    )
    assert tuple(out.shape) == _expected_shape(stereo_format, clip.shape[0])
    assert out.dtype == torch.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


@pytest.mark.parametrize(
    "method",
    ["mlbw_l2s", "mlbw_l2", "row_flow_v3", "forward_fill", "backward", "monobw", "mlbw_l2_inpaint", "forward_inpaint"],
)
def test_stereo_methods(nodes, clip, video_depth, method, artifacts_dir):
    depth, _ = video_depth
    (out,) = run(
        nodes.IW3Stereo,
        images=clip,
        depth=depth,
        method=method,
        divergence=3.0,
        convergence=0.25,
        foreground_scale=0.0,
        stereo_format="Full SBS",
    )
    assert out.shape[0] == clip.shape[0]
    left, right = out[..., :W, :], out[..., W:, :]
    # The two eyes differ (there is parallax) but each still resembles the source frame.
    assert (left - right).abs().mean() > 0.005
    assert (left - clip).abs().mean() < 0.1
    assert (right - clip).abs().mean() < 0.1
    if method == "mlbw_l2s":
        Image.fromarray((out[0].numpy() * 255).astype(np.uint8)).save(os.path.join(artifacts_dir, "sbs.png"))


def test_stereo_options(nodes, clip, video_depth):
    depth, _ = video_depth
    (out,) = run(
        nodes.IW3Stereo,
        images=clip,
        depth=depth,
        method="row_flow_v3_sym",
        divergence=6.0,
        convergence=0.5,
        foreground_scale=1.5,
        stereo_format="Full SBS",
        convergence_mode="sod_v1",
        synthetic_view="right",
        ipd_offset=2.0,
        preserve_screen_border=True,
        pad=0.1,
        max_output_width=640,
        keep_aspect_ratio=True,
        batch_size=3,
    )
    assert out.shape[0] == clip.shape[0]
    assert out.shape[2] == 640


def test_depth_from_image(nodes, clip, video_depth):
    _, depth_image = video_depth
    (depth,) = run(
        nodes.IW3DepthFromImage, depth_image=depth_image, batch_mode="video", invert=False, normalize="per_frame"
    )
    assert depth.depth.shape == (clip.shape[0], 1, H, W)
    (out,) = run(
        nodes.IW3Stereo,
        images=clip,
        depth=depth,
        method="mlbw_l2s",
        divergence=2.0,
        convergence=0.25,
        foreground_scale=0.0,
        stereo_format="Half SBS",
    )
    assert out.shape == (clip.shape[0], H, W, 3)


def test_scene_detect(nodes, depth_model):
    frames = render_clip(40, H, W, scene_cut_at=19)
    depth, _ = run(
        nodes.IW3EstimateDepth,
        images=frames,
        depth_model=depth_model,
        batch_mode="video",
        flicker_reduction=True,
        scene_detect=True,
    )
    assert len(depth) == 40
    assert depth.scene_boundaries & {18, 19, 20}, depth.scene_boundaries


def test_video_depth_anything_streaming(nodes, clip):
    (vda,) = run(nodes.IW3LoadDepthModel, depth_model="VDA_Stream_S")
    depth, _ = run(nodes.IW3EstimateDepth, images=clip, depth_model=vda, batch_mode="video", flicker_reduction=True)
    assert len(depth) == clip.shape[0]
    with pytest.raises(ValueError):
        run(nodes.IW3EstimateDepth, images=clip, depth_model=vda, batch_mode="images", flicker_reduction=True)


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "clip.mp4")
    write_video(path, render_clip(36, H, W), fps=24, with_audio=True)
    return path


def _convert(nodes, video, depth_model, **kwargs):
    params = dict(
        method="mlbw_l2s",
        divergence=2.0,
        convergence=0.25,
        foreground_scale=0.0,
        stereo_format="Full SBS",
        flicker_reduction=True,
    )
    params.update(kwargs)
    (out,) = run(nodes.IW3ConvertVideo, video=video, depth_model=depth_model, **params)
    return out


def test_convert_video(nodes, depth_model, video_file, artifacts_dir):
    from comfy_api.latest import InputImpl

    out = _convert(nodes, InputImpl.VideoFromFile(video_file), depth_model)
    path = out.get_stream_source()
    info = probe_video(path)
    assert (info["width"], info["height"]) == (W * 2, H)
    assert info["frames"] == 36
    assert info["has_audio"]
    Image.fromarray(read_frame(path, 10)).save(os.path.join(artifacts_dir, "convert_video_frame.png"))


def test_convert_video_trimmed_and_scene_detect(nodes, depth_model, video_file):
    from comfy_api.latest import InputImpl

    video = InputImpl.VideoFromFile(video_file).as_trimmed(start_time=0.5, duration=0.5)
    out = _convert(nodes, video, depth_model, stereo_format="Half SBS", scene_detect=True)
    info = probe_video(out.get_stream_source())
    assert (info["width"], info["height"]) == (W, H)
    assert 11 <= info["frames"] <= 13


def test_convert_video_from_frames(nodes, depth_model, clip):
    from fractions import Fraction

    from comfy_api.latest import InputImpl, Types

    video = InputImpl.VideoFromComponents(Types.VideoComponents(images=clip, audio=None, frame_rate=Fraction(16)))
    out = _convert(nodes, video, depth_model, stereo_format="Full TB", method="mlbw_l2_inpaint")
    info = probe_video(out.get_stream_source())
    assert (info["width"], info["height"]) == (W, H * 2)
    assert info["frames"] == clip.shape[0]


def test_model_offload(nodes, depth_model, clip):
    def device_of(handle):
        return next(handle.modules()[0].parameters()).device.type

    assert device_of(depth_model) == "cpu"
    (pinned,) = run(nodes.IW3LoadDepthModel, depth_model="Any_V2_S", keep_in_vram=True)
    run(nodes.IW3EstimateDepth, images=clip[:2], depth_model=pinned, batch_mode="video", flicker_reduction=True)
    assert device_of(pinned) == "cuda"
    assert device_of(depth_model) == "cpu"


def test_interrupt(nodes, depth_model, clip, video_depth, video_file):
    import comfy.model_management as mm
    from comfy_api.latest import InputImpl

    depth, _ = video_depth
    mm.interrupt_current_processing(True)
    try:
        with pytest.raises(mm.InterruptProcessingException):
            run(
                nodes.IW3Stereo,
                images=clip,
                depth=depth,
                method="mlbw_l2s",
                divergence=2.0,
                convergence=0.25,
                foreground_scale=0.0,
                stereo_format="Full SBS",
            )
        mm.interrupt_current_processing(True)
        with pytest.raises(mm.InterruptProcessingException):
            _convert(nodes, InputImpl.VideoFromFile(video_file), depth_model)
    finally:
        mm.interrupt_current_processing(False)


@pytest.fixture(scope="module")
def require_depth_pro2(nodes):
    if "DepthPro2_S" not in nodes.DEPTH_MODEL_TYPES:
        pytest.skip("this nunif checkout has no Depth Pro 2 (iw3/depth_models/depth_pro2_model.py)")


@pytest.fixture(scope="module")
def depth_pro2(nodes, require_gpu, require_depth_pro2):
    (handle,) = run(nodes.IW3LoadDepthModel, depth_model="DepthPro2_S")
    return handle


@pytest.mark.parametrize("low_vram", [False, True])
def test_depth_pro2_video(nodes, depth_pro2, low_vram):
    # 20 frames with a cut after frame 9: covers clip windows (16/4, or 8/2 with low_vram),
    # the flush at the scene cut and the final flush.
    frames = render_clip(20, H, W, scene_cut_at=9)
    depth, _ = run(
        nodes.IW3EstimateDepth,
        images=frames,
        depth_model=depth_pro2,
        batch_mode="video",
        flicker_reduction=True,
        scene_detect=True,
        low_vram=low_vram,
    )
    assert len(depth) == 20
    assert depth.scene_boundaries & {8, 9, 10}, depth.scene_boundaries
    d = depth.depth.float()
    h = d.shape[-2]
    assert d[..., -h // 8 :, :].mean() > d[..., : h // 8, :].mean() + 0.3


def test_depth_pro2_images(nodes, depth_pro2, clip):
    depth, _ = run(
        nodes.IW3EstimateDepth, images=clip[:3], depth_model=depth_pro2, batch_mode="images", flicker_reduction=True
    )
    assert len(depth) == 3 and not depth.is_video


def test_depth_pro2_stream(nodes, clip, require_gpu, require_depth_pro2):
    (stream,) = run(nodes.IW3LoadDepthModel, depth_model="DepthPro2_Stream_S")
    depth, _ = run(nodes.IW3EstimateDepth, images=clip, depth_model=stream, batch_mode="video", flicker_reduction=True)
    assert len(depth) == clip.shape[0]
    with pytest.raises(ValueError):
        run(nodes.IW3EstimateDepth, images=clip, depth_model=stream, batch_mode="images", flicker_reduction=True)


def test_depth_pro2_convert_video(nodes, depth_pro2, video_file):
    from comfy_api.latest import InputImpl

    out = _convert(nodes, InputImpl.VideoFromFile(video_file), depth_pro2, low_vram=True)
    info = probe_video(out.get_stream_source())
    assert (info["width"], info["height"]) == (W * 2, H)
    assert info["frames"] == 36
    assert info["has_audio"]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-10)))


def test_frames_are_encoded_at_high_quality(nodes, clip, tmp_path):
    from fractions import Fraction

    from comfy_api.latest import InputImpl, Types

    video_module = __import__(nodes.__name__.rsplit(".", 1)[0] + ".video", fromlist=["resolve_video_source"])
    video = InputImpl.VideoFromComponents(Types.VideoComponents(images=clip, audio=None, frame_rate=Fraction(16)))
    path, is_temporary = video_module.resolve_video_source(video, str(tmp_path))
    assert is_temporary
    default_path = str(tmp_path / "default.mp4")
    with open(default_path, "wb") as f:
        f.write(video.get_stream_source().getbuffer())

    source = (clip[5].numpy() * 255).round().astype(np.uint8)
    ours, default = _psnr(read_frame(path, 5), source), _psnr(read_frame(default_path, 5), source)
    print(f"PSNR intermediate={ours:.2f}dB comfy_default={default:.2f}dB")
    assert ours > 42.0 and ours > default + 5.0


def test_video_without_crop_support(nodes, video_file, tmp_path, monkeypatch):
    # ComfyUI <= v0.46.0 has no get_active_crop_rect on VIDEO inputs.
    from comfy_api.latest import Input, InputImpl

    monkeypatch.delattr(InputImpl.VideoFromFile, "get_active_crop_rect")
    monkeypatch.delattr(Input.Video, "get_active_crop_rect")
    video_module = __import__(nodes.__name__.rsplit(".", 1)[0] + ".video", fromlist=["resolve_video_source"])
    video = InputImpl.VideoFromFile(video_file)
    assert not hasattr(video, "get_active_crop_rect")
    assert video_module.resolve_video_source(video, str(tmp_path)) == (video_file, False)
