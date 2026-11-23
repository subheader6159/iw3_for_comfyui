import contextlib
import gc
import threading
from collections.abc import Iterator
from typing import Any

import comfy.model_management as mm
import torch
import torch.nn as nn
from iw3.base_depth_model import BaseDepthModel
from iw3.depth_model_factory import create_depth_model
from iw3.stereo_model_factory import create_stereo_model, get_mlbw_divergence_level

from .options import INPAINT_METHODS, device_to_gpu_id

# Rough working memory reserved for activations on top of model weights.
INFERENCE_RESERVE_BYTES = 2 * 1024**3

DEPTH_MODEL_TYPES = [m for m in BaseDepthModel.get_all_model_types() if m != "NULL"]
DEFAULT_DEPTH_MODEL = "Any_V3_Mono" if "Any_V3_Mono" in DEPTH_MODEL_TYPES else DEPTH_MODEL_TYPES[0]


def _modules_of(obj: Any) -> list[nn.Module]:
    return [m for m in obj if isinstance(m, nn.Module)]


def _module_bytes(modules: list[nn.Module]) -> int:
    total = 0
    for module in modules:
        for t in list(module.parameters()) + list(module.buffers()):
            total += t.numel() * t.element_size()
    return total


def _move(modules: list[nn.Module], device: torch.device) -> None:
    for module in modules:
        module.to(device)


def prepare_vram(device: torch.device, required_bytes: int) -> None:
    """Ask ComfyUI to unload its own models until `required_bytes` fit on `device`."""
    if device.type == "cpu":
        return
    mm.free_memory(required_bytes + INFERENCE_RESERVE_BYTES, device)


def release_vram() -> None:
    gc.collect()
    mm.soft_empty_cache()


class DepthModelHandle:
    """A loaded iw3 depth model. Weights stay in RAM and are moved to the GPU only while in use."""

    def __init__(
        self,
        model_type: str,
        device: torch.device,
        resolution: int | None = None,
        limit_resolution: bool = False,
        keep_in_vram: bool = False,
    ):
        cls = BaseDepthModel.get_depth_model_class(model_type)
        if cls is None:
            raise ValueError(f"Unknown depth model: {model_type}")
        if not cls.is_model_available(model_type):
            raise FileNotFoundError(
                f"`{model_type}` has no automatic download. Download its checkpoint manually and place it at "
                f"`{cls.get_model_path(model_type)}`."
            )
        self.model_type = model_type
        self.device = device
        self.keep_in_vram = keep_in_vram
        self.lock = threading.RLock()
        self.model = create_depth_model(model_type)
        prepare_vram(device, 0)
        self.model.load(gpu=device_to_gpu_id(device), resolution=resolution, limit_resolution=limit_resolution)
        if not self.model.loaded():
            raise RuntimeError(f"Failed to load depth model `{model_type}`")
        self.offload_device = torch.device("cpu")
        if not keep_in_vram:
            self._offload()

    def modules(self) -> list[nn.Module]:
        return _modules_of([self.model.model, self.model.depth_aa])

    def is_metric(self) -> bool:
        return bool(self.model.is_metric())

    def _offload(self) -> None:
        _move(self.modules(), self.offload_device)
        release_vram()

    @contextlib.contextmanager
    def use(self) -> Iterator[BaseDepthModel]:
        with self.lock:
            modules = self.modules()
            prepare_vram(self.device, _module_bytes(modules))
            _move(modules, self.device)
            try:
                yield self.model
            finally:
                self.model.reset()
                if not self.keep_in_vram:
                    self._offload()


class _StereoModelCache:
    """Keeps the most recently used stereo/inpaint model in RAM between runs."""

    def __init__(self) -> None:
        self.key: tuple | None = None
        self.model: Any = None
        self.lock = threading.RLock()

    @staticmethod
    def make_key(
        method: str,
        divergence: float,
        device: torch.device,
        inpaint_model: str | None,
        overlap_frames: list[int] | None,
    ) -> tuple:
        level = get_mlbw_divergence_level(divergence) if method.startswith("mlbw_") else 0
        if method in INPAINT_METHODS:
            inpaint_key = (method, level, inpaint_model, tuple(overlap_frames or []))
        else:
            inpaint_key = (method, level, None, ())
        return (*inpaint_key, str(device))

    def clear(self) -> None:
        with self.lock:
            self.key = None
            self.model = None
            release_vram()

    @contextlib.contextmanager
    def use(
        self,
        method: str,
        divergence: float,
        device: torch.device,
        inpaint_model: str | None = None,
        overlap_frames: list[int] | None = None,
        keep_in_vram: bool = False,
    ) -> Iterator[Any]:
        with self.lock:
            key = self.make_key(method, divergence, device, inpaint_model, overlap_frames)
            if key != self.key:
                self.clear()
                prepare_vram(device, 0)
                self.model = create_stereo_model(
                    method,
                    divergence=divergence,
                    device_id=device_to_gpu_id(device),
                    inpaint_model=inpaint_model if method in INPAINT_METHODS else None,
                    overlap_frames=overlap_frames if method in INPAINT_METHODS else None,
                )
                self.key = key
            model = self.model
            modules = _modules_of([model])
            prepare_vram(device, _module_bytes(modules))
            _move(modules, device)
            try:
                yield model
            finally:
                if model is not None and hasattr(model, "reset"):
                    model.reset()
                if not keep_in_vram:
                    _move(modules, torch.device("cpu"))
                    release_vram()


stereo_models = _StereoModelCache()
