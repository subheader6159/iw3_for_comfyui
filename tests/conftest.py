import importlib.util
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI")
PACKAGE_NAME = os.path.basename(REPO_ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, COMFYUI_ROOT)


def _import_custom_node():
    """Import the custom node package the way ComfyUI does (as a package named after its directory)."""
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, os.path.join(REPO_ROOT, "__init__.py"), submodule_search_locations=[REPO_ROOT]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def custom_node():
    return _import_custom_node()


@pytest.fixture(scope="session")
def nodes(custom_node):
    return sys.modules[f"{PACKAGE_NAME}.iw3_comfyui.nodes"]


@pytest.fixture(scope="session")
def artifacts_dir():
    path = os.environ.get("IW3_TEST_ARTIFACTS", os.path.join(REPO_ROOT, "test_artifacts"))
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
