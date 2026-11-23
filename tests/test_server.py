"""End-to-end test: start a real ComfyUI server with this custom node and run workflows through its HTTP API."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest
from conftest import COMFYUI_ROOT
from PIL import Image
from scene import probe_video, read_frame, render_clip, write_video

pytestmark = pytest.mark.server

H, W = 256, 448
IW3_NODES = ["IW3LoadDepthModel", "IW3EstimateDepth", "IW3DepthFromImage", "IW3Stereo", "IW3ConvertVideo"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ComfyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get(self, path: str):
        with urllib.request.urlopen(self.base_url + path, timeout=30) as res:
            return json.loads(res.read())

    def post(self, path: str, data: dict):
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                return json.loads(res.read())
        except urllib.error.HTTPError as e:
            raise AssertionError(f"POST {path} failed: {e.code} {e.read().decode()}") from e

    def run(self, prompt: dict, timeout: float = 900) -> dict:
        prompt_id = self.post("/prompt", {"prompt": prompt, "client_id": uuid.uuid4().hex})["prompt_id"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            history = self.get(f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") in {"success", "error"}:
                    assert status.get("status_str") == "success", json.dumps(status, indent=1)[-4000:]
                    return entry
            time.sleep(1)
        raise TimeoutError(prompt_id)


@pytest.fixture(scope="module")
def comfy(require_gpu, artifacts_dir):
    port = _free_port()
    log_path = Path(artifacts_dir) / "comfyui_server.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch"],
        cwd=COMFYUI_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    client = ComfyClient(f"http://127.0.0.1:{port}")
    try:
        deadline = time.time() + 300
        while True:
            if proc.poll() is not None:
                raise RuntimeError("ComfyUI exited:\n" + log_path.read_text()[-4000:])
            try:
                client.get("/system_stats")
                break
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("ComfyUI did not start:\n" + log_path.read_text()[-4000:])
                time.sleep(1)
        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@pytest.fixture(scope="module")
def object_info(comfy):
    return comfy.get("/object_info")


@pytest.fixture(scope="module")
def input_video():
    name = f"iw3_test_{uuid.uuid4().hex[:8]}.mp4"
    path = os.path.join(COMFYUI_ROOT, "input", name)
    write_video(path, render_clip(24, H, W), fps=16, with_audio=True)
    yield name
    os.unlink(path)


def node(object_info: dict, class_type: str, **inputs) -> dict:
    """Build an API-format node with required widget defaults filled from /object_info."""
    values = {}
    for name, spec in object_info[class_type]["input"].get("required", {}).items():
        if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
            values[name] = spec[1]["default"]
    values.update(inputs)
    return {"class_type": class_type, "inputs": values}


def _output_files(entry: dict, node_id: str) -> list[str]:
    files = []
    for item in entry["outputs"][node_id]["images"]:
        files.append(os.path.join(COMFYUI_ROOT, item["type"], item["subfolder"], item["filename"]))
    return files


def test_nodes_registered(object_info):
    for name in IW3_NODES:
        assert name in object_info, name
        assert object_info[name]["category"] == "iw3"
    depth_models = object_info["IW3LoadDepthModel"]["input"]["required"]["depth_model"]
    assert "Any_V2_S" in json.dumps(depth_models)
    # Advanced inputs are optional so API prompts can omit them.
    assert "divergence" in object_info["IW3Stereo"]["input"]["required"]
    assert "ipd_offset" in object_info["IW3Stereo"]["input"]["optional"]


def test_image_batch_workflow(comfy, object_info, input_video, artifacts_dir):
    prompt = {
        "1": node(object_info, "LoadVideo", file=input_video),
        "2": node(object_info, "GetVideoComponents", video=["1", 0]),
        "3": node(object_info, "IW3LoadDepthModel", depth_model="Any_V2_S"),
        "4": node(object_info, "IW3EstimateDepth", images=["2", 0], depth_model=["3", 0]),
        "5": node(object_info, "IW3Stereo", images=["2", 0], depth=["4", 0], stereo_format="Full SBS"),
        "6": node(object_info, "CreateVideo", images=["5", 0], fps=["2", 2], audio=["2", 1]),
        "7": node(
            object_info,
            "SaveVideo",
            video=["6", 0],
            filename_prefix="iw3_test/sbs_LRF_Full_SBS",
            format="mp4",
            **{"format.codec": "auto"},
        ),
    }
    entry = comfy.run(prompt)
    (path,) = _output_files(entry, "7")
    info = probe_video(path)
    assert (info["width"], info["height"]) == (W * 2, H)
    assert info["frames"] == 24
    assert info["has_audio"]
    Image.fromarray(read_frame(path, 12)).save(os.path.join(artifacts_dir, "server_image_batch_frame.png"))


def test_convert_video_workflow(comfy, object_info, input_video, artifacts_dir):
    prompt = {
        "1": node(object_info, "LoadVideo", file=input_video),
        "2": node(object_info, "IW3LoadDepthModel"),  # default model
        "3": node(object_info, "IW3ConvertVideo", video=["1", 0], depth_model=["2", 0], stereo_format="VR90"),
        "4": node(
            object_info,
            "SaveVideo",
            video=["3", 0],
            filename_prefix="iw3_test/vr_180x180_LR",
            format="mp4",
            **{"format.codec": "auto"},
        ),
    }
    entry = comfy.run(prompt)
    (preview,) = _output_files(entry, "3")
    assert os.path.isfile(preview)
    (path,) = _output_files(entry, "4")
    info = probe_video(path)
    assert info["frames"] == 24
    assert info["has_audio"]
    assert info["width"] == info["height"] * 2
    Image.fromarray(read_frame(path, 12)).save(os.path.join(artifacts_dir, "server_vr90_frame.png"))


def test_example_workflows(comfy, object_info):
    from conftest import REPO_ROOT

    workflow_dir = os.path.join(REPO_ROOT, "example_workflows")
    names = [os.path.splitext(f)[0] for f in os.listdir(workflow_dir) if f.endswith(".json")]
    assert set(comfy.get("/api/workflow_templates")["iw3_for_comfyui"]) == set(names)
    for name in names:
        with open(os.path.join(workflow_dir, f"{name}.json")) as f:
            workflow = json.load(f)
        for item in workflow["nodes"]:
            assert item["type"] in object_info, f"{name}: unknown node {item['type']}"
