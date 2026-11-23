#!/usr/bin/env bash
# Run ComfyUI with this custom node in Docker and serve the UI on http://localhost:8188 (local only).
# Files are shared with the host under comfyui_data/: put videos in comfyui_data/input,
# results appear in comfyui_data/output. Stop with Ctrl+C.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IW3_TEST_IMAGE:-iw3-for-comfyui-test}"
MODELS_VOLUME="${IW3_MODELS_VOLUME:-iw3-for-comfyui-models}"
DATA_DIR="${IW3_COMFYUI_DATA:-$REPO_ROOT/comfyui_data}"
PORT="${IW3_COMFYUI_PORT:-8188}"
NODE_DIR=/opt/ComfyUI/custom_nodes/iw3_for_comfyui

docker build -q -t "$IMAGE" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT" > /dev/null
mkdir -p "$DATA_DIR/input" "$DATA_DIR/output" "$DATA_DIR/user"
echo "ComfyUI: http://localhost:$PORT  (input: $DATA_DIR/input, output: $DATA_DIR/output)"

TTY_FLAGS=()
if [[ -t 0 ]]; then
    TTY_FLAGS=(-it)
fi

docker run --rm "${TTY_FLAGS[@]}" --name "${IW3_COMFYUI_CONTAINER:-iw3-comfyui}" --gpus all --shm-size 2g \
    --user "$(id -u):$(id -g)" \
    -p "127.0.0.1:$PORT:8188" \
    -v "$REPO_ROOT:$NODE_DIR" \
    -v "$MODELS_VOLUME:/opt/ComfyUI/models/iw3" \
    -v "$DATA_DIR/input:/opt/ComfyUI/input" \
    -v "$DATA_DIR/output:/opt/ComfyUI/output" \
    -v "$DATA_DIR/user:/opt/ComfyUI/user" \
    "$IMAGE" \
    python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch
