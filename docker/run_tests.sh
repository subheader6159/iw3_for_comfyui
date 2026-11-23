#!/usr/bin/env bash
# Build the ComfyUI test image and run the test suite on the GPU.
# Extra arguments are passed to pytest, e.g. `docker/run_tests.sh -k stereo`.
# IW3_NUNIF_DIR=/path/to/nunif tests against that checkout instead of the submodule (e.g. unpushed changes).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IW3_TEST_IMAGE:-iw3-for-comfyui-test}"
MODELS_VOLUME="${IW3_MODELS_VOLUME:-iw3-for-comfyui-models}"
NODE_DIR=/opt/ComfyUI/custom_nodes/iw3_for_comfyui

docker build -t "$IMAGE" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"
mkdir -p "$REPO_ROOT/test_artifacts"

NUNIF_MOUNT=()
if [[ -n "${IW3_NUNIF_DIR:-}" ]]; then
    NUNIF_MOUNT=(-v "$(cd "$IW3_NUNIF_DIR" && pwd):$NODE_DIR/nunif:ro")
fi

docker run --rm --gpus all --shm-size 2g \
    --user "$(id -u):$(id -g)" \
    -v "$REPO_ROOT:$NODE_DIR" \
    "${NUNIF_MOUNT[@]}" \
    -v "$MODELS_VOLUME:/opt/ComfyUI/models/iw3" \
    -e IW3_TEST_ARTIFACTS="$NODE_DIR/test_artifacts" \
    "$IMAGE" \
    python -m pytest -p no:cacheprovider -v "$NODE_DIR/tests" "$@"
