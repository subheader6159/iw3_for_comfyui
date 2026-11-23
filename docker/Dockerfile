# Test environment: ComfyUI + CUDA PyTorch. The custom node is bind-mounted at runtime (see docker/run_tests.sh).
FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# gcc: Triton (used by some PyTorch ops) compiles a small CUDA helper at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}

ARG COMFYUI_COMMIT=4c2b826d74a1827f5cc4a706608fc4653ed12972
RUN git init -q /opt/ComfyUI \
    && cd /opt/ComfyUI \
    && git remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git fetch -q --depth 1 origin ${COMFYUI_COMMIT} \
    && git checkout -q FETCH_HEAD \
    && rm -rf .git
RUN pip install -r /opt/ComfyUI/requirements.txt

COPY requirements.txt /tmp/iw3-requirements.txt
RUN pip install -r /tmp/iw3-requirements.txt pytest

# Tests run as the host user; let them write ComfyUI's data dirs and the model volume.
RUN mkdir -p /opt/ComfyUI/models/iw3 /opt/ComfyUI/custom_nodes /opt/ComfyUI/input /opt/ComfyUI/output \
        /opt/ComfyUI/temp /opt/ComfyUI/user \
    && chmod -R a+rwX /opt/ComfyUI

ENV HOME=/tmp \
    COMFYUI_ROOT=/opt/ComfyUI
WORKDIR /opt/ComfyUI
