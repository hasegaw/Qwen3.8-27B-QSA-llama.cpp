#!/bin/bash
# Vast.ai base image: supervisor wrapper (copy to /opt/supervisor-scripts/qwen38.sh). Set QWEN38_* in ${WORKSPACE}/.env or below.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export QWEN38_REPO="${QWEN38_REPO:-/workspace/src/llama.cpp}"
export QWEN38_MODEL="${QWEN38_MODEL:-/workspace/models/qwen38/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf}"
export QWEN38_MMPROJ="${QWEN38_MMPROJ-/workspace/models/qwen38/mmproj-F16.gguf}"
export QWEN38_CACHE_DIR="${QWEN38_CACHE_DIR:-/workspace/qwen38}"
# CUDA 12.x runtime and a CUDA 12 NCCL build (the CUDA 13 NCCL of the image has no sm_70 kernels)
export LD_LIBRARY_PATH="${QWEN38_NCCL_LIB:-/opt/nccl-cu129/usr/lib/x86_64-linux-gnu}:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
pty "$QWEN38_REPO/deploy/qwen38-v100/run-server.sh" 2>&1
