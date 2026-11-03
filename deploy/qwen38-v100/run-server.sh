#!/bin/bash
# qwen38-v100 launcher: llama-server for Qwen3.8-Flash-Next-IQ3E-Q8D-MTP on 2x V100 32 GB with the measured settings.
# Required: QWEN38_MODEL (the .gguf). Optional (defaults below): QWEN38_MMPROJ, QWEN38_HOST, QWEN38_PORT, QWEN38_CTX,
# QWEN38_IMAGE_MAX_TOKENS, QWEN38_CACHE_DIR, QWEN38_NUMA_NODE (empty = no binding), LLAMA_SERVER. Extra llama-server arguments: pass them to this script.
set -euo pipefail
QWEN38_DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$QWEN38_DEPLOY/../.." && pwd)"
: "${QWEN38_MODEL:?set QWEN38_MODEL to Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf}"
QWEN38_MMPROJ="${QWEN38_MMPROJ:-}"
QWEN38_IMAGE_MAX_TOKENS="${QWEN38_IMAGE_MAX_TOKENS:-}"
QWEN38_HOST="${QWEN38_HOST:-127.0.0.1}"
QWEN38_PORT="${QWEN38_PORT:-18080}"
QWEN38_CTX="${QWEN38_CTX:-196608}"
QWEN38_CACHE_DIR="${QWEN38_CACHE_DIR:-$HOME/.cache/qwen38}"
QWEN38_NUMA_NODE="${QWEN38_NUMA_NODE-0}"
LLAMA_SERVER="${LLAMA_SERVER:-$REPO/build/bin/llama-server}"
export QWEN38_DEPLOY QWEN38_CACHE_DIR
mkdir -p "$QWEN38_CACHE_DIR/pc-disk"
set -a; . "$QWEN38_DEPLOY/qwen38.env"; set +a

vision=()
if [ -n "$QWEN38_MMPROJ" ]; then
  vision=(--mmproj "$QWEN38_MMPROJ" --mmproj-device CUDA0)   # resident on CUDA0 (~0.9 GB + compute; images cap at ~4.1K tokens)
  if [ -n "$QWEN38_IMAGE_MAX_TOKENS" ]; then
    vision+=(--image-max-tokens "$QWEN38_IMAGE_MAX_TOKENS")  # larger images are downscaled: bounds the vision compute buffer
  fi
fi
numa=()
if [ -n "$QWEN38_NUMA_NODE" ] && command -v numactl >/dev/null; then
  numa=(numactl --cpunodebind="$QWEN38_NUMA_NODE")           # CPU threads next to CUDA0; --membind is not allowed in some containers
fi

exec "${numa[@]}" "$LLAMA_SERVER" \
  --model "$QWEN38_MODEL" "${vision[@]}" \
  --host "$QWEN38_HOST" --port "$QWEN38_PORT" --alias qwen3.8-flash-next \
  --ctx-size "$QWEN38_CTX" --parallel 1 --threads 8 --threads-batch 16 \
  --batch-size 3072 --ubatch-size 1536 --n-gpu-layers 99 --fit off \
  --split-mode layer --tensor-split 26,24 --main-gpu 0 --moe-expert-cache 0 --moe-expert-cache-inserts 2 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap \
  --cache-ram 4096 --ctx-checkpoints 24 --checkpoint-min-step 2048 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-sampling probabilistic --jinja \
  "$@"
