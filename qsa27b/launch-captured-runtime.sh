#!/usr/bin/env bash
set -euo pipefail
qsa_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qsa_model_dir="${QSA_MODEL_DIR:-$(dirname -- "$qsa_root")}"
export LD_LIBRARY_PATH="$qsa_root/bin:$qsa_root/lib:${CUDA_HOME:-/usr/local/cuda-12.9}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LLAMA_QSA_RESERVE_BLK=1
export LLAMA_QSA_BLK_REBUILD=1
export LLAMA_QSA27B_MASKED_PREFILL_MAX=8192
export LLAMA_MTP_KV_ONLY=1
export LLAMA_MTP_PAD=4
export LLAMA_MTP_N_UBATCH=64
export LLAMA_MTP_LAG=0
export LLAMA_MTP_ASYNC=0
export LLAMA_REUSE_OUT_ORDER=1
export GGML_CUDA_GRAPH_OPT=1
export GGML_FT_GRAPH_CACHE=1
export GGML_FT_GRAPH_CACHE_MIN_NODES=32
export CUDA_SCALE_LAUNCH_QUEUES=4x
export LLAMA_PP_PLAN_HINT=1
export LLAMA_PP_PLAN_AHEAD=1
export LLAMA_PP_PLAN_AHEAD_MAX_KV=0
export LLAMA_KV_PAD_GEO=2
export LLAMA_PP_REUSE_MAX_TOKENS=16
export LLAMA_CKPT_TAIL_UBATCH=1
exec "$qsa_root/bin/llama-server" \
  -m "$qsa_model_dir/Qwen3.8-27B-QSA-mm2-Q8_0.gguf" \
  --mmproj "$qsa_model_dir/mmproj-Qwen3.8-27B-QSA-F16.gguf" \
  --alias qwen3.8-27b-qsa,qwen3.8-27b-qsa-text,qwen3.8-27b-qsa-mm2 \
  -ngl all --device CUDA0,CUDA1 -sm layer -ts 1,1 -fa on -np 1 --fit off \
  -t 8 -tb 16 -c "${QSA_CONTEXT:-262144}" -b 2048 -ub 512 -ctk f16 -ctv f16 \
  --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 1024 --no-context-shift \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0 \
  --spec-draft-sampling probabilistic --host "${QSA_HOST:-127.0.0.1}" \
  --port "${QSA_PORT:-8091}" --jinja "$@"
