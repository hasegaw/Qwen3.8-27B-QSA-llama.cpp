#!/bin/bash
# Standalone timing/check harnesses of the A7 stages (they #include a7-gdn.cu). Run with the server stopped (they need ~3 GB of VRAM).
#   ./build.sh && ./bench_ffn 35 0   # router, gate/up (256 vs 512 threads, outputs compared), down; N = 35 experts, device 0
#   ./bench_attn 0                   # GDN projection stage, 256 vs 512 threads (outputs compared)
set -e
H="$(cd "$(dirname "$0")" && pwd)"; R="$(cd "$H/../../.." && pwd)"; NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
for b in bench_ffn bench_attn; do
  "$NVCC" -O3 -std=c++17 -arch=sm_70 -Wno-deprecated-gpu-targets -I"$R/ggml/src/ggml-cuda" -I"$R/ggml/src" -I"$R/ggml/include" -DGGML_USE_CUDA \
    -o "$H/$b" "$H/$b.cu" -L"$R/build/bin" -lggml-cuda -lggml-base -Xlinker -rpath="$R/build/bin"
done
