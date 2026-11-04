#!/usr/bin/env bash
set -euo pipefail
qsa_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
qsa_build="${QSA_BUILD_DIR:-$qsa_source/build-qsa}"
qsa_cuda="${CUDA_HOME:-/usr/local/cuda-12.9}"
cmake -S "$qsa_source" -B "$qsa_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$qsa_cuda/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${QSA_CUDA_ARCH:-70}" \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16' \
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON \
  -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX_VNNI=OFF \
  -DGGML_AVX512=OFF -DGGML_AVX512_BF16=OFF \
  -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=ON
cmake --build "$qsa_build" --target llama-server --parallel "${QSA_BUILD_JOBS:-8}"
