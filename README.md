# Qwen3.8-27B QSA for llama.cpp

![Qwen3.8-27B + QSA teaser](docs/assets/qsa-teaser.jpg)

[English](README.md) | [日本語](README.ja.md)

[QSA Paper](https://arxiv.org/abs/2608.30320) | [Inference Engine](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp) | [GGUF Model Series](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF) | [BF16 Weights](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16) | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | [X](https://x.com/_ryu15_)

This repository is the custom llama.cpp inference engine for Qwen3.8-27B with QSA, multimodal input, MTP, and live prefix-cache reuse. The [GGUF model series](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF) includes Q8_0 and Unsloth UD-Q4_K_M, UD-Q5_K_M, and UD-Q6_K variants. The architecture reference is [*On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*](https://arxiv.org/abs/2608.30320).

## Model variants

| Variant | Source and size |
| --- | --- |
| `Qwen3.8-27B-QSA-Q8_0.gguf` | Q8_0 conversion from the QSA BF16 release, 29.26 GB |
| `Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf` | Unsloth UD mixed-precision recipe, 16.67 GB |
| `Qwen3.8-27B-QSA-Unsloth-UD-Q5_K_M.gguf` | Unsloth UD mixed-precision recipe, 19.98 GB |
| `Qwen3.8-27B-QSA-Unsloth-UD-Q6_K.gguf` | Unsloth UD mixed-precision recipe, 22.19 GB |

The Unsloth variants copy all 866 original tensors, including 15 MTP tensors, from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) byte-for-byte. The same 80 FP32 QSA indexer tensors were added without retraining or requantization. UD is a mixed-precision recipe, not a uniform bit depth for every tensor.

## QSA in brief

Qwen Sparse Attention (QSA) uses a learned, compressed indexer to select context in four-token blocks. In this 27B adaptation, the 16 full-attention layers use a fixed token budget of K=2,048 plus the current block. The other 48 Gated DeltaNet layers retain their architecture.

All KV entries remain allocated. QSA reduces attention computation and KV reads; it does not compress the KV cache. The native engine supports text, images, and video with M-RoPE, MTP with up to two draft tokens, and target/indexer/MTP cache reuse in one live slot. The cache is not stored on disk and does not survive a server restart.

![Architecture of the original Qwen3.8-27B and the QSA adaptation](docs/assets/qsa-architecture-en.png)

Original Qwen3.8-27B (left) and the QSA adaptation with 16 learned indexers (right).

## Quick install and start

The captured runtime targets Linux x86_64 and NVIDIA sm70. Its reference configuration uses two 32GB GPUs, CUDA 12.9, FP16 KV, and a 262,144-token context. This quick install selects Q8_0 and downloads only the runtime files it needs:

```sh
python3 -m pip install --user -U huggingface_hub
hf download higa2515/Qwen3.8-27B-QSA-GGUF \
  Qwen3.8-27B-QSA-Q8_0.gguf mmproj-Qwen3.8-27B-QSA-F16.gguf \
  qsa27b-linux-x86_64-sm70.tar.gz SHA256SUMS --local-dir qsa27b
cd qsa27b
grep -E '  (Qwen3[.]8-27B-QSA-Q8_0[.]gguf|mmproj-Qwen3[.]8-27B-QSA-F16[.]gguf|qsa27b-linux-x86_64-sm70[.]tar[.]gz)$' SHA256SUMS | sha256sum -c -
tar -xzf qsa27b-linux-x86_64-sm70.tar.gz
QSA_MODEL_DIR="$PWD" ./qsa27b-runtime/launch.sh
```

The server listens on 127.0.0.1:8091 by default, with the OpenAI-compatible API at `http://127.0.0.1:8091/v1`. Set `QSA_HOST=0.0.0.0` only when remote access is required. Do not run another model on the same GPUs.

For Unsloth model selection, GPU-specific build settings, single- and multi-GPU launch examples, and measured VRAM use, see the [GPU setup guide](qsa27b/GPU-GUIDE.md).

## Build from source

Use a source build when the captured sm70 binary does not match the target system. CMake, Ninja, a C++ compiler, and the CUDA 12.9 toolkit are required.

```sh
git clone https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp.git
cd Qwen3.8-27B-QSA-llama.cpp
CUDA_HOME=/usr/local/cuda-12.9 bash qsa27b/build.sh
```

The default binary is `build-qsa/bin/llama-server`. Use the same launch settings with that binary by setting `QSA_BIN`:

```sh
QSA_BIN="$PWD/build-qsa/bin/llama-server" \
QSA_MODEL=Qwen3.8-27B-QSA-Q8_0.gguf \
QSA_MODEL_DIR=/path/to/gguf \
bash qsa27b/launch-captured-runtime.sh
```

`QSA_BUILD_DIR`, `QSA_CUDA_ARCH`, and `QSA_BUILD_JOBS` customize the build. A fresh source build is not claimed to reproduce the captured binary byte-for-byte. See the [runtime guide](qsa27b/RUNTIME.md) for conversion and verification details.

## Measured results

### Limited quality checks

These are BF16 research-reference comparisons against the original dense path, not Q8_0 native quality results.

| Check | Original dense | QSA | Observed comparison |
| --- | ---: | ---: | --- |
| Text PPL on 32,766 tokens, lower is better | 7.072960 | 7.071384 | QSA/dense ratio 0.99978 |
| Constructed 50-item four-choice DocVQA | 50/50 | 50/50 | 100% of the dense score on this probe |

The DocVQA probe is not standard open-ended DocVQA scoring. These small checks do not establish a general performance-retention percentage or broad quality equivalence. Native multimodal probes are limited to checks through 64k.

### Unsloth Q4/Q5/Q6 text checks

A teacher-forced evaluation used six fixed windows, up to 65,024 input tokens, FP16 KV, and MTP disabled. PPL differences are QSA relative to dense within each quantized model.

| Variant | Overall PPL difference | 64k capture | Same-budget oracle | 64k prose PPL difference |
| --- | ---: | ---: | ---: | ---: |
| UD-Q4_K_M | -1.41% | 73.82% | 82.93% | +2.44% |
| UD-Q5_K_M | -0.47% | 73.76% | 82.91% | +1.97% |
| UD-Q6_K | -0.35% | 73.89% | 83.04% | +1.76% |

Capture measures attention mass, not task accuracy. The oracle maximizes that mass under the same budget. These text-only checks do not establish general quality equivalence, free-generation quality, or a comparison with BF16. See [the complete evaluation](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF/tree/main/unsloth-qsa).

### Q8_0 throughput: QSA vs dense

The following results apply only to Q8_0, not to the Unsloth variants. The same Q8_0 GGUF, FP16 KV, MTP draft2, engine, and two sm70 32GB GPUs were used for both paths. Dense restores the 16 original full-attention layers with `LLAMA_QSA_DENSE=1`; the 48 Gated DeltaNet layers stay unchanged.

| Input tokens | QSA cold prefill | Dense cold prefill | QSA decode | Dense decode |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 660.75 | 658.87 | 49.28 | 49.87 |
| 261,120 | 1071.53 | 366.84 | 41.07 | 22.91 |

All values are token/s. At 261,120 input tokens, QSA was 1.793x faster in decode and 2.921x faster in cold prefill. From each path's re-measured 0k point, mixed decode declined 16.66% for QSA and 54.06% for dense. The workload mixed two coding tasks and two free-prose tasks. The 0k prompts still contain 490-563 tokens.

QSA's nonzero-depth samples were retained from the earlier matched run; QSA and dense were measured in separate sessions, not interleaved A/B repeats. No confidence interval or general quality claim is made. See the [full comparison data and methodology](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF/tree/main/benchmark).

![Measured QSA and Dense prefill and decode throughput](docs/assets/qsa-vs-dense.png)

## Contributors and credits

- [kuraneko1](https://github.com/kuraneko1): QSA 27B integration, multimodal runtime, and release engineering.
- [pentaCoxian](https://github.com/pentaCoxian): V100-oriented llama.cpp base and runtime optimizations.
- [Warabii1248](https://github.com/Warabii1248): V100 patch series included in the base engine.
- [ggml-org contributors](https://github.com/ggml-org/llama.cpp): upstream llama.cpp.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B): original model and the [QSA architecture paper](https://arxiv.org/abs/2608.30320).
- [Unsloth](https://github.com/unslothai/unsloth) and the [Unsloth Qwen3.8-27B GGUF repository](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF): UD mixed-precision recipes and the 866 source tensors reused in the Q4/Q5/Q6 variants.

The upstream README is preserved in [README.upstream.md](README.upstream.md). Engine code retains MIT and third-party notices. Model files retain Apache-2.0 and their upstream notices.
