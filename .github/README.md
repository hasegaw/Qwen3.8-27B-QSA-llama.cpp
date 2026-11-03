# llama.cpp for Qwen3.8-Flash-Next on V100

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) `v0.5.0` for
**[Qwen3.8-Flash-Next-IQ3E-Q8D-MTP](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF)**
on **two V100 32 GB cards**, without NVLink. It covers MTP speculative decoding, image input and a 192K context, and runs
on a host with 32 GB of RAM.

This is not the official llama.cpp. Report problems with this fork here, not upstream. The upstream README is
[`README.md`](../README.md); the work is on branch `qwen38-v100`.

## Speed

Two V100-SXM2-32GB cards, layer split, 200 W cap. Full method and more numbers:
[deploy/qwen38-v100/README.md](../deploy/qwen38-v100/README.md#measured).

| Workload | Decode tok/s | MTP acceptance |
|---|---:|---:|
| Code edit / JSON extraction | 187-188 | 94-95% |
| Code generation | 174 | 84% |
| Technical explanation | 130 | 62% |
| Long-context Q&A at 142K | 130 | 73% |
| Creative prose | 94 | 42% |

- **Prefill:** 2,433 tok/s at 34K and 2,558 tok/s at 142K.
- **Images:** 1.1 s for 640x360, 3.2 s for 1920x1080.
- **Prompt cache:** a follow-up turn on a long chat re-prefills only the new tokens.

## Quick start

```sh
git clone -b qwen38-v100 https://github.com/pentaCoxian/Qwen-V100-llama.cpp && cd Qwen-V100-llama.cpp
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_SCHED_MAX_COPIES=2 -DGGML_CUDA_NCCL=ON \
      -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda-12.9      # CUDA 12.x: CUDA 13 dropped sm_70
cmake --build build -j --target llama-server

hf download pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf --local-dir models
hf download unsloth/Qwen3.8-Flash-Next-GGUF mmproj-F16.gguf --revision c8b5954 --local-dir models

QWEN38_MODEL=models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf QWEN38_MMPROJ=models/mmproj-F16.gguf \
  deploy/qwen38-v100/run-server.sh          # OpenAI-compatible API on 127.0.0.1:18080/v1
```

[deploy/qwen38-v100/README.md](../deploy/qwen38-v100/README.md) covers the launcher options, the Vast.ai supervisor
files, every setting and switch, memory and VRAM headroom, the benchmarks and checks, and how to move to a newer upstream.

## Credits

- **[Warabii1248](https://github.com/Warabii1248)**: [qwen38-v100-patches](https://github.com/Warabii1248/qwen38-v100-patches)
  `pcie-20261016`, 56 prefill and decode features for two V100 cards without NVLink. Included unchanged as a commit
  authored by them.
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** and its contributors.

## License

- **Code:** MIT ([LICENSE](../LICENSE)). The FreeToken-derived files (`freetoken-*`) are Apache-2.0
  ([LICENSE-freetoken](../LICENSE-freetoken)), and Warabii1248's patches are MIT
  ([LICENSE-pcie-patches](../LICENSE-pcie-patches)).
- **Model weights:** not part of this repository. They are under the Qwen Community License 1.0.
