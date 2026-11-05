# GPU setup and launch guide

[日本語](GPU-GUIDE.ja.md) | [Main README](../README.md)

Use the [QSA-enabled engine](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp) with the [QSA GGUF files](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF). Stock llama.cpp does not implement this QSA path.

## Configurations tested

The following single-GPU configurations loaded and generated short text. VRAM figures are GPU-wide observations, including the F16 mmproj and MTP with up to two draft tokens. They can vary with the driver, other processes, and runtime settings.

| GPU | Model | Context capacity | Target KV | Observed VRAM |
| --- | --- | ---: | --- | ---: |
| RTX 3090 24 GB | UD-Q5_K_M | 65,536 | Q8_0 | 23,779 / 24,576 MiB |
| Tesla V100 32 GB | UD-Q6_K | 65,536 | Q8_0 | 25,944 / 32,768 MiB |
| Tesla V100 32 GB | UD-Q4_K_M | 262,144 | Q8_0 | about 28,102 / 32,768 MiB |

The Q5 and Q6 configurations placed all model tensors, including input embeddings, on the GPU. The Q4 configuration left the input embedding on the CPU. Context capacity is not the length of the input used in the startup check: Q5/Q6 generated short responses, while the Q4 trial used up to 16,064 input tokens. These are not broad quality or throughput benchmarks. Image/video support was loaded; the Q5/Q6 startup checks did not assess image/video answers.

## Build for your GPU

Linux, CMake, Ninja, a C++ compiler and the CUDA toolkit are required. The following architecture values are build targets; inclusion in this table does not mean the model has been tested on that GPU.

| GPU | `QSA_CUDA_ARCH` |
| --- | --- |
| Tesla V100 | `70` |
| RTX 3090 / 3090 Ti | `86` |
| RTX 4090 | `89` |

```sh
git clone https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp.git
cd Qwen3.8-27B-QSA-llama.cpp
# Example: RTX 3090. Set CUDA_HOME to your toolkit installation.
QSA_CUDA_ARCH=86 CUDA_HOME=/usr/local/cuda-12.9 bash qsa27b/build.sh
```

The server is built at `build-qsa/bin/llama-server` by default. The captured runtime archive targets Linux x86_64 and NVIDIA sm70; build from source for other GPU architectures. A source build is not guaranteed to be byte-identical to the captured runtime.

## Download a model

With the Hugging Face CLI installed, this example downloads the Q5 model, shared image/video projector and the three Unsloth variant checksums. Replace the Q5 filename with the Q4 or Q6 filename to select a different model. The projector is separate from the language model GGUF.

```sh
hf download higa2515/Qwen3.8-27B-QSA-GGUF \
  Qwen3.8-27B-QSA-Unsloth-UD-Q5_K_M.gguf \
  mmproj-Qwen3.8-27B-QSA-F16.gguf unsloth-qsa/SHA256SUMS \
  --local-dir models
(cd models && grep -F 'Qwen3.8-27B-QSA-Unsloth-UD-Q5_K_M.gguf' unsloth-qsa/SHA256SUMS | sha256sum -c -)
```

## Run on one GPU

This example follows the tested 65,536-context, Q8_0 target KV and MTP draft-2 setup. Select a GPU using its index from `nvidia-smi`; after selecting one visible GPU, the server calls it `CUDA0`. The example uses localhost. Replace the model path to use Q4 or Q6.

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export QSA_MODEL="$PWD/models/Qwen3.8-27B-QSA-Unsloth-UD-Q5_K_M.gguf"
export QSA_MMPROJ="$PWD/models/mmproj-Qwen3.8-27B-QSA-F16.gguf"
export LLAMA_QSA_RESERVE_BLK=1 LLAMA_QSA_BLK_REBUILD=1
export LLAMA_QSA27B_MASKED_PREFILL_MAX=8192 LLAMA_QSA_RAW_RING=8192
export LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_PAD=4 LLAMA_MTP_N_UBATCH=64
export LLAMA_MTP_LAG=0 LLAMA_MTP_ASYNC=0 LLAMA_SPEC_PREFILL_TAIL=2048
export LLAMA_REUSE_OUT_ORDER=1 LLAMA_KV_PAD_GEO=2 LLAMA_CKPT_TAIL_UBATCH=1
export GGML_FT_GRAPH_CACHE=0 LLAMA_GPU_ARGMAX=1 LLAMA_SAMPLE_TOPK=1
export GGML_CUDA_TOPK_DET=1 CUDA_SCALE_LAUNCH_QUEUES=4x

./build-qsa/bin/llama-server \
  -m "$QSA_MODEL" --mmproj "$QSA_MMPROJ" \
  --device CUDA0 --device-draft CUDA0 --mmproj-device CUDA0 \
  -ngl all -ngld all -sm none --fit off \
  --override-tensor token_embd.weight=CUDA0 \
  -fa on -np 1 -t 4 -tb 4 -td 4 -tbd 4 \
  -c 65536 -b 2048 -ub 256 \
  -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 \
  --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 1024 --no-context-shift \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0 \
  --spec-draft-sampling probabilistic --jinja --host 127.0.0.1 --port 8091
```

The explicit tensor override also works with an engine version whose default keeps the input embedding on the CPU. It uses additional VRAM; if memory is insufficient, reduce `-c` or change the override to `--override-tensor token_embd.weight=CPU` to place the embedding on the CPU. This does not guarantee that every model will fit. CPU offload can reduce speed.

## Using two GPUs

For a V100 and RTX 3090 source build, for example:

```sh
QSA_CUDA_ARCH='70;86' CUDA_HOME=/usr/local/cuda-12.9 bash qsa27b/build.sh
```

To adapt the single-GPU example, make both devices visible, for example `export CUDA_VISIBLE_DEVICES=0,1`. Replace `--device CUDA0` with `--device CUDA0,CUDA1`, and replace `-sm none` with `-sm layer -ts 1,1`. The example keeps the MTP draft and mmproj on `CUDA0`; check that device has enough free memory. `-ts` specifies a layer allocation ratio in the order of the visible devices. Adjust it for unequal GPU capacities. This is a starting point, not a validated configuration for every GPU pair or model.

## Check the server

```sh
curl -fsS http://127.0.0.1:8091/health
curl -fsS http://127.0.0.1:8091/v1/models
```

The web UI is at `http://127.0.0.1:8091/`. The example has no API key and binds to localhost. If remote access is needed, configure an appropriate access control before changing the host binding.
