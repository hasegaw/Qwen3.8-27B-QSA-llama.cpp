# qwen38-v100

A llama.cpp fork for **Qwen3.8-Flash-Next-IQ3E-Q8D-MTP** on **two V100 32 GB cards** (sm_70, CUDA 12.x),
with MTP speculative decoding, the vision encoder, and a 192K context. It runs on a host with 32 GB of RAM.
Maintained by pentaCoxian (pentacoxian-dev on Hugging Face). It replaces the patch files of the model repository.

Much of the prefill speed comes from **Warabii1248**'s
[qwen38-v100-patches](https://github.com/Warabii1248/qwen38-v100-patches) (`pcie-20261016`). This branch includes it
unchanged, as a commit authored by them.

## What is in the branch

The `qwen38-v100` branch is llama.cpp `v0.5.0` plus one commit per source, so you can review, bisect or rebase it:

| Commit | Source | License |
|---|---|---|
| Import `llama-v0.5.0-v100-v3.patch` | pentaCoxian, from the [model repo](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF) at revision `75e6411`: qwen4exp, MTP, sparse QSA attention, fused kernels | MIT (llama.cpp); FreeToken files Apache-2.0, see `LICENSE-freetoken` |
| Import `pcie-20261016` | Warabii1248, from [qwen38-v100-patches](https://github.com/Warabii1248/qwen38-v100-patches): 56 prefill and decode features for 2x V100 without NVLink | MIT, see `LICENSE-pcie-patches` |
| `llama-graph: reuse graphs whose output rows only moved` | this fork, `LLAMA_REUSE_OUT_ORDER=1` | MIT |
| `a7: IQ2_S gate/up stage at 512 threads with exact scales` | this fork, `GGML_A7_GU512=1` | MIT |
| `a7: F16 copy of the router weights` | this fork, `GGML_A7_ROUTER_F16=1` (`=2` is a check mode) | MIT |
| `qwen4exp: fused A7 FFN on full-attention layers` | this fork, `LLAMA_A7_FFN_FULLATTN=1` | MIT |
| `deploy/qwen38-v100` | this directory: config, launcher, draft vocabulary, benchmarks | MIT |

Every new switch is off unless it is set. `qwen38.env` sets all of them. Each commit message gives the measurement and the
correctness check behind its change.

The model weights are not part of this repository. They are under the Qwen Community License 1.0; see the model card.

## Build

You need CUDA 12.x (12.8 or later). CUDA 13 cannot build for sm_70. NCCL must be a CUDA 12 build: a CUDA 13 NCCL has no
sm_70 kernels and fails at the first AllReduce. The default layer split does not use NCCL.

```sh
git clone -b qwen38-v100 <this fork> llama.cpp && cd llama.cpp
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_SCHED_MAX_COPIES=2 \
      -DGGML_CUDA_NCCL=ON -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda-12.9
cmake --build build -j --target llama-server
```

## Model files

```sh
hf download pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
   --revision 75e6411f0157bb4b6d492a9868bc93c57d2b061c --local-dir models          # 85.9 GB
hf download unsloth/Qwen3.8-Flash-Next-GGUF mmproj-F16.gguf --revision c8b5954 --local-dir models   # 0.9 GB, for images
```

## Run

```sh
QWEN38_MODEL=models/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
QWEN38_MMPROJ=models/mmproj-F16.gguf \
deploy/qwen38-v100/run-server.sh            # OpenAI-compatible API on 127.0.0.1:18080/v1
```

| Variable | Default | Meaning |
|---|---|---|
| `QWEN38_MODEL` | required | the model GGUF |
| `QWEN38_MMPROJ` | empty | vision projector. When set, it is loaded once onto CUDA0 and stays there |
| `QWEN38_HOST`, `QWEN38_PORT` | `127.0.0.1`, `18080` | listen address |
| `QWEN38_CTX` | `196608` | context size |
| `QWEN38_CACHE_DIR` | `~/.cache/qwen38` | prompt-cache spill directory (`pc-disk/`). Give it a directory of its own |
| `QWEN38_NUMA_NODE` | `0` | `numactl --cpunodebind` node, normally the one next to CUDA0. Set it empty to disable |
| `LLAMA_SERVER` | `build/bin/llama-server` | binary |

Any arguments after the script name are passed through to llama-server.

On a Vast.ai base image, `vast/` holds the supervisor files. Copy `vast/qwen38.sh` to `/opt/supervisor-scripts/qwen38.sh`
and `vast/qwen38.conf` to `/etc/supervisor/conf.d/qwen38.conf`, then run
`supervisorctl reread && supervisorctl update`. `stopasgroup` and `killasgroup` matter: without them a restart leaves the old
llama-server holding the port.

## Measured

Two Tesla V100-SXM2-32GB cards, NVLink inactive, the GPUs on different NUMA nodes, Xeon E5-2680 v2, 200 W GPU power cap.
These numbers come from this branch launched with `run-server.sh` and the settings in `qwen38.env`, using the scripts in
`bench/`. Thinking was off except for the math task; each figure is the mean of 2 runs.

| Task | Decode tok/s | MTP acceptance |
|---|---:|---:|
| Code edit (rewrite a 130-line file) | 188 | 95% |
| JSON extraction | 187 | 94% |
| Code generation | 174 | 84% |
| Translation to Japanese | 156 | 78% |
| Math, thinking on | 146 | 71% |
| Summarization | 132 | 66% |
| Technical explanation | 130 | 62% |
| Long-context Q&A, 34K prompt | 135 | 72% |
| Long-context Q&A, 142K prompt | 130 | 73% |
| Creative prose | 94 | 42% |
| Japanese chat | 91 | 38% |

- **Mixed benchmark** (`bench/bench.py`): 115.1 tok/s at 54.2% acceptance, 22.75 ms per verify step. The author's
  configuration measured 26.74 ms on the same machine.
- **Prefill:** about 1,200 tok/s for prompts of 1-3K tokens, 2,433 tok/s at 34K and 2,558 tok/s at 142K.
- **Images:** 1.1 s total for 640x360, 3.2 s for 1920x1080, 7.9 s at 4K. Images are capped at about 4.1K tokens, so 8K
  inputs cost the same as 4K.
- **Prompt cache:** appending a turn, regenerating, or editing the last message of a 24K-token chat re-prefills 4-81
  tokens and finishes in under 1 s. A 142K-token conversation pushed out of RAM comes back from disk in 3.8 s; prefilling
  it again would take 57 s.
- **Host RAM:** peak anonymous memory was 5.6 GB in a stress run (142K + 34K + 34K documents, switching back, then an
  image). The 27.5 GB n-gram table and the 0.6 GB token embeddings are memory-mapped page cache, so the kernel can
  reclaim them. Keep `--load-mode mmap`: `mlock`, `none` and `dio` would pin the table or copy it into RAM.
- **VRAM:** CUDA0 peaks at 32.34 of 32.77 GB with vision resident. Check this before raising the context or changing the
  split.

## Settings that differ from the upstream measured config

| Setting | Why |
|---|---|
| CUDA graphs on during decode (`GGML_CUDA_DISABLE_GRAPHS` removed) | about 2,000 kernel launches per step on a slow host CPU; 25.1 -> 24.4 ms |
| `--batch-size 3072` (upstream 15360) | faster prefill (2,077 -> 2,433 tok/s at 34K). Context checkpoints are taken only between batches, so a change inside one long message now re-prefills at most ~2.5K tokens instead of ~12K |
| `--cache-ram 4096`, `--ctx-checkpoints 24`, `--checkpoint-min-step 2048`, `LLAMA_PROMPT_CACHE_DISK` | keeps a 32 GB host within bounds; older conversations spill to disk |
| Vision resident on CUDA0 (`MTMD_ON_DEMAND` removed) | about 0.65 s faster per image, text speed unchanged |
| `LLAMA_MTP_DRAFT_VOCAB=draftvocab-64k-ml.txt` | 64K-row draft head built from the model's own English, code, Japanese, Chinese and Korean output, and from local corpora (`tools/mkvocab*.py`). A list without CJK tokens costs Japanese about 20% |
| `LLAMA_MTP_PAD=4` with `LLAMA_REUSE_OUT_ORDER=1` | a constant draft batch shape: half as many draft-graph rebuilds |
| `GGML_A7_GU512`, `GGML_A7_ROUTER_F16`, `LLAMA_A7_FFN_FULLATTN` | this fork's kernel and graph changes, about 1.2 ms per step together |

## Measured and not worth it on this hardware

- **Tensor split** (with a CUDA 12 NCCL): 34 ms per step. Cross-socket AllReduces and graph-launch overhead outweigh the
  shared work.
- **Greedy drafts** (needed for an on-device draft chain): acceptance drops by 5 points, about -7% real tok/s.
- **`--spec-draft-n-max 2`:** asserts. The lookahead and draft-ahead features assume the 4-token window.
- **Lookahead off:** 2 ms per step slower, even at ~50% acceptance.
- **N-gram lookup in front of MTP:** about break-even on copy-heavy work (173 vs 188 tok/s). Windows over 8 tokens leave
  the MMVQ kernels, and drafts longer than the rollback depth (3) trigger ~35 ms recurrent-state checkpoints.
- **512 threads for the GDN projection stage:** bit-identical but not faster. It already reads ~80% of the card's
  practical bandwidth.
- **Down-stage redesign:** not done. The stage is bound by on-SM execution; the projected gain was under 1%.

The remaining verify window (~20 ms) is bound by the GPUs. The heavy A7 stages run as fast with their weights resident in
L2 as from DRAM, so they are limited by Volta's integer throughput for these quant formats, not by memory.

## Checks and benchmarks

- `bench/bench.py [reps]`: mixed decode benchmark. It reports tok/s, acceptance, ms per verify step, and tok/s
  normalized to 50% acceptance.
- `bench/tasks.py [reps] [task,task] [out.json]`: the per-task table above. The long-context tasks plant a fact in the
  document and check that the answer finds it.
- `bench/cachetest.py`: prompt-cache reuse in chat patterns. `bench/memstress.py`: host-memory peak.
- `bench/copybench.py FILE`: copy-heavy edit. `bench/imgtest.py IMG...`: image answers (run `bench/mkimages.py` first).
- `harness/build.sh`: standalone timing and bit-check harnesses for the A7 router, gate/up, down and GDN projection
  stages. Stop the server first.
- `LLAMA_A7_CHECK=1 GGML_CUDA_DISABLE_GRAPHS=1`: the upstream check mode. Each A7 op is compared with the unfused path,
  and the comparison is logged as `A7CHECK` lines.
- `GGML_A7_ROUTER_F16=2`: the host compares the F16 and F32 router top-10 sets. Run it with graphs disabled.

## Updating

To move onto a newer upstream patch or llama.cpp release:

1. Start a branch from the new base and import the new upstream patches as their own commits.
2. Cherry-pick this fork's commits (`git log v0.5.0..qwen38-v100`) on top.
3. Re-run `LLAMA_A7_CHECK=1` and `bench/` before switching the deployment.
