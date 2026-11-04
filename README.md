# Qwen3.8-27B QSA for llama.cpp

[English](README.md) | [日本語](README.ja.md)

This fork adds QSA, multimodal indexer support, MTP, and live prefix-cache reuse for Qwen3.8-27B. The engine targets Qwen3.8-27B, not Qwen3.8-FlashNext.

It is based on [pentaCoxian/Qwen-V100-llama.cpp](https://github.com/pentaCoxian/Qwen-V100-llama.cpp) at `6209237df53569f07429f3f79fc5354de5fe704f`.

[GGUF and captured runtime](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF) | [BF16 source and original-model restoration](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16)

## Implementation

The 16 full-attention layers use fixed K=2,048, four-token blocks, and the current block. The other 48 Gated DeltaNet layers retain their architecture. All KV entries are retained, so QSA does not compress the KV cache. Backbone, vision, projector, and MTP weights are unchanged before quantization.

Text, images, and video use native QSA with M-RoPE. MTP provides up to two draft tokens, with lag/async disabled. One live slot reuses target KV, pooled indexer keys, and MTP carry. This cache is not stored on disk and does not survive a server restart.

## Build and run

See the [build, conversion, and runtime instructions](qsa27b/RUNTIME.md). The captured runtime uses CUDA 12.9 on two sm70 32GB GPUs. Other hardware needs a source build and appropriate memory sizing.

The `mm2` strings in artifact names, checkpoint paths, and branch names are historical experiment identifiers retained for compatibility and reproducibility. They are not part of the QSA method name.

## Measured throughput and limits

A QSA-only text benchmark measured 48.94 token/s at 0k added context and 41.07 token/s at 261,120 input tokens, a 16.08% decline. The benchmark mixed two coding tasks and two free-prose tasks, with three repeats per task at each context depth. The 0k point still contains a 490-563-token task prompt.

A matched dense-attention throughput control has not been completed. Multimodal probes are limited to checks through 64k and do not establish broad quality equivalence.

## License and upstream

See the [original fork README](README.upstream.md). MIT and third-party notices are retained.
