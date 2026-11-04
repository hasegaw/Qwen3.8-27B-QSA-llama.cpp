# Qwen3.8-27B QSA MM2 for llama.cpp

Native sparse attention, MM2 multimodal indexers, MTP and live prefix-cache reuse.
Based on pentaCoxian/Qwen-V100-llama.cpp at `6209237df53569f07429f3f79fc5354de5fe704f`.

[GGUF and captured runtime](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-MM2-GGUF) | [BF16 source and original restoration](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-MM2-BF16)

The 16 full-attention layers use fixed K=2048, blocks of four, and the current block. The other 48 Gated DeltaNet layers retain their architecture. All KV is retained. Backbone, vision, projector and MTP weights are unchanged before quantization.

See [build, conversion and runtime instructions](qsa27b/RUNTIME.md). The captured runtime uses CUDA 12.9 on two sm70 32GB GPUs; other hardware needs a fresh build.

Text/image/video use native QSA with M-RoPE; MTP draft2 is enabled, lag/async disabled. One live slot reuses target KV, pooled indexer keys and MTP carry, not disk-persistent cache.

Controlled text throughput: 48.94 token/s at 0k added context, 41.07 at 261120 input tokens, mixing coding and free prose with three repeats. A matched dense throughput control is pending. Multimodal probes are limited through 64k and do not establish broad quality equivalence.

[Original fork README](README.upstream.md). MIT and third-party notices are retained.
