#pragma once

#include "common.cuh"

// GGML_OP_A7_GDN_FFN (A7 P1): the second half of a GDN layer (hc_mix(ffn) -> router -> MoE -> shared expert -> hc_combine(ffn)) of
// Qwen3.8-Flash-Next for a verify window of T = 4 tokens, as a chain of stage kernels (a7_ffn_*) ported from research gate0.cu.

void ggml_cuda_op_a7_gdn_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_a7_gdn_ffn_supported(int device, const ggml_tensor * op);

// GGML_OP_A7_GDN_ATTN (A7 P2): the first half of a GDN layer (hc_mix(attn) -> qkv / z / alpha / beta -> conv -> l2 norm -> delta rule -> gated norm ->
// ssm_out -> hc_combine(attn)), stage kernels a7_attn_*; reads and writes the recurrent state buffers (r_l / s_l) directly.

void ggml_cuda_op_a7_gdn_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_a7_gdn_attn_supported(int device, const ggml_tensor * op);
