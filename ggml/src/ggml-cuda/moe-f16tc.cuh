#pragma once
// [moe-f16tc] v4 routed-expert MUL_MAT_ID on Volta FP16 tensor cores, env GGML_CUDA_MOE_F16TC=N, GGML_CUDA_MOE_F16TC_LAYOUT=L (see patch-mt-moef16tc4.py)
#include "common.cuh"

bool ggml_cuda_moe_f16tc_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_id_f16tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// [moe-fuse] env GGML_CUDA_MOE_F16TC_FUSE=1 (read once): gate + up + swiglu + down of a routed-expert block on one shared preparation
// (see patch-mt-moefuse.py); the caller (ggml_cuda_try_fuse) checks the graph pattern and the [moe-f16tc] conditions of all three
bool ggml_cuda_moe_f16tc_fuse_enabled();
void ggml_cuda_moe_f16tc_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu, ggml_tensor * down);
