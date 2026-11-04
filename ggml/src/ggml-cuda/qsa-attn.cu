// Block-sparse attention over a Q8_0 K/V cache (QSA): every query attends to its own list of
// ratio-cell blocks, read straight from the cache. No gathered K/V copy and no cache-sized mask,
// so the cost follows the selected budget instead of the context length; the same kernel serves
// decode (a few queries, split over many blocks) and prompt batches.
//
// Expressed as a GGML_OP_FLASH_ATTN_EXT node with src[5] = block ids (see ggml_cuda_qsa_attn_is_sparse):
//   src[0] q     F32  [D, n_head, n_tok]
//   src[1] k     Q8_0 [D, n_head_kv, n_cells]   (a cache view; cell = block*ratio + c)
//   src[2] v     Q8_0 [D, n_head_kv, n_cells]
//   src[3] mask  F16  [n_wblk*ratio, 1, 1, n_tok]  0 or -inf per selected cell
//   src[5] ids   I32  [n_wblk, n_tok]
//   dst          F32  [D, n_head, n_tok]
//   op_params: [0] scale, [6] ratio

#include "qsa-attn.cuh"
#include "common.cuh"

#include "qsa-attn-kernels.cuh" // [qsa-dp4a] the device code lives there
#include <atomic> // [qsa-hpw6] one bounded selection diagnostic

bool ggml_cuda_qsa_attn_is_sparse(const ggml_tensor * dst) {
    return dst->op == GGML_OP_FLASH_ATTN_EXT && dst->src[5] != nullptr && dst->src[5]->type == GGML_TYPE_I32;
}

bool ggml_cuda_qsa_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    const ggml_tensor * ids = dst->src[5];
    if (q->type != GGML_TYPE_F32 || (k->type != GGML_TYPE_Q8_0 && k->type != GGML_TYPE_F16) || v->type != k->type || !m || m->type != GGML_TYPE_F16) {
        return false;
    }
    if (q->ne[0] != QSA_ATTN_D || k->ne[0] != QSA_ATTN_D || v->ne[0] != QSA_ATTN_D || ggml_get_op_params_i32(dst, 6) != QSA_ATTN_R) {
        return false;
    }
    const int gqa = q->ne[1] / k->ne[1];
    // the head groupings instantiated below: 24/12/8/6/4/3/2/1 query heads per KV head (6: 4 GPUs on the KV-duplicated model)
    if (q->ne[1] % k->ne[1] != 0 || !(gqa == 24 || gqa == 16 || gqa == 12 || gqa == 8 || gqa == 6 || gqa == 4 || gqa == 3 ||
            gqa == 2 || gqa == 1)) {
        return false;
    }
    return q->nb[0] == sizeof(float) && ggml_is_contiguous(m) && ggml_is_contiguous(ids) &&
           ids->ne[1] == q->ne[2] && m->ne[0] == ids->ne[0]*QSA_ATTN_R && m->ne[3] == q->ne[2] &&
           dst->ne[0] == QSA_ATTN_D && dst->ne[1] == q->ne[1] && dst->ne[2] == q->ne[2] && ggml_is_contiguous(dst);
}

void ggml_cuda_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * ids = dst->src[5];
    GGML_ASSERT(ggml_cuda_qsa_attn_supported(dst));

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_head = q->ne[1];
    const int n_tok  = q->ne[2];
    const int gqa    = n_head / k->ne[1];
    const int n_wblk = ids->ne[0];

    // 3 heads per warp for gqa 12 (4 warps per block); other group sizes pick a divisor
    const int hpw   = gqa % 3 == 0 && gqa/3 <= 8 ? 3 : gqa % 4 == 0 && gqa/4 <= 8 ? 4 : gqa % 2 == 0 && gqa/2 <= 8 ? 2 : 1;
    const int n_grp = gqa / hpw;
    GGML_ASSERT(n_grp <= 8);

    // enough blocks to fill the GPU: small batches split each query's block list
    const int n_sm    = ggml_cuda_info().devices[ctx.device].nsm;
    const int blocks0 = n_tok * (int) k->ne[1];
    int n_split = std::max(1, std::min(64, (n_sm*8 + blocks0 - 1) / blocks0));
    // An exhaustive list is used by dense MTP attention. Bound the work and
    // shared memory per CTA there; the usual ~512-block QSA lists are unchanged.
    if (k->type == GGML_TYPE_F16 && n_wblk > 1024) {
        n_split = std::max(n_split, std::min(512, (n_wblk + 127)/128));
    }
    n_split = std::min(n_split, n_wblk);

    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool());
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool());
    if (n_split > 1) {
        part_acc.alloc((size_t) n_tok*n_head*n_split*QSA_ATTN_D);
        part_ml.alloc((size_t) n_tok*n_head*n_split);
    }

    const dim3 grid(n_split, k->ne[1], n_tok);
    const dim3 block(WARP_SIZE, n_grp);
    cudaStream_t stream = ctx.stream();

    const int    per  = (n_wblk + n_split - 1) / n_split;
    const size_t smem = (size_t) per*(sizeof(int32_t) + QSA_ATTN_R*sizeof(float));
    // [qsa-dp4a] env GGML_CUDA_QSA_DP4A=1: q.k with int8 dot products (qsa_attn_kernel_dp4a), same launch shape
    static const bool use_dp4a = [] {
        const char * e = getenv("GGML_CUDA_QSA_DP4A");
        const bool on = e != nullptr && atoi(e) != 0;
        if (on) {
            GGML_LOG_WARN("[qsa-dp4a] qsa attention: q.k with int8 dot products (GGML_CUDA_QSA_DP4A=1)\n");
        }
        return on;
    }();
    // [qsa-hpw6] opt-in; never affects non-dp4a or other GQA shapes.
    static const bool use_hpw6 = [] {
        const char * e = getenv("GGML_CUDA_QSA_HPW6");
        return e != nullptr && atoi(e) != 0;
    }();
#define QSA_ARGS (const char *) q->data, (const char *) k->data, (const char *) v->data, (const half *) m->data, \
        (const int32_t *) ids->data, (float *) dst->data, part_acc.ptr, part_ml.ptr, \
        n_head, gqa, n_wblk, n_split, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], \
        m->nb[3], ids->nb[1], scale
#define QSA_LAUNCH(HPW, NGRP) \
    if (k->type == GGML_TYPE_F16) { qsa_attn_kernel<HPW, NGRP, true><<<grid, block, smem, stream>>>(QSA_ARGS); } \
    else if (use_dp4a) { qsa_attn_kernel_dp4a<HPW, NGRP><<<grid, block, smem, stream>>>(QSA_ARGS); } \
    else          { qsa_attn_kernel<HPW, NGRP><<<grid, block, smem, stream>>>(QSA_ARGS); }
    if (k->type == GGML_TYPE_Q8_0 && use_hpw6 && use_dp4a && gqa == 12) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_WARN("[qsa-hpw6] selected dp4a<6,2>: tokens=%d heads=%d kv=%lld splits=%d threads=64 smem=%zu\n",
                n_tok, n_head, (long long) k->ne[1], n_split, smem);
        }
        qsa_attn_kernel_dp4a<6, 2><<<grid, dim3(WARP_SIZE, 2), smem, stream>>>(QSA_ARGS);
    } else if (hpw == 3 && n_grp == 8) { QSA_LAUNCH(3, 8); }
    else if (hpw == 3 && n_grp == 4) { QSA_LAUNCH(3, 4); }
    else if (hpw == 3 && n_grp == 2) { QSA_LAUNCH(3, 2); }
    else if (hpw == 3 && n_grp == 1) { QSA_LAUNCH(3, 1); }
    else if (hpw == 4 && n_grp == 1) { QSA_LAUNCH(4, 1); }
    else if (hpw == 4 && n_grp == 2) { QSA_LAUNCH(4, 2); }
    else if (hpw == 4 && n_grp == 4) { QSA_LAUNCH(4, 4); }
    else if (hpw == 2 && n_grp == 4) { QSA_LAUNCH(2, 4); }
    else if (hpw == 2 && n_grp == 1) { QSA_LAUNCH(2, 1); }
    else if (hpw == 1 && n_grp == 1) { QSA_LAUNCH(1, 1); }
    else { GGML_ABORT("qsa attention: unsupported head grouping %d x %d", hpw, n_grp); }
#undef QSA_LAUNCH
#undef QSA_ARGS
    CUDA_CHECK(cudaGetLastError());

    if (n_split > 1) {
        qsa_attn_combine<<<dim3(n_head, n_tok), QSA_ATTN_D, 0, stream>>>(part_acc.ptr, part_ml.ptr, (float *) dst->data, n_split);
        CUDA_CHECK(cudaGetLastError());
    }
}
