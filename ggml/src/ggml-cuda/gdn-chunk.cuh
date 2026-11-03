#pragma once

// [gdn-chunk] chunked (WY form) Gated DeltaNet prefill on Volta tensor cores, a port of the single-pass kernel of upstream
// PR #29353 (gated-delta-net-mma.cu, Ampere+ / BF16) to mma.sync.m8n8k4 with FP16 hi/lo operand splitting.
// Enabled by env GGML_CUDA_GDN_CHUNK=1 (GGML_CUDA_GDN_CHUNK_V=32|64: value columns per block, default 32).
// The kernel covers the first n_c tokens (a multiple of the chunk length 32, leaving >= K tokens); the caller runs the
// recurrent kernel over the rest from the state the chunk kernel wrote, so the K snapshot slots are written as before.

#include "common.cuh"

#include <cstdint>

struct ggml_cuda_gdn_chunk_args {
    const float *   q;                // [S_k, H_k, n_tokens], token stride sq2, head stride sq1 (k uses the same strides)
    const float *   k;
    const float *   v;                // [S_v, H, n_tokens], strides sv1 / sv2
    const float *   g;                // [1, H, n_tokens], strides sb1 / sb2 (log decay)
    const float *   beta;             // [1, H, n_tokens], strides sb1 / sb2
    const float *   state;            // input state, M[col][i] = S[i][col], read like the recurrent kernel (seq 0)
    const int32_t * state_rows;       // deferred gather: state row of seq 0 (device), or nullptr
    int64_t         state_row_stride; // in floats
    float *         dst;              // attention output of token 0 (dst[(t*H + h)*S_v + c])
    float *         state_out;        // state after n_tokens tokens, same layout as an input state of seq 0
    int64_t         H;                // value heads
    int64_t         H_k;              // q/k heads (q/k head of value head h is h % H_k)
    int64_t         n_tokens;         // tokens the chunk kernel processes (n_c)
    int64_t         sq1, sq2;
    int64_t         sv1, sv2;
    int64_t         sb1, sb2;
    float           scale;
};

// number of leading tokens the chunk kernel takes (multiple of 32, n_tokens - n_c >= K), 0 = disabled or not eligible
int64_t ggml_cuda_gdn_chunk_tokens(int device, bool kda, int64_t S_v, int64_t S_k, int64_t H, int64_t H_k,
                                   int64_t n_tokens, int64_t n_seqs, int K);

// runs the chunk kernel over a.n_tokens tokens (as returned by ggml_cuda_gdn_chunk_tokens); n_tokens_total / K for the log
void ggml_cuda_gdn_chunk_launch(int device, const ggml_cuda_gdn_chunk_args & a, int64_t n_tokens_total, int K,
                                cudaStream_t stream);
