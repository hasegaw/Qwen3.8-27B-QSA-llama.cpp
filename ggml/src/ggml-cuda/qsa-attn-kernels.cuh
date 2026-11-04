// Device code of the QSA block-sparse attention (qsa-attn.cu): moved here unchanged by patch-mt-qsadp4a.py so that a standalone
// harness (qsa-bench.cu) can include the kernels; qsa_attn_kernel_dp4a is the [qsa-dp4a] variant.
#pragma once

#include "common.cuh"
#include <type_traits>

#define QSA_ATTN_D     256
#define QSA_ATTN_R     4

static constexpr int qsa_dpl = QSA_ATTN_D / WARP_SIZE; // dims per lane: 8

// raw Q8_0 data of one lane's 8 dims (dims 8*lane .. 8*lane+7) for the R cells of a block,
// loaded ahead of use so the next block's reads overlap this block's math
struct qsa_raw {
    uint32_t q[QSA_ATTN_R][2];
    float    d[QSA_ATTN_R];
};

static __device__ __forceinline__ void qsa_load_raw(const char * base, const size_t nb_cell, const int lane, qsa_raw & r) {
#pragma unroll
    for (int c = 0; c < QSA_ATTN_R; ++c) {
        const block_q8_0 * b = (const block_q8_0 *) (base + c*nb_cell) + lane/4;
        const uint16_t * q16 = (const uint16_t *) (b->qs + (lane % 4)*8);
        r.q[c][0] = (uint32_t) q16[0] | ((uint32_t) q16[1] << 16);
        r.q[c][1] = (uint32_t) q16[2] | ((uint32_t) q16[3] << 16);
        r.d[c]    = __half2float(b->d);
    }
}

static __device__ __forceinline__ void qsa_dequant(const qsa_raw & r, const int c, float (&x)[qsa_dpl]) {
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            x[4*w + b] = r.d[c] * (float) (int8_t) ((r.q[c][w] >> (8*b)) & 0xFF);
        }
    }
}

struct qsa_raw_f16 {
    uint4 words[QSA_ATTN_R];
};

static __device__ __forceinline__ void qsa_load_raw(const char * base, const size_t nb_cell, const int lane, qsa_raw_f16 & r) {
#pragma unroll
    for (int c = 0; c < QSA_ATTN_R; ++c) {
        r.words[c] = *(const uint4 *) (base + c*nb_cell + lane*qsa_dpl*sizeof(half));
    }
}

static __device__ __forceinline__ void qsa_dequant(const qsa_raw_f16 & r, const int c, float (&x)[qsa_dpl]) {
    const half * values = (const half *) &r.words[c];
#pragma unroll
    for (int d = 0; d < qsa_dpl; ++d) {
        x[d] = __half2float(values[d]);
    }
}

template <int HPW, int NGRP, bool CACHE_F16 = false>
__launch_bounds__(NGRP*WARP_SIZE)
static __global__ void qsa_attn_kernel(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const half * __restrict__ mask, const int32_t * __restrict__ ids, float * __restrict__ dst,
        float * __restrict__ part_acc, float2 * __restrict__ part_ml,
        const int n_head, const int gqa, const int n_wblk, const int n_split,
        const size_t nbq1, const size_t nbq2, const size_t nbk1, const size_t nbk2,
        const size_t nbv1, const size_t nbv2, const size_t nbm_tok, const size_t nbi_tok,
        const float scale) {
    static_assert(HPW*QSA_ATTN_R <= WARP_SIZE, "one lane per (head, cell) score");
    // one CUDA block per (split, kv head, token); its warps split the kv head's query heads, so they
    // read the same K/V rows (shared through L1)
    const int lane  = threadIdx.x;
    const int split = blockIdx.x;
    const int h_kv  = blockIdx.y;
    const int tok   = blockIdx.z;
    const int h0    = h_kv*gqa + threadIdx.y*HPW;   // first query head of this warp

    __shared__ float red[NGRP][WARP_SIZE][HPW*QSA_ATTN_R + 1];

    // this block's share of the query's block list
    const int per = (n_wblk + n_split - 1) / n_split;
    const int j0  = split*per;
    const int j1  = min(n_wblk, j0 + per);

    // the share's block ids and cell masks, loaded once for all warps of the block
    extern __shared__ char qsa_smem[];
    int32_t * s_ids  = (int32_t *) qsa_smem;
    float   * s_mask = (float *) (s_ids + per);
    const int32_t * id_t = (const int32_t *) ((const char *) ids + tok*nbi_tok);
    const half    * m_t  = (const half *) ((const char *) mask + tok*nbm_tok);
    for (int i = threadIdx.y*WARP_SIZE + lane; i < (j1 - j0)*QSA_ATTN_R; i += NGRP*WARP_SIZE) {
        s_mask[i] = __half2float(m_t[j0*QSA_ATTN_R + i]);
        if (i < j1 - j0) {
            s_ids[i] = id_t[j0 + i];
        }
    }
    __syncthreads();

    float qr[HPW][qsa_dpl];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const float * qh = (const float *) (q + tok*nbq2 + (h0 + h)*nbq1) + lane*qsa_dpl;
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            qr[h][d] = qh[d] * scale;
        }
    }

    float acc[HPW][qsa_dpl];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            acc[h][d] = 0.0f;
        }
    }
    // running max and sum of the head this lane scores (lanes h*R .. h*R + R-1 agree)
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int  my_c   = lane % QSA_ATTN_R;
    const bool scorer = lane < HPW*QSA_ATTN_R;

    // next block with a visible cell (padding blocks are off entirely)
    auto next_active = [&](int j) {
        while (j < j1) {
            const float * mj = s_mask + (j - j0)*QSA_ATTN_R;
            bool any = false;
#pragma unroll
            for (int c = 0; c < QSA_ATTN_R; ++c) {
                any |= mj[c] != -INFINITY;
            }
            if (any) {
                break;
            }
            ++j;
        }
        return j;
    };

    for (int j = next_active(j0); j < j1; j = next_active(j + 1)) {
        const float mv = s_mask[(j - j0)*QSA_ATTN_R + my_c];
        // K and V reads both issued before the math
        const int64_t cell0 = (int64_t) s_ids[j - j0] * QSA_ATTN_R;
        typename std::conditional<CACHE_F16, qsa_raw_f16, qsa_raw>::type kr, vr;
        qsa_load_raw(k + cell0*nbk2 + h_kv*nbk1, nbk2, lane, kr);
        qsa_load_raw(v + cell0*nbv2 + h_kv*nbv1, nbv2, lane, vr);

        // q.k partial sums over this lane's dims, for every (head, cell)
        float part[HPW][QSA_ATTN_R];
#pragma unroll
        for (int c = 0; c < QSA_ATTN_R; ++c) {
            float kx[qsa_dpl];
            qsa_dequant(kr, c, kx);
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                float sacc = 0.0f;
#pragma unroll
                for (int d = 0; d < qsa_dpl; ++d) {
                    sacc += qr[h][d] * kx[d];
                }
                part[h][c] = sacc;
            }
        }
        // transpose through shared memory: lane h*R + c sums (h, c) over all lanes
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
#pragma unroll
            for (int c = 0; c < QSA_ATTN_R; ++c) {
                red[threadIdx.y][lane][h*QSA_ATTN_R + c] = part[h][c];
            }
        }
        __syncwarp();
        float s = -INFINITY;
        if (scorer) {
            float sum = 0.0f;
#pragma unroll 8
            for (int i = 0; i < WARP_SIZE; ++i) {
                sum += red[threadIdx.y][i][lane];
            }
            s = mv == -INFINITY ? -INFINITY : sum + mv;
        }
        __syncwarp();

        // online softmax per head over the block's R cells (groups of R lanes)
        float bmax = s;
#pragma unroll
        for (int off = 1; off < QSA_ATTN_R; off <<= 1) {
            bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFF, bmax, off));
        }
        const float m_new = fmaxf(m_run, bmax);
        const float alpha = m_new == -INFINITY ? 1.0f : expf(m_run - m_new);
        const float p     = s == -INFINITY ? 0.0f : expf(s - m_new);
        float psum = p;
#pragma unroll
        for (int off = 1; off < QSA_ATTN_R; off <<= 1) {
            psum += __shfl_xor_sync(0xFFFFFFFF, psum, off);
        }
        l_run = l_run*alpha + psum;
        m_run = m_new;

        // acc = acc*alpha + sum_c p(h, c) * v_c
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float al = __shfl_sync(0xFFFFFFFF, alpha, h*QSA_ATTN_R);
#pragma unroll
            for (int d = 0; d < qsa_dpl; ++d) {
                acc[h][d] *= al;
            }
        }
#pragma unroll
        for (int c = 0; c < QSA_ATTN_R; ++c) {
            float vx[qsa_dpl];
            qsa_dequant(vr, c, vx);
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                const float pc = __shfl_sync(0xFFFFFFFF, p, h*QSA_ATTN_R + c);
#pragma unroll
                for (int d = 0; d < qsa_dpl; ++d) {
                    acc[h][d] += pc * vx[d];
                }
            }
        }
    }

    // per head (m, l) from its scoring lanes
    float mh[HPW], lh[HPW];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        mh[h] = __shfl_sync(0xFFFFFFFF, m_run, h*QSA_ATTN_R);
        lh[h] = __shfl_sync(0xFFFFFFFF, l_run, h*QSA_ATTN_R);
    }

    if (n_split == 1) {
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float inv = lh[h] > 0.0f ? 1.0f/lh[h] : 0.0f;
            float * o = dst + ((int64_t) tok*n_head + h0 + h)*QSA_ATTN_D + lane*qsa_dpl;
#pragma unroll
            for (int d = 0; d < qsa_dpl; ++d) {
                o[d] = acc[h][d]*inv;
            }
        }
        return;
    }

#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const int64_t row = ((int64_t) tok*n_head + h0 + h)*n_split + split;
        float * o = part_acc + row*QSA_ATTN_D + lane*qsa_dpl;
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            o[d] = acc[h][d];
        }
        if (lane == 0) {
            part_ml[row] = make_float2(mh[h], lh[h]);
        }
    }
}

// one block per (head, token): merge the splits' running softmax states
static __global__ void qsa_attn_combine(const float * __restrict__ part_acc, const float2 * __restrict__ part_ml,
        float * __restrict__ dst, const int n_split) {
    const int64_t row = (int64_t) blockIdx.y*gridDim.x + blockIdx.x;   // tok*n_head + head
    const float2 * ml = part_ml + row*n_split;
    float m = -INFINITY;
    for (int s = 0; s < n_split; ++s) {
        m = fmaxf(m, ml[s].x);
    }
    float l = 0.0f;
    float o = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float w = ml[s].x == -INFINITY ? 0.0f : expf(ml[s].x - m);
        l += w*ml[s].y;
        o += w*part_acc[(row*n_split + s)*QSA_ATTN_D + threadIdx.x];
    }
    dst[row*QSA_ATTN_D + threadIdx.x] = l > 0.0f ? o/l : 0.0f;
}

// [qsa-dp4a] qsa_attn_kernel with the q.k products done as int8 dot products (see patch-mt-qsadp4a.py): q is quantized once per
// (head, 32-dim q8 block); K is used as stored (int8 + scale), never dequantized. Everything else is as in qsa_attn_kernel.
template <int HPW, int NGRP>
__launch_bounds__(NGRP*WARP_SIZE)
static __global__ void qsa_attn_kernel_dp4a(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const half * __restrict__ mask, const int32_t * __restrict__ ids, float * __restrict__ dst,
        float * __restrict__ part_acc, float2 * __restrict__ part_ml,
        const int n_head, const int gqa, const int n_wblk, const int n_split,
        const size_t nbq1, const size_t nbq2, const size_t nbk1, const size_t nbk2,
        const size_t nbv1, const size_t nbv2, const size_t nbm_tok, const size_t nbi_tok,
        const float scale) {
    static_assert(HPW*QSA_ATTN_R <= WARP_SIZE, "one lane per (head, cell) score");
    // one CUDA block per (split, kv head, token); its warps split the kv head's query heads, so they
    // read the same K/V rows (shared through L1)
    const int lane  = threadIdx.x;
    const int split = blockIdx.x;
    const int h_kv  = blockIdx.y;
    const int tok   = blockIdx.z;
    const int h0    = h_kv*gqa + threadIdx.y*HPW;   // first query head of this warp

    __shared__ float red[NGRP][WARP_SIZE][HPW*QSA_ATTN_R + 1];

    // this block's share of the query's block list
    const int per = (n_wblk + n_split - 1) / n_split;
    const int j0  = split*per;
    const int j1  = min(n_wblk, j0 + per);

    // the share's block ids and cell masks, loaded once for all warps of the block
    extern __shared__ char qsa_smem[];
    int32_t * s_ids  = (int32_t *) qsa_smem;
    float   * s_mask = (float *) (s_ids + per);
    const int32_t * id_t = (const int32_t *) ((const char *) ids + tok*nbi_tok);
    const half    * m_t  = (const half *) ((const char *) mask + tok*nbm_tok);
    for (int i = threadIdx.y*WARP_SIZE + lane; i < (j1 - j0)*QSA_ATTN_R; i += NGRP*WARP_SIZE) {
        s_mask[i] = __half2float(m_t[j0*QSA_ATTN_R + i]);
        if (i < j1 - j0) {
            s_ids[i] = id_t[j0 + i];
        }
    }
    __syncthreads();

    float qr[HPW][qsa_dpl];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const float * qh = (const float *) (q + tok*nbq2 + (h0 + h)*nbq1) + lane*qsa_dpl;
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            qr[h][d] = qh[d] * scale;
        }
    }

    // [qsa-dp4a] q as int8 per (head, 32-dim block): this lane's 8 dims lie in q8 block lane/4, whose 4 lanes (4b .. 4b+3) share
    // the scale d_q = amax/127. Byte i of qq[h][w] is dim 4*w + i of the lane's 8 dims, as in kr.q[c][w] (qsa_load_raw).
    uint32_t qq[HPW][2];
    float    dq[HPW];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        float amax = 0.0f;
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            amax = fmaxf(amax, fabsf(qr[h][d]));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
        }
        dq[h] = amax / 127.0f;
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            uint32_t pk = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                int qi = dq[h] > 0.0f ? __float2int_rn(qr[h][4*w + b] / dq[h]) : 0;
                qi = max(-127, min(127, qi));
                pk |= ((uint32_t) qi & 0xFF) << (8*b);
            }
            qq[h][w] = pk;
        }
    }

    float acc[HPW][qsa_dpl];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            acc[h][d] = 0.0f;
        }
    }
    // running max and sum of the head this lane scores (lanes h*R .. h*R + R-1 agree)
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int  my_c   = lane % QSA_ATTN_R;
    const bool scorer = lane < HPW*QSA_ATTN_R;

    // next block with a visible cell (padding blocks are off entirely)
    auto next_active = [&](int j) {
        while (j < j1) {
            const float * mj = s_mask + (j - j0)*QSA_ATTN_R;
            bool any = false;
#pragma unroll
            for (int c = 0; c < QSA_ATTN_R; ++c) {
                any |= mj[c] != -INFINITY;
            }
            if (any) {
                break;
            }
            ++j;
        }
        return j;
    };

    for (int j = next_active(j0); j < j1; j = next_active(j + 1)) {
        const float mv = s_mask[(j - j0)*QSA_ATTN_R + my_c];
        // K and V reads both issued before the math
        const int64_t cell0 = (int64_t) s_ids[j - j0] * QSA_ATTN_R;
        qsa_raw kr, vr;
        qsa_load_raw(k + cell0*nbk2 + h_kv*nbk1, nbk2, lane, kr);
        qsa_load_raw(v + cell0*nbv2 + h_kv*nbv1, nbv2, lane, vr);

        // [qsa-dp4a] q.k partial sums over this lane's dims, for every (head, cell): 8 int8 products per __dp4a pair, no K dequant
        float part[HPW][QSA_ATTN_R];
#pragma unroll
        for (int c = 0; c < QSA_ATTN_R; ++c) {
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                const int dot = __dp4a((int) kr.q[c][0], (int) qq[h][0], __dp4a((int) kr.q[c][1], (int) qq[h][1], 0));
                part[h][c] = (float) dot * (kr.d[c] * dq[h]);
            }
        }
        // transpose through shared memory: lane h*R + c sums (h, c) over all lanes
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
#pragma unroll
            for (int c = 0; c < QSA_ATTN_R; ++c) {
                red[threadIdx.y][lane][h*QSA_ATTN_R + c] = part[h][c];
            }
        }
        __syncwarp();
        float s = -INFINITY;
        if (scorer) {
            float sum = 0.0f;
#pragma unroll 8
            for (int i = 0; i < WARP_SIZE; ++i) {
                sum += red[threadIdx.y][i][lane];
            }
            s = mv == -INFINITY ? -INFINITY : sum + mv;
        }
        __syncwarp();

        // online softmax per head over the block's R cells (groups of R lanes)
        float bmax = s;
#pragma unroll
        for (int off = 1; off < QSA_ATTN_R; off <<= 1) {
            bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFF, bmax, off));
        }
        const float m_new = fmaxf(m_run, bmax);
        const float alpha = m_new == -INFINITY ? 1.0f : expf(m_run - m_new);
        const float p     = s == -INFINITY ? 0.0f : expf(s - m_new);
        float psum = p;
#pragma unroll
        for (int off = 1; off < QSA_ATTN_R; off <<= 1) {
            psum += __shfl_xor_sync(0xFFFFFFFF, psum, off);
        }
        l_run = l_run*alpha + psum;
        m_run = m_new;

        // acc = acc*alpha + sum_c p(h, c) * v_c
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float al = __shfl_sync(0xFFFFFFFF, alpha, h*QSA_ATTN_R);
#pragma unroll
            for (int d = 0; d < qsa_dpl; ++d) {
                acc[h][d] *= al;
            }
        }
#pragma unroll
        for (int c = 0; c < QSA_ATTN_R; ++c) {
            float vx[qsa_dpl];
            qsa_dequant(vr, c, vx);
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                const float pc = __shfl_sync(0xFFFFFFFF, p, h*QSA_ATTN_R + c);
#pragma unroll
                for (int d = 0; d < qsa_dpl; ++d) {
                    acc[h][d] += pc * vx[d];
                }
            }
        }
    }

    // per head (m, l) from its scoring lanes
    float mh[HPW], lh[HPW];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        mh[h] = __shfl_sync(0xFFFFFFFF, m_run, h*QSA_ATTN_R);
        lh[h] = __shfl_sync(0xFFFFFFFF, l_run, h*QSA_ATTN_R);
    }

    if (n_split == 1) {
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float inv = lh[h] > 0.0f ? 1.0f/lh[h] : 0.0f;
            float * o = dst + ((int64_t) tok*n_head + h0 + h)*QSA_ATTN_D + lane*qsa_dpl;
#pragma unroll
            for (int d = 0; d < qsa_dpl; ++d) {
                o[d] = acc[h][d]*inv;
            }
        }
        return;
    }

#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const int64_t row = ((int64_t) tok*n_head + h0 + h)*n_split + split;
        float * o = part_acc + row*QSA_ATTN_D + lane*qsa_dpl;
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            o[d] = acc[h][d];
        }
        if (lane == 0) {
            part_ml[row] = make_float2(mh[h], lh[h]);
        }
    }
}
