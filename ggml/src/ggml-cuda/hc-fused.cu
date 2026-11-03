#include "hc-fused.cuh"

#include <cstdarg> // [hc-par] hc_par_log
#include <map>     // [hc-par] completion-counter slot per (device, stream)

#define HC_WARPS   8
#define HC_THREADS (HC_WARPS*WARP_SIZE)
#define HC_MAX_HC  8
#define HC_MAX_SMEM (96*1024)

// 4 int8 values of a Q8_0 block starting at qs + 4*i (the quants are only 2-byte aligned)
static __device__ __forceinline__ int hc_load_q8_0_int(const block_q8_0 & b, const int i) {
    const uint16_t * q = (const uint16_t *) (b.qs + 4*i);
    return q[0] | (q[1] << 16);
}

// symmetric int8 quantization of 32 values held one per lane: returns the scale, q gets the lane's quant
static __device__ __forceinline__ float hc_quant_warp(const float v, int8_t & q) {
    const float amax = warp_reduce_max(fabsf(v));
    const float d = amax / 127.0f;
    q = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
    return d;
}

// hc_mix_down, step 1: quantize x * w_norm to int8 in blocks of 32 (one warp per block) and store each block's sum of
// squares of x, so step 2 can apply 1/rms per stream afterwards
template <int T>
static __global__ void __launch_bounds__(HC_THREADS) hc_quant_x(
        const float * __restrict__ x, const float * __restrict__ w_norm, int8_t * __restrict__ xq, float * __restrict__ xd,
        float * __restrict__ ssq, const float * __restrict__ w_inject, const size_t ld_inject, float * __restrict__ inj,
        const int D, const int hc) {
    const int nb   = D / QK8_0;
    const int qb   = blockIdx.x*HC_WARPS + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (qb >= T*nb) {
        return;
    }
    const int t = qb / nb;
    const int e = (qb % nb)*QK8_0 + lane;
    const float xv = x[(size_t) t*D + e];
    const float xw = xv * w_norm[e];
    int8_t q;
    const float d = hc_quant_warp(xw, q);
    xq[(size_t) t*D + e] = q;
    const float sq = warp_reduce_sum(xv*xv);
    if (lane == 0) {
        xd[qb]  = d;
        ssq[qb] = sq;
    }
    // partial inject dots of this block, 1/rms applied in step 2
    if (w_inject != nullptr) {
        for (int r = 0; r < hc; ++r) {
            const float p = warp_reduce_sum(w_inject[(size_t) r*ld_inject + e] * xw);
            if (lane == 0) {
                inj[(size_t) qb*hc + r] = p;
            }
        }
    }
}

// hc_mix_down, step 2: rows [0, lr): silu(w_down . xn / hc), rows [lr, lr+hc): w_inject . xn, rows [lr+hc, lr+2hc):
// 1/rms per stream. HC_DOT_RPB rows per block, each row split over HC_WARPS/HC_DOT_RPB warps along K; dp4a against
// the Q8_0 weights, each quant block scaled by its stream's 1/rms.
#define HC_DOT_RPB 2
template <int T>
static __global__ void __launch_bounds__(HC_THREADS) hc_mix_down_dot(
        const int8_t * __restrict__ xq, const float * __restrict__ xd, const float * __restrict__ ssq,
        const char * __restrict__ w_down, const size_t nb_down, const float * __restrict__ inj,
        float * __restrict__ dst, const int n_embd, const int hc, const int lr, const float eps) {
    constexpr int wpr = HC_WARPS / HC_DOT_RPB; // warps per row
    const int D   = n_embd*hc;
    const int nb  = D / QK8_0;
    const int nbs = n_embd / QK8_0; // quant blocks per stream
    __shared__ float rr[HC_MAX_HC][T];
    __shared__ float part[HC_WARPS][T];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    for (int st = warp; st < hc*T; st += HC_WARPS) {
        const int s = st % hc;
        const int t = st / hc;
        float sum = 0.0f;
        for (int b = lane; b < nbs; b += WARP_SIZE) {
            sum += ssq[t*nb + s*nbs + b];
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            rr[s][t] = rsqrtf(sum / n_embd + eps);
        }
    }
    __syncthreads();

    const int row    = blockIdx.x*HC_DOT_RPB + warp / wpr;
    const int kw     = warp % wpr; // this warp's share of K
    const int n_rows = lr + hc;
    const int ne0    = lr + 2*hc;

    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = 0.0f;
    }
    if (row < lr) {
        const block_q8_0 * wrow = (const block_q8_0 *) (w_down + (size_t) row*nb_down);
        for (int b = kw*WARP_SIZE + lane; b < nb; b += wpr*WARP_SIZE) {
            const block_q8_0 & wb = wrow[b];
            int w[QK8_0/4];
#pragma unroll
            for (int i = 0; i < QK8_0/4; ++i) {
                w[i] = hc_load_q8_0_int(wb, i);
            }
            const float dw = __half2float(wb.d);
            const int   s  = b / nbs;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int4 * a4 = (const int4 *) (xq + (size_t) t*D + b*QK8_0);
                const int4 a0 = a4[0];
                const int4 a1 = a4[1];
                int sumi = 0;
                sumi = ggml_cuda_dp4a(w[0], a0.x, sumi);
                sumi = ggml_cuda_dp4a(w[1], a0.y, sumi);
                sumi = ggml_cuda_dp4a(w[2], a0.z, sumi);
                sumi = ggml_cuda_dp4a(w[3], a0.w, sumi);
                sumi = ggml_cuda_dp4a(w[4], a1.x, sumi);
                sumi = ggml_cuda_dp4a(w[5], a1.y, sumi);
                sumi = ggml_cuda_dp4a(w[6], a1.z, sumi);
                sumi = ggml_cuda_dp4a(w[7], a1.w, sumi);
                acc[t] += dw * xd[t*nb + b] * rr[s][t] * (float) sumi;
            }
        }
    } else if (row < n_rows && inj != nullptr) {
        const int r = row - lr;
        for (int b = kw*WARP_SIZE + lane; b < nb; b += wpr*WARP_SIZE) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[t] += inj[((size_t) t*nb + b)*hc + r] * rr[b / nbs][t];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            part[warp][t] = v;
        }
    }
    __syncthreads();
    if (kw == 0 && lane < T && row < n_rows) {
        const int t = lane;
        float v = 0.0f;
        for (int k = 0; k < wpr; ++k) {
            v += part[warp + k][t];
        }
        float out = v;
        if (row < lr) {
            const float z = v / hc;
            out = z / (1.0f + expf(-z));
        } else if (inj == nullptr) {
            out = 0.0f;
        }
        dst[(size_t) t*ne0 + row] = out;
    }
    if (blockIdx.x == 0 && threadIdx.x < T) {
        for (int s = 0; s < hc; ++s) {
            dst[(size_t) threadIdx.x*ne0 + n_rows + s] = rr[s][threadIdx.x];
        }
    }
}

// One warp per channel i, lanes grouped by stream: the 32/hc lanes of stream s compute gate row s*n_embd + i by dp4a
// against lo quantized in shared memory, then the gated mean over the streams.
#define HC_UP_WARPS 32
template <int T>
static __global__ void __launch_bounds__(HC_UP_WARPS*WARP_SIZE) hc_mix_up_q(
        const float * __restrict__ x, const float * __restrict__ w_norm, const char * __restrict__ w_up, const size_t nb_up,
        const float * __restrict__ down, float * __restrict__ dst, const int n_embd, const int hc, const int lr,
        block_q8_1 * __restrict__ yq) {
    extern __shared__ int hc_smem[];
    const int nb = lr / QK8_0;
    int8_t * lq = (int8_t *) hc_smem;             // [T][lr]
    float  * ld = (float *) (lq + (size_t) T*lr); // [T][nb]
    __shared__ float rr[HC_MAX_HC][T];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int ne_down = lr + 2*hc;

    for (int qb = warp; qb < T*nb; qb += HC_UP_WARPS) {
        const int t = qb / nb;
        const int j = (qb % nb)*QK8_0 + lane;
        int8_t q;
        const float d = hc_quant_warp(down[(size_t) t*ne_down + j], q);
        lq[t*lr + j] = q;
        if (lane == 0) {
            ld[t*nb + qb % nb] = d;
        }
    }
    if (threadIdx.x < T) {
        for (int s = 0; s < hc; ++s) {
            rr[s][threadIdx.x] = down[(size_t) threadIdx.x*ne_down + lr + hc + s];
        }
    }
    __syncthreads();

    const int i = blockIdx.x*HC_UP_WARPS + warp;
    const bool active = i < n_embd;
    __shared__ float mq[T][HC_UP_WARPS];

    const int lps = WARP_SIZE / hc; // lanes per stream
    const int s   = lane / lps;
    const int j0  = lane % lps;
    float g[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        g[t] = 0.0f;
    }
    if (s < hc && active) {
        const block_q8_0 * wrow = (const block_q8_0 *) (w_up + (size_t) (s*n_embd + i)*nb_up);
        for (int b = j0; b < nb; b += lps) {
            const block_q8_0 & wb = wrow[b];
            int w[QK8_0/4];
#pragma unroll
            for (int k = 0; k < QK8_0/4; ++k) {
                w[k] = hc_load_q8_0_int(wb, k);
            }
            const float dw = __half2float(wb.d);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int * a = (const int *) (lq + t*lr + b*QK8_0);
                int sumi = 0;
#pragma unroll
                for (int k = 0; k < QK8_0/4; ++k) {
                    sumi = ggml_cuda_dp4a(w[k], a[k], sumi);
                }
                g[t] += dw * ld[t*nb + b] * (float) sumi;
            }
        }
    }
    // sum within each group of lps lanes: every lane of the group ends with its stream's gate
#pragma unroll
    for (int t = 0; t < T; ++t) {
        for (int off = lps/2; off > 0; off >>= 1) {
            g[t] += __shfl_xor_sync(0xFFFFFFFF, g[t], off, WARP_SIZE);
        }
    }
    float mixed[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float contrib = 0.0f;
        if (s < hc && j0 == 0 && active) {
            const float xn = x[((size_t) t*hc + s)*n_embd + i] * rr[s][t] * w_norm[s*n_embd + i];
            contrib = xn / (1.0f + expf(-g[t]));
        }
        mixed[t] = warp_reduce_sum(contrib);
    }
    if (lane == 0 && active) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float v = mixed[t] / hc;
            dst[(size_t) t*n_embd + i] = v;
            mq[t][warp] = v;
        }
    }
    // the GEMV that reads mixed next finds it quantized (q8_1 cache): warp t quantizes token t's 32 channels,
    //     exactly as quantize_q8_1 does. Needs n_embd % 512 == 0: no padding blocks.
    if (yq != nullptr) {
        __syncthreads();
        if (warp < T) {
            const int t  = warp;
            const int ch = blockIdx.x*HC_UP_WARPS + lane;
            const float xi = ch < n_embd ? mq[t][lane] : 0.0f;
            float amax = fabsf(xi);
            float sum  = xi;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
            const int64_t i_cont = (int64_t) t*n_embd + ch;
            yq[i_cont / QK8_1].qs[i_cont % QK8_1] = q;
            if (lane == 0) {
                yq[i_cont / QK8_1].ds = make_half2(d, sum);
            }
        }
    }
}

static __global__ void hc_combine_f32(
        const float * __restrict__ x, const float * __restrict__ b, const float * __restrict__ down, float * __restrict__ dst,
        const int n_embd, const int hc, const int T, const int ne_down, const int lr) {
    const int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= (int64_t) n_embd*hc*T) {
        return;
    }
    const int i = k % n_embd;
    const int s = (k / n_embd) % hc;
    const int t = k / ((int64_t) n_embd*hc);
    const float inj = down[(size_t) t*ne_down + lr + s];
    const float w   = 2.0f / (1.0f + expf(-inj / hc));
    dst[k] = x[k] + b[(size_t) t*n_embd + i]*w;
}

// hc combine (mode 2) and step 1 of the next mix-down (hc_quant_x) in one pass: both split the [n_embd, hc, T] state
// into the same 32-element blocks, so each warp combines its block, stores it and quantizes it from registers
template <int T>
static __global__ void __launch_bounds__(HC_THREADS) hc_combine_quant_x(
        const float * __restrict__ cx, const float * __restrict__ cb, const float * __restrict__ cdown, float * __restrict__ x,
        const int ne_down, const int clr,
        const float * __restrict__ w_norm, int8_t * __restrict__ xq, float * __restrict__ xd,
        float * __restrict__ ssq, const float * __restrict__ w_inject, const size_t ld_inject, float * __restrict__ inj,
        const int n_embd, const int hc) {
    const int D    = n_embd*hc;
    const int nb   = D / QK8_0;
    const int qb   = blockIdx.x*HC_WARPS + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (qb >= T*nb) {
        return;
    }
    const int t = qb / nb;
    const int e = (qb % nb)*QK8_0 + lane;

    // as hc_combine_f32
    const int64_t k = (int64_t) t*D + e;
    {
        const int i = k % n_embd;
        const int s = (k / n_embd) % hc;
        const float inj_c = cdown[(size_t) t*ne_down + clr + s];
        const float w     = 2.0f / (1.0f + expf(-inj_c / hc));
        x[k] = cx[k] + cb[(size_t) t*n_embd + i]*w;
    }
    const float xv = x[k];

    // as hc_quant_x
    const float xw = xv * w_norm[e];
    int8_t q;
    const float d = hc_quant_warp(xw, q);
    xq[(size_t) t*D + e] = q;
    const float sq = warp_reduce_sum(xv*xv);
    if (lane == 0) {
        xd[qb]  = d;
        ssq[qb] = sq;
    }
    if (w_inject != nullptr) {
        for (int r = 0; r < hc; ++r) {
            const float p = warp_reduce_sum(w_inject[(size_t) r*ld_inject + e] * xw);
            if (lane == 0) {
                inj[(size_t) qb*hc + r] = p;
            }
        }
    }
}

// v2 kernels for hc == 4 (GGML_CUDA_NO_HC_V2=1 selects the kernels above). The v1 kernels were latency-bound: few
// blocks, 17 two-byte loads per 34-byte Q8_0 block, and per-block prologues ahead of the first weight load. Here each
// warp issues all of its weight loads at kernel start, then stages and consumes them one row (or row pair) at a time,
// so the dot products of the first rows overlap the loads still in flight. A lane dots whole Q8_0 blocks, read from
// shared memory with aligned 32-bit loads.

// per-block phase timestamps for the standalone benchmark (-DHC_TS); compiled out otherwise
#ifdef HC_TS
__device__ unsigned long long hc_ts[512][8];
#define HC_STAMP(i) do { if (threadIdx.x == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); hc_ts[blockIdx.x][i] = t_; } } while (0)
#else
#define HC_STAMP(i) do {} while (0)
#endif

// Q8_0 block at an even byte offset of a 4-byte-aligned shared buffer (at least 4 readable bytes past the block)
static __device__ __forceinline__ float hc_smem_q8_0(const int * base, const int off, int w[QK8_0/4]) {
    const int dword = base[off >> 2];
    const float d = __half2float(__ushort_as_half((unsigned short) (dword >> ((off & 2)*8))));
    const int q0 = off + (int) sizeof(half);
    const int a  = q0 >> 2;
    const int sh = (q0 & 2)*8;
#pragma unroll
    for (int k = 0; k < QK8_0/4; ++k) {
        w[k] = __funnelshift_r(base[a + k], base[a + k + 1], sh);
    }
    return d;
}

// sums N (a power of 2, 2..32) values over the warp in N - 1 + log2(32/N) shuffles instead of N*5: at each step a lane
// keeps half of its values and adds the partner's copy of that half. Lane l ends with the total of value
// l / (32/N) (each total is held by 32/N lanes).
static constexpr __host__ __device__ int hc_np2(const int x) {
    int p = 1;
    while (p < x) {
        p *= 2;
    }
    return p;
}

// one step of hc_warp_reduce_multi: n values left, partner at lane ^ off; recursion keeps every index static
template <int n, int off>
static __device__ __forceinline__ void hc_reduce_step(float * v, const bool hi) {
#pragma unroll
    for (int i = 0; i < n; ++i) {
        const float send = hi ? v[i] : v[i + n];
        const float keep = hi ? v[i + n] : v[i];
        v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, off, WARP_SIZE);
    }
}
template <int n, int off>
static __device__ __forceinline__ void hc_reduce_steps(float * v, const int lane) {
    hc_reduce_step<n, off>(v, lane & off);
    if constexpr (n > 1) {
        hc_reduce_steps<n/2, off/2>(v, lane);
    }
}

template <int N>
static __device__ __forceinline__ float hc_warp_reduce_multi(float v[N]) {
    const int lane = threadIdx.x % WARP_SIZE;
    if constexpr (N > 1) {
        hc_reduce_steps<N/2, WARP_SIZE/2>(v, lane);
    }
    float r = v[0];
#pragma unroll
    for (int off = WARP_SIZE/(2*N); off > 0; off /= 2) {
        r += __shfl_xor_sync(0xFFFFFFFF, r, off, WARP_SIZE);
    }
    return r;
}

static __device__ __forceinline__ int hc_dot_q8_0(const int w[QK8_0/4], const int4 a0, const int4 a1) {
    int sumi = 0;
    sumi = ggml_cuda_dp4a(w[0], a0.x, sumi);
    sumi = ggml_cuda_dp4a(w[1], a0.y, sumi);
    sumi = ggml_cuda_dp4a(w[2], a0.z, sumi);
    sumi = ggml_cuda_dp4a(w[3], a0.w, sumi);
    sumi = ggml_cuda_dp4a(w[4], a1.x, sumi);
    sumi = ggml_cuda_dp4a(w[5], a1.y, sumi);
    sumi = ggml_cuda_dp4a(w[6], a1.z, sumi);
    sumi = ggml_cuda_dp4a(w[7], a1.w, sumi);
    return sumi;
}

// hc_mix_down step 2, v2: HC_DN_R rows per block over the whole of K; warp w takes quant blocks [32w, 32w+32) of
// every row (one per lane, activations loaded once), NW = D/1024 warps. Block n_row_blocks writes the inject rows and
// the 1/rms rows.
#define HC_DN_R      4
#define HC_DN_CHUNK  (32*(int) sizeof(block_q8_0)/16) // int4 per row and warp (68)
#define HC_DN_STRIDE (HC_DN_CHUNK*4 + 4)              // words per warp staging area (16-byte aligned, 16 B slack)
#define HC_DN_MAX_THREADS 320                         // K <= 10240
template <int T>
static __global__ void __launch_bounds__(HC_DN_MAX_THREADS, 2) hc_mix_down_v2(
        const int8_t * __restrict__ xq, const float * __restrict__ xd, const float * __restrict__ ssq,
        const char * __restrict__ w_down, const size_t nb_down, const float * __restrict__ inj,
        float * __restrict__ dst, const int n_embd, const int lr, const float eps) {
    constexpr int hc = 4;
    __shared__ int   wsm[HC_DN_MAX_THREADS/WARP_SIZE][HC_DN_STRIDE];
    __shared__ float rr[hc][T];
    constexpr int NV  = hc_np2(2*HC_DN_R*T); // dot sums per warp: HC_DN_R*T for each of two streams
    constexpr int NSQ = hc_np2(hc*T);        // sums of squares (and inject rows) per warp
    static_assert(NV <= WARP_SIZE && NSQ <= WARP_SIZE, "hc_mix_down_v2: at most 4 tokens");
    __shared__ float part[HC_DN_MAX_THREADS/WARP_SIZE][NV];
    __shared__ float sqp[HC_DN_MAX_THREADS/WARP_SIZE][NSQ];

    const int NW   = blockDim.x / WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int D    = n_embd*hc;
    const int nb   = D / QK8_0;
    const int nbs  = n_embd / QK8_0;
    const int ne0  = lr + 2*hc;
    // block 0 writes the inject rows and the 1/rms rows; it goes first so it never waits for a free SM
    const bool dot_block = blockIdx.x > 0;
    const int row0 = ((int) blockIdx.x - 1)*HC_DN_R;
    const int b = warp*WARP_SIZE + lane;
    HC_STAMP(0);

    // small loads first; they are consumed only after the weight loads are issued
    float sq[hc*T];
#pragma unroll
    for (int st = 0; st < hc*T; ++st) {
        sq[st] = (int) threadIdx.x < nbs ? ssq[(st / hc)*nb + (st % hc)*nbs + threadIdx.x] : 0.0f;
    }
    int4  a[T][2];
    float xdv[T];
    float4 ij[T];
    if (dot_block) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int4 * a4 = (const int4 *) (xq + (size_t) t*D + (size_t) b*QK8_0);
            a[t][0] = a4[0];
            a[t][1] = a4[1];
            xdv[t]  = xd[t*nb + b];
        }
    } else {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            ij[t] = inj != nullptr && b < nb ? *(const float4 *) (inj + ((size_t) t*nb + b)*hc) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    int4 v[HC_DN_R][3];
    if (dot_block) {
#pragma unroll
        for (int r = 0; r < HC_DN_R; ++r) {
            const int4 * src = (const int4 *) (w_down + (size_t) (row0 + r)*nb_down + (size_t) warp*HC_DN_CHUNK*16);
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int c = lane + k*WARP_SIZE;
                if (c < HC_DN_CHUNK && row0 + r < lr) {
                    v[r][k] = src[c];
                }
            }
        }
    }

    // dot products, one row at a time as the row's weights arrive
    float acc[HC_DN_R][T];
    if (dot_block) {
        int4 * wsm4 = (int4 *) wsm[warp];
#pragma unroll
        for (int r = 0; r < HC_DN_R; ++r) {
            __syncwarp();
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int c = lane + k*WARP_SIZE;
                if (c < HC_DN_CHUNK && row0 + r < lr) {
                    wsm4[c] = v[r][k];
                }
            }
            __syncwarp();
            int w[QK8_0/4];
            const float dw = hc_smem_q8_0(wsm[warp], lane*(int) sizeof(block_q8_0), w);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[r][t] = dw * xdv[t] * (float) hc_dot_q8_0(w, a[t][0], a[t][1]);
            }
        }
    }

    HC_STAMP(1);
    // one transpose-reduce per warp: the hc*T sums of squares, and the dot products split by stream (a warp's 32
    // quant blocks touch at most two streams: s0 and s0 + 1), so 1/rms is applied once per warp at the end
    const int s0 = (warp*WARP_SIZE) / nbs;
    const int s  = b / nbs;
    float red[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        red[i] = 0.0f;
    }
    if (dot_block) {
#pragma unroll
        for (int r = 0; r < HC_DN_R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                red[r*T + t]             = s == s0 ? acc[r][t] : 0.0f; // static indices: no local memory
                red[HC_DN_R*T + r*T + t] = s == s0 ? 0.0f : acc[r][t];
            }
        }
    }
    float sqr[NSQ];
#pragma unroll
    for (int st = 0; st < NSQ; ++st) {
        sqr[st] = st < hc*T ? sq[st] : 0.0f;
    }
    const float dsum = dot_block ? hc_warp_reduce_multi<NV>(red) : 0.0f;
    const float ssum = hc_warp_reduce_multi<NSQ>(sqr);
    if (lane % (WARP_SIZE/NV) == 0) {
        ((float *) part[warp])[lane / (WARP_SIZE/NV)] = dsum;
    }
    if (lane % (WARP_SIZE/NSQ) == 0) {
        sqp[warp][lane / (WARP_SIZE/NSQ)] = ssum;
    }
    __syncthreads();
    if (threadIdx.x < hc*T) {
        float sum = 0.0f;
        for (int k = 0; k < NW; ++k) {
            sum += sqp[k][threadIdx.x];
        }
        rr[threadIdx.x % hc][threadIdx.x / hc] = rsqrtf(sum / n_embd + eps);
    }
    __syncthreads();
    HC_STAMP(2);

    if (!dot_block) {
        // inject rows: this thread's quant block (b < nb) scaled by its stream's 1/rms, reduced over the block
        const int sb = b < nb ? b / nbs : 0;
        float iv[NSQ];
#pragma unroll
        for (int i = 0; i < NSQ; ++i) {
            iv[i] = 0.0f;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float f = b < nb ? rr[sb][t] : 0.0f;
            iv[0*T + t] = ij[t].x * f;
            iv[1*T + t] = ij[t].y * f;
            iv[2*T + t] = ij[t].z * f;
            iv[3*T + t] = ij[t].w * f;
        }
        const float isum = hc_warp_reduce_multi<NSQ>(iv);
        __syncthreads(); // sqp is reused
        if (lane % (WARP_SIZE/NSQ) == 0) {
            sqp[warp][lane / (WARP_SIZE/NSQ)] = isum;
        }
        __syncthreads();
        if (threadIdx.x < hc*T) {
            const int r = threadIdx.x / T;
            const int t = threadIdx.x % T;
            float sum = 0.0f;
            for (int k = 0; k < NW; ++k) {
                sum += sqp[k][threadIdx.x];
            }
            dst[(size_t) t*ne0 + lr + r]      = sum;
            dst[(size_t) t*ne0 + lr + hc + r] = rr[r][t];
        }
        HC_STAMP(4);
        return;
    }

    HC_STAMP(3);
    if (threadIdx.x < HC_DN_R*T) {
        const int r = threadIdx.x / T;
        const int t = threadIdx.x % T;
        float sum = 0.0f;
        for (int k = 0; k < NW; ++k) {
            const int sk = (k*WARP_SIZE) / nbs;
            const float * pk = (const float *) part[k];
            sum += pk[r*T + t] * rr[sk][t];
            if (sk + 1 < hc) {
                sum += pk[HC_DN_R*T + r*T + t] * rr[sk + 1][t];
            }
        }
        if (row0 + r < lr) {
            const float z = sum / hc;
            dst[(size_t) t*ne0 + row0 + r] = z / (1.0f + expf(-z));
        }
    }
    HC_STAMP(4);
}

// hc_mix_up, v2: 32 channels per block, HC_UP_NW warps. Warp w owns rows (2w, 2w+1) + 2*HC_UP_NW*k of the block's
// 4x32 gate rows (row r = stream r/32, channel r%32); a row pair is 2*nb_up contiguous bytes in w_up. 16 lanes per
// row, one Q8_0 block of K = lr per lane, dp4a against lo quantized to int8 once per block.
#define HC_UP_C      32
#define HC_UP_NW     16
#define HC_UP_ROUNDS (4*HC_UP_C/(2*HC_UP_NW))
#define HC_UP_MAXW   6    // words per lane and round: a row pair of at most 6*32 words ...
#define HC_UP_STRIDE 196  // ... staged with slack
template <int T>
static __global__ void __launch_bounds__(HC_UP_NW*WARP_SIZE) hc_mix_up_v2(
        const float * __restrict__ x, const float * __restrict__ w_norm, const char * __restrict__ w_up, const size_t nb_up,
        const float * __restrict__ down, float * __restrict__ dst, const int n_embd, const int lr,
        block_q8_1 * __restrict__ yq) {
    constexpr int hc = 4;
    __shared__ int    wsm[HC_UP_NW][HC_UP_STRIDE];
    __shared__ int4   lq4[T][320/16];
    __shared__ float  ld[T][16];
    __shared__ float  rr[hc][T];
    __shared__ float  gs[hc][HC_UP_C][T];
    __shared__ float  mq[T][HC_UP_C];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int i0   = blockIdx.x*HC_UP_C;
    const int ne_down = lr + 2*hc;
    const int nbu     = lr / QK8_0;
    const int pw      = (int) nb_up/2;  // words per row pair

    float xs[hc];
    float wns[hc];
    const int mc = threadIdx.x / T;
    const int mt = threadIdx.x % T;
    if (threadIdx.x < HC_UP_C*T) {
#pragma unroll
        for (int s = 0; s < hc; ++s) {
            xs[s]  = x[((size_t) mt*hc + s)*n_embd + i0 + mc];
            wns[s] = w_norm[s*n_embd + i0 + mc];
        }
    }
    // lo -> int8 per 32-block, once per block; the values are prefetched (one per lane and round)
    constexpr int QR = (T*10 + HC_UP_NW - 1) / HC_UP_NW; // lr <= 320: at most 10 quant blocks per token
    float dv[QR];
#pragma unroll
    for (int k = 0; k < QR; ++k) {
        const int qb = warp + k*HC_UP_NW;
        if (qb < T*nbu) {
            const int t = qb / nbu;
            dv[k] = down[(size_t) t*ne_down + (qb % nbu)*QK8_0 + lane];
        }
    }
    const float rrv = threadIdx.x < hc*T ? down[(size_t) (threadIdx.x / hc)*ne_down + lr + hc + threadIdx.x % hc] : 0.0f;
    // the weights last: the small loads above are queued ahead of them
    int u[HC_UP_ROUNDS][HC_UP_MAXW];
#pragma unroll
    for (int k = 0; k < HC_UP_ROUNDS; ++k) {
        const int r = k*2*HC_UP_NW + 2*warp;
        const int * src = (const int *) (w_up + ((size_t) (r / HC_UP_C)*n_embd + i0 + r % HC_UP_C)*nb_up);
#pragma unroll
        for (int j = 0; j < HC_UP_MAXW; ++j) {
            const int c = lane + j*WARP_SIZE;
            if (c < pw) {
                u[k][j] = src[c];
            }
        }
    }
    int8_t * lq = (int8_t *) lq4;
#pragma unroll
    for (int k = 0; k < QR; ++k) {
        const int qb = warp + k*HC_UP_NW;
        if (qb < T*nbu) {
            const int t = qb / nbu;
            int8_t q;
            const float d = hc_quant_warp(dv[k], q);
            lq[t*lr + (qb % nbu)*QK8_0 + lane] = q;
            if (lane == 0) {
                ld[t][qb % nbu] = d;
            }
        }
    }
    if (threadIdx.x < hc*T) {
        rr[threadIdx.x % hc][threadIdx.x / hc] = rrv;
    }
    __syncthreads();

    const int h  = lane / 16;
    const int bq = lane % 16;
    const bool act = bq < nbu;
    int4  a[T][2];
    float ldv[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        if (act) {
            const int4 * a4 = (const int4 *) (lq + t*lr + bq*QK8_0);
            a[t][0] = a4[0];
            a[t][1] = a4[1];
            ldv[t]  = ld[t][bq];
        }
    }

#pragma unroll
    for (int k = 0; k < HC_UP_ROUNDS; ++k) {
        __syncwarp();
#pragma unroll
        for (int j = 0; j < HC_UP_MAXW; ++j) {
            const int c = lane + j*WARP_SIZE;
            if (c < pw) {
                wsm[warp][c] = u[k][j];
            }
        }
        __syncwarp();
        float gp[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            gp[t] = 0.0f;
        }
        if (act) {
            int w[QK8_0/4];
            const float dw = hc_smem_q8_0(wsm[warp], h*(int) nb_up + bq*(int) sizeof(block_q8_0), w);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                gp[t] = dw * ldv[t] * (float) hc_dot_q8_0(w, a[t][0], a[t][1]);
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                gp[t] += __shfl_xor_sync(0xFFFFFFFF, gp[t], off, WARP_SIZE);
            }
        }
        if (bq == 0) {
            const int r = k*2*HC_UP_NW + 2*warp + h;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                gs[r / HC_UP_C][r % HC_UP_C][t] = gp[t];
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < HC_UP_C*T) {
        float m = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) {
            m += xs[s] * rr[s][mt] * wns[s] / (1.0f + expf(-gs[s][mc][mt]));
        }
        m /= hc;
        dst[(size_t) mt*n_embd + i0 + mc] = m;
        mq[mt][mc] = m;
    }
    // as in hc_mix_up_q: the consumer GEMV's q8_1 input, exactly as quantize_q8_1 would produce it
    if (yq != nullptr) {
        __syncthreads();
        if (warp < T) {
            const int t  = warp;
            const int ch = i0 + lane;
            const float xi = mq[t][lane];
            float amax = fabsf(xi);
            float sum  = xi;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
            const int64_t i_cont = (int64_t) t*n_embd + ch;
            yq[i_cont / QK8_1].qs[i_cont % QK8_1] = q;
            if (lane == 0) {
                yq[i_cont / QK8_1].ds = make_half2(d, sum);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// [hc-par] env GGML_CUDA_HC_PAR (see patch-mt-hcpar.py): hc_mix_down_v2 / hc_mix_up_v2 split over more blocks / warps at T <= 4,
// bit-identical outputs. Unset / 0: nothing below is launched.

#define HC_PAR_NVAR  6   // variants per kernel, 0 = the v2 kernel
#define HC_PAR_SLOTS 8   // streams per device with their own completion counters
#define HC_PAR_NCNT  512 // counters per slot and kernel (mix_down: row blocks, mix_up: groups of 32 channels)

// completion counters of the variants whose blocks finish a row block / a q8_1 block together: zero at module load; the last block
// of a set resets its counter, so all are zero between launches. One slot per (device, stream): see hc_par_slot
static __device__ unsigned int hc_par_cnt[HC_PAR_SLOTS][2][HC_PAR_NCNT];

// hc_mix_down step 2 as hc_mix_down_v2, with R rows per block and the NW = D/1024 warp slices of a row (32 quant blocks each, the
// warps of hc_mix_down_v2) split over S blocks of NW/S warps: grid 1 + ceil(lr/R)*S. Block 1 + rb*S + ks computes rows
// [rb*R, rb*R + R) over the slices [ks*NW/S, (ks + 1)*NW/S). Per lane, per warp and over the slices the arithmetic is
// hc_mix_down_v2's (same products, same butterflies, the slices added in order 0..NW-1 with the same 1/rms), so the output is
// bit-identical. S > 1: each block stores its warps' sums in pg; the last block of a row block (counter cnt[rb]) adds all NW of them
// up. Block 0 writes the inject rows and the 1/rms rows (S > 1: its warps take the NW slices in S turns). Needs n_embd/32 <= 32*NW/S:
// the sums of squares are covered by the threads of one block, as in hc_mix_down_v2.
template <int T, int R, int S, int MINB>
static __global__ void __launch_bounds__(HC_DN_MAX_THREADS/S, MINB) hc_mix_down_par(
        const int8_t * __restrict__ xq, const float * __restrict__ xd, const float * __restrict__ ssq,
        const char * __restrict__ w_down, const size_t nb_down, const float * __restrict__ inj,
        float * __restrict__ dst, const int n_embd, const int lr, const float eps,
        float * __restrict__ pg, const int slot) {
    constexpr int hc    = 4;
    constexpr int NWMAX = HC_DN_MAX_THREADS/WARP_SIZE;
    unsigned int * cnt = hc_par_cnt[slot < 0 ? 0 : slot][0]; // used with S > 1 only (slot >= 0 then)
    constexpr int NV    = hc_np2(2*R*T); // dot sums per warp: R*T for each of two streams
    constexpr int NSQ   = hc_np2(hc*T);  // sums of squares (and inject rows) per warp
    static_assert(NV <= WARP_SIZE && NSQ <= WARP_SIZE, "hc_mix_down_par: at most 4 tokens");
    static_assert(NWMAX % S == 0, "hc_mix_down_par: S must divide the warp count");
    __shared__ int   wsm[NWMAX/S][HC_DN_STRIDE];
    __shared__ float rr[hc][T];
    __shared__ float part[NWMAX][NV];
    __shared__ float sqp[NWMAX][NSQ];
    __shared__ bool  last;

    const int NWB  = blockDim.x / WARP_SIZE; // warps of this block
    const int NW   = NWB*S;                  // warp slices of a row (hc_mix_down_v2's warps)
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int D    = n_embd*hc;
    const int nb   = D / QK8_0;
    const int nbs  = n_embd / QK8_0;
    const int ne0  = lr + 2*hc;
    // block 0 writes the inject rows and the 1/rms rows; it goes first so it never waits for a free SM
    const bool dot_block = blockIdx.x > 0;
    const int rb   = dot_block ? ((int) blockIdx.x - 1) / S : 0; // row block
    const int ks   = dot_block ? ((int) blockIdx.x - 1) % S : 0; // part of K
    const int row0 = rb*R;
    const int gw   = ks*NWB + warp;          // this warp's slice
    const int b    = gw*WARP_SIZE + lane;    // this lane's quant block

    // the weight loads first (hc_mix_down_v2 issues the small loads first; where registers are short, the compiler then spills
    // small values right after their loads and the threads wait for them before the weight loads go out)
    int4 v[R][3];
    if (dot_block) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int4 * src = (const int4 *) (w_down + (size_t) (row0 + r)*nb_down + (size_t) gw*HC_DN_CHUNK*16);
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int c = lane + k*WARP_SIZE;
                if (c < HC_DN_CHUNK && row0 + r < lr) {
                    v[r][k] = src[c];
                }
            }
        }
    }
    float sq[hc*T];
#pragma unroll
    for (int st = 0; st < hc*T; ++st) {
        sq[st] = (int) threadIdx.x < nbs ? ssq[(st / hc)*nb + (st % hc)*nbs + threadIdx.x] : 0.0f;
    }
    int4  a[T][2];
    float xdv[T];
    float4 ij[T];
    if (dot_block) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int4 * a4 = (const int4 *) (xq + (size_t) t*D + (size_t) b*QK8_0);
            a[t][0] = a4[0];
            a[t][1] = a4[1];
            xdv[t]  = xd[t*nb + b];
        }
    } else {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            ij[t] = inj != nullptr && b < nb ? *(const float4 *) (inj + ((size_t) t*nb + b)*hc) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // the sums of squares are reduced while the weights are in flight (hc_mix_down_v2 holds them until after the dot products, and
    // at T = 4 spills them: the spill store waits for the loads before the weight loads are issued); same butterfly as there
    float sqr[NSQ];
#pragma unroll
    for (int st = 0; st < NSQ; ++st) {
        sqr[st] = st < hc*T ? sq[st] : 0.0f;
    }
    const float ssum = hc_warp_reduce_multi<NSQ>(sqr);
    if (lane % (WARP_SIZE/NSQ) == 0) {
        sqp[warp][lane / (WARP_SIZE/NSQ)] = ssum;
    }

    // dot products, one row at a time as the row's weights arrive
    float acc[R][T];
    if (dot_block) {
        int4 * wsm4 = (int4 *) wsm[warp];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            __syncwarp();
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int c = lane + k*WARP_SIZE;
                if (c < HC_DN_CHUNK && row0 + r < lr) {
                    wsm4[c] = v[r][k];
                }
            }
            __syncwarp();
            int w[QK8_0/4];
            const float dw = hc_smem_q8_0(wsm[warp], lane*(int) sizeof(block_q8_0), w);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[r][t] = dw * xdv[t] * (float) hc_dot_q8_0(w, a[t][0], a[t][1]);
            }
        }
    }

    // one transpose-reduce per warp, as hc_mix_down_v2: the hc*T sums of squares, and the dot products split by stream (a slice's
    // 32 quant blocks touch at most two streams: s0 and s0 + 1)
    const int s0 = (gw*WARP_SIZE) / nbs;
    const int s  = b / nbs;
    float red[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        red[i] = 0.0f;
    }
    if (dot_block) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                red[r*T + t]       = s == s0 ? acc[r][t] : 0.0f; // static indices: no local memory
                red[R*T + r*T + t] = s == s0 ? 0.0f : acc[r][t];
            }
        }
    }
    const float dsum = dot_block ? hc_warp_reduce_multi<NV>(red) : 0.0f;
    if (lane % (WARP_SIZE/NV) == 0) {
        if constexpr (S == 1) {
            part[warp][lane / (WARP_SIZE/NV)] = dsum;
        } else {
            if (dot_block) {
                pg[((size_t) rb*NW + gw)*NV + lane / (WARP_SIZE/NV)] = dsum;
            }
        }
    }
    if constexpr (S > 1) {
        if (dot_block) {
            __threadfence(); // the slice sums in pg before the count below
        }
    }
    __syncthreads();
    // 1/rms: the threads < nbs (all in this block) hold the sums of squares, so the slices of the other blocks of hc_mix_down_v2's
    // layout would only add +0.0f
    if (threadIdx.x < hc*T) {
        float sum = 0.0f;
        for (int k = 0; k < NWB; ++k) {
            sum += sqp[k][threadIdx.x];
        }
        rr[threadIdx.x % hc][threadIdx.x / hc] = rsqrtf(sum / n_embd + eps);
    }
    if constexpr (S > 1) {
        if (dot_block && threadIdx.x == 0) {
            last = atomicAdd(cnt + rb, 1u) == S - 1;
        }
    }
    __syncthreads();

    if (!dot_block) {
        // inject rows: each slice's quant blocks (b < nb) scaled by their stream's 1/rms, reduced per slice as hc_mix_down_v2's warps
#pragma unroll
        for (int j = 0; j < S; ++j) {
            const int vw = warp + j*NWB; // slice
            const int bj = vw*WARP_SIZE + lane;
            if (j > 0) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    ij[t] = inj != nullptr && bj < nb ? *(const float4 *) (inj + ((size_t) t*nb + bj)*hc) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            const int sb = bj < nb ? bj / nbs : 0;
            float iv[NSQ];
#pragma unroll
            for (int i = 0; i < NSQ; ++i) {
                iv[i] = 0.0f;
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float f = bj < nb ? rr[sb][t] : 0.0f;
                iv[0*T + t] = ij[t].x * f;
                iv[1*T + t] = ij[t].y * f;
                iv[2*T + t] = ij[t].z * f;
                iv[3*T + t] = ij[t].w * f;
            }
            const float isum = hc_warp_reduce_multi<NSQ>(iv);
            if (j == 0) {
                __syncthreads(); // sqp is reused
            }
            if (lane % (WARP_SIZE/NSQ) == 0) {
                sqp[vw][lane / (WARP_SIZE/NSQ)] = isum;
            }
        }
        __syncthreads();
        if (threadIdx.x < hc*T) {
            const int r = threadIdx.x / T;
            const int t = threadIdx.x % T;
            float sum = 0.0f;
            for (int k = 0; k < NW; ++k) {
                sum += sqp[k][threadIdx.x];
            }
            dst[(size_t) t*ne0 + lr + r]      = sum;
            dst[(size_t) t*ne0 + lr + hc + r] = rr[r][t];
        }
        return;
    }

    if constexpr (S > 1) {
        // the last block of the row block adds up the slice sums of all S blocks
        if (!last) {
            return;
        }
        __threadfence();
        for (int i = threadIdx.x; i < NW*NV; i += blockDim.x) {
            part[i / NV][i % NV] = __ldcg(pg + (size_t) rb*NW*NV + i);
        }
        if (threadIdx.x == 0) {
            cnt[rb] = 0; // all S blocks have counted: zero for the next launch
        }
        __syncthreads();
    }

    if (threadIdx.x < R*T) {
        const int r = threadIdx.x / T;
        const int t = threadIdx.x % T;
        float sum = 0.0f;
        for (int k = 0; k < NW; ++k) {
            const int sk = (k*WARP_SIZE) / nbs;
            const float * pk = (const float *) part[k];
            sum += pk[r*T + t] * rr[sk][t];
            if (sk + 1 < hc) {
                sum += pk[R*T + r*T + t] * rr[sk + 1][t];
            }
        }
        if (row0 + r < lr) {
            const float z = sum / hc;
            dst[(size_t) t*ne0 + row0 + r] = z / (1.0f + expf(-z));
        }
    }
}

// hc_mix_up as hc_mix_up_v2 with C channels per block: 4*C gate rows (row r = stream r / C, channel r % C) = 2*C row pairs over NWU
// warps, 2*C/NWU pairs per warp, and the row-pair weights loaded as 8-byte words (v2: 4-byte). Per row the dot products (16 lanes x
// one Q8_0 block, lo quantized per 32 values by one warp), the 16-lane butterfly and the epilogue are hc_mix_up_v2's. The q8_1 block
// of 32 channels for the consumer GEMV: C = 32 as hc_mix_up_v2; C < 32: the last of the 32/C blocks of a group (counter cnt[g]) reads
// the group's 32 outputs back from dst and quantizes them with hc_mix_up_v2's code (lane = channel, the same values).
template <int T, int C, int NWU, int MINB>
static __global__ void __launch_bounds__(NWU*WARP_SIZE, MINB) hc_mix_up_par(
        const float * __restrict__ x, const float * __restrict__ w_norm, const char * __restrict__ w_up, const size_t nb_up,
        const float * __restrict__ down, float * __restrict__ dst, const int n_embd, const int lr,
        block_q8_1 * __restrict__ yq, const int slot) {
    constexpr int hc     = 4;
    constexpr int ROUNDS = 2*C/NWU;      // row pairs per warp
    unsigned int * cnt = hc_par_cnt[slot < 0 ? 0 : slot][1]; // used with C < 32 and yq only (slot >= 0 then)
    constexpr int MAXW2  = HC_UP_MAXW/2; // 8-byte words per lane and round: a row pair of at most 3*32 of them
    static_assert(ROUNDS >= 1 && ROUNDS*NWU == 2*C, "hc_mix_up_par: NWU must divide 2*C");
    static_assert(C % 2 == 0 && 32 % C == 0 && C*T <= NWU*WARP_SIZE && hc*T <= NWU*WARP_SIZE && T <= WARP_SIZE, "hc_mix_up_par shape");
    __shared__ __align__(16) int wsm[NWU][HC_UP_STRIDE];
    __shared__ int4   lq4[T][320/16];
    __shared__ float  ld[T][16];
    __shared__ float  rr[hc][T];
    __shared__ float  gs[hc][C][T];
    __shared__ float  mq[T][C];
    __shared__ float  xsm[hc][C*T];
    __shared__ float  wnsm[hc][C*T];
    __shared__ bool   last;

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int i0   = blockIdx.x*C;
    const int ne_down = lr + 2*hc;
    const int nbu     = lr / QK8_0;
    const int pw2     = (int) nb_up/4;   // 8-byte words per row pair

    // the weights first (hc_mix_down_v2 issues them last; with 64 registers the compiler spills small values right after their loads
    // and the threads wait for those before the weight loads go out)
    int2 u[ROUNDS][MAXW2];
#pragma unroll
    for (int k = 0; k < ROUNDS; ++k) {
        const int r = k*2*NWU + 2*warp;
        const int2 * src = (const int2 *) (w_up + ((size_t) (r / C)*n_embd + i0 + r % C)*nb_up);
#pragma unroll
        for (int j = 0; j < MAXW2; ++j) {
            const int c = lane + j*WARP_SIZE;
            if (c < pw2) {
                u[k][j] = src[c];
            }
        }
    }
    // lo -> int8 per 32-block, once per block; the values are prefetched (one per lane and round)
    constexpr int QR = (T*10 + NWU - 1) / NWU; // lr <= 320: at most 10 quant blocks per token
    float dv[QR];
#pragma unroll
    for (int k = 0; k < QR; ++k) {
        const int qb = warp + k*NWU;
        if (qb < T*nbu) {
            const int t = qb / nbu;
            dv[k] = down[(size_t) t*ne_down + (qb % nbu)*QK8_0 + lane];
        }
    }
    const float rrv = threadIdx.x < hc*T ? down[(size_t) (threadIdx.x / hc)*ne_down + lr + hc + threadIdx.x % hc] : 0.0f;
    // the epilogue's x and w_norm values, parked in shared memory below (held in registers through the rounds they made the
    // 64-register variants spill)
    float xs[hc];
    float wns[hc];
    const int mc = threadIdx.x / T;
    const int mt = threadIdx.x % T;
    if (threadIdx.x < C*T) {
#pragma unroll
        for (int s = 0; s < hc; ++s) {
            xs[s]  = x[((size_t) mt*hc + s)*n_embd + i0 + mc];
            wns[s] = w_norm[s*n_embd + i0 + mc];
        }
    }
    int8_t * lq = (int8_t *) lq4;
#pragma unroll
    for (int k = 0; k < QR; ++k) {
        const int qb = warp + k*NWU;
        if (qb < T*nbu) {
            const int t = qb / nbu;
            int8_t q;
            const float d = hc_quant_warp(dv[k], q);
            lq[t*lr + (qb % nbu)*QK8_0 + lane] = q;
            if (lane == 0) {
                ld[t][qb % nbu] = d;
            }
        }
    }
    if (threadIdx.x < hc*T) {
        rr[threadIdx.x % hc][threadIdx.x / hc] = rrv;
    }
    if (threadIdx.x < C*T) {
#pragma unroll
        for (int s = 0; s < hc; ++s) {
            xsm[s][threadIdx.x]  = xs[s];
            wnsm[s][threadIdx.x] = wns[s];
        }
    }
    __syncthreads();

    const int h  = lane / 16;
    const int bq = lane % 16;
    const bool act = bq < nbu;
    int4  a[T][2];
    float ldv[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        if (act) {
            const int4 * a4 = (const int4 *) (lq + t*lr + bq*QK8_0);
            a[t][0] = a4[0];
            a[t][1] = a4[1];
            ldv[t]  = ld[t][bq];
        }
    }

#pragma unroll
    for (int k = 0; k < ROUNDS; ++k) {
        __syncwarp();
#pragma unroll
        for (int j = 0; j < MAXW2; ++j) {
            const int c = lane + j*WARP_SIZE;
            if (c < pw2) {
                ((int2 *) wsm[warp])[c] = u[k][j];
            }
        }
        __syncwarp();
        float gp[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            gp[t] = 0.0f;
        }
        if (act) {
            int w[QK8_0/4];
            const float dw = hc_smem_q8_0(wsm[warp], h*(int) nb_up + bq*(int) sizeof(block_q8_0), w);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                gp[t] = dw * ldv[t] * (float) hc_dot_q8_0(w, a[t][0], a[t][1]);
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                gp[t] += __shfl_xor_sync(0xFFFFFFFF, gp[t], off, WARP_SIZE);
            }
        }
        if (bq == 0) {
            const int r = k*2*NWU + 2*warp + h;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                gs[r / C][r % C][t] = gp[t];
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < C*T) {
        float m = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) {
            m += xsm[s][threadIdx.x] * rr[s][mt] * wnsm[s][threadIdx.x] / (1.0f + expf(-gs[s][mc][mt]));
        }
        m /= hc;
        dst[(size_t) mt*n_embd + i0 + mc] = m;
        mq[mt][mc] = m;
    }
    // the consumer GEMV's q8_1 input, exactly as hc_mix_up_v2 (and quantize_q8_1) would produce it
    if (yq != nullptr) {
        if constexpr (C == 32) {
            __syncthreads();
            if (warp < T) {
                const int t  = warp;
                const int ch = i0 + lane;
                const float xi = mq[t][lane];
                float amax = fabsf(xi);
                float sum  = xi;
                amax = warp_reduce_max<QK8_1>(amax);
                sum  = warp_reduce_sum<QK8_1>(sum);
                const float  d = amax / 127.0f;
                const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
                const int64_t i_cont = (int64_t) t*n_embd + ch;
                yq[i_cont / QK8_1].qs[i_cont % QK8_1] = q;
                if (lane == 0) {
                    yq[i_cont / QK8_1].ds = make_half2(d, sum);
                }
            }
        } else {
            // the last of the 32/C blocks of this group of 32 channels quantizes the group's outputs (read back from dst)
            constexpr int G = 32/C;
            const int g = blockIdx.x / G;
            __threadfence(); // this block's outputs in dst before the count below
            __syncthreads();
            if (threadIdx.x == 0) {
                last = atomicAdd(cnt + g, 1u) == G - 1;
            }
            __syncthreads();
            if (last) {
                __threadfence();
                if (warp < T) {
                    const int t  = warp;
                    const int ch = g*32 + lane;
                    const float xi = __ldcg(dst + (size_t) t*n_embd + ch);
                    float amax = fabsf(xi);
                    float sum  = xi;
                    amax = warp_reduce_max<QK8_1>(amax);
                    sum  = warp_reduce_sum<QK8_1>(sum);
                    const float  d = amax / 127.0f;
                    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
                    const int64_t i_cont = (int64_t) t*n_embd + ch;
                    yq[i_cont / QK8_1].qs[i_cont % QK8_1] = q;
                    if (lane == 0) {
                        yq[i_cont / QK8_1].ds = make_half2(d, sum);
                    }
                }
                if (threadIdx.x == 0) {
                    cnt[g] = 0; // all blocks of the group have counted: zero for the next launch
                }
            }
        }
    }
}

// GGML_CUDA_HC_PAR, read once: "" / "0" = off; "n" = mix_down and mix_up variant n at every T; "d,u" = mix_down d, mix_up u;
// "d1,u1,d4,u4" = (d1, u1) at T = 1 and (d4, u4) at T = 2..4. Index [0] of dn / up: T = 1, [1]: T = 2..4
struct hc_par_cfg {
    int dn[2] = {0, 0};
    int up[2] = {0, 0};
};

static const hc_par_cfg & hc_par_get() {
    static const hc_par_cfg cfg = [] {
        hc_par_cfg c;
        const char * e = getenv("GGML_CUDA_HC_PAR");
        if (e == nullptr || *e == '\0') {
            return c;
        }
        int v[5] = {0, 0, 0, 0, 0};
        int n = 0;
        bool ok = true;
        const char * p = e;
        while (true) {
            char * end = nullptr;
            const long x = strtol(p, &end, 10);
            if (end == p || x < 0 || x >= HC_PAR_NVAR || n == 5) {
                ok = false;
                break;
            }
            v[n++] = (int) x;
            if (*end == '\0') {
                break;
            }
            if (*end != ',') {
                ok = false;
                break;
            }
            p = end + 1;
        }
        if (!ok || (n != 1 && n != 2 && n != 4)) {
            GGML_LOG_WARN("[hc-par] GGML_CUDA_HC_PAR=%s not understood (n | d,u | d1,u1,d4,u4 with variants 0..%d): off\n", e, HC_PAR_NVAR - 1);
            return c;
        }
        if (n == 1) {
            c.dn[0] = c.dn[1] = c.up[0] = c.up[1] = v[0];
        } else if (n == 2) {
            c.dn[0] = c.dn[1] = v[0];
            c.up[0] = c.up[1] = v[1];
        } else {
            c.dn[0] = v[0];
            c.up[0] = v[1];
            c.dn[1] = v[2];
            c.up[1] = v[3];
        }
        if (c.dn[0] || c.dn[1] || c.up[0] || c.up[1]) {
            GGML_LOG_WARN("[hc-par] GGML_CUDA_HC_PAR=%s: mix_down variant %d (T = 1) / %d (T = 2..4), mix_up variant %d / %d (0 = the v2 kernel)\n",
                e, c.dn[0], c.dn[1], c.up[0], c.up[1]);
        }
        return c;
    }();
    return cfg;
}

// this stream's slot of hc_par_cnt (the kernels address the __device__ array themselves: no CUDA call here, so nothing changes
// under graph capture); -1 when the device's HC_PAR_SLOTS slots are taken
static int hc_par_slot(const int device, cudaStream_t stream) {
    static std::mutex mutex;
    static std::map<std::pair<int, cudaStream_t>, int> slots;
    static int n_slots[GGML_CUDA_MAX_DEVICES] = {};
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(device, stream);
    auto it = slots.find(key);
    if (it == slots.end()) {
        int slot = -1;
        if (n_slots[device] < HC_PAR_SLOTS) {
            slot = n_slots[device]++;
        } else {
            GGML_LOG_WARN("[hc-par] device %d: more than %d streams, the split variants fall back to v2 on this one\n", device, HC_PAR_SLOTS);
        }
        it = slots.emplace(key, slot).first;
    }
    return it->second;
}

static void hc_par_log(const int kernel, const int var, const int T, const bool q8, const char * fmt, ...) {
    static bool logged[2][HC_PAR_NVAR][HC_FUSED_MAX_TOKENS + 1][2] = {};
    bool & l = logged[kernel][var][T][q8];
    if (l) {
        return;
    }
    l = true;
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    GGML_LOG_WARN("%s", buf);
}

// mix_down variants: R rows per block, K split over S blocks (variant 0: hc_mix_down_v2, R 4, S 1)
static constexpr int hc_par_dn_r[HC_PAR_NVAR] = {4, 2, 1, 4, 2, 4};
static constexpr int hc_par_dn_s[HC_PAR_NVAR] = {1, 1, 1, 2, 2, 1};
// mix_up variants: C channels per block, NWU warps (variant 0: hc_mix_up_v2, C 32, 16 warps)
static constexpr int hc_par_up_c[HC_PAR_NVAR] = {32, 32, 16, 8, 4, 32};
static constexpr int hc_par_up_w[HC_PAR_NVAR] = {16, 32, 16, 8, 4, 16};

// launches the selected mix_down variant; false (nothing launched) when GGML_CUDA_HC_PAR selects v2 here or the variant does not
// apply. Called where hc_mix_down_v2 would be launched (its conditions hold)
template <int T>
static bool hc_par_mix_down(ggml_backend_cuda_context & ctx, const int8_t * xq, const float * xd, const float * ssq,
        const ggml_tensor * w_down, const float * inj, float * dst, const int n_embd, const int lr, const float eps) {
    const int var = hc_par_get().dn[T == 1 ? 0 : 1];
    if (var == 0) {
        return false;
    }
    const int R   = hc_par_dn_r[var];
    const int S   = hc_par_dn_s[var];
    const int nb  = n_embd*4 / QK8_0;
    const int nw  = nb / WARP_SIZE;
    const int nbs = n_embd / QK8_0;
    const int nrb = (lr + R - 1) / R;
    if (nw % S != 0 || nbs > (nw/S)*WARP_SIZE || (S > 1 && nrb > HC_PAR_NCNT)) {
        hc_par_log(0, var, T, false, "[hc-par] mix_down variant %d not applicable (n_embd %d, lr %d): hc_mix_down_v2\n", var, n_embd, lr);
        return false;
    }
    cudaStream_t stream = ctx.stream();
    int cnt = -1; // counter slot
    ggml_cuda_pool_alloc<float> pg(ctx.pool());
    if (S > 1) {
        cnt = hc_par_slot(ctx.device, stream);
        if (cnt < 0) {
            return false;
        }
        pg.alloc((size_t) nrb*nw*WARP_SIZE); // NV <= 32 sums per slice
    }
    const dim3 grid(1 + nrb*S);
    const dim3 block((nw/S)*WARP_SIZE);
    const char * w = (const char *) w_down->data;
    const size_t nbd = w_down->nb[1];
    // minimum blocks per SM (register cap): the whole grid resident at once where the registers allow it without spilling (ptxas)
    switch (var) {
        case 1: hc_mix_down_par<T, 2, 1, 2><<<grid, block, 0, stream>>>(xq, xd, ssq, w, nbd, inj, dst, n_embd, lr, eps, pg.get(), cnt); break;
        case 2: hc_mix_down_par<T, 1, 1, T == 1 ? 5 : T == 2 ? 4 : T == 3 ? 3 : 2><<<grid, block, 0, stream>>>(xq, xd, ssq, w, nbd, inj, dst, n_embd, lr, eps, pg.get(), cnt); break;
        case 3: hc_mix_down_par<T, 4, 2, T <= 2 ? 4 : 3><<<grid, block, 0, stream>>>(xq, xd, ssq, w, nbd, inj, dst, n_embd, lr, eps, pg.get(), cnt); break;
        case 4: hc_mix_down_par<T, 2, 2, 4><<<grid, block, 0, stream>>>(xq, xd, ssq, w, nbd, inj, dst, n_embd, lr, eps, pg.get(), cnt); break;
        case 5: hc_mix_down_par<T, 4, 1, T == 4 ? 1 : 2><<<grid, block, 0, stream>>>(xq, xd, ssq, w, nbd, inj, dst, n_embd, lr, eps, pg.get(), cnt); break;
        default: GGML_ABORT("hc-par: mix_down variant %d", var);
    }
    hc_par_log(0, var, T, false, "[hc-par] mix_down variant %d: R %d rows per block, K over %d block(s), grid %u x %u threads (T %d, n_embd %d, lr %d)\n",
        var, R, S, grid.x, block.x, T, n_embd, lr);
    return true;
}

// launches the selected mix_up variant; false (nothing launched) when GGML_CUDA_HC_PAR selects v2 here or the variant does not
// apply. Called where hc_mix_up_v2 would be launched (its conditions hold)
template <int T>
static bool hc_par_mix_up(ggml_backend_cuda_context & ctx, const float * x, const float * w_norm, const ggml_tensor * w_up,
        const float * down, float * dst, const int n_embd, const int lr, block_q8_1 * yq) {
    if constexpr (T > 4) {
        GGML_UNUSED_VARS(ctx, x, w_norm, w_up, down, dst, n_embd, lr, yq);
        return false;
    } else {
        const int var = hc_par_get().up[T == 1 ? 0 : 1];
        if (var == 0) {
            return false;
        }
        const int C   = hc_par_up_c[var];
        const int NWU = hc_par_up_w[var];
        // the 8-byte weight loads: row pairs of whole 8-byte words at 8-byte aligned addresses
        if (w_up->nb[1] % 4 != 0 || (uintptr_t) w_up->data % 8 != 0 || n_embd % 32 != 0 || (yq != nullptr && C < 32 && n_embd/32 > HC_PAR_NCNT)) {
            hc_par_log(1, var, T, yq != nullptr, "[hc-par] mix_up variant %d not applicable (n_embd %d, lr %d, row %zu B): hc_mix_up_v2\n",
                var, n_embd, lr, (size_t) w_up->nb[1]);
            return false;
        }
        cudaStream_t stream = ctx.stream();
        int cnt = -1; // counter slot
        if (yq != nullptr && C < 32) {
            cnt = hc_par_slot(ctx.device, stream);
            if (cnt < 0) {
                return false;
            }
        }
        const dim3 grid(n_embd/C);
        const dim3 block(NWU*WARP_SIZE);
        const char * w = (const char *) w_up->data;
        const size_t nbu = w_up->nb[1];
        switch (var) {
            case 1: hc_mix_up_par<T, 32, 32, 1><<<grid, block, 0, stream>>>(x, w_norm, w, nbu, down, dst, n_embd, lr, yq, cnt); break;
            case 2: hc_mix_up_par<T, 16, 16, 2><<<grid, block, 0, stream>>>(x, w_norm, w, nbu, down, dst, n_embd, lr, yq, cnt); break;
            case 3: hc_mix_up_par<T,  8,  8, 4><<<grid, block, 0, stream>>>(x, w_norm, w, nbu, down, dst, n_embd, lr, yq, cnt); break;
            case 4: hc_mix_up_par<T,  4,  4, 8><<<grid, block, 0, stream>>>(x, w_norm, w, nbu, down, dst, n_embd, lr, yq, cnt); break;
            case 5: hc_mix_up_par<T, 32, 16, 1><<<grid, block, 0, stream>>>(x, w_norm, w, nbu, down, dst, n_embd, lr, yq, cnt); break;
            default: GGML_ABORT("hc-par: mix_up variant %d", var);
        }
        hc_par_log(1, var, T, yq != nullptr, "[hc-par] mix_up variant %d: %d channels per block, %d warps, grid %u x %u threads (T %d, n_embd %d, lr %d, q8_1 for the consumer GEMV: %s)\n",
            var, C, NWU, grid.x, block.x, T, n_embd, lr, yq != nullptr ? "yes" : "no");
        return true;
    }
}

static bool hc_v2_enabled() {
    static const bool off = getenv("GGML_CUDA_NO_HC_V2") != nullptr;
    return !off;
}

template <int T>
static void hc_fused_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int mode, const ggml_tensor * combine = nullptr) {
    const ggml_tensor * x = dst->src[0];
    const int n_embd = x->ne[0];
    const int hc     = x->ne[1];
    const int D      = n_embd*hc;
    cudaStream_t stream = ctx.stream();

    if (mode == 0) {
        const ggml_tensor * w_down   = dst->src[2];
        const ggml_tensor * w_inject = dst->src[3];
        const int lr = w_down->ne[1];
        const int nb = D / QK8_0;
        float eps;
        memcpy(&eps, (const int32_t *) dst->op_params + 1, sizeof(float));
        ggml_cuda_pool_alloc<int8_t> xq(ctx.pool(), (size_t) T*D);
        ggml_cuda_pool_alloc<float>  xd(ctx.pool(), (size_t) T*nb);
        ggml_cuda_pool_alloc<float>  ssq(ctx.pool(), (size_t) T*nb);
        ggml_cuda_pool_alloc<float>  inj(ctx.pool(), w_inject ? (size_t) T*nb*hc : 1);
        if (combine != nullptr) {
            hc_combine_quant_x<T><<<(T*nb + HC_WARPS - 1) / HC_WARPS, HC_THREADS, 0, stream>>>(
                (const float *) combine->src[0]->data, (const float *) combine->src[1]->data, (const float *) combine->src[2]->data,
                (float *) combine->data, combine->src[2]->ne[0], ggml_get_op_params_i32(combine, 1),
                (const float *) dst->src[1]->data, xq.get(), xd.get(), ssq.get(),
                w_inject ? (const float *) w_inject->data : nullptr, w_inject ? w_inject->nb[1] / sizeof(float) : 0, inj.get(),
                n_embd, hc);
        } else {
            hc_quant_x<T><<<(T*nb + HC_WARPS - 1) / HC_WARPS, HC_THREADS, 0, stream>>>(
                (const float *) x->data, (const float *) dst->src[1]->data, xq.get(), xd.get(), ssq.get(),
                w_inject ? (const float *) w_inject->data : nullptr, w_inject ? w_inject->nb[1] / sizeof(float) : 0, inj.get(), D, hc);
        }
        const int nw = nb / WARP_SIZE;
        if (hc_v2_enabled() && T <= 4 && hc == 4 && nb % WARP_SIZE == 0 && nw*WARP_SIZE <= HC_DN_MAX_THREADS && n_embd/QK8_0 <= nw*WARP_SIZE &&
                n_embd/QK8_0 >= WARP_SIZE &&
                (uintptr_t) w_down->data % 16 == 0 &&
                w_down->nb[1] % 16 == 0 && n_embd % QK8_0 == 0) {
            if constexpr (T <= 4) {
                // [hc-par] GGML_CUDA_HC_PAR: a split variant (false: unset / 0 / not applicable -> hc_mix_down_v2 as before)
                if (!hc_par_mix_down<T>(ctx, xq.get(), xd.get(), ssq.get(), w_down, w_inject ? inj.get() : nullptr, (float *) dst->data,
                        n_embd, lr, eps)) {
                    hc_mix_down_v2<T><<<(lr + HC_DN_R - 1)/HC_DN_R + 1, nw*WARP_SIZE, 0, stream>>>(
                        xq.get(), xd.get(), ssq.get(), (const char *) w_down->data, w_down->nb[1], w_inject ? inj.get() : nullptr,
                        (float *) dst->data, n_embd, lr, eps);
                }
            }
        } else {
            const int n_blocks = (lr + hc + HC_DOT_RPB - 1) / HC_DOT_RPB;
            hc_mix_down_dot<T><<<n_blocks, HC_THREADS, 0, stream>>>(
                xq.get(), xd.get(), ssq.get(), (const char *) w_down->data, w_down->nb[1], w_inject ? inj.get() : nullptr,
                (float *) dst->data, n_embd, hc, lr, eps);
        }
    } else if (mode == 1) {
        const ggml_tensor * w_up = dst->src[2];
        const int lr = w_up->ne[0];
        const size_t smem = (size_t) T*lr + (size_t) T*(lr/QK8_0)*sizeof(float);
        const int n_blocks = (n_embd + HC_UP_WARPS - 1) / HC_UP_WARPS;
        // quantize mixed for its consumer GEMV too (see ggml_backend_cuda_context::q8_1_cache)
        block_q8_1 * yq = nullptr;
        static const bool no_produce = getenv("GGML_CUDA_NO_Q8_PRODUCE") != nullptr || getenv("GGML_CUDA_NO_Q8_CACHE") != nullptr;
        const ggml_tensor * key = no_produce ? nullptr : ctx.q8_1_consumer(dst);
        if (key != nullptr && key->ne[0] == n_embd && n_embd % MATRIX_ROW_PADDING == 0 && HC_UP_WARPS == QK8_1) {
            yq = (block_q8_1 *) ctx.q8_1_cache_produce(key, dst, (size_t) T*n_embd/QK8_1*sizeof(block_q8_1));
        }
        if (hc_v2_enabled() && hc == 4 && n_embd % HC_UP_C == 0 && lr % QK8_0 == 0 && lr <= 320 &&
                w_up->nb[1] % 4 == 0 && (uintptr_t) w_up->data % 4 == 0 && (int) w_up->nb[1]/2 <= HC_UP_MAXW*WARP_SIZE &&
                (int) w_up->nb[1]/2 + 1 <= HC_UP_STRIDE) {
            // [hc-par] GGML_CUDA_HC_PAR: a split variant (false: unset / 0 / T > 4 / not applicable -> hc_mix_up_v2 as before)
            if (!hc_par_mix_up<T>(ctx, (const float *) x->data, (const float *) dst->src[1]->data, w_up,
                    (const float *) dst->src[3]->data, (float *) dst->data, n_embd, lr, yq)) {
                hc_mix_up_v2<T><<<n_embd/HC_UP_C, HC_UP_NW*WARP_SIZE, 0, stream>>>(
                    (const float *) x->data, (const float *) dst->src[1]->data, (const char *) w_up->data, w_up->nb[1],
                    (const float *) dst->src[3]->data, (float *) dst->data, n_embd, lr, yq);
            }
        } else {
            hc_mix_up_q<T><<<n_blocks, HC_UP_WARPS*WARP_SIZE, smem, stream>>>(
                (const float *) x->data, (const float *) dst->src[1]->data, (const char *) w_up->data, w_up->nb[1],
                (const float *) dst->src[3]->data, (float *) dst->data, n_embd, hc, lr, yq);
        }
    }
}

void ggml_cuda_op_hc_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 0);
    const ggml_tensor * x = dst->src[0];
    const int T = x->ne[2];

    if (mode == 2) {
        const ggml_tensor * down = dst->src[2];
        const int n_embd = x->ne[0];
        const int hc     = x->ne[1];
        const int64_t n  = (int64_t) n_embd*hc*T;
        hc_combine_f32<<<(n + 255) / 256, 256, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) dst->src[1]->data, (const float *) down->data, (float *) dst->data,
            n_embd, hc, T, down->ne[0], ggml_get_op_params_i32(dst, 1));
        return;
    }

    switch (T) {
        case 1: hc_fused_launch<1>(ctx, dst, mode); break;
        case 2: hc_fused_launch<2>(ctx, dst, mode); break;
        case 3: hc_fused_launch<3>(ctx, dst, mode); break;
        case 4: hc_fused_launch<4>(ctx, dst, mode); break;
        case 5: hc_fused_launch<5>(ctx, dst, mode); break;
        case 6: hc_fused_launch<6>(ctx, dst, mode); break;
        case 7: hc_fused_launch<7>(ctx, dst, mode); break;
        case 8: hc_fused_launch<8>(ctx, dst, mode); break;
        default: GGML_ABORT("hc_fused: %d tokens, at most %d", T, HC_FUSED_MAX_TOKENS);
    }
}

bool ggml_cuda_op_hc_combine_mix_down(ggml_backend_cuda_context & ctx, ggml_tensor * combine, ggml_tensor * down) {
    const ggml_tensor * x = down->src[0];
    const int T = x->ne[2];
    if (ggml_get_op_params_i32(combine, 0) != 2 || ggml_get_op_params_i32(down, 0) != 0 || x != combine ||
            !ggml_is_contiguous(combine) || !ggml_is_contiguous(combine->src[0]) || combine->src[0]->ne[0] != x->ne[0] ||
            combine->src[0]->ne[1] != x->ne[1] || combine->src[0]->ne[2] != T || T < 1 || T > HC_FUSED_MAX_TOKENS ||
            !ggml_cuda_hc_fused_supported(down)) {
        return false;
    }
    // the mix-down output is allocated after the combine, so it may sit on inputs the combine was the last to read
    for (int k = 0; k < GGML_MAX_SRC; ++k) {
        if (combine->src[k] != nullptr && ggml_cuda_ranges_overlap(down, combine->src[k])) {
            return false;
        }
    }
    switch (T) {
        case 1: hc_fused_launch<1>(ctx, down, 0, combine); break;
        case 2: hc_fused_launch<2>(ctx, down, 0, combine); break;
        case 3: hc_fused_launch<3>(ctx, down, 0, combine); break;
        case 4: hc_fused_launch<4>(ctx, down, 0, combine); break;
        case 5: hc_fused_launch<5>(ctx, down, 0, combine); break;
        case 6: hc_fused_launch<6>(ctx, down, 0, combine); break;
        case 7: hc_fused_launch<7>(ctx, down, 0, combine); break;
        case 8: hc_fused_launch<8>(ctx, down, 0, combine); break;
    }
    return true;
}

bool ggml_cuda_hc_fused_supported(const ggml_tensor * op) {
    const int mode = ggml_get_op_params_i32(op, 0);
    const ggml_tensor * x = op->src[0];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[1] > HC_MAX_HC) {
        return false;
    }
    if (mode == 2) {
        return true;
    }
    const int64_t D = x->ne[0]*x->ne[1];
    if (x->ne[0] % QK8_0 != 0) {
        return false; // a quant block must not cross streams
    }
    return x->ne[2] <= HC_FUSED_MAX_TOKENS && D % QK8_0 == 0 && WARP_SIZE % x->ne[1] == 0 &&
        op->src[2]->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(op->src[1]);
}
