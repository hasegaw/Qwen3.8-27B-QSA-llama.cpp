// [gdn-chunk] chunked Gated DeltaNet prefill on Volta (sm_70) tensor cores, see gdn-chunk.cuh.
// Port of gdn_single from upstream PR #29353 (gated-delta-net-mma.cu): per block one value head and V value columns,
// advancing by chunks of C = 32 tokens with the recurrent state in FP32 accumulator tiles. Products use the Volta
// ggml_cuda_mma tiles (mma.sync.m8n8k4, FP32 accumulation) on FP16 hi/lo operand pairs: x = hi + lo with
// hi = half(clamp(x, +-65504)), lo = half(x - hi); every product is lo*hi + hi*lo + hi*hi (the PR's BF16 scheme).
// All shared operands are stored "K contiguous" ([row][k] for A, [col][k] for B) so every tile load is one 16-byte LDS;
// rows are XOR swizzled in 16-byte units so that 8 consecutive rows hit distinct banks.

#include "gdn-chunk.cuh"
#include "mma.cuh"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace {

constexpr int GDN_D = 128; // S_k = S_v
constexpr int GDN_C = 32;  // chunk length = row count of a Volta tile

// R x L matrix of FP16 hi/lo pairs, row-major with the 16-byte units of each row XOR swizzled
template <int R, int L> struct hmat {
    static_assert(L == 32 || L == 128, "unsupported row length");
    half hi[R * L];
    half lo[R * L];

    static __device__ __forceinline__ int index(const int r, const int c) {
        const int sw = L == 128 ? (r & 7) : ((r >> 1) & 3);
        return r * L + (((c >> 3) ^ sw) << 3) + (c & 7);
    }

    static __device__ __forceinline__ float clamp(const float x) { return fminf(fmaxf(x, -65504.0f), 65504.0f); }

    __device__ __forceinline__ void store(const int r, const int c, const float x) {
        const int  i = index(r, c);
        const half h = __float2half_rn(clamp(x));
        hi[i]        = h;
        lo[i]        = __float2half_rn(x - __half2float(h));
    }

    // c even
    __device__ __forceinline__ void store2(const int r, const int c, const float2 x) {
        const int    i  = index(r, c);
        const half2  h  = __floats2half2_rn(clamp(x.x), clamp(x.y));
        const float2 hf = __half22float2(h);
        *reinterpret_cast<half2 *>(hi + i) = h;
        *reinterpret_cast<half2 *>(lo + i) = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
    }

    // c multiple of 4
    __device__ __forceinline__ void store4(const int r, const int c, const float4 x) {
        store2(r, c, make_float2(x.x, x.y));
        store2(r, c + 2, make_float2(x.z, x.w));
    }
};

template <int V> struct alignas(16) gdn_smem {
    union {
        hmat<GDN_C, GDN_D> q;  // q * scale [t][d], until the projections
        hmat<GDN_D, GDN_C> kT; // k transposed [d][t] for the state update
    } qk;
    hmat<GDN_C, GDN_D> k;       // [t][d]
    hmat<GDN_C, GDN_C> p;       // decayed causal q k^T [t][u]
    hmat<GDN_C, GDN_C> inverse; // (I + L)^-1 [t][u]
    hmat<V, GDN_C>     delta;   // [v][t]

    union {
        hmat<V, GDN_D> state;  // [v][d]
        hmat<V, GDN_C> solved; // [v][t]
    } scratch;

    float lower[GDN_C * GDN_C]; // beta-scaled decayed strictly lower k k^T [t][u]
    float prefix[GDN_C];
    float beta[GDN_C];
};

#ifdef VOLTA_MMA_AVAILABLE
using namespace ggml_cuda_mma;
using acc_t = tile<32, 8, float>;                                // 32 rows x 8 columns, FP32
using a_t   = tile<32, 4, half2>;                                // 32 rows x 8 k (FP16)
using b_t   = tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;   // 8 columns x 8 k (FP16), B addressed as B-transpose

// 8 consecutive k of row (row0 + tile row) of m: one 16-byte load
template <class T, int R, int L>
static __device__ __forceinline__ T load_tile(const hmat<R, L> & m, const bool low, const int row0, const int k0) {
    T          t;
    const int4 x   = *reinterpret_cast<const int4 *>((low ? m.lo : m.hi) + hmat<R, L>::index(row0 + T::get_i(0), k0));
    int *      xi  = reinterpret_cast<int *>(t.x);
    xi[0]          = x.x;
    xi[1]          = x.y;
    xi[2]          = x.z;
    xi[3]          = x.w;
    return t;
}

// c(32 x 8) += a[0..32)[0..K) * b[col..col+8)[0..K)^T, preserving both first-order residual terms (PR order)
template <int K, int RA, int LA, int RB, int LB>
static __device__ __forceinline__ void product(acc_t & c, const hmat<RA, LA> & a, const hmat<RB, LB> & b, const int col) {
#pragma unroll
    for (int k = 0; k < K; k += 8) {
        const a_t ah = load_tile<a_t>(a, false, 0, k);
        const a_t al = load_tile<a_t>(a, true, 0, k);
        const b_t bh = load_tile<b_t>(b, false, col, k);
        const b_t bl = load_tile<b_t>(b, true, col, k);
        mma(c, al, bh);
        mma(c, ah, bl);
        mma(c, ah, bh);
    }
}

// two products sharing the B operand
template <int K, int RA, int LA, int RB, int LB>
static __device__ __forceinline__ void product2(acc_t & c1, const hmat<RA, LA> & a1, acc_t & c2, const hmat<RA, LA> & a2,
                                                const hmat<RB, LB> & b, const int col) {
#pragma unroll
    for (int k = 0; k < K; k += 8) {
        const b_t bh  = load_tile<b_t>(b, false, col, k);
        const b_t bl  = load_tile<b_t>(b, true, col, k);
        const a_t a1h = load_tile<a_t>(a1, false, 0, k);
        const a_t a1l = load_tile<a_t>(a1, true, 0, k);
        mma(c1, a1l, bh);
        mma(c1, a1h, bl);
        mma(c1, a1h, bh);
        const a_t a2h = load_tile<a_t>(a2, false, 0, k);
        const a_t a2l = load_tile<a_t>(a2, true, 0, k);
        mma(c2, a2l, bh);
        mma(c2, a2h, bl);
        mma(c2, a2h, bh);
    }
}
#endif // VOLTA_MMA_AVAILABLE

// grid: H * (128 / V) blocks of 32 x 8 threads; n_seqs == 1, a.n_tokens a multiple of 32
template <int V>
static __global__ void __launch_bounds__(256, 1) gdn_chunk_volta(const ggml_cuda_gdn_chunk_args a) {
#ifdef VOLTA_MMA_AVAILABLE
    constexpr int C        = GDN_C;
    constexpr int D        = GDN_D;
    constexpr int NT       = V / 8;             // 8-column value tiles
    constexpr int PER_WARP = (D / 32) * NT / 8; // state tiles per warp
    static_assert(V == 32 || V == 64, "V must be 32 or 64");
    static_assert(acc_t::ne == 8 && a_t::ne == 4 && b_t::ne == 4, "unexpected Volta tile shapes");

    extern __shared__ __align__(16) unsigned char gdn_chunk_bytes[];
    gdn_smem<V> & s = *reinterpret_cast<gdn_smem<V> *>(gdn_chunk_bytes);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int h    = blockIdx.x / (D / V);
    const int v0   = (blockIdx.x % (D / V)) * V;

    const float * s_in  = a.state + (a.state_rows ? (int64_t) a.state_rows[0] * a.state_row_stride : 0) + (int64_t) h * D * D;
    float *       s_out = a.state_out + (int64_t) h * D * D;
    const int64_t qoff  = (int64_t) (h % a.H_k) * a.sq1 + 4 * lane;
    const float * vh    = a.v + (int64_t) h * a.sv1 + v0;
    const float * gh    = a.g + (int64_t) h * a.sb1;
    const float * bh    = a.beta + (int64_t) h * a.sb1;
    float *       out   = a.dst + (int64_t) h * D + v0;
    const int64_t ostr  = a.H * D;

    // state tile j of this warp: rows (d) sr(j)..+32, columns (v) sc..sc+8; tile index warp + 8 j as in the PR
    const int sc = (warp % NT) * 8;
    auto      sr = [&](const int j) { return ((warp + 8 * j) / NT) * 32; };
    // projection / output tile of this warp (value columns 8 pt..), if pt < NT; warps 0..3 also do the Gram / q k^T tiles
    const int pt = (warp + 4) % 8;

    acc_t state[PER_WARP];
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
#pragma unroll
        for (int l = 0; l < acc_t::ne; ++l) {
            state[j].x[l] = s_in[(int64_t) (v0 + sc + acc_t::get_j(l)) * D + sr(j) + acc_t::get_i(l)];
        }
    }

    // q / k rows of the next chunk are carried in registers to hide the global load latency
    float4 nq[C / 8], nk[C / 8];
    auto   read_qk = [&](const int64_t t0) {
#pragma unroll
        for (int j = 0; j < C / 8; ++j) {
            const int64_t off = qoff + (t0 + warp + 8 * j) * a.sq2;
            const float * qp  = a.q + off;
            const float * kp  = a.k + off;
            if ((((uintptr_t) qp | (uintptr_t) kp) & 15) == 0) {
                nq[j] = *reinterpret_cast<const float4 *>(qp);
                nk[j] = *reinterpret_cast<const float4 *>(kp);
            } else {
                nq[j] = make_float4(qp[0], qp[1], qp[2], qp[3]);
                nk[j] = make_float4(kp[0], kp[1], kp[2], kp[3]);
            }
        }
    };
    read_qk(0);

    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += C) {
        // ---- stage q * scale, k, gate prefix sums, beta and the state
#pragma unroll
        for (int j = 0; j < C / 8; ++j) {
            const int    t = warp + 8 * j;
            const float4 q = nq[j];
            s.qk.q.store4(t, 4 * lane, make_float4(q.x * a.scale, q.y * a.scale, q.z * a.scale, q.w * a.scale));
            s.k.store4(t, 4 * lane, nk[j]);
        }
        if (warp == 0) {
            float       g    = gh[(t0 + lane) * a.sb2];
            const float beta = bh[(t0 + lane) * a.sb2];
#pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                const float previous = __shfl_up_sync(0xffffffff, g, offset, 32);
                if (lane >= offset) {
                    g += previous;
                }
            }
            s.prefix[lane] = g;
            s.beta[lane]   = beta;
        }
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
#pragma unroll
            for (int l = 0; l < acc_t::ne; ++l) {
                s.scratch.state.store(sc + acc_t::get_j(l), sr(j) + acc_t::get_i(l), state[j].x[l]);
            }
        }
        __syncthreads();
        if (t0 + C < a.n_tokens) {
            read_qk(t0 + C);
        }

        // ---- Gram (k k^T) and q k^T tiles (warps 0..3), projections q S and k S (one value tile per warp)
        acc_t base_output;
        if (warp < 4) {
            const int c = 8 * warp;
            acc_t     xk, xq;
            product2<D>(xk, s.k, xq, s.qk.q, s.k, c);
#pragma unroll
            for (int l = 0; l < acc_t::ne; l += 2) {
                const int   rr = acc_t::get_i(l), cc = c + acc_t::get_j(l);
                const float d0 = cc <= rr ? expf(s.prefix[rr] - s.prefix[cc]) : 0.f;
                const float d1 = cc + 1 <= rr ? expf(s.prefix[rr] - s.prefix[cc + 1]) : 0.f;
                if (cc < rr) {
                    s.lower[rr * C + cc] = xk.x[l] * d0 * s.beta[rr];
                }
                if (cc + 1 < rr) {
                    s.lower[rr * C + cc + 1] = xk.x[l + 1] * d1 * s.beta[rr];
                }
                s.p.store2(rr, cc, make_float2(xq.x[l] * d0, xq.x[l + 1] * d1));
            }
        }
        if (pt < NT) {
            const int c = 8 * pt;
            float     vv[acc_t::ne];
#pragma unroll
            for (int l = 0; l < acc_t::ne; ++l) {
                vv[l] = vh[(t0 + acc_t::get_i(l)) * a.sv2 + c + acc_t::get_j(l)];
            }
            acc_t qs, ks;
            product2<D>(qs, s.qk.q, ks, s.k, s.scratch.state, c);
#pragma unroll
            for (int l = 0; l < acc_t::ne; ++l) {
                const int   t     = acc_t::get_i(l), v = c + acc_t::get_j(l);
                const float decay = expf(s.prefix[t]);
                s.delta.store(v, t, s.beta[t] * (vv[l] - decay * ks.x[l]));
                base_output.x[l] = decay * qs.x[l];
            }
        }
        __syncthreads();

        // ---- (I + L)^-1 by forward substitution in FP32, one row per warp at a time (lane = column)
        for (int row = warp; row < C; row += 8) {
            float x = lane == row ? 1.f : (lane < row ? -s.lower[row * C + lane] : 0.f);
            for (int pivot = row - 1; pivot > 0; --pivot) {
                const float xp = __shfl_sync(0xffffffff, x, pivot, 32);
                if (lane < pivot) {
                    x -= xp * s.lower[pivot * C + lane];
                }
            }
            s.inverse.store(row, lane, x);
        }
        __syncthreads();

        // ---- solved = (I + L)^-1 delta
        for (int ti = warp; ti < NT; ti += 8) {
            acc_t x;
            product<C>(x, s.inverse, s.delta, 8 * ti);
#pragma unroll
            for (int l = 0; l < acc_t::ne; ++l) {
                s.scratch.solved.store(8 * ti + acc_t::get_j(l), acc_t::get_i(l), x.x[l]);
            }
        }
        __syncthreads();

        // ---- outputs, decayed solved values for the state update, k transposed into the (now free) q buffer
        if (pt < NT) {
            const int c = 8 * pt;
            acc_t     x;
            product<C>(x, s.p, s.scratch.solved, c);
#pragma unroll
            for (int l = 0; l < acc_t::ne; ++l) {
                out[(t0 + acc_t::get_i(l)) * ostr + c + acc_t::get_j(l)] = base_output.x[l] + x.x[l];
            }
        }
        {
            const float last = s.prefix[C - 1];
            for (int i = tid; i < V * C; i += 256) {
                const int   v = i / C, t = i % C, j = hmat<V, C>::index(v, t);
                const float x = __half2float(s.scratch.solved.hi[j]) + __half2float(s.scratch.solved.lo[j]);
                s.delta.store(v, t, x * expf(last - s.prefix[t]));
            }
        }
        for (int i = tid; i < C * D / 2; i += 256) {
            const int   t  = i % C, d = 2 * (i / C);
            const int   src = hmat<C, D>::index(t, d);
            const int   d0 = hmat<D, C>::index(d, t), d1 = hmat<D, C>::index(d + 1, t);
            const half2 xh = *reinterpret_cast<const half2 *>(s.k.hi + src);
            const half2 xl = *reinterpret_cast<const half2 *>(s.k.lo + src);
            s.qk.kT.hi[d0] = __low2half(xh);
            s.qk.kT.hi[d1] = __high2half(xh);
            s.qk.kT.lo[d0] = __low2half(xl);
            s.qk.kT.lo[d1] = __high2half(xl);
        }
        __syncthreads();

        // ---- state = state * exp(G_last) + k^T delta (B shared by all tiles of the warp)
        {
            const float decay = expf(s.prefix[C - 1]);
#pragma unroll
            for (int j = 0; j < PER_WARP; ++j) {
#pragma unroll
                for (int l = 0; l < acc_t::ne; ++l) {
                    state[j].x[l] *= decay;
                }
            }
#pragma unroll
            for (int k = 0; k < C; k += 8) {
                const b_t bh8 = load_tile<b_t>(s.delta, false, sc, k);
                const b_t bl8 = load_tile<b_t>(s.delta, true, sc, k);
#pragma unroll
                for (int j = 0; j < PER_WARP; ++j) {
                    const a_t ah = load_tile<a_t>(s.qk.kT, false, sr(j), k);
                    const a_t al = load_tile<a_t>(s.qk.kT, true, sr(j), k);
                    mma(state[j], al, bh8);
                    mma(state[j], ah, bl8);
                    mma(state[j], ah, bh8);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
#pragma unroll
        for (int l = 0; l < acc_t::ne; ++l) {
            s_out[(int64_t) (v0 + sc + acc_t::get_j(l)) * D + sr(j) + acc_t::get_i(l)] = state[j].x[l];
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // VOLTA_MMA_AVAILABLE
}

static bool gdn_chunk_enabled() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_GDN_CHUNK");
        return e != nullptr && atoi(e) != 0;
    }();
    return v;
}

static int gdn_chunk_v() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_GDN_CHUNK_V");
        return e != nullptr && atoi(e) == 64 ? 64 : 32;
    }();
    return v;
}

// raise the dynamic shared memory limit once per device; false = the variant does not fit (fall back)
template <int V> static bool gdn_chunk_init(const int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool           available[GGML_CUDA_MAX_DEVICES] = {};
    std::call_once(once[device], [device] {
        const size_t nbytes = sizeof(gdn_smem<V>);
        if (nbytes > ggml_cuda_info().devices[device].smpbo) {
            return;
        }
        ggml_cuda_set_device(device);
        const cudaError_t status = cudaFuncSetAttribute((const void *) gdn_chunk_volta<V>,
                                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int) nbytes);
        available[device] = status == cudaSuccess;
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
    });
    return available[device];
}

} // namespace

int64_t ggml_cuda_gdn_chunk_tokens(const int device, const bool kda, const int64_t S_v, const int64_t S_k, const int64_t H,
                                   const int64_t H_k, const int64_t n_tokens, const int64_t n_seqs, const int K) {
    if (!gdn_chunk_enabled()) {
        return 0;
    }
    if (kda || S_v != GDN_D || S_k != GDN_D || n_seqs != 1 || H_k <= 0 || H % H_k != 0 || K < 1) {
        return 0;
    }
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc != GGML_CUDA_CC_VOLTA || !volta_mma_available(cc)) {
        return 0;
    }
    const int64_t n_c = ((n_tokens - K) / GDN_C) * GDN_C;
    if (n_c <= 0) {
        return 0;
    }
    const bool ok = gdn_chunk_v() == 64 ? gdn_chunk_init<64>(device) : gdn_chunk_init<32>(device);
    return ok ? n_c : 0;
}

void ggml_cuda_gdn_chunk_launch(const int device, const ggml_cuda_gdn_chunk_args & a, const int64_t n_tokens_total,
                                const int K, cudaStream_t stream) {
    GGML_UNUSED(device);
    GGML_ASSERT(a.n_tokens > 0 && a.n_tokens % GDN_C == 0);
    const int V = gdn_chunk_v();
    static std::atomic<bool> logged{ false };
    if (!logged.exchange(true)) {
        GGML_LOG_WARN("%s: [gdn-chunk] chunked Volta GDN in use: H_v=%lld H_k=%lld n_tokens=%lld n_c=%lld K=%d V=%d\n", __func__,
                      (long long) a.H, (long long) a.H_k, (long long) n_tokens_total, (long long) a.n_tokens, K, V);
    }
    const dim3 block_dims(32, 8, 1);
    if (V == 64) {
        gdn_chunk_volta<64><<<(unsigned) (a.H * (GDN_D / 64)), block_dims, sizeof(gdn_smem<64>), stream>>>(a);
    } else {
        gdn_chunk_volta<32><<<(unsigned) (a.H * (GDN_D / 32)), block_dims, sizeof(gdn_smem<32>), stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}
