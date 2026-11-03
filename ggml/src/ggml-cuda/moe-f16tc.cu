// [moe-f16tc] v4 routed-expert MUL_MAT_ID on Volta FP16 tensor cores (see patch-mt-moef16tc4.py)
#include "moe-f16tc.cuh"
#include "mma.cuh"
#include "mmid.cuh"
#include <cctype> // [moe-tile] bounded capture labels
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
#include "unary.cuh" // [moe-fuse] ggml_cuda_op_silu_single: the SILU of the CUDA GLU op (unary.cu op_silu)

using namespace ggml_cuda_mma;

#define MOE_TC_M_ALIGN 64               // the gate: M % 64 (every layout falls back to layout 0 when its RB does not divide M)
#define MOE_TC_K_ALIGN 128              // the gate: K % 128 (IQ4_NL)
#define MOE_TC_NW 8                     // warps per block: WR (rows) x WC (columns)
#define MOE_TC_NT (MOE_TC_NW*WARP_SIZE)

static int ggml_cuda_moe_f16tc_min_tokens() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_MOE_F16TC");
        return e ? atoi(e) : 0;
    }();
    return v;
}

// tiling of the kernel (see the table in patch-mt-moef16tc4.py); unset or invalid = 0 (= v3)
struct moe_f16tc_layout { int rb, cb, kt, sets; };
static const moe_f16tc_layout moe_f16tc_layouts[5] = {
    {  64, 64, 128, 2 },
    { 128, 64,  64, 1 },
    { 128, 32,  64, 1 },
    { 128, 64,  64, 2 },
    { 128, 32,  64, 2 },
};

static int ggml_cuda_moe_f16tc_layout() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_MOE_F16TC_LAYOUT");
        const int l = e ? atoi(e) : 0;
        return l >= 0 && l < 5 ? l : 0;
    }();
    return v;
}

// [moe-fuse] GGML_CUDA_MOE_F16TC_FUSE=1: gate + up + swiglu + down on one preparation (see patch-mt-moefuse.py); unset or 0 = off
bool ggml_cuda_moe_f16tc_fuse_enabled() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_MOE_F16TC_FUSE");
        return e != nullptr && atoi(e) != 0;
    }();
    return v;
}

bool ggml_cuda_moe_f16tc_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc) {
    const int min_tokens = ggml_cuda_moe_f16tc_min_tokens();
    if (min_tokens <= 0 || !volta_mma_available(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_IQ2_S && src0->type != GGML_TYPE_IQ4_NL) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    // IQ2_S: an even number of super-blocks per row keeps rows 4-byte aligned (the raw stages are read in 4-byte words)
    if (M % MOE_TC_M_ALIGN != 0 || K % (src0->type == GGML_TYPE_IQ2_S ? 2*QK_K : MOE_TC_K_ALIGN) != 0) {
        return false;
    }
    if (src1->ne[2] < min_tokens || src0->ne[2] > 1024 || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] % 4 != 0 || src0->nb[2] % 4 != 0 || (uintptr_t) src0->data % 4 != 0) {
        return false;
    }
    if ((src1->nb[1] % (2*sizeof(float))) != 0 || (src1->nb[2] % (2*sizeof(float))) != 0 || src1->nb[2] % src1->nb[1] != 0) {
        return false;
    }
    return true;
}

static __device__ __forceinline__ uint32_t moe_tc_ld_b2(const uint8_t * p) { // 4 bytes at a 2-byte aligned address
    const uint16_t * q = (const uint16_t *) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}

static __device__ __forceinline__ half2 moe_tc_h2(const uint32_t v) {
    half2 h;
    memcpy(&h, &v, sizeof(h));
    return h;
}

static __device__ __forceinline__ uint32_t moe_tc_u32(const half2 h) {
    uint32_t v;
    memcpy(&v, &h, sizeof(v));
    return v;
}

// sorted src1 rows -> FP16: y16[g] = src1[token, channel] with (token, channel) = divmod(ids_src[g], sis1).
// perm32: inside each block of 32 values, half2 p holds values p and p + 16 (the IQ4_NL nibble order, see the kernel)
template <bool perm32>
static __global__ void moe_f16tc_gather(const float * __restrict__ src1, const int32_t * __restrict__ ids_src, half2 * __restrict__ y16,
        const int K, const int64_t s11, const int64_t s12, const uint3 sis1_fd) {
    const int g = blockIdx.x;
    const uint2 tc = fast_div_modulo((uint32_t) ids_src[g], sis1_fd);
    const float * src = src1 + tc.x*s12 + tc.y*s11;
    half2 * out = y16 + (int64_t) g*(K/2);
    for (int i = threadIdx.x; i < K/2; i += blockDim.x) {
        if constexpr (perm32) {
            const int b = i / 16;
            const int p = i % 16;
            out[i] = __floats2half2_rn(src[32*b + p], src[32*b + 16 + p]);
        } else {
            out[i] = __float22half2_rn(((const float2 *) src)[i]);
        }
    }
}

// one block of 1024 threads: tile_start[e] = sum over e' < e of ceil(rows_e'/cb), tile_start[n_experts] = total (as [mmq-moe-pack])
static __global__ void moe_f16tc_tile_start(const int32_t * __restrict__ expert_bounds, const int n_experts, int32_t * __restrict__ tile_start, const int cb) {
    __shared__ int s[1024];
    const int e = threadIdx.x;
    int v = 0;
    if (e < n_experts) {
        v = (expert_bounds[e + 1] - expert_bounds[e] + cb - 1) / cb;
    }
    s[e] = v;
    __syncthreads();
    for (int off = 1; off < 1024; off *= 2) {
        const int t = e >= off ? s[e - off] : 0;
        __syncthreads();
        s[e] += t;
        __syncthreads();
    }
    if (e < n_experts) {
        tile_start[e + 1] = s[e];
    }
    if (e == 0) {
        tile_start[0] = 0;
    }
}

// raw weight data of one thread for one K stage: one 32-value sub-block of one row
template <ggml_type type> struct moe_tc_raw;
template <> struct moe_tc_raw<GGML_TYPE_IQ4_NL> { uint32_t d; uint32_t q[4]; };
template <> struct moe_tc_raw<GGML_TYPE_IQ2_S>  { uint32_t d; uint32_t sc; uint32_t qh; uint32_t qs; uint32_t sg; };

// raw stage = the bytes of every row of the block for raw_k values, read by the whole block in consecutive 4-byte words
// (segw words per row from the 4-byte aligned start at or below the row's raw_b bytes) and kept in shared memory (row stride segw_s)
template <ggml_type type> struct moe_tc_traits;
template <> struct moe_tc_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int raw_k  = 128;          // 4 blocks of 18 B = 72 B per row, always 4-byte aligned (K % 128)
    static constexpr int raw_b  = 72;
    static constexpr int segw   = 18;
    static constexpr int segw_s = 19;           // odd: the dequantizing threads (consecutive rows) hit distinct banks
    static constexpr int ntab   = 128;          // uint2 = half2[256]: byte -> the two kvalues
};
template <> struct moe_tc_traits<GGML_TYPE_IQ2_S> {
    static constexpr int raw_k  = 256;          // one 82-byte super-block per row, at offset 2 (mod 4) every other raw stage
    static constexpr int raw_b  = 82;
    static constexpr int segw   = 21;           // 84 B: [82j - 2, 82j + 82) for odd j, [82j, 82j + 84) for even j (j + 1 exists: K % 512)
    static constexpr int segw_s = 21;
    static constexpr int ntab   = 1024 + 16;    // iq2s_grid, then sign nibble -> 2 half2 xor masks
};

// words per thread per raw stage of RB rows
template <ggml_type type, int RB>
__host__ __device__ constexpr int moe_tc_nl() {
    return (RB*moe_tc_traits<type>::segw + MOE_TC_NT - 1)/MOE_TC_NT;
}

// static shared memory of the kernel: xs (RB rows) + ys (CB rows) of KT/2 + 4 half2, the tables, the raw stage
template <ggml_type type, int RB, int CB, int KT>
__host__ __device__ constexpr int moe_tc_smem_bytes() {
    return (RB + CB)*(KT/2 + 4)*4 + moe_tc_traits<type>::ntab*8 + RB*moe_tc_traits<type>::segw_s*4;
}

template <ggml_type type, int RB>
static __device__ __forceinline__ void moe_tc_raw_ld(uint32_t (&W)[moe_tc_nl<type, RB>()], const char * __restrict__ xb, const int64_t stride_row_x,
        const int rs, const int tid) {
    typedef moe_tc_traits<type> tr;
    const int off = (rs*tr::raw_b) & ~3;
#pragma unroll
    for (int i = 0; i < moe_tc_nl<type, RB>(); ++i) {
        const int w = tid + i*MOE_TC_NT;
        if ((RB*tr::segw) % MOE_TC_NT == 0 || w < RB*tr::segw) {
            W[i] = *(const uint32_t *) (xb + (w / tr::segw)*stride_row_x + off + 4*(w % tr::segw));
        }
    }
}

template <ggml_type type, int RB>
static __device__ __forceinline__ void moe_tc_raw_st(const uint32_t (&W)[moe_tc_nl<type, RB>()], uint32_t * __restrict__ raw, const int tid) {
    typedef moe_tc_traits<type> tr;
#pragma unroll
    for (int i = 0; i < moe_tc_nl<type, RB>(); ++i) {
        const int w = tid + i*MOE_TC_NT;
        if ((RB*tr::segw) % MOE_TC_NT == 0 || w < RB*tr::segw) {
            raw[(w / tr::segw)*tr::segw_s + w % tr::segw] = W[i];
        }
    }
}

template <ggml_type type>
static __device__ __forceinline__ void moe_tc_load(moe_tc_raw<type> & R, const char * __restrict__ xr, const int k0, const int sub);

template <>
__device__ __forceinline__ void moe_tc_load<GGML_TYPE_IQ4_NL>(moe_tc_raw<GGML_TYPE_IQ4_NL> & R, const char * __restrict__ xr, const int k0, const int sub) {
    const block_iq4_nl * b = (const block_iq4_nl *) xr + k0/QK4_NL + sub;
    R.d = *(const uint16_t *) &b->d;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        R.q[i] = moe_tc_ld_b2(b->qs + 4*i);
    }
}

template <>
__device__ __forceinline__ void moe_tc_load<GGML_TYPE_IQ2_S>(moe_tc_raw<GGML_TYPE_IQ2_S> & R, const char * __restrict__ xr, const int k0, const int sub) {
    const block_iq2_s * b = (const block_iq2_s *) xr + k0/QK_K;
    const int ib = (k0 % QK_K)/32 + sub;
    R.d  = *(const uint16_t *) &b->d;
    R.sc = b->scales[ib];
    R.qh = b->qh[ib];
    R.qs = moe_tc_ld_b2(b->qs + 4*ib);
    R.sg = moe_tc_ld_b2(b->qs + QK_K/8 + 4*ib);
}

// dequantize a thread's sub-block into its row of xs (half2 offset 16*sub of the stage)
template <ggml_type type>
static __device__ __forceinline__ void moe_tc_store(const moe_tc_raw<type> & R, half2 * __restrict__ xrow, const uint2 * __restrict__ tab);

template <>
__device__ __forceinline__ void moe_tc_store<GGML_TYPE_IQ4_NL>(const moe_tc_raw<GGML_TYPE_IQ4_NL> & R, half2 * __restrict__ xrow, const uint2 * __restrict__ tab) {
    // tab as half2[256]: byte q -> (kvalues[q & 15], kvalues[q >> 4]) = values p and p + 16 of the block (permuted K, as the gather)
    const half2 * kv = (const half2 *) tab;
    const half2 d2 = __half2half2(__ushort_as_half((unsigned short) R.d));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint4 out;
        half2 * o = (half2 *) &out;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = __hmul2(kv[(R.q[i] >> 8*j) & 0xff], d2);
        }
        *(uint4 *) (xrow + 4*i) = out;
    }
}

template <>
__device__ __forceinline__ void moe_tc_store<GGML_TYPE_IQ2_S>(const moe_tc_raw<GGML_TYPE_IQ2_S> & R, half2 * __restrict__ xrow, const uint2 * __restrict__ tab) {
    // tab: iq2s_grid (1024 x 8 B), then 16 x (masks of sign bits 0-1, masks of sign bits 2-3) of a nibble
    const uint2 * sgn = tab + 1024;
    // as dequantize_iq2_s: value j of group il = dl(il) * grid[j] * (sign bit j of qs[QK_K/8 + 4*ib + il] ? -1 : 1)
    const float d   = __half2float(__ushort_as_half((unsigned short) R.d));
    const float dl0 = __half2float(__float2half(d * (0.5f + (R.sc & 0xf)) * 0.25f));
    const float dl1 = __half2float(__float2half(d * (0.5f + (R.sc >>  4)) * 0.25f));
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const float dl  = il < 2 ? dl0 : dl1;
        const half2 dl2 = __float2half2_rn(dl);
        const half2 nd2 = __float2half2_rn(-1024.0f*dl);   // exact: dl is a half, times a power of 2
        const int idx   = ((R.qs >> 8*il) & 0xff) | ((R.qh << (8 - 2*il)) & 0x300);
        const uint2 gw  = tab[idx];
        const uint32_t sb = (R.sg >> 8*il) & 0xff;
        const uint2 s0  = sgn[sb & 15];
        const uint2 s1  = sgn[sb >> 4];
        // byte_perm puts 0x64 above each grid byte: half 0x64gg = 1024 + g; (1024 + g)*dl - 1024*dl in one rounding
        uint4 out;
        out.x = moe_tc_u32(__hfma2(moe_tc_h2(__byte_perm(gw.x, 0x64646464, 0x4140)), dl2, nd2)) ^ s0.x;
        out.y = moe_tc_u32(__hfma2(moe_tc_h2(__byte_perm(gw.x, 0x64646464, 0x4342)), dl2, nd2)) ^ s0.y;
        out.z = moe_tc_u32(__hfma2(moe_tc_h2(__byte_perm(gw.y, 0x64646464, 0x4140)), dl2, nd2)) ^ s1.x;
        out.w = moe_tc_u32(__hfma2(moe_tc_h2(__byte_perm(gw.y, 0x64646464, 0x4342)), dl2, nd2)) ^ s1.y;
        *(uint4 *) (xrow + 4*il) = out;
    }
}

// block = RB weight rows x CB sorted src1 columns of one expert, K stage KT, NW = 8 warps = WR (rows, 32 each) x WC (columns, CB/WC each),
// NSETS raw register sets (2: raw stages rs + 1 and rs + 2 in flight as v3, 1: rs + 1 only)
// [moe-fuse] FUSED = true: gate and up in one launch (grid.x = 2 x M/RB): row blocks blockIdx.x < gridDim.x/2 read the weights x (gate),
// the others x_up (up, same strides), and sorted column j, row r of the result goes to dst[j*s1 + which*s2 + r] (which = 0 gate, 1 up)
// instead of [slot, token, row]; loads, dequantization, mma and accumulation order are the plain kernel's. FUSED = false: the v4
// kernel (x_up is appended after the v4 parameters and unused).

// [moe-tile] exact supported env strings only; mode99 never dispatches candidate compute.
static int moe_tile_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_MOE_TILE");
        if (!e) { return 0; }
        return strcmp(e, "1") == 0 ? 1 : strcmp(e, "2") == 0 ? 2 : strcmp(e, "99") == 0 ? 99 : 0;
    }();
    return mode;
}

// Bounded, explicit capture mode only. moetile-capture.py disables BOTH graph paths for this run.
// Graph launch policy changes, not router/compute kernels. Never used in timed ABAB.
static void moe_tile_capture(ggml_backend_cuda_context & ctx, const int32_t * bounds, const int32_t * ids_dst, const int n_experts,
        const int n_tokens, const int n_used, const char * node_name) {
    const char * path = getenv("MOE_TILE_CAPTURE_FILE");
    if (moe_tile_mode() != 99 || !path || !*path || (n_tokens != 1536 && n_tokens != 570) || n_experts != 512 || n_used != 10) {
        return;
    }
    static std::mutex mutex;
    static int captured[2] = {0, 0};
    std::lock_guard<std::mutex> guard(mutex); // async MTP host calls share the process-wide bound
    const int slot = n_tokens == 1536 ? 0 : 1;
    if (captured[slot] >= 3) { return; }
    cudaStreamCaptureStatus capture_status;
    const cudaError_t capture_error = cudaStreamIsCapturing(ctx.stream(), &capture_status);
    if (capture_error != cudaSuccess || capture_status != cudaStreamCaptureStatusNone) {
        GGML_ABORT("[moe-tile] mode99 requires moetile-capture.py (DISABLE_GRAPHS=1 AND GRAPH_ALWAYS_MIN_TOKENS=0)");
    }
    std::vector<int32_t> host(n_experts + 1);
    std::vector<int32_t> sorted_ids(n_tokens*n_used);
    const cudaError_t copy_error = cudaMemcpyAsync(host.data(), bounds, host.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, ctx.stream());
    if (copy_error != cudaSuccess) {
        GGML_ABORT("[moe-tile] capture copy failed: %s", cudaGetErrorString(copy_error));
    }
    const cudaError_t ids_error = cudaMemcpyAsync(sorted_ids.data(), ids_dst, sorted_ids.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, ctx.stream());
    if (ids_error != cudaSuccess) {
        GGML_ABORT("[moe-tile] capture ids copy failed: %s", cudaGetErrorString(ids_error));
    }
    const cudaError_t sync_error = cudaStreamSynchronize(ctx.stream());
    if (sync_error != cudaSuccess) {
        GGML_ABORT("[moe-tile] capture sync failed: %s", cudaGetErrorString(sync_error));
    }
    if (host.front() != 0 || host.back() != n_tokens*n_used) {
        GGML_ABORT("[moe-tile] capture bounds total invalid");
    }
    for (int e = 0; e < n_experts; ++e) {
        if (host[e + 1] < host[e] || host[e + 1] - host[e] > n_tokens) {
            GGML_ABORT("[moe-tile] capture expert bounds invalid at %d", e);
        }
    }
    std::vector<int32_t> routes(n_tokens*n_used, -1);
    for (int e = 0; e < n_experts; ++e) {
        for (int j = host[e]; j < host[e + 1]; ++j) {
            const int position = sorted_ids[j];
            if (position < 0 || position >= n_tokens*n_used || routes[position] != -1) {
                GGML_ABORT("[moe-tile] capture ids_dst is not a permutation at %d", j);
            }
            routes[position] = e;
        }
    }
    for (int token = 0; token < n_tokens; ++token) {
        for (int used = 0; used < n_used; ++used) {
            const int expert = routes[token*n_used + used];
            if (expert < 0) { GGML_ABORT("[moe-tile] capture route position missing"); }
            for (int other = 0; other < used; ++other) {
                if (expert == routes[token*n_used + other]) {
                    GGML_ABORT("[moe-tile] capture repeated expert within token%d", token);
                }
            }
        }
    }
    std::string label = node_name && *node_name ? node_name : "unnamed";
    for (char & c : label) {
        if (std::isspace(static_cast<unsigned char>(c))) { c = '_'; }
    }
    label.resize(std::min(label.size(), size_t(160)));
    // Six records maximum, each <64KiB (512 counts + <=15360 IDs + bounded label).
    // Refuse a reused file on first record; append and flush every sampled ubatch node.
    FILE * file = fopen(path, captured[0] + captured[1] == 0 ? "wx" : "a");
    if (!file) { GGML_ABORT("[moe-tile] capture cannot open %s", path); }
    if (fseek(file, 0, SEEK_END) != 0 || ftell(file) < 0 || ftell(file) > 448*1024) {
        fclose(file);
        GGML_ABORT("[moe-tile] capture file exceeds bounded size or seek failed");
    }
    bool failed = fprintf(file, "MOETILE_ROWS_V1 %s %d %d %d", label.c_str(), n_tokens, n_used, n_experts) < 0;
    for (int e = 0; e < n_experts; ++e) {
        failed = fprintf(file, " %d", host[e + 1] - host[e]) < 0 || failed;
    }
    failed = fputs(" IDS", file) == EOF || failed;
    for (int expert : routes) {
        failed = fprintf(file, " %d", expert) < 0 || failed;
    }
    failed = fputc('\n', file) == EOF || failed;
    failed = fflush(file) != 0 || failed; // complete records survive request/stage timeout
    failed = fclose(file) != 0 || failed;
    if (failed) { GGML_ABORT("[moe-tile] capture write failed: %s", path); }
    ++captured[slot];
    GGML_LOG_WARN("[moe-tile] capture node=%s n=%d record=%d/3\n", label.c_str(), n_tokens, captured[slot]);
}

// Two independently prefixed descriptor lists; small offset0, large offsetcapacity.
// capacity is the old CB64 bound: adaptive positive experts use <= ceil(c/64) tiles.
static __global__ void moe_tile_prepare(const int32_t * __restrict__ bounds, const int n_experts,
        int2 * __restrict__ work, int32_t * __restrict__ state, const int capacity, const int mode) {
    __shared__ int small[1024];
    __shared__ int large[1024];
    const int e = threadIdx.x;
    const int c = e < n_experts ? bounds[e + 1] - bounds[e] : 0;
    const int ns = mode == 1 && c > 0 && c <= 32 ? 1 : 0;
    const int nl = c > 0 && ns == 0 ? (c + 63)/64 : 0;
    small[e] = ns;
    large[e] = nl;
    __syncthreads();
    for (int off = 1; off < 1024; off *= 2) {
        const int a = e >= off ? small[e - off] : 0;
        const int b = e >= off ? large[e - off] : 0;
        __syncthreads(); // every read completes before any thread overwrites scan data
        small[e] += a;
        large[e] += b;
        __syncthreads();
    }
    if (e == 0) {
        state[0] = small[1023];
        state[1] = large[1023];
        state[2] = 0;
        state[3] = 0;
    }
    if (e < n_experts) {
        if (ns) {
            work[small[e] - 1] = make_int2(e, bounds[e]);
        }
        for (int j = 0; j < nl; ++j) {
            work[capacity + large[e] - nl + j] = make_int2(e, bounds[e] + 64*j);
        }
    }
}

template <ggml_type type, int RB, int CB, int KT, int WR, int NSETS, bool FUSED = false>
__launch_bounds__(MOE_TC_NT, 2)
static __global__ void moe_f16tc_kernel(
        const char * __restrict__ x, const half2 * __restrict__ y16, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_start, float * __restrict__ dst,
        const int n_experts, const int K, const int64_t stride_row_x, const int64_t stride_expert_x,
        const int64_t s1, const int64_t s2, const uint3 nused_fd, const char * __restrict__ x_up) {
    typedef moe_tc_traits<type> tr;
    constexpr int NW     = MOE_TC_NW;
    constexpr int NT     = MOE_TC_NT;
    constexpr int WC     = NW/WR;                // column warps
    constexpr int WCOLS  = CB/WC;                // columns per warp
    constexpr int NIT    = WCOLS/8;              // 8-column B tiles per warp
    constexpr int LD     = KT/2 + 4;             // row stride of the shared tiles in half2 (conflict-free 16 B loads)
    constexpr int NL     = moe_tc_nl<type, RB>();
    constexpr int NSUB   = tr::raw_k/KT;
    constexpr int YP     = KT/8;                 // 16 B parts of a src1 column per K stage
    constexpr int YSTEP  = NT/YP;
    constexpr int NY     = CB*KT/8/NT;

    static_assert(WR*32 == RB, "warp rows");
    static_assert(WC*WR == NW, "warps");
    static_assert(WCOLS % 8 == 0, "columns per warp");
    static_assert(RB*(KT/32) == NT, "one 32-value sub-block per thread per K stage");
    static_assert(tr::raw_k % KT == 0, "K stage vs raw stage");
    static_assert(KT % 64 == 0, "KT");
    static_assert((CB*KT/8) % NT == 0, "src1 parts per thread");
    static_assert(NSETS == 1 || NSETS == 2, "NSETS");
    static_assert(moe_tc_smem_bytes<type, RB, CB, KT>() <= 48*1024, "static shared memory > 48 KB");

#if defined(VOLTA_MMA_AVAILABLE)
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;

    __shared__ half2    xs[RB*LD];
    __shared__ half2    ys[CB*LD];
    __shared__ uint2    tab[tr::ntab];
    __shared__ uint32_t raw[RB*tr::segw_s];

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;

    const int g = blockIdx.y;
    if (g >= tile_start[n_experts]) {
        return; // the whole block
    }
    int lo = 0;
    int hi = n_experts;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (tile_start[mid] <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int e    = lo;
    const int col0 = expert_bounds[e] + (g - tile_start[e])*CB;
    const int ncol = min(CB, expert_bounds[e + 1] - col0);
    // [moe-fuse] which = 0 gate / 1 up and the row block inside that matrix (FUSED = false: which = 0, row0 = blockIdx.x*RB as v4)
    [[maybe_unused]] const int which = FUSED ? (int) (blockIdx.x >= gridDim.x/2) : 0;
    int row0;
    if constexpr (FUSED) {
        row0 = (int) (blockIdx.x - which*(gridDim.x/2))*RB;
    } else {
        row0 = blockIdx.x*RB;
    }
    const int nrs  = K/tr::raw_k;

    // the block's weight rows: the next raw stage(s) in flight in registers while rs is dequantized from shared memory
    const char * xw = x;
    if constexpr (FUSED) {
        if (which) {
            xw = x_up;
        }
    }
    const char * xb = xw + e*stride_expert_x + (int64_t) row0*stride_row_x;
    uint32_t W0[NL];
    [[maybe_unused]] uint32_t W1[NSETS == 2 ? NL : 1];
    moe_tc_raw_ld<type, RB>(W0, xb, stride_row_x, 0, tid);
    if constexpr (NSETS == 2) {
        if (nrs > 1) {
            moe_tc_raw_ld<type, RB>(W1, xb, stride_row_x, 1, tid);
        }
    }

    const int wr = threadIdx.y % WR; // rows    wr*32 .. wr*32 + 31
    const int wc = threadIdx.y / WR; // columns wc*WCOLS .. wc*WCOLS + WCOLS - 1

    // dequantization: thread -> row tid % RB, sub-block tid / RB of the K stage (consecutive rows per warp: conflict-free)
    const int xr_i = tid % RB;
    const int sub  = tid / RB;
    half2 * xrow = xs + xr_i*LD + 16*sub;
    const char * rawrow = (const char *) (raw + xr_i*tr::segw_s);

    // src1: thread -> 16 B part tid % YP of columns tid / YP + YSTEP*u (YP threads read one column's KT*2 B)
    const int yq = tid % YP;
    const int yc = tid / YP;

    uint4 Y[NY];
    auto load_y = [&](const int kstart) { // src1 values kstart .. kstart + KT - 1 of this thread's parts
#pragma unroll
        for (int u = 0; u < NY; ++u) {
            const int c = yc + YSTEP*u;
            Y[u] = c < ncol ? *(const uint4 *) (y16 + (int64_t) (col0 + c)*(K/2) + kstart/2 + 4*yq) : make_uint4(0, 0, 0, 0);
        }
    };
    load_y(0);

    if constexpr (type == GGML_TYPE_IQ4_NL) {
        ((half2 *) tab)[tid] = __halves2half2(__int2half_rn(kvalues_iq4nl[tid & 15]), __int2half_rn(kvalues_iq4nl[tid >> 4]));
    } else {
        for (int i = tid; i < 1024; i += NT) {
            tab[i] = ((const uint2 *) iq2s_grid)[i];
        }
        if (tid < 16) { // sign nibble -> masks of its bits 0-1 and 2-3 (bit 2p -> bit 15, bit 2p + 1 -> bit 31)
            const uint32_t n = tid;
            tab[1024 + tid] = make_uint2(((n >> 0) & 1u) << 15 | ((n >> 1) & 1u) << 31, ((n >> 2) & 1u) << 15 | ((n >> 3) & 1u) << 31);
        }
    }
    moe_tc_raw_st<type, RB>(W0, raw, tid);
    if constexpr (NSETS == 2) {
        if (nrs > 2) {
            moe_tc_raw_ld<type, RB>(W0, xb, stride_row_x, 2, tid);
        }
    } else {
        if (nrs > 1) {
            moe_tc_raw_ld<type, RB>(W0, xb, stride_row_x, 1, tid);
        }
    }
    __syncthreads(); // tables, raw stage 0

    tile_C C[NIT];

    // raw stage rs is in shared memory and WN holds rs + 1 (NSETS == 2: the other register set holds rs + 2): dequantize and multiply
    // the NSUB K stages of rs, then copy WN to shared memory (after the last mma, the loads had the whole raw stage(s) to arrive) and
    // refill WN with rs + 1 + NSETS
    auto raw_stage = [&](const int rs, uint32_t (&WN)[NL]) {
        const char * rr = rawrow + ((rs*tr::raw_b) & 3);
#pragma unroll
        for (int h = 0; h < NSUB; ++h) {
            const int k0 = (rs*NSUB + h)*KT;
            {
                moe_tc_raw<type> R;
                moe_tc_load<type>(R, rr, h*KT, sub);
                moe_tc_store<type>(R, xrow, tab);
            }
#pragma unroll
            for (int u = 0; u < NY; ++u) {
                *(uint4 *) (ys + (yc + YSTEP*u)*LD + 4*yq) = Y[u];
            }
            __syncthreads();

            if (k0 + KT < K) { // next K stage of src1 into registers while the tensor cores work on this one
                load_y(k0 + KT);
            }

            // 8-column groups past the expert's rows skip their mma (ncol is the same in the whole block: warp-uniform)
            if (wc*WCOLS < ncol) {
#pragma unroll
                for (int kk = 0; kk < KT/2; kk += tile_A::J) {
                    tile_A A;
                    load_ldmatrix(A, xs + (wr*32)*LD + kk, LD);
#pragma unroll
                    for (int it = 0; it < NIT; ++it) {
                        if (wc*WCOLS + it*8 < ncol) {
                            tile_B B;
                            load_ldmatrix(B, ys + (wc*WCOLS + it*8)*LD + kk, LD);
                            mma(C[it], A, B);
                        }
                    }
                }
            }

            if (h == NSUB - 1 && rs + 1 < nrs) { // nobody reads raw between the barrier above and the one below
                moe_tc_raw_st<type, RB>(WN, raw, tid);
                if (rs + 1 + NSETS < nrs) {
                    moe_tc_raw_ld<type, RB>(WN, xb, stride_row_x, rs + 1 + NSETS, tid);
                }
            }
            __syncthreads();
        }
    };

    if constexpr (NSETS == 2) {
        for (int rs = 0; rs < nrs; rs += 2) {
            raw_stage(rs, W1);
            if (rs + 1 < nrs) {
                raw_stage(rs + 1, W0);
            }
        }
    } else {
        for (int rs = 0; rs < nrs; ++rs) {
            raw_stage(rs, W0);
        }
    }

    // C[it] element l: weight row wr*32 + get_i(l), sorted column wc*WCOLS + it*8 + get_j(l) -> dst[slot, token, row]
#pragma unroll
    for (int it = 0; it < NIT; ++it) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int c = wc*WCOLS + it*8 + tile_C::get_j(l);
            if (c < ncol) {
                const int r = row0 + wr*32 + tile_C::get_i(l);
                if constexpr (FUSED) { // [moe-fuse] sorted order: gu[j][which][r]
                    dst[(int64_t) (col0 + c)*s1 + which*s2 + r] = C[it].x[l];
                } else {
                    const uint2 ts = fast_div_modulo((uint32_t) ids_dst[col0 + c], nused_fd); // token, slot
                    dst[ts.y*s1 + ts.x*s2 + r] = C[it].x[l];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y16, ids_dst, expert_bounds, tile_start, dst, n_experts, K, stride_row_x, stride_expert_x, s1, s2, nused_fd, x_up);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// [moe-tile] arithmetic cloned mechanically; candidate exactness requires GPU output hashes.
template <ggml_type type, int CB, bool FUSED>
__launch_bounds__(MOE_TC_NT, 2)
static __global__ void moe_tile_kernel(
        const char * __restrict__ x, const half2 * __restrict__ y16, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int2 * __restrict__ work, int32_t * __restrict__ state, float * __restrict__ dst,
        const int capacity, const int nrowblocks, const int K, const int64_t stride_row_x, const int64_t stride_expert_x,
        const int64_t s1, const int64_t s2, const uint3 nused_fd, const char * __restrict__ x_up) {
    constexpr int RB = 128, KT = 64, WR = 4, NSETS = 1;
    typedef moe_tc_traits<type> tr;
    constexpr int NW     = MOE_TC_NW;
    constexpr int NT     = MOE_TC_NT;
    constexpr int WC     = NW/WR;                // column warps
    constexpr int WCOLS  = CB/WC;                // columns per warp
    constexpr int NIT    = WCOLS/8;              // 8-column B tiles per warp
    constexpr int LD     = KT/2 + 4;             // row stride of the shared tiles in half2 (conflict-free 16 B loads)
    constexpr int NL     = moe_tc_nl<type, RB>();
    constexpr int NSUB   = tr::raw_k/KT;
    constexpr int YP     = KT/8;                 // 16 B parts of a src1 column per K stage
    constexpr int YSTEP  = NT/YP;
    constexpr int NY     = CB*KT/8/NT;

    static_assert(WR*32 == RB, "warp rows");
    static_assert(WC*WR == NW, "warps");
    static_assert(WCOLS % 8 == 0, "columns per warp");
    static_assert(RB*(KT/32) == NT, "one 32-value sub-block per thread per K stage");
    static_assert(tr::raw_k % KT == 0, "K stage vs raw stage");
    static_assert(KT % 64 == 0, "KT");
    static_assert((CB*KT/8) % NT == 0, "src1 parts per thread");
    static_assert(NSETS == 1 || NSETS == 2, "NSETS");
    static_assert(moe_tc_smem_bytes<type, RB, CB, KT>() <= 48*1024, "static shared memory > 48 KB");

#if defined(VOLTA_MMA_AVAILABLE)
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;

    __shared__ half2    xs[RB*LD];
    __shared__ half2    ys[CB*LD];
    __shared__ uint2    tab[tr::ntab];
    __shared__ uint32_t raw[RB*tr::segw_s];

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;

    __shared__ int moe_tile_task;
    constexpr int group = CB == 32 ? 0 : 1;
    // All threads take the same task and the same exit; the cursor is independent per group.
    for (;;) {
        if (tid == 0) {
            moe_tile_task = atomicAdd(&state[2 + group], 1);
        }
        __syncthreads();
        const int task = moe_tile_task;
        if (task >= state[group]*nrowblocks) {
            break;
        }
        const int descriptor_index = task/nrowblocks;
        const int rowblock = task % nrowblocks;
        const int2 descriptor = work[group*capacity + descriptor_index];
        const int e = descriptor.x;
        const int col0 = descriptor.y;
        const int ncol = min(CB, expert_bounds[e + 1] - col0);
        const int rows_per_matrix = nrowblocks/(FUSED ? 2 : 1);
        [[maybe_unused]] const int which = FUSED ? rowblock/rows_per_matrix : 0;
        const int row0 = (rowblock % rows_per_matrix)*RB;
    const int nrs  = K/tr::raw_k;

    // the block's weight rows: the next raw stage(s) in flight in registers while rs is dequantized from shared memory
    const char * xw = x;
    if constexpr (FUSED) {
        if (which) {
            xw = x_up;
        }
    }
    const char * xb = xw + e*stride_expert_x + (int64_t) row0*stride_row_x;
    uint32_t W0[NL];
    [[maybe_unused]] uint32_t W1[NSETS == 2 ? NL : 1];
    moe_tc_raw_ld<type, RB>(W0, xb, stride_row_x, 0, tid);
    if constexpr (NSETS == 2) {
        if (nrs > 1) {
            moe_tc_raw_ld<type, RB>(W1, xb, stride_row_x, 1, tid);
        }
    }

    const int wr = threadIdx.y % WR; // rows    wr*32 .. wr*32 + 31
    const int wc = threadIdx.y / WR; // columns wc*WCOLS .. wc*WCOLS + WCOLS - 1

    // dequantization: thread -> row tid % RB, sub-block tid / RB of the K stage (consecutive rows per warp: conflict-free)
    const int xr_i = tid % RB;
    const int sub  = tid / RB;
    half2 * xrow = xs + xr_i*LD + 16*sub;
    const char * rawrow = (const char *) (raw + xr_i*tr::segw_s);

    // src1: thread -> 16 B part tid % YP of columns tid / YP + YSTEP*u (YP threads read one column's KT*2 B)
    const int yq = tid % YP;
    const int yc = tid / YP;

    uint4 Y[NY];
    auto load_y = [&](const int kstart) { // src1 values kstart .. kstart + KT - 1 of this thread's parts
#pragma unroll
        for (int u = 0; u < NY; ++u) {
            const int c = yc + YSTEP*u;
            Y[u] = c < ncol ? *(const uint4 *) (y16 + (int64_t) (col0 + c)*(K/2) + kstart/2 + 4*yq) : make_uint4(0, 0, 0, 0);
        }
    };
    load_y(0);

    if constexpr (type == GGML_TYPE_IQ4_NL) {
        ((half2 *) tab)[tid] = __halves2half2(__int2half_rn(kvalues_iq4nl[tid & 15]), __int2half_rn(kvalues_iq4nl[tid >> 4]));
    } else {
        for (int i = tid; i < 1024; i += NT) {
            tab[i] = ((const uint2 *) iq2s_grid)[i];
        }
        if (tid < 16) { // sign nibble -> masks of its bits 0-1 and 2-3 (bit 2p -> bit 15, bit 2p + 1 -> bit 31)
            const uint32_t n = tid;
            tab[1024 + tid] = make_uint2(((n >> 0) & 1u) << 15 | ((n >> 1) & 1u) << 31, ((n >> 2) & 1u) << 15 | ((n >> 3) & 1u) << 31);
        }
    }
    moe_tc_raw_st<type, RB>(W0, raw, tid);
    if constexpr (NSETS == 2) {
        if (nrs > 2) {
            moe_tc_raw_ld<type, RB>(W0, xb, stride_row_x, 2, tid);
        }
    } else {
        if (nrs > 1) {
            moe_tc_raw_ld<type, RB>(W0, xb, stride_row_x, 1, tid);
        }
    }
    __syncthreads(); // tables, raw stage 0

    tile_C C[NIT];

    // raw stage rs is in shared memory and WN holds rs + 1 (NSETS == 2: the other register set holds rs + 2): dequantize and multiply
    // the NSUB K stages of rs, then copy WN to shared memory (after the last mma, the loads had the whole raw stage(s) to arrive) and
    // refill WN with rs + 1 + NSETS
    auto raw_stage = [&](const int rs, uint32_t (&WN)[NL]) {
        const char * rr = rawrow + ((rs*tr::raw_b) & 3);
#pragma unroll
        for (int h = 0; h < NSUB; ++h) {
            const int k0 = (rs*NSUB + h)*KT;
            {
                moe_tc_raw<type> R;
                moe_tc_load<type>(R, rr, h*KT, sub);
                moe_tc_store<type>(R, xrow, tab);
            }
#pragma unroll
            for (int u = 0; u < NY; ++u) {
                *(uint4 *) (ys + (yc + YSTEP*u)*LD + 4*yq) = Y[u];
            }
            __syncthreads();

            if (k0 + KT < K) { // next K stage of src1 into registers while the tensor cores work on this one
                load_y(k0 + KT);
            }

            // 8-column groups past the expert's rows skip their mma (ncol is the same in the whole block: warp-uniform)
            if (wc*WCOLS < ncol) {
#pragma unroll
                for (int kk = 0; kk < KT/2; kk += tile_A::J) {
                    tile_A A;
                    load_ldmatrix(A, xs + (wr*32)*LD + kk, LD);
#pragma unroll
                    for (int it = 0; it < NIT; ++it) {
                        if (wc*WCOLS + it*8 < ncol) {
                            tile_B B;
                            load_ldmatrix(B, ys + (wc*WCOLS + it*8)*LD + kk, LD);
                            mma(C[it], A, B);
                        }
                    }
                }
            }

            if (h == NSUB - 1 && rs + 1 < nrs) { // nobody reads raw between the barrier above and the one below
                moe_tc_raw_st<type, RB>(WN, raw, tid);
                if (rs + 1 + NSETS < nrs) {
                    moe_tc_raw_ld<type, RB>(WN, xb, stride_row_x, rs + 1 + NSETS, tid);
                }
            }
            __syncthreads();
        }
    };

    if constexpr (NSETS == 2) {
        for (int rs = 0; rs < nrs; rs += 2) {
            raw_stage(rs, W1);
            if (rs + 1 < nrs) {
                raw_stage(rs + 1, W0);
            }
        }
    } else {
        for (int rs = 0; rs < nrs; ++rs) {
            raw_stage(rs, W0);
        }
    }

    // C[it] element l: weight row wr*32 + get_i(l), sorted column wc*WCOLS + it*8 + get_j(l) -> dst[slot, token, row]
#pragma unroll
    for (int it = 0; it < NIT; ++it) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int c = wc*WCOLS + it*8 + tile_C::get_j(l);
            if (c < ncol) {
                const int r = row0 + wr*32 + tile_C::get_i(l);
                if constexpr (FUSED) { // [moe-fuse] sorted order: gu[j][which][r]
                    dst[(int64_t) (col0 + c)*s1 + which*s2 + r] = C[it].x[l];
                } else {
                    const uint2 ts = fast_div_modulo((uint32_t) ids_dst[col0 + c], nused_fd); // token, slot
                    dst[ts.y*s1 + ts.x*s2 + r] = C[it].x[l];
                }
            }
        }
    }
        __syncthreads(); // [moe-tile] finish every output store before reusing shared arrays for the next task
    } // persistent tasks; C[] is constructed afresh inside each task
#else
    GGML_UNUSED_VARS(x, y16, ids_dst, expert_bounds, work, state, dst, capacity, nrowblocks, K, stride_row_x, stride_expert_x, s1, s2, nused_fd, x_up);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

struct moe_f16tc_args {
    int device;
    cudaStream_t stream;
    int64_t M;
    int64_t ntiles_max;
    const char * x;
    const half2 * y16;
    const int32_t * ids_dst;
    const int32_t * bounds;
    const int32_t * tstart;
    float * dst;
    int n_experts;
    int K;
    int64_t stride_row_x;
    int64_t stride_expert_x;
    int64_t s1;
    int64_t s2;
    uint3 nused_fd;
    const char * x_up; // [moe-fuse] the up weights of a FUSED launch (nullptr otherwise)
};

template <ggml_type type, int RB, int CB, int KT, int WR, int NSETS, bool FUSED>
static void moe_f16tc_launch(const moe_f16tc_args & a) {
    // two blocks per SM need up to 2 x 48,512 B of shared memory (IQ2_S layout 0): ask for the largest carveout once per device and instantiation
    static bool carveout[GGML_CUDA_MAX_DEVICES] = {};
    if (!carveout[a.device]) {
        carveout[a.device] = true;
        CUDA_CHECK(cudaFuncSetAttribute(moe_f16tc_kernel<type, RB, CB, KT, WR, NSETS, FUSED>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    }
    const dim3 grid((unsigned) ((FUSED ? 2 : 1)*(a.M/RB)), (unsigned) a.ntiles_max, 1); // [moe-fuse] FUSED: gate row blocks, then up row blocks
    const dim3 block(WARP_SIZE, MOE_TC_NW, 1);
    moe_f16tc_kernel<type, RB, CB, KT, WR, NSETS, FUSED><<<grid, block, 0, a.stream>>>(a.x, a.y16, a.ids_dst, a.bounds, a.tstart, a.dst,
        a.n_experts, a.K, a.stride_row_x, a.stride_expert_x, a.s1, a.s2, a.nused_fd, a.x_up);
}

template <ggml_type type, bool FUSED = false>
static void moe_f16tc_dispatch(const int layout, const moe_f16tc_args & a) {
    switch (layout) {
        case 1:  moe_f16tc_launch<type, 128, 64,  64, 4, 1, FUSED>(a); break;
        case 2:  moe_f16tc_launch<type, 128, 32,  64, 4, 1, FUSED>(a); break;
        case 3:  moe_f16tc_launch<type, 128, 64,  64, 4, 2, FUSED>(a); break;
        case 4:  moe_f16tc_launch<type, 128, 32,  64, 4, 2, FUSED>(a); break;
        default: moe_f16tc_launch<type,  64, 64, 128, 2, 2, FUSED>(a); break;
    }
}


// [moe-tile] candidate-only attributes. Pool allocations nest inside the caller and free LIFO.
template <ggml_type type, int CB, bool FUSED>
static void moe_tile_launch(const moe_f16tc_args & a, const int2 * work, int32_t * state, const int ctas) {
    static std::once_flag carveout[GGML_CUDA_MAX_DEVICES];
    std::call_once(carveout[a.device], [&] {
        CUDA_CHECK(cudaFuncSetAttribute(moe_tile_kernel<type, CB, FUSED>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    });
    const int nrowblocks = (FUSED ? 2 : 1)*(int) (a.M/128);
    const dim3 block(WARP_SIZE, MOE_TC_NW, 1);
    moe_tile_kernel<type, CB, FUSED><<<ctas, block, 0, a.stream>>>(a.x, a.y16, a.ids_dst, a.bounds,
        work, state, a.dst, (int) a.ntiles_max, nrowblocks, a.K, a.stride_row_x, a.stride_expert_x, a.s1, a.s2, a.nused_fd, a.x_up);
}

template <ggml_type type, bool FUSED = false>
static bool moe_tile_dispatch(ggml_backend_cuda_context & ctx, const int layout, const moe_f16tc_args & a) {
    const int mode = moe_tile_mode();
    if ((mode != 1 && mode != 2) || layout != 1 || a.M % 128 != 0 || a.n_experts <= 0 || a.n_experts > 1024) {
        return false;
    }
    const int nsm = ggml_cuda_info().devices[a.device].nsm;
    GGML_ASSERT(nsm > 0 && a.ntiles_max > 0 && a.ntiles_max <= 65535);
    const int ctas = std::min(2*nsm, 2048);
    static std::once_flag logged;
    std::call_once(logged, [&] {
        GGML_LOG_WARN("[moe-tile] mode=%d type=%s fused=%d RB=128 CB=32/64 KT=64 WR=4 sets=1 CTAs=%d SMs=%d candidate\n",
            mode, ggml_type_name(type), (int) FUSED, ctas, nsm);
    });
    ggml_cuda_pool_alloc<int2> work(ctx.pool(), 2*a.ntiles_max);
    ggml_cuda_pool_alloc<int32_t> state(ctx.pool(), 4);
    moe_tile_prepare<<<1, 1024, 0, a.stream>>>(a.bounds, a.n_experts, work.get(), state.get(), (int) a.ntiles_max, mode);
    moe_tile_launch<type, 32, FUSED>(a, work.get(), state.get(), ctas);
    moe_tile_launch<type, 64, FUSED>(a, work.get(), state.get(), ctas);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// first call of each type (IQ2_S, IQ4_NL) on this path, plain or [moe-fuse]: the v4 log line (text as v4; moved here by patch-mt-moefuse.py)
static void moe_f16tc_log_first(const ggml_type type, const int layout, const int64_t M, const int64_t K, const int64_t n_experts,
        const int64_t n_used, const int64_t n_tokens) {
    const moe_f16tc_layout lay = moe_f16tc_layouts[layout];
    static bool logged[2] = {false, false};
    if (!logged[type == GGML_TYPE_IQ4_NL]) {
        logged[type == GGML_TYPE_IQ4_NL] = true;
        GGML_LOG_WARN("[moe-f16tc] v4 layout %d (RB %d CB %d KT %d sets %d) %s rows %lld K %lld experts %lld used %lld tokens %lld (first call of this type)\n",
            layout, lay.rb, lay.cb, lay.kt, lay.sets,
            ggml_type_name(type), (long long) M, (long long) K, (long long) n_experts, (long long) n_used, (long long) n_tokens);
    }
}

void ggml_cuda_mul_mat_id_f16tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t K         = src0->ne[0];
    const int64_t M         = src0->ne[1];
    const int64_t n_experts = src0->ne[2];
    const int64_t n_used    = ids->ne[0];
    const int64_t n_tokens  = src1->ne[2];
    const int64_t total     = n_tokens*n_used;
    GGML_ASSERT(ids->ne[1] == n_tokens);

    int layout = ggml_cuda_moe_f16tc_layout();
    if (M % moe_f16tc_layouts[layout].rb != 0) {
        layout = 0;
    }
    const moe_f16tc_layout lay = moe_f16tc_layouts[layout];

    moe_f16tc_log_first(src0->type, layout, M, K, n_experts, n_used, n_tokens);

    ggml_cuda_pool_alloc<int32_t> ids_src(ctx.pool(), total);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), total);
    ggml_cuda_pool_alloc<int32_t> bounds (ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> tstart (ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<half2>   y16    (ctx.pool(), total*(K/2));

    const int si1  = (int) (ids->nb[1] / ggml_type_size(ids->type));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    GGML_ASSERT(sis1 > 0);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src.get(), ids_dst.get(), bounds.get(),
        (int) n_experts, (int) n_tokens, (int) n_used, (int) src1->ne[1], si1, sis1, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());
    moe_tile_capture(ctx, bounds.get(), ids_dst.get(), (int) n_experts, (int) n_tokens, (int) n_used, dst->name);

    const int64_t s11 = src1->nb[1]/sizeof(float);
    const int64_t s12 = src1->nb[2]/sizeof(float);
    const uint3 sis1_fd = init_fastdiv_values((uint32_t) sis1);
    if (src0->type == GGML_TYPE_IQ4_NL) {
        moe_f16tc_gather<true> <<<(unsigned) total, 256, 0, stream>>>((const float *) src1->data, ids_src.get(), y16.get(), (int) K, s11, s12, sis1_fd);
    } else {
        moe_f16tc_gather<false><<<(unsigned) total, 256, 0, stream>>>((const float *) src1->data, ids_src.get(), y16.get(), (int) K, s11, s12, sis1_fd);
    }
    moe_f16tc_tile_start<<<1, 1024, 0, stream>>>(bounds.get(), (int) n_experts, tstart.get(), lay.cb);

    const int64_t ntiles_used = (total + lay.cb - 1)/lay.cb + n_experts;
    const int64_t ntiles_all  = n_experts*((n_tokens + lay.cb - 1)/lay.cb);
    const int64_t ntiles_max  = ntiles_used < ntiles_all ? ntiles_used : ntiles_all;
    GGML_ASSERT(ntiles_max <= 65535);

    moe_f16tc_args a;
    a.device          = ctx.device;
    a.stream          = stream;
    a.M               = M;
    a.ntiles_max      = ntiles_max;
    a.x               = (const char *) src0->data;
    a.y16             = y16.get();
    a.ids_dst         = ids_dst.get();
    a.bounds          = bounds.get();
    a.tstart          = tstart.get();
    a.dst             = (float *) dst->data;
    a.n_experts       = (int) n_experts;
    a.K               = (int) K;
    a.stride_row_x    = (int64_t) src0->nb[1];
    a.stride_expert_x = (int64_t) src0->nb[2];
    a.s1              = dst->nb[1]/sizeof(float);
    a.s2              = dst->nb[2]/sizeof(float);
    a.nused_fd        = init_fastdiv_values((uint32_t) n_used);
    a.x_up            = nullptr;

    switch (src0->type) {
        case GGML_TYPE_IQ2_S:  if (!moe_tile_dispatch<GGML_TYPE_IQ2_S, false>(ctx, layout, a)) { moe_f16tc_dispatch<GGML_TYPE_IQ2_S>(layout, a); } break;
        case GGML_TYPE_IQ4_NL: if (!moe_tile_dispatch<GGML_TYPE_IQ4_NL, false>(ctx, layout, a)) { moe_f16tc_dispatch<GGML_TYPE_IQ4_NL>(layout, a); } break;
        default:
            GGML_ABORT("moe-f16tc: unsupported type %s", ggml_type_name(src0->type));
    }
    CUDA_CHECK(cudaGetLastError());
}

// [moe-fuse] SWIGLU of the fused gate/up result gu (sorted column j: M gate values, then M up values), written as the FP16 rows that
// the down call's gather would build from the GLU output: down row j = GLU row ids_src[j] = token*n_used + slot = (token, slot) of
// sorted column j (the GLU output is contiguous with n_used channels, checked by the caller). The value is computed exactly as the
// CUDA GLU op computes SWIGLU for F32 - unary.cu unary_gated_op_kernel<op_silu, float>: dst[i] = (T)(op((float)x[j0]) * (float)g[j1])
// with op_silu(x) = ggml_cuda_op_silu_single(x) (unary.cuh) - and rounded as moe_f16tc_gather<perm32> rounds (perm32: half2 p of each
// block of 32 holds values p and p + 16). One thread per output half2; consecutive threads read consecutive values of a column.
template <bool perm32>
static __global__ void moe_f16tc_swiglu(const float * __restrict__ gu, half2 * __restrict__ y16, const int M, const uint32_t n,
        const uint3 mh_fd) {
    const uint32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    const uint2 ji = fast_div_modulo(t, mh_fd); // sorted column j, half2 i of its M/2
    const float * g = gu + (int64_t) ji.x*(2*M);
    const float * u = g + M;
    const int i = (int) ji.y;
    int k0;
    int k1;
    if constexpr (perm32) {
        k0 = 32*(i/16) + i%16;
        k1 = k0 + 16;
    } else {
        k0 = 2*i;
        k1 = 2*i + 1;
    }
    const float v0 = ggml_cuda_op_silu_single(g[k0]) * u[k0];
    const float v1 = ggml_cuda_op_silu_single(g[k1]) * u[k1];
    if constexpr (perm32) {
        y16[t] = __floats2half2_rn(v0, v1);
    } else {
        y16[t] = __float22half2_rn(make_float2(v0, v1));
    }
}

// [moe-fuse] gate = MUL_MAT_ID(Wg, x, ids), up = MUL_MAT_ID(Wu, x, ids), glu = SWIGLU(gate, up), down = MUL_MAT_ID(Wd, glu, ids) on one
// mm_ids_helper, one gather of x, one tile_start per distinct CB, one gate + up launch, a SWIGLU kernel and the plain down launch.
// Writes only down->data; bit-identical to the three plain [moe-f16tc] calls and the GLU op (see patch-mt-moefuse.py).
void ggml_cuda_moe_f16tc_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu, ggml_tensor * down) {
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * wg   = gate->src[0];
    const ggml_tensor * wu   = up->src[0];
    const ggml_tensor * src1 = gate->src[1]; // x, shared by gate and up
    const ggml_tensor * ids  = gate->src[2]; // shared by gate, up and down
    const ggml_tensor * wd   = down->src[0];

    const int64_t K         = wg->ne[0];     // hidden = K of gate / up = rows of down
    const int64_t M         = wg->ne[1];     // FFN = rows of gate / up = K of down
    const int64_t Md        = wd->ne[1];
    const int64_t n_experts = wg->ne[2];
    const int64_t n_used    = ids->ne[0];
    const int64_t n_tokens  = src1->ne[2];
    const int64_t total     = n_tokens*n_used;
    GGML_ASSERT(ids->ne[1] == n_tokens);
    GGML_ASSERT(up->src[1] == src1 && up->src[2] == ids && down->src[1] == glu && down->src[2] == ids);
    GGML_ASSERT(wu->type == wg->type && wu->ne[0] == K && wu->ne[1] == M && wu->ne[2] == n_experts && wu->nb[1] == wg->nb[1] && wu->nb[2] == wg->nb[2]);
    GGML_ASSERT(wd->ne[0] == M && wd->ne[2] == n_experts);
    GGML_ASSERT(glu->ne[0] == M && glu->ne[1] == n_used && glu->ne[2] == n_tokens && ggml_is_contiguous(glu));
    GGML_ASSERT(total*(M/2) <= (int64_t) UINT32_MAX);

    // the layouts the plain calls would choose (a layout whose RB does not divide the rows falls back to 0)
    int layout_gu = ggml_cuda_moe_f16tc_layout();
    if (M % moe_f16tc_layouts[layout_gu].rb != 0) {
        layout_gu = 0;
    }
    int layout_d = ggml_cuda_moe_f16tc_layout();
    if (Md % moe_f16tc_layouts[layout_d].rb != 0) {
        layout_d = 0;
    }
    const moe_f16tc_layout lay_gu = moe_f16tc_layouts[layout_gu];
    const moe_f16tc_layout lay_d  = moe_f16tc_layouts[layout_d];

    moe_f16tc_log_first(wg->type, layout_gu, M,  K, n_experts, n_used, n_tokens);
    moe_f16tc_log_first(wd->type, layout_d,  Md, M, n_experts, n_used, n_tokens);
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_WARN("[moe-fuse] gate+up+swiglu+down fused: rows %d/%d K %d experts %d used %d tokens %d layout %d%s\n",
            (int) M, (int) Md, (int) K, (int) n_experts, (int) n_used, (int) n_tokens, layout_gu,
            layout_d != layout_gu ? " (down: layout 0, its RB does not divide the rows)" : "");
    }

    // pool (LIFO): ids, tile starts, gu; y16 in an inner scope that ends before y16d -> peak = gu + y16 + ids
    ggml_cuda_pool_alloc<int32_t> ids_src (ctx.pool(), total);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), total);
    ggml_cuda_pool_alloc<int32_t> bounds  (ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> tstart  (ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> tstart_d(ctx.pool());
    if (lay_d.cb != lay_gu.cb) {
        tstart_d.alloc(n_experts + 1);
    }

    // as the plain gate / up call: the sorted order and ids_dst / bounds depend on ids only, so they serve the down call as well
    const int si1  = (int) (ids->nb[1] / ggml_type_size(ids->type));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    GGML_ASSERT(sis1 > 0);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src.get(), ids_dst.get(), bounds.get(),
        (int) n_experts, (int) n_tokens, (int) n_used, (int) src1->ne[1], si1, sis1, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());
    moe_tile_capture(ctx, bounds.get(), ids_dst.get(), (int) n_experts, (int) n_tokens, (int) n_used, gate->name);

    moe_f16tc_tile_start<<<1, 1024, 0, stream>>>(bounds.get(), (int) n_experts, tstart.get(), lay_gu.cb);
    if (lay_d.cb != lay_gu.cb) {
        moe_f16tc_tile_start<<<1, 1024, 0, stream>>>(bounds.get(), (int) n_experts, tstart_d.get(), lay_d.cb);
    }

    const auto ntiles = [&](const int cb) {
        const int64_t ntiles_used = (total + cb - 1)/cb + n_experts;
        const int64_t ntiles_all  = n_experts*((n_tokens + cb - 1)/cb);
        const int64_t ntiles_max  = ntiles_used < ntiles_all ? ntiles_used : ntiles_all;
        GGML_ASSERT(ntiles_max <= 65535);
        return ntiles_max;
    };

    ggml_cuda_pool_alloc<float> gu(ctx.pool(), total*2*M);
    {
        ggml_cuda_pool_alloc<half2> y16(ctx.pool(), total*(K/2));

        const int64_t s11 = src1->nb[1]/sizeof(float);
        const int64_t s12 = src1->nb[2]/sizeof(float);
        const uint3 sis1_fd = init_fastdiv_values((uint32_t) sis1);
        if (wg->type == GGML_TYPE_IQ4_NL) {
            moe_f16tc_gather<true> <<<(unsigned) total, 256, 0, stream>>>((const float *) src1->data, ids_src.get(), y16.get(), (int) K, s11, s12, sis1_fd);
        } else {
            moe_f16tc_gather<false><<<(unsigned) total, 256, 0, stream>>>((const float *) src1->data, ids_src.get(), y16.get(), (int) K, s11, s12, sis1_fd);
        }

        moe_f16tc_args a;
        a.device          = ctx.device;
        a.stream          = stream;
        a.M               = M;
        a.ntiles_max      = ntiles(lay_gu.cb);
        a.x               = (const char *) wg->data;
        a.y16             = y16.get();
        a.ids_dst         = ids_dst.get();
        a.bounds          = bounds.get();
        a.tstart          = tstart.get();
        a.dst             = gu.get();
        a.n_experts       = (int) n_experts;
        a.K               = (int) K;
        a.stride_row_x    = (int64_t) wg->nb[1];
        a.stride_expert_x = (int64_t) wg->nb[2];
        a.s1              = 2*M;              // sorted column j
        a.s2              = M;                // which (0 gate, 1 up)
        a.nused_fd        = init_fastdiv_values((uint32_t) n_used);
        a.x_up            = (const char *) wu->data;

        switch (wg->type) {
            case GGML_TYPE_IQ2_S:  if (!moe_tile_dispatch<GGML_TYPE_IQ2_S, true>(ctx, layout_gu, a)) { moe_f16tc_dispatch<GGML_TYPE_IQ2_S,  true>(layout_gu, a); } break;
            case GGML_TYPE_IQ4_NL: if (!moe_tile_dispatch<GGML_TYPE_IQ4_NL, true>(ctx, layout_gu, a)) { moe_f16tc_dispatch<GGML_TYPE_IQ4_NL, true>(layout_gu, a); } break;
            default:
                GGML_ABORT("moe-fuse: unsupported type %s", ggml_type_name(wg->type));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_pool_alloc<half2> y16d(ctx.pool(), total*(M/2));
    {
        const uint32_t nh = (uint32_t) (total*(M/2));
        const uint3 mh_fd = init_fastdiv_values((uint32_t) (M/2));
        if (wd->type == GGML_TYPE_IQ4_NL) {
            moe_f16tc_swiglu<true> <<<(nh + 255)/256, 256, 0, stream>>>(gu.get(), y16d.get(), (int) M, nh, mh_fd);
        } else {
            moe_f16tc_swiglu<false><<<(nh + 255)/256, 256, 0, stream>>>(gu.get(), y16d.get(), (int) M, nh, mh_fd);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // the plain down launch on the same sorted order
    moe_f16tc_args a;
    a.device          = ctx.device;
    a.stream          = stream;
    a.M               = Md;
    a.ntiles_max      = ntiles(lay_d.cb);
    a.x               = (const char *) wd->data;
    a.y16             = y16d.get();
    a.ids_dst         = ids_dst.get();
    a.bounds          = bounds.get();
    a.tstart          = lay_d.cb != lay_gu.cb ? tstart_d.get() : tstart.get();
    a.dst             = (float *) down->data;
    a.n_experts       = (int) n_experts;
    a.K               = (int) M;
    a.stride_row_x    = (int64_t) wd->nb[1];
    a.stride_expert_x = (int64_t) wd->nb[2];
    a.s1              = down->nb[1]/sizeof(float);
    a.s2              = down->nb[2]/sizeof(float);
    a.nused_fd        = init_fastdiv_values((uint32_t) n_used);
    a.x_up            = nullptr;

    switch (wd->type) {
        case GGML_TYPE_IQ2_S:  if (!moe_tile_dispatch<GGML_TYPE_IQ2_S, false>(ctx, layout_d, a)) { moe_f16tc_dispatch<GGML_TYPE_IQ2_S>(layout_d, a); } break;
        case GGML_TYPE_IQ4_NL: if (!moe_tile_dispatch<GGML_TYPE_IQ4_NL, false>(ctx, layout_d, a)) { moe_f16tc_dispatch<GGML_TYPE_IQ4_NL>(layout_d, a); } break;
        default:
            GGML_ABORT("moe-fuse: unsupported type %s", ggml_type_name(wd->type));
    }
    CUDA_CHECK(cudaGetLastError());
}
