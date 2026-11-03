// GGML_OP_A7_GDN_FFN (A7 P1, 2026-10-16): the second half of a GDN layer of Qwen3.8-Flash-Next (hc_mix(ffn) -> router -> MoE -> shared expert ->
// hc_combine(ffn)) for a verify window of T = 4 tokens.
//
// The stage code is ported from research/ceiling-20261015/gate0/gate0.cu (10-16 version, graph mode: one kernel per stage, k_stage<THREADS, S>):
//   hc1f (stage_h1 / stage_h1_b2), hc2f (stage_h2 / stage_h2_b2), router, gateup, qh, down (with hc_combine(ffn)).
// Differences from gate0:
//   - the weights are not synthetic copies in __constant__ memory: the pointers come from the ggml tensors of the layer (ggml_a7_ffn_layer) and are passed
//     to the kernels by value (kernel argument struct A7P, __grid_constant__);
//   - a first kernel (a7_ffn_rquant) makes the q8 activations of R * hc_ffn_norm (gate0: the down stage of the previous layer made them);
//   - the routing table (a7-route.cuh: softmax, top-10, union of the 4 tokens, the tables uni / umask / pref / pn / pt / wsel and the weights wk) is built on
//     the GPU by the last block to finish the router stage (a7_route_build; P1b), or with LLAMA_A7_ROUTE_SEPARATE=1 by one extra block (a7_ffn_route kernel).
//     gateup / qh / down copy the table into shared memory first;
//   - down writes the new residual to dst (R is read only); with a next layer on the same GPU it also makes the q8 of dst * hc_attn_norm(next) into the work
//     area (as gate0 does; work only, nothing reads it in P1);
//   - work areas (counters, q8 activations, routing table, ...) are one persistent allocation per (device, stream).
// Kernel names start with a7_ffn_.
#include "a7-gdn.cuh"
#include "ggml-a7.h"
#include "a7-route.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#define A7_FULL 0xffffffffu

namespace a7ffn {

constexpr int D = 2560, HCD = 10240, LR = 320;
constexpr int FF = 640;
constexpr int TILEB = 3328;                // per-warp smem tile for staged weight rows
constexpr int XDOFF = 24576;               // q8 activations: xq int8 [T][UT][32] at 0, xd float [T][UT] at XDOFF (UT <= 192)
constexpr int ACTB = 32768;
constexpr int GRIDOFF = 12288;             // IQ2_S grid (8 KiB) aliased inside the activation area (gate/up stage uses UT = 80 only)
constexpr int SCRF = 4096;                 // floats of static scratch smem
constexpr int IQ2ROW = 820;                // IQ2_S row, K = 2560
constexpr int IQ4ROW = 360;                // IQ4_NL row, K = 640
constexpr int EXP_GU_BYTES = FF * IQ2ROW;      // IQ2_S gate (or up) of one expert: 524,800
constexpr int EXP_D_BYTES = D * IQ4ROW;        // IQ4_NL down of one expert: 921,600
constexpr int NCH = D / 32;                // 80 chunks of 32 columns

// the routing table (A7Route, T, NEXP, TOPK, MAXU, MAXP) and its builds are in a7-route.cuh

// kernel argument: weights (pointers into the ggml tensors), input / output residual, work areas
struct A7P {
  const float *norm_f;                     // hc_ffn_norm [4][2560]
  const float *norm_next;                  // hc_attn_norm of the next layer (same GPU) or nullptr: down makes the q8 of the new R * norm_next (work only)
  const uint8_t *hcd_f, *hcu_f;            // hc_ffn_down Q8_0 10240->320 (rows 320 x 10880 B), hc_ffn_up Q8_0 320->10240 (rows 10240 x 340 B)
  const float *inj_f;                      // hc_ffn_inject [4][10240]
  const float *router;                     // ffn_gate_inp [512][2560]
  const float *sgw;                        // ffn_gate_inp_shexp [2560]
  const uint8_t *eg, *eu, *ed;             // IQ2_S, IQ2_S [512][640][820 B]; IQ4_NL [512][2560][360 B]
  const uint8_t *sg, *su, *sd;             // Q8_0 shared expert gate/up (640 rows x 2720 B), down (2560 rows x 680 B)
  const float *R;                          // input residual [4 t][4 s][2560] (not modified)
  float *Rout;                             // output residual (dst)
  float *rr, *part, *injp, *mixed, *logits, *sgl, *hbuf;
  unsigned *cnt;                           // last-arriver counters (never reset, except cnt[A7_CNT_ROUTE]: reset by the last block) and tickets
  int *h2cnt;                              // [2][NCH] last-arriver counters of stage_h2_b2 (reset by the last arriver)
  int *balcnt;                             // [NCH] last-arriver counters of the leftover chunks of stage_down (reset by the last arriver)
  int8_t *mq; float *md;                   // q8 of mixed [T][D], scales [T][80]
  int8_t *loq; float *lod;                 // q8 of lo [T][320], scales [T][10]
  int8_t *Rq; float *Rd, *Rss;             // q8 of R*norm per stream [4][T][D], scales and sums of squares [4][T][80]
  int8_t *hq; float *hd;                   // q8 of h per pair: [MAXP][640], [MAXP][20]
  A7Route *route;
  int route_inline;                        // 1: the last block of the router stage builds the routing table; 0: the separate kernel a7_ffn_route does
  int route_ref;                           // 10-16 diag: 1 = use a7_route_build_ref (the P1 build: smaller code, faster in production) in either place
  float eps;                               // rms eps of hc_mix (stage_h1): the P1 op sets 1e-6f, the P2 op hparams.f_norm_rms_eps
  // ---- P2 (attn op) only; for the attn op norm_f / hcd_f / hcu_f / inj_f are the hc_attn_* weights, norm_next is hc_ffn_norm (ssmout makes the q8 of
  // R_new * hc_ffn_norm for the ffn op), R is the input residual, Rout the output (dst)
  const uint8_t *wqkv, *wz, *wout;         // Q8_0: qkv 2560->10240 and z 2560->6144 (rows 2720 B), ssm_out 6144->2560 (rows 6528 B)
  const float *alpha, *beta;               // F32 [48][2560]
  const float *dt, *Aa;                    // F32 [48]: ssm_dt (bias of alpha), ssm_a (= -exp(A_log))
  const float *conv;                       // F32 [10240][4] (ggml ne = {4, 10240}: taps of channel c at c * 4 + j)
  const float *snorm;                      // F32 [128]
  float *conv_all, *ssm_all;               // r_l / s_l of the layer: read at row s_copy[0], written at rows s * mem_size + kv_head (s < 4)
  const int *s_copy;                       // I32 [1], on the device
  int kv_head, mem_size;
  float *qkv_o, *z_o, *a_o, *b_o;          // [4][10240] conv + silu output, [4][6144] z, [4][48] alpha, [4][48] beta (raw)
  int8_t *oq; float *od, *osq;             // q8 of the gated-norm input [4][6144], scales [4][192], sums of squares of o [4][192]
};

extern __shared__ __align__(16) uint8_t g_dsm[];
__shared__ __align__(16) float g_scr[SCRF];
__shared__ __align__(16) A7Route g_rt;

// ------------------------------------------------------------------ device helpers (gate0.cu)
static __device__ __noinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(A7_FULL, v, o);
  return v;
}
static __device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(A7_FULL, v, o));
  return v;
}
// fast variants (hardware exp2/rcp: ~1e-7 relative error)
static __device__ __forceinline__ float fsilu(float x) { return __fdividef(x, 1.f + __expf(-x)); }
static __device__ __forceinline__ float fsigmoid(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
static __device__ __forceinline__ float fsoftplus(float x) { return x > 20.f ? x : __logf(1.f + __expf(x)); }  // P2 (gate0.cu)

// q8 quantization of one block of 32 (lane holds element `lane`): d = amax/127, q = round(x/d)
static __device__ __noinline__ uint2 q8_warp_ni(float x) {  // one shared copy of the code
  const float am = warp_max(fabsf(x));
  const float d = am / 127.f;
  const int q = am == 0.f ? 0 : __float2int_rn(x * (127.f / am));
  return make_uint2((unsigned)(q & 0xFF), __float_as_uint(d));
}
static __device__ __forceinline__ void q8_warp(float x, int8_t &q, float &d) {
  const uint2 r = q8_warp_ni(x);
  q = (int8_t)r.x;
  d = __uint_as_float(r.y);
}

// reduce-scatter of N (power of 2, <= 32) per-lane values over the warp: lane l returns the total of value index (l & (N-1)).
template <int N, int MAXO = 32>
static __device__ __forceinline__ float rscatter(float (&v)[N], int lane) {
#pragma unroll
  for (int st = N / 2; st >= 1; st >>= 1) {
    const bool up = (lane & st) != 0;
#pragma unroll
    for (int i = 0; i < st; i++) {
      const float send = up ? v[i] : v[i + st];
      const float keep = up ? v[i + st] : v[i];
      v[i] = keep + __shfl_xor_sync(A7_FULL, send, st);
    }
  }
  float r = v[0];
#pragma unroll
  for (int o = N; o < MAXO; o <<= 1) r += __shfl_xor_sync(A7_FULL, r, o);
  return r;
}

static __device__ __forceinline__ int get_int_b2(const void *x, int i32) {
  const uint16_t *x16 = (const uint16_t *)x;
  return (int)x16[2 * i32 + 0] | ((int)x16[2 * i32 + 1] << 16);
}
// quantized unit in smem = 2-byte scale + NW*4 bytes of quants, units 2-byte aligned (address mod 4 is 0 or 2): the NW ints of the quants from NW+1 aligned
// 32-bit loads and funnel shifts (instead of 2 x 16-bit loads per int); returns the scale
template <int NW>
static __device__ __forceinline__ float unit_words(const uint8_t *b, int (&w)[NW]) {
  const uintptr_t a = (uintptr_t)b;
  const uint32_t *p = (const uint32_t *)(a & ~(uintptr_t)3);
  const bool even = (a & 2) == 0;
  uint32_t v[NW + 1];
#pragma unroll
  for (int i = 0; i <= NW; i++) v[i] = p[i];
  const int sh = even ? 16 : 32;
#pragma unroll
  for (int i = 0; i < NW; i++) w[i] = (int)__funnelshift_rc(v[i], v[i + 1], sh);
  return __half2float(__ushort_as_half((unsigned short)(even ? (v[0] & 0xFFFFu) : (v[0] >> 16))));
}
static __device__ __forceinline__ float h2f(const uint8_t *p) { return __half2float(*(const half *)p); }
__host__ __device__ constexpr uint32_t pack4(int a, int b, int c, int d) {
  return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) | ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}
static __device__ __forceinline__ int2 table16(int q4) {  // llama.cpp get_int_from_table_16 with the IQ4_NL table
  const uint32_t t0 = pack4(-127, -104, -83, -65), t1 = pack4(-49, -35, -22, -10), t2 = pack4(1, 13, 25, 38), t3 = pack4(53, 69, 89, 113);
  uint32_t tmp[2];
  const uint32_t sel = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low = __byte_perm(t0, t1, q4 >> shift);
    const uint32_t high = __byte_perm(t2, t3, q4 >> shift);
    tmp[i] = __byte_perm(low, high, sel >> shift);
  }
  return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

#define A7_XQ(sm) ((const int4 *)(sm))
#define A7_XD(sm) ((const float *)((sm) + XDOFF))

// copy R rows (PB bytes each, row r at src + r*rs) into the warp's smem tile (row r at dst + r*PB) with coalesced 8 B streaming loads
template <int R, int PB>
static __device__ __forceinline__ void stage_tile(uint8_t *dst, const uint8_t *src, size_t rs, int lane) {
  constexpr int CHR = PB / 8, CH = R * CHR;
  __syncwarp();
  for (int i0 = lane; i0 < CH; i0 += 32 * 8) {
    uint2 v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int idx = i0 + 32 * j;
      if (idx < CH) {
        const int r = idx / CHR, c = idx - r * CHR;
        v[j] = __ldcs(((const uint2 *)(src + (size_t)r * rs)) + c);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int idx = i0 + 32 * j;
      if (idx < CH) ((uint2 *)dst)[idx] = v[j];
    }
  }
  __syncwarp();
}

// register-staged variant of stage_tile: load() issues the global loads (all independent, kept in registers) so that they can be in flight while the
// warp computes on the previous tile; store() later writes them to the smem tile.
template <int R, int PB, int VB = 16>
struct TileRegs {
  using V = typename std::conditional<VB == 16, uint4, uint2>::type;
  static constexpr int CHR = PB / VB, CH = R * CHR, NV = (CH + 31) / 32;
  V v[NV];
  __device__ __forceinline__ void load(const uint8_t *src, size_t rs, int lane) {
#pragma unroll
    for (int j = 0; j < NV; j++) {
      const int idx = lane + 32 * j;
      if (idx < CH) {
        const int r = R == 1 ? 0 : (idx >= CHR ? 1 : 0), c = idx - r * CHR;
        v[j] = __ldcs(((const V *)(src + (size_t)r * rs)) + c);
      }
    }
  }
  __device__ __forceinline__ void store(uint8_t *dst, int lane) {
#pragma unroll
    for (int j = 0; j < NV; j++) {
      const int idx = lane + 32 * j;
      if (idx < CH) ((V *)dst)[idx] = v[j];
    }
  }
};

// Q8_0 weights of R rows x NU units in the warp tile (34 B units) against T q8 activations (smem; token stride UT units, unit offset ua)
template <int R, int NU>
static __device__ __forceinline__ void q8_tile_dot(const uint8_t *tile, const int4 *xq, const float *xd, int UT, int ua, int lane,
                                                   float (&acc)[R][T]) {
#pragma unroll 1
  for (int jj = 0; jj < (NU + 31) / 32; jj++) {
    const int u = lane + 32 * jj;
    if (u < NU) {
      int4 a[T][2];
      float dy[T];
#pragma unroll
      for (int t = 0; t < T; t++) {
        a[t][0] = xq[(t * UT + ua + u) * 2];
        a[t][1] = xq[(t * UT + ua + u) * 2 + 1];
        dy[t] = xd[t * UT + ua + u];
      }
#pragma unroll
      for (int r = 0; r < R; r++) {
        const uint8_t *b = tile + (r * NU + u) * 34;
        int w[8];
        const float d = unit_words<8>(b, w);
#pragma unroll
        for (int t = 0; t < T; t++) {
          int s = 0;
          s = __dp4a(w[0], a[t][0].x, s); s = __dp4a(w[1], a[t][0].y, s); s = __dp4a(w[2], a[t][0].z, s); s = __dp4a(w[3], a[t][0].w, s);
          s = __dp4a(w[4], a[t][1].x, s); s = __dp4a(w[5], a[t][1].y, s); s = __dp4a(w[6], a[t][1].z, s); s = __dp4a(w[7], a[t][1].w, s);
          acc[r][t] += d * dy[t] * (float)s;
        }
      }
    }
  }
}

// one Q8_0 unit (pointer b into smem) against the T activations' unit `au` (token stride UT)
static __device__ __forceinline__ void q8_unit(const uint8_t *b, const int4 *xq, const float *xd, int UT, int au, float (&acc)[T]) {
  int w[8];
  const float d = unit_words<8>(b, w);
#pragma unroll
  for (int t = 0; t < T; t++) {
    const int4 a0 = xq[(t * UT + au) * 2], a1 = xq[(t * UT + au) * 2 + 1];
    const float dy = xd[t * UT + au];
    int s = 0;
    s = __dp4a(w[0], a0.x, s); s = __dp4a(w[1], a0.y, s); s = __dp4a(w[2], a0.z, s); s = __dp4a(w[3], a0.w, s);
    s = __dp4a(w[4], a1.x, s); s = __dp4a(w[5], a1.y, s); s = __dp4a(w[6], a1.z, s); s = __dp4a(w[7], a1.w, s);
    acc[t] += d * dy * (float)s;
  }
}

// F32 matvec of one weight row (K = 2560) against T fp32 activation vectors x[t*2560 + i]; lane l returns the total of token (l & 3)
static __device__ __noinline__ float f32_dot4(const float *w, const float *x, int lane) {
  float acc[T] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int c = lane; c < D / 4; c += 32) {
    const float4 wv = __ldcs(((const float4 *)w) + c);
#pragma unroll
    for (int t = 0; t < T; t++) {
      const float4 xv = ((const float4 *)(x + t * D))[c];
      acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
  }
  return rscatter<4>(acc, lane);
}

// inject partial: injp[t][s][o] = sum_i W_inj[o][s*2560+i] * (R[t][s][i] * norm[s][i])   (one warp)
static __device__ __forceinline__ void inject_partial(const A7P &P, int s, int o, int lane) {
  float acc[T] = {0.f, 0.f, 0.f, 0.f};
  const float4 *wv = (const float4 *)(P.inj_f + (size_t)o * HCD + s * D);
  const float4 *nv = (const float4 *)(P.norm_f + s * D);
  for (int c = lane; c < D / 4; c += 32) {
    const float4 wq = __ldcs(wv + c), nq = nv[c];
#pragma unroll
    for (int t = 0; t < T; t++) {
      const float4 rq = ((const float4 *)(P.R + (size_t)(t * 4 + s) * D))[c];
      acc[t] += wq.x * (rq.x * nq.x) + wq.y * (rq.y * nq.y) + wq.z * (rq.z * nq.z) + wq.w * (rq.w * nq.w);
    }
  }
  float v = 0.f;
#pragma unroll
  for (int t = 0; t < T; t++) { const float s_ = warp_sum(acc[t]); if (lane == t) v = s_; }
  if (lane < T) P.injp[(lane * 4 + s) * 4 + o] = v;
}

// last-arriver: all lanes of the warp call it after writing their results; returns true (warp-uniformly) for the warp that completes a group of `need`
// arrivals. Counters are never reset (arrivals per launch are multiples of `need`).
static __device__ __forceinline__ bool la_arrive(unsigned *cnt, unsigned need, int lane) {
  __threadfence();
  __syncwarp();
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(cnt, 1u);
  old = __shfl_sync(A7_FULL, old, 0);
  const bool last = (old % need) == need - 1;
  if (last) __threadfence();
  return last;
}

// compact q8 activations made by a producer (int8 [T][UT*32], float scale [T][UT], same layout as the smem area) -> smem; loads and stores split so
// that other loads can be issued in between
template <int NTH, int NQ4, int ND4>
struct Q8Regs {
  static constexpr int TOT = NQ4 + ND4, NV = (TOT + NTH - 1) / NTH;
  int4 v[NV];
  __device__ __forceinline__ void load(const int4 *qs, const int4 *ds) {
#pragma unroll
    for (int j = 0; j < NV; j++) {
      const int idx = threadIdx.x + j * NTH;
      if (idx < TOT) v[j] = idx < NQ4 ? __ldcg(qs + idx) : __ldcg(ds + (idx - NQ4));
    }
  }
  __device__ __forceinline__ void store(uint8_t *sm) const {
#pragma unroll
    for (int j = 0; j < NV; j++) {
      const int idx = threadIdx.x + j * NTH;
      if (idx < TOT) {
        if (idx < NQ4) ((int4 *)sm)[idx] = v[j];
        else ((int4 *)(sm + XDOFF))[idx - NQ4] = v[j];
      }
    }
  }
};

// the routing table (global work area) -> shared memory g_rt; ends with a block barrier
template <int NTH>
static __device__ __forceinline__ void load_route(const A7P &P) {
  constexpr int NW = (int)(sizeof(A7Route) / 4);
  const int *src = (const int *)P.route;
  int *dst = (int *)&g_rt;
  for (int i = threadIdx.x; i < NW; i += NTH) dst[i] = __ldcg(src + i);
  __syncthreads();
}

// last arriver of a 32-row chunk of the hc_down stage: lo = silu(sum_s r_s * part_s / 4) of the chunk's 4 tokens, quantized (q8) for the next stage
static __device__ __forceinline__ void h1_lo(const A7P &P, int ch, int lane) {
  const int r0 = ch * 32 + lane;
  float rrv[16];
#pragma unroll
  for (int k = 0; k < 16; k++) rrv[k] = __ldcg(P.rr + k);
  float pv[16];
#pragma unroll
  for (int k = 0; k < 16; k++) pv[k] = __ldcg(P.part + k * LR + r0);
#pragma unroll
  for (int t = 0; t < T; t++) {
    float sum = rrv[t * 4 + 0] * pv[t * 4 + 0];
    sum += rrv[t * 4 + 1] * pv[t * 4 + 1];
    sum += rrv[t * 4 + 2] * pv[t * 4 + 2];
    sum += rrv[t * 4 + 3] * pv[t * 4 + 3];
    int8_t q; float d;
    q8_warp(fsilu(sum * 0.25f), q, d);
    P.loq[t * LR + r0] = q;
    if (lane == 0) P.lod[t * 10 + ch] = d;
  }
}

// ------------------------------------------------------------------ hc1f: hc_mix part 1 (rms, hc_down partial products per stream)
// units (row, stream) = one warp each; the last of the 128 units of a 32-row chunk (all rows, all streams) forms lo = silu(sum_s r_s*part_s/4) of
// its 4 tokens, quantizes it (q8) and writes loq/lod for the next stage.
template <int NTH>
static __device__ __forceinline__ void stage_h1(const A7P &P, int stg) {
  uint8_t *sm = g_dsm;
  const uint8_t *hcd = P.hcd_f;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  uint8_t *wtile = sm + ACTB + warp * TILEB;
  constexpr int tps = (LR + nwb - 1) / nwb;
  constexpr int NT = 4 * tps;
  const int ntk = (NT - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  auto tsrc = [&](int i) -> const uint8_t * {
    const int task = blockIdx.x + i * gridDim.x;
    const int s = task & 3, row = (task >> 2) * nwb + warp;
    return row < LR ? hcd + (size_t)row * 10880 + s * 2720 : nullptr;
  };
  const int s = blockIdx.x & 3;  // the grid is a multiple of 4: every task of the block has the same stream
  Q8Regs<NTH, T * D / 16, T * 80 / 4> qr;
  qr.load((const int4 *)(P.Rq + (size_t)s * T * D), (const int4 *)(P.Rd + s * T * 80));  // q8 of R*norm made by a7_ffn_rquant; activations first
  float ssv[3] = {0.f, 0.f, 0.f};
  if (warp < 4) {
#pragma unroll
    for (int j = 0; j < 3; j++) { const int u = lane + 32 * j; if (u < 80) ssv[j] = __ldcg(P.Rss + (s * T + warp) * 80 + u); }
  }
  TileRegs<1, 2720> regs;  // then the first weight tile
  if (ntk > 0) { const uint8_t *p = tsrc(0); if (p) regs.load(p, 0, lane); }
  qr.store(sm);
  if (warp < 4) {  // review 10-16 m-1: rr written before the barrier, so the arriving warp's fence in la_arrive also orders it
    float v = 0.f;
    v += ssv[0]; v += ssv[1]; v += ssv[2];
    v = warp_sum(v);
    if (lane == 0) P.rr[warp * 4 + s] = rsqrtf(v / (float)D + P.eps);
  }
  __syncthreads();
  for (int i = 0; i < ntk; i++) {
    const int task = blockIdx.x + i * gridDim.x;
    const int rg = task >> 2;
    const int row = rg * nwb + warp;
    if (row < LR) {
      __syncwarp();
      regs.store(wtile, lane);
      __syncwarp();
      if (i + 1 < ntk) { const uint8_t *p = tsrc(i + 1); if (p) regs.load(p, 0, lane); }
      float acc[1][T] = {{0.f, 0.f, 0.f, 0.f}};
      q8_tile_dot<1, 80>(wtile, A7_XQ(sm), A7_XD(sm), 80, 0, lane, acc);
      float v[4] = {acc[0][0], acc[0][1], acc[0][2], acc[0][3]};
      const float r = rscatter<4>(v, lane);
      if (lane < 4) P.part[(lane * 4 + s) * LR + row] = r;
      if (la_arrive(P.cnt + 1024 + stg * 16 + (row >> 5), 128u, lane)) h1_lo(P, row >> 5, lane);
    }
  }
}

// --balance 2 (72 SMs): stage_h1 with the 320 rows of each stream dealt out evenly over the blocks of the stream. Block (bs = blockIdx.x >> 2, stream s = blockIdx.x & 3)
// takes the rows row = bs + j*NBS (NBS = G/4, row < LR), the j-th row on warp j % nwb (the warp's m-th row is j = warp + m*nwb); the next row of the same warp is
// prefetched. One (row, stream) is still processed exactly once, so the 128 arrivals per 32-row chunk are unchanged. The activations / rr part is as stage_h1.
template <int NTH>
static __device__ __forceinline__ void stage_h1_b2(const A7P &P, int stg) {
  uint8_t *sm = g_dsm;
  const uint8_t *hcd = P.hcd_f;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  uint8_t *wtile = sm + ACTB + warp * TILEB;
  const int s = blockIdx.x & 3;  // the grid is a multiple of 4
  const int bs = (int)(blockIdx.x >> 2), NBS = (int)gridDim.x >> 2;
  const int nrow = bs < LR ? (LR - bs + NBS - 1) / NBS : 0;         // rows of this block
  const int nm = warp < nrow ? (nrow - warp + nwb - 1) / nwb : 0;   // rows of this warp
  auto rsrc = [&](int m) -> const uint8_t * {
    const int row = bs + (warp + m * nwb) * NBS;
    return hcd + (size_t)row * 10880 + s * 2720;
  };
  Q8Regs<NTH, T * D / 16, T * 80 / 4> qr;
  qr.load((const int4 *)(P.Rq + (size_t)s * T * D), (const int4 *)(P.Rd + s * T * 80));  // q8 of R*norm made by a7_ffn_rquant; activations first
  float ssv[3] = {0.f, 0.f, 0.f};
  if (warp < 4) {
#pragma unroll
    for (int j = 0; j < 3; j++) { const int u = lane + 32 * j; if (u < 80) ssv[j] = __ldcg(P.Rss + (s * T + warp) * 80 + u); }
  }
  TileRegs<1, 2720> regs;  // then the first weight tile
  if (nm > 0) regs.load(rsrc(0), 0, lane);
  qr.store(sm);
  if (warp < 4) {  // review 10-16 m-1: rr written before the barrier, so the arriving warp's fence in la_arrive also orders it
    float v = 0.f;
    v += ssv[0]; v += ssv[1]; v += ssv[2];
    v = warp_sum(v);
    if (lane == 0) P.rr[warp * 4 + s] = rsqrtf(v / (float)D + P.eps);
  }
  __syncthreads();
  for (int m = 0; m < nm; m++) {
    const int row = bs + (warp + m * nwb) * NBS;
    __syncwarp();
    regs.store(wtile, lane);
    __syncwarp();
    if (m + 1 < nm) regs.load(rsrc(m + 1), 0, lane);
    float acc[1][T] = {{0.f, 0.f, 0.f, 0.f}};
    q8_tile_dot<1, 80>(wtile, A7_XQ(sm), A7_XD(sm), 80, 0, lane, acc);
    float v[4] = {acc[0][0], acc[0][1], acc[0][2], acc[0][3]};
    const float r = rscatter<4>(v, lane);
    if (lane < 4) P.part[(lane * 4 + s) * LR + row] = r;
    if (la_arrive(P.cnt + 1024 + stg * 16 + (row >> 5), 128u, lane)) h1_lo(P, row >> 5, lane);
  }
}

// ------------------------------------------------------------------ hc2f: hc_up on lo (q8 from stage 1), mixed = 1/4 sum_s xn_s*sigmoid(gate_s)
// A block owns chunks of 32 output columns (8 warps x 4 columns); after a chunk the block quantizes mixed (q8) for the expert stages.
template <int NTH>
static __device__ __forceinline__ void stage_h2(const A7P &P, int stg) {
  uint8_t *sm = g_dsm; float *scr = g_scr;
  const float *norm = P.norm_f;
  const uint8_t *hcu = P.hcu_f;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  uint8_t *wtile = sm + ACTB + warp * TILEB;
  float *mx_s = scr;  // [T][32]
  if (stg == 1 && blockIdx.x == 0 && threadIdx.x == 0) P.cnt[3001] = 0;  // P2 (hc2a): ticket counter of the next stage (projection), as gate0
  const int ncj = (NCH - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int gcount = warp < 8 ? (8 - warp + nwb - 1) / nwb : 0;
  const int nitems = ncj * gcount * 2;
  auto isrc = [&](int it) -> const uint8_t * {
    const int m = it >> 1, h = it & 1;
    const int cj = m / gcount, gi = m - cj * gcount;
    const int ig = (blockIdx.x + cj * gridDim.x) * 8 + warp + nwb * gi;
    return hcu + ((size_t)(2 * h) * D + 4 * ig) * 340;
  };
  TileRegs<2, 1360> regs;
  {
    Q8Regs<NTH, T * LR / 16, T * 10 / 4> qr;
    qr.load((const int4 *)P.loq, (const int4 *)P.lod);  // lo (q8 made by the previous stage) first, then the first weight tile
    if (nitems > 0) regs.load(isrc(0), (size_t)D * 340, lane);
    qr.store(sm);
  }
  __syncthreads();
  const int4 *xq = A7_XQ(sm);
  const float *xd = A7_XD(sm);
  int it = 0;
  for (int cj = 0; cj < ncj; cj++) {
    const int c = blockIdx.x + cj * gridDim.x;
    for (int gi = 0; gi < gcount; gi++) {
      const int gl = warp + nwb * gi, ig = c * 8 + gl;
      float macc[T] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
      for (int h = 0; h < 2; h++, it++) {  // two halves (planes 2h, 2h+1) per 4-column group
        __syncwarp();
        regs.store(wtile, lane);
        __syncwarp();
        if (it + 1 < nitems) regs.load(isrc(it + 1), (size_t)D * 340, lane);
        const int r = lane >> 2, q = lane & 3, sp = r >> 2, il = r & 3;
        float acc[T] = {0.f, 0.f, 0.f, 0.f};
        const uint8_t *rowp = wtile + r * 340;
#pragma unroll
        for (int k = 0; k < 3; k++) {
          const int u = q + 4 * k;
          if (u < 10) q8_unit(rowp + u * 34, xq, xd, 10, u, acc);
        }
#pragma unroll
        for (int t = 0; t < T; t++) {
          acc[t] += __shfl_xor_sync(A7_FULL, acc[t], 1);
          acc[t] += __shfl_xor_sync(A7_FULL, acc[t], 2);
        }
        const int s = 2 * h + sp, i = 4 * ig + il;
#pragma unroll
        for (int t = 0; t < T; t++) {
          const float xn = P.R[(size_t)(t * 4 + s) * D + i] * P.rr[t * 4 + s] * norm[s * D + i];
          float term = xn * fsigmoid(acc[t]);
          term += __shfl_xor_sync(A7_FULL, term, 16);
          macc[t] += term;
        }
      }
      if ((lane & 0x13) == 0) {
        const int il = (lane >> 2) & 3;
#pragma unroll
        for (int t = 0; t < T; t++) {
          const float v = macc[t] * 0.25f;
          P.mixed[t * D + 4 * ig + il] = v;
          mx_s[t * 32 + gl * 4 + il] = v;
        }
      }
    }
    __syncthreads();
    for (int t = warp; t < T; t += nwb) {
      int8_t q; float d;
      q8_warp(mx_s[t * 32 + lane], q, d);
      P.mq[t * D + c * 32 + lane] = q;
      if (lane == 0) P.md[t * 80 + c] = d;
    }
    __syncthreads();
  }
  (void)stg;
}

// --balance 2 (72 SMs): stage_h2 with the 640 units (chunk c, group gl; u = c*8 + gl; 4 columns x 2 halves, the work of one warp for one chunk in stage_h2) dealt out
// evenly: unit u goes to block u % G (u = blockIdx.x + k*G), the block's k-th unit to warp k % nwb (the warp's m-th unit is k = warp + m*nwb). Same arithmetic and
// P.mixed stores as stage_h2; mx_s is not used. The chunk is quantized by its last arriver: every unit writer warp fences its P.mixed stores and adds 1 to
// P.h2cnt[(hc2f: 1)*NCH + c]; the warp that gets 7 (= the 8th unit) fences again, reads the chunk's 4 tokens x 32 columns of P.mixed with __ldcg,
// writes mq / md as stage_h2 does and resets the counter. The next stage reads mq / md after the kernel boundary.
template <int NTH>
static __device__ __forceinline__ void stage_h2_b2(const A7P &P, int stg) {
  uint8_t *sm = g_dsm;
  const float *norm = P.norm_f;
  const uint8_t *hcu = P.hcu_f;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  uint8_t *wtile = sm + ACTB + warp * TILEB;
  constexpr int NUN = NCH * 8;
  if (stg == 1 && blockIdx.x == 0 && threadIdx.x == 0) P.cnt[3001] = 0;  // P2 (hc2a): ticket counter of the next stage (projection), as gate0
  const int G = (int)gridDim.x;
  const int nk = NUN > (int)blockIdx.x ? (NUN - (int)blockIdx.x + G - 1) / G : 0;  // units of this block
  const int nm = warp < nk ? (nk - warp + nwb - 1) / nwb : 0;                       // units of this warp
  const int nitems = nm * 2;
  auto isrc = [&](int it) -> const uint8_t * {
    const int m = it >> 1, h = it & 1;
    const int ig = (int)blockIdx.x + (warp + m * nwb) * G;
    return hcu + ((size_t)(2 * h) * D + 4 * ig) * 340;
  };
  TileRegs<2, 1360> regs;
  {
    Q8Regs<NTH, T * LR / 16, T * 10 / 4> qr;
    qr.load((const int4 *)P.loq, (const int4 *)P.lod);  // lo (q8 made by the previous stage) first, then the first weight tile
    if (nitems > 0) regs.load(isrc(0), (size_t)D * 340, lane);
    qr.store(sm);
  }
  __syncthreads();
  const int4 *xq = A7_XQ(sm);
  const float *xd = A7_XD(sm);
  int *hc = P.h2cnt + (stg == 1 ? 0 : 1) * NCH;
  int it = 0;
  for (int m = 0; m < nm; m++) {
    const int ig = (int)blockIdx.x + (warp + m * nwb) * G;  // = u = c*8 + gl
    const int c = ig >> 3;
    float macc[T] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int h = 0; h < 2; h++, it++) {  // two halves (planes 2h, 2h+1) per 4-column group
      __syncwarp();
      regs.store(wtile, lane);
      __syncwarp();
      if (it + 1 < nitems) regs.load(isrc(it + 1), (size_t)D * 340, lane);
      const int r = lane >> 2, q = lane & 3, sp = r >> 2, il = r & 3;
      float acc[T] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t *rowp = wtile + r * 340;
#pragma unroll
      for (int k = 0; k < 3; k++) {
        const int u = q + 4 * k;
        if (u < 10) q8_unit(rowp + u * 34, xq, xd, 10, u, acc);
      }
#pragma unroll
      for (int t = 0; t < T; t++) {
        acc[t] += __shfl_xor_sync(A7_FULL, acc[t], 1);
        acc[t] += __shfl_xor_sync(A7_FULL, acc[t], 2);
      }
      const int s = 2 * h + sp, i = 4 * ig + il;
#pragma unroll
      for (int t = 0; t < T; t++) {
        const float xn = P.R[(size_t)(t * 4 + s) * D + i] * P.rr[t * 4 + s] * norm[s * D + i];
        float term = xn * fsigmoid(acc[t]);
        term += __shfl_xor_sync(A7_FULL, term, 16);
        macc[t] += term;
      }
    }
    if ((lane & 0x13) == 0) {
      const int il = (lane >> 2) & 3;
#pragma unroll
      for (int t = 0; t < T; t++) P.mixed[t * D + 4 * ig + il] = macc[t] * 0.25f;
    }
    // last arriver of the chunk (8 units): fence the writes, count, the 8th warp quantizes the whole chunk
    __threadfence();
    __syncwarp();
    int old = 0;
    if (lane == 0) old = atomicAdd(&hc[c], 1);
    old = __shfl_sync(A7_FULL, old, 0);
    if (old == 7) {
      __threadfence();
#pragma unroll
      for (int t = 0; t < T; t++) {
        int8_t q; float d;
        q8_warp(__ldcg(P.mixed + t * D + c * 32 + lane), q, d);
        P.mq[t * D + c * 32 + lane] = q;
        if (lane == 0) P.md[t * 80 + c] = d;
      }
      if (lane == 0) hc[c] = 0;
    }
  }
}

// ------------------------------------------------------------------ router logits, shared-expert gate, ffn inject partials
// P.cnt[A7_CNT_ROUTE]: block-level last-arriver counter of stage_router (arrivals of the grid's blocks). The other users of P.cnt: [1104 .. 1113] =
// 1024 + 5 * 16 + chunk (hc1f, stage_h1 / stage_h1_b2 with stg = 5; the chunks are row >> 5 < 10) and [3000] = the gate/up ticket (reset by block 0 of this stage).
// The index 2048 is in neither range (and the other counters, h2cnt / balcnt, are separate arrays). The last arriver resets it to 0 after the route is
// decided: the number of arrivals per launch (gridDim.x = 80 or 72) is not a divisor of 2^32, so a counter that is never reset would lose its alignment
// when it wraps (after 2^32 / 80 launches, about 53 million).
constexpr int A7_CNT_ROUTE = 2048;

template <int NTH>
static __device__ __forceinline__ void stage_router(const A7P &P) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  const int gw = blockIdx.x * nwb + warp, GW = gridDim.x * nwb;
  if (blockIdx.x == 0 && threadIdx.x == 0) P.cnt[3000] = 0;  // ticket counter of the next stage (gate/up)
  for (int item = gw; item < NEXP + 1 + 16; item += GW) {
    if (item <= NEXP) {
      const float v = f32_dot4(item < NEXP ? P.router + (size_t)item * D : P.sgw, P.mixed, lane);
      if (lane < T) {
        if (item < NEXP) P.logits[lane * NEXP + item] = v;
        else P.sgl[lane] = v;
      }
    } else {
      const int idx = item - NEXP - 1;
      inject_partial(P, idx >> 2, idx & 3, lane);
    }
  }
  if (!P.route_inline) return;  // the separate kernel a7_ffn_route builds the table (the same for every block)
  // P1b: the last block to finish builds the routing table from the logits of all blocks (block level last arriver: fence, barrier, one atomicAdd, barrier)
  __threadfence();
  __syncthreads();
  unsigned *flag = (unsigned *)g_dsm;  // the dynamic shared memory is unused by this stage: [0, 16) the flag, then the work area of the build
  if (threadIdx.x == 0) {
    __threadfence();  // review P1b m-1: cumulative fence by the arriving thread right before the atomic (grid-sync form)
    const unsigned old = atomicAdd(P.cnt + A7_CNT_ROUTE, 1u);
    const bool last = (old % gridDim.x) == gridDim.x - 1u;
    if (last) atomicExch(P.cnt + A7_CNT_ROUTE, 0u);
    flag[0] = last ? 1u : 0u;
  }
  __syncthreads();
  if (flag[0] != 0u) {
    __threadfence();
    if (P.route_ref) a7_route_build_ref(P.logits, P.route, (A7RouteSmem *)(g_dsm + 16));
    else a7_route_build(P.logits, P.route, (A7RouteSmem *)(g_dsm + 16));  // all threads; the logits are read with __ldcg
  }
}

// ------------------------------------------------------------------ expert gate/up (IQ2_S) + shared expert gate/up (Q8_0)
struct Iq2g { int gl[4], gh[4]; int sc; float d; };
// decode one 32-weight group of an IQ2_S block located in shared memory (same block layout as ggml: d, qs[64], qh[8], scales[8])
static __device__ __forceinline__ void iq2s_decode_s(const uint8_t *blk, int ib, const uint2 *grid_s, Iq2g &w) {
  const int qs = get_int_b2(blk + 2, ib);
  const int signs = get_int_b2(blk + 2, 8 + ib);
  const int qh = blk[66 + ib];
  w.sc = blk[74 + ib];
  w.d = h2f(blk);
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint2 g = grid_s[((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300)];
    const int s = (signs >> (8 * l)) & 0xFF;
    const int sp0 = ((s & 0xF) * 0x00204081) & 0x01010101;
    const int sp1 = ((s >> 4) * 0x00204081) & 0x01010101;
    w.gl[l] = (int(g.x) ^ (sp0 * 0xFF)) + sp0;
    w.gh[l] = (int(g.y) ^ (sp1 * 0xFF)) + sp1;
  }
}
static __device__ __forceinline__ int iq2s_sumi(const Iq2g &w, const int4 a0, const int4 a1) {
  int s0 = 0, s1 = 0;
  s0 = __dp4a(w.gl[0], a0.x, s0); s0 = __dp4a(w.gh[0], a0.y, s0);
  s0 = __dp4a(w.gl[1], a0.z, s0); s0 = __dp4a(w.gh[1], a0.w, s0);
  s1 = __dp4a(w.gl[2], a1.x, s1); s1 = __dp4a(w.gh[2], a1.y, s1);
  s1 = __dp4a(w.gl[3], a1.z, s1); s1 = __dp4a(w.gh[3], a1.w, s1);
  return (s0 * (w.sc & 0x0F) + s1 * (w.sc >> 4) + (s0 + s1) / 2) / 4;
}
// 2 rows of gate + 2 rows of up of one expert (each 1640 B contiguous) in registers: idx < 205 gate, 205 <= idx < 410 up
struct Iq2Regs {
  uint2 v[13];
  __device__ __forceinline__ void load(const uint8_t *g, const uint8_t *u, int lane) {
#pragma unroll
    for (int j = 0; j < 13; j++) {
      const int idx = lane + 32 * j;
      if (idx < 205) v[j] = __ldcs(((const uint2 *)g) + idx);
      else if (idx < 410) v[j] = __ldcs(((const uint2 *)u) + (idx - 205));
    }
  }
  __device__ __forceinline__ void store(uint8_t *dst, int lane) {
#pragma unroll
    for (int j = 0; j < 13; j++) {
      const int idx = lane + 32 * j;
      if (idx < 410) ((uint2 *)dst)[idx] = v[j];
    }
  }
  // one Q8_0 row of the shared expert (2720 B = 340 uint2)
  __device__ __forceinline__ void load_row(const uint8_t *p, int lane) {
#pragma unroll
    for (int j = 0; j < 11; j++) {
      const int idx = lane + 32 * j;
      if (idx < 340) v[j] = __ldcs(((const uint2 *)p) + idx);
    }
  }
  __device__ __forceinline__ void store_row(uint8_t *dst, int lane) {
#pragma unroll
    for (int j = 0; j < 11; j++) {
      const int idx = lane + 32 * j;
      if (idx < 340) ((uint2 *)dst)[idx] = v[j];
    }
  }
};

template <int NTH>
static __device__ __forceinline__ void stage_gateup(const A7P &P) {
  uint8_t *sm = g_dsm;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  const int gw = blockIdx.x * nwb + warp, GW = gridDim.x * nwb;
  uint8_t *wtile = sm + ACTB + warp * TILEB;
  uint2 *grid_s = (uint2 *)(sm + GRIDOFF);
  Q8Regs<NTH, T * D / 16, T * 80 / 4> qr;
  qr.load((const int4 *)P.mq, (const int4 *)P.md);
  load_route<NTH>(P);           // routing table -> shared memory
  const int N = g_rt.N;
  const int NI = N * 320;       // IQ2_S warp tasks: (expert n, row pair tp)
  const int NTOT = NI + FF;     // + one task per shared-expert row (gate and up row)
  unsigned *ctr = P.cnt + 3000;
  Iq2Regs regs;
  auto load_task = [&](int task, int pass) {
    if (task < NI) {
      const int n = task / 320, tp = task - n * 320;
      const size_t off = (size_t)g_rt.uni[n] * EXP_GU_BYTES + (size_t)tp * 2 * IQ2ROW;
      regs.load(P.eg + off, P.eu + off, lane);
    } else {
      const int row = task - NI;
      regs.load_row((pass == 0 ? P.sg : P.su) + (size_t)row * 2720, lane);
    }
  };
  int task = gw;
  if (task < NTOT) load_task(task, 0);  // first tile in flight during the prologue
  unsigned traw = 0;
  if (lane == 0) traw = atomicAdd(ctr, 1u);  // ticket of the second task
#pragma unroll 4
  for (int i = threadIdx.x; i < 1024; i += NTH) grid_s[i] = ((const uint2 *)iq2s_grid)[i];
  qr.store(sm);
  __syncthreads();
  const int4 *xq = A7_XQ(sm);
  const float *xd = A7_XD(sm);
  while (task < NTOT) {
    const int nxt = GW + (int)__shfl_sync(A7_FULL, traw, 0);
    if (lane == 0) traw = atomicAdd(ctr, 1u);  // ticket of the task after nxt (consumed in the next iteration)
    if (task < NI) {
      const int n = task / 320, tp = task - n * 320, row0 = tp * 2;
      __syncwarp();
      regs.store(wtile, lane);
      __syncwarp();
      if (nxt < NTOT) load_task(nxt, 0);  // next task's tile in flight during the decode below
      const int mask = g_rt.umask[n];
      const int r = lane >> 4, par = (lane >> 3) & 1, ib = lane & 7;
      const uint8_t *gt = wtile + r * IQ2ROW, *ut = wtile + 1640 + r * IQ2ROW;
      float ag[T] = {0.f, 0.f, 0.f, 0.f}, au[T] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
      for (int k = 0; k < 5; k++) {
        const int m = par + 2 * k;
        Iq2g gg, gu;
        iq2s_decode_s(gt + m * 82, ib, grid_s, gg);
        iq2s_decode_s(ut + m * 82, ib, grid_s, gu);
        const int g = m * 8 + ib;
#pragma unroll
        for (int t = 0; t < T; t++) {
          if ((mask >> t) & 1) {
            const int4 a0 = xq[(t * 80 + g) * 2], a1 = xq[(t * 80 + g) * 2 + 1];
            const float dy = xd[t * 80 + g];
            ag[t] += gg.d * dy * (float)iq2s_sumi(gg, a0, a1);
            au[t] += gu.d * dy * (float)iq2s_sumi(gu, a0, a1);
          }
        }
      }
      float v8[8] = {ag[0], ag[1], ag[2], ag[3], au[0], au[1], au[2], au[3]};
      const float res = rscatter<8, 16>(v8, lane);  // sums over the 16 lanes of a row: lane (l & 7) < 4 has gate[t], (l & 7) >= 4 has up[t - 4]
      const float upv = __shfl_down_sync(A7_FULL, res, 4);
      if ((lane & 15) < 4 && ((mask >> (lane & 3)) & 1)) P.hbuf[(size_t)(n * 4 + (lane & 3)) * FF + row0 + r] = fsilu(res) * upv;
    } else {  // shared expert row (Q8_0, all tokens): gate row, then up row
      const int row = task - NI;
      float vg[4] = {0.f, 0.f, 0.f, 0.f}, vu[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
      for (int pass = 0; pass < 2; pass++) {
        float acc[1][T] = {{0.f, 0.f, 0.f, 0.f}};
        __syncwarp();
        regs.store_row(wtile, lane);
        __syncwarp();
        if (pass == 0) load_task(task, 1);
        else if (nxt < NTOT) load_task(nxt, 0);
        q8_tile_dot<1, 80>(wtile, xq, xd, 80, 0, lane, acc);
        if (pass == 0) { vg[0] = acc[0][0]; vg[1] = acc[0][1]; vg[2] = acc[0][2]; vg[3] = acc[0][3]; }
        else { vu[0] = acc[0][0]; vu[1] = acc[0][1]; vu[2] = acc[0][2]; vu[3] = acc[0][3]; }
      }
      const float gs = rscatter<4>(vg, lane), us = rscatter<4>(vu, lane);
      if (lane < T) P.hbuf[(size_t)(N * 4 + lane) * FF + row] = fsilu(gs) * us;
    }
    task = nxt;
  }
}

// ------------------------------------------------------------------ q8 of h (gate/up output) per (expert, token) member
// units (member pair, 32-element chunk of h): up to 44 pairs x 20 chunks over the warps.
template <int NTH>
static __device__ __forceinline__ void stage_qh(const A7P &P) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  const int gw = blockIdx.x * nwb + warp, GW = gridDim.x * nwb;
  load_route<NTH>(P);
  const int NU = g_rt.pref[g_rt.N + 1] * 20;
#pragma unroll 1
  for (int base = 0; base < NU; base += 2 * GW) {  // 2 units per warp and round (one round at >= 256 threads x 1 block/SM on 80 SMs)
    float hv[2];
    int idxs[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int u = base + gw + i * GW;
      hv[i] = 0.f; idxs[i] = -1;
      if (u < NU) {
        const int p = u / 20, ch = u - p * 20;
        const int n = g_rt.pn[p], t = g_rt.pt[p];
        idxs[i] = u;  // pair-major: hq[p][640], hd[p][20]
        hv[i] = __ldcg(P.hbuf + (size_t)(n * 4 + t) * FF + ch * 32 + lane);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      if (idxs[i] >= 0) {
        int8_t q; float d;
        q8_warp(hv[i], q, d);
        P.hq[(size_t)idxs[i] * 32 + lane] = q;
        if (lane == 0) P.hd[idxs[i]] = d;
      }
    }
  }
}

// A block owns chunks of 32 consecutive rows (output columns) of R. rquant_chunk: the new residual values of the chunk (Rn_s[(t*4+s)*32 + row], smem) are
// quantized as R*norm_next (q8, the activation of the next layer's hc_mix) and their sums of squares are stored (for the rms); layout per stream:
// Rq[(s*T+t)*D + i], Rd[(s*T+t)*80 + chunk], Rss[(s*T+t)*80 + chunk]. Block-wide: call between two __syncthreads. (Work only in P1: nothing reads it.)
template <int NTH>
static __device__ __forceinline__ void rquant_chunk(const A7P &P, int c, const float *Rn_s, int warp, int lane) {
  constexpr int nwb = NTH / 32;
#pragma unroll 1
  for (int p = warp; p < 16; p += nwb) {
    const int t = p >> 2, s = p & 3;
    const float x = Rn_s[p * 32 + lane];
    int8_t q; float d;
    q8_warp(x * P.norm_next[s * D + c * 32 + lane], q, d);
    const float ss = warp_sum(x * x);
    P.Rq[(size_t)(s * T + t) * D + c * 32 + lane] = q;
    if (lane == 0) { P.Rd[(s * T + t) * 80 + c] = d; P.Rss[(s * T + t) * 80 + c] = ss; }
  }
}

// BAL >= 1: end of the leftover-chunk work of stage_down. The block has written (global P.Rout) its units of the leftover chunks c = base*G + u/upc,
// units u = blockIdx.x + j*G (j < nxu, increasing, so the units of one chunk are consecutive in j). Per chunk: the block adds the number of its units to
// P.balcnt[c]; the block that completes the chunk's `upc` units is the last arriver and quantizes the chunk (rquant_chunk) from the Rout in global memory.
// Order: every writer thread fences its stores, block barrier, thread 0 adds (atomic) and shares the verdict through smem (*flag); the last arriver fences
// again, reads the 32 rows x 16 (t,s) of Rout with __ldcg (L2) into Rn_s, quantizes, and resets the counter.
// Block-wide: all threads of the block call it (uniform loop).
template <int NTH>
static __device__ __forceinline__ void bal_finish(const A7P &P, int base, int nxu, int upc, float *Rn_s, int *flag) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int G = (int)gridDim.x;
  __threadfence();
  __syncthreads();
  int j = 0;
  while (j < nxu) {
    const int cx = ((int)blockIdx.x + j * G) / upc;
    int j2 = j + 1;
    while (j2 < nxu && ((int)blockIdx.x + j2 * G) / upc == cx) j2++;
    const int c = base * G + cx, nu = j2 - j;
    if (threadIdx.x == 0) {
      const int old = (int)atomicAdd((unsigned *)&P.balcnt[c], (unsigned)nu);
      *flag = (old + nu == upc) ? 1 : 0;
    }
    __syncthreads();
    if (*flag) {
      __threadfence();
      for (int idx = threadIdx.x; idx < 16 * 32; idx += NTH) Rn_s[idx] = __ldcg(P.Rout + (size_t)(idx >> 5) * D + c * 32 + (idx & 31));
      __syncthreads();
      rquant_chunk<NTH>(P, c, Rn_s, warp, lane);
      __syncthreads();
      if (threadIdx.x == 0) P.balcnt[c] = 0;
    }
    __syncthreads();  // the next chunk's verdict must not overwrite *flag while it is still being read
    j = j2;
  }
}

// ------------------------------------------------------------------ down (IQ4_NL experts + Q8_0 shared expert), hc_combine(ffn)
// Work items (task j of this block = 8 output rows, expert n) are dealt round-robin to the warps, tiles prefetched one item ahead, member activations
// of all (expert, token) pairs staged in smem; partial sums per (task, warp) in smem, reduced once at the end (no block barrier between tasks).
// The new residual R_out = R + f * 2*sigmoid(inject / 4) is written to P.Rout (R is read only). With P.norm_next != nullptr the stage also makes the q8
// of R_out * norm_next (rquant_chunk, as gate0's down stage does for the next layer): work only, the next op of P1 does not read it.
// BAL >= 1 (72 SMs): the NCH % G leftover chunks are dealt out as tasks of 8 rows to all blocks.
template <int NTH, int BAL>
static __device__ __forceinline__ void stage_down(const A7P &P) {
  uint8_t *sm = g_dsm; float *scr = g_scr;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  uint8_t *wtile = sm + ACTB + warp * TILEB;
  load_route<NTH>(P);
  const int N = g_rt.N;
  float *wn_s = scr;                   // [(N+1)*4]  (shared expert: sigmoid gate)
  float *inj_s = scr + 312;            // [16]
  float *part_s = scr + 384;           // [8][nwb][32]
  float *Rn_s = scr + 3456;            // [16][32]
  const bool qnext = P.norm_next != nullptr;  // kernel argument: uniform over the grid
  uint8_t *aq = sm;                  // pair p: int8 [640] at p*640
  float *ad = (float *)(sm + 28160);   // pair p: float [20] at p*20
  const int G = (int)gridDim.x;
  // BAL >= 1: the block owns base = NCH / G chunks; the rem leftover chunks (c >= base*G) are dealt out as tasks (8 rows): unit u (0 .. rem*4-1) goes to
  // block u % G, the block's jx-th unit is u = blockIdx.x + jx*G, rows base*G*32 + u*8 ... +7. BAL == 0: rem = 0, the original deal.
  const int base = NCH / G;
  const int rem = BAL >= 1 ? NCH - base * G : 0;
  const int nch = BAL >= 1 ? base : (NCH - (int)blockIdx.x + G - 1) / G;  // chunks of 32 output rows owned by this block
  const int ntask = 4 * nch;                                           // tasks of 8 rows (own chunks)
  const int nxt = rem * 4 > (int)blockIdx.x ? (rem * 4 - (int)blockIdx.x + G - 1) / G : 0;  // leftover tasks of this block
  auto row0_of = [&](int j) -> int {
    if (j >= ntask) return base * G * 32 + ((int)blockIdx.x + (j - ntask) * G) * 8;  // leftover task
    return ((int)blockIdx.x + (j >> 2) * (int)gridDim.x) * 32 + (j & 3) * 8;
  };
  const int items = (ntask + nxt) * (N + 1);
  TileRegs<1, 2880> regs;
  auto isrc = [&](int i) -> const uint8_t * {  // IQ4_NL tile of item i, nullptr for the shared-expert item
    const int j = i / (N + 1), n = i - j * (N + 1);
    if (n >= N) return nullptr;
    return P.ed + (size_t)g_rt.uni[n] * EXP_D_BYTES + (size_t)row0_of(j) * IQ4ROW;
  };
  // prologue: loads of the small inputs first (member activations, expert weights, inject), then the first weight tile, then the smem stores
  constexpr int NAV = (44 * 45 + NTH - 1) / NTH;
  const int tot = g_rt.pref[N + 1];
  int4 av[NAV];
#pragma unroll
  for (int j = 0; j < NAV; j++) {
    const int idx = threadIdx.x + j * NTH;
    if (idx < tot * 45) av[j] = idx < tot * 40 ? __ldcg((const int4 *)P.hq + idx) : __ldcg((const int4 *)P.hd + (idx - tot * 40));  // pair-major: plain copies
  }
  for (int idx = threadIdx.x; idx < (N + 1) * 4; idx += NTH) {
    if (idx < N * 4) { const int k = g_rt.wsel[idx]; wn_s[idx] = k >= 0 ? g_rt.wk[k] : 0.f; }
    else wn_s[idx] = fsigmoid(__ldcg(P.sgl + (idx - N * 4)));
  }
  if (threadIdx.x < 16) {
    const int t = threadIdx.x >> 2, s = threadIdx.x & 3;
    float v = P.rr[t * 4 + 0] * P.injp[(t * 4 + 0) * 4 + s];
    v += P.rr[t * 4 + 1] * P.injp[(t * 4 + 1) * 4 + s];
    v += P.rr[t * 4 + 2] * P.injp[(t * 4 + 2) * 4 + s];
    v += P.rr[t * 4 + 3] * P.injp[(t * 4 + 3) * 4 + s];
    inj_s[threadIdx.x] = 2.f * fsigmoid(v * 0.25f);  // combine gain
  }
  if (warp < items) { const uint8_t *s0 = isrc(warp); if (s0) regs.load(s0, 0, lane); }
  for (int i = threadIdx.x; i < 8 * nwb * 32; i += NTH) part_s[i] = 0.f;
#pragma unroll
  for (int j = 0; j < NAV; j++) {
    const int idx = threadIdx.x + j * NTH;
    if (idx < tot * 45) {
      if (idx < tot * 40) ((int4 *)aq)[idx] = av[j];
      else ((int4 *)ad)[idx - tot * 40] = av[j];
    }
  }
  __syncthreads();
  for (int i = warp; i < items; i += nwb) {
    const int j = i / (N + 1), n = i - j * (N + 1);
    const int row0 = row0_of(j);
    const int pb = g_rt.pref[n];
    const int mask = n < N ? g_rt.umask[n] : 15;
    float *ps = part_s + (j * nwb + warp) * 32;
    if (n < N) {
      __syncwarp();
      regs.store(wtile, lane);
      __syncwarp();
      if (i + nwb < items) { const uint8_t *s1 = isrc(i + nwb); if (s1) regs.load(s1, 0, lane); }
      const int rr = lane >> 2, q = lane & 3;
      int pi[T];
#pragma unroll
      for (int t = 0; t < T; t++) pi[t] = pb + __popc(mask & ((1 << t) - 1));
      float tmp[T] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
      for (int k = 0; k < 5; k++) {
        const int u = q * 5 + k;
        const uint8_t *b = wtile + rr * IQ4ROW + u * 18;
        int w4[4];
        const float d = unit_words<4>(b, w4);
        int lo[4], hi[4];
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
          const int2 qq = table16(w4[ii]);
          lo[ii] = qq.x; hi[ii] = qq.y;
        }
#pragma unroll
        for (int t = 0; t < T; t++) {
          if ((mask >> t) & 1) {
            const int4 *base = (const int4 *)(aq + pi[t] * 640 + u * 32);
            const int4 a0 = base[0], a1 = base[1];
            const float dy = ad[pi[t] * 20 + u];
            int s = 0;
            s = __dp4a(lo[0], a0.x, s); s = __dp4a(lo[1], a0.y, s); s = __dp4a(lo[2], a0.z, s); s = __dp4a(lo[3], a0.w, s);
            s = __dp4a(hi[0], a1.x, s); s = __dp4a(hi[1], a1.y, s); s = __dp4a(hi[2], a1.z, s); s = __dp4a(hi[3], a1.w, s);
            tmp[t] += d * dy * (float)s;
          }
        }
      }
      const float v = rscatter<4, 4>(tmp, lane);  // lane q = token q's total over the 4 lanes of the row
      if ((mask >> q) & 1) ps[rr * 4 + q] += wn_s[n * 4 + q] * v;
    } else {  // shared expert down, Q8_0 (rows of 20 units x 34 B), two halves of 4 rows
      if (i + nwb < items) { const uint8_t *s1 = isrc(i + nwb); if (s1) regs.load(s1, 0, lane); }
#pragma unroll 1
      for (int h2 = 0; h2 < 2; h2++) {
        stage_tile<1, 2720>(wtile, P.sd + (size_t)(row0 + 4 * h2) * 680, 0, lane);
        const int rr = lane >> 3, q = lane & 7;
        float tmp[T] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 3; k++) {
          const int u = q + 8 * k;
          if (u < 20) {
            const uint8_t *b = wtile + rr * 680 + u * 34;
            int wv[8];
            const float d = unit_words<8>(b, wv);
#pragma unroll
            for (int t = 0; t < T; t++) {
              const int4 *base = (const int4 *)(aq + (pb + t) * 640 + u * 32);
              const int4 a0 = base[0], a1 = base[1];
              const float dy = ad[(pb + t) * 20 + u];
              int s = 0;
              s = __dp4a(wv[0], a0.x, s); s = __dp4a(wv[1], a0.y, s); s = __dp4a(wv[2], a0.z, s); s = __dp4a(wv[3], a0.w, s);
              s = __dp4a(wv[4], a1.x, s); s = __dp4a(wv[5], a1.y, s); s = __dp4a(wv[6], a1.z, s); s = __dp4a(wv[7], a1.w, s);
              tmp[t] += d * dy * (float)s;
            }
          }
        }
        const float v = rscatter<4, 8>(tmp, lane);  // lane q < 4 = token q's total over the 8 lanes of the row
        if (q < 4) ps[(h2 * 4 + rr) * 4 + q] += wn_s[N * 4 + q] * v;
      }
    }
  }
  __syncthreads();
  for (int k = 0; k < nch; k++) {
    const int c = blockIdx.x + k * gridDim.x;
    for (int idx = threadIdx.x; idx < 4 * 32; idx += NTH) {
      const int jl = idx >> 5, rr = (idx >> 2) & 7, t = idx & 3;
      const int j = k * 4 + jl;
      float f = 0.f;
      for (int ww = 0; ww < nwb; ww++) f += part_s[(j * nwb + ww) * 32 + rr * 4 + t];
      const int i = c * 32 + jl * 8 + rr;
#pragma unroll
      for (int s = 0; s < 4; s++) {
        const float rn = P.R[(size_t)(t * 4 + s) * D + i] + f * inj_s[t * 4 + s];
        P.Rout[(size_t)(t * 4 + s) * D + i] = rn;
        Rn_s[(t * 4 + s) * 32 + jl * 8 + rr] = rn;
      }
    }
    if (qnext) {
      __syncthreads();
      rquant_chunk<NTH>(P, c, Rn_s, warp, lane);
      __syncthreads();
    }
  }
  if (rem > 0) {  // leftover tasks: f of the 8 rows x 4 tokens (same sum over the warps' part_s), Rout to global, then the last arriver quantizes the chunk
    for (int jx = 0; jx < nxt; jx++) {
      const int j = ntask + jx;
      if (threadIdx.x < 32) {
        const int rr = threadIdx.x >> 2, t = threadIdx.x & 3;
        float f = 0.f;
        for (int ww = 0; ww < nwb; ww++) f += part_s[(j * nwb + ww) * 32 + rr * 4 + t];
        const int i = row0_of(j) + rr;
#pragma unroll
        for (int s = 0; s < 4; s++) P.Rout[(size_t)(t * 4 + s) * D + i] = P.R[(size_t)(t * 4 + s) * D + i] + f * inj_s[t * 4 + s];
      }
    }
    if (qnext) bal_finish<NTH>(P, base, nxt, 4, Rn_s, (int *)(scr + 3456 + 512));  // all blocks (block-uniform: nxt may be 0)
  }
}

// ================================================================== P2: the attn half (GGML_OP_A7_GDN_ATTN), ported from gate0.cu
// hc1a / hc2a are stage_h1 / stage_h2 (and the _b2 variants) with stg = 0 / 1; proj, gdn and ssmout follow. Differences from gate0 (the 2 sections of SPEC-a7-p2):
//   - the conv state row is read from r_l at row s_copy[0] with the production layout (channel c: elements c * 3 + j, j = 0 oldest) and the 4 frames
//     (slot s: columns 4 - s .. 6 - s of [state(3) | new tokens(4)]) are written to rows s * mem_size + kv_head (gate0: j * 10240 + c, slot 0 only);
//   - the ssm state row is read from s_l at row s_copy[0] and the state after token t is written to slot 3 - t (gate0: Sout[t]);
//   - the q / k head of the value head h is h % 16 (gate0: h / 3);
//   - eps (l2 norm, gated norm, hc rms) is P.eps = hparams.f_norm_rms_eps (gate0: 1e-6);
//   - R is not modified: ssmout writes P.Rout (gate0: in place).
constexpr int NVH = 48, HD = 128, INNER = 6144, QKVC = 10240;
constexpr int CONVROW = 3 * QKVC;                    // n_embd_r(): floats of one conv state row
constexpr size_t SSMROW = (size_t) NVH * HD * HD;    // n_embd_s(): floats of one ssm state row: element h * 16384 + col * 128 + i

// ------------------------------------------------------------------ proj: qkv, z (Q8_0) and alpha, beta (F32) projections, conv + silu, conv state frames
// Tiles (2 rows) are handed out dynamically (atomic ticket, taken one tile ahead); the first tile of every warp is static and prefetched.
// A conv channel (tile row) is read (its 3 state values) and its 4 frames are written by the same lane group of one warp, the write after the read
// (the lanes of the warp are converged at the shuffles in between): no other warp touches the channel (SPEC-a7-p2 2.7).
template <int NTH>
static __device__ __forceinline__ void attn_proj(const A7P &P) {
  uint8_t *sm = g_dsm;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  const int gw = blockIdx.x * nwb + warp, GW = gridDim.x * nwb;
  uint8_t *wtile = sm + ACTB + warp * TILEB;
  constexpr int NQ = (QKVC + INNER) / 2;  // 8192 tiles of 2 rows
  unsigned *ctr = P.cnt + 3001;
  const float *cs0 = P.conv_all + (size_t)(unsigned)P.s_copy[0] * CONVROW;  // the state row read (2.1)
  auto src = [&](int tile, int p) -> const uint8_t * {
    const bool isq = tile < QKVC / 2;
    return (isq ? P.wqkv : P.wz) + (size_t)(isq ? tile : tile - QKVC / 2) * 2 * 2720 + p * 1360;
  };
  Q8Regs<NTH, T * D / 16, T * 80 / 4> qr;
  qr.load((const int4 *)P.mq, (const int4 *)P.md);  // activations (q8 made by hc2a) first, then the first weight tile
  TileRegs<2, 1360> regs;
  int tile = gw;
  if (tile < NQ) regs.load(src(tile, 0), 2720, lane);
  unsigned traw = 0;
  if (lane == 0) traw = atomicAdd(ctr, 1u);  // ticket of the second tile
  qr.store(sm);
  __syncthreads();
  const int4 *xq = A7_XQ(sm);
  const float *xd = A7_XD(sm);
#pragma unroll 1
  for (int g = gw; g < 12 * NVH; g += GW) {  // alpha / beta (F32) rows: one per selected warp (every 6th), more than one round if the grid has < 576 warps
    if (g % 6 != 0) continue;
    const int f = g / 6;
    const float *wf = f < NVH ? P.alpha + (size_t)f * D : P.beta + (size_t)(f - NVH) * D;
    const float v = f32_dot4(wf, P.mixed, lane);
    if (lane < T) {
      if (f < NVH) P.a_o[lane * NVH + f] = v;
      else P.b_o[lane * NVH + (f - NVH)] = v;
    }
  }
  while (tile < NQ) {
    const int nxt = GW + (int)__shfl_sync(A7_FULL, traw, 0);
    if (lane == 0) traw = atomicAdd(ctr, 1u);  // ticket of the tile after nxt (consumed in the next iteration)
    const bool isq = tile < QKVC / 2;
    float4 cw4 = make_float4(0.f, 0.f, 0.f, 0.f);  // conv taps and saved conv state of the channels of this tile (prefetched)
    float cs3[3] = {0.f, 0.f, 0.f};
    if (isq && lane < 8) {
      const int crow = tile * 2 + (lane >> 2);
      cw4 = *(const float4 *)(P.conv + (size_t)crow * 4);
#pragma unroll
      for (int j = 0; j < 3; j++) cs3[j] = cs0[(size_t)crow * 3 + j];  // production layout: channel-major, j = 0 oldest
    }
    float acc[2][T];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int t = 0; t < T; t++) acc[r][t] = 0.f;
#pragma unroll 1
    for (int p = 0; p < 2; p++) {
      __syncwarp();
      regs.store(wtile, lane);
      __syncwarp();
      const uint8_t *np = p == 0 ? src(tile, 1) : (nxt < NQ ? src(nxt, 0) : nullptr);  // next pass (or the next tile's first pass) in flight during the dot products
      if (np) regs.load(np, 2720, lane);
      q8_tile_dot<2, 40>(wtile, xq, xd, 80, p * 40, lane, acc);
    }
    float v[8];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int t = 0; t < T; t++) v[r * 4 + t] = acc[r][t];
    const float res = rscatter<8>(v, lane);
    if (isq) {  // conv (4 taps over the saved state and the 4 new tokens of the channel) + silu; lane (r, t) holds the raw value of row r, token t
      float x[4];
#pragma unroll
      for (int tp = 0; tp < 4; tp++) x[tp] = __shfl_sync(A7_FULL, res, (lane & 4) | tp);
      const int t = lane & 3;
      auto rawv = [&](int tau) -> float {
        return tau < 0 ? (tau == -3 ? cs3[0] : tau == -2 ? cs3[1] : cs3[2]) : (tau == 0 ? x[0] : tau == 1 ? x[1] : tau == 2 ? x[2] : x[3]);
      };
      float acc2 = 0.f;
      acc2 += cw4.x * rawv(t - 3);
      acc2 += cw4.y * rawv(t - 2);
      acc2 += cw4.z * rawv(t - 1);
      acc2 += cw4.w * rawv(t);
      __syncwarp();  // the frame writes below may hit the saved-state row read above by other lanes (read row == written row): order them after all reads
      if (lane < 8) {
        const int row = tile * 2 + (lane >> 2);
        P.qkv_o[t * QKVC + row] = fsilu(acc2);
        // conv state frames (2.3): comb = [state 0, 1, 2, x0, x1, x2, x3] (7 columns); the lane of token t writes slot s = t = columns 4 - s .. 6 - s
        // to row s * mem_size + kv_head, in the same channel-major layout (64-bit address)
        auto cmb = [&](int k) -> float {
          return k == 0 ? cs3[0] : k == 1 ? cs3[1] : k == 2 ? cs3[2] : k == 3 ? x[0] : k == 4 ? x[1] : k == 5 ? x[2] : x[3];
        };
        float *fr = P.conv_all + ((size_t)t * (size_t)P.mem_size + (size_t)P.kv_head) * CONVROW + (size_t)row * 3;
        fr[0] = cmb(4 - t);
        fr[1] = cmb(5 - t);
        fr[2] = cmb(6 - t);
      }
    } else if (lane < 8) {
      const int r = lane >> 2, t = lane & 3, row = (tile - QKVC / 2) * 2 + r;
      P.z_o[t * INNER + row] = res;
    }
    tile = nxt;
  }
}

// ------------------------------------------------------------------ gdn: l2 norm, delta rule (state after every token), gated-norm q8 activations
static __device__ __forceinline__ float dot4(const float4 a, const float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// A block handles up to MB chunk tasks (head h, 32 columns) as one batch: all loads of the batch are issued together (conv'ed q/k/v, gates, z, then the state
// columns), then l2 norm, delta rule per task (warp = 4 columns; the state of the next task is prefetched), and the gated-norm q8 epilogue. The convolution
// + silu was done by the projection stage. State: column (h, col) (128 floats) is read from the row s_copy[0] of s_l by one warp, which then writes the
// state after token t to the slot 3 - t, row s * mem_size + kv_head (the read of the column is complete before its first write; no other warp touches it).
template <int NTH>
static __device__ __forceinline__ void attn_gdn(const A7P &P) {
  float *scr = g_scr;
  static_assert(NTH >= 128, "gdn stage needs >= 4 warps");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  constexpr int MB = 3;
  constexpr int NCT = NVH * 4;  // 192 chunk tasks
  float *qn = scr, *kn = scr + MB * 512, *vv = scr + MB * 1024, *os = scr + MB * 1152, *bet = scr + MB * 1280, *gdc = scr + MB * 1284;
  const float *Sin = P.ssm_all + (size_t)(unsigned)P.s_copy[0] * SSMROW;   // state row read (2.1, 2.4)
  float *Sw = P.ssm_all + (size_t)P.kv_head * SSMROW;                       // row kv_head of slot 0; slot s is + s * mem_size * SSMROW (2.5)
  const size_t slot_stride = (size_t)P.mem_size * SSMROW;
  const int ntk = (int)blockIdx.x < NCT ? (NCT - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  for (int j0 = 0; j0 < ntk; j0 += MB) {
    const int nb = min(MB, ntk - j0);
    // ---- loads: conv'ed q/k/v as float4 (per task and token: 32 float4 q, 32 float4 k, 8 float4 v)
    constexpr int NE4 = (MB * 288 + NTH - 1) / NTH;
    float4 xv[NE4];
#pragma unroll
    for (int j = 0; j < NE4; j++) {
      const int u = threadIdx.x + j * NTH;
      xv[j] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (u < nb * 288) {
        const int jt = u / 288, rem = u - jt * 288, t = rem / 72, e4 = rem - t * 72;
        const int id = blockIdx.x + (j0 + jt) * gridDim.x, h = id >> 2, kh = h & 15, jb = (id & 3) * 32;  // kh = h % 16 (2.6; gate0: h / 3)
        const int c = e4 < 32 ? kh * 128 + e4 * 4 : e4 < 64 ? 2048 + kh * 128 + (e4 - 32) * 4 : 4096 + h * 128 + jb + (e4 - 64) * 4;
        xv[j] = __ldcg((const float4 *)(P.qkv_o + t * QKVC + c));
      }
    }
    float gb = 0.f, ga = 0.f, gA = 0.f;
    if (threadIdx.x < nb * 4) {
      const int jt = threadIdx.x >> 2, t = threadIdx.x & 3;
      const int id = blockIdx.x + (j0 + jt) * gridDim.x, h = id >> 2;
      gb = __ldcg(P.b_o + t * NVH + h);
      ga = __ldcg(P.a_o + t * NVH + h) + P.dt[h];
      gA = P.Aa[h];
    }
    float zr[MB], sn[MB];
#pragma unroll
    for (int jt = 0; jt < MB; jt++) {
      zr[jt] = 0.f; sn[jt] = 0.f;
      if (jt < nb && warp < T) {
        const int id = blockIdx.x + (j0 + jt) * gridDim.x, h = id >> 2, jb = (id & 3) * 32;
        zr[jt] = __ldcg(P.z_o + warp * INNER + h * 128 + jb + lane);
        sn[jt] = P.snorm[jb + lane];
      }
    }
    auto ldstate = [&](int jt, float4 (&sl)[4]) {  // state columns warp*4..+3 of task jt (first pass of this warp)
      const int id = blockIdx.x + (j0 + jt) * gridDim.x, h = id >> 2, jb = (id & 3) * 32;
#pragma unroll
      for (int c = 0; c < 4; c++) sl[c] = __ldcs(((const float4 *)(Sin + ((size_t)h * 128 + jb + warp * 4 + c) * 128)) + lane);
    };
    float4 sa[4], sb[4];
    if (warp < 8) ldstate(0, sa);
    // ---- smem
#pragma unroll
    for (int j = 0; j < NE4; j++) {
      const int u = threadIdx.x + j * NTH;
      if (u < nb * 288) {
        const int jt = u / 288, rem = u - jt * 288, t = rem / 72, e4 = rem - t * 72;
        float *dst = e4 < 32 ? qn + jt * 512 + t * 128 + e4 * 4 : e4 < 64 ? kn + jt * 512 + t * 128 + (e4 - 32) * 4 : vv + jt * 128 + t * 32 + (e4 - 64) * 4;
        *(float4 *)dst = xv[j];
      }
    }
    if (threadIdx.x < nb * 4) {
      bet[threadIdx.x] = fsigmoid(gb);
      gdc[threadIdx.x] = __expf(fsoftplus(ga) * gA);
    }
    __syncthreads();
    for (int vec = warp; vec < nb * 8; vec += nwb) {  // l2 normalize (q: also times 1/sqrt(128))
      const int jt = vec >> 3, v8 = vec & 7;
      float *p = (v8 < 4 ? qn : kn) + jt * 512 + (v8 & 3) * 128;
      float ss = 0.f;
#pragma unroll
      for (int m = 0; m < 4; m++) ss += p[lane + 32 * m] * p[lane + 32 * m];
      ss = warp_sum(ss);
      float sc = rsqrtf(ss + P.eps);  // production build_gdn_l2_norm: x / sqrt(sum x^2 + eps), eps = f_norm_rms_eps
      if (v8 < 4) sc *= 0.08838834764831845f;
#pragma unroll
      for (int m = 0; m < 4; m++) p[lane + 32 * m] *= sc;
    }
    __syncthreads();
    // ---- delta rule
#pragma unroll 1
    for (int jt = 0; jt < nb; jt++) {
      const int id = blockIdx.x + (j0 + jt) * gridDim.x, h = id >> 2, jb = (id & 3) * 32;
      if (jt + 1 < nb && warp < 8) ldstate(jt + 1, sb);
      for (int cgp = warp; cgp < 8; cgp += nwb) {  // 4 columns per warp pass
        const int j0c = cgp * 4;
        float4 sl[4];
        if (cgp == warp) {
#pragma unroll
          for (int c = 0; c < 4; c++) sl[c] = sa[c];
        } else {
#pragma unroll
          for (int c = 0; c < 4; c++) sl[c] = __ldcs(((const float4 *)(Sin + ((size_t)h * 128 + jb + j0c + c) * 128)) + lane);
        }
#pragma unroll 1
        for (int t = 0; t < T; t++) {
          const float4 k4 = ((const float4 *)(kn + jt * 512 + t * 128))[lane];
          const float4 q4 = ((const float4 *)(qn + jt * 512 + t * 128))[lane];
          const float g = gdc[jt * 4 + t], be = bet[jt * 4 + t];
          float kp[4];
#pragma unroll
          for (int c = 0; c < 4; c++) {
            sl[c].x *= g; sl[c].y *= g; sl[c].z *= g; sl[c].w *= g;
            kp[c] = dot4(k4, sl[c]);
          }
          const float kr = rscatter<4>(kp, lane);  // lane l holds the total of column (l & 3)
          float op[4];
          float *Sslot = Sw + (size_t)(T - 1 - t) * slot_stride;  // the state after token t goes to slot 3 - t (2.5)
#pragma unroll
          for (int c = 0; c < 4; c++) {
            const float kv = __shfl_sync(A7_FULL, kr, c);
            const float u = vv[jt * 128 + t * 32 + j0c + c] - kv;
            const float cf = be * u;
            sl[c].x += cf * k4.x; sl[c].y += cf * k4.y; sl[c].z += cf * k4.z; sl[c].w += cf * k4.w;
            op[c] = dot4(q4, sl[c]);
            __stcs(((float4 *)(Sslot + ((size_t)h * 128 + jb + j0c + c) * 128)) + lane, sl[c]);
          }
          const float orr = rscatter<4>(op, lane);
          if (lane < 4) os[jt * 128 + t * 32 + j0c + lane] = orr;
        }
      }
      if (jt + 1 < nb) {
#pragma unroll
        for (int c = 0; c < 4; c++) sa[c] = sb[c];
      }
    }
    __syncthreads();
    // ---- gated norm input (rms scale applied later through d): w = o * ssm_norm * sigmoid(z), q8
#pragma unroll
    for (int jt = 0; jt < MB; jt++) {
      if (jt < nb) {
        const int id = blockIdx.x + (j0 + jt) * gridDim.x, h = id >> 2, ch = id & 3, jb = ch * 32;
        for (int t = warp; t < T; t += nwb) {
          const float o = os[jt * 128 + t * 32 + lane];
          const float wv = o * sn[jt] * fsigmoid(zr[jt]);
          int8_t q; float d;
          q8_warp(wv, q, d);
          P.oq[t * INNER + h * 128 + jb + lane] = q;
          const float ss = warp_sum(o * o);
          if (lane == 0) { P.od[t * 192 + h * 4 + ch] = d; P.osq[t * 192 + h * 4 + ch] = ss; }
        }
      }
    }
    __syncthreads();
  }
  {  // inject partials of the attention hc_mix (R is untouched until the next stage): 16 tasks on the last blocks (they have the fewest chunk tasks)
    const int ib = (int)blockIdx.x - ((int)gridDim.x - 16);
    if (ib >= 0 && warp == 0) inject_partial(P, ib >> 2, ib & 3, lane);
  }
}

// ------------------------------------------------------------------ ssmout: ssm_out (Q8_0) + hc_combine(attn), and the q8 of R_new * hc_ffn_norm for the ffn op
// As stage_down: the new residual is written to P.Rout (R is read only); the chunk is quantized by rquant_chunk with P.norm_next = hc_ffn_norm
// (always present for the attn op). BAL >= 1 (72 SMs): the NCH % G leftover chunks are dealt out as tiles.
template <int NTH, int BAL>
static __device__ __forceinline__ void attn_out(const A7P &P) {
  uint8_t *sm = g_dsm; float *scr = g_scr;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int nwb = NTH / 32;
  uint8_t *wtile = sm + ACTB + warp * TILEB;
  float *inj_s = scr;        // [T][4]
  float *Rn_s = scr + 64;    // [16][32]
  const int G = (int)gridDim.x;
  // BAL >= 1: the block owns base = NCH / G chunks (c = blockIdx.x + k*G, k < base); the rem leftover chunks (c >= base*G) are dealt out as tiles:
  // unit u (0 .. rem*16-1) goes to block u % G, the block's j-th unit (u = blockIdx.x + j*G) to warp j % nwb. BAL == 0: rem = 0, the original deal.
  const int base = NCH / G;
  const int rem = BAL >= 1 ? NCH - base * G : 0;
  const int nch = BAL >= 1 ? base : (NCH - (int)blockIdx.x + G - 1) / G;
  const int tpw = (16 - warp + nwb - 1) / nwb;  // tiles (2 rows) of a 32-row chunk per warp
  const int nxu = rem * 16 > (int)blockIdx.x ? (rem * 16 - (int)blockIdx.x + G - 1) / G : 0;  // leftover tiles of this block
  const int nxw = nxu > warp ? (nxu - warp + nwb - 1) / nwb : 0;                                 // ... of this warp
  const int nitems = nch * tpw;
  const int nall = nitems + nxw;
  auto tile_of = [&](int it) -> int {
    if (it >= nitems) return 16 * base * G + (int)blockIdx.x + (warp + (it - nitems) * nwb) * G;  // leftover tile
    const int k = it / tpw, ti = it - k * tpw;
    return 16 * (blockIdx.x + k * gridDim.x) + warp + nwb * ti;
  };
  auto src = [&](int tile, int p) -> const uint8_t * { return P.wout + (size_t)tile * 2 * 6528 + p * 1632; };
  Q8Regs<NTH, T * INNER / 16, 0> qr;
  qr.load((const int4 *)P.oq, nullptr);  // activations first, then the first weight tile
  constexpr int NXD = (T * 192 + NTH - 1) / NTH;
  float odv[NXD], rho[NXD];
#pragma unroll
  for (int j = 0; j < NXD; j++) {
    const int idx = threadIdx.x + j * NTH;
    odv[j] = 0.f; rho[j] = 0.f;
    if (idx < T * 192) {
      const int t = idx / 192, h = (idx - t * 192) >> 2;
      const float4 sq = __ldcg((const float4 *)(P.osq + t * 192 + h * 4));
      odv[j] = __ldcg(P.od + idx);
      rho[j] = rsqrtf((((sq.x + sq.y) + sq.z) + sq.w) / 128.f + P.eps);
    }
  }
  TileRegs<2, 1632> regs;
  if (nall > 0) regs.load(src(tile_of(0), 0), 6528, lane);
  if (threadIdx.x < 16) {
    const int t = threadIdx.x >> 2, s = threadIdx.x & 3;
    float v = P.rr[t * 4 + 0] * P.injp[(t * 4 + 0) * 4 + s];
    v += P.rr[t * 4 + 1] * P.injp[(t * 4 + 1) * 4 + s];
    v += P.rr[t * 4 + 2] * P.injp[(t * 4 + 2) * 4 + s];
    v += P.rr[t * 4 + 3] * P.injp[(t * 4 + 3) * 4 + s];
    inj_s[threadIdx.x] = 2.f * fsigmoid(v * 0.25f);  // combine gain
  }
  qr.store(sm);
#pragma unroll
  for (int j = 0; j < NXD; j++) {
    const int idx = threadIdx.x + j * NTH;
    if (idx < T * 192) ((float *)(sm + XDOFF))[idx] = odv[j] * rho[j];
  }
  __syncthreads();
  const int4 *xq = A7_XQ(sm);
  const float *xd = A7_XD(sm);
  int it = 0;
  // phases k < nch: own chunks (tiles, then block barrier + rquant_chunk); phase k == nch (only if this block has leftover tiles): the leftover tiles of the
  // warp, R written to global only (the chunk is quantized by its last arriver, bal_finish)
  for (int k = 0; k < nch + (nxu > 0 ? 1 : 0); k++) {
    const bool ext = k >= nch;
    const int c = blockIdx.x + k * gridDim.x;
    const int ntl = ext ? nxw : tpw;
    for (int ti = 0; ti < ntl; ti++, it++) {
      const int tile = tile_of(it);
      float acc[2][T];
#pragma unroll
      for (int r = 0; r < 2; r++)
#pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.f;
#pragma unroll 1
      for (int p = 0; p < 4; p++) {
        __syncwarp();
        regs.store(wtile, lane);
        __syncwarp();
        const uint8_t *np = p == 3 ? (it + 1 < nall ? src(tile_of(it + 1), 0) : nullptr) : src(tile, p + 1);
        if (np) regs.load(np, 6528, lane);
        q8_tile_dot<2, 48>(wtile, xq, xd, 192, p * 48, lane, acc);
      }
      float v[8];
#pragma unroll
      for (int r = 0; r < 2; r++)
#pragma unroll
        for (int t = 0; t < T; t++) v[r * 4 + t] = acc[r][t];
      const float y = rscatter<8>(v, lane);
      if (lane < 8) {
        const int r = lane >> 2, t = lane & 3, i = tile * 2 + r;
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const float rn = P.R[(size_t)(t * 4 + s) * D + i] + y * inj_s[t * 4 + s];
          P.Rout[(size_t)(t * 4 + s) * D + i] = rn;
          if (!ext) Rn_s[(t * 4 + s) * 32 + (i - c * 32)] = rn;
        }
      }
    }
    if (!ext) {
      __syncthreads();
      rquant_chunk<NTH>(P, c, Rn_s, warp, lane);
      __syncthreads();
    }
  }
  if (rem > 0) bal_finish<NTH>(P, base, nxu, 16, Rn_s, (int *)(scr + 576));  // all blocks (block-uniform: nxu may be 0)
}

// ------------------------------------------------------------------ kernels
// q8 of R * hc_ffn_norm per stream (the activation of hc1f) and the sums of squares of R (for the rms). One block per 32-column chunk. Same formulas as
// gate0's rquant_chunk: Rq[(s*T+t)*D + i], Rd[(s*T+t)*80 + chunk], Rss[(s*T+t)*80 + chunk].
__global__ void __launch_bounds__(256, 1) a7_ffn_rquant(const __grid_constant__ A7P P) {
  const int c = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll 1
  for (int p = warp; p < 16; p += 8) {
    const int t = p >> 2, s = p & 3;
    const float x = P.R[(size_t)p * D + c * 32 + lane];
    int8_t q; float d;
    q8_warp(x * P.norm_f[s * D + c * 32 + lane], q, d);
    const float ss = warp_sum(x * x);
    P.Rq[(size_t)(s * T + t) * D + c * 32 + lane] = q;
    if (lane == 0) { P.Rd[(s * T + t) * 80 + c] = d; P.Rss[(s * T + t) * 80 + c] = ss; }
  }
}

// one stage per kernel (gate0 graph mode k_stage<THREADS, S>, renamed): S = 5 hc1f, 6 hc2f, 7 router, 8 gateup, 9 qh, 10 down (the numbers of gate0's stages).
// 256 threads (80 SMs): balance 0; 384 threads (72 SMs): balance 2.
template <int THREADS, int S>
__global__ void __launch_bounds__(THREADS, 1) a7_ffn_stage(const __grid_constant__ A7P P) {
  constexpr int BAL = THREADS == 384 ? 2 : 0;
  if constexpr (S == 5) { if constexpr (BAL == 2) stage_h1_b2<THREADS>(P, 5); else stage_h1<THREADS>(P, 5); }
  else if constexpr (S == 6) { if constexpr (BAL == 2) stage_h2_b2<THREADS>(P, 6); else stage_h2<THREADS>(P, 6); }
  else if constexpr (S == 7) stage_router<THREADS>(P);
  else if constexpr (S == 8) stage_gateup<THREADS>(P);
  else if constexpr (S == 9) stage_qh<THREADS>(P);
  else if constexpr (S == 10) stage_down<THREADS, BAL>(P);
}

// routing table in a kernel of its own (LLAMA_A7_ROUTE_SEPARATE=1; the default builds it in the last block of the router stage): one block of 128 threads
__global__ void __launch_bounds__(128, 1) a7_ffn_route(const __grid_constant__ A7P P) {
  __shared__ A7RouteSmem sm;
  if (P.route_ref) a7_route_build_ref(P.logits, P.route, &sm);
  else a7_route_build(P.logits, P.route, &sm);
}

// check mode (review 10-16 I-1: measured on the delta the op adds, not on R + delta): one block per token t (n floats each),
// out[t * 8 + ...] = { sum (a-b)^2, sum (b-r)^2, sum r^2, max |a-b|, max |b-r| }  with a = A7, b = production, r = the residual in
__global__ void __launch_bounds__(1024, 1) a7_ffn_check(const float *a, const float *b, const float *r, int n, float *out) {
  __shared__ float sh[5][32];
  const size_t o = (size_t) blockIdx.x * n;
  float d2 = 0.f, db2 = 0.f, r2 = 0.f, md = 0.f, mdb = 0.f;
  for (int i = threadIdx.x; i < n; i += 1024) {
    const float d = a[o + i] - b[o + i], db = b[o + i] - r[o + i];
    d2 += d * d; db2 += db * db; r2 += r[o + i] * r[o + i];
    md = fmaxf(md, fabsf(d)); mdb = fmaxf(mdb, fabsf(db));
  }
  d2 = warp_sum(d2); db2 = warp_sum(db2); r2 = warp_sum(r2); md = warp_max(md); mdb = warp_max(mdb);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) { sh[0][warp] = d2; sh[1][warp] = db2; sh[2][warp] = r2; sh[3][warp] = md; sh[4][warp] = mdb; }
  __syncthreads();
  if (warp == 0) {
    d2 = warp_sum(sh[0][lane]); db2 = warp_sum(sh[1][lane]); r2 = warp_sum(sh[2][lane]); md = warp_max(sh[3][lane]); mdb = warp_max(sh[4][lane]);
    if (lane == 0) { float *q = out + blockIdx.x * 8; q[0] = d2; q[1] = db2; q[2] = r2; q[3] = md; q[4] = mdb; }
  }
}

// ------------------------------------------------------------------ P2 kernels (names start with a7_attn_)
// q8 of R * hc_attn_norm per stream (the activation of hc1a); the same code as a7_ffn_rquant (P.norm_f is the hc_attn_norm of the attn op)
__global__ void __launch_bounds__(256, 1) a7_attn_rquant(const __grid_constant__ A7P P) {
  const int c = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll 1
  for (int p = warp; p < 16; p += 8) {
    const int t = p >> 2, s = p & 3;
    const float x = P.R[(size_t)p * D + c * 32 + lane];
    int8_t q; float d;
    q8_warp(x * P.norm_f[s * D + c * 32 + lane], q, d);
    const float ss = warp_sum(x * x);
    P.Rq[(size_t)(s * T + t) * D + c * 32 + lane] = q;
    if (lane == 0) { P.Rd[(s * T + t) * 80 + c] = d; P.Rss[(s * T + t) * 80 + c] = ss; }
  }
}

// one stage per kernel: S = 0 hc1a, 1 hc2a, 2 proj, 3 gdn, 4 ssmout (the numbers of gate0's stages). 256 threads (80 SMs): balance 0; 384 threads (72 SMs): balance 2.
template <int THREADS, int S>
__global__ void __launch_bounds__(THREADS, 1) a7_attn_stage(const __grid_constant__ A7P P) {
  constexpr int BAL = THREADS == 384 ? 2 : 0;
  if constexpr (S == 0) { if constexpr (BAL == 2) stage_h1_b2<THREADS>(P, 0); else stage_h1<THREADS>(P, 0); }
  else if constexpr (S == 1) { if constexpr (BAL == 2) stage_h2_b2<THREADS>(P, 1); else stage_h2<THREADS>(P, 1); }
  else if constexpr (S == 2) attn_proj<THREADS>(P);
  else if constexpr (S == 3) attn_gdn<THREADS>(P);
  else if constexpr (S == 4) attn_out<THREADS, BAL>(P);
}

// check mode: the state rows of the 4 slots (row s * mem_size + kv_head) of the A7 result (work copies) and of the production result.
// grid (blocks, 4 slots, 2 kinds: 0 ssm, 1 conv); out[(kind * 4 + s) * 2 + {0: sum (a-b)^2, 1: sum b^2}] (float atomics, zeroed before)
__global__ void __launch_bounds__(256, 1) a7_attn_scheck(const float *ssm_a, const float *ssm_b, const float *conv_a, const float *conv_b, int mem_size, int kv_head, float *out) {
  __shared__ float sh[2][8];
  const int s = blockIdx.y, kind = blockIdx.z;
  const size_t n = kind ? (size_t) CONVROW : SSMROW;
  const size_t off = ((size_t) s * mem_size + kv_head) * n;
  const float *a = (kind ? conv_a : ssm_a) + off, *b = (kind ? conv_b : ssm_b) + off;
  float d2 = 0.f, b2 = 0.f;
  for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
    const float d = a[i] - b[i];
    d2 += d * d; b2 += b[i] * b[i];
  }
  d2 = warp_sum(d2); b2 = warp_sum(b2);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) { sh[0][warp] = d2; sh[1][warp] = b2; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sd = 0.f, sb = 0.f;
    for (int w = 0; w < (int) (blockDim.x >> 5); w++) { sd += sh[0][w]; sb += sh[1][w]; }
    atomicAdd(out + (kind * 4 + s) * 2 + 0, sd);
    atomicAdd(out + (kind * 4 + s) * 2 + 1, sb);
  }
}

// LLAMA_A7_RQ_VERIFY=1: the q8 of the work area (made by the previous op) against one made now at the head of the op (work-area copy v*): counts of
// mismatches res[0] = Rq (exact), res[1] = Rd (exact bits), res[2] = Rss (relative 1e-6)
__global__ void __launch_bounds__(256, 1) a7_rq_verify(const int8_t *q1, const int8_t *q2, const float *d1, const float *d2, const float *s1, const float *s2, unsigned *res) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 16 * D && q1[i] != q2[i]) atomicAdd(res + 0, 1u);
  if (i < 16 * 80) {
    if (__float_as_uint(d1[i]) != __float_as_uint(d2[i])) atomicAdd(res + 1, 1u);
    const float a = s1[i], b = s2[i];
    if (!(fabsf(a - b) <= 1e-6f * fmaxf(fabsf(a), fabsf(b)))) atomicAdd(res + 2, 1u);
  }
}

}  // namespace a7ffn

// ================================================================== host side
// work area: one persistent allocation per (device, stream), zero-filled once (counters are never reset: arrivals per launch are multiples of `need`; cnt[A7_CNT_ROUTE] is reset by the last block)
// P2: the q8 of R * norm (Rq / Rd / Rss) is handed from one op to the next in this area: the attn op's ssmout stage makes the ffn-side q8 of its layer, the ffn
// op's down stage the attn-side q8 of the next layer. The host-side tag says which one is in the area now; an op that reads it (rq_ready = 1) aborts on a mismatch.
// The ops of one stream run in the order they are enqueued, so the tag needs no synchronization.
enum { A7_SIDE_NONE = -1, A7_SIDE_ATTN = 0, A7_SIDE_FFN = 1 };
struct a7_work {
  void * base = nullptr;
  a7ffn::A7P p;   // work-area pointers only (weights and R are filled per call)
  float * chk = nullptr;
  float * schk = nullptr;                       // check mode of the attn op: state compare sums (16 floats)
  int8_t * vRq = nullptr; float * vRd = nullptr; float * vRss = nullptr;  // LLAMA_A7_RQ_VERIFY: the q8 made at the head of an op with rq_ready = 1
  unsigned * vres = nullptr;                    // [4] mismatch counts of the verification
  int tag_il = -1;                              // the q8 in the work area is the one of layer tag_il, side tag_side
  int tag_side = A7_SIDE_NONE;
};

static a7_work * a7_get_work(int device, cudaStream_t stream) {
  static std::mutex mutex;
  static std::map<std::pair<int, cudaStream_t>, a7_work *> works;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = works.find({device, stream});
  if (it != works.end()) {
    return it->second;
  }
  using namespace a7ffn;
  size_t off = 0;
  auto take = [&](size_t bytes) { const size_t o = off; off += (bytes + 255) & ~(size_t) 255; return o; };
  const size_t o_cnt = take(4096 * 4), o_h2 = take(2 * NCH * 4), o_bal = take(NCH * 4), o_rr = take(64), o_part = take((size_t) T * 4 * LR * 4), o_injp = take(256),
               o_mixed = take((size_t) T * D * 4), o_logits = take((size_t) T * NEXP * 4), o_sgl = take(64),
               o_hbuf = take((size_t) (MAXU + 1) * 4 * FF * 4), o_mq = take((size_t) T * D), o_md = take((size_t) T * 80 * 4),
               o_loq = take((size_t) T * LR), o_lod = take((size_t) T * 10 * 4), o_Rq = take((size_t) 16 * D), o_Rd = take((size_t) 16 * 80 * 4),
               o_Rss = take((size_t) 16 * 80 * 4), o_hq = take((size_t) MAXP * FF), o_hd = take((size_t) MAXP * 20 * 4), o_route = take(sizeof(A7Route)),
               o_chk = take((size_t) T * 8 * sizeof(float)),
               // P2 (attn op)
               o_qkv = take((size_t) T * QKVC * 4), o_z = take((size_t) T * INNER * 4), o_a = take((size_t) T * NVH * 4), o_b = take((size_t) T * NVH * 4),
               o_oq = take((size_t) T * INNER), o_od = take((size_t) T * 192 * 4), o_osq = take((size_t) T * 192 * 4), o_schk = take(16 * sizeof(float)),
               o_vRq = take((size_t) 16 * D), o_vRd = take((size_t) 16 * 80 * 4), o_vRss = take((size_t) 16 * 80 * 4), o_vres = take(4 * sizeof(unsigned));
  a7_work * w = new a7_work();
  CUDA_CHECK(cudaMalloc(&w->base, off));
  CUDA_CHECK(cudaMemsetAsync(w->base, 0, off, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  char * b = (char *) w->base;
  memset(&w->p, 0, sizeof(w->p));
  w->p.cnt = (unsigned *) (b + o_cnt);   w->p.h2cnt = (int *) (b + o_h2); w->p.balcnt = (int *) (b + o_bal);
  w->p.rr = (float *) (b + o_rr);        w->p.part = (float *) (b + o_part);   w->p.injp = (float *) (b + o_injp);
  w->p.mixed = (float *) (b + o_mixed);  w->p.logits = (float *) (b + o_logits); w->p.sgl = (float *) (b + o_sgl);
  w->p.hbuf = (float *) (b + o_hbuf);
  w->p.mq = (int8_t *) (b + o_mq);       w->p.md = (float *) (b + o_md);
  w->p.loq = (int8_t *) (b + o_loq);     w->p.lod = (float *) (b + o_lod);
  w->p.Rq = (int8_t *) (b + o_Rq);       w->p.Rd = (float *) (b + o_Rd);       w->p.Rss = (float *) (b + o_Rss);
  w->p.hq = (int8_t *) (b + o_hq);       w->p.hd = (float *) (b + o_hd);
  w->p.route = (A7Route *) (b + o_route);
  w->chk = (float *) (b + o_chk);
  w->p.qkv_o = (float *) (b + o_qkv);    w->p.z_o = (float *) (b + o_z);       w->p.a_o = (float *) (b + o_a);       w->p.b_o = (float *) (b + o_b);
  w->p.oq = (int8_t *) (b + o_oq);       w->p.od = (float *) (b + o_od);       w->p.osq = (float *) (b + o_osq);
  w->schk = (float *) (b + o_schk);
  w->vRq = (int8_t *) (b + o_vRq);       w->vRd = (float *) (b + o_vRd);       w->vRss = (float *) (b + o_vRss);     w->vres = (unsigned *) (b + o_vres);
  works.emplace(std::make_pair(std::make_pair(device, stream), w));
  return w;
}

// shape and type of one layer tensor
static bool a7_tensor_ok(const ggml_tensor * t, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
  return t != nullptr && t->type == type && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 && t->ne[3] == 1 &&
         ggml_is_contiguous(t) && t->data != nullptr && t->extra == nullptr;
}

bool ggml_cuda_a7_gdn_ffn_supported(int device, const ggml_tensor * op) {
  using namespace a7ffn;
  const ggml_tensor * r = op->src[0];
  if (r == nullptr || op->src[1] == nullptr || op->type != GGML_TYPE_F32 || r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) ||
      r->ne[0] != D || r->ne[1] != 4 || r->ne[2] != T || r->ne[3] != 1 || !ggml_are_same_shape(op, r)) {
    return false;
  }
  const ggml_tensor * ref = op->src[2];
  if (ref != nullptr && (ref->type != GGML_TYPE_F32 || !ggml_are_same_shape(ref, r) || !ggml_is_contiguous(ref))) {
    return false;
  }
  const ggml_a7_ffn_layer * L = nullptr;
  memcpy(&L, op->op_params, sizeof(L));
  if (L == nullptr) {
    return false;
  }
  if (!a7_tensor_ok(L->hc_norm, GGML_TYPE_F32, D, 4, 1) || !a7_tensor_ok(L->hc_down, GGML_TYPE_Q8_0, HCD, LR, 1) ||
      !a7_tensor_ok(L->hc_up, GGML_TYPE_Q8_0, LR, HCD, 1) || !a7_tensor_ok(L->hc_inject, GGML_TYPE_F32, HCD, 4, 1) ||
      !a7_tensor_ok(L->gate_inp, GGML_TYPE_F32, D, NEXP, 1) || L->gate_inp_shexp == nullptr || L->gate_inp_shexp->type != GGML_TYPE_F32 ||
      ggml_nelements(L->gate_inp_shexp) != D || !ggml_is_contiguous(L->gate_inp_shexp) || L->gate_inp_shexp->data == nullptr ||
      !a7_tensor_ok(L->gate_exps, GGML_TYPE_IQ2_S, D, FF, NEXP) || !a7_tensor_ok(L->up_exps, GGML_TYPE_IQ2_S, D, FF, NEXP) ||
      !a7_tensor_ok(L->down_exps, GGML_TYPE_IQ4_NL, FF, D, NEXP) || !a7_tensor_ok(L->gate_shexp, GGML_TYPE_Q8_0, D, FF, 1) ||
      !a7_tensor_ok(L->up_shexp, GGML_TYPE_Q8_0, D, FF, 1) || !a7_tensor_ok(L->down_shexp, GGML_TYPE_Q8_0, FF, D, 1)) {
    return false;
  }
  if (L->gate_exps->nb[2] != (size_t) EXP_GU_BYTES || L->up_exps->nb[2] != (size_t) EXP_GU_BYTES || L->down_exps->nb[2] != (size_t) EXP_D_BYTES) {
    return false;
  }
  const int nsm = ggml_cuda_info().devices[device].nsm;
  if (nsm != 80 && nsm != 72) {
    return false;
  }
  const int threads = nsm == 80 ? 256 : 384;
  const size_t smem = (size_t) ACTB + (size_t) (threads / 32) * TILEB + sizeof(float) * SCRF + sizeof(A7Route);
  return smem <= ggml_cuda_info().devices[device].smpbo;
}

// hc_attn_norm of the next layer as a device pointer, or nullptr when there is none or it does not sit in device memory of `device`
// (a layer on the other GPU, or a host buffer): the q8 of the new R for the next layer is then skipped. Cached per (layer, device).
static const float * a7_norm_next(const ggml_a7_ffn_layer * L, int device) {
  const ggml_tensor * t = L->hc_norm_next;
  if (t == nullptr || t->data == nullptr || t->type != GGML_TYPE_F32 || t->ne[0] != a7ffn::D || t->ne[1] != 4 || t->ne[2] != 1 || !ggml_is_contiguous(t)) {
    return nullptr;
  }
  static std::mutex mutex;
  static std::map<std::pair<int, int>, bool> ok;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = ok.find({L->il, device});
  if (it == ok.end()) {
    cudaPointerAttributes attr;
    bool good = cudaPointerGetAttributes(&attr, t->data) == cudaSuccess && attr.type == cudaMemoryTypeDevice && attr.device == device;
    cudaGetLastError();  // clear a possible error of the query
    it = ok.emplace(std::make_pair(L->il, device), good).first;
  }
  return it->second ? (const float *) t->data : nullptr;
}

// 10-16 diag (review of P1-result C-1: the production layer after an A7 op is about 17 us slower): LLAMA_A7_TAIL=1 launches, after the op, an
// empty kernel (one block per SM) whose shared-memory carveout preference is max L1, =2 the same empty kernel with the default preference.
// Tests whether the L1 / shared split chosen for the A7 stages (58-71 KB of dynamic shared memory) stays for the next production kernels.
static __global__ void a7_tail_l1() {}
static __global__ void a7_tail_def() {}
static int a7_tail_mode() {
  static const int m = [] {
    const char * e = getenv("LLAMA_A7_TAIL");
    const int v = e ? atoi(e) : 0;
    if (v != 0) fprintf(stderr, "A7: tail kernel mode %d\n", v);
    return v;
  }();
  return m;
}

template <int TH>
static void a7_launch(const a7ffn::A7P & P, cudaStream_t st, int grid, int device, int rq_ready) {
  using namespace a7ffn;
  const size_t dyn = (size_t) ACTB + (size_t) (TH / 32) * TILEB;
  static std::mutex mutex;
  static bool done[GGML_CUDA_MAX_DEVICES] = {};
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!done[device]) {
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_ffn_stage<TH, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_ffn_stage<TH, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_ffn_stage<TH, 7>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_ffn_stage<TH, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_ffn_stage<TH, 9>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_ffn_stage<TH, 10>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_tail_l1, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxL1));
      done[device] = true;
    }
  }
  const dim3 gd(grid), bl(TH);
  if (!rq_ready) {  // rq_ready = 1 (P2): the q8 of R * hc_ffn_norm was made by the ssmout stage of the attn op of the same layer
    a7_ffn_rquant<<<NCH, 256, 0, st>>>(P);
  }
  a7_ffn_stage<TH, 5><<<gd, bl, dyn, st>>>(P);   // hc1f
  a7_ffn_stage<TH, 6><<<gd, bl, dyn, st>>>(P);   // hc2f
  a7_ffn_stage<TH, 7><<<gd, bl, dyn, st>>>(P);   // router
  if (!P.route_inline) a7_ffn_route<<<1, 128, 0, st>>>(P);  // routing table (default: built in the router stage)
  a7_ffn_stage<TH, 8><<<gd, bl, dyn, st>>>(P);   // gateup
  a7_ffn_stage<TH, 9><<<gd, bl, dyn, st>>>(P);   // qh
  a7_ffn_stage<TH, 10><<<gd, bl, dyn, st>>>(P);  // down + hc_combine(ffn)
  if (a7_tail_mode() == 1) a7_tail_l1<<<gd, 32, 0, st>>>();
  else if (a7_tail_mode() == 2) a7_tail_def<<<gd, 32, 0, st>>>();
  CUDA_CHECK(cudaGetLastError());
}

// ---- P2: the q8 hand-over between the attn op and the ffn op (tag) and its verification
// rq_ready = 1: the op reads the q8 of R * norm from the work area instead of making it; abort when the tag is not {il, side}
static void a7_tag_require(const a7_work * w, const char * who, int il, int side) {
  if (w->tag_il != il || w->tag_side != side) {
    GGML_ABORT("A7: %s op of layer %d has rq_ready=1 but the q8 in the work area is tagged {il=%d, side=%d}, expected {il=%d, side=%d}", who, il, w->tag_il,
               w->tag_side, il, side);
  }
}

static bool a7_rq_verify_on() {
  static const bool v = [] {
    const char * e = getenv("LLAMA_A7_RQ_VERIFY");
    const bool on = e != nullptr && atoi(e) == 1;
    if (on) fprintf(stderr, "A7: LLAMA_A7_RQ_VERIFY=1 (rq_ready ops make the q8 again and compare; needs GGML_CUDA_DISABLE_GRAPHS=1)\n");
    return on;
  }();
  return v;
}

// LLAMA_A7_RQ_VERIFY=1 and rq_ready = 1: make the q8 of the head of the op once more into a separate area (v*) and compare it with the one in the work
// area (Rq, Rd exact; Rss relative 1e-6). Called before the stages of the op, while the work-area q8 is still the one the op will read.
static void a7_rq_verify_run(a7_work * w, const a7ffn::A7P & P, cudaStream_t st, int il, const char * side, bool attn) {
  using namespace a7ffn;
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(st, &cap));
  if (cap != cudaStreamCaptureStatusNone) {
    GGML_ABORT("A7: LLAMA_A7_RQ_VERIFY=1 needs GGML_CUDA_DISABLE_GRAPHS=1 (the verification synchronizes the stream)");
  }
  A7P Pv = P;
  Pv.Rq = w->vRq; Pv.Rd = w->vRd; Pv.Rss = w->vRss;
  if (attn) a7_attn_rquant<<<NCH, 256, 0, st>>>(Pv);
  else a7_ffn_rquant<<<NCH, 256, 0, st>>>(Pv);
  CUDA_CHECK(cudaMemsetAsync(w->vres, 0, 4 * sizeof(unsigned), st));
  a7_rq_verify<<<(16 * D + 255) / 256, 256, 0, st>>>(P.Rq, w->vRq, P.Rd, w->vRd, P.Rss, w->vRss, w->vres);
  CUDA_CHECK(cudaGetLastError());
  unsigned h[4] = {0, 0, 0, 0};
  CUDA_CHECK(cudaMemcpyAsync(h, w->vres, sizeof(h), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  if (h[0] != 0 || h[1] != 0 || h[2] != 0) {
    fprintf(stderr, "A7RQ mismatch layer=%d side=%s Rq=%u Rd=%u Rss=%u\n", il, side, h[0], h[1], h[2]);
  }
}

void ggml_cuda_op_a7_gdn_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
  using namespace a7ffn;
  GGML_ASSERT(ggml_cuda_a7_gdn_ffn_supported(ctx.device, dst));
  const ggml_a7_ffn_layer * L = nullptr;
  memcpy(&L, dst->op_params, sizeof(L));
  const int il = ((const int32_t *) dst->op_params)[2];
  const int rq_ready = ((const int32_t *) dst->op_params)[3];  // P2: the q8 of R * hc_ffn_norm is in the work area (made by the attn op of this layer)
  const ggml_tensor * res = dst->src[0];
  const ggml_tensor * ref = dst->src[2];
  cudaStream_t st = ctx.stream();

  {  // once per layer: where the op runs
    static std::mutex mutex;
    static std::set<std::pair<int, int>> seen;
    std::lock_guard<std::mutex> lock(mutex);
    if (seen.insert({il, ctx.device}).second) {
      fprintf(stderr, "A7: ffn op on layer %d (device %d)\n", il, ctx.device);
    }
  }

  a7_work * w = a7_get_work(ctx.device, st);
  A7P P = w->p;
  P.norm_f = (const float *) L->hc_norm->data;
  P.hcd_f = (const uint8_t *) L->hc_down->data;
  P.hcu_f = (const uint8_t *) L->hc_up->data;
  P.inj_f = (const float *) L->hc_inject->data;
  P.router = (const float *) L->gate_inp->data;
  P.sgw = (const float *) L->gate_inp_shexp->data;
  P.eg = (const uint8_t *) L->gate_exps->data;
  P.eu = (const uint8_t *) L->up_exps->data;
  P.ed = (const uint8_t *) L->down_exps->data;
  P.sg = (const uint8_t *) L->gate_shexp->data;
  P.su = (const uint8_t *) L->up_shexp->data;
  P.sd = (const uint8_t *) L->down_shexp->data;
  P.R = (const float *) res->data;
  P.Rout = (float *) dst->data;
  P.norm_next = a7_norm_next(L, ctx.device);
  P.eps = 1e-6f;  // hc_mix rms eps of the P1 op (as before P2)
  if (rq_ready) {
    a7_tag_require(w, "ffn", il, A7_SIDE_FFN);
    if (a7_rq_verify_on()) a7_rq_verify_run(w, P, st, il, "ffn", false);
  }
  {  // LLAMA_A7_ROUTE_SEPARATE=1: the routing table by a kernel of its own instead of the last block of the router stage (read once)
    static const int route_inline = [] {
      const char * s = getenv("LLAMA_A7_ROUTE_SEPARATE");
      const bool sep = s != nullptr && strcmp(s, "1") == 0;
      fprintf(stderr, "A7: routing table %s\n", sep ? "by the separate kernel a7_ffn_route" : "built in the router stage (last arriver)");
      return sep ? 0 : 1;
    }();
    P.route_inline = route_inline;
    static const int route_ref = [] {  // 10-16 diag: LLAMA_A7_ROUTE_REF=1 = the P1 build of the table (see A7P::route_ref)
      const char * s = getenv("LLAMA_A7_ROUTE_REF");
      const bool r = s != nullptr && strcmp(s, "1") == 0;
      if (r) fprintf(stderr, "A7: routing table with the P1 build (a7_route_build_ref)\n");
      return r ? 1 : 0;
    }();
    P.route_ref = route_ref;
  }

  if (ggml_cuda_info().devices[ctx.device].nsm == 80) {
    a7_launch<256>(P, st, 80, ctx.device, rq_ready);
  } else {
    a7_launch<384>(P, st, 72, ctx.device, rq_ready);
  }
  // the work area now holds the q8 of the new R * hc_attn_norm of the next layer (when the down stage made it), else nothing valid
  if (P.norm_next != nullptr) {
    w->tag_il = il + 1;
    w->tag_side = A7_SIDE_ATTN;
  } else {
    w->tag_il = -1;
    w->tag_side = A7_SIDE_NONE;
  }

  if (ref != nullptr) {  // check mode: compare with the production result, then continue with the production values
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;  // review 10-16 m-6: the check synchronizes the stream
    CUDA_CHECK(cudaStreamIsCapturing(st, &cap));
    if (cap != cudaStreamCaptureStatusNone) {
      GGML_ABORT("A7: LLAMA_A7_CHECK=1 needs GGML_CUDA_DISABLE_GRAPHS=1 (the check synchronizes the stream)");
    }
    const int nt = (int) dst->ne[2];
    const int n = (int) (ggml_nelements(dst) / nt);
    GGML_ASSERT((size_t) nt * 8 * sizeof(float) <= (size_t) T * 8 * sizeof(float));
    a7_ffn_check<<<nt, 1024, 0, st>>>((const float *) dst->data, (const float *) ref->data, (const float *) res->data, n, w->chk);
    CUDA_CHECK(cudaGetLastError());
    float h[64];
    CUDA_CHECK(cudaMemcpyAsync(h, w->chk, nt * 8 * sizeof(float), cudaMemcpyDeviceToHost, st));
    A7Route hr;  // the routing table of this call (read-only for the stages after the router): N and the number of pairs
    CUDA_CHECK(cudaMemcpyAsync(&hr, w->p.route, sizeof(A7Route), cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    // d_rel_l2 = |A7 - prod| / |prod delta| over all tokens; tok_max = the worst token of the same; d_max_rel = max|A7 - prod| / max|prod delta|;
    // dR = |prod delta| / |R in| (how much the layer adds; the old rel_l2 on R_out is about d_rel_l2 * dR)
    double d2 = 0, db2 = 0, r2 = 0, md = 0, mdb = 0, tok = 0;
    for (int t = 0; t < nt; t++) {
      const float * q = h + t * 8;
      d2 += q[0]; db2 += q[1]; r2 += q[2]; md = fmax(md, (double) q[3]); mdb = fmax(mdb, (double) q[4]);
      tok = fmax(tok, q[1] > 0.f ? sqrt((double) q[0] / q[1]) : (q[0] > 0.f ? INFINITY : 0.0));
    }
    const double rel = db2 > 0 ? sqrt(d2 / db2) : (d2 > 0 ? INFINITY : 0.0);
    const double mrel = mdb > 0 ? md / mdb : (md > 0 ? INFINITY : 0.0);
    const double dR = r2 > 0 ? sqrt(db2 / r2) : 0.0;
    // review 10-16 (2nd, minor 1): the device-side fmaxf drops NaN, and a NaN in a or b makes d2 / db2 NaN or inf: report nan, not a 0 that looks like an exact match
    const bool fin = std::isfinite(d2) && std::isfinite(db2);
    const double nanv = std::nan("");
    const int npairs = (hr.N >= 0 && hr.N <= MAXU) ? hr.pref[hr.N + 1] : -1;
    fprintf(stderr, "A7CHECK layer=%d dev=%d d_rel_l2=%.6g tok_max=%.6g d_max_rel=%.6g dR=%.6g N=%d pairs=%d\n", il, ctx.device, fin ? rel : nanv,
            fin ? tok : nanv, fin ? mrel : nanv, dR, hr.N, npairs);
    CUDA_CHECK(cudaMemcpyAsync(dst->data, ref->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, st));
  }
}

// ================================================================== P2 host side: GGML_OP_A7_GDN_ATTN
// op_params: [0..1] L, [2] kv_head, [3] mem_size, [4] K, [5] rq_ready, [6] eps (float bits), [7] mode (0 compute, 1 check-mode comparison), [8] check.
// srcs, mode 0: [0] res_hc, [1] anchor (hc_attn_norm), [2] s_copy, [3] conv_all (check: work copy), [4] ssm_all (check: work copy).
// srcs, mode 1: [0] A7 result, [1] production result, [2] s_copy, [3] conv_w, [4] ssm_w, [5] conv_all, [6] ssm_all, [7] input residual.

// a recurrent state buffer (r_l / s_l, or its work copy): F32, rows of n floats, at least K * mem_size rows
static bool a7_attn_state_ok(const ggml_tensor * t, int64_t n, int K, int mem_size) {
  return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == n && t->ne[1] >= (int64_t) K * mem_size && t->ne[2] == 1 &&
         t->ne[3] == 1;
}

static bool a7_attn_same_f32(const ggml_tensor * t, const ggml_tensor * r) {
  return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_are_same_shape(t, r);
}

bool ggml_cuda_a7_gdn_attn_supported(int device, const ggml_tensor * op) {
  using namespace a7ffn;
  const int32_t * prm = (const int32_t *) op->op_params;
  const int kv_head = prm[2], mem_size = prm[3], K = prm[4], mode = prm[7];
  const ggml_tensor * r = op->src[0];
  if (r == nullptr || op->type != GGML_TYPE_F32 || r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) || r->ne[0] != D || r->ne[1] != 4 || r->ne[2] != T ||
      r->ne[3] != 1 || !ggml_are_same_shape(op, r)) {
    return false;
  }
  if ((mode != 0 && mode != 1) || K != T || mem_size < 1 || kv_head < 0 || kv_head >= mem_size) {
    return false;
  }
  const ggml_a7_attn_layer * L = nullptr;
  memcpy(&L, op->op_params, sizeof(L));
  if (L == nullptr) {
    return false;
  }
  if (!a7_tensor_ok(L->hc_norm, GGML_TYPE_F32, D, 4, 1) || !a7_tensor_ok(L->hc_down, GGML_TYPE_Q8_0, HCD, LR, 1) ||
      !a7_tensor_ok(L->hc_up, GGML_TYPE_Q8_0, LR, HCD, 1) || !a7_tensor_ok(L->hc_inject, GGML_TYPE_F32, HCD, 4, 1) ||
      !a7_tensor_ok(L->wqkv, GGML_TYPE_Q8_0, D, QKVC, 1) || !a7_tensor_ok(L->wqkv_gate, GGML_TYPE_Q8_0, D, INNER, 1) ||
      !a7_tensor_ok(L->ssm_alpha, GGML_TYPE_F32, D, NVH, 1) || !a7_tensor_ok(L->ssm_beta, GGML_TYPE_F32, D, NVH, 1) ||
      !a7_tensor_ok(L->ssm_dt, GGML_TYPE_F32, NVH, 1, 1) || !a7_tensor_ok(L->ssm_a, GGML_TYPE_F32, NVH, 1, 1) ||
      !a7_tensor_ok(L->ssm_conv1d, GGML_TYPE_F32, 4, QKVC, 1) || !a7_tensor_ok(L->ssm_norm, GGML_TYPE_F32, HD, 1, 1) ||
      !a7_tensor_ok(L->ssm_out, GGML_TYPE_Q8_0, INNER, D, 1) || !a7_tensor_ok(L->hc_ffn_norm, GGML_TYPE_F32, D, 4, 1)) {
    return false;
  }
  const ggml_tensor * s_copy = op->src[2];
  if (s_copy == nullptr || s_copy->type != GGML_TYPE_I32 || ggml_nelements(s_copy) < 1) {
    return false;
  }
  if (mode == 1) {
    if (!a7_attn_same_f32(op->src[1], r) || !a7_attn_same_f32(op->src[7], r) || !a7_attn_state_ok(op->src[3], CONVROW, K, mem_size) ||
        !a7_attn_state_ok(op->src[5], CONVROW, K, mem_size) || !a7_attn_state_ok(op->src[4], (int64_t) SSMROW, K, mem_size) ||
        !a7_attn_state_ok(op->src[6], (int64_t) SSMROW, K, mem_size)) {
      return false;
    }
  } else if (op->src[1] == nullptr || !a7_attn_state_ok(op->src[3], CONVROW, K, mem_size) || !a7_attn_state_ok(op->src[4], (int64_t) SSMROW, K, mem_size)) {
    return false;
  }
  const int nsm = ggml_cuda_info().devices[device].nsm;
  if (nsm != 80 && nsm != 72) {
    return false;
  }
  const int threads = nsm == 80 ? 256 : 384;
  const size_t smem = (size_t) ACTB + (size_t) (threads / 32) * TILEB + sizeof(float) * SCRF + sizeof(A7Route);
  return smem <= ggml_cuda_info().devices[device].smpbo;
}

template <int TH>
static void a7_attn_launch(const a7ffn::A7P & P, cudaStream_t st, int grid, int device, int rq_ready) {
  using namespace a7ffn;
  const size_t dyn = (size_t) ACTB + (size_t) (TH / 32) * TILEB;
  static std::mutex mutex;
  static bool done[GGML_CUDA_MAX_DEVICES] = {};
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!done[device]) {
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_attn_stage<TH, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_attn_stage<TH, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_attn_stage<TH, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_attn_stage<TH, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      CUDA_CHECK(cudaFuncSetAttribute((const void *) a7_attn_stage<TH, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dyn));
      done[device] = true;
    }
  }
  const dim3 gd(grid), bl(TH);
  if (!rq_ready) {  // rq_ready = 1: the q8 of R * hc_attn_norm was made by the down stage of the ffn op of the previous layer
    a7_attn_rquant<<<NCH, 256, 0, st>>>(P);
  }
  a7_attn_stage<TH, 0><<<gd, bl, dyn, st>>>(P);  // hc1a
  a7_attn_stage<TH, 1><<<gd, bl, dyn, st>>>(P);  // hc2a
  a7_attn_stage<TH, 2><<<gd, bl, dyn, st>>>(P);  // proj
  a7_attn_stage<TH, 3><<<gd, bl, dyn, st>>>(P);  // gdn
  a7_attn_stage<TH, 4><<<gd, bl, dyn, st>>>(P);  // ssmout + hc_combine(attn)
  CUDA_CHECK(cudaGetLastError());
}

// check mode (mode 1): compare the A7 result with the production result (the same four ratios as the ffn op, on the delta of the layer) and the state rows of
// the 4 slots (A7 work copies against the production buffers), print A7CHECK_ATTN, then pass the production result on
static void a7_attn_cmp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
  using namespace a7ffn;
  const ggml_a7_attn_layer * L = nullptr;
  memcpy(&L, dst->op_params, sizeof(L));
  const int il = L->il;
  const int kv_head = ((const int32_t *) dst->op_params)[2], mem_size = ((const int32_t *) dst->op_params)[3];
  const ggml_tensor * r_a7 = dst->src[0], * r_prod = dst->src[1], * conv_w = dst->src[3], * ssm_w = dst->src[4], * conv_all = dst->src[5],
                    * ssm_all = dst->src[6], * res_in = dst->src[7];
  cudaStream_t st = ctx.stream();
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(st, &cap));
  if (cap != cudaStreamCaptureStatusNone) {
    GGML_ABORT("A7: LLAMA_A7_CHECK=1 needs GGML_CUDA_DISABLE_GRAPHS=1 (the check synchronizes the stream)");
  }
  a7_work * w = a7_get_work(ctx.device, st);
  const int nt = (int) dst->ne[2];
  const int n = (int) (ggml_nelements(dst) / nt);
  GGML_ASSERT((size_t) nt * 8 * sizeof(float) <= (size_t) T * 8 * sizeof(float));
  a7_ffn_check<<<nt, 1024, 0, st>>>((const float *) r_a7->data, (const float *) r_prod->data, (const float *) res_in->data, n, w->chk);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemsetAsync(w->schk, 0, 16 * sizeof(float), st));
  a7_attn_scheck<<<dim3(64, 4, 2), 256, 0, st>>>((const float *) ssm_w->data, (const float *) ssm_all->data, (const float *) conv_w->data,
                                                (const float *) conv_all->data, mem_size, kv_head, w->schk);
  CUDA_CHECK(cudaGetLastError());
  float h[64];
  float sc[16];
  int32_t src_row = -1;  // s_copy[0]: the state row the window read (slot = row / mem_size; slot >= 1 = after a partial rollback)
  CUDA_CHECK(cudaMemcpyAsync(h, w->chk, nt * 8 * sizeof(float), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaMemcpyAsync(sc, w->schk, sizeof(sc), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaMemcpyAsync(&src_row, dst->src[2]->data, sizeof(int32_t), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  // same ratios as A7CHECK of the ffn op: d_rel_l2 = |A7 - prod| / |prod delta|, tok_max = the worst token, d_max_rel = max|A7 - prod| / max|prod delta|,
  // dR = |prod delta| / |R in|
  double d2 = 0, db2 = 0, r2 = 0, md = 0, mdb = 0, tok = 0;
  for (int t = 0; t < nt; t++) {
    const float * q = h + t * 8;
    d2 += q[0]; db2 += q[1]; r2 += q[2]; md = fmax(md, (double) q[3]); mdb = fmax(mdb, (double) q[4]);
    tok = fmax(tok, q[1] > 0.f ? sqrt((double) q[0] / q[1]) : (q[0] > 0.f ? INFINITY : 0.0));
  }
  const double rel = db2 > 0 ? sqrt(d2 / db2) : (d2 > 0 ? INFINITY : 0.0);
  const double mrel = mdb > 0 ? md / mdb : (md > 0 ? INFINITY : 0.0);
  const double dR = r2 > 0 ? sqrt(db2 / r2) : 0.0;
  const bool fin = std::isfinite(d2) && std::isfinite(db2);
  const double nanv = std::nan("");
  // state: relative L2 per slot, the worst of the 4 slots (kind 0 = ssm, 1 = conv)
  double srel[2] = {0.0, 0.0};
  bool sfin = true;
  for (int kind = 0; kind < 2; kind++) {
    for (int s = 0; s < 4; s++) {
      const double sd = sc[(kind * 4 + s) * 2 + 0], sb = sc[(kind * 4 + s) * 2 + 1];
      if (!std::isfinite(sd) || !std::isfinite(sb)) { sfin = false; continue; }
      srel[kind] = fmax(srel[kind], sb > 0 ? sqrt(sd / sb) : (sd > 0 ? INFINITY : 0.0));
    }
  }
  fprintf(stderr, "A7CHECK_ATTN layer=%d dev=%d d_rel_l2=%.6g tok_max=%.6g d_max_rel=%.6g dR=%.6g s_rel_l2=%.6g c_rel_l2=%.6g src_row=%d src_slot=%d\n", il,
          ctx.device, fin ? rel : nanv, fin ? tok : nanv, fin ? mrel : nanv, dR, sfin ? srel[0] : nanv, sfin ? srel[1] : nanv, (int) src_row,
          mem_size > 0 && src_row >= 0 ? (int) (src_row / mem_size) : -1);
  CUDA_CHECK(cudaMemcpyAsync(dst->data, r_prod->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, st));
}

void ggml_cuda_op_a7_gdn_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
  using namespace a7ffn;
  GGML_ASSERT(ggml_cuda_a7_gdn_attn_supported(ctx.device, dst));
  const int32_t * prm = (const int32_t *) dst->op_params;
  if (prm[7] == 1) {
    a7_attn_cmp(ctx, dst);
    return;
  }
  const ggml_a7_attn_layer * L = nullptr;
  memcpy(&L, dst->op_params, sizeof(L));
  const int il = L->il;
  const int kv_head = prm[2], mem_size = prm[3], rq_ready = prm[5];
  float eps = 0.f;
  memcpy(&eps, &prm[6], sizeof(eps));
  const ggml_tensor * res = dst->src[0];
  cudaStream_t st = ctx.stream();

  {  // once per layer: where the op runs
    static std::mutex mutex;
    static std::set<std::pair<int, int>> seen;
    std::lock_guard<std::mutex> lock(mutex);
    if (seen.insert({il, ctx.device}).second) {
      fprintf(stderr, "A7: attn op on layer %d (device %d) rq_ready=%d\n", il, ctx.device, rq_ready);
    }
  }

  a7_work * w = a7_get_work(ctx.device, st);
  A7P P = w->p;
  P.norm_f = (const float *) L->hc_norm->data;
  P.hcd_f = (const uint8_t *) L->hc_down->data;
  P.hcu_f = (const uint8_t *) L->hc_up->data;
  P.inj_f = (const float *) L->hc_inject->data;
  P.wqkv = (const uint8_t *) L->wqkv->data;
  P.wz = (const uint8_t *) L->wqkv_gate->data;
  P.wout = (const uint8_t *) L->ssm_out->data;
  P.alpha = (const float *) L->ssm_alpha->data;
  P.beta = (const float *) L->ssm_beta->data;
  P.dt = (const float *) L->ssm_dt->data;
  P.Aa = (const float *) L->ssm_a->data;
  P.conv = (const float *) L->ssm_conv1d->data;
  P.snorm = (const float *) L->ssm_norm->data;
  P.norm_next = (const float *) L->hc_ffn_norm->data;   // ssmout makes the q8 of R_new * hc_ffn_norm
  P.R = (const float *) res->data;
  P.Rout = (float *) dst->data;
  P.s_copy = (const int *) dst->src[2]->data;           // read on the device by the kernels (never on the host)
  P.conv_all = (float *) dst->src[3]->data;
  P.ssm_all = (float *) dst->src[4]->data;
  P.kv_head = kv_head;
  P.mem_size = mem_size;
  P.eps = eps;

  if (rq_ready) {
    a7_tag_require(w, "attn", il, A7_SIDE_ATTN);
    if (a7_rq_verify_on()) a7_rq_verify_run(w, P, st, il, "attn", true);
  }

  if (ggml_cuda_info().devices[ctx.device].nsm == 80) {
    a7_attn_launch<256>(P, st, 80, ctx.device, rq_ready);
  } else {
    a7_attn_launch<384>(P, st, 72, ctx.device, rq_ready);
  }
  // ssmout made the q8 of R_new * hc_ffn_norm of this layer
  w->tag_il = il;
  w->tag_side = A7_SIDE_FFN;
}
