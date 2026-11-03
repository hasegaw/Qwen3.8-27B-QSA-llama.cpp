#pragma once
// A7 P1b (2026-10-16): the routing table of GGML_OP_A7_GDN_FFN, shared by a7-gdn.cu and the check / timing tool (research .../a7/route-test/route-test.cu).
//
// Input: the router logits of the 4 tokens of the verify window, logits[T][NEXP] (global memory). Output: the A7Route table (global memory): the union of the
// experts of the 4 tokens (uni / umask), the (expert, token) pair tables (pref / pn / pt), wsel (union index, token) -> slot in wk, and the weights wk.
//
// Both functions are called by ALL threads of the block (128, 256 or 384 threads). Only warps 0..3 work (warp = token); the other warps only take part in
// __syncthreads(). `sm` is a work area in shared memory (A7RouteSmem, 16 B aligned, at least sizeof(A7RouteSmem) bytes). The caller may use the stage's dynamic
// shared memory for it. The functions depend on nothing else of a7-gdn.cu. The table is complete when the kernel (or the stage) ends; there is no barrier
// at the end of the functions (the caller's kernel boundary orders the global writes).
//
//   a7_route_build      the fast build (this file's 1-2): rank based, one __syncthreads()
//   a7_route_build_ref  the previous build (10 rounds of argmax over the warp, 4 __syncthreads()), the body of the old a7_ffn_route kernel. It is the reference
//                       of the test: the selected experts and every table must be bit identical to a7_route_build (wk: a few ulp).
//
// Selection rule shared by both: key e = expf(logit - max(logit of the token)); expert a beats expert b iff e_a > e_b, or e_a == e_b and a < b (a strict
// total order when no e is NaN). The top 10 by this order are the experts of the token; slot k of the token is the k-th best (k = 0 the best).
// Differences of a7_route_build to the old build (and to a7_route_build_ref), on purpose:
//   - a NaN logit is replaced by -INFINITY when it is read, so it is never chosen ahead of a finite logit (old: a NaN e is never chosen, and when all
//     e are NaN the old code selected expert 511 ten times). A NaN e (-INF - -INF, +INF - +INF) is replaced by 0. The order stays total.
//   - the softmax sum zs is not computed (it cancels in the weights, see below).
//   - the weight is e / sum(e of the 10) (old: (e / zs) / max(sum(e / zs), 6.1035e-5)): equal up to a few ulp.
//   - when all 10 e are 0 (only possible with non-finite logits) the weights are 1/10 instead of NaN.
#include <cuda_runtime.h>

#ifndef A7_FULL
#define A7_FULL 0xffffffffu
#endif
// instrumentation hooks of the test tool (route-test.cu defines them before including this file); empty in production
#ifndef A7_ROUTE_MARK
#define A7_ROUTE_MARK(i) ((void)0)
#endif
#ifndef A7_ROUTE_SLOW
#define A7_ROUTE_SLOW() ((void)0)
#endif

namespace a7ffn {

constexpr int T = 4;                       // verify window tokens
constexpr int NEXP = 512, TOPK = 10;
constexpr int MAXU = 40;                   // max size of the union of the experts of the 4 tokens
constexpr int MAXP = 48;                   // max (expert, token) pairs (<= 40 + 4 of the shared expert), padded

// routing table built by a7_route_build (global work area), copied to shared memory by the stages that use it
struct A7Route {
  int N;                                   // size of the union
  int uni[MAXU];                           // union index -> expert
  int umask[MAXU];                         // union index -> bit mask of the tokens that selected it
  int pref[MAXU + 2];                      // (expert, token) pairs before union index n; pref[N] = first pair of the shared expert, pref[N + 1] = total
  int pn[MAXP], pt[MAXP];                  // pair -> union index (N for the shared expert), token
  int wsel[(MAXU + 1) * 4];                // (union index, token) -> slot t * TOPK + k in wk, or -1
  float wk[T * TOPK];                      // renormalized top-k weights of token t at [t * TOPK + k]
};

// work area (shared memory) of the two builds: 1,616 B (1,604 B of members, aligned to 16)
struct __align__(16) A7RouteSmem {
  unsigned msk[4][16];                     // per token: 512-bit mask of the selected experts
  unsigned umk[16];                        // union (the reference build only)
  int pre[17];                             // exclusive prefix of the union's popcounts per mask word (the reference build only)
  int s_um[MAXU + 8];                      // umask copy
  float cv[4][32];                         // per token: the candidates (value, expert) of a7_route_build, 32 slots
  int ci[4][32];
};

static __device__ __forceinline__ float rt_warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(A7_FULL, v, o));
  return v;
}
static __device__ __forceinline__ float rt_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(A7_FULL, v, o);
  return v;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------
// the fast build. PROF is for the test tool only (0 in production): 1 = clock64 marks, 2 = count the tokens that take the slow path.
//
// Selection of the top 10 of a token (a warp; lane l owns the experts l + 32k, k = 0..15; "beats" is the total order above):
//   1. rep(l) = the best of the 16 experts of lane l.
//   2. rank(l) = the number of reps that beat rep(l) (31 shuffles). The reps are 32 different experts, so the ranks are 0..31, each once.
//      tau = the rep with rank 9.
//   3. candidates = the experts that beat tau, plus tau itself. The top 10 are all candidates: rank 0..9 reps are 10 different experts, each is tau or beats
//      it. Any expert y that tau beats is beaten by tau and by the 9 reps that beat tau (the order is transitive), i.e. by >= 10 experts, so y is not in the top 10.
//      For an expert with e equal to tau's e, "beats tau" means a smaller expert number, so such an expert is a candidate iff its number <= tau's number: the
//      ties are cut exactly as the old build does (smaller number first). The number of candidates C is >= 10. They are packed into the shared table with
//      __ballot_sync + popc (in increasing k, then lane).
//   4. C <= 32: lane j < C takes candidate j and counts the candidates that beat it (the same order, ties by the expert number) = its rank among the
//      candidates. The candidates with rank < 10 are the top 10, and rank = slot k (the old build's "selected in round r" = slot r).
//      C > 32 (the top experts crowd into a few lanes): the old build's 10 rounds of argmax over the warp, on the same keys.
//   5. weight = e / (sum of the 10 selected e). The production formula is p / max(sum p, 6.1035e-5) with p = e / zs; zs cancels. The 10 selected p sum to
//      >= 10 / 512 (the 10 largest of 512 probabilities that sum to 1 are at least the average of all), so the lower bound 6.1035e-5 never applies.
// Then the union as before: 512-bit masks with atomicOr, prefix over the 16 words, the list, wsel, the pair table. The masks are zeroed by the warp that
// ORs into them (no barrier before), the prefix is computed by each warp itself from the masks (no barrier after), and the pair table needs only a
// __syncwarp(): a single __syncthreads() (masks complete) in the whole function.
template <int PROF = 0>
static __device__ __forceinline__ void a7_route_build(const float *logits, A7Route *rt, A7RouteSmem *sm) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  A7_ROUTE_MARK(0);
  for (int i = tid; i < (MAXU + 1) * 4; i += blockDim.x) rt->wsel[i] = -1;   // complete long before the first write to wsel (after the barrier)
  bool sel = false;      // this lane holds a selected expert
  int slot = 0, myidx = 0;
  float myw = 0.f;
  if (warp < 4) {
    if (lane < 16) sm->msk[warp][lane] = 0u;
    const float *lg = logits + warp * NEXP;
    float e[16];
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < 16; k++) {
      float l = __ldcg(lg + lane + 32 * k);   // the last arriver of the router stage reads what other blocks wrote: not through L1
      l = (l != l) ? -INFINITY : l;
      e[k] = l;
      mx = fmaxf(mx, l);
    }
    mx = rt_warp_max(mx);
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const float x = expf(e[k] - mx);
      e[k] = (x == x) ? x : 0.f;
    }
    A7_ROUTE_MARK(1);
    // 1. rep of the lane (strict >: the first k, i.e. the smaller expert number, wins a tie)
    float rv = e[0];
    int ri = lane;
#pragma unroll
    for (int k = 1; k < 16; k++) {
      if (e[k] > rv) { rv = e[k]; ri = lane + 32 * k; }
    }
    // 2. rank of the rep among the 32 reps, tau = the rep of rank 9
    int rank = 0;
#pragma unroll
    for (int d = 1; d < 32; d++) {
      const float ov = __shfl_sync(A7_FULL, rv, (lane + d) & 31);
      const int oi = __shfl_sync(A7_FULL, ri, (lane + d) & 31);
      rank += (ov > rv || (ov == rv && oi < ri)) ? 1 : 0;
    }
    const int tl = __ffs((int)__ballot_sync(A7_FULL, rank == 9)) - 1;   // exactly one lane (the ranks are a permutation of 0..31)
    const float tv = __shfl_sync(A7_FULL, rv, tl);
    const int ti = __shfl_sync(A7_FULL, ri, tl);
    // 3. candidates: beat tau or are tau
    int base = 0;
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const int idx = lane + 32 * k;
      const bool c = (e[k] > tv) || (e[k] == tv && idx <= ti);
      const unsigned b = __ballot_sync(A7_FULL, c);
      if (c) {
        const int pos = base + __popc(b & ((1u << lane) - 1u));
        if (pos < 32) { sm->cv[warp][pos] = e[k]; sm->ci[warp][pos] = idx; }
      }
      base += __popc(b);
    }
    const int C = base;   // warp uniform, >= 10
    float ev = 0.f;
    if (C <= 32) {
      // 4. rank among the candidates
      __syncwarp();
      const float cvj = lane < C ? sm->cv[warp][lane] : -1.f;
      const int cij = lane < C ? sm->ci[warp][lane] : 0x7fffffff;
      int r = 0;
#pragma unroll 4
      for (int i = 0; i < C; i++) {
        const float ov = sm->cv[warp][i];
        const int oi = sm->ci[warp][i];
        r += (ov > cvj || (ov == cvj && oi < cij)) ? 1 : 0;
      }
      sel = lane < C && r < TOPK;
      slot = r;
      myidx = cij;
      ev = cvj;
    } else {
      // 4'. the old build: 10 rounds of argmax over the warp (ties: smaller expert first)
      A7_ROUTE_SLOW();
      unsigned taken = 0;
#pragma unroll 1
      for (int r = 0; r < TOPK; r++) {
        float bv = -1.f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int k = 0; k < 16; k++) {
          if (!((taken >> k) & 1u) && e[k] > bv) { bv = e[k]; bi = lane + 32 * k; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          const float ov = __shfl_xor_sync(A7_FULL, bv, o);
          const int oi = __shfl_xor_sync(A7_FULL, bi, o);
          if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
        if (lane == r) { ev = bv; myidx = bi; }
      }
      sel = lane < TOPK;
      slot = lane;
    }
    A7_ROUTE_MARK(2);
    // 5. weights
    const float S = rt_warp_sum(sel ? ev : 0.f);
    myw = S > 0.f ? __fdiv_rn(ev, S) : 1.f / (float)TOPK;
    __syncwarp();   // the zeroing of the masks (lanes 0..15) is ordered before the atomicOr of any lane
    if (sel) atomicOr(&sm->msk[warp][myidx >> 5], 1u << (myidx & 31));
    A7_ROUTE_MARK(3);
  }
  __syncthreads();   // the masks of the 4 tokens are complete
  A7_ROUTE_MARK(4);
  if (warp < 4) {
    unsigned m0 = 0, m1 = 0, m2 = 0, m3 = 0;
    if (lane < 16) { m0 = sm->msk[0][lane]; m1 = sm->msk[1][lane]; m2 = sm->msk[2][lane]; m3 = sm->msk[3][lane]; }
    const unsigned u = m0 | m1 | m2 | m3;   // union word `lane` (lanes 0..15)
    const int c = __popc(u);
    int inc = c;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
      const int v = __shfl_up_sync(A7_FULL, inc, o);
      if (lane >= o) inc += v;
    }
    const int pre_l = inc - c;                         // first union index of word `lane`
    const int total = __shfl_sync(A7_FULL, inc, 15);   // N
    A7_ROUTE_MARK(5);
    {  // wsel / wk of this token's selected experts
      const int x = sel ? (myidx >> 5) : 0;
      const int prex = __shfl_sync(A7_FULL, pre_l, x);
      const unsigned ux = __shfl_sync(A7_FULL, u, x);
      if (sel) {
        const int b = myidx & 31;
        const int idx = prex + __popc(ux & ((1u << b) - 1u));
        rt->wsel[idx * 4 + warp] = warp * TOPK + slot;
        rt->wk[warp * TOPK + slot] = myw;
      }
    }
    if (warp == 0) {
      if (lane < 16) {   // the union list, in increasing expert order
        unsigned uu = u;
        int idx = pre_l;
        while (uu) {
          const int b = __ffs(uu) - 1;
          uu &= uu - 1;
          const int m = (int)((m0 >> b) & 1u) | (int)(((m1 >> b) & 1u) << 1) | (int)(((m2 >> b) & 1u) << 2) | (int)(((m3 >> b) & 1u) << 3);
          rt->uni[idx] = lane * 32 + b;
          rt->umask[idx] = m;
          sm->s_um[idx] = m;
          idx++;
        }
      }
      if (lane == 0) rt->N = total;
      __syncwarp();
      A7_ROUTE_MARK(6);
      // pair tables: entries n = 0 .. N (N = the shared expert, all 4 tokens), two per lane
      const int N = total;
      const int n0 = 2 * lane, n1 = 2 * lane + 1;
      const int pm0 = n0 < N ? sm->s_um[n0] : (n0 == N ? 15 : 0);
      const int pm1 = n1 < N ? sm->s_um[n1] : (n1 == N ? 15 : 0);
      const int c0 = __popc(pm0), c1 = __popc(pm1);
      int pinc = c0 + c1;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(A7_FULL, pinc, o);
        if (lane >= o) pinc += v;
      }
      const int pbase = pinc - (c0 + c1);
      if (n0 <= N) rt->pref[n0] = pbase;
      if (n1 <= N) rt->pref[n1] = pbase + c0;
      int p = pbase;
#pragma unroll
      for (int t = 0; t < T; t++) if ((pm0 >> t) & 1) { rt->pn[p] = n0; rt->pt[p] = t; p++; }
#pragma unroll
      for (int t = 0; t < T; t++) if ((pm1 >> t) & 1) { rt->pn[p] = n1; rt->pt[p] = t; p++; }
      if (lane == 31) rt->pref[N + 1] = pinc;
    }
  }
  A7_ROUTE_MARK(7);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------
// the previous build (the body of the old a7_ffn_route kernel, calculation unchanged; made a function for all block sizes: only warps 0..3 compute).
// Softmax over the 512 router logits, top-10 (ties: smaller expert first), the 10 weights divided by their sum (lower bound 6.1035e-5,
// llama-graph.cpp build_moe_ffn), the union of the 4 tokens in increasing expert order and the tables.
static __device__ __forceinline__ void a7_route_build_ref(const float *logits, A7Route *rt, A7RouteSmem *sm) {
  unsigned (*msk)[16] = sm->msk;   // per token: 512-bit mask of the selected experts
  unsigned *umk = sm->umk;         // union
  int *pre = sm->pre;              // exclusive prefix of the union's popcounts per mask word; pre[16] = N
  int *s_um = sm->s_um;            // umask copy
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x < 64) (&msk[0][0])[threadIdx.x] = 0u;
  for (int i = threadIdx.x; i < (MAXU + 1) * 4; i += blockDim.x) rt->wsel[i] = -1;
  __syncthreads();
  const int tk = warp;
  int mysel = 0;
  float wgt = 0.f;
  if (warp < 4) {
    const float *lg = logits + tk * NEXP;
    float e[16];
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < 16; k++) { e[k] = __ldcg(lg + lane + 32 * k); mx = fmaxf(mx, e[k]); }
    mx = rt_warp_max(mx);
    float zs = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) { e[k] = expf(e[k] - mx); zs += e[k]; }
    zs = rt_warp_sum(zs);
    unsigned taken = 0;
    float tot = 0.f, myw = 0.f;
#pragma unroll 1
    for (int r = 0; r < TOPK; r++) {
      float bv = -1.f;
      int bi = 0x7fffffff;
#pragma unroll
      for (int k = 0; k < 16; k++) {
        if (!((taken >> k) & 1u) && e[k] > bv) { bv = e[k]; bi = lane + 32 * k; }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(A7_FULL, bv, o);
        const int oi = __shfl_xor_sync(A7_FULL, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      bi = min(bi, NEXP - 1);  // review 10-16 m-2: all-NaN logits leave bi unset
      if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
      if (lane == r) { myw = bv; mysel = bi; }
      tot += bv;
    }
    // probabilities p = e / zs; weight = p / max(sum of the 10 selected p, 6.103515625e-5)
    const float ssum = fmaxf(tot / zs, 6.103515625e-5f);
    wgt = (myw / zs) / ssum;
    if (lane < TOPK) atomicOr(&msk[tk][mysel >> 5], 1u << (mysel & 31));
  }
  __syncthreads();
  if (warp == 0) {
    unsigned u = 0;
    int c = 0;
    if (lane < 16) { u = msk[0][lane] | msk[1][lane] | msk[2][lane] | msk[3][lane]; umk[lane] = u; c = __popc(u); }
    int inc = c;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
      const int v = __shfl_up_sync(A7_FULL, inc, o);
      if (lane >= o) inc += v;
    }
    if (lane < 16) pre[lane] = inc - c;
    const int total = __shfl_sync(A7_FULL, inc, 15);
    if (lane == 0) { pre[16] = total; rt->N = total; }
  }
  __syncthreads();
  if (threadIdx.x < 16) {
    const int x = threadIdx.x;
    unsigned u = umk[x];
    int idx = pre[x];
    while (u) {
      const int b = __ffs(u) - 1;
      u &= u - 1;
      const int m = (int)((msk[0][x] >> b) & 1u) | (int)(((msk[1][x] >> b) & 1u) << 1) | (int)(((msk[2][x] >> b) & 1u) << 2) | (int)(((msk[3][x] >> b) & 1u) << 3);
      rt->uni[idx] = x * 32 + b;
      rt->umask[idx] = m;
      s_um[idx] = m;
      idx++;
    }
  }
  if (warp < 4 && lane < TOPK) {
    const int x = mysel >> 5, b = mysel & 31;
    const int idx = pre[x] + __popc(umk[x] & ((1u << b) - 1u));
    rt->wsel[idx * 4 + tk] = tk * TOPK + lane;
    rt->wk[tk * TOPK + lane] = wgt;
  }
  __syncthreads();
  if (warp == 0) {  // pair tables: entries n = 0 .. N (N = the shared expert, all 4 tokens), two per lane
    const int N = pre[16];
    const int n0 = 2 * lane, n1 = 2 * lane + 1;
    const int m0 = n0 < N ? s_um[n0] : (n0 == N ? 15 : 0);
    const int m1 = n1 < N ? s_um[n1] : (n1 == N ? 15 : 0);
    const int c0 = __popc(m0), c1 = __popc(m1);
    int inc = c0 + c1;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(A7_FULL, inc, o);
      if (lane >= o) inc += v;
    }
    const int base = inc - (c0 + c1);
    if (n0 <= N) rt->pref[n0] = base;
    if (n1 <= N) rt->pref[n1] = base + c0;
    int p = base;
#pragma unroll
    for (int t = 0; t < T; t++) if ((m0 >> t) & 1) { rt->pn[p] = n0; rt->pt[p] = t; p++; }
#pragma unroll
    for (int t = 0; t < T; t++) if ((m1 >> t) & 1) { rt->pn[p] = n1; rt->pt[p] = t; p++; }
    const int total = __shfl_sync(A7_FULL, inc, 31);
    if (lane == 0) rt->pref[N + 1] = total;
  }
}

}  // namespace a7ffn
