#include <algorithm>
// standalone timing harness for the A7 FFN gate/up (S=8) and down (S=10) stages
#include "a7-gdn.cu"
#include <vector>
#include <random>
using namespace a7ffn;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
template <class T> T *dalloc(size_t n, bool rnd = true) { T *p; CK(cudaMalloc(&p, n * sizeof(T))); if (rnd) { std::vector<uint8_t> h(n * sizeof(T)); std::mt19937 g(1); for (auto &b : h) b = g(); CK(cudaMemcpy(p, h.data(), h.size(), cudaMemcpyHostToDevice)); } else CK(cudaMemset(p, 0, n * sizeof(T))); return p; }
__global__ void stream_read(const int4 *p, size_t n, int *out) { int4 a = {0,0,0,0}; for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { int4 v = __ldcs(p + i); a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w; } if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345) *out = 1; }
__global__ void zero_cnt(unsigned *c) { c[3000] = 0; for (int i = 0; i < 80; i++) c[4000 + i] = 0; }
int main(int argc, char **argv) {
  const int NU = argc > 1 ? atoi(argv[1]) : 35;
  CK(cudaSetDevice(argc > 2 ? atoi(argv[2]) : 0));
  A7P P = {};
  P.eg = dalloc<uint8_t>((size_t)NEXP * EXP_GU_BYTES); P.eu = dalloc<uint8_t>((size_t)NEXP * EXP_GU_BYTES); P.ed = dalloc<uint8_t>((size_t)NEXP * EXP_D_BYTES);
  // sanitize fp16 scales: random bytes can give inf/nan; that does not change timing, so leave them
  P.sg = dalloc<uint8_t>(640 * 2720); P.su = dalloc<uint8_t>(640 * 2720); P.sd = dalloc<uint8_t>(2560 * 680);
  P.mq = dalloc<int8_t>(T * D); P.md = dalloc<float>(T * 80);
  auto fixd = [](uint8_t *dp, size_t bytes, size_t blk) { std::vector<uint8_t> h(bytes); CK(cudaMemcpy(h.data(), dp, bytes, cudaMemcpyDeviceToHost)); std::mt19937 g(3);
    for (size_t o = 0; o + 1 < bytes; o += blk) { const uint16_t v = 0x1c00 + (g() & 0x3ff); h[o] = v & 0xff; h[o + 1] = v >> 8; } CK(cudaMemcpy(dp, h.data(), bytes, cudaMemcpyHostToDevice)); };
  fixd((uint8_t *)P.eg, (size_t)NEXP * EXP_GU_BYTES, 82); fixd((uint8_t *)P.eu, (size_t)NEXP * EXP_GU_BYTES, 82); fixd((uint8_t *)P.ed, (size_t)NEXP * EXP_D_BYTES, 18);
  fixd((uint8_t *)P.sg, 640 * 2720, 34); fixd((uint8_t *)P.su, 640 * 2720, 34); fixd((uint8_t *)P.sd, 2560 * 680, 34);
  { std::vector<float> h(T * 80, 0.01f); CK(cudaMemcpy((void *)P.md, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
  P.hbuf = dalloc<float>((MAXU + 1) * 4 * FF, false); P.hq = dalloc<int8_t>(MAXP * 640); P.hd = dalloc<float>(MAXP * 20);
  P.router = dalloc<float>((size_t)NEXP * D); P.sgw = dalloc<float>(D); P.norm_f = dalloc<float>(4 * D);
  P.R = dalloc<float>(16 * D); P.Rout = dalloc<float>(16 * D, false); P.inj_f = dalloc<float>(4 * HCD);
  P.cnt = dalloc<unsigned>(8192, false); P.balcnt = dalloc<int>(256, false); P.h2cnt = dalloc<int>(512, false);
  P.injp = dalloc<float>(64 * HCD, false); P.mixed = dalloc<float>(T * D, false); P.sgl = dalloc<float>(16, false);
  P.rr = dalloc<float>(64, false); P.part = dalloc<float>(4096 * 16, false); P.logits = dalloc<float>(T * NEXP, false);
  // routing table: NU distinct experts spread over the 512, each used by 1-4 tokens; 10 per token
  A7Route rt = {}; rt.N = NU; std::mt19937 g(7); std::vector<int> perm(NEXP); for (int i = 0; i < NEXP; i++) perm[i] = i; std::shuffle(perm.begin(), perm.end(), g);
  int np = 0; for (int n = 0; n < NU; n++) { rt.uni[n] = perm[n]; rt.umask[n] = 0; }
  for (int t = 0; t < T; t++) { int c = 0; for (int k = 0; c < TOPK && k < 4 * NU; k++) { int n = (t * 9 + k) % NU; if (!((rt.umask[n] >> t) & 1)) { rt.umask[n] |= 1 << t; c++; } } }
  for (int n = 0; n < NU; n++) rt.umask[n] = getenv("ONE_TOKEN") ? 1 : (rt.umask[n] ? rt.umask[n] : 1);
  for (int n = 0; n < NU; n++) { rt.pref[n] = np; for (int t = 0; t < T; t++) if ((rt.umask[n] >> t) & 1) { if (np < MAXP) { rt.pn[np] = n; rt.pt[np] = t; } np++; } }
  rt.pref[NU] = np; for (int t = 0; t < T; t++) { rt.pn[np] = NU; rt.pt[np] = t; np++; } rt.pref[NU + 1] = np;
  for (int i = 0; i < (MAXU + 1) * 4; i++) rt.wsel[i] = -1; for (int i = 0; i < T * TOPK; i++) rt.wk[i] = 0.1f;
  printf("NU=%d pairs=%d (MAXP %d)\n", NU, np, MAXP);
  CK(cudaMalloc(&P.route, sizeof(A7Route))); CK(cudaMemcpy(P.route, &rt, sizeof(rt), cudaMemcpyHostToDevice));
  const int smem = 59392;
  CK(cudaFuncSetAttribute(a7_ffn_stage<256, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CK(cudaFuncSetAttribute(a7_ffn_stage<256, 10>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto run = [&](auto kern, const char *name, double mb) {
    for (int i = 0; i < 20; i++) { zero_cnt<<<1, 1>>>(P.cnt); kern<<<80, 256, smem>>>(P); }
    CK(cudaDeviceSynchronize());
    const int IT = 300; float tot = 0;
    for (int i = 0; i < IT; i++) { zero_cnt<<<1, 1>>>(P.cnt); cudaEventRecord(e0); kern<<<80, 256, smem>>>(P); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); tot += ms; }
    CK(cudaGetLastError());
    const double us = 1000.0 * tot / IT;
    printf("%-10s %7.1f us  %6.1f MB  %6.0f GB/s\n", name, us, mb, mb / us * 1000.0);
  };
  { // router stage: with the inline routing-table build (last block) and without it
    auto fixf = [](const float *p, size_t n) { std::vector<float> h(n); std::mt19937 g(5); for (auto &x : h) x = ((int)(g() % 2001) - 1000) * 1e-3f; CK(cudaMemcpy((void *)p, h.data(), n * 4, cudaMemcpyHostToDevice)); };
    fixf(P.router, (size_t)NEXP * D); fixf(P.sgw, D); fixf(P.mixed, T * D); fixf(P.norm_f, 4 * D); fixf(P.R, 16 * D); fixf(P.inj_f, 4 * HCD);
    CK(cudaFuncSetAttribute(a7_ffn_stage<256, 7>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    for (int mode = 0; mode < 4; mode++) {
      P.route_inline = mode < 2 ? 1 : 0; P.route_ref = mode & 1; P.router16 = nullptr; P.logits16 = nullptr;
      for (int i = 0; i < 20; i++) a7_ffn_stage<256, 7><<<80, 256, smem>>>(P);
      CK(cudaDeviceSynchronize()); float tot = 0;
      for (int i = 0; i < 300; i++) { cudaEventRecord(e0); a7_ffn_stage<256, 7><<<80, 256, smem>>>(P); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); tot += ms; }
      CK(cudaGetLastError());
      printf("router stage %-28s %6.1f us\n", mode == 0 ? "inline build (fast)" : mode == 1 ? "inline build (ref, production)" : "no build", 1000.0 * tot / 300);
      if (mode == 2) break;
    }
  }
  run(a7_ffn_stage<256, 8>, "gateup", (NU * 2.0 * EXP_GU_BYTES + 2 * 640 * 2720) / 1e6);
  {
    const int smem512 = ACTB_GU + 16 * TILEB;
    CK(cudaFuncSetAttribute(a7_ffn_stage<512, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem512));
    for (int i = 0; i < 20; i++) { zero_cnt<<<1, 1>>>(P.cnt); a7_ffn_stage<512, 8><<<80, 512, smem512>>>(P); }
    CK(cudaDeviceSynchronize());
    float tot = 0; for (int i = 0; i < 300; i++) { zero_cnt<<<1, 1>>>(P.cnt); cudaEventRecord(e0); a7_ffn_stage<512, 8><<<80, 512, smem512>>>(P); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); tot += ms; }
    CK(cudaGetLastError()); const double us = 1000.0 * tot / 300, mb = (NU * 2.0 * EXP_GU_BYTES + 2 * 640 * 2720) / 1e6;
    printf("%-10s %7.1f us  %6.1f MB  %6.0f GB/s\n", "gateup512", us, mb, mb / us * 1000.0);
  }
  run(a7_ffn_stage<256, 10>, "down", (NU * (double)EXP_D_BYTES + 2560 * 680) / 1e6);
  { // practical streaming-read ceiling over a buffer of the gate/up size, at several grid shapes
    const size_t n = (size_t)(NU * 2.0 * EXP_GU_BYTES) / 16; int *o; CK(cudaMalloc(&o, 4));
    for (int cfg = 0; cfg < 3; cfg++) { int gx = cfg == 0 ? 80 : cfg == 1 ? 160 : 640, bx = cfg == 0 ? 256 : cfg == 1 ? 512 : 256;
      for (int i = 0; i < 10; i++) stream_read<<<gx, bx>>>((const int4 *)P.eg, n, o);
      cudaEventRecord(e0); for (int i = 0; i < 100; i++) stream_read<<<gx, bx>>>((const int4 *)P.eg, n, o); cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); printf("stream_read grid %d x %d: %6.0f GB/s (%.1f us for %.1f MB)\n", gx, bx, n * 16 / (ms / 100 * 1e-3) / 1e9, ms * 10, n * 16 / 1e6); }
  }
  { // correctness: hbuf of the 256-thread kernel vs the 512-thread kernel
    const size_t nh = (size_t)(MAXU + 1) * 4 * FF; std::vector<float> h0(nh), h1(nh);
    CK(cudaMemset((void *)P.hbuf, 0, nh * 4)); zero_cnt<<<1, 1>>>(P.cnt); a7_ffn_stage<256, 8><<<80, 256, smem>>>(P); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h0.data(), P.hbuf, nh * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemset((void *)P.hbuf, 0, nh * 4)); zero_cnt<<<1, 1>>>(P.cnt); a7_ffn_stage<512, 8><<<80, 512, ACTB_GU + 16 * TILEB>>>(P); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h1.data(), P.hbuf, nh * 4, cudaMemcpyDeviceToHost));
    double md = 0, mx = 0; size_t ndiff = 0, nz = 0; for (size_t i = 0; i < nh; i++) { const double d = fabs(h0[i] - h1[i]); if (d > 0) ndiff++; if (h0[i] != 0) nz++; md = fmax(md, d); mx = fmax(mx, fabs(h0[i])); }
    std::vector<double> rel; for (size_t i = 0; i < nh; i++) if (fabs(h0[i]) > 1e-3 * mx) rel.push_back(fabs(h0[i] - h1[i]) / fabs(h0[i])); std::sort(rel.begin(), rel.end());
    if (!rel.empty()) printf("relative diff (|v| > 1e-3 max): median %.2e  p99 %.2e  max %.2e  (n=%zu)\n", rel[rel.size()/2], rel[rel.size()*99/100], rel.back(), rel.size());
    printf("check 256 vs 512: nonzero %zu, differing %zu, max |diff| %.3g, max |val| %.3g\n", nz, ndiff, md, mx);
  }
  return 0;
}
