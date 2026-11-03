#include "a7-gdn.cu"
#include <vector>
#include <random>
using namespace a7ffn;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
template <class T> T *dalloc(size_t n, int mode = 1) { T *p; CK(cudaMalloc(&p, n * sizeof(T))); std::vector<uint8_t> h(n * sizeof(T)); std::mt19937 g(n);
  if (mode == 1) for (auto &b : h) b = g(); else if (mode == 2) { float *f = (float *)h.data(); for (size_t i = 0; i < n; i++) f[i] = (int)(g() % 2001 - 1000) * 1e-3f; }
  CK(cudaMemcpy(p, h.data(), h.size(), cudaMemcpyHostToDevice)); return p; }
static void fixd(uint8_t *dp, size_t bytes, size_t blk) { std::vector<uint8_t> h(bytes); CK(cudaMemcpy(h.data(), dp, bytes, cudaMemcpyDeviceToHost)); std::mt19937 g(3);
  for (size_t o = 0; o + 1 < bytes; o += blk) { const uint16_t v = 0x1c00 + (g() & 0x3ff); h[o] = v & 0xff; h[o + 1] = v >> 8; } CK(cudaMemcpy(dp, h.data(), bytes, cudaMemcpyHostToDevice)); }
__global__ void zc(unsigned *c) { c[3001] = 0; }
int main(int argc, char **argv) {
  CK(cudaSetDevice(argc > 1 ? atoi(argv[1]) : 0));
  A7P P = {};
  uint8_t *wqkv = dalloc<uint8_t>((size_t)QKVC * 2720), *wz = dalloc<uint8_t>((size_t)INNER * 2720); fixd(wqkv, (size_t)QKVC * 2720, 34); fixd(wz, (size_t)INNER * 2720, 34);
  P.wqkv = wqkv; P.wz = wz;
  P.alpha = dalloc<float>(NVH * D, 2); P.beta = dalloc<float>(NVH * D, 2); P.mixed = dalloc<float>(T * D, 2);
  P.mq = dalloc<int8_t>(T * D); { float *md = dalloc<float>(T * 80, 2); P.md = md; std::vector<float> h(T * 80, 0.01f); CK(cudaMemcpy(md, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
  const int mem_size = 8; P.mem_size = mem_size; P.kv_head = 1;
  const size_t ncs = (size_t)(4 * mem_size + 2) * CONVROW; P.conv_all = dalloc<float>(ncs, 2); P.conv = dalloc<float>((size_t)QKVC * 4, 2);
  int *sc = dalloc<int>(1, 0); CK(cudaMemset(sc, 0, 4)); P.s_copy = sc;
  P.qkv_o = dalloc<float>(4 * QKVC, 0); P.z_o = dalloc<float>(4 * INNER, 0); P.a_o = dalloc<float>(4 * NVH, 0); P.b_o = dalloc<float>(4 * NVH, 0);
  P.cnt = dalloc<unsigned>(8192, 0); CK(cudaMemset(P.cnt, 0, 8192 * 4));
  std::vector<float> conv0(ncs); CK(cudaMemcpy(conv0.data(), P.conv_all, ncs * 4, cudaMemcpyDeviceToHost));
  const int dyn = ACTB + 8 * TILEB, dyn512 = ACTB_GU + 16 * TILEB;
  CK(cudaFuncSetAttribute(a7_attn_stage<256, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
  CK(cudaFuncSetAttribute(a7_attn_stage<512, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn512));
  auto outs = [&](std::vector<float> &v) { v.resize(4 * QKVC + 4 * INNER + 8 * NVH + ncs); float *p = v.data();
    CK(cudaMemcpy(p, P.qkv_o, 4 * QKVC * 4, cudaMemcpyDeviceToHost)); p += 4 * QKVC; CK(cudaMemcpy(p, P.z_o, 4 * INNER * 4, cudaMemcpyDeviceToHost)); p += 4 * INNER;
    CK(cudaMemcpy(p, P.a_o, 4 * NVH * 4, cudaMemcpyDeviceToHost)); p += 4 * NVH; CK(cudaMemcpy(p, P.b_o, 4 * NVH * 4, cudaMemcpyDeviceToHost)); p += 4 * NVH;
    CK(cudaMemcpy(p, P.conv_all, ncs * 4, cudaMemcpyDeviceToHost)); };
  std::vector<float> o0, o1;
  CK(cudaMemcpy(P.conv_all, conv0.data(), ncs * 4, cudaMemcpyHostToDevice)); zc<<<1,1>>>(P.cnt); a7_attn_stage<256, 2><<<80, 256, dyn>>>(P); CK(cudaDeviceSynchronize()); outs(o0);
  CK(cudaMemcpy(P.conv_all, conv0.data(), ncs * 4, cudaMemcpyHostToDevice)); zc<<<1,1>>>(P.cnt); a7_attn_stage<512, 2><<<80, 512, dyn512>>>(P); CK(cudaDeviceSynchronize()); outs(o1);
  size_t nd = 0, nz = 0; for (size_t i = 0; i < o0.size(); i++) { if (o0[i] != o1[i]) nd++; if (o0[i] != 0) nz++; }
  printf("check 256 vs 512: %zu values (%zu nonzero), %zu differ\n", o0.size(), nz, nd);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int v = 0; v < 2; v++) for (int rep = 0; rep < 2; rep++) {
    float tot = 0; for (int i = 0; i < 300; i++) { zc<<<1,1>>>(P.cnt); cudaEventRecord(e0); if (v == 0) a7_attn_stage<256, 2><<<80, 256, dyn>>>(P); else a7_attn_stage<512, 2><<<80, 512, dyn512>>>(P); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); tot += ms; }
    const double us = 1000.0 * tot / 300, mb = ((QKVC + INNER) * 2720.0 + 2 * NVH * D * 4) / 1e6;
    printf("proj %d threads: %6.1f us  %5.1f MB  %5.0f GB/s\n", v ? 512 : 256, us, mb, mb / us * 1000);
  }
}
