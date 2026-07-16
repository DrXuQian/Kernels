/*
 * Numerical test for the GGUF int4 path on the PPU Marlin (marlin_gguf_ppu.cuh): gs=32 (group_blocks=2),
 * per-32 in-tile scale. This is what Q4_0 needs -- Q4_0 is exactly (q-8)*d with a per-32-block d.
 *
 * The weight is generated as Bq[n][k] in [-8,7] (the dequantized-to-integer weight = nibble-8) plus a
 * per-32 scale d[n][kb], and the reference is C = A @ (Bq * d). The kernel is fed the Marlin B packing
 * (nibble = Bq+8) and the _scale_perm'd per-32 scales, and must reproduce the reference -- which it can
 * only do if the per-32 (gs=32) scale is applied correctly WITHIN each 64-k tile (the whole point of
 * the new in-tile scale path). A kernel that applied one scale per tile would fail on any case where
 * the two 32-blocks of a tile have different d.
 *
 * Build (box only, ppu001 -- uses ppu.mma): nvcc -O3 -std=c++17 -o t test_marlin_gguf.cu
 * PTX syntax check anywhere:                 nvcc -std=c++17 --expt-relaxed-constexpr -arch=sm_90 -ptx -c test_marlin_gguf.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include "marlin_gguf_ppu.cuh"

static int run_case(int M, int N, int K) {
  using namespace marlin_gguf_ppu;
  const int gs = 32, max_par = 128;
  const int NG = K / gs;                             // groups per column
  std::mt19937 rng(999 + M * 3 + N * 7 + K * 13);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> qd(-8, 7);

  std::vector<half> hA((size_t)M * K), hC((size_t)M * N, __float2half(0.f));
  for (auto& x : hA) x = __float2half(0.1f * nd(rng));

  // Bq[n][k] in [-8,7] and a DISTINCT per-32 scale d[n][kb] -- distinct so a per-tile (not per-32)
  // scale bug cannot pass: the two 32-blocks of one 64-k tile get different d.
  std::vector<int>  Bq((size_t)N * K);
  std::vector<half> d((size_t)N * NG);
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) Bq[(size_t)n * K + k] = qd(rng);
    for (int g = 0; g < NG; g++) d[(size_t)n * NG + g] = __float2half(0.4f + 0.6f * ((rng() % 100) / 100.0f));
  }

  // Reference: C[m][n] = sum_k A[m][k] * Bq[n][k] * d[n][k/32]
  std::vector<float> ref((size_t)M * N);
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      double acc = 0;
      for (int k = 0; k < K; k++)
        acc += (double)__half2float(hA[(size_t)m * K + k]) * Bq[(size_t)n * K + k] * __half2float(d[(size_t)n * NG + k / 32]);
      ref[(size_t)m * N + n] = acc;
    }

  // Pack B into Marlin's exact ktile-major per-lane layout (verbatim from test_marlin_classic_group).
  auto Bu = [&](int n, int k) { return (Bq[(size_t)n * K + k] + 8) & 0xf; };   // nibble = Bq + 8
  std::vector<int> hB((size_t)(K / 16) * (N * 16 / 32) * 4);
  for (int ktile = 0; ktile < K / 16; ktile++)
    for (int nblock = 0; nblock < N / 16; nblock++)
      for (int lane = 0; lane < 32; lane++) {
        int n = nblock * 16 + lane / 4, kb = ktile * 16 + (lane % 4) * 2, q = 0;
        q |= Bu(n, kb)         << 0;  q |= Bu(n, kb + 1)     << 16;
        q |= Bu(n, kb + 8)     << 4;  q |= Bu(n, kb + 9)     << 20;
        q |= Bu(n + 8, kb)     << 8;  q |= Bu(n + 8, kb + 1) << 24;
        q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
        size_t idx = (size_t)(N / 2) * ktile + (nblock / 4) * 32 + lane;
        hB[idx * 4 + (nblock % 4)] = q;
      }

  // Scales: plain [NG][N] then _scale_perm (the kernel's default read).
  std::vector<half> sPlain((size_t)NG * N), sDev((size_t)NG * N);
  for (int g = 0; g < NG; g++)
    for (int n = 0; n < N; n++) sPlain[(size_t)g * N + n] = d[(size_t)n * NG + g];
  marlin_permute_scales(sPlain.data(), sDev.data(), NG, N);

  half *dA, *dC, *dS; int *dB, *dWS;
  cudaMalloc(&dA, hA.size() * 2); cudaMalloc(&dB, hB.size() * 4); cudaMalloc(&dC, hC.size() * 2);
  cudaMalloc(&dS, sDev.size() * 2); cudaMalloc(&dWS, (N / 128 + 1) * max_par * 4);
  cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dS, sDev.data(), sDev.size() * 2, cudaMemcpyHostToDevice);
  cudaMemset(dWS, 0, (N / 128 + 1) * max_par * 4);

  int ret = marlin_cuda(dA, dB, dC, dS, M, N, K, dWS, gs);
  cudaError_t e = cudaDeviceSynchronize();
  if (ret || e) { printf("  M%-4d N%-5d K%-4d gs=32: ret=%d err=%s\n", M, N, K, ret, cudaGetErrorString(e)); return 2; }
  cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost);

  double maxabs = 0, refmax = 0;
  for (size_t i = 0; i < (size_t)M * N; i++) {
    maxabs = fmax(maxabs, fabs(__half2float(hC[i]) - ref[i]));
    refmax = fmax(refmax, fabs(ref[i]));
  }
  double rel = maxabs / (refmax + 1e-9);
  const bool bad = rel >= 3e-2;
  printf("  Q4_0/gs=32  M%-4d N%-5d K%-5d  rel %.2e (|ref|max=%.1f) -> %s\n", M, N, K, rel, refmax, bad ? "MISMATCH" : "MATCH");
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dS); cudaFree(dWS);
  return bad ? 1 : 0;
}

int main() {
  printf("GGUF int4 on PPU Marlin, gs=32 (per-32 in-tile scale):\n");
  int bad = 0;
  bad |= run_case(16,  128, 512);
  bad |= run_case(32,  256, 512);
  bad |= run_case(64,  256, 1024);
  bad |= run_case(128, 512, 1024);
  printf("%s\n", bad ? "SOME CASES FAILED" : "all gs=32 cases MATCH");
  return bad;
}
