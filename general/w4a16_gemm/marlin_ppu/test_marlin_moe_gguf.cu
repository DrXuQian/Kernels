// Q4_K MoE grouped GEMM vs a CPU reference over the routing.
//
// What this has to catch, and what a naive test would not: the padding rule. If an expert's row segment
// does not fill whole m-tiles, two experts share a tile and the tile applies ONE expert's B to both --
// wrong rows, no error, and it only shows when a segment length is not a multiple of the m-tile. So the
// routing here is deliberately RAGGED (random experts, so segments land on arbitrary lengths) rather than
// an even split, which would pad to nothing and test nothing.
#include "marlin_moe_gguf_ppu.cuh"
#include <cstdio>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static int run(int num_rows, int num_experts, int N, int K, bool affine) {
  using namespace marlin_moe_gguf_ppu;
  const int gs = 32, G = K / gs, max_par = 128;
  // marlin_cuda picks thread_m_blocks by prob_m; the plan must pad to the LARGEST it can choose, or a
  // segment sized for a small tile gets scheduled with a big one. 4 m-blocks = 64 rows is that bound.
  const int m_tile = 64;

  std::mt19937 rng(31 + num_rows * 7 + num_experts * 13 + N);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> qd(-8, 7), ed(0, num_experts - 1);

  std::vector<int> rows_expert(num_rows);
  for (auto& e : rows_expert) e = ed(rng);
  std::vector<half> hA((size_t) num_rows * K);
  for (auto& x : hA) x = __float2half(0.1f * nd(rng));
  std::vector<int> Bq((size_t) num_experts * N * K);
  for (auto& x : Bq) x = qd(rng);
  std::vector<half> sPlain((size_t) num_experts * G * N), mPlain((size_t) num_experts * G * N);
  for (size_t i = 0; i < sPlain.size(); i++) {
    sPlain[i] = __float2half(0.4f + 0.6f * ((rng() % 100) / 100.0f));
    mPlain[i] = __float2half(affine ? ((int) (rng() % 100) - 50) / 20.0f : 0.0f);
  }

  // CPU reference: each token through ITS expert only.
  std::vector<double> ref((size_t) num_rows * N);
  for (int t = 0; t < num_rows; t++) {
    const int e = rows_expert[t];
    for (int n = 0; n < N; n++) {
      double acc = 0;
      for (int k = 0; k < K; k++) {
        const size_t sIdx = (size_t) e * G * N + (size_t)(k / gs) * N + n;
        acc += (double) __half2float(hA[(size_t) t * K + k]) *
               ((double) Bq[((size_t) e * N + n) * K + k] * __half2float(sPlain[sIdx])
                + (affine ? __half2float(mPlain[sIdx]) : 0.0));
      }
      ref[(size_t) t * N + n] = acc;
    }
  }

  // Marlin B packing + _scale_perm, per expert.
  std::vector<int> hB((size_t) num_experts * (K / 16) * (N * 16 / 32) * 4);
  std::vector<half> sDev(sPlain.size()), mDev(mPlain.size());
  for (int e = 0; e < num_experts; e++) {
    const int* Be = Bq.data() + (size_t) e * N * K;
    auto Bu = [&](int n, int k) { return (Be[(size_t) n * K + k] + 8) & 0xf; };
    int* out = hB.data() + (size_t) e * (K / 16) * (N * 16 / 32) * 4;
    for (int kt = 0; kt < K / 16; kt++)
      for (int nb = 0; nb < N / 16; nb++)
        for (int l = 0; l < 32; l++) {
          int n = nb * 16 + l / 4, kb = kt * 16 + (l % 4) * 2, q = 0;
          q |= Bu(n, kb) << 0;          q |= Bu(n, kb + 1) << 16;
          q |= Bu(n, kb + 8) << 4;      q |= Bu(n, kb + 9) << 20;
          q |= Bu(n + 8, kb) << 8;      q |= Bu(n + 8, kb + 1) << 24;
          q |= Bu(n + 8, kb + 8) << 12; q |= Bu(n + 8, kb + 9) << 28;
          size_t idx = (size_t) (N / 2) * kt + (nb / 4) * 32 + l;
          out[idx * 4 + (nb % 4)] = q;
        }
    marlin_gguf_ppu::marlin_permute_scales(sPlain.data() + (size_t) e * G * N,
                                           sDev.data() + (size_t) e * G * N, G, N);
    marlin_gguf_ppu::marlin_permute_scales(mPlain.data() + (size_t) e * G * N,
                                           mDev.data() + (size_t) e * G * N, G, N);
  }

  const MoePlan plan = moe_plan(rows_expert.data(), num_rows, num_experts, m_tile);

  half *dA, *dAg, *dC, *dCg, *dS, *dM; int *dB, *dWS, *dSrc;
  CK(cudaMalloc(&dA, hA.size() * 2));
  CK(cudaMalloc(&dAg, (size_t) plan.padded_rows * K * 2));
  CK(cudaMalloc(&dC, (size_t) num_rows * N * 2));
  CK(cudaMalloc(&dCg, (size_t) plan.padded_rows * N * 2));
  CK(cudaMalloc(&dB, hB.size() * 4));
  CK(cudaMalloc(&dS, sDev.size() * 2)); CK(cudaMalloc(&dM, mDev.size() * 2));
  CK(cudaMalloc(&dWS, (N / 128 + 1) * max_par * 4));
  CK(cudaMalloc(&dSrc, plan.row_src.size() * 4));
  CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dS, sDev.data(), sDev.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dM, mDev.data(), mDev.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSrc, plan.row_src.data(), plan.row_src.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dWS, 0, (N / 128 + 1) * max_par * 4));
  CK(cudaMemset(dC, 0, (size_t) num_rows * N * 2));

  {
    const long long nv = (long long) plan.padded_rows * (K / 8);
    gather_rows<<<(int) ((nv + 255) / 256), 256>>>(dA, dAg, dSrc, plan.padded_rows, K);
  }
  const int ret = moe_cuda(dAg, dB, dCg, dS, affine ? dM : nullptr, plan, N, K, dWS, gs, num_experts, max_par);
  {
    const long long nv = (long long) plan.padded_rows * (N / 8);
    scatter_rows<<<(int) ((nv + 255) / 256), 256>>>(dCg, dC, dSrc, plan.padded_rows, N);
  }
  CK(cudaDeviceSynchronize());
  if (ret) { printf("  rows=%d E=%d N=%d K=%d: ret=%d\n", num_rows, num_experts, N, K, ret); return 2; }

  std::vector<half> got((size_t) num_rows * N);
  CK(cudaMemcpy(got.data(), dC, got.size() * 2, cudaMemcpyDeviceToHost));
  double ma = 0, rm = 0;
  for (size_t i = 0; i < got.size(); i++) {
    ma = fmax(ma, fabs(__half2float(got[i]) - ref[i])); rm = fmax(rm, fabs(ref[i]));
  }
  const double rel = ma / (rm + 1e-9);
  const bool bad = rel >= 3e-2;
  printf("  MoE Q4_K %-3s rows=%-5d E=%-3d N=%-5d K=%-5d  padded=%-5d  rel %.2e -> %s\n",
         affine ? "aff" : "sym", num_rows, num_experts, N, K, plan.padded_rows, rel,
         bad ? "MISMATCH" : "MATCH");
  cudaFree(dA); cudaFree(dAg); cudaFree(dC); cudaFree(dCg); cudaFree(dB);
  cudaFree(dS); cudaFree(dM); cudaFree(dWS); cudaFree(dSrc);
  return bad ? 1 : 0;
}

int main() {
  printf("GGUF Q4_K MoE grouped GEMM on PPU (ragged routing, so segments do NOT align to m-tiles)\n");
  int bad = 0;
  for (bool aff : {false, true}) {
    bad |= run(256,  4, 256,  512,  aff);
    bad |= run(1024, 8, 512,  1024, aff);
    bad |= run(100,  8, 256,  512,  aff);   // fewer rows than experts*m_tile: several empty/short segments
  }
  printf("%s\n", bad ? "SOME CASES FAILED" : "all MoE cases MATCH");
  return bad ? 1 : 0;
}
