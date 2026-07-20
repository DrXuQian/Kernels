// AIU Q4_K MoE vs the Marlin-derived MoE, on identical routing/weights/scales.
//
// The reference is marlin_moe_gguf_ppu, which passes ragged routing and short/empty segments. Comparing
// against a CPU loop as well would be better still, but the point of THIS test is the two index
// assumptions the AIU kernel is built on and cannot justify from first principles: the scale's column
// mapping within the n16 tile, and the AIU cube's k offset against the absolute k. Both are exactly the
// kind of thing that has been wrong every single time in this work -- A_IDX, warp_k, asum's reduction
// width, mtile_expert's granularity -- so they get a reference, not an argument.
//
// Symmetric only for now: the min is a separate pass here (moe_min_correct) and gets its own case once
// the base path is right. Wiring both at once would leave two candidates for any mismatch.
#include "marlin_moe_aiu_ppu.cuh"
#include "marlin_moe_gguf_ppu.cuh"
#include <cstdio>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

int main() {
  using namespace marlin_moe_aiu_ppu;
  const int n_experts = 4, N = 256, K = 512, gs = 32, G = K / gs;
  const int BMR = 128, NST = 4;
  printf("AIU Q4_K MoE vs Marlin-derived MoE: E=%d N=%d K=%d gs=%d BMR=%d NST=%d\n",
         n_experts, N, K, gs, BMR, NST);

  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> qd(-8, 7), ed(0, n_experts - 1);

  // ragged routing, tokens already grouped by expert (the AIU kernel takes true bounds, no padding)
  const int total_rows = 300;
  std::vector<int> rows_expert(total_rows);
  for (auto& e : rows_expert) e = ed(rng);
  std::sort(rows_expert.begin(), rows_expert.end());
  std::vector<int> bounds(n_experts + 1, 0);
  for (int t = 0; t < total_rows; t++) bounds[rows_expert[t] + 1]++;
  for (int e = 0; e < n_experts; e++) bounds[e + 1] += bounds[e];

  std::vector<half> hA((size_t) total_rows * K);
  for (auto& x : hA) x = __float2half(0.1f * nd(rng));
  std::vector<int> Bq((size_t) n_experts * N * K);
  for (auto& x : Bq) x = qd(rng);
  std::vector<half> sPlain((size_t) n_experts * G * N);
  for (auto& x : sPlain) x = __float2half(0.4f + 0.6f * ((rng() % 100) / 100.0f));

  // Marlin packing per expert, then the AIU image (the swzl permutation of the same bytes).
  const size_t bwords = (size_t) (K / 16) * (N * 16 / 32) * 4;
  std::vector<int> hB((size_t) n_experts * bwords);
  for (int e = 0; e < n_experts; e++) {
    const int* Be = Bq.data() + (size_t) e * N * K;
    auto Bu = [&](int n, int k) { return (Be[(size_t) n * K + k] + 8) & 0xf; };
    int* out = hB.data() + (size_t) e * bwords;
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
  }
  const int IH = aiu_img_h(N), IW = aiu_img_w(K);
  std::vector<unsigned short> img((size_t) n_experts * IH * IW, 0);
  for (int e = 0; e < n_experts; e++)
    marlin_b_to_aiu_image(hB.data() + (size_t) e * bwords, N, K, img.data() + (size_t) e * IH * IW);

  // ---- reference: the Marlin-derived MoE (padded plan + per-expert launches) ----
  std::vector<half> sDev(sPlain.size());
  for (int e = 0; e < n_experts; e++)
    marlin_gguf_ppu::marlin_permute_scales(sPlain.data() + (size_t) e * G * N,
                                           sDev.data() + (size_t) e * G * N, G, N);
  const int m_tile = 16 * MARLIN_MAX_MB;
  const auto plan = marlin_moe_gguf_ppu::moe_plan(rows_expert.data(), total_rows, n_experts, m_tile);

  half *dA, *dAg, *dCg, *dS; int *dB, *dWS, *dSrc;
  CK(cudaMalloc(&dA, hA.size() * 2));
  CK(cudaMalloc(&dAg, (size_t) plan.padded_rows * K * 2));
  CK(cudaMalloc(&dCg, (size_t) plan.padded_rows * N * 2));
  CK(cudaMalloc(&dB, hB.size() * 4)); CK(cudaMalloc(&dS, sDev.size() * 2));
  CK(cudaMalloc(&dWS, (N / 128 + 1) * 128 * 4)); CK(cudaMalloc(&dSrc, plan.row_src.size() * 4));
  CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dS, sDev.data(), sDev.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSrc, plan.row_src.data(), plan.row_src.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dWS, 0, (N / 128 + 1) * 128 * 4));
  {
    const long long nv = (long long) plan.padded_rows * (K / 8);
    marlin_moe_gguf_ppu::gather_rows<<<(int) ((nv + 255) / 256), 256>>>(dA, dAg, dSrc, plan.padded_rows, K);
  }
  const int rr = marlin_moe_gguf_ppu::moe_cuda(dAg, dB, dCg, dS, nullptr, plan, N, K, dWS, gs, n_experts);
  CK(cudaDeviceSynchronize());
  if (rr) { printf("reference failed ret=%d\n", rr); return 2; }
  std::vector<half> refPad((size_t) plan.padded_rows * N);
  CK(cudaMemcpy(refPad.data(), dCg, refPad.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<double> ref((size_t) total_rows * N);
  for (int r = 0; r < plan.padded_rows; r++) {
    const int src = plan.row_src[r];
    if (src < 0) continue;
    for (int n = 0; n < N; n++) ref[(size_t) src * N + n] = __half2float(refPad[(size_t) r * N + n]);
  }

  // ---- AIU kernel: true bounds, no padding, contiguous A (rows already grouped by expert) ----
  unsigned short* dImg; half* dSp; float* dC; int *dBounds, *dMblk;
  CK(cudaMalloc(&dImg, img.size() * 2)); CK(cudaMalloc(&dSp, sPlain.size() * 2));
  CK(cudaMalloc(&dC, (size_t) total_rows * N * 4));
  CK(cudaMalloc(&dBounds, (n_experts + 1) * 4));
  CK(cudaMemcpy(dImg, img.data(), img.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSp, sPlain.data(), sPlain.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBounds, bounds.data(), (n_experts + 1) * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0, (size_t) total_rows * N * 4));
  const auto sch = moe_sched(bounds.data(), n_experts, N, BMR);
  CK(cudaMalloc(&dMblk, (n_experts + 1) * 4));
  CK(cudaMemcpy(dMblk, sch.mblk_prefix.data(), (n_experts + 1) * 4, cudaMemcpyHostToDevice));
  printf("  total_tiles=%d  rows/expert:", sch.total_tiles);
  for (int e = 0; e < n_experts; e++) printf(" %d", bounds[e + 1] - bounds[e]);
  printf("\n");

  launch_moe_q4k<4, 128>(dA, dImg, dSp, dBounds, dMblk, dC, total_rows, N, K, n_experts, sch.total_tiles, gs);
  cudaError_t e2 = cudaDeviceSynchronize();
  if (e2) { printf("  AIU kernel error: %s\n", cudaGetErrorString(e2)); return 2; }
  std::vector<float> got((size_t) total_rows * N);
  CK(cudaMemcpy(got.data(), dC, got.size() * 4, cudaMemcpyDeviceToHost));

  double ma = 0, rm = 0;
  for (size_t i = 0; i < got.size(); i++) { ma = fmax(ma, fabs(got[i] - ref[i])); rm = fmax(rm, fabs(ref[i])); }
  const double rel = ma / (rm + 1e-9);
  printf("  AIU vs Marlin-MoE: rel %.2e (|ref|max=%.1f) -> %s\n", rel, rm, rel < 3e-2 ? "MATCH" : "MISMATCH");
  return rel < 3e-2 ? 0 : 1;
}
