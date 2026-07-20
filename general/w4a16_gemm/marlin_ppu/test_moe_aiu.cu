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
#include "affine_split_ppu.cuh"   // asum: per TOKEN and expert-independent, so one pass for the whole batch
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

// dist: 0 = uniform-ish, 1 = SKEWED (a few fat experts, many tiny) -- the fp16 kernel's own worst case,
//        2 = some EMPTY experts.
// affine: run the min as a SEPARATE pass (symmetric AIU kernel + Asum @ m') against a reference that
// folds it into the mainloop. Untested until now, and "it is the same shape as the dense split" is not
// evidence -- gs=128 affine survived a whole round on exactly that reasoning while multiplying by an
// uninitialised frag_m.
static int run(int n_experts, int N, int K, int total_rows, int dist, bool affine = false, int BMR_ = 128) {
  using namespace marlin_moe_aiu_ppu;
  const int gs = 32, G = K / gs;
  const int BMR = BMR_;

  std::mt19937 rng(9 + n_experts * 31 + N + K + total_rows + dist * 7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> qd(-8, 7), ed(0, n_experts - 1);

  // ragged routing, tokens already grouped by expert (the AIU kernel takes true bounds, no padding)
  std::vector<int> rows_expert(total_rows);
  if (dist == 0) { for (auto& e : rows_expert) e = ed(rng); }
  else if (dist == 1) {   // skew: 1/8 of experts take most rows -> experts spanning MANY m-tiles
    for (auto& e : rows_expert) e = (rng() % 8 == 0) ? (int) (rng() % n_experts) : (int) (rng() % (n_experts / 4 + 1));
  } else {                // leave the upper half of the experts EMPTY
    for (auto& e : rows_expert) e = (int) (rng() % (n_experts / 2));
  }
  std::sort(rows_expert.begin(), rows_expert.end());
  std::vector<int> bounds(n_experts + 1, 0);
  for (int t = 0; t < total_rows; t++) bounds[rows_expert[t] + 1]++;
  for (int e = 0; e < n_experts; e++) bounds[e + 1] += bounds[e];

  std::vector<half> hA((size_t) total_rows * K);
  for (auto& x : hA) x = __float2half(0.1f * nd(rng));
  std::vector<int> Bq((size_t) n_experts * N * K);
  for (auto& x : Bq) x = qd(rng);
  std::vector<half> sPlain((size_t) n_experts * G * N), mPlain((size_t) n_experts * G * N);
  for (size_t i = 0; i < sPlain.size(); i++) {
    sPlain[i] = __float2half(0.4f + 0.6f * ((rng() % 100) / 100.0f));
    // both signs, magnitude comparable to |q|*s, so dropping the term cannot pass
    mPlain[i] = __float2half(affine ? ((int) (rng() % 100) - 50) / 20.0f : 0.0f);
  }

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
  std::vector<half> sDev(sPlain.size()), mDev(mPlain.size());
  for (int e = 0; e < n_experts; e++) {
    marlin_gguf_ppu::marlin_permute_scales(sPlain.data() + (size_t) e * G * N,
                                           sDev.data() + (size_t) e * G * N, G, N);
    marlin_gguf_ppu::marlin_permute_scales(mPlain.data() + (size_t) e * G * N,
                                           mDev.data() + (size_t) e * G * N, G, N);
  }
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
  half* dMref = nullptr;
  if (affine) { CK(cudaMalloc(&dMref, mDev.size() * 2));
                CK(cudaMemcpy(dMref, mDev.data(), mDev.size() * 2, cudaMemcpyHostToDevice)); }
  const int rr = marlin_moe_gguf_ppu::moe_cuda(dAg, dB, dCg, dS, dMref, plan, N, K, dWS, gs, n_experts);
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
  int maxmb = 0;
  for (int e = 0; e < n_experts; e++) maxmb = std::max(maxmb, sch.mblk_prefix[e + 1] - sch.mblk_prefix[e]);

  if (BMR == 32) launch_moe_q4k<4, 32, 2>(dA, dImg, dSp, dBounds, dMblk, dC, total_rows, N, K, n_experts, sch.total_tiles, gs);
  else           launch_moe_q4k<4, 128>   (dA, dImg, dSp, dBounds, dMblk, dC, total_rows, N, K, n_experts, sch.total_tiles, gs);
  if (affine) {
    // Asum is per TOKEN and does NOT depend on the expert, so it is one pass for the whole batch; only the
    // second (G-deep) GEMM is grouped. mins go in PLAIN [G][N] here -- the permuted layout is Marlin's
    // fragment contract and means nothing to this pass.
    half *dAsum, *dMp; int* dRowE;
    CK(cudaMalloc(&dAsum, (size_t) total_rows * G * 2));
    CK(cudaMalloc(&dMp, mPlain.size() * 2)); CK(cudaMalloc(&dRowE, total_rows * 4));
    CK(cudaMemcpy(dMp, mPlain.data(), mPlain.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dRowE, rows_expert.data(), total_rows * 4, cudaMemcpyHostToDevice));
    affine_split_ppu::asum_launch(dA, dAsum, total_rows, K, gs);
    moe_min_correct<<<total_rows, 128>>>(dAsum, dMp, dRowE, dC, total_rows, N, G);
    CK(cudaDeviceSynchronize());
    cudaFree(dAsum); cudaFree(dMp); cudaFree(dRowE);
  }
  cudaError_t e2 = cudaDeviceSynchronize();
  if (e2) { printf("  AIU kernel error: %s\n", cudaGetErrorString(e2)); return 2; }
  std::vector<float> got((size_t) total_rows * N);
  CK(cudaMemcpy(got.data(), dC, got.size() * 4, cudaMemcpyDeviceToHost));

  double ma = 0, rm = 0;
  for (size_t i = 0; i < got.size(); i++) { ma = fmax(ma, fabs(got[i] - ref[i])); rm = fmax(rm, fabs(ref[i])); }
  const double rel = ma / (rm + 1e-9);
  const bool bad = rel >= 3e-2;
  printf("  E=%-3d N=%-4d K=%-4d rows=%-5d dist=%d %-3s tiles=%-4d maxtiles/expert=%d  rel %.2e -> %s\n",
         n_experts, N, K, total_rows, dist, affine ? "aff" : "sym", sch.total_tiles, maxmb, rel,
         bad ? "MISMATCH" : "MATCH");
  cudaFree(dA); cudaFree(dAg); cudaFree(dCg); cudaFree(dB); cudaFree(dS); cudaFree(dWS); cudaFree(dSrc);
  cudaFree(dImg); cudaFree(dSp); cudaFree(dC); cudaFree(dBounds); cudaFree(dMblk); if (dMref) cudaFree(dMref);
  return bad ? 1 : 0;
}

int main() {
  printf("AIU Q4_K MoE vs Marlin-derived MoE (BMR=128, NST=4)\n");
  int bad = 0;
  // The first case is the one that already passed, kept as a control. Everything after it exists because
  // that case did NOT exercise the scheduler: every expert had fewer rows than BMR, so each was exactly
  // one m-block, mblk_prefix was [0,1,2,...], the binary search never had a choice to make, and
  // (mblk_g - mblk_prefix[e]) was identically zero. The grid-stride queue never wrapped either. So the
  // one thing this kernel adds over the reference was untested -- the same shape as the asum bug, which
  // survived because only the groupsize that hides it was ever run.
  bad |= run(4,   256,  512,  300,  0);   // control: every expert < BMR (1 m-tile each)
  bad |= run(4,   256,  512,  2000, 0);   // experts span MANY m-tiles; m0 advances; queue wraps
  bad |= run(16,  256,  512,  4000, 1);   // skewed: a few fat experts, many tiny -- the fp16 kernel's worst case
  bad |= run(8,   256,  512,  1500, 2);   // half the experts EMPTY (zero rows -> zero m-blocks)
  bad |= run(8,   512,  1024, 3000, 1);   // larger N/K, more n-blocks per tile row

  // The grid-stride loop STILL has not wrapped: launch_moe_q4k clamps grid to total_tiles, which peaked at
  // 112 against a #CU*blk/CU grid, so every block took exactly one tile and the `for (iter...)` body ran
  // once. Forcing a small grid is the only way to make iter>0 happen, and a queue that never wraps is not
  // a persistent scheduler -- it is a static grid wearing one.
  printf("--- forced small grid (MOE_GRID): makes the grid-stride queue actually wrap ---\n");
  setenv("MOE_GRID", "8", 1);
  bad |= run(16,  256,  512,  4000, 1);   // 82 tiles over 8 blocks -> ~10 iterations per block
  bad |= run(8,   512,  1024, 3000, 1);
  setenv("MOE_GRID", "1", 1);
  bad |= run(4,   256,  512,  2000, 0);   // one block does every tile: the extreme of the same path
  unsetenv("MOE_GRID");

  // AFFINE: the min as its own pass, against a reference that folds it into the mainloop. Both
  // distributions, since the correction is indexed by each row's expert and a skewed routing is where a
  // row->expert mix-up would actually show.
  printf("--- affine: min as a separate pass (symmetric AIU + Asum @ m') ---\n");
  bad |= run(4,   256,  512,  2000, 0, true);
  bad |= run(16,  256,  512,  4000, 1, true);
  bad |= run(8,   256,  512,  1500, 2, true);

  // BMR=32 via MWARPS=2 -- a 4-warp block. Newly expressible, so it needs its own correctness pass: the
  // warp grid changed, and the warp grid is what produced the earlier out-of-range column.
  printf("--- BMR=32 (MWARPS=2, 4-warp block) ---\n");
  bad |= run(4,   256,  512,  2000, 0, false, 32);
  bad |= run(16,  256,  512,  4000, 1, false, 32);
  bad |= run(8,   256,  512,  1500, 2, true,  32);
  printf("%s\n", bad ? "SOME CASES FAILED" : "all AIU MoE cases MATCH");
  return bad ? 1 : 0;
}
