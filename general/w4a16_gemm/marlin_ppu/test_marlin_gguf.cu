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
#include <cstring>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include "marlin_gguf_ppu.cuh"
#ifndef GGUF_BUILD_SHA
#define GGUF_BUILD_SHA "unknown-sha (pass -DGGUF_BUILD_SHA=...)"
#endif

// uniform=true: every per-32 scale is EXACTLY 1.0. The staging path is byte-for-byte the same (still NG
// group-rows, still cp.async'd into shared, still read per k-block) -- only the VALUES are all equal, so a
// wrong group INDEX becomes invisible. Bisection: uniform MATCH + distinct MISMATCH => the bug is purely
// in which group we read. Both MISMATCH => the bug is in the B pack / dequant / mma, not the scales.
// SM_*: which axis the per-32 scale actually varies along. All four run the identical staging path;
// they differ only in which class of index error can show up in the result:
//   SM_ONE  s=1.0        both group- and column-index errors invisible -> tests everything BUT the scales
//   SM_COL  s = f(n)     group errors invisible, COLUMN errors visible
//   SM_GRP  s = f(g)     column errors invisible, GROUP errors visible
//   SM_FULL s = f(n,g)   everything visible (the real case)
enum ScaleMode { SM_ONE, SM_COL, SM_GRP, SM_FULL, SM_READOUT };
static const char* sm_name(int m) { return m==SM_ONE?"s=1.0":m==SM_COL?"s=f(n)":m==SM_GRP?"s=f(g)":m==SM_READOUT?"readout":"gs=32"; }

static int run_case(int M, int N, int K, int mode = SM_FULL, int gs = 32) {
  using namespace marlin_gguf_ppu;
  const int max_par = 128;
  const int NG = K / gs;                             // groups per column
  std::mt19937 rng(999 + M * 3 + N * 7 + K * 13);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> qd(-8, 7);

  std::vector<half> hA((size_t)M * K), hC((size_t)M * N, __float2half(0.f));
  for (auto& x : hA) x = __float2half(0.1f * nd(rng));
  // READOUT: stop inferring which group the kernel read and just make it print the index. Row m of A is
  // one-hot on the m-th 32-block, every Bq is 1, and the scale VALUE is the group number itself (g+1). So
  // C[m][n] = 32 * (whatever group index the kernel used for block m, +1) -- an exact small integer in
  // half. Dividing by 32 reads the kernel's group index straight out, per (block, column).
  if (mode == SM_READOUT)
    for (int m = 0; m < M; m++)
      for (int k = 0; k < K; k++)
        hA[(size_t)m * K + k] = __float2half((k / gs) == m ? 1.0f : 0.0f);

  // Bq[n][k] in [-8,7] and a DISTINCT per-32 scale d[n][kb] -- distinct so a per-tile (not per-32)
  // scale bug cannot pass: the two 32-blocks of one 64-k tile get different d.
  auto hashf = [](uint32_t i) {                         // uncorrelated in [0.4, 1.0)
    i = (i + 0x9e3779b9u) * 0x85ebca6bu; i ^= i >> 13; i *= 0xc2b2ae35u; i ^= i >> 16;
    return 0.4f + 0.6f * ((i % 1024) / 1024.0f);
  };
  std::vector<int>  Bq((size_t)N * K);
  std::vector<half> d((size_t)N * NG);
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) Bq[(size_t)n * K + k] = (mode == SM_READOUT) ? 1 : qd(rng);
    for (int g = 0; g < NG; g++) {
      float dv = 0.4f + 0.6f * ((rng() % 100) / 100.0f);      // draw regardless, so rng state matches
      // HIGH CONTRAST, and deliberately so. The first version of these was 0.4+0.6*((i%97)/97), which is
      // CONTINUOUS in the index: neighbours differ by 0.6%, so an off-by-one read produced ~0.6% error and
      // passed a 3% tolerance. Both axes "MATCHed" while the real case sat at 24% -- the probe was blind
      // to exactly the bug class it existed to find. hashf decorrelates neighbours, so ANY wrong index
      // gives a full-magnitude miss.
      float v = mode == SM_ONE  ? 1.0f
              : mode == SM_COL  ? hashf(n)
              : mode == SM_GRP  ? hashf(g)
              : mode == SM_READOUT ? (float)(g + 1)
                                : dv;
      d[(size_t)n * NG + g] = __float2half(v);
    }
  }

  // Reference: C[m][n] = sum_k A[m][k] * Bq[n][k] * d[n][k/32]
  std::vector<float> ref((size_t)M * N);
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      double acc = 0;
      for (int k = 0; k < K; k++)
        acc += (double)__half2float(hA[(size_t)m * K + k]) * Bq[(size_t)n * K + k] * __half2float(d[(size_t)n * NG + k / gs]);
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

  int ret = marlin_cuda(dA, dB, dC, dS, nullptr /*mins: symmetric*/, M, N, K, dWS, gs);
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
  printf("  Q4_0/%-8s gs=%-3d M%-4d N%-5d K%-5d  rel %.2e (|ref|max=%.1f) -> %s\n",
         sm_name(mode), gs, M, N, K, rel, refmax, bad ? "MISMATCH" : "MATCH");
  // The group axis is the one that is broken (s=f(n) sits on the s=1.0 baseline, s=f(g) does not). Rather
  // than keep reading the kernel and guessing, ask which group it ACTUALLY read: rebuild the reference
  // under a list of candidate wrong mappings and see which one reproduces the GPU output. The one that
  // matches names the bug outright; if none matches, the fault is not a uniform remap of the group index
  // and the next probe has to be per-(stage,k) rather than per-g.
  if (mode == SM_READOUT) {
    printf("    group index the kernel actually used (row = the 32-block being probed):\n");
    printf("      block |  n=0    n=1    n=8    n=64   n=65  |  expected\n");
    const int ns[5] = {0, 1, 8, 64, 65};
    int wrong = 0;
    for (int m = 0; m < M && m < NG; m++) {
      printf("      %5d |", m);
      for (int t = 0; t < 5; t++) {
        int n = ns[t] % N;
        float v = __half2float(hC[(size_t)m * N + n]) / 32.0f - 1.0f;   // == the group index it read
        printf(" %6.2f", v);
        if (fabs(v - m) > 0.01) wrong++;
      }
      printf("  |  %5d%s\n", m, wrong ? "" : "");
    }
    printf("    (a column of constant wrong values = a group remap; values varying along n = the group\n"
           "     read depends on the column, i.e. the two axes are not separable after all)\n");
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dS); cudaFree(dWS);
    return 0;
  }
  if (bad && mode == SM_GRP) {
    struct Cand { const char* name; int (*f)(int, int); };
    static const Cand cands[] = {
      {"g^1        (two groups of a k-tile swapped)", [](int g, int NG){ return g ^ 1; }},
      {"2*(g/2)    (always the tile's FIRST group)",  [](int g, int NG){ return 2 * (g / 2); }},
      {"2*(g/2)+1  (always the tile's SECOND group)", [](int g, int NG){ return 2 * (g / 2) + 1; }},
      {"g+1        (off by one group)",               [](int g, int NG){ return (g + 1) % NG; }},
      {"g-1        (off by one group)",               [](int g, int NG){ return (g - 1 + NG) % NG; }},
      {"g+2        (off by one k-TILE)",              [](int g, int NG){ return (g + 2) % NG; }},
      {"g-2        (off by one k-TILE)",              [](int g, int NG){ return (g - 2 + NG) % NG; }},
      {"g/2        (read as if gs=64)",               [](int g, int NG){ return g / 2; }},
      {"(2*g)%NG   (read as if gs=16)",               [](int g, int NG){ return (2 * g) % NG; }},
      {"0          (group index stuck at 0)",         [](int g, int NG){ return 0; }},
    };
    printf("    which group did the kernel read? (rebuilding the reference under candidate mappings)\n");
    for (const auto& c : cands) {
      double ma = 0, rm = 0;
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
          double acc = 0;
          for (int k = 0; k < K; k++)
            acc += (double)__half2float(hA[(size_t)m * K + k]) * Bq[(size_t)n * K + k]
                 * __half2float(__float2half(hashf(c.f(k / gs, NG))));
          ma = fmax(ma, fabs(__half2float(hC[(size_t)m * N + n]) - acc));
          rm = fmax(rm, fabs(acc));
        }
      double r = ma / (rm + 1e-9);
      printf("      %-44s rel %.2e%s\n", c.name, r, r < 3e-2 ? "   <== THIS IS WHAT THE KERNEL DOES" : "");
    }
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dS); cudaFree(dWS);
  return bad ? 1 : 0;
}

// Q4_K: build REAL Q4_K blocks (d/dmin/6-bit packed sc+m), repack via marlin_repack_q4k, and check the
// kernel against a CPU reference that dequantizes the SAME blocks the way ggml does. This exercises the
// full affine path: two-level scale expanded to per-32 (s, m'), and hfma2 instead of hmul2.
static int run_q4k(int M, int N, int K) {
  using namespace marlin_gguf_ppu;
  const int gs = 32, max_par = 128, NG = K / 32, nblk = K / 256;
  std::mt19937 rng(4242 + M + N * 5 + K * 11);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> d6(1, 63), q4(0, 15);

  std::vector<half> hA((size_t)M * K), hC((size_t)M * N, __float2half(0.f));
  for (auto& x : hA) x = __float2half(0.1f * nd(rng));

  std::vector<gguf_block_q4_K> blocks((size_t)N * nblk);
  for (auto& b : blocks) {
    // distinct-per-sub-block sc/m so a per-tile (not per-32) scale bug cannot pass
    uint8_t sc[8], mn[8];
    for (int g = 0; g < 8; g++) { sc[g] = (uint8_t)d6(rng); mn[g] = (uint8_t)d6(rng); }
    b.d = __float2half(0.002f + 0.001f * ((rng() % 100) / 100.0f));
    b.dmin = __float2half(0.001f + 0.001f * ((rng() % 100) / 100.0f));
    memset(b.scales, 0, 12);
    for (int g = 0; g < 4; g++) { b.scales[g] = sc[g] & 63; b.scales[g + 4] = mn[g] & 63; }
    for (int g = 4; g < 8; g++) {
      b.scales[g + 4] = (uint8_t)((sc[g] & 0xF) | ((mn[g] & 0xF) << 4));
      b.scales[g - 4] |= (uint8_t)((sc[g] >> 4) << 6);
      b.scales[g - 0] |= (uint8_t)((mn[g] >> 4) << 6);
    }
    for (int i = 0; i < 128; i++) b.qs[i] = (uint8_t)(q4(rng) | (q4(rng) << 4));
  }

  // CPU reference: dequantize exactly as ggml does -- w = d*sc_b*q - dmin*m_b
  std::vector<float> ref((size_t)M * N);
  for (int n = 0; n < N; n++) {
    std::vector<float> w(K);
    for (int blk = 0; blk < nblk; blk++) {
      const gguf_block_q4_K& B = blocks[(size_t)n * nblk + blk];
      uint8_t sc[8], mn[8]; gguf_q4k_scales(B.scales, sc, mn);
      const float d = __half2float(B.d), dmin = __half2float(B.dmin);
      for (int b = 0; b < 8; b++)
        for (int j = 0; j < 32; j++) {
          const int byte = (b / 2) * 32 + j;
          const int q = (b % 2 == 0) ? (B.qs[byte] & 0xF) : (B.qs[byte] >> 4);
          w[blk * 256 + b * 32 + j] = d * sc[b] * q - dmin * mn[b];
        }
    }
    for (int m = 0; m < M; m++) {
      double acc = 0;
      for (int k = 0; k < K; k++) acc += (double)__half2float(hA[(size_t)m * K + k]) * w[k];
      ref[(size_t)m * N + n] = acc;
    }
  }

  std::vector<int>  hB((size_t)(K / 16) * (N * 16 / 32) * 4);
  std::vector<half> hS((size_t)NG * N), hM((size_t)NG * N);
  marlin_repack_q4k(blocks.data(), N, K, hB.data(), hS.data(), hM.data());

  half *dA, *dC, *dS, *dM; int *dB, *dWS;
  cudaMalloc(&dA, hA.size() * 2); cudaMalloc(&dB, hB.size() * 4); cudaMalloc(&dC, hC.size() * 2);
  cudaMalloc(&dS, hS.size() * 2); cudaMalloc(&dM, hM.size() * 2); cudaMalloc(&dWS, (N / 128 + 1) * max_par * 4);
  cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dM, hM.data(), hM.size() * 2, cudaMemcpyHostToDevice);
  cudaMemset(dWS, 0, (N / 128 + 1) * max_par * 4);

  int ret = marlin_cuda(dA, dB, dC, dS, dM /*affine mins*/, M, N, K, dWS, gs);
  cudaError_t e = cudaDeviceSynchronize();
  if (ret || e) { printf("  Q4_K M%-4d N%-5d K%-5d: ret=%d err=%s\n", M, N, K, ret, cudaGetErrorString(e)); return 2; }
  cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost);

  double maxabs = 0, refmax = 0;
  for (size_t i = 0; i < (size_t)M * N; i++) {
    maxabs = fmax(maxabs, fabs(__half2float(hC[i]) - ref[i]));
    refmax = fmax(refmax, fabs(ref[i]));
  }
  const double rel = maxabs / (refmax + 1e-9);
  const bool bad = rel >= 3e-2;
  printf("  Q4_K/gs=32  M%-4d N%-5d K%-5d  rel %.2e (|ref|max=%.1f) -> %s\n", M, N, K, rel, refmax, bad ? "MISMATCH" : "MATCH");
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dS); cudaFree(dM); cudaFree(dWS);
  return bad ? 1 : 0;
}

int main() {
  printf("BUILD %s  (%s %s)\n", GGUF_BUILD_SHA, __DATE__, __TIME__);
  // s=1.0 already came back MATCH on ppu001 (rel 5e-4), so the B pack / dequant / mma / A path is sound
  // and the fault is somewhere in the scales. These two split WHICH scale index is wrong -- s=1.0 could
  // not, because it hid the group and the column error alike.
  printf("--- probe: isolate the scale axis (identical staging path in all cases) ---\n");
  int p_one = run_case(64, 256, 1024, SM_ONE);
  int p_col = run_case(64, 256, 1024, SM_COL);
  int p_grp = run_case(16, 128, 512, SM_GRP);      // small: the candidate scan below is O(M*N*K) per candidate
  printf("  => %s\n",
      p_one ? "s=1.0 FAILED: base path broken, scales are not the bug"
    : (p_col && p_grp) ? "BOTH axes wrong"
    : p_col ? "COLUMN index wrong (s_sh_rd / _scale_perm), group index OK"
    : p_grp ? "GROUP index wrong ((k%b_sh_wr_iters)/2 / staging), column index OK"
            : "both axes OK alone -- fault needs BOTH to vary (aliasing in the packed int4 scale read)");
  printf("  (compare each rel against the s=1.0 baseline, not just against the threshold)\n");
  // gs=128 goes through group_blocks=8, i.e. the group_blocks >= thread_k_blocks path that marlin_classic
  // already verified on this box. It shares the B packing, the scale layout and _scale_perm with gs=32 and
  // differs ONLY in the group machinery, so it is the control: if it passes, the fault is confined to the
  // group_blocks==2 branch I added, and nothing under it is suspect.
  int probe = p_one | p_col | p_grp;
  printf("--- READOUT: make the kernel print the group index it used ---\n");
  run_case(16, 128, 512, SM_READOUT);
  printf("--- control: gs=128 (group_blocks=8, the path marlin_classic already verified) ---\n");
  probe |= run_case(64, 256, 1024, SM_FULL, 128);
  probe |= run_case(128, 512, 1024, SM_FULL, 128);
  printf("GGUF int4 on PPU Marlin, gs=32 (per-32 in-tile scale):\n");
  int bad = 0;
  int bad2 = probe;
  bad |= run_case(16,  128, 512);
  bad |= run_case(32,  256, 512);
  bad |= run_case(64,  256, 1024);
  bad |= run_case(128, 512, 1024);
  printf("--- Q4_K (affine: two-level scale expanded to per-32 (s, m'), hfma2) ---\n");
  bad |= run_q4k(16,  128, 512);
  bad |= run_q4k(32,  256, 512);
  bad |= run_q4k(64,  256, 1024);
  bad |= bad2;
  printf("%s\n", bad ? "SOME CASES FAILED" : "all gs=32 cases MATCH");
  return bad;
}
