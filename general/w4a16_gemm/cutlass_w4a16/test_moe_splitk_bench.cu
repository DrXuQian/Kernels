// Grouped mixed-input GEMM at splitk=1 and splitk>1, on the REAL grouped shape.
//
// This produces two of the three winners to profile; the third (the CUDA-core GEMV) comes from
// test_gemv_perf.cu. They are reported the same way on purpose.
//
// WHY THIS EXISTS. The split-K question was first put to a DENSE ladder at m=8, where TileM=16 gives mt=1 and
// the launch is 64 CTAs on a 72-CU part (acu: DRAM 4.43%). That measurement could not answer anything, and
// no dense shape can: matching grouped's grid needs mt=8, but a dense m=128 shares ONE B across its 8 m-tiles
// while grouped's 8 experts have 8 different B's that cannot be shared. So the question only has an answer on
// the grouped path.
//
// WHAT SPLIT-K CAN AND CANNOT BUY, stated up front so the numbers are read against a prediction. Decode today
// is 512 CTAs x 64 threads = 1024 warps = 14.2 warps/CU with acu measuring 13.65 achieved -- every warp of the
// launch resident at once, no second wave, so occupancy is bounded by WORK and not by smem or registers.
// Split-K multiplies the CTA count by S, so it buys resident warps up to the theoretical ceiling acu reported
// (18 warps/CU, 28.13%) and no further: about 1.27x. It also multiplies A and scale traffic by S while leaving
// weight traffic alone, and adds the partial buffer plus the merge pass. If the measured speedup exceeds
// 1.27x, the occupancy model is what was wrong.
//
// Build: TARGET=test_moe_splitk_bench ./build.sh
// Run:   $BIN/test_moe_splitk_bench [L] [Rows] [N] [K] [gs] [mode]
//          mode 3 = DECODE batch=1 (default here), 2 = skewed prefill band, 0 = uniform
//   SPLITK_ONLY=<substring>  run only rows whose tag contains this
//   SPLITK_ACU=1             ONE COLD launch per row (a capture, not a timing)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "lowbit_moe_bench.hpp"     // Band, time_it, moe_ok, HBM_GBS, PEAK
#include "moe_splitk_ppu.cuh"

static const char* sk_only() { return std::getenv("SPLITK_ONLY"); }
static bool sk_acu() { return std::getenv("SPLITK_ACU") != nullptr; }
static double pct_of(double gbs) { return 100.0 * gbs / HBM_GBS; }
static bool sk_selected(const char* tag) {
  const char* f = sk_only();
  return !f || std::strstr(tag, f) != nullptr;
}

// Two winners, kept apart: the whole question is whether S>1 beats S==1, and one combined "best" would hide it.
struct SkBest { char tag[80] = ""; double us = 1e30; int S = 0; };
static void sk_upd(SkBest& b, const char* t, double us, int S) {
  if (us > 0 && us < b.us) { std::snprintf(b.tag, sizeof(b.tag), "%s", t); b.us = us; b.S = S; }
}

struct SkCtx {
  cutlass::DeviceAllocation<half_t>* dPart;     // S_max * total * N partials
  cutlass::DeviceAllocation<half_t*>* pdAll;    // L * S_max output pointers
  cutlass::DeviceAllocation<half_t>* dD;        // final output
  cutlass::DeviceAllocation<GS>* gsdSlice;      // device per-expert shapes with the slice K
  std::vector<GS>* gshSlice;                    // host scratch for the same
  int S_max;
};

// One (config, S) row.
template <int TM, int TN, int TK, int WM, int WN, int Stages>
static void sk_row(Band const& bd, SkCtx const& cx, cutlass::DeviceAllocation<int4_t>& dB,
                   int slices, SkBest& b1, SkBest& bS) {
  if constexpr (!moe_ok<TM, TN, TK, WM, WN, Stages, 4>()) { (void)slices; return; }
  else {
    char tag[80];
    std::snprintf(tag, sizeof(tag), "i4 %dx%d:%d w%dx%d s%d  S=%d", TM, TN, TK, WM, WN, Stages, slices);
    if (!sk_selected(tag)) return;
    const char* why = "";
    if (!moe_splitk_ppu::splitk_ok(bd.K, slices, bd.gs, TK, &why)) {
      std::printf("  %-34s %10s | ILLEGAL: %s\n", tag, "-", why);
      return;
    }

    auto go = [&] {
      moe_splitk_ppu::launch_splitk<QM::FinegrainedScaleZero, TM, TN, TK, WM, WN, Stages, int4_t>(
          bd.dA, dB.get(), bd.dSc, bd.dZr,
          cx.dD->get(), cx.dPart->get(),
          slices == 1 ? bd.pd : cx.pdAll->get(), bd.sd, bd.gm,
          bd.Mmax, bd.N, bd.K, bd.L, bd.gs, slices,
          bd.rdev, bd.rsh.data(), *cx.gshSlice, cx.gsdSlice->get(),
          bd.mode ? bd.offdev : nullptr, bd.total,
          bd.ws, bd.wsb, nullptr);
    };

    int const f0 = moe_grouped_ppu::moeg_fail_count();
    double us;
    if (sk_acu()) { us = time_it(go, 0); std::printf("  [acu] ONE COLD launch (not a timing): %s\n", tag); return; }
    us = time_it(go, 20);
    if (moe_grouped_ppu::moeg_fail_count() != f0) {
      std::printf("  %-34s %10s | DID NOT RUN (launch refused) -- excluded\n", tag, "-");
      return;
    }

    // Traffic. Weights are read ONCE regardless of S (each slice reads its own K-range), but A, scales and
    // zeros are read once PER SLICE, and the partials are written then read back by the merge.
    double const wb = double(bd.N) * bd.K * 4 / 8.0;                      // int4 codes
    double const sb = double(bd.scale_k) * bd.N * 2.0 * 2.0;              // scale + zero, whole matrix
    double const ab = double(bd.total) * bd.K * 2.0 * slices;             // A re-read per slice
    double const pb = (slices > 1) ? (2.0 * double(slices) * bd.total * bd.N * 2.0) : 0.0;  // write + read back
    double const db = double(bd.total) * bd.N * 2.0;
    double const bytes = double(bd.active) * (wb + sb) + ab + pb + db;
    double const gbs = bytes / (us * 1e-6) / 1e9;

    int mt = 0;
    for (int e = 0; e < bd.L; ++e) mt += (bd.me[e] + TM - 1) / TM;
    int64_t const ctas = int64_t(mt) * ((bd.N + TN - 1) / TN) * slices;
    double const wkwrp_cu = double(ctas) * (double(TM / WM) * (TN / WN)) / 72.0;

    std::printf("  %-34s %8.2f us | %7.1f GB/s | %5.1f%% HBM | cta %6lld | wkwrp/CU %6.1f%s\n",
                tag, us, gbs, pct_of(gbs), (long long)ctas, wkwrp_cu,
                gbs > HBM_GBS ? "  <-- IMPLIES > HBM PEAK, excluded" : "");
    if (gbs > HBM_GBS) return;
    sk_upd(slices == 1 ? b1 : bS, tag, us, slices);
  }
}

int main(int argc, char** argv) {
  Band bd{};
  bd.L    = argc > 1 ? atoi(argv[1]) : 8;
  bd.Rows = argc > 2 ? atoi(argv[2]) : 1;
  bd.N    = argc > 3 ? atoi(argv[3]) : 2048;
  bd.K    = argc > 4 ? atoi(argv[4]) : 2048;
  bd.gs   = argc > 5 ? atoi(argv[5]) : 32;
  bd.mode = argc > 6 ? atoi(argv[6]) : 3;     // decode batch=1 by default: that is the band in question
  bd.scale_k = bd.K / bd.gs;

  bd.me.resize(bd.L); bd.offs.resize(bd.L);
  bd.total = 0; bd.Mmax = 0; bd.active = 0;
  for (int e = 0; e < bd.L; ++e) {
    if (bd.mode == 0)      bd.me[e] = bd.Rows;
    else if (bd.mode == 3) bd.me[e] = (e < bd.Rows || bd.Rows >= bd.L) ? 1 : 0;
    else {
      unsigned h = (unsigned)e * 2654435761u >> 13;
      if ((h % 8) == 0)      bd.me[e] = 0;
      else if ((h % 8) == 1) bd.me[e] = int(bd.Rows * 3 + (h % 37));
      else                   bd.me[e] = int(bd.Rows / 2 + (h % (unsigned)(bd.Rows + 1)));
    }
    if (bd.me[e]) ++bd.active;
    bd.offs[e] = bd.total; bd.total += bd.me[e]; bd.Mmax = std::max(bd.Mmax, bd.me[e]);
  }
  if (bd.Mmax == 0) { std::printf("all experts empty\n"); return 1; }

  std::printf("== grouped mixed-input GEMM: splitk=1 vs splitk>1 ==\n");
  std::printf("   L=%d rows=%d mode=%d N=%d K=%d gs=%d | total=%d Mmax=%d active=%d | HBM %.0f GB/s\n",
              bd.L, bd.Rows, bd.mode, bd.N, bd.K, bd.gs, bd.total, bd.Mmax, bd.active, HBM_GBS);
  if (sk_only()) std::printf("   SPLITK_ONLY=\"%s\"\n", sk_only());
  if (sk_acu())  std::printf("   *** SPLITK_ACU: ONE COLD LAUNCH PER ROW. Captures, not timings. ***\n");

  // int4 memory roof: weights + scale/zero once per active expert, plus A and D.
  double const roof = double(bd.active) * (double(bd.N) * bd.K * 4 / 8.0 + double(bd.scale_k) * bd.N * 4.0)
                    + double(bd.total) * bd.K * 2.0 + double(bd.total) * bd.N * 2.0;
  std::printf("   int4 memory roof: %.2f us (%.2f MB)\n", roof / (HBM_GBS * 1e9) * 1e6, roof / 1e6);

  std::vector<half_t> hA((size_t)bd.total * bd.K), hSc((size_t)bd.L * bd.scale_k * bd.N),
                      hZr((size_t)bd.L * bd.scale_k * bd.N);
  for (auto& v : hA)  v = half_t(0.01f);
  for (auto& v : hSc) v = half_t(0.0625f);
  for (auto& v : hZr) v = half_t(-0.0625f);

  cutlass::DeviceAllocation<half_t> dA((size_t)bd.total * bd.K), dSc((size_t)bd.L * bd.scale_k * bd.N),
                                    dZr((size_t)bd.L * bd.scale_k * bd.N), dD((size_t)bd.total * bd.N);
  dA.copy_from_host(hA.data()); dSc.copy_from_host(hSc.data()); dZr.copy_from_host(hZr.data());

  int const S_MAX = 8;
  cutlass::DeviceAllocation<half_t> dPart((size_t)S_MAX * bd.total * bd.N);

  bd.rsh.resize(bd.L);
  std::vector<half_t*> pdh(bd.L); std::vector<DStride> sdh(bd.L); std::vector<int> gmh(bd.L);
  std::vector<half_t*> pdAllh((size_t)bd.L * S_MAX);
  for (int e = 0; e < bd.L; ++e) {
    bd.rsh[e] = cute::make_shape(bd.me[e], bd.N, bd.K);
    pdh[e] = dD.get() + (size_t)bd.offs[e] * bd.N;
    sdh[e] = cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(bd.me[e], bd.N, 1));
    gmh[e] = bd.me[e];
    // Slice s writes into partial plane s, so the merge finds the S planes contiguous.
    for (int s = 0; s < S_MAX; ++s)
      pdAllh[(size_t)s * bd.L + e] = dPart.get() + ((size_t)s * bd.total + bd.offs[e]) * bd.N;
  }
  cutlass::DeviceAllocation<GS> rdev(bd.L);            rdev.copy_from_host(bd.rsh.data());
  cutlass::DeviceAllocation<half_t*> pd(bd.L);         pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(bd.L);         sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(bd.L);             gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(bd.L);         offdev.copy_from_host(bd.offs.data());
  cutlass::DeviceAllocation<half_t*> pdAll((size_t)bd.L * S_MAX); pdAll.copy_from_host(pdAllh.data());
  cutlass::DeviceAllocation<GS> gsdSlice(bd.L);
  std::vector<GS> gshSlice(bd.L);
  cutlass::DeviceAllocation<char> ws(1 << 20);

  bd.dA = dA.get(); bd.dSc = dSc.get(); bd.dZr = dZr.get();
  bd.pd = pd.get(); bd.sd = sd.get(); bd.rdev = rdev.get(); bd.gm = gm.get(); bd.offdev = offdev.get();
  bd.ws = ws.get(); bd.wsb = ws.size();

  SkCtx cx{ &dPart, &pdAll, &dD, &gsdSlice, &gshSlice, S_MAX };

  // int4 weights, packed once per (TM,TN,TK,WM,WN) fold shape. Every config below shares TK, WN and the fold
  // factor within a group, so the pack is hoisted per group rather than per row.
  SkBest b1, bS;

#define SK_PACK_AND_RUN(TM, TN, TK, WM, WN, ST)                                                        \
  do {                                                                                                 \
    constexpr int _F = moe_fold<4>(TK);                                                                \
    size_t const _per = (size_t)bd.K * bd.N * 4 / 8;                                                    \
    std::vector<int8_t> _bb((size_t)bd.L * _per);                                                       \
    { std::vector<uint8_t> _q((size_t)bd.K * bd.N);                                                     \
      for (size_t i = 0; i < _q.size(); ++i) _q[i] = uint8_t((i * 2654435761u >> 5) & 0xFu);            \
      xplane::place_derived<4, TM, TN, TK, WM, WN, _F>(_bb.data(), _q, bd.N, bd.K);                     \
      for (int e = 1; e < bd.L; ++e) std::memcpy(_bb.data() + (size_t)e * _per, _bb.data(), _per); }     \
    cutlass::DeviceAllocation<int4_t> _db((size_t)bd.L * _per);                                         \
    _db.copy_from_host(reinterpret_cast<int4_t const*>(_bb.data()));                                    \
    for (int S : {1, 2, 4, 8}) sk_row<TM, TN, TK, WM, WN, ST>(bd, cx, _db, S, b1, bS);                  \
  } while (0)

  std::printf("\n-- decode-shaped configs (TileK=256: a slice of 2048/8 is exactly one offline tile) --\n");
  SK_PACK_AND_RUN(16,  32, 256, 16, 16, 2);   // the recorded decode winner
  SK_PACK_AND_RUN(16,  64, 256, 16, 32, 2);
  std::printf("\n-- shallower TileK --\n");
  SK_PACK_AND_RUN(16,  32,  64, 16, 16, 4);
  SK_PACK_AND_RUN(32,  64,  64, 32, 32, 4);
  SK_PACK_AND_RUN(64, 128,  64, 32, 32, 4);   // prefill-shaped, for contrast
#undef SK_PACK_AND_RUN

  if (!sk_acu()) {
    std::printf("\n== the two winners ==\n");
    if (b1.us < 1e29) std::printf("  splitk=1 : %-34s %8.2f us\n", b1.tag, b1.us);
    else              std::printf("  splitk=1 : none ran\n");
    if (bS.us < 1e29) std::printf("  splitk>1 : %-34s %8.2f us\n", bS.tag, bS.us);
    else              std::printf("  splitk>1 : none ran\n");
    if (b1.us < 1e29 && bS.us < 1e29) {
      double const sp = b1.us / bS.us;
      std::printf("  speedup from split-K: %.3fx   (the occupancy model predicts at most 1.27x -- more than\n"
                  "                                 that means the 18 warps/CU ceiling was wrong)\n", sp);
    }
    std::printf("\n  To profile one: SPLITK_ONLY=\"<tag substring>\" SPLITK_ACU=1 acu -o splitk.report --set full -f "
                "$BIN/test_moe_splitk_bench %d %d %d %d %d %d\n", bd.L, bd.Rows, bd.N, bd.K, bd.gs, bd.mode);
  }
  std::printf("\n  launches refused: %d\n", moe_grouped_ppu::moeg_fail_count());
  return 0;
}
