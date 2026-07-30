// Shared harness for the grouped split-K bench. The config rows live in one generated .cu each, NOT here --
// every row instantiates the AIU mixed-input collective, and on the box the expensive stages are LLVM opt/llc,
// single-threaded and minutes per kernel. Five rows in one translation unit is five of those in series.
//
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

#pragma once

#include "lowbit_moe_bench.hpp"     // Band, time_it, moe_ok, HBM_GBS, PEAK
#include "moe_splitk_ppu.cuh"

inline const char* sk_only() { return std::getenv("SPLITK_ONLY"); }
inline bool sk_acu() { return std::getenv("SPLITK_ACU") != nullptr; }
inline double pct_of(double gbs) { return 100.0 * gbs / HBM_GBS; }
inline bool sk_selected(const char* tag) {
  const char* f = sk_only();
  return !f || std::strstr(tag, f) != nullptr;
}

// Two winners, kept apart: the whole question is whether S>1 beats S==1, and one combined "best" would hide it.
struct SkBest { char tag[80] = ""; double us = 1e30; int S = 0; };
inline void sk_upd(SkBest& b, const char* t, double us, int S) {
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
inline void sk_row(Band const& bd, SkCtx const& cx, cutlass::DeviceAllocation<int4_t>& dB,
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


// PACK + THE FULL S LADDER FOR ONE CONFIG. The offline pack depends on (TM,TN,TK,WM,WN) through the fold
// factor, so unlike the GEMV sweep it cannot be hoisted to a group -- each unit owns its buffer. The S values
// are a RUNTIME loop: split-K does not change the instantiation, so sweeping S costs no extra compile.
template <int TM, int TN, int TK, int WM, int WN, int Stages>
inline void sk_config(Band const& bd, SkCtx const& cx, SkBest& b1, SkBest& bS) {
  if constexpr (moe_ok<TM, TN, TK, WM, WN, Stages, 4>()) {
    constexpr int _F = moe_fold<4>(TK);
    size_t const per = (size_t)bd.K * bd.N * 4 / 8;
    std::vector<int8_t> bb((size_t)bd.L * per);
    {
      std::vector<uint8_t> q((size_t)bd.K * bd.N);
      for (size_t i = 0; i < q.size(); ++i) q[i] = uint8_t((i * 2654435761u >> 5) & 0xFu);
      xplane::place_derived<4, TM, TN, TK, WM, WN, _F>(bb.data(), q, bd.N, bd.K);
      for (int e = 1; e < bd.L; ++e) std::memcpy(bb.data() + (size_t)e * per, bb.data(), per);
    }
    cutlass::DeviceAllocation<int4_t> db((size_t)bd.L * per);
    db.copy_from_host(reinterpret_cast<int4_t const*>(bb.data()));
    for (int S : {1, 2, 4, 8}) sk_row<TM, TN, TK, WM, WN, Stages>(bd, cx, db, S, b1, bS);
  } else {
    std::printf("  i4 %dx%d:%d w%dx%d s%d  -- ILLEGAL SHAPE (moe_ok), not built\n",
                TM, TN, TK, WM, WN, Stages);
  }
}
