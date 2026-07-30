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

// SK_QUANT: which quantisation channel the row uses, so its cost can be measured by removing it.
//   2 (default) FinegrainedScaleZero -- per-group scale AND zero, what ships
//   1           FinegrainedScaleOnly -- drops the zero: one fewer f16x2 pass per atom and no Z smem reload
//   0           PerColScaleOnly      -- drops the per-GROUP reload entirely, one scale per column read once, so the
//                                      FINE path's 8 dependent smem loads per k-tile disappear. This is the one that
//                                      isolates what TODO #11 (prefetch the next group's scale) could ever be worth.
// The tag carries it so a log cannot be mistaken for the default -- the same lesson as the A-path banner.
#ifndef SK_QUANT
#define SK_QUANT 2
#endif
#if SK_QUANT == 2
#define SK_QUANT_MODE QM::FinegrainedScaleZero
#define SK_QUANT_NAME "sz"
#elif SK_QUANT == 1
#define SK_QUANT_MODE QM::FinegrainedScaleOnly
#define SK_QUANT_NAME "s "
#else
#define SK_QUANT_MODE QM::PerColScaleOnly
#define SK_QUANT_NAME "pc"
#endif


inline const char* sk_only() { return std::getenv("SPLITK_ONLY"); }
inline bool sk_acu() { return std::getenv("SPLITK_ACU") != nullptr; }
// A/B IN ONE BINARY, so the comparison cannot be two different builds. Off by default: it changes the answer
// for any expert with more than one row, and launch() refuses it above Mmax == 1.
// SPLITK_ABCAST is gone with the a_row_broadcast parameter it drove: that zeroed the AIU descriptor's row PITCH
// (dim_w) to request what dim_h already provides, and returned NaN. See moe_grouped_ppu.cuh.
inline double pct_of(double gbs) { return 100.0 * gbs / HBM_GBS; }
// SPLITK_ONLY matches the whole tag, which means counting the spaces the format string pads out -- and that
// already cost an acu run ("No kernels were profiled") after the abcast marker changed the spacing from two to
// three. SPLITK_CFG and SPLITK_S match the two halves independently, so a capture never depends on whitespace.
// Same fix as GEMV_FMT / GEMV_CFG on the GEMV side, which this should have copied in the first place.
inline bool sk_selected(const char* tag) {
  const char* f = sk_only();
  if (f && std::strstr(tag, f) == nullptr) return false;
  const char* cfg = std::getenv("SPLITK_CFG");          // e.g. "16x128:256 w16x16 s2"
  if (cfg && std::strstr(tag, cfg) == nullptr) return false;
  const char* sv = std::getenv("SPLITK_S");             // e.g. "2"
  if (sv) {
    char want[16];
    std::snprintf(want, sizeof(want), "S=%s", sv);
    size_t const n = std::strlen(want);
    const char* p = std::strstr(tag, want);
    // exact match on the S value: "S=1" must not select "S=16"
    if (!p || (p[n] != '\0' && p[n] != ' ')) return false;
  }
  return true;
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
  cutlass::DeviceAllocation<half_t>* dRef;      // this config's S=1 output, for the S>1 comparison
  cutlass::DeviceAllocation<half_t*>* pdOne;    // the plain L-entry ptr_D, for slices == 1
  cutlass::DeviceAllocation<DStride>* sdAll;    // L*S_MAX strides (the L strides repeated)
  cutlass::DeviceAllocation<DStride>* sdOne;    // the plain L-entry stride_D
  int S_max;
};

// One (config, S) row.
template <int TM, int TN, int TK, int WM, int WN, int Stages>
inline void sk_row(Band const& bd, SkCtx const& cx, cutlass::DeviceAllocation<int4_t>& dB,
                   int slices, SkBest& b1, SkBest& bS) {
  if constexpr (!moe_ok<TM, TN, TK, WM, WN, Stages, 4>()) { (void)slices; return; }
  else {
    char tag[80];
    std::snprintf(tag, sizeof(tag), "i4 %s %dx%d:%d w%dx%d s%d  S=%d", SK_QUANT_NAME, TM, TN, TK, WM, WN, Stages, slices);
    if (!sk_selected(tag)) return;
    const char* why = "";
    if (!moe_splitk_ppu::splitk_ok(bd.K, slices, TK, &why)) {
      std::printf("  %-34s %10s | ILLEGAL: %s\n", tag, "-", why);
      return;
    }

    auto go = [&] {
      moe_splitk_ppu::launch_splitk<SK_QUANT_MODE, TM, TN, TK, WM, WN, Stages, int4_t>(
          bd.dA, dB.get(), bd.dSc, bd.dZr,
          cx.dD->get(), cx.dPart->get(),
          slices == 1 ? cx.pdOne->get() : cx.pdAll->get(),
          slices == 1 ? cx.sdOne->get() : cx.sdAll->get(), bd.gm,
          bd.Mmax, bd.N, bd.K, bd.L, bd.gs, slices,
          bd.rdev, bd.rsh.data(),
          bd.mode ? bd.offdev : nullptr, bd.total,
          // B2 IS LEFT TO ITS DEFAULT ON PURPOSE. Passing `nullptr` explicitly makes deduction of PlaneB2 from
          // std::nullptr_t fail, and a failed deduction is not rescued by the parameter's default template
          // argument -- so the call stops matching. The single-TU version worked only because it never named B2.
          bd.ws, bd.wsb, hggcStream_t(0));
    };

    int const f0 = moe_grouped_ppu::moeg_fail_count();
    double us;
    if (sk_acu()) { us = time_it(go, 0); std::printf("  [acu] ONE COLD launch (not a timing): %s\n", tag); return; }
    us = time_it(go, 20);
    if (moe_grouped_ppu::moeg_fail_count() != f0) {
      std::printf("  %-34s %10s | DID NOT RUN (launch refused) -- excluded\n", tag, "-");
      return;
    }

    // CORRECTNESS, NOT JUST TIME. The split is STRIDED in k-tiles, so the per-group scale has to be addressed
    // from the k-tile coordinate rather than by advancing a pointer per tile -- and nothing here had checked
    // that. S=1 stashes this config's output as the reference; every S>1 row is compared against it, and a row
    // that fails is EXCLUDED from the winner rather than reported as fast (the same net as moe_row_ran).
    {
      // A NAMED COUNT, because `std::vector<half_t> got(size_t(nelem))` is the most vexing parse: it declares a
      // FUNCTION taking size_t and returning a vector, and the errors it produces name copy_to_host instead.
      size_t const nelem = size_t(bd.total) * size_t(bd.N);
      std::vector<half_t> got(nelem);
      cutlass::device_memory::copy_to_host(got.data(), cx.dD->get(), nelem);
      if (slices == 1) {
        cutlass::device_memory::copy_to_device(cx.dRef->get(), got.data(), nelem);
      } else {
        std::vector<half_t> ref(nelem);
        cutlass::device_memory::copy_to_host(ref.data(), cx.dRef->get(), nelem);
        // Non-finite values are COUNTED, never compared. d = |NaN - NaN| is NaN and `NaN > 4e-3` is false, so an
        // all-NaN pair on both sides used to score bad=0 and be reported as agreeing with S=1. std::max(x, NaN)
        // also returns x, so `worst` and `maxr` stayed clean while the data was garbage. This check called an
        // all-NaN kernel numerically correct in test_moe_grouped_verify's sibling case; same shape of bug.
        double maxr = 1e-30, worst = 0.0; int bad = 0, nonfinite = 0;
        for (size_t i = 0; i < nelem; ++i) {
          double const r = double(float(ref[i]));
          if (std::isfinite(r)) maxr = std::max(maxr, std::fabs(r)); else ++nonfinite;
        }
        for (size_t i = 0; i < nelem; ++i) {
          double const g = double(float(got[i])), r = double(float(ref[i]));
          if (!std::isfinite(g) || !std::isfinite(r)) { ++bad; continue; }
          double const d = std::fabs(g - r);
          worst = std::max(worst, d);
          if (d / maxr > 4e-3) ++bad;
        }
        if (nonfinite)
          std::printf("  %-34s %8s | %d NON-FINITE values in the S=1 reference itself\n", tag, "-", nonfinite);
        if (bad) {
          std::printf("  %-34s %8.2f us | NUMERICS: %d/%lld differ from S=1 by >4e-3 rel "
                      "(worst abs %.3g, |ref|max %.3g) -- EXCLUDED\n",
                      tag, us, bad, (long long)nelem, worst, maxr);
          return;
        }
      }
    }

    // Traffic. Weights, scales and A are each read ONCE in total regardless of S: the slices partition the
    // k-tiles, they do not repeat them. Only the partial buffer is new -- written by the epilogue and read back
    // by the merge. (The host-slicing form did re-read A per slice; the in-kernel form does not.)
    double const wb = double(bd.N) * bd.K * 4 / 8.0;                      // int4 codes
    double const sb = double(bd.scale_k) * bd.N * 2.0 * 2.0;              // scale + zero, whole matrix
    double const ab = double(bd.total) * bd.K * 2.0;
    double const pb = (slices > 1) ? (2.0 * double(slices) * bd.total * bd.N * 2.0) : 0.0;  // write + read back
    double const db = double(bd.total) * bd.N * 2.0;
    double const bytes = double(bd.active) * (wb + sb) + ab + pb + db;
    double const gbs = bytes / (us * 1e-6) / 1e9;

    // TWO CTA NUMBERS, KEPT APART. `cta/L` is the grid of ONE launch -- unchanged by S, because slicing K does
    // not add CTAs to a launch -- and it is the only one occupancy depends on. `tot` sums across the S launches.
    // The first version of this printed only the sum, which made the grid look like it grew 8x with S when the
    // concurrent grid was constant; that is the number that made a serialised implementation look plausible.
    int mt = 0;
    for (int e = 0; e < bd.L; ++e) mt += (bd.me[e] + TM - 1) / TM;
    // ONE launch now, with the slice on gridDim.z, so the concurrent grid IS mt*ntile*S.
    int64_t const cta_per_launch = int64_t(mt) * ((bd.N + TN - 1) / TN) * slices;
    int64_t const cta_total = cta_per_launch;
    double const wkwrp_cu = double(cta_per_launch) * (double(TM / WM) * (TN / WN)) / 72.0;

    std::printf("  %-34s %8.2f us | %7.1f GB/s | %5.1f%% HBM | cta/L %5lld tot %6lld | wkwrp/CU %5.1f%s\n",
                tag, us, gbs, pct_of(gbs), (long long)cta_per_launch, (long long)cta_total, wkwrp_cu,
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
