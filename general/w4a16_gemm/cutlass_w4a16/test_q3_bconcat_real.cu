// Q3_K B-CONCAT (ONE GEMM, two B bit planes) on REAL GGUF weights [box-only].
//
// The dedicated 2-plane mainloop loads BOTH planes and combines them in the converter, so a real Q3_K weight runs
// as a SINGLE GEMM while staying 3-bit in memory:
//     low  plane: int2  = q & 3            (drives the main swzl / tCrB_mma)
//     high plane: int1  = q >> 2
//     converter emits fp16(low + 4*high) -> the affine then does dl*q + zero, zero = -4*dl (Q3_K's -4 center)
// so ONLY sc_lo (= dl) and zr_lo (= -4*dl) are used here; sc_hi from the .bin is NOT needed, because the factor 4
// is baked into the high bit's mantissa placement (b+2) instead of into a second scale.
//
// Compare against the A-concat (two GEMMs summed, test_q3_concat_real): same golden, but that costs 2x the mma.
//   .bin: <4 i32: M,N,K,gs> A[M*K]f16 | low2[K*N]u8 | high1[K*N]u8 | sc_lo|zr_lo|sc_hi[(K/gs)*N]f16 | gold[M*N]f16
//   Build: TARGET=test_q3_bconcat_real ./build.sh ; run: ./<bin> real_weight/real_q3k_concat.bin
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "xplane_offline.hpp"
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
using uint2_t = cutlass::uint2b_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

template <class T> static std::vector<T> rd(FILE* f, size_t n) {
  std::vector<T> v(n);
  if (fread(v.data(), sizeof(T), n, f) != n) { std::printf("short read\n"); exit(1); }
  return v;
}

// transpose q [K][N] -> [N][K], pack ELTS_PER_BYTE per byte, run the offline preprocess for that plane's format
template <int ELTS_PER_BYTE, QuantTypeClass QTC, int FoldTN = 0, int FoldTK = 0>
static std::vector<int8_t> pack_plane(const std::vector<uint8_t>& q, int K, int N) {
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n * K + k] = q[(size_t)k * N + n];
  const int BITS = 8 / ELTS_PER_BYTE, MASK = (1 << BITS) - 1;
  std::vector<int8_t> packed((size_t)K * N / ELTS_PER_BYTE, 0);
  for (size_t i = 0; i < packed.size(); ++i) {
    int8_t b = 0;
    for (int t = 0; t < ELTS_PER_BYTE; ++t) b |= int8_t((qT[ELTS_PER_BYTE * i + t] & MASK) << (BITS * t));
    packed[i] = b;
  }
  std::vector<int8_t> out(packed.size());
  preprocess_weights_for_mixed_gemm<false, 256>(out.data(), packed.data(), {(size_t)K, (size_t)N}, QTC);
  // PER-PLANE N-FOLD: fold this plane iff ITS contiguous run at FoldTK is under the AIU's 32 B minimum. FoldTK == 0
  // means "no fold", byte-identical to what shipped -- which is what keeps the Block_K=256 run below a true control.
  // (f) THE FOLD BRANCH IS GONE. Every call passes FoldTK = 0 -- folded planes are built by xplane::place_derived /
  // place_int1 inside the row macros, per configuration, because the map depends on the tile. What is left here is the
  // unfolded path, which the derived preprocess_weights_for_mixed_gemm covers on its own.
  return out;
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "real_weight/real_q3k_concat.bin";
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::printf("cannot open %s\n", path); return 1; }
  int32_t hdr[4]; if (fread(hdr, 4, 4, f) != 4) { std::printf("bad header\n"); return 1; }
  const int M = hdr[0], N = hdr[1], K = hdr[2], gs = hdr[3], L = 1;
  const int scale_k = K / gs;
  std::printf("[q3-bconcat-real] %s  M=%d N=%d K=%d gs=%d  (ONE GEMM, int2 low + int1 high)\n",
              path, M, N, K, gs);

  auto A_h    = rd<uint16_t>(f, (size_t)M * K);
  auto low2   = rd<uint8_t> (f, (size_t)K * N);
  auto high1  = rd<uint8_t> (f, (size_t)K * N);
  auto sc_lo  = rd<uint16_t>(f, (size_t)scale_k * N);
  auto zr_lo  = rd<uint16_t>(f, (size_t)scale_k * N);
  /*sc_hi unused*/ rd<uint16_t>(f, (size_t)scale_k * N);
  auto gold_h = rd<uint16_t>(f, (size_t)M * N);
  std::fclose(f);

  auto Blo = pack_plane<4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(low2,  K, N);
  auto Bhi = pack_plane<8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(high1, K, N);

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dBlo((size_t)K*N);
  cutlass::DeviceAllocation<uint1_t> dBhi((size_t)K*N);
  dA.copy_from_host(reinterpret_cast<half_t const*>(A_h.data()));
  dSc.copy_from_host(reinterpret_cast<half_t const*>(sc_lo.data()));
  dZr.copy_from_host(reinterpret_cast<half_t const*>(zr_lo.data()));
  dBlo.copy_from_host(reinterpret_cast<uint2_t const*>(Blo.data()));
  dBhi.copy_from_host(reinterpret_cast<uint1_t const*>(Bhi.data()));

  std::vector<GS> shp(L, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  std::vector<half_t*> pdh{dD.get()};
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M, N, 1))};
  std::vector<int> gmh{M}, offs{0};
  cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(L);     gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  const size_t wsb = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*(size_t)L*64;
  cutlass::DeviceAllocation<char> ws(wsb);

  // Shared Block_K = 256: bounded below by the SPARSEST plane's AIU 32B minimum (int1 -> K/8 >= 32 -> K >= 256).
  // PlaneB2 = uint1b_t as the 8th template arg routes the builder to the 2-plane mainloop; the high plane's device
  // pointer is the trailing argument.
  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3,
                                  cutlass::uint2b_t, cutlass::uint1b_t>(
      dA.get(), dBlo.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr,
      dBhi.get());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  auto check = [&](const char* tag) {
    std::vector<half_t> h((size_t)M*N); dD.copy_to_host(h.data());
    const half_t* g = reinterpret_cast<const half_t*>(gold_h.data());
    int b_ = 0; double mr = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
      double got = (double)float(h[i]), exp = (double)float(g[i]);
      double rel = std::abs(got - exp) / (std::abs(exp) + 1e-3);
      if (rel > mr) mr = rel;
      if (std::abs(got - exp) > 2e-2 + 6e-2 * std::abs(exp)) ++b_;
    }
    std::printf("  %-38s bad=%d/%d max_rel=%.3e %s\n", tag, b_, M * N, mr, b_ == 0 ? "MATCH" : "MISMATCH");
    return b_ == 0;
  };
  bool ok256 = check("Block_K=256 (unfolded, CONTROL)");

  // A LADDER, one variable per rung. The Block_K=64 attempt moved THREE at once (TN 64->128, WN 32->64, TK 128->64
  // with both planes folding) and came back bad=13689/32768, which localises nothing. The composition check cannot
  // separate them either: it DEFINES the high plane's placement, so testing the pairing against it is circular. A
  // numeric ladder does separate them, in one box round -- the same technique that localised int4's unexplained
  // 10-point drop.
  //
  // Every rung drives the high plane through xplane::place_int1, which is byte-identical to pack_plane's fold at
  // F1=1/F2=2 (l47, l53 at both K=512 and the box's K=256). So rung 1 is the known-MATCH configuration and also
  // double-checks that byte-identity on the box; if rung 1 breaks, place_int1 is the problem and nothing else matters.
#define RUNG(TAG, TMv, TNv, TKv, WMv, WNv, Sv, F1v, F2v, QMODEv) do {                                        \
  /* PLANE 1: once cols_per_word > 1 (WN=64) the whole-word packer cannot express the fragment, and it also    \
     groups the folded columns strided while the kernel's MmaView groups them adjacent. So drive plane 1 off the  \
     SAME derived map plane 2 already uses. Byte-identical to the whole-word packer wherever that one is          \
     box-validated -- gated in fold_derivation/l61. */                                                           \
  std::vector<int8_t> blo_((size_t)K * N / 4);                                                        \
  if constexpr ((F1v) > 1) {                                                                          \
    xplane::place_derived<2, TMv, TNv, TKv, WMv, WNv, F1v>(blo_.data(), low2, N, K);                  \
  } else {                                                                                            \
    blo_ = pack_plane<4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY, TNv, 0>(low2, K, N);                \
  }                                                                                                   \
  cutlass::DeviceAllocation<cutlass::uint2b_t> dblo_((size_t)K*N);                                   \
  dblo_.copy_from_host(reinterpret_cast<cutlass::uint2b_t const*>(blo_.data()));                     \
  std::vector<int8_t> bhi_((size_t)K * N / 8);                                                       \
  xplane::place_int1<TMv, TNv, TKv, WMv, WNv, F2v, F1v>(bhi_.data(), high1, N, K);                   \
  cutlass::DeviceAllocation<cutlass::uint1b_t> dbhi_((size_t)K*N);                                   \
  dbhi_.copy_from_host(reinterpret_cast<cutlass::uint1b_t const*>(bhi_.data()));                     \
  CUTLASS_PPU_CHECK(hggcMemset(dD.get(), 0, sizeof(half_t) * (size_t)M * N));                        \
  moe_grouped_ppu::filter_and_run<QMODEv, TMv, TNv, TKv, WMv, WNv, Sv,                                \
                                  cutlass::uint2b_t, cutlass::uint1b_t>(                             \
      dA.get(), dblo_.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                     \
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr, dbhi_.get());    \
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());                                                        \
  if (!check(TAG)) { if (first_bad < 0) first_bad = rung_no; }                                       \
  ++rung_no;                                                                                         \
} while (0)

  // ---------------------------------------------------------------- CONTROLLED-INPUT PROBE
  // PROBE=hi / PROBE=lo reads the COLUMN PERMUTATION straight off the hardware, which is what is left once the ladder
  // has narrowed to one rung whose remaining variables (TK, F1, F2) cannot be separated by configuration -- TK decides
  // both fold factors. Same instrument that cracked the int2 N-half pairing bug.
  //
  //   scale = 1, zero = 0, one plane all-zero, the other set to high[k][n] = delta(k, n), and A[m][k] = k. Then
  //       D[m][n] = sum_k A[m][k] * (low + 4*high) * s = MULT * sigma(n)
  //   so D[0][n]/MULT is literally "which column's bit column n actually received". A correct map prints n. Any other
  //   value IS the permutation, read directly, no derivation.
  if (const char* pr = std::getenv("PROBE")) {
    const bool hi_mode = (pr[0] == 'h');
    const int MULT = hi_mode ? 4 : 1;
    std::printf("  [PROBE=%s] rung-5 config (64,128,64) w64x64 F1=2 F2=4 -- D[0][n]/%d should equal n\n",
                hi_mode ? "hi" : "lo", MULT);
    std::vector<uint8_t> plo((size_t)K*N, 0), phi((size_t)K*N, 0);
    for (int n = 0; n < N && n < K; ++n) (hi_mode ? phi : plo)[(size_t)n * N + n] = 1;   // delta(k, n)
    { std::vector<half_t> a((size_t)M*K), s1((size_t)scale_k*N, half_t(1.f)), z0((size_t)scale_k*N, half_t(0.f));
      for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) a[(size_t)m*K + k] = half_t(float(k));
      dA.copy_from_host(a.data()); dSc.copy_from_host(s1.data()); dZr.copy_from_host(z0.data()); }
    std::vector<int8_t> blo_((size_t)K * N / 4);
    xplane::place_derived<2, 64, 128, 64, 64, 64, 2>(blo_.data(), plo, N, K);
    cutlass::DeviceAllocation<cutlass::uint2b_t> dblo_((size_t)K*N);
    dblo_.copy_from_host(reinterpret_cast<cutlass::uint2b_t const*>(blo_.data()));
    std::vector<int8_t> bhi_((size_t)K * N / 8);
    xplane::place_int1<64, 128, 64, 64, 64, 4, 2>(bhi_.data(), phi, N, K);
    cutlass::DeviceAllocation<cutlass::uint1b_t> dbhi_((size_t)K*N);
    dbhi_.copy_from_host(reinterpret_cast<cutlass::uint1b_t const*>(bhi_.data()));
    CUTLASS_PPU_CHECK(hggcMemset(dD.get(), 0, sizeof(half_t) * (size_t)M * N));
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 128, 64, 64, 64, 3,
                                    cutlass::uint2b_t, cutlass::uint1b_t>(
        dA.get(), dblo_.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
        M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr, dbhi_.get());
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::vector<half_t> h((size_t)M*N); dD.copy_to_host(h.data());
    int wrong = 0;
    for (int n = 0; n < N; ++n) {
      const int got = int(std::lround(double(float(h[(size_t)0*N + n])) / MULT));
      if (got != n) { if (wrong < 24) std::printf("    n=%3d -> %4d   (delta %+d)\n", n, got, got - n); ++wrong; }
    }
    std::printf("  => %d / %d columns permuted%s\n", wrong, N,
                wrong ? "  <-- the deltas above ARE the map error" : "  (this plane's column map is correct)");
    return wrong == 0 ? 0 : 1;
  }

  int rung_no = 1, first_bad = -1;
  RUNG("1 (64, 64,128) w32x32 F1=1 F2=2  BASE", 64,  64, 128, 32, 32, 3, 1, 2, QM::FinegrainedScaleZero);
  RUNG("2 (64,128,128) w32x32   TN 64->128 ", 64, 128, 128, 32, 32, 3, 1, 2, QM::FinegrainedScaleZero);
  RUNG("3 (64,128,128) w32x64   WN 32->64  ", 64, 128, 128, 32, 64, 3, 1, 2, QM::FinegrainedScaleZero);
  RUNG("4 (64,128,128) w64x64   WM 32->64  ", 64, 128, 128, 64, 64, 3, 1, 2, QM::FinegrainedScaleZero);
  RUNG("5 (64,128, 64) w64x64   TK 128->64, F1=2 F2=4", 64, 128, 64, 64, 64, 3, 2, 4, QM::FinegrainedScaleZero);
  // PLAN #20 PHASE 1 GATE. Same real Q3_K weights, same native golden, but NO ZERO CHANNEL: the -4*dl that rungs 1-5
  // carry in `zero` is now the converter's own additive immediate (kSymBias2Plane<2,1> == 4, l92). The zero buffer is
  // still passed and still holds -4*dl, so if ScaleOnly wrongly consulted it the result would be off by exactly that
  // and this rung would fail -- i.e. the check catches both a missing bias and a double-applied one.
  //
  // This, NOT test_lowbit_grouped, is the honest gate: that harness's oracle is the same kernel at L=1 (it isolates
  // per-expert addressing), so it cannot see a wrong dequant constant at all. Only a harness with an INDEPENDENT
  // golden can, and this file has one straight out of the gguf.
  RUNG("6 (64,128,128) w64x64   ScaleOnly, bias in converter", 64, 128, 128, 64, 64, 3, 1, 2,
       QM::FinegrainedScaleOnly);
  RUNG("7 (64,128, 64) w64x64   ScaleOnly + F1=2 F2=4      ", 64, 128, 64, 64, 64, 3, 2, 4,
       QM::FinegrainedScaleOnly);
#undef RUNG
  std::printf("  => control %s; ladder: %s\n", ok256 ? "OK" : "BROKEN (look there first)",
              first_bad < 0 ? "all rungs MATCH" :
              first_bad == 1 ? "rung 1 already wrong -- place_int1 or the base config, not the ladder" :
                               "first failure at rung -- the variable that rung introduces is the cause");
  if (first_bad > 1) std::printf("     first failing rung: %d\n", first_bad);

  // dD still holds the LAST rung's output, so the per-element dump below shows that rung's error shape.
  std::vector<half_t> hD((size_t)M*N); dD.copy_to_host(hD.data());
  const half_t* gold = reinterpret_cast<const half_t*>(gold_h.data());
  int bad = 0, shown = 0; double maxrel = 0;
  for (size_t i = 0; i < (size_t)M * N; ++i) {
    double got = (double)float(hD[i]), exp = (double)float(gold[i]);
    double rel = std::abs(got - exp) / (std::abs(exp) + 1e-3);
    if (rel > maxrel) maxrel = rel;
    if (std::abs(got - exp) > 2e-2 + 6e-2 * std::abs(exp)) ++bad;
  }
  std::printf("  last rung (%d) vs native Q3_K golden: bad=%d/%d max_rel=%.3e %s\n", rung_no - 1,
              bad, M * N, maxrel, bad == 0 ? "MATCH" : "MISMATCH");
  // On MISMATCH the SHAPE of the error localizes it: a whole-K shift or a low/high N-half swap points at the
  // plane-2 swzl lane delivery (the one thing the local xplane.py check could not cover); a per-element factor of
  // ~4 or a 0/4 offset points at the high bit's mantissa placement instead.
  for (int m = 0; m < M && shown < 8; ++m) for (int n = 0; n < N && shown < 8; ++n) {
    size_t i = (size_t)m * N + n;
    double got = (double)float(hD[i]), exp = (double)float(gold[i]);
    if (std::abs(got - exp) > 2e-2 + 6e-2 * std::abs(exp)) {
      std::printf("    m=%d n=%d | got=%.4f exp=%.4f ratio=%.3f\n", m, n, got, exp, exp != 0 ? got / exp : 0.0);
      ++shown;
    }
  }
  return bad == 0 ? 0 : 1;
}
