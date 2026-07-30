// MULTI-EXPERT + RAGGED correctness gate for the grouped mixed-input GEMM, now with CONTIGUOUS output.
//
// Output is contiguous [total][N] via the ptr-array epilogue: ptr_D[e] = D + offs[e]*N (like DeepGemm), NOT the
// old padded [L][Mmax][N]. The grouped-L=1 kernel is a proven bit-exact oracle, so running each expert alone and
// requiring D[e]'s rows == the standalone golden validates BOTH the grouping AND the contiguous layout.
// run: ./test_moe_grouped_verify [L] [m_base] [ragged?]   (any 4th arg => ragged: m_e = m_base*(e%4+1))
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"   // preprocess_weights_for_mixed_gemm
#include "moe_grouped_ppu.cuh"

using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;

// One grouped launch (fixed 64x64x128/s3). ptr_D/stride_D/group_M are caller-built device arrays.
static void run_grouped(const half_t* A, const int4_t* B, const half_t* scales,
                        half_t** ptr_D, DStride* stride_D, int const* group_M,
                        int Mmax, int N, int K, int L, int gs,
                        GS* gsd, GS const* gsh, int const* offsets, char* ws, size_t wsb) {
  // TK must satisfy gs<=TK<=2*gs (SK=ceil(TK/gs)<=2; SK>=4 is broken). gs=128->TK=128, gs=32/64->TK=64.
  if (gs >= 128)
    moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64, 64, 128, 32, 32, 3>(
        A, B, scales, nullptr, ptr_D, stride_D, group_M, Mmax, N, K, L, gs, gsd, gsh, offsets, ws, wsb, nullptr);
  else
    moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64, 64, 64, 32, 32, 3>(
        A, B, scales, nullptr, ptr_D, stride_D, group_M, Mmax, N, K, L, gs, gsd, gsh, offsets, ws, wsb, nullptr);
}

int main(int argc, char** argv) {
  // Positional args only. A '-'-prefixed argument would atoi() to 0 and silently give L=0 or Mb=0, which runs
  // nothing and reports success -- the same vacuous-pass shape the refusal check below exists to stop.
  for (int i = 1; i < argc; ++i)
    if (argv[i][0] == '-') {
      std::printf("usage: %s [L] [Mb] [ragged?] [gs]   -- POSITIONAL, no --flags (got '%s')\n"
                  "  e.g. %s 8 1    (Mmax==1, which PPU_A_CUBE_H=1 requires)\n", argv[0], argv[i], argv[0]);
      return 64;
    }
  const int  L      = argc > 1 ? atoi(argv[1]) : 4;
  const int  Mb     = argc > 2 ? atoi(argv[2]) : 128;
  const bool ragged = argc > 3;
  const int  gs     = argc > 4 ? atoi(argv[4]) : 128;   // 128/64/32 (gs=32 needs actlize_gs32.patch)
  const int  N = 1024, K = 2048;              // %256==0 -> interleaved-256
  const int  scale_k = K / gs;

  std::vector<int> me(L), offs(L);
  int total = 0, Mmax = 0;
  for (int e = 0; e < L; ++e) { me[e] = ragged ? Mb * ((e % 4) + 1) : Mb; offs[e] = total; total += me[e]; Mmax = std::max(Mmax, me[e]); }

  // A REFUSED LAUNCH IS NOT A PASS, and until now it read as one. With PPU_A_CUBE_H=1 every launch here was
  // refused (a_row_broadcast requires Mmax<=1, and the default Mb is 128), so BOTH the run and its same-kernel
  // L=1 oracle left their buffers untouched -- identical, rel=0 everywhere, worst_e never assigned, "bad=0 ->
  // MATCH", exit 0. The tell was `worst e=-1`: no element ever became the worst, i.e. nothing was compared.
  int const fails_before = moe_grouped_ppu::moeg_fail_count();

  srand(1234);
  std::vector<float>  hA((size_t)total * K), hSc((size_t)L * N * scale_k);
  std::vector<int8_t> rawB((size_t)L * N * K);
  for (auto& a : hA)  a = (rand() % 7 - 3) * 0.25f;
  for (auto& s : hSc) s = 0.02f + (rand() % 8) * 0.01f;
  for (auto& b : rawB) b = int8_t(rand() % 15 - 7);

  std::vector<int8_t> packed((size_t)L * N * K / 2), Bbuf((size_t)L * N * K / 2);
  for (int e = 0; e < L; ++e) {
    for (int i = 0; i < N * K / 2; ++i) {
      int8_t lo = rawB[(size_t)e * N * K + 2 * i] & 0xF, hi = rawB[(size_t)e * N * K + 2 * i + 1] & 0xF;
      packed[(size_t)e * N * K / 2 + i] = int8_t((hi << 4) | lo);
    }
    preprocess_weights_for_mixed_gemm<false, 256>(
        (int8_t*)&Bbuf[(size_t)e * N * K / 2], (int8_t*)&packed[(size_t)e * N * K / 2],
        {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);
  }
  auto to_half = [](std::vector<float> const& f) { std::vector<half_t> h(f.size()); for (size_t i = 0; i < f.size(); ++i) h[i] = half_t(f[i]); return h; };
  auto out_stride = [&](int m) { return cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(m, N, 1)); };

  // ---- FULL run: all L experts, CONTIGUOUS output D[total][N] ----
  cutlass::DeviceAllocation<half_t> A((size_t)total * K), scales((size_t)L * N * scale_k), D((size_t)total * N);
  cutlass::DeviceAllocation<int4_t> B((size_t)L * N * K);
  { auto h = to_half(hA);  A.copy_from_host(h.data()); }
  { auto h = to_half(hSc); scales.copy_from_host(h.data()); }
  B.copy_from_host(reinterpret_cast<int4_t const*>(Bbuf.data()));

  std::vector<GS> rshapes(L); for (int e = 0; e < L; ++e) rshapes[e] = cute::make_shape(me[e], N, K);
  cutlass::DeviceAllocation<GS>  rdev(L);   rdev.copy_from_host(rshapes.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  // ptr-array outputs: contiguous ptr_D[e] = D + offs[e]*N
  std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);
  for (int e = 0; e < L; ++e) { pdh[e] = D.get() + (size_t)offs[e] * N; sdh[e] = out_stride(me[e]); gmh[e] = me[e]; }
  cutlass::DeviceAllocation<half_t*> pd(L);  pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L);  sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int>     gm(L);  gm.copy_from_host(gmh.data());
  const size_t ws = (size_t)cutlass::ceil_div(Mmax, 16) * cutlass::ceil_div(N, 64) * (size_t)L * 64;
  cutlass::DeviceAllocation<char> wsr(ws);

  run_grouped(A.get(), B.get(), scales.get(), pd.get(), sd.get(), gm.get(),
              Mmax, N, K, L, gs, rdev.get(), rshapes.data(), ragged ? offdev.get() : nullptr, wsr.get(), ws);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hD((size_t)total * N); D.copy_to_host(hD.data());

  // ---- ORACLE: each expert alone (L=1), compare vs the contiguous D rows ----
  half_t hD0[4] = {half_t(0.f), half_t(0.f), half_t(0.f), half_t(0.f)};   // expert 0's oracle rows, for the dump below
  // gold_absmax, not worst_e, is what decides whether this comparison had anything in it. worst_e stays -1 on a
  // BIT-EXACT pass -- rel is 0 for every element, so `rel > max_rel` never fires -- and the L=1 oracle is
  // bit-exact by construction. Keying vacuity on worst_e therefore failed a genuine pass, which is how this check
  // first reported itself wrong.
  // NaN DEFEATS EVERY COMPARISON-BASED CHECK IN THIS FILE, so it is counted explicitly rather than compared.
  // Every `if (x > y)` is FALSE when x is NaN: rel = |got-NaN|/(NaN+1e-3) is NaN, so `rel > max_rel` and
  // `rel > 5e-2` both fail, max_rel stays 0, bad stays 0, worst_e stays -1, and abs(NaN) > gold_absmax fails too.
  // A buffer full of NaN therefore printed 'max_rel=0.000e+00 bad=0 -> MATCH' on BOTH the L=1 and the cross-build
  // comparisons while gold_absmax reported the golden as all-zero -- three mutually contradictory readings that
  // only NaN reconciles. Non-finite values are now bad by definition, on either side.
  double max_rel = 0, gold_absmax = 0, got_absmax = 0; int bad = 0, worst_e = -1, nan_gold = 0, nan_got = 0;
  for (int e = 0; e < L; ++e) {
    const int Me = me[e];
    cutlass::DeviceAllocation<half_t> Ae((size_t)Me * K), Se((size_t)N * scale_k), De((size_t)Me * N);
    cutlass::DeviceAllocation<int4_t> Be((size_t)N * K);
    { std::vector<half_t> h((size_t)Me * K); for (size_t i = 0; i < (size_t)Me * K; ++i) h[i] = half_t(hA[(size_t)offs[e] * K + i]); Ae.copy_from_host(h.data()); }
    { std::vector<half_t> h((size_t)N * scale_k); for (size_t i = 0; i < (size_t)N * scale_k; ++i) h[i] = half_t(hSc[(size_t)e * N * scale_k + i]); Se.copy_from_host(h.data()); }
    Be.copy_from_host(reinterpret_cast<int4_t const*>(&Bbuf[(size_t)e * N * K / 2]));

    std::vector<GS> gs1(1, cute::make_shape(Me, N, K));
    cutlass::DeviceAllocation<GS> gs1d(1); gs1d.copy_from_host(gs1.data());
    std::vector<half_t*> pdo{De.get()}; std::vector<DStride> sdo{out_stride(Me)}; std::vector<int> gmo{Me};
    cutlass::DeviceAllocation<half_t*> pd1(1); pd1.copy_from_host(pdo.data());
    cutlass::DeviceAllocation<DStride> sd1(1); sd1.copy_from_host(sdo.data());
    cutlass::DeviceAllocation<int>     gm1(1); gm1.copy_from_host(gmo.data());
    const size_t ws1 = (size_t)cutlass::ceil_div(Me, 16) * cutlass::ceil_div(N, 64) * 64;
    cutlass::DeviceAllocation<char> ws1b(ws1);

    run_grouped(Ae.get(), Be.get(), Se.get(), pd1.get(), sd1.get(), gm1.get(),
                Me, N, K, /*L=*/1, gs, gs1d.get(), gs1.data(), /*offsets=*/nullptr, ws1b.get(), ws1);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::vector<half_t> hDe((size_t)Me * N); De.copy_to_host(hDe.data());
    if (e == 0) for (int t = 0; t < 4 && t < Me * N; ++t) hD0[t] = hDe[t];

    for (int i = 0; i < Me; ++i)
      for (int j = 0; j < N; ++j) {
        double gold = (double)float(hDe[(size_t)i * N + j]);
        double got  = (double)float(hD[((size_t)offs[e] + i) * N + j]);   // CONTIGUOUS: row offs[e]+i
        if (!std::isfinite(gold)) { ++nan_gold; ++bad; if (worst_e < 0) worst_e = e; continue; }
        if (!std::isfinite(got))  { ++nan_got;  ++bad; if (worst_e < 0) worst_e = e; continue; }
        if (std::abs(gold) > gold_absmax) gold_absmax = std::abs(gold);
        if (std::abs(got)  > got_absmax)  got_absmax  = std::abs(got);
        double rel  = std::abs(got - gold) / (std::abs(gold) + 1e-3);
        if (rel > max_rel) { max_rel = rel; worst_e = e; }
        if (rel > 5e-2) ++bad;
      }
  }
  // Oracle liveness: an all-zero golden makes every rel exactly 0 and the comparison vacuous. Independent of the
  // refusal count, because a silently-not-launched kernel is not the only way to get an untouched buffer.
  bool const oracle_dead = (gold_absmax == 0.0);

  std::printf("verify(grouping+contiguous): L=%d %s Mb=%d N=%d K=%d gs=%d total=%d Mmax=%d\n",
              L, ragged ? "ragged" : "uniform", Mb, N, K, gs, total, Mmax);
  std::printf("  grouped-L=%d(contiguous D[total][N]) vs grouped-L=1 oracle: max_rel=%.3e (worst e=%d) bad=%d -> %s\n",
              L, max_rel, worst_e, bad, bad == 0 ? "MATCH" : "MISMATCH");
  // What is actually in the buffers. Printed unconditionally: the whole point is that a verdict computed from
  // comparisons could not distinguish 'equal', 'both zero' and 'both NaN'.
  std::printf("  |gold|max=%.4g  |got|max=%.4g  non-finite: gold=%d got=%d   gold[0..3]=%g %g %g %g  got[0..3]=%g %g %g %g\n",
              gold_absmax, got_absmax, nan_gold, nan_got,
              (double)float(hD0[0]), (double)float(hD0[1]), (double)float(hD0[2]), (double)float(hD0[3]),
              (double)float(hD[0]),  (double)float(hD[1]),  (double)float(hD[2]),  (double)float(hD[3]));
  if (nan_gold || nan_got) {
    std::printf("  *** NON-FINITE OUTPUT: %d in the L=1 oracle, %d in the grouped run. FAILURE. ***\n", nan_gold, nan_got);
    return 5;
  }

  int const refused = moe_grouped_ppu::moeg_fail_count() - fails_before;
  if (refused) {
    std::printf("  *** %d LAUNCH(ES) REFUSED -- NOTHING WAS VERIFIED. This is a FAILURE, not a match. ***\n"
                "      PPU_A_CUBE_H=1 needs Mmax==1: run with Mb=1, e.g. ./test_moe_grouped_verify 8 1\n", refused);
    return 2;
  }
  // Cross-build oracle. The L=1 comparison above shares this binary's collective, so a defect common to both
  // sides is invisible to it -- exactly the hole PPU_A_CUBE_H sits in. Dump from a build without the macro, check
  // from a build with it, and the reference is a DIFFERENT compilation rather than the same one twice.
  if (const char* f = std::getenv("MOEG_DUMP")) {
    FILE* fp = std::fopen(f, "wb");
    if (!fp) { std::printf("  MOEG_DUMP: cannot open %s\n", f); return 4; }
    std::fwrite(hD.data(), sizeof(half_t), hD.size(), fp);
    std::fclose(fp);
    std::printf("  MOEG_DUMP: %zu halfs -> %s\n", hD.size(), f);
  }
  if (const char* f = std::getenv("MOEG_CHECK")) {
    FILE* fp = std::fopen(f, "rb");
    if (!fp) { std::printf("  MOEG_CHECK: cannot open %s\n", f); return 4; }
    std::vector<half_t> ref(hD.size());
    size_t const rd = std::fread(ref.data(), sizeof(half_t), ref.size(), fp);
    std::fclose(fp);
    if (rd != ref.size()) { std::printf("  MOEG_CHECK: %s has %zu halfs, expected %zu\n", f, rd, ref.size()); return 4; }
    int xbad = 0, xnan = 0; double xmax = 0, ref_absmax = 0; size_t xi = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      double g = (double)float(ref[i]), o = (double)float(hD[i]);
      // 'g != 0' was TRUE for NaN, which is how an all-NaN reference passed itself off as live.
      if (!std::isfinite(g) || !std::isfinite(o)) { ++xnan; ++xbad; continue; }
      if (std::abs(g) > ref_absmax) ref_absmax = std::abs(g);
      double r = std::abs(o - g) / (std::abs(g) + 1e-3);
      if (r > xmax) { xmax = r; xi = i; }
      if (r > 5e-2) ++xbad;
    }
    std::printf("  MOEG_CHECK: |ref|max=%.4g non-finite=%d\n", ref_absmax, xnan);
    if (ref_absmax == 0) { std::printf("  *** MOEG_CHECK: the reference has no finite nonzero value -- it verifies nothing. ***\n"); return 3; }
    std::printf("  cross-build vs %s: max_rel=%.3e (worst idx=%zu) bad=%d -> %s\n",
                f, xmax, xi, xbad, xbad == 0 ? "MATCH" : "MISMATCH");
    if (xbad) return 1;
  }
  if (oracle_dead) {
    std::printf("  *** VACUOUS: the oracle never produced a nonzero value, so rel==0 everywhere. FAILURE. ***\n");
    return 3;
  }

  return bad == 0 ? 0 : 1;
}
