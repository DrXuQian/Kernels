// P1.3f: FIRST box run of the N-FOLD path -- single-plane int2 folded to TileShape.K=64 (FoldF=2, so the B operand's
// AIU contiguous run is 2*64=128 elements = 32B, reusing the VALIDATED int2@TK128 AiuContElemSize/swzl/converter).
//
// Why int2-fold@TK64 first (not the concat): it is the simplest possible fold, and its target geometry is exactly
// int4@TK64 which measures 55.8% MFU (gs=32) / 53.1% (gs=16) on this box, vs int2@TK128's 37.1% / 30.9%. So a working
// fold should move int2 from ~37% toward ~55%.
//
// Correctness: A=identity, scale=1, zero=0 => D[m][n] == the dequantized weight at (m,n), so a wrong fold shows up as
// a permutation/garbage immediately (same controlled-input trick that cracked the sigma_n and 2-plane bugs).
//   Build: TARGET=test_fold_int2 ./build.sh ; run: ./<bin> [N] [K] [gs]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
using uint2_t = cutlass::uint2b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

static int M, N, K, gs;

int main(int argc, char** argv) {
  N  = argc > 1 ? atoi(argv[1]) : 256;
  K  = argc > 2 ? atoi(argv[2]) : 256;
  gs = argc > 3 ? atoi(argv[3]) : 32;
  M = K;                                   // identity A
  const int scale_k = K / gs;
  std::printf("[fold-int2] M=K=%d N=%d gs=%d  FOLD: TileShape.K=64, FoldF=2 (operand run = 128 elems = 32B)\n",
              M, N, gs);

  // q2 codes, transposed to [N][K] and packed 4/byte, then the OFFLINE FOLD (FoldTK=64) in preprocess.
  // LABELLED input (argv[4]): make the weight code SPELL OUT its own source index, so a wrong fold reads back as a
  // decodable (k,n) instead of a value we have to guess-match. int2 holds only 4 values, so label one 2-bit field at
  // a time: mode 1 -> q = (n >> shift) & 3, mode 2 -> q = (k >> shift) & 3 (shift from argv[5]).
  //   run: <bin> N K gs 1 0   # decodes n bits 0-1 that the kernel actually consumed
  const int label_mode  = argc > 4 ? atoi(argv[4]) : 0;
  const int label_shift = argc > 5 ? atoi(argv[5]) : 0;
  std::vector<uint8_t> q((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
    size_t i = (size_t)k * N + n;
    q[i] = label_mode == 1 ? (uint8_t)((n >> label_shift) & 3)
         : label_mode == 2 ? (uint8_t)((k >> label_shift) & 3)
                           : (uint8_t)((i * 7 + i / 13) % 4);
  }
  if (label_mode) std::printf("  LABEL mode=%d shift=%d: q encodes %s bits %d-%d\n", label_mode, label_shift,
                              label_mode == 1 ? "n" : "k", label_shift, label_shift + 1);
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n * K + k] = q[(size_t)k * N + n];
  std::vector<int8_t> packed((size_t)K * N / 4, 0);
  for (size_t i = 0; i < packed.size(); ++i) {
    int8_t b = 0;
    for (int t = 0; t < 4; ++t) b |= int8_t((qT[4 * i + t] & 0x3) << (2 * t));
    packed[i] = b;
  }
  std::vector<int8_t> Bp(packed.size());
  // FoldTK=64: the N-fold step (nfold_column_pairs_ppu) runs after interleave-256.
  // DERIVED placement (nfold_place_derived_int2) INSTEAD of the standard relayout: it maps each logical (n,k) of a
  // 64x64 tile straight to the physical (word, crumb) the folded kernel reads. Verified locally: 4096 logical
  // positions -> 4096 distinct physical positions, no collisions/misses. Set NFOLD_STD=1 to fall back to the standard
  // relayout for an A/B (that baseline gave n=0..31 correct at k=0, everything else wrong).
  // Standard preprocess FIRST (all 5 steps), then the fold is a WORD-LEVEL regroup on its output that preserves the
  // pipeline's own crumb order -- see nfold_regroup_words_int2. NFOLD_STD=1 skips the regroup (baseline A/B).
  preprocess_weights_for_mixed_gemm<false, 256, 0>(
      Bp.data(), packed.data(), {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY);
  if (!getenv("NFOLD_STD")) {
    std::vector<int8_t> tmp(Bp.size());
    nfold_regroup_gmem_int2(tmp.data(), Bp.data(), {(size_t)K, (size_t)N}, /*fold_tn=*/64, /*fold_tk=*/64);
    Bp.swap(tmp);
  }

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dB((size_t)K*N);
  { std::vector<half_t> a((size_t)M*K, half_t(0.f));
    for (int m = 0; m < M; ++m) a[(size_t)m*K + m] = half_t(1.f);
    std::vector<half_t> s((size_t)scale_k*N, half_t(1.f)), z((size_t)scale_k*N, half_t(0.f));
    dA.copy_from_host(a.data()); dSc.copy_from_host(s.data()); dZr.copy_from_host(z.data()); }
  dB.copy_from_host(reinterpret_cast<uint2_t const*>(Bp.data()));

  std::vector<GS> shp(1, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd(1); shpd.copy_from_host(shp.data());
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M, N, 1))};
  std::vector<int> gmh{M}, offs{0};
  std::vector<half_t*> pdh{dD.get()};
  cutlass::DeviceAllocation<half_t*> pd(1); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(1); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(1); gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(1); offdev.copy_from_host(offs.data());
  const size_t wsb = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*64;
  cutlass::DeviceAllocation<char> ws(wsb);

  // FOLD launch: TileShape.K = 64 (A-smem halved vs int2@TK128) + KernelAiuFold<2, Gs-base> schedule.
  // filter_and_run picks the group-size schedule; the fold wrapper is applied inside launch via MOEG_CALL, so here
  // we go through filter_and_run with TK=64 -- which for plain int2 would be ILLEGAL (16B run) and is exactly what
  // the fold makes legal.
  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 64, 32, 32, 3, uint2_t>(
      dA.get(), dB.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<half_t> hD((size_t)M*N); dD.copy_to_host(hD.data());
  int bad = 0, shown = 0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    double got = (double)float(hD[(size_t)m*N + n]);
    double exp = (double)q[(size_t)m*N + n];          // scale=1, zero=0 => D == code
    if (std::abs(got - exp) > 0.25) {
      ++bad;
      if (shown < 16) { std::printf("    m=%d n=%d | got=%.2f exp=%.2f%s\n", m, n, got, exp,
          label_mode == 1 ? "   (got = the n-field the kernel actually read)" :
          label_mode == 2 ? "   (got = the k-field the kernel actually read)" : ""); ++shown; }
    }
  }
  std::printf("  fold int2 TK=64 vs host codes: bad=%d/%d %s\n", bad, M*N, bad == 0 ? "MATCH" : "MISMATCH");

  // PERF (only meaningful once correct): the whole point of the fold is A-smem = TileM*TK*2 at TK=64 instead of 128,
  // i.e. int2 should move from its TK=128 ceiling (37.1% gs=32 / 30.9% gs=16) toward int4@TK64's measured 55.8%/53.1%.
  if (bad == 0 && getenv("FOLD_PERF")) {
    const int PM = atoi(getenv("FOLD_PERF"));            // FOLD_PERF=<M> e.g. 2048
    const int PN = 4096, PK = 4096;
    const int psk = PK / gs;
    cutlass::DeviceAllocation<half_t> pA((size_t)PM*PK), pS((size_t)psk*PN), pZ((size_t)psk*PN), pD((size_t)PM*PN);
    cutlass::DeviceAllocation<uint2_t> pB((size_t)PK*PN);
    { std::vector<half_t> a((size_t)PM*PK, half_t(0.01f)), s((size_t)psk*PN, half_t(0.05f)), z((size_t)psk*PN, half_t(0.f));
      pA.copy_from_host(a.data()); pS.copy_from_host(s.data()); pZ.copy_from_host(z.data()); }
    std::vector<GS> ps(1, cute::make_shape(PM, PN, PK));
    cutlass::DeviceAllocation<GS> psd(1); psd.copy_from_host(ps.data());
    std::vector<DStride> psdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(PM, PN, 1))};
    std::vector<int> pgm{PM}, pof{0}; std::vector<half_t*> ppd{pD.get()};
    cutlass::DeviceAllocation<half_t*> ppD(1); ppD.copy_from_host(ppd.data());
    cutlass::DeviceAllocation<DStride> psD(1); psD.copy_from_host(psdh.data());
    cutlass::DeviceAllocation<int> pgM(1); pgM.copy_from_host(pgm.data());
    cutlass::DeviceAllocation<int> pOf(1); pOf.copy_from_host(pof.data());
    const size_t pwsb = (size_t)cutlass::ceil_div(PM,16)*cutlass::ceil_div(PN,64)*64;
    cutlass::DeviceAllocation<char> pws(pwsb);
    // FOLD_INT4=1: run int4@TK64 instead -- SAME geometry (TM64/TK64, A-smem 8KB, SK=TK/gs, 4 mma K-atoms), only the
    // B format differs, so an acu A/B isolates why int2-fold is more gs-sensitive (42.0 vs 53.1 at gs=16, while at
    // gs=32 it is 52.2 vs 55.8).
    cutlass::DeviceAllocation<cutlass::int4b_t> pB4((size_t)PK*PN);
    const bool use_i4 = getenv("FOLD_INT4") != nullptr;
    auto run = [&]{
      if (use_i4)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 64, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 64, 32, 32, 3, uint2_t>(
            pA.get(), pB.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr); };
    if (getenv("FOLD_ONCE")) {                 // acu: emit exactly ONE kernel launch
      run(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
      std::printf("  [acu] one launch emitted (%s, gs=%d)\n", use_i4 ? "int4@TK64" : "int2-fold@TK64", gs);
      return 0;
    }
    for (int i = 0; i < 3; ++i) run();
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    hggcEvent_t e0, e1; hggcEventCreate(&e0); hggcEventCreate(&e1);
    hggcEventRecord(e0); for (int i = 0; i < 30; ++i) run(); hggcEventRecord(e1);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    float ms = 0; hggcEventElapsedTime(&ms, e0, e1);
    const double us = (double)ms * 1e3 / 30, tf = 2.0*PM*PN*PK / (us*1e-6) / 1e12;
    std::printf("  [fold perf] M=%d N=%d K=%d gs=%d : %8.2f us | %6.1f TFLOP/s (%4.1f%% MFU)\n",
                PM, PN, PK, gs, us, tf, 100.0*tf*1e12/500.0e12);
    std::printf("     compare: int2@TK128 = 37.1%% (gs=32) / 30.9%% (gs=16) ; int4@TK64 = 55.8%% / 53.1%%\n");
  }
  return bad == 0 ? 0 : 1;
}
