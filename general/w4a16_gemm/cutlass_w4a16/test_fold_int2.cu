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
using uint1_t = cutlass::uint1b_t;
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
  const int fbits = getenv("FOLD_BITS") ? atoi(getenv("FOLD_BITS")) : 2;   // 1 = int1, 2 = int2
  // FOLD_TK: the fold's TileShape.K. NOT free -- a swzl delivery is a fixed 16 B/thread, so it carries
  // (16*8/bits) codes, while the mma fragment has 8*(TN/32)*(TK/16) slots. They must be EQUAL or the converter
  // produces data with nowhere to go. int2 balances at TK=64 (64 codes = 64 slots); int1 packs twice as densely and
  // balances at TK=128 (128 = 128). At int1/TK=64 it would be 128 codes into 64 slots -- every shape assert still
  // passes (B MMA_N=2 matches accum, MMA_K=4 matches A), so it compiles and silently drops half the data, which is
  // exactly the 44.9%-random result measured for int1 fold at TK=64.
  // Binding constraint: fragment slots == swzl delivery, where
  //     slots    = 8*(TN/32)*(TK/16)          (fp16 B fragment per thread)
  //     delivery = 16 bytes/thread = 16*8/bits codes
  // TN is a FREE VARIABLE too -- fixating on TK was my error, and it forced int1 to TK=128 (A-smem 16KB, 5 blk).
  //     int1 TN=64  TK=64  : slots  64 < 128 -> OVERFLOW, half the converted data dropped (the 44.9% result)
  //     int1 TN=128 TK=64  : slots 128 = 128 -> BALANCED, A-smem back to 8KB like int4/int2-fold, and since int1
  //                          weights are the smallest the total smem is smaller than int4's: 9 blk vs 8.
  // (int4@TK64 has slots 64 > delivery 32, i.e. it UNDER-delivers and issues more swzl steps -- workable. Only
  //  OVER-delivery is fatal, which is exactly what broke int1.)
  // Reaching 50%+ needs A-smem = TM*TK*2 down to 8KB. TK is not the only lever -- TM is one too, and int1's only
  // VERIFIED-correct fold is F=2 (F=4 is broken: the decode showed n_used = n%Ng and k_used = (m/8)*8+m%4, i.e. the
  // fragment only ever covers 64 of the run's 256 codes, which offline placement cannot fix). So instead of forcing
  // F=4, keep F=2 and halve TM:
  //     int1 TM=32 TN=128 TK=128 F=2 -> A 8KB, 8 blk, 4 warps/blk, 50% warp occupancy, slots 256 > deliv 128
  //   which matches int4 (8 blk, 4 warps/blk, 50%, under-delivery) column for column, with a smaller B tile.
  const int ftm = getenv("FOLD_TM") ? atoi(getenv("FOLD_TM")) : (fbits == 1 ? 32 : 64);
  const int ftk = getenv("FOLD_TK") ? atoi(getenv("FOLD_TK")) : (fbits == 1 ? 128 : 64);
  const int ftn = getenv("FOLD_TN") ? atoi(getenv("FOLD_TN")) : (fbits == 1 ? 128 : 64);
  // FOLD_BITPACK dispatches at (64,128,64) with a 32x64 warp tile, so the banner must say so -- printing the
  // env-derived shape instead is how a run can look like it measured one tile while measuring another.
  const bool bitpack_ = getenv("FOLD_BITPACK") != nullptr;
  const int dtm = bitpack_ ? 64 : ftm, dtn = bitpack_ ? 128 : ftn, dtk = bitpack_ ? 64 : ftk;
  const int dwn = bitpack_ ? 64 : 32;
  // slots = WN*TK/32, MEASURED against partition_B on the builder's real TiledMma (fold_derivation/l5_slots.cu).
  // The old form here was 8*(TN/32)*(TK/16) = TN*TK/64, which is only right when TN == 2*WN.
  std::printf("[fold] M=K=%d N=%d gs=%d bits=%d  TileShape=(%d,%d,%d) warp=%dx%d FoldF=%d | slots=%d delivery=%d%s\n",
              M, N, gs, fbits, dtm, dtn, dtk, 32, dwn, (32 * 8 / fbits) / dtk,
              dwn * dtk / 32, 16 * 8 / fbits, bitpack_ ? "  [BITPACK]" : "");

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
  // FOLD_BITS=1 -> int1 (F=4 at TK=64), default int2 (F=2). The fold derivation is bit-width agnostic; only the
  // packing density (codes/byte) and the element type change. int1 is the bigger prize: it is stuck at 2 blk/12%
  // occupancy at its forced TK=256, and folding to TK=64 should give 10 blk/62% (even past int4's 8 blk, since its
  // weights are smaller). It is also the prerequisite for Q3/Q5 reaching the ceiling.
  const int epb   = 8 / fbits;                      // codes per byte
  const int cmask = (1 << fbits) - 1;
  for (auto& v : q) v = (uint8_t)(v & cmask);       // clamp codes to the format
  for (auto& v : qT) v &= cmask;
  std::vector<int8_t> packed((size_t)K * N / epb, 0);
  for (size_t i = 0; i < packed.size(); ++i) {
    int8_t b = 0;
    for (int t = 0; t < epb; ++t) b |= int8_t((qT[epb * i + t] & cmask) << (fbits * t));
    packed[i] = b;
  }
  std::vector<int8_t> Bp(packed.size());   // same byte count as `packed` (regroup only permutes whole words)
  // The N-fold runs after interleave-256, as a whole-uint32 regroup (nfold_regroup_gmem).
  // DERIVED placement (nfold_place_derived_int2) INSTEAD of the standard relayout: it maps each logical (n,k) of a
  // 64x64 tile straight to the physical (word, crumb) the folded kernel reads. Verified locally: 4096 logical
  // positions -> 4096 distinct physical positions, no collisions/misses. Set NFOLD_STD=1 to fall back to the standard
  // relayout for an A/B (that baseline gave n=0..31 correct at k=0, everything else wrong).
  // Standard preprocess FIRST (all 5 steps), then the fold is a WORD-LEVEL regroup on its output that preserves the
  // pipeline's own crumb order -- see nfold_regroup_words_int2. NFOLD_STD=1 skips the regroup (baseline A/B).
  const QuantTypeClass qtc = (fbits == 1) ? QuantTypeClass::PACKED_INT1_WEIGHT_ONLY
                                          : QuantTypeClass::PACKED_INT2_WEIGHT_ONLY;
  // FOLD_BITPACK=1: int1 at TK=64 with WN=64. The whole-word regroup cannot express this placement (the fragment
  // wants TWO logical columns inside each 32-bit word), so skip the five relayout steps and write the folded
  // buffer directly from the DERIVED map -- fold_derivation/l10_placement.cu, which regresses to 0/16384 against
  // the shipped offline on the TK=128 config before generating this one.
  const bool bitpack = bitpack_;
  if (bitpack) {
    if (fbits != 1 || ftk != 64 || ftn != 128) {
      std::printf("  FOLD_BITPACK is derived for int1 TN=128 TK=64 only (got int%d TN=%d TK=%d)\n", fbits, ftn, ftk);
      return 2;
    }
    std::vector<int8_t> nk((size_t)K * N / 8, 0);      // row-major (n,k), one code per bit
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) {
      const size_t i = (size_t)n * K + k;
      if (qT[(size_t)n * K + k] & 1) nk[i / 8] |= int8_t(1 << (i % 8));
    }
    nfold_place_bits_int1_tk64(Bp.data(), nk.data(), N, K, ftn, ftk);
  } else {
    preprocess_weights_for_mixed_gemm<false, 256, 0>(Bp.data(), packed.data(), {(size_t)K, (size_t)N}, qtc);
    if (!getenv("NFOLD_STD")) {
      std::vector<int8_t> tmp(Bp.size());
      nfold_regroup_gmem(tmp.data(), Bp.data(), {(size_t)K, (size_t)N}, ftn, ftk, fbits);
      Bp.swap(tmp);
    }
  }

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dB((size_t)K*N);
  cutlass::DeviceAllocation<uint1_t> dB1((size_t)K*N);
  { std::vector<half_t> a((size_t)M*K, half_t(0.f));
    for (int m = 0; m < M; ++m) a[(size_t)m*K + m] = half_t(1.f);
    std::vector<half_t> s((size_t)scale_k*N, half_t(1.f)), z((size_t)scale_k*N, half_t(0.f));
    dA.copy_from_host(a.data()); dSc.copy_from_host(s.data()); dZr.copy_from_host(z.data()); }
  if (fbits == 1) dB1.copy_from_host(reinterpret_cast<uint1_t const*>(Bp.data()));
  else            dB.copy_from_host(reinterpret_cast<uint2_t const*>(Bp.data()));

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
// The int1 branch is compile-time gated. At WN=32 the fragment has slots = WN*TK/32 = TK, while one swzl delivery
// carries 128 int1 codes, so int1 below TK=128 OVER-DELIVERS and the surplus is never fetched -- that is the
// configuration ppu001 measured as garbage. The ladder below instantiates every (TM,TN,TK) combination regardless
// of the runtime ftk, so without `if constexpr` it instantiates int1 at TK=64 and trips fold::CheckDelivery at
// compile time. int1 at TK=64 is reachable, but only at WN=64 -- that is what BITPACK_DISPATCH is for.
#define CORR_DISPATCH(TMV, TNV, TKV)                                                                          \
  do {                                                                                                        \
    if (fbits == 1) {                                                                                         \
      if constexpr (fold::deliverable<1, TNV, TKV, 32, 32>)                                                   \
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, TMV, TNV, TKV, 32, 32, 3, uint1_t>(         \
            dA.get(), dB1.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                          \
            M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);                     \
      else {                                                                                                  \
        std::printf("  int1 at TK=%d over-delivers at WN=32 (slots=%d < 128 codes). Use FOLD_BITPACK=1 with"  \
                    " FOLD_TN=128 FOLD_TK=64, which dispatches at WN=64.\n", TKV, 32 * TKV / 32);              \
        return 2;                                                                                             \
      }                                                                                                       \
    } else                                                                                                    \
      moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, TMV, TNV, TKV, 32, 32, 3, uint2_t>(           \
          dA.get(), dB.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                             \
          M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);                      \
  } while (0)
// Same as CORR_DISPATCH but with the warp N extent at 64 -- int1 only, since that is the case that needs it.
// slots = WN*TK/32, so int1 at TK=64 needs WN >= 64; the 32x32 ladder below can never satisfy it.
#define BITPACK_DISPATCH(TMV, TNV, TKV)                                                                       \
  do {                                                                                                        \
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, TMV, TNV, TKV, 32, 64, 3, uint1_t>(             \
        dA.get(), dB1.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                              \
        M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);                        \
  } while (0)
  if (bitpack) BITPACK_DISPATCH(64, 128, 64);
  else if (ftm == 32) { if (ftk == 128) CORR_DISPATCH(32, 128, 128); else CORR_DISPATCH(32, 128, 64); }
  else if (ftn == 128) { if (ftk == 128) CORR_DISPATCH(64, 128, 128); else CORR_DISPATCH(64, 128, 64); }
  else                 { if (ftk == 128) CORR_DISPATCH(64,  64, 128); else CORR_DISPATCH(64,  64,  64); }
  if (false)
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 64, 32, 32, 3, uint2_t>(
        dA.get(), dB.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
        M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  // FOLD_DECODE=1: recover WHICH logical (n,k) each output slot actually consumed, instead of theorising about it.
  // A 1-bit code can carry one label bit, so the index is decoded bit-plane by bit-plane: run the kernel once per bit
  // with q = ((n or k) >> b) & mask, read the bit back out of D (scale=1, zero=0, A=identity => D == the code), and
  // OR it into place. This is the technique that cracked int2; three successive mechanistic theories for int1
  // (degenerate Ng, slots-vs-delivery, pairing) were all wrong, so measure instead.
  if (getenv("FOLD_DECODE") && bitpack) {
    // The decode path re-runs the preprocess+word-regroup offline and dispatches on the 32x32 ladder. Under
    // FOLD_BITPACK both of those are wrong, and it would print a plausible-looking (n,k) map from a different
    // configuration -- exactly the silent cross-config mismatch that invalidated the int1 MFU numbers.
    std::printf("  FOLD_DECODE is not wired for FOLD_BITPACK (different offline AND different warp shape)\n");
    return 2;   // `bad` is computed further down; 2 is the same "unsupported combination" code the shape check uses
  }
  if (getenv("FOLD_DECODE")) {
    auto run_once = [&](std::vector<uint8_t> const& qq, std::vector<half_t>& out) {
      std::vector<int> t2((size_t)K*N);
      for (int k2 = 0; k2 < K; ++k2) for (int n2 = 0; n2 < N; ++n2) t2[(size_t)n2*K+k2] = qq[(size_t)k2*N+n2];
      std::vector<int8_t> pk((size_t)K*N/epb, 0);
      for (size_t i = 0; i < pk.size(); ++i) { int8_t bb=0;
        for (int t = 0; t < epb; ++t) bb |= int8_t((t2[epb*i+t] & cmask) << (fbits*t)); pk[i]=bb; }
      std::vector<int8_t> bp(pk.size());
      preprocess_weights_for_mixed_gemm<false, 256, 0>(bp.data(), pk.data(), {(size_t)K,(size_t)N}, qtc);
      if (!getenv("NFOLD_STD")) { std::vector<int8_t> tm(bp.size());
        nfold_regroup_gmem(tm.data(), bp.data(), {(size_t)K,(size_t)N}, ftn, ftk, fbits); bp.swap(tm); }
      if (fbits == 1) dB1.copy_from_host(reinterpret_cast<uint1_t const*>(bp.data()));
      else            dB.copy_from_host(reinterpret_cast<uint2_t const*>(bp.data()));
      if (ftm == 32) { if (ftk == 128) CORR_DISPATCH(32, 128, 128); else CORR_DISPATCH(32, 128, 64); }
  else if (ftn == 128) { if (ftk == 128) CORR_DISPATCH(64, 128, 128); else CORR_DISPATCH(64, 128, 64); }
  else                 { if (ftk == 128) CORR_DISPATCH(64,  64, 128); else CORR_DISPATCH(64,  64,  64); }
      CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
      out.resize((size_t)M*N); dD.copy_to_host(out.data());
    };
    for (int axis = 0; axis < 2; ++axis) {                    // 0 = decode n, 1 = decode k
      std::vector<int> idx((size_t)M*N, 0);
      const int nbits_idx = (axis == 0) ? 8 : 8;              // N,K <= 256
      for (int b = 0; b < nbits_idx; ++b) {
        std::vector<uint8_t> qq((size_t)K*N);
        for (int k2 = 0; k2 < K; ++k2) for (int n2 = 0; n2 < N; ++n2)
          qq[(size_t)k2*N+n2] = (uint8_t)((((axis == 0) ? n2 : k2) >> b) & cmask);
        std::vector<half_t> dd; run_once(qq, dd);
        for (size_t i = 0; i < (size_t)M*N; ++i)
          idx[i] |= (((int)std::lround((double)float(dd[i]))) & 1) << b;
      }
      int bad2 = 0, shown2 = 0;
      std::printf("  [decode %s_used]\n", axis == 0 ? "n" : "k");
      // FULL map for one slice, matched or not -- the first-10-mismatches view hides the shape of the permutation.
      if (axis == 0) {                       // n_used across a whole row (m=0)
        std::printf("    m=0 n_used[n=0..%d]:", (N < 64 ? N : 64) - 1);
        for (int n2 = 0; n2 < N && n2 < 64; ++n2) std::printf(" %d", idx[(size_t)0*N + n2]);
        std::printf("\n");
      } else {                               // k_used down a column (n=0), want == m
        std::printf("    n=0 k_used[m=0..%d]:", (M < 64 ? M : 64) - 1);
        for (int m2 = 0; m2 < M && m2 < 64; ++m2) std::printf(" %d", idx[(size_t)m2*N + 0]);
        std::printf("\n");
      }
      for (int m2 = 0; m2 < M && shown2 < 10; ++m2) for (int n2 = 0; n2 < N && shown2 < 10; ++n2) {
        const int want = (axis == 0) ? n2 : m2;               // identity A: output row m reads k=m
        const int got  = idx[(size_t)m2*N+n2];
        if (got != want) { std::printf("    (m=%3d,n=%3d) %s_used=%3d want=%3d delta=%+4d xor=%3d\n",
                                       m2, n2, axis == 0 ? "n" : "k", got, want, got-want, got^want); ++shown2; }
      }
      for (int m2 = 0; m2 < M; ++m2) for (int n2 = 0; n2 < N; ++n2)
        if (idx[(size_t)m2*N+n2] != ((axis == 0) ? n2 : m2)) ++bad2;
      std::printf("    -> bad=%d/%d %s\n", bad2, M*N, bad2 ? "PERMUTED" : "identity");
    }
    return 0;
  }
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
    cutlass::DeviceAllocation<uint1_t> pB1((size_t)PK*PN);
    const bool use_i4 = getenv("FOLD_INT4") != nullptr;
    // FOLD_SCALEONLY=1 -> FinegrainedScaleOnly (zeros=nullptr). CRITICAL for fair comparison: the recorded int4
    // "ceiling" numbers (55.8% gs=32 / 53.1% gs=16) came from the bench's I4 macro, which uses ScaleOnly, while the
    // fold test used ScaleZero. In the FINE path every mma atom reloads the scale, and WITH zero it reloads twice --
    // so the zero cost grows as gs shrinks (gs=16: 4 reloads -> 8). That, not the weight format, is what the earlier
    // "int2-fold 42.0 vs int4 53.1" gap was measuring.
    const bool scale_only = getenv("FOLD_SCALEONLY") != nullptr;
    auto run = [&]{
      if (use_i4 && ftk == 64 && scale_only)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 64, 64, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (use_i4 && ftk == 64)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 64, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (!use_i4 && fbits == 2 && ftk == 128 && scale_only)   // int2 @TK=128 (unfolded) -- same-TK control
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 64, 128, 32, 32, 3, uint2_t>(
            pA.get(), pB.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (!use_i4 && fbits == 2 && ftk == 128)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 128, 32, 32, 3, uint2_t>(
            pA.get(), pB.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (use_i4 && ftk == 128 && scale_only)                  // int4 @TK=128 -- same-TK control
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 64, 128, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (use_i4 && ftk == 128)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 128, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      // int1: TK is pinned at 128 (TK*Bits >= 128, see fold_derivation/). ftm picks between the two legal
      // tiles, exactly as the correctness ladder does -- these branches used to hardcode 64,128,64, which is
      // the F=4 shape the decode proved broken, so the printed MFU came from a DIFFERENT tile than the bad=0.
      // FOLD_BITPACK must be dispatched at WN=64 here too. Without this branch the correctness check runs the
      // bitpack config and the perf number comes from (32,128,128) at WN=32 -- the same cross-config mismatch that
      // invalidated the int1 numbers in the first place, reintroduced in the very fix for it.
      else if (bitpack && scale_only)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 128, 64, 32, 64, 3, uint1_t>(
            pA.get(), pB1.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (bitpack)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 128, 64, 32, 64, 3, uint1_t>(
            pA.get(), pB1.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (fbits == 1 && ftm == 32 && scale_only)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 3, uint1_t>(
            pA.get(), pB1.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (fbits == 1 && ftm == 32)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 32, 128, 128, 32, 32, 3, uint1_t>(
            pA.get(), pB1.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (fbits == 1 && scale_only)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 64, 128, 32, 32, 3, uint1_t>(
            pA.get(), pB1.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (fbits == 1)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 128, 32, 32, 3, uint1_t>(
            pA.get(), pB1.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (scale_only)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 64, 64, 32, 32, 3, uint2_t>(
            pA.get(), pB.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 64, 32, 32, 3, uint2_t>(
            pA.get(), pB.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr); };
    if (getenv("FOLD_ONCE")) {                 // acu: emit exactly ONE kernel launch
      run(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
      std::printf("  [acu] one launch: %s %s gs=%d\n",
                  use_i4 ? (ftk == 128 ? "int4 (64,64,128)" : "int4 (64,64,64)")
                       : bitpack ? "int1 (64,128,64) w32x64"
                       : (fbits == 1 ? (ftm == 32 ? "int1 (32,128,128)" : "int1 (64,64,128)")
                                     : (ftk == 128 ? "int2 (64,64,128)" : "int2 (64,64,64)")),
                  scale_only ? "ScaleOnly" : "ScaleZero", gs);
      return 0;
    }
    for (int i = 0; i < 3; ++i) run();
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    hggcEvent_t e0, e1; hggcEventCreate(&e0); hggcEventCreate(&e1);
    hggcEventRecord(e0); for (int i = 0; i < 30; ++i) run(); hggcEventRecord(e1);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    float ms = 0; hggcEventElapsedTime(&ms, e0, e1);
    const double us = (double)ms * 1e3 / 30, tf = 2.0*PM*PN*PK / (us*1e-6) / 1e12;
    std::printf("  [fold perf] %-17s %-9s M=%d N=%d K=%d gs=%d : %8.2f us | %6.1f TFLOP/s (%4.1f%% MFU)\n",
                use_i4 ? (ftk == 128 ? "int4 (64,64,128)" : "int4 (64,64,64)")
                       : bitpack ? "int1 (64,128,64) w32x64"
                       : (fbits == 1 ? (ftm == 32 ? "int1 (32,128,128)" : "int1 (64,64,128)")
                                     : (ftk == 128 ? "int2 (64,64,128)" : "int2 (64,64,64)")),
                scale_only ? "ScaleOnly" : "ScaleZero",
                PM, PN, PK, gs, us, tf, 100.0*tf*1e12/500.0e12);
    std::printf("     compare: int2@TK128 = 37.1%% (gs=32) / 30.9%% (gs=16) ; int4@TK64 = 55.8%% / 53.1%%\n");
  }
  return bad == 0 ? 0 : 1;
}
