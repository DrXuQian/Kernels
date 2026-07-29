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
// (f) #13: THE LAST CONSUMER OF THE LEGACY PACKERS, now migrated. It was the last one because it picks its shape at
// RUNTIME (env vars) while xplane::place_derived needs the tile as template arguments -- so the migration is a
// runtime-to-compile-time dispatch, which is FOLD_CONFIGS below. legacy_pipeline.hpp is no longer included: the
// packers survive only as the independent reference the fold_derivation gates compare against (l58/l61/l64), which is
// the one place they must survive, or "the derived walk equals the derived walk" becomes a tautology.
#include "xplane_offline.hpp"
#include "fold_traits.hpp"
#include "moe_grouped_ppu.cuh"

// -------------------------------------------------------------------------------------------------------------------
// THE ONE TABLE. Every (bits, tile, warp) this harness supports, written ONCE; the offline placement, the banner and
// the kernel launch are all generated from these rows.
//
// WHY A TABLE AND NOT THREE LADDERS. That is what was here before, and the three did not agree. The banner printed the
// raw FOLD_TM while the launch ladder collapsed everything that was not 32 down to 64, so `FOLD_TM=128` measured
// (64,...) under a banner that said 128; and the offline ladder was a third copy, duplicated verbatim between the main
// path and FOLD_DECODE's run_once. The file's own comments warn twice about "the same cross-config mismatch that
// invalidated the int1 MFU numbers" -- three independent derivations of one decision is how that keeps happening.
//
// MEMBERSHIP IS THE DELIVERY BOUND. A row exists iff `slots = WN*TK/32 >= delivery = 128/bits`. That REPLACES the
// `if constexpr` guard that used to sit inside CORR_DISPATCH: an over-delivering shape is not a shape the table has,
// so asking for one prints the table instead of silently dropping half the converted data (the 44.9%-random result).
// int1 at TK=64 therefore appears only at WN=64 -- which is precisely why it used to need its own packer.
#define FOLD_CONFIGS(EMIT)                  /* bits  TM   TN   TK  WM  WN */                                       \
  EMIT(2,  32, 128, 128, 32, 32)  /* slots 128 >= 64  */  EMIT(2,  32, 128,  64, 32, 32)  /* slots  64 >= 64 */    \
  EMIT(2,  64, 128, 128, 32, 32)  /* slots 128 >= 64  */  EMIT(2,  64, 128,  64, 32, 32)  /* slots  64 >= 64 */    \
  EMIT(2,  64,  64, 128, 32, 32)  /* slots 128 >= 64  */  EMIT(2,  64,  64,  64, 32, 32)  /* slots  64 >= 64 */    \
  EMIT(1,  32, 128, 128, 32, 32)  /* slots 128 >= 128 */  EMIT(1,  64, 128, 128, 32, 32)  /* slots 128 >= 128 */   \
  EMIT(1,  64,  64, 128, 32, 32)  /* slots 128 >= 128 */  EMIT(1,  64, 128,  64, 32, 64)  /* F=4, needs WN=64 */
#define FOLD_ELEM_1 uint1_t
#define FOLD_ELEM_2 uint2_t
#define FOLD_DBUF_1 dB1
#define FOLD_DBUF_2 dB

// If this fails, the actlize SUBMODULE on the box is stale: the Kernels gitlink moved but `git submodule update` did
// not run, so the build would silently produce a binary identical to the previous one. That ambiguity is what made an
// "every acu counter is identical" A/B uninterpretable -- so it is a compile error, not a runtime surprise.
#if !defined(PPU_SCALE_FRAGMENT_API) || PPU_SCALE_FRAGMENT_API < 3
#error "stale actlize submodule: run `git submodule update --init third_party/actlize` and rebuild"
#endif

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
  // FOLD_BITPACK used to mean "switch the offline to the bit-granular packer", because the whole-word regroup could not
  // express int1 at TK=64 (a 32-bit word must carry TWO logical columns there). place_derived is bit-granular ALWAYS,
  // so the flag no longer selects an offline -- it only selects that shape, which the table now reaches on its own.
  // Kept, and kept implying int1, so every existing invocation still names the same configuration.
  // It is read BEFORE the shape, because it CHOOSES the shape: reading it after was how the banner and the launch
  // ended up disagreeing.
  const bool bitpack_ = getenv("FOLD_BITPACK") != nullptr;
  const int fbits = getenv("FOLD_BITS") ? atoi(getenv("FOLD_BITS")) : (bitpack_ ? 1 : 2);   // 1 = int1, 2 = int2
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
  const int ftm = getenv("FOLD_TM") ? atoi(getenv("FOLD_TM")) : (bitpack_ ? 64 : fbits == 1 ? 32 : 64);
  const int ftk = getenv("FOLD_TK") ? atoi(getenv("FOLD_TK")) : (bitpack_ ? 64 : fbits == 1 ? 128 : 64);
  const int ftn = getenv("FOLD_TN") ? atoi(getenv("FOLD_TN")) : (bitpack_ ? 128 : fbits == 1 ? 128 : 64);
  // Resolve the requested shape against the table ONCE, and take the warp tile and FoldF FROM the row rather than
  // recomputing them here. FoldF's old inline form was (32*8/bits)/TK, which silently returns 0 for any unfolded shape.
  int fwm = 0, fwn = 0, ff = 0;
#define FOLD_MATCH(BITS, TMV, TNV, TKV, WMV, WNV)                                                                  \
  if (fbits == (BITS) && ftm == (TMV) && ftn == (TNV) && ftk == (TKV))                                             \
    { fwm = (WMV); fwn = (WNV); ff = fold::FoldTraits<BITS, TMV, TNV, TKV, 3, WMV, WNV>::F; }
  FOLD_CONFIGS(FOLD_MATCH)
#undef FOLD_MATCH
  if (!ff) {
    std::printf("  int%d (%d,%d,%d) is not a supported configuration. The table is:\n", fbits, ftm, ftn, ftk);
#define FOLD_SHOW(BITS, TMV, TNV, TKV, WMV, WNV)                                                                   \
    std::printf("    FOLD_BITS=%d FOLD_TM=%-3d FOLD_TN=%-3d FOLD_TK=%-3d  w%dx%-2d F=%d  slots=%d delivery=%d\n",  \
                BITS, TMV, TNV, TKV, WMV, WNV, fold::FoldTraits<BITS, TMV, TNV, TKV, 3, WMV, WNV>::F,              \
                fold::FoldTraits<BITS, TMV, TNV, TKV, 3, WMV, WNV>::slots,                                         \
                fold::FoldTraits<BITS, TMV, TNV, TKV, 3, WMV, WNV>::delivery);
    FOLD_CONFIGS(FOLD_SHOW)
#undef FOLD_SHOW
    return 2;
  }
  // The banner prints the RESOLVED row, so a run can no longer look like it measured one tile while measuring another.
  // Declared up here, not next to the buffers: the BANNER prints it and the golden further down uses it, and the
  // banner comes first. (The local syntax gate caught this after the first placement compiled only for the golden.)
  const bool svary = getenv("FOLD_SVARY") != nullptr;
  // PERIOD 13, NOT 8, AND THAT IS THE WHOLE POINT. My first pattern was 1 + (1/16)*((n+g)&7) -- period 8 in n --
  // while the displacements a broken broadcast can actually produce are 8 (the val mode's n-bit) and 32 (MMA_N).
  // Both are multiples of 8, so that pattern gave the SAME scale after a misassignment and the probe was blind to
  // exactly the errors it existed to catch. It reported bad=0 on the box, which proved nothing.
  // 13 is coprime to 8/16/32, so the minimum |delta| is 0.0625 at offset 8, 0.125 at 16 and 0.25 at 32 -- all above
  // the 0.02 tolerance. Values are 1.0 .. 1.75 in steps of 1/16, exact in fp16. g carries a coefficient too, so a
  // scale-GROUP misassignment shows up as well.
  auto sc_at = [](int g, int n) { return 1.f + 0.0625f * (float)((5 * n + 3 * g) % 13); };
  // slots = WN*TK/32, MEASURED against partition_B on the builder's real TiledMma (fold_derivation/l5_slots.cu).
  // The old form here was 8*(TN/32)*(TK/16) = TN*TK/64, which is only right when TN == 2*WN.
  std::printf("[fold] M=K=%d N=%d gs=%d bits=%d  TileShape=(%d,%d,%d) warp=%dx%d FoldF=%d | slots=%d delivery=%d%s%s\n",
              M, N, gs, fbits, ftm, ftn, ftk, fwm, fwn, ff,
              fwn * ftk / 32, 16 * 8 / fbits, bitpack_ ? "  [BITPACK]" : "",
              // FOLD_SVARY leaves no trace otherwise, and a log that cannot show which inputs were used is how a
              // round of int1 MFU numbers got invalidated. Say it either way, so "blind" is visible too.
              svary ? "  [SVARY]" : "  [scale=1.0 EVERYWHERE -- BLIND to scale plumbing]");

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
  std::vector<int8_t> Bp(packed.size());   // same byte count as `packed` (a placement only permutes bits)
  // THE OFFLINE, from the SAME table row the kernel launches. xplane::place_derived walks the destination as a cute
  // layout at BIT granularity, so one call covers both destination walks -- the folded one and the interleave-256 one
  // -- and both bit widths. That is what retires nfold_regroup_gmem (whole-uint32 moves, hence one logical column per
  // word, hence correct only at WN=32) and its bit-granular escape hatch nfold_place_bits_int1_tk64.
  //
  // Byte-identity with the pair it replaces is gated locally, on the configurations being migrated rather than on two
  // neighbours of them: fold_derivation/l64_migrate_gate.cu diffs every (bits, TM, TN, TK) row of this table.
  //
  // NFOLD_STD=1 still means what it always meant -- the standard relayout with NO fold, i.e. deliberately wrong data
  // for a folded shape. It is a negative control: it gave "n=0..31 correct at k=0, everything else wrong", and a
  // harness that cannot produce a known-wrong buffer cannot show that its bad-count means anything.
  const QuantTypeClass qtc = (fbits == 1) ? QuantTypeClass::PACKED_INT1_WEIGHT_ONLY
                                          : QuantTypeClass::PACKED_INT2_WEIGHT_ONLY;
  // place_derived reads RAW CODES in k-major (`q[k*N + n]`) and does its own packing, so it takes `q`, not `packed`.
#define FOLD_PACK(BITS, TMV, TNV, TKV, WMV, WNV)                                                                   \
  if (fbits == (BITS) && ftm == (TMV) && ftn == (TNV) && ftk == (TKV))                                             \
    xplane::place_derived<BITS, TMV, TNV, TKV, WMV, WNV,                                                           \
                          fold::FoldTraits<BITS, TMV, TNV, TKV, 3, WMV, WNV>::F>(dst, src, N, K);
  auto pack_B = [&](std::vector<uint8_t> const& src, std::vector<int8_t>& into) {
    int8_t* dst = into.data();
    FOLD_CONFIGS(FOLD_PACK)
    (void)dst;
  };
  if (getenv("NFOLD_STD")) {
    preprocess_weights_for_mixed_gemm<false, 256, 0>(Bp.data(), packed.data(), {(size_t)K, (size_t)N}, qtc);
  } else {
    pack_B(q, Bp);
  }

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dB((size_t)K*N);
  cutlass::DeviceAllocation<uint1_t> dB1((size_t)K*N);
  { std::vector<half_t> a((size_t)M*K, half_t(0.f));
    for (int m = 0; m < M; ++m) a[(size_t)m*K + m] = half_t(1.f);
    // FOLD_SVARY=1 -- WITHOUT THIS, A SCALE-PLUMBING CHANGE IS UNTESTED. The default scale is 1.0 EVERYWHERE, so
    // every scale register holds the same value and it does not matter which one a given mma slot reads. That makes
    // this harness blind to exactly the thing the stride-0 broadcast fragment changes (which register each slot
    // reads), and it reported bad=0 for that change without exercising it at all.
    //
    // With A=identity and 1-bit codes, D[m][n] is either 0 or scale(g,n) -- the output IS the scale the slot used, so
    // a misassignment cannot hide. Values are exact in fp16 (steps of 1/16) so the tolerance can be tight: a
    // one-position error is 0.0625 against a 0.02 tolerance. Only the SCALE is varied, not the zero, because the
    // correctness dispatch hardcodes its quant mode -- and tCrS and tCrZ come from the SAME make_scale_fragment
    // call, so validating the scale validates the zero's layout too.
    std::vector<half_t> s((size_t)scale_k*N, half_t(1.f)), z((size_t)scale_k*N, half_t(0.f));
    if (svary)
      for (int g = 0; g < scale_k; ++g)
        for (int n = 0; n < N; ++n) s[(size_t)g*N + n] = half_t(sc_at(g, n));
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
// THE LAUNCH, from the same rows as the offline. This used to be two macros -- CORR_DISPATCH for the 32x32 ladder plus
// BITPACK_DISPATCH for the one WN=64 shape -- behind a nested runtime ladder that was ALSO duplicated inside
// FOLD_DECODE. Both the warp shape and the element type now come from the row, so `fold::deliverable`'s `if constexpr`
// guard is gone: it existed because the ladder instantiated every (TM,TN,TK) for BOTH bit widths regardless of the
// runtime request, and the table only contains rows that satisfy the bound. It also instantiates 10 kernels where the
// two macros instantiated 12.
#define CORR_DISPATCH()                                                                                            \
  do {                                                                                                             \
    FOLD_CONFIGS(FOLD_RUN)                                                                                          \
  } while (0)
#define FOLD_RUN(BITS, TMV, TNV, TKV, WMV, WNV)                                                                    \
  if (fbits == (BITS) && ftm == (TMV) && ftn == (TNV) && ftk == (TKV))                                             \
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, TMV, TNV, TKV, WMV, WNV, 3, FOLD_ELEM_##BITS>(       \
        dA.get(), FOLD_DBUF_##BITS.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                      \
        M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CORR_DISPATCH();
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
  // FOLD_DECODE used to REFUSE the WN=64 int1 shape, because its own copies of the offline and the ladder were the
  // 32x32 ones -- it would have printed a plausible (n,k) map from a different configuration. It now shares pack_B and
  // CORR_DISPATCH with the main path, so every row of the table decodes, including that one.
  if (getenv("FOLD_DECODE")) {
    auto run_once = [&](std::vector<uint8_t> const& qq, std::vector<half_t>& out) {
      std::vector<int8_t> bp((size_t)K*N/epb, 0);
      if (getenv("NFOLD_STD")) {
        std::vector<int> t2((size_t)K*N);
        for (int k2 = 0; k2 < K; ++k2) for (int n2 = 0; n2 < N; ++n2) t2[(size_t)n2*K+k2] = qq[(size_t)k2*N+n2];
        std::vector<int8_t> pk((size_t)K*N/epb, 0);
        for (size_t i = 0; i < pk.size(); ++i) { int8_t bb=0;
          for (int t = 0; t < epb; ++t) bb |= int8_t((t2[epb*i+t] & cmask) << (fbits*t)); pk[i]=bb; }
        preprocess_weights_for_mixed_gemm<false, 256, 0>(bp.data(), pk.data(), {(size_t)K,(size_t)N}, qtc);
      } else {
        pack_B(qq, bp);
      }
      if (fbits == 1) dB1.copy_from_host(reinterpret_cast<uint1_t const*>(bp.data()));
      else            dB.copy_from_host(reinterpret_cast<uint2_t const*>(bp.data()));
      CORR_DISPATCH();
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
    // A is the identity and M==K, so the k of output (m,n) is m and its scale group is m/gs.
    double exp = (double)q[(size_t)m*N + n] * (svary ? (double)sc_at(m / gs, n) : 1.0);
    if (std::abs(got - exp) > (svary ? 0.02 : 0.25)) {
      ++bad;
      if (shown < 16) { std::printf("    m=%d n=%d | got=%.2f exp=%.2f%s\n", m, n, got, exp,
          label_mode == 1 ? "   (got = the n-field the kernel actually read)" :
          label_mode == 2 ? "   (got = the k-field the kernel actually read)" : ""); ++shown; }
    }
  }
  // Say WHICH width and tile actually ran. The literal "int2 TK=64" here printed for every configuration, including
  // the int1 BITPACK run at (64,128,64) -- the same class of "the log does not name what ran" problem that
  // invalidated a round of MFU numbers, just in the correctness line instead of the perf line.
  std::printf("  fold int%d (%d,%d,%d) w%dx%d F=%d vs host %s: bad=%d/%d %s\n",
              fbits, ftm, ftn, ftk, fwm, fwn, ff, svary ? "codes x scale(g,n)" : "codes",
              bad, M*N, bad == 0 ? "MATCH" : "MISMATCH");

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
    // One row -> both quant modes. Two modes x 10 rows is 20 instantiations against the old ladder's 10, which buys the
    // guarantee that the shape in the [fold perf] line is the shape in the [fold] banner and the shape the offline
    // packed. The old ladder was cheaper and measured the wrong tile.
#define PERF_PB_1 pB1
#define PERF_PB_2 pB
#define PERF_RUN(BITS, TMV, TNV, TKV, WMV, WNV)                                                                    \
      else if (!use_i4 && fbits == (BITS) && ftm == (TMV) && ftn == (TNV) && ftk == (TKV) && scale_only)            \
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, TMV, TNV, TKV, WMV, WNV, 3, FOLD_ELEM_##BITS>(    \
            pA.get(), PERF_PB_##BITS.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),                     \
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);                          \
      else if (!use_i4 && fbits == (BITS) && ftm == (TMV) && ftn == (TNV) && ftk == (TKV))                          \
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, TMV, TNV, TKV, WMV, WNV, 3, FOLD_ELEM_##BITS>(    \
            pA.get(), PERF_PB_##BITS.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),                    \
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
    // The label was a FOURTH place the shape was decided, by nested ternaries that named tiles the launch no longer
    // picks. Build it once, from the resolved row.
    char shape_lbl[48];
    if (use_i4) std::snprintf(shape_lbl, sizeof shape_lbl, "int4 (64,64,%d)", ftk == 128 ? 128 : 64);
    else        std::snprintf(shape_lbl, sizeof shape_lbl, "int%d (%d,%d,%d) w%dx%d", fbits, ftm, ftn, ftk, fwm, fwn);
    auto run = [&]{
      if (use_i4 && ftk == 64 && scale_only)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 64, 64, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (use_i4 && ftk == 64)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 64, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      // (the int2 @TK=128 "same-TK control" that used to sit here is table row (2,*,*,128): it hardcoded TN=64, so at
      //  FOLD_TN=128 it measured a narrower tile than the correctness launch, and the table reaches it correctly.)
      else if (use_i4 && ftk == 128 && scale_only)                  // int4 @TK=128 -- same-TK control
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 64, 128, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), nullptr, ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      else if (use_i4 && ftk == 128)
        moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 128, 32, 32, 3, cutlass::int4b_t>(
            pA.get(), pB4.get(), pS.get(), pZ.get(), ppD.get(), psD.get(), pgM.get(),
            PM, PN, PK, 1, gs, psd.get(), ps.data(), pOf.get(), pws.get(), pwsb, nullptr);
      // ...and the low-bit shapes come from THE TABLE, like the offline and the correctness launch. This was a THIRD
      // copy of the shape decision and it agreed with neither: at fbits==1, ftm!=32 and no FOLD_BITPACK it ran
      // (64,64,128) whatever FOLD_TN said, so FOLD_TN=128 printed an MFU for a tile the bad=0 never covered -- the
      // cross-config mismatch this block's own comment was written to prevent, still present one ladder over. int4
      // stays hand-written below/above because int4 does not fold and therefore has no row.
      FOLD_CONFIGS(PERF_RUN) };
    if (getenv("FOLD_ONCE")) {                 // acu: emit exactly ONE kernel launch
      run(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
      std::printf("  [acu] one launch: %s %s gs=%d\n", shape_lbl, scale_only ? "ScaleOnly" : "ScaleZero", gs);
      return 0;
    }
    for (int i = 0; i < 3; ++i) run();
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    hggcEvent_t e0, e1; hggcEventCreate(&e0); hggcEventCreate(&e1);
    hggcEventRecord(e0); for (int i = 0; i < 30; ++i) run(); hggcEventRecord(e1);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    float ms = 0; hggcEventElapsedTime(&ms, e0, e1);
    const double us = (double)ms * 1e3 / 30, tf = 2.0*PM*PN*PK / (us*1e-6) / 1e12;
    std::printf("  [fold perf] %-22s %-9s M=%d N=%d K=%d gs=%d : %8.2f us | %6.1f TFLOP/s (%4.1f%% MFU)\n",
                shape_lbl, scale_only ? "ScaleOnly" : "ScaleZero",
                PM, PN, PK, gs, us, tf, 100.0*tf*1e12/500.0e12);
    std::printf("     compare: int2@TK128 = 37.1%% (gs=32) / 30.9%% (gs=16) ; int4@TK64 = 55.8%% / 53.1%%\n");
  }
  return bad == 0 ? 0 : 1;
}
