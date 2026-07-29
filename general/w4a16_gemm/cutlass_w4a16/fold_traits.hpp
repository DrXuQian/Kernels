// FoldTraits: ONE place that derives every quantity the N-fold depends on, so the ones that are currently
// computed independently (swzl delivery size, mma fragment slots, converter emission, gmem address layout) can no
// longer disagree silently.
//
// READ THIS FIRST -- A RETRACTION. An earlier version of this header carried a "theorem" that a folded column
// could never be narrower than a half-run, hence TK*Bits >= 128, hence F <= 2, hence int1 could never run at
// TK=64. That was WRONG. The two measurements it rested on are sound (see fold_derivation/), but the inference
// was not: LEG 2 shows the four lanes of a lane/4 group demand the same SET of N, and I read it as the same
// SINGLE N. A thread demands many columns, so nothing forces one column per half-run.
//
// What actually kills int1 at TK=64 is the OFFLINE PACKER, and it is fixable. Per-thread demand vs delivery
// (leg5_perthread.cu, straight out of cute's partition_B):
//     int1 TN=128 TK=64 : thread wants 128 slots = 8 columns x 16 k ; swzl gives 4 words x 32 codes
//                         a column needs 16 k and a word holds 32 codes  =>  TWO columns must share each word
// nfold_regroup_gmem moves whole uint32 words (dst[dst_w] = src[src_w]), so it can only ever put ONE column in a
// word -- it supplies 4 of the 8 columns the thread needs. That is exactly the measured n_used = n % Ng. The fix
// is a bit-granular offline, not a bigger converter and not a larger TK. (int4 at F=1 already does a non-1:1
// word<->column mapping, in the other direction: a word holds 8 codes while a column needs 16 k.)
//
// The one constraint that survives is OVER-DELIVERY: a thread cannot use more codes than its fragment has slots,
// because the surplus is never fetched. That is what rules out int1 at TN=64/TK=64 (128 codes into 64 slots) and
// it is what CheckDelivery<> asserts. It also says a fold at a given TK needs a minimum TN.
//
// BOX REFERENCE POINTS (7 correct + 2 broken), with what each invariant says about them:
//   int4 (64, 64, 64)  F=1  deliv  32  slots  64   OK   55.9%
//   int2 (64, 64, 64)  F=2  deliv  64  slots  64   OK   53.2%
//   int2 (64,128, 64)  F=2  deliv  64  slots 128   OK   bad=0
//   int2 (64, 64,128)  F=1  deliv  64  slots 128   OK   original validated path
//   int1 (32,128,128)  F=2  deliv 128  slots 256   OK   bad=0 (MFU UNVERIFIED, see below)
//   int1 (64, 64,128)  F=2  deliv 128  slots 128   OK   bad=0 (MFU UNVERIFIED, see below)
//   int1 (64, 64,256)  F=1  deliv 128  slots 256   OK   original validated path
//   int1 (64, 64, 64)  F=4  deliv 128  slots  64   BAD  over-delivery -- genuinely impossible
//   int1 (64,128, 64)  F=4  deliv 128  slots 128   BAD  counting is FINE; the offline packer is what fails
// The last row is the one worth having in mind: nothing structural forbids it.
//
// CAUTION ON int1's RECORDED MFU. Every int1 throughput number on file (54.3% at gs=32, 45.3% at gs=16) came from
// a harness whose perf lambda hardcoded (64,128,64) while its correctness check ran the env-selected
// (32,128,128) -- bad=0 and the MFU came from DIFFERENT tiles. The CORRECTNESS results stand; the throughput
// numbers do not. Fixed in test_fold_int2.cu; the labels now carry the launched TileShape.
#pragma once
#include <cstddef>
#include <type_traits>

namespace fold {

// WM/WN default to the common 32x32 warp tile. They only feed the occupancy estimate -- pass the real ones when
// analysing a config that uses something else (the Q3 B-concat sweep runs 16x32 warps at TM=16, which divides by
// zero under a hardcoded /32).
template <int Bits, int TM, int TN, int TK, int Stages = 3, int WM = 32, int WN = 32>
struct FoldTraits {
  // ---- fold geometry ----
  static constexpr int contig_bytes = TK * Bits / 8;              // bytes one N-column contributes along K
  static constexpr int F            = contig_bytes >= 32 ? 1 : 32 / contig_bytes;   // N-columns folded into a 32B run
  static constexpr int Ng           = TN / F;                     // physical rows per tile
  static constexpr int run_codes    = F * TK;                     // codes in one 32B contiguous run
  static constexpr int codes_per_word = 32 / Bits;

  // ---- the two quantities that must agree ----
  static constexpr int delivery = 16 * 8 / Bits;                  // one swzl instruction hands a thread 16 BYTES
  // MEASURED against partition_B on the builder's real TiledMma over 12 configs (fold_derivation/l5_slots.cu).
  // Independent of TN: B is split across the warps in N, so a wider tile is more work, not a bigger fragment.
  static constexpr int slots    = WN * TK / 32;                   // fp16 B-fragment slots per thread

  // ---- occupancy ----
  static constexpr int a_smem   = TM * TK * 2;                    // fp16 A tile per stage
  static constexpr int b_smem   = Ng * (F * TK * Bits / 8);
  static constexpr int smem     = (a_smem + b_smem) * Stages;
  static constexpr int warps    = ((TM + WM - 1) / WM) * ((TN + WN - 1) / WN);
  static constexpr int blk_smem = 262144 / smem;                  // 256KB per CU (hard cap)
  static constexpr int blk_warp = 64 / (warps > 0 ? warps : 1);   // 64 warps per CU
  // WARNING: this omits the REGISTER limit and is therefore not the occupancy the hardware gives you. acu showed
  // shared memory allowing 10 blocks where registers allowed 4. Use fold::warps_per_cu<> below for anything
  // performance-related; this stays only because ft_check prints it alongside the fold geometry.
  static constexpr int blocks   = blk_smem < blk_warp ? blk_smem : blk_warp;

  // ---- invariants ----
  // I1 is the real one: OVER-delivery is fatal, because the surplus is never fetched into the fragment at all.
  // Under-delivery is fine -- int4 lives there (32 codes into 64 slots) and simply issues more swzl steps.
  static_assert(delivery <= slots,
      "fold: swzl delivers more codes than the mma fragment has slots; the surplus is silently never read "
      "(this is int1 F=4 at TN=64). Raise TN or TK, or lower F.");
  // Structural sanity, not fold theory.
  static_assert(TN % F == 0, "fold: TileN must divide into F groups");
  // The mma atom is 16x16x16, so 16 -- not 32 -- is the real granularity. TM=16 is a shipped shape (the Q3
  // B-concat sweep's small-M configs) and must not be rejected.
  static_assert(TK % 16 == 0 && TN % 16 == 0 && TM % 16 == 0, "fold: tile must be mma-atom aligned");
  static_assert(contig_bytes * F == 32 || F == 1, "fold: the folded run must be exactly 32B");
  // DERIVED, not asserted: how many logical columns the offline must fit into ONE 32-bit word. 1 means the
  // simple whole-word packer works; >1 means it must interleave columns at bit granularity. int1 at TK=64
  // needs 2, and that -- not any hardware limit -- is what the current packer cannot do.
  // Also measured on the real TiledMma (same file): one thread demands WN/8 columns x TK/4 k each, and one swzl
  // delivery is 4 words of 32/Bits codes. Both are functions of WN, not TN.
  static constexpr int cols_demanded = WN / 8;
  static constexpr int k_per_col     = TK / 4;
  static constexpr int words_per_dlv = 4;
  // NO cols_per_word HERE, ON PURPOSE. Two derivations of it disagree and neither is trustworthy yet:
  //   * cols_demanded / words_per_dlv = WN/32 -- wrong, it assumes each column lives in exactly one word, but a
  //     column's k is spread across several words while a word also holds several columns;
  //   * the L2 o L3 o L4 composition (l6_derive.cu) says F, and that CONTRADICTS the shipped packer:
  //     nfold_regroup_gmem does dst[dst_w] = src[src_w] with src_w indexed by a single n_log, so the folded buffer
  //     provably has ONE column per word -- and it measures bad=0 on ppu001 at exactly the config the
  //     composition claims needs two.
  // So one leg of that composition is wrong despite each leg having been checked in isolation. Until the
  // calibration in l6_derive.cu's header is done, any cols_per_word constant here would be a guess with a
  // static_assert's authority. See fold_derivation/README.md.
};

// CheckDelivery<> is what a kernel instantiation fires. It asserts the ONE hard constraint: a thread cannot use
// more codes than its mma fragment has slots, because the surplus is never fetched.
//
//     delivery = 16 * 8 / Bits          one swzl hands a thread 16 BYTES
//     slots    = WN * TK / 32           fp16 B-fragment slots per thread
//
// The slots formula is MEASURED against cute's partition_B on the builder's real TiledMma -- WarpOnN = TN/WN and
// PermN = WarpOnN*16 -- over twelve configurations, including all nine ppu001 reference points and three WN
// variants (fold_derivation/l5_slots.cu). Note what it does NOT contain: **TN**. B is partitioned across the
// warps in N, so widening the tile widens the work, not the per-thread fragment. That is why raising TN never
// bought anything, and it is why an earlier version of this file -- which used slots = 8*(TN/32)*(TK/16) -- was
// wrong. That form is right only when TN == 2*WN, and at TN=128/WN=32 it over-estimated slots 2x and would have
// PASSED int1 (64,128,64), a configuration measured broken on the box.
//
// Over-delivery alone separates all nine reference points, 9/9. Written out, the condition is
//     WN * TK * Bits >= 4096
// which at the WM=WN=32 every fold test passes reduces to TK*Bits >= 128 -- int1 >= 128, int2 >= 64, int4 >= 32.
//
// THE ESCAPE, and it is real: slots scales with WN. At WN=64 int1 at TK=64 has slots == delivery == 128 and is
// feasible. But raising WN also raises how many logical columns must share one 32-bit word,
//     cols_per_word = WN / 32
// (also measured, same file) -- so WN=64 needs a packer that interleaves TWO columns inside a word, which
// nfold_regroup_gmem's whole-uint32 moves cannot express. The two constraints pincer int1 at TK=64: WN must rise
// to fix delivery, and raising WN forces the packer upgrade. Both, or neither.
// The predicate itself, exposed so a CALLER can gate on it with `if constexpr` instead of hand-writing an
// equivalent condition. A dispatch ladder that instantiates every (TM,TN,TK) triple regardless of the runtime
// choice will otherwise instantiate illegal ones and trip the static_assert -- which is what happened on the box
// with int1 at TK=64 from CORR_DISPATCH(64,64,64). Gate and guard must be the SAME expression or they drift.
// The warp tiling must divide the block tile, or get_tiled_mma degenerates (WarpOnM = blockM/warpM becomes 0) and
// the collective builder returns `int`, which surfaces as "CollectiveEpilogue (aka int) cannot be used prior to ::"
// deep inside gemm_universal_adapter.h. Pure arithmetic, so it is checkable without instantiating anything -- which
// matters because the local front-end gate CANNOT see this class: the stub environment's own errors abort template
// instantiation before a config is ever reached. TM=16 with WM=32 is the case that got to the box.
template <int TM, int TN, int WM, int WN>
inline constexpr bool warp_shape_ok = (WM > 0 && WN > 0 && TM % WM == 0 && TN % WN == 0
                                       && TM / WM >= 1 && TN / WN >= 1 && (TM / WM) * (TN / WN) <= 16);

// REGISTER PRESSURE per thread. An ESTIMATE, not the compiler's allocation, so it warns rather than blocks -- the
// over-budget configs below do compile, they just run at 4.5% MFU. Derived and then checked against the box:
//     accum = WM*WN/32          fp32, and note it depends ONLY on the warp shape, not on TM or TN
//     A     = WM*TK/64          fp16, A is duplicated across the N warps
//     B     = WN*TK/64          fp16, = slots/2
//     S     = WN*TK/256         fp16, cosize is slots/4 (measured in l21), doubled with zero
// Measured agreement (int1, gs=32, ScaleOnly unless noted):
//     (32,128,128) w32x32  176 regs  blocks  7  -> 42.0%      (32,128,128) w32x32 s2  176  blocks 11 -> 46.5%
//     (32,128, 64) w32x64  176 regs  blocks 23  -> 50.0%      (32,128,256) w32x32 s2  320  blocks  5 -> 23.0%
//     (32,128,256) w32x32 s2 with zero  352 regs             ->  5.8%   (gs=16: 352 -> 4.5%)
//     (32,128,128) w32x64 s3            288 regs             -> 39.0%
// TK=256 was pursued on a "more atoms per iteration hides the scale reload better" theory that the box refuted --
// the register file, not hiding, is what TK controls.
template <int TM, int TN, int TK, int WM, int WN, bool Zero = false>
inline constexpr int regs_per_thread = WM * WN / 32              // accumulator, fp32
                                     + WM * TK / 64              // A fragment
                                     + WN * TK / 64              // B fragment
                                     // Scale fragment: cosize is 32 halves = 16 registers at every shape in the tree
                                     // (l21, l27), i.e. WN*TK/256. It briefly read WN/16 for the stride-0 broadcast
                                     // experiment; that was REVERTED as a no-op, and leaving the formula behind was an
                                     // inconsistency ft_check caught (it asserts 272 for the amort=4 shape and got 260).
                                     + WN * TK / 256 * (Zero ? 2 : 1);
template <int TM, int TN, int TK, int WM, int WN, bool Zero = false>
inline constexpr bool regs_ok = regs_per_thread<TM, TN, TK, WM, WN, Zero> <= 256;


// OCCUPANCY, corrected by acu. My blk formula was min(smem_limit, warp_limit) and it was MISSING THE REGISTER
// LIMIT, which is why blk kept failing to order the data. acu on ladder rungs 3 and 4:
//     rung 3  Regs=112  256 thr/blk  Block Limit Registers ...  theoretical 32 warp/CU  achieved 31.0
//     rung 4  Regs=186  128 thr/blk  Block Limit Registers 4    theoretical 16 warp/CU  achieved 15.2
// and it says so outright: "theoretical occupancy (25.0%) is limited by the number of required registers", with
// shared memory allowing 10 blocks where registers allow 4.
//
// REGISTERS ARE BILLED ROUNDED UP TO A POWER OF TWO. Reverse it from acu's own warp counts: 131072/32 warps =
// 128 reg/thread for a kernel reporting 112, and 131072/16 = 256 for one reporting 186. So 112 is billed as 128
// and 186 as 256 -- 129 registers cost exactly as much as 256. That cliff, not the raw count, is what matters,
// and it is why regs=176 and regs=104 behave like different animals while 176 vs 240 do not.
// ESTIMATE-TO-MEASURED OFFSET. regs_per_thread is consistently ~22 BELOW what acu reports, at both points measured:
//     estimate 176 -> acu 186        estimate 128 -> acu 142     (int1, (32,128,64) w32x64, unchunked / B-chunked)
// so ~10 and ~14; 12 splits them. IT DOES NOT EXTRAPOLATE RELIABLY: the chunked sweep gained +10.2 points at
// (32,128,128) w32x32, which only makes sense if that config crossed into the 128 bucket, yet 120 + 12 = 132 says it
// did not. Two anchor points cannot call a boundary. Treat the model as ordering configs, not as deciding buckets --
// only acu on the specific config does that.
// The gap is address arithmetic, pipeline temporaries and the packed source registers -- things the model does not
// count. It lives here rather than in a comment somewhere because every billing decision compares against 128 or 256,
// so the estimate is useless without it: 120 looks like it clears 128 and does not.
//
// THIRD ANCHOR, AND IT BREAKS THE OFFSET FOR CHUNKED CONFIGS (acu, 2026-07-27, the shipping best config
// (64,128,64) w64x64 s2 gs=32 ScaleOnly int1 with PPU_B_CHUNK=1): acu reports **140 regs**, grid (64,32,1)x(64,1,1),
// 260.90 us on ONE cold launch. regs_per_thread_chunked predicts 224 (accum 128 + A 64 + B 16 + scale 16), so the
// error is **-84**, not +12. Both earlier anchors (176->186, 128->142) were UNCHUNKED. The compiler is evidently not
// keeping accum and the whole A fragment live simultaneously -- 140 is below accum+A = 192 on its own.
// So: regs_measured_offset applies to the UNCHUNKED model only, and even there two anchors cannot call a bucket
// boundary. For chunked configs the model is an UPPER BOUND and an ordering, nothing more. Consequence worth keeping
// in mind: at 140 regs that config's register limit is 131072/(140*32) = 29 warps = 14 blocks, while smem allows 13 --
// so it is SMEM-limited at ~13 blocks / 26 warps per CU, not register-limited at blk=4 as the model claimed.
inline constexpr int regs_measured_offset = 12;
inline constexpr int regs_per_cu = 131072;
template <int R>
inline constexpr int regs_billed = R <= 32 ? 32 : R <= 64 ? 64 : R <= 128 ? 128 : R <= 256 ? 256 : 512;
template <int TM, int TN, int TK, int WM, int WN, bool Zero = false>
inline constexpr int regs_billed_measured =
    regs_billed<regs_per_thread<TM, TN, TK, WM, WN, Zero> + regs_measured_offset>;
template <int TM, int TN, int TK, int WM, int WN, int Stages, int Bits, int Gs = 32, bool Zero = false>
inline constexpr int warps_per_cu = [] {
  constexpr int warps = (TM / WM) * (TN / WN);
  // TOTAL FOR DEGENERATE SHAPES, and that is not defensive dressing. WM > TM or WN > TN makes `warps` zero and
  // the divisions below ill-formed -- and a caller cannot protect this by putting the use inside
  // `if constexpr (legal)`: naming a constexpr VARIABLE TEMPLATE as an argument instantiates it even in a
  // discarded branch. The MoE bench did exactly that and the box build died on
  // `warps_per_cu_chunked<128,32,256,32,64,...>` with "division by zero", for a config moe_ok rejects. Third time
  // in this work that a template named in a discarded statement was instantiated anyway, so the rule is: make the
  // template TOTAL, do not rely on the guard. 0 warps is the honest answer for a shape with no legal warp tiling.
  if constexpr (warps < 1) { return 0; } else {
  constexpr int smem  = (TM * TK * 2 + TN * TK * Bits / 8 + TN * (TK / Gs) * 2 * (Zero ? 2 : 1)) * Stages;
  constexpr int blk_s = 262144 / smem;
  constexpr int blk_w = 64 / warps;
  constexpr int blk_r = regs_per_cu
                      / (regs_billed<regs_per_thread<TM, TN, TK, WM, WN, Zero>> * warps * 32);
  constexpr int blk   = blk_s < blk_w ? (blk_s < blk_r ? blk_s : blk_r) : (blk_w < blk_r ? blk_w : blk_r);
  return blk * warps; }
}();
// CHUNK-AWARE variant. The B fragment holds ONE k-atom instead of MMA_K of them, so its term is 4*MMA_N = WN/4
// instead of WN*TK/64. Added because the sweep printed the UNCHUNKED model in a chunked build -- regs=260/bill=512 next
// to a 63.7% MFU that only makes sense at bill=256 -- which is the third time in this work that a log failed to
// describe the run that produced it.
template <int TM, int TN, int TK, int WM, int WN, bool Zero = false>
inline constexpr int regs_per_thread_chunked = WM * WN / 32          // accumulator, fp32
                                             + WM * TK / 64          // A fragment, still whole
                                             + WN / 4                // B: 4*MMA_N = 4*(WN/16), one k-atom
                                             + WN * TK / 256 * (Zero ? 2 : 1);
template <int TM, int TN, int TK, int WM, int WN, int Stages, int Bits, int Gs = 32, bool Zero = false>
inline constexpr int warps_per_cu_chunked = [] {
  constexpr int warps = (TM / WM) * (TN / WN);
  // TOTAL FOR DEGENERATE SHAPES, and that is not defensive dressing. WM > TM or WN > TN makes `warps` zero and
  // the divisions below ill-formed -- and a caller cannot protect this by putting the use inside
  // `if constexpr (legal)`: naming a constexpr VARIABLE TEMPLATE as an argument instantiates it even in a
  // discarded branch. The MoE bench did exactly that and the box build died on
  // `warps_per_cu_chunked<128,32,256,32,64,...>` with "division by zero", for a config moe_ok rejects. Third time
  // in this work that a template named in a discarded statement was instantiated anyway, so the rule is: make the
  // template TOTAL, do not rely on the guard. 0 warps is the honest answer for a shape with no legal warp tiling.
  if constexpr (warps < 1) { return 0; } else {
  constexpr int smem  = (TM * TK * 2 + TN * TK * Bits / 8 + TN * (TK / Gs) * 2 * (Zero ? 2 : 1)) * Stages;
  constexpr int blk_r = regs_per_cu
      / (regs_billed<regs_per_thread_chunked<TM, TN, TK, WM, WN, Zero> + regs_measured_offset> * warps * 32);
  constexpr int blk_s = 262144 / smem, blk_w = 64 / warps;
  constexpr int blk   = blk_s < blk_w ? (blk_s < blk_r ? blk_s : blk_r) : (blk_w < blk_r ? blk_w : blk_r);
  return blk * warps; }
}();

// THE MODEL, and it took three tries to get here because each earlier attempt varied only ONE of two independent
// factors. Over 25 measured points across all three widths (int1 sweep + the int4/int2/int1 ladders):
//
//                        cvt/mma = 4              cvt/mma = 8
//     warps/CU >= 32     mean 52.1%  n=6          mean 39.1%  n=6     (-13.0)
//     warps/CU <= 16     mean 46.1%  n=16         mean 32.0%  n=2     (-20.1)
//                             (-6.0)
//
// The cvt/mma axis separates with NO overlap in either row -- it is a genuine cliff, ~13 points. The warps/CU axis
// OVERLAPS at cvt/mma=4 (47.8-50.2 appears in both cells), so it is a mean shift of ~6-7 points, not a cliff. The
// two are roughly additive.
//
// WHY THE EARLIER STORIES LOOKED CONTRADICTORY. The int1 sweep's high-warp configs were all WM=16 (cvt/mma=8), so
// raising warps/CU appeared to HURT; the ladder's high-warp configs were WM=32 (cvt/mma=4), so raising warps/CU
// appeared to HELP. Same quantity, opposite apparent sign, because each experiment moved the other factor too.
//
// PRESCRIPTION: cvt/mma = 4 (WM >= 32) AND warps/CU >= 32. The second needs registers billed at <= 128, which at
// TK=64 means WN=32 -- and the delivery bound forbids that for int1. That is int1's ceiling, stated in the two
// quantities that actually govern it.

// CONVERTER AMORTISATION -- the objective, and it is NOT the register count.
//
// A PRESCRIPTION DERIVED FROM regs_per_thread ALONE FAILED ON THE BOX. accum = WM*WN/32 depends only on the warp
// shape and the delivery bound constrains only WN, so w16x64 keeps int1's TK=64 legal while halving the
// accumulator: 128 estimated regs against 176. Measured 39.7% against 50.0%. Fewer registers, comparable
// occupancy, 10.3 points worse.
//
// What separates all 20 measured points, with no overlap, is WM:
//     WM=32, regs<=256 : 40.9 - 50.2%   (10 points)
//     WM=16, regs<=256 : 31.9 - 39.8%   ( 8 points)
// and the best WM=16 point has blk=32, the HIGHEST occupancy in the sweep, while the worst WM=32 point has blk=4,
// the lowest -- so `blocks` orders WITHIN a group but cannot cross between them.
//
// THE MECHANISM (counted from cute in fold_derivation/l26_convert_amort.cu, not argued). In a mixed-input GEMM
// every B fragment element must be converted (lop3 + fma) and scaled before it can enter an mma. Per thread per
// k-tile:
//     mma instructions  = (WM/16)*(WN/16)*(TK/16)
//     B elems to cvt    = 8*(WN/16)*(TK/16)
//     cvt elems per mma = 128/WM                    <- WN and TK cancel EXACTLY; only WM survives
// Each B fragment feeds WM/16 mma instructions down the M direction, so WM/16 is the amortisation factor. WM=16
// means every converted element feeds exactly one mma and the converter cost per unit of math doubles. This is the
// quantised-B analogue of arithmetic intensity, at the register file rather than at HBM -- and it is why a
// prescription that bought registers by cutting WM was exactly backwards.
//
// cvt/mma IS A THRESHOLD, NOT A RATE -- and the register cliff is not at 256. Both corrections come from ONE
// measurement: w64x64, the only int1 shape that reaches cvt/mma=2, run against its cvt/mma=4 peer at EQUAL blk.
//     (64,128,64) w32x64 s2   cvt/mma=4  regs=176  blk=13  ->  48.4%
//     (64,128,64) w64x64 s2   cvt/mma=2  regs=272  blk=13  ->  48.7%
// Halving the converter work AND spending 96 extra registers nets +0.3 points. So:
//
//   (a) cvt/mma=8 is throughput-bound on the B convert path, and cvt/mma<=4 is not. The signature is occupancy
//       sensitivity, measured over the same ~5x blk span at regs<=176, TK<=128:
//           cvt/mma=8 : blk  8 -> 32   MFU 38.4 - 39.8%   spread 1.4 points   FLAT -- more blocks do nothing
//           cvt/mma=4 : blk  4 -> 23   MFU 40.8 - 50.2%   spread 9.4 points   blocks ARE the lever
//       A flat response to 4x the occupancy is what a throughput ceiling looks like. Once past it, cutting cvt/mma
//       further buys nothing, which is why WM=64 gains +0.3 and not another 10 points.
//
//   (b) the register penalty is GRADED and its knee is above 272, not at 256. Against a same-blk 176-reg peer:
//           272 regs -> +0.3 points (nothing)      288 regs -> -2.6      320 regs -> -19.9
//       So 256 was a guess that happened to sit below the real knee. Treat >288 as the danger zone.
//
// THE PRESCRIPTION, and int1's measured optimum is exactly it: get cvt/mma to 4 (WM >= 32), stay under ~288 regs,
// then maximise blocks -> w32x64 s2 at TK=64.
//
// RETRACTED: "int1 is capped at amort=2, and int4 (64,64,32) w64x32 is an untried lever." The cap is real (WM=64
// needs 272 regs at every shape legal under WN*TK >= 4096; asserted in ft_check) but it is HARMLESS, because
// cvt/mma=2 is worth +0.3 points. int4's production config is already at cvt/mma=4, so pushing it to 2 has nothing
// to gain either -- do not spend a box run on it.


template <int WM>
inline constexpr int amort = WM / 16;                 // mma instructions each B fragment element feeds
template <int WM>
inline constexpr int cvt_per_mma = 128 / WM;          // B elements converted per mma instruction issued
// Below this the B convert path stops being the constraint; above it, occupancy cannot compensate.
inline constexpr int cvt_per_mma_target = 4;

template <int Bits, int TN, int TK, int WM, int WN>
inline constexpr bool deliverable = (Bits <= 0 || Bits >= 8) ? true : ((16 * 8 / Bits) <= (WN * TK / 32));

template <int Bits, int TN, int TK, int WM, int WN, class = void>
struct CheckDelivery { static constexpr bool ok = true; };

template <int Bits, int TN, int TK, int WM, int WN>
struct CheckDelivery<Bits, TN, TK, WM, WN, std::enable_if_t<(Bits > 0 && Bits < 8)>> {
  static constexpr int delivery = 16 * 8 / Bits;   // one swzl delivery, in codes
  static constexpr int slots    = WN * TK / 32;    // fp16 B-fragment slots per thread
  static_assert(delivery <= slots,
      "fold: one swzl delivery carries more codes than this thread's mma fragment has slots, and the surplus is "
      "never fetched. Need WN*TK*Bits >= 4096. Raise TileShape.K, or raise the warp N extent WN -- raising "
      "TileShape.N does NOT help, because B is split across the warps in N and slots does not depend on TN. "
      "Note that raising WN past 32 also requires a bit-granular offline packer (cols_per_word = WN/32).");
  static constexpr bool ok = true;
};

} // namespace fold
