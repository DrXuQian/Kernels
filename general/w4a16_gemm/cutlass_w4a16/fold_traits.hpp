// FoldTraits: ONE place that derives every quantity the N-fold depends on, so the four that are currently computed
// independently (swzl delivery size, mma fragment slots, converter CPY_VEC, gmem address layout) can no longer
// disagree silently.
//
// WHY THIS EXISTS. Every fold failure in this project was one of three kinds:
//   1. the same quantity computed in several places, disagreeing -> SILENT wrong data
//      (int1 F=4 compiled cleanly and dropped 3/4 of the weights; measured 41% random, decoded as
//       n_used = n % Ng and k_used = (m/8)*8 + m%4, i.e. the fragment covered 64 of the run's 256 codes);
//   2. an edit landing on a dead branch (load_init_B has two, and a %256-aligned shape takes the interleaved one,
//      which builds shape/stride from N,K,kCon and never reads mainloop_params.dB) -> four "fixes", zero movement;
//   3. hand-composing a multi-stage index chain (source order, coord_h, plane-major, tile period) -> one error each.
// This header attacks (1) directly and gives (2)/(3) a single source of truth to derive from.
//
// VALIDATED AGAINST EVERY BOX REFERENCE POINT (7 correct + 2 known-broken); the invariants below are exactly the
// predicate that separates them:
//   int4 (64, 64, 64)  F=1  deliv  32  slots  64  colw 8   OK   55.9%
//   int2 (64, 64, 64)  F=2  deliv  64  slots  64  colw 4   OK   53.2%
//   int2 (64,128, 64)  F=2  deliv  64  slots 128  colw 4   OK   bad=0
//   int2 (64, 64,128)  F=1  deliv  64  slots 128  colw 8   OK   original validated path
//   int1 (32,128,128)  F=2  deliv 128  slots 256  colw 4   OK   bad=0 (MFU UNVERIFIED, see below)
//   int1 (64, 64,128)  F=2  deliv 128  slots 128  colw 4   OK   bad=0 (MFU UNVERIFIED, see below)
//   int1 (64, 64,256)  F=1  deliv 128  slots 256  colw 8   OK   original validated path
//   int1 (64, 64, 64)  F=4  deliv 128  slots  64  colw 2   BAD  over-delivery (I1 catches it too)
//   int1 (64,128, 64)  F=4  deliv 128  slots 128  colw 2   BAD  balanced yet broken (only I2 catches it)
//   int4 (64, 64,128)  F=1  deliv  32  slots 128  colw16   OK   untested, must stay ALLOWED (colw > run)
// Row 9 is why I1 alone is not enough; row 10 is why I2 must be a one-sided bound rather than an equality.
// I2 separates all nine measured points on its own, but I1 stays because it is independent: it catches TN too
// small at a legal colw (e.g. int1 TN=32 TK=128 delivers 128 codes into 64 slots).
#pragma once
#include <cstddef>
#include <type_traits>

namespace fold {

// WHY F IS CAPPED AT 2 -- derived, not probe-fitted, and verifiable locally with no box.
// (scratchpad/leg1_runword.cpp, leg2_frag.cu, leg3_predicate.cpp)
//
// LEG 1, from ppu_tsm_ld_swzl_sim's SWAP=true branch (the B operand's instantiation, and the one calibrated
// against hardware by the w2a16 cube-width bug). Strip the two swizzle terms -- they cancel against the AIU
// write's matching .swzl -- and the LOGICAL word a thread reads is
//         row      = (v/2)*8 + lane/4
//         run-word = (v%2)*4 + lane%4
// so a row's 32B run is addressed as a 2x4 GRID: v%2 picks a 4-word HALF, lane%4 picks the word inside it.
// Nothing else reaches a run-word (checked for all 128 (lane,v) pairs).
//
// LEG 2, from cute's partition_B on the real TiledMMA: the fragment's N is a function of lane/4 and the value
// bits ONLY. lane%4 moves K, at stride 2, and never touches N.
//
// THEOREM. The four lanes of a lane/4 group all demand the SAME N, and they receive the four different words of
// a half. Every word of a half must therefore carry the same logical column, so a folded column can never be
// narrower than a half-run (4 words = 16 B), and a 32B run holds at most two columns:
//         a folded column is at least a half-run   <=>   TK * Bits >= 128   (and then F <= 2 follows)
// A column WIDER than a run is unaffected -- it just spans several 32B slices, which is the ordinary unfolded
// case (int4 at TK=128 is 64 B/column and perfectly legal). The bound is one-sided.
// Checked against all nine box reference points: 0 mismatches (leg3_predicate.cpp).
//
// COROLLARY, and the reason this constant is not simply "raise it later": F=4 is NOT blocked by the converter.
// The converter's bases {0,32,2,34} look like a 2-way N split, and the obvious move is to write a 4-way variant
// with bases {0,16,32,48}. That was tried and reverted, because it cannot work in principle -- with F=4 the
// third and fourth columns of a run are delivered to DIFFERENT LANES (f = 2*(v%2) + (lane%4)/2), and a converter
// only relabels registers inside one thread. No converter reaches another lane's registers.
//
// CONSEQUENCE. int1's smallest legal TK is 128, so at gs=16 it carries SK=8 while int2/int4 carry SK=4.
//
// CAUTION ON int1's RECORDED MFU. Every int1 throughput number on file (54.3% at gs=32, 45.3% at gs=16) was
// produced by a harness whose perf lambda hardcoded (64,128,64) -- the F=4 shape this header forbids -- while
// its correctness check ran the env-selected (32,128,128). bad=0 and the MFU therefore came from DIFFERENT
// tiles. The CORRECTNESS results stand; the throughput numbers do not, and int1 must be re-measured before any
// conclusion is drawn from them. Fixed in test_fold_int2.cu; the labels now carry the launched TileShape.
inline constexpr int kMaxFold = 2;

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
  static constexpr int slots    = 8 * (TN / 32) * (TK / 16);      // fp16 B-fragment slots per thread
  static constexpr int col_words = TK * Bits / 32;                // uint32 words one folded column occupies

  // ---- occupancy ----
  static constexpr int a_smem   = TM * TK * 2;                    // fp16 A tile per stage
  static constexpr int b_smem   = Ng * (F * TK * Bits / 8);
  static constexpr int smem     = (a_smem + b_smem) * Stages;
  static constexpr int warps    = ((TM + WM - 1) / WM) * ((TN + WN - 1) / WN);
  static constexpr int blk_smem = 262144 / smem;                  // 256KB per CU (hard cap)
  static constexpr int blk_warp = 64 / (warps > 0 ? warps : 1);   // 64 warps per CU
  static constexpr int blocks   = blk_smem < blk_warp ? blk_smem : blk_warp;

  // ---- invariants (both needed; each one alone mis-predicts a real reference point) ----
  // I1: OVER-delivery is fatal. Under-delivery is fine -- int4 lives there (32 into 64 slots) and simply issues more
  //     swzl steps. Over-delivery has no such mechanism: the surplus is never fetched into the fragment at all.
  static_assert(delivery <= slots,
      "fold: swzl delivers more codes than the mma fragment has slots; the surplus is silently never read "
      "(this is int1 F=4 at TN=64). Raise TN or TK, or lower F.");
  // I2: a folded column must be at least a whole 4-word half of the run. Otherwise the column a lane receives
  //     varies with lane%4, and LEG 2 says the fragment's N never does -- so the surplus columns land on the
  //     wrong lanes. This is int1 F=4 at TN=128, which satisfies I1 yet still fails.
  //     Not fixable in the converter (see kMaxFold above); the only lever is TK.
  //     Only a LOWER bound: a column wider than a run (int4 at TK=128 -> 64 B) just spans several slices, which
  //     is the ordinary unfolded case and always fine. What is forbidden is a column NARROWER than a half-run.
  static_assert(TK * Bits >= 128,
      "fold: a folded column must be at least 16 B (TK*Bits >= 128) so that the four words a half-run hands to "
      "the four lanes all belong to ONE column. Below that, the column varies with lane%4 and the mma fragment "
      "has no such N. int1 therefore cannot go below TK=128, int2 below TK=64, int4 below TK=32.");
  static_assert(F <= kMaxFold, "fold: F > 2 is unreachable -- see the derivation above");
  static_assert(TN % F == 0, "fold: TileN must divide into F groups");
  // The mma atom is 16x16x16, so 16 -- not 32 -- is the real granularity. TM=16 is a shipped shape (the Q3
  // B-concat sweep's small-M configs) and must not be rejected.
  static_assert(TK % 16 == 0 && TN % 16 == 0 && TM % 16 == 0, "fold: tile must be mma-atom aligned");
  static_assert(contig_bytes * F == 32 || F == 1, "fold: the folded run must be exactly 32B");
};

// Check<> is what a kernel instantiation fires. It deliberately asserts ONLY the theorem -- the TK*Bits bound --
// and NOT the rest of FoldTraits, for a reason worth stating:
//
//   * TK*Bits >= 128 depends on nothing but the B width and TileShape.K. No warp layout, no TN, no epilogue. It is
//     the one predicate I can verify by inspection against every instantiation in the tree (sweep_shapes.cpp does
//     exactly that), so putting it in the build cannot break a config that works today.
//   * I1 (delivery <= slots) is real but its `slots` term assumes the 32x32 warp tile. The Q3 B-concat sweep runs
//     16x32 warps, so a kernel-side I1 would be asserting a number I do not model faithfully. It stays in
//     FoldTraits for offline analysis, where the caller can pass the true WM/WN.
//
// Bits outside (0,8) means "not a sub-byte B plane" (fp16 / int8 / an absent second plane) and is skipped.
// Worth knowing, from I1 rather than the theorem: folding (F=2) additionally needs TN >= 64. At TN=32 a thread is
// handed more codes than its fragment has slots.
template <int Bits, int TK, class = void>
struct Check { static constexpr bool ok = true; };

template <int Bits, int TK>
struct Check<Bits, TK, std::enable_if_t<(Bits > 0 && Bits < 8)>> {
  static_assert(TK * Bits >= 128,
      "fold: a folded column must be at least 16 B (TK*Bits >= 128) so that the four words a half-run hands to "
      "the four lanes all belong to ONE column. Below that the column varies with lane%4, and the mma fragment "
      "has no such N -- the surplus columns are delivered to other lanes entirely. int1 cannot go below TK=128, "
      "int2 below TK=64, int4 below TK=32. See fold_derivation/ for the two-leg proof.");
  static constexpr bool ok = true;
};

} // namespace fold
