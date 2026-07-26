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
  // Straight from cute's partition_B (leg5_perthread.cu, four shapes predicted and measured): one thread demands
  //     TN/16 columns  x  TK/4 k each,  and receives  TN*TK*Bits/2048  words of 32/Bits codes.
  // Dividing gives how many logical columns must share ONE 32-bit word:
  static constexpr int cols_demanded  = TN / 16;
  static constexpr int k_per_col      = TK / 4;
  static constexpr int words_per_thr  = TN * TK * Bits / 2048;
  static constexpr int cols_per_word_num = 128;               // cols_per_word = 128 / (TK*Bits), as a ratio
  static constexpr int cols_per_word_den = TK * Bits;
  // <=1 means the whole-word packer in nfold_regroup_gmem suffices (it moves uint32s, so one column per word).
  // >1 means several columns must be INTERLEAVED inside a word, which needs a bit-granular offline. int1 at
  // TK=64 needs 2 -- and asserting this away as "impossible" is the mistake this header used to make.
  static constexpr bool wholeword_packer_ok = cols_per_word_num <= cols_per_word_den;
};

// CheckDelivery<> is what a kernel instantiation fires. It asserts ONE thing: the swzl must not hand a thread
// more codes than its mma fragment has slots. Over-delivery has no recovery -- the surplus is simply never
// fetched. Under-delivery is fine and common (int4 lives there and just issues more swzl steps).
//
// `slots = 8*(TN/32)*(TK/16)` was verified against cute's partition_B, but only for the DEFAULT 32x32 warp tile
// (leg5_perthread.cu: 256/64/64/128 predicted and measured for the four shapes tried). The Q3 B-concat sweep
// runs 16x32 warps at TM=16, where the TiledMMA -- and therefore the slot count -- is a different shape. So the
// guard skips anything that is not 32x32 rather than assert a number it does not model.
//
// WHAT IS DELIBERATELY *NOT* ASSERTED: an earlier version added `TK*Bits >= 128`, on the theory that a folded
// column could not be narrower than a half-run. That was wrong; see the retraction in fold_derivation/README.md.
// Bits outside (0,8) means "not a sub-byte B plane" (fp16 / int8 / an absent second plane) and is skipped.
template <int Bits, int TN, int TK, int WM, int WN, class = void>
struct CheckDelivery { static constexpr bool ok = true; };

template <int Bits, int TN, int TK, int WM, int WN>
struct CheckDelivery<Bits, TN, TK, WM, WN, std::enable_if_t<(Bits > 0 && Bits < 8 && WM == 32 && WN == 32)>> {
  static constexpr int delivery = 16 * 8 / Bits;              // one swzl hands a thread 16 BYTES
  static constexpr int slots    = 8 * (TN / 32) * (TK / 16);  // fp16 B-fragment slots per thread
  static_assert(delivery <= slots,
      "fold: one swzl delivery carries more codes than this thread's mma fragment has slots, and the surplus is "
      "never fetched. Raise TileShape.N (a fold at this TK needs TN >= 128 for int1, >= 64 for int2), or raise "
      "TileShape.K.");
  static constexpr bool ok = true;
};

} // namespace fold
