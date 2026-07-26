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
//   int4 (64, 64, 64)  F=1  deliv  32  slots  64   OK   55.9%
//   int2 (64, 64, 64)  F=2  deliv  64  slots  64   OK   53.2%
//   int2 (64,128, 64)  F=2  deliv  64  slots 128   OK   bad=0
//   int2 (64, 64,128)  F=1  deliv  64  slots 128   OK   original validated path
//   int1 (32,128,128)  F=2  deliv 128  slots 256   OK   54.3%
//   int1 (64, 64,128)  F=2  deliv 128  slots 128   OK   36.4%
//   int1 (64, 64,256)  F=1  deliv 128  slots 256   OK   original validated path
//   int1 (64, 64, 64)  F=4  deliv 128  slots  64   BAD  over-delivery (I1 catches it)
//   int1 (64,128, 64)  F=4  deliv 128  slots 128   BAD  balanced yet broken (only I2 catches it)
// The last row is why I1 alone is not enough.
#pragma once
#include <cstddef>

namespace fold {

// How many N-groups the dequant converter can address inside one register. The int1/int2 converters place the
// N-half with  base = (32|16)*(v&1) + 2*(v>=2), i.e. a TWO-way split -- so a fold that packs 4 N-columns into one
// run has no way to route groups 2 and 3, which is exactly the int1 F=4 failure. Raising this is a converter change
// (4-way N placement), and it is what would let int1 drop to TK=64 and stop paying double FINE-scale cost at gs=16
// (int1 ScaleOnly 54.3% at gs=32 but only 45.3% at gs=16, while int2/int4 are flat, because SK = TK/gs and int1 is
// pinned at TK=128).
#if defined(MIXGEMM_INT1_NWAY4)
inline constexpr int kConverterNWays = 4;   // 4-way int1 converter enabled (bases {0,16,32,48})
#else
inline constexpr int kConverterNWays = 2;
#endif

template <int Bits, int TM, int TN, int TK, int Stages = 3>
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

  // ---- occupancy ----
  static constexpr int a_smem   = TM * TK * 2;                    // fp16 A tile per stage
  static constexpr int b_smem   = Ng * (F * TK * Bits / 8);
  static constexpr int smem     = (a_smem + b_smem) * Stages;
  static constexpr int warps    = (TM / 32) * (TN / 32);
  static constexpr int blk_smem = 262144 / smem;                  // 256KB per CU (hard cap)
  static constexpr int blk_warp = 64 / warps;                     // 64 warps per CU
  static constexpr int blocks   = blk_smem < blk_warp ? blk_smem : blk_warp;

  // ---- invariants (both needed; each one alone mis-predicts a real reference point) ----
  // I1: OVER-delivery is fatal. Under-delivery is fine -- int4 lives there (32 into 64 slots) and simply issues more
  //     swzl steps. Over-delivery has no such mechanism: the surplus is never fetched into the fragment at all.
  static_assert(delivery <= slots,
      "fold: swzl delivers more codes than the mma fragment has slots; the surplus is silently never read "
      "(this is int1 F=4 at TN=64). Raise TN or TK, or lower F.");
  // I2: the converter can only route kConverterNWays N-groups per register.
  static_assert(F <= kConverterNWays,
      "fold: F exceeds the converter's N-way capacity, so the extra N-groups cannot be routed and read back as "
      "group 0 (this is int1 F=4 at TN=128, which satisfies I1 yet still fails). Raise kConverterNWays only "
      "together with a converter that places that many N-groups.");
  static_assert(TN % F == 0, "fold: TileN must divide into F groups");
  static_assert(TK % 16 == 0 && TN % 32 == 0 && TM % 32 == 0, "fold: tile must be mma-atom aligned");
  static_assert(contig_bytes * F == 32 || F == 1, "fold: the folded run must be exactly 32B");
};

} // namespace fold
