// Plan #20 phase 2, the DECODE: native GGUF scale bytes -> the (scale, zero) an mma atom needs.
//
// NEW FILE. The fp16 scale path is untouched; this is the object a packed-scale collective calls instead of reading a
// pre-multiplied fp16 plane. Nothing here restates a bit position: the field extraction comes from
// gguf_scale_layout.hpp's Layouts (gated on real gguf data in fold_derivation/l91: 4096 superblocks x 8 groups, 0 bad)
// and the code->fp16 conversion comes from the SAME magic-number identity the B converter uses.
//
// WHAT THE KERNEL ACTUALLY NEEDS, per (n, group):   scale = d * sc     zero = -dmin * mn
// with sc/mn the 6-bit codes of the superblock's 12 scale bytes and d/dmin its two fp16 values. Both products are one
// fp16 multiply and both are EXACT-then-rounded-once, because sc, mn <= 63 need 6 bits and d, dmin are already fp16:
// the real product has at most 17 significant bits, so fp32 holds it exactly and fp16 rounds it once. Measured over a
// real tensor's 136192 groups in real_weight/dump_packed_scale.py gate (b): bit-identical to today's host-side form.
//
// WHY THE ZERO IS -dmin*mn AND NOT 8*scale - dmin*mn: the stored 4-bit codes already are the unsigned nib, so a
// converter running with Bias = 0 (plan #20 phase 1 made that immediate a parameter) leaves nib in the mainloop and the
// zero needs no +8*scale term. Gate (c) of the same file measured both: they agree to 0.0003 of a quantisation step, so
// the choice is cost -- one product instead of a product plus an fma -- not precision.
//
// WHY THIS IS THE SHAPE OF THE WIN. The FINE path reloads the scale (and the zero) once per scale group, i.e. 16 s2r
// reads per k-tile at gs=32/TileK=256, and the two channels' reads outnumber their writes 9.1:1. One superblock's 12
// bytes cover EIGHT groups, so one contiguous read plus register decode replaces 8 (x2) strided reads. That is the
// "16 reloads -> 1" in the plan's re-priced section, and it is why the decode is worth ALU it does not save.
#pragma once

#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "gguf_scale_layout.hpp"
#include "cutlass/gguf_packed_scale.h"

namespace gguf_scale {

using cutlass::half_t;

// ---------------------------------------------------------------------------------------------------------------
// CODE -> fp16, ONE rule for every format. With v+128 placed in the mantissa of the fp16 pattern 0x6400 (= 1024.0) the
// value is exactly 1024 + v + 128, so a single subtract of 1152 yields v. This is MixGemmChunkEmit's identity at
// bpos = 0 -- same mechanism, restated only because a 6-bit code living one-per-byte is not one of that struct's
// packings (it assumes codes at bpos = i*Bits inside a vreg).
//
// The +128 offset is what makes it cover SIGNED codes (Q6_K's int8 scales) with no second path: v in [-128, 895] gives
// a mantissa field in [0, 1023] and both 1024+v+128 and v are integers below 2048, hence exact in fp16. l93 checks
// every v in that range against the integer, not against a formula.
CUTLASS_HOST_DEVICE half_t int_to_half_small(int v) {
  uint16_t const bits = uint16_t(0x6400u | uint32_t((v + 128) & 0x3FF));
  return cutlass::half_t::bitcast(bits) - half_t(1152.f);
}

// The code's own centre is a per-format constant (Q3_K's -32; zero for the others) and lives in Traits::kScaleBias,
// read off dump_real_weights.py. Subtracting it BEFORE the conversion keeps one magic constant instead of two.
template <KType T>
CUTLASS_HOST_DEVICE half_t scale_code_to_half(int sc) {
  return int_to_half_small(sc - Traits<T>::kScaleBias);
}

// The two products, named once so a caller cannot form them a second way. `zero` carries the SIGN the mainloop's
// `plus{}` pass expects -- it adds `zero`, and a k-quant's affine term is -dmin*mn.
struct GroupScale {
  half_t scale;
  half_t zero;
};

template <KType T>
CUTLASS_HOST_DEVICE GroupScale make_group_scale(int sc, int mn, half_t d, half_t dmin) {
  GroupScale g;
  g.scale = d * scale_code_to_half<T>(sc);
  g.zero  = -(dmin * int_to_half_small(mn));      // the min code has no centre in any k-quant that has a min
  return g;
}

// Scale only (Q3_K/Q6_K after phase 1, and any symmetric format): no min channel at all, and the code's own centre is
// the converter's Bias, so nothing is left for a zero to carry.
template <KType T>
CUTLASS_HOST_DEVICE half_t make_group_scale_only(int sc, half_t d) {
  return d * scale_code_to_half<T>(sc);
}

// ---------------------------------------------------------------------------------------------------------------
// ONE SUPERBLOCK, ONE COLUMN: the object that replaces eight strided fp16 reads with one contiguous read plus decode.
// `bytes` is the format's scale block (12 B for Q4_K/Q3_K, 16 B for Q2_K/Q6_K) as it sits in memory -- no repacking,
// which is the point of phase 2: smem holds the native bytes.
template <KType T>
struct Superblock {
  static constexpr int kGroups     = Traits<T>::kGroups;
  static constexpr int kBlockBytes = Traits<T>::kBlockBytes;
  static constexpr bool kHasMin    = Traits<T>::kHasMin;

  GroupScale g[kGroups];

  CUTLASS_HOST_DEVICE void decode(uint8_t const* bytes, half_t d, half_t dmin) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kGroups; ++i) {
      int const sc = scale_of<T>(bytes, i);
      if constexpr (kHasMin) {
        g[i] = make_group_scale<T>(sc, min_of<T>(bytes, i), d, dmin);
      } else {
        g[i].scale = make_group_scale_only<T>(sc, d);
        g[i].zero  = half_t(0.f);
      }
    }
  }
};

// The traffic this replaces, as an expression rather than a comment, so a shape change cannot silently invalidate the
// claim: fp16 scale (+ zero) per group per column against the native bytes per superblock per column.
template <KType T>
CUTLASS_HOST_DEVICE constexpr float bytes_per_group_per_col() {
  return float(Traits<T>::kBlockBytes) / float(Traits<T>::kGroups);
}
template <KType T>
CUTLASS_HOST_DEVICE constexpr float bytes_per_group_per_col_fp16() {
  return Traits<T>::kHasMin ? 4.f : 2.f;   // scale + zero, or scale alone
}


// THE PACKED UNIT AND ITS DECODE NOW LIVE IN actlize: cutlass/gguf_packed_scale.h, because the mainloop needs them and
// the mainloop is there. This header re-exports them so existing callers (fold_derivation/l94, the offline harnesses) are
// unchanged, and so there is exactly ONE definition of the bit map -- copying it here is the failure this work keeps
// recording. The Q4_K parameters are named where they are used, not baked into the re-export.
using cutlass::gguf_packed::PackBits;
using cutlass::gguf_packed::kUnitBytes;
static constexpr int kPackBitBase = cutlass::gguf_packed::kBitBase;
CUTLASS_HOST_DEVICE constexpr int packed_bit_of(int g, int which) { return cutlass::gguf_packed::bit_of(g, which); }
CUTLASS_HOST_DEVICE int  packed_code(uint8_t const* u, int g, int which) { return cutlass::gguf_packed::code_of(u, g, which); }
CUTLASS_HOST_DEVICE void packed_put (uint8_t* u, int g, int which, int v) { cutlass::gguf_packed::put_code(u, g, which, v); }
template <KType T>
CUTLASS_HOST_DEVICE GroupScale packed_group(uint8_t const* unit, int g) {
  auto const g2 = cutlass::gguf_packed::group_of<Traits<T>::kScaleBias, Traits<T>::kHasMin>(unit, g);
  return GroupScale{g2.scale, g2.zero};
}

}  // namespace gguf_scale
