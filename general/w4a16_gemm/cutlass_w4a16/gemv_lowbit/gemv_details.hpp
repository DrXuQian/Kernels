// Compile-time description of one GEMV instantiation: A type, weight format, weight layout, and the
// per-thread K step that ties them together.
#pragma once

#include "gemv_wformat.hpp"

namespace ppu_gemv {

// ---------------------------------------------------------------------------------------------------
// Activation type details.
struct FP16DetailsA {
  using Type  = half;
  using Type2 = half2;
  static constexpr int kElemBits = 16;
  static constexpr bool kIsBF16 = false;
};

#if defined(ENABLE_BF16)
struct BF16DetailsA {
  using Type  = __nv_bfloat16;
  using Type2 = __nv_bfloat162;
  static constexpr int kElemBits = 16;
  static constexpr bool kIsBF16 = true;
};
#endif

// ---------------------------------------------------------------------------------------------------
// Vector access type for a per-thread byte count. Widest whole-power-of-two vector that divides it.
//
// The static_assert here is the reason kHiBits == 0 must NOT be fed in as a byte count of zero: naming
// PackedAccess<0> in a using-declaration instantiates it even when nothing reads it, so the guard has to
// be on the ARGUMENT, not around the use. (Same shape as the FoldTraits-above-the-gate failure: a
// constexpr entity named as a template argument is instantiated regardless of any if-constexpr around it.)
template <int Bytes>
struct PackedAccess {
  static_assert(Bytes >= 4, "per-thread access must be at least one 4-byte word -- raise StepK");
  static_assert(Bytes % 4 == 0, "per-thread access must be a whole number of 4-byte words");
  using Vec = std::conditional_t<Bytes % 16 == 0, uint4,
              std::conditional_t<Bytes % 8 == 0, uint2, uint32_t>>;
  static constexpr int kVecBytes = static_cast<int>(sizeof(Vec));
  static constexpr int kCount    = Bytes / kVecBytes;
  static constexpr int kBytes    = Bytes;
};

// ---------------------------------------------------------------------------------------------------
// One instantiation.
template <typename ADetails_, WFormat Fmt, WLayout Lay, int StepK_, int Threads_, int TileSizeK_ = 256>
struct KernelDetails {
  using ADetails = ADetails_;
  using AType    = typename ADetails::Type;
  using AType2   = typename ADetails::Type2;

  static constexpr WFormat kFormat   = Fmt;
  static constexpr WLayout kLayout   = Lay;
  static constexpr int kLoBits       = lo_bits_of(Fmt);
  static constexpr int kHiBits       = hi_bits_of(Fmt);
  static constexpr int kTotalBits    = kLoBits + kHiBits;
  static constexpr bool kTwoPlane    = is_two_plane(Fmt);
  static constexpr int kMinBits      = kTwoPlane ? (kLoBits < kHiBits ? kLoBits : kHiBits) : kLoBits;

  static constexpr int kStepK        = StepK_;
  static constexpr int kThreads      = Threads_;
  static constexpr int kTileSizeK    = TileSizeK_;
  static constexpr int kCtaK         = kStepK * kThreads;
  static constexpr int kWarpSize     = 32;
  static constexpr int kWarpNum      = kThreads / kWarpSize;

  // Every plane must deliver at least a 4-byte word per thread per iteration. For the sparsest plane this
  // is what forces StepK up: int1 needs StepK >= 32, int2 needs >= 16.
  static_assert(kStepK * kMinBits >= 32,
                "StepK too small for the sparsest plane: StepK*min(bits) must be >= 32 bits");
  static_assert(kThreads % kWarpSize == 0, "Threads must be a multiple of the warp size");
  // A conversion run must be a whole number of 32-bit source words, per plane.
  static_assert(kStepK % (32 / kLoBits) == 0, "StepK must be a whole number of low-plane source words");
  static_assert(!kTwoPlane || kStepK % (32 / (kHiBits ? kHiBits : 1)) == 0,
                "StepK must be a whole number of high-plane source words");

  using LoAccess = PackedAccess<kStepK * kLoBits / 8>;
  // GUARD ON THE ARGUMENT, not around the use -- see PackedAccess.
  using HiAccess = PackedAccess<(kHiBits ? kStepK * kHiBits / 8 : 4)>;
  using AAccess  = PackedAccess<kStepK * int(sizeof(AType))>;

  // Defined as cute Layouts in gemv_wformat.hpp; the byte triple is derived from them.
  using LoLayout = WFormatTraits<Lay, kLoBits, kStepK, kThreads, kTileSizeK>;
  using HiLayout = WFormatTraits<Lay, (kHiBits ? kHiBits : 1), kStepK, kThreads, kTileSizeK>;

  static constexpr int kLoStepBytes = kStepK * kLoBits / 8;
  static constexpr int kHiStepBytes = kHiBits ? kStepK * kHiBits / 8 : 0;

  static const char* format_name() { return name_of(kFormat); }
};

}  // namespace ppu_gemv
