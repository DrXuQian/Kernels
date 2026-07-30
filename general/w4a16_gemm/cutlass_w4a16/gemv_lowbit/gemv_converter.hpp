// Packed-integer -> float conversion for the GEMV, uniform in the bit width.
//
// THE ONE FORMULA. Take a 32-bit source word w holding 32/B elements of width B, element j at bits
// [jB, jB+B). For pair index p in [0, 16/B):
//
//     wp = w >> (p*B)
//     h  = (wp & MASK) | MAGIC          MASK = ((1<<B)-1) * 0x00010001,  MAGIC = 0x64006400
//
// h is a half2 holding (2^10 + e_p, 2^10 + e_{p+16/B}). The shift can never contaminate the low field
// because p*B + B <= 16 over that range, and the extracted value's SCALE IS ALWAYS 1 -- there is no
// per-slot 2^-b correction to apply, which is what makes one code path serve int1/int2/int4/int8.
//
// THE 2^10 OFFSET MUST COME OFF HERE, IN THE INTEGER DOMAIN. The tempting alternative -- leave it in and
// fold (-2^10)*s into a modified zero z' = z - 2^10*s, so dequant is a single hfma2 (TODO #18) -- is
// numerically wrong in fp16 and the gate catches it: with s = 0.01 and z = 0.1 you get raw*s = 10.31 against
// z' = -10.14, i.e. two quantities of magnitude 10 cancelling to 0.17 while fp16's spacing at 10.24 is
// 0.0078. Measured max relative error 1.9e-2 to 4.6e-2, and ScaleOnly (z = 0) is worse still because the
// entire result is the cancellation. Subtracting first is exact -- 2^10 + q is an integer in [1024, 2048)
// where fp16's spacing is exactly 1 -- and costs one hsub2 per pair, which is what TRT-LLM's converter also
// pays (h -= 0x64086408). Any signed convention (GPTQ's q-8, Q4_K's dmin) still folds into the caller's zero,
// where the magnitudes are comparable and there is nothing to cancel.
//
// OUTPUT IS IN SLOT ORDER, not logical order: slot 2p+0 is element p and slot 2p+1 is element p + 16/B.
// mapper(logical) -> slot resolves that at compile time inside pack_to_vec2, so it costs nothing. This is
// the same trick as TRT-LLM's LayoutDetails::Mapper -- and unlike the AIU path's offline permutation, it is
// free precisely because a register-resident converter has no fixed hardware emission order to obey.
#pragma once

#include "gemv_math.hpp"

namespace ppu_gemv {

// ---------------------------------------------------------------------------------------------------
// fp16 magic-number path. Exact for any code that fits fp16's 11-bit exact-integer range: 2^10 + q with
// q <= 1023, so every width up to int8 (q <= 255) and every two-plane combination we ship is exact.
template <typename ADetails, int Bits, bool SubOffset>
struct RawConverter {
  using Type  = typename ADetails::Type;
  using Type2 = typename ADetails::Type2;

  static constexpr int kPairsPerWord = 16 / Bits;   // half2 outputs per 32-bit source word
  static constexpr int kElemsPerWord = 32 / Bits;
  static_assert(Bits == 1 || Bits == 2 || Bits == 4 || Bits == 8, "unsupported weight width");

  static constexpr uint32_t kMask  = ((1u << Bits) - 1u) * 0x00010001u;
  static constexpr uint32_t kMagic = ADetails::kIsBF16 ? 0x43004300u : 0x64006400u;
  // 2^10 for fp16, 2^7 for bf16. bf16 has only 8 exact-integer bits, so a code must stay under 128 there.
  static constexpr int kOffset = ADetails::kIsBF16 ? 128 : 1024;
  static_assert(!ADetails::kIsBF16 || Bits <= 4,
                "bf16 has 8 exact-integer bits: int8 codes would not round-trip through the magic path");

  // slot of a logical element, within a run of N elements (N a whole number of source words).
  __host__ __device__ static constexpr int mapper(int logical) {
    int const w = logical / kElemsPerWord;
    int const l = logical % kElemsPerWord;
    return w * kElemsPerWord + (l % kPairsPerWord) * 2 + (l / kPairsPerWord);
  }

  // N logical elements -> N slots of Type.
  template <int N>
  __device__ __forceinline__ static void convert(void const* src, void* dst) {
    static_assert(N % kElemsPerWord == 0, "conversion run must be a whole number of source words");
    uint32_t const* pw = reinterpret_cast<uint32_t const*>(src);
    uint32_t* po = reinterpret_cast<uint32_t*>(dst);   // one uint32_t per half2 slot pair
#pragma unroll
    for (int wi = 0; wi < N / kElemsPerWord; ++wi) {
      uint32_t const w = pw[wi];
#pragma unroll
      for (int p = 0; p < kPairsPerWord; ++p) {
        uint32_t const h = ((w >> (p * Bits)) & kMask) | kMagic;
        po[wi * kPairsPerWord + p] = h;
      }
    }
    if constexpr (SubOffset) {
      // High plane of a two-plane format: the offset must come off BEFORE the shift-and-add combine, or
      // the intermediate (q + 2^10*(1+2^LoBits)) leaves fp16's exact-integer range.
      using M = MathWrapper<ADetails>;
      Type2* p2 = reinterpret_cast<Type2*>(dst);
      Type2 const off = M::to_vec2(M::from_float(float(kOffset)));
#pragma unroll
      for (int i = 0; i < N / 2; ++i) p2[i] = M::sub2(p2[i], off);
    }
  }
};

// ---------------------------------------------------------------------------------------------------
// Portable reference conversion. Per-element extraction, IDENTITY mapper, no magic numbers. Exists so the
// gate has an independent oracle: the fast path above is validated against this (and its mapper is
// discovered by one-hot probing rather than trusted), because a silently permuted converter output is a
// failure mode this project has already paid for twice.
template <typename ADetails, int Bits, bool SubOffset>
struct RefRawConverter {
  using Type  = typename ADetails::Type;
  static constexpr int kElemsPerWord = 32 / Bits;
  static constexpr int kOffset = SubOffset ? 0 : (ADetails::kIsBF16 ? 128 : 1024);

  __host__ __device__ static constexpr int mapper(int logical) { return logical; }

  template <int N>
  __device__ __forceinline__ static void convert(void const* src, void* dst) {
    uint8_t const* pb = reinterpret_cast<uint8_t const*>(src);
    Type* po = reinterpret_cast<Type*>(dst);
#pragma unroll
    for (int j = 0; j < N; ++j) {
      int const bit = j * Bits;
      uint32_t word;
      // Bits divides 8 for every supported width, so an element never straddles a byte.
      word = uint32_t(pb[bit >> 3]) >> (bit & 7);
      uint32_t const q = word & ((1u << Bits) - 1u);
      po[j] = static_cast<Type>(float(int(q) + kOffset));
    }
  }
};

}  // namespace ppu_gemv
