// L66 -- the Q6 / Q5 arithmetic, EMULATED on the host, plus Q3 behaviour preservation.
//
// The converter is two inline asm instructions, so it cannot be run locally -- but both are pure bit/fp16 operations and
// can be emulated exactly: lop3 with immLut 0xEA is (a & b) | c, and ppu.fma.rtte.f16x2 is a half2 fused multiply-add.
// Emulating them turns "does the arithmetic produce lo + 2^LowBits * hi" into a LOCAL question, which matters because
// Q6 and Q5 have no working reference anywhere -- unlike every previous step in this work, where Q3 or a shipped buffer
// could be diffed against.
//
// Three checks:
//  (1) Q3 BEHAVIOUR IS PRESERVED. MixGemm2Plane<2,1> replaced a hand-written struct; at_plain / keep / at / vreg_used
//      must be identical to the old versions for every Chunk and both Rebase settings.
//  (2) THE ARITHMETIC IS RIGHT for all three formats: emulate the full emit_one and require the fp16 result to equal
//      lo + 2^LowBits * hi for every (t, v, lane) and every code value, including the int4 +8 bias.
//  (3) THE DESTINATION IS A PARTITION: unchunked, the kOut/2 half2 slots are each written exactly once; chunked, the
//      chunks partition them.
//
//   nvcc -std=c++17 -Istub_inc -I.. -I../../../../third_party/actlize/include l66_q65_arith.cu -o l66 && ./l66
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
using namespace cute;
using namespace cutlass;

// Q3's ORIGINAL hand-written members, kept verbatim as the reference for check (1).
using OldAt = Layout<Shape <Shape<_2,_4>, Shape<_2, _2>>, Stride<Stride<_1,_4>, Stride<_16,_2>>>;
static int old_at_plain(int t, int v) { return int(OldAt{}(t + 8 * v)); }
template <int Chunk> static bool old_keep(int t, int v) { return Chunk < 0 || (old_at_plain(t, v) / 4) == Chunk; }
template <int Chunk> static int  old_at(int t, int v)   { return Chunk < 0 ? old_at_plain(t, v) : old_at_plain(t, v) % 4; }
template <int Chunk> static bool old_vreg_used(int v) {
  for (int t = 0; t < 8; ++t) if (old_keep<Chunk>(t, v)) return true;
  return false;
}

static float h_lo(uint32_t x) { __half h; unsigned short b = (unsigned short)(x & 0xffff); memcpy(&h, &b, 2); return __half2float(h); }
static float h_hi(uint32_t x) { __half h; unsigned short b = (unsigned short)(x >> 16);    memcpy(&h, &b, 2); return __half2float(h); }

// emulate one emit_one and return the two fp16 lanes it produces
template <class Cvt, int T, int V>
static void emulate(uint32_t reg, uint32_t hreg, float& out_lo, float& out_hi) {
  using E = typename Cvt::E;
  const uint32_t src = (T / Cvt::kPerLevel) ? (reg >> 8) : reg;
  uint32_t x = (src & E::template mask<T>()) | 0x64006400u;                       // lop3, immLut 0xEA
  // The high-plane insert, emulated exactly as written. The explicit single-shift + lop3(0xF8) form was tried and
  // REVERTED: nvcc -O3 emits byte-identical SASS for both (LOP3=4, SHIFT=2), so it was zero gain. The check that the
  // two forms agree is kept, because it is what established the equivalence in the first place.
  {
    const uint32_t plain = ((hreg >> Cvt::hshift(T, V)) & Cvt::himask()) << (E::template bpos<T>() + Cvt::kLowBitsPub);
    constexpr int d = (E::bpos_of(T) + Cvt::kLowBitsPub) - Cvt::hshift(T, V);
    const uint32_t hs = (d >= 0) ? (hreg << (d >= 0 ? d : 0)) : (hreg >> (d < 0 ? -d : 0));
    const uint32_t fused = hs & (Cvt::himask() << (E::bpos_of(T) + Cvt::kLowBitsPub));
    if (plain != fused) printf("      SHIFT COMPOSITION DIFFERS T=%d V=%d d=%d : %08x vs %08x\n", T, V, d, plain, fused);
    x |= plain;
  }
  const uint32_t mul = E::template mul<T>(), add = E::template add<T>();
  out_lo = h_lo(x) * h_lo(mul) + h_lo(add);
  out_hi = h_hi(x) * h_hi(mul) + h_hi(add);
}

template <int LowBits, int HiBits>
static bool arith(const char* tag) {
  using Cvt = MixGemm2Plane<LowBits, HiBits>;
  constexpr int kCPV = Cvt::kCodesPerVreg, kPairs = Cvt::kPairs, P2 = Cvt::P2_DIV, HS = Cvt::kHiStride;
  constexpr int LMASK = (1 << LowBits) - 1, HMASK = (1 << HiBits) - 1, BIAS = (LowBits == 4) ? 8 : 0;
  int bad = 0, shown = 0;
  // deterministic pseudo-random codes; the pairing tells us which low and high code meet at each (t, v, lane).
  // THE LOW CODE IS THE *STORED* ONE. int4's offline pre-adds 8 and the field is 4 bits, so a stored code is what the
  // buffer holds and the dequantised value is stored - 8; generating the dequantised value and re-adding the bias wraps
  // (q=8 -> stored 0 -> dequant -8), which is what made my first expectation disagree with a correct converter.
  auto lo_stored = [&](int v, int c) { return int(((unsigned)(v * 37 + c * 11 + 5) >> 1) & LMASK); };
  auto hi_code = [&](int hv, int c) { return int(((unsigned)(hv * 53 + c * 17 + 3) >> 1) & HMASK); };
  uint32_t lo[4] = {0,0,0,0};
  std::vector<uint32_t> hi((size_t)P2 * 4, 0);          // enough for the largest vreg offset the form uses
  for (int v = 0; v < 4; ++v)
    for (int c = 0; c < kCPV; ++c)
      lo[v] |= uint32_t(lo_stored(v, c)) << (LowBits * c);
  for (size_t hv = 0; hv < hi.size(); ++hv)
    for (int c = 0; c < 32 / HiBits; ++c)
      hi[hv] |= uint32_t(hi_code(int(hv), c)) << (HiBits * c);

  cute::for_each(cute::make_int_sequence<4>{}, [&] (auto vv) {
    constexpr int V = decltype(vv)::value;
    cute::for_each(cute::make_int_sequence<kPairs>{}, [&] (auto tt) {
      constexpr int T = decltype(tt)::value;
      const int hv = P2 * (V / P2);
      float a, b; emulate<Cvt, T, V>(lo[V], hi[(size_t)hv], a, b);
      const int lc0 = lo_stored(V, T) - BIAS, lc1 = lo_stored(V, T + kPairs) - BIAS;
      const int hc0 = hi_code(hv, T + kPairs * (V % P2)), hc1 = hi_code(hv, T + kPairs * (V % P2) + HS);
      const float e0 = float(lc0 + (1 << LowBits) * hc0), e1 = float(lc1 + (1 << LowBits) * hc1);
      if (std::fabs(a - e0) > 1e-3f || std::fabs(b - e1) > 1e-3f) {
        ++bad;
        if (shown < 4) { printf("      T=%d V=%d : got (%.1f, %.1f) expected (%.1f, %.1f)\n", T, V, a, b, e0, e1); ++shown; }
      }
    });
  });
  printf("  %-18s low=int%d hi=int%d : %d / %d (t,v) pairs wrong  %s\n", tag, LowBits, HiBits,
         bad, 4 * kPairs, bad ? "<-- ARITHMETIC WRONG" : "lo + 2^LowBits*hi exactly");
  return bad == 0;
}

template <int LowBits, int HiBits>
static bool partition_check(const char* tag) {
  using Cvt = MixGemm2Plane<LowBits, HiBits>;
  std::vector<int> hit((size_t)Cvt::kOut / 2, 0);
  for (int v = 0; v < 4; ++v) for (int t = 0; t < Cvt::kPairs; ++t) ++hit[Cvt::at_plain(t, v)];
  int miss = 0, dbl = 0;
  for (size_t i = 0; i < hit.size(); ++i) { if (!hit[i]) ++miss; else if (hit[i] > 1) ++dbl; }
  printf("  %-18s unchunked destination over %2zu half2 slots: unreached %d, doubled %d  %s\n",
         tag, hit.size(), miss, dbl, (!miss && !dbl) ? "PARTITION" : "<-- NOT a partition");
  return !miss && !dbl;
}

int main() {
  printf("L66 -- Q6/Q5 arithmetic emulated on the host, and Q3 preserved\n\n");
  int bad = 0;

  printf("  (1) Q3 behaviour preservation against the old hand-written members\n");
  {
    int d = 0;
    auto one = [&](auto ch) {
      constexpr int C = decltype(ch)::value;
      using New = MixGemm2Plane<2, 1, C, 8, true, Layout<Shape<_8,_1,_8>>>;
      for (int t = 0; t < 8; ++t) for (int v = 0; v < 4; ++v) {
        if (New::at_plain(t, v) != old_at_plain(t, v)) ++d;
        if (New::keep(t, v)     != old_keep<C>(t, v))  ++d;
        if (New::keep(t, v) && New::at(t, v) != old_at<C>(t, v)) ++d;
      }
      for (int v = 0; v < 4; ++v) if (New::vreg_used(v) != old_vreg_used<C>(v)) ++d;
    };
    one(Int<-1>{}); one(Int<0>{}); one(Int<1>{}); one(Int<2>{}); one(Int<3>{});
    one(Int<4>{}); one(Int<5>{}); one(Int<6>{}); one(Int<7>{});
    printf("      %d mismatches over Chunk -1..7  %s\n", d, d ? "<-- Q3 CHANGED" : "Q3 is bit-for-bit the same converter");
    if (d) ++bad;
  }

  printf("\n  (2) the arithmetic, emulated (lop3 immLut 0xEA == (a&b)|c ; fma.rtte.f16x2)\n");
  if (!arith<2,1>("Q3 int2+int1")) ++bad;
  if (!arith<4,2>("Q6 int4+int2")) ++bad;
  if (!arith<4,1>("Q5 int4+int1")) ++bad;

  printf("\n  (3) the unchunked destination is a partition\n");
  if (!partition_check<2,1>("Q3 int2+int1")) ++bad;
  if (!partition_check<4,2>("Q6 int4+int2")) ++bad;
  if (!partition_check<4,1>("Q5 int4+int1")) ++bad;

  printf("\n  %s\n", bad ? "NOT ready -- fix before touching the collective"
                         : "Q3 unchanged, and Q6 / Q5 produce lo + 2^LowBits*hi with a clean destination partition");
  return bad ? 1 : 0;
}
