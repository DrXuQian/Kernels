// L65 -- the Q6 / Q5 design, gated against Q3's SHIPPED constants before any of it is written into the collective.
//
// Q3 = int2 + int1 works. Q6 = int4 + int2 and Q5 = int4 + int1 are the other two formats the bit-plane architecture
// promised. Writing a second and third hand-derived pairing is exactly what cost this session a day, so instead there is
// ONE closed form for all three, and the test of it is that it REPRODUCES Q3's shipped numbers -- a derivation that
// agrees with working code at the one point where working code exists.
//
// The four claims:
//
//  (1) MixGemmChunkEmit covers int4 once `add` carries the +8 bias. Its mask and mul already matched; with
//      x = 1024 + c*2^bpos and mul = 2^-bpos, add must be -(2^(10-bpos) + Bias), and 2^(10-bpos)+8 has exponent field
//      25-bpos and mantissa 2^(bpos+3). Checked against the two constants the shipped int4 converter hardcodes.
//
//  (2) The destination is the LOW plane's own emission: at_plain(T,V) == MixGemmEmit<LowBits>::index(T,V)/2. The high
//      bits only add magnitude, never position, so a 2-plane output lands in the same fp16 slot a single-plane
//      conversion would use. Checked against Q3's hand-written AtLayout for all 32 (t,v).
//
//  (3) The high-plane extraction shift, for any pair:
//          hshift(T,V) = HiBits * (T + kPairs*(V % P2_DIV))     kPairs = 16/LowBits,  P2_DIV = LowBits/HiBits
//      Q3 (2,1) gives T + 8*(V&1) -- the shipped constant. Q6 (4,2) gives 2T + 8*(V&1); Q5 (4,1) gives T + 4V.
//
//  (4) The high vreg a low vreg reads: hreg = hi[P2_DIV * (V / P2_DIV)]. Q3 gives hi[2*(V>>1)] -- shipped.
//
// And the derivation behind (3)/(4): a delivery is 16 B/thread either way, so the low plane brings 128/LowBits codes in
// 4 vregs while the high plane's 16 B covers 128/HiBits, i.e. P2_DIV = LowBits/HiBits times more than one low delivery
// needs. Within one high vreg the two half2 lanes must be 16 bits apart, so the paired high codes are (a, a + 16/HiBits),
// and idx(v_local, c) = (c % kPairs) + kPairs*v_local + (16/HiBits)*(c / kPairs) is a bijection onto that vreg's codes
// because kPairs * P2_DIV == 16/HiBits.
//
//   nvcc -std=c++17 -Istub_inc -I.. -I../../../../third_party/actlize/include l65_q65_design.cu -o l65 && ./l65
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using namespace cute;
using namespace cutlass;

// Q3's SHIPPED AtLayout and pairing, copied here as the reference. Do not simplify.
using ShippedAt = Layout<Shape <Shape<_2,_4>, Shape<_2, _2>>,
                         Stride<Stride<_1,_4>, Stride<_16,_2>>>;
static int shipped_at_plain(int t, int v) { return int(ShippedAt{}(t + 8 * v)); }
static int shipped_hshift(int t, int v)   { return 8 * (v & 1) + t; }
static int shipped_hvreg(int v)           { return 2 * (v >> 1); }

// the general form
template <int LowBits, int HiBits> struct Pair {
  static constexpr int kPairs  = 16 / LowBits;
  static constexpr int P2_DIV  = LowBits / HiBits;
  static constexpr int kCodes  = 32 / LowBits;             // low codes per vreg
  static constexpr int hstride = 16 / HiBits;              // high codes between the two half2 lanes
  static constexpr int hvregs  = 4 * HiBits / LowBits;     // high vregs one low delivery consumes
  static int hshift(int t, int v) { return HiBits * (t + kPairs * (v % P2_DIV)); }
  static int hvreg (int v)        { return P2_DIV * (v / P2_DIV); }
  static int at_plain(int t, int v) { return MixGemmEmit<LowBits>::index(t, v) / 2; }
};

int main() {
  printf("L65 -- the Q6/Q5 closed form, gated against Q3's shipped constants\n\n");
  int bad = 0;

  // (1) int4's two constants
  {
    using E4 = MixGemmChunkEmit<4, -1, 1>;
    const uint32_t a0 = E4::add<0>(), a1 = E4::add<1>();
    const uint32_t m0 = E4::mask<0>(), m1 = E4::mask<1>();
    const uint32_t u0 = E4::mul<0>(),  u1 = E4::mul<1>();
    const bool ok = (a0 == 0xE408E408u) && (a1 == 0xD480D480u)
                 && (m0 == 0x000f000fu) && (m1 == 0x00f000f0u)
                 && (u0 == 0x3c003c00u) && (u1 == 0x2c002c00u);
    printf("  (1) int4 via MixGemmChunkEmit: mask %08x/%08x mul %08x/%08x add %08x/%08x  %s\n",
           m0, m1, u0, u1, a0, a1,
           ok ? "== the shipped 1032/72 constants" : "<-- DIFFERS from the shipped converter");
    if (!ok) ++bad;
  }

  // (2) the destination is the low plane's own emission
  {
    int d = 0;
    for (int t = 0; t < 8; ++t) for (int v = 0; v < 4; ++v)
      if (Pair<2,1>::at_plain(t, v) != shipped_at_plain(t, v)) ++d;
    printf("  (2) at_plain == MixGemmEmit<2>::index/2 : %d / 32 differ  %s\n", d,
           d ? "<-- the hand-written AtLayout is NOT the low plane's emission" : "identical");
    if (d) ++bad;
  }

  // (3)(4) the pairing, at Q3 where shipped values exist
  {
    int d3 = 0, d4 = 0;
    for (int t = 0; t < 8; ++t) for (int v = 0; v < 4; ++v)
      if (Pair<2,1>::hshift(t, v) != shipped_hshift(t, v)) ++d3;
    for (int v = 0; v < 4; ++v) if (Pair<2,1>::hvreg(v) != shipped_hvreg(v)) ++d4;
    printf("  (3) hshift  : %d / 32 differ  %s\n", d3, d3 ? "<-- WRONG" : "reproduces 8*(V&1)+T");
    printf("  (4) hvreg   : %d / 4  differ  %s\n", d4, d4 ? "<-- WRONG" : "reproduces 2*(V>>1)");
    if (d3 || d4) ++bad;
  }

  // (5) the pairing is a BIJECTION for all three formats -- every high code of every consumed vreg hit exactly once
  auto bij = [&](const char* tag, int LowBits, int HiBits, int kPairs, int P2_DIV, int hstride, int hvregs) {
    const int codes_per_hvreg = 32 / HiBits;
    std::vector<int> hit((size_t)hvregs * codes_per_hvreg, 0);
    for (int v = 0; v < 4; ++v)
      for (int t = 0; t < kPairs; ++t)
        for (int half = 0; half < 2; ++half) {
          const int hv = P2_DIV * (v / P2_DIV);
          const int idx = (t + kPairs * (v % P2_DIV)) + hstride * half;
          if (hv >= hvregs * P2_DIV || idx >= codes_per_hvreg) { printf("      %s OUT OF RANGE hv=%d idx=%d\n", tag, hv, idx); ++bad; return; }
          ++hit[(size_t)(hv / P2_DIV) * codes_per_hvreg + idx];
        }
    int miss = 0, dbl = 0;
    for (size_t i = 0; i < hit.size(); ++i) { if (!hit[i]) ++miss; else if (hit[i] > 1) ++dbl; }
    const int need = 4 * (32 / LowBits);      // low codes in one delivery == high codes needed
    printf("  (5) %-22s low=int%d hi=int%d kPairs=%d P2_DIV=%d hstride=%2d hvregs=%d : need %3d, "
           "space %3zu, unreached %d, doubled %d  %s\n", tag, LowBits, HiBits, kPairs, P2_DIV, hstride, hvregs,
           need, hit.size(), miss, dbl, (!miss && !dbl) ? "BIJECTION" : "<-- NOT a bijection");
    if (miss || dbl) ++bad;
  };
  bij("Q3 = int2 + int1", 2, 1, 8, 2, 16, 2);
  bij("Q6 = int4 + int2", 4, 2, 4, 2,  8, 2);
  bij("Q5 = int4 + int1", 4, 1, 4, 4, 16, 1);

  printf("\n  %s\n", bad ? "the closed form is WRONG somewhere -- do not write it into the collective"
                         : "the closed form reproduces Q3's shipped constants AND is a bijection for Q6 and Q5");
  return bad ? 1 : 0;
}
