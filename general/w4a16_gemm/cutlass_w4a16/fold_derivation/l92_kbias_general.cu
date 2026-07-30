// L92 -- PHASE 1 GATE: MixGemmChunkEmit's kBias is now a template parameter. Verify the magic-number identity
// NUMERICALLY for every (Bits, Bias, bpos, code), not just that two hex constants were reproduced.
//
// The scheme: lop3 gives x = 1024 + c*2^bpos in fp16, then one fma with
//     mul = 2^-bpos                 encoded as exponent field 15-bpos
//     add = -(2^(10-bpos) + Bias)   encoded as sign | (25-bpos)<<10 | Bias<<bpos
// must yield exactly c - Bias. Every intermediate is an exactly-representable fp16 integer or power of two in the
// ranges used (x <= 2047, 1024*2^-bpos integral for bpos <= 10, |add| < 2048), so double arithmetic decides it.
//
// Exactness of `add` needs its mantissa field to fit 10 bits: Bias << bpos_max < 1024. That is a static_assert in
// the header; here it is exercised by trying every legal Bias for each width.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
using cutlass::MixGemmChunkEmit;

static double h2d(uint16_t h) {                       // fp16 bits -> double, normals and zero only (all we emit)
  int s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  double v = (e == 0) ? std::ldexp(double(m), -24) : std::ldexp(1.0 + double(m) / 1024.0, e - 15);
  return s ? -v : v;
}

static int g_fail = 0;
template <int Bits, int Bias>
static void check(char const* name) {
  using E = MixGemmChunkEmit<Bits, -1, 1, true,
      cute::Layout<cute::Shape<cute::Shape<cute::_2,cute::_2,cute::_2>, cute::_4, cute::_4>,
                   cute::Stride<cute::Stride<cute::_1,cute::_2,cute::_4>, cute::_32, cute::_8>>, Bias>;
  constexpr int kPairs = E::kPairsPerVreg;
  int bad = 0;
  cute::for_each(cute::make_int_sequence<kPairs>{}, [&] (auto t_) {
    constexpr int T    = decltype(t_)::value;
    constexpr int bpos = E::template bpos<T>();
    uint16_t const mulh = uint16_t(E::template mul<T>() & 0xFFFF);
    uint16_t const addh = uint16_t(E::template add<T>() & 0xFFFF);
    double const mul = h2d(mulh), add = h2d(addh);
    for (int c = 0; c < (1 << Bits); ++c) {
      double x = 1024.0 + double(c) * std::ldexp(1.0, bpos);      // what the lop3 leaves in the mantissa
      double y = x * mul + add;
      if (y != double(c - Bias)) {
        if (bad < 4) std::printf("  [FAIL] %s bpos=%d c=%d: got %g want %d (mul=%.10g add=%.10g)\n",
                                 name, bpos, c, y, c - Bias, mul, add);
        ++bad;
      }
    }
    if (T < 4) std::printf("  %-14s bpos=%d  mul=0x%04X (%.10g)  add=0x%04X (%.10g)\n",
                           name, bpos, mulh, mul, addh, add);
  });
  std::printf("  %-14s %d codes x %d bpos -> %d bad\n", name, 1 << Bits, kPairs, bad);
  g_fail += bad;
}

int main() {
  std::printf("== plan #20 phase 1: generalised kBias ==\n");
  check<4, 8>("int4 Bias=8");     // the SHIPPED path -- must be bit-identical to before
  check<2, 0>("int2 Bias=0");     // the shipped int2 path
  check<1, 0>("int1 Bias=0");
  check<2, 4>("int2 Bias=4");     // NEW: what Q3_K's int2 plane needs to become ScaleOnly
  check<1, 1>("int1 Bias=1");
  check<4, 0>("int4 Bias=0");

  // The four constants plan #20 derived by hand for int2 Bias=4, checked against what the header now emits.
  {
    using E4 = MixGemmChunkEmit<2, -1, 1, true,
        cute::Layout<cute::Shape<cute::Shape<cute::_2,cute::_2,cute::_2>, cute::_4, cute::_4>,
                     cute::Stride<cute::Stride<cute::_1,cute::_2,cute::_4>, cute::_32, cute::_8>>, 4>;
    uint16_t want[4] = {0xE404, 0xDC10, 0xD440, 0xCD00};   // plan's 0x6404/0x5C10/0x5440/0x4D00 with the sign bit
    uint16_t got[4]  = {uint16_t(E4::template add<0>() & 0xFFFF), uint16_t(E4::template add<1>() & 0xFFFF),
                        uint16_t(E4::template add<2>() & 0xFFFF), uint16_t(E4::template add<3>() & 0xFFFF)};
    for (int i = 0; i < 4; ++i) {
      bool ok = got[i] == want[i];
      std::printf("  int2 Bias=4 add[bpos=%d] = 0x%04X (plan 0x%04X) -> %s\n", 2 * i, got[i], want[i], ok?"ok":"FAIL");
      if (!ok) ++g_fail;
    }
  }
  // Regression: int4's shipped constants must be untouched by the generalisation.
  {
    using E8 = MixGemmChunkEmit<4>;
    uint16_t a0 = uint16_t(E8::template add<0>() & 0xFFFF), a1 = uint16_t(E8::template add<1>() & 0xFFFF);
    bool ok = (a0 == 0xE408) && (a1 == 0xD480);            // FP16_TOP_MAGIC_NUM's -1032 and NEG_72 from the int4 converter
    std::printf("  int4 Bias=8 add[0]=0x%04X add[1]=0x%04X (want 0xE408 / 0xD480) -> %s\n", a0, a1, ok?"ok":"FAIL");
    if (!ok) ++g_fail;
  }
  std::printf("== %s: %d mismatches ==\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
