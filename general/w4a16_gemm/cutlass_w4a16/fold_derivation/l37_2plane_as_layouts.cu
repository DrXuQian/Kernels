// L37 -- the 2-plane cross-plane correspondence as a TUPLE OF LAYOUTS (task #8).
//
// MixGemm2Plane_uint2_uint1 composes an int2 low plane and an int1 high plane into one fp16 fragment. Today the
// correspondence is hand-written across eight _E2 lines: the low plane reuses the single-plane int2 offsets while the
// high plane is addressed as
//     hreg = hi[2 * (v >> 1)]      hs = 8 * (v & 1)      high bit = hs + t      mantissa position = 2*(t%4) + 2
// Three separate hand-derived index maps now exist (single-plane int2, single-plane int1, and this), and the chunked
// conversion needs to gate BOTH planes consistently -- which would make a fourth. This file checks that the pair is
// just two layouts over ONE domain, so the gate composes instead of being re-derived.
//
// THE CLAIM. Flattening the high plane's source as vreg*32 + bit, the hardcoded addressing is
//     hi_src = 32*(2*(v>>1)) + 8*(v&1) + t  =  64*(v>>1) + 8*(v&1) + t
// which is a sum of per-mode weights, i.e. a cute Layout over (t, (v0, v1)) with strides (1, (8, 64)). The low plane is
// MixGemmEmit<2>::index(t, v), already a Layout. So the correspondence is a TUPLE of two layouts sharing a domain.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l37_2plane_as_layouts.cu -o l37 && ./l37
#include "cute/tensor.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
using namespace cute;
using cutlass::MixGemmEmit;

int main() {
  printf("L37 -- the 2-plane correspondence as two layouts over one domain\n\n");
  // the high plane's source index, as a Layout over (t, (v&1, v>>1))
  auto HI = make_layout(make_shape (_8{}, make_shape(_2{}, _2{})),
                        make_stride(_1{}, make_stride(_8{}, _64{})));
  printf("  HI layout : "); print(HI); printf("\n");
  printf("  LO layout : MixGemmEmit<2>::EmitLayout (already a Layout)\n\n");
  printf("   v  t | hardcoded lo h2  HI hard  HI layout | mantissa pos\n");
  long bad = 0;
  for (int v = 0; v < 4; ++v)
    for (int t = 0; t < 8; ++t) {
      // hardcoded, transcribed from the _E2 lines
      static const int off[8] = {0, 1, 4, 5, 8, 9, 12, 13};
      const int base   = 16 * (v & 1) + 2 * (v >= 2);
      const int lo_h2  = base + off[t];
      const int hi_vr  = 2 * (v >> 1), hs = 8 * (v & 1);
      const int hi_src = 32 * hi_vr + hs + t;
      const int hipos  = 2 * (t % 4) + 2;
      // layout-derived
      const int lo_l   = MixGemmEmit<2>::index(t, v) / 2;
      const int hi_l   = int(HI(make_coord(t, make_coord(v & 1, v >> 1))));
      const bool ok = (lo_h2 == lo_l) && (hi_src == hi_l);
      if (!ok) ++bad;
      if (v < 2 && t < 4)
        printf("  %2d %2d | %15d  %7d  %9d | %d%s\n", v, t, lo_h2, hi_src, hi_l, hipos, ok ? "" : "  <-- DIFFER");
    }
  printf("\n  %ld / 32 differ  %s\n", bad,
         bad ? "<-- NOT a layout pair" : "<-- the correspondence IS a tuple of two layouts");
  printf("\n  Consequence for the chunked 2-plane path (task #12): the chunk gate on the LOW plane is the existing\n");
  printf("  keep() over MixGemmEmit<2>, and the HIGH plane's gate is the SAME predicate pulled through HI -- one\n");
  printf("  composition, not a fourth hand-derived index map.\n");
  printf("\n  ALL: %s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
