// L22 -- can cute express a WIDTH-EXPANDING convert? Answer: the position/width bookkeeping yes, the arithmetic no.
//
// Three separate things get conflated in "the converter":
//   (a) the POSITION map      code i -> fp16 element j          <- a plain cute Layout, shown below
//   (b) the WIDTH change      Bits-wide code -> 16-bit element   <- cute's recast/upcast/downcast, and Copy_Traits'
//                                                                  Src/Dst layouts are already in BITS for this
//   (c) the ARITHMETIC        (code & MASK) | 0x6400, then fma    <- cute has no vocabulary for values at all
//
// (a) is a Layout only because the converter is 1:1 in element COUNT. A 2-plane bit-plane concat is 2:1 (int2 low +
// int1 high -> one fp16), so its position map is not one Layout but a TUPLE of layouts sharing a codomain -- which is
// exactly the shape of MixGemm2Plane_uint2_uint1 with its hi[kb + 2*(v>>1)] pairing.
#include "cute/tensor.hpp"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
using namespace cute;

// (a) MixGemmEmit as an actual cute Layout, per width. Coord is ((c0..c_{nb-2}, half), (v%2, v/2)).
using Emit1 = Layout<Shape <Shape<_2,_2,_2,_2,_2>, Shape<_2,_2>>,
                     Stride<Stride<_2,_8,_16,_32,_1>, Stride<_64,_4>>>;
using Emit2 = Layout<Shape <Shape<_2,_2,_2,_2>,    Shape<_2,_2>>,
                     Stride<Stride<_2,_8,_16,_1>,   Stride<_32,_4>>>;
using Emit4 = Layout<Shape <Shape<_2,_2,_2>,       Shape<_2,_2>>,
                     Stride<Stride<_2,_8,_1>,      Stride<_16,_4>>>;

template <int Bits, class L>
static bool check(const char*, L lay) {
  constexpr int CPW = 32 / Bits;
  int bad = 0;
  for (int v = 0; v < 4; ++v)
    for (int c = 0; c < CPW; ++c) {
      int got;
      if constexpr (Bits == 1) got = lay(make_coord(make_coord(c&1,(c>>1)&1,(c>>2)&1,(c>>3)&1,(c>>4)&1),
                                                   make_coord(v&1, v>>1)));
      else if constexpr (Bits == 2) got = lay(make_coord(make_coord(c&1,(c>>1)&1,(c>>2)&1,(c>>3)&1),
                                                        make_coord(v&1, v>>1)));
      else got = lay(make_coord(make_coord(c&1,(c>>1)&1,(c>>2)&1), make_coord(v&1, v>>1)));
      if (got != cutlass::MixGemmEmit<Bits>::index(c, v)) ++bad;
    }
  printf("  int%-2d  ", Bits); print(lay); printf("   vs MixGemmEmit: %d / %d differ %s\n",
         bad, 4*CPW, bad ? "" : "<-- IS a cute Layout");
  return bad == 0;
}

int main() {
  printf("L22 -- is the converter's position map a cute Layout?\n\n");
  bool ok = true;
  ok &= check<1>("int1", Emit1{});
  ok &= check<2>("int2", Emit2{});
  ok &= check<4>("int4", Emit4{});

  printf("\n  (b) the WIDTH change is cute's existing vocabulary, not something new:\n");
  printf("      recast<T> / upcast<N> / downcast<N> rescale a layout when the element width changes -- that is how\n");
  printf("      cvt_in = recast<uint1b_t>(tCrB_load(_,_,k)) turns 16 int8 into 128 codes (measured in l9).\n");
  printf("      And Copy_Traits' SrcLayout/DstLayout are already declared in BITS, precisely so a copy between two\n");
  printf("      different widths is describable: Copy_Atom derives NumValSrc = size<1>(SrcLayout)/sizeof_bits<Src>.\n");
  printf("      So a Bits->16 'copy' has a perfectly good cute description; the swzl atom's own DstLayout\n");
  printf("      ((_32,(_32,_4)) with bit strides) is exactly that form.\n");

  printf("\n  (c) the ARITHMETIC is out of scope by construction. A Layout maps coordinates to integers; it has no\n");
  printf("      notion of the VALUE at a position. (code & MASK) | 0x6400 and the fma live in a functor\n");
  printf("      (NumericArrayConverter / TransformB), which is how cutlass separates the two everywhere.\n");

  printf("\n  WHY (a) IS A SINGLE LAYOUT AT ALL: the converter is 1:1 in element count -- N codes in, N fp16 out.\n");
  printf("      A 2-plane bit-plane concat is 2:1 (int2 low + int1 high -> one fp16), so its position map is a TUPLE\n");
  printf("      of layouts into a shared codomain, not one layout. That is the actual shape of\n");
  printf("      MixGemm2Plane_uint2_uint1, whose hi[kb + 2*(v>>1)] pairing IS the second layout, written by hand.\n");
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
