// L29 -- is MixGemmEmit REALLY what the converters do? Encode each converter's ACTUAL emission from its own source
// and compare, so the claim can stop being a comment.
//
// WHY THIS IS THE GATE FOR MAKING MixGemmEmit THE SOURCE OF TRUTH. MixGemmEmit exists, self-asserts bijectivity, and
// has ZERO callers -- the hand-written emission order in each converter is still the truth, so a NEW bit width still
// requires hand-deriving it. That is where the int2 sigma_n = n & ~8 bug came from. But "replace the hand-written
// order with MixGemmEmit" is only safe if they agree EXACTLY, and the three converters have three different shapes:
//
//   int2 <half_t,uint2b_t,64>   a flat table: base = 16*(v&1) + 2*(v>=2), offsets {0,1,4,5,8,9,12,13}, and the
//                               lop3 masks put crumb c<8 in the LOW half2 lane and c+8 in the HIGH lane
//   int4 <DstType,int4b_t,32>   COMPOSED: base-8's internal order, then a vreg permutation (0,2,1,3), then a
//                               4-element block swap (1<->2, 5<->6)
//   int1 <half_t,uint1b_t,128>  two levels (reg and reg>>8) with a 16-offset step, and a STALE comment ABOVE the
//                               struct describing the previous implementation -- encoding that comment instead of
//                               the code cost 120/128 mismatches on the first run of this file
//
// So this file encodes all three from their source and diffs them against MixGemmEmit<Bits>::index(). int4's is the
// interesting one: three separate permutations compose to exactly the closed-form formula, which is much stronger
// evidence for the model than any single flat table.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l29_emit_is_converter.cu -o l29 && ./l29
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using cutlass::MixGemmEmit;

// ---- int2 <half_t, uint2b_t, 64>: read straight off the _E(...) lines ----
static int emit_int2(int c, int v) {
  static const int off[8] = {0, 1, 4, 5, 8, 9, 12, 13};   // h2 offsets of the 8 lop3 pairs, in crumb order
  const int base = 16 * (v & 1) + 2 * (v >= 2);
  return 2 * (base + off[c & 7]) + (c >= 8 ? 1 : 0);      // c<8 -> low half2 lane, c>=8 -> high lane
}

// ---- int4 <DstType, int4b_t, 32>: base-8 order, then vreg perm, then the 4-block swap ----
static int emit_int4(int c, int v) {
  const int l  = 2 * (c & 3) + (c >= 4 ? 1 : 0);          // base-8: h[c&3], low lane for c<4, high for c>=4
  const int p[4] = {0, 2, 1, 3};                          // result_ptr[0]=s0, [2]=s1, [1]=s2, [3]=s3
  const int g0 = 8 * p[v] + l;
  int b = g0 / 4;                                         // elt_b16[1]<->[2], elt_b16[5]<->[6]
  if      (b == 1) b = 2; else if (b == 2) b = 1;
  else if (b == 5) b = 6; else if (b == 6) b = 5;
  return 4 * b + (g0 % 4);
}

// ---- int1 <half_t, uint1b_t, 128>: from the CODE, not the comment above the struct ----
// MY FIRST VERSION OF THIS FUNCTION WAS WRONG and l29 caught it: I encoded the comment block ABOVE the struct
// ("16 atoms, atom a (g=a/8, j=a%8) ..."), which describes the STAGE-1 magic-OR implementation that the current
// Stage-2 lop3 code replaced. 120 of 128 entries mismatched. The comment INSIDE convert() is the accurate one.
// Actual structure: base = 32*(v&1) + 2*(v>=2); 16 pairs, t<8 use src=reg and t>=8 use src=reg>>8, at
// h2[base + {0,1,4,5,8,9,12,13}[t%8] + 16*(t>=8)], mask (1<<b)|(1<<(16+b)) so the low half2 lane takes bit b of
// src and the high lane bit 16+b. Bit c of reg therefore has b = c&7, level = (c>>3)&1, lane = (c>>4)&1.
static int emit_int1(int c, int v) {
  static const int off[8] = {0, 1, 4, 5, 8, 9, 12, 13};
  const int base = 32 * (v & 1) + 2 * (v >= 2);
  const int b = c & 7, level = (c >> 3) & 1, lane = (c >> 4) & 1;
  return 2 * (base + off[b] + 16 * level) + lane;
}

template <int Bits>
static bool check(const char* tag, int (*f)(int, int)) {
  constexpr int CPW = 32 / Bits;
  long bad = 0; int shown = 0;
  std::vector<int> seen(4 * CPW, -1);
  for (int v = 0; v < 4; ++v)
    for (int c = 0; c < CPW; ++c) {
      const int a = f(c, v), b = MixGemmEmit<Bits>::index(c, v);
      if (a != b) { ++bad; if (shown < 4) {
        printf("      MISMATCH vreg%d code%2d: converter -> %3d, MixGemmEmit -> %3d\n", v, c, a, b); ++shown; } }
      if (a >= 0 && a < 4 * CPW) seen[a] = v * CPW + c;
    }
  int unmapped = 0; for (int e = 0; e < 4 * CPW; ++e) if (seen[e] < 0) ++unmapped;
  printf("  %-34s int%d: %2d codes x 4 vregs -> %3d outputs | %ld mismatch, %d output(s) unwritten %s\n",
         tag, Bits, CPW, 4 * CPW, bad, unmapped,
         (bad == 0 && unmapped == 0) ? "<-- MixGemmEmit IS the converter" : "<-- DIVERGES");
  return bad == 0 && unmapped == 0;
}

int main() {
  printf("L29 -- MixGemmEmit vs what each converter actually emits\n\n");
  bool ok = true;
  ok &= check<2>("int2 flat table (masks + offsets)", emit_int2);
  ok &= check<4>("int4 COMPOSED (base8 o perm o swap)", emit_int4);
  ok &= check<1>("int1 flat table (lop3 stage 2)", emit_int1);
  printf("\n  int4 is the load-bearing one: base-8's internal order, a vreg permutation and a 4-element block swap\n");
  printf("  are three independent permutations, and their composition landing exactly on the closed form is much\n");
  printf("  stronger than any single flat table agreeing.\n");
  printf("\n  ALL: %s\n", ok ? "PASS -- MixGemmEmit can become the source of truth" : "FAIL -- do NOT replace anything");
  return ok ? 0 : 1;
}
