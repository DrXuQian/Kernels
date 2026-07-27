// L15 -- what of the remaining chain is expressible as a stock cute Layout, and what needs more?
// A cute Layout is a stride-based affine map, so a piece qualifies iff it is a pure PERMUTATION OF INDEX BITS
// (for power-of-two extents). Anything with a carry -- e.g. a rotate that is not an XOR -- needs a Swizzle or a
// custom functor instead. That distinction decides how much work "make it derivable in cute" actually is.
#include <cstdio>
#include <vector>
#include <set>
#include <initializer_list>
#include <utility>

// int4's L3, as modelled in l12 (4x base-8 + two explicit permutations)
static int l3_int4(int c) { const int g=c/8, ci=c%8, G=((g&1)<<1)|((g>>1)&1);
  const int e8=2*(ci&3)+((ci>>2)&1), idx=8*G+e8, b=idx/4, r=idx%4;
  return 4*((b&~3)|((b&1)<<1)|((b>>1)&1)) + r; }

// is f a pure bit permutation on [0,N)?  <=> GF(2)-linear AND each basis vector maps to a single set bit
template <class F> static void bitperm(const char* tag, int N, F f) {
  bool linear = true, perm = true;
  const int f0 = f(0);
  int nb = 0; while ((1 << nb) < N) ++nb;
  for (int x = 0; x < N && linear; ++x)
    for (int b = 0; b < nb && linear; ++b)
      if (f(x ^ (1 << b)) != (f(x) ^ f(1 << b) ^ f0)) linear = false;
  for (int b = 0; b < nb; ++b) { const int v = f(1 << b) ^ f0; if (v == 0 || (v & (v - 1))) perm = false; }
  printf("      %-28s affine=%-3s single-bit basis=%-3s -> %s\n", tag, linear?"yes":"no", perm?"yes":"no",
         (linear && perm && f0 == 0) ? "STOCK cute Layout (mode permutation)"
                                     : (linear ? "Layout + Swizzle (XOR terms)" : "needs a custom functor"));
}

// the swzl read's cross-slice term, straight out of ppu_tsm_ld_swzl_sim
static int ssv(int slice) { return (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2; }

int main() {
  printf("L15 -- expressibility of what is left\n");
  printf("  *** CORRECTION: the cross-slice conclusion below was WRONG, and l19_multislice.cu overturned it. ***\n"
         "  The rotate that carries for slices 2 and 3 is part of the SWIZZLE, which the AIU write cancels, so the\n"
         "  LOGICAL map has no rotate and needs no custom functor -- the slice is one more mode with stride 8.\n"
         "  l19 measured it: {strip,keep} x {slice-major,rowblock-major}, and only strip+slice-major yields the fp16\n"
         "  identity at 2 and 4 slices. What follows is about the PHYSICAL address, which is not what has to be\n"
         "  modelled. Kept because the distinction is the whole lesson.\n\n");
  printf("  == converter emission (L3), per width\n");
  bitperm("int4 (4x base8 + 2 swaps)", 32, l3_int4);
  bitperm("int1", 128, [](int j){ const int v=j/32, c=j%32, b0=c&1,b1=(c>>1)&1,b2=(c>>2)&1,cc=(c>>3)&1,d=(c>>4)&1;
    return 64*(v%2)+4*(v/2)+2*b0+8*b1+16*b2+32*cc+d; });
  bitperm("int2", 64, [](int x){ const int v=x/16, c=x%16, m=c&3, lv=(c>>2)&1, hf=(c>>3)&1;
    return 2*(16*(v&1)+2*(v>=2)+(m&1)+4*((m>>1)&1)+8*lv)+hf; });

  printf("\n  == swzl read, cross-slice term  vreg_vec_idx_swz2 = (vec + ssv(slice)) %% 8\n");
  printf("      slice : ssv : is (+ssv)%%8 the same as (^ssv) on 3 bits?\n");
  for (int s = 0; s < 4; ++s) {
    bool xorable = true;
    for (int v = 0; v < 8; ++v) if (((v + ssv(s)) % 8) != (v ^ ssv(s))) xorable = false;
    printf("        %d   :  %d  : %s\n", s, ssv(s), xorable ? "yes -> stock Swizzle" : "NO (carries) -> custom functor");
  }
  printf("\n      cube width -> slices: AiuContByteSize = min(TK*Bits/8, 128), slice = 32B\n");
  for (auto p : {std::pair<const char*,int>{"int4 TK=64 (production)",32}, {"int4 TK=128",64},
                 {"int4 TK=256",128}, {"int2 TK=64 fold",32}, {"int1 TK=128 fold",32}}) {
    const int slices = p.second / 32;
    printf("        %-24s %3dB = %d slice(s) -> %s\n", p.first, p.second, slices,
           slices <= 2 ? "stock cute (all ssv are XOR-able)" : "needs the custom functor for slices 2,3");
  }
  return 0;
}
