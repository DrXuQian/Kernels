// L35 -- at() and keep() as LAYOUT COMPOSITIONS, proven identical to the hand arithmetic they replace.
//
// The old versions were
//     keep = ((e/8) % MMA_K) == Chunk        at = ((e%8) + 8*(e/(8*MMA_K))) / 2
// where /8, %8 and 8*MMA_K are tCrB_mma's strides typed out by hand. Getting those wrong -- I assumed a compact
// (8, 32) where the fragment is ((1,2,4), 32, 8) -- is what produced "MMA_N atom 0 right, atoms 1-3 wrong" on the box.
// As layouts, with L_src taken FROM the fragment instead of restated:
//     L_dst = shape(L_src) : (val strides,  size(val), 0)      compact in n, stride-0 in k = ONE k-atom
//     L_k   = shape(L_src) : (zeros,        0,         1)      which k-atom an element belongs to
//     At = composition(L_dst, right_inverse(L_src))    Ka = composition(L_k, right_inverse(L_src))
// cute simplifies both symbolically to (8,MMA_N,MMA_K):(1,8,0) and (8,MMA_N,MMA_K):(0,0,1).
//
// This is the same lesson as l30: prefer the algebra cute already has over index arithmetic that happens to work. The
// difference here is that the arithmetic did NOT happen to work, and the layout version cannot go wrong the same way
// because it FOLLOWS the fragment rather than restating it.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l35_at_keep_is_layout.cu -o l35 && ./l35
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
using cutlass::MixGemmEmit; using cutlass::MixGemmChunkEmit;
int main(){
  constexpr int MMA_K=4, NPAIR=16; long bad=0;
  for (int c=0;c<MMA_K;++c) for (int v=0;v<4;++v) for (int t=0;t<NPAIR;++t) {
    const int e=MixGemmEmit<1>::index(t,v);
    const bool hk = ((e/8)%MMA_K)==c, ck = MixGemmChunkEmit<1, 0,4>::keep(t,v) && c==0 ? true : false;
    // compare per-chunk via the templated keep
    bool ck2=false; int ca=-1;
    if (c==0){ck2=MixGemmChunkEmit<1, 0,4>::keep(t,v); ca=MixGemmChunkEmit<1, 0,4>::at(t,v);}
    if (c==1){ck2=MixGemmChunkEmit<1, 1,4>::keep(t,v); ca=MixGemmChunkEmit<1, 1,4>::at(t,v);}
    if (c==2){ck2=MixGemmChunkEmit<1, 2,4>::keep(t,v); ca=MixGemmChunkEmit<1, 2,4>::at(t,v);}
    if (c==3){ck2=MixGemmChunkEmit<1, 3,4>::keep(t,v); ca=MixGemmChunkEmit<1, 3,4>::at(t,v);}
    const int ha = ((e%8)+8*(e/(8*MMA_K)))/2;
    if (ck2!=hk || (hk && ca!=ha)) { ++bad; if(bad<=4) printf("  MISMATCH c%d t%2d v%d: keep %d/%d at %d/%d\n",c,t,v,ck2,hk,ca,ha); }
  }
  printf("  layout-based keep/at vs hand arithmetic: %ld / %d differ %s\n", bad, 4*4*NPAIR,
         bad?"<-- REGRESSION":"<-- IDENTICAL");
  return bad?1:0;
}
