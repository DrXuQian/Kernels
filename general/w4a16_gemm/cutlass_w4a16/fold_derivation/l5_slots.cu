// L5/1 -- measure `slots` with the REAL TiledMma shape from the builder, not a stub warp layout.
// builder: WarpOnM = TM/WM, WarpOnN = TN/WN, Tile<TM/WM*16, TN/WN*16, PermK>  (PermK = TK when folding, else 16)
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <set>
using namespace cute;
struct F16Atom {};
namespace cute {
template <> struct MMA_Traits<F16Atom> {
  using ValTypeD=float; using ValTypeA=cutlass::half_t; using ValTypeB=cutlass::half_t; using ValTypeC=float;
  using Shape_MNK=Shape<_16,_16,_16>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
}
template <int Bits,int TM,int TN,int TK,int WM,int WN>
void probe(const char* tag, const char* measured) {
  constexpr int WOM = TM/WM, WON = TN/WN;
  constexpr int contig = TK*Bits/8, F = contig>=32 ? 1 : 32/contig;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;     // PermK = TK (the fold builder's choice)
  auto part = Mma{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
  std::set<int> ns, ks;
  for (int i=0;i<int(size(part));++i){ ns.insert(get<0>(part(i))); ks.insert(get<1>(part(i))); }
  const int slots = int(size(part)), deliv = 16*8/Bits;
  const int words = deliv/(32/Bits);
  printf("  %-20s int%d (%d,%d,%d) w%dx%d F=%d | slots=%4d deliv=%4d %-14s | cols=%2zu k=%2zu words=%2d cols/word=%.2f | %s\n",
    tag,Bits,TM,TN,TK,WM,WN,F,slots,deliv,
    deliv>slots?"OVER-DELIVERY":(deliv==slots?"tight":"under"),
    ns.size(),ks.size(),words, double(ns.size())/words, measured);
}
int main(){
  printf("L5/1 -- slots with the REAL builder TiledMma (WarpOnN = TN/WN, PermN = WarpOnN*16)\n\n");
  printf("  -- the nine reference points, all at the WM=WN=32 the fold tests actually pass --\n");
  probe<4,64, 64, 64,32,32>("int4","OK 55.9%");
  probe<2,64, 64, 64,32,32>("int2 fold","OK 53.2%");
  probe<2,64,128, 64,32,32>("int2 fold wideN","OK bad=0");
  probe<2,64, 64,128,32,32>("int2 unfolded","OK");
  probe<1,32,128,128,32,32>("int1 fold winner","OK bad=0");
  probe<1,64, 64,128,32,32>("int1 fold alt","OK bad=0");
  probe<1,64, 64,256,32,32>("int1 unfolded","OK");
  probe<1,64, 64, 64,32,32>("int1 F=4 narrow","BAD random");
  probe<1,64,128, 64,32,32>("int1 F=4 wide","BAD random");
  printf("\n  -- the escape the retraction pointed at: raise WN so the fragment gets more slots --\n");
  probe<1,64,128, 64,32,64>("int1 TK=64 WN=64","UNTESTED");
  probe<1,64,256, 64,32,64>("int1 TK=64 TN=256","UNTESTED");
  probe<1,64,128, 64,32,128>("int1 TK=64 WN=128","UNTESTED");
  return 0;
}
