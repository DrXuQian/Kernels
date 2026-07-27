// L34 -- what tCrB_mma's layout ACTUALLY is, and what make_fragment_like does to a slice of it.
//
// I assumed the B fragment was compact in (val, MMA_N, MMA_K) order, i.e. e = val + 8*n + 32*k. It is not:
//     tCrB_mma : ((2,2,2), MMA_N, MMA_K) : ((1,2,4), 32, 8)      MMA_N stride 32, MMA_K stride 8
// so e = val + 32*n + 8*k and e/32 == n_atom, NOT k_atom. That one assumption produced a wrong chunk-axis name, a
// wrong keep() predicate and a wrong at(), and l32 "verified" the split -- correctly, but of the wrong model, which
// reads as confirmation. On the box it showed as MMA_N atom 0 correct and atoms 1..3 wrong.
//
// It also shows make_fragment_like re-compacting the slice: the parent slice has MMA_N stride 32 (inherited) while
// the fresh fragment gets 8. That is CORRECT and wanted -- a standalone one-atom buffer should be compact -- but it
// means the emission has to target val + 8*n_atom, not the parent's offsets.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l34_fragment_layout.cu -o l34 && ./l34
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
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
int main(){
  constexpr int TM=64,TN=128,TK=64,WM=32,WN=64, WOM=TM/WM, WON=TN/WN;
  using Mma=TiledMMA<MMA_Atom<F16Atom>,Layout<Shape<Int<WOM>,Int<WON>,_1>>,Tile<Int<WOM*16>,Int<WON*16>,Int<TK>>>;
  auto sB=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                      make_layout(Shape<Int<TN>,Int<TK>>{},Stride<Int<TK>,_1>{}));
  auto tCrB_mma = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  printf("tCrB_mma            : "); print(tCrB_mma.layout()); printf("\n");
  auto slice = tCrB_mma(_,_,Int<0>{});
  printf("tCrB_mma(_,_,0)     : "); print(slice.layout()); printf("\n");
  auto one = make_fragment_like(slice);
  printf("make_fragment_like  : "); print(one.layout()); printf("\n");
  printf("\n  flat offsets of (val, n_atom) -- slice vs one-atom fragment:\n");
  long bad=0;
  for (int n=0;n<int(size<1>(slice));++n) for (int v=0;v<8;++v) {
    const int a=int(slice.layout()(v,n)), b=int(one.layout()(v,n));
    // the slice's offsets are relative to the parent, so compare against the slice minus its own base
    const int a0=a-int(slice.layout()(0,0));
    if (a0!=b) { ++bad; if (bad<=8) printf("    n_atom=%d val=%d : slice %3d (rel %3d)  one %3d  <-- DIFFER\n",n,v,a,a0,b); }
  }
  printf("  %ld / %d differ  %s\n", bad, 8*int(size<1>(slice)),
         bad ? "<-- make_fragment_like REORDERS; that is the MMA_N stride bug" : "<-- identical, layout is not the bug");
  return 0;
}
