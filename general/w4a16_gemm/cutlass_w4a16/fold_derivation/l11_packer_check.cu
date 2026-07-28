// L11 -- does the packer reproduce the derived placement? Same binary-label trick as l7, but now against
// nfold_place_bits_int1_tk64 instead of the shipped offline, and compared with l10's chain map.
//
// Two checks:
//   1. the packer's map is a bijection over the tile
//   2. it agrees with the chain (L2 o L3 o pi o L4) at EVERY position -- 0 mismatch or the packer is wrong
// Plus a regression reminder: the same chain code is what reproduces the shipped offline 0/16384 in l10 step 1.
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"
#include <cstdio>
#include <vector>
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
int main(){
  const int NN=128, KK=64, TN=128, TK=64, F=4, Ng=TN/F, CPW=32, W_ROW=8;
  const size_t nbytes=(size_t)NN*KK/8;
  // recover the packer's map with a binary label (NN*KK = 8192 -> 13 bits)
  std::vector<int> id(nbytes*8, 0);
  for (int b=0;b<13;++b){
    std::vector<int8_t> in(nbytes,0), out(nbytes,0);
    for (int n=0;n<NN;++n) for (int k=0;k<KK;++k){ size_t i=(size_t)n*KK+k;
      if ((i>>b)&1) in[i/8] |= int8_t(1<<(i%8)); }
    legacy::nfold_place_bits_int1_tk64(out.data(), in.data(), NN, KK, TN, TK);
    for (size_t p=0;p<nbytes*8;++p) if ((out[p/8]>>(p%8))&1) id[p] |= (1<<b);
  }
  std::set<int> uniq(id.begin(), id.end());
  printf("L11 -- packer vs the derived placement\n\n  packer map: %zu distinct ids over %zu positions -> %s\n",
         uniq.size(), nbytes*8, uniq.size()==nbytes*8 ? "bijection" : "NOT a bijection");

  // the chain, with pi from the collective's row-major stride
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<_2,_2,_1>>, Tile<_32,_32,Int<TK>>>;  // TM64/WM32, TN128/WN64
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));
  // pi = frag.layout()^-1, and cute HAS this: right_inverse(Layout). It used to be a hand-rolled loop here (and in
  // seven sibling files); l30_should_have_been_cute.cu verifies the two agree on three shapes including int4's
  // production one. Worth replacing beyond tidiness: l20 is the file scheduled to become the production offline.
  auto pi = right_inverse(frag.layout());
  auto l3=[](int j,int v){ int b0=j&1,b1=(j>>1)&1,b2=(j>>2)&1,c=(j>>3)&1,d=(j>>4)&1;
    return 64*(v%2)+4*(v/2)+2*b0+8*b1+16*b2+32*c+d; };
  int bad=0, shown=0;
  for (int t=0;t<128;++t){
    const int lane=t%32, w=t/32, warp_n=w/2;   // WOM = TM/WM = 2
    auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
    for (int v=0;v<4;++v){
      const int row=16*warp_n+(v/2)*8+lane/4, wd=(v%2)*4+lane%4;
      for (int j=0;j<CPW;++j){
        auto c=part(pi(l3(j,v)));
        const int gt=id[((size_t)row*W_ROW+wd)*CPW+j];
        if (int(get<0>(c))!=gt/KK || int(get<1>(c))!=gt%KK){ ++bad;
          if (shown<5){ printf("    MISMATCH row%2d w%d bit%2d: chain(n%3d,k%2d) packer(n%3d,k%2d)\n",
            row,wd,j,int(get<0>(c)),int(get<1>(c)),gt/KK,gt%KK); ++shown; } }
      }
    }
  }
  printf("  packer vs chain: %d / %d mismatch  %s\n", bad, Ng*W_ROW*CPW,
         bad ? "" : "<-- PACKER MATCHES THE DERIVED PLACEMENT");
  return bad ? 1 : 0;
}
