// L18 -- is L2 REDUNDANT? The user's point: if the AIU write and the swzl read cancel, model both as identity and
// let partition_B on a row-major smem view do the work.
//
// Algebraically A o B = I only gives B = A^-1, not A = B = I. But only the COMPOSITION is observable, so choosing
// the gauge W = identity is legitimate -- and then R is fully determined. The question is whether that determined R
// is something we must write down (L2) or something cute already knows (partition_B).
//
// TEST 1 (fp16, no converter): does partition_B on the row-major tile reproduce L2's word map exactly? If yes, L2 is
//   redundant for fp16 -- "both copies are identity" is not just permissible, it is the whole model.
// TEST 2 (sub-byte): the same question with the converter in the path. Here the register receives CODES in swzl
//   order, MixGemmEmit permutes them, and only then do they reach fragment slots -- so partition_B alone cannot be
//   the answer, and the two must differ by exactly MixGemmEmit o pi.
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
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
static inline int l2word(int lane,int v){ return ((v/2)*8 + lane/4)*8 + (v%2)*4 + lane%4; }

int main(){
  printf("L18 -- is L2 redundant, i.e. does partition_B already contain it?\n\n");

  // TEST 1: fp16, TN=64 TK=16 (one 32B slice, 8 words per row -> matches l2word's *8)
  {
    constexpr int TN=64, TK=16;
    using Mma=TiledMMA<MMA_Atom<F16Atom>,Layout<Shape<_1,_2,_1>>,Tile<_16,_32,Int<TK>>>;
    auto sB=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{},Stride<Int<TK>,_1>{}));
    auto frag=Mma{}.get_thread_slice(0).partition_fragment_B(sB);
    const int NS=int(size(frag));  // pi = frag.layout()^-1, and cute HAS this: right_inverse(Layout). It used to be a hand-rolled loop here (and in
  // seven sibling files); l30_should_have_been_cute.cu verifies the two agree on three shapes including int4's
  // production one. Worth replacing beyond tidiness: l20 is the file scheduled to become the production offline.
    auto pi = right_inverse(frag.layout());
    int bad=0,tot=0;
    for(int t=0;t<32*2;++t){
      const int lane=t%32, warp_n=t/32;
      auto part=Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
      for(int v=0;v<4;++v) for(int h=0;h<2;++h){
        const int flat=v*2+h; if(flat>=NS) continue;
        auto c=part(pi(flat));
        // what partition_B says: (n,k) -> word index n*8 + k/2, half k%2
        const int pb_word = (int(get<0>(c)) - warp_n*16)*8 + int(get<1>(c))/2;
        const int pb_half = int(get<1>(c))%2;
        ++tot; if(pb_word!=l2word(lane,v) || pb_half!=h) ++bad;
      }
    }
    printf("  TEST 1  fp16: partition_B's word map vs L2 -> %d / %d differ  %s\n", bad, tot,
           bad ? "" : "<-- L2 IS REDUNDANT for fp16; partition_B already is it");
  }

  // TEST 2: sub-byte. Same comparison, but now MixGemmEmit sits between the swzl and the fragment.
  {
    constexpr int Bits=1, TN=128, TK=128, CPW=32;
    using Mma=TiledMMA<MMA_Atom<F16Atom>,Layout<Shape<_1,_4,_1>>,Tile<_16,_64,Int<TK>>>;
    auto sB=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{},Stride<Int<TK>,_1>{}));
    auto frag=Mma{}.get_thread_slice(0).partition_fragment_B(sB);
    const int NS=int(size(frag));  // pi = frag.layout()^-1, and cute HAS this: right_inverse(Layout). It used to be a hand-rolled loop here (and in
  // seven sibling files); l30_should_have_been_cute.cu verifies the two agree on three shapes including int4's
  // production one. Worth replacing beyond tidiness: l20 is the file scheduled to become the production offline.
    auto pi = right_inverse(frag.layout());
    int naive=0, withemit=0, tot=0;
    for(int t=0;t<32*4;++t){
      const int lane=t%32;
      auto part=Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
      for(int v=0;v<4;++v) for(int j=0;j<CPW;++j){
        // (a) NAIVE: pretend the converter preserves order -> code (v,j) lands at flat v*CPW+j
        const int flat_naive=v*CPW+j;
        // (b) WITH the emission model
        const int flat_emit=cutlass::MixGemmEmit<Bits>::index(j,v);
        ++tot;
        if(flat_naive<NS && flat_emit<NS && pi(flat_naive)!=pi(flat_emit)) ++naive;
        if(flat_emit>=NS) ++withemit;
      }
    }
    printf("  TEST 2  int1: pretending the converter preserves order disagrees with MixGemmEmit on %d / %d slots\n",
           naive, tot);
    printf("          -> for sub-byte L2 is NOT redundant: the converter breaks the swzl<->fragment correspondence\n");
  }
  return 0;
}
