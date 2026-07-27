// L21 -- map the scale/zero path. A shape survey, plus two corrections to my own first reading of it.
//
// Nothing had ever modelled this path and it carries the gs=16 penalty, so this is about being able to say what it
// does before proposing a change.
//
// CORRECTION 1 -- A FALSE ALARM. My first probe used partition_B on the (TN,1) scale tensor to read its coordinates
// and got garbage (4164, -1492614480) at most slots, which reads exactly like "the wrong scale is applied to B". It
// was an OUT-OF-BOUNDS host read: partition_B computes addresses, and a K extent of 1 against an mma whose K tile is
// TK indexes past the tensor. The collective never does that -- it uses partition_fragment_B only to borrow the
// SHAPE for make_fragment_like, and the values arrive through copy() from partition_S.
//
// CORRECTION 2 -- I REPORTED THE WRONG QUANTITY. I read the scale fragment's register cost off size(), which counts
// COORDINATES. What costs registers is cosize(), the storage extent, and this layout has stride-0 modes:
//     tCrS  ((_2,_2,_2),_2,_8):((_0,_0,_1),_16,_2)     size=128  cosize=32
// make_fragment_like PRESERVES the zero strides, so the fragment is 32 half = 16 regs/thread, not the 64 I claimed.
// The stride-0 modes are the mma atom's K direction, where a per-(n, group) scale has nothing to vary -- so the
// broadcast is exactly right and there is no waste to reclaim. Measured below; tCrB_mma is printed alongside for
// contrast (size == cosize there, because B really does vary in every mode).
//
//     TK=128:  scale 32 half = 16 regs/thread   (x2 with zero)      B fragment 128 half
//     TK=256:  scale 64 half = 32 regs/thread                       B fragment 256 half
//
// So the answer to "did registers go up" is no, twice over: nothing in this work changed kernel code at all, and the
// scale fragment was never as large as size() suggested.
//
// WHAT REMAINS TRUE: the cost formula from the FINE branch. APG = gs/16 atoms per group, reload at
// atom_idx % APG == 0, so reloads per k-iteration = MMA_K/APG = TK/gs = SK, doubled with zero. int1 is pinned at
// TK>=128 by the delivery bound, so at gs=16 it carries SK=8 where int2/int4 at TK=64 carry SK=4. That is read off
// the code rather than inferred, and it is the only quantitative claim here that a box run should be asked to
// confirm.
//
// CORRECTION 3 -- MEASURED ON ppu001, AND IT OVERTURNED MY MODEL. I predicted that halving TK would halve the
// scale cost, because SK = TK/gs halves. Wrong, and the reason is one line of algebra I should have done first:
//
//     total reloads over the whole K  =  (K/TK) * SK  =  (K/TK) * (TK/gs)  =  K/gs      -- INDEPENDENT of TK
//     reloads per mma atom            =  1/APG        =  16/gs                          -- also independent of TK
//
// Halving TK halves the reloads per k-iteration and doubles the number of iterations. Net zero. TK is not a lever on
// the scale path at all.
//
// int1, M=2048 N=K=4096, MFU:
//                        TK=128 (WN=32)   TK=64 (WN=64)
//     ScaleOnly gs=32        42.0             46.4        +4.4
//     ScaleOnly gs=16        37.3             38.9        +1.6
//     ScaleZero gs=32        36.4             41.2        +4.8
//     ScaleZero gs=16        30.4             31.2        +0.8
//
// The gs=32 gain is real but comes from OCCUPANCY (blocks 8 -> 9) and a smaller A tile per stage, not from scale. And
// the gs=16 penalty got WORSE in absolute time at the smaller TK: +57.7us against +41.8us for ScaleOnly, +106.8
// against +73.8 for ScaleZero. At gs=16, APG = gs/16 = 1 means every mma atom triggers a reload, and TK=64 leaves
// only 4 atoms per iteration to hide it behind where TK=128 has 8. Fewer atoms, worse overlap.
//
// SO, CORRECTED: for the scale path a LARGER TK is better, the opposite of what I claimed. The two levers that
// remain are
//   1. cheaper per reload -- sS and sZ are separate smem tensors and the FINE branch issues two copies from the same
//      group index. Interleaving them into one tensor with a trailing extent-2 mode makes it one copy of twice the
//      width. Zero costs 51-107us here, i.e. up to 20% of runtime at TK=64/gs=16, so this is the biggest single item.
//   2. better hiding -- more mma atoms per scale group, or issuing the reload earlier in the iteration.
// And for int1 at gs=16 specifically, TK=128 plus the zero interleave is the more promising combination, not TK=64.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l21_scalezero_shapes.cu -o l21 && ./l21
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
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
template<int TN,int TK,int SK,int WON> void go(const char* tag){
  using Mma=TiledMMA<MMA_Atom<F16Atom>,Layout<Shape<_1,Int<WON>,_1>>,Tile<_16,Int<WON*16>,Int<TK>>>;
  auto thr=Mma{}.get_thread_slice(0);
  auto sSl=tile_to_shape(Layout<Shape<_8,_1>>{}, make_shape(Int<TN>{},Int<1>{},Int<SK>{}));
  auto sS=make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), sSl);
  printf("  %s TN=%d TK=%d SK=%d\n", tag, TN, TK, SK);
  printf("     sS              "); print(sSl); printf("\n");
  auto fragB = thr.partition_fragment_B(make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                  make_layout(Shape<Int<TN>,Int<TK>>{},Stride<Int<TK>,_1>{})));
  printf("     tCrB_mma        "); print(fragB.layout()); printf("  size=%d\n", int(size(fragB)));
  auto fragS = thr.partition_fragment_B(sS(_,_,Int<0>{}));
  printf("     tCrS (frag)     "); print(fragS.layout()); printf("  size=%d\n", int(size(fragS)));
  auto tiledS = make_tiled_copy_B(Copy_Atom<DefaultCopy, cutlass::half_t>{}, Mma{});
  auto thrS = tiledS.get_thread_slice(0);
  auto tCsS = thrS.partition_S(sS);
  printf("     tCsS            "); print(tCsS.layout()); printf("  rank=%d\n", int(rank(tCsS)));
  auto ownS = make_fragment_like<cutlass::half_t>(fragS);
  // size() counts COORDINATES; cosize() is the storage extent, which is what costs registers. A layout with
  // stride-0 modes has cosize < size, and reporting size as the register cost is simply wrong.
  printf("     tCrS partition   size=%d cosize=%d\n", int(size(fragS)), int(cosize(fragS.layout())));
  printf("     make_fragment_like  layout "); print(ownS.layout());
  printf("   size=%d cosize=%d -> %d half = %d regs/thread (x2 with zero)\n",
         int(size(ownS)), int(cosize(ownS.layout())), int(cosize(ownS.layout())), int(cosize(ownS.layout()))/2);
  printf("     tCrB_mma for comparison   size=%d cosize=%d\n", int(size(fragB)), int(cosize(fragB.layout())));
  auto view = thrS.retile_D(ownS);
  printf("     tCrS_copy_view  "); print(view.layout()); printf("  size=%d\n", int(size(view)));
  printf("     transform slice tCrS(_,_,0) size = %d ;  copy dst view(_,_,0) size = %d\n\n",
         int(size(fragS(_,_,0))), int(size(view(_,_,0))));
}
int main(){ printf("scale path shapes\n\n");
  go<128,128,4,4>("int1 TK128 gs32"); go<64,64,2,2>("int2 TK64 gs32"); go<64,256,8,2>("int1 TK256 gs32"); return 0; }
