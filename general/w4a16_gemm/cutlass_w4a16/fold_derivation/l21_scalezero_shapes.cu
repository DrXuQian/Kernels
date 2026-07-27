// L21 -- map the scale/zero path. NOT a correctness proof: a shape survey plus two things worth measuring.
//
// Nothing had ever modelled this path, and it carries the gs=16 penalty (7-14 points), so "we would not know what to
// optimise" was the honest state.
//
// FIRST, A FALSE ALARM I ALMOST REPORTED. My initial probe used partition_B on the (TN,1) scale tensor to read its
// coordinates and found garbage (4164, -1492614480) at most slots, which reads exactly like "the wrong scale is
// applied". It was an OUT-OF-BOUNDS host read: partition_B computes addresses, and a K extent of 1 against an mma
// whose K tile is TK indexes past the tensor. The collective never does that -- it uses partition_fragment_B only to
// borrow the SHAPE, and the values arrive through copy() from partition_S. Lesson repeated: a probe that reads
// garbage looks identical to a bug.
//
// WHAT THE SHAPES SAY (dumped below):
//   * tCrS is partitioned LIKE B, so its size follows B's fragment -- 128 half at TK=128, 256 at TK=256.
//     make_fragment_like allocates that compactly: 64 regs/thread at TK=128, 128 at TK=256, DOUBLED with zero.
//     Against 256 regs/thread that is a large footprint for a value that is per-(n, group).
//   * the transform consumes tCrS(_,_,0) -- 16 elements -- and always slice 0, i.e. the SAME 16 every atom, while
//     each copy writes the full 128. So the copy is 8x wider than what is read.
//   * tCsS's strides are mostly 0 (((_0,(_0,_0,_8,_0)),_64,_0,_128)), so the smem side is largely a broadcast
//     rather than 128 distinct loads -- the register footprint is the real cost, not the smem traffic.
//
// TWO THINGS TO MEASURE ON THE BOX before optimising, because register pressure is exactly the kind of thing I have
// no business predicting from a layout dump:
//   1. the actual regs/thread and occupancy for ScaleOnly vs ScaleZero at TK=128 vs TK=256
//   2. whether shrinking tCrS to the 16 elements the transform uses changes anything, or the compiler already
//      eliminates the rest
//
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
  printf("     make_fragment_like(tCrS)  size=%d half = %d regs/thread   (x2 with zero = %d)\n",
         int(size(ownS)), int(size(ownS))/2, int(size(ownS)));
  auto view = thrS.retile_D(ownS);
  printf("     tCrS_copy_view  "); print(view.layout()); printf("  size=%d\n", int(size(view)));
  printf("     transform slice tCrS(_,_,0) size = %d ;  copy dst view(_,_,0) size = %d\n\n",
         int(size(fragS(_,_,0))), int(size(view(_,_,0))));
}
int main(){ printf("scale path shapes\n\n");
  go<128,128,4,4>("int1 TK128 gs32"); go<64,64,2,2>("int2 TK64 gs32"); go<64,256,8,2>("int1 TK256 gs32"); return 0; }
