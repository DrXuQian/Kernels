// L43 -- rank(partition_S(gB)) per plane, with the REAL AIU copy atom.
//
// I claimed this "needs the AIU atom to model" and stopped. Wrong, and not just as phrasing: partition_S is PURE
// LAYOUT ALGEBRA -- make_tiled_copy(atom, thr_layout, val_layout) uses the atom only for its ThrID and element type,
// no PTX -- so including cute/arch/copy_ppu0010_aiu.hpp + its traits answers it locally. Stopping there meant shipping
// a fix (slice_last) without knowing whether the ranks actually differ; if they do not, the fix is a no-op and the box
// fails again on something else.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l43_partition_rank.cu -o l43 && ./l43
#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
using namespace cute;

template <class Elem, int BlockMN, int ContElem, int TM, int TN, int TK, int F>
static void probe(const char* tag, int N, int K) {
  constexpr int bits_per_aiu = BlockMN * ContElem * sizeof_bits<Elem>::value;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Elem, false>;
  using GTC = decltype(make_tiled_copy(Copy_Atom<CopyInst, Elem>{},
                                       Layout<Shape<_1,_1>, Stride<_1,_1>>{},
                                       Layout<Shape<Int<BlockMN>, Int<ContElem>>>{}));
  constexpr int kCon = 256;
  const int Nphys = N / F, Kphys = K * F;
  auto mB = make_counting_tensor(make_layout(
      make_shape(Nphys, make_shape(Int<kCon>{}, Kphys / kCon)),
      make_stride(ScaledBasis<_1,1>{}, make_stride(ScaledBasis<_1,0>{}, ScaledBasis<int,1>{Nphys}))));
  auto gB = local_tile(mB, make_shape(Int<TM>{}, Int<TN/F>{}, Int<F*TK>{}), make_coord(0,0,_), Step<X,_1,_1>{});
  auto t  = GTC{}.get_thread_slice(0).partition_S(gB);
  printf("  %-22s F=%d tile=(%d,%d)  gB rank=%d  partition_S rank=%d  k-mode=",
         tag, F, TN/F, F*TK, int(rank(gB.layout())), int(rank(t.layout())));
  print(shape<3>(t.layout()));
  // THE EXPRESSION THAT FAILED ON THE BOX. Slice with THIS plane's own k coordinate, which is what the fix does;
  // the bug was feeding plane 1's coordinate (a tuple, because its k mode comes out nested) to plane 2 (scalar).
  auto crd = cute::idx2crd(0, cute::shape<2>(gB));
  auto sl  = t(_,_,_,crd);
  printf("   sliced OK, rank=%d\n", int(rank(sl.layout())));
}

int main() {
  printf("real-weight shape N=256 K=256, config (64,64,128) w32x32 -- the one that failed at 2plane.hpp:808\n");
  probe<cutlass::uint2b_t, 64, 128, 64, 64, 128, 1>("plane1 int2 F=1", 256, 256);
  probe<cutlass::uint1b_t, 32, 256, 64, 64, 128, 2>("plane2 int1 F=2", 256, 256);
  // A cross-check block used to sit here that rebuilt the k shape with FULLY STATIC extents (Int<Kphys/kCon>).
  // It coalesced both planes to _2 and reported them identical -- contradicting the lines above and hiding the very
  // difference it was written to show. The real code's K/kCon is DYNAMIC, and that is what makes plane 1's k mode a
  // nested (_2, 1) while plane 2's folded tiler consumes kCon exactly and leaves a scalar. Deleted rather than fixed:
  // the partition_S lines above already carry the fact, from the real shapes.

  printf("\nthe shipping TK=256 config (64,64,256), both planes F=1 -- must come out EQUAL\n");
  probe<cutlass::uint2b_t, 64, 256, 64, 64, 256, 1>("plane1 int2 F=1", 256, 256);
  probe<cutlass::uint1b_t, 64, 256, 64, 64, 256, 1>("plane2 int1 F=1", 256, 256);
  return 0;
}
