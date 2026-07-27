// L9 -- MEASURE cvt_in's layout, which is what convert_tensor's chunk offsets come from.
// convert_tensor slices in(_, ii) / out(_, ii) -- MODE 1 -- and writes ConversionVectorWidth CONTIGUOUS elements
// from that pointer. out shares in's layout. So converter chunk ii writes flat offsets [L(0,ii), +VEC), where
// L = cvt_in.layout(). Everything about pi is in that one layout, and I had been deriving it instead of asking.
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
using namespace cute;
struct S8Atom {};
namespace cute {
template <> struct MMA_Traits<S8Atom> {
  using ValTypeD=int32_t; using ValTypeA=int8_t; using ValTypeB=int8_t; using ValTypeC=int32_t;
  using Shape_MNK=Shape<_16,_16,_32>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2,_2>>,Stride<Stride<_64,_1>,Stride<_16,_256,_8>>>;
  using BLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2,_2>>,Stride<Stride<_64,_1>,Stride<_16,_256,_8>>>;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
}
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
template<int Bits,int TN,int TK,int WM,int WN> void probe(const char* tag){
  constexpr int WOM=32/WM, WON=TN/WN;
  constexpr int contig=TK*Bits/8, F=contig>=32?1:32/contig, Ng=TN/F, BPR=F*TK*Bits/8;
  using F16=TiledMMA<MMA_Atom<F16Atom>,Layout<Shape<Int<WOM>,Int<WON>,_1>>,Tile<Int<WOM*16>,Int<WON*16>,Int<TK>>>;
  using S8 =TiledMMA<MMA_Atom<S8Atom >,Layout<Shape<Int<WOM>,Int<WON>,_1>>,Tile<Int<WOM*16>,Int<WON*16>,_32>>;
  auto mma = F16{}.get_thread_slice(0).partition_fragment_B(
      make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), make_layout(Shape<Int<TN>,Int<TK>>{},Stride<Int<TK>,_1>{})));
  auto ld = S8{}.get_thread_slice(0).partition_fragment_B(
      make_tensor(make_smem_ptr((int8_t*)nullptr), make_layout(Shape<Int<Ng>,Int<BPR>>{},Stride<Int<BPR>,_1>{})));
  // MUST recast to the ACTUAL element type. This line used to hardcode uint1b_t, which made the int2/int4 rows
  // report bit counts instead of code counts (chunk offsets 0,128,...,896 for a 64-element fragment -- an
  // impossible answer that I very nearly read as an out-of-bounds bug in the collective).
  using EB = cute::conditional_t<Bits==1, cutlass::uint1b_t,
             cute::conditional_t<Bits==2, cutlass::uint2b_t, cutlass::int4b_t>>;
  auto in = recast<EB>(ld(_,_,0));
  printf("  %-16s int%d TN=%3d TK=%3d w%dx%d | F=%d Ng=%3d BPR=%2d\n", tag,Bits,TN,TK,WM,WN,F,Ng,BPR);
  printf("      tCrB_mma  "); print(mma.layout()); printf("   size=%d\n", int(size(mma)));
  printf("      tCrB_load "); print(ld.layout());  printf("   size=%d int8\n", int(size(ld)));
  printf("      cvt_in    "); print(in.layout());  printf("   size=%d codes\n", int(size(in)));
  const int VEC = 4*32/Bits, N = int(size(in));
  printf("      CPY_VEC=%d -> NumIterations=%d ; chunk offsets L(0,ii): ", VEC, N/VEC);
  for (int ii=0; ii<N/VEC; ++ii) printf("%d ", int(in.layout()(make_coord(0, ii))));
  printf("\n\n");
}
int main(){
  printf("L9 -- cvt_in's layout, i.e. where convert_tensor actually writes each chunk\n\n");
  probe<1,128,128,32,32>("calib F=2");
  probe<1,128, 64,32,64>("target F=2");
  probe<2, 64, 64,32,32>("int2 fold F=2");
  printf("  == F=1 configs: these need SEVERAL copy instances, which no earlier model covered\n");
  probe<4, 64, 64,32,32>("int4 F=1 (production)");
  probe<2, 64,128,32,32>("int2 F=1");
  probe<1, 64,256,32,32>("int1 F=1");
  return 0;
}
