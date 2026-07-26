// L5, part 1 -- the piece L2/L3/L4 could not supply: when a fragment needs several copy-atom instances, how many
// are there and along which axis? Answered by instantiating the collective's OWN types, not by reading them.
//
// Mirrors ppu_mma_aiu_fold.hpp exactly:
//     BFoldBlockN = TN / FoldF      BFoldBlockK = FoldF * TK
//     SmemCopyAtomB = Copy_Atom<PPU0010_TSM_LD_SWZL<ElemB, BFoldBlockN, AiuContElemSize, true, false, InstNum>>
//     TiledMma_S8   = TiledMMA<s8 atom, Layout<Shape<warpOnM,warpOnN,_1>>, Tile<PermM,PermN,_32>>
//     tCrB_load     = thr_mma_s8.partition_fragment_B(sB_s8(_,_,0))      sB_s8 = (BFoldBlockN, BFoldBlockK*bits/8)
//     tCrB_copy_view= smem_thr_copy_B.retile_D(tCrB_load)                -> (CPY, CPY_N, CPY_K)
//     K_BLOCK_MAX     = size<2>(tCrB_copy_view)
//     K_ATOM_PER_COPY = size<2>(tCrB_mma) / K_BLOCK_MAX
//
// CPY_N indexes copy instances along N (a different coord_h), CPY_K along K (a different slice / coord_w).
// The converter is called CPY_VEC at a time over cvt_in = recast<ElemB>(tCrB_load(_,_,k_block)), and that tensor
// is compact, so converter call c consumes codes [c*CPY_VEC, (c+1)*CPY_VEC) -- i.e. it walks CPY_N first.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l5_copy_instances.cu -o l5 && ./l5
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
using namespace cute;

// the s8 mma atom's traits, as the collective's TiledMma_S8 uses them
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

template <class ElemB, int TM, int TN, int TK, int WM, int WN>
void probe(const char* tag) {
  constexpr int Bits = sizeof_bits<ElemB>::value;
  constexpr int contig = TK * Bits / 8;
  constexpr int FoldF  = contig >= 32 ? 1 : 32 / contig;
  constexpr int BFoldN = TN / FoldF, BFoldK = FoldF * TK;
  // MixGemm_AIU_Operand, verbatim
  constexpr int BlockContSize   = BFoldK * Bits / 8;
  constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  constexpr int AiuContElem     = AiuContByteSize * 8 / Bits;
  constexpr int InstNum         = BFoldK / AiuContElem;

  using SmemCopyOp   = PPU0010_TSM_LD_SWZL<ElemB, BFoldN, AiuContElem, true, false, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, ElemB>;

  constexpr int warpOnM = TM / WM, warpOnN = TN / WN;
  using F16Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<warpOnM>,Int<warpOnN>,_1>>,
                          Tile<Int<WM>, Int<WN>, Int<TK>>>;
  using PermN  = decltype(F16Mma{}.template permutation_mnk<1>());
  using PermM  = decltype(F16Mma{}.template permutation_mnk<0>());
  using S8Mma  = TiledMMA<MMA_Atom<S8Atom>, Layout<Shape<Int<warpOnM>,Int<warpOnN>,_1>>,
                          Tile<PermM, PermN, _32>>;

  // tCrB_mma comes from the LOGICAL (TN, TK) view
  auto tCrB_mma = F16Mma{}.get_thread_slice(0).partition_fragment_B(
      make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), make_layout(Shape<Int<TN>,Int<TK>>{})));
  // tCrB_load comes from the PHYSICAL (BFoldN, BFoldK*bits/8) int8 view
  constexpr int BPR = BFoldK * Bits / 8;
  auto sB_s8    = make_tensor(make_smem_ptr((int8_t*)nullptr), make_layout(Shape<Int<BFoldN>,Int<BPR>>{}));
  auto tCrB_load = S8Mma{}.get_thread_slice(0).partition_fragment_B(sB_s8);

  auto tiled_copy = make_tiled_copy_B(SmemCopyAtom{}, S8Mma{});
  auto thr_copy   = tiled_copy.get_thread_slice(0);
  auto cpy_view   = thr_copy.retile_D(tCrB_load);

  const int KBM   = int(size<2>(cpy_view));
  const int KAPC  = int(size<2>(tCrB_mma)) / (KBM > 0 ? KBM : 1);
  const int CPYV  = 4 * 32 / Bits;                     // CPY_VEC in the collective
  const int codes = int(size(tCrB_load)) * 8 / Bits;   // codes in the whole load fragment

  printf("  %-22s int%d (%d,%d,%d) warp %dx%d  FoldF=%d  swzl CUBE<%d,%d> InstNum=%d\n",
         tag, Bits, TM, TN, TK, WM, WN, FoldF, BFoldN, AiuContElem, InstNum);
  printf("      tCrB_mma  (MMA=%d, MMA_N=%d, MMA_K=%d)   tCrB_load = %d int8 = %d codes\n",
         int(size<0>(tCrB_mma)), int(size<1>(tCrB_mma)), int(size<2>(tCrB_mma)), int(size(tCrB_load)), codes);
  printf("      cpy_view  (CPY=%d, CPY_N=%d, CPY_K=%d)  -> K_BLOCK_MAX=%d  K_ATOM_PER_COPY=%d\n",
         int(size<0>(cpy_view)), int(size<1>(cpy_view)), int(size<2>(cpy_view)), KBM, KAPC);
  printf("      converter: CPY_VEC=%d, so %d call(s) per k_block; instances per call along CPY_N\n\n",
         CPYV, codes / (KBM > 0 ? KBM : 1) / CPYV);
}

int main() {
  printf("L5/1 -- copy-atom instance structure, from the collective's own types\n\n");
  probe<cutlass::uint1b_t, 32, 128, 128, 32, 32>("int1 F=2 winner");
  probe<cutlass::uint1b_t, 64, 128,  64, 32, 32>("int1 F=4 target");
  probe<cutlass::uint2b_t, 64,  64,  64, 32, 32>("int2 F=2");
  probe<cutlass::int4b_t,  64,  64,  64, 32, 32>("int4 F=1");
  return 0;
}
