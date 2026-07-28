// ***** RETRACTED. The criterion below is WRONG, and so is the conclusion it produced ("chunked B conversion is a
// ***** correctness prerequisite at Block_K=64"). What the flat write actually needs is only that it cover tCrB_mma
// ***** BIJECTIVELY, and it does: at Block_K=64 the fragment is ((2,2,2),4,4):((1,2,4),32,8), so ii=0 writes memory
// ***** offsets [0,64) = atoms (n=0..1, k=0..3) and ii=1 writes [64,128) = (n=2..3, k=0..3) -- each exactly once. And
// ***** plane_map's `flat = inst*VEC + e` walks the SAME memory offsets, with pi = right_inverse(frag.layout()) already
// ***** accounting for the folded layout. Requiring cvt_in's mode-1 stride to EQUAL tCrB_mma's MMA_N stride was a
// ***** condition I invented; nothing needs it.
// *****
// ***** Fifth wrong derivation in this line of work, and the same root cause as the other four: a criterion asserted
// ***** rather than gated. The Block_K=64 failure moved THREE variables at once (TN, WN, TK-with-fold), so the next
// ***** step is a numeric LADDER, one variable per rung -- see test_q3_bconcat_real. Kept for the retraction.
//
// L55 -- the side my composition never modelled: where the CONVERTED fp16 is written.
//
// transform_B_kblock does  cvt_out = make_tensor(tCrB_mma(_,_,kb*K_ATOM).data(), cvt_in.layout())  -- it aliases
// tCrB_mma's registers THROUGH the LOAD fragment's layout. That is only valid if cvt_in's mode-1 stride equals
// tCrB_mma's MMA_N stride. Unfolded they coincide. With plane 1 FOLDED, tCrB_mma comes from the fold-in-N LOGICAL view
// (MMA_N = TN/16) while cvt_in comes from the PHYSICAL fragment (mode 1 = Ng/16), so they need not.
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
using namespace cute;

template <int TM,int TN,int TK,int WM,int WN,int F1>
static void probe(const char* tag) {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int warpM = (WM>16)?WM:16, warpN = (WN>16)?WN:16;
  constexpr int WOM = TM/warpM, WON = TN/warpN;
  constexpr int MmaPermK = (F1>1) ? TK : (32*8/2);
  using Mma   = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>,Int<WON>,_1>>, Tile<Int<WOM*16>,Int<WON*16>,Int<MmaPermK>>>;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>,Int<WON>,_1>>, Tile<Int<WOM*16>,Int<WON*16>,_32>>;
  constexpr int Ng = TN/F1, RowB = F1*TK*2/8;
  // tCrB_mma from the fold-in-N LOGICAL view (F1>1) or the plain (TN,TK) one
  auto logical = make_layout(make_shape (make_shape(Int<F1>{}, Int<Ng>{}), Int<TK>{}),
                             make_stride(make_stride(Int<TK>{}, Int<F1*TK>{}), _1{}));
  auto sB   = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), logical);
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  auto s8   = make_tensor(make_smem_ptr((int8_t*)nullptr),
                          make_layout(make_shape(Int<Ng>{}, Int<RowB>{}), make_stride(Int<RowB>{}, _1{})));
  auto load = MmaS8{}.get_thread_slice(0).partition_fragment_B(s8);
  auto cvt  = recast<cutlass::uint2b_t>(load(_,_,Int<0>{}));
  const int mmaN_stride = int(stride<1>(frag.layout()));
  const int cvt1_stride = int(stride<1>(cvt.layout()));
  printf("  %-30s F1=%d | tCrB_mma MMA_N=%d stride=%d | cvt_in mode1=%d stride=%d  -> %s\n",
         tag, F1, int(size<1>(frag.layout())), mmaN_stride,
         int(size<1>(cvt.layout())), cvt1_stride,
         mmaN_stride == cvt1_stride ? "MATCH: the flat write is valid"
                                   : "MISMATCH: the flat write scatters into the wrong registers");
}

int main() {
  printf("L55 -- does the converter's flat write into tCrB_mma hold once plane 1 folds?\n\n");
  probe<64, 64,256,32,32,1>("TK=256 unfolded (control)");
  probe<32,128,128,32,32,1>("TK=128 unfolded (works, bad=0)");
  probe<64, 64,128,32,32,1>("TK=128 unfolded (works, bad=0)");
  probe<64,128, 64,64,64,2>("TK=64  plane 1 FOLDED F1=2");
  return 0;
}
