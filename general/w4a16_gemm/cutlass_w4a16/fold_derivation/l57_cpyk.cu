// L57 -- the last unmodelled segment: the A operand and the SHARED k_block index.
//
// l56 closed the loop on B and it is clean at every rung, including the failing Block_K=64 -- so the defect is not in
// B's offline / delivery / emission / destination / scale schedule. What the loop never touched is A, and the mainloop
// drives BOTH operands off ONE index:
//     K_BLOCK_MAX = size<2>(tCrB_copy_view);                                    // B's CPY_K
//     copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
// If A's CPY_K exceeds B's, the slots above K_BLOCK_MAX are never loaded and part of A is garbage. A's swzl atom
// parameters move with TK (Block_K stays UNFOLDED for A: "A stays blockK"), so this ratio is exactly the kind of thing
// that can differ between Block_K 128 and 64.
//
// Computed with the REAL atoms rather than hand-derived -- my hand derivation gave A:B = 2:1 at TK=128, which would
// make the WORKING config broken, so it is wrong.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l57_cpyk.cu -o l57 && ./l57
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
using namespace cute;

// MixGemm_AIU_Operand's own arithmetic, from ppu_mma_builder.inl
template <int Bits, int BlockK> struct AiuOp {
  static constexpr int BlockCont = BlockK * Bits / 8;
  static constexpr int AiuByte   = BlockCont > 128 ? 128 : BlockCont;
  static constexpr int AiuElem   = AiuByte * 8 / Bits;
  static constexpr int InstNum   = BlockK / AiuElem;
};

template <int TM,int TN,int TK,int WM,int WN,int F1>
static void probe(const char* tag) {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int warpM = (WM>16)?WM:16, warpN = (WN>16)?WN:16;
  constexpr int WOM = TM/warpM, WON = TN/warpN;
  constexpr int MmaPermK = (F1>1) ? TK : (32*8/2);
  using Mma   = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>,Int<WON>,_1>>, Tile<Int<WOM*16>,Int<WON*16>,Int<MmaPermK>>>;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>,Int<WON>,_1>>, Tile<Int<WOM*16>,Int<WON*16>,_32>>;

  // A: Block_MN = TM, Block_K = TK (UNFOLDED), fp16
  using OA = AiuOp<16, TK>;
  using AtomA = Copy_Atom<PPU0010_TSM_LD_SWZL<cutlass::half_t, TM, OA::AiuElem, true, false, OA::InstNum>, cutlass::half_t>;
  auto sA   = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                          make_layout(Shape<Int<TM>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto tCrA = Mma{}.get_thread_slice(0).partition_fragment_A(sA);
  auto tcA  = make_tiled_copy_A(AtomA{}, Mma{}).get_thread_slice(0);
  auto vA   = tcA.retile_D(tCrA);

  // B plane 1: Block_MN = TN/F1, Block_K = F1*TK, int2 -- but the swzl atom is int8-typed (AiuContElemSize/4 for int2)
  constexpr int BK = F1 * TK;
  using OB = AiuOp<2, BK>;
  constexpr int Ng = TN / F1, RowB = BK * 2 / 8;
  using AtomB = Copy_Atom<PPU0010_TSM_LD_SWZL<int8_t, Ng, OB::AiuElem / 4, true, false, OB::InstNum>, int8_t>;
  auto sB8  = make_tensor(make_smem_ptr((int8_t*)nullptr),
                          make_layout(make_shape(Int<Ng>{}, Int<RowB>{}), make_stride(Int<RowB>{}, _1{})));
  auto tCrB = MmaS8{}.get_thread_slice(0).partition_fragment_B(sB8);
  auto tcB  = make_tiled_copy_B(AtomB{}, MmaS8{}).get_thread_slice(0);
  auto vB   = tcB.retile_D(tCrB);

  const int cpyA = int(size<2>(vA.layout())), cpyB = int(size<2>(vB.layout()));
  printf("  %-30s F1=%d | A: MMA_K=%d AiuElem=%d InstNum=%d CPY_K=%d | B: MMA_K=%d AiuElem=%d InstNum=%d CPY_K=%d  -> %s\n",
         tag, F1, int(size<2>(tCrA.layout())), OA::AiuElem, OA::InstNum, cpyA,
         int(size<2>(tCrB.layout())), OB::AiuElem, OB::InstNum, cpyB,
         cpyA == cpyB ? "equal -- one k_block index serves both"
                      : "DIFFER <-- the shared k_block cannot cover both operands");
}

int main() {
  printf("L57 -- A's CPY_K vs B's, which the mainloop's single k_block index must satisfy\n\n");
  printf("  working configs (box bad=0)\n");
  probe<64, 64,256,32,32,1>("(64, 64,256) w32x32");
  probe<64, 64,128,32,32,1>("(64, 64,128) w32x32");
  probe<32,128,128,32,32,1>("(32,128,128) w32x32");
  printf("\n  the ladder's later rungs\n");
  probe<64,128,128,32,64,1>("(64,128,128) w32x64");
  probe<64,128,128,64,64,1>("(64,128,128) w64x64");
  probe<64,128, 64,64,64,2>("(64,128, 64) w64x64 F1=2");
  return 0;
}
