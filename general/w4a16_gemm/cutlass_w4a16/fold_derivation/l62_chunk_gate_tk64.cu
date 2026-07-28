// L62 -- what task #12 (chunked B on the 2-plane) actually needs at Block_K=64, measured rather than assumed.
//
// Block_K=64 is now the fastest B-concat configuration on the box: 377.62 us / 36.4% MFU at (64,128,64) w64x64 s2,
// against 473.11 at Block_K=128 and 822.60 at 256. The remaining gap is large -- the SAME geometry runs 63.7% MFU as a
// single-plane int1 GEMM -- and what took that config from 50.2% to 63.7% was the chunked B conversion. Its
// prerequisite holds here: cvt/mma = 128/WM = 2 and WM = 64 >= 32, so the "worth nothing if WM == 16" exclusion in the
// handoff does not apply.
//
// But the 2-plane collective guards chunking with
//     static_assert(!kBChunk || K_ATOM_PER_COPY == MixGemm2Plane_uint2_uint1<>::kAtoms)     // kAtoms == 8
// and K_ATOM_PER_COPY = MMA_K / CPY_K, which was 8 only because the 2-plane path was pinned to TileShape.K = 256
// (MMA_K 16, CPY_K 2). At smaller Block_K, MMA_K shrinks.
//
// THE INVARIANT IS PROBABLY NOT "K_ATOM_PER_COPY == 8". One int2 delivery is 16 B/thread = 64 codes, and the converter
// emits 64 fp16 into a flat window; whether tCrB_mma reads that window as 1 n-atom x 8 k-atoms (TK=256, MMA_N=2) or
// 2 n-atoms x 4 k-atoms (TK=64, MMA_N=4) is a property of its LAYOUT, not of the emission. If the product
// MMA_N_per_delivery * K_ATOM_PER_COPY is 8 in both, the gate is over-strict and #12 becomes a relaxation rather than a
// redesign. Printed here for all three Block_K so the change is derived from the real atoms.
//
//   nvcc -std=c++17 -Istub_inc -I.. -I../../../../third_party/actlize/include l62_chunk_gate_tk64.cu -o l62 && ./l62
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
using namespace cute;

template <int Bits, int BlockK> struct AiuOp {
  static constexpr int BlockCont = BlockK * Bits / 8;
  static constexpr int AiuByte   = BlockCont > 128 ? 128 : BlockCont;
  static constexpr int AiuElem   = AiuByte * 8 / Bits;
  static constexpr int InstNum   = BlockK / AiuElem;
};

template <int TM, int TN, int TK, int WM, int WN, int F1, int F2>
static void probe(const char* tag) {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int warpM = (WM > 16) ? WM : 16, warpN = (WN > 16) ? WN : 16;
  constexpr int WOM = TM / warpM, WON = TN / warpN;
  constexpr int MmaPermK = (F1 > 1) ? TK : (32 * 8 / 2);
  using Mma   = TiledMMA<MMA_Atom<FInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>, Tile<Int<WOM*16>, Int<WON*16>, Int<MmaPermK>>>;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>, Tile<Int<WOM*16>, Int<WON*16>, _32>>;

  // tCrB_mma comes from the fold-in-N LOGICAL view, exactly SmemLayoutB_MmaView
  auto mma_view = make_layout(make_shape (make_shape(Int<F1>{}, Int<TN/F1>{}), Int<TK>{}),
                             make_stride(make_stride(Int<TK>{}, Int<F1*TK>{}), _1{}));
  auto sB       = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), mma_view);
  auto tCrB_mma = Mma{}.get_thread_slice(0).partition_fragment_B(sB);

  // tCrB_load / tCrB_copy_view come from the PHYSICAL int8 tensor through the int8 atom
  constexpr int BK1 = F1 * TK, Ng1 = TN / F1, RowB1 = BK1 * 2 / 8;
  using OB1 = AiuOp<2, BK1>;
  using AtomB1 = Copy_Atom<PPU0010_TSM_LD_SWZL<int8_t, Ng1, OB1::AiuElem / 4, true, false, OB1::InstNum>, int8_t>;
  auto sB8      = make_tensor(make_smem_ptr((int8_t*)nullptr),
                              make_layout(make_shape(Int<Ng1>{}, Int<RowB1>{}), make_stride(Int<RowB1>{}, _1{})));
  auto tCrB_load = MmaS8{}.get_thread_slice(0).partition_fragment_B(sB8);
  auto copy_view = make_tiled_copy_B(AtomB1{}, MmaS8{}).get_thread_slice(0).retile_D(tCrB_load);

  const int MMA_N = int(size<1>(tCrB_mma)), MMA_K = int(size<2>(tCrB_mma));
  const int CPY_N = int(size<1>(copy_view)), CPY_K = int(size<2>(copy_view));
  const int KAPC  = MMA_K / CPY_K;                       // the quantity the assert tests
  const int NAPC  = MMA_N / CPY_N;                       // n-atoms per copy step -- the assert ignores this
  const int one   = int(size(tCrB_mma(_, _, Int<0>{}))); // tCrB_one: 8 * MMA_N fp16
  const int kAtoms = cutlass::MixGemm2Plane_uint2_uint1<>::kAtoms;

  printf("  %-34s MMA_N=%2d MMA_K=%2d | CPY_N=%2d CPY_K=%2d | K_ATOM_PER_COPY=%d N_ATOM_PER_COPY=%d"
         "  product=%2d  tCrB_one=%2d\n",
         tag, MMA_N, MMA_K, CPY_N, CPY_K, KAPC, NAPC, KAPC * NAPC, one);
  printf("  %-34s assert as written (KAPC == %d): %s     product == %d: %s\n", "",
         kAtoms, KAPC == kAtoms ? "PASSES" : "FIRES  <-- blocks #12 here",
         kAtoms, KAPC * NAPC == kAtoms ? "holds -- the gate is over-strict, relax it to the product"
                                       : "ALSO fails -- #12 needs real work at this shape");
}

int main() {
  printf("L62 -- the 2-plane chunk gate across Block_K (kAtoms = %d)\n\n",
         cutlass::MixGemm2Plane_uint2_uint1<>::kAtoms);
  probe<64,  64, 256, 32, 32, 1, 1>("TK=256 (64, 64,256) w32x32   original");
  probe<32, 128, 256, 32, 32, 1, 1>("TK=256 (32,128,256) w32x32   original");
  probe<64,  64, 128, 32, 32, 1, 2>("TK=128 (64, 64,128) w32x32   F1=1 F2=2");
  probe<64, 128, 128, 32, 64, 1, 2>("TK=128 (64,128,128) w32x64   F1=1 F2=2");
  probe<64, 128,  64, 64, 64, 2, 4>("TK= 64 (64,128, 64) w64x64   THE WINNER");
  probe<128,128,  64, 64, 64, 2, 4>("TK= 64 (128,128,64) w64x64   F1=2 F2=4");
  probe<64, 256,  64, 64, 64, 2, 4>("TK= 64 (64,256, 64) w64x64   F1=2 F2=4");
  return 0;
}
