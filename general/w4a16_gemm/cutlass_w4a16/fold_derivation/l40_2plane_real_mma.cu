// L40 -- the 2-plane fragment layouts derived LOCALLY from the REAL types, not a stub. Supersedes l39's guess.
//
// I said the emission index space "cannot be settled offline" because tiled_mma carries PermutationM/N and tCrB_load
// goes through the int8 atom. That was wrong, and the reason it was wrong is instructive: PermutationM/N are DERIVED
// from tiled_mma (not extra inputs), the int8 atom's traits are in the actlize headers this harness already includes,
// and the ONE thing my l39 stub got wrong was a value I could have read from the builder:
//
//   ppu_mma_builder.inl:588   MmaPermK = FoldF > 0 ? TileShape.K : (32*8/bits)
//
// The FOLD path permutes K by TileShape.K; every OTHER path (including the 2-plane) permutes by the 32 B swzl run
// span -- 128 for int2, 256 for int1, 64 for int4. l39 used Int<TK> for both, so its 2-plane fragment was a fiction.
// That single constant is the whole difference between the fold path's layout and this one.
//
// Everything below is the builder's and the collective's own type algebra, copied by REFERENCE not by hand:
//   tiled_mma     = TiledMMA<MMA_Atom<GetMmaInst<half,half,float>>, Layout<WarpOnM,WarpOnN,1>,
//                            Tile<blockM/warpM*InstM, blockN/warpN*InstN, MmaPermK>>      (builder, 249-254)
//   TiledMma_S8   = TiledMMA<MMA_Atom<..._S32S8S8S32_TN>,          same warp layout,
//                            Tile<PermutationM, PermutationN, _32>>                        (2plane collective, 586)
//   tCrB_mma      = thr_mma.partition_fragment_B(sB(_,_,0))          sB   : (TN, Block_K)  quantized element
//   tCrB_load     = thr_mma_s8.partition_fragment_B(sB_s8(_,_,0))    sB_s8: (TN, TN_bytes) int8 view
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l40_2plane_real_mma.cu -o l40 && ./l40
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <vector>
using namespace cute;

template <int Bits> struct QElem;
template <> struct QElem<1> { using type = cutlass::uint1b_t; };
template <> struct QElem<2> { using type = cutlass::uint2b_t; };
template <> struct QElem<4> { using type = cutlass::int4b_t;  };

// Bits = the plane's width; Fold = the fold path's MmaPermK rule (TileShape.K) instead of the 32B-run one.
template <int TM, int TN, int TK, int WM, int WN, int Bits, bool Fold = false>
static void probe(const char* tag) {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  using WarpOnM = Int<TM / warpM>; using WarpOnN = Int<TN / warpN>;
  constexpr int PermM = TM / warpM * InstM, PermN = TN / warpN * InstN;
  constexpr int MmaPermK = Fold ? TK : (32 * 8 / Bits);
  using WarpLayout = Layout<Shape<WarpOnM, WarpOnN, _1>>;

  using Mma   = TiledMMA<MMA_Atom<FInst>, WarpLayout, Tile<Int<PermM>, Int<PermN>, Int<MmaPermK>>>;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, WarpLayout, Tile<Int<PermM>, Int<PermN>, _32>>;

  using QE = typename QElem<Bits>::type;
  auto sB    = make_tensor(make_smem_ptr((QE*)nullptr),
                           make_layout(Shape<Int<TN>, Int<TK>>{}, Stride<Int<TK>, _1>{}));
  auto sB_s8 = recast<int8_t>(sB);                                    // (TN, TK*Bits/8)
  auto frag  = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  auto load  = MmaS8{}.get_thread_slice(0).partition_fragment_B(sB_s8);

  printf("\n%s  (%d,%d,%d) w%dx%d int%d %s\n", tag, TM, TN, TK, WM, WN, Bits, Fold ? "[FOLD MmaPermK=TK]" : "");
  printf("   MmaPermK=%d  PermM=%d PermN=%d  WarpOn=(%d,%d)\n",
         MmaPermK, PermM, PermN, int(WarpOnM{}), int(WarpOnN{}));
  printf("   tCrB_mma  : "); print(frag.layout());
  printf("   size=%d\n", int(size(frag.layout())));
  printf("   tCrB_load : "); print(load.layout());
  printf("   size=%d (int8)  CPY_K=%d\n", int(size(load.layout())), int(size<2>(load.layout())));
  auto cvt_in = recast<QE>(load(_, _, Int<0>{}));
  printf("   cvt_in    : "); print(cvt_in.layout());
  printf("   size=%d codes  mode0=%d mode1=%d stride1=%d\n", int(size(cvt_in.layout())),
         int(size<0>(cvt_in.layout())), int(size<1>(cvt_in.layout())), int(stride<1>(cvt_in.layout())));

  const int MMA_N = int(size<1>(frag.layout())), MMA_K = int(size<2>(frag.layout()));
  const int APC = MMA_K / int(size<2>(load.layout()));                 // K_ATOM_PER_COPY
  printf("   MMA_N=%d MMA_K=%d K_ATOM_PER_COPY=%d   n-stride=%d (8*MMA_K=%d, 8*MMA_N=%d)\n",
         MMA_N, MMA_K, APC, int(stride<1>(frag.layout())), 8 * MMA_K, 8 * MMA_N);

  // THE QUESTION: cvt_out aliases tCrB_mma's registers THROUGH cvt_in's layout, based at the copy step's first atom.
  // So for step s and chunk ii, the converter writes offsets base(s) + stride1*ii + [0, mode0). Does the union over
  // (s, ii) hit each of tCrB_mma's slots EXACTLY ONCE? If yes the flat space is sound and a chunk gate can be built
  // on it; if no, my reading of the plumbing is wrong (Q3 numerics say the code is right, so that would be on me).
  const int kOut = int(size<0>(cvt_in.layout())), NumIter = int(size<1>(cvt_in.layout()));
  const int stride1 = int(stride<1>(cvt_in.layout())), FSZ = int(size(frag.layout()));
  std::vector<int> hits(FSZ, 0);
  bool oob = false;
  for (int s = 0; s < int(size<2>(load.layout())); ++s) {
    const int base = int(frag.layout()(0, 0, s * APC));
    for (int ii = 0; ii < NumIter; ++ii)
      for (int e = 0; e < kOut; ++e) {
        const int off = base + stride1 * ii + e;
        if (off < 0 || off >= FSZ) { oob = true; continue; }
        ++hits[off];
      }
  }
  int once = 0, zero = 0, many = 0;
  for (int h : hits) { if (h == 1) ++once; else if (h == 0) ++zero; else ++many; }
  printf("   flat-write coverage of tCrB_mma's %d slots: once=%d  never=%d  twice+=%d%s  -> %s\n",
         FSZ, once, zero, many, oob ? "  (some writes OUT OF RANGE)" : "",
         (once == FSZ && !oob) ? "EXACT: flat index space is sound"
                               : "NOT a partition -- the flat space is not tCrB_mma's");
}

int main() {
  printf("== FOLD path (MmaPermK = TileShape.K): the configs the chunked emitter was validated on ==\n");
  probe<64,128, 64,64,64,1,true>("fold BEST");
  probe<32,128,128,32,32,1,true>("fold Q3/Q5 TK=128");
  printf("\n== NON-FOLD path (MmaPermK = 32B run span): what the 2-plane actually gets ==\n");
  probe<64, 64,256,32,32,2>("2plane LOW  int2");
  probe<64, 64,256,32,32,1>("2plane HIGH int1");
  probe<32,128,256,32,32,2>("2plane LOW  int2 (32,128,256)");
  printf("\n== the same shape under BOTH MmaPermK rules, to isolate that one constant ==\n");
  probe<64, 64,256,32,32,2,true>("int2 TK=256 with the FOLD rule");
  return 0;
}
