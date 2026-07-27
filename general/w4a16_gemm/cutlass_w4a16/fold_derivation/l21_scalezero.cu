// L21 -- model the scale/zero path: is it correct, and where does the gs=16 cost come from?
//
// Nothing had ever verified this path, and it is where the gs=16 penalty lives (7-14 points), so "how would we even
// know what to optimise" was the honest state.
//
// HOW THE COLLECTIVE DOES IT (ppu_mma_aiu_fold.hpp):
//     tCrS = make_fragment_like<ElementScale>(thr_mma.partition_fragment_B(sS(_,_,0)))
// -- the scale register layout is partitioned LIKE B, over a tensor of shape (ScaleTileShape.N, 1) -- and it is
// applied elementwise:
//     transform(tCrB_mma(_,_,atom), tCrS(_,_,0), tCrB_mma(_,_,atom), multiplies{})   (+ plus{} for zero)
//
// So slot (mma_val, mma_n) of B is scaled by slot (mma_val, mma_n) of tCrS. CORRECTNESS therefore reduces to one
// question cute can answer: does the scale slot carry the same n as the B slot it multiplies? That is Q1.
//
// COST (Q2) comes out of the FINE branch:
//     APG_ = MMA_KA_ / Scale_TileK = (TK/16) / (TK/gs) = gs/16      atoms per scale group
//     reload when atom_idx % APG_ == 0                              -> MMA_KA_/APG_ = TK/gs = SK reloads per k-iter
//     with zero, TWO copies per reload
// which is the SK story, now read off the code rather than inferred from two measurements.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l21_scalezero.cu -o l21 && ./l21
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <set>
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

// Q1: does the scale that lands in a thread's register carry the same n as the B slot it multiplies?
//
// MY FIRST ATTEMPT WAS WRONG and nearly reported a bug. I used partition_B on the (TN,1) scale tensor to get its
// coordinates -- but partition_B computes ADDRESSES, and with a K extent of 1 against an mma whose K tile is TK it
// indexes past the tensor. The values it read (4164, -1492614480) were out-of-bounds host memory, not scales.
//
// The collective never does that. It uses partition_fragment_B only to borrow the SHAPE for register allocation,
// and the values arrive through the copy:
//     smem_tiled_copy_S = make_tiled_copy_B(SmemCopyAtomScale{}, tiled_mma)   // DefaultCopy
//     tCsS              = smem_thr_copy_S.partition_S(sS)                     // (CPY, CPY_N, CPY_K, sk)
//     tCrS_copy_view    = smem_thr_copy_S.retile_D(tCrS)
//     copy(smem_tiled_copy_S, tCsS(_,_,0,sk), tCrS_copy_view(_,_,0))
// so the n a register ends up with comes from partition_S, and the correspondence to fragment slots is retile_D.
template <int TM, int TN, int TK, int WM, int WN, int SK>
static bool q1(const char* tag) {
  constexpr int WOM = TM/WM, WON = TN/WN;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  using CopyS = Copy_Atom<DefaultCopy, cutlass::half_t>;
  auto tiled_S = make_tiled_copy_B(CopyS{}, Mma{});

  // sS as the collective builds it: tile_to_shape(Layout<Shape<_8,_1>>, (TN, 1, SK)); values encode n
  std::vector<int> sval((size_t)TN * SK);
  auto sS_layout = tile_to_shape(Layout<Shape<_8,_1>>{}, make_shape(Int<TN>{}, Int<1>{}, Int<SK>{}));
  for (int g = 0; g < SK; ++g) for (int n = 0; n < TN; ++n) sval[sS_layout(make_coord(n, 0, g))] = n;
  auto sS = make_tensor(sval.data(), sS_layout);

  std::vector<int> bval((size_t)TN*TK);
  for (int n = 0; n < TN; ++n) for (int k = 0; k < TK; ++k) bval[(size_t)n*TK + k] = n;
  auto tB = make_tensor(bval.data(), make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));

  long bad = 0, tot = 0; int shown = 0, ns = 0, nb = 0;
  for (int t = 0; t < 32*WOM*WON; ++t) {
    auto thrS = tiled_S.get_thread_slice(t);
    auto tCsS = thrS.partition_S(sS);                       // (CPY, CPY_N, CPY_K, sk)
    auto pB   = Mma{}.get_thread_slice(t).partition_B(tB);   // (MMA, MMA_N, MMA_K)
    ns = int(size(tCsS(_, _, 0, 0)));
    nb = int(size<0>(pB)) * int(size<1>(pB));
    if (ns != nb) { printf("  %-20s SHAPE: scale copy delivers %d, B slice wants %d -- transform ill-formed\n",
                           tag, ns, nb); return false; }
    auto sl = tCsS(_, _, 0, 0);
    for (int i = 0; i < ns; ++i) {
      const int nS = sl(i), nB = pB(i);
      ++tot;
      if (nS != nB) { ++bad;
        if (shown < 4) { printf("      MISMATCH thr%d slot %d: B wants n=%d, scale carries n=%d\n",
                                t, i, nB, nS); ++shown; } }
    }
  }
  printf("  %-20s (%d,%d,%d) w%dx%d SK=%d | scale copy %d elems, B slice %d | n agrees %ld/%ld %s\n",
         tag, TM, TN, TK, WM, WN, SK, ns, nb, tot - bad, tot,
         bad ? "<-- WRONG SCALE APPLIED" : "<-- correct by construction");
  return bad == 0;
}

int main() {
  printf("L21 -- the scale/zero path\n\n");
  printf("  == Q1 CORRECTNESS: the scale slot must carry the same n as the B slot it multiplies\n");
  bool ok = true;
  ok &= q1<32, 128, 128, 32, 32, 4>("int1 TK128 gs32");
  ok &= q1<32, 128, 128, 32, 32, 8>("int1 TK128 gs16");
  ok &= q1<64,  64,  64, 32, 32, 2>("int2 TK64 gs32");
  ok &= q1<64,  64,  64, 32, 32, 4>("int4 TK64 gs16");
  ok &= q1<64,  64, 256, 32, 32, 8>("int1 TK256 gs32");

  printf("\n  == Q2 COST: reloads and transforms per k-iteration, from the FINE branch\n");
  printf("      APG = gs/16 atoms per group;  reload at atom_idx %% APG == 0;  MMA_K = TK/16 atoms\n\n");
  printf("      %-6s %-4s %-5s %-5s %-6s %-9s %-9s %-11s %s\n",
         "width", "TK", "gs", "SK", "MMA_K", "APG", "reloads", "copies", "transforms");
  for (int bits : {1, 2, 4})
    for (int tk : {64, 128, 256}) {
      if (tk * bits < 128) continue;                 // below the delivery bound (fold_traits)
      for (int gs : {128, 32, 16}) {
        const int mma_k = tk / 16, sk = tk / gs, apg = gs / 16;
        const bool fine = sk > 1;
        const int reloads = fine ? (mma_k / (apg > 0 ? apg : 1)) : 1;
        printf("      int%-3d %-4d %-5d %-5d %-6d %-9d %-9d %-11s %s\n", bits, tk, gs, sk, mma_k,
               fine ? apg : 0, reloads,
               fine ? "S:1 SZ:2 each" : "once/group",
               fine ? "1 or 2 per atom" : "1 or 2 per atom");
      }
    }
  printf("\n      so per k-iteration the scale smem->reg traffic is  SK * (1 or 2) copies, SK = TK/gs.\n");
  printf("      int1 is pinned at TK>=128 by the delivery bound, so at gs=16 it carries SK=8 where int2/int4\n");
  printf("      at TK=64 carry SK=4 -- and with zero that is 16 copies against 8. That is the whole mechanism\n");
  printf("      behind the measured gs=16 penalty, and it is now read off the code rather than inferred.\n");

  printf("\n  == what an optimisation would have to change\n");
  printf("      sS and sZ are SEPARATE smem tensors with the same layout, and the FINE branch issues two copies\n");
  printf("      from the same group index sk. Interleaving them into one tensor with a trailing extent-2 mode\n");
  printf("      would make it ONE copy of twice the width, halving the ScaleZero traffic. That is a change to\n");
  printf("      SmemLayoutAtomScale plus the gmem->smem staging, and it does not touch B at all.\n");
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
