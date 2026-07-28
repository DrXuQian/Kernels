// L42 -- PER-PLANE N-FOLD for the 2-plane (B-concat) path, derived from the real types before touching the collective.
//
// WHY. B-concat is locked at TK=256 because ONE MOEG_FOLD is computed from plane 1's bit width, and below TK=256 the
// two planes need DIFFERENT fold factors (F = 32/(TK*bits/8)):
//     TK=128 : int2 F=1, int1 F=2      TK=64 : int2 F=2, int1 F=4
// TK=256 is the only TK where they agree (both 1), which is why it is what shipped. TK=256 costs A-smem 32 KB/stage
// (blk=2, 12.5% occupancy) and 128 fp16 registers each for A and B -- measured 16.7% MFU against int4's 53.2%.
//
// WHAT THE FOLD COLLECTIVE ACTUALLY DOES (ppu_mma_aiu_fold.hpp, the "division of views" comment). This is what makes
// per-plane fold plausible rather than a redesign:
//     * the swzl READ (tCrB_load / tCsB) partitions the PHYSICAL (Ng, FoldF*TKe) shape, Ng = TN/FoldF
//     * the MMA fragment partitions a LOGICAL (TileShape.N, TileShape.K) VIEW of the same bytes
// The 2-plane collective already has fully separate load machinery for plane 2 (SmemLayoutAtomB2, SmemCopyAtomB2,
// GmemTiledCopyB2, sB2_s8, its own copy_aiu) and only ONE tCrB_mma. So each plane can partition its own physical
// shape while both converge on one logical fragment -- IF cute agrees, which is what this file asks.
//
// AND ONE THING THAT DEFINITELY CHANGES: the builder picks MmaPermK = FoldF > 0 ? TileShape.K : 32*8/bits
// (ppu_mma_builder.inl:588). So a folded 2-plane runs the OTHER MmaPermK rule, tCrB_mma is a different layout, and
// the chunk gate derived in l41 (at_plain/4, valid at MmaPermK = 32B-run) does NOT automatically carry.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l42_2plane_fold.cu -o l42 && ./l42
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
using namespace cute;

static constexpr int foldF(int bits, int tk) { const int c = tk*bits/8; return c >= 32 ? 1 : 32/c; }

// One plane's PHYSICAL swzl-read partition. Static extents: partition_fragment_B builds an OWNING register tensor and
// cute rejects dynamic owning tensors ("Dynamic owning tensors not supported") -- the shapes must be Int<>.
template <class MmaS8, int Ng, int RowB, int Bits>
static int phys(const char* name) {
  auto s8 = make_tensor(make_smem_ptr((int8_t*)nullptr),
                        make_layout(Shape<Int<Ng>, Int<RowB>>{}, Stride<Int<RowB>, _1>{}));
  auto ld = MmaS8{}.get_thread_slice(0).partition_fragment_B(s8);
  printf("   %s physical (%3d,%3d B) -> tCrB_load : ", name, Ng, RowB); print(ld.layout());
  const int cpyk = int(size<2>(ld.layout()));
  const int per_step = cpyk ? int(size(ld.layout()))*8/Bits/cpyk : 0;
  printf("   CPY_K=%d  codes/thread/step=%d  delivery=%d codes %s\n",
         cpyk, per_step, 128/Bits, per_step >= 128/Bits ? "" : " <-- OVER-DELIVERY");
  return cpyk;
}

// Bits1 = low plane (dense), Bits2 = high plane (sparse). Fold = use the fold path's MmaPermK rule.
template <int TM, int TN, int TK, int WM, int WN, int Bits1, int Bits2, bool Fold>
static void probe(const char* tag) {
  using FInst = PPU0015_16x16x16_F32F16F16F32_TN;
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  using WarpOnM = Int<TM/warpM>; using WarpOnN = Int<TN/warpN>;
  constexpr int PermM = TM/warpM*InstM, PermN = TN/warpN*InstN;
  constexpr int MmaPermK = Fold ? TK : (32*8/Bits1);
  using WarpLayout = Layout<Shape<WarpOnM, WarpOnN, _1>>;
  using Mma   = TiledMMA<MMA_Atom<FInst>, WarpLayout, Tile<Int<PermM>, Int<PermN>, Int<MmaPermK>>>;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, WarpLayout, Tile<Int<PermM>, Int<PermN>, _32>>;

  constexpr int F1 = foldF(Bits1, TK), F2 = foldF(Bits2, TK);
  constexpr int Ng1 = TN/F1, Ng2 = TN/F2;
  constexpr int row1 = F1*TK*Bits1/8, row2 = F2*TK*Bits2/8;  // physical int8 row bytes per plane
  printf("\n%s  (%d,%d,%d) w%dx%d  int%d+int%d  %s\n", tag, TM, TN, TK, WM, WN, Bits1, Bits2,
         Fold ? "[FOLD rule: MmaPermK=TK]" : "[non-fold rule: MmaPermK=32B run]");
  printf("   F1=%d F2=%d   physical rows: plane1 (%d, %d B)  plane2 (%d, %d B)   %s\n",
         F1, F2, Ng1, row1, Ng2, row2, (row1 == 32 && row2 == 32) ? "both 32 B (AIU minimum met exactly)" : "");

  // the LOGICAL (N,K) view both planes' mma fragment shares
  auto sB_log = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                            make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB_log);
  printf("   tCrB_mma (logical view) : "); print(frag.layout());
  printf("   MMA_N=%d MMA_K=%d size=%d\n",
         int(size<1>(frag.layout())), int(size<2>(frag.layout())), int(size(frag.layout())));

  // each plane's PHYSICAL swzl-read partition, in its own fold
  const int c1 = phys<MmaS8, Ng1, row1, Bits1>("plane1");
  const int c2 = phys<MmaS8, Ng2, row2, Bits2>("plane2");
  printf("   P2_DIV = CPY_K1/CPY_K2 = %d/%d = %s\n", c1, c2,
         (c2 && c1 % c2 == 0) ? (c1/c2 == 1 ? "1  (1:1 -- SIMPLER than today's 2)" : "int") : "NOT INTEGRAL <-- blocker");

  // THE CHECK THAT WOULD HAVE CAUGHT THE BOX FAILURE. A folded plane's gmem tensor is PHYSICALLY (N/F, K*F), so the
  // per-CTA gmem tile must be cut with the FOLDED tiler; cutting it with TileShape gives a (TN, TK) gmem tile against
  // a (TN/F, F*TK) smem tile and cute/algorithm/copy.hpp's `size<1>(src) == size<1>(dst)` fires. That is pure shape
  // arithmetic -- no copy atom needed to see it -- and syntax_check.sh structurally cannot: the stubs make the builder
  // fail, so nothing downstream of CollectiveMma is instantiated locally and the whole mainloop is invisible to it.
  {
    struct Tile2 { int n, k; };
    auto smem  = [](int TNv, int TKv, int F) { return Tile2{TNv / F, F * TKv}; };
    auto gmemTileShape = Tile2{TN, TK};                       // local_tile(mB2_nk, TileShape{}, ...)  <-- the bug
    auto gmemFolded    = Tile2{TN / F2, F2 * TK};             // local_tile(mB2_nk, FoldTilerB2{}, ...) <-- the fix
    auto s1 = smem(TN, TK, F1), s2 = smem(TN, TK, F2);
    printf("   gmem/smem tile agreement: plane1 smem(%d,%d) vs gmem(%d,%d) %s | plane2 smem(%d,%d) vs "
           "TileShape(%d,%d) %s , folded(%d,%d) %s\n",
           s1.n, s1.k, TN / F1, F1 * TK, (s1.n == TN / F1 && s1.k == F1 * TK) ? "OK" : "MISMATCH",
           s2.n, s2.k, gmemTileShape.n, gmemTileShape.k,
           (s2.n == gmemTileShape.n && s2.k == gmemTileShape.k) ? "OK" : "MISMATCH <-- copy.hpp static_assert",
           gmemFolded.n, gmemFolded.k,
           (s2.n == gmemFolded.n && s2.k == gmemFolded.k) ? "OK" : "MISMATCH");
  }

  // TILE **RANK**, not just extent. The box failed a second time at 2plane.hpp:808 with tB2gB2 having FIVE modes where
  // the slice passes four. local_tile on the interleaved-branch counting tensor -- whose K is a NESTED mode
  // (kCon, K/kCon) -- produces a rest whose rank depends on how the tiler's K divides kCon. Both planes must land on
  // the SAME rank or the hardcoded (_,_,_,k) slice is wrong for one of them. This is pure cute: no copy atom, no
  // builder, so it is answerable here even though syntax_check.sh structurally cannot see the mainloop at all.
  {
    constexpr int kCon = 256;
    auto counting = [&](int Nphys, int Kphys) {
      return make_counting_tensor(make_layout(
          make_shape(Nphys, make_shape(Int<kCon>{}, Kphys / kCon)),
          make_stride(ScaledBasis<_1,1>{}, make_stride(ScaledBasis<_1,0>{}, ScaledBasis<int,1>{Nphys}))));
    };
    auto rank_of = [&](int Nphys, int Kphys, int tileN, int tileK) {
      // the k coordinate is `_` in the real call (take<0,3>(blk_coord_mnkl) keeps the k ITERATION mode, which is what
      // the (_,_,_,k_iter_crd) slice later consumes). Passing a concrete 0 collapses it and reports rank 2 -- my first
      // attempt did exactly that and "verified" a model that could not fail.
      auto g = local_tile(counting(Nphys, Kphys),
                          make_shape(Int<TM>{}, tileN, tileK), make_coord(0, 0, _), Step<X,_1,_1>{});
      return int(rank(g.layout()));
    };
    const int N = 256, K = 256;                       // the real-weight test's shape
    const int r1 = rank_of(N / F1, K * F1, TN / F1, F1 * TK);
    const int r2 = rank_of(N / F2, K * F2, TN / F2, F2 * TK);
    printf("   local_tile RANK: plane1 gB rank=%d (tileK=%d vs kCon=%d)  plane2 gB2 rank=%d (tileK=%d)  %s\n",
           r1, F1 * TK, kCon, r2, F2 * TK,
           r1 == r2 ? "equal -- the (_,_,_,k) slice fits both"
                    : "DIFFER <-- the hardcoded 4-mode slice is wrong for one plane (2plane.hpp:808)");
  }

  // delivery bound per plane: slots = WN*TK/32 codes must cover one 16 B delivery = 128/bits codes
  for (int p = 0; p < 2; ++p) {
    const int b = p ? Bits2 : Bits1, slots = WN*TK/32, deliv = 128/b;
    printf("   delivery bound plane%d: slots=%d codes vs delivery=%d  %s\n", p+1, slots, deliv,
           deliv <= slots ? "OK" : "<-- ILLEGAL, needs larger WN or TK");
  }
}

int main() {
  printf("L42 -- per-plane N-fold for the 2-plane path\n");
  printf("\n======== TODAY: TK=256, both planes F=1, no fold at all ========\n");
  probe<64, 64,256,32,32, 2,1,false>("Q3 shipping");
  printf("\n======== STAGE 1: TK=128 -- only the HIGH plane folds (F1=1, F2=2) ========\n");
  printf("  the strictly smaller change: plane 1's load layer is untouched, plane 2 gains a fold.\n");
  probe<32,128,128,32,32, 2,1,true>("Q3 TK=128 w32x32");
  probe<64,128,128,32,64, 2,1,true>("Q3 TK=128 w32x64");
  printf("\n======== STAGE 2: TK=64 -- BOTH planes fold (F1=2, F2=4), WN must be 64 for int1 ========\n");
  probe<64,128, 64,64,64, 2,1,true>("Q3 TK=64 w64x64  (int1's 63.7% geometry)");
  printf("\n======== Q5 = int4 + int1, and Q6 = int4 + int2 (Q6 needs NO fold at TK=128) ========\n");
  probe<64,128,128,32,64, 4,1,true>("Q5 TK=128");
  probe<64,128,128,32,64, 4,2,false>("Q6 TK=128  (F1=F2=1 -- non-fold rule applies)");
  return 0;
}
