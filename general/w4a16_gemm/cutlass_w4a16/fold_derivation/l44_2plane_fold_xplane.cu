// ***** RETRACTED (l47). The conclusion below -- "the two planes disagree on thread->column, the data a thread needs
// ***** is in another thread's registers" -- is WRONG, and the error is in this file's premise, not its arithmetic.
// ***** It labels each plane's PHYSICAL row as a logical N column (n = f + F*g). That identification holds only if the
// ***** offline is the identity, and it is not: the offline permutes. So both sides were physical rows dressed up as
// ***** logical, and the comparison was meaningless.
// *****
// ***** What actually settled it: l47 emits the buffer and diffs it against the shipped offline. At F2=1 the derived
// ***** cross-plane buffer is BIT-IDENTICAL to what runs today (0/16384, 0/65536) -- so the generator is right -- and
// ***** at F2=2 it is ALSO bit-identical to plane 2's own single-plane rule. The offline was never wrong. The defect
// ***** behind bad=15010/32768 is kernel-side: cvt_hi(_, ii) is out of range once the fold drops plane 2 to MMA_N=1.
// *****
// ***** Kept for the method, not the claim: derive, then GATE THE DERIVATION AGAINST SOMETHING THAT ALREADY WORKS.
// ***** A self-consistency check (T2) cannot catch a wrong premise shared by both sides of the comparison.
//
// L44 -- which LOGICAL N columns each plane's vregs carry once plane 2 is folded, and therefore what the converter's
// high-plane offset must become. The box says bad=15010/32768 on the Block_K=128 folded path while the unfolded
// control is bad=0, so the gmem/smem plumbing is right and the cross-plane REGISTER correspondence is not.
//
// The correspondence shipped today was derived at P2_DIV=2 with plane 2 UNFOLDED:
//     hi_p = &cvt_hi(_, ii) + (k_block % P2_DIV);   then the converter indexes hi[2*(v>>1)]
// i.e. the high vreg index decomposes as (k_block parity) + 2*(N-half). Folding plane 2 changes what those four
// vregs mean: its fragment loses N extent (MMA_N 2 -> 1, l42) because F2 adjacent logical N columns now live in ONE
// physical row. So `cvt_hi(_, ii)` is not just wrong, it is out of range for ii >= MMA_N2.
//
// Physical->logical for a folded plane, from the fold collective's own logical view
// SmemLayoutB_Logical = ((FoldF, Ng), TKe) : ((TKe, FoldF*TKe), 1):
//     physical row g holds [f=0's TK codes][f=1's TK codes]...,  logical n = f + F*g,  k = c % TK
// F == 1 degenerates to n = g, which is what plane 1 does today.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l44_2plane_fold_xplane.cu -o l44 && ./l44
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <set>
#include <map>
using namespace cute;

// one plane's fragment, and the LOGICAL (n,k) each of its (vreg, code) slots carries, for one thread
template <int TM, int TN, int TK, int WM, int WN, int Bits, int F>
static std::map<int, std::set<int>> plane_cols(int thr, const char* tag, bool print) {
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  using WarpOnM = Int<TM/warpM>; using WarpOnN = Int<TN/warpN>;
  constexpr int PermM = TM/warpM*InstM, PermN = TN/warpN*InstN;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<WarpOnM, WarpOnN, _1>>,
                         Tile<Int<PermM>, Int<PermN>, _32>>;
  constexpr int Ng = TN / F, RowB = F * TK * Bits / 8;      // physical rows, bytes per physical row

  // partition an IDENTITY tensor over the PHYSICAL (Ng, RowB) int8 tile: gives, per slot, the (row, byte) it reads
  auto ident = make_identity_tensor(make_shape(Int<Ng>{}, Int<RowB>{}));
  auto part  = MmaS8{}.get_thread_slice(thr).partition_B(ident);       // (val, MMA_N, MMA_K)
  const int NV = int(size<1>(part.layout()));

  std::map<int, std::set<int>> cols;                  // mode1 index -> the logical N columns it touches
  for (int n1 = 0; n1 < NV; ++n1)
    for (int v = 0; v < int(size<0>(part.layout())); ++v) {
      auto c = part(v, n1, 0);
      const int g = int(get<0>(c)), byte = int(get<1>(c));
      const int code0 = byte * 8 / Bits;               // first code in this byte
      const int f = code0 / TK;                        // which folded column
      cols[n1].insert(f + F * g);                      // logical n
    }
  if (print) {
    printf("  %-26s F=%d  phys(%d rows, %d B)  fragment mode1 extent=%d\n", tag, F, Ng, RowB, NV);
    for (auto& kv : cols) {
      printf("      mode1=%d -> logical N {", kv.first);
      int c = 0; for (int n : kv.second) { if (c++) printf(","); printf("%d", n); }
      printf("}\n");
    }
  }
  return cols;
}

int main() {
  printf("L44 -- cross-plane logical-N correspondence, thread 0, config (64,64,128) w32x32 (the failing one)\n\n");
  auto p1 = plane_cols<64,64,128,32,32, 2, 1>(0, "plane1 int2 (unfolded)", true);
  auto p2 = plane_cols<64,64,128,32,32, 1, 2>(0, "plane2 int1 (F=2)",      true);

  printf("\n  what the converter does today: for ii in [0, %d) it reads cvt_hi(_, ii)\n", (int)p1.size());
  printf("  plane 2 has only %d mode1 entr%s -> ii=%d.. is OUT OF RANGE\n",
         (int)p2.size(), p2.size() == 1 ? "y" : "ies", (int)p2.size());

  printf("\n  the mapping that must replace it -- for each of plane 1's ii, which plane-2 entry holds its columns:\n");
  bool clean = true;
  for (auto& a : p1) {
    int hit = -1;
    for (auto& b : p2) {
      bool subset = true;
      for (int n : a.second) if (!b.second.count(n)) subset = false;
      if (subset) { hit = b.first; break; }
    }
    printf("      plane1 ii=%d {", a.first);
    int c = 0; for (int n : a.second) { if (c++) printf(","); printf("%d", n); }
    printf("}  ->  plane2 mode1=%d %s\n", hit, hit < 0 ? " <-- NO single entry covers it" : "");
    if (hit < 0) clean = false;
  }
  printf("\n  => %s\n", clean
      ? "every plane-1 ii maps into ONE plane-2 entry: the fix is an index map, not a re-derivation of the bit order"
      : "a plane-1 ii spans MORE THAN ONE plane-2 entry -- the converter must split the read");
  return 0;
}
