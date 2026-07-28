// L45 -- can plane 2's offline placement be DERIVED CROSS-PLANE, and is the required map even a bijection?
//
// l44's finding: with different fold factors the two planes disagree on thread->column by construction (the fold
// halves plane 2's physical row count and the mma maps threads to PHYSICAL rows), so at (64,64,128) w32x32 thread 0
// holds low codes for logical N {0,8,32,40} and high bits for {0,1,16,17}. No hi_p offset repairs that.
//
// But the fold is only a LOAD-LAYER device, and the offline is derived rather than hand-written, so we are free to
// place plane 2's bits wherever we like. The question this file answers, before any packer is written:
//
//     define  T2 : (plane-2 physical bit position) -> (logical n, k)
//             "the logical element whose HIGH bit the converter will merge at this position"
//     is T2 a BIJECTION over the tile?
//
// If yes, its inverse IS the offline placement and the kernel needs no change at all. If no -- if two physical
// positions want the same logical element, or some element is never requested -- then the cross-plane merge cannot be
// satisfied by any placement and the fold must be equalized (same F on both planes) instead.
//
// Everything here is the same chain l20 uses for the single-plane offline, with the TARGET swapped from "plane 2's
// own single-plane rule" to "what plane 1's fragment demands".
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l45_xplane_offline.cu -o l45 && ./l45
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <vector>
#include <map>
using namespace cute;

template <int TM, int TN, int TK, int WM, int WN>
static bool probe(const char* tag) {
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  using WarpOnM = Int<TM/warpM>; using WarpOnN = Int<TN/warpN>;
  constexpr int PermM = TM/warpM*InstM, PermN = TN/warpN*InstN;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<WarpOnM, WarpOnN, _1>>,
                         Tile<Int<PermM>, Int<PermN>, _32>>;
  constexpr int NTHR = 32 * (TM/warpM) * (TN/warpN);

  // plane 1: int2, F1 = 1 at TK=128 -> physical == logical, rows TN, bytes TK*2/8
  constexpr int R1 = TN, B1 = TK * 2 / 8;
  // plane 2: int1, F2 = 2 at TK=128 -> rows TN/2, bytes 2*TK*1/8
  constexpr int F2 = 2, R2 = TN / F2, B2 = F2 * TK * 1 / 8;

  auto id1 = make_identity_tensor(make_shape(Int<R1>{}, Int<B1>{}));
  auto id2 = make_identity_tensor(make_shape(Int<R2>{}, Int<B2>{}));

  // T2[physical bit of plane 2] = the logical (n,k) whose high bit belongs there, encoded n*TK + k. -1 = unclaimed.
  std::vector<int> T2((size_t)R2 * B2 * 8, -1);
  long claims = 0, conflicts = 0;
  std::map<int,int> want;                                   // logical element -> times demanded

  for (int t = 0; t < NTHR; ++t) {
    auto p1 = MmaS8{}.get_thread_slice(t).partition_B(id1);   // (val, MMA_N, MMA_K)
    auto p2 = MmaS8{}.get_thread_slice(t).partition_B(id2);
    const int V1 = int(size<0>(p1.layout())), N1 = int(size<1>(p1.layout())), K1 = int(size<2>(p1.layout()));
    const int V2 = int(size<0>(p2.layout())), N2 = int(size<1>(p2.layout())), K2 = int(size<2>(p2.layout()));
    // The converter consumes plane 1 and plane 2 in LOCK STEP over the thread's own slots: the i-th low code this
    // thread converts is merged with the i-th high bit it holds. That IS the correspondence -- the emission order is
    // a permutation applied to both sides identically (l38: each _E2 line reads the low crumb AND the high bit and
    // writes one h2), so a slot-index pairing is the right granularity to derive a placement at.
    long i1 = 0, i2 = 0;
    for (int k = 0; k < K1; ++k) for (int n1 = 0; n1 < N1; ++n1) for (int v = 0; v < V1; ++v) {
      auto c = p1(v, n1, k);
      const int row = int(get<0>(c)), byte = int(get<1>(c));
      for (int sub = 0; sub < 4; ++sub, ++i1) {              // 4 int2 codes per int8
        const int n = row, kk = byte * 4 + sub;              // plane 1 unfolded: physical row == logical n
        // the i2-th high bit this thread holds, in plane 2's PHYSICAL space
        const long slot = i1;                                 // lock step
        const int kb = int(slot / (V2 * 8L * N2)), rest = int(slot % (V2 * 8L * N2));
        const int nn = rest / (V2 * 8), r2 = rest % (V2 * 8);
        if (kb >= K2) { continue; }
        auto c2 = p2(r2 / 8, nn, kb);
        const int g = int(get<0>(c2)), b2 = int(get<1>(c2));
        const long bitpos = ((long)g * B2 + b2) * 8 + (r2 % 8);
        const int elem = n * TK + kk;
        ++claims; ++want[elem];
        if (T2[bitpos] >= 0 && T2[bitpos] != elem) ++conflicts;
        T2[bitpos] = elem;
        ++i2;
      }
    }
  }
  long unclaimed = 0; for (int e : T2) if (e < 0) ++unclaimed;
  // MULTIPLICITY IS NOT A CONFLICT. B is split across the N warps only, so all TM/WM warps in M read the SAME B --
  // every element is legitimately demanded WOM times. My first version required multiplicity 1 and would have
  // reported (64,64,128) w32x32 (WOM=2) infeasible when it is fine. The real criteria are: no two threads want
  // DIFFERENT elements at the same physical bit (conflicts), and no bit goes unused (unclaimed).
  constexpr int WOM = TM / warpM;
  long badmult = 0; for (auto& kv : want) if (kv.second != WOM) ++badmult;
  const bool ok = (conflicts == 0 && unclaimed == 0 && badmult == 0);
  printf("  %-28s bits=%zu claims=%ld conflicts=%ld unclaimed=%ld mult!=WOM(%d)=%ld -> %s\n",
         tag, T2.size(), claims, conflicts, unclaimed, WOM, badmult,
         ok ? "BIJECTION: the inverse IS the offline placement"
            : "NOT a bijection at this pairing -- see below");
  return ok;
}

int main() {
  printf("L45 -- is the cross-plane target map a bijection? (feasibility, before writing any packer)\n\n");
  bool a = probe<64, 64,128,32,32>("Q3 (64,64,128) w32x32");
  bool b = probe<32,128,128,32,32>("Q3 (32,128,128) w32x32");
  printf("\n  NOTE ON THE PAIRING. This assumes the converter merges the thread's i-th low code with its i-th high\n");
  printf("  bit in slot order. That is the assumption to attack if the counts balance but the numbers still differ:\n");
  printf("  the true pairing is whatever MixGemm2Plane_uint2_uint1 emits, and l37/l38 have it as a tuple of layouts.\n");
  printf("  A count-level bijection is NECESSARY, not sufficient -- it only says no placement is ruled out.\n");
  printf("\n  verdict: %s\n", (a && b) ? "feasible at both shapes"
                                       : "at least one shape is infeasible under this pairing");
  return 0;
}
