// LEG 4 -- the step my "theorem" got wrong, checked numerically.
//
// I claimed: the four lanes of a lane/4 group demand the SAME N, so all four words of a half-run must carry one
// column, hence a folded column cannot be narrower than 16 B, hence F <= 2.
//
// LEG 2 only showed the four lanes demand the same SET of N. A thread demands many columns. So the question is
// not "same column?" but a COUNTING one:
//
//   swzl hands a thread  4 vregs x 32/Bits*4 ... = 16 BYTES = (16*8/Bits) codes.
//   the fragment demands (8 * TN/32 * TK/16) fp16 slots from that thread.
//
// The offline relayout is a FREE bijection -- that is the whole reason sub-byte formats need one at all (it
// compensates the converter's fixed emission order). So if delivery == slots, a placement EXISTS by construction:
// put the code that slot s wants at the physical position that reaches s. Whether a given offline is correct is
// a separate question from whether one exists.
//
// This program prints, per config: delivery, slots, the per-thread demanded (n,k) footprint, and the verdict.
//   nvcc -std=c++17 -Istub_inc -I<actlize>/include leg4_bijection.cu -o leg4 && ./leg4
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <set>
#include <map>
using namespace cute;

struct StubMma {};
namespace cute {
template <> struct MMA_Traits<StubMma> {
  using ValTypeD=float; using ValTypeA=cutlass::half_t; using ValTypeB=cutlass::half_t; using ValTypeC=float;
  using Shape_MNK=Shape<_16,_16,_16>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
}

template <int Bits, int TN, int TK>
void probe(const char* tag, const char* measured) {
  using Mma = TiledMMA<MMA_Atom<StubMma>, Layout<Shape<_2,_2,_1>>, Tile<_32,_32,Int<TK>>>;
  Mma mma;
  auto idB = make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{}));

  // per distinct B-consuming thread: how many slots, how many distinct n, how many distinct k
  std::map<int, std::set<std::pair<int,int>>> nk;   // lane -> demanded (n,k)
  std::set<int> ns0, ks0;
  for (int t = 0; t < 128; ++t) {
    auto part = mma.get_thread_slice(t).partition_B(idB);
    for (int i = 0; i < int(size(part)); ++i) {
      auto c = part(i);
      nk[t % 32].insert({get<0>(c), get<1>(c)});
      if (t % 32 == 0) { ns0.insert(get<0>(c)); ks0.insert(get<1>(c)); }
    }
  }
  const int slots    = 8 * (TN / 32) * (TK / 16);
  const int delivery = 16 * 8 / Bits;
  const int contig   = TK * Bits / 8;
  const int F        = contig >= 32 ? 1 : 32 / contig;

  printf("  %-26s int%d (TN=%3d, TK=%3d)  F=%d\n", tag, Bits, TN, TK, F);
  printf("      swzl delivers %3d codes/thread   fragment wants %3d slots/thread   -> %s\n",
         delivery, slots,
         delivery == slots ? "EXACT: a bijection exists"
                           : (delivery < slots ? "under-delivery: more swzl steps, fine"
                                               : "OVER-delivery: surplus unreachable, IMPOSSIBLE"));
  printf("      one thread's demand spans %zu distinct n x %zu distinct k  (so it needs MANY columns --\n"
         "      'the four lanes want one column' was never true)\n", ns0.size(), ks0.size());
  printf("      measured on ppu001: %s\n\n", measured);
}

int main() {
  printf("LEG 4 -- does a valid offline placement EXIST? (existence, not correctness of any particular one)\n\n");
  probe<1,  64,  64>("int1 F=4, narrow N",  "random ~41%  -- explained by over-delivery");
  probe<1, 128,  64>("int1 F=4, wide N",    "random ~41%  -- NOT explained by counting");
  probe<1, 128, 128>("int1 F=2 (winner)",   "bad=0");
  probe<2,  64,  64>("int2 F=2",            "bad=0");
  printf("  => TN=128/TK=64 has delivery == slots, so an offline placement EXISTS. Its box failure says the\n");
  printf("     placement I wrote is wrong, not that none is possible. The F<=2 'theorem' overreached.\n");
  return 0;
}
