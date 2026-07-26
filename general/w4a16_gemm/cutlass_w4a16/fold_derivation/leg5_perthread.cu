// LEG 5 -- per-THREAD (not per-lane-across-warps; leg4 merged 4 warps and inflated the counts).
//
// What one thread must receive, versus what the swzl hands it, in STRUCTURE not just count:
//   delivery = 4 vregs, each vreg = one 32-bit word = 32/Bits codes drawn from ONE (row, run-word) slot.
// The offline decides what sits in a word. The cheap offline (the one in the tree) puts ONE logical column's
// consecutive k there. If a thread's demand needs more distinct columns than it has words, that offline cannot
// work -- but a bit-granular offline, mixing several columns inside one word, still could.
//
// So print, per thread: #slots, #distinct n, #distinct k, and the n->k structure. Then compare with
//   words_per_swzl = 4,  swzl_calls = slots/delivery,  words_total = 4 * swzl_calls.
//   nvcc -std=c++17 -Istub_inc -I<actlize>/include leg5_perthread.cu -o leg5 && ./leg5
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

  auto part = mma.get_thread_slice(0).partition_B(idB);   // ONE thread
  std::map<int, std::set<int>> k_of_n;
  for (int i = 0; i < int(size(part)); ++i) k_of_n[get<0>(part(i))].insert(get<1>(part(i)));

  const int slots    = int(size(part));
  const int delivery = 16 * 8 / Bits;
  const int contig   = TK * Bits / 8;
  const int F        = contig >= 32 ? 1 : 32 / contig;
  const int calls    = slots / delivery;                 // swzl instructions to fill the fragment
  const int words    = 4 * (calls > 0 ? calls : 1);      // 4 vregs per call
  const int cpw      = 32 / Bits;                        // codes per 32-bit word
  const int ndist    = int(k_of_n.size());
  const int kdist    = int(k_of_n.begin()->second.size());

  printf("  %-24s int%d TN=%3d TK=%3d  F=%d\n", tag, Bits, TN, TK, F);
  printf("      thread demands %3d slots = %2d columns x %2d k each\n", slots, ndist, kdist);
  printf("      swzl gives     %3d codes/call x %d call(s) = %2d words x %2d codes\n", delivery, calls, words, cpw);
  printf("      one-column-per-word offline can supply %2d columns -- %s\n", words,
         words >= ndist ? "ENOUGH" : "NOT ENOUGH (needs a bit-granular offline that mixes columns in a word)");
  printf("      k per column: need %2d, a word holds %2d -> %s\n", kdist, cpw,
         cpw >= kdist ? "one word covers a column's k" : "a column's k spans several words");
  printf("      measured: %s\n\n", measured);
}

int main() {
  printf("LEG 5 -- per-thread demand structure vs what a word can carry\n\n");
  probe<1, 128, 128>("int1 F=2 (WORKS)",   "bad=0");
  probe<2,  64,  64>("int2 F=2 (WORKS)",   "bad=0");
  probe<4,  64,  64>("int4 F=1 (WORKS)",   "bad=0, 55.9%");
  probe<1, 128,  64>("int1 F=4 (FAILS)",   "random ~41%");
  probe<1,  64,  64>("int1 F=4 narrow",    "random ~41%, also over-delivers");
  return 0;
}
