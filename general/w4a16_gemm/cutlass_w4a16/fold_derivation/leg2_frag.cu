// LEG 2: does the mma B-fragment's logical N ever depend on lane%4?
//
// LEG 1 showed the 8-word run is a 2x4 grid: (v%2) picks a 4-word half, (lane%4) picks the word in it. So if a
// folded column occupies FEWER than 4 words, the column a thread receives must vary with lane%4. This leg asks
// cute whether the fragment can express that: for each thread, dump the set of logical N it demands, grouped by
// lane%4 (holding lane/4 fixed). If N is constant across lane%4, a lane%4-dependent column is inexpressible.
//   nvcc -std=c++17 -I<cutlass>/include leg2_frag.cu -o leg2 && ./leg2
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

template <int TN, int TK>
void probe(const char* tag) {
  using Mma = TiledMMA<MMA_Atom<StubMma>, Layout<Shape<_2,_2,_1>>, Tile<_32,_32,Int<TK>>>;
  Mma mma;
  // identity tensor over the logical (N,K) B tile: each element carries its own (n,k) coordinate
  auto idB = make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{}));

  std::map<int, std::map<int, std::set<int>>> N_by_lane;  // lane/4 -> lane%4 -> set of n
  std::map<int, std::map<int, std::set<int>>> K_by_lane;

  for (int t = 0; t < 128; ++t) {                       // 128 threads = 4 warps (2x2)
    auto thr  = mma.get_thread_slice(t);
    auto part = thr.partition_B(idB);
    int lane = t % 32;
    for (int i = 0; i < int(size(part)); ++i) {
      auto c = part(i);
      N_by_lane[lane / 4][lane % 4].insert(get<0>(c));
      K_by_lane[lane / 4][lane % 4].insert(get<1>(c));
    }
  }

  // is the demanded N set independent of lane%4 (for fixed lane/4)?
  bool n_indep = true, k_indep = true;
  for (auto& [hi, m] : N_by_lane) { auto ref = m.begin()->second; for (auto& [lo,s] : m) if (s != ref) n_indep = false; }
  for (auto& [hi, m] : K_by_lane) { auto ref = m.begin()->second; for (auto& [lo,s] : m) if (s != ref) k_indep = false; }

  printf("  %-22s TN=%3d TK=%3d\n", tag, TN, TK);
  printf("      demanded N depends on lane%%4 ? %-3s   demanded K depends on lane%%4 ? %s\n",
         n_indep ? "NO" : "yes", k_indep ? "NO" : "yes");
  // show it concretely for lane/4 == 0
  printf("      lane/4=0:");
  for (auto& [lo, s] : N_by_lane[0]) { printf("  lane%%4=%d N={", lo); int c=0; for (int n : s) { if (c++<6) printf("%d,", n); } printf(c>6?"..}":"}"); }
  printf("\n      lane/4=0:");
  for (auto& [lo, s] : K_by_lane[0]) { printf("  lane%%4=%d K={", lo); int c=0; for (int k : s) { if (c++<6) printf("%d,", k); } printf(c>6?"..}":"}"); }
  printf("\n");
}

int main() {
  printf("LEG 2 -- what the mma B-fragment demands per thread, split by lane%%4\n\n");
  probe<64,  64>("int2 fold (works)");
  probe<128, 128>("int1 fold (works)");
  probe<64,  128>("F=1 baseline");
  probe<128, 64>("int1 F=4 (fails)");
  printf("\n  => N is a function of lane/4 (and value bits) ONLY; lane%%4 moves K, never N.\n");
  return 0;
}
