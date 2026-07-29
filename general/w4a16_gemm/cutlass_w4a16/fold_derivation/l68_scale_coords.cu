// L68 -- (j) the scale/zero smem view: is splitting the flattened K mode the SAME memory?
//
// SmemCopyLayoutScale is tile_to_shape(atom, (Scale_TileN, 1, Scale_TileK * Stages)) -- the group index and the pipeline
// stage FLATTENED into one mode -- so both consumers re-flatten by hand:
//     COARSE   scale_k_idx = read_stage * Scale_TileK + k_block / GroupK
//     FINE     sk          = read_stage * Scale_TileK + atom_idx / APG
// One rule, two copies, structurally identical to the hi_vreg0 defect (one rule in two copies, only one fixed). And the
// STORAGE layout SmemLayoutScale already keeps (Scale_TileK, Stages) as separate modes -- only the copy view flattens.
//
// So: give the copy view four modes and index by coordinate. That is only legal if the memory is unchanged, i.e.
//     L4(n, 0, g, s)  ==  L3(n, 0, s * Scale_TileK + g)      for every (n, g, s)
// and cosize(L4) == cosize(L3). Checked here for the real (Scale_TileN, Scale_TileK, Stages) combinations, because a
// silent stride change would corrupt the scale path in a way no local numeric gate can see.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l68_scale_coords.cu -o l68 && ./l68
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
using namespace cute;

// the builder's atom for scales: rank-2, N-major over the group count
template <int TileN, int TileK, int Stages>
static bool same_memory(const char* tag) {
  using Atom = Layout<Shape<Int<TileN>, _1>, Stride<_1, Int<TileN>>>;   // SmemLayoutAtomScale's shape family
  auto L3 = tile_to_shape(Atom{}, make_shape(Int<TileN>{}, Int<1>{}, Int<TileK * Stages>{}));
  auto L4 = tile_to_shape(Atom{}, make_shape(Int<TileN>{}, Int<1>{}, Int<TileK>{}, Int<Stages>{}));
  int bad = 0, shown = 0;
  for (int n = 0; n < TileN; ++n)
    for (int g = 0; g < TileK; ++g)
      for (int s = 0; s < Stages; ++s) {
        const int a = int(L3(n, 0, s * TileK + g));
        const int b = int(L4(n, 0, g, s));
        if (a != b) { ++bad; if (shown < 4) { printf("      n=%d g=%d s=%d : L3 %d  L4 %d\n", n, g, s, a, b); ++shown; } }
      }
  const bool cos_ok = (int(cosize(L3)) == int(cosize(L4)));
  printf("  %-34s TileN=%3d TileK=%d Stages=%d : %d coords differ, cosize %d vs %d  %s\n",
         tag, TileN, TileK, Stages, bad, int(cosize(L3)), int(cosize(L4)),
         (!bad && cos_ok) ? "SAME MEMORY" : "<-- DIFFERENT, do not split");
  return !bad && cos_ok;
}

// (j2/j4) the divide/mod pair as layouts, the same idiom HVregL uses: atom_idx -> (position in group, group)
template <int APG, int NGroups>
static bool divmod_as_layout(const char* tag) {
  using InGroupL = Layout<Shape<Int<APG>, Int<NGroups>>, Stride<_1, _0>>;   // atom_idx -> position within its group
  using GroupL   = Layout<Shape<Int<APG>, Int<NGroups>>, Stride<_0, _1>>;   // atom_idx -> which group
  int bad = 0;
  for (int a = 0; a < APG * NGroups; ++a) {
    if (int(InGroupL{}(a)) != a % APG) ++bad;
    if (int(GroupL{}(a))   != a / APG) ++bad;
  }
  printf("  %-34s APG=%d groups=%d : %d of %d disagree with %% and /  %s\n", tag, APG, NGroups, bad, 2 * APG * NGroups,
         bad ? "<-- WRONG" : "the layouts ARE the divide/mod pair");
  return bad == 0;
}

int main() {
  printf("L68 -- splitting the scale view's flattened (group, stage) mode\n\n");
  int bad = 0;
  printf("  (1) is it the same memory?\n");
  if (!same_memory<128, 8, 3>("Q3 gs=16 TK=128 s3")) ++bad;   // Scale_TileK = TK/gs
  if (!same_memory<128, 4, 2>("Q3 gs=16 TK= 64 s2")) ++bad;
  if (!same_memory<128, 4, 3>("Q6 gs=16 TK= 64 s3")) ++bad;
  if (!same_memory< 64, 8, 3>("TileN 64")) ++bad;
  if (!same_memory<256, 2, 4>("TileN 256, few groups, s4")) ++bad;
  if (!same_memory<128, 1, 3>("one group per tile (COARSE)")) ++bad;

  printf("\n  (2) the divide/mod pair as layouts\n");
  if (!divmod_as_layout<1, 4>("APG=1 (gs=16, every atom)")) ++bad;
  if (!divmod_as_layout<2, 2>("APG=2 (gs=32)")) ++bad;
  if (!divmod_as_layout<4, 4>("APG=4")) ++bad;
  if (!divmod_as_layout<8, 1>("one group")) ++bad;

  printf("\n  %s\n", bad ? "NOT safe to split" : "the split is the same memory and the coordinates are the divide/mod pair");
  return bad ? 1 : 0;
}
