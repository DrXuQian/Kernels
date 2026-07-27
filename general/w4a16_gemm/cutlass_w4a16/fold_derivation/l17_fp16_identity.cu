// L17 (step 3b) -- the free calibration: for fp16 the whole chain must be the IDENTITY.
//
// fp16 weights need NO offline relayout, and the reason is that the AIU write's .swzl and the ldmatrix read's .swzl
// are a matched hardware pair that cancels. So with no converter in the way:
//
//     gmem tile (n,k)  --AIU(linear)-->  smem (row=n, word=k/2, half=k%2)
//                      --L2^-1-------->  (lane, vreg, half)
//                      --fragment----->  mma wants (n', k')
//     and n'==n, k'==k must hold for every position.
//
// This is a stronger and more basic check than any of the sub-byte calibrations, because it has no converter and no
// offline to hide a compensating error in: if L2 is wrong, there is nothing left to cancel it. It also needs no box.
//
// SCOPE. L2 models ONE 32B slice. For fp16 a 32B run is 16 half, so TK=16 gives a single-slice cube; TK=64 would be
// a 128B/4-slice cube whose cross-slice term carries (l15). The single-slice case is the one that isolates L2.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l17_fp16_identity.cu -o l17 && ./l17
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <map>
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

// L2, the validated swzl delivery: (lane, vreg) -> logical 32-bit word of the cube.
//   row = (v/2)*8 + lane/4     word-in-row = (v%2)*4 + lane%4
static inline void l2(int lane, int v, int& row, int& wd) { row = (v/2)*8 + lane/4; wd = (v%2)*4 + lane%4; }

template <int TN, int TK, int WM, int WN>
static bool check(const char* tag) {
  constexpr int WOM = 32/WM, WON = TN/WN;
  constexpr int HPW = 2;                       // half per 32-bit word
  constexpr int W_ROW = TK/HPW;                // words per row of the tile
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));
  std::vector<int> pi(NS,-1);
  for (int f=0; f<NS; ++f) { const int o = frag.layout()(f); if (o>=0 && o<NS) pi[o]=f; }

  const int RPI = WON*16, NINST = TN/RPI, VEC = 8;   // one swzl delivers 16 B = 8 half
  printf("  %-22s fp16 TN=%3d TK=%2d w%dx%d | W_ROW=%d rows/instance=%d instances=%d frag=%d\n",
         tag, TN, TK, WM, WN, W_ROW, RPI, NINST, NS);
  if (W_ROW != 8) { printf("      cube is %d words wide (%d slices) -- L2 models one 32B slice, SKIPPED\n",
                           W_ROW, W_ROW/8); return true; }

  long bad = 0, tot = 0; int shown = 0;
  for (int i = 0; i < NINST; ++i)
    for (int t = 0; t < 32*WOM*WON; ++t) {
      const int lane = t%32, w = t/32, warp_n = w/WOM;
      auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
      for (int v = 0; v < 4; ++v) {
        int row0, wd; l2(lane, v, row0, wd);
        const int row = i*RPI + 16*warp_n + row0;
        for (int h = 0; h < HPW; ++h) {
          // fp16: no converter, so the delivered half lands at flat offset (v*HPW + h) of this call's VEC block
          const int flat = i*VEC + v*HPW + h;
          if (flat < 0 || flat >= NS) continue;
          auto c = part(pi[flat]);
          const int n_want = int(get<0>(c)), k_want = int(get<1>(c));
          const int n_have = row, k_have = wd*HPW + h;      // smem is row-major (n,k) and the AIU write is linear
          ++tot;
          if (n_want != n_have || k_want != k_have) {
            ++bad;
            if (shown < 6) { printf("      MISMATCH i%d lane%2d v%d h%d: delivered (n%2d,k%2d) but mma wants (n%2d,k%2d)\n",
                                    i, lane, v, h, n_have, k_have, n_want, k_want); ++shown; }
          }
        }
      }
    }
  printf("      identity over %ld positions: %ld mismatch  %s\n\n", tot, bad,
         bad ? "<-- L2 (or the linear-AIU assumption) is WRONG for fp16" : "<-- IDENTITY HOLDS");
  return bad == 0;
}

int main() {
  printf("L17 -- fp16 must come out as the identity, with no converter and no offline to hide an error in\n\n");
  bool ok = true;
  ok &= check<64, 16, 32, 32>("single-slice");
  ok &= check<128, 16, 32, 32>("single-slice wide N");
  ok &= check<64, 64, 32, 32>("4-slice (expected skip)");
  printf("  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
