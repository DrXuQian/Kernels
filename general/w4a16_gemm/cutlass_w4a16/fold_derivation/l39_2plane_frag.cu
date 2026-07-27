// L39 -- what tCrB_mma ACTUALLY is at the 2-PLANE config, and whether the existing contiguous-64 write is even the
// right index space to chunk. Prerequisite for task #12; also the answer to "why did TK=256 come back bit-identical".
//
// TWO facts collided:
//   * l34 printed the fold path's fragment as ((2,2,2), MMA_N, MMA_K) : ((1,2,4), 32, 8)  -- k INNER, n stride 8*MMA_K
//   * the 2-plane collective writes ONE COPY STEP as a CONTIGUOUS run of 64 fp16 starting at
//     tCrB_mma(_,_,k_block*K_ATOM_PER_COPY).data(), which is only correct if k is the OUTER mode (stride 8*MMA_N)
// l34's config was (64,128,64) w32x64 -> MMA_N = 4 AND MMA_K = 4, so 8*MMA_K and 8*MMA_N are BOTH 32: that measurement
// cannot tell the two apart. The 2-plane runs (64,64,256) w32x32 -> MMA_N=2, MMA_K=16, where they are 128 vs 16. So
// the whole shape of #12's gate (what "chunk" indexes, and what at() must return) hangs on a fact I never measured at
// a config that could distinguish it. Print it.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l39_2plane_frag.cu -o l39 && ./l39
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
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

// One config: print the real fragment, then ask whether a CONTIGUOUS kOut-element run at the copy step's base covers
// exactly that step's (n, k) atoms -- the assumption the 2-plane convert is built on.
template <int TM, int TN, int TK, int WM, int WN, int Bits>
static bool probe(const char* tag) {
  constexpr int WOM = TM / WM, WON = TN / WN;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  constexpr int MMA_N = int(size<1>(decltype(frag.layout()){}));
  constexpr int MMA_K = int(size<2>(decltype(frag.layout()){}));
  // delivery: 16 B/thread = 128/Bits codes; one (n,k) atom costs 8 codes, a step spans all MMA_N -> atoms per step
  constexpr int kCodes = 128 / Bits;
  constexpr int kAtomPerCopy = kCodes / (8 * MMA_N);
  constexpr int kOut  = kCodes;                                  // fp16 outputs one delivery produces
  printf("\n%s  (%d,%d,%d) w%dx%d int%d : MMA_N=%d MMA_K=%d  delivery=%d codes  K_ATOM_PER_COPY=%d  KBM=%d\n",
         tag, TM, TN, TK, WM, WN, Bits, MMA_N, MMA_K, kCodes, kAtomPerCopy, MMA_K / kAtomPerCopy);
  printf("   tCrB_mma : "); print(frag.layout()); printf("\n");
  printf("   n stride %d : 8*MMA_K=%d (k inner)  8*MMA_N=%d (k outer)  -> %s\n",
         int(stride<1>(frag.layout())), 8*MMA_K, 8*MMA_N,
         (MMA_N == MMA_K) ? "AMBIGUOUS at this config" :
         (int(stride<1>(frag.layout())) == 8*MMA_K ? "k INNER" : "k OUTER"));
  if (kAtomPerCopy == 0 || MMA_K % kAtomPerCopy) { printf("   (step does not divide the tile -- skip)\n"); return true; }

  // The assumption under test: for every copy step, the kOut CONTIGUOUS offsets starting at the slice's base are
  // exactly the offsets of that step's atoms {(n, k) : n < MMA_N, k in [step*APC, step*APC + APC)}.
  bool ok = true;
  for (int step = 0; step < MMA_K / kAtomPerCopy; ++step) {
    const int base = int(frag.layout()(0, 0, step * kAtomPerCopy));
    long owned = 0, hit = 0;
    for (int n = 0; n < MMA_N; ++n)
      for (int k = step * kAtomPerCopy; k < (step + 1) * kAtomPerCopy; ++k)
        for (int v = 0; v < 8; ++v) {
          const int off = int(frag.layout()(v, n, k)) - base;
          ++owned;
          if (off >= 0 && off < kOut) ++hit;                     // inside the contiguous run the convert writes
        }
    if (hit != owned) { ok = false;
      printf("   step %d: only %ld / %ld of this step's atoms land in the contiguous [%d, %d) run"
             "  <-- the contiguous-run write is WRONG here\n", step, hit, owned, base, base + kOut); }
  }
  if (ok) printf("   every copy step's atoms lie in ONE contiguous %d-element run  <-- flat index space is valid\n", kOut);
  return ok;
}

int main() {
  printf("== the config l34 measured: MMA_N == MMA_K, so it cannot distinguish k-inner from k-outer ==\n");
  bool a = probe<64,128,64,32,64,1>("fold best-ish");
  printf("\n== the fold path's actual best config ==\n");
  bool b = probe<64,128,64,64,64,1>("fold BEST (64,128,64) w64x64");
  printf("\n== the 2-plane's LOCKED config: MMA_N=2, MMA_K=16 -- here the two differ by 8x ==\n");
  bool c = probe<64,64,256,32,32,2>("2plane low (int2)");
  bool d = probe<64,64,256,32,32,1>("2plane high (int1)");
  printf("\n== other 2-plane sweep shapes ==\n");
  bool e = probe<32,128,256,32,32,2>("2plane (32,128,256) w32x32 int2");
  printf("\nVERDICT: contiguous-run index space valid at every probed config: %s\n",
         (a&&b&&c&&d&&e) ? "YES" : "NO -- see the WRONG lines above");
  return 0;
}
