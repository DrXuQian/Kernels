// L8 -- derive pi STRUCTURALLY instead of calibrating it, so it transfers to any MMA_N.
//
// l7 measured pi (converter element index -> mma fragment logical index) as a fixed GF(2)-linear rotation of the
// high four element bits, at MMA_N=2. That calibration cannot be carried to the WN=64 target, where MMA_N=4 and
// the fragment's mode split changes. Hence this.
//
// TWO HYPOTHESES, both now CONFIRMED against l7's calibration, and agreeing with each other at MMA_N=4.
//
// H1  pi = tCrB_mma.layout()^-1.
//     In the collective, cvt_out = make_tensor(tCrB_mma(...).data(), cvt_in.layout()), and convert_tensor writes
//     ConversionVectorWidth CONTIGUOUS elements per chunk from that raw pointer. cvt_in is compact (measured, see
//     l9_cvtin.cu: ((_32,_2,_2),_1):((_1,_32,_64),_0), one iteration for int1), so converter element e lands at
//     FLAT offset e. But partition_fragment_B is make_fragment_like(partition_B(...)), which compacts in the order
//     the partition's strides imply -- and for the collective's ROW-major SmemLayoutB_MmaView that puts MMA_K
//     ahead of MMA_N: ((2,2,2),2,8):((1,2,4),64,8). So logical index and flat offset differ, and the (n,k) at flat
//     e is the one whose logical index f has layout(f) == e.
//
// H2  the same thing stated without cute: the converter's element index nests (val, MMA_K, MMA_N) while the
//     fragment's logical index nests (val, MMA_N, MMA_K), so pi re-reads the group above `val` with the two modes
//     swapped. Useful as a cross-check because it contains no cute call.
//
// THE BUG THAT MADE H1 LOOK FALSE THE FIRST TIME. This file originally built sB as make_layout(Shape<TN,TK>),
// which is cute's default LayoutLeft -- COLUMN major. The collective uses make_stride(Int<TKe>{}, _1{}), row
// major. That one stride transposes the partition and changes which mode make_fragment_like compacts first, so
// the fragment came out ((_1,_2,_4),_8,_16) instead of ((_1,_2,_4),_64,_8) and pi came out as the identity.
// The lesson is the same one as L4 and L6: the step that looked too obvious to check was the broken one.
//
// CONSEQUENCE: pi needs no calibration constant. It is derivable from cute for any (Bits, TN, TK, WM, WN), which
// is what unblocks generating the placement for a config the shipped offline has never seen.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l8_pi_structural.cu -o l8 && ./l8
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <map>
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

// l7's MEASURED pi, at MMA_N=2: the high four element bits rotate by one.
static int pi_measured(int e) { const int hi = (e >> 3) & 0xF; return (e & 7) | ((((hi << 1) | (hi >> 3)) & 0xF) << 3); }

// HYPOTHESIS 2 (structural, config-general). The fragment's flat index nests (val, MMA_N, MMA_K):
//     flat = val + VAL*(mma_n + MMA_N*mma_k)
// The converter's element index nests the OTHER WAY, (val, MMA_K, MMA_N):
//     e    = val + VAL*(mma_k + MMA_K*mma_n)
// so pi just re-reads the group above `val` with the two modes swapped. No calibration constant in it -- only
// VAL, MMA_N and MMA_K, all of which cute reports for any config.
static int pi_swap(int e, int VAL, int MMA_N, int MMA_K) {
  const int val  = e % VAL, rest = e / VAL;
  const int mk   = rest % MMA_K, mn = rest / MMA_K;
  return val + VAL * (mn + MMA_N * mk);
}

template <int TN, int TK, int WM, int WN>
void probe(const char* tag, bool compare_measured) {
  constexpr int WOM = 32 / WM, WON = TN / WN;   // TM is irrelevant to B; use 32 so WOM=1 for the calib shape
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto thr = Mma{}.get_thread_slice(0);
  // THE STRIDE MATTERS. The collective's SmemLayoutB_MmaView is make_stride(Int<TKe>{}, _1{}) -- ROW major over
  // the folded bytes. Building sB with make_layout(Shape<TN,TK>) gets cute's default LayoutLeft (column major),
  // which transposes the partition and hence changes which mode make_fragment_like compacts first. That single
  // wrong stride is what made H1 look false the first time this file ran.
  auto sB  = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                         make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));

  auto part = thr.partition_B(sB);                 // the partition; its layout is what make_fragment_like reorders
  auto frag = thr.partition_fragment_B(sB);        // == make_fragment_like(part)

  printf("  %s  TN=%d TK=%d warp %dx%d | MMA=%d MMA_N=%d MMA_K=%d\n",
         tag, TN, TK, WM, WN, int(size<0>(frag)), int(size<1>(frag)), int(size<2>(frag)));
  printf("      partition layout : "); print(part.layout()); printf("\n");
  printf("      fragment layout  : "); print(frag.layout()); printf("\n");

  // pi = frag.layout()^-1 : flat offset -> logical index
  const int n = int(size(frag));
  std::vector<int> inv(n, -1);
  bool onto = true;
  for (int f = 0; f < n; ++f) {
    const int off = frag.layout()(f);
    if (off < 0 || off >= n) { onto = false; continue; }
    inv[off] = f;
  }
  for (int e = 0; e < n; ++e) if (inv[e] < 0) onto = false;
  printf("      frag.layout() is a bijection on [0,%d): %s\n", n, onto ? "yes" : "NO");

  bool ident = true;
  for (int e = 0; e < n && onto; ++e) if (inv[e] != e) ident = false;
  printf("      pi = frag.layout()^-1 is the identity: %s\n", ident ? "yes" : "NO");

  if (onto) {
    printf("      pi on the element-index basis: ");
    for (int b = 0; (1 << b) < n; ++b) printf("bit%d->%d  ", b, inv[1 << b]);
    printf("\n");
  }
  if (compare_measured && onto) {
    int bad = 0;
    for (int e = 0; e < n; ++e) if (inv[e] != pi_measured(e)) ++bad;
    printf("      H1 pi == frag.layout()^-1 : %d / %d differ -> %s\n",
           bad, n, bad == 0 ? "CONFIRMED" : "WRONG");
  }
  // HYPOTHESIS 2, checked against the measured pi wherever we have it, and printed everywhere
  {
    const int VAL = int(size<0>(frag)), MN = int(size<1>(frag)), MK = int(size<2>(frag));
    int bad = 0;
    for (int e = 0; e < n; ++e) if (pi_swap(e, VAL, MN, MK) != pi_measured(e)) ++bad;
    printf("      H2 pi == (val, K, N) -> (val, N, K) swap, with VAL=%d MMA_N=%d MMA_K=%d\n", VAL, MN, MK);
    if (compare_measured)
      printf("         vs l7's MEASURED pi: %d / %d differ -> %s\n", bad, n, bad == 0 ? "CONFIRMED" : "WRONG");
    printf("         basis: ");
    for (int b = 0; (1 << b) < n; ++b) printf("bit%d->%d  ", b, pi_swap(1 << b, VAL, MN, MK));
    printf("\n");
    // must be a bijection for any config, or it is not a relabelling
    std::vector<int> hit(n, 0); bool bij = true;
    for (int e = 0; e < n; ++e) { int f = pi_swap(e, VAL, MN, MK); if (f < 0 || f >= n || hit[f]++) bij = false; }
    printf("         bijection on [0,%d): %s\n", n, bij ? "yes" : "NO");
  }
  printf("\n");
}

int main() {
  printf("L8 -- is pi just the fragment layout's inverse?\n\n");
  printf("  == the config l7 calibrated on (MMA_N=2). This is the test that matters.\n");
  probe<128, 128, 32, 32>("calib ", true);
  printf("  == the WN=64 target, plus neighbours, to see how pi moves with the mode split\n");
  probe<128,  64, 32, 64>("target", false);
  probe< 64,  64, 32, 32>("int2-ish shape", false);
  probe<128, 256, 32, 32>("int1 unfolded", false);
  return 0;
}
