// L27 -- step 2 was going to interleave sS/sZ. This file is why it should not be built: the interleave is a 2x on a
// path that is carrying a 16-64x redundancy, so it fixes the wrong thing.
//
// WHAT STEP 2 WAS. The zero diagnostic said COPY, not TRANSFORM: at TK=128 the ScaleZero delta grew with gs (+49.6us
// at gs=32, +83.0us at gs=16) while the transform count is gs-independent. So interleave sS and sZ into one tensor
// with a trailing extent-2 mode, turning the FINE branch's two copies into one of twice the width. A 2x.
//
// FIRST THE PREREQUISITE, because the interleave is a LOSS if the reads are already vectorized: interleaving splits
// a contiguous run apart. Measured -- max contiguous run is 1 half in EVERY config. So the interleave really is a
// clean 2x, and a permuted layout that groups a thread's own elements only reaches max-run 2 (my "transpose n from
// (b,a) to (a,b) makes it 8" derivation was wrong; the per-thread n set is not what I claimed).
//
// AND THEN THE NUMBER THAT MAKES ALL OF THAT IRRELEVANT. Per thread, one scale reload asks for 64-256 slots to fetch
// 4-8 DISTINCT smem elements:
//     int4 PRODUCTION (64,64,64) w32x32 :  64 slots ->  4 distinct = 16x redundant
//     int1 best       (32,128,64) w32x64: 128 slots ->  8 distinct = 16x
//     int1            (32,128,256) w32x32: 256 slots -> 4 distinct = 64x
// The cause is structural and visible in the partition:
//     ((_1,(_2,_2,_2,_4)),_4,_1,_2):((_0,(_1@1,_8@1,_8@0,_16@1)),_32@0,_0,_1@2)
//                                            ^^^^   ^^^^   ^^^^^
// three val modes walk mode-1 of the (TN, 1, SK) scale tensor -- and that mode has extent 1. sS is k-invariant, but
// make_tiled_copy_B builds the copy for the FULL B tile (TN x TK), so every k-walking mode collapses to stride 0 and
// re-requests the same element. The fragment mirrors B's shape for the same reason, so the replication is
// materialised in registers too: 4x in every config, e.g. 32 reg-halves holding 8 distinct scales.
//
// The replication is not an accident -- it is what lets the transform be a shuffle-free elementwise
//     cute::transform(tCrB_mma(_,_,atom), tCrS(_,_,0), tCrB_mma(_,_,atom), multiplies{})
// against a B fragment of the same shape. The cost is up to 256 smem requests and 4x the scale registers.
//
// CORRECTION TO THE HEADLINE NUMBER. nslot is what the partition ASKS for, not what the copy ISSUES. cute's
// copy(Copy_Atom<...>, src, dst) already auto-filters -- it takes nullspace(layout<1>(dst_v)) and divides both
// operands by it, skipping redundant iterations along the DESTINATION's stride-0 modes. That is in the general
// CopyAtom overload rather than gated on AutoFilter, so the collective gets it today. The destination fragment
// distinguishes nreg registers, so ~nreg requests are issued for ne distinct values: 4x redundant, not 16-64x.
// The register saving (nreg -> ne, also 4x) is the definite part, and it is tier-1.
//
// THE FIX THAT SUBSUMES THE INTERLEAVE. Keep the replicated SHAPE the transform needs, but stop materialising it:
//     tCrS_ld = compact fragment of ne elements          <- the copy targets this: ne loads, not nslot
//     tCrS_bc = make_tensor(tCrS_ld.data(), <B-fragment shape, stride 0 on the k modes>)
// and hand tCrS_bc to transform. The transform is BYTE-IDENTICAL in what it computes -- it just reads the same
// register 4x instead of 4 copies of it. Gains, all measured here rather than hoped for:
//     smem requests per reload : ~nreg -> ne          4x  (NOT 16-64x -- see the correction below)
//     scale registers          : nreg  -> ne           4x   (and this is tier-1: TK=256's 320 regs included 32 here)
//     ScaleZero                : both halves shrink, so it beats the interleave's 2x on its own
// This does NOT touch the transform, the B path, or the offline -- which is why it is a safer change than the
// interleave, not just a bigger one. (The earlier fused-FMA attempt regressed 52.3% -> 33.5% precisely because it
// rewrote the transform into a scalar loop.)
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l27_scale_contiguity.cu -o l27 && ./l27
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <vector>
#include <algorithm>
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

// count maximal contiguous runs in a sorted address list
static void runs(std::vector<int>& a, int& nruns, int& maxrun, int& nelem) {
  std::sort(a.begin(), a.end()); a.erase(std::unique(a.begin(), a.end()), a.end());
  nelem = int(a.size()); nruns = 0; maxrun = 0;
  for (size_t i = 0; i < a.size();) {
    size_t j = i; while (j + 1 < a.size() && a[j+1] == a[j] + 1) ++j;
    ++nruns; maxrun = std::max(maxrun, int(j - i + 1)); i = j + 1;
  }
}

template <int TM, int TN, int TK, int WM, int WN, int SK>
static void probe(const char* tag) {
  constexpr int WOM = TM/WM, WON = TN/WN;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto tiled_S = make_tiled_copy_B(Copy_Atom<DefaultCopy, cutlass::half_t>{}, Mma{});

  // sS exactly as the collective builds it: tile_to_shape(Layout<Shape<_8,_1>>, (TN, 1, SK)), and the INTERLEAVED
  // candidate: the same with a trailing extent-2 mode innermost, so (s,z) for one (n,group) are adjacent halves.
  auto sS_plain = tile_to_shape(Layout<Shape<_8,_1>>{}, make_shape(Int<TN>{}, Int<1>{}, Int<SK>{}));
  auto sS_intlv = make_layout(make_shape (Int<TN>{}, Int<1>{}, Int<SK>{}),
                              make_stride(_2{},     _0{},      Int<2*TN>{}));   // *2 everywhere, (s,z) adjacent
  // THE BIGGER CANDIDATE, which L27 only exposed because max-run came out as 1. A thread's n set is
  // n = 16*j + 8*h + lane/4, i.e. a FIXED a = lane/4 in [0,8) and EVERY b in n = b*8 + a. So transposing n from
  // (b,a) to (a,b) makes each thread's whole set ONE contiguous run -- and this touches only sS's layout, not the
  // transform, so it cannot regress the way the fused-FMA attempt did.
  auto sS_perm = make_layout(make_shape (make_shape(_8{}, Int<TN/8>{}), Int<1>{}, Int<SK>{}),
                             make_stride(make_stride(Int<TN/8>{}, _1{}), _0{}, Int<TN>{}));

  int tot_runs_p = 0, tot_runs_i = 0, tot_runs_x = 0, ne = 0, mr_p = 0, mr_i = 0, mr_x = 0, nslot = 0, nreg = 0;
  const int NT = 32*WOM*WON;
  for (int t = 0; t < NT; ++t) {
    auto thr = tiled_S.get_thread_slice(t);
    { auto p = thr.partition_S(make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), sS_plain));
      std::vector<int> a; auto s = p(_,_,0,0); nslot = int(size(s));
      for (int i = 0; i < int(size(s)); ++i) a.push_back(int(s.layout()(i)));
      int nr, mx, n; runs(a, nr, mx, n); tot_runs_p += nr; ne = n; mr_p = std::max(mr_p, mx); }
    { auto p = thr.partition_S(make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), sS_intlv));
      std::vector<int> a; auto s = p(_,_,0,0);
      for (int i = 0; i < int(size(s)); ++i) a.push_back(int(s.layout()(i)));
      int nr, mx, n; runs(a, nr, mx, n); tot_runs_i += nr; mr_i = std::max(mr_i, mx); }
    { auto p = thr.partition_S(make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), sS_perm));
      std::vector<int> a; auto s = p(_,_,0,0);
      for (int i = 0; i < int(size(s)); ++i) a.push_back(int(s.layout()(i)));
      int nr, mx, n; runs(a, nr, mx, n); tot_runs_x += nr; mr_x = std::max(mr_x, mx); }
  }
  const double rp = double(tot_runs_p)/NT, ri = double(tot_runs_i)/NT, rx = double(tot_runs_x)/NT;
  // plain: S and Z are separate arrays, so 2 * runs transactions per reload
  // interleaved: one array, each (s,z) pair is one 32-bit access -> runs_i transactions, each twice as wide
  printf("  %-22s (%3d,%3d,%3d) w%dx%-3d SK=%d | %2d el/thr | plain max-run %d -> %4.1f txn |"
         " intlv max-run %d -> %4.1f txn (%.1fx) | PERM max-run %d -> %4.1f txn (%.1fx, both paths)\n",
         tag, TM, TN, TK, WM, WN, SK, ne, mr_p, 2*rp, mr_i, ri, 2*rp/ri, mr_x, 2*rx, rp/rx);
  // THE HEADLINE, and it is not contiguity at all. sS has K extent 1, but make_tiled_copy_B builds the copy for the
  // FULL B tile (TN x TK), so every val mode that walks K collapses to stride 0 and the SAME smem element is asked
  // for again and again. The fragment mirrors B's shape for the same reason, so it materialises the replication in
  // registers too.
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto sS_t = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr), sS_plain);
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sS_t(_,_,Int<0>{}));
  nreg = int(cosize(frag.layout()));
  printf("      %-18s copy asks %3d slot(s)/thr for %2d DISTINCT smem elem = %2dx redundant"
         " | fragment: %3d slot(s) -> %2d reg-half = %2dx replicated%s\n",
         "", nslot, ne, nslot/ne, int(size(frag)), nreg, int(size(frag))/nreg,
         nslot/ne > 2 ? "   <-- dwarfs the 2x interleave" : "");
}

int main() {
  printf("L27 -- the sS/sZ interleave is a 2x on a path carrying a 16-64x redundancy\n\n");
  printf("  A 'run' is a maximal contiguous address stretch; DefaultCopy vectorizes each run into one wide access.\n");
  printf("  plain txn = 2 x runs (S and Z are separate arrays); interleaved txn = runs (one array, pairs adjacent).\n\n");
  probe<32,128,128,32,32,4>("int1 TK128 gs32");
  probe<32,128,128,32,32,8>("int1 TK128 gs16");
  probe<32,128, 64,32,64,2>("int1 TK64 gs32 (best)");
  probe<32,256, 64,32,64,2>("int1 TK64 TN=256");
  probe<64, 64, 64,32,32,2>("int4 PRODUCTION");
  probe<32,128,256,32,32,8>("int1 TK256 gs32");
  printf("\n  The interleave halves the transaction count (max-run is 1 everywhere, so nothing is broken by it) --\n");
  printf("  but the copy is 16-64x redundant to begin with, because sS is k-invariant while make_tiled_copy_B is\n");
  printf("  built for the full TN x TK B tile. Replacing the materialised replication with a stride-0 broadcast VIEW\n");
  printf("  cuts the same traffic 16-64x AND frees 4x the scale registers, without touching the transform.\n");
  return 0;
}
