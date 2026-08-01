// L98 -- WHICH Swizzle<B,M,S> MAKES THE SCALE READ CONFLICT-FREE? Swept against the collective's own layout, not
// reasoned about.
//
// WHY A SWIZZLE AND NOT PADDING. `PPU_SCALE_PAD` added halfs to the group stride and LOST: an additive pad turns the
// address into non-power-of-two arithmetic, and the multiply costs more than the conflict. A swizzle is an XOR on
// address bits -- free -- and cute composes it onto the layout, so the g2s cp.async's partition_D, the s2r
// make_tiled_copy_B read, and the packed decode's explicit `sS(n, Int<G>{}, stage) =` stores ALL derive from the one
// object. That is the property that matters here: this task's recurring defect is two places having to agree, and a
// composed layout removes the second place.
//
// WHY IT IS WORTH DOING, with the numbers that bound it. TODO.md prices the whole per-group scale reload at 7.3%
// (SK_QUANT=0 deletes it) and prefetching -- which removes only the WAITING -- recovered 0.7%. So nine tenths of that
// channel's cost is work, not stall. Bank conflicts are work: a 4-way conflict is four shared-pipe services for one
// instruction, which prefetch provably cannot reach. De-conflicting therefore attacks the 6.6% that prefetch left,
// and is bounded above by 7.3%.
//
// The metric is DISTINCT ADDRESSES per bank, not lanes per bank: the scale fragment is deliberately k-broadcast, so
// lanes DO share addresses, and lanes sharing an address broadcast rather than conflict. l94 (2) makes the same point
// and reports today's map as 4-way on 4 banks; this file reproduces that as the sweep's baseline, so a disagreement
// shows up as the two files disagreeing rather than as a silent wrong answer.
//
// The scale tile is NOT an AIU tile -- SmemCopyAtomScale is Copy_Atom<DefaultCopy, half_t> (pinned by l95) -- so it
// carries none of the swzl read/write pairing constraints that make A and B unswizzleable.
//
//   nvcc -std=c++17 -x cu -arch=sm_80 -w -I fold_derivation/stub_inc -I <actlize>/include -o /tmp/l98 l98...cu
#include <cstdio>
#include <set>
#include <vector>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/numeric_types.h"
using namespace cute;
using cutlass::half_t;

// The stub mma, byte-identical to the collective's by l95's static_asserts.
struct F16Atom {};
namespace cute {
template <> struct MMA_Traits<F16Atom> {
  using ValTypeD=float; using ValTypeA=half_t; using ValTypeB=half_t; using ValTypeC=float;
  using Shape_MNK=Shape<_16,_16,_16>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
}
using SmemCopyAtomS = Copy_Atom<DefaultCopy, half_t>;
using AtomScale   = Layout<Shape<_8,_1>>;                        // the collective's SmemLayoutAtomScale
// EVERY WIDTH, not the one I first hardcoded. The conflict map runs through the mma B operand's TV layout, which
// depends on Scale_TileN, so a swizzle chosen at TN=128 is not a swizzle for TN=32 -- and the first version of this
// file swept only 128 and the collective then applied the winner to all three. WarpN is 16 throughout, so the warp
// count is TN/16.
template <int TN> struct Cfg {
  using TMma  = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<_1,Int<TN/16>,_1>>, Tile<_16,Int<TN>,_64>>;
  using Plain = decltype(tile_to_shape(AtomScale{}, make_shape(Int<TN>{}, _8{}, _2{})));
};

// distinct addresses per bank over warp 0, for a layout given as a callable offset map
template <int TN, class LayoutT>
static std::pair<int,int> bank_profile(LayoutT const& lay, size_t* n_addr = nullptr) {
  auto sS      = make_tensor(make_smem_ptr(static_cast<half_t*>(nullptr)), typename Cfg<TN>::Plain{});
  auto tiled_s = make_tiled_copy_B(SmemCopyAtomS{}, typename Cfg<TN>::TMma{});
  auto cS      = make_identity_tensor(shape(sS));
  // ONE VALUE INDEX AT A TIME: a bank conflict is a property of ONE instruction, i.e. the 32 lanes' addresses for a
  // single element of the fragment. My first version aggregated every value of every lane and reported 64-way on 512
  // addresses, which is not a conflict measurement at all -- l94 (2) uses tCsC(0,0,0,0), one address per lane, and the
  // cross-check I printed against it is what caught this. The profile is the WORST value index.
  int mx = 0, used = 0; size_t addrs = 0;
  {
    auto thr0 = tiled_s.get_thread_slice(0);
    auto tC0  = thr0.partition_S(cS);
    int const nv = int(size<0>(tC0)) * int(size<1>(tC0));
    for (int v = 0; v < nv; ++v) {
      std::set<int> per_bank[32], all;
      for (int t = 0; t < 32; ++t) {
        auto thr  = tiled_s.get_thread_slice(t);
        auto tCsC = thr.partition_S(cS);
        auto c    = tCsC(v % int(size<0>(tCsC)), v / int(size<0>(tCsC)), 0, 0);
        int const n = int(get<0>(c)), g = int(get<1>(c));
        int const w = (int(lay(n, g, 0)) * 2) / 4;               // halfs -> bytes -> 4-byte bank words
        per_bank[w & 31].insert(w);
        all.insert(w);
      }
      int m = 0, u = 0;
      for (int b = 0; b < 32; ++b) { if (!per_bank[b].empty()) ++u; if (int(per_bank[b].size()) > m) m = int(per_bank[b].size()); }
      if (m > mx) { mx = m; used = u; addrs = all.size(); }
    }
  }
  if (n_addr) *n_addr = addrs;
  return {mx, used};
}

// One swept candidate. B bits at position M+S are xored into the B bits at position M, on the HALF offset.
template <int TN, int B, int M, int S>
static void try_one(int& best_way, int& best_b, int& best_m, int& best_s) {
  auto lay = composition(Swizzle<B,M,S>{}, typename Cfg<TN>::Plain{});
  size_t addrs = 0;
  auto p = bank_profile<TN>(lay, &addrs);
  bool const better = (p.first < best_way);
  if (better) { best_way = p.first; best_b = B; best_m = M; best_s = S; }
  if (p.first <= 2 || better)
    std::printf("      Swizzle<%d,%d,%d>%*s %d-way on %2d banks (%2zu addrs)%s\n",
                B, M, S, (B<10&&M<10&&S<10)?2:0, "", p.first, p.second, addrs, better ? "   <-- best so far" : "");
}

template <int TN>
static void sweep() {
  size_t base_addrs = 0;
  auto base = bank_profile<TN>(typename Cfg<TN>::Plain{}, &base_addrs);
  int bw = base.first, bb = -1, bm = -1, bs = -1;
#define TRY(b,m,s) try_one<TN,b,m,s>(bw, bb, bm, bs);
  TRY(1,0,1)
  TRY(1,0,2)
  TRY(1,0,3)
  TRY(1,0,4)
  TRY(1,0,5)
  TRY(1,0,6)
  TRY(1,0,7)
  TRY(1,0,8)
  TRY(1,1,1)
  TRY(1,1,2)
  TRY(1,1,3)
  TRY(1,1,4)
  TRY(1,1,5)
  TRY(1,1,6)
  TRY(1,1,7)
  TRY(1,1,8)
  TRY(1,2,1)
  TRY(1,2,2)
  TRY(1,2,3)
  TRY(1,2,4)
  TRY(1,2,5)
  TRY(1,2,6)
  TRY(1,2,7)
  TRY(1,2,8)
  TRY(1,3,1)
  TRY(1,3,2)
  TRY(1,3,3)
  TRY(1,3,4)
  TRY(1,3,5)
  TRY(1,3,6)
  TRY(1,3,7)
  TRY(1,3,8)
  TRY(1,4,1)
  TRY(1,4,2)
  TRY(1,4,3)
  TRY(1,4,4)
  TRY(1,4,5)
  TRY(1,4,6)
  TRY(1,4,7)
  TRY(2,0,2)
  TRY(2,0,3)
  TRY(2,0,4)
  TRY(2,0,5)
  TRY(2,0,6)
  TRY(2,0,7)
  TRY(2,0,8)
  TRY(2,1,2)
  TRY(2,1,3)
  TRY(2,1,4)
  TRY(2,1,5)
  TRY(2,1,6)
  TRY(2,1,7)
  TRY(2,1,8)
  TRY(2,2,2)
  TRY(2,2,3)
  TRY(2,2,4)
  TRY(2,2,5)
  TRY(2,2,6)
  TRY(2,2,7)
  TRY(2,2,8)
  TRY(2,3,2)
  TRY(2,3,3)
  TRY(2,3,4)
  TRY(2,3,5)
  TRY(2,3,6)
  TRY(2,3,7)
  TRY(2,4,2)
  TRY(2,4,3)
  TRY(2,4,4)
  TRY(2,4,5)
  TRY(2,4,6)
  TRY(3,0,3)
  TRY(3,0,4)
  TRY(3,0,5)
  TRY(3,0,6)
  TRY(3,0,7)
  TRY(3,0,8)
  TRY(3,1,3)
  TRY(3,1,4)
  TRY(3,1,5)
  TRY(3,1,6)
  TRY(3,1,7)
  TRY(3,1,8)
  TRY(3,2,3)
  TRY(3,2,4)
  TRY(3,2,5)
  TRY(3,2,6)
  TRY(3,2,7)
  TRY(3,3,3)
  TRY(3,3,4)
  TRY(3,3,5)
  TRY(3,3,6)
  TRY(3,4,3)
  TRY(3,4,4)
  TRY(3,4,5)
#undef TRY
  std::printf("   TN=%-3d  plain %d-way on %2d banks   ->  ", TN, base.first, base.second);
  if (bb < 0) std::printf("NOTHING beats it (keep the plain layout)\n");
  else        std::printf("Swizzle<%d,%d,%d>  %d-way\n", bb, bm, bs, bw);
}

int main() {
  std::printf("== l98: the best Swizzle<B,M,S> for the scale read, PER Scale_TileN ==\n");
  std::printf("   (the metric is DISTINCT bank-word addresses per bank over one warp for ONE value index, worst over\n"
              "    value indices -- lanes sharing an address broadcast, and l94 (2) reports 4-way on 4 banks at TN=128)\n");
  sweep<32>();
  sweep<64>();
  sweep<128>();

  // The two views must agree before AND after: partition_extra_inputs uses (n, group, stage) and
  // partition_extra_mma_info uses (n, 1, stage*Scale_TileK + group) on the same buffer.
  {
    using P = Cfg<128>::Plain;
    using C = decltype(tile_to_shape(AtomScale{}, make_shape(_128{}, _1{}, Int<16>{})));
    int bad_plain = 0, bad_swz = 0;
    for (int s2 = 0; s2 < 2; ++s2) for (int g = 0; g < 8; ++g) for (int n = 0; n < 128; ++n) {
      if (int(P{}(n,g,s2)) != int(C{}(n,0,s2*8+g))) ++bad_plain;
      if (int(composition(Swizzle<2,3,5>{}, P{})(n,g,s2)) != int(composition(Swizzle<2,3,5>{}, C{})(n,0,s2*8+g))) ++bad_swz;
    }
    std::printf("   two views agree: %d bad plain, %d bad under Swizzle<2,3,5>\n", bad_plain, bad_swz);
  }
  // 16 B cp.async contiguity: MBase=3 leaves the low three bits alone, so 8-half runs keep one xor.
  {
    auto swz = composition(Swizzle<2,3,5>{}, Cfg<128>::Plain{});
    int broken = 0, misaligned = 0;
    for (int s2 = 0; s2 < 2; ++s2) for (int g = 0; g < 8; ++g) for (int n0 = 0; n0 < 128; n0 += 8) {
      int const base = int(swz(n0,g,s2));
      if (base % 8) ++misaligned;
      for (int j = 1; j < 8; ++j) if (int(swz(n0+j,g,s2)) != base + j) ++broken;
    }
    std::printf("   16 B cp.async under Swizzle<2,3,5>: %d non-contiguous, %d misaligned\n", broken, misaligned);
  }
  // BIJECTION, on the swizzle actually selected. The first version spot-checked Swizzle<2,1,6> -- a constant left over
  // from an earlier edit -- which proves nothing about the one that ships.
  {
    std::set<int> img;
    for (int s2 = 0; s2 < 2; ++s2) for (int g = 0; g < 8; ++g) for (int n = 0; n < 128; ++n)
      img.insert(int(composition(Swizzle<2,3,5>{}, Cfg<128>::Plain{})(n,g,s2)));
    std::printf("   Swizzle<2,3,5> is a permutation of the allocation: %zu distinct images of %d coords, cosize %d\n",
                img.size(), 128*8*2, int(cosize(Cfg<128>::Plain{})));
  }
  return 0;
}
