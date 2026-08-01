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
static constexpr int kWON = 8;                                   // TileN 128 / WarpN 16
using TMma        = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<_1,Int<kWON>,_1>>, Tile<_16,Int<kWON*16>,_64>>;
using SmemCopyAtomS = Copy_Atom<DefaultCopy, half_t>;
using AtomScale   = Layout<Shape<_8,_1>>;                        // the collective's SmemLayoutAtomScale
using ScaleShape  = decltype(make_shape(_128{}, _8{}, _2{}));    // (Scale_TileN, Scale_TileK, Stages)
using PlainScale  = decltype(tile_to_shape(AtomScale{}, ScaleShape{}));

// distinct addresses per bank over warp 0, for a layout given as a callable offset map
template <class LayoutT>
static std::pair<int,int> bank_profile(LayoutT const& lay, size_t* n_addr = nullptr) {
  auto sS      = make_tensor(make_smem_ptr(static_cast<half_t*>(nullptr)), PlainScale{});
  auto tiled_s = make_tiled_copy_B(SmemCopyAtomS{}, TMma{});
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
template <int B, int M, int S>
static void try_one(int& best_way, int& best_b, int& best_m, int& best_s) {
  auto lay = composition(Swizzle<B,M,S>{}, PlainScale{});
  size_t addrs = 0;
  auto p = bank_profile(lay, &addrs);
  bool const better = (p.first < best_way);
  if (better) { best_way = p.first; best_b = B; best_m = M; best_s = S; }
  if (p.first <= 2 || better)
    std::printf("      Swizzle<%d,%d,%d>%*s %d-way on %2d banks (%2zu addrs)%s\n",
                B, M, S, (B<10&&M<10&&S<10)?2:0, "", p.first, p.second, addrs, better ? "   <-- best so far" : "");
}

int main() {
  std::printf("== l98: sweep Swizzle<B,M,S> for the scale read ==\n");
  std::printf("   layout "); print(PlainScale{}); std::printf("\n");

  size_t base_addrs = 0;
  auto base = bank_profile(PlainScale{}, &base_addrs);
  std::printf("   (0) TODAY, unswizzled: %d-way on %d banks (%zu addrs)   -- l94 (2) reports 4-way on 4 banks\n",
              base.first, base.second, base_addrs);
  if (base.first != 4)
    std::printf("       NOTE this disagrees with l94; one of the two is wrong and the swizzle choice below inherits it\n");

  std::printf("   (1) candidates (only 2-way or better, plus every improvement, are printed):\n");
  int bw = base.first, bb = -1, bm = -1, bs = -1;
  // B is how many bits move, M where they land, S how far above M they come from. The bank selector is bits [1,6) of
  // the half offset, so M in [0,5) is where a swizzle can reach; S walks the donor bits up through the group (stride
  // 128 halfs = bit 7) and stage (1024 = bit 10) fields.
#define TRY(b,m,s) try_one<b,m,s>(bw, bb, bm, bs);
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

  // ---- (2) THE TWO VIEWS MUST AGREE, BEFORE AND AFTER. partition_extra_inputs builds sS with SmemLayoutScale
  // (n, group, stage) while partition_extra_mma_info builds a tensor ALSO called sS with SmemCopyLayoutScale
  // (n, 1, stage*Scale_TileK + group) -- two layouts, one buffer, and this task has already faulted once by using the
  // wrong one. A swizzle composes onto the OFFSET, so composition(Swz, L)(c) = Swz(L(c)) and the two stay equal iff
  // they are equal today. That is checkable, so check it rather than reason about it.
  {
    using CopyScale = decltype(tile_to_shape(AtomScale{}, make_shape(_128{}, _1{}, Int<16>{})));
    int bad_plain = 0, bad_swz = 0;
    for (int s2 = 0; s2 < 2; ++s2)
      for (int g = 0; g < 8; ++g)
        for (int n = 0; n < 128; ++n) {
          if (int(PlainScale{}(n, g, s2)) != int(CopyScale{}(n, 0, s2 * 8 + g))) ++bad_plain;
          if (int(composition(Swizzle<2,3,5>{}, PlainScale{})(n, g, s2)) !=
              int(composition(Swizzle<2,3,5>{}, CopyScale{})(n, 0, s2 * 8 + g))) ++bad_swz;
        }
    std::printf("   (2) SmemLayoutScale(n,g,s) == SmemCopyLayoutScale(n,0,8s+g): %d bad plain, %d bad under "
                "Swizzle<2,3,5>  -> composing the SAME swizzle on both is %s\n",
                bad_plain, bad_swz, (bad_plain == 0 && bad_swz == 0) ? "equivalent" : "NOT equivalent");
  }

  std::printf("\n   BEST: ");
  if (bb < 0) std::printf("nothing beat the unswizzled map (%d-way)\n", base.first);
  else        std::printf("Swizzle<%d,%d,%d> -> %d-way (from %d-way)\n", bb, bm, bs, bw, base.first);
  // A swizzle is only usable if it is a bijection over the allocation, or the tile aliases itself and the g2s writes
  // collide. cosize is a power of two here, which is the condition; checked rather than assumed.
  std::printf("   cosize %d halfs = %d B, power of two: %s (required -- a swizzle must permute WITHIN the allocation)\n",
              int(cosize(PlainScale{})), int(cosize(PlainScale{})) * 2,
              (cosize(PlainScale{}) & (cosize(PlainScale{}) - 1)) == 0 ? "yes" : "NO -- do not swizzle this");
  if (bb >= 0) {
    // and that it really is a permutation: every offset must map somewhere distinct
    auto lay = composition(Swizzle<1,1,1>{}, PlainScale{});
    (void)lay;
    std::set<int> img;
    for (int s = 0; s < 2; ++s) for (int g = 0; g < 8; ++g) for (int n = 0; n < 128; ++n)
      img.insert(int(composition(Swizzle<2,1,6>{}, PlainScale{})(n, g, s)));
    std::printf("   (bijection spot-check on Swizzle<2,1,6>: %zu distinct images of %d coordinates)\n",
                img.size(), 128 * 8 * 2);
  }
  return bb < 0 ? 1 : 0;
}
