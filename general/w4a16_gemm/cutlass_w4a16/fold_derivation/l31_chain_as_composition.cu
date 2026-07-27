// L31 -- PASSES, 7/7 configs, 0 mismatch. Link 3 of 6 is cute.
//
// The offline's logical side is now ONE expression instead of four hand-spliced lines:
//     TV    = thrfrg_B(sB).layout().compose(btile, _).compose(right_inverse(thr_layout_vmnk), _)
//     Frag  = composition(pi, emit_all)
//     logical(t, i) = TV(make_coord(t, Frag(i)))          -> n*TK + k
//
// THREE REAL BUGS ON THE WAY, all found by PRINTING candidates next to ground truth rather than by swapping
// expressions and diffing totals. Four rounds went to the latter and produced nothing; the former produced each
// answer in one step. Same technique that localised the ladder's 10 points and the int2 pairing bug.
//
//   1. WARP ORDER IS OPPOSITE between the two views, readable straight off the layouts:
//          thr_layout_vmnk       = (32,2,2,1):(1,32,64,0)            linear t: warpM stride 32, warpN 64
//          thrfrg_B(sB) thr part = ((4,8),(2,1)):((2,64),(1024,0))   first mode (lane,(warpN,warpM))
//      t=32 is warpM=1/warpN=0 with true base 0, but thrfrg_B reads warpN=1 and adds 1024; t=64 mirrors it. Six of
//      seven configs were off by a constant n, and (32,128,128) w32x32 passed only because with two warps the swap
//      is a no-op -- a single passing config is not evidence.
//   2. right_inverse(vmnk) MUST BE COMPOSED AS A LAYOUT, not evaluated. Applying it as an integer inside make_coord
//      changed nothing at all, because evaluating it discards the tuple structure and thrfrg_B re-decomposes the
//      integer colex, bringing the wrong order straight back.
//   3. .layout() BEFORE .compose(). Composing the TENSOR returns a tensor over sB's nullptr, and evaluating it
//      DEREFERENCES -- that was the segfault, and it printed nothing because stdout was buffered.
//
// get_layoutB_TV() cannot be used directly here for two reasons its source makes plain: its reference is
// make_layout(make_shape(tile_size_mnk<1>(), tile_size_mnk<2>())) with NO stride, hence COLUMN-major (codomain
// n + N*k, not n*TK + k), and it is sized by the MMA tile (WON*16) rather than the block TN. Applying its two
// composes to OUR sB is what works.
//
// L31 -- link 3 of 6: the offline placement as a COMPOSITION of layouts instead of hand-spliced index arithmetic.
//
// WHERE THE 813 HAND-WRITTEN LINES COME FROM. Four of the chain's links are now cute Layouts (SmemLayoutB_Logical,
// LogicalTV, MixGemmEmit, pi = right_inverse), but l20's tile_map still splices them together BY HAND:
//
//     e    = MixGemmEmit<Bits>::index(j, v);
//     flat = inst * VEC + e;
//     c    = part(pi(flat));
//     m[(row*8 + wd)*CPW + j] = get<0>(c)*TK + get<1>(c);
//
// Every one of those lines is a composition cute can do, and hand-splicing is exactly what has to be re-derived per
// format. cute's composition(A, B)(i) == A(B(i)), so the whole logical side is one expression:
//
//     emit_all = append(emit, Layout<NINST, VEC>)        add the instance mode, stride VEC
//     Loc      = composition(part.layout(), composition(pi, emit_all))
//              : (code bits, vreg bits, inst) -> n*TK + k
//
// part.layout() already maps a fragment index to an OFFSET in an (TN,TK):(TK,1) tensor, and that offset IS n*TK+k, so
// no coordinate unpacking is needed either.
//
// THE GATE: Loc must reproduce l20's hand-written map at every point, for every config in the tree. If it does, the
// derived offline becomes "walk the domain, ask Loc" -- and adding a format stops meaning re-deriving index math.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l31_chain_as_composition.cu -o l31 && ./l31
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include <cstdio>
#include <vector>
using namespace cute;
using cutlass::MixGemmEmit;

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

template <int Bits, int TM, int TN, int TK, int WM, int WN>
static bool check(const char* tag) {
  constexpr int WOM = TM/WM, WON = TN/WN, CPW = 32/Bits;
  constexpr int contig = TK*Bits/8, F = contig >= 32 ? 1 : 32/contig, Ng = TN/F;
  constexpr int RPI = WON*16, VEC = 4*CPW;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag)), NINST = NS / VEC;
  if (NINST < 1) { printf("  %-30s int%d: fragment %d < one instance %d -- skipped\n", tag, Bits, NS, VEC); return true; }

  auto pi   = right_inverse(frag.layout());
  auto emit = typename MixGemmEmit<Bits>::EmitLayout{};
  // the instance mode: fragment index = inst*VEC + e
  auto emit_all = append(emit, make_layout(Int<NINST>{}, Int<VEC>{}));

  // THE THREAD MODE MUST BE PART OF THE LAYOUT. thr.partition_B(sB) bakes the thread's base into the ITERATOR, not
  // the layout, so part.layout()(0) is 0 for EVERY thread while the real coordinate is not -- which showed up as a
  // constant offset for every t != 0 on the first version of this file. get_layoutB_TV() is the thread-inclusive
  // form, (thr, val) -> tile offset, so nothing is lost and no per-thread slicing is needed at all.
  // THE DERIVATION, taken from cute's own get_layoutB_TV() source rather than guessed. That function builds
  //     thrfrg_B(ref_B).compose(btile, _).compose(thridx_2_thrid, _)
  // and its only problem for us is ref_B: make_layout(make_shape(tile_size_mnk<1>(), tile_size_mnk<2>())) has NO
  // stride, so it is COLUMN-major, and it is sized by the MMA tile (WON*16) rather than the block TN. Apply the same
  // two composes to OUR sB and the codomain is sB's own offset = n*TK + k.
  //
  // What each compose fixes, read off the printed layouts:
  //   thr_layout_vmnk       = (32,2,2,1):(1,32,64,0)      linear t: warpM at stride 32, warpN at 64
  //   thrfrg_B(sB) thr part = ((4,8),(2,1)):((2,64),(1024,0))   first mode is (lane,(warpN,warpM)) -- warpN FIRST
  // The two warp orders are OPPOSITE, which is the whole bug: t=32 is warpM=1/warpN=0 (true base 0) but thrfrg_B
  // reads it as warpN=1 (+1024), and t=64 is the mirror. thridx_2_thrid = right_inverse(thr_layout_vmnk) is exactly
  // that reordering -- and feeding its INTEGER output into make_coord (my previous attempt) throws the tuple
  // structure away, so thrfrg_B re-decomposes it colex and the wrong order comes straight back. It has to be
  // composed as a LAYOUT, not evaluated and re-packed.
  auto vmnk  = Mma{}.get_thr_layout_vmnk();
  auto btile = make_tile(_, make_tile(make_layout(make_shape (size<1>(vmnk), size<2>(vmnk)),
                                                  make_stride(     Int<0>{},      Int<1>{})),   // B ignores ThrM
                                      _));
  auto TV    = Mma{}.thrfrg_B(sB).layout().compose(btile, _).compose(right_inverse(vmnk), _);  // .layout()! composing
  // the TENSOR instead returns a tensor over sB's nullptr, and evaluating it DEREFERENCES -- that was the segfault.
  auto Frag  = composition(pi, emit_all);                           // (code, vreg, inst) -> fragment index
  long bad = 0; int shown = 0;
  for (int t = 0; t < 32*WOM*WON; ++t) {
    auto id   = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{},Int<TK>{})));
    for (int inst = 0; inst < NINST; ++inst)
      for (int v = 0; v < 4; ++v)
        for (int j = 0; j < CPW; ++j) {
          const int e = MixGemmEmit<Bits>::index(j, v), flat = inst*VEC + e;
          if (flat < 0 || flat >= NS) continue;
          auto c = id(pi(flat));
          const int hand = int(get<0>(c))*TK + int(get<1>(c));      // exactly l20's line
          const int comp = int(TV(make_coord(t, Frag(j + CPW * v + VEC * inst))));   // one cute expression
          ++bad;
          if (hand == comp) --bad;
          else if (shown < 4) {
            printf("      MISMATCH inst%d vreg%d code%2d: hand %5d, composition %5d\n", inst, v, j, hand, comp);
            ++shown; }
        }
  }
  printf("  %-30s int%d (%d,%d,%d) w%dx%d F=%d | frag %3d = %d inst x %3d | %ld mismatch %s\n",
         tag, Bits, TM, TN, TK, WM, WN, F, NS, NINST, VEC, bad,
         bad ? "<-- composition is NOT the hand-spliced map" : "<-- ONE cute expression replaces the splice");
  return bad == 0;
}

int main() {
  printf("L31 -- the offline's logical side as composition(part, composition(pi, emit))\n\n");
  bool ok = true;
  ok &= check<1, 32,128,128, 32,32>("int1 fold TK=128");
  ok &= check<1, 64,128, 64, 32,64>("int1 fold TK=64 (bitpack)");
  ok &= check<2, 64, 64, 64, 32,32>("int2 fold TK=64");
  ok &= check<2, 64,128, 64, 32,32>("int2 fold wideN");
  ok &= check<4, 64, 64, 64, 32,32>("int4 PRODUCTION");
  ok &= check<4, 64, 64,128, 32,32>("int4 TK=128");
  ok &= check<1, 64, 64,256, 32,32>("int1 unfolded TK=256");
  printf("\n  What is left after this is the PHYSICAL side -- (row, wd) from LogicalTV plus the instance stride -- and\n");
  printf("  the sub-byte packing loop, which is an engine concern and correctly stays a loop.\n");
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
