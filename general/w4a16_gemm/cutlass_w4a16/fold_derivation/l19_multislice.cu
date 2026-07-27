// L19 -- the fp16 IDENTITY check, for any number of 32B slices. (l17_fp16_identity.cu was merged in here; it only
// covered a single slice and reported the wider cubes as SKIPPED, which is the case that actually needed deciding.)
//
// WHY THE fp16 IDENTITY IS THE STRONGEST CHECK AVAILABLE. fp16 has no converter and no offline relayout, so
//     gmem tile (n,k) -> AIU (linear) -> smem -> L2^-1 -> fragment -> mma wants (n',k')
// must come out as n'==n, k'==k everywhere. Nothing exists in that path to hide a compensating error: if L2 were
// wrong there is no converter emission and no offline permutation left to cancel it. And it needs no box.
//
// L2 was derived for ONE 32B slice by stripping both swizzle terms from ppu_tsm_ld_swzl_sim:
//     vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2)          <- XOR
//     vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8   <- ROTATE, ssv in {0,4,2,6} by slice
// For slice 0 ssv is 0, so stripping it was a no-op and the single-slice result could not distinguish "the rotate
// is part of the swizzle the AIU write cancels" from "the rotate is part of the logical address". At slices 1..3 it
// matters, and guessing is how the last several rounds went wrong.
//
// THE ARBITER. fp16 has no converter and no offline relayout, so the whole chain must come out as the identity.
// That is a decisive, box-free test: try each candidate and keep the one that yields 0 mismatch. Two independent
// binary choices, four combinations:
//     ROT   -- is the cross-slice rotate stripped (part of the swizzle) or kept (part of the address)?
//     ORDER -- do the copy instances walk slice-major or row-block-major?
// If exactly one combination gives identity, that is the answer and it is measured, not chosen.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l19_multislice.cu -o l19 && ./l19
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
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

static inline int ssv(int slice) { return (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2; }

template <int TN, int TK, int WM, int WN>
static void probe(const char* tag) {
  constexpr int WOM = 32 / WM, WON = TN / WN, HPW = 2;
  constexpr int SLICE_HALF = 16;                                  // one 32B slice = 16 half
  constexpr int NSLICE = TK / SLICE_HALF;
  constexpr int RPI = WON * 16, NROWBLK = TN / RPI;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto frag = Mma{}.get_thread_slice(0).partition_fragment_B(sB);
  const int NS = int(size(frag));
  // pi = frag.layout()^-1, and cute HAS this: right_inverse(Layout). It used to be a hand-rolled loop here (and in
  // seven sibling files); l30_should_have_been_cute.cu verifies the two agree on three shapes including int4's
  // production one. Worth replacing beyond tidiness: l20 is the file scheduled to become the production offline.
  auto pi = right_inverse(frag.layout());
  const int NINST = NS / 8;
  printf("  %-18s fp16 TN=%3d TK=%3d w%dx%d | %d slice(s), %d row-block(s), frag=%d -> %d instance(s)\n",
         tag, TN, TK, WM, WN, NSLICE, NROWBLK, NS, NINST);
  if (NINST != NSLICE * NROWBLK) { printf("      instance count does not factor as slices x row-blocks -- skipped\n\n"); return; }

  for (int rot = 0; rot < 2; ++rot)
    for (int order = 0; order < 2; ++order) {
      long bad = 0, tot = 0;
      for (int i = 0; i < NINST; ++i) {
        const int slice = order == 0 ? (i % NSLICE) : (i / NROWBLK);      // 0 = slice-major, 1 = row-block-major
        const int rblk  = order == 0 ? (i / NSLICE) : (i % NROWBLK);
        for (int t = 0; t < 32 * WOM * WON; ++t) {
          const int lane = t % 32, w = t / 32, warp_n = w / WOM;
          auto part = Mma{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{})));
          for (int v = 0; v < 4; ++v) {
            const int row  = (v / 2) * 8 + lane / 4;
            const int vec  = (row % 4) * 2 + (v % 2);
            // The sim's slice-internal offset is line*32 + vecf*4 + lane%4. With the rotate STRIPPED that reduces
            // exactly to row*8 + (v%2)*4 + lane%4, so the word within the row is (v%2)*4 + lane%4 -- independent of
            // the slice. My first version left the (row%4)*8 term inside vec instead of letting row*8 absorb it,
            // which is why the 1-slice control failed even though l17 passes on the same config. That control is
            // there precisely to catch this.
            int word_in_row;
            if (!rot) { word_in_row = (v % 2) * 4 + lane % 4; }
            else {
              // the rotate mixes the row bits into the vec index, so the row's own words are no longer contiguous
              const int vecf = (vec + ssv(slice)) % 8;
              word_in_row = vecf * 4 + lane % 4 - (row % 4) * 8;
              if (word_in_row < 0 || word_in_row >= 8) { ++tot; ++bad; continue; }  // leaves the row: cannot be right
            }
            for (int h = 0; h < HPW; ++h) {
              const int flat = i * 8 + v * HPW + h;
              if (flat < 0 || flat >= NS) continue;
              auto c = part(pi(flat));
              const int n_have = rblk * RPI + 16 * warp_n + row;
              const int k_have = slice * SLICE_HALF + word_in_row * HPW + h;
              ++tot;
              if (int(get<0>(c)) != n_have || int(get<1>(c)) != k_have) ++bad;
            }
          }
        }
      }
      printf("      rot=%-4s order=%-14s : %ld / %ld mismatch %s\n", rot ? "keep" : "strip",
             order == 0 ? "slice-major" : "rowblock-major", bad, tot, bad ? "" : "<-- IDENTITY");
    }
  printf("\n");
}

int main() {
  printf("L19 -- the fp16 identity, over 1/2/4 slices, arbitrating the cross-slice term\n\n");
  probe<64,  16, 32, 32>("1 slice (control)");
  probe<128, 16, 32, 32>("1 slice wide N");
  probe<64,  32, 32, 32>("2 slices");
  probe<64,  64, 32, 32>("4 slices (dense fp16 shape)");
  probe<128, 64, 32, 32>("4 slices wide N");
  return 0;
}
