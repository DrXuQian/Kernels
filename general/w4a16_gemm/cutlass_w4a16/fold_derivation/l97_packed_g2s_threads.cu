// L97 -- DOES A THREAD BEYOND THE COPY'S THREAD LAYOUT WRITE OUTSIDE ITS TILE? The last unexamined write into
// smem_scale_raw, and the only known undefined behaviour left on the packed path.
//
// GmemTiledCopyScalePacked is make_tiled_copy(atom, Layout<Shape<Int<Scale_TileN>,_1>>, Layout<Shape<_1,_16>>), i.e.
// Scale_TileN = 128 threads. Its get_slice(thread_idx) is called by ALL size(TiledMma) = 256 threads of the CTA. If
// cute does not wrap the index, threads 128..255 partition at n = 128..255 of a tile whose N extent is 128 -- and
// SmemLayoutScaleRawStaged's stride is (16, 1, Scale_TileN*16), so n = 128 is EXACTLY the first byte of the next
// stage. With Stages = 2 that means half the CTA scribbles over the stage the other half is about to read.
//
// This is worth its own file because rowC's failure is intermittent: it was bad=128, then bad=724, then MATCH with no
// semantic source change (the =&r fix was refuted by rebuilding with "=r" and passing). A latent out-of-range write
// whose damage depends on scheduling behaves exactly like that.
//
//   nvcc -std=c++17 -x cu -arch=sm_80 -w -I fold_derivation/stub_inc -I <actlize>/include -o /tmp/l97 l97...cu
#include <cstdio>
#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
using namespace cute;

int main() {
  constexpr int TN = 128, UNIT = 16, STAGES = 2, THREADS = 256;
  auto tiled = make_tiled_copy(Copy_Atom<DefaultCopy, uint8_t>{},
                               Layout<Shape<Int<TN>, _1>>{},
                               Layout<Shape<_1, Int<UNIT>>>{});
  std::printf("== l97: %d-thread copy sliced by %d threads ==\n", int(size(tiled)), THREADS);

  // the staged raw tile, exactly SmemLayoutScaleRawStaged
  auto lay = make_layout(make_shape(Int<TN>{}, Int<UNIT>{}, Int<STAGES>{}),
                         make_stride(Int<UNIT>{}, _1{}, Int<TN * UNIT>{}));
  std::printf("   tile layout "); print(lay); std::printf("   cosize %d bytes\n", int(cosize(lay)));

  uint8_t* base = nullptr;
  int worst = -1, first_oob = -1, oob = 0;
  for (int t = 0; t < THREADS; ++t) {
    auto thr = tiled.get_slice(t);
    Tensor s  = make_tensor(make_smem_ptr(base), lay);
    Tensor td = thr.partition_D(s);
    // the byte this thread's slice starts at, for stage 0
    int const off = int(reinterpret_cast<intptr_t>(&td(0, 0, 0, 0)));
    if (t < 4 || t == TN - 1 || t == TN || t == TN + 1 || t == THREADS - 1)
      std::printf("   thread %3d -> byte %6d%s\n", t, off,
                  off >= int(cosize(lay)) ? "   <-- PAST THE WHOLE TILE"
                  : (off >= TN * UNIT ? "   <-- inside the NEXT STAGE" : ""));
    if (off >= TN * UNIT) { ++oob; if (first_oob < 0) first_oob = t; }
    if (off > worst) worst = off;
  }
  std::printf("\n   threads whose stage-0 slice lands at or past stage 1: %d (first = %d)\n", oob, first_oob);
  std::printf("   highest start byte %d against a tile of %d bytes\n", worst, int(cosize(lay)));
  bool const wraps = (oob == 0);
  std::printf("== l97 %s: cute %s the thread index ==\n", wraps ? "PASS" : "FAIL",
              wraps ? "WRAPS (out-of-range threads are harmless)"
                    : "does NOT wrap -- threads beyond the layout write into other stages");
  return wraps ? 0 : 1;
}
