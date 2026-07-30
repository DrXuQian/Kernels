// The two pieces of grouped split-K that do NOT need the PPU cutlass stack: the merge kernel and the
// legality rule. Separated so both can be gated LOCALLY on any CUDA device -- the merge is what produces the
// final numbers, so "compiles on the box" is not an acceptable level of assurance for it.
#pragma once

#include <cstdio>
#include <algorithm>
#include <cstdint>
#if defined(__HGGCCC__)
#include <hggc_fp16.h>
#else
#include <cuda_fp16.h>
#endif

namespace moe_splitk_ppu {

// D[i] = sum over slices of partial[s][i]. fp32 accumulate, one fp16 rounding at the end.
// Grid-stride over total_rows*N so the launch shape is independent of the problem.
//
// TYPED AS __half, NOT cutlass::half_t, and deliberately: an elementwise fp16 sum needs nothing from cutlass,
// and actlize gates CUTLASS_HOST_DEVICE on __HGGCCC__ -- so a cutlass-typed kernel cannot be COMPILED AND RUN
// locally at all (every half_t method degrades to host `inline`). The caller reinterpret_casts, which is free:
// cutlass::half_t wraps __half with the same layout. This is the difference between a merge that has been
// measured against an fp64 reference and one that has only been observed to compile.
// STATIC, because this header is now included by SIX translation units (main plus one per generated config)
// and a __global__ function has EXTERNAL linkage: the split into per-config units turned a working header into
// `multiple definition of moeg_splitk_reduce` at link time. A __global__ cannot be `inline` in CUDA, so internal
// linkage is the fix -- six copies of a grid-stride elementwise add cost nothing.
static __global__ void moeg_splitk_reduce(__half* __restrict__ D,
                                   __half const* __restrict__ partials,
                                   int64_t elems, int slices) {
  int64_t const stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < elems; i += stride) {
    float acc = 0.f;
    for (int s = 0; s < slices; ++s) acc += __half2float(partials[int64_t(s) * elems + i]);
    D[i] = __float2half(acc);
  }
}

// Is S a legal split for this problem? Every slice must still satisfy the constraints the collective and the
// offline layout impose on K, so the answer is not "any divisor".
inline bool splitk_ok(int k, int slices, int group_size, int tile_k, const char** why = nullptr) {
  auto no = [&](const char* m) { if (why) *why = m; return false; };
  if (slices < 1) return no("slices < 1");
  if (slices == 1) { if (why) *why = ""; return true; }
  if (k % slices) return no("K not divisible by the slice count");
  int const ks = k / slices;
  // The AIU offline interleave is [K/256][N][256], so a slice boundary must fall on a 256-element tile or the
  // slice's B pointer lands mid-tile.
  if (ks % 256) return no("slice K not a multiple of the 256-element offline tile");
  // A slice must still be a whole number of scale groups, and at least one K-tile of the mainloop.
  if (group_size > 0 && ks % group_size) return no("slice K not a whole number of scale groups");
  if (tile_k > 0 && ks % tile_k) return no("slice K not a whole number of Block_K");
  if (why) *why = "";
  return true;
}

}  // namespace moe_splitk_ppu
