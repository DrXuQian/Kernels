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

// Is S a legal split? WITH IN-KERNEL SPLIT-K THE RULE COLLAPSED. The host-slicing form needed a slice to start
// on a 256-element offline tile (its B pointer moved) and to cover whole scale groups; the in-kernel form walks
// k-tiles z, z+S, z+2S with gA/gB untouched, so neither applies. What is left is that the k-tile count divides by
// S, because the kernel's k_tile_count is a ceil and an indivisible count would let the last slice step past the
// end of the coordinate space. A whole k-tile always covers whole scale groups (the collective already requires
// gs <= Block_K), so the group size does not enter.
inline bool splitk_ok(int k, int slices, int tile_k, const char** why = nullptr) {
  auto no = [&](const char* m) { if (why) *why = m; return false; };
  if (slices < 1) return no("slices < 1");
  if (slices == 1) { if (why) *why = ""; return true; }
  if (tile_k <= 0) return no("Block_K unknown");
  int const kt = (k + tile_k - 1) / tile_k;
  if (kt % slices) return no("k-tile count not divisible by the slice count");
  if (why) *why = "";
  return true;
}

}  // namespace moe_splitk_ppu
