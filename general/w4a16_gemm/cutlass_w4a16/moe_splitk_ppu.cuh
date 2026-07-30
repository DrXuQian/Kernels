// Split-K for the GROUPED mixed-input GEMM, done as S ordinary grouped launches plus one light reduce.
//
// ONE LAUNCH, THE SLICE ON gridDim.z. The first version of this sliced K on the HOST -- S calls to
// filter_and_run -- and that raises no occupancy at all: consecutive launches on one stream serialise, so the
// grid AT ANY INSTANT is unchanged and the only effect is S times the launch cost. Measured 23.49 / 44.51 /
// 69.70 / 121.37 us for S = 1,2,4,8, linear in S, which is exactly what serialisation predicts. All S slices
// have to be resident TOGETHER, which means one launch.
//
// The kernel side is small because cute already has the piece: SplitkCoordIterator walks k-tiles z, z+S, z+2S ...
// so gA/gB are untouched and only the traversal changes. Two consequences worth stating:
//   * the B pointer never moves, so the host-slicing constraint that a slice fall on the 256-element offline tile
//     boundary is GONE. What remains is that the k-tile count must divide by S (the kernel's count is a ceil).
//   * the split is STRIDED, not contiguous, so the per-group scale must be addressed from the k-tile coordinate
//     rather than by advancing a pointer per tile. The dense split-K path shares this mainloop, which is evidence
//     but not proof -- hence the S>1 vs S=1 output comparison in the bench.
//
// WHY A SEPARATE REDUCE AND NOT cutlass::Semaphore. The serial split-K kernel that already exists
// (ppu_aiu_gemm_mixed_input_splitk_serial.hpp) is dense-only, and its epilogue chains the slices through
// gmem in fp16: slice s+1 reads slice s's output, adds, writes back. That serialises the tail and rounds S
// times. Slicing K on the HOST instead needs no new GemmUniversal specialisation at all -- each slice is a
// complete grouped GEMM over k/S -- and the merge becomes one elementwise kernel that accumulates the S
// partials in fp32. This is the shape the user asked for ("splitk 可以在另外一个 kernel 轻量 reduce"), and
// it is strictly better numerically than the serial chain: fold_derivation/l70_splitk_fp16_merge.cu measured
// the fp16 chain at 1 ulp for S<=4 and 2 ulp for S>=8, whereas a single fp32 accumulation of S fp16 partials
// is correctly rounded once.
//
// WHAT IT COSTS. The partial buffer is S * total_rows * N halfs, and the reduce reads all of it and writes
// 1/S of it. At decode (total_rows = experts) that is nothing; at prefill it is real and is the reason this
// is an axis to measure rather than a default. Activation traffic also multiplies: every slice re-reads its
// own A and B slices, so total weight traffic is unchanged but A is read once per slice per n-tile.
//
// WHAT IT BUYS. S times the CTAs. That is the whole point: at decode the grouped launch is 512 CTAs on a
// 72-CU part with every warp resident at once (acu: 13.65 of a theoretical 18 warps/CU), so there is no
// second wave and no latency hiding. Whether more CTAs actually convert into throughput is a measurement on
// the real grouped shape -- which is what this file exists to make possible, after a dense proxy at m=8
// turned out to run 64 CTAs and answer nothing.
#pragma once

#include <cstdio>
#include <vector>
#include "moe_grouped_ppu.cuh"
#include "moe_splitk_reduce.cuh"   // the merge kernel and the legality rule (locally gated)

namespace moe_splitk_ppu {

using moe_grouped_ppu::GroupShape;
using moe_grouped_ppu::DStride;
using moe_grouped_ppu::QuantMode;

// One split-K grouped GEMM: a single launch with the slice on gridDim.z, then the merge.
//
// ptr_D_all: device array of L*slices output pointers, slice z of expert e at index e + z*L, pointing into
// `partials`. stride_D_all likewise holds L*slices entries (the L strides repeated), because the epilogue
// indexes both arrays with the shifted coordinate. When slices == 1 the caller may pass the plain L arrays.
template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages,
          class ElementB = cutlass::int4b_t, class PlaneB2 = void>
void launch_splitk(const cutlass::half_t* A, const ElementB* B, const cutlass::half_t* scales,
                   const cutlass::half_t* zeros,
                   cutlass::half_t* D,                  // final output, total_rows x N
                   cutlass::half_t* partials,           // slices * total_rows * N, unused when slices == 1
                   cutlass::half_t** ptr_D_all, DStride* stride_D_all, int const* group_M,
                   int m, int n, int k, int L, int group_size, int slices,
                   GroupShape* gsd, GroupShape const* gsh,
                   int const* group_row_offsets, int64_t total_rows,
                   char* ws, size_t ws_bytes, hggcStream_t stream,
                   bool /*unused, was a_row_broadcast*/ = false,
                   const PlaneB2* B2 = nullptr) {
  if (slices < 1) slices = 1;

  moe_grouped_ppu::filter_and_run<QuantOp, TM, TN, TK, WM, WN, Stages, ElementB, PlaneB2>(
      A, B, scales, zeros, ptr_D_all, stride_D_all, group_M, m, n, k, L, group_size,
      gsd, gsh, group_row_offsets, ws, ws_bytes, stream, B2,
      /*k_full=*/-1, /*prefix_ready=*/false, /*splitk=*/slices, false);

  if (slices == 1) return;

  int64_t const elems = total_rows * int64_t(n);
  int const threads = 256;
  int const blocks  = int(std::min<int64_t>((elems + threads - 1) / threads, 4096));
  // cutlass::half_t wraps __half with the same layout; the merge is typed on the raw type so it stays locally
  // testable (see moe_splitk_reduce.cuh).
  moeg_splitk_reduce<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<__half*>(D), reinterpret_cast<__half const*>(partials), elems, slices);
}

}  // namespace moe_splitk_ppu
