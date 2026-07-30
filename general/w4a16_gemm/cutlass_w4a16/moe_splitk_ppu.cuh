// Split-K for the GROUPED mixed-input GEMM, done as S ordinary grouped launches plus one light reduce.
//
// WHY THIS FORM AND NOT cutlass::Semaphore. The serial split-K kernel that already exists
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

// One split-K grouped GEMM.
//
// ptr_D_all: device array of L*slices output pointers, slice s owning entries [s*L, (s+1)*L). Slice s must
// point into partials + s*total_rows*N so the reduce below finds them contiguously. When slices == 1 this
// degenerates to a single ordinary launch and the caller may pass the plain ptr_D.
template <QuantMode QuantOp, int TM, int TN, int TK, int WM, int WN, int Stages,
          class ElementB = cutlass::int4b_t, class PlaneB2 = void>
void launch_splitk(const cutlass::half_t* A, const ElementB* B, const cutlass::half_t* scales,
                   const cutlass::half_t* zeros,
                   cutlass::half_t* D,                  // final output, total_rows x N
                   cutlass::half_t* partials,           // slices * total_rows * N, unused when slices == 1
                   cutlass::half_t** ptr_D_all, DStride* stride_D, int const* group_M,
                   int m, int n, int k, int L, int group_size, int slices,
                   GroupShape* gsd, GroupShape const* gsh_full,   // host shapes with the FULL k
                   std::vector<GroupShape>& gsh_slice,            // scratch: host shapes with the slice k
                   GroupShape* gsd_slice,                         // device copy of gsh_slice
                   int const* group_row_offsets, int64_t total_rows,
                   char* ws, size_t ws_bytes, hggcStream_t stream,
                   const PlaneB2* B2 = nullptr) {
  const char* why = "";
  if (!splitk_ok(k, slices, group_size, TK, &why)) {
    std::printf("[moe_splitk] S=%d refused: %s\n", slices, why);
    ++moe_grouped_ppu::moeg_fail_count();
    return;
  }

  if (slices == 1) {
    moe_grouped_ppu::filter_and_run<QuantOp, TM, TN, TK, WM, WN, Stages, ElementB, PlaneB2>(
        A, B, scales, zeros, ptr_D_all, stride_D, group_M, m, n, k, L, group_size,
        gsd, gsh_full, group_row_offsets, ws, ws_bytes, stream, B2);
    return;
  }

  int const ks = k / slices;
  int const scale_k_slice = (group_size > 0) ? (ks / group_size) : 0;

  // Per-expert problem shapes with the SLICE k. The M values are unchanged -- split-K does not touch the
  // ragged row distribution.
  gsh_slice.resize(L);
  for (int e = 0; e < L; ++e)
    gsh_slice[e] = cute::make_shape(int(cute::get<0>(gsh_full[e])), n, ks);
  hggcMemcpy(gsd_slice, gsh_slice.data(), sizeof(GroupShape) * L, hggcMemcpyHostToDevice);

  // Element offsets of slice s. B's offline layout is tile-major in K, so a whole number of 256-tiles is a
  // flat element offset; A is row-major so the slice is a COLUMN range, which is why launch() needs k_full
  // for the row pitch and only the pointer moves here.
  int64_t const b_slice_elems  = int64_t(ks) * n;                        // codes, not bytes
  int64_t const s_slice_elems  = int64_t(scale_k_slice) * n;

  for (int s = 0; s < slices; ++s) {
    moe_grouped_ppu::filter_and_run<QuantOp, TM, TN, TK, WM, WN, Stages, ElementB, PlaneB2>(
        A + int64_t(s) * ks,
        B + int64_t(s) * b_slice_elems,
        scales + int64_t(s) * s_slice_elems,
        zeros ? (zeros + int64_t(s) * s_slice_elems) : nullptr,
        ptr_D_all + int64_t(s) * L, stride_D, group_M,
        m, n, ks, L, group_size,
        gsd_slice, gsh_slice.data(), group_row_offsets,
        ws, ws_bytes, stream,
        B2 ? (B2 + int64_t(s) * b_slice_elems) : nullptr,
        /*k_full=*/k);
  }

  int64_t const elems = total_rows * int64_t(n);
  int const threads = 256;
  int const blocks  = int(std::min<int64_t>((elems + threads - 1) / threads, 4096));
  // cutlass::half_t wraps __half with the same layout; the merge is typed on the raw type so it stays
  // locally testable (see moe_splitk_reduce.cuh).
  moeg_splitk_reduce<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<__half*>(D), reinterpret_cast<__half const*>(partials), elems, slices);
}

}  // namespace moe_splitk_ppu
