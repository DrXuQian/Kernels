// Split-K for the GROUPED mixed-input GEMM, done as S ordinary grouped launches plus one light reduce.
//
// THE SLICES MUST RUN CONCURRENTLY, OR THIS BUYS NOTHING. S launches on ONE stream serialise, so the grid at
// any instant is unchanged and the only effect is S times the launch cost -- measured: time grew ~linearly in S
// (23.49 -> 44.51 -> 69.70 -> 121.37 us for S = 1,2,4,8) which is exactly what serialisation predicts. The whole
// point of split-K is that all S slices' CTAs are resident AT THE SAME TIME. So launch_splitk takes an array of
// streams and the caller joins them; the alternative, one launch with the slice on gridDim.z, needs a new
// GemmUniversal specialisation and is the thing to write only if the stream form shows the idea has legs.
//
// Two things had to move for streams to overlap at all:
//   * the ragged m-tile prefix write in launch() is a BLOCKING hggcMemcpy, which serialises the host and with it
//     every stream. It depends only on the per-expert M values, which K-slicing does not touch, so the caller
//     writes it once and passes prefix_ready=true.
//   * the workspace is shared by all slices. Safe here because the non-persistent grouped kernel only READS the
//     prefix from it; it is not scratch.
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
                   char* ws, size_t ws_bytes,
                   hggcStream_t const* streams, int num_streams,   // slice s runs on streams[s % num_streams]
                   const PlaneB2* B2 = nullptr) {
  const char* why = "";
  if (!splitk_ok(k, slices, group_size, TK, &why)) {
    std::printf("[moe_splitk] S=%d refused: %s\n", slices, why);
    ++moe_grouped_ppu::moeg_fail_count();
    return;
  }

  // num_streams <= 0 (or a null array) means the DEFAULT stream for every slice, i.e. the serialised form.
  // That is the current default on purpose: hggcStreamCreate exists in fold_derivation/stub_inc as a no-op that
  // returns success WITHOUT setting the stream, and nowhere in actlize -- so the concurrent form cannot be
  // validated locally at all, and shipping it as the default would repeat the cuda_fp16.h mistake.
  bool const have_streams = (streams != nullptr && num_streams > 0);
  hggcStream_t const s0 = have_streams ? streams[0] : hggcStream_t(0);
  if (slices == 1) {
    moe_grouped_ppu::filter_and_run<QuantOp, TM, TN, TK, WM, WN, Stages, ElementB, PlaneB2>(
        A, B, scales, zeros, ptr_D_all, stride_D, group_M, m, n, k, L, group_size,
        gsd, gsh_full, group_row_offsets, ws, ws_bytes, s0, B2);
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

  // The ragged m-tile prefix, written ONCE. Identical for every slice (K-slicing leaves the M values alone),
  // and blocking, so leaving it inside the per-slice launch would serialise the streams it exists to overlap.
  {
    int const TMv = TM;
    std::vector<int> pfx(L + 1);
    pfx[0] = 0;
    for (int e = 0; e < L; ++e)
      pfx[e + 1] = pfx[e] + int((int(cute::get<0>(gsh_full[e])) + TMv - 1) / TMv);
    if (ws) hggcMemcpy(ws, pfx.data(), sizeof(int) * (L + 1), hggcMemcpyHostToDevice);
  }

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
        ws, ws_bytes, have_streams ? streams[s % num_streams] : hggcStream_t(0),
        B2 ? (B2 + int64_t(s) * b_slice_elems) : nullptr,
        /*k_full=*/k, /*prefix_ready=*/true);
  }

  // Join before the merge: the reduce reads every slice's partials.
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  int64_t const elems = total_rows * int64_t(n);
  int const threads = 256;
  int const blocks  = int(std::min<int64_t>((elems + threads - 1) / threads, 4096));
  // cutlass::half_t wraps __half with the same layout; the merge is typed on the raw type so it stays
  // locally testable (see moe_splitk_reduce.cuh).
  moeg_splitk_reduce<<<blocks, threads, 0, s0>>>(
      reinterpret_cast<__half*>(D), reinterpret_cast<__half const*>(partials), elems, slices);
}

}  // namespace moe_splitk_ppu
