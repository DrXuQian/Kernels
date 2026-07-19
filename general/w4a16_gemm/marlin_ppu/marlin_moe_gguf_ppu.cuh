#ifndef MARLIN_MOE_GGUF_PPU_CUH
#define MARLIN_MOE_GGUF_PPU_CUH
// GGUF Q4_K grouped (MoE) GEMM on the PPU, built on the verified dense W4A16 Marlin.
//
// WHY THIS EXISTS: qwen35moe routes its expert FFN through ggml_mul_mat_id, which our dense kernels do not
// cover at all -- and for an A3B model that path IS the prefill compute. See ppu-gemv-routing-coverage.
//
// THE STRUCTURAL POINT that makes this cheap to build: sort the tokens by expert and pad each expert's row
// segment up to a multiple of the m-tile (16*thread_m_blocks). Then every m-tile belongs to exactly ONE
// expert, and the expert dimension collapses onto Marlin's existing `parallel` dimension -- the only thing
// that varies per m-tile is the B / s / mins base pointer. No change to the mainloop, the scheduler, or
// any of the group machinery that took this long to get right.
//
// THE PADDING IS NOT OPTIONAL. An expert segment that does not fill whole m-tiles puts two experts in one
// tile, and the tile uses one B for both -- silently wrong rows, no error. The same trap bit the public
// DeepGEMM .so port (see llamacpp-public-so-decoupling: per-expert row segments must be padded to 128).
//
// KNOWN SHAPE PROBLEM, from the fp16 AIU MoE GEMM (see memory ppu-moe-gemm-design). That kernel is
// weight/memory bound, so the quantity it minimizes is the TOTAL NUMBER OF M-TILES, sum_e ceil(rows_e/BMR)
// -- weight traffic scales with it, and its ragged default is BMR=128. Marlin's m-tile is 16*MARLIN_MAX_MB
// = 32 rows, A QUARTER of that, so each expert's weight is re-streamed ~4x as often. MARLIN_MAX_MB is
// capped at 2 by register spills, so this is structural to the Marlin path, not a tuning knob.
// => measure B's DRAM traffic here, not just TFLOP/s; expect weight-bound before compute-bound.
//
// That kernel also MASKS rather than pads (true per-expert bounds, `if (row < m_end)` on the store, padz
// for the real OOB, and over-reading into the next expert's rows is harmless since those rows are never
// written). Padding as done here additionally costs rows in A and a larger gather. Worth revisiting if the
// m-tile problem forces a rewrite anyway.
//
// v1 issues one marlin_cuda per expert over that expert's contiguous rows. Correct, and the baseline the
// persistent version has to beat. NEXT STEP, from the fp16 MoE work that reached DeepGemm parity
// (ppu-moe-aiu-persist-matches-deepgemm): a single persistent launch, grid = #CU * blocks-per-CU, with a
// grid-stride queue over a flattened (expert, m-tile, n-tile) space and the expert looked up from a cumsum.
// That removes the per-expert launch (~2us each, and an A3B layer has many) and the tail of every small
// expert, which is exactly what a per-expert loop cannot fix.
//
// AFFINE: v1 uses the fused min. The dense measurements say the split (symmetric + Asum @ m') is worth
// ~7.6 MFU points at gs=32, and it carries over -- Asum is per TOKEN and independent of the expert, so it
// is computed once for all tokens and only the second GEMM is grouped. Deliberately not done here: the
// fused path is the one with a correctness test behind it, and the split needs its own reference.
#include "marlin_gguf_ppu.cuh"
#include "marlin_moe_kernel_ppu.cuh"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>

namespace marlin_moe_gguf_ppu {

using marlin_gguf_ppu::ceildiv;

// Routing plan, host side. Tokens are grouped by expert into padded segments.
struct MoePlan {
  int m_tile;                    // rows per m-tile = 16 * thread_m_blocks of the chosen config
  int padded_rows;               // total rows after per-expert padding
  std::vector<int> row_src;      // padded_rows: source token, or -1 for pad rows
  std::vector<int> expert_off;   // num_experts+1: start row of each expert's padded segment
};

// rows_expert[i] = expert of token i (one expert per token; for top-k, call once per k with the
// corresponding assignment, which is what mul_mat_id does anyway).
inline MoePlan moe_plan(const int* rows_expert, int num_rows, int num_experts, int m_tile) {
  MoePlan p;
  p.m_tile = m_tile;
  std::vector<std::vector<int>> by_e(num_experts);
  for (int i = 0; i < num_rows; i++) by_e[rows_expert[i]].push_back(i);
  p.expert_off.assign(num_experts + 1, 0);
  for (int e = 0; e < num_experts; e++) {
    p.expert_off[e] = (int) p.row_src.size();
    for (int t : by_e[e]) p.row_src.push_back(t);
    // pad up to a whole number of m-tiles -- see the note above, this is a correctness requirement
    while (p.row_src.size() % m_tile) p.row_src.push_back(-1);
  }
  p.expert_off[num_experts] = (int) p.row_src.size();
  p.padded_rows = (int) p.row_src.size();
  return p;
}

// Gather A rows into the sorted/padded order. Pad rows are zeroed so their output is harmless.
__global__ void gather_rows(const half* __restrict__ A, half* __restrict__ Ag,
                            const int* __restrict__ row_src, int padded_rows, int K) {
  const long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;   // in int4 units
  const long long nvec = (long long) padded_rows * (K / 8);
  if (idx >= nvec) return;
  const int r = (int) (idx / (K / 8)), off = (int) (idx % (K / 8));
  const int src = row_src[r];
  int4 v = make_int4(0, 0, 0, 0);
  if (src >= 0) v = reinterpret_cast<const int4*>(A)[(long long) src * (K / 8) + off];
  reinterpret_cast<int4*>(Ag)[idx] = v;
}

// Scatter the padded output back to token order; pad rows are dropped.
__global__ void scatter_rows(const half* __restrict__ Cg, half* __restrict__ C,
                             const int* __restrict__ row_src, int padded_rows, int N) {
  const long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  const long long nvec = (long long) padded_rows * (N / 8);
  if (idx >= nvec) return;
  const int r = (int) (idx / (N / 8)), off = (int) (idx % (N / 8));
  const int dst = row_src[r];
  if (dst < 0) return;
  reinterpret_cast<int4*>(C)[(long long) dst * (N / 8) + off] =
      reinterpret_cast<const int4*>(Cg)[idx];
}

// B/s/mins are per expert, contiguous: B + e*(K/16)*(N*16/32)*4 ints, s/mins + e*(K/gs)*N halves.
// Ag/Cg are the gathered buffers (padded_rows x K, padded_rows x N).
inline int moe_cuda(const half* Ag, const int* B, half* Cg, const half* s, const half* mins,
                    const MoePlan& plan, int N, int K, void* workspace, int gs,
                    int num_experts, int max_par = 128) {
  const size_t b_stride = (size_t) (K / 16) * (N * 16 / 32) * 4;   // ints per expert
  const size_t s_stride = (size_t) (K / gs) * N;                   // halves per expert
  for (int e = 0; e < num_experts; e++) {
    const int r0 = plan.expert_off[e], rows = plan.expert_off[e + 1] - r0;
    if (rows == 0) continue;
    const int ret = marlin_gguf_ppu::marlin_cuda(
        Ag + (size_t) r0 * K, B + e * b_stride, Cg + (size_t) r0 * N,
        (void*) (s + e * s_stride), mins ? (void*) (mins + e * s_stride) : nullptr,
        rows, N, K, workspace, gs, 0, 0, -1, -1, -1, max_par);
    if (ret) return ret;
  }
  return 0;
}

// SINGLE-LAUNCH grouped GEMM: every expert's tiles in one grid. v1 above issues one marlin_cuda per
// expert, which pays a launch (~2us) and a wave tail PER EXPERT -- and an A3B layer has many experts each
// holding few rows, so that is the worst case for it. Here the expert is looked up per m-tile from
// mtile_expert and only the B / s / mins bases move; Marlin's own scheduler already flattens
// (k_tile, n_tile, parallel) across a grid of #CU and stripes it, so one launch over the padded rows IS
// the persistent grouped kernel.
//
// mtile_expert must be built at moe_m_tile()'s granularity -- the same one moe_plan padded to.
inline int moe_cuda_fused(const half* Ag, const int* B, half* Cg, const half* s, const half* mins,
                          const MoePlan& plan, int N, int K, void* workspace, int gs,
                          int num_experts, const int* d_mtile_expert, int max_par = 128) {
  const long long b_stride_i4 = (long long) (K / 16) * (N * 16 / 32) / 4;   // int4 per expert
  const long long s_stride_i4 = (long long) (K / gs) * N / 8;               // int4 per expert
  return marlin_moe_kernel_ppu::marlin_cuda(
      Ag, B, Cg, (void*) s, mins ? (void*) mins : nullptr,
      plan.padded_rows, N, K, workspace, gs, 0, 0, -1, -1, -1, max_par,
      d_mtile_expert, b_stride_i4, s_stride_i4);
}

// Expert of each m-tile, at the KERNEL's tile granularity -- which is NOT necessarily the plan's.
// MARLIN_MAX_MB is 2, so the kernel tiles at 32 rows while a plan may pad to 64. Padding to a larger
// multiple is safe (every 32-row tile still sits inside one expert), but building this array at the
// PLAN's granularity while the kernel indexes at its own silently shifts every expert. The plan's m_tile
// must therefore be a multiple of the kernel's, and the array is built at the kernel's.
inline std::vector<int> moe_mtile_expert(const MoePlan& plan, int num_experts, int kernel_m_tile) {
  if (plan.m_tile % kernel_m_tile) {
    printf("moe_mtile_expert: plan m_tile=%d is not a multiple of the kernel's %d -- an m-tile would "
           "straddle two experts\n", plan.m_tile, kernel_m_tile);
    return {};
  }
  std::vector<int> v(plan.padded_rows / kernel_m_tile);
  for (int e = 0; e < num_experts; e++)
    for (int r = plan.expert_off[e]; r < plan.expert_off[e + 1]; r += kernel_m_tile)
      v[r / kernel_m_tile] = e;
  return v;
}

}  // namespace marlin_moe_gguf_ppu
#endif
