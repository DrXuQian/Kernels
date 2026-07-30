// Grouped MIXED-INPUT GEMM kernel for actlize v1.0.0 -- the actlize analogue of trtllm's MoeFCGemm<mixed-input Mma>.
//
// This is a NEW GemmUniversal specialization that combines:
//   - the GroupScheduler + GroupProblemShape ragged-M machinery from ppu_aiu_gemm_array_group.hpp, and
//   - the mixed-input collective's scale-aware drive (load_init + operator) from ppu_aiu_gemm_mixed_input.hpp.
// The existing array_group kernel drives the PLAIN (BatchArray) collective with a gA/gB-passed interface and
// has no scales; this one drives the mixed-input collective (which carries scale/zero internally), so W4A16
// grouped GEMM works. enable_if keys on the mixed-input schedule + a GroupProblemShape, so it does not collide
// with the single-GEMM mixed-input specialization (which requires a rank-3/4 ProblemShape).
//
// STEP 2a (this file): DEGENERATE / uniform-M. The mixed-input collective still slices A by l_coord with a
// uniform L-stride (mA_mkl(_,_,l_coord)), so per-expert M must be uniform -- equivalent to step 1 but routed
// through the GroupScheduler. Purpose: prove the scheduler+collective wiring compiles and runs.
// STEP 2b (next): ragged A -- give the mixed-input collective a per-expert A base (ptr_A_array[l_coord] +
// M_e) so token counts can vary. B/scale/zero stay L-strided (uniform per-expert) and are untouched.
//
// UNTESTED on box (private SDK; cannot compile here). Integration points most likely to need a fix are marked
// [Q1]..[Q4] inline.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cute/ppu_util.hpp"
#include "cutlass/utils.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class GemmUniversal<
  ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_,
  cute::enable_if_t<
    cute::is_base_of_v<KernelAiuMultistageMixedInput, typename CollectiveMainloop_::DispatchPolicy::Schedule>
    && isGroupProblemShape_v<ProblemShape_>>>          // <- mixed-input schedule AND a group problem shape
{
public:
  using ProblemShape = ProblemShape_;
  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 3
             or rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "UnderlyingProblemShape should be <M,N,K> or <M,N,K,L>");

  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static constexpr bool IsGroupedGemmKernel = isGroupProblemShape_v<ProblemShape>;
  using TileScheduler = typename detail::TileSchedulerSelector<
      GroupScheduler, ArchTag, TileShape, ClusterShape, ProblemShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  struct SharedStorage {
    union SharedTensorStorage {
      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
      MainloopSharedStorage mainloop;
      EpilogueSharedStorage epilogue;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  // PPU_MAXREG: cap registers per thread by asking __launch_bounds__ for more resident blocks. device_kernel.h
  // passes MinBlocksPerMultiprocessor straight into __launch_bounds__, so the compiler must fit
  // 131072 / (blocks * threads) registers. Expressed as a REGISTER target and converted here, because the block
  // count that implies depends on MaxThreadsPerBlock -- 10 blocks means 102 registers at 128 threads and 51 at 256,
  // and hardcoding the block count would silently over-constrain the wider tiles.
  //
  // Whether it pays is a separate question: at S=1 the grid supplies 4 blocks/CU for a 16x64 tile while 106
  // registers already allow 9, so registers are NOT the binding limit there. Measured, not assumed.
#if defined(PPU_MAXREG) && (PPU_MAXREG > 0)
  static constexpr uint32_t MinBlocksPerMultiprocessor =
      (131072u / (uint32_t(PPU_MAXREG) * MaxThreadsPerBlock)) > 0
          ? (131072u / (uint32_t(PPU_MAXREG) * MaxThreadsPerBlock)) : 1u;
#else
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
#endif
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr int MinWorkspaceAlignment = 16;   // [Q1] array_group takes this from an outer scope; define locally

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    int probe = 0;   // debug: 0=normal; 1=ROUTING probe (skip GEMM, write expert+1 to every output element)
    int const* group_M = nullptr;   // device [L] per-expert M_e; O(L) decode of blockIdx.x -> (expert,m_tile)
    int mtiles_uniform = 0;         // >0: all experts have this many m-tiles -> O(1) decode; 0: ragged, scan group_M
    // IN-KERNEL SPLIT-K. >1 slices the K loop across gridDim.z so all S slices' CTAs are resident in ONE launch.
    // Doing it on the host instead -- S launches on a stream -- serialises them and raises no occupancy at all:
    // measured 23.49 / 44.51 / 69.70 / 121.37 us for S = 1,2,4,8, i.e. linear in S.
    //
    // The split is STRIDED (slice z takes k-tiles z, z+S, z+2S ...), which is what cute's SplitkCoordIterator
    // does, so the B pointer never moves and there is NO constraint that a slice fall on the 256-element offline
    // tile boundary -- that constraint belonged to the host-slicing form and is gone.
    //
    // With splitk > 1 the epilogue's ptr_D/stride_D arrays must hold L*splitk entries: slice z of expert e writes
    // plane e + z*L, so the S partials are contiguous for the merge kernel.
    int splitk = 1;
  };
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    TileSchedulerParams scheduler;
    void* workspace;
    int probe = 0;
    int const* group_M = nullptr;
    int mtiles_uniform = 0;
    int splitk = 1;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    ProblemShape problem_shapes = args.problem_shape;
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0)
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    KernelHardwareInfo hw_info{args.hw_info.device_id, cu_count};

    // Only the GroupScheduler needs workspace. The mixed-input vectorized epilogue needs none (its
    // get_workspace_size is (ProblemShape,Args) and returns 0 for this non-persistent path).
    void* scheduler_workspace = workspace;

    constexpr uint32_t NumEpilogueSubTiles = 1;
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        problem_shapes, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);

    // [Q2] The mixed-input collective + epilogue want ONE (M,N,K,L) shape (K,N uniform across experts; L=groups)
    // to compute scale_k=K/gs, the (N,scale_k,L) scale layout, and the L-strided D. Build a representative
    // rank-4 shape from the group shapes so those L-strided planes span all experts.
    auto host0 = problem_shapes.get_host_problem_shape(0);
    auto rep_mnkl = cute::make_shape(int(get<0>(host0)), int(get<1>(host0)), int(get<2>(host0)), problem_shapes.groups());

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(rep_mnkl, args.mainloop, /*workspace=*/nullptr),
      CollectiveEpilogue::to_underlying_arguments(rep_mnkl, args.epilogue, /*workspace=*/nullptr),
      hw_info, scheduler, workspace, args.probe, args.group_M, args.mtiles_uniform,
      args.splitk < 1 ? 1 : args.splitk
    };
  }

  static bool can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGrouped || args.mode == GemmUniversalMode::kArray;
  }
  static int get_workspace_size(Arguments const& args) {
    // GroupScheduler workspace only (epilogue needs none on this path).
    size_t s = TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
        args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    return int(round_nearest(s, MinWorkspaceAlignment));
  }
  static cutlass::Status initialize_workspace(Arguments const&, void* = nullptr, hggcStream_t = nullptr, HostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  // NON-PERSISTENT grid: one block per (m_tile, n_tile, expert), like the standard mixed-input kernel
  // (ppu_aiu_gemm_mixed_input.hpp, ~49% MFU). The persistent GroupScheduler launched only grid=(72,1,1) = #CU
  // blocks and ran tiles serially -> acu measured 2 active warps/CU, 3.1% achieved occupancy, 16% CU throughput.
  // Thousands of blocks let the HW hide per-tile load/epilogue latency across blocks. X uses Mmax over experts;
  // experts with fewer M-tiles early-exit in-kernel. (N uniform across experts.)
  // FLAT non-persistent grid: gridDim.x = SUM_e ceil(M_e/TM) (NOT Mmax*L) -> no per-expert padding, so ragged
  // experts launch ZERO idle blocks. gridDim.y = N-tiles. blockIdx.x is decoded to (expert, local m_tile) in
  // operator() by a per-expert m-tile prefix scan (L is small). (Earlier grid was (ceil(Mmax/TM), N/TN, L),
  // which over-launched idle blocks for small experts.)
  static dim3 get_grid_shape(Params const& params) {
    int const L = params.problem_shape.groups();
    int const TM = cute::size<0>(TileShape{}), TN = cute::size<1>(TileShape{});
    int total_m_tiles = 0, N = 1, mt0 = -1, mt_max = 0; bool uni = true;
    for (int e = 0; e < L; ++e) {
      auto ps = params.problem_shape.get_host_problem_shape(e);
      int const mte = int(cute::ceil_div(int(cute::get<0>(ps)), TM));
      total_m_tiles += mte;  N = int(cute::get<1>(ps));  mt_max = mte > mt_max ? mte : mt_max;
      if (mt0 < 0) mt0 = mte; else if (mte != mt0) uni = false;
    }
    int const Nt = int(cute::ceil_div(N, TN));
    bool const force3d = std::getenv("MOEG_FORCE3D") != nullptr;   // diagnostic: force 3D Mmax grid for ragged
    // UNIFORM (or forced): 3D grid (mt_max, N_tiles, L), blockIdx.z=expert, O(1) decode. small experts' extra
    // m-tiles early-exit (idle). RAGGED default: flat grid (total,N,1), no idle blocks but O(L) decode scan.
    int const S = params.splitk < 1 ? 1 : params.splitk;
    // THE SLICE LIVES ON z. The ragged (flat) path leaves z unused, so it is free there; the uniform path already
    // spends z on the expert, so the two are packed as z = expert*S + slice -- the same packing
    // ppu_aiu_gemm_parallel.hpp uses for (l, slice).
    if (uni || force3d) return dim3(uni ? mt0 : mt_max, Nt, L * S);
    return dim3(total_m_tiles, Nt, S);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE typename TileScheduler::WorkTileInfo
  fetch_next_work(typename TileScheduler::WorkTileInfo& work_tile_info, TileScheduler& scheduler) const {
    if (scheduler.continue_current_work(work_tile_info)) return work_tile_info;
    scheduler.advance_to_next_work();
    return scheduler.get_current_work();
  }

  CUTLASS_DEVICE void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};
    int const num_groups = params.problem_shape.groups();   // L extent for the collective's B/S/Z slice

    // FLAT non-persistent: decode blockIdx.x -> (expert, local m_idx) via a per-expert m-tile prefix scan
    // (num_groups is small). Every block maps to a REAL tile of some expert -> no idle blocks (vs the old
    // Mmax-sized (.,.,L) grid). l_coord = expert selects this expert's A/B/scale/zero plane in the collective.
    int const TM = int(size<0>(blk_shape));
    int const S = params.splitk < 1 ? 1 : params.splitk;
    int flat = int(blockIdx.x), n_idx = int(blockIdx.y);
    int expert = -1, m_idx = 0, slice = 0;
    if (params.mtiles_uniform > 0) {
      // UNIFORM: 3D grid (mt0, N_tiles, L) -> expert = blockIdx.z (O(1)), m_idx = blockIdx.x (local m-tile).
      // Same grid/raster as the old fast path; recovers the ~5% the flat grid lost on the balanced case.
      // z = expert*S + slice
      expert = int(blockIdx.z) / S;  slice = int(blockIdx.z) % S;  m_idx = flat;
    } else {
      // A (O(log L) ragged): binary-search the m-tile prefix [L+1] (written by launch into the scheduler-unused
      // workspace) for the largest e with prefix[e] <= flat. Replaces the O(L) linear scan (the ~5% the flat
      // grid lost at high expert count -- avg ~64 iters at L=128 -> now ~7).
      slice = int(blockIdx.z);        // flat path: z carries the slice alone
      int const* pfx = reinterpret_cast<int const*>(params.workspace);
      int lo = 0, hi = num_groups;
      while (lo + 1 < hi) { int const mid = (lo + hi) >> 1; if (pfx[mid] <= flat) lo = mid; else hi = mid; }
      expert = lo;  m_idx = flat - pfx[lo];
    }
    if (expert < 0 || expert >= num_groups) return;            // guard (grid.x is exactly SUM_e mtiles_e)
    auto pe = append<4>(params.problem_shape.get_problem_shape(expert), Int<1>{});  // ONE struct read for N,K
    int const M = int(get<0>(pe)), N = int(get<1>(pe)), K = int(get<2>(pe));
    if (m_idx * TM >= M) return;                               // idle m-tile (3D Mmax grid: small experts); no-op for flat
    if (n_idx * int(size<1>(blk_shape)) >= N) return;          // N uniform, guard anyway

    auto problem_shape_MNKL = make_shape(M, N, K, num_groups);
    auto blk_coord_mnkl = make_coord(m_idx, n_idx, _, expert);

    CollectiveMainloop collective_mainloop;
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, params.mainloop);
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    auto m_max = M - size<0>(gA) * m_idx;
    auto n_max = N - size<0>(gB) * n_idx;
    auto k_res = K - size<1>(gA) * size<2>(gA);
    auto residue_mnk = make_tuple(m_max, n_max, k_res);

    TiledMma tiled_mma;
    Tensor accumulators = make_fragment_like<ElementCompute>(partition_fragment_C(tiled_mma, take<0,2>(blk_shape)));
    clear(accumulators);
    // STRIDED K SPLIT. slice z walks k-tiles z, z+S, z+2S ... so gA/gB are untouched and only the traversal
    // changes. k_tile_count uses ceil, so an indivisible tile count would let the last slice step past the end;
    // the host refuses that case (see moe_grouped_ppu::launch) rather than relying on mainloop predication.
    auto k_tile_iter  = cute::make_splitk_coord_iterator(shape<2>(gA), slice, S);
    int  k_tile_count = (size<2>(gA) + S - 1) / S;

    if (params.probe == 1) {
      // ROUTING PROBE: skip the GEMM, tag every output element with (expert+1). See test_moe_grouped_probe.
      cute::fill(accumulators, ElementCompute(expert + 1));
    } else {
      collective_mainloop(params.mainloop, load_inputs, accumulators, k_tile_iter, k_tile_count, thread_idx, smem_buf);
    }

    // EACH SLICE WRITES ITS OWN D PLANE, e + slice*L, so the S partials are contiguous for the merge. The L
    // EXTENT has to grow with them or the epilogue's own bounds check would reject the shifted coordinate -- the
    // mainloop above already ran with the unshifted expert, which is what selects the right B/scale plane.
    auto problem_shape_epi = make_shape(M, N, K, num_groups * S);
    auto blk_coord_epi     = make_coord(m_idx, n_idx, _, expert + slice * num_groups);
    CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
    epilogue(problem_shape_epi, blk_shape, blk_coord_epi, accumulators, tiled_mma, residue_mnk,
             thread_idx, (char*)&shared_storage.tensors.epilogue);
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
