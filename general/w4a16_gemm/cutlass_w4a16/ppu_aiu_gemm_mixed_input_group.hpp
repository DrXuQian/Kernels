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
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
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
      hw_info, scheduler, workspace, args.probe
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
  static dim3 get_grid_shape(Params const& params) {
    int const L = params.problem_shape.groups();
    int Mmax = 0, N = 1;
    for (int e = 0; e < L; ++e) {
      auto ps = params.problem_shape.get_host_problem_shape(e);
      int Me = int(cute::get<0>(ps));
      if (Me > Mmax) Mmax = Me;
      N = int(cute::get<1>(ps));
    }
    int const TM = cute::size<0>(TileShape{}), TN = cute::size<1>(TileShape{});
    return dim3(int(cute::ceil_div(Mmax, TM)), int(cute::ceil_div(N, TN)), L);
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

    // NON-PERSISTENT: this block owns exactly one (m_tile, n_tile, expert), coords from blockIdx (like the
    // standard mixed-input kernel). No while-loop -> no serial per-tile latency. Experts with fewer M-tiles than
    // the Mmax-sized grid early-exit. l_coord = expert selects this expert's A/B/scale/zero plane in the collective.
    int const expert = int(blockIdx.z);
    if (expert >= num_groups) return;
    auto ps = append<4>(params.problem_shape.get_problem_shape(expert), Int<1>{});   // per-expert [M_e,N,K,1]
    int const M = int(get<0>(ps)), N = int(get<1>(ps)), K = int(get<2>(ps));
    int const m_idx = int(blockIdx.x), n_idx = int(blockIdx.y);
    if (m_idx * size<0>(blk_shape) >= M || n_idx * size<1>(blk_shape) >= N) return;   // beyond this expert's tiles

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
    auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
    int  k_tile_count = size<2>(gA);

    if (params.probe == 1) {
      // ROUTING PROBE: skip the GEMM, tag every output element with (expert+1). See test_moe_grouped_probe.
      cute::fill(accumulators, ElementCompute(expert + 1));
    } else {
      collective_mainloop(params.mainloop, load_inputs, accumulators, k_tile_iter, k_tile_count, thread_idx, smem_buf);
    }

    CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
    epilogue(problem_shape_MNKL, blk_shape, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk,
             thread_idx, (char*)&shared_storage.tensors.epilogue);   // writes this expert's D plane (L-strided)
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
