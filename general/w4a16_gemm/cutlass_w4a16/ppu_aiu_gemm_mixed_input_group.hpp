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
  };
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    TileSchedulerParams scheduler;
    void* workspace;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    ProblemShape problem_shapes = args.problem_shape;
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0)
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    KernelHardwareInfo hw_info{args.hw_info.device_id, cu_count};

    uint8_t* wp = reinterpret_cast<uint8_t*>(workspace);
    size_t off = 0;
    void* scheduler_workspace = wp;
    off += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
        args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    off = round_nearest(off, MinWorkspaceAlignment);
    void* epilogue_workspace = wp + off;
    off += CollectiveEpilogue::get_workspace_size(problem_shapes, args.epilogue, cu_count);
    off = round_nearest(off, MinWorkspaceAlignment);

    constexpr uint32_t NumEpilogueSubTiles = 1;
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        problem_shapes, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);

    // [Q2] The mixed-input collective's to_underlying_arguments wants ONE (M,N,K[,L]) to compute scale_k=K/gs
    // and the (N,scale_k,L) scale layout. K,N are uniform across experts; L = number of groups. Build a
    // representative rank-4 shape [M=nominal, N, K, L=num_groups] from the group shapes so the collective's
    // L-strided B/scale/zero span all experts.
    auto host0 = problem_shapes.get_host_problem_shape(0);   // [Q2] API: per-group host shape accessor
    int Nrep = get<1>(host0), Krep = get<2>(host0);
    int Lrep = problem_shapes.groups();                      // [Q2] API: group count
    auto rep_mnkl = cute::make_shape(int(get<0>(host0)), Nrep, Krep, Lrep);

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(rep_mnkl, args.mainloop, /*workspace=*/nullptr),
      CollectiveEpilogue::to_underlying_arguments(problem_shapes, args.epilogue, epilogue_workspace),
      hw_info, scheduler, workspace
    };
  }

  static bool can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGrouped || args.mode == GemmUniversalMode::kArray;
  }
  static int get_workspace_size(Arguments const& args) {
    // scheduler + epilogue workspace (mirrors to_underlying_arguments)
    int cu = args.hw_info.cu_count > 0 ? args.hw_info.cu_count
           : KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    size_t s = TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
        args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    s = round_nearest(s, MinWorkspaceAlignment);
    s += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, cu);
    return int(round_nearest(s, MinWorkspaceAlignment));
  }
  static cutlass::Status initialize_workspace(Arguments const&, void* = nullptr, hggcStream_t = nullptr, HostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>)
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN
                      ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
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
    int const num_groups = params.problem_shape.groups();   // [Q2] L extent for the collective's B/S/Z slice

    TileScheduler scheduler{params.scheduler};
    auto work_tile_info = scheduler.get_current_work();

    while (work_tile_info.is_valid()) {
      if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
        work_tile_info = fetch_next_work(work_tile_info, scheduler);
        continue;
      }
      int expert = work_tile_info.L_idx;
      auto ps = append<4>(params.problem_shape.get_problem_shape(expert), Int<1>{});   // per-expert [M_e,N,K,1]
      auto M = get<0>(ps); auto N = get<1>(ps); auto K = get<2>(ps);
      // Drive the mixed-input collective with L = num_groups so its B/scale/zero L-slice spans all experts and
      // l_coord = expert selects this expert's plane. (Uniform-M for step 2a; A is sliced by l_coord too.)
      auto problem_shape_MNKL = make_shape(int(M), int(N), int(K), num_groups);   // [Q3] uniform-M assumption
      auto blk_coord_mnkl = make_coord(work_tile_info.M_idx, work_tile_info.N_idx, _, expert);

      CollectiveMainloop collective_mainloop;
      auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, params.mainloop);
      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);

      auto m_max = M - size<0>(gA) * work_tile_info.M_idx;
      auto n_max = N - size<0>(gB) * work_tile_info.N_idx;
      auto k_res = K - size<1>(gA) * size<2>(gA);
      auto residue_mnk = make_tuple(m_max, n_max, k_res);

      TiledMma tiled_mma;
      Tensor accumulators = make_fragment_like<ElementCompute>(partition_fragment_C(tiled_mma, take<0,2>(blk_shape)));
      clear(accumulators);
      auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
      int  k_tile_count = size<2>(gA);

      collective_mainloop(params.mainloop, load_inputs, accumulators, k_tile_iter, k_tile_count, thread_idx, smem_buf);

      CollectiveEpilogue epilogue{params.epilogue, shared_storage.tensors.epilogue};
      epilogue(problem_shape_MNKL, blk_shape, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk,
               thread_idx, (char*)&shared_storage.tensors.epilogue);   // [Q4] epilogue must write to this expert's D plane (L-strided by l_coord)

      work_tile_info = fetch_next_work(work_tile_info, scheduler);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
