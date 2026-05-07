#include "quantization/machete/cutlass55_backend.cuh"

#include "machete_standalone_check.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace machete_standalone {

std::vector<Cutlass55Config> supported_cutlass55_configs()
{
    return {
        {"128x128x64_1x1x1", 128, 128, 1, 1, 1},
        {"128x64x64_1x1x1", 128, 64, 1, 1, 1},
        {"128x256x64_1x1x1", 128, 256, 1, 1, 1},
        {"256x128x64_1x1x1", 256, 128, 1, 1, 1},
        {"128x128x64_2x1x1", 128, 128, 2, 1, 1},
        {"128x64x64_2x1x1", 128, 64, 2, 1, 1},
        {"128x256x64_2x1x1", 128, 256, 2, 1, 1},
        {"256x128x64_2x1x1", 256, 128, 2, 1, 1},
    };
}

Cutlass55Config default_cutlass55_config()
{
    return supported_cutlass55_configs().front();
}

struct Cutlass55Plan
{
    virtual ~Cutlass55Plan() = default;
    virtual void run(cudaStream_t stream) = 0;
};

namespace {

using namespace cute;

template <typename ElementA_, int TileN, int TileM, int ClusterN, int ClusterM, int ClusterK>
struct Cutlass55ScaleOnlyKernel
{
    using MmaType = ElementA_;
    using ElementA = MmaType;
    using ElementB = cutlass::int4b_t;
    using ElementScale = MmaType;
    using ElementC = void;
    using ElementD = MmaType;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;
    using LayoutScale = cutlass::layout::RowMajor;
    using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;

    static int constexpr AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
    static int constexpr AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
    static int constexpr AlignmentC = 0;
    static int constexpr AlignmentD = 128 / cutlass::sizeof_bits_v<ElementD>;
    static int constexpr TileShapeK = 128 * 8 / cutlass::sizeof_bits_v<MmaType>;

    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
    using StrideCRef = cutlass::detail::TagToStrideC_t<LayoutC>;

    using ValueShuffle = Layout<Shape<_2, _4>, Stride<_4, _1>>;
    static int constexpr NumShuffleAtoms = 1;
    using MmaAtomShape = Layout<Shape<_1, Int<NumShuffleAtoms>>>;
    using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
    using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int, int, int>, StrideB>{}));

    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using TileShape = Shape<Int<TileN>, Int<TileM>, cute::Int<TileShapeK>>;
    using ClusterShape = Shape<Int<ClusterN>, Int<ClusterM>, Int<ClusterK>>;
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
        TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator, ElementC,
        typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC, ElementD,
        typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD, EpilogueSchedule>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass,
        cute::tuple<ElementB, ElementScale>, LayoutB_Reordered, AlignmentB, ElementA, LayoutA_Transpose, AlignmentA,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using StrideScale = typename CollectiveMainloop::StrideScale;
    using Arguments = typename Gemm::Arguments;

    static LayoutB_Reordered make_reordered_B_layout(int n, int k)
    {
        auto shape_B = cute::make_shape(n, k, 1);
        return cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
    }

    static void reorder_B(cudaStream_t stream, ElementB const* b_in, ElementB* b_out, int k, int n)
    {
        (void) stream;
        auto shape_B = cute::make_shape(n, k, 1);
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
        auto layout_B = cute::make_layout(shape_B, stride_B);
        auto layout_B_reordered = make_reordered_B_layout(n, k);
        cutlass::reorder_tensor(b_in, layout_B, b_out, layout_B_reordered);
    }

    static Arguments create_arguments(ElementA const* A, ElementB const* B_reordered, ElementScale const* scales,
        ElementC const* C, ElementD* D, int m, int n, int k, int group_size)
    {
        if (group_size == -1)
        {
            group_size = k;
        }
        int const scale_k = cutlass::ceil_div(k, group_size);
        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
        auto stride_S = cutlass::make_cute_packed_stride(StrideScale{}, cute::make_shape(n, scale_k, 1));
        auto layout_B_reordered = make_reordered_B_layout(n, k);
        (void) C;

        return Arguments{cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1},
            {B_reordered, layout_B_reordered, A, stride_A, scales, stride_S, group_size},
            {{1.0f, 0.0f}, nullptr, {}, D, stride_D}};
    }

    static size_t get_workspace_size(Arguments const& args) { return Gemm::get_workspace_size(args); }

    static bool can_implement(Arguments const& args) { return Gemm::can_implement(args) == cutlass::Status::kSuccess; }

    static void run(Arguments const& args, void* workspace, cudaStream_t stream)
    {
        Gemm gemm_op;
        cutlass::Status status = gemm_op.initialize(args, workspace, stream);
        MACHETE_CHECK(status == cutlass::Status::kSuccess, "CUTLASS55 kernel failed to initialize workspace");
        status = gemm_op.run(stream);
        MACHETE_CHECK(status == cutlass::Status::kSuccess, "CUTLASS55 kernel failed");
    }
};

template <typename ElementA, int TileN, int TileM, int ClusterN, int ClusterM, int ClusterK>
using Cutlass55Kernel = Cutlass55ScaleOnlyKernel<ElementA, TileN, TileM, ClusterN, ClusterM, ClusterK>;

#define CUTLASS55_DISPATCH_CONFIG(ELEMENT_A, ...)                                                                      \
    do                                                                                                                  \
    {                                                                                                                   \
        if (config.tile_n == 128 && config.tile_m == 128 && config.cluster_n == 1 && config.cluster_m == 1)             \
        {                                                                                                               \
            using Kernel = Cutlass55Kernel<ELEMENT_A, 128, 128, 1, 1, 1>;                                               \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (config.tile_n == 128 && config.tile_m == 64 && config.cluster_n == 1 && config.cluster_m == 1)              \
        {                                                                                                               \
            using Kernel = Cutlass55Kernel<ELEMENT_A, 128, 64, 1, 1, 1>;                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (config.tile_n == 128 && config.tile_m == 256 && config.cluster_n == 1 && config.cluster_m == 1)             \
        {                                                                                                               \
            using Kernel = Cutlass55Kernel<ELEMENT_A, 128, 256, 1, 1, 1>;                                               \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (config.tile_n == 256 && config.tile_m == 128 && config.cluster_n == 1 && config.cluster_m == 1)             \
        {                                                                                                               \
            using Kernel = Cutlass55Kernel<ELEMENT_A, 256, 128, 1, 1, 1>;                                               \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (config.tile_n == 128 && config.tile_m == 128 && config.cluster_n == 2 && config.cluster_m == 1)             \
        {                                                                                                               \
            using Kernel = Cutlass55Kernel<ELEMENT_A, 128, 128, 2, 1, 1>;                                               \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (config.tile_n == 128 && config.tile_m == 64 && config.cluster_n == 2 && config.cluster_m == 1)              \
        {                                                                                                               \
            using Kernel = Cutlass55Kernel<ELEMENT_A, 128, 64, 2, 1, 1>;                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (config.tile_n == 128 && config.tile_m == 256 && config.cluster_n == 2 && config.cluster_m == 1)             \
        {                                                                                                               \
            using Kernel = Cutlass55Kernel<ELEMENT_A, 128, 256, 2, 1, 1>;                                               \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (config.tile_n == 256 && config.tile_m == 128 && config.cluster_n == 2 && config.cluster_m == 1)             \
        {                                                                                                               \
            using Kernel = Cutlass55Kernel<ELEMENT_A, 256, 128, 2, 1, 1>;                                               \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
    } while (0)

template <typename Kernel>
void run_kernel(cudaStream_t stream, typename Kernel::ElementA const* A, typename Kernel::ElementB const* B,
    typename Kernel::ElementScale const* scales, typename Kernel::ElementC const* C, typename Kernel::ElementD* D,
    int m, int n, int k, int group_size, void* workspace, size_t workspace_bytes)
{
    auto args = Kernel::create_arguments(A, B, scales, C, D, m, n, k, group_size);
    MACHETE_CHECK(Kernel::can_implement(args), "CUTLASS55 kernel cannot implement the requested problem");
    size_t const needed = Kernel::get_workspace_size(args);
    MACHETE_CHECK(workspace_bytes >= needed, "insufficient CUTLASS55 workspace", workspace_bytes, "needed", needed);
    Kernel::run(args, workspace, stream);
}

template <typename Kernel>
struct TypedCutlass55Plan : Cutlass55Plan
{
    using Arguments = typename Kernel::Arguments;
    using Gemm = typename Kernel::Gemm;

    TypedCutlass55Plan(Arguments const& args, void* workspace, cudaStream_t stream)
        : args_(args)
    {
        cutlass::Status status = gemm_.initialize(args_, workspace, stream);
        MACHETE_CHECK(status == cutlass::Status::kSuccess, "CUTLASS55 kernel failed to initialize workspace");
    }

    void run(cudaStream_t stream) override
    {
        cutlass::Status status = gemm_.run(stream);
        MACHETE_CHECK(status == cutlass::Status::kSuccess, "CUTLASS55 kernel failed");
    }

    Arguments args_;
    Gemm gemm_;
};

template <typename Kernel>
Cutlass55Plan* create_plan(cudaStream_t stream, typename Kernel::ElementA const* A, typename Kernel::ElementB const* B,
    typename Kernel::ElementScale const* scales, typename Kernel::ElementC const* C, typename Kernel::ElementD* D,
    int m, int n, int k, int group_size, void* workspace, size_t workspace_bytes)
{
    auto args = Kernel::create_arguments(A, B, scales, C, D, m, n, k, group_size);
    MACHETE_CHECK(Kernel::can_implement(args), "CUTLASS55 kernel cannot implement the requested problem");
    size_t const needed = Kernel::get_workspace_size(args);
    MACHETE_CHECK(workspace_bytes >= needed, "insufficient CUTLASS55 workspace", workspace_bytes, "needed", needed);
    return new TypedCutlass55Plan<Kernel>(args, workspace, stream);
}

template <typename Kernel>
size_t workspace_size(typename Kernel::ElementA const* A, typename Kernel::ElementB const* B,
    typename Kernel::ElementScale const* scales, typename Kernel::ElementC const* C, typename Kernel::ElementD* D,
    int m, int n, int k, int group_size)
{
    auto args = Kernel::create_arguments(A, B, scales, C, D, m, n, k, group_size);
    MACHETE_CHECK(Kernel::can_implement(args), "CUTLASS55 kernel cannot implement the requested problem");
    return Kernel::get_workspace_size(args);
}

} // namespace

void cutlass55_reorder_B_fp16_s4(cudaStream_t stream, cutlass::int4b_t const* b_in, cutlass::int4b_t* b_out,
    int k, int n)
{
    using Kernel = Cutlass55Kernel<cutlass::half_t, 128, 128, 1, 1, 1>;
    Kernel::reorder_B(stream, b_in, b_out, k, n);
}

void cutlass55_reorder_B_bf16_s4(cudaStream_t stream, cutlass::int4b_t const* b_in, cutlass::int4b_t* b_out,
    int k, int n)
{
    using Kernel = Cutlass55Kernel<cutlass::bfloat16_t, 128, 128, 1, 1, 1>;
    Kernel::reorder_B(stream, b_in, b_out, k, n);
}

void cutlass55_mm_fp16_s4(cudaStream_t stream, cutlass::half_t const* A, cutlass::int4b_t const* B_reordered,
    cutlass::half_t const* scales, cutlass::half_t const* C, cutlass::half_t* D, int m, int n, int k,
    int group_size, void* workspace, size_t workspace_bytes)
{
    using Kernel = Cutlass55Kernel<cutlass::half_t, 128, 128, 1, 1, 1>;
    run_kernel<Kernel>(stream, A, B_reordered, scales, C, D, m, n, k, group_size, workspace, workspace_bytes);
}

void cutlass55_mm_bf16_s4(cudaStream_t stream, cutlass::bfloat16_t const* A, cutlass::int4b_t const* B_reordered,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* C, cutlass::bfloat16_t* D, int m, int n,
    int k, int group_size, void* workspace, size_t workspace_bytes)
{
    using Kernel = Cutlass55Kernel<cutlass::bfloat16_t, 128, 128, 1, 1, 1>;
    run_kernel<Kernel>(stream, A, B_reordered, scales, C, D, m, n, k, group_size, workspace, workspace_bytes);
}

size_t cutlass55_get_workspace_size_fp16_s4(cutlass::half_t const* A, cutlass::int4b_t const* B_reordered,
    cutlass::half_t const* scales, cutlass::half_t const* C, cutlass::half_t* D, int m, int n, int k,
    int group_size, Cutlass55Config config)
{
    CUTLASS55_DISPATCH_CONFIG(cutlass::half_t,
        return workspace_size<Kernel>(A, B_reordered, scales, C, D, m, n, k, group_size));
    MACHETE_CHECK(false, "unsupported CUTLASS55 config", config.name);
}

size_t cutlass55_get_workspace_size_bf16_s4(cutlass::bfloat16_t const* A, cutlass::int4b_t const* B_reordered,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* C, cutlass::bfloat16_t* D, int m, int n,
    int k, int group_size, Cutlass55Config config)
{
    CUTLASS55_DISPATCH_CONFIG(cutlass::bfloat16_t,
        return workspace_size<Kernel>(A, B_reordered, scales, C, D, m, n, k, group_size));
    MACHETE_CHECK(false, "unsupported CUTLASS55 config", config.name);
}

Cutlass55Plan* cutlass55_create_plan_fp16_s4(cudaStream_t stream, cutlass::half_t const* A,
    cutlass::int4b_t const* B_reordered, cutlass::half_t const* scales, cutlass::half_t const* C, cutlass::half_t* D,
    int m, int n, int k, int group_size, Cutlass55Config config, void* workspace, size_t workspace_bytes)
{
    CUTLASS55_DISPATCH_CONFIG(cutlass::half_t,
        return create_plan<Kernel>(
            stream, A, B_reordered, scales, C, D, m, n, k, group_size, workspace, workspace_bytes));
    MACHETE_CHECK(false, "unsupported CUTLASS55 config", config.name);
}

Cutlass55Plan* cutlass55_create_plan_bf16_s4(cudaStream_t stream, cutlass::bfloat16_t const* A,
    cutlass::int4b_t const* B_reordered, cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* C,
    cutlass::bfloat16_t* D, int m, int n, int k, int group_size, Cutlass55Config config, void* workspace,
    size_t workspace_bytes)
{
    CUTLASS55_DISPATCH_CONFIG(cutlass::bfloat16_t,
        return create_plan<Kernel>(
            stream, A, B_reordered, scales, C, D, m, n, k, group_size, workspace, workspace_bytes));
    MACHETE_CHECK(false, "unsupported CUTLASS55 config", config.name);
}

void cutlass55_run_plan(Cutlass55Plan* plan, cudaStream_t stream)
{
    MACHETE_CHECK(plan != nullptr, "CUTLASS55 plan is null");
    plan->run(stream);
}

void cutlass55_destroy_plan(Cutlass55Plan* plan)
{
    delete plan;
}

} // namespace machete_standalone
