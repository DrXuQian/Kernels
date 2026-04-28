#pragma once

#include "machete_collective_builder.cuh"
#include "machete_prepack_kernel.cuh"
#include "machete_prepacked_layout.cuh"
#include "machete_standalone_check.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/vllm_custom_types.cuh"
#include "cutlass_extensions/vllm_numeric_conversion.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace machete_standalone {

using namespace cute;

enum class ActType
{
    FP16,
    BF16,
};

enum class QuantType
{
    GPTQ_U4B8,
    AWQ_U4,
};

struct MacheteSchedule
{
    char const* name;
    int tile_n;
    int tile_m;
    int cluster_n;
    int cluster_m;
    int cluster_k;
};

struct MacheteGemmConfig
{
    ActType act_type = ActType::FP16;
    QuantType quant_type = QuantType::GPTQ_U4B8;
    int group_size = 128;
    MacheteSchedule schedule{};
};

inline std::vector<MacheteSchedule> supported_schedules()
{
    return {
        {"128x128_2x1x1_TmaMI_TmaCoop_streamK", 128, 128, 2, 1, 1},
        {"128x128_1x1x1_TmaMI_TmaCoop_streamK", 128, 128, 1, 1, 1},
        {"128x256_2x1x1_TmaMI_TmaCoop_streamK", 128, 256, 2, 1, 1},
        {"128x64_2x1x1_TmaMI_TmaCoop_streamK", 128, 64, 2, 1, 1},
        {"128x32_2x1x1_TmaMI_TmaCoop_streamK", 128, 32, 2, 1, 1},
        {"256x128_2x1x1_TmaMI_TmaCoop_streamK", 256, 128, 2, 1, 1},
        {"128x16_1x1x1_TmaMI_TmaCoop_streamK", 128, 16, 1, 1, 1},
        {"256x64_2x1x1_TmaMI_TmaCoop_streamK", 256, 64, 2, 1, 1},
        {"256x32_2x1x1_TmaMI_TmaCoop_streamK", 256, 32, 2, 1, 1},
        {"256x16_1x1x1_TmaMI_TmaCoop_streamK", 256, 16, 1, 1, 1},
    };
}

inline bool same_schedule(MacheteSchedule const& a, MacheteSchedule const& b)
{
    return a.tile_n == b.tile_n && a.tile_m == b.tile_m && a.cluster_n == b.cluster_n
        && a.cluster_m == b.cluster_m && a.cluster_k == b.cluster_k;
}

inline std::optional<MacheteSchedule> schedule_from_name(std::string const& name)
{
    for (auto const& schedule : supported_schedules())
    {
        if (name == schedule.name)
        {
            return schedule;
        }
    }
    return std::nullopt;
}

// This is the vLLM Machete H100 heuristic from csrc/quantization/machete/generate.py.
inline MacheteSchedule select_schedule(int m, int n, int k)
{
    auto const schedules = supported_schedules();
    auto find = [&](char const* name) -> MacheteSchedule {
        auto schedule = schedule_from_name(name);
        MACHETE_CHECK(schedule.has_value(), "missing schedule", name);
        return *schedule;
    };

    if (m > 256 && k <= 16384 && n <= 4096)
    {
        return find("128x128_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 256)
    {
        return find("128x256_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 128 && k <= 4096 && n <= 4096)
    {
        return find("128x64_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 128 && k <= 8192 && n <= 8192)
    {
        return find("128x128_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 128)
    {
        return find("128x256_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 64 && k <= 4069 && n <= 4069)
    {
        return find("128x32_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 64 && k <= 4069 && n <= 8192)
    {
        return find("128x64_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 64 && k >= 8192 && n >= 12288)
    {
        return find("256x128_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 64)
    {
        return find("128x128_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 32 && k <= 6144 && n <= 6144)
    {
        return find("128x16_1x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 32 && k >= 16384 && n >= 12288)
    {
        return find("256x64_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 32)
    {
        return find("128x64_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 16 && k <= 12288 && n <= 8192)
    {
        return find("128x32_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (m > 16)
    {
        return find("256x32_2x1x1_TmaMI_TmaCoop_streamK");
    }
    if (n >= 26624)
    {
        return find("256x16_1x1x1_TmaMI_TmaCoop_streamK");
    }
    return find("128x16_1x1x1_TmaMI_TmaCoop_streamK");
}

template <int TileN, int TileM, int ClusterN, int ClusterM, int ClusterK>
struct ScheduleTemplate
{
    using TileShapeNM = Shape<Int<TileN>, Int<TileM>>;
    using ClusterShape = Shape<Int<ClusterN>, Int<ClusterM>, Int<ClusterK>>;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileScheduler = cutlass::gemm::StreamKScheduler;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

template <typename ElementA_, typename ElementB_, typename ElementD_, typename AccumulatorT, typename GroupScaleT,
    typename GroupZeroT, typename ScheduleConfig>
struct MacheteKernelTemplate
{
    static constexpr bool with_group_scales = !std::is_same_v<GroupScaleT, void>;
    static constexpr bool with_group_zeropoints = !std::is_same_v<GroupZeroT, void>;

    using MmaType = ElementA_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementD = ElementD_;
    using ElementC = void;
    using ElementAccumulator = AccumulatorT;
    using ElementSGroup = cute::conditional_t<with_group_scales, GroupScaleT, MmaType>;
    using ElementZGroup = cute::conditional_t<with_group_zeropoints, GroupZeroT, MmaType>;
    using ElementConvertGroup = cute::conditional_t<with_group_scales, GroupScaleT, MmaType>;

    using BTypeTuple = cute::conditional_t<with_group_scales,
        cute::conditional_t<with_group_zeropoints, cute::tuple<ElementB, ElementSGroup, ElementZGroup>,
            cute::tuple<ElementB, ElementSGroup>>,
        ElementB>;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using LayoutScale = cutlass::layout::RowMajor;
    using LayoutBUnused = cutlass::layout::ColumnMajor;
    using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
    using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using PrepackedLayoutB = machete::PrepackedLayoutBTemplate<ElementA, ElementB, ElementConvertGroup,
        ElementAccumulator, LayoutA_Transpose, KernelSchedule>;

    static int constexpr TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value;
    static int constexpr AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
    static int constexpr AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
    static int constexpr AlignmentD = 128 / cutlass::sizeof_bits_v<ElementD>;

    using TileShape = decltype(append(typename ScheduleConfig::TileShapeNM{}, cute::Int<TileShapeK>{}));
    using ClusterShape = typename ScheduleConfig::ClusterShape;
    using EpilogueSchedule = typename ScheduleConfig::EpilogueSchedule;
    using EpilogueTileType = typename ScheduleConfig::EpilogueTileType;
    using TileScheduler = typename ScheduleConfig::TileScheduler;

    using StoreEpilogueCompute =
        typename cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
        TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator, ElementC, LayoutD_Transpose,
        0, ElementD, LayoutD_Transpose, AlignmentD, EpilogueSchedule, StoreEpilogueCompute>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::VLLMCollectiveBuilder<
        cutlass::gemm::collective::MacheteKernelTag, ArchTag, OperatorClass, BTypeTuple, PrepackedLayoutB, AlignmentB,
        ElementA, LayoutA_Transpose, AlignmentA, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
        CollectiveEpilogue, TileScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideD = cutlass::detail::TagToStrideA_t<LayoutD>;
    using StrideSGroup = cutlass::detail::TagToStrideA_t<LayoutScale>;
    using StrideZGroup = StrideSGroup;
    using StrideBUnused = cutlass::detail::TagToStrideB_t<LayoutBUnused>;

    using Arguments = typename Gemm::Arguments;
    using MainloopArguments = typename GemmKernel::MainloopArguments;
    using EpilogueArguments = typename GemmKernel::EpilogueArguments;

    static Arguments create_arguments(ElementA const* A, ElementB const* B, ElementD* D,
        ElementSGroup const* group_scales, ElementZGroup const* group_zeros, int m, int n, int k, int group_size)
    {
        static_assert(!with_group_zeropoints || with_group_scales);
        if (group_size == -1)
        {
            group_size = k;
        }
        int const scale_k = (k + group_size - 1) / group_size;

        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
        auto stride_Dt = permute_layout<1, 0, 2>(make_layout(make_shape(m, n, 1), stride_D)).stride();

        MainloopArguments mainloop_arguments{};
        EpilogueArguments epilogue_arguments{{}, nullptr, {}, D, stride_Dt};

        if constexpr (with_group_scales && with_group_zeropoints)
        {
            MACHETE_CHECK(group_scales != nullptr, "group scales are required");
            MACHETE_CHECK(group_zeros != nullptr, "group zeros are required");
            auto stride_S = cutlass::make_cute_packed_stride(StrideSGroup{}, cute::make_shape(scale_k, n, 1));
            auto stride_St = permute_layout<1, 0, 2>(make_layout(make_shape(scale_k, n, 1), stride_S)).stride();
            mainloop_arguments
                = MainloopArguments{B, StrideBUnused{}, A, stride_A, group_scales, stride_St, group_size, group_zeros};
        }
        else if constexpr (with_group_scales)
        {
            MACHETE_CHECK(group_scales != nullptr, "group scales are required");
            auto stride_S = cutlass::make_cute_packed_stride(StrideSGroup{}, cute::make_shape(scale_k, n, 1));
            auto stride_St = permute_layout<1, 0, 2>(make_layout(make_shape(scale_k, n, 1), stride_S)).stride();
            mainloop_arguments = MainloopArguments{B, StrideBUnused{}, A, stride_A, group_scales, stride_St, group_size};
        }
        else
        {
            mainloop_arguments = MainloopArguments{B, StrideBUnused{}, A, stride_A};
        }

        return Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1}, mainloop_arguments, epilogue_arguments};
    }

    static size_t get_workspace_size(Arguments const& args) { return Gemm::get_workspace_size(args); }

    static bool can_implement(Arguments const& args) { return Gemm::can_implement(args) == cutlass::Status::kSuccess; }

    static void run(Arguments const& args, void* workspace, cudaStream_t stream)
    {
        Gemm gemm_op;
        cutlass::Status status = gemm_op.initialize(args, workspace, stream);
        MACHETE_CHECK(status == cutlass::Status::kSuccess, "Machete kernel failed to initialize workspace");
        status = gemm_op.run(stream);
        MACHETE_CHECK(status == cutlass::Status::kSuccess, "Machete kernel failed");
    }
};

template <typename ElementA, typename ElementB, typename ElementConvert, typename AccumulatorT>
using PrepackLayout = machete::PrepackedLayoutBTemplate<ElementA, ElementB, ElementConvert, AccumulatorT,
    typename cutlass::layout::LayoutTranspose<cutlass::layout::RowMajor>::type,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>;

template <typename ElementA, typename ElementB, typename ElementConvert, typename AccumulatorT>
void prepack_B(cudaStream_t stream, ElementB const* b_in, ElementB* b_out, int k, int n)
{
    using Layout = PrepackLayout<ElementA, ElementB, ElementConvert, AccumulatorT>;
    int constexpr elements_per_storage = 32 / cutlass::sizeof_bits_v<ElementB>;
    MACHETE_CHECK(k % elements_per_storage == 0, "K must be divisible by storage pack factor");
    MACHETE_CHECK(k % size<1>(typename Layout::PPBlockShape_NK{}) == 0, "K must be divisible by prepack block K");
    MACHETE_CHECK(n % size<0>(typename Layout::PPBlockShape_NK{}) == 0, "N must be divisible by prepack block N");

    int const packed_k = k / elements_per_storage;
    auto layout_Bt = make_layout(make_shape(n, k, 1), make_stride(packed_k * elements_per_storage, 1, n * k));
    machete::prepack_B_template<Layout>(stream, b_in, layout_Bt, b_out);
}

void prepack_B_fp16_u4b8(cudaStream_t stream, cutlass::vllm_uint4b8_t const* b_in, cutlass::vllm_uint4b8_t* b_out,
    int k, int n);
void prepack_B_bf16_u4b8(cudaStream_t stream, cutlass::vllm_uint4b8_t const* b_in, cutlass::vllm_uint4b8_t* b_out,
    int k, int n);
void prepack_B_fp16_u4(cudaStream_t stream, cutlass::uint4b_t const* b_in, cutlass::uint4b_t* b_out, int k, int n);
void prepack_B_bf16_u4(cudaStream_t stream, cutlass::uint4b_t const* b_in, cutlass::uint4b_t* b_out, int k, int n);

void machete_mm_fp16_u4b8(cudaStream_t stream, cutlass::half_t const* A, cutlass::vllm_uint4b8_t const* B,
    cutlass::half_t const* scales, cutlass::half_t* D, int m, int n, int k, int group_size,
    MacheteSchedule schedule, void* workspace, size_t workspace_bytes);

void machete_mm_bf16_u4b8(cudaStream_t stream, cutlass::bfloat16_t const* A, cutlass::vllm_uint4b8_t const* B,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t* D, int m, int n, int k, int group_size,
    MacheteSchedule schedule, void* workspace, size_t workspace_bytes);

void machete_mm_fp16_u4(cudaStream_t stream, cutlass::half_t const* A, cutlass::uint4b_t const* B,
    cutlass::half_t const* scales, cutlass::half_t const* zeros, cutlass::half_t* D, int m, int n, int k,
    int group_size, MacheteSchedule schedule, void* workspace, size_t workspace_bytes);

void machete_mm_bf16_u4(cudaStream_t stream, cutlass::bfloat16_t const* A, cutlass::uint4b_t const* B,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* zeros, cutlass::bfloat16_t* D, int m, int n,
    int k, int group_size, MacheteSchedule schedule, void* workspace, size_t workspace_bytes);

size_t machete_get_workspace_size_fp16_u4b8(cutlass::half_t const* A, cutlass::vllm_uint4b8_t const* B,
    cutlass::half_t const* scales, cutlass::half_t* D, int m, int n, int k, int group_size, MacheteSchedule schedule);

size_t machete_get_workspace_size_bf16_u4b8(cutlass::bfloat16_t const* A, cutlass::vllm_uint4b8_t const* B,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t* D, int m, int n, int k, int group_size,
    MacheteSchedule schedule);

size_t machete_get_workspace_size_fp16_u4(cutlass::half_t const* A, cutlass::uint4b_t const* B,
    cutlass::half_t const* scales, cutlass::half_t const* zeros, cutlass::half_t* D, int m, int n, int k,
    int group_size, MacheteSchedule schedule);

size_t machete_get_workspace_size_bf16_u4(cutlass::bfloat16_t const* A, cutlass::uint4b_t const* B,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* zeros, cutlass::bfloat16_t* D, int m, int n,
    int k, int group_size, MacheteSchedule schedule);

} // namespace machete_standalone
