#include "quantization/machete/machete_standalone_gemm.cuh"

namespace machete_standalone {
namespace {

template <typename ElementA, typename Schedule>
using GPTQKernel = MacheteKernelTemplate<ElementA, cutlass::vllm_uint4b8_t, ElementA, float, ElementA, void, Schedule>;

template <typename ElementA, typename Schedule>
using AWQKernel = MacheteKernelTemplate<ElementA, cutlass::uint4b_t, ElementA, float, ElementA, ElementA, Schedule>;

template <typename Kernel, typename ElementA>
typename Kernel::Arguments make_args(cudaStream_t, ElementA const* A, typename Kernel::ElementB const* B,
    ElementA const* scales, ElementA const* zeros, ElementA* D, int m, int n, int k, int group_size)
{
    return Kernel::create_arguments(A, B, D, scales, zeros, m, n, k, group_size);
}

template <typename Kernel, typename ElementA>
void run_kernel(cudaStream_t stream, ElementA const* A, typename Kernel::ElementB const* B, ElementA const* scales,
    ElementA const* zeros, ElementA* D, int m, int n, int k, int group_size, void* workspace, size_t workspace_bytes)
{
    auto args = make_args<Kernel>(stream, A, B, scales, zeros, D, m, n, k, group_size);
    MACHETE_CHECK(Kernel::can_implement(args), "Machete kernel cannot implement the requested problem");
    size_t const needed = Kernel::get_workspace_size(args);
    MACHETE_CHECK(workspace_bytes >= needed, "insufficient workspace", workspace_bytes, "needed", needed);
    Kernel::run(args, workspace, stream);
}

template <typename Kernel, typename ElementA>
size_t workspace_size(cudaStream_t stream, ElementA const* A, typename Kernel::ElementB const* B, ElementA const* scales,
    ElementA const* zeros, ElementA* D, int m, int n, int k, int group_size)
{
    auto args = make_args<Kernel>(stream, A, B, scales, zeros, D, m, n, k, group_size);
    MACHETE_CHECK(Kernel::can_implement(args), "Machete kernel cannot implement the requested problem");
    return Kernel::get_workspace_size(args);
}

#define MACHETE_DISPATCH_SCHEDULE(ELEMENT_A, KERNEL_ALIAS, ...)                                                         \
    do                                                                                                                  \
    {                                                                                                                   \
        if (schedule.tile_n == 128 && schedule.tile_m == 128 && schedule.cluster_n == 2 && schedule.cluster_m == 1)     \
        {                                                                                                               \
            using Sch = ScheduleTemplate<128, 128, 2, 1, 1>;                                                            \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 128 && schedule.tile_m == 128 && schedule.cluster_n == 1 && schedule.cluster_m == 1)     \
        {                                                                                                               \
            using Sch = ScheduleTemplate<128, 128, 1, 1, 1>;                                                            \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 128 && schedule.tile_m == 256 && schedule.cluster_n == 2 && schedule.cluster_m == 1)     \
        {                                                                                                               \
            using Sch = ScheduleTemplate<128, 256, 2, 1, 1>;                                                            \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 128 && schedule.tile_m == 64 && schedule.cluster_n == 2 && schedule.cluster_m == 1)      \
        {                                                                                                               \
            using Sch = ScheduleTemplate<128, 64, 2, 1, 1>;                                                             \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 128 && schedule.tile_m == 32 && schedule.cluster_n == 2 && schedule.cluster_m == 1)      \
        {                                                                                                               \
            using Sch = ScheduleTemplate<128, 32, 2, 1, 1>;                                                             \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 256 && schedule.tile_m == 128 && schedule.cluster_n == 2 && schedule.cluster_m == 1)     \
        {                                                                                                               \
            using Sch = ScheduleTemplate<256, 128, 2, 1, 1>;                                                            \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 128 && schedule.tile_m == 16 && schedule.cluster_n == 1 && schedule.cluster_m == 1)      \
        {                                                                                                               \
            using Sch = ScheduleTemplate<128, 16, 1, 1, 1>;                                                             \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 256 && schedule.tile_m == 64 && schedule.cluster_n == 2 && schedule.cluster_m == 1)      \
        {                                                                                                               \
            using Sch = ScheduleTemplate<256, 64, 2, 1, 1>;                                                             \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 256 && schedule.tile_m == 32 && schedule.cluster_n == 2 && schedule.cluster_m == 1)      \
        {                                                                                                               \
            using Sch = ScheduleTemplate<256, 32, 2, 1, 1>;                                                             \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
        if (schedule.tile_n == 256 && schedule.tile_m == 16 && schedule.cluster_n == 1 && schedule.cluster_m == 1)      \
        {                                                                                                               \
            using Sch = ScheduleTemplate<256, 16, 1, 1, 1>;                                                             \
            using Kernel = KERNEL_ALIAS<ELEMENT_A, Sch>;                                                                \
            __VA_ARGS__;                                                                                                \
        }                                                                                                               \
    } while (0)

} // namespace

void prepack_B_fp16_u4b8(
    cudaStream_t stream, cutlass::vllm_uint4b8_t const* b_in, cutlass::vllm_uint4b8_t* b_out, int k, int n)
{
    prepack_B<cutlass::half_t, cutlass::vllm_uint4b8_t, cutlass::half_t, float>(stream, b_in, b_out, k, n);
}

void prepack_B_bf16_u4b8(
    cudaStream_t stream, cutlass::vllm_uint4b8_t const* b_in, cutlass::vllm_uint4b8_t* b_out, int k, int n)
{
    prepack_B<cutlass::bfloat16_t, cutlass::vllm_uint4b8_t, cutlass::bfloat16_t, float>(stream, b_in, b_out, k, n);
}

void prepack_B_fp16_u4(
    cudaStream_t stream, cutlass::uint4b_t const* b_in, cutlass::uint4b_t* b_out, int k, int n)
{
    prepack_B<cutlass::half_t, cutlass::uint4b_t, cutlass::half_t, float>(stream, b_in, b_out, k, n);
}

void prepack_B_bf16_u4(
    cudaStream_t stream, cutlass::uint4b_t const* b_in, cutlass::uint4b_t* b_out, int k, int n)
{
    prepack_B<cutlass::bfloat16_t, cutlass::uint4b_t, cutlass::bfloat16_t, float>(stream, b_in, b_out, k, n);
}

void machete_mm_fp16_u4b8(cudaStream_t stream, cutlass::half_t const* A, cutlass::vllm_uint4b8_t const* B,
    cutlass::half_t const* scales, cutlass::half_t* D, int m, int n, int k, int group_size, MacheteSchedule schedule,
    void* workspace, size_t workspace_bytes)
{
    MACHETE_DISPATCH_SCHEDULE(cutlass::half_t, GPTQKernel,
        return run_kernel<Kernel, cutlass::half_t>(
            stream, A, B, scales, nullptr, D, m, n, k, group_size, workspace, workspace_bytes));
    MACHETE_CHECK(false, "unsupported schedule", schedule.name);
}

void machete_mm_bf16_u4b8(cudaStream_t stream, cutlass::bfloat16_t const* A, cutlass::vllm_uint4b8_t const* B,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t* D, int m, int n, int k, int group_size,
    MacheteSchedule schedule, void* workspace, size_t workspace_bytes)
{
    MACHETE_DISPATCH_SCHEDULE(cutlass::bfloat16_t, GPTQKernel,
        return run_kernel<Kernel, cutlass::bfloat16_t>(
            stream, A, B, scales, nullptr, D, m, n, k, group_size, workspace, workspace_bytes));
    MACHETE_CHECK(false, "unsupported schedule", schedule.name);
}

void machete_mm_fp16_u4(cudaStream_t stream, cutlass::half_t const* A, cutlass::uint4b_t const* B,
    cutlass::half_t const* scales, cutlass::half_t const* zeros, cutlass::half_t* D, int m, int n, int k,
    int group_size, MacheteSchedule schedule, void* workspace, size_t workspace_bytes)
{
    MACHETE_DISPATCH_SCHEDULE(cutlass::half_t, AWQKernel,
        return run_kernel<Kernel, cutlass::half_t>(
            stream, A, B, scales, zeros, D, m, n, k, group_size, workspace, workspace_bytes));
    MACHETE_CHECK(false, "unsupported schedule", schedule.name);
}

void machete_mm_bf16_u4(cudaStream_t stream, cutlass::bfloat16_t const* A, cutlass::uint4b_t const* B,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* zeros, cutlass::bfloat16_t* D, int m, int n, int k,
    int group_size, MacheteSchedule schedule, void* workspace, size_t workspace_bytes)
{
    MACHETE_DISPATCH_SCHEDULE(cutlass::bfloat16_t, AWQKernel,
        return run_kernel<Kernel, cutlass::bfloat16_t>(
            stream, A, B, scales, zeros, D, m, n, k, group_size, workspace, workspace_bytes));
    MACHETE_CHECK(false, "unsupported schedule", schedule.name);
}

size_t machete_get_workspace_size_fp16_u4b8(cutlass::half_t const* A, cutlass::vllm_uint4b8_t const* B,
    cutlass::half_t const* scales, cutlass::half_t* D, int m, int n, int k, int group_size, MacheteSchedule schedule)
{
    cudaStream_t stream{};
    MACHETE_DISPATCH_SCHEDULE(cutlass::half_t, GPTQKernel,
        return workspace_size<Kernel, cutlass::half_t>(stream, A, B, scales, nullptr, D, m, n, k, group_size));
    MACHETE_CHECK(false, "unsupported schedule", schedule.name);
}

size_t machete_get_workspace_size_bf16_u4b8(cutlass::bfloat16_t const* A, cutlass::vllm_uint4b8_t const* B,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t* D, int m, int n, int k, int group_size,
    MacheteSchedule schedule)
{
    cudaStream_t stream{};
    MACHETE_DISPATCH_SCHEDULE(cutlass::bfloat16_t, GPTQKernel,
        return workspace_size<Kernel, cutlass::bfloat16_t>(stream, A, B, scales, nullptr, D, m, n, k, group_size));
    MACHETE_CHECK(false, "unsupported schedule", schedule.name);
}

size_t machete_get_workspace_size_fp16_u4(cutlass::half_t const* A, cutlass::uint4b_t const* B,
    cutlass::half_t const* scales, cutlass::half_t const* zeros, cutlass::half_t* D, int m, int n, int k,
    int group_size, MacheteSchedule schedule)
{
    cudaStream_t stream{};
    MACHETE_DISPATCH_SCHEDULE(cutlass::half_t, AWQKernel,
        return workspace_size<Kernel, cutlass::half_t>(stream, A, B, scales, zeros, D, m, n, k, group_size));
    MACHETE_CHECK(false, "unsupported schedule", schedule.name);
}

size_t machete_get_workspace_size_bf16_u4(cutlass::bfloat16_t const* A, cutlass::uint4b_t const* B,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* zeros, cutlass::bfloat16_t* D, int m, int n,
    int k, int group_size, MacheteSchedule schedule)
{
    cudaStream_t stream{};
    MACHETE_DISPATCH_SCHEDULE(cutlass::bfloat16_t, AWQKernel,
        return workspace_size<Kernel, cutlass::bfloat16_t>(stream, A, B, scales, zeros, D, m, n, k, group_size));
    MACHETE_CHECK(false, "unsupported schedule", schedule.name);
}

} // namespace machete_standalone
