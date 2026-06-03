// Standalone vLLM-style block-wise FP8 GEMM benchmark for SM90.
//
// Computes row-major D[M,N] = A[M,K] * B[K,N], where A and B are FP8 E4M3
// and scales are FP32 block scales with granularity A=(1,128), B=(128,128).
// This is adapted from vLLM's cutlass_scaled_mm_blockwise_sm90_fp8 path, but
// uses raw CUDA pointers instead of torch::Tensor/ATen.
//
// Usage:
//   general/bench_cutlass_block_fp8_gemm --m=3823 --n=6144 --k=3072 --bench 0 1
//   general/bench_cutlass_block_fp8_gemm --m=1 --n=6144 --k=3072 --bench 0 1
//
// vLLM pads Hopper CUTLASS block-FP8 GEMM to M % 4 == 0. This benchmark does
// the same allocation-side padding and times only the GEMM kernel.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"

#include "bench_timer.h"

#define CHECK_CUDA(e)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t _e = (e);                                                                                          \
        if (_e != cudaSuccess)                                                                                         \
        {                                                                                                              \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_CUTLASS(e)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cutlass::Status _e = (e);                                                                                      \
        if (_e != cutlass::Status::kSuccess)                                                                           \
        {                                                                                                              \
            std::fprintf(stderr, "CUTLASS %s:%d: status=%d\n", __FILE__, __LINE__, static_cast<int>(_e));             \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace
{

using namespace cute;

template <class Stride0, class Stride2, class Shape>
CUTE_HOST_DEVICE constexpr auto make_packed_stride(cute::Stride<Stride0, cute::Int<1>, Stride2>, Shape const& shape)
{
    int64_t const dim0 = static_cast<int64_t>(cute::get<0>(shape));
    int64_t const dim1 = static_cast<int64_t>(cute::get<1>(shape));
    int64_t const dim2 = static_cast<int64_t>(cute::get<2>(shape));

    Stride0 s0{};
    if constexpr (!cute::is_static<Stride0>::value)
    {
        s0 = static_cast<Stride0>(dim1);
    }

    Stride2 s2{};
    if constexpr (!cute::is_static<Stride2>::value)
    {
        s2 = static_cast<Stride2>((dim2 == 1) ? 0 : (dim0 * dim1));
    }

    return cute::make_stride(s0, cute::Int<1>{}, s2);
}

template <class Stride1, class Stride2, class Shape>
CUTE_HOST_DEVICE constexpr auto make_packed_stride(cute::Stride<cute::Int<1>, Stride1, Stride2>, Shape const& shape)
{
    int64_t const dim0 = static_cast<int64_t>(cute::get<0>(shape));
    int64_t const dim1 = static_cast<int64_t>(cute::get<1>(shape));
    int64_t const dim2 = static_cast<int64_t>(cute::get<2>(shape));

    Stride1 s1{};
    if constexpr (!cute::is_static<Stride1>::value)
    {
        s1 = static_cast<Stride1>(dim0);
    }

    Stride2 s2{};
    if constexpr (!cute::is_static<Stride2>::value)
    {
        s2 = static_cast<Stride2>((dim2 == 1) ? 0 : (dim0 * dim1));
    }

    return cute::make_stride(cute::Int<1>{}, s1, s2);
}

struct Options
{
    int m = 3823;
    int n = 6144;
    int k = 3072;
    std::string out_dtype = "fp16";
    bool pad_m4 = true;
};

bool starts_with(char const* s, char const* prefix)
{
    return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

int parse_int(char const* s)
{
    return std::atoi(s);
}

Options parse_args(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--m") == 0)
        {
            opt.m = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--m="))
        {
            opt.m = parse_int(argv[i] + 4);
        }
        else if (std::strcmp(argv[i], "--n") == 0)
        {
            opt.n = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--n="))
        {
            opt.n = parse_int(argv[i] + 4);
        }
        else if (std::strcmp(argv[i], "--k") == 0)
        {
            opt.k = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--k="))
        {
            opt.k = parse_int(argv[i] + 4);
        }
        else if (std::strcmp(argv[i], "--out-dtype") == 0 || std::strcmp(argv[i], "--output-dtype") == 0)
        {
            opt.out_dtype = argv[++i];
        }
        else if (starts_with(argv[i], "--out-dtype="))
        {
            opt.out_dtype = argv[i] + 12;
        }
        else if (starts_with(argv[i], "--output-dtype="))
        {
            opt.out_dtype = argv[i] + 15;
        }
        else if (std::strcmp(argv[i], "--pad-m4") == 0)
        {
            opt.pad_m4 = true;
        }
        else if (std::strcmp(argv[i], "--no-pad-m4") == 0)
        {
            opt.pad_m4 = false;
        }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("Usage: %s --m M --n N --k K [--out-dtype fp16|bf16] [--pad-m4|--no-pad-m4]"
                        " [--bench W I]\n",
                argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            std::exit(1);
        }
    }

    if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0)
    {
        std::fprintf(stderr, "m, n, and k must be positive\n");
        std::exit(1);
    }
    if (opt.out_dtype != "fp16" && opt.out_dtype != "bf16")
    {
        std::fprintf(stderr, "out-dtype must be fp16 or bf16\n");
        std::exit(1);
    }
    if (opt.k % 128 != 0 || opt.n % 128 != 0)
    {
        std::fprintf(stderr, "block-FP8 path requires n and k to be multiples of 128\n");
        std::exit(1);
    }
    return opt;
}

int round_up(int value, int multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

template <class OutType, int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK,
    class MmaTileShape, class ClusterShape, class EpilogueScheduler, class MainloopScheduler>
struct BlockFp8Gemm
{
    using ElementAB = cutlass::float_e4m3_t;

    using ElementA = ElementAB;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB = ElementAB;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    using ElementD = OutType;
    using LayoutD = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ElementC = void;
    using LayoutC = LayoutD;
    static constexpr int AlignmentC = AlignmentD;

    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBlockScale = float;

    using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
        ScaleGranularityK, cute::GMMA::Major::MN, cute::GMMA::Major::K>;

    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using ElementScalar = float;
    using DefaultOperation
        = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
        MmaTileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, EpilogueScheduler,
        DefaultOperation>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA,
        cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
        ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopScheduler>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
        CollectiveEpilogue>;
};

template <typename OutType>
using VllmSm90BlockFp8Gemm = BlockFp8Gemm<OutType, 1, 128, 128, Shape<_128, _128, _128>, Shape<_1, _2, _1>,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum>;

template <typename Gemm>
size_t scale_a_elements(int m, int n, int k)
{
    using ScaleConfig = typename Gemm::ScaleConfig;
    auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    return static_cast<size_t>(cute::size(layout_sfa));
}

template <typename Gemm>
size_t scale_b_elements(int m, int n, int k)
{
    using ScaleConfig = typename Gemm::ScaleConfig;
    auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    return static_cast<size_t>(cute::size(layout_sfb));
}

template <typename OutType>
int run_block_fp8_gemm(Options const& opt, BenchTimer& timer)
{
    using Gemm = VllmSm90BlockFp8Gemm<OutType>;
    using GemmKernel = typename Gemm::GemmKernel;
    using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using LayoutSFA = typename Gemm::LayoutSFA;
    using LayoutSFB = typename Gemm::LayoutSFB;
    using ScaleConfig = typename Gemm::ScaleConfig;
    using ElementAB = typename Gemm::ElementAB;
    using ElementD = typename Gemm::ElementD;
    using ElementBlockScale = typename Gemm::ElementBlockScale;

    int const m_kernel = opt.pad_m4 ? round_up(opt.m, 4) : opt.m;
    if (m_kernel % 4 != 0)
    {
        std::fprintf(stderr, "block-FP8 SM90 CUTLASS path requires padded m %% 4 == 0; got m=%d\n", m_kernel);
        return 1;
    }

    ElementAB* d_a = nullptr;
    ElementAB* d_b = nullptr;
    ElementBlockScale* d_a_scales = nullptr;
    ElementBlockScale* d_b_scales = nullptr;
    ElementD* d_out = nullptr;
    void* d_workspace = nullptr;

    size_t const a_elems = static_cast<size_t>(m_kernel) * opt.k;
    size_t const b_elems = static_cast<size_t>(opt.k) * opt.n;
    size_t const out_elems = static_cast<size_t>(m_kernel) * opt.n;
    size_t const a_scale_elems = scale_a_elements<Gemm>(m_kernel, opt.n, opt.k);
    size_t const b_scale_elems = scale_b_elements<Gemm>(m_kernel, opt.n, opt.k);

    CHECK_CUDA(cudaMalloc(&d_a, a_elems * sizeof(ElementAB)));
    CHECK_CUDA(cudaMalloc(&d_b, b_elems * sizeof(ElementAB)));
    CHECK_CUDA(cudaMalloc(&d_a_scales, a_scale_elems * sizeof(ElementBlockScale)));
    CHECK_CUDA(cudaMalloc(&d_b_scales, b_scale_elems * sizeof(ElementBlockScale)));
    CHECK_CUDA(cudaMalloc(&d_out, out_elems * sizeof(ElementD)));

    CHECK_CUDA(cudaMemset(d_a, 0, a_elems * sizeof(ElementAB)));
    CHECK_CUDA(cudaMemset(d_b, 0, b_elems * sizeof(ElementAB)));
    CHECK_CUDA(cudaMemset(d_out, 0, out_elems * sizeof(ElementD)));
    std::vector<float> h_a_scales(a_scale_elems, 1.0f);
    std::vector<float> h_b_scales(b_scale_elems, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_a_scales, h_a_scales.data(), a_scale_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_scales, h_b_scales.data(), b_scale_elems * sizeof(float), cudaMemcpyHostToDevice));

    StrideA a_stride = make_packed_stride(StrideA{}, cute::make_shape(m_kernel, opt.k, 1));
    StrideB b_stride = make_packed_stride(StrideB{}, cute::make_shape(opt.n, opt.k, 1));
    StrideC c_stride = make_packed_stride(StrideC{}, cute::make_shape(m_kernel, opt.n, 1));

    LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m_kernel, opt.n, opt.k, 1));
    LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m_kernel, opt.n, opt.k, 1));

    typename GemmKernel::MainloopArguments mainloop_args{};
    mainloop_args.ptr_A = d_a;
    mainloop_args.dA = a_stride;
    mainloop_args.ptr_B = d_b;
    mainloop_args.dB = b_stride;
    mainloop_args.ptr_SFA = d_a_scales;
    mainloop_args.layout_SFA = layout_sfa;
    mainloop_args.ptr_SFB = d_b_scales;
    mainloop_args.layout_SFB = layout_sfb;

    typename GemmKernel::EpilogueArguments epilogue_args{{}, d_out, c_stride, d_out, c_stride};
    auto problem_shape = cute::make_shape(m_kernel, opt.n, opt.k, 1);

    cutlass::KernelHardwareInfo hw_info;
    typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, problem_shape, mainloop_args,
        epilogue_args, hw_info, {}};

    GemmOp gemm_op;
    CHECK_CUTLASS(gemm_op.can_implement(args));
    size_t const workspace_size = gemm_op.get_workspace_size(args);
    if (workspace_size > 0)
    {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    }

    cudaStream_t stream = nullptr;
    std::printf("block_fp8_gemm: requested_m=%d kernel_m=%d n=%d k=%d out=%s a_scale_elems=%zu b_scale_elems=%zu"
                " workspace=%zu\n",
        opt.m, m_kernel, opt.n, opt.k, opt.out_dtype.c_str(), a_scale_elems, b_scale_elems, workspace_size);

    timer.run([&]() {
        cutlass::Status status = gemm_op.run(args, d_workspace, stream);
        if (status != cutlass::Status::kSuccess)
        {
            std::fprintf(stderr, "CUTLASS run failed: status=%d\n", static_cast<int>(status));
            std::exit(1);
        }
    });
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(d_workspace);
    cudaFree(d_out);
    cudaFree(d_b_scales);
    cudaFree(d_a_scales);
    cudaFree(d_b);
    cudaFree(d_a);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    Options opt = parse_args(argc, argv);
    if (opt.out_dtype == "bf16")
    {
        return run_block_fp8_gemm<cutlass::bfloat16_t>(opt, timer);
    }
    return run_block_fp8_gemm<cutlass::half_t>(opt, timer);
}
