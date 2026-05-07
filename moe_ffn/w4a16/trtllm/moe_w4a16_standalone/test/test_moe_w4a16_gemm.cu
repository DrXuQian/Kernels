/*
 * Standalone smoke test for TensorRT-LLM MoE grouped GEMM:
 *   activation: FP16 or BF16
 *   weight: INT4 (cutlass::uint4b_t), groupwise scale-only
 *
 * The input rows are already grouped by expert. This test intentionally does
 * not include routing, permutation, activation fusion, or finalize fusion.
 */

#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h"

#include "cutlass/numeric_types.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tk = tensorrt_llm::kernels::cutlass_kernels;
namespace tc = tensorrt_llm::cutlass_extensions;

namespace
{

struct Args
{
    std::string dtype = "fp16";
    int experts = 4;
    int m_per_expert = 16;
    int n = 128;
    int k = 128;
    int group_size = 128;
    int warmup = 5;
    int iters = 20;
    int tile_enum = static_cast<int>(tc::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64);
    int stages = 2;
    std::string tactic_file;
    bool verify = false;
    bool list_configs = false;
    bool sweep_configs = false;
};

void check_cuda(cudaError_t status, char const* expr, char const* file, int line)
{
    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error at %s:%d: %s: %s (%d)\n", file, line, expr, cudaGetErrorString(status),
            static_cast<int>(status));
        std::exit(1);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

template <typename T>
T from_float(float value);

template <>
half from_float<half>(float value)
{
    return __float2half(value);
}

template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <typename T>
float to_float(T value);

template <>
float to_float<half>(half value)
{
    return __half2float(value);
}

template <>
float to_float<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T>
char const* dtype_name()
{
    return std::is_same_v<T, half> ? "fp16" : "bf16";
}

bool parse_int(char const* arg, char const* key, int& out)
{
    size_t const len = std::strlen(key);
    if (std::strncmp(arg, key, len) != 0)
    {
        return false;
    }
    out = std::strtol(arg + len, nullptr, 10);
    return true;
}

void print_usage(char const* name)
{
    std::printf(
        "Usage: %s [--dtype=fp16|bf16] [--experts=N] [--m_per_expert=N] [--n=N] [--k=N]\n"
        "          [--group_size=N] [--warmup=N] [--iters=N] [--tile_enum=N] [--stages=N]\n"
        "          [--tactic=<file>] [--verify] [--list_configs] [--sweep_configs]\n",
        name);
}

bool parse_args(int argc, char** argv, Args& args)
{
    for (int i = 1; i < argc; ++i)
    {
        char const* arg = argv[i];
        if (std::strncmp(arg, "--dtype=", 8) == 0)
        {
            args.dtype = arg + 8;
        }
        else if (parse_int(arg, "--experts=", args.experts) || parse_int(arg, "--m_per_expert=", args.m_per_expert)
            || parse_int(arg, "--n=", args.n) || parse_int(arg, "--k=", args.k)
            || parse_int(arg, "--group_size=", args.group_size) || parse_int(arg, "--warmup=", args.warmup)
            || parse_int(arg, "--iters=", args.iters) || parse_int(arg, "--tile_enum=", args.tile_enum)
            || parse_int(arg, "--stages=", args.stages))
        {
        }
        else if (std::strcmp(arg, "--verify") == 0)
        {
            args.verify = true;
        }
        else if (std::strcmp(arg, "--list_configs") == 0)
        {
            args.list_configs = true;
        }
        else if (std::strcmp(arg, "--sweep_configs") == 0)
        {
            args.sweep_configs = true;
        }
        else if (std::strncmp(arg, "--tactic=", 9) == 0)
        {
            args.tactic_file = arg + 9;
        }
        else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0)
        {
            print_usage(argv[0]);
            return false;
        }
        else
        {
            std::fprintf(stderr, "Unknown argument: %s\n", arg);
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

tc::CutlassGemmConfig make_sm80_config(int tile_enum, int stages)
{
    tc::CutlassGemmConfig config{};
    config.tile_config_sm80 = static_cast<tc::CutlassTileConfig>(tile_enum);
    config.stages = stages;
    config.split_k_factor = 1;
    config.split_k_style = tc::SplitKStyle::NO_SPLIT_K;
    config.sm_version = 80;
    config.enableCudaKernel = false;
    return config;
}

std::string serialize_config(tc::CutlassGemmConfig const& c)
{
    std::ostringstream s;
    if (c.enableCudaKernel)
    {
        s << "cuda=1";
    }
    else if (c.is_tma_warp_specialized)
    {
        s << "cuda=0,tma=1,sm=" << c.sm_version << ",tile90=" << static_cast<int>(c.tile_config_sm90)
          << ",ml=" << static_cast<int>(c.mainloop_schedule) << ",el=" << static_cast<int>(c.epilogue_schedule)
          << ",cl=" << static_cast<int>(c.cluster_shape);
    }
    else
    {
        s << "cuda=0,tma=0"
          << ",tile80=" << static_cast<int>(c.tile_config_sm80) << ",stages=" << c.stages
          << ",splitk=" << c.split_k_factor << ",sk_style=" << static_cast<int>(c.split_k_style);
    }
    return s.str();
}

bool deserialize_config(std::string const& str, tc::CutlassGemmConfig& out)
{
    out = tc::CutlassGemmConfig{};

    auto get_int = [&](char const* key) -> int
    {
        std::string needle = std::string(key) + "=";
        auto pos = str.find(needle);
        if (pos == std::string::npos)
        {
            return -999;
        }
        return std::atoi(str.c_str() + pos + needle.size());
    };

    int const cuda = get_int("cuda");
    if (cuda == 1)
    {
        out.enableCudaKernel = true;
        return true;
    }

    int const tma = get_int("tma");
    if (tma == 1)
    {
        out.is_tma_warp_specialized = true;
        out.sm_version = get_int("sm");
        out.tile_config_sm90 = static_cast<tc::CutlassTileConfigSM90>(get_int("tile90"));
        out.mainloop_schedule = static_cast<tc::MainloopScheduleType>(get_int("ml"));
        out.epilogue_schedule = static_cast<tc::EpilogueScheduleType>(get_int("el"));
        out.cluster_shape = static_cast<tc::ClusterShape>(get_int("cl"));
        return true;
    }

    out.tile_config_sm80 = static_cast<tc::CutlassTileConfig>(get_int("tile80"));
    out.stages = get_int("stages");
    out.split_k_factor = get_int("splitk");
    out.split_k_style = static_cast<tc::SplitKStyle>(get_int("sk_style"));
    out.sm_version = 80;
    return true;
}

// Tactic file serialization mirrors fpA_intB_standalone:
//   key|cuda=0,tma=0,tile80=...,stages=...,splitk=...,sk_style=...
// The MoE key includes dtype and expert layout because those are part of the
// grouped-GEMM problem shape.
std::string tactic_key(char const* dtype, int experts, int m_per_expert, int n, int k, int group_size)
{
    std::ostringstream s;
    s << dtype << "," << experts << "," << m_per_expert << "," << n << "," << k << "," << group_size << "|";
    return s.str();
}

bool load_tactic(std::string const& path, char const* dtype, int experts, int m_per_expert, int n, int k, int group_size,
    tc::CutlassGemmConfig& config)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        return false;
    }

    std::string const prefix = tactic_key(dtype, experts, m_per_expert, n, k, group_size);
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        if (line.compare(0, prefix.size(), prefix) == 0)
        {
            return deserialize_config(line.substr(prefix.size()), config);
        }
    }
    return false;
}

void save_tactic(std::string const& path, char const* dtype, int experts, int m_per_expert, int n, int k,
    int group_size,
    tc::CutlassGemmConfig const& config)
{
    std::ofstream f(path, std::ios::app);
    if (!f.is_open())
    {
        std::fprintf(stderr, "Warning: cannot write tactic file %s\n", path.c_str());
        return;
    }
    f << tactic_key(dtype, experts, m_per_expert, n, k, group_size) << serialize_config(config) << "\n";
}

bool same_sm80_config(tc::CutlassGemmConfig const& a, tc::CutlassGemmConfig const& b)
{
    return !a.enableCudaKernel && !b.enableCudaKernel && !a.is_tma_warp_specialized && !b.is_tma_warp_specialized
        && a.tile_config_sm80 == b.tile_config_sm80 && a.stages == b.stages && a.split_k_factor == b.split_k_factor
        && a.split_k_style == b.split_k_style && a.sm_version == b.sm_version;
}

template <typename T>
std::vector<tc::CutlassGemmConfig> benchmark_configs(const Args& args)
{
    if (!args.sweep_configs)
    {
        return {make_sm80_config(args.tile_enum, args.stages)};
    }

    auto configs = tk::MoeGemmRunner<T, cutlass::uint4b_t, T>::getConfigs(80, false);
    std::vector<tc::CutlassGemmConfig> out;
    std::set<std::tuple<int, int, int, int>> seen;
    for (auto config : configs)
    {
        if (config.is_tma_warp_specialized || config.split_k_factor != 1)
        {
            continue;
        }
        auto const key = std::make_tuple(
            static_cast<int>(config.tile_config_sm80), config.stages, config.split_k_factor, config.sm_version);
        if (!seen.insert(key).second)
        {
            continue;
        }
        out.push_back(config);
    }
    return out;
}

template <typename T>
bool is_supported_config(tc::CutlassGemmConfig const& config)
{
    if (config.enableCudaKernel || config.is_tma_warp_specialized)
    {
        return false;
    }
    Args sweep_args{};
    sweep_args.sweep_configs = true;
    for (auto const& candidate : benchmark_configs<T>(sweep_args))
    {
        if (same_sm80_config(config, candidate))
        {
            return true;
        }
    }
    return false;
}

template <typename T>
void print_configs()
{
    auto configs = tk::MoeGemmRunner<T, cutlass::uint4b_t, T>::getConfigs(80, false);
    std::set<std::tuple<int, int, int, int>> seen;
    for (auto const& config : configs)
    {
        if (config.is_tma_warp_specialized)
        {
            continue;
        }
        auto const key = std::make_tuple(
            static_cast<int>(config.tile_config_sm80), config.stages, config.split_k_factor, config.sm_version);
        if (!seen.insert(key).second)
        {
            continue;
        }
        std::printf("tile_enum=%d stages=%d split_k=%d sm=%d\n", static_cast<int>(config.tile_config_sm80),
            config.stages, config.split_k_factor, config.sm_version);
    }
}

template <typename T>
int run(const Args& args)
{
    if (args.experts <= 0 || args.m_per_expert <= 0 || args.n <= 0 || args.k <= 0 || args.group_size <= 0)
    {
        std::fprintf(stderr, "All shape arguments must be positive.\n");
        return 1;
    }
    if (args.warmup < 0 || args.iters <= 0)
    {
        std::fprintf(stderr, "warmup must be >= 0 and iters must be > 0.\n");
        return 1;
    }
    if (args.k % args.group_size != 0)
    {
        std::fprintf(stderr, "k must be divisible by group_size.\n");
        return 1;
    }
    if (args.group_size != 128 && args.group_size != args.k)
    {
        std::fprintf(stderr, "This CUTLASS MoE W4A16 path supports group_size=128 or group_size=k.\n");
        return 1;
    }
    if (args.k % 64 != 0 || args.n % 128 != 0)
    {
        std::fprintf(stderr, "Default config expects k multiple of 64 and n multiple of 128.\n");
        return 1;
    }

    if (args.list_configs)
    {
        print_configs<T>();
        return 0;
    }

    int const total_rows = args.experts * args.m_per_expert;
    size_t const a_elems = static_cast<size_t>(total_rows) * args.k;
    size_t const c_elems = static_cast<size_t>(total_rows) * args.n;
    size_t const packed_weight_bytes = static_cast<size_t>(args.experts) * args.k * args.n / 2;
    size_t const scale_elems = static_cast<size_t>(args.experts) * (args.k / args.group_size) * args.n;

    std::vector<T> h_a(a_elems);
    std::vector<T> h_c(c_elems);
    std::vector<T> h_scales(scale_elems, from_float<T>(1.0f));
    std::vector<uint8_t> h_b(packed_weight_bytes, 0x99);
    std::vector<int64_t> h_offsets(args.experts);

    for (int expert = 0; expert < args.experts; ++expert)
    {
        h_offsets[expert] = static_cast<int64_t>(expert + 1) * args.m_per_expert;
    }

    for (size_t i = 0; i < a_elems; ++i)
    {
        float const value = (static_cast<int>(i % 17) - 8) * 0.03125f;
        h_a[i] = from_float<T>(value);
    }

    T* d_a = nullptr;
    T* d_c = nullptr;
    T* d_scales = nullptr;
    uint8_t* d_b = nullptr;
    int64_t* d_offsets = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, a_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_c, c_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_scales, scale_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b, packed_weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int64_t)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_elems * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), c_elems * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales, h_scales.data(), scale_elems * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), packed_weight_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    tk::GroupedGemmInput<T, cutlass::uint4b_t, T, T> inputs{.A = d_a,
        .total_tokens_including_expert = d_offsets,
        .B = reinterpret_cast<cutlass::uint4b_t const*>(d_b),
        .scales = d_scales,
        .zeros = nullptr,
        .biases = nullptr,
        .C = d_c,
        .alpha_scales = nullptr,
        .occupancy = nullptr,
        .activation_type = tk::ActivationType::Identity,
        .num_rows = total_rows,
        .n = args.n,
        .k = args.k,
        .num_experts = args.experts,
        .groupwise_quant_group_size = args.group_size,
        .bias_is_broadcast = true,
        .use_fused_moe = false,
        .stream = 0,
        .gemm_config = make_sm80_config(args.tile_enum, args.stages)};

    tk::MoeGemmRunner<T, cutlass::uint4b_t, T> runner;
    tk::TmaWarpSpecializedGroupedGemmInput hopper_inputs{};

    float best_ms = std::numeric_limits<float>::infinity();
    tc::CutlassGemmConfig best_config{};

    std::vector<tc::CutlassGemmConfig> configs_to_run;
    bool save_best_tactic = false;
    if (!args.tactic_file.empty() && !args.sweep_configs)
    {
        tc::CutlassGemmConfig cached_config{};
        if (load_tactic(args.tactic_file, dtype_name<T>(), args.experts, args.m_per_expert, args.n, args.k,
                args.group_size, cached_config))
        {
            if (!is_supported_config<T>(cached_config))
            {
                std::printf("tactic cache entry unsupported for this MoE W4A16 build: profiling...\n");
                Args sweep_args = args;
                sweep_args.sweep_configs = true;
                configs_to_run = benchmark_configs<T>(sweep_args);
                save_best_tactic = true;
            }
            else
            {
                std::printf("tactic cache HIT from %s\n", args.tactic_file.c_str());
                configs_to_run.push_back(cached_config);
            }
        }
        else
        {
            std::printf("tactic cache MISS: profiling...\n");
            Args sweep_args = args;
            sweep_args.sweep_configs = true;
            configs_to_run = benchmark_configs<T>(sweep_args);
            save_best_tactic = true;
        }
    }
    else
    {
        configs_to_run = benchmark_configs<T>(args);
    }

    for (auto const& config : configs_to_run)
    {
        inputs.gemm_config = config;
        try
        {
            for (int i = 0; i < args.warmup; ++i)
            {
                runner.moeGemm(inputs, hopper_inputs);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t start{};
            cudaEvent_t stop{};
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < args.iters; ++i)
            {
                runner.moeGemm(inputs, hopper_inputs);
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));

            float const avg_ms = elapsed_ms / std::max(args.iters, 1);
            if (args.sweep_configs)
            {
                std::printf("config dtype=%s tile_enum=%d stages=%d split_k=%d sm=%d avg_ms=%.4f\n",
                    dtype_name<T>(), static_cast<int>(config.tile_config_sm80), config.stages,
                    config.split_k_factor, config.sm_version, avg_ms);
            }
            if (avg_ms < best_ms)
            {
                best_ms = avg_ms;
                best_config = config;
            }
        }
        catch (std::exception const& e)
        {
            if (args.sweep_configs)
            {
                std::fprintf(stderr, "config dtype=%s tile_enum=%d stages=%d failed: %s\n",
                    dtype_name<T>(), static_cast<int>(config.tile_config_sm80), config.stages, e.what());
            }
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    if (!std::isfinite(best_ms))
    {
        std::fprintf(stderr, "No valid GEMM config ran.\n");
        return 1;
    }
    if (save_best_tactic)
    {
        save_tactic(args.tactic_file, dtype_name<T>(), args.experts, args.m_per_expert, args.n, args.k,
            args.group_size, best_config);
        std::printf("tactic saved to %s\n", args.tactic_file.c_str());
    }
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, c_elems * sizeof(T), cudaMemcpyDeviceToHost));

    bool ok = true;
    float max_abs_err = 0.0f;
    if (args.verify)
    {
        for (int row = 0; row < total_rows; ++row)
        {
            float expected = 0.0f;
            for (int kk = 0; kk < args.k; ++kk)
            {
                expected += to_float(h_a[static_cast<size_t>(row) * args.k + kk]);
            }
            for (int nn = 0; nn < args.n; ++nn)
            {
                float const got = to_float(h_c[static_cast<size_t>(row) * args.n + nn]);
                float const abs_err = std::fabs(got - expected);
                max_abs_err = std::max(max_abs_err, abs_err);
                if (abs_err > 0.35f)
                {
                    ok = false;
                }
            }
        }
    }

    std::printf("dtype=%s experts=%d m_per_expert=%d n=%d k=%d group_size=%d tile_enum=%d stages=%d avg_ms=%.4f",
        dtype_name<T>(), args.experts, args.m_per_expert, args.n, args.k, args.group_size,
        static_cast<int>(best_config.tile_config_sm80), best_config.stages, best_ms);
    if (args.verify)
    {
        std::printf(" verify=%s max_abs_err=%.6f", ok ? "ok" : "failed", max_abs_err);
    }
    std::printf("\n");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_scales));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_offsets));

    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parse_args(argc, argv, args))
    {
        return 1;
    }

    if (args.dtype == "fp16")
    {
        return run<half>(args);
    }
    if (args.dtype == "bf16")
    {
        return run<__nv_bfloat16>(args);
    }

    std::fprintf(stderr, "Unsupported dtype: %s\n", args.dtype.c_str());
    return 1;
}
