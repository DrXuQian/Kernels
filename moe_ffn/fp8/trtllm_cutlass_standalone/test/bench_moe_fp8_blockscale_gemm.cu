/*
 * Standalone MiniMax-style FP8 block-scale grouped MoE GEMM benchmark.
 *
 * Source lineage:
 *   TensorRT-LLM fp8_blockscale_gemm runner, vendored from the TRT-LLM/CUTLASS
 *   kernel layer shipped in FlashInfer. This binary does not link FlashInfer,
 *   TVM, Python, or TensorRT plugins.
 *
 * Timed region:
 *   One grouped-with-offset block-FP8 GEMM kernel. Inputs, scales, offsets, and
 *   output storage are created before timing.
 */

#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"

#include "bench_timer.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fp8bs = tensorrt_llm::kernels::fp8_blockscale_gemm;

namespace
{

#define CHECK_CUDA(expr)                                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
        cudaError_t _status = (expr);                                                                                 \
        if (_status != cudaSuccess)                                                                                   \
        {                                                                                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s: %s (%d)\n", __FILE__, __LINE__, #expr,                    \
                cudaGetErrorString(_status), static_cast<int>(_status));                                              \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

struct Args
{
    int experts = 8;
    int m_per_expert = 3823;
    int n = 3072;
    int k = 3072;
    int warmup = 0;
    int iters = 1;
    bool deep_gemm = false;
    int tile_m = 0;
    int tile_n = 0;
    bool sweep_configs = false;
    std::string tactic_file;
};

struct StaticTileConfig
{
    int tile_m = 128;
    int tile_n = 64;
};

StaticTileConfig default_static_tile_config(Args const& args)
{
    if (args.m_per_expert <= 64)
    {
        return {64, 128};
    }
    return {128, 64};
}

bool starts_with(char const* s, char const* prefix)
{
    return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

int parse_int(char const* s)
{
    return std::atoi(s);
}

void usage(char const* prog)
{
    std::printf(
        "Usage: %s [--experts=N] [--m_per_expert=N] [--n=N] [--k=N]\n"
        "          [--bench W I] [--warmup=N] [--iters=N] [--deep-gemm]\n"
        "          [--tile_m=64|128] [--tile_n=64|128]\n"
        "          [--tactic=file] [--sweep_configs]\n"
        "\n"
        "MiniMax examples:\n"
        "  gate/up decode:   %s --m_per_expert=1 --n=3072 --k=3072 --bench 0 1\n"
        "  down decode:      %s --m_per_expert=1 --n=3072 --k=1536 --bench 0 1\n"
        "  gate/up prefill:  %s --m_per_expert=3823 --n=3072 --k=3072 --bench 0 1\n"
        "  down prefill:     %s --m_per_expert=3823 --n=3072 --k=1536 --bench 0 1\n",
        prog, prog, prog, prog, prog);
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        char const* a = argv[i];
        if (starts_with(a, "--experts="))
        {
            args.experts = parse_int(a + 10);
        }
        else if (starts_with(a, "--m_per_expert="))
        {
            args.m_per_expert = parse_int(a + 15);
        }
        else if (starts_with(a, "--n="))
        {
            args.n = parse_int(a + 4);
        }
        else if (starts_with(a, "--k="))
        {
            args.k = parse_int(a + 4);
        }
        else if (starts_with(a, "--warmup="))
        {
            args.warmup = parse_int(a + 9);
        }
        else if (starts_with(a, "--iters="))
        {
            args.iters = parse_int(a + 8);
        }
        else if (starts_with(a, "--tile_m="))
        {
            args.tile_m = parse_int(a + 9);
        }
        else if (starts_with(a, "--tile_n="))
        {
            args.tile_n = parse_int(a + 9);
        }
        else if (starts_with(a, "--tactic="))
        {
            args.tactic_file = a + 9;
        }
        else if (std::strcmp(a, "--sweep_configs") == 0)
        {
            args.sweep_configs = true;
        }
        else if (std::strcmp(a, "--bench") == 0)
        {
            if (i + 2 >= argc)
            {
                usage(argv[0]);
                std::exit(1);
            }
            args.warmup = parse_int(argv[++i]);
            args.iters = parse_int(argv[++i]);
        }
        else if (std::strcmp(a, "--deep-gemm") == 0)
        {
            args.deep_gemm = true;
        }
        else if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0)
        {
            usage(argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", a);
            usage(argv[0]);
            std::exit(1);
        }
    }

    if (args.experts <= 0 || args.m_per_expert <= 0 || args.n <= 0 || args.k <= 0)
    {
        std::fprintf(stderr, "experts, m_per_expert, n, and k must be positive\n");
        std::exit(1);
    }
    if (args.n % 64 != 0 || args.k % 128 != 0)
    {
        std::fprintf(stderr, "this SM90 block-FP8 path requires n %% 64 == 0 and k %% 128 == 0\n");
        std::exit(1);
    }
    if (args.tile_m != 0 && args.tile_m != 64 && args.tile_m != 128)
    {
        std::fprintf(stderr, "--tile_m must be 64 or 128\n");
        std::exit(1);
    }
    if (args.tile_n != 0 && args.tile_n != 64 && args.tile_n != 128)
    {
        std::fprintf(stderr, "--tile_n must be 64 or 128\n");
        std::exit(1);
    }
    if (args.sweep_configs && args.deep_gemm)
    {
        std::fprintf(stderr, "--sweep_configs only covers the static TMA fallback; omit --deep-gemm\n");
        std::exit(1);
    }
    return args;
}

std::string tactic_key(Args const& args)
{
    std::ostringstream s;
    s << args.experts << "," << args.m_per_expert << "," << args.n << "," << args.k << ",1x128,128x128|";
    return s.str();
}

std::string serialize_config(StaticTileConfig cfg)
{
    std::ostringstream s;
    s << "tile=" << cfg.tile_m << "x" << cfg.tile_n << "x128"
      << ",layoutA=row,layoutB=col,layoutD=row,output=bf16,deep_gemm=0";
    return s.str();
}

bool deserialize_config(std::string const& text, StaticTileConfig& cfg)
{
    auto pos = text.find("tile=");
    if (pos == std::string::npos)
    {
        return false;
    }
    int tm = 0;
    int tn = 0;
    int tk = 0;
    if (std::sscanf(text.c_str() + pos, "tile=%dx%dx%d", &tm, &tn, &tk) != 3)
    {
        return false;
    }
    if ((tm != 64 && tm != 128) || (tn != 64 && tn != 128) || tk != 128)
    {
        return false;
    }
    cfg.tile_m = tm;
    cfg.tile_n = tn;
    return true;
}

bool load_tactic(std::string const& path, Args const& args, StaticTileConfig& cfg)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        return false;
    }
    std::string const key = tactic_key(args);
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        if (line.compare(0, key.size(), key) == 0)
        {
            return deserialize_config(line.substr(key.size()), cfg);
        }
    }
    return false;
}

void save_tactic(std::string const& path, Args const& args, StaticTileConfig cfg)
{
    std::ofstream f(path, std::ios::app);
    if (!f.is_open())
    {
        std::fprintf(stderr, "warning: cannot write tactic file: %s\n", path.c_str());
        return;
    }
    f << tactic_key(args) << serialize_config(cfg) << "\n";
}

void set_static_tile_config(StaticTileConfig cfg)
{
    setenv("TRTLLM_FP8_STATIC_TILE_M", std::to_string(cfg.tile_m).c_str(), 1);
    setenv("TRTLLM_FP8_STATIC_TILE_N", std::to_string(cfg.tile_n).c_str(), 1);
}

std::vector<StaticTileConfig> static_tile_search_space()
{
    return {{64, 64}, {64, 128}, {128, 64}, {128, 128}};
}

// Matches TRT-LLM deep_gemm::compute_padded_offset(offset, problem_idx) without
// including the JIT helper in the benchmark translation unit.
int64_t padded_scale_rows(int64_t total_rows, int experts)
{
    constexpr int64_t alignment = 32;
    return (total_rows + experts * (alignment - 1)) / alignment * alignment;
}

void* cuda_alloc_bytes(size_t bytes, char const* name)
{
    void* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, bytes));
    CHECK_CUDA(cudaMemset(ptr, 0, bytes));
    std::printf("  alloc %-12s %zu bytes\n", name, bytes);
    return ptr;
}

template <typename F>
float measure_kernel_ms(F&& launch, int warmup, int iters)
{
    iters = std::max(iters, 1);
    for (int i = 0; i < warmup; ++i)
    {
        launch();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i)
    {
        cudaEvent_t start{};
        cudaEvent_t stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        launch();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&times[static_cast<size_t>(i)], start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

} // namespace

int main(int argc, char** argv)
{
    Args const args = parse_args(argc, argv);
    if (!args.deep_gemm)
    {
        setenv("TRTLLM_DG_ENABLED", "0", 1);
    }

    StaticTileConfig selected_config = default_static_tile_config(args);
    bool const explicit_tile = args.tile_m != 0 || args.tile_n != 0;
    bool auto_selected_tile = true;
    if (explicit_tile)
    {
        selected_config.tile_m = args.tile_m != 0 ? args.tile_m : selected_config.tile_m;
        selected_config.tile_n = args.tile_n != 0 ? args.tile_n : selected_config.tile_n;
        auto_selected_tile = false;
        std::printf("static config from CLI: %s\n", serialize_config(selected_config).c_str());
    }
    else if (!args.tactic_file.empty() && load_tactic(args.tactic_file, args, selected_config))
    {
        auto_selected_tile = false;
        std::printf("tactic cache HIT from %s\n", args.tactic_file.c_str());
        std::printf("selected config: %s\n", serialize_config(selected_config).c_str());
    }
    else if (!args.tactic_file.empty())
    {
        std::printf("tactic cache MISS from %s, using default static config\n", args.tactic_file.c_str());
    }
    if (auto_selected_tile && !args.sweep_configs)
    {
        std::fprintf(stderr,
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "[WARNING][FP8-MOE] AUTOMATIC STATIC TILE SELECTION IS ACTIVE.\n"
            "[WARNING][FP8-MOE] selected config: %s\n"
            "[WARNING][FP8-MOE] For final benchmark numbers, pass --tactic=<cache>\n"
            "[WARNING][FP8-MOE] or explicit --tile_m/--tile_n so the tile choice is fixed.\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "\n",
            serialize_config(selected_config).c_str());
    }
    set_static_tile_config(selected_config);

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    if (prop.major != 9)
    {
        std::fprintf(stderr, "SM90/Hopper is required, got sm_%d%d\n", prop.major, prop.minor);
        return 1;
    }

    int const total_m = args.experts * args.m_per_expert;
    int const k_blocks = (args.k + 127) / 128;
    int const n_blocks = (args.n + 127) / 128;
    int64_t const padded_rows = padded_scale_rows(total_m, args.experts);

    std::vector<int64_t> h_offsets(args.experts + 1);
    for (int i = 0; i <= args.experts; ++i)
    {
        h_offsets[i] = static_cast<int64_t>(i) * args.m_per_expert;
    }

    auto* d_a = static_cast<__nv_fp8_e4m3*>(cuda_alloc_bytes(
        static_cast<size_t>(total_m) * args.k * sizeof(__nv_fp8_e4m3), "A_fp8"));
    auto* d_b = static_cast<__nv_fp8_e4m3*>(cuda_alloc_bytes(
        static_cast<size_t>(args.experts) * args.n * args.k * sizeof(__nv_fp8_e4m3), "B_fp8"));
    auto* d_d = static_cast<__nv_bfloat16*>(cuda_alloc_bytes(
        static_cast<size_t>(total_m) * args.n * sizeof(__nv_bfloat16), "D_bf16"));
    auto* d_scale_a = static_cast<float*>(cuda_alloc_bytes(
        static_cast<size_t>(padded_rows) * k_blocks * sizeof(float), "scale_a"));
    auto* d_scale_b = static_cast<float*>(cuda_alloc_bytes(
        static_cast<size_t>(args.experts) * n_blocks * k_blocks * sizeof(float), "scale_b"));
    auto* d_offsets = static_cast<int64_t*>(
        cuda_alloc_bytes(h_offsets.size() * sizeof(int64_t), "offsets"));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    fp8bs::CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16> runner;
    size_t const workspace_bytes = runner.getWorkspaceSize(
        static_cast<size_t>(args.m_per_expert), static_cast<size_t>(args.n), static_cast<size_t>(args.k),
        static_cast<size_t>(args.experts), static_cast<size_t>(args.experts));
    if (workspace_bytes != 0)
    {
        void* workspace = cuda_alloc_bytes(workspace_bytes, "workspace");
        runner.configureWorkspace(static_cast<char*>(workspace));
    }

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::printf("trtllm_cutlass_fp8_blockscale_moe_gemm: experts=%d m_per_expert=%d total_m=%d n=%d k=%d\n",
        args.experts, args.m_per_expert, total_m, args.n, args.k);
    std::printf("  scales: a_rows_padded=%lld a_blocks=%d b_blocks=(%d,%d) deep_gemm=%d workspace=%zu\n",
        static_cast<long long>(padded_rows), k_blocks, n_blocks, k_blocks, args.deep_gemm ? 1 : 0, workspace_bytes);
    std::printf("  static_config: %s\n", serialize_config(selected_config).c_str());

    auto launch_gemm = [&]() {
        runner.moeGemm(d_d, d_a, d_b, d_offsets, static_cast<size_t>(args.experts), static_cast<size_t>(args.n),
            static_cast<size_t>(args.k), stream, d_scale_a, d_scale_b);
    };

    if (args.sweep_configs)
    {
        float best_ms = std::numeric_limits<float>::infinity();
        StaticTileConfig best_cfg{};
        std::printf("sweep static FP8 configs: key=%s\n", tactic_key(args).c_str());
        int const sweep_warmup = std::max(args.warmup, 1);
        for (StaticTileConfig cfg : static_tile_search_space())
        {
            set_static_tile_config(cfg);
            float const ms = measure_kernel_ms(launch_gemm, sweep_warmup, args.iters);
            std::printf("  config %s median=%.4f ms\n", serialize_config(cfg).c_str(), ms);
            if (ms < best_ms)
            {
                best_ms = ms;
                best_cfg = cfg;
            }
        }
        selected_config = best_cfg;
        set_static_tile_config(selected_config);
        std::printf("best static config: %s median=%.4f ms\n", serialize_config(selected_config).c_str(), best_ms);
        if (!args.tactic_file.empty())
        {
            save_tactic(args.tactic_file, args, selected_config);
            std::printf("saved tactic: %s%s\n", tactic_key(args).c_str(), serialize_config(selected_config).c_str());
        }
    }

    BenchTimer timer;
    timer.warmup = args.warmup;
    timer.iters = args.iters;
    timer.active = true;
    timer.run(launch_gemm);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_scale_b));
    CHECK_CUDA(cudaFree(d_scale_a));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_a));
    return 0;
}
