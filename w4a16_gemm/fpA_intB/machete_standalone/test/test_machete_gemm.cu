#include "quantization/machete/cutlass55_backend.cuh"
#include "quantization/machete/machete_standalone_gemm.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct Args
{
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int group_size = 128;
    int warmup = 10;
    int iters = 100;
    bool list_schedules = false;
    bool list_cutlass55_configs = false;
    bool search_cutlass55_configs = false;
    bool no_checksum = false;
    bool offline_prepack = false;
    bool profile_gemm_only = false;
    std::string backend = "machete";
    std::string schedule;
    std::string cutlass55_config = "128x128x64_1x1x1";
    std::string cutlass55_tactic_file;
    std::string save_cutlass55_tactic_file;
    std::string act = "fp16";
    std::string quant = "gptq_u4b8";
    std::string offline_prepack_path;
    std::string save_prepacked_path;
};

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
    std::printf("Usage: %s [--m=N] [--n=N] [--k=N] [--group_size=N|-1] [--warmup=N] [--iters=N] "
                "[--backend=machete|cutlass55] [--act=fp16|bf16] [--quant=gptq_u4b8|awq_u4|cutlass_s4] "
                "[--schedule=<name>] [--cutlass55_config=<name>] [--search_cutlass55_configs] "
                "[--cutlass55_tactic=PATH] [--save_cutlass55_tactic=PATH] "
                "[--save_prepacked=PATH] [--offline_prepack[=PATH]] [--profile_gemm_only] [--no_checksum] "
                "[--list_schedules] [--list_cutlass55_configs]\n",
        name);
}

void parse_args(int argc, char** argv, Args& args)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (parse_int(argv[i], "--m=", args.m) || parse_int(argv[i], "--n=", args.n)
            || parse_int(argv[i], "--k=", args.k) || parse_int(argv[i], "--group_size=", args.group_size)
            || parse_int(argv[i], "--warmup=", args.warmup) || parse_int(argv[i], "--iters=", args.iters))
        {
            continue;
        }
        if (std::strncmp(argv[i], "--schedule=", 11) == 0)
        {
            args.schedule = argv[i] + 11;
            continue;
        }
        if (std::strncmp(argv[i], "--cutlass55_config=", 19) == 0)
        {
            args.cutlass55_config = argv[i] + 19;
            continue;
        }
        if (std::strncmp(argv[i], "--cutlass55-config=", 19) == 0)
        {
            args.cutlass55_config = argv[i] + 19;
            continue;
        }
        if (std::strncmp(argv[i], "--cutlass55_tactic=", 19) == 0)
        {
            args.cutlass55_tactic_file = argv[i] + 19;
            continue;
        }
        if (std::strncmp(argv[i], "--cutlass55-tactic=", 19) == 0)
        {
            args.cutlass55_tactic_file = argv[i] + 19;
            continue;
        }
        if (std::strncmp(argv[i], "--save_cutlass55_tactic=", 24) == 0)
        {
            args.save_cutlass55_tactic_file = argv[i] + 24;
            continue;
        }
        if (std::strncmp(argv[i], "--save-cutlass55-tactic=", 24) == 0)
        {
            args.save_cutlass55_tactic_file = argv[i] + 24;
            continue;
        }
        if (std::strncmp(argv[i], "--backend=", 10) == 0)
        {
            args.backend = argv[i] + 10;
            continue;
        }
        if (std::strncmp(argv[i], "--act=", 6) == 0)
        {
            args.act = argv[i] + 6;
            continue;
        }
        if (std::strncmp(argv[i], "--quant=", 8) == 0)
        {
            args.quant = argv[i] + 8;
            continue;
        }
        if (std::strcmp(argv[i], "--list_schedules") == 0)
        {
            args.list_schedules = true;
            continue;
        }
        if (std::strcmp(argv[i], "--list_cutlass55_configs") == 0
            || std::strcmp(argv[i], "--list-cutlass55-configs") == 0)
        {
            args.list_cutlass55_configs = true;
            continue;
        }
        if (std::strcmp(argv[i], "--search_cutlass55_configs") == 0
            || std::strcmp(argv[i], "--search-cutlass55-configs") == 0)
        {
            args.search_cutlass55_configs = true;
            args.backend = "cutlass55";
            continue;
        }
        if (std::strcmp(argv[i], "--profile_gemm_only") == 0 || std::strcmp(argv[i], "--profile-gemm-only") == 0)
        {
            args.profile_gemm_only = true;
            continue;
        }
        if (std::strcmp(argv[i], "--no_checksum") == 0 || std::strcmp(argv[i], "--no-checksum") == 0)
        {
            args.no_checksum = true;
            continue;
        }
        if (std::strncmp(argv[i], "--offline_prepack=", 18) == 0)
        {
            args.offline_prepack = true;
            args.offline_prepack_path = argv[i] + 18;
            continue;
        }
        if (std::strncmp(argv[i], "--offline-prepack=", 18) == 0)
        {
            args.offline_prepack = true;
            args.offline_prepack_path = argv[i] + 18;
            continue;
        }
        if (std::strcmp(argv[i], "--offline_prepack") == 0 || std::strcmp(argv[i], "--offline-prepack") == 0)
        {
            args.offline_prepack = true;
            continue;
        }
        if (std::strncmp(argv[i], "--save_prepacked=", 17) == 0)
        {
            args.save_prepacked_path = argv[i] + 17;
            continue;
        }
        if (std::strncmp(argv[i], "--save-prepacked=", 17) == 0)
        {
            args.save_prepacked_path = argv[i] + 17;
            continue;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(1);
    }
}

std::vector<uint32_t> read_prepacked_file(std::string const& path, size_t bytes)
{
    std::vector<uint32_t> data(bytes / sizeof(uint32_t));
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::fprintf(stderr, "Failed to open offline prepack file: %s\n", path.c_str());
        std::exit(1);
    }
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (in.gcount() != static_cast<std::streamsize>(bytes))
    {
        std::fprintf(stderr, "Offline prepack file has wrong size: %s expected=%zu read=%lld\n", path.c_str(), bytes,
            static_cast<long long>(in.gcount()));
        std::exit(1);
    }
    return data;
}

void write_prepacked_file(std::string const& path, uint32_t const* data, size_t bytes)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        std::fprintf(stderr, "Failed to create prepacked file: %s\n", path.c_str());
        std::exit(1);
    }
    out.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(bytes));
    if (!out)
    {
        std::fprintf(stderr, "Failed to write prepacked file: %s\n", path.c_str());
        std::exit(1);
    }
}

void check_cuda(cudaError_t status, char const* label)
{
    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error at %s: %s (%d)\n", label, cudaGetErrorString(status), int(status));
        std::abort();
    }
}

template <typename T>
T make_value(size_t i, float scale)
{
    return T(float(i % 113) * scale);
}

template <typename T>
float to_float(T const& value)
{
    return float(value);
}

std::vector<uint32_t> make_packed_u4b8_col_major(int k, int n)
{
    int constexpr pack = 8;
    int const packed_k = k / pack;
    std::vector<uint32_t> packed(static_cast<size_t>(packed_k) * n, 0);
    for (int col = 0; col < n; ++col)
    {
        for (int pk = 0; pk < packed_k; ++pk)
        {
            uint32_t word = 0;
            for (int i = 0; i < pack; ++i)
            {
                int const row = pk * pack + i;
                uint32_t const q = static_cast<uint32_t>((row + col) & 0xF);
                word |= q << (4 * i);
            }
            packed[static_cast<size_t>(col) * packed_k + pk] = word;
        }
    }
    return packed;
}

std::vector<uint32_t> make_synthetic_prepacked_words(size_t words)
{
    std::vector<uint32_t> packed(words);
    uint32_t state = 0x6d2b79f5u;
    for (size_t i = 0; i < words; ++i)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        packed[i] = state;
    }
    return packed;
}

machete_standalone::Cutlass55Config find_cutlass55_config(std::string const& name)
{
    for (auto const& config : machete_standalone::supported_cutlass55_configs())
    {
        if (name == config.name)
        {
            return config;
        }
    }
    std::fprintf(stderr, "Unknown CUTLASS55 config: %s\n", name.c_str());
    std::fprintf(stderr, "Use --list_cutlass55_configs to list supported configs.\n");
    std::exit(1);
}

std::string cutlass55_tactic_key(int m, int n, int k, int group_size, std::string const& act)
{
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%d,%d,%d,%d,%s|", m, n, k, group_size, act.c_str());
    return buf;
}

bool load_cutlass55_tactic(
    std::string const& path, int m, int n, int k, int group_size, std::string const& act, std::string& config_name)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        return false;
    }

    std::string const prefix = cutlass55_tactic_key(m, n, k, group_size, act);
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        if (line.compare(0, prefix.size(), prefix) != 0)
        {
            continue;
        }
        std::string const payload = line.substr(prefix.size());
        std::string const needle = "config=";
        auto pos = payload.find(needle);
        if (pos == std::string::npos)
        {
            return false;
        }
        pos += needle.size();
        auto end = payload.find(',', pos);
        config_name = payload.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        return !config_name.empty();
    }
    return false;
}

void save_cutlass55_tactic(std::string const& path, int m, int n, int k, int group_size, std::string const& act,
    std::string const& config_name, double avg_us)
{
    std::ofstream f(path, std::ios::app);
    if (!f.is_open())
    {
        std::fprintf(stderr, "Warning: cannot write CUTLASS55 tactic file %s\n", path.c_str());
        return;
    }
    f << cutlass55_tactic_key(m, n, k, group_size, act) << "config=" << config_name << ",avg_us=" << avg_us << "\n";
}

template <typename Element>
double run_case(Args const& args)
{
    using namespace machete_standalone;

    bool const is_cutlass55 = args.backend == "cutlass55";
    bool const is_machete = args.backend == "machete";
    if (!is_cutlass55 && !is_machete)
    {
        std::fprintf(stderr, "Unsupported --backend=%s\n", args.backend.c_str());
        std::exit(1);
    }

    int const group_size = args.group_size == -1 ? args.k : args.group_size;

    MacheteSchedule schedule{};
    if (!is_cutlass55 && args.schedule.empty())
    {
        schedule = select_schedule(args.m, args.n, args.k);
    }
    else if (!is_cutlass55)
    {
        auto parsed = schedule_from_name(args.schedule);
        if (!parsed)
        {
            std::fprintf(stderr, "Unknown schedule: %s\n", args.schedule.c_str());
            std::exit(1);
        }
        schedule = *parsed;
    }

    if (args.k % 64 != 0 || args.n % 128 != 0 || args.k % group_size != 0)
    {
        std::fprintf(stderr, "Invalid shape: K must be multiple of 64/group_size and N must be multiple of 128.\n");
        std::exit(1);
    }

    bool const is_gptq = args.quant == "gptq_u4b8";
    bool const is_awq = args.quant == "awq_u4";
    bool const is_cutlass_s4 = args.quant == "cutlass_s4";
    if (is_cutlass55)
    {
        if (!is_gptq && !is_cutlass_s4)
        {
            std::fprintf(stderr, "CUTLASS55 backend only supports scale-only signed int4. Use --quant=cutlass_s4 or omit --quant.\n");
            std::exit(1);
        }
    }
    else if (!is_gptq && !is_awq)
    {
        std::fprintf(stderr, "Unsupported --quant=%s\n", args.quant.c_str());
        std::exit(1);
    }

    Cutlass55Config cutlass55_config = default_cutlass55_config();
    if (is_cutlass55)
    {
        std::string config_name = args.cutlass55_config;
        if (!args.cutlass55_tactic_file.empty())
        {
            std::string cached_config;
            if (load_cutlass55_tactic(
                    args.cutlass55_tactic_file, args.m, args.n, args.k, group_size, args.act, cached_config))
            {
                std::printf("cutlass55 tactic cache HIT from %s: %s\n", args.cutlass55_tactic_file.c_str(),
                    cached_config.c_str());
                config_name = cached_config;
            }
            else
            {
                std::printf("cutlass55 tactic cache MISS from %s\n", args.cutlass55_tactic_file.c_str());
            }
        }
        cutlass55_config = find_cutlass55_config(config_name);
    }

    std::printf("m=%d n=%d k=%d group_size=%d backend=%s act=%s quant=%s\n", args.m, args.n, args.k, args.group_size,
        args.backend.c_str(), args.act.c_str(), is_cutlass55 ? "cutlass_s4" : args.quant.c_str());
    if (is_cutlass55)
    {
        std::printf("selected backend config: cutlass55 mode1 scale-only %dx%dx64 cluster=%dx%dx%d (%s)\n",
            cutlass55_config.tile_n, cutlass55_config.tile_m, cutlass55_config.cluster_n,
            cutlass55_config.cluster_m, cutlass55_config.cluster_k, cutlass55_config.name);
    }
    else
    {
        std::printf("selected schedule: %s\n", schedule.name);
    }
    bool const offline_prepack_from_file = args.offline_prepack && !args.offline_prepack_path.empty();
    bool const offline_prepack_synthetic = args.offline_prepack && args.offline_prepack_path.empty();
    if (offline_prepack_from_file)
    {
        std::printf("offline_prepack: loading %s\n", args.offline_prepack_path.c_str());
    }
    if (offline_prepack_synthetic)
    {
        std::printf("offline_prepack: synthetic prepacked data; no runtime prepack or file IO\n");
    }
    if (!args.save_prepacked_path.empty())
    {
        std::printf("save_prepacked: writing %s\n", args.save_prepacked_path.c_str());
    }
    if (args.profile_gemm_only)
    {
        std::printf("profile_gemm_only: cudaProfilerStart/Stop wraps timed GEMM loop\n");
    }

    std::vector<Element> h_a(static_cast<size_t>(args.m) * args.k);
    std::vector<Element> h_scales(static_cast<size_t>(args.k / group_size) * args.n);
    std::vector<Element> h_zeros(static_cast<size_t>(args.k / group_size) * args.n);
    std::vector<Element> h_out(static_cast<size_t>(args.m) * args.n);
    for (size_t i = 0; i < h_a.size(); ++i)
    {
        h_a[i] = make_value<Element>(i, 0.01f);
    }
    for (size_t i = 0; i < h_scales.size(); ++i)
    {
        h_scales[i] = make_value<Element>(i, 0.001f);
        h_zeros[i] = make_value<Element>(i, 0.01f);
    }
    int constexpr weight_pack = 8;
    size_t const b_words = static_cast<size_t>(args.k / weight_pack) * args.n;
    std::vector<uint32_t> h_b;
    if (!args.offline_prepack)
    {
        h_b = make_packed_u4b8_col_major(args.k, args.n);
    }

    Element* d_a = nullptr;
    Element* d_scales = nullptr;
    Element* d_zeros = nullptr;
    Element* d_out = nullptr;
    uint32_t* d_b_raw = nullptr;
    uint32_t* d_b_prepacked = nullptr;

    size_t const a_bytes = h_a.size() * sizeof(Element);
    size_t const scales_bytes = h_scales.size() * sizeof(Element);
    size_t const out_bytes = h_out.size() * sizeof(Element);
    size_t const b_bytes = b_words * sizeof(uint32_t);

    check_cuda(cudaMalloc(&d_a, a_bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_scales, scales_bytes), "cudaMalloc d_scales");
    check_cuda(cudaMalloc(&d_zeros, scales_bytes), "cudaMalloc d_zeros");
    check_cuda(cudaMalloc(&d_out, out_bytes), "cudaMalloc d_out");
    if (!args.offline_prepack)
    {
        check_cuda(cudaMalloc(&d_b_raw, b_bytes), "cudaMalloc d_b_raw");
    }
    check_cuda(cudaMalloc(&d_b_prepacked, b_bytes), "cudaMalloc d_b_prepacked");
    check_cuda(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    check_cuda(cudaMemcpy(d_scales, h_scales.data(), scales_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_scales");
    check_cuda(cudaMemcpy(d_zeros, h_zeros.data(), scales_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_zeros");
    if (offline_prepack_from_file)
    {
        std::vector<uint32_t> h_b_prepacked = read_prepacked_file(args.offline_prepack_path, b_bytes);
        check_cuda(cudaMemcpy(d_b_prepacked, h_b_prepacked.data(), b_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy d_b_prepacked");
    }
    else if (offline_prepack_synthetic)
    {
        std::vector<uint32_t> h_b_prepacked = make_synthetic_prepacked_words(b_words);
        check_cuda(cudaMemcpy(d_b_prepacked, h_b_prepacked.data(), b_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy synthetic d_b_prepacked");
    }
    else if (!args.offline_prepack)
    {
        check_cuda(cudaMemcpy(d_b_raw, h_b.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_b_raw");
    }

    cudaStream_t stream{};
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    if (!args.offline_prepack)
    {
        if (is_cutlass55)
        {
            if constexpr (std::is_same_v<Element, cutlass::half_t>)
            {
                cutlass55_reorder_B_fp16_s4(stream, reinterpret_cast<cutlass::int4b_t const*>(d_b_raw),
                    reinterpret_cast<cutlass::int4b_t*>(d_b_prepacked), args.k, args.n);
            }
            else
            {
                cutlass55_reorder_B_bf16_s4(stream, reinterpret_cast<cutlass::int4b_t const*>(d_b_raw),
                    reinterpret_cast<cutlass::int4b_t*>(d_b_prepacked), args.k, args.n);
            }
        }
        else
        {
            if constexpr (std::is_same_v<Element, cutlass::half_t>)
            {
                if (is_gptq)
                {
                    prepack_B_fp16_u4b8(stream, reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_raw),
                        reinterpret_cast<cutlass::vllm_uint4b8_t*>(d_b_prepacked), args.k, args.n);
                }
                else
                {
                    prepack_B_fp16_u4(stream, reinterpret_cast<cutlass::uint4b_t const*>(d_b_raw),
                        reinterpret_cast<cutlass::uint4b_t*>(d_b_prepacked), args.k, args.n);
                }
            }
            else
            {
                if (is_gptq)
                {
                    prepack_B_bf16_u4b8(stream, reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_raw),
                        reinterpret_cast<cutlass::vllm_uint4b8_t*>(d_b_prepacked), args.k, args.n);
                }
                else
                {
                    prepack_B_bf16_u4(stream, reinterpret_cast<cutlass::uint4b_t const*>(d_b_raw),
                        reinterpret_cast<cutlass::uint4b_t*>(d_b_prepacked), args.k, args.n);
                }
            }
        }
        check_cuda(cudaGetLastError(), is_cutlass55 ? "cutlass55 reorder launch" : "prepack launch");
        check_cuda(cudaStreamSynchronize(stream), is_cutlass55 ? "cutlass55 reorder sync" : "prepack sync");
        if (!args.save_prepacked_path.empty())
        {
            std::vector<uint32_t> h_b_prepacked(b_words);
            check_cuda(cudaMemcpy(h_b_prepacked.data(), d_b_prepacked, b_bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy save prepacked");
            write_prepacked_file(args.save_prepacked_path, h_b_prepacked.data(), b_bytes);
        }
    }
    else
    {
        check_cuda(cudaGetLastError(), "offline prepack");
    }

    size_t workspace_bytes = 0;
    if constexpr (std::is_same_v<Element, cutlass::half_t>)
    {
        if (is_cutlass55)
        {
            workspace_bytes = cutlass55_get_workspace_size_fp16_s4(d_a,
                reinterpret_cast<cutlass::int4b_t const*>(d_b_prepacked), d_scales, nullptr, d_out, args.m, args.n,
                args.k, args.group_size, cutlass55_config);
        }
        else if (is_gptq)
        {
            workspace_bytes = machete_get_workspace_size_fp16_u4b8(d_a,
                reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked), d_scales, d_out, args.m, args.n,
                args.k, args.group_size, schedule);
        }
        else
        {
            workspace_bytes = machete_get_workspace_size_fp16_u4(d_a,
                reinterpret_cast<cutlass::uint4b_t const*>(d_b_prepacked), d_scales, d_zeros, d_out, args.m, args.n,
                args.k, args.group_size, schedule);
        }
    }
    else
    {
        if (is_cutlass55)
        {
            workspace_bytes = cutlass55_get_workspace_size_bf16_s4(d_a,
                reinterpret_cast<cutlass::int4b_t const*>(d_b_prepacked), d_scales, nullptr, d_out, args.m, args.n,
                args.k, args.group_size, cutlass55_config);
        }
        else if (is_gptq)
        {
            workspace_bytes = machete_get_workspace_size_bf16_u4b8(d_a,
                reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked), d_scales, d_out, args.m, args.n,
                args.k, args.group_size, schedule);
        }
        else
        {
            workspace_bytes = machete_get_workspace_size_bf16_u4(d_a,
                reinterpret_cast<cutlass::uint4b_t const*>(d_b_prepacked), d_scales, d_zeros, d_out, args.m, args.n,
                args.k, args.group_size, schedule);
        }
    }
    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
    {
        check_cuda(cudaMalloc(&d_workspace, workspace_bytes), "cudaMalloc d_workspace");
    }
    std::printf("workspace bytes: %zu\n", workspace_bytes);

    Cutlass55Plan* cutlass55_plan = nullptr;
    if (is_cutlass55)
    {
        if constexpr (std::is_same_v<Element, cutlass::half_t>)
        {
            cutlass55_plan = cutlass55_create_plan_fp16_s4(stream, d_a,
                reinterpret_cast<cutlass::int4b_t const*>(d_b_prepacked), d_scales, nullptr, d_out, args.m, args.n,
                args.k, args.group_size, cutlass55_config, d_workspace, workspace_bytes);
        }
        else
        {
            cutlass55_plan = cutlass55_create_plan_bf16_s4(stream, d_a,
                reinterpret_cast<cutlass::int4b_t const*>(d_b_prepacked), d_scales, nullptr, d_out, args.m, args.n,
                args.k, args.group_size, cutlass55_config, d_workspace, workspace_bytes);
        }
    }

    auto launch = [&]() {
        if constexpr (std::is_same_v<Element, cutlass::half_t>)
        {
            if (is_cutlass55)
            {
                cutlass55_run_plan(cutlass55_plan, stream);
            }
            else if (is_gptq)
            {
                machete_mm_fp16_u4b8(stream, d_a, reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked),
                    d_scales, d_out, args.m, args.n, args.k, args.group_size, schedule, d_workspace, workspace_bytes);
            }
            else
            {
                machete_mm_fp16_u4(stream, d_a, reinterpret_cast<cutlass::uint4b_t const*>(d_b_prepacked), d_scales,
                    d_zeros, d_out, args.m, args.n, args.k, args.group_size, schedule, d_workspace, workspace_bytes);
            }
        }
        else
        {
            if (is_cutlass55)
            {
                cutlass55_run_plan(cutlass55_plan, stream);
            }
            else if (is_gptq)
            {
                machete_mm_bf16_u4b8(stream, d_a, reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked),
                    d_scales, d_out, args.m, args.n, args.k, args.group_size, schedule, d_workspace, workspace_bytes);
            }
            else
            {
                machete_mm_bf16_u4(stream, d_a, reinterpret_cast<cutlass::uint4b_t const*>(d_b_prepacked), d_scales,
                    d_zeros, d_out, args.m, args.n, args.k, args.group_size, schedule, d_workspace, workspace_bytes);
            }
        }
    };

    for (int i = 0; i < args.warmup; ++i)
    {
        launch();
    }
    if (args.profile_gemm_only)
    {
        check_cuda(cudaStreamSynchronize(stream), "profile pre-sync");
        check_cuda(cudaProfilerStart(), "cudaProfilerStart");
    }
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (int i = 0; i < args.iters; ++i)
    {
        launch();
    }
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    if (args.profile_gemm_only)
    {
        check_cuda(cudaProfilerStop(), "cudaProfilerStop");
    }
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    if (!args.no_checksum)
    {
        check_cuda(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_out");
    }

    double checksum = 0.0;
    if (!args.no_checksum)
    {
        for (size_t i = 0; i < std::min<size_t>(h_out.size(), 1024); ++i)
        {
            checksum += to_float(h_out[i]);
        }
    }
    double const flops = 2.0 * double(args.m) * args.n * args.k;
    double const avg_us = double(ms) * 1000.0 / args.iters;
    if (!args.no_checksum)
    {
        std::printf("checksum=%f\n", checksum);
    }
    std::printf("Avg kernel time: %.3f us (%.1f TFLOPS, %d iters, %d warmup)\n", avg_us, flops / (avg_us * 1e-6) / 1e12,
        args.iters, args.warmup);
    if (is_cutlass55 && !args.save_cutlass55_tactic_file.empty())
    {
        save_cutlass55_tactic(args.save_cutlass55_tactic_file, args.m, args.n, args.k, group_size, args.act,
            cutlass55_config.name, avg_us);
    }

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    cutlass55_destroy_plan(cutlass55_plan);
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    if (d_workspace)
    {
        check_cuda(cudaFree(d_workspace), "cudaFree workspace");
    }
    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_scales), "cudaFree d_scales");
    check_cuda(cudaFree(d_zeros), "cudaFree d_zeros");
    check_cuda(cudaFree(d_out), "cudaFree d_out");
    if (d_b_raw)
    {
        check_cuda(cudaFree(d_b_raw), "cudaFree d_b_raw");
    }
    check_cuda(cudaFree(d_b_prepacked), "cudaFree d_b_prepacked");
    return avg_us;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    parse_args(argc, argv, args);
    if (args.list_cutlass55_configs)
    {
        auto configs = machete_standalone::supported_cutlass55_configs();
        std::printf("Supported CUTLASS55 configs (%zu):\n", configs.size());
        for (size_t i = 0; i < configs.size(); ++i)
        {
            std::printf("  %zu: %s\n", i, configs[i].name);
        }
        return 0;
    }
    if (args.list_schedules)
    {
        auto schedules = machete_standalone::supported_schedules();
        std::printf("Supported schedules (%zu):\n", schedules.size());
        for (size_t i = 0; i < schedules.size(); ++i)
        {
            std::printf("  %zu: %s\n", i, schedules[i].name);
        }
        return 0;
    }

    try
    {
        auto run_one = [](Args const& run_args) -> double {
            if (run_args.act == "fp16")
            {
                return run_case<cutlass::half_t>(run_args);
            }
            else if (run_args.act == "bf16")
            {
                return run_case<cutlass::bfloat16_t>(run_args);
            }
            else
            {
                std::fprintf(stderr, "Unsupported --act=%s\n", run_args.act.c_str());
                std::exit(1);
            }
        };

        if (args.search_cutlass55_configs)
        {
            args.backend = "cutlass55";
            auto configs = machete_standalone::supported_cutlass55_configs();
            double best_us = std::numeric_limits<double>::infinity();
            std::string best_config;
            for (auto const& config : configs)
            {
                Args run_args = args;
                run_args.cutlass55_config = config.name;
                run_args.cutlass55_tactic_file.clear();
                run_args.save_cutlass55_tactic_file.clear();
                std::printf("\n===== CUTLASS55 config: %s =====\n", config.name);
                double const avg_us = run_one(run_args);
                if (avg_us < best_us)
                {
                    best_us = avg_us;
                    best_config = config.name;
                }
            }
            std::printf("\nBest CUTLASS55 config: %s avg_us=%.3f\n", best_config.c_str(), best_us);
            int const group_size = args.group_size == -1 ? args.k : args.group_size;
            if (!args.save_cutlass55_tactic_file.empty())
            {
                save_cutlass55_tactic(
                    args.save_cutlass55_tactic_file, args.m, args.n, args.k, group_size, args.act, best_config, best_us);
            }
            return 0;
        }

        if (args.act == "fp16")
        {
            run_case<cutlass::half_t>(args);
        }
        else if (args.act == "bf16")
        {
            run_case<cutlass::bfloat16_t>(args);
        }
        else
        {
            std::fprintf(stderr, "Unsupported --act=%s\n", args.act.c_str());
            return 1;
        }
    }
    catch (std::exception const& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
