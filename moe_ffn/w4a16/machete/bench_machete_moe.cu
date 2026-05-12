#include "quantization/machete/machete_standalone_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace
{

struct Args
{
    int experts = 8;
    int m_per_expert = 3823;
    int n = 2048;
    int k = 3072;
    int group_size = 128;
    int warmup = 20;
    int iters = 100;
    bool no_checksum = true;
    bool profile_gemm_only = false;
    bool verify = false;
    bool verify_reference = false;
    bool sequential = false;
    int verify_samples = 4096;
    std::string schedule;
    std::string tactic_file;
};

void check_cuda(cudaError_t status, char const* label)
{
    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error at %s: %s (%d)\n", label, cudaGetErrorString(status), int(status));
        std::abort();
    }
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
    std::printf("Usage: %s [--experts=N] [--m_per_expert=N] [--n=N] [--k=N] [--group_size=N]\n"
                "          [--schedule=<machete schedule>] [--tactic=PATH] [--warmup=N] [--iters=N]\n"
                "          [--profile_gemm_only] [--verify] [--verify_reference]\n"
                "          [--verify_samples=N] [--sequential] [--no_checksum]\n",
        name);
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (parse_int(argv[i], "--experts=", args.experts) || parse_int(argv[i], "--m_per_expert=", args.m_per_expert)
            || parse_int(argv[i], "--n=", args.n) || parse_int(argv[i], "--k=", args.k)
            || parse_int(argv[i], "--group_size=", args.group_size) || parse_int(argv[i], "--warmup=", args.warmup)
            || parse_int(argv[i], "--iters=", args.iters) || parse_int(argv[i], "--verify_samples=", args.verify_samples))
        {
            continue;
        }
        if (std::strncmp(argv[i], "--schedule=", 11) == 0)
        {
            args.schedule = argv[i] + 11;
            continue;
        }
        if (std::strncmp(argv[i], "--tactic=", 9) == 0)
        {
            args.tactic_file = argv[i] + 9;
            continue;
        }
        if (std::strncmp(argv[i], "--machete_tactic=", 17) == 0
            || std::strncmp(argv[i], "--machete-tactic=", 17) == 0)
        {
            args.tactic_file = argv[i] + 17;
            continue;
        }
        if (std::strcmp(argv[i], "--profile_gemm_only") == 0 || std::strcmp(argv[i], "--profile-gemm-only") == 0)
        {
            args.profile_gemm_only = true;
            continue;
        }
        if (std::strcmp(argv[i], "--verify") == 0)
        {
            args.verify = true;
            continue;
        }
        if (std::strcmp(argv[i], "--verify_reference") == 0 || std::strcmp(argv[i], "--verify-reference") == 0)
        {
            args.verify_reference = true;
            continue;
        }
        if (std::strcmp(argv[i], "--sequential") == 0)
        {
            args.sequential = true;
            continue;
        }
        if (std::strcmp(argv[i], "--no_checksum") == 0 || std::strcmp(argv[i], "--no-checksum") == 0)
        {
            args.no_checksum = true;
            continue;
        }
        if (std::strcmp(argv[i], "--checksum") == 0)
        {
            args.no_checksum = false;
            continue;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(1);
    }
    if (args.experts <= 0 || args.m_per_expert <= 0 || args.n <= 0 || args.k <= 0 || args.iters <= 0
        || args.warmup < 0 || args.verify_samples <= 0)
    {
        std::fprintf(stderr,
            "experts, m_per_expert, n, k, iters, verify_samples must be positive and warmup must be non-negative.\n");
        std::exit(1);
    }
    if (args.k % 64 != 0 || args.n % 128 != 0 || args.group_size <= 0 || args.k % args.group_size != 0)
    {
        std::fprintf(stderr, "Invalid shape: K must be multiple of 64/group_size and N must be multiple of 128.\n");
        std::exit(1);
    }
    return args;
}

std::string tactic_key(int experts, int m_per_expert, int n, int k, int group_size)
{
    char buf[192];
    std::snprintf(buf, sizeof(buf), "fp16,%d,%d,%d,%d,%d|", experts, m_per_expert, n, k, group_size);
    return buf;
}

bool load_tactic_schedule(
    std::string const& path, int experts, int m_per_expert, int n, int k, int group_size, std::string& schedule_name)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        return false;
    }

    std::string const prefix = tactic_key(experts, m_per_expert, n, k, group_size);
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
        std::string const needle = "schedule=";
        auto pos = payload.find(needle);
        if (pos == std::string::npos)
        {
            return false;
        }
        pos += needle.size();
        auto const end = payload.find(',', pos);
        schedule_name = payload.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        return !schedule_name.empty();
    }
    return false;
}

cutlass::half_t make_half_value(size_t i, float scale)
{
    return cutlass::half_t(float((i * 13 + 7) & 1023) * (scale / 1024.0f));
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

std::vector<uint32_t> make_packed_u4b8_col_major(int experts, int k, int n)
{
    int constexpr pack = 8;
    int const packed_k = k / pack;
    std::vector<uint32_t> packed(static_cast<size_t>(experts) * packed_k * n, 0);
    for (int expert = 0; expert < experts; ++expert)
    {
        for (int col = 0; col < n; ++col)
        {
            for (int pk = 0; pk < packed_k; ++pk)
            {
                uint32_t word = 0;
                for (int i = 0; i < pack; ++i)
                {
                    int const row = pk * pack + i;
                    uint32_t const q = static_cast<uint32_t>((row + 3 * col + 5 * expert) & 0xF);
                    word |= q << (4 * i);
                }
                packed[(static_cast<size_t>(expert) * n + col) * packed_k + pk] = word;
            }
        }
    }
    return packed;
}

float dequant_u4b8(uint32_t word, int k_mod_pack)
{
    int const q = static_cast<int>((word >> (4 * k_mod_pack)) & 0xF);
    return static_cast<float>(q - 8);
}

template <typename LaunchFn>
void verify_zero_input(cutlass::half_t* d_a, cutlass::half_t const* h_a, size_t a_elems, cutlass::half_t* d_out,
    size_t out_elems, cudaStream_t stream, LaunchFn const& launch_all_experts)
{
    size_t const a_bytes = a_elems * sizeof(cutlass::half_t);
    size_t const out_bytes = out_elems * sizeof(cutlass::half_t);

    check_cuda(cudaMemset(d_a, 0, a_bytes), "verify cudaMemset d_a");
    check_cuda(cudaMemset(d_out, 0x7f, out_bytes), "verify cudaMemset d_out sentinel");
    launch_all_experts();
    check_cuda(cudaGetLastError(), "verify launch");
    check_cuda(cudaStreamSynchronize(stream), "verify sync");

    std::vector<cutlass::half_t> h_out(out_elems);
    check_cuda(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), "verify cudaMemcpy d_out");

    double max_abs = 0.0;
    size_t bad_count = 0;
    size_t first_bad = 0;
    double first_bad_value = 0.0;
    for (size_t i = 0; i < h_out.size(); ++i)
    {
        double const value = std::fabs(static_cast<double>(float(h_out[i])));
        max_abs = std::max(max_abs, value);
        if (value > 1e-3)
        {
            if (bad_count == 0)
            {
                first_bad = i;
                first_bad_value = static_cast<double>(float(h_out[i]));
            }
            ++bad_count;
        }
    }

    check_cuda(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice), "verify restore d_a");
    check_cuda(cudaMemset(d_out, 0, out_bytes), "verify clear d_out");

    if (bad_count != 0)
    {
        std::fprintf(stderr,
            "verify_zero_input FAILED: max_abs=%.8e bad_count=%zu first_bad_index=%zu first_bad_value=%.8e\n",
            max_abs, bad_count, first_bad, first_bad_value);
        std::exit(2);
    }
    std::printf("verify_zero_input PASS: checked=%zu max_abs=%.8e\n", h_out.size(), max_abs);
}

template <typename LaunchFn>
void verify_cpu_reference(Args const& args, std::vector<cutlass::half_t> const& h_a,
    std::vector<uint32_t> const& h_b_raw, std::vector<cutlass::half_t> const& h_scales, cutlass::half_t* d_out,
    size_t out_elems, cudaStream_t stream, LaunchFn const& launch_all_experts)
{
    int constexpr pack = 8;
    int const packed_k = args.k / pack;
    size_t const out_bytes = out_elems * sizeof(cutlass::half_t);

    check_cuda(cudaMemset(d_out, 0, out_bytes), "verify_reference clear d_out");
    launch_all_experts();
    check_cuda(cudaGetLastError(), "verify_reference launch");
    check_cuda(cudaStreamSynchronize(stream), "verify_reference sync");

    std::vector<cutlass::half_t> h_out(out_elems);
    check_cuda(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), "verify_reference copy d_out");

    size_t const sample_count = std::min<size_t>(out_elems, static_cast<size_t>(args.verify_samples));
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t bad_count = 0;
    size_t first_bad = 0;
    double first_gpu = 0.0;
    double first_ref = 0.0;

    for (size_t sample = 0; sample < sample_count; ++sample)
    {
        size_t const idx = (sample_count == out_elems) ? sample : (sample * out_elems / sample_count);
        int const col = static_cast<int>(idx % args.n);
        size_t const tmp = idx / args.n;
        int const row = static_cast<int>(tmp % args.m_per_expert);
        int const expert = static_cast<int>(tmp / args.m_per_expert);

        float acc = 0.0f;
        size_t const a_base = (static_cast<size_t>(expert) * args.m_per_expert + row) * args.k;
        size_t const b_base = static_cast<size_t>(expert) * args.n * packed_k + static_cast<size_t>(col) * packed_k;
        size_t const scale_base = static_cast<size_t>(expert) * (args.k / args.group_size) * args.n;
        for (int kk = 0; kk < args.k; ++kk)
        {
            uint32_t const word = h_b_raw[b_base + kk / pack];
            float const w = dequant_u4b8(word, kk % pack);
            float const scale = float(h_scales[scale_base + static_cast<size_t>(kk / args.group_size) * args.n + col]);
            acc += float(h_a[a_base + kk]) * w * scale;
        }

        float const ref = float(cutlass::half_t(acc));
        float const gpu = float(h_out[idx]);
        double const abs_err = std::fabs(static_cast<double>(gpu) - static_cast<double>(ref));
        double const rel_err = abs_err / std::max(1e-6, std::fabs(static_cast<double>(ref)));
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
        if (abs_err > 5e-2 && rel_err > 5e-2)
        {
            if (bad_count == 0)
            {
                first_bad = idx;
                first_gpu = gpu;
                first_ref = ref;
            }
            ++bad_count;
        }
    }

    if (bad_count != 0)
    {
        std::fprintf(stderr,
            "verify_cpu_reference FAILED: samples=%zu max_abs=%.8e max_rel=%.8e bad_count=%zu "
            "first_bad_index=%zu gpu=%.8e ref=%.8e\n",
            sample_count, max_abs, max_rel, bad_count, first_bad, first_gpu, first_ref);
        std::exit(3);
    }
    std::printf("verify_cpu_reference PASS: samples=%zu max_abs=%.8e max_rel=%.8e\n", sample_count, max_abs, max_rel);
}

} // namespace

int main(int argc, char** argv)
{
    using namespace machete_standalone;

    Args const args = parse_args(argc, argv);

    MacheteSchedule schedule{};
    if (!args.tactic_file.empty())
    {
        std::string schedule_name;
        if (!load_tactic_schedule(
                args.tactic_file, args.experts, args.m_per_expert, args.n, args.k, args.group_size, schedule_name))
        {
            std::fprintf(stderr, "machete tactic cache MISS from %s for key %s\n", args.tactic_file.c_str(),
                tactic_key(args.experts, args.m_per_expert, args.n, args.k, args.group_size).c_str());
            return 1;
        }
        auto parsed = schedule_from_name(schedule_name);
        if (!parsed)
        {
            std::fprintf(stderr, "Unknown schedule in tactic cache: %s\n", schedule_name.c_str());
            return 1;
        }
        schedule = *parsed;
        std::printf("machete tactic cache HIT from %s: %s\n", args.tactic_file.c_str(), schedule.name);
    }
    else if (args.schedule.empty())
    {
        schedule = select_schedule(args.m_per_expert, args.n, args.k);
    }
    else
    {
        auto parsed = schedule_from_name(args.schedule);
        if (!parsed)
        {
            std::fprintf(stderr, "Unknown schedule: %s\n", args.schedule.c_str());
            return 1;
        }
        schedule = *parsed;
    }

    int constexpr weight_pack = 8;
    size_t const a_elems = static_cast<size_t>(args.experts) * args.m_per_expert * args.k;
    size_t const scales_elems
        = static_cast<size_t>(args.experts) * (args.k / args.group_size) * static_cast<size_t>(args.n);
    size_t const out_elems = static_cast<size_t>(args.experts) * args.m_per_expert * args.n;
    size_t const b_words_per_expert = static_cast<size_t>(args.k / weight_pack) * args.n;
    size_t const b_words = static_cast<size_t>(args.experts) * b_words_per_expert;

    std::vector<cutlass::half_t> h_a(a_elems);
    std::vector<cutlass::half_t> h_scales(scales_elems);
    for (size_t i = 0; i < h_a.size(); ++i)
    {
        h_a[i] = make_half_value(i, 0.01f);
    }
    for (size_t i = 0; i < h_scales.size(); ++i)
    {
        h_scales[i] = make_half_value(i, 0.001f);
    }
    std::vector<uint32_t> h_b_raw;
    std::vector<uint32_t> h_b_prepacked;
    if (args.verify_reference)
    {
        h_b_raw = make_packed_u4b8_col_major(args.experts, args.k, args.n);
    }
    else
    {
        h_b_prepacked = make_synthetic_prepacked_words(b_words);
    }

    cutlass::half_t* d_a = nullptr;
    cutlass::half_t* d_scales = nullptr;
    cutlass::half_t* d_out = nullptr;
    uint32_t* d_b_raw = nullptr;
    uint32_t* d_b_prepacked = nullptr;

    size_t const a_bytes = h_a.size() * sizeof(cutlass::half_t);
    size_t const scales_bytes = h_scales.size() * sizeof(cutlass::half_t);
    size_t const out_bytes = out_elems * sizeof(cutlass::half_t);
    size_t const b_bytes = b_words * sizeof(uint32_t);

    check_cuda(cudaMalloc(&d_a, a_bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_scales, scales_bytes), "cudaMalloc d_scales");
    check_cuda(cudaMalloc(&d_out, out_bytes), "cudaMalloc d_out");
    if (args.verify_reference)
    {
        check_cuda(cudaMalloc(&d_b_raw, b_bytes), "cudaMalloc d_b_raw");
    }
    check_cuda(cudaMalloc(&d_b_prepacked, b_bytes), "cudaMalloc d_b_prepacked");
    check_cuda(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    check_cuda(cudaMemcpy(d_scales, h_scales.data(), scales_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_scales");

    cudaStream_t stream{};
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    if (args.verify_reference)
    {
        check_cuda(cudaMemcpy(d_b_raw, h_b_raw.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_b_raw");
        for (int expert = 0; expert < args.experts; ++expert)
        {
            auto const b_offset = static_cast<size_t>(expert) * b_words_per_expert;
            prepack_B_fp16_u4b8(stream, reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_raw + b_offset),
                reinterpret_cast<cutlass::vllm_uint4b8_t*>(d_b_prepacked + b_offset), args.k, args.n);
        }
        check_cuda(cudaGetLastError(), "verify_reference prepack launch");
        check_cuda(cudaStreamSynchronize(stream), "verify_reference prepack sync");
    }
    else
    {
        check_cuda(cudaMemcpy(d_b_prepacked, h_b_prepacked.data(), b_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy d_b_prepacked");
    }

    size_t const workspace_bytes = args.sequential
        ? machete_get_workspace_size_fp16_u4b8(d_a, reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked),
            d_scales, d_out, args.m_per_expert, args.n, args.k, args.group_size, schedule)
        : machete_grouped_get_workspace_size_fp16_u4b8(d_a,
            reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked), d_scales, d_out, args.experts,
            args.m_per_expert, args.n, args.k, args.group_size, schedule);
    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
    {
        check_cuda(cudaMalloc(&d_workspace, workspace_bytes), "cudaMalloc d_workspace");
    }

    auto launch_moe_gemm = [&]() {
        if (args.sequential)
        {
            for (int expert = 0; expert < args.experts; ++expert)
            {
                auto const a_offset = static_cast<size_t>(expert) * args.m_per_expert * args.k;
                auto const b_offset = static_cast<size_t>(expert) * b_words_per_expert;
                auto const scale_offset = static_cast<size_t>(expert) * (args.k / args.group_size) * args.n;
                auto const out_offset = static_cast<size_t>(expert) * args.m_per_expert * args.n;
                machete_mm_fp16_u4b8(stream, d_a + a_offset,
                    reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked + b_offset),
                    d_scales + scale_offset, d_out + out_offset, args.m_per_expert, args.n, args.k, args.group_size,
                    schedule, d_workspace, workspace_bytes);
            }
            return;
        }
        machete_grouped_mm_fp16_u4b8(stream, d_a, reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked),
            d_scales, d_out, args.experts, args.m_per_expert, args.n, args.k, args.group_size, schedule, d_workspace,
            workspace_bytes);
    };

    std::printf("Machete MoE W4A16 prefill bench: experts=%d m_per_expert=%d n=%d k=%d group_size=%d\n",
        args.experts, args.m_per_expert, args.n, args.k, args.group_size);
    std::printf("selected schedule: %s\n", schedule.name);
    std::printf("dispatch: %s\n", args.sequential ? "sequential_per_expert" : "grouped_batched");
    if (args.verify_reference)
    {
        std::printf("offline_prepack: raw GPTQ u4b8 data prepacked before timing for CPU reference\n");
    }
    else
    {
        std::printf("offline_prepack: synthetic prepacked data; no runtime prepack or file IO\n");
    }
    std::printf("workspace bytes: %zu\n", workspace_bytes);

    if (args.verify)
    {
        verify_zero_input(d_a, h_a.data(), h_a.size(), d_out, out_elems, stream, launch_moe_gemm);
    }
    if (args.verify_reference)
    {
        verify_cpu_reference(args, h_a, h_b_raw, h_scales, d_out, out_elems, stream, launch_moe_gemm);
    }

    for (int i = 0; i < args.warmup; ++i)
    {
        launch_moe_gemm();
    }
    check_cuda(cudaStreamSynchronize(stream), "warmup sync");
    if (args.profile_gemm_only)
    {
        check_cuda(cudaProfilerStart(), "cudaProfilerStart");
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (int i = 0; i < args.iters; ++i)
    {
        launch_moe_gemm();
    }
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    if (args.profile_gemm_only)
    {
        check_cuda(cudaProfilerStop(), "cudaProfilerStop");
    }

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    double checksum = 0.0;
    if (!args.no_checksum)
    {
        std::vector<cutlass::half_t> h_out(std::min<size_t>(out_elems, 1024));
        check_cuda(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy d_out checksum");
        for (auto const& value : h_out)
        {
            checksum += float(value);
        }
        std::printf("checksum=%f\n", checksum);
    }

    double const avg_us = double(ms) * 1000.0 / args.iters;
    double const flops = 2.0 * args.experts * double(args.m_per_expert) * args.n * args.k;
    int const kernels_per_iter = args.sequential ? args.experts : 1;
    std::printf("Avg MoE GEMM time: %.3f us (%.1f TFLOPS, %d experts, %d kernels/iter, %d iters, %d warmup)\n",
        avg_us, flops / (avg_us * 1e-6) / 1e12, args.experts, kernels_per_iter, args.iters, args.warmup);

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    if (d_workspace)
    {
        check_cuda(cudaFree(d_workspace), "cudaFree d_workspace");
    }
    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_scales), "cudaFree d_scales");
    check_cuda(cudaFree(d_out), "cudaFree d_out");
    if (d_b_raw)
    {
        check_cuda(cudaFree(d_b_raw), "cudaFree d_b_raw");
    }
    check_cuda(cudaFree(d_b_prepacked), "cudaFree d_b_prepacked");
    return 0;
}
