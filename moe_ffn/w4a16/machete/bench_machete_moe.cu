#include "quantization/machete/machete_standalone_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    std::string schedule;
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
                "          [--schedule=<machete schedule>] [--warmup=N] [--iters=N]\n"
                "          [--profile_gemm_only] [--no_checksum]\n",
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
            || parse_int(argv[i], "--iters=", args.iters))
        {
            continue;
        }
        if (std::strncmp(argv[i], "--schedule=", 11) == 0)
        {
            args.schedule = argv[i] + 11;
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
        || args.warmup < 0)
    {
        std::fprintf(stderr, "experts, m_per_expert, n, k, iters must be positive and warmup must be non-negative.\n");
        std::exit(1);
    }
    if (args.k % 64 != 0 || args.n % 128 != 0 || args.group_size <= 0 || args.k % args.group_size != 0)
    {
        std::fprintf(stderr, "Invalid shape: K must be multiple of 64/group_size and N must be multiple of 128.\n");
        std::exit(1);
    }
    return args;
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

} // namespace

int main(int argc, char** argv)
{
    using namespace machete_standalone;

    Args const args = parse_args(argc, argv);

    MacheteSchedule schedule{};
    if (args.schedule.empty())
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
    std::vector<uint32_t> h_b_prepacked = make_synthetic_prepacked_words(b_words);

    cutlass::half_t* d_a = nullptr;
    cutlass::half_t* d_scales = nullptr;
    cutlass::half_t* d_out = nullptr;
    uint32_t* d_b_prepacked = nullptr;

    size_t const a_bytes = h_a.size() * sizeof(cutlass::half_t);
    size_t const scales_bytes = h_scales.size() * sizeof(cutlass::half_t);
    size_t const out_bytes = out_elems * sizeof(cutlass::half_t);
    size_t const b_bytes = h_b_prepacked.size() * sizeof(uint32_t);

    check_cuda(cudaMalloc(&d_a, a_bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_scales, scales_bytes), "cudaMalloc d_scales");
    check_cuda(cudaMalloc(&d_out, out_bytes), "cudaMalloc d_out");
    check_cuda(cudaMalloc(&d_b_prepacked, b_bytes), "cudaMalloc d_b_prepacked");
    check_cuda(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    check_cuda(cudaMemcpy(d_scales, h_scales.data(), scales_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_scales");
    check_cuda(cudaMemcpy(d_b_prepacked, h_b_prepacked.data(), b_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy d_b_prepacked");

    cudaStream_t stream{};
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    size_t const workspace_bytes = machete_get_workspace_size_fp16_u4b8(d_a,
        reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked), d_scales, d_out, args.m_per_expert, args.n,
        args.k, args.group_size, schedule);
    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
    {
        check_cuda(cudaMalloc(&d_workspace, workspace_bytes), "cudaMalloc d_workspace");
    }

    auto launch_all_experts = [&]() {
        for (int expert = 0; expert < args.experts; ++expert)
        {
            auto const a_offset = static_cast<size_t>(expert) * args.m_per_expert * args.k;
            auto const b_offset = static_cast<size_t>(expert) * b_words_per_expert;
            auto const scale_offset = static_cast<size_t>(expert) * (args.k / args.group_size) * args.n;
            auto const out_offset = static_cast<size_t>(expert) * args.m_per_expert * args.n;
            machete_mm_fp16_u4b8(stream, d_a + a_offset,
                reinterpret_cast<cutlass::vllm_uint4b8_t const*>(d_b_prepacked + b_offset), d_scales + scale_offset,
                d_out + out_offset, args.m_per_expert, args.n, args.k, args.group_size, schedule, d_workspace,
                workspace_bytes);
        }
    };

    std::printf("Machete MoE W4A16 prefill bench: experts=%d m_per_expert=%d n=%d k=%d group_size=%d\n",
        args.experts, args.m_per_expert, args.n, args.k, args.group_size);
    std::printf("selected schedule: %s\n", schedule.name);
    std::printf("offline_prepack: synthetic prepacked data; no runtime prepack or file IO\n");
    std::printf("workspace bytes per expert: %zu\n", workspace_bytes);

    for (int i = 0; i < args.warmup; ++i)
    {
        launch_all_experts();
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
        launch_all_experts();
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
    std::printf("Avg MoE GEMM time: %.3f us (%.1f TFLOPS, %d experts, %d kernels/iter, %d iters, %d warmup)\n",
        avg_us, flops / (avg_us * 1e-6) / 1e12, args.experts, args.experts, args.iters, args.warmup);

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
    check_cuda(cudaFree(d_b_prepacked), "cudaFree d_b_prepacked");
    return 0;
}
