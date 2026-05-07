// Standalone sampling-stage benchmarks for Qwen3.5-122B-A10B decode.
//
// Stages:
//   lm_head    : fp16 GEMM, (1, hidden) x (hidden, vocab) -> (1, vocab)
//   topk_mask  : FlashInfer radix top-k logits mask
//   softmax    : FlashInfer online softmax
//   top_p      : FlashInfer top-p sampling from probabilities
//
// This file keeps only the single-row, profile-oriented subset needed by the
// Qwen3.5 decode microbenchmarks. Sampling kernels are FlashInfer kernels.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

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

#define CHECK_CUBLAS(e)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t _e = (e);                                                                                       \
        if (_e != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            std::fprintf(stderr, "cuBLAS %s:%d: status=%d\n", __FILE__, __LINE__, static_cast<int>(_e));             \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace
{

constexpr int kMaxTopK = 128;
constexpr size_t kFlashInferWorkspaceBytes = 1024 * 1024;

struct Options
{
    std::string op = "all";
    int hidden = 3072;
    int vocab = 248320;
    int top_k = 50;
    float top_p = 0.9f;
};

bool starts_with(char const* s, char const* prefix)
{
    return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

int parse_int(char const* s)
{
    return std::atoi(s);
}

float parse_float(char const* s)
{
    return std::strtof(s, nullptr);
}

Options parse_args(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--op") == 0 || std::strcmp(argv[i], "--kernel") == 0)
        {
            opt.op = argv[++i];
        }
        else if (starts_with(argv[i], "--op="))
        {
            opt.op = argv[i] + 5;
        }
        else if (starts_with(argv[i], "--kernel="))
        {
            opt.op = argv[i] + 9;
        }
        else if (std::strcmp(argv[i], "--hidden") == 0)
        {
            opt.hidden = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--hidden="))
        {
            opt.hidden = parse_int(argv[i] + 9);
        }
        else if (std::strcmp(argv[i], "--vocab") == 0)
        {
            opt.vocab = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--vocab="))
        {
            opt.vocab = parse_int(argv[i] + 8);
        }
        else if (std::strcmp(argv[i], "--top-k") == 0 || std::strcmp(argv[i], "--top_k") == 0)
        {
            opt.top_k = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--top-k="))
        {
            opt.top_k = parse_int(argv[i] + 8);
        }
        else if (starts_with(argv[i], "--top_k="))
        {
            opt.top_k = parse_int(argv[i] + 8);
        }
        else if (std::strcmp(argv[i], "--top-p") == 0 || std::strcmp(argv[i], "--top_p") == 0)
        {
            opt.top_p = parse_float(argv[++i]);
        }
        else if (starts_with(argv[i], "--top-p="))
        {
            opt.top_p = parse_float(argv[i] + 8);
        }
        else if (starts_with(argv[i], "--top_p="))
        {
            opt.top_p = parse_float(argv[i] + 8);
        }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("Usage: %s [--op lm_head|topk_mask|softmax|top_p|all] [--hidden N] [--vocab N]\n"
                        "          [--top-k K] [--top-p P] [--bench W I]\n",
                argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            std::exit(1);
        }
    }

    if (opt.hidden <= 0 || opt.vocab <= 0)
    {
        std::fprintf(stderr, "hidden and vocab must be positive\n");
        std::exit(1);
    }
    if (opt.top_k <= 0 || opt.top_k > kMaxTopK)
    {
        std::fprintf(stderr, "top_k must be in [1, %d]\n", kMaxTopK);
        std::exit(1);
    }
    if (opt.top_k > opt.vocab)
    {
        std::fprintf(stderr, "top_k must be <= vocab\n");
        std::exit(1);
    }
    if (!(opt.top_p > 0.0f && opt.top_p <= 1.0f))
    {
        std::fprintf(stderr, "top_p must be in (0, 1]\n");
        std::exit(1);
    }
    if (opt.op != "lm_head" && opt.op != "topk_mask" && opt.op != "softmax" && opt.op != "top_p"
        && opt.op != "all")
    {
        std::fprintf(stderr, "unknown op: %s\n", opt.op.c_str());
        std::exit(1);
    }
    return opt;
}

__global__ void init_half_kernel(half* ptr, long long n, float scale)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = idx; i < n; i += stride)
    {
        float v = static_cast<float>((i * 13 + 7) & 1023) * (scale / 1024.0f);
        ptr[i] = __float2half(v);
    }
}

__global__ void init_logits_kernel(float* logits, int vocab)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < vocab; i += stride)
    {
        int x = (i * 1103515245 + 12345) & 0x7fffffff;
        logits[i] = static_cast<float>(x % 20000) * 0.0005f - 5.0f;
    }
}

__global__ void init_probs_kernel(float* probs, int vocab, int top_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float denom = static_cast<float>(top_k * (top_k + 1)) * 0.5f;
    for (int i = idx; i < vocab; i += stride)
    {
        probs[i] = (i < top_k) ? static_cast<float>(top_k - i) / denom : 0.0f;
    }
}

void launch_lm_head(cublasHandle_t handle, half const* weight, half const* hidden, float* logits, int vocab, int hidden_dim)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, vocab, 1, hidden_dim, &alpha, weight, CUDA_R_16F,
        vocab, hidden, CUDA_R_16F, hidden_dim, &beta, logits, CUDA_R_32F, vocab, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void launch_topk_mask(float* logits, float* masked_logits, int vocab, int top_k, void* row_states)
{
    CHECK_CUDA((flashinfer::sampling::RadixTopKMaskLogitsMultiCTA<float, int>(
        logits, masked_logits, nullptr, 1, static_cast<uint32_t>(top_k), static_cast<uint32_t>(vocab),
        static_cast<flashinfer::sampling::RadixRowState*>(row_states))));
}

void launch_softmax(float* logits, float* probs, int vocab, void* workspace)
{
    CHECK_CUDA((flashinfer::sampling::OnlineSoftmax<float>(
        logits, probs, 1, static_cast<uint32_t>(vocab), nullptr, 1.0f, workspace, kFlashInferWorkspaceBytes, false)));
}

void launch_top_p(float* probs, int* token, bool* valid, int vocab, float top_p)
{
    CHECK_CUDA((flashinfer::sampling::TopPSamplingFromProb<float, int>(
        probs, token, valid, nullptr, nullptr, 1, top_p, static_cast<uint32_t>(vocab), true, nullptr, 1,
        nullptr, 0)));
}

} // namespace

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    Options opt = parse_args(argc, argv);

    bool const need_lm = (opt.op == "lm_head" || opt.op == "all");
    bool const need_logits = (need_lm || opt.op == "topk_mask" || opt.op == "softmax");
    bool const need_masked = (opt.op == "topk_mask" || opt.op == "all");
    bool const need_probs = (opt.op == "softmax" || opt.op == "top_p" || opt.op == "all");
    bool const need_softmax_workspace = (opt.op == "softmax" || opt.op == "all");
    bool const need_topk_row_states = (opt.op == "topk_mask" || opt.op == "all");
    bool const need_sampling_valid = (opt.op == "top_p" || opt.op == "all");

    std::printf("sampling bench: op=%s hidden=%d vocab=%d top_k=%d top_p=%.3f\n", opt.op.c_str(), opt.hidden,
        opt.vocab, opt.top_k, opt.top_p);

    half* d_hidden = nullptr;
    half* d_weight = nullptr;
    float* d_logits = nullptr;
    float* d_masked = nullptr;
    float* d_probs = nullptr;
    int* d_token = nullptr;
    bool* d_valid = nullptr;
    void* d_softmax_workspace = nullptr;
    void* d_topk_row_states = nullptr;

    if (need_lm)
    {
        CHECK_CUDA(cudaMalloc(&d_hidden, static_cast<size_t>(opt.hidden) * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_weight, static_cast<size_t>(opt.vocab) * opt.hidden * sizeof(half)));
        int blocks_hidden = (opt.hidden + 255) / 256;
        int blocks_weight = 4096;
        init_half_kernel<<<blocks_hidden, 256>>>(d_hidden, opt.hidden, 0.25f);
        init_half_kernel<<<blocks_weight, 256>>>(d_weight, static_cast<long long>(opt.vocab) * opt.hidden, 0.03125f);
    }
    if (need_logits)
    {
        CHECK_CUDA(cudaMalloc(&d_logits, static_cast<size_t>(opt.vocab) * sizeof(float)));
        if (!need_lm)
        {
            init_logits_kernel<<<std::min(1024, (opt.vocab + 255) / 256), 256>>>(d_logits, opt.vocab);
        }
    }
    if (need_masked)
    {
        CHECK_CUDA(cudaMalloc(&d_masked, static_cast<size_t>(opt.vocab) * sizeof(float)));
    }
    if (need_probs)
    {
        CHECK_CUDA(cudaMalloc(&d_probs, static_cast<size_t>(opt.vocab) * sizeof(float)));
        if (opt.op == "top_p")
        {
            init_probs_kernel<<<std::min(1024, (opt.vocab + 255) / 256), 256>>>(d_probs, opt.vocab, opt.top_k);
        }
    }
    CHECK_CUDA(cudaMalloc(&d_token, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_token, 0, sizeof(int)));
    if (need_sampling_valid)
    {
        CHECK_CUDA(cudaMalloc(&d_valid, sizeof(bool)));
        CHECK_CUDA(cudaMemset(d_valid, 0, sizeof(bool)));
    }
    if (need_softmax_workspace)
    {
        CHECK_CUDA(cudaMalloc(&d_softmax_workspace, kFlashInferWorkspaceBytes));
    }
    if (need_topk_row_states)
    {
        CHECK_CUDA(cudaMalloc(&d_topk_row_states, kFlashInferWorkspaceBytes));
        CHECK_CUDA(cudaMemset(d_topk_row_states, 0, kFlashInferWorkspaceBytes));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle = nullptr;
    if (need_lm)
    {
        CHECK_CUBLAS(cublasCreate(&handle));
    }

    if (opt.op == "lm_head")
    {
        timer.run([&] { launch_lm_head(handle, d_weight, d_hidden, d_logits, opt.vocab, opt.hidden); });
    }
    else if (opt.op == "topk_mask")
    {
        timer.run([&] { launch_topk_mask(d_logits, d_masked, opt.vocab, opt.top_k, d_topk_row_states); });
    }
    else if (opt.op == "softmax")
    {
        timer.run([&] { launch_softmax(d_logits, d_probs, opt.vocab, d_softmax_workspace); });
    }
    else if (opt.op == "top_p")
    {
        timer.run([&] { launch_top_p(d_probs, d_token, d_valid, opt.vocab, opt.top_p); });
    }
    else
    {
        timer.run([&] {
            launch_lm_head(handle, d_weight, d_hidden, d_logits, opt.vocab, opt.hidden);
            launch_topk_mask(d_logits, d_masked, opt.vocab, opt.top_k, d_topk_row_states);
            launch_softmax(d_masked, d_probs, opt.vocab, d_softmax_workspace);
            launch_top_p(d_probs, d_token, d_valid, opt.vocab, opt.top_p);
        });
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int token = -1;
    CHECK_CUDA(cudaMemcpy(&token, d_token, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("selected_token=%d\n", token);

    if (handle)
    {
        CHECK_CUBLAS(cublasDestroy(handle));
    }
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
    CHECK_CUDA(cudaFree(d_masked));
    CHECK_CUDA(cudaFree(d_probs));
    CHECK_CUDA(cudaFree(d_token));
    CHECK_CUDA(cudaFree(d_valid));
    CHECK_CUDA(cudaFree(d_softmax_workspace));
    CHECK_CUDA(cudaFree(d_topk_row_states));
    return 0;
}
