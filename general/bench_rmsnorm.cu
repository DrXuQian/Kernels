// Standalone RMSNorm benchmark adapted from TensorRT-LLM generalRmsNorm.
// Source reference:
//   NVIDIA/TensorRT-LLM cpp/tensorrt_llm/kernels/rmsnormKernels.cu
//
// y = x * rsqrt(mean(x^2) + eps) * gamma (+ beta)
// Usage:
//   ./bench_rmsnorm --batch 1 --embed 3072 --dtype fp16 --bench 20 100

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

namespace trtllm_rmsnorm_standalone
{

template <typename T>
__host__ __device__ inline float to_float(T v)
{
    return static_cast<float>(v);
}

template <>
__host__ __device__ inline float to_float<half>(half v)
{
    return __half2float(v);
}

template <>
__host__ __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

template <typename T>
__host__ __device__ inline T from_float(float v)
{
    return static_cast<T>(v);
}

template <>
__host__ __device__ inline half from_float<half>(float v)
{
    return __float2half(v);
}

template <>
__host__ __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v)
{
    return __float2bfloat16(v);
}

__device__ inline float block_reduce_sum(float value)
{
    __shared__ float warp_sums[32];
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const num_warps = (blockDim.x + 31) >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if (lane == 0)
    {
        warp_sums[warp] = value;
    }
    __syncthreads();

    value = 0.0f;
    if (warp == 0)
    {
        value = (lane < num_warps) ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    return value;
}

template <typename T, bool USE_SHMEM, bool HAS_BETA>
__global__ void general_rmsnorm_kernel(T const* __restrict__ input, T const* __restrict__ gamma,
    T const* __restrict__ beta, T* __restrict__ output, int tokens, int hidden_dim, float eps)
{
    int const row = blockIdx.x;
    if (row >= tokens)
    {
        return;
    }

    extern __shared__ __align__(sizeof(float)) unsigned char smem_raw[];
    T* shmem = reinterpret_cast<T*>(smem_raw);

    __shared__ float s_inv_rms;
    T const* input_row = input + static_cast<long long>(row) * hidden_dim;
    T* output_row = output + static_cast<long long>(row) * hidden_dim;

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x)
    {
        T const x = input_row[col];
        if (USE_SHMEM)
        {
            shmem[col] = x;
        }
        float const xf = to_float(x);
        local_sum += xf * xf;
    }

    float const row_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0)
    {
        s_inv_rms = rsqrtf(row_sum / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x)
    {
        T const x = USE_SHMEM ? shmem[col] : input_row[col];
        float yf = to_float(x) * s_inv_rms * to_float(gamma[col]);
        if (HAS_BETA)
        {
            yf += to_float(beta[col]);
        }
        output_row[col] = from_float<T>(yf);
    }
}

inline int round_block_size(int hidden_dim)
{
    int block = std::min(hidden_dim, 1024);
    block = 32 * ((block + 31) / 32);
    return std::max(block, 32);
}

template <typename T, bool HAS_BETA>
void launch_general_rmsnorm(
    T const* input, T const* gamma, T const* beta, T* output, int tokens, int hidden_dim, float eps, cudaStream_t stream)
{
    dim3 grid(tokens);
    dim3 block(round_block_size(hidden_dim));
    size_t const shmem_size = static_cast<size_t>(hidden_dim) * sizeof(T);

    bool use_shmem = true;
    if (shmem_size >= (48u << 10))
    {
        cudaError_t const ret = cudaFuncSetAttribute(general_rmsnorm_kernel<T, true, HAS_BETA>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        use_shmem = (ret == cudaSuccess);
    }

    if (use_shmem)
    {
        general_rmsnorm_kernel<T, true, HAS_BETA>
            <<<grid, block, shmem_size, stream>>>(input, gamma, beta, output, tokens, hidden_dim, eps);
    }
    else
    {
        general_rmsnorm_kernel<T, false, HAS_BETA>
            <<<grid, block, 0, stream>>>(input, gamma, beta, output, tokens, hidden_dim, eps);
    }
}

template <typename T>
void fill_input(std::vector<T>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        float const v = 0.01f * static_cast<float>(static_cast<int>(i % 101) - 50);
        data[i] = from_float<T>(v);
    }
}

template <typename T>
void fill_gamma(std::vector<T>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        float const v = 0.75f + 0.0005f * static_cast<float>(i % 257);
        data[i] = from_float<T>(v);
    }
}

template <typename T>
void fill_beta(std::vector<T>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        float const v = 0.001f * static_cast<float>(static_cast<int>(i % 17) - 8);
        data[i] = from_float<T>(v);
    }
}

template <typename T>
void run_rmsnorm(int batch, int embed, float eps, bool has_beta, bool check, BenchTimer& timer)
{
    size_t const total = static_cast<size_t>(batch) * embed;
    size_t const total_bytes = total * sizeof(T);
    size_t const weight_bytes = static_cast<size_t>(embed) * sizeof(T);

    std::vector<T> h_input(total);
    std::vector<T> h_gamma(embed);
    std::vector<T> h_beta(embed);
    fill_input(h_input);
    fill_gamma(h_gamma);
    fill_beta(h_beta);

    T* d_input = nullptr;
    T* d_gamma = nullptr;
    T* d_beta = nullptr;
    T* d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_gamma, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, total_bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), total_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), weight_bytes, cudaMemcpyHostToDevice));

    if (has_beta)
    {
        CHECK_CUDA(cudaMalloc(&d_beta, weight_bytes));
        CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), weight_bytes, cudaMemcpyHostToDevice));
    }

    if (has_beta)
    {
        timer.run([&]() { launch_general_rmsnorm<T, true>(d_input, d_gamma, d_beta, d_output, batch, embed, eps, 0); });
    }
    else
    {
        timer.run([&]() { launch_general_rmsnorm<T, false>(d_input, d_gamma, nullptr, d_output, batch, embed, eps, 0); });
    }
    CHECK_CUDA(cudaGetLastError());

    if (check)
    {
        std::vector<T> h_output(total);
        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, total_bytes, cudaMemcpyDeviceToHost));

        double max_abs = 0.0;
        for (int row = 0; row < batch; ++row)
        {
            double sum_sq = 0.0;
            for (int col = 0; col < embed; ++col)
            {
                double const x = to_float(h_input[static_cast<size_t>(row) * embed + col]);
                sum_sq += x * x;
            }
            double const inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(embed) + eps);
            for (int col = 0; col < embed; ++col)
            {
                size_t const idx = static_cast<size_t>(row) * embed + col;
                double ref = to_float(h_input[idx]) * inv_rms * to_float(h_gamma[col]);
                if (has_beta)
                {
                    ref += to_float(h_beta[col]);
                }
                double const got = to_float(h_output[idx]);
                max_abs = std::max(max_abs, std::abs(ref - got));
            }
        }

        double const tol = std::is_same<T, float>::value ? 2e-5 : std::is_same<T, half>::value ? 2e-3 : 2e-2;
        std::printf("check: max_abs=%g tol=%g %s\n", max_abs, tol, max_abs <= tol ? "PASS" : "FAIL");
        if (max_abs > tol)
        {
            std::exit(1);
        }
    }

    char const* dtype = std::is_same<T, half>::value ? "fp16"
        : std::is_same<T, __nv_bfloat16>::value        ? "bf16"
                                                       : "fp32";
    size_t const bytes = total_bytes * 2 + weight_bytes + (has_beta ? weight_bytes : 0);
    std::printf("bench rmsnorm: batch=%d embed=%d dtype=%s eps=%g beta=%d bytes_per_launch=%zu\n", batch, embed, dtype,
        eps, static_cast<int>(has_beta), bytes);

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
}

} // namespace trtllm_rmsnorm_standalone

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int batch = 1;
    int embed = 3072;
    float eps = 1e-6f;
    bool has_beta = false;
    bool check = true;
    std::string dtype = "fp16";

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
        {
            batch = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--embed") == 0 && i + 1 < argc)
        {
            embed = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--dtype") == 0 && i + 1 < argc)
        {
            dtype = argv[++i];
        }
        else if (std::strcmp(argv[i], "--eps") == 0 && i + 1 < argc)
        {
            eps = std::strtof(argv[++i], nullptr);
        }
        else if (std::strcmp(argv[i], "--beta") == 0)
        {
            has_beta = true;
        }
        else if (std::strcmp(argv[i], "--no-check") == 0)
        {
            check = false;
        }
        else
        {
            std::fprintf(stderr,
                "Usage: %s [--batch N] [--embed N] [--dtype fp16|bf16|fp32] [--eps F] [--beta] [--no-check] "
                "[--bench W I]\n",
                argv[0]);
            return 1;
        }
    }

    if (batch <= 0 || embed <= 0)
    {
        std::fprintf(stderr, "batch and embed must be positive\n");
        return 1;
    }

    using namespace trtllm_rmsnorm_standalone;
    if (dtype == "fp16" || dtype == "float16" || dtype == "half")
    {
        run_rmsnorm<half>(batch, embed, eps, has_beta, check, timer);
    }
    else if (dtype == "bf16" || dtype == "bfloat16")
    {
        run_rmsnorm<__nv_bfloat16>(batch, embed, eps, has_beta, check, timer);
    }
    else if (dtype == "fp32" || dtype == "float32" || dtype == "float")
    {
        run_rmsnorm<float>(batch, embed, eps, has_beta, check, timer);
    }
    else
    {
        std::fprintf(stderr, "unsupported dtype: %s\n", dtype.c_str());
        return 1;
    }
    return 0;
}
