// Linear-attention dense FP16/BF16 adjunct benchmarks for Qwen3.5-style blocks.
//
// Isolates:
//   in_proj_a/b : cuBLAS GEMM, (tokens, hidden) x (hidden, out_dim) -> (tokens, out_dim)
//   residual_add: output = residual + update, shape (tokens, hidden)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include <cublas_v2.h>
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

struct Options
{
    std::string op = "in_proj_a";
    int tokens = 3823;
    int hidden = 3072;
    int out_dim = 64;
    std::string dtype = "fp16";
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
        if (std::strcmp(argv[i], "--op") == 0)
        {
            opt.op = argv[++i];
        }
        else if (starts_with(argv[i], "--op="))
        {
            opt.op = argv[i] + 5;
        }
        else if (std::strcmp(argv[i], "--tokens") == 0 || std::strcmp(argv[i], "--batch") == 0)
        {
            opt.tokens = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--tokens="))
        {
            opt.tokens = parse_int(argv[i] + 9);
        }
        else if (starts_with(argv[i], "--batch="))
        {
            opt.tokens = parse_int(argv[i] + 8);
        }
        else if (std::strcmp(argv[i], "--hidden") == 0 || std::strcmp(argv[i], "--embed") == 0)
        {
            opt.hidden = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--hidden="))
        {
            opt.hidden = parse_int(argv[i] + 9);
        }
        else if (starts_with(argv[i], "--embed="))
        {
            opt.hidden = parse_int(argv[i] + 8);
        }
        else if (std::strcmp(argv[i], "--out-dim") == 0 || std::strcmp(argv[i], "--out_dim") == 0)
        {
            opt.out_dim = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--out-dim="))
        {
            opt.out_dim = parse_int(argv[i] + 10);
        }
        else if (starts_with(argv[i], "--out_dim="))
        {
            opt.out_dim = parse_int(argv[i] + 10);
        }
        else if (std::strcmp(argv[i], "--dtype") == 0)
        {
            opt.dtype = argv[++i];
        }
        else if (starts_with(argv[i], "--dtype="))
        {
            opt.dtype = argv[i] + 8;
        }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("Usage: %s --op in_proj_a|in_proj_b|residual_add [--tokens N] [--hidden N]\n"
                        "          [--out-dim N] [--dtype fp16|bf16] [--bench W I]\n",
                argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            std::exit(1);
        }
    }

    if (opt.op != "in_proj_a" && opt.op != "in_proj_b" && opt.op != "residual_add")
    {
        std::fprintf(stderr, "unknown op: %s\n", opt.op.c_str());
        std::exit(1);
    }
    if (opt.dtype != "fp16" && opt.dtype != "bf16")
    {
        std::fprintf(stderr, "dtype must be fp16 or bf16\n");
        std::exit(1);
    }
    if (opt.tokens <= 0 || opt.hidden <= 0 || opt.out_dim <= 0)
    {
        std::fprintf(stderr, "tokens, hidden, and out_dim must be positive\n");
        std::exit(1);
    }
    return opt;
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value)
{
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T>
__global__ void init_tensor_kernel(T* ptr, long long n, float scale)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = idx; i < n; i += stride)
    {
        float v = static_cast<float>((i * 13 + 7) & 1023) * (scale / 1024.0f);
        ptr[i] = from_float<T>(v);
    }
}

template <typename T>
__global__ void residual_add_kernel(T* __restrict__ out, T const* __restrict__ residual,
    T const* __restrict__ update, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = idx; i < n; i += stride)
    {
        out[i] = from_float<T>(to_float<T>(residual[i]) + to_float<T>(update[i]));
    }
}

template <typename T>
cudaDataType_t cuda_type();

template <>
cudaDataType_t cuda_type<half>()
{
    return CUDA_R_16F;
}

template <>
cudaDataType_t cuda_type<__nv_bfloat16>()
{
    return CUDA_R_16BF;
}

template <typename T>
int run_gemm(Options const& opt, BenchTimer& timer)
{
    T* d_hidden = nullptr;
    T* d_weight = nullptr;
    T* d_out = nullptr;
    size_t hidden_elems = static_cast<size_t>(opt.tokens) * opt.hidden;
    size_t weight_elems = static_cast<size_t>(opt.hidden) * opt.out_dim;
    size_t out_elems = static_cast<size_t>(opt.tokens) * opt.out_dim;

    CHECK_CUDA(cudaMalloc(&d_hidden, hidden_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out, out_elems * sizeof(T)));
    init_tensor_kernel<T><<<std::min(4096LL, (static_cast<long long>(hidden_elems) + 255) / 256), 256>>>(
        d_hidden, hidden_elems, 0.25f);
    init_tensor_kernel<T><<<std::min(4096LL, (static_cast<long long>(weight_elems) + 255) / 256), 256>>>(
        d_weight, weight_elems, 0.03125f);
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;
    timer.run([&] {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, opt.out_dim, opt.tokens, opt.hidden, &alpha,
            d_weight, cuda_type<T>(), opt.out_dim, d_hidden, cuda_type<T>(), opt.hidden, &beta, d_out,
            cuda_type<T>(), opt.out_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    });
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}

template <typename T>
int run_residual_add(Options const& opt, BenchTimer& timer)
{
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_out = nullptr;
    long long elems = static_cast<long long>(opt.tokens) * opt.hidden;
    CHECK_CUDA(cudaMalloc(&d_a, elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b, elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out, elems * sizeof(T)));
    init_tensor_kernel<T><<<std::min(4096LL, (elems + 255) / 256), 256>>>(d_a, elems, 0.25f);
    init_tensor_kernel<T><<<std::min(4096LL, (elems + 255) / 256), 256>>>(d_b, elems, 0.125f);
    CHECK_CUDA(cudaDeviceSynchronize());

    int threads = 256;
    int blocks = static_cast<int>(std::min(4096LL, (elems + threads - 1) / threads));
    timer.run([&] { residual_add_kernel<T><<<blocks, threads>>>(d_out, d_a, d_b, elems); });
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}

template <typename T>
int dispatch(Options const& opt, BenchTimer& timer)
{
    if (opt.op == "residual_add")
    {
        return run_residual_add<T>(opt, timer);
    }
    return run_gemm<T>(opt, timer);
}

} // namespace

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    Options opt = parse_args(argc, argv);
    std::printf("linear ops bench: op=%s tokens=%d hidden=%d out_dim=%d dtype=%s\n", opt.op.c_str(), opt.tokens,
        opt.hidden, opt.out_dim, opt.dtype.c_str());
    if (opt.op == "residual_add")
    {
        std::printf("shape: (%d,%d) + (%d,%d) -> (%d,%d)\n", opt.tokens, opt.hidden, opt.tokens, opt.hidden,
            opt.tokens, opt.hidden);
    }
    else
    {
        std::printf("shape: (%d,%d) x (%d,%d) -> (%d,%d)\n", opt.tokens, opt.hidden, opt.hidden, opt.out_dim,
            opt.tokens, opt.out_dim);
    }

    if (opt.dtype == "fp16")
    {
        return dispatch<half>(opt, timer);
    }
    return dispatch<__nv_bfloat16>(opt, timer);
}
