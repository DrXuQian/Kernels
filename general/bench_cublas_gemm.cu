// Generic cuBLAS FP16/BF16 GEMM benchmark.
//
// Computes row-major C[M,N] = A[M,K] * B[K,N].
//
// Usage:
//   general/bench_cublas_gemm --m=3823 --n=64 --k=3072 --dtype=fp16 --bench 0 1
//   general/bench_cublas_gemm --m=1 --n=248320 --k=3072 --dtype=fp16 --out-dtype=fp32 --bench 0 1

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
    int m = 3823;
    int n = 64;
    int k = 3072;
    std::string dtype = "fp16";
    std::string out_dtype = "same";
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
        else if (std::strcmp(argv[i], "--dtype") == 0)
        {
            opt.dtype = argv[++i];
        }
        else if (starts_with(argv[i], "--dtype="))
        {
            opt.dtype = argv[i] + 8;
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
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("Usage: %s --m M --n N --k K [--dtype fp16|bf16] [--out-dtype same|fp16|bf16|fp32]"
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
    if (opt.dtype != "fp16" && opt.dtype != "bf16")
    {
        std::fprintf(stderr, "dtype must be fp16 or bf16\n");
        std::exit(1);
    }
    if (opt.out_dtype == "same")
    {
        opt.out_dtype = opt.dtype;
    }
    if (opt.out_dtype != "fp16" && opt.out_dtype != "bf16" && opt.out_dtype != "fp32")
    {
        std::fprintf(stderr, "out-dtype must be same, fp16, bf16, or fp32\n");
        std::exit(1);
    }
    return opt;
}

template <typename T>
T host_from_float(float value);

template <>
half host_from_float<half>(float value)
{
    return __float2half(value);
}

template <>
__nv_bfloat16 host_from_float<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <typename T>
void fill_tensor_host(std::vector<T>& data, float scale)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        float v = static_cast<float>((i * 13 + 7) & 1023) * (scale / 1024.0f);
        data[i] = host_from_float<T>(v);
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

cudaDataType_t out_cuda_type(std::string const& dtype)
{
    if (dtype == "fp16")
    {
        return CUDA_R_16F;
    }
    if (dtype == "bf16")
    {
        return CUDA_R_16BF;
    }
    return CUDA_R_32F;
}

size_t out_type_size(std::string const& dtype)
{
    return dtype == "fp32" ? sizeof(float) : sizeof(half);
}

template <typename T>
int run_gemm(Options const& opt, BenchTimer& timer)
{
    T* d_a = nullptr;
    T* d_b = nullptr;
    void* d_c = nullptr;
    long long a_elems = static_cast<long long>(opt.m) * opt.k;
    long long b_elems = static_cast<long long>(opt.k) * opt.n;
    long long c_elems = static_cast<long long>(opt.m) * opt.n;

    CHECK_CUDA(cudaMalloc(&d_a, a_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_b, b_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_c, c_elems * out_type_size(opt.out_dtype)));

    std::vector<T> h_a(static_cast<size_t>(a_elems));
    std::vector<T> h_b(static_cast<size_t>(b_elems));
    fill_tensor_host(h_a, 0.25f);
    fill_tensor_host(h_b, 0.03125f);
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_elems * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), b_elems * sizeof(T), cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    float alpha = 1.0f;
    float beta = 0.0f;
    cudaDataType_t c_type = out_cuda_type(opt.out_dtype);
    timer.run([&] {
        // Row-major C[M,N] = A[M,K] * B[K,N].
        // cuBLAS sees column-major C^T[N,M] = B^T[N,K] * A^T[K,M].
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, opt.n, opt.m, opt.k, &alpha, d_b,
            cuda_type<T>(), opt.n, d_a, cuda_type<T>(), opt.k, &beta, d_c, c_type, opt.n, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    });
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    Options opt = parse_args(argc, argv);
    std::printf("bench cublas_gemm: m=%d n=%d k=%d dtype=%s out_dtype=%s\n", opt.m, opt.n, opt.k,
        opt.dtype.c_str(), opt.out_dtype.c_str());
    std::printf("shape: A=(%d,%d), B=(%d,%d), C=(%d,%d)\n", opt.m, opt.k, opt.k, opt.n, opt.m, opt.n);

    if (opt.dtype == "fp16")
    {
        return run_gemm<half>(opt, timer);
    }
    return run_gemm<__nv_bfloat16>(opt, timer);
}
