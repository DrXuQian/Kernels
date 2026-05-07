// Shared-expert adjunct benchmarks for Qwen3.5 MoE-FFN.
//
// Source semantics:
//   TensorRT-LLM and vLLM Qwen3-Next paths compute
//     shared = shared_expert(hidden_states)
//     shared = sigmoid(shared_expert_gate(hidden_states)) * shared
//     output = routed_moe_output + shared
//
// This standalone benchmark isolates:
//   gate_gemv        : shared_expert_gate, dense FP16/BF16 GEMM with N=1
//   router_gate_gemm : router gate, dense FP16/BF16 GEMM with N=num_experts
//   sigmoid_mul_add  : output = routed + sigmoid(gate) * shared

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
    std::string op = "gate_gemv";
    int tokens = 3823;
    int hidden = 3072;
    int out_dim = 1;
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
        else if (std::strcmp(argv[i], "--out-dim") == 0 || std::strcmp(argv[i], "--experts") == 0)
        {
            opt.out_dim = parse_int(argv[++i]);
        }
        else if (starts_with(argv[i], "--out-dim="))
        {
            opt.out_dim = parse_int(argv[i] + 10);
        }
        else if (starts_with(argv[i], "--experts="))
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
            std::printf("Usage: %s --op gate_gemv|router_gate_gemm|sigmoid_mul_add [--tokens N] [--hidden N]"
                        " [--out-dim N|--experts N] [--dtype fp16|bf16] [--bench W I]\n",
                argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            std::exit(1);
        }
    }

    if (opt.op != "gate_gemv" && opt.op != "router_gate_gemm" && opt.op != "sigmoid_mul_add")
    {
        std::fprintf(stderr, "unknown op: %s\n", opt.op.c_str());
        std::exit(1);
    }
    if (opt.dtype != "fp16" && opt.dtype != "bf16")
    {
        std::fprintf(stderr, "dtype must be fp16 or bf16\n");
        std::exit(1);
    }
    if (opt.op == "gate_gemv")
    {
        opt.out_dim = 1;
    }
    else if (opt.op == "router_gate_gemm" && opt.out_dim == 1)
    {
        opt.out_dim = 256;
    }
    if (opt.tokens <= 0 || opt.hidden <= 0 || opt.out_dim <= 0)
    {
        std::fprintf(stderr, "tokens, hidden, and out_dim must be positive\n");
        std::exit(1);
    }
    return opt;
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
__global__ void sigmoid_mul_add_kernel(
    T* __restrict__ out, T const* __restrict__ routed, T const* __restrict__ shared, T const* __restrict__ gate,
    int tokens, int hidden)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(tokens) * hidden;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = idx; i < total; i += stride)
    {
        int row = static_cast<int>(i / hidden);
        float g = to_float<T>(gate[row]);
        float s = 1.0f / (1.0f + expf(-g));
        float v = to_float<T>(routed[i]) + s * to_float<T>(shared[i]);
        out[i] = from_float<T>(v);
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
void init_tensor(T* ptr, long long n, float scale)
{
    int blocks = 256;
    init_tensor_kernel<T><<<blocks, 256>>>(ptr, n, scale);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
int run_gate_gemm(Options const& opt, BenchTimer& timer)
{
    T* d_hidden = nullptr;
    T* d_weight = nullptr;
    T* d_out = nullptr;
    long long hidden_elems = static_cast<long long>(opt.tokens) * opt.hidden;
    long long weight_elems = static_cast<long long>(opt.hidden) * opt.out_dim;
    long long out_elems = static_cast<long long>(opt.tokens) * opt.out_dim;
    CHECK_CUDA(cudaMalloc(&d_hidden, hidden_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out, out_elems * sizeof(T)));
    init_tensor(d_hidden, hidden_elems, 0.2f);
    init_tensor(d_weight, weight_elems, 0.1f);
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    float alpha = 1.0f;
    float beta = 0.0f;
    auto launch = [&]() {
        // Row-major C[M,N] = A[M,K] * B[K,N].
        // cuBLAS sees column-major C^T[N,M] = B^T[N,K] * A^T[K,M].
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, opt.out_dim, opt.tokens, opt.hidden, &alpha,
            d_weight, cuda_type<T>(), opt.out_dim, d_hidden, cuda_type<T>(), opt.hidden, &beta, d_out,
            cuda_type<T>(), opt.out_dim, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    timer.run(launch);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}

template <typename T>
int run_sigmoid_mul_add(Options const& opt, BenchTimer& timer)
{
    T* d_routed = nullptr;
    T* d_shared = nullptr;
    T* d_gate = nullptr;
    T* d_out = nullptr;
    long long hidden_elems = static_cast<long long>(opt.tokens) * opt.hidden;
    CHECK_CUDA(cudaMalloc(&d_routed, hidden_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_shared, hidden_elems * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_gate, opt.tokens * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out, hidden_elems * sizeof(T)));
    init_tensor(d_routed, hidden_elems, 0.2f);
    init_tensor(d_shared, hidden_elems, 0.3f);
    init_tensor(d_gate, opt.tokens, 0.1f);
    CHECK_CUDA(cudaDeviceSynchronize());

    int threads = 256;
    int blocks = static_cast<int>((hidden_elems + threads - 1) / threads);
    blocks = blocks < 1 ? 1 : (blocks > 65535 ? 65535 : blocks);
    auto launch = [&]() {
        sigmoid_mul_add_kernel<T><<<blocks, threads>>>(d_out, d_routed, d_shared, d_gate, opt.tokens, opt.hidden);
        CHECK_CUDA(cudaGetLastError());
    };

    timer.run(launch);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_routed));
    CHECK_CUDA(cudaFree(d_shared));
    CHECK_CUDA(cudaFree(d_gate));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}

template <typename T>
int dispatch(Options const& opt, BenchTimer& timer)
{
    if (opt.op == "gate_gemv" || opt.op == "router_gate_gemm")
    {
        return run_gate_gemm<T>(opt, timer);
    }
    return run_sigmoid_mul_add<T>(opt, timer);
}

} // namespace

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);
    Options opt = parse_args(argc, argv);

    std::printf("bench shared_expert: op=%s tokens=%d hidden=%d out_dim=%d dtype=%s\n", opt.op.c_str(), opt.tokens,
        opt.hidden, opt.out_dim, opt.dtype.c_str());
    if (opt.op == "gate_gemv" || opt.op == "router_gate_gemm")
    {
        std::printf("shape: hidden_states=(%d,%d), weight=(%d,%d), output=(%d,%d)\n", opt.tokens, opt.hidden,
            opt.hidden, opt.out_dim, opt.tokens, opt.out_dim);
    }
    else
    {
        std::printf("shape: routed/shared=(%d,%d), gate=(%d,1), output=(%d,%d)\n", opt.tokens, opt.hidden,
            opt.tokens, opt.tokens, opt.hidden);
    }

    if (opt.dtype == "fp16")
    {
        return dispatch<half>(opt, timer);
    }
    return dispatch<__nv_bfloat16>(opt, timer);
}
