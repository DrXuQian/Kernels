#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (x);                                                                                         \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_CUBLAS(x)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t err = (x);                                                                                      \
        if (err != CUBLAS_STATUS_SUCCESS)                                                                              \
        {                                                                                                              \
            std::fprintf(stderr, "cuBLAS %s:%d: status=%d\n", __FILE__, __LINE__, static_cast<int>(err));             \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace
{

struct Options
{
    std::string op = "all";
    int n = 248320;
    int k = 3072;
    int warps_per_block = 8;
    int warmup = 100;
    int iters = 200;
};

bool starts_with(char const* s, char const* prefix)
{
    return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

Options parse_args(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        if (starts_with(argv[i], "--op="))
        {
            opt.op = argv[i] + 5;
        }
        else if (std::strcmp(argv[i], "--op") == 0)
        {
            opt.op = argv[++i];
        }
        else if (starts_with(argv[i], "--n="))
        {
            opt.n = std::atoi(argv[i] + 4);
        }
        else if (std::strcmp(argv[i], "--n") == 0)
        {
            opt.n = std::atoi(argv[++i]);
        }
        else if (starts_with(argv[i], "--k="))
        {
            opt.k = std::atoi(argv[i] + 4);
        }
        else if (std::strcmp(argv[i], "--k") == 0)
        {
            opt.k = std::atoi(argv[++i]);
        }
        else if (starts_with(argv[i], "--warps-per-block="))
        {
            opt.warps_per_block = std::atoi(argv[i] + 18);
        }
        else if (std::strcmp(argv[i], "--warps-per-block") == 0)
        {
            opt.warps_per_block = std::atoi(argv[++i]);
        }
        else if (starts_with(argv[i], "--warmup="))
        {
            opt.warmup = std::atoi(argv[i] + 9);
        }
        else if (std::strcmp(argv[i], "--warmup") == 0)
        {
            opt.warmup = std::atoi(argv[++i]);
        }
        else if (starts_with(argv[i], "--iters="))
        {
            opt.iters = std::atoi(argv[i] + 8);
        }
        else if (std::strcmp(argv[i], "--iters") == 0)
        {
            opt.iters = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("Usage: %s [--op all|shared|global|cublas|copy|copy_u8]\n"
                        "          [--n vocab] [--k hidden] [--warps-per-block 4|8|16]\n"
                        "          [--warmup N] [--iters N]\n",
                argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            std::exit(1);
        }
    }

    if (opt.n <= 0 || opt.k <= 0 || opt.warmup < 0 || opt.iters <= 0)
    {
        std::fprintf(stderr, "n/k/iters must be positive and warmup must be non-negative\n");
        std::exit(1);
    }
    if (opt.warps_per_block != 4 && opt.warps_per_block != 8 && opt.warps_per_block != 16)
    {
        std::fprintf(stderr, "warps-per-block must be 4, 8, or 16\n");
        std::exit(1);
    }
    return opt;
}

template <typename F>
float median_time_ms(F&& fn, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i)
    {
        fn();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i)
    {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        fn();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&times[i], start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int WarpsPerBlock, bool CacheActivation>
__global__ void lm_head_gemv_kernel(half const* __restrict__ hidden, half const* __restrict__ weight,
    float* __restrict__ logits, int n, int k)
{
    extern __shared__ half s_hidden[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row = blockIdx.x * WarpsPerBlock + warp;

    if constexpr (CacheActivation)
    {
        for (int i = tid; i < k; i += blockDim.x)
        {
            s_hidden[i] = hidden[i];
        }
        __syncthreads();
    }

    if (row >= n)
    {
        return;
    }

    int k2 = k / 2;
    half2 const* w2 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row) * k);
    half2 const* h2 = CacheActivation ? reinterpret_cast<half2 const*>(s_hidden) : reinterpret_cast<half2 const*>(hidden);

    float acc = 0.0f;
    for (int i = lane; i < k2; i += 32)
    {
        half2 hv = h2[i];
        half2 wv = w2[i];
        float2 hf = __half22float2(hv);
        float2 wf = __half22float2(wv);
        acc = fmaf(hf.x, wf.x, acc);
        acc = fmaf(hf.y, wf.y, acc);
    }

    if ((k & 1) && lane == 0)
    {
        int i = k - 1;
        half hv = CacheActivation ? s_hidden[i] : hidden[i];
        acc = fmaf(__half2float(hv), __half2float(weight[static_cast<long long>(row) * k + i]), acc);
    }

    acc = warp_sum(acc);
    if (lane == 0)
    {
        logits[row] = acc;
    }
}

__global__ void copy_u8_kernel(ulonglong4* __restrict__ dst, ulonglong4 const* __restrict__ src, size_t n)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < n; i += stride)
    {
        dst[i] = src[i];
    }
}

template <int WarpsPerBlock, bool CacheActivation>
float run_lm_head_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + WarpsPerBlock - 1) / WarpsPerBlock;
    size_t smem = CacheActivation ? static_cast<size_t>(opt.k) * sizeof(half) : 0;
    return median_time_ms(
        [&] {
            lm_head_gemv_kernel<WarpsPerBlock, CacheActivation>
                <<<blocks, threads, smem>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head(Options const& opt, bool cache_activation, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (cache_activation)
    {
        if (opt.warps_per_block == 4)
        {
            return run_lm_head_kernel<4, true>(opt, d_hidden, d_weight, d_logits);
        }
        if (opt.warps_per_block == 8)
        {
            return run_lm_head_kernel<8, true>(opt, d_hidden, d_weight, d_logits);
        }
        return run_lm_head_kernel<16, true>(opt, d_hidden, d_weight, d_logits);
    }

    if (opt.warps_per_block == 4)
    {
        return run_lm_head_kernel<4, false>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_kernel<8, false>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_kernel<16, false>(opt, d_hidden, d_weight, d_logits);
}

void print_bw(char const* name, float ms, double bytes)
{
    double gbps = bytes / (ms * 1.0e-3) / 1.0e9;
    std::printf("%-14s median=%.4f ms traffic=%.3f MB bw=%.1f GB/s %.3f TB/s\n", name, ms, bytes / 1.0e6,
        gbps, gbps / 1000.0);
}

void run_gemv_cases(Options const& opt, bool run_shared, bool run_global)
{
    size_t hidden_bytes = static_cast<size_t>(opt.k) * sizeof(half);
    size_t weight_bytes = static_cast<size_t>(opt.n) * opt.k * sizeof(half);
    size_t logits_bytes = static_cast<size_t>(opt.n) * sizeof(float);
    half* d_hidden = nullptr;
    half* d_weight = nullptr;
    float* d_logits = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hidden, hidden_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_logits, logits_bytes));
    CHECK_CUDA(cudaMemset(d_hidden, 0x3a, hidden_bytes));
    CHECK_CUDA(cudaMemset(d_weight, 0x1d, weight_bytes));
    CHECK_CUDA(cudaMemset(d_logits, 0, logits_bytes));

    double mandatory_traffic = static_cast<double>(weight_bytes + logits_bytes);
    std::printf("lm_head: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d\n", opt.n, opt.k,
        weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block);
    if (run_shared)
    {
        float ms = run_lm_head(opt, true, d_hidden, d_weight, d_logits);
        CHECK_CUDA(cudaGetLastError());
        print_bw("shared_a", ms, mandatory_traffic);
    }
    if (run_global)
    {
        float ms = run_lm_head(opt, false, d_hidden, d_weight, d_logits);
        CHECK_CUDA(cudaGetLastError());
        print_bw("global_a", ms, mandatory_traffic);
    }

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_copy_case(Options const& opt)
{
    size_t bytes = static_cast<size_t>(opt.n) * opt.k * sizeof(half);
    bytes = bytes / sizeof(ulonglong4) * sizeof(ulonglong4);
    void* src = nullptr;
    void* dst = nullptr;
    CHECK_CUDA(cudaMalloc(&src, bytes));
    CHECK_CUDA(cudaMalloc(&dst, bytes));
    CHECK_CUDA(cudaMemset(src, 0x5a, bytes));
    CHECK_CUDA(cudaMemset(dst, 0, bytes));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int threads = 256;
    int blocks = prop.multiProcessorCount * 16;
    size_t n_vec = bytes / sizeof(ulonglong4);
    float ms = median_time_ms(
        [&] { copy_u8_kernel<<<blocks, threads>>>(static_cast<ulonglong4*>(dst), static_cast<ulonglong4 const*>(src), n_vec); },
        opt.warmup, opt.iters);
    CHECK_CUDA(cudaGetLastError());
    print_bw("copy_u8", ms, static_cast<double>(bytes) * 2.0);

    CHECK_CUDA(cudaFree(src));
    CHECK_CUDA(cudaFree(dst));
}

void run_cublas_case(Options const& opt)
{
    size_t hidden_bytes = static_cast<size_t>(opt.k) * sizeof(half);
    size_t weight_bytes = static_cast<size_t>(opt.k) * opt.n * sizeof(half);
    size_t logits_bytes = static_cast<size_t>(opt.n) * sizeof(float);
    half* d_hidden = nullptr;
    half* d_weight_kn = nullptr;
    float* d_logits = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hidden, hidden_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight_kn, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_logits, logits_bytes));
    CHECK_CUDA(cudaMemset(d_hidden, 0x3a, hidden_bytes));
    CHECK_CUDA(cudaMemset(d_weight_kn, 0x1d, weight_bytes));
    CHECK_CUDA(cudaMemset(d_logits, 0, logits_bytes));

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;

    // Row-major C[1,N] = A[1,K] * B[K,N], expressed as column-major GEMM:
    // C_col[N,1] = B_col[N,K] * A_col[K,1].
    float ms = median_time_ms(
        [&] {
            CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, opt.n, 1, opt.k, &alpha, d_weight_kn,
                CUDA_R_16F, opt.n, d_hidden, CUDA_R_16F, opt.k, &beta, d_logits, CUDA_R_32F, opt.n, CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        },
        opt.warmup, opt.iters);
    CHECK_CUDA(cudaGetLastError());

    std::printf("cublas lm_head: n=%d k=%d weight=%.3f MB logits=%.3f MB\n", opt.n, opt.k, weight_bytes / 1.0e6,
        logits_bytes / 1.0e6);
    print_bw("cublas", ms, static_cast<double>(weight_bytes + logits_bytes));

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight_kn));
    CHECK_CUDA(cudaFree(d_logits));
}

} // namespace

int main(int argc, char** argv)
{
    Options opt = parse_args(argc, argv);
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("device=%s sms=%d op=%s warmup=%d iters=%d\n", prop.name, prop.multiProcessorCount, opt.op.c_str(),
        opt.warmup, opt.iters);

    if (opt.op == "all")
    {
        run_gemv_cases(opt, true, true);
        run_cublas_case(opt);
        run_copy_case(opt);
    }
    else if (opt.op == "shared")
    {
        run_gemv_cases(opt, true, false);
    }
    else if (opt.op == "global")
    {
        run_gemv_cases(opt, false, true);
    }
    else if (opt.op == "copy" || opt.op == "copy_u8")
    {
        run_copy_case(opt);
    }
    else if (opt.op == "cublas")
    {
        run_cublas_case(opt);
    }
    else
    {
        std::fprintf(stderr, "unknown op: %s\n", opt.op.c_str());
        return 1;
    }
    return 0;
}
