#include <algorithm>
#include <cstdio>
#include <cstdint>
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
            std::printf("Usage: %s [--op all|shared|global|ptx|ptx_u4|ptx_r2u4|cublas|copy|copy_u8]\n"
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

__device__ __forceinline__ uint32_t ptx_ld_global_u32(void const* ptr)
{
    uint32_t value;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
}

__device__ __forceinline__ float ptx_fma_rn_f32(float a, float b, float c)
{
    float value;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(value) : "f"(a), "f"(b), "f"(c));
    return value;
}

__device__ __forceinline__ void ptx_st_global_f32(float* ptr, float value)
{
    asm volatile("st.global.f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

__device__ __forceinline__ half2 half2_from_u32(uint32_t packed)
{
    unsigned short lo = static_cast<unsigned short>(packed & 0xffffu);
    unsigned short hi = static_cast<unsigned short>(packed >> 16);
    return __halves2half2(__ushort_as_half(lo), __ushort_as_half(hi));
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

template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= n)
    {
        return;
    }

    int k2 = k / 2;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w2 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row) * k);

    float acc = 0.0f;
    for (int i = lane; i < k2; i += 32)
    {
        uint32_t h_pack = ptx_ld_global_u32(h2 + i);
        uint32_t w_pack = ptx_ld_global_u32(w2 + i);
        float2 h = __half22float2(half2_from_u32(h_pack));
        float2 w = __half22float2(half2_from_u32(w_pack));
        acc = ptx_fma_rn_f32(h.x, w.x, acc);
        acc = ptx_fma_rn_f32(h.y, w.y, acc);
    }

    // Qwen3.5 K=3072 is even. Keep a generic odd-K fallback outside the main path.
    if ((k & 1) && lane == 0)
    {
        int i = k - 1;
        acc = ptx_fma_rn_f32(__half2float(hidden[i]), __half2float(weight[static_cast<long long>(row) * k + i]), acc);
    }

    acc = warp_sum(acc);
    if (lane == 0)
    {
        ptx_st_global_f32(logits + row, acc);
    }
}

template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_u4_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= n)
    {
        return;
    }

    int k2 = k / 2;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w2 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row) * k);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    int i = lane;
    for (; i + 96 < k2; i += 128)
    {
        uint32_t h_pack0 = ptx_ld_global_u32(h2 + i);
        uint32_t w_pack0 = ptx_ld_global_u32(w2 + i);
        uint32_t h_pack1 = ptx_ld_global_u32(h2 + i + 32);
        uint32_t w_pack1 = ptx_ld_global_u32(w2 + i + 32);
        uint32_t h_pack2 = ptx_ld_global_u32(h2 + i + 64);
        uint32_t w_pack2 = ptx_ld_global_u32(w2 + i + 64);
        uint32_t h_pack3 = ptx_ld_global_u32(h2 + i + 96);
        uint32_t w_pack3 = ptx_ld_global_u32(w2 + i + 96);

        float2 h0 = __half22float2(half2_from_u32(h_pack0));
        float2 w0 = __half22float2(half2_from_u32(w_pack0));
        float2 h1 = __half22float2(half2_from_u32(h_pack1));
        float2 w1 = __half22float2(half2_from_u32(w_pack1));
        float2 h2v = __half22float2(half2_from_u32(h_pack2));
        float2 w2v = __half22float2(half2_from_u32(w_pack2));
        float2 h3 = __half22float2(half2_from_u32(h_pack3));
        float2 w3 = __half22float2(half2_from_u32(w_pack3));

        acc0 = ptx_fma_rn_f32(h0.x, w0.x, acc0);
        acc0 = ptx_fma_rn_f32(h0.y, w0.y, acc0);
        acc1 = ptx_fma_rn_f32(h1.x, w1.x, acc1);
        acc1 = ptx_fma_rn_f32(h1.y, w1.y, acc1);
        acc2 = ptx_fma_rn_f32(h2v.x, w2v.x, acc2);
        acc2 = ptx_fma_rn_f32(h2v.y, w2v.y, acc2);
        acc3 = ptx_fma_rn_f32(h3.x, w3.x, acc3);
        acc3 = ptx_fma_rn_f32(h3.y, w3.y, acc3);
    }

    float acc_tail = 0.0f;
    for (; i < k2; i += 32)
    {
        uint32_t h_pack = ptx_ld_global_u32(h2 + i);
        uint32_t w_pack = ptx_ld_global_u32(w2 + i);
        float2 h = __half22float2(half2_from_u32(h_pack));
        float2 w = __half22float2(half2_from_u32(w_pack));
        acc_tail = ptx_fma_rn_f32(h.x, w.x, acc_tail);
        acc_tail = ptx_fma_rn_f32(h.y, w.y, acc_tail);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3) + acc_tail;
    if ((k & 1) && lane == 0)
    {
        int kk = k - 1;
        acc = ptx_fma_rn_f32(__half2float(hidden[kk]), __half2float(weight[static_cast<long long>(row) * k + kk]), acc);
    }

    acc = warp_sum(acc);
    if (lane == 0)
    {
        ptx_st_global_f32(logits + row, acc);
    }
}

template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_r2u4_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row0 = (blockIdx.x * WarpsPerBlock + warp) * 2;
    if (row0 >= n)
    {
        return;
    }
    int row1 = row0 + 1;
    bool has_row1 = row1 < n;

    int k2 = k / 2;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w20 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row0) * k);
    half2 const* w21 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row1) * k);

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc02 = 0.0f;
    float acc03 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;

    int i = lane;
    for (; i + 96 < k2; i += 128)
    {
        uint32_t h_pack0 = ptx_ld_global_u32(h2 + i);
        uint32_t h_pack1 = ptx_ld_global_u32(h2 + i + 32);
        uint32_t h_pack2 = ptx_ld_global_u32(h2 + i + 64);
        uint32_t h_pack3 = ptx_ld_global_u32(h2 + i + 96);
        uint32_t w0_pack0 = ptx_ld_global_u32(w20 + i);
        uint32_t w0_pack1 = ptx_ld_global_u32(w20 + i + 32);
        uint32_t w0_pack2 = ptx_ld_global_u32(w20 + i + 64);
        uint32_t w0_pack3 = ptx_ld_global_u32(w20 + i + 96);

        float2 h0 = __half22float2(half2_from_u32(h_pack0));
        float2 h1 = __half22float2(half2_from_u32(h_pack1));
        float2 h2v = __half22float2(half2_from_u32(h_pack2));
        float2 h3 = __half22float2(half2_from_u32(h_pack3));
        float2 w00 = __half22float2(half2_from_u32(w0_pack0));
        float2 w01 = __half22float2(half2_from_u32(w0_pack1));
        float2 w02 = __half22float2(half2_from_u32(w0_pack2));
        float2 w03 = __half22float2(half2_from_u32(w0_pack3));

        acc00 = ptx_fma_rn_f32(h0.x, w00.x, acc00);
        acc00 = ptx_fma_rn_f32(h0.y, w00.y, acc00);
        acc01 = ptx_fma_rn_f32(h1.x, w01.x, acc01);
        acc01 = ptx_fma_rn_f32(h1.y, w01.y, acc01);
        acc02 = ptx_fma_rn_f32(h2v.x, w02.x, acc02);
        acc02 = ptx_fma_rn_f32(h2v.y, w02.y, acc02);
        acc03 = ptx_fma_rn_f32(h3.x, w03.x, acc03);
        acc03 = ptx_fma_rn_f32(h3.y, w03.y, acc03);

        if (has_row1)
        {
            uint32_t w1_pack0 = ptx_ld_global_u32(w21 + i);
            uint32_t w1_pack1 = ptx_ld_global_u32(w21 + i + 32);
            uint32_t w1_pack2 = ptx_ld_global_u32(w21 + i + 64);
            uint32_t w1_pack3 = ptx_ld_global_u32(w21 + i + 96);
            float2 w10 = __half22float2(half2_from_u32(w1_pack0));
            float2 w11 = __half22float2(half2_from_u32(w1_pack1));
            float2 w12 = __half22float2(half2_from_u32(w1_pack2));
            float2 w13 = __half22float2(half2_from_u32(w1_pack3));
            acc10 = ptx_fma_rn_f32(h0.x, w10.x, acc10);
            acc10 = ptx_fma_rn_f32(h0.y, w10.y, acc10);
            acc11 = ptx_fma_rn_f32(h1.x, w11.x, acc11);
            acc11 = ptx_fma_rn_f32(h1.y, w11.y, acc11);
            acc12 = ptx_fma_rn_f32(h2v.x, w12.x, acc12);
            acc12 = ptx_fma_rn_f32(h2v.y, w12.y, acc12);
            acc13 = ptx_fma_rn_f32(h3.x, w13.x, acc13);
            acc13 = ptx_fma_rn_f32(h3.y, w13.y, acc13);
        }
    }

    float tail0 = 0.0f;
    float tail1 = 0.0f;
    for (; i < k2; i += 32)
    {
        uint32_t h_pack = ptx_ld_global_u32(h2 + i);
        uint32_t w0_pack = ptx_ld_global_u32(w20 + i);
        float2 h = __half22float2(half2_from_u32(h_pack));
        float2 w0 = __half22float2(half2_from_u32(w0_pack));
        tail0 = ptx_fma_rn_f32(h.x, w0.x, tail0);
        tail0 = ptx_fma_rn_f32(h.y, w0.y, tail0);
        if (has_row1)
        {
            uint32_t w1_pack = ptx_ld_global_u32(w21 + i);
            float2 w1 = __half22float2(half2_from_u32(w1_pack));
            tail1 = ptx_fma_rn_f32(h.x, w1.x, tail1);
            tail1 = ptx_fma_rn_f32(h.y, w1.y, tail1);
        }
    }

    float acc0 = (acc00 + acc01) + (acc02 + acc03) + tail0;
    float acc1 = (acc10 + acc11) + (acc12 + acc13) + tail1;
    if ((k & 1) && lane == 0)
    {
        int kk = k - 1;
        float h = __half2float(hidden[kk]);
        acc0 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row0) * k + kk]), acc0);
        if (has_row1)
        {
            acc1 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row1) * k + kk]), acc1);
        }
    }

    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0)
    {
        ptx_st_global_f32(logits + row0, acc0);
        if (has_row1)
        {
            ptx_st_global_f32(logits + row1, acc1);
        }
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

template <int WarpsPerBlock>
float run_lm_head_ptx_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + WarpsPerBlock - 1) / WarpsPerBlock;
    return median_time_ms(
        [&] { lm_head_gemv_ptx_kernel<WarpsPerBlock><<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k); },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_u4_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + WarpsPerBlock - 1) / WarpsPerBlock;
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_u4_kernel<WarpsPerBlock>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_u4(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_u4_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_u4_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_u4_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_r2u4_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + 2 * WarpsPerBlock - 1) / (2 * WarpsPerBlock);
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_r2u4_kernel<WarpsPerBlock>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_r2u4(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_r2u4_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_r2u4_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_r2u4_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

void print_bw(char const* name, float ms, double bytes)
{
    double gbps = bytes / (ms * 1.0e-3) / 1.0e9;
    std::printf("%-14s median=%.4f ms traffic=%.3f MB bw=%.1f GB/s %.3f TB/s\n", name, ms, bytes / 1.0e6,
        gbps, gbps / 1000.0);
}

void run_gemv_cases(
    Options const& opt, bool run_shared, bool run_global, bool run_ptx, bool run_ptx_u4, bool run_ptx_r2u4)
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
    if (run_ptx)
    {
        float ms = run_lm_head_ptx(opt, d_hidden, d_weight, d_logits);
        CHECK_CUDA(cudaGetLastError());
        print_bw("ptx_global", ms, mandatory_traffic);
    }
    if (run_ptx_u4)
    {
        float ms = run_lm_head_ptx_u4(opt, d_hidden, d_weight, d_logits);
        CHECK_CUDA(cudaGetLastError());
        print_bw("ptx_u4", ms, mandatory_traffic);
    }
    if (run_ptx_r2u4)
    {
        float ms = run_lm_head_ptx_r2u4(opt, d_hidden, d_weight, d_logits);
        CHECK_CUDA(cudaGetLastError());
        print_bw("ptx_r2u4", ms, mandatory_traffic);
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
        run_gemv_cases(opt, true, true, true, true, true);
        run_cublas_case(opt);
        run_copy_case(opt);
    }
    else if (opt.op == "shared")
    {
        run_gemv_cases(opt, true, false, false, false, false);
    }
    else if (opt.op == "global")
    {
        run_gemv_cases(opt, false, true, false, false, false);
    }
    else if (opt.op == "ptx")
    {
        run_gemv_cases(opt, false, false, true, false, false);
    }
    else if (opt.op == "ptx_u4")
    {
        run_gemv_cases(opt, false, false, false, true, false);
    }
    else if (opt.op == "ptx_r2u4")
    {
        run_gemv_cases(opt, false, false, false, false, true);
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
