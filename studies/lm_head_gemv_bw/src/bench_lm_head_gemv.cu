#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
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
    int rows_per_warp = 1;
    int k_unroll = 4;
    int warmup = 100;
    int iters = 200;
    bool verify = true;
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
        else if (starts_with(argv[i], "--rows-per-warp="))
        {
            opt.rows_per_warp = std::atoi(argv[i] + 16);
        }
        else if (std::strcmp(argv[i], "--rows-per-warp") == 0)
        {
            opt.rows_per_warp = std::atoi(argv[++i]);
        }
        else if (starts_with(argv[i], "--k-unroll="))
        {
            opt.k_unroll = std::atoi(argv[i] + 11);
        }
        else if (std::strcmp(argv[i], "--k-unroll") == 0)
        {
            opt.k_unroll = std::atoi(argv[++i]);
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
        else if (std::strcmp(argv[i], "--no-verify") == 0)
        {
            opt.verify = false;
        }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("Usage: %s [--op all|shared|global|ptx|ptx_u4|ptx_r2u4|ptx_r2u8|ptx_chunk4|ptx_r2_chunk4|ptx_r2_chunk4u2|ptx_r2_chunk4u4|ptx_r2_chunk4u2_smem|ptx_cpasync|ptx_r4_chunk4|ptx_ru|ptx_rlean|ptx_persistent|ptx_tma_ws|cublas|copy|copy_u8]\n"
                        "          [--n vocab] [--k hidden] [--warps-per-block 4|8|16]\n"
                        "          [--rows-per-warp 1|2|4|8] [--k-unroll 4|8|16]\n"
                        "          [--warmup N] [--iters N] [--no-verify]\n",
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
    if (opt.rows_per_warp != 1 && opt.rows_per_warp != 2 && opt.rows_per_warp != 4 && opt.rows_per_warp != 8)
    {
        std::fprintf(stderr, "rows-per-warp must be 1, 2, 4, or 8\n");
        std::exit(1);
    }
    if (opt.k_unroll != 4 && opt.k_unroll != 8 && opt.k_unroll != 16)
    {
        std::fprintf(stderr, "k-unroll must be 4, 8, or 16\n");
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

__device__ __forceinline__ void ptx_ld_global_v4_u32(
    void const* ptr, uint32_t& x0, uint32_t& x1, uint32_t& x2, uint32_t& x3)
{
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                 : "l"(ptr));
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

// Outstanding-load study variant. Same 2-rows-per-warp mapping as ptx_r2u4, but
// the K loop is unrolled 8x and every packed load for the iteration is issued
// before any conversion or FMA. That keeps ~24 vector loads in flight per warp
// (8 hidden + 8 weight row0 + 8 weight row1) versus ~8 for ptx_r2u4. It targets
// backends whose block timeline shows the SM idle behind s.wait/vmcnt load
// waits: more outstanding loads give the memory subsystem more parallelism to
// hide DRAM latency. w21 is clamped to a valid row so the hot loop stays
// branch-free and all 24 loads are unconditional.
template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_r2u8_kernel(
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
    int safe_row1 = has_row1 ? row1 : row0;

    int k2 = k / 2;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w20 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row0) * k);
    half2 const* w21 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(safe_row1) * k);

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc02 = 0.0f;
    float acc03 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;

    int i = lane;
    for (; i + 32 * 7 < k2; i += 32 * 8)
    {
        // Issue all 24 packed loads up front: this load batching is the only
        // structural change versus ptx_r2u4 and is what widens the window.
        uint32_t h_pack0 = ptx_ld_global_u32(h2 + i);
        uint32_t h_pack1 = ptx_ld_global_u32(h2 + i + 32);
        uint32_t h_pack2 = ptx_ld_global_u32(h2 + i + 64);
        uint32_t h_pack3 = ptx_ld_global_u32(h2 + i + 96);
        uint32_t h_pack4 = ptx_ld_global_u32(h2 + i + 128);
        uint32_t h_pack5 = ptx_ld_global_u32(h2 + i + 160);
        uint32_t h_pack6 = ptx_ld_global_u32(h2 + i + 192);
        uint32_t h_pack7 = ptx_ld_global_u32(h2 + i + 224);

        uint32_t w0_pack0 = ptx_ld_global_u32(w20 + i);
        uint32_t w0_pack1 = ptx_ld_global_u32(w20 + i + 32);
        uint32_t w0_pack2 = ptx_ld_global_u32(w20 + i + 64);
        uint32_t w0_pack3 = ptx_ld_global_u32(w20 + i + 96);
        uint32_t w0_pack4 = ptx_ld_global_u32(w20 + i + 128);
        uint32_t w0_pack5 = ptx_ld_global_u32(w20 + i + 160);
        uint32_t w0_pack6 = ptx_ld_global_u32(w20 + i + 192);
        uint32_t w0_pack7 = ptx_ld_global_u32(w20 + i + 224);

        uint32_t w1_pack0 = ptx_ld_global_u32(w21 + i);
        uint32_t w1_pack1 = ptx_ld_global_u32(w21 + i + 32);
        uint32_t w1_pack2 = ptx_ld_global_u32(w21 + i + 64);
        uint32_t w1_pack3 = ptx_ld_global_u32(w21 + i + 96);
        uint32_t w1_pack4 = ptx_ld_global_u32(w21 + i + 128);
        uint32_t w1_pack5 = ptx_ld_global_u32(w21 + i + 160);
        uint32_t w1_pack6 = ptx_ld_global_u32(w21 + i + 192);
        uint32_t w1_pack7 = ptx_ld_global_u32(w21 + i + 224);

        // Convert each packed value just before use so f32 temporaries do not
        // pin extra registers; 4 accumulators per row absorb the FMA chain.
        float2 h = __half22float2(half2_from_u32(h_pack0));
        float2 a = __half22float2(half2_from_u32(w0_pack0));
        float2 b = __half22float2(half2_from_u32(w1_pack0));
        acc00 = ptx_fma_rn_f32(h.x, a.x, acc00);
        acc00 = ptx_fma_rn_f32(h.y, a.y, acc00);
        acc10 = ptx_fma_rn_f32(h.x, b.x, acc10);
        acc10 = ptx_fma_rn_f32(h.y, b.y, acc10);

        h = __half22float2(half2_from_u32(h_pack1));
        a = __half22float2(half2_from_u32(w0_pack1));
        b = __half22float2(half2_from_u32(w1_pack1));
        acc01 = ptx_fma_rn_f32(h.x, a.x, acc01);
        acc01 = ptx_fma_rn_f32(h.y, a.y, acc01);
        acc11 = ptx_fma_rn_f32(h.x, b.x, acc11);
        acc11 = ptx_fma_rn_f32(h.y, b.y, acc11);

        h = __half22float2(half2_from_u32(h_pack2));
        a = __half22float2(half2_from_u32(w0_pack2));
        b = __half22float2(half2_from_u32(w1_pack2));
        acc02 = ptx_fma_rn_f32(h.x, a.x, acc02);
        acc02 = ptx_fma_rn_f32(h.y, a.y, acc02);
        acc12 = ptx_fma_rn_f32(h.x, b.x, acc12);
        acc12 = ptx_fma_rn_f32(h.y, b.y, acc12);

        h = __half22float2(half2_from_u32(h_pack3));
        a = __half22float2(half2_from_u32(w0_pack3));
        b = __half22float2(half2_from_u32(w1_pack3));
        acc03 = ptx_fma_rn_f32(h.x, a.x, acc03);
        acc03 = ptx_fma_rn_f32(h.y, a.y, acc03);
        acc13 = ptx_fma_rn_f32(h.x, b.x, acc13);
        acc13 = ptx_fma_rn_f32(h.y, b.y, acc13);

        h = __half22float2(half2_from_u32(h_pack4));
        a = __half22float2(half2_from_u32(w0_pack4));
        b = __half22float2(half2_from_u32(w1_pack4));
        acc00 = ptx_fma_rn_f32(h.x, a.x, acc00);
        acc00 = ptx_fma_rn_f32(h.y, a.y, acc00);
        acc10 = ptx_fma_rn_f32(h.x, b.x, acc10);
        acc10 = ptx_fma_rn_f32(h.y, b.y, acc10);

        h = __half22float2(half2_from_u32(h_pack5));
        a = __half22float2(half2_from_u32(w0_pack5));
        b = __half22float2(half2_from_u32(w1_pack5));
        acc01 = ptx_fma_rn_f32(h.x, a.x, acc01);
        acc01 = ptx_fma_rn_f32(h.y, a.y, acc01);
        acc11 = ptx_fma_rn_f32(h.x, b.x, acc11);
        acc11 = ptx_fma_rn_f32(h.y, b.y, acc11);

        h = __half22float2(half2_from_u32(h_pack6));
        a = __half22float2(half2_from_u32(w0_pack6));
        b = __half22float2(half2_from_u32(w1_pack6));
        acc02 = ptx_fma_rn_f32(h.x, a.x, acc02);
        acc02 = ptx_fma_rn_f32(h.y, a.y, acc02);
        acc12 = ptx_fma_rn_f32(h.x, b.x, acc12);
        acc12 = ptx_fma_rn_f32(h.y, b.y, acc12);

        h = __half22float2(half2_from_u32(h_pack7));
        a = __half22float2(half2_from_u32(w0_pack7));
        b = __half22float2(half2_from_u32(w1_pack7));
        acc03 = ptx_fma_rn_f32(h.x, a.x, acc03);
        acc03 = ptx_fma_rn_f32(h.y, a.y, acc03);
        acc13 = ptx_fma_rn_f32(h.x, b.x, acc13);
        acc13 = ptx_fma_rn_f32(h.y, b.y, acc13);
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

template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_chunk4_kernel(
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
    int full_k2 = (k2 / 128) * 128;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w2 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row) * k);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int tile = 0; tile < full_k2; tile += 128)
    {
        int base = tile + lane * 4;
        uint32_t h_pack0;
        uint32_t h_pack1;
        uint32_t h_pack2;
        uint32_t h_pack3;
        uint32_t w_pack0;
        uint32_t w_pack1;
        uint32_t w_pack2;
        uint32_t w_pack3;
        ptx_ld_global_v4_u32(h2 + base, h_pack0, h_pack1, h_pack2, h_pack3);
        ptx_ld_global_v4_u32(w2 + base, w_pack0, w_pack1, w_pack2, w_pack3);

        float2 h0 = __half22float2(half2_from_u32(h_pack0));
        float2 h1 = __half22float2(half2_from_u32(h_pack1));
        float2 h2v = __half22float2(half2_from_u32(h_pack2));
        float2 h3 = __half22float2(half2_from_u32(h_pack3));
        float2 w0 = __half22float2(half2_from_u32(w_pack0));
        float2 w1 = __half22float2(half2_from_u32(w_pack1));
        float2 w2v = __half22float2(half2_from_u32(w_pack2));
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

    float tail = 0.0f;
    for (int i = full_k2 + lane; i < k2; i += 32)
    {
        uint32_t h_pack = ptx_ld_global_u32(h2 + i);
        uint32_t w_pack = ptx_ld_global_u32(w2 + i);
        float2 h = __half22float2(half2_from_u32(h_pack));
        float2 w = __half22float2(half2_from_u32(w_pack));
        tail = ptx_fma_rn_f32(h.x, w.x, tail);
        tail = ptx_fma_rn_f32(h.y, w.y, tail);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3) + tail;
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
__global__ void lm_head_gemv_ptx_r4_chunk4_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row0 = (blockIdx.x * WarpsPerBlock + warp) * 4;
    if (row0 >= n)
    {
        return;
    }
    int row1 = row0 + 1;
    int row2 = row0 + 2;
    int row3 = row0 + 3;
    bool has_row1 = row1 < n;
    bool has_row2 = row2 < n;
    bool has_row3 = row3 < n;

    int k2 = k / 2;
    int full_k2 = (k2 / 128) * 128;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w20 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row0) * k);
    half2 const* w21 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row1) * k);
    half2 const* w22 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row2) * k);
    half2 const* w23 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row3) * k);

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int tile = 0; tile < full_k2; tile += 128)
    {
        int base = tile + lane * 4;
        uint32_t h_pack0, h_pack1, h_pack2, h_pack3;
        uint32_t w0_pack0, w0_pack1, w0_pack2, w0_pack3;
        ptx_ld_global_v4_u32(h2 + base, h_pack0, h_pack1, h_pack2, h_pack3);
        ptx_ld_global_v4_u32(w20 + base, w0_pack0, w0_pack1, w0_pack2, w0_pack3);

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
            uint32_t w_pack0, w_pack1, w_pack2, w_pack3;
            ptx_ld_global_v4_u32(w21 + base, w_pack0, w_pack1, w_pack2, w_pack3);
            float2 w10 = __half22float2(half2_from_u32(w_pack0));
            float2 w11 = __half22float2(half2_from_u32(w_pack1));
            float2 w12 = __half22float2(half2_from_u32(w_pack2));
            float2 w13 = __half22float2(half2_from_u32(w_pack3));
            acc10 = ptx_fma_rn_f32(h0.x, w10.x, acc10);
            acc10 = ptx_fma_rn_f32(h0.y, w10.y, acc10);
            acc11 = ptx_fma_rn_f32(h1.x, w11.x, acc11);
            acc11 = ptx_fma_rn_f32(h1.y, w11.y, acc11);
            acc12 = ptx_fma_rn_f32(h2v.x, w12.x, acc12);
            acc12 = ptx_fma_rn_f32(h2v.y, w12.y, acc12);
            acc13 = ptx_fma_rn_f32(h3.x, w13.x, acc13);
            acc13 = ptx_fma_rn_f32(h3.y, w13.y, acc13);
        }
        if (has_row2)
        {
            uint32_t w_pack0, w_pack1, w_pack2, w_pack3;
            ptx_ld_global_v4_u32(w22 + base, w_pack0, w_pack1, w_pack2, w_pack3);
            float2 w20 = __half22float2(half2_from_u32(w_pack0));
            float2 w21 = __half22float2(half2_from_u32(w_pack1));
            float2 w22 = __half22float2(half2_from_u32(w_pack2));
            float2 w23 = __half22float2(half2_from_u32(w_pack3));
            acc20 = ptx_fma_rn_f32(h0.x, w20.x, acc20);
            acc20 = ptx_fma_rn_f32(h0.y, w20.y, acc20);
            acc21 = ptx_fma_rn_f32(h1.x, w21.x, acc21);
            acc21 = ptx_fma_rn_f32(h1.y, w21.y, acc21);
            acc22 = ptx_fma_rn_f32(h2v.x, w22.x, acc22);
            acc22 = ptx_fma_rn_f32(h2v.y, w22.y, acc22);
            acc23 = ptx_fma_rn_f32(h3.x, w23.x, acc23);
            acc23 = ptx_fma_rn_f32(h3.y, w23.y, acc23);
        }
        if (has_row3)
        {
            uint32_t w_pack0, w_pack1, w_pack2, w_pack3;
            ptx_ld_global_v4_u32(w23 + base, w_pack0, w_pack1, w_pack2, w_pack3);
            float2 w30 = __half22float2(half2_from_u32(w_pack0));
            float2 w31 = __half22float2(half2_from_u32(w_pack1));
            float2 w32 = __half22float2(half2_from_u32(w_pack2));
            float2 w33 = __half22float2(half2_from_u32(w_pack3));
            acc30 = ptx_fma_rn_f32(h0.x, w30.x, acc30);
            acc30 = ptx_fma_rn_f32(h0.y, w30.y, acc30);
            acc31 = ptx_fma_rn_f32(h1.x, w31.x, acc31);
            acc31 = ptx_fma_rn_f32(h1.y, w31.y, acc31);
            acc32 = ptx_fma_rn_f32(h2v.x, w32.x, acc32);
            acc32 = ptx_fma_rn_f32(h2v.y, w32.y, acc32);
            acc33 = ptx_fma_rn_f32(h3.x, w33.x, acc33);
            acc33 = ptx_fma_rn_f32(h3.y, w33.y, acc33);
        }
    }

    float tail0 = 0.0f;
    float tail1 = 0.0f;
    float tail2 = 0.0f;
    float tail3 = 0.0f;
    for (int i = full_k2 + lane; i < k2; i += 32)
    {
        uint32_t h_pack = ptx_ld_global_u32(h2 + i);
        uint32_t w_pack = ptx_ld_global_u32(w20 + i);
        float2 h = __half22float2(half2_from_u32(h_pack));
        float2 w = __half22float2(half2_from_u32(w_pack));
        tail0 = ptx_fma_rn_f32(h.x, w.x, tail0);
        tail0 = ptx_fma_rn_f32(h.y, w.y, tail0);
        if (has_row1)
        {
            w_pack = ptx_ld_global_u32(w21 + i);
            w = __half22float2(half2_from_u32(w_pack));
            tail1 = ptx_fma_rn_f32(h.x, w.x, tail1);
            tail1 = ptx_fma_rn_f32(h.y, w.y, tail1);
        }
        if (has_row2)
        {
            w_pack = ptx_ld_global_u32(w22 + i);
            w = __half22float2(half2_from_u32(w_pack));
            tail2 = ptx_fma_rn_f32(h.x, w.x, tail2);
            tail2 = ptx_fma_rn_f32(h.y, w.y, tail2);
        }
        if (has_row3)
        {
            w_pack = ptx_ld_global_u32(w23 + i);
            w = __half22float2(half2_from_u32(w_pack));
            tail3 = ptx_fma_rn_f32(h.x, w.x, tail3);
            tail3 = ptx_fma_rn_f32(h.y, w.y, tail3);
        }
    }

    float out0 = (acc00 + acc01) + (acc02 + acc03) + tail0;
    float out1 = (acc10 + acc11) + (acc12 + acc13) + tail1;
    float out2 = (acc20 + acc21) + (acc22 + acc23) + tail2;
    float out3 = (acc30 + acc31) + (acc32 + acc33) + tail3;
    if ((k & 1) && lane == 0)
    {
        int kk = k - 1;
        float h = __half2float(hidden[kk]);
        out0 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row0) * k + kk]), out0);
        if (has_row1)
        {
            out1 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row1) * k + kk]), out1);
        }
        if (has_row2)
        {
            out2 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row2) * k + kk]), out2);
        }
        if (has_row3)
        {
            out3 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row3) * k + kk]), out3);
        }
    }

    out0 = warp_sum(out0);
    out1 = warp_sum(out1);
    out2 = warp_sum(out2);
    out3 = warp_sum(out3);
    if (lane == 0)
    {
        ptx_st_global_f32(logits + row0, out0);
        if (has_row1)
        {
            ptx_st_global_f32(logits + row1, out1);
        }
        if (has_row2)
        {
            ptx_st_global_f32(logits + row2, out2);
        }
        if (has_row3)
        {
            ptx_st_global_f32(logits + row3, out3);
        }
    }
}

template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_r2_chunk4_kernel(
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
    int full_k2 = (k2 / 128) * 128;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w20 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row0) * k);
    half2 const* w21 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row1) * k);

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;

    for (int tile = 0; tile < full_k2; tile += 128)
    {
        int base = tile + lane * 4;
        uint32_t h_pack0, h_pack1, h_pack2, h_pack3;
        uint32_t w0_pack0, w0_pack1, w0_pack2, w0_pack3;
        ptx_ld_global_v4_u32(h2 + base, h_pack0, h_pack1, h_pack2, h_pack3);
        ptx_ld_global_v4_u32(w20 + base, w0_pack0, w0_pack1, w0_pack2, w0_pack3);

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
            uint32_t w1_pack0, w1_pack1, w1_pack2, w1_pack3;
            ptx_ld_global_v4_u32(w21 + base, w1_pack0, w1_pack1, w1_pack2, w1_pack3);
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
    for (int i = full_k2 + lane; i < k2; i += 32)
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

    float out0 = (acc00 + acc01) + (acc02 + acc03) + tail0;
    float out1 = (acc10 + acc11) + (acc12 + acc13) + tail1;
    if ((k & 1) && lane == 0)
    {
        int kk = k - 1;
        float h = __half2float(hidden[kk]);
        out0 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row0) * k + kk]), out0);
        if (has_row1)
        {
            out1 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row1) * k + kk]), out1);
        }
    }

    out0 = warp_sum(out0);
    out1 = warp_sum(out1);
    if (lane == 0)
    {
        ptx_st_global_f32(logits + row0, out0);
        if (has_row1)
        {
            ptx_st_global_f32(logits + row1, out1);
        }
    }
}

// Outstanding-load study variant, wide-load version. Combines the v4 (128-bit)
// loads of ptx_r2_chunk4 with the wide outstanding window of ptx_r2u8: the K
// loop processes 2 chunk4 tiles per step and issues all 6 v4 loads (2 hidden +
// 2 weight row0 + 2 weight row1) before any conversion or FMA. Each lane keeps
// 96 bytes per row pair in flight -- the same bytes as ptx_r2u8, but via 6 wide
// loads instead of 24 narrow ones. Comparing the three isolates the two levers:
// ptx_r2_chunk4 (wide, shallow) vs ptx_r2u8 (narrow, deep) vs this (wide, deep).
template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_r2_chunk4u2_kernel(
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
    int safe_row1 = has_row1 ? row1 : row0;

    int k2 = k / 2;
    int full_k2 = (k2 / 256) * 256;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w20 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row0) * k);
    half2 const* w21 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(safe_row1) * k);

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc02 = 0.0f;
    float acc03 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;

    for (int step = 0; step < full_k2; step += 256)
    {
        int base0 = step + lane * 4;
        int base1 = step + 128 + lane * 4;

        // Issue all 6 v4 loads before any conversion or FMA. w21 is clamped to a
        // valid row so every load is unconditional and the hot loop has no branch.
        uint32_t h0_0, h0_1, h0_2, h0_3;
        uint32_t h1_0, h1_1, h1_2, h1_3;
        uint32_t a0_0, a0_1, a0_2, a0_3;
        uint32_t a1_0, a1_1, a1_2, a1_3;
        uint32_t b0_0, b0_1, b0_2, b0_3;
        uint32_t b1_0, b1_1, b1_2, b1_3;
        ptx_ld_global_v4_u32(h2 + base0, h0_0, h0_1, h0_2, h0_3);
        ptx_ld_global_v4_u32(h2 + base1, h1_0, h1_1, h1_2, h1_3);
        ptx_ld_global_v4_u32(w20 + base0, a0_0, a0_1, a0_2, a0_3);
        ptx_ld_global_v4_u32(w20 + base1, a1_0, a1_1, a1_2, a1_3);
        ptx_ld_global_v4_u32(w21 + base0, b0_0, b0_1, b0_2, b0_3);
        ptx_ld_global_v4_u32(w21 + base1, b1_0, b1_1, b1_2, b1_3);

        float2 h = __half22float2(half2_from_u32(h0_0));
        float2 a = __half22float2(half2_from_u32(a0_0));
        float2 b = __half22float2(half2_from_u32(b0_0));
        acc00 = ptx_fma_rn_f32(h.x, a.x, acc00);
        acc00 = ptx_fma_rn_f32(h.y, a.y, acc00);
        acc10 = ptx_fma_rn_f32(h.x, b.x, acc10);
        acc10 = ptx_fma_rn_f32(h.y, b.y, acc10);

        h = __half22float2(half2_from_u32(h0_1));
        a = __half22float2(half2_from_u32(a0_1));
        b = __half22float2(half2_from_u32(b0_1));
        acc01 = ptx_fma_rn_f32(h.x, a.x, acc01);
        acc01 = ptx_fma_rn_f32(h.y, a.y, acc01);
        acc11 = ptx_fma_rn_f32(h.x, b.x, acc11);
        acc11 = ptx_fma_rn_f32(h.y, b.y, acc11);

        h = __half22float2(half2_from_u32(h0_2));
        a = __half22float2(half2_from_u32(a0_2));
        b = __half22float2(half2_from_u32(b0_2));
        acc02 = ptx_fma_rn_f32(h.x, a.x, acc02);
        acc02 = ptx_fma_rn_f32(h.y, a.y, acc02);
        acc12 = ptx_fma_rn_f32(h.x, b.x, acc12);
        acc12 = ptx_fma_rn_f32(h.y, b.y, acc12);

        h = __half22float2(half2_from_u32(h0_3));
        a = __half22float2(half2_from_u32(a0_3));
        b = __half22float2(half2_from_u32(b0_3));
        acc03 = ptx_fma_rn_f32(h.x, a.x, acc03);
        acc03 = ptx_fma_rn_f32(h.y, a.y, acc03);
        acc13 = ptx_fma_rn_f32(h.x, b.x, acc13);
        acc13 = ptx_fma_rn_f32(h.y, b.y, acc13);

        h = __half22float2(half2_from_u32(h1_0));
        a = __half22float2(half2_from_u32(a1_0));
        b = __half22float2(half2_from_u32(b1_0));
        acc00 = ptx_fma_rn_f32(h.x, a.x, acc00);
        acc00 = ptx_fma_rn_f32(h.y, a.y, acc00);
        acc10 = ptx_fma_rn_f32(h.x, b.x, acc10);
        acc10 = ptx_fma_rn_f32(h.y, b.y, acc10);

        h = __half22float2(half2_from_u32(h1_1));
        a = __half22float2(half2_from_u32(a1_1));
        b = __half22float2(half2_from_u32(b1_1));
        acc01 = ptx_fma_rn_f32(h.x, a.x, acc01);
        acc01 = ptx_fma_rn_f32(h.y, a.y, acc01);
        acc11 = ptx_fma_rn_f32(h.x, b.x, acc11);
        acc11 = ptx_fma_rn_f32(h.y, b.y, acc11);

        h = __half22float2(half2_from_u32(h1_2));
        a = __half22float2(half2_from_u32(a1_2));
        b = __half22float2(half2_from_u32(b1_2));
        acc02 = ptx_fma_rn_f32(h.x, a.x, acc02);
        acc02 = ptx_fma_rn_f32(h.y, a.y, acc02);
        acc12 = ptx_fma_rn_f32(h.x, b.x, acc12);
        acc12 = ptx_fma_rn_f32(h.y, b.y, acc12);

        h = __half22float2(half2_from_u32(h1_3));
        a = __half22float2(half2_from_u32(a1_3));
        b = __half22float2(half2_from_u32(b1_3));
        acc03 = ptx_fma_rn_f32(h.x, a.x, acc03);
        acc03 = ptx_fma_rn_f32(h.y, a.y, acc03);
        acc13 = ptx_fma_rn_f32(h.x, b.x, acc13);
        acc13 = ptx_fma_rn_f32(h.y, b.y, acc13);
    }

    float tail0 = 0.0f;
    float tail1 = 0.0f;
    for (int i = full_k2 + lane; i < k2; i += 32)
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

// Configurable-depth wide-load variant. Same structure as ptx_r2_chunk4u2, but
// the number of chunk4 tiles batched per K step is a template parameter: Tiles
// chunk4 tiles == 3*Tiles v4 (128-bit) loads issued before any FMA == 64*Tiles
// bytes/lane in flight per row pair. Used to push the outstanding window past
// ptx_r2u8 / ptx_r2_chunk4u2 on backends that can sustain more in-flight loads.
// ptx_r2_chunk4u4 instantiates this at Tiles=4 (12 v4 loads, 192 bytes/lane).
template <int WarpsPerBlock, int Tiles>
__global__ void lm_head_gemv_ptx_r2_chunk4u_kernel(
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
    int safe_row1 = has_row1 ? row1 : row0;

    int k2 = k / 2;
    int step_half2 = 128 * Tiles;
    int full_k2 = (k2 / step_half2) * step_half2;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w20 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row0) * k);
    half2 const* w21 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(safe_row1) * k);

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int step = 0; step < full_k2; step += step_half2)
    {
        uint32_t hp[Tiles][4];
        uint32_t ap[Tiles][4];
        uint32_t bp[Tiles][4];

        // Issue all 3*Tiles v4 loads before any conversion or FMA. w21 is clamped
        // to a valid row so every load is unconditional and the loop is branch-free.
#pragma unroll
        for (int t = 0; t < Tiles; ++t)
        {
            int base = step + t * 128 + lane * 4;
            ptx_ld_global_v4_u32(h2 + base, hp[t][0], hp[t][1], hp[t][2], hp[t][3]);
        }
#pragma unroll
        for (int t = 0; t < Tiles; ++t)
        {
            int base = step + t * 128 + lane * 4;
            ptx_ld_global_v4_u32(w20 + base, ap[t][0], ap[t][1], ap[t][2], ap[t][3]);
        }
#pragma unroll
        for (int t = 0; t < Tiles; ++t)
        {
            int base = step + t * 128 + lane * 4;
            ptx_ld_global_v4_u32(w21 + base, bp[t][0], bp[t][1], bp[t][2], bp[t][3]);
        }

#pragma unroll
        for (int t = 0; t < Tiles; ++t)
        {
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                float2 h = __half22float2(half2_from_u32(hp[t][c]));
                float2 a = __half22float2(half2_from_u32(ap[t][c]));
                float2 b = __half22float2(half2_from_u32(bp[t][c]));
                acc0[c] = ptx_fma_rn_f32(h.x, a.x, acc0[c]);
                acc0[c] = ptx_fma_rn_f32(h.y, a.y, acc0[c]);
                acc1[c] = ptx_fma_rn_f32(h.x, b.x, acc1[c]);
                acc1[c] = ptx_fma_rn_f32(h.y, b.y, acc1[c]);
            }
        }
    }

    float tail0 = 0.0f;
    float tail1 = 0.0f;
    for (int i = full_k2 + lane; i < k2; i += 32)
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

    float r0 = (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]) + tail0;
    float r1 = (acc1[0] + acc1[1]) + (acc1[2] + acc1[3]) + tail1;
    if ((k & 1) && lane == 0)
    {
        int kk = k - 1;
        float h = __half2float(hidden[kk]);
        r0 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row0) * k + kk]), r0);
        if (has_row1)
        {
            r1 = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row1) * k + kk]), r1);
        }
    }

    r0 = warp_sum(r0);
    r1 = warp_sum(r1);
    if (lane == 0)
    {
        ptx_st_global_f32(logits + row0, r0);
        if (has_row1)
        {
            ptx_st_global_f32(logits + row1, r1);
        }
    }
}

// Clean isolation test for the redundant-hidden-read hypothesis. Identical to
// ptx_r2_chunk4u2 (2 rows/warp, 6-v4 outstanding window, inline-PTX weight
// loads) except the hidden vector is cooperatively staged into shared memory
// once per block; the inner loop then reads hidden from smem and only the 4
// weight v4 loads hit global. If a backend does not cache the 6 KB hidden
// vector, every warp re-reads it from DRAM (~763 MB total for this shape);
// staging it in smem cuts that to one read per block. Comparing this against
// ptx_r2_chunk4u2 changes only the hidden path, nothing else.
template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_r2_chunk4u2_smem_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    extern __shared__ half s_hidden[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    // Stage hidden into shared memory before the row-range check so every
    // thread in the block reaches __syncthreads().
    for (int i = tid; i < k; i += WarpsPerBlock * 32)
    {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int row0 = (blockIdx.x * WarpsPerBlock + warp) * 2;
    if (row0 >= n)
    {
        return;
    }
    int row1 = row0 + 1;
    bool has_row1 = row1 < n;
    int safe_row1 = has_row1 ? row1 : row0;

    int k2 = k / 2;
    int full_k2 = (k2 / 256) * 256;
    half2 const* sh2 = reinterpret_cast<half2 const*>(s_hidden);
    half2 const* w20 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row0) * k);
    half2 const* w21 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(safe_row1) * k);

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc02 = 0.0f;
    float acc03 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;

    for (int step = 0; step < full_k2; step += 256)
    {
        int base0 = step + lane * 4;
        int base1 = step + 128 + lane * 4;

        // hidden from shared memory; only the 4 weight v4 loads hit global.
        uint4 hv0 = *reinterpret_cast<uint4 const*>(sh2 + base0);
        uint4 hv1 = *reinterpret_cast<uint4 const*>(sh2 + base1);
        uint32_t a0_0, a0_1, a0_2, a0_3;
        uint32_t a1_0, a1_1, a1_2, a1_3;
        uint32_t b0_0, b0_1, b0_2, b0_3;
        uint32_t b1_0, b1_1, b1_2, b1_3;
        ptx_ld_global_v4_u32(w20 + base0, a0_0, a0_1, a0_2, a0_3);
        ptx_ld_global_v4_u32(w20 + base1, a1_0, a1_1, a1_2, a1_3);
        ptx_ld_global_v4_u32(w21 + base0, b0_0, b0_1, b0_2, b0_3);
        ptx_ld_global_v4_u32(w21 + base1, b1_0, b1_1, b1_2, b1_3);

        float2 h = __half22float2(half2_from_u32(hv0.x));
        float2 a = __half22float2(half2_from_u32(a0_0));
        float2 b = __half22float2(half2_from_u32(b0_0));
        acc00 = ptx_fma_rn_f32(h.x, a.x, acc00);
        acc00 = ptx_fma_rn_f32(h.y, a.y, acc00);
        acc10 = ptx_fma_rn_f32(h.x, b.x, acc10);
        acc10 = ptx_fma_rn_f32(h.y, b.y, acc10);

        h = __half22float2(half2_from_u32(hv0.y));
        a = __half22float2(half2_from_u32(a0_1));
        b = __half22float2(half2_from_u32(b0_1));
        acc01 = ptx_fma_rn_f32(h.x, a.x, acc01);
        acc01 = ptx_fma_rn_f32(h.y, a.y, acc01);
        acc11 = ptx_fma_rn_f32(h.x, b.x, acc11);
        acc11 = ptx_fma_rn_f32(h.y, b.y, acc11);

        h = __half22float2(half2_from_u32(hv0.z));
        a = __half22float2(half2_from_u32(a0_2));
        b = __half22float2(half2_from_u32(b0_2));
        acc02 = ptx_fma_rn_f32(h.x, a.x, acc02);
        acc02 = ptx_fma_rn_f32(h.y, a.y, acc02);
        acc12 = ptx_fma_rn_f32(h.x, b.x, acc12);
        acc12 = ptx_fma_rn_f32(h.y, b.y, acc12);

        h = __half22float2(half2_from_u32(hv0.w));
        a = __half22float2(half2_from_u32(a0_3));
        b = __half22float2(half2_from_u32(b0_3));
        acc03 = ptx_fma_rn_f32(h.x, a.x, acc03);
        acc03 = ptx_fma_rn_f32(h.y, a.y, acc03);
        acc13 = ptx_fma_rn_f32(h.x, b.x, acc13);
        acc13 = ptx_fma_rn_f32(h.y, b.y, acc13);

        h = __half22float2(half2_from_u32(hv1.x));
        a = __half22float2(half2_from_u32(a1_0));
        b = __half22float2(half2_from_u32(b1_0));
        acc00 = ptx_fma_rn_f32(h.x, a.x, acc00);
        acc00 = ptx_fma_rn_f32(h.y, a.y, acc00);
        acc10 = ptx_fma_rn_f32(h.x, b.x, acc10);
        acc10 = ptx_fma_rn_f32(h.y, b.y, acc10);

        h = __half22float2(half2_from_u32(hv1.y));
        a = __half22float2(half2_from_u32(a1_1));
        b = __half22float2(half2_from_u32(b1_1));
        acc01 = ptx_fma_rn_f32(h.x, a.x, acc01);
        acc01 = ptx_fma_rn_f32(h.y, a.y, acc01);
        acc11 = ptx_fma_rn_f32(h.x, b.x, acc11);
        acc11 = ptx_fma_rn_f32(h.y, b.y, acc11);

        h = __half22float2(half2_from_u32(hv1.z));
        a = __half22float2(half2_from_u32(a1_2));
        b = __half22float2(half2_from_u32(b1_2));
        acc02 = ptx_fma_rn_f32(h.x, a.x, acc02);
        acc02 = ptx_fma_rn_f32(h.y, a.y, acc02);
        acc12 = ptx_fma_rn_f32(h.x, b.x, acc12);
        acc12 = ptx_fma_rn_f32(h.y, b.y, acc12);

        h = __half22float2(half2_from_u32(hv1.w));
        a = __half22float2(half2_from_u32(a1_3));
        b = __half22float2(half2_from_u32(b1_3));
        acc03 = ptx_fma_rn_f32(h.x, a.x, acc03);
        acc03 = ptx_fma_rn_f32(h.y, a.y, acc03);
        acc13 = ptx_fma_rn_f32(h.x, b.x, acc13);
        acc13 = ptx_fma_rn_f32(h.y, b.y, acc13);
    }

    float tail0 = 0.0f;
    float tail1 = 0.0f;
    for (int i = full_k2 + lane; i < k2; i += 32)
    {
        float2 h = __half22float2(sh2[i]);
        uint32_t w0_pack = ptx_ld_global_u32(w20 + i);
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
        float h = __half2float(s_hidden[kk]);
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

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, void const* global_ptr)
{
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(global_ptr));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// cp.async-prefetched variant. The structural change vs every other variant:
// the weight stream is copied global -> shared memory with cp.async, so the
// in-flight load data lands in shared memory instead of registers. That
// decouples the outstanding-load window depth from register pressure -- the
// register-bound ceiling that caps the ptx_r2u8 / ptx_r2_chunk4u* family. The
// kernel keeps very few registers (one accumulator + addressing), so occupancy
// is high, while a Stages-deep software pipeline keeps Stages-1 weight tiles in
// flight. Hidden is staged in shared memory once per block; each warp computes
// one output row over 128-half2 weight tiles.
template <int WarpsPerBlock, int Stages>
__global__ void lm_head_gemv_ptx_cpasync_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    extern __shared__ char cpasync_smem[];
    half* s_hidden = reinterpret_cast<half*>(cpasync_smem);
    int k2 = k / 2;
    // Weight ring buffer starts 16-byte aligned after the staged hidden vector.
    int ring_off = ((k * static_cast<int>(sizeof(half)) + 15) / 16) * 16;
    half2* s_wbuf = reinterpret_cast<half2*>(cpasync_smem + ring_off);

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    for (int i = tid; i < k; i += WarpsPerBlock * 32)
    {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= n)
    {
        return;
    }

    half2 const* h2 = reinterpret_cast<half2 const*>(s_hidden);
    half2 const* w2 = reinterpret_cast<half2 const*>(weight + static_cast<long long>(row) * k);
    half2* my_ring = s_wbuf + warp * (Stages * 128);

    int full_tiles = k2 / 128;
    float acc = 0.0f;

    // Prologue: issue the first Stages-1 tile copies.
#pragma unroll
    for (int s = 0; s < Stages - 1; ++s)
    {
        if (s < full_tiles)
        {
            cp_async_cg_16(&my_ring[s * 128 + lane * 4], &w2[s * 128 + lane * 4]);
        }
        cp_async_commit();
    }

    for (int t = 0; t < full_tiles; ++t)
    {
        cp_async_wait<Stages - 2>();
        __syncwarp();

        int pre = t + Stages - 1;
        if (pre < full_tiles)
        {
            int slot = pre % Stages;
            cp_async_cg_16(&my_ring[slot * 128 + lane * 4], &w2[pre * 128 + lane * 4]);
        }
        cp_async_commit();

        half2 const* wt = my_ring + (t % Stages) * 128;
        half2 const* ht = h2 + t * 128;
#pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            float2 hv = __half22float2(ht[lane + 32 * u]);
            float2 wv = __half22float2(wt[lane + 32 * u]);
            acc = ptx_fma_rn_f32(hv.x, wv.x, acc);
            acc = ptx_fma_rn_f32(hv.y, wv.y, acc);
        }
    }
    cp_async_wait<0>();

    for (int i = full_tiles * 128 + lane; i < k2; i += 32)
    {
        float2 hv = __half22float2(h2[i]);
        uint32_t w_pack = ptx_ld_global_u32(w2 + i);
        float2 wv = __half22float2(half2_from_u32(w_pack));
        acc = ptx_fma_rn_f32(hv.x, wv.x, acc);
        acc = ptx_fma_rn_f32(hv.y, wv.y, acc);
    }
    if ((k & 1) && lane == 0)
    {
        int kk = k - 1;
        acc = ptx_fma_rn_f32(
            __half2float(s_hidden[kk]), __half2float(weight[static_cast<long long>(row) * k + kk]), acc);
    }

    acc = warp_sum(acc);
    if (lane == 0)
    {
        ptx_st_global_f32(logits + row, acc);
    }
}

// Register-lean many-rows variant. Same all-loads-first structure as ptx_ru,
// but ONE accumulator per row instead of RowsPerWarp*KUnroll. The KUnroll
// hidden-load registers are fixed per-warp overhead; more rows amortize them
// across more weight streams, so weight-loads-per-register -- which bounds the
// chip-wide in-flight bytes of a register-resident kernel -- rises with
// RowsPerWarp (~0.40 at 2 rows, ~0.59 at 4). The lean accumulator is what lets
// rows=4/8 fit the register budget where ptx_ru's acc[R][U] would spill; the R
// independent rows already give the FMA pipeline enough ILP for one acc each.
template <int WarpsPerBlock, int RowsPerWarp, int KUnroll>
__global__ void lm_head_gemv_ptx_rlean_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row_base = (blockIdx.x * WarpsPerBlock + warp) * RowsPerWarp;
    if (row_base >= n)
    {
        return;
    }

    int k2 = k / 2;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w2[RowsPerWarp];
    bool has_row[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r)
    {
        int row = row_base + r;
        has_row[r] = row < n;
        int safe_row = has_row[r] ? row : row_base;
        w2[r] = reinterpret_cast<half2 const*>(weight + static_cast<long long>(safe_row) * k);
    }

    float acc[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r)
    {
        acc[r] = 0.0f;
    }

    int i = lane;
    for (; i + 32 * (KUnroll - 1) < k2; i += 32 * KUnroll)
    {
        uint32_t h_pack[KUnroll];
        uint32_t w_pack[RowsPerWarp][KUnroll];
        // All hidden + weight loads issued before any FMA.
#pragma unroll
        for (int u = 0; u < KUnroll; ++u)
        {
            h_pack[u] = ptx_ld_global_u32(h2 + i + 32 * u);
        }
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r)
        {
#pragma unroll
            for (int u = 0; u < KUnroll; ++u)
            {
                w_pack[r][u] = ptx_ld_global_u32(w2[r] + i + 32 * u);
            }
        }
#pragma unroll
        for (int u = 0; u < KUnroll; ++u)
        {
            float2 h = __half22float2(half2_from_u32(h_pack[u]));
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r)
            {
                float2 w = __half22float2(half2_from_u32(w_pack[r][u]));
                acc[r] = ptx_fma_rn_f32(h.x, w.x, acc[r]);
                acc[r] = ptx_fma_rn_f32(h.y, w.y, acc[r]);
            }
        }
    }
    for (; i < k2; i += 32)
    {
        float2 h = __half22float2(half2_from_u32(ptx_ld_global_u32(h2 + i)));
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r)
        {
            float2 w = __half22float2(half2_from_u32(ptx_ld_global_u32(w2[r] + i)));
            acc[r] = ptx_fma_rn_f32(h.x, w.x, acc[r]);
            acc[r] = ptx_fma_rn_f32(h.y, w.y, acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r)
    {
        if (has_row[r])
        {
            float out = acc[r];
            if ((k & 1) && lane == 0)
            {
                int kk = k - 1;
                out = ptx_fma_rn_f32(__half2float(hidden[kk]),
                    __half2float(weight[static_cast<long long>(row_base + r) * k + kk]), out);
            }
            out = warp_sum(out);
            if (lane == 0)
            {
                ptx_st_global_f32(logits + row_base + r, out);
            }
        }
    }
}

// Persistent-style variant. Same per-row work as the lean 2-row, 8-way unroll
// (which empirically peaked at ~60% on the latency-bound target), but each warp
// grid-strides over many row pairs instead of doing one and exiting. Launched
// with blocks = multiProcessorCount * factor so the grid is just enough to fill
// the chip in one wave; warps live long, amortizing prologue/epilogue and
// kernel-launch overhead. Expected gain is bounded by the prologue/epilogue
// fraction of per-warp time (~3-5%) plus any cross-row pipeline benefit
// (drains avoided at row boundaries) -- order ~5-15%, not a structural break.
template <int WarpsPerBlock>
__global__ void lm_head_gemv_ptx_persistent_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    constexpr int Rows = 2;
    constexpr int Unroll = 8;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row_groups = (n + Rows - 1) / Rows;
    int warp_stride = gridDim.x * WarpsPerBlock;
    int k2 = k / 2;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);

    for (int rg = blockIdx.x * WarpsPerBlock + warp; rg < row_groups; rg += warp_stride)
    {
        int row_base = rg * Rows;
        half2 const* w2[Rows];
        bool has_row[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r)
        {
            int row = row_base + r;
            has_row[r] = row < n;
            int safe_row = has_row[r] ? row : row_base;
            w2[r] = reinterpret_cast<half2 const*>(weight + static_cast<long long>(safe_row) * k);
        }

        float acc[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r)
        {
            acc[r] = 0.0f;
        }

        int i = lane;
        for (; i + 32 * (Unroll - 1) < k2; i += 32 * Unroll)
        {
            uint32_t h_pack[Unroll];
            uint32_t w_pack[Rows][Unroll];
#pragma unroll
            for (int u = 0; u < Unroll; ++u)
            {
                h_pack[u] = ptx_ld_global_u32(h2 + i + 32 * u);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r)
            {
#pragma unroll
                for (int u = 0; u < Unroll; ++u)
                {
                    w_pack[r][u] = ptx_ld_global_u32(w2[r] + i + 32 * u);
                }
            }
#pragma unroll
            for (int u = 0; u < Unroll; ++u)
            {
                float2 h = __half22float2(half2_from_u32(h_pack[u]));
#pragma unroll
                for (int r = 0; r < Rows; ++r)
                {
                    float2 w = __half22float2(half2_from_u32(w_pack[r][u]));
                    acc[r] = ptx_fma_rn_f32(h.x, w.x, acc[r]);
                    acc[r] = ptx_fma_rn_f32(h.y, w.y, acc[r]);
                }
            }
        }
        for (; i < k2; i += 32)
        {
            float2 h = __half22float2(half2_from_u32(ptx_ld_global_u32(h2 + i)));
#pragma unroll
            for (int r = 0; r < Rows; ++r)
            {
                float2 w = __half22float2(half2_from_u32(ptx_ld_global_u32(w2[r] + i)));
                acc[r] = ptx_fma_rn_f32(h.x, w.x, acc[r]);
                acc[r] = ptx_fma_rn_f32(h.y, w.y, acc[r]);
            }
        }

#pragma unroll
        for (int r = 0; r < Rows; ++r)
        {
            if (has_row[r])
            {
                float out = acc[r];
                if ((k & 1) && lane == 0)
                {
                    int kk = k - 1;
                    out = ptx_fma_rn_f32(__half2float(hidden[kk]),
                        __half2float(weight[static_cast<long long>(row_base + r) * k + kk]), out);
                }
                out = warp_sum(out);
                if (lane == 0)
                {
                    ptx_st_global_f32(logits + row_base + r, out);
                }
            }
        }
    }
}

// Hopper TMA (Tensor Memory Accelerator) + mbarrier helpers. sm_90a.
// cp.async.bulk.tensor.2d copies a (TileK x Rows) submatrix of weight global ->
// shared in a single instruction; mbarrier handles producer-consumer sync.
__device__ __forceinline__ void mbarrier_init_b64(uint64_t* bar, uint32_t count)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_b64(uint64_t* bar)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{ .reg .b64 t; mbarrier.arrive.shared::cta.b64 t, [%0]; }" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{ .reg .b64 t; mbarrier.arrive.expect_tx.shared::cta.b64 t, [%0], %1; }"
                 ::"r"(addr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, uint32_t phase)
{
    // Loop in C++ instead of PTX labels: NVCC expands `%=` to `$NNN`, but some
    // non-NVIDIA PTX consumers (e.g. PPU) reject `$` in identifiers.
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t done = 0;
    while (!done)
    {
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     "  selp.b32 %0, 1, 0, p;\n"
                     "}\n"
                     : "=r"(done)
                     : "r"(addr), "r"(phase));
    }
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    void* smem_dst, void const* tensor_map, int coord_c, int coord_n, uint64_t* mbar)
{
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        ::"r"(dst), "l"(tensor_map), "r"(coord_c), "r"(coord_n), "r"(bar)
        : "memory");
}

// Warp-specialized GEMV with TMA. Block layout:
//   warp 0:           producer (only lane 0 issues TMA + mbarrier arrives)
//   warps 1..NCons:   consumer warps (each owns RowsPerBlock/NConsumers rows)
//
// Hidden is staged into smem once at the start. Weight tiles stream into a
// Stages-deep ring buffer in smem via cp.async.bulk.tensor.2d; the producer
// blocks on mbar_empty[s] when the ring slot still holds a tile the consumers
// haven't drained, and arrives on mbar_full[s] (with the TMA tx-count) when it
// issues a new tile. Consumers wait on mbar_full[s], FMA into per-row
// accumulators, and arrive on mbar_empty[s]. In-flight load data lives in
// shared memory -- the register file does NOT bound the outstanding window.
template <int WarpsPerBlock, int RowsPerBlock, int TileKHalf2, int Stages, int NConsumers>
__global__ void lm_head_gemv_tma_ws_kernel(
    half const* __restrict__ hidden,
    __grid_constant__ CUtensorMap const tensor_map_weight,
    float* __restrict__ logits, int n, int k)
{
    static_assert(NConsumers >= 1, "need at least one consumer");
    static_assert(NConsumers + 1 <= WarpsPerBlock, "need 1 producer + NConsumers consumers <= WarpsPerBlock");
    static_assert(RowsPerBlock % NConsumers == 0, "RowsPerBlock must split evenly across consumers");
    constexpr int RowsPerConsumer = RowsPerBlock / NConsumers;
    constexpr int TileBytes = RowsPerBlock * TileKHalf2 * static_cast<int>(sizeof(half2));

    extern __shared__ char smem_raw[];
    int hidden_align = ((k * static_cast<int>(sizeof(half)) + 127) / 128) * 128;
    half* s_hidden = reinterpret_cast<half*>(smem_raw);
    half2* s_weight = reinterpret_cast<half2*>(smem_raw + hidden_align);
    int ring_bytes = Stages * RowsPerBlock * TileKHalf2 * static_cast<int>(sizeof(half2));
    uint64_t* mbar_full = reinterpret_cast<uint64_t*>(smem_raw + hidden_align + ring_bytes);
    uint64_t* mbar_empty = mbar_full + Stages;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int block_threads = WarpsPerBlock * 32;
    int row_base = blockIdx.x * RowsPerBlock;

    // One-time mbarrier init (before any use).
    if (tid < Stages)
    {
        mbarrier_init_b64(&mbar_full[tid], 1);
    }
    if (tid >= Stages && tid < Stages * 2)
    {
        mbarrier_init_b64(&mbar_empty[tid - Stages], NConsumers * 32);
    }

    // Stage hidden once.
    int k2 = k / 2;
    half2 const* g_h2 = reinterpret_cast<half2 const*>(hidden);
    half2* s_h2 = reinterpret_cast<half2*>(s_hidden);
    for (int i = tid; i < k2; i += block_threads)
    {
        s_h2[i] = g_h2[i];
    }
    __syncthreads();

    if (row_base >= n)
    {
        return;
    }

    int num_tiles = (k2 + TileKHalf2 - 1) / TileKHalf2;

    if (warp == 0)
    {
        if (lane == 0)
        {
            for (int t = 0; t < num_tiles; ++t)
            {
                int s = t % Stages;
                if (t >= Stages)
                {
                    uint32_t phase = ((t - Stages) / Stages) & 1;
                    mbarrier_wait_parity(&mbar_empty[s], phase);
                }
                int coord_c = t * TileKHalf2 * 2;
                int coord_n = row_base;
                mbarrier_arrive_expect_tx(&mbar_full[s], TileBytes);
                cp_async_bulk_tensor_2d(&s_weight[s * RowsPerBlock * TileKHalf2],
                    &tensor_map_weight, coord_c, coord_n, &mbar_full[s]);
            }
        }
    }
    else if (warp <= NConsumers)
    {
        int cons_id = warp - 1;
        int my_row_start = cons_id * RowsPerConsumer;

        float acc[RowsPerConsumer];
#pragma unroll
        for (int r = 0; r < RowsPerConsumer; ++r)
        {
            acc[r] = 0.0f;
        }

        for (int t = 0; t < num_tiles; ++t)
        {
            int s = t % Stages;
            uint32_t phase = (t / Stages) & 1;
            mbarrier_wait_parity(&mbar_full[s], phase);

            half2 const* w = &s_weight[s * RowsPerBlock * TileKHalf2];
            half2 const* h = reinterpret_cast<half2 const*>(s_hidden) + t * TileKHalf2;
#pragma unroll
            for (int u = lane; u < TileKHalf2; u += 32)
            {
                float2 hv = __half22float2(h[u]);
#pragma unroll
                for (int r = 0; r < RowsPerConsumer; ++r)
                {
                    float2 wv = __half22float2(w[(my_row_start + r) * TileKHalf2 + u]);
                    acc[r] = ptx_fma_rn_f32(hv.x, wv.x, acc[r]);
                    acc[r] = ptx_fma_rn_f32(hv.y, wv.y, acc[r]);
                }
            }

            mbarrier_arrive_b64(&mbar_empty[s]);
        }

#pragma unroll
        for (int r = 0; r < RowsPerConsumer; ++r)
        {
            float out = warp_sum(acc[r]);
            int row = row_base + my_row_start + r;
            if (lane == 0 && row < n)
            {
                logits[row] = out;
            }
        }
    }
}

template <int WarpsPerBlock, int RowsPerWarp, int KUnroll>
__global__ void lm_head_gemv_ptx_ru_kernel(
    half const* __restrict__ hidden, half const* __restrict__ weight, float* __restrict__ logits, int n, int k)
{
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row_base = (blockIdx.x * WarpsPerBlock + warp) * RowsPerWarp;
    if (row_base >= n)
    {
        return;
    }

    int k2 = k / 2;
    half2 const* h2 = reinterpret_cast<half2 const*>(hidden);
    half2 const* w2[RowsPerWarp];
    bool has_row[RowsPerWarp];

#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r)
    {
        int row = row_base + r;
        has_row[r] = row < n;
        int safe_row = has_row[r] ? row : row_base;
        w2[r] = reinterpret_cast<half2 const*>(weight + static_cast<long long>(safe_row) * k);
    }

    float acc[RowsPerWarp][KUnroll];
    float tail[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r)
    {
        tail[r] = 0.0f;
#pragma unroll
        for (int u = 0; u < KUnroll; ++u)
        {
            acc[r][u] = 0.0f;
        }
    }

    int i = lane;
    for (; i + 32 * (KUnroll - 1) < k2; i += 32 * KUnroll)
    {
        uint32_t h_pack[KUnroll];
        uint32_t w_pack[RowsPerWarp][KUnroll];

#pragma unroll
        for (int u = 0; u < KUnroll; ++u)
        {
            h_pack[u] = ptx_ld_global_u32(h2 + i + 32 * u);
        }

#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r)
        {
            if (has_row[r])
            {
#pragma unroll
                for (int u = 0; u < KUnroll; ++u)
                {
                    w_pack[r][u] = ptx_ld_global_u32(w2[r] + i + 32 * u);
                }
            }
        }

#pragma unroll
        for (int u = 0; u < KUnroll; ++u)
        {
            float2 h = __half22float2(half2_from_u32(h_pack[u]));
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r)
            {
                if (has_row[r])
                {
                    float2 w = __half22float2(half2_from_u32(w_pack[r][u]));
                    acc[r][u] = ptx_fma_rn_f32(h.x, w.x, acc[r][u]);
                    acc[r][u] = ptx_fma_rn_f32(h.y, w.y, acc[r][u]);
                }
            }
        }
    }

    for (; i < k2; i += 32)
    {
        uint32_t h_pack = ptx_ld_global_u32(h2 + i);
        float2 h = __half22float2(half2_from_u32(h_pack));
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r)
        {
            if (has_row[r])
            {
                uint32_t w_pack = ptx_ld_global_u32(w2[r] + i);
                float2 w = __half22float2(half2_from_u32(w_pack));
                tail[r] = ptx_fma_rn_f32(h.x, w.x, tail[r]);
                tail[r] = ptx_fma_rn_f32(h.y, w.y, tail[r]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r)
    {
        if (has_row[r])
        {
            float out = tail[r];
#pragma unroll
            for (int u = 0; u < KUnroll; ++u)
            {
                out += acc[r][u];
            }
            if ((k & 1) && lane == 0)
            {
                int kk = k - 1;
                float h = __half2float(hidden[kk]);
                out = ptx_fma_rn_f32(h, __half2float(weight[static_cast<long long>(row_base + r) * k + kk]), out);
            }
            out = warp_sum(out);
            if (lane == 0)
            {
                ptx_st_global_f32(logits + row_base + r, out);
            }
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
float run_lm_head_ptx_chunk4_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + WarpsPerBlock - 1) / WarpsPerBlock;
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_chunk4_kernel<WarpsPerBlock>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_chunk4(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_chunk4_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_chunk4_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_chunk4_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_r2_chunk4_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + 2 * WarpsPerBlock - 1) / (2 * WarpsPerBlock);
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_r2_chunk4_kernel<WarpsPerBlock>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_r2_chunk4(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_r2_chunk4_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_r2_chunk4_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_r2_chunk4_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_r2_chunk4u2_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + 2 * WarpsPerBlock - 1) / (2 * WarpsPerBlock);
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_r2_chunk4u2_kernel<WarpsPerBlock>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_r2_chunk4u2(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_r2_chunk4u2_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_r2_chunk4u2_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_r2_chunk4u2_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock, int Tiles>
float run_lm_head_ptx_r2_chunk4u_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + 2 * WarpsPerBlock - 1) / (2 * WarpsPerBlock);
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_r2_chunk4u_kernel<WarpsPerBlock, Tiles>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

template <int Tiles>
float run_lm_head_ptx_r2_chunk4u(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_r2_chunk4u_kernel<4, Tiles>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_r2_chunk4u_kernel<8, Tiles>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_r2_chunk4u_kernel<16, Tiles>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_r2_chunk4u2_smem_kernel(
    Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + 2 * WarpsPerBlock - 1) / (2 * WarpsPerBlock);
    size_t smem = static_cast<size_t>(opt.k) * sizeof(half);
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_r2_chunk4u2_smem_kernel<WarpsPerBlock>
                <<<blocks, threads, smem>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_r2_chunk4u2_smem(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_r2_chunk4u2_smem_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_r2_chunk4u2_smem_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_r2_chunk4u2_smem_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock, int Stages>
float run_lm_head_ptx_cpasync_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + WarpsPerBlock - 1) / WarpsPerBlock;
    int ring_off = ((opt.k * static_cast<int>(sizeof(half)) + 15) / 16) * 16;
    size_t smem = static_cast<size_t>(ring_off) + static_cast<size_t>(WarpsPerBlock) * Stages * 128 * sizeof(half2);
    // Deep pipelines exceed the 48 KB default; opt into the larger limit.
    CHECK_CUDA(cudaFuncSetAttribute(lm_head_gemv_ptx_cpasync_kernel<WarpsPerBlock, Stages>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_cpasync_kernel<WarpsPerBlock, Stages>
                <<<blocks, threads, smem>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

// --k-unroll selects the cp.async pipeline depth (Stages). Deeper Stages keeps
// more weight tiles in flight in shared memory: per warp the in-flight window
// is (Stages-1) tiles * 512 B. Stages=4 is shallower than the register-resident
// chunk4 window; Stages=16 is ~2.5x deeper.
template <int WarpsPerBlock>
float run_lm_head_ptx_cpasync_stages(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.k_unroll == 4)
    {
        return run_lm_head_ptx_cpasync_kernel<WarpsPerBlock, 4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.k_unroll == 8)
    {
        return run_lm_head_ptx_cpasync_kernel<WarpsPerBlock, 8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_cpasync_kernel<WarpsPerBlock, 16>(opt, d_hidden, d_weight, d_logits);
}

float run_lm_head_ptx_cpasync(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_cpasync_stages<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_cpasync_stages<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_cpasync_stages<16>(opt, d_hidden, d_weight, d_logits);
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

template <int WarpsPerBlock>
float run_lm_head_ptx_r2u8_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + 2 * WarpsPerBlock - 1) / (2 * WarpsPerBlock);
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_r2u8_kernel<WarpsPerBlock>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_r2u8(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_r2u8_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_r2u8_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_r2u8_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_r4_chunk4_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + 4 * WarpsPerBlock - 1) / (4 * WarpsPerBlock);
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_r4_chunk4_kernel<WarpsPerBlock>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_r4_chunk4(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_r4_chunk4_kernel<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_r4_chunk4_kernel<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_r4_chunk4_kernel<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock, int RowsPerWarp, int KUnroll>
float run_lm_head_ptx_ru_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int rows_per_block = WarpsPerBlock * RowsPerWarp;
    int blocks = (opt.n + rows_per_block - 1) / rows_per_block;
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_ru_kernel<WarpsPerBlock, RowsPerWarp, KUnroll>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

template <int WarpsPerBlock, int RowsPerWarp>
float run_lm_head_ptx_ru_rows(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.k_unroll == 4)
    {
        return run_lm_head_ptx_ru_kernel<WarpsPerBlock, RowsPerWarp, 4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.k_unroll == 8)
    {
        return run_lm_head_ptx_ru_kernel<WarpsPerBlock, RowsPerWarp, 8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_ru_kernel<WarpsPerBlock, RowsPerWarp, 16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_ru_warps(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.rows_per_warp == 1)
    {
        return run_lm_head_ptx_ru_rows<WarpsPerBlock, 1>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.rows_per_warp == 2)
    {
        return run_lm_head_ptx_ru_rows<WarpsPerBlock, 2>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.rows_per_warp == 4)
    {
        return run_lm_head_ptx_ru_rows<WarpsPerBlock, 4>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_ru_rows<WarpsPerBlock, 8>(opt, d_hidden, d_weight, d_logits);
}

float run_lm_head_ptx_ru(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_ru_warps<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_ru_warps<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_ru_warps<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock, int RowsPerWarp, int KUnroll>
float run_lm_head_ptx_rlean_kernel(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int rows_per_block = WarpsPerBlock * RowsPerWarp;
    int blocks = (opt.n + rows_per_block - 1) / rows_per_block;
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_rlean_kernel<WarpsPerBlock, RowsPerWarp, KUnroll>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

template <int WarpsPerBlock, int RowsPerWarp>
float run_lm_head_ptx_rlean_rows(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.k_unroll == 4)
    {
        return run_lm_head_ptx_rlean_kernel<WarpsPerBlock, RowsPerWarp, 4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.k_unroll == 8)
    {
        return run_lm_head_ptx_rlean_kernel<WarpsPerBlock, RowsPerWarp, 8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_rlean_kernel<WarpsPerBlock, RowsPerWarp, 16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_rlean_warps(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.rows_per_warp == 1)
    {
        return run_lm_head_ptx_rlean_rows<WarpsPerBlock, 1>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.rows_per_warp == 2)
    {
        return run_lm_head_ptx_rlean_rows<WarpsPerBlock, 2>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.rows_per_warp == 4)
    {
        return run_lm_head_ptx_rlean_rows<WarpsPerBlock, 4>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_rlean_rows<WarpsPerBlock, 8>(opt, d_hidden, d_weight, d_logits);
}

float run_lm_head_ptx_rlean(Options const& opt, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_rlean_warps<4>(opt, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_rlean_warps<8>(opt, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_rlean_warps<16>(opt, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock>
float run_lm_head_ptx_persistent_kernel(
    Options const& opt, int blocks, half const* d_hidden, half const* d_weight, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    return median_time_ms(
        [&] {
            lm_head_gemv_ptx_persistent_kernel<WarpsPerBlock>
                <<<blocks, threads>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_persistent(
    Options const& opt, int blocks, half const* d_hidden, half const* d_weight, float* d_logits)
{
    if (opt.warps_per_block == 4)
    {
        return run_lm_head_ptx_persistent_kernel<4>(opt, blocks, d_hidden, d_weight, d_logits);
    }
    if (opt.warps_per_block == 8)
    {
        return run_lm_head_ptx_persistent_kernel<8>(opt, blocks, d_hidden, d_weight, d_logits);
    }
    return run_lm_head_ptx_persistent_kernel<16>(opt, blocks, d_hidden, d_weight, d_logits);
}

template <int WarpsPerBlock, int RowsPerBlock, int TileKHalf2, int Stages, int NConsumers>
float run_lm_head_ptx_tma_ws_kernel(
    Options const& opt, CUtensorMap const& weight_map, half const* d_hidden, float* d_logits)
{
    int threads = WarpsPerBlock * 32;
    int blocks = (opt.n + RowsPerBlock - 1) / RowsPerBlock;
    int hidden_align = ((opt.k * static_cast<int>(sizeof(half)) + 127) / 128) * 128;
    int ring_bytes = Stages * RowsPerBlock * TileKHalf2 * static_cast<int>(sizeof(half2));
    int mbar_bytes = 2 * Stages * static_cast<int>(sizeof(uint64_t));
    size_t smem = static_cast<size_t>(hidden_align + ring_bytes + mbar_bytes);
    CHECK_CUDA(cudaFuncSetAttribute(
        lm_head_gemv_tma_ws_kernel<WarpsPerBlock, RowsPerBlock, TileKHalf2, Stages, NConsumers>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    return median_time_ms(
        [&] {
            lm_head_gemv_tma_ws_kernel<WarpsPerBlock, RowsPerBlock, TileKHalf2, Stages, NConsumers>
                <<<blocks, threads, smem>>>(d_hidden, weight_map, d_logits, opt.n, opt.k);
        },
        opt.warmup, opt.iters);
}

float run_lm_head_ptx_tma_ws(
    Options const& opt, CUtensorMap const& weight_map, half const* d_hidden, float* d_logits)
{
    // --k-unroll selects pipeline Stages (4/8/16). Other dims fixed:
    // WPB=5 (1 producer + 4 consumers), Rows=16, TileK=128 half2.
    if (opt.k_unroll == 4)
    {
        return run_lm_head_ptx_tma_ws_kernel<5, 16, 128, 4, 4>(opt, weight_map, d_hidden, d_logits);
    }
    if (opt.k_unroll == 8)
    {
        return run_lm_head_ptx_tma_ws_kernel<5, 16, 128, 8, 4>(opt, weight_map, d_hidden, d_logits);
    }
    return run_lm_head_ptx_tma_ws_kernel<5, 16, 128, 16, 4>(opt, weight_map, d_hidden, d_logits);
}

static CUtensorMap make_weight_tensormap(
    half* d_weight, int n, int k, uint32_t tile_k_halves, uint32_t rows_per_tile)
{
    CUtensorMap tmap{};
    uint64_t size[2] = {static_cast<uint64_t>(k), static_cast<uint64_t>(n)};
    uint64_t stride[1] = {static_cast<uint64_t>(k) * sizeof(half)};
    uint32_t box[2] = {tile_k_halves, rows_per_tile};
    uint32_t elem_stride[2] = {1, 1};
    CUresult rc = cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, /*rank=*/2, d_weight, size,
        stride, box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (rc != CUDA_SUCCESS)
    {
        char const* msg = nullptr;
        cuGetErrorString(rc, &msg);
        std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", msg ? msg : "(unknown)");
        std::exit(1);
    }
    return tmap;
}

void print_bw(char const* name, float ms, double bytes)
{
    double gbps = bytes / (ms * 1.0e-3) / 1.0e9;
    std::printf("%-14s median=%.4f ms traffic=%.3f MB bw=%.1f GB/s %.3f TB/s\n", name, ms, bytes / 1.0e6,
        gbps, gbps / 1000.0);
}

void run_gemv_cases(Options const& opt, bool run_shared, bool run_global, bool run_ptx, bool run_ptx_u4,
    bool run_ptx_r2u4, bool run_ptx_chunk4, bool run_ptx_r2_chunk4, bool run_ptx_r4_chunk4)
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
    if (run_ptx_chunk4)
    {
        float ms = run_lm_head_ptx_chunk4(opt, d_hidden, d_weight, d_logits);
        CHECK_CUDA(cudaGetLastError());
        print_bw("ptx_chunk4", ms, mandatory_traffic);
    }
    if (run_ptx_r2_chunk4)
    {
        float ms = run_lm_head_ptx_r2_chunk4(opt, d_hidden, d_weight, d_logits);
        CHECK_CUDA(cudaGetLastError());
        print_bw("ptx_r2_chunk4", ms, mandatory_traffic);
    }
    if (run_ptx_r4_chunk4)
    {
        float ms = run_lm_head_ptx_r4_chunk4(opt, d_hidden, d_weight, d_logits);
        CHECK_CUDA(cudaGetLastError());
        print_bw("ptx_r4_chunk4", ms, mandatory_traffic);
    }

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_ru_case(Options const& opt)
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
    std::printf("lm_head ptx_ru: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d rows/warp=%d k_unroll=%d\n",
        opt.n, opt.k, weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block, opt.rows_per_warp,
        opt.k_unroll);
    float ms = run_lm_head_ptx_ru(opt, d_hidden, d_weight, d_logits);
    CHECK_CUDA(cudaGetLastError());
    char name[64];
    std::snprintf(name, sizeof(name), "ptx_ru_r%d_u%d", opt.rows_per_warp, opt.k_unroll);
    print_bw(name, ms, mandatory_traffic);

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_rlean_case(Options const& opt)
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
    std::printf("lm_head ptx_rlean: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d rows/warp=%d k_unroll=%d\n",
        opt.n, opt.k, weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block, opt.rows_per_warp,
        opt.k_unroll);
    float ms = run_lm_head_ptx_rlean(opt, d_hidden, d_weight, d_logits);
    CHECK_CUDA(cudaGetLastError());
    char name[64];
    std::snprintf(name, sizeof(name), "ptx_rlean_r%d_u%d", opt.rows_per_warp, opt.k_unroll);
    print_bw(name, ms, mandatory_traffic);

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_persistent_case(Options const& opt)
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

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int persistent_factor = 4;
    int blocks = prop.multiProcessorCount * persistent_factor;

    double mandatory_traffic = static_cast<double>(weight_bytes + logits_bytes);
    std::printf("lm_head ptx_persistent: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d blocks=%d (sms=%d)\n",
        opt.n, opt.k, weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block, blocks,
        prop.multiProcessorCount);
    float ms = run_lm_head_ptx_persistent(opt, blocks, d_hidden, d_weight, d_logits);
    CHECK_CUDA(cudaGetLastError());
    print_bw("ptx_persistent", ms, mandatory_traffic);

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_tma_ws_case(Options const& opt)
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

    // Baseline TMA+WS: 16 rows/block, TileK = 256 halves = 128 half2 per tile, 4 stages, 4 consumers.
    CUtensorMap weight_map
        = make_weight_tensormap(d_weight, opt.n, opt.k, /*tile_k_halves=*/256, /*rows_per_tile=*/16);

    double mandatory_traffic = static_cast<double>(weight_bytes + logits_bytes);
    std::printf("lm_head ptx_tma_ws: n=%d k=%d weight=%.3f MB logits=%.3f MB R=16 TileK=256 Stages=4 NCons=4 WPB=5\n",
        opt.n, opt.k, weight_bytes / 1.0e6, logits_bytes / 1.0e6);

    if (opt.verify)
    {
        // Correctness check vs the simple inline-PTX GEMV: constant inputs make
        // every output row identical -- logits[0] and logits[n-1] must match.
        lm_head_gemv_ptx_kernel<8><<<(opt.n + 7) / 8, 256>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        float ref = 0.0f;
        CHECK_CUDA(cudaMemcpy(&ref, d_logits, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemset(d_logits, 0, logits_bytes));
        {
            int blocks = (opt.n + 15) / 16;
            int threads = 5 * 32;
            int hidden_align = ((opt.k * static_cast<int>(sizeof(half)) + 127) / 128) * 128;
            int ring_bytes = 4 * 16 * 128 * static_cast<int>(sizeof(half2));
            int mbar_bytes = 2 * 4 * static_cast<int>(sizeof(uint64_t));
            size_t smem = static_cast<size_t>(hidden_align + ring_bytes + mbar_bytes);
            CHECK_CUDA(cudaFuncSetAttribute(lm_head_gemv_tma_ws_kernel<5, 16, 128, 4, 4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
            lm_head_gemv_tma_ws_kernel<5, 16, 128, 4, 4>
                <<<blocks, threads, smem>>>(d_hidden, weight_map, d_logits, opt.n, opt.k);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        float v0 = 0.0f;
        float vn = 0.0f;
        CHECK_CUDA(cudaMemcpy(&v0, d_logits, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&vn, d_logits + (opt.n - 1), sizeof(float), cudaMemcpyDeviceToHost));
        float tol = (ref < 0.0f ? -ref : ref) * 1.0e-3f + 1.0e-3f;
        float d0 = (v0 - ref < 0.0f) ? ref - v0 : v0 - ref;
        float dn = (vn - ref < 0.0f) ? ref - vn : vn - ref;
        bool ok = d0 <= tol && dn <= tol;
        std::printf("  tma_ws verify: ref=%.4f v[0]=%.4f v[n-1]=%.4f -> %s\n", ref, v0, vn, ok ? "OK" : "MISMATCH");
    }

    float ms = run_lm_head_ptx_tma_ws(opt, weight_map, d_hidden, d_logits);
    CHECK_CUDA(cudaGetLastError());
    print_bw("ptx_tma_ws", ms, mandatory_traffic);

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_r2u8_case(Options const& opt)
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
    std::printf("lm_head ptx_r2u8: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d\n", opt.n, opt.k,
        weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block);
    float ms = run_lm_head_ptx_r2u8(opt, d_hidden, d_weight, d_logits);
    CHECK_CUDA(cudaGetLastError());
    print_bw("ptx_r2u8", ms, mandatory_traffic);

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_r2_chunk4u2_case(Options const& opt)
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
    std::printf("lm_head ptx_r2_chunk4u2: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d\n", opt.n, opt.k,
        weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block);
    float ms = run_lm_head_ptx_r2_chunk4u2(opt, d_hidden, d_weight, d_logits);
    CHECK_CUDA(cudaGetLastError());
    print_bw("ptx_r2_chunk4u2", ms, mandatory_traffic);

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_r2_chunk4u4_case(Options const& opt)
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
    std::printf("lm_head ptx_r2_chunk4u4: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d\n", opt.n, opt.k,
        weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block);
    float ms = run_lm_head_ptx_r2_chunk4u<4>(opt, d_hidden, d_weight, d_logits);
    CHECK_CUDA(cudaGetLastError());
    print_bw("ptx_r2_chunk4u4", ms, mandatory_traffic);

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_r2_chunk4u2_smem_case(Options const& opt)
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
    std::printf("lm_head ptx_r2_chunk4u2_smem: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d\n", opt.n, opt.k,
        weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block);
    float ms = run_lm_head_ptx_r2_chunk4u2_smem(opt, d_hidden, d_weight, d_logits);
    CHECK_CUDA(cudaGetLastError());
    print_bw("ptx_r2_chunk4u2_smem", ms, mandatory_traffic);

    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_logits));
}

void run_gemv_cpasync_case(Options const& opt)
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
    std::printf("lm_head ptx_cpasync: n=%d k=%d weight=%.3f MB logits=%.3f MB warps/block=%d\n", opt.n, opt.k,
        weight_bytes / 1.0e6, logits_bytes / 1.0e6, opt.warps_per_block);

    if (opt.verify)
    {
        // Correctness check vs the simple inline-PTX GEMV: the benchmark fills the
        // inputs with constant bytes, so every output row is identical -- logits[0]
        // and logits[n-1] must match the reference and each other.
        lm_head_gemv_ptx_kernel<8><<<(opt.n + 7) / 8, 256>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        float ref = 0.0f;
        CHECK_CUDA(cudaMemcpy(&ref, d_logits, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemset(d_logits, 0, logits_bytes));
        {
            int ring_off = ((opt.k * static_cast<int>(sizeof(half)) + 15) / 16) * 16;
            size_t smem = static_cast<size_t>(ring_off) + static_cast<size_t>(8) * 4 * 128 * sizeof(half2);
            lm_head_gemv_ptx_cpasync_kernel<8, 4>
                <<<(opt.n + 7) / 8, 256, smem>>>(d_hidden, d_weight, d_logits, opt.n, opt.k);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        float v0 = 0.0f;
        float vn = 0.0f;
        CHECK_CUDA(cudaMemcpy(&v0, d_logits, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&vn, d_logits + (opt.n - 1), sizeof(float), cudaMemcpyDeviceToHost));
        float tol = (ref < 0.0f ? -ref : ref) * 1.0e-3f + 1.0e-3f;
        float d0 = (v0 - ref < 0.0f) ? ref - v0 : v0 - ref;
        float dn = (vn - ref < 0.0f) ? ref - vn : vn - ref;
        bool ok = d0 <= tol && dn <= tol;
        std::printf("  cpasync verify: ref=%.4f v[0]=%.4f v[n-1]=%.4f -> %s\n", ref, v0, vn, ok ? "OK" : "MISMATCH");
    }

    float ms = run_lm_head_ptx_cpasync(opt, d_hidden, d_weight, d_logits);
    CHECK_CUDA(cudaGetLastError());
    print_bw("ptx_cpasync", ms, mandatory_traffic);

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
    std::printf("device=%s sms=%d op=%s warmup=%d iters=%d rows/warp=%d k_unroll=%d\n", prop.name,
        prop.multiProcessorCount, opt.op.c_str(), opt.warmup, opt.iters, opt.rows_per_warp, opt.k_unroll);

    if (opt.op == "all")
    {
        run_gemv_cases(opt, true, true, true, true, true, true, true, true);
        run_cublas_case(opt);
        run_copy_case(opt);
    }
    else if (opt.op == "shared")
    {
        run_gemv_cases(opt, true, false, false, false, false, false, false, false);
    }
    else if (opt.op == "global")
    {
        run_gemv_cases(opt, false, true, false, false, false, false, false, false);
    }
    else if (opt.op == "ptx")
    {
        run_gemv_cases(opt, false, false, true, false, false, false, false, false);
    }
    else if (opt.op == "ptx_u4")
    {
        run_gemv_cases(opt, false, false, false, true, false, false, false, false);
    }
    else if (opt.op == "ptx_r2u4")
    {
        run_gemv_cases(opt, false, false, false, false, true, false, false, false);
    }
    else if (opt.op == "ptx_r2u8")
    {
        run_gemv_r2u8_case(opt);
    }
    else if (opt.op == "ptx_r2_chunk4u2")
    {
        run_gemv_r2_chunk4u2_case(opt);
    }
    else if (opt.op == "ptx_r2_chunk4u4")
    {
        run_gemv_r2_chunk4u4_case(opt);
    }
    else if (opt.op == "ptx_r2_chunk4u2_smem")
    {
        run_gemv_r2_chunk4u2_smem_case(opt);
    }
    else if (opt.op == "ptx_cpasync")
    {
        run_gemv_cpasync_case(opt);
    }
    else if (opt.op == "ptx_chunk4")
    {
        run_gemv_cases(opt, false, false, false, false, false, true, false, false);
    }
    else if (opt.op == "ptx_r2_chunk4")
    {
        run_gemv_cases(opt, false, false, false, false, false, false, true, false);
    }
    else if (opt.op == "ptx_r4_chunk4")
    {
        run_gemv_cases(opt, false, false, false, false, false, false, false, true);
    }
    else if (opt.op == "ptx_ru")
    {
        run_gemv_ru_case(opt);
    }
    else if (opt.op == "ptx_rlean")
    {
        run_gemv_rlean_case(opt);
    }
    else if (opt.op == "ptx_persistent")
    {
        run_gemv_persistent_case(opt);
    }
    else if (opt.op == "ptx_tma_ws")
    {
        run_gemv_tma_ws_case(opt);
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
