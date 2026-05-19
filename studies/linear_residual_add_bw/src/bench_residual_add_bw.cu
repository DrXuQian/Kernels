#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

namespace
{

struct Options
{
    std::string op = "all";
    long long tokens = 3823;
    long long hidden = 3072;
    size_t mib = 0;
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
        else if (starts_with(argv[i], "--tokens="))
        {
            opt.tokens = std::atoll(argv[i] + 9);
        }
        else if (std::strcmp(argv[i], "--tokens") == 0)
        {
            opt.tokens = std::atoll(argv[++i]);
        }
        else if (starts_with(argv[i], "--hidden="))
        {
            opt.hidden = std::atoll(argv[i] + 9);
        }
        else if (std::strcmp(argv[i], "--hidden") == 0)
        {
            opt.hidden = std::atoll(argv[++i]);
        }
        else if (starts_with(argv[i], "--mib="))
        {
            opt.mib = static_cast<size_t>(std::atoll(argv[i] + 6));
        }
        else if (std::strcmp(argv[i], "--mib") == 0)
        {
            opt.mib = static_cast<size_t>(std::atoll(argv[++i]));
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
            std::printf("Usage: %s [--op all|scalar|half8|copy|memcpy|copy_u4|copy_u8]\n"
                        "          [--tokens N] [--hidden N] [--mib MiB] [--warmup N] [--iters N]\n",
                argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            std::exit(1);
        }
    }

    if (opt.tokens <= 0 || opt.hidden <= 0 || opt.warmup < 0 || opt.iters <= 0)
    {
        std::fprintf(stderr, "tokens/hidden/iters must be positive and warmup must be non-negative\n");
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

__global__ void residual_add_scalar_kernel(
    half* __restrict__ out, half const* __restrict__ residual, half const* __restrict__ update, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = idx; i < n; i += stride)
    {
        out[i] = __float2half(__half2float(residual[i]) + __half2float(update[i]));
    }
}

struct alignas(16) Half8
{
    half2 x0;
    half2 x1;
    half2 x2;
    half2 x3;
};

__global__ void residual_add_half8_kernel(
    half* __restrict__ out, half const* __restrict__ residual, half const* __restrict__ update, long long n)
{
    long long vec_n = n / 8;
    Half8* __restrict__ out_vec = reinterpret_cast<Half8*>(out);
    Half8 const* __restrict__ residual_vec = reinterpret_cast<Half8 const*>(residual);
    Half8 const* __restrict__ update_vec = reinterpret_cast<Half8 const*>(update);

    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = idx; i < vec_n; i += stride)
    {
        Half8 a = residual_vec[i];
        Half8 b = update_vec[i];
        Half8 c;
        c.x0 = __hadd2(a.x0, b.x0);
        c.x1 = __hadd2(a.x1, b.x1);
        c.x2 = __hadd2(a.x2, b.x2);
        c.x3 = __hadd2(a.x3, b.x3);
        out_vec[i] = c;
    }

    long long tail_start = vec_n * 8;
    for (long long i = tail_start + idx; i < n; i += stride)
    {
        out[i] = __hadd(residual[i], update[i]);
    }
}

__global__ void copy_u4_kernel(uint4* __restrict__ dst, uint4 const* __restrict__ src, size_t n)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < n; i += stride)
    {
        dst[i] = src[i];
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

void print_bw(char const* name, float ms, double bytes)
{
    double gbps = bytes / (ms * 1.0e-3) / 1.0e9;
    std::printf("%-18s median=%.4f ms traffic=%.3f MB bw=%.1f GB/s %.3f TB/s\n", name, ms, bytes / 1.0e6,
        gbps, gbps / 1000.0);
}

void run_residual_cases(Options const& opt, bool run_scalar, bool run_half8)
{
    long long elems = opt.tokens * opt.hidden;
    size_t bytes = static_cast<size_t>(elems) * sizeof(half);
    half* d_a = nullptr;
    half* d_b = nullptr;
    half* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemset(d_a, 0x3a, bytes));
    CHECK_CUDA(cudaMemset(d_b, 0x1d, bytes));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int threads = 256;
    int scalar_blocks = static_cast<int>(std::min(4096LL, (elems + threads - 1) / threads));
    int half8_blocks = static_cast<int>(std::min(4096LL, ((elems + 7) / 8 + threads - 1) / threads));
    double residual_traffic = static_cast<double>(bytes) * 3.0;

    std::printf("residual shape=(%lld,%lld) elems=%lld blocks scalar=%d half8=%d\n", opt.tokens, opt.hidden, elems,
        scalar_blocks, half8_blocks);
    if (run_scalar)
    {
        float ms = median_time_ms(
            [&] { residual_add_scalar_kernel<<<scalar_blocks, threads>>>(d_out, d_a, d_b, elems); }, opt.warmup,
            opt.iters);
        CHECK_CUDA(cudaGetLastError());
        print_bw("scalar_residual", ms, residual_traffic);
    }
    if (run_half8)
    {
        float ms = median_time_ms(
            [&] { residual_add_half8_kernel<<<half8_blocks, threads>>>(d_out, d_a, d_b, elems); }, opt.warmup,
            opt.iters);
        CHECK_CUDA(cudaGetLastError());
        print_bw("half8_residual", ms, residual_traffic);
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
}

void run_copy_cases(Options const& opt, bool run_memcpy, bool run_u4, bool run_u8)
{
    size_t bytes = opt.mib > 0 ? opt.mib * 1024ULL * 1024ULL : static_cast<size_t>(opt.tokens * opt.hidden) * sizeof(half);
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
    double copy_traffic = static_cast<double>(bytes) * 2.0;
    std::printf("copy bytes=%.3f MiB blocks=%d\n", static_cast<double>(bytes) / (1024.0 * 1024.0), blocks);

    if (run_memcpy)
    {
        float ms = median_time_ms(
            [&] { CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice)); }, opt.warmup, opt.iters);
        print_bw("cudaMemcpyD2D", ms, copy_traffic);
    }
    if (run_u4)
    {
        size_t n = bytes / sizeof(uint4);
        float ms = median_time_ms(
            [&] { copy_u4_kernel<<<blocks, threads>>>(static_cast<uint4*>(dst), static_cast<uint4 const*>(src), n); },
            opt.warmup, opt.iters);
        CHECK_CUDA(cudaGetLastError());
        print_bw("copy_u4_kernel", ms, copy_traffic);
    }
    if (run_u8)
    {
        size_t n = bytes / sizeof(ulonglong4);
        float ms = median_time_ms(
            [&] {
                copy_u8_kernel<<<blocks, threads>>>(
                    static_cast<ulonglong4*>(dst), static_cast<ulonglong4 const*>(src), n);
            },
            opt.warmup, opt.iters);
        CHECK_CUDA(cudaGetLastError());
        print_bw("copy_u8_kernel", ms, copy_traffic);
    }

    CHECK_CUDA(cudaFree(src));
    CHECK_CUDA(cudaFree(dst));
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
        run_residual_cases(opt, true, true);
        run_copy_cases(opt, true, true, true);
    }
    else if (opt.op == "scalar")
    {
        run_residual_cases(opt, true, false);
    }
    else if (opt.op == "half8")
    {
        run_residual_cases(opt, false, true);
    }
    else if (opt.op == "copy")
    {
        run_copy_cases(opt, true, true, true);
    }
    else if (opt.op == "memcpy")
    {
        run_copy_cases(opt, true, false, false);
    }
    else if (opt.op == "copy_u4")
    {
        run_copy_cases(opt, false, true, false);
    }
    else if (opt.op == "copy_u8")
    {
        run_copy_cases(opt, false, false, true);
    }
    else
    {
        std::fprintf(stderr, "unknown op: %s\n", opt.op.c_str());
        return 1;
    }
    return 0;
}

