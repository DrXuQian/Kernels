// Fused RMS Norm + Sigmoid Gate
// y = (x / rms) * weight * sigmoid(z)
// Usage: ./bench_fused_rms_norm_gate [N] [D] [--dtype fp16|bf16] [--bench W I]
//   122B DeltaNet: N=64(heads), D=128(head_dim) for decode
//   prefill: N=64*seq_chunks, D=128
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "bench_timer.h"

#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
    return __float2half(value);
}

template <typename T>
T host_from_float(float value);

template <>
__nv_bfloat16 host_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
__half host_from_float<__half>(float value) {
    return __float2half(value);
}

template <typename T>
__global__ void fused_rms_norm_gate_kernel(
    const T* __restrict__ x,
    const T* __restrict__ z,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const T* x_row = x + (long long)row * D;
    const T* z_row = z + (long long)row * D;
    T* out_row = output + (long long)row * D;

    float local_sum_sq = 0.0f;
    for (int i = tid; i < D; i += nthreads) {
        float v = to_float<T>(x_row[i]);
        local_sum_sq += v * v;
    }

    __shared__ float warp_sums[32];
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int nwarps = (nthreads + 31) / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    if (lane == 0) warp_sums[warp_id] = local_sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        warp_sums[0] = total;
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] / D + eps);

    for (int i = tid; i < D; i += nthreads) {
        float xv = to_float<T>(x_row[i]) * inv_rms;
        float wv = to_float<T>(weight[i]);
        float zv = to_float<T>(z_row[i]);
        float gate = 1.0f / (1.0f + expf(-zv));
        out_row[i] = from_float<T>(xv * wv * gate);
    }
}

template <typename T>
void fused_rms_norm_gate_launch(
    const T* x, const T* z, const T* weight,
    T* output, int N, int D, float eps, cudaStream_t stream = 0)
{
    fused_rms_norm_gate_kernel<T><<<N, 256, 0, stream>>>(x, z, weight, output, N, D, eps);
}

struct Options {
    int N = 64;
    int D = 128;
    std::string dtype = "bf16";
};

Options parse_options(int argc, char** argv) {
    Options opt;
    opt.N = 64;
    opt.D = 128;
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            opt.dtype = argv[++i];
        } else if (strncmp(argv[i], "--dtype=", 8) == 0) {
            opt.dtype = argv[i] + 8;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [N] [D] [--dtype fp16|bf16] [--bench W I]\n", argv[0]);
            exit(0);
        } else {
            int value = atoi(argv[i]);
            if (positional == 0) opt.N = value;
            else if (positional == 1) opt.D = value;
            else {
                fprintf(stderr, "unexpected positional argument: %s\n", argv[i]);
                exit(1);
            }
            ++positional;
        }
    }
    if (opt.dtype != "fp16" && opt.dtype != "bf16") {
        fprintf(stderr, "dtype must be fp16 or bf16\n");
        exit(1);
    }
    return opt;
}

template <typename T>
int run(Options const& opt, BenchTimer& timer) {
    float eps = 1e-6f;
    printf("bench fused_rms_norm_gate: N=%d D=%d dtype=%s\n", opt.N, opt.D, opt.dtype.c_str());

    long long total = (long long)opt.N * opt.D;
    T *d_x, *d_z, *d_w, *d_out;
    CHECK(cudaMalloc(&d_x, total * sizeof(T)));
    CHECK(cudaMalloc(&d_z, total * sizeof(T)));
    CHECK(cudaMalloc(&d_w, opt.D * sizeof(T)));
    CHECK(cudaMalloc(&d_out, total * sizeof(T)));

    srand(42);
    auto fill = [](T* d, long long n) {
        std::vector<T> h(n);
        for (auto& v : h) v = host_from_float<T>(((float)rand()/RAND_MAX - 0.5f) * 0.2f);
        cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    };
    fill(d_x, total); fill(d_z, total); fill(d_w, opt.D);

    timer.run([&]() {
        fused_rms_norm_gate_launch<T>(d_x, d_z, d_w, d_out, opt.N, opt.D, eps);
    });
    CHECK(cudaGetLastError());

    printf("Done.\n");
    cudaFree(d_x); cudaFree(d_z); cudaFree(d_w); cudaFree(d_out);
    return 0;
}

// ── Bench ──
int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    Options opt = parse_options(argc, argv);
    return opt.dtype == "fp16" ? run<__half>(opt, timer) : run<__nv_bfloat16>(opt, timer);
}
