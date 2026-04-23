// Fused RMS Norm + Sigmoid Gate
// y = (x / rms) * weight * sigmoid(z)
// Usage: ./bench_fused_rms_norm_gate [N] [D] [--bench W I]
//   122B DeltaNet: N=64(heads), D=128(head_dim) for decode
//   prefill: N=64*seq_chunks, D=128
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "bench_timer.h"

#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)

__global__ void fused_rms_norm_gate_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const __nv_bfloat16* x_row = x + (long long)row * D;
    const __nv_bfloat16* z_row = z + (long long)row * D;
    __nv_bfloat16* out_row = output + (long long)row * D;

    float local_sum_sq = 0.0f;
    for (int i = tid; i < D; i += nthreads) {
        float v = __bfloat162float(x_row[i]);
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
        float xv = __bfloat162float(x_row[i]) * inv_rms;
        float wv = __bfloat162float(weight[i]);
        float zv = __bfloat162float(z_row[i]);
        float gate = 1.0f / (1.0f + expf(-zv));
        out_row[i] = __float2bfloat16(xv * wv * gate);
    }
}

void fused_rms_norm_gate_launch(
    const __nv_bfloat16* x, const __nv_bfloat16* z, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int N, int D, float eps, cudaStream_t stream = 0)
{
    fused_rms_norm_gate_kernel<<<N, 256, 0, stream>>>(x, z, weight, output, N, D, eps);
}

// ── Bench ──
int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    // 122B DeltaNet: after recurrent, output is (sab_heads, head_dim) = (64, 128)
    int N = (argc > 1) ? atoi(argv[1]) : 64;
    int D = (argc > 2) ? atoi(argv[2]) : 128;
    float eps = 1e-6f;
    printf("bench fused_rms_norm_gate: N=%d D=%d\n", N, D);

    long long total = (long long)N * D;
    __nv_bfloat16 *d_x, *d_z, *d_w, *d_out;
    CHECK(cudaMalloc(&d_x, total * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_z, total * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_w, D * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_out, total * sizeof(__nv_bfloat16)));

    srand(42);
    auto fill = [](auto* d, long long n) {
        std::vector<__nv_bfloat16> h(n);
        for (auto& v : h) v = __float2bfloat16(((float)rand()/RAND_MAX - 0.5f) * 0.2f);
        cudaMemcpy(d, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    };
    fill(d_x, total); fill(d_z, total); fill(d_w, D);

    timer.run([&]() {
        fused_rms_norm_gate_launch(d_x, d_z, d_w, d_out, N, D, eps);
    });
    CHECK(cudaGetLastError());

    printf("Done.\n");
    cudaFree(d_x); cudaFree(d_z); cudaFree(d_w); cudaFree(d_out);
    return 0;
}
