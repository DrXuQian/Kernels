// Bench K5: fused_rms_norm_gate
// Usage: ./bench_k5 [N]
//   prefill: N = seq_len (default 3823)
//   decode:  N = 1
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include "deltanet.h"

constexpr int D = 8192;  // NUM_V_HEADS * HEAD_DIM
constexpr float EPS = 1e-6f;

__global__ void fill_bf16(__nv_bfloat16* d, int n, float s, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState st; curand_init(seed, i, 0, &st);
    d[i] = __float2bfloat16(curand_normal(&st) * s);
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 3823;
    printf("bench_k5: fused_rms_norm_gate  N=%d  D=%d\n", N, D);

    long long total = (long long)N * D;
    __nv_bfloat16 *d_x, *d_z, *d_w, *d_out;
    cudaMalloc(&d_x, total * sizeof(__nv_bfloat16));
    cudaMalloc(&d_z, total * sizeof(__nv_bfloat16));
    cudaMalloc(&d_w, D * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, total * sizeof(__nv_bfloat16));

    fill_bf16<<<((int)total+255)/256, 256>>>(d_x, (int)total, 1.0f, 1);
    fill_bf16<<<((int)total+255)/256, 256>>>(d_z, (int)total, 1.0f, 2);
    fill_bf16<<<(D+255)/256, 256>>>(d_w, D, 1.0f, 3);
    cudaDeviceSynchronize();

    fused_rms_norm_gate(d_x, d_z, d_w, d_out, N, D, EPS);
    cudaDeviceSynchronize();

    cudaFree(d_x); cudaFree(d_z); cudaFree(d_w); cudaFree(d_out);
    return 0;
}
