// Bench K2: causal_conv1d_update (decode, single step)
// Usage: ./bench_k2
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include "deltanet.h"

constexpr int BATCH = 1;
constexpr int CONV_DIM = 12288;
constexpr int CONV_KERNEL = 4;

__global__ void fill_bf16(__nv_bfloat16* d, int n, float s, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState st; curand_init(seed, i, 0, &st);
    d[i] = __float2bfloat16(curand_normal(&st) * s);
}

int main() {
    printf("bench_k2: causal_conv1d_update  dim=%d  kernel=%d\n", CONV_DIM, CONV_KERNEL);

    __nv_bfloat16 *d_x, *d_y, *d_w, *d_b, *d_st;
    cudaMalloc(&d_x, (long long)BATCH * CONV_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_y, (long long)BATCH * CONV_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_w, (long long)CONV_DIM * CONV_KERNEL * sizeof(__nv_bfloat16));
    cudaMalloc(&d_b, CONV_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_st, (long long)BATCH * CONV_DIM * (CONV_KERNEL - 1) * sizeof(__nv_bfloat16));

    fill_bf16<<<(CONV_DIM+255)/256, 256>>>(d_x, CONV_DIM, 0.5f, 1);
    fill_bf16<<<(CONV_DIM*CONV_KERNEL+255)/256, 256>>>(d_w, CONV_DIM*CONV_KERNEL, 0.1f, 2);
    fill_bf16<<<(CONV_DIM+255)/256, 256>>>(d_b, CONV_DIM, 0.01f, 3);
    fill_bf16<<<(CONV_DIM*(CONV_KERNEL-1)+255)/256, 256>>>(d_st, CONV_DIM*(CONV_KERNEL-1), 0.2f, 4);
    cudaDeviceSynchronize();

    causal_conv1d_update(d_x, d_st, d_w, d_b, d_y, BATCH, CONV_DIM, CONV_KERNEL);
    cudaDeviceSynchronize();

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b); cudaFree(d_st);
    return 0;
}
