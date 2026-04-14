// Bench K4: fused_recurrent_gated_delta_rule (decode, single step)
// Usage: ./bench_k4
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include "deltanet.h"

constexpr int BATCH = 1;
constexpr int HEADS = 64;
constexpr int HEAD_DIM = 128;

__global__ void fill_bf16(__nv_bfloat16* d, int n, float s, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState st; curand_init(seed, i, 0, &st);
    d[i] = __float2bfloat16(curand_normal(&st) * s);
}
__global__ void fill_logsig(float* d, int n, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState st; curand_init(seed, i, 0, &st);
    d[i] = logf(1.0f / (1.0f + expf(-curand_normal(&st))));
}
__global__ void fill_sig(float* d, int n, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState st; curand_init(seed, i, 0, &st);
    d[i] = 1.0f / (1.0f + expf(-curand_normal(&st)));
}
__global__ void fill_f32(float* d, int n, float s, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState st; curand_init(seed, i, 0, &st);
    d[i] = curand_normal(&st) * s;
}

int main() {
    printf("bench_k4: fused_recurrent  heads=%d  dim=%d\n", HEADS, HEAD_DIM);

    int qkv_n = HEADS * HEAD_DIM;
    long long st_n = (long long)BATCH * HEADS * HEAD_DIM * HEAD_DIM;

    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_out;
    float *d_g, *d_beta, *d_state;
    cudaMalloc(&d_Q, qkv_n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K, qkv_n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V, qkv_n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, qkv_n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_g, HEADS * sizeof(float));
    cudaMalloc(&d_beta, HEADS * sizeof(float));
    cudaMalloc(&d_state, st_n * sizeof(float));

    fill_bf16<<<(qkv_n+255)/256, 256>>>(d_Q, qkv_n, 0.3f, 1);
    fill_bf16<<<(qkv_n+255)/256, 256>>>(d_K, qkv_n, 0.3f, 2);
    fill_bf16<<<(qkv_n+255)/256, 256>>>(d_V, qkv_n, 0.3f, 3);
    fill_logsig<<<(HEADS+255)/256, 256>>>(d_g, HEADS, 4);
    fill_sig<<<(HEADS+255)/256, 256>>>(d_beta, HEADS, 5);
    fill_f32<<<((int)st_n+255)/256, 256>>>(d_state, (int)st_n, 0.01f, 6);
    cudaDeviceSynchronize();

    fused_recurrent_gated_delta_rule(d_Q, d_K, d_V, d_g, d_beta,
                                     d_state, d_out,
                                     BATCH, HEADS, HEAD_DIM, true);
    cudaDeviceSynchronize();

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    cudaFree(d_g); cudaFree(d_beta); cudaFree(d_state);
    return 0;
}
