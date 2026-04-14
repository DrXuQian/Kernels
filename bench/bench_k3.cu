// Bench K3: chunk_gated_delta_rule (prefill)
// Usage: ./bench_k3 [seq_len]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include "deltanet.h"

constexpr int BATCH = 1;
constexpr int HEADS = 64;
constexpr int HEAD_DIM = 128;
constexpr int CHUNK_SIZE = 64;

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
    float x = curand_normal(&st);
    d[i] = logf(1.0f / (1.0f + expf(-x)));
}
__global__ void fill_sig(float* d, int n, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState st; curand_init(seed, i, 0, &st);
    d[i] = 1.0f / (1.0f + expf(-curand_normal(&st)));
}

int main(int argc, char** argv) {
    int seq = (argc > 1) ? atoi(argv[1]) : 3823;
    printf("bench_k3: chunk_gated_delta_rule  seq=%d  heads=%d  dim=%d\n", seq, HEADS, HEAD_DIM);

    long long qkv_n = (long long)BATCH * seq * HEADS * HEAD_DIM;
    long long gb_n  = (long long)BATCH * seq * HEADS;
    long long st_n  = (long long)BATCH * HEADS * HEAD_DIM * HEAD_DIM;

    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_out;
    float *d_g, *d_beta, *d_state;
    cudaMalloc(&d_Q, qkv_n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K, qkv_n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V, qkv_n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, qkv_n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_g, gb_n * sizeof(float));
    cudaMalloc(&d_beta, gb_n * sizeof(float));
    cudaMalloc(&d_state, st_n * sizeof(float));
    cudaMemset(d_state, 0, st_n * sizeof(float));

    fill_bf16<<<((int)qkv_n+255)/256, 256>>>(d_Q, (int)qkv_n, 0.3f, 1);
    fill_bf16<<<((int)qkv_n+255)/256, 256>>>(d_K, (int)qkv_n, 0.3f, 2);
    fill_bf16<<<((int)qkv_n+255)/256, 256>>>(d_V, (int)qkv_n, 0.3f, 3);
    fill_logsig<<<((int)gb_n+255)/256, 256>>>(d_g, (int)gb_n, 4);
    fill_sig<<<((int)gb_n+255)/256, 256>>>(d_beta, (int)gb_n, 5);
    cudaDeviceSynchronize();

    chunk_gated_delta_rule(d_Q, d_K, d_V, d_g, d_beta,
                           d_out, d_state,
                           BATCH, seq, HEADS, HEAD_DIM, CHUNK_SIZE, true);
    cudaDeviceSynchronize();

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    cudaFree(d_g); cudaFree(d_beta); cudaFree(d_state);
    return 0;
}
