#pragma once
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define BENCH_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// Host-side random bf16 init → cudaMemcpy to device
static inline void host_rand_bf16(__nv_bfloat16* d_ptr, long long n, float scale, unsigned seed) {
    srand(seed);
    std::vector<__nv_bfloat16> buf(n);
    for (long long i = 0; i < n; i++) {
        float r = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        buf[i] = __float2bfloat16(r);
    }
    BENCH_CHECK(cudaMemcpy(d_ptr, buf.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

// Host-side log-sigmoid init (negative values for decay gates)
static inline void host_rand_logsig(float* d_ptr, long long n, unsigned seed) {
    srand(seed);
    std::vector<float> buf(n);
    for (long long i = 0; i < n; i++) {
        float r = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        buf[i] = logf(1.0f / (1.0f + expf(-r)));
    }
    BENCH_CHECK(cudaMemcpy(d_ptr, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

// Host-side sigmoid init (0,1) for beta
static inline void host_rand_sig(float* d_ptr, long long n, unsigned seed) {
    srand(seed);
    std::vector<float> buf(n);
    for (long long i = 0; i < n; i++) {
        float r = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        buf[i] = 1.0f / (1.0f + expf(-r));
    }
    BENCH_CHECK(cudaMemcpy(d_ptr, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

// Host-side random float init
static inline void host_rand_f32(float* d_ptr, long long n, float scale, unsigned seed) {
    srand(seed);
    std::vector<float> buf(n);
    for (long long i = 0; i < n; i++)
        buf[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    BENCH_CHECK(cudaMemcpy(d_ptr, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}
