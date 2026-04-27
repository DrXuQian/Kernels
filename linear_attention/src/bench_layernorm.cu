// Standalone LayerNorm benchmark (OneFlow kernel)
// Usage: ./bench_layernorm --batch 13824 --embed 1152 --dtype float16
//
// Build: nvcc -O2 -std=c++17 -arch=sm_80 bench_layernorm.cu -o bench_layernorm

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Include the OneFlow LayerNorm kernel (header-only)
#include "oneflow_layernorm.cuh"

#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)

template<typename T>
void run_layernorm(int batch, int embed) {
    using ComputeType = typename oneflow::cuda::layer_norm::DefaultComputeType<T>::type;

    size_t input_size = (size_t)batch * embed;
    size_t input_bytes = input_size * sizeof(T);
    size_t stats_bytes = batch * sizeof(ComputeType);

    // Allocate
    T *d_input, *d_output;
    ComputeType *d_mean, *d_inv_var;
    CHECK(cudaMalloc(&d_input, input_bytes));
    CHECK(cudaMalloc(&d_output, input_bytes));
    CHECK(cudaMalloc(&d_mean, stats_bytes));
    CHECK(cudaMalloc(&d_inv_var, stats_bytes));

    // Init: deterministic pattern
    std::vector<T> h_input(input_size);
    for (size_t i = 0; i < input_size; i++) {
        float val = 0.01f * ((int)(i % 101) - 50);  // [-0.5, 0.5]
        if constexpr (std::is_same_v<T, half>)
            h_input[i] = __float2half(val);
        else
            h_input[i] = static_cast<T>(val);
    }
    CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, input_bytes));
    CHECK(cudaMemset(d_mean, 0, stats_bytes));
    CHECK(cudaMemset(d_inv_var, 0, stats_bytes));

    // Setup load/store
    oneflow::cuda::layer_norm::DirectLoad<T, ComputeType> load(d_input, embed);
    oneflow::cuda::layer_norm::DirectStore<ComputeType, T> store(d_output, embed);

    double epsilon = 1e-5;

    // Single kernel launch
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    cudaError_t err = oneflow::cuda::layer_norm::DispatchLayerNorm<
        decltype(load), decltype(store), ComputeType>(
        0, load, store, batch, embed, epsilon, d_mean, d_inv_var);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    if (err != cudaSuccess) {
        fprintf(stderr, "LayerNorm dispatch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Read back and verify non-zero
    std::vector<T> h_output(input_size);
    CHECK(cudaMemcpy(h_output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost));

    int nonzero = 0;
    float min_val = 1e30f, max_val = -1e30f;
    for (size_t i = 0; i < input_size; i++) {
        float v;
        if constexpr (std::is_same_v<T, half>)
            v = __half2float(h_output[i]);
        else
            v = static_cast<float>(h_output[i]);
        if (v != 0.0f) nonzero++;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    // Bandwidth: read input + write output + read/write stats
    size_t total_bytes = input_bytes * 2 + stats_bytes * 2;
    float bw_gb = total_bytes / (ms / 1000.0f) / 1e9f;

    printf("batch=%d embed=%d dtype=%s\n", batch, embed,
           std::is_same_v<T, half> ? "float16" : std::is_same_v<T, float> ? "float32" : "unknown");
    printf("kernel time: %.3f us\n", ms * 1000.0f);
    printf("bandwidth:   %.1f GB/s\n", bw_gb);
    printf("output: nonzero=%d/%zu min=%.6f max=%.6f\n",
           nonzero, input_size, min_val, max_val);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_mean); cudaFree(d_inv_var);
}

int main(int argc, char** argv) {
    int batch = 13824;
    int embed = 1152;
    const char* dtype = "float16";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--embed") == 0 && i+1 < argc) embed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dtype") == 0 && i+1 < argc) dtype = argv[++i];
        else { fprintf(stderr, "Usage: %s [--batch N] [--embed N] [--dtype float16|float32]\n", argv[0]); return 1; }
    }

    if (strcmp(dtype, "float16") == 0 || strcmp(dtype, "fp16") == 0 || strcmp(dtype, "half") == 0) {
        run_layernorm<half>(batch, embed);
    } else if (strcmp(dtype, "float32") == 0 || strcmp(dtype, "fp32") == 0 || strcmp(dtype, "float") == 0) {
        run_layernorm<float>(batch, embed);
    } else {
        fprintf(stderr, "Unsupported dtype: %s\n", dtype);
        return 1;
    }
    return 0;
}
