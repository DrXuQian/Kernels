// Standalone cuBLAS BF16 GEMM benchmark
// Shape: M=1024, N=1024, K=1024
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

struct BenchResult {
    double median_ms;
    double min_ms;
    double max_ms;
    double tflops;
};

BenchResult benchmark_gemm(
    cublasHandle_t handle,
    int M, int N, int K,
    void* A, void* B, void* C,
    cudaDataType_t dtype,
    int warmup, int repeat,
    const char* label)
{
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, dtype, N,
            A, dtype, K,
            &beta,
            C, dtype, N,
            compute, CUBLAS_GEMM_DEFAULT));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure
    std::vector<float> times(repeat);
    for (int i = 0; i < repeat; i++) {
        cudaEvent_t start, end;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&end));
        CHECK_CUDA(cudaEventRecord(start));

        CHECK_CUBLAS(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, dtype, N,
            A, dtype, K,
            &beta,
            C, dtype, N,
            compute, CUBLAS_GEMM_DEFAULT));

        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));
        CHECK_CUDA(cudaEventElapsedTime(&times[i], start, end));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(end));
    }

    // Sort for median
    std::sort(times.begin(), times.end());
    double median = times[repeat / 2];
    double min_t = times[0];
    double max_t = times[repeat - 1];
    double flops = 2.0 * M * N * K;
    double tflops = flops / (median / 1000.0) / 1e12;

    printf("\n============================================================\n");
    printf("  %s\n", label);
    printf("============================================================\n");
    printf("  M=%d, N=%d, K=%d\n", M, N, K);
    printf("  Median : %.4f ms\n", median);
    printf("  Min    : %.4f ms\n", min_t);
    printf("  Max    : %.4f ms\n", max_t);
    printf("  TFLOPS : %.4f\n", tflops);
    printf("============================================================\n");

    return {median, min_t, max_t, tflops};
}

int main(int argc, char* argv[]) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int warmup = 50, repeat = 200;

    // Print GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shape: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Warmup=%d, Repeat=%d\n\n", warmup, repeat);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    // Use TF32 for FP32 accumulation but BF16 inputs
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    size_t size = (size_t)M * K;
    size_t size_out = (size_t)M * N;

    // === BF16 GEMM ===
    {
        nv_bfloat16 *A, *B, *C;
        CHECK_CUDA(cudaMalloc(&A, size * sizeof(nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&B, (size_t)K * N * sizeof(nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&C, size_out * sizeof(nv_bfloat16)));

        // Init with random (zero is fine for benchmark)
        CHECK_CUDA(cudaMemset(A, 0x3C, size * sizeof(nv_bfloat16)));  // ~1.0 in bf16
        CHECK_CUDA(cudaMemset(B, 0x3C, (size_t)K * N * sizeof(nv_bfloat16)));
        CHECK_CUDA(cudaMemset(C, 0, size_out * sizeof(nv_bfloat16)));

        auto r_bf16 = benchmark_gemm(handle, M, N, K, A, B, C,
            CUDA_R_16BF, warmup, repeat, "BF16 GEMM (cuBLAS)");

        CHECK_CUDA(cudaFree(A));
        CHECK_CUDA(cudaFree(B));
        CHECK_CUDA(cudaFree(C));
    }

    // === FP16 GEMM ===
    {
        half *A, *B, *C;
        CHECK_CUDA(cudaMalloc(&A, size * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&B, (size_t)K * N * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&C, size_out * sizeof(half)));

        CHECK_CUDA(cudaMemset(A, 0x3C, size * sizeof(half)));
        CHECK_CUDA(cudaMemset(B, 0x3C, (size_t)K * N * sizeof(half)));
        CHECK_CUDA(cudaMemset(C, 0, size_out * sizeof(half)));

        auto r_fp16 = benchmark_gemm(handle, M, N, K, A, B, C,
            CUDA_R_16F, warmup, repeat, "FP16 GEMM (cuBLAS)");

        CHECK_CUDA(cudaFree(A));
        CHECK_CUDA(cudaFree(B));
        CHECK_CUDA(cudaFree(C));
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
