// copy_bw.cu
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t err = (expr);                                            \
        if (err != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d: %s\n",                              \
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__ void copy_kernel(
        const uint4 * __restrict__ src,
        uint4       * __restrict__ dst,
        size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        dst[i] = src[i]; // 16 B load + 16 B store
    }
}

int main(int argc, char ** argv) {
    // 默认对应 12288 x 12288 BF16，约 288 MiB。
    size_t bytes = 2ULL * 12288 * 12288;

    // 也可以通过参数指定 MiB。
    if (argc > 1) {
        bytes = std::strtoull(argv[1], nullptr, 10) << 20;
    }

    if (bytes % sizeof(uint4) != 0) {
        std::fprintf(stderr, "bytes must be a multiple of 16\n");
        return 1;
    }

    uint4 * src = nullptr;
    uint4 * dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemset(src, 1, bytes));
    CUDA_CHECK(cudaMemset(dst, 0, bytes));

    const size_t n       = bytes / sizeof(uint4);
    const int    threads = 256;
    const int    blocks  = (int) ((n + threads - 1) / threads);

    for (int i = 0; i < 10; ++i) {
        copy_kernel<<<blocks, threads>>>(src, dst, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int iterations = 100;

    cudaEvent_t begin;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < iterations; ++i) {
        copy_kernel<<<blocks, threads>>>(src, dst, n);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, begin, end));

    const double ms_per_copy = total_ms / iterations;

    // 一次 copy 包含 bytes 读取和 bytes 写入。
    const double bandwidth =
        2.0 * bytes * iterations / (total_ms * 1.0e-3) / 1.0e9;

    std::printf("size:       %.2f MiB\n", bytes / 1048576.0);
    std::printf("time:       %.3f ms/copy\n", ms_per_copy);
    std::printf("bandwidth:  %.2f GB/s (read + write)\n", bandwidth);

    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    return 0;
}
