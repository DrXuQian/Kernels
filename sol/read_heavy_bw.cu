// read_heavy_bw.cu
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        const cudaError_t err = (expr);                                      \
        if (err != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d: %s\n",                              \
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void init_kernel(uint4 * dst, size_t count) {
    size_t i      = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) blockDim.x * gridDim.x;

    for (; i < count; i += stride) {
        const uint32_t x = (uint32_t) i;

        dst[i] = make_uint4(
            x * 1664525u + 1013904223u,
            x * 22695477u + 1u,
            x * 1103515245u + 12345u,
            x * 214013u + 2531011u);
    }
}

// One block reads one complete BF16 matrix row and writes one uint32 result.
__global__ __launch_bounds__(128)
void read_heavy_kernel(
        const uint4 * __restrict__ src,
        uint32_t    * __restrict__ dst,
        int vecs_per_row) {
    const int row  = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const uint4 * row_src =
        src + (size_t) row * (size_t) vecs_per_row;

    uint32_t acc = 0;

    // For 12288 BF16 elements:
    // 24576 bytes / 16 bytes = 1536 uint4 loads per row.
    // With 128 threads, each thread performs 12 loads.
    for (int i = threadIdx.x; i < vecs_per_row; i += blockDim.x) {
        const uint4 value = row_src[i];

        // Keep every global load observable while adding little arithmetic.
        acc += value.x;
        acc += value.y;
        acc += value.z;
        acc += value.w;
    }

    // Warp reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ uint32_t warp_results[4];

    if (lane == 0) {
        warp_results[warp] = acc;
    }

    __syncthreads();

    // Reduce the four warp results and write only 4 bytes per row.
    if (warp == 0) {
        acc = lane < 4 ? warp_results[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if (lane == 0) {
            dst[row] = acc;
        }
    }
}

int main(int argc, char ** argv) {
    int rows       = 12288;
    int cols       = 12288;
    int iterations = 100;

    if (argc > 1) {
        rows = std::atoi(argv[1]);
    }
    if (argc > 2) {
        cols = std::atoi(argv[2]);
    }
    if (argc > 3) {
        iterations = std::atoi(argv[3]);
    }

    if (rows <= 0 || cols <= 0 || iterations <= 0) {
        std::fprintf(
            stderr,
            "usage: %s [rows] [cols] [iterations]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    // Eight BF16 values fit in one 16-byte uint4.
    if (cols % 8 != 0) {
        std::fprintf(stderr, "cols must be a multiple of 8\n");
        return EXIT_FAILURE;
    }

    constexpr int threads = 128;

    const int    vecs_per_row = cols / 8;
    const size_t vector_count =
        (size_t) rows * (size_t) vecs_per_row;

    const size_t input_bytes =
        (size_t) rows * (size_t) cols * sizeof(uint16_t);

    const size_t output_bytes =
        (size_t) rows * sizeof(uint32_t);

    uint4 *    src = nullptr;
    uint32_t * dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, input_bytes));
    CUDA_CHECK(cudaMalloc(&dst, output_bytes));

    int init_blocks =
        (int) ((vector_count + 255) / 256);

    if (init_blocks > 65535) {
        init_blocks = 65535;
    }

    init_kernel<<<init_blocks, 256>>>(src, vector_count);
    CUDA_CHECK(cudaMemset(dst, 0, output_bytes));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warm up.
    for (int i = 0; i < 10; ++i) {
        read_heavy_kernel<<<rows, threads>>>(
            src, dst, vecs_per_row);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t begin;
    cudaEvent_t end;

    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(begin));

    for (int i = 0; i < iterations; ++i) {
        read_heavy_kernel<<<rows, threads>>>(
            src, dst, vecs_per_row);
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, begin, end));

    // Read back the sink to ensure that the result is externally visible.
    std::vector<uint32_t> host_dst(rows);
    CUDA_CHECK(cudaMemcpy(
        host_dst.data(),
        dst,
        output_bytes,
        cudaMemcpyDeviceToHost));

    uint64_t checksum = 0;
    for (uint32_t value : host_dst) {
        checksum += value;
    }

    const double elapsed_s =
        (double) total_ms * 1.0e-3;

    const double total_read_bytes =
        (double) input_bytes * iterations;

    const double total_write_bytes =
        (double) output_bytes * iterations;

    const double read_gb_s =
        total_read_bytes / elapsed_s / 1.0e9;

    const double total_traffic_gb_s =
        (total_read_bytes + total_write_bytes) /
        elapsed_s / 1.0e9;

    std::printf("shape:              %d x %d BF16\n", rows, cols);
    std::printf("input:              %.2f MiB\n",
                input_bytes / 1048576.0);
    std::printf("output:             %.2f KiB\n",
                output_bytes / 1024.0);
    std::printf("read/write ratio:   %.1f : 1\n",
                (double) input_bytes / output_bytes);
    std::printf("blocks:             %d\n", rows);
    std::printf("threads/block:      %d\n", threads);
    std::printf("time:               %.3f ms/kernel\n",
                total_ms / iterations);
    std::printf("read bandwidth:     %.2f GB/s\n", read_gb_s);
    std::printf("total traffic:      %.2f GB/s\n",
                total_traffic_gb_s);
    std::printf("checksum:           %llu\n",
                (unsigned long long) checksum);

    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));

    return EXIT_SUCCESS;
}
