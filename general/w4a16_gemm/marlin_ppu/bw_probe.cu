// What bandwidth can this part ACTUALLY reach?
//
// Every %HBM figure in this directory divides by a nameplate peak (2700 GB/s for the PPU). Nameplate is not achievable:
// a real streaming kernel typically lands at 70-85% of it. We never checked. That matters, because our tuned W4A16
// GEMV reaches 59.2% "of peak" on N=14336, while UPSTREAM Marlin -- a GEMM being misused as a GEMV -- reaches 69.7% of
// peak on an RTX 5090. Either our denominator is wrong, or our kernel is much worse than we think. This tells us which.
//
// Deliberately dumb: grid-stride uint4 reads, one accumulate, no shared, no unroll games. The same shape as ggml's
// convert kernel, which the ppu-ptx skill says saturates HBM. If THIS cannot hit the nameplate, nothing can.
//
// Sweeps the buffer past LLC (the PPU's is 64-128 MB, measured) so the plateau is DRAM, not cache.
//
// build:  nvcc -O3 -std=c++17 -o bw_probe bw_probe.cu     (on the box: no -arch)
// run:    ./bw_probe [peak_GBs]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

// read-only stream: every 16-byte line is consumed, and the accumulator is written out so nothing is dead code.
__global__ void read_bw(const uint4* __restrict__ src, unsigned* __restrict__ sink, size_t n4) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (; i < n4; i += stride) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeef) sink[0] = acc;     // never true; keeps the loads alive
}

int main(int argc, char** argv) {
    const double peak = argc > 1 ? atof(argv[1]) : 2700.0;
    int cu = 0; CHECK(cudaDeviceGetAttribute(&cu, cudaDevAttrMultiProcessorCount, 0));
    printf("CUs=%d   nameplate peak=%.0f GB/s\n\n", cu, peak);
    printf("  %10s %12s %10s %8s   %s\n", "buffer", "time(us)", "GB/s", "% peak", "grid");

    unsigned* sink; CHECK(cudaMalloc(&sink, 4));
    const int THREADS = 256;

    for (size_t mb = 8; mb <= 1024; mb *= 2) {
        const size_t bytes = mb << 20, n4 = bytes / 16;
        uint4* buf;
        if (cudaMalloc(&buf, bytes)) { printf("  %8zu MB   alloc fail\n", mb); break; }
        CHECK(cudaMemset(buf, 1, bytes));

        // enough blocks to fill the machine several times over; grid-stride handles the rest
        int blocks = (int) ((n4 + THREADS - 1) / THREADS);
        if (blocks > cu * 32) blocks = cu * 32;

        for (int i = 0; i < 3; i++) read_bw<<<blocks, THREADS>>>(buf, sink, n4);
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        const int iters = bytes > (256u << 20) ? 5 : 20;
        cudaEventRecord(a);
        for (int i = 0; i < iters; i++) read_bw<<<blocks, THREADS>>>(buf, sink, n4);
        cudaEventRecord(b); CHECK(cudaEventSynchronize(b));
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;

        const double gbs = bytes / (ms * 1e-3) / 1e9;
        printf("  %8zu MB %10.2f %10.0f %7.1f%%   %d x %d\n", mb, ms * 1e3, gbs, 100.0 * gbs / peak, blocks, THREADS);

        cudaEventDestroy(a); cudaEventDestroy(b);
        CHECK(cudaFree(buf));
    }
    printf("\nThe plateau past the LLC is the achievable read bandwidth. Divide by THAT, not by the nameplate.\n");
    CHECK(cudaFree(sink));
    return 0;
}
