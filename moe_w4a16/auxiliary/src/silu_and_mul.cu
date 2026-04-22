// Fused SiLU activation + element-wise multiply kernel
// Computes: out[i] = SiLU(gate[i]) * up[i]
// Where input layout is [..., 2*d] with gate in first half, up in second half.
// Extracted from vllm/csrc/activation_kernels.cu
#include "moe_compat.h"

__device__ __forceinline__ __half silu_h(__half x) {
    float f = __half2float(x);
    return __float2half(f / (1.0f + expf(-f)));
}

__global__ void silu_and_mul_kernel(
    __half* __restrict__ out,          // [..., d]
    const __half* __restrict__ input,  // [..., 2*d]
    const int d)
{
    const __half* gate = input + (long long)blockIdx.x * 2 * d;
    const __half* up   = gate + d;
    __half* out_ptr    = out + (long long)blockIdx.x * d;

    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu_g = g / (1.0f + expf(-g));
        out_ptr[idx] = __float2half(silu_g * u);
    }
}

// ---- Standalone launch ----
void silu_and_mul_launch(
    __half* out,            // [num_tokens, d]
    const __half* input,    // [num_tokens, 2*d]
    int num_tokens, int d,
    cudaStream_t stream)
{
    dim3 grid(num_tokens);
    dim3 block(min(d, 1024));
    silu_and_mul_kernel<<<grid, block, 0, stream>>>(out, input, d);
}

// ============================================================================
#ifdef BENCH
#include <cstdlib>
#include <vector>
#include "bench_timer.h"

// Usage: ./silu_and_mul [num_tokens] [top_k] [hidden_size] [--bench warmup iters]
// Actual rows = num_tokens * top_k (each token replicated per expert)
int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int M = (argc > 1) ? atoi(argv[1]) : 1;
    int topk = (argc > 2) ? atoi(argv[2]) : 8;
    int D = (argc > 3) ? atoi(argv[3]) : 5632;
    int rows = M * topk;
    printf("bench silu_and_mul: tokens=%d top_k=%d rows=%d hidden=%d\n", M, topk, rows, D);

    __half *d_in, *d_out;
    cudaMalloc(&d_in, (long long)rows * 2 * D * sizeof(__half));
    cudaMalloc(&d_out, (long long)rows * D * sizeof(__half));

    std::vector<__half> h(rows * 2 * D);
    srand(42);
    for (auto& v : h) v = __float2half((float)rand() / RAND_MAX * 0.1f);
    cudaMemcpy(d_in, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice);

    timer.run([&]() {
        silu_and_mul_launch(d_out, d_in, rows, D, 0);
    });

    cudaFree(d_in); cudaFree(d_out);
    printf("Done.\n");
    return 0;
}
#endif
