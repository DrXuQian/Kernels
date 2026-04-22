// MoE sum kernel: aggregate topk expert outputs
// Extracted from vllm/csrc/moe/moe_align_sum_kernels.cu
#include "moe_compat.h"

namespace vllm { namespace moe {

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const scalar_t* __restrict__ input,  // [num_tokens, topk, d]
    const int d)
{
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        scalar_t x = 0;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k)
            x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
        out[token_idx * d + idx] = x;
    }
}

}} // namespace vllm::moe

// ---- Standalone launch ----
void moe_sum_launch(
    __half* out,            // [num_tokens, hidden]
    const __half* input,    // [num_tokens, topk, hidden]
    int num_tokens, int topk, int hidden_size,
    cudaStream_t stream)
{
    dim3 grid(num_tokens);
    dim3 block(min(hidden_size, 1024));

    switch (topk) {
        case 2: vllm::moe::moe_sum_kernel<__half, 2><<<grid, block, 0, stream>>>(out, input, hidden_size); break;
        case 4: vllm::moe::moe_sum_kernel<__half, 4><<<grid, block, 0, stream>>>(out, input, hidden_size); break;
        case 8: vllm::moe::moe_sum_kernel<__half, 8><<<grid, block, 0, stream>>>(out, input, hidden_size); break;
        default: fprintf(stderr, "moe_sum: unsupported topk=%d\n", topk); abort();
    }
}

// ============================================================================
#ifdef BENCH
#include <cstdlib>
#include <vector>
#include "bench_timer.h"

// Usage: ./moe_sum [num_tokens] [topk] [hidden_size] [--bench warmup iters]
int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int M = (argc > 1) ? atoi(argv[1]) : 1;
    int K = (argc > 2) ? atoi(argv[2]) : 8;
    int D = (argc > 3) ? atoi(argv[3]) : 5632;
    printf("bench moe_sum: tokens=%d topk=%d hidden=%d\n", M, K, D);

    __half *d_in, *d_out;
    cudaMalloc(&d_in, (long long)M * K * D * sizeof(__half));
    cudaMalloc(&d_out, (long long)M * D * sizeof(__half));

    std::vector<__half> h(M * K * D);
    srand(42);
    for (auto& v : h) v = __float2half((float)rand() / RAND_MAX * 0.1f);
    cudaMemcpy(d_in, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice);

    timer.run([&]() {
        moe_sum_launch(d_out, d_in, M, K, D, 0);
    });

    cudaFree(d_in); cudaFree(d_out);
    printf("Done.\n");
    return 0;
}
#endif
