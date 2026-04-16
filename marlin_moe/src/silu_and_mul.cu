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

// Usage: ./silu_and_mul [num_tokens] [hidden_size]
// hidden_size = intermediate_size (N) of the MoE FFN
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1;
    int D = (argc > 2) ? atoi(argv[2]) : 5632;
    printf("bench silu_and_mul: tokens=%d hidden=%d\n", M, D);

    __half *d_in, *d_out;
    cudaMalloc(&d_in, (long long)M * 2 * D * sizeof(__half));
    cudaMalloc(&d_out, (long long)M * D * sizeof(__half));

    std::vector<__half> h(M * 2 * D);
    srand(42);
    for (auto& v : h) v = __float2half((float)rand() / RAND_MAX * 0.1f);
    cudaMemcpy(d_in, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice);

    silu_and_mul_launch(d_out, d_in, M, D, 0);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    printf("Done.\n");
    return 0;
}
#endif
