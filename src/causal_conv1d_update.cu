#include "deltanet.h"

// ============================================================================
// Kernel 2: Decode conv1d update (single timestep)
// Grid: (batch, ceildiv(dim, 256)), Block: (256)
// ============================================================================
__global__ void causal_conv1d_update_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    int dim, int kernel_size)
{
    const int b = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    float w[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        w[i] = (i < kernel_size) ? __bfloat162float(weight[d * kernel_size + i]) : 0.0f;
    float bias_val = bias ? __bfloat162float(bias[d]) : 0.0f;

    __nv_bfloat16* sp = state + (long long)b * dim * (kernel_size - 1) + (long long)d * (kernel_size - 1);

    float buf[8] = {0};
    for (int i = 0; i < kernel_size - 1; i++)
        buf[i] = __bfloat162float(sp[i]);
    buf[kernel_size - 1] = __bfloat162float(x[(long long)b * dim + d]);

    float out = bias_val;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < kernel_size) out += w[k] * buf[k];
    }
    y[(long long)b * dim + d] = __float2bfloat16(out);

    for (int i = 0; i < kernel_size - 2; i++)
        sp[i] = __float2bfloat16(buf[i + 1]);
    sp[kernel_size - 2] = __float2bfloat16(buf[kernel_size - 1]);
}

void causal_conv1d_update(
    const __nv_bfloat16* x, __nv_bfloat16* state,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* y,
    int batch, int dim, int kernel_size,
    cudaStream_t stream)
{
    const int threads = 256;
    dim3 grid(batch, (dim + threads - 1) / threads);
    causal_conv1d_update_kernel<<<grid, threads, 0, stream>>>(
        x, state, weight, bias, y, dim, kernel_size);
}

// ============================================================================
#ifdef BENCH
#include "bench_utils.h"

// Usage: ./causal_conv1d_update
int main() {
    const int BATCH = 1, DIM = 12288, KS = 4;
    printf("bench K2: causal_conv1d_update  dim=%d kernel=%d\n", DIM, KS);

    __nv_bfloat16 *d_x, *d_y, *d_w, *d_b, *d_st;
    BENCH_CHECK(cudaMalloc(&d_x, (long long)BATCH * DIM * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_y, (long long)BATCH * DIM * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_w, (long long)DIM * KS * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_b, DIM * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_st, (long long)BATCH * DIM * (KS - 1) * sizeof(__nv_bfloat16)));

    host_rand_bf16(d_x, (long long)BATCH * DIM, 0.5f, 1);
    host_rand_bf16(d_w, (long long)DIM * KS, 0.1f, 2);
    host_rand_bf16(d_b, DIM, 0.01f, 3);
    host_rand_bf16(d_st, (long long)BATCH * DIM * (KS - 1), 0.2f, 4);

    causal_conv1d_update(d_x, d_st, d_w, d_b, d_y, BATCH, DIM, KS);
    BENCH_CHECK(cudaDeviceSynchronize());

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b); cudaFree(d_st);
    return 0;
}
#endif
