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

    // Build full window: [state[0], state[1], ..., state[ks-2], new_x]
    float buf[8] = {0};
    for (int i = 0; i < kernel_size - 1; i++)
        buf[i] = __bfloat162float(sp[i]);
    buf[kernel_size - 1] = __bfloat162float(x[(long long)b * dim + d]);

    // Compute convolution
    float out = bias_val;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < kernel_size) out += w[k] * buf[k];
    }
    y[(long long)b * dim + d] = __float2bfloat16(out);

    // Update state: shift left, append new x
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
