#include "deltanet.h"

// ============================================================================
// Kernel 1: Prefill causal conv1d
// Grid: (batch, ceildiv(dim, 256)), Block: (256)
// Each thread handles one channel across all timesteps.
// ============================================================================
__global__ void causal_conv1d_fn_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ state_out,
    int dim, int seq_len, int kernel_size)
{
    const int b = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    // Load weight for this channel: weight layout is (dim, 1, kernel_size)
    float w[8]; // max supported kernel_size
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        w[i] = (i < kernel_size) ? __bfloat162float(weight[d * kernel_size + i]) : 0.0f;
    }
    float bias_val = bias ? __bfloat162float(bias[d]) : 0.0f;

    // Pointers for this (batch, channel) pair
    const __nv_bfloat16* x_ptr = x + (long long)b * dim * seq_len + (long long)d * seq_len;
    __nv_bfloat16* y_ptr = y + (long long)b * dim * seq_len + (long long)d * seq_len;

    // Sliding window history buffer (holds last kernel_size values)
    float hist[8] = {0};

    for (int t = 0; t < seq_len; t++) {
        // Shift history left and insert new value
        #pragma unroll
        for (int i = 0; i < 7; i++)
            hist[i] = hist[i + 1];
        hist[kernel_size - 1] = __bfloat162float(x_ptr[t]);

        // Causal convolution: y[t] = bias + sum_k w[k] * x[t - (kernel_size-1) + k]
        float out = bias_val;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            if (k < kernel_size) out += w[k] * hist[k];
        }
        y_ptr[t] = __float2bfloat16(out);
    }

    // Save conv state: last kernel_size-1 input values
    if (state_out) {
        __nv_bfloat16* sp = state_out + (long long)b * dim * (kernel_size - 1) + (long long)d * (kernel_size - 1);
        for (int i = 0; i < kernel_size - 1; i++)
            sp[i] = __float2bfloat16(hist[i + 1]);
    }
}

void causal_conv1d_fn(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* y, __nv_bfloat16* state_out,
    int batch, int dim, int seq_len, int kernel_size,
    cudaStream_t stream)
{
    const int threads = 256;
    dim3 grid(batch, (dim + threads - 1) / threads);
    causal_conv1d_fn_kernel<<<grid, threads, 0, stream>>>(
        x, weight, bias, y, state_out, dim, seq_len, kernel_size);
}
