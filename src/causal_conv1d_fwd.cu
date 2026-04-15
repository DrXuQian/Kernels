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

    float w[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        w[i] = (i < kernel_size) ? __bfloat162float(weight[d * kernel_size + i]) : 0.0f;
    float bias_val = bias ? __bfloat162float(bias[d]) : 0.0f;

    const __nv_bfloat16* x_ptr = x + (long long)b * dim * seq_len + (long long)d * seq_len;
    __nv_bfloat16* y_ptr = y + (long long)b * dim * seq_len + (long long)d * seq_len;

    float hist[8] = {0};

    for (int t = 0; t < seq_len; t++) {
        #pragma unroll
        for (int i = 0; i < 7; i++)
            hist[i] = hist[i + 1];
        hist[kernel_size - 1] = __bfloat162float(x_ptr[t]);

        float out = bias_val;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            if (k < kernel_size) out += w[k] * hist[k];
        }
        y_ptr[t] = __float2bfloat16(out);
    }

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

// ============================================================================
#ifdef BENCH
#include "bench_utils.h"

// Usage: ./causal_conv1d_fwd [seq_len]
int main(int argc, char** argv) {
    int seq = (argc > 1) ? atoi(argv[1]) : 3823;
    const int BATCH = 1, DIM = 12288, KS = 4;
    printf("bench K1: causal_conv1d_fn  seq=%d dim=%d kernel=%d\n", seq, DIM, KS);

    long long io = (long long)BATCH * DIM * seq;
    long long st = (long long)BATCH * DIM * (KS - 1);

    __nv_bfloat16 *d_x, *d_y, *d_w, *d_b, *d_st;
    BENCH_CHECK(cudaMalloc(&d_x, io * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_y, io * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_w, (long long)DIM * KS * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_b, DIM * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_st, st * sizeof(__nv_bfloat16)));

    host_rand_bf16(d_x, io, 0.5f, 1);
    host_rand_bf16(d_w, (long long)DIM * KS, 0.1f, 2);
    host_rand_bf16(d_b, DIM, 0.01f, 3);

    causal_conv1d_fn(d_x, d_w, d_b, d_y, d_st, BATCH, DIM, seq, KS);
    BENCH_CHECK(cudaDeviceSynchronize());

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b); cudaFree(d_st);
    return 0;
}
#endif
