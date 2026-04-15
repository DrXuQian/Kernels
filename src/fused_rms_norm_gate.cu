#include "deltanet.h"

// ============================================================================
// Kernel 5: Fused RMS Norm + Sigmoid Gate
//
// y = (x / rms) * weight * sigmoid(z)
// where rms = sqrt(mean(x^2) + eps)
//
// Grid: (N,), Block: (256)
// Each block handles one row of (N, D).
// ============================================================================
__global__ void fused_rms_norm_gate_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const __nv_bfloat16* x_row = x + (long long)row * D;
    const __nv_bfloat16* z_row = z + (long long)row * D;
    __nv_bfloat16* out_row = output + (long long)row * D;

    float local_sum_sq = 0.0f;
    for (int i = tid; i < D; i += nthreads) {
        float v = __bfloat162float(x_row[i]);
        local_sum_sq += v * v;
    }

    __shared__ float warp_sums[32];
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int nwarps = (nthreads + 31) / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    if (lane == 0) warp_sums[warp_id] = local_sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        warp_sums[0] = total;
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] / D + eps);

    for (int i = tid; i < D; i += nthreads) {
        float xv = __bfloat162float(x_row[i]) * inv_rms;
        float wv = __bfloat162float(weight[i]);
        float zv = __bfloat162float(z_row[i]);
        float gate = 1.0f / (1.0f + expf(-zv));
        out_row[i] = __float2bfloat16(xv * wv * gate);
    }
}

void fused_rms_norm_gate(
    const __nv_bfloat16* x, const __nv_bfloat16* z, const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int N, int D, float eps,
    cudaStream_t stream)
{
    const int block_size = 256;
    fused_rms_norm_gate_kernel<<<N, block_size, 0, stream>>>(
        x, z, weight, output, N, D, eps);
}

// ============================================================================
#ifdef BENCH
#include "bench_utils.h"

// Usage: ./fused_rms_norm_gate [N]
//   prefill: N = seq_len (default 3823)
//   decode:  N = 1
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 3823;
    const int D = 8192;
    const float EPS = 1e-6f;
    printf("bench K5: fused_rms_norm_gate  N=%d D=%d\n", N, D);

    long long total = (long long)N * D;
    __nv_bfloat16 *d_x, *d_z, *d_w, *d_out;
    BENCH_CHECK(cudaMalloc(&d_x, total * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_z, total * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_w, D * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_out, total * sizeof(__nv_bfloat16)));

    host_rand_bf16(d_x, total, 1.0f, 1);
    host_rand_bf16(d_z, total, 1.0f, 2);
    host_rand_bf16(d_w, D, 1.0f, 3);

    fused_rms_norm_gate(d_x, d_z, d_w, d_out, N, D, EPS);
    BENCH_CHECK(cudaDeviceSynchronize());

    cudaFree(d_x); cudaFree(d_z); cudaFree(d_w); cudaFree(d_out);
    return 0;
}
#endif
