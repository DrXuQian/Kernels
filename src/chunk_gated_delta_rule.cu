#include "deltanet.h"

// ============================================================================
// Kernel 3: Prefill gated delta rule (recurrent implementation)
//
// Each block handles one (batch, head) pair.
// Thread vid owns COLUMN vid of the (head_dim × head_dim) state matrix.
// No cross-thread reductions needed for core matrix-vector operations.
// ============================================================================

#define HEAD_DIM_MAX 128

static __device__ __forceinline__ float reduce_sum_128(float val, float* warp_buf, float* result_buf) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (lane == 0) warp_buf[warp_id] = val;
    __syncthreads();

    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < 4; i++) sum += warp_buf[i];
        result_buf[0] = sum;
    }
    __syncthreads();
    return result_buf[0];
}

__global__ void chunk_gated_delta_rule_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ state_out,
    int seq_len, int heads, int head_dim,
    float scale, bool use_l2norm)
{
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh % heads;
    const int vid = threadIdx.x;

    float state[HEAD_DIM_MAX];
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_MAX; k++)
        state[k] = 0.0f;

    __shared__ float s_vec[HEAD_DIM_MAX];
    __shared__ float s_warp[4];
    __shared__ float s_result[1];
    __shared__ float s_scalar[2];

    for (int t = 0; t < seq_len; t++) {
        const long long bhd_base = (long long)b * seq_len * heads * head_dim
                                 + (long long)t * heads * head_dim
                                 + (long long)h * head_dim;
        const long long bh_base  = (long long)b * seq_len * heads
                                 + (long long)t * heads + h;

        s_vec[vid] = __bfloat162float(K[bhd_base + vid]);
        if (vid == 0) {
            s_scalar[0] = g[bh_base];
            s_scalar[1] = beta[bh_base];
        }
        __syncthreads();

        if (use_l2norm) {
            float norm = reduce_sum_128(s_vec[vid] * s_vec[vid], s_warp, s_result);
            norm = sqrtf(norm + 1e-6f);
            s_vec[vid] /= norm;
            __syncthreads();
        }

        const float g_val = s_scalar[0];
        const float beta_val = s_scalar[1];
        const float v_val = __bfloat162float(V[bhd_base + vid]);

        const float decay = expf(g_val);
        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            state[k] *= decay;

        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            dot += state[k] * s_vec[k];

        const float v_new = beta_val * (v_val - dot);

        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            state[k] += s_vec[k] * v_new;

        __syncthreads();
        s_vec[vid] = __bfloat162float(Q[bhd_base + vid]);
        __syncthreads();

        if (use_l2norm) {
            float norm = reduce_sum_128(s_vec[vid] * s_vec[vid], s_warp, s_result);
            norm = sqrtf(norm + 1e-6f);
            s_vec[vid] /= norm;
            __syncthreads();
        }

        float out_val = 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            out_val += state[k] * s_vec[k];
        out_val *= scale;

        output[bhd_base + vid] = __float2bfloat16(out_val);
        __syncthreads();
    }

    if (state_out) {
        float* sp = state_out + (long long)b * heads * head_dim * head_dim
                              + (long long)h * head_dim * head_dim;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            sp[k * head_dim + vid] = state[k];
    }
}

void chunk_gated_delta_rule(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const float* g, const float* beta,
    __nv_bfloat16* output, float* state_out,
    int batch, int seq_len, int heads, int head_dim, int /*chunk_size*/,
    bool use_l2norm, cudaStream_t stream)
{
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = batch * heads;
    chunk_gated_delta_rule_kernel<<<num_blocks, head_dim, 0, stream>>>(
        Q, K, V, g, beta, output, state_out,
        seq_len, heads, head_dim, scale, use_l2norm);
}

// ============================================================================
#ifdef BENCH
#include "bench_utils.h"

// Usage: ./chunk_gated_delta_rule [seq_len]
int main(int argc, char** argv) {
    int seq = (argc > 1) ? atoi(argv[1]) : 3823;
    const int BATCH = 1, HEADS = 64, HD = 128, CS = 64;
    printf("bench K3: chunk_gated_delta_rule  seq=%d heads=%d dim=%d\n", seq, HEADS, HD);

    long long qkv_n = (long long)BATCH * seq * HEADS * HD;
    long long gb_n  = (long long)BATCH * seq * HEADS;
    long long st_n  = (long long)BATCH * HEADS * HD * HD;

    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_out;
    float *d_g, *d_beta, *d_state;
    BENCH_CHECK(cudaMalloc(&d_Q, qkv_n * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_K, qkv_n * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_V, qkv_n * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_out, qkv_n * sizeof(__nv_bfloat16)));
    BENCH_CHECK(cudaMalloc(&d_g, gb_n * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_beta, gb_n * sizeof(float)));
    BENCH_CHECK(cudaMalloc(&d_state, st_n * sizeof(float)));
    BENCH_CHECK(cudaMemset(d_state, 0, st_n * sizeof(float)));

    host_rand_bf16(d_Q, qkv_n, 0.3f, 1);
    host_rand_bf16(d_K, qkv_n, 0.3f, 2);
    host_rand_bf16(d_V, qkv_n, 0.3f, 3);
    host_rand_logsig(d_g, gb_n, 4);
    host_rand_sig(d_beta, gb_n, 5);

    chunk_gated_delta_rule(d_Q, d_K, d_V, d_g, d_beta,
                           d_out, d_state,
                           BATCH, seq, HEADS, HD, CS, true);
    BENCH_CHECK(cudaDeviceSynchronize());

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    cudaFree(d_g); cudaFree(d_beta); cudaFree(d_state);
    return 0;
}
#endif
