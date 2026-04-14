#include "deltanet.h"

// ============================================================================
// Kernel 3: Prefill gated delta rule
//
// Implements the recurrent form for correctness (sequential over timesteps).
// Each block handles one (batch, head) pair.
// Thread vid owns COLUMN vid of the (head_dim × head_dim) state matrix,
// i.e., state[k][vid] for k=0..head_dim-1.
//
// This avoids cross-thread reductions for the core matrix-vector operations.
//
// Performance note: a chunked implementation would be faster for long sequences
// by enabling intra-chunk parallelism, but this recurrent version is correct
// and simpler to verify.
// ============================================================================

#define HEAD_DIM_MAX 128

// Block-level sum reduction for 128 threads (4 warps)
static __device__ __forceinline__ float reduce_sum_128(float val, float* warp_buf, float* result_buf) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Warp-level reduction
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
    const int bh = blockIdx.x;        // (batch * heads)
    const int b = bh / heads;
    const int h = bh % heads;
    const int vid = threadIdx.x;      // 0..head_dim-1

    // State registers: column vid of the (head_dim × head_dim) matrix
    // state[kd] represents h_matrix[kd][vid]
    float state[HEAD_DIM_MAX];
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_MAX; k++)
        state[k] = 0.0f;

    // Shared memory for broadcasting vectors and scalars
    __shared__ float s_vec[HEAD_DIM_MAX];  // for k or q vector
    __shared__ float s_warp[4];            // for warp reduction
    __shared__ float s_result[1];          // for reduction result
    __shared__ float s_scalar[2];          // [g, beta]

    for (int t = 0; t < seq_len; t++) {
        const long long bhd_base = (long long)b * seq_len * heads * head_dim
                                 + (long long)t * heads * head_dim
                                 + (long long)h * head_dim;
        const long long bh_base  = (long long)b * seq_len * heads
                                 + (long long)t * heads + h;

        // --- Load K[t] into shared memory ---
        s_vec[vid] = __bfloat162float(K[bhd_base + vid]);
        if (vid == 0) {
            s_scalar[0] = g[bh_base];
            s_scalar[1] = beta[bh_base];
        }
        __syncthreads();

        // L2 normalize K if requested
        if (use_l2norm) {
            float norm = reduce_sum_128(s_vec[vid] * s_vec[vid], s_warp, s_result);
            norm = sqrtf(norm + 1e-6f);
            s_vec[vid] /= norm;
            __syncthreads();
        }

        const float g_val = s_scalar[0];
        const float beta_val = s_scalar[1];
        const float v_val = __bfloat162float(V[bhd_base + vid]);

        // --- 1. Decay state by exp(g) ---
        const float decay = expf(g_val);
        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            state[k] *= decay;

        // --- 2. h^T @ k: dot = sum_{kd} state[kd] * k[kd] ---
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            dot += state[k] * s_vec[k];

        // --- 3. Delta update: v_new = beta * (v - dot) ---
        const float v_new = beta_val * (v_val - dot);

        // --- 4. Rank-1 update: state[kd] += k[kd] * v_new ---
        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            state[k] += s_vec[k] * v_new;

        // --- 5. Load Q[t] into shared memory ---
        __syncthreads();
        s_vec[vid] = __bfloat162float(Q[bhd_base + vid]);
        __syncthreads();

        // L2 normalize Q if requested
        if (use_l2norm) {
            float norm = reduce_sum_128(s_vec[vid] * s_vec[vid], s_warp, s_result);
            norm = sqrtf(norm + 1e-6f);
            s_vec[vid] /= norm;
            __syncthreads();
        }

        // --- 6. Output: out[vid] = scale * sum_{kd} state[kd] * q[kd] ---
        float out_val = 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM_MAX; k++)
            out_val += state[k] * s_vec[k];
        out_val *= scale;

        output[bhd_base + vid] = __float2bfloat16(out_val);
        __syncthreads();
    }

    // Store final state to global memory (fp32)
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

    // Request enough registers per thread for the state array
    chunk_gated_delta_rule_kernel<<<num_blocks, head_dim, 0, stream>>>(
        Q, K, V, g, beta, output, state_out,
        seq_len, heads, head_dim, scale, use_l2norm);
}
