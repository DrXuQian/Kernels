#include "deltanet.h"

// ============================================================================
// Kernel 4: Decode gated delta rule (single-step recurrent)
//
// Same algorithm as kernel 3 but for a single timestep during autoregressive
// decode. State is read from and written back to global memory.
//
// Grid: (batch * heads), Block: (head_dim)
// Thread vid owns column vid of the state matrix.
// ============================================================================

#define HEAD_DIM_MAX 128

static __device__ __forceinline__ float reduce_sum_128_v2(float val, float* warp_buf, float* result_buf) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (lane == 0) warp_buf[warp_id] = val;
    __syncthreads();

    if (tid == 0) {
        float s = 0;
        for (int i = 0; i < 4; i++) s += warp_buf[i];
        result_buf[0] = s;
    }
    __syncthreads();
    return result_buf[0];
}

__global__ void fused_recurrent_gated_delta_rule_kernel(
    const __nv_bfloat16* __restrict__ Q,    // (batch, 1, heads, head_dim)
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ g,             // (batch, 1, heads)
    const float* __restrict__ beta,          // (batch, 1, heads)
    float* __restrict__ state,               // (batch, heads, head_dim, head_dim) in/out
    __nv_bfloat16* __restrict__ output,      // (batch, 1, heads, head_dim)
    int heads, int head_dim,
    float scale, bool use_l2norm)
{
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh % heads;
    const int vid = threadIdx.x;

    // Pointers — seq_len is 1 for decode
    const long long qkv_off = (long long)b * heads * head_dim + (long long)h * head_dim + vid;
    const long long gb_off  = (long long)b * heads + h;

    // Load state column from global memory
    float* sp = state + (long long)b * heads * head_dim * head_dim
                      + (long long)h * head_dim * head_dim;
    float st[HEAD_DIM_MAX];
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_MAX; k++)
        st[k] = sp[k * head_dim + vid];

    __shared__ float s_vec[HEAD_DIM_MAX];
    __shared__ float s_warp[4];
    __shared__ float s_result[1];
    __shared__ float s_scalar[2];

    // Load K
    s_vec[vid] = __bfloat162float(K[qkv_off]);
    if (vid == 0) {
        s_scalar[0] = g[gb_off];
        s_scalar[1] = beta[gb_off];
    }
    __syncthreads();

    if (use_l2norm) {
        float norm = reduce_sum_128_v2(s_vec[vid] * s_vec[vid], s_warp, s_result);
        norm = sqrtf(norm + 1e-6f);
        s_vec[vid] /= norm;
        __syncthreads();
    }

    const float g_val = s_scalar[0];
    const float beta_val = s_scalar[1];
    const float v_val = __bfloat162float(V[qkv_off]);

    // 1. Decay
    const float decay = expf(g_val);
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_MAX; k++)
        st[k] *= decay;

    // 2. h^T @ k
    float dot = 0.0f;
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_MAX; k++)
        dot += st[k] * s_vec[k];

    // 3. Delta
    const float v_new = beta_val * (v_val - dot);

    // 4. Rank-1 update
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_MAX; k++)
        st[k] += s_vec[k] * v_new;

    // 5. Load Q
    __syncthreads();
    s_vec[vid] = __bfloat162float(Q[qkv_off]);
    __syncthreads();

    if (use_l2norm) {
        float norm = reduce_sum_128_v2(s_vec[vid] * s_vec[vid], s_warp, s_result);
        norm = sqrtf(norm + 1e-6f);
        s_vec[vid] /= norm;
        __syncthreads();
    }

    // 6. Output
    float out_val = 0.0f;
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_MAX; k++)
        out_val += st[k] * s_vec[k];
    out_val *= scale;

    output[qkv_off] = __float2bfloat16(out_val);

    // 7. Write state back
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_MAX; k++)
        sp[k * head_dim + vid] = st[k];
}

void fused_recurrent_gated_delta_rule(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const float* g, const float* beta,
    float* state, __nv_bfloat16* output,
    int batch, int heads, int head_dim,
    bool use_l2norm, cudaStream_t stream)
{
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = batch * heads;

    fused_recurrent_gated_delta_rule_kernel<<<num_blocks, head_dim, 0, stream>>>(
        Q, K, V, g, beta, state, output,
        heads, head_dim, scale, use_l2norm);
}
