#include "ggml_compat.h"

template <int S_v, bool KDA>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            gated_delta_net_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            gated_delta_net_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// ============================================================================
#ifdef BENCH
#include <cstdlib>
#include <cmath>
#include <vector>

// Usage: ./gated_delta_net [n_tokens] [heads] [head_dim] [n_seqs]
// Runs the recurrent gated delta net kernel (prefill + decode).
// Note: this kernel is recurrent (sequential over tokens), same for prefill and decode.
int main(int argc, char** argv) {
    int n_tokens = (argc > 1) ? atoi(argv[1]) : 1;
    int H        = (argc > 2) ? atoi(argv[2]) : 64;
    int S_v      = (argc > 3) ? atoi(argv[3]) : 128;
    int n_seqs   = (argc > 4) ? atoi(argv[4]) : 1;
    printf("bench gated_delta_net: tokens=%d heads=%d dim=%d seqs=%d\n", n_tokens, H, S_v, n_seqs);

    // Contiguous layout: q,k shape (S_v, H, n_tokens, n_seqs), v shape (S_v, H, n_tokens, n_seqs)
    // Strides: sq1=S_v (head stride), sq2=S_v*H (token stride), sq3=S_v*H*n_tokens (seq stride)
    int64_t sq1 = 1, sq2 = H, sq3 = (int64_t)H * n_tokens;
    int64_t sv1 = S_v, sv2 = (int64_t)S_v * H, sv3 = (int64_t)S_v * H * n_tokens;
    int64_t sb1 = 1, sb2 = H, sb3 = (int64_t)H * n_tokens;

    long long qk_size = (long long)S_v * H * n_tokens * n_seqs;
    long long v_size  = (long long)S_v * H * n_tokens * n_seqs;
    long long gb_size = (long long)H * n_tokens * n_seqs;
    long long state_size = (long long)n_seqs * H * S_v * S_v;
    long long dst_size = v_size + state_size; // output + state

    float *d_q, *d_k, *d_v, *d_g, *d_beta, *d_state, *d_dst;
    cudaMalloc(&d_q, qk_size * sizeof(float));
    cudaMalloc(&d_k, qk_size * sizeof(float));
    cudaMalloc(&d_v, v_size * sizeof(float));
    cudaMalloc(&d_g, gb_size * sizeof(float));
    cudaMalloc(&d_beta, gb_size * sizeof(float));
    cudaMalloc(&d_state, state_size * sizeof(float));
    cudaMalloc(&d_dst, dst_size * sizeof(float));
    cudaMemset(d_state, 0, state_size * sizeof(float));
    cudaMemset(d_dst, 0, dst_size * sizeof(float));

    // Host init
    auto fill = [](float* d, long long n, float scale, unsigned seed) {
        srand(seed);
        std::vector<float> h(n);
        for (auto& v : h) v = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * scale;
        cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    };
    fill(d_q, qk_size, 0.1f, 1);
    fill(d_k, qk_size, 0.1f, 2);
    fill(d_v, v_size, 0.1f, 3);
    // g: log-sigmoid (negative)
    {
        srand(4);
        std::vector<float> h(gb_size);
        for (auto& v : h) { float x = ((float)rand()/RAND_MAX - 0.5f)*4.0f; v = logf(1.0f/(1.0f+expf(-x))); }
        cudaMemcpy(d_g, h.data(), gb_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    // beta: sigmoid (0,1)
    {
        srand(5);
        std::vector<float> h(gb_size);
        for (auto& v : h) { float x = ((float)rand()/RAND_MAX - 0.5f)*4.0f; v = 1.0f/(1.0f+expf(-x)); }
        cudaMemcpy(d_beta, h.data(), gb_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    float scale = 1.0f / sqrtf((float)S_v);

    launch_gated_delta_net<false>(
        d_q, d_k, d_v, d_g, d_beta, d_state, d_dst,
        S_v, H, n_tokens, n_seqs,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
        /*neqk1=*/H, /*rq3=*/1,
        scale, 0);
    cudaDeviceSynchronize();

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_g); cudaFree(d_beta); cudaFree(d_state); cudaFree(d_dst);
    printf("Done.\n");
    return 0;
}
#endif
