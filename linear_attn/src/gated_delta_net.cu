#include "ggml_compat.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

template <typename T>
__device__ __forceinline__ float gdn_to_float(T x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float gdn_to_float<half>(half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float gdn_to_float<nv_bfloat16>(nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T gdn_from_float(float x);

template <>
__device__ __forceinline__ float gdn_from_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ half gdn_from_float<half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ nv_bfloat16 gdn_from_float<nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename T, int S_v, bool KDA>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const T *   q,
                     const T *   k,
                     const T *   v,
                     const float * g,
                     const float * beta,
                     const float * curr_state,
                     T *         attn_data,
                     float *     state,
                     int64_t     H,
                     int64_t     n_tokens,
                     int64_t     n_seqs,
                     int64_t     sq1,
                     int64_t     sq2,
                     int64_t     sq3,
                     int64_t     sv1,
                     int64_t     sv2,
                     int64_t     sv3,
                     int64_t     sb1,
                     int64_t     sb2,
                     int64_t     sb3,
                     const uint3 neqk1_magic,
                     const uint3 rq3_magic,
                     float       scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

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
        const T * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const T * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const T * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

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
            k_reg[r] = gdn_to_float(k_t[i]);
            q_reg[r] = gdn_to_float(q_t[i]);
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
            float delta_col = (gdn_to_float(v_t[col]) - g_val * kv_col) * beta_val;

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
                attn_data[col] = gdn_from_float<T>(attn_col * scale);
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
            float delta_col = (gdn_to_float(v_t[col]) - kv_col) * beta_val;

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
                attn_data[col] = gdn_from_float<T>(attn_col * scale);
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

template <typename T, bool KDA>
static void launch_gated_delta_net(
        const T * q_d, const T * k_d, const T * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        T * out_d, float * state_out_d,
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
            gated_delta_net_cuda<T, 16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, out_d, state_out_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<T, 32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, out_d, state_out_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            gated_delta_net_cuda<T, 64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, out_d, state_out_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            gated_delta_net_cuda<T, 128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, out_d, state_out_d, H,
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
#include <cstring>
#include <string>
#include <vector>
#include "bench_timer.h"

template <typename T>
static T host_from_float(float x);

template <>
float host_from_float<float>(float x) {
    return x;
}

template <>
half host_from_float<half>(float x) {
    return __float2half(x);
}

template <>
nv_bfloat16 host_from_float<nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// Usage: ./gated_delta_net [n_tokens] [heads] [head_dim] [n_seqs] [--bench warmup iters]
// Runs the recurrent gated delta net kernel (prefill + decode).
// Note: this kernel is recurrent (sequential over tokens), same for prefill and decode.
template <typename T>
static int run_gated_delta_net(BenchTimer& timer, int n_tokens, int H, int S_v, int n_seqs, char const* dtype_name) {
    printf("bench gated_delta_net: tokens=%d heads=%d dim=%d seqs=%d dtype=%s\n", n_tokens, H, S_v, n_seqs, dtype_name);

    // Contiguous layout: q,k shape (S_v, H, n_tokens, n_seqs), v shape (S_v, H, n_tokens, n_seqs)
    // Strides: sq1=S_v (head stride), sq2=S_v*H (token stride), sq3=S_v*H*n_tokens (seq stride)
    int64_t sq1 = 1, sq2 = H, sq3 = (int64_t)H * n_tokens;
    int64_t sv1 = S_v, sv2 = (int64_t)S_v * H, sv3 = (int64_t)S_v * H * n_tokens;
    int64_t sb1 = 1, sb2 = H, sb3 = (int64_t)H * n_tokens;

    long long qk_size = (long long)S_v * H * n_tokens * n_seqs;
    long long v_size  = (long long)S_v * H * n_tokens * n_seqs;
    long long gb_size = (long long)H * n_tokens * n_seqs;
    long long state_size = (long long)n_seqs * H * S_v * S_v;
    long long out_size = v_size;

    T *d_q, *d_k, *d_v, *d_out;
    float *d_g, *d_beta, *d_state, *d_state_out;
    cudaMalloc(&d_q, qk_size * sizeof(T));
    cudaMalloc(&d_k, qk_size * sizeof(T));
    cudaMalloc(&d_v, v_size * sizeof(T));
    cudaMalloc(&d_out, out_size * sizeof(T));
    cudaMalloc(&d_g, gb_size * sizeof(float));
    cudaMalloc(&d_beta, gb_size * sizeof(float));
    cudaMalloc(&d_state, state_size * sizeof(float));
    cudaMalloc(&d_state_out, state_size * sizeof(float));
    cudaMemset(d_state, 0, state_size * sizeof(float));
    cudaMemset(d_state_out, 0, state_size * sizeof(float));
    cudaMemset(d_out, 0, out_size * sizeof(T));

    // Host init
    auto fill_t = [](T* d, long long n, float scale, unsigned seed) {
        srand(seed);
        std::vector<T> h(n);
        for (auto& v : h) v = host_from_float<T>(((float)rand()/RAND_MAX - 0.5f) * 2.0f * scale);
        cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    };
    fill_t(d_q, qk_size, 0.1f, 1);
    fill_t(d_k, qk_size, 0.1f, 2);
    fill_t(d_v, v_size, 0.1f, 3);
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

    timer.run([&]() {
        launch_gated_delta_net<T, false>(
            d_q, d_k, d_v, d_g, d_beta, d_state, d_out, d_state_out,
            S_v, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            /*neqk1=*/H, /*rq3=*/1,
            scale, 0);
    });

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_g); cudaFree(d_beta); cudaFree(d_state); cudaFree(d_out); cudaFree(d_state_out);
    printf("Done.\n");
    return 0;
}

int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    std::vector<char*> positional;
    std::string dtype = "float";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dtype" && i + 1 < argc) {
            dtype = argv[++i];
        } else if (arg.rfind("--dtype=", 0) == 0) {
            dtype = arg.substr(strlen("--dtype="));
        } else {
            positional.push_back(argv[i]);
        }
    }

    int n_tokens = (positional.size() > 0) ? atoi(positional[0]) : 1;
    int H        = (positional.size() > 1) ? atoi(positional[1]) : 64;
    int S_v      = (positional.size() > 2) ? atoi(positional[2]) : 128;
    int n_seqs   = (positional.size() > 3) ? atoi(positional[3]) : 1;

    if (dtype == "fp16" || dtype == "half") {
        return run_gated_delta_net<half>(timer, n_tokens, H, S_v, n_seqs, "fp16");
    }
    if (dtype == "bf16" || dtype == "bfloat16") {
        return run_gated_delta_net<nv_bfloat16>(timer, n_tokens, H, S_v, n_seqs, "bf16");
    }
    if (dtype == "fp32" || dtype == "float") {
        return run_gated_delta_net<float>(timer, n_tokens, H, S_v, n_seqs, "float");
    }

    fprintf(stderr, "Unsupported dtype '%s'. Use fp16, bf16, or float.\n", dtype.c_str());
    return 1;
}
#endif
