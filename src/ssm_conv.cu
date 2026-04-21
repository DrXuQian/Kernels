#include "ggml_compat.h"


template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            ssm_conv_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                       dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9 right now.");
    }
}

// ============================================================================
#ifdef BENCH
#include <cstdlib>
#include <cmath>
#include <vector>

// Usage: ./ssm_conv [n_tokens] [d_inner] [d_conv] [n_seqs]
// d_inner = CONV_DIM (e.g. 12288 for Qwen3.5)
// d_conv = kernel size (e.g. 4)
int main(int argc, char** argv) {
    int n_t   = (argc > 1) ? atoi(argv[1]) : 1;
    int nr    = (argc > 2) ? atoi(argv[2]) : 12288;
    int nc    = (argc > 3) ? atoi(argv[3]) : 4;
    int n_s   = (argc > 4) ? atoi(argv[4]) : 1;
    printf("bench ssm_conv: tokens=%d d_inner=%d d_conv=%d seqs=%d\n", n_t, nr, nc, n_s);

    // src0: conv_x shape (d_conv + n_t - 1, d_inner, n_s) — padded input
    // src1: weight shape (d_conv, d_inner)
    // dst:  output shape (d_inner, n_t, n_s)
    int64_t x_cols = nc + n_t - 1; // d_conv - 1 history + n_t new tokens
    long long x_size = (long long)x_cols * nr * n_s;
    long long w_size = (long long)nc * nr;
    long long y_size = (long long)nr * n_t * n_s;

    // Strides in bytes
    int src0_nb0 = sizeof(float);                      // contiguous innermost
    int src0_nb1 = x_cols * sizeof(float);             // stride to next channel (d_inner)
    int src0_nb2 = x_cols * nr * sizeof(float);        // stride to next seq
    int src1_nb1 = nc * sizeof(float);                 // weight stride per channel
    int dst_nb0  = sizeof(float);                      // output innermost (d_inner)
    int dst_nb1  = nr * sizeof(float);                 // stride to next token
    int dst_nb2  = nr * n_t * sizeof(float);           // stride to next seq

    float *d_x, *d_w, *d_y;
    cudaMalloc(&d_x, x_size * sizeof(float));
    cudaMalloc(&d_w, w_size * sizeof(float));
    cudaMalloc(&d_y, y_size * sizeof(float));

    srand(42);
    std::vector<float> hx(x_size), hw(w_size);
    for (auto& v : hx) v = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
    for (auto& v : hw) v = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
    cudaMemcpy(d_x, hx.data(), x_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), w_size * sizeof(float), cudaMemcpyHostToDevice);

    ssm_conv_f32_cuda<false>(d_x, d_w,
        src0_nb0, src0_nb1, src0_nb2, src1_nb1,
        d_y, dst_nb0, dst_nb1, dst_nb2,
        nc, nr, n_t, n_s, 0);
    cudaDeviceSynchronize();

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_y);
    printf("Done.\n");
    return 0;
}
#endif
