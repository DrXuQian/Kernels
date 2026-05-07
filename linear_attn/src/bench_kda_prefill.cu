// Standalone bench for cuLA's chunked KDA (gated delta rule) prefill kernel
// Extracted from https://github.com/inclusionAI/cuLA
// Usage: ./bench_kda_prefill [seq_len] [num_heads] [head_dim] [num_seqs]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/arch/arch.h>

#include "kda/sm90/prefill_kernel.hpp"
#include "bench_timer.h"

#define CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int seq_len   = (argc > 1) ? atoi(argv[1]) : 3823;
    int num_heads = (argc > 2) ? atoi(argv[2]) : 64;
    int head_dim  = (argc > 3) ? atoi(argv[3]) : 128;
    int num_seqs  = (argc > 4) ? atoi(argv[4]) : 1;
    int total_seq = seq_len * num_seqs;

    printf("bench kda_prefill (cuLA SM90): seq=%d heads=%d dim=%d seqs=%d\n",
           seq_len, num_heads, head_dim, num_seqs);

    using bf16 = cute::bfloat16_t;

    // Q, K, V: [total_seq, num_heads, head_dim] in bf16
    long long qkv_size = (long long)total_seq * num_heads * head_dim;
    // alpha: [total_seq, num_heads, head_dim] in fp32
    long long alpha_size = (long long)total_seq * num_heads * head_dim;
    // beta: [total_seq, num_heads] in fp32
    long long beta_size = (long long)total_seq * num_heads;
    // state: [num_seqs, num_heads, head_dim, head_dim] in fp32
    long long state_size = (long long)num_seqs * num_heads * head_dim * head_dim;
    // cu_seqlens: [num_seqs + 1] in int32
    // workspace: generous buffer

    bf16 *d_q, *d_k, *d_v, *d_output;
    float *d_alpha, *d_beta, *d_state;
    int32_t *d_cu_seqlens;
    uint8_t *d_workspace;

    CHECK(cudaMalloc(&d_q, qkv_size * sizeof(bf16)));
    CHECK(cudaMalloc(&d_k, qkv_size * sizeof(bf16)));
    CHECK(cudaMalloc(&d_v, qkv_size * sizeof(bf16)));
    CHECK(cudaMalloc(&d_output, qkv_size * sizeof(bf16)));
    CHECK(cudaMalloc(&d_alpha, alpha_size * sizeof(float)));
    CHECK(cudaMalloc(&d_beta, beta_size * sizeof(float)));
    CHECK(cudaMalloc(&d_state, state_size * sizeof(float)));
    CHECK(cudaMalloc(&d_cu_seqlens, (num_seqs + 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_workspace, 128 * 1024 * 1024));  // 128MB workspace

    CHECK(cudaMemset(d_state, 0, state_size * sizeof(float)));
    CHECK(cudaMemset(d_output, 0, qkv_size * sizeof(bf16)));

    // Host init
    srand(42);
    auto rand_bf16 = [](bf16* d, long long n) {
        std::vector<bf16> h(n);
        for (auto& v : h)
            v = cute::bfloat16_t(((float)rand()/RAND_MAX - 0.5f) * 0.2f);
        cudaMemcpy(d, h.data(), n * sizeof(bf16), cudaMemcpyHostToDevice);
    };
    auto rand_f32 = [](float* d, long long n, float lo, float hi) {
        std::vector<float> h(n);
        for (auto& v : h) v = lo + (hi - lo) * (float)rand()/RAND_MAX;
        cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    };

    rand_bf16(d_q, qkv_size);
    rand_bf16(d_k, qkv_size);
    rand_bf16(d_v, qkv_size);
    // alpha: log-sigmoid decay (negative values, applied as exp(alpha))
    {
        std::vector<float> h(alpha_size);
        for (auto& v : h) { float x = ((float)rand()/RAND_MAX - 0.5f)*4.0f; v = logf(1.0f/(1.0f+expf(-x))); }
        cudaMemcpy(d_alpha, h.data(), alpha_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    // beta: sigmoid write strength
    rand_f32(d_beta, beta_size, 0.0f, 1.0f);

    // cu_seqlens: [0, seq_len, 2*seq_len, ...]
    std::vector<int32_t> h_cu(num_seqs + 1);
    for (int i = 0; i <= num_seqs; i++) h_cu[i] = i * seq_len;
    CHECK(cudaMemcpy(d_cu_seqlens, h_cu.data(), (num_seqs + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)head_dim);

    int sm_count = 0;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

    // Launch
    timer.run([&]() {
        kda::sm90::launch_kda_fwd_prefill_kernel<cutlass::arch::Sm90, bf16, bf16, float, float>(
            /*stream=*/0,
            d_output,
            d_state,       // output_state
            d_q, d_k, d_v,
            nullptr,       // input_state (null = zero init)
            d_alpha,
            d_beta,
            d_cu_seqlens,
            d_workspace,
            num_seqs,
            num_heads,
            head_dim,
            total_seq,
            scale,
            /*safe_gate=*/true,
            sm_count);
    });
    CHECK(cudaGetLastError());

    printf("Done.\n");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_output);
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_state);
    cudaFree(d_cu_seqlens); cudaFree(d_workspace);
    return 0;
}
