// Standalone Marlin MoE W4A16 benchmark
// FP16 activation, INT4 weight, group_size=128
// Usage: ./bench_marlin_moe [num_tokens] [num_experts] [top_k] [K] [N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "scalar_type.hpp"
#include "bench_timer.h"

// Forward declare marlin_mm from dispatch.cu
namespace marlin_moe_wna16 {
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, void* sorted_token_ids,
               void* expert_ids, void* num_tokens_past_padded,
               void* topk_weights, int moe_block_size, int num_experts,
               int top_k, bool mul_topk_weights, int prob_m, int prob_n,
               int prob_k, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k,
               int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float);
}

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

    // Default: decode-like, Qwen3.5-MoE dimensions
    int M           = (argc > 1) ? atoi(argv[1]) : 1;      // num tokens
    int num_experts = (argc > 2) ? atoi(argv[2]) : 64;
    int top_k       = (argc > 3) ? atoi(argv[3]) : 8;
    int K           = (argc > 4) ? atoi(argv[4]) : 2048;    // hidden_size
    int N           = (argc > 5) ? atoi(argv[5]) : 5632;    // intermediate_size (must be % 64 == 0)

    int group_size  = 128;
    int moe_block_size = (M * top_k <= 8) ? 8 : 16;
    int num_groups = K / group_size;
    int pack_factor = 8; // 32 / 4bits

    printf("Marlin MoE W4A16 bench: M=%d experts=%d top_k=%d K=%d N=%d group=%d moe_block=%d\n",
           M, num_experts, top_k, K, N, group_size, moe_block_size);

    // Validate alignment
    if (K % 16 != 0 || N % 64 != 0) {
        fprintf(stderr, "K must be %%16==0, N must be %%64==0\n");
        return 1;
    }

    // ---- Allocate ----
    // A: FP16 [M, K]
    half* d_A;
    CHECK(cudaMalloc(&d_A, (long long)M * K * sizeof(half)));

    // B: packed U4 weights [num_experts, K/16, N/pack_factor * 16]
    // Marlin layout: [num_experts, K/tile_size, N*tile_size/pack_factor]
    // Simplified: total elements per expert = K * N / pack_factor (as uint32)
    long long B_expert_size = (long long)(K / 16) * (N * 16 / pack_factor);
    int4* d_B;
    CHECK(cudaMalloc(&d_B, (long long)num_experts * B_expert_size * sizeof(int4) / 4));
    // Actually Marlin stores as int4 (128-bit), expert_size in int4 units:
    long long B_bytes_per_expert = (long long)K * N / 2; // 4-bit per element = 0.5 byte
    CHECK(cudaFree(d_B));
    CHECK(cudaMalloc(&d_B, (long long)num_experts * B_bytes_per_expert));

    // C: FP16 [M * top_k, N]
    half* d_C;
    CHECK(cudaMalloc(&d_C, (long long)M * top_k * N * sizeof(half)));
    CHECK(cudaMemset(d_C, 0, (long long)M * top_k * N * sizeof(half)));

    // C_tmp: FP32 for reduction (0 = not used when use_atomic_add=true)
    float* d_C_tmp;
    CHECK(cudaMalloc(&d_C_tmp, sizeof(float))); // minimal

    // scales: FP16 [num_experts, num_groups, N]
    half* d_scales;
    CHECK(cudaMalloc(&d_scales, (long long)num_experts * num_groups * N * sizeof(half)));

    // Allocate dummy buffers for unused optional params (kernel may still deref)
    void *d_bias, *d_a_scales, *d_g_scales, *d_zp, *d_g_idx, *d_perm, *d_a_tmp;
    CHECK(cudaMalloc(&d_bias, 16));
    CHECK(cudaMalloc(&d_a_scales, 16));
    CHECK(cudaMalloc(&d_g_scales, 16));
    CHECK(cudaMalloc(&d_zp, 16));
    CHECK(cudaMalloc(&d_g_idx, (long long)num_experts * K * sizeof(int)));
    CHECK(cudaMemset(d_g_idx, 0, (long long)num_experts * K * sizeof(int)));
    CHECK(cudaMalloc(&d_perm, (long long)num_experts * K * sizeof(int)));
    CHECK(cudaMemset(d_perm, 0, (long long)num_experts * K * sizeof(int)));
    CHECK(cudaMalloc(&d_a_tmp, (long long)M * top_k * K * sizeof(half)));
    CHECK(cudaMemset(d_a_tmp, 0, (long long)M * top_k * K * sizeof(half)));

    // ---- MoE routing data ----
    // Simulate: each token selects top_k experts.
    // Marlin expects sorted_token_ids grouped by expert, padded to moe_block_size.
    // We send all M*top_k token slots to expert 0 (simplest valid routing for profiling).
    int num_moe_blocks = (M * top_k + moe_block_size - 1) / moe_block_size;
    int total_tokens_padded = num_moe_blocks * moe_block_size;

    std::vector<int32_t> h_sorted_ids(total_tokens_padded);
    std::vector<int32_t> h_expert_ids(num_moe_blocks);
    std::vector<float> h_topk_weights(M * top_k);
    int32_t h_num_tokens_padded[1] = {total_tokens_padded};

    // All valid tokens map to sequential indices, assigned to expert 0
    for (int i = 0; i < M * top_k; i++) {
        h_sorted_ids[i] = i;
        h_topk_weights[i] = 1.0f / top_k;
    }
    // Pad with invalid token ids (>= M * top_k → skipped by kernel)
    for (int i = M * top_k; i < total_tokens_padded; i++)
        h_sorted_ids[i] = M * top_k;

    // All blocks assigned to expert 0
    for (int i = 0; i < num_moe_blocks; i++)
        h_expert_ids[i] = 0;

    int32_t *d_sorted_ids, *d_expert_ids, *d_num_tokens_padded;
    float* d_topk_weights;
    CHECK(cudaMalloc(&d_sorted_ids, total_tokens_padded * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_expert_ids, h_expert_ids.size() * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_num_tokens_padded, sizeof(int32_t)));
    CHECK(cudaMalloc(&d_topk_weights, M * top_k * sizeof(float)));

    CHECK(cudaMemcpy(d_sorted_ids, h_sorted_ids.data(), total_tokens_padded * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_expert_ids, h_expert_ids.data(), h_expert_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_num_tokens_padded, h_num_tokens_padded, sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_topk_weights, h_topk_weights.data(), M * top_k * sizeof(float), cudaMemcpyHostToDevice));

    // Fill A, B, scales with random data (doesn't matter for profiling)
    {
        std::vector<half> h_A(M * K);
        for (int i = 0; i < M * K; i++) h_A[i] = __float2half(0.01f * (rand() % 100 - 50));
        CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));

        // Random bytes for packed weights
        std::vector<uint8_t> h_B(num_experts * B_bytes_per_expert);
        for (auto& b : h_B) b = rand() & 0xFF;
        CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice));

        // Scales: small positive FP16
        std::vector<half> h_s((long long)num_experts * num_groups * N);
        for (auto& s : h_s) s = __float2half(0.01f);
        CHECK(cudaMemcpy(d_scales, h_s.data(), h_s.size() * sizeof(half), cudaMemcpyHostToDevice));
    }

    // Workspace for locks
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int workspace_size = sms * 4;
    int* d_workspace;
    CHECK(cudaMalloc(&d_workspace, workspace_size * sizeof(int)));
    CHECK(cudaMemset(d_workspace, 0, workspace_size * sizeof(int)));

    // ---- Launch Marlin MoE ----
    timer.run([&]() {
        marlin_moe_wna16::marlin_mm(
            /*A=*/d_A,
            /*B=*/d_B,
            /*C=*/d_C,
            /*C_tmp=*/d_C_tmp,
            /*b_bias=*/d_bias,
            /*a_s=*/d_a_scales,
            /*b_s=*/d_scales,
            /*g_s=*/d_g_scales,
            /*zp=*/d_zp,
            /*g_idx=*/d_g_idx,
            /*perm=*/d_perm,
            /*a_tmp=*/d_a_tmp,
            /*sorted_token_ids=*/d_sorted_ids,
            /*expert_ids=*/d_expert_ids,
            /*num_tokens_past_padded=*/d_num_tokens_padded,
            /*topk_weights=*/d_topk_weights,
            /*moe_block_size=*/moe_block_size,
            /*num_experts=*/num_experts,
            /*top_k=*/top_k,
            /*mul_topk_weights=*/true,
            /*prob_m=*/M,
            /*prob_n=*/N,
            /*prob_k=*/K,
            /*workspace=*/d_workspace,
            /*a_type=*/vllm::kFloat16,
            /*b_type=*/vllm::kU4,
            /*c_type=*/vllm::kFloat16,
            /*s_type=*/vllm::kFloat16,
            /*has_bias=*/false,
            /*has_act_order=*/false,
            /*is_k_full=*/true,
            /*has_zp=*/false,
            /*num_groups=*/num_groups,
            /*group_size=*/group_size,
            /*dev=*/0,
            /*stream=*/0,
            /*thread_k=*/-1,  // auto
            /*thread_n=*/-1,  // auto
            /*sms=*/sms,
            /*blocks_per_sm=*/-1, // auto
            /*use_atomic_add=*/true,
            /*use_fp32_reduce=*/false,
            /*is_zp_float=*/false);
    });
    CHECK(cudaGetLastError());

    printf("Done.\n");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_tmp);
    cudaFree(d_scales); cudaFree(d_sorted_ids); cudaFree(d_expert_ids);
    cudaFree(d_num_tokens_padded); cudaFree(d_topk_weights);
    cudaFree(d_workspace);
    cudaFree(d_bias); cudaFree(d_a_scales); cudaFree(d_g_scales);
    cudaFree(d_zp); cudaFree(d_g_idx); cudaFree(d_perm); cudaFree(d_a_tmp);
    return 0;
}
