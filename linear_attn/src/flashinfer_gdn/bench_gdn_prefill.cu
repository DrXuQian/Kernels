// Standalone bench: FlashInfer chunk_gated_delta_rule prefill (SM90 WGMMA)
// Qwen3.5-122B DeltaNet GVA config:
//   num_k_heads=16, num_v_heads=64, head_dim=128
//
// Usage: ./bench_gdn_prefill [seqlen] [num_q_heads] [num_v_heads] [head_dim] [num_seqs] [--bench W I]
// Defaults: Qwen3.5-122B DeltaNet GVA (seqlen=3823, q=16, v=64, dim=128, seqs=1)
// Examples:
//   ./bench_gdn_prefill                                # defaults
//   ./bench_gdn_prefill 3823                           # just seqlen
//   ./bench_gdn_prefill 3823 16 64 128                 # GVA: q=16, v=64
//   ./bench_gdn_prefill 3823 32 32 128                 # uniform h=32 (TP=2)
//   ./bench_gdn_prefill 3823 64 64 128                 # uniform h=64
//   ./bench_gdn_prefill 3823 16 64 128 1 --bench 10 50 # with timing
//   ncu --set full ./bench_gdn_prefill 3823 16 64 128

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "bench_timer.h"

// Forward declare dispatch function
namespace flat {
void launch_gdn_prefill_bf16(
    cudaStream_t stream,
    nv_bfloat16* output, float* output_state,
    nv_bfloat16 const* q, nv_bfloat16 const* k, nv_bfloat16 const* v,
    float const* input_state, float const* alpha, float const* beta,
    int64_t const* cu_seqlens, uint8_t* workspace,
    int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads,
    int32_t num_v_heads, int32_t num_o_heads, int32_t head_size,
    int64_t total_seqlen, float scale, int32_t sm_count);
}

#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}  }while(0)

int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    // Usage: ./bench_gdn_prefill [seqlen] [num_q_heads] [num_v_heads] [head_dim] [num_seqs] [--bench W I]
    // Defaults: Qwen3.5-122B DeltaNet GVA config
    int total_seqlen  = (argc > 1) ? atoi(argv[1]) : 3823;
    int num_q_heads   = (argc > 2) ? atoi(argv[2]) : 16;
    int num_v_heads   = (argc > 3) ? atoi(argv[3]) : 64;
    int head_dim      = (argc > 4) ? atoi(argv[4]) : 128;
    int num_seqs      = (argc > 5) ? atoi(argv[5]) : 1;
    int num_k_heads   = num_q_heads;
    int num_o_heads   = (num_q_heads > num_v_heads) ? num_q_heads : num_v_heads;
    int num_sab_heads = num_o_heads;
    float scale       = 1.0f / sqrtf((float)head_dim);

    printf("bench gdn_prefill (FlashInfer SM90): seqlen=%d seqs=%d "
           "q_heads=%d k_heads=%d v_heads=%d dim=%d\n",
           total_seqlen, num_seqs, num_q_heads, num_k_heads, num_v_heads, head_dim);

    using T = nv_bfloat16;

    // Q: [total_seqlen, num_q_heads, head_dim]
    // K: [total_seqlen, num_k_heads, head_dim]
    // V: [total_seqlen, num_v_heads, head_dim]
    // O: [total_seqlen, num_o_heads, head_dim]
    int64_t q_size = (int64_t)total_seqlen * num_q_heads * head_dim;
    int64_t k_size = (int64_t)total_seqlen * num_k_heads * head_dim;
    int64_t v_size = (int64_t)total_seqlen * num_v_heads * head_dim;
    int64_t o_size = (int64_t)total_seqlen * num_o_heads * head_dim;
    // alpha, beta: [total_seqlen, num_sab_heads]
    int64_t gate_size = (int64_t)total_seqlen * num_sab_heads;
    // state: [num_seqs, num_sab_heads, head_dim, head_dim]
    int64_t state_size = (int64_t)num_seqs * num_sab_heads * head_dim * head_dim;

    T *d_q, *d_k, *d_v, *d_o;
    float *d_alpha, *d_beta, *d_state;
    int64_t *d_cu_seqlens;
    uint8_t *d_workspace;

    CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
    CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
    CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
    CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
    CHECK(cudaMalloc(&d_alpha, gate_size * sizeof(float)));
    CHECK(cudaMalloc(&d_beta, gate_size * sizeof(float)));
    CHECK(cudaMalloc(&d_state, state_size * sizeof(float)));
    CHECK(cudaMalloc(&d_cu_seqlens, (num_seqs + 1) * sizeof(int64_t)));
    size_t ws_size = 128 * 1024 * 1024;  // 128 MB
    CHECK(cudaMalloc(&d_workspace, ws_size));

    // Init
    {
        srand(42);
        auto fill_bf16 = [](T* d, int64_t n) {
            std::vector<T> h(n);
            for (auto& v : h) v = __float2bfloat16(((float)rand()/RAND_MAX - 0.5f) * 0.1f);
            cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        };
        fill_bf16(d_q, q_size);
        fill_bf16(d_k, k_size);
        fill_bf16(d_v, v_size);

        // alpha/beta: sigmoid-like values in [0,1]
        std::vector<float> h_gate(gate_size);
        for (auto& v : h_gate) v = 0.5f + 0.3f * ((float)rand()/RAND_MAX - 0.5f);
        CHECK(cudaMemcpy(d_alpha, h_gate.data(), gate_size * sizeof(float), cudaMemcpyHostToDevice));
        for (auto& v : h_gate) v = 0.5f + 0.3f * ((float)rand()/RAND_MAX - 0.5f);
        CHECK(cudaMemcpy(d_beta, h_gate.data(), gate_size * sizeof(float), cudaMemcpyHostToDevice));

        // cu_seqlens
        std::vector<int64_t> h_cu(num_seqs + 1);
        for (int i = 0; i <= num_seqs; i++)
            h_cu[i] = (int64_t)total_seqlen * i / num_seqs;
        CHECK(cudaMemcpy(d_cu_seqlens, h_cu.data(), h_cu.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    }

    CHECK(cudaMemset(d_o, 0, o_size * sizeof(T)));
    CHECK(cudaMemset(d_state, 0, state_size * sizeof(float)));
    CHECK(cudaMemset(d_workspace, 0, ws_size));

    int sm_count;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

    timer.run([&]() {
        flat::launch_gdn_prefill_bf16(
            0, d_o, d_state, d_q, d_k, d_v,
            nullptr,  // no initial state
            d_alpha, d_beta, d_cu_seqlens, d_workspace,
            num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads,
            head_dim, total_seqlen, scale, sm_count);
    });
    CHECK(cudaGetLastError());

    printf("Done.\n");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_state);
    cudaFree(d_cu_seqlens); cudaFree(d_workspace);
    return 0;
}
