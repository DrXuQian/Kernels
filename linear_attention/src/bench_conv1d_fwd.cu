// Bench: causal_conv1d prefill (Tri Dao, 128-bit vectorized)
// Extracted from https://github.com/Dao-AILab/causal-conv1d
// Usage: ./bench_conv1d_fwd [seq_len] [dim] [width] [batch]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "causal_conv1d.h"

// Forward declare
template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)

int main(int argc, char** argv) {
    int seq   = (argc > 1) ? atoi(argv[1]) : 3823;
    int dim   = (argc > 2) ? atoi(argv[2]) : 12288;
    int width = (argc > 3) ? atoi(argv[3]) : 4;
    int batch = (argc > 4) ? atoi(argv[4]) : 1;
    printf("bench conv1d_fwd: seq=%d dim=%d width=%d batch=%d\n", seq, dim, width, batch);

    using T = __nv_bfloat16;
    long long x_size = (long long)batch * dim * seq;
    long long w_size = (long long)dim * width;
    long long state_size = (long long)batch * dim * (width - 1);

    T *d_x, *d_w, *d_bias, *d_out, *d_final_states;
    CHECK(cudaMalloc(&d_x, x_size * sizeof(T)));
    CHECK(cudaMalloc(&d_w, w_size * sizeof(T)));
    CHECK(cudaMalloc(&d_bias, dim * sizeof(T)));
    CHECK(cudaMalloc(&d_out, x_size * sizeof(T)));
    CHECK(cudaMalloc(&d_final_states, state_size * sizeof(T)));

    // Host init
    srand(42);
    auto fill = [](T* d, long long n) {
        std::vector<T> h(n);
        for (auto& v : h) v = __float2bfloat16(((float)rand()/RAND_MAX - 0.5f) * 0.2f);
        cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    };
    fill(d_x, x_size);
    fill(d_w, w_size);
    fill(d_bias, dim);

    // Setup params (channel-first layout: x is [batch, dim, seq])
    ConvParamsBase params = {};
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seq;
    params.width = width;
    params.silu_activation = false;
    params.x_ptr = d_x;
    params.weight_ptr = d_w;
    params.bias_ptr = d_bias;
    params.out_ptr = d_out;
    params.conv_state_ptr = nullptr;
    params.cache_seqlens = nullptr;
    params.conv_state_indices_ptr = nullptr;
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = d_final_states;

    // Channel-first strides
    params.x_batch_stride = dim * seq;
    params.x_c_stride = seq;
    params.x_l_stride = 1;
    params.weight_c_stride = width;
    params.weight_width_stride = 1;
    params.out_batch_stride = dim * seq;
    params.out_c_stride = seq;
    params.out_l_stride = 1;
    params.final_states_batch_stride = dim * (width - 1);
    params.final_states_c_stride = width - 1;
    params.final_states_l_stride = 1;

    causal_conv1d_fwd_cuda<T, T>(params, 0);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    printf("Done.\n");
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_bias); cudaFree(d_out); cudaFree(d_final_states);
    return 0;
}
