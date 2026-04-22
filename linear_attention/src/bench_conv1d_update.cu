// Bench: causal_conv1d decode/update (Tri Dao, state-based single step)
// Extracted from https://github.com/Dao-AILab/causal-conv1d
// Usage: ./bench_conv1d_update [dim] [width] [batch]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "causal_conv1d.h"

// Forward declare
template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream);

#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)

int main(int argc, char** argv) {
    int dim   = (argc > 1) ? atoi(argv[1]) : 12288;
    int width = (argc > 2) ? atoi(argv[2]) : 4;
    int batch = (argc > 3) ? atoi(argv[3]) : 1;
    printf("bench conv1d_update: dim=%d width=%d batch=%d\n", dim, width, batch);

    using T = __nv_bfloat16;
    int conv_state_len = width - 1;

    T *d_x, *d_w, *d_bias, *d_out, *d_conv_state;
    CHECK(cudaMalloc(&d_x, (long long)batch * dim * sizeof(T)));
    CHECK(cudaMalloc(&d_w, (long long)dim * width * sizeof(T)));
    CHECK(cudaMalloc(&d_bias, dim * sizeof(T)));
    CHECK(cudaMalloc(&d_out, (long long)batch * dim * sizeof(T)));
    CHECK(cudaMalloc(&d_conv_state, (long long)batch * dim * conv_state_len * sizeof(T)));

    srand(42);
    auto fill = [](T* d, long long n) {
        std::vector<T> h(n);
        for (auto& v : h) v = __float2bfloat16(((float)rand()/RAND_MAX - 0.5f) * 0.2f);
        cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    };
    fill(d_x, (long long)batch * dim);
    fill(d_w, (long long)dim * width);
    fill(d_bias, dim);
    fill(d_conv_state, (long long)batch * dim * conv_state_len);

    // Setup params — decode: x is [batch, dim, 1], conv_state is [batch, dim, width-1]
    ConvParamsBase params = {};
    params.batch = batch;
    params.dim = dim;
    params.seqlen = 1;
    params.width = width;
    params.silu_activation = false;
    params.x_ptr = d_x;
    params.weight_ptr = d_w;
    params.bias_ptr = d_bias;
    params.out_ptr = d_out;
    params.conv_state_ptr = d_conv_state;
    params.cache_seqlens = nullptr;   // linear (non-circular) state
    params.conv_state_indices_ptr = nullptr;
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;

    // x strides: [batch, dim] (seqlen=1 so x_l_stride doesn't matter)
    params.x_batch_stride = dim;
    params.x_c_stride = 1;
    params.x_l_stride = 1;
    params.weight_c_stride = width;
    params.weight_width_stride = 1;
    params.out_batch_stride = dim;
    params.out_c_stride = 1;
    params.out_l_stride = 1;
    params.conv_state_len = conv_state_len;
    params.conv_state_batch_stride = dim * conv_state_len;
    params.conv_state_c_stride = conv_state_len;
    params.conv_state_l_stride = 1;

    causal_conv1d_update_cuda<T, T>(params, 0);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    printf("Done.\n");
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_bias); cudaFree(d_out); cudaFree(d_conv_state);
    return 0;
}
