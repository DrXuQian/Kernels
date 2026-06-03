// Bench: causal_conv1d decode/update (Tri Dao, state-based single step)
// Extracted from https://github.com/Dao-AILab/causal-conv1d
// Usage: ./bench_conv1d_update [dim] [width] [batch] [--dtype fp16|bf16]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "causal_conv1d.h"
#include "bench_timer.h"

// Forward declare
template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream);

#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)

template <typename T>
T host_from_float(float value);

template <>
__nv_bfloat16 host_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
__half host_from_float<__half>(float value) {
    return __float2half(value);
}

struct Options {
    int dim = 12288;
    int width = 4;
    int batch = 1;
    std::string dtype = "bf16";
};

Options parse_options(int argc, char** argv) {
    Options opt;
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            opt.dtype = argv[++i];
        } else if (strncmp(argv[i], "--dtype=", 8) == 0) {
            opt.dtype = argv[i] + 8;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [dim] [width] [batch] [--dtype fp16|bf16] [--bench W I]\n", argv[0]);
            exit(0);
        } else {
            int value = atoi(argv[i]);
            if (positional == 0) opt.dim = value;
            else if (positional == 1) opt.width = value;
            else if (positional == 2) opt.batch = value;
            else {
                fprintf(stderr, "unexpected positional argument: %s\n", argv[i]);
                exit(1);
            }
            ++positional;
        }
    }
    if (opt.dtype != "fp16" && opt.dtype != "bf16") {
        fprintf(stderr, "dtype must be fp16 or bf16\n");
        exit(1);
    }
    return opt;
}

template <typename T>
int run(Options const& opt, BenchTimer& timer) {
    printf("bench conv1d_update: dim=%d width=%d batch=%d dtype=%s\n",
        opt.dim, opt.width, opt.batch, opt.dtype.c_str());

    int conv_state_len = opt.width - 1;

    T *d_x, *d_w, *d_bias, *d_out, *d_conv_state;
    CHECK(cudaMalloc(&d_x, (long long)opt.batch * opt.dim * sizeof(T)));
    CHECK(cudaMalloc(&d_w, (long long)opt.dim * opt.width * sizeof(T)));
    CHECK(cudaMalloc(&d_bias, opt.dim * sizeof(T)));
    CHECK(cudaMalloc(&d_out, (long long)opt.batch * opt.dim * sizeof(T)));
    CHECK(cudaMalloc(&d_conv_state, (long long)opt.batch * opt.dim * conv_state_len * sizeof(T)));

    srand(42);
    auto fill = [](T* d, long long n) {
        std::vector<T> h(n);
        for (auto& v : h) v = host_from_float<T>(((float)rand()/RAND_MAX - 0.5f) * 0.2f);
        cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    };
    fill(d_x, (long long)opt.batch * opt.dim);
    fill(d_w, (long long)opt.dim * opt.width);
    fill(d_bias, opt.dim);
    fill(d_conv_state, (long long)opt.batch * opt.dim * conv_state_len);

    // Setup params — decode: x is [batch, dim, 1], conv_state is [batch, dim, width-1]
    ConvParamsBase params = {};
    params.batch = opt.batch;
    params.dim = opt.dim;
    params.seqlen = 1;
    params.width = opt.width;
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
    params.x_batch_stride = opt.dim;
    params.x_c_stride = 1;
    params.x_l_stride = 1;
    params.weight_c_stride = opt.width;
    params.weight_width_stride = 1;
    params.out_batch_stride = opt.dim;
    params.out_c_stride = 1;
    params.out_l_stride = 1;
    params.conv_state_len = conv_state_len;
    params.conv_state_batch_stride = opt.dim * conv_state_len;
    params.conv_state_c_stride = conv_state_len;
    params.conv_state_l_stride = 1;

    timer.run([&]() {
        causal_conv1d_update_cuda<T, T>(params, 0);
    });
    CHECK(cudaGetLastError());

    printf("Done.\n");
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_bias); cudaFree(d_out); cudaFree(d_conv_state);
    return 0;
}

int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    Options opt = parse_options(argc, argv);
    return opt.dtype == "fp16" ? run<__half>(opt, timer) : run<__nv_bfloat16>(opt, timer);
}
