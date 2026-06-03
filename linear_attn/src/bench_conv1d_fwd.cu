// Bench: causal_conv1d prefill (Tri Dao, 128-bit vectorized)
// Extracted from https://github.com/Dao-AILab/causal-conv1d
// Usage: ./bench_conv1d_fwd [seq_len] [dim] [width] [batch] [--dtype fp16|bf16]
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
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

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
    int seq = 3823;
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
            printf("Usage: %s [seq_len] [dim] [width] [batch] [--dtype fp16|bf16] [--bench W I]\n", argv[0]);
            exit(0);
        } else {
            int value = atoi(argv[i]);
            if (positional == 0) opt.seq = value;
            else if (positional == 1) opt.dim = value;
            else if (positional == 2) opt.width = value;
            else if (positional == 3) opt.batch = value;
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
    printf("bench conv1d_fwd: seq=%d dim=%d width=%d batch=%d dtype=%s\n",
        opt.seq, opt.dim, opt.width, opt.batch, opt.dtype.c_str());

    long long x_size = (long long)opt.batch * opt.dim * opt.seq;
    long long w_size = (long long)opt.dim * opt.width;
    long long state_size = (long long)opt.batch * opt.dim * (opt.width - 1);

    T *d_x, *d_w, *d_bias, *d_out, *d_final_states;
    CHECK(cudaMalloc(&d_x, x_size * sizeof(T)));
    CHECK(cudaMalloc(&d_w, w_size * sizeof(T)));
    CHECK(cudaMalloc(&d_bias, opt.dim * sizeof(T)));
    CHECK(cudaMalloc(&d_out, x_size * sizeof(T)));
    CHECK(cudaMalloc(&d_final_states, state_size * sizeof(T)));

    // Host init
    srand(42);
    auto fill = [](T* d, long long n) {
        std::vector<T> h(n);
        for (auto& v : h) v = host_from_float<T>(((float)rand()/RAND_MAX - 0.5f) * 0.2f);
        cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    };
    fill(d_x, x_size);
    fill(d_w, w_size);
    fill(d_bias, opt.dim);

    // Setup params (channel-first layout: x is [batch, dim, seq])
    ConvParamsBase params = {};
    params.batch = opt.batch;
    params.dim = opt.dim;
    params.seqlen = opt.seq;
    params.width = opt.width;
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
    params.x_batch_stride = opt.dim * opt.seq;
    params.x_c_stride = opt.seq;
    params.x_l_stride = 1;
    params.weight_c_stride = opt.width;
    params.weight_width_stride = 1;
    params.out_batch_stride = opt.dim * opt.seq;
    params.out_c_stride = opt.seq;
    params.out_l_stride = 1;
    params.final_states_batch_stride = opt.dim * (opt.width - 1);
    params.final_states_c_stride = opt.width - 1;
    params.final_states_l_stride = 1;

    timer.run([&]() {
        causal_conv1d_fwd_cuda<T, T>(params, 0);
    });
    CHECK(cudaGetLastError());

    printf("Done.\n");
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_bias); cudaFree(d_out); cudaFree(d_final_states);
    return 0;
}

int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    Options opt = parse_options(argc, argv);
    return opt.dtype == "fp16" ? run<__half>(opt, timer) : run<__nv_bfloat16>(opt, timer);
}
