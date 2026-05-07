// TensorRT-LLM MoE pipeline helper kernels, standalone FP16/BF16 subset.
// Extracted and specialized from:
//   cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu

#include "trtllm_aux_compat.h"

#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace trtllm_aux
{

static constexpr int EXPAND_THREADS_PER_BLOCK = 256;
static constexpr int ACTIVATION_THREADS_PER_BLOCK = 256;
static constexpr int FINALIZE_THREADS_PER_BLOCK = 256;

template <typename T>
__global__ void expandInputRowsKernel(T const* __restrict__ unpermuted_input, T* __restrict__ permuted_output,
    float const* __restrict__ unpermuted_scales, float* __restrict__ permuted_scales,
    int const* __restrict__ permuted_row_to_unpermuted_row, int64_t num_tokens, int64_t hidden_size, int64_t topk,
    int64_t valid_rows)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    for (int64_t permuted_row = blockIdx.x; permuted_row < valid_rows; permuted_row += gridDim.x)
    {
        int64_t const unpermuted_row = permuted_row_to_unpermuted_row[permuted_row];
        int64_t const source_k_rank = unpermuted_row / num_tokens;
        int64_t const source_row = unpermuted_row % num_tokens;
        T const* src = unpermuted_input + source_row * hidden_size;
        T* dst = permuted_output + permuted_row * hidden_size;
        for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x)
        {
            dst[col] = src[col];
        }
        if (permuted_scales && threadIdx.x == 0)
        {
            int64_t const source_k_idx = source_row * topk + source_k_rank;
            permuted_scales[permuted_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T>
void expand_input_rows_launch(T const* unpermuted_input, T* permuted_output, float const* unpermuted_scales,
    float* permuted_scales, int const* permuted_row_to_unpermuted_row, int64_t num_tokens, int64_t hidden_size,
    int64_t topk, int64_t valid_rows, cudaStream_t stream)
{
    int device = 0;
    int sm_count = 0;
    TRTLLM_AUX_CUDA_CHECK(cudaGetDevice(&device));
    TRTLLM_AUX_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int const blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(sm_count * 4, valid_rows)));
    expandInputRowsKernel<T><<<blocks, EXPAND_THREADS_PER_BLOCK, 0, stream>>>(unpermuted_input, permuted_output,
        unpermuted_scales, permuted_scales, permuted_row_to_unpermuted_row, num_tokens, hidden_size, topk, valid_rows);
}

template <typename T>
__global__ void gatedActivationKernel(T* __restrict__ output, T const* __restrict__ gemm_result,
    int64_t const* __restrict__ expert_first_token_offset, int64_t inter_size, int64_t num_experts_per_node,
    int64_t valid_rows)
{
    int64_t const row = blockIdx.x;
    if (row >= expert_first_token_offset[num_experts_per_node] || row >= valid_rows)
    {
        return;
    }

    T const* gate = gemm_result + row * inter_size * 2;
    T const* up = gate + inter_size;
    T* out = output + row * inter_size;

    for (int64_t col = threadIdx.x; col < inter_size; col += blockDim.x)
    {
        float const g = trtllm_aux_to_float(gate[col]);
        float const u = trtllm_aux_to_float(up[col]);
        float const silu = g / (1.0f + expf(-g));
        out[col] = trtllm_aux_from_float<T>(silu * u);
    }
}

template <typename T>
void gated_activation_launch(T* output, T const* gemm_result, int64_t const* expert_first_token_offset,
    int64_t inter_size, int64_t valid_rows, int64_t num_experts_per_node, cudaStream_t stream)
{
    gatedActivationKernel<T>
        <<<valid_rows, ACTIVATION_THREADS_PER_BLOCK, 0, stream>>>(output, gemm_result, expert_first_token_offset,
            inter_size, num_experts_per_node, valid_rows);
}

template <typename T, bool SCALE>
__global__ void finalizeMoeRoutingKernel(T const* __restrict__ expanded_permuted_rows,
    T* __restrict__ reduced_unpermuted_output, float const* __restrict__ scales,
    int const* __restrict__ unpermuted_row_to_permuted_row, int const* __restrict__ token_selected_experts,
    int64_t num_rows, int64_t hidden_size, int64_t topk, int64_t num_experts_per_node, int start_expert_id)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int64_t const row = blockIdx.x;
    if (row >= num_rows)
    {
        return;
    }
    T* out = reduced_unpermuted_output + row * hidden_size;

    for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x)
    {
        float acc = 0.0f;
        for (int64_t k_idx = 0; k_idx < topk; ++k_idx)
        {
            int64_t const k_offset = row * topk + k_idx;
            int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
            if (expert_id < 0 || expert_id >= num_experts_per_node)
            {
                continue;
            }
            int64_t const expanded_original_row = row + k_idx * num_rows;
            int64_t const expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];
            float const row_scale = SCALE ? scales[k_offset] : 1.0f;
            acc += row_scale * trtllm_aux_to_float(expanded_permuted_rows[expanded_permuted_row * hidden_size + col]);
        }
        out[col] = trtllm_aux_from_float<T>(acc);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T>
void finalize_moe_routing_launch(T const* expanded_permuted_rows, T* reduced_unpermuted_output, float const* scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts, int64_t num_rows,
    int64_t hidden_size, int64_t topk, int64_t num_experts_per_node, bool use_scales, cudaStream_t stream)
{
    if (use_scales)
    {
        finalizeMoeRoutingKernel<T, true><<<num_rows, FINALIZE_THREADS_PER_BLOCK, 0, stream>>>(expanded_permuted_rows,
            reduced_unpermuted_output, scales, unpermuted_row_to_permuted_row, token_selected_experts, num_rows,
            hidden_size, topk, num_experts_per_node, 0);
    }
    else
    {
        finalizeMoeRoutingKernel<T, false><<<num_rows, FINALIZE_THREADS_PER_BLOCK, 0, stream>>>(expanded_permuted_rows,
            reduced_unpermuted_output, scales, unpermuted_row_to_permuted_row, token_selected_experts, num_rows,
            hidden_size, topk, num_experts_per_node, 0);
    }
}

template void expand_input_rows_launch<half>(half const*, half*, float const*, float*, int const*, int64_t, int64_t,
    int64_t, int64_t, cudaStream_t);
template void gated_activation_launch<half>(half*, half const*, int64_t const*, int64_t, int64_t, int64_t,
    cudaStream_t);
template void finalize_moe_routing_launch<half>(half const*, half*, float const*, int const*, int const*, int64_t,
    int64_t, int64_t, int64_t, bool, cudaStream_t);

#if ENABLE_BF16
template void expand_input_rows_launch<__nv_bfloat16>(__nv_bfloat16 const*, __nv_bfloat16*, float const*, float*,
    int const*, int64_t, int64_t, int64_t, int64_t, cudaStream_t);
template void gated_activation_launch<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, int64_t const*, int64_t,
    int64_t, int64_t, cudaStream_t);
template void finalize_moe_routing_launch<__nv_bfloat16>(__nv_bfloat16 const*, __nv_bfloat16*, float const*,
    int const*, int const*, int64_t, int64_t, int64_t, int64_t, bool, cudaStream_t);
#endif

} // namespace trtllm_aux

#ifdef BENCH_EXPAND
#include "bench_timer.h"

template <typename T>
int run_expand(int tokens, int topk, int hidden, BenchTimer& timer)
{
    int64_t const expanded_rows = static_cast<int64_t>(tokens) * topk;
    std::vector<T> h_input(static_cast<size_t>(tokens) * hidden);
    std::vector<float> h_scales(static_cast<size_t>(tokens) * topk);
    std::vector<int> h_map(expanded_rows);
    for (int64_t i = 0; i < static_cast<int64_t>(h_input.size()); ++i)
    {
        h_input[i] = trtllm_aux_from_float<T>(static_cast<float>((i * 13 + 7) % 127) * 0.001f);
    }
    for (int t = 0; t < tokens; ++t)
    {
        for (int k = 0; k < topk; ++k)
        {
            h_scales[t * topk + k] = 1.0f / static_cast<float>(topk);
            h_map[k * tokens + t] = k * tokens + t;
        }
    }

    T* d_input = nullptr;
    T* d_output = nullptr;
    float* d_scales = nullptr;
    float* d_permuted_scales = nullptr;
    int* d_map = nullptr;
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(expanded_rows) * hidden * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_scales, h_scales.size() * sizeof(float)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_permuted_scales, static_cast<size_t>(expanded_rows) * sizeof(float)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_map, h_map.size() * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_map, h_map.data(), h_map.size() * sizeof(int), cudaMemcpyHostToDevice));

    timer.run([&]() {
        trtllm_aux::expand_input_rows_launch<T>(
            d_input, d_output, d_scales, d_permuted_scales, d_map, tokens, hidden, topk, expanded_rows, 0);
    });
    TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scales);
    cudaFree(d_permuted_scales);
    cudaFree(d_map);
    return 0;
}

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);
    int tokens = (argc > 1) ? std::atoi(argv[1]) : 1;
    int topk = (argc > 2) ? std::atoi(argv[2]) : 8;
    int hidden = (argc > 3) ? std::atoi(argv[3]) : 2048;
    std::string dtype = (argc > 4) ? argv[4] : "fp16";
    std::printf("bench trtllm expand_input_rows: tokens=%d topk=%d hidden=%d dtype=%s\n", tokens, topk, hidden,
        dtype.c_str());
    if (dtype == "fp16" || dtype == "half")
        return run_expand<half>(tokens, topk, hidden, timer);
#if ENABLE_BF16
    if (dtype == "bf16")
        return run_expand<__nv_bfloat16>(tokens, topk, hidden, timer);
#endif
    std::fprintf(stderr, "unsupported dtype: %s\n", dtype.c_str());
    return 1;
}
#endif

#ifdef BENCH_GATED
#include "bench_timer.h"

template <typename T>
int run_gated(int tokens, int topk, int inter_size, BenchTimer& timer)
{
    int64_t const expanded_rows = static_cast<int64_t>(tokens) * topk;
    std::vector<T> h_input(static_cast<size_t>(expanded_rows) * inter_size * 2);
    std::vector<int64_t> h_offsets(topk + 1);
    for (int64_t i = 0; i < static_cast<int64_t>(h_input.size()); ++i)
    {
        h_input[i] = trtllm_aux_from_float<T>(static_cast<float>((i * 17 + 5) % 97) * 0.001f);
    }
    for (int k = 0; k <= topk; ++k)
    {
        h_offsets[k] = static_cast<int64_t>(k) * tokens;
    }

    T* d_input = nullptr;
    T* d_output = nullptr;
    int64_t* d_offsets = nullptr;
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(expanded_rows) * inter_size * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int64_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    timer.run([&]() {
        trtllm_aux::gated_activation_launch<T>(d_output, d_input, d_offsets, inter_size, expanded_rows, topk, 0);
    });
    TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_offsets);
    return 0;
}

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);
    int tokens = (argc > 1) ? std::atoi(argv[1]) : 1;
    int topk = (argc > 2) ? std::atoi(argv[2]) : 8;
    int inter_size = (argc > 3) ? std::atoi(argv[3]) : 3072;
    std::string dtype = (argc > 4) ? argv[4] : "fp16";
    std::printf("bench trtllm gated_activation: tokens=%d topk=%d rows=%lld inter=%d dtype=%s\n", tokens, topk,
        static_cast<long long>(static_cast<int64_t>(tokens) * topk), inter_size, dtype.c_str());
    if (dtype == "fp16" || dtype == "half")
        return run_gated<half>(tokens, topk, inter_size, timer);
#if ENABLE_BF16
    if (dtype == "bf16")
        return run_gated<__nv_bfloat16>(tokens, topk, inter_size, timer);
#endif
    std::fprintf(stderr, "unsupported dtype: %s\n", dtype.c_str());
    return 1;
}
#endif

#ifdef BENCH_FINALIZE
#include "bench_timer.h"

template <typename T>
int run_finalize(int tokens, int topk, int hidden, bool use_scales, BenchTimer& timer)
{
    int64_t const expanded_rows = static_cast<int64_t>(tokens) * topk;
    std::vector<T> h_input(static_cast<size_t>(expanded_rows) * hidden);
    std::vector<float> h_scales(static_cast<size_t>(tokens) * topk);
    std::vector<int> h_unperm_to_perm(expanded_rows);
    std::vector<int> h_experts(static_cast<size_t>(tokens) * topk);
    for (int64_t i = 0; i < static_cast<int64_t>(h_input.size()); ++i)
    {
        h_input[i] = trtllm_aux_from_float<T>(static_cast<float>((i * 19 + 3) % 89) * 0.001f);
    }
    for (int t = 0; t < tokens; ++t)
    {
        for (int k = 0; k < topk; ++k)
        {
            h_unperm_to_perm[k * tokens + t] = k * tokens + t;
            h_experts[t * topk + k] = k;
            h_scales[t * topk + k] = 1.0f / static_cast<float>(topk);
        }
    }

    T* d_input = nullptr;
    T* d_output = nullptr;
    float* d_scales = nullptr;
    int* d_unperm_to_perm = nullptr;
    int* d_experts = nullptr;
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(tokens) * hidden * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_scales, h_scales.size() * sizeof(float)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_unperm_to_perm, h_unperm_to_perm.size() * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_experts, h_experts.size() * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(
        d_unperm_to_perm, h_unperm_to_perm.data(), h_unperm_to_perm.size() * sizeof(int), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_experts, h_experts.data(), h_experts.size() * sizeof(int), cudaMemcpyHostToDevice));

    timer.run([&]() {
        trtllm_aux::finalize_moe_routing_launch<T>(
            d_input, d_output, d_scales, d_unperm_to_perm, d_experts, tokens, hidden, topk, topk, use_scales, 0);
    });
    TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scales);
    cudaFree(d_unperm_to_perm);
    cudaFree(d_experts);
    return 0;
}

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);
    int tokens = (argc > 1) ? std::atoi(argv[1]) : 1;
    int topk = (argc > 2) ? std::atoi(argv[2]) : 8;
    int hidden = (argc > 3) ? std::atoi(argv[3]) : 1024;
    std::string dtype = (argc > 4) ? argv[4] : "fp16";
    bool use_scales = true;
    for (int i = 5; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--no-scales") == 0)
        {
            use_scales = false;
        }
    }
    std::printf("bench trtllm finalize_moe_routing: tokens=%d topk=%d hidden=%d dtype=%s scales=%d\n", tokens, topk,
        hidden, dtype.c_str(), static_cast<int>(use_scales));
    if (dtype == "fp16" || dtype == "half")
        return run_finalize<half>(tokens, topk, hidden, use_scales, timer);
#if ENABLE_BF16
    if (dtype == "bf16")
        return run_finalize<__nv_bfloat16>(tokens, topk, hidden, use_scales, timer);
#endif
    std::fprintf(stderr, "unsupported dtype: %s\n", dtype.c_str());
    return 1;
}
#endif
