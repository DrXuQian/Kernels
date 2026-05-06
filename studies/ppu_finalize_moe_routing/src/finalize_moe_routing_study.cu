// Isolated study for TensorRT-LLM finalizeMoeRoutingKernel source-level
// compensation on PPU. This file intentionally stays outside production
// auxiliary kernels and root benchmark orchestration.

#include "bench_timer.h"
#include "trtllm_aux_compat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace finalize_study
{

static constexpr int FINALIZE_THREADS_PER_BLOCK = 256;
static constexpr int TOPK_MAX_UNROLL = 8;

template <typename T, bool SCALE>
__global__ void finalizeMoeRoutingBaselineKernel(T const* __restrict__ expanded_permuted_rows,
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

template <typename T, bool SCALE>
__global__ void finalizeMoeRoutingOptimizedKernel(T const* __restrict__ expanded_permuted_rows,
    T* __restrict__ reduced_unpermuted_output, float const* __restrict__ scales,
    int const* __restrict__ unpermuted_row_to_permuted_row, int const* __restrict__ token_selected_experts,
    int64_t num_rows, int64_t hidden_size, int64_t topk, int64_t num_experts_per_node, int start_expert_id)
{
    static_assert(sizeof(T) == 2, "optimized finalize study expects 16-bit element types");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int64_t const row = blockIdx.x;
    if (row >= num_rows)
    {
        return;
    }

    extern __shared__ unsigned char smem_raw[];
    float* s_scale = reinterpret_cast<float*>(smem_raw);
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(s_scale + topk);
    base_addr = (base_addr + alignof(T const*) - 1) & ~(static_cast<uintptr_t>(alignof(T const*) - 1));
    T const** s_base = reinterpret_cast<T const**>(base_addr);

    for (int64_t k = threadIdx.x; k < topk; k += blockDim.x)
    {
        int64_t const k_offset = row * topk + k;
        int const expert_id = token_selected_experts[k_offset] - start_expert_id;
        bool const valid = (expert_id >= 0 && expert_id < num_experts_per_node);
        int64_t const expanded_original_row = row + k * num_rows;
        int64_t const expanded_permuted_row = valid ? unpermuted_row_to_permuted_row[expanded_original_row] : 0;
        s_base[k] = expanded_permuted_rows + expanded_permuted_row * hidden_size;
        s_scale[k] = valid ? (SCALE ? scales[k_offset] : 1.0f) : 0.0f;
    }
    __syncthreads();

    T* out = reduced_unpermuted_output + row * hidden_size;
    bool const vectorizable = ((hidden_size & 3) == 0);
    int64_t const vec_count = vectorizable ? (hidden_size / 4) : 0;
    int64_t const tail_start = vec_count * 4;

    if (vectorizable)
    {
        for (int64_t v = threadIdx.x; v < vec_count; v += blockDim.x)
        {
            uint2 pack_data[TOPK_MAX_UNROLL];
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
            for (int k = 0; k < TOPK_MAX_UNROLL; ++k)
            {
                if (k >= topk)
                {
                    break;
                }
                pack_data[k] = *reinterpret_cast<uint2 const*>(s_base[k] + v * 4);
            }

#pragma unroll
            for (int k = 0; k < TOPK_MAX_UNROLL; ++k)
            {
                if (k >= topk)
                {
                    break;
                }
                T const* lane = reinterpret_cast<T const*>(&pack_data[k]);
                float const s = s_scale[k];
                acc[0] += s * trtllm_aux_to_float(lane[0]);
                acc[1] += s * trtllm_aux_to_float(lane[1]);
                acc[2] += s * trtllm_aux_to_float(lane[2]);
                acc[3] += s * trtllm_aux_to_float(lane[3]);
            }

            for (int64_t k = TOPK_MAX_UNROLL; k < topk; ++k)
            {
                T const* lane = s_base[k] + v * 4;
                float const s = s_scale[k];
                acc[0] += s * trtllm_aux_to_float(lane[0]);
                acc[1] += s * trtllm_aux_to_float(lane[1]);
                acc[2] += s * trtllm_aux_to_float(lane[2]);
                acc[3] += s * trtllm_aux_to_float(lane[3]);
            }

            uint2 out_pack;
            T* out_lane = reinterpret_cast<T*>(&out_pack);
            out_lane[0] = trtllm_aux_from_float<T>(acc[0]);
            out_lane[1] = trtllm_aux_from_float<T>(acc[1]);
            out_lane[2] = trtllm_aux_from_float<T>(acc[2]);
            out_lane[3] = trtllm_aux_from_float<T>(acc[3]);
            *reinterpret_cast<uint2*>(out + v * 4) = out_pack;
        }
    }

    for (int64_t col = tail_start + threadIdx.x; col < hidden_size; col += blockDim.x)
    {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < TOPK_MAX_UNROLL; ++k)
        {
            if (k >= topk)
            {
                break;
            }
            acc += s_scale[k] * trtllm_aux_to_float(s_base[k][col]);
        }
        for (int64_t k = TOPK_MAX_UNROLL; k < topk; ++k)
        {
            acc += s_scale[k] * trtllm_aux_to_float(s_base[k][col]);
        }
        out[col] = trtllm_aux_from_float<T>(acc);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

inline size_t optimized_smem_bytes(int64_t topk)
{
    return static_cast<size_t>(topk) * sizeof(float) + (alignof(void*) - 1)
        + static_cast<size_t>(topk) * sizeof(void*);
}

template <typename T>
void finalize_baseline_launch(T const* expanded_permuted_rows, T* reduced_unpermuted_output, float const* scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts, int64_t num_rows,
    int64_t hidden_size, int64_t topk, int64_t num_experts_per_node, bool use_scales, cudaStream_t stream)
{
    if (use_scales)
    {
        finalizeMoeRoutingBaselineKernel<T, true><<<num_rows, FINALIZE_THREADS_PER_BLOCK, 0, stream>>>(
            expanded_permuted_rows, reduced_unpermuted_output, scales, unpermuted_row_to_permuted_row,
            token_selected_experts, num_rows, hidden_size, topk, num_experts_per_node, 0);
    }
    else
    {
        finalizeMoeRoutingBaselineKernel<T, false><<<num_rows, FINALIZE_THREADS_PER_BLOCK, 0, stream>>>(
            expanded_permuted_rows, reduced_unpermuted_output, scales, unpermuted_row_to_permuted_row,
            token_selected_experts, num_rows, hidden_size, topk, num_experts_per_node, 0);
    }
}

template <typename T>
void finalize_optimized_launch(T const* expanded_permuted_rows, T* reduced_unpermuted_output, float const* scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts, int64_t num_rows,
    int64_t hidden_size, int64_t topk, int64_t num_experts_per_node, bool use_scales, cudaStream_t stream)
{
    size_t const smem = optimized_smem_bytes(topk);
    if (use_scales)
    {
        finalizeMoeRoutingOptimizedKernel<T, true><<<num_rows, FINALIZE_THREADS_PER_BLOCK, smem, stream>>>(
            expanded_permuted_rows, reduced_unpermuted_output, scales, unpermuted_row_to_permuted_row,
            token_selected_experts, num_rows, hidden_size, topk, num_experts_per_node, 0);
    }
    else
    {
        finalizeMoeRoutingOptimizedKernel<T, false><<<num_rows, FINALIZE_THREADS_PER_BLOCK, smem, stream>>>(
            expanded_permuted_rows, reduced_unpermuted_output, scales, unpermuted_row_to_permuted_row,
            token_selected_experts, num_rows, hidden_size, topk, num_experts_per_node, 0);
    }
}

enum class Mode
{
    kBaseline,
    kOptimized,
    kBoth,
};

template <typename T>
int run_finalize(int tokens, int topk, int hidden, bool use_scales, bool check, Mode mode, BenchTimer& timer)
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
    T* d_baseline = nullptr;
    T* d_optimized = nullptr;
    float* d_scales = nullptr;
    int* d_unperm_to_perm = nullptr;
    int* d_experts = nullptr;
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_baseline, static_cast<size_t>(tokens) * hidden * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_optimized, static_cast<size_t>(tokens) * hidden * sizeof(T)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_scales, h_scales.size() * sizeof(float)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_unperm_to_perm, h_unperm_to_perm.size() * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_experts, h_experts.size() * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(
        d_unperm_to_perm, h_unperm_to_perm.data(), h_unperm_to_perm.size() * sizeof(int), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_experts, h_experts.data(), h_experts.size() * sizeof(int), cudaMemcpyHostToDevice));

    if (check || mode == Mode::kBoth)
    {
        finalize_baseline_launch<T>(
            d_input, d_baseline, d_scales, d_unperm_to_perm, d_experts, tokens, hidden, topk, topk, use_scales, 0);
        finalize_optimized_launch<T>(
            d_input, d_optimized, d_scales, d_unperm_to_perm, d_experts, tokens, hidden, topk, topk, use_scales, 0);
        TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
        TRTLLM_AUX_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> h_baseline(static_cast<size_t>(tokens) * hidden);
        std::vector<T> h_optimized(static_cast<size_t>(tokens) * hidden);
        TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(
            h_baseline.data(), d_baseline, h_baseline.size() * sizeof(T), cudaMemcpyDeviceToHost));
        TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(
            h_optimized.data(), d_optimized, h_optimized.size() * sizeof(T), cudaMemcpyDeviceToHost));

        double max_abs = 0.0;
        for (size_t i = 0; i < h_baseline.size(); ++i)
        {
            double const a = trtllm_aux_to_float(h_baseline[i]);
            double const b = trtllm_aux_to_float(h_optimized[i]);
            max_abs = std::max(max_abs, std::abs(a - b));
        }
        std::printf("  check baseline_vs_optimized max_abs=%g\n", max_abs);
        if (max_abs > 1e-3)
        {
            std::fprintf(stderr, "baseline/optimized mismatch: max_abs=%g\n", max_abs);
            return 1;
        }
    }

    if (mode == Mode::kBaseline || mode == Mode::kBoth)
    {
        std::printf("  baseline:\n");
        timer.run([&]() {
            finalize_baseline_launch<T>(
                d_input, d_baseline, d_scales, d_unperm_to_perm, d_experts, tokens, hidden, topk, topk, use_scales, 0);
        });
        TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
    }
    if (mode == Mode::kOptimized || mode == Mode::kBoth)
    {
        std::printf("  optimized:\n");
        timer.run([&]() {
            finalize_optimized_launch<T>(
                d_input, d_optimized, d_scales, d_unperm_to_perm, d_experts, tokens, hidden, topk, topk, use_scales, 0);
        });
        TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
    }

    cudaFree(d_input);
    cudaFree(d_baseline);
    cudaFree(d_optimized);
    cudaFree(d_scales);
    cudaFree(d_unperm_to_perm);
    cudaFree(d_experts);
    return 0;
}

} // namespace finalize_study

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
    bool check = true;
    finalize_study::Mode mode = finalize_study::Mode::kBoth;

    for (int i = 5; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--no-scales") == 0)
        {
            use_scales = false;
        }
        else if (std::strcmp(argv[i], "--no-check") == 0)
        {
            check = false;
        }
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            std::string value = argv[++i];
            if (value == "baseline")
            {
                mode = finalize_study::Mode::kBaseline;
            }
            else if (value == "optimized")
            {
                mode = finalize_study::Mode::kOptimized;
            }
            else if (value == "both")
            {
                mode = finalize_study::Mode::kBoth;
            }
            else
            {
                std::fprintf(stderr, "unsupported mode: %s\n", value.c_str());
                return 1;
            }
        }
    }

    std::printf("bench finalize_moe_routing study: tokens=%d topk=%d hidden=%d dtype=%s scales=%d check=%d\n",
        tokens, topk, hidden, dtype.c_str(), static_cast<int>(use_scales), static_cast<int>(check));
    std::printf("  optimized dynamic_smem=%zu bytes\n", finalize_study::optimized_smem_bytes(topk));

    if (dtype == "fp16" || dtype == "half")
    {
        return finalize_study::run_finalize<half>(tokens, topk, hidden, use_scales, check, mode, timer);
    }
#if ENABLE_BF16
    if (dtype == "bf16")
    {
        return finalize_study::run_finalize<__nv_bfloat16>(tokens, topk, hidden, use_scales, check, mode, timer);
    }
#endif
    std::fprintf(stderr, "unsupported dtype: %s\n", dtype.c_str());
    return 1;
}
