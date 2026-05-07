// TensorRT-LLM custom MoE routing kernel standalone benchmark.
// Extracted from:
//   cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu

#include "moe_topk_funcs.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstring>
#include <string>
#include <vector>

namespace trtllm_aux
{
namespace cg = cooperative_groups;

static constexpr int BLOCK_SIZE = 1024;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

template <typename DataType>
__device__ DataType calcSoftmax(
    cg::thread_block_tile<WARP_SIZE> const& warp, DataType score, int32_t laneIdx, int32_t numTopExperts)
{
    float maxScore = -INFINITY;
    if (laneIdx < numTopExperts)
    {
        maxScore = static_cast<float>(score);
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<float>());

    float sumScore = 0.f;
    float newScore = 0.f;
    if (laneIdx < numTopExperts)
    {
        newScore = expf(static_cast<float>(score) - maxScore);
        sumScore += newScore;
    }
    sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

    if (laneIdx < numTopExperts)
    {
        score = static_cast<DataType>(newScore / sumScore);
    }
    return score;
}

template <typename DataType, int VecSize>
__device__ void calcSoftmax(cg::thread_block_tile<WARP_SIZE> const& warp, DataType (&scores)[VecSize])
{
    float maxScore = -INFINITY;
    float sumScore = 0.f;
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        maxScore = max(maxScore, static_cast<float>(scores[i]));
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<float>());

#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        float e = expf(static_cast<float>(scores[i]) - maxScore);
        scores[i] = static_cast<DataType>(e);
        sumScore += e;
    }
    sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        scores[i] = static_cast<DataType>(static_cast<float>(scores[i]) / sumScore);
    }
}

template <typename InputT, typename OutputT, typename IdxT, int MaxNumExperts, int MaxNumTopExperts,
    bool DoSoftmaxBeforeTopK>
__global__ void customMoeRoutingKernel(
    InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices, int32_t numTokens, int32_t numExperts, int32_t topK)
{
    using BaseType = std::conditional_t<DoSoftmaxBeforeTopK, float, InputT>;
    uint32_t const tIdx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    uint32_t const warpIdx = tIdx / WARP_SIZE;
    uint32_t const laneIdx = tIdx % WARP_SIZE;
    uint32_t const warpNum = gridDim.x * WARPS_PER_BLOCK;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    BaseType minScore = BaseType{-INFINITY};

    for (uint32_t tokenId = warpIdx; tokenId < static_cast<uint32_t>(numTokens); tokenId += warpNum)
    {
        auto scoreOffset = tokenId * numExperts;
        auto outputOffset = tokenId * topK;

        BaseType inputScore[MaxNumExperts / WARP_SIZE];
        int32_t inputIndex[MaxNumExperts / WARP_SIZE];
        BaseType warpTopKScore[MaxNumTopExperts];
        int32_t warpTopKExpertIdx[MaxNumTopExperts];

#pragma unroll
        for (uint32_t i = 0; i < MaxNumExperts / WARP_SIZE; ++i)
        {
            auto expertIdx = i * WARP_SIZE + laneIdx;
            inputScore[i]
                = expertIdx < static_cast<uint32_t>(numExperts) ? static_cast<BaseType>(routerLogits[scoreOffset + expertIdx]) : minScore;
            inputIndex[i] = expertIdx;
        }

        if constexpr (DoSoftmaxBeforeTopK)
        {
            calcSoftmax(warp, inputScore);
        }

        reduce_topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, inputScore, inputIndex, minScore);

        if constexpr (DoSoftmaxBeforeTopK)
        {
            if (laneIdx < static_cast<uint32_t>(topK))
            {
                topkValues[outputOffset + laneIdx] = static_cast<OutputT>(warpTopKScore[laneIdx]);
                topkIndices[outputOffset + laneIdx] = warpTopKExpertIdx[laneIdx];
            }
        }
        else
        {
            auto softmaxScore = calcSoftmax(warp,
                laneIdx < static_cast<uint32_t>(topK) ? static_cast<float>(warpTopKScore[laneIdx])
                                                      : static_cast<float>(minScore),
                laneIdx, topK);
            if (laneIdx < static_cast<uint32_t>(topK))
            {
                topkValues[outputOffset + laneIdx] = static_cast<OutputT>(softmaxScore);
                topkIndices[outputOffset + laneIdx] = warpTopKExpertIdx[laneIdx];
            }
        }
    }
}

#define CASE_MAX_TOPK(MAX_NUM_EXPERTS)                                                                                 \
    case MAX_NUM_EXPERTS:                                                                                              \
        switch (maxNumTopExperts)                                                                                      \
        {                                                                                                              \
        case 1: kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 1, DoSoftmaxBeforeTopK>; break; \
        case 2: kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 2, DoSoftmaxBeforeTopK>; break; \
        case 4: kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 4, DoSoftmaxBeforeTopK>; break; \
        case 8: kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 8, DoSoftmaxBeforeTopK>; break; \
        default: kernelInstance = nullptr; break;                                                                      \
        }                                                                                                              \
        break

template <typename InputT, typename OutputT, typename IdxT, bool DoSoftmaxBeforeTopK>
void custom_moe_routing_launch(
    InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices, int numTokens, int numExperts, int topK, cudaStream_t stream)
{
    uint32_t const maxNumBlocks = 1024;
    uint32_t const numBlocks = std::min(static_cast<uint32_t>((numTokens - 1) / WARPS_PER_BLOCK + 1), maxNumBlocks);
    uint32_t const maxNumExperts = std::max(32, trtllm_aux_next_power_of_two(numExperts));
    uint32_t const maxNumTopExperts = trtllm_aux_next_power_of_two(topK);

    auto* kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, 128, 8, DoSoftmaxBeforeTopK>;
    switch (maxNumExperts)
    {
    CASE_MAX_TOPK(32);
    CASE_MAX_TOPK(64);
    CASE_MAX_TOPK(96);
    CASE_MAX_TOPK(128);
    default: kernelInstance = nullptr; break;
    }

    TRTLLM_AUX_CHECK(kernelInstance != nullptr, "unsupported num_experts=%d topK=%d", numExperts, topK);
    kernelInstance<<<numBlocks, BLOCK_SIZE, 0, stream>>>(routerLogits, topkValues, topkIndices, numTokens, numExperts, topK);
}

#undef CASE_MAX_TOPK

} // namespace trtllm_aux

#ifdef BENCH
#include "bench_timer.h"

template <typename InputT>
int run_bench(int tokens, int experts, int topk, bool softmaxBeforeTopK, BenchTimer& timer)
{
    std::vector<InputT> h_logits(tokens * experts);
    for (int i = 0; i < tokens * experts; ++i)
    {
        float v = static_cast<float>((i * 17 + 11) % 101) / 37.0f;
        h_logits[i] = trtllm_aux_from_float<InputT>(v);
    }

    InputT* d_logits = nullptr;
    float* d_values = nullptr;
    int32_t* d_indices = nullptr;
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_logits, h_logits.size() * sizeof(InputT)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_values, tokens * topk * sizeof(float)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_indices, tokens * topk * sizeof(int32_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(InputT), cudaMemcpyHostToDevice));

    timer.run([&]() {
        if (softmaxBeforeTopK)
        {
            trtllm_aux::custom_moe_routing_launch<InputT, float, int32_t, true>(
                d_logits, d_values, d_indices, tokens, experts, topk, 0);
        }
        else
        {
            trtllm_aux::custom_moe_routing_launch<InputT, float, int32_t, false>(
                d_logits, d_values, d_indices, tokens, experts, topk, 0);
        }
    });

    TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_logits));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_values));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_indices));
    return 0;
}

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int tokens = (argc > 1) ? std::atoi(argv[1]) : 1;
    int experts = (argc > 2) ? std::atoi(argv[2]) : 64;
    int topk = (argc > 3) ? std::atoi(argv[3]) : 8;
    std::string dtype = "float";
    bool softmaxBeforeTopK = false;
    for (int i = 4; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--softmax-before-topk") == 0)
        {
            softmaxBeforeTopK = true;
        }
        else
        {
            dtype = argv[i];
        }
    }

    std::printf("bench trtllm custom_moe_routing: tokens=%d experts=%d topk=%d dtype=%s softmax_before_topk=%d\n",
        tokens, experts, topk, dtype.c_str(), static_cast<int>(softmaxBeforeTopK));

    if (dtype == "float" || dtype == "fp32")
    {
        return run_bench<float>(tokens, experts, topk, softmaxBeforeTopK, timer);
    }
    if (dtype == "half" || dtype == "fp16")
    {
        return run_bench<half>(tokens, experts, topk, softmaxBeforeTopK, timer);
    }
#if ENABLE_BF16
    if (dtype == "bf16")
    {
        return run_bench<__nv_bfloat16>(tokens, experts, topk, softmaxBeforeTopK, timer);
    }
#endif

    std::fprintf(stderr, "unsupported dtype: %s\n", dtype.c_str());
    return 1;
}
#endif
