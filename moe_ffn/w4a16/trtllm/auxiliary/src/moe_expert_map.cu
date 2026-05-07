// TensorRT-LLM MoE expert-map prologue standalone benchmark.
// Extracted from:
//   cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu

#include "trtllm_aux_compat.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace trtllm_aux
{

int64_t compute_num_tokens_per_block(int64_t const num_tokens, int64_t const num_experts_per_node)
{
    for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2)
    {
        int64_t const num_blocks_per_seq = CEILDIV(num_tokens, num_tokens_per_block);
        if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block)
        {
            return num_tokens_per_block;
        }
    }
    return 1024;
}

inline int expert_log2_bins(int num_experts_per_node)
{
    int const values = num_experts_per_node + 1;
    int bits = 0;
    while ((1 << bits) <= values)
    {
        ++bits;
    }
    return bits;
}

inline bool fused_expert_map_supported(int64_t num_tokens, int num_experts_per_node, int experts_per_token)
{
    if (num_tokens > 256)
    {
        return false;
    }
    if (expert_log2_bins(num_experts_per_node) > 9)
    {
        return false;
    }
    return experts_per_token == 1 || experts_per_token == 2 || experts_per_token == 4 || experts_per_token == 6
        || experts_per_token == 8;
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel(int const* const token_selected_experts,
    int* const permuted_row_to_unpermuted_row, int* const unpermuted_row_to_permuted_row,
    int64_t* const expert_first_token_offset, int64_t const num_tokens, int const experts_per_token,
    int const start_expert, int const end_expert, int const num_experts_per_node)
{
    assert(gridDim.x == 1);
    assert(start_expert <= end_expert);
    assert(num_experts_per_node == (end_expert - start_expert));
    assert(num_experts_per_node <= (1 << LOG2_NUM_EXPERTS));

    int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool is_valid_token = token < num_tokens;

    int local_token_selected_experts[EXPERTS_PER_TOKEN];
    int local_token_permuted_indices[EXPERTS_PER_TOKEN];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

#pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
    {
        int const expert = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i] : num_experts_per_node;
        bool is_valid_expert = expert >= start_expert && expert < end_expert;
        local_token_selected_experts[i] = !is_valid_token ? num_experts_per_node + 1
            : is_valid_expert                             ? (expert - start_expert)
                                                          : num_experts_per_node;
    }

    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    extern __shared__ unsigned char temp_storage[];
    auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

    static_assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= (1 << LOG2_NUM_EXPERTS));
    assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= num_experts_per_node);

    int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];

    cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);
    BlockRadixRank(sort_temp).RankKeys(
        local_token_selected_experts, local_token_permuted_indices, extractor, local_expert_first_token_offset);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    if (is_valid_token)
    {
#pragma unroll
        for (int i = 0; i < EXPERTS_PER_TOKEN; i++)
        {
            int const unpermuted_row = i * num_tokens + token;
            int const permuted_row = local_token_permuted_indices[i];
            permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
            unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
        }
    }

#pragma unroll
    for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++)
    {
        int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
        if (out_expert_id < num_experts_per_node + 1)
        {
            expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
        }
    }
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fused_build_expert_maps_sort_first_token_dispatch(int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token, int const start_expert,
    int const end_expert, cudaStream_t stream)
{
    TRTLLM_AUX_CHECK(num_experts_per_node == (end_expert - start_expert),
        "num_experts_per_node must equal end_expert - start_expert");
    int const threads = BLOCK_SIZE;
    int const blocks = CEILDIV(num_tokens, threads);
    TRTLLM_AUX_CHECK(blocks == 1, "fused expert-map implementation requires a single CTA");

    using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
    size_t const shared_size = sizeof(typename BlockRadixRank::TempStorage);
    auto kernel = &fusedBuildExpertMapsSortFirstTokenKernel<BLOCK_SIZE, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;

    int device = 0;
    int max_smem_per_block = 0;
    TRTLLM_AUX_CUDA_CHECK(cudaGetDevice(&device));
    TRTLLM_AUX_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (shared_size >= static_cast<size_t>(max_smem_per_block))
    {
        return false;
    }

    TRTLLM_AUX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
    kernel<<<blocks, threads, shared_size, stream>>>(token_selected_experts, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens, experts_per_token, start_expert,
        end_expert, num_experts_per_node);
    return true;
}

template <int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fused_build_expert_maps_sort_first_token_block_size(int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token, int const start_expert,
    int const end_expert, cudaStream_t stream)
{
    if (num_tokens > 256)
    {
        return false;
    }

    auto func = &fused_build_expert_maps_sort_first_token_dispatch<32, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
    if (num_tokens > 32 && num_tokens <= 64)
    {
        func = &fused_build_expert_maps_sort_first_token_dispatch<64, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
    }
    else if (num_tokens > 64 && num_tokens <= 128)
    {
        func = &fused_build_expert_maps_sort_first_token_dispatch<128, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
    }
    else if (num_tokens > 128 && num_tokens <= 256)
    {
        func = &fused_build_expert_maps_sort_first_token_dispatch<256, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
    }

    return func(token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
        expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token, start_expert, end_expert,
        stream);
}

template <int LOG2_NUM_EXPERTS>
bool fused_build_expert_maps_sort_first_token_block_size(int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token, int const start_expert,
    int const end_expert, cudaStream_t stream)
{
    auto func = &fused_build_expert_maps_sort_first_token_block_size<1, LOG2_NUM_EXPERTS>;
    switch (experts_per_token)
    {
    case 1: func = &fused_build_expert_maps_sort_first_token_block_size<1, LOG2_NUM_EXPERTS>; break;
    case 2: func = &fused_build_expert_maps_sort_first_token_block_size<2, LOG2_NUM_EXPERTS>; break;
    case 4: func = &fused_build_expert_maps_sort_first_token_block_size<4, LOG2_NUM_EXPERTS>; break;
    case 6: func = &fused_build_expert_maps_sort_first_token_block_size<6, LOG2_NUM_EXPERTS>; break;
    case 8: func = &fused_build_expert_maps_sort_first_token_block_size<8, LOG2_NUM_EXPERTS>; break;
    default: return false;
    }
    return func(token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
        expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token, start_expert, end_expert,
        stream);
}

bool fused_build_expert_maps_sort_first_token(int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset, int64_t const num_tokens,
    int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
    cudaStream_t stream)
{
    int const expert_log = expert_log2_bins(num_experts_per_node);
    if (expert_log <= 9)
    {
        auto funcs = std::array{&fused_build_expert_maps_sort_first_token_block_size<1>,
            &fused_build_expert_maps_sort_first_token_block_size<2>,
            &fused_build_expert_maps_sort_first_token_block_size<3>,
            &fused_build_expert_maps_sort_first_token_block_size<4>,
            &fused_build_expert_maps_sort_first_token_block_size<5>,
            &fused_build_expert_maps_sort_first_token_block_size<6>,
            &fused_build_expert_maps_sort_first_token_block_size<7>,
            &fused_build_expert_maps_sort_first_token_block_size<8>,
            &fused_build_expert_maps_sort_first_token_block_size<9>};

        return funcs[expert_log - 1](token_selected_experts, permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens, num_experts_per_node,
            experts_per_token, start_expert, end_expert, stream);
    }
    return false;
}

template <int kNumTokensPerBlock>
__global__ void blockExpertPrefixSumKernel(int const* token_selected_experts, int* blocked_expert_counts,
    int* blocked_row_to_unpermuted_row, int64_t const num_tokens, int64_t const num_experts_per_token,
    int const start_expert_id)
{
    using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;
    int const num_blocks_per_seq = gridDim.y;
    int const token_id = block_id * kNumTokensPerBlock + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int expanded_token_id = -1;
    if (token_id < num_tokens)
    {
        for (int i = 0; i < num_experts_per_token; i++)
        {
            int const expert_id = token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
            if (expert_id == target_expert_id)
            {
                expanded_token_id = i * num_tokens + token_id;
                break;
            }
        }
    }

    int const has_matched = expanded_token_id >= 0 ? 1 : 0;
    int index;
    BlockScan(temp_storage).ExclusiveSum(has_matched, index);

    if (has_matched)
    {
        blocked_row_to_unpermuted_row[target_expert_id * num_tokens + block_id * kNumTokensPerBlock + index]
            = expanded_token_id;
    }
    if (threadIdx.x == kNumTokensPerBlock - 1)
    {
        blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] = index + has_matched;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void block_expert_prefix_sum(int const* token_selected_experts, int* blocked_expert_counts,
    int* blocked_row_to_unpermuted_row, int64_t const num_tokens, int64_t const num_experts_per_node,
    int64_t const num_experts_per_token, int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq,
    int const start_expert_id, cudaStream_t stream)
{
    dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
    dim3 const threads(num_tokens_per_block);

    auto func = blockExpertPrefixSumKernel<1024>;
    if (num_tokens_per_block <= 32)
    {
        func = blockExpertPrefixSumKernel<32>;
    }
    else if (num_tokens_per_block <= 64)
    {
        func = blockExpertPrefixSumKernel<64>;
    }
    else if (num_tokens_per_block <= 128)
    {
        func = blockExpertPrefixSumKernel<128>;
    }
    else if (num_tokens_per_block <= 256)
    {
        func = blockExpertPrefixSumKernel<256>;
    }
    else if (num_tokens_per_block <= 512)
    {
        func = blockExpertPrefixSumKernel<512>;
    }

    func<<<blocks, threads, 0, stream>>>(
        token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row, num_tokens,
        num_experts_per_token, start_expert_id);
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumLargeKernel(int const* blocked_expert_counts, int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset, int64_t const num_experts_per_node, int64_t const num_blocks_per_seq,
    int64_t const num_elem_per_thread)
{
    using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int offset = threadIdx.x * num_elem_per_thread;
    int cnt = 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    for (int i = 0; i < num_elem_per_thread; i++)
    {
        if (offset + i < num_experts_per_node * num_blocks_per_seq)
        {
            cnt += blocked_expert_counts[offset + i];
        }
    }

    int cumsum;
    BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

    for (int i = 0; i < num_elem_per_thread; i++)
    {
        if (offset + i < num_experts_per_node * num_blocks_per_seq)
        {
            blocked_expert_counts_cumsum[offset + i] = cumsum;
            if ((offset + i) % num_blocks_per_seq == 0)
            {
                expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
            }
            cumsum += blocked_expert_counts[offset + i];
            if ((offset + i) == num_experts_per_node * num_blocks_per_seq - 1)
            {
                expert_first_token_offset[num_experts_per_node] = cumsum;
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumKernel(int const* blocked_expert_counts, int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset, int64_t const num_experts_per_node, int64_t const num_blocks_per_seq)
{
    using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int const cnt = threadIdx.x < num_experts_per_node * num_blocks_per_seq ? blocked_expert_counts[threadIdx.x] : 0;
    int cumsum;
    BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

    if (threadIdx.x < num_experts_per_node * num_blocks_per_seq)
    {
        blocked_expert_counts_cumsum[threadIdx.x] = cumsum;
        if (threadIdx.x % num_blocks_per_seq == 0)
        {
            expert_first_token_offset[threadIdx.x / num_blocks_per_seq] = cumsum;
        }
        if (threadIdx.x == num_experts_per_node * num_blocks_per_seq - 1)
        {
            expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void global_expert_prefix_sum(int const* blocked_expert_counts, int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset, int64_t const num_experts_per_node, int64_t const num_blocks_per_seq,
    cudaStream_t stream)
{
    int64_t const num_elements = num_experts_per_node * num_blocks_per_seq;

    if (num_elements <= 1024)
    {
        auto func = globalExpertPrefixSumKernel<1024>;
        int block_dim = 1024;
        if (num_elements <= 32)
        {
            func = globalExpertPrefixSumKernel<32>;
            block_dim = 32;
        }
        else if (num_elements <= 64)
        {
            func = globalExpertPrefixSumKernel<64>;
            block_dim = 64;
        }
        else if (num_elements <= 128)
        {
            func = globalExpertPrefixSumKernel<128>;
            block_dim = 128;
        }
        else if (num_elements <= 256)
        {
            func = globalExpertPrefixSumKernel<256>;
            block_dim = 256;
        }
        else if (num_elements <= 512)
        {
            func = globalExpertPrefixSumKernel<512>;
            block_dim = 512;
        }
        func<<<1, block_dim, 0, stream>>>(
            blocked_expert_counts, blocked_expert_counts_cumsum, expert_first_token_offset,
            num_experts_per_node, num_blocks_per_seq);
    }
    else
    {
        int64_t const num_elem_per_thread = CEILDIV(num_elements, 1024);
        globalExpertPrefixSumLargeKernel<1024><<<1, 1024, 0, stream>>>(blocked_expert_counts,
            blocked_expert_counts_cumsum, expert_first_token_offset, num_experts_per_node, num_blocks_per_seq,
            num_elem_per_thread);
    }
}

__global__ void mergeExpertPrefixSumKernel(int const* blocked_expert_counts, int const* blocked_expert_counts_cumsum,
    int const* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int const num_tokens)
{
    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;
    int const num_blocks_per_seq = gridDim.y;
    int const token_id = block_id * blockDim.x + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int const cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
    int const offset = blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
    if (threadIdx.x < cnt)
    {
        int const unpermuted_row = blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
        int const permuted_row = offset + threadIdx.x;
        permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
        permuted_token_selected_experts[permuted_row] = target_expert_id;
        unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void merge_expert_prefix_sum(int const* blocked_expert_counts, int const* blocked_expert_counts_cumsum,
    int const* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t const num_tokens, int64_t const num_experts_per_node,
    int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq, cudaStream_t stream)
{
    dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
    dim3 const threads(num_tokens_per_block);
    mergeExpertPrefixSumKernel<<<blocks, threads, 0, stream>>>(blocked_expert_counts, blocked_expert_counts_cumsum,
        blocked_row_to_unpermuted_row, permuted_token_selected_experts, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, num_tokens);
}

void three_step_build_expert_maps_sort_first_token(int const* token_selected_experts,
    int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset, int* blocked_expert_counts, int* blocked_expert_counts_cumsum,
    int* blocked_row_to_unpermuted_row, int64_t const num_tokens, int64_t const num_experts_per_node,
    int64_t const num_experts_per_token, int const start_expert_id, cudaStream_t stream)
{
    int64_t const num_tokens_per_block = compute_num_tokens_per_block(num_tokens, num_experts_per_node);
    int64_t const num_blocks_per_seq = CEILDIV(num_tokens, num_tokens_per_block);

    block_expert_prefix_sum(token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row, num_tokens,
        num_experts_per_node, num_experts_per_token, num_tokens_per_block, num_blocks_per_seq, start_expert_id, stream);
    global_expert_prefix_sum(
        blocked_expert_counts, blocked_expert_counts_cumsum, expert_first_token_offset, num_experts_per_node,
        num_blocks_per_seq, stream);
    merge_expert_prefix_sum(blocked_expert_counts, blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row,
        permuted_token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, num_tokens,
        num_experts_per_node, num_tokens_per_block, num_blocks_per_seq, stream);
}

} // namespace trtllm_aux

#ifdef BENCH
#include "bench_timer.h"

namespace
{

enum class ExpertMapMode
{
    Auto,
    Fused,
    ThreeStep,
};

ExpertMapMode parse_mode(char const* value)
{
    if (std::strcmp(value, "auto") == 0)
    {
        return ExpertMapMode::Auto;
    }
    if (std::strcmp(value, "fused") == 0)
    {
        return ExpertMapMode::Fused;
    }
    if (std::strcmp(value, "three_step") == 0 || std::strcmp(value, "three-step") == 0 || std::strcmp(value, "three") == 0)
    {
        return ExpertMapMode::ThreeStep;
    }
    std::fprintf(stderr, "unsupported mode: %s\n", value);
    std::exit(1);
}

char const* mode_name(ExpertMapMode mode)
{
    switch (mode)
    {
    case ExpertMapMode::Auto: return "auto";
    case ExpertMapMode::Fused: return "fused";
    case ExpertMapMode::ThreeStep: return "three_step";
    }
    return "unknown";
}

ExpertMapMode resolve_mode(ExpertMapMode mode, int tokens, int experts, int topk)
{
    if (mode == ExpertMapMode::Auto)
    {
        return trtllm_aux::fused_expert_map_supported(tokens, experts, topk) ? ExpertMapMode::Fused
                                                                            : ExpertMapMode::ThreeStep;
    }
    if (mode == ExpertMapMode::Fused)
    {
        TRTLLM_AUX_CHECK(trtllm_aux::fused_expert_map_supported(tokens, experts, topk),
            "fused expert map requires tokens<=256, experts representable with <=9 bits, and topk in {1,2,4,6,8}");
    }
    return mode;
}

} // namespace

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int tokens = (argc > 1) ? std::atoi(argv[1]) : 1;
    int experts = (argc > 2) ? std::atoi(argv[2]) : 64;
    int topk = (argc > 3) ? std::atoi(argv[3]) : 8;
    ExpertMapMode requested_mode = (argc > 4) ? parse_mode(argv[4]) : ExpertMapMode::Auto;
    ExpertMapMode selected_mode = resolve_mode(requested_mode, tokens, experts, topk);

    TRTLLM_AUX_CHECK(tokens > 0, "tokens must be > 0");
    TRTLLM_AUX_CHECK(experts > 0, "experts must be > 0");
    TRTLLM_AUX_CHECK(topk > 0 && topk <= experts, "topk must be in [1, experts]");

    int64_t const expanded_tokens = static_cast<int64_t>(tokens) * topk;
    int64_t const num_tokens_per_block = trtllm_aux::compute_num_tokens_per_block(tokens, experts);
    int64_t const num_blocks_per_seq = CEILDIV(tokens, num_tokens_per_block);

    std::printf(
        "bench trtllm moe_expert_map: tokens=%d experts=%d topk=%d requested=%s selected=%s "
        "tokens_per_block=%lld blocks_per_seq=%lld\n",
        tokens, experts, topk, mode_name(requested_mode), mode_name(selected_mode),
        static_cast<long long>(num_tokens_per_block), static_cast<long long>(num_blocks_per_seq));

    std::vector<int> h_selected(expanded_tokens);
    for (int token = 0; token < tokens; ++token)
    {
        for (int k = 0; k < topk; ++k)
        {
            h_selected[token * topk + k] = (token + k) % experts;
        }
    }

    int* d_selected = nullptr;
    int* d_permuted_token_selected_experts = nullptr;
    int* d_permuted_row_to_unpermuted_row = nullptr;
    int* d_unpermuted_row_to_permuted_row = nullptr;
    int* d_blocked_expert_counts = nullptr;
    int* d_blocked_expert_counts_cumsum = nullptr;
    int* d_blocked_row_to_unpermuted_row = nullptr;
    int64_t* d_expert_first_token_offset = nullptr;

    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_selected, h_selected.size() * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_permuted_token_selected_experts, expanded_tokens * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_permuted_row_to_unpermuted_row, expanded_tokens * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_unpermuted_row_to_permuted_row, expanded_tokens * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_expert_first_token_offset, (experts + 1) * sizeof(int64_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_blocked_expert_counts, experts * num_blocks_per_seq * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_blocked_expert_counts_cumsum, experts * num_blocks_per_seq * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_blocked_row_to_unpermuted_row, static_cast<int64_t>(experts) * tokens * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(
        cudaMemcpy(d_selected, h_selected.data(), h_selected.size() * sizeof(int), cudaMemcpyHostToDevice));
    TRTLLM_AUX_CUDA_CHECK(cudaMemset(d_permuted_token_selected_experts, 0, expanded_tokens * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemset(d_permuted_row_to_unpermuted_row, 0, expanded_tokens * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemset(d_unpermuted_row_to_permuted_row, 0, expanded_tokens * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemset(d_expert_first_token_offset, 0, (experts + 1) * sizeof(int64_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemset(d_blocked_expert_counts, 0, experts * num_blocks_per_seq * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemset(d_blocked_expert_counts_cumsum, 0, experts * num_blocks_per_seq * sizeof(int)));
    TRTLLM_AUX_CUDA_CHECK(
        cudaMemset(d_blocked_row_to_unpermuted_row, 0, static_cast<int64_t>(experts) * tokens * sizeof(int)));

    timer.run([&]() {
        if (selected_mode == ExpertMapMode::Fused)
        {
            bool ok = trtllm_aux::fused_build_expert_maps_sort_first_token(d_selected,
                d_permuted_row_to_unpermuted_row, d_unpermuted_row_to_permuted_row, d_expert_first_token_offset,
                tokens, experts, topk, 0, experts, 0);
            TRTLLM_AUX_CHECK(ok, "fused_build_expert_maps_sort_first_token returned false");
        }
        else
        {
            trtllm_aux::three_step_build_expert_maps_sort_first_token(d_selected,
                d_permuted_token_selected_experts, d_permuted_row_to_unpermuted_row,
                d_unpermuted_row_to_permuted_row, d_expert_first_token_offset, d_blocked_expert_counts,
                d_blocked_expert_counts_cumsum, d_blocked_row_to_unpermuted_row, tokens, experts, topk, 0, 0);
        }
    });

    TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
    TRTLLM_AUX_CUDA_CHECK(cudaDeviceSynchronize());

    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_selected));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_permuted_token_selected_experts));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_permuted_row_to_unpermuted_row));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_unpermuted_row_to_permuted_row));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_expert_first_token_offset));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_blocked_expert_counts));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_blocked_expert_counts_cumsum));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_blocked_row_to_unpermuted_row));

    return 0;
}
#endif
