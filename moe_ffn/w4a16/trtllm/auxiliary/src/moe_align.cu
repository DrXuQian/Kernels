// TensorRT-LLM MoE align kernel standalone benchmark.
// Extracted from:
//   cpp/tensorrt_llm/kernels/moeAlignKernels.cu

#include "trtllm_aux_compat.h"

#include <cstring>
#include <string>
#include <vector>

namespace trtllm_aux
{

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts, int32_t padded_num_experts,
    int32_t experts_per_warp, int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded)
{
    extern __shared__ int32_t shared_counts[];

    for (size_t it = threadIdx.x; it < static_cast<size_t>(max_num_tokens_padded); it += blockDim.x)
    {
        sorted_token_ids[it] = numel;
    }

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const my_expert_start = warp_id * experts_per_warp;

    for (int i = 0; i < experts_per_warp; ++i)
    {
        if (my_expert_start + i < padded_num_experts)
        {
            shared_counts[warp_id * experts_per_warp + i] = 0;
        }
    }
    __syncthreads();

    size_t const tid = threadIdx.x;
    size_t const stride = blockDim.x;

    for (size_t i = tid; i < numel; i += stride)
    {
        int expert_id = topk_ids[i];
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }
    __syncthreads();

    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_count = 0;
    int expert_id = threadIdx.x;
    if (expert_id < num_experts)
    {
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
        expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    int cumsum_val;
    BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
    if (expert_id <= num_experts)
    {
        cumsum[expert_id] = cumsum_val;
    }
    if (expert_id == num_experts)
    {
        *total_tokens_post_pad = cumsum_val;
    }
    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size)
        {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    size_t const fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
    size_t const expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x)
    {
        expert_ids[i] = 0;
    }
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer, size_t numel)
{
    size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < numel; i += stride)
    {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}

template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts, int32_t block_size, size_t numel,
    int32_t max_num_tokens_padded)
{
    for (size_t it = threadIdx.x; it < static_cast<size_t>(max_num_tokens_padded); it += blockDim.x)
    {
        sorted_token_ids[it] = numel;
    }

    size_t const tid = threadIdx.x;
    size_t const stride = blockDim.x;

    extern __shared__ int32_t shared_mem[];
    int32_t* cumsum = shared_mem;
    int32_t* tokens_cnts = shared_mem + num_experts + 1;

    for (int i = 0; i < num_experts; ++i)
    {
        tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride)
    {
        ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
    }
    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        tokens_cnts[threadIdx.x] = 0;
        for (int i = 1; i <= blockDim.x; ++i)
        {
            tokens_cnts[i * num_experts + threadIdx.x] += tokens_cnts[(i - 1) * num_experts + threadIdx.x];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i)
        {
            cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) * block_size;
        }
        *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
    }
    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size)
        {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    size_t const fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
    size_t const expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x)
    {
        expert_ids[i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride)
    {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = tokens_cnts[threadIdx.x * num_experts + expert_id] + cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
        ++tokens_cnts[threadIdx.x * num_experts + expert_id];
    }
}

template <typename scalar_t>
void moe_align_block_size_small_launch(scalar_t const* topk_ids, int32_t* sorted_token_ids, int32_t* expert_ids,
    int32_t* num_tokens_post_pad, int32_t num_experts, int32_t block_size, int32_t numel,
    int32_t max_num_tokens_padded, cudaStream_t stream)
{
    int32_t const thread_count = std::max(num_experts, WARP_SIZE);
    int32_t const shared_mem_size = ((thread_count + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
    moe_align_block_size_small_batch_expert_kernel<scalar_t><<<1, thread_count, shared_mem_size, stream>>>(
        topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, num_experts, block_size, numel,
        max_num_tokens_padded);
}

template <typename scalar_t>
void moe_align_block_size_generic_launch(scalar_t const* topk_ids, int32_t* sorted_token_ids, int32_t* expert_ids,
    int32_t* num_tokens_post_pad, int32_t* cumsum_buffer, int32_t num_experts, int32_t block_size, int32_t numel,
    int32_t max_num_tokens_padded, cudaStream_t stream)
{
    int32_t const padded_num_experts = CEILDIV(num_experts, WARP_SIZE) * WARP_SIZE;
    int32_t const experts_per_warp = WARP_SIZE;
    int32_t const threads = 1024;
    int32_t const num_warps = CEILDIV(padded_num_experts, experts_per_warp);
    size_t const shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

    TRTLLM_AUX_CUDA_CHECK(cudaMemsetAsync(cumsum_buffer, 0, (num_experts + 1) * sizeof(int32_t), stream));
    moe_align_block_size_kernel<scalar_t><<<1, threads, shared_mem_size, stream>>>(topk_ids, sorted_token_ids,
        expert_ids, num_tokens_post_pad, num_experts, padded_num_experts, experts_per_warp, block_size, numel,
        cumsum_buffer, max_num_tokens_padded);

    int const block_threads = 256;
    int const num_blocks = (numel + block_threads - 1) / block_threads;
    int const actual_blocks = std::min(num_blocks, 65535);
    count_and_sort_expert_tokens_kernel<scalar_t>
        <<<actual_blocks, block_threads, 0, stream>>>(topk_ids, sorted_token_ids, cumsum_buffer, numel);
}

template <typename scalar_t>
void moe_align_block_size_launch(scalar_t const* topk_ids, int32_t* sorted_token_ids, int32_t* expert_ids,
    int32_t* num_tokens_post_pad, int32_t* cumsum_buffer, int32_t num_experts, int32_t block_size, int32_t numel,
    int32_t max_num_tokens_padded, char const* mode, cudaStream_t stream)
{
    bool const small_auto = (numel < 1024) && (num_experts <= 64);
    bool const use_small = std::strcmp(mode, "small") == 0 || (std::strcmp(mode, "auto") == 0 && small_auto);
    if (use_small)
    {
        moe_align_block_size_small_launch(topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, num_experts,
            block_size, numel, max_num_tokens_padded, stream);
    }
    else
    {
        moe_align_block_size_generic_launch(topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum_buffer,
            num_experts, block_size, numel, max_num_tokens_padded, stream);
    }
}

} // namespace trtllm_aux

#ifdef BENCH
#include "bench_timer.h"

int main(int argc, char** argv)
{
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int tokens = (argc > 1) ? std::atoi(argv[1]) : 1;
    int experts = (argc > 2) ? std::atoi(argv[2]) : 64;
    int topk = (argc > 3) ? std::atoi(argv[3]) : 8;
    int block_size = (argc > 4) ? std::atoi(argv[4]) : 16;
    char const* mode = (argc > 5) ? argv[5] : "auto";
    int numel = tokens * topk;
    int max_padded = CEILDIV(numel, block_size) * block_size * experts;

    std::printf("bench trtllm moe_align: tokens=%d experts=%d topk=%d block=%d mode=%s max_padded=%d\n",
        tokens, experts, topk, block_size, mode, max_padded);

    std::vector<int32_t> h_ids(numel);
    for (int i = 0; i < numel; ++i)
    {
        h_ids[i] = (i * 17 + 3) % experts;
    }

    int32_t* d_ids = nullptr;
    int32_t* d_sorted = nullptr;
    int32_t* d_expert_ids = nullptr;
    int32_t* d_num_post_pad = nullptr;
    int32_t* d_cumsum = nullptr;
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_ids, numel * sizeof(int32_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_sorted, max_padded * sizeof(int32_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_expert_ids, CEILDIV(max_padded, block_size) * sizeof(int32_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_num_post_pad, sizeof(int32_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMalloc(&d_cumsum, (experts + 1) * sizeof(int32_t)));
    TRTLLM_AUX_CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), numel * sizeof(int32_t), cudaMemcpyHostToDevice));

    timer.run([&]() {
        trtllm_aux::moe_align_block_size_launch(d_ids, d_sorted, d_expert_ids, d_num_post_pad, d_cumsum, experts,
            block_size, numel, max_padded, mode, 0);
    });

    TRTLLM_AUX_CUDA_CHECK(cudaGetLastError());
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_ids));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_sorted));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_expert_ids));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_num_post_pad));
    TRTLLM_AUX_CUDA_CHECK(cudaFree(d_cumsum));
    return 0;
}
#endif
