// MoE token alignment kernel: sort tokens by expert, pad to block_size
// Extracted from vllm/csrc/moe/moe_align_sum_kernels.cu
// Uses CUB BlockScan for prefix sum.
#include "moe_compat.h"

namespace vllm { namespace moe {

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t padded_num_experts,
    int32_t experts_per_warp, int32_t block_size, size_t numel,
    int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded)
{
    extern __shared__ int32_t shared_counts[];

    if (blockIdx.x % 2) {
        for (size_t i = threadIdx.x; i < static_cast<size_t>(max_num_tokens_padded); i += blockDim.x)
            sorted_token_ids[i] = numel;
        return;
    }

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const my_expert_start = warp_id * experts_per_warp;
    for (int i = 0; i < experts_per_warp; ++i) {
        if (my_expert_start + i < padded_num_experts)
            shared_counts[warp_id * experts_per_warp + i] = 0;
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
        int expert_id = topk_ids[i];
        if (expert_id >= num_experts)
            continue;
        int const warp_idx = expert_id / experts_per_warp;
        int const expert_offset = expert_id % experts_per_warp;
        atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }
    __syncthreads();

    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_count = 0;
    int const expert_id = threadIdx.x;
    if (expert_id < num_experts) {
        int const warp_idx = expert_id / experts_per_warp;
        int const expert_offset = expert_id % experts_per_warp;
        expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
        expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    int cumsum_val = 0;
    BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
    if (expert_id <= num_experts)
        cumsum[expert_id] = cumsum_val;
    if (expert_id == num_experts)
        total_tokens_post_pad[0] = cumsum_val;
    __syncthreads();

    if (threadIdx.x < static_cast<unsigned>(num_experts)) {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size)
            expert_ids[i / block_size] = threadIdx.x;
    }

    size_t const fill_start = cumsum[num_experts] / block_size + threadIdx.x;
    int32_t const max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start; i < static_cast<size_t>(max_num_m_blocks); i += blockDim.x)
        expert_ids[i] = -1;
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel, int32_t num_experts, int32_t max_num_tokens_padded)
{
    size_t const tid = blockIdx.y * blockDim.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.y;

    for (size_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        if (expert_id >= num_experts)
            continue;
        int32_t const rank = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank] = i;
    }
}

// Small-batch-expert variant: all work in one threadblock, uses shared memory
template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded)
{
    int32_t max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);

    if (threadIdx.x < fill_threads) {
        for (size_t i = threadIdx.x; i < max_num_tokens_padded; i += fill_threads)
            sorted_token_ids[i] = numel;
        __syncthreads(); __syncthreads(); __syncthreads();
        return;
    }

    const size_t tid = threadIdx.x - fill_threads;
    const size_t stride = blockDim.x - fill_threads;

    extern __shared__ int32_t shared_mem[];
    int32_t* cumsum = shared_mem;
    int32_t* tokens_cnts = shared_mem + num_experts + 1;

    for (int i = 0; i < num_experts; ++i)
        tokens_cnts[(tid + 1) * num_experts + i] = 0;

    for (size_t i = tid; i < numel; i += stride) {
        int32_t eid = topk_ids[i];
        if (eid < num_experts) tokens_cnts[(tid + 1) * num_experts + eid]++;
    }
    __syncthreads();

    if (tid < (size_t)num_experts) {
        tokens_cnts[tid] = 0;
        for (size_t i = 1; i <= stride; ++i)
            tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
    }
    __syncthreads();

    if (tid == 0) {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i)
            cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
        total_tokens_post_pad[0] = cumsum[num_experts];
    }
    __syncthreads();

    if (tid < (size_t)num_experts)
        for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size)
            expert_ids[i / block_size] = tid;

    size_t fill_start = cumsum[num_experts] / block_size + tid;
    for (size_t i = fill_start; i < (size_t)max_num_m_blocks; i += stride)
        expert_ids[i] = -1;

    for (size_t i = tid; i < numel; i += stride) {
        int32_t eid = topk_ids[i];
        if (eid < num_experts) {
            int32_t rank = tokens_cnts[tid * num_experts + eid] + cumsum[eid];
            sorted_token_ids[rank] = i;
            ++tokens_cnts[tid * num_experts + eid];
        }
    }
}

}} // namespace vllm::moe

// ---- Standalone launch ----
void moe_align_block_size_launch(
    const int32_t* topk_ids,          // [num_tokens * topk]
    int32_t* sorted_token_ids,        // [max_num_tokens_padded]
    int32_t* expert_ids,              // [max_num_m_blocks]
    int32_t* num_tokens_post_pad,     // [1]
    int32_t* cumsum_buffer,           // [num_experts + 1]
    int num_tokens_x_topk, int num_experts, int block_size,
    int max_num_tokens_padded, cudaStream_t stream)
{
    bool const small_batch_expert_mode = (num_tokens_x_topk < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode) {
        constexpr int fill_threads = 256;
        int threads = max(num_experts, 32);
        int shared = ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

        vllm::moe::moe_align_block_size_small_batch_expert_kernel<int32_t, fill_threads>
            <<<1, fill_threads + threads, shared, stream>>>(
                topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad,
                num_experts, block_size, num_tokens_x_topk, max_num_tokens_padded);
    } else {
        int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        int experts_per_warp = WARP_SIZE;
        int threads = 1024;
        size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
        size_t shared = num_warps * experts_per_warp * sizeof(int32_t);

        vllm::moe::moe_align_block_size_kernel<int32_t>
            <<<2, threads, shared, stream>>>(
                topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad,
                num_experts, padded_num_experts, experts_per_warp, block_size,
                num_tokens_x_topk, cumsum_buffer, max_num_tokens_padded);

        int const block_threads = 256;
        int const num_blocks = (num_tokens_x_topk + block_threads - 1) / block_threads;
        int const actual_blocks = min(num_blocks, 65535);
        dim3 grid(1, actual_blocks);
        vllm::moe::count_and_sort_expert_tokens_kernel<int32_t>
            <<<grid, block_threads, 0, stream>>>(
                topk_ids, sorted_token_ids, cumsum_buffer, num_tokens_x_topk,
                num_experts, max_num_tokens_padded);
    }
}

// ============================================================================
#ifdef BENCH
#include <cstdlib>
#include <vector>
#include "bench_timer.h"

// Usage: ./moe_align [num_tokens] [num_experts] [topk] [block_size] [--bench warmup iters]
int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int M = (argc > 1) ? atoi(argv[1]) : 1;
    int E = (argc > 2) ? atoi(argv[2]) : 64;
    int K = (argc > 3) ? atoi(argv[3]) : 8;
    int BS = (argc > 4) ? atoi(argv[4]) : 16;
    int numel = M * K;
    int padded = numel + E * (BS - 1);
    if (numel < E)
        padded = min(numel * BS, padded);
    printf("bench moe_align: tokens=%d experts=%d topk=%d block=%d\n", M, E, K, BS);

    std::vector<int32_t> h_ids(numel);
    srand(42);
    for (auto& v : h_ids) v = rand() % E;

    int32_t *d_ids, *d_sorted, *d_experts, *d_npost, *d_cumsum;
    cudaMalloc(&d_ids, numel * sizeof(int32_t));
    cudaMalloc(&d_sorted, padded * sizeof(int32_t));
    cudaMalloc(&d_experts, (padded / BS) * sizeof(int32_t));
    cudaMalloc(&d_npost, sizeof(int32_t));
    cudaMalloc(&d_cumsum, (E + 1) * sizeof(int32_t));
    cudaMemcpy(d_ids, h_ids.data(), numel * sizeof(int32_t), cudaMemcpyHostToDevice);

    timer.run([&]() {
        moe_align_block_size_launch(d_ids, d_sorted, d_experts, d_npost,
                                    d_cumsum, numel, E, BS, padded, 0);
    });
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "moe_align launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_ids); cudaFree(d_sorted); cudaFree(d_experts); cudaFree(d_npost); cudaFree(d_cumsum);
        return 1;
    }

    cudaFree(d_ids); cudaFree(d_sorted); cudaFree(d_experts); cudaFree(d_npost); cudaFree(d_cumsum);
    printf("Done.\n");
    return 0;
}
#endif
