// MoE token alignment kernel: sort tokens by expert, pad to block_size
// Extracted from vllm/csrc/moe/moe_align_sum_kernels.cu
// Uses CUB BlockScan for prefix sum.
#include "moe_compat.h"

namespace vllm { namespace moe {

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
    int num_tokens_x_topk, int num_experts, int block_size,
    int max_num_tokens_padded, cudaStream_t stream)
{
    constexpr int fill_threads = 256;
    int threads = max(num_experts, 32);
    int shared = ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

    vllm::moe::moe_align_block_size_small_batch_expert_kernel<int32_t, fill_threads>
        <<<1, fill_threads + threads, shared, stream>>>(
            topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad,
            num_experts, block_size, num_tokens_x_topk, max_num_tokens_padded);
}

// ============================================================================
#ifdef BENCH
#include <cstdlib>
#include <vector>

// Usage: ./moe_align [num_tokens] [num_experts] [topk] [block_size]
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1;
    int E = (argc > 2) ? atoi(argv[2]) : 64;
    int K = (argc > 3) ? atoi(argv[3]) : 8;
    int BS = (argc > 4) ? atoi(argv[4]) : 16;
    int numel = M * K;
    int padded = CEILDIV(numel, BS) * BS * E;  // generous padding
    printf("bench moe_align: tokens=%d experts=%d topk=%d block=%d\n", M, E, K, BS);

    std::vector<int32_t> h_ids(numel);
    srand(42);
    for (auto& v : h_ids) v = rand() % E;

    int32_t *d_ids, *d_sorted, *d_experts, *d_npost;
    cudaMalloc(&d_ids, numel * sizeof(int32_t));
    cudaMalloc(&d_sorted, padded * sizeof(int32_t));
    cudaMalloc(&d_experts, (padded / BS) * sizeof(int32_t));
    cudaMalloc(&d_npost, sizeof(int32_t));
    cudaMemcpy(d_ids, h_ids.data(), numel * sizeof(int32_t), cudaMemcpyHostToDevice);

    moe_align_block_size_launch(d_ids, d_sorted, d_experts, d_npost,
                                numel, E, BS, padded, 0);
    cudaDeviceSynchronize();

    cudaFree(d_ids); cudaFree(d_sorted); cudaFree(d_experts); cudaFree(d_npost);
    printf("Done.\n");
    return 0;
}
#endif
