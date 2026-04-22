// TopK Gating kernel: fused softmax + topK selection
// Extracted from vllm/csrc/moe/topk_softmax_kernels.cu
// Supports power-of-2 expert counts (up to 512) with warp-level butterfly reduce.
#include "moe_compat.h"
#include <type_traits>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)

using namespace vllm::moe;

// ---- Fused topkGating kernel (power-of-2 expert counts) ----
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,
          int WARP_SIZE_PARAM, typename IndType, typename InputType, ScoringFunc SF>
__launch_bounds__(WARPS_PER_CTA * WARP_SIZE_PARAM)
__global__ void topkGating(
    const InputType* input, const bool* finished, float* output,
    const int num_rows, IndType* indices, int* source_rows,
    const int k, const int start_expert, const int end_expert,
    const bool renormalize, const float* bias)
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    if (thread_row >= num_rows) return;
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt;

    // Load and convert to float
    float row_chunk[VPT];
    for (int ii = 0; ii < VPT; ++ii) {
        int col = first_elt + (ii / ELTS_PER_LDG) * ELTS_PER_LDG * THREADS_PER_ROW + ii % ELTS_PER_LDG;
        row_chunk[ii] = (col < NUM_EXPERTS) ? toFloat(thread_row_ptr[col]) : 0.0f;
    }

    // Apply scoring
    if constexpr (SF == SCORING_SOFTMAX) {
        float thread_max = row_chunk[0];
        for (int ii = 1; ii < VPT; ++ii) thread_max = max(thread_max, row_chunk[ii]);
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
            thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
        float row_sum = 0;
        for (int ii = 0; ii < VPT; ++ii) { row_chunk[ii] = expf(row_chunk[ii] - thread_max); row_sum += row_chunk[ii]; }
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
            row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
        for (int ii = 0; ii < VPT; ++ii) row_chunk[ii] /= row_sum;
    } else {
        for (int ii = 0; ii < VPT; ++ii)
            row_chunk[ii] = 1.0f / (1.0f + __expf(-row_chunk[ii]));
    }

    // TopK selection with butterfly argmax
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float row_chunk_for_choice[VPT];
    for (int ii = 0; ii < VPT; ++ii) {
        int col = first_elt + (ii / ELTS_PER_LDG) * COLS_PER_GROUP_LDG + ii % ELTS_PER_LDG;
        row_chunk_for_choice[ii] = row_chunk[ii] + ((bias && col < NUM_EXPERTS) ? bias[col] : 0.0f);
    }

    int start_col = first_elt;
    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        float max_val_c = row_chunk_for_choice[0], max_val = row_chunk[0];
        int expert = start_col;
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
                if (row_chunk_for_choice[ldg * ELTS_PER_LDG + ii] > max_val_c) {
                    max_val_c = row_chunk_for_choice[ldg * ELTS_PER_LDG + ii];
                    max_val = row_chunk[ldg * ELTS_PER_LDG + ii];
                    expert = col + ii;
                }
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float o_c = VLLM_SHFL_XOR_SYNC_WIDTH(max_val_c, mask, THREADS_PER_ROW);
            float o_v = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int o_e = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);
            if (o_c > max_val_c || (o_c == max_val_c && o_e < expert)) { max_val_c = o_c; max_val = o_v; expert = o_e; }
        }
        if (thread_group_idx == 0) {
            bool use = row_is_active && expert >= start_expert && expert < end_expert;
            int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = use ? (expert - start_expert) : NUM_EXPERTS;
            source_rows[idx] = k_idx * num_rows + thread_row;
            if (renormalize) selected_sum += max_val;
        }
        if (k_idx + 1 < k) {
            int ldg_g = expert / COLS_PER_GROUP_LDG;
            int t_clear = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (thread_group_idx == t_clear)
                row_chunk_for_choice[ldg_g * ELTS_PER_LDG + expert % ELTS_PER_LDG] = -10000.f;
        }
    }
    if (renormalize && thread_group_idx == 0) {
        float denom = selected_sum > 0.f ? selected_sum : 1.f;
        for (int k_idx = 0; k_idx < k; ++k_idx) output[k * thread_row + k_idx] /= denom;
    }
}

// ---- Launch helper ----
namespace detail {
template <int E, int BPL, int WS, typename IT>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BPL / sizeof(IT);
    static constexpr int VECs_PER_THREAD = MAX(1, E / (ELTS_PER_LDG * WS));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = E / VPT;
    static constexpr int ROWS_PER_WARP = WS / THREADS_PER_ROW;
};
}

// Standalone launch: float gating input, int32 indices, softmax scoring
void topk_gating_softmax_launch(
    const float* gating_output,  // [num_tokens, num_experts]
    float* topk_weights,         // [num_tokens, topk]
    int* topk_indices,           // [num_tokens, topk]
    int* source_rows,            // [num_tokens * topk]
    int num_tokens, int num_experts, int topk, bool renormalize,
    cudaStream_t stream)
{
    constexpr int WARPS = 4;
    constexpr int BPL = 16;
    constexpr int WS = 32;

    #define LAUNCH_CASE(NE) { \
        using C = detail::TopkConstants<NE, MIN(BPL, (int)sizeof(float)*NE), WS, float>; \
        int rows_per_warp = C::ROWS_PER_WARP; \
        int nw = (num_tokens + rows_per_warp - 1) / rows_per_warp; \
        int nb = (nw + WARPS - 1) / WARPS; \
        dim3 block(WS, WARPS); \
        topkGating<C::VPT, NE, WARPS, MIN(BPL, (int)sizeof(float)*NE), WS, int, float, SCORING_SOFTMAX> \
            <<<nb, block, 0, stream>>>(gating_output, nullptr, topk_weights, num_tokens, \
                topk_indices, source_rows, topk, 0, num_experts, renormalize, nullptr); }

    switch (num_experts) {
        case 8:   LAUNCH_CASE(8); break;
        case 16:  LAUNCH_CASE(16); break;
        case 32:  LAUNCH_CASE(32); break;
        case 64:  LAUNCH_CASE(64); break;
        case 128: LAUNCH_CASE(128); break;
        case 256: LAUNCH_CASE(256); break;
        default: fprintf(stderr, "topk_gating: unsupported num_experts=%d\n", num_experts); abort();
    }
    #undef LAUNCH_CASE
}

// ============================================================================
#ifdef BENCH
#include <cstdlib>
#include <vector>
#include "bench_timer.h"

// Usage: ./bench_topk_gating [num_tokens] [num_experts] [topk] [--bench warmup iters]
int main(int argc, char** argv) {
    BenchTimer timer;
    timer.parse(argc, argv);
    argc = BenchTimer::strip_bench_args(argc, argv);

    int M = (argc > 1) ? atoi(argv[1]) : 1;
    int E = (argc > 2) ? atoi(argv[2]) : 64;
    int K = (argc > 3) ? atoi(argv[3]) : 8;
    printf("bench topk_gating: tokens=%d experts=%d topk=%d\n", M, E, K);

    float *d_gating, *d_weights;
    int *d_indices, *d_source;
    cudaMalloc(&d_gating, (long long)M * E * sizeof(float));
    cudaMalloc(&d_weights, (long long)M * K * sizeof(float));
    cudaMalloc(&d_indices, (long long)M * K * sizeof(int));
    cudaMalloc(&d_source, (long long)M * K * sizeof(int));

    std::vector<float> h(M * E);
    srand(42);
    for (auto& v : h) v = (float)rand() / RAND_MAX;
    cudaMemcpy(d_gating, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);

    timer.run([&]() {
        topk_gating_softmax_launch(d_gating, d_weights, d_indices, d_source, M, E, K, true, 0);
    });

    cudaFree(d_gating); cudaFree(d_weights); cudaFree(d_indices); cudaFree(d_source);
    printf("Done.\n");
    return 0;
}
#endif
