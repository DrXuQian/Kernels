// Correct split-sequence study for FlashInfer GDN prefill.
//
// The default GDN prefill kernel exposes only num_seqs * num_v_heads CTAs. For
// Qwen3.5-122B-A10B prefill that is 1 * 64 CTAs, which underfills H800.
// This study uses the existing checkpoint and init-state paths to test whether
// splitting one long sequence into several shorter logical sequences can raise
// the second-stage grid without resetting the recurrent state.
//
// Usage:
//   ./bench_gdn_splitseq_study_single_tu [seqlen] [q_heads] [v_heads] [head_dim] \
//       --segment-tokens 1024 --mode split --bench 10 50
//
// Modes:
//   checkpoint: time the full-sequence checkpoint pass.
//   split:      prepare checkpoints once, then time only the split init-state pass.
//   both:       time checkpoint + pack-state + split pass together.
//   state_checkpoint/state_split/state_both:
//               same, but the checkpoint pass uses the study-only state-only kernel.
//   scan_transition/scan_split/scan_both:
//               compute per-segment state transitions in parallel, compose the
//               segment prefix states, then run the split output pass.
//   zero_split: compute per-segment zero-state output and state transitions.
//   correction_full:
//               prepare scan prefix states, then time an exact correction pass
//               using InitStateFromInput=true and V=0.
//   cluster_scan_split/cluster_scan_both:
//               same as scan_split/scan_both, but compose segment prefix states
//               with one thread-block-cluster kernel.
//   zero_v_correction_full:
//               same prefix prep as correction_full, then time a study-only
//               InitStateFromInput correction collective that skips V loads.
//   stream_segments:
//               process segments sequentially with state handoff across launches.
//   stream_segments_post_serial/stream_segments_post_overlap:
//               add a synthetic per-segment consumer kernel after each GDN
//               segment, either serialized on the GDN stream or overlapped on a
//               second stream. This tests whether idle SMs can be used by later
//               chunk-local work without changing GDN recurrence semantics.
//   stream_segments_rms_gate_serial/stream_segments_rms_gate_overlap:
//               same scheduling test using the real linear-attention fused
//               RMSNorm+sigmoid-gate kernel shape, (segment_tokens*num_v_heads, head_dim).

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bench_timer.h"

namespace cg = cooperative_groups;

namespace gdn_splitseq_study {
using T = nv_bfloat16;

void launch_gdn_prefill_bf16_gva_checkpoint(
    cudaStream_t stream, T* output, float* output_state, T const* q, T const* k, T const* v,
    float const* alpha, float const* beta, int64_t const* cu_seqlens, uint8_t* workspace,
    int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads,
    int32_t num_o_heads, int32_t head_size, int64_t total_seqlen, float scale, int32_t sm_count,
    float* state_checkpoints, int64_t const* checkpoint_cu_starts,
    int32_t checkpoint_every_n_tokens);

void launch_gdn_prefill_bf16_gva_zero_state(
    cudaStream_t stream, T* output, float* output_state, T const* q, T const* k, T const* v,
    float const* alpha, float const* beta, int64_t const* cu_seqlens, uint8_t* workspace,
    int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads,
    int32_t num_o_heads, int32_t head_size, int64_t total_seqlen, float scale,
    int32_t sm_count);

void launch_gdn_prefill_bf16_gva_initstate(
    cudaStream_t stream, T* output, float* output_state, T const* q, T const* k, T const* v,
    float const* input_state, float const* alpha, float const* beta, int64_t const* cu_seqlens,
    uint8_t* workspace, int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads,
    int32_t num_v_heads, int32_t num_o_heads, int32_t head_size, int64_t total_seqlen,
    float scale, int32_t sm_count);
}  // namespace gdn_splitseq_study

namespace gdn_state_only_study {
using T = nv_bfloat16;

void launch_gdn_prefill_bf16_gva_state_only_checkpoint(
    cudaStream_t stream, T* output, float* output_state, T const* q, T const* k, T const* v,
    float const* alpha, float const* beta, int64_t const* cu_seqlens, uint8_t* workspace,
    int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads,
    int32_t num_o_heads, int32_t head_size, int64_t total_seqlen, float scale, int32_t sm_count,
    float* state_checkpoints, int64_t const* checkpoint_cu_starts,
    int32_t checkpoint_every_n_tokens);

void launch_gdn_prefill_bf16_gva_state_only_transition(
    cudaStream_t stream, T* output, float* output_state, T const* q, T const* k, T const* v,
    float const* alpha, float const* beta, int64_t const* cu_seqlens, uint8_t* workspace,
    int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads,
    int32_t num_o_heads, int32_t head_size, int64_t total_seqlen, float scale, int32_t sm_count);
}  // namespace gdn_state_only_study

namespace gdn_zero_v_study {
using T = nv_bfloat16;

void launch_gdn_prefill_bf16_gva_zero_v_initstate(
    cudaStream_t stream, T* output, float* output_state, T const* q, T const* k, T const* v,
    float const* input_state, float const* alpha, float const* beta, int64_t const* cu_seqlens,
    uint8_t* workspace_buffer, int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads,
    int32_t num_v_heads, int32_t num_o_heads, int32_t head_size, int64_t total_seqlen,
    float scale, int32_t sm_count);
}  // namespace gdn_zero_v_study

#define BENCH_CUDA_CHECK(e)                                                                         \
  do {                                                                                              \
    cudaError_t _e = (e);                                                                           \
    if (_e != cudaSuccess) {                                                                        \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));             \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)

__global__ void pack_segment_input_states(float* dst, float const* checkpoints, int segments,
                                          int64_t state_elems_per_segment) {
  int64_t total = static_cast<int64_t>(segments) * state_elems_per_segment;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int seg = static_cast<int>(idx / state_elems_per_segment);
    int64_t off = idx - static_cast<int64_t>(seg) * state_elems_per_segment;
    dst[idx] = (seg == 0) ? 0.0f : checkpoints[static_cast<int64_t>(seg - 1) *
                                               state_elems_per_segment + off];
  }
}

__global__ void compute_segment_coeffs(float* coeffs, float const* alpha,
                                       int64_t const* cu_seqlens, int segments, int num_heads) {
  int idx = blockIdx.x;
  if (idx >= segments * num_heads) {
    return;
  }
  int seg = idx / num_heads;
  int head = idx - seg * num_heads;
  int64_t begin = cu_seqlens[seg];
  int64_t end = cu_seqlens[seg + 1];
  float log_prod = 0.0f;
  for (int64_t tok = begin + threadIdx.x; tok < end; tok += blockDim.x) {
    log_prod += log2f(alpha[tok * num_heads + head] + 1.0e-10f);
  }

  __shared__ float smem[256];
  smem[threadIdx.x] = log_prod;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    coeffs[idx] = exp2f(smem[0]);
  }
}

__global__ void compose_segment_input_states(float* dst, float const* segment_states,
                                             float const* coeffs, int segments, int num_heads,
                                             int64_t state_elems_per_head) {
  int64_t state_elems_per_segment = static_cast<int64_t>(num_heads) * state_elems_per_head;
  int64_t total = static_cast<int64_t>(segments) * state_elems_per_segment;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int seg = static_cast<int>(idx / state_elems_per_segment);
    int64_t rem = idx - static_cast<int64_t>(seg) * state_elems_per_segment;
    int head = static_cast<int>(rem / state_elems_per_head);
    int64_t elem = rem - static_cast<int64_t>(head) * state_elems_per_head;
    float running = 0.0f;
    for (int j = 0; j < seg; ++j) {
      int64_t off = (static_cast<int64_t>(j) * num_heads + head) * state_elems_per_head + elem;
      running = segment_states[off] + coeffs[j * num_heads + head] * running;
    }
    dst[idx] = running;
  }
}

__global__ void compose_segment_input_states_cluster(float* dst, float const* segment_states,
                                                     float* coeffs, float const* alpha,
                                                     int64_t const* cu_seqlens, int segments,
                                                     int num_heads,
                                                     int64_t state_elems_per_head) {
  cg::cluster_group cluster = cg::this_cluster();
  int seg = cluster.block_rank();
  int head = static_cast<int>(blockIdx.x) / segments;
  int64_t begin = cu_seqlens[seg];
  int64_t end = cu_seqlens[seg + 1];

  float log_prod = 0.0f;
  for (int64_t tok = begin + threadIdx.x; tok < end; tok += blockDim.x) {
    log_prod += log2f(alpha[tok * num_heads + head] + 1.0e-10f);
  }

  __shared__ float smem[256];
  smem[threadIdx.x] = log_prod;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    coeffs[seg * num_heads + head] = exp2f(smem[0]);
  }
  __threadfence();
  cluster.sync();

  int64_t dst_base = (static_cast<int64_t>(seg) * num_heads + head) * state_elems_per_head;
  for (int64_t elem = threadIdx.x; elem < state_elems_per_head; elem += blockDim.x) {
    float running = 0.0f;
    for (int j = 0; j < seg; ++j) {
      int64_t off = (static_cast<int64_t>(j) * num_heads + head) * state_elems_per_head + elem;
      running = segment_states[off] + coeffs[j * num_heads + head] * running;
    }
    dst[dst_base + elem] = running;
  }
}

__global__ void post_consumer_bf16_kernel(nv_bfloat16 const* input, float* output,
                                          int64_t elems, int rounds) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elems;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float acc = __bfloat162float(input[idx]);
    #pragma unroll 4
    for (int r = 0; r < rounds; ++r) {
      acc = fmaf(acc, 1.00012207f, 0.000244140625f);
    }
    output[idx] = acc;
  }
}

__global__ void fused_rms_norm_gate_segment_kernel(
    nv_bfloat16 const* __restrict__ x, nv_bfloat16 const* __restrict__ z,
    nv_bfloat16 const* __restrict__ weight, nv_bfloat16* __restrict__ output, int rows, int dim,
    float eps) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  int tid = threadIdx.x;
  nv_bfloat16 const* x_row = x + static_cast<int64_t>(row) * dim;
  nv_bfloat16 const* z_row = z + static_cast<int64_t>(row) * dim;
  nv_bfloat16* out_row = output + static_cast<int64_t>(row) * dim;

  float local_sum_sq = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    float v = __bfloat162float(x_row[i]);
    local_sum_sq += v * v;
  }

  __shared__ float warp_sums[32];
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int nwarps = (blockDim.x + 31) / 32;

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
  }
  if (lane == 0) {
    warp_sums[warp_id] = local_sum_sq;
  }
  __syncthreads();

  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < nwarps; ++i) {
      total += warp_sums[i];
    }
    warp_sums[0] = total;
  }
  __syncthreads();

  float inv_rms = rsqrtf(warp_sums[0] / dim + eps);
  for (int i = tid; i < dim; i += blockDim.x) {
    float xv = __bfloat162float(x_row[i]) * inv_rms;
    float wv = __bfloat162float(weight[i]);
    float zv = __bfloat162float(z_row[i]);
    float gate = 1.0f / (1.0f + expf(-zv));
    out_row[i] = __float2bfloat16(xv * wv * gate);
  }
}

static int strip_int_arg(int argc, char** argv, char const* name, int* value) {
  int out = 0;
  size_t name_len = strlen(name);
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], name) == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", name);
        exit(1);
      }
      *value = atoi(argv[++i]);
      continue;
    }
    if (strncmp(argv[i], name, name_len) == 0 && argv[i][name_len] == '=') {
      *value = atoi(argv[i] + name_len + 1);
      continue;
    }
    argv[out++] = argv[i];
  }
  return out;
}

static int strip_mode_arg(int argc, char** argv, int* mode, char const** mode_name) {
  int out = 0;
  for (int i = 0; i < argc; ++i) {
    char const* value = nullptr;
    if (strcmp(argv[i], "--mode") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "--mode requires checkpoint, split, both, state_checkpoint, state_split, state_both, scan_transition, scan_split, scan_both, zero_split, correction_full, cluster_scan_split, cluster_scan_both, zero_v_correction_full, stream_segments, stream_segments_post_serial, stream_segments_post_overlap, stream_segments_rms_gate_serial, or stream_segments_rms_gate_overlap\n");
        exit(1);
      }
      value = argv[++i];
    } else if (strncmp(argv[i], "--mode=", 7) == 0) {
      value = argv[i] + 7;
    }
    if (value != nullptr) {
      if (strcmp(value, "checkpoint") == 0) {
        *mode = 0;
      } else if (strcmp(value, "split") == 0) {
        *mode = 1;
      } else if (strcmp(value, "both") == 0) {
        *mode = 2;
      } else if (strcmp(value, "state_checkpoint") == 0) {
        *mode = 3;
      } else if (strcmp(value, "state_split") == 0) {
        *mode = 4;
      } else if (strcmp(value, "state_both") == 0) {
        *mode = 5;
      } else if (strcmp(value, "scan_transition") == 0) {
        *mode = 6;
      } else if (strcmp(value, "scan_split") == 0) {
        *mode = 7;
      } else if (strcmp(value, "scan_both") == 0) {
        *mode = 8;
      } else if (strcmp(value, "zero_split") == 0) {
        *mode = 9;
      } else if (strcmp(value, "correction_full") == 0) {
        *mode = 10;
      } else if (strcmp(value, "cluster_scan_split") == 0) {
        *mode = 11;
      } else if (strcmp(value, "cluster_scan_both") == 0) {
        *mode = 12;
      } else if (strcmp(value, "zero_v_correction_full") == 0) {
        *mode = 13;
      } else if (strcmp(value, "stream_segments") == 0) {
        *mode = 14;
      } else if (strcmp(value, "stream_segments_post_serial") == 0) {
        *mode = 15;
      } else if (strcmp(value, "stream_segments_post_overlap") == 0) {
        *mode = 16;
      } else if (strcmp(value, "stream_segments_rms_gate_serial") == 0) {
        *mode = 17;
      } else if (strcmp(value, "stream_segments_rms_gate_overlap") == 0) {
        *mode = 18;
      } else {
        fprintf(stderr, "unsupported --mode=%s; use checkpoint, split, both, state_checkpoint, state_split, state_both, scan_transition, scan_split, scan_both, zero_split, correction_full, cluster_scan_split, cluster_scan_both, zero_v_correction_full, stream_segments, stream_segments_post_serial, stream_segments_post_overlap, stream_segments_rms_gate_serial, or stream_segments_rms_gate_overlap\n", value);
        exit(1);
      }
      *mode_name = value;
      continue;
    }
    argv[out++] = argv[i];
  }
  return out;
}

static int strip_check_arg(int argc, char** argv, bool* check) {
  int out = 0;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "--check") == 0) {
      *check = true;
      continue;
    }
    argv[out++] = argv[i];
  }
  return out;
}

static void usage(char const* argv0) {
  printf("Usage: %s [seqlen] [q_heads] [v_heads] [head_dim] "
         "[--segment-tokens N] [--post-rounds N] [--mode checkpoint|split|both|state_checkpoint|state_split|state_both|scan_transition|scan_split|scan_both|zero_split|correction_full|cluster_scan_split|cluster_scan_both|zero_v_correction_full|stream_segments|stream_segments_post_serial|stream_segments_post_overlap|stream_segments_rms_gate_serial|stream_segments_rms_gate_overlap] [--check] [--bench W I]\n",
         argv0);
  printf("Default shape: 3823 16 64 128, segment_tokens=1024, mode=split\n");
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    }
  }

  BenchTimer timer;
  timer.parse(argc, argv);
  argc = BenchTimer::strip_bench_args(argc, argv);

  int segment_tokens = 1024;
  argc = strip_int_arg(argc, argv, "--segment-tokens", &segment_tokens);
  int post_rounds = 64;
  argc = strip_int_arg(argc, argv, "--post-rounds", &post_rounds);
  int mode = 1;
  char const* mode_name = "split";
  argc = strip_mode_arg(argc, argv, &mode, &mode_name);
  bool check = false;
  argc = strip_check_arg(argc, argv, &check);

  int total_seqlen = (argc > 1) ? atoi(argv[1]) : 3823;
  int num_q_heads = (argc > 2) ? atoi(argv[2]) : 16;
  int num_v_heads = (argc > 3) ? atoi(argv[3]) : 64;
  int head_dim = (argc > 4) ? atoi(argv[4]) : 128;
  int num_k_heads = num_q_heads;
  int num_o_heads = std::max(num_q_heads, num_v_heads);
  int num_sab_heads = num_o_heads;
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  if (head_dim != 128) {
    fprintf(stderr, "this study only instantiates head_dim=128\n");
    return 1;
  }
  if (num_v_heads <= num_q_heads) {
    fprintf(stderr, "this study only instantiates the GVA path: v_heads must be > q_heads\n");
    return 1;
  }
  if (segment_tokens <= 0 || (segment_tokens % 64) != 0) {
    fprintf(stderr, "--segment-tokens must be a positive multiple of 64\n");
    return 1;
  }

  int segments = (total_seqlen + segment_tokens - 1) / segment_tokens;
  if (segments < 2) {
    fprintf(stderr, "segment_tokens=%d creates only one segment; choose a smaller segment size\n",
            segment_tokens);
    return 1;
  }
  int checkpoint_count = total_seqlen / segment_tokens;
  int num_blocks = (total_seqlen + 63) / 64;
  int checkpoint_block_interval = segment_tokens / 64;
  int checkpoint_storage_count = num_blocks / checkpoint_block_interval;
  checkpoint_storage_count = std::max(checkpoint_storage_count, checkpoint_count);
  int needed_input_states = segments;
  if (checkpoint_count < segments - 1) {
    fprintf(stderr, "internal error: not enough checkpoints for segment inputs\n");
    return 1;
  }

  printf("bench gdn split-seq study: seqlen=%d q_heads=%d k_heads=%d v_heads=%d dim=%d "
         "segment_tokens=%d segments=%d checkpoint_count=%d checkpoint_storage=%d mode=%s\n",
         total_seqlen, num_q_heads, num_k_heads, num_v_heads, head_dim, segment_tokens,
         segments, checkpoint_count, checkpoint_storage_count, mode_name);

  using T = nv_bfloat16;

  int64_t q_size = static_cast<int64_t>(total_seqlen) * num_q_heads * head_dim;
  int64_t k_size = static_cast<int64_t>(total_seqlen) * num_k_heads * head_dim;
  int64_t v_size = static_cast<int64_t>(total_seqlen) * num_v_heads * head_dim;
  int64_t o_size = static_cast<int64_t>(total_seqlen) * num_o_heads * head_dim;
  int64_t gate_size = static_cast<int64_t>(total_seqlen) * num_sab_heads;
  int64_t state_elems = static_cast<int64_t>(num_sab_heads) * head_dim * head_dim;
  int64_t state_elems_per_head = static_cast<int64_t>(head_dim) * head_dim;
  int64_t split_state_size = static_cast<int64_t>(needed_input_states) * state_elems;
  int64_t checkpoint_size = static_cast<int64_t>(checkpoint_storage_count) * state_elems;

  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_v_zero = nullptr;
  T *d_o = nullptr, *d_o_split = nullptr, *d_gate_z = nullptr, *d_gate_weight = nullptr;
  T* d_gate_out = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_full_final_state = nullptr, *d_split_input_state = nullptr;
  float *d_split_output_state = nullptr, *d_checkpoints = nullptr;
  float *d_segment_state = nullptr, *d_segment_coeffs = nullptr;
  float *d_pipeline_states = nullptr, *d_post_scratch = nullptr;
  int64_t *d_cu_full = nullptr, *d_cu_split = nullptr, *d_ckpt_cu = nullptr;
  uint8_t* d_workspace = nullptr;

  BENCH_CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_v_zero, v_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_o_split, o_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_gate_z, o_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_gate_weight, head_dim * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_gate_out, o_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_alpha, gate_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_beta, gate_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_full_final_state, state_elems * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_split_input_state, split_state_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_split_output_state, split_state_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_segment_state, split_state_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_segment_coeffs,
                              static_cast<int64_t>(segments) * num_sab_heads * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_pipeline_states,
                              static_cast<int64_t>(segments + 1) * state_elems * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_post_scratch, o_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_checkpoints, checkpoint_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_cu_full, 2 * sizeof(int64_t)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_cu_split, (segments + 1) * sizeof(int64_t)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_ckpt_cu, 2 * sizeof(int64_t)));
  size_t ws_size = 128 * 1024 * 1024;
  BENCH_CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

  srand(42);
  auto fill_bf16 = [](T* d, int64_t n) {
    std::vector<T> h(n);
    for (auto& x : h) {
      x = __float2bfloat16((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    }
    BENCH_CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  };
  fill_bf16(d_q, q_size);
  fill_bf16(d_k, k_size);
  fill_bf16(d_v, v_size);
  fill_bf16(d_gate_z, o_size);
  fill_bf16(d_gate_weight, head_dim);
  BENCH_CUDA_CHECK(cudaMemset(d_v_zero, 0, v_size * sizeof(T)));

  std::vector<float> h_gate(gate_size);
  for (auto& x : h_gate) {
    x = 0.5f + 0.3f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
  }
  BENCH_CUDA_CHECK(cudaMemcpy(d_alpha, h_gate.data(), gate_size * sizeof(float), cudaMemcpyHostToDevice));
  for (auto& x : h_gate) {
    x = 0.5f + 0.3f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
  }
  BENCH_CUDA_CHECK(cudaMemcpy(d_beta, h_gate.data(), gate_size * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<int64_t> h_full = {0, total_seqlen};
  BENCH_CUDA_CHECK(cudaMemcpy(d_cu_full, h_full.data(), h_full.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice));
  std::vector<int64_t> h_split(segments + 1);
  for (int i = 0; i <= segments; ++i) {
    h_split[i] = std::min<int64_t>(static_cast<int64_t>(i) * segment_tokens, total_seqlen);
  }
  BENCH_CUDA_CHECK(cudaMemcpy(d_cu_split, h_split.data(), h_split.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice));
  std::vector<int64_t> h_ckpt_cu = {0, checkpoint_storage_count};
  BENCH_CUDA_CHECK(cudaMemcpy(d_ckpt_cu, h_ckpt_cu.data(), h_ckpt_cu.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice));

  BENCH_CUDA_CHECK(cudaMemset(d_o, 0, o_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMemset(d_o_split, 0, o_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMemset(d_gate_out, 0, o_size * sizeof(T)));
  BENCH_CUDA_CHECK(cudaMemset(d_full_final_state, 0, state_elems * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMemset(d_split_input_state, 0, split_state_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMemset(d_split_output_state, 0, split_state_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMemset(d_segment_state, 0, split_state_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMemset(d_segment_coeffs, 0,
                              static_cast<int64_t>(segments) * num_sab_heads * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMemset(d_pipeline_states, 0,
                              static_cast<int64_t>(segments + 1) * state_elems * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMemset(d_post_scratch, 0, o_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMemset(d_checkpoints, 0, checkpoint_size * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMemset(d_workspace, 0, ws_size));

  int sm_count = 0;
  BENCH_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

  auto run_checkpoint = [&]() {
    gdn_splitseq_study::launch_gdn_prefill_bf16_gva_checkpoint(
        0, d_o, d_full_final_state, d_q, d_k, d_v, d_alpha, d_beta, d_cu_full, d_workspace,
        1, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_dim, total_seqlen, scale,
        sm_count, d_checkpoints, d_ckpt_cu, segment_tokens);
  };

  auto run_zero_split = [&]() {
    gdn_splitseq_study::launch_gdn_prefill_bf16_gva_zero_state(
        0, d_o_split, d_split_output_state, d_q, d_k, d_v, d_alpha, d_beta, d_cu_split,
        d_workspace, segments, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_dim,
        total_seqlen, scale, sm_count);
  };

  auto run_state_checkpoint = [&]() {
    gdn_state_only_study::launch_gdn_prefill_bf16_gva_state_only_checkpoint(
        0, d_o, d_full_final_state, d_q, d_k, d_v, d_alpha, d_beta, d_cu_full, d_workspace,
        1, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_dim, total_seqlen, scale,
        sm_count, d_checkpoints, d_ckpt_cu, segment_tokens);
  };

  auto run_transition = [&]() {
    gdn_state_only_study::launch_gdn_prefill_bf16_gva_state_only_transition(
        0, d_o, d_segment_state, d_q, d_k, d_v, d_alpha, d_beta, d_cu_split, d_workspace,
        segments, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_dim, total_seqlen,
        scale, sm_count);
  };

  auto pack_states = [&]() {
    int threads = 256;
    int blocks = static_cast<int>(
        std::min<int64_t>(4096, (split_state_size + threads - 1) / threads));
    pack_segment_input_states<<<blocks, threads>>>(d_split_input_state, d_checkpoints, segments,
                                                   state_elems);
  };

  auto compose_states = [&]() {
    int coeff_threads = 256;
    int coeff_blocks = segments * num_sab_heads;
    compute_segment_coeffs<<<coeff_blocks, coeff_threads>>>(d_segment_coeffs, d_alpha, d_cu_split,
                                                            segments, num_sab_heads);
    int threads = 256;
    int blocks = static_cast<int>(
        std::min<int64_t>(4096, (split_state_size + threads - 1) / threads));
    compose_segment_input_states<<<blocks, threads>>>(d_split_input_state, d_segment_state,
                                                      d_segment_coeffs, segments, num_sab_heads,
                                                      state_elems_per_head);
  };

  auto compose_states_cluster = [&]() {
#if CUDART_VERSION >= 12000
    int threads = 256;
    cudaError_t attr_err = cudaFuncSetAttribute(
        compose_segment_input_states_cluster,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (attr_err != cudaSuccess) {
      cudaGetLastError();
    }

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(segments * num_sab_heads, 1, 1);
    config.blockDim = dim3(threads, 1, 1);
    config.dynamicSmemBytes = 0;

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = segments;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;

    BENCH_CUDA_CHECK(cudaLaunchKernelEx(
        &config, compose_segment_input_states_cluster, d_split_input_state, d_segment_state,
        d_segment_coeffs, d_alpha, d_cu_split, segments, num_sab_heads,
        state_elems_per_head));
#else
    fprintf(stderr, "cluster_scan modes require CUDA runtime >= 12.0\n");
    exit(1);
#endif
  };

  auto run_split = [&]() {
    gdn_splitseq_study::launch_gdn_prefill_bf16_gva_initstate(
        0, d_o_split, d_split_output_state, d_q, d_k, d_v, d_split_input_state, d_alpha, d_beta,
        d_cu_split, d_workspace, segments, num_q_heads, num_k_heads, num_v_heads, num_o_heads,
        head_dim, total_seqlen, scale, sm_count);
  };

  auto run_correction_full_to = [&](T* output, float* output_state) {
    gdn_splitseq_study::launch_gdn_prefill_bf16_gva_initstate(
        0, output, output_state, d_q, d_k, d_v_zero, d_split_input_state, d_alpha,
        d_beta, d_cu_split, d_workspace, segments, num_q_heads, num_k_heads, num_v_heads,
        num_o_heads, head_dim, total_seqlen, scale, sm_count);
  };

  auto run_correction_full = [&]() {
    run_correction_full_to(d_o_split, d_split_output_state);
  };

  auto run_zero_v_correction_full_to = [&](T* output, float* output_state) {
    gdn_zero_v_study::launch_gdn_prefill_bf16_gva_zero_v_initstate(
        0, output, output_state, d_q, d_k, d_v_zero, d_split_input_state, d_alpha, d_beta,
        d_cu_split, d_workspace, segments, num_q_heads, num_k_heads, num_v_heads, num_o_heads,
        head_dim, total_seqlen, scale, sm_count);
  };

  auto run_zero_v_correction_full = [&]() {
    run_zero_v_correction_full_to(d_o_split, d_split_output_state);
  };

  auto run_stream_segments = [&](int post_kind, bool overlap_post) {
    cudaStream_t gdn_stream = nullptr;
    cudaStream_t post_stream = nullptr;
    BENCH_CUDA_CHECK(cudaStreamCreateWithFlags(&gdn_stream, cudaStreamNonBlocking));
    BENCH_CUDA_CHECK(cudaStreamCreateWithFlags(&post_stream, cudaStreamNonBlocking));
    std::vector<cudaEvent_t> segment_done(segments);
    for (int i = 0; i < segments; ++i) {
      BENCH_CUDA_CHECK(cudaEventCreateWithFlags(&segment_done[i], cudaEventDisableTiming));
    }

    BENCH_CUDA_CHECK(cudaMemsetAsync(d_pipeline_states, 0, state_elems * sizeof(float),
                                     gdn_stream));
    for (int i = 0; i < segments; ++i) {
      int64_t seg_begin = h_split[i];
      int64_t seg_end = h_split[i + 1];
      int64_t seg_tokens = seg_end - seg_begin;
      gdn_splitseq_study::launch_gdn_prefill_bf16_gva_initstate(
          gdn_stream, d_o_split, d_pipeline_states + static_cast<int64_t>(i + 1) * state_elems,
          d_q, d_k, d_v, d_pipeline_states + static_cast<int64_t>(i) * state_elems, d_alpha,
          d_beta, d_cu_split + i, d_workspace, 1, num_q_heads, num_k_heads, num_v_heads,
          num_o_heads, head_dim, total_seqlen, scale, sm_count);

      if (post_kind != 0) {
        BENCH_CUDA_CHECK(cudaEventRecord(segment_done[i], gdn_stream));
        cudaStream_t consumer_stream = overlap_post ? post_stream : gdn_stream;
        BENCH_CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, segment_done[i], 0));
        int64_t elems = seg_tokens * static_cast<int64_t>(num_o_heads) * head_dim;
        int64_t offset = seg_begin * static_cast<int64_t>(num_o_heads) * head_dim;
        if (post_kind == 1) {
          int threads = 256;
          int blocks =
              static_cast<int>(std::min<int64_t>(4096, (elems + threads - 1) / threads));
          post_consumer_bf16_kernel<<<blocks, threads, 0, consumer_stream>>>(
              d_o_split + offset, d_post_scratch + offset, elems, post_rounds);
        } else {
          int rows = static_cast<int>(seg_tokens * static_cast<int64_t>(num_o_heads));
          fused_rms_norm_gate_segment_kernel<<<rows, 256, 0, consumer_stream>>>(
              d_o_split + offset, d_gate_z + offset, d_gate_weight, d_gate_out + offset, rows,
              head_dim, 1.0e-6f);
        }
      }
    }

    BENCH_CUDA_CHECK(cudaStreamSynchronize(gdn_stream));
    BENCH_CUDA_CHECK(cudaStreamSynchronize(post_stream));
    for (int i = 0; i < segments; ++i) {
      BENCH_CUDA_CHECK(cudaEventDestroy(segment_done[i]));
    }
    BENCH_CUDA_CHECK(cudaStreamDestroy(post_stream));
    BENCH_CUDA_CHECK(cudaStreamDestroy(gdn_stream));
  };

  try {
    if (mode == 0) {
      timer.run(run_checkpoint);
    } else if (mode == 1) {
      run_checkpoint();
      pack_states();
      BENCH_CUDA_CHECK(cudaGetLastError());
      BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      timer.run(run_split);
    } else if (mode == 2) {
      timer.run([&]() {
        run_checkpoint();
        pack_states();
        run_split();
      });
    } else if (mode == 3) {
      timer.run(run_state_checkpoint);
    } else if (mode == 4) {
      run_state_checkpoint();
      pack_states();
      BENCH_CUDA_CHECK(cudaGetLastError());
      BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      timer.run(run_split);
    } else if (mode == 5) {
      timer.run([&]() {
        run_state_checkpoint();
        pack_states();
        run_split();
      });
    } else if (mode == 6) {
      timer.run(run_transition);
    } else if (mode == 7) {
      run_transition();
      compose_states();
      BENCH_CUDA_CHECK(cudaGetLastError());
      BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      timer.run(run_split);
    } else if (mode == 8) {
      timer.run([&]() {
        run_transition();
        compose_states();
        run_split();
      });
    } else if (mode == 9) {
      timer.run(run_zero_split);
    } else if (mode == 10) {
      run_transition();
      compose_states();
      BENCH_CUDA_CHECK(cudaGetLastError());
      BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      timer.run(run_correction_full);
    } else if (mode == 11) {
      run_transition();
      compose_states_cluster();
      BENCH_CUDA_CHECK(cudaGetLastError());
      BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      timer.run(run_split);
    } else if (mode == 12) {
      timer.run([&]() {
        run_transition();
        compose_states_cluster();
        run_split();
      });
    } else if (mode == 13) {
      run_transition();
      compose_states();
      BENCH_CUDA_CHECK(cudaGetLastError());
      BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      timer.run(run_zero_v_correction_full);
    } else if (mode == 14) {
      timer.run([&]() { run_stream_segments(0, false); });
    } else if (mode == 15) {
      timer.run([&]() { run_stream_segments(1, false); });
    } else if (mode == 16) {
      timer.run([&]() { run_stream_segments(1, true); });
    } else if (mode == 17) {
      timer.run([&]() { run_stream_segments(2, false); });
    } else {
      timer.run([&]() { run_stream_segments(2, true); });
    }
  } catch (std::exception const& e) {
    fprintf(stderr, "launch failed: %s\n", e.what());
    return 1;
  }
  BENCH_CUDA_CHECK(cudaGetLastError());

  if (check) {
    if (mode == 0 || mode == 3 || mode == 6) {
      printf("check skipped: checkpoint-only modes do not produce split output\n");
    } else {
      BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      if (mode == 13) {
        run_correction_full_to(d_o, d_full_final_state);
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      } else if (mode == 4 || mode == 5 || mode == 7 || mode == 8 || mode == 11 ||
                 mode == 12 || mode == 14 || mode == 15 || mode == 16 || mode == 17 ||
                 mode == 18) {
        run_checkpoint();
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
      }
      std::vector<T> h_full(o_size);
      std::vector<T> h_split_out(o_size);
      BENCH_CUDA_CHECK(cudaMemcpy(h_full.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
      BENCH_CUDA_CHECK(
          cudaMemcpy(h_split_out.data(), d_o_split, o_size * sizeof(T), cudaMemcpyDeviceToHost));
      float max_abs = 0.0f;
      float max_rel = 0.0f;
      for (int64_t i = 0; i < o_size; ++i) {
        float a = __bfloat162float(h_full[i]);
        float b = __bfloat162float(h_split_out[i]);
        float diff = fabsf(a - b);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / std::max(1.0e-6f, fabsf(a)));
      }
      printf("check: max_abs=%.6g max_rel=%.6g elements=%lld\n", max_abs, max_rel,
             static_cast<long long>(o_size));
    }
  }

  printf("Done.\n");

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_v_zero);
  cudaFree(d_o);
  cudaFree(d_o_split);
  cudaFree(d_gate_z);
  cudaFree(d_gate_weight);
  cudaFree(d_gate_out);
  cudaFree(d_alpha);
  cudaFree(d_beta);
  cudaFree(d_full_final_state);
  cudaFree(d_split_input_state);
  cudaFree(d_split_output_state);
  cudaFree(d_segment_state);
  cudaFree(d_segment_coeffs);
  cudaFree(d_pipeline_states);
  cudaFree(d_post_scratch);
  cudaFree(d_checkpoints);
  cudaFree(d_cu_full);
  cudaFree(d_cu_split);
  cudaFree(d_ckpt_cu);
  cudaFree(d_workspace);
  return 0;
}
