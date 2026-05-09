// Isolated FlashInfer GDN prefill block-DV study.
// Usage:
//   ./bench_gdn_blockdv_study [seqlen] [num_q_heads] [num_v_heads] [head_dim] [num_seqs] --tile 64 --block-dv 64 [--bench W I]

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bench_timer.h"

namespace gdn_blockdv_study {
void launch_gdn_prefill_bf16_gva_blockdv(
    int tile_tokens, int block_dv, int variant_id, cudaStream_t stream, nv_bfloat16* output, float* output_state,
    nv_bfloat16 const* q, nv_bfloat16 const* k, nv_bfloat16 const* v, float const* alpha,
    float const* beta, int64_t const* cu_seqlens, uint8_t* workspace, int32_t num_seqs,
    int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads, int32_t num_o_heads,
    int32_t head_size, int64_t total_seqlen, float scale, int32_t sm_count);
}  // namespace gdn_blockdv_study

#define CHECK(e)                                                                                    \
  do {                                                                                              \
    cudaError_t _e = (e);                                                                           \
    if (_e != cudaSuccess) {                                                                        \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));             \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)

static int strip_tile_arg(int argc, char** argv, int* tile_tokens) {
  int out = 0;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "--tile") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "--tile requires a value: 64 or 128\n");
        exit(1);
      }
      *tile_tokens = atoi(argv[++i]);
      continue;
    }
    if (strncmp(argv[i], "--tile=", 7) == 0) {
      *tile_tokens = atoi(argv[i] + 7);
      continue;
    }
    argv[out++] = argv[i];
  }
  return out;
}

static int strip_block_dv_arg(int argc, char** argv, int* block_dv) {
  int out = 0;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "--block-dv") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "--block-dv requires a value: 64\n");
        exit(1);
      }
      *block_dv = atoi(argv[++i]);
      continue;
    }
    if (strncmp(argv[i], "--block-dv=", 11) == 0) {
      *block_dv = atoi(argv[i] + 11);
      continue;
    }
    argv[out++] = argv[i];
  }
  return out;
}

static int strip_variant_arg(int argc, char** argv, int* variant_id, char const** variant_name) {
  int out = 0;
  for (int i = 0; i < argc; ++i) {
    char const* value = nullptr;
    if (strcmp(argv[i], "--variant") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "--variant requires a value: default, k2, q3, or v3\n");
        exit(1);
      }
      value = argv[++i];
    } else if (strncmp(argv[i], "--variant=", 10) == 0) {
      value = argv[i] + 10;
    }
    if (value != nullptr) {
      if (strcmp(value, "default") == 0) {
        *variant_id = 0;
      } else if (strcmp(value, "k2") == 0) {
        *variant_id = 1;
      } else if (strcmp(value, "q3") == 0) {
        *variant_id = 2;
      } else if (strcmp(value, "v3") == 0) {
        *variant_id = 3;
      } else {
        fprintf(stderr, "unsupported --variant=%s; use default, k2, q3, or v3\n", value);
        exit(1);
      }
      *variant_name = value;
      continue;
    }
    argv[out++] = argv[i];
  }
  return out;
}

static void usage(char const* argv0) {
  printf("Usage: %s [seqlen] [num_q_heads] [num_v_heads] [head_dim] [num_seqs] --tile 64 --block-dv 64 [--variant default|k2|q3|v3] [--bench W I]\n",
         argv0);
  printf("Default shape: 3823 16 64 128 1\n");
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

  int tile_tokens = 64;
  argc = strip_tile_arg(argc, argv, &tile_tokens);
  int block_dv = 64;
  argc = strip_block_dv_arg(argc, argv, &block_dv);
  int variant_id = 0;
  char const* variant_name = "default";
  argc = strip_variant_arg(argc, argv, &variant_id, &variant_name);

  int total_seqlen = (argc > 1) ? atoi(argv[1]) : 3823;
  int num_q_heads = (argc > 2) ? atoi(argv[2]) : 16;
  int num_v_heads = (argc > 3) ? atoi(argv[3]) : 64;
  int head_dim = (argc > 4) ? atoi(argv[4]) : 128;
  int num_seqs = (argc > 5) ? atoi(argv[5]) : 1;
  int num_k_heads = num_q_heads;
  int num_o_heads = std::max(num_q_heads, num_v_heads);
  int num_sab_heads = num_o_heads;
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  if (tile_tokens != 64 || block_dv != 64) {
    fprintf(stderr, "unsupported --tile=%d --block-dv=%d; this study compiles 64x64x128 with block_dv=64\n", tile_tokens, block_dv);
    return 1;
  }
  if (head_dim != 128) {
    fprintf(stderr, "this study only instantiates head_dim=128\n");
    return 1;
  }
  if (num_v_heads <= num_q_heads) {
    fprintf(stderr, "this study only instantiates the GVA path: num_v_heads must be > num_q_heads\n");
    return 1;
  }

  printf("bench gdn_prefill block-DV study: seqlen=%d seqs=%d q_heads=%d k_heads=%d v_heads=%d dim=%d tile=%d block_dv=%d variant=%s\n",
         total_seqlen, num_seqs, num_q_heads, num_k_heads, num_v_heads, head_dim, tile_tokens,
         block_dv, variant_name);

  using T = nv_bfloat16;

  int64_t q_size = static_cast<int64_t>(total_seqlen) * num_q_heads * head_dim;
  int64_t k_size = static_cast<int64_t>(total_seqlen) * num_k_heads * head_dim;
  int64_t v_size = static_cast<int64_t>(total_seqlen) * num_v_heads * head_dim;
  int64_t o_size = static_cast<int64_t>(total_seqlen) * num_o_heads * head_dim;
  int64_t gate_size = static_cast<int64_t>(total_seqlen) * num_sab_heads;
  int64_t state_size = static_cast<int64_t>(num_seqs) * num_sab_heads * head_dim * head_dim;

  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr, *d_state = nullptr;
  int64_t* d_cu_seqlens = nullptr;
  uint8_t* d_workspace = nullptr;

  CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
  CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
  CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
  CHECK(cudaMalloc(&d_alpha, gate_size * sizeof(float)));
  CHECK(cudaMalloc(&d_beta, gate_size * sizeof(float)));
  CHECK(cudaMalloc(&d_state, state_size * sizeof(float)));
  CHECK(cudaMalloc(&d_cu_seqlens, (num_seqs + 1) * sizeof(int64_t)));
  size_t ws_size = 128 * 1024 * 1024;
  CHECK(cudaMalloc(&d_workspace, ws_size));

  srand(42);
  auto fill_bf16 = [](T* d, int64_t n) {
    std::vector<T> h(n);
    for (auto& x : h) {
      x = __float2bfloat16((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    }
    CHECK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  };
  fill_bf16(d_q, q_size);
  fill_bf16(d_k, k_size);
  fill_bf16(d_v, v_size);

  std::vector<float> h_gate(gate_size);
  for (auto& x : h_gate) {
    x = 0.5f + 0.3f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
  }
  CHECK(cudaMemcpy(d_alpha, h_gate.data(), gate_size * sizeof(float), cudaMemcpyHostToDevice));
  for (auto& x : h_gate) {
    x = 0.5f + 0.3f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
  }
  CHECK(cudaMemcpy(d_beta, h_gate.data(), gate_size * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<int64_t> h_cu(num_seqs + 1);
  for (int i = 0; i <= num_seqs; ++i) {
    h_cu[i] = static_cast<int64_t>(total_seqlen) * i / num_seqs;
  }
  CHECK(cudaMemcpy(d_cu_seqlens, h_cu.data(), h_cu.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

  CHECK(cudaMemset(d_o, 0, o_size * sizeof(T)));
  CHECK(cudaMemset(d_state, 0, state_size * sizeof(float)));
  CHECK(cudaMemset(d_workspace, 0, ws_size));

  int sm_count = 0;
  CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

  try {
    timer.run([&]() {
      gdn_blockdv_study::launch_gdn_prefill_bf16_gva_blockdv(
          tile_tokens, block_dv, variant_id, 0, d_o, d_state, d_q, d_k, d_v, d_alpha, d_beta, d_cu_seqlens,
          d_workspace, num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_dim,
          total_seqlen, scale, sm_count);
    });
  } catch (std::exception const& e) {
    fprintf(stderr, "launch failed: %s\n", e.what());
    return 1;
  }
  CHECK(cudaGetLastError());

  printf("Done.\n");

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
  cudaFree(d_alpha);
  cudaFree(d_beta);
  cudaFree(d_state);
  cudaFree(d_cu_seqlens);
  cudaFree(d_workspace);
  return 0;
}
