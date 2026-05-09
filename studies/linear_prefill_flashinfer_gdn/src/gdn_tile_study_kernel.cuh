#pragma once

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/device_memory.h"
#include "flashinfer/flat/common.hpp"
#include "flashinfer/flat/hopper/device/device_universal.hpp"
#include "flashinfer/flat/hopper/kernel/flat_kernel_builder_delta_rule.hpp"

namespace gdn_tile_study {

using namespace cute;

template <int TileTokens, int StagesQCount, int StagesKCount, int StagesVCount>
void launch_gdn_prefill_bf16_gva_tile(
    cudaStream_t stream, nv_bfloat16* output, float* output_state, nv_bfloat16 const* q,
    nv_bfloat16 const* k, nv_bfloat16 const* v, float const* alpha, float const* beta,
    int64_t const* cu_seqlens, uint8_t* workspace_buffer, int32_t num_seqs,
    int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads, int32_t num_o_heads,
    int32_t head_size, int64_t total_seqlen, float scale, int32_t sm_count) {
#if defined(FLAT_SM90A_ENABLED)
  static_assert(TileTokens == 64 || TileTokens == 128, "FlashInfer GDN supports 64 or 128 token tiles");

  using namespace flat::kernel;
  using T = flat::map_to_cutlass_t<nv_bfloat16>;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = sm_count;

  using Options = std::tuple<
      Option<Tag::kIsDeltaRule, cute::true_type>,
      Option<Tag::kIsGVA, cute::true_type>,
      Option<Tag::kNeedsBeta, cute::true_type>,
      Option<Tag::kNeedsAlpha, cute::true_type>,
      Option<Tag::kInitStateFromInput, cute::false_type>,
      Option<Tag::kEnableCheckpointing, cute::false_type>,
      Option<Tag::kStagesQ, Int<StagesQCount>>,
      Option<Tag::kStagesK, Int<StagesKCount>>,
      Option<Tag::kStagesV, Int<StagesVCount>>>;

  using TileShape = Shape<Int<TileTokens>, Int<TileTokens>, _128>;
  using Scheduler = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using Operation = cutlass::device::Universal<typename flat::kernel::FlatBuilderDeltaRule<
      T, float, float, TileShape,
      /*LayoutQ=*/cute::tuple<int64_t, _1, int32_t>,
      /*LayoutK=*/cute::tuple<int64_t, _1, int32_t>,
      /*LayoutV=*/cute::tuple<int64_t, _1, int32_t>,
      /*LayoutO=*/cute::tuple<int64_t, _1, int32_t>, Scheduler, Options>::Kernel>;
  using Arguments = typename Operation::Arguments;

  int32_t num_sab_heads = std::max(num_q_heads, num_v_heads);

  int32_t q_tok_stride = num_q_heads * head_size;
  int32_t o_tok_stride = num_o_heads * head_size;
  int32_t k_tok_stride = num_k_heads * head_size;
  int32_t v_tok_stride = num_v_heads * head_size;

  int32_t q_head_stride = head_size;
  int32_t o_head_stride = head_size;
  int32_t k_head_stride = head_size;
  int32_t v_head_stride = head_size;

  Operation op;
  Arguments arguments{.problem_size =
                          {
                              .cu_seqlens = cu_seqlens,
                              .total_seqlen = total_seqlen,
                              .num_seqs = num_seqs,
                              .num_q_heads = num_q_heads,
                              .num_k_heads = num_k_heads,
                              .num_v_heads = num_v_heads,
                              .num_o_heads = num_o_heads,
                              .num_sab_heads = num_sab_heads,
                              .head_size = head_size,
                          },
                      .mainloop =
                          {
                              // clang-format off
              .ptr_Q = (T*)q,      .dQ = {q_tok_stride, _1{}, q_head_stride},
              .ptr_K = (T*)k,      .dK = {k_tok_stride, _1{}, k_head_stride},
              .ptr_V = (T*)v,      .dV = {v_tok_stride, _1{}, v_head_stride},
              .ptr_O = (T*)output, .dO = {o_tok_stride, _1{}, o_head_stride},
              .ptr_output_state = (float*)output_state,
              .ptr_input_state  = nullptr,
              .scale = scale,
              .alpha_ptr = alpha, .alpha_stride = {num_sab_heads, 1},
              .beta_ptr  = beta,  .beta_stride  = {num_sab_heads, 1},
              .ptr_state_checkpoints = nullptr,
              .checkpoint_cu_starts = nullptr,
              .checkpoint_every_n_tokens = 0,
      },  // clang-format on
                      .hw_info = hw_info};

  cutlass::Status status = op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("can_implement failed");
  }

  status = op.initialize(arguments, workspace_buffer, stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("initialize failed");
  }

  status = op.run(stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("run failed");
  }
#else
  (void)stream;
  (void)output;
  (void)output_state;
  (void)q;
  (void)k;
  (void)v;
  (void)alpha;
  (void)beta;
  (void)cu_seqlens;
  (void)workspace_buffer;
  (void)num_seqs;
  (void)num_q_heads;
  (void)num_k_heads;
  (void)num_v_heads;
  (void)num_o_heads;
  (void)head_size;
  (void)total_seqlen;
  (void)scale;
  (void)sm_count;
  throw std::runtime_error("FLAT_SM90A_ENABLED is required");
#endif
}

}  // namespace gdn_tile_study
