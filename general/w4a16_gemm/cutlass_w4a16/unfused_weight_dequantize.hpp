/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cute/tensor.hpp"
#include "xplane_offline.hpp"

#include <hggc.h>
#include "helper.h"

template <class QuantizedElement, 
          class DequantizedElement,
          class OperandLayout,
          class ElementScale,
          class ElementZero,
          class ScaleBroadCastLayout,
          class ThrLayout>
__global__ void dequantize_weight_kernel(DequantizedElement* dq_buffer,
                                         QuantizedElement const* q_buffer,
                                         OperandLayout const operand_layout,
                                         ElementScale const* scale_buffer,
                                         ElementZero const* zero_buffer,
                                         ScaleBroadCastLayout const broadcasted_scale_layout,
                                         ThrLayout thr_layout) {
  using namespace cute;

  // Represent the full tensors to gmem elements. 
  // These are expected to have shape [MN, K, L]
  Tensor gmem_op_dq = make_tensor(make_gmem_ptr(dq_buffer), operand_layout);
  auto init_quantized_iterator = [&]() {
    if constexpr (cute::sizeof_bits_v<QuantizedElement> >= 8) {
      return make_gmem_ptr(q_buffer);
    } else {
      return subbyte_iterator<const QuantizedElement>(q_buffer);
    }
  };
  Tensor gmem_op_q  = make_tensor(init_quantized_iterator(), operand_layout);
  // While the scales are expected to have shape [MN, G, L] but with a stride to allow broadcasting
  // It is expected that K % G == 0
  Tensor gmem_scale_broadcasted = make_tensor(make_gmem_ptr(scale_buffer), broadcasted_scale_layout);
  Tensor gmem_zero_broadcasted = make_tensor(make_gmem_ptr(zero_buffer), broadcasted_scale_layout);

  // Assign 1 thread per element in the thread block
  auto blk_shape = make_shape(size<0>(thr_layout), _1{}, _1{}); // 
  auto blk_coord = make_coord(_, blockIdx.x, blockIdx.y);  // (MN, K, L)

  // Tile across the block
  auto gOp_dq = local_tile(gmem_op_dq, blk_shape, blk_coord);
  auto gScale = local_tile(gmem_scale_broadcasted, blk_shape, blk_coord);
  auto gZero  = local_tile(gmem_zero_broadcasted,  blk_shape, blk_coord);
  auto gOp_q  = local_tile(gmem_op_q, blk_shape, blk_coord);
  
  auto tOpDq_gOpDq = local_partition(gOp_dq, thr_layout, threadIdx.x);
  auto tScale_gScale = local_partition(gScale, thr_layout, threadIdx.x);
  auto tZero_gZero = local_partition(gZero, thr_layout, threadIdx.x);
  auto tOpQ_gOpQ = local_partition(gOp_q, thr_layout, threadIdx.x);

  // Make a fragment of registers to hold gmem loads
  Tensor rmem_op_q = make_fragment_like(tOpQ_gOpQ(_, _, _, 0));
  Tensor rmem_scale = make_fragment_like(tScale_gScale(_, _, _, 0));
  Tensor rmem_zero = make_fragment_like(tZero_gZero(_, _, _, 0));
  Tensor rmem_op_dq = make_fragment_like(tOpDq_gOpDq(_, _, _, 0));
  Tensor rmem_op_scaled = make_fragment_like<ElementScale>(rmem_op_dq);
  Tensor rmem_zero_buf = make_fragment_like<ElementScale>(rmem_zero);

  Tensor pred_id = make_identity_tensor(shape(operand_layout));
  auto pred_blk_tile = local_tile(pred_id, blk_shape, blk_coord);
  auto pred_thr_partition = local_partition(pred_blk_tile, thr_layout, threadIdx.x);

  const auto num_iters = size<3>(tOpDq_gOpDq);
  
  for (int ii = 0; ii < num_iters; ++ii) {
    const auto thread_offset = get<0>(pred_thr_partition(0, 0, 0, ii));
    if (thread_offset < size<0>(operand_layout)) {
      copy(tOpQ_gOpQ(_, _, _, ii), rmem_op_q);
      copy(tScale_gScale(_, _, _, ii), rmem_scale);
      copy(tZero_gZero(_, _, _, ii), rmem_zero);
      transform(rmem_op_q, rmem_op_scaled, [] (const QuantizedElement& elt) { return ElementScale(elt); } );
      transform(rmem_zero, rmem_zero_buf, [] (const ElementZero& elt) { return ElementScale(elt); } );
      transform(rmem_op_scaled, rmem_scale, rmem_op_scaled, multiplies{});
      transform(rmem_op_scaled, rmem_zero_buf, rmem_op_scaled, plus{});
      transform(rmem_op_scaled, rmem_op_dq, [] (const ElementScale& elt) { return DequantizedElement(elt); } );
      copy(rmem_op_dq, tOpDq_gOpDq(_, _, _, ii));
    }
  }
}

template <class QuantizedElement, 
          class DequantizedElement,
          class OperandLayout,
          class ElementScale,
          class ElementZero,
          class ScaleLayout>
void dequantize_weight(DequantizedElement* dq_buffer,
                       QuantizedElement const* q_buffer,
                       OperandLayout const operand_layout,
                       ElementScale const* scale_buffer,
                       ElementZero const* zero_buffer,
                       ScaleLayout const scale_layout,
                       int const group_size) {
  
  using namespace cute;

  constexpr int tpb = 128;
  auto thr_layout = make_layout(make_shape(Int<tpb>{}));

  const auto num_rows = get<0>(shape(operand_layout));
  const auto gemm_k = get<1>(shape(operand_layout));   // [MN, K, L]
  const auto batches = get<2>(shape(operand_layout));  // [MN, K, L]
  const auto scale_k = get<1>(shape(scale_layout));    // [MN, Scale_K, L]

  if (num_rows != size<0>(scale_layout)) {
    std::cerr << "Invalid first dimension for scales. Must match first dim for weights."
              << " But got shapes " << shape(operand_layout) << " " << shape(scale_layout) 
              << std::endl;
    exit(-1);
  }

  const auto scale_stride0 = get<0>(stride(scale_layout));
  const auto scale_stride1 = get<1>(stride(scale_layout));
  const auto scale_stride2 = get<2>(stride(scale_layout));

  auto scale_shape_bcast = make_shape(num_rows, make_shape(group_size, scale_k), batches);
  auto scale_stride_bcast = make_stride(scale_stride0, make_stride(0, scale_stride1), scale_stride2);
  auto scale_layout_bcast = make_layout(scale_shape_bcast, scale_stride_bcast);

  const auto blocks_x = gemm_k;
  const auto blocks_y = batches;

  dim3 blocks(blocks_x, blocks_y, 1);
  dequantize_weight_kernel<<<blocks, tpb>>>(dq_buffer, q_buffer, operand_layout, scale_buffer, zero_buffer, scale_layout_bcast, thr_layout);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
}

enum class QuantTypeClass {
    INT8_WEIGHT_ONLY,
    PACKED_INT4_WEIGHT_ONLY,
    PACKED_INT2_WEIGHT_ONLY,  // W2A16 (4 uint2/byte); mirrors PACKED_INT4 with ELTS_PER_BYTE=4
    PACKED_INT1_WEIGHT_ONLY   // W1A16 (8 uint1/byte); Q3/Q5 high plane; mirrors PACKED_INT2 with ELTS_PER_BYTE=8
};

int get_bits_in_quant_type(QuantTypeClass quant_type)
{
    switch (quant_type) {
        case QuantTypeClass::INT8_WEIGHT_ONLY:
            return 8;
        case QuantTypeClass::PACKED_INT4_WEIGHT_ONLY:
            return 4;
        case QuantTypeClass::PACKED_INT2_WEIGHT_ONLY:
            return 2;
        case QuantTypeClass::PACKED_INT1_WEIGHT_ONLY:
            return 1;
        default:
            //FT_CHECK_WITH_INFO(false, "Invalid quant_type");
            return -1;
    }
}

// The data is permuted such that:
// For int8, each group of 16 rows is permuted using the map below:
//  0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
// For int4, each group of 32 rows is permuted using the map below:
//  0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22 23 30 31
// For int4 with int8 mma, each group of 32 rows is permuted using the map below:
//  0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23 8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31
// ------------------------------------------------------------------------------------------------------------------
// THE FIVE-STEP OFFLINE PIPELINE USED TO LIVE HERE and has been DELETED:
//     subbyte_transpose -> permute_B_rows_for_mixed_gemm -> subbyte_transpose -> interleave_column_major_tensor_ppu
//     -> add_bias_and_interleave_{int8,int4,int2,int1}s_inplace
// ~500 lines, none of which had a caller outside preprocess_weights_for_mixed_gemm below.
//
// What they computed is a POSITION map, and that map is now DERIVED rather than hand-written: xplane::plane_map
// composes pi = right_inverse(partition_fragment_B(...).layout()) with partition_B, the swzl atom's LogicalTV and
// MixGemmEmit, and xplane::place_from_map walks it straight into the buffer. One pass instead of five, and 2.05x
// faster. The rung-5 defect is the argument for doing this: a hand-written placement can be wrong in a way no amount
// of self-consistency reveals, because the map and the writer were two expressions of one premise.
//
// The originals are preserved in fold_derivation/legacy_pipeline.hpp as the REFERENCE l61 gates against -- deleting
// them there too would make the gate compare the derived walk with itself. They must never regain a production caller.
// ------------------------------------------------------------------------------------------------------------------


// (f) nfold_regroup_gmem AND nfold_place_bits_int1_tk64 USED TO LIVE HERE and have moved to
// fold_derivation/legacy_pipeline.hpp, gate-side only, next to the deleted five-step pipeline.
//
// nfold_regroup_gmem moved whole uint32 words, so each word carried ONE logical column -- correct only while
// cols_per_word == 1, i.e. warp N extent 32. Every caller was w32x32, so all of them were right; but w64x32 measured
// +7 to +9 points on int2 and int4 this session, so WN=64 is where the tuning is going and the first caller to follow it
// would have miscomputed silently. nfold_place_bits_int1_tk64 was the bit-granular escape hatch for exactly that case,
// int1 only.
//
// xplane::place_derived replaces both: it covers the fold walk AND the interleave-256 walk, so one call also absorbs
// preprocess_weights_for_mixed_gemm. Byte-identical on all 27 configurations the callers actually use (l64) -- which
// now includes every row of test_fold_int2's FOLD_CONFIGS, the last non-gate consumer, migrated under #13 -- and
// correct where the whole-word packer is not.
//
// So there is now NO consumer outside fold_derivation/, and build.sh FAILS the build if a CMake-built source grows
// one: quarantine by structure rather than by the reader noticing this comment.

// The derived replacement for the deleted five steps. Same signature, so no call site changes.
//
// TWO THINGS THE FIVE STEPS DID THAT A POSITION MAP DOES NOT, both handled here:
//   * int4's +8 (add_bias_and_interleave_int4s) is a VALUE transform, so MixGemmEmit deliberately omits it. int2 and
//     int1 apply no bias -- their own comments say so explicitly.
//   * the row-major / column-major input convention, resolved while unpacking.
//
// A REPRESENTATIVE TILE. plane_map needs a kernel tile; this function is not told one. That is sound because the
// unfolded placement is TILE-INVARIANT, verified byte-identical to the deleted pipeline across 11 configurations
// (TM 32/64/128, TN 64/128/256, TK 64/128/256, w32x32 / w32x64 / w64x64) and all three live widths in
// fold_derivation/l61. Any (TN, TK) dividing (N, K) within the delivery bound gives the same buffer, so TN=64 w32x32
// and the narrowest legal TK are used. The bound is DL >= 1 (TK*Bits >= 256) and RPS >= 1 (the 32 B run must fit the
// 256-element interleave), which pins int1 to TK=256 exactly and allows int2 128 and int4 64.
//
// Anything outside that is a HARD ERROR rather than a silent fallback: int8 is unreachable (no QuantType in the tree
// is 8-bit) and MixGemmEmit covers 1/2/4 only, so a fallback would only hide a new caller's mistake.
template<bool is_rowmajor, int RowsPerTile, int FoldTK = 0>
void preprocess_weights_for_mixed_gemm(int8_t*                    preprocessed_quantized_weight,
                                       const int8_t*              row_major_quantized_weight,
                                       const std::vector<size_t>& shape,
                                       QuantTypeClass             quant_type)
{
    static_assert(RowsPerTile == 256,
        "only the interleave-256 destination is derived; RowsPerTile == -1 had no live caller when the five steps went");
    static_assert(FoldTK == 0,
        "the fold is applied by the caller (xplane::place_derived / place_int1), not by a FoldTK parameter here");

    const size_t L = shape.size() == 3 ? shape[0] : 1;
    const int    K = int(shape[shape.size() - 2]), N = int(shape[shape.size() - 1]);
    const int    Bits = get_bits_in_quant_type(quant_type);
    const int    EPB = 8 / Bits, MASK = (1 << Bits) - 1;
    const size_t nb  = (size_t)K * N * Bits / 8;

    if (Bits != 1 && Bits != 2 && Bits != 4)
      throw std::runtime_error("preprocess_weights_for_mixed_gemm: only 1/2/4-bit codes are derived (MixGemmEmit)");
    if (N % 64 != 0)
      throw std::runtime_error("preprocess_weights_for_mixed_gemm: N must be a multiple of 64 for the representative tile");
    const int TKrep = Bits == 1 ? 256 : Bits == 2 ? 128 : 64;
    if (K % TKrep != 0)
      throw std::runtime_error("preprocess_weights_for_mixed_gemm: K must be a multiple of the representative TK");

    for (size_t b = 0; b < L; ++b) {
      const int8_t* src = row_major_quantized_weight + b * nb;
      std::vector<uint8_t> q((size_t)K * N);
      for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
          const size_t lin = is_rowmajor ? (size_t)k * N + n : (size_t)n * K + k;
          int v = (int(src[lin / EPB]) >> (Bits * int(lin % EPB))) & MASK;
          if (Bits == 4) v = (v + 8) & MASK;      // (sign_extend(v) + 8) == (v + 8) mod 16
          q[(size_t)k * N + n] = uint8_t(v);
        }
      int8_t* dst = preprocessed_quantized_weight + b * nb;
      if      (Bits == 1) xplane::place_derived<1, 64, 64, 256, 32, 32, 1>(dst, q, N, K);
      else if (Bits == 2) xplane::place_derived<2, 64, 64, 128, 32, 32, 1>(dst, q, N, K);
      else                xplane::place_derived<4, 64, 64,  64, 32, 32, 1>(dst, q, N, K);
    }
}
