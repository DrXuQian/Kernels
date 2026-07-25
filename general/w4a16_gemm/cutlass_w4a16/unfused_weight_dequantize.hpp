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
void permute_B_rows_for_mixed_gemm(int8_t*                    permuted_quantized_tensor,
                                   const int8_t*              quantized_tensor,
                                   const std::vector<size_t>& shape,
                                   QuantTypeClass             quant_type,
                                   const int64_t              arch_version,
                                   bool                       is_int8_mma)
{

    // We only want to run this step for weight only quant.
    //FT_CHECK(quant_type == QuantTypeClass::PACKED_INT4_WEIGHT_ONLY || quant_type == QuantTypeClass::INT8_WEIGHT_ONLY);

    //FT_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    // printf("num_rows = %d, num_cols = %d\n", num_rows, num_cols);

    const int BITS_PER_ELT  = get_bits_in_quant_type(quant_type);
    const int K             = 16 / BITS_PER_ELT;
    const int ELTS_PER_BYTE = 8 / BITS_PER_ELT;
    const int ELTS_PER_REG  = 32 / BITS_PER_ELT;

    const uint32_t* input_byte_ptr  = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t*       output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

    int       MMA_SHAPE_N    = 8;
    int       B_ROWS_PER_MMA = 8 * K;
    const int elts_in_int32  = 32 / BITS_PER_ELT;

    const int num_vec_cols = num_cols / elts_in_int32;

    // The code is written as below so it works for both int8 and packed int4.
    for (int expert = 0; expert < num_experts; ++expert) {
        const int64_t matrix_offset = expert * int64_t(num_rows) * int64_t(num_vec_cols);
        for (int base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA) {
            for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {

                for (int write_col = 0; write_col < num_vec_cols; ++write_col) {
                    const int write_row = base_row + tile_row;
                    int tile_read_row = 0;
                    if (is_int8_mma) {
                        tile_read_row = (tile_row % 8) / 4 * 16 + tile_row / 8 * 4 + tile_row % 4;
                    } else {
                        tile_read_row = 8 * (((tile_row % ELTS_PER_REG) / 2)) + tile_row % 2 + 2 * (tile_row / ELTS_PER_REG);
                    }
                    const int read_row = base_row + tile_read_row;
                    const int read_col = write_col;

                    const int64_t read_offset  = matrix_offset + int64_t(read_row) * num_vec_cols + read_col;
                    const int64_t write_offset = matrix_offset + int64_t(write_row) * num_vec_cols + write_col;

                    output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
                }
            }
        }
    }
}

void add_bias_and_interleave_int8s_inplace(int8_t* int8_tensor, const size_t num_elts)
{
    for (int ii = 0; ii < num_elts; ++ii) {
        int8_tensor[ii] = int8_t(int(int8_tensor[ii]) + 128);
    }

    // Step 2 will transform the layout of a 32-bit register in device in order to match the int4 layout. This has no
    // performance benefit and is purely so that int4 and int8 have the same layout.
    // Pictorially, this does the following:
    // bit 32                                                      0
    //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 8 bits)
    //
    // And it will rearrange the output 32 bit register to be the following:
    // bit 32                                                      0
    //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)
    // FT_CHECK_WITH_INFO(num_elts % 4 == 0, "Dimensions of int8 tensor must be a multiple of 4 for register relayout");
    for (size_t base = 0; base < num_elts; base += 4) {
        std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
    }
}

void add_bias_and_interleave_int4s_inplace(int8_t* packed_int4_tensor, const size_t num_elts)
{
    const int num_bytes = num_elts / 2;

    // Step 1 will be to transform all the int4s to unsigned in order to make the dequantize take as little
    // instructions as possible in the device code.
    for (size_t ii = 0; ii < num_bytes; ++ii) {
        int8_t transformed_packed_int4s = 0;
        int8_t transformed_first_elt =
            (int8_t(packed_int4_tensor[ii] << 4) >> 4) + 8;  // The double shift here is to ensure sign extension
        int8_t transformed_second_elt = (packed_int4_tensor[ii] >> 4) + 8;

        // FT_CHECK_WITH_INFO(transformed_first_elt >= 0 && transformed_first_elt <= 15,
        //                    "Illegal result for int4 transform (first elt)");
        // FT_CHECK_WITH_INFO(transformed_second_elt >= 0 && transformed_second_elt <= 15,
        //                    "Illegal result for int4 transform (second elt)");

        // We don't need to mask in these ops since everything should be in the range 0-15
        transformed_packed_int4s |= transformed_first_elt;
        transformed_packed_int4s |= (transformed_second_elt << 4);
        packed_int4_tensor[ii] = transformed_packed_int4s;
    }

    // Step 2 will transform the layout of a 32-bit register in device in order to minimize the number of shift & logical
    // instructions That are needed to extract the int4s in the GEMM main loop. Pictorially, the loop below will do the
    // following: Take as input a 32 bit register with layout: bit 32 0
    //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt occupies 4 bits)
    //
    // And it will rearrange the output 32 bit register to be the following:
    // bit 32                                                      0
    //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt occupies 4 bits)

    // FT_CHECK_WITH_INFO(num_bytes % 4 == 0, "Dimensions of int4 tensor must be a multiple of 8 for register relayout");
    const size_t num_registers = num_bytes / 4;

    uint32_t* register_ptr = reinterpret_cast<uint32_t*>(packed_int4_tensor);
    for (size_t ii = 0; ii < num_registers; ++ii) {
        const uint32_t current_register     = register_ptr[ii];
        uint32_t       transformed_register = 0;

        for (int dest_idx = 0; dest_idx < 8; ++dest_idx) {
            const int src_idx    = dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
            const int src_shift  = 4 * src_idx;
            const int dest_shift = 4 * dest_idx;

            const uint32_t src_bits = (current_register >> src_shift) & 0xF;
            transformed_register |= (src_bits << dest_shift);
        }
        register_ptr[ii] = transformed_register;
    }
}

void add_bias_and_interleave_int2s_inplace(int8_t* packed_int2_tensor, const size_t num_elts)
{
    // W2A16 / uint2b_t: NO +bias (uint2 already [0,3]; the per-group affine 'zero' absorbs the offset). Register
    // relayout to match the int2 lop3 magic converter (Stage 2) -- mirror of int4's split-at-4, here split-at-8
    // (16 crumbs / 32-bit register vs int4's 8 nibbles). Rearranges one 32-bit reg (the 16 K of ONE N-row):
    //   input : [c15 c14 ... c1 c0]  (each crumb 2 bits)
    //   output: [c15 c13 c11 c9 c7 c5 c3 c1 | c14 c12 c10 c8 c6 c4 c2 c0]
    //   dest d <- src crumb (d<8 ? 2d : 2(d-8)+1). Then the converter's lop3 h[t] = (crumb 2t, crumb 2t+1) reads
    //   the crumbs back in sequential order 0..15 (each lop3 pairs a crumb with the one 16 bits = 8 crumbs away).
    const size_t num_bytes     = num_elts / 4;   // 4 crumbs per byte
    const size_t num_registers = num_bytes / 4;
    uint32_t* register_ptr = reinterpret_cast<uint32_t*>(packed_int2_tensor);
    for (size_t ii = 0; ii < num_registers; ++ii) {
        const uint32_t current_register     = register_ptr[ii];
        uint32_t       transformed_register = 0;
        for (int dest_idx = 0; dest_idx < 16; ++dest_idx) {
            const int      src_idx   = dest_idx < 8 ? 2 * dest_idx : 2 * (dest_idx - 8) + 1;
            const uint32_t src_crumb = (current_register >> (2 * src_idx)) & 0x3u;
            transformed_register |= (src_crumb << (2 * dest_idx));
        }
        register_ptr[ii] = transformed_register;
    }
}

void add_bias_and_interleave_int1s_inplace(int8_t* packed_int1_tensor, const size_t num_elts)
{
    // W1A16: NO +bias. Register relayout for the int1 lop3 magic converter -- mirror of int2's split-at-8, here
    // split-at-16 (32 bits / 32-bit register). dest d <- src bit (d<16 ? 2d : 2(d-16)+1); then the converter's
    // lop3 h[t] = (bit 2t, bit 2t+1) reads the bits back as sequential adjacent pairs (the validated magic-OR order).
    const size_t num_bytes     = num_elts / 8;   // 8 bits per byte
    const size_t num_registers = num_bytes / 4;
    uint32_t* register_ptr = reinterpret_cast<uint32_t*>(packed_int1_tensor);
    for (size_t ii = 0; ii < num_registers; ++ii) {
        const uint32_t current_register     = register_ptr[ii];
        uint32_t       transformed_register = 0;
        for (int dest_idx = 0; dest_idx < 32; ++dest_idx) {
            const int      src_idx = dest_idx < 16 ? 2 * dest_idx : 2 * (dest_idx - 16) + 1;
            const uint32_t src_bit = (current_register >> src_idx) & 0x1u;
            transformed_register |= (src_bit << dest_idx);
        }
        register_ptr[ii] = transformed_register;
    }
}

void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantTypeClass quant_type)
{
    if (quant_type == QuantTypeClass::INT8_WEIGHT_ONLY) {
        add_bias_and_interleave_int8s_inplace(tensor, num_elts);
    }
    else if (quant_type == QuantTypeClass::PACKED_INT4_WEIGHT_ONLY) {
        add_bias_and_interleave_int4s_inplace(tensor, num_elts);
    }
    else if (quant_type == QuantTypeClass::PACKED_INT2_WEIGHT_ONLY) {
        // Stage-2: was identity; now the split-at-8 register relayout that the int2 lop3 magic converter needs.
        add_bias_and_interleave_int2s_inplace(tensor, num_elts);
    }
    else if (quant_type == QuantTypeClass::PACKED_INT1_WEIGHT_ONLY) {
        // W1A16 lop3: split-at-16 register relayout (mirror of the int2 split-at-8) that the int1 lop3 converter needs.
        add_bias_and_interleave_int1s_inplace(tensor, num_elts);
    }
    else {
        // FT_CHECK_WITH_INFO(false, "Invalid quantization type for interleaving.");
        assert(false);
    }
}

void interleave_column_major_tensor_ppu(int8_t*                    interleaved_quantized_tensor,
                                        const int8_t*              quantized_tensor,
                                        const std::vector<size_t>& shape,
                                        QuantTypeClass             quant_type,
                                        const int                  rows_per_tile)
{

    // We only want to run this step for weight only quant.
    // FT_CHECK(quant_type == QuantTypeClass::PACKED_INT4_WEIGHT_ONLY || quant_type == QuantTypeClass::INT8_WEIGHT_ONLY);

    // FT_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];     // k
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];     // n

    const int BITS_PER_ELT  = get_bits_in_quant_type(quant_type);
    const int elts_in_int32 = 32 / BITS_PER_ELT;

    // FT_CHECK_WITH_INFO(
    //     !(num_rows % elts_in_int32),
    //     fmtstr("The number of rows must be a multiple of %d but the number of rows is %d.", elts_in_int32, num_rows));

    // FT_CHECK_WITH_INFO(!(num_cols % rows_per_tile),
    //                    fmtstr("The number of columns must be a multiple of %d but the number of columns is %ld",
    //                           rows_per_tile,
    //                           num_cols));

    const uint32_t* input_byte_ptr  = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t*       output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

    const int num_vec_rows      = num_rows / elts_in_int32;
    const int vec_rows_per_tile = rows_per_tile / elts_in_int32;

    for (int expert = 0; expert < num_experts; ++expert) {
        const int64_t matrix_offset = expert * int64_t(num_vec_rows) * int64_t(num_cols);
        for (int read_col = 0; read_col < num_cols; ++read_col) {
            for (int vec_read_row = 0; vec_read_row <num_vec_rows; ++vec_read_row) {
                const int64_t read_offset = matrix_offset + int64_t(read_col) * num_vec_rows + vec_read_row;
                const int64_t num_tile = vec_read_row / vec_rows_per_tile;
                const int64_t tile_idx = vec_read_row % vec_rows_per_tile;
                const int64_t write_offset = matrix_offset + num_tile * vec_rows_per_tile * num_cols + read_col * vec_rows_per_tile + tile_idx;
                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}

// We need to use this transpose to correctly handle packed int4 and int8 data
// The reason this code is relatively complex is that the "trivial" loops took a substantial
// amount of time to transpose leading to long preprocessing times. This seemed to be a big
// issue for relatively large models.
template<QuantTypeClass quant_type>
void subbyte_transpose_impl(int8_t*                    transposed_quantized_tensor,
                            const int8_t*              quantized_tensor,
                            const std::vector<size_t>& shape)
{
    const int bits_per_elt = get_bits_in_quant_type(quant_type);

    // FT_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    const size_t col_bytes       = num_cols * bits_per_elt / 8;
    const size_t col_bytes_trans = num_rows * bits_per_elt / 8;
    const size_t num_bytes       = size_t(num_experts) * num_rows * col_bytes;

    const uint8_t* input_byte_ptr  = reinterpret_cast<const uint8_t*>(quantized_tensor);
    uint8_t*       output_byte_ptr = reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

    static_assert(quant_type == QuantTypeClass::INT8_WEIGHT_ONLY || quant_type == QuantTypeClass::PACKED_INT4_WEIGHT_ONLY
                  || quant_type == QuantTypeClass::PACKED_INT2_WEIGHT_ONLY || quant_type == QuantTypeClass::PACKED_INT1_WEIGHT_ONLY, "");
    static constexpr int ELTS_PER_BYTE = quant_type == QuantTypeClass::INT8_WEIGHT_ONLY ? 1
                                       : (quant_type == QuantTypeClass::PACKED_INT4_WEIGHT_ONLY ? 2
                                       : (quant_type == QuantTypeClass::PACKED_INT2_WEIGHT_ONLY ? 4 : 8));

    static constexpr int M_TILE_L1 = 64;
    static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
    uint8_t              cache_buf[M_TILE_L1][N_TILE_L1];

    static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

    // We assume the dims are a multiple of vector width. Our kernels only handle dims which are multiples
    // of 64 for weight-only quantization. As a result, this seemed like a reasonable tradeoff because it
    // allows GCC to emit vector instructions.
    if (col_bytes_trans % VECTOR_WIDTH || col_bytes % VECTOR_WIDTH) {
        auto err_msg = "Number of bytes for rows and cols must be a multiple of " + std::to_string(VECTOR_WIDTH)
                + ". However, num_rows_bytes = " + std::to_string(col_bytes_trans)
                + " and num_col_bytes = " + std::to_string(col_bytes) + ".";
        throw std::runtime_error(err_msg);
    }

    const int num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
    const int num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

    for (size_t expert = 0; expert < num_experts; ++expert) {
        const size_t matrix_offset = expert * num_rows * col_bytes;
        for (size_t row_tile_start = 0; row_tile_start < num_rows; row_tile_start += M_TILE_L1) {
            for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes; col_tile_start_byte += N_TILE_L1) {

                const int row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
                const int col_limit = std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

                for (int ii = 0; ii < M_TILE_L1; ++ii) {
                    const int row = row_tile_start + ii;

                    for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
                        const int col = col_tile_start_byte + jj;

                        const size_t logical_src_offset = matrix_offset + row * col_bytes + col;

                        if (row < row_limit && col < col_limit) {
                            for (int v = 0; v < VECTOR_WIDTH; ++v) {
                                cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
                            }
                        }
                    }
                }

                if (quant_type == QuantTypeClass::INT8_WEIGHT_ONLY) {
                    for (int ii = 0; ii < M_TILE_L1; ++ii) {
                        for (int jj = ii + 1; jj < N_TILE_L1; ++jj) {
                            std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
                        }
                    }
                }
                else if (quant_type == QuantTypeClass::PACKED_INT4_WEIGHT_ONLY) {

                    for (int ii = 0; ii < M_TILE_L1; ++ii) {
                        // Using M_TILE_L1 here is deliberate since we assume that the cache tile
                        // is square in the number of elements (not necessarily the number of bytes).
                        for (int jj = ii + 1; jj < M_TILE_L1; ++jj) {
                            const int ii_byte       = ii / ELTS_PER_BYTE;
                            const int ii_bit_offset = ii % ELTS_PER_BYTE;

                            const int jj_byte       = jj / ELTS_PER_BYTE;
                            const int jj_bit_offset = jj % ELTS_PER_BYTE;

                            uint8_t src_elt = 0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
                            uint8_t tgt_elt = 0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

                            cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
                            cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

                            cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
                            cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
                        }
                    }
                }
                else if (quant_type == QuantTypeClass::PACKED_INT2_WEIGHT_ONLY) {
                    // 2-bit crumb transpose: mirror of the int4 nibble path, ELTS_PER_BYTE=4 (>>2*off, mask 0x3).
                    for (int ii = 0; ii < M_TILE_L1; ++ii) {
                        for (int jj = ii + 1; jj < M_TILE_L1; ++jj) {
                            const int ii_byte       = ii / ELTS_PER_BYTE;
                            const int ii_bit_offset = ii % ELTS_PER_BYTE;
                            const int jj_byte       = jj / ELTS_PER_BYTE;
                            const int jj_bit_offset = jj % ELTS_PER_BYTE;
                            uint8_t src_elt = 0x3 & (cache_buf[ii][jj_byte] >> (2 * jj_bit_offset));
                            uint8_t tgt_elt = 0x3 & (cache_buf[jj][ii_byte] >> (2 * ii_bit_offset));
                            cache_buf[ii][jj_byte] &= uint8_t(~(0x3 << (2 * jj_bit_offset)));
                            cache_buf[jj][ii_byte] &= uint8_t(~(0x3 << (2 * ii_bit_offset)));
                            cache_buf[ii][jj_byte] |= (tgt_elt << (2 * jj_bit_offset));
                            cache_buf[jj][ii_byte] |= (src_elt << (2 * ii_bit_offset));
                        }
                    }
                }
                else if (quant_type == QuantTypeClass::PACKED_INT1_WEIGHT_ONLY) {
                    // 1-bit transpose: mirror of int2, ELTS_PER_BYTE=8 (>>1*off, mask 0x1).
                    for (int ii = 0; ii < M_TILE_L1; ++ii) {
                        for (int jj = ii + 1; jj < M_TILE_L1; ++jj) {
                            const int ii_byte       = ii / ELTS_PER_BYTE;
                            const int ii_bit_offset = ii % ELTS_PER_BYTE;
                            const int jj_byte       = jj / ELTS_PER_BYTE;
                            const int jj_bit_offset = jj % ELTS_PER_BYTE;
                            uint8_t src_elt = 0x1 & (cache_buf[ii][jj_byte] >> jj_bit_offset);
                            uint8_t tgt_elt = 0x1 & (cache_buf[jj][ii_byte] >> ii_bit_offset);
                            cache_buf[ii][jj_byte] &= uint8_t(~(0x1 << jj_bit_offset));
                            cache_buf[jj][ii_byte] &= uint8_t(~(0x1 << ii_bit_offset));
                            cache_buf[ii][jj_byte] |= (tgt_elt << jj_bit_offset);
                            cache_buf[jj][ii_byte] |= (src_elt << ii_bit_offset);
                        }
                    }
                }
                else {
                    // FT_CHECK_WITH_INFO(false, "Unsupported quantization type.");
                    assert(false);
                }

                const size_t row_tile_start_trans      = col_tile_start_byte * ELTS_PER_BYTE;
                const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

                const int row_limit_trans = std::min(row_tile_start_trans + M_TILE_L1, num_cols);
                const int col_limit_trans = std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

                for (int ii = 0; ii < M_TILE_L1; ++ii) {
                    const int row = row_tile_start_trans + ii;
                    for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
                        const int col = col_tile_start_byte_trans + jj;

                        const size_t logical_tgt_offset = matrix_offset + row * col_bytes_trans + col;

                        if (row < row_limit_trans && col < col_limit_trans) {
                            for (int v = 0; v < VECTOR_WIDTH; ++v) {
                                output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
                            }
                        }
                    }
                }
            }
        }
    }
}

void subbyte_transpose(int8_t*                    transposed_quantized_tensor,
                       const int8_t*              quantized_tensor,
                       const std::vector<size_t>& shape,
                       QuantTypeClass             quant_type)
{

    if (quant_type == QuantTypeClass::INT8_WEIGHT_ONLY) {
        subbyte_transpose_impl<QuantTypeClass::INT8_WEIGHT_ONLY>(transposed_quantized_tensor, quantized_tensor, shape);
    }
    else if (quant_type == QuantTypeClass::PACKED_INT4_WEIGHT_ONLY) {
        subbyte_transpose_impl<QuantTypeClass::PACKED_INT4_WEIGHT_ONLY>(
            transposed_quantized_tensor, quantized_tensor, shape);
    }
    else if (quant_type == QuantTypeClass::PACKED_INT2_WEIGHT_ONLY) {
        subbyte_transpose_impl<QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(
            transposed_quantized_tensor, quantized_tensor, shape);
    }
    else if (quant_type == QuantTypeClass::PACKED_INT1_WEIGHT_ONLY) {
        subbyte_transpose_impl<QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(
            transposed_quantized_tensor, quantized_tensor, shape);
    }
    else {
        // FT_CHECK_WITH_INFO(false, "Invalid quant_tye");
        assert(false);
    }
}

// N-FOLD (P1.1): frees a sparse format's TileShape.K from the AIU 32-byte-contiguous-K floor. After
// interleave_column_major_tensor_ppu has laid each column's K into 256-K super-tiles of VRPT uint32-vecs, this
// interleaves N-column groups PAIRED ACROSS HALVES (column g with g + N/F, ...) at FoldTK-vec granularity, so each AIU
// contiguous run (still 32B, i.e. 256 elems for the sparse plane) is [n_a's FoldTK-K][n_b's FoldTK-K][...]. That lets
// the kernel run at TileShape.K = FoldTK (A-smem = TileM*FoldTK*2, halved/quartered) while the AIU still reads a legal
// 32B run. The two folded N-columns land in the lower vs upper mma-K atom blocks (cute_nfold2.cu: 0/64 cross-contam),
// so the mainloop consumes them as 2 output-N groups reusing one A -- no reduce, no converter change.
// Placement VERIFIED bijective and consistent with the fragment split in scratchpad/nfold_p11.cpp.
inline void nfold_column_pairs_ppu(int8_t*                    out,
                                   const int8_t*              in,
                                   const std::vector<size_t>& shape,
                                   QuantTypeClass             quant_type,
                                   const int                  fold_tk,   // TileShape.K in ELEMENTS (e.g. 64)
                                   const int                  fold_tn = 64)  // TileShape.N -- the fold group PERIOD
{
    const int    bits        = get_bits_in_quant_type(quant_type);
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t K           = shape.size() == 2 ? shape[0] : shape[1];   // rows
    const size_t N           = shape.size() == 2 ? shape[1] : shape[2];   // cols
    const int    EPV  = 32 / bits;              // elements per uint32 vec
    const int    VRPT = 256 / EPV;              // uint32-vecs per 256-K super-tile (matches interleave RowsPerTile=256)
    const int    TKv  = fold_tk / EPV;          // uint32-vecs per FoldTK
    const int    F    = (32 * 8 / bits) / fold_tk;   // fold factor: how many N columns fill one 32B run
    const int    nsuper = int(K) / 256;
    const int    chunks = VRPT / TKv;
    assert(F >= 2 && "nfold: FoldTK too large (single column already >= 32B, no fold needed)");
    assert(int(N) % F == 0 && VRPT % TKv == 0 && int(K) % 256 == 0 && "nfold: shape not fold-divisible");
    const uint32_t* src = reinterpret_cast<const uint32_t*>(in);
    uint32_t*       dst = reinterpret_cast<uint32_t*>(out);
    for (size_t e = 0; e < num_experts; ++e) {
        const size_t moff = e * (N * K / EPV);
        for (int T = 0; T < nsuper; ++T)
            for (size_t g = 0; g < N / F; ++g)
                for (int ci = 0; ci < chunks; ++ci)
                    for (int f = 0; f < F; ++f)
                        for (int j = 0; j < TKv; ++j) {
                            // CROSS-HALF pairing: the MMA pairs logical column n with n + N/F (not n with n+1).
                            // Derived from the fragment slot maps (scratchpad/cute_nfold7.cu): at mma K-atom j the MMA
                            // wants (n, k=16j+..) from swzl atom j and (n + Ng, same k) from swzl atom j+F/..., i.e.
                            // the SECOND half of the folded run must carry the column Ng away, not the neighbour.
                            // Adjacent pairing (n = g*F + f) is what produced the measured n_used = n & ~1.
                            // TILE-PERIODIC grouping (not global): the kernel iterates B in TileShape.N-wide tiles, so a
                            // column may only be folded with a partner INSIDE its own tile -- pairing n with n+N/F
                            // (global halves) puts the partner in a different tile, which the block never loads. The
                            // fold group must therefore repeat with period fold_tn: within each tile of fold_tn
                            // columns, physical row g takes logical columns g and g + fold_tn/F.
                            // Two candidate placements exist for which logical column each half of a physical row
                            // carries; the in-tile-half one measured n_used = n//2 (i.e. output n read the wrong
                            // half), so switch to ADJACENT: physical row r carries logical columns F*r + f.
                            // MOEG_FOLD_CROSSHALF=1 restores the other candidate for A/B comparison.
                            const size_t tile   = g / (fold_tn / F);
                            const size_t g_in   = g % (fold_tn / F);
#if defined(MOEG_FOLD_CROSSHALF)
                            const size_t n      = tile * fold_tn + f * (fold_tn / F) + g_in;   // in-tile half pairing
#else
                            const size_t n      = tile * fold_tn + g_in * F + f;               // ADJACENT pairing
#endif
                            const int    i     = ci * TKv + j;                          // tile_idx within super-tile
                            const size_t o_off = (size_t)T * VRPT * N + g * F * VRPT
                                               + (size_t)ci * F * TKv + (size_t)f * TKv + j;
                            const size_t i_off = (size_t)T * VRPT * N + n * VRPT + i;
                            dst[moff + o_off] = src[moff + i_off];
                        }
    }
}

// FoldTK: 0 = no N-fold (default; existing callers byte-identical). >0 = TileShape.K in ELEMENTS to fold to.
template<bool is_rowmajor, int RowsPerTile, int FoldTK = 0>
void preprocess_weights_for_mixed_gemm(int8_t*                    preprocessed_quantized_weight,
                                       const int8_t*              row_major_quantized_weight,
                                       const std::vector<size_t>& shape,
                                       QuantTypeClass             quant_type)
{
    // FT_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    size_t num_elts = 1;
    for (const auto& dim : shape) {
        num_elts *= dim;
    }

    const size_t num_bytes = num_elts * get_bits_in_quant_type(quant_type) / 8;

    std::vector<int8_t> src_buf(num_bytes);
    std::vector<int8_t> dst_buf(num_bytes);
    std::copy(row_major_quantized_weight, row_major_quantized_weight + num_bytes, src_buf.begin());

    if constexpr(!is_rowmajor) {
      // transpose to row major
      subbyte_transpose(dst_buf.data(), src_buf.data(), {shape[1], shape[0]}, quant_type);
      src_buf.swap(dst_buf);
    }

    permute_B_rows_for_mixed_gemm(dst_buf.data(), src_buf.data(), shape, quant_type, 80, false);
    src_buf.swap(dst_buf);

    // transpose to column major
    subbyte_transpose(dst_buf.data(), src_buf.data(), shape, quant_type);
    src_buf.swap(dst_buf);

    if constexpr (RowsPerTile != -1) {
        // column major -> column interleaved 256 major
        interleave_column_major_tensor_ppu(dst_buf.data(), src_buf.data(), shape, quant_type, RowsPerTile);
        src_buf.swap(dst_buf);
    }
    if constexpr (FoldTK > 0) {
        // N-fold: interleave adjacent N-column groups at FoldTK-vec granularity (after interleave-256, before split).
        // Operates on whole uint32-vecs so it commutes with the per-uint32 split below. Off by default (FoldTK=0).
        nfold_column_pairs_ppu(dst_buf.data(), src_buf.data(), shape, quant_type, FoldTK);
        src_buf.swap(dst_buf);
    }
    add_bias_and_interleave_quantized_tensor_inplace(src_buf.data(), num_elts, quant_type);
    std::copy(src_buf.begin(), src_buf.end(), preprocessed_quantized_weight);
}
