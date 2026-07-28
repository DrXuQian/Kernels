#pragma once
// LEGACY five-step offline pipeline -- DELETED FROM PRODUCTION, kept here ONLY as the reference l61 gates against.
//
// These are subbyte_transpose -> permute_B_rows_for_mixed_gemm -> subbyte_transpose -> interleave_column_major_ppu ->
// add_bias_and_interleave, verbatim as they stood in unfused_weight_dequantize.hpp. What they compute is a POSITION
// map, and that map is now derived from cute instead (xplane::plane_map composes pi = right_inverse of
// partition_fragment_B's layout with partition_B, the swzl LogicalTV and MixGemmEmit; xplane::place_from_map walks it).
//
// WHY KEEP THEM AT ALL. If the gate's reference is deleted along with the code, the gate becomes a tautology -- it
// would be comparing the derived walk against itself and reporting 0. Keeping the old implementation on the GATE side
// is what makes "bit-identical" mean something. It has no callers in production and must never gain one.
//
// Everything below is the original code and the original comments, unedited, including the corrections recorded in
// them. Do not fix anything here; a reference that drifts is not a reference.
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../unfused_weight_dequantize.hpp"

namespace legacy {

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
    // N-FOLD is not a step here; the caller applies it afterwards. A CORRECTION to what this comment used to say:
    // it claimed the pipeline above "already interleaves several N columns into one 32B contiguous run AT CRUMB
    // LEVEL", citing vreg0/crumb0 -> (n0,k0) versus vreg0/crumb2 -> (n32,k0). That is FALSE. Running the pipeline
    // and recovering its map bit by bit (fold_derivation/l7_groundtruth.cu, and l13 across int1/int2/int4) shows
    // every 32-bit word holds ONE logical column: the two transposes preserve the n axis, permute_B_rows permutes
    // K, interleave-256 relocates whole uint32 vecs, and add_bias_and_interleave reorders bits WITHIN a word.
    //
    // The old claim also argued against the approach that actually works: nfold_regroup_gmem IS "a whole-uint32
    // permutation after interleave-256", and it is the validated one. What it must do -- and what the version of
    // this file before fold_derivation/l13 got wrong -- is invert interleave-256 properly rather than treat its
    // output as n-major, which only holds at K == 256.
    static_assert(FoldTK == 0,
        "the fold is applied by the caller (nfold_regroup_gmem, or nfold_place_bits_* when a word must carry "
        "several logical columns), not by a FoldTK parameter here");
    add_bias_and_interleave_quantized_tensor_inplace(src_buf.data(), num_elts, quant_type);
    std::copy(src_buf.begin(), src_buf.end(), preprocessed_quantized_weight);
}

} // namespace legacy
