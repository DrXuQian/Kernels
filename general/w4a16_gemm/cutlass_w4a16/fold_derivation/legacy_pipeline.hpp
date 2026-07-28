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

// ------------------------------------------------------------------------------------------------------------------
// (f) THE TWO N-FOLD PACKERS, moved out of the production header. Same standing as the five steps above: kept only so
// the gates have a reference that is not the derived walk itself. nfold_regroup_gmem moves whole uint32 words, correct
// only while cols_per_word == 1 (warp N extent 32); xplane::place_derived replaces both and is byte-identical on every
// configuration the old callers used (l64). Do not fix anything here -- a reference that drifts is not a reference.
// ------------------------------------------------------------------------------------------------------------------
// Placement VERIFIED bijective and consistent with the fragment split in scratchpad/nfold_p11.cpp.
// N-FOLD offline placement, derived from the KERNEL'S OWN gmem address arithmetic (not from a guessed arrangement).
// Read straight out of the fold collective's load_init_B (interleaved-256 branch) + AiuDesc::init:
//     folded mB layout : shape (N/F, (kCon, K*F/kCon)), stride (kCon, (1, kCon*(N/F)))
//     AIU descriptor   : dim_h = N/F, dim_w = kCon, cube_h = Ng, cube_w = AiuContElemSize
//     gB tiler         : N steps by Ng, K steps by F*TK
// => the buffer is (N/F) PHYSICAL ROWS, each kCon elements contiguous, and within a row each F*TK-element run holds
//    F logical N columns x TK k each. Output block [n0, n0+TN) reads physical rows [n0/F, n0/F + Ng).
// Everything above is pure layout arithmetic from source -- no hardware unknown -- which is why this replaces the
// previous five guessed arrangements (each of which measured 72-75%, i.e. random, because the global arrangement
// disagreed with this walk regardless of the within-run placement).
// The WITHIN-run element order keeps the standard pipeline's crumb order, so run this on the standard preprocess
// output and only MOVE whole 16-code words.
// BITS-parameterised: int2 uses F=2 / 16 codes per uint32, int1 uses F=4 / 32 codes. Everything else in the
// derivation is bit-width agnostic (it only moves whole uint32 words, preserving the pipeline's crumb order).
// LANDMINE, kept only until its remaining WN=32 callers move to xplane::place_derived. This moves whole uint32
// words, so each word carries ONE logical column -- correct only while cols_per_word == 1, i.e. warp N extent 32. At
// WN=64 the fragment wants TWO columns per word and no whole-word move can express it; line 676 additionally groups
// the folded columns STRIDED (n = g + f*Ng) where the kernel's SmemLayoutB_MmaView groups them ADJACENT
// (n = f + P1Fold*g). Both defects were invisible until (64,128,64) w64x64 F=2, which measured 32768/65536 slots
// misplaced (fold_derivation/l61) and half the output columns off by +32 on hardware.
inline void nfold_regroup_gmem(int8_t* out, const int8_t* in_std,
                               const std::vector<size_t>& shape, int fold_tn, int fold_tk, int bits)
{
    const size_t K = shape.size() == 2 ? shape[0] : shape[1];
    const size_t N = shape.size() == 2 ? shape[1] : shape[2];
    const int    F   = (32 * 8 / bits) / fold_tk;   // columns needed to fill the 32B run (int2@64 -> 2, int1@64 -> 4)
    const int    CPW = 32 / bits;                   // codes per uint32 word (int2 -> 16, int1 -> 32)
    const int    Ng  = fold_tn / F;
    const int    kCon = 256;
    const int    WPK = fold_tk / CPW;           // words per (n, K-tile)
    const int    W_ROW = kCon / CPW;            // words per physical row segment (kCon elements)
    const uint32_t* src = reinterpret_cast<const uint32_t*>(in_std);
    uint32_t*       dst = reinterpret_cast<uint32_t*>(out);
    const size_t nrow = N / F, nkb = (K * F) / kCon;
    for (size_t r = 0; r < nrow; ++r)                       // physical row
      for (size_t kb = 0; kb < nkb; ++kb)                   // super-tile along folded K
        for (int t = 0; t < kCon / (F * fold_tk); ++t)      // K-tiles inside this super-tile
          for (int f = 0; f < F; ++f)
            for (int w = 0; w < WPK; ++w) {
              // which logical (n, K-tile) supplies this word
              const size_t tile_n0 = (r / Ng) * fold_tn;                     // output block this row serves
              const size_t n_log   = tile_n0 + (r % Ng) + (size_t)f * Ng;    // partner column is n + Ng
              const size_t ktile   = (kb * (kCon / (F * fold_tk)) + t);      // which fold_tk block along K
              // BUG FIXED HERE. This used to be  n_log * WPN + ktile * WPK + w,  i.e. it read the
              // interleave-256 output as if it were n-major with row pitch WPN = K/CPW. It is not:
              // interleave_column_major_tensor_ppu writes  dst[nt*(vrpt*N) + c*vrpt + ti]  with nt = vr/vrpt,
              // ti = vr%vrpt, vrpt = 256/CPW -- the k-SUPERTILE is the outer index, not n. The two coincide only
              // when nvr == vrpt, i.e. K == 256, exactly one supertile. Every box run of the fold used the
              // harness default 256x256, so the mistake never showed: at K=512 the old form fetched
              // (n=0, k=256) where (n=64, k=0) was wanted -- measured in fold_derivation/l13_wholebuffer.cu,
              // which is a whole-buffer regression rather than the single-tile ones that missed it.
              const size_t vrpt   = 256 / CPW;                               // uint32 vecs per column per supertile
              const size_t vr     = ktile * WPK + w;                         // vec index within column n_log
              const size_t src_w  = (vr / vrpt) * (vrpt * N) + n_log * vrpt + (vr % vrpt);
              // destination: row r, element offset within row = t*(F*fold_tk) + f*fold_tk (+ w*16)
              // destination = PLANE-major: stride (kCon, (1, kCon*(N/F))) makes super-tile kb a separate plane of
              // (N/F) rows, so kb selects the plane and r indexes rows inside it. (Verified locally: 4096/4096 words
              // written, zero collisions, zero out-of-range.)
              const size_t dst_w = kb * (nrow * (size_t)W_ROW)
                                 + r * (size_t)W_ROW
                                 + (size_t)t * (F * WPK) + (size_t)f * WPK + w;
              dst[dst_w] = src[src_w];
            }
}

// BIT-GRANULAR fold placement. Writes the folded gmem buffer DIRECTLY from the row-major (n,k) codes -- it does
// not run the five relayout steps, because the placement it needs is not "five steps then a whole-word regroup".
//
// WHY A NEW PACKER AT ALL. nfold_regroup_gmem moves whole uint32s, so every word it produces holds ONE logical
// column. That is exactly what the mma wants while cols_per_word == 1, which holds for every configuration with a
// 32x32 warp tile. But over-delivery (delivery <= slots, slots = WN*TK/32) forbids int1 below TK=128 at WN=32, and
// the only escape is a wider warp N extent. At WN=64 the fragment asks for TWO columns inside each word, and a
// whole-word move cannot express that.
//
// WHERE THE FORMULA COMES FROM. Derived, not probed: fold_derivation/l10_placement.cu composes the swzl delivery
// (L2), the converter's emission order (L3), pi = partition_fragment_B(...).layout()^-1 (L8), and cute's
// partition_B (L4), then fits a GF(2)-affine form and verifies it over every position. The same chain regresses to
// 0/16384 against the REAL preprocess_weights_for_mixed_gemm + nfold_regroup_gmem on the box-verified
// int1 (32,128,128) config, which is what makes it trustworthy for a config the shipped offline has never seen.
//
// int1, TN=128, TK=64, WN=64  (F=4, Ng=32, 8 words per row, 32 bits per word):
//     n = row + 64*(wd>>2) + 32*((j>>3)&1)
//     k = 2*(wd&3) + 8*(j&7) + (j>>4)
// inverted, which is what this function walks:
//     row = n & 31
//     wd  = ((k >> 1) & 3) | (((n >> 6) & 1) << 2)
//     j   = ((k >> 3) & 7) | (((n >> 5) & 1) << 3) | ((k & 1) << 4)
// Compared with the TK=128 form, the single change is that j's bit 3 moves from k += 64 to n += 32: TK halving
// frees a k bit and F doubling needs an n bit. That migration IS the second column inside each word.
//
// `in_nk` is row-major (n, k) one code per bit, exactly as the caller packs `qT`. NOT the preprocess output.
inline void nfold_place_bits_int1_tk64(int8_t* out, const int8_t* in_nk, size_t N, size_t K,
                                       int fold_tn = 128, int fold_tk = 64)
{
    const int F = 4, Ng = fold_tn / F, W_ROW = 8, CPW = 32;
    assert(fold_tn == 128 && fold_tk == 64 && "derived for this shape only -- re-run l10_placement for others");
    assert(N % fold_tn == 0 && K % fold_tk == 0 && "shape must tile");
    const size_t nrow_total = N / F;
    std::fill(out, out + (N * K / 8), int8_t(0));
    for (size_t n = 0; n < N; ++n)
      for (size_t k = 0; k < K; ++k) {
        const size_t src_bit = n * K + k;
        if (!((in_nk[src_bit / 8] >> (src_bit % 8)) & 1)) continue;
        const size_t tile_n = n / fold_tn, kb = k / fold_tk;
        const int    nl = int(n % fold_tn), kl = int(k % fold_tk);
        const int    row = nl & (Ng - 1);
        const int    wd  = ((kl >> 1) & 3) | (((nl >> 6) & 1) << 2);
        const int    j   = ((kl >> 3) & 7) | (((nl >> 5) & 1) << 3) | ((kl & 1) << 4);
        const size_t dst_bit = ((kb * nrow_total + tile_n * Ng + row) * W_ROW + wd) * CPW + j;
        out[dst_bit / 8] |= int8_t(1 << (dst_bit % 8));
      }
}

// nfold_column_pairs_ppu USED TO LIVE HERE and has been deleted. It was dead code (nothing called it) whose
// comments carried a DISPROVEN derivation -- that the pipeline interleaves several N columns inside one vreg
// "at crumb level". fold_derivation/l7_groundtruth.cu measures the shipped pipeline as SINGLE-column per
// 32-bit word, and l13 confirms it across int1/int2/int4. Keeping a wrong explanation next to working code
// is not free: I independently re-derived the same wrong placement in l6 and believed it BECAUSE it matched
// this comment. The working placement is nfold_regroup_gmem (whole-uint32) and, for cols_per_word > 1,
// nfold_place_bits_int1_tk64 (bit-granular, generated by l10 from the verified chain).


} // namespace legacy
