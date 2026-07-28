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
