// Low-bit weight-only GEMV for PPU -- shared declarations.
//
// STRUCTURE mirrors TRT-LLM's weightOnlyBatchedGemv (common/details/converter/utility/kernel/dispatcher/launcher)
// so the two can be read side by side. What is GENERALISED here relative to that reference:
//
//   * weight width 8/4/2/1 bits, plus the three GGUF two-plane decompositions
//     (Q6 = int4 low + int2 high, Q5 = int4 low + int1 high, Q3 = int2 low + int1 high).
//   * grouped / MoE: an expert dimension on grid.z with ragged per-expert row counts, so the SAME kernel
//     serves dense decode and MoE decode. The reference is dense only.
//   * two weight layouts (see gemv_details.hpp): the GGUF-native [N][K] and the tile-K interleave
//     [K/TS][N][TS] that the prefill AIU collective already reads. Native needs NO offline pass at all.
//
// WHY A GEMV EXISTS AT ALL, given a working mixed-input tensor-core GEMM. At decode the grouped GEMM's
// launch is 512 CTAs x 64 threads = 1024 warps = 14.2 warps/CU on a 72-CU part, and acu measured 13.65
// achieved: EVERY warp of the launch is resident at once, so there is no second wave and latency has
// nothing to hide behind. Occupancy there is bounded by the amount of WORK, not by smem or registers --
// which is why every tile/warp/stage knob moved it less than 8%. This kernel's grid is
// (rows/CtaM) x (n/(CtaN*Interleave)) x experts = ~2048 CTAs x 128 threads = ~7 waves, and its main loop
// touches no shared memory at all, so the resource ceiling stops being the question.
//
// K IS SPLIT 'Threads' WAYS BY CONSTRUCTION here (thread t owns the K-strided slice starting at t*StepK and
// the warp reduces at the end), so this subsumes split-K for the m<=CtaM_max band without a partial buffer,
// a semaphore or a second kernel.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

// PLATFORM HEADERS. The intrinsic SPELLINGS are the same on both sides -- half2, __hfma2, __hsub2,
// __half2half2, __shfl_xor_sync, __float2half all exist under hgcc -- but they arrive from a different
// header. Getting this wrong is invisible to a local syntax check, because nvcc has cuda_fp16.h whether or
// not __HGGCCC__ is defined; only the box says so. See fold_derivation/ppu_portability_check.sh.
#if defined(__HGGCCC__)
#include <hggc_fp16.h>
#if defined(ENABLE_BF16)
#include <hggc_bf16.h>
#endif
#else
#include <cuda_fp16.h>
#if defined(ENABLE_BF16)
#include <cuda_bf16.h>
#endif
#endif

// The stream type differs, and it comes from the RUNTIME header, not the fp16 one. Every other file in this
// tree gets hggcStream_t transitively through a cutlass header; this one is standalone, so it has to ask.
// (Caught by syntax_check.sh only because the new baseline line was READ rather than accepted: a bare
// `error: identifier "hggcStream_t" is undefined` is not fatal, so nothing stops it becoming "known noise".)
#if defined(__HGGCCC__)
#include <hggc_runtime.h>
using gemv_stream_t = hggcStream_t;
#else
#include <cuda_runtime.h>
using gemv_stream_t = cudaStream_t;
#endif

namespace ppu_gemv {

// ---------------------------------------------------------------------------------------------------
// Weight formats.
//
// A format is (low-plane bits, high-plane bits). The high plane is 0 for the single-plane formats. The
// two-plane entries are the GGUF bit-plane decompositions: the quantised code is reconstructed as
//     q = q_lo + (q_hi << LoBits)
// which in the dequantised domain is one extra hfma2 per pair (see gemv_utility.hpp::combine_planes).
// 3-bit does not divide a byte, which is exactly why Q3/Q5/Q6 are stored as two power-of-two planes.
enum class WFormat : int {
  Int8 = 0,   // 8 bits, one plane
  Int4,       // 4 bits, one plane   (GPTQ / AWQ / Q4_K)
  Int2,       // 2 bits, one plane   (Q2_K)
  Int1,       // 1 bit,  one plane
  Q6_42,      // 6 bits = int4 low + int2 high   (Q6_K)
  Q5_41,      // 5 bits = int4 low + int1 high   (Q5_K)
  Q3_21,      // 3 bits = int2 low + int1 high   (Q3_K)
};

#if defined(__CUDACC__)
#define PPU_GEMV_HD __host__ __device__
#else
#define PPU_GEMV_HD
#endif

PPU_GEMV_HD constexpr int lo_bits_of(WFormat f) {
  return f == WFormat::Int8  ? 8
       : f == WFormat::Int4  ? 4
       : f == WFormat::Int2  ? 2
       : f == WFormat::Int1  ? 1
       : f == WFormat::Q6_42 ? 4
       : f == WFormat::Q5_41 ? 4
       :                       2;   // Q3_21
}
PPU_GEMV_HD constexpr int hi_bits_of(WFormat f) {
  return f == WFormat::Q6_42 ? 2
       : f == WFormat::Q5_41 ? 1
       : f == WFormat::Q3_21 ? 1
       :                       0;
}
PPU_GEMV_HD constexpr int total_bits_of(WFormat f) { return lo_bits_of(f) + hi_bits_of(f); }
PPU_GEMV_HD constexpr bool is_two_plane(WFormat f) { return hi_bits_of(f) != 0; }

inline const char* name_of(WFormat f) {
  switch (f) {
    case WFormat::Int8:  return "int8";
    case WFormat::Int4:  return "int4";
    case WFormat::Int2:  return "int2";
    case WFormat::Int1:  return "int1";
    case WFormat::Q6_42: return "q6(4+2)";
    case WFormat::Q5_41: return "q5(4+1)";
    case WFormat::Q3_21: return "q3(2+1)";
  }
  return "?";
}

// ---------------------------------------------------------------------------------------------------
// How the per-group affine is supplied.
//
// The CONVERTER always produces the raw unsigned code as a float type; every signed convention is folded
// into (scale, zero) by the caller. GPTQ's symmetric int4 (value = q - 8) is therefore ScaleOnly with the
// -8 pushed into a zero term, and Q4_K's (d, dmin) affine is ScaleZero directly -- the same collapse the
// AIU collective makes ("int4's -8 folds into zero").
enum class QuantOp : int {
  PerColScaleOnly = 0,     // scales [1, n],        no zero
  FinegrainedScaleOnly,    // scales [k/gs, n],     no zero
  FinegrainedScaleZero,    // scales + zeros [k/gs, n]
};

PPU_GEMV_HD constexpr bool is_finegrained(QuantOp q) { return q != QuantOp::PerColScaleOnly; }
PPU_GEMV_HD constexpr bool has_zero(QuantOp q) { return q == QuantOp::FinegrainedScaleZero; }

inline const char* name_of(QuantOp q) {
  switch (q) {
    case QuantOp::PerColScaleOnly:      return "per-col";
    case QuantOp::FinegrainedScaleOnly: return "fg-scale";
    case QuantOp::FinegrainedScaleZero: return "fg-scale+zero";
  }
  return "?";
}

// ---------------------------------------------------------------------------------------------------
// Weight memory layouts.
enum class WLayout : int {
  Native = 0,   // [N][K] column-major, K contiguous. GGUF/GPTQ native -- zero offline pass.
  TileK,        // [K/TS][N][TS]. Matches cutlass::layout::ColumnMajorInterleaved<TS> with TS = TileSizeK,
                // i.e. the buffer the prefill AIU collective already reads for int4.
  FoldCube,     // The sub-byte offline layout: cube-tiled with a within-cube permutation that compensates
                // for the AIU converter's fixed emission order (xplane_offline.hpp). Parameterised by the
                // PREFILL tile shape (tm,tn,tk,wm,wn,F), and NOT affine in k at StepK granularity -- a
                // contiguous logical-k run of one column scatters inside the cube. Describable here, and
                // deliberately refused by the affine GEMV path rather than read wrongly.
};

inline const char* name_of(WLayout l) {
  return l == WLayout::Native ? "native" : l == WLayout::TileK ? "tileK" : "foldcube";
}

// ---------------------------------------------------------------------------------------------------
// Runtime arguments. num_experts == 0 selects the dense path; otherwise the grouped/MoE path.
//
// GROUPED CONVENTION matches the existing MoE grouped harness: activations and outputs are GATHERED, i.e.
// expert e owns rows [row_offsets[e], row_offsets[e+1]) of a single [total_rows, k] A and [total_rows, n] D,
// and the weights are L-strided by whole matrices. That means a caller already producing data for
// moe_grouped_ppu::launch can call this with the same pointers.
struct Params {
  void const* act        = nullptr;  // [total_rows, k] row-major, A element type
  void const* act_scale  = nullptr;  // [k] or null (per-K activation scale; AWQ-style)
  void const* weight     = nullptr;  // packed low plane
  void const* weight_hi  = nullptr;  // packed high plane, or null for single-plane formats
  void const* scales     = nullptr;  // [k/gs, n] row-major, or [1, n] when groupsize == 0
  void const* zeros      = nullptr;  // same shape as scales, or null
  void const* bias       = nullptr;  // [n] or null
  void*       out        = nullptr;  // [total_rows, n] row-major
  float       alpha      = 1.f;
  bool        apply_alpha_in_advance = false;

  int m = 0;              // dense: rows. grouped: ignored (use row_offsets).
  int n = 0;
  int k = 0;
  int groupsize = 0;      // 0 => per-column scale

  WFormat format  = WFormat::Int4;
  QuantOp quant   = QuantOp::FinegrainedScaleOnly;
  WLayout layout  = WLayout::Native;
  bool    is_bf16 = false;

  // ---- grouped / MoE (num_experts == 0 => dense) ----
  int        num_experts = 0;
  int const* row_offsets = nullptr;  // device [L+1] cumulative gathered row starts
  int        max_rows    = 0;        // host-side max over experts of (row_offsets[e+1]-row_offsets[e])
  int64_t    w_bytes_per_expert   = 0;  // low-plane bytes between consecutive experts
  int64_t    w_hi_bytes_per_expert = 0; // high-plane bytes between consecutive experts
  int64_t    scale_elems_per_expert = 0; // scale (and zero) elements between consecutive experts

  // Optional format record for the weight buffer (gemv_wformat.hpp). When present the launcher REFUSES a
  // buffer whose recorded layout is not the one this instantiation reads, instead of computing wrong
  // numbers from bytes it misread. Null keeps the old behaviour: caller asserts the format by construction.
  void const* record = nullptr;   // WeightFormatRecord const*

  Params() = default;
};

// A launch that did not happen must be visible to the caller: a harness that times a refused launch
// measures an empty call and will happily rank it fastest. Same net as moe_grouped_ppu::moeg_fail_count().
inline int& gemv_fail_count() { static int c = 0; return c; }

inline void gemv_refuse(const char* why) {
  ++gemv_fail_count();
  std::printf("[gemv] launch refused: %s\n", why);
}

}  // namespace ppu_gemv
