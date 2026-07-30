// Host dispatch. Structured as the reference's nested check_* chain so each runtime axis collapses to one
// template argument, with two deliberate differences:
//
//   * the group-size list is a MACRO, so a build that only ships gs=32 and gs=128 can drop the rest. Every
//     entry is a full kernel instantiation and this project has already paid for a sweep that instantiated
//     more than it needed.
//   * every refusal goes through gemv_refuse(), which bumps a counter. A harness that times a launch which
//     never happened will otherwise rank it fastest -- that has happened here twice.
#pragma once

#include "gemv_kernel.hpp"

namespace ppu_gemv {

// Which group sizes get instantiated. 16 is Q3_K's, 32 is the fine Q4_K/Q6_K one, 128 is GPTQ's, 0 is
// per-column. Narrow this in a build that does not need all of them.
#ifndef GEMV_GS_LIST
#define GEMV_GS_LIST(EMIT) EMIT(0) EMIT(16) EMIT(32) EMIT(64) EMIT(128)
#endif

// Rows per CTA. The activation tile is CtaM*StepK halfs in registers, which is what caps this.
#ifndef GEMV_CTAM_MAX
#define GEMV_CTAM_MAX 4
#endif

// Which quant ops get instantiated, for the same reason as GEMV_GS_LIST: every entry is a full set of kernels
// and a perf sweep that only ever calls two of the three should not pay for the third. Default is all three.
#ifndef GEMV_QUANT_LIST
#define GEMV_QUANT_LIST(EMIT, G)                                                                   \
  EMIT(QuantOp::PerColScaleOnly, G) EMIT(QuantOp::FinegrainedScaleOnly, G) EMIT(QuantOp::FinegrainedScaleZero, G)
#endif

// ---------------------------------------------------------------------------------------------------
// Compile-time legality of one (GS, QuantOp, StepK) triple. Checked with if constexpr inside a TEMPLATE so
// the illegal combinations are never instantiated -- an if constexpr in a plain function does not suppress
// instantiation, which is a trap this codebase has hit before.
template <typename Details, int GS, QuantOp QOp>
constexpr bool gemv_combo_ok() {
  if constexpr (QOp == QuantOp::PerColScaleOnly) return GS == 0;
  else if constexpr (GS == 0) return false;
  else return gs_step_ok<GS, Details::kStepK, Details::kCtaK>();
}

template <typename Details, int CtaM, int CtaN, int Chunk, int GS, QuantOp QOp, bool EnableBias, bool Grouped>
void gemv_exec(Params const& p, KernelArgs const& args, int grid_m, gemv_stream_t s) {
  dim3 const grid(grid_m, p.n / CtaN, Grouped ? p.num_experts : 1);
  dim3 const block(Details::kThreads);
  gemv_kernel<Details, CtaM, CtaN, Chunk, GS, QOp, /*EnableActScale=*/false, EnableBias,
              /*ApplyAlphaInAdvance=*/false, Grouped><<<grid, block, 0, s>>>(args);
}

// ---- CtaM ----
template <typename Details, int CtaN, int Chunk, int GS, QuantOp QOp, bool EnableBias, bool Grouped>
bool gemv_dispatch_ctam(Params const& p, KernelArgs const& args, int rows_max, gemv_stream_t s) {
#define GEMV_TRY_CTAM(CM)                                                                        \
  if (rows_max <= (CM) || (CM) == GEMV_CTAM_MAX) {                                                 \
    constexpr int _cm = (CM);                                                                      \
    int const gm = (rows_max + _cm - 1) / _cm;                                                     \
    gemv_exec<Details, _cm, CtaN, Chunk, GS, QOp, EnableBias, Grouped>(p, args, gm, s);             \
    return true;                                                                                   \
  }
  GEMV_TRY_CTAM(1)
#if GEMV_CTAM_MAX >= 2
  GEMV_TRY_CTAM(2)
#endif
#if GEMV_CTAM_MAX >= 3
  GEMV_TRY_CTAM(3)
#endif
#if GEMV_CTAM_MAX >= 4
  GEMV_TRY_CTAM(4)
#endif
#undef GEMV_TRY_CTAM
  return false;
}

// ---- bias ----
template <typename Details, int CtaN, int Chunk, int GS, QuantOp QOp, bool Grouped>
bool gemv_dispatch_bias(Params const& p, KernelArgs const& args, int rows_max, gemv_stream_t s) {
  if (p.bias) return gemv_dispatch_ctam<Details, CtaN, Chunk, GS, QOp, true, Grouped>(p, args, rows_max, s);
  return gemv_dispatch_ctam<Details, CtaN, Chunk, GS, QOp, false, Grouped>(p, args, rows_max, s);
}

// ---- grouped ----
template <typename Details, int CtaN, int Chunk, int GS, QuantOp QOp>
bool gemv_dispatch_grouped(Params const& p, KernelArgs const& args, int rows_max, gemv_stream_t s) {
  if (p.num_experts > 0)
    return gemv_dispatch_bias<Details, CtaN, Chunk, GS, QOp, true>(p, args, rows_max, s);
  return gemv_dispatch_bias<Details, CtaN, Chunk, GS, QOp, false>(p, args, rows_max, s);
}

// ---- (quant op, group size) ----
template <typename Details, int CtaN, int Chunk>
bool gemv_dispatch_quant(Params const& p, KernelArgs const& args, int rows_max, gemv_stream_t s) {
#define GEMV_TRY_QGS(QO, G)                                                                        \
  if (p.quant == (QO) && p.groupsize == (G)) {                                                     \
    if constexpr (gemv_combo_ok<Details, (G), (QO)>())                                             \
      return gemv_dispatch_grouped<Details, CtaN, Chunk, (G), (QO)>(p, args, rows_max, s);         \
    else { gemv_refuse("group size illegal for this StepK"); return false; }                       \
  }
#define GEMV_EMIT_GS(G) GEMV_QUANT_LIST(GEMV_TRY_QGS, G)
  GEMV_GS_LIST(GEMV_EMIT_GS)
#undef GEMV_EMIT_GS
#undef GEMV_TRY_QGS
  gemv_refuse("group size not instantiated (see GEMV_GS_LIST)");
  return false;
}

// ---------------------------------------------------------------------------------------------------
// Entry point for one compiled shape. Returns false (and bumps gemv_fail_count) without launching if the
// problem does not fit the instantiation.
template <typename Details, int CtaN, int Chunk>
bool launch_gemv(Params const& p, gemv_stream_t s) {
  // The buffer's own record wins over anything the caller claims in Params.
  if (p.record) {
    char const* why = "";
    if (!wfmt_matches<Details>(*reinterpret_cast<WeightFormatRecord const*>(p.record),
                              p.n, p.k, p.groupsize, p.quant, &why)) {
      char msg[128];
      std::snprintf(msg, sizeof(msg), "weight format record disagrees: %s", why);
      gemv_refuse(msg);
      return false;
    }
  }
  if (p.format != Details::kFormat) { gemv_refuse("format mismatch"); return false; }
  if (p.layout != Details::kLayout) { gemv_refuse("layout mismatch"); return false; }
  if (p.is_bf16 != Details::ADetails::kIsBF16) { gemv_refuse("A type mismatch"); return false; }
  if (p.act_scale) { gemv_refuse("act_scale is not instantiated"); return false; }
  // n % CtaN also carries the ALIGNMENT the vectorised scale load needs: a group row starts at row*n + n0 with
  // n0 a multiple of CtaN, so row*n being a multiple of CtaN makes the whole index CtaN-aligned.
  if (p.n % CtaN) { gemv_refuse("n must be a multiple of CtaN"); return false; }
  if (p.k % Details::kCtaK) {
    // CtaK = StepK*Threads. GGUF guarantees k % 256 == 0, so this only bites for a deliberately odd shape.
    gemv_refuse("k must be a multiple of StepK*Threads");
    return false;
  }
  if (is_two_plane(Details::kFormat) && !p.weight_hi) { gemv_refuse("two-plane format needs weight_hi"); return false; }
  if (has_zero(p.quant) && !p.zeros) { gemv_refuse("ScaleZero needs zeros"); return false; }

  int rows_max = 0;
  if (p.num_experts > 0) {
    if (!p.row_offsets) { gemv_refuse("grouped needs row_offsets"); return false; }
    rows_max = p.max_rows;
  } else {
    rows_max = p.m;
  }
  if (rows_max <= 0) { gemv_refuse("no rows"); return false; }
  if (rows_max > GEMV_CTAM_MAX * 4096) { gemv_refuse("rows out of range"); return false; }

  KernelArgs args{};
  args.act = p.act;  args.act_scale = p.act_scale;
  args.w_lo = p.weight;  args.w_hi = p.weight_hi;
  args.scales = p.scales;  args.zeros = p.zeros;  args.bias = p.bias;
  args.out = p.out;  args.alpha = p.alpha;
  args.n = p.n;  args.k = p.k;  args.rows = p.m;
  args.row_offsets = p.row_offsets;
  args.w_lo_stride_e = p.w_bytes_per_expert;
  args.w_hi_stride_e = p.w_hi_bytes_per_expert;
  args.scale_stride_e = p.scale_elems_per_expert;
  args.lo_s = Details::LoLayout::strides(p.n, p.k);
  args.hi_s = Details::HiLayout::strides(p.n, p.k);

  return gemv_dispatch_quant<Details, CtaN, Chunk>(p, args, rows_max, s);
}

}  // namespace ppu_gemv
