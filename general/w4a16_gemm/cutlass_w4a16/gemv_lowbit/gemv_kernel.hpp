// The GEMV kernel itself, dense and grouped in one instantiation axis.
#pragma once

#include "gemv_utility.hpp"

namespace ppu_gemv {

// Device-side arguments. Params (gemv_common.hpp) is the host-facing form; this is what crosses to the GPU.
struct KernelArgs {
  void const* act;
  void const* act_scale;
  void const* w_lo;
  void const* w_hi;
  void const* scales;
  void const* zeros;
  void const* bias;
  void*       out;
  float       alpha;
  int         n;
  int         k;
  int         rows;            // dense row count (grouped: ignored)
  int const*  row_offsets;     // grouped: [L+1]
  int64_t     w_lo_stride_e;   // bytes
  int64_t     w_hi_stride_e;   // bytes
  int64_t     scale_stride_e;  // elements
};

// A group size is legal when a thread's StepK run never straddles a group boundary and a CTA step is a
// whole number of groups. Expressed with if constexpr so the GS == 0 case never parses `% GS`.
template <int GS, int StepK, int CtaK>
PPU_GEMV_HD constexpr bool gs_step_ok() {
  if constexpr (GS == 0) return true;
  else return (GS < StepK ? (StepK % GS == 0) : (GS % StepK == 0)) && (CtaK % GS == 0);
}

// grid = (ceil(rows/CtaM), n/CtaN, experts)   block = Details::kThreads
//
// K IS SPLIT kThreads WAYS: thread t owns the slice starting at t*StepK and striding by CtaK, and the
// epilogue reduces across the CTA. There is no partial buffer, no semaphore and no second kernel -- which is
// why this covers the decode band without the grouped tensor-core path's split-K machinery.
template <typename Details, int CtaM, int CtaN, int Chunk, int GS, QuantOp QOp,
          bool EnableActScale, bool EnableBias, bool ApplyAlphaInAdvance, bool Grouped>
__global__ void gemv_kernel(KernelArgs args) {
  using ADetails = typename Details::ADetails;
  using M  = MathWrapper<ADetails>;
  using T  = typename Details::AType;
  using T2 = typename Details::AType2;

  constexpr int StepK  = Details::kStepK;
  constexpr int CtaK   = Details::kCtaK;
  constexpr int Pairs  = CtaN / 2;
  constexpr int ChunkP = Chunk / 2;
  constexpr bool TwoPlane = Details::kTwoPlane;
  constexpr bool HasZero  = has_zero(QOp);

  static_assert(CtaN % Chunk == 0, "CtaN must be a whole number of column chunks");
  static_assert(Chunk % 2 == 0 && Chunk >= 2, "column chunk must be even");
  // gs must either sit inside one thread-step or tile it exactly, so a step never straddles a group. The
  // check lives in a template with if constexpr rather than in the && chain below: a short-circuited
  // `StepK % GS` with GS == 0 still gets diagnosed as a division by zero at parse time.
  static_assert(gs_step_ok<GS, StepK, CtaK>(), "group size incompatible with StepK / CtaK");

  constexpr int kSub = (GS == 0) ? 1 : (GS < StepK ? StepK / GS : 1);
  constexpr int kEPS = (GS == 0) ? StepK : (GS < StepK ? GS : StepK);   // elements per sub-group

  // BOTH planes subtract the magic offset in the converter -- see gemv_converter.hpp for why folding it
  // into the zero instead loses 2-5% to fp16 cancellation.
  using LoCvt = RawConverter<ADetails, Details::kLoBits, true>;
  // Guard on the ARGUMENT: RawConverter<_,0,_> would trip its own width assert even unused.
  using HiCvt = RawConverter<ADetails, (Details::kHiBits ? Details::kHiBits : 1), true>;

  using LoAcc = typename Details::LoAccess;
  using HiAcc = typename Details::HiAccess;
  using AAcc  = typename Details::AAccess;
  using LoLay = typename Details::LoLayout;
  using HiLay = typename Details::HiLayout;

  int const tid = threadIdx.x;
  int const e   = Grouped ? int(blockIdx.z) : 0;

  int row_begin = 0, rows = args.rows;
  if constexpr (Grouped) {
    row_begin = args.row_offsets[e];
    rows      = args.row_offsets[e + 1] - row_begin;
  }
  int const offset_m = int(blockIdx.x) * CtaM;
  if (offset_m >= rows) return;   // uniform across the CTA, so returning before the epilogue's barrier is safe

  int const n0 = int(blockIdx.y) * CtaN;
  int const n  = args.n, k = args.k;

  uint8_t const* w_lo = reinterpret_cast<uint8_t const*>(args.w_lo)
                      + (Grouped ? int64_t(e) * args.w_lo_stride_e : 0);
  uint8_t const* w_hi = reinterpret_cast<uint8_t const*>(args.w_hi)
                      + (Grouped && TwoPlane ? int64_t(e) * args.w_hi_stride_e : 0);
  T const* scales = reinterpret_cast<T const*>(args.scales)
                  + (Grouped ? int64_t(e) * args.scale_stride_e : 0);
  T const* zeros  = reinterpret_cast<T const*>(args.zeros)
                  + (Grouped && HasZero ? int64_t(e) * args.scale_stride_e : 0);

  // Rows past this expert's end read row 0 instead of going out of bounds, and are masked at store time.
  T const* a_rows[CtaM];
  unsigned row_mask = 0;
#pragma unroll
  for (int i = 0; i < CtaM; ++i) {
    int const r  = offset_m + i;
    bool const ok = r < rows;
    row_mask |= unsigned(ok) << i;
    a_rows[i] = reinterpret_cast<T const*>(args.act) + int64_t(row_begin + (ok ? r : 0)) * k + tid * StepK;
  }
  T* out_ptr = reinterpret_cast<T*>(args.out) + int64_t(row_begin + offset_m) * n + n0;
  T const* bias = reinterpret_cast<T const*>(args.bias) + n0;
  T const* act_scale = reinterpret_cast<T const*>(args.act_scale) + tid * StepK;

  ByteIterator<true, typename LoAcc::Vec, LoAcc::kCount> it_lo(
      w_lo, LoLay::base(n0, tid, n, k), LoLay::step(n, k), LoLay::stride(n, k));
  ByteIterator<TwoPlane, typename HiAcc::Vec, HiAcc::kCount> it_hi(
      w_hi, HiLay::base(n0, tid, n, k), HiLay::step(n, k), HiLay::stride(n, k));

  T2 acc[CtaM * Pairs];
  fill<CtaM * Pairs>(acc, M::to_vec2(M::from_float(0.f)));

  int const iters = k / CtaK;

  for (int iter = 0; iter < iters; ++iter) {
    // ---- scales / zeros for this step ----
    T2 s2[kSub * Pairs], z2[kSub * Pairs];
    {
      int const krow = (GS == 0) ? 0 : (iter * CtaK + tid * StepK) / GS;
#pragma unroll
      for (int g = 0; g < kSub; ++g) {
        int64_t const row_off = (GS == 0) ? 0 : int64_t(krow + g) * n;
#pragma unroll
        for (int cp = 0; cp < Pairs; ++cp) {
          T const sa = scales[row_off + n0 + 2 * cp + 0];
          T const sb = scales[row_off + n0 + 2 * cp + 1];
          T2 const sv = M::pack2(sa, sb);
          T2 zv = M::to_vec2(M::from_float(0.f));
          if constexpr (HasZero) {
            zv = M::pack2(zeros[row_off + n0 + 2 * cp + 0], zeros[row_off + n0 + 2 * cp + 1]);
          }
          s2[g * Pairs + cp] = sv;
          z2[g * Pairs + cp] = zv;   // the converter already removed the magic offset, in the integer domain
        }
      }
    }

    // ---- activations ----
    T tile_a[CtaM * StepK];
#pragma unroll
    for (int m = 0; m < CtaM; ++m) {
      typename AAcc::Vec const* src =
          reinterpret_cast<typename AAcc::Vec const*>(a_rows[m] + int64_t(iter) * CtaK);
#pragma unroll
      for (int j = 0; j < AAcc::kCount; ++j)
        reinterpret_cast<typename AAcc::Vec*>(tile_a + m * StepK)[j] = src[j];
    }
    if constexpr (EnableActScale) {
      T vas[StepK];
      typename AAcc::Vec const* src =
          reinterpret_cast<typename AAcc::Vec const*>(act_scale + int64_t(iter) * CtaK);
#pragma unroll
      for (int j = 0; j < AAcc::kCount; ++j) reinterpret_cast<typename AAcc::Vec*>(vas)[j] = src[j];
#pragma unroll
      for (int m = 0; m < CtaM; ++m)
#pragma unroll
        for (int j = 0; j < StepK / 2; ++j)
          reinterpret_cast<T2*>(tile_a + m * StepK)[j] =
              M::mul2(reinterpret_cast<T2*>(tile_a + m * StepK)[j], reinterpret_cast<T2*>(vas)[j]);
    }

    // ---- ALL quantised weight loads first (maximum memory-level parallelism), then convert in column
    //      chunks (minimum registers). Issuing the loads up front costs CtaN*StepK*bits/8 bytes of
    //      registers, which is small; holding CtaN dequantised columns at once is what is not.
    typename LoAcc::Vec q_lo[CtaN * LoAcc::kCount];
    typename HiAcc::Vec q_hi[TwoPlane ? CtaN * HiAcc::kCount : 1];
#pragma unroll
    for (int c = 0; c < CtaN; ++c) it_lo.load(q_lo + c * LoAcc::kCount, iter, c);
    if constexpr (TwoPlane) {
#pragma unroll
      for (int c = 0; c < CtaN; ++c) it_hi.load(q_hi + c * HiAcc::kCount, iter, c);
    }

#pragma unroll
    for (int nc = 0; nc < CtaN / Chunk; ++nc) {
      T raw_lo[Chunk * StepK];
      T raw_hi[TwoPlane ? Chunk * StepK : 1];
#pragma unroll
      for (int c = 0; c < Chunk; ++c) {
        LoCvt::template convert<StepK>(q_lo + (nc * Chunk + c) * LoAcc::kCount, raw_lo + c * StepK);
        if constexpr (TwoPlane)
          HiCvt::template convert<StepK>(q_hi + (nc * Chunk + c) * HiAcc::kCount, raw_hi + c * StepK);
      }
      T2 wp[ChunkP * StepK];
      transpose_combine_affine<Details, Chunk, TwoPlane, LoCvt, HiCvt>(
          wp, raw_lo, raw_hi, s2 + nc * ChunkP, z2 + nc * ChunkP, Pairs, kEPS);
      mma_acc<Details, CtaM, ChunkP>(acc, Pairs, nc * ChunkP, wp, tile_a);
    }
  }

  epilogue<Details, CtaM, CtaN, EnableBias, ApplyAlphaInAdvance>(
      out_ptr, n, acc, bias, args.alpha, row_mask);
}

}  // namespace ppu_gemv
