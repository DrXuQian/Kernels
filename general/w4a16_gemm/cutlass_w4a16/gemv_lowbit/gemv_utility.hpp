// Building blocks for the GEMV main loop: global iterators, the slot->n-pair transpose that also carries the
// plane combine and the affine, the fma tree, and the CTA-wide reduction.
#pragma once

#include "gemv_converter.hpp"

namespace ppu_gemv {

// ---------------------------------------------------------------------------------------------------
// Byte-addressed strided iterator. Disabled instances hold no pointer and emit no code, so an optional
// input (act_scale, zeros, the high plane) costs nothing when absent.
template <bool Enable, typename Vec, int Continuous>
class ByteIterator {
 public:
  __device__ __forceinline__ ByteIterator(void const* p, int64_t off, int64_t step, int64_t stride)
      : base_(Enable ? (reinterpret_cast<uint8_t const*>(p) + off) : nullptr), step_(step), stride_(stride) {}

  __device__ __forceinline__ void load(void* dst, int iter, int ii = 0) const {
    if constexpr (Enable) {
      Vec const* src = reinterpret_cast<Vec const*>(base_ + int64_t(iter) * step_ + int64_t(ii) * stride_);
#pragma unroll
      for (int j = 0; j < Continuous; ++j) reinterpret_cast<Vec*>(dst)[j] = src[j];
    }
  }

 private:
  uint8_t const* base_;
  int64_t step_;
  int64_t stride_;
};

template <int N, typename T>
__device__ __forceinline__ void fill(void* tile, T v) {
#pragma unroll
  for (int i = 0; i < N; ++i) reinterpret_cast<T*>(tile)[i] = v;
}

// ---------------------------------------------------------------------------------------------------
// Slot-order converter output -> n-paired, plane-combined, dequantised half2 tile.
//
// WHY THE n-PAIRED DOMAIN IS THE RIGHT PLACE FOR ALL THE DEQUANT MATH. The mma below consumes weights as
// half2 over ADJACENT COLUMNS (one activation scalar broadcast against two columns), so the transpose from
// the converter's k-paired slots to n-paired registers has to happen regardless -- and it is free, being
// pure compile-time indexing. Doing the combine and the affine there instead of in the converter's own
// k-paired domain costs the same op count and dissolves the cross-plane pairing problem: the low plane pairs
// (e_p, e_{p+16/LoBits}) and the high plane pairs (e_p, e_{p+16/HiBits}), which never line up, but once the
// pairing axis is n each plane resolves its own k independently through its own mapper.
//
// Arithmetic, with raw = 2^Off + q from the converter and u = q_hi from the high plane:
//     q   = q_lo + 2^LoBits * q_hi
//     w   = q*s + z = (raw_lo + 2^LoBits*u)*s + (z - 2^Off*s)
// so the caller pre-folds (-2^Off - zero_point)*s into z'. Per (k, n-pair) that is one hfma2 for the combine
// (two-plane only) and one hfma2 for the affine -- nothing else.
template <typename Details, int Chunk, bool TwoPlane, typename LoCvt, typename HiCvt>
__device__ __forceinline__ void transpose_combine_affine(
    typename Details::AType2* wp,                  // [Chunk/2][StepK]
    typename Details::AType const* raw_lo,         // [Chunk][StepK] slot order (low plane)
    typename Details::AType const* raw_hi,         // [Chunk][StepK] slot order (high plane), unused if !TwoPlane
    typename Details::AType2 const* s2,            // [kSubGroups][pair_stride], pre-offset to this chunk
    typename Details::AType2 const* z2,            // ditto
    int pair_stride, int elems_per_subgroup) {
  using M  = MathWrapper<typename Details::ADetails>;
  using T  = typename Details::AType;
  using T2 = typename Details::AType2;
  constexpr int StepK = Details::kStepK;
  constexpr int Pairs = Chunk / 2;
  static_assert(Chunk % 2 == 0, "column chunk must be even: the mma pairs adjacent columns");

  T2 const hi_scale = M::to_vec2(M::from_float(float(1 << Details::kLoBits)));

#pragma unroll
  for (int cp = 0; cp < Pairs; ++cp) {
#pragma unroll
    for (int k = 0; k < StepK; ++k) {
      int const sl = LoCvt::mapper(k);
      T2 v = M::pack2(raw_lo[(2 * cp) * StepK + sl], raw_lo[(2 * cp + 1) * StepK + sl]);
      if constexpr (TwoPlane) {
        int const sh = HiCvt::mapper(k);
        T2 const u = M::pack2(raw_hi[(2 * cp) * StepK + sh], raw_hi[(2 * cp + 1) * StepK + sh]);
        v = M::fma2(u, hi_scale, v);
      }
      int const g = elems_per_subgroup >= StepK ? 0 : (k / elems_per_subgroup);
      wp[cp * StepK + k] = M::fma2(v, s2[g * pair_stride + cp], z2[g * pair_stride + cp]);
    }
  }
}

// ---------------------------------------------------------------------------------------------------
// acc[m][n-pair] += sum_k w[n-pair][k] * a[m][k]
//
// Accumulation is in the activation type, not fp32, and that is safe BECAUSE K is split Threads ways: each
// thread only ever sums k/Threads products into a given output (16 of them at k=2048, Threads=128), so the
// fp16 accumulation depth is 16, not 2048. The cross-thread sum in the epilogue is done in fp32.
template <typename Details, int CtaM, int Pairs>
__device__ __forceinline__ void mma_acc(typename Details::AType2* acc, int acc_pair_stride, int pair_base,
                                        typename Details::AType2 const* wp,
                                        typename Details::AType const* tile_a) {
  using M = MathWrapper<typename Details::ADetails>;
  constexpr int StepK = Details::kStepK;
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
#pragma unroll
    for (int cp = 0; cp < Pairs; ++cp) {
      auto a = acc[m * acc_pair_stride + pair_base + cp];
#pragma unroll
      for (int k = 0; k < StepK; ++k) {
        a = M::fma2(wp[cp * StepK + k], M::to_vec2(tile_a[m * StepK + k]), a);
      }
      acc[m * acc_pair_stride + pair_base + cp] = a;
    }
  }
}

// ---------------------------------------------------------------------------------------------------
// Full 32-lane reduction. Both supported layouts have no column interleave inside a warp (a k-tile of a
// single column is covered by TileSizeK/StepK consecutive lanes and every lane of the CTA works on the same
// column set), so every lane of the warp holds a partial sum of the SAME output and the reduction is the
// plain butterfly. Column-interleaved layouts would need the reference's partial butterfly instead.
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) val += __shfl_xor_sync(~0u, val, off);
  return val;
}

// out is [rows, n] row-major; `row_ok` masks the ragged tail of a grouped problem.
template <typename Details, int CtaM, int CtaN, bool EnableBias, bool ApplyAlphaInAdvance>
__device__ __forceinline__ void epilogue(typename Details::AType* out, int stride,
                                         typename Details::AType2 const* acc,
                                         typename Details::AType const* bias, float alpha,
                                         unsigned row_mask) {
  using M = MathWrapper<typename Details::ADetails>;
  using T = typename Details::AType;
  constexpr int WarpNum = Details::kWarpNum;
  constexpr int Pairs   = CtaN / 2;

  __shared__ float shmem[CtaM * CtaN * WarpNum];
  int const tid = threadIdx.x, warp_id = tid / Details::kWarpSize, lane = tid % Details::kWarpSize;

#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
#pragma unroll
    for (int cp = 0; cp < Pairs; ++cp) {
      float2 v;
      v.x = M::to_float(M::lo(acc[m * Pairs + cp]));
      v.y = M::to_float(M::hi(acc[m * Pairs + cp]));
      v.x = warp_reduce_sum(v.x);
      v.y = warp_reduce_sum(v.y);
      if (lane == 0) {
        shmem[warp_id * CtaM * CtaN + m * CtaN + cp * 2 + 0] = v.x;
        shmem[warp_id * CtaM * CtaN + m * CtaN + cp * 2 + 1] = v.y;
      }
    }
  }
  __syncthreads();

  for (int ii = tid; ii < CtaM * CtaN; ii += Details::kThreads) {
    int const m = ii / CtaN, nn = ii % CtaN;
    if (((row_mask >> m) & 1u) == 0u) continue;   // bitmask, not bool[CtaM]: a runtime index into a small
                                                 // array would spill it to local memory
    float val = 0.f;
#pragma unroll
    for (int w = 0; w < WarpNum; ++w) val += shmem[w * CtaM * CtaN + ii];
    float b = 0.f;
    if constexpr (EnableBias) b = M::to_float(bias[nn]);
    if constexpr (ApplyAlphaInAdvance) out[m * stride + nn] = M::from_float(val + b);
    else                              out[m * stride + nn] = M::from_float(alpha * val + b);
  }
}

}  // namespace ppu_gemv
