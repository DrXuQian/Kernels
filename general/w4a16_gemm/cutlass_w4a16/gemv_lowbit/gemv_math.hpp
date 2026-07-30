// Packed-pair arithmetic, one specialisation per activation type. Kept in its own header because both the
// converter and the kernel body need it and neither should include the other.
#pragma once

#include "gemv_details.hpp"

namespace ppu_gemv {

template <typename ADetails>
struct MathWrapper;

template <>
struct MathWrapper<FP16DetailsA> {
  using Type  = half;
  using Type2 = half2;

  __device__ __forceinline__ static Type2 to_vec2(Type const& v) { return __half2half2(v); }
  // MEMBER-WISE, not __halves2half2: that intrinsic appears nowhere in actlize, so it cannot be assumed to
  // exist under hgcc. half2's .x/.y are relied on by the tree itself (cutlass/float8.h:441), so this is the
  // construction with evidence behind it.
  __device__ __forceinline__ static Type2 pack2(Type const& a, Type const& b) {
    Type2 r; r.x = a; r.y = b; return r;
  }
  __device__ __forceinline__ static Type lo(Type2 const& v) { return v.x; }
  __device__ __forceinline__ static Type hi(Type2 const& v) { return v.y; }
  __device__ __forceinline__ static Type2 fma2(Type2 const& a, Type2 const& b, Type2 const& c) { return __hfma2(a, b, c); }
  __device__ __forceinline__ static Type2 mul2(Type2 const& a, Type2 const& b) { return __hmul2(a, b); }
  __device__ __forceinline__ static Type2 sub2(Type2 const& a, Type2 const& b) { return __hsub2(a, b); }
  __device__ __forceinline__ static Type2 add2(Type2 const& a, Type2 const& b) { return __hadd2(a, b); }
  __device__ __forceinline__ static Type from_float(float v) { return __float2half(v); }
  __device__ __forceinline__ static float to_float(Type v) { return __half2float(v); }
};

#if defined(ENABLE_BF16)
template <>
struct MathWrapper<BF16DetailsA> {
  using Type  = __nv_bfloat16;
  using Type2 = __nv_bfloat162;

  __device__ __forceinline__ static Type2 to_vec2(Type const& v) { return __bfloat162bfloat162(v); }
  // Same reasoning as fp16. NOTE: the bf16 path is NOT compiled in any current build (ENABLE_BF16 is never
  // defined), so the PPU spelling of the bf16 vector type is UNVERIFIED -- treat enabling it as a change that
  // needs its own build, not as a working feature.
  __device__ __forceinline__ static Type2 pack2(Type const& a, Type const& b) {
    Type2 r; r.x = a; r.y = b; return r;
  }
  __device__ __forceinline__ static Type lo(Type2 const& v) { return v.x; }
  __device__ __forceinline__ static Type hi(Type2 const& v) { return v.y; }
  __device__ __forceinline__ static Type2 fma2(Type2 const& a, Type2 const& b, Type2 const& c) { return __hfma2(a, b, c); }
  __device__ __forceinline__ static Type2 mul2(Type2 const& a, Type2 const& b) { return __hmul2(a, b); }
  __device__ __forceinline__ static Type2 sub2(Type2 const& a, Type2 const& b) { return __hsub2(a, b); }
  __device__ __forceinline__ static Type2 add2(Type2 const& a, Type2 const& b) { return __hadd2(a, b); }
  __device__ __forceinline__ static Type from_float(float v) { return __float2bfloat16(v); }
  __device__ __forceinline__ static float to_float(Type v) { return __bfloat162float(v); }
};
#endif

}  // namespace ppu_gemv
