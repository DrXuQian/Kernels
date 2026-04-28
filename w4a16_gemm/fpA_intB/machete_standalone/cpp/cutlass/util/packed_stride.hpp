/*
 * Minimal compatibility shim for CUTLASS variants that reference
 * `cutlass/util/packed_stride.hpp` but don't ship it.
 *
 * TensorRT-LLM's SM90 fpA_intB launcher uses `cutlass::make_cute_packed_stride` to
 * materialize runtime strides from the CUTLASS/CuTe stride "tag" types.
 */

#pragma once

#include "cute/layout.hpp"

#include <cstdint>

namespace cutlass
{

// Stride tags for GEMM A/B/D/S in CUTLASS generally follow one of these patterns:
//   - (dynamic, 1, dynamic)  -> contiguous dim is the 2nd mode
//   - (1, dynamic, dynamic)  -> contiguous dim is the 1st mode
// where the 3rd mode is the batch stride (often unused for L=1).

template <class Stride0, class Stride2, class Shape>
CUTE_HOST_DEVICE constexpr auto make_cute_packed_stride(cute::Stride<Stride0, cute::Int<1>, Stride2>,
    Shape const& shape)
{
    int64_t const dim0 = static_cast<int64_t>(cute::get<0>(shape));
    int64_t const dim1 = static_cast<int64_t>(cute::get<1>(shape));
    int64_t const dim2 = static_cast<int64_t>(cute::get<2>(shape));

    Stride0 s0{};
    if constexpr (!cute::is_static<Stride0>::value)
    {
        s0 = static_cast<Stride0>(dim1);
    }

    Stride2 s2{};
    if constexpr (!cute::is_static<Stride2>::value)
    {
        s2 = static_cast<Stride2>((dim2 == 1) ? 0 : (dim0 * dim1));
    }

    return cute::make_stride(s0, cute::Int<1>{}, s2);
}

template <class Stride1, class Stride2, class Shape>
CUTE_HOST_DEVICE constexpr auto make_cute_packed_stride(cute::Stride<cute::Int<1>, Stride1, Stride2>,
    Shape const& shape)
{
    int64_t const dim0 = static_cast<int64_t>(cute::get<0>(shape));
    int64_t const dim1 = static_cast<int64_t>(cute::get<1>(shape));
    int64_t const dim2 = static_cast<int64_t>(cute::get<2>(shape));

    Stride1 s1{};
    if constexpr (!cute::is_static<Stride1>::value)
    {
        s1 = static_cast<Stride1>(dim0);
    }

    Stride2 s2{};
    if constexpr (!cute::is_static<Stride2>::value)
    {
        s2 = static_cast<Stride2>((dim2 == 1) ? 0 : (dim0 * dim1));
    }

    return cute::make_stride(cute::Int<1>{}, s1, s2);
}

} // namespace cutlass
