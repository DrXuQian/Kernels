// L79 -- HOW WIDE IS THE A gmem->register COPY? PPU_A_IN_REG measured 1.5-1.8x SLOWER than the swzl path, and the
// swzl atom moves 4096 bits in ONE instruction (InstNum=4 over a 128-element fragment). If cute picked a 1-element
// vector for the replacement copy, A's delivery costs ~128 loads per thread per k-tile and vectorising is an 8x
// lever; if it already picked 8, the instruction count is irreducible and the route is finished.
//
// Reported: max_common_vector for the (gmem partition, fragment) pair, plus the fragment size for scale.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
using namespace cute;

// The LAYOUT is what settles how many of these loads are redundant: with the gmem m-stride at 0, any fragment
// mode whose resulting stride is 0 addresses the SAME half for every index along it, so those loads are provably
// duplicates. Reported as types so the compiler prints the full shape:stride.
template <class TCGA_LAYOUT, class TCRA_LAYOUT, int MAX_COMMON_VECTOR, int FRAG_ELEMS> struct REPORT;

using TileShape      = Shape<_16,_128,_256>;
using ScaleTileShape = Shape<_128,_8>;
using WarpShape      = Shape<_16,_16,_256>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;
using TMma = typename Mainloop::TiledMma;

// One A k-tile as launch() hands it over: m-stride 0 (a_row_broadcast), k contiguous.
static auto make_gA() {
  return make_tensor(make_gmem_ptr(static_cast<cutlass::half_t*>(nullptr)),
                     make_shape(size<0>(TileShape{}), size<2>(TileShape{})),
                     make_stride(_0{}, _1{}));
}
static auto tCgA() { return TMma{}.get_thread_slice(0).partition_A(make_gA()); }
static auto tCrA() { return TMma{}.get_thread_slice(0).partition_fragment_A(make_gA()); }

static constexpr int MCV        = int(decltype(max_common_vector(tCgA(), tCrA())){});
static constexpr int FRAG_ELEMS = int(cute::size(decltype(tCrA()){}));
// one mma atom's slice, which is what the innermost loop copies
static auto tCgA_atom() { return tCgA()(cute::_, cute::_, cute::Int<0>{}); }
static auto tCrA_atom() { return tCrA()(cute::_, cute::_, cute::Int<0>{}); }
REPORT<decltype(cute::layout(tCgA_atom())), decltype(cute::layout(tCrA_atom())), MCV, FRAG_ELEMS> report;
