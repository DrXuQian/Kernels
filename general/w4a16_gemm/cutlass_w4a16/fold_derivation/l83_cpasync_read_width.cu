// L83 -- WHY IS THE cp.async PATH 1.4x SLOWER? A's smem->reg read is now make_tiled_copy_A(DefaultCopy, mma) over a
// STRIDE-0 sA. Reported: the vector width cute can use for that pair, the per-atom element count, and therefore the
// loads per mma to compare against the swzl path's 0.26. The stride-0 M mode makes slots that differ in m share an
// address, which is what capped the gmem variant at 2.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
using namespace cute;

template <class SRC_ATOM_LAYOUT, class DST_ATOM_LAYOUT, int MCV, int FRAG> struct REPORT;

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
using SmemLayoutA = typename Mainloop::SmemLayoutA;

static auto sA() { return make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutA{}); }
static auto tiled() {
  return make_tiled_copy_A(Copy_Atom<DefaultCopy, cutlass::half_t>{}, TMma{});
}
static auto tCsA() { return tiled().get_thread_slice(0).partition_S(sA()); }
static auto tCrA() {
  Tensor f = make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)),
                         tile_to_shape(typename Mainloop::InternalSmemLayoutAtomA{},
                                       make_shape(size<0>(TileShape{}), size<2>(TileShape{}), _2{})));
  return TMma{}.get_thread_slice(0).partition_fragment_A(f(_,_,0));
}
static auto tCrA_view() { return tiled().get_thread_slice(0).retile_D(decltype(tCrA()){}); }

// one mma atom's slice on each side
static auto src_atom() { return tCsA()(_,_,Int<0>{},Int<0>{}); }
static auto dst_atom() { return tCrA_view()(_,_,Int<0>{}); }

static constexpr int MCV          = int(decltype(max_common_vector(src_atom(), dst_atom())){});
static constexpr int PER_ATOM_SRC = int(cute::size(decltype(src_atom()){}));
static constexpr int PER_ATOM_DST = int(cute::size(decltype(dst_atom()){}));
static constexpr int FRAG         = int(cute::size(decltype(tCrA()){}));
static constexpr int COSIZE_A     = int(cute::cosize_v<SmemLayoutA>);
// distinct addresses one atom's slice touches: cosize of the source slice's layout (stride 0 collapses m)
static constexpr int UNIQUE       = int(cute::cosize(decltype(cute::layout(src_atom())){}));
REPORT<decltype(cute::layout(src_atom())), decltype(cute::layout(dst_atom())), MCV, FRAG> report;
