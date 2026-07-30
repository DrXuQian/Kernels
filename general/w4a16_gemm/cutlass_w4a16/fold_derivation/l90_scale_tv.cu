// L90 -- THE SCALE READ, THROUGH THE KERNEL'S OWN CONSTRUCTION. l89 hand-rolled partition_S and got a slice that
// still carried the group and stage strides; this uses the collective's own scale_fragment_layout (the host twin of
// make_scale_fragment) and the same make_tiled_copy_B, so what is printed is what the kernel asks for.
//
// The question that matters: for ONE group, which smem address does each of the 32 lanes read? Two lanes on the same
// address is a same-address access -- a broadcast on some hardware, a 2-way conflict on others -- and acu measures
// 278,528 conflicts against 272,384 scale loads, i.e. 1.02 each, which is what a permanent 2-way conflict looks like.
// The TV layout below is what settles it; padding cannot (l90's predecessor PPU_SCALE_PAD cost 9.7%).
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
using namespace cute;

template <class SMEM_SCALE, class FRAG_LAYOUT, class SRC_TV, int FRAG_COSIZE> struct REPORT;

using TileShape      = Shape<_16,_128,_256>;          // the config the instruction mix was captured on
using ScaleTileShape = Shape<_128,_8>;                // gs=32, TileK=256 -> Scale_TileK = 8
using WarpShape      = Shape<_16,_16,_256>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;

using SmemLayoutScale = typename Mainloop::SmemLayoutScale;
using TMma            = typename Mainloop::TiledMma;
static auto sS()  { return make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutScale{}); }
static auto frag_layout() { return Mainloop::scale_fragment_layout(TMma{}.get_thread_slice(0), sS()); }
static auto tiled()  { return make_tiled_copy_B(typename Mainloop::SmemCopyAtomScale{}, TMma{}); }
// (thread, value) -> coordinate in the SOURCE tensor: this is the lane -> address map, before SmemLayoutScale
static auto src_tv()  { return tiled().get_layoutS_TV(); }

REPORT<SmemLayoutScale, decltype(frag_layout()), decltype(src_tv()), Mainloop::scale_frag_cosize()> report;
