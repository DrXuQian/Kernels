// L77 -- did PPU_A_CUBE_H=1 actually give partition_S an M mode, and did the allocation collapse?
// Front end only; the values arrive as template arguments in the incomplete-type diagnostic.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
using namespace cute;

template <class TCSA, class TCRA, int COSIZE_A> struct REPORT;

using TileShape      = Shape<_16,_32,_256>;
using ScaleTileShape = Shape<_32,_8>;
using WarpShape      = Shape<_16,_16,_256>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;

using SmemLayoutA = typename Mainloop::SmemLayoutA;
static constexpr int COSIZE_A = cute::cosize_v<SmemLayoutA>;

// CPY_M: the M extent partition_S exposes on the smem->reg copy.
using AtomA = typename Mainloop::SmemCopyAtomA;
using TMma  = typename Mainloop::TiledMma;
static auto make_tCsA() {
  Tensor sA = make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutA{});
  return make_tiled_copy_A(AtomA{}, TMma{}).get_thread_slice(0).partition_S(make_mix_tensor_like(sA));
}
static auto make_tCrA() {
  Tensor sA = make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutA{});
  return TMma{}.get_thread_slice(0).partition_fragment_A(sA(_,_,0));
}
REPORT<decltype(make_tCsA().layout()), decltype(make_tCrA().layout()), COSIZE_A> report;
