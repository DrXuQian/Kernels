// L78 -- WHAT DOES THE ATOM DELIVER? CUBE_H=1 shrank the allocation; does it also shrink the DELIVERY into
// registers? The mma A operand is Shape<_16,...>, so 16 rows of register data are needed even though 15 rows of
// RESULTS are discarded by the residue mask. If one atom call carries 1/16 of the bits, the slots the mma reads
// for row 0 may hold whatever was already in those registers -- a NaN mechanism that never faults.
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

template <int COSIZE_A, int SRC_BITS, int DST_BITS, int TCSA_SIZE, int TCRA_SIZE, int MMA_M, int MMA_K> struct REPORT;

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
using AtomA = typename Mainloop::InternalSmemCopyAtomA;   // what is ACTUALLY used on sA
using TMma  = typename Mainloop::TiledMma;
static auto make_tCsA() {
  Tensor sA = make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutA{});
  return make_tiled_copy_A(AtomA{}, TMma{}).get_thread_slice(0).partition_S(make_mix_tensor_like(sA));
}
static auto make_tCrA() {
  Tensor sA = make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutA{});
  return TMma{}.get_thread_slice(0).partition_fragment_A(sA(_,_,0));
}
// Copy_Atom's first template argument IS the op here, so peel it off to name its traits.
template <class> struct AtomOp;
template <class Op, class T> struct AtomOp<cute::Copy_Atom<Op, T>> { using type = Op; };
using TraitsA = cute::Copy_Traits<typename AtomOp<AtomA>::type>;
static constexpr int SRC_BITS  = int(cute::size(typename TraitsA::SrcLayout{}));
static constexpr int DST_BITS  = int(cute::size(typename TraitsA::DstLayout{}));
static constexpr int TCSA_SIZE = int(cute::size(decltype(make_tCsA()){}));
static constexpr int TCRA_SIZE = int(cute::size(decltype(make_tCrA()){}));
static constexpr int MMA_M = int(cute::size<0>(typename TMma::AtomShape_MNK{}));
static constexpr int MMA_K = int(cute::size<2>(typename TMma::AtomShape_MNK{}));
REPORT<COSIZE_A, SRC_BITS, DST_BITS, TCSA_SIZE, TCRA_SIZE, MMA_M, MMA_K> report;
