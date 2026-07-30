// L81 -- ARE THE AIU WRITE AND THE SWZL READ ONE MATCHED PAIR, AND DOES A CUBE OVERRIDE MOVE BOTH?
//
// The PPU_A_CUBE_H NaN is now attributed to changing ONE side: MixGemm_AIU_Operand derives the write's payload
// (bits_per_aiu), the write's cube shape (GmemTiledCopy's value layout, which becomes desc_.cube_h through
// get<0>(Tiler_MN)) and the read's CUBE_H all from Block_MN, and that macro only reached the read. This prints all
// three so a future override can be checked for consistency instead of assumed.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
using namespace cute;

template <int WRITE_BITS, int WRITE_CUBE_M, int READ_CUBE_H, int COSIZE_A, int CPY_M, int CPY_K, int FRAG> struct REPORT;
// CPY_M is the decider for leg 5: if a 1-row read atom makes cute tile 16x in M, the read goes from InstNum=4 to
// 64 issues and the whole triple is a net loss unless one read plus a register broadcast replaces them.

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

// write side: the atom carries bits_per_aiu; the tiler's M becomes desc_.cube_h in load_init
using WriteCopy  = typename Mainloop::GmemTiledCopyA;
using WriteAtom  = typename WriteCopy::Copy_Atom;
using WriteTiler = typename WriteCopy::Tiler_MN;
// read side: the atom actually used on sA
using ReadAtom = typename Mainloop::InternalSmemCopyAtomA;
static constexpr int COSIZE_A = int(cute::cosize_v<typename Mainloop::SmemLayoutA>);
using SmemLayoutA = typename Mainloop::SmemLayoutA;
using TMma = typename Mainloop::TiledMma;
static auto tCsA() {
  Tensor sA = make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutA{});
  return make_tiled_copy_A(ReadAtom{}, TMma{}).get_thread_slice(0).partition_S(make_mix_tensor_like(sA));
}
static auto tCrA() {
  Tensor sA = make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutA{});
  return TMma{}.get_thread_slice(0).partition_fragment_A(sA(cute::_,cute::_,0));
}
template <class T> struct AtomBits;
template <class Op, class E> struct AtomBits<cute::Copy_Atom<Op, E>> { using type = Op; };
template <class N, class E, bool T, bool S, class G> struct BitsOf;
static constexpr int WRITE_BITS = []{
  return int(cute::size(typename cute::Copy_Traits<typename AtomBits<WriteAtom>::type>::SrcLayout{}));
}();
REPORT<WRITE_BITS, int(cute::size<0>(WriteTiler{})), 0, COSIZE_A,
       int(cute::size<1>(decltype(tCsA()){})), int(cute::size<2>(decltype(tCsA()){})),
       int(cute::size(decltype(tCrA()){}))> report;
