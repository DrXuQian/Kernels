// L89 -- WHY DOES ONE SCALE GROUP TAKE ~4 SHARED LOADS PER WARP? acu, sz against pc: the scale/zero reads are +272k
// instructions (2.08 per mma) with +252k bank conflicts, about 1.02 per read, and % Peak only 28 -- so it is
// transactions x latency, and both the WIDTH of each read and the bank pattern are ours to change.
//
// Reported: the smem-side and register-side layouts of ONE group's scale slice, the vector width cute can use, and
// the element counts. The strides tell the bank story; max_common_vector tells the width story.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
using namespace cute;

template <class SMEM_LAYOUT, class SRC_ATOM, class DST_ATOM, int MCV, int SRC_N, int DST_N> struct REPORT;

using TileShape      = Shape<_16,_64,_256>;
using ScaleTileShape = Shape<_64,_8>;                 // gs=32 with TileK=256 -> Scale_TileK = 8
using WarpShape      = Shape<_16,_16,_256>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;
using TMma = typename Mainloop::TiledMma;
using SmemLayoutScale = typename Mainloop::SmemLayoutScale;

static auto sS() { return make_tensor(make_smem_ptr(static_cast<cutlass::half_t*>(nullptr)), SmemLayoutScale{}); }
static auto tiled() { return make_tiled_copy_B(typename Mainloop::SmemCopyAtomScale{}, TMma{}); }
static auto tCsS() { return tiled().get_thread_slice(0).partition_S(sS()); }
static auto frag() {
  return make_fragment_like<cutlass::half_t>(TMma{}.get_thread_slice(0).partition_fragment_B(sS()(_,_,0)));
}
static auto tCrS() { return tiled().get_thread_slice(0).retile_D(decltype(frag()){}); }
// one group's slice, as transform_B_kblock reads it: (_,_,0,sk)
static auto src1() { return tCsS()(_,_,Int<0>{},Int<0>{}); }
static auto dst1() { return tCrS()(_,_,Int<0>{}); }

static constexpr int MCV   = int(decltype(max_common_vector(src1(), dst1())){});
static constexpr int SRC_N = int(cute::size(decltype(src1()){}));
static constexpr int DST_N = int(cute::size(decltype(dst1()){}));
REPORT<SmemLayoutScale, decltype(cute::layout(src1())), decltype(cute::layout(dst1())), MCV, SRC_N, DST_N> report;
