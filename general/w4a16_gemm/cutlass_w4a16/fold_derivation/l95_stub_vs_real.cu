// L95 -- IS l21's STUB THE SAME OBJECT AS THE REAL COLLECTIVE'S? Type identity, not a map comparison.
//
// l94's bank and lane->n results are computed on l21's stub mma, because the CollectiveBuilder's types can be NAMED
// locally but not CALLED: calling partition_S needs -D__HGGCCC__ so CUTLASS_DEVICE lands in device code, and then
// cute's namespace-scope `_` is not device-visible under nvcc.
//
// Comparing the two lane->n maps element by element would be the weak check -- it would only cover the coordinates I
// happened to print. If the three TYPES the map is derived from are identical, every derived quantity is identical by
// construction. So this file asserts type identity and nothing else, and it needs no function calls at all.
//
// A failing static_assert here invalidates l94's (2) and (3) rows, so it is worth its own file.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include <cstdio>
#include <type_traits>
using namespace cute;

// ---- the REAL one, exactly as l90 builds it (the decode-band config the instruction mix was captured on)
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

// ---- the STUB, exactly as l21 and l94 build it
struct F16Atom {};
namespace cute {
template <> struct MMA_Traits<F16Atom> {
  using ValTypeD=float; using ValTypeA=cutlass::half_t; using ValTypeB=cutlass::half_t; using ValTypeC=float;
  using Shape_MNK=Shape<_16,_16,_16>; using ThrID=Layout<_32>;
  using ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using CLayout=Layout<Shape<Shape<_4,_8>,Shape<_4,_2>>,Stride<Stride<_16,_1>,Stride<_64,_8>>>;
};
}
static constexpr int kWON = 8;
using StubMma       = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<_1,Int<kWON>,_1>>, Tile<_16,Int<kWON*16>,_64>>;
using StubSmemScale = decltype(tile_to_shape(Layout<Shape<_8,_1>>{}, make_shape(_128{}, _8{}, _2{})));
using StubCopyAtomS = Copy_Atom<DefaultCopy, cutlass::half_t>;

// THE THREE THINGS l94's map is derived from. Layout identity is what matters, so compare the LAYOUTS (a TiledMMA's
// identity includes its atom type, which differs by construction -- the stub's atom is a different C++ type carrying
// the same traits, and that is exactly the substitution being justified).
static_assert(std::is_same_v<StubSmemScale, typename Mainloop::SmemLayoutScale>,
              "l94's SmemLayoutScale is not the collective's -- its (2)/(3) rows are invalid");
static_assert(std::is_same_v<StubCopyAtomS, typename Mainloop::SmemCopyAtomScale>,
              "l94's SmemCopyAtomScale is not the collective's");
static_assert(std::is_same_v<decltype(StubMma{}.get_layoutB_TV()),
                             decltype(typename Mainloop::TiledMma{}.get_layoutB_TV())>,
              "the stub mma's B (thread,value) layout differs from the collective's -- l94's lane->n map is invalid");
static_assert(std::is_same_v<typename StubMma::ThrLayoutVMNK,
                             typename Mainloop::TiledMma::ThrLayoutVMNK>,
              "the stub mma's thread layout differs from the collective's");
// (TiledShape_MNK is not a member of this cute's TiledMMA; get_layoutB_TV identity above covers it.)

// NOTHING is printed and nothing is called. Every cute entry point (print, product, get_layoutB_TV) instantiates a
// device path under -D__HGGCCC__ that references namespace-scope constants nvcc cannot see in device code. The
// static_asserts above are unevaluated contexts, so THEY are the check: if this file compiles, the stub is the
// collective's own object. `nvcc ... l95_stub_vs_real.cu` compiling clean IS the pass condition.
int main() {
  std::printf("== l95 PASS: it compiled, so every static_assert above holds ==\n");
  std::printf("   stub SmemLayoutScale == Mainloop::SmemLayoutScale\n");
  std::printf("   stub SmemCopyAtomScale == Mainloop::SmemCopyAtomScale\n");
  std::printf("   stub mma layoutB_TV == Mainloop::TiledMma layoutB_TV   (with Tile K = 64, NOT TileK=256)\n");
  std::printf("   stub mma ThrLayoutVMNK == Mainloop::TiledMma ThrLayoutVMNK\n");
  return 0;
}
