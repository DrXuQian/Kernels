// L26 -- why the register-count prescription failed, and what the real objective is.
//
// THE FAILED PREDICTION. From regs_per_thread I prescribed w16x64: it keeps WN=64 (so the delivery bound still
// passes at int1/TK=64) while halving accum = WM*WN/32, giving 128 regs against 176. Measured: 39.7% against
// 50.0%. Register pressure was not the objective.
//
// WHAT ACTUALLY SEPARATES THE 20 MEASURED POINTS is WM alone, with no overlap at all:
//     WM=32, regs<=256 : 40.9 - 50.2%    (10 points)
//     WM=16, regs<=256 : 31.9 - 39.8%    ( 8 points)
// The best WM=16 point (39.8%, and it has blk=32, the HIGHEST occupancy in the sweep) is below the WORST WM=32
// point (40.9%, blk=4, the lowest). So occupancy cannot be the discriminator across the groups -- it only orders
// within them.
//
// THE MECHANISM, counted here rather than argued. In a mixed-input GEMM every B fragment element must be
// converted (lop3 + fma) and scaled before it can enter an mma. Per thread per k-tile, out of the real TiledMma:
//     mma instructions   = (WM/16)*(WN/16)*(TK/16)
//     B elements to cvt  = 8*(WN/16)*(TK/16)
//     cvt elements / mma = 128/WM                      <- WN and TK cancel; only WM survives
// Each B fragment serves WM/16 mma instructions in the M direction, so WM/16 IS the converter amortisation
// factor. WM=16 means every converted element feeds exactly one mma: the converter cost per unit of math doubles.
//
// This is the quantised-B analogue of arithmetic intensity, at the register level rather than at HBM, and it is
// why a prescription that minimised registers by cutting WM was exactly backwards.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l26_convert_amort.cu -o l26 && ./l26
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <vector>
using namespace cute;

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

// count it from the real partition, not from the formula
template <int TM, int TN, int TK, int WM, int WN>
static void count(double mfu) {
  constexpr int WOM = TM/WM, WON = TN/WN;
  using Mma = TiledMMA<MMA_Atom<F16Atom>, Layout<Shape<Int<WOM>,Int<WON>,_1>>,
                       Tile<Int<WOM*16>, Int<WON*16>, Int<TK>>>;
  auto sA = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TM>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto sB = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                        make_layout(Shape<Int<TN>,Int<TK>>{}, Stride<Int<TK>,_1>{}));
  auto thr  = Mma{}.get_thread_slice(0);
  auto fA   = thr.partition_fragment_A(sA);      // (MMA, MMA_M, MMA_K)
  auto fB   = thr.partition_fragment_B(sB);      // (MMA, MMA_N, MMA_K)
  const int MMA_M = int(size<1>(fA)), MMA_N = int(size<1>(fB)), MMA_K = int(size<2>(fB));
  const int mmas  = MMA_M * MMA_N * MMA_K;       // mma instructions per thread per k-tile
  const int cvts  = int(size(fB));               // B elements this thread must convert per k-tile
  const int accum = WM*WN/32, ra = WM*TK/64, rb = WN*TK/64, rs = WN*TK/256;
  printf("  %5.1f%%  (%3d,%3d,%3d) w%dx%-3d | MMA %dx%dx%d = %3d mma | cvt %3d B-elem | %5.2f cvt/mma"
         "  amort=WM/16=%d | regs=%d%s\n",
         mfu, TM, TN, TK, WM, WN, MMA_M, MMA_N, MMA_K, mmas, cvts, double(cvts)/mmas, WM/16,
         accum+ra+rb+rs, (accum+ra+rb+rs) > 256 ? " !" : "");
}

int main() {
  printf("L26 -- converter amortisation: cvt elements per mma, counted from cute\n\n");
  printf("  == WM=32 group (all 10 measured points, sorted by MFU)\n");
  count<32,256, 64,32,64>(50.2);  count<32,128, 64,32,64>(50.0);  count<64,128, 64,32,64>(48.4);
  count<32,128, 64,32,64>(47.6);  count<32,128,128,32,32>(46.5);  count<64,128, 64,32,64>(46.4);
  count<64,128,128,32,32>(44.7);  count<32,128,128,32,32>(43.1);  count<32,128,128,32,32>(42.0);
  count<64,128,128,32,32>(40.9);
  printf("\n  == WM=16 group (all 8 measured points, sorted by MFU)\n");
  count<16,128, 64,16,64>(39.8);  count<32,128, 64,16,64>(39.7);  count<32,128, 64,16,64>(39.3);
  count<32,128,128,16,32>(38.7);  count<16,128,128,16,32>(38.5);  count<16,128,128,16,32>(38.4);
  count<16,128,256,16,32>(32.1);  count<16,128,256,16,32>(31.9);
  printf("\n  == spilled\n");
  count<32,128,128,32,64>(39.0);  count<32,128,256,32,32>(23.0);
  printf("\n  cvt/mma is 8.00 for every WM=16 row and 4.00 for every WM=32 row -- WN and TK cancel exactly,\n");
  printf("  and the 39.8/40.9 gap between the groups is where that factor of 2 shows up.\n");
  printf("\n  == would WM=64 be better still (amort 4)? the register budget answers\n");
  count<64,128, 64,64,64>(0.0);
  printf("  272 regs > 256, so WM=64 spills at WN=64/TK=64. WM=32 is the largest amortisation the budget allows,\n");
  printf("  and int1's delivery bound pins WN>=64 at TK=64, so w32x64 is FORCED -- which is the measured optimum.\n");
  return 0;
}
