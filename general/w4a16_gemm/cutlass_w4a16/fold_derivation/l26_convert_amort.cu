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
// BUT IT IS A THRESHOLD, NOT A RATE -- see the decisive point at the bottom of main(). Pushing cvt/mma from 4 to 2
// (w64x64, the only int1 shape that reaches it) is worth +0.3 points at equal blk. cvt/mma=8 is throughput-bound
// and flat in occupancy; cvt/mma=4 is latency-bound and occupancy is the lever; below 4 there is nothing left.
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
  printf("\n  == THE DECISIVE POINT, and it corrects the two claims above. w64x64 is the only int1 shape at\n");
  printf("     cvt/mma=2, and it ran -- 272 regs did NOT collapse:\n");
  count<64,128, 64,64,64>(48.7);
  printf("     against its cvt/mma=4 peer at the SAME blk=13:\n");
  count<64,128, 64,32,64>(48.4);
  printf("\n  Half the converter work plus 96 extra registers = +0.3 points. So cvt/mma is a THRESHOLD, not a rate,\n");
  printf("  and the register knee is above 272 rather than at 256. The signature is occupancy sensitivity, over the\n");
  printf("  same ~5x blk span at regs<=176 and TK<=128:\n\n");
  printf("      cvt/mma=8 : blk  8 -> 32   MFU 38.4 - 39.8%%   spread 1.4 pts   FLAT -- throughput-bound on B cvt\n");
  printf("      cvt/mma=4 : blk  4 -> 23   MFU 40.8 - 50.2%%   spread 9.4 pts   latency-bound -- blocks are the lever\n");
  printf("\n  A flat response to 4x the occupancy is what a throughput ceiling looks like; once past it there is\n");
  printf("  nothing left for a smaller cvt/mma to recover. Register penalty, vs a same-blk 176-reg peer:\n");
  printf("      272 -> +0.3 (nothing)      288 -> -2.6      320 -> -19.9\n");
  printf("\n  PRESCRIPTION: reach cvt/mma=4 (WM>=32), stay under ~288 regs, then maximise blocks. int1's measured\n");
  printf("  optimum w32x64 s2 at TK=64 is exactly that, so its amort=2 cap is harmless -- and the int4 amort=4\n");
  printf("  lever proposed off the earlier model is RETRACTED, since int4 is already at cvt/mma=4.\n");
  return 0;
}
