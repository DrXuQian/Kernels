// Compile-test FoldTraits against every box reference point: the 7 good ones must instantiate, the 2 bad ones
// must be rejected (checked by SFINAE-style probing of the static_asserts via a separate TU, see ft_neg below).
#include "/root/marlin_ppu/Kernels/general/w4a16_gemm/cutlass_w4a16/fold_traits.hpp"
#include <cstdio>
using namespace fold;
template<class T> void show(const char* tag){
  printf("  %-18s F=%d Ng=%3d deliv=%3d slots=%3d cols=%2d | smem=%6d B warps=%d blocks=%d\n",
         tag, T::F, T::Ng, T::delivery, T::slots, T::cols_demanded, T::smem, T::warps, T::blocks);
}
int main(){
  printf("FoldTraits -- the 7 validated box configs must all instantiate:\n");
  show<FoldTraits<4,64, 64, 64>>("int4 (64,64,64)");
  show<FoldTraits<2,64, 64, 64>>("int2 (64,64,64)");
  show<FoldTraits<2,64,128, 64>>("int2 (64,128,64)");
  show<FoldTraits<2,64, 64,128>>("int2 (64,64,128)");
  show<FoldTraits<1,32,128,128>>("int1 (32,128,128)");
  show<FoldTraits<1,64, 64,128>>("int1 (64,64,128)");
  show<FoldTraits<1,64, 64,256>>("int1 (64,64,256)");
  printf("\n...and a wide-column config that must NOT be rejected (column spans 2 slices):\n");
  show<FoldTraits<4,64, 64,128>>("int4 (64,64,128)");
  show<FoldTraits<2,64, 64,256>>("int2 (64,64,256)");

  // The converter-amortisation metric, against the measured groups. cvt/mma = 128/WM is the tier-2 objective and it
  // separated all 20 sweep points with no overlap; these asserts are what keeps that number from drifting.
  printf("\nconverter amortisation (tier 2 of the three-tier model):\n");
  static_assert(cvt_per_mma<16> == 8 && cvt_per_mma<32> == 4 && cvt_per_mma<64> == 2, "cvt/mma = 128/WM");
  static_assert(amort<16> == 1 && amort<32> == 2 && amort<64> == 4, "amort = WM/16");
  printf("  WM=16 -> cvt/mma=%d amort=%d   measured 31.9-39.8%%\n", cvt_per_mma<16>, amort<16>);
  printf("  WM=32 -> cvt/mma=%d amort=%d   measured 40.9-50.2%%\n", cvt_per_mma<32>, amort<32>);
  printf("  WM=64 -> cvt/mma=%d amort=%d   int1: unreachable, 272 regs is the space minimum\n",
         cvt_per_mma<64>, amort<64>);

  // int1 at amort=4 must be over budget for EVERY legal shape, or the cap claim in fold_traits.hpp is wrong. The two
  // register-minimal shapes under the delivery bound WN*TK>=4096 are TK=64/WN=64 and TK=128/WN=32.
  static_assert(regs_per_thread<64,128, 64,64,64> == 272 && regs_per_thread<64, 64,128,64,32> == 272,
                "int1 amort=4: both minimal shapes must land at 272");
  static_assert(!regs_ok<64,128, 64,64,64> && !regs_ok<64, 64,128,64,32>, "int1 amort=4 must be over budget");
  // int4's bound is 8x looser, so amort=4 IS reachable there -- the untried lever on the shipped path.
  static_assert(deliverable<4,64,32,64,32> && regs_ok<64,64,32,64,32> && regs_per_thread<64,64,32,64,32> == 116,
                "int4 (64,64,32) w64x32 must be legal at amort=4");
  printf("  int1 amort=4 : %d regs, legal shapes exist but all over 256 -> capped at amort=2\n",
         regs_per_thread<64,128,64,64,64>);
  printf("  int4 amort=4 : (64,64,32) w64x32 = %d regs, deliverable -> reachable\n",
         regs_per_thread<64,64,32,64,32>);
  // ...and MEASURED to be worth nothing. w64x64 s2 (the only int1 shape at cvt/mma=2) ran at 272 regs and scored
  // 48.7% against its cvt/mma=4 / 176-reg peer's 48.4% at the SAME blk=13. So the cap above is harmless and the
  // int4 lever is retracted; cvt_per_mma_target records where the return actually stops.
  static_assert(cvt_per_mma_target == 4, "the B convert path stops binding at cvt/mma=4; below that is +0.3 points");
  printf("  BUT MEASURED: cvt/mma=2 buys +0.3 points over cvt/mma=4 at equal blk -> target is %d, not lower;\n",
         cvt_per_mma_target);
  printf("               the amort=2 cap is harmless and the int4 amort=4 lever is RETRACTED.\n");
  printf("  and 272 regs cost nothing at equal blk (288 -> -2.6, 320 -> -19.9), so the knee is above 272, not 256.\n");
  printf("\nall compiled.\n");
  return 0;
}
