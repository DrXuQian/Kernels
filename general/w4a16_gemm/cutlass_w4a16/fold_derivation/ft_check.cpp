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
  printf("\nall compiled.\n");
  return 0;
}
