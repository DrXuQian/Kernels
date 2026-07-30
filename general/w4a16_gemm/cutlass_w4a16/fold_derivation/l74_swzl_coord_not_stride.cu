// L74 -- WHY stride-0 ON SmemLayoutA CANNOT CHANGE A's SMEM ADDRESSING. Host side, minimal cute, no PPU types.
//
// The box faulted with an illegal access when SmemLayoutA's M stride was set to 0 and the allocation shrank 16x,
// and the disassembly showed the M step still 512 B (s.add.i32 sreg64, sreg75, 0xc00 / 0xe00). This explains it
// from the definition rather than by inspection of one disassembly:
//
//   the swzl copy atom's copy_unpack passes  src.data().coord_  to the asm -- a LOGICAL COORDINATE, not a linear
//   offset (copy_traits_ppu0010_aiu.hpp; its own comment says partition_S's result "becomes the coord_ handed to
//   the asm"). make_mix_tensor exists precisely to carry (ptr, coord) instead of a resolved pointer.
//
//   MEASURED BELOW: the coordinate at (m,0,0) is (0, m, _0, 0) for BOTH layouts -- the m index enters RAW, not
//   scaled by any stride -- while the linear offsets differ (0,256,512,768 vs 0,0,0,0). So the stride-0 change
//   altered exactly the quantity this path does not use, and left untouched the one it does. The hardware turns
//   coordinate m into base + m*(cube row pitch) either way, which is the 512-byte step the disassembly showed.
//
//   nvcc -std=c++17 -arch=sm_90 -Istub_inc -I<actlize>/include l74.cu -o l74 && ./l74
#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include <cstdio>
using namespace cute;

int main() {
  // A's smem tile: (M, K, Stage) = (16, 256, 2) halfs.
  auto compact = make_layout(make_shape(_16{}, _256{}, _2{}),
                             make_stride(_256{}, _1{}, Int<16*256>{}));   // what tile_to_shape gives, in spirit
  auto bcast   = make_layout(make_shape(_16{}, _256{}, _2{}),
                             make_stride( _0{}, _1{}, _256{}));           // the change that faulted

  std::printf("compact : "); print(compact); std::printf("   cosize=%d halfs\n", int(cosize(compact)));
  std::printf("bcast   : "); print(bcast);   std::printf("   cosize=%d halfs  <- 16x smaller allocation\n",
                                                         int(cosize(bcast)));

  auto show = [](auto lay, const char* tag) {
    // A mix tensor carries (ptr, coord). This is what the swzl atom's copy_unpack forwards to the asm.
    Tensor t = make_mix_tensor(static_cast<half_t*>(nullptr), lay);
    std::printf("  %s  coord_ at (m,0,0) for m=0..3:", tag);
    for (int m = 0; m < 4; ++m) { std::printf("  m%d -> ", m); print(t(m, 0, 0)); }
    std::printf("\n  %s  and the LINEAR offset the layout would give: ", tag);
    for (int m = 0; m < 4; ++m) std::printf(" %d", int(lay(m, 0, 0)));
    std::printf("\n");
  };
  show(compact, "compact");
  show(bcast,   "bcast  ");

  std::printf("\nREAD THIS AS: the linear offsets DO differ -- the layout genuinely changed. The COORDINATES do\n"
              "not: m enters raw as the second component. copy_unpack forwards the coordinate, so the hardware\n"
              "still strides one cube row per m and reads 16x past a 16x smaller allocation.\n"
              "NO LAYOUT CHANGE CAN SHRINK A's SMEM. The fix has to pin the M COORDINATE, i.e. what partition_S\n"
              "yields on the M mode -- copy(atom, tCsA_p(_,0,k), tCrA_copy_view(_,i,k)) for each destination i.\n"
              "Whether that is even a saving depends on CPY_M = size<1>(tCrA_copy_view) being > 1; the box's\n"
              "disassembly showed three tsm.ld.swzl at 512-byte-apart bases, which says it is, but that is an\n"
              "inference from three instructions and wants confirming.\n");
  return 0;
}
