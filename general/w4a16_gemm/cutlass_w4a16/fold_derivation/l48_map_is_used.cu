// L48 -- is the cross-plane tile map actually DIFFERENT from plane 2's own single-plane map, or did l47's equality
// come from the map never being used / degenerating?
//
// l47 reported the F2=2 cross-plane buffer bit-identical to the single-plane rule. That is either a real fact (the
// two rules coincide) or an artifact. The box then got WORSE with the new kernel index (15010 -> 29666 bad), which
// says the pairing is wrong -- so the equality is the thing to attack first. Two direct checks, no box needed:
//   1. compare the two tile maps entry by entry
//   2. perturb the cross-plane map and see whether the emitted buffer changes at all
#include "cute/tensor.hpp"
#include "unfused_weight_dequantize.hpp"
#include "xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;

int main() {
  constexpr int TM=64, TN=64, TK=128, WM=32, WN=32, F2=2;
  auto mx = xplane::tile_map_int1<TM,TN,TK,WM,WN,F2>();

  // plane 2's OWN single-plane map, l20's tile_map for int1 at the folded geometry: destination identical, source
  // taken from int1's own fragment + MixGemmEmit<1> instead of from plane 1.
  long diff = 0, both = 0;
  {
    using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
    constexpr int WOM = TM/32, WON = TN/32, RPI = WON*16, CPW = 32, R2 = TN/F2, B2 = F2*TK/8;
    using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>,Int<WON>,_1>>, Tile<Int<WOM*16>,Int<WON*16>,_32>>;
    auto id2 = make_identity_tensor(make_shape(Int<R2>{}, Int<B2>{}));
    std::vector<int> m2((size_t)R2*8*CPW, -1);
    for (int t = 0; t < 32*WOM*WON; ++t) {
      const int lane = t%32, w = t/32, warp_n = w/WOM;
      auto p2 = MmaS8{}.get_thread_slice(t).partition_B(id2);
      for (int inst = 0; inst < R2/RPI; ++inst)
        for (int v = 0; v < 4; ++v) {
          const int row = inst*RPI + 16*warp_n + (v/2)*8 + lane/4, wd = (v%2)*4 + lane%4;
          for (int j = 0; j < CPW; ++j) {
            auto c = p2((32*v + j)/8, inst, 0);
            const int g = int(get<0>(c)), byte = int(get<1>(c));
            const int code = byte*8 + (32*v + j)%8, f = code/TK;
            m2[((size_t)row*8 + wd)*CPW + j] = (f + F2*g)*TK + code%TK;
          }
        }
    }
    for (size_t i = 0; i < mx.size(); ++i) { if (mx[i] >= 0 && m2[i] >= 0) { ++both; if (mx[i] != m2[i]) ++diff; } }
    printf("  1. cross-plane map vs plane-2's own map : %ld / %ld entries differ  %s\n", diff, both,
           diff ? "<-- genuinely different maps" : "<-- IDENTICAL, so l47's equality was never a test of anything");
  }

  // 2. does the emitted buffer even depend on the map?
  {
    std::vector<uint8_t> hi((size_t)128 * 256);
    for (size_t i = 0; i < hi.size(); ++i) hi[i] = uint8_t((i * 2654435761u >> 7) & 1);
    std::vector<int8_t> a((size_t)256*128/8);
    xplane::place_int1<TM,TN,TK,WM,WN,F2>(a.data(), hi, 256, 128);
    long nz = 0; for (int8_t b : a) if (b) ++nz;
    printf("  2. emitted buffer: %ld / %zu bytes nonzero %s\n", nz, a.size(),
           nz ? "" : "<-- ALL ZERO: the walk never fired, nothing was tested");
  }
  return 0;
}
