// L52 -- the one side of the composition never checked against shipped code: PLANE 1's map.
//
// l51 looked like it validated everything, but half of it is circular: tile_map_int1 is DERIVED from the HV=1 pairing,
// so checking the pairing against it must pass. The non-circular half is real -- the derived high map equals the
// shipped fold byte for byte -- which validates plane 2. Plane 1 was never checked. If plane_map<2,...> is not what
// preprocess_weights_for_mixed_gemm actually produces for int2, the whole composition rests on it and its "0
// mispaired" means nothing, which would explain the contradiction with the box.
#include "cute/tensor.hpp"
#include "unfused_weight_dequantize.hpp"
#include "xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;

template <int TM,int TN,int TK,int WM,int WN>
static bool check(const char* tag, int N, int K) {
  constexpr int Bits = 2, CPW = 32 / Bits, DL = (TK * Bits / 8) / 32;
  const int kCon = 256, contig = TK * Bits / 8;
  const int AiuByte = contig > 128 ? 128 : contig, AiuElem = AiuByte * 8 / Bits, RPS = kCon / AiuElem;
  const auto m = xplane::plane_map<Bits, TM, TN, TK, WM, WN, 1>();

  std::vector<int> codes((size_t)N * K);
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = int((i * 2654435761u >> 5) & 3);
  std::vector<int8_t> derived((size_t)N * K * Bits / 8, 0);
  for (int tn = 0; tn < N/TN; ++tn) for (int ki = 0; ki < K/TK; ++ki)
    for (int row = 0; row < TN; ++row) for (int dl = 0; dl < DL; ++dl)
      for (int wd = 0; wd < 8; ++wd) for (int j = 0; j < CPW; ++j) {
        const int loc = m[(((size_t)row * DL + dl) * 8 + wd) * CPW + j]; if (loc < 0) continue;
        const int n = tn*TN + loc/TK, k = ki*TK + loc%TK;
        const size_t bp = (size_t)(((size_t)(ki/RPS)*N + (size_t)tn*TN + row)*kCon
                                   + (ki%RPS)*AiuElem + (dl*8 + wd)*CPW + j) * Bits;
        derived[bp/8] |= int8_t((codes[(size_t)n*K + k] & 3) << (bp%8));
      }

  std::vector<int8_t> packed((size_t)N*K*Bits/8, 0);
  for (size_t i = 0; i < (size_t)N*K; ++i) packed[i/4] |= int8_t((codes[i] & 3) << (2*(i%4)));
  std::vector<int8_t> ship(packed.size());
  preprocess_weights_for_mixed_gemm<false,256,0>(ship.data(), packed.data(), {(size_t)K,(size_t)N},
                                                 QuantTypeClass::PACKED_INT2_WEIGHT_ONLY);
  size_t d = 0; for (size_t i = 0; i < ship.size(); ++i) if (derived[i] != ship[i]) ++d;
  printf("  %-28s %dx%d DL=%d : %zu / %zu bytes differ  %s\n", tag, N, K, DL, d, ship.size(),
         d ? "<-- plane 1's map is NOT the shipped one; the composition rests on it" : "<-- BIT-IDENTICAL");
  return d == 0;
}

int main() {
  printf("L52 -- is plane 1's map the shipped int2 offline?\n\n");
  bool a = check<64,64,128,32,32>("int2 (64,64,128) F=1", 256, 512);
  bool b = check<64,64,256,32,32>("int2 (64,64,256) F=1 DL=2", 256, 512);
  printf("\n  %s\n", (a&&b) ? "both sides of the composition are now validated against shipped code"
                            : "the composition's low side is unvalidated -- fix before trusting its verdict");
  return 0;
}
