// L58 -- the BOX-VALIDATED gate for F=4 that I never used.
//
// nfold_place_bits_int1_tk64 is the offline the single-plane int1 (64,128,64) w64x64 config runs -- the one that
// measures 63.7% MFU. It was derived by l10 from the same chain (swzl delivery o MixGemmEmit o pi o partition_B) and
// is bit-granular precisely because whole-word moves CANNOT express F=4. So it is ground truth for int1 at F=4, and
// xplane::plane_map<1,...,F=4> must reproduce it. That comparison has never been made, and F=4 is exactly the fold
// plane 2 uses at Block_K=64 -- the rung that fails.
//
// If they differ, the swzl TV formula (row = inst*RPI + 16*warp_n + (v/2)*8 + lane/4, wd = (v%2)*4 + lane%4) that
// every one of my maps uses is not the right one at F=4, and every derivation built on it inherits that.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include -I.. l58_f4_gate.cu -o l58 && ./l58
#include "cute/tensor.hpp"
#include "unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"
#include "xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;

template <int TM,int TN,int TK,int WM,int WN,int F>
static bool check(const char* tag, int N, int K) {
  constexpr int CPW = 32, Ng = TN / F;
  const int W_ROW_OFF = 256 / CPW, RUNS = W_ROW_OFF / 8, nrow = N / F;
  const auto m = xplane::plane_map<1, TM, TN, TK, WM, WN, F>();

  std::vector<uint8_t> codes((size_t)N * K);                       // logical [n][k], one bit per byte
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t((i * 2654435761u >> 7) & 1);

  // emit from plane_map
  std::vector<int8_t> a((size_t)N * K / 8, 0);
  for (int tn = 0; tn < N/TN; ++tn) for (int ki = 0; ki < K/TK; ++ki)
    for (int row = 0; row < Ng; ++row) for (int wd = 0; wd < 8; ++wd) for (int j = 0; j < CPW; ++j) {
      const int loc = m[((size_t)row * 8 + wd) * CPW + j]; if (loc < 0) continue;
      const int n = tn*TN + loc/TK, k = ki*TK + loc%TK;
      if (!codes[(size_t)n*K + k]) continue;
      const int kb = ki/RUNS, tt = ki%RUNS;
      const size_t b = (size_t)((((size_t)kb*nrow + (size_t)tn*Ng + row)*W_ROW_OFF + tt*8 + wd)*CPW + j);
      a[b/8] |= int8_t(1 << (b%8));
    }

  // emit from the box-validated l10 packer: it takes [n][k] packed one bit per bit
  std::vector<int8_t> nk((size_t)N*K/8, 0);
  for (size_t i = 0; i < (size_t)N*K; ++i) if (codes[i]) nk[i/8] |= int8_t(1 << (i%8));
  std::vector<int8_t> b_((size_t)N*K/8, 0);
  legacy::nfold_place_bits_int1_tk64(b_.data(), nk.data(), N, K, TN, TK);

  size_t d = 0; for (size_t i = 0; i < a.size(); ++i) if (a[i] != b_[i]) ++d;
  printf("  %-34s %dx%d : %zu / %zu bytes differ  %s\n", tag, N, K, d, a.size(),
         d ? "<-- plane_map DISAGREES with the box-validated F=4 placement" : "<-- BIT-IDENTICAL");
  return d == 0;
}

int main() {
  printf("L58 -- plane_map at F=4 vs legacy::nfold_place_bits_int1_tk64(the 63.7% config's own offline)\n\n");
  bool a = check<64,128,64,64,64,4>("int1 (64,128,64) w64x64 F=4", 256, 512);
  bool b = check<64,128,64,64,64,4>("int1 (64,128,64) w64x64 F=4", 256, 256);
  printf("\n  %s\n", (a&&b) ? "the TV formula and plane_map hold at F=4"
                            : "F=4 is where the shared TV formula breaks -- every map built on it inherits this");
  return 0;
}
