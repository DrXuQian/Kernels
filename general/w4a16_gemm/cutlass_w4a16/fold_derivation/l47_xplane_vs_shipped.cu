// L47 -- close the gap: does the CROSS-PLANE generator reproduce the SHIPPED plane-2 buffer at the unfolded config?
//
// I said the frame precondition (8*CPW == one 32B AIU run per row) ruled the TK=256 control out. Too broad: in
// tile_map_int1 that precondition binds only PLANE 2, whose destination uses the swzl TV formula. Plane 1 enters only
// through partition_B for the logical (n,k) and carries no frame assumption. Plane 2 at TK=256, F2=1 is 8*32 = 256,
// so the control IS available -- the earlier failure was my partition_B-based destination, now replaced.
//
// This is the check that matters: at F2=1 the derived buffer must equal what preprocess_weights_for_mixed_gemm
// produces for the int1 plane today, byte for byte. If it does, the SAME generator at F2=2 is trustworthy.
#include "cute/tensor.hpp"
#include "unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"
#include "xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;

template <int TM, int TN, int TK, int WM, int WN, int F2>
static bool check(const char* tag, int N, int K) {
  std::vector<uint8_t> hi((size_t)K * N);
  for (size_t i = 0; i < hi.size(); ++i) hi[i] = uint8_t((i * 2654435761u >> 7) & 1);

  std::vector<int8_t> derived((size_t)N * K / 8);
  xplane::place_int1<TM,TN,TK,WM,WN,F2>(derived.data(), hi, N, K);

  // the shipped path for the high plane: transpose to [N][K], pack 8/byte, preprocess (+ its own fold if F2 > 1)
  std::vector<int8_t> packed((size_t)N * K / 8, 0);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n)
    if (hi[(size_t)k * N + n] & 1) { size_t i = (size_t)n * K + k; packed[i / 8] |= int8_t(1 << (i % 8)); }
  std::vector<int8_t> shipped(packed.size());
  preprocess_weights_for_mixed_gemm<false, 256, 0>(shipped.data(), packed.data(),
      {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);
  if (F2 > 1) { std::vector<int8_t> f(shipped.size());
                legacy::nfold_regroup_gmem(f.data(), shipped.data(), {(size_t)K,(size_t)N}, TN, TK, 1); shipped.swap(f); }

  size_t d = 0, nzD = 0, nzS = 0;
  for (size_t i = 0; i < derived.size(); ++i) { if (derived[i] != shipped[i]) ++d; if (derived[i]) ++nzD; if (shipped[i]) ++nzS; }
  // nonzero counts on BOTH sides: two all-zero buffers also "agree", and l48 says the maps differ in 8160/8192
  // entries, so an agreement here needs corroborating that anything was compared at all.
  printf("  %-34s %dx%d : %zu / %zu differ | nonzero derived=%zu shipped=%zu  %s\n", tag, N, K, d, derived.size(),
         nzD, nzS, d ? "<-- DIFFERS" : "<-- BIT-IDENTICAL");
  return d == 0;
}

int main() {
  printf("L47 -- cross-plane generator vs the shipped offline\n\n");
  printf("  CONTROL: unfolded, where the derived buffer must equal what runs correctly today\n");
  bool c = check<64, 64,256,32,32, 1>("F2=1 (64,64,256)", 256, 512);
  bool c2 = check<64, 64,256,32,32, 1>("F2=1 (64,64,256) larger", 512, 1024);
  printf("\n  FOLDED: no shipped equivalent exists -- printed to show HOW MUCH it differs from plane 2's own rule,\n");
  printf("  which is the buffer that measured bad=15010/32768 on the box\n");
  // THE REAL TEST'S SHAPE. K=256 is 2 k-tiles, not 4 -- if place_int1's walk depends on K/TK the equality can hold at
  // one and not the other, and the box runs K=256. That difference is the only thing left that distinguishes the run
  // which measured 29666 from what the composition predicts.
  check<64, 64,128,32,32, 2>("F2=2 (64,64,128) K=512", 256, 512);
  check<64, 64,128,32,32, 2>("F2=2 (64,64,128) K=256  <-- the box's shape", 256, 256);
  check<64, 64,128,32,32, 2>("F2=2 (64,64,128) K=128", 256, 128);
  printf("\n  verdict: %s\n", (c && c2) ? "generator validated on the control"
                                        : "generator does NOT reproduce the shipped buffer -- do not trust the folded map");
  return 0;
}
