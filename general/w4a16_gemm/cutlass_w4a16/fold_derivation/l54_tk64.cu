// L54 -- is the cross-plane composition complete and consistent at Block_K=64, where BOTH planes fold (F1=2, F2=4)?
// The low side is gated (l52: plane_map<2,...,F=2> is byte-identical to the shipped int2 offline at TK=64). There is no
// shipped 2-plane reference at TK=64, so the checks available are completeness (every plane-2 slot demanded) and
// consistency (no slot demanded for two different logical elements) -- plus the F1=1 cases still reproducing shipped,
// which l53 covers.
#include "cute/tensor.hpp"
#include "xplane_offline.hpp"
#include <cstdio>
#include <vector>
#include <map>
int main() {
  auto show = [&](const char* tag, const std::vector<int>& m, int expect) {
    long unset = 0; std::map<int,int> mult;
    for (int e : m) { if (e < 0) ++unset; else ++mult[e]; }
    long bad = 0; for (auto& kv : mult) if (kv.second != 1) ++bad;
    printf("  %-34s slots=%zu unset=%ld distinct=%zu slots-per-element!=1:%ld  %s\n",
           tag, m.size(), unset, mult.size(), bad,
           (!unset && !bad && (int)mult.size() == expect) ? "<-- COMPLETE bijection" : "<-- incomplete");
  };
  // Block_K=128, the config that measures bad=0 on the box -- the reference shape for this check
  show("TK=128 F1=1 F2=2 (32,128,128)", xplane::tile_map_int1<32,128,128,32,32, 2, 1>(), 128*128);
  show("TK=128 F1=1 F2=2 (64, 64,128)", xplane::tile_map_int1<64, 64,128,32,32, 2, 1>(),  64*128);
  // Block_K=64: both planes fold. WN must be 64 (int1's delivery bound: WN*TK*bits >= 4096).
  show("TK=64  F1=2 F2=4 (64,128,64) w64x64", xplane::tile_map_int1<64,128,64,64,64, 4, 2>(), 128*64);
  return 0;
}
