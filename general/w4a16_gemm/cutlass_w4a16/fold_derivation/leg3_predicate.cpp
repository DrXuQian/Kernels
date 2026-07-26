// LEG 3 -- what separates the nine ppu001 reference points. Third and, I believe, final version.
//
// v1 tested TK*Bits >= 128 and called it a structural limit on the fold. v2 retracted that and blamed the offline
// packer. Both were partly wrong, because both used a STUB warp layout instead of the builder's real one. With
// the real TiledMma (WarpOnN = TN/WN, PermN = WarpOnN*16) the per-thread fragment is
//
//     slots = WN * TK / 32          -- measured over 12 configs in l5_slots.cu; note it does NOT contain TN,
//                                      because B is split across the warps in N
//
// and then ONE predicate separates all nine points, 9/9:
//
//     delivery <= slots      i.e.   WN * TK * Bits >= 4096
//
// At the WM=WN=32 that every fold test passes, that reduces to TK*Bits >= 128 -- numerically v1's predicate. So
// v1 had the right inequality for the wrong reason, and v2 threw away a correct bound. The reason matters because
// it says where the escape is: slots scales with **WN**, not with TN. At WN=64, int1 at TK=64 becomes feasible.
//
// The price of that escape is the thing v2 discovered, which is real but only bites once WN > 32:
//
//     cols_per_word = WN / 32       -- how many logical columns must share one 32-bit word
//
// nfold_regroup_gmem moves whole uint32s, so it can only do 1. int1 at TK=64 therefore needs BOTH a higher WN and
// a bit-granular packer; the two constraints pincer it.
//
//   g++ -O2 -std=c++17 leg3_predicate.cpp -o leg3 && ./leg3
#include <cstdio>
#include <initializer_list>
#include <string>

struct Ref { const char* name; int bits, tm, tn, tk, wm, wn; bool works; const char* note; };

int main() {
  Ref refs[] = {
    {"int4", 4, 64,  64,  64, 32, 32, true,  "55.9% MFU"},
    {"int2", 2, 64,  64,  64, 32, 32, true,  "53.2% MFU (fold)"},
    {"int2", 2, 64, 128,  64, 32, 32, true,  "bad=0"},
    {"int2", 2, 64,  64, 128, 32, 32, true,  "original unfolded path"},
    {"int1", 1, 32, 128, 128, 32, 32, true,  "bad=0 (MFU unverified)"},
    {"int1", 1, 64,  64, 128, 32, 32, true,  "bad=0 (MFU unverified)"},
    {"int1", 1, 64,  64, 256, 32, 32, true,  "original unfolded path"},
    {"int1", 1, 64,  64,  64, 32, 32, false, "random ~41%"},
    {"int1", 1, 64, 128,  64, 32, 32, false, "random ~41%"},
  };

  printf("LEG 3 -- one predicate, measured slots, 9/9\n\n");
  printf("  %-5s %-14s %-6s %6s %6s %-14s %5s %5s %8s %-9s %s\n",
         "type", "tile(M,N,K)", "warp", "deliv", "slots", "verdict", "cols", "k", "cols/wd", "measured", "note");

  int agree = 0;
  for (auto& r : refs) {
    const int slots    = r.wn * r.tk / 32;              // MEASURED formula (l5_slots.cu), independent of TN
    const int delivery = 16 * 8 / r.bits;
    const int cols     = r.wn / 8;
    const int kpc      = r.tk / 4;
    const int cpw      = r.wn / 32;
    const bool ok = delivery <= slots;
    char tile[24]; snprintf(tile, sizeof tile, "(%d,%d,%d)", r.tm, r.tn, r.tk);
    char warp[12]; snprintf(warp, sizeof warp, "%dx%d", r.wm, r.wn);
    agree += (ok == r.works);
    printf("  %-5s %-14s %-6s %6d %6d %-14s %5d %5d %8d %-9s %s\n",
           r.name, tile, warp, delivery, slots,
           delivery > slots ? "OVER-DELIVERY" : (delivery == slots ? "tight" : "under"),
           cols, kpc, cpw, r.works ? "ok" : "BAD", r.note);
  }
  printf("\n  over-delivery alone: %d/9 agree\n", agree);

  printf("\n  the escape -- same TK=64 that fails at WN=32, at higher WN:\n");
  for (int wn : {32, 64, 128}) {
    const int slots = wn * 64 / 32, deliv = 128, cpw = wn / 32;
    printf("    int1 TK=64 WN=%-3d : slots=%3d deliv=%3d %-14s cols/word=%d %s\n", wn, slots, deliv,
           deliv > slots ? "OVER-DELIVERY" : (deliv == slots ? "tight" : "under"), cpw,
           deliv <= slots ? (cpw > 1 ? "-- feasible, needs the bit-granular packer" : "-- feasible as is")
                          : "-- impossible");
  }

  printf("  smallest legal TK, by width and WN (WN*TK*Bits >= 4096):\n    %-8s", "");
  for (int wn : {32, 64, 128}) printf("  WN=%-6d", wn);
  printf("\n");
  for (int bits : {4, 2, 1}) {
    printf("    int%-5d", bits);
    for (int wn : {32, 64, 128}) {
      int tk = 4096 / (wn * bits);
      printf("  %-8s", tk < 16 ? "16" : (std::to_string(tk)).c_str());
    }
    printf("\n");
  }
  printf("    (WN>32 costs cols_per_word = WN/32 > 1, i.e. the bit-granular packer)\n");
  return 0;
}
