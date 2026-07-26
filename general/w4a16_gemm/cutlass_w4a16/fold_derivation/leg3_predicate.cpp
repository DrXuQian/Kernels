// LEG 3: the predicate that LEG 1 + LEG 2 imply, checked against every box reference point.
//
// LEG 1 (leg1_runword.cpp, exact copy of ppu_tsm_ld_swzl_sim SWAP=true):
//     run-word = (v%2)*4 + (lane%4)     -- the 32B run is a 2x4 grid: v%2 picks a 4-word HALF,
//                                          lane%4 picks the word inside that half.
// LEG 2 (leg2_frag.cu, cute partition_B of the real TiledMMA):
//     the fragment's N depends on lane/4 and the value bits ONLY; lane%4 moves K (stride 2), never N.
//
// THEOREM. All four lanes of a lane/4 group demand the SAME N, and they receive the four different words of a
// half. So every word of a half must carry the same logical column => a folded column occupies EXACTLY 4 words
// = 16 bytes. A 32B run therefore holds at most 2 columns:
//         F <= 2      <=>      TK * Bits in {128 (F=2), 256 (F=1)}
// This is a property of the swzl delivery grid vs the mma fragment. It is NOT a converter limitation, so no
// converter rewrite can lift it.
//   g++ -O2 -std=c++17 leg3_predicate.cpp -o leg3 && ./leg3
#include <cstdio>
#include <initializer_list>

struct Ref { const char* name; int bits, tm, tn, tk; bool works; const char* note; };

int main() {
  // every configuration ever run on ppu001, with its measured verdict
  Ref refs[] = {
    {"int4", 4, 64,  64,  64,  true,  "55.9% MFU"},
    {"int2", 2, 64,  64,  64,  true,  "53.2% MFU (fold)"},
    {"int2", 2, 64, 128,  64,  true,  "bad=0"},
    {"int2", 2, 64,  64, 128,  true,  "original unfolded path"},
    {"int1", 1, 32, 128, 128,  true,  "54.3% MFU (fold, winner)"},
    {"int1", 1, 64,  64, 128,  true,  "36.4% MFU"},
    {"int1", 1, 64,  64, 256,  true,  "original unfolded path"},
    {"int1", 1, 64,  64,  64,  false, "measured random ~41%"},
    {"int1", 1, 64, 128,  64,  false, "measured random ~41%"},
  };

  printf("LEG 3 -- predicate  words_per_column == 4  (i.e. TK*Bits in {128,256}, i.e. F <= 2)\n\n");
  printf("  %-5s %-14s %5s %6s %8s %6s  %-9s %-9s %s\n",
         "type", "tile(M,N,K)", "F", "run B", "col words", "TK*b", "predicts", "measured", "note");

  int wrong = 0;
  for (auto& r : refs) {
    int contig  = r.tk * r.bits / 8;                 // bytes one column contributes along K
    int F       = contig >= 32 ? 1 : 32 / contig;    // columns folded into the 32B run
    int colw    = r.tk * r.bits / 32;                // uint32 words one folded column occupies
    bool pred   = (colw == 4 || colw == 8);          // 4 words (F=2) or the whole run (F=1)
    char tile[24]; snprintf(tile, sizeof tile, "(%d,%d,%d)", r.tm, r.tn, r.tk);
    bool agree = (pred == r.works);
    if (!agree) ++wrong;
    printf("  %-5s %-14s %5d %6d %8d %6d  %-9s %-9s %s%s\n",
           r.name, tile, F, contig * F, colw, r.tk * r.bits,
           pred ? "OK" : "BAD", r.works ? "OK" : "BAD", r.note, agree ? "" : "   <-- MISMATCH");
  }
  printf("\n  mismatches: %d / %d\n", wrong, int(sizeof(refs)/sizeof(refs[0])));

  printf("\n  reachable TK per width (the ONLY legal fold points):\n");
  for (int bits : {8, 4, 2, 1})
    printf("    int%-2d : TK = %3d (F=1, run is one column)   TK = %3d (F=2, two columns of 16B)\n",
           bits, 256 / bits, 128 / bits);
  printf("\n  => int1's smallest legal TK is 128. TK=64 needs F=4, which asks the fragment for an N that varies\n");
  printf("     with lane%%4 -- LEG 2 says that does not exist. A 4-way converter cannot help: the data for the\n");
  printf("     other two columns is delivered to the WRONG LANES, and a converter only relabels registers\n");
  printf("     inside one thread.\n");
  return 0;
}
