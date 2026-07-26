// LEG 3 -- what actually separates the nine ppu001 reference points, after the retraction.
//
// The first version of this file tested `TK*Bits >= 128` and reported 0/9 mismatches. True, but misleading: it
// fits the data while blaming the wrong thing. Two predicates are at work, with completely different status.
//
//   P1  OVER-DELIVERY (hardware).  delivery <= slots.  A thread cannot use more codes than its fragment has
//       slots; the surplus is never fetched. Only a bigger TN or TK fixes this.
//
//   P2  WHOLE-WORD PACKER (software).  cols_per_word = 128/(TK*Bits) <= 1.  nfold_regroup_gmem moves whole
//       uint32 words, so at most ONE logical column can live in a word. When a thread's demand needs two
//       columns per word, that packer supplies half the columns -- exactly the measured n_used = n % Ng.
//       A bit-granular offline lifts it.
//
// P1 alone gets 8/9. P1 && P2 gets 9/9. Reporting only the conjunction is what let me write "int1 can never run
// at TK=64" into a header, when the honest statement is "int1 at TK=64 needs a better offline".
//
//   g++ -O2 -std=c++17 leg3_predicate.cpp -o leg3 && ./leg3
#include <cstdio>
#include <initializer_list>

struct Ref { const char* name; int bits, tm, tn, tk; bool works; const char* note; };

int main() {
  Ref refs[] = {
    {"int4", 4, 64,  64,  64,  true,  "55.9% MFU"},
    {"int2", 2, 64,  64,  64,  true,  "53.2% MFU (fold)"},
    {"int2", 2, 64, 128,  64,  true,  "bad=0"},
    {"int2", 2, 64,  64, 128,  true,  "original unfolded path"},
    {"int1", 1, 32, 128, 128,  true,  "bad=0 (MFU unverified)"},
    {"int1", 1, 64,  64, 128,  true,  "bad=0 (MFU unverified)"},
    {"int1", 1, 64,  64, 256,  true,  "original unfolded path"},
    {"int1", 1, 64,  64,  64,  false, "random ~41%"},
    {"int1", 1, 64, 128,  64,  false, "random ~41%"},
  };

  printf("LEG 3 -- two predicates, only one of which is a hardware limit\n\n");
  printf("  %-5s %-14s %6s %6s %8s %5s %6s  %-4s %-4s %-7s %-8s %s\n",
         "type", "tile(M,N,K)", "deliv", "slots", "cols/wd", "cols", "words", "P1", "P2", "P1&&P2", "measured", "note");

  int only_p1 = 0, both = 0;
  for (auto& r : refs) {
    const int slots    = 8 * (r.tn / 32) * (r.tk / 16);
    const int delivery = 16 * 8 / r.bits;
    const int cols     = r.tn / 16;                       // columns one thread demands (cute: partition_B)
    const int words    = r.tn * r.tk * r.bits / 2048;     // 32-bit words it is handed
    const bool p1 = delivery <= slots;                    // hardware
    const bool p2 = 128 <= r.tk * r.bits;                 // whole-word packer, i.e. cols_per_word <= 1
    char tile[24]; snprintf(tile, sizeof tile, "(%d,%d,%d)", r.tm, r.tn, r.tk);
    char cpw[12];  snprintf(cpw, sizeof cpw, "%.2f", 128.0 / (r.tk * r.bits));
    only_p1 += (p1 == r.works);
    both    += ((p1 && p2) == r.works);
    printf("  %-5s %-14s %6d %6d %8s %5d %6d  %-4s %-4s %-7s %-8s %s%s\n",
           r.name, tile, delivery, slots, cpw, cols, words,
           p1 ? "ok" : "BAD", p2 ? "ok" : "BAD", (p1 && p2) ? "ok" : "BAD",
           r.works ? "ok" : "BAD", r.note,
           (p1 != r.works && (p1 && p2) == r.works) ? "   <-- only P2 fails here" : "");
  }
  printf("\n  P1 alone : %d/9 agree\n  P1 && P2 : %d/9 agree\n", only_p1, both);

  printf("\n  The row where they differ is int1 (64,128,64): 128 codes into 128 slots, so the COUNTING is exact\n"
         "  and a valid placement exists. Its thread wants 8 columns and gets 4 words, so each word must carry\n"
         "  TWO columns -- which a uint32-granular packer cannot express. Nothing in the hardware forbids it.\n");

  printf("\n  cols-per-word by (width, TK). >1 is where a bit-granular offline is required:\n    %-7s", "");
  for (int tk : {32, 64, 128, 256}) printf("  TK=%-5d", tk);
  printf("\n");
  for (int bits : {4, 2, 1}) {
    printf("    int%-4d", bits);
    for (int tk : {32, 64, 128, 256}) printf("  %-8.2f", 128.0 / (tk * bits));
    printf("\n");
  }
  printf("    (0.50 is the opposite case -- one column spanning two words -- which int4 already does today)\n");
  return 0;
}
