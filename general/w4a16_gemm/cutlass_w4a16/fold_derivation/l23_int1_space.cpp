// L23 -- enumerate int1's reachable config space and predict, so a sweep is informed rather than brute force.
//
// Facts the enumeration rests on, all measured:
//   * delivery bound      WN*TK*Bits >= 4096          (l5_slots, 9/9 on the reference points)
//   * placement is WN-INVARIANT for TK in {128,256}    (l20: shipped offline bit-identical at WN=32 and WN=64)
//   * TK=64 needs the bit-granular packer              (l20: 11264/16384 differ from the shipped offline)
//   * scale reloads over K = K/gs, INDEPENDENT of TK   (so TK is not a scale lever -- the box overturned my model)
//   * hiding improves with atoms per iteration = TK/16 (gs=16 puts a reload on every atom, APG = gs/16 = 1)
//   * smem/stage = TM*TK*2 (A) + TN*TK*Bits/8 (B, F cancels) + TN*SK*2 (scale, x2 with zero)
//
// MEASURED BASELINES to beat: int1 ScaleOnly gs=32  TK=128/WN=32 = 42.0%,  TK=64/WN=64 = 46.4%
//   g++ -O2 -std=c++17 l23_int1_space.cpp -o l23 && ./l23
#include <cstdio>
#include <vector>
#include <algorithm>
#include <initializer_list>

struct Cfg { int tm, tn, tk, wm, wn, st; int warps, blocks, smem, atoms; bool bitpack; };

int main() {
  const int Bits = 1, gs = 32, SK = 0;
  printf("L23 -- int1 reachable configs, ranked by predicted occupancy (gs=32, ScaleOnly)\n\n");
  std::vector<Cfg> v;
  for (int tm : {16, 32, 64})
    for (int tn : {64, 128, 256})
      for (int tk : {64, 128, 256})
        for (int wn : {32, 64, 128})
          for (int st : {2, 3, 4}) {
            const int wm = 32;
            if (tm % wm || tn % wn) continue;
            if ((long)wn * tk * Bits < 4096) continue;          // delivery bound
            const int warps = (tm / wm) * (tn / wn);
            if (warps < 1 || warps > 16) continue;
            const int sk = tk / gs;
            const int smem = (tm * tk * 2 + tn * tk * Bits / 8 + tn * sk * 2) * st;
            if (smem > 262144) continue;
            const int blocks = std::min(262144 / smem, 64 / warps);
            if (blocks < 1) continue;
            v.push_back({tm, tn, tk, wm, wn, st, warps, blocks, smem, tk / 16, tk == 64});
          }
  std::stable_sort(v.begin(), v.end(), [](const Cfg& a, const Cfg& b) {
    if (a.blocks != b.blocks) return a.blocks > b.blocks;
    return a.atoms > b.atoms;                                   // tie-break on hiding
  });
  printf("  %-16s %-6s %-5s %-4s %-7s %-8s %-7s %s\n",
         "tile(M,N,K)", "warp", "st", "warps", "smem", "blocks", "atoms", "offline");
  int shown = 0;
  for (auto& c : v) {
    if (shown++ >= 22) break;
    printf("  (%3d,%3d,%3d)    32x%-3d %-5d %-4d %-7d %-8d %-7d %s\n",
           c.tm, c.tn, c.tk, c.wn, c.st, c.warps, c.smem, c.blocks, c.atoms,
           c.bitpack ? "needs bitpack (TN=128 only today)" : "shipped offline works");
  }
  printf("\n  the two measured points, for calibration of the ranking:\n");
  for (auto& c : v) {
    if (c.tm==32 && c.tn==128 && c.tk==128 && c.wn==32 && c.st==3)
      printf("    (32,128,128) 32x32 s3 : blocks=%d atoms=%d  -> MEASURED 42.0%%\n", c.blocks, c.atoms);
    if (c.tm==64 && c.tn==128 && c.tk==64 && c.wn==64 && c.st==3)
      printf("    (64,128, 64) 32x64 s3 : blocks=%d atoms=%d  -> MEASURED 46.4%%\n", c.blocks, c.atoms);
  }
  printf("\n  Ranking by occupancy alone is a weak predictor -- the two measured points differ by 4.4 points and the\n");
  printf("  model has to explain that from blocks and atoms only. Treat this as a shortlist, not a prediction.\n");
  return 0;
}
