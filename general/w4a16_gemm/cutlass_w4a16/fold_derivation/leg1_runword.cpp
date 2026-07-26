// LEG 1: what LOGICAL word of a row's 32B run does thread (lane, vreg) receive?
//
// Faithful copy of ppu_tsm_ld_swzl_sim's SWAP=true branch (copy_ppu0010_aiu.hpp:365-413) -- SWAP=true is what the
// B operand instantiates and is the hardware-calibrated variant (see the w2a16 cube-width bug: hardware delivers
// [low N | high N] = src0,1 low / src2,3 high, which is exactly SWAP=true's vreg_row_idx = (v/2)*8 + ...).
//
// The AIU write .swzl and this read .swzl are a matched HARDWARE pair, so the two swizzle terms
// (^ vreg_line_idx%2, + slice_start_vec) cancel with the write. Strip them and what is left is the LOGICAL word
// index inside the cube -- i.e. the gmem word, which is what the offline relayout controls.
//   g++ -O2 -std=c++17 leg1_runword.cpp -o leg1 && ./leg1
#include <cstdio>
#include <set>
#include <map>

int main() {
  const int W_ROW = 8;   // words per row per slice: kCon=256 codes at 32 codes/word (int1) = 32 B = one AIU run

  printf("LEG 1 -- logical run-word delivered to (lane, v), SWAP=true\n");
  printf("  physical: line*32 + vec_swz*4 + lane%%4    logical (swizzles cancelled): line*32 + vec*4 + lane%%4\n\n");

  std::map<int, std::set<int>> word_of_vmod2;      // v%2 -> set of run-words seen
  std::map<int, std::set<int>> word_of_lanemod4;   // lane%4 -> set of run-words seen
  bool row_ok = true, word_ok = true;

  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 4; ++v) {
      int lane_row_idx = lane / 4;                        // coord_h = 0
      int vreg_row_idx = (v / 2) * 8 + lane_row_idx;      // SWAP=true
      int vreg_line_idx = vreg_row_idx / 4;
      int vreg_vec_idx  = (vreg_row_idx % 4) * 2 + (v % 2);  // SWAP=true
      int lane_col_idx  = lane % 4;
      int logical_u32   = vreg_line_idx * 32 + vreg_vec_idx * 4 + lane_col_idx;

      int row  = logical_u32 / W_ROW;
      int word = logical_u32 % W_ROW;

      // the two claims
      if (row  != vreg_row_idx)                      row_ok  = false;
      if (word != (v % 2) * 4 + (lane % 4))          word_ok = false;

      word_of_vmod2[v % 2].insert(word);
      word_of_lanemod4[lane % 4].insert(word);

      if (lane < 8 && v < 4)
        printf("  lane %2d v%d -> u32 %3d = row %2d, run-word %d\n", lane, v, logical_u32, row, word);
    }
  }

  printf("\n  CLAIM A  row      == (v/2)*8 + lane/4      : %s\n", row_ok  ? "HOLDS for all 128 (lane,v)" : "FAILS");
  printf("  CLAIM B  run-word == (v%%2)*4 + lane%%4      : %s\n", word_ok ? "HOLDS for all 128 (lane,v)" : "FAILS");

  printf("\n  run-words reachable per v%%2   (v%%2 partitions the run into halves):\n");
  for (auto& [k, s] : word_of_vmod2) { printf("    v%%2=%d : {", k); for (int w : s) printf(" %d", w); printf(" }\n"); }
  printf("  run-words reachable per lane%%4 (lane%%4 picks one word inside each half):\n");
  for (auto& [k, s] : word_of_lanemod4) { printf("    lane%%4=%d : {", k); for (int w : s) printf(" %d", w); printf(" }\n"); }

  printf("\n  => the 8-word run is addressed as a 2x4 grid: (v%%2) selects a 4-word HALF, (lane%%4) selects the\n");
  printf("     word inside that half. Nothing else reaches a run-word.\n");
  return 0;
}
