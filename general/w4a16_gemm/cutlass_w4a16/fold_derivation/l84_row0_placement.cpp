// L82 -- CAN ONE swzl READ TOUCH ONE ROW? Host replay of ppu_tsm_ld_swzl_sim's arithmetic, verbatim, so the answer
// comes from the same object LogicalTV was derived from (validated 0-mismatch against hardware in l2l3/l17/l7/l10/
// l12/l13/l16). The question this settles: is the 16-row span a parameter (fixable by cute-ifying the addressing)
// or the instruction's lane/vreg structure (not fixable at all)?
#include <cstdio>
#include <set>
#include <cstdlib>

// verbatim from cute/arch/copy_ppu0010_aiu.hpp: ppu_tsm_ld_swzl_sim, SWAP=true branch (fp16 A)
static int word_of(int lane, int v, int coord_h, int slice_idx, int CUBE_H, int* out_row) {
  int slice_word_base   = CUBE_H * 8 * slice_idx;                 // CUBE_H appears ONLY here (and in stage)
  int slice_start_vec   = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;
  int lane_row_idx      = lane / 4 + coord_h;
  int lane_col_idx      = lane % 4;
  int vreg_row_idx      = (v / 2) * 8 + lane_row_idx;              // SWAP=true
  int vreg_line_idx     = vreg_row_idx / 4;
  int vreg_vec_idx      = (vreg_row_idx % 4) * 2 + (v % 2);
  int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
  int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;
  *out_row = vreg_row_idx;
  return slice_word_base + vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
}

// WHICH WORDS BELONG TO ROW 0? If they are one contiguous run, a 16-row cube can be backed by one row of storage
// plus 15 rows of anything. If they are scattered at the slice stride, the cube span must be allocated even though
// only 1/16 of it carries A.
static void row0_map(int CUBE_H, int SLICES) {
  std::set<int> w0, wall;
  for (int lane = 0; lane < 32; ++lane)
    for (int v = 0; v < 4; ++v)
      for (int s = 0; s < SLICES; ++s) {
        int row; int w = word_of(lane, v, 0, s, CUBE_H, &row);
        wall.insert(w);
        if (row == 0) w0.insert(w);
      }
  std::printf("CUBE_H=%-2d  row0 owns %2zu of %3zu words;  row0 offsets:", CUBE_H, w0.size(), wall.size());
  int prev = -99, runs = 0;
  for (int w : w0) { if (w != prev + 1) { ++runs; std::printf(" |"); } std::printf(" %d", w); prev = w; }
  std::printf("   -> %d contiguous run(s)\n", runs);
}

int main() {
  row0_map(16, 4);
  row0_map(16, 1);
  const int SLICES = 4;   // CUBE_W = 64 halfs = 128 B = 32 words -> 4 slices of 32 B
  for (int CUBE_H : {16, 1}) {
    std::set<int> rows, words;
    for (int lane = 0; lane < 32; ++lane)
      for (int v = 0; v < 4; ++v)
        for (int s = 0; s < SLICES; ++s) {
          int row; int w = word_of(lane, v, /*coord_h=*/0, s, CUBE_H, &row);
          rows.insert(row); words.insert(w);
        }
    std::printf("CUBE_H=%-2d  distinct ROWS touched = %-3zu  distinct 32-bit WORDS = %-4zu  word range [%d, %d]\n",
                CUBE_H, rows.size(), words.size(), *words.begin(), *words.rbegin());
  }
  // does CUBE_H change WHICH rows a given (lane,vreg) reads?
  int diff = 0;
  for (int lane = 0; lane < 32; ++lane)
    for (int v = 0; v < 4; ++v) {
      int r16, r1;
      word_of(lane, v, 0, 0, 16, &r16);
      word_of(lane, v, 0, 0, 1,  &r1);
      if (r16 != r1) ++diff;
    }
  std::printf("(lane,vreg) -> row differs between CUBE_H 16 and 1 in %d of 128 cases\n", diff);
  // and how many words collide once the slice stride shrinks?
  for (int CUBE_H : {16, 1}) {
    std::set<int> w;
    int total = 0;
    for (int lane = 0; lane < 32; ++lane)
      for (int v = 0; v < 4; ++v)
        for (int s = 0; s < SLICES; ++s) { int r; w.insert(word_of(lane,v,0,s,CUBE_H,&r)); ++total; }
    std::printf("CUBE_H=%-2d  %d reads land on %zu distinct words -> %d ALIASED reads\n",
                CUBE_H, total, w.size(), total - int(w.size()));
  }
  return 0;
}
