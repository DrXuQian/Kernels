// L85 -- CAN THE CUBES OVERLAP? Row 0 owns only 32 of a cube's 512 words (l84). The other 480 must be READABLE, not
// A's. So pack the cubes by shrinking the BASE stride -- CUBE_H*CUBE_W (512 words) down to CUBE_W (32) -- while the
// cube GEOMETRY stays 16x64 so the swizzle and the write/read pairing are untouched. That is the opposite of
// PPU_A_CUBE_MN, which changed the geometry and corrupted everything.
//
// Checks: do any two (cube, stage) row-0 runs collide, and what is the total span.
#include <cstdio>
#include <set>
#include <map>

static int word_of(int lane, int v, int coord_h, int slice_idx, int CUBE_H, int* out_row) {
  int slice_word_base   = CUBE_H * 8 * slice_idx;
  int slice_start_vec   = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;
  int lane_row_idx      = lane / 4 + coord_h;
  int lane_col_idx      = lane % 4;
  int vreg_row_idx      = (v / 2) * 8 + lane_row_idx;
  int vreg_line_idx     = vreg_row_idx / 4;
  int vreg_vec_idx      = (vreg_row_idx % 4) * 2 + (v % 2);
  int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
  int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;
  *out_row = vreg_row_idx;
  return slice_word_base + vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
}

// row-0 words of one cube, relative to its own base
static std::set<int> row0_words(int CUBE_H, int SLICES) {
  std::set<int> w;
  for (int lane = 0; lane < 32; ++lane)
    for (int v = 0; v < 4; ++v)
      for (int s = 0; s < SLICES; ++s) { int r; int o = word_of(lane, v, 0, s, CUBE_H, &r); if (r == 0) w.insert(o); }
  return w;
}
static int cube_span(int CUBE_H, int SLICES) {
  int mx = 0;
  for (int lane = 0; lane < 32; ++lane)
    for (int v = 0; v < 4; ++v)
      for (int s = 0; s < SLICES; ++s) { int r; int o = word_of(lane, v, 0, s, CUBE_H, &r); if (o > mx) mx = o; }
  return mx + 1;
}

int main() {
  const int CUBE_H = 16, CUBE_W_HALFS = 64, SLICES = CUBE_W_HALFS * 16 / 32 / 8;   // 32 words/row / 8 = 4
  auto r0 = row0_words(CUBE_H, SLICES);
  int span = cube_span(CUBE_H, SLICES);
  std::printf("one cube: span %d words (%d B), row0 owns %zu words (%d B) = 1/%.0f\n",
              span, span * 4, r0.size(), int(r0.size()) * 4, double(span) / r0.size());

  // pack N sub-tiles (cubes x stages) at BASE_STRIDE words apart; a collision means two of them need the same word
  for (int nsub : {8, 6, 4}) {
    for (int stride : {512, 128, 64, 32, 16, 8}) {
      std::map<int,int> owner;  int clash = 0, hi = 0;
      for (int i = 0; i < nsub; ++i)
        for (int w : r0) {
          int a = i * stride + w;
          if (owner.count(a) && owner[a] != i) ++clash;
          owner[a] = i;
          if (a > hi) hi = a;
        }
      int total_span = hi + 1 + (span - 1 - *r0.rbegin());   // the last sub-tile still READS to its cube end
      std::printf("  %d sub-tiles, base stride %3d words: %s  total read span %5d words (%6d B)%s\n",
                  nsub, stride, clash ? "COLLIDE" : "clean  ", total_span, total_span * 4,
                  clash ? "" : "  <-- usable");
    }
    std::printf("  (for reference, no overlap = %d x %d = %d words, %d B)\n\n",
                nsub, span, nsub * span, nsub * span * 4);
  }
  return 0;
}
