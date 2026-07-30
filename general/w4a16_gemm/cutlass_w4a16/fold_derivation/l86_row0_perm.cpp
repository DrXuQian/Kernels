// L86 -- THE ROW-0 PERMUTATION, exported for the packed-cube writer. Row 0 owns 32 of a cube's 512 words (l84) and
// the cubes can be packed at an 8-word base stride with zero collisions (l85). But the AIU write is .padz -- it
// writes the whole 2048 B cube, zeros included -- so a packed layout would have each cube's zero-fill erase its
// neighbour's row 0. The write therefore has to be a cp.async that touches ONLY row 0's words, which means knowing
// exactly which physical word each LOGICAL k of row 0 lands on.
//
// Read off ppu_tsm_ld_swzl_sim: for row 0 only lanes 0..3 participate (lane_row_idx = lane/4 + coord_h == 0), and
// the logical word index within the row is (v%2)*4 + lane%4 + 8*slice.
#include <cstdio>
#include <set>
#include <map>

static int phys_word(int lane, int v, int slice_idx, int CUBE_H) {
  int slice_word_base   = CUBE_H * 8 * slice_idx;
  int slice_start_vec   = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;
  int lane_row_idx      = lane / 4;                 // coord_h = 0
  int lane_col_idx      = lane % 4;
  int vreg_row_idx      = (v / 2) * 8 + lane_row_idx;
  int vreg_line_idx     = vreg_row_idx / 4;
  int vreg_vec_idx      = (vreg_row_idx % 4) * 2 + (v % 2);
  int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
  int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;
  return slice_word_base + vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
}

int main() {
  const int CUBE_H = 16, SLICES = 4;          // CUBE_W = 64 halfs = 32 words/row -> 4 slices of 8 words
  std::map<int,int> logical_to_phys;          // logical word of row 0 -> physical word in the cube
  for (int lane = 0; lane < 4; ++lane)        // only lane/4 == 0 sees row 0
    for (int v = 0; v < 4; ++v)
      for (int s = 0; s < SLICES; ++s) {
        int vreg_row = (v / 2) * 8 + lane / 4;
        if (vreg_row != 0) continue;          // v/2 must be 0 -> v in {0,1}
        int logical = (v % 2) * 4 + (lane % 4) + 8 * s;
        int phys    = phys_word(lane, v, s, CUBE_H);
        if (logical_to_phys.count(logical) && logical_to_phys[logical] != phys)
          std::printf("  !! logical %d maps to both %d and %d\n", logical, logical_to_phys[logical], phys);
        logical_to_phys[logical] = phys;
      }
  std::printf("row0 logical word -> physical word (%zu entries):\n", logical_to_phys.size());
  for (auto& [l, p] : logical_to_phys) std::printf("  k_word %2d -> phys %3d %s\n", l, p, (l % 4 == 3) ? "" : "");
  // contiguity of the destination: how many runs, and is the SOURCE order preserved inside each run?
  std::set<int> phys;
  for (auto& kv : logical_to_phys) phys.insert(kv.second);
  int runs = 0, prev = -99;
  for (int p : phys) { if (p != prev + 1) ++runs; prev = p; }
  std::printf("destination occupies %zu words in %d contiguous run(s)\n", phys.size(), runs);
  bool order_ok = true;
  for (auto& [l, p] : logical_to_phys) {
    // within a run of 4 (one slice-half), logical k should advance with phys
    auto it = logical_to_phys.find(l + 1);
    if (it != logical_to_phys.end() && (l % 4) != 3 && it->second != p + 1) order_ok = false;
  }
  std::printf("logical k advances with physical word inside each run: %s\n", order_ok ? "YES -> each run is a plain contiguous copy" : "NO -> the writer must permute inside the run");
  return 0;
}
