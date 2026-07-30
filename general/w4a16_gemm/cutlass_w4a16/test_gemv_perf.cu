// Decode-band performance for the low-bit GEMV: the CUDA-CORE-ONLY winner. main() only -- every config is a
// separate generated translation unit so `make -j` compiles them concurrently (see gemv_perf_common.hpp).
//
// TWO SHAPE FAMILIES, because they have DIFFERENT parallelism knobs and conflating them hid a factor of 8
// earlier in this work:
//
//   * MoE decode (L experts x 1 row). grid = 1 x n/CtaN x L. The expert dimension multiplies the grid, so at
//     L=8, n=2048, CtaN=8 this is 2048 CTAs x 128 threads.
//   * DENSE decode (one matrix, m rows). grid = ceil(m/CtaM) x n/CtaN x 1. There is NO expert dimension, so
//     at m=1, n=2048, CtaN=8 it is 256 CTAs = 1024 warps = 14.2 warps/CU -- the SAME warp count as the
//     grouped GEMM's decode launch. For dense, parallelism has to be bought with CtaN, and the price is A
//     re-read n/CtaN times (4 KB at m=1, so it should live in L2 -- a prediction to check, not an assumption).
//
// Build: TARGET=test_gemv_perf ./build.sh      Run: $BIN/test_gemv_perf [shape index]
//   GEMV_ONLY=<substring>  run only rows whose tag contains this (one row -> one launch, for acu)
//   GEMV_ACU=1             ONE COLD launch per row, no warmup, no repeat -- a capture, NOT a timing
#include "gemv_perf_common.hpp"
#include "gemv_perf_units.inc"     // GENERATED: unit declarations + gemv_run_all()

int main(int argc, char** argv) {
  int const only_shape = (argc > 1) ? std::atoi(argv[1]) : -1;

  std::printf("== low-bit GEMV, decode band (CUDA-CORE-ONLY winner) ==\n");
  std::printf("   rev %d, %d generated units, %d format groups | HBM peak %.0f GB/s, %d CU, CtaM max %d\n",
              GEMV_PERF_REV, GEMV_UNIT_COUNT, GEMV_GROUP_COUNT, HBM_GBS, CU, GEMV_CTAM_MAX);
  if (only_filter()) std::printf("   GEMV_ONLY=\"%s\"\n", only_filter());
  if (std::getenv("GEMV_FMT")) std::printf("   GEMV_FMT=\"%s\"\n", std::getenv("GEMV_FMT"));
  if (std::getenv("GEMV_CFG")) std::printf("   GEMV_CFG=\"%s\"\n", std::getenv("GEMV_CFG"));
  if (acu_mode()) std::printf("   *** GEMV_ACU: ONE COLD LAUNCH PER ROW. These are captures, not timings. ***\n");

  Shape const shapes[] = {
    {"MoE  L=8  x1 row  N=K=2048", 8, 1, 2048, 2048, 32, QuantOp::FinegrainedScaleZero},
    {"MoE  L=8  x1 row  N=K=4096", 8, 1, 4096, 4096, 32, QuantOp::FinegrainedScaleZero},
    {"MoE  L=64 x1 row  N=K=2048", 64, 1, 2048, 2048, 32, QuantOp::FinegrainedScaleZero},
    {"dense m=1         N=K=2048", 0, 1, 2048, 2048, 32, QuantOp::FinegrainedScaleZero},
    {"dense m=1         N=K=4096", 0, 1, 4096, 4096, 32, QuantOp::FinegrainedScaleZero},
    {"dense m=1  N=12288 K=4096",  0, 1, 12288, 4096, 128, QuantOp::FinegrainedScaleOnly},
    {"dense m=4         N=K=2048", 0, 4, 2048, 2048, 32, QuantOp::FinegrainedScaleZero},
    {"dense m=8         N=K=2048", 0, 8, 2048, 2048, 32, QuantOp::FinegrainedScaleZero},
  };
  int const ns = int(sizeof(shapes) / sizeof(shapes[0]));

  // One Best per format GROUP (format x layout), so the verdict is per thing a caller would actually pick.
  std::vector<std::vector<Best>> bests(ns, std::vector<Best>(GEMV_GROUP_COUNT));
  std::vector<Best> overall(ns);

  for (int i = 0; i < ns; ++i) {
    if (only_shape >= 0 && i != only_shape) continue;
    Shape const& sh = shapes[i];
    int const experts = sh.experts > 0 ? sh.experts : 1;
    int const sk = sh.K / sh.gs;
    double const floor_b = double(experts) * (double(sh.N) * sh.K * 4 / 8.0 + double(sk) * sh.N * 4.0)
                         + double(experts) * sh.rows * sh.K * 2.0 + double(experts) * sh.rows * sh.N * 2.0;
    std::printf("\n-- [%d] %s  gs=%d %s --\n     int4 memory roof: %.2f us (%.2f MB at %.0f GB/s)\n",
                i, sh.name, sh.gs, name_of(sh.quant), floor_b / (HBM_GBS * 1e9) * 1e6, floor_b / 1e6, HBM_GBS);
    gemv_run_all(sh, bests[i].data());
    for (auto const& b : bests[i]) if (b.us < overall[i].us) overall[i] = b;
    if (!acu_mode() && overall[i].us < 1e29)
      std::printf("     WINNER %s  %.2f us  (%.1f%% HBM)\n", overall[i].tag, overall[i].us, overall[i].pct);
  }

  if (!acu_mode()) {
    std::printf("\n== per-shape, per-format winners ==\n");
    for (int i = 0; i < ns; ++i) {
      if (only_shape >= 0 && i != only_shape) continue;
      std::printf("  [%d] %s\n", i, shapes[i].name);
      std::vector<int> ord(GEMV_GROUP_COUNT);
      for (int g = 0; g < GEMV_GROUP_COUNT; ++g) ord[g] = g;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) { return bests[i][a].us < bests[i][b].us; });
      for (int g : ord)
        if (bests[i][g].us < 1e29)
          std::printf("        %-12s %-34s %8.2f us  %5.1f%% HBM%s\n", gemv_group_names[g], bests[i][g].tag,
                      bests[i][g].us, bests[i][g].pct, g == ord[0] ? "  <-- fastest" : "");
    }
    std::printf("\n  To profile one: GEMV_ONLY=\"<tag substring>\" GEMV_ACU=1 acu -o gemv.report --set full -f "
                "$BIN/test_gemv_perf <shape index>\n");
  }
  std::printf("\n  launches refused: %d\n", gemv_fail_count());
  return 0;
}
