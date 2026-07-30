// Grouped mixed-input GEMM at splitk=1 and splitk>1: main() only. Every config row is a separate generated
// translation unit so `make -j` compiles them concurrently -- see moe_splitk_bench_common.hpp.
//
// Build: TARGET=test_moe_splitk_bench ./build.sh
// Run:   $BIN/test_moe_splitk_bench [L] [Rows] [N] [K] [gs] [mode]
//          mode 3 = DECODE batch=1 (default), 2 = skewed prefill band, 0 = uniform
//   SPLITK_ONLY=<substring>  run only rows whose tag contains this
//   SPLITK_ACU=1             ONE COLD launch per row (a capture, not a timing)
#include "moe_splitk_bench_common.hpp"
#include "moe_splitk_units.inc"     // GENERATED: unit declarations + splitk_run_all()

int main(int argc, char** argv) {
  Band bd{};
  bd.L    = argc > 1 ? atoi(argv[1]) : 8;
  bd.Rows = argc > 2 ? atoi(argv[2]) : 1;
  bd.N    = argc > 3 ? atoi(argv[3]) : 2048;
  bd.K    = argc > 4 ? atoi(argv[4]) : 2048;
  bd.gs   = argc > 5 ? atoi(argv[5]) : 32;
  bd.mode = argc > 6 ? atoi(argv[6]) : 3;     // decode batch=1 by default: that is the band in question
  bd.scale_k = bd.K / bd.gs;

  bd.me.resize(bd.L); bd.offs.resize(bd.L);
  bd.total = 0; bd.Mmax = 0; bd.active = 0;
  for (int e = 0; e < bd.L; ++e) {
    if (bd.mode == 0)      bd.me[e] = bd.Rows;
    // MODE 3: `Rows` IS THE TOP-K, NOT ROWS PER EXPERT. Same convention as test_lowbit_moe_bench.cu, and the
    // trap that produced a 1-expert measurement: `8 1 ...` reads like "8 experts, 1 row each" and means "8
    // experts, 1 of them active". The recorded decode band is L=64 with top-k=8, i.e. `64 8 ...`.
    else if (bd.mode == 3) bd.me[e] = (e < bd.Rows) ? 1 : 0;
    else {
      unsigned h = (unsigned)e * 2654435761u >> 13;
      if ((h % 8) == 0)      bd.me[e] = 0;
      else if ((h % 8) == 1) bd.me[e] = int(bd.Rows * 3 + (h % 37));
      else                   bd.me[e] = int(bd.Rows / 2 + (h % (unsigned)(bd.Rows + 1)));
    }
    if (bd.me[e]) ++bd.active;
    bd.offs[e] = bd.total; bd.total += bd.me[e]; bd.Mmax = std::max(bd.Mmax, bd.me[e]);
  }
  if (bd.Mmax == 0) { std::printf("all experts empty\n"); return 1; }
  if (bd.mode == 3 && bd.Rows > bd.L) {
    std::printf("mode 3: top-k=%d exceeds L=%d\n", bd.Rows, bd.L); return 1;
  }

  std::printf("== grouped mixed-input GEMM: splitk=1 vs splitk>1 ==\n");
  std::printf("   L=%d rows=%d mode=%d N=%d K=%d gs=%d | total=%d Mmax=%d active=%d | HBM %.0f GB/s\n",
              bd.L, bd.Rows, bd.mode, bd.N, bd.K, bd.gs, bd.total, bd.Mmax, bd.active, HBM_GBS);
  if (sk_only()) std::printf("   SPLITK_ONLY=\"%s\"\n", sk_only());
  if (std::getenv("SPLITK_CFG")) std::printf("   SPLITK_CFG=\"%s\"\n", std::getenv("SPLITK_CFG"));
  if (std::getenv("SPLITK_S"))   std::printf("   SPLITK_S=\"%s\"\n", std::getenv("SPLITK_S"));
  if (sk_acu())  std::printf("   *** SPLITK_ACU: ONE COLD LAUNCH PER ROW. Captures, not timings. ***\n");

  // A GRID BELOW ONE WAVE CANNOT ANSWER ANYTHING, so this REFUSES rather than warns. A printed warning was not
  // enough: the run that prompted it printed this very banner, named the corrected command, and was then
  // re-issued unchanged -- so a verdict table appeared for a 1-expert problem twice. Three rounds of this
  // session have now been spent on numbers taken from a grid too small to measure (a dense m=8 ladder at 64
  // CTAs, acu DRAM 4.43%, withdrawn; then this shape twice).
  //
  // The bound is computed from the LARGEST TileM and SMALLEST TileN actually swept, so it is a property of the
  // sweep and not a guess: if even the widest row cannot fill 72 CUs at S=1, no row can.
  {
    int mt_min = 0;
    for (int e = 0; e < bd.L; ++e) mt_min += (bd.me[e] + 63) / 64;
    int const cta_best = mt_min * ((bd.N + 31) / 32);
    if (bd.active < 2 || cta_best < 72) {
      std::printf("\n  *** REFUSING TO RUN: THIS GEOMETRY CANNOT ANSWER ANYTHING ***\n"
                  "      active experts = %d, and the widest row here launches only %d CTAs on 72 CUs at S=1,\n"
                  "      so every number would be latency on an empty machine.\n\n"
                  "      In mode 3, `Rows` is the TOP-K (how many experts are ACTIVE), not rows per expert.\n"
                  "      You ran L=%d Rows=%d. The recorded decode band is:\n\n"
                  "          $BIN/test_moe_splitk_bench 64 8 %d %d %d 3\n\n"
                  "      Expect active=8, roof ~7.6 us (21 MB), and cta 512 for 16x32:256 at S=1.\n"
                  "      To measure this geometry anyway: SPLITK_FORCE=1\n",
                  bd.active, cta_best, bd.L, bd.Rows, bd.N, bd.K, bd.gs);
      if (!std::getenv("SPLITK_FORCE")) return 2;
      std::printf("      SPLITK_FORCE=1 set -- proceeding. These numbers do not bound anything.\n\n");
    }
  }

  // int4 memory roof: weights + scale/zero once per active expert, plus A and D.
  double const roof = double(bd.active) * (double(bd.N) * bd.K * 4 / 8.0 + double(bd.scale_k) * bd.N * 4.0)
                    + double(bd.total) * bd.K * 2.0 + double(bd.total) * bd.N * 2.0;
  std::printf("   int4 memory roof: %.2f us (%.2f MB)\n", roof / (HBM_GBS * 1e9) * 1e6, roof / 1e6);

  std::vector<half_t> hA((size_t)bd.total * bd.K), hSc((size_t)bd.L * bd.scale_k * bd.N),
                      hZr((size_t)bd.L * bd.scale_k * bd.N);
  for (auto& v : hA)  v = half_t(0.01f);
  for (auto& v : hSc) v = half_t(0.0625f);
  for (auto& v : hZr) v = half_t(-0.0625f);

  cutlass::DeviceAllocation<half_t> dA((size_t)bd.total * bd.K), dSc((size_t)bd.L * bd.scale_k * bd.N),
                                    dZr((size_t)bd.L * bd.scale_k * bd.N), dD((size_t)bd.total * bd.N);
  dA.copy_from_host(hA.data()); dSc.copy_from_host(hSc.data()); dZr.copy_from_host(hZr.data());

  int const S_MAX = 8;
  cutlass::DeviceAllocation<half_t> dPart((size_t)S_MAX * bd.total * bd.N);

  bd.rsh.resize(bd.L);
  std::vector<half_t*> pdh(bd.L); std::vector<DStride> sdh(bd.L); std::vector<int> gmh(bd.L);
  std::vector<half_t*> pdAllh((size_t)bd.L * S_MAX);
  for (int e = 0; e < bd.L; ++e) {
    bd.rsh[e] = cute::make_shape(bd.me[e], bd.N, bd.K);
    pdh[e] = dD.get() + (size_t)bd.offs[e] * bd.N;
    sdh[e] = cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(bd.me[e], bd.N, 1));
    gmh[e] = bd.me[e];
    // Slice s writes into partial plane s, so the merge finds the S planes contiguous.
    for (int s = 0; s < S_MAX; ++s)
      pdAllh[(size_t)s * bd.L + e] = dPart.get() + ((size_t)s * bd.total + bd.offs[e]) * bd.N;
  }
  cutlass::DeviceAllocation<GS> rdev(bd.L);            rdev.copy_from_host(bd.rsh.data());
  cutlass::DeviceAllocation<half_t*> pd(bd.L);         pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(bd.L);         sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(bd.L);             gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(bd.L);         offdev.copy_from_host(bd.offs.data());
  cutlass::DeviceAllocation<half_t*> pdAll((size_t)bd.L * S_MAX); pdAll.copy_from_host(pdAllh.data());
  // The epilogue indexes ptr_D AND stride_D with the shifted coordinate e + slice*L, so the stride array has to
  // be L*S long too -- the same L strides repeated, since slicing K does not change any output stride.
  std::vector<DStride> sdAllh((size_t)bd.L * S_MAX);
  for (int s = 0; s < S_MAX; ++s)
    for (int e = 0; e < bd.L; ++e) sdAllh[(size_t)s * bd.L + e] = sdh[e];
  cutlass::DeviceAllocation<DStride> sdAll((size_t)bd.L * S_MAX); sdAll.copy_from_host(sdAllh.data());
  cutlass::DeviceAllocation<half_t> dRef((size_t)bd.total * bd.N);   // each config's S=1 output
  cutlass::DeviceAllocation<char> ws(1 << 20);

  bd.dA = dA.get(); bd.dSc = dSc.get(); bd.dZr = dZr.get();
  bd.pd = pd.get(); bd.sd = sd.get(); bd.rdev = rdev.get(); bd.gm = gm.get(); bd.offdev = offdev.get();
  bd.ws = ws.get(); bd.wsb = ws.size();

  SkCtx cx{};
  cx.dPart = &dPart; cx.pdAll = &pdAll; cx.pdOne = &pd; cx.sdAll = &sdAll; cx.sdOne = &sd; cx.dRef = &dRef;
  cx.dD = &dD; cx.S_max = S_MAX;
  std::printf("   in-kernel split-K: ONE launch, the slice on gridDim.z, so the concurrent grid is mt*ntile*S\n");
  SkBest b1, bS;
  std::printf("\n-- %d generated config units, S in {1,2,4,8} (legal when S divides the k-tile count) --\n",
              SPLITK_UNIT_COUNT);
  splitk_run_all(bd, cx, b1, bS);

  if (!sk_acu()) {
    std::printf("\n== the two winners ==\n");
    if (b1.us < 1e29) std::printf("  splitk=1 : %-34s %8.2f us\n", b1.tag, b1.us);
    else              std::printf("  splitk=1 : none ran\n");
    if (bS.us < 1e29) std::printf("  splitk>1 : %-34s %8.2f us\n", bS.tag, bS.us);
    else              std::printf("  splitk>1 : none ran\n");
    if (b1.us < 1e29 && bS.us < 1e29) {
      double const sp = b1.us / bS.us;
      std::printf("  speedup from split-K: %.3fx   (the occupancy model predicts at most 1.27x -- more than\n"
                  "                                 that means the 18 warps/CU ceiling was wrong)\n", sp);
    }
    std::printf("\n  To profile one: SPLITK_ONLY=\"<tag substring>\" SPLITK_ACU=1 acu -o splitk.report --set full -f "
                "$BIN/test_moe_splitk_bench %d %d %d %d %d %d\n", bd.L, bd.Rows, bd.N, bd.K, bd.gs, bd.mode);
  }
  std::printf("\n  launches refused: %d\n", moe_grouped_ppu::moeg_fail_count());
  return 0;
}
