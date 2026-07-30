// Decode-band performance for the low-bit GEMV: the CUDA-CORE-ONLY winner.
//
// This is one of the three winners to dump and then profile. The other two (grouped GEMM at splitk=1 and at
// splitk>1) come from test_moe_splitk_bench.cu, so all three are directly comparable: same shapes, same
// traffic model, same reporting.
//
// TWO SHAPE FAMILIES, because they have DIFFERENT parallelism knobs and conflating them hid a factor of 8
// earlier in this work:
//
//   * MoE decode (L experts x 1 row). grid = 1 x n/CtaN x L. The expert dimension multiplies the grid, so at
//     L=8, n=2048, CtaN=8 this is 2048 CTAs x 128 threads = 8192 warps = ~7 waves on 72 CUs.
//   * DENSE decode (one matrix, m rows). grid = ceil(m/CtaM) x n/CtaN x 1. There is NO expert dimension, so
//     at m=1, n=2048, CtaN=8 it is 256 CTAs = 1024 warps = 14.2 warps/CU -- the SAME warp count as the
//     grouped GEMM's decode launch. For dense, parallelism has to be bought with CtaN, and the price is that
//     A is re-read n/CtaN times (4 KB at m=1, so it should live in L2 -- that is a prediction to check, not
//     an assumption).
//
// B IS RE-READ ONCE PER m-TILE, so the traffic model counts it grid.x times. That is what makes this kernel a
// decode kernel and not a GEMM: at m=8 with CtaM=4 the weights come in twice.
//
// Build: TARGET=test_gemv_perf ./build.sh      Run: $BIN/test_gemv_perf
//   GEMV_ONLY=<substring>  run only rows whose tag contains this (one row -> one launch, for acu)
//   GEMV_ACU=1             ONE COLD launch per row, no warmup, no repeat -- a capture, NOT a timing
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

// Narrow the instantiation set for the bench: these are the group sizes the decode path actually ships.
#define GEMV_GS_LIST(EMIT) EMIT(32) EMIT(128)
#include "gemv_lowbit/gemv_launcher.hpp"
#include "gemv_lowbit/gemv_rt.hpp"

using namespace ppu_gemv;

// ppu001. Same constants the MoE bench uses, so the percentages are comparable across the three winners.
static constexpr double HBM_GBS = 2766.0;
static constexpr int    CU      = 72;

static const char* only_filter() { return std::getenv("GEMV_ONLY"); }
static bool acu_mode() { return std::getenv("GEMV_ACU") != nullptr; }
static bool row_selected(const char* tag) {
  const char* f = only_filter();
  return !f || std::strstr(tag, f) != nullptr;
}

// chrono + a device sync: the same timing shape lowbit_moe_bench.hpp uses, so it works under both runtimes.
template <typename F>
static double time_it(F&& f, int iters) {
  if (iters == 0) { f(); rt_sync("cold launch"); return 0.0; }
  for (int i = 0; i < 5; ++i) f();
  rt_sync("warmup");
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) f();
  rt_sync("timed");
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

struct Best { char tag[96] = ""; double us = 1e30; double pct = 0; };
static void upd(Best& b, const char* t, double us, double pct) {
  if (us > 0 && us < b.us) { std::snprintf(b.tag, sizeof(b.tag), "%s", t); b.us = us; b.pct = pct; }
}

// ---------------------------------------------------------------------------------------------------
// One benchmark problem. `experts == 0` is dense.
struct Shape {
  const char* name;
  int experts;      // 0 = dense
  int rows;         // dense: m. MoE: rows per expert.
  int N, K;
  int gs;
  QuantOp quant;
};

struct Bufs {
  DevBuf A, W, Wh, S, Z, O, Off;
  std::vector<int> offs;
  int total_rows = 0;
};

// Pack a plane the way gemv_wformat.hpp defines the layout. Deliberately the same bit-position expression as
// the correctness gate's packer -- there is one convention and it lives in one form.
static std::vector<uint8_t> pack_plane(WLayout lay, int bits, int TS, int N, int K, uint32_t seed) {
  std::vector<uint8_t> out(size_t(N) * K * bits / 8, 0);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      size_t bitpos = (lay == WLayout::Native)
          ? (size_t(n) * K + size_t(k)) * bits
          : ((size_t(k / TS) * N * TS) + size_t(n) * TS + size_t(k % TS)) * bits;
      uint32_t const v = ((uint32_t(n) * 2654435761u + uint32_t(k) * 40503u + seed) >> 7) & ((1u << bits) - 1u);
      out[bitpos >> 3] |= uint8_t(v << (bitpos & 7));
    }
  return out;
}

template <typename Details>
static Bufs make_bufs(Shape const& sh) {
  constexpr int LoBits = Details::kLoBits, HiBits = Details::kHiBits, TS = Details::kTileSizeK;
  constexpr bool TwoPlane = Details::kTwoPlane;
  constexpr WLayout Lay = Details::kLayout;

  int const experts = sh.experts > 0 ? sh.experts : 1;
  Bufs b;
  b.offs.assign(experts + 1, 0);
  for (int e = 0; e < experts; ++e) b.offs[e + 1] = b.offs[e] + sh.rows;
  b.total_rows = b.offs[experts];

  int const sk = (sh.gs == 0) ? 1 : sh.K / sh.gs;

  std::vector<uint16_t> hA(size_t(b.total_rows) * sh.K, 0x3000);   // ~0.125 in fp16
  std::vector<uint16_t> hS(size_t(experts) * sk * sh.N, 0x2C00);   // ~0.0625
  std::vector<uint16_t> hZ(size_t(experts) * sk * sh.N, 0xA800);   // ~-0.0625
  auto plo = pack_plane(Lay, LoBits, TS, sh.N, sh.K, 1u);
  std::vector<uint8_t> phi;
  if (TwoPlane) phi = pack_plane(Lay, HiBits, TS, sh.N, sh.K, 7u);

  // One expert's weights replicated: the pattern does not depend on e and packing L times is what made the
  // MoE sweep unaffordable at N=K=2048 before.
  std::vector<uint8_t> wl(size_t(experts) * plo.size());
  for (int e = 0; e < experts; ++e) std::memcpy(wl.data() + size_t(e) * plo.size(), plo.data(), plo.size());
  b.A = DevBuf(hA.size() * 2);  b.A.from_host(hA.data());
  b.S = DevBuf(hS.size() * 2);  b.S.from_host(hS.data());
  if (has_zero(sh.quant)) { b.Z = DevBuf(hZ.size() * 2); b.Z.from_host(hZ.data()); }
  b.W = DevBuf(wl.size());      b.W.from_host(wl.data());
  if (TwoPlane) {
    std::vector<uint8_t> wh(size_t(experts) * phi.size());
    for (int e = 0; e < experts; ++e) std::memcpy(wh.data() + size_t(e) * phi.size(), phi.data(), phi.size());
    b.Wh = DevBuf(wh.size());   b.Wh.from_host(wh.data());
  }
  b.O = DevBuf(size_t(b.total_rows) * sh.N * 2);
  rt_memset0(b.O.p, b.O.bytes);
  if (sh.experts > 0) { b.Off = DevBuf(b.offs.size() * 4); b.Off.from_host(b.offs.data()); }
  return b;
}

// ---------------------------------------------------------------------------------------------------
template <typename Details, int CtaN, int Chunk>
static void run_row(Shape const& sh, Bufs const& b, Best& best) {
  constexpr int LoBits = Details::kLoBits, HiBits = Details::kHiBits;
  constexpr int TotalBits = LoBits + HiBits;
  constexpr int StepK = Details::kStepK, Threads = Details::kThreads;

  char tag[96];
  std::snprintf(tag, sizeof(tag), "%-7s %-6s s%-2d/t%-3d N%d C%d", Details::format_name(),
                name_of(Details::kLayout), StepK, Threads, CtaN, Chunk);
  if (!row_selected(tag)) return;

  int const experts = sh.experts > 0 ? sh.experts : 1;
  int const sk = (sh.gs == 0) ? 1 : sh.K / sh.gs;

  Params p;
  p.act = b.A.p; p.weight = b.W.p; p.weight_hi = b.Wh.p; p.scales = b.S.p; p.zeros = b.Z.p; p.out = b.O.p;
  p.m = b.total_rows; p.n = sh.N; p.k = sh.K; p.groupsize = sh.gs;
  p.format = Details::kFormat; p.quant = sh.quant; p.layout = Details::kLayout;
  if (sh.experts > 0) {
    p.num_experts = sh.experts; p.row_offsets = b.Off.as<int>(); p.max_rows = sh.rows;
    p.w_bytes_per_expert = int64_t(sh.N) * sh.K * LoBits / 8;
    p.w_hi_bytes_per_expert = HiBits ? int64_t(sh.N) * sh.K * HiBits / 8 : 0;
    p.scale_elems_per_expert = int64_t(sk) * sh.N;
  }

  int const f0 = gemv_fail_count();
  auto go = [&] { launch_gemv<Details, CtaN, Chunk>(p, 0); };
  double const us = acu_mode() ? (time_it(go, 0), 0.0) : time_it(go, 100);
  if (acu_mode()) { std::printf("  [acu] ONE COLD launch (not a timing): %s\n", tag); return; }
  if (gemv_fail_count() != f0) {
    std::printf("  %-34s %10s | DID NOT RUN (launch refused) -- excluded\n", tag, "-");
    return;
  }

  // Compulsory traffic. B counted grid.x times because every m-tile re-reads it; A counted once (it is tiny
  // at decode and should be served by L2 across the n-tiles -- if the measured rate exceeds this model, that
  // assumption is what broke).
  int const ctam = std::min(sh.rows, GEMV_CTAM_MAX);
  int const grid_m = (sh.rows + ctam - 1) / ctam;
  double const wb  = double(sh.N) * sh.K * TotalBits / 8.0;
  double const sb  = double(sk) * sh.N * 2.0 * (has_zero(sh.quant) ? 2 : 1);
  double const ab  = double(b.total_rows) * sh.K * 2.0;
  double const db  = double(b.total_rows) * sh.N * 2.0;
  double const bytes = double(experts) * (wb + sb) * grid_m + ab + db;
  double const gbs = bytes / (us * 1e-6) / 1e9;
  double const pct = 100.0 * gbs / HBM_GBS;

  int64_t const ctas = int64_t(grid_m) * (sh.N / CtaN) * experts;
  // WARPS OF WORK PER CU, not achieved occupancy -- the same quantity the MoE bench prints as grid_wrp/CU.
  // Naming it "wrp/CU" invited exactly the misreading that cost rounds earlier: 14.2 there was the TOTAL work,
  // and the reason occupancy could not exceed it. `wave` divides by the 64-warp/CU hardware maximum, so it is
  // a LOWER bound on the wave count (real occupancy is below 64, so real waves are more).
  double const wkwrp_cu = double(ctas) * (Threads / 32.0) / CU;

  std::printf("  %-34s %8.2f us | %7.1f GB/s | %5.1f%% HBM | cta %6lld | wkwrp/CU %6.1f | wave>=%5.1f%s\n",
              tag, us, gbs, pct, (long long)ctas, wkwrp_cu, wkwrp_cu / 64.0,
              gbs > HBM_GBS ? "  <-- IMPLIES > HBM PEAK, excluded" : "");
  if (gbs <= HBM_GBS) upd(best, tag, us, pct);
}

// ---------------------------------------------------------------------------------------------------
// The config table. One place, so an axis cannot be half-swept.
//
// (StepK, Threads) must satisfy StepK*min(plane bits) >= 32 (a 4-byte floor per plane per thread) and
// StepK*Threads <= K dividing K. At K=2048 that admits (8,256), (16,128) and (32,64) for int4/int8, drops
// (8,256) for int2, and leaves only (32,64) for anything with an int1 plane.
#define GEMV_CFGS_WIDE(EMIT)  /* sparsest plane >= 4 bits */                              \
  EMIT( 8, 256, 2, 2) EMIT( 8, 256, 4, 2) EMIT( 8, 256, 8, 2)                             \
  EMIT(16, 128, 2, 2) EMIT(16, 128, 4, 2) EMIT(16, 128, 8, 2) EMIT(16, 128, 8, 4)         \
  EMIT(32,  64, 2, 2) EMIT(32,  64, 4, 2) EMIT(32,  64, 8, 2)
#define GEMV_CFGS_MID(EMIT)   /* sparsest plane == 2 bits */                              \
  EMIT(16, 128, 2, 2) EMIT(16, 128, 4, 2) EMIT(16, 128, 8, 2) EMIT(16, 128, 8, 4)         \
  EMIT(32,  64, 2, 2) EMIT(32,  64, 4, 2) EMIT(32,  64, 8, 2)
#define GEMV_CFGS_NARROW(EMIT) /* sparsest plane == 1 bit */                              \
  EMIT(32,  64, 2, 2) EMIT(32,  64, 4, 2) EMIT(32,  64, 8, 2) EMIT(32,  64, 8, 4)

// THE WEIGHT BUFFERS ARE PACKED ONCE PER (format, layout), NOT PER CONFIG. Packing is O(N*K) per plane, and
// it depends only on the format and the layout -- not on StepK, Threads, CtaN or Chunk. Doing it per row
// would be ~50x the work at N=K=2048, which is the same mistake the MoE sweep's MOE2 macro documents having
// made per EXPERT.
template <WFormat F, WLayout L, int Tier>
static void sweep_fmt(Shape const& sh, Best& best) {
  using D0 = KernelDetails<FP16DetailsA, F, L, 32, 64>;   // buffers do not depend on StepK/Threads
  Bufs const b = make_bufs<D0>(sh);
#define EMIT_ROW(SK, TH, CN, CH) \
  run_row<KernelDetails<FP16DetailsA, F, L, SK, TH>, CN, CH>(sh, b, best);
  if constexpr (Tier == 0)      { GEMV_CFGS_WIDE(EMIT_ROW) }
  else if constexpr (Tier == 1) { GEMV_CFGS_MID(EMIT_ROW) }
  else                          { GEMV_CFGS_NARROW(EMIT_ROW) }
#undef EMIT_ROW
}

static void sweep(Shape const& sh, Best& best) {
  sweep_fmt<WFormat::Int4,  WLayout::Native, 0>(sh, best);
  sweep_fmt<WFormat::Int4,  WLayout::TileK,  0>(sh, best);
  sweep_fmt<WFormat::Int2,  WLayout::Native, 1>(sh, best);
  sweep_fmt<WFormat::Q6_42, WLayout::Native, 1>(sh, best);
  sweep_fmt<WFormat::Q3_21, WLayout::Native, 2>(sh, best);
  sweep_fmt<WFormat::Int1,  WLayout::Native, 2>(sh, best);
}

int main(int argc, char** argv) {
  int const only_shape = (argc > 1) ? std::atoi(argv[1]) : -1;

  std::printf("== low-bit GEMV, decode band (CUDA-CORE-ONLY winner) ==\n");
  std::printf("   HBM peak %.0f GB/s, %d CU, CtaM max %d, gs list {32,128}\n", HBM_GBS, CU, GEMV_CTAM_MAX);
  if (only_filter()) std::printf("   GEMV_ONLY=\"%s\"\n", only_filter());
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

  std::vector<Best> bests(ns);
  for (int i = 0; i < ns; ++i) {
    if (only_shape >= 0 && i != only_shape) continue;
    Shape const& sh = shapes[i];
    int const experts = sh.experts > 0 ? sh.experts : 1;
    // roofline: the weights (once) plus scales, A and D -- the least any implementation can move
    int const sk = sh.K / sh.gs;
    double const floor_b = double(experts) * (double(sh.N) * sh.K * 4 / 8.0 + double(sk) * sh.N * 4.0)
                         + double(experts) * sh.rows * sh.K * 2.0 + double(experts) * sh.rows * sh.N * 2.0;
    std::printf("\n-- [%d] %s  gs=%d %s --\n     int4 memory roof: %.2f us (%.2f MB at %.0f GB/s)\n",
                i, sh.name, sh.gs, name_of(sh.quant), floor_b / (HBM_GBS * 1e9) * 1e6, floor_b / 1e6, HBM_GBS);
    sweep(sh, bests[i]);
    if (!acu_mode() && bests[i].us < 1e29)
      std::printf("     WINNER %s  %.2f us  (%.1f%% HBM)\n", bests[i].tag, bests[i].us, bests[i].pct);
  }

  if (!acu_mode()) {
    std::printf("\n== per-shape winners (CUDA-core GEMV) ==\n");
    for (int i = 0; i < ns; ++i)
      if (bests[i].us < 1e29)
        std::printf("  [%d] %-30s %-34s %8.2f us  %5.1f%% HBM\n", i, shapes[i].name, bests[i].tag,
                    bests[i].us, bests[i].pct);
    std::printf("\n  To profile one: GEMV_ONLY=\"<tag substring>\" GEMV_ACU=1 acu -o gemv.report --set full -f "
                "$BIN/test_gemv_perf <shape index>\n");
  }
  return 0;
}
