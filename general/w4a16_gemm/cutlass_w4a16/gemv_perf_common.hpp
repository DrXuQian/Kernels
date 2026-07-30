// Shared harness for the GEMV perf sweep. The sweep itself lives in one generated .cu per config, NOT here.
//
// WHY SPLIT AT ALL -- the same reason lowbit_moe_bench.hpp gives: one .cu is ONE compiler invocation, so a
// 42-row sweep in a single translation unit compiles strictly serially. cutlass_build_dev_kernels emits one
// add_custom_command per .cu, so N sources become N independent commands and `make -j` runs them
// concurrently. Splitting is the only way to instantiate MORE and wait LESS.
//
// make_bufs is deliberately NOT a template: the weight packing depends on (format, layout, TileSizeK) only, so
// the buffers are built once per GROUP in the caller's translation unit and passed to every config unit of
// that group. Packing per unit would repeat an O(N*K) pass 42 times.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

// NARROW THE INSTANTIATION SET TO WHAT THE SHAPE LIST ACTUALLY CALLS. Every (gs, quant) pair is a full set of
// kernels per unit; the shapes below use gs 32 and 128 and only the two finegrained quant ops, so paying for
// gs {0,16,64} and PerColScaleOnly would triple the per-unit compile for coverage nothing exercises.
#define GEMV_GS_LIST(EMIT) EMIT(32) EMIT(128)
#define GEMV_QUANT_LIST(EMIT, G) \
  EMIT(QuantOp::FinegrainedScaleOnly, G) EMIT(QuantOp::FinegrainedScaleZero, G)
#include "gemv_lowbit/gemv_launcher.hpp"
#include "gemv_lowbit/gemv_rt.hpp"

using namespace ppu_gemv;

#define GEMV_PERF_REV 1

// ppu001. Same constants the MoE bench uses, so the percentages are comparable across the three winners.
static constexpr double HBM_GBS = 2766.0;
static constexpr int    CU      = 72;

inline const char* only_filter() { return std::getenv("GEMV_ONLY"); }
inline bool acu_mode() { return std::getenv("GEMV_ACU") != nullptr; }
// GEMV_ONLY is a substring of the whole tag, which means counting the spaces a "%-7s %-6s" pads out -- fragile
// for an acu capture, where selecting the wrong number of rows wastes the whole run. GEMV_FMT and GEMV_CFG match
// the two halves independently, so `GEMV_FMT=int4 GEMV_CFG="s16/t128 N2 C2"` picks exactly one row.
inline bool row_selected(const char* tag) {
  const char* f = only_filter();
  if (f && std::strstr(tag, f) == nullptr) return false;
  const char* fmt = std::getenv("GEMV_FMT");
  if (fmt && std::strncmp(tag, fmt, std::strlen(fmt)) != 0) return false;   // format name starts the tag
  const char* cfg = std::getenv("GEMV_CFG");
  if (cfg && std::strstr(tag, cfg) == nullptr) return false;
  return true;
}

// chrono + a device sync: the same timing shape lowbit_moe_bench.hpp uses, so it works under both runtimes.
template <typename F>
inline double time_it(F&& f, int iters) {
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
inline void upd(Best& b, const char* t, double us, double pct) {
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
inline std::vector<uint8_t> pack_plane(WLayout lay, int bits, int TS, int N, int K, uint32_t seed) {
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

inline Bufs make_bufs(WFormat fmt, WLayout Lay, int TS, Shape const& sh) {
  int const LoBits = lo_bits_of(fmt), HiBits = hi_bits_of(fmt);
  bool const TwoPlane = is_two_plane(fmt);

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
inline void run_row(Shape const& sh, Bufs const& b, Best& best) {
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

