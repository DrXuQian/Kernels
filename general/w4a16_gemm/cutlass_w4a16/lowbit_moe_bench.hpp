// Shared harness for the MoE shape-band sweep. The sweep itself lives in one moe_bench_*.cu per format, NOT here.
//
// WHY SPLIT AT ALL. Every row of the sweep is a full kernel instantiation, and one .cu is ONE hgcc invocation, so a
// 200-row sweep in a single translation unit compiles strictly serially. cutlass_build_dev_kernels emits one
// add_custom_command per .cu (cmake/PPUToolchain.cmake), so N sources become N independent commands and `make -j`
// runs them concurrently. Splitting is the only way to instantiate MORE and wait LESS -- the alternative,
// instantiating less, is not on the table.
//
// The split is by (format, WarpN) rather than by format alone so the largest unit stays near the size of the old
// single-file sweep: q3/q5 have one legal WarpN so they are one file each; q6/i2/i4 have two, so they are two.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "xplane_offline.hpp"
#include "fold_traits.hpp"   // warps_per_cu_chunked: the occupancy model WITH the register term
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
// THE QUANT MODE, and it used to be hardcoded ScaleZero at BOTH launch sites -- so every low-bit number this harness
// has ever produced, decode band included, was an affine run. That was invisible because the 2-plane driver ignored
// QuantOp anyway (fixed: moe_grouped_ppu's NoZero slot), so asking for ScaleOnly changed nothing and nobody noticed.
// Default is unchanged; LOWBIT_QMODE=1 selects ScaleOnly. The mode goes in the header line, because a row that cannot
// be told apart from the other mode is how "(ScaleOnly)" ended up labelling affine runs for months.
#if defined(LOWBIT_QMODE) && (LOWBIT_QMODE == 1)
#  define LOWBIT_QMODE_SEL QM::FinegrainedScaleOnly
#  define LOWBIT_QMODE_STR "ScaleOnly"
#else
#  define LOWBIT_QMODE_SEL QM::FinegrainedScaleZero
#  define LOWBIT_QMODE_STR "ScaleZero"
#endif

using int4_t  = cutlass::int4b_t;
using uint2_t = cutlass::uint2b_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

static constexpr double PEAK    = 500.0e12;
// HBM peak for ppu001 (see the ppu-ptx skill). Used to turn the traffic model into a percentage, which is the only form
// in which "is this bandwidth-bound?" is answerable.
static constexpr double HBM_GBS = 2766.0;

// WHICH SOURCE DID THIS BINARY COMPILE AGAINST. Bump this whenever a macro in this header changes shape. The banner prints
// it, so a run can no longer be ambiguous about whether it used the fixed header -- a failed build showed a macro expansion
// containing `fold::FoldTraits` while the checked-out tree had `moe_fold`, and there was no way to tell from here whether
// the box had built an older commit or the overlay had a stale copy. That ambiguity has cost rounds twice in this work (the
// per-unit PPU_B_CHUNK vote exists for the same reason), so it gets an invariant instead of a guess.
#define LOWBIT_MOE_BENCH_REV 11

// TileK is a BUILD knob, not a row: it changes the per-plane fold factor, so sweeping it at runtime would mean packing
// every row twice. One extra build (PPU_DEFS=MOE_TK=128) covers it.
#ifndef MOE_TK
#define MOE_TK 64
#endif

// SELF-DESCRIBING RUN, NOW PER TRANSLATION UNIT. Whether PPU_B_CHUNK was active has been undeterminable from the build
// output twice already (the device compiles are add_custom_command with a COMMENT, so make.log holds no compile line).
// Splitting the sweep adds a NEW way for that to go wrong: if one moe_bench_*.cu misses the -D, it silently runs the
// unchunked collective while main's banner -- compiled in a different TU -- still says chunked. So every sweep prints
// its OWN state and a disagreement is visible in the log rather than buried in a perf number.
// `#x` stringizes a macro PARAMETER only, so `#PPU_B_CHUNK` outside a function-like macro is ill-formed -- two levels.
#define PPU_STR2_(x) #x
#define PPU_STR1_(x) PPU_STR2_(x)
#if defined(PPU_B_CHUNK)
#define PPU_CHUNK_STR "PPU_B_CHUNK=" PPU_STR1_(PPU_B_CHUNK)
#else
#define PPU_CHUNK_STR "PPU_B_CHUNK=off"
#endif

// EVERY UNIT VOTES ON ITS OWN PPU_B_CHUNK. The 8 hand-written units each printed their state; 128 generated ones cannot
// each print a line, but the check must survive, because a unit that misses the -D silently runs the UNCHUNKED collective
// while main's banner -- a different translation unit -- still says chunked. Each unit registers its state at static-init
// time and main reports the tally, so a split brain is one line in the log instead of a perf number nobody can explain.
// DECODE A BROADCAST, as an A/B in ONE binary. A's m-stride goes to 0 so the TileM-1 padding rows read the
// expert's real row instead of 15 other rows -- legal only because the epilogue's residue mask discards them,
// and only when every expert has ONE row, which launch() enforces. Off by default: above Mmax == 1 it would be
// refused, and a sweep that silently refused half its rows is worse than one that never tried.
inline bool moe_abcast() { static int c = -1; if (c < 0) c = std::getenv("MOE_ABCAST") ? 1 : 0; return c != 0; }

struct MoeChunkTally { int on = 0, off = 0; };
inline MoeChunkTally& moe_chunk_tally() { static MoeChunkTally t; return t; }
inline void moe_chunk_vote(bool on) { if (on) ++moe_chunk_tally().on; else ++moe_chunk_tally().off; }
#if defined(PPU_B_CHUNK)
#define PPU_CHUNK_ON 1
#else
#define PPU_CHUNK_ON 0
#endif

// ROW SELECTION, for acu and for cheap re-runs. acu needs EXACTLY ONE kernel launch in the process, and the sweep issues
// 336 x 21 of them; there was no way to profile a single row without editing and rebuilding, which is how a cross-config
// mismatch gets introduced (the same trap FOLD_ONCE exists for in test_fold_int2).
//
//   MOE_ONLY=<substring>   run only rows whose tag contains it, e.g. MOE_ONLY="i2 64x128:64 w64x32 s3" or MOE_ONLY=i4
//   MOE_ACU=1              with MOE_ONLY, issue ONE launch instead of 1 warmup + 20 timed
//
// The shape-level test matches in BOTH directions on purpose: a full row tag ("i2 64x128:64 w64x32 s3") contains the shape
// string, and a loose filter ("i4") is contained BY it. Getting this backwards would silently pack nothing and print an
// empty sweep, which reads exactly like a broken build.
inline const char* moe_only() { static const char* v = std::getenv("MOE_ONLY"); return v; }
inline bool moe_acu() { static const bool v = std::getenv("MOE_ACU") != nullptr; return v; }
inline bool moe_row_selected(const char* tag) {
  const char* f = moe_only();
  return !f || std::strstr(tag, f) != nullptr;
}
inline bool moe_shape_selected(const char* shape) {
  const char* f = moe_only();
  return !f || std::strstr(shape, f) != nullptr || std::strstr(f, shape) != nullptr;
}

// Everything a sweep needs about the band. Passed by const reference so the per-format translation units share ONE
// set of device buffers instead of each allocating its own.
struct Band {
  int L, N, K, gs, Rows, mode;
  int total, Mmax, scale_k, active;
  std::vector<int>   me, offs;
  std::vector<GS>    rsh;                 // host-side; filter_and_run takes a host pointer for this one
  half_t*   dA;   half_t*  dSc;  half_t* dZr;
  half_t**  pd;   DStride* sd;   GS*     rdev;  int* gm;  int* offdev;
  char*     ws;   size_t   wsb;
};

struct Best { char tag[64]; double us; };
inline void upd(Best& b, const char* t, double u) { if (u < b.us) { b.us = u; std::snprintf(b.tag, 64, "%s", t); } }

// iters == 0 means EXACTLY ONE launch and no warmup: acu attributes counters to the whole process, so a warmup would
// double-count and 20 iterations would make the report meaningless.
template <class F> inline double time_it(F&& f, int iters) {
  if (iters == 0) {
    auto a = std::chrono::high_resolution_clock::now();
    f(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    auto b = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(b - a).count();
  }
  f(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) f();
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// F WITHOUT INSTANTIATING FoldTraits, and that distinction is the whole point. FoldTraits carries static_asserts --
// `delivery <= slots` among them -- so computing F from it fires those asserts for configurations moe_ok would have
// rejected. That is what broke the TileK=32 build: moe_ok correctly rejects int2 at WarpN=32 (slots = 32*32/32 = 32 against
// delivery = 128/2 = 64), but `constexpr int _F = FoldTraits<...>::F` sat ABOVE the `if constexpr (moe_ok<...>)` gate, and
// a template argument is instantiated whether or not a later gate would have discarded the statement. Same closed form
// here, no asserts; FoldTraits still fires for the configurations that actually launch, from inside the collective.
template <int Bits> constexpr int moe_fold(int TK) { const int c = TK * Bits / 8; return c >= 32 ? 1 : 32 / c; }

// THE TRAFFIC MODEL LIVES HERE, not in prose. I argued "not bandwidth-bound" from traffic computed by hand in a
// message; that number changes with every shape and tile, so the tool has to produce it or the argument is
// unfalsifiable.
//
// AND IT MUST BE A LOWER BOUND. The first version printed only the DMA-issued upper bound and reported 116-181% of
// HBM, which answers nothing: a bound looser than the hardware peak is not a measurement. The over-count is A --
// 60% of the total on the worst row -- because `mt * (N/TN) * TM * K * 2` assumes every n-tile column re-reads all of
// A from DRAM, when the CTAs sharing an m-tile read the IDENTICAL A tile (128 KB at TM=32) and will hit L2.
//
//   floor  every byte crosses the bus AT LEAST once:  A = total_rows*K*2, B+S once per ACTIVE expert, D = rows*N*2
//   ceil   the DMA-issued bound above
//   noreuse = ceil/floor -- how much reuse the L2 must be supplying. A ratio, deliberately not a percentage.
//
// WHICH END IS CONCLUSIVE, and I had this BACKWARDS in the first version of this comment and in three messages built on it.
// Achieved traffic lies in [floor, ceil], so achieved bandwidth lies in [floor/t, ceil/t]:
//   * ceil/t  < peak  =>  DEFINITELY NOT saturated. CONCLUSIVE, and it is the ceiling that gives it.
//   * floor/t ~ peak  =>  DEFINITELY saturated. CONCLUSIVE the other way.
//   * a LOW floor/t says NOTHING -- the truth may sit at the ceiling. That is the error: "low floor means not
//     bandwidth-bound" is false, because the lower bound being small does not bound the actual from above.
// Both ends are printed now. And the strongest argument needs neither: if two configs of the same format differ by 1.4-2.8x
// in ceiling traffic and land within 0.4-6% in TIME, traffic is not the limiter -- that is what the decode run showed.
// OCCUPANCY IS PRINTED, not inferred. The decode measurement showed 256 CTAs and 128 CTAs of the same format landing within
// 0.4% of each other -- both fit in one wave, so the time is ONE CTA's K-loop latency and adding CTAs cannot touch it. What
// can is the number of tile loads in flight, which per CTA is exactly `Stages`, and per CU is `blocks/CU * Stages`. That
// quantity was never on screen, so the hypothesis could only be argued.
//
// `blk` COMES FROM fold::warps_per_cu_chunked, NOT from a formula written here. The first version of this used
// `min(262144/smem, 64/warps)` -- which fold_traits.hpp and the README both record as KNOWN-WRONG: acu said outright
// "theoretical occupancy (25.0%) is limited by the number of required registers", shared memory allowing 10 blocks where
// registers allowed 4, and that missing register term is why an earlier `blk` "kept failing to order the data". The correct
// model bills registers ROUNDED UP TO A POWER OF TWO (129 costs exactly what 256 costs) and is pinned by static_assert to
// acu's measured points. Re-deriving it here would have been the same mistake a second time.
// `wav` uses CU = 72, which acu confirmed independently through its own wave arithmetic (1024 - 3*288 = 160).
//
// AND `ifl` IS INVARIANT IN Stages WHENEVER smem BINDS, which it does at every decode shape here (blk_smem 19 < blk_warp 32
// for i4 (32,64,32) w32x32). Since blk = 256KB / (smem_per_stage * Stages), the product blk*Stages collapses to
// 256KB / smem_per_stage: i4 (32,64,32) gives ifl 76 at s4 and 72 at s12 -- deepening the pipeline does not add in-flight
// loads, it CONCENTRATES the same total into fewer CTAs and costs warps. So "more stages" and "more occupancy" pull opposite
// ways here, and the lever that actually raises the total is SHRINKING smem per stage: TileN=32 halves both the B and the
// scale term (3328 -> 2688 B, ifl <= 97). Dropping the zero tile (#20 Phase 1) is only 128 B of 3328, i.e. 3.8%.
// Which of depth-vs-width wins at a fixed total is exactly what the sweep is for.
inline void report(const Band& bd, const char* tag, double us, int TM, int TN, int TK, int WM, int WN, int Stages,
                   int bits_total, int wcu) {
  long long mt = 0, mt_max = 0;
  for (int e = 0; e < bd.L; ++e) {
    const long long t = (bd.me[e] + TM - 1) / TM; mt += t; mt_max = std::max(mt_max, t);
  }
  const double masked = mt ? 1.0 - double(bd.total) / (double(TM) * double(mt)) : 0.0;
  // skew = the heaviest expert's m-tile count against the mean. With a non-persistent scheduler this is what the
  // last-wave tail is made of, so it belongs next to the time rather than in a separate analysis.
  const double skew  = mt ? double(mt_max) / (double(mt) / double(bd.L)) : 0.0;
  const double ntile = std::ceil(double(bd.N) / double(TN));
  const double wb    = double(bd.N) * double(bd.K) * double(bits_total) / 8.0;         // one expert's weights
  const double sb    = double(bd.N) * (double(bd.K) / double(bd.gs)) * 2.0 * 2.0;      // scale + zero, fp16
  // A IS PADDED IN THE FLOOR. The kernel fetches TM rows whether or not they are real, so `mt*TM*K*2` is compulsory and
  // `total*K*2` was an under-count -- at decode that is 1.05 MB against 32 KB of real rows.
  //
  // AND AT DECODE THE TRAFFIC IS LOCKED, NOT A BRACKET, which is the correction that produced this version: `mt == active`
  // there, so B is exactly `active*wb` and S exactly `active*sb` -- each active expert's weights read once. The 2.6x
  // "noreuse" the previous version printed came ENTIRELY from A's cross-n-tile term (33.5 MB of a 54.5 MB ceiling), and
  // that term assumes 32 CTAs each re-fetch their m-tile's A from DRAM when the whole distinct A footprint is 8 x 128 KB =
  // 1 MB shared inside one wave. Calling that a bandwidth question is not defensible, so it is no longer folded into a
  // bracket: the floor is reported as THE number and A's reuse factor is printed beside it as `Ax`.
  const double a_pad  = double(mt) * double(TM) * double(bd.K) * 2.0;
  const double dfl    = a_pad + double(bd.active) * (wb + sb) + double(bd.total) * double(bd.N) * 2.0;
  const double dce    = ntile * a_pad + double(mt) * (wb + sb) + double(bd.total) * double(bd.N) * 2.0;
  const double gbs = dfl / (us * 1e-6) / 1e9;
  const double tf  = 2.0 * double(bd.total) * double(bd.N) * double(bd.K) / (us * 1e-6) / 1e12;
  // S's SHARE OF THE FLOOR is what #20 can possibly buy, and without it the floor cannot answer that question. At decode
  // (1 row per active expert) the weight term is read exactly once per active expert, so floor and ceiling coincide on B
  // and S and this share is a tight number rather than a bound.
  const double s_share = dfl > 0 ? double(bd.active) * sb / dfl : 0.0;
  const double gbs_c = dce / (us * 1e-6) / 1e9;
  // THE TWO QUANTITIES TileK CONTROLS, both candidates for why a memory-bound kernel misses its own roofline.
  //   run  = F*TK*bits/8, the AIU CONTIGUOUS run in bytes. The fold targets the AIU's 32 B minimum, which is right on
  //          prefill (it shrinks A-smem) and may be exactly wrong at decode: 24.8% of peak is suspiciously close to
  //          32/128, i.e. one 32 B run per 128 B line.
  //   kit  = K/TK, the k-iteration count. 64 dependent iterations at TK=32 against 8 at TK=256.
  // Both improve monotonically with TileK, so they are confounded across a TileK scan -- report them so the confound is
  // visible rather than discovered later.
  const int  fold_l = (TK * bits_total / 8) >= 32 ? 1 : 32 / (TK * bits_total / 8);
  const int  run_b  = fold_l * TK * bits_total / 8;
  const long kit    = bd.K / TK;
  const int  warps  = (TM / WM) * (TN / WN);
  const int  blk    = warps > 0 ? wcu / warps : 0;          // wcu comes from fold::warps_per_cu_chunked at the call site
  const long ctas   = mt * (long)ntile;
  const double waves = (blk > 0) ? double(ctas) / (72.0 * double(blk)) : 0.0;   // CU = 72, confirmed by acu wave arithmetic
  // ACHIEVED occupancy is set by the GRID, not by the per-CU limits, and the identity says which knobs can move it:
  //     grid warps = ctas * warps/blk = [mt * N/TN] * [(TM/WM)(TN/WN)] = mt * N * TM / (WM * WN)
  // TileN and TileK CANCEL. At the decode winner that is 8*2048*16/(16*32) = 512 warps over 72 CUs = 7.1 warps/CU, and acu
  // measured achieved 6.97 -- the identity closes. So the only levers are mt (fixed by the router), TM/WM, and WN; smem caps
  // THEORETICAL occupancy (acu: 21.88%, Block Limit Shared Mem 7) while ACHIEVED (10.90%) is grid-limited, which is why
  // shrinking the scale or zero tile raises a ceiling that is not being reached.
  const double gwarps = double(mt) * double(bd.N) * double(TM) / (double(WM) * double(WN));
  const double wcu_grid = gwarps / 72.0;
  std::printf("    %-30s %8.2f us | %6.1f TF/s (%4.1f%% MFU) | mt=%-5lld msk=%4.1f%% skw=%.1fx |"
              " HBM %4.1f%% S=%4.1f%% | blk %-2d wrp/CU %-3d grid_wrp/CU %5.1f cta=%-5ld wav=%4.2f run=%-3dB kit=%-3ld%s\n",
              tag, us, tf, 100.0 * tf * 1e12 / PEAK, mt, 100.0 * masked, skew,
              100.0 * gbs / HBM_GBS, 100.0 * s_share,
              blk, blk * warps, wcu_grid, ctas, waves, run_b, kit,
              gbs_c < 0.9 * HBM_GBS ? "  NOT-BW" : "");
}

// DID THIS ROW ACTUALLY RUN? Two independent nets, because the first only catches causes we already know about.
//   1. moeg_fail_count() grew  -> can_implement / workspace / initialize refused and launch() returned without a kernel.
//   2. the compulsory weight traffic could not have crossed the bus in the measured time -> whatever the cause, nothing ran.
// Net 2 is what would have caught this class immediately: the bench reported `q5 128x128:256 w32x64 s3` at 3.17 us as its
// FASTEST config, which is 6.6 TB/s against a 2.77 TB/s peak. A win that violates the hardware is not a win, and a harness
// that cannot say so will rank its own failures.
inline bool moe_row_ran(const Band& bd, const char* tag, double us, int fail0, int bits_total) {
  const double wb  = double(bd.N) * double(bd.K) * double(bits_total) / 8.0;
  const double gbs = (double(bd.active) * wb) / (us * 1e-6) / 1e9;   // weights alone: the least it can possibly have moved
  const bool refused    = moe_grouped_ppu::moeg_fail_count() > fail0;
  const bool impossible = gbs > HBM_GBS;
  if (refused || impossible) {
    std::printf("    %-30s %8.2f us | DID NOT RUN (%s) -- excluded from the verdict\n", tag, us,
                refused ? "launch refused" : "implies > HBM peak");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// LEGALITY, as one constexpr predicate rather than as a hand-maintained row list.
//
// The delivery bound is per PLANE and the binding plane is the SPARSEST one: one swzl delivery hands a thread a fixed
// 16 B = 128/bits codes, the fp16 B fragment has slots = WN*TK/32, and over-delivery silently drops the surplus. So
// every plane needs WN*TK*bits >= 4096, i.e. the minimum over the planes decides. That is why Q3/Q5 (an int1 plane)
// are pinned to WN=64 at TK=64 while Q6 (int4+int2) may run WN=32 -- a fact this file's own section headers assert and
// the box has measured, so the predicate is cross-checked against something other than my own derivation.
//
// smem: A is TM*TK*2 per stage; each plane's B is Ng_p * (F_p*TK*bits_p/8) and Ng_p = TN/F_p, so the fold CANCELS and
// B is simply TN*TK*bits_total/8 whatever the folding does. 256 KB per CU is a hard cap (not configurable higher).
template <int TM, int TN, int TK, int WM, int WN, int Stages, int BitsLo, int BitsHi = 0>
constexpr bool moe_ok() {
  constexpr int bits_total = BitsLo + BitsHi;
  constexpr int bmin = BitsHi ? (BitsLo < BitsHi ? BitsLo : BitsHi) : BitsLo;
  return WM <= TM && WN <= TN && TM % WM == 0 && TN % WN == 0
      && (TM / WM) * (TN / WN) <= 32                                   // <= 1024 threads per block
      // ACCUMULATOR REGISTERS. The fp32 C fragment is WM*WN elements over 32 lanes, so WM*WN/32 registers per thread
      // against a hard ceiling of 256 (see the occupancy notes for CC 8.0). w64x128 hits exactly 256 -- the accumulator
      // alone would consume the entire budget -- so it must be REJECTED here rather than left to fail deep in the
      // collective. This matters the moment TileK shrinks: at TK=32 an int1 plane needs WN >= 4096/32 = 128, which is
      // precisely that dead config, so the bound is what makes "int1 cannot reach TK=32" a filter instead of a build break.
      && (WM * WN) / 32 <= 192
      && (long long)WN * TK * bmin >= 4096                             // swzl delivery <= fragment slots, per plane
      // SMEM MUST INCLUDE THE SCALE AND ZERO TILES. Leaving them out let q5 (128,128,256) s3 through at a computed 252 KB
      // when the real figure is 264 KB, and the launch then failed at runtime with "[moe_grouped] init failed" -- which the
      // bench timed as 3.17 us and reported as its FASTEST config. fold_traits.hpp already had the complete expression
      // (TM*TK*2 + TN*TK*Bits/8 + TN*(TK/Gs)*2*(Zero?2:1)); this one was written again, and short.
      // gs is a RUNTIME argument, so size for the smallest gs supported (16) and stay conservative: a filter that is too
      // tight loses a config, one that is too loose produces a fake winner.
      && ((long long)TM * TK * 2 + (long long)TN * TK * bits_total / 8
          + (long long)TN * (TK / 16) * 2 * 2) * Stages <= 262144;
}

// (The hand-written (TileM, WarpM) grid that used to sit here is gone. One unit fixes one shape, so TileM and WarpM are
// generated axes now, and moe_ok's WM <= TM rejects the illegal corner -- the L-shape is a CONSEQUENCE of the predicate
// rather than a list someone maintains. The five points it used to enumerate were missing (128,32) and (256,32).)

// STAGES IS AN AXIS, and it was not one. It was baked into each hand-written row, which left int4's single-plane
// winner at s3 and int2's IDENTICAL shape at s2 -- so `i4 382.76 vs i2 420.83` was not a format comparison, it was s3
// vs s2. On dense the same tile moved 1.46x between s2 and s3. 4 is included because it is the toolchain's default
// and therefore the value someone gets by accident.
// DEEPER THAN 4, because `Stages` IS the per-CTA count of tile loads in flight and the decode band is latency-bound with a
// tiny tile: i4 (32,64,32) is ~3.25 KB/stage, so s16 costs 52 KB of the 256 KB budget. The collective caps nothing above
// `Stages >= 2`; smem does, and moe_ok's smem predicate filters the rest, so these cost nothing on the prefill shapes where
// A-smem is 8-32 KB/stage and they simply do not fit.
// MOE_STAGES narrows this axis from the build, for the same reason the tile lists can be narrowed: the cost is entirely
// front-end template instantiation at ~3.0 s per kernel, so halving the stage list halves the per-unit time.
//   PPU_DEFS="PPU_B_CHUNK=1 MOE_STAGES_4 MOE_STAGES_8"   -- define the ones you want; none defined = all six
#if defined(MOE_STAGES_2) || defined(MOE_STAGES_3) || defined(MOE_STAGES_4) || \
    defined(MOE_STAGES_6) || defined(MOE_STAGES_8) || defined(MOE_STAGES_12)
   // at least one was requested: leave the others undefined so only the requested stages are emitted
#else
#  define MOE_STAGES_2
#  define MOE_STAGES_3
#  define MOE_STAGES_4
#  define MOE_STAGES_6
#  define MOE_STAGES_8
#  define MOE_STAGES_12
#endif
#define MOE_STAGE_LIST(F, ...)                                                                 \
  MOE_STG_2(F, __VA_ARGS__)  MOE_STG_3(F, __VA_ARGS__)  MOE_STG_4(F, __VA_ARGS__)              \
  MOE_STG_6(F, __VA_ARGS__)  MOE_STG_8(F, __VA_ARGS__)  MOE_STG_12(F, __VA_ARGS__)
#ifdef MOE_STAGES_2
#  define MOE_STG_2(F, ...)  F(__VA_ARGS__, 2)
#else
#  define MOE_STG_2(F, ...)
#endif
#ifdef MOE_STAGES_3
#  define MOE_STG_3(F, ...)  F(__VA_ARGS__, 3)
#else
#  define MOE_STG_3(F, ...)
#endif
#ifdef MOE_STAGES_4
#  define MOE_STG_4(F, ...)  F(__VA_ARGS__, 4)
#else
#  define MOE_STG_4(F, ...)
#endif
#ifdef MOE_STAGES_6
#  define MOE_STG_6(F, ...)  F(__VA_ARGS__, 6)
#else
#  define MOE_STG_6(F, ...)
#endif
#ifdef MOE_STAGES_8
#  define MOE_STG_8(F, ...)  F(__VA_ARGS__, 8)
#else
#  define MOE_STG_8(F, ...)
#endif
#ifdef MOE_STAGES_12
#  define MOE_STG_12(F, ...)  F(__VA_ARGS__, 12)
#else
#  define MOE_STG_12(F, ...)
#endif

// ---- two-plane: pack ONCE per shape, then time every stage count against the same device buffer.
//
// The pack used to run per EXPERT, 64 times, on byte-identical input -- the weight pattern does not depend on e. At
// N=K=2048 that is 268 M positions per row for one row's worth of information. Pack expert 0 and memcpy the rest; the
// sweep only became affordable at this size because of it.
#define MOE2_TIME(BD,BEST,NAME,LOELEM,HIELEM,LOB,HIB,TMv,TNv,TKv,WMv,WNv,Sv)                                       \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,Sv,LOB,HIB>()) {                                                        \
    char _t[64]; std::snprintf(_t, 64, NAME " %dx%d:%d w%dx%d s%d%s", TMv, TNv, TKv, WMv, WNv, Sv,                 \
                              moe_abcast() ? " B" : "");                                                            \
    if (moe_row_selected(_t)) {                                                                                    \
      auto _go = [&]{                                                                                              \
        moe_grouped_ppu::filter_and_run<LOWBIT_QMODE_SEL,TMv,TNv,TKv,WMv,WNv,Sv,LOELEM,HIELEM>(            \
            (BD).dA, _b1.get(), (BD).dSc, (BD).dZr, (BD).pd, (BD).sd, (BD).gm,                                     \
            (BD).Mmax, (BD).N, (BD).K, (BD).L, (BD).gs, (BD).rdev, (BD).rsh.data(),                                \
            (BD).mode ? (BD).offdev : nullptr, (BD).ws, (BD).wsb, nullptr, _b2.get(),                                \
            /*k_full=*/-1, /*prefix_ready=*/false, /*splitk=*/1, moe_abcast()); };                                  \
      double u; const int _f0 = moe_grouped_ppu::moeg_fail_count();                                                \
      if (moe_acu()) { u = time_it(_go, 0); std::printf("  [acu] ONE COLD launch (not a timing): %s\n", _t); }                         \
      else             u = time_it(_go, 20);                                                                       \
      if (moe_row_ran(BD, _t, u, _f0, (LOB)+(HIB))) { report(BD,_t,u,TMv,TNv,TKv,WMv,WNv,Sv,(LOB)+(HIB),fold::warps_per_cu_chunked<TMv,TNv,TKv,WMv,WNv,Sv,(LOB)+(HIB),32,true>);  upd(BEST, _t, u); } \
    }                                                                                                              \
  }

#define MOE2(BD,BEST,NAME,LOELEM,HIELEM,LOB,HIB,TMv,TNv,TKv,WMv,WNv) do {                                          \
  char _sh[64]; std::snprintf(_sh, 64, NAME " %dx%d:%d w%dx%d", TMv, TNv, TKv, WMv, WNv);                           \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,2,LOB,HIB>()) if (moe_shape_selected(_sh)) {                            \
    constexpr int _F1 = moe_fold<LOB>(TKv);                                                                        \
    constexpr int _F2 = moe_fold<HIB>(TKv);                                                                        \
    const size_t _lo = (size_t)(BD).K*(BD).N*(LOB)/8, _hi = (size_t)(BD).K*(BD).N*(HIB)/8;                          \
    std::vector<int8_t> _blo((size_t)(BD).L*_lo), _bhi((size_t)(BD).L*_hi);                                        \
    { std::vector<uint8_t> _l((size_t)(BD).K*(BD).N), _h((size_t)(BD).K*(BD).N);                                    \
      for (size_t i = 0; i < _l.size(); ++i) {                                                                     \
        const int q = int((i * 2654435761u >> 5) % (unsigned)((1u<<(LOB))<<(HIB)));                                 \
        _l[i] = uint8_t(q & ((1u<<(LOB))-1u)); _h[i] = uint8_t(q >> (LOB)); }                                       \
      xplane::place_derived<LOB,TMv,TNv,TKv,WMv,WNv,_F1>(_blo.data(), _l, (BD).N, (BD).K);                          \
      xplane::place_hi<LOB,HIB,TMv,TNv,TKv,WMv,WNv,_F2,_F1>(_bhi.data(), _h, (BD).N, (BD).K);                       \
      for (int e = 1; e < (BD).L; ++e) {                                                                           \
        std::memcpy(_blo.data() + (size_t)e*_lo, _blo.data(), _lo);                                                 \
        std::memcpy(_bhi.data() + (size_t)e*_hi, _bhi.data(), _hi); } }                                            \
    cutlass::DeviceAllocation<LOELEM> _b1((size_t)(BD).L*_lo);                                                     \
    _b1.copy_from_host(reinterpret_cast<LOELEM const*>(_blo.data()));                                              \
    cutlass::DeviceAllocation<HIELEM> _b2((size_t)(BD).L*_hi);                                                     \
    _b2.copy_from_host(reinterpret_cast<HIELEM const*>(_bhi.data()));                                              \
    MOE_STAGE_LIST(MOE2_TIME, BD,BEST,NAME,LOELEM,HIELEM,LOB,HIB,TMv,TNv,TKv,WMv,WNv)                              \
  } } while (0)

// ---- single plane, same structure.
#define MOE1_TIME(BD,BEST,NAME,ELEM,BITS,TMv,TNv,TKv,WMv,WNv,Sv)                                                   \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,Sv,BITS>()) {                                                           \
    char _t[64]; std::snprintf(_t, 64, NAME " %dx%d:%d w%dx%d s%d%s", TMv, TNv, TKv, WMv, WNv, Sv,                 \
                              moe_abcast() ? " B" : "");                                                            \
    if (moe_row_selected(_t)) {                                                                                    \
      auto _go = [&]{                                                                                              \
        /* PlaneB2 is NAMED (void) on purpose: passing nullptr for B2 while letting PlaneB2 deduce makes the    \
           deduction fail on std::nullptr_t, and a failed deduction is NOT rescued by the default template       \
           argument -- the call simply stops matching. */                                                        \
        moe_grouped_ppu::filter_and_run<LOWBIT_QMODE_SEL,TMv,TNv,TKv,WMv,WNv,Sv,ELEM,void>(                \
            (BD).dA, _db.get(), (BD).dSc, (BD).dZr, (BD).pd, (BD).sd, (BD).gm,                                     \
            (BD).Mmax, (BD).N, (BD).K, (BD).L, (BD).gs, (BD).rdev, (BD).rsh.data(),                                \
            (BD).mode ? (BD).offdev : nullptr, (BD).ws, (BD).wsb, nullptr,                                          \
            /*B2=*/nullptr, /*k_full=*/-1, /*prefix_ready=*/false, /*splitk=*/1, moe_abcast()); };                   \
      double u; const int _f0 = moe_grouped_ppu::moeg_fail_count();                                                \
      if (moe_acu()) { u = time_it(_go, 0); std::printf("  [acu] ONE COLD launch (not a timing): %s\n", _t); }                         \
      else             u = time_it(_go, 20);                                                                       \
      if (moe_row_ran(BD, _t, u, _f0, (BITS))) { report(BD,_t,u,TMv,TNv,TKv,WMv,WNv,Sv,(BITS),fold::warps_per_cu_chunked<TMv,TNv,TKv,WMv,WNv,Sv,(BITS),32,true>);  upd(BEST, _t, u); } \
    }                                                                                                              \
  }

#define MOE1(BD,BEST,NAME,ELEM,BITS,TMv,TNv,TKv,WMv,WNv) do {                                                      \
  char _sh[64]; std::snprintf(_sh, 64, NAME " %dx%d:%d w%dx%d", TMv, TNv, TKv, WMv, WNv);                           \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,2,BITS>()) if (moe_shape_selected(_sh)) {                               \
    constexpr int _F = moe_fold<BITS>(TKv);                                                                        \
    const size_t _per = (size_t)(BD).K*(BD).N*(BITS)/8;                                                            \
    std::vector<int8_t> _bb((size_t)(BD).L*_per);                                                                  \
    { std::vector<uint8_t> _q((size_t)(BD).K*(BD).N);                                                              \
      for (size_t i = 0; i < _q.size(); ++i) _q[i] = uint8_t((i * 2654435761u >> 5) & ((1u<<(BITS))-1u));           \
      xplane::place_derived<BITS,TMv,TNv,TKv,WMv,WNv,_F>(_bb.data(), _q, (BD).N, (BD).K);                           \
      for (int e = 1; e < (BD).L; ++e) std::memcpy(_bb.data() + (size_t)e*_per, _bb.data(), _per); }                \
    cutlass::DeviceAllocation<ELEM> _db((size_t)(BD).L*_per);                                                      \
    _db.copy_from_host(reinterpret_cast<ELEM const*>(_bb.data()));                                                 \
    MOE_STAGE_LIST(MOE1_TIME, BD,BEST,NAME,ELEM,BITS,TMv,TNv,TKv,WMv,WNv)                                          \
  } } while (0)

// The unit entry points are DECLARED IN GENERATED CODE (moe_bench_units.inc, written by CMakeLists), one per shape.
// Nothing is declared here on purpose: 128 hand-written declarations would be a second copy of the enumeration, and the
// whole reason the sweep is generated is that the enumeration must live in exactly one place.
