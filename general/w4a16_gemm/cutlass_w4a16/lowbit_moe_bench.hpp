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
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "xplane_offline.hpp"
#include "fold_traits.hpp"
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
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

template <class F> inline double time_it(F&& f, int iters) {
  f(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) f();
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

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
// Read the FLOOR: it is conclusive in one direction only. Low means NOT bandwidth-bound however much re-reading
// happens; high means look at acu, because the floor cannot tell you where between the two ends the truth sits.
inline void report(const Band& bd, const char* tag, double us, int TM, int TN, int bits_total) {
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
  const double dfl   = double(bd.total) * double(bd.K) * 2.0 + double(bd.active) * (wb + sb)
                     + double(bd.total) * double(bd.N) * 2.0;
  const double dce   = double(mt) * ntile * double(TM) * double(bd.K) * 2.0 + double(mt) * wb
                     + double(mt) * ntile * double(TN) * (double(bd.K) / double(bd.gs)) * 2.0 * 2.0
                     + double(bd.total) * double(bd.N) * 2.0;
  const double gbs = dfl / (us * 1e-6) / 1e9;
  const double tf  = 2.0 * double(bd.total) * double(bd.N) * double(bd.K) / (us * 1e-6) / 1e12;
  std::printf("    %-30s %8.2f us | %6.1f TF/s (%4.1f%% MFU) | mt=%-5lld msk=%4.1f%% skw=%.1fx |"
              " floor %6.0f MB %6.0f GB/s (%4.1f%% HBM) noreuse %.1fx\n",
              tag, us, tf, 100.0 * tf * 1e12 / PEAK, mt, 100.0 * masked, skew,
              dfl / 1e6, gbs, 100.0 * gbs / HBM_GBS, dfl > 0 ? dce / dfl : 0.0);
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
      && (long long)WN * TK * bmin >= 4096                             // swzl delivery <= fragment slots, per plane
      && ((long long)TM * TK * 2 + (long long)TN * TK * bits_total / 8) * Stages <= 262144;
}

// THE MATCHED GRID: the full (TileM, WarpM) product with WarpM <= TileM, shared by every format so it cannot drift.
//
// The previous version was five hand-picked points on an L-shape -- (32,32) (64,32) (64,64) (128,64) (256,64) -- chosen
// so WarpM at fixed TileM=64 separated from TileM at fixed WarpM=64. That reasoning was right and the coverage was not:
// (128,32) and (256,32) were simply absent, and `cvt/mma = 128/WM` being worse at WM=32 is a PREDICTION, which is
// exactly the kind of claim that has been wrong twice here (int4 was declared "not the ceiling" only because it had
// never been tuned on the same grid; w64x32 was missing from a whole scan and worth +7 to +9 points).
#define MOE_GRID(EMIT)                                                                  \
  EMIT( 32,  32)  EMIT( 64,  32)  EMIT( 64,  64)  EMIT(128,  32)                        \
  EMIT(128,  64)  EMIT(256,  32)  EMIT(256,  64)

// STAGES IS AN AXIS, and it was not one. It was baked into each hand-written row, which left int4's single-plane
// winner at s3 and int2's IDENTICAL shape at s2 -- so `i4 382.76 vs i2 420.83` was not a format comparison, it was s3
// vs s2. On dense the same tile moved 1.46x between s2 and s3. 4 is included because it is the toolchain's default
// and therefore the value someone gets by accident.
#define MOE_STAGE_LIST(F, ...)  F(__VA_ARGS__, 2)  F(__VA_ARGS__, 3)  F(__VA_ARGS__, 4)

// ---- two-plane: pack ONCE per shape, then time every stage count against the same device buffer.
//
// The pack used to run per EXPERT, 64 times, on byte-identical input -- the weight pattern does not depend on e. At
// N=K=2048 that is 268 M positions per row for one row's worth of information. Pack expert 0 and memcpy the rest; the
// sweep only became affordable at this size because of it.
#define MOE2_TIME(BD,BEST,NAME,LOELEM,HIELEM,LOB,HIB,TMv,TNv,TKv,WMv,WNv,Sv)                                       \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,Sv,LOB,HIB>()) {                                                        \
    const double u = time_it([&]{                                                                                  \
      moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TMv,TNv,TKv,WMv,WNv,Sv,LOELEM,HIELEM>(              \
          (BD).dA, _b1.get(), (BD).dSc, (BD).dZr, (BD).pd, (BD).sd, (BD).gm,                                       \
          (BD).Mmax, (BD).N, (BD).K, (BD).L, (BD).gs, (BD).rdev, (BD).rsh.data(),                                  \
          (BD).mode ? (BD).offdev : nullptr, (BD).ws, (BD).wsb, nullptr, _b2.get()); }, 20);                        \
    char _t[64]; std::snprintf(_t, 64, NAME " %dx%d:%d w%dx%d s%d", TMv, TNv, TKv, WMv, WNv, Sv);                  \
    report(BD, _t, u, TMv, TNv, (LOB)+(HIB));  upd(BEST, _t, u);                                                   \
  }

#define MOE2(BD,BEST,NAME,LOELEM,HIELEM,LOB,HIB,TMv,TNv,TKv,WMv,WNv) do {                                          \
  constexpr int _F1 = fold::FoldTraits<LOB,TMv,TNv,TKv,3,WMv,WNv>::F;                                              \
  constexpr int _F2 = fold::FoldTraits<HIB,TMv,TNv,TKv,3,WMv,WNv>::F;                                              \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,2,LOB,HIB>()) {                                                         \
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
    const double u = time_it([&]{                                                                                  \
      moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TMv,TNv,TKv,WMv,WNv,Sv,ELEM>(                       \
          (BD).dA, _db.get(), (BD).dSc, (BD).dZr, (BD).pd, (BD).sd, (BD).gm,                                       \
          (BD).Mmax, (BD).N, (BD).K, (BD).L, (BD).gs, (BD).rdev, (BD).rsh.data(),                                  \
          (BD).mode ? (BD).offdev : nullptr, (BD).ws, (BD).wsb, nullptr); }, 20);                                   \
    char _t[64]; std::snprintf(_t, 64, NAME " %dx%d:%d w%dx%d s%d", TMv, TNv, TKv, WMv, WNv, Sv);                  \
    report(BD, _t, u, TMv, TNv, (BITS));  upd(BEST, _t, u);                                                        \
  }

#define MOE1(BD,BEST,NAME,ELEM,BITS,TMv,TNv,TKv,WMv,WNv) do {                                                      \
  constexpr int _F = fold::FoldTraits<BITS,TMv,TNv,TKv,3,WMv,WNv>::F;                                              \
  if constexpr (moe_ok<TMv,TNv,TKv,WMv,WNv,2,BITS>()) {                                                            \
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

// One entry point per translation unit. Each prints its own section header and folds into the shared Best.
void sweep_q3     (const Band&, Best&);
void sweep_q5     (const Band&, Best&);
void sweep_q6_wn32(const Band&, Best&);
void sweep_q6_wn64(const Band&, Best&);
void sweep_i2_wn32(const Band&, Best&);
void sweep_i2_wn64(const Band&, Best&);
void sweep_i4_wn32(const Band&, Best&);
void sweep_i4_wn64(const Band&, Best&);
