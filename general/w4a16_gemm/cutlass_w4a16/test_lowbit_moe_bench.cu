// MoE SHAPE BAND perf sweep for every low-bit format [box-only]. Separate from test_lowbit_grouped on purpose: mixing a
// timing lambda into a correctness harness is how a sweep once came to time a different tile than the one it checked.
//
// WHY THE DENSE WINNER CANNOT BE INHERITED. On dense M=2048 all six formats peak at (64,128,64) w64x64 s2, and TN=128
// wins there because it halves A's re-read count -- the tile was selected by A TRAFFIC. In MoE the bottleneck is the
// WEIGHTS, and the recorded rule is to minimise the TOTAL m-tile count, which favours LARGE TileM. Those point opposite
// ways. But large TileM also wastes rows: at ~128 rows per expert, TM=256 leaves half of every expert's single m-tile
// MASKED. So the question is not "which tile is fastest" but WHICH OF THE TWO EFFECTS DOMINATES, and the honest
// instrument prints both quantities next to the time instead of asserting one of them:
//
//     m_tiles     = sum_e ceil(me[e]/TM)      what the weight-bound rule says to minimise
//     masked_frac = 1 - sum_e me[e] / (TM * m_tiles)      what large TileM costs in wasted rows
//
// MFU is computed on the REAL rows (2 * total_rows * N * K), not on the padded m-tile area, so a configuration cannot
// buy MFU by wasting work.
//
//   Build: TARGET=test_lowbit_moe_bench ./build.sh    (PPU_DEFS=PPU_B_CHUNK=1 -- and CHECK the verified line)
//   Run:   ./<bin> [L] [rows_per_expert] [N] [K] [gs] [ragged 0|1]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "xplane_offline.hpp"
#include "moe_grouped_ppu.cuh"

// SELF-DESCRIBING RUN. Whether PPU_B_CHUNK was active has now been undeterminable from the build output twice: the
// device compiles are add_custom_command with a COMMENT so make.log holds no compile line, and the build.sh line that
// does check scrolls away above a long result. The binary reports its own configuration instead -- a log that does not
// describe its own run has cost this work three separate rounds.
// `#x` stringizes a macro PARAMETER only, so `#PPU_B_CHUNK` outside a function-like macro is ill-formed -- two levels.
#define PPU_STR2_(x) #x
#define PPU_STR1_(x) PPU_STR2_(x)
#if defined(PPU_B_CHUNK)
#define PPU_CHUNK_STR "PPU_B_CHUNK=" PPU_STR1_(PPU_B_CHUNK)
#else
#define PPU_CHUNK_STR "PPU_B_CHUNK=off"
#endif


using half_t  = cutlass::half_t;
using int4_t  = cutlass::int4b_t;
using uint2_t = cutlass::uint2b_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

static constexpr double PEAK = 500.0e12;
// HBM peak for ppu001 (see the ppu-ptx skill). Used to turn the traffic model into a percentage, which is the only form
// in which "is this bandwidth-bound?" is answerable.
static constexpr double HBM_GBS = 2766.0;
static int L = 64, Rows = 128, N = 2048, K = 2048, gs = 32;
// SKEWED IS THE DEFAULT. Uniform row counts are the case that teaches least: every expert has the same m-tile count, so
// there is no masking, no load imbalance, and no tail -- i.e. none of what makes MoE different from a batched GEMM. Real
// row counts come from a router. mode 0 (uniform) is kept only as a control, and mode 1 exists to be pointed at: its
// counts are all multiples of TileM, so it never enters the masked path at all.
static int mode_ = 2;    // 0 uniform (control), 1 coarse ragged (multiples of TM -- masked path NEVER entered), 2 skewed

// sub-byte pointers advance in BYTES (sizeof(uint2b_t) == 1), so per-expert offsets must be byte counts
template <class T> static T const* eptr(T const* b, size_t e, size_t bytes) {
  return reinterpret_cast<T const*>(reinterpret_cast<int8_t const*>(b) + e * bytes);
}

static std::vector<int> me_, offs_;
static int total_ = 0, Mmax_ = 0;

template <class F> static double time_it(F&& f, int iters) {
  f(); CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) f();
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

struct Best { char tag[48]; double us; };
static void upd(Best& b, const char* t, double u) { if (u < b.us) { b.us = u; std::snprintf(b.tag, 48, "%s", t); } }


// THE TRAFFIC MODEL LIVES HERE, not in prose. I argued "not bandwidth-bound" from traffic computed by hand in a message;
// that number changes with every shape and tile, so the tool has to produce it or the argument is unfalsifiable.
//
// DMA-ISSUED BYTES, AN UPPER BOUND ON DRAM TRAFFIC. Each CTA covers one (TM, TN) output tile and streams the full K:
//     A   m_tiles * (N/TN) * TM * K * 2            A is re-read once per n-tile; masked rows are still fetched
//     B   m_tiles * N * K * bits_total / 8         the whole weight slab per m-tile, BOTH planes for a 2-plane format
//     S   m_tiles * (N/TN) * TN * (K/gs) * 2 * 2   scale and zero
//     D   total_rows * N * 2
// It is an UPPER bound because consecutive m-tiles of one expert may hit L2 on B, so read the percentage as "at most this
// much of HBM": a LOW number is conclusive (not bandwidth-bound), a high one needs acu to confirm.
static void report(const char* tag, double us, int TM, int TN, int bits_total) {
  long long mt = 0, mt_max = 0;
  for (int e = 0; e < L; ++e) { const long long t = (me_[e] + TM - 1) / TM; mt += t; mt_max = std::max(mt_max, t); }
  const double masked = mt ? 1.0 - double(total_) / (double(TM) * double(mt)) : 0.0;
  // skew = the heaviest expert's m-tile count against the mean. With a non-persistent scheduler this is what the
  // last-wave tail is made of, so it belongs next to the time rather than in a separate analysis.
  const double skew  = mt ? double(mt_max) / (double(mt) / double(L)) : 0.0;
  const double ntile = std::ceil(double(N) / double(TN));
  const double bytes = double(mt) * ntile * double(TM) * double(K) * 2.0
                     + double(mt) * double(N) * double(K) * double(bits_total) / 8.0
                     + double(mt) * ntile * double(TN) * (double(K) / double(gs)) * 2.0 * 2.0
                     + double(total_) * double(N) * 2.0;
  const double gbs = bytes / (us * 1e-6) / 1e9;
  const double fl  = 2.0 * double(total_) * double(N) * double(K);
  const double tf  = fl / (us * 1e-6) / 1e12;
  std::printf("    %-42s %8.2f us | %6.1f TF/s (%4.1f%% MFU) | mt=%-5lld msk=%4.1f%% skw=%.1fx | %7.0f MB %6.0f GB/s (%4.1f%% HBM)\n",
              tag, us, tf, 100.0 * tf * 1e12 / PEAK, mt, 100.0 * masked, skew,
              bytes / 1e6, gbs, 100.0 * gbs / HBM_GBS);
}

int main(int argc, char** argv) {
  if (argc > 1) L      = atoi(argv[1]);
  if (argc > 2) Rows   = atoi(argv[2]);
  if (argc > 3) N      = atoi(argv[3]);
  if (argc > 4) K      = atoi(argv[4]);
  if (argc > 5) gs     = atoi(argv[5]);
  if (argc > 6) mode_  = atoi(argv[6]);
  const int scale_k = K / gs;
  // ROW-COUNT DISTRIBUTION. mode 0 = uniform, 1 = the old coarse ragged, 2 = SKEWED, which is the realistic one.
  //
  // The first ragged generator was (Rows/2)*((e%4)+1) -- 64, 128, 192, 256. Every count a multiple of 64, hence a
  // multiple of every TileM being swept, hence **the masked path was never entered at all**. That is the same mistake as
  // a probe whose period aliases the displacement it is hunting, in different clothing: a "ragged" model whose raggedness
  // is invisible to the thing under test measures the uniform case twice.
  //
  // Real MoE row counts come from a router, so mode 2 gives: arbitrary counts (not multiples of anything), a heavy tail
  // (one expert several times the mean), and ZERO-ROW experts, which are common at small batch and are their own path --
  // filter_and_run has to skip them rather than launch an empty tile.
  me_.resize(L); offs_.resize(L);
  int zeros = 0, mn = 1 << 30;
  for (int e = 0; e < L; ++e) {
    if (mode_ == 0)      me_[e] = Rows;
    else if (mode_ == 1) me_[e] = (Rows / 2) * ((e % 4) + 1);
    else {
      // skewed: ~12% of experts get nothing, one in eight gets a heavy share, the rest scatter around the mean
      const unsigned h = (unsigned)(e * 2654435761u) >> 13;
      if ((h % 8) == 0)      me_[e] = 0;
      else if ((h % 8) == 1) me_[e] = int(Rows * 3 + (h % 37));       // heavy tail
      else                   me_[e] = int(Rows / 2 + (h % (unsigned)(Rows + 1)));   // arbitrary, not a multiple of TM
    }
    if (me_[e] == 0) ++zeros;
    mn = std::min(mn, me_[e]);
    offs_[e] = total_; total_ += me_[e]; Mmax_ = std::max(Mmax_, me_[e]);
  }
  if (Mmax_ == 0) { std::printf("all experts empty; nothing to measure\n"); return 1; }
  const char* mname = mode_ == 0 ? "uniform" : (mode_ == 1 ? "ragged-coarse(multiples of TM: masked path NEVER entered)"
                                              : "skewed(arbitrary counts + zero-row experts)");
  std::printf("[lowbit-moe] %s  L=%d rows/expert=%d mode=%s N=%d K=%d gs=%d\n", PPU_CHUNK_STR, L, Rows, mname, N, K, gs);
  std::printf("             total=%d Mmax=%d Mmin=%d zero-row experts=%d  PEAK=%.0f TFLOP/s\n",
              total_, Mmax_, mn, zeros, PEAK / 1e12);
  std::printf("             MFU is on the REAL rows (2*total*N*K), so wasting rows cannot buy MFU.\n");

  std::vector<half_t> hA((size_t)total_ * K), hSc((size_t)L * scale_k * N), hZr((size_t)L * scale_k * N);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = half_t(0.01f);
  for (size_t i = 0; i < hSc.size(); ++i) { hSc[i] = half_t(0.0625f); hZr[i] = half_t(0.5f); }
  cutlass::DeviceAllocation<half_t> dA((size_t)total_ * K), dSc((size_t)L * scale_k * N),
                                    dZr((size_t)L * scale_k * N), dD((size_t)total_ * N);
  dA.copy_from_host(hA.data()); dSc.copy_from_host(hSc.data()); dZr.copy_from_host(hZr.data());

  std::vector<GS> rsh(L); std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);
  for (int e = 0; e < L; ++e) {
    rsh[e] = cute::make_shape(me_[e], N, K); pdh[e] = dD.get() + (size_t)offs_[e] * N;
    sdh[e] = cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(me_[e], N, 1)); gmh[e] = me_[e];
  }
  cutlass::DeviceAllocation<GS> rdev(L);      rdev.copy_from_host(rsh.data());
  cutlass::DeviceAllocation<half_t*> pd(L);   pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L);   sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(L);       gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(L);   offdev.copy_from_host(offs_.data());
  const size_t wsb = (size_t)cutlass::ceil_div(Mmax_,16)*cutlass::ceil_div(N,64)*(size_t)L*64;
  cutlass::DeviceAllocation<char> ws(wsb);

  Best b2p{"",1e18}, b1p{"",1e18};

  // ---- two-plane
#define MOE2(NAME,BEST,LOWELEM,HIELEM,LOWB,HIB,TMv,TNv,TKv,WMv,WNv,Sv,F1v,F2v) do {                                     \
    const size_t lo_per = (size_t)K * N * (LOWB) / 8, hi_per = (size_t)K * N * (HIB) / 8;                                \
    std::vector<int8_t> blo((size_t)L * lo_per), bhi((size_t)L * hi_per);                                                \
    { std::vector<uint8_t> lo((size_t)K*N), hi((size_t)K*N);                                                             \
      for (size_t i = 0; i < lo.size(); ++i) { const int q = int((i * 2654435761u >> 5) % (unsigned)((1<<(LOWB))<<(HIB)));\
        lo[i] = uint8_t(q & ((1<<(LOWB))-1)); hi[i] = uint8_t(q >> (LOWB)); }                                             \
      for (int e = 0; e < L; ++e) {                                                                                      \
        xplane::place_derived<LOWB,TMv,TNv,TKv,WMv,WNv,F1v>(blo.data() + (size_t)e*lo_per, lo, N, K);                     \
        xplane::place_hi<LOWB,HIB,TMv,TNv,TKv,WMv,WNv,F2v,F1v>(bhi.data() + (size_t)e*hi_per, hi, N, K); } }              \
    cutlass::DeviceAllocation<LOWELEM> b1((size_t)L*lo_per); b1.copy_from_host(reinterpret_cast<LOWELEM const*>(blo.data()));\
    cutlass::DeviceAllocation<HIELEM>  b2((size_t)L*hi_per); b2.copy_from_host(reinterpret_cast<HIELEM  const*>(bhi.data()));\
    double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TMv,TNv,TKv,WMv,WNv,Sv,LOWELEM,HIELEM>(\
        dA.get(), b1.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                                          \
        Mmax_, N, K, L, gs, rdev.get(), rsh.data(), mode_ ? offdev.get() : nullptr, ws.get(), wsb, nullptr, b2.get()); }, 20);\
    report(NAME " " #TMv "x" #TNv ":" #TKv " w" #WMv "x" #WNv " s" #Sv, u, TMv, TNv, (LOWB)+(HIB));                                          \
    upd(BEST, NAME " " #TMv "x" #TNv ":" #TKv " s" #Sv, u); } while (0)

  // ---- single plane
#define MOE1(NAME,BEST,ELEM,BITS,TMv,TNv,TKv,WMv,WNv,Sv,Fv) do {                                                         \
    const size_t per = (size_t)K * N * (BITS) / 8;                                                                        \
    std::vector<int8_t> bb((size_t)L * per);                                                                              \
    { std::vector<uint8_t> q((size_t)K*N);                                                                                \
      for (size_t i = 0; i < q.size(); ++i) q[i] = uint8_t((i * 2654435761u >> 5) & ((1u<<(BITS))-1u));                    \
      for (int e = 0; e < L; ++e) xplane::place_derived<BITS,TMv,TNv,TKv,WMv,WNv,Fv>(bb.data() + (size_t)e*per, q, N, K); }\
    cutlass::DeviceAllocation<ELEM> db((size_t)L*per); db.copy_from_host(reinterpret_cast<ELEM const*>(bb.data()));       \
    double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TMv,TNv,TKv,WMv,WNv,Sv,ELEM>(       \
        dA.get(), db.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                                          \
        Mmax_, N, K, L, gs, rdev.get(), rsh.data(), mode_ ? offdev.get() : nullptr, ws.get(), wsb, nullptr); }, 20);     \
    report(NAME " " #TMv "x" #TNv ":" #TKv " w" #WMv "x" #WNv " s" #Sv, u, TMv, TNv, (BITS));                                          \
    upd(BEST, NAME " " #TMv "x" #TNv ":" #TKv " s" #Sv, u); } while (0)

  // THE SAME (TileM, WarpM) GRID FOR EVERY FORMAT, and the two axes SEPARATED. The first version swept TileM
  // 32/64/128/256 and put TM=32 at w32x64 -- because TM=32 forces warpM <= 32 -- so the TM=32 rung moved TWO variables,
  // and only Q3 got that rung at all. Two mistakes already written down this session and then made one section later: a
  // rung that changes more than one thing, and comparing formats on grids that are not the same.
  //
  // WN is NOT free across formats and cannot be matched: int1 as the high plane forces WN >= 4096/TK = 64 at TK=64,
  // int2 needs 32, int4 needs 16. So each format is swept at ITS legal WN and only (TileM, WarpM) is matched:
  //     (32,32)  (64,32)  (64,64)  (128,64)  (256,64)
  // giving TileM at fixed WarpM=64 (rows 3-5) AND WarpM at fixed TileM=64 (rows 2-3), so cvt/mma = 128/WM is separable
  // from the m-tile count instead of confounded with it.
  std::printf("  --- Q3 = int2 + int1   (WN=64 forced by the int1 plane) ---\n");
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1,  32,128, 64,32,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1,  64,128, 64,32,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1,  64,128, 64,64,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1, 128,128, 64,64,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1, 256,128, 64,64,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1, 128,128, 64,64,64,3,2,4);
  std::printf("  --- Q6 = int4 + int2   (WN >= 32, so w*x32 is legal here unlike Q3/Q5) ---\n");
  MOE2("q6", b2p, int4_t, uint2_t, 4,2,  32,128, 64,32,64,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2,  64,128, 64,32,64,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2,  64,128, 64,64,64,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2, 128,128, 64,64,64,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2, 256,128, 64,64,64,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2,  64,128, 64,64,32,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2, 128,128, 64,64,32,3,1,2);
  std::printf("  --- Q5 = int4 + int1   (WN=64 forced; the SAME grid as Q3 -- that IS the Q3-vs-Q5 comparison) ---\n");
  MOE2("q5", b2p, int4_t, uint1_t, 4,1,  32,128, 64,32,64,2,1,4);
  MOE2("q5", b2p, int4_t, uint1_t, 4,1,  64,128, 64,32,64,2,1,4);
  MOE2("q5", b2p, int4_t, uint1_t, 4,1,  64,128, 64,64,64,2,1,4);
  MOE2("q5", b2p, int4_t, uint1_t, 4,1, 128,128, 64,64,64,2,1,4);
  MOE2("q5", b2p, int4_t, uint1_t, 4,1, 256,128, 64,64,64,2,1,4);
  std::printf("  --- single plane int2 and int4, same (TileM, WarpM) grid at their own WN ---\n");
  MOE1("i2", b1p, uint2_t, 2,  32, 64, 64,32,32,2,2);
  MOE1("i2", b1p, uint2_t, 2,  64, 64, 64,32,32,2,2);
  MOE1("i2", b1p, uint2_t, 2,  64, 64, 64,64,32,2,2);
  MOE1("i2", b1p, uint2_t, 2, 128, 64, 64,64,32,3,2);
  MOE1("i2", b1p, uint2_t, 2, 256, 64, 64,64,32,3,2);
  MOE1("i2", b1p, uint2_t, 2, 128,128, 64,64,32,3,2);
  MOE1("i4", b1p, int4_t,  4,  32, 64, 64,32,32,2,1);
  MOE1("i4", b1p, int4_t,  4,  64, 64, 64,32,32,2,1);
  MOE1("i4", b1p, int4_t,  4,  64, 64, 64,64,32,3,1);
  MOE1("i4", b1p, int4_t,  4, 128, 64, 64,64,32,3,1);
  MOE1("i4", b1p, int4_t,  4, 256, 64, 64,64,32,3,1);
  MOE1("i4", b1p, int4_t,  4, 128,128, 64,64,32,3,1);

  std::printf("\n  ================= VERDICT =================\n");
  std::printf("  two-plane best : %-40s %8.2f us\n", b2p.tag, b2p.us);
  std::printf("  single-plane   : %-40s %8.2f us\n", b1p.tag, b1p.us);
  std::printf("  %%HBM is the COMPULSORY floor -- every byte crossing the bus at least once -- so it is conclusive: a low\n");
  std::printf("  figure means the kernel is NOT bandwidth-bound whatever the re-reads do. noreuse is how much larger the\n");
  std::printf("  per-CTA re-read count is, i.e. how much reuse the L2 must be supplying; it is a ratio, not a percentage,\n");
  std::printf("  because the first version of this printed 116-181%% of HBM and a bound looser than the peak says nothing.\n");
  std::printf("  Then read mt and msk: smallest mt winning => the weight-bound rule holds; least msk => the dense\n");
  std::printf("  reasoning carries over; neither => the lever is occupancy and the next instrument is acu.\n");
  return 0;
}
