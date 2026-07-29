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
static int L = 64, Rows = 128, N = 2048, K = 2048, gs = 32;
static bool ragged = false;

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

static void report(const char* tag, double us, int TM) {
  long long mt = 0; for (int e = 0; e < L; ++e) mt += (me_[e] + TM - 1) / TM;
  const double masked = 1.0 - double(total_) / (double(TM) * double(mt));
  const double fl = 2.0 * double(total_) * double(N) * double(K);
  const double tf = fl / (us * 1e-6) / 1e12;
  std::printf("    %-44s %8.2f us | %6.1f TFLOP/s (%4.1f%% MFU) | m_tiles=%-6lld masked=%4.1f%%\n",
              tag, us, tf, 100.0 * tf * 1e12 / PEAK, mt, 100.0 * masked);
}

int main(int argc, char** argv) {
  if (argc > 1) L      = atoi(argv[1]);
  if (argc > 2) Rows   = atoi(argv[2]);
  if (argc > 3) N      = atoi(argv[3]);
  if (argc > 4) K      = atoi(argv[4]);
  if (argc > 5) gs     = atoi(argv[5]);
  if (argc > 6) ragged = atoi(argv[6]) != 0;
  const int scale_k = K / gs;
  me_.resize(L); offs_.resize(L);
  for (int e = 0; e < L; ++e) {
    me_[e] = ragged ? (Rows / 2) * ((e % 4) + 1) : Rows;   // ragged spans 0.5x..2x the nominal rows
    offs_[e] = total_; total_ += me_[e]; Mmax_ = std::max(Mmax_, me_[e]);
  }
  std::printf("[lowbit-moe] %s  L=%d rows/expert=%d %s N=%d K=%d gs=%d total=%d Mmax=%d  PEAK=%.0f TFLOP/s\n",
              PPU_CHUNK_STR, L, Rows, ragged ? "ragged" : "uniform", N, K, gs, total_, Mmax_, PEAK / 1e12);
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
        Mmax_, N, K, L, gs, rdev.get(), rsh.data(), ragged ? offdev.get() : nullptr, ws.get(), wsb, nullptr, b2.get()); }, 20);\
    report(NAME " " #TMv "x" #TNv ":" #TKv " w" #WMv "x" #WNv " s" #Sv, u, TMv);                                          \
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
        Mmax_, N, K, L, gs, rdev.get(), rsh.data(), ragged ? offdev.get() : nullptr, ws.get(), wsb, nullptr); }, 20);     \
    report(NAME " " #TMv "x" #TNv ":" #TKv " w" #WMv "x" #WNv " s" #Sv, u, TMv);                                          \
    upd(BEST, NAME " " #TMv "x" #TNv ":" #TKv " s" #Sv, u); } while (0)

  // THE TileM SWEEP IS THE POINT. The dense winner is TM=64; the weight-bound rule wants TM as large as possible; masked
  // rows push back. At rows/expert = 128 the m-tile counts are 4x (TM=32), 2x (64), 1x (128) and 1x-with-50%-masked (256).
  std::printf("  --- Q3 = int2 + int1, TileM sweep at the dense winner's TN/TK/warp ---\n");
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1,  32,128, 64,32,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1,  64,128, 64,64,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1, 128,128, 64,64,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1, 256,128, 64,64,64,2,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1, 128,128, 64,64,64,3,2,4);
  MOE2("q3", b2p, uint2_t, uint1_t, 2,1, 128,256, 64,64,64,2,2,4);
  std::printf("  --- Q6 = int4 + int2 ---\n");
  MOE2("q6", b2p, int4_t, uint2_t, 4,2,  64,128, 64,64,64,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2, 128,128, 64,64,64,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2, 256,128, 64,64,64,2,1,2);
  MOE2("q6", b2p, int4_t, uint2_t, 4,2, 128,128, 64,64,32,3,1,2);
  std::printf("  --- Q5 = int4 + int1 ---\n");
  MOE2("q5", b2p, int4_t, uint1_t, 4,1,  64,128, 64,64,64,2,1,4);
  MOE2("q5", b2p, int4_t, uint1_t, 4,1, 128,128, 64,64,64,2,1,4);
  MOE2("q5", b2p, int4_t, uint1_t, 4,1, 256,128, 64,64,64,2,1,4);
  std::printf("  --- single plane: int2 and int4 (the reference) ---\n");
  MOE1("i2", b1p, uint2_t, 2,  64, 64, 64,64,32,2,2);
  MOE1("i2", b1p, uint2_t, 2, 128, 64, 64,64,32,3,2);
  MOE1("i2", b1p, uint2_t, 2, 128,128, 64,64,32,3,2);
  MOE1("i2", b1p, uint2_t, 2, 256,128, 64,64,32,3,2);
  MOE1("i4", b1p, int4_t,  4,  64, 64, 64,64,32,3,1);
  MOE1("i4", b1p, int4_t,  4, 128, 64, 64,64,32,3,1);
  MOE1("i4", b1p, int4_t,  4, 128,128, 64,64,32,3,1);
  MOE1("i4", b1p, int4_t,  4, 256,128, 64,64,32,3,1);

  std::printf("\n  ================= VERDICT =================\n");
  std::printf("  two-plane best : %-40s %8.2f us\n", b2p.tag, b2p.us);
  std::printf("  single-plane   : %-40s %8.2f us\n", b1p.tag, b1p.us);
  std::printf("  READ m_tiles AND masked TOGETHER: if the winner has the smallest m_tiles the weight-bound rule holds;\n");
  std::printf("  if it has the least masking the dense reasoning carries over; if neither, the tile is not the lever.\n");
  return 0;
}
