// MULTI-EXPERT (L>1) + RAGGED gate for every LOW-BIT format on the grouped path [box-only]:
//   two-plane    Q3 = int2+int1, Q6 = int4+int2, Q5 = int4+int1
//   single-plane int2, int1, int4
//
// Named for what it covers. It was test_q65_grouped, and that read as "Q6 and Q5 only" even though Q3 was the first
// entry -- the name is the only part a reader sees, so a name that misdescribes the coverage is a real defect.
//
// WHY THIS EXISTS. Everything measured for the two-plane formats so far runs L=1. The plumbing L>1 adds is per-expert
// ADDRESSING, and the two-plane path has twice as much of it as a single-plane one: a second weight base pointer with its
// OWN L-stride. In the interleaved branch that stride is a hand-written byte count,
//     int64_t(N) * int64_t(K) * sizeof_bits<PlaneB2>::value / 8
// and the collective's own comment records what a wrong version did: "every expert read plane 0; wrong scale -> e>=2 read
// OOB garbage". Nothing has ever tested it.
//
// THE ORACLE IS THE L=1 KERNEL, not a CPU golden -- the same instrument test_w1a16_grouped uses. Run all L experts
// together into a contiguous D[total][N], then run each expert ALONE at L=1 and compare. Any difference is per-expert
// addressing and nothing else, because the arithmetic is identical between the two runs.
//
// EVERY EXPERT GETS DIFFERENT WEIGHTS, in both planes and in the scales. With identical experts, "every expert read
// plane 0" passes silently -- which is exactly the defect the comment above describes. The codes are seeded on e.
//
// int1 IS A CHECK ON THE HARNESS, not on the kernel. It already has an independent multi-expert gate
// (test_w1a16_grouped) that passes, so int1 failing here while passing there means this harness is wrong. A new test
// with no known-good row in it can only tell you that something is broken, never which of the two.
//
//   Build: TARGET=test_lowbit_grouped ./build.sh      (PPU_DEFS=PPU_B_CHUNK=1 for the chunked path)
//   Run:   ./<bin> [L] [ragged 0|1] [Mb] [N] [K] [gs]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
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


// SUB-BYTE POINTER ARITHMETIC IS IN BYTES, NOT ELEMENTS -- and that is what broke the first version of this test.
// cutlass::uint2b_t / uint1b_t / int4b_t all have sizeof == 1 (a byte-sized wrapper holding 2, 1 or 4 bits), so
// `ptr + n` on one of them advances n BYTES. Offsetting an expert by e*K*N therefore over-advanced by 4x for int2, 8x
// for int1 and 2x for int4, and expert 1 already read past the end of the allocation.
//
// The failure was legible before any debugging, and only because the harness contains a row whose answer is already
// known: EVERY format failed including int1, which has an independent multi-expert gate that passes. And the grouped run
// returned the SAME value for e=1 at L=4 and L=8 while the ORACLE returned two different ones -- so the thing that varied
// was the reference, not the subject. A new test with no known-good row could not have said which side was wrong.
template <class T>
static T const* expert_ptr(T const* base, size_t e, size_t bytes_per_expert) {
  return reinterpret_cast<T const*>(reinterpret_cast<int8_t const*>(base) + e * bytes_per_expert);
}
template <class T>
static T* expert_ptr(T* base, size_t e, size_t bytes_per_expert) {
  return reinterpret_cast<T*>(reinterpret_cast<int8_t*>(base) + e * bytes_per_expert);
}

static int L = 4, Mb = 64, N = 256, K = 256, gs = 32;
static bool ragged = true;

int main(int argc, char** argv) {
  if (argc > 1) L      = atoi(argv[1]);
  if (argc > 2) ragged = atoi(argv[2]) != 0;
  if (argc > 3) Mb     = atoi(argv[3]);
  if (argc > 4) N      = atoi(argv[4]);
  if (argc > 5) K      = atoi(argv[5]);
  if (argc > 6) gs     = atoi(argv[6]);
  const int scale_k = K / gs;

  std::vector<int> me(L), offs(L);
  int total = 0, Mmax = 0;
  for (int e = 0; e < L; ++e) {
    me[e] = ragged ? Mb * ((e % 4) + 1) : Mb;
    offs[e] = total; total += me[e]; Mmax = std::max(Mmax, me[e]);
  }
  std::printf("[lowbit-grouped] %s  L=%d %s Mb=%d N=%d K=%d gs=%d total=%d Mmax=%d  (two-plane, per-expert addressing)\n",
              PPU_CHUNK_STR, L, ragged ? "ragged" : "uniform", Mb, N, K, gs, total, Mmax);

  // ---- A and the scales. Both are [total][K] and [L][scale_k][N], i.e. the layouts the grouped path expects.
  std::vector<half_t> hA((size_t)total * K), hSc((size_t)L * scale_k * N), hZr((size_t)L * scale_k * N);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = half_t(float(int((i * 2654435761u >> 9) & 7) - 3) * 0.125f);
  for (int e = 0; e < L; ++e)
    for (int i = 0; i < scale_k * N; ++i) {
      // seeded on e as well, so a wrong scale L-stride cannot pass
      const float dl = float(int(((size_t(e) * 7919 + i) * 40503u >> 3) & 7) + 1) * 0.0625f;
      hSc[(size_t)e * scale_k * N + i] = half_t(dl);
      hZr[(size_t)e * scale_k * N + i] = half_t(+8.0f * dl);   // cancels int4's -8; harmless for int2 (bias 0) if unused
    }

  cutlass::DeviceAllocation<half_t> dA((size_t)total * K), dSc((size_t)L * scale_k * N),
                                    dZr((size_t)L * scale_k * N), dD((size_t)total * N), dD1((size_t)total * N);
  dA.copy_from_host(hA.data()); dSc.copy_from_host(hSc.data()); dZr.copy_from_host(hZr.data());

  auto out_stride = [&](int m) { return cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(m, N, 1)); };
  int fails = 0;

  // LowBits/HiBits pick the format; TM..S the tile. F1/F2 are the two planes' fold factors at this Block_K.
#define GROUPED(TAG, LOWELEM, HIELEM, LOWB, HIB, TMv, TNv, TKv, WMv, WNv, Sv, F1v, F2v, QMODE) do {                    \
    const size_t lo_per = (size_t)K * N * (LOWB) / 8, hi_per = (size_t)K * N * (HIB) / 8;                              \
    std::vector<int8_t> blo((size_t)L * lo_per), bhi((size_t)L * hi_per);                                              \
    for (int e = 0; e < L; ++e) {                                                                                      \
      std::vector<uint8_t> lo((size_t)K * N), hi((size_t)K * N);                                                       \
      for (size_t i = 0; i < lo.size(); ++i) {                                                                         \
        const int q = int(((size_t(e) * 104729 + i) * 2654435761u >> 5) % (unsigned)((1 << (LOWB)) << (HIB)));          \
        lo[i] = uint8_t(q & ((1 << (LOWB)) - 1));                                                                       \
        hi[i] = uint8_t(q >> (LOWB));                                                                                    \
      }                                                                                                                 \
      xplane::place_derived<LOWB, TMv, TNv, TKv, WMv, WNv, F1v>(blo.data() + (size_t)e * lo_per, lo, N, K);             \
      xplane::place_hi<LOWB, HIB, TMv, TNv, TKv, WMv, WNv, F2v, F1v>(bhi.data() + (size_t)e * hi_per, hi, N, K);        \
    }                                                                                                                   \
    /* SIZED IN BYTES, because sizeof(uint2b_t) == 1: DeviceAllocation<LOWELEM>(L*K*N) would allocate L*K*N BYTES and
       copy_from_host would then read L*K*N bytes out of a host buffer holding only L*K*N*BITS/8 -- an out-of-bounds host
       read. L*lo_per is both the exact host size and exactly what the kernel's element-based L-stride resolves to. */    \
    cutlass::DeviceAllocation<LOWELEM> dblo((size_t)L * lo_per);                                                         \
    dblo.copy_from_host(reinterpret_cast<LOWELEM const*>(blo.data()));                                                  \
    cutlass::DeviceAllocation<HIELEM> dbhi((size_t)L * hi_per);                                                          \
    dbhi.copy_from_host(reinterpret_cast<HIELEM const*>(bhi.data()));                                                   \
    /* ---- the full L>1 run, contiguous D[total][N] via the ptr-array epilogue */                                       \
    std::vector<GS> rsh(L); std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);          \
    for (int e = 0; e < L; ++e) { rsh[e] = cute::make_shape(me[e], N, K); pdh[e] = dD.get() + (size_t)offs[e] * N;       \
                                  sdh[e] = out_stride(me[e]); gmh[e] = me[e]; }                                          \
    cutlass::DeviceAllocation<GS> rdev(L);       rdev.copy_from_host(rsh.data());                                       \
    cutlass::DeviceAllocation<half_t*> pd(L);    pd.copy_from_host(pdh.data());                                          \
    cutlass::DeviceAllocation<DStride> sd(L);    sd.copy_from_host(sdh.data());                                          \
    cutlass::DeviceAllocation<int> gm(L);        gm.copy_from_host(gmh.data());                                          \
    cutlass::DeviceAllocation<int> offdev(L);    offdev.copy_from_host(offs.data());                                     \
    const size_t wsb = (size_t)cutlass::ceil_div(Mmax,16)*cutlass::ceil_div(N,64)*(size_t)L*64;                          \
    cutlass::DeviceAllocation<char> ws(wsb);                                                                             \
    CUTLASS_PPU_CHECK(hggcMemset(dD.get(), 0, sizeof(half_t) * (size_t)total * N));                                     \
    moe_grouped_ppu::filter_and_run<QMODE, TMv, TNv, TKv, WMv, WNv, Sv, LOWELEM, HIELEM>(                               \
        dA.get(), dblo.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                                       \
        Mmax, N, K, L, gs, rdev.get(), rsh.data(), ragged ? offdev.get() : nullptr, ws.get(), wsb, nullptr, dbhi.get()); \
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());                                                                          \
    /* ---- the ORACLE: each expert alone at L=1, into the same contiguous rows */                                       \
    CUTLASS_PPU_CHECK(hggcMemset(dD1.get(), 0, sizeof(half_t) * (size_t)total * N));                                    \
    for (int e = 0; e < L; ++e) {                                                                                        \
      const int Me = me[e];                                                                                             \
      std::vector<GS> s1{cute::make_shape(Me, N, K)};                                                                    \
      std::vector<half_t*> p1{dD1.get() + (size_t)offs[e] * N};                                                          \
      std::vector<DStride> t1{out_stride(Me)}; std::vector<int> g1{Me}, o1{0};                                           \
      cutlass::DeviceAllocation<GS> s1d(1);       s1d.copy_from_host(s1.data());                                         \
      cutlass::DeviceAllocation<half_t*> p1d(1);  p1d.copy_from_host(p1.data());                                          \
      cutlass::DeviceAllocation<DStride> t1d(1);  t1d.copy_from_host(t1.data());                                          \
      cutlass::DeviceAllocation<int> g1d(1);      g1d.copy_from_host(g1.data());                                          \
      cutlass::DeviceAllocation<int> o1d(1);      o1d.copy_from_host(o1.data());                                          \
      const size_t w1b = (size_t)cutlass::ceil_div(Me,16)*cutlass::ceil_div(N,64)*64;                                    \
      cutlass::DeviceAllocation<char> w1(w1b);                                                                            \
      moe_grouped_ppu::filter_and_run<QMODE, TMv, TNv, TKv, WMv, WNv, Sv, LOWELEM, HIELEM>(                             \
          dA.get() + (size_t)offs[e] * K, expert_ptr(dblo.get(), (size_t)e, lo_per),                                    \
          dSc.get() + (size_t)e * scale_k * N, dZr.get() + (size_t)e * scale_k * N,                                     \
          p1d.get(), t1d.get(), g1d.get(), Me, N, K, 1, gs, s1d.get(), s1.data(), nullptr, w1.get(), w1b, nullptr,      \
          expert_ptr(dbhi.get(), (size_t)e, hi_per));                                                                               \
      CUTLASS_PPU_CHECK(hggcDeviceSynchronize());                                                                        \
    }                                                                                                                    \
    std::vector<half_t> h((size_t)total*N), h1((size_t)total*N);                                                         \
    dD.copy_to_host(h.data()); dD1.copy_to_host(h1.data());                                                              \
    int bad = 0, worst_e = -1; double mr = 0;                                                                             \
    for (int e = 0; e < L; ++e)                                                                                          \
      for (int i = 0; i < me[e]; ++i)                                                                                    \
        for (int j = 0; j < N; ++j) {                                                                                     \
          const size_t idx = ((size_t)offs[e] + i) * N + j;                                                              \
          const double a = double(float(h[idx])), b = double(float(h1[idx]));                                            \
          const double rel = std::abs(a - b) / (std::abs(b) + 1e-3);                                                     \
          if (rel > 2e-2) { if (bad < 4) std::printf("      e=%d i=%d j=%d grouped=%.4f oracle=%.4f\n", e, i, j, a, b);  \
                            ++bad; }                                                                                      \
          if (rel > mr) { mr = rel; worst_e = e; }                                                                        \
        }                                                                                                                 \
    std::printf("    %-46s bad=%d/%d max_rel=%.3e (worst e=%d) %s\n", TAG, bad, total * N, mr, worst_e,                  \
                bad ? "MISMATCH" : "MATCH");                                                                              \
    if (bad) ++fails;                                                                                                     \
  } while (0)

  // Single-plane variant: identical structure, no B2 argument. Same L=1-per-expert oracle, so it isolates the same
  // thing -- per-expert addressing -- for the formats that ship without a companion plane.
#define GROUPED1(TAG, ELEM, BITS, TMv, TNv, TKv, WMv, WNv, Sv, Fv, QMODE) do {                                          \
    const size_t per = (size_t)K * N * (BITS) / 8;                                                                       \
    std::vector<int8_t> bb((size_t)L * per);                                                                             \
    for (int e = 0; e < L; ++e) {                                                                                        \
      std::vector<uint8_t> q((size_t)K * N);                                                                             \
      for (size_t i = 0; i < q.size(); ++i)                                                                              \
        q[i] = uint8_t(((size_t(e) * 104729 + i) * 2654435761u >> 5) & ((1u << (BITS)) - 1u));                            \
      xplane::place_derived<BITS, TMv, TNv, TKv, WMv, WNv, Fv>(bb.data() + (size_t)e * per, q, N, K);                     \
    }                                                                                                                    \
    cutlass::DeviceAllocation<ELEM> db((size_t)L * per);   /* bytes -- see the note in the two-plane macro */            \
    db.copy_from_host(reinterpret_cast<ELEM const*>(bb.data()));                                                          \
    std::vector<GS> rsh(L); std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);            \
    for (int e = 0; e < L; ++e) { rsh[e] = cute::make_shape(me[e], N, K); pdh[e] = dD.get() + (size_t)offs[e] * N;         \
                                  sdh[e] = out_stride(me[e]); gmh[e] = me[e]; }                                            \
    cutlass::DeviceAllocation<GS> rdev(L);       rdev.copy_from_host(rsh.data());                                         \
    cutlass::DeviceAllocation<half_t*> pd(L);    pd.copy_from_host(pdh.data());                                            \
    cutlass::DeviceAllocation<DStride> sd(L);    sd.copy_from_host(sdh.data());                                            \
    cutlass::DeviceAllocation<int> gm(L);        gm.copy_from_host(gmh.data());                                            \
    cutlass::DeviceAllocation<int> offdev(L);    offdev.copy_from_host(offs.data());                                       \
    const size_t wsb = (size_t)cutlass::ceil_div(Mmax,16)*cutlass::ceil_div(N,64)*(size_t)L*64;                            \
    cutlass::DeviceAllocation<char> ws(wsb);                                                                               \
    CUTLASS_PPU_CHECK(hggcMemset(dD.get(), 0, sizeof(half_t) * (size_t)total * N));                                       \
    moe_grouped_ppu::filter_and_run<QMODE, TMv, TNv, TKv, WMv, WNv, Sv, ELEM>(                                            \
        dA.get(), db.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                                          \
        Mmax, N, K, L, gs, rdev.get(), rsh.data(), ragged ? offdev.get() : nullptr, ws.get(), wsb, nullptr);              \
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());                                                                            \
    CUTLASS_PPU_CHECK(hggcMemset(dD1.get(), 0, sizeof(half_t) * (size_t)total * N));                                      \
    for (int e = 0; e < L; ++e) {                                                                                          \
      const int Me = me[e];                                                                                              \
      std::vector<GS> s1{cute::make_shape(Me, N, K)};                                                                     \
      std::vector<half_t*> p1{dD1.get() + (size_t)offs[e] * N};                                                            \
      std::vector<DStride> t1{out_stride(Me)}; std::vector<int> g1{Me}, o1{0};                                             \
      cutlass::DeviceAllocation<GS> s1d(1);       s1d.copy_from_host(s1.data());                                           \
      cutlass::DeviceAllocation<half_t*> p1d(1);  p1d.copy_from_host(p1.data());                                            \
      cutlass::DeviceAllocation<DStride> t1d(1);  t1d.copy_from_host(t1.data());                                            \
      cutlass::DeviceAllocation<int> g1d(1);      g1d.copy_from_host(g1.data());                                            \
      const size_t w1b = (size_t)cutlass::ceil_div(Me,16)*cutlass::ceil_div(N,64)*64;                                      \
      cutlass::DeviceAllocation<char> w1(w1b);                                                                              \
      moe_grouped_ppu::filter_and_run<QMODE, TMv, TNv, TKv, WMv, WNv, Sv, ELEM>(                                          \
          dA.get() + (size_t)offs[e] * K, expert_ptr(db.get(), (size_t)e, per),                                          \
          dSc.get() + (size_t)e * scale_k * N, dZr.get() + (size_t)e * scale_k * N,                                      \
          p1d.get(), t1d.get(), g1d.get(), Me, N, K, 1, gs, s1d.get(), s1.data(), nullptr, w1.get(), w1b, nullptr);      \
      CUTLASS_PPU_CHECK(hggcDeviceSynchronize());                                                                          \
    }                                                                                                                      \
    std::vector<half_t> h((size_t)total*N), h1((size_t)total*N);                                                           \
    dD.copy_to_host(h.data()); dD1.copy_to_host(h1.data());                                                                \
    int bad = 0, worst_e = -1; double mr = 0;                                                                               \
    for (int e = 0; e < L; ++e)                                                                                            \
      for (int i = 0; i < me[e]; ++i)                                                                                      \
        for (int j = 0; j < N; ++j) {                                                                                       \
          const size_t idx = ((size_t)offs[e] + i) * N + j;                                                                \
          const double a = double(float(h[idx])), b = double(float(h1[idx]));                                              \
          const double rel = std::abs(a - b) / (std::abs(b) + 1e-3);                                                       \
          if (rel > 2e-2) { if (bad < 4) std::printf("      e=%d i=%d j=%d grouped=%.4f oracle=%.4f\n", e, i, j, a, b);   \
                            ++bad; }                                                                                        \
          if (rel > mr) { mr = rel; worst_e = e; }                                                                          \
        }                                                                                                                   \
    std::printf("    %-46s bad=%d/%d max_rel=%.3e (worst e=%d) %s\n", TAG, bad, total * N, mr, worst_e,                    \
                bad ? "MISMATCH" : "MATCH");                                                                                \
    if (bad) ++fails;                                                                                                       \
  } while (0)

  std::printf("  --- Q3 = int2 + int1 ---\n");
  GROUPED("(64,128, 64) w64x64 s2  F1=2 F2=4", uint2_t, uint1_t, 2, 1, 64,128, 64,64,64,2,2,4, QM::FinegrainedScaleZero);
  GROUPED("(64,128,128) w32x64 s3  F1=1 F2=2", uint2_t, uint1_t, 2, 1, 64,128,128,32,64,3,1,2, QM::FinegrainedScaleZero);
  std::printf("  --- Q6 = int4 + int2 ---\n");
  GROUPED("(64,128, 64) w64x64 s2  F1=1 F2=2", int4_t, uint2_t, 4, 2, 64,128, 64,64,64,2,1,2, QM::FinegrainedScaleZero);
  GROUPED("(64,128,128) w32x32 s3  F1=1 F2=1", int4_t, uint2_t, 4, 2, 64,128,128,32,32,3,1,1, QM::FinegrainedScaleZero);
  std::printf("  --- Q5 = int4 + int1 ---\n");
  GROUPED("(64,128, 64) w64x64 s2  F1=1 F2=4", int4_t, uint1_t, 4, 1, 64,128, 64,64,64,2,1,4, QM::FinegrainedScaleZero);
  GROUPED("(64,128,128) w32x32 s3  F1=1 F2=2", int4_t, uint1_t, 4, 1, 64,128,128,32,32,3,1,2, QM::FinegrainedScaleZero);

  std::printf("  --- single plane: int2 (the gap), int1 (HARNESS check), int4 ---\n");
  GROUPED1("int2 (64, 64, 64) w64x32 s2  F=2", uint2_t, 2, 64, 64, 64,64,32,2,2, QM::FinegrainedScaleZero);
  GROUPED1("int2 (64,128, 64) w64x32 s3  F=2", uint2_t, 2, 64,128, 64,64,32,3,2, QM::FinegrainedScaleZero);
  GROUPED1("int2 (64, 64,128) w32x32 s3  F=1", uint2_t, 2, 64, 64,128,32,32,3,1, QM::FinegrainedScaleZero);
  GROUPED1("int1 (64,128, 64) w64x64 s2  F=4", uint1_t, 1, 64,128, 64,64,64,2,4, QM::FinegrainedScaleZero);
  GROUPED1("int4 (64, 64, 64) w64x32 s3  F=1", int4_t,  4, 64, 64, 64,64,32,3,1, QM::FinegrainedScaleZero);

  std::printf("\n  => %d failing configuration(s)\n", fails);
  return fails ? 1 : 0;
}
