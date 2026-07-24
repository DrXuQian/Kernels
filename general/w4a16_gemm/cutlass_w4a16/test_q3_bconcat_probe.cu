// B-CONCAT CONTROLLED-INPUT PROBE [box-only]: decode the ACTUAL (n,k) that each plane's bit came from.
//
// Why a probe and not a local cute-layout derivation: the previous int2 sigma_n bug proved the hardware swzl
// delivery differs from what the cute/sim layout says ("不是 sim 的 v-parity"), so only a controlled input run on
// real hardware settles the register<->element correspondence. Same technique that cracked that bug.
//
// Setup that makes D a direct readout of the weight:  scale=1, zero=0, A = IDENTITY (M=K)
//     D[m][n] = sum_k A[m][k]*W[k][n] = W[m][n] = low_used(m,n) + 4*high_used(m,n)
// so with one plane zeroed, D is EXACTLY the other plane's code, at the (m,n) slot that consumed it.
//
// LADDER (each rung isolates one failure mode; a rung failing tells you to stop and fix THAT):
//   rung 0  low=rand(0..3), high=0        -> golden low            : the low path + affine inside the 2-plane kernel
//   rung 1  low=rand(0..3), high=ALL ONES -> golden low+4          : the high bit's MANTISSA PLACEMENT (the "+4").
//                                                                    Permutation-blind: every high bit is 1, so a
//                                                                    misrouted bit still reads 1. Isolates placement.
//   rung 2  low=rand,       high=rand     -> golden low+4*high     : reproduces the real-weight MISMATCH synthetically
//   rung 3  BIT LABELS: encode the source index in the bit values, read the index back out of D.
//           high plane (1 bit/elt): high[k][n] = (k>>b)&1 for b=0..7  -> k_used(m,n);  then (n>>b)&1 -> n_used(m,n)
//           low  plane (2 bit/elt): low[k][n]  = (k>>2b)&3 for b=0..3 -> k_used;       then (n>>2b)&3 -> n_used
//           Binary encoding over the FULL K/N is injective, which avoids the classic probe trap of a label that
//           aliases inside the tile (k%4 etc. cannot distinguish k from k+4).
//
// Build: TARGET=test_q3_bconcat_probe ./build.sh ; run: ./<bin> [N] [K]   (M is forced = K for the identity A)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
using uint2_t = cutlass::uint2b_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

static int M, N, K, gs = 16, scale_k;

template <int ELTS_PER_BYTE, QuantTypeClass QTC>
static std::vector<int8_t> pack_plane(const std::vector<uint8_t>& q) {
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n * K + k] = q[(size_t)k * N + n];
  const int BITS = 8 / ELTS_PER_BYTE, MASK = (1 << BITS) - 1;
  std::vector<int8_t> packed((size_t)K * N / ELTS_PER_BYTE, 0);
  for (size_t i = 0; i < packed.size(); ++i) {
    int8_t b = 0;
    for (int t = 0; t < ELTS_PER_BYTE; ++t) b |= int8_t((qT[ELTS_PER_BYTE * i + t] & MASK) << (BITS * t));
    packed[i] = b;
  }
  std::vector<int8_t> out(packed.size());
  preprocess_weights_for_mixed_gemm<false, 256>(out.data(), packed.data(), {(size_t)K, (size_t)N}, QTC);
  return out;
}

// device state reused across all runs (only B planes change)
struct Ctx {
  cutlass::DeviceAllocation<half_t>  dA, dSc, dZr, dD;
  cutlass::DeviceAllocation<uint2_t> dBlo;
  cutlass::DeviceAllocation<uint1_t> dBhi;
  cutlass::DeviceAllocation<GS>      shpd;
  cutlass::DeviceAllocation<half_t*> pd;
  cutlass::DeviceAllocation<DStride> sd;
  cutlass::DeviceAllocation<int>     gm, offdev;
  cutlass::DeviceAllocation<char>    ws;
  std::vector<GS> shp;
  size_t wsb;
};

// run the 2-plane GEMM on the given planes, return D on the host
static std::vector<half_t> run(Ctx& c, const std::vector<uint8_t>& low, const std::vector<uint8_t>& high) {
  auto Blo = pack_plane<4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(low);
  auto Bhi = pack_plane<8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(high);
  c.dBlo.copy_from_host(reinterpret_cast<uint2_t const*>(Blo.data()));
  c.dBhi.copy_from_host(reinterpret_cast<uint1_t const*>(Bhi.data()));
  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3,
                                  cutlass::uint2b_t, cutlass::uint1b_t>(
      c.dA.get(), c.dBlo.get(), c.dSc.get(), c.dZr.get(), c.pd.get(), c.sd.get(), c.gm.get(),
      M, N, K, 1, gs, c.shpd.get(), c.shp.data(), c.offdev.get(), c.ws.get(), c.wsb, nullptr,
      c.dBhi.get());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  std::vector<half_t> hD((size_t)M * N);
  c.dD.copy_to_host(hD.data());
  return hD;
}

static void set_scales(Ctx& c, const std::vector<half_t>& sc, const std::vector<half_t>& zr) {
  c.dSc.copy_from_host(sc.data()); c.dZr.copy_from_host(zr.data());
}

// rung 5: with A=identity, D[m][n] must be sc[m/gs][n]*(low+4*high) + zr[m/gs][n]. The rungs above all ran
// scale=1/zero=0, which makes ANY scale-group misindexing invisible -- that is exactly the blind spot that let the
// synthetic rungs pass while the real Q3_K weight (per-group dl and -4dl) still mismatched.
static bool check_affine(const char* tag, const std::vector<half_t>& D,
                         const std::vector<uint8_t>& low, const std::vector<uint8_t>& high,
                         const std::vector<half_t>& sc, const std::vector<half_t>& zr) {
  int bad = 0, shown = 0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    const int g = m / gs;
    double q   = (double)low[(size_t)m * N + n] + 4.0 * (double)high[(size_t)m * N + n];
    double exp = (double)float(sc[(size_t)g * N + n]) * q + (double)float(zr[(size_t)g * N + n]);
    double got = (double)float(D[(size_t)m * N + n]);
    if (std::abs(got - exp) > 2e-2 + 3e-2 * std::abs(exp)) {
      ++bad;
      if (shown < 6) { std::printf("      m=%d n=%d (g=%d) | got=%.4f exp=%.4f  sc=%.4f zr=%.4f q=%.0f\n",
                                   m, n, g, got, exp, (double)float(sc[(size_t)g*N+n]), (double)float(zr[(size_t)g*N+n]), q); ++shown; }
    }
  }
  std::printf("    %-46s bad=%d/%d %s\n", tag, bad, M * N, bad == 0 ? "MATCH" : "MISMATCH");
  return bad == 0;
}

// rung 6: make the SCALE itself the label. low=1, high=0, zero=0, sc[g][n] = 2^g -> D = 2^g_used exactly (fp16 holds
// 2^15), so the scale group each output row actually consumed reads straight out of D.
static void decode_group(Ctx& c) {
  std::vector<uint8_t> low((size_t)K * N, 1), high((size_t)K * N, 0);
  std::vector<half_t> sc((size_t)scale_k * N), zr((size_t)scale_k * N, half_t(0.f));
  for (int g = 0; g < scale_k; ++g) for (int n = 0; n < N; ++n) sc[(size_t)g * N + n] = half_t(float(1 << g));
  set_scales(c, sc, zr);
  auto D = run(c, low, high);
  int bad = 0, shown = 0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    double d = (double)float(D[(size_t)m * N + n]);
    int g_used = (d > 0) ? (int)std::lround(std::log2(d)) : -1;
    const int want = m / gs;
    if (g_used != want) {
      ++bad;
      if (shown < 12) { std::printf("      (m=%3d,n=%3d) g_used=%3d want=%3d  D=%.1f\n", m, n, g_used, want, d); ++shown; }
    }
  }
  std::printf("    SCALE group index: bad=%d/%d %s\n", bad, M * N, bad ? "MISINDEXED" : "identity");
}

// rung 7: DENSE A. Every rung above used A = identity, whose one-nonzero-per-row structure means each m-tile only
// ever exercises the k values equal to its own row indices -- a dense A uses ALL k in EVERY m-tile. That is the
// structural blind spot that let all the synthetic rungs pass while the real Q3_K GEMM (dense A) still mismatched.
// A entries are multiples of 1/4 in [-1,1] so the fp16 products are exact; the host golden accumulates in fp32.
static bool rung7(Ctx& c, const std::vector<uint8_t>& low, const std::vector<uint8_t>& high,
                  const std::vector<half_t>& sc, const std::vector<half_t>& zr, const char* tag) {
  std::vector<half_t> Ad((size_t)M * K);
  for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k)
    Ad[(size_t)m * K + k] = half_t(0.25f * float((int)((m * 31 + k * 17) % 9) - 4));   // {-1,-0.75,..,1}
  c.dA.copy_from_host(Ad.data());
  set_scales(c, sc, zr);
  auto D = run(c, low, high);
  int bad = 0, shown = 0; double maxrel = 0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    double acc = 0;
    for (int k = 0; k < K; ++k) {
      const int g = k / gs;
      double w = (double)float(sc[(size_t)g * N + n]) * ((double)low[(size_t)k * N + n] + 4.0 * (double)high[(size_t)k * N + n])
               + (double)float(zr[(size_t)g * N + n]);
      acc += (double)float(Ad[(size_t)m * K + k]) * w;
    }
    double got = (double)float(D[(size_t)m * N + n]);
    double rel = std::abs(got - acc) / (std::abs(acc) + 1e-3);
    if (rel > maxrel) maxrel = rel;
    if (std::abs(got - acc) > 5e-2 + 5e-2 * std::abs(acc)) {
      ++bad;
      if (shown < 8) { std::printf("      m=%d n=%d | got=%.4f exp=%.4f\n", m, n, got, acc); ++shown; }
    }
  }
  std::printf("    %-46s bad=%d/%d max_rel=%.2e %s\n", tag, bad, M * N, maxrel, bad == 0 ? "MATCH" : "MISMATCH");
  // restore identity A for any later rung
  std::vector<half_t> Ai((size_t)M * K, half_t(0.f));
  for (int m = 0; m < M && m < K; ++m) Ai[(size_t)m * K + m] = half_t(1.f);
  c.dA.copy_from_host(Ai.data());
  return bad == 0;
}

// rung 8: RECOVER the high plane the kernel actually consumed, exactly. With low=0, scale=1, zero=0 and A=identity,
// D[m][n] == 4*high_used[m][n], so high_used is recovered as an exact integer -- no fitting, no guessing. Then the
// map is read off by testing candidates against the recovered plane.
// Why this and not rung2/rung7: those used rhigh = (i*13/3)%2, a SELF-SIMILAR pattern. Measured locally, it still
// agrees 21846/32768 under a candidate permutation, so it is a weak witness and can pass while the plane is wrong --
// which is exactly what happened (the real-weight ablation shows high=0 clean, low=0 broken). rung8 uses a strong
// hash instead, so no permutation can hide.
static uint32_t hbit(int k, int n) {                     // strong 1-bit hash, no self-similarity in k or n
  uint32_t h = (uint32_t)k * 2654435761u ^ (uint32_t)n * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
  return h & 1u;
}
static void rung8(Ctx& c) {
  std::vector<uint8_t> low((size_t)K * N, 0), high((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) high[(size_t)k * N + n] = (uint8_t)hbit(k, n);
  std::vector<half_t> sc1((size_t)scale_k * N, half_t(1.f)), zr0((size_t)scale_k * N, half_t(0.f));
  set_scales(c, sc1, zr0);
  auto D = run(c, low, high);
  // exact recovery: D == 4*high_used
  std::vector<uint8_t> used((size_t)M * N);
  int nonbin = 0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    double d = (double)float(D[(size_t)m * N + n]);
    int v = (int)std::lround(d / 4.0);
    if (std::abs(d - 4.0 * v) > 0.25 || v < 0 || v > 1) ++nonbin;
    used[(size_t)m * N + n] = (uint8_t)(v & 1);
  }
  int bad = 0;
  for (int k = 0; k < K && k < M; ++k) for (int n = 0; n < N; ++n)
    if (used[(size_t)k * N + n] != high[(size_t)k * N + n]) ++bad;
  std::printf("    rung8 strong-random high, recovered exactly: bad=%d/%d  (non-binary D: %d) %s\n",
              bad, (K < M ? K : M) * N, nonbin, bad == 0 ? "MATCH" : "MISMATCH");
  if (bad == 0) { std::printf("      => the high plane is bit-exact even under a strong witness.\n"); return; }
  // name the map: which (k',n') does slot (k,n) actually read?
  struct Cand { const char* name; int dk, dn; bool andmask; };
  static const Cand cands[] = {
    {"n^1",0,1,false},{"n^2",0,2,false},{"n^4",0,4,false},{"n^8",0,8,false},{"n^16",0,16,false},
    {"n^32",0,32,false},{"n^64",0,64,false},{"n^128",0,128,false},
    {"k^1",1,0,false},{"k^2",2,0,false},{"k^4",4,0,false},{"k^8",8,0,false},{"k^16",16,0,false},
    {"k^32",32,0,false},{"k^64",64,0,false},{"k^128",128,0,false},
    {"n&~8",0,8,true},{"n&~16",0,16,true},{"k&~8",8,0,true},{"k&~16",16,0,true},{"k&~64",64,0,true},
  };
  const int rows = (K < M ? K : M);
  for (auto const& cd : cands) {
    int hit = 0;
    for (int k = 0; k < rows; ++k) for (int n = 0; n < N; ++n) {
      int kk = cd.andmask ? (cd.dk ? (k & ~cd.dk) : k) : (k ^ cd.dk);
      int nn = cd.andmask ? (cd.dn ? (n & ~cd.dn) : n) : (n ^ cd.dn);
      if (used[(size_t)k * N + n] == high[(size_t)kk * N + nn]) ++hit;
    }
    if (hit > rows * N * 90 / 100)
      std::printf("      => candidate '%-6s' explains %d/%d\n", cd.name, hit, rows * N);
  }
  // structure of the failures: which n and k residues are affected
  std::printf("      bad-rate by n%%16:");
  for (int r = 0; r < 16; ++r) { int b = 0, t = 0;
    for (int k = 0; k < rows; ++k) for (int n = r; n < N; n += 16) { ++t; if (used[(size_t)k*N+n] != high[(size_t)k*N+n]) ++b; }
    std::printf(" %d:%2d%%", r, 100 * b / (t ? t : 1)); }
  std::printf("\n      bad-rate by k%%16:");
  for (int r = 0; r < 16; ++r) { int b = 0, t = 0;
    for (int k = r; k < rows; k += 16) for (int n = 0; n < N; ++n) { ++t; if (used[(size_t)k*N+n] != high[(size_t)k*N+n]) ++b; }
    std::printf(" %d:%2d%%", r, 100 * b / (t ? t : 1)); }
  std::printf("\n");
}

// rung 0-2: compare D against low + 4*high
static bool check(const char* tag, const std::vector<half_t>& D,
                  const std::vector<uint8_t>& low, const std::vector<uint8_t>& high) {
  int bad = 0, shown = 0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    double got = (double)float(D[(size_t)m * N + n]);
    double exp = (double)low[(size_t)m * N + n] + 4.0 * (double)high[(size_t)m * N + n];
    if (std::abs(got - exp) > 0.25) {
      ++bad;
      if (shown < 6) { std::printf("      m=%d n=%d | got=%.2f exp=%.2f\n", m, n, got, exp); ++shown; }
    }
  }
  std::printf("    %-46s bad=%d/%d %s\n", tag, bad, M * N, bad == 0 ? "MATCH" : "MISMATCH");
  return bad == 0;
}

// rung 3: reconstruct, for every output slot (m,n), the source index that the consumed bit actually carried
enum Axis { AXIS_K, AXIS_N };
static void decode(Ctx& c, bool high_plane, Axis axis) {
  const int nbits   = high_plane ? 1 : 2;                       // label bits carried per element
  const int idx_max = (axis == AXIS_K) ? K : N;
  int rounds = 0; while ((1 << (nbits * rounds)) < idx_max) ++rounds;
  std::vector<int> idx((size_t)M * N, 0);
  std::vector<uint8_t> low((size_t)K * N, 0), high((size_t)K * N, 0);
  for (int r = 0; r < rounds; ++r) {
    const int sh = nbits * r, mask = (1 << nbits) - 1;
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
      const int v = (((axis == AXIS_K) ? k : n) >> sh) & mask;
      (high_plane ? high : low)[(size_t)k * N + n] = (uint8_t)v;
    }
    auto D = run(c, low, high);
    for (size_t i = 0; i < (size_t)M * N; ++i) {
      // high plane contributes 4*bit, low plane contributes its code directly
      int v = (int)std::lround((double)float(D[i]) / (high_plane ? 4.0 : 1.0));
      idx[i] |= (v & mask) << sh;
    }
  }
  // summary: is it the identity map we expect?
  const char* pname = high_plane ? "HIGH(int1)" : "LOW (int2)";
  const char* aname = (axis == AXIS_K) ? "k" : "n";
  int bad = 0, shown = 0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    const int want = (axis == AXIS_K) ? m : n, got = idx[(size_t)m * N + n];
    if (got != want) {
      ++bad;
      if (shown < 12) { std::printf("      (m=%3d,n=%3d) %s_used=%3d want=%3d  delta=%+4d  xor=%3d\n",
                                    m, n, aname, got, want, got - want, got ^ want); ++shown; }
    }
  }
  std::printf("    %s %s_used map: bad=%d/%d %s\n", pname, aname, bad, M * N, bad ? "PERMUTED" : "identity");
  if (bad) {  // test the usual suspects so the fix is named, not guessed
    struct Cand { const char* name; int (*f)(int, int); };
    static const Cand cands[] = {
      {"idx ^ 8",   [](int w, int){ return w ^ 8; }},   {"idx & ~8",  [](int w, int){ return w & ~8; }},
      {"idx ^ 16",  [](int w, int){ return w ^ 16; }},  {"idx ^ 1",   [](int w, int){ return w ^ 1; }},
      {"idx ^ 32",  [](int w, int){ return w ^ 32; }},  {"idx ^ 2",   [](int w, int){ return w ^ 2; }},
      {"idx ^ 64",  [](int w, int){ return w ^ 64; }},  {"idx ^ 128", [](int w, int){ return w ^ 128; }},
    };
    for (auto const& cd : cands) {
      int hit = 0;
      for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        const int want = (axis == AXIS_K) ? m : n;
        if (idx[(size_t)m * N + n] == cd.f(want, 0)) ++hit;
      }
      if (hit > M * N / 2) std::printf("      => matches candidate '%s' on %d/%d\n", cd.name, hit, M * N);
    }
  }
}

int main(int argc, char** argv) {
  N = argc > 1 ? atoi(argv[1]) : 256;
  K = argc > 2 ? atoi(argv[2]) : 256;
  M = K;                                       // identity A needs a square A
  scale_k = K / gs;
  std::printf("[bconcat-probe] M=K=%d N=%d gs=%d  (A=identity, scale=1, zero=0 -> D == the weight code)\n", M, N, gs);

  Ctx c;
  c.dA.reset((size_t)M * K); c.dSc.reset((size_t)scale_k * N); c.dZr.reset((size_t)scale_k * N);
  c.dD.reset((size_t)M * N); c.dBlo.reset((size_t)K * N); c.dBhi.reset((size_t)K * N);
  std::vector<half_t> A_h((size_t)M * K, half_t(0.f));
  for (int m = 0; m < M; ++m) A_h[(size_t)m * K + m] = half_t(1.f);
  std::vector<half_t> sc((size_t)scale_k * N, half_t(1.f)), zr((size_t)scale_k * N, half_t(0.f));
  c.dA.copy_from_host(A_h.data()); c.dSc.copy_from_host(sc.data()); c.dZr.copy_from_host(zr.data());

  c.shp.assign(1, cute::make_shape(M, N, K));
  c.shpd.reset(1); c.shpd.copy_from_host(c.shp.data());
  std::vector<half_t*> pdh{c.dD.get()};
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M, N, 1))};
  std::vector<int> gmh{M}, offs{0};
  c.pd.reset(1); c.pd.copy_from_host(pdh.data());
  c.sd.reset(1); c.sd.copy_from_host(sdh.data());
  c.gm.reset(1); c.gm.copy_from_host(gmh.data());
  c.offdev.reset(1); c.offdev.copy_from_host(offs.data());
  c.wsb = (size_t)cutlass::ceil_div(M, 16) * cutlass::ceil_div(N, 64) * 64;
  c.ws.reset(c.wsb);

  std::vector<uint8_t> rlow((size_t)K * N), rhigh((size_t)K * N), z((size_t)K * N, 0), ones((size_t)K * N, 1);
  for (size_t i = 0; i < rlow.size(); ++i) { rlow[i] = (uint8_t)(i * 7 % 4); rhigh[i] = (uint8_t)((i * 13 / 3) % 2); }

  std::printf("  --- rung 0/1/2: is it PLACEMENT or PERMUTATION? ---\n");
  bool r0 = check("rung0 low=rand high=0     (low path+affine)", run(c, rlow, z), rlow, z);
  bool r1 = check("rung1 low=rand high=1     (the +4 placement)", run(c, rlow, ones), rlow, ones);
  bool r2 = check("rung2 low=rand high=rand  (reproduces the bug)", run(c, rlow, rhigh), rlow, rhigh);
  if (!r0) std::printf("  !! rung0 failed: the fault is NOT the high plane at all -- fix the low path first.\n");
  else if (!r1) std::printf("  !! rung1 failed: the high bit's MANTISSA PLACEMENT is wrong (not a permutation).\n");
  else if (!r2) std::printf("  !! rung2 failed: rungs 0+1 OK => pure PERMUTATION of the high bits (see rung3).\n");
  else std::printf("  ==> rungs 0-2 all OK: q = low + 4*high is exact under a UNIFORM scale.\n");

  // rung 5/6: the blind spot of every rung above -- a VARYING per-group scale/zero.
  std::printf("  --- rung 5/6: per-group scale/zero (invisible to rungs 0-3) ---\n");
  { std::vector<half_t> sc((size_t)scale_k * N), zr((size_t)scale_k * N);
    for (int g = 0; g < scale_k; ++g) for (int n = 0; n < N; ++n) {
      sc[(size_t)g * N + n] = half_t(0.125f * float(1 + (g * 5 + n) % 7));      // varies in BOTH g and n
      zr[(size_t)g * N + n] = half_t(-0.5f  * float(1 + (g * 3 + n) % 5));
    }
    set_scales(c, sc, zr);
    check_affine("rung5 varying scale+zero, both planes", run(c, rlow, rhigh), rlow, rhigh, sc, zr);
  }
  decode_group(c);

  // rung 8: exact recovery of the consumed high plane under a STRONG witness (the headline diagnostic).
  std::printf("  --- rung 8: recover the consumed high plane exactly (strong random) ---\n");
  rung8(c);

  // rung 7: the LAST structural difference from the real test -- a dense A.
  std::printf("  --- rung 7: DENSE A (closes the identity-A blind spot) ---\n");
  { std::vector<half_t> sc1((size_t)scale_k * N, half_t(1.f)), zr0((size_t)scale_k * N, half_t(0.f));
    std::vector<half_t> scv((size_t)scale_k * N), zrv((size_t)scale_k * N);
    for (int g = 0; g < scale_k; ++g) for (int n = 0; n < N; ++n) {
      scv[(size_t)g * N + n] = half_t(0.125f * float(1 + (g * 5 + n) % 7));
      zrv[(size_t)g * N + n] = half_t(-0.5f  * float(1 + (g * 3 + n) % 5));
    }
    std::vector<uint8_t> z2((size_t)K * N, 0);
    bool a1 = rung7(c, rlow, z2,    sc1, zr0, "rung7a denseA, high=0,    uniform scale");
    bool a2 = rung7(c, rlow, rhigh, sc1, zr0, "rung7b denseA, high=rand, uniform scale");
    bool a3 = rung7(c, rlow, rhigh, scv, zrv, "rung7c denseA, high=rand, VARYING scale");
    if (a1 && a2 && a3) std::printf("  ==> dense A is clean too: the kernel is correct on synthetic data end-to-end,\n"
                                    "      so the real-weight MISMATCH is in the .bin/golden path, not the kernel.\n");
    else if (a1 && !a2)  std::printf("  !! dense A + high plane fails while high=0 passes => HIGH-PLANE k coverage bug\n"
                                    "      (only the k's on the identity diagonal were ever right).\n");
    else if (!a1)        std::printf("  !! dense A fails even with high=0 => the fault is in the SHARED path under dense A.\n");
    else                 std::printf("  !! only the VARYING-scale dense case fails => scale group vs k under dense A.\n");
  }
  { std::vector<half_t> sc1((size_t)scale_k * N, half_t(1.f)), zr0((size_t)scale_k * N, half_t(0.f));
    set_scales(c, sc1, zr0); }   // restore uniform for rung 3

  std::printf("  --- rung 3: decode the source index each consumed bit carried ---\n");
  decode(c, false, AXIS_K);   // low plane as CONTROL (validated standalone -> must be identity)
  decode(c, false, AXIS_N);
  decode(c, true,  AXIS_K);   // the suspect
  decode(c, true,  AXIS_N);
  return 0;
}
