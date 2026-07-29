// Q6 (int4+int2) and Q5 (int4+int1) B-CONCAT -- ONE GEMM, two B bit planes [box-only].
//
// Q3 = int2+int1 already ships. These are the other two formats the bit-plane decomposition promised, and they run on
// the SAME mainloop: MixGemm2Plane is one closed form over (LowBits, HiBits), gated against Q3's shipped constants in
// fold_derivation/l65, emulated for all three formats in l66, and its offline (tile_map_hi) checked byte-identical for
// Q3 and a complete bijection for Q6/Q5 in l67. So there is no new index algebra here to be wrong -- what remains
// unverified until this test runs is the PLUMBING: builder fold factors, gmem strides, and the scale path.
//
// THE int4 BIAS, and why Q3 did not need this. The int4 converter subtracts 8 (the shipped 1032/72 constants), so with
// low = q & 15 and high = q >> 4 the emitted value is
//     (q & 15) - 8 + 16 * (q >> 4)  ==  q - 8         for EVERY q, no wrapping
// i.e. a uniform -8 offset, which the caller folds into the zero point. int2 carries no bias, so Q3's zero is just
// Q3_K's -4 centre. Here zero = -8*dl exactly, so the dequantised weight is dl*q and the golden is a plain sum -- any
// bias error shows up as a systematic offset rather than hiding in the data.
//
// The low plane is written by place_derived DIRECTLY, not through preprocess_weights_for_mixed_gemm: the shim adds int4's
// +8 to reproduce the legacy pipeline, and here the stored code must be q & 15 with the bias handled by the zero point.
//
//   Build: TARGET=test_q65_bconcat_real ./build.sh          (PPU_DEFS=PPU_B_CHUNK=1 to exercise the chunked path)
//   Run:   ./<bin>                                          (synthetic, full code range, CPU golden)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "xplane_offline.hpp"
#include "moe_grouped_ppu.cuh"

using half_t  = cutlass::half_t;
using int4_t  = cutlass::int4b_t;
using uint2_t = cutlass::uint2b_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

static int M = 128, N = 256, K = 256, gs = 32, L = 1;

int main(int argc, char** argv) {
  if (argc > 1) M = atoi(argv[1]);
  if (argc > 2) N = atoi(argv[2]);
  if (argc > 3) K = atoi(argv[3]);
  if (argc > 4) gs = atoi(argv[4]);
  const int scale_k = K / gs;
  std::printf("[q65-bconcat] M=%d N=%d K=%d gs=%d   Q6 = int4+int2 (q in [0,64)), Q5 = int4+int1 (q in [0,32))\n",
              M, N, K, gs);

  // ---- A, scales. THE ZERO POINT MUST CANCEL int4's BIAS, WITH A PLUS SIGN. apply_scale_atom is `multiplies` then
  // `plus`, i.e. w = dl * emitted + zero, and the converter emits q - 8, so w = dl*q needs zero = +8*dl.
  //
  // My first version wrote -8*dl and every Q6/Q5 configuration came out bad=32768/32768 with IDENTICAL values across
  // configurations -- and, decisively, with got - exp = -16 * sum_k A*dl carrying NO q dependence at all (the six
  // printed columns were exact multiples 1,7,6,5,4,3 of one another, which is the shape of the dl generator). That
  // forces sum_k A*dl*(emitted - q) = -8 * sum_k A*dl for every (m,n), hence emitted == q - 8 exactly. In other words
  // the failure PROVED the two-plane combination lo + 16*hi was already right on hardware and only the test's sign was
  // wrong. A whole-output MISMATCH whose residual is independent of the quantised data is never the kernel.
  //
  // A real Q6_K would put its own centre here instead (w = d*sc*(q - 32) means zero = (8 - 32)*d*sc); zero = +8*dl is
  // chosen so the dequantised weight is exactly dl*q and the golden is a plain sum.
  std::vector<half_t> A((size_t)M * K), Sc((size_t)scale_k * N), Zr((size_t)scale_k * N);
  for (size_t i = 0; i < A.size(); ++i)  A[i]  = half_t(float(int((i * 2654435761u >> 9) & 7) - 3) * 0.125f);
  for (size_t i = 0; i < Sc.size(); ++i) {
    const float dl = float(int((i * 40503u >> 3) & 7) + 1) * 0.0625f;   // varying, period coprime to 8/16/32
    Sc[i] = half_t(dl);
    Zr[i] = half_t(+8.0f * dl);                                        // cancels the converter's -8
  }

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N), dD((size_t)M*N);
  dA.copy_from_host(A.data()); dSc.copy_from_host(Sc.data()); dZr.copy_from_host(Zr.data());
  std::vector<GS> shp(L, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd(L); shpd.copy_from_host(shp.data());
  std::vector<half_t*> pdh{dD.get()};
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M, N, 1))};
  std::vector<int> gmh{M}, offs{0};
  cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(L);     gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  const size_t wsb = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*(size_t)L*64;
  cutlass::DeviceAllocation<char> ws(wsb);

  int fails = 0;

  // BITS = the HIGH plane's width. Q6 -> 2, Q5 -> 1. The low plane is int4 either way, so the code range is
  // 16 << HiBits and the split is (q & 15, q >> 4).
  auto run_format = [&](const char* fmt, int HiBits, auto hi_elem_tag, auto&& rungs) {
    const int qmax = 16 << HiBits;
    std::vector<uint8_t> lo((size_t)K * N), hi((size_t)K * N);      // [K][N], one code per byte
    std::vector<int> q((size_t)K * N);
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = int((i * 2654435761u >> 5) % (unsigned)qmax);          // FULL range, so the top plane is exercised
      lo[i] = uint8_t(q[i] & 15);
      hi[i] = uint8_t(q[i] >> 4);
    }
    // CPU golden in double: D = sum_k A * (dl * q)
    std::vector<double> gold((size_t)M * N, 0.0);
    for (int k = 0; k < K; ++k)
      for (int m = 0; m < M; ++m) {
        const double a = double(float(A[(size_t)m * K + k]));
        if (a == 0.0) continue;
        for (int n = 0; n < N; ++n) {
          const double dl = double(float(Sc[(size_t)(k / gs) * N + n]));
          gold[(size_t)m * N + n] += a * dl * double(q[(size_t)k * N + n]);
        }
      }
    std::printf("  --- %s ---\n", fmt);
    rungs(lo, hi, gold);
  };

  auto check = [&](const char* tag, const std::vector<double>& gold) {
    std::vector<half_t> h((size_t)M*N); dD.copy_to_host(h.data());
    int bad = 0; double mr = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
      const double got = double(float(h[i])), exp = gold[i];
      const double rel = std::abs(got - exp) / (std::abs(exp) + 1e-3);
      if (rel > 2e-2) { if (bad < 6) std::printf("      i=%zu got=%.4f exp=%.4f rel=%.3e\n", i, got, exp, rel); ++bad; }
      if (rel > mr) mr = rel;
    }
    std::printf("    %-42s bad=%d/%d max_rel=%.3e %s\n", tag, bad, M * N, mr, bad ? "MISMATCH" : "MATCH");
    if (bad) ++fails;
    return bad == 0;
  };

#define RUNG65(TAG, HIB, HIELEM, TMv, TNv, TKv, WMv, WNv, Sv, F2v) do {                              \
    std::vector<int8_t> blo((size_t)K * N / 2), bhi((size_t)K * N * (HIB) / 8);                       \
    xplane::place_derived<4, TMv, TNv, TKv, WMv, WNv, 1>(blo.data(), lo, N, K);                       \
    xplane::place_hi<4, HIB, TMv, TNv, TKv, WMv, WNv, F2v, 1>(bhi.data(), hi, N, K);                  \
    cutlass::DeviceAllocation<int4_t> dblo((size_t)K*N);                                              \
    dblo.copy_from_host(reinterpret_cast<int4_t const*>(blo.data()));                                 \
    cutlass::DeviceAllocation<HIELEM> dbhi((size_t)K*N);                                               \
    dbhi.copy_from_host(reinterpret_cast<HIELEM const*>(bhi.data()));                                 \
    CUTLASS_PPU_CHECK(hggcMemset(dD.get(), 0, sizeof(half_t) * (size_t)M * N));                       \
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, TMv, TNv, TKv, WMv, WNv, Sv,            \
                                    int4_t, HIELEM>(                                                  \
        dA.get(), dblo.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),                     \
        M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr, dbhi.get());    \
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());                                                       \
    check(TAG, gold);                                                                                 \
  } while (0)

  // Q6 = int4 + int2. int4's run is 32 B at TK=64 already, so the LOW plane never folds; int2 folds by 2 at TK=64.
  // int2's delivery bound is WN >= 2048/TK, so TK=64 needs WN >= 32 -- w64x32 IS legal, the shape that paid +7 to +9
  // points on the single-plane sweeps.
  run_format("Q6 = int4 + int2", 2, uint2_t{}, [&](auto& lo, auto& hi, auto& gold) {
    RUNG65("(64,128,128) w32x32 s3  F1=1 F2=1", 2, uint2_t, 64, 128, 128, 32, 32, 3, 1);
    RUNG65("(64,128,128) w64x32 s3  F1=1 F2=1", 2, uint2_t, 64, 128, 128, 64, 32, 3, 1);
    RUNG65("(64,128, 64) w64x32 s3  F1=1 F2=2", 2, uint2_t, 64, 128,  64, 64, 32, 3, 2);
    RUNG65("(64,128, 64) w64x64 s2  F1=1 F2=2", 2, uint2_t, 64, 128,  64, 64, 64, 2, 2);
    RUNG65("(64, 64,128) w32x32 s3  F1=1 F2=1", 2, uint2_t, 64,  64, 128, 32, 32, 3, 1);
  });

  // Q5 = int4 + int1. int1's bound is WN >= 4096/TK, so TK=256 allows w32x32, TK=128 needs WN >= 32, TK=64 needs 64.
  run_format("Q5 = int4 + int1", 1, uint1_t{}, [&](auto& lo, auto& hi, auto& gold) {
    RUNG65("(64,128,256) w32x32 s3  F1=1 F2=1", 1, uint1_t, 64, 128, 256, 32, 32, 3, 1);
    RUNG65("(64,128,128) w32x32 s3  F1=1 F2=2", 1, uint1_t, 64, 128, 128, 32, 32, 3, 2);
    RUNG65("(64,128,128) w64x32 s3  F1=1 F2=2", 1, uint1_t, 64, 128, 128, 64, 32, 3, 2);
    RUNG65("(64,128, 64) w64x64 s2  F1=1 F2=4", 1, uint1_t, 64, 128,  64, 64, 64, 2, 4);
  });

  std::printf("\n  => %d failing configuration(s)\n", fails);
  return fails ? 1 : 0;
}
