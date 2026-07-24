// Q3_K B-CONCAT (ONE GEMM, two B bit planes) on REAL GGUF weights [box-only].
//
// The dedicated 2-plane mainloop loads BOTH planes and combines them in the converter, so a real Q3_K weight runs
// as a SINGLE GEMM while staying 3-bit in memory:
//     low  plane: int2  = q & 3            (drives the main swzl / tCrB_mma)
//     high plane: int1  = q >> 2
//     converter emits fp16(low + 4*high) -> the affine then does dl*q + zero, zero = -4*dl (Q3_K's -4 center)
// so ONLY sc_lo (= dl) and zr_lo (= -4*dl) are used here; sc_hi from the .bin is NOT needed, because the factor 4
// is baked into the high bit's mantissa placement (b+2) instead of into a second scale.
//
// Compare against the A-concat (two GEMMs summed, test_q3_concat_real): same golden, but that costs 2x the mma.
//   .bin: <4 i32: M,N,K,gs> A[M*K]f16 | low2[K*N]u8 | high1[K*N]u8 | sc_lo|zr_lo|sc_hi[(K/gs)*N]f16 | gold[M*N]f16
//   Build: TARGET=test_q3_bconcat_real ./build.sh ; run: ./<bin> real_weight/real_q3k_concat.bin
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

template <class T> static std::vector<T> rd(FILE* f, size_t n) {
  std::vector<T> v(n);
  if (fread(v.data(), sizeof(T), n, f) != n) { std::printf("short read\n"); exit(1); }
  return v;
}

// transpose q [K][N] -> [N][K], pack ELTS_PER_BYTE per byte, run the offline preprocess for that plane's format
template <int ELTS_PER_BYTE, QuantTypeClass QTC>
static std::vector<int8_t> pack_plane(const std::vector<uint8_t>& q, int K, int N) {
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

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "real_weight/real_q3k_concat.bin";
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::printf("cannot open %s\n", path); return 1; }
  int32_t hdr[4]; if (fread(hdr, 4, 4, f) != 4) { std::printf("bad header\n"); return 1; }
  const int M = hdr[0], N = hdr[1], K = hdr[2], gs = hdr[3], L = 1;
  const int scale_k = K / gs;
  std::printf("[q3-bconcat-real] %s  M=%d N=%d K=%d gs=%d  (ONE GEMM, int2 low + int1 high)\n",
              path, M, N, K, gs);

  auto A_h    = rd<uint16_t>(f, (size_t)M * K);
  auto low2   = rd<uint8_t> (f, (size_t)K * N);
  auto high1  = rd<uint8_t> (f, (size_t)K * N);
  auto sc_lo  = rd<uint16_t>(f, (size_t)scale_k * N);
  auto zr_lo  = rd<uint16_t>(f, (size_t)scale_k * N);
  /*sc_hi unused*/ rd<uint16_t>(f, (size_t)scale_k * N);
  auto gold_h = rd<uint16_t>(f, (size_t)M * N);
  std::fclose(f);

  auto Blo = pack_plane<4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(low2,  K, N);
  auto Bhi = pack_plane<8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(high1, K, N);

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dBlo((size_t)K*N);
  cutlass::DeviceAllocation<uint1_t> dBhi((size_t)K*N);
  dA.copy_from_host(reinterpret_cast<half_t const*>(A_h.data()));
  dSc.copy_from_host(reinterpret_cast<half_t const*>(sc_lo.data()));
  dZr.copy_from_host(reinterpret_cast<half_t const*>(zr_lo.data()));
  dBlo.copy_from_host(reinterpret_cast<uint2_t const*>(Blo.data()));
  dBhi.copy_from_host(reinterpret_cast<uint1_t const*>(Bhi.data()));

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

  // Shared Block_K = 256: bounded below by the SPARSEST plane's AIU 32B minimum (int1 -> K/8 >= 32 -> K >= 256).
  // PlaneB2 = uint1b_t as the 8th template arg routes the builder to the 2-plane mainloop; the high plane's device
  // pointer is the trailing argument.
  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3,
                                  cutlass::uint2b_t, cutlass::uint1b_t>(
      dA.get(), dBlo.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr,
      dBhi.get());
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<half_t> hD((size_t)M*N); dD.copy_to_host(hD.data());
  const half_t* gold = reinterpret_cast<const half_t*>(gold_h.data());
  int bad = 0, shown = 0; double maxrel = 0;
  for (size_t i = 0; i < (size_t)M * N; ++i) {
    double got = (double)float(hD[i]), exp = (double)float(gold[i]);
    double rel = std::abs(got - exp) / (std::abs(exp) + 1e-3);
    if (rel > maxrel) maxrel = rel;
    if (std::abs(got - exp) > 2e-2 + 6e-2 * std::abs(exp)) ++bad;
  }
  std::printf("  B-concat (1 GEMM) vs native Q3_K golden: bad=%d/%d max_rel=%.3e %s\n",
              bad, M * N, maxrel, bad == 0 ? "MATCH" : "MISMATCH");
  // On MISMATCH the SHAPE of the error localizes it: a whole-K shift or a low/high N-half swap points at the
  // plane-2 swzl lane delivery (the one thing the local xplane.py check could not cover); a per-element factor of
  // ~4 or a 0/4 offset points at the high bit's mantissa placement instead.
  for (int m = 0; m < M && shown < 8; ++m) for (int n = 0; n < N && shown < 8; ++n) {
    size_t i = (size_t)m * N + n;
    double got = (double)float(hD[i]), exp = (double)float(gold[i]);
    if (std::abs(got - exp) > 2e-2 + 6e-2 * std::abs(exp)) {
      std::printf("    m=%d n=%d | got=%.4f exp=%.4f ratio=%.3f\n", m, n, got, exp, exp != 0 ? got / exp : 0.0);
      ++shown;
    }
  }
  return bad == 0 ? 0 : 1;
}
