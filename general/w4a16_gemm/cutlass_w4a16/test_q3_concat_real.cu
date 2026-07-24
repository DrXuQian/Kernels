// Q3_K BIT-PLANE CONCAT on REAL GGUF WEIGHTS [box-only]. Q3_K is natively 2-bit + 1-bit:
//     W = dl*(low2 + 4*high1 - 4),  dl = d*(sc6-32),  gs=16
// and the -4 center folds into the LOW plane's affine zero, so it is exactly two plain W*A16 GEMMs summed:
//     int2 plane:  q=low2  scale=dl     zero=-4*dl
//     int1 plane:  q=high1 scale=4*dl   zero=0
//     D = D_low + D_high  ==  A @ dequant(Q3_K)
// This ONE test therefore validates (a) int1 on REAL GGUF bits, (b) int2 on real bits, and (c) the A-concat.
// The python side (real_weight/dump_q3_concat.py) already checked the decomposition itself: two-plane vs native
// dequant max|d| = 6e-5 (pure fp16 rounding of the scale arrays).
//   .bin: <4 i32: M,N,K,gs> A[M*K]f16 | low2[K*N]u8 | high1[K*N]u8 | sc_lo|zr_lo|sc_hi[(K/gs)*N]f16 | gold[M*N]f16
//   Build: TARGET=test_q3_concat_real ./build.sh ; run: ./<bin> [real_weight/real_q3k_concat.bin]
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

// One plane: transpose q [K][N] -> [N][K], pack ELTS_PER_BYTE per byte, preprocess, run the grouped L=1 kernel.
template <class QuantT, int ELTS_PER_BYTE, QuantTypeClass QTC>
static void run_plane(const std::vector<uint8_t>& q,            // [K][N], values already masked
                      const std::vector<uint16_t>& sc,          // [(K/gs)][N] f16 bits
                      const std::vector<uint16_t>& zr,          // [(K/gs)][N] f16 bits
                      const std::vector<uint16_t>& A_h,         // [M][K] f16 bits
                      int M, int N, int K, int gs,
                      std::vector<half_t>& out /*[M*N]*/) {
  const int scale_k = K / gs, L = 1;
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n * K + k] = q[(size_t)k * N + n];
  std::vector<int8_t> packed((size_t)K * N / ELTS_PER_BYTE, 0);
  const int BITS = 8 / ELTS_PER_BYTE, MASK = (1 << BITS) - 1;
  for (size_t i = 0; i < (size_t)K * N / ELTS_PER_BYTE; ++i) {
    int8_t b = 0;
    for (int t = 0; t < ELTS_PER_BYTE; ++t) b |= int8_t((qT[ELTS_PER_BYTE * i + t] & MASK) << (BITS * t));
    packed[i] = b;
  }
  std::vector<int8_t> Bbuf((size_t)K * N / ELTS_PER_BYTE);
  preprocess_weights_for_mixed_gemm<false, 256>(Bbuf.data(), packed.data(), {(size_t)K, (size_t)N}, QTC);

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<QuantT> dB((size_t)K*N);
  dA.copy_from_host(reinterpret_cast<half_t const*>(A_h.data()));
  dSc.copy_from_host(reinterpret_cast<half_t const*>(sc.data()));
  dZr.copy_from_host(reinterpret_cast<half_t const*>(zr.data()));
  dB.copy_from_host(reinterpret_cast<QuantT const*>(Bbuf.data()));

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

  // TK=256: required by int1 (Block_K%256==0) and legal for int2; gs=16 -> the FINE per-mma-atom scale path.
  moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3, QuantT>(
      dA.get(), dB.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
      M, N, K, L, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  out.resize((size_t)M * N);
  dD.copy_to_host(out.data());
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "real_weight/real_q3k_concat.bin";
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::printf("cannot open %s\n", path); return 1; }
  int32_t hdr[4]; if (fread(hdr, 4, 4, f) != 4) { std::printf("bad header\n"); return 1; }
  const int M = hdr[0], N = hdr[1], K = hdr[2], gs = hdr[3];
  const int scale_k = K / gs;
  std::printf("[q3-concat-real] %s  M=%d N=%d K=%d gs=%d\n", path, M, N, K, gs);

  auto A_h    = rd<uint16_t>(f, (size_t)M * K);
  auto low2   = rd<uint8_t> (f, (size_t)K * N);
  auto high1  = rd<uint8_t> (f, (size_t)K * N);
  auto sc_lo  = rd<uint16_t>(f, (size_t)scale_k * N);
  auto zr_lo  = rd<uint16_t>(f, (size_t)scale_k * N);
  auto sc_hi  = rd<uint16_t>(f, (size_t)scale_k * N);
  auto gold_h = rd<uint16_t>(f, (size_t)M * N);
  std::fclose(f);
  std::vector<uint16_t> zr_hi((size_t)scale_k * N, 0);   // high plane: zero = 0

  std::vector<half_t> Dlo, Dhi;
  run_plane<uint2_t, 4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(low2,  sc_lo, zr_lo, A_h, M, N, K, gs, Dlo);
  run_plane<uint1_t, 8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(high1, sc_hi, zr_hi, A_h, M, N, K, gs, Dhi);

  const half_t* gold = reinterpret_cast<const half_t*>(gold_h.data());
  int bad = 0, shown = 0; double maxrel = 0;
  for (size_t i = 0; i < (size_t)M * N; ++i) {
    double got = (double)float(Dlo[i]) + (double)float(Dhi[i]);      // A-concat: sum the two planes
    double exp = (double)float(gold[i]);
    double rel = std::abs(got - exp) / (std::abs(exp) + 1e-3);
    if (rel > maxrel) maxrel = rel;
    if (std::abs(got - exp) > 2e-2 + 6e-2 * std::abs(exp)) ++bad;
  }
  std::printf("  D_low + D_high vs native Q3_K golden: bad=%d/%d max_rel=%.3e %s\n",
              bad, M * N, maxrel, bad == 0 ? "MATCH" : "MISMATCH");
  for (int m = 0; m < M && shown < 8; ++m) for (int n = 0; n < N && shown < 8; ++n) {
    size_t i = (size_t)m * N + n;
    double got = (double)float(Dlo[i]) + (double)float(Dhi[i]), exp = (double)float(gold[i]);
    if (std::abs(got - exp) > 2e-2 + 6e-2 * std::abs(exp)) {
      std::printf("    m=%d n=%d | lo=%.4f hi=%.4f sum=%.4f exp=%.4f\n",
                  m, n, (double)float(Dlo[i]), (double)float(Dhi[i]), got, exp);
      ++shown;
    }
  }
  return bad == 0 ? 0 : 1;
}
