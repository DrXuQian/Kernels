// Q3_K B-CONCAT vs A-CONCAT speed [box-only]. Numerics are already settled (test_q3_bconcat_real bad=0); this only
// times. Both routes run on the SAME synthetic large shape in the SAME process, INTERLEAVED (B,A,B,A,...), because on
// the shared box only interleaved same-session A/B is trustworthy (see ppu-w2a16 GOTCHA4). Weights are synthetic --
// speed is data-independent for these kernels.
//   B-concat: ONE 2-plane GEMM (int2 low + int1 high), 3-bit in memory.
//   A-concat: TWO validated single-plane GEMMs (int2 then int1), summed -> the oracle/fallback, ~2x the mma.
//   Build: TARGET=test_q3_bconcat_bench ./build.sh ; run: ./<bin> [M] [N] [K]   (defaults 2048 4096 4096, gs=16)
#include <cstdio>
#include <cstdlib>
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

template <int EPB, QuantTypeClass QTC>
static std::vector<int8_t> pack_plane(const std::vector<uint8_t>& q) {
  std::vector<int> qT((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) qT[(size_t)n * K + k] = q[(size_t)k * N + n];
  const int BITS = 8 / EPB, MASK = (1 << BITS) - 1;
  std::vector<int8_t> packed((size_t)K * N / EPB, 0);
  for (size_t i = 0; i < packed.size(); ++i) {
    int8_t b = 0;
    for (int t = 0; t < EPB; ++t) b |= int8_t((qT[EPB * i + t] & MASK) << (BITS * t));
    packed[i] = b;
  }
  std::vector<int8_t> out(packed.size());
  preprocess_weights_for_mixed_gemm<false, 256>(out.data(), packed.data(), {(size_t)K, (size_t)N}, QTC);
  return out;
}

int main(int argc, char** argv) {
  M = argc > 1 ? atoi(argv[1]) : 2048;
  N = argc > 2 ? atoi(argv[2]) : 4096;
  K = argc > 3 ? atoi(argv[3]) : 4096;
  scale_k = K / gs;
  std::printf("[q3-bconcat-bench] M=%d N=%d K=%d gs=%d  (B-concat = 1 GEMM 3-bit; A-concat = 2 GEMMs)\n", M, N, K, gs);

  std::vector<uint8_t> low((size_t)K*N), high((size_t)K*N);
  for (size_t i = 0; i < low.size(); ++i) { low[i] = (uint8_t)(i % 4); high[i] = (uint8_t)((i / 4) % 2); }
  auto Blo = pack_plane<4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(low);
  auto Bhi = pack_plane<8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(high);

  cutlass::DeviceAllocation<half_t> dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N),
                                    dD((size_t)M*N), dDhi((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dBlo((size_t)K*N);
  cutlass::DeviceAllocation<uint1_t> dBhi((size_t)K*N);
  { std::vector<half_t> a((size_t)M*K, half_t(0.01f)), s((size_t)scale_k*N, half_t(0.05f)), z((size_t)scale_k*N, half_t(-0.2f));
    dA.copy_from_host(a.data()); dSc.copy_from_host(s.data()); dZr.copy_from_host(z.data()); }
  dBlo.copy_from_host(reinterpret_cast<uint2_t const*>(Blo.data()));
  dBhi.copy_from_host(reinterpret_cast<uint1_t const*>(Bhi.data()));

  std::vector<GS> shp(1, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd(1); shpd.copy_from_host(shp.data());
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M, N, 1))};
  std::vector<int> gmh{M}, offs{0};
  cutlass::DeviceAllocation<DStride> sd(1); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(1); gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(1); offdev.copy_from_host(offs.data());
  std::vector<half_t*> pdh{dD.get()}, pdhi{dDhi.get()};
  cutlass::DeviceAllocation<half_t*> pd(1); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<half_t*> pdh2(1); pdh2.copy_from_host(pdhi.data());
  const size_t wsb = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*64;
  cutlass::DeviceAllocation<char> ws(wsb);

  auto bconcat = [&]{  // ONE 2-plane GEMM
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3, uint2_t, uint1_t>(
        dA.get(), dBlo.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
        M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr, dBhi.get());
  };
  auto aconcat = [&]{  // TWO single-plane GEMMs (int2 winner 64x64:32x32:s3 TK=128; int1 winner 32x128:32x32:s3 TK=256)
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 128, 32, 32, 3, uint2_t>(
        dA.get(), dBlo.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
        M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 32, 128, 256, 32, 32, 3, uint1_t>(
        dA.get(), dBhi.get(), dSc.get(), dZr.get(), pdh2.get(), sd.get(), gm.get(),
        M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
  };

  auto time_it = [&](auto fn, int iters) -> double {
    for (int i = 0; i < 3; ++i) fn();                      // warmup
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    hggcEvent_t a, b; hggcEventCreate(&a); hggcEventCreate(&b);
    hggcEventRecord(a); for (int i = 0; i < iters; ++i) fn(); hggcEventRecord(b);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    float ms = 0; hggcEventElapsedTime(&ms, a, b);
    hggcEventDestroy(a); hggcEventDestroy(b);
    return (double)ms * 1e3 / iters;                       // us
  };

  const double flops = 2.0 * M * N * K;                    // one A@W
  const double PEAK = 197.0e12;                            // ppu001 fp16 tensor peak (approx, for MFU%)
  auto report = [&](const char* tag, double us, double wbytes){
    double tf = flops / (us * 1e-6) / 1e12;
    std::printf("  %-24s %8.2f us | %6.1f TFLOP/s (%4.1f%% MFU) | weight %5.1f MB @ %6.0f GB/s\n",
                tag, us, tf, 100.0*tf*1e12/PEAK, wbytes/1e6, wbytes/(us*1e-6)/1e9);
  };
  const double wb_b = (double)K*N*3/8;                     // 3-bit
  const double wb_a = (double)K*N*2/8 + (double)K*N*1/8;   // int2 + int1 = same 3-bit total, but 2 passes

  std::printf("  interleaved B,A,B,A,... (only same-session A/B is trustworthy on the shared box)\n");
  for (int r = 0; r < 4; ++r) {
    double tb = time_it(bconcat, 50);
    double ta = time_it(aconcat, 50);
    std::printf("  --- round %d ---\n", r);
    report("B-concat (1 GEMM)", tb, wb_b);
    report("A-concat (2 GEMMs)", ta, wb_a);
    std::printf("  %-24s B/A = %.2fx\n", "speedup", ta / tb);
  }
  return 0;
}
