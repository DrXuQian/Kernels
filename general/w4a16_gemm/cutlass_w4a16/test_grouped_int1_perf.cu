// GROUPED single-plane int1 (and int2 reference) PERFORMANCE verification [box-only].
//
// Why: the recorded int1 285.9us is from the DENSE path (bench_cutlass_w4a16 / fpA_intB); the GROUPED path
// (moe_grouped_ppu, what B-concat uses) has NEVER been perf-verified for int1. The sweep bench showed grouped int1
// ~3052us at N=K=4096 -- 10x the dense record and 6x slower than B-concat, despite int1's 32x128 tile having BETTER
// occupancy than B-concat. This isolates grouped int1 in the CLEANEST possible setup so acu can say why.
//
// Matches the VALIDATED-correct test_w1a16_grouped exactly (FinegrainedScaleOnly, PACKED_INT1, 32x128:256:s3, L=1),
// so ScaleZero-vs-ScaleOnly and any earlier harness quirk are ruled out. int2 (64x64:128:s3, PACKED_INT2) is the
// same-shape reference. ONE kernel per call -> a clean acu target.
//   Build: TARGET=test_grouped_int1_perf ./build.sh ; run: ./<bin> [M] [N] [K] [gs] [which:0=both/1=int1/2=int2]
//   acu:   acu --set full ./<bin> 2048 4096 4096 128 1     # profile grouped int1 alone
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
using uint1_t = cutlass::uint1b_t;
using uint2_t = cutlass::uint2b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

static int M, N, K, gs, scale_k;

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

template <class Fn> static double time_it(Fn fn, int iters) {
  for (int i = 0; i < 3; ++i) fn();
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  hggcEvent_t a, b; hggcEventCreate(&a); hggcEventCreate(&b);
  hggcEventRecord(a); for (int i = 0; i < iters; ++i) fn(); hggcEventRecord(b);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  float ms = 0; hggcEventElapsedTime(&ms, a, b);
  hggcEventDestroy(a); hggcEventDestroy(b);
  return (double)ms * 1e3 / iters;
}

int main(int argc, char** argv) {
  M  = argc > 1 ? atoi(argv[1]) : 2048;
  N  = argc > 2 ? atoi(argv[2]) : 4096;
  K  = argc > 3 ? atoi(argv[3]) : 4096;
  gs = argc > 4 ? atoi(argv[4]) : 128;
  int which = argc > 5 ? atoi(argv[5]) : 0;              // 0 both, 1 int1 only, 2 int2 only (for a clean acu target)
  scale_k = K / gs;
  const double flops = 2.0 * M * N * K;
  std::printf("[grouped-int1-perf] M=%d N=%d K=%d gs=%d  (GROUPED path, L=1, FinegrainedScaleOnly -- matches the "
              "validated test_w1a16_grouped call)\n", M, N, K, gs);

  std::vector<uint8_t> q1((size_t)K*N), q2((size_t)K*N);
  for (size_t i = 0; i < q1.size(); ++i) { q1[i] = (uint8_t)(i & 1); q2[i] = (uint8_t)(i & 3); }
  auto B1 = pack_plane<8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(q1);
  auto B2 = pack_plane<4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(q2);

  cutlass::DeviceAllocation<half_t> A((size_t)M*K), Sc((size_t)scale_k*N), Zr((size_t)scale_k*N), D((size_t)M*N);
  cutlass::DeviceAllocation<uint1_t> dB1((size_t)K*N);
  cutlass::DeviceAllocation<uint2_t> dB2((size_t)K*N);
  { std::vector<half_t> a((size_t)M*K, half_t(0.01f)), s((size_t)scale_k*N, half_t(0.05f)), z((size_t)scale_k*N, half_t(-0.2f));
    A.copy_from_host(a.data()); Sc.copy_from_host(s.data()); Zr.copy_from_host(z.data()); }
  dB1.copy_from_host(reinterpret_cast<uint1_t const*>(B1.data()));
  dB2.copy_from_host(reinterpret_cast<uint2_t const*>(B2.data()));

  std::vector<GS> shp(1, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd(1); shpd.copy_from_host(shp.data());
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M, N, 1))};
  std::vector<int> gmh{M}, offs{0};
  std::vector<half_t*> pdh{D.get()};
  cutlass::DeviceAllocation<half_t*> pd(1); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(1); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(1); gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(1); offdev.copy_from_host(offs.data());
  const size_t wsb = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*64;
  cutlass::DeviceAllocation<char> ws(wsb);

  auto rep = [&](const char* tag, double us){
    double tf = flops/(us*1e-6)/1e12;
    std::printf("  %-30s %8.2f us | %6.1f TFLOP/s (%4.1f%% MFU)\n", tag, us, tf, 100.0*tf*1e12/500.0e12);
  };

  // PROFILE MODE (argv[6]): emit EXACTLY ONE kernel launch (no warmup, no timing loop) so `acu --set full` gets a
  // single clean kernel to profile instead of dozens. 1=int1 ScaleOnly, 2=int1 ScaleZero, 3=int2 ScaleZero.
  //   acu --set full ./<bin> 2048 4096 4096 128 1 2   # profile int1 ScaleZero alone
  int prof = argc > 6 ? atoi(argv[6]) : 0;
  if (prof) {
    if (prof == 1) moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 32,128,256, 32,32, 3, uint1_t>(
        A.get(), dB1.get(), Sc.get(), nullptr, pd.get(), sd.get(), gm.get(),
        M,N,K,1,gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
    else if (prof == 2) moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 32,128,256, 32,32, 3, uint1_t>(
        A.get(), dB1.get(), Sc.get(), Zr.get(), pd.get(), sd.get(), gm.get(),
        M,N,K,1,gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
    else moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64,64,128, 32,32, 3, uint2_t>(
        A.get(), dB2.get(), Sc.get(), Zr.get(), pd.get(), sd.get(), gm.get(),
        M,N,K,1,gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::printf("[profile] emitted ONE kernel (prof=%d) -- for acu\n", prof);
    return 0;
  }

  // ScaleOnly (zeros=nullptr) vs ScaleZero (per-group zeros): the sweep bench used ScaleZero and got int1=3052us,
  // this harness with ScaleOnly got 417us -- SAME config/shape/gs. So the suspect is the zero-loading path for
  // int1 specifically (int2+ScaleZero was fine at ~251us). These four measurements pin it.
  if (which != 2) {
    double u0 = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 32,128,256, 32,32, 3, uint1_t>(
        A.get(), dB1.get(), Sc.get(), nullptr, pd.get(), sd.get(), gm.get(),
        M,N,K,1,gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr); }, 30);
    rep("int1 32x128:256 s3  ScaleOnly", u0);
    double uz = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 32,128,256, 32,32, 3, uint1_t>(
        A.get(), dB1.get(), Sc.get(), Zr.get(), pd.get(), sd.get(), gm.get(),
        M,N,K,1,gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr); }, 30);
    rep("int1 32x128:256 s3  ScaleZero", uz);
    std::printf("    -> int1 zero penalty = %.2fx\n", uz / u0);
  }
  if (which != 1) {
    double u0 = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64,64,128, 32,32, 3, uint2_t>(
        A.get(), dB2.get(), Sc.get(), nullptr, pd.get(), sd.get(), gm.get(),
        M,N,K,1,gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr); }, 30);
    rep("int2 64x64:128  s3  ScaleOnly (ref)", u0);
    double uz = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64,64,128, 32,32, 3, uint2_t>(
        A.get(), dB2.get(), Sc.get(), Zr.get(), pd.get(), sd.get(), gm.get(),
        M,N,K,1,gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr); }, 30);
    rep("int2 64x64:128  s3  ScaleZero (ref)", uz);
    std::printf("    -> int2 zero penalty = %.2fx\n", uz / u0);
  }
  return 0;
}
