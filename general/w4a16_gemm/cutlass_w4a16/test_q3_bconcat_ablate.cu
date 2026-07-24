// Q3_K B-CONCAT ABLATION on REAL data [box-only]: run the 2-plane kernel on the REAL .bin numbers, but with a
// HOST golden recomputed for each ablation, to localize the real-weight MISMATCH the synthetic probe could not
// reproduce. Established so far: A-concat (two validated single-plane kernels) is bad=0 on this same .bin, so every
// INPUT is good and the fault is in the 2-plane collective, triggered only by the real high-plane VALUES (the
// synthetic rhigh in the probe is self-similar enough to mask it). This test uses the real values.
//
// Ablations (all use the REAL low/high/sc/zr; the host golden is recomputed to match each):
//   A  real low + real high, real A     -> must reproduce MISMATCH; also asserts host golden == .bin gold (formula OK)
//   B  real low + high=0,     real A     -> single-plane int2 equivalent; isolates the LOW path in the 2-plane kernel
//   C  low=0    + real high,  real A     -> isolates the HIGH plane (the prime suspect)
//   D  real low + real high, SYNTH dense A-> isolates whether the real A matters (it should not; A-concat used it fine)
//   Build: TARGET=test_q3_bconcat_ablate ./build.sh ; run: ./<bin> real_weight/real_q3k_concat.bin
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

static int M, N, K, gs, scale_k;

template <class T> static std::vector<T> rd(FILE* f, size_t n) {
  std::vector<T> v(n);
  if (fread(v.data(), sizeof(T), n, f) != n) { std::printf("short read\n"); exit(1); }
  return v;
}

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
  const char* path = argc > 1 ? argv[1] : "real_weight/real_q3k_concat.bin";
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::printf("cannot open %s\n", path); return 1; }
  int32_t hdr[4]; if (fread(hdr, 4, 4, f) != 4) { std::printf("bad header\n"); return 1; }
  M = hdr[0]; N = hdr[1]; K = hdr[2]; gs = hdr[3]; scale_k = K / gs;
  std::printf("[q3-bconcat-ablate] %s  M=%d N=%d K=%d gs=%d\n", path, M, N, K, gs);
  auto A_h   = rd<uint16_t>(f, (size_t)M * K);
  auto low2  = rd<uint8_t>(f, (size_t)K * N);
  auto high1 = rd<uint8_t>(f, (size_t)K * N);
  auto sc_h  = rd<uint16_t>(f, (size_t)scale_k * N);
  auto zr_h  = rd<uint16_t>(f, (size_t)scale_k * N);
  /*sc_hi*/ rd<uint16_t>(f, (size_t)scale_k * N);
  auto gold  = rd<uint16_t>(f, (size_t)M * N);
  std::fclose(f);

  const half_t* A_real = reinterpret_cast<const half_t*>(A_h.data());
  const half_t* sc     = reinterpret_cast<const half_t*>(sc_h.data());
  const half_t* zr     = reinterpret_cast<const half_t*>(zr_h.data());
  const half_t* goldp  = reinterpret_cast<const half_t*>(gold.data());
  // synthetic dense A (ablation D): multiples of 1/4 in [-1,1] -> exact fp16 products
  std::vector<half_t> A_syn((size_t)M * K);
  for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k)
    A_syn[(size_t)m * K + k] = half_t(0.25f * float((int)((m * 31 + k * 17) % 9) - 4));

  cutlass::DeviceAllocation<half_t>  dA((size_t)M*K), dSc((size_t)scale_k*N), dZr((size_t)scale_k*N), dD((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> dBlo((size_t)K*N);
  cutlass::DeviceAllocation<uint1_t> dBhi((size_t)K*N);
  dSc.copy_from_host(sc); dZr.copy_from_host(zr);
  std::vector<GS> shp(1, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd(1); shpd.copy_from_host(shp.data());
  std::vector<half_t*> pdh{dD.get()};
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M, N, 1))};
  std::vector<int> gmh{M}, offs{0};
  cutlass::DeviceAllocation<half_t*> pd(1); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(1); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm(1); gm.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> offdev(1); offdev.copy_from_host(offs.data());
  const size_t wsb = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*64;
  cutlass::DeviceAllocation<char> ws(wsb);

  auto run_case = [&](const char* tag, const half_t* A, const std::vector<uint8_t>& low,
                      const std::vector<uint8_t>& high, const uint16_t* cmp_gold) {
    auto Blo = pack_plane<4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(low);
    auto Bhi = pack_plane<8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(high);
    dA.copy_from_host(A);
    dBlo.copy_from_host(reinterpret_cast<uint2_t const*>(Blo.data()));
    dBhi.copy_from_host(reinterpret_cast<uint1_t const*>(Bhi.data()));
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 256, 32, 32, 3,
                                    cutlass::uint2b_t, cutlass::uint1b_t>(
        dA.get(), dBlo.get(), dSc.get(), dZr.get(), pd.get(), sd.get(), gm.get(),
        M, N, K, 1, gs, shpd.get(), shp.data(), offdev.get(), ws.get(), wsb, nullptr, dBhi.get());
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::vector<half_t> hD((size_t)M*N); dD.copy_to_host(hD.data());
    // host golden for these exact inputs
    std::vector<double> hg((size_t)M*N, 0);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) {
        int g = k / gs;
        double w = (double)float(sc[(size_t)g*N+n]) * ((double)low[(size_t)k*N+n] + 4.0*(double)high[(size_t)k*N+n])
                 + (double)float(zr[(size_t)g*N+n]);
        acc += (double)float(A[(size_t)m*K+k]) * w;
      }
      hg[(size_t)m*N+n] = acc;
    }
    int bad = 0, shown = 0; double mr = 0;
    for (size_t i = 0; i < (size_t)M*N; ++i) {
      double got = (double)float(hD[i]), exp = hg[i];
      double rel = std::abs(got-exp)/(std::abs(exp)+1e-3); if (rel>mr) mr=rel;
      if (std::abs(got-exp) > 5e-2 + 5e-2*std::abs(exp)) { ++bad;
        if (shown<6){ std::printf("      i=%zu m=%zu n=%zu | got=%.4f exp=%.4f\n", i, i/N, i%N, got, exp); ++shown; } }
    }
    std::printf("  %-40s kernel vs HOST golden: bad=%d/%d max_rel=%.2e %s\n",
                tag, bad, M*N, mr, bad==0?"MATCH":"MISMATCH");
    if (cmp_gold) {  // does my host formula reproduce the .bin's own golden? (sanity: formula + read are right)
      int gb = 0; double gm2 = 0;
      for (size_t i = 0; i < (size_t)M*N; ++i) {
        double a = hg[i], b = (double)float(reinterpret_cast<const half_t*>(cmp_gold)[i]);
        double rel = std::abs(a-b)/(std::abs(b)+1e-3); if (rel>gm2) gm2=rel;
        if (std::abs(a-b) > 5e-2 + 5e-2*std::abs(b)) ++gb;
      }
      std::printf("     (host golden vs .bin gold: bad=%d/%d max_rel=%.2e -> formula %s)\n",
                  gb, M*N, gm2, gb==0?"CONFIRMED":"WRONG");
    }
    return bad == 0;
  };

  std::vector<uint8_t> z((size_t)K*N, 0);
  std::printf("  --- ablations (all REAL low/high/sc/zr values) ---\n");
  bool A = run_case("A real low + real high, real A",   A_real, low2, high1, gold.data());
  bool B = run_case("B real low + high=0,    real A",   A_real, low2, z,     nullptr);
  bool C = run_case("C low=0    + real high,  real A",   A_real, z,    high1, nullptr);
  bool D = run_case("D real low + real high, SYNTH A",   A_syn.data(), low2, high1, nullptr);
  std::printf("  --- verdict ---\n");
  if (A) std::printf("  A MATCH: the real-weight bug did NOT reproduce here -> suspect .bin gold read or A layout.\n");
  else if (B && !C) std::printf("  ==> HIGH PLANE is the fault: low-only (B) clean, high-only (C) broken on real data.\n");
  else if (!B)      std::printf("  ==> LOW path in the 2-plane kernel breaks under a dense real A (B failed).\n");
  else if (C && !D) std::printf("  ==> triggered by the REAL A specifically (synth A clean).\n");
  else              std::printf("  A=%d B=%d C=%d D=%d -- read the per-case lines above.\n", A,B,C,D);
  return 0;
}
