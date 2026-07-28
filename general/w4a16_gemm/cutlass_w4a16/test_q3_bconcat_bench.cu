// Q3_K B-CONCAT vs A-CONCAT speed + CONFIG SWEEP [box-only]. Numerics settled (test_q3_bconcat_real bad=0); this
// times. acu on the 64x64:256:s3 B-concat showed Duration=1.04ms == wall-clock, occupancy 12.5% LIMITED BY SHARED
// MEMORY (2 blocks/CU), Memory-Dependency-bound -- so absolute wall-clock IS the kernel time (host overhead ~0), and
// the lever is shared-memory usage. TK is locked at 256 by the int1 plane, so the way to cut A-smem (32KB/stage at
// TM64) and lift occupancy is smaller TileM / fewer stages. This sweeps those.
//
// A-concat is measured HONESTLY: sweep int2 and int1 SEPARATELY, take each one's best, and SUM -- not two hardcoded
// (gs=128-winner) configs run back-to-back, which is what made the earlier 4196us bogus.
//   Build: TARGET=test_q3_bconcat_bench ./build.sh ; run: ./<bin> [M] [N] [K] [gs]  (defaults 2048 4096 4096 16)
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
using int4_t  = cutlass::int4b_t;   // ceiling reference: int4@TK64 IS the geometry int2/int1-fold target
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

static int M, N, K, gs = 16, scale_k;

template <int EPB, QuantTypeClass QTC, int FoldTN = 0, int FoldTK = 0>
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
  // PER-PLANE N-FOLD: a plane whose contiguous run at this TileShape.K is under the AIU's 32 B minimum must be folded
  // in N, and each plane folds by its OWN factor -- that is the whole point of the change. FoldTK == 0 means "this
  // config does not fold", which reproduces the previous buffer byte for byte.
  if (FoldTK > 0) {
    const int contig = FoldTK * BITS / 8, F = contig >= 32 ? 1 : 32 / contig;
    if (F > 1) {
      std::vector<int8_t> f(out.size());
      nfold_regroup_gmem(f.data(), out.data(), {(size_t)K, (size_t)N}, FoldTN, FoldTK, BITS);
      out.swap(f);
    }
  }
  return out;
}

// globals filled in main, so the sweep macros stay short
static cutlass::DeviceAllocation<half_t>  *dA, *dSc, *dZr, *dD, *dDhi;
static cutlass::DeviceAllocation<uint2_t> *dBlo;
static cutlass::DeviceAllocation<int4_t>  *dB4;
static cutlass::DeviceAllocation<uint1_t> *dBhi;
static cutlass::DeviceAllocation<uint1_t> *dBhi128;   // int1 folded F=2 for Block_K=128 (per-plane fold)
static cutlass::DeviceAllocation<GS>       *shpd;
static cutlass::DeviceAllocation<half_t*>  *pd, *pd2;
static cutlass::DeviceAllocation<DStride>  *sd;
static cutlass::DeviceAllocation<int>      *gm, *offdev;
static cutlass::DeviceAllocation<char>     *ws;
static std::vector<GS>                     *shpv;
static size_t                               wsb;

template <class Fn> static double time_it(Fn fn, int iters) {
  for (int i = 0; i < 3; ++i) fn();
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  hggcEvent_t a, b; hggcEventCreate(&a); hggcEventCreate(&b);
  hggcEventRecord(a); for (int i = 0; i < iters; ++i) fn(); hggcEventRecord(b);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  float ms = 0; hggcEventElapsedTime(&ms, a, b);
  hggcEventDestroy(a); hggcEventDestroy(b);
  return (double)ms * 1e3 / iters;                       // us
}

static double flops;
static void report(const char* tag, double us) {
  double tf = flops / (us * 1e-6) / 1e12;
  std::printf("    %-28s %8.2f us | %6.1f TFLOP/s (%4.1f%% MFU)\n", tag, us, tf, 100.0*tf*1e12/500.0e12);
}

struct Best { char tag[32]; double us; };
static void upd(Best& b, const char* t, double u) { if (u < b.us) { b.us = u; std::snprintf(b.tag, 32, "%s", t); } }

// B-concat (2 planes): TK must be 256 (int1 plane). Vary TileM / TileN / stages to trade A-smem for occupancy.
#define BC(TM,TN,TK,WM,WN,S) do { \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TM,TN,TK,WM,WN,S,uint2_t,uint1_t>( \
      dA->get(), dBlo->get(), dSc->get(), dZr->get(), pd->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr, dBhi->get()); }, 30); \
  report("BC " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S, u); upd(bBC, #TM "x" #TN ":" #TK " s" #S, u); } while (0)

// PER-PLANE FOLD: Block_K=128 is now reachable -- int2 keeps F=1 (32 B run) while int1 folds F=2. That halves A-smem
// against TK=256 (the whole reason B-concat sat at 12.5% occupancy) and needs a SEPARATELY FOLDED high plane, hence
// dBhi128 rather than dBhi.
#define BC128(TM,TN,TK,WM,WN,S) do { \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TM,TN,TK,WM,WN,S,uint2_t,uint1_t>( \
      dA->get(), dBlo->get(), dSc->get(), dZr->get(), pd->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr, dBhi128->get()); }, 30); \
  report("BC " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S " [per-plane fold]", u); \
  upd(bBC, #TM "x" #TN ":" #TK " s" #S " fold", u); } while (0)

#define I2(TM,TN,TK,WM,WN,S) do { \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TM,TN,TK,WM,WN,S,uint2_t>( \
      dA->get(), dBlo->get(), dSc->get(), dZr->get(), pd->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr); }, 30); \
  report("i2 " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S, u); upd(bI2, #TM "x" #TN ":" #TK " s" #S, u); } while (0)

// A-concat's int1 plane has zero == 0 (the Q3_K -4 center folds entirely into the LOW/int2 plane), so it runs
// ScaleOnly -- NOT ScaleZero. The earlier sweep wrongly forced ScaleZero here, dragging int1 onto the 7x
// FINE-per-atom-zero path (3052us) and making A-concat look 6x worse than it is. Real A-concat int1 is ScaleOnly.
// int4 CEILING reference (ScaleOnly). int4@TK64 is exactly the A-smem/occupancy/tile geometry that int2-fold@TK64
// and int1-fold@TK64 are engineered to reach (TM64, TK64, A-smem 8KB, ~50% occ, full MMA reuse). Also shows int4's
// own TK sensitivity (TK64 vs TK128) => whether folding int4 further would even help.
#define I4(TM,TN,TK,WM,WN,S) do { \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly,TM,TN,TK,WM,WN,S,int4_t>( \
      dA->get(), dB4->get(), dSc->get(), nullptr, pd->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr); }, 30); \
  report("i4 " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S, u); upd(bI4, #TM "x" #TN ":" #TK " s" #S, u); } while (0)

#define I1(TM,TN,TK,WM,WN,S) do { \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly,TM,TN,TK,WM,WN,S,uint1_t>( \
      dA->get(), dBhi->get(), dSc->get(), nullptr, pd2->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr); }, 30); \
  report("i1 " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S " (ScaleOnly)", u); upd(bI1, #TM "x" #TN ":" #TK " s" #S, u); } while (0)

int main(int argc, char** argv) {
  M = argc > 1 ? atoi(argv[1]) : 2048;
  N = argc > 2 ? atoi(argv[2]) : 4096;
  K = argc > 3 ? atoi(argv[3]) : 4096;
  gs = argc > 4 ? atoi(argv[4]) : 16;
  scale_k = K / gs;
  flops = 2.0 * M * N * K;
  std::printf("[q3-bconcat-bench] M=%d N=%d K=%d gs=%d  (config sweep; PEAK=500 TFLOP/s; wall-clock==acu here)\n",
              M, N, K, gs);

  std::vector<uint8_t> low((size_t)K*N), high((size_t)K*N);
  for (size_t i = 0; i < low.size(); ++i) { low[i] = (uint8_t)(i % 4); high[i] = (uint8_t)((i / 4) % 2); }
  auto Blo = pack_plane<4, QuantTypeClass::PACKED_INT2_WEIGHT_ONLY>(low);
  auto Bhi = pack_plane<8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY>(high);

  cutlass::DeviceAllocation<half_t> A_((size_t)M*K), S_((size_t)scale_k*N), Z_((size_t)scale_k*N),
                                    D_((size_t)M*N), Dh_((size_t)M*N);
  cutlass::DeviceAllocation<uint2_t> Blo_((size_t)K*N);
  std::vector<uint8_t> q4((size_t)K*N); for (size_t i=0;i<q4.size();++i) q4[i]=(uint8_t)(i%16);
  auto B4 = pack_plane<2, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY>(q4);
  cutlass::DeviceAllocation<int4_t> B4_((size_t)K*N); B4_.copy_from_host(reinterpret_cast<int4_t const*>(B4.data()));
  cutlass::DeviceAllocation<uint1_t> Bhi_((size_t)K*N);
  auto Bhi128 = pack_plane<8, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY, 128, 128>(high);   // int1 F=2 at Block_K=128
  cutlass::DeviceAllocation<uint1_t> Bhi128_((size_t)K*N);
  { std::vector<half_t> a((size_t)M*K, half_t(0.01f)), s((size_t)scale_k*N, half_t(0.05f)), z((size_t)scale_k*N, half_t(-0.2f));
    A_.copy_from_host(a.data()); S_.copy_from_host(s.data()); Z_.copy_from_host(z.data()); }
  Blo_.copy_from_host(reinterpret_cast<uint2_t const*>(Blo.data()));
  Bhi_.copy_from_host(reinterpret_cast<uint1_t const*>(Bhi.data()));
  Bhi128_.copy_from_host(reinterpret_cast<uint1_t const*>(Bhi128.data()));

  std::vector<GS> shp(1, cute::make_shape(M, N, K));
  cutlass::DeviceAllocation<GS> shpd_(1); shpd_.copy_from_host(shp.data());
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(M, N, 1))};
  std::vector<int> gmh{M}, offs{0};
  cutlass::DeviceAllocation<DStride> sd_(1); sd_.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int> gm_(1); gm_.copy_from_host(gmh.data());
  cutlass::DeviceAllocation<int> off_(1); off_.copy_from_host(offs.data());
  std::vector<half_t*> pdh{D_.get()}, pdhi{Dh_.get()};
  cutlass::DeviceAllocation<half_t*> pd_(1); pd_.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<half_t*> pd2_(1); pd2_.copy_from_host(pdhi.data());
  wsb = (size_t)cutlass::ceil_div(M,16)*cutlass::ceil_div(N,64)*64;
  cutlass::DeviceAllocation<char> ws_(wsb);

  dA=&A_; dSc=&S_; dZr=&Z_; dD=&D_; dDhi=&Dh_; dBlo=&Blo_; dBhi=&Bhi_; dBhi128=&Bhi128_; dB4=&B4_; shpd=&shpd_;
  pd=&pd_; pd2=&pd2_; sd=&sd_; gm=&gm_; offdev=&off_; ws=&ws_; shpv=&shp;

  Best bBC{"",1e18}, bI2{"",1e18}, bI1{"",1e18}, bI4{"",1e18};

  std::printf("  --- B-concat sweep (TK locked 256; smaller TileM / fewer stages cut A-smem to lift occupancy) ---\n");
  BC(64,64,256,32,32,3);   // baseline (acu: 12.5% occ, shared-limited)
  BC(64,64,256,32,32,2);   // stages 3->2 : smem x2/3
  BC(32,64,256,32,32,3);   // TileM 64->32 : A-smem halved
  BC(32,64,256,32,32,2);
  BC(32,32,256,32,32,3);   // smallest A tile
  BC(32,128,256,32,32,3);  // wide N (int1's own winner shape)
  BC(16,128,256,16,32,3);  // very small M
  BC(16,64,256,16,32,3);   // TileM=16 min, keep A-smem floor
  BC(16,256,256,16,32,3);  // TileM=16 + widest N: tiny A-smem, big tile area (best occ/reuse trade)

  std::printf("  --- B-concat with PER-PLANE FOLD (Block_K 128: int2 F=1, int1 F=2 -- A-smem halved) ---\n");
  BC128(64, 64,128,32,32,3);
  BC128(64, 64,128,32,32,2);
  BC128(32,128,128,32,32,3);
  BC128(64,128,128,32,64,3);
  BC128(32, 64,128,32,32,3);
  BC128(16,128,128,16,32,3);

  std::printf("  --- int2 single-plane sweep (TK may be 128) ---\n");
  I2(64,64,128,32,32,3);
  I2(64,64,128,32,32,4);
  I2(32,64,128,32,32,3);
  I2(64,128,128,32,64,3);
  I2(32,128,128,32,32,3);

  std::printf("  --- int4 CEILING ref (TK64 = fold's target geometry; TK128 = int4's own TK sensitivity) ---\n");
  I4(64,64,64,32,32,3);    // int4 native winner: TM64 TK64, A-smem 8KB, ~50% occ -- the ceiling
  I4(32,64,64,32,32,3);
  I4(64,128,64,32,64,3);
  I4(64,64,128,32,32,3);   // int4 @ TK128 (bigger A-smem) -> if slower, small TK helps int4 too
  I4(64,64,64,32,32,4);

  std::printf("  --- int1 single-plane sweep (TK must be 256) ---\n");
  I1(32,128,256,32,32,3);
  I1(64,64,256,32,32,3);
  I1(32,64,256,32,32,3);
  I1(32,32,256,32,32,3);
  I1(16,128,256,16,32,3);
  I1(16,64,256,16,32,3);    // TileM=16 min
  I1(16,256,256,16,32,3);   // TileM=16 + widest N

  std::printf("  ================= VERDICT =================\n");
  std::printf("  B-concat  best: %-16s %8.2f us\n", bBC.tag, bBC.us);
  std::printf("  int2      best: %-16s %8.2f us\n", bI2.tag, bI2.us);
  std::printf("  int1      best: %-16s %8.2f us\n", bI1.tag, bI1.us);
  std::printf("  int4 CEIL best: %-16s %8.2f us  <- fold target; int2/int1-fold ceiling\n", bI4.tag, bI4.us);
  double A = bI2.us + bI1.us;
  std::printf("  A-concat (best int2 + best int1, the honest sum): %8.2f us\n", A);
  std::printf("  => B-concat / A-concat = %.2fx  (%s)\n", bBC.us / A,
              bBC.us < A ? "B-concat wins" : "A-concat wins -- 2 lean GEMMs beat 1 shared-limited GEMM");
  return 0;
}
