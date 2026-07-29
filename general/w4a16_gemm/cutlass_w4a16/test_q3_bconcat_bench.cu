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
#include "xplane_offline.hpp"
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
  // (f) THE FOLD BRANCH IS GONE. Every call passes FoldTK = 0 -- folded planes are built by xplane::place_derived /
  // place_int1 inside the row macros, per configuration, because the map depends on the tile. What is left here is the
  // unfolded path, which the derived preprocess_weights_for_mixed_gemm covers on its own.
  return out;
}

// globals filled in main, so the sweep macros stay short
static cutlass::DeviceAllocation<half_t>  *dA, *dSc, *dZr, *dD, *dDhi;
static cutlass::DeviceAllocation<uint2_t> *dBlo;
static cutlass::DeviceAllocation<int4_t>  *dB4;
static cutlass::DeviceAllocation<uint1_t> *dBhi;
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

// PER-PLANE FOLD, both planes from the DERIVED map (xplane), per configuration. Two reasons this is not the old
// pack_plane pair:
//   * the high plane must use the CROSS-PLANE map. The bench used to pack it with the single-plane rule, which is the
//     buffer the hi_vreg0 finding says is wrong; it went unnoticed because a bench only reads the clock. Timing is
//     unaffected, but a row that looks validated and is not is worse than no row.
//   * plane 1 folds too once Block_K drops to 64 (int2 F=2, int1 F=4), and the whole-word packer cannot express
//     cols_per_word > 1 at WN=64 -- that was the rung-5 defect.
// Both buffers are rebuilt per row because the map depends on the tile. Host-side, outside the timed region.
#define BCF(TM,TN,TK,WM,WN,S,F1,F2) do { \
  std::vector<int8_t> blo_((size_t)K*N/4), bhi_((size_t)K*N/8); \
  xplane::place_derived<2,TM,TN,TK,WM,WN,F1>(blo_.data(), low, N, K); \
  xplane::place_int1<TM,TN,TK,WM,WN,F2,F1>(bhi_.data(), high, N, K); \
  cutlass::DeviceAllocation<uint2_t> b1_((size_t)K*N); b1_.copy_from_host(reinterpret_cast<uint2_t const*>(blo_.data())); \
  cutlass::DeviceAllocation<uint1_t> b2_((size_t)K*N); b2_.copy_from_host(reinterpret_cast<uint1_t const*>(bhi_.data())); \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TM,TN,TK,WM,WN,S,uint2_t,uint1_t>( \
      dA->get(), b1_.get(), dSc->get(), dZr->get(), pd->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr, b2_.get()); }, 30); \
  report("BC " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S " [F1=" #F1 " F2=" #F2 "]", u); \
  upd(bBC, #TM "x" #TN ":" #TK " s" #S " F" #F1 #F2, u); } while (0)

// int2 alone WITH its own fold, so Block_K=64 is reachable for the single-plane reference too. Without this the
// "int2 best" line is a TK=128 number and the concat overhead is measured against the wrong ceiling.
// Q6 = int4 + int2 and Q5 = int4 + int1 on the SAME 2-plane mainloop. int4's contiguous run is already 32 B at TK=64,
// so the LOW plane never folds (F1=1) and only the high plane does -- structurally simpler than Q3, where both fold at
// Block_K=64. Delivery bounds: Q6's int2 high needs WN >= 2048/TK (32 at TK=64, so w64x32 is legal here), Q5's int1 high
// needs WN >= 4096/TK (64 at TK=64). The low plane is written by place_derived DIRECTLY with q & 15 -- the shim's +8 is
// for reproducing the legacy pipeline, and here int4's -8 is absorbed by the zero point.
#define Q65(NAME,BEST,HIB,HIELEM,TM,TN,TK,WM,WN,S,F2) do { \
  std::vector<int8_t> blo_((size_t)K*N/2), bhi_((size_t)K*N*(HIB)/8); \
  xplane::place_derived<4,TM,TN,TK,WM,WN,1>(blo_.data(), q65lo, N, K); \
  xplane::place_hi<4,HIB,TM,TN,TK,WM,WN,F2,1>(bhi_.data(), q65hi##HIB, N, K); \
  cutlass::DeviceAllocation<int4_t> b1_((size_t)K*N); b1_.copy_from_host(reinterpret_cast<int4_t const*>(blo_.data())); \
  cutlass::DeviceAllocation<HIELEM> b2_((size_t)K*N); b2_.copy_from_host(reinterpret_cast<HIELEM const*>(bhi_.data())); \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TM,TN,TK,WM,WN,S,int4_t,HIELEM>( \
      dA->get(), b1_.get(), dSc->get(), dZr->get(), pd->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr, b2_.get()); }, 30); \
  report(NAME " " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S " [F2=" #F2 "]", u); \
  upd(BEST, #TM "x" #TN ":" #TK " s" #S " F2=" #F2, u); } while (0)

#define I2F(TM,TN,TK,WM,WN,S,F) do { \
  std::vector<int8_t> b_((size_t)K*N/4); \
  xplane::place_derived<2,TM,TN,TK,WM,WN,F>(b_.data(), low, N, K); \
  cutlass::DeviceAllocation<uint2_t> bb_((size_t)K*N); bb_.copy_from_host(reinterpret_cast<uint2_t const*>(b_.data())); \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero,TM,TN,TK,WM,WN,S,uint2_t>( \
      dA->get(), bb_.get(), dSc->get(), dZr->get(), pd->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr); }, 30); \
  report("i2 " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S " [F=" #F "]", u); \
  upd(bI2, #TM "x" #TN ":" #TK " s" #S " F" #F, u); } while (0)

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

// int1 alone WITH its own fold. The "TK must be 256" note on the i1 sweep below is STALE for the same reason the
// B-concat one was: per-plane fold reaches Block_K 128 (F=2) and 64 (F=4). It matters a lot here -- the recorded int1
// optimum is (64,128,64) w64x64 s2 at 215.62 us / 63.7% MFU, i.e. TK=64, so a TK=256-only sweep reports 26.7% and
// understates the single-plane ceiling by 2.4x. Two caveats when comparing against 63.7%: that number was measured at
// gs=32 (this bench defaults to gs=16, which forces FINE) and WITH the chunked B conversion, which the 2-plane path
// does not have yet (task #12).
#define I1F(TM,TN,TK,WM,WN,S,F) do { \
  std::vector<int8_t> b_((size_t)K*N/8); \
  xplane::place_derived<1,TM,TN,TK,WM,WN,F>(b_.data(), high, N, K); \
  cutlass::DeviceAllocation<uint1_t> bb_((size_t)K*N); bb_.copy_from_host(reinterpret_cast<uint1_t const*>(b_.data())); \
  double u = time_it([&]{ moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly,TM,TN,TK,WM,WN,S,uint1_t>( \
      dA->get(), bb_.get(), dSc->get(), nullptr, pd2->get(), sd->get(), gm->get(), \
      M,N,K,1,gs, shpd->get(), shpv->data(), offdev->get(), ws->get(), wsb, nullptr); }, 30); \
  report("i1 " #TM "x" #TN ":" #TK " w" #WM "x" #WN " s" #S " (ScaleOnly) [F=" #F "]", u); \
  upd(bI1, #TM "x" #TN ":" #TK " s" #S " F" #F, u); } while (0)

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
  // Q6/Q5 planes: low = q & 15 for BOTH (int4), high = q >> 4 with 2 bits for Q6 and 1 for Q5. Full code range, so the
  // top plane is actually exercised rather than sitting at zero.
  std::vector<uint8_t> q65lo((size_t)K*N), q65hi2((size_t)K*N), q65hi1((size_t)K*N);
  for (size_t i = 0; i < q65lo.size(); ++i) {
    const int q6 = int((i * 2654435761u >> 5) % 64u), q5 = int((i * 2654435761u >> 5) % 32u);
    q65lo[i]  = uint8_t(q6 & 15);          // the int4 low plane is shared by both rows below
    q65hi2[i] = uint8_t(q6 >> 4);
    q65hi1[i] = uint8_t(q5 >> 4);
  }
  auto B4 = pack_plane<2, QuantTypeClass::PACKED_INT4_WEIGHT_ONLY>(q4);
  cutlass::DeviceAllocation<int4_t> B4_((size_t)K*N); B4_.copy_from_host(reinterpret_cast<int4_t const*>(B4.data()));
  cutlass::DeviceAllocation<uint1_t> Bhi_((size_t)K*N);
  { std::vector<half_t> a((size_t)M*K, half_t(0.01f)), s((size_t)scale_k*N, half_t(0.05f)), z((size_t)scale_k*N, half_t(-0.2f));
    A_.copy_from_host(a.data()); S_.copy_from_host(s.data()); Z_.copy_from_host(z.data()); }
  Blo_.copy_from_host(reinterpret_cast<uint2_t const*>(Blo.data()));
  Bhi_.copy_from_host(reinterpret_cast<uint1_t const*>(Bhi.data()));

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

  dA=&A_; dSc=&S_; dZr=&Z_; dD=&D_; dDhi=&Dh_; dBlo=&Blo_; dBhi=&Bhi_; dB4=&B4_; shpd=&shpd_;
  pd=&pd_; pd2=&pd2_; sd=&sd_; gm=&gm_; offdev=&off_; ws=&ws_; shpv=&shp;

  Best bBC{"",1e18}, bI2{"",1e18}, bI1{"",1e18}, bI4{"",1e18}, bQ6{"",1e18}, bQ5{"",1e18};

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
  // Block_K=128: int2 needs no fold (32 B run), int1 folds by 2.
  BCF(64, 64,128,32,32,3,1,2);
  BCF(64, 64,128,32,32,2,1,2);
  BCF(32,128,128,32,32,3,1,2);
  BCF(64,128,128,32,64,3,1,2);
  BCF(32, 64,128,32,32,3,1,2);
  BCF(16,128,128,16,32,3,1,2);
  // Same opening for the two-plane path: at TK=128 the int1 plane's bound is WN >= 32, so w64x32 is legal. Its TK=128
  // rows are all w32x32 / w32x64, and w64x32 is worth +7 points on int2 -- whether that survives two planes is exactly
  // the question. (At TK=64 it is illegal, so the current 252.72 us winner cannot be improved this way.)
  BCF(64,128,128,64,32,3,1,2);
  BCF(64,128,128,64,32,2,1,2);
  BCF(128,128,128,64,32,3,1,2);
  // Block_K=64: BOTH planes fold (int2 F=2, int1 F=4). int1's over-delivery bound is delivery <= WN*TK/32, i.e.
  // 128 <= WN*64/32, so WN=64 is FORCED here -- w32x* cannot run at this Block_K. A-smem is TM*TK*2 = TM*128 B per
  // stage, a quarter of TK=256's, which is the whole point: TileM and stages are the levers that were unaffordable.
  BCF(64,128, 64,64,64,3,2,4);
  BCF(64,128, 64,64,64,2,2,4);
  BCF(64,128, 64,32,64,3,2,4);
  BCF(32,128, 64,32,64,3,2,4);
  BCF(128,128,64,64,64,3,2,4);
  BCF(64,256, 64,64,64,3,2,4);
  BCF(128,256,64,64,64,2,2,4);
  // STAGES IS THE DOMINANT LEVER HERE, and the first sweep only sampled s2 once: (64,128,64) w64x64 went 551.04 us
  // (24.9%) at s3 and 377.33 us (36.4%) at s2 -- 1.46x from stages alone, while every other Block_K=64 row ran s3 and
  // landed in a flat ~425 us band. So give each geometry its s2, and probe s4 in case the curve turns again.
  BCF(64,128, 64,32,64,2,2,4);
  BCF(32,128, 64,32,64,2,2,4);
  BCF(128,128,64,64,64,2,2,4);
  BCF(64,256, 64,64,64,2,2,4);
  BCF(64,128, 64,64,64,4,2,4);
  BCF(32,128, 64,32,64,4,2,4);

  std::printf("  --- int2 single-plane sweep (TK 128, then FOLDED 64) ---\n");
  I2(64,64,128,32,32,3);
  I2(64,64,128,32,32,4);
  I2(32,64,128,32,32,3);
  I2(64,128,128,32,64,3);
  I2(32,128,128,32,32,3);
  I2F(64, 64, 64,32,32,3,2);
  I2F(64, 64, 64,32,32,2,2);
  I2F(32,128, 64,32,32,3,2);
  I2F(64,128, 64,64,64,2,2);
  // w64x32 -- THE WARP SHAPE THE SWEEP NEVER HAD, and the only one where int2 gets chunking AND cvt/mma = 2 at once.
  //
  // MMA_N = TN / PermN = TN / ((TN/warpN)*16) = warpN / 16, so it depends ONLY on warpN -- TN and TM are free. Chunking
  // needs 8*MMA_N*MMA_K == kOut = 64, i.e. MMA_N == 2 at TK=64, i.e. warpN == 32. That is a constraint on WN alone, NOT
  // on tile size, so w64x32 buys chunking, cvt/mma = 128/WM = 2, AND a 128x128 tile simultaneously.
  //
  // Why that matters here: int2's best row (64,64,64) w32x32 is SLOWER than the two-plane B-concat's (128,128,64)
  // w64x64, which is not a paradox -- B-concat merges the planes in the converter, so its mma count is identical and
  // only its HBM traffic and ALU are larger. At M=2048 N=K=4096, A is re-read N/TN times and B M/TM times:
  //     int2 (64,64,64)     A 64x16.78MB = 1074MB + B 32x4.19MB = 134MB  ~= 1208 MB
  //     int3 (128,128,64)   A 32x16.78MB =  537MB + B 16x6.29MB = 101MB  ~=  638 MB
  // so the two-plane kernel moves 1.89x FEWER bytes despite carrying an extra bit plane. int2 was never tile-limited --
  // it was warp-shape-limited: every swept row is w32x32 (chunking on but cvt/mma = 4, where freeing registers under a
  // throughput ceiling measured ~0) or w64x64 (cvt/mma = 2 but MMA_N = 4, so the predicate turns chunking OFF, which is
  // why that row is both 418 us and unchunked).
  I2F(64, 64, 64,64,32,3,2);
  I2F(64, 64, 64,64,32,2,2);
  I2F(64,128, 64,64,32,3,2);
  I2F(64,128, 64,64,32,2,2);
  I2F(128,64, 64,64,32,3,2);
  I2F(128,128,64,64,32,3,2);

  std::printf("  --- int4 CEILING ref (TK64 = fold's target geometry; TK128 = int4's own TK sensitivity) ---\n");
  I4(64,64,64,32,32,3);    // int4 native winner: TM64 TK64, A-smem 8KB, ~50% occ -- the ceiling
  I4(32,64,64,32,32,3);
  I4(64,128,64,32,64,3);
  I4(64,64,128,32,32,3);   // int4 @ TK128 (bigger A-smem) -> if slower, small TK helps int4 too
  I4(64,64,64,32,32,4);
  // w64x32 FOR int4 TOO -- and this is no longer a nicety. int2 at w64x32 measured 232.96 us / 59.0%, i.e. FASTER than
  // int4's 243.54, so int4 is no longer a ceiling; it is just another format whose sweep is missing the same warp
  // shape (every i4 row above is w32x32 or w32x64). Every "x% of the ceiling" statement rests on this reference, so it
  // has to be tuned on the same grid before any of them mean anything. int4 does not chunk (the predicate covers 1 and
  // 2 bits only), so this is purely the warp shape: cvt/mma = 128/WM = 2 instead of 4.
  I4(64,64,64,64,32,3);
  I4(64,64,64,64,32,2);
  I4(64,128,64,64,32,3);
  I4(128,64,64,64,32,3);
  I4(64,64,128,64,32,3);

  std::printf("  --- int1 single-plane sweep (TK 256 unfolded, then FOLDED 128/64) ---\n");
  I1(32,128,256,32,32,3);
  I1(64,64,256,32,32,3);
  I1(32,64,256,32,32,3);
  I1(32,32,256,32,32,3);
  I1(16,128,256,16,32,3);
  I1(16,64,256,16,32,3);    // TileM=16 min
  I1(16,256,256,16,32,3);   // TileM=16 + widest N
  // FOLDED int1 -- the geometries the recorded 63.7% actually used. int1's over-delivery bound is 128 <= WN*TK/32, so
  // TK=64 forces WN=64 and TK=128 allows WN=32.
  I1F(64,128, 64,64,64,2,4);          // the recorded optimum's tile and stage count
  I1F(64,128, 64,64,64,3,4);
  I1F(32,128, 64,32,64,2,4);
  I1F(128,128,64,64,64,2,4);
  I1F(64,128,128,32,32,3,2);
  I1F(32,128,128,32,32,3,2);
  I1F(32,128,128,32,32,2,2);
  // int1 at TK=64 needs WN >= 64 (delivery 128 <= slots WN*TK/32), so w64x32 is illegal there -- but at TK=128 the bound
  // is WN >= 32, so w64x32 IS legal and was never swept. Chunking stays on: MMA_N = warpN/16 = 2, MMA_K = 8, and
  // 8*2*8 == kOut = 128 for int1.
  I1F(64,128,128,64,32,3,2);
  I1F(64,128,128,64,32,2,2);
  I1F(128,128,128,64,32,3,2);

  std::printf("  --- Q6 = int4 + int2 (one GEMM, 6-bit in memory) ---\n");
  Q65("q6", bQ6, 2, uint2_t, 64,128,128,32,32,3,1);
  Q65("q6", bQ6, 2, uint2_t, 64,128,128,64,32,3,1);
  Q65("q6", bQ6, 2, uint2_t, 64,128, 64,64,32,3,2);
  Q65("q6", bQ6, 2, uint2_t, 64,128, 64,64,32,2,2);
  Q65("q6", bQ6, 2, uint2_t, 64,128, 64,64,64,2,2);
  Q65("q6", bQ6, 2, uint2_t,128,128, 64,64,32,3,2);
  std::printf("  --- Q5 = int4 + int1 (one GEMM, 5-bit in memory) ---\n");
  Q65("q5", bQ5, 1, uint1_t, 64,128,256,32,32,3,1);
  Q65("q5", bQ5, 1, uint1_t, 64,128,128,32,32,3,2);
  Q65("q5", bQ5, 1, uint1_t, 64,128,128,64,32,3,2);
  Q65("q5", bQ5, 1, uint1_t, 64,128, 64,64,64,2,4);
  Q65("q5", bQ5, 1, uint1_t,128,128, 64,64,64,2,4);
  // NEIGHBOURS OF THE TWO WINNERS. Both formats peaked at (64,128,64) w64x64 s2 -- the same geometry as Q3 -- and on Q3
  // the rows within 2% of that were (128,128,64) w64x64 s3 and (64,256,64) w64x64 s3, neither of which was sampled here.
  // Q6 can additionally use w32x32 at TK=64: its int2 high plane needs only WN >= 2048/TK = 32 and the int4 low plane
  // WN >= 1024/TK = 16, so the whole w*x32 family is legal for Q6 and was never tried. Q5's int1 high pins it to w*x64.
  std::printf("  --- Q6 / Q5 neighbours of the winners ---\n");
  Q65("q6", bQ6, 2, uint2_t, 64,128, 64,64,64,3,2);
  Q65("q6", bQ6, 2, uint2_t,128,128, 64,64,64,3,2);
  Q65("q6", bQ6, 2, uint2_t,128,128, 64,64,64,2,2);
  Q65("q6", bQ6, 2, uint2_t, 64,256, 64,64,64,3,2);
  Q65("q6", bQ6, 2, uint2_t, 64,256, 64,64,64,2,2);
  Q65("q6", bQ6, 2, uint2_t, 64,128, 64,32,32,3,2);
  Q65("q6", bQ6, 2, uint2_t, 64,128, 64,32,64,2,2);
  Q65("q5", bQ5, 1, uint1_t, 64,128, 64,64,64,3,4);
  Q65("q5", bQ5, 1, uint1_t,128,128, 64,64,64,3,4);
  Q65("q5", bQ5, 1, uint1_t, 64,256, 64,64,64,2,4);
  Q65("q5", bQ5, 1, uint1_t, 64,256, 64,64,64,3,4);
  Q65("q5", bQ5, 1, uint1_t, 32,128, 64,32,64,2,4);

  std::printf("  ================= VERDICT =================\n");
  std::printf("  B-concat  best: %-16s %8.2f us\n", bBC.tag, bBC.us);
  std::printf("  int2      best: %-16s %8.2f us\n", bI2.tag, bI2.us);
  std::printf("  int1      best: %-16s %8.2f us\n", bI1.tag, bI1.us);
  std::printf("  int4 CEIL best: %-16s %8.2f us  <- fold target; int2/int1-fold ceiling\n", bI4.tag, bI4.us);
  std::printf("  Q6 (int4+int2)  best: %-16s %8.2f us   vs int4 alone %.2fx\n", bQ6.tag, bQ6.us, bQ6.us / bI4.us);
  std::printf("  Q5 (int4+int1)  best: %-16s %8.2f us   vs int4 alone %.2fx\n", bQ5.tag, bQ5.us, bQ5.us / bI4.us);
  double A = bI2.us + bI1.us;
  std::printf("  A-concat (best int2 + best int1, the honest sum): %8.2f us\n", A);
  std::printf("  => B-concat / A-concat = %.2fx  (%s)\n", bBC.us / A,
              bBC.us < A ? "B-concat wins" : "A-concat wins -- 2 lean GEMMs beat 1 shared-limited GEMM");
  return 0;
}
