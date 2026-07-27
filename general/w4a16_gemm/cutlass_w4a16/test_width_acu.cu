// int1 vs int2 vs int4 at ONE shared tile config, for acu. Isolates BIT WIDTH from tile shape.
//
// THE OPEN QUESTION. int1's best is 50.2% MFU and int4's production config is 55.9%, but they run DIFFERENT tiles
// (int1 (32,128,64) w32x64 s2, int4 (64,64,64) w32x32 s3), so the 5.7-point gap could be either width or shape. Every
// model built so far says it should NOT be width:
//     cvt/mma = 128/WM        identical (4) -- both at the threshold where the B convert path stops binding
//     regs_per_thread         identical (176) -- the B fragment is fp16 after conversion, so width does not enter
//     HBM traffic             FAVOURS int1 -- B is 4x smaller, and TN=128 vs 64 halves A traffic
//     B smem bank pattern     IDENTICAL, and this one is by construction: the fold sets F = 32/(TK*Bits/8), so the
//                             contiguous run is F*TK*Bits/8 = 32 B for EVERY width (int1 F=4, int2 F=2, int4 F=1).
//                             Rows are 32 B apart in all three. A bank-conflict explanation is therefore dead.
// So the gap sits outside every model, which is exactly when to stop predicting and read counters.
//
// WHY A SEPARATE FILE. test_fold_int2.cu hardcodes a different tile per width (int4 -> (64,64,64) w32x32, int1
// bitpack -> (64,128,64) w32x64) and does not expose WM/WN, so it structurally cannot answer this. Here the tile is
// ONE compile-time constant shared by all three widths.
//
// THE SHARED CONFIG: (32,128,64) w32x64 s2. It is the narrowest tile all three widths can legally run --
//     slots = WN*TK/32 = 128,  delivery = 16*8/Bits = 128 / 64 / 32  for int1 / int2 / int4
// so int1 sits EXACTLY on the delivery bound and the others under it. It is also int1's measured optimum, so int4 is
// the one being moved off its home shape -- which is the right direction: if int4 still wins here, the gap is width.
//
// OCCUPANCY IS NOT EQUAL AT EQUAL STAGES, but it can be made equal. B smem is TN*TK*Bits/8 = 1024 / 2048 / 4096 B,
// so at s2 the blocks/CU come out 23 / 19 / 15 -- which FAVOURS int1, so an int4 win at s2 is already a lower bound
// on the width cost. To remove the last asymmetry, ACU_STAGES=3 puts int1 at 16896 B and int4 at s2's 17408 B:
// BOTH land at blk=15. int1@s3 against int4@s2 is therefore same-shape, same-occupancy, width-only.
// The banner prints blk either way, so the comparison being made is in the log rather than implied.
//
//   Build: TARGET=test_width_acu ./build.sh
//   Run:   ./test_width_acu [bits] [M] [N] [K] [gs]        bits in {1,2,4}, default 1
//          ACU_ONE=1 ./test_width_acu 4                    exactly ONE kernel launch, for acu
//          ACU_ZERO=1                                      ScaleZero instead of ScaleOnly
//          ACU_STAGES=3                                    int1@s3 and int4@s2 BOTH land at blk=15 -- the
//                                                          same-shape same-occupancy control
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "fold_traits.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"

using cutlass::half_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

// THE shared config. Changing it changes it for all three widths, which is the entire point of this file.
static constexpr int TM = 32, TN = 128, TK = 64, WM = 32, WN = 64;
static_assert(fold::warp_shape_ok<TM, TN, WM, WN>, "shared config: warp tile must divide the block tile");
static_assert(fold::deliverable<1, TN, TK, WM, WN>, "shared config must be legal for int1 (the tightest bound)");
static_assert(fold::deliverable<2, TN, TK, WM, WN>, "shared config must be legal for int2");
static_assert(fold::deliverable<4, TN, TK, WM, WN>, "shared config must be legal for int4");

static int PM = 2048, PN = 4096, PK = 4096;

// Build the weight buffer for a given width at TK=64. All three fold differently and each has its own validated
// offline: int1 F=4 needs the bit-granular packer (whole-word moves cannot put two logical columns in one word),
// int2 F=2 and int4 F=1 go through the shipped preprocess (+ regroup where it folds).
template <int Bits>
static void pack_weights(std::vector<int8_t>& out) {
  constexpr int EPB = 8 / Bits, MASK = (1 << Bits) - 1;
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  const size_t bytes = (size_t)PK * PN / EPB;
  std::vector<int8_t> nk(bytes, 0);
  for (size_t i = 0; i < (size_t)PK * PN; ++i)                       // row-major (n,k) codes
    nk[i / EPB] |= int8_t((int((i * 2654435761u >> 7) & MASK)) << (Bits * (i % EPB)));
  out.assign(bytes, 0);
  if (Bits == 1 && TK == 64) {
    nfold_place_bits_int1_tk64(out.data(), nk.data(), PN, PK, TN, TK);
  } else {
    const auto qtc = Bits == 1 ? QuantTypeClass::PACKED_INT1_WEIGHT_ONLY
                   : Bits == 2 ? QuantTypeClass::PACKED_INT2_WEIGHT_ONLY
                               : QuantTypeClass::PACKED_INT4_WEIGHT_ONLY;
    preprocess_weights_for_mixed_gemm<false, 256, 0>(out.data(), nk.data(), {(size_t)PK, (size_t)PN}, qtc);
    if (F > 1) { std::vector<int8_t> f(bytes);
                 nfold_regroup_gmem(f.data(), out.data(), {(size_t)PK, (size_t)PN}, TN, TK, Bits); out.swap(f); }
  }
}

template <int Bits, int ST, class QElem>
static void run_width(int gs, bool zero) {
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  const int sk = PK / gs;

  cutlass::DeviceAllocation<half_t> A((size_t)PM * PK), S((size_t)sk * PN), Z((size_t)sk * PN), D((size_t)PM * PN);
  cutlass::DeviceAllocation<QElem>  B((size_t)PK * PN);
  { std::vector<half_t> a((size_t)PM * PK, half_t(0.01f)), s((size_t)sk * PN, half_t(0.05f)),
                        z((size_t)sk * PN, half_t(0.f));
    A.copy_from_host(a.data()); S.copy_from_host(s.data()); Z.copy_from_host(z.data()); }
  { std::vector<int8_t> w; pack_weights<Bits>(w);
    B.copy_from_host(reinterpret_cast<QElem const*>(w.data())); }

  std::vector<GS> shp_h{cute::make_shape(PM, PN, PK)};
  cutlass::DeviceAllocation<GS> shp(1); shp.copy_from_host(shp_h.data());
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(PM, PN, 1))};
  cutlass::DeviceAllocation<DStride> sD(1); sD.copy_from_host(sdh.data());
  std::vector<half_t*> pdh{D.get()}; cutlass::DeviceAllocation<half_t*> pD(1); pD.copy_from_host(pdh.data());
  std::vector<int> gmh{PM}, ofh{0};
  cutlass::DeviceAllocation<int> gm(1), off(1); gm.copy_from_host(gmh.data()); off.copy_from_host(ofh.data());
  const size_t wsb = (size_t)cutlass::ceil_div(PM, 16) * cutlass::ceil_div(PN, 64) * 64;
  cutlass::DeviceAllocation<char> ws(wsb);

  auto once = [&](bool z) {
    if (z) moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, TM, TN, TK, WM, WN, ST, QElem>(
             A.get(), B.get(), S.get(), Z.get(), pD.get(), sD.get(), gm.get(), PM, PN, PK, 1, gs,
             shp.get(), shp_h.data(), off.get(), ws.get(), wsb, nullptr);
    else   moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, TM, TN, TK, WM, WN, ST, QElem>(
             A.get(), B.get(), S.get(), nullptr, pD.get(), sD.get(), gm.get(), PM, PN, PK, 1, gs,
             shp.get(), shp_h.data(), off.get(), ws.get(), wsb, nullptr);
  };

  const int warps = (TM / WM) * (TN / WN);
  const int smem  = (TM * TK * 2 + TN * TK * Bits / 8 + TN * (TK / gs) * 2 * (zero ? 2 : 1)) * ST;
  const int blk   = std::min(262144 / smem, 64 / warps);
  constexpr int R = fold::regs_per_thread<TM, TN, TK, WM, WN, false>;

  if (getenv("ACU_ONE")) {                          // exactly ONE launch, so acu sees a clean kernel
    once(zero);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::printf("  [acu] ONE launch: int%d F=%d (%d,%d,%d) w%dx%d s%d %s gs=%d | smem=%dB blk=%d regs=%d cvt/mma=%d\n",
                Bits, F, TM, TN, TK, WM, WN, ST, zero ? "ScaleZero" : "ScaleOnly", gs,
                smem, blk, R, fold::cvt_per_mma<WM>);
    return;
  }
  for (int i = 0; i < 3; ++i) once(zero);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  hggcEvent_t e0, e1; hggcEventCreate(&e0); hggcEventCreate(&e1);
  hggcEventRecord(e0); for (int i = 0; i < 30; ++i) once(zero); hggcEventRecord(e1);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  float ms = 0; hggcEventElapsedTime(&ms, e0, e1);
  const double us = (double)ms * 1e3 / 30, tf = 2.0 * PM * PN * PK / (us * 1e-6) / 1e12;
  std::printf("  int%d F=%d (%d,%d,%d) w%dx%d s%d %s gs=%-3d | smem=%6dB blk=%-2d regs=%d cvt/mma=%d"
              " | %8.2f us  %5.1f%% MFU\n",
              Bits, F, TM, TN, TK, WM, WN, ST, zero ? "ScaleZero" : "ScaleOnly", gs,
              smem, blk, R, fold::cvt_per_mma<WM>, us, 100.0 * tf * 1e12 / 500.0e12);
}

int main(int argc, char** argv) {
  const int bits = argc > 1 ? atoi(argv[1]) : 1;
  PM = argc > 2 ? atoi(argv[2]) : 2048;
  PN = argc > 3 ? atoi(argv[3]) : 4096;
  PK = argc > 4 ? atoi(argv[4]) : 4096;
  const int gs = argc > 5 ? atoi(argv[5]) : 32;
  const bool zero = getenv("ACU_ZERO") != nullptr;
  const int st = getenv("ACU_STAGES") ? atoi(getenv("ACU_STAGES")) : 2;
  if (st != 2 && st != 3) { std::printf("  ACU_STAGES must be 2 or 3\n"); return 1; }
  std::printf("width isolation: ONE shared config (%d,%d,%d) w%dx%d s%d, bits is the only variable\n"
              "  M=%d N=%d K=%d gs=%d %s   (B smem differs by width, so blk differs -- it FAVOURS int1)\n",
              TM, TN, TK, WM, WN, st, PM, PN, PK, gs, zero ? "ScaleZero" : "ScaleOnly");
  // ACU_STAGES lets the occupancy advantage be CANCELLED rather than just noted. B smem is 1024/2048/4096 B, so at
  // equal stages int1 gets more blocks; but int1 at s3 (16896 B) and int4 at s2 (17408 B) both land at blk=15, which
  // is a same-shape same-occupancy control where bit width really is the only difference left.
  if (st == 2) switch (bits) {
    case 1: run_width<1, 2, cutlass::uint1b_t>(gs, zero); break;
    case 2: run_width<2, 2, cutlass::uint2b_t>(gs, zero); break;
    case 4: run_width<4, 2, cutlass::int4b_t >(gs, zero); break;
    default: std::printf("  bits must be 1, 2 or 4\n"); return 1;
  } else switch (bits) {
    case 1: run_width<1, 3, cutlass::uint1b_t>(gs, zero); break;
    case 2: run_width<2, 3, cutlass::uint2b_t>(gs, zero); break;
    case 4: run_width<4, 3, cutlass::int4b_t >(gs, zero); break;
    default: std::printf("  bits must be 1, 2 or 4\n"); return 1;
  }
  return 0;
}
