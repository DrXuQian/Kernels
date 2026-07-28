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
//   Build: TARGET=test_width_acu ./build.sh                   PLAIN build -- int1 measures 48.6% at the best config
//          PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_width_acu ./build.sh    CHUNKED -- 63.7%. The define is NOT the default;
//          the printed line ends in CHUNK or plain, so check it before believing an acu capture.
//   Run:   ./test_width_acu [bits] [M] [N] [K] [gs]        bits in {1,2,4}, default 1
//          ACU_ONE=1 ./test_width_acu 4                    exactly ONE kernel launch, for acu
//          ACU_ZERO=1                                      ScaleZero instead of ScaleOnly
//          ACU_STAGES=3                                    int1@s3 and int4@s2 BOTH land at blk=15 -- the
//                                                          same-shape same-occupancy control
//          ACU_LADDER=1 ./test_width_acu 4                 walk int4's home tile -> int1's tile, ONE variable per
//                                                          rung, to localise its unexplained 10-point drop
//          ACU_LADDER=1 ACU_RUNG=n ACU_ONE=1               ONE rung, ONE launch -- the acu form
//          ACU_BEST=1 ACU_ONE=1                            the new best config (64,128,64) w64x64 s2, 63.7% chunked
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include "fold_traits.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "xplane_offline.hpp"
#include "moe_grouped_ppu.cuh"

// If this fails, the actlize SUBMODULE on the box is stale: the Kernels gitlink moved but `git submodule update` did
// not run, so the build would silently produce a binary identical to the previous one. That ambiguity is what made an
// "every acu counter is identical" A/B uninterpretable -- so it is a compile error, not a runtime surprise.
#if !defined(PPU_SCALE_FRAGMENT_API) || PPU_SCALE_FRAGMENT_API < 3
#error "stale actlize submodule: run `git submodule update --init third_party/actlize` and rebuild"
#endif

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

// Build the weight buffer for a given width at TK=64. All three widths fold differently -- int1 F=4, int2 F=2, int4 F=1
// -- and each used to need its own packer. One derived call now covers all of them. WNp is a parameter because the map
// depends on the warp N extent and the rungs sweep w32x32 / w32x64 / w64x64.
template <int Bits, int TNp, int WNp>
static void pack_weights(std::vector<int8_t>& out) {
  constexpr int MASK = (1 << Bits) - 1, EPB = 8 / Bits;
  constexpr int contig = TK * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  // ONE derived call for every width and TK: place_derived covers the fold walk and the interleave-256 walk, so the
  // bit-granular / preprocess+nfold split is gone. int4 needs the +8 the old preprocess applied, since a position map
  // carries no value transform. Byte-identical to the old path at every (Bits, TN, TK) this harness uses (l64).
  std::vector<uint8_t> q((size_t)PK * PN);                          // [K][N], one code per byte
  for (size_t i = 0; i < q.size(); ++i) {
    const int v = int((i * 2654435761u >> 7) & MASK);
    q[i] = uint8_t(Bits == 4 ? ((v + 8) & MASK) : v);
  }
  out.assign((size_t)PK * PN / EPB, 0);
  xplane::place_derived<Bits, 64, TNp, TK, 32, WNp, F>(out.data(), q, PN, PK);
}

template <int Bits, int TMr, int TNr, int TKr, int WMr, int WNr, int ST, class QElem>
static void run_rung(int gs, bool zero, const char* note) {
  constexpr int contig = TKr * Bits / 8, F = contig >= 32 ? 1 : 32 / contig;
  const int sk = PK / gs;

  cutlass::DeviceAllocation<half_t> A((size_t)PM * PK), S((size_t)sk * PN), Z((size_t)sk * PN), D((size_t)PM * PN);
  cutlass::DeviceAllocation<QElem>  B((size_t)PK * PN);
  { std::vector<half_t> a((size_t)PM * PK, half_t(0.01f)), s((size_t)sk * PN, half_t(0.05f)),
                        z((size_t)sk * PN, half_t(0.f));
    A.copy_from_host(a.data()); S.copy_from_host(s.data()); Z.copy_from_host(z.data()); }
  { std::vector<int8_t> w; pack_weights<Bits, TNr, WNr>(w);
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
    if (z) moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, TMr, TNr, TKr, WMr, WNr, ST, QElem>(
             A.get(), B.get(), S.get(), Z.get(), pD.get(), sD.get(), gm.get(), PM, PN, PK, 1, gs,
             shp.get(), shp_h.data(), off.get(), ws.get(), wsb, nullptr);
    else   moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, TMr, TNr, TKr, WMr, WNr, ST, QElem>(
             A.get(), B.get(), S.get(), nullptr, pD.get(), sD.get(), gm.get(), PM, PN, PK, 1, gs,
             shp.get(), shp_h.data(), off.get(), ws.get(), wsb, nullptr);
  };

  const int warps = (TMr / WMr) * (TNr / WNr);
  const int smem  = (TMr * TKr * 2 + TNr * TKr * Bits / 8 + TNr * (TKr / gs) * 2 * (zero ? 2 : 1)) * ST;
  const int blk   = std::min(262144 / smem, 64 / warps);
  // WHICH KERNEL WAS MEASURED. This binary used to be completely chunk-blind: no witness that PPU_B_CHUNK reached the
  // compile line, and regs= always the UNCHUNKED model. int1 at (64,128,64) w64x64 s2 is 63.7% chunked and 48.6%
  // plain, so a ~50% acu capture of "the best config" is the PLAIN BUILD, not a bad measurement -- and nothing in the
  // output said so. An acu capture that cannot name the kernel it profiled is not evidence. (kBChunk in the collective
  // is int1-only, hence chunk-N/A rather than CHUNK for widths 2 and 4.)
#if defined(PPU_B_CHUNK) && (PPU_B_CHUNK != 0)
  constexpr bool kCh = true;
  constexpr int R = fold::regs_per_thread_chunked<TMr, TNr, TKr, WMr, WNr, false>;
#else
  constexpr bool kCh = false;
  constexpr int R = fold::regs_per_thread<TMr, TNr, TKr, WMr, WNr, false>;
#endif
  const char* chtag = kCh ? (Bits == 1 ? "CHUNK" : "chunk-N/A") : "plain";

  if (getenv("ACU_ONE")) {                          // exactly ONE launch, so acu sees a clean kernel
    once(zero);
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    std::printf("  [acu] ONE launch: int%d F=%d (%d,%d,%d) w%dx%d s%d %s gs=%d | smem=%dB blk=%d regs=%d bill=%d cvt/mma=%d %s | %s\n",
                Bits, F, TMr, TNr, TKr, WMr, WNr, ST, zero ? "ScaleZero" : "ScaleOnly", gs,
                smem, blk, R, fold::regs_billed<R>, fold::cvt_per_mma<WMr>, chtag, note);
    return;
  }
  for (int i = 0; i < 3; ++i) once(zero);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  hggcEvent_t e0, e1; hggcEventCreate(&e0); hggcEventCreate(&e1);
  hggcEventRecord(e0); for (int i = 0; i < 30; ++i) once(zero); hggcEventRecord(e1);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  float ms = 0; hggcEventElapsedTime(&ms, e0, e1);
  const double us = (double)ms * 1e3 / 30, tf = 2.0 * PM * PN * PK / (us * 1e-6) / 1e12;
  std::printf("  int%d F=%d (%2d,%3d,%2d) w%dx%-2d s%d %s gs=%-3d | warps=%d smem=%6dB blk=%-2d regs=%d bill=%d cvt/mma=%d %s"
              " | %8.2f us  %5.1f%% MFU  %s\n",
              Bits, F, TMr, TNr, TKr, WMr, WNr, ST, zero ? "ScaleZero" : "ScaleOnly", gs,
              warps, smem, blk, R, fold::regs_billed<R>, fold::cvt_per_mma<WMr>, chtag,
              us, 100.0 * tf * 1e12 / 500.0e12, note);
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
  // ACU_LADDER=1: walk from int4's HOME tile to int1's tile changing ONE variable per rung. int4 measures 55.9% at
  // home and 45.9% at int1's tile, a 10-point gap that blk (8 vs 15, backwards), warps/CU (32 vs 30, equal) and HBM
  // traffic (1342 vs 1074 MB, backwards) all fail to explain. Four things differ at once -- TM, TN, WN and stages --
  // so a ladder localises the drop to one of them, and may settle it without acu at all.
  //
  //   rung 1  (64, 64,64) w32x32 s3   home                        55.9% measured
  //   rung 2  (64, 64,64) w32x32 s2   stages 3 -> 2
  //   rung 3  (64,128,64) w32x32 s2   TN     64 -> 128
  //   rung 4  (64,128,64) w32x64 s2   WN     32 -> 64
  //   rung 5  (32,128,64) w32x64 s2   TM     64 -> 32               int1's tile, 45.9% measured
  //
  // Rungs 1-3 have WN=32 so slots=64: legal for int4 (deliv 32) and int2 (64) but NOT int1 (128), hence the
  // if constexpr gate rather than a comment saying "int1 skips these".
  // ACU_BEST=1: the new best config, (64,128,64) w64x64 s2, which measures 63.7% with PPU_B_CHUNK=1 against 48.6%
  // without. It is not the shared config above (that one is w32x64, chosen so all three widths can run it), so it
  // needs its own entry point. Legal for int1: slots = WN*TK/32 = 128 = delivery, and TM/WM=1, TN/WN=2 -> 2 warps.
  if (getenv("ACU_BEST")) {
    switch (bits) {
      case 1: run_rung<1, 64, 128, 64, 64, 64, 2, cutlass::uint1b_t>(gs, zero, "NEW BEST w64x64 s2 (63.7% chunked)"); break;
      case 2: run_rung<2, 64, 128, 64, 64, 64, 2, cutlass::uint2b_t>(gs, zero, "w64x64 s2"); break;
      case 4: run_rung<4, 64, 128, 64, 64, 64, 2, cutlass::int4b_t >(gs, zero, "w64x64 s2"); break;
      default: std::printf("  bits must be 1, 2 or 4\n"); return 1;
    }
    return 0;
  }

  if (getenv("ACU_LADDER")) {
    // ACU_RUNG=n runs ONLY rung n, so ACU_ONE=1 ACU_RUNG=n is a single clean launch for acu. Without it,
    // ACU_ONE + the full ladder would emit five launches and acu would have nothing to attribute.
    const int only = getenv("ACU_RUNG") ? atoi(getenv("ACU_RUNG")) : 0;
    std::printf("\n== LADDER: one variable per rung, int4 home -> int1's tile (rungs 1-3 need WN=32, illegal for int1)%s\n",
                only ? "  [single rung]" : "");
    auto ladder = [&](auto tag) {
      constexpr int Bt = decltype(tag)::value;
      using QE = std::conditional_t<Bt == 1, cutlass::uint1b_t,
                 std::conditional_t<Bt == 2, cutlass::uint2b_t, cutlass::int4b_t>>;
      if constexpr (fold::deliverable<Bt, 64, 64, 32, 32>) {
        if (!only || only == 1) run_rung<Bt, 64,  64, 64, 32, 32, 3, QE>(gs, zero, "rung 1: int4 HOME (55.9% measured)");
        if (!only || only == 2) run_rung<Bt, 64,  64, 64, 32, 32, 2, QE>(gs, zero, "rung 2: stages 3 -> 2");
        if (!only || only == 3) run_rung<Bt, 64, 128, 64, 32, 32, 2, QE>(gs, zero, "rung 3: TN 64 -> 128");
      } else if (only <= 3) {
        std::printf("  int%d: rungs 1-3 skipped -- slots=64 < delivery=%d (over-delivery)\n", Bt, 16 * 8 / Bt);
      }
      if (!only || only == 4) run_rung<Bt, 64, 128, 64, 32, 64, 2, QE>(gs, zero, "rung 4: WN 32 -> 64");
      if (!only || only == 5) run_rung<Bt, 32, 128, 64, 32, 64, 2, QE>(gs, zero, "rung 5: TM 64 -> 32  = int1's tile (45.9% measured)");
    };
    switch (bits) {
      case 1: ladder(std::integral_constant<int,1>{}); break;
      case 2: ladder(std::integral_constant<int,2>{}); break;
      case 4: ladder(std::integral_constant<int,4>{}); break;
      default: std::printf("  bits must be 1, 2 or 4\n"); return 1;
    }
    return 0;
  }

  // Default: the single shared config, which is what the width comparison uses. ACU_STAGES cancels the occupancy
  // asymmetry -- int1@s3 (16896 B) and int4@s2 (17408 B) both land at blk=15.
  if (st == 2) switch (bits) {
    case 1: run_rung<1, TM, TN, TK, WM, WN, 2, cutlass::uint1b_t>(gs, zero, "shared config"); break;
    case 2: run_rung<2, TM, TN, TK, WM, WN, 2, cutlass::uint2b_t>(gs, zero, "shared config"); break;
    case 4: run_rung<4, TM, TN, TK, WM, WN, 2, cutlass::int4b_t >(gs, zero, "shared config"); break;
    default: std::printf("  bits must be 1, 2 or 4\n"); return 1;
  } else switch (bits) {
    case 1: run_rung<1, TM, TN, TK, WM, WN, 3, cutlass::uint1b_t>(gs, zero, "shared config"); break;
    case 2: run_rung<2, TM, TN, TK, WM, WN, 3, cutlass::uint2b_t>(gs, zero, "shared config"); break;
    case 4: run_rung<4, TM, TN, TK, WM, WN, 3, cutlass::int4b_t >(gs, zero, "shared config"); break;
    default: std::printf("  bits must be 1, 2 or 4\n"); return 1;
  }
  return 0;
}
