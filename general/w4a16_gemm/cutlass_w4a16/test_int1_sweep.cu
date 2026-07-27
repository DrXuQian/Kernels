// int1 tile sweep + a diagnostic that separates where the ScaleZero cost actually comes from.
//
// WHY A SEPARATE FILE. test_fold_int2.cu is the verified correctness harness; a sweep needs many extra kernel
// instantiations and would slow its build for everyone. Nothing here is on the correctness path.
//
// WHAT THE SWEEP IS FOR. After the box overturned my scale model, only two knobs remain for int1 and they pull
// against each other:
//     occupancy  -- smaller TM*TK per stage means more blocks
//     hiding     -- atoms per k-iteration = TK/16; at gs=16 APG = gs/16 = 1 puts a reload on EVERY atom, so more
//                   atoms per iteration means more work to overlap the reload against
// Measured so far (ScaleOnly gs=32): (32,128,128) w32x32 s3 = 42.0%, (64,128,64) w32x64 s3 = 46.4%. Occupancy alone
// does not explain a 4.4 point gap (blocks 7 vs 8), so this is exploration, not confirmation of a prediction.
//
// THE WEIGHT BUFFER DEPENDS ON (Bits, TN, TK) ONLY -- it is WN-INVARIANT, verified byte-for-byte in
// fold_derivation/l20_derived_offline.cu at WN=32 and WN=64. So configs sharing a TK share a buffer, and the sweep
// regroups by TK instead of rebuilding per config. TK=64 is the exception: it needs the bit-granular packer, so it
// gets its own buffer from nfold_place_bits_int1_tk64.
//
// THE ZERO DIAGNOSTIC (FOLD_ZDIAG=1). ScaleZero costs 51-107us and I do not know whether that is the extra smem
// COPY per reload or the extra TRANSFORM per atom -- the two scale differently (reloads = K/gs, transforms = K/16),
// and the obvious fix differs completely: interleaving sS/sZ in smem helps only the copy, fusing multiplies+plus
// into one f16x2 FMA helps only the transform. Timing ScaleOnly against ScaleZero at two group sizes separates them,
// so this measures before anything gets built. (A previous fused-FMA attempt regressed 52.3% -> 33.5% because it
// replaced a vectorized cute::transform with a scalar fp16->float->fp16 loop; the idea was never tested, only that
// implementation.)
//
//   Build: TARGET=test_int1_sweep ./build.sh ; run: ./<bin> [N] [K] [gs]
//   FOLD_ZDIAG=1 adds the ScaleOnly/ScaleZero pairs at both group sizes.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <string>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"

using cutlass::half_t;
using uint1_t = cutlass::uint1b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using QM      = moe_grouped_ppu::QuantMode;

static int PM = 2048, PN = 4096, PK = 4096;

struct Buf {
  cutlass::DeviceAllocation<half_t>  A, S, Z, D;
  cutlass::DeviceAllocation<uint1_t> B;
  cutlass::DeviceAllocation<GS>      shp;
  cutlass::DeviceAllocation<half_t*> pD;
  cutlass::DeviceAllocation<DStride> sD;
  cutlass::DeviceAllocation<int>     gm, off;
  cutlass::DeviceAllocation<char>    ws;
  std::vector<GS> shp_h;
  size_t wsb = 0;
};

static void make_buffers(Buf& b, int gs) {
  const int sk = PK / gs;
  b.A.reset((size_t)PM * PK); b.S.reset((size_t)sk * PN); b.Z.reset((size_t)sk * PN);
  b.D.reset((size_t)PM * PN); b.B.reset((size_t)PK * PN);
  { std::vector<half_t> a((size_t)PM * PK, half_t(0.01f)), s((size_t)sk * PN, half_t(0.05f)),
                        z((size_t)sk * PN, half_t(0.f));
    b.A.copy_from_host(a.data()); b.S.copy_from_host(s.data()); b.Z.copy_from_host(z.data()); }
  b.shp_h.assign(1, cute::make_shape(PM, PN, PK));
  b.shp.reset(1); b.shp.copy_from_host(b.shp_h.data());
  std::vector<DStride> sdh{cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(PM, PN, 1))};
  b.sD.reset(1); b.sD.copy_from_host(sdh.data());
  std::vector<half_t*> pdh{b.D.get()}; b.pD.reset(1); b.pD.copy_from_host(pdh.data());
  std::vector<int> gmh{PM}, ofh{0};
  b.gm.reset(1); b.gm.copy_from_host(gmh.data());
  b.off.reset(1); b.off.copy_from_host(ofh.data());
  b.wsb = (size_t)cutlass::ceil_div(PM, 16) * cutlass::ceil_div(PN, 64) * 64;
  b.ws.reset(b.wsb);
}

// One timed config. TK selects which weight buffer is valid, but the buffer is uploaded by the caller.
template <QM Q, int TM, int TN, int TK, int WM, int WN, int ST>
static void run_cfg(Buf& b, int gs, const char* note) {
  // warpOnM = TM/WM and warpOnN = TN/WN must both be >= 1, or get_tiled_mma degenerates and the collective builder
  // returns `int` -- which surfaces as "CollectiveEpilogue (aka int) cannot be used prior to ::" deep in
  // gemm_universal_adapter.h. TM=16 therefore needs WM=16, which is what test_q3_bconcat_bench.cu uses and what I
  // got wrong here by copying WM=32 across.
  static_assert(fold::warp_shape_ok<TM, TN, WM, WN>,
                "run_cfg: warp tile must divide the block tile (TM=16 needs WM=16, not WM=32)");
  static_assert(fold::deliverable<1, TN, TK, WM, WN>, "run_cfg: violates the delivery bound WN*TK*Bits >= 4096");
  auto once = [&] {
    moe_grouped_ppu::filter_and_run<Q, TM, TN, TK, WM, WN, ST, uint1_t>(
        b.A.get(), b.B.get(), b.S.get(), Q == QM::FinegrainedScaleZero ? b.Z.get() : nullptr,
        b.pD.get(), b.sD.get(), b.gm.get(), PM, PN, PK, 1, gs,
        b.shp.get(), b.shp_h.data(), b.off.get(), b.ws.get(), b.wsb, nullptr);
  };
  for (int i = 0; i < 3; ++i) once();
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  hggcEvent_t e0, e1; hggcEventCreate(&e0); hggcEventCreate(&e1);
  hggcEventRecord(e0); for (int i = 0; i < 30; ++i) once(); hggcEventRecord(e1);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  float ms = 0; hggcEventElapsedTime(&ms, e0, e1);
  const double us = (double)ms * 1e3 / 30, tf = 2.0 * PM * PN * PK / (us * 1e-6) / 1e12;
  const int warps = (TM / WM) * (TN / WN), sk = TK / gs;
  const int smem = (TM * TK * 2 + TN * TK / 8 + TN * sk * 2 * (Q == QM::FinegrainedScaleZero ? 2 : 1)) * ST;
  std::printf("  (%3d,%3d,%3d) w%dx%-3d s%d %-9s gs=%-3d | warps=%d smem=%6dB atoms=%-2d | %8.2f us  %5.1f%% MFU  %s\n",
              TM, TN, TK, WM, WN, ST, Q == QM::FinegrainedScaleZero ? "ScaleZero" : "ScaleOnly", gs,
              warps, smem, TK / 16, us, 100.0 * tf * 1e12 / 500.0e12, note);
}

// upload the weight buffer for a given TK. TK=64 needs the bit-granular packer; the rest use the shipped offline.
template <int TN, int TK>
static void upload_weights(Buf& b) {
  const size_t bytes = (size_t)PK * PN / 8;
  std::vector<int8_t> nk(bytes, 0), out(bytes, 0);
  for (size_t i = 0; i < (size_t)PK * PN; ++i)             // row-major (n,k), pseudo-random 1-bit codes
    if ((i * 2654435761u >> 7) & 1) nk[i / 8] |= int8_t(1 << (i % 8));
  if (TK == 64) {
    nfold_place_bits_int1_tk64(out.data(), nk.data(), PN, PK, TN, TK);
  } else {
    // nk is already the [N][K] one-code-per-bit packing preprocess expects, so it goes in directly.
    preprocess_weights_for_mixed_gemm<false, 256, 0>(out.data(), nk.data(),
        {(size_t)PK, (size_t)PN}, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);
    constexpr int contig = TK / 8, F = contig >= 32 ? 1 : 32 / contig;
    if (F > 1) { std::vector<int8_t> f(bytes);
                 nfold_regroup_gmem(f.data(), out.data(), {(size_t)PK, (size_t)PN}, TN, TK, 1); out.swap(f); }
  }
  b.B.copy_from_host(reinterpret_cast<uint1_t const*>(out.data()));
}

int main(int argc, char** argv) {
  PN = argc > 1 ? atoi(argv[1]) : 4096;
  PK = argc > 2 ? atoi(argv[2]) : 4096;
  const int gs = argc > 3 ? atoi(argv[3]) : 32;
  PM = argc > 4 ? atoi(argv[4]) : 2048;
  std::printf("int1 sweep  M=%d N=%d K=%d gs=%d   (buffer depends on (TN,TK) only -- WN-invariant, see l20)\n\n",
              PM, PN, PK, gs);
  Buf b; make_buffers(b, gs);

  std::printf("== TK=128 group (shipped offline). A: vary TM at fixed TK. D: WN is free -- same buffer.\n");
  upload_weights<128, 128>(b);
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 128, 16, 32, 3>(b, gs, "A: TM=16");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 3>(b, gs, "A: TM=32  <- measured 42.0% at gs=32");
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 128, 32, 32, 3>(b, gs, "A: TM=64");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 64, 3>(b, gs, "D: WN=64, same buffer");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 2>(b, gs, "C: stages=2");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 4>(b, gs, "C: stages=4");
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 128, 16, 32, 2>(b, gs, "A+C: TM=16 s2");

  std::printf("\n== TK=256 group (shipped offline, F=1). B: most atoms per iteration = best hiding.\n");
  upload_weights<128, 256>(b);
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 256, 16, 32, 2>(b, gs, "B: TK=256 TM=16 s2");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 256, 32, 32, 2>(b, gs, "B: TK=256 TM=32 s2");
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 256, 16, 32, 3>(b, gs, "B: TK=256 TM=16 s3");

  std::printf("\n== TK=64 group (bit-granular packer, WN>=64 required by the delivery bound).\n");
  upload_weights<128, 64>(b);
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 64, 32, 64, 3>(b, gs, "measured 46.4% at gs=32");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 64, 32, 64, 3>(b, gs, "B: TM=32");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 64, 32, 64, 2>(b, gs, "B+C: TM=32 s2");

  if (getenv("FOLD_ZDIAG")) {
    std::printf("\n== ZERO DIAGNOSTIC. reloads = K/gs, transforms = K/16. If the zero cost tracks gs it is the COPY\n");
    std::printf("   (interleave sS/sZ); if it is flat in gs it is the TRANSFORM (fuse multiplies+plus into one FMA).\n");
    for (int g : {32, 16}) {
      Buf bz; make_buffers(bz, g);
      upload_weights<128, 128>(bz);
      run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 3>(bz, g, "ScaleOnly TK=128");
      run_cfg<QM::FinegrainedScaleZero, 32, 128, 128, 32, 32, 3>(bz, g, "ScaleZero TK=128");
      upload_weights<128, 256>(bz);
      run_cfg<QM::FinegrainedScaleOnly, 32, 128, 256, 32, 32, 2>(bz, g, "ScaleOnly TK=256 (2x transforms/iter)");
      run_cfg<QM::FinegrainedScaleZero, 32, 128, 256, 32, 32, 2>(bz, g, "ScaleZero TK=256 (2x transforms/iter)");
    }
    std::printf("   TK=256 doubles transforms per iteration while halving iterations -- transforms total is\n");
    std::printf("   invariant, but reloads per iteration double. Comparing the zero delta at TK=128 vs TK=256\n");
    std::printf("   separates a per-reload cost from a per-transform one.\n");
  }
  return 0;
}
