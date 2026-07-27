// int1 tile sweep + a diagnostic that separates where the ScaleZero cost actually comes from.
//
// WHY A SEPARATE FILE. test_fold_int2.cu is the verified correctness harness; a sweep needs many extra kernel
// instantiations and would slow its build for everyone. Nothing here is on the correctness path.
//
// WHAT THE SWEEP FOUND (20 configs, 42.0% -> 50.2%), and it took three attempts to state it correctly. Two
// INDEPENDENT factors, measured over 25 points across all three widths (this sweep plus the int4/int2/int1 ladders in
// test_width_acu) -- full derivation in fold_traits.hpp and fold_derivation/README.md:
//
//                        cvt/mma = 4              cvt/mma = 8
//     warps/CU >= 32     mean 52.1%  n=6          mean 39.1%  n=6     (-13.0)
//     warps/CU <= 16     mean 46.1%  n=16         mean 32.0%  n=2     (-20.1)
//                             (-6.0)
//
//   cvt/mma = 128/WM   converter amortisation. Separates with NO overlap in either row: a real ~13-point cliff.
//   warps/CU           occupancy, and it MUST include the register limit -- acu showed shared memory allowing 10
//                      blocks where registers allowed 4, and registers are billed rounded up to a POWER OF TWO, so
//                      129 costs as much as 256. Overlaps at cvt/mma=4, so ~6 points of mean shift, not a cliff.
//
// The two earlier stories looked contradictory because each varied only one axis: this sweep's high-warp configs are
// all WM=16 (cvt/mma=8), so warps/CU appeared to HURT, while the ladder's are WM=32, so it appeared to HELP. The
// REFUTED w16x64 rows below are kept as the evidence for the cvt/mma axis, and the TEST amort=4 row as the evidence
// that pushing cvt/mma below 4 buys +0.3 points.
//
// THE WEIGHT BUFFER DEPENDS ON (Bits, TN, TK) ONLY -- it is WN-INVARIANT, verified byte-for-byte in
// fold_derivation/l20_derived_offline.cu at WN=32 and WN=64. So configs sharing a TK share a buffer, and the sweep
// regroups by TK instead of rebuilding per config. TK=64 is the exception: it needs the bit-granular packer, so it
// gets its own buffer from nfold_place_bits_int1_tk64.
//
// THE ZERO DIAGNOSTIC (FOLD_ZDIAG=1), and its ANSWER -- which was not one of the two options it offered. The split
// was "if the zero cost tracks gs it is the COPY (reloads = K/gs); if it is flat in gs it is the TRANSFORM
// (= K/16)". The cost does track gs (+49.5us at gs=32, +81.9us at gs=16 at TK=128) -- and then the stride-0 scale
// broadcast cut the reload's smem requests to a QUARTER and changed nothing at all: delta +49.50 vs +49.6 recorded,
// +81.94 vs +83.0. So a third quantity also tracks gs: the per-reload smem round-trip LATENCY, which depends on the
// NUMBER of reloads and not on how wide each one is. Consistent with the kernel being latency-bound at cvt/mma=4.
// The remedy is an EARLIER reload (prefetch the next group's scale), not a narrower one, and not the fused FMA
// either -- that attempt regressed 52.3% -> 33.5% by replacing a vectorized cute::transform with a scalar
// fp16->float->fp16 loop.
//
//   Build: TARGET=test_int1_sweep ./build.sh ; run: ./<bin> [N] [K] [gs]
//   FOLD_ZDIAG=1 adds the ScaleOnly/ScaleZero pairs at both group sizes.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include "fold_traits.hpp"
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
  // regs is an ESTIMATE (fold_traits.hpp) and it is printed rather than asserted, because the over-budget configs
  // compile and run -- they just collapse. TK=256 with zero measures 4.5% MFU at 352 estimated regs, which is what
  // spilling looks like, and having the number on the same line as the timing is how that became visible at all.
  constexpr int R = fold::regs_per_thread<TM, TN, TK, WM, WN, Q == QM::FinegrainedScaleZero>;
  const int blocks = std::min(262144 / smem, 64 / warps);
  // cvt/mma = 128/WM is the discriminator the register count missed: it separated the first 20 points of this sweep
  // with no overlap (4.00 -> 40.9-50.2%, 8.00 -> 31.9-39.8%). Printed next to blk so the two can be read against
  // each other, since they frequently pull in opposite directions.
  std::printf("  (%3d,%3d,%3d) w%dx%-3d s%d %-9s gs=%-3d | warps=%d smem=%6dB blk=%-2d regs=%-3d%s cvt/mma=%d | %8.2f us  %5.1f%% MFU  %s\n",
              TM, TN, TK, WM, WN, ST, Q == QM::FinegrainedScaleZero ? "ScaleZero" : "ScaleOnly", gs,
              warps, smem, blocks, R, R > 256 ? "!" : " ", fold::cvt_per_mma<WM>,
              us, 100.0 * tf * 1e12 / 500.0e12, note);
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
  // Which scale-fragment path this binary was BUILT with, printed at RUN time. A build-time message() is useless
  // here: build.sh redirects cmake's stdout to cmake.log, so an A/B could silently compare two identical binaries
  // and the operator would have no way to notice. regs_per_thread is gated on the same macro, so the number moves
  // with the path and is a second, independent witness.
#if defined(PPU_SCALE_BCAST) && (PPU_SCALE_BCAST == 0)
  const char* scale_path = "MATERIALISED (old path, PPU_SCALE_BCAST=0)";
#else
  const char* scale_path = "stride-0 BROADCAST (default)";
#endif
  std::printf("int1 sweep  M=%d N=%d K=%d gs=%d   (buffer depends on (TN,TK) only -- WN-invariant, see l20)\n"
              "  scale fragment: %s   -> regs at (32,128,64) w32x64 = %d ScaleOnly / %d ScaleZero\n\n",
              PM, PN, PK, gs, scale_path,
              fold::regs_per_thread<32,128,64,32,64,false>, fold::regs_per_thread<32,128,64,32,64,true>);
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
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 64, 32, 64, 2>(b, gs, "B+C: TM=32 s2   <- 50.1% measured, best so far");
  // A PRESCRIPTION DERIVED FROM REGISTERS ALONE, AND HOW IT FAILED. accum = WM*WN/32 depends only on the warp
  // shape and the delivery bound constrains only WN, so w16x64 keeps TK=64 legal while halving the accumulator:
  // 128 estimated regs against 176. Measured 39.7% against 50.0% -- fewer registers, comparable occupancy, 10.3
  // points WORSE. These four rows are kept because they are the evidence that the objective is cvt/mma = 128/WM
  // (fold_traits.hpp), not the register count: every w16 row lands in 38.4-39.8% and every w32 row in 40.9-50.2%.
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 64, 16, 64, 2>(b, gs, "REFUTED w16x64 s2 (cvt/mma 8)");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 64, 16, 64, 3>(b, gs, "REFUTED w16x64 s3 (cvt/mma 8)");
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 64, 16, 64, 2>(b, gs, "REFUTED w16x64 s2 TM=16 (blk=32, still 39.8)");
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 64, 32, 64, 2>(b, gs, "w32x64 s2 TM=64");
  run_cfg<QM::FinegrainedScaleOnly, 32, 256, 64, 32, 64, 2>(b, gs, "w32x64 s2 TN=256  <- 50.2% measured, best");

  // THE DECISIVE TEST: tier 1 (register budget) against tier 2 (converter amortisation), which the three-tier model
  // says must be resolved in favour of tier 1. w64x64 is the ONLY int1 shape that reaches amort=4 (cvt/mma=2, half
  // the converter work of the current best) and the register minimum over the whole legal space is 272 -- 16 over
  // budget, and regs_per_thread is an ESTIMATE, so 272 is exactly the margin where it might be wrong.
  //   predicted: collapses to ~25-40% despite the best cvt/mma in the sweep
  //   if instead it beats 50.2%: the 256 threshold is wrong and int1 has another factor-of-2 available
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 64, 64, 64, 2>(b, gs, "TEST amort=4: cvt/mma=2 but 272 regs");

  // stages=2 was the single biggest knob at TK=128 (42.0 -> 46.5), so finish exploring it there. This needs the
  // TK=128 buffer back, so it is re-uploaded rather than run against the TK=64 one.
  std::printf("\n== back to TK=128 to finish the stages=2 line\n");
  upload_weights<128, 128>(b);
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 128, 32, 32, 2>(b, gs, "TK=128 TM=64 s2");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 16, 32, 2>(b, gs, "TK=128 w16x32 s2");

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
