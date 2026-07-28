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
#include "xplane_offline.hpp"
#include "moe_grouped_ppu.cuh"

// If this fails, the actlize SUBMODULE on the box is stale: the Kernels gitlink moved but `git submodule update` did
// not run, so the build would silently produce a binary identical to the previous one. That ambiguity is what made an
// "every acu counter is identical" A/B uninterpretable -- so it is a compile error, not a runtime surprise.
#if !defined(PPU_SCALE_FRAGMENT_API) || PPU_SCALE_FRAGMENT_API < 3
#error "stale actlize submodule: run `git submodule update --init third_party/actlize` and rebuild"
#endif

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
  // WHICH MODEL, and whether this binary was built with chunking. The previous version always printed the UNCHUNKED
  // model, so a chunked run showed regs=260/bill=512 beside a 63.7% MFU that is only possible at bill=256. A log has
  // to describe the run that produced it -- third time in this work.
#if defined(PPU_B_CHUNK) && (PPU_B_CHUNK != 0)
  constexpr bool kCh = true;
  constexpr int R = fold::regs_per_thread_chunked<TM, TN, TK, WM, WN, Q == QM::FinegrainedScaleZero>;
  const int wcu   = fold::warps_per_cu_chunked<TM, TN, TK, WM, WN, ST, 1, 32, Q == QM::FinegrainedScaleZero>;
#else
  constexpr bool kCh = false;
  constexpr int R = fold::regs_per_thread<TM, TN, TK, WM, WN, Q == QM::FinegrainedScaleZero>;
  const int wcu   = fold::warps_per_cu<TM, TN, TK, WM, WN, ST, 1, 32, Q == QM::FinegrainedScaleZero>;
#endif
  // NChunk, so a config whose chunking silently did not apply is visible. TK=256 came back bit-identical to the
  // unchunked run, which is what a skipped path looks like.
  constexpr int kSlots = WN * TK / 32, kKbm = kSlots / 128, kNChunk = kKbm ? (TK / 16) / kKbm : 0;
  // blk USED TO BE min(smem, warps) HERE AND THAT WAS WRONG -- it omitted the register limit, so it printed 23 for
  // the best config where the hardware gives 8 (131072/(256*64)); acu confirmed the register limit directly
  // ("Block Limit Registers 4" against "Block Limit Shared Mem 10" on a ladder rung). Every blk number in the early
  // analysis of this sweep was inflated for the 256-billing configs. warps/CU is the quantity the two-factor model
  // uses, so print that, and keep the smem-only bound beside it since it shows WHICH limit binds.
  const int blk_smem = 262144 / smem;
  const int blocks = wcu / warps;
  // cvt/mma = 128/WM is the discriminator the register count missed: it separated the first 20 points of this sweep
  // with no overlap (4.00 -> 40.9-50.2%, 8.00 -> 31.9-39.8%). Printed next to blk so the two can be read against
  // each other, since they frequently pull in opposite directions.
  std::printf("  (%3d,%3d,%3d) w%dx%-3d s%d %-9s gs=%-3d | wrp=%d smem=%6dB blk=%-2d w/CU=%-2d regs=%-3d bill=%-3d cvt=%d %s | %8.2f us  %5.1f%% MFU  %s\n",
              TM, TN, TK, WM, WN, ST, Q == QM::FinegrainedScaleZero ? "ScaleZero" : "ScaleOnly", gs,
              warps, smem, blocks, wcu, R, fold::regs_billed<R + fold::regs_measured_offset>, fold::cvt_per_mma<WM>,
              kCh ? (kNChunk ? "CHUNK" : "chunk-N/A") : "plain",
              us, 100.0 * tf * 1e12 / 500.0e12, note);
}

// upload the weight buffer for a given TK. ONE derived call covers every TK: place_derived walks the fold destination
// when F > 1 and the interleave-256 one when F == 1, so the old two-branch form -- bit-granular packer for TK=64,
// preprocess+nfold_regroup_gmem otherwise -- collapses. WN is a parameter because int1's delivery bound is WN >= 64 at
// TK=64 and WN >= 32 at TK >= 128, and the map depends on it; passing 32 for a TK=64 buffer produced a buffer for a
// configuration that cannot run. Byte-identical to the old path on every TK here (fold_derivation/l64).
template <int TN, int TK, int WN>
static void upload_weights(Buf& b) {
  constexpr int contig = TK / 8, F = contig >= 32 ? 1 : 32 / contig;
  std::vector<uint8_t> q((size_t)PK * PN);                       // [K][N], one code per byte
  for (size_t i = 0; i < q.size(); ++i) q[i] = uint8_t((i * 2654435761u >> 7) & 1);
  std::vector<int8_t> out((size_t)PK * PN / 8, 0);
  xplane::place_derived<1, 64, TN, TK, 32, WN, F>(out.data(), q, PN, PK);
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
  upload_weights<128, 128, 32>(b);
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 128, 16, 32, 3>(b, gs, "A: TM=16");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 3>(b, gs, "A: TM=32  <- measured 42.0% at gs=32");
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 128, 32, 32, 3>(b, gs, "A: TM=64");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 64, 3>(b, gs, "D: WN=64, same buffer");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 2>(b, gs, "C: stages=2");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 4>(b, gs, "C: stages=4");
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 128, 16, 32, 2>(b, gs, "A+C: TM=16 s2");

  std::printf("\n== TK=256 group (shipped offline, F=1). B: most atoms per iteration = best hiding.\n");
  upload_weights<128, 256, 32>(b);
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 256, 16, 32, 2>(b, gs, "B: TK=256 TM=16 s2");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 256, 32, 32, 2>(b, gs, "B: TK=256 TM=32 s2");
  run_cfg<QM::FinegrainedScaleOnly, 16, 128, 256, 16, 32, 3>(b, gs, "B: TK=256 TM=16 s3");

  std::printf("\n== TK=64 group (bit-granular packer, WN>=64 required by the delivery bound).\n");
  upload_weights<128, 64, 64>(b);
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
  upload_weights<128, 128, 32>(b);
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 128, 32, 32, 2>(b, gs, "TK=128 TM=64 s2");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 16, 32, 2>(b, gs, "TK=128 w16x32 s2");

  // CHUNK GROUP. Build with PPU_DEFS=PPU_B_CHUNK=1 and run this twice, with and without, for the A/B.
  //
  // The B fragment drops from 4*MMA_N*MMA_K to 4*MMA_N registers, so the saving is 4*(WN/16)*(TK/16 - 1) -- it GROWS
  // with TK. That moves the interesting configs away from TK=64: at TK=64 the estimate goes 164 -> 116, and
  // 116 + 22 (the measured offset) = 138 > 128, so it stays in the 256 bucket. acu confirms exactly that -- 186 -> 142
  // registers with occupancy unchanged. At TK=128 the estimate reaches 106, i.e. 128 measured, which is EXACTLY the
  // boundary: the offset is fitted on two points, so +-2 flips the bucket and the model cannot call it.
  //
  // TK=256 is the one worth the most: it currently measures 23.0% because 320 estimated registers bill at 512. With
  // chunking the estimate is 170 -> 192 measured -> 256 billed, and warps/CU doubles. That is a config going from
  // register-dead to alive, not a tuning delta.
  //
  // NChunk = K_ATOM_PER_COPY = MMA_K / K_BLOCK_MAX, so it is 4 at TK=64 and 8 at TK=128/256 with WN=32. at() and
  // keep() are layout compositions over tCrB_mma's own layout, so they follow MMA_N/MMA_K without code changes --
  // which is the point of having made them derived rather than hand-typed.
  std::printf("\n== CHUNK GROUP (PPU_B_CHUNK). B regs 4*MMA_N*MMA_K -> 4*MMA_N, so the saving grows with TK.\n");
  upload_weights<128, 128, 32>(b);
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 2>(b, gs, "CHUNK: est 162->106, bucket 256->128 (BOUNDARY)");
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 128, 32, 32, 2>(b, gs, "CHUNK: est 162->106, warps/CU 16->32");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 3>(b, gs, "CHUNK: same at s3");
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 64, 3>(b, gs, "CHUNK: est 260->148, bucket 512->256");
  upload_weights<128, 256, 32>(b);
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 256, 32, 32, 2>(b, gs, "CHUNK: was 23.0% (512 bucket) -> 256");
  run_cfg<QM::FinegrainedScaleOnly, 64, 128, 256, 32, 32, 2>(b, gs, "CHUNK: TK=256 TM=64");
  upload_weights<128, 64, 64>(b);
  run_cfg<QM::FinegrainedScaleOnly, 32, 128, 64, 32, 64, 2>(b, gs, "CHUNK control: TK=64 should NOT cross (138>128)");

  if (getenv("FOLD_ZDIAG")) {
    std::printf("\n== ZERO DIAGNOSTIC. reloads = K/gs, transforms = K/16. If the zero cost tracks gs it is the COPY\n");
    std::printf("   (interleave sS/sZ); if it is flat in gs it is the TRANSFORM (fuse multiplies+plus into one FMA).\n");
    for (int g : {32, 16}) {
      Buf bz; make_buffers(bz, g);
      upload_weights<128, 128, 32>(bz);
      run_cfg<QM::FinegrainedScaleOnly, 32, 128, 128, 32, 32, 3>(bz, g, "ScaleOnly TK=128");
      run_cfg<QM::FinegrainedScaleZero, 32, 128, 128, 32, 32, 3>(bz, g, "ScaleZero TK=128");
      upload_weights<128, 256, 32>(bz);
      run_cfg<QM::FinegrainedScaleOnly, 32, 128, 256, 32, 32, 2>(bz, g, "ScaleOnly TK=256 (2x transforms/iter)");
      run_cfg<QM::FinegrainedScaleZero, 32, 128, 256, 32, 32, 2>(bz, g, "ScaleZero TK=256 (2x transforms/iter)");
    }
    std::printf("   TK=256 doubles transforms per iteration while halving iterations -- transforms total is\n");
    std::printf("   invariant, but reloads per iteration double. Comparing the zero delta at TK=128 vs TK=256\n");
    std::printf("   separates a per-reload cost from a per-transform one.\n");
  }
  return 0;
}
