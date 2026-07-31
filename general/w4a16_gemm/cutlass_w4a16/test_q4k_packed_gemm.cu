// END TO END: real Q4_K weights whose scale/zero reach the device in the GGUF's OWN form. Plan #20 option E, step 4.
//
// Everything else on this path feeds the kernel two pre-multiplied fp16 planes. Here the device gets one 16 B unit per
// (superblock, column) -- d, dmin and the 8+8 six-bit codes -- and the mainloop decodes it in registers. This is the
// gate for the whole packed channel: if it matches the fixture's golden, then the offline layout, the cp.async, the smem
// tile, the per-lane unit load and the register decode are all right together.
//
// THE PLANE IS BUILT HERE, IN C++, WITH THE KERNEL'S OWN put_code. Not in python: the bit map would then exist twice,
// and every defect this work has hit came from one relation living in two places. dump_packed_scale.py stays as it is,
// emitting sc/mn as int8 and d/dmin as fp16 (RWMOEP); this file only rearranges those into the unit the kernel reads.
//
// WHY TileK MUST BE 256 HERE: the packed path is applicable only when Scale_TileK == 8 (kPackedScaleOn), which at gs=32
// means TileK == 256 -- one superblock per k-tile. At smaller Scale_TileK the unit still carries eight groups' codes
// while the tile needs fewer, and l94 (4) measured that as a LOSS. A build with a different TileK silently keeps the
// fp16 path, so this harness pins the one shape that exercises the new code.
//
//   fixture: python3 real_weight/dump_packed_scale.py <file.gguf> --tensor blk.11.ffn_down.weight --ncols 128 --m 16 \
//                       --out real_weight/q4k_packed.bin
//   build:   PPU_DEFS=PPU_PACKED_SCALE=1 TARGET=test_q4k_packed_gemm ./build.sh
//   run:     $BIN real_weight/q4k_packed.bin
//
// Built WITHOUT the macro it must still MATCH -- the fp16 planes are then rebuilt on the host from the same codes, so a
// failure there is the fixture or this file, not the packed channel. That A/B is the point: same data, same golden, two
// paths through the kernel.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "unfused_weight_dequantize.hpp"
#include "moe_grouped_ppu.cuh"
#include "cutlass/gguf_packed_scale.h"
#include "rwmoep_loader.hpp"

using cutlass::half_t;
using int4_t  = cutlass::int4b_t;
using QM      = moe_grouped_ppu::QuantMode;
using DStride = moe_grouped_ppu::DStride;
using GS      = moe_grouped_ppu::GroupShape;

int main(int argc, char** argv) {
  char const* path = argc > 1 ? argv[1] : "real_weight/q4k_packed.bin";
  rwmoep::File f;
  if (!f.load(path)) return 64;
  if (f.mode != 1 || f.ktype != 4) { std::printf("expected a Q4_K scale+zero fixture\n"); return 64; }
  int const L = f.L, M = f.M, N = f.N, K = f.K, gs = f.gs, sb = f.sb;
  int const scale_k = f.scale_k(), nsb = f.nsb(), gpsb = sb / gs;      // groups per superblock, 8 for Q4_K
  // BOTH must be %256, and this is what the last three failing runs were about. moe_grouped_ppu.cuh:352 selects the
  // interleaved B layout by `n % 256 == 0 && k % 256 == 0`; below that it takes ColumnMajor instead. The B here is
  // preprocessed with interleave-256, so an N that is not a multiple of 256 hands an interleaved buffer to a kernel
  // reading it as plain column-major -- deterministic garbage, and INDEPENDENT of the scale mode and the tile, which is
  // exactly the signature that showed up: rowA, rowB and rowC produced bit-identical wrong output.
  //
  // The signature is worth remembering: output that does not change when you change the quant mode or the tile is not a
  // scale bug. It is upstream of both.
  if (K % 256 || N % 256) {
    std::printf("need K%%256==0 AND N%%256==0: moe_grouped_ppu.cuh:352 picks the interleaved layout only then, and this\n"
                "  harness preprocesses B with interleave-256 (got K=%d N=%d) -- regenerate the fixture with --ncols 256\n",
                K, N);
    return 64;
  }
  std::printf("[q4k-packed] L=%d M=%d N=%d K=%d gs=%d nsb=%d groups/superblock=%d\n", L, M, N, K, gs, nsb, gpsb);

  // ---- THE PACKED PLANE, [L][nsb][N][16] bytes, through the kernel's own put_code -------------------------------
  std::vector<uint8_t> plane((size_t)L * nsb * N * cutlass::gguf_packed::kUnitBytes, 0);
  for (int e = 0; e < L; ++e)
    for (int b = 0; b < nsb; ++b)
      for (int n = 0; n < N; ++n) {
        uint8_t* unit = plane.data()
                      + ((size_t)(e * nsb + b) * N + n) * cutlass::gguf_packed::kUnitBytes;
        uint16_t const dbits  = f.d   [(size_t)e * nsb * N + (size_t)b * N + n].raw();
        uint16_t const dmbits = f.dmin[(size_t)e * nsb * N + (size_t)b * N + n].raw();
        std::memcpy(unit + 0, &dbits,  2);
        std::memcpy(unit + 2, &dmbits, 2);
        for (int g = 0; g < gpsb; ++g) {
          size_t const gi = (size_t)e * scale_k * N + (size_t)(b * gpsb + g) * N + n;
          cutlass::gguf_packed::put_code(unit, g, 0, int(f.sc[gi]));
          cutlass::gguf_packed::put_code(unit, g, 1, int(f.mn[gi]));
        }
      }
  // Read it straight back with the DECODE the mainloop uses, and require the fp16 planes it yields to equal what the
  // host would have computed. A wrong offline arrangement then fails HERE, with a group index, instead of surfacing as a
  // wrong GEMM result with nothing to point at.
  {
    int bad = 0;
    for (int e = 0; e < L && bad < 5; ++e)
      for (int b = 0; b < nsb && bad < 5; ++b)
        for (int n = 0; n < N && bad < 5; ++n) {
          uint8_t const* unit = plane.data() + ((size_t)(e * nsb + b) * N + n) * cutlass::gguf_packed::kUnitBytes;
          for (int g = 0; g < gpsb; ++g) {
            size_t const gi = (size_t)e * scale_k * N + (size_t)(b * gpsb + g) * N + n;
            auto const got = cutlass::gguf_packed::group_of<0, true, 8>(unit, g);
            float const ws = float(f.d[(size_t)e*nsb*N + (size_t)b*N + n]) * float(int(f.sc[gi]));
            float const wz = 8.f * ws - float(f.dmin[(size_t)e*nsb*N + (size_t)b*N + n]) * float(int(f.mn[gi]));
            if (float(got.scale) != float(half_t(ws)) || std::abs(float(got.zero) - wz) > 1e-2f * (std::abs(wz) + 1.f)) {
              std::printf("  [plane] e=%d b=%d n=%d g=%d scale %.6g/%.6g zero %.6g/%.6g\n",
                          e, b, n, g, float(got.scale), ws, float(got.zero), wz);
              ++bad;
            }
          }
        }
    std::printf("  plane round trip through group_of<0,true,8>: %s\n", bad ? "MISMATCH" : "ok");
    if (bad) return 3;
  }

  // ---- the fp16 planes, for the build WITHOUT the macro (same data, so the A/B is honest) -----------------------
  std::vector<half_t> hSc((size_t)L * scale_k * N), hZr((size_t)L * scale_k * N);
  for (int e = 0; e < L; ++e)
    for (int g = 0; g < scale_k; ++g)
      for (int n = 0; n < N; ++n) {
        size_t const gi = (size_t)e * scale_k * N + (size_t)g * N + n;
        size_t const bi = (size_t)e * nsb * N + (size_t)(g / gpsb) * N + n;
        float const s = float(f.d[bi]) * float(int(f.sc[gi]));
        hSc[gi] = half_t(s);
        hZr[gi] = half_t(8.f * s - float(f.dmin[bi]) * float(int(f.mn[gi])));
      }

  int const total = L * M;

  // ---- HOST GOLDENS, so the kernel is compared against a model this file can defend --------------------------
  // hAff must agree with the fixture's own golden; if it does not, the disagreement is in this file's understanding of
  // the fixture and nothing about the kernel is implicated. hSco is the SAME data with the zero forced to 0, which is the
  // only configuration the grouped int4 path has ever been validated in (bench_cutlass_w4a16's xcheck_grouped runs
  // ScaleOnly with zeros=nullptr). Having both separates "affine is broken" from "my layout is wrong".
  std::vector<float> hAff((size_t)total * N, 0.f), hSco((size_t)total * N, 0.f);
  for (int e = 0; e < L; ++e)
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k) {
        float const a = float(f.A[((size_t)e * M + m) * K + k]);
        if (a == 0.f) continue;
        size_t const grow = (size_t)e * scale_k * N + (size_t)(k / gs) * N;
        for (int n = 0; n < N; ++n) {
          float const q = float(f.q[(size_t)e * N * K + (size_t)n * K + k]);   // q is [N][K]
          float const sv = float(hSc[grow + n]);
          hAff[((size_t)e * M + m) * N + n] += a * (sv * q + float(hZr[grow + n]));
          hSco[((size_t)e * M + m) * N + n] += a * (sv * q);
        }
      }
  {
    double mr = 0;
    for (size_t i = 0; i < (size_t)total * N; ++i) {
      double const g = double(float(f.gold[i]));
      mr = std::max(mr, std::abs(hAff[i] - g) / (std::abs(g) + 1e-3));
    }
    std::printf("  host affine model vs the fixture's golden: max_rel=%.3e %s\n", mr,
                mr < 5e-2 ? "ok" : "  <-- THIS FILE misreads the fixture; the kernel is not implicated");
    if (mr >= 5e-2) return 3;
  }

  // ---- B: nibbles then preprocess, exactly as the other real-weight harnesses --------------------------------
  std::vector<int8_t> Bbuf((size_t)L * K * N / 2);
  for (int e = 0; e < L; ++e) {
    std::vector<int8_t> packed((size_t)K * N / 2);
    // LINEAR PAIRS OVER THE FILE'S OWN ORDER, then dims {K,N} -- byte for byte what test_moe_grouped_real.cu does, and
    // that path is validated against real GPTQ and Q4_K weights.
    //
    // My first version "transposed back" because the writer stores q.T as [N][K] and the packer is told {K,N}, which
    // looks like a contradiction. It is not one to undo: the packer consumes the BUFFER, and the validated convention
    // pairs consecutive file elements. Transposing produced bad=2005/2048 on the REFERENCE path, i.e. before the packed
    // channel was even involved -- which is exactly why the reference build runs first.
    for (size_t i = 0; i < (size_t)K * N / 2; ++i) {
      int8_t const lo = f.q[(size_t)e * K * N + 2 * i] & 0xF;
      int8_t const hi = f.q[(size_t)e * K * N + 2 * i + 1] & 0xF;
      packed[i] = int8_t((hi << 4) | lo);
    }
    preprocess_weights_for_mixed_gemm<false, 256>(
        (int8_t*)&Bbuf[(size_t)e * K * N / 2], packed.data(), {(size_t)K, (size_t)N},
        QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);
  }

  cutlass::DeviceAllocation<half_t> dA((size_t)total * K), dD((size_t)total * N);
  cutlass::DeviceAllocation<int4_t> dB((size_t)L * K * N);
  cutlass::DeviceAllocation<uint8_t> dPlane(plane.size());
  cutlass::DeviceAllocation<half_t> dSc((size_t)L * scale_k * N), dZr((size_t)L * scale_k * N);
  dA.copy_from_host(f.A.data());
  dB.copy_from_host(reinterpret_cast<int4_t const*>(Bbuf.data()));
  dPlane.copy_from_host(plane.data());
  dSc.copy_from_host(hSc.data());
  dZr.copy_from_host(hZr.data());

  std::vector<GS> shp(L); std::vector<half_t*> pd(L); std::vector<DStride> sd(L); std::vector<int> gm(L), offs(L);
  auto out_stride = [&](int m) { return cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(m, N, 1)); };
  for (int e = 0; e < L; ++e) {
    shp[e] = cute::make_shape(M, N, K); pd[e] = dD.get() + (size_t)e * M * N;
    sd[e] = out_stride(M); gm[e] = M; offs[e] = e * M;
  }
  cutlass::DeviceAllocation<GS> shpd(L);        shpd.copy_from_host(shp.data());
  cutlass::DeviceAllocation<half_t*> pdd(L);    pdd.copy_from_host(pd.data());
  cutlass::DeviceAllocation<DStride> sdd(L);    sdd.copy_from_host(sd.data());
  cutlass::DeviceAllocation<int> gmd(L);        gmd.copy_from_host(gm.data());
  cutlass::DeviceAllocation<int> offd(L);       offd.copy_from_host(offs.data());
  size_t const wsb = (size_t)cutlass::ceil_div(M,16) * cutlass::ceil_div(N,64) * (size_t)L * 64;
  cutlass::DeviceAllocation<char> ws(wsb);
  // WHICH POINTER ROW 2 GETS depends on the build, and it has to: with PPU_PACKED_SCALE the collective reinterprets
  // ptr_S as the byte plane, without it the same pointer is an fp16 plane. Row 1's tile has Scale_TileK == 2, so it is
  // always on the fp16 path and always takes dSc.
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
  half_t const* scale_arg = reinterpret_cast<half_t const*>(dPlane.get());
  char const*   which     = "row2 reads the gguf's own 16 B units";
#else
  half_t const* scale_arg = dSc.get();
  char const*   which     = "row2 reads fp16 planes (reference build)";
#endif
  std::printf("  %s\n", which);

  // TWO ROWS, so a failure is attributable. Row 1 is the tile validated against real weights in
  // test_moe_grouped_real.cu (TK=64 at gs=32) -- its Scale_TileK is 2, so kPackedScaleOn is FALSE there and it exercises
  // only this harness, the fixture and the B packing. Row 2 is the decode band's winner (16x128:256 w16x16 s2), the tile
  // the splitk bench validates numerically at TK=256, which is where Scale_TileK == 8 and the packed path turns on.
  //
  // The first version ran ONE row at 64,64,256,32,32,3 -- a tile no harness has ever validated -- so when it failed there
  // was no way to tell a bad tile from a bad channel. Both rows print their own verdict.
  auto run_and_check = [&](char const* tag, float const* ref, auto&& launch) {
    CUTLASS_PPU_CHECK(hggcMemset(dD.get(), 0, sizeof(half_t) * (size_t)total * N));
    int const before = moe_grouped_ppu::moeg_fail_count();
    launch();
    CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
    if (moe_grouped_ppu::moeg_fail_count() != before) {
      std::printf("  %-28s REFUSED to launch -- nothing was computed\n", tag);
      return 2;
    }
    std::vector<half_t> hD((size_t)total * N);
    dD.copy_to_host(hD.data());
    int bad = 0, nonfinite = 0; double mr = 0, gmax = 0;
    for (size_t i = 0; i < (size_t)total * N; ++i) {
      double const g = double(ref[i]), d = double(float(hD[i]));
      if (!std::isfinite(d)) { ++nonfinite; continue; }        // NaN compares false against everything; count it
      gmax = std::max(gmax, std::abs(g));
      double const rel = std::abs(d - g) / (std::abs(g) + 1e-3);
      if (rel > mr) mr = rel;
      if (std::abs(d - g) > 2e-2 + 6e-2 * std::abs(g)) { if (bad < 4) std::printf("      i=%zu got=%.4f gold=%.4f\n", i, d, g); ++bad; }
    }
    std::printf("  %-28s bad=%d/%d max_rel=%.3e |gold|max=%.4g %s%s\n", tag, bad, total * N, mr, gmax,
                bad || nonfinite ? "MISMATCH" : "MATCH", gmax == 0 ? "  <-- VACUOUS" : "");
    if (nonfinite) std::printf("      NON-FINITE outputs: %d\n", nonfinite);
    return bad + nonfinite + (gmax == 0 ? 1 : 0);
  };

  int fail = 0;
  // rowA is bench_cutlass_w4a16::xcheck_grouped's configuration EXACTLY -- ScaleOnly, zeros=nullptr, offsets=nullptr,
  // 64x64:128 w32x32 s3 -- the only single-plane int4 setup ever checked against an external golden. Compared against
  // the host's scale-only model. If rowA fails, the fault is in A/B/scale layout and has nothing to do with the zero.
  fail += run_and_check("rowA scale-only 64x64:128", hSco.data(), [&]{
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleOnly, 64, 64, 128, 32, 32, 3>(
        dA.get(), dB.get(), dSc.get(), nullptr, pdd.get(), sdd.get(), gmd.get(),
        M, N, K, L, gs, shpd.get(), shp.data(), nullptr, ws.get(), wsb, nullptr);
  });
  // rowB adds ONLY the zero channel. rowA ok + rowB bad isolates the affine path itself.
  fail += run_and_check("rowB affine   64x64:128", hAff.data(), [&]{
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 64, 64, 128, 32, 32, 3>(
        dA.get(), dB.get(), dSc.get(), dZr.get(), pdd.get(), sdd.get(), gmd.get(),
        M, N, K, L, gs, shpd.get(), shp.data(), nullptr, ws.get(), wsb, nullptr);
  });
  // rowC is the only tile where Scale_TileK == 8, i.e. the one the packed channel runs in.
  fail += run_and_check("rowC affine   16x128:256", hAff.data(), [&]{
    moe_grouped_ppu::filter_and_run<QM::FinegrainedScaleZero, 16, 128, 256, 16, 16, 2>(
        dA.get(), dB.get(), scale_arg, dZr.get(), pdd.get(), sdd.get(), gmd.get(),
        M, N, K, L, gs, shpd.get(), shp.data(), nullptr, ws.get(), wsb, nullptr);
  });

  std::printf("== %s ==   (rowA = the only validated single-plane int4 setup; rowB adds the zero channel; rowC is the\n"
              "             Scale_TileK == 8 tile, and %s)\n", fail ? "FAIL" : "PASS", which);
  return fail ? 1 : 0;
}
