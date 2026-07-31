// LOAD THE NATIVE Q4_K scale/zero AND DECODE IT ON THE DEVICE. Plan #20, the loading half.
//
// Everything shipped so far consumes fp16 scale/zero that the offline pre-multiplied; the gguf's own form -- 6-bit sc,
// 6-bit mn, one fp16 d and dmin per 256 k -- never reaches the device. This harness closes that: it loads an RWMOEP
// fixture, uploads the NATIVE planes (int8 codes + the two fp16 per superblock), decodes them ON THE DEVICE through
// gguf_scale_decode.hpp, and checks the result against the same relation computed on the host in fp16.
//
// Why this is the right first gate, and not the GEMM: the decode is the ONE piece of new device arithmetic. Landing it
// inside the mainloop means moving four transform arms onto a byte-granular scale tile, and if that comes out wrong
// there is no way to tell "the decode is wrong" from "the tile/fragment plumbing is wrong". Here the decode runs on the
// hardware, on real weights, in isolation -- so the collective change inherits a verified decode.
//
// It also checks the thing a bit-map test cannot: that d/dmin are paired with the right SUPERBLOCK. Every group inside
// one superblock shares d, so an off-by-one in b = (g*gs)/sb shifts eight groups at once and shows up here as a whole
// band of columns being wrong.
//
//   fixture: python3 real_weight/dump_packed_scale.py <file.gguf> --tensor blk.11.ffn_down.weight --ncols 64 --m 16 \
//                       --out real_weight/q4k_packed.bin
//   build:   TARGET=test_q4k_native_scale ./build.sh
//   run:     $BIN real_weight/q4k_packed.bin
#include <cstdio>
#include <cmath>
#include <vector>
#include "cutlass/half.h"
#include "cutlass/util/device_memory.h"
#include "ppu_include.hpp"
#include "helper.h"                        // CUTLASS_PPU_CHECK
#include "rwmoep_loader.hpp"

using cutlass::half_t;

int main(int argc, char** argv) {
  char const* path = argc > 1 ? argv[1] : "real_weight/q4k_packed.bin";
  rwmoep::File f;
  if (!f.load(path)) {
    std::printf("  (generate it with real_weight/dump_packed_scale.py -- see the header of this file)\n");
    return 64;
  }
  int const sk = f.scale_k(), nb = f.nsb(), N = f.N, L = f.L;
  size_t const per = (size_t)sk * N, tot = (size_t)L * per;
  int const has_min = (f.mode == 1) ? 1 : 0;

  cutlass::DeviceAllocation<int8_t> dsc(tot), dmn(has_min ? tot : 1);
  cutlass::DeviceAllocation<half_t> dd((size_t)L * nb * N), ddm(has_min ? (size_t)L * nb * N : 1),
                                    dS(tot), dZ(tot);
  dsc.copy_from_host(f.sc.data());
  dd.copy_from_host(f.d.data());
  if (has_min) { dmn.copy_from_host(f.mn.data()); ddm.copy_from_host(f.dmin.data()); }

  int const threads = 128, blocks = int((tot + threads - 1) / threads);
  rwmoep::decode_planes<<<blocks, threads>>>(dsc.get(), has_min ? dmn.get() : nullptr, dd.get(),
                                             has_min ? ddm.get() : nullptr, dS.get(), dZ.get(),
                                             L, N, sk, nb, f.gs, f.sb, has_min, f.z_mul);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<half_t> hS(tot), hZ(tot);
  dS.copy_to_host(hS.data());
  dZ.copy_to_host(hZ.data());

  // HOST REFERENCE, in fp16, formed the way the offline used to form its fp16 planes: the exact product rounded once.
  // sc, mn <= 63 need 6 bits and d, dmin are already fp16, so the product has at most 17 significant bits and fp32
  // holds it exactly -- which is why this is an EQUALITY check and not a tolerance.
  int bad_s = 0, bad_z = 0, nonfinite = 0;
  double max_s = 0, max_z = 0;
  for (int e = 0; e < L; ++e)
    for (int g = 0; g < sk; ++g)
      for (int n = 0; n < N; ++n) {
        size_t const i = (size_t)e * per + (size_t)g * N + n;
        int const b = (g * f.gs) / f.sb;
        size_t const bi = (size_t)e * nb * N + (size_t)b * N + n;
        double const dv = double(float(f.d[bi]));
        double const dm = has_min ? double(float(f.dmin[bi])) : 0.0;
        float const ws = float(half_t(float(dv * double(int(f.sc[i])))));
        float  wz = has_min ? float(half_t(float(-dm * double(int(f.mn[i]))))) : 0.f;
        if (f.z_mul) wz = float(half_t(wz + float(f.z_mul) * ws));
        float const gs_ = float(hS[i]), gz = float(hZ[i]);
        if (!std::isfinite(gs_) || !std::isfinite(gz)) { ++nonfinite; continue; }
        if (gs_ != ws) { if (bad_s < 4) std::printf("    [scale] e=%d g=%d n=%d dev=%.9g host=%.9g (sc=%d d=%.9g)\n",
                                                    e, g, n, gs_, ws, int(f.sc[i]), dv); ++bad_s; }
        if (gz != wz)  { if (bad_z < 4) std::printf("    [zero ] e=%d g=%d n=%d dev=%.9g host=%.9g (mn=%d dmin=%.9g)\n",
                                                    e, g, n, gz, wz, has_min ? int(f.mn[i]) : 0, dm); ++bad_z; }
        max_s = std::max(max_s, std::abs(double(gs_) - double(ws)));
        max_z = std::max(max_z, std::abs(double(gz) - double(wz)));
      }
  std::printf("  device decode vs host fp16 reference: %zu groups | scale %d bad (max_abs %.3e) | zero %d bad (max_abs %.3e)\n",
              tot, bad_s, max_s, bad_z, max_z);
  // Non-finite must be counted, never compared: NaN != NaN makes every comparison "pass" and an all-NaN buffer would
  // score a clean run. This bit me three times in this codebase's harnesses.
  if (nonfinite) std::printf("  NON-FINITE outputs: %d of %zu -- the decode produced NaN/Inf, not a mismatch\n",
                             nonfinite, tot);

  // The traffic this form actually carries, so the claim is a number off the fixture rather than prose.
  double const native = double(tot) * (1.0 + (has_min ? 1.0 : 0.0)) + double((size_t)L * nb * N) * (has_min ? 4.0 : 2.0);
  double const fp16   = double(tot) * (has_min ? 4.0 : 2.0);
  std::printf("  scale channel bytes: native %.0f vs fp16 %.0f  (%.2fx smaller)\n", native, fp16, fp16 / native);

  int const fail = bad_s + bad_z + nonfinite;
  std::printf("== %s: %d ==\n", fail ? "FAIL" : "PASS", fail);
  return fail ? 1 : 0;
}
