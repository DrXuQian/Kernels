// Plan #20: LOAD THE NATIVE scale/zero. New file; the fp16 path is untouched.
//
// FLAT, next to the harnesses, not under real_weight/. build.sh overlays the harness directory's *.hpp into the actlize
// example dir but only descends into the ONE subdirectory listed in _overlay_dirs (gemv_lowbit), so a compiled header
// parked in real_weight/ vanishes at build time -- which is exactly what happened. real_weight/ holds fixtures and the
// python that writes them; anything the box compiles belongs up here.
//
// Today every consumer loads pre-multiplied fp16 planes: the offline computes d*sc and -dmin*mn + 8*scale in fp32 and
// rounds each to one fp16, so the kernel never sees what the gguf actually stores. RWMOEP (dump_packed_scale.py) keeps
// the native form instead -- int8 codes plus the superblock's fp16 -- and this header is its C++ side:
//
//   char[8] "RWMOEP\0\0"
//   i32 x10 L, M, N, K, gs, mode, ktype, sb, z_mul, cvt_bias
//   f16 A  [L][M][K]        i8 q_signed [L][N][K]  (transposed, as RWMOE)
//   i8  sc [L][scale_k][N]  i8 mn [L][scale_k][N]   (mn only when mode == 1)
//   f16 d  [L][K/sb][N]     f16 dmin [L][K/sb][N]   (dmin only when mode == 1)
//   f16 gold [L][M][N]
//
// and the DEVICE decode, which is the same call the mainloop will make once the scale tile holds bytes:
//     scale = d[k/sb][n] * sc[k/gs][n]        zero = -dmin[k/sb][n] * mn[k/gs][n] + z_mul * scale
// z_mul comes from the file, so the fixture -- not this header -- decides whether the converter's -8 is cancelled here
// or left to the converter (both measured equivalent, see the plan; the offline currently writes z_mul = 0).
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "cutlass/half.h"
#include "gguf_scale_decode.hpp"

namespace rwmoep {

using cutlass::half_t;

struct File {
  int L = 0, M = 0, N = 0, K = 0, gs = 0, mode = 0, ktype = 0, sb = 0, z_mul = 0, cvt_bias = 0;
  std::vector<half_t> A, d, dmin, gold;
  std::vector<int8_t> q, sc, mn;

  int scale_k() const { return K / gs; }
  int nsb()     const { return K / sb; }

  template <class T> bool rd(std::FILE* f, std::vector<T>& v, size_t n) {
    v.resize(n);
    return n == 0 || std::fread(v.data(), sizeof(T), n, f) == n;
  }

  // Returns false with a printed reason rather than reading a short file as zeros -- a silently truncated plane would
  // pass a comparison against a golden that was also read short.
  bool load(char const* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("[rwmoep] cannot open %s\n", path); return false; }
    char magic[8];
    int  h[10];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "RWMOEP\0\0", 8) != 0) {
      std::printf("[rwmoep] %s is not RWMOEP (an RWMOE fixture has fp16 scale/zero -- wrong file)\n", path);
      std::fclose(f); return false;
    }
    if (std::fread(h, 4, 10, f) != 10) { std::printf("[rwmoep] short header\n"); std::fclose(f); return false; }
    L = h[0]; M = h[1]; N = h[2]; K = h[3]; gs = h[4]; mode = h[5];
    ktype = h[6]; sb = h[7]; z_mul = h[8]; cvt_bias = h[9];
    if (gs <= 0 || sb <= 0 || K % gs || K % sb) {
      std::printf("[rwmoep] bad geometry L=%d M=%d N=%d K=%d gs=%d sb=%d\n", L, M, N, K, gs, sb);
      std::fclose(f); return false;
    }
    size_t const sk = (size_t)scale_k(), nb = (size_t)nsb();
    bool ok = rd(f, A, (size_t)L * M * K)
           && rd(f, q, (size_t)L * N * K)
           && rd(f, sc, (size_t)L * sk * N)
           && (mode != 1 || rd(f, mn, (size_t)L * sk * N))
           && rd(f, d, (size_t)L * nb * N)
           && (mode != 1 || rd(f, dmin, (size_t)L * nb * N))
           && rd(f, gold, (size_t)L * M * N);
    // Trailing bytes mean the writer and this reader disagree about the layout, which is exactly the failure a
    // header-only check cannot see.
    long extra = 0;
    if (ok) { long here = std::ftell(f); std::fseek(f, 0, SEEK_END); extra = std::ftell(f) - here; }
    std::fclose(f);
    if (!ok)    { std::printf("[rwmoep] truncated planes\n"); return false; }
    if (extra)  { std::printf("[rwmoep] %ld trailing bytes -- layout mismatch with the writer\n", extra); return false; }
    std::printf("[rwmoep] %s: L=%d M=%d N=%d K=%d gs=%d mode=%d ktype=%d sb=%d z_mul=%d cvt_bias=%d\n",
                path, L, M, N, K, gs, mode, ktype, sb, z_mul, cvt_bias);
    return true;
  }
};

// THE DEVICE DECODE. One thread per (expert, group, n): exactly the arithmetic the mainloop will do per (n, group),
// through the same gguf_scale_decode.hpp entry points, so what passes here is what the mainloop inherits.
__global__ void decode_planes(int8_t const* sc, int8_t const* mn, half_t const* d, half_t const* dmin,
                              half_t* scale_out, half_t* zero_out,
                              int L, int N, int scale_k, int nsb, int gs, int sb, int has_min, int z_mul) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  int const per = scale_k * N;
  if (i >= L * per) return;
  int const e = i / per, r = i % per, g = r / N, n = r % N;
  int const b = (g * gs) / sb;                                  // which superblock this group belongs to
  half_t const dv  = d[(size_t)e * nsb * N + (size_t)b * N + n];
  half_t const dmv = has_min ? dmin[(size_t)e * nsb * N + (size_t)b * N + n] : half_t(0.f);
  int const sci = int(sc[i]);
  int const mni = has_min ? int(mn[i]) : 0;
  half_t s = dv * gguf_scale::int_to_half_small(sci);            // the file's codes are already centre-free (Q4_K)
  half_t z = has_min ? -(dmv * gguf_scale::int_to_half_small(mni)) : half_t(0.f);
  if (z_mul) z = z + half_t(float(z_mul)) * s;
  scale_out[i] = s;
  zero_out[i]  = z;
}

}  // namespace rwmoep
