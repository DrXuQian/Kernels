// L70 -- SERIAL SPLIT-K's NUMERICAL COST, answered on the host before any kernel is written.
//
// ppu_aiu_gemm_mixed_input_splitk_serial.hpp merges slices through the EPILOGUE with beta: slice s reads D (fp16),
// adds its fp32 accumulator, and writes fp16 back. So S slices cost S fp16 roundings on a value that is a sum of K
// products, where a single-slice kernel costs exactly one. That is a real accuracy change and it is the one split-K
// risk that needs no hardware to settle -- so settle it first, because if S=4 already breaks the tolerance the rest
// of the work is pointless.
//
// The reference is fp32 accumulation of the whole K, which is what the current kernel does (ElementAccumulator=float).
// Magnitudes are the real ones: int4 codes in [-8,7], fp16 scales around 2^-4, fp16 activations around 1e-2 -- taken
// from what test_lowbit_moe_bench actually fills (A = 0.01, scale = 0.0625).
//
//   nvcc -std=c++17 -Istub_inc -I.. -I../../../../third_party/actlize/include l70_splitk_fp16_merge.cu -o l70 && ./l70
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
using half_t = cutlass::half_t;

// one merge step exactly as the epilogue does it: fp16(prev_fp16 + fp32_partial)
static inline float merge(float prev_stored, float partial) {
  return float(half_t(prev_stored + partial));
}

int main() {
  const int K = 2048, gs = 32, TRIALS = 4096;
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> q(-8, 7);

  std::printf("L70 -- serial split-K fp16 merge error, K=%d gs=%d, %d trials\n", K, gs, TRIALS);
  std::printf("  reference = fp32 accumulate over all K, then ONE fp16 store (what split_k=1 does)\n\n");
  // fp16's own eps is 2^-11 = 4.88e-4, so the meaningful unit is ULP, not a bare relative number: a few ulp is what
  // extra roundings must cost, and anything much larger means the merge is losing real information.
  std::printf("  fp16 eps = 2^-11 = 4.88e-04, so read the ulp column: S-1 merges should cost O(S) ulp, not more\n\n");
  std::printf("  %-6s %-14s %-14s %-14s\n", "S", "max |rel|", "mean |rel|", "max |ulp16|");

  int bad = 0;
  for (int S : {1, 2, 4, 8, 16, 32}) {
    double worst = 0.0, sum = 0.0; double worst_ulp = 0.0;
    for (int t = 0; t < TRIALS; ++t) {
      std::vector<float> a(K), w(K);
      for (int i = 0; i < K; ++i) {
        a[i] = float(half_t(0.01f * (1.0f + 0.5f * float(q(rng)) / 8.0f)));   // activations ~1e-2
        const float sc = float(half_t(0.0625f));                              // fp16 scale, as the harness fills
        w[i] = sc * float(q(rng));                                            // dequantised int4 weight
      }
      double ref = 0.0;                            // fp32 accumulate, one fp16 store
      for (int i = 0; i < K; ++i) ref += double(a[i]) * double(w[i]);
      const float ref16 = float(half_t(float(ref)));

      // K TILES are what a slice owns, and the slices are STRIDED by S (k_step = gridDim.z), not contiguous --
      // read off make_splitk_coord_iterator(shape, start_k=k_coord, k_step=gridDim.z). The grouping matters for the
      // per-slice partial magnitude, so model the real one rather than a contiguous split.
      const int TK = 256, ktiles = K / TK;
      float stored = 0.0f;
      for (int s = 0; s < S; ++s) {
        float part = 0.0f;                          // fp32 within a slice
        for (int tile = s; tile < ktiles; tile += S)
          for (int i = tile * TK; i < (tile + 1) * TK; ++i) part += a[i] * w[i];
        stored = (s == 0) ? float(half_t(part)) : merge(stored, part);
      }
      const double rel = std::fabs(double(stored) - double(ref16)) / std::max(1e-30, std::fabs(double(ref16)));
      // ULP OF AN fp16, computed from the exponent. The first version used std::nextafter on a FLOAT and converted
      // back to half -- which returns the same half, so the denominator was ~0 and the column printed 9.8e26. fp16 has
      // a 10-bit stored mantissa, so ulp(v) = 2^(ilogb(v) - 10).
      const double ulp16 = (ref16 == 0.0f) ? 1.0 : std::ldexp(1.0, std::ilogb(std::fabs(ref16)) - 10);
      const double ulp = std::fabs(double(stored) - double(ref16)) / ulp16;
      worst = std::max(worst, rel); sum += rel; worst_ulp = std::max(worst_ulp, ulp);
    }
    const double mean = sum / TRIALS;
    std::printf("  %-6d %-14.3e %-14.3e %-14.1f%s\n", S, worst, mean, worst_ulp,
                worst > 5e-3 ? "   <-- EXCEEDS 5e-3" : "");
    if (worst > 5e-3) ++bad;
  }
  std::printf("\n  %s\n", bad ? "some S exceed 5e-3 relative -- serial fp16 merge is NOT free, cap S or accumulate in fp32"
                              : "every S within 5e-3 relative -- the serial fp16 merge is acceptable at these magnitudes");
  std::printf("  CONSEQUENCE FOR THE GATE: split-K is NOT bit-identical, so test_lowbit_grouped's exact-match oracle\n");
  std::printf("  (max_rel == 0) CANNOT be used on it. A split-K row needs a tolerance, and the number above is what to set.\n");
  return bad ? 1 : 0;
}
