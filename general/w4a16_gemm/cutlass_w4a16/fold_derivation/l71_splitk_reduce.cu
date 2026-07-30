// L71 -- local gate for the two pieces of grouped split-K that compute the answer, so neither has to be
// trusted on the strength of "it compiled on the box".
//
//   (A) THE LEGALITY RULE. splitk_ok() decides which S a problem admits. It is a table of divisibility
//       conditions, and a table is exactly the kind of thing that looks right and is not, so every condition
//       gets a case that TRIPS it and a neighbouring case that does not. Without the negative half, a rule
//       that accepted everything would pass.
//
//   (B) THE MERGE. moeg_splitk_reduce sums S fp16 partials in fp32. Checked against an fp64 reference over
//       the magnitudes the MoE harness actually produces, and alongside the fp16 SERIAL chain that the
//       existing dense split-K epilogue uses (measured in l70: 1 ulp at S<=4, 2 ulp at S>=8), so the claim
//       "the separate reduce is numerically better" is a measurement rather than an assertion.
//
//   nvcc -std=c++17 -arch=sm_90 -I../../../../third_party/actlize/include -Istub_inc l71_splitk_reduce.cu -o l71 && ./l71
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include "../moe_splitk_reduce.cuh"

using moe_splitk_ppu::splitk_ok;
using moe_splitk_ppu::moeg_splitk_reduce;

static int pass = 0, fail = 0;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------------
// (A) legality
struct OkCase { int k, S, gs, tk; bool want; const char* note; };

static void gate_legality() {
  // Every row names the ONE condition it exercises. The `want == true` rows are the neighbours that keep a
  // rule from being vacuously strict; the `false` rows each trip exactly one clause.
  OkCase const cs[] = {
    {2048, 1,  32,  64, true,  "S=1 always legal"},
    {2048, 2,  32,  64, true,  "clean: 1024 per slice"},
    {2048, 4,  32,  64, true,  "clean: 512 per slice"},
    {2048, 8,  32,  64, true,  "clean: 256 per slice, exactly one offline tile"},
    {2048, 16, 32,  64, false, "slice K 128 < the 256-element offline tile"},
    {2048, 3,  32,  64, false, "K not divisible by S"},
    {2048, 0,  32,  64, false, "S < 1"},
    {2048, 2,  32, 256, true,  "slice 1024 is a whole number of Block_K=256"},
    {1536, 2,  32, 256, false, "slice 768 is not a whole number of Block_K=256"},
    {2048, 8,  16,  64, true,  "gs=16 divides the 256 slice"},
    {2560, 5,  32,  64, false, "slice 512 ok but 2560/5=512 -> 512%256==0; kept legal"},
    {2048, 2, 512,  64, false, "slice 1024 ok, gs=512 divides it; legal"},
  };
  // The last two rows are deliberately self-contradicting notes -- recompute the truth instead of trusting me.
  for (auto& c : cs) {
    const char* why = "";
    bool const got = splitk_ok(c.k, c.S, c.gs, c.tk, &why);
    // independent recomputation of the rule, written out rather than called
    bool ref;
    if (c.S < 1) ref = false;
    else if (c.S == 1) ref = true;
    else {
      int ks = (c.k % c.S) ? -1 : c.k / c.S;
      ref = (ks > 0) && (ks % 256 == 0) && (c.gs <= 0 || ks % c.gs == 0) && (c.tk <= 0 || ks % c.tk == 0);
    }
    if (got == ref) { ++pass; }
    else {
      std::printf("  [FAIL] legality k=%d S=%d gs=%d tk=%d: got %d ref %d (%s) -- %s\n",
                  c.k, c.S, c.gs, c.tk, int(got), int(ref), why, c.note);
      ++fail;
    }
  }
  // Coverage assertion: the table must actually contain both verdicts, or it proves nothing.
  int t = 0, f = 0;
  for (auto& c : cs) { const char* w = ""; (splitk_ok(c.k, c.S, c.gs, c.tk, &w) ? t : f)++; }
  if (t > 0 && f > 0) { std::printf("  [ok]   legality  %zu rows, %d legal / %d refused, all match an independent recomputation\n",
                                    sizeof(cs)/sizeof(cs[0]), t, f); ++pass; }
  else { std::printf("  [FAIL] legality table is one-sided (%d legal, %d refused) -- it cannot detect a vacuous rule\n", t, f); ++fail; }
}

// ---------------------------------------------------------------------------------------------------
// (B) merge numerics
static void gate_merge(int slices, int elems, double a_mag) {
  std::mt19937 gen(0xBEEF ^ (slices * 7919 + elems));
  std::uniform_real_distribution<double> d(-a_mag, a_mag);

  std::vector<__half> h_part(size_t(slices) * elems);
  std::vector<double> ref(elems, 0.0);
  for (int s = 0; s < slices; ++s)
    for (int i = 0; i < elems; ++i) {
      __half const v = __float2half(float(d(gen)));
      h_part[size_t(s) * elems + i] = v;
      ref[i] += double(__half2float(v));          // exact sum of the values the kernel will see
    }

  __half *dP, *dD;
  CK(cudaMalloc(&dP, h_part.size() * sizeof(__half)));
  CK(cudaMalloc(&dD, size_t(elems) * sizeof(__half)));
  CK(cudaMemcpy(dP, h_part.data(), h_part.size() * sizeof(__half), cudaMemcpyHostToDevice));
  moeg_splitk_reduce<<<64, 256>>>(dD, dP, elems, slices);
  CK(cudaDeviceSynchronize());
  std::vector<__half> h_out(elems);
  CK(cudaMemcpy(h_out.data(), dD, size_t(elems) * sizeof(__half), cudaMemcpyDeviceToHost));

  // The fp32 accumulation is correctly rounded once, so the result must equal fp16(exact sum) EXACTLY --
  // not "within a tolerance". Anything else means the kernel is not doing what it claims.
  int bad_exact = 0;
  double worst_abs = 0.0;
  double max_ref = 1e-30;
  for (int i = 0; i < elems; ++i) max_ref = std::max(max_ref, std::fabs(ref[i]));
  for (int i = 0; i < elems; ++i) {
    float const got = __half2float(h_out[i]);
    float const want = __half2float(__float2half(float(ref[i])));
    if (got != want) {
      ++bad_exact;
      worst_abs = std::max(worst_abs, double(std::fabs(got - want)));
    }
  }

  // For comparison: the SERIAL fp16 chain the dense split-K epilogue uses, prev = fp16(prev + partial).
  int chain_bad = 0; double chain_worst_abs = 0.0;
  for (int i = 0; i < elems; ++i) {
    float acc = 0.f;
    for (int s = 0; s < slices; ++s) acc = __half2float(__float2half(acc + __half2float(h_part[size_t(s) * elems + i])));
    float const want = __half2float(__float2half(float(ref[i])));
    if (acc != want) {
      ++chain_bad;
      chain_worst_abs = std::max(chain_worst_abs, double(std::fabs(acc - want)));
    }
  }

  bool const ok = (bad_exact == 0);
  // ULP IS THE WRONG METRIC HERE and reporting it would be misleading: independent partials cancel, so the
  // reference is sometimes near zero and the ulp ratio explodes on elements whose ABSOLUTE error is tiny. What
  // is meaningful is (a) the fp32 reduce reproduces fp16(exact sum) with ZERO mismatches, and (b) how far the
  // serial fp16 chain drifts relative to the largest output in the array.
  std::printf("  %s merge     S=%-3d elems=%-6d |A|<=%.3g : fp32 reduce %d bad (max abs err %.3g)"
              " | fp16 serial chain %d bad = %4.1f%% of elements, max rel err %.2e\n",
              ok ? "[ok]  " : "[FAIL]", slices, elems, a_mag, bad_exact, worst_abs,
              chain_bad, 100.0 * chain_bad / elems, chain_worst_abs / max_ref);
  ok ? ++pass : ++fail;
  CK(cudaFree(dP)); CK(cudaFree(dD));
}

int main() {
  std::printf("== (A) split-K legality rule ==\n");
  gate_legality();
  std::printf("\n== (B) merge: fp32 reduce vs the fp16 serial chain the dense epilogue uses ==\n");
  // magnitudes the MoE harness produces: A ~ 0.01, scale ~ 0.0625, K=2048 -> partials of order 1
  for (int s : {2, 4, 8, 16, 32}) gate_merge(s, 1 << 14, 1.0);
  gate_merge(8, 1 << 14, 64.0);    // larger partials: fp16 spacing coarsens, chain should degrade further
  std::printf("\n== summary ==\n  %d passed, %d failed\n", pass, fail);
  return fail == 0 ? 0 : 1;
}
