/*
 * Classic Marlin FP16xINT4 GEMM (Elias Frantar, Apache-2.0) ported to the PPU (ppu001).
 * The original NVIDIA kernel is kept VERBATIM except for the PPU-specific spots (search "PPU:"):
 *   (1) mma()  -> mma_n16(): two NVIDIA m16n8 MMAs fuse into one ppu.mma.m16n16 (B={b0,b1}, 8-float acc).
 *   (2) write_result: the C fragment layout differs on PPU -> acc_i/acc_j drive the smem staging indices; the
 *       coalesced int4 read-out to global is layout-agnostic (-DMARLIN_WRITE_DIRECT keeps the old per-half store).
 *   (3) global_reduce: same C-layout issue; FIXME, only used on split-K.
 * thread_block_reduce is UNCHANGED (register-index-aligned across K-warps -> correct on PPU too).
 * ppu_ldmatrix_x4 is kept (A operand layout == NVIDIA; if PPU rejects plain ldmatrix.x4, swap to element-wise frag_i/frag_j).
 * Build: box only, ppu001 (no -arch).
 */
#ifndef MARLIN_GGUF_PPU_CUH
#define MARLIN_GGUF_PPU_CUH

// -DMARLIN_MINS_NOAPPLY: RESULT IS WRONG ON PURPOSE. Keeps every byte of the mins path -- the extra
// cp.async per stage, the extra shared int4 per k, the registers -- and drops ONLY the hfma2, falling back
// to the symmetric hmul2. It brackets what the affine min actually costs:
//   time returns to the sym line -> the cost is the ARITHMETIC (and the header's "free, hmul2 -> hfma2 is
//                                   the same instruction count" claim is wrong about more than count)
//   time stays at the aff line   -> the cost is MOVING the mins, and trimming the math cannot touch it
// Measured affine cost to explain: -3.8 to -4.5 MFU points at gs=128, -10.6 to -11.3 at gs=32. Same shared
// read per k in both; what differs is cp.async per stage, 1 against thread_k_blocks/2.
// Same convention as MARLIN_SCALE_NONE / GEMV_KTPG -- isolate one half, do not guess which half.
#ifdef MARLIN_MINS_NOAPPLY
#define MARLIN_APPLY_MIN 0
#else
#define MARLIN_APPLY_MIN 1
#endif

// -DMARLIN_MINS_CONST: the THIRD point, and the reason it is needed. MARLIN_MINS_NOAPPLY does not keep the
// mins path alive as its comment claimed: with frag_m unread, ptxas dead-code-eliminates the per-k shared
// int4 load, so that probe drops the LOAD and the hfma2 together and cannot separate them. This one keeps
// the hfma2 and feeds it a compile-time constant instead, dropping only the load. Three points then read:
//   real (load + hfma2)            29.9% MFU at gs=32
//   NOAPPLY (neither, via DCE)     40.2%
//   CONST (hfma2, no load)         -> near 29.9 means the ARITHMETIC costs; near 40.2 means the LOAD does
// Wrong results by design, timing only.
#ifdef MARLIN_MINS_CONST
#define MARLIN_MIN_VAL(fm) __float2half2_rn(0.25f)
#else
#define MARLIN_MIN_VAL(fm) __half2half2(reinterpret_cast<__half*>(&(fm))[i])
#endif

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>   // getenv / atoi for the MARLIN_* runtime knobs
#include <iostream>
#include <vector>

namespace marlin_gguf_ppu {

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

template <typename T, int n>
struct Vec { T elems[n]; __device__ __forceinline__ T& operator[](int i) { return elems[i]; } };

using I4 = Vec<int, 4>;
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
// PPU: ONE FragC per n16 mma = the 8 accumulator floats in the instruction's NATIVE order (d0..d7). NVIDIA splits
// them into two 4-float n8 accs; keeping that split forced a d[8] repack around every mma. Native order also means
// float l maps straight to (acc_i(lane,l), acc_j(lane,l)) -- no reassembly in write_result/global_reduce.
using FragC = Vec<float, 8>;
using FragS = Vec<half2, 1>;

// PPU: A/B operand and C fragment (lane,l) -> (row,col), verified in test_marlin_n8x2_to_n16.cu.
__device__ __forceinline__ int frag_i(int lane, int l) { return (lane / 4) + (l / 2) * 8; }   // A outer (m), B outer (n)
__device__ __forceinline__ int frag_j(int lane, int l) { return (lane % 4) + (l % 2) * 4; }   // A/B k half2-col
__device__ __forceinline__ int acc_i(int lane, int l) { return (lane / 4) + (((l >> 2) & 1) << 3); }
__device__ __forceinline__ int acc_j(int lane, int l) { return (lane % 4) + ((l % 4) << 2); }

// PPU: no L2 cache-hint eviction policy asm -> use a plain cp.async.cg (correctness; L2 policy is a perf tweak).
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES));
}
// PPU: the classic guards the A copy with a PREDICATED cp.async (".reg .pred p; setp; @p cp.async"). PPU lowers
// cp.async to an intrinsic (as it does ppu.mma), and an inline-asm @p cannot predicate an intrinsic -- the copy can
// fall back to a slow compat expansion (ld+st). Plain unpredicated cp.async is known-good on PPU, so guard with a
// BRANCH. Semantically identical: a thread that issues no copy still commits an empty group, so the cp.async group
// stack stays consistent -- the same pattern the scale path and the AIU bulk loads already use.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  if (pred) cp_async4(smem_ptr, glob_ptr);
}
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) { cp_async4(smem_ptr, glob_ptr); }
__device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ inline void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

// PPU: two NVIDIA m16n8 MMAs (b0=cols0-7, b1=cols8-15) fused into ONE ppu.mma.m16n16k16. The 8 acc floats bind
// DIRECTLY to the asm "+f" operands -- no d[8] repack, no FragC& pair. That repack (plus the & taken on frag_c in
// the reduces) is what stopped hggc from promoting frag_c into registers, leaving a 516-byte stack frame and 8
// local ld + 8 local st per mma. Acc float l lives at (acc_i(lane,l), acc_j(lane,l)); l in {0,1,4,5} are the b0
// columns 0-7 and {2,3,6,7} the b1 columns 8-15 (verified in ppu_tests/test_marlin_n8x2_to_n16.cu).
__device__ __forceinline__ void mma_n16(const FragA& a, const FragB& b0, const FragB& b1, FragC& c) {
  const unsigned* A = reinterpret_cast<const unsigned*>(&a);
  unsigned B[4] = { *reinterpret_cast<const unsigned*>(&b0.elems[0]), *reinterpret_cast<const unsigned*>(&b0.elems[1]),
                    *reinterpret_cast<const unsigned*>(&b1.elems[0]), *reinterpret_cast<const unsigned*>(&b1.elems[1]) };
  asm volatile(
    "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3,%4,%5,%6,%7};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]));
}

// PPU ldmatrix (non-swizzle form -- reads ordinary cp.async-filled shared memory, no AIU cube descriptor):
//   ppu.ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type  r, [p];
//   .shape={.m8n8,.m16n16}  .num={.x1,.x2,.x4}  .ss={.shared{::cta}}  .type={.b16}   (.m16n16 requires .trans)
// REGISTER ORDER: ldmatrix.x4 returns the four 8x8 matrices in the order its 32 lane addresses named them. With
// Marlin's a_sh_rd (lanes 0-7 -> rows 0-7/k0, 8-15 -> rows 8-15/k0, 16-23 -> rows 0-7/k8, 24-31 -> rows 8-15/k8):
//     v0=(r0-7,k0-7)  v1=(r8-15,k0-7)  v2=(r0-7,k8-15)  v3=(r8-15,k8-15)
// But the PPU mma's A operand (frag_i = lane/4 + (l/2)*8, frag_j = lane%4 + (l%2)*4) wants
//     a0=(r0-7,k0-7)  a1=(r0-7,k8-15)  a2=(r8-15,k0-7)  a3=(r8-15,k8-15)
// i.e. **v1 and v2 are transposed**. "PPU A/B fragment layout == NVIDIA" holds for the SET of (row,col) a lane owns
// (lane0 k={0,1,8,9}), NOT for the register ORDER. -DMARLIN_A_LDSM_NOSWAP drops the swap to A/B this.
// Notes: the `ppu.` prefix is a NO-OP for ldmatrix (bare and prefixed lower to the same instruction and gave
// bit-identical results); so is "l" vs a cvta'd "r" address. -DMARLIN_A_LDSM_NV keeps the bare mnemonic.
// The swap is FREE: bind the asm's 2nd/3rd outputs straight to a[2]/a[1]. No temporaries, no movs.
__device__ __forceinline__ void ppu_ldmatrix_x4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
#ifdef MARLIN_A_LDSM_NV
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];\n"
#else
  asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
#endif
#ifdef MARLIN_A_LDSM_NOSWAP
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "l"(smem_ptr));
#else
    : "=r"(a[0]), "=r"(a[2]), "=r"(a[1]), "=r"(a[3]) : "l"(smem_ptr));   // v1 <-> v2
#endif
}

// PPU tensor-core operand loads (from ppu_tests/fa_ppu.cu). A goes via AIU bulk load -> swzl TSM, then ld_swzl gives
// the 4-uint32 A fragment in ONE instruction (vs 8 scalar loads + make_half2 element-wise) -> cuts I-Fetch/Mem-Dep +
// registers. cube_w MUST be 64 (b16 128B swzl).
__device__ __forceinline__ unsigned cvta_s(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void aiu_load(unsigned smem, const void* gmem,
        int dim_h, int dim_w, int cube_h, int cube_w, int coord_h, int coord_w) {
  asm volatile(
    "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], {%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
    :: "r"(smem), "l"(gmem), "r"(1), "r"(dim_h), "r"(dim_w), "r"(0), "r"(coord_h), "r"(coord_w), "r"(1), "r"(cube_h), "r"(cube_w), "r"(1));
}
__device__ __forceinline__ void ld_swzl(uint32_t* v, const void* cube_base, int coord_h, int coord_w, int cube_h) {
  const int cbo = coord_w * (int) sizeof(__half);
  asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};\n"
    : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
    : "l"(cube_base), "r"(0), "r"(coord_h), "r"(1), "r"(cube_h), "r"(1), "r"(cbo));
}

template <int lut> __device__ inline int lop3(int a, int b, int c) {
  int res; asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)); return res;
}
__device__ __forceinline__ FragB dequant(int q) {
  const int LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX), hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64086408, MUL = 0x2c002c00, ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}
// DO NOT fold this multiply into dequant's constants. It is the obvious optimization and it is NUMERICALLY WRONG.
//
// Grouped scales (groupsize=128) cost 130 us of the 581 us at 4096^3 -- 24 FALU ops per matmul call (8 __half2half2 +
// 16 __hmul2), 50.3M warp-instructions, and they flip the kernel from tensor-bound to issue-bound. The tempting fix is
// to fold s into dequant's magic constants, since s is constant within a group:
//     (lo - SUB) * s      -> __hfma2(lo, s2, -SUB*s2)
//     (hi*MUL + ADD) * s  -> __hfma2(hi, MUL*s2, ADD*s2)
// That would delete all 24 ops. It also destroys the lop3 trick.
//
// The trick works because lo = 1024 + q and SUB = 1032 are BOTH exactly representable in fp16 (spacing is exactly 1.0
// at that magnitude), so `lo - SUB` gives q-8 with ZERO error. Fold s in and you instead round 1024*s and 1032*s
// separately (each off by up to half an ULP = 0.5) and then subtract them to get a result of magnitude ~10. Textbook
// catastrophic cancellation. Measured over 20k random (q, s in [0.5,1.5)):
//
//   current (exact hsub2, then hmul2)         median 0        p99 4.6e-04   max 4.9e-04
//   fold v[0]: fma(lo, s, -SUB*s)             median 4.3e-02  p99 3.7e-01   max 5.0e-01
//   fold v[1]: fma(hi, MUL*s, ADD*s)          median 2.9e-03  p99 2.5e-02   max 3.5e-02
//
// Note the trap in that last row: folding only v[1] gives a MEDIAN error of 2.9e-03, which sails through the 3e-2
// threshold in test_marlin_classic_group -- while the p99 and the max are already over it. It would report MATCH and
// quietly degrade the model.
//
// So the 16 __hmul2 are the floor: the scale must be applied AFTER the exact subtraction. The 8 __half2half2 are
// removable (have the host emit each scale pre-broadcast as a half2), worth ~41 us / +8.7%, at the cost of a scale
// format change. The 8 scalar shared loads were worth 6.6 us, and THAT one is now taken: the permuted layout turned
// out to be upstream's own _scale_perm, so it cost nothing to adopt -- see fetch_to_registers.
__device__ __forceinline__ void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s); frag_b[1] = __hmul2(frag_b[1], s);
}

// AFFINE dequant (Q4_1, Q4_K): w = q*s + m. dequant() already produces (q-8), and
//     q*s + m == (q-8)*s + (m + 8s)
// so folding 8s into the min ON THE HOST (m' = m + 8s) lets the affine formats reuse the symmetric
// dequant UNCHANGED -- the only kernel cost is hmul2 -> hfma2, which is the SAME instruction count.
// That is the big structural advantage of the fp16 (W4A16) path over int8 (W4A8) for these formats:
// in int8 the min cannot be folded into the mma at all (the tensor core sees raw integers), so it needs
// a separate rank-1 correction AND an int32-accumulator flush at every scale boundary. Here it is free.
__device__ __forceinline__ void scale_affine(FragB& frag_b, FragS& frag_s, FragS& frag_m, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  half2 m = MARLIN_MIN_VAL(frag_m);
  frag_b[0] = __hfma2(frag_b[0], s, m); frag_b[1] = __hfma2(frag_b[1], s, m);
}
// HOST: the grouped-scale layout the kernel expects by default. This is upstream Marlin's _scale_perm,
// bit for bit -- an 8x8 transpose inside every 64-column chunk of each group row:
//     out[64c + 8i + j] = plain[64c + i + 8j]
// It falls out of the B-fragment column layout (see fetch_to_registers), which PPU shares with NVIDIA,
// which is why a vLLM / GPTQ-Marlin checkpoint's scales need NO conversion -- they are already like this.
// You only need this function when you hold RAW [num_groups][prob_n] scales (e.g. straight off a GPTQ
// checkpoint). The alternative is -DMARLIN_SCALE_PLAIN, which reads that layout directly but costs 8
// scalar half loads per k instead of one int4.
//
// ONE definition, shared by the kernel's tests and by any integration -- so the two cannot drift apart.
// Requires prob_n % 64 == 0. Only for groupsize != -1 (gs == -1 scales are applied in write_result and
// are indexed by column there; they are NOT permuted).
inline void marlin_permute_scales(const half* plain, half* out, int num_groups, int prob_n) {
  for (int g = 0; g < num_groups; g++)
    for (int c0 = 0; c0 < prob_n; c0 += 64)
      for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
          out[(size_t) g * prob_n + c0 + 8 * i + j] = plain[(size_t) g * prob_n + c0 + i + 8 * j];
}

// ============================================================================================
// HOST: GGUF Q4_K -> this kernel's inputs (the "reorder" that makes the vectorized 16 B shared load
// possible). Marlin's ktile-major per-lane B layout exists precisely so ONE int4 load gives a lane
// every word it needs; GGUF's native layout scatters them, which is what costs ~4x on the PPU.
//
// Q4_K is affine and two-level: w = (d*sc_b)*q - dmin*m_b, per 32-value sub-block b of a 256 super-block,
// with sc_b/m_b 6-bit packed in 12 bytes. Both levels are PRE-EXPANDED here into per-32 fp16 pairs:
//     s  = d * sc_b
//     m' = -dmin * m_b + 8*s          <- the +8s folds the (q-8) of the symmetric dequant, so the KERNEL
//                                        needs no new dequant: it just does hfma2(q-8, s, m') == q*s + m.
// Weights stay 4-bit; only the scale metadata grows (12 B per 256 -> 2 fp16 per 32), which is small.
// ============================================================================================
struct gguf_block_q4_K { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; };
static_assert(sizeof(gguf_block_q4_K) == 144, "q4_K");

// The 12 packed bytes -> 8 six-bit scales + 8 six-bit mins (llama.cpp's get_scale_min_k4).
inline void gguf_q4k_scales(const uint8_t* q, uint8_t* sc, uint8_t* mn) {
  for (int g = 0; g < 4; g++) { sc[g] = q[g] & 63; mn[g] = q[g + 4] & 63; }
  for (int g = 4; g < 8; g++) {
    sc[g] = (uint8_t)((q[g + 4] & 0xF) | ((q[g - 4] >> 6) << 4));
    mn[g] = (uint8_t)((q[g + 4] >> 4)  | ((q[g - 0] >> 6) << 4));
  }
}

// src: [prob_n][prob_k/256] Q4_K blocks (row-major over n). Outputs are ready for marlin_cuda(groupsize=32).
//   B_out : (prob_k/16) * (prob_n*16/32) ints  -- Marlin ktile-major per-lane layout
//   s_out : (prob_k/32) * prob_n halves        -- _scale_perm'd
//   m_out : (prob_k/32) * prob_n halves        -- _scale_perm'd
inline void marlin_repack_q4k(const gguf_block_q4_K* src, int prob_n, int prob_k,
                              int* B_out, half* s_out, half* m_out) {
  const int NG = prob_k / 32, nblk = prob_k / 256;
  std::vector<half> sp((size_t)NG * prob_n), mp((size_t)NG * prob_n);

  // nibble(n,k): sub-block b = (k%256)/32, value j = k%32; Q4_K stores value j of sub-block b in
  // qs[(b/2)*32 + j], low nibble for even b and high for odd b.
  auto nib = [&](int n, int k) -> int {
    const gguf_block_q4_K& blk = src[(size_t)n * nblk + k / 256];
    const int b = (k % 256) / 32, j = k % 32, byte = (b / 2) * 32 + j;
    return (b % 2 == 0) ? (blk.qs[byte] & 0xF) : (blk.qs[byte] >> 4);
  };

  for (int n = 0; n < prob_n; n++)
    for (int blk = 0; blk < nblk; blk++) {
      const gguf_block_q4_K& B = src[(size_t)n * nblk + blk];
      uint8_t sc[8], mn[8]; gguf_q4k_scales(B.scales, sc, mn);
      const float d = __half2float(B.d), dmin = __half2float(B.dmin);
      for (int b = 0; b < 8; b++) {
        const float sv = d * sc[b];
        sp[(size_t)(blk * 8 + b) * prob_n + n] = __float2half(sv);
        mp[(size_t)(blk * 8 + b) * prob_n + n] = __float2half(-dmin * mn[b] + 8.0f * sv);
      }
    }
  marlin_permute_scales(sp.data(), s_out, NG, prob_n);
  marlin_permute_scales(mp.data(), m_out, NG, prob_n);

  // Marlin B packing (identical to the classic layout; a pure nibble permutation, format-agnostic).
  for (int ktile = 0; ktile < prob_k / 16; ktile++)
    for (int nb = 0; nb < prob_n / 16; nb++)
      for (int lane = 0; lane < 32; lane++) {
        const int n = nb * 16 + lane / 4, kb = ktile * 16 + (lane % 4) * 2;
        int q = 0;
        q |= nib(n, kb)         << 0;  q |= nib(n, kb + 1)     << 16;
        q |= nib(n, kb + 8)     << 4;  q |= nib(n, kb + 9)     << 20;
        q |= nib(n + 8, kb)     << 8;  q |= nib(n + 8, kb + 1) << 24;
        q |= nib(n + 8, kb + 8) << 12; q |= nib(n + 8, kb + 9) << 28;
        const size_t idx = (size_t)(prob_n / 2) * ktile + (nb / 4) * 32 + lane;
        B_out[idx * 4 + (nb % 4)] = q;
      }
}


// HOST: gs == -1 (per-column scales). THIS ONE IS THE OPPOSITE CASE from the grouped scales above, and
// the difference is worth stating because it is the whole reason the two behave differently.
//
// A per-column scale is CONSTANT along K, so it is applied ONCE after the K reduction -- in write_result,
// against frag_c. That means its layout is dictated by the C (accumulator) fragment, and PPU's C layout is
// genuinely NOT NVIDIA's (NVIDIA gives a lane two ADJACENT columns {0,1}; PPU gives four stride-4 columns
// {0,4,8,12}). So upstream's _scale_perm_single does NOT carry over, and our write_result instead reads
// sh_s BY COLUMN -- i.e. this kernel wants PLAIN [1][prob_n] scales for gs == -1.
//
// (The grouped scale is the opposite: it VARIES along K, so it must be applied inside the K-loop, before
// the accumulation -- onto the B fragment. Its layout is therefore dictated by B, which PPU shares with
// NVIDIA bit for bit, which is why upstream's _scale_perm DOES carry over. Upstream needing two separate
// tables at all is the proof that the two scales hang off two different fragments.)
//
// So: raw GPTQ per-column scales are already plain and need nothing. Only a checkpoint that upstream
// already packed with _scale_perm_single needs undoing -- and that permutation is an INVOLUTION, so
// undoing it is applying it again. Writing p = 8i + 2a + b, it maps to 2i + 8a + b: a 4x4 transpose of
// the (i, a) digits, hence self-inverse. Chunk width is 32 columns here, NOT the 64 of _scale_perm.
// Verified by round-trip against upstream's table.
inline void marlin_unpermute_scales_single(const half* packed, half* out, int prob_n) {
  for (int c0 = 0; c0 < prob_n; c0 += 32)
    for (int p = 0; p < 32; p++) {
      const int q = 2 * (p / 8) + 8 * ((p % 8) / 2) + (p % 2);   // == _scale_perm_single[p], self-inverse
      out[c0 + q] = packed[c0 + p];
    }
}

__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) { int state = -1;
    do asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock)); while (state != count); }
  __syncthreads();
}
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) { lock[0] = 0; return; }
    int val = 1; asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val)); }
}

// ============================================================================
// The classic Marlin kernel, VERBATIM except the PPU spots (search "PPU:").
// ============================================================================
// PPU: give the compiler room for frag_c (thread_m_blocks*4*8 floats = 128 for the (4,16,4) prefill config). The
// per-thread register cap is 256 either way (131072/CU), so minBlocks=2 costs no register headroom vs 1 while still
// asking for 2 blocks/CU -- and (4,16,4) needs 66KB shared, so 2 blocks fit. Symptom when frag_c does not stay in
// registers: ptxas reports a ~516-byte stack frame with 0 spilled VReg, and asys shows vmem.ld/vmem.st ~8.1 per
// v.mma (ppu.mma.m16n16k16 has exactly 8 acc floats) -- 68M local accesses vs 4.2M mma.
// MIN_BLOCKS=2 pins theoretical occupancy at 25% (2 blocks x 8 warps; 131072/(2*256) = the 256-reg/thread hw cap),
// and that is the RIGHT choice: this kernel is ILP-bound behind a 4-stage cp.async pipeline, not latency-bound.
// Swept on 2048x4096x14336: MIN_BLOCKS=2 -> 292.5 TFLOP/s (0 stack, 0 spill); =3 -> 167.6 (42-132 spilled VReg);
// =4 -> 164.0 (1328 spilled VReg). Buying occupancy with real register spills is a large net loss here.
// (Shared is not the limit: (2,16,4) uses 50KB of the 256KB/CU, i.e. 5 blocks' worth.)
#ifndef MARLIN_MIN_BLOCKS
#define MARLIN_MIN_BLOCKS 2
#endif
template <const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const int stages, const int group_blocks = -1,
          const bool AFFINE = false>   // AFFINE: w = q*s + m' (Q4_K/Q4_1); false = symmetric (q-8)*s
__global__ void __launch_bounds__(threads, MARLIN_MIN_BLOCKS)
Marlin(const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C,
       const int4* __restrict__ s, const int4* __restrict__ mn, int prob_m, int prob_n, int prob_k,
       int* locks) {
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) { parallel = prob_m / (16 * thread_m_blocks); prob_m = 16 * thread_m_blocks; }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
  // GGUF gs=32 (group_blocks==2 < thread_k_blocks): the group is SMALLER than the k-tile, so every tile
  // already spans a whole number of groups -- no iter rounding, and the group/tile division below is
  // replaced by the in-tile per-32 scale path. Only the coarse-group case (group_blocks>=thread_k_blocks,
  // i.e. gs>=64) needs slices aligned to group boundaries.
  if constexpr (group_blocks != -1 && group_blocks >= thread_k_blocks)
    iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters, slice_count = 0, slice_idx;

  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  auto init_slice = [&] () {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1; slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0)) slice_idx = slice_count - 1;
      else { slice_idx = slice_count - 1 - delta_first / iters; if (col_off > 0) slice_idx--; }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8; C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles; slice_col = 0;
    }
  };
  init_slice();

  int a_gl_stride = prob_k / 8;
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

  int b_gl_stride = 16 * prob_n / 32;
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  // GGUF gs=32: a k-tile (thread_k_blocks*16 = 64 k) spans 2 groups (32 k each), so each pipeline stage
  // stages 2 group-rows of scales instead of 1. Every other format stages 1.
  constexpr int groups_per_stage = (group_blocks == 2) ? (thread_k_blocks / 2) : 1;
  constexpr int s_sh_stage = groups_per_stage * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;

  int s_gl_rd = (group_blocks != -1)
      ? s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x
      : s_sh_stride * slice_col + threadIdx.x;
  int s_sh_wr = threadIdx.x;
  // NVIDIA's s_sh_rd is gone: both scale paths now index sh_s by COLUMN (see fetch_to_registers and write_result).
  // Its lane mapping encodes NVIDIA's C/B fragment layouts, which PPU does not share.

  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++) a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // PPU: A smem->reg. DEFAULT = the classic ldmatrix.x4 + XOR swizzle (1 instruction per m-block). The PPU A/B operand
  // fragment layout == NVIDIA, so ppu_ldmatrix_x4 should land correctly -- marlin_classic_num is the oracle.
  // -DMARLIN_A_ELEMENTWISE falls back to the element-wise frag_i/frag_j read (identity transform, no swizzle), which
  // is known-correct but issues ~4 tsm.ld per fragment instead of 1.
#ifdef MARLIN_A_ELEMENTWISE
  auto transform_a = [&] (int i) { return i; };
#else
  auto transform_a = [&] (int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
#endif
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++) a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);

  const int4* B_ptr[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  extern __shared__ int4 sh[];
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s = sh_b + (stages * b_sh_stage);
  int4* sh_m = sh_s + (stages * s_sh_stage);   // AFFINE: mins staged in parallel with the scales
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4];   // PPU: one 8-float n16 acc per (m-block, n-block), native mma order
  FragS frag_s[2][4];
  FragS frag_m[2][4];   // AFFINE (Q4_K/Q4_1): the folded min m' = m + 8s, same layout as frag_s

  // PPU: index frag_c directly -- reinterpret_cast<float*>(frag_c) decays the array to a pointer, which alone is
  // enough to stop hggc from keeping the accumulator in registers.
  auto zero_accums = [&] () {
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int l = 0; l < 8; l++) frag_c[i][j][l] = 0;
  };

  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++)
        cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]], &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off], a_sh_wr_pred[i]);
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) { cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]); B_ptr[i] += b_gl_rd_delta_o; }
      if constexpr (group_blocks == 2) {
        // GGUF gs=32: stage groups_per_stage (=2) group-rows every stage; each 32-block has its own scale.
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;
        int4* sh_m_stage = sh_m + s_sh_stage * pipe;
        #pragma unroll
        for (int g = 0; g < groups_per_stage; g++)
          if (s_sh_wr_pred) {
            cp_async4_stream(&sh_s_stage[g * s_sh_stride + s_sh_wr], &s[s_gl_rd + g * s_gl_stride]);
            if (AFFINE) cp_async4_stream(&sh_m_stage[g * s_sh_stride + s_sh_wr], &mn[s_gl_rd + g * s_gl_stride]);
          }
        s_gl_rd += groups_per_stage * s_gl_rd_delta;
      } else if (group_blocks != -1 && pipe % (group_blocks / thread_k_blocks) == 0) {
        // The mins were MISSING from this branch. group_blocks==2 staged them, this one did not, while
        // matmul applies scale_affine() whenever AFFINE regardless of group_blocks -- so gs>=64 + affine
        // read an UNINITIALIZED frag_m and produced wrong numbers. Not caught because the only affine test
        // case (run_q4k) is gs=32, and the bench does not check correctness at all: it timed a gs=128
        // affine that was never doing the work, which is what made affine look free there.
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;
        if (s_sh_wr_pred) {
          cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
          if (AFFINE) cp_async4_stream(&(sh_m + s_sh_stage * pipe)[s_sh_wr], &mn[s_gl_rd]);
        }
        s_gl_rd += s_gl_rd_delta;
      }
    }
    cp_async_fence();
  };
  auto wait_for_stage = [&] () { cp_async_wait<stages - 2>(); __syncthreads(); };

  auto fetch_to_registers = [&] (int k, int pipe) {
    // GROUPED SCALES (group_blocks != -1, i.e. groupsize=128 -- what every real model uses).
    //
    // matmul() applies these to the B FRAGMENTS, before the mma -- not to C. So the layout they need is
    // dictated by where dequant puts those fragments:
    //     dequant(q)      -> column n     = (warp_n*4 + j)*16 + lane/4     scaled by frag_s[.][j][0]
    //     dequant(q >> 8) -> column n + 8                                  scaled by frag_s[.][j][1]
    // and PPU's B fragment layout is IDENTICAL to NVIDIA's (same (n,k) set AND same register order --
    // which is exactly why the whole B path is verbatim upstream). Therefore upstream's scale layout is
    // PPU-correct, and this reads it with upstream's single int4 per lane.
    //
    // Working it through: lane (warp_n=w, c=lane/4) needs the 8 halves s[64w + 8t + c], t=0..7. One int4
    // at s_sh_rd = 8w + c covers shared halves [64w + 8c .. +7]. So the buffer must hold
    //     buf[64w + 8c + t] == s_plain[64w + 8t + c]
    // i.e. an 8x8 transpose inside every 64-column chunk -- which is precisely upstream's _scale_perm
    // ("for i in range(8): perm += [i + 8*j for j in range(8)]"). marlin_permute_scales() below is that
    // transpose; use it, do not re-derive it.
    //
    // VERIFIED ON ppu001 (test_marlin_classic_group, all four configs, 1 group AND 4/8 groups): MATCH.
    // A long-standing comment here claimed this mapping was "wrong on PPU". It was not. It had only ever
    // been run against an UNPERMUTED buffer, where of course it fails -- and that failure was misread as
    // the mapping being incompatible. The permutation is a HOST-side format, not a kernel property.
    //
    // Consequence for integration: a vLLM / GPTQ-Marlin checkpoint's B AND grouped scales are both
    // directly edible, with no transform at load. (gs == -1 is different: that scale is applied in
    // write_result, so it IS tied to the C layout, which PPU genuinely does not share -- upstream's
    // _scale_perm_single does NOT carry over. Real models use gs=128.)
    //
    // -DMARLIN_SCALE_PLAIN takes raw [num_groups][prob_n] scales instead (e.g. straight off a GPTQ
    // checkpoint, whose scales are already in that layout). Correct, but 8 scalar half loads per k
    // instead of one int4: ncu measured +10.1M shared loads, +10.0M converts and +10.1M bank conflicts
    // against the int4 form (the PPU issues one conflict per half load -- it does not merge the two
    // halves of a 4-byte word). Worth ~6.6 us on the 4096-cube, so the default is both correct AND the
    // faster of the two.
    if constexpr (group_blocks == 2) {
      // GGUF gs=32: the current stage staged 2 group-rows; k-block k belongs to group (k/2) within it.
      // frag_s[k%2] is loaded (via the k-pipeline) with THAT group's scales, so matmul(k) -- which is
      // unchanged and applies frag_s[k%2] to the B fragments -- gets per-32 scaling. _scale_perm layout,
      // one int4 per lane, same as the default coarse-group read.
      // WARPS SPLIT THE K DIMENSION -- that is what thread_block_reduce() exists to undo, and it is the
      // thing I originally got wrong here. b_sh_wr_iters is NOT thread_k_blocks: it is 8*NB*KB/threads,
      // which is 2 for every config in the CALL_IF list while thread_k_blocks is 4 or 8. The k-block a
      // thread actually works on is therefore split between the loop counter and its warp id, exactly as
      // a_sh_rd encodes it (a_sh_rd += 2 * ((threadIdx.x/32) / (thread_n_blocks/4)) at init, stepped by
      // a_sh_rd_delta_o = 2*NWK per iteration):
      //     ktile_in_stage = (k % b_sh_wr_iters) * NWK + warp_k
      // Using (k % b_sh_wr_iters) / 2 made the group index identically 0, so every warp read the stage's
      // FIRST group -- which is precisely what the readout probe showed (blocks 0-3 -> 0, 4-7 -> 4, ...).
      constexpr int NWK = (threads / 32) / (thread_n_blocks / 4);
      const int warp_k = (threadIdx.x / 32) / (thread_n_blocks / 4);
      const int grp = ((k % b_sh_wr_iters) * NWK + warp_k) / 2;   // gs=32 = 2 k-blocks of 16
      int4* sh_s_stage = sh_s + s_sh_stage * pipe + grp * s_sh_stride;
      const int s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
      if (AFFINE) reinterpret_cast<int4*>(&frag_m[k % 2])[0] =
          (sh_m + s_sh_stage * pipe + grp * s_sh_stride)[s_sh_rd];
    } else if (group_blocks != -1) {
#ifdef MARLIN_SCALE_NONE
      // SPEED-ONLY, RESULT IS WRONG. Removes ALL grouped-scale work from the mainloop (this load and the
      // scale() calls in matmul) while keeping the s staging, the group_blocks branches and the address
      // math -- it BRACKETS what the grouped-scale path costs.
#else
      int4* sh_s_stage = sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
#ifdef MARLIN_SCALE_PLAIN
      // Diagnostic path only, and it does NOT read the mins -- fail the build rather than repeat the bug
      // this branch already had once (affine applied against an uninitialized frag_m).
      static_assert(!AFFINE, "MARLIN_SCALE_PLAIN does not implement the affine mins");
      // Raw [num_groups][prob_n] scales, read by column. No host permutation needed; 8 scalar half loads.
      const half* ss = reinterpret_cast<const half*>(sh_s_stage);
      const int warp_n = (threadIdx.x / 32) % (thread_n_blocks / 4), lane = threadIdx.x % 32;
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        const int c = (warp_n * 4 + j) * 16 + lane / 4;   // column within the tile
        half* fs = reinterpret_cast<half*>(&frag_s[k % 2][j]);
        fs[0] = ss[c];                                    // scales frag_b0 (column n)
        fs[1] = ss[c + 8];                                // scales frag_b1 (column n + 8)
      }
#else
      // DEFAULT: upstream's _scale_perm layout (see marlin_permute_scales). One int4 per lane.
      const int s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];   // max 8*(NB/4-1)+7 < s_sh_stride
      // mins, staged at the same stage offset as the scales (see fetch_to_shared above).
      if (AFFINE) reinterpret_cast<int4*>(&frag_m[k % 2])[0] =
          (sh_m + (sh_s_stage - sh_s))[s_sh_rd];
#endif
#endif
    }
#ifndef MARLIN_A_ELEMENTWISE
    // A smem->reg via ldmatrix.x4 (the classic path): one instruction per m-block, reading the XOR-swizzled stage.
    {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++)
        ppu_ldmatrix_x4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    }
#else
    // Element-wise A read. Plain A smem [16*m_blocks rows][16*thread_k_blocks halves], K-contiguous (no swizzle).
    // k-tile within the stage = (k%iters)*N_WARPS_K + warp_k (interleaved per a_sh_rd_delta_o = 2*N_WARPS_K int4).
    {
      constexpr int NWK = (threads / 32) / (thread_n_blocks / 4);
      const int warp_k = (threadIdx.x / 32) / (thread_n_blocks / 4), lane = threadIdx.x % 32;
      const half * sha = reinterpret_cast<const half *>(sh_a + a_sh_stage * pipe);
      const int rowstride = 16 * thread_k_blocks;               // halves per row
      const int ktile = (k % b_sh_wr_iters) * NWK + warp_k;
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
          int row = i * 16 + frag_i(lane, l), hc = ktile * 16 + 2 * frag_j(lane, l);
          frag_a[k % 2][i][l] = make_half2(sha[row * rowstride + hc], sha[row * rowstride + hc + 1]);
        }
    }
#endif
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };

  // asys (4096^3, max_par=128) says the kernel is NOT memory bound (Memory Dependency 0.018, Shared Memory Pipe Busy
  // 0.004) and NOT warp-starved (Not Selected 0.225 -- warps are ready, the scheduler just picks others). The top stall
  // is Compute Dependency 0.586, and Stall Pipe Busy 0.412 splits as Tensor 0.159 + FALU 0.122 + IALU 0.116: dequant's
  // ALU work (lop3 / hsub2 / hfma2) loads the ALU pipes 1.5x harder than the MMAs load the tensor pipe. That ratio is
  // structural -- MARLIN_MAX_MB=2 feeds each B fragment to 2 mmas instead of 4, so dequant:mma is 4:1 -- and raising MB
  // to pay it back loses to register spills (see MARLIN_MAX_MB).
  //
  // What IS addressable is the dependency CHAIN: below, each j's mma waits on that same j's dequant (lop3 -> hsub2 ->
  // mma). -DMARLIN_DQ_BATCH hoists all four j's dequants first, giving the scheduler 4 independent lop3 chains to
  // interleave and 8 mutually independent mmas afterwards. SAME instructions, just reordered (verified in PTX: 256 mma,
  // 768 lop3, 1536 dequant ALU ops either way); 8 live FragB = 64 B, well under the ~512 B SROA cliff; 0 spill.
  //
  // MEASURED, and it is shape-dependent -- which is why it stays opt-in:
  //   4096^3              302.9 -> 302.0  (-0.3%, inside the 0.6% noise floor)
  //   2048x4096x14336     313.5 -> 319.7  (+2.0%, real)
  // The win tracks K, not M: the long-K shape runs 3.5x more matmul(k) iterations, so the mainloop's dependency chain
  // dominates a larger share of its runtime. On 4096^3 the same reorder buys nothing measurable.
  //
  // The residual Compute Dependency (0.586) is structural, and hoisting cannot touch it:
  //   (a) dequant's own lop3 -> hsub2/hfma2, two levels deep, nothing left to spread;
  //   (b) the mma -> frag_c[i][j] accumulator chain: matmul(k)'s 8 mmas (4 j x 2 i) are mutually independent, but each
  //       matmul(k+1) mma waits on matmul(k)'s same (i,j). MB=2 yields only 8 independent mmas to hide mma latency --
  //       the SECOND cost of MARLIN_MAX_MB=2, on top of the dequant amortization. Deepening it needs MB>2 (real spills)
  //       and an f16 accumulator to pay for the registers, but fp16 accumulation over K=4096 is ~3e-2 relative error.
  auto matmul = [&] (int k) {
#ifdef MARLIN_DQ_BATCH
    FragB fb0[4], fb1[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j], b_quant_shift = b_quant >> 8;
      fb0[j] = dequant(b_quant);
#ifndef MARLIN_SCALE_NONE
      if (group_blocks != -1) { if (AFFINE && MARLIN_APPLY_MIN) scale_affine(fb0[j], frag_s[k % 2][j], frag_m[k % 2][j], 0);
                                else            scale(fb0[j], frag_s[k % 2][j], 0); }
#endif
      fb1[j] = dequant(b_quant_shift);
#ifndef MARLIN_SCALE_NONE
      if (group_blocks != -1) { if (AFFINE && MARLIN_APPLY_MIN) scale_affine(fb1[j], frag_s[k % 2][j], frag_m[k % 2][j], 1);
                                else            scale(fb1[j], frag_s[k % 2][j], 1); }
#endif
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++)
        mma_n16(frag_a[k % 2][i], fb0[j], fb1[j], frag_c[i][j]);      // PPU: 1 n16 (was 2 n8)
#else
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j], b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant(b_quant);
#ifndef MARLIN_SCALE_NONE
      if (group_blocks != -1) { if (AFFINE && MARLIN_APPLY_MIN) scale_affine(frag_b0, frag_s[k % 2][j], frag_m[k % 2][j], 0);
                                else            scale(frag_b0, frag_s[k % 2][j], 0); }
#endif
      FragB frag_b1 = dequant(b_quant_shift);
#ifndef MARLIN_SCALE_NONE
      if (group_blocks != -1) { if (AFFINE && MARLIN_APPLY_MIN) scale_affine(frag_b1, frag_s[k % 2][j], frag_m[k % 2][j], 1);
                                else            scale(frag_b1, frag_s[k % 2][j], 1); }
#endif
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++)
        mma_n16(frag_a[k % 2][i], frag_b0, frag_b1, frag_c[i][j]);   // PPU: 1 n16 (was 2 n8)
    }
#endif
  };

  // PPU: UNCHANGED -- register-index-aligned cross-K-warp reduce is correct on PPU.
  auto thread_block_reduce = [&] () {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
      #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
        #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
            #pragma unroll
            // PPU: the staged float4 chunk j of m_block is frag_c[m_block][j/2][4*(j%2) .. +3] -- same flat order as
            // the NVIDIA int4 view, but reached by direct indices so frag_c is never address-taken.
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              const int jj = j / 2, l0 = 4 * (j % 2);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
                #pragma unroll
                for (int k = 0; k < 4; k++) frag_c[m_block][jj][l0 + k] += c_rd[k] + c_wr[k];
              }
              *reinterpret_cast<float4*>(&sh[red_sh_wr]) = make_float4(
                  frag_c[m_block][jj][l0 + 0], frag_c[m_block][jj][l0 + 1],
                  frag_c[m_block][jj][l0 + 2], frag_c[m_block][jj][l0 + 3]);
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
          #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
            #pragma unroll
            for (int j = 0; j < 4; j++) frag_c[m_block][i / 2][4 * (i % 2) + j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // PPU: cross-CTA split-K reduce (slice_count>1) redone with the PPU C layout (acc_i/acc_j) -- same (row,col) as
  //      write_result, so partials chain correctly. Each CTA (except first) reads the previous partial from C (fp16),
  //      adds its frag_c (fp32), writes back (unless last -> left in frag_c for write_result). Direct global access
  //      (perf TODO: the NVIDIA cp.async-staged version). Only N-warps hold the reduced frag_c after thread_block_reduce.
  auto global_reduce = [&] (bool first = false, bool last = false) {
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warp_n = warp % (thread_n_blocks / 4);
      half* Ch = reinterpret_cast<half*>(C);
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          const int ncol0 = (slice_col * thread_n_blocks + warp_n * 4 + j) * 16;
          #pragma unroll
          for (int l = 0; l < 8; l++) {
            int r = 16 * i + acc_i(lane, l), cc = ncol0 + acc_j(lane, l);
            if (r < prob_m) {
              if (!first) frag_c[i][j][l] += __half2float(Ch[(long long) r * prob_n + cc]);   // previous CTA's partial
              if (!last)  Ch[(long long) r * prob_n + cc] = __float2half(frag_c[i][j][l]);    // hand to the next CTA
            }
          }
        }
    }
  };

  // PPU: smem-staged COALESCED write-back. The direct per-element store (still available under -DMARLIN_WRITE_DIRECT)
  //      issues ONE 2-byte global store per acc float -- 16*thread_m_blocks * 4 * 8 of them per CTA -- and each one
  //      pulls a full 128 B line, because a lane's columns are strided by 4 (PPU C layout: col = lane%4 + 4*(l%4)).
  //      Instead: stage the whole output tile through shared in row-major order, then have ALL threads stream it to
  //      global as int4 (8 halves), one full sector per store.
  //
  //      The PPU C layout only affects the smem WRITE indices (acc_i/acc_j); the read-out is layout-agnostic. NVIDIA's
  //      version packs a half2 per store because its lanes own ADJACENT columns -- PPU lanes do not (lane0 owns cols
  //      {0,4,8,12}), so the smem side stays per-half. That is fine: shared stores are cheap, the global ones were not.
  //
  //      Staged tile: (16*thread_m_blocks) rows x (2*thread_n_blocks + 1) int4. The trailing +1 int4 (8 halves) pads
  //      the row stride so the strided smem writes don't all land in the same banks. Only the N-warps hold the reduced
  //      frag_c, so only they write; every thread participates in the read-out.
  //      PER-COLUMN SCALE (group_blocks == -1). This used to be dropped on the floor -- frag_s was loaded into
  //      registers and never multiplied in -- and every test passed because they all set s = 1.0, the multiplicative
  //      identity. The kernel therefore returned unscaled results for any real quantized weight.
  //
  //      We do NOT use NVIDIA's frag_s lane mapping (its C layout gives each lane two ADJACENT columns; PPU's gives
  //      four stride-4 columns, so the mapping does not carry over). Read the scale straight out of sh_s instead:
  //      the host stages sh_s[i] = s[s_sh_stride*slice_col + i] with s_sh_stride = 2*thread_n_blocks int4, i.e.
  //      16*thread_n_blocks halves -- exactly one tile's worth of columns, in column order. So
  //          ((const half*) sh_s)[column_within_tile]
  //      is the scale for that column, for any C layout. group_blocks != -1 already scales inside matmul().
  //      -DMARLIN_SKIP_WRITE_SCALE restores the old (wrong) behaviour, to demonstrate the tests now catch it.
  auto write_result = [&] () {
#ifdef MARLIN_WRITE_DIRECT
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warp_n = warp % (thread_n_blocks / 4);
      half* Ch = reinterpret_cast<half*>(C);
      const half* sh_s_h = reinterpret_cast<const half*>(sh_s);
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          const int ncol_t = (warp_n * 4 + j) * 16;                       // column within the tile -> indexes sh_s
          const int ncol0  = slice_col * thread_n_blocks * 16 + ncol_t;   // global column -> indexes C
          #pragma unroll
          for (int l = 0; l < 8; l++) {
            const int cj = acc_j(lane, l);
            int r = 16 * i + acc_i(lane, l), cc = ncol0 + cj;
            float v = frag_c[i][j][l];
#ifndef MARLIN_SKIP_WRITE_SCALE
            if (group_blocks == -1) v *= __half2float(sh_s_h[ncol_t + cj]);
#endif
            if (r < prob_m) Ch[(long long) r * prob_n + cc] = __float2half(v);
          }
        }
    }
#else
    constexpr int c_sh_stride   = 2 * thread_n_blocks + 1;   // int4 per staged row (incl. the pad)
    constexpr int c_sh_stride_h = 8 * c_sh_stride;           // halves per staged row
    constexpr int c_int4_per_row = 2 * thread_n_blocks;      // int4 per REAL row (tile is 16*thread_n_blocks halves)
    constexpr int c_rows_per_iter = threads / c_int4_per_row;
    constexpr int c_sh_rd_delta = c_sh_stride * c_rows_per_iter;

    const int c_gl_stride = prob_n / 8;
    int c_gl_wr = c_gl_stride * (threadIdx.x / c_int4_per_row) + (threadIdx.x % c_int4_per_row)
                + c_int4_per_row * slice_col;
    int c_sh_rd = c_sh_stride * (threadIdx.x / c_int4_per_row) + (threadIdx.x % c_int4_per_row);
    const int c_gl_wr_delta = c_gl_stride * c_rows_per_iter;
    const int c_gl_wr_end = c_gl_stride * prob_m;   // row >= prob_m overshoots this -> the OOB guard, as in NVIDIA marlin

    // Read the scales BEFORE sh is repurposed: the staging area overlaps sh_a/sh_b, not sh_s (which lives at
    // stages*(a_sh_stage + b_sh_stage)), but keep the load ahead of the barrier so the dependency is obvious.
    const half* sh_s_h = reinterpret_cast<const half*>(sh_s);
    __syncthreads();   // sh is thread_block_reduce's scratch; take it over only once every warp is done reading it
    half* sh_h = reinterpret_cast<half*>(sh);
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warp_n = warp % (thread_n_blocks / 4);
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          const int ncol0 = (warp_n * 4 + j) * 16;   // column INSIDE the tile -- slice_col is folded into c_gl_wr
          #pragma unroll
          for (int l = 0; l < 8; l++) {
            const int cj = acc_j(lane, l);
            float v = frag_c[i][j][l];
#ifndef MARLIN_SKIP_WRITE_SCALE
            if (group_blocks == -1) v *= __half2float(sh_s_h[ncol0 + cj]);
#endif
            sh_h[(16 * i + acc_i(lane, l)) * c_sh_stride_h + ncol0 + cj] = __float2half(v);
          }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, c_rows_per_iter); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();   // the next slice's start_pipes() overwrites sh_a/sh_b -- don't let it race the read-out above
#endif
  };

  auto start_pipes = [&] () {
    #pragma unroll
    for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);
    zero_accums(); wait_for_stage(); fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();

  while (slice_iters) {
    #pragma unroll
    for (int pipe = 0; pipe < stages;) {
      #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters - 2) { fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages); pipe++; wait_for_stage(); }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      if (group_blocks == -1 && last) { if (s_sh_wr_pred) cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]); cp_async_fence(); }
      thread_block_reduce();
      // Land the scales in sh_s and make them visible; write_result indexes them by column, so no frag_s staging is
      // needed on PPU. (NVIDIA loaded frag_s here via s_sh_rd, whose lane mapping assumes NVIDIA's C layout. We used to
      // do the same and then never multiply by it -- the load was dead, and s = 1.0 in every test hid that.)
      if (group_blocks == -1 && last) { cp_async_wait<0>(); __syncthreads(); }
#ifdef MARLIN_BENCH_NOREDUCE
      // SPEED-ONLY PROBE, RESULT IS WRONG ON PURPOSE (no reduce ever runs). Models a splitk_PARALLEL mainloop: skip
      // barrier_acquire/global_reduce/barrier_release, and have each slice write its own gmem slot. The slot offset is
      // the whole C plane, so slices never touch the same cache line.
      //
      // The slot is what makes this a valid probe. v1 of this flag let every slice race to write the SAME C, which for
      // slice_count=112 put 112 CTAs on one cache line -- and measured identical to the barrier (7961.06 vs 7960.59 us).
      // That proved only that a racy write serializes as badly as a barrier, not that splitk_parallel would. Traffic and
      // contention are different quantities; the v1 comment conflated them.
      //
      // Caller must size C as slice_count * (prob_m * prob_n / 8) int4. bench_marlin over-allocates when NOREDUCE is on.
      {
        int4* C_save = C;
        C += (long long) slice_idx * prob_m * (prob_n / 8);
        write_result();
        C = C_save;
      }
#else
      if (slice_count > 1) {
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }
      if (last) write_result();
#endif
      slice_row = 0; slice_col_par++; slice_col++; init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
        #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
          #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] -= b_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

// STAGES = cp.async pipeline depth. asys puts Stall Sync at 0.142 -- the wait_for_stage() __syncthreads(), and the
// 2nd-largest addressable stall after Compute Dependency. A deeper pipeline gives cp.async more slack before the
// barrier, at the cost of shared memory. The whole pipeline (wait_for_stage, pipe % stages, start_pipes) is already
// written against the `stages` template param, so this is the only constant that was pinned.
//
// Budget (per CU: 256 KB shared; __launch_bounds__(threads, MARLIN_MIN_BLOCKS=2) wants 2 resident blocks):
//   (2,16,4):  STAGES=4 -> 51200 B/blk (2 blk = 100 KB)   5 -> 64000 (125 KB)   6 -> 76800 (150 KB)   all fit
// Occupancy is NOT the concern here (asys: Not Selected 0.225, warps are already idle-ready) -- the concern is only
// that 2 blocks still fit, since MARLIN_MIN_BLOCKS=2 is what keeps the register budget at 256/thread and spills at 0.
#ifndef MARLIN_STAGES
#define MARLIN_STAGES 4
#endif
static const int THREADS = 256, STAGES = MARLIN_STAGES;
// hggc's SROA gives up on an alloca of >= 512 bytes: frag_c is thread_m_blocks*4*8 floats, so MB=4 (512 B) is NEVER
// promoted to registers -- ptxas says "516 bytes stack frame, 0 spilled VReg" (0 spilled = never promoted, NOT a
// register-budget spill) and the SASS round-trips all 8 acc floats to local memory around every single mma. MB=3
// (384 B) promotes. MB=2 (256 B) promotes cleanly (0/0). Measured on 2048x14336x4096: MB=4 -> 32.7 TFLOP/s,
// MB=2 -> 287.1. Worth ~4.2M extra lop3 (dequant amortizes over 2 mmas, not 4) to kill 68M local accesses.
#ifndef MARLIN_MAX_MB
#define MARLIN_MAX_MB 2
#endif

// PPU: use the ACTUAL per-config shared (not a fixed 96KB) so occupancy isn't wasted. Per stage (int4):
// a_sh_stage=32*KB*MB, b_sh_stage=8*NB*KB, s_sh_stage=2*NB. x STAGES x 16 bytes. (Reuse for the reduce fits.)
// The scale term is 2*NB int4 per group per stage; GGUF gs=32 (GB==2) stages KB/2 groups per stage.
#define MARLIN_SHMEM(MB, NB, KB, GB, AFF) (STAGES * (32 * (KB) * (MB) + 8 * (NB) * (KB) + ((AFF) ? 2 : 1) * ((GB) == 2 ? (KB) / 2 : 1) * 2 * (NB)) * 16)

#define CALL_IF_A(MB, NB, KB, GB, AFF) \
  else if (thread_m_blocks == MB && thread_n_blocks == NB && thread_k_blocks == KB && group_blocks == GB && affine == AFF) { \
    const int shmem = MARLIN_SHMEM(MB, NB, KB, GB, AFF); \
    cudaFuncSetAttribute(Marlin<THREADS, MB, NB, KB, STAGES, GB, AFF>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem); \
    Marlin<THREADS, MB, NB, KB, STAGES, GB, AFF><<<blocks, THREADS, shmem, stream>>>(A_ptr, B_ptr, C_ptr, s_ptr, m_ptr, prob_m, prob_n, prob_k, locks); }
#define CALL_IF(MB, NB, KB, GB) CALL_IF_A(MB, NB, KB, GB, false) CALL_IF_A(MB, NB, KB, GB, true)

// max_par = how many m-tiles ONE launch covers. This function issues ceil(tot_m_blocks / MARLIN_MAX_MB / par) kernels
// back-to-back on `stream`, so a profiler only ever shows you ONE of them.
//
// WORKSPACE CONTRACT: `workspace` (the locks array) must hold at least (prob_n/128 + 1) * max_par ints, zeroed. It
// scales with max_par -- raise one without the other and the kernel writes out of bounds. (prob_n/128, not /256:
// the prob_m<=16 configs use thread_n=128, so they have twice the n-tiles.)
//
// PPU (72 CU): NVIDIA's max_par=16 was tuned for a 108-SM A100. Here it pinned the grid at cols*16 = 256 CTA -> 4
// waves whose last fills 40/72 CUs, and that tail was paid once per launch, plus ~2.1us of launch latency each.
// Measured on 4096^3: max_par 16 -> 266.3 TFLOP/s, 32 -> 272.4, 64 -> 293.9, 128 -> 303.6 (+14%). On
// 2048x4096x14336 it saturates at par = tot_m_blocks/MB = 64: 294.9 -> 312.5. Hence the default below.
// Decode (prob_m <= 16) has tot_m_blocks == 1, never enters the par branch, and is unaffected.
static int marlin_cuda(const void* A, const void* B, void* C, void* s, void* mins, int prob_m, int prob_n, int prob_k,
        void* workspace, int groupsize = -1, int dev = 0, cudaStream_t stream = 0,
        int thread_k = -1, int thread_n = -1, int sms = -1, int max_par = 128) {
  int tot_m = prob_m, tot_m_blocks = ceildiv(tot_m, 16), pad = 16 * tot_m_blocks - tot_m;
  if (sms == -1) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) { thread_k = 128; thread_n = 128; } else { thread_k = 64; thread_n = 256; }
  }
  int thread_k_blocks = thread_k / 16, thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16, blocks = sms;
  // Split-K heuristic. With grid=#SMs the stripe dispatcher flattens (tile,k) and cuts it evenly, so stripe boundaries
  // land mid-tile and split K *even when there are far more tiles than SMs* -> global_reduce runs for nothing. Setting
  // grid=#output-tiles (cols*par) gives each CTA a full tile's K (iters=k_tiles, slice_count=1) and global_reduce
  // never runs. Benched: no-split-K wins big whenever #tiles >= #SMs (prefill 1.4-1.9x; decode N=14336 1.7x) and LOSES
  // when #tiles < #SMs (decode N=4096 K=14336: 0.7x -- 40 SMs idle grinding a long K). So: don't split K if the tiles
  // already fill the machine. MARLIN_NOSPLITK / MARLIN_SPLITK force either way.
  static const bool force_nosplit = getenv("MARLIN_NOSPLITK") != nullptr;
  static const bool force_split   = getenv("MARLIN_SPLITK")   != nullptr;
  // MARLIN_BLOCKS overrides the grid outright (diagnostic only -- see below, a bigger grid never wins). Safe at any
  // size: iters = ceildiv(k_tiles*n_tiles*par, gridDim.x), and init_slice() retires any CTA whose slice_col_par lands
  // past n_tiles*par, so extras idle. `locks` is indexed by slice_col < n_tiles, so unlike max_par the workspace does
  // NOT scale with it.
  //
  // MEASURED (decode, M=1, N=4096): raising blocks past the heuristic's choice is CATASTROPHIC, monotonically.
  //   blocks    72     144     288     576    1024    3584
  //   slice_cnt  3       4       8      16      32     112     <- barrier_acquire chain length per output tile
  //   K=4096   17.8    18.4    47.5   176.2   588.1     --  us
  //   K=14336  27.1    28.7    87.7   182.2   474.8  7960.6 us
  // t ~ slice_count^1.5. The linear part is global_reduce: the slices hand partials down one at a time, fully
  // serialized. The superlinear part is barrier_acquire spinning on locks[slice_col] -- only n_tiles locks exist, so
  // more CTAs means more contention. Split-K's parallelism is exhausted at slice_count 3-4, and blocks=sms lands
  // exactly there. NOT a traffic problem (M=1 writes 128 halves per slice); serialization + lock contention.
  //
  // Decode's real limit is upstream of all this: M<=16 selects thread_n=128, so only N/128 output tiles exist -- 32
  // for N=4096 against 72 CUs (0.44x). N=14336 has 112 tiles and reaches 65.7% HBM; N=4096 gets 17.5%. Fixing that
  // needs more tiles (thread_n=64 requires thread_k=256, since b_sh_wr_iters = b_sh_stride*KB/threads must be >= 2 or
  // the `k == b_sh_wr_iters - 2` pipeline trigger never fires) -- or a purpose-built W4A16 GEMV, which is the honest
  // answer: Marlin's tile is 16x128 and decode uses one of those 16 rows.
  static const char* blocks_env = getenv("MARLIN_BLOCKS");
  int cols = prob_n / thread_n;
  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 || (group_blocks != -1 && prob_k % group_blocks != 0)) return 1;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0) return 0;
  const int4* A_ptr = (const int4*) A; const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C; const int4* s_ptr = (const int4*) s;
  const int4* m_ptr = (const int4*) mins;      // AFFINE mins (m' = m + 8s); may be null for symmetric
  const bool affine = mins != nullptr;         // Q4_K/Q4_1 pass mins; Q4_0/IQ4 pass nullptr
  int* locks = (int*) workspace;
  int ret = 0;
  const int MB = MARLIN_MAX_MB;   // see MARLIN_MAX_MB: caps frag_c below hggc's ~512-byte SROA cliff
  for (int i = 0; i < tot_m_blocks; i += MB) {
    int thread_m_blocks = tot_m_blocks - i; prob_m = tot_m - 16 * i; int par = 1;
    if (thread_m_blocks > MB) { par = (16 * thread_m_blocks - pad) / (16 * MB); if (par > max_par) par = max_par;
      prob_m = 16 * MB * par; i += MB * (par - 1); thread_m_blocks = MB; }
    int tiles = cols * par;                // one CTA per output tile -> full-K, no split
    blocks = (force_nosplit || (!force_split && tiles >= sms)) ? tiles : sms;
    if (blocks_env) blocks = atoi(blocks_env);
    if (false) {}
    // Only instantiate configs the dispatcher can actually reach (thread_m_blocks <= MARLIN_MAX_MB). Otherwise the
    // unreachable (3,·)/(4,·) kernels still get compiled and their spill/stack-frame lines pollute -Xptxas -v.
    CALL_IF(1, 8, 8, -1) CALL_IF(1, 8, 8, 8) CALL_IF(1, 8, 8, 2) CALL_IF(1, 16, 4, -1) CALL_IF(1, 16, 4, 8) CALL_IF(1, 16, 4, 2)
#if MARLIN_MAX_MB >= 2
    CALL_IF(2, 16, 4, -1) CALL_IF(2, 16, 4, 8) CALL_IF(2, 16, 4, 2)
#endif
#if MARLIN_MAX_MB >= 3
    CALL_IF(3, 16, 4, -1) CALL_IF(3, 16, 4, 8) CALL_IF(3, 16, 4, 2)
#endif
#if MARLIN_MAX_MB >= 4
    CALL_IF(4, 16, 4, -1) CALL_IF(4, 16, 4, 8) CALL_IF(4, 16, 4, 2)
#endif
    else ret = 2;
    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }
  return ret;
}

} // namespace marlin_gguf_ppu

#endif
