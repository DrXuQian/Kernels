// L46 -- GENERATE plane 2's cross-plane offline placement, using the converter's TRUE pairing.
//
// l45 showed a count-level bijection exists, but paired the thread's i-th low code with its i-th high bit in slot
// order -- necessary, not sufficient. The real pairing is what MixGemm2Plane_uint2_uint1 emits, and it is explicit in
// the _E2 lines (l37 verified it):
//
//   line (t, v), t in [0,8), v in [0,4)
//     LOW : two crumbs of lo[v] -- code index within the 64-code chunk = 16*v + (t%4) + 4*(t/4), and +8
//     HIGH: two bits of hi[2*(v>>1)] at 8*(v&1)+t and +16
//           == bit 64*(v>>1) + 8*(v&1) + t of the 128-bit (4-vreg) high chunk, and +16
//
// So ONE _E2 line consumes 2 low codes and 2 high bits, and the pairing between them is (low half <-> bit, high half
// <-> bit+16). That is the granularity a placement must be derived at.
//
// KERNEL-SIDE INDEXING ASSUMED BELOW (and it is a change, because the shipped one is out of range once plane 2 folds):
//     hi chunk   = cvt_hi(_, ii % MMA_N2)
//     uint32 off = (k_block % P2_DIV) + P2_DIV * (ii / MMA_N2)
// P2_DIV==1 and MMA_N2==MMA_N1 reproduces the shipped expression exactly, so the unfolded path is untouched.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l46_xplane_gen.cu -o l46 && ./l46
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
#include <vector>
#include <map>
#include "unfused_weight_dequantize.hpp"
using namespace cute;

template <int TM, int TN, int TK, int WM, int WN, int F2>
static bool gen(const char* tag, std::vector<int>* out_T2 = nullptr) {
  using SInst = PPU0015_16x16x32_S32S8S8S32_TN;
  constexpr int InstM = 16, InstN = 16;
  constexpr int warpM = (WM > InstM) ? WM : InstM, warpN = (WN > InstN) ? WN : InstN;
  constexpr int WOM = TM / warpM, WON = TN / warpN, NTHR = 32 * WOM * WON;
  using MmaS8 = TiledMMA<MMA_Atom<SInst>, Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                         Tile<Int<WOM*16>, Int<WON*16>, _32>>;
  constexpr int R1 = TN,      B1 = TK * 2 / 8;          // plane 1 int2, F1 = 1
  constexpr int R2 = TN / F2, B2 = F2 * TK * 1 / 8;     // plane 2 int1, folded by F2

  auto p1t = MmaS8{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<R1>{}, Int<B1>{})));
  auto p2t = MmaS8{}.get_thread_slice(0).partition_B(make_identity_tensor(make_shape(Int<R2>{}, Int<B2>{})));
  auto pi2 = right_inverse(make_layout(shape<0>(p2t.layout())));
  const int V1 = int(size<0>(p1t.layout())), N1 = int(size<1>(p1t.layout())), KB1 = int(size<2>(p1t.layout()));
  const int V2 = int(size<0>(p2t.layout())), N2 = int(size<1>(p2t.layout())), KB2 = int(size<2>(p2t.layout()));
  const int P2_DIV = KB2 ? KB1 / KB2 : 0;
  if (!KB2 || KB1 % KB2) { printf("  %-26s P2_DIV not integral (%d/%d)\n", tag, KB1, KB2); return false; }

  // T2[plane-2 physical bit] = logical element (n*TK + k) whose HIGH bit belongs there
  std::vector<int> T2((size_t)R2 * B2 * 8, -1);
  std::map<int,int> mult; long conflicts = 0;

  for (int t = 0; t < NTHR; ++t) {
    auto p1 = MmaS8{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<R1>{}, Int<B1>{})));
    auto p2 = MmaS8{}.get_thread_slice(t).partition_B(make_identity_tensor(make_shape(Int<R2>{}, Int<B2>{})));
    for (int kb = 0; kb < KB1; ++kb)
      for (int ii = 0; ii < N1; ++ii) {
        const int hi_chunk = ii % N2;                      // cvt_hi(_, ii % MMA_N2)
        const int hi_vreg0 = (kb % P2_DIV) + P2_DIV * (ii / N2);   // uint32 offset into the 4-vreg chunk
        const int kb2      = kb / P2_DIV;
        for (int v = 0; v < 4; ++v)
          for (int lt = 0; lt < 8; ++lt)
            for (int half = 0; half < 2; ++half) {
              // ---- low side: which logical (n,k)
              const int code = 16 * v + (lt % 4) + 4 * (lt / 4) + 8 * half;   // within the 64-code chunk
              auto c1 = p1(code / 4, ii, kb);                                  // int8 slot -> (row, byte)
              const int n = int(get<0>(c1));                                   // plane 1 unfolded: row == logical n
              const int k = int(get<1>(c1)) * 4 + (code % 4);
              // ---- high side: which plane-2 physical bit
              const int w   = hi_vreg0 + 2 * (v >> 1);                          // vreg within the chunk
              const int bit = 8 * (v & 1) + lt + 16 * half;                     // bit within that vreg
              if (w >= V2 / 4) { printf("  %-26s high vreg %d out of range (chunk has %d)\n", tag, w, V2/4); return false; }
              // pi = right_inverse(frag.layout()), the same composition l20's tile_map uses. The delivered order is
              // NOT partition_B's slot order; skipping pi made the control buffer differ in every one of its 16384
              // bytes, which is exactly what that diff exists to catch.
              auto c2 = p2(pi2(w * 4 + bit / 8), hi_chunk, kb2);                // int8 slot -> (g, byte)
              const long bp = ((long)int(get<0>(c2)) * B2 + int(get<1>(c2))) * 8 + (bit % 8);
              const int e = n * TK + k;
              if (T2[bp] >= 0 && T2[bp] != e) ++conflicts;
              T2[bp] = e; ++mult[e];
            }
      }
  }
  long unclaimed = 0; for (int e : T2) if (e < 0) ++unclaimed;
  long badmult = 0; for (auto& kv : mult) if (kv.second != WOM) ++badmult;
  const bool ok = (conflicts == 0 && unclaimed == 0 && badmult == 0);
  printf("  %-26s F2=%d MMA_N %d/%d P2_DIV=%d | bits=%zu conflicts=%ld unclaimed=%ld mult!=%d:%ld -> %s\n",
         tag, F2, N1, N2, P2_DIV, T2.size(), conflicts, unclaimed, WOM, badmult,
         ok ? "PLACEMENT DERIVED" : "INCONSISTENT under the true pairing");
  if (out_T2) *out_T2 = T2;
  return ok;
}

// Emit plane 2's whole buffer from the derived tile map, using the SAME interleave-256 addressing place_derived uses,
// and diff it against the shipped offline. Consistency (l46 above) only says no placement is ruled out; THIS is what
// says the generator is right, because at F2=1 it must reproduce the buffer that is running correctly today.
template <int TM, int TN, int TK, int WM, int WN, int F2>
static bool emit_and_diff(const char* tag, int N, int K) {
  std::vector<int> T2;
  if (!gen<TM,TN,TK,WM,WN,F2>("(map)", &T2)) return false;
  constexpr int Bits = 1, CPW = 32 / Bits, R2 = TN / F2, B2 = F2 * TK * Bits / 8;
  const int kCon = 256, AiuByte = B2 > 128 ? 128 : B2, AiuElem = AiuByte * 8 / Bits, RPS = kCon / AiuElem;
  const int W_ROW_OFF = 256 / CPW, RUNS = W_ROW_OFF / 8, nrow = N / F2;

  std::vector<int> codes((size_t)N * K);
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = int((i * 2654435761u >> 7) & 1);
  std::vector<int8_t> derived((size_t)N * K / 8, 0), shipped;
  for (int tn = 0; tn < N / TN; ++tn)
    for (int ki = 0; ki < K / TK; ++ki)
      for (int row = 0; row < R2; ++row)
        for (int cc = 0; cc < B2 * 8; ++cc) {
          const int loc = T2[(size_t)row * B2 * 8 + cc];
          if (loc < 0) continue;
          const int n = tn * TN + loc / TK, k = ki * TK + loc % TK;
          if (!(codes[(size_t)n * K + k] & 1)) continue;
          const int wd = cc / CPW, j = cc % CPW;
          size_t bitpos;
          if (F2 > 1) { const int kb = ki / RUNS, t = ki % RUNS;
                        bitpos = (size_t)((((size_t)kb * nrow + (size_t)tn * R2 + row) * W_ROW_OFF + t * 8 + wd) * CPW + j); }
          else        { bitpos = (size_t)(((size_t)(ki / RPS) * N + (size_t)tn * TN + row) * kCon
                                          + (ki % RPS) * AiuElem + wd * CPW + j); }
          derived[bitpos / 8] |= int8_t(1 << (bitpos % 8));
        }
  { std::vector<int8_t> packed((size_t)N * K / 8, 0);
    for (size_t i = 0; i < (size_t)N * K; ++i) if (codes[i] & 1) packed[i / 8] |= int8_t(1 << (i % 8));
    // the caller packs qT[n*K+k]; codes above are already in that order
    shipped.assign(packed.size(), 0);
    preprocess_weights_for_mixed_gemm<false, 256, 0>(shipped.data(), packed.data(),
        {(size_t)K, (size_t)N}, QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);
    if (F2 > 1) { std::vector<int8_t> f(shipped.size());
                  nfold_regroup_gmem(f.data(), shipped.data(), {(size_t)K,(size_t)N}, TN, TK, 1); shipped.swap(f); } }
  size_t d = 0; for (size_t i = 0; i < derived.size(); ++i) if (derived[i] != shipped[i]) ++d;
  printf("  %-30s %dx%d : %zu / %zu bytes differ  %s\n", tag, N, K, d, derived.size(),
         d ? "<-- DIFFERS from the shipped offline" : "<-- BIT-IDENTICAL");
  return d == 0;
}

int main() {
  printf("L46 -- cross-plane placement generation under the converter's TRUE pairing\n\n");
  printf("  control: the SHIPPING unfolded shape, where the derived map must reproduce today's working placement\n");
  bool c = gen<64, 64,256,32,32, 1>("Q3 (64,64,256) F2=1");
  printf("\n  the folded shapes\n");
  bool a = gen<64, 64,128,32,32, 2>("Q3 (64,64,128) F2=2");
  bool b = gen<32,128,128,32,32, 2>("Q3 (32,128,128) F2=2");
  printf("\n  and the check that makes it trustworthy: at F2=1 the derived buffer must EQUAL the shipped one\n");
  bool d = emit_and_diff<64, 64,256,32,32, 1>("control F2=1 vs shipped", 256, 512);
  printf("\n  verdict: %s\n",
         (c && a && b && d) ? "generator validated on the control -- the folded map can be trusted"
                       : "at least one shape is inconsistent; the kernel-side indexing assumed at the top is the suspect");
  return 0;
}
