// L51 -- does plane_map<1,...,F=2> (my reconstruction of "plane 2's own rule") equal what nfold_regroup_gmem actually
// produces? Everything downstream hinges on it: l49 measured "demanded vs in-use = 4096 differ" against that
// reconstruction, and if the reconstruction is not the shipped fold then that number describes nothing.
#include "cute/tensor.hpp"
#include "unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"
#include "xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;

template <int TM,int TN,int TK,int WM,int WN,int F2>
static void emit_from(const std::vector<int>& m, std::vector<int8_t>& out,
                      const std::vector<uint8_t>& hi, int N, int K) {
  constexpr int CPW = 32, R2 = TN / F2;
  const int W_ROW_OFF = 256 / CPW, RUNS = W_ROW_OFF / 8, nrow = N / F2;
  out.assign((size_t)N * K / 8, 0);
  for (int tn = 0; tn < N/TN; ++tn) for (int ki = 0; ki < K/TK; ++ki)
    for (int row = 0; row < R2; ++row) for (int wd = 0; wd < 8; ++wd) for (int j = 0; j < CPW; ++j) {
      const int loc = m[((size_t)row*8 + wd)*CPW + j]; if (loc < 0) continue;
      const int n = tn*TN + loc/TK, k = ki*TK + loc%TK;
      if (!(hi[(size_t)k*N + n] & 1)) continue;
      const int kb = ki/RUNS, t = ki%RUNS;
      const size_t b = (size_t)((((size_t)kb*nrow + (size_t)tn*R2 + row)*W_ROW_OFF + t*8 + wd)*CPW + j);
      out[b/8] |= int8_t(1 << (b%8));
    }
}

int main() {
  constexpr int TM=64,TN=64,TK=128,WM=32,WN=32,F2=2; const int N=256,K=512;
  std::vector<uint8_t> hi((size_t)K*N);
  for (size_t i=0;i<hi.size();++i) hi[i]=uint8_t((i*2654435761u>>7)&1);
  std::vector<int8_t> packed((size_t)N*K/8,0);
  for (int k=0;k<K;++k) for (int n=0;n<N;++n) if (hi[(size_t)k*N+n]&1){size_t i=(size_t)n*K+k; packed[i/8]|=int8_t(1<<(i%8));}
  std::vector<int8_t> ship(packed.size());
  preprocess_weights_for_mixed_gemm<false,256,0>(ship.data(),packed.data(),{(size_t)K,(size_t)N},QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);
  { std::vector<int8_t> f(ship.size()); legacy::nfold_regroup_gmem(f.data(),ship.data(),{(size_t)K,(size_t)N},TN,TK,1); ship.swap(f); }

  std::vector<int8_t> a,b;
  emit_from<TM,TN,TK,WM,WN,F2>(xplane::plane_map<1,TM,TN,TK,WM,WN,F2>(), a, hi, N, K);
  emit_from<TM,TN,TK,WM,WN,F2>(xplane::tile_map_int1<TM,TN,TK,WM,WN,F2>(), b, hi, N, K);
  auto diff=[&](const std::vector<int8_t>&x,const char*t){ size_t d=0; for(size_t i=0;i<x.size();++i) if(x[i]!=ship[i])++d;
    printf("  %-40s vs shipped fold: %zu / %zu differ %s\n", t, d, x.size(), d?"":"<-- IDENTICAL"); };
  diff(a, "plane_map<1,..,F=2>  (my reconstruction)");
  diff(b, "tile_map_int1        (composed, HV=1)");
  return 0;
}
