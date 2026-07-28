// L53 -- is place_int1 == the shipped fold at the BOX'S shape (K=256, i.e. 2 k-tiles)? l47 only ever tested K=512.
#include "cute/tensor.hpp"
#include "unfused_weight_dequantize.hpp"
#include "legacy_pipeline.hpp"
#include "xplane_offline.hpp"
#include <cstdio>
#include <vector>
using namespace cute;
int main() {
  constexpr int TM=64,TN=64,TK=128,WM=32,WN=32,F2=2;
  for (int K : {512, 256}) {
    const int N = 256;
    std::vector<uint8_t> hi((size_t)K*N);
    for (size_t i=0;i<hi.size();++i) hi[i]=uint8_t((i*2654435761u>>7)&1);
    std::vector<int8_t> derived((size_t)N*K/8 + 64, 0);      // +64 guard: catch an overrun instead of corrupting the heap
    xplane::place_int1<TM,TN,TK,WM,WN,F2>(derived.data(), hi, N, K);
    long guard = 0; for (size_t i = (size_t)N*K/8; i < derived.size(); ++i) if (derived[i]) ++guard;
    std::vector<int8_t> packed((size_t)N*K/8,0);
    for (int k=0;k<K;++k) for (int n=0;n<N;++n) if (hi[(size_t)k*N+n]&1){size_t i=(size_t)n*K+k; packed[i/8]|=int8_t(1<<(i%8));}
    std::vector<int8_t> ship(packed.size());
    preprocess_weights_for_mixed_gemm<false,256,0>(ship.data(),packed.data(),{(size_t)K,(size_t)N},QuantTypeClass::PACKED_INT1_WEIGHT_ONLY);
    { std::vector<int8_t> f(ship.size()); legacy::nfold_regroup_gmem(f.data(),ship.data(),{(size_t)K,(size_t)N},TN,TK,1); ship.swap(f); }
    size_t d=0; for (size_t i=0;i<ship.size();++i) if (derived[i]!=ship[i]) ++d;
    fprintf(stderr, "  N=%d K=%d : %zu / %zu differ, guard-bytes-touched=%ld  %s\n",
            N, K, d, ship.size(), guard, d ? "<-- DIFFERS" : "<-- identical");
  }
  return 0;
}
