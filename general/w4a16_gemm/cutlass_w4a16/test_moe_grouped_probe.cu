// ROUTING PROBE for the grouped mixed-input kernel. Run with MOEG_PROBE=1 to make the kernel skip the GEMM
// and write (expert+1) into every output element of each tile it processes. Then this decodes, per D-plane e,
// what landed there -- pinning down the multi-expert bug deterministically:
//
//   plane e all == e+1        -> scheduler issued expert e's tiles AND epilogue wrote them to plane e (GOOD)
//   plane e all == 0          -> no tile ever wrote plane e (scheduler never produced L_idx==e, or grid too small)
//   plane e has mixed tags    -> epilogue L-slice wrong (tiles of several experts land in the same plane)
//   plane e uniformly == c+1  -> every tile that should be expert e was labeled expert c (L_idx assignment off)
//
// This isolates scheduler + epilogue ROUTING from B/scale slicing and the mma. B/scale are left uninitialized
// (the probe path never reads their data; load_init only needs valid pointers).
// run: MOEG_PROBE=1 ./test_moe_grouped_probe [L] [m_base] [ragged?]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "moe_grouped_ppu.cuh"

#include "cutlass/util/packed_stride.hpp"
using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using GS      = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;

int main(int argc, char** argv) {
  const int  L      = argc > 1 ? atoi(argv[1]) : 4;
  const int  Mb     = argc > 2 ? atoi(argv[2]) : 128;
  const bool ragged = argc > 3;
  const int  N = 1024, K = 2048, gs = 128;
  const int  scale_k = K / gs;
  if (!std::getenv("MOEG_PROBE")) std::printf("WARNING: MOEG_PROBE not set -> this runs the real GEMM, not the probe.\n");

  std::vector<int> me(L), offs(L);
  int total=0, Mmax=0;
  for (int e=0;e<L;e++){ me[e] = ragged ? Mb*((e%4)+1) : Mb; offs[e]=total; total+=me[e]; Mmax=std::max(Mmax,me[e]); }

  // Uninitialized device buffers -- the probe path ignores the data (only needs valid pointers). D contiguous.
  cutlass::DeviceAllocation<half_t> A((size_t)total*K), scales((size_t)L*N*scale_k), D((size_t)total*N);
  cutlass::DeviceAllocation<int4_t> B((size_t)L*N*K);
  { std::vector<half_t> z((size_t)total*N, half_t(0.f)); D.copy_from_host(z.data()); }   // zero D so "unwritten" reads as 0

  std::vector<GS> rshapes(L); for(int e=0;e<L;e++) rshapes[e]=cute::make_shape(me[e],N,K);
  cutlass::DeviceAllocation<GS>  rdev(L);   rdev.copy_from_host(rshapes.data());
  cutlass::DeviceAllocation<int> offdev(L); offdev.copy_from_host(offs.data());
  std::vector<half_t*> pdh(L); std::vector<DStride> sdh(L); std::vector<int> gmh(L);
  for (int e=0;e<L;e++){ pdh[e]=D.get()+(size_t)offs[e]*N; sdh[e]=cutlass::make_cute_packed_stride(DStride{}, cute::make_shape(me[e],N,1)); gmh[e]=me[e]; }
  cutlass::DeviceAllocation<half_t*> pd(L); pd.copy_from_host(pdh.data());
  cutlass::DeviceAllocation<DStride> sd(L); sd.copy_from_host(sdh.data());
  cutlass::DeviceAllocation<int>     gm(L); gm.copy_from_host(gmh.data());
  const size_t ws = (size_t)cutlass::ceil_div(Mmax,16)*cutlass::ceil_div(N,64)*(size_t)L*64;
  cutlass::DeviceAllocation<char> wsr(ws);

  std::printf("probe: L=%d %s Mb=%d N=%d K=%d gs=%d total=%d Mmax=%d  (expect plane e uniformly == %s)\n",
              L, ragged?"ragged":"uniform", Mb, N, K, gs, total, Mmax, "e+1");
  moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64,64,128, 32,32, 3>(
      A.get(), B.get(), scales.get(), nullptr, pd.get(), sd.get(), gm.get(), Mmax, N, K, L, gs,
      rdev.get(), rshapes.data(), ragged?offdev.get():nullptr, wsr.get(), ws, nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());

  std::vector<half_t> hD((size_t)total*N); D.copy_to_host(hD.data());

  // Decode each D-plane: histogram of the integer tags found in the VALID (me[e] x N) region.
  bool all_good = true;
  for (int e=0;e<L;e++){
    int cnt_expected=0, cnt_zero=0, cnt_other=0, other_example=-1;
    for (int i=0;i<me[e];i++)
      for (int j=0;j<N;j++){
        int v = (int)std::lround((double)float(hD[((size_t)offs[e] + i)*N + j]));   // contiguous row offs[e]+i
        if      (v == e+1) ++cnt_expected;
        else if (v == 0)   ++cnt_zero;
        else { ++cnt_other; if (other_example<0) other_example=v; }
      }
    int tot = me[e]*N;
    const char* verdict = (cnt_expected==tot) ? "GOOD (all e+1)"
                        : (cnt_zero==tot)      ? "UNWRITTEN (all 0) -> scheduler never issued this expert"
                        : (cnt_other>0 && other_example>0) ? "WRONG-TAG -> some tiles labeled a different expert"
                        :                        "MIXED -> epilogue wrote several experts into this plane";
    if (cnt_expected != tot) all_good=false;
    std::printf("  plane e=%d: expected(%d)=%d zero=%d other=%d (first other tag=%d) / %d -> %s\n",
                e, e+1, cnt_expected, cnt_zero, cnt_other, other_example, tot, verdict);
  }
  std::printf("ROUTING: %s\n", all_good ? "ALL PLANES GOOD -> bug is in data-slice/compute, NOT routing"
                                        : "ROUTING BROKEN -> see per-plane verdicts above");
  return 0;
}
