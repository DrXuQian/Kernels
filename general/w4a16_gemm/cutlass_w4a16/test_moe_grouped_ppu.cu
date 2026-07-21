// STEP 2a gate: grouped mixed-input GEMM via the GroupScheduler, uniform m_per_expert.
// Proves the GroupProblemShape + GroupScheduler + mixed-input collective type stack compiles/runs (equivalent
// to step 1 but routed through the grouped kernel). Timing only. run: ./test_moe_grouped_ppu [E] [m] [n] [k] [gs]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "moe_grouped_ppu.cuh"

int main(int argc, char** argv) {
  int L = argc > 1 ? atoi(argv[1]) : 8;
  int m = argc > 2 ? atoi(argv[2]) : 512;
  int n = argc > 3 ? atoi(argv[3]) : 1024;
  int k = argc > 4 ? atoi(argv[4]) : 2048;
  int g = argc > 5 ? atoi(argv[5]) : 128;
  using half_t = cutlass::half_t; using int4_t = cutlass::int4b_t;
  const int scale_k = (k + g - 1) / g;

  cutlass::DeviceAllocation<half_t> A((size_t)L*m*k), scales((size_t)L*n*scale_k), D((size_t)L*m*n);
  cutlass::DeviceAllocation<int4_t> B((size_t)L*n*k);
  const size_t ws_bytes = (size_t)cutlass::ceil_div(m,16)*cutlass::ceil_div(n,64)*L*64;  // generous
  cutlass::DeviceAllocation<char>   ws(ws_bytes);

  // GroupProblemShape: uniform [m,n,k] per expert.
  using GS = moe_grouped_ppu::GroupShape;
  std::vector<GS> host_shapes(L, cute::make_shape(m, n, k));
  cutlass::DeviceAllocation<GS> dev_shapes(L);
  dev_shapes.copy_from_host(host_shapes.data());

  std::printf("MoE grouped (2a, uniform m): experts=%d m=%d n=%d k=%d gs=%d\n", L,m,n,k,g);
  auto launch = [&]{
    moe_grouped_ppu::filter_and_run<moe_grouped_ppu::QuantMode::FinegrainedScaleOnly, 64,64,128, 32,32, 3>(
        A.get(), B.get(), scales.get(), nullptr, D.get(), m, n, k, L, g,
        dev_shapes.get(), host_shapes.data(), ws.get(), ws_bytes, nullptr);
  };
  launch();
  std::printf("launch issued (compile + can_implement + run gate).\n");

  const int warmup = 10, iters = 50;
  for (int i=0;i<warmup;i++) launch();
  PpuTimer t; t.start(); for (int i=0;i<iters;i++) launch(); t.stop();
  double us = double(t.elapsed_millis())*1e3/iters;
  double tf = 2.0*L*m*n*k/(us*1e-6)/1e12;
  std::printf("  [MoE grouped 64x64x128/s3] %.2f us | %.1f TFLOP/s (%.1f%% MFU)\n", us, tf, 100.0*tf/500.0);
  return 0;
}
