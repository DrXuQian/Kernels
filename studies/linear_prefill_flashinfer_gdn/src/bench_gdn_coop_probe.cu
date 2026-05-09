// Probe whether a cooperative single-kernel prefix design is mechanically
// possible for the GDN study kernels.
//
// This does not launch the real GDN kernel cooperatively. It reports the real
// CUTLASS/FlashInfer kernel resource usage, then optionally launches a tiny
// cooperative_groups::grid_group dummy kernel with the same block size and
// dynamic shared-memory request. That answers whether a future fused
// transition/prefix/output kernel could legally run enough CTAs concurrently.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "flashinfer/flat/common.hpp"
#include "flashinfer/flat/hopper/kernel/flat_kernel_builder_delta_rule.hpp"
#include "state_only/flat_kernel_builder_delta_rule_state_only.hpp"

namespace cg = cooperative_groups;

namespace {

using T = nv_bfloat16;
using Element = flat::map_to_cutlass_t<T>;
using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
using Scheduler = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Layout = cute::tuple<int64_t, cute::_1, int32_t>;

template <bool InitStateFromInput, bool EnableCheckpointing>
using FullOptions = std::tuple<
    flat::kernel::Option<flat::kernel::Tag::kIsDeltaRule, cute::true_type>,
    flat::kernel::Option<flat::kernel::Tag::kIsGVA, cute::true_type>,
    flat::kernel::Option<flat::kernel::Tag::kNeedsBeta, cute::true_type>,
    flat::kernel::Option<flat::kernel::Tag::kNeedsAlpha, cute::true_type>,
    flat::kernel::Option<flat::kernel::Tag::kInitStateFromInput,
                         std::conditional_t<InitStateFromInput, cute::true_type,
                                            cute::false_type>>,
    flat::kernel::Option<flat::kernel::Tag::kEnableCheckpointing,
                         std::conditional_t<EnableCheckpointing, cute::true_type,
                                            cute::false_type>>>;

template <bool InitStateFromInput, bool EnableCheckpointing>
using FullKernel = typename flat::kernel::FlatBuilderDeltaRule<
    Element, float, float, TileShape, Layout, Layout, Layout, Layout, Scheduler,
    FullOptions<InitStateFromInput, EnableCheckpointing>>::Kernel;

template <bool EnableCheckpointing>
using StateOnlyOptions = std::tuple<
    flat::kernel::Option<flat::kernel::Tag::kIsDeltaRule, cute::true_type>,
    flat::kernel::Option<flat::kernel::Tag::kIsGVA, cute::true_type>,
    flat::kernel::Option<flat::kernel::Tag::kNeedsBeta, cute::true_type>,
    flat::kernel::Option<flat::kernel::Tag::kNeedsAlpha, cute::true_type>,
    flat::kernel::Option<flat::kernel::Tag::kInitStateFromInput, cute::false_type>,
    flat::kernel::Option<flat::kernel::Tag::kEnableCheckpointing,
                         std::conditional_t<EnableCheckpointing, cute::true_type,
                                            cute::false_type>>>;

template <bool EnableCheckpointing>
using StateOnlyKernel = typename flat::kernel::FlatBuilderDeltaRuleStateOnly<
    Element, float, float, TileShape, Layout, Layout, Layout, Layout, Scheduler,
    StateOnlyOptions<EnableCheckpointing>>::Kernel;

void check(cudaError_t err, char const* expr) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA %s failed: %s (%d)\n", expr, cudaGetErrorString(err),
            static_cast<int>(err));
    std::exit(1);
  }
}

__global__ void cooperative_probe_kernel(int* out) {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *out = gridDim.x;
  }
}

void try_dummy_cooperative_launch(char const* name, int block_threads, int smem_size,
                                  int sm_count) {
  int max_active = 0;
  cudaError_t err = cudaFuncSetAttribute(
      cooperative_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  if (err != cudaSuccess) {
    printf("  dummy_set_smem=%s (%d)\n", cudaGetErrorString(err), static_cast<int>(err));
    cudaGetLastError();
    return;
  }

  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active, cooperative_probe_kernel, block_threads, smem_size);
  if (err != cudaSuccess) {
    printf("  dummy_occupancy=%s (%d)\n", cudaGetErrorString(err), static_cast<int>(err));
    cudaGetLastError();
    return;
  }

  int max_grid = max_active * sm_count;
  printf("  dummy_coop_capacity: active_blocks_per_sm=%d max_grid=%d\n", max_active,
         max_grid);

  int* d_out = nullptr;
  check(cudaMalloc(&d_out, sizeof(int)), "cudaMalloc d_out");
  int targets[] = {64, 128, 192, 256, 320, 384, 512, 960};
  for (int grid : targets) {
    int zero = 0;
    check(cudaMemcpy(d_out, &zero, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy zero");
    void* args[] = {&d_out};
    err = cudaLaunchCooperativeKernel(reinterpret_cast<void*>(cooperative_probe_kernel),
                                      dim3(grid), dim3(block_threads), args, smem_size);
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
      int got = 0;
      check(cudaMemcpy(&got, d_out, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy got");
      printf("    launch grid=%-4d ok out=%d\n", grid, got);
    } else {
      printf("    launch grid=%-4d failed: %s (%d)\n", grid, cudaGetErrorString(err),
             static_cast<int>(err));
      cudaGetLastError();
    }
  }
  check(cudaFree(d_out), "cudaFree d_out");
  printf("  note: dummy launch only validates CUDA cooperative-launch residency for %s;\n",
         name);
  printf("        it is not a real GDN kernel launch or a CUTLASS cluster launch.\n");
}

template <typename Kernel>
void report_kernel(char const* name, int sm_count, bool launch_dummy) {
  constexpr int block_threads = Kernel::MaxThreadsPerBlock;
  constexpr int smem_size = Kernel::SharedStorageSize;

  printf("\n== %s ==\n", name);
  printf("  max_threads_per_block=%d shared_storage=%d bytes\n", block_threads,
         smem_size);

  cudaFuncAttributes attr{};
  cudaError_t err = cudaFuncGetAttributes(&attr, cutlass::device_kernel<Kernel>);
  if (err == cudaSuccess) {
    printf("  func_attrs: binary=%d ptx=%d const=%zu local=%zu max_threads=%d "
           "static_smem=%zu\n",
           attr.binaryVersion, attr.ptxVersion, attr.constSizeBytes, attr.localSizeBytes,
           attr.maxThreadsPerBlock, attr.sharedSizeBytes);
  } else {
    printf("  func_attrs failed: %s (%d)\n", cudaGetErrorString(err),
           static_cast<int>(err));
    cudaGetLastError();
  }

  err = cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  printf("  set_dynamic_smem=%s (%d)\n", cudaGetErrorString(err), static_cast<int>(err));
  if (err != cudaSuccess) {
    cudaGetLastError();
  }

  int active = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, cutlass::device_kernel<Kernel>, block_threads, smem_size);
  if (err == cudaSuccess) {
    int max_grid = active * sm_count;
    printf("  occupancy: active_blocks_per_sm=%d max_resident_grid=%d\n", active,
           max_grid);
    int targets[] = {64, 128, 192, 256, 320, 384, 512, 960};
    printf("  resident target grids:");
    for (int grid : targets) {
      printf(" %d:%s", grid, grid <= max_grid ? "yes" : "no");
    }
    printf("\n");
  } else {
    printf("  occupancy failed: %s (%d)\n", cudaGetErrorString(err),
           static_cast<int>(err));
    cudaGetLastError();
  }

  if (launch_dummy) {
    try_dummy_cooperative_launch(name, block_threads, smem_size, sm_count);
  }
}

void usage(char const* argv0) {
  printf("Usage: %s [--launch-dummy]\n", argv0);
  printf("Reports real FlashInfer/CUTLASS GDN kernel resource usage. With "
         "--launch-dummy, also tests a tiny cooperative grid-sync kernel using the "
         "same block size and dynamic shared memory.\n");
}

}  // namespace

int main(int argc, char** argv) {
  bool launch_dummy = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--launch-dummy") == 0) {
      launch_dummy = true;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      return 1;
    }
  }

  int device = 0;
  check(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  int cooperative = 0;
  int sm_count = 0;
  int optin_smem = 0;
  int max_smem = 0;
  check(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device),
        "cudaDevAttrCooperativeLaunch");
  check(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device),
        "cudaDevAttrMultiProcessorCount");
  check(cudaDeviceGetAttribute(&optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
        "cudaDevAttrMaxSharedMemoryPerBlockOptin");
  check(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device),
        "cudaDevAttrMaxSharedMemoryPerBlock");

  printf("device=%d name=%s cc=%d.%d sms=%d cooperative_launch=%d\n", device, prop.name,
         prop.major, prop.minor, sm_count, cooperative);
  printf("shared_memory: per_block=%d optin_per_block=%d\n", max_smem, optin_smem);

  report_kernel<FullKernel<false, false>>("full_gdn_original", sm_count, launch_dummy);
  report_kernel<FullKernel<false, true>>("full_gdn_checkpoint", sm_count, launch_dummy);
  report_kernel<FullKernel<true, false>>("full_gdn_init_state_split", sm_count,
                                         launch_dummy);
  report_kernel<StateOnlyKernel<false>>("state_only_transition", sm_count, launch_dummy);
  report_kernel<StateOnlyKernel<true>>("state_only_checkpoint", sm_count, launch_dummy);

  return 0;
}
