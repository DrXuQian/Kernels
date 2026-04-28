/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ONEFLOW_CORE_CUDA_LAYER_NORM_H_
#define ONEFLOW_CORE_CUDA_LAYER_NORM_H_

#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>

namespace oneflow {
namespace cuda {
namespace layer_norm {

constexpr int kWarpSize = 32;

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask, thread_group_width));
  }
  return val;
}

template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
  return a / b;
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

template<typename T>
__inline__ __device__ T Rsqrt(T x);

template<>
__inline__ __device__ float Rsqrt<float>(float x) {
  return rsqrt(x);
}

template<>
__inline__ __device__ double Rsqrt<double>(double x) {
  return rsqrt(x);
}

template<class Func>
inline cudaError_t GetNumBlocks(Func func, int64_t block_size, size_t dynamic_smem_size,
                                int64_t max_blocks, int64_t waves, int* num_blocks) {
  int dev;
  { cudaError_t err = cudaGetDevice(&dev); if (err != cudaSuccess) { return err; } }
  int sm_count;
  { cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev); if (err != cudaSuccess) { return err; } }
  int max_active_blocks;
  { cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, func, block_size, dynamic_smem_size); }
  *num_blocks = std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return cudaSuccess;
}

template<typename T>
struct DefaultComputeType { using type = T; };

template<>
struct DefaultComputeType<half> { using type = float; };

#if CUDA_VERSION >= 11000
template<>
struct DefaultComputeType<nv_bfloat16> { using type = float; };
#endif

template<typename T>
class HasCanPackAs {
  typedef char one;
  struct two { char x[2]; };
  template<typename C> static one test(decltype(&C::CanPackAs));
  template<typename C> static two test(...);
public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template<typename T>
typename std::enable_if<HasCanPackAs<T>::value == true, bool>::type CanPackAs(T t, size_t pack_size) {
  return t.CanPackAs(pack_size);
}

template<typename T>
typename std::enable_if<HasCanPackAs<T>::value == false, bool>::type CanPackAs(T t, size_t pack_size) {
  return true;
}

template<typename T, int N>
struct GetPackType { using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type; };

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {}
  PackType<T, N> storage;
  T elem[N];
};

template<typename SRC, typename DST>
struct DirectLoad {
  using LoadType = DST;
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  const SRC* src;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

template<typename T>
inline __device__ void WelfordCombine(T val, T* mean, T* m2, T* count) {
  *count += 1;
  T delta1 = val - *mean;
  *mean += Div(delta1, *count);
  T delta2 = val - *mean;
  *m2 += delta1 * delta2;
}

template<typename T>
inline __device__ void WelfordCombine(T b_mean, T b_m2, T b_count, T* mean, T* m2, T* count) {
  if (b_count == 0) { return; }
  T new_count = *count + b_count;
  T nb_over_n = Div(b_count, new_count);
  T delta = b_mean - *mean;
  *mean += delta * nb_over_n;
  *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
  *count = new_count;
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count) {
  *mean = thread_mean; *m2 = thread_m2; *count = thread_count;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count) {
  WelfordWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_count, mean, m2, count);
  *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

template<typename T>
__inline__ __device__ void WelfordBlockAllReduce(T thread_mean, T thread_m2, T thread_count,
                                                 T* result_mean, T* result_m2, T* result_count) {
  __shared__ T mean_shared[kWarpSize];
  __shared__ T m2_shared[kWarpSize];
  __shared__ T count_shared[kWarpSize];
  __shared__ T mean_result_broadcast;
  __shared__ T m2_result_broadcast;
  __shared__ T count_result_broadcast;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  T warp_mean = 0, warp_m2 = 0, warp_count = 0;
  WelfordWarpReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
  __syncthreads();
  if (lid == 0) { mean_shared[wid] = warp_mean; m2_shared[wid] = warp_m2; count_shared[wid] = warp_count; }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) { warp_mean = mean_shared[lid]; warp_m2 = m2_shared[lid]; warp_count = count_shared[lid]; }
    else { warp_mean = 0; warp_m2 = 0; warp_count = 0; }
    __syncwarp();
    T block_mean = 0, block_m2 = 0, block_count = 0;
    WelfordWarpReduce(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
    if (lid == 0) { mean_result_broadcast = block_mean; m2_result_broadcast = block_m2; count_result_broadcast = block_count; }
  }
  __syncthreads();
  *result_mean = mean_result_broadcast; *result_m2 = m2_result_broadcast; *result_count = count_result_broadcast;
}

// Forward declare all the dispatch functions (implementations follow)
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int max_cols_per_thread,
         int min_cols_per_thread, int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType* mean, ComputeType* inv_variance);

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int max_cols_per_thread,
         int min_cols_per_thread, int thread_group_width, int rows_per_access, bool padding>
inline cudaError_t LaunchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols,
                                           const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks = (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  { cudaError_t err = GetNumBlocks(LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, max_cols_per_thread, min_cols_per_thread, thread_group_width, rows_per_access, padding>, block_size, 0, num_blocks, waves, &grid_dim_x); if (err != cudaSuccess) { return err; } }
  LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, max_cols_per_thread, min_cols_per_thread, thread_group_width, rows_per_access, padding><<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int max_cols_per_thread,
         int min_cols_per_thread, int thread_group_width, int rows_per_access>
inline cudaError_t DispatchLayerNormWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols,
                                                    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols == max_cols_per_thread * thread_group_width) {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, max_cols_per_thread, max_cols_per_thread, thread_group_width, rows_per_access, false>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
  } else {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, max_cols_per_thread, min_cols_per_thread, thread_group_width, rows_per_access, true>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width) \
  else if (cols <= (thread_group_width)*pack_size) { \
    if (rows % 2 == 0) { return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, 0, thread_group_width, 2>(stream, load, store, rows, cols, epsilon, mean, inv_variance); } \
    else { return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, 0, thread_group_width, 1>(stream, load, store, rows, cols, epsilon, mean, inv_variance); } \
  }
  DEFINE_ONE_ELIF(4) DEFINE_ONE_ELIF(8) DEFINE_ONE_ELIF(16) DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col) \
  else if (cols <= (max_col)*kWarpSize) { return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, max_col, min_col, kWarpSize, 1>(stream, load, store, rows, cols, epsilon, mean, inv_variance); }
  DEFINE_ONE_ELIF(2, 1) DEFINE_ONE_ELIF(4, 2) DEFINE_ONE_ELIF(8, 4) DEFINE_ONE_ELIF(12, 8)
  DEFINE_ONE_ELIF(16, 12) DEFINE_ONE_ELIF(20, 16) DEFINE_ONE_ELIF(24, 20) DEFINE_ONE_ELIF(28, 24) DEFINE_ONE_ELIF(32, 28)
#undef DEFINE_ONE_ELIF
  else { return cudaErrorInvalidValue; }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width) \
  else if (cols <= (thread_group_width)*pack_size) { \
    if (rows % 2 == 0) { return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, 0, thread_group_width, 2>(stream, load, store, rows, cols, epsilon, mean, inv_variance); } \
    else { return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, 0, thread_group_width, 1>(stream, load, store, rows, cols, epsilon, mean, inv_variance); } \
  }
  DEFINE_ONE_ELIF(4) DEFINE_ONE_ELIF(8) DEFINE_ONE_ELIF(16) DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col) \
  else if ((cols <= (max_col)*kWarpSize) && (cols > (min_col)*kWarpSize)) { return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, max_col, min_col, kWarpSize, 1>(stream, load, store, rows, cols, epsilon, mean, inv_variance); }
  DEFINE_ONE_ELIF(4, 2) DEFINE_ONE_ELIF(8, 4) DEFINE_ONE_ELIF(12, 8) DEFINE_ONE_ELIF(16, 12)
  DEFINE_ONE_ELIF(20, 16) DEFINE_ONE_ELIF(24, 20) DEFINE_ONE_ELIF(28, 24) DEFINE_ONE_ELIF(32, 28)
#undef DEFINE_ONE_ELIF
  else { return cudaErrorInvalidValue; }
}

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
                         const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
    if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) && CanPackAs<STORE>(store, 2)) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 2>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 1>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                             const int64_t rows, const int64_t cols,
                                             const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  return DispatchLayerNormWarpImplPackSize<LOAD, STORE, ComputeType>()(stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

// Block shared memory implementation
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LayerNormBlockSMemImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                       const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  using LoadType = typename LOAD::LoadType;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<LoadType*>(shared_buf);
  const int tid = threadIdx.x;
  const int num_packs = static_cast<int>(cols) / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0, thread_m2 = 0, thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        WelfordCombine(static_cast<ComputeType>(pack[i]), &thread_mean, &thread_m2, &thread_count);
      }
    }
    ComputeType row_mean = 0, row_m2 = 0, row_count = 0;
    WelfordBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2, &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) { mean[row] = row_mean; inv_variance[row] = row_inv_var; }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      for (int i = 0; i < pack_size; ++i) { pack[i] = (static_cast<ComputeType>(buf[i * num_packs + pack_id]) - row_mean) * row_inv_var; }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
inline cudaError_t LaunchLayerNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, int smem,
                                                const int64_t rows, const int64_t cols, const double epsilon,
                                                ComputeType* mean, ComputeType* inv_variance) {
  constexpr int waves = 32;
  int grid_dim_x;
  { cudaError_t err = GetNumBlocks(LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>, block_size, smem, rows, waves, &grid_dim_x); if (err != cudaSuccess) { return err; } }
  LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size><<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename Func>
cudaError_t MaximizeDynamicSharedMemorySize(Func func, const int max_smem_size) {
  cudaFuncAttributes attr{};
  cudaError_t err = cudaFuncGetAttributes(&attr, func);
  if (err != cudaSuccess) { return err; }
  constexpr int reserved_smem = 1024;
  return cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_size - attr.sharedSizeBytes - reserved_smem);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline cudaError_t TryDispatchLayerNormBlockSMemImplBlockSize(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance, bool* success) {
  constexpr int block_size_conf_1 = 128, block_size_conf_2 = 256, block_size_conf_3 = 512, block_size_conf_4 = 1024;
  int dev = 0;
  { cudaError_t err = cudaGetDevice(&dev); if (err != cudaSuccess) { return err; } }
  int sm_count = 0;
  { cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev); if (err != cudaSuccess) { return err; } }
  static const bool max_smem_configed = [=]() {
    int max_smem_size = 0;
    cudaError_t err = cudaDeviceGetAttribute(&max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (err != cudaSuccess) { return false; }
    err = MaximizeDynamicSharedMemorySize(LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>, max_smem_size); if (err != cudaSuccess) { return false; }
    err = MaximizeDynamicSharedMemorySize(LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>, max_smem_size); if (err != cudaSuccess) { return false; }
    err = MaximizeDynamicSharedMemorySize(LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>, max_smem_size); if (err != cudaSuccess) { return false; }
    err = MaximizeDynamicSharedMemorySize(LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>, max_smem_size); if (err != cudaSuccess) { return false; }
    return true;
  }();
  const size_t smem = cols * sizeof(typename LOAD::LoadType);
  int max_active_blocks_conf_1;
  { cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_conf_1, LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>, block_size_conf_1, smem); if (err != cudaSuccess) { return err; } }
  if (max_active_blocks_conf_1 <= 0) { *success = false; return cudaSuccess; }
  int max_active_blocks_conf_4;
  { cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_conf_4, LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>, block_size_conf_4, smem); if (err != cudaSuccess) { return err; } }
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1 || (max_active_blocks_conf_4 > 0 && rows <= sm_count)) { *success = true; return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>(stream, load, store, smem, rows, cols, epsilon, mean, inv_variance); }
  int max_active_blocks_conf_3;
  { cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_conf_3, LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>, block_size_conf_3, smem); if (err != cudaSuccess) { return err; } }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1 || (max_active_blocks_conf_3 > 0 && rows <= sm_count)) { *success = true; return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>(stream, load, store, smem, rows, cols, epsilon, mean, inv_variance); }
  int max_active_blocks_conf_2;
  { cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_conf_2, LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>, block_size_conf_2, smem); if (err != cudaSuccess) { return err; } }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1 || (max_active_blocks_conf_2 > 0 && rows <= sm_count)) { *success = true; return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>(stream, load, store, smem, rows, cols, epsilon, mean, inv_variance); }
  *success = true;
  return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>(stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType>
struct TryDispatchLayerNormBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
                         const double epsilon, ComputeType* mean, ComputeType* inv_variance, bool* success) {
    if (cols % 4 == 0 && CanPackAs<LOAD>(load, 4) && CanPackAs<STORE>(store, 4)) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 4>(stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    } else if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) && CanPackAs<STORE>(store, 2)) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2>(stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    } else {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1>(stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t TryDispatchLayerNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                     const int64_t rows, const int64_t cols, const double epsilon,
                                                     ComputeType* mean, ComputeType* inv_variance, bool* success) {
  return TryDispatchLayerNormBlockSMemImplPackSize<LOAD, STORE, ComputeType>()(stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
}

// Block uncached implementation
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void __launch_bounds__(1024) LayerNormBlockUncachedImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                               const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  using LoadType = typename LOAD::LoadType;
  const int tid = threadIdx.x;
  const int num_packs = static_cast<int>(cols) / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0, thread_m2 = 0, thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
      for (int i = 0; i < pack_size; ++i) { WelfordCombine(static_cast<ComputeType>(pack[i]), &thread_mean, &thread_m2, &thread_count); }
    }
    ComputeType row_mean = 0, row_m2 = 0, row_count = 0;
    WelfordBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2, &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) { mean[row] = row_mean; inv_variance[row] = row_inv_var; }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[pack_size]; ComputeType dst_pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
      for (int i = 0; i < pack_size; ++i) { dst_pack[i] = (static_cast<ComputeType>(pack[i]) - row_mean) * row_inv_var; }
      store.template store<pack_size>(dst_pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline cudaError_t LaunchLayerNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols, const double epsilon,
                                                    ComputeType* mean, ComputeType* inv_variance) {
  constexpr int block_size = 1024, waves = 32;
  int grid_dim_x;
  { cudaError_t err = GetNumBlocks(LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>, block_size, 0, rows, waves, &grid_dim_x); if (err != cudaSuccess) { return err; } }
  LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size><<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
                         const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
    if (cols % 4 == 0 && CanPackAs<LOAD>(load, 4) && CanPackAs<STORE>(store, 4)) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 4>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) && CanPackAs<STORE>(store, 2)) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 2>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 1>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                      const int64_t rows, const int64_t cols, const double epsilon,
                                                      ComputeType* mean, ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImplPackSize<LOAD, STORE, ComputeType>()(stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

// Main dispatch
template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNorm(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
                  const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 1024) {
    return DispatchLayerNormWarpImpl<LOAD, STORE, ComputeType>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
  } else {
    bool dispatch_smem_impl_success;
    { cudaError_t err = TryDispatchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType>(stream, load, store, rows, cols, epsilon, mean, inv_variance, &dispatch_smem_impl_success); if (err != cudaSuccess) { return err; } }
    if (!dispatch_smem_impl_success) {
      return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNorm(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
                  const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType>(stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

// Warp implementation kernel body (must be after all helper functions)
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int max_cols_per_thread,
         int min_cols_per_thread, int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  using LoadType = typename LOAD::LoadType;
  constexpr int max_num_packs = max_cols_per_thread / pack_size;
  constexpr int min_num_packs = min_cols_per_thread / pack_size;
  ComputeType buf[rows_per_access][max_cols_per_thread];
  const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int64_t num_global_thread_group = gridDim.x * blockDim.y;
  const int64_t lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_mean[rows_per_access], thread_m2[rows_per_access], thread_count[rows_per_access];
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_mean[row_id] = 0; thread_m2[row_id] = 0; thread_count[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
      for (int pack_id = 0; pack_id < min_num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        LoadType pack[pack_size];
        load.template load<pack_size>(pack, row + row_id, col);
        for (int i = 0; i < pack_size; ++i) { row_buf[pack_id * pack_size + i] = static_cast<ComputeType>(pack[i]); WelfordCombine(row_buf[pack_id * pack_size + i], thread_mean + row_id, thread_m2 + row_id, thread_count + row_id); }
      }
      for (int pack_id = min_num_packs; pack_id < max_num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          LoadType pack[pack_size];
          load.template load<pack_size>(pack, row + row_id, col);
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_id * pack_size + i] = static_cast<ComputeType>(pack[i]); WelfordCombine(row_buf[pack_id * pack_size + i], thread_mean + row_id, thread_m2 + row_id, thread_count + row_id); }
        } else { for (int i = 0; i < pack_size; ++i) { row_buf[pack_id * pack_size + i] = 0; } }
      }
    }
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      int global_row_id = row + row_id;
      ComputeType* row_buf = buf[row_id];
      ComputeType warp_mean, warp_m2, warp_count;
      WelfordWarpAllReduce<ComputeType, thread_group_width>(thread_mean[row_id], thread_m2[row_id], thread_count[row_id], &warp_mean, &warp_m2, &warp_count);
      ComputeType row_mean = warp_mean;
      ComputeType row_variance = max(Div(warp_m2, warp_count), static_cast<ComputeType>(0.0));
      ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
      if (lane_id == 0) { mean[global_row_id] = row_mean; inv_variance[global_row_id] = row_inv_var; }
      for (int i = 0; i < max_cols_per_thread; ++i) { row_buf[i] = (row_buf[i] - row_mean) * row_inv_var; }
      for (int i = 0; i < min_num_packs; ++i) { const int col = (i * thread_group_width + lane_id) * pack_size; store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col); }
      for (int i = min_num_packs; i < max_num_packs; ++i) { const int col = (i * thread_group_width + lane_id) * pack_size; if (!padding || col < cols) { store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col); } }
    }
  }
}

}  // namespace layer_norm
}  // namespace cuda
}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_LAYER_NORM_H_
// Standalone LayerNorm benchmark (OneFlow kernel)
// Usage: ./bench_layernorm --batch 13824 --embed 1152 --dtype float16
//
// Build: nvcc -O2 -std=c++17 -arch=sm_80 bench_layernorm.cu -o bench_layernorm

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Include the OneFlow LayerNorm kernel (header-only)

#define CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)

template<typename T>
void run_layernorm(int batch, int embed) {
    using ComputeType = typename oneflow::cuda::layer_norm::DefaultComputeType<T>::type;

    size_t input_size = (size_t)batch * embed;
    size_t input_bytes = input_size * sizeof(T);
    size_t stats_bytes = batch * sizeof(ComputeType);

    // Allocate
    T *d_input, *d_output;
    ComputeType *d_mean, *d_inv_var;
    CHECK(cudaMalloc(&d_input, input_bytes));
    CHECK(cudaMalloc(&d_output, input_bytes));
    CHECK(cudaMalloc(&d_mean, stats_bytes));
    CHECK(cudaMalloc(&d_inv_var, stats_bytes));

    // Init: deterministic pattern
    std::vector<T> h_input(input_size);
    for (size_t i = 0; i < input_size; i++) {
        float val = 0.01f * ((int)(i % 101) - 50);  // [-0.5, 0.5]
        if constexpr (std::is_same_v<T, half>)
            h_input[i] = __float2half(val);
        else
            h_input[i] = static_cast<T>(val);
    }
    CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, input_bytes));
    CHECK(cudaMemset(d_mean, 0, stats_bytes));
    CHECK(cudaMemset(d_inv_var, 0, stats_bytes));

    // Setup load/store
    oneflow::cuda::layer_norm::DirectLoad<T, ComputeType> load(d_input, embed);
    oneflow::cuda::layer_norm::DirectStore<ComputeType, T> store(d_output, embed);

    double epsilon = 1e-5;

    // Single kernel launch
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    cudaError_t err = oneflow::cuda::layer_norm::DispatchLayerNorm<
        decltype(load), decltype(store), ComputeType>(
        0, load, store, batch, embed, epsilon, d_mean, d_inv_var);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    if (err != cudaSuccess) {
        fprintf(stderr, "LayerNorm dispatch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Read back and verify non-zero
    std::vector<T> h_output(input_size);
    CHECK(cudaMemcpy(h_output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost));

    int nonzero = 0;
    float min_val = 1e30f, max_val = -1e30f;
    for (size_t i = 0; i < input_size; i++) {
        float v;
        if constexpr (std::is_same_v<T, half>)
            v = __half2float(h_output[i]);
        else
            v = static_cast<float>(h_output[i]);
        if (v != 0.0f) nonzero++;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    // Bandwidth: read input + write output + read/write stats
    size_t total_bytes = input_bytes * 2 + stats_bytes * 2;
    float bw_gb = total_bytes / (ms / 1000.0f) / 1e9f;

    printf("batch=%d embed=%d dtype=%s\n", batch, embed,
           std::is_same_v<T, half> ? "float16" : std::is_same_v<T, float> ? "float32" : "unknown");
    printf("kernel time: %.3f us\n", ms * 1000.0f);
    printf("bandwidth:   %.1f GB/s\n", bw_gb);
    printf("output: nonzero=%d/%zu min=%.6f max=%.6f\n",
           nonzero, input_size, min_val, max_val);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_mean); cudaFree(d_inv_var);
}

int main(int argc, char** argv) {
    int batch = 13824;
    int embed = 1152;
    const char* dtype = "float16";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--embed") == 0 && i+1 < argc) embed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dtype") == 0 && i+1 < argc) dtype = argv[++i];
        else { fprintf(stderr, "Usage: %s [--batch N] [--embed N] [--dtype float16|float32]\n", argv[0]); return 1; }
    }

    if (strcmp(dtype, "float16") == 0 || strcmp(dtype, "fp16") == 0 || strcmp(dtype, "half") == 0) {
        run_layernorm<half>(batch, embed);
    } else if (strcmp(dtype, "float32") == 0 || strcmp(dtype, "fp32") == 0 || strcmp(dtype, "float") == 0) {
        run_layernorm<float>(batch, embed);
    } else {
        fprintf(stderr, "Unsupported dtype: %s\n", dtype);
        return 1;
    }
    return 0;
}
