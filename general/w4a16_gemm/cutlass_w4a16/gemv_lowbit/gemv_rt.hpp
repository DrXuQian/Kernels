// Minimal device-runtime shim, so ONE source builds and runs both locally under nvcc and on the box.
//
// This exists because the two runtimes are spelled differently and the difference is not cosmetic: every
// box-built source here uses hggcMalloc / hggcDeviceSynchronize (cutlass::DeviceAllocation in actlize is
// implemented on hggc*), while a locally-runnable source must use cuda*. A gate that only builds on the box
// cannot be run against a golden before shipping; a gate that only builds locally never validates the PPU.
// Five functions is the whole surface these harnesses need, so both are cheap to provide.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstddef>

#if defined(__HGGCCC__)
#include "cutlass/util/device_memory.h"     // actlize: implemented on hggcMalloc / hggcFree
#else
#include <cuda_runtime.h>
#endif

namespace ppu_gemv {

#if defined(__HGGCCC__)

inline void* rt_malloc(size_t bytes) {
  return static_cast<void*>(cutlass::device_memory::allocate<char>(bytes));
}
inline void rt_free(void* p) { cutlass::device_memory::free<char>(static_cast<char*>(p)); }
inline void rt_h2d(void* dst, void const* src, size_t bytes) {
  cutlass::device_memory::copy_to_device(static_cast<char*>(dst), static_cast<char const*>(src), bytes);
}
inline void rt_d2h(void* dst, void const* src, size_t bytes) {
  cutlass::device_memory::copy_to_host(static_cast<char*>(dst), static_cast<char const*>(src), bytes);
}
inline void rt_sync(const char* where) {
  hggcError_t e = hggcDeviceSynchronize();
  if (e != hggcSuccess) { std::printf("[gemv] device error at %s: %s\n", where, hggcGetErrorString(e)); std::exit(1); }
}

#else

inline void* rt_malloc(size_t bytes) {
  void* p = nullptr;
  cudaError_t e = cudaMalloc(&p, bytes);
  if (e != cudaSuccess) { std::printf("[gemv] cudaMalloc(%zu): %s\n", bytes, cudaGetErrorString(e)); std::exit(1); }
  return p;
}
inline void rt_free(void* p) { cudaFree(p); }
inline void rt_h2d(void* dst, void const* src, size_t bytes) {
  cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) { std::printf("[gemv] H2D: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
inline void rt_d2h(void* dst, void const* src, size_t bytes) {
  cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) { std::printf("[gemv] D2H: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
inline void rt_sync(const char* where) {
  cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) { std::printf("[gemv] device error at %s: %s\n", where, cudaGetErrorString(e)); std::exit(1); }
}

#endif

inline void rt_memset0(void* p, size_t bytes) {
  // No hggcMemset wrapper in cutlass's device_memory, and the one place this is needed (zeroing D so an
  // unwritten element is visible as zero rather than as garbage) is cheap to do from the host.
  void* z = std::calloc(bytes, 1);
  rt_h2d(p, z, bytes);
  std::free(z);
}

// RAII, so a harness with twenty allocations does not leak one per early return.
struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  DevBuf() = default;
  explicit DevBuf(size_t n) : p(n ? rt_malloc(n) : nullptr), bytes(n) {}
  DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) { if (p) rt_free(p); p = o.p; bytes = o.bytes; o.p = nullptr; o.bytes = 0; }
    return *this;
  }
  DevBuf(DevBuf const&) = delete;
  DevBuf& operator=(DevBuf const&) = delete;
  ~DevBuf() { if (p) rt_free(p); }
  template <typename T> T* as() const { return static_cast<T*>(p); }
  void from_host(void const* src) { if (p && bytes) rt_h2d(p, src, bytes); }
  void to_host(void* dst) const { if (p && bytes) rt_d2h(dst, p, bytes); }
};

}  // namespace ppu_gemv
