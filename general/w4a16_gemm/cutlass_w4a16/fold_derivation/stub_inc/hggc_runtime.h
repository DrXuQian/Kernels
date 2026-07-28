#pragma once
#include <cstdio>
#include <cstddef>
#include <cstdarg>
// A DISTINCT type, not `int`. cutlass/core_io.h defines operator<<(ostream&, hggcError_t); with the typedef to int
// that overload competed with ostream::operator<<(int) and every `std::cerr << <some int>` in CUTLASS_PPU_CHECK became
// ambiguous -- three "more than one operator <<" errors that I had misread as a template-parse artifact.
enum class hggcError_t : int { hggcSuccess_ = 0 };
inline constexpr hggcError_t hggcSuccess = hggcError_t::hggcSuccess_;
inline bool operator==(hggcError_t a, int b) { return int(a) == b; }
inline bool operator!=(hggcError_t a, int b) { return int(a) != b; }
typedef struct CUstream_st* hggcStream_t;
struct hggcLaunchAttributeValue { int x; };
struct hggcLaunchAttribute { int id; hggcLaunchAttributeValue val; };
struct hggcLaunchConfig_t { int gridDim; int blockDim; size_t dynamicSmemBytes; hggcStream_t stream; hggcLaunchAttribute* attrs; int numAttrs; };
enum { hggcLaunchAttributeProgrammaticStreamSerialization = 1 };
static inline const char* hggcGetErrorName(hggcError_t){ return ""; }
static inline const char* hggcGetErrorString(hggcError_t){ return ""; }
static inline hggcError_t hggcDeviceSynchronize(){ return hggcSuccess; }
static inline hggcError_t hggcGetLastError(){ return hggcSuccess; }
static inline hggcError_t hggcPeekAtLastError(){ return hggcSuccess; }
static inline hggcError_t hggcMemsetAsync(void*,int,size_t,hggcStream_t){ return hggcSuccess; }
template<typename... A> static inline hggcError_t hggcLaunchKernelEx(A...){ return hggcSuccess; }

// -- host-only stubs so unfused_weight_dequantize.hpp (via helper.h) compiles for the ground-truth extractor.
//    Nothing here runs; l7_groundtruth.cu only calls the CPU relayout functions.
typedef struct hggcEvent_st* hggcEvent_t;
inline hggcError_t hggcEventCreate(hggcEvent_t*) { return hggcSuccess; }
inline hggcError_t hggcEventDestroy(hggcEvent_t) { return hggcSuccess; }
inline hggcError_t hggcEventRecord(hggcEvent_t, hggcStream_t = nullptr) { return hggcSuccess; }
inline hggcError_t hggcEventSynchronize(hggcEvent_t) { return hggcSuccess; }
inline hggcError_t hggcEventElapsedTime(float* ms, hggcEvent_t, hggcEvent_t) { if (ms) *ms = 0.f; return hggcSuccess; }

// -- more host-only stubs, so `nvcc -cuda` can run the FRONT END on the harnesses locally. This is what turns
//    "the box build finds my typos" into "a local check finds them" -- it caught `bad` used before declaration
//    only after a round trip to ppu001, which was avoidable.
inline constexpr hggcError_t hggcErrorUnknown      = hggcError_t(1);
inline constexpr hggcError_t hggcErrorInvalidValue = hggcError_t(2);
typedef int hggcMemcpyKind;
enum { hggcMemcpyHostToDevice = 1, hggcMemcpyDeviceToHost = 2, hggcMemcpyDeviceToDevice = 3,
       hggcMemcpyHostToHost = 4, hggcMemcpyDefault = 5 };
inline hggcError_t hggcMalloc(void** p, size_t n) { *p = ::operator new(n); return hggcSuccess; }
inline hggcError_t hggcFree(void* p) { ::operator delete(p); return hggcSuccess; }
inline hggcError_t hggcMemcpy(void*, const void*, size_t, hggcMemcpyKind) { return hggcSuccess; }
inline hggcError_t hggcMemcpyAsync(void*, const void*, size_t, hggcMemcpyKind, hggcStream_t = nullptr) { return hggcSuccess; }
inline hggcError_t hggcMemset(void*, int, size_t) { return hggcSuccess; }
inline hggcError_t hggcStreamCreate(hggcStream_t*) { return hggcSuccess; }
inline hggcError_t hggcStreamDestroy(hggcStream_t) { return hggcSuccess; }
inline hggcError_t hggcStreamSynchronize(hggcStream_t) { return hggcSuccess; }
inline hggcError_t hggcGetDevice(int* d) { if (d) *d = 0; return hggcSuccess; }
inline hggcError_t hggcSetDevice(int) { return hggcSuccess; }
typedef int hggcDeviceAttr_t;
enum { hggcDevAttrMultiProcessorCount = 1 };
inline hggcError_t hggcDeviceGetAttribute(int* v, hggcDeviceAttr_t, int) { if (v) *v = 64; return hggcSuccess; }

// -- enough of the launch/attribute API that cutlass/util and gemm_universal_* parse. Without these the front end
//    emits a large, file-independent noise set, and a noise-pattern filter then has to be so loose that it hides real
//    errors -- exactly how a TM=16/WM=32 template failure slipped past syntax_check.sh.
struct CUstream_st;
enum { hggcFuncAttributeMaxDynamicSharedMemorySize = 1, hggcOccupancyDisableCachingOverride = 2 };
typedef unsigned long long HGdeviceptr;
inline hggcError_t hggcFuncSetAttribute(const void*, int, int) { return hggcSuccess; }
inline hggcError_t hggcDeviceSetLimit(int, size_t) { return hggcSuccess; }
inline hggcError_t hggcOccupancyMaxActiveBlocksPerMultiprocessor(int* n, const void*, int, size_t) {
  if (n) *n = 1; return hggcSuccess; }
enum { HGGC_SUCCESS = 0 };
// VARIADIC on purpose. Fixed arity here silently amputates the local check: the real signatures take (id, threads[, x]),
// so a 2- or 3-argument call failed with "too many arguments", the collective it lives in never instantiated, and
// EVERY static error downstream of CollectiveMma became invisible locally. Two compile errors reached the box that
// way (a folded gB2 cut with the unfolded TileShape, and plane 1's k coordinate fed to plane 2).
template <class... Ts> inline void __ppu_barrier_arrive(Ts...) {}
template <class... Ts> inline void __ppu_barrier_sync(Ts...) {}

// -- driver-level symbols cutlass/workspace.h uses. Missing, they cost five errors and (worse) pushed the front end
//    past its instantiation budget, which is how the whole mainloop stayed invisible locally.
typedef int HGresult;
inline HGresult hgMemsetD8Async (unsigned long long, unsigned char,  size_t, hggcStream_t = nullptr) { return HGGC_SUCCESS; }
inline HGresult hgMemsetD16Async(unsigned long long, unsigned short, size_t, hggcStream_t = nullptr) { return HGGC_SUCCESS; }
inline HGresult hgMemsetD32Async(unsigned long long, unsigned int,   size_t, hggcStream_t = nullptr) { return HGGC_SUCCESS; }
inline HGresult hgGetErrorString(HGresult, const char** s) { if (s) *s = ""; return HGGC_SUCCESS; }
