#pragma once
#include <cstdio>
#include <cstddef>
#include <cstdarg>
typedef int hggcError_t;
enum { hggcSuccess = 0 };
typedef void* hggcStream_t;
struct hggcLaunchAttributeValue { int x; };
struct hggcLaunchAttribute { int id; hggcLaunchAttributeValue val; };
struct hggcLaunchConfig_t { int gridDim; int blockDim; size_t dynamicSmemBytes; hggcStream_t stream; hggcLaunchAttribute* attrs; int numAttrs; };
enum { hggcLaunchAttributeProgrammaticStreamSerialization = 1 };
static inline const char* hggcGetErrorName(hggcError_t){ return ""; }
static inline const char* hggcGetErrorString(hggcError_t){ return ""; }
static inline hggcError_t hggcDeviceSynchronize(){ return 0; }
static inline hggcError_t hggcGetLastError(){ return 0; }
static inline hggcError_t hggcPeekAtLastError(){ return 0; }
static inline hggcError_t hggcMemsetAsync(void*,int,size_t,hggcStream_t){ return 0; }
template<typename... A> static inline hggcError_t hggcLaunchKernelEx(A...){ return 0; }

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
enum { hggcErrorUnknown = 1, hggcErrorInvalidValue = 2 };
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
