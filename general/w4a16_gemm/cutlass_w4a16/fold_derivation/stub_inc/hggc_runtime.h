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
