cmake_minimum_required(VERSION 3.18)
project(fpA_intB_standalone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# GPU architecture selection.
# Use -DGPU_ARCH=sm_90a (or sm_80, sm_89, etc.) to set the target arch.
# This bypasses CMAKE_CUDA_ARCHITECTURES which doesn't support the 'a' suffix
# in CMake < 3.28.
set(GPU_ARCH "sm_90a" CACHE STRING "CUDA architecture (e.g. sm_80, sm_90a)")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "Disable CMake CUDA arch flags for raw -arch builds" FORCE)
set(_FPA_INTB_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} ")
string(REGEX REPLACE " +(-arch=sm_[0-9]+a?|--gpu-architecture=sm_[0-9]+a?|--cuda-gpu-arch=sm_[0-9]+a?)" " "
    _FPA_INTB_CUDA_FLAGS "${_FPA_INTB_CUDA_FLAGS}")
string(STRIP "${_FPA_INTB_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${GPU_ARCH}")
string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
message(STATUS "GPU_ARCH=${GPU_ARCH}; CUDA flags=${CMAKE_CUDA_FLAGS}")

# Auto-detect SM90 for COMPILE_HOPPER_TMA_GEMMS
if(GPU_ARCH MATCHES "^sm_90" OR CMAKE_CUDA_FLAGS MATCHES "(^| )-arch=sm_90a?($| )")
    set(ENABLE_SM90 TRUE)
else()
    set(ENABLE_SM90 FALSE)
endif()

find_package(CUDAToolkit REQUIRED)

set(_FPA_INTB_CUTLASS_CANDIDATES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/cutlass"
    "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass"
    "$ENV{HOME}/TensorRT-LLM/3rdparty/cutlass"
)
set(_FPA_INTB_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass")
foreach(_candidate IN LISTS _FPA_INTB_CUTLASS_CANDIDATES)
    if(EXISTS "${_candidate}/include/cutlass/cutlass.h")
        set(_FPA_INTB_DEFAULT_CUTLASS_DIR "${_candidate}")
        break()
    endif()
endforeach()
set(CUTLASS_DIR "${_FPA_INTB_DEFAULT_CUTLASS_DIR}" CACHE PATH "Path to CUTLASS")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found under ${CUTLASS_DIR}. Set CUTLASS_DIR.")
endif()

set(FPA_INTB_SOURCES
    cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
    cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fp16_int4_gemm_fg_scalebias.cu
    cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_gemm_sm80_wrappers.cu
    cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedTrue.cu
)

add_library(fpA_intB_gemm STATIC ${FPA_INTB_SOURCES})
set_property(TARGET fpA_intB_gemm PROPERTY CUDA_ARCHITECTURES OFF)
set_property(TARGET fpA_intB_gemm PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

set(FPA_INTB_COMPILE_DEFS
    ENABLE_BF16=0
    ENABLE_FP8=0
    ENABLE_FP4=0
)

# Enable Hopper fpA_intB kernels when building for SM90.
if(ENABLE_SM90)
    list(APPEND FPA_INTB_COMPILE_DEFS
        COMPILE_HOPPER_TMA_GEMMS=1
        CUTLASS_ARCH_MMA_SM90_SUPPORTED=1
    )
endif()

target_include_directories(fpA_intB_gemm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm/cutlass_extensions/include
    ${CUTLASS_DIR}/include
)

target_compile_definitions(fpA_intB_gemm PUBLIC ${FPA_INTB_COMPILE_DEFS})

target_compile_options(fpA_intB_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(fpA_intB_gemm
    CUDA::cudart
)

add_executable(test_fpA_intB_gemm test/test_fpA_intB_gemm.cu)
set_property(TARGET test_fpA_intB_gemm PROPERTY CUDA_ARCHITECTURES OFF)

target_link_libraries(test_fpA_intB_gemm
    fpA_intB_gemm
    CUDA::cudart
)

target_compile_definitions(test_fpA_intB_gemm PRIVATE
    ${FPA_INTB_COMPILE_DEFS}
)

target_compile_options(test_fpA_intB_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
