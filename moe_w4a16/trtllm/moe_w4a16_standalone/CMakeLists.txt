cmake_minimum_required(VERSION 3.18)
project(moe_w4a16_standalone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# SM80 only by default (allow override for PTX JIT).
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

set(_CUDA_CCCL_INCLUDE_CANDIDATES
    "${CUDAToolkit_TARGET_DIR}/include/cccl"
    "$ENV{CUDA_ROOT}/targets/x86_64-linux/include/cccl"
)
set(CUDA_CCCL_INCLUDE_DIR "" CACHE PATH "Path to CUDA CCCL include directory")
foreach(_candidate IN LISTS _CUDA_CCCL_INCLUDE_CANDIDATES)
    if(NOT CUDA_CCCL_INCLUDE_DIR AND EXISTS "${_candidate}/cuda/std/utility")
        set(CUDA_CCCL_INCLUDE_DIR "${_candidate}" CACHE PATH "Path to CUDA CCCL include directory" FORCE)
    endif()
endforeach()
if(CUDA_CCCL_INCLUDE_DIR)
    message(STATUS "CUDA CCCL include: ${CUDA_CCCL_INCLUDE_DIR}")
else()
    message(WARNING "CUDA CCCL include directory was not found; CUTLASS C++ sources may fail to include cuda/std headers.")
endif()

set(_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass")
if(NOT EXISTS "${_DEFAULT_CUTLASS_DIR}/include/cutlass/cutlass.h")
    set(_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass")
endif()
set(CUTLASS_DIR "${_DEFAULT_CUTLASS_DIR}" CACHE PATH "Path to CUTLASS")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found under ${CUTLASS_DIR}. Set CUTLASS_DIR.")
endif()

set(MOE_W4A16_SOURCES
    cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
    cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint4.cu
    cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint4.cu
)

add_library(moe_w4a16_gemm STATIC ${MOE_W4A16_SOURCES})

set(MOE_W4A16_COMPILE_DEFS
    ENABLE_BF16
)

target_include_directories(moe_w4a16_gemm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm/cutlass_extensions/include
    ${CUTLASS_DIR}/include
    ${CUDA_CCCL_INCLUDE_DIR}
)

target_compile_definitions(moe_w4a16_gemm PUBLIC ${MOE_W4A16_COMPILE_DEFS})

target_compile_options(moe_w4a16_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(moe_w4a16_gemm
    CUDA::cudart
)

add_executable(test_moe_w4a16_gemm test/test_moe_w4a16_gemm.cu)

target_link_libraries(test_moe_w4a16_gemm
    moe_w4a16_gemm
    CUDA::cudart
)

target_compile_definitions(test_moe_w4a16_gemm PRIVATE
    ${MOE_W4A16_COMPILE_DEFS}
)

target_compile_options(test_moe_w4a16_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
