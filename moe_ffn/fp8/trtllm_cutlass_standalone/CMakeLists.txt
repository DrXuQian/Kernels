cmake_minimum_required(VERSION 3.18)
project(moe_fp8_trtllm_cutlass_standalone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GPU_ARCH "sm_90a" CACHE STRING "CUDA architecture, e.g. sm_90a")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "Disable CMake CUDA arch flags for raw -arch builds" FORCE)
set(_FP8_MOE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} ")
string(REGEX REPLACE " +(-arch=sm_[0-9]+a?|--gpu-architecture=sm_[0-9]+a?|--cuda-gpu-arch=sm_[0-9]+a?)" " "
    _FP8_MOE_CUDA_FLAGS "${_FP8_MOE_CUDA_FLAGS}")
string(STRIP "${_FP8_MOE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${GPU_ARCH}")
string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
message(STATUS "GPU_ARCH=${GPU_ARCH}; CUDA flags=${CMAKE_CUDA_FLAGS}")

if(NOT GPU_ARCH MATCHES "^sm_90")
    message(FATAL_ERROR "TRT-LLM CUTLASS FP8 block-scale MoE standalone requires Hopper SM90/SM90a.")
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

set(_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass")
if(NOT EXISTS "${_DEFAULT_CUTLASS_DIR}/include/cutlass/cutlass.h")
    set(_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/cutlass")
endif()
set(CUTLASS_DIR "${_DEFAULT_CUTLASS_DIR}" CACHE PATH "Path to CUTLASS")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found under ${CUTLASS_DIR}. Set CUTLASS_DIR.")
endif()

set(W4A16_TRTLLM_STANDALONE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../w4a16/trtllm/moe_w4a16_standalone"
    CACHE PATH "Existing local TensorRT-LLM standalone shim for common headers")

set(FP8_BLOCKSCALE_SOURCES
    cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
)

add_library(moe_fp8_blockscale_gemm STATIC ${FP8_BLOCKSCALE_SOURCES})
set_property(TARGET moe_fp8_blockscale_gemm PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(moe_fp8_blockscale_gemm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm
    ${W4A16_TRTLLM_STANDALONE_DIR}/cpp/include
    ${W4A16_TRTLLM_STANDALONE_DIR}/cpp
    ${W4A16_TRTLLM_STANDALONE_DIR}/cpp/tensorrt_llm
    ${W4A16_TRTLLM_STANDALONE_DIR}/cpp/tensorrt_llm/cutlass_extensions/include
    ${CUTLASS_DIR}/include
    ${CUDA_CCCL_INCLUDE_DIR}
)

target_compile_definitions(moe_fp8_blockscale_gemm PUBLIC
    ENABLE_BF16
    ENABLE_FP8
    ENABLE_FP8_BLOCK_SCALE
    COMPILE_HOPPER_TMA_GEMMS
)

target_compile_options(moe_fp8_blockscale_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(moe_fp8_blockscale_gemm
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
)

add_executable(bench_moe_fp8_blockscale_gemm test/bench_moe_fp8_blockscale_gemm.cu)
set_property(TARGET bench_moe_fp8_blockscale_gemm PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(bench_moe_fp8_blockscale_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tensorrt_llm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${W4A16_TRTLLM_STANDALONE_DIR}/cpp/include
    ${W4A16_TRTLLM_STANDALONE_DIR}/cpp
    ${W4A16_TRTLLM_STANDALONE_DIR}/cpp/tensorrt_llm
    ${CUTLASS_DIR}/include
    ${CUDA_CCCL_INCLUDE_DIR}
)

target_compile_definitions(bench_moe_fp8_blockscale_gemm PRIVATE
    ENABLE_BF16
    ENABLE_FP8
    ENABLE_FP8_BLOCK_SCALE
    COMPILE_HOPPER_TMA_GEMMS
)

target_compile_options(bench_moe_fp8_blockscale_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(bench_moe_fp8_blockscale_gemm
    moe_fp8_blockscale_gemm
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
)
