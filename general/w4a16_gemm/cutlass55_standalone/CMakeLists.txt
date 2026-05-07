cmake_minimum_required(VERSION 3.18)
project(cutlass55_standalone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GPU_ARCH "sm_90a" CACHE STRING "CUDA architecture, e.g. sm_90a")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "Disable CMake CUDA arch flags for raw -arch builds" FORCE)
set(_CUTLASS55_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} ")
string(REGEX REPLACE " +(-arch=sm_[0-9]+a?|--gpu-architecture=sm_[0-9]+a?|--cuda-gpu-arch=sm_[0-9]+a?)" " "
    _CUTLASS55_CUDA_FLAGS "${_CUTLASS55_CUDA_FLAGS}")
string(STRIP "${_CUTLASS55_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${GPU_ARCH}")
string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
message(STATUS "GPU_ARCH=${GPU_ARCH}; CUDA flags=${CMAKE_CUDA_FLAGS}")

if(NOT GPU_ARCH MATCHES "^sm_90")
    message(FATAL_ERROR "CUTLASS 55 standalone requires Hopper SM90/SM90a. Configure with -DGPU_ARCH=sm_90a.")
endif()

find_package(CUDAToolkit REQUIRED)

set(_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass")
if(NOT EXISTS "${_DEFAULT_CUTLASS_DIR}/include/cutlass/cutlass.h")
    set(_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass")
endif()
set(CUTLASS_DIR "${_DEFAULT_CUTLASS_DIR}" CACHE PATH "Path to CUTLASS")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found under ${CUTLASS_DIR}. Set CUTLASS_DIR.")
endif()

function(add_cutlass55_target target_name)
    add_executable(${target_name} cutlass55_gemm.cu)
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES OFF)
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
        ${CUTLASS_DIR}/examples/common
        ${CUTLASS_DIR}/examples/55_hopper_mixed_dtype_gemm
    )
    target_compile_definitions(${target_name} PRIVATE
        CUTLASS_ARCH_MMA_SM90_SUPPORTED=1
    )
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_link_libraries(${target_name} CUDA::cudart)
endfunction()

add_cutlass55_target(cutlass55_bf16_gemm)
add_cutlass55_target(cutlass55_fp16_gemm)
target_compile_definitions(cutlass55_fp16_gemm PRIVATE CUTLASS_USE_FP16=1)
