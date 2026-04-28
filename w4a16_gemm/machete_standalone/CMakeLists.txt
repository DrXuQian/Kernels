cmake_minimum_required(VERSION 3.18)
project(machete_standalone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GPU_ARCH "sm_90a" CACHE STRING "CUDA architecture, e.g. sm_90a")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "Disable CMake CUDA arch flags for raw -arch builds" FORCE)
set(_MACHETE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} ")
string(REGEX REPLACE " +(-arch=sm_[0-9]+a?|--gpu-architecture=sm_[0-9]+a?|--cuda-gpu-arch=sm_[0-9]+a?)" " "
    _MACHETE_CUDA_FLAGS "${_MACHETE_CUDA_FLAGS}")
string(STRIP "${_MACHETE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${GPU_ARCH}")
string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
message(STATUS "GPU_ARCH=${GPU_ARCH}; CUDA flags=${CMAKE_CUDA_FLAGS}")

if(NOT GPU_ARCH MATCHES "^sm_90")
    message(FATAL_ERROR "Machete requires Hopper SM90/SM90a. Configure with -DGPU_ARCH=sm_90a.")
endif()

find_package(CUDAToolkit REQUIRED)

set(_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass")
if(NOT EXISTS "${_DEFAULT_CUTLASS_DIR}/include/cutlass/cutlass.h")
    set(_DEFAULT_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/cutlass")
endif()
set(CUTLASS_DIR "${_DEFAULT_CUTLASS_DIR}" CACHE PATH "Path to CUTLASS")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found under ${CUTLASS_DIR}. Set CUTLASS_DIR.")
endif()

add_library(machete_gemm STATIC
    cpp/quantization/machete/cutlass55_backend.cu
    cpp/quantization/machete/machete_standalone_gemm.cu
)
set_property(TARGET machete_gemm PROPERTY CUDA_ARCHITECTURES OFF)
set_property(TARGET machete_gemm PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_include_directories(machete_gemm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

target_compile_definitions(machete_gemm PUBLIC
    CUTLASS_ARCH_MMA_SM90_SUPPORTED=1
)

target_compile_options(machete_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(machete_gemm CUDA::cudart)

add_executable(test_machete_gemm test/test_machete_gemm.cu)
set_property(TARGET test_machete_gemm PROPERTY CUDA_ARCHITECTURES OFF)

target_link_libraries(test_machete_gemm machete_gemm CUDA::cudart)
target_compile_definitions(test_machete_gemm PRIVATE CUTLASS_ARCH_MMA_SM90_SUPPORTED=1)
target_compile_options(test_machete_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
