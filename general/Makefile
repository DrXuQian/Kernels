CUDA_ROOT ?=
ifeq ($(origin NVCC), undefined)
ifneq ($(CUDA_ROOT),)
NVCC := $(CUDA_ROOT)/bin/nvcc
else
NVCC := nvcc
endif
endif

CFLAGS ?= -O2 -std=c++17 --expt-relaxed-constexpr -diag-suppress 3288
CUTLASS_CFLAGS ?= -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 3288 -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1
ARCH ?= -arch=sm_80
CUDA_CCCL_INC ?= $(if $(CUDA_ROOT),-I$(CUDA_ROOT)/targets/x86_64-linux/include/cccl,)
INCLUDES := -I../include
CUTLASS_DIR ?= ../third_party/cutlass
CUTLASS_INCLUDES := -I$(CUTLASS_DIR)/include

BINS := bench_layernorm bench_rmsnorm bench_cublas_gemm bench_cuda_core_gemv bench_cutlass_block_fp8_gemm

.PHONY: all clean

all: $(BINS)

bench_layernorm: bench_layernorm.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(CUDA_CCCL_INC) $(INCLUDES) $< -o $@

bench_rmsnorm: bench_rmsnorm.cu ../include/bench_timer.h
	$(NVCC) $(CFLAGS) $(ARCH) $(CUDA_CCCL_INC) $(INCLUDES) $< -o $@

bench_cublas_gemm: bench_cublas_gemm.cu ../include/bench_timer.h
	$(NVCC) $(CFLAGS) $(ARCH) $(CUDA_CCCL_INC) $(INCLUDES) $< -o $@ -lcublas -lcublasLt

bench_cuda_core_gemv: bench_cuda_core_gemv.cu ../include/bench_timer.h
	$(NVCC) $(CFLAGS) $(ARCH) $(CUDA_CCCL_INC) $(INCLUDES) $< -o $@

bench_cutlass_block_fp8_gemm: bench_cutlass_block_fp8_gemm.cu ../include/bench_timer.h
	$(NVCC) $(CUTLASS_CFLAGS) $(ARCH) $(CUDA_CCCL_INC) $(INCLUDES) $(CUTLASS_INCLUDES) $< -o $@

clean:
	rm -f $(BINS)
