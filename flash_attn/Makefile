CUDA_ROOT ?=
ifeq ($(origin NVCC), undefined)
ifneq ($(CUDA_ROOT),)
NVCC := $(CUDA_ROOT)/bin/nvcc
else
NVCC := nvcc
endif
endif

CFLAGS ?= -O2 -std=c++17 --expt-relaxed-constexpr -diag-suppress 3288
ARCH ?= -arch=sm_80
CUDA_CCCL_INC ?= $(if $(CUDA_ROOT),-I$(CUDA_ROOT)/targets/x86_64-linux/include/cccl,)
INCLUDES := $(CUDA_CCCL_INC) -I../include

BINS := bench_rmsnorm

.PHONY: all clean

all: $(BINS)

# Shared RMSNorm, built into this category for attention layer benches.
bench_rmsnorm: ../general/bench_rmsnorm.cu ../include/bench_timer.h
	$(NVCC) $(CFLAGS) $(ARCH) $(INCLUDES) $< -o $@

clean:
	rm -f $(BINS)
