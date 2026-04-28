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

BINS := bench_layernorm

.PHONY: all clean

all: $(BINS)

bench_layernorm: bench_layernorm.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(CUDA_CCCL_INC) $< -o $@

clean:
	rm -f $(BINS)
