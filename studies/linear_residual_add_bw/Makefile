CUDA_ROOT ?=
ifeq ($(origin NVCC), undefined)
ifneq ($(CUDA_ROOT),)
NVCC := $(CUDA_ROOT)/bin/nvcc
else
NVCC := nvcc
endif
endif

ARCH ?= -arch=sm_90a
CFLAGS ?= -O3 -std=c++17 --expt-relaxed-constexpr

.PHONY: all clean

all: bench_residual_add_bw

bench_residual_add_bw: src/bench_residual_add_bw.cu
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

clean:
	rm -f bench_residual_add_bw *.o

