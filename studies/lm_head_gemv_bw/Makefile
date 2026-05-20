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

all: bench_lm_head_gemv

bench_lm_head_gemv: src/bench_lm_head_gemv.cu
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@ -lcublas

clean:
	rm -f bench_lm_head_gemv *.o
