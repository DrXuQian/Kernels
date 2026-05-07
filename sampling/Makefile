CUDA_ROOT ?=
ifeq ($(origin NVCC), undefined)
ifneq ($(CUDA_ROOT),)
NVCC := $(CUDA_ROOT)/bin/nvcc
else
NVCC := nvcc
endif
endif

CFLAGS ?= -O2 -std=c++17 --expt-relaxed-constexpr -diag-suppress 3288 -diag-suppress 177
ARCH ?= -arch=sm_80
CUDA_CCCL_INC ?= $(if $(CUDA_ROOT),-I$(CUDA_ROOT)/targets/x86_64-linux/include/cccl,)
FLASHINFER_INCLUDE ?= third_party/flashinfer/include
INCLUDES := $(CUDA_CCCL_INC) -I$(FLASHINFER_INCLUDE) -I../include

BINS := bench_sampling

.PHONY: all clean

all: $(BINS)

bench_sampling: bench_sampling.cu ../include/bench_timer.h
	$(NVCC) $(CFLAGS) $(ARCH) $(INCLUDES) $< -o $@ -lcublas -lcurand

clean:
	rm -f $(BINS)
