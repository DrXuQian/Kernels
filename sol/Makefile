CUDA_ROOT ?=
ifeq ($(origin NVCC), undefined)
ifneq ($(CUDA_ROOT),)
NVCC := $(CUDA_ROOT)/bin/nvcc
else
NVCC := nvcc
endif
endif

ARCH ?= -arch=sm_90a
CFLAGS ?= -O3 -std=c++17
BINS := read_heavy_bw copy_bw

.PHONY: all clean

all: $(BINS)

$(BINS): %: %.cu
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

clean:
	rm -f $(BINS)
