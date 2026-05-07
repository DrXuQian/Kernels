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

SUBDIRS := w4a16/vllm/marlin w4a16/vllm/auxiliary w4a16/trtllm/auxiliary
BINS := bench_rmsnorm bench_shared_expert

.PHONY: all clean $(SUBDIRS)

all: $(BINS) $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

# Shared RMSNorm, built into this category for FFN/MoE layer benches.
bench_rmsnorm: ../general/bench_rmsnorm.cu ../include/bench_timer.h
	$(NVCC) $(CFLAGS) $(ARCH) $(INCLUDES) $< -o $@

bench_shared_expert: bench_shared_expert.cu ../include/bench_timer.h
	$(NVCC) $(CFLAGS) $(ARCH) $(INCLUDES) $< -o $@ -lcublas

clean:
	rm -f $(BINS)
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
