CUDA_ROOT  ?=
ifeq ($(origin NVCC), undefined)
ifneq ($(CUDA_ROOT),)
NVCC       := $(CUDA_ROOT)/bin/nvcc
else
NVCC       := nvcc
endif
endif
CUTLASS    := ../../../third_party/cutlass
ARCH       := -arch=sm_90a
CFLAGS     := -O2 -std=c++17 $(ARCH) --expt-relaxed-constexpr -diag-suppress 3288
CUDA_CCCL_INC ?= $(if $(CUDA_ROOT),-I$(CUDA_ROOT)/targets/x86_64-linux/include/cccl,)
INC        := $(CUDA_CCCL_INC) -I. -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include \
              -I../../../include -DFLAT_SM90A_ENABLED

BINS       := bench_gdn_prefill

.PHONY: all clean

all: $(BINS)

bench_gdn_prefill: bench_gdn_prefill_single_tu.cu dispatch.cu bench_gdn_prefill.cu
	$(NVCC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm -f *.o $(BINS)
