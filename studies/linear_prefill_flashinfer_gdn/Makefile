CUDA_ROOT ?=
NVCC ?= $(if $(CUDA_ROOT),$(CUDA_ROOT)/bin/nvcc,nvcc)
CUTLASS_DIR ?= ../../third_party/cutlass
GPU_ARCH ?= sm_90a

ARCH := -arch=$(GPU_ARCH)
CFLAGS := -O2 -std=c++17 $(ARCH) --expt-relaxed-constexpr -diag-suppress 3288 -diag-suppress 177

ifneq ($(CUDA_ROOT),)
CCCL_CANDIDATE := $(CUDA_ROOT)/targets/x86_64-linux/include/cccl
ifneq ($(wildcard $(CCCL_CANDIDATE)/cuda/std/utility),)
CCCL_INC := -I$(CCCL_CANDIDATE)
endif
endif

INC := $(CCCL_INC) \
       -Isrc \
       -I../../linear_attn/src/flashinfer_gdn \
       -I$(CUTLASS_DIR)/include \
       -I$(CUTLASS_DIR)/tools/util/include \
       -I../../include \
       -DFLAT_SM90A_ENABLED

BINS := bench_gdn_tile_study
OBJS := bench_gdn_tile_study.o gdn_tile_study_dispatch.o

.PHONY: all clean

all: $(BINS)

bench_gdn_tile_study.o: src/bench_gdn_tile_study.cu
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

gdn_tile_study_dispatch.o: src/gdn_tile_study_dispatch.cu src/gdn_tile_study_kernel.cuh
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

bench_gdn_tile_study: $(OBJS)
	$(NVCC) $(ARCH) $^ -o $@

clean:
	rm -f $(OBJS) $(BINS)
