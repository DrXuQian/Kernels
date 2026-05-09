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

BINS := bench_gdn_tile_study bench_gdn_blockdv_study bench_gdn_splitseq_study
OBJS := bench_gdn_tile_study.o gdn_tile_study_dispatch.o
BLOCKDV_OBJS := bench_gdn_blockdv_study.o gdn_blockdv_study_dispatch.o
SPLITSEQ_OBJS := bench_gdn_splitseq_study.o gdn_splitseq_study_dispatch.o gdn_state_only_study_dispatch.o

.PHONY: all single_tu blockdv_single_tu splitseq_single_tu clean

all: $(BINS)

single_tu: bench_gdn_tile_study_single_tu

blockdv_single_tu: bench_gdn_blockdv_study_single_tu

splitseq_single_tu: bench_gdn_splitseq_study_single_tu

bench_gdn_tile_study.o: src/bench_gdn_tile_study.cu
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

gdn_tile_study_dispatch.o: src/gdn_tile_study_dispatch.cu src/gdn_tile_study_kernel.cuh
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

bench_gdn_tile_study: $(OBJS)
	$(NVCC) $(ARCH) $^ -o $@

bench_gdn_blockdv_study.o: src/bench_gdn_blockdv_study.cu
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

gdn_blockdv_study_dispatch.o: src/gdn_blockdv_study_dispatch.cu src/gdn_blockdv_study_kernel.cuh src/block_dv/flat_kernel_builder_delta_rule_blockdv.hpp src/block_dv/flat_collective_tma_warpspecialized_delta_rule_blockdv.hpp src/block_dv/flat_tile_scheduler_blockdv.hpp src/block_dv/flat_collective_load_blockdv.hpp src/block_dv/flat_collective_store_blockdv.hpp
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

bench_gdn_blockdv_study: $(BLOCKDV_OBJS)
	$(NVCC) $(ARCH) $^ -o $@

bench_gdn_splitseq_study.o: src/bench_gdn_splitseq_study.cu
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

gdn_splitseq_study_dispatch.o: src/gdn_splitseq_study_dispatch.cu
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

gdn_state_only_study_dispatch.o: src/gdn_state_only_study_dispatch.cu src/state_only/flat_kernel_builder_delta_rule_state_only.hpp src/state_only/flat_collective_tma_warpspecialized_delta_rule_state_only.hpp
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

bench_gdn_splitseq_study: $(SPLITSEQ_OBJS)
	$(NVCC) $(ARCH) $^ -o $@

bench_gdn_tile_study_single_tu: src/bench_gdn_tile_study_single_tu.cu src/bench_gdn_tile_study.cu src/gdn_tile_study_dispatch.cu src/gdn_tile_study_kernel.cuh
	$(NVCC) $(CFLAGS) $(INC) $< -o $@

bench_gdn_blockdv_study_single_tu: src/bench_gdn_blockdv_study_single_tu.cu src/bench_gdn_blockdv_study.cu src/gdn_blockdv_study_dispatch.cu src/gdn_blockdv_study_kernel.cuh src/block_dv/flat_kernel_builder_delta_rule_blockdv.hpp src/block_dv/flat_collective_tma_warpspecialized_delta_rule_blockdv.hpp src/block_dv/flat_tile_scheduler_blockdv.hpp src/block_dv/flat_collective_load_blockdv.hpp src/block_dv/flat_collective_store_blockdv.hpp
	$(NVCC) $(CFLAGS) $(INC) $< -o $@

bench_gdn_splitseq_study_single_tu: src/bench_gdn_splitseq_study_single_tu.cu src/bench_gdn_splitseq_study.cu src/gdn_splitseq_study_dispatch.cu src/gdn_state_only_study_dispatch.cu src/state_only/flat_kernel_builder_delta_rule_state_only.hpp src/state_only/flat_collective_tma_warpspecialized_delta_rule_state_only.hpp
	$(NVCC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm -f $(OBJS) $(BLOCKDV_OBJS) $(SPLITSEQ_OBJS) $(BINS) bench_gdn_tile_study_single_tu bench_gdn_blockdv_study_single_tu bench_gdn_splitseq_study_single_tu
