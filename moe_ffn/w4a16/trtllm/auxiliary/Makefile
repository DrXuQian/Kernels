NVCC        ?= nvcc
ARCH        ?= -arch=sm_90a
CUDA_CCCL_INC ?=
ENABLE_BF16 ?= 1
CFLAGS      := -O2 -std=c++17 $(ARCH) $(CUDA_CCCL_INC) -Iinclude -I../../../../include --expt-relaxed-constexpr -DENABLE_BF16=$(ENABLE_BF16)

BINS        := bench_custom_moe_routing bench_moe_align bench_expert_map bench_expand_input_rows bench_gated_activation bench_finalize_moe_routing

.PHONY: all clean

all: $(BINS)

bench_custom_moe_routing: src/custom_moe_routing.cu include/trtllm_aux_compat.h include/moe_topk_funcs.cuh
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_moe_align: src/moe_align.cu include/trtllm_aux_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_expert_map: src/moe_expert_map.cu include/trtllm_aux_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_expand_input_rows: src/moe_pipeline_kernels.cu include/trtllm_aux_compat.h
	$(NVCC) $(CFLAGS) -DBENCH_EXPAND $< -o $@

bench_gated_activation: src/moe_pipeline_kernels.cu include/trtllm_aux_compat.h
	$(NVCC) $(CFLAGS) -DBENCH_GATED $< -o $@

bench_finalize_moe_routing: src/moe_pipeline_kernels.cu include/trtllm_aux_compat.h
	$(NVCC) $(CFLAGS) -DBENCH_FINALIZE $< -o $@

clean:
	rm -f *.o $(BINS)
