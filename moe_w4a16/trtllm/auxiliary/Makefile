NVCC        ?= nvcc
ARCH        ?= -arch=sm_90a
CUDA_CCCL_INC ?=
ENABLE_BF16 ?= 1
CFLAGS      := -O2 -std=c++17 $(ARCH) $(CUDA_CCCL_INC) -Iinclude -I../../../include --expt-relaxed-constexpr -DENABLE_BF16=$(ENABLE_BF16)

BINS        := bench_custom_moe_routing bench_moe_align

.PHONY: all clean

all: $(BINS)

bench_custom_moe_routing: src/custom_moe_routing.cu include/trtllm_aux_compat.h include/moe_topk_funcs.cuh
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_moe_align: src/moe_align.cu include/trtllm_aux_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

clean:
	rm -f *.o $(BINS)
