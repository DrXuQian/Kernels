NVCC        ?= nvcc
ARCH        ?= -arch=sm_90a
ENABLE_BF16 ?= 1
CUDA_CCCL_INC ?=

ROOT_DIR    := ../..
AUX_DIR     := $(ROOT_DIR)/moe_w4a16/trtllm/auxiliary
CFLAGS      := -O2 -std=c++17 $(ARCH) $(CUDA_CCCL_INC) \
               -I$(ROOT_DIR)/include -I$(AUX_DIR)/include \
               --expt-relaxed-constexpr -DENABLE_BF16=$(ENABLE_BF16)

.PHONY: all clean

all: bench_finalize_moe_routing_study

bench_finalize_moe_routing_study: src/finalize_moe_routing_study.cu \
    $(ROOT_DIR)/include/bench_timer.h \
    $(AUX_DIR)/include/trtllm_aux_compat.h
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f bench_finalize_moe_routing_study
