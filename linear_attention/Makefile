CUDA_ROOT  ?=
ifeq ($(origin NVCC), undefined)
ifneq ($(CUDA_ROOT),)
NVCC       := $(CUDA_ROOT)/bin/nvcc
else
NVCC       := nvcc
endif
endif
CUTLASS    := ../third_party/cutlass
CFLAGS     ?= -O2 -std=c++17 --expt-relaxed-constexpr -diag-suppress 3288

ARCH       ?= -arch=sm_80
ARCH_SM90  ?= -arch=sm_90a

CUDA_CCCL_INC ?= $(if $(CUDA_ROOT),-I$(CUDA_ROOT)/targets/x86_64-linux/include/cccl,)
INC        := $(CUDA_CCCL_INC) -Isrc -Iinclude -I../include
INC_GDN    := $(INC) -Isrc/flashinfer_gdn -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include -DFLAT_SM90A_ENABLED

BINS       := bench_conv1d_fwd bench_conv1d_update bench_gated_delta_net bench_fused_rms_norm_gate bench_gdn_prefill

.PHONY: all clean

all: $(BINS)

# --- Conv1d prefill (Dao-AILab, 128-bit vectorized) ---
conv1d_fwd.o: src/causal_conv1d_fwd.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(INC) -dc $< -o $@

bench_conv1d_fwd.o: src/bench_conv1d_fwd.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(INC) -dc $< -o $@

bench_conv1d_fwd: bench_conv1d_fwd.o conv1d_fwd.o
	$(NVCC) $(ARCH) $^ -o $@

# --- Conv1d decode (Dao-AILab, state update) ---
conv1d_update.o: src/causal_conv1d_update.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(INC) -dc $< -o $@

bench_conv1d_update.o: src/bench_conv1d_update.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(INC) -dc $< -o $@

bench_conv1d_update: bench_conv1d_update.o conv1d_update.o
	$(NVCC) $(ARCH) $^ -o $@

# --- Gated Delta Net recurrent (llama.cpp) ---
bench_gated_delta_net: src/gated_delta_net.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(INC) -DBENCH $< -o $@

# --- Fused RMSNorm + Sigmoid Gate ---
bench_fused_rms_norm_gate: src/fused_rms_norm_gate.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(INC) $< -o $@

# --- FlashInfer GDN prefill (SM90 only) ---
gdn_kernel_inst.o: src/flashinfer_gdn/kernel_inst.cu
	$(NVCC) $(CFLAGS) $(ARCH_SM90) $(INC_GDN) -dc $< -o $@

gdn_dispatch.o: src/flashinfer_gdn/dispatch.cu
	$(NVCC) $(CFLAGS) $(ARCH_SM90) $(INC_GDN) -dc $< -o $@

bench_gdn.o: src/flashinfer_gdn/bench_gdn_prefill.cu
	$(NVCC) $(CFLAGS) $(ARCH_SM90) $(INC_GDN) -dc $< -o $@

bench_gdn_prefill: bench_gdn.o gdn_dispatch.o gdn_kernel_inst.o
	$(NVCC) $(ARCH_SM90) $^ -o $@

clean:
	rm -f *.o $(BINS) bench_gdn_prefill
