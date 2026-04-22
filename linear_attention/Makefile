NVCC       := nvcc
CUTLASS    := ../third_party/cutlass
CFLAGS     := -O2 -std=c++17 --expt-relaxed-constexpr -diag-suppress 3288

ARCH       := -arch=sm_80
ARCH_SM90  := -arch=sm_90a

INC        := -Isrc -Iinclude
INC_KDA    := $(INC) -Isrc/kerutils/include -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include -DCULA_SM90A_ENABLED

BINS       := bench_conv1d_fwd bench_conv1d_update bench_gated_delta_net bench_kda_prefill

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

# --- cuLA chunked KDA prefill (SM90 only) ---
kda_safe_gate.o: src/kda/sm90/kda_fwd_sm90_safe_gate.cu
	$(NVCC) $(CFLAGS) $(ARCH_SM90) $(INC_KDA) -dc $< -o $@

kda_dispatch.o: src/kda/sm90/kda_fwd_sm90.cu
	$(NVCC) $(CFLAGS) $(ARCH_SM90) $(INC_KDA) -dc $< -o $@

bench_kda.o: src/bench_kda_prefill.cu
	$(NVCC) $(CFLAGS) $(ARCH_SM90) $(INC_KDA) -dc $< -o $@

bench_kda_prefill: bench_kda.o kda_dispatch.o kda_safe_gate.o
	$(NVCC) $(ARCH_SM90) $^ -o $@

clean:
	rm -f *.o $(BINS)
