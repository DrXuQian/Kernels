NVCC       := nvcc
CUTLASS    := third_party/cutlass
CFLAGS_BASE := -O2 -std=c++17 --expt-relaxed-constexpr -diag-suppress 3288
INC_KDA    := -Isrc -Isrc/kerutils/include -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include

ARCH_GENERIC := -arch=sm_80
ARCH_SM90    := -arch=sm_90a

BENCH_BINS := bench_conv1d_fwd bench_conv1d_update bench_gated_delta_net bench_kda_prefill

.PHONY: all clean

all: $(BENCH_BINS)

# ---- Dao-AILab/causal-conv1d (SM80+, 128-bit vectorized) ----
conv1d_fwd.o: src/causal_conv1d_fwd.cu src/causal_conv1d.h src/causal_conv1d_common.h src/static_switch.h
	$(NVCC) $(CFLAGS_BASE) $(ARCH_GENERIC) -Isrc -dc $< -o $@

conv1d_update.o: src/causal_conv1d_update.cu src/causal_conv1d.h src/causal_conv1d_common.h src/static_switch.h
	$(NVCC) $(CFLAGS_BASE) $(ARCH_GENERIC) -Isrc -dc $< -o $@

bench_conv1d_fwd.o: src/bench_conv1d_fwd.cu src/causal_conv1d.h
	$(NVCC) $(CFLAGS_BASE) $(ARCH_GENERIC) -Isrc -dc $< -o $@

bench_conv1d_update.o: src/bench_conv1d_update.cu src/causal_conv1d.h
	$(NVCC) $(CFLAGS_BASE) $(ARCH_GENERIC) -Isrc -dc $< -o $@

bench_conv1d_fwd: bench_conv1d_fwd.o conv1d_fwd.o
	$(NVCC) $(ARCH_GENERIC) $^ -o $@

bench_conv1d_update: bench_conv1d_update.o conv1d_update.o
	$(NVCC) $(ARCH_GENERIC) $^ -o $@

# ---- llama.cpp gated_delta_net (SM80+, recurrent) ----
bench_gated_delta_net: src/gated_delta_net.cu include/ggml_compat.h
	$(NVCC) $(CFLAGS_BASE) $(ARCH_GENERIC) -Iinclude -DBENCH $< -o $@

# ---- cuLA chunked KDA prefill (SM90 only, CUTLASS) ----
kda_safe_gate.o: src/kda/sm90/kda_fwd_sm90_safe_gate.cu
	$(NVCC) $(CFLAGS_BASE) $(ARCH_SM90) $(INC_KDA) -dc $< -o $@

kda_dispatch.o: src/kda/sm90/kda_fwd_sm90.cu
	$(NVCC) $(CFLAGS_BASE) $(ARCH_SM90) $(INC_KDA) -dc $< -o $@

bench_kda.o: src/bench_kda_prefill.cu src/kda/sm90/prefill_kernel.hpp
	$(NVCC) $(CFLAGS_BASE) $(ARCH_SM90) $(INC_KDA) -dc $< -o $@

bench_kda_prefill: bench_kda.o kda_dispatch.o kda_safe_gate.o
	$(NVCC) $(ARCH_SM90) $^ -o $@

clean:
	rm -f *.o $(BENCH_BINS)
