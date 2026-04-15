NVCC       := nvcc
ARCH       := -arch=sm_120
CFLAGS     := -O2 -std=c++17 $(ARCH) -Iinclude --expt-relaxed-constexpr
LDFLAGS    := -lcudart

# Kernel source files (each is one kernel)
K1_SRC := src/causal_conv1d_fwd.cu
K2_SRC := src/causal_conv1d_update.cu
K3_SRC := src/chunk_gated_delta_rule.cu
K4_SRC := src/fused_recurrent.cu
K5_SRC := src/fused_rms_norm_gate.cu

K1_OBJ := causal_conv1d_fwd.o
K2_OBJ := causal_conv1d_update.o
K3_OBJ := chunk_gated_delta_rule.o
K4_OBJ := fused_recurrent.o
K5_OBJ := fused_rms_norm_gate.o

KERN_OBJS  := $(K1_OBJ) $(K2_OBJ) $(K3_OBJ) $(K4_OBJ) $(K5_OBJ)

# Bench binaries — same .cu compiled with -DBENCH to include main()
BENCH_BINS := bench_k1 bench_k2 bench_k3 bench_k4 bench_k5

.PHONY: all bench clean

all: deltanet_test
bench: $(BENCH_BINS)

# ---- Kernel objects (library mode, no main) ----
$(K1_OBJ): $(K1_SRC) include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

$(K2_OBJ): $(K2_SRC) include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

$(K3_OBJ): $(K3_SRC) include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

$(K4_OBJ): $(K4_SRC) include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

$(K5_OBJ): $(K5_SRC) include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

# ---- Main test binary (links all kernels) ----
main.o: main.cu include/deltanet.h include/naive_reference.h include/bench_utils.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

deltanet_test: main.o $(KERN_OBJS)
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS)

# ---- Bench binaries: same source + -DBENCH → standalone executable ----
bench_k1: $(K1_SRC) include/deltanet.h include/bench_utils.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@ $(LDFLAGS)

bench_k2: $(K2_SRC) include/deltanet.h include/bench_utils.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@ $(LDFLAGS)

bench_k3: $(K3_SRC) include/deltanet.h include/bench_utils.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@ $(LDFLAGS)

bench_k4: $(K4_SRC) include/deltanet.h include/bench_utils.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@ $(LDFLAGS)

bench_k5: $(K5_SRC) include/deltanet.h include/bench_utils.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o deltanet_test $(BENCH_BINS)
