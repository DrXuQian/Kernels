NVCC       := nvcc
ARCH       := -arch=sm_120
CFLAGS     := -O2 -std=c++17 $(ARCH) -Iinclude --expt-relaxed-constexpr
LDFLAGS    := -lcudart

# 5 kernel object files — each kernel one file
K1_OBJ := causal_conv1d_fwd.o
K2_OBJ := causal_conv1d_update.o
K3_OBJ := chunk_gated_delta_rule.o
K4_OBJ := fused_recurrent.o
K5_OBJ := fused_rms_norm_gate.o

KERN_OBJS  := $(K1_OBJ) $(K2_OBJ) $(K3_OBJ) $(K4_OBJ) $(K5_OBJ)
BENCH_BINS := bench_k1 bench_k2 bench_k3 bench_k4 bench_k5

.PHONY: all bench clean

all: deltanet_test
bench: $(BENCH_BINS)

# ---- Kernel objects (1 file = 1 kernel) ----
$(K1_OBJ): src/causal_conv1d_fwd.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

$(K2_OBJ): src/causal_conv1d_update.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

$(K3_OBJ): src/chunk_gated_delta_rule.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

$(K4_OBJ): src/fused_recurrent.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

$(K5_OBJ): src/fused_rms_norm_gate.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

# ---- Main test binary (links all kernels) ----
main.o: main.cu include/deltanet.h include/naive_reference.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

deltanet_test: main.o $(KERN_OBJS)
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS)

# ---- Bench binaries (each links only its kernel) ----
bench_k1.o: bench/bench_k1.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@
bench_k2.o: bench/bench_k2.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@
bench_k3.o: bench/bench_k3.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@
bench_k4.o: bench/bench_k4.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@
bench_k5.o: bench/bench_k5.cu include/deltanet.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

bench_k1: bench_k1.o $(K1_OBJ)
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS)

bench_k2: bench_k2.o $(K2_OBJ)
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS)

bench_k3: bench_k3.o $(K3_OBJ)
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS)

bench_k4: bench_k4.o $(K4_OBJ)
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS)

bench_k5: bench_k5.o $(K5_OBJ)
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS)

clean:
	rm -f *.o deltanet_test $(BENCH_BINS)
