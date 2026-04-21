NVCC       := nvcc
ARCH       := -arch=sm_120
CFLAGS     := -O2 -std=c++17 $(ARCH) -Iinclude --expt-relaxed-constexpr

# Extracted from llama.cpp: gated_delta_net (prefill+decode) and ssm_conv (causal conv1d)
BENCH_BINS := bench_gated_delta_net bench_ssm_conv

.PHONY: all clean

all: $(BENCH_BINS)

bench_gated_delta_net: src/gated_delta_net.cu include/ggml_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_ssm_conv: src/ssm_conv.cu include/ggml_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

clean:
	rm -f *.o $(BENCH_BINS)
