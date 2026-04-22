NVCC       := nvcc
ARCH       := -arch=sm_90a
CFLAGS     := -O2 -std=c++17 $(ARCH) -Iinclude --expt-relaxed-constexpr

BINS       := bench_topk_gating bench_moe_align bench_silu_and_mul bench_moe_sum

.PHONY: all clean

all: $(BINS)

bench_topk_gating: src/topk_gating.cu
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_moe_align: src/moe_align.cu
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_silu_and_mul: src/silu_and_mul.cu
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_moe_sum: src/moe_sum.cu
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

clean:
	rm -f *.o $(BINS)
