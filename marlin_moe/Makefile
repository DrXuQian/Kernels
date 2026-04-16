NVCC    := nvcc
ARCH    := -arch=sm_120
CFLAGS  := -O2 -std=c++17 $(ARCH) -Iinclude --expt-relaxed-constexpr

BENCH_BINS := bench_marlin_moe bench_topk_gating bench_moe_align bench_silu_and_mul bench_moe_sum

.PHONY: all clean

all: $(BENCH_BINS)

# ---- Marlin MoE GEMM (separate compilation for templates) ----
kernel_fp16_u4.o: src/kernel_fp16_u4.cu include/kernel.h include/marlin_template.h \
                  include/marlin.cuh include/marlin_dtypes.cuh include/dequant.h \
                  include/marlin_mma.h include/scalar_type.hpp include/compat.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

dispatch.o: src/dispatch.cu include/kernel.h include/kernel_selector.h \
            include/marlin.cuh include/scalar_type.hpp include/compat.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

bench_marlin.o: bench_marlin_moe.cu include/scalar_type.hpp include/compat.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

bench_marlin_moe: bench_marlin.o dispatch.o kernel_fp16_u4.o
	$(NVCC) $(ARCH) $^ -o $@

# ---- Auxiliary MoE kernels (single-file + -DBENCH) ----
bench_topk_gating: src/topk_gating.cu include/moe_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_moe_align: src/moe_align.cu include/moe_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_silu_and_mul: src/silu_and_mul.cu include/moe_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

bench_moe_sum: src/moe_sum.cu include/moe_compat.h
	$(NVCC) $(CFLAGS) -DBENCH $< -o $@

clean:
	rm -f *.o $(BENCH_BINS)
