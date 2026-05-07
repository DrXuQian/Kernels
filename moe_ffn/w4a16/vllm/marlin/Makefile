NVCC       := nvcc
ARCH       := -arch=sm_90a
CFLAGS     := -O2 -std=c++17 $(ARCH) -Iinclude -I../../../../include --expt-relaxed-constexpr

BINS       := bench_marlin_moe

.PHONY: all clean

all: $(BINS)

kernel_fp16_u4.o: src/kernel_fp16_u4.cu
	$(NVCC) $(CFLAGS) -dc $< -o $@

dispatch.o: src/dispatch.cu
	$(NVCC) $(CFLAGS) -dc $< -o $@

bench_marlin.o: bench_marlin_moe.cu
	$(NVCC) $(CFLAGS) -dc $< -o $@

bench_marlin_moe: bench_marlin.o dispatch.o kernel_fp16_u4.o
	$(NVCC) $(ARCH) $^ -o $@

clean:
	rm -f *.o $(BINS)
