NVCC    := nvcc
ARCH    := -arch=sm_120
CFLAGS  := -O2 -std=c++17 $(ARCH) -Iinclude --expt-relaxed-constexpr

.PHONY: all clean

all: bench_marlin_moe

# Kernel instantiations (slow to compile — template heavy)
kernel_fp16_u4.o: src/kernel_fp16_u4.cu include/kernel.h include/marlin_template.h \
                  include/marlin.cuh include/marlin_dtypes.cuh include/dequant.h \
                  include/marlin_mma.h include/scalar_type.hpp include/compat.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

# Dispatch logic (config selection + kernel launch)
dispatch.o: src/dispatch.cu include/kernel.h include/kernel_selector.h \
            include/marlin.cuh include/scalar_type.hpp include/compat.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

# Bench main
bench.o: bench_marlin_moe.cu include/scalar_type.hpp include/compat.h
	$(NVCC) $(CFLAGS) -dc $< -o $@

bench_marlin_moe: bench.o dispatch.o kernel_fp16_u4.o
	$(NVCC) $(ARCH) $^ -o $@

clean:
	rm -f *.o bench_marlin_moe
