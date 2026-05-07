NVCC       := nvcc
CUTLASS    := ../../../third_party/cutlass
ARCH       := -arch=sm_90a
CFLAGS     := -O2 -std=c++17 $(ARCH) --expt-relaxed-constexpr -diag-suppress 3288
INC        := -I. -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include \
              -I../../../include -DFLAT_SM90A_ENABLED

BINS       := bench_gdn_prefill

.PHONY: all clean

all: $(BINS)

# Kernel template instantiation (bf16 GVA only, slow compile)
kernel_inst.o: kernel_inst.cu
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

# Dispatch (runtime bool → template param mapping)
dispatch.o: dispatch.cu
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

# Bench main (fast compile, no heavy templates)
bench.o: bench_gdn_prefill.cu
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

bench_gdn_prefill: bench.o dispatch.o kernel_inst.o
	$(NVCC) $(ARCH) $^ -o $@

clean:
	rm -f *.o $(BINS)
