# Build the standalone Marlin FP16xINT4 GEMM experiments for the PPU.
#
# PPU target = ppu001: build with NO -arch (nvcc default). A forced sm_XX routes onto ppu0015 and rejects
# the ppu001-only asm atoms. See the ppu-ptx skill / memory ppu-build-target-and-m8n16.

NVCC     ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda

# Rebuild whenever NVCCFLAGS changes. Without this, `make NVCCFLAGS="... -DFOO" tgt` silently keeps a binary built
# with different flags (the deps are only the sources), which quietly invalidates any -D A/B comparison.
.PHONY: all clean force
.flags: force
	@echo '$(NVCCFLAGS)' | cmp -s - $@ || echo '$(NVCCFLAGS)' > $@

all: dequant_smoke marlin_ppu_test marlin_classic_num marlin_classic_splitk

# INT4 dequant (lop3) in isolation vs CPU.
dequant_smoke: dequant_smoke.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# plain-CUDA PPU Marlin self-consistent correctness test.
marlin_ppu_test: test_marlin_ppu.cu marlin_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ported classic Marlin (marlin_classic_ppu.cuh) compile/run gate.
marlin_classic_test: test_marlin_classic_ppu.cu marlin_classic_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ported classic Marlin NUMERICAL test (self-consistent B packing) vs CPU.
marlin_classic_num: test_marlin_classic_num.cu marlin_classic_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# classic Marlin split-K test (exercises global_reduce, cross-CTA K reduce).
marlin_classic_splitk: test_marlin_classic_splitk.cu marlin_classic_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# perf bench for the classic Marlin PPU port (random data, TFLOP/s).
bench_marlin: bench_marlin.cu marlin_classic_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ground truth for ppu.ldmatrix.x4 register distribution (which 8x8 matrix lands in which reg).
dump_ldmatrix: dump_ldmatrix.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<


clean:
	rm -f dequant_smoke dump_ldmatrix marlin_ppu_test marlin_classic_test marlin_classic_num marlin_classic_splitk bench_marlin *.o .flags
