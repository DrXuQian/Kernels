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

all: dequant_smoke marlin_ppu_test marlin_classic_num marlin_classic_splitk marlin_classic_group mma_int8_smoke

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

# classic Marlin GROUPED-scale test (groupsize=128 -> 4 groups; the group_blocks != -1 path).
# CAN WE EAT A vLLM/MARLIN CHECKPOINT AS-IS?  Build it two ways and let the answer be a PASS/FAIL:
#   plain (default)                                 -> our column-indexed read + unpermuted [G][N] scales
#   -DTEST_SCALE_PERM -DMARLIN_NVIDIA_SCALE_MAP     -> upstream's _scale_perm + upstream's one-int4 lane read
# The second is the one that matters: _scale_perm is derived from the B-FRAGMENT column layout, and B's
# layout is identical on PPU, so it SHOULD pass. If it does, vLLM's grouped scales need no un-permuting.
marlin_classic_group: test_marlin_classic_group.cu marlin_classic_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# classic Marlin split-K test (exercises global_reduce, cross-CTA K reduce).
marlin_classic_splitk: test_marlin_classic_splitk.cu marlin_classic_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# perf bench for the classic Marlin PPU port (random data, TFLOP/s).
bench_marlin: bench_marlin.cu marlin_classic_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# INT8 tensor-core gate for the llama.cpp MMQ port: does the PPU have m16n16k32 s32.s8.s8.s32?
# NVIDIA reference path (validates the probe harness):  make NVCCFLAGS="-O3 -std=c++17 -arch=sm_90" mma_int8_smoke
# PPU path (box, ppu001, NO -arch):                     make NVCCFLAGS="-O3 -std=c++17 -DUSE_PPU" mma_int8_smoke
mma_int8_smoke: mma_int8_smoke.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Standalone llama.cpp MMQ (Q4_0 + Q4_K), self-checking.
# NVIDIA reference:  make NVCCFLAGS="-O3 -std=c++17 -arch=sm_90 -DMMQ_NV" mmq_ppu
# PPU (ppu001, NO -arch): make NVCCFLAGS="-O3 -std=c++17 -DMMQ_PPU" mmq_ppu
mmq_ppu: mmq_ppu.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Tensor-core PEAK: the DENOMINATOR for every compute-utilization number. Reports int8 and fp16
# back-to-back mma throughput and their ratio. Sweep NACC (argv[1]) to prove it isn't latency-bound.
#   PPU:  make NVCCFLAGS="-O3 -std=c++17" mma_peak && ./mma_peak 4
mma_peak: mma_peak.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ground truth for ppu.ldmatrix.x4 register distribution (which 8x8 matrix lands in which reg).
dump_ldmatrix: dump_ldmatrix.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Achievable read bandwidth: the denominator for every %HBM figure. No ppu.* asm.
bw_probe: bw_probe.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# W4A16 GEMV on an N-MAJOR INT4 layout (a SECOND weight tensor). The v8 structure from the bf16 sweep: lean register
# LDG, no shared / cp.async / split-K. Self-checking. Sweep with -DNM_COLS / -DNM_THREADS / -DNM_UNROLL.
gemv_nmajor: gemv_w4a16_nmajor.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# W4A16 GEMV for decode: reads Marlin's B format in place, no ppu.* asm (so it also builds on stock CUDA).
# Self-checking (packs B on the host, compares against a CPU dot product) and reports GB/s + %HBM.
gemv_w4a16: gemv_w4a16_ppu.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<


clean:
	rm -f dequant_smoke mma_int8_smoke mmq_ppu mma_peak dump_ldmatrix marlin_ppu_test marlin_classic_test marlin_classic_num marlin_classic_splitk marlin_classic_group bench_marlin gemv_w4a16 gemv_nmajor bw_probe *.o .flags
