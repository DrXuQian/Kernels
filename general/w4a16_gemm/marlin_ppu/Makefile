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

# GGUF int4 on the PPU Marlin (marlin_gguf_ppu.cuh): gs=32 per-32 in-tile scale (what Q4_0 needs).
# Box only (ppu.mma). PTX check: nvcc -std=c++17 --expt-relaxed-constexpr -arch=sm_90 -ptx -c test_marlin_gguf.cu
# The sha is compiled IN and printed at startup: a stale checkout / wrong branch shows up in the output
# instead of masquerading as "the fix did not work" (that misread cost a full debug cycle once already).
marlin_gguf: test_marlin_gguf.cu marlin_gguf_ppu.cuh affine_split_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -lcublas -DGGUF_BUILD_SHA='"$(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)$(shell git diff --quiet 2>/dev/null || echo -dirty)"' -o $@ $<

# perf bench for the GGUF W4A16 port. Pairs gs=128 vs gs=32 (and sym vs affine) on each shape, because the
# question is the gs=32 DELTA against marlin_classic's 47% MFU, not an absolute number.
bench_marlin_gguf: bench_marlin_gguf.cu marlin_gguf_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Brackets the affine min cost: stages and loads the mins exactly as the real path does, applies hmul2
# instead of hfma2. WRONG RESULTS BY DESIGN, timing only. If the aff lines fall back to the sym lines the
# cost is arithmetic; if they stay put it is moving the mins.
bench_marlin_gguf_nomin: bench_marlin_gguf.cu marlin_gguf_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -DMARLIN_MINS_NOAPPLY -o $@ $<

# hfma2 KEPT, min value replaced by a constant -- drops the per-k shared load only. Needed because
# _nomin's unread frag_m gets dead-code-eliminated along with its load, so that build drops both and
# cannot tell the arithmetic from the load. WRONG RESULTS, timing only.
# Is the affine min cheaper as its OWN kernel? It never touches B, so it factors into
# Asum[M x G] @ m'[G x N] with G = K/32 -- ~3% of the main GEMM's FLOPs. Needs cublas for the skinny gemm.
# Which path wins at which M: fused affine vs split affine vs dequant-to-fp16 + dense hgemm.
# All three verified against one CPU reference before any timing.
# GGUF Q4_K MoE (grouped) GEMM -- qwen35moe's expert FFN goes through mul_mat_id, which the dense kernels
# do not cover at all and which IS the prefill compute for an A3B model.
# Gate for the Q4_K MoE direction: can Marlin-format INT4 ride the AIU bulk load + swzl ldmatrix the way
# the fp16 MoE GEMM moves bf16? Box only (ppu.* asm), NO -arch.
# AIU Q4_K MoE vs the Marlin-derived MoE (which passes ragged routing). Box only.
moe_aiu: test_moe_aiu.cu marlin_moe_aiu_ppu.cuh marlin_moe_gguf_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Same, with the column-range guard compiled in: turns an illegal access into a printed reason.
moe_aiu_guard: test_moe_aiu.cu marlin_moe_aiu_ppu.cuh marlin_moe_gguf_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -DMOE_AIU_GUARD -o $@ $<

aiu_int4_smoke: aiu_int4_smoke.cu .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

marlin_moe_gguf: test_marlin_moe_gguf.cu marlin_moe_gguf_ppu.cuh marlin_gguf_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bench_prefill_paths: bench_prefill_paths.cu marlin_gguf_ppu.cuh affine_split_ppu.cuh dequant_w4_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcublas

bench_affine_split: bench_affine_split.cu marlin_gguf_ppu.cuh affine_split_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcublas

bench_marlin_gguf_cmin: bench_marlin_gguf.cu marlin_gguf_ppu.cuh .flags
	$(NVCC) $(NVCCFLAGS) -DMARLIN_MINS_CONST -o $@ $<

# classic Marlin GROUPED-scale test (groupsize=128; the group_blocks != -1 path).
# ANSWERED, on ppu001, all four configs, 1 group AND 4/8 groups: a vLLM / GPTQ-Marlin checkpoint's B AND
# grouped scales are BOTH directly edible -- no conversion at load. _scale_perm is derived from the
# B-FRAGMENT column layout, and B's fragment layout is identical on PPU, so upstream's layout is the
# kernel's DEFAULT contract now (one int4 per lane -- correct AND ~6.6 us faster than the by-column read).
#   default              -> upstream's _scale_perm layout (marlin_permute_scales in the header)
#   -DMARLIN_SCALE_PLAIN -> raw [num_groups][prob_n] scales, e.g. straight off a GPTQ checkpoint
#   -DSCALE_DECODE       -> also print, per column, WHICH scale index the kernel actually read
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

# Same GEMV with Marlin's _scale_perm applied to the scale/min tensors, so a lane's 8 scales are ONE int4
# load instead of 8 scattered halves. Aimed at gs=32, where the scale:B load ratio is 4x worse than gs=128.
gemv_w4a16_perm: gemv_w4a16_ppu.cu .flags
	$(NVCC) $(NVCCFLAGS) -DGEMV_SCALE_PERM -o $@ $<

# Per-group flush done in half (one v.cnvt instead of three/four). acu says gs=32 is ALU-bound on the PPU:
# v.fma.f16 and v.lop3.i are IDENTICAL to gs=128 while v.cnvt is +295% and the fp32 flush ops are +300%.
gemv_w4a16_h16: gemv_w4a16_ppu.cu .flags
	$(NVCC) $(NVCCFLAGS) -DGEMV_FLUSH_H16 -o $@ $<

# ONE warp per block. Two quantities move in opposite directions here and both have to be held:
#   flushes   ~ groups x WARPS PER GROUP        (T=32 halves it: one warp owns the group's 2 ktiles)
#   occupancy ~ blocks x WARPS PER BLOCK        (T=32 halves it too: 1024 blocks x 1 warp = 14/CU = 22%)
# So T=32 alone trades one for the other. Pair it with SPLIT_K=32 and both are held: 2048 blocks x 1 warp
# is 28 warps/CU, identical to today's 1024 x 2, while each group is flushed once instead of twice. The
# only new cost is 2x reduce -- and that is exactly what (T=64, sk=32) paid WITHOUT buying fewer flushes,
# which is why it lost on the box (33.26 -> 36.76).
# MEASURED, and the two builds are NOT interchangeable -- pick by groupsize (box, cold, N4096 K14336):
#            T=64/sk16   T=32
#   gs=32      35.73     26.54   -26%   <- GGUF Q4_K lives here: use gemv_w4a16_t32
#   gs=-1      19.17     20.02    +4%
#   gs=128     19.89     25.49   +28%   <- gs=128 cannot reach sk=32 (28 ktiles/slice vs an 8-ktile group),
#                                          so T=32 only costs it half its warps. Use gemv_w4a16.
# Small group -> T=32 (it can take sk=32, which is what pays for the halved warps-per-block).
# gs>=128     -> T=64. There is no single default; this is the same per-format templating the MMQ port
# needed for its X and MIN_BLOCKS.
gemv_w4a16_t32: gemv_w4a16_ppu.cu .flags
	$(NVCC) $(NVCCFLAGS) -DGEMV_THREADS=32 -DGEMV_UNROLL=2 -o $@ $<


clean:
	rm -f moe_aiu_guard moe_aiu aiu_int4_smoke marlin_moe_gguf bench_prefill_paths bench_affine_split bench_marlin_gguf_cmin bench_marlin_gguf_nomin gemv_w4a16_t32 gemv_w4a16_h16 gemv_w4a16_perm bench_marlin_gguf dequant_smoke mma_int8_smoke mmq_ppu mma_peak marlin_gguf dump_ldmatrix marlin_ppu_test marlin_classic_test marlin_classic_num marlin_classic_splitk marlin_classic_group bench_marlin gemv_w4a16 gemv_nmajor bw_probe *.o .flags
