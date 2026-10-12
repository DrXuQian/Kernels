NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: all force
all: bench_aiu_read

# This is small: always rebuild so changed SDK/flags cannot reuse a stale binary.
# PPU 1.0 uses the PPU SDK default target; do not inherit -arch=sm_XX.
bench_aiu_read: bench_aiu_read.cu force
	$(NVCC) $(NVCCFLAGS) $< -o $@
