NVCC ?= nvcc
ARCH ?= -arch=sm_80
CFLAGS ?= -O2 -std=c++17 --expt-relaxed-constexpr -diag-suppress 177,179,39

BINS := marlin_standalone

.PHONY: all clean

all: $(BINS)

marlin_standalone: marlin_standalone.cu
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@

clean:
	rm -f $(BINS)
