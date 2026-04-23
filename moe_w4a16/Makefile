SUBDIRS := marlin auxiliary

ALL_BINS := bench_marlin_moe bench_topk_gating bench_moe_align bench_silu_and_mul bench_moe_sum

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)
	@# Copy all binaries to this directory
	@cp -f marlin/bench_marlin_moe . 2>/dev/null || true
	@cp -f auxiliary/bench_topk_gating auxiliary/bench_moe_align \
	       auxiliary/bench_silu_and_mul auxiliary/bench_moe_sum . 2>/dev/null || true
	@echo "All binaries copied to moe_w4a16/"

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
	rm -f $(ALL_BINS)
