SUBDIRS := general linear_attention moe_w4a16/vllm/marlin moe_w4a16/vllm/auxiliary

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
