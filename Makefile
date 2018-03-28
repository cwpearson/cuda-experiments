NVCC = nvcc

MODULES := access-counters \
	atomics \
	atomics.1 \
	cpu-touch \
	coherence \
	ctx \
	mgpu-sync \
	stream-thread \
	stream-warp \
	system-atomics

# Look in each module for include files
NVCCFLAGS += $(patsubst %,-I%,$(MODULES)) -I. -lineinfo

#each module will add to this
TARGETS :=

#include the description for
#each module
include $(patsubst %,%/module.mk,$(MODULES))

.PHONY : all
.DEFAULT_GOAL := all
all: $(TARGETS)

clean:
	rm -f $(TARGETS)
