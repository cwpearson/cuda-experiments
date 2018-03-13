NVCC = nvcc

MODULES := cpu-touch ctx coherence stream-thread

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
