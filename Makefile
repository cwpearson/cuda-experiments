NVCC = nvcc

NVCC_VERSION := $(shell nvcc -V | grep -oP "release \K([0-9]{1,}\.)+[0-9]{1,}")
DRIVER_VERSION := $(shell nvidia-smi | grep -oP "Driver Version: \K([0-9]{1,}\.)+[0-9]{1,}")

$(info $(NVCC_VERSION) "/" $(DRIVER_VERSION) )

MODULES := access-counters \
	atomics \
	atomics.1 \
	cpu-touch \
	coherence \
	ctx \
	mgpu-sync \
	prefetch-bw \
	stream-thread \
	stream-warp \
	system-atomics

# Look in each module for include files
#NVCCFLAGS += $(patsubst %,-I%,$(MODULES)) -I. -lineinfo
NVCCFLAGS += -I. -lineinfo

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
