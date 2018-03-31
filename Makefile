NVCC = nvcc

#NVCC_VER_MAJOR := $(shell nvcc -V | grep -oP "release \K([0-9]{1,}\.)+[0-9]{1,}")
NVCC_VER_MAJOR := $(shell nvcc -V | grep -oP "release \K([0-9])")
DRIVER_VERSION := $(shell nvidia-smi | grep -oP "Driver Version: \K([0-9]{1,}\.)+[0-9]{1,}")

$(info $(NVCC_VER_MAJOR) "/" $(DRIVER_VERSION) )

ifeq ($(NVCC_VER_MAJOR),9)
MODULES += mgpu-sync \
           system-atomics
GENCODE := -gencode arch=compute_50,code=compute_50 \
           -gencode arch=compute_52,code=compute_52 \
           -gencode arch=compute_60,code=compute_60 \
           -gencode arch=compute_61,code=compute_61 \
		   -gencode arch=compute_62,code=compute_62 \
		   -gencode arch=compute_70,code=compute_70
else ifeq ($(NVCC_VER_MAJOR),8)
GENCODE := -gencode arch=compute_60,code=compute_60 \
           -gencode arch=compute_61,code=compute_61 \
		   -gencode arch=compute_62,code=compute_62
else
$(error Unrecognized nvcc version)
endif

MODULES = access-counters \
	atomics \
	atomics.1 \
	cpu-touch \
	coherence-bw \
	coherence-latency \
	ctx \
	pageable \
	pinned \
	prefetch-bw \
	stream-thread \
	stream-warp 

# Look in each module for include files
#NVCCFLAGS += $(patsubst %,-I%,$(MODULES)) -I. -lineinfo
NVCCFLAGS += -I. -lineinfo $(GENCODE) -lnuma

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
