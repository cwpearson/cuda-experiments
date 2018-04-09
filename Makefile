include config.mk

USE_THIRDPARTY=0

NVCC = nvcc

#NVCC_VER_MAJOR := $(shell nvcc -V | grep -oP "release \K([0-9]{1,}\.)+[0-9]{1,}")
NVCC_VER_MAJOR := $(shell nvcc -V | grep -oP "release \K([0-9])")
NVCC_VER_MINOR := $(shell nvcc -V | grep -oP "release [0-9]\.\K([0-9])")
DRIVER_VERSION := $(shell nvidia-smi | grep -oP "Driver Version: \K([0-9]{1,}\.)+[0-9]{1,}")

$(info $(NVCC_VER_MAJOR).$(NVCC_VER_MINOR) "/" $(DRIVER_VERSION) )

GENCODE := -gencode arch=compute_$(CUDA_CC),code=compute_$(CUDA_CC)

CC_GT_70 := $(shell echo $(CUDA_CC)\>=70 | bc )
CC_GT_60 := $(shell echo $(CUDA_CC)\>=60 | bc )

$(info Compute Capability >=70: $(CC_GT_70))
$(info Compute Capability >=60: $(CC_GT_60))

MODULES += common \
	access-counters \
	atomics \
	atomics.1 \
	ctx \
	direct-peer-srcwr \
	direct-peer-dstrd \
	memcpy-nopeer \
	memcpy-peer \
	pageable \
	pinned \
	stream-thread \
	stream-warp \
	um-cc35-bw \
	wc

ifeq ($(CC_GT_70),1)
MODULES += \
	mgpu-sync \
	system-atomics
endif

ifeq ($(CC_GT_60),1)
MODULES += \
	coherence-bw \
	coherence-latency \
	cpu-touch \
	prefetch-bw 
endif





# Look in each module for include files
#NVCCFLAGS += $(patsubst %,-I%,$(MODULES)) -I. -lineinfo
NVCCFLAGS += -I. -lineinfo $(GENCODE)

ifeq ($(USE_THIRDPARTY),1)
NVCCFLAGS += -Ithirdparty/include 
endif

NVCC_LD_FLAGS += -lnuma -lnvToolsExt -Lthirdparty/lib -Xcompiler '"-Wl,-rpath=thirdparty/lib"'

#each module will add to this
TARGETS :=
CLEAN_TARGETS :=
COMMON_OBJECTS :=
COMMON_HEADERS :=

#include the description for
#each module
include $(patsubst %,%/module.mk,$(MODULES))

%.o: %.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) $< -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -o $@ -c

.PHONY : all
.DEFAULT_GOAL := all
all: $(TARGETS)

clean:
	rm -f $(TARGETS) $(CLEAN_TARGETS)

bench: $(TARGETS)
	for m in $(MODULES); do \
		echo $$m; \
		$$m/main >> $$m.csv; \
	done;