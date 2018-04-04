USE_THIRDPARTY=0

NVCC = nvcc

#NVCC_VER_MAJOR := $(shell nvcc -V | grep -oP "release \K([0-9]{1,}\.)+[0-9]{1,}")
NVCC_VER_MAJOR := $(shell nvcc -V | grep -oP "release \K([0-9])")
DRIVER_VERSION := $(shell nvidia-smi | grep -oP "Driver Version: \K([0-9]{1,}\.)+[0-9]{1,}")

$(info $(NVCC_VER_MAJOR) "/" $(DRIVER_VERSION) )

ifeq ($(NVCC_VER_MAJOR),9)
MODULES += coherence-bw \
	coherence-latency \
	cpu-touch \
           mgpu-sync \
           system-atomics
	prefetch-bw \
GENCODE := -gencode arch=compute_50,code=compute_50 \
           -gencode arch=compute_52,code=compute_52 \
           -gencode arch=compute_60,code=compute_60 \
           -gencode arch=compute_61,code=compute_61 \
           -gencode arch=compute_62,code=compute_62 \
           -gencode arch=compute_70,code=compute_70
else ifeq ($(NVCC_VER_MAJOR),8)
MODULES += coherence-bw \
	coherence-latency \
	cpu-touch \
	prefetch-bw \

GENCODE := -gencode arch=compute_60,code=compute_60 \
           -gencode arch=compute_61,code=compute_61 \
           -gencode arch=compute_62,code=compute_62
else ifeq ($(NVCC_VER_MAJOR),7)
GENCODE := -gencode arch=compute_35,code=compute_35 \
           -gencode arch=compute_50,code=compute_50 \
           -gencode arch=compute_52,code=compute_52
else
$(error Unrecognized nvcc version)
endif

MODULES = common \
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
	wc

# Look in each module for include files
#NVCCFLAGS += $(patsubst %,-I%,$(MODULES)) -I. -lineinfo
NVCCFLAGS += -I. -lineinfo $(GENCODE)

ifeq ($(USE_THIRDPARTY),1)
NVCCFLAGS += -Ithirdparty/include -Lthirdparty/lib -Xcompiler '"-Wl,-rpath=thirdparty/lib"'
endif

NVCCFLAGS += -lnuma

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
