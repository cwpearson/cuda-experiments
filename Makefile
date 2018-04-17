USE_THIRDPARTY=0

NVCC = nvcc

#NVCC_VER_MAJOR := $(shell nvcc -V | grep -oP "release \K([0-9]{1,}\.)+[0-9]{1,}")
NVCC_VER_MAJOR := $(shell nvcc -V | grep -oP "release \K([0-9])")
# DRIVER_VERSION ?= $(shell nvidia-smi | grep -oP "Driver Version: \K([0-9]{1,}\.)+[0-9]{1,}")

$(info $(NVCC_VER_MAJOR).$(NVCC_VER_MINOR))

NVCC_GTE_9 := $(shell echo $(NVCC_VER_MAJOR)\>=9 | bc )
NVCC_GTE_8 := $(shell echo $(NVCC_VER_MAJOR)\>=8 | bc )
NVCC_GTE_7 := $(shell echo $(NVCC_VER_MAJOR)\>=7 | bc )
NVCC_GTE_6 := $(shell echo $(NVCC_VER_MAJOR)\>=6 | bc )

$(info nvcc >= 9: $(NVCC_GTE_9))
$(info nvcc >= 8: $(NVCC_GTE_8))
$(info nvcc >= 7: $(NVCC_GTE_7))
$(info nvcc >= 6: $(NVCC_GTE_6))

MODULES += common \
	access-counters \
	atomics \
	atomics.1 \
	cpu-cpu \
	cpu-cpu-df \
	ctx \
	direct-peer-srcwr \
	direct-peer-dstrd \
	memcpy-nopeer \
	memcpy-peer \
	pageable \
	pageable-host \
	pinned \
	std-memcpy \
	std-memcpy-df \
	um-cc35-bw \
	wc

ifeq ($(NVCC_GTE_9),1)
MODULES += \
	mgpu-sync \
	system-atomics
endif

ifeq ($(NVCC_GTE_8),1)
MODULES += \
	coherence-bw \
	coherence-latency \
	cpu-touch \
	prefetch-bw
endif

ifeq ($(NVCC_GTE_7),1)
MODULES += \
	stream-thread \
	stream-warp
endif


# Look in each module for include files
#NVCCFLAGS += $(patsubst %,-I%,$(MODULES)) -I. -lineinfo
NVCCFLAGS += -I. -lineinfo -Wno-deprecated-gpu-targets
CXXFLAGS += -I.

ifeq ($(USE_THIRDPARTY),1)
NVCCFLAGS += -Ithirdparty/include
CXXFLAGS += -Ithirdparty/include 
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
	$(NVCC) $(NVCCFLAGS) $< -std=c++11 -Xcompiler -Wall,-Wextra,-O3,-Wshadow -o $@ -c

%.o: %.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $< -std=c++11 -Wall -Wextra -O3 -Wshadow -fopenmp -c -o $@

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