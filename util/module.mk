MODULE := util

TARGETS += $(MODULE)/lscpu $(MODULE)/lsgpu
CLEAN_TARGETS += $(MODULE)/lscpu.o $(MODULE)/lsgpu.o

$(MODULE)/lscpu: $(MODULE)/lscpu.o common/common.o
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -o $@ -lnuma

$(MODULE)/lsgpu: $(MODULE)/lsgpu.o common/common.o
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -o $@ -lnuma
