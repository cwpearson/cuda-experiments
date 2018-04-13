MODULE := mgpu-sync

TARGETS += $(MODULE)/main
CLEAN_TARGETS += $(MODULE)/main.o $(MODULE)/main.ptx

$(MODULE)/main: $(MODULE)/main.o common/common.o
	$(NVCC) $(NVCCFLAGS) -arch=sm_70 -std=c++11 -Xcompiler -Wall,-Wextra,-O3 $^ -o $@ -lcuda -lnvToolsExt -lnuma

$(MODULE)/main.o: $(MODULE)/main.cu
	$(NVCC) $(NVCCFLAGS) -arch=sm_70 -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -dc $^ -o $@ -lnvToolsExt
