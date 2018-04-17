MODULE := system-atomics

TARGETS += $(MODULE)/main
CLEAN_TARGETS += $(MODULE)/main.o

$(MODULE)/main: $(MODULE)/main.o common/common.o
	$(NVCC) $(NVCCFLAGS) -arch=sm_70 -std=c++11 -Xcompiler -Wall,-Wextra,-O3 $^ -o $@ -lcuda -lnuma -lnvToolsExt

$(MODULE)/main.o: $(MODULE)/main.cu
	$(NVCC) $(NVCCFLAGS) -arch=sm_70 -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -dc $< -o $@ -lnvToolsExt
