MODULE := mgpu-sync

TARGETS += $(MODULE)/main
CLEAN_TARGETS += $(MODULE)/main.o $(MODULE)/main.ptx

$(MODULE)/main: $(MODULE)/main.o common/common.o
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -o $@ -lcuda -lnvToolsExt -lnuma

$(MODULE)/main.ptx: $(MODULE)/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -rdc=true -ptx -src-in-ptx -o $@ -lnvToolsExt
