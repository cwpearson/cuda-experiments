MODULE := mgpu-sync

TARGETS += $(MODULE)/main $(MODULE)/main.ptx

$(MODULE)/main: mgpu-sync/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -gencode $(GENCODE) -rdc=true -o $@ -lnvToolsExt

$(MODULE)/main.ptx: mgpu-sync/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -gencode $(GENCODE) -rdc=true -ptx -src-in-ptx -o $@ -lnvToolsExt
