TARGETS += mgpu-sync/main

mgpu-sync/main: mgpu-sync/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -arch=sm_60 -rdc=true -o $@ -lnvToolsExt