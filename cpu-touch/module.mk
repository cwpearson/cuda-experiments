TARGETS += cpu-touch/main

cpu-touch/main: cpu-touch/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -o $@ -ldl -lcuda -lnvToolsExt