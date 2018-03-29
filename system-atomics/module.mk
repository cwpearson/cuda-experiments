TARGETS += system-atomics/main

system-atomics/main: system-atomics/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -rdc=true -o $@ -lnvToolsExt