TARGETS += access-counters/main

access-counters/main: access-counters/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -o $@ -lcuda -lnvToolsExt