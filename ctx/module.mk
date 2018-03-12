TARGETS += ctx/main

ctx/main: ctx/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -G -g -o $@ -ldl -lcuda -lnvToolsExt