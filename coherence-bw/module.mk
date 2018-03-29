MODULE := coherence-bw

TARGETS += $(MODULE)/main

$(MODULE)/main: $(MODULE)/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -o $@ -lcuda -lnvToolsExt