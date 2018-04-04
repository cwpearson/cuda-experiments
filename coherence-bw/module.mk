MODULE := coherence-bw

TARGETS += $(MODULE)/main
CLEAN_TARGETS += $(MODULE)/main.o

$(MODULE)/main.o: $(MODULE)/main.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) $< -std=c++11 -Xcompiler -Wall,-Wextra,-O3,-fopenmp -c -o $@ -lcuda -lnvToolsExt -lnuma

$(MODULE)/main: $(MODULE)/main.o $(COMMON_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3,-fopenmp -o $@ -lcuda -lnvToolsExt -lnuma