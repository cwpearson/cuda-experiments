MODULE := cpu-cpu

TARGETS += $(MODULE)/main
CLEAN_TARGETS += $(MODULE)/main.o

$(MODULE)/main: $(MODULE)/main.o $(MODULE)/read.o $(COMMON_OBJECTS)
	$(NVCC) $^ -std=c++11  -o $@ -lnuma -lcudart -lgomp