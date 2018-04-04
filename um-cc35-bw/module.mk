MODULE := um-cc35-bw

TARGETS += $(MODULE)/main
CLEAN_TARGETS += $(MODULE)/main.o

$(MODULE)/main: $(MODULE)/main.o common/common.o
	$(NVCC) $^ -std=c++11 -Xcompiler -Wall,-Wextra,-O3 $(NVCC_LD_FLAGS) -o $@