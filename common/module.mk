MODULE := common

TARGETS += $(MODULE)/common.o

$(MODULE)/common.o: $(MODULE)/common.cpp common/common.hpp
	$(NVCC) $(NVCCFLAGS) $< -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -c -o $@