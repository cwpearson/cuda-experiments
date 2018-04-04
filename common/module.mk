MODULE := common

TARGETS += $(MODULE)/common.o
COMMON_OBJECTS += $(MODULE)/common.o
COMMON_HEADERS += $(MODULE)/common.hpp

$(MODULE)/common.o: $(MODULE)/common.cpp common/common.hpp
	$(NVCC) $(NVCCFLAGS) $< -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -c -o $@