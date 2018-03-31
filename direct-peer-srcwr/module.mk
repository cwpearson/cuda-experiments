MODULE := direct-peer-srcwr

TARGETS += $(MODULE)/main

$(MODULE)/main: $(MODULE)/main.cu common/common.hpp
	$(NVCC) $(NVCCFLAGS) $< -std=c++11 -Xcompiler -Wall,-Wextra,-O3 -o $@ -lcuda -lnvToolsExt -lnuma
