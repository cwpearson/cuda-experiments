MODULE := std-memcpy

TARGETS += $(MODULE)/main-src $(MODULE)/main-dst
CLEAN_TARGETS += $(MODULE)/main_src.o $(MODULE)/main_dst.o $(MODULE)/op.o

$(MODULE)/main-src: $(MODULE)/main_src.o $(MODULE)/op.o $(COMMON_OBJECTS)
	$(NVCC) $^ -std=c++11  -o $@ -lnuma -lcudart -lgomp

$(MODULE)/main-dst: $(MODULE)/main_dst.o $(MODULE)/op.o $(COMMON_OBJECTS)
	$(NVCC) $^ -std=c++11  -o $@ -lnuma -lcudart -lgomp

$(MODULE)/main_src.o: $(MODULE)/main.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $< -DOP_SRC -std=c++11 -Wall -Wextra -O3 -fopenmp -c -o $@

$(MODULE)/main_dst.o: $(MODULE)/main.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $< -DOP_DST -std=c++11 -Wall -Wextra -O3 -fopenmp -c -o $@
