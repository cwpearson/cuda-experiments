MODULE := cpu-cpu

TARGETS += $(MODULE)/dst-rd-src $(MODULE)/src-wr-dst
CLEAN_TARGETS += $(MODULE)/main_rd.o $(MODULE)/main_wr.o $(MODULE)/op.o

$(MODULE)/dst-rd-src: $(MODULE)/main_rd.o $(MODULE)/op.o $(COMMON_OBJECTS)
	$(NVCC) $^ -std=c++11  -o $@ -lnuma -lcudart -lgomp

$(MODULE)/src-wr-dst: $(MODULE)/main_wr.o $(MODULE)/op.o $(COMMON_OBJECTS)
	$(NVCC) $^ -std=c++11  -o $@ -lnuma -lcudart -lgomp

$(MODULE)/main_wr.o: $(MODULE)/main.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $< -DOP=WR -std=c++11 -Wall -Wextra -O3 -fopenmp -c -o $@

$(MODULE)/main_rd.o: $(MODULE)/main.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $< -DOP=RD -std=c++11 -Wall -Wextra -O3 -fopenmp -c -o $@