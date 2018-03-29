#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <nvToolsExt.h>
#include <cooperative_groups.h>

#include <unistd.h>

#include "common/common.hpp"

template <size_t NUM_ITERS>
__global__ void gpu_sync(long long *clocks, const int dev)
{

  using namespace cooperative_groups;

  const size_t gx = blockDim.x * blockIdx.x + threadIdx.x;

  multi_grid_group mg = this_multi_grid();

  long long start = clock64();
#pragma unroll(NUM_ITERS)
  for (int iter = 0; iter < NUM_ITERS; ++iter)
  {
    mg.sync();
  }
  long long end = clock64();

  clocks[gx] = (end - start) / NUM_ITERS;
}

std::vector<void *> box(long long *clocks, const int dev)
{

#define BOX(v, T)                            \
  {                                          \
    auto box = new T;                        \
    *box = v;                                \
    ret.push_back(static_cast<void *>(box)); \
  }

  std::vector<void *> ret;

  BOX(clocks, long long *);
  BOX(dev, int);

#undef BOX

  return ret;
}

int main(void)
{

  std::vector<int> devices = {0, 1};

  // create streams
  std::vector<cudaStream_t> streams(2);
  for (size_t i = 0; i < streams.size(); ++i)
  {
    RT_CHECK(cudaSetDevice(i));
    RT_CHECK(cudaStreamCreate(&streams[i]));
  }

  // determine kernel parameters
  dim3 gridDim(1);
  dim3 blockDim(32);
  const size_t numThreads = gridDim.x * blockDim.x;

  // allocate clock arrays
  std::vector<long long *> devClocks(2);
  std::vector<long long *> hostClocks(2);
  for (size_t dev = 0; dev < devClocks.size(); ++dev)
  {
    RT_CHECK(cudaSetDevice(dev));
    RT_CHECK(cudaMalloc(&devClocks[dev], numThreads * sizeof(long long)));
    hostClocks[dev] = new long long[numThreads];
  }

  // create argument lists
  std::vector<std::vector<void *>> kernelArgsList(2);
  for (size_t dev = 0; dev < kernelArgsList.size(); ++dev)
  {
    kernelArgsList[dev] = box(devClocks[dev], dev);
  }

  // create launch parameters lists
  std::vector<cudaLaunchParams> paramsList(2);

  for (size_t i = 0; i < paramsList.size(); ++i)
  {
    auto &params = paramsList[i];

    params.func = (void *)gpu_sync<1>;
    params.gridDim = gridDim;
    params.blockDim = blockDim;
    params.args = &(kernelArgsList[i][0]);
    params.sharedMem = 0;
    params.stream = streams[i];
  }

  RT_CHECK(cudaLaunchCooperativeKernelMultiDevice(&paramsList[0], 2));

  // copy clocks back to host
  for (size_t dev = 0; dev < devClocks.size(); ++dev)
  {
    RT_CHECK(cudaMemcpy(hostClocks[dev], devClocks[dev], numThreads * sizeof(long long), cudaMemcpyDefault));
  }

  // Print some host clocks:
  for (size_t dev = 0; dev < devClocks.size(); ++dev)
  {
    std::cout << "dev: " << dev << std::endl;
    for (size_t i = 0; i < numThreads; ++i)
    {
      std::cerr << hostClocks[dev][i] << " ";
    }
    std::cout << std::endl;
  }

  nvtxRangePush("cleanup");
  for (auto &stream : streams)
  {
    RT_CHECK(cudaStreamDestroy(stream));
  }

  // for (auto &kernelArgs : kernelArgsList)
  // {
  //   for (auto &arg : kernelArgs)
  //   {
  //     delete arg;
  //     arg = nullptr;
  //   }
  // }

  nvtxRangePop();

  return 0;
}
