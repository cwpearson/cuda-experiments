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
__global__ void gpu_touch(long long *clocks, int *hist, const int dev, const int numIters)
{

  using namespace cooperative_groups;

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  auto mg = this_multi_grid();

  int ppCount = 0;
#pragma unroll(NUM_ITERS)
  for (int iter = 0; iter < NUM_ITERS; ++iter)
  {
    mg.sync();

    if (iter % dev == 0)
    {
      long long start = clock64();
      atomicAdd_system(&hist[0], 1);
      long long end = clock64();
      clocks[gx * NUM_ITERS + ppCount] = end - start;
      ++ppCount;
    }
  }

  while (ppCount < NUM_ITERS)
  {
    clocks[gx * NUM_ITERS + ppCount] = -1;
    ++ppCount;
  }
}

std::vector<void *> box(long long *clocks, int *hist, const int dev, const int numIters)
{

#define BOX(v, T)                            \
  {                                          \
    auto box = new T;                        \
    *box = v;                                \
    ret.push_back(static_cast<void *>(box)); \
  }

  std::vector<void *> ret;

  BOX(clocks, long long *);
  BOX(hist, int *);
  BOX(dev, int);
  BOX(numIters, int);

#undef BOX

  return ret;
}

int main(void)
{

  // number of ping-pongs
  const int numIters = 10;

  // create streams
  std::vector<cudaStream_t> streams(2);
  for (size_t i = 0; i < streams.size(); ++i)
  {
    RT_CHECK(cudaSetDevice(i));
    RT_CHECK(cudaStreamCreate(&streams[i]));
  }

  // allocate histogram
  int *hist;
  RT_CHECK(cudaMallocManaged(&hist, 4096));

  // allocate clock arrays
  std::vector<long long *> devClocks(2);
  std::vector<long long *> hostClocks(2);
  for (size_t dev = 0; dev < devClocks.size(); ++dev)
  {
    RT_CHECK(cudaSetDevice(dev));
    RT_CHECK(cudaMalloc((void **)&devClocks[dev][0], 32 * sizeof(long long)));
    hostClocks[dev] = new long long[32];
  }

  // create argument lists
  std::vector<std::vector<void *>> kernelArgsList(2);
  for (size_t dev = 0; dev < kernelArgsList.size(); ++dev)
  {
    kernelArgsList[dev] = box(devClocks[dev], hist, dev, numIters);
  }

  // create launch parameters lists
  std::vector<cudaLaunchParams> paramsList(2);

  for (size_t i = 0; i < paramsList.size(); ++i)
  {
    auto &params = paramsList[i];

    params.func = (void *)gpu_touch<1>;
    params.gridDim = dim3(1);
    params.blockDim = dim3(32);
    params.args = &(kernelArgsList[i][0]);
    params.sharedMem = 0;
    params.stream = streams[i];
  }

  RT_CHECK(cudaLaunchCooperativeKernelMultiDevice(&paramsList[0], 2));

  nvtxRangePush("cleanup");
  for (auto &stream : streams)
  {
    RT_CHECK(cudaStreamDestroy(stream));
  }

  RT_CHECK(cudaFree(hist));

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
