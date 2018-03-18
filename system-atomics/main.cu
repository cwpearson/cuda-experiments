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

__global__ void gpu_touch(const int dev)
{
  // should come in as args
  const int numIters = 10;
  int *hist;

  using namespace cooperative_groups;

  auto mg = this_multi_grid();
  // auto mg = this_thread_block();

  for (int iter = 0; iter < numIters; ++iter)
  {
    mg.sync();

    if (iter % dev == 0)
    {
      long long start = clock64();
      atomicAdd_system(&hist[0], 1);
      long long end = clock64();
    }
  }
}

std::vector<void *> box(int dev, int *hist)
{
  std::vector<void *> ret;

  {
    auto box = new int;
    *box = dev;
    ret.push_back(static_cast<void *>(box));
  }

  {
    auto box = new int *;
    *box = hist;
    ret.push_back(static_cast<void *>(box));
  }

  return ret;
}

int main(void)
{

  // create streams
  std::vector<cudaStream_t> streams(2);
  for (size_t i = 0; i < streams.size(); ++i)
  {
    RT_CHECK(cudaSetDevice(i));
    RT_CHECK(cudaStreamCreate(&streams[i]));
  }

  int *hist;
  RT_CHECK(cudaMallocManaged(&hist, 4096));

  // create argument lists
  std::vector<std::vector<void *>> kernelArgsList(2);
  for (size_t i = 0; i < kernelArgsList.size(); ++i)
  {
    kernelArgsList[i] = box(i, hist);
  }

  // create launch parameters lists
  std::vector<cudaLaunchParams> paramsList(2);

  for (size_t i = 0; i < paramsList.size(); ++i)
  {
    auto &params = paramsList[i];

    params.func = (void *)gpu_touch;
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
