#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <vector>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/common.hpp"

static void prefetch_bw(const int dstDev, const int srcDev, const size_t count, const size_t stride)
{

  if (srcDev != cudaCpuDeviceId)
  {
    RT_CHECK(cudaSetDevice(srcDev));
    RT_CHECK(cudaFree(0));
  }

  if (dstDev != cudaCpuDeviceId)
  {
    RT_CHECK(cudaSetDevice(dstDev));
    RT_CHECK(cudaFree(0));
  }

  if (srcDev != cudaCpuDeviceId)
  {
    RT_CHECK(cudaSetDevice(srcDev));
  }

  void *ptr;

  RT_CHECK(cudaMallocManaged(&ptr, count));

  double totalWork = 0;
  const size_t numIters = 20;
  for (size_t i = 0; i < numIters; ++i)
  {
    // Try to get allocation on source
    nvtxRangePush("move to src");
    RT_CHECK(cudaMemPrefetchAsync(ptr, count, srcDev));
    RT_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // Prefetch to device and time.
    nvtxRangePush("tx");
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaMemPrefetchAsync(ptr, count, dstDev));
    RT_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    std::chrono::duration<double> txSeconds = end - start;
    totalWork += txSeconds.count();
  }

  std::cout << "," << count / 1024.0 / 1024.0 / (totalWork / numIters);
  RT_CHECK(cudaFree(ptr));
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);

  int numDevs;
  RT_CHECK(cudaGetDeviceCount(&numDevs));

  std::vector<int> devIds;
  for (int dev = 0; dev < 3; ++dev)
  {
    devIds.push_back(dev);
  }
  devIds.push_back(cudaCpuDeviceId);

  // print header
  std::cout << "Transfer Size (MB),";
  for (const auto src : devIds)
  {
    for (const auto dst : devIds)
    {
      if (src != dst)
      {
	std::cout << src << ":" << dst <<",";
      }
    }
  }
  std::cout << "\n";

  for (size_t count = 2048; count <= 4 * 1024ul * 1024ul * 1024ul; count *= 2) {
    std::cout << count / 1024.0 / 1024.0;
    for (const auto src : devIds)
    {
      for (const auto dst : devIds)
      {
        if (src != dst)
        {
            prefetch_bw(dst, src, count, pageSize);
        }
      }
    }
            std::cout << "\n";
  }

  return 0;
}
