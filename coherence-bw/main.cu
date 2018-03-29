#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/common.hpp"

template <typename data_type>
void cpu_write(data_type *ptr, const size_t count, const size_t stride)
{

  const size_t numElems = count / sizeof(data_type);
  const size_t elemsPerStride = stride / sizeof(data_type);

  for (size_t i = 0; i < numElems; i += elemsPerStride)
  {
    ptr[i] = i * 31ul + 7ul;
  }
}

template <typename data_type, bool NOOP = false>
__global__ void gpu_write(data_type *ptr, const size_t count, const size_t stride)
{
  if (NOOP)
  {
    return;
  }

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = gx & 31;
  // warp ID
  size_t wx = gx / 32;
  // number of warps in the grid
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;
  // number of strides in N bytes
  const size_t numStrides = count / stride;
  const size_t numData = count / sizeof(data_type);
  // number of data_types in each
  const size_t dataPerStride = stride / sizeof(data_type);

  if (0 == lx)
  {

    for (; wx < numStrides; wx += numWarps)
    {
      const size_t id = wx * dataPerStride;
      if (id < numData)
      {
        ptr[id] = id * 31ul + 7ul;
      }
    }
  }
}

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

  // Determine grid dimensions
  dim3 blockDim(256);
  const size_t numStrides = (count + stride - 1) / stride;
  dim3 gridDim((numStrides + blockDim.x - 1) / blockDim.x);

  void *ptr;

  RT_CHECK(cudaMallocManaged(&ptr, count));

  std::vector<double> times;
  const size_t numIters = 20;
  for (size_t i = 0; i < numIters; ++i)
  {
    // Try to get allocation on source
    nvtxRangePush("prefetch to src");
    RT_CHECK(cudaMemPrefetchAsync(ptr, count, srcDev));
    RT_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    if (dstDev != cudaCpuDeviceId)
    {
      RT_CHECK(cudaSetDevice(dstDev));
    }

    // Access from Device and Time
    nvtxRangePush("dst");
    auto start = std::chrono::high_resolution_clock::now();
    if (cudaCpuDeviceId == dstDev)
    {
      cpu_write((int *)ptr, count, stride);
    }
    else
    {
      gpu_write<<<gridDim, blockDim>>>((int *)ptr, count, stride);
      RT_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    nvtxRangePop();
    times.push_back(txSeconds.count());
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  const double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  std::cout << "," << count / 1024.0 / 1024.0 / (minTime);
  RT_CHECK(cudaFree(ptr));
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);

  int numDevs;
  RT_CHECK(cudaGetDeviceCount(&numDevs));

  std::vector<int> devIds;
  for (int dev = 0; dev < numDevs; ++dev)
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
        std::cout << src << ":" << dst << ",";
      }
    }
  }
  std::cout << "\n";

  for (size_t count = 2048; count <= 4 * 1024ul * 1024ul * 1024ul; count *= 2)
  {
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
