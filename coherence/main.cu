#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <chrono>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/common.hpp"

size_t cpu_touch(char *c, const size_t e, const size_t n)
{
  for (size_t i = 0; i < n; i += e)
  {
    c[i] = i * 31ul + 7ul;
  }
  return (n / e);
}

template <typename data_type>
__global__ void gpu_touch(data_type *c, const size_t stride, const size_t n, const bool noop = false)
{
  if (noop)
  {
    return;
  }

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = (blockDim.x * blockIdx.x + threadIdx.x) & 31;
  // warp ID
  size_t wx = gx / 32;
  // number of warps in the grid
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;
  // number of strides in N bytes
  const size_t numStrides = (n + stride - 1) / stride;
  // number of data_types in each
  const size_t elemsPerStride = stride / sizeof(data_type);

  if (0 == lx)
  {

    for (; wx < numStrides; wx += numWarps)
    {
      const size_t id = wx * elemsPerStride;
      if (id < numStrides)
      {
        c[id] = id * 31ul + 7ul;
      }
    }
  }
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);
  std::stringstream buffer;

  RT_CHECK(cudaFree(0));

  size_t memTotal, memAvail;
  RT_CHECK(cudaMemGetInfo(&memAvail, &memTotal));

  char *cm;

  RT_CHECK(cudaMallocManaged(&cm, pageSize * 32));
  RT_CHECK(cudaDeviceSynchronize());

  const int numIters = 80;

  for (size_t n = 4; n <= pageSize * 32; n *= 2)
  {
    buffer.str("");
    buffer << n;
    nvtxRangePush(buffer.str().c_str());

    RT_CHECK(cudaMallocManaged(&cm, n));
    RT_CHECK(cudaDeviceSynchronize());

    const size_t stride = 4096;

    // create enough warps to cover all the strides
    const size_t numThreads = 32 * ((n + stride - 1) / stride);
    dim3 dimBlock(128);
    dim3 dimGrid((numThreads + 128 - 1) / 128);
    // std::cout << numThreads << " " << dimGrid.x << "\n";

    // Loop with work
    nvtxRangePush("work");
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numIters; ++i)
    {

      nvtxRangePush("cpu");
      cpu_touch(cm, stride, n);
      RT_CHECK(cudaDeviceSynchronize());
      nvtxRangePop();
      nvtxRangePush("gpu");
      gpu_touch<<<dimGrid, dimBlock>>>(cm, stride, n);
      RT_CHECK(cudaDeviceSynchronize());
      nvtxRangePop();
    }
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    std::chrono::duration<double> workSeconds = end - start;

    // empty loop
    nvtxRangePush("noop");
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numIters; ++i)
    {
      nvtxRangePush("cpu");
      RT_CHECK(cudaDeviceSynchronize());
      nvtxRangePop();
      nvtxRangePush("gpu");
      gpu_touch<<<dimGrid, dimBlock>>>(cm, stride, n, true /*no-op*/);
      RT_CHECK(cudaDeviceSynchronize());
      nvtxRangePop();
    }
    end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    std::chrono::duration<double> emptySeconds = end - start;

    std::cout << "n=" << n << ": " << (workSeconds.count() - emptySeconds.count()) / numIters << " s/iter (" << numIters << ")\n";

    RT_CHECK(cudaFree(cm));
    nvtxRangePop();
  }

  return 0;
}
