#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <chrono>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/cuda_check.hpp"
#include "common/common.hpp"

template <typename data_type>
void cpu_touch(data_type *c, const size_t stride, const size_t n)
{

  const size_t numElems = n / sizeof(data_type);
  const size_t elemsPerStride = stride / sizeof(data_type);

  for (size_t i = 0; i < numElems; i += elemsPerStride)
  {
    c[i] = i * 31ul + 7ul;
  }
}

template <typename data_type>
__global__ void gpu_touch(data_type *ptr, const size_t numWay, const size_t stride, const size_t numIters, const size_t footprint, const bool noop = false)
{
  if (noop)
  {
    return;
  }

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // way ID
  const size_t wx = gx / numWay;
  // convert stride in bytes to stride in data_type
  const size_t ptr_stride = stride / sizeof(data_type);
  // convert n in bytes to n in data_type
  const size_t ptr_footprint = footprint / sizeof(data_type);
  // where this thread will atomic
  const size_t idx = wx * ptr_stride;

  if (idx < ptr_footprint)
#pragma unroll
    for (size_t iter = 0; iter < numIters; ++iter)
    {
      {
        atomicAdd(&ptr[idx], 1);
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

  typedef int data_type;
  data_type *ptr;

  RT_CHECK(cudaMallocManaged(&ptr, pageSize * 32));
  RT_CHECK(cudaDeviceSynchronize());

  const int numIters = 100;
  const size_t footprint = pageSize * 32;

  // number of warps making atomic accesses to same location
  // const size_t interWarpConflict;
  // number of threads w/in a warp making atomic accesses
  // const size_t intraWarpConflict;

  for (size_t numWay = 1; numWay <= 64; numWay *= 2)
  {
    for (size_t stride = sizeof(data_type); stride <= footprint; stride *= 2)
    {

      buffer.str("");
      buffer << "w=" << numWay << " s=" << stride;
      nvtxRangePush(buffer.str().c_str());

      RT_CHECK(cudaMallocManaged(&ptr, footprint));
      RT_CHECK(cudaMemset(ptr, 0, footprint));
      RT_CHECK(cudaDeviceSynchronize());

      // numWays threads per stride within footprint
      const size_t numThreads = numWay * (footprint / stride);
      dim3 dimBlock(256);
      dim3 dimGrid((numThreads + dimBlock.x - 1) / dimBlock.x);
      const size_t numAtomics = numThreads * numIters;

      // work loop
      nvtxRangePush("work");
      auto start = std::chrono::high_resolution_clock::now();
      gpu_touch<<<dimGrid, dimBlock>>>(ptr, numWay, stride, numIters, footprint);
      RT_CHECK(cudaDeviceSynchronize());
      auto end = std::chrono::high_resolution_clock::now();
      nvtxRangePop();
      std::chrono::duration<double> workSeconds = end - start;

      // empty loop
      nvtxRangePush("noop");
      start = std::chrono::high_resolution_clock::now();
      gpu_touch<<<dimGrid, dimBlock>>>(ptr, numWay, stride, numIters, footprint, true /*no-op*/);
      RT_CHECK(cudaDeviceSynchronize());
      end = std::chrono::high_resolution_clock::now();
      nvtxRangePop();
      std::chrono::duration<double> noopSeconds = end - start;

      std::cout << "w=" << numWay << ": "
                << "s=" << stride << ": "
                << numAtomics / (workSeconds.count() - noopSeconds.count()) << " atomics/s (" << numAtomics << ") "
                << (workSeconds.count() - noopSeconds.count()) / numIters << "s latency\n";

      // check correctness
      const size_t ptrStride = stride / sizeof(data_type);
      const size_t ptrFootprint = footprint / sizeof(data_type);
      for (size_t i = 0; i < ptrFootprint; i += ptrStride)
      {
        if (ptr[i] != numWay * numIters)
        {
          std::cerr << "gah\n";
          assert(0);
        }
      }

      //free memory
      RT_CHECK(cudaFree(ptr));

      nvtxRangePop();
    }
  }
  return 0;
}
