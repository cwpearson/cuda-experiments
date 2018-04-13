#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/common.hpp"

template <typename data_type, size_t REPEATS = 1024>
__global__ void
gpu_touch(data_type *__restrict__ hist, const size_t *__restrict__ idx,
          double *__restrict__ threadTimes, const bool noop = false)
{
  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // where to increment
  const size_t voteIdx = idx[gx];

  hist[voteIdx] = 0;
  __syncthreads();

  const long long int start = clock64();
  if (!noop)
  {
#pragma unroll REPEATS
    for (size_t iter = 0; iter < REPEATS; ++iter)
    {
      atomicAdd(&hist[voteIdx], gx);
    }
  }
  const long long int end = clock64();

  threadTimes[gx] = double(end - start) / REPEATS;
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);
  std::stringstream buffer;

  RT_CHECK(cudaFree(0));

  size_t memTotal, memAvail;
  RT_CHECK(cudaMemGetInfo(&memAvail, &memTotal));

  typedef int data_type;

  // number of warps making atomic accesses to same location
  // const size_t interWarpConflict;
  // number of threads w/in a warp making atomic accesses
  // const size_t intraWarpConflict;

  for (size_t stride = sizeof(data_type); stride <= 1024 * sizeof(data_type);
       stride *= 2)
  {
    for (size_t numWay = 4; numWay <= 4; ++numWay)
    {

      assert(stride % sizeof(data_type) == 0);

      buffer.str("");
      buffer << "w=" << numWay << " s=" << stride;
      nvtxRangePush(buffer.str().c_str());

      nvtxRangePush("setup");

      // Number of threads
      dim3 dimGrid(1);
      dim3 dimBlock(32);
      const size_t numThreads = dimGrid.x * dimBlock.x;

      // Allocation tracking of thread times
      double *threadTimes_h, *threadTimes_d;
      threadTimes_h = new double[numThreads];
      RT_CHECK(cudaMalloc(&threadTimes_d, numThreads * sizeof(double)));

      // Allocate the histogram
      data_type *hist;
      RT_CHECK(cudaMalloc(&hist, stride * numThreads * sizeof(data_type)));

      // Allocate thread access indices
      size_t *idx_h, *idx_d;
      idx_h = new size_t[numThreads];
      RT_CHECK(cudaMalloc(&idx_d, numThreads * sizeof(size_t)));

      // initialize thread access indices
      for (size_t i = 0; i < numThreads; ++i)
      {
        size_t lx = i % 32;
        size_t idx;
        if (lx < numWay)
        {
          idx = 0;
        }
        else
        {
          idx = i;
        }
        idx_h[i] = idx * (stride / sizeof(data_type));
      }

      // Copy thread access indices to device
      RT_CHECK(cudaMemcpy(idx_d, idx_h, numThreads * sizeof(float),
                          cudaMemcpyDefault));
      RT_CHECK(cudaDeviceSynchronize());
      nvtxRangePop();

      // work loop
      nvtxRangePush("work");
      gpu_touch<<<dimGrid, dimBlock>>>(hist, idx_d, threadTimes_d);
      RT_CHECK(cudaGetLastError());
      nvtxRangePop();

      // Get thread times back
      RT_CHECK(cudaMemcpy(threadTimes_h, threadTimes_d,
                          sizeof(double) * numThreads, cudaMemcpyDefault));

      // Average, min, max thread times
      const double maxCycles =
          *std::max_element(threadTimes_h, &threadTimes_h[numThreads]);

      // std::cout << "s=" << stride << ": "
      //           << "w=" << numWay << ": " << maxCycles << "\n";
      std::cout << stride << ", " << numWay << ", " << maxCycles << "\n";

      // free memory
      nvtxRangePush("cleanup");
      RT_CHECK(cudaFree(hist));
      RT_CHECK(cudaFree(idx_d));
      RT_CHECK(cudaFree(threadTimes_d));
      delete[] idx_h;
      delete[] threadTimes_h;
      nvtxRangePop();

      nvtxRangePop();
    }
  }
  return 0;
}
