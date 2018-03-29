#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/common.hpp"

template <bool NOOP = false>
__global__ void gpu_traverse(size_t *ptr, const size_t steps, long long *clocks)
{

  if (NOOP)
  {
    return;
  }
  size_t next = 0;
  long long start = clock64();
  for (int i = 0; i < steps; ++i)
  {
    next = ptr[next];
  }
  long long end = clock64();
  ptr[next] = 1;

  clocks[0] = end - start;
}

template <typename data_type, bool NOOP = false>
__global__ void gpu_write(data_type *ptr, long long *clocks)
{
  if (NOOP)
  {
    return;
  }

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;

  if (gx == 0)
  {
    long long start = clock64();
    atomicAdd(&ptr[gx], 31);
    long long end = clock64();
    clocks[gx] = end - start;
  }
}

static void prefetch_bw(const int dstDev, const size_t steps)
{

  // Determine grid dimensions
  dim3 blockDim(1);
  dim3 gridDim(1);

  size_t *managedPtr, *explicitPtr;
  long long *clocks_d, *clocks_h;

  RT_CHECK(cudaSetDevice(dstDev));
  RT_CHECK(cudaMalloc(&clocks_d, sizeof(long long)));
  clocks_h = new long long;

  const size_t stride = 65536ul * 2ul;

  const size_t count = sizeof(size_t) * (steps + 1) * stride;
  RT_CHECK(cudaMallocManaged(&managedPtr, count));
  RT_CHECK(cudaMalloc(&explicitPtr, count));

  // set up stride
  for (size_t i = 0; i < steps; ++i)
  {
    managedPtr[i * stride] = (i + 1) * stride;
  }
  RT_CHECK(cudaMemcpy(explicitPtr, managedPtr, count, cudaMemcpyDefault));
  RT_CHECK(cudaDeviceSynchronize());

  std::vector<double> managedTimes, explicitTimes;
  const size_t numIters = 10;
  for (size_t i = 0; i < numIters; ++i)
  {
    // Try to get allocation on source
    nvtxRangePush("prefetch to cpu");
    RT_CHECK(cudaMemPrefetchAsync(managedPtr, count, cudaCpuDeviceId));
    RT_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // Access from Device and Time
    nvtxRangePush("managed traverse");
    auto start = std::chrono::high_resolution_clock::now();
    gpu_traverse<<<gridDim, blockDim>>>(managedPtr, steps, clocks_d);
    RT_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    RT_CHECK(cudaMemcpy(clocks_h, clocks_d, sizeof(long long), cudaMemcpyDefault));
    managedTimes.push_back((end - start).count() / 1e3);

    // std::cout << ":" << *clocks_h << " " << (end - start).count() << "\n";

    // Explicit traverse
    nvtxRangePush("explicit traverse");
    start = std::chrono::high_resolution_clock::now();
    gpu_traverse<<<gridDim, blockDim>>>(explicitPtr, steps, clocks_d);
    RT_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    RT_CHECK(cudaMemcpy(clocks_h, clocks_d, sizeof(long long), cudaMemcpyDefault));
    explicitTimes.push_back((end - start).count());

    // std::cout << ":" << *clocks_h << " " << (end - start).count() << "\n";
  }

  printf("%lu", steps);
  assert(managedTimes.size());
  const double minTime = *std::min_element(managedTimes.begin(), managedTimes.end());
  const double minExplicitTime = *std::min_element(explicitTimes.begin(), explicitTimes.end());
  const double avgTime = std::accumulate(managedTimes.begin(), managedTimes.end(), 0.0) / managedTimes.size();

  printf(",%.2f,%.2f", minTime, minExplicitTime);
  RT_CHECK(cudaFree(explicitPtr));
  RT_CHECK(cudaFree(managedPtr));
  RT_CHECK(cudaFree(clocks_d));
  delete clocks_h;
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

  // print header
  printf("# Strides,");

  for (const auto dst : devIds)
  {
    printf("GPU%d Traversal Time (us) [managed], GPU%d Traversal Time (us) [explicit]", dst, dst);
  }

  printf("\n");

  for (size_t numSteps = 4; numSteps < 24; ++numSteps)
  {
    for (const auto dst : devIds)
    {
      prefetch_bw(dst, numSteps);
    }
    printf("\n");
  }

  return 0;
}
