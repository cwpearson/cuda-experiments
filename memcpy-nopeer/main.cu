#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <vector>

#include <numa.h>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/cuda_check.hpp"
#include "common/common.hpp"

static void gpu_gpu_bw(const Device &dst, const Device &src, const size_t count)
{

  assert(src.is_gpu() && dst.is_gpu());

  void *srcPtr, *dstPtr;

  RT_CHECK(cudaSetDevice(src.id()));
  RT_CHECK(cudaMalloc(&srcPtr, count));
  {
    cudaError_t err = cudaDeviceDisablePeerAccess(dst.id());
    if (err != cudaErrorPeerAccessNotEnabled)
    {
      RT_CHECK(err);
    }
  }
  RT_CHECK(cudaSetDevice(dst.id()));
  RT_CHECK(cudaMalloc(&dstPtr, count));
  {
    cudaError_t err = cudaDeviceDisablePeerAccess(src.id());
    if (err != cudaErrorPeerAccessNotEnabled)
    {
      RT_CHECK(err);
    }
  }

  std::vector<double> times;
  const size_t numIters = 20;
  for (size_t i = 0; i < numIters; ++i)
  {
    nvtxRangePush("dst");
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaMemcpy(dstPtr, srcPtr, count, cudaMemcpyDefault));
    RT_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    nvtxRangePop();
    times.push_back(txSeconds.count());
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  const double avgTime =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  printf(",%.2f", count / 1024.0 / 1024.0 / minTime);
  RT_CHECK(cudaFree(srcPtr));
  RT_CHECK(cudaFree(dstPtr));
}

int main(void)
{

  const size_t numNodes = numa_max_node();

  const long pageSize = sysconf(_SC_PAGESIZE);

  std::vector<Device> gpus = get_gpus();

  // print header
  printf("Transfer Size (MB)");
  for (const auto dst : gpus)
  {
    for (const auto src : gpus)
    {
      if (src != dst)
      {
        printf(",%s to %s (no peer)", src.name().c_str(), dst.name().c_str());
      }
    }
  }

  printf("\n");

  auto freeMem = gpu_free_memory(gpus);
  auto counts = Sequence::geometric(2048, freeMem, 2) |
                Sequence::geometric(2048 * 1.5, freeMem, 2);

  for (auto count : counts)
  {
    printf("%f", count / 1024.0 / 1024.0);
    for (const auto dst : gpus)
    {
      for (const auto src : gpus)
      {

        if (src != dst)
        {

          gpu_gpu_bw(dst, src, count);
        }
      }
    }

    printf("\n");
  }

  return 0;
}
