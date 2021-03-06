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

static void gpu_gpu_bw(const Device &dst, const Device &src, const size_t count, const int numIters)
{

  assert(src.is_gpu() && dst.is_gpu());

  void *srcPtr, *dstPtr;

  RT_CHECK(cudaSetDevice(src.id()));
  RT_CHECK(cudaMalloc(&srcPtr, count));
  {
    cudaError_t err = cudaDeviceEnablePeerAccess(dst.id(), 0);
    if (err != cudaErrorPeerAccessAlreadyEnabled)
    {
      RT_CHECK(err);
    }
  }
  RT_CHECK(cudaSetDevice(dst.id()));
  RT_CHECK(cudaMalloc(&dstPtr, count));
  {
    cudaError_t err = cudaDeviceEnablePeerAccess(src.id(), 0);
    if (err != cudaErrorPeerAccessAlreadyEnabled)
    {
      RT_CHECK(err);
    }
  }

  std::vector<double> times;
  for (int i = 0; i < numIters; ++i)
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

int main(int argc, char **argv)
{
  int numIters = 10;
  std::vector<int> gpuIds;
  if (option_as_int(argc, argv, "-n", numIters))
  {
    fprintf(stderr, "Using %d iterations\n", numIters);
  }

  if (option_as_int_list(argc, argv, "-g", gpuIds))
  {
    fprintf(stderr, "Using GPU subset\n");
  }

  std::vector<Device> gpus = get_gpus(gpuIds);

  // print header
  printf("Transfer Size (MB)");
  for (const auto dst : gpus)
  {
    for (const auto src : gpus)
    {
      if (src != dst)
      {
        int can;
        RT_CHECK(cudaDeviceCanAccessPeer(&can, src.id(), dst.id()));
        if (can)
        {
          printf(",%s to %s (peer)", src.name().c_str(), dst.name().c_str());
        }
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
          int can;
          RT_CHECK(cudaDeviceCanAccessPeer(&can, src.id(), dst.id()));
          if (can)
          {

            gpu_gpu_bw(dst, src, count, numIters);
          }
        }
      }
    }

    printf("\n");
  }

  return 0;
}
