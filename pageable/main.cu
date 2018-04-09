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

#include "common/common.hpp"

static void pinned_bw(const Device &dst, const Device &src, const size_t count)
{

  assert((src.is_cpu()) ^ (dst.is_cpu()));

  void *devPtr, *hostPtr;
  void *srcPtr, *dstPtr;

  if (src.is_gpu())
  {
    RT_CHECK(cudaSetDevice(src.id()));
    bind_cpu(dst);
  }
  else
  {
    RT_CHECK(cudaSetDevice(dst.id()));
    bind_cpu(src);
  }

  RT_CHECK(cudaFree(0));
  RT_CHECK(cudaMalloc(&devPtr, count));
  hostPtr = new char[count];
  assert((0 == count) || hostPtr);

  if (src.is_gpu())
  {
    srcPtr = devPtr;
    dstPtr = hostPtr;
  }
  else
  {
    srcPtr = hostPtr;
    dstPtr = devPtr;
  }

  std::vector<double> times;
  const size_t numIters = 20;
  for (size_t i = 0; i < numIters; ++i)
  {
    nvtxRangePush("dst");
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaMemcpy(dstPtr, srcPtr, count, cudaMemcpyDefault));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    nvtxRangePop();
    times.push_back(txSeconds.count());
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  const double avgTime =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  printf(",%.2f", count / 1024.0 / 1024.0 / minTime);
  delete[](char *) hostPtr;
  RT_CHECK(cudaFree(devPtr));
}

int main(void)
{

  const size_t numNodes = numa_max_node();

  const long pageSize = sysconf(_SC_PAGESIZE);

  std::vector<Device> gpus = get_gpus();
  std::vector<Device> cpus = get_cpus();

  // print header
  printf("Transfer Size (MB)");
  // cpu->gpu
  for (const auto cpu : cpus)
  {
    for (const auto gpu : gpus)
    {
      printf(",%s to %s (pageable)", cpu.name().c_str(), gpu.name().c_str());
    }
  }
  //gpu->cpu
  for (const auto cpu : cpus)
  {
    for (const auto gpu : gpus)
    {
      printf(",%s to %s (pageable)", gpu.name().c_str(), cpu.name().c_str());
    }
  }

  printf("\n");

  auto freeMem = gpu_free_memory(gpus);
  auto counts = Sequence::geometric(2048, freeMem, 2) |
                Sequence::geometric(2048 * 1.5, freeMem, 2);

  for (auto count : counts)
  {
    printf("%f", count / 1024.0 / 1024.0);
    //cpu->gpu
    for (const auto cpu : cpus)
    {
      for (const auto gpu : gpus)
      {
        pinned_bw(gpu, cpu, count);
      }
    }
    //gpu->cpu
    for (const auto cpu : cpus)
    {
      for (const auto gpu : gpus)
      {
        pinned_bw(cpu, gpu, count);
      }
    }

    printf("\n");
  }

  return 0;
}
