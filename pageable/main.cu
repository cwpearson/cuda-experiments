#include <algorithm>
#include <cstring>
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

static void pageable_bw(const Device &dst, const Device &src, const size_t count, const int numIters)
{

  assert((src.is_cpu()) ^ (dst.is_cpu()));
  const long pageSize = sysconf(_SC_PAGESIZE);

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
  hostPtr = aligned_alloc(pageSize, count);
  assert((0 == count) || hostPtr);
  std::memset(hostPtr, 0, count);

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
  for (int i = 0; i < numIters; ++i)
  {
    nvtxRangePush("dst");
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaMemcpy(dstPtr, srcPtr, count, cudaMemcpyDefault));
    RT_CHECK(cudaDeviceSynchronize()); // cudaMemcpy may return before DMA is done for pageable
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    nvtxRangePop();
    times.push_back(txSeconds.count());
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  const double avgTime =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  printf(",%.2f", count / 1024.0 / 1024.0 / minTime);
  free(hostPtr);
  RT_CHECK(cudaFree(devPtr));
}

int main(int argc, char **argv)
{

  int numIters = 10;
  std::vector<int> gpuIds;
  std::vector<int> numaIds;
  if (option_as_int(argc, argv, "-n", numIters))
  {
    fprintf(stderr, "Using %d iterations\n", numIters);
  }
  if (option_as_int_list(argc, argv, "-c", numaIds))
  {
    fprintf(stderr, "Using CPU subset\n");
  }
  if (option_as_int_list(argc, argv, "-g", gpuIds))
  {
    fprintf(stderr, "Using GPU subset\n");
  }

  std::vector<Device> gpus = get_gpus(gpuIds);
  std::vector<Device> cpus = get_cpus(numaIds);

  if (gpus.empty())
  {
    fprintf(stderr, "no gpus\n");
    return 1;
  }

  if (cpus.empty())
  {
    fprintf(stderr, "no cpus\n");
    return 1;
  }

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
        pageable_bw(gpu, cpu, count, numIters);
      }
    }
    //gpu->cpu
    for (const auto cpu : cpus)
    {
      for (const auto gpu : gpus)
      {
        pageable_bw(cpu, gpu, count, numIters);
      }
    }

    printf("\n");
  }

  return 0;
}
