#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/common.hpp"

static void prefetch_bw(const Device &dstDev, const Device &srcDev, const size_t count)
{

  if (srcDev.is_gpu())
  {
    RT_CHECK(cudaSetDevice(srcDev.cuda_device_id()));
    RT_CHECK(cudaFree(0));
  }

  if (dstDev.is_gpu())
  {
    RT_CHECK(cudaSetDevice(dstDev.cuda_device_id()));
    RT_CHECK(cudaFree(0));
  }

  if (srcDev.is_gpu())
  {
    RT_CHECK(cudaSetDevice(srcDev.cuda_device_id()));
  }

  void *ptr;

  RT_CHECK(cudaMallocManaged(&ptr, count));

  std::vector<double> times;
  const size_t numIters = 20;
  for (size_t i = 0; i < numIters; ++i)
  {
    // Try to get allocation on source
    nvtxRangePush("move to src");
    RT_CHECK(cudaMemPrefetchAsync(ptr, count, srcDev.cuda_device_id()));
    RT_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // Prefetch to device and time.
    nvtxRangePush("tx");
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaMemPrefetchAsync(ptr, count, dstDev.cuda_device_id()));
    RT_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    std::chrono::duration<double> txSeconds = end - start;
    times.push_back(txSeconds.count());
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  const double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  printf(",%.2f", count / 1024.0 / 1024.0 / (minTime));

  RT_CHECK(cudaFree(ptr));
}

int main(void)
{
  const long pageSize = sysconf(_SC_PAGESIZE);

  int numDevs;
  RT_CHECK(cudaGetDeviceCount(&numDevs));

  std::vector<int> devIds;
  auto cpus = get_cpus();
  auto gpus = get_gpus();
  auto devs = cpus;
  for (const auto g : gpus)
  {
    devs.push_back(g);
  }

  size_t freeMem = -1ul;
  for (auto d : devs)
  {
    if (d.is_gpu())
    {
      size_t fr, to;
      RT_CHECK(cudaMemGetInfo(&fr, &to));

      if (fr < freeMem)
      {
        freeMem = fr;
      }
    }
  }

  // print header
  printf("Transfer Size (MB),");
  for (const auto src : devs)
  {
    for (const auto dst : devs)
    {
      if (src != dst)
      {
        printf("%s to %s (prefetch),", src.name().c_str(), dst.name().c_str());
      }
    }
  }
  printf("\n");

  auto counts = Sequence::geometric(2048, freeMem, 2) |
                Sequence::geometric(2048 * 1.5, freeMem, 2);

  for (auto count : counts)
  {
    printf("%f", count / 1024.0 / 1024.0);
    for (const auto src : devs)
    {
      for (const auto dst : devs)
      {
        if (src != dst)
        {
          prefetch_bw(dst, src, count);
        }
      }
    }
    printf("\n");
  }

  return 0;
}
