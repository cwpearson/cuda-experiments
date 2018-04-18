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

static void memcpy_bw(const Device &dst, const Device &src, const size_t count, const int numIters)
{

  assert((src.is_cpu()) && (dst.is_cpu()));
  const long pageSize = sysconf(_SC_PAGESIZE);

  void *srcPtr, *dstPtr;

  bind_cpu(src);
  srcPtr = aligned_alloc(pageSize, count);
  std::memset(srcPtr, 0, count);

  bind_cpu(dst);
  dstPtr = aligned_alloc(pageSize, count);
  std::memset(dstPtr, 0, count);

  std::vector<double> times;
  for (int i = 0; i < numIters; ++i)
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

  printf(",%.2f", count / 1024.0 / 1024.0 / minTime);
  free(srcPtr);
  free(dstPtr);
}

int main(int argc, char **argv)
{
  int numIters = 10;
  std::vector<int> numaIds;
  if (option_as_int(argc, argv, "-n", numIters))
  {
    fprintf(stderr, "Using %d iterations\n", numIters);
  }

  if (option_as_int_list(argc, argv, "-c", numaIds))
  {
    fprintf(stderr, "Using CPU subset\n");
  }

  std::vector<Device> cpus = get_cpus(numaIds);

  // print header
  printf("Transfer Size (MB)");
  for (const auto src : cpus)
  {
    for (const auto dst : cpus)
    {
      printf(",%s to %s", src.name().c_str(), dst.name().c_str());
    }
  }

  printf("\n");

  auto freeMem = cpu_free_memory(cpus);
  freeMem /= 2; // two allocations
  freeMem = std::min(freeMem, 4ll * 1024ll * 1024ll * 1024ll);
  auto counts = Sequence::geometric(2048, freeMem, 2) |
                Sequence::geometric(2048 * 1.5, freeMem, 2);

  for (auto count : counts)
  {
    printf("%f", count / 1024.0 / 1024.0);
    for (const auto src : cpus)
    {
      for (const auto dst : cpus)
      {
        memcpy_bw(src, dst, count, numIters);
      }
    }

    printf("\n");
  }

  return 0;
}
