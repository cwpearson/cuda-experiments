#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include <unistd.h>

#include "common/common.hpp"
#include "read.hpp"

static void prefetch_bw(const Device &dstDev, const Device &srcDev, const size_t count, const size_t stride)
{

  assert(srcDev.is_cpu() && dstDev.is_cpu());
  const long pageSize = sysconf(_SC_PAGESIZE);

  // create source allocation
  bind_cpu(srcDev);
  double *srcPtr = static_cast<double *>(aligned_alloc(pageSize, count));

  // bind to dst
  bind_cpu(dstDev);

  std::vector<double> times;
  const size_t numIters = 15;
  for (size_t i = 0; i < numIters; ++i)
  {
    // Try to get allocation on source

    // Access from Device and Time
    auto start = std::chrono::high_resolution_clock::now();
    cpu_write_8(srcPtr, count, stride);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    times.push_back(txSeconds.count());
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  const double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  printf(",%.2f", count / 1024.0 / 1024.0 / minTime);
  free(srcPtr);
}

int main(void)
{
  auto cpus = get_cpus();

  // print header
  printf("Transfer Size (MB),");
  for (const auto src : cpus)
  {
    for (const auto dst : cpus)
    {
      if (src != dst)
      {
        printf("%s to %s,", src.name().c_str(), dst.name().c_str());
      }
    }
  }
  printf("\n");

  auto counts = Sequence::geometric(2048, 4ul * 1024ul * 1024ul * 1024ul, 2) |
                Sequence::geometric(2048 * 1.5, 4ul * 1024ul * 1024ul * 1024ul, 2);

  for (auto count : counts)
  {
    printf("%f", count / 1024.0 / 1024.0);
    for (const auto src : cpus)
    {
      for (const auto dst : cpus)
      {
        if (src != dst)
        {
          prefetch_bw(dst, src, count, 8);
        }
      }
    }
    printf("\n");
  }

  return 0;
}
