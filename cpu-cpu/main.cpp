#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

#include <unistd.h>
#include <omp.h>
#include <numa.h>

#include "common/common.hpp"
#include "op.hpp"

static int get_num_cpus(const Device &d) {
  bitmask *mask = numa_allocate_cpumask();
  numa_node_to_cpus(d.id(), mask);
  int num_cpus = numa_bitmask_weight(mask);
  numa_free_cpumask(mask);
  return num_cpus;
}

static void prefetch_bw(const Device &dstDev, const Device &srcDev, const size_t count, const size_t stride)
{

  assert(srcDev.is_cpu() && dstDev.is_cpu());
  const long pageSize = sysconf(_SC_PAGESIZE);

  // create source allocation
#ifdef OP_RD
  //printf("\nalloc on %s\n", srcDev.name().c_str());
  bind_cpu(srcDev);
#elif OP_WR
  bind_cpu(dstDev);
#else
#error "woah"
#endif
  double *ptr = static_cast<double *>(aligned_alloc(pageSize, count));
  std::memset(ptr, 0, count);

int num_cpus = 0;
#ifdef OP_RD
  bind_cpu(dstDev);
  num_cpus = get_num_cpus(dstDev);
#elif OP_WR
  bind_cpu(srcDev);
  num_cpus = get_num_cpus(srcDev);
#else
#error "woah"
#endif

  assert(num_cpus >0);
  //printf("\n::%d::\n", num_cpus);
  omp_set_num_threads(num_cpus);

#pragma omp parallel
{
#ifdef OP_RD
  //printf("\nrd on %s\n", dstDev.name().c_str());
  bind_cpu(dstDev);
#elif OP_WR
  bind_cpu(srcDev);
#else
#error "woah"
#endif
}

  std::vector<double> times;
  double dummy;
  const size_t numIters = 15;
  for (size_t i = 0; i < numIters; ++i)
  {
    // Access from Device and Time

    auto start = std::chrono::high_resolution_clock::now();
#ifdef OP_RD
    cpu_read_8(&dummy, ptr, count, stride);
#elif OP_WR
    cpu_write_8(&dummy, ptr, count, stride);
#else
#error "woah"
#endif
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    times.push_back(txSeconds.count());
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  const double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  printf(",%.2f", count / 1024.0 / 1024.0 / minTime);
  free(ptr);
}

int main(void)
{
  numa_set_strict(1);
  numa_exit_on_error = 1;
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
