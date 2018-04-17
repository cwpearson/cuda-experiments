#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <climits>

#include <unistd.h>
#include <omp.h>
#include <numa.h>

#include "common/common.hpp"
#include "op.hpp"

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

  int nCpus = 0;
#ifdef OP_RD
  bind_cpu(dstDev);
  nCpus = num_cpus(dstDev);
#elif OP_WR
  bind_cpu(srcDev);
  nCpus = num_cpus(srcDev);
#else
#error "woah"
#endif

  assert(nCpus > 0);
  omp_set_num_threads(nCpus);

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
    //#pragma omp parallel
    {
      //#pragma omp barrier
      auto start = std::chrono::high_resolution_clock::now();
#ifdef OP_RD
      cpu_read_8(&dummy, ptr, count, stride);
#elif OP_WR
      cpu_write_8(&dummy, ptr, count, stride);
#else
#error "woah"
#endif

      //#pragma omp barrier
      //    if (omp_get_thread_num() == 0) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> txSeconds = end - start;
      times.push_back(txSeconds.count());
      //    }
    }
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  //const double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

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

  long long freeMem = cpu_free_memory(cpus);
  freeMem = std::min(freeMem, 8ll * 1024ll * 1024ll * 1024ll);

  auto counts = Sequence::geometric(2048, freeMem, 2) |
                Sequence::geometric(2048 * 1.5, freeMem, 2);

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
