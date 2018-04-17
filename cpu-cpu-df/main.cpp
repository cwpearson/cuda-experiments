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

static void prefetch_bw(const Device &dstDev, const Device &srcDev, const size_t count, const size_t stride, const size_t nCpus)
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

#ifdef OP_RD
  bind_cpu(dstDev);
#elif OP_WR
  bind_cpu(srcDev);
#else
#error "woah"
#endif

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
      printf("%lu,%s,%s,%lu,%.2f\n", count, srcDev.name().c_str(), dstDev.name().c_str(), nCpus, count / txSeconds.count());
      //    }
    }
  }

  free(ptr);
}

int main(void)
{
  numa_set_strict(1);
  numa_exit_on_error = 1;
  auto cpus = get_cpus();

  size_t numCpusPerNode = min_cpus_per_node(cpus);
  numCpusPerNode = std::min(size_t(16), numCpusPerNode);

  // Print column headers
  printf("transfer_size,src,dst,threads,bandwidth\n");

  long long freeMem = cpu_free_memory(cpus);
  freeMem = std::min(freeMem, 8ll * 1024ll * 1024ll * 1024ll);

  auto counts = Sequence::geometric(2048, freeMem, 2) |
                Sequence::geometric(2048 * 1.5, freeMem, 2);

  for (auto count : counts)
  {
    for (size_t par = 1; par <= numCpusPerNode; par *= 2)
    {
      for (const auto src : cpus)
      {
        for (const auto dst : cpus)
        {
          if (src != dst)
          {
            prefetch_bw(dst, src, count, 8, par);
          }
        }
      }
    }
  }

  return 0;
}
