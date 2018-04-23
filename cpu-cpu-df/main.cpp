#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <vector>

#include <numa.h>
#include <omp.h>
#include <unistd.h>

#include "common/common.hpp"
#include "op.hpp"

static void omp_bind(const Device &d) {
  bind_cpu(d);
#pragma omp parallel
  { bind_cpu(d); }
}

#ifdef OP_WR
static void wr_bw(const Device &dstDev, const Device &srcDev,
                  const size_t count, const size_t stride, const size_t nCpus) {

  assert(srcDev.is_cpu() && dstDev.is_cpu());
  omp_set_num_threads(nCpus);
  const long pageSize = sysconf(_SC_PAGESIZE);

  // create source allocation
  bind_cpu(dstDev);
  double *ptr = static_cast<double *>(aligned_alloc(pageSize, count));
  std::memset(ptr, 0, count);

  omp_bind(srcDev);

  double dummy;
  const size_t numIters = 15;
  for (size_t i = 0; i < numIters; ++i) {

    // invalidate allocation in src caches
    omp_bind(dstDev);
    std::memset(ptr, 0, count);

    // Access from Device and Time
    omp_bind(srcDev);

    auto start = std::chrono::high_resolution_clock::now();
    cpu_write_8(&dummy, ptr, count, stride);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    printf("%lu,%s,%s,%lu,%.2f\n", count, srcDev.name().c_str(),
           dstDev.name().c_str(), nCpus, count / txSeconds.count());
  }

  free(ptr);
}
#endif

#ifdef OP_RD
static void rd_bw(const Device &dstDev, const Device &srcDev,
                  const size_t count, const size_t stride, const size_t nCpus) {

  assert(srcDev.is_cpu() && dstDev.is_cpu());
  const long pageSize = sysconf(_SC_PAGESIZE);
  omp_set_num_threads(nCpus);

  // create source allocation
  omp_bind(srcDev);
  double *ptr = static_cast<double *>(aligned_alloc(pageSize, count));
  std::memset(ptr, 0, count);

  omp_bind(dstDev);

  double dummy;
  const size_t numIters = 15;
  for (size_t i = 0; i < numIters; ++i) {

    // invalidate data in dst cache
    omp_bind(srcDev);
    std::memset(ptr, 0, count);

    // Access from Device and Time
    omp_bind(dstDev);

    auto start = std::chrono::high_resolution_clock::now();

    cpu_read_8(&dummy, ptr, count, stride);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    printf("%lu,%s,%s,%lu,%.2f\n", count, srcDev.name().c_str(),
           dstDev.name().c_str(), nCpus, count / txSeconds.count());
  }
  free(ptr);
}
#endif

int main(void) {
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

  for (auto count : counts) {
    for (size_t par = 1; par <= numCpusPerNode; par *= 2) {
      for (const auto src : cpus) {
        for (const auto dst : cpus) {
          if (src != dst) {
#ifdef OP_RD
            rd_bw(dst, src, count, 8, par);
#elif OP_WR
            wr_bw(dst, src, count, 8, par);
#else
#error "woah"
#endif
          }
        }
      }
    }
  }

  return 0;
}
