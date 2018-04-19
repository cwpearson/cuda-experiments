#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <vector>

#include <nvToolsExt.h>

#include "common/common.hpp"
#include "common/cuda_check.hpp"

static void prefetch_bw(const Device &dstDev, const Device &srcDev,
                        const size_t count, const int numIters) {

  // If srd is a CPU, make sure ptr is allocted on that cpu
  if (srcDev.is_cpu()) {
    bind_cpu(srcDev);
  }

  void *ptr;
  RT_CHECK(cudaMallocManaged(&ptr, count));
  std::memset(ptr, 0, count);

  std::vector<double> times;
  for (size_t i = 0; i < numIters; ++i) {
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
  const double avgTime =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  printf(",%.2f", count / 1024.0 / 1024.0 / (minTime));

  RT_CHECK(cudaFree(ptr));
}

int main(int argc, char **argv) {
  int numIters = 10;
  std::vector<int> numaIds, gpuIds;
  option_as_int(argc, argv, "-n", numIters);
  option_as_int_list(argc, argv, "-c", numaIds);
  option_as_int_list(argc, argv, "-g", gpuIds);

  auto gpus = get_gpus(gpuIds);
  auto cpus = get_cpus(numaIds);
  auto devs = cpus;
  for (const auto g : gpus) {
    devs.push_back(g);
  }

  size_t freeMem = free_memory(devs);

  // print header
  printf("Transfer Size (MB),");
  for (const auto src : devs) {
    for (const auto dst : devs) {
      if (src != dst && (src.is_gpu() || dst.is_gpu())) {
        printf("%s to %s (prefetch),", src.name().c_str(), dst.name().c_str());
      }
    }
  }
  printf("\n");

  auto counts = Sequence::geometric(2048, freeMem, 2) |
                Sequence::geometric(2048 * 1.5, freeMem, 2);

  for (auto count : counts) {
    printf("%f", count / 1024.0 / 1024.0);
    for (const auto src : devs) {
      for (const auto dst : devs) {
        if (src != dst && (src.is_gpu() || dst.is_gpu())) {
          prefetch_bw(dst, src, count, numIters);
        }
      }
    }
    printf("\n");
  }

  return 0;
}
