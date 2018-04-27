#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include <nvToolsExt.h>

#include "common/cuda_check.hpp"
#include "common/common.hpp"

template <bool NOOP = false>
__global__ void gpu_traverse(size_t *ptr, const size_t steps)
{

  if (NOOP)
  {
    return;
  }
  size_t next = 0;
  for (int i = 0; i < steps; ++i)
  {
    next = ptr[next];
  }
  ptr[next] = 1;
}

template <bool NOOP = false>
void cpu_traverse(size_t *ptr, const size_t steps)
{

  if (NOOP)
  {
    return;
  }
  size_t next = 0;
  for (size_t i = 0; i < steps; ++i)
  {
    next = ptr[next];
  }
  ptr[next] = 1;
}

static void coherence_latency(const Device &dst, const Device &src, const size_t steps, const int numIters)
{

  assert(src.is_gpu() || dst.is_gpu());

  if (src.is_cpu())
  {
    bind_cpu(src);
  }
  else if (dst.is_cpu())
  {
    bind_cpu(dst);
  }

  // Determine grid dimensions
  dim3 blockDim(1);
  dim3 gridDim(1);
  const size_t stride = 65536ul * 2ul;
  const size_t count = sizeof(size_t) * (steps + 1) * stride;

  size_t *managedPtr, *explicitPtr;

  // explicit dst allocation
  if (dst.is_gpu())
  {
    RT_CHECK(cudaSetDevice(dst.cuda_device_id()));
    RT_CHECK(cudaMalloc(&explicitPtr, count));
  }
  else if (dst.is_cpu())
  {
    explicitPtr = new size_t[count / sizeof(size_t)];
  }

  // managed allocation
  RT_CHECK(cudaMallocManaged(&managedPtr, count));

  // set up stride
  for (size_t i = 0; i < steps; ++i)
  {
    managedPtr[i * stride] = (i + 1) * stride;
  }
  RT_CHECK(cudaMemcpy(explicitPtr, managedPtr, count, cudaMemcpyDefault));
  RT_CHECK(cudaDeviceSynchronize());

  std::vector<double> managedTimes, explicitTimes;
  for (int i = 0; i < numIters; ++i)
  {
    // Try to get allocation on source
    nvtxRangePush("prefetch to src");
    RT_CHECK(cudaMemPrefetchAsync(managedPtr, count, src.cuda_device_id()));
    if (src.is_gpu())
    {
      RT_CHECK(cudaSetDevice(src.cuda_device_id()));
      RT_CHECK(cudaDeviceSynchronize());
    }
    if (dst.is_gpu())
    {
      RT_CHECK(cudaSetDevice(dst.cuda_device_id()));
      RT_CHECK(cudaDeviceSynchronize());
    }
    nvtxRangePop();

    // Access from Device and Time
    nvtxRangePush("managed traverse");
    auto start = std::chrono::high_resolution_clock::now();
    if (dst.is_gpu())
    {
      gpu_traverse<<<gridDim, blockDim>>>(managedPtr, steps);
      RT_CHECK(cudaDeviceSynchronize());
    }
    else
    {
      cpu_traverse(managedPtr, steps);
    }
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    managedTimes.push_back((end - start).count() / 1e3);

    // Explicit traverse
    nvtxRangePush("explicit traverse");
    start = std::chrono::high_resolution_clock::now();
    if (dst.is_gpu())
    {
      gpu_traverse<<<gridDim, blockDim>>>(explicitPtr, steps);
      RT_CHECK(cudaDeviceSynchronize());
    }
    else
    {
      cpu_traverse(explicitPtr, steps);
    }
    end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    explicitTimes.push_back((end - start).count() / 1e3);
  }

  assert(managedTimes.size());
  const double minTime = *std::min_element(managedTimes.begin(), managedTimes.end());
  const double minExplicitTime = *std::min_element(explicitTimes.begin(), explicitTimes.end());
  const double avgTime = std::accumulate(managedTimes.begin(), managedTimes.end(), 0.0) / managedTimes.size();

  printf(",%.2f,%.2f", minTime, minExplicitTime);
  if (dst.is_gpu())
  {
    RT_CHECK(cudaFree(explicitPtr));
  }
  else
  {
    delete[] explicitPtr;
  }
  explicitPtr = nullptr;
  RT_CHECK(cudaFree(managedPtr));
}

int main(int argc, char **argv)
{
  int numIters = 10;
  std::vector<int> numaIds, gpuIds;
  option_as_int(argc, argv, "-n", numIters);
  option_as_int_list(argc, argv, "-c", numaIds);
  option_as_int_list(argc, argv, "-g", gpuIds);

  auto gpus = get_gpus(gpuIds);
  auto cpus = get_cpus(numaIds);

  std::vector<Device> devs;
  for (const auto &d : gpus)
  {
    devs.push_back(d);
  }
  for (const auto &d : cpus)
  {
    devs.push_back(d);
  }

  // print header
  printf("# Strides");

  for (const auto src : devs)
  {
    for (const auto dst : devs)
    {
      if (src != dst && (src.is_gpu() || dst.is_gpu()))
      {
        printf(",%s:%s Traversal Time (us) [managed], %s:%s Traversal Time (us) [explicit]", src.name().c_str(), dst.name().c_str(), src.name().c_str(), dst.name().c_str());
      }
    }
  }

  printf("\n");

  for (size_t numSteps = 4; numSteps < 34; ++numSteps)
  {
    printf("%lu", numSteps);
    for (const auto src : devs)
    {
      for (const auto dst : devs)
      {
        if (src != dst && (src.is_gpu() || dst.is_gpu()))
        {
          coherence_latency(dst, src, numSteps, numIters);
        }
      }
    }
    printf("\n");
  }

  return 0;
}
