#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <vector>
#include <tuple>

#include <numa.h>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/cuda_check.hpp"
#include "common/common.hpp"

typedef std::tuple<Device, Device> DevicePair;

static void pinned_bw(std::vector<DevicePair> pairs, const size_t count)
{

  // Set up allocations
  std::vector<void *> srcPtrs, dstPtrs;
  std::vector<void *> cudaFrees, cudaFreeHosts;
  for (const auto &p : pairs)
  {
    const auto &srcDev = std::get<0>(p);
    const auto &dstDev = std::get<1>(p);

    void *srcPtr = nullptr;
    if (srcDev.is_cpu())
    {
      bind_cpu(srcDev);
      RT_CHECK(cudaMallocHost(&srcPtr, count));
      cudaFreeHosts.push_back(srcPtr);
    }
    else
    {
      RT_CHECK(cudaSetDevice(srcDev.cuda_device_id()));
      RT_CHECK(cudaMalloc(&srcPtr, count));
      cudaFrees.push_back(srcPtr);
    }
    srcPtrs.push_back(srcPtr);

    void *dstPtr = nullptr;
    if (dstDev.is_cpu())
    {
      bind_cpu(dstDev);
      RT_CHECK(cudaMallocHost(&dstPtr, count));
      cudaFreeHosts.push_back(dstPtr);
    }
    else
    {
      RT_CHECK(cudaSetDevice(dstDev.cuda_device_id()));
      RT_CHECK(cudaMalloc(&dstPtr, count));
      cudaFrees.push_back(dstPtr);
    }
    srcPtrs.push_back(dstPtr);
  }

  // set up streams
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> startEvents;
  std::vector<cudaEvent_t> endEvents;
  for (auto &p : pairs)
  {
    cudaStream_t stream;
    RT_CHECK(cudaStreamCreate(&stream));
    streams.push_back(stream);

    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    RT_CHECK(cudaEventCreate(&startEvent));
    RT_CHECK(cudaEventCreate(&endEvent));
    startEvents.push_back(startEvent);
    endEvents.push_back(endEvent);
  }

  // Engage memcpys
  for (size_t i = 0; i < streams.size(); ++i)
  {
    void *dst = dstPtrs[i];
    void *src = srcPtrs[i];
    cudaStream_t stream = streams[i];
    auto startEvent = startEvents[i];
    auto endEvent = endEvents[i];
    RT_CHECK(cudaEventRecord(startEvent, stream));
    RT_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
    RT_CHECK(cudaEventRecord(endEvent, stream));
  }

  // Get times
  for (size_t i = 0; i < streams.size(); ++i)
  {
    //Get runtime of my_kernel in ms
    float runtime = 0.0f;
    auto startEvent = startEvents[i];
    auto endEvent = endEvents[i];
    auto stream = streams[i];
    RT_CHECK(cudaEventSynchronize(endEvent));
    RT_CHECK(cudaEventElapsedTime(&runtime, startEvent, endEvent));
  }

  // free resources
  for (auto p : cudaFrees)
  {
    RT_CHECK(cudaFree(p));
  }
  for (auto p : cudaFreeHosts)
  {
    RT_CHECK(cudaFreeHost(p));
  }
  for (auto e : startEvents)
  {
    RT_CHECK(cudaEventDestroy(e));
  }
  for (auto e : endEvents)
  {
    RT_CHECK(cudaEventDestroy(e));
  }
  for (auto s : streams)
  {
    RT_CHECK(cudaStreamDestroy(s));
  }
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
      printf(",%s to %s (pinned)", cpu.name().c_str(), gpu.name().c_str());
    }
  }
  //gpu->cpu
  for (const auto cpu : cpus)
  {
    for (const auto gpu : gpus)
    {
      printf(",%s to %s (pinned)", gpu.name().c_str(), cpu.name().c_str());
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
        // pinned_bw(gpu, cpu, count, numIters);
      }
    }
    //gpu->cpu
    for (const auto cpu : cpus)
    {
      for (const auto gpu : gpus)
      {
        // pinned_bw(cpu, gpu, count, numIters);
      }
    }

    printf("\n");
  }

  return 0;
}
