#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <vector>

#include <numa.h>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/cuda_check.hpp"
#include "common/common.hpp"

#include <vector>
#include <cstring>

#define RT_RET(ans)         \
  {                         \
    err = (ans);            \
    if (err != cudaSuccess) \
    {                       \
      goto cleanup;         \
    }                       \
  }

cudaError_t micro_cuda_memcpy_async(
    float *millis,              // milliseconds to complete all transfers
    char *const *const srcPtrs, // source pointers, one per transfer
    char **dstPtrs,             // destination pointers, one per transfer
    const size_t count,         // transfer size
    const size_t n              // number of transfers
)
{
  cudaError_t err = cudaSuccess;

  // Set up stream and event for each transfer
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> startEvents;
  std::vector<cudaEvent_t> endEvents;
  for (size_t i = 0; i < n; ++i)
  {
    cudaStream_t stream;
    RT_RET(cudaStreamCreate(&stream));
    streams.push_back(stream);

    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    RT_RET(cudaEventCreate(&startEvent));
    RT_RET(cudaEventCreate(&endEvent));
    startEvents.push_back(startEvent);
    endEvents.push_back(endEvent);
  }

  // Engage memcpys
  for (size_t i = 0; i < n; ++i)
  {
    auto dst = dstPtrs[i];
    auto src = srcPtrs[i];
    cudaStream_t stream = streams[i];
    auto startEvent = startEvents[i];
    auto endEvent = endEvents[i];
    RT_RET(cudaEventRecord(startEvent, stream));
    RT_RET(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
    RT_RET(cudaEventRecord(endEvent, stream));
  }

  // Wait for memcpys to finish
  for (size_t i = 0; i < n; ++i)
  {
    auto endEvent = endEvents[i];
    RT_RET(cudaEventSynchronize(endEvent));
  }

  // Get the longest time between start and end events
  {
    float maxRuntime = 0.0;
    for (size_t startIdx = 0; startIdx < n; ++startIdx)
    {
      for (size_t endIdx = 0; endIdx < n; ++endIdx)
      {
        float runtime = 0.0f;
        auto startEvent = startEvents[startIdx];
        auto endEvent = endEvents[endIdx];
        RT_RET(cudaEventSynchronize(endEvent));
        RT_RET(cudaEventElapsedTime(&runtime, startEvent, endEvent));
        if (runtime > maxRuntime)
        {
          maxRuntime = runtime;
        }
      }
    }
    *millis = maxRuntime;
  }

// free resources
cleanup:
  for (auto e : startEvents)
  {
    cudaEventDestroy(e);
  }
  for (auto e : endEvents)
  {
    cudaEventDestroy(e);
  }
  for (auto s : streams)
  {
    cudaStreamDestroy(s);
  }
  return err;
}

cudaError_t micro_pageable(float *millis,
                           int src_is_host,
                           int dst_is_host,
                           const int gpu,
                           const size_t count,
                           const int full_duplex)
{
  cudaError_t err = cudaSuccess;

  // Set up allocations
  std::vector<char *> srcPtrs, dstPtrs;
  std::vector<char *> cudaFrees, frees;

  char *srcPtr = nullptr;
  char *dstPtr = nullptr;

  // first allocation (src to dst)
  if (src_is_host)
  {
    srcPtr = (char *)malloc(count);
    std::memset(srcPtr, 0, count);
    frees.push_back(srcPtr);
  }
  else
  {
    RT_RET(cudaSetDevice(gpu));
    RT_RET(cudaMalloc(&srcPtr, count));
    cudaFrees.push_back(srcPtr);
  }
  if (dst_is_host)
  {
    dstPtr = (char *)malloc(count);
    std::memset(srcPtr, 0, count);
    frees.push_back(dstPtr);
  }
  else
  {
    RT_RET(cudaSetDevice(gpu));
    RT_RET(cudaMalloc(&dstPtr, count));
    cudaFrees.push_back(dstPtr);
  }
  srcPtrs.push_back(srcPtr);
  dstPtrs.push_back(dstPtr);

  // full-duplex allocation (dst to src)
  if (full_duplex)
  {
    if (src_is_host)
    {
      srcPtr = (char *)malloc(count);
      std::memset(srcPtr, 0, count);
      frees.push_back(srcPtr);
    }
    else
    {
      RT_RET(cudaSetDevice(gpu));
      RT_RET(cudaMalloc(&srcPtr, count));
      cudaFrees.push_back(srcPtr);
    }
    if (dst_is_host)
    {
      dstPtr = (char *)malloc(count);
      std::memset(srcPtr, 0, count);
      frees.push_back(dstPtr);
    }
    else
    {
      RT_RET(cudaSetDevice(gpu));
      RT_RET(cudaMalloc(&dstPtr, count));
      cudaFrees.push_back(dstPtr);
    }
    srcPtrs.push_back(dstPtr);
    dstPtrs.push_back(srcPtr);
  }

  // run the benchmark
  RT_RET(micro_cuda_memcpy_async(
      millis,
      srcPtrs.data(),
      dstPtrs.data(),
      count,
      srcPtrs.size()));

// free resources
cleanup:
  for (auto p : cudaFrees)
  {
    cudaFree(p);
  }
  for (auto p : frees)
  {
    free(p);
  }
  return err;
}

static void pinned_bw(const Device &dst, const Device &src, const size_t count, const int numIters)
{

  assert((src.is_cpu()) ^ (dst.is_cpu()));

  void *devPtr, *hostPtr;
  void *srcPtr, *dstPtr;

  if (src.is_gpu())
  {
    RT_CHECK(cudaSetDevice(src.id()));
    bind_cpu(dst);
  }
  else
  {
    RT_CHECK(cudaSetDevice(dst.id()));
    bind_cpu(src);
  }

  RT_CHECK(cudaFree(0));
  RT_CHECK(cudaMalloc(&devPtr, count))
  RT_CHECK(cudaMallocHost(&hostPtr, count));
  std::memset(hostPtr, 0, count);

  if (src.is_gpu())
  {
    srcPtr = devPtr;
    dstPtr = hostPtr;
  }
  else
  {
    srcPtr = hostPtr;
    dstPtr = devPtr;
  }

  for (int i = 0; i < numIters; ++i)
  {
    nvtxRangePush("dst");
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaMemcpy(dstPtr, srcPtr, count, cudaMemcpyDefault));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    nvtxRangePop();
    printf("%s,%s,%lu,%.5f\n", src.name().c_str(), dst.name().c_str(), count, txSeconds.count());
  }

  RT_CHECK(cudaFreeHost(hostPtr));
  RT_CHECK(cudaFree(devPtr));
}

int main(int argc, char **argv)
{
  int numIters = 10;
  std::vector<int> gpuIds;
  std::vector<int> numaIds;
  bool src_is_cpu = false;
  bool dst_is_cpu = false;
  int src_id = -1;
  int dst_id = -1;

  if (option_as_int(argc, argv, "-n", numIters))
  {
    fprintf(stderr, "Using %d iterations\n", numIters);
  }
  if (option_as_int(argc, argv, "--src-numa", src_id))
  {
    fprintf(stderr, "Using src CPU %d\n", src_id);
    src_is_cpu = true;
  }
  if (option_as_int(argc, argv, "--src-gpu", src_id))
  {
    fprintf(stderr, "Using src GPU\n\n");
  }
  if (option_as_int(argc, argv, "--dst-numa", dst_id))
  {
    fprintf(stderr, "Using dst CPU\n");
    dst_is_cpu = true;
  }
  if (option_as_int(argc, argv, "--dst-gpu", dst_id))
  {
    fprintf(stderr, "Using dst GPU\n");
  }

  auto srcIds = std::vector<int>(1, src_id);
  auto dstIds = std::vector<int>(1, dst_id);
  auto srcs = src_is_cpu ? get_cpus(srcIds) : get_gpus(srcIds);
  auto dsts = dst_is_cpu ? get_cpus(dstIds) : get_gpus(dstIds);

  if (srcs.empty())
  {
    fprintf(stderr, "no srcs\n");
    return 1;
  }

  if (dsts.empty())
  {
    fprintf(stderr, "no dsts\n");
    return 1;
  }
  auto src = srcs[0];
  auto dst = dsts[0];
  assert(src.is_cpu() ^ dst.is_cpu());
  auto gpus = src.is_cpu() ? dsts : srcs;

  auto freeMem = gpu_free_memory(gpus);
  auto counts = Sequence::geometric(256 * 1024, freeMem, 2) |
                Sequence::geometric(256 * 1024 * 1.5, freeMem, 2);

  for (auto count : counts)
  {
    pinned_bw(dst, src, count, numIters);
  }

  return 0;
}
