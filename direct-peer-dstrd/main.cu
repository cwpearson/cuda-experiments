#include <algorithm>
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

#include "common/common.hpp"

template <typename data_type>
__global__ void gpu_read(data_type *ptr, const size_t stride, const size_t count)
{
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t wx = gx >> 5;            // warp id
  const size_t lx = threadIdx.x & 0x1F; // lane id
  const size_t warpsInGrid = gridDim.x * blockDim.x / 32;

  const size_t dataInStride = stride / sizeof(data_type);
  const size_t dataInCount = count / sizeof(data_type);

  data_type acc = 0;

  for (size_t i = wx * dataInStride; i < dataInCount; i += warpsInGrid * dataInStride)
  {
    for (size_t strideOff = lx; strideOff < dataInStride && (i + strideOff < dataInCount); strideOff += 32)
    {
      acc += ptr[i + strideOff];
    }
  }
  if (wx * dataInStride < dataInCount)
  {
    ptr[wx * dataInStride] = acc;
  }
}

static void gpu_gpu_bw(const Device &dst, const Device &src, const size_t count)
{

  assert(src.is_gpu() && dst.is_gpu());

  int *srcPtr;

  RT_CHECK(cudaSetDevice(src.id()));
  RT_CHECK(cudaMalloc(&srcPtr, count));
  {
    cudaError_t err = cudaDeviceEnablePeerAccess(dst.id(), 0);
    if (err != cudaErrorPeerAccessAlreadyEnabled)
    {
      RT_CHECK(err);
    }
  }
  RT_CHECK(cudaSetDevice(dst.id()));
  {
    cudaError_t err = cudaDeviceEnablePeerAccess(src.id(), 0);
    if (err != cudaErrorPeerAccessAlreadyEnabled)
    {
      RT_CHECK(err);
    }
  }

  // fill up GPU with blocks
  const size_t numMps = num_mps(dst);
  const size_t maxBlocksPerMp = max_blocks_per_mp(dst);
  const size_t maxThreadsPerMp = max_threads_per_mp(dst);
  dim3 gridDim(numMps * maxBlocksPerMp);
  dim3 blockDim(maxThreadsPerMp / maxBlocksPerMp);

  const long pageSize = sysconf(_SC_PAGESIZE);

  std::vector<double> times;
  const size_t numIters = 20;
  for (size_t i = 0; i < numIters; ++i)
  {
    RT_CHECK(cudaSetDevice(dst.id()));
    RT_CHECK(cudaDeviceSynchronize());
    nvtxRangePush("dst");
    auto start = std::chrono::high_resolution_clock::now();
    gpu_read<<<gridDim, blockDim>>>(srcPtr, pageSize, count);
    RT_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> txSeconds = end - start;
    nvtxRangePop();
    times.push_back(txSeconds.count());
  }

  const double minTime = *std::min_element(times.begin(), times.end());
  const double avgTime =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  printf(",%.2f", count / 1024.0 / 1024.0 / minTime);
  RT_CHECK(cudaFree(srcPtr));
}

int main(void)
{

  const size_t numNodes = numa_max_node();

  std::vector<Device> gpus = get_gpus();

  // print header
  printf("Transfer Size (MB)");
  for (const auto dst : gpus)
  {
    for (const auto src : gpus)
    {
      if (src != dst)
      {
        int can;
        RT_CHECK(cudaDeviceCanAccessPeer(&can, src.id(), dst.id()));
        if (can)
        {
          printf(",%s:%s", src.name().c_str(), dst.name().c_str());
        }
      }
    }
  }

  printf("\n");

  auto freeMem = gpu_free_memory(gpus);
  auto counts = Sequence::geometric(2048, freeMem, 2) |
                Sequence::geometric(2048 * 1.5, freeMem, 2);

  for (auto count : counts)
  {
    printf("%f", count / 1024.0 / 1024.0);
    for (const auto dst : gpus)
    {
      for (const auto src : gpus)
      {

        if (src != dst)
        {
          int can;
          RT_CHECK(cudaDeviceCanAccessPeer(&can, src.id(), dst.id()));
          if (can)
          {

            gpu_gpu_bw(dst, src, count);
          }
        }
      }
    }

    printf("\n");
  }

  return 0;
}
