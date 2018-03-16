#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <chrono>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/common.hpp"

template <typename data_type>
void cpu_touch(volatile data_type *ptr)
{
  ptr[0] = 0;
}

template <typename data_type>
__global__ void gpu_touch(data_type *ptr, const size_t pageSize, const size_t startCount, const size_t endCount, const bool noop = false)
{
  if (noop)
  {
    return;
  }

  const size_t pageSizeElems = pageSize / sizeof(data_type);

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < endCount - startCount; i += blockDim.x * gridDim.x)
  {
    const size_t pageOffset = pageSizeElems * i;
    const size_t accessCount = startCount + i;
    for (size_t access = 0; access < accessCount; ++access) {
      ptr[pageOffset] += access;
    }
  }
}

template <typename data_type>
__global__ void sm_touch(data_type *ptr, const size_t footprint, const size_t numTouch, const bool noop = false)
{
  if (noop)
  {
    return;
  }

  const size_t numElems = footprint / sizeof(data_type);

  const size_t tx = threadIdx.x;
  for (size_t i = tx; i < numElems; i += blockDim.x)
  {
    for (size_t c = 0; c < numTouch; ++c) {
      ptr[i] += c * 31 + 7;
    }
  }
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);
  std::stringstream buffer;

  typedef int data_type;
  data_type *ptr;
  const int srcDev = 0;
  const int dstDev = 1;
  RT_CHECK(cudaSetDevice(srcDev));
  RT_CHECK(cudaFree(0));
  RT_CHECK(cudaSetDevice(dstDev));
  RT_CHECK(cudaFree(0));

  RT_CHECK(cudaSetDevice(srcDev));
  size_t memTotal, memAvail;
  RT_CHECK(cudaMemGetInfo(&memAvail, &memTotal));
  RT_CHECK(cudaMallocManaged(&ptr, pageSize));
  //RT_CHECK(cudaMemAdvise(ptr, pageSize, cudaMemAdviseSetPreferredLocation, 0));

  nvtxRangePush("src");
  RT_CHECK(cudaSetDevice(srcDev));
    sm_touch<<<84000, 256>>>(ptr, pageSize, 100);
  RT_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  nvtxRangePush("dst");
  RT_CHECK(cudaSetDevice(dstDev));
    sm_touch<<<1, 256>>>(ptr, pageSize, 1);
    // gpu_touch<<<dimGrid, dimBlock>>>(ptr, pageSize, 1, 2);
    // gpu_touch<<<dimGrid, dimBlock>>>(&ptr[pageSize / sizeof(data_type) * 400], pageSize, 1, 2);
    // gpu_touch<<<dimGrid, dimBlock>>>(&ptr[pageSize / sizeof(data_type) * 800], pageSize, 1, 2);
  RT_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  for (int i = 0; i < 100; ++i) {
  nvtxRangePush("src");
  RT_CHECK(cudaSetDevice(srcDev));
  sm_touch<<<84000, 256>>>(ptr, pageSize, 100);
  RT_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  nvtxRangePush("dst");
  RT_CHECK(cudaSetDevice(dstDev));
  sm_touch<<<1, 256>>>(ptr, pageSize, 1);
  RT_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
  }


  RT_CHECK(cudaFree(ptr));

  return 0;
}
