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
__global__ void gpu_touch(data_type *ptr, const bool noop = false)
{
  if (noop)
  {
    return;
  }

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  if (gx == 0)
  {
    ptr[0] = 0;
  }
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);
  std::stringstream buffer;

  RT_CHECK(cudaFree(0));

  size_t memTotal, memAvail;
  RT_CHECK(cudaMemGetInfo(&memAvail, &memTotal));

  typedef int data_type;
  data_type *ptr;
  const size_t srcCount = 10;
  const size_t dstCount = 1;
  const dim3 dimGrid(1);
  const dim3 dimBlock(1);

  RT_CHECK(cudaMallocManaged(&ptr, pageSize));

  nvtxRangePush("src");
  for (size_t srcI = 0; srcI < srcCount; ++srcI)
  {
    cpu_touch(ptr);
  }
  nvtxRangePop();

  nvtxRangePush("dst");
  for (size_t dstI = 0; dstI < dstCount; ++dstI)
  {
    gpu_touch<<<dimGrid, dimBlock>>>(ptr);
  }
  RT_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  RT_CHECK(cudaFree(ptr));

  return 0;
}
