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

size_t cpu_touch(char *c, const size_t e, const size_t n)
{
  for (size_t i = 0; i < n; i += e)
  {
    c[i] = i * 31ul + 7ul;
  }
  return (n / e);
}

__global__ void gpu_touch(char *c, const size_t e, const size_t n)
{
  // const size_t bx = blockIdx.x;
  // const size_t tx = threadIdx.x;
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = gx; i < n; i += e)
  {
    c[i] = i * 31ul + 7ul;
  }
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);
  std::stringstream buffer;

  RT_CHECK(cudaFree(0));

  size_t memTotal, memAvail;
  RT_CHECK(cudaMemGetInfo(&memAvail, &memTotal));

  char *cm;

  RT_CHECK(cudaMallocManaged(&cm, pageSize * 32));
  RT_CHECK(cudaDeviceSynchronize());

  dim3 dimGrid(1);
  dim3 dimBlock(1);

  for (size_t i = 0; i < 20; ++i)
  {
    nvtxRangePush("iter");
    nvtxRangePush("cpu");
    cpu_touch(cm, pageSize, 1);
    RT_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();
    nvtxRangePush("gpu");
    gpu_touch<<<dimGrid, dimBlock>>>(cm, pageSize, 1);
    RT_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();
    nvtxRangePop();
  }

  return 0;
}
