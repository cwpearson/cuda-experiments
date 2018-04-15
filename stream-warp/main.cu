#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <chrono>

#include <nvToolsExt.h>

#include <unistd.h>

#include "common/cuda_check.hpp"
#include "common/common.hpp"

enum class op_type
{
  READ,
  WRITE
};

const size_t STRIDE_64K = 65536;
const size_t STRIDE_4K = 4096;

template <typename data_type, op_type op, size_t STRIDE>
__global__ void stream_warp(data_type *ptr, const size_t size, data_type *output, const data_type val)
{
  int lane_id = threadIdx.x & 31;
  size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
  int warps_per_grid = (blockDim.x * gridDim.x) >> 5;
  size_t warp_total = (size + STRIDE - 1) / STRIDE;

  size_t n = size / sizeof(data_type);
  data_type accum = 0;

  for (; warp_id < warp_total; warp_id += warps_per_grid)
  {
#pragma unroll
    for (int rep = 0; rep < STRIDE / sizeof(data_type) / 32; rep++)
    {
      size_t ind = warp_id * STRIDE / sizeof(data_type) + rep * 32 + lane_id;
      if (ind < n)
      {
        if (op == op_type::READ)
          accum += ptr[ind];
        else
          ptr[ind] = val;
      }
    }
  }

  if (op == op_type::READ)
    output[threadIdx.x + blockIdx.x * blockDim.x] = accum;
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);
  std::stringstream buffer;

  RT_CHECK(cudaFree(0));

  size_t memTotal, memAvail;
  RT_CHECK(cudaSetDevice(0));
  RT_CHECK(cudaMemGetInfo(&memAvail, &memTotal));

  cudaDeviceProp prop;
  RT_CHECK(cudaGetDeviceProperties(&prop, 0));

  const size_t numSMs = prop.multiProcessorCount;
  const size_t threadsPerSM = prop.maxThreadsPerMultiProcessor;
  const size_t major = prop.major;
  size_t blocksPerSM;
  if (major < 3)
  {
    blocksPerSM = 8;
  }
  else if (major < 6)
  {
    blocksPerSM = 16;
  }
  else
  {
    blocksPerSM = 32;
  }

  std::cout << prop.name << ":\n";
  std::cout << "\t" << numSMs << " SMs\n";
  std::cout << "\t" << threadsPerSM << " threads/SM\n";
  std::cout << "\t" << blocksPerSM << " blocks/SM\n";

  typedef float data_type;
  data_type *output, *ptr;

  for (size_t n = 128; n < memAvail; n *= 2)
  {
    dim3 dimGrid;
    dim3 dimBlock;

    for (dimGrid.x = numSMs; dimGrid.x <= blocksPerSM * numSMs; dimGrid.x = dimGrid.x * 2)
    {
      for (dimBlock.x = 32; dimBlock.x <= 1024; dimBlock.x *= 2)
      {
        RT_CHECK(cudaMallocManaged(&ptr, n));
        RT_CHECK(cudaMalloc(&output, dimGrid.x * dimBlock.x * sizeof(data_type)));
        RT_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        if (pageSize == STRIDE_64K)
        {
          stream_warp<data_type, op_type::READ, STRIDE_64K><<<dimGrid, dimBlock>>>(ptr, n, output, 1.0);
        }
        else if (pageSize == STRIDE_4K)
        {
          stream_warp<data_type, op_type::READ, STRIDE_4K><<<dimGrid, dimBlock>>>(ptr, n, output, 1.0);
        }
        else
        {
          assert(0);
        }
        RT_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "n=" << n << " (" << dimGrid.x << "x" << dimBlock.x << ") " << elapsed_seconds.count() << "s " << n / 1e9 / elapsed_seconds.count() << "GB/s\n";

        RT_CHECK(cudaFree(ptr));
        RT_CHECK(cudaFree(output));
      }
    }
  }

  return 0;
}
