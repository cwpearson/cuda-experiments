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

template <typename data_type, op_type op>
__global__ void stream_thread(data_type *ptr, const size_t size,
                              data_type *output, const data_type val)
{
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t n = size / sizeof(data_type);
  data_type accum = 0;

  for (; tid < n; tid += blockDim.x * gridDim.x)
    if (op == op_type::READ)
      accum += ptr[tid];
    else
      ptr[tid] = val;

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

  const int numSMs = prop.multiProcessorCount;
  const int threadsPerSM = prop.maxThreadsPerMultiProcessor;
  const int major = prop.major;
  int blocksPerSM;
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
        stream_thread<data_type, op_type::READ><<<dimGrid, dimBlock>>>(ptr, n, output, 1.0);
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
