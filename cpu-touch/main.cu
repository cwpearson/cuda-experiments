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

size_t touch(char *c, const size_t e, const size_t n)
{
  for (size_t i = 0; i < n; i += e)
  {
    c[i] = i * 31ul + 7ul;
  }
  return (n / e);
}

int main(void)
{

  const long pageSize = sysconf(_SC_PAGESIZE);
  std::stringstream buffer;

  RT_CHECK(cudaFree(0));

  size_t memTotal, memAvail;
  RT_CHECK(cudaMemGetInfo(&memAvail, &memTotal));

  char *cm;

  for (size_t n = 4; n < memAvail; n = ceil(n * 1.5))
  {
    buffer.str("");
    buffer << "n " << n;
    nvtxRangePush(buffer.str().c_str());
    for (size_t e = 1; e < n && e < pageSize * 32; e *= 2)
    {
      buffer.str("");
      buffer << "e " << e;
      nvtxRangePush(buffer.str().c_str());

      // Allocate memory
      buffer.str("");
      buffer << "cudaMallocManaged " << n;
      nvtxRangePush(buffer.str().c_str());
      RT_CHECK(cudaMallocManaged(&cm, n));
      RT_CHECK(cudaMemPrefetchAsync(cm, n, 0));
      RT_CHECK(cudaDeviceSynchronize());
      nvtxRangePop();

      {
        buffer.str("");
        buffer << "touch(" << e << ", " << n << ")";
        nvtxRangePush(buffer.str().c_str());
        auto start = std::chrono::high_resolution_clock::now();
        const size_t numTouch = touch(cm, e, n);
        auto end = std::chrono::high_resolution_clock::now();
        nvtxRangePop();
        RT_CHECK(cudaDeviceSynchronize());
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "n=" << n << " e=" << e << ": " << numTouch / elapsed_seconds.count() << " " << e << "writes/s (" << numTouch << " total)\n";
      }

      // Free Memory
      nvtxRangePush("free");
      RT_CHECK(cudaFree(cm));
      nvtxRangePop();
      nvtxRangePop();
    }
    nvtxRangePop();
  }

  /*
  {
    buffer.clear();
    buffer << "touch " << N;
    nvtxRangePush(buffer.str().c_str());
    auto start = std::chrono::high_resolution_clock::now();
    const size_t numTouch = touch(fm, pageSize, N);
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    RT_CHECK(cudaDeviceSynchronize());
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "bw (1): " << numTouch / elapsed_seconds.count() << " " << pageSize << "B-elems/s\n";
  }

  {
    buffer.clear();
    buffer << "touch " << N;
    nvtxRangePush(buffer.str().c_str());
    auto start = std::chrono::high_resolution_clock::now();
    const size_t numTouch = touch(fm, pageSize, N);
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    RT_CHECK(cudaDeviceSynchronize());
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "bw (2): " << numTouch / elapsed_seconds.count() << " " << pageSize << "B-elems/s\n";
  }

  {
    buffer.clear();
    buffer << "touch " << N;
    nvtxRangePush(buffer.str().c_str());
    auto start = std::chrono::high_resolution_clock::now();
    const size_t numTouch = touch(fm, pageSize, N);
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    RT_CHECK(cudaDeviceSynchronize());
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "bw (3): " << numTouch / elapsed_seconds.count() << " " << pageSize << "B-elems/s\n";
  }
  */

  return 0;
}
