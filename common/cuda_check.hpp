#ifndef CUDA_CHECK_HPP
#define CUDA_CHECK_HPP

#include <iostream>

#include <cuda_runtime_api.h>
#include <cuda.h>

#define RT_CHECK(ans)                    \
  {                                      \
    rtAssert((ans), __FILE__, __LINE__); \
  }
inline void rtAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    std::cerr << "CUDA_CHECK: " << cudaGetErrorString(code) << " " << file << " "
              << line << std::endl;
    if (abort)
      exit(code);
  }
}

#define DR_CHECK(ans)                    \
  {                                      \
    drAssert((ans), __FILE__, __LINE__); \
  }
inline void drAssert(CUresult code, const char *file, int line, bool abort = true)
{
  if (code != CUDA_SUCCESS)
  {
    const char *str;
    cuGetErrorString(code, &str);
    std::cerr << "DR_CHECK: " << str << " " << file << " "
              << line << std::endl;
    if (abort)
      exit(code);
  }
}

#endif
