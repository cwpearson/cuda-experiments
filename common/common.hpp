#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <vector>
#include <cstdint>
#include <set>
#include <cassert>
#include <algorithm>

#include <numa.h>
#include <errno.h>

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

class Device
{
public:
  Device();
  Device(const bool cpu, const int id);

  std::string name() const;
  bool is_cpu() const;
  bool is_gpu() const;
  int cuda_device_id() const;
  int id() const;

  bool operator==(const Device &other) const;
  bool operator!=(const Device &other) const;

private:
  bool cpu_;
  int id_;
};

std::vector<Device> get_gpus();
std::vector<Device> get_cpus();
void bind_cpu(const Device &d);
size_t num_mps(const Device &d);
size_t max_threads_per_mp(const Device &d);
size_t max_blocks_per_mp(const Device &d);

class Sequence
{
public:
  typedef int64_t value_type;
  typedef std::vector<value_type> container_type;

private:
  container_type seq_;

public:
  static Sequence geometric(value_type min, value_type max, double step);
  static Sequence neighborhood(const Sequence &orig, const double window_scale, const size_t window_elems);

  Sequence &operator|=(const Sequence &rhs);
  Sequence operator|(const Sequence &rhs) const;

  container_type::const_iterator begin() const;
  container_type::const_iterator end() const;
};

#endif
