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
  Device() {}
  Device(const bool cpu, const int id) : cpu_(cpu), id_(id) {}

  std::string name() const
  {
    std::string s;
    if (is_cpu())
    {
      s = "cpu";
    }
    else
    {
      s = "gpu";
    }

    return s + std::to_string(id_);
  }
  bool is_cpu() const { return cpu_; }
  bool is_gpu() const { return !cpu_; }
  int cuda_device_id() { return is_cpu() ? cudaCpuDeviceId : id_; }
  int id() const { return id_; }

private:
  bool cpu_;
  int id_;
};

inline std::vector<Device> get_gpus()
{
  int numDevices;
  RT_CHECK(cudaGetDeviceCount(&numDevices));
  std::vector<Device> gpus(numDevices);
  for (int i = 0; i < numDevices; ++i)
  {
    gpus[i] = Device(false, i);
  }
  return gpus;
}

inline std::vector<Device> get_cpus()
{
  if (-1 != numa_available())
  {
    std::set<int> nodes;
    for (int i = 0; i < numa_num_configured_cpus(); ++i)
    {
      nodes.insert(numa_node_of_cpu(i));
    }
    const int numNodes = nodes.size();
    assert(nodes.size() >= 1);
    std::vector<Device> cpus;
    for (const auto &i : nodes)
    {
      cpus.push_back(Device(true, i));
    }
    return cpus;
  }
  else
  {
    return {{Device(true, 0)}};
  }
}

void bind_cpu(const Device &d)
{
  if (-1 != numa_available())
  {
    if (d.is_cpu())
    {
      bitmask *mask = numa_allocate_nodemask();
      assert(0 == numa_node_to_cpus(d.id(), mask));
      numa_bind(mask);
      numa_free_nodemask(mask);
    }
  }
}

std::vector<int64_t> sequence_geometric(int64_t min, int64_t max, double step)
{
  double min_d = static_cast<double>(min);
  double max_d = static_cast<double>(max);

  std::vector<int64_t> seq;

  for (double i = min_d; i < max_d; i *= step)
  {
    seq.push_back(i);
  }
  return seq;
}

std::vector<int64_t> merge(const std::vector<int64_t> &a, const std::vector<int64_t> &b)
{
  std::vector<int64_t> seq(a.size() + b.size());
  std::vector<int64_t> a_s(a);
  std::vector<int64_t> b_s(b);
  std::sort(a_s.begin(), a_s.end());
  std::sort(b_s.begin(), b_s.end());
  std::merge(a_s.begin(), a_s.end(), b_s.begin(), b_s.end(), seq.begin());
  auto end = std::unique(seq.begin(), seq.end());
  seq.resize(std::distance(seq.begin(), end));
  return seq;
}

#endif
