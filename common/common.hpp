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
  int cuda_device_id() const { return is_cpu() ? cudaCpuDeviceId : id_; }
  int id() const { return id_; }

  bool operator==(const Device &other) const
  {
    return (cpu_ == other.cpu_) && (id_ == other.id_);
  }

  bool operator!=(const Device &other) const
  {
    return !((*this) == other);
  }

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
      numa_bitmask_setbit(mask, d.id());
      numa_bind(mask);
      numa_free_nodemask(mask);
    }
  }
}

size_t num_mps(const Device &d)
{
  assert(d.is_gpu());
  cudaDeviceProp prop;
  RT_CHECK(cudaGetDeviceProperties(&prop, d.id()));
  return prop.multiProcessorCount;
}

size_t max_threads_per_mp(const Device &d)
{
  assert(d.is_gpu());
  cudaDeviceProp prop;
  RT_CHECK(cudaGetDeviceProperties(&prop, d.id()));
  return prop.maxThreadsPerMultiProcessor;
}

size_t max_blocks_per_mp(const Device &d)
{
  assert(d.is_gpu());
  cudaDeviceProp prop;
  RT_CHECK(cudaGetDeviceProperties(&prop, d.id()));
  if (prop.major <= 2)
  {
    return 8;
  }
  else if (prop.major <= 3)
  {
    return 16;
  }
  else if (prop.major <= 7)
  {
    return 32;
  }
  else
  {
    assert(0 && "Unexpected CC major version");
  }
  return prop.multiProcessorCount;
}

class Sequence
{
public:
  typedef int64_t value_type;
  typedef std::vector<value_type> container_type;

private:
  container_type seq_;

public:
  static Sequence geometric(value_type min, value_type max, double step)
  {
    double min_d = static_cast<double>(min);
    double max_d = static_cast<double>(max);

    Sequence s;

    for (double i = min_d; i < max_d; i *= step)
    {
      s.seq_.push_back(i);
    }

    return s;
  }

  Sequence &operator|=(const Sequence &rhs)
  {
    std::vector<value_type> newSeq(seq_.size() + rhs.seq_.size());

    std::merge(seq_.begin(), seq_.end(), rhs.seq_.begin(), rhs.seq_.end(), newSeq.begin());
    auto end = std::unique(newSeq.begin(), newSeq.end());
    newSeq.resize(std::distance(newSeq.begin(), end));
    seq_ = std::move(newSeq);

    return *this;
  }

  Sequence operator|(const Sequence &rhs)
  {
    Sequence s = *this;
    return s |= rhs;
  }

  container_type::const_iterator begin() const
  {
    return seq_.begin();
  }

  container_type::const_iterator end() const
  {
    return seq_.end();
  }
};

// std::vector<int64_t>
// sequence_geometric(int64_t min, int64_t max, double step)
// {
//   double min_d = static_cast<double>(min);
//   double max_d = static_cast<double>(max);

//   std::vector<int64_t> seq;

//   for (double i = min_d; i < max_d; i *= step)
//   {
//     seq.push_back(i);
//   }
//   return seq;
// }

// std::vector<int64_t> merge(const std::vector<int64_t> &a, const std::vector<int64_t> &b)
// {
//   std::vector<int64_t> seq(a.size() + b.size());
//   std::vector<int64_t> a_s(a);
//   std::vector<int64_t> b_s(b);
//   std::sort(a_s.begin(), a_s.end());
//   std::sort(b_s.begin(), b_s.end());
//   std::merge(a_s.begin(), a_s.end(), b_s.begin(), b_s.end(), seq.begin());
//   auto end = std::unique(seq.begin(), seq.end());
//   seq.resize(std::distance(seq.begin(), end));
//   return seq;
// }

#endif
