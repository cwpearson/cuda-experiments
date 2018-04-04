#include "common.hpp"

Sequence Sequence::neighborhood(const Sequence &orig, const double window_scale, const size_t window_elems)
{
  Sequence s;

  for (const auto &val : orig)
  {
    for (size_t i = 1; i <= window_elems; ++i)
    {
      double mod = window_scale * (i / static_cast<double>(window_elems));
      s.seq_.push_back(val * (1.0 + mod));
      s.seq_.push_back(val / (1.0 + mod));
    }
    s.seq_.push_back(val);
  }

  std::sort(s.seq_.begin(), s.seq_.end());

  return s;
}

Sequence &Sequence::operator|=(const Sequence &rhs)
{
  std::vector<value_type> newSeq(seq_.size() + rhs.seq_.size());

  std::merge(seq_.begin(), seq_.end(), rhs.seq_.begin(), rhs.seq_.end(), newSeq.begin());
  auto end = std::unique(newSeq.begin(), newSeq.end());
  newSeq.resize(std::distance(newSeq.begin(), end));
  seq_ = std::move(newSeq);

  return *this;
}

Sequence Sequence::operator|(const Sequence &rhs) const
{
  Sequence s = *this;
  return s |= rhs;
}

Sequence Sequence::geometric(value_type min, value_type max, double step)
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

Sequence::container_type::const_iterator Sequence::begin() const
{
  return seq_.begin();
}

Sequence::container_type::const_iterator Sequence::end() const
{
  return seq_.end();
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

std::vector<Device> get_gpus()
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

std::vector<Device> get_cpus()
{
  if (-1 != numa_available())
  {
    std::set<int> nodes;
    for (int i = 0; i < numa_num_configured_cpus(); ++i)
    {
      nodes.insert(numa_node_of_cpu(i));
    }
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

Device::Device() {}
Device::Device(const bool cpu, const int id) : cpu_(cpu), id_(id) {}

std::string Device::name() const
{
  std::string s;
  if (is_cpu())
  {
    s = "CPU";
  }
  else
  {
    s = "GPU";
  }

  return s + std::to_string(id_);
}
bool Device::is_cpu() const { return cpu_; }
bool Device::is_gpu() const { return !cpu_; }
int Device::cuda_device_id() const { return is_cpu() ? cudaCpuDeviceId : id_; }
int Device::id() const { return id_; }

bool Device::operator==(const Device &other) const
{
  return (cpu_ == other.cpu_) && (id_ == other.id_);
}

bool Device::operator!=(const Device &other) const
{
  return !((*this) == other);
}
