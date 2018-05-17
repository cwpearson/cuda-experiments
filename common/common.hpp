#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <string>

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
std::vector<Device> get_gpus(const std::vector<int> &ids);
std::vector<Device> get_cpus(const std::vector<int> &ids);

void bind_cpu(const Device &d);
size_t num_cpus(const Device &d);
size_t num_mps(const Device &d);
size_t max_threads_per_mp(const Device &d);
size_t max_blocks_per_mp(const Device &d);
size_t gpu_free_memory(const std::vector<Device> &devs);
long long cpu_free_memory(const std::vector<Device> &devs);
size_t free_memory(const std::vector<Device> &devs);

size_t min_cpus_per_node(const std::vector<Device> &devs);

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

bool option_as_int(const int argc, char *const *const argv, const char *opt, int &val);
bool option_as_int_list(const int argc, char *const *const argv, const char *opt, std::vector<int> &vals);
bool option_as_ull(const int argc, char *const *const argv, const char *opt, unsigned long long &val);

#endif
