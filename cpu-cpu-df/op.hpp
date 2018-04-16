#ifndef CPUCPU_OP_HPP
#define CPUCPU_OP_HPP

#include <cstdlib>
void cpu_write_8(double *dummy, double *ptr, const size_t count, const size_t stride);
void cpu_read_8(double *dummy, double *ptr, const size_t count, const size_t stride);

#endif