#include "op.hpp"

void cpu_write_8(double *__restrict__ dummy, double *__restrict__ ptr, const size_t count, const size_t stride)
{
    (void)dummy;
    const size_t numElems = count / sizeof(double);
    const size_t elemsPerStride = stride / sizeof(double);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numElems; i += elemsPerStride)
    {
        ptr[i] = i * 31ul + 7ul;
    }
}

void cpu_read_8(double *__restrict__ dummy, double *__restrict__ ptr, const size_t count, const size_t stride)
{

    const size_t numElems = count / sizeof(double);
    const size_t elemsPerStride = stride / sizeof(double);

    double acc = 0;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numElems; i += elemsPerStride)
    {
        acc += ptr[i];
    }

    *dummy += acc;
}
