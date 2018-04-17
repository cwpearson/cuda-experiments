#include "op.hpp"

double a;

void dummy(double *ptr)
{
    a = *ptr;
    ptr[0] = 1;
}