#include <cstdio>
#include <climits>
#include <algorithm>

#include <cuda_runtime.h>

int main(void)
{

    int nDev;
    int ccMajor, ccMinor;
    cudaError_t err;
    cudaDeviceProp prop;

    if ((err = cudaGetDeviceCount(&nDev)) != cudaSuccess)
    {
        return err;
    }

    if (nDev == 0)
    {
        ccMajor = 0;
        ccMinor = 0;
    }
    else
    {
        ccMajor = INT_MAX;
        ccMinor = INT_MAX;
        for (int i = 0; i < nDev; ++i)
        {
            if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess)
            {
                return err;
            }

            ccMajor = std::min(ccMajor, prop.major);
            ccMinor = std::min(ccMinor, prop.minor);
        }
    }

    printf("%d%d\n", ccMajor, ccMinor);
    return 0;
}