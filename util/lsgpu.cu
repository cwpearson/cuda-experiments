#include <cstdio>

int main(void) {
    int n;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) {
        return -1;
    }

    for (int i = 0; i < n; ++i) {
        printf("%d ",i);
    }
    printf("\n");
}