#include <chrono>

#include "common/common.hpp"

int main(void) {

    {
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaFree(0));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }
    RT_CHECK(cudaDeviceReset());
    {
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaFree(0));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }
    RT_CHECK(cudaDeviceReset());
    {
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(cudaFree(0));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }

}