#include <cstdio>

#include "common/common.hpp"

int main(void) {
    auto cpus = get_cpus();

    for (const auto cpu : cpus) {
        printf("%d ", cpu.id());
    }
    printf("\n");
}