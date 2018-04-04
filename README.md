# cuda-experiments

Supports NVVC 7, 8, and 9

Build all supported tests

    make

Run all supported tests and dump outputs in `$MODULE.csv`

    make bench

## stream-thread

https://devblogs.nvidia.com/maximizing-unified-memory-performance-cuda/

## stream-warp

https://devblogs.nvidia.com/maximizing-unified-memory-performance-cuda/

## To-dos:

Check write performance impact of set read mostly, with increasing number of GPUs that have a read-only copy of the page.

Check peer access enable on GPU coherence bandwidth

