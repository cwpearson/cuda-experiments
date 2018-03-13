#! /bin/bash

set -eu

MACHINE=$1
modules=(coherence cpu-touch ctx stream-thread stream-warp)

make

for m in "${modules[@]}"; do
    mkdir -pv "results/$MACHINE/$m"
    nvprof -o "results/$MACHINE/$m/timeline.nvvp" -f "$m"/main
done
