#! /bin/bash

set -eu

MACHINE=$1
modules=(cpu-touch ctx)

make

for m in "${modules[@]}"; do
    mkdir -pv "results/$MACHINE/$m"
    nvprof -o "results/$MACHINE/$m/timeline.nvvp" -f "$m"/main
done
