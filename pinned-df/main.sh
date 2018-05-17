#! /bin/bash

for cpu in `util/lscpu`; do
  for gpu in `util/lsgpu`; do
     echo $cpu $gpu;
  done
done