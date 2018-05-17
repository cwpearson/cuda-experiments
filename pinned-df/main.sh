#! /bin/bash

echo "src,dst,count,time"

for cpu in `util/lscpu`; do
  for gpu in `util/lsgpu`; do
     pinned-df/main --src-numa $cpu --dst-gpu $gpu -n 15
     pinned-df/main --src-gpu $gpu --dst-numa $cpu -n 15
  done
done