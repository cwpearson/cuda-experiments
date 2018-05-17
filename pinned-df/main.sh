#! /bin/bash



for cpu in `util/lscpu`; do
  for gpu in `util/lsgpu`; do
     echo "src,dst,count,time" > pinned-df/cpu${cpu}-gpu${gpu}.csv
     pinned-df/main --src-numa $cpu --dst-gpu $gpu -n 15 | tee -a pinned-df/cpu${cpu}-gpu${gpu}.csv
     echo "src,dst,count,time" > pinned-df/gpu${gpu}-cpu${cpu}.csv
     pinned-df/main --src-gpu $gpu --dst-numa $cpu -n 15 | tee -a pinned-df/gpu${gpu}-cpu${cpu}.csv
  done
done