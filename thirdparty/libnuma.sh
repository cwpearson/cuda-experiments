#! /bin/bash

set -eou pipefail
set -x

URL="https://github.com/numactl/numactl/archive/v2.0.11.tar.gz"

wget $URL -O libnuma.tar.gz
tar -xf libnuma.tar.gz

LIBNUMA_PREFIX=$(readlink -f .)

## Install libnuma to EXTERNAL_DIR
cd numactl-2.0.11 && \
   ./autogen.sh && \
   ./configure --prefix="$LIBNUMA_PREFIX" && \
   make \
   && make install \
   && cd -
