#!/bin/bash

CUDADIR=$(dirname $(dirname $(which nvcc)))
sed -e "s|^CUDADIR=$|&${CUDADIR}|" \
    -e "s|^BOOSTDIR=$|&${HOME}/boost|" \
    -e "s|^TBB=$|&${HOME}/tbb|" \
    Makefile.template > Makefile
