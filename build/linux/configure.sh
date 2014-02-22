#!/bin/bash

CUDADIR=$(dirname $(dirname $(which nvcc)))
sed -e "s|^CUDADIR=$|&${CUDADIR}|" -e "s|^BOOSTDIR=$|&${HOME}|" Makefile.template > Makefile
