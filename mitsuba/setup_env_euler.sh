#!/bin/bash

module purge

module unload gcc
module unload python
module unload boost
module unload cmake
module unload open_mpi
module unload opencv

module load modules
module load new

module load gcc/5.2.0
module load python/3.7.1
module load python/2.7
module load qt/5.9.6
module load boost/1.62.0
module load cmake/3.9
module load eigen/3.2.1
module load opencv
module load fftw

source setpath.sh

PATH=$PATH:$(pwd)/dependencies/local/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/dependencies/lib:$(pwd)/dependencies/lib64
PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/local/lib64/python2.7/site-packages

export PYTHONPATH
export PATH
export LD_LIBRARY_PATH




