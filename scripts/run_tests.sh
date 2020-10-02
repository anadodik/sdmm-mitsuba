#! /bin/bash

SCRIPT_DIR=`dirname $0`

cd mitsuba && scons -j 32 && source setpath.sh && cd ..

python3 ${SCRIPT_DIR}/run_tests.py \
    --scene torus \
    --integrator sdmm \
    --name perf \
    --processors 32 \
    --maxDepth 10 \
    --sampleCount 1024 \
    --option samplesPerIteration 8

python3 ${SCRIPT_DIR}/run_tests.py \
    --scene torus \
    --integrator ppg \
    --name perf \
    --processors 32 \
    --maxDepth 10 \
    --sampleCount 1024
