#! /bin/bash

SCRIPT_DIR=`dirname $0`

cd mitsuba && scons -j 8 && source setpath.sh && cd ..

python3 ${SCRIPT_DIR}/run_tests.py \
    --scene torus \
    --integrator sdmm \
    --name perf \
    --processors 128 \
    --maxDepth 10 \
    --sampleCount 1024 \
    --option correctStateDensity false

python3 ${SCRIPT_DIR}/run_tests.py \
    --scene torus \
    --integrator ppg \
    --name perf \
    --processors 128 \
    --maxDepth 10 \
    --sampleCount 1024
