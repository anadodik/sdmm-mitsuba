#! /bin/bash

SCRIPT_DIR=`dirname $0`

cd mitsuba && scons -j 32 && source setpath.sh && cd ..

python3 ${SCRIPT_DIR}/run_tests.py \
    --scene all \
    --integrator sdmm \
    --name async_4spp_8min \
    --processors 16 \
    --maxDepth 10 \
    --sampleCount 1024 \
    --option samplesPerIteration 4

# python3 ${SCRIPT_DIR}/run_tests.py \
#     --scene all \
#     --integrator ppg \
#     --name perf \
#     --processors 16 \
#     --maxDepth 10 \
#     --sampleCount 1024

