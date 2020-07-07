#! /bin/bash

cd mitsuba && scons -j 8 && source setpath.sh && cd ..

python3 run_tests.py \
    --scene torus \
    --integrator sdmm \
    --name perf \
    --processors 128 \
    --maxDepth 10 \
    --sampleCount 1024 \
    --option correctStateDensity false

python3 run_tests.py \
    --scene torus \
    --integrator ppg \
    --name perf \
    --processors 128 \
    --maxDepth 10 \
    --sampleCount 1024

# python3 run_tests.py \
#     --scene necklace \
#     --integrator gt \
#     --name gt \
#     --processors 128 \
#     --maxDepth 10 \
#     --sampleCount 131072
