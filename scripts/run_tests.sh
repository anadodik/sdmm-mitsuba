#! /bin/bash

SCRIPT_DIR=`dirname $0`

cd mitsuba && scons -j 32 && source setpath.sh && cd ..


for run in {1..1}
do
    # Config cleanup, fixing the tab width,
    # changing default number of components to 16,
    python3 ${SCRIPT_DIR}/run_tests.py \
        -n sdmm_radiance_$run \
        --scene cornell-box \
        --integrator sdmm \
        --processors 12 \
        --maxDepth 10 \
        --sampleCount 1024 \
        --option samplesPerIteration 4

    python3 ${SCRIPT_DIR}/run_tests.py \
        -n sdmm_old_$run \
        --scene cornell-box \
        --integrator sdmm \
        --processors 12 \
        --maxDepth 10 \
        --sampleCount 1024 \
        --option samplesPerIteration 4 \
        --option optimizeAsync false \
        --option flushDenormals false

    # python3 ${SCRIPT_DIR}/run_tests.py \
    #     -n sdmm_product_$run \
    #     --scene all \
    #     --integrator sdmm \
    #     --processors 12 \
    #     --maxDepth 10 \
    #     --sampleCount 1024 \
    #     --option samplesPerIteration 4 \
    #     --option sampleProduct true
    
    # python3 ${SCRIPT_DIR}/run_tests.py \
    #     -n sdmm_bsdf_$run \
    #     --scene all \
    #     --integrator sdmm \
    #     --processors 12 \
    #     --maxDepth 10 \
    #     --sampleCount 1024 \
    #     --option samplesPerIteration 4 \
    #     --option bsdfOnly true

    # python3 ${SCRIPT_DIR}/run_tests.py \
    #     -n unidirectional_path_tracing_$run \
    #     --scene all \
    #     --integrator ppg \
    #     --processors 12 \
    #     --maxDepth 10 \
    #     --sampleCount 1024

done

