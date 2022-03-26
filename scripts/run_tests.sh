#! /bin/bash

SCRIPT_DIR=`dirname $0`

cd mitsuba && scons -j 32 && source setpath.sh && cd ..


for run in {2..5}
do
    # Config cleanup, fixing the tab width,
    # changing default number of components to 16,
    # python3 ${SCRIPT_DIR}/run_tests.py \
    #     -n zoom_$run \
    #     --scene glossy-cbox \
    #     --integrator sdmm \
    #     --processors 12 \
    #     --maxDepth 10 \
    #     --sampleCount 1024 \
    #     --option optimizeAsync false \
    #     --option samplesPerIteration 4

    python3 ${SCRIPT_DIR}/run_tests.py \
        -n radiance_timings_$run \
        --scene all \
        --integrator sdmm \
        --processors 12 \
        --maxDepth 10 \
        --sampleCount 1024 \
        --option optimizeAsync false \
        --option samplesPerIteration 4
        # --option sampleProduct true \

    # python3 ${SCRIPT_DIR}/run_tests.py \
    #     -n product_2048leafs_16000st_delay_on_$run \
    #     --scene all \
    #     --integrator sdmm \
    #     --processors 12 \
    #     --maxDepth 10 \
    #     --sampleCount 1024 \
    #     --option optimizeAsync false \
    #     --option sampleProduct true \
    #     --option samplesPerIteration 4

    # python3 ${SCRIPT_DIR}/run_tests.py \
    #     -n vmm_comp_$run \
    #     --scene torus \
    #     --integrator sdmm \
    #     --processors 12 \
    #     --maxDepth 10 \
    #     --sampleCount 1024 \
    #     --option optimizeAsync false \
    #     --option samplesPerIteration 4

    # python3 ${SCRIPT_DIR}/run_tests.py \
    #     -n sdmm_old_$run \
    #     --scene cornell-box \
    #     --integrator sdmm \
    #     --processors 12 \
    #     --maxDepth 10 \
    #     --sampleCount 1024 \
    #     --option samplesPerIteration 4 \
    #     --option optimizeAsync false \
    #     --option flushDenormals false

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
    #     -n zoom_gt_$run \
    #     --scene glossy-cbox \
    #     --integrator ppg \
    #     --processors 12 \
    #     --maxDepth 10 \
    #     --sampleCount 20000

    #   --option bsdfSamplingFraction 1.0 \
done

