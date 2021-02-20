#!/usr/bin/env python3

import os
import re
import shutil

from test_suite_utils import SCENES, RESULTS_PATH

ALLOWED_TYPES = r'(.*\.json|.*\.exr|.*\.log|.*\.asdmm)$'
OUT_PATH = os.path.join(RESULTS_PATH, 'results_stats')

def copy(from_paths):
    from_pattern = re.compile(from_paths)
    types_pattern = re.compile(ALLOWED_TYPES)
    top = RESULTS_PATH
    for root, _, files in os.walk(top):
        match = from_pattern.match(root)
        if match and match.group(1) in SCENES:
            new_dir = os.path.join(OUT_PATH, root[len(top) + 1:])
            os.makedirs(new_dir, exist_ok=True)
            for f in files:
                if not 'var' in f and not 'sqr' in f and types_pattern.match(f):
                    from_path = os.path.join(root, f)
                    to_path = os.path.join(new_dir, f)
                    # print(from_path, ' --> ', to_path)
                    shutil.copy2(from_path, to_path)

SDMM_FROM_PATHS = [
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_1/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_2/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_3/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_4/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_5/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',

    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_1/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_2/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_3/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_4/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_5/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',

    # RESULTS_PATH + '/(.*)/sdmm/sdmm_bsdf_1/bsdfOnly=true,maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_bsdf_2/bsdfOnly=true,maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_bsdf_3/bsdfOnly=true,maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_bsdf_4/bsdfOnly=true,maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_bsdf_5/bsdfOnly=true,maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',

    # RESULTS_PATH + '/(.*)/sdmm/dmm_radiance_1/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/dmm_radiance_2/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/dmm_radiance_3/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/dmm_radiance_4/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/dmm_radiance_5/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',

    # RESULTS_PATH + '/(.*)/sdmm/dmm_product_1/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/dmm_product_2/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/dmm_product_3/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/dmm_product_4/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/dmm_product_5/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',

    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_16c_1/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_16c_2/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_16c_3/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_16c_4/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_radiance_16c_5/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',

    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_16c_1/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_16c_2/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_16c_3/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_16c_4/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    # RESULTS_PATH + '/(.*)/sdmm/sdmm_product_16c_5/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
    RESULTS_PATH + '/(.*)/sdmm/sdmm_stats_1/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',
    RESULTS_PATH + '/(.*)/sdmm/dmm_stats_1/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=4',

    # RESULTS_PATH + '/(.*)/sdmm/sdmm_single_mixture_radiance_648c/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=4',
]
GT_FROM_PATHS = RESULTS_PATH + '/(.*)/gt'
PPG_FROM_PATHS = [
    # RESULTS_PATH + '/(.*)/ppg/baseline_1/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024',
    # RESULTS_PATH + '/(.*)/ppg/baseline_2/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024',
    # RESULTS_PATH + '/(.*)/ppg/baseline_3/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024',
    # RESULTS_PATH + '/(.*)/ppg/baseline_4/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024',
    # RESULTS_PATH + '/(.*)/ppg/baseline_5/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024',
]
PATH_FROM_PATHS = [
    # RESULTS_PATH + '/(.*)/ppg/unidirectional_path_tracing_1/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024'
    # RESULTS_PATH + '/(.*)/ppg/unidirectional_path_tracing_2/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024'
    # RESULTS_PATH + '/(.*)/ppg/unidirectional_path_tracing_3/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024'
    # RESULTS_PATH + '/(.*)/ppg/unidirectional_path_tracing_4/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024'
    # RESULTS_PATH + '/(.*)/ppg/unidirectional_path_tracing_5/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024'
]

# copy(GT_FROM_PATHS)
for path in SDMM_FROM_PATHS:
    copy(path)
for path in PPG_FROM_PATHS:
    copy(path)
for path in PATH_FROM_PATHS:
    copy(path)
