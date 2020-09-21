#! python3

import os
import re
import shutil

from test_suite_utils import SCENES, RESULTS_PATH

ALLOWED_TYPES = r'(.*\.json|.*\.exr)$'
OUT_PATH = os.path.join(RESULTS_PATH, 'thomas_results_v3')

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
    RESULTS_PATH + '/(.*)/sdmm/radiance_3/maxDepth=10,rrDepth=10,sampleCo=1024,samplesP=8',
    RESULTS_PATH + '/(.*)/sdmm/product_3/maxDepth=10,rrDepth=10,sampleCo=1024,samplePr=true,samplesP=8',
]
GT_FROM_PATHS = RESULTS_PATH + '/(.*)/gt'
PPG_FROM_PATHS = RESULTS_PATH + '/(.*)/ppg/comparison/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024'
PATH_FROM_PATHS = RESULTS_PATH + '/(.*)/ppg/standard_path_tracing/budget=1024,budgetTy=spp,maxDepth=10,rrDepth=10,sampleCo=1024'

for path in SDMM_FROM_PATHS:
    copy(path)
# copy(GT_FROM_PATHS)
# copy(PPG_FROM_PATHS)
# copy(PATH_FROM_PATHS)
