#!/usr/bin/env python3 

import argparse
from enum import Enum, auto
from collections import OrderedDict
from colorama import Fore, Style
import sys
import subprocess
from glob import glob
import json
import numpy as np
import skimage as ski
import os
import re
import smartexr as exr
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from test_suite_utils import get_gt_path, MrSE, MAPE, SMAPE, aggregate
from test_suite_utils import SCENES, SCENE_TITLES, RESULTS_PATH

from matplotlib import rc
rc('interactive') == False
# rc('text', usetex=True)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
        
def tonemap_exrs(experiment_path):
    image_files = [os.path.join(experiment_path, filename) for filename in os.listdir(experiment_path)
        if filename[-4:] == '.exr']
    output_directory = os.path.join(experiment_path, 'tonemapped')
    os.makedirs(output_directory, exist_ok=True)
    
    for image_file in image_files:
        filename = os.path.basename(image_file)
        png_filename = os.path.splitext(filename)[0] + '.png'
        output_file = os.path.join(output_directory, png_filename)
        print(f'Tonemapping {image_file}.')
        process = subprocess.Popen(['mtsutil', 'tonemap', '-o', output_file, image_file])
        process.wait()

def load_iteration(image_file, image_sqr_file):
    image_exr = exr.SmartExr(image_file)
    attributes = image_exr.attributes

    spp = attributes['spp']
    image_data = exr.read(image_file)
    image_sqr_data = exr.read(image_sqr_file)
    return image_data, image_sqr_data, spp


class ErrorType(Enum):
    CUMULATIVE = auto()
    INDIVIDUAL = auto()

def get_errors(run_dir, combination_type='var', error_type=ErrorType.CUMULATIVE, integrator='sdmm'):
    run_dir = os.path.join(run_dir, "") # adds to the end '/' if needed
    out_dir = os.path.dirname(run_dir)

    print(f"Combining run with weighing heuristic: {combination_type}.")

    rexp = r'^iteration_[0-9]*\.exr$' if integrator == 'ppg' else r'^iteration[0-9]*\.exr$'
    image_files = sorted([
        filename
        for filename in os.listdir(run_dir)
        if re.search(rexp, filename)
    ])

    json_dir = os.path.join(run_dir, "stats.json")

    scene_name = Path(out_dir).parents[2].name
    gt_path = get_gt_path(scene_name)
    gt_image = exr.read(gt_path)

    final_image = None
    total_reciprocal_weight = 0.0

    var_data=[]
    MrSE_data = []
    MAPE_data = []
    SMAPE_data = []
    spp_data=[]
    total_spp = 0

    stats_json = None
    if not os.path.exists(json_dir):
        print('Cannot find PPG stats.json.')
    with open(json_dir) as json_file:
        stats_json = json.load(json_file)

    elapsed_seconds = []
    var_data = []
    spp_data = []
    for i in range(len(stats_json)):
        elapsed_seconds.append(stats_json[i]['total_elapsed_seconds'])
        var_data.append(stats_json[i]['mean_pixel_variance'])
        spp_data.append(stats_json[i]['spp'])

    for iteration, image_file in tqdm(enumerate(image_files), total=len(image_files)):
        iteration_image = exr.read(os.path.join(run_dir, image_file))

        var = var_data[iteration]
        spp = spp_data[iteration]
        if combination_type == 'var':
            weight = var_data[iteration]
        elif combination_type == 'uniform':
            weight = 1 / spp

        if (integrator == 'ppg' and weight is not None) or (integrator == 'sdmm' and iteration >= 4): # iteration >= 3: # or iteration > 10:
            total_reciprocal_weight += 1.0 / weight
            if final_image is None:
                final_image = iteration_image / weight
            else:
                final_image += iteration_image / weight

        if final_image is not None and error_type == ErrorType.CUMULATIVE:
            combined_estimate = final_image / total_reciprocal_weight
            MrSE_data.append(aggregate(MrSE(np.clip(combined_estimate, 0, 1), np.clip(gt_image, 0, 1))))
            MAPE_data.append(aggregate(MAPE(combined_estimate, gt_image)))
            SMAPE_data.append(aggregate(SMAPE(combined_estimate, gt_image)))
        elif final_image is None or error_type == ErrorType.INDIVIDUAL:
            MrSE_data.append(aggregate(MrSE(np.clip(iteration_image, 0, 1), np.clip(gt_image, 0, 1))))
            MAPE_data.append(aggregate(MAPE(iteration_image, gt_image)))
            SMAPE_data.append(aggregate(SMAPE(iteration_image, gt_image)))

    final_image /= total_reciprocal_weight

    SMAPE_final = aggregate(SMAPE(final_image, gt_image))
    MAPE_final = aggregate(MAPE(final_image, gt_image))
    MrSE_final = aggregate(MrSE(np.clip(final_image, 0, 1), np.clip(gt_image, 0, 1)))

    if len(elapsed_seconds) > 0:
        print(
            f"{Fore.GREEN}"
            f"Total time: {elapsed_seconds[-1]:.3f}s, "
            f"MAPE: {MAPE_final:.3f}s, "
            f"{Style.RESET_ALL}"
        )

    return {
        'MAPE': (elapsed_seconds, MAPE_data, MAPE_final),
        'SMAPE': (elapsed_seconds, SMAPE_data, SMAPE_final),
        'MrSE': (elapsed_seconds, MrSE_data, MrSE_final),
    }


def combine_renders(run_dir, combination_type='var', error_type=ErrorType.CUMULATIVE):
    run_dir = os.path.join(run_dir, "") # adds to the end '/' if needed
    out_dir = os.path.dirname(run_dir)

    print(f"Combining run with weighing heuristic: {combination_type}.")

    image_files = sorted([
        filename
        for filename in os.listdir(run_dir)
        if re.search(r'^iteration[0-9]*\.exr$', filename)
    ])

    image_sqr_files = sorted([
        filename
        for filename in os.listdir(run_dir)
        if re.search(r'^iteration_sqr[0-9]*\.exr$', filename)
    ])

    scene_name = Path(out_dir).parents[2].name
    gt_path = get_gt_path(scene_name)
    gt_image = exr.read(gt_path)

    final_image = None
    total_reciprocal_weight = 0.0

    var_data=[]
    MrSE_data = []
    MAPE_data = []
    SMAPE_data = []
    spp_data=[]
    total_spp = 0

    for image_file, image_sqr_file in tqdm(
        zip(image_files, image_sqr_files), total=len(image_files)
    ):
        iteration = int(
            re.search(r'^iteration([0-9]*)\.exr$', image_file).group(1)
        )
        # print('Processing iteration {}.'.format(iteration))
        iteration_image, iteration_square_image, spp = load_iteration(
            os.path.join(run_dir, image_file),
            os.path.join(run_dir, image_sqr_file)
        )

        image_shape = iteration_image.shape[:2]

        image_variance = spp / (spp - 1) * (
            iteration_square_image / spp - (iteration_image / spp) ** 2
        )
        # image_variance = ski.filters.gaussian(
        #     image_variance, sigma=3, multichannel=True
        # )
        image_variance = np.clip(image_variance, 0, 2000)
        per_channel_variance = np.mean(image_variance, axis=(0,1))
        max_variance = np.maximum(image_variance, per_channel_variance)

        exr.write(os.path.join(out_dir, f'iteration_var{iteration}.exr'), image_variance)

        total_spp += spp
        spp_data.append(total_spp)
        # print('Iteration {} spp={}.'.format(iteration, spp))

        var_data.append(aggregate(image_variance))
        if iteration >= 3: # or iteration > 10:
            # print('Including iteration {}.'.format(iteration))
            if combination_type == 'var':
                weight = per_channel_variance
            elif combination_type == 'max_var':
                weight = max_variance
            elif combination_type == 'uniform':
                weight = 1 / spp
            total_reciprocal_weight += 1.0 / weight
            if final_image is None:
                final_image = iteration_image / weight
            else:
                final_image += iteration_image / weight


        if final_image is not None and error_type == ErrorType.CUMULATIVE:
            combined_estimate = final_image / total_reciprocal_weight
            # out_path = os.path.join(out_dir, f'combined_{iteration:02d}' + '.exr')
            # if os.path.exists(out_path):
            #     os.remove(out_path)
            # exr.write(out_path, combined_estimate.astype(np.float32))
            MrSE_data.append(aggregate(MrSE(np.clip(combined_estimate, 0, 1), np.clip(gt_image, 0, 1))))
            MAPE_data.append(aggregate(MAPE(combined_estimate, gt_image)))
            SMAPE_data.append(aggregate(SMAPE(combined_estimate, gt_image)))
        elif final_image is None or error_type == ErrorType.INDIVIDUAL:
            MrSE_data.append(aggregate(MrSE(np.clip(iteration_image, 0, 1), np.clip(gt_image, 0, 1))))
            MAPE_data.append(aggregate(MAPE(iteration_image, gt_image)))
            SMAPE_data.append(aggregate(SMAPE(iteration_image, gt_image)))

    if len(var_data) <= 1:
        return {}

    json_dir = os.path.join(run_dir, "stats.json")
    total_elapsed_seconds = None
    mean_path_length = None
    elapsed_seconds = []
    if os.path.exists(json_dir):
        with open(json_dir) as json_file:
            stats_json = json.load(json_file)
            for i in range(len(stats_json)):
                stats_json[i]['mean_pixel_variance']=float(var_data[i])
                stats_json[i]['ttuv']=float(var_data[i]) * stats_json[i]['elapsed_seconds']
                elapsed_seconds.append(stats_json[i]['total_elapsed_seconds'])
            last_stats = stats_json[-1]
            total_elapsed_seconds=last_stats['total_elapsed_seconds']
            mean_path_length=last_stats['mean_path_length']
        with open(json_dir, 'w') as json_file:
            json_file.write(json.dumps(stats_json, sort_keys=True, indent=4))

    plot_filename = 'error.pdf'
    plot_file = os.path.join(out_dir, plot_filename) 
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', axis='both', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', axis='both', linestyle=':', linewidth='0.5')
    ax.semilogy(elapsed_seconds, var_data, basey=10, label='Estimated VAR')
    ax.semilogy(elapsed_seconds, SMAPE_data, basey=10, label='SMAPE')
    ax.semilogy(elapsed_seconds, MrSE_data, basey=10, label='MRSE')
    ax.semilogy(elapsed_seconds, MAPE_data, basey=10, label='MAPE')
    # ax.plot(var_data, label='Estimated VAR')
    # ax.plot(SMAPE_data, label='SMAPE')
    # ax.plot(MrSE_data, label='MRSE')
    # ax.set_yticks(var_data)
    ax.legend()
    fig.tight_layout()
    plt.savefig(plot_file, format=os.path.splitext(plot_filename)[-1][1:], dpi=fig.dpi)

    if final_image is None:
        print('No final image produced!')
        return

    final_image /= total_reciprocal_weight

    SMAPE_final = aggregate(SMAPE(final_image, gt_image))
    MAPE_final = aggregate(MAPE(final_image, gt_image))
    MrSE_final = aggregate(MrSE(np.clip(final_image, 0, 1), np.clip(gt_image, 0, 1)))

    print(f"{Fore.GREEN}MAPE={MAPE_final:.3f}, SMAPE={SMAPE_final:.3f}, MrSE={MrSE_final:.3f}{Style.RESET_ALL}")
    if total_elapsed_seconds and mean_path_length:
        print(
            f"{Fore.GREEN}"
            f"Total time: {total_elapsed_seconds:.3f}s, "
            f"Mean path length: {mean_path_length:.3f}."
            f"{Style.RESET_ALL}"
        )

    out_path = os.path.join(out_dir, scene_name + '.exr')
    if os.path.exists(out_path):
        os.remove(out_path)
    exr.write(out_path, final_image.astype(np.float32))
    return {
        'MAPE': (elapsed_seconds, MAPE_data, MAPE_final),
        'SMAPE': (elapsed_seconds, SMAPE_data, SMAPE_final),
        'MrSE': (elapsed_seconds, MrSE_data, MrSE_final),
    }


def get_all_subdirs(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir()]

N_COMP_RUNS = OrderedDict([
    ('comp-80-spatial-48-directional-fixed-init', "$80$ spatial locations"),
    ('comp-170-spatial-48-directional-fixed-init', "$170$ spatial locations"),
    ('comp-360-spatial-48-directional-fixed-init', "$360$ spatial locations"),
])

PRODUCT = OrderedDict([
    ('comp-170-spatial-48-directional-fixed-init', "No product"),
    # ('comp-product', "Product"),
    # ('comp-product-per', "Diffuse Product"),
    ('comp-product-no-jac', "Diffuse Product"),
])

RUNS = OrderedDict([
    # ('radiance', "Radiance Guiding"),
    # ('radiance_2', "Radiance Guiding v2"),
    ('radiance_3', "Radiance Guiding v3"),
    ('radiance_4', "Radiance Guiding v4"),
    # ('radiance_3', "Radiance Performance"),
    # ('radiance_init', "Radiance Init"),
    # ('radiance_max_ch', "Radiance Max Ch."),
    # ('radiance_init_it', "Radiance Init Iteration"),
    ('comparison', "PPG"),
    # ('no_curv_1e-5', "No Curvature, 1e-5"),
    # ('no_curv_5e-8', "No Curvature, 5e-8"),
    # ('radiance_all_offset', "Radiance Guiding (All Offset)"),
    # ('radiance_none_offset', "Radiance Guiding (None Offset)"),
    # ('radiance_all_offset_r2', "Radiance Guiding (All Offset) 2"),
    # ('radiance_none_offset_r2', "Radiance Guiding (None Offset) 2"),
    # ('radiance_01_all_offset', "Radiance Guiding (All Offset, 0.1)"),

    # ('radiance_32c', "Radiance Guiding (32 Components)"),

    # ('product', "Product Guiding"),
    # ('product_2', "Product Guiding v2"),
    # ('product_3', "Product Guiding v3"),
    ('product_4', "Product Guiding v4"),
    # ('product_all_offset', "Product Guiding (All Offset)"),
    # ('product_all_offset_r2', "Product Guiding (All Offset) 2"),
    # ('product_none_offset', "Product Guiding (None Offset)"),
    # ('product_none_offset_r2', "Product Guiding (None Offset) 2"),
    # ('product_03', "Product Guiding (0.3 Ratio)"),
    # ('product_01', "Product Guiding (0.1 Ratio)"),
    # ('product_01_bsdf', "Product Guiding (0.1 BSDF Sampling)"),
    # ('product', "Product Guiding"),
    # ('product_32c', "Product Guiding (32 Components)"),
])

def make_comparison_figure(allowed_runs, name_prefix, error_type):
    scene_errors = {}
    for scene in SCENES:
        # if scene not in ['torus']: # ['cornell-box', 'glossy-cbox', 'torus', 'glossy-kitchen']:
        #     continue
        all_errors = {}
        for integrator in ['sdmm', 'ppg']:
            scene_path = os.path.join(RESULTS_PATH, scene, integrator)
            experiments = get_all_subdirs(scene_path)
            print(experiments)
            for experiment in experiments:
                if not os.path.basename(experiment) in allowed_runs.keys():
                    continue
                runs = get_all_subdirs(experiment)
                print(f'Found runs: {runs}.')
                assert(len(runs) == 1)
                for run in runs:
                    # print(f'Combining renders: {run}.')
                    errors = get_errors(run, 'var', error_type, integrator)
                    # print(f'errors={errors}')
                    all_errors[os.path.basename(experiment)] = errors
            scene_errors[scene] = all_errors
    plots = {
        # 'mrse.pdf': ['MrSE'],
        'mape.pdf': ['MAPE'],
        # 'smape.pdf': ['SMAPE'],
    }
    n_rows = 2 if len(scene_errors) > 1 else 1
    for plot_filename, allowed_errors in plots.items():
        result_path = os.path.join(RESULTS_PATH, '_plots', name_prefix + plot_filename)
        fig, ax = plt.subplots(nrows=n_rows, ncols=int(np.ceil(len(scene_errors) / 2)), sharex=True, sharey=False, figsize=(4 * len(scene_errors), 6), squeeze=False)
        ax = ax.flatten()
        for scene_i, (scene_name, all_errors) in enumerate(sorted(scene_errors.items())):
            ax[scene_i].set_axisbelow(True)
            ax[scene_i].minorticks_on()
            ax[scene_i].grid(which='major', axis='both', linestyle='-', linewidth='0.5')
            ax[scene_i].grid(which='minor', axis='both', linestyle=':', linewidth='0.5')
            for experiment_name in allowed_runs.keys():
                # print(f'Experiment name={experiment_name}')
                experiment_errors = all_errors[experiment_name]
                for error_name, errors in experiment_errors.items():
                    if error_name not in allowed_errors:
                        continue
                    print(f"Label[{experiment_name}]={allowed_runs[experiment_name]}")
                    ax[scene_i].semilogy(
                        errors[0],
                        errors[1],
                        label=f'{allowed_runs[experiment_name]}'
                    )
            # ax.set_yscale('log')
            # ax[scene_i].legend(fontsize="x-large")
            ax[scene_i].set_title(SCENE_TITLES[scene_name], fontsize="xx-large")

        labels_handles = {
            label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
        }

        lgd = fig.legend(
            labels_handles.values(),
            labels_handles.keys(),
            # loc="lower center",
            # ncol=len(allowed_runs),
            # bbox_to_anchor=(0.5, 0),
            # fontsize="xx-large",
        )
        # fig.tight_layout(rect=[0,0.1,1,1])
        print(f"Saving figure to {result_path}.")
        plt.savefig(
            result_path,
            bbox_extra_artists=(lgd,),
            bbox_inches='tight',
            format=os.path.splitext(plot_filename)[-1][1:],
            dpi=fig.dpi
        )


def compare_all_runs():
    allowed_runs = {
        'radiance': "Radiance Guiding",
        'radiance_32c': "Radiance Guiding (32 Components)",
        'product': "Product Guiding",
        'product_32c': "Product Guiding (32 Components)",
    }
    for scene in SCENES:
        if scene not in ['glossy-bathroom']:
            continue
        all_errors = {}
        scene_path = os.path.join(RESULTS_PATH, scene, 'sdmm')
        experiments = get_all_subdirs(scene_path)
        print(experiments)
        for experiment in experiments:
            if not os.path.basename(experiment) in allowed_runs.keys():
                continue
            runs = get_all_subdirs(experiment)
            # print(f'Found runs: {runs}.')
            assert(len(runs) == 1)
            for run in runs:
                # print(f'Combining renders: {run}.')
                errors = combine_renders(run, 'var')
                # print(f'errors={errors}')
                all_errors[os.path.basename(experiment)] = errors

        plots = {
            'mrse.svg': ['MrSE'],
            'mape.svg': ['MAPE'],
            'smape.svg': ['SMAPE'],
        }
        for plot_filename, allowed_errors in plots.items():
            plot_file = os.path.join(os.path.join(scene_path), plot_filename) 
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_axisbelow(True)
            ax.minorticks_on()
            ax.grid(which='major', axis='both', linestyle='-', linewidth='0.5')
            ax.grid(which='minor', axis='both', linestyle=':', linewidth='0.5')
            for experiment_name, experiment_errors in all_errors.items():
                # print(f'Experiment name={experiment_name}')
                for error_name, errors in experiment_errors.items():
                    if error_name not in allowed_errors:
                        continue
                    ax.semilogy(errors[0], label=f'{allowed_runs[experiment_name]}')
            # ax.set_yscale('log')
            ax.legend(fontsize="x-large")
            fig.tight_layout()
            plt.savefig(plot_file, format=os.path.splitext(plot_filename)[-1][1:], dpi=fig.dpi)

        for plot_filename, allowed_errors in plots.items():
            plot_file = os.path.join(os.path.join(scene_path), 'final_' + plot_filename) 
            fig, ax = plt.subplots(figsize=(12, 9))
        
            experiment_names = []
            final_errors = []
            for experiment_name, experiment_errors in all_errors.items():
                # print(f'Experiment name={experiment_name}')
                for error_name, errors in experiment_errors.items():
                    if error_name not in allowed_errors:
                        continue
                    print(f"{experiment_name}: {errors[1]}")
                    experiment_names.append(experiment_name)
                    final_errors.append(errors[1])
            x = np.arange(len(experiment_names))
            ax.set_xticks(x)
            ax.set_xticklabels(experiment_names)
            rects = ax.bar(x, final_errors)
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.03f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            # ax.legend()
            fig.tight_layout()
            plt.savefig(plot_file, format=os.path.splitext(plot_filename)[-1][1:], dpi=fig.dpi)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Combine SDMM Runs!')
    parser.add_argument('-t', '--tonemap', action='store_true')
    parser.add_argument('-c', '--combine_type', type=str, default='var')
    parser.add_argument('-o', '--comparison_out_file', type=str, nargs='?')
    parser.add_argument('-a', '--all', action='store_true')

    parser.add_argument('run_dirs', type=str, nargs='*')

    args = parser.parse_args()

    if args.all:
        # compare_all_runs()
        make_comparison_figure(RUNS, 'radiance_init_it_', ErrorType.CUMULATIVE)
        quit()

    all_errors = {}
    for run_dir in args.run_dirs:
        experiment_name = Path(run_dir).parent.name
        errors = combine_renders(run_dir, args.combine_type)
        all_errors[experiment_name] = errors
        if args.tonemap:
            tonemap_exrs(run_dir)

    if len(args.run_dirs) > 1 and args.comparison_out_file is not None:
        plots = {
            'mrse.svg': ['MrSE'],
            'mape.svg': ['MAPE', 'SMAPE'],
        }
        for plot_filename, allowed_errors in plots.items():
            plot_file = os.path.join(args.comparison_out_file, plot_filename) 
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_axisbelow(True)
            ax.minorticks_on()
            ax.grid(which='major', axis='both', linestyle='-', linewidth='0.5')
            ax.grid(which='minor', axis='both', linestyle=':', linewidth='0.5')
            for experiment_name, experiment_errors in all_errors.items():
                print(f'Experiment name={experiment_name}')
                for error_name, error_list in experiment_errors.items():
                    if error_name not in allowed_errors:
                        continue
                    ax.semilogy(error_list, subsy=error_list, basey=2, label=f'{experiment_name}: {error_name}')
            ax.legend()
            fig.tight_layout()
            plt.savefig(plot_file, format=os.path.splitext(plot_filename)[-1][1:], dpi=fig.dpi)
