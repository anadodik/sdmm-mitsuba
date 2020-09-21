import re, sys, os, subprocess, shutil
import argparse
from colorama import Fore, Style
import json
import numpy as np

from test_suite_utils import MITSUBA_PATH, SCENES
from test_suite_utils import get_scene_path, get_experiment_path, get_gt_path, get_combined_path, get_plots_path
from combine_renders import combine_renders

def render_experiment(scene_name, experiment_name, integrator, parameters, processors):
    print('Running experiment {}, scene {}, with {} and {}'.format(experiment_name, scene_name, integrator, parameters))
    experiment_path = get_experiment_path(scene_name, experiment_name, integrator, parameters)
    scene_path = get_scene_path(scene_name)
    output_path = os.path.join(experiment_path, experiment_name + '.exr')
    combined_path = get_combined_path(experiment_path, scene_name)

    shutil.rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path, exist_ok=True)

    parameters = {'integrator' : integrator, **parameters}
    mts_arguments = ['-D' + name + '=' + value for name, value in parameters.items()]

    stdout_path = os.path.join(experiment_path, 'stdout.log')
    stderr_path = os.path.join(experiment_path, 'stderr.log')
    with open(stdout_path, 'wb+', 0) as stdout_file, open(stderr_path, 'wb+', 0) as stderr_file:
        os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libprofiler.so'
        os.environ['CPUPROFILE'] = 'mts.prof'
        os.environ['CPUPROFILE_FREQUENCY'] = '800'
        mts_call_list = [
            'mitsuba',
            '-z',
            '-p',
            processors,
            *mts_arguments,
            '-o',
            output_path,
            scene_path
        ]
        print('Calling ' + ' '.join(mts_call_list))

        process = subprocess.Popen(
            mts_call_list,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=MITSUBA_PATH
        )
        # for line in process.stdout:
        #     sys.stdout.write(line.decode('utf-8'))
        #     stdout_file.write(line)

        process.wait()

    duration_regex = re.compile('Render time: ([0-9]+\.[0-9]+)(s|m)')
    path_length_regex = re.compile('Average path length : ([0-9]+\.[0-9]+)')
    duration_s = None
    path_length = None
    with open(stdout_path, 'r') as f:
        for line in f:
            duration_found = duration_regex.search(line)
            if duration_found:
                print(f"{Fore.GREEN}{duration_found.group(0)}.{Style.RESET_ALL}")
                duration_s = float(duration_found.group(1))
                if duration_found.group(2) == 'm':
                    duration_s *= 60
            path_length_found = path_length_regex.search(line)
            if path_length_found:
                print(f"{Fore.GREEN}{path_length_found.group(0).replace(' :', ':')} rays.{Style.RESET_ALL}")
                path_length = float(path_length_found.group(1))
    print(f"{Fore.GREEN}Time per ray: {duration_s / path_length:0.2f}s.{Style.RESET_ALL}")

    json_dir = os.path.join(experiment_path, "stats.json")
    if os.path.exists(json_dir):
        total_elapsed_seconds = None
        with open(json_dir) as json_file:
            stats_json = json.load(json_file)
            last_stats = stats_json[-1]
            total_elapsed_seconds=last_stats['total_elapsed_seconds']
        print(
            f"{Fore.YELLOW}"
            f"Total time: {total_elapsed_seconds:.3f}s."
            f"{Style.RESET_ALL}"
        )
        

def combine(scene_name, experiment_name, integrator, parameters):
    experiment_path = get_experiment_path(scene_name, experiment_name, integrator, parameters)
    combine_renders(experiment_path, 'var')

def render_gt(scene_name, parameters):
    print('Running ground truth with scene {}, with {}'.format(scene_name, parameters))
    output_path = get_gt_path(scene_name)
    scene_path = get_scene_path(scene_name)

    output_directory = os.path.dirname(output_path)
    os.makedirs(output_directory, exist_ok=True)

    parameters = {'integrator' : 'gt', **parameters}
    mts_arguments = ['-D' + name + '=' + value for name, value in parameters.items()]
    stdout_path = os.path.join(output_directory, 'stdout.log')
    stderr_path = os.path.join(output_directory, 'stderr.log')

    with open(stdout_path, 'wb+', 0) as stdout_file, open(stderr_path, 'wb+', 0) as stderr_file:
        mts_call_list = ['mitsuba', '-z', '-b', '8', '-r', '180', *mts_arguments, '-o', output_path, scene_path]
        print('Calling ' + ' '.join(mts_call_list))
        process = subprocess.Popen(
            mts_call_list,
            stdout=stdout_file, stderr=stderr_file, cwd=MITSUBA_PATH)
        process.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the SDMM test suite.')
    parser.add_argument('--name', type=str, help='Name of the experiment.', required=True)
    parser.add_argument('--integrator', type=str, help='Name of the integrator to use.', required=True)
    parser.add_argument('--scene', type=str, help='Name of the scene to render.', required=True)
    parser.add_argument('--processors', type=str, help='Override the default number of processors.', required=True)
    parser.add_argument('--combine', action='store_true', help='Do not render, only combine renders', default=False)

    settings_parser = parser.add_argument_group('Render settings')
    settings_parser.add_argument('--sampleCount', type=int, help='Total samples per pixel.', required=True)
    settings_parser.add_argument('--maxDepth', type=int, help='Maximum path depth.', default=10)
    settings_parser.add_argument('--rrDepth', type=int, help='Minimum Russian roulette depth.', default=10)

    sdmm_parser = parser.add_argument_group('SDMM settings')
    sdmm_parser.add_argument('--option', action='append', nargs='+')
    args = parser.parse_args()
    print(args.option)

    parameters = {
        'sampleCount' : str(args.sampleCount),
        'maxDepth' : str(args.maxDepth),
        'rrDepth' : str(args.rrDepth),
    }

    if args.integrator == 'sdmm':
        for name, value in args.option:
            parameters[name] = value
    elif args.integrator == 'gt':
        parameters['nee'] = 'true'
    elif args.integrator == 'ppg':
        parameters['budget'] = str(args.sampleCount)
        parameters['budgetType'] = 'spp'

    def run(scene_name):
        if args.integrator == 'gt':
            render_gt(scene_name, parameters)
        elif args.integrator == 'path':
            render_experiment(scene_name, args.name, args.integrator, parameters, args.processors)
        elif args.integrator == 'ppg':
            render_experiment(scene_name, args.name, args.integrator, parameters, args.processors)
        elif args.integrator == 'sdmm':
            if args.combine:
                combine(scene_name, args.name, args.integrator, parameters)
            else:
                render_experiment(scene_name, args.name, args.integrator, parameters, args.processors)
                combine(scene_name, args.name, args.integrator, parameters)

    if args.scene == 'all':
        for scene_name in SCENES:
            run(scene_name)
    else:
        assert(args.scene in SCENES)
        run(args.scene)
