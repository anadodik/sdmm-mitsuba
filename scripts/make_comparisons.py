import sys, os, shutil, pprint
from pathlib import Path
import re

import smartexr as exr
import numpy as np
import pandas as pd

from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, NoEscape, Package, Description, SubFigure, Command, Table, StandAloneGraphic
from pylatex.utils import italic

from test_suite_utils import SCENES, SCENE_TITLES, RESULTS_PATH
from test_suite_utils import get_experiment_path_param_string, get_gt_path, get_combined_path, get_combined_path_ppg
from test_suite_utils import imsave, SSIM, MrSE, MAPE, SMAPE, L2, L1, format_value, to_precision, aggregate


def get_gt_images():
    images = {}
    for scene_name in SCENES:
        gt_path = get_gt_path(scene_name)
        if not os.path.exists(gt_path):
            continue
        gt = exr.read(gt_path)
        images[scene_name] = gt
    return images

def calculate_metrics_with_images(experiment_name, integrator, parameters):
    losses = {}
    images = {}
    for scene_name in SCENES:
        gt_path = get_gt_path(scene_name)
        if not os.path.exists(gt_path):
            continue
        gt = exr.read(gt_path)
        clipped_gt = np.clip(gt, 0.0, 1.0)
        experiment_path = get_experiment_path_param_string(scene_name, experiment_name, integrator, parameters)
        if integrator == 'ppg':
            combined_path = get_combined_path_ppg(experiment_path, experiment_name)
        else:
            combined_path = get_combined_path(experiment_path, scene_name)
        if not os.path.exists(combined_path):
            # print('{} does not exist!'.format(combined_path))
            continue
        img = exr.read(combined_path)
        clipped_img = np.clip(img, 0.0, 1.0)
        scene_losses = {
            "MAPE"   : aggregate(MAPE(img, gt)),
            # "SMAPE"  : aggregate(SMAPE(img, gt)),
            # "L1"     : aggregate(L1(img, gt)),
            "MRSE"   : aggregate(MrSE(clipped_img, clipped_gt)),
            # "L2"     : aggregate(L2(clipped_img, clipped_gt)),
            # "1-SSIM" : aggregate(1. - SSIM(clipped_img, clipped_gt)),
        }
        losses[SCENE_TITLES[scene_name]] = scene_losses
        images[scene_name] = img
    return losses, images

def calculate_metrics(experiment_name, integrator, parameters):
    losses = {}
    for scene_name in SCENES:
        gt_path = get_gt_path(scene_name)
        if not os.path.exists(gt_path):
            continue
        gt = exr.read(gt_path)
        clipped_gt = np.clip(gt, 0.0, 1.0)
        experiment_path = get_experiment_path_param_string(scene_name, experiment_name, integrator, parameters)
        if integrator == 'ppg':
            combined_path = get_combined_path_ppg(experiment_path, experiment_name)
        else:
            combined_path = get_combined_path(experiment_path, scene_name)
        if not os.path.exists(combined_path):
            # print('{} does not exist!'.format(combined_path))
            continue
        img = exr.read(combined_path)
        clipped_img = np.clip(img, 0.0, 1.0)
        scene_losses = {
            "MAPE"   : aggregate(MAPE(img, gt)),
            # "SMAPE"  : aggregate(SMAPE(img, gt)),
            # "L1"     : aggregate(L1(img, gt)),
            "MRSE"   : aggregate(MrSE(clipped_img, clipped_gt)),
            # "L2"     : aggregate(L2(clipped_img, clipped_gt)),
            # "1-SSIM" : aggregate(1. - SSIM(clipped_img, clipped_gt)),
        }
        losses[scene_name] = scene_losses
    return losses

def calculate_all_with_images(integrator, experiment_name, re_filter=''):
    results_path = Path(RESULTS_PATH)
    glob_path = '*/' + integrator + '/' + experiment_name + '/*'
    experiments = sorted(results_path.glob(glob_path))
    experiment_parameters = list(set([experiment.name for experiment in experiments]))
    experiment_parameters = [experiment_parameter for experiment_parameter in experiment_parameters if experiment_parameter[0] != '.']
    assert(len(experiment_parameters) == 1)
    parameters_string = experiment_parameters[0]
    assert(re.search(re_filter, parameters_string))
    return calculate_metrics_with_images(experiment_name, integrator, parameters_string)

def calculate_all(integrator, experiment_name, re_filter=''):
    results_path = Path(RESULTS_PATH)
    glob_path = '*/' + integrator + '/' + experiment_name + '/*'
    experiments = sorted(results_path.glob(glob_path))
    experiment_parameters = list(set([experiment.name for experiment in experiments]))

    all_losses = {}
    parameters_strings = {}
    ctr = 0
    for parameters_string in experiment_parameters:
        if not re.search(re_filter, parameters_string):
            continue
        losses = calculate_metrics(experiment_name, integrator, parameters_string)
        if losses == {}:
            continue
        formated_ctr = '{:03d}'.format(ctr)
        parameters_strings[formated_ctr] = parameters_string
        all_losses[formated_ctr] = losses
        ctr += 1
    return all_losses, parameters_strings

def make_comparisons_methods():
    ppg_losses, ppg_images = calculate_all_with_images('ppg', 'comp-new-ppg')
    sdmm_losses, sdmm_images = calculate_all_with_images('sdmm', 'comp-360-spatial-48-directional-fixed-init')
    product_losses, product_images = calculate_all_with_images('sdmm', 'comp-product-no-jac')
    gt_images = get_gt_images()
    # print(ppg_losses)
    # print(sdmm_losses)

    image_directory = os.path.join(RESULTS_PATH, 'comparison_images')
    ppg_image_directory = os.path.join(image_directory, 'ppg')
    sdmm_image_directory = os.path.join(image_directory, 'sdmm')
    product_image_directory = os.path.join(image_directory, 'sdmm-prod')
    gt_image_directory = os.path.join(image_directory, 'gt')

    os.makedirs(image_directory, exist_ok=True)
    os.makedirs(ppg_image_directory, exist_ok=True)
    os.makedirs(sdmm_image_directory, exist_ok=True)
    os.makedirs(product_image_directory, exist_ok=True)
    os.makedirs(gt_image_directory, exist_ok=True)

    geometry_options = {"tmargin": "1cm", "lmargin": "10cm"}
    doc = Document(geometry_options=['a4paper', 'margin=0.2in'])
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('longtable'))
    doc.packages.append(Package('multirow'))
    doc.packages.append(Package('adjustbox'))
    doc.packages.append(Package('caption'))
    doc.packages.append(Package('xcolor', options='dvipsnames'))

    with doc.create(Section('Results', numbering=False)):
        doc.append(NoEscape(r'\definecolor{myblue}{HTML}{1F77B4}'))
        data_frame = pd.concat({
            'PPG': pd.DataFrame.from_dict(ppg_losses, 'columns'),
            'SDMM-radiance': pd.DataFrame.from_dict(sdmm_losses, 'columns'),
            'SDMM-product': pd.DataFrame.from_dict(product_losses, 'columns'),
        }, axis=0)
        data_frame = data_frame.sort_index(axis=1)

        def fmt(num):
            return f"{num:.3f}"
        str_data_frame = data_frame.applymap(fmt)
        for error in ['MAPE', 'MRSE']:
            sdmm_better = np.logical_and(
                data_frame.loc['SDMM-radiance', error] <= data_frame.loc['PPG', error],
                data_frame.loc['SDMM-radiance', error] <= data_frame.loc['SDMM-product', error]
            )
            ppg_better = np.logical_and(
                data_frame.loc['PPG', error] <= data_frame.loc['SDMM-radiance', error],
                data_frame.loc['PPG', error] <= data_frame.loc['SDMM-product', error]
            )
            product_better = np.logical_and(
                data_frame.loc['SDMM-product', error] <= data_frame.loc['PPG', error],
                data_frame.loc['SDMM-product', error] <= data_frame.loc['SDMM-radiance', error]
            )
            str_data_frame.loc['PPG', error][ppg_better] = r'\textbf{\color{myblue} ' + str_data_frame.loc['PPG', error][ppg_better] + '}'
            str_data_frame.loc['SDMM-radiance', error][sdmm_better] = r'\textbf{\color{myblue}' + str_data_frame.loc['SDMM-radiance', error][sdmm_better] + '}'
            str_data_frame.loc['SDMM-product', error][product_better] = r'\textbf{\color{myblue}' + str_data_frame.loc['SDMM-product', error][product_better] + '}'

        print(str_data_frame)
        with doc.create(Table(position='h!')) as table:
            doc.append(Command('centering'))
            doc.append(NoEscape(r'\begin{adjustbox}{width=1\textwidth}'))
            doc.append(NoEscape(str_data_frame.to_latex(sparsify=True, escape=False, bold_rows=True, multirow=True)))
            doc.append(NoEscape(r'\end{adjustbox}'))
            table.add_caption(NoEscape(
                r'A comparison of PPG \cite{Vorba:2019:PGP:3305366.3328091} '
                r'against two versions of our method, SDMM-radiance, '
                r'which uses only radiance guiding and $360$ spatial components, '
                r'and SDMM-product, which uses product guiding and $170$ spatial components. '
                r'The best result for each scene and error metric has been highlighted for visibility.'
            ))
            doc.append(NoEscape(r'\label{tab:ppg-comp}'))
        
        for counter, image_name in enumerate(sorted(sdmm_images.keys())):
            ppg_image = ppg_images[image_name]
            sdmm_image = sdmm_images[image_name]
            product_image = product_images[image_name]
            gt_image = gt_images[image_name]
            ppg_image_path = os.path.join(ppg_image_directory, image_name + '.jpg') 
            sdmm_image_path = os.path.join(sdmm_image_directory, image_name + '.jpg') 
            product_image_path = os.path.join(product_image_directory, image_name + '.jpg') 
            gt_image_path = os.path.join(gt_image_directory, image_name + '.jpg') 
            imsave(ppg_image_path, ppg_image)
            imsave(sdmm_image_path, sdmm_image)
            imsave(product_image_path, product_image)
            imsave(gt_image_path, gt_image)
            with doc.create(Figure(position='!p')) as comparison_figure:
                doc.append(Command('centering'))
                with doc.create(SubFigure(
                        width=NoEscape(r'0.24\linewidth'))) as subfigure:
                    doc.append(Command('centering'))
                    subfigure.append(StandAloneGraphic(
                        image_options=r'width=\textwidth, height=0.122\textheight, keepaspectratio',
                        filename=os.path.join('comparison_images', 'gt', image_name + '.jpg')
                    ))
                with doc.create(SubFigure(
                        width=NoEscape(r'0.24\linewidth'))) as subfigure:
                    doc.append(Command('centering'))
                    subfigure.append(StandAloneGraphic(
                        image_options=r'width=\textwidth, height=0.122\textheight, keepaspectratio',
                        filename=os.path.join('comparison_images', 'ppg', image_name + '.jpg')
                    ))
                with doc.create(SubFigure(
                        width=NoEscape(r'0.24\linewidth'))) as subfigure:
                    doc.append(Command('centering'))
                    subfigure.append(StandAloneGraphic(
                        image_options=r'width=\textwidth, height=0.122\textheight, keepaspectratio',
                        filename=os.path.join('comparison_images', 'sdmm', image_name + '.jpg')
                    ))
                with doc.create(SubFigure(
                        width=NoEscape(r'0.24\linewidth'))) as subfigure:
                    doc.append(Command('centering'))
                    subfigure.append(StandAloneGraphic(
                        image_options=r'width=\textwidth, height=0.122\textheight, keepaspectratio',
                        filename=os.path.join('comparison_images', 'sdmm-prod', image_name + '.jpg')
                    ))
                # comparison_figure.add_caption(image_name)
                doc.append(Command('ContinuedFloat'))
                if counter == len(sdmm_images) - 1:
                    comparison_figure.add_caption(NoEscape(
                        r'A visual comparison of methods from Table \ref{tab:ppg-comp}. '
                        r'From left to right: Ground Truth, PPG, SDMM-radiance, SDMM-product.'
                    ))
                    doc.append(NoEscape(r'\label{fig:ppg-comp}'))
    doc.generate_pdf(os.path.join(RESULTS_PATH, 'comparison'), clean=True, clean_tex=True, compiler='pdflatex')
    doc.generate_tex(os.path.join(RESULTS_PATH, 'comparison'))

def make_comparisons_experiments():
    experiment_name = 'comparison'
    all_losses, parameters_strings = calculate_all('sdmm', experiment_name)
    geometry_options = {"tmargin": "1cm", "lmargin": "10cm"}
    doc = Document(geometry_options=['a4paper', 'margin=0.2in'])
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('multirow'))

    with doc.create(Section(experiment_name, numbering=False)):
        for ctr, losses in all_losses.items():
            data_frame = pd.DataFrame.from_dict(losses)
            # data_frame = pd.concat({
            #     k: pd.DataFrame.from_dict(v, 'index') for k, v in all_losses.items()
            # }, axis=0).swaplevel()
            data_frame = data_frame.sort_index(axis=1)
            print(data_frame)
            with doc.create(Subsection(ctr, numbering=False)):
                doc.append(NoEscape(data_frame.to_latex(float_format="%.3f", bold_rows=True, multirow=True)))
                with doc.create(Description()) as desc:
                    # for idx, string in parameters_strings.items():
                    desc.add_item(ctr, NoEscape(', '.join(parameters_strings[ctr].split(','))))
    doc.generate_pdf(os.path.join(RESULTS_PATH, experiment_name), clean=True, clean_tex=True, compiler='pdflatex')

if __name__ == '__main__':
    make_comparisons_methods()
