import sys, os, subprocess, shutil
import math
import numpy as np
import scipy.misc
from scipy.ndimage.filters import convolve1d
from PIL import Image

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MITSUBA_PATH = os.path.join(ROOT_PATH, 'mitsuba')
SCENES_PATH = os.path.join(ROOT_PATH, 'test-suite', 'scenes')
RESULTS_PATH = os.path.join(ROOT_PATH, 'test-suite', 'results')

SCENES = [
    'cornell-box',
    'torus',
    'glossy-cbox',
    'bookshelf', # NEW
    'pool',
    'bathroom2',
    'bedroom',
    'glossy-kitchen', # NEW
    'veach-door', # NEW
    'water-caustic', # NEW
    'hairball',
    'necklace', # NEW
    'glossy-bathroom2', # NEW
    # 'living-room-2',
    # 'cbox',
    # 'cbox-white',
    # 'cornell-box-big',
    # 'glass-of-water',
    # 'pool-rerendering',
    # 'lucy',
    # 'lucy-flipped',
    # 'cornell-box-motion',
    # 'cornell-box-moving-emitter',
    # 'floor',
    # 'pool-diffuse',
    # 'floor-constant-sky',
    # 'floor-zoom-out',
]

SCENE_TITLES = {
    # 'cbox',
    'cornell-box': 'Cornell Box',
    # 'cornell-box-big',
    'bathroom2': 'Salle de Bain',
    'living-room-2': 'White Room',
    'bedroom': 'Bedroom',
    # 'glass-of-water': 'Glass of Water',
    'pool': 'Swimming Pool',
    # 'pool-rerendering',
    'torus': 'Torus',
    'hairball': 'Hairball',
    'glossy-cbox': 'Glossy Cornell Box',
    # 'lucy',
    # 'lucy-flipped',
    # 'cornell-box-motion',
    # 'cornell-box-moving-emitter',
    # 'floor',
    # 'pool-diffuse',
    # 'floor-constant-sky',
    # 'floor-zoom-out',
}

def get_experiment_path(scene_name, experiment_name, integrator, parameters):
    parameters_string = ','.join("{!s}={!s}".format(key[0:8],val) for (key,val) in sorted(parameters.items()) if 'integrator' not in key)
    experiment_path = os.path.join(RESULTS_PATH, scene_name, integrator, experiment_name, parameters_string)
    return experiment_path

def get_experiment_path_param_string(scene_name, experiment_name, integrator, parameters_string):
    experiment_path = os.path.join(RESULTS_PATH, scene_name, integrator, experiment_name, parameters_string)
    return experiment_path

def get_scene_path(scene_name):
    return os.path.join(SCENES_PATH, scene_name, scene_name + '.xml')

def get_combined_path(experiment_path, scene_name, suffix=''):
    return os.path.join(experiment_path, scene_name + suffix + '.exr')

def get_combined_path_ppg(experiment_path, experiment_name):
    return os.path.join(experiment_path, experiment_name + '.exr')

def get_plots_path(experiment_path, scene_name):
    return os.path.join(experiment_path, scene_name + '.jpg')

def get_gt_path(scene_name):
    output_directory = os.path.join(RESULTS_PATH, scene_name, 'gt')
    output_path = os.path.join(output_directory, scene_name + '.exr')
    return output_path

def imsave(filename, img, gamma=2.2, divide_by=1.0, exposure=0.0):
    def tonemapped(image, gamma):
        return image**(1.0 / gamma)
    image = img / divide_by
    image = 2**exposure * image
    image = np.clip(image, 0.0, 1.0)
    image = tonemapped(image, gamma)
    filename = os.path.abspath(filename)
    # filename = os.path.splitext(filename)[0]
    # filename = filename.replace('.', '_')
    if os.path.splitext(filename)[-1].lower() in ['.jpg', '.jpeg']:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        image = (image * 255).astype('uint8')
        pil_img = Image.fromarray(image, 'RGB')
        pil_img.save(filename, format='JPEG', subsampling=0, quality=94)
    else:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(filename)

def aggregate(error, trim=None):
    if trim is None:
        return error.mean()
    error.flatten().sort()
    return float(error[trim:-trim].mean())

def luminance(a):
    a = np.maximum(0, a)**0.4545454545
    return 0.2126 * a[:,:,0] + 0.7152 * a[:,:,1] + 0.0722 * a[:,:,2]

def rgb_mean(a):
    return np.mean(a, axis=2)

def SSIM(a, b):
    def blur(a):
        k = np.array([0.120078, 0.233881, 0.292082, 0.233881, 0.120078])
        x = convolve1d(a, k, axis=0)
        return convolve1d(x, k, axis=1)
    a = luminance(a)
    b = luminance(b)
    mA = blur(a)
    mB = blur(b)
    sA = blur(a*a) - mA**2
    sB = blur(b*b) - mB**2
    sAB = blur(a*b) - mA*mB
    c1 = 0.01**2
    c2 = 0.03**2
    p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
    p2 = (2.0*sAB + c2)/(sA + sB + c2)
    error = p1 * p2
    return error

def MrSE(img, ref):
    return rgb_mean((img - ref)**2 / (1e-2 + ref**2))

def MAPE(img, ref):
    return rgb_mean(abs(img - ref) / (1e-2 + ref))

def SMAPE(img, ref):
    return rgb_mean(2 * abs(img - ref) / (1e-2 + ref + img))

def L2(img, ref):
    return rgb_mean((img - ref)**2)

def L1(img, ref):
    return rgb_mean(abs(img - ref))

def format_value(value, min_value):
    if value > min_value:
        return '%.4f' % value
    return '\\textbf{%.4f}' % value

def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)
