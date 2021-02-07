import os, sys

BUILDDIR       = '#build/release'
DISTDIR        = '#dist'
CXX            = 'g++'
CC             = 'gcc'
# TODO: DISABLE SSE
CCFLAGS        = ['-DNDEBUG', '-march=native', '-mno-avx512f', '-O3', '-Wno-deprecated-declarations', '-g', '-DMTS_DEBUG', '-DSINGLE_PRECISION', '-DSPECTRUM_SAMPLES=3', '-DMTS_SSE', '-DMTS_HAS_COHERENT_RT', '-Wextra', '-std=gnu++17', '-fopenmp', '-ftree-vectorize', '-funsafe-math-optimizations', '-fno-rounding-math', '-fno-signaling-nans', '-fno-math-errno', '-fno-stack-protector', '-ffp-contract=fast', '-fomit-frame-pointer', '-fPIC']
# CXXFLAGS       = ['-O3', '-DNDEBUG', '-w', '-Wall', '-Wfloat-conversion', '-fPIC', '-g', '-pipe', '-march=native', '-mno-avx512f', '-msse2avx', '-mvzeroupper', '-mtune=intel', '-ftree-vectorize', '-funsafe-math-optimizations', '-fno-rounding-math', '-fno-signaling-nans', '-fno-math-errno', '-DMTS_DEBUG', '-DSINGLE_PRECISION', '-DSPECTRUM_SAMPLES=3', '-DMTS_SSE', '-DMTS_HAS_COHERENT_RT', '-fopenmp', '-fvisibility=hidden', '-mtls-dialect=gnu2', '-std=gnu++17', '-Wno-error=deprecated', '-fno-stack-protector', '-ffp-contract=fast', '-fomit-frame-pointer']
LINKFLAGS      = []
SHLINKFLAGS    = ['-rdynamic', '-shared', '-fPIC', '-lstdc++'] # '-lprofiler'
BASEINCLUDE    = ['#include']
BASELIB        = ['dl', 'm', 'pthread', 'gomp'] # 'profiler'
EIGENINCLUDE   = ['/usr/include/eigen3']
OEXRINCLUDE    = ['/usr/local/include/OpenEXR']
OEXRLIB        = ['Half-2_5', 'IlmImf-2_5', 'z']
PNGLIB         = ['png']
JPEGLIB        = ['jpeg']
XERCESINCLUDE  = []
XERCESLIB      = ['xerces-c']
GLLIB          = ['GL', 'GLU', 'GLEWmx', 'Xxf86vm', 'X11']
GLFLAGS        = ['-DGLEW_MX']
GLLIBDIR       = ['#dependencies/lib']
GLINCLUDE      = ['#dependencies/include']
BOOSTLIB       = ['boost_system', 'boost_filesystem', 'boost_thread', 'boost_iostreams', 'boost_serialization']
COLLADAINCLUDE = ['/usr/include/collada-dom2.4', '/usr/include/collada-dom2.4/1.4']
COLLADALIB     = ['collada-dom2.4-dp']
FFTWLIB        = ['fftw3_threads', 'fftw3']
SDMMINCLUDE    = ['#../sdmm/include', '#../sdmm/ext/enoki/include'] 
SDMMLIB        = ['fmt', 'spdlog'] 

# The following runs a helper script to search for installed Python
# packages that have a Boost Python library of matching version.
# A Mitsuba binding library will be compiled for each such pair.
# Alternatively, you could also specify the paths and libraries manually
# using the variables PYTHON27INCLUDE, PYTHON27LIB, PYTHON27LIBDIR etc.

import sys, os
sys.path.append(os.path.abspath('../data/scons'))
from detect_python import detect_python
locals().update(detect_python())
