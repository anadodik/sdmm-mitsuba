BUILDDIR       = '#build/release'
DISTDIR        = '#Mitsuba.app'
CXX            = 'clang++'
CC             = 'clang'
XCODE_SDK      = '/Applications/Xcode_12.0.0.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'
CCFLAGS        = ['-mmacosx-version-min=12.00', '-DNDEBUG', '-march=native', '-isysroot', XCODE_SDK, '-O3', '-Wno-deprecated-declarations', '-g', '-DMTS_DEBUG', '-DSINGLE_PRECISION', '-DSPECTRUM_SAMPLES=3', '-DMTS_SSE', '-DMTS_HAS_COHERENT_RT', '-Wextra', '-stdlib=libc++']
LINKFLAGS      = ['-framework', 'OpenGL', '-framework', 'Cocoa', '-mmacosx-version-min=12.00', '-Wl,-syslibroot', XCODE_SDK, '-Wl,-headerpad,128', '-stdlib=libc++']
CXXFLAGS       = ['-std=c++17']
BASEINCLUDE    = ['#include', '#dependencies/include']
BASELIBDIR     = ['#dependencies/lib']
BASELIB        = ['m', 'pthread']
JPEGLIB        = ['jpeg']
JPEGINCLUDE    = ['#dependencies/include/libjpeg']
PNGLIBDIR      = ['#dependencies/lib']
PNGLIB         = ['libpng.dylib']
XERCESLIB      = ['xerces-c']
GLLIB          = ['GLEWmx', 'objc']
GLFLAGS        = ['-DGLEW_MX']
# COLLADAINCLUDE = ['#dependencies/include/collada-dom', '#dependencies/include/collada-dom/1.4']
# COLLADALIB     = ['collada14dom24']
QTDIR          = '#dependencies'
FFTWLIB        = ['fftw3']
SDMMINCLUDE    = ['#../sdmm/include', '#../sdmm/ext/enoki/include']
SDMMLIB        = []
