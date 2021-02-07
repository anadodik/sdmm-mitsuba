# SDMM Mitsuba

Mitsuba Implementation of SDMM Path Guiding.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
Clone the library, and pull the submodules. Make sure git-lfs is installed and pull the git lfs files.

```
git clone git@github.com:anadodik/sdmm-mitsuba.git
git submodule update --init --recursive
git lfs pull
```

### Prerequisites

This library depends on other projects, and they have to be available first. You will need to install some SDMM specific dependencies using `conan.io`. Make sure conan is installed. Make sure `settings.cppstd=17` in your conan profile (to create a profile, use `conan profile new default --detect`, and to update it use `conan profile update settings.cppstd=17 default`). If on Linux, also make sure that you've set your conan profile to include `compiler.libcxx=libcstd++11` by running `conan profile update settings.compiler.libcxx=libstdc++11 default`.
```
cd mitsuba/build
mkdir conan-out
cd conan-out
conan install ..
```
In case there is no pre-built binary in the conan repositories for the dependency `dep`, use `conan install --build=dep ..`.

Follow the platform-specific Mitsuba dependency installation instruction at https://www.mitsuba-renderer.org/releases/current/documentation.pdf. Note that conan.io already includes some dependencies so not all of them might be necessary. For example, on Fedora, you would need to use the command `sudo dnf install xerces-c-devel collada-dom-devel fftw3-devel libX11-devel libXxf86vm-devel libjpeg-devel`, together with the precompiled libraries in [this](https://github.com/mitsuba-renderer/dependencies_fedora) repository.

Lastly, to compile Mitsuba, make sure you have *scons for Python 3* installed (`pip3 install scons`), and symlink the correct configuration file to the mitsuba directory.

## Authors

**Copyright (c) 2020 by Ana Dodik.**

## License

This project is licensed under the MIT Licence - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

I would like to thank my advisors, Thomas Müller, Cengiz Öztireli, and Marios Papas, whose insightful comments and ideas significantly improved the quality of the work.
