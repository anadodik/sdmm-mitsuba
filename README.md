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

This library depends on other projects, and they have to be available first. You will need to install some SDMM specific dependencies using conan.io. Make sure conan is installed. Make sure `settings.cppstd=17` in your conan profile (`conan profile update settings.cppstd=17 default`). If on Linux, also make sure that you've set your conan profile to include `compiler.libcxx=libcstd++11`.
```
cd mitsuba/build
mkdir conan-out
cd conan-out
conan install ..
```
In case there is no pre-built binary in the conan repositories for the dependency `dep`, use `conan install --build=dep ..`.

## Authors

**Copyright (c) 2020 by Ana Dodik.**

## License

This project is licensed under the MIT Licence - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

I would like to thank my advisors, Thomas Müller, Cengiz Öztireli, and Marios Papas, whose insightful comments and ideas significantly improved the quality of the work.
