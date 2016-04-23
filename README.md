# nmfgpu4R
R binding for the [nmfgpu](https://github.com/razorx89/nmfgpu) library

## About
This package is a binding of the nmfgpu library for the R language. By default the package installs the relevant release of nmfgpu from github into the package directory.
If you want to use a custom compiled version, then define the `NMFGPU_ROOT` environment variable. Please visit the [nmfgpu](https://github.com/razorx89/nmfgpu) project page
or read the package documentation for further information.

## Citation
TBA

## Licence
This library is distributed under the terms of the *General Public Licence Version 3 (GPLv3)*.

![GPLv3 Logo](http://www.gnu.org/graphics/gplv3-127x51.png "GPLv3 Logo")

## Installation

### Prerequisites
- Compiled version of [nmfgpu](https://github.com/razorx89/nmfgpu)
- CUDA Toolkit (version depends on nmfgpu compilation)
- CUDA capable computation device (minimum: Kepler 3.0)
- *Windows only*: If you build from source, then you have to install [RTools](https://cran.rstudio.com/bin/windows/Rtools/)

### Instructions
The latest stable version of `nmfgpu4R` can be installed from CRAN using
```
install.packages("nmfgpu4R")
```

When using [devtools](https://cran.r-project.org/web/packages/devtools/index.html), the latest git version can be installed using:
```
install_github("razorx89/nmfgpu4R")
```

### Known issues
- Currently `nmfgpu` is only prebuild for Windows and Linux using the CUDA 7.5 toolkit. If you are using Mac OS X or CUDA 7.0 then you have to compile `nmfgpu` according to the installation instructions.
- Installation/Loading of the package fails if the nmfgpu library cannot be loaded
  - Do you have created a `NMFGPU_ROOT` environment variable?
  - Do you have all required dependencies installed? (CUDA, Visual Studio Runtime on Windows platforms, ...)
  - __Important__: Loading of the nmfgpu library can fail even if it can be found in the filesystem. Such an error can occur if one of the required dependencies cannot be loaded (e.g. CUDA runtime or cuBLAS from the CUDA Toolkit)
