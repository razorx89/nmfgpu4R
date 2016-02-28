# nmfgpu4R
R binding for the [nmfgpu](https://github.com/razorx89/nmfgpu) library

## About

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
- Installation/Loading of the package fails if the nmfgpu library cannot be loaded
  - Do you have created a *NMFGPU_ROOT* environment variable?
  - Do you have all required dependencies installed? (CUDA, Visual Studio Runtime on Windows platforms, ...)
  - __Important__: Loading of the nmfgpu library can fail even if it can be found in the filesystem. Such an error can occur if one of the required dependencies cannot be loaded (e.g. CUDA runtime or cuBLAS from the CUDA Toolkit)
- Compilation step may be replaced in the future by an one-time download of pre-build binaries


## References
