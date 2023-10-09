---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{currentmodule} tskit
```

(sec_installation)=


# Installation

There are two basic options for installing `tskit`: either through
pre-built binary packages using {ref}`sec_installation_conda` or
by compiling locally using {ref}`sec_installation_pip`. We recommend using `conda`
for most users, although `pip` can be more convenient in certain cases.
Tskit is installed to provide succinct tree sequence functionality
to other software (such as [msprime](https://github.com/tskit-dev/msprime)),
so it may already be installed if you use such software.

(sec_installation_requirements)=


## Requirements

Tskit requires Python 3.8+. There are no external C library dependencies. Python
dependencies are installed automatically by `pip` or `conda`.

(sec_installation_conda)=


## Conda

Pre-built binary packages for `tskit` are available through
[conda](https://conda.io/docs/), and built using [conda-forge](https://conda-forge.org/).
Packages for recent version of Python are available for Linux, OSX and Windows. Install
using:

```bash
$ conda install -c conda-forge tskit
```

### Quick Start

1. Install `conda` using [miniconda ](https://conda.io/miniconda.html).
   Make sure you follow the instructions to fully activate your `conda`
   installation!
2. Set up the [conda-forge channel ](https://conda-forge.org/) using
   `conda config --add channels conda-forge`.
3. Install tskit: `conda install tskit`.
4. Try it out: `tskit --version`.


There are several different ways to obtain `conda`. Please see the
[anaconda installation documentation](https://docs.anaconda.com/anaconda/install/)
for full details.

(sec_installation_pip)=


## Pip

Installing using `pip` is somewhat more flexible than `conda` and
may result in code that is (slightly) faster on your specific hardware.
`Pip` is the recommended method when using the system provided Python
installations. Installation is straightforward:

```bash
$ python3 -m pip install tskit
```

(sec_installation_development_versions)=


## Development versions

For general use, we do not recommend installing development versions.
Occasionally pre-release versions are made available, which can be
installed using `python3 -m pip install --pre tskit`. If you really need to install a
bleeding-edge version, see {ref}`sec_development_installing`.
