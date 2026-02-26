# tskit  <img align="right" width="145" height="90" src="https://github.com/tskit-dev/administrative/blob/main/tskit_logo.svg">

[![License](https://img.shields.io/github/license/tskit-dev/tskit)](https://github.com/tskit-dev/tskit/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors/tskit-dev/tskit)](https://github.com/tskit-dev/tskit/graphs/contributors)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tskit-dev/tskit)](https://github.com/tskit-dev/tskit/commits/main)
[![Coverage](https://codecov.io/gh/tskit-dev/tskit/branch/main/graph/badge.svg)](https://codecov.io/gh/tskit-dev/tskit)
![OS](https://img.shields.io/badge/OS-linux%20%7C%20OSX%20%7C%20win--64-steelblue)

[Documentation (stable)](https://tskit.dev/tskit/docs/stable/) • [Documentation (latest)](https://tskit.dev/tskit/docs/latest/)

[![Docs Build](https://github.com/tskit-dev/tskit/actions/workflows/docs.yml/badge.svg)](https://github.com/tskit-dev/tskit/actions/workflows/docs.yml)[![Tests](https://github.com/tskit-dev/tskit/actions/workflows/tests.yml/badge.svg)](https://github.com/tskit-dev/tskit/actions/workflows/tests.yml)


The succinct tree sequence (`tskit`) format is an efficient way of representing
the genetic history - sometimes known as an
[Ancestral Recombination Graph or ARG](https://doi.org/10.1093/genetics/iyae100) -
of a set of related DNA sequences. `Tskit` is used
by a number of software libraries and programs (such as
[msprime](https://github.com/tskit-dev/msprime),
[SLiM](https://github.com/MesserLab/SLiM),
[fwdpp](http://molpopgen.github.io/fwdpp/), and
[tsinfer](https://tskit.dev/tsinfer/docs/stable/)) that either simulate or infer
the evolutionary ancestry of genetic sequences.

The `tskit` library provides the underlying functionality used to load, examine, and
manipulate ARGs in the tree sequence format, including efficient access to the
sequence of correlated trees along a genome and general methods to calculate
genetic statistics. `Tskit` often forms part of an installation of other
software packages such as those listed above. Please see the
[documentation](https://tskit.dev/tskit/docs/stable/) for further details, which
includes
[installation instructions](https://tskit.dev/tskit/docs/stable/installation.html).

To get started with tskit, tutorials and other content are at http://tskit.dev. For help
and support from the community you can use
[discussions](https://github.com/tskit-dev/tskit/discussions) here on github, or raise an
issue for a specific bug or feature request.

We warmly welcome contributions from the community. Raise an issue if you have an
idea you'd like to work on, or submit a PR for comments and help.

The base `tskit` library provides both a [Python](https://tskit.dev/tskit/docs/stable/python-api.html)
and [C](https://tskit.dev/tskit/docs/stable/c-api.html) API. A Rust API is provided in the
[tskit-rust](https://github.com/tskit-dev/tskit-rust) repository.


#### Python API
[![PyPI version](https://img.shields.io/pypi/v/tskit.svg)](https://pypi.org/project/tskit/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/tskit.svg)](https://pypi.org/project/tskit/)
[![Wheel](https://img.shields.io/pypi/wheel/tskit)](https://pypi.org/project/tskit/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Most users of `tskit` will use the python API as it provides a convenient, high-level API
to access, analyse and create tree sequences. Full documentation is
[here](https://tskit.dev/tskit/docs/stable/python-api.html).   

#### C API
[![C99](https://img.shields.io/badge/Language-C99-steelblue.svg)](https://en.wikipedia.org/wiki/C99)


The `tskit` C API provides comprehensive, low-level methods for manipulating and
processing tree-sequences. Written to the C99 standard and fully thread-safe, it can be
used with either C or C++. Full documentation is
[here](https://tskit.dev/tskit/docs/stable/c-api.html).

## Installation

```bash
python -m pip install tskit
# or
conda install -c conda-forge tskit
```
