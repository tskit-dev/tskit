# tskit  <img align="right" width="145" height="90" src="https://github.com/tskit-dev/administrative/blob/master/tskit_logo.svg">

[![PyPI version](https://img.shields.io/pypi/v/tskit.svg)](https://pypi.org/project/tskit/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/tskit.svg)](https://pypi.org/project/tskit/)
[![PyPI license](https://img.shields.io/pypi/l/tskit.svg)](https://pypi.python.org/pypi/tskit/)


[![CircleCI](https://circleci.com/gh/tskit-dev/tskit.svg?style=shield)](https://circleci.com/gh/tskit-dev/tskit)
[![Travis](https://img.shields.io/travis/tskit-dev/tskit)](https://travis-ci.org/github/tskit-dev/tskit)
[![Coverage](https://codecov.io/gh/tskit-dev/tskit/branch/master/graph/badge.svg)](https://codecov.io/gh/tskit-dev/tskit)
[![Docs](https://readthedocs.org/projects/tskit/badge/?version=stable&style=flat)](https://tskit.readthedocs.io/en/stable/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  

[![PyPI download total](https://img.shields.io/pypi/dm/tskit)](https://pypi.python.org/pypi/tskit/)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tskit-dev/tskit)](https://github.com/tskit-dev/tskit/commits/master)


Succinct tree sequences are a highly efficient way of storing a set of related DNA sequences by encoding their ancestral history as a set of correlated trees along the genome. The tree sequence format is output by a number of software libraries and programs (such as [msprime](https://github.com/tskit-dev/msprime), [SLiM](https://github.com/MesserLab/SLiM), [fwdpp](http://molpopgen.github.io/fwdpp/), and [tsinfer](https://tsinfer.readthedocs.io/en/latest/)) that either simulate or infer the evolutionary history of genetic sequences.

The `tskit` library provides the underlying functionality used to load, examine, and manipulate tree sequences. It often forms part of an installation of other software packages such as those listed above. Please see the [documentation](https://tskit.readthedocs.io/en/latest/) for further details, which includes [installation instructions](https://tskit.readthedocs.io/en/latest/installation.html).
