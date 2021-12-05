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

(sec_introduction)=

# Introduction

This is the documentation for `tskit`, the tree sequence toolkit. Succinct tree sequences
provide a highly efficient way of storing a set of related DNA sequences by encoding
their ancestral history as a set of correlated trees along the genome.  The evolutionary
history of genetic sequences is often technically referred to as an Ancestral
Recombination Graph (ARG); succinct tree sequences are fully compatible with this
formulation, and tskit is a therefore a powerful platform for processing ARGs.

The tree sequence format is output by a number of external software libraries
and programs (such as [msprime](https://github.com/tskit-dev/msprime), 
[SLiM](https://github.com/MesserLab/SLiM), 
[fwdpp](http://molpopgen.github.io/fwdpp/), and 
[tsinfer](https://tsinfer.readthedocs.io/en/latest/)) that either simulate or
infer the evolutionary history of genetic sequences. This library provides the
underlying functionality that such software uses to load, examine, and
manipulate tree sequences, including efficient methods for calculating
{ref}`genetic statistics<sec_stats>`.

For a gentle introduction, you might like to read "{ref}`tutorials:sec_what_is`"
on our {ref}`tutorials site<tutorials:sec_intro>`. There you can also find further
tutorial material to introduce you to the key concepts behind succinct tree sequences.

:::{note}
This documentation is under active development and may be incomplete
in some areas. If you would like to help improve it, please open an issue or
pull request at [GitHub](https://github.com/tskit-dev/tskit).
:::
