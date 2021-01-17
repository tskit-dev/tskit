.. _sec_introduction:

============
Introduction
============

This is the documentation for tskit, the tree sequence toolkit. Tree sequences
are a highly efficient way of storing a set of related DNA sequences by encoding
their ancestral history as a set of correlated trees along the genome.

The tree sequence format is output by a number of external software libraries
and programs (such as `msprime <https://github.com/tskit-dev/msprime>`_, 
`SLiM <https://github.com/MesserLab/SLiM>`_, 
`fwdpp <http://molpopgen.github.io/fwdpp/>`_, and 
`tsinfer <https://tsinfer.readthedocs.io/en/latest/>`_) that either simulate or
infer the evolutionary history of genetic sequences. This library provides the
underlying functionality that such software uses to load, examine, and
manipulate tree sequences, including efficient methods for calculating
:ref:`genetic statistics<sec_stats>`.

.. note:: This documentation is under active development and may be incomplete
    in some areas. If you would like to help improve it, please open an issue or
    pull request at `GitHub <https://github.com/tskit-dev/tskit>`_.
