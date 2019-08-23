.. _sec_general_stats:

##########
Statistics
##########

There is a unified interface for computing many types of summary statistics from tree sequences.
These are implemented in a flexible way that
-- like the tree sequence itself --
exploits the duality between mutations and branches in the trees
to compute statistics of both genome sequence
(whose definition does not depend on the trees)
and of the underlying trees (whose definition does not depend on the genome sequence).
Furthermore, these statistics have a common interface to easily compute
(a) averaged statistics in windows along the genome,
and (b) summary statistics of many combinations of sets of samples simultaneously.
All methods return numpy arrays whose dimensions are
determined by the parameters (see :ref:`sec_general_stats_output_dimensions`).

.. warning:: :ref:`sec_data_model_missing_data` is not currently
   handled correctly by site statistics defined here, as we always
   impute missing data to be equal to the ancestral state. Later
   versions will add this behaviour as an option and will account
   for the presence of missing data by default.


********************
Available statistics
********************

Here are the statistics that can be computed using ``tskit``,
grouped by basic classification and type.

++++++++++++++++++++++
Single site statistics
++++++++++++++++++++++

- :meth:`.TreeSequence.allele_frequency_spectrum`
- :meth:`.TreeSequence.diversity`
- :meth:`.TreeSequence.divergence`
- :meth:`.TreeSequence.segregating_sites`

------------------------
Patterson's f statistics
------------------------

These are the `f` statistics (also called `F` statistics) introduced in
`Reich et al (2009) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2842210/>`_.
See the documentation (link below) for the definition,
and `Peter (2016) <https://www.genetics.org/content/202/4/1485>`_ for readable
discussion of their use.

- :meth:`.TreeSequence.f4`
- :meth:`.TreeSequence.f3`
- :meth:`.TreeSequence.f2`

------------
Y statistics
------------

These are the `Y` statistics introduced by
`Ashander et al (2018) <https://www.biorxiv.org/content/10.1101/354530v1>`_
as a three-sample intermediate between diversity/divergence (which are
pairwise) and Patterson's f statistics (which are four-way).

- :meth:`.TreeSequence.Y3`
- :meth:`.TreeSequence.Y2`

------------------
Trait correlations
------------------

These methods compute correlations and covariances of traits (i.e., an
arbitrary vector) with allelic state, possibly in the context of a multivariate
regression with other covariates (as in GWAS).

- :meth:`.TreeSequence.trait_covariance`
- :meth:`.TreeSequence.trait_correlation`

------------------
Derived statistics
------------------

The other statistics above all have the property that `mode="branch"` and
`mode="site"` are "dual" in the sense that they are equal, on average, under
a high neutral mutation rate. The following statistics do not have this
property (since both are ratios of statistics that do have this property).

- :meth:`.TreeSequence.Fst`
- :meth:`.TreeSequence.Tajimas_D`

---------------
General methods
---------------

These methods allow access to the general method of computing statistics,
using weights or sample counts, and summary functions. See the documentation
for more details. The pre-implemented statistics above will be faster than
using these methods directly, so they should be preferred.

- :meth:`.TreeSequence.general_stat`
- :meth:`.TreeSequence.sample_count_stat`


.. _sec_general_stats_interface:

*********
Interface
*********


.. _sec_general_stats_mode:

++++
Mode
++++

There are three **modes** of statistic: ``site``, ``branch``, and ``node``,
that each summarize aspects of the tree sequence in different but related ways.
Roughly speaking, these answer the following sorts of question:

site
   How many mutations differentiate these two genomes?

branch
   How long since these genomes' common ancestor?

node
   On how much of the genome is each node an ancestor of only one of these genomes, but not both?

These three examples can all be answered in the same way with the tree sequence:
first, draw all the paths from one genome to the other through the tree sequence
(back up to their common ancestor and back down in each marginal tree).
Then,
(``site``) count the number of mutations falling on the paths,
(``branch``) measure the length of the paths, or
(``node``) count how often the path goes through each node.
There is more discussion of this correspondence in the paper describing these statistics,
and precise definitions are given in each statistic.

One important thing to know is that ``node`` statistics have somewhat different output.
While ``site`` and ``branch`` statistics naturally return one number
for each portion of the genome (and thus incorporates information about many nodes: see below),
the ``node`` statistics return one number **for each node** in the tree sequence (and for each window).
There can be a lot of nodes in the tree sequence, so beware.

Also remember that in a tree sequence the "sites" are usually just the **variant** sites,
e.g., the sites of the SNPs.
(Although tree sequence may in principle have monomorphic sites, those produced by simulation usually don't.)

.. _sec_general_stats_windowing:

+++++++++
Windowing
+++++++++

Each statistic has an argument, ``windows``,
which defines a collection of contiguous windows spanning the genome.
``windows`` should be a list of ``n+1`` increasing numbers beginning with 0
and ending with the ``sequence_length``.
The statistic will be computed separately in each of the ``n`` windows,
and the ``k``-th row of the output will report the values of the statistic
in the ``k``-th window, i.e., from (and including) ``windows[k]`` to (but not including) ``windows[k+1]``.

Most windowed statistics by default return **averages** within each of the windows,
so the values are comparable between windows, even of different lengths.
(However, shorter windows may be noisier.)
Suppose for instance  that you compute some statistic with ``windows = [a, b, c]``
for some valid positions ``a < b < c``,
and get an output array ``S`` with two rows.
Then, computing the same statistic with ``windows = [a, c]``
would be equivalent to averaging the rows of ``S``,
obtaining ``((b - a) * S[0] + (c - b) * S[1]) / (c - a)``.

There are some shortcuts to other useful options:

``windows = None``
   This is the default and computes statistics in single window over the whole
   sequence. As the first returned array contains only a single
   value, we drop this dimension as described in the :ref:`output dimensions
   <sec_general_stats_output_dimensions>` section. **Note:** if you really do
   want to have an array with a single value as the result, please use
   ``windows = [0.0, ts.sequence_length]``.

``windows = "trees"``
   This says that you want statistics computed separately on the portion of the genome
   spanned by each tree, so is equivalent to passing ``windows = ts.breakpoints()``.
   (Beware: there can be a lot of trees!)

``windows = "sites"``
   This says to output one set of values for **each site**,
   and is equivalent to passing ``windows = [s.position for s in ts.sites()] + [ts.sequence_length]``.
   This will return one statistic for each site (beware!);
   since the windows are all different sizes you probably want to also pass
   ``span_normalise=False`` (see below).


.. _sec_general_stats_span_normalise:

+++++++++++++
Normalisation
+++++++++++++

In addition to windowing there is an option, ``span_normalise`` (default ``True``),
that if ``False`` returns the **sum** of the relevant statistic across each window rather than the average.
The statistic that is returned by default is an average because we divide by
rather than normalizing (i.e., dividing) by the length of the window.
As above, if the statistic ``S`` was computed with ``span_normalise=False``,
then the value obtained with ``windows = [a, c]`` would be equal to ``S[0] + S[1]``.
However, you probably usually want the (default) normalized version:
don't get unnormalised values unless you're sure that's what you want.
The exception is when computing a site statistic with ``windows = "sites"``:
this case, computes a statistic with the pattern of genotypes at each site,
and normalising would divide these statistics by the distance to the previous variant site
(probably not what you want to do).

To explain normalization a bit more:
a good way to think about these statistics in general
is that they all have a way of summarizing something **locally**,
i.e., at each point along the genome,
and this summary is then **averaged** across each window.
For instance, pairwise sequence divergence between two samples
is the density of sites that differ between them;
this is computed for each window by counting up the number of sites
at which the two differ, and dividing by the total length of the window.
Branch statistics do just the same thing,
except that we average over **all** locations on the sequence,
not just the locations of mutations.
So, usually "divergence" gives us the average number of differing sites
per unit of genome length; but if we set ``span_normalise=False``
then we'd just obtain the number of differing sites per window.

And, a final note about "length": in tree sequences produced by ``msprime``
coordinates along the sequence are **continuous**,
so the "lengths" used here may not correspond to distance along the genome in (say) base pairs.
For instance, pairwise sequence divergence is usually a number between 0 and 1
because it is the proportion of bases that differ;
this will only be true if length is measured in base pairs
(which you ensure in ``msprime`` by setting recombination and mutation rates equal to the values
in units of crossovers and mutations per base pair, respectively).


.. _sec_general_stats_sample_sets:

+++++++++++++++++++++++
Sample sets and indexes
+++++++++++++++++++++++

Many standard population genetics statistics
are defined with respect to some number of groups of genomes,
usually called "populations".
(However, it's clear from the correspondence to descriptors of tree shape
that the definitions can usefully describe *something*
even if the groups of samples don't come from "separate populations" in some sense.)
Basically, statistics defined in terms of sample sets can use the frequency of any allele
in each of the sample sets when computing the statistic.
For instance, nucleotide divergence is defined for a *pair* of groups of samples,
so if you wanted to compute pairwise divergences between some groups of samples,
you'd specify these as your ``sample_sets``.
Then, if ``p[i]`` is the derived allele frequency in sample set ``i``,
under the hood we (essentially) compute the divergence between sample sets ``i`` and ``j``
by averaging ``p[i] * (1 - p[j]) + (1 - p[i]) * p[j]`` across the genome.

So, what if you
have samples from each of 10 populations,
and want to compute **all** fourty-five pairwise divergences?
You could call ``divergence`` fourty-five times, but this would be tedious
and also inefficient, because the allele frequencies for one population
gets used in computing many of those values.
So, statistics that take a ``sample_sets`` argument also take an ``indexes`` argument,
which for a statistic that operates on ``k`` sample sets will be a list of ``k``-tuples.
If ``indexes`` is a length ``n`` list of ``k``-tuples,
then the output will have ``n`` columns,
and if ``indexes[j]`` is a tuple ``(i0, ..., ik)``,
then the ``j``-th column will contain values of the statistic computed on
``(sample_sets[i0], sample_sets[i1], ..., sample_sets[ik])``.

To recap: ``indexes`` must be a list of tuples, each of length ``k``,
of integers between ``0`` and ``len(sample_sets) - 1``.
The appropriate value of ``k`` depends on the statistic.

Here are some additional special cases:

``indexes = None``
   If the statistic takes ``k`` inputs for ``k > 1``,
   and there are exactly ``k`` lists in ``sample_sets``,
   then this will compute just one statistic, and is equivalent to passing
   ``indexes = (0, 1, ..., k-1)``. Note that this also drops the last
   dimension of the output, as described in the :ref:`sec_general_stats_output_dimensions`
   section.
   If there are not exactly ``k`` sample sets, this will throw an error.

``k=1`` does not allow ``indexes``:
   Statistics that operate on one sample set at a time (i.e., ``k=1``)
   do **not** take the ``indexes`` argument,
   and instead just return the value of the statistic separately for each of ``sample_sets``
   in the order they are given.
   (This would be equivalent to passing ``indexes = [[0], [1], ..., [len(sample_sets)]]``,
   were that allowed.)


.. _sec_general_stats_output_format:

+++++++++++++
Output format
+++++++++++++

Each of the statistics methods returns a ``numpy`` ndarray.
Suppose that the output is named ``out``.
If ``windows`` has been specified, the number of rows of the output is equal to the
number of windows, so that ``out.shape[0]`` is equal to ``len(windows) - 1``
and ``out[i]`` is an array of statistics describing the portion of the tree sequence
from ``windows[i]`` to ``windows[i + 1]`` (including the left but not the right endpoint).
What is returned within each window depends on the :ref:`mode <sec_general_stats_mode>`:

``mode="site"`` or ``mode="branch"``
   The output is a two-dimensional array,
   with columns corresponding to the different statistics computed: ``out[i, j]`` is the ``j``-th statistic
   in the ``i``-th window.

``mode="node"``
   The output is a three-dimensional array,
   with the second dimension corresponding to node id.
   In other words, ``out.shape[1]`` is equal to ``ts.num_nodes``,
   and ``out[i,j]`` is an array of statistics computed for node ``j`` on the ``i``-th window.

The final dimension of the arrays in other cases is specified by the method.

Note, however, that empty dimensions can optionally be dropped,
as described in the :ref:`sec_general_stats_output_dimensions` section.

A note about **default values** and **division by zero**:
Under the hood, statistics computation fills in zeros everywhere, then updates these
(since statistics are all **additive**, this makes sense).
But now suppose that you've got a statistic that returns ``nan``
("not a number") sometimes, like if you're taking the diversity of a sample set with only ``n=1`` sample,
which involves dividing by ``n * (n - 1)``.
Usually, you'll just get ``nan`` everywhere that the division by zero happens.
But there's a couple of caveats.
For ``site`` statistics, any windows without any sites in them never get touched,
so they will have a value of 0.
For ``branch`` statistics, any windows with **no branches** will similarly remain 0.
That said, you should **not** rely on the specific behavior of whether ``0`` or ``nan`` is returned
for "empty" cases like these: it is subject to change.

.. _sec_general_stats_output_dimensions:

+++++++++++++++++
Output dimensions
+++++++++++++++++

In the general case, tskit outputs two dimensional (or three dimensional, in the case of node
stats) numpy arrays, as described in the :ref:`sec_general_stats_output_format` section.
The first dimension corresponds to the window along the genome
such that for some result array ``x``, ``x[j]`` contains information about the jth window.
The last dimension corresponds to the statistics being computed, so that ``x[j, k]`` is the
value of the kth statistic in the jth window (in the two dimensional case). This is
a powerful and general interface, but in many cases we will not use this full generality
and the extra dimensions in the numpy arrays are inconvenient.

Tskit optionally removes empty dimensions from the output arrays following a few
simple rules.

1. If ``windows`` is None we are computing over the single window covering the
   full sequence. We therefore drop the first dimension of the array.

2. In one-way stats, if the ``sample_sets`` argument is a 1D array we interpret
   this as specifying a single sample set (and therefore a single statistic), and
   drop the last dimension of the output array. If ``sample_sets`` is None
   (the default), we use the sample set ``ts.samples()``, invoking
   this rule (we therefore drop the last dimension by default).

3. In k-way stats, if the ``indexes`` argument is a 1D array of length k
   we intepret this as specifying a single statistic and drop the last
   dimension of the array. If ``indexes`` is None (the default) and
   there are k sample sets, we compute the statistic from these sample sets
   and drop the last dimension.

Rules 2 and 3 can be summarised by "the dimensions of the input determines
the dimensions of the output". Note that dropping these dimensions is
**optional**: it is always possible to keep the full dimensions of the
output arrays.

Please see the :ref:`tutorial <sec_tutorial_stats>` for examples of the
various output dimension options.

.. _sec_general_stats_general_api:

***********
General API
***********

The methods :meth:`.TreeSequence.general_stat` and :meth:`.TreeSequence.sample_count_stat`
provide access to the general-purpose algorithm for computing statistics.
Here is a bit more discussion of how to use these.

.. _sec_general_stats_polarisation:

++++++++++++
Polarisation
++++++++++++

Many statistics calculated from genome sequence treat all alleles on equal footing,
as one must without knowledge of the ancestral state and sequence of mutations that produced the data.
Separating out the *ancestral* allele (e.g., as inferred using an outgroup)
is known as *polarisiation*.
For instance, in the allele frequency spectrum, a site with alleles at 20% and 80% frequency
is no different than another whose alleles are at 80% and 20%,
unless we know in each case which allele is ancestral,
and so while the unpolarised allele frequency spectrum gives the distribution of frequencies of *all* alleles,
the *polarised* allele frequency spectrum gives the distribution of frequencies of only *derived* alleles.

This concept is extended to more general statistics as follows.
For site statistics, summary functions are applied to the total weight or number of samples
associated with each allele; but if polarised, then the ancestral allele is left out of this sum.
For branch or node statistics, summary functions are applied to the total weight or number of samples
below, and above each branch or node; if polarised, then only the weight below is used.

.. _sec_general_stats_summary_functions:

+++++++++++++++++
Summary functions
+++++++++++++++++

For convenience, here are the summary functions used for many of the statistics.
Below, :math:`x` denotes the number of samples in a sample set below a node,
`n` denotes the total size of a sample set,
and boolean expressions (e.g., :math:`(x > 0)`) are interpreted as 0/1.

``diversity``
   :math:`f(x) = \frac{x (n - x)}{n (n-1)}`

``segregating_sites``
   :math:`f(x) =  (x > 0) (1 - x / n)`

   (Note: this works because if :math:`\sum_i p_1 = 1` then :math:`\sum_{i=1}^k (1-p_i) = k-1`.)

``Y1``
   :math:`f(x) = \frac{x (n - x) (n - x - 1)}{n (n-1) (n-2)}`

``divergence``
   :math:`f(x_1, x_2) = \frac{x_1 (n_2 - x_2)}{n_1 n_2}`,

   unless the two indices are the same, when the diversity function is used.

``Y2``
   :math:`f(x_1, x_2) = \frac{x_1 (n_2 - x_2) (n_2 - x_2 - 1)}{n_1 n_2 (n_2 - 1)}`

``f2``
   :math:`f(x_1, x_2) = \frac{x_1 (x_1 - 1) (n_2 - x_2) (n_2 - x_2 - 1)}{n_1 (n_1 - 1) n_2 (n_2 - 1)}`

``Y3``
   :math:`f(x_1, x_2, x_3) = \frac{x_1 (n_2 - x_2) (n_3 - x_3)}{n_1 n_2 n_3}`

``f3``
   :math:`f(x_1, x_2, x_3) = \frac{x_1 (x_1 - 1) (n_2 - x_2) (n_3 - x_3)}{n_1 (n_1 - 1) n_2 n_3}`

``f4``
   :math:`f(x_1, x_2, x_3, x_4) = \frac{x_1 x_3 (n_2 - x_2) (n_4 - x_4)}{n_1 n_2 n_3 n_4}`

``trait_covariance``
   :math:`f(w) = \frac{w^2}{2 (n-1)^2}`,

   where :math:`w` is the sum of all trait values of the samples below the node.

``trait_correlation``
   :math:`f(w, x) = \frac{w^2}{2 x (1 - x/n) (n - 1)}`,

   where as before :math:`x` is the total number of samples below the node,
   and :math:`n` is the total number of samples.

``trait_regression``
   :math:`f(w, z, x) = \frac{1}{2}\left( \frac{w - \sum_{j=1}^k z_j v_j}{x - \sum_{j=1}^k z_j^2} \right)^2`,

   where :math:`w` and :math:`x` are as before,
   :math:`z_j` is the sum of the j-th normalised covariate values below the node,
   and :math:`v_j` is the covariance of the trait with the j-th covariate.
