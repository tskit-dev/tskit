.. currentmodule:: tskit
.. _sec_stats:

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
determined by the parameters (see :ref:`sec_stats_output_dimensions`).

Please see the :ref:`tutorial <sec_tutorial_stats>` for examples of the
statistics API in use.

.. warning:: :ref:`sec_data_model_missing_data` is not currently
   handled correctly by site statistics defined here, as we always
   assign missing data to be equal to the ancestral state. Later
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

- :meth:`TreeSequence.diversity`
- :meth:`TreeSequence.divergence`
- :meth:`TreeSequence.segregating_sites`
- :meth:`TreeSequence.allele_frequency_spectrum`
- :meth:`TreeSequence.genetic_relatedness`

------------------------
Patterson's f statistics
------------------------

These are the `f` statistics (also called `F` statistics) introduced in
`Reich et al (2009) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2842210/>`_.
See the documentation (link below) for the definition,
and `Peter (2016) <https://www.genetics.org/content/202/4/1485>`_ for readable
discussion of their use.

- :meth:`TreeSequence.f4`
- :meth:`TreeSequence.f3`
- :meth:`TreeSequence.f2`

------------
Y statistics
------------

These are the `Y` statistics introduced by
`Ashander et al (2018) <https://www.biorxiv.org/content/10.1101/354530v1>`_
as a three-sample intermediate between diversity/divergence (which are
pairwise) and Patterson's f statistics (which are four-way).

- :meth:`TreeSequence.Y3`
- :meth:`TreeSequence.Y2`

------------------
Trait correlations
------------------

These methods compute correlations and covariances of traits (i.e., an
arbitrary vector) with allelic state, possibly in the context of a multivariate
linear model with other covariates (as in GWAS).

- :meth:`TreeSequence.trait_covariance`
- :meth:`TreeSequence.trait_correlation`
- :meth:`TreeSequence.trait_linear_model`

------------------
Derived statistics
------------------

The other statistics above all have the property that `mode="branch"` and
`mode="site"` are "dual" in the sense that they are equal, on average, under
a high neutral mutation rate. The following statistics do not have this
property (since both are ratios of statistics that do have this property).

- :meth:`TreeSequence.Fst`
- :meth:`TreeSequence.Tajimas_D`

---------------
General methods
---------------

These methods allow access to the general method of computing statistics,
using weights or sample counts, and summary functions. See the documentation
for more details. The pre-implemented statistics above will be faster than
using these methods directly, so they should be preferred.

- :meth:`TreeSequence.general_stat`
- :meth:`TreeSequence.sample_count_stat`


.. _sec_stats_afs:

+++++++++++++++++++++++++
Allele frequency spectrum
+++++++++++++++++++++++++

We compute allele frequency spectra (AFS), including windowed and joint spectra,
using the same general pattern as other statistics,
but some of the details about how it is defined,
especially in the presence of multiple alleles per site, need to be explained.
If all sites are biallelic, then the result is just as you'd expect:
see the method documentation at :meth:`TreeSequence.allele_frequency_spectrum` 
for the description.
Note that with ``mode="site"``, we really do tabulate *allele* counts:
if more than one mutation on different parts of the tree produce the same allele,
it is the total number with this allele (i.e., inheriting *either* mutation)
that goes into the AFS.
The AFS with ``mode="branch"`` is the expected value for the Site AFS
with infinite-sites, biallelic mutation, so there is nothing surprising there,
either.

But, how do we deal with sites at which there are more than two alleles?
At each site, we iterate over the distinct alleles at that site,
and for each allele, count how many samples in each sample set
have inherited that allele.
For a concrete example, suppose that we are computing the AFS of a single
sample set with 10 samples, and are considering a site with three alleles:
*a*, *b*, and *c*,
which have been inherited by 6, 3, and 1 samples, respectively,
and that allele *a* is ancestral.
What we do at this site depends on if the AFS is polarised or not.

If we are computing the *polarised* AFS,
we add 1 to each entry of the output corresponding to each allele count
*except* the ancestral allele.
In our example, we'd add 1 to both ``AFS[3]`` and ``AFS[1]``.
This means that the sum of all entries of a polarised, site AFS
should equal the total number of non-ancestral alleles in the tree sequence
that are ancestral to at least one of the samples in the tree sequence.
The reason for this last caveat is that like with most statistics,
mutations that are not ancestral to *any* samples (not just those in the sample sets)
are not counted (and so don't even enter into ``AFS[0]``.

Now, if we are computing the *unpolarised* AFS,
we add *one half* to each entry of the *folded* output
corresponding to each allele count *including* the ancestral allele.
What does this mean?
Well, ``polarised=False`` means that we cannot distinguish between an
allele count of 6 and an allele count of 4.
So, *folding* means that we would add our allele that is seen in 6 samples
to ``AFS[4]`` instead of ``AFS[6]``.
So, in total, we will add 0.5 to each of ``AFS[4]``, ``AFS[3]``, and ``AFS[1]``.
This means that the sum of an unpolarised AFS
will be equal to the total number of alleles that are inherited
by any of the samples in the tree sequence, divided by two.
Why one-half? Well, notice that if in fact the mutation that produced the *b*
allele had instead produced an *a* allele,
so that the site had only two alleles, with frequencies 7 and 3.
Then, we would have added 0.5 to ``AFS[3]`` *twice*.


.. _sec_stats_interface:

*********
Interface
*********

Tskit offers a powerful and flexible interface for computing population genetic
statistics. Consequently, the interface is a little complicated and there are a
lot of options. However, we provide sensible defaults for these options and
``tskit`` should do what you want in most cases. There are several major options
shared by many statistics, which we describe in detail in the following subsections:

:ref:`sec_stats_mode`
    What are we summarising information about?

:ref:`sec_stats_windows`
    What section(s) of the genome are we interested in?

:ref:`sec_stats_span_normalise`
    Should the statistic calculated for each window be normalised by the span
    (i.e. the sequence length) of that window?

The statistics functions are highly efficient and are based where possible
on numpy arrays. Each of these statistics will return the results as a numpy
array, and the format of this array will depend on the statistic being
computed (see the :ref:`sec_stats_output_format` section for details).
A convenient feature of the statistics API is that the dimensions of the
output array is defined in a simple and intuitive manner by the
parameters provided. The :ref:`sec_stats_output_dimensions` section
defines the rules that are used.

Please see the :ref:`tutorial <sec_tutorial_stats>` for examples of the
statistics API in use.

.. _sec_stats_mode:

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

.. _sec_stats_sample_sets:

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


How multiple statistics are handled differs slightly between statistics
that operate on single sample sets and multiple sample sets.

.. _sec_stats_sample_sets_one_way:

---------------
One-way methods
---------------

One-way statistics such as :meth:`TreeSequence.diversity` are defined over a single
sample set. For these methods, ``sample_sets`` is interpreted in the following way:

- If it is a single list of node IDs (e.g., ``sample_sets=[0, 1 ,2]``), this is
  interpreted as running the calculation over one sample set and we remove
  the last dimension in the result array as described in the
  :ref:`sec_stats_output_dimensions` section.

- If it is ``None`` (the default), this is equivalent to ``sample_sets=ts.samples()``,
  and we therefore compute the statistic over all samples in the tree sequence. **Note
  that we also drop the outer dimension of the result array in this case**.

- If it is a list of lists of samples we return an array for each window in the output,
  which contains the value of the statistic separately for each of ``sample_sets``
  in the order they are given.

.. _sec_stats_sample_sets_multi_way:

-----------------
Multi-way methods
-----------------

Multi-way statistics such as :meth:`TreeSequence.divergence` are defined over a
``k`` sample sets.

In this case, ``sample_sets`` must be a list of lists of sample IDs, and there
is no default.

The ``indexes`` parameter is interpreted in the following way:

- If it is a single ``k``-tuple, this is interpreted as computing a single
  statistic selecting the specified sample sets and we remove the last dimension
  in the result array as described in the :ref:`sec_stats_output_dimensions` section.

- If if is ``None`` and ``sample_sets`` contains exactly ``k`` sample sets,
  this is equivalent to ``indexes=range(k)``. **Note
  that we also drop the outer dimension of the result array in this case**.

- If is is a list of ``k``-tuples (each consisting of integers
  of integers between ``0`` and ``len(sample_sets) - 1``) of length ``n`` we
  compute ``n`` statistics based on these selections of sample sets.


.. _sec_stats_windows:

+++++++
Windows
+++++++

Each statistic has an argument, ``windows``,
which defines a collection of contiguous windows spanning the genome.
``windows`` should be a list of ``n+1`` increasing numbers beginning with 0
and ending with the ``sequence_length``.
The statistic will be computed separately in each of the ``n`` windows,
and the ``k``-th row of the output will report the values of the statistic
in the ``k``-th window, i.e., from (and including) ``windows[k]`` to (but not including) ``windows[k+1]``.

Most windowed statistics by default return **averages** within each of the windows,
so the values are comparable between windows, even of different spans.
(However, shorter windows may be noisier.)
Suppose for instance  that you compute some statistic with ``windows = [0, a, b]``
for some valid positions ``0 < a < b``,
and get an output array ``S`` with two rows.
Then, computing the same statistic with ``windows = [0, b]``
would be equivalent to averaging the rows of ``S``,
obtaining ``((a - 0) * S[0] + (b - a) * S[1]) / (b - 0)``.

There are some shortcuts to other useful options:

``windows = None``
   This is the default and computes statistics in single window over the whole
   sequence. As the first returned array contains only a single
   value, we drop this dimension as described in the :ref:`output dimensions
   <sec_stats_output_dimensions>` section. **Note:** if you really do
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


.. _sec_stats_span_normalise:

++++++++++++++
Span normalise
++++++++++++++

In addition to windowing there is an option, ``span_normalise`` (which defaults to ``True``),
All the primary statistics defined here are *sums* across locations in the genome:
something is computed for each position, and these values are added up across all positions in each window.
Whether the total span of the window is then taken into account is determined by the option ``span_normalise``:
if it is ``True`` (the default), the sum for each window is converted into an *average*,
by dividing by the window's *span* (i.e. the length of genome that it covers).
Otherwise, the sum itself is returned.
The default is ``span_normalise=True``,
because this makes the values comparable across windows of different sizes.
To make this more concrete: :meth:`pairwise sequence divergence <.TreeSequence.divergence>`
between two samples with ``mode="site"`` is the density of sites that differ between the samples;
this is computed for each window by counting up the number of sites
at which the two differ, and dividing by the total span of the window.
If we wanted the number of sites at which the two differed in each window,
we'd calculate divergence with ``span_normalise=False``.

Following on from above, suppose we computed the statistic ``S`` with
``windows = [0, a, b]`` and ``span_normalise=True``,
and then computed ``T`` in just the same way except with ``span_normalize=False``.
Then ``S[0]`` would be equal to ``T[0] / a`` and ``S[1] = T[1] / (b - a)``.
Furthermore, the value obtained with ``windows = [0, b]`` would be equal to ``T[0] + T[1]``.
However, you probably usually want the (default) normalized version:
don't get unnormalised values unless you're sure that's what you want.
The exception is when computing a site statistic with ``windows = "sites"``:
this case, computes a statistic with the pattern of genotypes at each site,
and normalising would divide these statistics by the distance to the previous variant site
(probably not what you want to do).

And, a final note about "span": in tree sequences produced by ``msprime``
coordinates along the sequence are **continuous**,
so the "spans" used here may not correspond to distance along the genome in (say) base pairs.
For instance, pairwise sequence divergence is usually a number between 0 and 1
because it is the proportion of bases that differ;
this will only be true if the ``sequence_length``, and hence the "spans" are measured in base pairs
(which you ensure in ``msprime`` by setting recombination and mutation rates equal to the values
in units of crossovers and mutations per base pair, respectively).


.. _sec_stats_output_format:

+++++++++++++
Output format
+++++++++++++

Each of the statistics methods returns a ``numpy`` ndarray.
Suppose that the output is named ``out``.
If ``windows`` has been specified, the number of rows of the output is equal to the
number of windows, so that ``out.shape[0]`` is equal to ``len(windows) - 1``
and ``out[i]`` is an array of statistics describing the portion of the tree sequence
from ``windows[i]`` to ``windows[i + 1]`` (including the left but not the right endpoint).
What is returned within each window depends on the :ref:`mode <sec_stats_mode>`:

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
as described in the :ref:`sec_stats_output_dimensions` section.

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

.. _sec_stats_output_dimensions:

+++++++++++++++++
Output dimensions
+++++++++++++++++

In the general case, tskit outputs two dimensional (or three dimensional, in the case of node
stats) numpy arrays, as described in the :ref:`sec_stats_output_format` section.
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

.. _sec_stats_general_api:

***********
General API
***********

The methods :meth:`TreeSequence.general_stat` and :meth:`TreeSequence.sample_count_stat`
provide access to the general-purpose algorithm for computing statistics.
Here is a bit more discussion of how to use these.

.. _sec_stats_polarisation:

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

.. _sec_stats_summary_functions:

+++++++++++++++++
Summary functions
+++++++++++++++++

For convenience, here are the summary functions used for many of the statistics.
Below, :math:`x` denotes the number of samples in a sample set below a node,
`n` denotes the total size of a sample set, :math:`p = x / n`,
and boolean expressions (e.g., :math:`(x > 0)`) are interpreted as 0/1.

``diversity``
   :math:`f(x) = \frac{x (n - x)}{n (n-1)}`

   For an unpolarized statistic with biallelic loci, this calculates
   :math:`2 p (1-p)`.

``segregating_sites``
   :math:`f(x) =  (x > 0) (1 - x / n)`

   (Note: this works because if :math:`\sum_i p_1 = 1` then :math:`\sum_{i=1}^k (1-p_i) = k-1`.)

``Y1``
   :math:`f(x) = \frac{x (n - x) (n - x - 1)}{n (n-1) (n-2)}`

``divergence``
   :math:`f(x_1, x_2) = \frac{x_1 (n_2 - x_2)}{n_1 n_2}`,

   unless the two indices are the same, when the diversity function is used.

   For an unpolarized statistic with biallelic loci, this calculates
   :math:`p_1 (1-p_2) + (1 - p_1) p_2`.

``genetic_relatedness``
   :math:`f(x_i, x_j) = \frac{1}{2}(x_i - m)(x_j - m)`,

   where :math:`m = \frac{1}{n}\sum_{k=1}^n x_k` with :math:`n` the total number
   of samples.

``Y2``
   :math:`f(x_1, x_2) = \frac{x_1 (n_2 - x_2) (n_2 - x_2 - 1)}{n_1 n_2 (n_2 - 1)}`

``f2``
   :math:`f(x_1, x_2) = \frac{x_1 (x_1 - 1) (n_2 - x_2) (n_2 - x_2 - 1)}{n_1 (n_1 - 1) n_2 (n_2 - 1)} - \frac{x_1 (n_1 - x_1) (n_2 - x_2) x_2}{n_1 (n_1 - 1) n_2 (n_2 - 1)}`

   For an unpolarized statistic with biallelic loci, this calculates
   :math:`((p_1 - p_2)^2 - (p_1 (1-p_2)^2 + (1-p_1) p_2^2)/n_1 - (p_1^2 (1-p_2) + (1-p_1)^2 p_2)/n_2`
   :math:`+ (p_1 p_2 + (1-p_1)(1-p_2))/ n_1 n_2)(1 + \frac{1}{n_1 - 1})(1 + \frac{1}{n_2 - 1})`,
   which is the unbiased estimator for :math:`(p_1 - p_2)^2` from a finite sample.

``Y3``
   :math:`f(x_1, x_2, x_3) = \frac{x_1 (n_2 - x_2) (n_3 - x_3)}{n_1 n_2 n_3}`

``f3``
   :math:`f(x_1, x_2, x_3) = \frac{x_1 (x_1 - 1) (n_2 - x_2) (n_3 - x_3)}{n_1 (n_1 - 1) n_2 n_3} - \frac{x_1 (n_1 - x_1) (n_2 - x_2) x_3}{n_1 (n_1 - 1) n_2 n_3}`

   For an unpolarized statistic with biallelic loci, this calculates
   :math:`((p_1 - p_2)(p_1 - p_3) - p_1 (1-p_2)(1-p_3)/n_1 - (1-p_1) p_2 p_3/n_1)(1 + \frac{1}{n_1 - 1})`,
   which is the unbiased estimator for :math:`(p_1 - p_2)(p_1 - p_3)` from a finite sample.

``f4``
   :math:`f(x_1, x_2, x_3, x_4) = \frac{x_1 x_3 (n_2 - x_2) (n_4 - x_4)}{n_1 n_2 n_3 n_4} - \frac{x_1 x_4 (n_2 - x_2) (n_3 - x_3)}{n_1 n_2 n_3 n_4}`

   For an unpolarized statistic with biallelic loci, this calculates
   :math:`(p_1 - p_2)(p_3 - p_4)`.

``trait_covariance``
   :math:`f(w) = \frac{w^2}{2 (n-1)^2}`,

   where :math:`w` is the sum of all trait values of the samples below the node.

``trait_correlation``
   :math:`f(w, x) = \frac{w^2}{2 x (1 - x/n) (n - 1)}`,

   where as before :math:`x` is the total number of samples below the node,
   and :math:`n` is the total number of samples.

``trait_linear_model``
   :math:`f(w, z, x) = \frac{1}{2}\left( \frac{w - \sum_{j=1}^k z_j v_j}{x - \sum_{j=1}^k z_j^2} \right)^2`,

   where :math:`w` and :math:`x` are as before,
   :math:`z_j` is the sum of the j-th normalised covariate values below the node,
   and :math:`v_j` is the covariance of the trait with the j-th covariate.
