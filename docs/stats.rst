.. _sec_general_stats:

############################
General, windowed statistics
############################

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
All methods return numpy arrays,
whose rows correspond to the windows along the genome,
and whose remaining dimensions are determined by the statistic.

.. warning:: :ref:`sec_data_model_missing_data` is not currently
   handled correctly by site statistics defined here, as we always
   impute missing data to be equal to the ancestral state. Later
   versions will add this behaviour as an option and will account
   for the presence of missing data by default.


.. _sec_general_stats_type:

**************
Statistic mode
**************

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

*********
Windowing
*********

By default, statistics

``windows = None``
   This is the default, and equivalent to passing ``windows = [0.0, ts.sequence_length]``.
   The output will still be a two-dimensional array, but with only one row.

Each statistic has an argument, ``windows``,
which defines a collection of contiguous windows along the genome.
If ``windows`` is a list of ``n+1`` increasing numbers between 0 and the ``sequence_length``,
then the statistic will be computed separately in each of the ``n`` windows,
and the ``k``-th row of the output will report the values of the statistic
in the ``k``-th window, i.e., from (and including) ``windows[k]`` to (but not including) ``windows[k+1]``.

All windowed statistics by default return **averages** within each of the windows,
so the values are comparable between windows, even of different lengths.
(However, shorter windows may be noisier.)
Suppose for instance  that you compute some statistic with ``windows = [a, b, c]``
for some valid positions ``a < b < c``,
and get an output array ``S`` with two rows.
Then, computing the same statistic with ``windows = [a, c]``
would be equivalent to averaging the rows of ``S``,
obtaining ``((b - a) * S[0] + (c - b) * S[1]) / (c - a)``.

There are some shortcuts to other useful options:

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

Furthermore, there is an option, ``span_normalise`` (default ``True``),
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

***********************
Sample sets and indexes
***********************

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
   ``indexes = [(0, 1, ..., k-1)]``.
   If there are not exactly ``k`` sample sets, this will throw an error.

``k=1`` does not allow ``indexes``:
   Statistics that operate on one sample set at a time (i.e., ``k=1``)
   do **not** take the ``indexes`` argument,
   and instead just return the value of the statistic separately for each of ``sample_sets``
   in the order they are given.
   (This would be equivalent to passing ``indexes = [[0], [1], ..., [len(sample_sets)]]``,
   were that allowed.)


.. _sec_general_stats_output:

******
Output
******

Each of the statistics methods returns a ``numpy`` ndarray.
Suppose that the output is name ``out``.
In all cases, the number of rows of the output is equal to the number of windows,
so that ``out.shape[0]`` is equal to ``len(windows) - 1``
and ``out[i]`` is an array of statistics describing the portion of the tree sequence
from ``windows[i]`` to ``windows[i + 1]`` (including the left but not the right endpoint).

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

---------------
General methods
---------------

These methods allow access to the general method of computing statistics,
using weights or sample counts, and summary functions. See the documentation
for more details. The pre-implemented statistics above will be faster than
using these methods directly, so they should be preferred.

- :meth:`.TreeSequence.general_stat`
- :meth:`.TreeSequence.sample_count_stat`

------------------
Derived statistics
------------------

The other statistics above all have the property that `mode="branch"` and
`mode="site"` are "dual" in the sense that they are equal, on average, under
a high neutral mutation rate. The following statistics do not have this
property (since both are ratios of statistics that do have this property).

- :meth:`.TreeSequence.Fst`
- :meth:`.TreeSequence.TajimasD`

