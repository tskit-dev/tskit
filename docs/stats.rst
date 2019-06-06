.. _sec_general_stats:

############################
General, windowed statistics
############################

There is a unified interface for computing many types of summary statistics from tree sequences.
These are implemented in a flexible way that
-- like the tree sequence itself --
exploits the duality between mutations and branches in the trees
to compute statistics from both genome sequence
(whose definition does not depend on the trees)
and from the underlying trees (whose definition does not depend on the genome sequence).
Furthermore, these statistics have a common interface to easily compute
(a) averaged statistics in windows along the genome,
and (b) summary statistics of many combinations of sets of samples simultaneously.
All statistics return a two-dimensional numpy array,
whose rows correspond to the windows along the genome,
and whose columns are determined by the statistic.

.. _sec_general_stats_type:

**************
Statistic type
**************

There are three types of statistic: ``site``, ``branch``, and ``node``,
that each summarize aspects of the tree sequence in different but related ways.
Roughly speaking, these answer the following sorts of question:

site
   How many mutations differentiate these two genomes?

branch
   How long since these genomes' common ancestor?

node
   On how much of the genome is each nodes an ancestors of only one of these genomes, but not both?

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
for each collection of nodes (described more below),
the ``node`` statistics return one number **for each node** (and for each window).
There can be a lot of nodes in the tree sequence, so beware.

.. _sec_general_stats_windowing:

*********
Windowing
*********

Each statistic has an argument, ``windows``,
which defines a collection of contiguous windows along the genome.
If ``windows`` is a list of ``n+1`` increasing numbers between 0 and the ``sequence_length``,
then the statistic will be computed separately in each of the ``n`` windows,
and the ``k``-th row of the output will report the values of the statistic
in the ``k``-th window, i.e., between ``windows[k]`` and ``windows[k+1]``.

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

``windows = None``
   This is the default, and equivalent to passing ``windows = [0.0, ts.sequence_length]``.
   The output will still be a two-dimensional array, but with only one row.

``windows = "treewise"``
   This says that you want statistics computed separately on the portion of the genome
   spanned by each tree, so is equivalent to passing ``windows = ts.breakpoints()``.
   (Beware: there can be a lot of trees!)

``windows = "sitewise"``
   This says to output one set of values for **each site**.
   This is windowing option does *not* return an average across some region
   (because sites occupy single points, not regions).

Furthermore, there is an option, ``unnormalized``,
that returns the **sum** of the relevant statistic across each window rather than the average.
The statistic that is returned by default is an average because we divide by
rather than normalizing (i.e., dividing) by the length of the window.
As above, if the statistic ``S`` was computed with ``unnormalized=True``,
then the value obtained with ``windows = [a, c]`` would be equal to ``S[0] + S[1]``.
However, you probably want the (default) normalized version:
don't get unnormalized values unless you're sure that's what you want.

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
per unit of genome length; but if we set ``unnormalized=True``
then we'd just obtain the number of differing sites per window.

And, a final note about "length": in tree sequences produced by ``msprime``
coordinates along the sequence are **continuous**,
so the the "lengths" used here may not correspond to distance along the genome in (say) base pairs.
For instance, pairwise sequence divergence is usually a number between 0 and 1
because it is the proportion of bases that differ;
this will only be true if length is measured in base pairs
(which you ensure in ``msprime`` by setting recombination and mutation rates equal to the values
in units of crossovers and mutations per base pair, respectively).


.. _sec_general_stats_sample_sets:

***********************
Sample sets and indices
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
under the hood we (essentially) compute the divergency between sample sets ``i`` and ``j``
by averaging ``p[i] * (1 - p[j]) + (1 - p[i]) * p[j]`` across the genome.

So, what if you
have samples from each of 10 populations,
and want to compute **all** fourty-five pairwise divergences?
You could call ``divergence`` fourty-five times, but this would be tedious
and would be inefficient, because the allele frequencies for one population
gets used in computing many of those values.
So, statistics that take a ``sample_sets`` argument also take an ``indices`` argument,
which for a statistic that operates on ``k`` sample sets will be a list of ``k``-tuples.
If ``indices`` is a length ``n`` list of ``k``-tuples,
then the output will have ``n`` columns,
and the ``j``-th column will contain values of the statistic computed on
``(sample_sets[indices[j][0]], sample_sets[indices[j][1]], ..., sample_sets[indices[j][k]])``.

To recap: ``indices`` must be a list of tuples, each of length ``k``,
of integers between ``0`` and ``len(sample-sets)``.
The appropriate value of ``k`` depends on the statistic.

Here are some additional special cases:

``indices = None``
   If the statistic takes ``k`` inputs for ``k > 1``,
   and there are exactly ``k`` lists in ``sample_sets``,
   then this will compute just one statistic, and is equivalent to passing
   ``indices = [(0, 1, ..., k-1)]``.
   If there are not exactly ``k`` sample sets, this will throw an error.

``k=1`` does not allow ``indices``:
   Statistics that operate on one sample set at a time (i.e., ``k=1``)
   do **not** take the ``indexes`` argument,
   and instead just return the value of the statistic separately for each of ``sample_sets``
   in the order they are given.
   (This would be equivalent to passing ``indices = [[0], [1], ..., [len(sample_sets)]]``,
   were that allowed.)

``stat_type = "node"`` does not allow ``indices``:
   Since node statistics output one value per node (unlike the other types, which output
   something summed across all nodes), it is an error to specify ``indices`` when computing
   a node statistic (consequently, you need to have exactly ``k`` sample sets).

.. Commenting these out for now as they are duplicates of the methods in the TreeSequence
   and sphinx is unhappy.

.. ********************
.. Statistics functions
.. ********************

.. .. autofunction:: tskit.TreeSequence.diversity

.. .. autofunction:: tskit.TreeSequence.divergence

.. .. autofunction:: tskit.TreeSequence.f4

.. .. autofunction:: tskit.TreeSequence.f3

.. .. autofunction:: tskit.TreeSequence.f2

.. .. autofunction:: tskit.TreeSequence.Y3

.. .. autofunction:: tskit.TreeSequence.Y2
