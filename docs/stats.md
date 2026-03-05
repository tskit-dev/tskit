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


(sec_stats)=

# Statistics

The `tskit` library provides a large number of functions for calculating population
genetic statistics from tree sequences. Statistics can reflect either the distribution
of genetic variation or the underlying trees that generate it; the
[duality](https://doi.org/10.1534/genetics.120.303253) between
mutations and branch lengths on trees means that statistics based on genetic variation
often have an corresponding version based on branch lengths in trees in a tree sequence.

Note that `tskit` provides a unified interface for computing so-called
"single site statistics" that are summaries across sites in windows of the genome, as
well as a standard method for specifying "multi-way" statistics that are calculated
over many combinations of sets of samples simultaneously.

Please see the {ref}`tutorial <tutorials:sec_tutorial_stats>` for examples of the
statistics API in use.

:::{warning}
Site statistics defined here currently treat :ref:`missing data<sec_data_model_missing_data>`
in the same way as in earlier versions of tskit: when computing site-based
statistics, isolated samples without mutations directly above them are treated
as carrying the ancestral allele rather than as missing. Future versions of
tskit may expose options to treat missing data differently in statistics; for
now, if you need explicit control over how missing data is handled you should
use the low-level genotype/variant APIs (for example with
``isolated_as_missing=True``) together with your own summary logic.
:::

(sec_stats_available)=

## Available statistics

Here are the statistics that can be computed using `tskit`,
grouped by basic classification and type. Single-site statistics are ones that are
averages across sites in windows of the genome, returning numpy arrays whose
dimensions are determined by the parameters (see {ref}`sec_stats_output_dimensions`).

{ref}`sec_stats_sample_sets_one_way` are defined over a single sample set, 
whereas {ref}`sec_stats_sample_sets_multi_way` compare 2 or more sets of samples.

Some of the methods below benefit from a little extra discussion, provided in the
{ref}`sec_stats_notes` section at the end of this chapter: if so, a link to the note
appears beside the listed method.

* Single site
    * One-way
        * {meth}`~TreeSequence.allele_frequency_spectrum` (see {ref}`notes<sec_stats_notes_afs>`)
        * {meth}`~TreeSequence.diversity`
        * {meth}`~TreeSequence.segregating_sites`
        * {meth}`~TreeSequence.trait_covariance`
          {meth}`~TreeSequence.trait_correlation`
          {meth}`~TreeSequence.trait_linear_model`
          (see {ref}`sec_stats_notes_trait`)
        * {meth}`~TreeSequence.Tajimas_D` (see {ref}`notes<sec_stats_notes_derived>`)
    * Multi-way
        * {meth}`~TreeSequence.divergence`
        * {meth}`~TreeSequence.genetic_relatedness`
          {meth}`~TreeSequence.genetic_relatedness_weighted`
          {meth}`~TreeSequence.genetic_relatedness_vector`
          {meth}`~TreeSequence.genetic_relatedness_matrix`
        * {meth}`~TreeSequence.f4`
          {meth}`~TreeSequence.f3`
          {meth}`~TreeSequence.f2`
          (see {ref}`sec_stats_notes_f`)
        * {meth}`~TreeSequence.Y3`
          {meth}`~TreeSequence.Y2`
          (see {ref}`sec_stats_notes_y`)
        * {meth}`~TreeSequence.genealogical_nearest_neighbours` (see {ref}`sec_stats_notes_gnn`)
        * {meth}`~TreeSequence.Fst` (see {ref}`sec_stats_notes_derived`)
* Multi site
    * {meth}`~LdCalculator` (note this is soon to be deprecated)

:::{note}
There is a general framework provided for calculating additional single site
statistics (see the {ref}`sec_stats_general_api` section). However, the
pre-implemented statistics in the table above will be faster than rederiving
them using the general framework directly, so the versions above should be preferred.
:::


(sec_stats_examples)=

### Quick examples

```{code-cell} ipython3
:"tags": ["hide-input"]
from IPython.display import Markdown
import msprime
import numpy as np

demography = msprime.Demography()
demography.add_population(name="A", initial_size=10_000)
demography.add_population(name="B", initial_size=10_000)
demography.set_symmetric_migration_rate(["A", "B"], 0.001)
ts = msprime.sim_ancestry(
    samples={"A": 2, "B": 2},
    sequence_length=1000,
    demography=demography,
    recombination_rate=2e-8,
    random_seed=12)
ts = msprime.sim_mutations(ts, rate=2e-8, random_seed=12)
Markdown(
    f"These examples use a tree sequence of {ts.num_samples} samples "
    f"in {ts.num_populations} populations, "
    f"with a sequence length of {int(ts.sequence_length)}. "
    f"There are {ts.num_trees} trees and "
    f"{ts.num_sites} variable sites in the tree sequence."
)
```

#### Basic calling convention

```{code-cell} ipython3
pi = ts.diversity()
print(pi) # Genetic diversity within the sample set
```

#### Restrict to {ref}`sample sets<sec_stats_sample_sets>`

```{code-cell} ipython3
pi_0 = ts.diversity(sample_sets=ts.samples(population=0))
print(pi_0)  # Genetic diversity within population 0
```

#### Summarise in genomic {ref}`windows<sec_stats_windows>`

```{code-cell} ipython3
pi_window = ts.diversity(sample_sets=ts.samples(population=1), windows=[0, 400,  600, 1000])
print(pi_window)  # Genetic diversity within population 1 in three windows along the genome
```

#### Compare {ref}`between<sec_stats_sample_sets_multi_way>` sample sets

```{code-cell} ipython3
dxy = ts.divergence(sample_sets=[ts.samples(population=0), ts.samples(population=1)])
print(dxy)  # Av number of differences per bp between samples in population 0 and 1
```

#### Change the {ref}`mode<sec_stats_mode>`

```{code-cell} ipython3
bl = ts.divergence(
    mode="branch",  # Use branch lengths rather than genetic differences
    sample_sets=[ts.samples(population=0), ts.samples(population=1)],
)
print(bl)  # Av branch length separating samples in population 0 and 1
```

(sec_stats_single_site)=

## Single site statistics


(sec_stats_interface)=

### Interface

Tskit offers a powerful and flexible interface for computing population genetic
statistics. Consequently, the interface is a little complicated and there are a
lot of options. However, we provide sensible defaults for these options and
`tskit` should do what you want in most cases. There are several major options
shared by many statistics, which we describe in detail in the following subsections:

{ref}`sec_stats_mode`
: What are we summarising information about?

{ref}`sec_stats_windows`
: What section(s) of the genome are we interested in?

{ref}`sec_stats_span_normalise`
: Should the statistic calculated for each window be normalised by the span
  (i.e. the sequence length) of that window?

The statistics functions are highly efficient and are based where possible
on numpy arrays. Each of these statistics will return the results as a numpy
array, and the format of this array will depend on the statistic being
computed (see the {ref}`sec_stats_output_format` section for details).
A convenient feature of the statistics API is that the dimensions of the
output array is defined in a simple and intuitive manner by the
parameters provided. The {ref}`sec_stats_output_dimensions` section
defines the rules that are used.

Please see the {ref}`tutorial <sec_tutorial_stats>` for examples of the
statistics API in use.


(sec_stats_mode)=

#### Mode

There are three **modes** of statistic: `site`, `branch`, and `node`,
that each summarize aspects of the tree sequence in different but related ways.
Roughly speaking, these answer the following sorts of question:

site
: How many mutations differentiate these two genomes?

branch
: How long since these genomes' common ancestor?

node
: On how much of the genome is each node an ancestor of only one of these genomes, but not both?

These three examples can all be answered in the same way with the tree sequence:
first, draw all the paths from one genome to the other through the tree sequence
(back up to their common ancestor and back down in each marginal tree).
Then,
(`site`) count the number of mutations falling on the paths,
(`branch`) measure the length of the paths, or
(`node`) count how often the path goes through each node.
There is more discussion of this correspondence in the paper describing these statistics,
and precise definitions are given in each statistic.

Here's an example of using the {meth}`~TreeSequence.diversity` statistic to return the
average branch length between all pairs of samples:

```{code-cell} ipython3
ts.diversity(mode="branch")
```

One important thing to know is that `node` statistics have somewhat different output.
While `site` and `branch` statistics naturally return one number
for each portion of the genome (and thus incorporates information about many nodes: see below),
the `node` statistics return one number **for each node** in the tree sequence (and for each window).
There can be a lot of nodes in the tree sequence, so beware.

Also remember that in a tree sequence the "sites" are usually just the **variant** sites,
e.g., the sites of the SNPs. Although the tree sequence may in principle have monomorphic
sites, those produced by simulation usually don't.


(sec_stats_sample_sets)=

#### Sample sets and indexes

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
you'd specify these as your `sample_sets`.
Then, if `p[i]` is the derived allele frequency in sample set `i`,
under the hood we (essentially) compute the divergence between sample sets `i` and `j`
by averaging `p[i] * (1 - p[j]) + (1 - p[i]) * p[j]` across the genome.

Concretely, `sample_sets` specifies the IDs of the nodes to compute statistics of.
Importantly, these nodes must be {ref}`samples <sec_data_model_definitions_sample>`.

Here's an example of calculating the average
{meth}`genetic diversity<TreeSequence.diversity>` within a specific population:

```{code-cell} ipython3
ts.diversity(sample_sets=ts.samples(population=0))
```


So, what if you
have samples from each of 10 populations,
and want to compute **all** fourty-five pairwise divergences?
You could call `divergence` fourty-five times, but this would be tedious
and also inefficient, because the allele frequencies for one population
gets used in computing many of those values.
So, statistics that take a `sample_sets` argument also take an `indexes` argument,
which for a statistic that operates on `k` sample sets will be a list of `k`-tuples.
If `indexes` is a length `n` list of `k`-tuples,
then the output will have `n` columns,
and if `indexes[j]` is a tuple `(i0, ..., ik)`,
then the `j`-th column will contain values of the statistic computed on
`(sample_sets[i0], sample_sets[i1], ..., sample_sets[ik])`.


How multiple statistics are handled differs slightly between statistics
that operate on single sample sets and multiple sample sets.


(sec_stats_sample_sets_one_way)=

##### One-way methods

One-way statistics such as {meth}`TreeSequence.diversity` are defined over a single
sample set. For these methods, `sample_sets` is interpreted in the following way:

- If it is a single list of node IDs (e.g., `sample_sets=[0, 1 ,2]`), this is
  interpreted as running the calculation over one sample set and we remove
  the last dimension in the result array as described in the
  {ref}`sec_stats_output_dimensions` section.

- If it is `None` (the default), this is equivalent to `sample_sets=ts.samples()`,
  and we therefore compute the statistic over all samples in the tree sequence. **Note
  that we also drop the outer dimension of the result array in this case**.

- If it is a list of lists of samples we return an array for each window in the output,
  which contains the value of the statistic separately for each of `sample_sets`
  in the order they are given.


(sec_stats_sample_sets_multi_way)=

##### Multi-way methods

Multi-way statistics such as {meth}`TreeSequence.divergence` are defined over a
`k` sample sets. In this case, `sample_sets` must be a list of lists of sample IDs,
and there is no default. For example, this finds the average
{meth}`genetic divergence<TreeSequence.divergence>` between samples in populations
0 and 1

```{code-cell} ipython3
ts.divergence(
    sample_sets=[
        ts.samples(population=0),
        ts.samples(population=1),
    ]
)
```


The `indexes` parameter is interpreted in the following way:

- If it is a single `k`-tuple, this is interpreted as computing a single
  statistic selecting the specified sample sets and we remove the last dimension
  in the result array as described in the {ref}`sec_stats_output_dimensions` section.

- If it is `None` and `sample_sets` contains exactly `k` sample sets,
  this is equivalent to `indexes=range(k)`. **Note
  that we also drop the outer dimension of the result array in this case**.

- If it is a list of `k`-tuples (each consisting of integers
  between `0` and `len(sample_sets) - 1`) of length `n` we
  compute `n` statistics based on these selections of sample sets.


(sec_stats_windows)=

#### Windows

Each statistic has an argument, `windows`,
which defines a collection of contiguous windows spanning the genome.
`windows` should be a list of `n+1` increasing numbers beginning with 0
and ending with the `sequence_length`.
The statistic will be computed separately in each of the `n` windows,
and the `k`-th row of the output will report the values of the statistic
in the `k`-th window, i.e., from (and including) `windows[k]` to
(but not including) `windows[k+1]`. For example, this calculates
{meth}`Tajima's D<TreeSequence.Tajimas_D>` in four evenly spaced windows along the
genome:

```{code-cell} ipython3
num_windows = 4
ts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
```

Most windowed statistics by default return **averages** within each of the windows,
so the values are comparable between windows, even of different spans.
(However, shorter windows may be noisier.)
Suppose for instance  that you compute some statistic with `windows = [0, a, b]`
for some valid positions `0 < a < b`,
and get an output array `S` with two rows.
Then, computing the same statistic with `windows = [0, b]`
would be equivalent to averaging the rows of `S`,
obtaining `((a - 0) * S[0] + (b - a) * S[1]) / (b - 0)`.

There are some shortcuts to other useful options:

`windows = None`
   This is the default and computes statistics in single window over the whole
   sequence. As the first returned array contains only a single
   value, we drop this dimension as described in the
   {ref}`output dimensions <sec_stats_output_dimensions>` section. **Note:** if you
   really do want to have an array with a single value as the result, please use
   `windows = [0.0, ts.sequence_length]`.

`windows = "trees"`
   This says that you want statistics computed separately on the portion of the genome
   spanned by each tree, so is equivalent to passing `windows = ts.breakpoints()`.
   (Beware: there can be a lot of trees!)

`windows = "sites"`
   This says to output one set of values for **each site**,
   and is equivalent to passing `windows = [s.position for s in ts.sites()] + [ts.sequence_length]`.
   This will return one statistic for each site (beware!);
   since the windows are all different sizes you probably want to also pass
   `span_normalise=False` (see below).


(sec_stats_span_normalise)=

#### Span normalise

In addition to windowing there is an option, `span_normalise` (which defaults to `True`),
All the primary statistics defined here are *sums* across locations in the genome:
something is computed for each position, and these values are added up across all positions in each window.
Whether the total span of the window is then taken into account is determined by the option `span_normalise`:
if it is `True` (the default), the sum for each window is converted into an *average*,
by dividing by the window's *span* (i.e. the length of genome that it covers).
Otherwise, the sum itself is returned.
The default is `span_normalise=True`,
because this makes the values comparable across windows of different sizes.
To make this more concrete: {meth}`pairwise sequence divergence <.TreeSequence.divergence>`
between two samples with `mode="site"` is the density of sites that differ between the samples;
this is computed for each window by counting up the number of sites
at which the two differ, and dividing by the total span of the window.
If we wanted the number of sites at which the two differed in each window,
we'd calculate divergence with `span_normalise=False`.

Following on from above, suppose we computed the statistic `S` with
`windows = [0, a, b]` and `span_normalise=True`,
and then computed `T` in just the same way except with `span_normalise=False`.
Then `S[0]` would be equal to `T[0] / a` and `S[1] = T[1] / (b - a)`.
Furthermore, the value obtained with `windows = [0, b]` would be equal to `T[0] + T[1]`.
However, you probably usually want the (default) normalized version:
don't get unnormalised values unless you're sure that's what you want.
The exception is when computing a site statistic with `windows = "sites"`:
this case, computes a statistic with the pattern of genotypes at each site,
and normalising would divide these statistics by the distance to the previous variant site
(probably not what you want to do).

:::{note}
The resulting values are scaled "per unit of sequence length" - for instance, pairwise
sequence divergence is measured in "differences per unit of sequence length". Functions
such as {func}`msprime:msprime.sim_mutations` will by default add mutations in discrete
coordinates, usually interpreted as base pairs, in which
case span normalised statistics are in units of "per base pair".
:::

(sec_stats_output_format)=

#### Output format

Each of the statistics methods returns a `numpy` ndarray.
Suppose that the output is named `out`.
If `windows` has been specified, the number of rows of the output is equal to the
number of windows, so that `out.shape[0]` is equal to `len(windows) - 1`
and `out[i]` is an array of statistics describing the portion of the tree sequence
from `windows[i]` to `windows[i + 1]` (including the left but not the right endpoint).
What is returned within each window depends on the {ref}`mode <sec_stats_mode>`:

`mode="site"` or `mode="branch"`
   The output is a two-dimensional array,
   with columns corresponding to the different statistics computed: `out[i, j]` is the `j`-th statistic
   in the `i`-th window.

`mode="node"`
   The output is a three-dimensional array,
   with the second dimension corresponding to node id.
   In other words, `out.shape[1]` is equal to `ts.num_nodes`,
   and `out[i,j]` is an array of statistics computed for node `j` on the `i`-th window.

The final dimension of the arrays in other cases is specified by the method.

Note, however, that empty dimensions can optionally be dropped,
as described in the {ref}`sec_stats_output_dimensions` section.

A note about **default values** and **division by zero**:
Under the hood, statistics computation fills in zeros everywhere, then updates these
(since statistics are all **additive**, this makes sense).
But now suppose that you've got a statistic that returns `nan`
("not a number") sometimes, like if you're taking the diversity of a sample set with only `n=1` sample,
which involves dividing by `n * (n - 1)`.
Usually, you'll just get `nan` everywhere that the division by zero happens.
But there's a couple of caveats.
For `site` statistics, any windows without any sites in them never get touched,
so they will have a value of 0.
For `branch` statistics, any windows with **no branches** will similarly remain 0.
That said, you should **not** rely on the specific behavior of whether `0` or `nan` is returned
for "empty" cases like these: it is subject to change.


(sec_stats_output_dimensions)=

#### Output dimensions

In the general case, tskit outputs two dimensional (or three dimensional, in the case of node
stats) numpy arrays, as described in the {ref}`sec_stats_output_format` section.
The first dimension corresponds to the window along the genome
such that for some result array `x`, `x[j]` contains information about the jth window.
The last dimension corresponds to the statistics being computed, so that `x[j, k]` is the
value of the kth statistic in the jth window (in the two dimensional case). This is
a powerful and general interface, but in many cases we will not use this full generality
and the extra dimensions in the numpy arrays are inconvenient.

Tskit optionally removes empty dimensions from the output arrays following a few
simple rules.

1. If `windows` is None we are computing over the single window covering the
   full sequence. We therefore drop the first dimension of the array.

2. In one-way stats, if the `sample_sets` argument is a 1D array we interpret
   this as specifying a single sample set (and therefore a single statistic), and
   drop the last dimension of the output array. If `sample_sets` is None
   (the default), we use the sample set `ts.samples()`, invoking
   this rule (we therefore drop the last dimension by default).

3. In k-way stats, if the `indexes` argument is a 1D array of length k
   we intepret this as specifying a single statistic and drop the last
   dimension of the array. If `indexes` is None (the default) and
   there are k sample sets, we compute the statistic from these sample sets
   and drop the last dimension.

4. If, after dropping these dimensions, the dimension is 0, we return a numpy
   scalar (instead of an array of dimension 0).

Rules 2 and 3 can be summarised by "the dimensions of the input determines
the dimensions of the output". Note that dropping these dimensions is
**optional**: it is always possible to keep the full dimensions of the
output arrays.

Please see the {ref}`tutorial <sec_tutorial_stats>` for examples of the
various output dimension options.


(sec_stats_general_api)=

### General API

The methods {meth}`TreeSequence.general_stat` and {meth}`TreeSequence.sample_count_stat`
provide access to the general-purpose algorithm for computing statistics.
Here is a bit more discussion of how to use these.


(sec_stats_polarisation)=

#### Polarisation

Many statistics calculated from genome sequence treat all alleles on equal footing,
as one must without knowledge of the ancestral state and sequence of mutations that produced the data.
Separating out the *ancestral* allele (e.g., as inferred using an outgroup)
is known as *polarisation*.
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

(sec_stats_strictness)=

### Strictness, and which branches count?

Most statistics are not affected by invariant sites,
and hence do not depend on any part of the tree that is not ancestral to any of the sample sets.
However, some statistics are different: for instance, 
given a pair of samples, {meth}`TreeSequence.genetic_relatedness`
with `centre=False` and `polarised=False`
adds up the total number of alleles (or total area of branches) that is
either ancestral to both samples *or ancestral to neither*.
So, it depends on what else is in the tree sequence.
(For this reason, we don't recommend actually *using* this combination of options for genetic
relatedness; the default for that method is `polarised=True`.)

In terms of the summary function {math}`f(x)`, "not affected by invariant sites" translates to
{math}`f(0) = f(n) = 0`, where {math}`n` is the vector of sample set sizes.
By default, {meth}`TreeSequence.general_stat` checks if the summary function satisfies this condition,
and throws an error if not; this check can be disabled by setting `strict=False`.


(sec_stats_summary_functions)=

#### Summary functions

For convenience, here are the summary functions used for many of the statistics.
Below, {math}`x` denotes the number of samples in a sample set below a node,
`n` denotes the total size of a sample set, {math}`p = x / n`,
and boolean expressions (e.g., {math}`(x > 0)`) are interpreted as 0/1.

`diversity`
: {math}`f(x) = \frac{x (n - x)}{n (n-1)}`

  For an unpolarized statistic with biallelic loci, this calculates
  {math}`2 p (1-p)`.

`segregating_sites`
: {math}`f(x) =  (x > 0) (1 - x / n)`

  (Note: this works because if {math}`\sum_i p_1 = 1` then {math}`\sum_{i=1}^k (1-p_i) = k-1`.)

`Y1`
: {math}`f(x) = \frac{x (n - x) (n - x - 1)}{n (n-1) (n-2)}`

`divergence`
: {math}`f(x_1, x_2) = \frac{x_1 (n_2 - x_2)}{n_1 n_2}`,

  unless the two indices are the same, when the diversity function is used.

  For an unpolarised statistic with biallelic loci, this calculates
  {math}`p_1 (1-p_2) + (1 - p_1) p_2`.

`genetic_relatedness, centre=True`
: {math}`f(x_i, x_j) = (x_i / n_i - m)(x_j / n_j - m)`,

  where {math}`m = \frac{1}{n}\sum_{k=1}^n x_k` with {math}`n` the total number
  of sample sets.
  For a polarised statistic (the default) with biallelic loci, this calculates
  {math}`(p_1 - \bar{p}) (p_2 - \bar{p})`, where {math}`\bar{p}` is the average
  derived allele frequency across sample sets.

`genetic_relatedness, centre=False`
: {math}`f(x_i, x_j) = (x_i / n_i) (x_j / n_j)`.

  For an polarised statistic (the default) with biallelic loci, this calculates
  {math}`p_1 p_2`.

`genetic_relatedness_weighted, centre=True`
: {math}`f(w_i, w_j, x_i, x_j) = (x_i - w_i p) (x_j - w_j p)`,

  where {math}`p` is the proportion of all samples below the focal node,
  and {math}`w_j = \sum_{k=1}^n W_{kj}` is the sum of the weights in the {math}`j`th column of the weight matrix.

`genetic_relatedness_weighted, centre=False`
: {math}`f(w_i, w_j, x_i, x_j) = x_i x_j`.


`Y2`
: {math}`f(x_1, x_2) = \frac{x_1 (n_2 - x_2) (n_2 - x_2 - 1)}{n_1 n_2 (n_2 - 1)}`

`f2`
: {math}`f(x_1, x_2) = \frac{x_1 (x_1 - 1) (n_2 - x_2) (n_2 - x_2 - 1)}{n_1 (n_1 - 1) n_2 (n_2 - 1)} - \frac{x_1 (n_1 - x_1) (n_2 - x_2) x_2}{n_1 (n_1 - 1) n_2 (n_2 - 1)}`

  For an unpolarized statistic with biallelic loci, this calculates
  {math}`((p_1 - p_2)^2 - (p_1 (1-p_2)^2 + (1-p_1) p_2^2)/n_1 - (p_1^2 (1-p_2) + (1-p_1)^2 p_2)/n_2`
  {math}`+ (p_1 p_2 + (1-p_1)(1-p_2))/ n_1 n_2)(1 + \frac{1}{n_1 - 1})(1 + \frac{1}{n_2 - 1})`,
  which is the unbiased estimator for {math}`(p_1 - p_2)^2` from a finite sample.

`Y3`
: {math}`f(x_1, x_2, x_3) = \frac{x_1 (n_2 - x_2) (n_3 - x_3)}{n_1 n_2 n_3}`

`f3`
: {math}`f(x_1, x_2, x_3) = \frac{x_1 (x_1 - 1) (n_2 - x_2) (n_3 - x_3)}{n_1 (n_1 - 1) n_2 n_3} - \frac{x_1 (n_1 - x_1) (n_2 - x_2) x_3}{n_1 (n_1 - 1) n_2 n_3}`

  For an unpolarized statistic with biallelic loci, this calculates
  {math}`((p_1 - p_2)(p_1 - p_3) - p_1 (1-p_2)(1-p_3)/n_1 - (1-p_1) p_2 p_3/n_1)(1 + \frac{1}{n_1 - 1})`,
  which is the unbiased estimator for {math}`(p_1 - p_2)(p_1 - p_3)` from a finite sample.

`f4`
: {math}`f(x_1, x_2, x_3, x_4) = \frac{x_1 x_3 (n_2 - x_2) (n_4 - x_4)}{n_1 n_2 n_3 n_4} - \frac{x_1 x_4 (n_2 - x_2) (n_3 - x_3)}{n_1 n_2 n_3 n_4}`

  For an unpolarized statistic with biallelic loci, this calculates
  {math}`(p_1 - p_2)(p_3 - p_4)`.

`trait_covariance`
: {math}`f(w) = \frac{w^2}{2 (n-1)^2}`,

  where {math}`w` is the sum of all trait values of the samples below the node.

`trait_correlation`
: {math}`f(w, x) = \frac{w^2}{2 x (1 - x/n) (n - 1)}`,

  where as before {math}`x` is the total number of samples below the node,
  and {math}`n` is the total number of samples.

`trait_linear_model`
: {math}`f(w, z, x) = \frac{1}{2}\left( \frac{w - \sum_{j=1}^k z_j v_j}{x - \sum_{j=1}^k z_j^2} \right)^2`,

  where {math}`w` and {math}`x` are as before,
  {math}`z_j` is the sum of the j-th normalised covariate values below the node,
  and {math}`v_j` is the covariance of the trait with the j-th covariate.


(sec_stats_multi_site)=

## Multi site statistics

(sec_stats_two_locus)=

### Two-locus statistics

The {meth}`~TreeSequence.ld_matrix` method provides an interface to
a collection of two-locus statistics with predefined summary functions (see
{ref}`sec_stats_two_locus_summary_functions`).
The LD matrix method differs from other
statistics methods in that it provides a unified API with an argument to
specify different two-locus summaries of the data. It otherwise behaves
similarly to most other functions with respect to `sample_sets` and `indexes`.

Two-locus statistics can be computed using two {ref}`modes <sec_stats_mode>`,
either `site` or `branch`, and these should be interpreted in the same way as
these modes in the single-site statistics. That is, the `site` mode computes LD
over observed alleles at pairs of sites, while the `branch` model computes
expected LD conditioned on pairs of trees.

(sec_stats_two_locus_site)=

#### Site mode

The `"site"` mode computes two-locus statistics summarized over alleles between
all pairs of specified sites. The default behavior, leaving `sites`
unspecified, will compute a matrix for all pairs of sites, with one row and
column for each site in the tree sequence (i.e., an {math}`n \times n` matrix
where {math}`n` is the number of sites in the tree sequence). We can also
restrict the output to a subset of sites, either by specifying a single vector
of site indexes for both rows and columns or a pair of vectors for the row
sites and column sites separately.

The following computes a matrix of the {math}`r^2` measure of linkage
disequilibrium (LD) computed pairwise between the first 4 sites in the tree
sequence among all samples. The `sites` must be given as a list of lists, and
with a single list of sites specified, we obtain a symmetric square matrix. 

```{code-cell} ipython3
ld = ts.ld_matrix(sites=[[0, 1, 2, 3]])
print(ld)
```

If a list of two lists of site indexes is provided, these specify the row and
column sites. For instance, here we specify 2 rows and 3 columns, which
computes a subset of the matrix shown above.

```{code-cell} ipython3
ld = ts.ld_matrix(sites=[[1, 2], [1, 2, 3]])
print(ld)
```

#### Computational details

Because we allow for two-locus statistics to be computed for multi-allelic
data, we need to be able to combine statistical results from each pair of
alleles into one summary for a pair of sites. This does not affect biallelic
data (and so this section can be skipped on first reading).
We use two implementations for
combining results from multiple alleles: `hap_weighted` and `total_weighted`.
These are statistic-specific and not chosen by the user, with choices motivated
by [Zhao (2007)](https://doi.org/10.1017/S0016672307008634).

Briefly, consider a pair of sites with {math}`n` alleles at the first locus and
{math}`m` alleles at the second. (Whether this includes the ancestral allele
depends on whether the statistic is polarised.) Write {math}`f_{ij}` as the
statistic computed for focal alleles {math}`A_i` and {math}`B_j`. Then the
weighting schemes are defined as:

- `hap_weighted`: {math}`\sum_{i=1}^{n}\sum_{j=1}^{m}p(A_{i}B_{j})f_{ij}`,
  where {math}`p(A_{i}B_{j})` is the frequency of haplotype {math}`A_{i}B_{j}`.
  This method was first introduced in [Karlin
  (1981)](https://doi.org/10.1111/j.1469-1809.1981.tb00308.x) and reviewed in
  [Zhao (2007)](https://doi.org/10.1017/S0016672307008634).

- `total_weighted`: {math}`\frac{1}{n m}\sum_{i=1}^{n}\sum_{j=1}^{m}f_{ij}`.
  This method assigns equal weight to each of the possible pairs of focal
  alleles at the two sites, taking the arithmetic mean of statistics over
  focal haplotypes.

Out of all of the available summary functions, only {math}`r^2` uses
`hap_weighted` normalisation, with the remainder using uniform weighting
(`total_weighted`).

Within this framework, statistics may be either polarised or unpolarised. For
statistics that are polarised, we compute statistic values for pairs of derived
alleles. (For this purpose, the "derived" alleles at a site are all alleles
except that stored as the ``ancestral_state`` for the site.) Unpolarised
statistics compute statistics over all pairs of alleles, derived and ancestral.
In either case, the result is averaged over these values, using one of the
weighting scheme (described below for each statistics). The option for
polarisation is not exposed to the user, and we list which statistics are
polarised below.

(sec_stats_two_locus_branch)=

#### Branch mode

The `"branch"` mode computes expected two-locus statistics between pairs of
trees, conditioned on the marginal topologies and branch lengths of those
trees. The trees for which we compute statistics are specified by positions,
and for a pair of positions we consider all possible haplotypes that could be
generated by a single mutation occurring on each of the two trees.

For two trees, one with {math}`n` branches and the other with {math}`m`
branches, there are {math}`nm` possible pairs of branches that may carry the
pair of mutations. For each pair, we compute the two-locus statistic, and then
sum these values weighted by the product of the two branch lengths. Given that
the two mutations occur, this accounts for the relative probability that the
two mutations fall on any pair of branches.

In other words, imagine we place two mutations uniformly, one on each tree, and
then compute the statistic. The branch mode computes the expected value of the
statistic over this process, multiplied by the product of the total branch
lengths of each tree. This weighting accounts for mutational opportunity, so that
the sum of the branch-mode statistic over all positions in a genomic region,
multiplied by a mutation rate,  is equal to the expected sum of the two-locus site
statistic over all mutations falling in that region under an infinite-sites model.

The time complexity of this method is quadratic in the number of samples,
due to the pairwise comparisons of branches from each pair of trees.
By default, this method computes
a symmetric matrix for all pairs of trees, with rows and columns representing
each tree in the tree sequence. Similar to the site method, we can restrict the
output to a subset of trees, either by specifying a vector of positions or
a pair of vectors for row and column positions separately. To select a specific
tree, the specified positions must land in the tree span (`[start, end)`).

In the following, we compute a matrix of expected {math}`r^2` within and
between the first 4 trees in the tree sequence. The tree breakpoints are
a convenient way to specify those first four trees.

```{code-cell} ipython3
ld = ts.ld_matrix(
    mode="branch",
    positions=[ts.breakpoints(as_array=True)[0:4]]
)
print(ld)
```

We note that these values are quite large: as described above, the statistic is
scaled by the product of the total branch lengths of each pair of trees. To
compute the expected {math}`r^2` value for a pair of mutations that each land
uniformly on the pair of trees, we can divide by the product of the total
branch lengths:

```{code-cell} ipython3
total_branch_lengths = [tree.total_branch_length for tree in ts.trees()]
prod_branch_lengths = np.outer(total_branch_lengths, total_branch_lengths)
print(ld / prod_branch_lengths[0:4, 0:4])
```

To compute the average {math}`r^2` for a uniformly chosen pair of mutations, we also
weight by tree span:

```{code-cell} ipython3
tree_spans = np.array([t.span for t in ts.trees()])
total_opportunity = np.sum(tree_spans * total_branch_lengths)
all_ld = ts.ld_matrix(mode="branch")
mean_ld = np.sum(all_ld * np.outer(tree_spans, tree_spans)) / total_opportunity ** 2
print("mean infinite-sites LD:", mean_ld)
```

As with the `"site"` mode above, we can specify the row and column trees
separately.

```{code-cell} ipython3
breakpoints = ts.breakpoints(as_array=True)
ld = ts.ld_matrix(
    mode="branch",
    positions=[breakpoints[[0]], breakpoints[0:4]]
)
print(ld)
```

(sec_stats_two_locus_sample_sets)=

#### Sample Sets

Without specifying `sample_sets` or `indexes`, the `ld_matrix()` method
computes statistics over a single sample set that includes all samples in the
tree sequence. The API allows for the specification of a subset or multiple
subsets of samples, so that a separate LD matrix can be computed for each. If
`sample_sets` is specified as a single list of samples, then a single LD matrix
is returned. A list of lists of samples will return a 3D array containing an LD
matrix for each list of samples.

Some LD statistics can be computed between sample sets (two-way statistics are
specified below), in which case `indexes` must be specified that reference the
indexes of the `sample_sets`, which must be a list of lists of sample nodes.
This results in an LD matrix computed for each list of indexes. The statistics
are selected in the same way (with the `stat` argument), and these are limited
to a handful of statistics (see
{ref}`sec_stats_two_locus_summary_functions_two_way`). The dimension-dropping
rules for the result follow the rest of the tskit stats API in that a single
list or tuple will produce a single two-dimensional matrix, while a list of
these will produce a three-dimensional array, with the first dimension of
length equal to the length of the list.

For example, to compute the {math}`r^2` LD matrix over a subset of samples in
the tree sequence (such as sample nodes 0 through 7), we would specify the
samples as follows:

```{code-cell} ipython3
ts = msprime.sim_ancestry(
    20,
    population_size=10000,
    sequence_length=1000,
    recombination_rate=2e-8,
    random_seed=12)
ts = msprime.sim_mutations(ts, rate=2e-8, random_seed=12)

ld = ts.ld_matrix(mode="site", sample_sets=range(8))
print(ld)
```

We would get the following dimensions with the specified
`sample_sets` and `indexes` arguments.

```
# one-way
ts.ld_matrix(sample_sets=None) # -> 2 dimensions
ts.ld_matrix(sample_sets=[0, 1, 2, 3]) # -> 2 dimensions
ts.ld_matrix(sample_sets=[[0, 1, 2, 3]]) # -> 3 dimensions
# two-way
ts.ld_matrix(sample_sets=[[0, 1, 2, 3], [4, 5, 6, 7]], indexes=(0, 1)) # -> 2 dimensions
ts.ld_matrix(sample_sets=[[0, 1, 2, 3], [4, 5, 6, 7]], indexes=[(0, 1)]) # -> 3 dimensions
```

#### Why are there `nan` values in the LD matrix?

For some statistics, it is possible to observe `nan` entries in the LD matrix,
which can be surprising and may numerically impact downstream analyses. A `nan`
entry occurs if the denominator of a ratio statistic (including {math}`r` and
{math}`r^2`) is zero, indicating that one or both of the alleles in the pair is
fixed or absent in the given sample set(s). This can happen for
a number of reasons:

- Some mutation models allow for reversible mutations, so a back mutation at
  a site can result in a single allele despite multiple mutations in the
  history of the sample.
- LD is computed for a subsample of individuals, and some sites are not
  variable among the sample nodes in the subsample.
- A mutation exists above the root of the local tree, so that all samples carry
  the mutation, and one or more sites are not variable.

The `branch` mode will also return `nan` values for ratio statistics if there
are branches in either tree on which a mutation would not result in
a polymorphism within a sample set.

:::{warning}
This means there are two common situations in which many or all LD values will be `nan`.
These are:

1. A branch-mode ratio statistic computed on less than the full set of samples
    will always be `nan`, since part of the trees are ancestral to none of the samples.
2. A site-mode ratio statistic will be `nan` at any sites at which there are alleles found
    in the entire set of samples that are not seen in the provided sample set.

This behavior **may change in the future**,
because possibly more natural behavior not currently implemented
would be to ignore the branches/alleles not ancestral
to any of the provided samples.
:::

(sec_stats_two_locus_sample_one_way_stats)=

#### One-way Statistics

One-way statistics are summaries of two loci in a single sample set, using
a triple of haplotype counts {math}`\{n_{AB}, n_{Ab}, n_{aB}\}` and the size of
the sample set {math}`n`, where the capitalized and lowercase letters in our
notation represent alternate alleles.

(sec_stats_two_locus_sample_two_way_stats)=

#### Two-way Statistics

Two-way statistics are summaries of haplotype counts between two sample sets,
which operate on the three haplotype counts (as in one-way stats, above)
computed from each sample set, indexed by `(i, j)`. These statistics take on
a different meaning from their one-way counterparts. For instance `stat="D2"`
over a pair of sample sets computes {math}`D_i D_j`, which is the product of
the covariance measure of LD within each sample set and is related to the
covariance of {math}`D` between sample sets.

Only a subset of our summary functions are two-way statistics (see
{ref}`sec_two_locus_summary_functions_two_way`). Note that the unbiased two-way
statistics expect non-overlapping sample sets (see [Ragsdale and Gravel
(2020)](https://doi.org/10.1093/molbev/msz265)), and we do not make any
assertions about the sample sets and assume that `i` and `j` represent disjoint
sets of samples (see also the note in {meth}`~TreeSequence.divergence`).

(sec_stats_two_locus_summary_functions)=

#### Summary Functions

(sec_stats_two_locus_summary_functions_one_way)=

##### One-way

The two-locus summary functions all take haplotype counts and sample set size
as input. Each of our summary functions has the signature
{math}`f(n_{AB}, n_{Ab}, n_{aB}, n)`, converting to haplotype frequencies
{math}`\{p_{AB}, p_{Ab}, p_{aB}\}` by dividing by {math}`n`. Below,
{math}`n_{ab} = n - n_{AB} - n_{Ab} - n_{aB}`, {math}`n_A = n_{AB} + n_{Ab}`
and {math}`n_B = n_{AB} + n_{aB}`, with frequencies {math}`p` found by dividing
by {math}`n`.

Our convention is to use {math}`A,B` to denote derived alleles, and {math}`a,b`
ancestral alleles (or other alleles, if the site is multi-allelic). For
polarised statistics, we average statistics over all non-ancestral alleles. For
unpolarised statistics, the labeling is arbitrary as we average over all
alleles (derived and ancestral).

`D`
: {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = p_{AB}p_{ab} - p_{Ab}p_{aB} \, (=p_{AB} - p_A p_B)`

  This statistic is polarised, as the unpolarised result, which averages over
  allele labelings, is zero. Uses the `total` weighting method.

`D_prime`
: {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = \frac{D}{D_{\max}}`,

  where {math}`D_{\max} = \begin{cases}
            \min\{p_A (1-p_B), p_B (1-p_B)\} & \textrm{if }D>=0 \\
            \min\{p_A p_B, (1-p_B) (1-p_B)\} & \textrm{if }D<0
        \end{cases}`

  and {math}`D` is defined above. Polarised, `total` weighted.

`D2`
: {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = D^2`

  and {math}`D` is defined above. Unpolarised, `total` weighted.

`Dz`
:  {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = D (1 - 2 p_A) (1 - 2 p_B)`,

  where {math}`D` is defined above. Unpolarised, `total` weighted.

`pi2`
: {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = p_A (1-p_A) p_B (1-p_B)`

  Unpolarised, `total` weighted.

`r`
: {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = \frac{D}{\sqrt{p_A (1-p_A) p_B (1-p_B)}}`,

  where {math}`D` is defined above. Polarised, `total` weighted.

`r2`
: {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = \frac{D^{2}}{p_A (1-p_A) p_B (1-p_B))}`,

  where {math}`D` is defined above. Unpolarised, `haplotype` weighted.

Unbiased two-locus statistics from the Hill-Robertson (1968) system are
computed from haplotype counts. Definitions of these unbiased estimators can
be found in [Ragsdale and Gravel
(2020)](https://doi.org/10.1093/molbev/msz265). They require at least 4 samples
to be valid and are specified as `stat="D2_unbiased"`, `"Dz_unbiased"`, or
`"pi2_unbiased"`.

(sec_two_locus_summary_functions_two_way)=

(sec_stats_two_locus_summary_functions_two_way)=

##### Two-way

Two-way statistics are indexed by sample sets {math}`i, j` and compute values
using haplotype counts within pairs of sample sets.

`D2`
: {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = D_i D_j`,

  where {math}`D_i` denotes {math}`D` computed within sample set {math}`i`,
  and {math}`D` is defined above. Unpolarised, `total` weighted.

`r2`
: {math}`f(n_{AB}, n_{Ab}, n_{aB}, n) = r_i r_j`,

  where {math}`r_i` denotes {math}`r` computed within sample set {math}`i`,
  and {math}`r` is defined above. Unpolarised, `haplotype` weighted.

And `D2_unbiased`, which can be found in [Ragsdale and Gravel
(2020)](https://doi.org/10.1093/molbev/msz265).


(sec_stats_notes)=

## Notes


(sec_stats_notes_afs)=

### Allele frequency spectrum

Most single site statistics are based on the summaries of the allele frequency spectra
(AFS). The `tskit` AFS interface includes windowed and joint spectra,
using the same general pattern as other statistics,
but some of the details about how it is defined,
especially in the presence of multiple alleles per site, need to be explained.
If all sites are biallelic, then the result is just as you'd expect:
see the method documentation at {meth}`~TreeSequence.allele_frequency_spectrum` 
for the description.
Note that with `mode="site"`, we really do tabulate *allele* counts:
if more than one mutation on different parts of the tree produce the same allele,
it is the total number with this allele (i.e., inheriting *either* mutation)
that goes into the AFS.
The AFS with `mode="branch"` is the expected value for the Site AFS
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
In our example, we'd add 1 to both `AFS[3]` and `AFS[1]`.
This means that the sum of all entries of a polarised, site AFS
should equal the total number of non-ancestral alleles in the tree sequence
that are ancestral to at least one of the samples in the tree sequence
but not ancestral to all of them.
The reason for this last caveat is that like with most statistics,
mutations that are not ancestral to *any* samples (not just those in the sample sets)
are not counted (and so don't even enter into `AFS[0]`),
and similarly for those alleles inherited by *all* samples.

Now, if we are computing the *unpolarised* AFS,
we add *one half* to each entry of the *folded* output
corresponding to each allele count *including* the ancestral allele.
What does this mean?
Well, `polarised=False` means that we cannot distinguish between an
allele count of 6 and an allele count of 4.
So, *folding* means that we would add our allele that is seen in 6 samples
to `AFS[4]` instead of `AFS[6]`.
So, in total, we will add 0.5 to each of `AFS[4]`, `AFS[3]`, and `AFS[1]`.
This means that the sum of an unpolarised AFS
will be equal to the total number of alleles that are inherited
by any of the samples in the tree sequence, divided by two.
Why one-half? Well, notice that if in fact the mutation that produced the *b*
allele had instead produced an *a* allele,
so that the site had only two alleles, with frequencies 7 and 3.
Then, we would have added 0.5 to `AFS[3]` *twice*.


(sec_stats_notes_trait)=

### Trait correlations

{meth}`~TreeSequence.trait_covariance`, {meth}`~TreeSequence.trait_correlation`, and
{meth}`~TreeSequence.trait_linear_model` compute correlations and covariances of traits
(i.e., an arbitrary vector) with allelic state, possibly in the context of a multivariate
linear model with other covariates (as in GWAS).


(sec_stats_notes_f)=

### Patterson's f statistics

{meth}`~TreeSequence.f4`, {meth}`~TreeSequence.f3`, and {meth}`~TreeSequence.f2`
are the `f` statistics (also called `F` statistics) introduced in
[Reich et al (2009)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2842210/).
See the documentation (link below) for the definition,
and [Peter (2016)](https://www.genetics.org/content/202/4/1485) for readable
discussion of their use.


(sec_stats_notes_y)=

### Y statistics

{meth}`~TreeSequence.Y3` and {meth}`~TreeSequence.Y2` are the `Y` statistics introduced
by [Ashander et al (2018)](https://www.biorxiv.org/content/10.1101/354530v1)
as a three-sample intermediate between diversity/divergence (which are
pairwise) and Patterson's f statistics (which are four-way).


(sec_stats_notes_derived)=

### Derived statistics

Most statistics have the property that `mode="branch"` and
`mode="site"` are "dual" in the sense that they are equal, on average, under
a high neutral mutation rate. {meth}`~TreeSequence.Fst` and {meth}`~TreeSequence.Tajimas_D`
do not have this property (since both are ratios of statistics that do have this property).


(sec_stats_notes_gnn)=

### Genealogical nearest neighbours

The {meth}`~TreeSequence.genealogical_nearest_neighbours` statistic is not based on branch
lengths, but on topologies. therefore it currently has a slightly different interface to
the other single site statistics. This may be revised in the future.
