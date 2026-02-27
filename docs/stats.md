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
    * {meth}`~TreeSequence.ld_matrix` (see {ref}`sec_stats_two_locus`)

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

:::{todo}
Document statistics that use information about correlation between sites, such as
LdCalculator (and perhaps reference {ref}`sec_identity`). Note that if we have a general
framework which has the same calling conventions as the single site stats,
we can rework the sections above.
:::

(sec_stats_two_locus)=

### Two-locus statistics

The {meth}`~TreeSequence.ld_matrix` method provides an interface to a collection
of two-locus statistics with predefined summary functions (see
{ref}`sec_stats_two_locus_summary_functions`) and `site` and `branch` {ref}`modes
<sec_stats_mode>`. This method differs from the other stats methods because it
provides a collection of statistics, instead of the usual one method per
stat. Otherwise, it behaves similarly to most other functions with respect to
`sample_sets` and `indexes`. Site statistics can be computed from multi-allelic
data, while branch statistics are inherently biallelic. Within this framework,
we also implement polarisation, but do not expose it to the user, opting to
provide statistics that are polarised where appropriate.

(sec_stats_two_locus_site)=

#### Site

The `site` mode computes two-locus statistics from pairs of alleles on sites. By
default, this method will compute a matrix for all pairs of sites, with rows and
columns representing each site in the tree sequence (ie an n×n matrix where n is
the number of sites in the tree sequence). We can also restrict the output to a
subset of sites, either by specifying a single vector for both rows and columns
or a pair of vectors for the row sites and column sites separately.

The following example produces a matrix containing {math}`r^2` computed pairwise
between the first 4 sites in the tree sequence (all samples included). In our
computations, row sites are used as the lefthand locus and column sites are used
as the righthand locus (though in most cases, the LD matrix will be symmetric).

```{code-cell} ipython3
ld = ts.ld_matrix(sites=[[0, 1, 2, 3]])
print(ld)
```

Here's an example demonstrating how we can specify the row and column sites
independently of each other. We're specifying 4 columns and 1 row, selecting the
first row from the matrix shown above.
```{code-cell} ipython3
ld = ts.ld_matrix(sites=[[0], [0, 1, 2, 3]])
print(ld)
```

Because we implement two-locus statistics for multi-allelic data, we need a
method for combining the statistical results from each allele into one summary
for a pair of sites. We have two methods for combining results from multiple
alleles:

`hap_weighted`
: {math}`\sum_{i=1}^{n}\sum_{j=1}^{m}p(A_{i}B_{j})f_{ij}`

  where {math}`f_{ij}` is the result of computing the summary function on data
  from the {math}`i`th row site and the {math}`j`th column site and
  {math}`p(A_{i}B_{j})` is the frequency of haplotype {math}`A_{i}B_{j}`. This
  method was first introduced in
  [Karlin (1981)](https://doi.org/10.1111/j.1469-1809.1981.tb00308.x), and was
  reviewed in [Zhao (2007)](https://doi.org/10.1017/S0016672307008634).

  This method multiplies each result by the frequency of the haplotype under
  consideration (the {math}`[i,j]` alleles for a given pair of sites), then sums
  the results to obtain the statistic for the two focal sites.

`total_weighted`
: {math}`\frac{1}{(n-\mathbb{1}_{p}) (m-\mathbb{1}_{p})}\sum_{i=1}^{n}\sum_{j=1}^{m}f_{ij}`

  where {math}`\mathbb{1}_{p}` is an indicator function conditioned on whether
  or not our statistic is polarized, {math}`n` is the number of alleles in site
  {math}`a`, and {math}`m` is the number of alleles in site {math}`b`.

  This method assigns equal weight to each of the possible haplotypes for a
  given pair of sites, it might be more aptly named "uniform_weighted". In
  essence, we are taking the arithmetic mean of each statistic.

Out of all of the available summary functions, only {math}`r^2` uses
`hap_weighted` normalisation. The rest behave correctly under a uniform weighting
(`total_weighted`).

(sec_stats_two_locus_branch)=

#### Branch

The `branch` mode computes two-locus statistics between pairs of trees, the
resulting statistic is a summary of the branch lengths between both trees, over
the area formed by multiplying the spans of the two trees. To perform this
computation, we use the partition formed by each node in the tree to generate
haplotype counts, then we sum each pairwise comparison weighted by the branch
length above the node. The time complexity of this method is quadratic, due to
the pairwise comparisons of nodes from a pair of trees. By default, this method
computes a symmetric matrix for all pairs of trees, with rows and columns
representing each tree in the tree sequence. Similar to the site method, we can
restrict the output to a subset of trees, either by specifying a vector of
positions or a pair of vectors for row and column positions separately. To
select a specific tree, the specified positions must land in the tree span
(`[start, end)`).

The following example produces a matrix containing {math}`r^2` computed pairwise
between the first 4 trees in the tree sequence. The tree breakpoints are a
convenient way to address each tree.

```{code-cell} ipython3
ld = ts.ld_matrix(mode="branch", positions=[ts.breakpoints(as_array=True)[0:4]])
print(ld)
```

We can also specify the row and column trees separately.

```{code-cell} ipython3
breakpoints = ts.breakpoints(as_array=True)
ld = ts.ld_matrix(mode="branch", positions=[breakpoints[[0]], breakpoints[0:4]])
print(ld)
```

(sec_stats_two_locus_sample_sets)=

#### Sample Sets

The two-locus API provides a mechanism by which to subset the samples under
consideration, providing the ability to compute a separate LD matrix for each
sample set or an LD matrix between sample sets. Output dimensions are handled in
the same manner as the rest of the stats api (see
{ref}`sec_stats_output_dimensions`). The two-locus API differs from the rest of
the stats API in that we handle one-way and two-way statistics in the same
function call. To compute a two-way two-locus statistic, the `index` argument
must be provided. The statistics are selected in the same way (with the `stat`
argument), but we provide a restricted set of two-way statistics (see
{ref}`sec_stats_two_locus_summary_functions_two_way`). The dimension-dropping
rules for the result follow the rest of the tskit stats API in that scalar
indexes will produce a n×m matrix, while lists of indexes will produce a k×n×m
matrix, even if there is only one index present in the list.

Here's a summary of the dimensions we would expect to observe for all of the
possible types of sample set inputs.
```
# one-way
ts.ld_matrix(sample_sets=None) -> 2 dimensions
ts.ld_matrix(samples=[...]) -> 2 dimensions
ts.ld_matrix(samples=[[...]]) -> 3 dimensions
# two-way
ts.ld_matrix(samples=[[...]], indexes=(i, j)) -> 2 dimensions
ts.ld_matrix(samples=[[...]], indexes=[(i, j)]) -> 3 dimensions
```

(sec_stats_two_locus_sample_one_way_stats)=

##### One-way Statistics

One-way statistics are summaries of two loci in a single sample set, using a
triple of haplotype counts {math}`\{w_{Ab}, w_{aB}, w_{AB}\}` and the size of
the sample set {math}`n`, where the capitalized letter in our notation
represents the derived allele and lowercase represents the ancestral
allele. When the statistics are polarised and we're considering a multi-allelic
site, we exclude the ancestral state, computing our two-locus summary only on
the derived alleles. When we're considering a pair of biallelic sites or a pair
of trees (branch statistics), we compute our summary stat for both derived
alleles ({math}`A`/{math}`B`).

(sec_stats_two_locus_sample_two_way_stats)=

##### Two-way Statistics

Two-way statistics are summaries of haplotype counts between two sample sets,
which operate on the three mentioned haplotype counts computed from each
sample set, indexed by `(i, j)`. These statistics take on a different
meaning from their one-way counterparts. For instance {math}`D_i D_j` can be
thought of as the covariance between two loci when computed for a haplotype in a
single sample set. When computed between sample sets {math}`i` and {math}`j`, we
have a measure of the covariance of covariance between two loci in two
populations (or, more succinctly, the covariance of LD). Only a subset of our
summary functions are two-way statistics (see
{ref}`sec_two_locus_summary_functions_two_way`). Note that the unbiased two-way
statistics expect nonoverlapping sample sets, we do not make any assertions
about the sample sets and assume that `i` and `j` represent disjoint sets of
samples (see also the note in {meth}`~TreeSequence.divergence`).

(sec_stats_two_locus_summary_functions)=

#### Summary Functions

(sec_stats_two_locus_summary_functions_one_way)=

##### One-way

The two-locus summary functions all take haplotype counts and sample set size as
input. Each of our summary functions has the signature
{math}`f(w_{Ab}, w_{aB}, w_{AB}, n)`, converting to haplotype frequencies
{math}`\{p_{Ab}, p_{aB}, p_{AB}\}` where appropriate by dividing by {math}`n`.

`D`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = p_{ab} - p_{a}p_{b}`

  This statistic is inherently polarised, as the unpolarised result of this
  statistic is expected to be zero. Uses the `total` normalisation method.

`D_prime`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = \frac{D}{D_{\max}}`

  Where {math}```D_{\max} = \begin{cases}
        \min\{p_A (1-p_B), p_B (1-p_B)\} & \textrm{if }D>=0 \\
        \min\{p_A p_B, (1-p_B) (1-p_B)\} & \textrm{otherwise}
        \end{cases}```
  and {math}`D` is defined above.

`D2`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = (p_{ab} - p_{a}p_{b})^2`

  Unpolarised, total weighted.

`Dz`
:  {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = D (1 - 2 p_{a})(1-2p_{b})`

  Where {math}`D` is defined above. {math}`D_z` measures positive covariance of
  low frequency genomic variants and is sensitive to changes in demographic
  history. Unpolarised, total weighted.

`pi2`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = p_{a}p_{b}(1-p_{a})(1-p_{b})`

  {math}`\pi2` is the joint heterozygosity between two loci and can be used for
  normalizing D statistics to control for variable mutation rate and for
  estimating coalescent times. Unpolarised, total weighted.

`r`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = \frac{D}{\sqrt{p_{a}p_{b}(1-p_{a})(1-p_{b})}}`

  Where {math}`D` is defined above. Polarised, total weighted.

`r2`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = \frac{D^{2}}{p_{a}p_{b}(1-p_{a})(1-p_{b})}`

  Where {math}`D` is defined above. Unpolarised, haplotype weighted.

Unbiased two-locus statistics are computed from haplotype counts Derivations for
these unbiased estimators can be found in [Ragsdale and Gravel
(2020)](https://doi.org/10.1093/molbev/msz265). They require at least 4 samples
to be valid.

`D2_unbiased`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = \frac{1}{n (n - 1) (n - 2) (n - 3)} \begin{pmatrix} w_{aB}^2 (w_{Ab} - 1) w_{Ab} \\ + (w_{ab} - 1) w_{ab} (w_{AB} - 1) w_{AB} \\ - w_{aB} w_{Ab} (w_{Ab} + (2 w_{ab} w_{AB}) - 1) \end{pmatrix} `

`Dz_unbiased`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = \\ \frac{1}{n (n - 1) (n - 2) (n - 3))} \begin{pmatrix} ((w_{Ab} + w_{ab} - w_{AB} - w_{aB}) \\ (w_{AB} w_{ab} - w_{Ab} w_{aB})(w_{aB} + w_{ab} - w_{AB} - w_{Ab})) \\ - w_{AB} w_{ab} (w_{AB} + w_{ab} - w_{Ab} - w_{aB} - 2) \\ - w_{Ab} w_{aB} (w_{Ab} + w_{aB} - w_{AB} - w_{ab} - 2) \end{pmatrix}`

`pi2_unbiased`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = \\ \frac{1}{n (n - 1) (n - 2) (n - 3)} \begin{pmatrix} (w_{AB} + w_{Ab}) (w_{aB} + w_{ab}) (w_{AB} + w_{aB}) (w_{Ab} + w_{ab}) \\ - w_{AB} w_{ab} (w_{AB} + w_{ab} + 3 w_{Ab} + 3 w_{aB} - 1) \\ - w_{Ab} w_{aB} (w_{Ab} + w_{aB} + 3 w_{AB} + 3 w_{ab} - 1) \\ \end{pmatrix}`

(sec_two_locus_summary_functions_two_way)=

(sec_stats_two_locus_summary_functions_two_way)=

##### Two-way

Our two-way statistics index into a collection of haplotype counts with their
user provided sample set index {math}`i, j` (i.e. {math}`w_{AB_i}`).

`D2`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = D_i * D_j`

Where {math}`D` is defined above.

`D2_unbiased`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = \begin{cases} \frac {w_{AB_i} (w_{AB_i} - 1) w_{ab_i} (w_{ab_i} - 1) + w_{Ab_i} (w_{Ab_i} - 1) w_{aB_i} (w_{aB_i} - 1) - 2 w_{AB_i} w_{Ab_i} w_{aB_i} w_{ab_i}} {n_i (n_i - 1) (n_i - 2) (n_i - 3)} & \textrm{if }i=j \\ \frac {(w_{Ab_i} w_{aB_i} - w_{AB_i} w_{ab_i}) (w_{Ab_j} w_{aB_j} - w_{AB_j} w_{ab_j})} {n_i (n_i - 1) n_j (n_j - 1)} & \textrm{if }i \neq j \\ \end{cases}`

  Where {math}`w_{ab_i} = n_i - (w_{AB_i} + w_{Ab_i} + w_{aB_i})`
  
`r2`
: {math}`f(w_{Ab}, w_{aB}, w_{AB}, n) = \frac{(p_{AB_i} - (p_{A_i}  p_{B_i})) (p_{AB_j} - (p_{A_j}  p_{B_j}))}{\sqrt{p_{A_i} (1 - p_{A_i}) p_{B_i} (1 - p_{B_i})}\sqrt{p_{A_j} (1 - p_{A_j}) p_{B_j} (1 - p_{B_j})}}`

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
