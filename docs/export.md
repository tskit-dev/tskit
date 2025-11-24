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

```{code-cell}
:tags: [hide-input]

from IPython.display import display
```

(sec_export)=

# Data export

(sec_export_vcf)=
## Variant Call Format

Tskit supports exporting data to the standard
[Variant Call Format](http://samtools.github.io/hts-specs/VCFv4.3.pdf)
via the `tskit vcf` {ref}`command line interface<sec_cli>` command
and the {meth}`TreeSequence.write_vcf` method in the {ref}`sec_python_api`.
Conversion is quite efficient, with tskit producing VCF data at several
hundred megabytes per second (for large files), which is usually as fast as
it can be written to storage or consumed by programs in a pipeline.

::::{tip}
If we have a tree sequence file the
{ref}`command line interface<sec_cli>` is often the most
convenient way to convert to VCF:

:::{code-block} bash
$ tskit vcf example.trees > example.vcf
:::

See the {ref}`sec_export_vcf_compression` section for information
on how to compress the VCF output.
::::

For tree sequences produced by recent versions of programs such as
``msprime``, ``SLiM``, ``fwdpy11`` or ``tsinfer``, VCF output will
"do the right thing" and no further arguments are needed.
For example, here we simulate 3 diploid individuals
with mutations using ``msprime``, and convert to VCF.

```{code-cell}
import sys
import msprime
ts = msprime.sim_ancestry(
    samples=3, ploidy=2, sequence_length=10, random_seed=2)
ts = msprime.sim_mutations(ts, rate=0.1, random_seed=2)
ts.write_vcf(sys.stdout)
```

In the output VCF we have 3 diploid samples
(see the {ref}`sec_export_vcf_terminology` section)
corresponding to samples specified in the ancestry simulation
with IDs ``tsk_0``, ``tsk_1`` and ``tsk_2``
(see the {ref}`sec_export_vcf_individual_names`
section for how to override these default labels).
We then have a line for every row
in the {ref}`site table<sec_site_table_definition>`, and
the data is derived directly from the {meth}`TreeSequence.variants`
method; e.g.

```{code-cell}
for var in ts.variants():
    print(var.site.position, var.site.id, var.alleles, var.genotypes, sep="\t")
```

We can see the ``POS`` value is equal to the site's position
(see the {ref}`sec_export_vcf_modifying_coordinates` for information
on how we deal with continuous coordinates), the ``ID`` value
is the site's ID, and the ``REF`` and ``ALT`` values
are derived from the variant's ``alleles``.

The ``GT`` values for the three diploid individuals are derived from the
variant's genotypes (see the {ref}`sec_export_vcf_terminology` section).
For this simulation, the diploid individuals correspond to
adjacent sample nodes in order, and we can see there is a direct
correspondence between the phased ``GT`` values and variant's genotypes.
See the {ref}`sec_export_vcf_constructing_gt` section for
more information on how this done in general and for options
to control the VCF sample and ``GT`` values.

::::{important}
In these examples we write the VCF data to ``sys.stdout`` so that we can see
the output. Usually, however, you'd write to a file:

:::{code-block}
with open("output.vcf", "w") as vcf_file:
    ts.write_vcf(vcf_file)
:::

:::{seealso}
See the {ref}`sec_export_vcf_compression` section for information
on how to compress the output or convert to BCF.
:::

::::

(sec_export_vcf_terminology)=

### Terminology

There are some mismatches between the terminology for tskit and VCF.
In VCF a "sample" is a multiploid individual, but in tskit a sample
refers to a single **node** (monoploid genome), and an individual
consists of one or more nodes (e.g., two nodes for a diploid).
Similarly, in VCF a "genotype" refers to the observed allelic state
for a sample **individual** at a particular site,
whereas in tskit a genotype is the observed allelic state
for a **node** (see {attr}`.Variant.genotypes`).

:::{seealso}
See the {ref}`sec_glossary` for more details on tskit's data model
 and terminology.
:::

(sec_export_vcf_compression)=

### Compressed output

The simplest way to compress the VCF output is to use the
`tskit vcf` {ref}`command line interface<sec_cli>`
and pipe the output to `bgzip`:

:::{code-block} bash
$ tskit vcf example.trees | bgzip -c > example.vcf.gz
:::
A general way to convert VCF data to various formats is to pipe the text
produced by ``tskit`` into ``bcftools`` using the command
line interface:

:::{code-block} bash
$ tskit vcf example.trees | bcftools view -O b > example.bcf
:::

If you need more control over the form of the output (or want to work
directly in Python), the following recipe has the same effect:

:::{code-block}

import os
import subprocess

read_fd, write_fd = os.pipe()
write_pipe = os.fdopen(write_fd, "w")
with open("output.bcf", "w") as bcf_file:
    proc = subprocess.Popen(
        ["bcftools", "view", "-O", "b"], stdin=read_fd, stdout=bcf_file
    )
    ts.write_vcf(write_pipe)
    write_pipe.close()
    os.close(read_fd)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("bcftools failed with status:", proc.returncode)
:::


The VCF output can also be compressed using the {mod}`gzip` Python module:

:::{code-block}

import gzip

with gzip.open("output.vcf.gz", "wt") as f:
    ts.write_vcf(f)
:::

However, this gzipped VCF won't be fully compatible with downstream tools
such as tabix, which usually require the VCF to use the specialised bgzip format.

(sec_export_vcf_masking_output)=

### Masking output

The {meth}`TreeSequence.write_vcf` method provides the
``site_mask`` and ``sample_mask`` arguments to
omit or mark parts of the output as missing.

```{code-cell}
ts = msprime.sim_ancestry(
    samples=3, ploidy=2, sequence_length=10, random_seed=2)
ts = msprime.sim_mutations(ts, rate=0.1, random_seed=2)
ts.tables.sites
```

The ``sample_mask`` argument provides a general way to mask out
parts of the output, which can be helpful when simulating missing
data. In this (contrived) example, we create a sample mask function
that marks one genotype missing in each variant in a regular
pattern:

:::{code-block}

def sample_mask(variant):
    sample_mask = np.zeros(ts.num_samples, dtype=bool)
    sample_mask[variant.site.id % ts.num_samples] = 1
    return sample_mask


ts.write_vcf(sys.stdout, sample_mask=sample_mask)
:::

(sec_export_vcf_constructing_gt)=

### Constructing GT values

The core elements of the tskit
{ref}`data model<sec_data_model>`
are {ref}`nodes<sec_node_table_definition>`,
{ref}`edges<sec_node_table_definition>`,
{ref}`sites<sec_site_table_definition>` and
{ref}`mutations<sec_mutation_table_definition>`.
These four tables allow us to completely describe the
genetic ancestry of a set of sampled monoploid
genomes and their genetic variation.
The {ref}`individual table<sec_individual_table_definition>`
defines a set of individual *organisms*, and it can
be used to define the inheritance relationships between
then (the pedigree). An individual may be associated
with one or more nodes, and these nodes may or
may not be samples (see the {ref}`sec_glossary`
for clarification of these terms).
Thus, there is some complexity in how the per-individual GT values
are generated, which we explain in this section.

#### Without individuals

We start with an example in which there are no individuals
defined (which was the default in msprime before version 1.0):

```{code-cell}
import tskit
tables = tskit.Tree.generate_balanced(4, span=10).tree_sequence.dump_tables()
tables.sites.add_row(3, ancestral_state="A")
tables.mutations.add_row(site=0, node=0, derived_state="T")
ts = tables.tree_sequence()
display(ts.draw_svg())
display(ts)
ts.write_vcf(sys.stdout)
```

Here we define a tree sequence consisting of a single tree, which
has a variant site at position 3 and a mutation over node 0.
There is no information about individuals in this tree sequence,
and so we assume that each of the nodes corresponds to a single
haploid individual.

Users of msprime simulations would often be interested in producing
VCFs for diploid organisms. Because of the assumptions made
by these simulations, this means arbitrarily combining the sample
nodes into pairs. This is what the ``ploidy`` option does:

```{code-cell}
ts.write_vcf(sys.stdout, ploidy=2)
```

Thus, the ``GT`` values for the (synthetic) diploid individual ``tsk_0``
are generated by combining nodes 0 and 1, and ``tsk_1``
by combining nodes 2 and 3.

:::{important}
Software packages modelling multiploid individuals are encouraged to
use the individual table to make their assumptions explicit. Recent
versions of simulators and inference methods should all do this,
and so the ``ploidy`` argument is really only intended to support
legacy code. It is therefore an error to supply a value for ``ploidy``
when individual information is present in a tree sequence.
:::

#### With individuals

Extending the example in the previous section, we add some individual data
defining a pair of diploid sibs and their parents.

:::{note}
We set the nodes for (e.g.) individual 2 to [1, 3] here to illustrate
that nodes for a given individual are not necessarily contiguous.
:::

```{code-cell}
tables.individuals.add_row(parents=[-1, -1])
tables.individuals.add_row(parents=[-1, -1])
tables.individuals.add_row(parents=[0, 1])
tables.individuals.add_row(parents=[0, 1])
node_individual = tables.nodes.individual
node_individual[[1, 3]] = 2
node_individual[[0, 2]] = 3
tables.nodes.individual = node_individual
display(tables.individuals)
display(tables.nodes)
ts = tables.tree_sequence()
ts.write_vcf(sys.stdout)
```

In this model we have four individuals defined, but only
individuals 2 and 3 are associated with nodes (more specifically,
**sample** nodes). Thus, we output **two** VCF sample individuals
composed of the linked nodes.

:::{note}
Note that the labels are ``tsk_0`` and ``tsk_1`` even though
the individual IDs are 2 and 3. See the
{ref}`sec_export_vcf_individual_names` section for how to change the
these default labels.
:::

If some individuals have no associated nodes, they are omitted from the
VCF output. By default, only nodes that are marked as samples contribute
to the VCF genotypes; to include non-sample nodes as well (e.g., internal
nodes that have been marked as individuals), set
``include_non_sample_nodes=True`` when calling :meth:`TreeSequence.write_vcf`.

:::{note}
At present, :meth:`TreeSequence.write_vcf` only supports sites with up to
9 distinct alleles; attempting to write a site with more than 9 alleles
will result in a :class:`ValueError`.
:::

(sec_export_vcf_individual_names)=

### Individual names

By default the VCF samples are given the labels ``tsk_0``, ``tsk_1``,
..., ``tsk_{N - 1}``, where ``N`` is the number of individuals to
be output (see the {ref}`sec_export_vcf_constructing_gt` section).

We can change this default labelling using the ``individual_names``
argument::

```{code-cell}
import sys
import msprime
ts = msprime.sim_ancestry(
    samples=3, ploidy=2, sequence_length=10, random_seed=2)
ts = msprime.sim_mutations(ts, rate=0.1, random_seed=2)
ts.write_vcf(sys.stdout, individual_names=["A", "B", "C"])
```

#### Exporting to plink

The default VCF sample IDs produced by ``tskit`` do not work well
with plink because it parses the individual
IDs based on a particular format, and does not allow ``0`` as a valid
identifier. We get an error like this:

```
Error: Sample ID ends with "_0", which induces an invalid IID of '0`.
```

This can be fixed by using the ``individual_names`` argument
to set the names to anything where the first name doesn't end with ``_0``.
An example implementation for diploid individuals is:

:::{code-block}
n_dip_indv = int(ts.num_samples / 2)
indv_names = [f"tsk_{i}indv" for i in range(n_dip_indv)]
with open("output.vcf", "w") as vcf_file:
    ts.write_vcf(vcf_file, individual_names=indv_names)
:::

Adding a second ``_`` (eg: ``tsk_0_indv``) is not recommended as
``plink`` uses ``_`` as the default separator for separating family
id and individual id, and two underscores will throw an error.

(sec_export_vcf_modifying_coordinates)=

### Modifying coordinates

Tree sequence site positions can be floating point values, whereas VCF
requires positive integers. The ``position_transform`` argument
controls how tskit maps coordinates into VCF. Translating non-integer
positions necessarily loses precision; by default we round to the nearest
integer, so multiple sites may share the same output position. 
Furthermore, tskit's coordinate system starts at zero,
whereas the VCF standard requires positions to be positive,
and so a site at position 0 is not valid in the VCF standard.
Because VCF parsers differ, we do **not** do anything to account for this.

The simplest resolution of this discrepancy in convention between tskit and VCF
positions is deal with any site at position 0 as a special case (for instance,
by discarding or ignoring it).
A different interpretation of this difference between tskit's position
and VCF's POS field
is that they are different coordinate systems: tskit coordinates are
"distance to the right of the left end of the chromosome",
while VCF coordinates are "which number site, counting from the left end
of the chromosome and starting at one".
Under this interpretation, the solution is to supply an explicit
``position_transform`` that adds 1 to the coordinate when outputting
to VCF (or, using the ``"legacy"`` option described below). However, this can
easily lead to off-by-one errors converting between the coordinate systems,
so should only be used if you really are using 0-based coordinates for your
tree sequence.

:::{warning}
Most VCF tools cannot deal with a POS value of 0. If your tree
sequence contains a site with position 0, this will likely cause an error.
:::

Internally, the coordinates used in the VCF output are obtained by applying
the ``position_transform`` function to the array of site positions (and, for
the contig length, to the tree sequence :attr:`.TreeSequence.sequence_length`).
This function must return a one-dimensional array of the same length as its
input; otherwise a :class:`ValueError` is raised. In addition to accepting a
callable, tskit also supports the string value ``"legacy"`` here, which
selects the pre-0.2.0 behaviour used by the original VCF exporter:
positions are rounded to the nearest integer, starting at 1, and are forced
to be strictly increasing by incrementing ties.

The VCF specification does not allow positions to be 0. By default, if any
transformed position is 0, :meth:`TreeSequence.write_vcf` will raise a
:class:`ValueError`. If you wish to retain these records you can either:

- set ``allow_position_zero=True`` to write such sites anyway;
- mask the offending sites using the ``site_mask`` argument; or
- choose a ``position_transform`` that maps 0 to a valid positive position.

For example, to shift all coordinates by 1, we could define:

```{code-cell}
def one_based_positions(positions):
    return [int(round(x)) + 1 for x in positions]

ts.write_vcf(sys.stdout, position_transform=one_based_positions)
```

:::{note}
The msprime 0.x legacy API simulates using continuous coordinates. It may
be simpler to update your code to use the msprime 1.0 API (which uses
discrete coordinates by default) than to work out how to transform
coordinates in a way that is suitable for your application.
:::
