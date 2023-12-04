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

(sec_python_api)=

# Python API

This page documents the full tskit Python API. Brief thematic summaries of common
classes and methods are presented first. The {ref}`sec_python_api_reference` section
at the end then contains full details which aim to be concise, precise and exhaustive.
Note that this may not therefore be the best place to start if you are new
to a particular piece of functionality.

(sec_python_api_trees_and_tree_sequences)=

## Trees and tree sequences

The {class}`TreeSequence` class represents a sequence of correlated
evolutionary trees along a genome. The {class}`Tree` class represents a
single tree in this sequence. These classes are the interfaces used to
interact with the trees and mutational information stored in a tree sequence,
for example as returned from a simulation or inferred from a set of DNA
sequences.


(sec_python_api_tree_sequences)=

### {class}`TreeSequence` API


(sec_python_api_tree_sequences_properties)=

#### General properties

```{eval-rst}
.. autosummary::
  TreeSequence.time_units
  TreeSequence.nbytes
  TreeSequence.sequence_length
  TreeSequence.max_root_time
  TreeSequence.discrete_genome
  TreeSequence.discrete_time
  TreeSequence.metadata
  TreeSequence.metadata_schema
  TreeSequence.reference_sequence
```

#### Efficient table column access

The {class}`.TreeSequence` class provides access to underlying numerical
data defined in the {ref}`data model<sec_data_model>` in two ways:

1. Via the {attr}`.TreeSequence.tables` property and the
    {ref}`Tables API<sec_tables_api_accessing_table_data>`
2. Via a set of properties on the ``TreeSequence`` class that provide
   direct and efficient access to the underlying memory.

:::{warning}
Accessing table data via {attr}`.TreeSequence.tables` can be very inefficient
at the moment because accessing the `.tables` property incurs a **full copy**
of the data model. While we intend to implement this as a read-only view
in the future, the engineering involved is nontrivial, and so we recommend
using the properties listed here like ``ts.nodes_time`` in favour of
``ts.tables.nodes.time``.
Please see [issue #760](https://github.com/tskit-dev/tskit/issues/760)
for more information.
:::


```{eval-rst}
.. autosummary::
  TreeSequence.individuals_flags
  TreeSequence.nodes_time
  TreeSequence.nodes_flags
  TreeSequence.nodes_population
  TreeSequence.nodes_individual
  TreeSequence.edges_left
  TreeSequence.edges_right
  TreeSequence.edges_parent
  TreeSequence.edges_child
  TreeSequence.sites_position
  TreeSequence.mutations_site
  TreeSequence.mutations_node
  TreeSequence.mutations_parent
  TreeSequence.mutations_time
  TreeSequence.migrations_left
  TreeSequence.migrations_right
  TreeSequence.migrations_right
  TreeSequence.migrations_node
  TreeSequence.migrations_source
  TreeSequence.migrations_dest
  TreeSequence.migrations_time
  TreeSequence.indexes_edge_insertion_order
  TreeSequence.indexes_edge_removal_order
```

(sec_python_api_tree_sequences_loading_and_saving)=

#### Loading and saving

There are several methods for loading data into a {class}`TreeSequence`
instance. The simplest and most convenient is the use the {func}`tskit.load`
function to load a {ref}`tree sequence file <sec_tree_sequence_file_format>`. For small
scale data and debugging, it is often convenient to use the {func}`tskit.load_text`
function to read data in the {ref}`text file format<sec_text_file_format>`.
The {meth}`TableCollection.tree_sequence` function
efficiently creates a {class}`TreeSequence` object from a
{class}`collection of tables<TableCollection>`
using the {ref}`Tables API <sec_tables_api>`.

```{eval-rst}
Load a tree sequence
    .. autosummary::
      load
      load_text
      TableCollection.tree_sequence

Save a tree sequence
    .. autosummary::
      TreeSequence.dump
```

:::{seealso}
Tree sequences with a single simple topology can also be created from scratch by
{ref}`generating<sec_python_api_trees_creating>` a {class}`Tree` and accessing its
{attr}`~Tree.tree_sequence` property.
:::

(sec_python_api_tree_sequences_obtaining_trees)=

#### Obtaining trees

The following properties and methods return information about the
{class}`trees<Tree>` that are generated along a tree sequence.

```{eval-rst}
.. autosummary::

  TreeSequence.num_trees
  TreeSequence.trees
  TreeSequence.breakpoints
  TreeSequence.coiterate
  TreeSequence.first
  TreeSequence.last
  TreeSequence.aslist
  TreeSequence.at
  TreeSequence.at_index
```

#### Obtaining other objects

(sec_python_api_tree_sequences_obtaining_other_objects)=

Various components make up a tree sequence, such as nodes and edges, sites and
mutations, and populations and individuals. These can be counted or converted into
Python objects using the following classes, properties, and methods.

##### Tree topology

```{eval-rst}
Nodes
    .. autosummary::
      Node
      TreeSequence.num_nodes
      TreeSequence.nodes
      TreeSequence.node
      TreeSequence.num_samples
      TreeSequence.samples

Edges
    .. autosummary::
      Edge
      TreeSequence.num_edges
      TreeSequence.edges
      TreeSequence.edge
```

##### Genetic variation

```{eval-rst}
Sites
    .. autosummary::
      Site
      TreeSequence.num_sites
      TreeSequence.sites
      TreeSequence.site
      Variant
      TreeSequence.variants
      TreeSequence.genotype_matrix
      TreeSequence.haplotypes
      TreeSequence.alignments

Mutations
    .. autosummary::
      Mutation
      TreeSequence.num_mutations
      TreeSequence.mutations
      TreeSequence.mutation
```

##### Demography

```{eval-rst}
Populations
    .. autosummary::
      Population
      TreeSequence.num_populations
      TreeSequence.populations
      TreeSequence.population

Migrations
    .. autosummary::
      Migration
      TreeSequence.num_migrations
      TreeSequence.migrations
      TreeSequence.migration
```

##### Other

```{eval-rst}
Individuals
    .. autosummary::
      Individual
      TreeSequence.num_individuals
      TreeSequence.individuals
      TreeSequence.individual


Provenance entries (also see :ref:`sec_python_api_provenance`)
    .. autosummary::
      Provenance
      TreeSequence.num_provenances
      TreeSequence.provenances
      TreeSequence.provenance
```

(sec_python_api_tree_sequences_modification)=

#### Tree sequence modification

Although tree sequences are immutable, several methods will taken an existing tree
sequence and return a modifed version. These are thin wrappers around the
{ref}`identically named methods of a TableCollection<sec_tables_api_modification>`,
which perform the same actions but modify the {class}`TableCollection` in place.

```{eval-rst}
.. autosummary::
  TreeSequence.simplify
  TreeSequence.subset
  TreeSequence.union
  TreeSequence.keep_intervals
  TreeSequence.delete_intervals
  TreeSequence.delete_sites
  TreeSequence.trim
  TreeSequence.split_edges
  TreeSequence.decapitate
  TreeSequence.extend_edges
```

(sec_python_api_tree_sequences_ibd)=

#### Identity by descent

The {meth}`.TreeSequence.ibd_segments` method allows us to compute
identity relationships between pairs of samples. See the
{ref}`sec_identity` section for more details and examples
and the {ref}`sec_python_api_reference_identity` section for
API documentation on the associated classes.

```{eval-rst}
.. autosummary::
  TreeSequence.ibd_segments
```

(sec_python_api_tree_sequences_tables)=

#### Tables

The underlying data in a tree sequence is stored in a
{ref}`collection of tables<sec_tables_api>`. The following methods give access
to tables and associated functionality. Since tables can be modified, this
allows tree sequences to be edited: see the {ref}`sec_tables` tutorial for
an introduction.

```{eval-rst}
.. autosummary::
  TreeSequence.tables
  TreeSequence.dump_tables
  TreeSequence.table_metadata_schemas
  TreeSequence.tables_dict
```


(sec_python_api_tree_sequences_statistics)=

#### Statistics

```{eval-rst}

Single site
    .. autosummary::
      TreeSequence.allele_frequency_spectrum
      TreeSequence.divergence
      TreeSequence.diversity
      TreeSequence.f2
      TreeSequence.f3
      TreeSequence.f4
      TreeSequence.Fst
      TreeSequence.genealogical_nearest_neighbours
      TreeSequence.genetic_relatedness
      TreeSequence.genetic_relatedness_weighted
      TreeSequence.general_stat
      TreeSequence.segregating_sites
      TreeSequence.sample_count_stat
      TreeSequence.mean_descendants
      TreeSequence.Tajimas_D
      TreeSequence.trait_correlation
      TreeSequence.trait_covariance
      TreeSequence.trait_linear_model
      TreeSequence.Y2
      TreeSequence.Y3

Comparative
    .. autosummary::
      TreeSequence.kc_distance
```

(sec_python_api_tree_sequences_topological_analysis)=

#### Topological analysis

The topology of a tree in a tree sequence refers to the relationship among
samples ignoring branch lengths. Functionality as described in
{ref}`sec_topological_analysis` is mainly provided via
{ref}`methods on trees<sec_python_api_trees_topological_analysis>`, but more
efficient methods sometimes exist for entire tree sequences:

```{eval-rst}
.. autosummary::
  TreeSequence.count_topologies
```

(sec_python_api_tree_sequences_display)=

#### Display

```{eval-rst}
.. autosummary::
  TreeSequence.draw_svg
  TreeSequence.draw_text
  TreeSequence.__str__
  TreeSequence._repr_html_
```


(sec_python_api_tree_sequences_export)=

#### Export
```{eval-rst}
.. autosummary::
  TreeSequence.as_fasta
  TreeSequence.as_nexus
  TreeSequence.dump_text
  TreeSequence.to_macs
  TreeSequence.write_fasta
  TreeSequence.write_nexus
  TreeSequence.write_vcf
```


(sec_python_api_trees)=

### {class}`Tree<Tree>` API

A tree is an instance of the {class}`Tree` class. These trees cannot exist
independently of the {class}`TreeSequence` from which they are generated.
Usually, therefore, a {class}`Tree` instance is created by
{ref}`sec_python_api_tree_sequences_obtaining_trees` from an existing tree
sequence (although it is also possible to generate a new instance of a
{class}`Tree` belonging to the same tree sequence using {meth}`Tree.copy`).

:::{note}
For efficiency, each instance of a {class}`Tree` is a state-machine
whose internal state corresponds to one of the trees in the parent tree sequence:
{ref}`sec_python_api_trees_moving_to` in the tree sequence does not require a
new instance to be created, but simply the internal state to be changed.
:::

(sec_python_api_trees_general_properties)=

#### General properties


```{eval-rst}
.. autosummary::
  Tree.tree_sequence
  Tree.total_branch_length
  Tree.root_threshold
  Tree.virtual_root
  Tree.num_edges
  Tree.num_roots
  Tree.has_single_root
  Tree.has_multiple_roots
  Tree.root
  Tree.roots
  Tree.index
  Tree.interval
  Tree.span
```


(sec_python_api_trees_creating)=

#### Creating new trees

It is sometimes useful to create an entirely new tree sequence consisting
of just a single tree (a "one-tree sequence"). The follow methods create such an
object and return a {class}`Tree` instance corresponding to that tree.
The new tree sequence to which the tree belongs is available through the
{attr}`~Tree.tree_sequence` property.

```{eval-rst}
Creating a new tree
    .. autosummary::
      Tree.generate_balanced
      Tree.generate_comb
      Tree.generate_random_binary
      Tree.generate_star

Creating a new tree from an existing tree
    .. autosummary::
      Tree.split_polytomies
```

:::{seealso}
{meth}`Tree.unrank` for creating a new one-tree sequence from its
{ref}`topological rank<sec_python_api_trees_topological_analysis>`.
:::

:::{note}
Several of these methods are {func}`static<python:staticmethod>`, so should
be called e.g. as `tskit.Tree.generate_balanced(4)` rather than used on
a specific {class}`Tree` instance.
:::


(sec_python_api_trees_node_measures)=

#### Node measures

Often it is useful to access information pertinant to a specific node or set of nodes
but which might also change from tree to tree in the tree sequence. Examples include
the encoding of the tree via `parent`, `left_child`, etc.
(see {ref}`sec_data_model_tree_structure`), the number of samples under a node,
or the most recent common ancestor (MRCA) of two nodes. This sort of information is
available via simple and high performance {class}`Tree` methods


(sec_python_api_trees_node_measures_simple)=

##### Simple measures

These return a simple number, or (usually) short list of numbers relevant to a specific
node or limited set of nodes.

```{eval-rst}
Node information
    .. autosummary::
      Tree.is_sample
      Tree.is_isolated
      Tree.is_leaf
      Tree.is_internal
      Tree.parent
      Tree.num_children
      Tree.time
      Tree.branch_length
      Tree.depth
      Tree.population
      Tree.right_sib
      Tree.left_sib
      Tree.right_child
      Tree.left_child
      Tree.children
      Tree.edge

Descendant nodes
    .. autosummary::
      Tree.leaves
      Tree.samples
      Tree.num_samples
      Tree.num_tracked_samples

Multiple nodes
    .. autosummary::
      Tree.is_descendant
      Tree.mrca
      Tree.tmrca
```


(sec_python_api_trees_node_measures_array)=

##### Array access

These all return a numpy array whose length corresponds to
the total number of nodes in the tree sequence. They provide direct access
to the underlying memory structures, and are thus very efficient, providing a
high performance interface which can be used in conjunction with the equivalent
{ref}`traversal methods<sec_python_api_trees_traversal>`.

```{eval-rst}
.. autosummary::
  Tree.parent_array
  Tree.left_child_array
  Tree.right_child_array
  Tree.left_sib_array
  Tree.right_sib_array
  Tree.num_children_array
  Tree.edge_array
```


(sec_python_api_trees_traversal)=

#### Tree traversal

Moving around within a tree usually involves visiting the tree nodes in some sort of
order. Often, given a particular order, it is convenient to iterate over each node
using the {meth}`Tree.nodes` method. However, for high performance algorithms, it
may be more convenient to access the node indices for a particular order as
an array, and use this, for example, to index into one of the node arrays (see
{ref}`sec_topological_analysis_traversal`).

```{eval-rst}
Iterator access
    .. autosummary::

      Tree.nodes

Array access
    .. autosummary::

      Tree.postorder
      Tree.preorder
      Tree.timeasc
      Tree.timedesc
```


(sec_python_api_trees_topological_analysis)=

#### Topological analysis

The topology of a tree refers to the simple relationship among samples
(i.e. ignoring branch lengths), see {ref}`sec_combinatorics` for more details. These
methods provide ways to enumerate and count tree topologies.

Briefly, the position of a tree in the enumeration `all_trees` can be obtained using
the tree's {meth}`~Tree.rank` method. Inversely, a {class}`Tree` can be constructed
from a position in the enumeration with {meth}`Tree.unrank`.


```{eval-rst}
Methods of a tree
    .. autosummary::
      Tree.rank
      Tree.count_topologies

Functions and static methods
    .. autosummary::
      Tree.unrank
      all_tree_shapes
      all_tree_labellings
      all_trees
```


(sec_python_api_trees_comparing)=

#### Comparing trees

```{eval-rst}
.. autosummary::
  Tree.kc_distance
```

(sec_python_api_trees_balance)=

#### Balance/imbalance indices

```{eval-rst}
.. autosummary::
  Tree.colless_index
  Tree.sackin_index
  Tree.b1_index
  Tree.b2_index
```

(sec_python_api_trees_sites_mutations)=

#### Sites and mutations

```{eval-rst}
.. autosummary::
  Tree.sites
  Tree.num_sites
  Tree.mutations
  Tree.num_mutations
  Tree.map_mutations
```


(sec_python_api_trees_moving_to)=

#### Moving to other trees

```{eval-rst}
.. autosummary::

  Tree.next
  Tree.prev
  Tree.first
  Tree.last
  Tree.seek
  Tree.seek_index
  Tree.clear
```

#### Display

```{eval-rst}
.. autosummary::

  Tree.draw_svg
  Tree.draw_text
  Tree.__str__
  Tree._repr_html_
```

#### Export

```{eval-rst}
.. autosummary::

  Tree.as_dict_of_dicts
  Tree.as_newick
```


(sec_tables_api)=

## Tables and Table Collections

The information required to construct a tree sequence is stored in a collection
of *tables*, each defining a different aspect of the structure of a tree
sequence. These tables are described individually in
{ref}`the next section<sec_tables_api_table>`. However, these are interrelated,
and so many operations work
on the entire collection of tables, known as a *table collection*.

(sec_tables_api_table_collection)=

### `TableCollection` API

The {class}`TableCollection` and {class}`TreeSequence` classes are
deeply related. A `TreeSequence` instance is based on the information
encoded in a `TableCollection`. Tree sequences are **immutable**, and
provide methods for obtaining trees from the sequence. A `TableCollection`
is **mutable**, and does not have any methods for obtaining trees.
The `TableCollection` class thus allows creation and modification of
tree sequences (see the {ref}`sec_tables` tutorial).


#### General properties

Specific {ref}`tables<sec_tables_api_table>` in the {class}`TableCollection`
are be accessed using the plural version of their name, so that, for instance, the
individual table can be accessed using `table_collection.individuals`. A table
collection also has other properties containing, for example, number of bytes taken
to store it and the top-level metadata associated with the tree sequence as a whole.

```{eval-rst}
Table access
    .. autosummary::
      TableCollection.individuals
      TableCollection.nodes
      TableCollection.edges
      TableCollection.migrations
      TableCollection.sites
      TableCollection.mutations
      TableCollection.populations
      TableCollection.provenances

Other properties
    .. autosummary::
      TableCollection.file_uuid
      TableCollection.indexes
      TableCollection.nbytes
      TableCollection.table_name_map
      TableCollection.metadata
      TableCollection.metadata_bytes
      TableCollection.metadata_schema
      TableCollection.sequence_length
      TableCollection.time_units
```


(sec_tables_api_transformation)=

#### Transformation

These methods act in-place to transform the contents of a {class}`TableCollection`,
either by modifying the underlying tables (removing, editing, or adding to them) or
by adjusting the table collection so that it meets the
{ref}`sec_valid_tree_sequence_requirements`.


(sec_tables_api_modification)=

##### Modification

These methods modify the data stored in a {class}`TableCollection`. They also have
{ref}`equivalant TreeSequence versions<sec_python_api_tree_sequences_modification>`
(unlike the methods described below those do *not* operate in place, but rather act in
a functional way, returning a new tree sequence while leaving the original unchanged).

```{eval-rst}
.. autosummary::
  TableCollection.clear
  TableCollection.simplify
  TableCollection.subset
  TableCollection.delete_intervals
  TableCollection.keep_intervals
  TableCollection.delete_sites
  TableCollection.trim
  TableCollection.union
  TableCollection.delete_older
```

(sec_tables_api_creating_valid_tree_sequence)=

##### Creating a valid tree sequence

These methods can be used to help reorganise or rationalise the
{class}`TableCollection` so that it is in the form
{ref}`required<sec_valid_tree_sequence_requirements>` for
it to be {meth}`converted<TableCollection.tree_sequence>`
into a {class}`TreeSequence`. This may require sorting the tables,
ensuring they are logically consistent, and adding {ref}`sec_table_indexes`.

:::{note}
These methods are not guaranteed to make valid a {class}`TableCollection` which is
logically inconsistent, for example if multiple edges have the same child at a
given position on the genome or if non-existent node IDs are referenced.
:::

```{eval-rst}
Sorting
    .. autosummary::
      TableCollection.sort
      TableCollection.sort_individuals
      TableCollection.canonicalise

Logical consistency
    .. autosummary::
      TableCollection.compute_mutation_parents
      TableCollection.compute_mutation_times
      TableCollection.deduplicate_sites

Indexing
    .. autosummary::
      TableCollection.has_index
      TableCollection.build_index
      TableCollection.drop_index
```

#### Miscellaneous methods

```{eval-rst}
.. autosummary::
  TableCollection.copy
  TableCollection.equals
  TableCollection.link_ancestors
```

#### Export
```{eval-rst}
.. autosummary::
  TableCollection.tree_sequence
  TableCollection.dump
```

(sec_tables_api_table)=

### Table APIs

Here we outline the table classes and the common methods and variables available for
each. For description and definition of each table's meaning
and use, see {ref}`the table definitions <sec_table_definitions>`.

```{eval-rst}
.. autosummary::

  IndividualTable
  NodeTable
  EdgeTable
  MigrationTable
  SiteTable
  MutationTable
  PopulationTable
  ProvenanceTable
```

(sec_tables_api_accessing_table_data)=

#### Accessing table data

The tables API provides an efficient way of working
with and interchanging {ref}`tree sequence data <sec_data_model>`. Each table class
(e.g, {class}`NodeTable`, {class}`EdgeTable`, {class}`SiteTable`) has a specific set
of columns with fixed types, and a set of methods for setting and getting the data
in these columns. The number of rows in the table `t` is given by `len(t)`.

```{code-cell} ipython3
import tskit
t = tskit.EdgeTable()
t.add_row(left=0, right=1, parent=10, child=11)
t.add_row(left=1, right=2, parent=9, child=11)
print("The table contains", len(t), "rows")
print(t)
```

Each table supports accessing the data either by row or column. To access the data in
a *column*, we can use standard attribute access which will
return a copy of the column data as a numpy array:

```{code-cell} ipython3
t.left
```

```{code-cell} ipython3
t.parent
```

To access the data in a *row*, say row number `j` in table `t`, simply use `t[j]`:

```{code-cell} ipython3
t[0]
```

This also works as expected with negative `j`, counting rows from the end of the table

```{code-cell} ipython3
t[-1]
```

The returned row has attributes allowing contents to be accessed by name, e.g.
`site_table[0].position`, `site_table[0].ancestral_state`, `site_table[0].metadata`
etc.:

```{code-cell} ipython3
t[-1].right
```

Row attributes cannot be modified directly. Instead, the `replace` method of a row
object can be used to create a new row with one or more changed column
values, which can then be used to replace the original. For example:

```{code-cell} ipython3
t[-1] = t[-1].replace(child=4, right=3)
print(t)
```

Tables also support the {mod}`pickle` protocol, and so can be easily serialised and
deserialised. This can be useful, for example, when performing parallel computations
using the {mod}`multiprocessing` module (however, pickling will not be as efficient
as storing tables in the native {ref}`format <sec_tree_sequence_file_format>`).

```{code-cell} ipython3
import pickle
serialised = pickle.dumps(t)
t2 = pickle.loads(serialised)
print(t2)
```

Tables support the equality operator `==` based on the data
held in the columns:

```{code-cell} ipython3
t == t2
```

```{code-cell} ipython3
t is t2
```

```{code-cell} ipython3
t2.add_row(0, 1, 2, 3)
print(t2)
t == t2
```

:::{todo}
Move some or all of these examples into a suitable alternative chapter.
:::


(sec_tables_api_text_columns)=

##### Text columns

As described in the {ref}`sec_encoding_ragged_columns`, working with
variable length columns is somewhat more involved. Columns
encoding text data store the **encoded bytes** of the flattened
strings, and the offsets into this column in two separate
arrays.

Consider the following example:

```{code-cell} ipython3
t = tskit.SiteTable()
t.add_row(0, "A")
t.add_row(1, "BB")
t.add_row(2, "")
t.add_row(3, "CCC")
print(t)
print(t[0])
print(t[1])
print(t[2])
print(t[3])
```

Here we create a {class}`SiteTable` and add four rows, each with a different
`ancestral_state`. We can then access this information from each
row in a straightforward manner. Working with columns of text data
is a little trickier, however:

```{code-cell} ipython3
print(t.ancestral_state)
print(t.ancestral_state_offset)
```

```{code-cell} ipython3
tskit.unpack_strings(t.ancestral_state, t.ancestral_state_offset)
```

Here, the `ancestral_state` array is the UTF8 encoded bytes of the flattened
strings, and the `ancestral_state_offset` is the offset into this array
for each row. The {func}`tskit.unpack_strings` function, however, is a convient
way to recover the original strings from this encoding. We can also use the
{func}`tskit.pack_strings` to insert data using this approach:

```{code-cell} ipython3
a, off = tskit.pack_strings(["0", "12", ""])
t.set_columns(position=[0, 1, 2], ancestral_state=a, ancestral_state_offset=off)
print(t)
```

When inserting many rows with standard infinite sites mutations (i.e.,
ancestral state is "0"), it is more efficient to construct the
numpy arrays directly than to create a list of strings and use
{func}`pack_strings`. When doing this, it is important to note that
it is the **encoded** byte values that are stored; by default, we
use UTF8 (which corresponds to ASCII for simple printable characters).:

```{code-cell} ipython3
import numpy as np
t_s = tskit.SiteTable()
m = 10
a = ord("0") + np.zeros(m, dtype=np.int8)
off = np.arange(m + 1, dtype=np.uint32)
t_s.set_columns(position=np.arange(m), ancestral_state=a, ancestral_state_offset=off)
print(t_s)
print("ancestral state data", t_s.ancestral_state)
print("ancestral state offsets", t_s.ancestral_state_offset)
```


In the mutation table, the derived state of each mutation can be handled similarly:

```{code-cell} ipython3
t_m = tskit.MutationTable()
site = np.arange(m, dtype=np.int32)
d, off = tskit.pack_strings(["1"] * m)
node = np.zeros(m, dtype=np.int32)
t_m.set_columns(site=site, node=node, derived_state=d, derived_state_offset=off)
print(t_m)
```

:::{todo}
Move some or all of these examples into a suitable alternative chapter.
:::


(sec_tables_api_binary_columns)=

##### Binary columns

Columns storing binary data take the same approach as
{ref}`sec_tables_api_text_columns` to encoding
{ref}`variable length data <sec_encoding_ragged_columns>`.
The difference between the two is only raw {class}`bytes` values are accepted: no
character encoding or decoding is done on the data. Consider the following example
where a table has no `metadata_schema` such that arbitrary bytes can be stored and
no automatic encoding or decoding of objects is performed by the Python API and we can
store and retrieve raw `bytes`. (See {ref}`sec_metadata` for details):

Below, we add two rows to a {class}`NodeTable`, with different
{ref}`metadata <sec_metadata_definition>`. The first row contains a simple
byte string, and the second contains a Python dictionary serialised using
{mod}`pickle`.

```{code-cell} ipython3
t = tskit.NodeTable()
t.add_row(metadata=b"these are raw bytes")
t.add_row(metadata=pickle.dumps({"x": 1.1}))
print(t)
```

Note that the pickled dictionary is encoded in 24 bytes containing unprintable
characters. It appears to be unrelated to the original contents, because the binary
data is [base64 encoded](https://en.wikipedia.org/wiki/Base64) to ensure that it is
print-safe (and doesn't break your terminal). (See the
{ref}`sec_metadata_definition` section for more information on the
use of base64 encoding.).

We can access the metadata in a row (e.g., `t[0].metadata`) which returns a Python
bytes object containing precisely the bytes that were inserted.

```{code-cell} ipython3
print(t[0].metadata)
print(t[1].metadata)
```

The metadata containing the pickled dictionary can be unpickled using
{func}`pickle.loads`:

```{code-cell} ipython3
print(pickle.loads(t[1].metadata))
```

As previously, the `replace` method can be used to change the metadata,
by overwriting an existing row with an updated one:

```{code-cell} ipython3
t[0] = t[0].replace(metadata=b"different raw bytes")
print(t)
```

Finally, when we print the `metadata` column, we see the raw byte values
encoded as signed integers. As for {ref}`sec_tables_api_text_columns`,
the `metadata_offset` column encodes the offsets into this array. So, we
see that the first metadata value is 9 bytes long and the second is 24.

```{code-cell} ipython3
print(t.metadata)
print(t.metadata_offset)
```

The {func}`tskit.pack_bytes` and {func}`tskit.unpack_bytes` functions are
also useful for encoding data in these columns.

:::{todo}
Move some or all of these examples into a suitable alternative chapter.
:::



#### Table functions

```{eval-rst}
.. autosummary::

  parse_nodes
  parse_edges
  parse_sites
  parse_mutations
  parse_individuals
  parse_populations
  parse_migrations
  pack_strings
  unpack_strings
  pack_bytes
  unpack_bytes
```


(sec_python_api_metadata)=

## Metadata API

The `metadata` module provides validation, encoding and decoding of metadata
using a schema. See {ref}`sec_metadata`, {ref}`sec_metadata_api_overview` and
{ref}`sec_tutorial_metadata`.

```{eval-rst}
.. autosummary::
  MetadataSchema
  register_metadata_codec
```

:::{seealso}
Refer to the top level metadata-related properties of TreeSequences and TableCollections,
such as {attr}`TreeSequence.metadata` and {attr}`TreeSequence.metadata_schema`. Also the
metadata fields of
{ref}`objects accessed<sec_python_api_tree_sequences_obtaining_other_objects>` through
the {class}`TreeSequence` API.
:::


(sec_python_api_provenance)=

## Provenance

We provide some preliminary support for validating JSON documents against the
{ref}`provenance schema <sec_provenance>`. Programmatic access to provenance
information is planned for future versions.


```{eval-rst}
.. autosummary::
  validate_provenance
```

(sec_utility_api)=

## Utility functions

Miscellaneous top-level utility functions.

```{eval-rst}
.. autosummary::
  is_unknown_time
  random_nucleotides
```


(sec_python_api_reference)=

## Reference documentation

(sec_python_api_constants)=

### Constants

The following constants are used throughout the `tskit` API.

```{eval-rst}
.. automodule:: tskit
   :members:
```

(sec_python_api_exceptions)=

### Exceptions

```{eval-rst}
.. autoexception:: DuplicatePositionsError
.. autoexception:: MetadataEncodingError
.. autoexception:: MetadataSchemaValidationError
.. autoexception:: MetadataValidationError
.. autoexception:: ProvenanceValidationError
```

(sec_python_api_functions)=

### Top-level functions

```{eval-rst}
.. autofunction:: all_trees
.. autofunction:: all_tree_shapes
.. autofunction:: all_tree_labellings
.. autofunction:: is_unknown_time
.. autofunction:: load
.. autofunction:: load_text
.. autofunction:: pack_bytes
.. autofunction:: pack_strings
.. autofunction:: parse_edges
.. autofunction:: parse_individuals
.. autofunction:: parse_mutations
.. autofunction:: parse_nodes
.. autofunction:: parse_populations
.. autofunction:: parse_migrations
.. autofunction:: parse_sites
.. autofunction:: random_nucleotides
.. autofunction:: register_metadata_codec
.. autofunction:: validate_provenance
.. autofunction:: unpack_bytes
.. autofunction:: unpack_strings

```

### Tree and tree sequence classes

#### The {class}`Tree` class

Also see the {ref}`sec_python_api_trees` summary.

```{eval-rst}
.. autoclass:: Tree()
    :members:
    :special-members: __str__
    :private-members: _repr_html_
```

#### The {class}`TreeSequence` class

Also see the {ref}`sec_python_api_tree_sequences` summary.

```{eval-rst}
.. autoclass:: TreeSequence()
    :members:
    :special-members: __str__
    :private-members: _repr_html_
```

### Simple container classes

#### The {class}`Individual` class

```{eval-rst}
.. autoclass:: Individual()
    :members:
```

#### The {class}`Node` class

```{eval-rst}
.. autoclass:: Node()
    :members:
```

#### The {class}`Edge` class

```{eval-rst}
.. autoclass:: Edge()
    :members:
```

#### The {class}`Site` class

```{eval-rst}
.. autoclass:: Site()
    :members:
```

#### The {class}`Mutation` class

```{eval-rst}
.. autoclass:: Mutation()
    :members:
```

#### The {class}`Variant` class

```{eval-rst}
.. autoclass:: Variant()
    :members:
```

#### The {class}`Migration` class

```{eval-rst}
.. autoclass:: Migration()
    :members:
```

#### The {class}`Population` class

```{eval-rst}
.. autoclass:: Population()
    :members:
```

#### The {class}`Provenance` class

```{eval-rst}
.. autoclass:: Provenance()
    :members:
```

#### The {class}`Interval` class

```{eval-rst}
.. autoclass:: Interval()
    :members:
```

#### The {class}`Rank` class

```{eval-rst}
.. autoclass:: Rank()
    :members:
```

### TableCollection and Table classes

#### The {class}`TableCollection` class

Also see the {ref}`sec_tables_api_table_collection` summary.

```{eval-rst}
.. autoclass:: TableCollection
    :inherited-members:
    :members:
```

% Overriding the default signatures for the tables here as they will be
% confusing to most users.


#### {class}`IndividualTable` classes

```{eval-rst}
.. autoclass:: IndividualTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

##### Associated row class

A row returned from an {class}`IndividualTable` is an instance of the following
basic class, where each attribute matches an identically named attribute in the
{class}`Individual` class.

```{eval-rst}
.. autoclass:: IndividualTableRow()
    :members:
    :inherited-members:
```


#### {class}`NodeTable` classes

```{eval-rst}
.. autoclass:: NodeTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

##### Associated row class

A row returned from a {class}`NodeTable` is an instance of the following
basic class, where each attribute matches an identically named attribute in the
{class}`Node` class.

```{eval-rst}
.. autoclass:: NodeTableRow()
    :members:
    :inherited-members:
```


#### {class}`EdgeTable` classes

```{eval-rst}
.. autoclass:: EdgeTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

##### Associated row class

A row returned from an {class}`EdgeTable` is an instance of the following
basic class, where each attribute matches an identically named attribute in the
{class}`Edge` class.

```{eval-rst}
.. autoclass:: EdgeTableRow()
    :members:
    :inherited-members:
```


#### {class}`MigrationTable` classes

```{eval-rst}
.. autoclass:: MigrationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

##### Associated row class

A row returned from a {class}`MigrationTable` is an instance of the following
basic class, where each attribute matches an identically named attribute in the
{class}`Migration` class.

```{eval-rst}
.. autoclass:: MigrationTableRow()
    :members:
    :inherited-members:
```


#### {class}`SiteTable` classes

```{eval-rst}
.. autoclass:: SiteTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

##### Associated row class

A row returned from a {class}`SiteTable` is an instance of the following
basic class, where each attribute matches an identically named attribute in the
{class}`Site` class.

```{eval-rst}
.. autoclass:: SiteTableRow()
    :members:
    :inherited-members:
```


#### {class}`MutationTable` classes

```{eval-rst}
.. autoclass:: MutationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

##### Associated row class

A row returned from a {class}`MutationTable` is an instance of the following
basic class, where each attribute matches an identically named attribute in the
{class}`Mutation` class.

```{eval-rst}
.. autoclass:: MutationTableRow()
    :members:
    :inherited-members:
```


#### {class}`PopulationTable` classes

```{eval-rst}
.. autoclass:: PopulationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

##### Associated row class

A row returned from a {class}`PopulationTable` is an instance of the following
basic class, where each attribute matches an identically named attribute in the
{class}`Population` class.

```{eval-rst}
.. autoclass:: PopulationTableRow()
    :members:
    :inherited-members:
```


#### {class}`ProvenanceTable` classes

Also see the {ref}`sec_provenance` and
{ref}`provenance API methods<sec_python_api_provenance>`.

```{eval-rst}
.. autoclass:: ProvenanceTable()
    :members:
    :inherited-members:
```

##### Associated row class

A row returned from a {class}`ProvenanceTable` is an instance of the following
basic class, where each attribute matches an identically named attribute in the
{class}`Provenance` class.

```{eval-rst}
.. autoclass:: ProvenanceTableRow()
    :members:
    :inherited-members:
```

(sec_python_api_reference_identity)=

### Identity classes

The classes documented in this section are associated with summarising
identity relationships between pairs of samples. See the {ref}`sec_identity`
section for more details and examples.

#### The {class}`IdentitySegments` class

```{eval-rst}
.. autoclass:: IdentitySegments()
    :members:
```

#### The {class}`IdentitySegmentList` class

```{eval-rst}
.. autoclass:: IdentitySegmentList()
    :members:
```

#### The {class}`IdentitySegment` class

```{eval-rst}
.. autoclass:: IdentitySegment()
    :members:
```

### Miscellaneous classes

#### The {class}`ReferenceSequence` class

```{eval-rst}
.. todo:: Add a top-level summary section that we can link to from here.
```

```{eval-rst}
.. autoclass:: ReferenceSequence()
    :members:
    :inherited-members:
```

#### The {class}`MetadataSchema` class

Also see the {ref}`sec_python_api_metadata` summary.

```{eval-rst}
.. autoclass:: MetadataSchema
    :members:
    :inherited-members:
```

#### The {class}`TableMetadataSchemas` class

```{eval-rst}
.. autoclass:: TableMetadataSchemas
    :members:
```

#### The {class}`TopologyCounter` class

```{eval-rst}
.. autoclass:: TopologyCounter
```

#### The {class}`LdCalculator` class

```{eval-rst}
.. autoclass:: LdCalculator
    :members:
```

#### The {class}`TableCollectionIndexes` class

```{eval-rst}
.. autoclass:: TableCollectionIndexes
    :members:
```

#### The {class}`SVGString` class

```{eval-rst}
.. autoclass:: SVGString
    :members:
    :private-members: _repr_svg_
```
