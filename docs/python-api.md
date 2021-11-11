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

This page provides detailed documentation for the `tskit` Python API.

(sec_python_api_trees_and_tree_sequences)=

## Trees and tree sequences

The {class}`TreeSequence` class represents a sequence of correlated
evolutionary trees along a genome. The {class}`Tree` class represents a
single tree in this sequence. These classes are the interfaces used to
interact with the trees and mutational information stored in a tree sequence,
for example as returned from a simulation or inferred from a set of DNA
sequences. This library also provides methods for loading stored tree
sequences, for example using {func}`tskit.load`.


### The `TreeSequence` class

```{eval-rst}
.. autoclass:: TreeSequence()
    :members:
    :autosummary:
```


### The `Tree` class

Trees in a tree sequence can be obtained by iterating over
{meth}`TreeSequence.trees` and specific trees can be accessed using methods
such as {meth}`TreeSequence.first`, {meth}`TreeSequence.at` and
{meth}`TreeSequence.at_index`. Each tree is an instance of the following
class which provides methods, for example, to access information
about particular nodes in the tree.

```{eval-rst}
.. autoclass:: Tree()
    :members:
    :autosummary:
```

### Simple container classes

These classes are simple shallow containers representing the entities defined
in the {ref}`sec_data_model_definitions`. These classes are not intended to be
instantiated directly, but are the return types for the various iterators provided
by the {class}`TreeSequence` and {class}`Tree` classes.

```{eval-rst}
.. autoclass:: Individual()
    :members:
```

```{eval-rst}
.. autoclass:: Node()
    :members:
```

```{eval-rst}
.. autoclass:: Edge()
    :members:
```

```{eval-rst}
.. autoclass:: Interval()
    :members:
```

```{eval-rst}
.. autoclass:: Site()
    :members:
```

```{eval-rst}
.. autoclass:: Mutation()
    :members:
```

```{eval-rst}
.. autoclass:: Variant()
    :members:
```

```{eval-rst}
.. autoclass:: Migration()
    :members:
```

```{eval-rst}
.. autoclass:: Population()
    :members:
```

```{eval-rst}
.. autoclass:: Provenance()
    :members:
```

### Loading data

There are several methods for loading data into a {class}`TreeSequence`
instance. The simplest and most convenient is the use the {func}`tskit.load`
function to load a {ref}`tree sequence file <sec_tree_sequence_file_format>`. For small
scale data and debugging, it is often convenient to use the {func}`tskit.load_text`
function to read data in the {ref}`text file format<sec_text_file_format>`.
The {meth}`TableCollection.tree_sequence` function
efficiently creates a {class}`TreeSequence` object from a set of tables
using the {ref}`Tables API <sec_tables_api>`.


```{eval-rst}
.. autofunction:: load
```

```{eval-rst}
.. autofunction:: load_text
```

(sec_tables_api)=

## Tables and Table Collections

The information required to construct a tree sequence is stored in a collection
of *tables*, each defining a different aspect of the structure of a tree
sequence. These tables are described individually in
{ref}`the next section<sec_tables_api_tables>`. However, these are interrelated,
and so many operations work
on the entire collection of tables, known as a `TableCollection`.
The {class}`TableCollection` and {class}`TreeSequence` classes are
deeply related. A `TreeSequence` instance is based on the information
encoded in a `TableCollection`. Tree sequences are **immutable**, and
provide methods for obtaining trees from the sequence. A `TableCollection`
is **mutable**, and does not have any methods for obtaining trees.
The `TableCollection` class thus allows dynamic creation and modification of
tree sequences.


(sec_tables_api_table_collection)=

### The `TableCollection` class

Note that several `TableCollection` methods have identical names to methods of a
`TreeSequence`. These tree sequence methods are often a simple wrapper around the
equivalent table collection method, acting on a copy of the tree sequence's table
collection.

```{eval-rst}
.. autoclass:: TableCollection(sequence_length=0)
    :members:
    :autosummary:
```

(sec_tables_api_tables)=

### Tables

The {ref}`tables API <sec_binary_interchange>` provides an efficient way of working
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


(sec_tables_api_text_columns)=

#### Text columns

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

(sec_tables_api_binary_columns)=

#### Binary columns

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

### Table classes

This section describes the methods and variables available for each
table class. For description and definition of each table's meaning
and use, see {ref}`the table definitions <sec_table_definitions>`.

% Overriding the default signatures for the tables here as they will be
% confusing to most users.

```{eval-rst}
.. autoclass:: IndividualTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

```{eval-rst}
.. autoclass:: NodeTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

```{eval-rst}
.. autoclass:: EdgeTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

```{eval-rst}
.. autoclass:: MigrationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

```{eval-rst}
.. autoclass:: SiteTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

```{eval-rst}
.. autoclass:: MutationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

```{eval-rst}
.. autoclass:: PopulationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__
```

```{eval-rst}
.. autoclass:: ProvenanceTable()
    :members:
    :inherited-members:
```

### Table functions

```{eval-rst}
.. autofunction:: parse_nodes
```

```{eval-rst}
.. autofunction:: parse_edges
```

```{eval-rst}
.. autofunction:: parse_sites
```

```{eval-rst}
.. autofunction:: parse_mutations
```

```{eval-rst}
.. autofunction:: parse_individuals
```

```{eval-rst}
.. autofunction:: parse_populations
```

```{eval-rst}
.. autofunction:: pack_strings
```

```{eval-rst}
.. autofunction:: unpack_strings
```

```{eval-rst}
.. autofunction:: pack_bytes
```

```{eval-rst}
.. autofunction:: unpack_bytes
```


(sec_constants_api)=

## Constants

The following constants are used throughout the `tskit` API.

```{eval-rst}
.. autodata:: NULL
```

```{eval-rst}
.. autodata:: NODE_IS_SAMPLE
```

```{eval-rst}
.. autodata:: MISSING_DATA
```

```{eval-rst}
.. autodata:: FORWARD
```

```{eval-rst}
.. autodata:: REVERSE
```

```{eval-rst}
.. autodata:: ALLELES_ACGT
```

```{eval-rst}
.. autodata:: UNKNOWN_TIME
```


(sec_metadata_api)=

## Metadata API

The `metadata` module provides validation, encoding and decoding of metadata
using a schema. See {ref}`sec_metadata`, {ref}`sec_metadata_api_overview` and
{ref}`sec_tutorial_metadata`.

```{eval-rst}
.. autoclass:: MetadataSchema
    :members:
    :inherited-members:
```

```{eval-rst}
.. autofunction:: register_metadata_codec
```

(sec_combinatorics_api)=

## Combinatorics API

The combinatorics API deals with tree topologies, allowing them to be counted,
listed and generated: see {ref}`sec_combinatorics` for a detailed description. Briefly,
the position of a tree in the enumeration `all_trees` can be obtained using the tree's
{meth}`~Tree.rank` method. Inversely, a {class}`Tree` can be constructed from a position
in the enumeration with {meth}`Tree.unrank`. Generated trees are associated with a new
tree sequence containing only that tree for the entire genome (i.e. with
{attr}`~TreeSequence.num_trees` = 1 and a {attr}`~TreeSequence.sequence_length` equal to
the {attr}`~Tree.span` of the tree).

```{eval-rst}
.. autofunction:: all_trees
```

```{eval-rst}
.. autofunction:: all_tree_shapes
```

```{eval-rst}
.. autofunction:: all_tree_labellings
```

```{eval-rst}
.. autoclass:: TopologyCounter
```

## Linkage disequilibrium

:::{note}
This API will soon be deprecated in favour of multi-site extensions
to the {ref}`sec_stats` API.
:::

```{eval-rst}
.. autoclass:: LdCalculator(tree_sequence)
    :members:
```

(sec_provenance_api)=

## Provenance

We provide some preliminary support for validating JSON documents against the
{ref}`provenance schema <sec_provenance>`. Programmatic access to provenance
information is planned for future versions.

```{eval-rst}
.. autofunction:: validate_provenance
```

```{eval-rst}
.. autoexception:: ProvenanceValidationError
```

(sec_utility_api)=

## Utility functions

Some top-level utility functions.

```{eval-rst}
.. autofunction:: is_unknown_time
```

```{eval-rst}
.. autofunction:: random_nucleotides
```
