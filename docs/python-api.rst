.. currentmodule:: tskit
.. _sec_python_api:

==========
Python API
==========

This page provides detailed documentation for the ``tskit`` Python API.

.. _sec_python_api_trees_and_tree_sequences:

************************
Trees and tree sequences
************************

The :class:`TreeSequence` class represents a sequence of correlated
evolutionary trees along a genome. The :class:`Tree` class represents a
single tree in this sequence. These classes are the interfaces used to
interact with the trees and mutational information stored in a tree sequence,
for example as returned from a simulation or inferred from a set of DNA
sequences. This library also provides methods for loading stored tree
sequences, for example using :func:`tskit.load`.


++++++++++++++++++++++++++
The ``TreeSequence`` class
++++++++++++++++++++++++++

.. autoclass:: TreeSequence()
    :members:
    :autosummary:


++++++++++++++++++
The ``Tree`` class
++++++++++++++++++

Trees in a tree sequence can be obtained by iterating over
:meth:`TreeSequence.trees` and specific trees can be accessed using methods
such as :meth:`TreeSequence.first`, :meth:`TreeSequence.at` and
:meth:`TreeSequence.at_index`. Each tree is an instance of the following
class which provides methods, for example, to access information
about particular nodes in the tree.

.. autoclass:: Tree()
    :members:
    :autosummary:

+++++++++
Constants
+++++++++

.. autodata:: NULL
    :annotation: = -1

.. autodata:: NODE_IS_SAMPLE
    :annotation: = 1

.. autodata:: MISSING_DATA
    :annotation: = -1

.. autodata:: FORWARD
    :annotation: = 1

.. autodata:: REVERSE
    :annotation: = -1

.. autodata:: ALLELES_ACGT


++++++++++++++++++++++++
Simple container classes
++++++++++++++++++++++++

These classes are simple shallow containers representing the entities defined
in the :ref:`sec_data_model_definitions`. These classes are not intended to be instantiated
directly, but are the return types for the various iterators provided by the
:class:`TreeSequence` and :class:`Tree` classes.

.. autoclass:: Individual()
    :members:

.. autoclass:: Node()
    :members:

.. autoclass:: Edge()
    :members:

.. autoclass:: Interval()
    :members:

.. autoclass:: Site()
    :members:

.. autoclass:: Mutation()
    :members:

.. autoclass:: Variant()
    :members:

.. autoclass:: Migration()
    :members:

.. autoclass:: Population()
    :members:

.. autoclass:: Provenance()
    :members:

++++++++++++
Loading data
++++++++++++

There are several methods for loading data into a :class:`TreeSequence`
instance. The simplest and most convenient is the use the :func:`tskit.load`
function to load a :ref:`tree sequence file <sec_tree_sequence_file_format>`. For small
scale data and debugging, it is often convenient to use the
:func:`tskit.load_text` to read data in the :ref:`text file format
<sec_text_file_format>`. The :meth:`TableCollection.tree_sequence` function
efficiently creates a :class:`TreeSequence` object from a set of tables
using the :ref:`Tables API <sec_tables_api>`.


.. autofunction:: load

.. autofunction:: load_text


.. _sec_tables_api:

****************************
Tables and Table Collections
****************************

The information required to construct a tree sequence is stored in a collection
of *tables*, each defining a different aspect of the structure of a tree
sequence. These tables are described individually in :ref:`the next section
<sec_tables_api_tables>`. However, these are interrelated, and so many operations work
on the entire collection of tables, known as a ``TableCollection``.
The :class:`TableCollection` and :class:`TreeSequence` classes are
deeply related. A ``TreeSequence`` instance is based on the information
encoded in a ``TableCollection``. Tree sequences are **immutable**, and
provide methods for obtaining trees from the sequence. A ``TableCollection``
is **mutable**, and does not have any methods for obtaining trees.
The ``TableCollection`` class thus allows dynamic creation and modification of
tree sequences.

+++++++++++++++++++++++++++++
The ``TableCollection`` class
+++++++++++++++++++++++++++++

Many of the ``TreeSequence`` methods that return a modified tree sequence
are in fact wrappers around a corresponding ``TableCollection`` method
that modifies a copy of the origin tree sequence's table collection.

.. autoclass:: TableCollection(sequence_length=0)
    :members:
    :autosummary:


.. _sec_tables_api_tables:

++++++
Tables
++++++

The :ref:`tables API <sec_binary_interchange>` provides an efficient way of working
with and interchanging :ref:`tree sequence data <sec_data_model>`. Each table
class (e.g, :class:`NodeTable`, :class:`EdgeTable`) has a specific set of
columns with fixed types, and a set of methods for setting and getting the data
in these columns. The number of rows in the table ``t`` is given by ``len(t)``.
Each table supports accessing the data either by row or column. To access the
row ``j`` in table ``t`` simply use ``t[j]``. The value returned by such an
access is an instance of :func:`collections.namedtuple`, and therefore supports
either positional or named attribute access. To access the data in
a column, we can use standard attribute access which will return a numpy array
of the data. For example::

    >>> import tskit
    >>> t = tskit.EdgeTable()
    >>> t.add_row(left=0, right=1, parent=10, child=11)
    0
    >>> t.add_row(left=1, right=2, parent=9, child=11)
    1
    >>> print(t)
    id      left            right           parent  child
    0       0.00000000      1.00000000      10      11
    1       1.00000000      2.00000000      9       11
    >>> t[0]
    EdgeTableRow(left=0.0, right=1.0, parent=10, child=11)
    >>> t[-1]
    EdgeTableRow(left=1.0, right=2.0, parent=9, child=11)
    >>> t.left
    array([ 0.,  1.])
    >>> t.parent
    array([10,  9], dtype=int32)
    >>> len(t)
    2
    >>>

Tables also support the :mod:`pickle` protocol, and so can be easily
serialised and deserialised (for example, when performing parallel
computations using the :mod:`multiprocessing` module). ::

    >>> serialised = pickle.dumps(t)
    >>> t2 = pickle.loads(serialised)
    >>> print(t2)
    id      left            right           parent  child
    0       0.00000000      1.00000000      10      11
    1       1.00000000      2.00000000      9       11

However, pickling will not be as efficient as storing tables
in the native :ref:`format <sec_tree_sequence_file_format>`.

Tables support the equality operator ``==`` based on the data
held in the columns::

    >>> t == t2
    True
    >>> t is t2
    False
    >>> t2.add_row(0, 1, 2, 3)
    2
    >>> print(t2)
    id      left            right           parent  child
    0       0.00000000      1.00000000      10      11
    1       1.00000000      2.00000000      9       11
    2       0.00000000      1.00000000      2       3
    >>> t == t2
    False



.. _sec_tables_api_text_columns:

++++++++++++
Text columns
++++++++++++

As described in the :ref:`sec_encoding_ragged_columns`, working with
variable length columns is somewhat more involved. Columns
encoding text data store the **encoded bytes** of the flattened
strings, and the offsets into this column in two separate
arrays.

Consider the following example::

    >>> t = tskit.SiteTable()
    >>> t.add_row(0, "A")
    >>> t.add_row(1, "BB")
    >>> t.add_row(2, "")
    >>> t.add_row(3, "CCC")
    >>> print(t)
    id      position        ancestral_state metadata
    0       0.00000000      A
    1       1.00000000      BB
    2       2.00000000
    3       3.00000000      CCC
    >>> t[0]
    SiteTableRow(position=0.0, ancestral_state='A', metadata=b'')
    >>> t[1]
    SiteTableRow(position=1.0, ancestral_state='BB', metadata=b'')
    >>> t[2]
    SiteTableRow(position=2.0, ancestral_state='', metadata=b'')
    >>> t[3]
    SiteTableRow(position=3.0, ancestral_state='CCC', metadata=b'')

Here we create a :class:`SiteTable` and add four rows, each with a different
``ancestral_state``. We can then access this information from each
row in a straightforward manner. Working with the data in the columns
is a little trickier, however::

    >>> t.ancestral_state
    array([65, 66, 66, 67, 67, 67], dtype=int8)
    >>> t.ancestral_state_offset
    array([0, 1, 3, 3, 6], dtype=uint32)
    >>> tskit.unpack_strings(t.ancestral_state, t.ancestral_state_offset)
    ['A', 'BB', '', 'CCC']

Here, the ``ancestral_state`` array is the UTF8 encoded bytes of the flattened
strings, and the ``ancestral_state_offset`` is the offset into this array
for each row. The :func:`tskit.unpack_strings` function, however, is a convient
way to recover the original strings from this encoding. We can also use the
:func:`tskit.pack_strings` to insert data using this approach::

    >>> a, off = tskit.pack_strings(["0", "12", ""])
    >>> t.set_columns(position=[0, 1, 2], ancestral_state=a, ancestral_state_offset=off)
    >>> print(t)
    id      position        ancestral_state metadata
    0       0.00000000      0
    1       1.00000000      12
    2       2.00000000

When inserting many rows with standard infinite sites mutations (i.e.,
ancestral state is "0"), it is more efficient to construct the
numpy arrays directly than to create a list of strings and use
:func:`pack_strings`. When doing this, it is important to note that
it is the **encoded** byte values that are stored; by default, we
use UTF8 (which corresponds to ASCII for simple printable characters).::

    >>> t_s = tskit.SiteTable()
    >>> m = 10
    >>> a = ord("0") + np.zeros(m, dtype=np.int8)
    >>> off = np.arange(m + 1, dtype=np.uint32)
    >>> t_s.set_columns(position=np.arange(m), ancestral_state=a, ancestral_state_offset=off)
    >>> print(t_s)
    id      position        ancestral_state metadata
    0       0.00000000      0
    1       1.00000000      0
    2       2.00000000      0
    3       3.00000000      0
    4       4.00000000      0
    5       5.00000000      0
    6       6.00000000      0
    7       7.00000000      0
    8       8.00000000      0
    9       9.00000000      0
    >>> t_s.ancestral_state
    array([48, 48, 48, 48, 48, 48, 48, 48, 48, 48], dtype=int8)
    >>> t_s.ancestral_state_offset
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=uint32)

Here we create 10 sites at regular positions, each with ancestral state equal to
"0". Note that we use ``ord("0")`` to get the ASCII code for "0" (48), and create
10 copies of this by adding it to an array of zeros. We have done this for
illustration purposes: it is equivalent (though slower for large examples) to do
``a, off = tskit.pack_strings(["0"] * m)``.

Mutations can be handled similarly::

    >>> t_m = tskit.MutationTable()
    >>> site = np.arange(m, dtype=np.int32)
    >>> d, off = tskit.pack_strings(["1"] * m)
    >>> node = np.zeros(m, dtype=np.int32)
    >>> t_m.set_columns(site=site, node=node, derived_state=d, derived_state_offset=off)
    >>> print(t_m)
    id      site    node    derived_state   parent  metadata
    0       0       0       1       -1
    1       1       0       1       -1
    2       2       0       1       -1
    3       3       0       1       -1
    4       4       0       1       -1
    5       5       0       1       -1
    6       6       0       1       -1
    7       7       0       1       -1
    8       8       0       1       -1
    9       9       0       1       -1
    >>>


.. _sec_tables_api_binary_columns:

++++++++++++++
Binary columns
++++++++++++++

Columns storing binary data take the same approach as
:ref:`sec_tables_api_text_columns` to encoding
:ref:`variable length data <sec_encoding_ragged_columns>`.
The difference between the two is only raw :class:`bytes` values are accepted: no
character encoding or decoding is done on the data. Consider the following example
where a table has no ``metadata_schema`` such that arbitrary bytes can be stored and
no automatic encoding or decoding of objects is performed by the Python API and we can
store and retrive raw ``bytes``. (See :ref:`sec_metadata` for details)::

    >>> t = tskit.NodeTable()
    >>> t.add_row(metadata=b"raw bytes")
    >>> t.add_row(metadata=pickle.dumps({"x": 1.1}))
    >>> t[0].metadata
    b'raw bytes'
    >>> t[1].metadata
    b'\x80\x03}q\x00X\x01\x00\x00\x00xq\x01G?\xf1\x99\x99\x99\x99\x99\x9as.'
    >>> pickle.loads(t[1].metadata)
    {'x': 1.1}
    >>> print(t)
    id      flags   population      time    metadata
    0       0       -1      0.00000000000000        cmF3IGJ5dGVz
    1       0       -1      0.00000000000000        gAN9cQBYAQAAAHhxAUc/8ZmZmZmZmnMu
    >>> t.metadata
    array([ 114,   97,  119,   32,   98,  121,  116,  101,  115, -128,    3,
            125,  113,    0,   88,    1,    0,    0,    0,  120,  113,    1,
             71,   63,  -15, -103, -103, -103, -103, -103, -102,  115,   46], dtype=int8)
    >>> t.metadata_offset
    array([ 0,  9, 33], dtype=uint32)


Here we add two rows to a :class:`NodeTable`, with different
:ref:`metadata <sec_metadata_definition>`. The first row contains a simple
byte string, and the second contains a Python dictionary serialised using
:mod:`pickle`. We then show several different (and seemingly incompatible!)
different views on the same data.

When we access the data in a row (e.g., ``t[0].metadata``) we are returned
a Python bytes object containing precisely the bytes that were inserted.
The pickled dictionary is encoded in 24 bytes containing unprintable
characters, and when we unpickle it using :func:`pickle.loads`, we obtain
the original dictionary.

When we print the table, however, we see some data which is seemingly
unrelated to the original contents. This is because the binary data is
`base64 encoded <https://en.wikipedia.org/wiki/Base64>`_ to ensure
that it is print-safe (and doesn't break your terminal). (See the
:ref:`sec_metadata_definition` section for more information on the
use of base64 encoding.).

Finally, when we print the ``metadata`` column, we see the raw byte values
encoded as signed integers. As for :ref:`sec_tables_api_text_columns`,
the ``metadata_offset`` column encodes the offsets into this array. So, we
see that the first metadata value is 9 bytes long and the second is 24.

The :func:`tskit.pack_bytes` and :func:`tskit.unpack_bytes` functions are
also useful for encoding data in these columns.

+++++++++++++
Table classes
+++++++++++++

This section describes the methods and variables available for each
table class. For description and definition of each table's meaning
and use, see :ref:`the table definitions <sec_table_definitions>`.

.. Overriding the default signatures for the tables here as they will be
.. confusing to most users.

.. autoclass:: IndividualTable()
    :members:
    :inherited-members:
    :special-members: __getitem__

.. autoclass:: NodeTable()
    :members:
    :inherited-members:
    :special-members: __getitem__

.. autoclass:: EdgeTable()
    :members:
    :inherited-members:
    :special-members: __getitem__

.. autoclass:: MigrationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__

.. autoclass:: SiteTable()
    :members:
    :inherited-members:
    :special-members: __getitem__

.. autoclass:: MutationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__

.. autoclass:: PopulationTable()
    :members:
    :inherited-members:
    :special-members: __getitem__

.. autoclass:: ProvenanceTable()
    :members:
    :inherited-members:

+++++++++++++++
Table functions
+++++++++++++++

.. autofunction:: parse_nodes

.. autofunction:: parse_edges

.. autofunction:: parse_sites

.. autofunction:: parse_mutations

.. autofunction:: parse_individuals

.. autofunction:: parse_populations

.. autofunction:: pack_strings

.. autofunction:: unpack_strings

.. autofunction:: pack_bytes

.. autofunction:: unpack_bytes

.. _sec_metadata_api:

************
Metadata API
************

The ``metadata`` module provides validation, encoding and decoding of metadata
using a schema. See :ref:`sec_metadata`, :ref:`sec_metadata_api_overview` and
:ref:`sec_tutorial_metadata`.

.. autoclass:: MetadataSchema
    :members:
    :inherited-members:

.. autofunction:: register_metadata_codec

.. _sec_combinatorics_api:

*****************
Combinatorics API
*****************

The combinatorics API deals with tree topologies, allowing them to be counted,
listed and generated: see :ref:`sec_combinatorics` for a detailed description. Briefly,
the position of a tree in the enumeration ``all_trees`` can be obtained using the tree's
:meth:`~Tree.rank` method. Inversely, a :class:`Tree` can be constructed from a position
in the enumeration with :meth:`Tree.unrank`. Generated trees are associated with a new
tree sequence containing only that tree for the entire genome (i.e. with
:attr:`~TreeSequence.num_trees` = 1 and a :attr:`~TreeSequence.sequence_length` equal to
the :attr:`~Tree.span` of the tree).

.. autofunction:: all_trees

.. autofunction:: all_tree_shapes

.. autofunction:: all_tree_labellings

.. autoclass:: TopologyCounter

**********************
Linkage disequilibrium
**********************

.. note:: This API will soon be deprecated in favour of multi-site extensions
    to the :ref:`sec_stats` API.

.. autoclass:: LdCalculator(tree_sequence)
    :members:


.. _sec_provenance_api:

**********
Provenance
**********

We provide some preliminary support for validating JSON documents against the
:ref:`provenance schema <sec_provenance>`. Programmatic access to provenance
information is planned for future versions.

.. autofunction:: validate_provenance

.. autoexception:: ProvenanceValidationError

