.. _sec_c_api:

=====
C API
=====

This is the documentation for the ``tskit`` C API, a low-level library
for manipulating and processing :ref:`tree sequence data <sec_data_model>`.
The library is written using the C99 standard and is fully thread safe.
Tskit uses `kastore <https://kastore.readthedocs.io/>`_ to define a
simple storage format for the tree sequence data.

To see the API in action, please see :ref:`sec_c_api_examples` section.

********
Overview
********

--------------------
Do I need the C API?
--------------------

The ``tskit`` C API is generally useful in the following situations:

- You want to use the ``tskit`` API in a larger C/C++ application (e.g.,
  in order to output data in the ``.trees`` format);
- You need to perform lots of tree traversals/loops etc. to analyse some
  data that is in tree sequence form.

For high level operations that are not performance sensitive, the :ref:`sec_python_api`
is generally more useful. Python is *much* more convenient that C,
and since the ``tskit`` Python module is essentially a wrapper for this
C library, there's often no real performance penalty for using it.

-------------------------------
Differences with the Python API
-------------------------------

Much of the explanatory material (for example tutorials) about the Python API applies to
the C-equivalent methods as the Python API wraps this API.

The main area of difference is, unlike the Python API, the C API doesn't do any
decoding, encoding or schema validation of :ref:`sec_metadata` fields,
instead only handling the byte sting representation of the metadata. Metadata is therefore
never used directly by any tskit C API method, just stored.

----------------------
API stability contract
----------------------

Since the C API 1.0 release we pledge to make **no** breaking changes
to the documented API in subsequent releases in the 1.0 series.
What this means is that any code that compiles under the 1.0 release
should also compile without changes in subsequent 1.x releases. We
will not change the semantics of documented functions,  unless it is to
fix clearly buggy behaviour. We will not change the values of macro
constants.

Undocumented functions do not have this guarantee, and may be changed
arbitrarily between releases.

.. note::
    We do not currently make any guarantees about
    `ABI <https://en.wikipedia.org/wiki/Application_binary_interface>`__
    stability, since the primary use-case is for tskit to be embedded
    within another application rather than used as a shared library. If you
    do intend to use tskit as a shared library and ABI stability is
    therefore imporant to you, please let us know and we can plan
    accordingly.

.. _sec_c_api_overview_structure:

-------------
API structure
-------------

Tskit uses a set of conventions to provide a pseudo object-oriented API. Each
'object' is represented by a C struct and has a set of 'methods'. This is
most easily explained by an example:

.. literalinclude:: ../c/examples/api_structure.c
    :language: c

In this program we create a :c:type:`tsk_edge_table_t` instance, add five rows
using :c:func:`tsk_edge_table_add_row`, print out its contents using the
:c:func:`tsk_edge_table_print_state` debugging method, and finally free
the memory used by the edge table object. We define this edge table
'class' by using some simple naming conventions which are adhered
to throughout ``tskit``. This is simply a naming convention that helps to
keep code written in plain C logically structured; there are no extra C++ style features.
We use object oriented terminology freely throughout this documentation
with this understanding.

In this convention, a class is defined by a struct ``tsk_class_name_t`` (e.g.
``tsk_edge_table_t``) and its methods all have the form ``tsk_class_name_method_name``
whose first argument is always a pointer to an instance of the class (e.g.,
``tsk_edge_table_add_row`` above).
Each class has an initialise and free method, called ``tsk_class_name_init``
and ``tsk_class_name_free``, respectively. The init method must
be called to ensure that the object is correctly initialised (except
for functions such as for :c:func:`tsk_table_collection_load`
and :c:func:`tsk_table_collection_copy` which automatically initialise
the object by default for convenience). The free
method must always be called to avoid leaking memory, even in the
case of an error occurring during initialisation. If ``tsk_class_name_init`` has
been called successfully, we say the object has been "initialised"; if not,
it is "uninitialised". After ``tsk_class_name_free`` has been called,
the object is again uninitialised.

It is important to note that the init methods only allocate *internal* memory;
the memory for the instance itself must be allocated either on the
heap or the stack:

.. code-block:: c

    // Instance allocated on the stack
    tsk_node_table_t nodes;
    tsk_node_table_init(&nodes, 0);
    tsk_node_table_free(&nodes);

    // Instance allocated on the heap
    tsk_edge_table_t *edges = malloc(sizeof(tsk_edge_table_t));
    tsk_edge_table_init(edges, 0);
    tsk_edge_table_free(edges);
    free(edges);


.. _sec_c_api_error_handling:

--------------
Error handling
--------------

C does not have a mechanism for propagating exceptions, and great care
must be taken to ensure that errors are correctly and safely handled.
The convention adopted in ``tskit`` is that
every function (except for trivial accessor methods) returns
an integer. If this return value is negative an error has occured which
must be handled. A description of the error that occured can be obtained
using the :c:func:`tsk_strerror` function. The following example illustrates
the key conventions around error handling in ``tskit``:

.. literalinclude:: ../c/examples/error_handling.c
    :language: c

In this example we load a tree sequence from file and print out a summary
of the number of nodes and edges it contains. After calling
:c:func:`tsk_treeseq_load` we check the return value ``ret`` to see
if an error occured. If an error has occured we exit with an error
message produced by :c:func:`tsk_strerror`. Note that in this example we call
:c:func:`tsk_treeseq_free` whether or not an error occurs: in general,
once a function that initialises an object (e.g., ``X_init``, ``X_copy``
or ``X_load``) is called, then ``X_free`` must
be called to ensure that memory is not leaked.

Most functions in ``tskit`` return an error status; we recommend that **every**
return value is checked.

.. _sec_c_api_memory_allocation_strategy:

--------------------------
Memory allocation strategy
--------------------------

To reduce the frequency of memory allocations tskit pre-allocates space for
additional table rows in each table, along with space for the contents of
ragged columns. The default behaviour is to start with space for 1,024 rows
in each table and 65,536 bytes in each ragged column. The table then grows
as needed by doubling, until a maximum pre-allocation of 2,097,152 rows for
a table or 104,857,600 bytes for a ragged column. This behaviour can be
disabled and a fixed increment used, on a per-table and per-ragged-column
basis using the ``tsk_X_table_set_max_rows_increment`` and
``tsk_provenance_table_set_max_X_length_increment`` methods where ``X`` is
the name of the table or column.

---------------------------
Using tskit in your project
---------------------------

Tskit is built as a standard C library and so there are many different ways
in which it can be included in downstream projects. It is possible to
install ``tskit`` onto a system (i.e., installing a shared library and
header files to a standard locations on Unix) and linking against it,
but there are many different ways in which this can go wrong. In the
interest of simplicity and improving the end-user experience we recommend
embedding ``tskit`` directly into your applications.

There are many different build systems and approaches to compiling
code, and so it's not possible to give definitive documentation on
how ``tskit`` should be included in downstream projects. Please
see the `build examples <https://github.com/tskit-dev/tskit-build-examples>`_
repo for some examples of how to incorporate ``tskit`` into
different project structures and build systems.

Tskit uses the `meson <https://mesonbuild.com>`_ build system internally,
and supports being used a `meson subproject <https://mesonbuild.com/Subprojects.html>`_.
We show an `example <https://github.com/tskit-dev/tskit-build-examples/tree/main/meson>`_
in which this is combined with the tskit distribution tarball to neatly
abstract many details of cross-platform C development.

Some users may choose to check the source for ``tskit`` directly into their source
control repositories. If you wish to do this, the code is in the ``c`` subdirectory of the
`tskit <https://github.com/tskit-dev/tskit/tree/main/c>`_ repo.
The following header files should be placed in the search path:
``subprojects/kastore/kastore.h``, ``tskit.h``, and ``tskit/*.h``.
The C files ``subprojects/kastore/kastore.c`` and ``tskit/*.c`` should be compiled.
For those who wish to minimise the size of their compiled binaries,
``tskit`` is quite modular, and C files can be omitted if not needed.
For example, if you are just using the :ref:`sec_c_api_tables_api` then
only the files ``tskit/core.[c,h]`` and ``tskit/tables.[c,h]`` are
needed.

However you include ``tskit`` in your project, however, please
ensure that it is a **released version**. Released versions are
tagged on GitHub using the convention ``C_{VERSION}``. The code
can either be downloaded from GitHub on the `releases page
<https://github.com/tskit-dev/tskit/releases>`_ where each release has a distribution
tarball for example
https://github.com/tskit-dev/tskit/releases/download/C_1.0.0/tskit-1.0.0.tar.xz
Alternatively the code can be checked out
using git. For example, to check out the ``C_1.0.0`` release::

    $ git clone https://github.com/tskit-dev/tskit.git
    $ cd tskit
    $ git checkout C_1.0.0



***********
Basic Types
***********

.. doxygentypedef:: tsk_id_t
.. doxygentypedef:: tsk_size_t
.. doxygentypedef:: tsk_flags_t
.. doxygentypedef:: tsk_bool_t

**************
Common options
**************

.. doxygengroup:: GENERIC_FUNCTION_OPTIONS
   :content-only:

**********
Tables API
**********

The tables API section of ``tskit`` is defined in the ``tskit/tables.h`` header.

-----------------
Table collections
-----------------

.. doxygenstruct:: tsk_table_collection_t
    :members:

.. doxygenstruct:: tsk_bookmark_t
    :members:

.. doxygengroup:: TABLE_COLLECTION_API_GROUP
    :content-only:

-----------
Individuals
-----------

.. doxygenstruct:: tsk_individual_t
    :members:

.. doxygenstruct:: tsk_individual_table_t
    :members:

.. doxygengroup:: INDIVIDUAL_TABLE_API_GROUP
   :content-only:

-----
Nodes
-----

.. doxygenstruct:: tsk_node_t
    :members:

.. doxygenstruct:: tsk_node_table_t
    :members:

.. doxygengroup:: NODE_TABLE_API_GROUP
   :content-only:

-----
Edges
-----

.. doxygenstruct:: tsk_edge_t
    :members:

.. doxygenstruct:: tsk_edge_table_t
    :members:

.. doxygengroup:: EDGE_TABLE_API_GROUP
    :content-only:

----------
Migrations
----------

.. doxygenstruct:: tsk_migration_t
    :members:

.. doxygenstruct:: tsk_migration_table_t
    :members:

.. doxygengroup:: MIGRATION_TABLE_API_GROUP
    :content-only:

-----
Sites
-----

.. doxygenstruct:: tsk_site_t
    :members:

.. doxygenstruct:: tsk_site_table_t
    :members:

.. doxygengroup:: SITE_TABLE_API_GROUP
    :content-only:

---------
Mutations
---------

.. doxygenstruct:: tsk_mutation_t
    :members:

.. doxygenstruct:: tsk_mutation_table_t
    :members:

.. doxygengroup:: MUTATION_TABLE_API_GROUP
    :content-only:

-----------
Populations
-----------

.. doxygenstruct:: tsk_population_t
    :members:

.. doxygenstruct:: tsk_population_table_t
    :members:

.. doxygengroup:: POPULATION_TABLE_API_GROUP
    :content-only:

-----------
Provenances
-----------

.. doxygenstruct:: tsk_provenance_t
    :members:

.. doxygenstruct:: tsk_provenance_table_t
    :members:

.. doxygengroup:: PROVENANCE_TABLE_API_GROUP
    :content-only:


.. _sec_c_api_table_indexes:

-------------
Table indexes
-------------

Along with the tree sequence :ref:`ordering requirements
<sec_valid_tree_sequence_requirements>`, the :ref:`sec_table_indexes`
allow us to take a table collection and efficiently operate
on the trees defined within it. This section defines the rules
for safely operating on table indexes and their life-cycle.

The edge index used for tree generation consists of two arrays,
each holding ``N`` edge IDs (where ``N`` is the size of the edge
table). When the index is computed using
:c:func:`tsk_table_collection_build_index`, we store the current size
of the edge table along with the two arrays of edge IDs. The
function :c:func:`tsk_table_collection_has_index` then returns true
iff (a) both of these arrays are not NULL and (b) the stored
number of edges is the same as the current size of the edge table.

Updating the edge table does not automatically invalidate the indexes.
Thus, if we call :c:func:`tsk_edge_table_clear` on an edge table
which has an index, this index will still exist. However, it will
not be considered a valid index by
:c:func:`tsk_table_collection_has_index` because of the size mismatch.
Similarly for functions that increase the size of the table.
Note that it is possible then to have
:c:func:`tsk_table_collection_has_index` return true, but the index
is not actually valid, if, for example, the user has manipulated the
node and edge tables to describe a different topology, which happens
to have the same number of edges. The behaviour of methods that
use the indexes will be undefined in this case.

Thus, if you are manipulating an existing table collection that may
be indexed, it is always recommended to call
:c:func:`tsk_table_collection_drop_index` first.

.. _sec_c_api_tree_sequences:

**************
Tree sequences
**************

.. doxygenstruct:: tsk_treeseq_t
    :members:

.. doxygengroup:: TREESEQ_API_GROUP
    :content-only:

.. _sec_c_api_trees:

*****
Trees
*****

.. doxygenstruct:: tsk_tree_t
    :members:

---------
Lifecycle
---------

.. doxygengroup:: TREE_API_LIFECYCLE_GROUP
    :content-only:

.. _sec_c_api_trees_null:

----------
Null state
----------

Trees are initially in a "null state" where each sample is a
root and there are no branches. The ``index`` of a tree in the
null state is ``-1``.

We must call one of the
:ref:`seeking<sec_c_api_trees_seeking>` methods
to make the state of the tree object correspond to a particular tree
in the sequence.

.. _sec_c_api_trees_seeking:

-------
Seeking
-------

When we are examining many trees along a tree sequence,
we usually allocate a single :c:struct:`tsk_tree_t` object
and update its state. This allows us to efficiently transform
the state of a tree into nearby trees, using the underlying succinct tree
sequence data structure.

The simplest example to visit trees left-to-right along the genome:

.. code-block:: c
    :linenos:

    int
    visit_trees(const tsk_treeseq_t *ts)
    {
        tsk_tree_t tree;
        int ret;

        ret = tsk_tree_init(&tree, &ts, 0);
        if (ret != 0) {
            goto out;
        }
        for (ret = tsk_tree_first(&tree); ret == TSK_TREE_OK; ret = tsk_tree_next(&tree)) {
            printf("\ttree %lld covers interval left=%f right=%f\n",
                (long long) tree.index, tree.interval.left, tree.interval.right);
        }
        if (ret != 0) {
            goto out;
        }
        // Do other things in the function...
    out:
        tsk_tree_free(&tree);
        return ret;
    }


In this example we first initialise a :c:struct:`tsk_tree_t` object,
associating it with the input tree sequence. We then iterate over the
trees along the sequence using a ``for`` loop, with the ``ret`` variable
controlling iteration. The usage of ``ret`` here follows a slightly
different pattern to other functions in the tskit C API
(see the :ref:`sec_c_api_error_handling` section).
The interaction between error handling and states
of the ``tree`` object here is somewhat subtle, and is worth explaining
in detail.

After successful initialisation (after line 10), the tree is in the
:ref:`null state<sec_c_api_trees_null>` where all samples are roots.
The ``for`` loop begins by calling :c:func:`tsk_tree_first` which
transforms the state of the tree into the first (leftmost) tree in
the sequence. If this operation is successful, :c:func:`tsk_tree_first`
returns :c:data:`TSK_TREE_OK`. We then check the value of ``ret``
in the loop condition to see if it is equal
to :c:data:`TSK_TREE_OK` and execute the loop body for the
first tree in the sequence.

On completing the loop body for the first tree in the sequence,
we then execute the ``for`` loop increment operation, which
calls :c:func:`tsk_tree_next` and assigns the returned value to
``ret``. This function efficiently transforms the current state
of ``tree`` so that it represents the next tree along the genome,
and returns :c:data:`TSK_TREE_OK` if the operation succeeds.
When :c:func:`tsk_tree_next` is called on the last tree in the
sequence, the state of ``tree`` is set back to the
:ref:`null state<sec_c_api_trees_null>` and the return value is 0.

Thus, the loop on lines 11-14 can exit in two ways:

1. Either we successfully iterate over all trees in the sequence and
   ``ret`` has the value ``0`` at line 15; or
2. An error occurs during :c:func:`tsk_tree_first` or
   :c:func:`tsk_tree_next`, and ret contains a negative value.

.. warning::
    It is **vital** that you check the value of ``ret`` immediately
    after the loop exits like we do here at line 15, or errors can be silently
    lost. (Although it's redundant here, as we don't do anything else in the
    function.)

.. seealso::
    See the :ref:`examples<sec_c_api_examples_tree_iteration>` section for
    more examples of sequential seeking, including
    an example of using
    use :c:func:`tsk_tree_last` and :c:func:`tsk_tree_prev`
    to iterate from right-to-left.

.. note::
    Seeking functions
    :c:func:`tsk_tree_first`,
    :c:func:`tsk_tree_last`,
    :c:func:`tsk_tree_next`
    :c:func:`tsk_tree_prev`,
    and :c:func:`tsk_tree_seek`
    can be called in any order and from any non-error state.

.. doxygengroup:: TREE_API_SEEKING_GROUP
    :content-only:

------------
Tree queries
------------

.. doxygengroup:: TREE_API_TREE_QUERY_GROUP
    :content-only:

------------
Node queries
------------

.. doxygengroup:: TREE_API_NODE_QUERY_GROUP
    :content-only:

----------------
Traversal orders
----------------

.. doxygengroup:: TREE_API_TRAVERSAL_GROUP
    :content-only:


.. _sec_c_api_low_level_sorting:

*****************
Low-level sorting
*****************

In some highly performance sensitive cases it can be useful to
have more control over the process of sorting tables. This low-level
API allows a user to provide their own edge sorting function.
This can be useful, for example, to use parallel sorting algorithms,
or to take advantage of the more efficient sorting procedures
available in C++. It is the user's responsibility to ensure that the
edge sorting requirements are fulfilled by this function.

.. todo::
    Create an idiomatic C++11 example where we load a table collection
    file from argv[1], and sort the edges  using std::sort, based
    on the example in tests/test_minimal_cpp.cpp. We can include
    this in the examples below, and link to it here.

.. doxygenstruct:: _tsk_table_sorter_t
    :members:

.. doxygengroup:: TABLE_SORTER_API_GROUP
    :content-only:

******************
Decoding genotypes
******************

Obtaining genotypes for samples at specific sites is achieved via
:c:struct:`tsk_variant_t` and its methods.

.. doxygenstruct:: tsk_variant_t
    :members:

.. doxygengroup:: VARIANT_API_GROUP
    :content-only:


***********************
Miscellaneous functions
***********************

.. doxygenfunction:: tsk_strerror

.. doxygenfunction:: tsk_is_unknown_time


*************************
Function Specific Options
*************************

-------------
Load and init
-------------
.. doxygengroup:: API_FLAGS_LOAD_INIT_GROUP
    :content-only:

--------------------------
:c:func:`tsk_treeseq_init`
--------------------------
.. doxygengroup:: API_FLAGS_TS_INIT_GROUP
    :content-only:

-----------------------------------------------------------------------
:c:func:`tsk_treeseq_simplify`, :c:func:`tsk_table_collection_simplify`
-----------------------------------------------------------------------
.. doxygengroup:: API_FLAGS_SIMPLIFY_GROUP
    :content-only:

----------------------------------------------
:c:func:`tsk_table_collection_check_integrity`
----------------------------------------------
.. doxygengroup:: API_FLAGS_CHECK_INTEGRITY_GROUP
    :content-only:

------------------------------------
:c:func:`tsk_table_collection_clear`
------------------------------------
.. doxygengroup:: API_FLAGS_CLEAR_GROUP
    :content-only:

-----------------------------------
:c:func:`tsk_table_collection_copy`
-----------------------------------
.. doxygengroup:: API_FLAGS_COPY_GROUP
    :content-only:

----------------------
All equality functions
----------------------
.. doxygengroup:: API_FLAGS_CMP_GROUP
    :content-only:

-------------------------------------
:c:func:`tsk_table_collection_subset`
-------------------------------------
.. doxygengroup:: API_FLAGS_SUBSET_GROUP
    :content-only:

------------------------------------
:c:func:`tsk_table_collection_union`
------------------------------------
.. doxygengroup:: API_FLAGS_UNION_GROUP
    :content-only:


*********
Constants
*********

-----------
API Version
-----------

.. doxygengroup:: API_VERSION_GROUP
    :content-only:

.. _sec_c_api_error_codes:

----------------
Common constants
----------------

.. doxygengroup:: GENERIC_CONSTANTS
   :content-only:

.. _sec_c_api_tables_api:

--------------
Generic Errors
--------------

.. doxygengroup:: GENERAL_ERROR_GROUP
        :content-only:

------------------
File format errors
------------------

.. doxygengroup:: FILE_FORMAT_ERROR_GROUP
        :content-only:

--------------------
Out-of-bounds errors
--------------------

.. doxygengroup:: OOB_ERROR_GROUP
        :content-only:

-----------
Edge errors
-----------

.. doxygengroup:: EDGE_ERROR_GROUP
        :content-only:


-----------
Site errors
-----------

.. doxygengroup:: SITE_ERROR_GROUP
        :content-only:


---------------
Mutation errors
---------------

.. doxygengroup:: MUTATION_ERROR_GROUP
        :content-only:


----------------
Migration errors
----------------

.. doxygengroup:: MIGRATION_ERROR_GROUP
        :content-only:

-------------
Sample errors
-------------

.. doxygengroup:: SAMPLE_ERROR_GROUP
        :content-only:

------------
Table errors
------------

.. doxygengroup:: TABLE_ERROR_GROUP
        :content-only:

------------------------
Genotype decoding errors
------------------------

.. doxygengroup:: GENOTYPE_ERROR_GROUP
        :content-only:

------------
Union errors
------------

.. doxygengroup:: UNION_ERROR_GROUP
        :content-only:

---------------
Simplify errors
---------------

.. doxygengroup:: SIMPLIFY_ERROR_GROUP
        :content-only:

-----------------
Individual errors
-----------------

.. doxygengroup:: INDIVIDUAL_ERROR_GROUP
        :content-only:

-------------------
Extend edges errors
-------------------

.. doxygengroup:: EXTEND_EDGES_ERROR_GROUP
        :content-only:


.. _sec_c_api_examples:

********
Examples
********

------------------------
Basic forwards simulator
------------------------

This is an example of using the tables API to define a simple
haploid Wright-Fisher simulator. Because this simple example
repeatedly sorts the edge data, it is quite inefficient and
should not be used as the basis of a large-scale simulator.

.. note::

   This example uses the C function ``rand`` and constant
   ``RAND_MAX`` for random number generation.  These methods
   are used for example purposes only and a high-quality
   random number library should be preferred for code
   used for research.  Examples include, but are not
   limited to:

   1. The `GNU Scientific Library <https://www.gnu.org/software/gsl>`_,
      which is licensed under the GNU General Public License, version
      3 (`GPL3+ <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.
   2. For C++ projects using C++11 or later,
      the built-in `random <https://en.cppreference.com/w/cpp/numeric/random>`_
      number library.
   3. The `numpy C API <https://numpy.org/devdocs/reference/random/c-api.html>`_
      may be useful for those writing Python extension modules in C/C++.

.. todo::
    Give a pointer to an example that caches and flushes edge data efficiently.
    Probably using the C++ API?

.. literalinclude:: ../c/examples/haploid_wright_fisher.c
    :language: c

.. _sec_c_api_examples_tree_iteration:

--------------
Tree iteration
--------------

.. literalinclude:: ../c/examples/tree_iteration.c
    :language: c


.. _sec_c_api_examples_tree_traversals:

---------------
Tree traversals
---------------

In this example we load a tree sequence file, and then traverse the first
tree in four different ways:

1. We first traverse the tree in preorder and postorder using the
   :c:func:`tsk_tree_preorder`
   :c:func:`tsk_tree_postorder` functions to fill an array of
   nodes in the appropriate orders. This is the recommended approach
   and will be convenient and efficient for most purposes.

2. As an example of how we might build our own traveral algorithms, we
   then traverse the tree in preorder using recursion. This is a very
   common way of navigating around trees and can be convenient for
   some applications. For example, here we compute the depth of each node
   (i.e., it's distance from the root) and use this when printing out the
   nodes as we visit them.

3. Then we traverse the tree in preorder using an iterative approach. This
   is a little more efficient than using recursion, and is sometimes
   more convenient than structuring the calculation recursively.

4. In the third example we iterate upwards from the samples rather than
   downwards from the root.

.. literalinclude:: ../c/examples/tree_traversal.c
    :language: c

.. _sec_c_api_examples_file_streaming:

--------------
File streaming
--------------

It is often useful to read tree sequence files from a stream rather than
from a fixed filename. This example shows how to do this using the
:c:func:`tsk_table_collection_loadf` and
:c:func:`tsk_table_collection_dumpf` functions. Here, we sequentially
load table collections from the ``stdin`` stream and write them
back out to ``stdout`` with their mutations removed.

.. literalinclude:: ../c/examples/streaming.c
    :language: c

Note that we use the value :c:macro:`TSK_ERR_EOF` to detect when the stream
ends, as we don't know how many tree sequences to expect on the input.
In this case, :c:macro:`TSK_ERR_EOF` is not considered an error and we exit
normally.

Running this program on some tree sequence files we might get::

    $ cat tmp1.trees tmp2.trees | ./build/streaming > no_mutations.trees
    Tree sequence 0 had 38 mutations
    Tree sequence 1 had 132 mutations

Then, running this program again on the output of the previous command,
we see that we now have two tree sequences with their mutations removed
stored in the file ``no_mutations.trees``::

    $ ./build/streaming < no_mutations.trees > /dev/null
    Tree sequence 0 had 0 mutations
    Tree sequence 1 had 0 mutations
