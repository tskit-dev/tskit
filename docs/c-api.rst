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
- You need to perform lots of tree traversals/loops etc to analyse some
  data that is in tree sequence form.

For high level operations that are not performance sensitive, the :ref:`sec_python_api`
is generally more useful. Python is *much* more convenient that C,
and since the ``tskit`` Python module is essentially a wrapper for this
C library, there's often no real performance penalty for using it.

.. _sec_c_api_overview_structure:

-------------
API structure
-------------

Tskit uses a set of conventions to provide a pseudo object oriented API. Each
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

In this convention, a class is defined by a struct ``class_name_t`` (e.g.
``edge_table_t``) and its methods all have the form ``class_name_method_name``
whose first argument is always a pointer to an instance of the class (e.g.,
``edge_table_add_row`` above).
Each class has an initialise and free method, called ``class_name_init``
and ``class_name_free``, respectively. The init method must
be called to ensure that the object is correctly initialised (except
for functions such as for :c:func:`tsk_table_collection_load`
and :c:func:`tsk_table_collection_copy` which automatically initialise
the object by default for convenience). The free
method must always be called to avoid leaking memory, even in the
case of an error occuring during intialisation. If ``class_name_init`` has
been called succesfully, we say the object has been "initialised"; if not,
it is "uninitialised". After ``class_name_free`` has been called,
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
We show an `example <https://github.com/tskit-dev/tskit-build-examples/tree/master/meson>`_
in which this is combined with
`git submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_ to neatly
abstract many details of cross platform C development.

Some users may choose to check the source for ``tskit`` (and ``kastore``)
directly into their source control repositories. If you wish to do this,
the code is in the ``c`` subdirectory of the
`tskit <https://github.com/tskit-dev/tskit/tree/master/c>`_ and
`kastore <https://github.com/tskit-dev/kastore/tree/master/c>`__ repos.
The following header files should be placed in the search path:
``kastore.h``, ``tskit.h``, and ``tskit/*.h``.
The C files ``kastore.c`` and ``tskit*.c`` should be compiled.
For those who wish to minimise the size of their compiled binaries,
``tskit`` is quite modular, and C files can be omitted if not needed.
For example, if you are just using the :ref:`sec_c_api_tables_api` then
only the files ``tskit/core.[c,h]`` and ``tskit/tables.[c,h]`` are
needed.

However you include ``tskit`` in your project, however, please
ensure that it is a **released version**. Released versions are
tagged on GitHub using the convention ``C_{VERSION}``. The code
can either be downloaded from GitHub on the `releases page
<https://github.com/tskit-dev/tskit/releases>`_ or checked out
using git. For example, to check out the ``C_0.99.1`` release::

    $ git clone https://github.com/tskit-dev/tskit.git
    $ cd tskit
    $ git checkout C_0.99.1

Git submodules may also be considered---see the
`example <https://github.com/tskit-dev/tskit-build-examples/tree/master/meson>`_
for how to set these up and to check out at a specific release.


***********
Basic Types
***********

.. doxygentypedef:: tsk_id_t
.. doxygentypedef:: tsk_size_t
.. doxygentypedef:: tsk_flags_t

**************
Common options
**************

.. doxygengroup:: TABLES_API_FUNCTION_OPTIONS
   :content-only:

.. _sec_c_api_tables_api:

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


.. _sec_c_api_tree_sequences:

**************
Tree sequences
**************

.. warning:: This part of the API is more preliminary and may be subject to change.

.. doxygenstruct:: tsk_treeseq_t
    :members:

.. doxygengroup:: TREESEQ_API_GROUP
    :content-only:

*****
Trees
*****

.. warning:: This part of the API is more preliminary and may be subject to change.

.. doxygenstruct:: tsk_tree_t
    :members:

.. doxygengroup:: TREE_API_GROUP
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


***********************
Miscellaneous functions
***********************

.. doxygenfunction:: tsk_strerror

*********
Constants
*********

-----------
API Version
-----------

.. doxygengroup:: API_VERSION_GROUP
    :content-only:

.. _sec_c_api_error_codes:

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


.. todo:: Add in groups for rest of the error types and document.



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
tree in three different ways:

1. We first traverse the tree in preorder using recursion. This is a very
   common way of navigating around trees and can be very convenient for
   some applications. For example, here we compute the depth of each node
   (i.e., it's distance from the root) and use this when printing out the
   nodes as we visit them.

2. Then we traverse the tree in preorder using an iterative approach. This
   is a little more efficient than using recursion, and is sometimes
   more convenient than structuring the calculation recursively. Note that
   we allocate a stack here with space to hold the total number of nodes
   in the tree sequence. This is safe, but it likely to be a massive
   over estimate. However, this makes very little difference in practise
   even for tree sequences with millions of nodes since it's likely
   only the first page (usually 4K) will be written to and the
   rest of the stack will never therefore be mapped to physical memory.

3. In the third example we iterate upwards from the samples rather than
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
