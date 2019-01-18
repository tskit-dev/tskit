.. _sec_c_api:

=====
C API
=====

.. warning::
        **This section is under construction, incomplete and experiemental!!**

********
Overview
********

The ``tskit`` C API is a low-level library for manipulating and processing
:ref:`tree sequence data <sec_data_model>`.

For high level operations
and operations that are not performance sensitive, the :ref:`sec_python_api`
is much more useful. The Python code uses this C API under the hood and
so there's often no real performance penalty for using Python, and Python
is *much* more convenient than C. This C API is useful in the following
situtations:

- You want to use the ``tskit`` API in a larger C/C++ application;
- You need to do lots of tree traversals/loops etc to analyse some data.

The library is written using the C99 standard and is fully thread safe.

---------------------------
Using tskit in your project
---------------------------

Tskit is intended to be embedded directly into applications that use it.
That is, rather than linking against a shared ``tskit`` library, applications
compile and embed their own copy of ``tskit``. As ``tskit`` is quite small, consisting
of only a handful of C files, this is much more convenient and avoids many
of the headaches caused by shared libraries.

The simplest way to include ``tskit`` in your C/C++ project is to
use git submodule.

.. todo:: Set up an example project repo on GitHub and go through
    the steps of getting the submodule set up properly.


If you don't use git (or prefer not to use submodules), then you can simply
copy the ``tskit`` and ``kastore`` source files into your own repository.
Please ensure that the files you use correspond to a **released version**
of the API by checking out the appropriate tag on git.

We may distribute ``tskit`` as a shared library in the future, however.

-----------------
Code organisation
-----------------

Tskit is organised in a modular way, allowing users to pick and choose which
parts of the library that they compile into their application. The functionality
defined in each header file corresponds to one C file, giving fine-grained access
to the functionality that is required for different applications.

Core functionality such as error handling required by all of ``tskit`` is
defined in ``tsk_core.[c,h]``. Client code should not need to include ``tsk_core.h``
as it is included in all other ``tskit`` headers.

The :ref:`sec_c_api_tables_api` is defined in ``tsk_tables.[c,h]``. Tree sequence
and tree :ref:`functionality <sec_c_api_tree_sequences>` is defined in
``tsk_trees.[c,h]``.

.. todo:: When the remaining types have been finalised and documented add the
    descriptions in here.

For convenience, there is also a ``tskit.h`` header file that includes all
of the functionality in ``tskit``.


.. _sec_c_api_overview_structure:

-------------
API structure
-------------

Tskit uses a set of conventions to provide pseudo object oriented API. Each
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
Each class has an allocator and deallocator, called ``class_name_alloc``
and ``class_name_free``, respectively. The allocator method must
be called to ensure that the object is correctly initialised (except
for :c:func:`tsk_table_collection_load` and :c:func:`tsk_treeseq_load`
which call the allocator internally for convenience). The deallocator
method must always be called to avoid leaking memory, even in the
case of an error occuring in the allocator. If ``class_name_alloc`` has
been called, we say the object has been "initialised"; if not,
it is "uninitialised".

It is important to note that the allocator methods only allocate *internal* memory;
the memory for the instance itself must be allocated either on the
heap or the stack:

.. code-block:: c

    // Instance allocated on the stack
    tsk_node_table_t nodes;
    tsk_node_table_alloc(&nodes, 0);
    tsk_node_table_free(&nodes);

    // Instance allocated on the heap
    tsk_edge_table_t *edges = malloc(sizeof(tsk_edge_table_t));
    tsk_edge_table_alloc(edges, 0);
    tsk_edge_table_free(edges);
    free(edges);


--------------
Error handling
--------------

Every function in ``tskit`` (except for trivial accessor methods) returns
an integer. If this return value is negative an error has occured which
must be handled.

.. literalinclude:: ../c/examples/error_handling.c
    :language: c

In this example we load a tree sequence from file and print out a summary
of the number of nodes and edges it contains. After calling
:c:func:`tsk_treeseq_load` we check it's return value ``ret`` to see
if an error occured. If an error happens we with an error message produced with
:c:func:`tsk_strerror`. Note that in this example we call
``tsk_treeseq_free`` whether or not an error occurs: in general,
once ``X_alloc`` (or ``load`` here) is called ``X_free`` must also
be called to ensure that memory is not leaked.

Most functions in ``tskit`` can return an error status, and we
**strongly** recommend that every return value is checked.

***********
Basic Types
***********

.. doxygentypedef:: tsk_id_t
.. doxygentypedef:: tsk_size_t
.. doxygentypedef:: tsk_flags_t



.. doxygenstruct:: tsk_edge_t
    :members:
.. doxygenstruct:: tsk_migration_t
    :members:
.. doxygenstruct:: tsk_site_t
    :members:
.. doxygenstruct:: tsk_mutation_t
    :members:
.. doxygenstruct:: tsk_population_t
    :members:
.. doxygenstruct:: tsk_provenance_t
    :members:


.. _sec_c_api_tables_api:

**********
Tables API
**********

The tables API section of ``tskit`` is defined in ``tsk_tables.h``.

-----------------
Table collections
-----------------

.. doxygenstruct:: tsk_table_collection_t
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

----------
Edge table
----------

.. doxygenstruct:: tsk_edge_table_t
    :members:

.. doxygengroup:: EDGE_TABLE_API_GROUP
    :content-only:

---------------
Migration table
---------------

.. doxygenstruct:: tsk_migration_table_t
    :members:

.. doxygengroup:: MIGRATION_TABLE_API_GROUP
    :content-only:

----------
Site table
----------

.. doxygenstruct:: tsk_site_table_t
    :members:

.. doxygengroup:: SITE_TABLE_API_GROUP
    :content-only:

--------------
Mutation table
--------------

.. doxygenstruct:: tsk_mutation_table_t
    :members:

.. doxygengroup:: MUTATION_TABLE_API_GROUP
    :content-only:

----------------
Population table
----------------

.. doxygenstruct:: tsk_population_table_t
    :members:

.. doxygengroup:: POPULATION_TABLE_API_GROUP
    :content-only:

----------------
Provenance table
----------------

.. doxygenstruct:: tsk_provenance_table_t
    :members:

.. doxygengroup:: PROVENANCE_TABLE_API_GROUP
    :content-only:


.. _sec_c_api_tree_sequences:

**************
Tree sequences
**************

.. doxygenstruct:: tsk_treeseq_t
    :members:

.. doxygengroup:: TREESEQ_API_GROUP
    :content-only:

*****
Trees
*****

.. doxygenstruct:: tsk_tree_t
    :members:

.. doxygengroup:: TREE_API_GROUP
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




********
Examples
********

This is an example of using the tables API to define a simple
haploid Wright-Fisher simulator.

.. literalinclude:: ../c/examples/haploid_wright_fisher.c
    :language: c

