.. _sec_c_api:

=====
C API
=====

.. warning::
        **This section is under construction, incomplete and experiemental!!**

***********
Basic Types
***********

.. doxygentypedef:: tsk_id_t
.. doxygentypedef:: tsk_size_t

.. doxygenstruct:: tsk_individual_t
    :members:

.. todo:: Need to document the members in these structs.

.. doxygenstruct:: tsk_node_t
    :members:
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


**********
Tables API
**********

This is an example of using the tables API to define a simple
haploid Wright-Fisher simulator.

.. literalinclude:: ../c/examples/haploid_wright_fisher.c
    :language: c

----------------
Table collection
----------------

.. doxygenstruct:: tsk_table_collection_t
    :members:

.. doxygengroup:: TABLE_COLLECTION_API_GROUP
    :content-only:

----------------
Individual table
----------------

.. doxygenstruct:: tsk_individual_table_t
    :members:

.. doxygengroup:: INDIVIDUAL_TABLE_API_GROUP
    :content-only:

----------
Node table
----------

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


**************
Tree sequences
**************

.. doxygenstruct:: tsk_treeseq_t
    :members:

.. doxygenfunction:: tsk_treeseq_alloc

.. doxygenfunction:: tsk_treeseq_load


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
