.. _sec_c_api:

=====
C API
=====

.. warning::
        **This section is under construction, incomplete and experiemental!!**

**********
Tables API
**********

.. doxygenstruct:: tsk_tbl_collection_t
    :members:

.. doxygenfunction:: tsk_tbl_collection_alloc

.. doxygenfunction:: tsk_tbl_collection_load

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
