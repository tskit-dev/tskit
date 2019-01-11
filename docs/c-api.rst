.. _sec_c_api:

===================
C API Documentation
===================

This is the C API documentation for kastore.

.. todo:: Give a short example program.


******************
General principles
******************

--------------
Error handling
--------------

Functions return 0 to indicate success or an
:ref:`error code <sec_c_api_error_codes>` to indicate a failure condition.
Thus, the return value of all functions must be checked to ensure safety.

-------------
Array lengths
-------------

The length of arrays is specified in terms of the number of elements not bytes.

*********
Top level
*********

.. doxygenstruct:: kastore_t

.. doxygenfunction:: kastore_open
.. doxygenfunction:: kastore_close

.. doxygenfunction:: kas_strerror

.. _sec_c_api_get:

*************
Get functions
*************

Get functions provide the interface for querying a store. The most general interface
is :c:func:`kastore_get`, but it is usually more convenient to use one of the
:ref:`typed get functions <sec_c_api_typed_get>`.

.. doxygenfunction:: kastore_get
.. doxygenfunction:: kastore_gets

.. _sec_c_api_typed_get:

----------
Typed gets
----------

The functions listed here provide a convenient short-cut for accessing arrays
where the key is a standard NULL terminated C string and the type of the
array is known in advance.

.. doxygengroup:: TYPED_GETS_GROUP
        :content-only:

.. _sec_c_api_put:

*************
Put functions
*************

Put functions provide the interface for inserting data into store. The most
general interface is :c:func:`kastore_put` which allows keys to be arbitrary
bytes, but it is usually more convenient to use one of the :ref:`typed put
functions <sec_c_api_typed_put>`.

.. doxygenfunction:: kastore_put
.. doxygenfunction:: kastore_puts

.. _sec_c_api_typed_put:

----------
Typed puts
----------

The functions listed here provide a convenient short-cut for inserting
key-array pairs where the key is a standard NULL terminated C string and the
type of the array is known in advance.

.. doxygengroup:: TYPED_PUTS_GROUP
        :content-only:


*********
Constants
*********

.. _sec_c_api_error_codes:

------
Errors
------

.. doxygengroup:: ERROR_GROUP
        :content-only:

-----
Types
-----

.. doxygengroup:: TYPE_GROUP
        :content-only:

-------------------
Version information
-------------------

.. doxygengroup:: API_VERSION_GROUP
        :content-only:

.. doxygengroup:: FILE_VERSION_GROUP
        :content-only:

***********************
Miscellaneous functions
***********************

.. doxygenstruct:: kas_version_t
    :members:

.. doxygenfunction:: kas_version

.. doxygenfunction:: kas_dynamic_api_init

