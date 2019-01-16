---------------------
[0.99.1] - 2019-XX-XX
---------------------

Refinements to the C API as we move towards 1.0.0. Changes:

- Change the ``_tbl_`` abbreviation to ``_table_`` to improve readability.
  Hence, we now have, e.g., ``tsk_node_table_t`` etc.
- Change ``tsk_tbl_size_t`` to ``tsk_size_t``.
- Standardise public API to use ``tsk_size_t`` and ``tsk_id_t`` as appropriate.
- Add ``tsk_flags_t`` typedef and consistently use this as the type used to
  encode bitwise flags. To avoid confusion, functions now have an ``options``
  parameter.

---------------------
[0.99.0] - 2019-01-14
---------------------

Initial alpha version of the tskit C API tagged. Version 0.99.x
represents the series of releases leading to version 1.0.0 which
will be the first stable release. After 1.0.0, semver rules
regarding API/ABI breakage will apply; however, in the 0.99.x
series arbitrary changes may happen.

--------------------
[0.0.0] - 2019-01-10
--------------------

Initial extraction of tskit code from msprime. Relicense to MIT.
Code copied at hash 29921408661d5fe0b1a82b1ca302a8b87510fd23
