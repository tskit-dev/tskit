---------------------
[0.99.3] - 2020-XX-XX
---------------------

In development.

**Breaking changes**

- ``tsk_mutation_table_add_row`` has an extra ``time`` argument. If the time
  is unknown ``TSK_UNKNOWN_TIME`` should be passed.
  (:user:`benjeffery`, :pr:`672`)

- Change genotypes from unsigned to signed to accommodate missing data
  (see :issue:`144` for discussion). This only affects users of the
  ``tsk_vargen_t`` class. Genotypes are now stored as int8_t and int16_t
  types rather than the former unsigned types. The field names in the
  genotypes union of the ``tsk_variant_t`` struct returned by ``tsk_vargen_next``
  have been renamed to ``i8`` and ``i16`` accordingly; care should be
  taken when updating client code to ensure that types are correct. The number
  of distinct alleles supported by 8 bit genotypes has therefore dropped
  from 255 to 127, with a similar reduction for 16 bit genotypes.

- Change the ``tsk_vargen_init`` method to take an extra parameter ``alleles``.
  To keep the current behaviour, set this parameter to NULL.

- Edges can now have metadata. Hence edge methods now take two extra arguments:
  metadata and metadata length. The file format has also changed to accommodate this,
  but is backwards compatible.
  (:user:`benjeffery`, :pr:`496`)

- Migrations can now have metadata. Hence migration methods now take two extra arguments:
  metadata and metadata length. The file format has also changed to accommodate this,
  but is backwards compatible.
  (:user:`benjeffery`, :pr:`505`)

- The text dump of tables with metadata now includes the metadata schema as a header.
  (:user:`benjeffery`, :pr:`493`)

**New features**

- Mutations now have an optional double-precision floating-point ``time`` column.
  If not specified, this defaults to a particular NaN value (``TSK_UNKNOWN_TIME``)
  indicating that the time is unknown. For a tree sequence to be considered valid
  it must meet new criteria for mutation times, see :ref:`sec_mutation_requirements`.
  Add ``tsk_table_collection_compute_mutation_times`` and new flag to
  ``tsk_table_collection_check_integrity``:``TSK_CHECK_MUTATION_TIME``.
  (:user:`benjeffery`, :pr:`672`)

- Add ``metadata`` and ``metadata_schema`` fields to table collection, with accessors on
  tree sequence. These store arbitrary bytes and are optional in the file format.
  (:user: `benjeffery`, :pr:`641`)

- Add the ``TSK_KEEP_UNARY`` option to simplify (:user:`gtsambos`). See :issue:`1`
  and :pr:`143`.

- Add a ``set_root_threshold`` option to tsk_tree_t which allows us to set the
  number of samples a node must be an ancestor of to be considered a root
  (:pr:`462`).

- Change the semantics of tsk_tree_t so that sample counts are always
  computed, and add a new ``TSK_NO_SAMPLE_COUNTS`` option to turn this
  off (:pr:`462`).

- Tables with metadata now have an optional `metadata_schema` field that can contain
  arbitrary bytes. (:user:`benjeffery`, :pr:`493`)

- Tables loaded from a file can now be edited in the same way as any other
  table collection (:user:`jeromekelleher`, :issue:`536`, :pr:`530`.

- Support for reading/writing to arbitrary file streams with the loadf/dumpf
  variants for tree sequence and table collection load/dump
  (:user:`jeromekelleher`, :user:`grahamgower`, :issue:`565`, :pr:`599`).

**Deprecated**

- The ``TSK_SAMPLE_COUNTS`` options is now ignored and  will print out a warning
  if used (:pr:`462`).

---------------------
[0.99.2] - 2019-03-27
---------------------

Bugfix release. Changes:

- Fix incorrect errors on tbl_collection_dump (#132)
- Catch table overflows (#157)

---------------------
[0.99.1] - 2019-01-24
---------------------

Refinements to the C API as we move towards 1.0.0. Changes:

- Change the ``_tbl_`` abbreviation to ``_table_`` to improve readability.
  Hence, we now have, e.g., ``tsk_node_table_t`` etc.
- Change ``tsk_tbl_size_t`` to ``tsk_size_t``.
- Standardise public API to use ``tsk_size_t`` and ``tsk_id_t`` as appropriate.
- Add ``tsk_flags_t`` typedef and consistently use this as the type used to
  encode bitwise flags. To avoid confusion, functions now have an ``options``
  parameter.
- Rename ``tsk_table_collection_position_t`` to ``tsk_bookmark_t``.
- Rename ``tsk_table_collection_reset_position`` to ``tsk_table_collection_truncate``
  and ``tsk_table_collection_record_position`` to ``tsk_table_collection_record_num_rows``.
- Generalise ``tsk_table_collection_sort`` to take a bookmark as start argument.
- Relax restriction that nodes in the ``samples`` argument to simplify must
  currently be marked as samples. (https://github.com/tskit-dev/tskit/issues/72)
- Allow ``tsk_table_collection_simplify`` to take a NULL samples argument to
  specify "all samples in the current tables".
- Add support for building as a meson subproject.

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
