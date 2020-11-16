---------------------
[0.99.8] - 2020-XX-XX
---------------------

**Breaking changes**

- Added an ``options`` argument to ``tsk_table_collection_equals``
  and table equality methods to allow for more flexible equality criteria
  (e.g., ignore top-level metadata and schema or provenance tables).
  Existing code should add an extra final parameter ``0`` to retain the
  current behaviour. (:user:`mufernando`, :user:`jeromekelleher`,
  :issue:`896`, :pr:`897`, :issue:`913`, :pr:`917`).

- Changed default behaviour of ``tsk_table_collection_clear`` to not clear
  provenances and added ``options`` argument to optionally clear provenances
  and schemas.
  (:user:`benjeffery`, :issue:`929`, :pr:`1001`)

- Exposed ``tsk_table_collection_set_indexes`` to the API.
  (:user:`benjeffery`, :issue:`870`, :pr:`921`)

- Renamed ``ts.trait_regression`` to ``ts.trait_linear_model``.

---------------------
[0.99.7] - 2020-09-29
---------------------

- Added ``TSK_INCLUDE_TERMINAL`` option to ``tsk_diff_iter_init`` to output the last edges
  at the end of a tree sequence (:user:`hyanwong`, :issue:`783`, :pr:`787`)

- Added ``tsk_bug_assert`` for assertions that should be compiled into release binaries
  (:user:`benjeffery`, :pr:`860`)

---------------------
[0.99.6] - 2020-09-04
---------------------

**Bugfixes**

- :issue:`823` - Fix mutation time error when using
  ``tsk_table_collection_simplify`` with ``TSK_KEEP_INPUT_ROOTS``
  (:user:`petrelharp`, :pr:`823`).

---------------------
[0.99.5] - 2020-08-27
---------------------

**Breaking changes**

- The macro ``TSK_IMPUTE_MISSING_DATA`` is renamed to ``TSK_ISOLATED_NOT_MISSING``
  (:user:`benjeffery`, :issue:`716`, :pr:`794`)

**New features**

- Add a ``TSK_KEEP_INPUT_ROOTS`` option to simplify which, if enabled, adds edges
  from the MRCAs of samples in the simplified tree sequence back to the roots
  in the input tree sequence (:user:`jeromekelleher`, :issue:`775`, :pr:`782`).

**Bugfixes**

- :issue:`777` - Mutations over isolated samples were incorrectly decoded as
  missing data. (:user:`jeromekelleher`, :pr:`778`)

- :issue:`776` - Fix a segfault when a partial list of samples
  was provided to the ``variants`` iterator. (:user:`jeromekelleher`, :pr:`778`)

---------------------
[0.99.4] - 2020-08-12
---------------------

**Note**

- The ``TSK_VERSION_PATCH`` macro was incorrectly set to ``4`` for 0.99.3, so both
  0.99.4 and 0.99.3 have the same value.

**Changes**

- Mutation times can be a mixture of known and unknown as long as for each
  individual site  they are either all known or all unknown (:user:`benjeffery`, :pr:`761`).

**Bugfixes**

- Fix for including core.h under C++ (:user:`petrelharp`, :pr:`755`).

---------------------
[0.99.3] - 2020-07-27
---------------------

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
  but is backwards compatible. Edge metadata can be disabled for a table collection with
  the TSK_NO_EDGE_METADATA flag.
  (:user:`benjeffery`, :pr:`496`, :pr:`712`)

- Migrations can now have metadata. Hence migration methods now take two extra arguments:
  metadata and metadata length. The file format has also changed to accommodate this,
  but is backwards compatible.
  (:user:`benjeffery`, :pr:`505`)

- The text dump of tables with metadata now includes the metadata schema as a header.
  (:user:`benjeffery`, :pr:`493`)

- Bad tree topologies are detected earlier, so that it is no longer possible
  to create a tsk_treeseq_t object which contains a parent with contradictory
  children on an interval. Previously an error occured when some operation
  building the trees was attempted (:user:`jeromekelleher`, :pr:`709`).

**New features**

- New methods to perform set operations on table collections.
  ``tsk_table_collection_subset`` subsets and reorders table collections by nodes
  (:user:`mufernando`, :user:`petrelharp`, :pr:`663`, :pr:`690`).
  ``tsk_table_collection_union`` forms the node-wise union of two table collections
  (:user:`mufernando`, :user:`petrelharp`, :issue:`381`, :pr:`623`).

- Mutations now have an optional double-precision floating-point ``time`` column.
  If not specified, this defaults to a particular NaN value (``TSK_UNKNOWN_TIME``)
  indicating that the time is unknown. For a tree sequence to be considered valid
  it must meet new criteria for mutation times, see :ref:`sec_mutation_requirements`.
  Add ``tsk_table_collection_compute_mutation_times`` and new flag to
  ``tsk_table_collection_check_integrity``:``TSK_CHECK_MUTATION_TIME``. Table sorting
  orders mutations by non-increasing time per-site, which is also a requirement for a
  valid tree sequence.
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

- Add low-level sorting API and ``TSK_NO_CHECK_INTEGRITY`` flag
  (:user:`jeromekelleher`, :pr:`627`, :issue:`626`).

- Add extension of Kendall-Colijn tree distance metric for tree sequences
  computed by ``tsk_treeseq_kc_distance``
  (:user:`daniel-goldstein`, :pr:`548`)

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
