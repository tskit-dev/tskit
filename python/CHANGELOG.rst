--------------------
[0.2.4] - 20XX-XX-XX
--------------------

In development

**Breaking changes**

- The default display order for tree visualisations has been changed to ``minlex``
  (see below) to stabilise the node ordering and to make trees more readily
  comparable. The old behaviour is still available with ``order="tree"``.

- File system operations such as dump/load now raise an appropriate OSError
  instead of tskit.FileFormatError. Loading from an empty file now raises
  and EOFError.

**New features**

- Mutations now have an optional double-precision floating-point ``time`` column.
  If not specified, this defaults to a particular NaN value (``tskit.UNKNOWN_TIME``)
  indicating that the time is unknown. For a tree sequence to be considered valid
  it must meet new criteria for mutation times, see :ref:`sec_mutation_requirements`.
  Also added function ``TableCollection.compute_mutation_times``.
  (:user:`benjeffery`, :pr:`672`)

- Add support for trees with internal samples for the Kendall-Colijn tree distance
  metric. (:user:`daniel-goldstein`, :pr:`610`)

- Add background shading to SVG tree sequences to reflect tree position along the
  sequence (:user:`hyanwong`, :pr:`563`)

- Tables with a metadata column now have a ``metadata_schema`` that is used to
  validate and encode metadata that is passed to ``add_row`` and decode metadata
  on calls to ``table[j]`` and e.g. ``tree_sequence.node(j)`` See :ref:`sec_metadata`.
  (:user:`benjeffery`, :pr:`491`, :pr:`542`, :pr:`543`, :pr:`601`)

- The tree-sequence now has top-level metadata with a schema.
  (:user:`benjeffery`, :pr:`666`, :pr:`644`, :pr:`642`)

- Add classes to SVG drawings to allow easy adjustment and styling, and document the new
  ``tskit.Tree.draw_svg()`` and ``tskit.TreeSequence.draw_svg()`` methods. This also fixes
  :issue:`467` for duplicate SVG entity ``id`` s in Jupyter notebooks.
  (:user:`hyanwong`, :pr:`555`)

- Add a ``nexus`` function that outputs a tree sequence in Nexus format
  (:user:`saunack`, :pr:`550`).

- Add extension of Kendall-Colijn tree distance metric for tree sequences
  computed by ``TreeSequence.kc_distance``
  (:user:`daniel-goldstein`, :pr:`548`)

- Add an optional node traversal order in ``tskit.Tree`` that uses the minimum
  lexicographic order of leaf nodes visited. This ordering (``"minlex_postorder"``)
  adds more determinism because it constraints the order in which children of
  a node are visited (:user:`brianzhang01`, :pr:`411`).

- Add an ``order`` argument to the tree visualisation functions which supports
  two node orderings: ``"tree"`` (the previous default) and ``"minlex"``
  which stabilises the node ordering (making it easier to compare trees).
  The default node ordering is changed to ``"minlex"``.
  (:user:`brianzhang01`, :user:`jeromekelleher`, :issue:`389`, :pr:`566`).

- Add ``_repr_html_`` to tables, so that jupyter notebooks render them as
  html tables (:user:`benjeffery`, :pr:`514`)

- Remove support for ``kc_distance`` on trees with unary nodes
  (:user:`daniel-goldstein`, :pr:`508`)

- Improve Kendall-Colijn tree distance algorithm to operate in O(n^2) time
  instead of O(n^2 * log(n)) where n is the number of samples
  (:user:`daniel-goldstein`, :pr:`490`)

- Add a metadata column to the migrations table. Works similarly to existing
  metadata columns on other tables(:user:`benjeffery`, :pr:`505`).

- Add a metadata column to the edges table. Works similarly to existing
  metadata columns on other tables(:user:`benjeffery`, :pr:`496`).

- Allow sites with missing data to be output by the ``haplotypes`` method, by
  default replacing with ``-``. Errors are no longer raised for missing data
  with ``impute_missing_data=False``; the error types returned for bad alleles
  (e.g. multiletter or non-ascii) have also changed from ``_tskit.LibraryError``
  to TypeError, or ValueError if the missing data character clashes
  (:user:`hyanwong`, :pr:`426`).

- Access the number of children of a node in a tree directly using
  ``tree.num_children(u)`` (:user:`hyanwong`, :pr:`436`).

- User specified allele mapping for genotypes in ``variants`` and
  ``genotype_matrix`` (:user:`jeromekelleher`, :pr:`430`).

- New ``root_threshold`` option for the Tree class, which allows
  us to efficiently iterate over 'real' roots when we have
  missing data (:user:`jeromekelleher`, :pr:`462`).

- Add pickle support for ``TreeSequence`` (:user:`terhorst`, :pr:`473`).

**Bugfixes**

- :issue:`453` - Fix LibraryError when ``tree.newick()`` is called with large node time
  values (:user:`jeromekelleher`, :pr:`637`).

**Deprecated**

- The ``sample_counts`` feature has been deprecated and is now
  ignored. Sample counts are now always computed.


--------------------
[0.2.3] - 2019-11-22
--------------------

Minor feature release, providing a tree distance metric and various
method to manipulate tree sequence data.

**New features**

- Kendall-Colijn tree distance metric computed by ``Tree.kc_distance``
  (:user:`awohns`, :pr:`172`).
- New "timeasc" and "timedesc" orders for tree traversals
  (:user:`benjeffery`, :issue:`246`, :pr:`399`).
- Up to 2X performance improvements to tree traversals (:user:`benjeffery`,
  :pr:`400`).
- Add ``trim``, ``delete_sites``, ``keep_intervals`` and ``delete_intervals``
  methods to edit tree sequence data. (:user:`hyanwong`, :pr:`364`,
  :pr:`372`, :pr:`377`, :pr:`390`).
- Initial online documentation for CLI (:user:`hyanwong`, :pr:`414`).
- Various documentation improvements (:user:`hyanwong`, :user:`jeromekelleher`,
  :user:`petrelharp`).
- Rename the ``map_ancestors`` function to ``link_ancestors``
  (:user:`hyanwong`, :user:`gtsambos`; :pr:`406`,
  :issue:`262`). The original function is retained as an deprecated alias.

**Bugfixes**

- Fix height scaling issues with SVG tree drawing (:user:`jeromekelleher`,
  :pr:`407`, :issue:`383`, :pr:`378`).
- Do not reuse buffers in ``LdCalculator`` (:user:`jeromekelleher`). See :pr:`397` and
  :issue:`396`.

--------------------
[0.2.2] - 2019-09-01
--------------------

Minor bugfix release.

Relaxes overly-strict input requirements on individual location data that
caused some SLiM tree sequences to fail loading in version 0.2.1
(see :issue:`351`).

**New features**

- Add log_time height scaling option for drawing SVG trees
  (:user:`marianne-aspbury`). See :pr:`324` and :issue:`303`.

**Bugfixes**

- Allow 4G metadata columns (:user:`jeromekelleher`). See :pr:`342` and
  :issue:`341`.


--------------------
[0.2.1] - 2019-08-23
--------------------

Major feature release, adding support for population genetic statistics,
improved VCF output and many other features.

**Note:** Version 0.2.0 was skipped because of an error uploading to PyPI
which could not be undone.

**Breaking changes**

- Genotype arrays returned by ``TreeSequence.variants`` and
  ``TreeSequence.genotype_matrix`` have changed from unsigned 8 bit values
  to signed 8 bit values to accomodate missing data (see :issue:`144` for
  discussion). Specifically, the dtype of the genotypes arrays have changed
  from numpy "u8" to "i8". This should not affect client code in any way
  unless it specifically depends on the type of the returned numpy array.

- The VCF written by the ``write_vcf`` is no longer compatible with previous
  versions, which had significant shortcomings. Position values are now rounded
  to the nearest integer by default, REF and ALT values are derived from the
  actual allelic states (rather than always being A and T). Sample names
  are now of the form ``tsk_j`` for sample ID j. Most of the legacy behaviour
  can be recovered with new options, however.

- The positional parameter ``reference_sets`` in ``genealogical_nearest_neighbours``
  and ``mean_descendants`` TreeSequence methods has been renamed to
  ``sample_sets``.

**New features**

- Support for general windowed statistics. Implementations of diversity,
  divergence, segregating sites, Tajima's D, Fst, Patterson's F statistics,
  Y statistics, trait correlations and covariance, and k-dimensional allele
  frequency specra (:user:`petrelharp`, :user:`jeromekelleher`, :user:`molpopgen`).

- Add the ``keep_unary`` option to simplify (:user:`gtsambos`). See :issue:`1`
  and :pr:`143`.

- Add the ``map_ancestors`` method to TableCollection (user:`gtsambos`). See :pr:`175`.

- Add the ``squash`` method to EdgeTable (:user:`gtsambos`). See :issue:`59` and
  :pr:`285`.

- Add support for individuals to VCF output, and fix major issues with output
  format (:user:`jeromekelleher`). Position values are transformed in a much
  more straightforward manner and output has been generalised substantially.
  Adds ``individual_names`` and ``position_transform`` arguments.
  See :pr:`286`, and issues :issue:`2`, :issue:`30` and :issue:`73`.

- Control height scale in SVG trees using 'tree_height_scale' and 'max_tree_height'
  (:user:`hyanwong`, :user:`jeromekelleher`). See :issue:`167`, :pr:`168`.
  Various other improvements to tree drawing (:pr:`235`, :pr:`241`, :pr:`242`,
  :pr:`252`, :pr:`259`).

- Add ``Tree.max_root_time`` property (:user:`hyanwong`, :user:`jeromekelleher`).
  See :pr:`170`.

- Improved input checking on various methods taking numpy arrays as parameters
  (:user:`hyanwong`). See :issue:`8` and :pr:`185`.

- Define the branch length over roots in trees to be zero (previously raise
  an error; :user:`jeromekelleher`). See :issue:`188` and :pr:`191`.

- Implementation of the genealogical nearest neighbours statistic
  (:user:`hyanwong`, :user:`jeromekelleher`).

- New ``delete_intervals`` and ``keep_intervals`` method for the TableCollection
  to allow slicing out of topology from specific intervals (:user:`hyanwong`,
  :user:`andrewkern`, :user:`petrelharp`, :user:`jeromekelleher`). See
  :pr:`225` and :pr:`261`.

- Support for missing data via a topological definition (:user:`jeromekelleher`).
  See :issue:`270` and :pr:`272`.

- Add ability to set columns directly in the Tables API (:user:`jeromekelleher`).
  See :issue:`12` and :pr:`307`.

- Various documentation improvements from :user:`brianzhang01`, :user:`hyanwong`,
  :user:`petrelharp` and :user:`jeromekelleher`.

**Deprecated**

- Deprecate ``Tree.length`` in favour of ``Tree.span`` (:user:`hyanwong`).
  See :pr:`169`.

- Deprecate ``TreeSequence.pairwise_diversity`` in favour of the new
  ``diversity`` method. See :issue:`215`, :pr:`312`.

**Bugfixes**

- Catch NaN and infinity values within tables (:user:`hyanwong`).
  See :issue:`293` and :pr:`294`.

--------------------
[0.1.5] - 2019-03-27
--------------------

This release removes support for Python 2, adds more flexible tree access and a
new ``tskit`` command line interface.

**New features**

- Remove support for Python 2 (:user:`hugovk`). See :issue:`137` and :pr:`140`.
- More flexible tree API (:pr:`121`). Adds ``TreeSequence.at`` and
  ``TreeSequence.at_index`` methods to find specific trees, and efficient support
  for backwards traversal using ``reversed(ts.trees())``.
- Add initial ``tskit`` CLI (:issue:`80`)
- Add ``tskit info`` CLI command (:issue:`66`)
- Enable drawing SVG trees with coloured edges (:user:`hyanwong`; :issue:`149`).
- Add ``Tree.is_descendant`` method (:issue:`120`)
- Add ``Tree.copy`` method (:issue:`122`)

**Bugfixes**

- Fixes to the low-level C API (:issue:`132` and :issue:`157`)


--------------------
[0.1.4] - 2019-02-01
--------------------


Minor feature update. Using the C API 0.99.1.

**New features**

- Add interface for setting TableCollection.sequence_length:
  https://github.com/tskit-dev/tskit/issues/107
- Add support for building and dropping TableCollection indexes:
  https://github.com/tskit-dev/tskit/issues/108


--------------------
[0.1.3] - 2019-01-14
--------------------

Bugfix release.

**Bugfixes**

- Fix missing provenance schema: https://github.com/tskit-dev/tskit/issues/81

--------------------
[0.1.2] - 2019-01-14
--------------------

Bugfix release.

**Bugfixes**

- Fix memory leak in table collection. https://github.com/tskit-dev/tskit/issues/76

--------------------
[0.1.1] - 2019-01-11
--------------------

Fixes broken distribution tarball for 0.1.0.

--------------------
[0.1.0] - 2019-01-11
--------------------

Initial release after separation from msprime 0.6.2. Code that reads tree sequence
files and processes them should be able to work without changes.

**Breaking changes**

- Removal of the previously deprecated ``sort_tables``, ``simplify_tables``
  and ``load_tables`` functions. All code should change to using corresponding
  TableCollection methods.

- Rename ``SparseTree`` class to ``Tree``.

----------------------
[1.1.0a1] - 2019-01-10
----------------------

Initial alpha version posted to PyPI for bootstrapping.

--------------------
[0.0.0] - 2019-01-10
--------------------

Initial extraction of tskit code from msprime. Relicense to MIT.

Code copied at hash 29921408661d5fe0b1a82b1ca302a8b87510fd23
