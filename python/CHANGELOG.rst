----------------------
[0.4.0] - 2021-12-10
----------------------

**Breaking changes**

- The ``Tree.num_nodes`` method is now deprecated with a warning, because it confusingly
  returns the number of nodes in the entire tree sequence, rather than in the tree. Text
  summaries of trees (e.g. ``str(tree)``) now return the number of nodes in the tree,
  not in the entire tree sequence (:user:`hyanwong`, :issue:`1966` :pr:`1968`)

- The CLI ``info`` command now gives more detailed information on the tree sequence
  (:user:`benjeffery`, :pr:`1611`)

- 64 bits are now used to store the sizes of ragged table columns such as metadata,
  allowing them to hold more data. This change is fully backwards and forwards compatible
  for all tree-sequences whose ragged column sizes fit into 32 bits. New tree-sequences with
  large offset arrays that require 64 bits will fail to load in previous versions with
  error ``_tskit.FileFormatError: An incompatible type for a column was found in the
  file``.
  (:user:`jeromekelleher`, :issue:`343`, :issue:`1527`, :issue:`1528`, :issue:`1530`,
  :issue:`1554`, :issue:`1573`, :issue:`1589`,:issue:`1598`,:issue:`1628`, :pr:`1571`,
  :pr:`1579`, :pr:`1585`, :pr:`1590`, :pr:`1602`, :pr:`1618`, :pr:`1620`, :pr:`1652`).

- The Tree class now conceptually has an extra node, the "virtual root" whose
  children are the roots of the tree. The quintuply linked tree arrays
  (parent_array, left_child_array, right_child_array, left_sib_array and right_sib_array)
  all have one extra element.
  (:user:`jeromekelleher`, :issue:`1691`, :pr:`1704`).

- Tree traversal orders returned by the ``nodes`` method have changed when there
  are multiple roots. Previously orders were defined locally for each root, but
  are now globally across all roots. (:user:`jeromekelleher`, :pr:`1704`).

- Individuals are no longer guaranteed or required to be topologically sorted in a tree sequence.
  ``TableCollection.sort`` no longer sorts individuals.
  (:user:`benjeffery`, :issue:`1774`, :pr:`1789`)

- Metadata encoding errors now raise ``MetadataEncodingError``
  (:user:`benjeffery`, :issue:`1505`, :pr:`1827`).

- For ``TreeSequence.samples`` all arguments after ``population`` are now keyword only
  (:user:`benjeffery`, :issue:`1715`, :pr:`1831`).

- Remove the method ``TreeSequence.to_nexus`` and replace with ``TreeSequence.as_nexus``.
  As the old method was not generating standards-compliant output, it seems unlikely
  that it was used by anyone. Calls to ``to_nexus`` will result in a
  NotImplementedError, informing users of the change. See below for details on
  ``as_nexus``.

- Change default value for ``missing_data_char`` in the ``TreeSequence.haplotypes``
  method from "-" to "N". This is a more idiomatic usage to indicate
  missing data rather than a gap in an alignment. (:user:`jeromekelleher`,
  :issue:`1893`, :pr:`1894`)

**Features**

- Add the ``ibd_segments`` method and associated classes to compute, summarise
  and store segments of identity by descent from a tree sequence
  (:user:`gtsambos`, :user:`jeromekelleher`).

- Allow skipping of site and mutation tables in ``TableCollection.sort``
  (:user:`benjeffery`, :issue:`1475`, :pr:`1826`).

- Add ``TableCollection.sort_individuals`` to sort the individuals as this is no longer done by the
  default sort (:user:`benjeffery`, :issue:`1774`, :pr:`1789`).

- Add ``__setitem__`` to all tables allowing single rows to be updated. For example
  ``tables.nodes[0] = tables.nodes[0].replace(flags=tskit.NODE_IS_SAMPLE)``
  (:user:`jeromekelleher`, :user:`benjeffery`, :issue:`1545`, :pr:`1600`).

- Added a new parameter ``time`` to ``TreeSequence.samples()`` allowing to select
  samples at a specific time point or time interval.
  (:user:`mufernando`, :user:`petrelharp`, :issue:`1692`, :pr:`1700`)

- Add ``table.metadata_vector`` to all table classes to allow easy extraction of a single
  metadata key into an array
  (:user:`petrelharp`, :issue:`1676`, :pr:`1690`).

- Add ``time_units`` to ``TreeSequence`` to describe the units of the time dimension of the
  tree sequence. This is then used to generate an error if ``time_units`` is ``uncalibrated`` when
  using the branch lengths in statistics. (:user:`benjeffery`, :issue:`1644`, :pr:`1760`, :pr:`1832`)

- Add the ``virtual_root`` property to the Tree class (:user:`jeromekelleher`, :pr:`1704`).

- Add the ``num_edges`` property to the Tree class (:user:`jeromekelleher`, :pr:`1704`).

- Improved performance for tree traversal methods in the ``nodes`` iterator.
  Roughly a 10X performance increase for "preorder", "postorder", "timeasc"
  and "timedesc" (:user:`jeromekelleher`, :pr:`1704`).

- Substantial performance improvement for ``Tree.total_branch_length``
  (:user:`jeromekelleher`, :issue:`1794` :pr:`1799`)

- Add the ``discrete_genome`` property to the TreeSequence class which is true if
  all coordinates are discrete (:user:`jeromekelleher`, :issue:`1144`, :pr:`1819`)

- Add a ``random_nucleotides`` function. (user:`jeromekelleher`, :pr:`1825`)

- Add the ``TreeSequence.alignments`` method. (user:`jeromekelleher`, :pr:`1825`)

- Add alignment export in the FASTA and nexus formats using the
  ``TreeSequence.write_nexus`` and ``TreeSequence.write_fasta`` methods.
  (:user:`jeromekelleher`, :user:`hyanwong`, :pr:`1894`)

- Add the ``discrete_time`` property to the TreeSequence class which is true if
  all time coordinates are discrete or unknown (:user:`benjeffery`, :issue:`1839`, :pr:`1890`)

- Add the ``skip_tables`` option to ``load`` to support only loading
  top-level information from a file. Also add the ``ignore_tables`` option to
  ``TableCollection.equals`` and ``TableCollection.assert_equals`` to
  compare only top-level information. (:user:`clwgg`, :pr:`1882`, :issue:`1854`).

- Add the ``skip_reference_sequence`` option to ``load``. Also add the
  ``ignore_reference_sequence`` option ``equals`` to compare two table
  collections without comparing their reference sequence. (:user:`clwgg`,
  :pr:`2019`, :issue:`1971`).

- tskit now supports python 3.10 (:user:`benjeffery`, :issue:`1895`, :pr:`1949`)


**Fixes**

- `dump_tables` omitted individual parents. (:user:`benjeffery`, :issue:`1828`, :pr:`1884`)

- Add the ``Tree.as_newick`` method and deprecate ``Tree.newick``. The
  ``as_newick`` method by default labels samples with the pattern ``"n{node_id}"``
  which is much more useful that the behaviour of ``Tree.newick`` (which mimics
  ``ms`` output). (:user:`jeromekelleher`, :issue:`1671`, :pr:`1838`.)

- Add the ``as_nexus`` and ``write_nexus`` methods to the TreeSequence class,
  replacing the broken ``to_nexus`` method (see above). This uses the same
  sample labelling pattern as ``as_newick``.
  (:user:`jeetsukumaran`, :user:`jeromekelleher`, :issue:`1785`, :pr:`1835`,
  :pr:`1836`, :pr:`1838`)

- `load_text` created additional populations even if the population table was specified,
  and didn't strip newlines from input text (:user:`hyanwong`, :issue:`1909`, :pr:`1910`)


--------------------
[0.3.7] - 2021-07-08
--------------------

**Features**

- ``map_mutations`` now allows the ancestral state to be specified
  (:user:`hyanwong`, :user:`jeromekelleher`, :issue:`1542`, :pr:`1550`)

--------------------
[0.3.6] - 2021-05-14
--------------------

**Breaking changes**

- ``Mutation.position`` and ``Mutation.index`` which were deprecated in 0.2.2 (Sep '19) have
  been removed.

**Features**

- Add direct, copy-free access to the arrays representing the quintuply-linked structure
  of ``Tree`` (e.g. ``left_child_array``). Allows performant algorithms over the tree
  structure using, for example, numba
  (:user:`jeromekelleher`, :issue:`1299`, :pr:`1320`).

- Add fancy indexing to tables. E.g. ``table[6:86]`` returns a new table with the
  specified rows. Supports slices, index arrays and boolean masks
  (:user:`benjeffery`, :issue:`1221`, :pr:`1348`, :pr:`1342`).

- Add ``Table.append`` method for adding rows from classes such as ``SiteTableRow`` and
  ``Site`` (:user:`benjeffery`, :issue:`1111`, :pr:`1254`).

- SVG visualization of a tree sequence can be restricted to displaying between left
  and right genomic coordinates using the ``x_lim`` parameter. The default settings
  now mean that if the left or right flanks of a tree sequence are entirely empty,
  these regions will not be plotted in the SVG (:user:`hyanwong`, :pr:`1288`).

- SVG visualization of a single tree allows all mutations on an edge to be plotted
  via the ``all_edge_mutations`` param (:user:`hyanwong`,:issue:`1253`, :pr:`1258`).

- Entity classes such as ``Mutation``, ``Node`` are now python dataclasses
  (:user:`benjeffery`, :pr:`1261`).

- Metadata decoding for table row access is now lazy (:user:`benjeffery`, :pr:`1261`).

- Add html notebook representation for ``Tree`` and change ``Tree.__str__`` from dict
  representation to info table. (:user:`benjeffery`, :issue:`1269`, :pr:`1304`).

- Improve display of tables when ``print``ed, limiting lines set via
  ``tskit.set_print_options`` (:user:`benjeffery`,:issue:`1270`, :pr:`1300`).

- Add ``Table.assert_equals`` and ``TableCollection.assert_equals`` which give an exact
  report of any differences. (:user:`benjeffery`,:issue:`1076`, :pr:`1328`)

**Changes**

- In drawing methods ``max_tree_height`` and ``tree_height_scale`` have been deprecated
  in favour of ``max_time`` and ``time_scale``
  (:user:`benjeffery`,:issue:`1262`, :pr:`1331`).

**Fixes**

- Tree sequences were not properly init'd after unpickling
  (:user:`benjeffery`, :issue:`1297`, :pr:`1298`)

--------------------
[0.3.5] - 2021-03-16
--------------------

**Features**

- SVG visualization plots mutations at the correct time, if it exists, and a y-axis,
  with label can be drawn. Both x- and y-axes can be plotted on trees as well as
  tree sequences (:user:`hyanwong`,:issue:`840`, :issue:`580`, :pr:`1236`)

- SVG visualization now uses squares for sample nodes and red crosses for mutations,
  with the site/mutation positions marked on the x-axis. Additionally, an x-axis
  label can be set (:user:`hyanwong`,:issue:`1155`, :issue:`1194`, :pr:`1182`, :pr:`1213`)

- Add ``parents`` column to the individual table to allow recording of pedigrees
  (:user:`ivan-krukov`, :user:`benjeffery`, :issue:`852`, :pr:`1125`, :pr:`866`, :pr:`1153`, :pr:`1177`, :pr:`1192` :pr:`1199`).

- Added ``Tree.generate_random_binary`` static method to create random
  binary trees (:user:`hyanwong`, :user:`jeromekelleher`, :pr:`1037`).

- Change the default behaviour of Tree.split_polytomies to generate
  the shortest possible branch lengths instead of a fixed epsilon of
  1e-10. (:user:`jeromekelleher`, :issue:`1089`, :pr:`1090`)

- Default value metadata in ``add_row`` functions is now schema-dependant, so that
  ``metadata={}`` is no longer needed as an argument when a schema is present
  (:user:`benjeffery`, :issue:`1084`).

- ``default`` in metadata schemas is used to fill in missing values when encoding for
  the struct codec. (:user:`benjeffery`, :issue:`1073`, :pr:`1116`).

- Added ``canonical`` option to table collection sorting (:user:`mufernando`,
  :user:`petrelharp`, :issue:`705`)

- Added various arguments to ``TreeSequence.subset``, to allow for stable
  population indexing and lossless node reordering with subset.
  (:user:`petrelharp`, :pr:`1097`)

**Changes**

- Allow mutations that have the same derived state as their parent mutation.
  (:user:`benjeffery`, :issue:`1180`, :pr:`1233`)

- File minor version change to support individual parents

**Breaking changes**

- tskit now requires Python 3.7 (:user:`benjeffery`, :pr:`1235`)

--------------------
[0.3.4] - 2020-12-02
--------------------

Minor bugfix release.


**Bugfixes**

- Reinstate the unused zlib_compression option to tskit.dump, as msprime < 1.0
  still uses it (:user:`jeromekelleher`, :issue:`1067`).

--------------------
[0.3.3] - 2020-11-27
--------------------

**Features**

- Add ``TreeSequence.genetic_relatedness`` for calculating genetic relatedness between
  pairs of sets of nodes (:user:`brieuclehmann`, :issue:`1021`, :pr:`1023`, :issue:`974`,
  :issue:`973`, :pr:`898`).

- Expose ``TreeSequence.coiterate()`` method to allow iteration over 2 sequences
  simultaneously, aiding comparison of trees from two sequences
  (:user:`jeromekelleher`, :user:`hyanwong`, :issue:`1021`, :pr:`1022`).

- tskit is now supported on, and has wheels for, python3.9
  (:user:`benjeffery`, :issue:`982`, :pr:`907`).

- ``Tree.newick()`` now has extra option ``include_branch_lengths`` to allow branch
  lengths to be omitted (:user:`hyanwong`, :pr:`931`).

- Added ``Tree.generate_star`` static method to create star-topologies (:user:`hyanwong`,
  :pr:`934`).

- Added ``Tree.generate_comb`` and ``Tree.generate_balanced`` methods to create
  example trees. (:user:`jeromekelleher`, :pr:`1026`).

- Added ``equals`` method to TreeSequence, TableCollection and each of the tables which
  provides more flexible equality comparisons, for example, allowing
  users to ignore metadata or provenance in the comparison
  (:user:`mufernando`, :user:`jeromekelleher`, :issue:`896`, :pr:`897`,
  :issue:`913`, :pr:`917`).

- Added ``__eq__`` to TreeSequence
  (:user:`benjeffery`, :issue:`1011`, :pr:`1020`).

- ``ts.dump`` and ``tskit.load`` now support reading and writing file objects such as
  FIFOs and sockets (:user:`benjeffery`, :issue:`657`, :pr:`909`).

- Added ``tskit.write_ms`` for writing to MS format
  (:user:`saurabhbelsare`, :issue:`727`, :pr:`854`).

- Added ``TableCollection.indexes`` for access to the edge insertion/removal order indexes
  (:user:`benjeffery`, :issue:`4`, :pr:`916`).

- The dictionary representation of a TableCollection now contains its index
  (:user:`benjeffery`, :issue:`870`, :pr:`921`).

- Added ``TreeSequence._repr_html_`` for use in jupyter notebooks
  (:user:`benjeffery`, :issue:`872`, :pr:`923`).

- Added ``TreeSequence.__str__`` to display a summary for terminal usage
  (:user:`benjeffery`, :issue:`938`, :pr:`985`).

- Added ``TableCollection.dump`` and ``TableCollection.load``. This allows table
  collections that are not valid tree sequences to be manipulated
  (:user:`benjeffery`, :issue:`14`, :pr:`986`).

- Added ``nbytes`` method to tables, ``TableCollection`` and ``TreeSequence`` which
  reports the size in bytes of those objects
  (:user:`jeromekelleher`, :user:`benjeffery`, :issue:`54`, :pr:`871`).

- Added ``TableCollection.clear`` to clear data table rows and optionally
  provenances, table schemas and tree-sequence level metadata and schema
  (:user:`benjeffery`, :issue:`929`, :pr:`1001`).

**Bugfixes**

- ``LightWeightTableCollection.asdict`` and ``TableCollection.asdict`` now return copies
  of arrays (:user:`benjeffery`, :issue:`1025`, :pr:`1029`).

- The ``map_mutations`` method previously used the Fitch parsimony method, but this
  does not produce parsimonious results on non-binary trees. We now now use the
  Hartigan parsimony algorithm, which does (:user:`jeromekelleher`,
  :issue:`987`, :pr:`1030`).

- The ``flag`` argument to tables' ``add_row`` was treating the value as signed
  (:user:`benjeffery`, :issue:`1027`, :pr:`1031`).

**Breaking changes**

- The argument to ``ts.dump`` and ``tskit.load`` has been renamed `file` from `path`.
- All arguments to ``Tree.newick()`` except precision are now keyword-only.
- Renamed ``ts.trait_regression`` to ``ts.trait_linear_model``.

--------------------
[0.3.2] - 2020-09-29
--------------------

**Breaking changes**

- The argument order of ``Tree.unrank`` and ``combinatorics.num_labellings`` now
  positions the number of leaves before the tree rank
  (:user:`daniel-goldstein`, :issue:`950`, :pr:`978`)

- Change several methods (``simplify()``, ``trees()``, ``Tree()``) so most parameters
  are keyword only, not positional. This allows reordering of parameters, so
  that deprecated parameters can be moved, and the parameter order in similar functions,
  e.g. ``TableCollection.simplify`` and ``TreeSequence.simplify()`` can be made
  consistent (:user:`hyanwong`, :issue:`374`, :issue:`846`, :pr:`851`)


**Features**

- Add ``split_polytomies`` method to the Tree class
  (:user:`hyanwong`, :user:`jeromekelleher`, :issue:`809`, :pr:`815`)

- Tree accessor functions (e.g. ``ts.first()``, ``ts.at()`` pass extra parameters such as
  ``sample_indexes`` to the underlying ``Tree`` constructor; also ``root_threshold`` can
  be specified when calling ``ts.trees()`` (:user:`hyanwong`, :issue:`847`, :pr:`848`)

- Genomic intervals returned by python functions are now namedtuples, allowing ``.left``
  ``.right`` and ``.span`` usage (:user:`hyanwong`, :issue:`784`, :pr:`786`, :pr:`811`)

- Added ``include_terminal`` parameter to edge diffs iterator, to output the last edges
  at the end of a tree sequence (:user:`hyanwong`, :issue:`783`, :pr:`787`)

- :issue:`832` - Add ``metadata_bytes`` method to allow access to raw
  TableCollection metadata (:user:`benjeffery`, :pr:`842`)

- New ``tree.is_isolated(u)`` method (:user:`hyanwong`, :pr:`443`).

- ``tskit.is_unknown_time`` can now check arrays. (:user:`benjeffery`, :pr:`857`).

--------------------
[0.3.1] - 2020-09-04
--------------------

**Bugfixes**

- :issue:`823` - Fix mutation time error when using
  ``simplify(keep_input_roots=True)`` (:user:`petrelharp`, :pr:`823`).

- :issue:`821` - Fix mutation rows with unknown time never being
  equal (:user:`petrelharp`, :pr:`822`).

--------------------
[0.3.0] - 2020-08-27
--------------------

Major feature release for metadata schemas, set-like operations, mutation times,
SVG drawing improvements and many others.

**Breaking changes**

- The default display order for tree visualisations has been changed to ``minlex``
  (see below) to stabilise the node ordering and to make trees more readily
  comparable. The old behaviour is still available with ``order="tree"``.

- File system operations such as dump/load now raise an appropriate OSError
  instead of ``tskit.FileFormatError``. Loading from an empty file now raises
  and ``EOFError``.

- Bad tree topologies are detected earlier, so that it is no longer possible
  to create a ``TreeSequence`` object which contains a parent with contradictory
  children on an interval. Previously an error was thrown when some operation
  building the trees was attempted (:user:`jeromekelleher`, :pr:`709`).

- The ``TableCollection object`` no longer implements the iterator protocol.
  Previously ``list(tables)`` returned a sequence of (table_name, table_instance)
  tuples. This has been replaced with the more intuitive and future-proof
  ``TableCollection.name_map`` and ``TreeSequence.tables_dict`` attributes, which
  perform the same function (:user:`jeromekelleher`, :issue:`500`,
  :pr:`694`).

- The arguments to ``TreeSequence.genotype_matrix``, ``TreeSequence.haplotypes``
  and ``TreeSequence.variants`` must now be keyword arguments, not positional. This
  is to support the change from ``impute_missing_data`` to ``isolated_as_missing``
  in the arguments to these methods. (:user:`benjeffery`, :issue:`716`, :pr:`794`)

**New features**

- New methods to perform set operations on TableCollections and TreeSequences.
  ``TableCollection.subset`` subsets and reorders table collections by nodes
  (:user:`mufernando`, :user:`petrelharp`, :pr:`663`, :pr:`690`).
  ``TableCollection.union`` forms the node-wise union of two table collections
  (:user:`mufernando`, :user:`petrelharp`, :issue:`381` :pr:`623`).

- Mutations now have an optional double-precision floating-point ``time`` column.
  If not specified, this defaults to a particular ``NaN`` value (``tskit.UNKNOWN_TIME``)
  indicating that the time is unknown. For a tree sequence to be considered valid
  it must meet new criteria for mutation times, see :ref:`sec_mutation_requirements`.
  Also added function ``TableCollection.compute_mutation_times``. Table sorting orders
  mutations by non-increasing time per-site, which is also a requirement for a valid tree
  sequence (:user:`benjeffery`, :pr:`672`).

- Add support for trees with internal samples for the Kendall-Colijn tree distance
  metric. (:user:`daniel-goldstein`, :pr:`610`)

- Add background shading to SVG tree sequences to reflect tree position along the
  sequence (:user:`hyanwong`, :pr:`563`).

- Tables with a metadata column now have a ``metadata_schema`` that is used to
  validate and encode metadata that is passed to ``add_row`` and decode metadata
  on calls to ``table[j]`` and e.g. ``tree_sequence.node(j)`` See :ref:`sec_metadata`
  (:user:`benjeffery`, :pr:`491`, :pr:`542`, :pr:`543`, :pr:`601`).

- The tree-sequence now has top-level metadata with a schema
  (:user:`benjeffery`, :pr:`666`, :pr:`644`, :pr:`642`).

- Add classes to SVG drawings to allow easy adjustment and styling, and document the new
  ``tskit.Tree.draw_svg()`` and ``tskit.TreeSequence.draw_svg()`` methods. This also fixes
  :issue:`467` for duplicate SVG entity ``id`` s in Jupyter notebooks
  (:user:`hyanwong`, :pr:`555`).

- Add a ``to_nexus`` function that outputs a tree sequence in Nexus format
  (:user:`saunack`, :pr:`550`).

- Add extension of Kendall-Colijn tree distance metric for tree sequences
  computed by ``TreeSequence.kc_distance``
  (:user:`daniel-goldstein`, :pr:`548`).

- Add an optional node traversal order in ``tskit.Tree`` that uses the minimum
  lexicographic order of leaf nodes visited. This ordering (``"minlex_postorder"``)
  adds more determinism because it constraints the order in which children of
  a node are visited (:user:`brianzhang01`, :pr:`411`).

- Add an ``order`` argument to the tree visualisation functions which supports
  two node orderings: ``"tree"`` (the previous default) and ``"minlex"``
  which stabilises the node ordering (making it easier to compare trees).
  The default node ordering is changed to ``"minlex"``
  (:user:`brianzhang01`, :user:`jeromekelleher`, :issue:`389`, :pr:`566`).

- Add ``_repr_html_`` to tables, so that jupyter notebooks render them as
  html tables (:user:`benjeffery`, :pr:`514`).

- Remove support for ``kc_distance`` on trees with unary nodes
  (:user:`daniel-goldstein`, :pr:`508`).

- Improve Kendall-Colijn tree distance algorithm to operate in O(n^2) time
  instead of O(n^2 * log(n)) where n is the number of samples
  (:user:`daniel-goldstein`, :pr:`490`).

- Add a metadata column to the migrations table. Works similarly to existing
  metadata columns on other tables (:user:`benjeffery`, :pr:`505`).

- Add a metadata column to the edges table. Works similarly to existing
  metadata columns on other tables (:user:`benjeffery`, :pr:`496`).

- Allow sites with missing data to be output by the ``haplotypes`` method, by
  default replacing with ``-``. Errors are no longer raised for missing data
  with ``isolated_as_missing=True``; the error types returned for bad alleles
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

- Add ``tree.as_dict_of_dicts()`` function to enable use with networkx. See
  :ref:`sec_tutorial_networkx` (:user:`winni2k`, :pr:`457`).

- Add ``tree_sequence.to_macs()`` function to convert tree sequence to MACS
  format (:user:`winni2k`, :pr:`727`)

- Add a ``keep_input_roots`` option to simplify which, if enabled, adds edges
  from the MRCAs of samples in the simplified tree sequence back to the roots
  in the input tree sequence (:user:`jeromekelleher`, :issue:`775`, :pr:`782`).

**Bugfixes**

- :issue:`453` - Fix LibraryError when ``tree.newick()`` is called with large node time
  values (:user:`jeromekelleher`, :pr:`637`).

- :issue:`777` - Mutations over isolated samples were incorrectly decoded as
  missing data. (:user:`jeromekelleher`, :pr:`778`)

- :issue:`776` - Fix a segfault when a partial list of samples
  was provided to the ``variants`` iterator. (:user:`jeromekelleher`, :pr:`778`)

**Deprecated**

- The ``sample_counts`` feature has been deprecated and is now
  ignored. Sample counts are now always computed.

- For ``TreeSequence.genotype_matrix``, ``TreeSequence.haplotypes``
  and ``TreeSequence.variants`` the ``impute_missing_data`` argument is deprecated
  and replaced with ``isolated_as_missing``. Note that to get the same behaviour
  ``impute_missing_data=True`` should be replaced with ``isolated_as_missing=False``.
  (:user:`benjeffery`, :issue:`716`, :pr:`794`)

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
