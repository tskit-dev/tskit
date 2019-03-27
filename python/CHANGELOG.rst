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
