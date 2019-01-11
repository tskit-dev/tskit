--------------------
[0.1.0] - 2019-01-11
--------------------

Initial release after separation from msprime 0.6.2. Code that reads tree sequence
files and processes them should be able to work without changes.

**Breaking changes**

- Removal of the previously deprecated ``sort_tables``, ``simplify_tables``
  and ``load_tables`` functions. All code should change to using corresponding
  TableCollection methods.

----------------------
[0.1.0a1] - 2019-01-19
----------------------

Initial alpha version posted to PyPI for bootstrapping.

--------------------
[0.0.0] - 2019-01-19
--------------------

Initial extraction of tskit code from msprime. Relicense to MIT.

Code copied at hash 29921408661d5fe0b1a82b1ca302a8b87510fd23
