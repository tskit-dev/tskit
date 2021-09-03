--------------------
[0.1.4] - 2021-09-02
--------------------

- Offset columns are now 64 bit in tskit. For compatibility, offset arrays that fit into
  32bits will be a 32bit array in the dict encoding. Large arrays that require 64 bit
  will fail to ``fromdict`` in previous versions with the error:
  ``TypeError: Cannot cast array data from dtype('uint64') to dtype('uint32') according
  to the rule 'safe'``
  A ``force_offset_64`` option on ``asdict`` allows the easy creation of 64bit arrays for
  testing.

--------------------
[0.1.3] - 2021-02-01
--------------------

- Added optional ``parents`` to individual table.

--------------------
[0.1.2] - 2020-10-22
--------------------

 - Added optional top-level key ``indexes`` which has contains ``edge_insertion_order`` and
   ``edge_removal_order``