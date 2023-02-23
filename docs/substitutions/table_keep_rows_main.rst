Updates this table in-place according to the specified boolean
array, and returns the resulting mapping from old to new row IDs.
For each row ``j``, if ``keep[j]`` is True, that row will be
retained in the output; otherwise, the row will be deleted.
Rows are retained in their original ordering.

The returned ``id_map`` is an array of the same length as
this table before the operation, such that ``id_map[j] = -1``
(:data:`tskit.NULL`) if row ``j`` was deleted, and ``id_map[j]``
is the new ID of that row, otherwise.

.. todo::
    This needs some examples to link to. See
    https://github.com/tskit-dev/tskit/issues/2708
