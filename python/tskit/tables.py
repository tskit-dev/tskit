#
# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
# Copyright (c) 2017 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Tree sequence IO via the tables API.
"""
import collections.abc
import dataclasses
import datetime
import json
import numbers
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np

import _tskit
import tskit
import tskit.metadata as metadata
import tskit.provenance as provenance
import tskit.util as util
from tskit import UNKNOWN_TIME

dataclass_options = {"frozen": True}


# Needed for cases where `None` can be an appropriate kwarg value,
# we override the meta so that it looks good in the docs.
class NotSetMeta(type):
    def __repr__(cls):
        return "Not set"


class NOTSET(metaclass=NotSetMeta):
    pass


@metadata.lazy_decode()
@dataclass(**dataclass_options)
class IndividualTableRow(util.Dataclass):
    """
    A row in an :class:`IndividualTable`.
    """

    __slots__ = ["flags", "location", "parents", "metadata"]
    flags: int
    """
    See :attr:`Individual.flags`
    """
    location: np.ndarray
    """
    See :attr:`Individual.location`
    """
    parents: np.ndarray
    """
    See :attr:`Individual.parents`
    """
    metadata: Optional[Union[bytes, dict]]
    """
    See :attr:`Individual.metadata`
    """

    # We need a custom eq for the numpy arrays
    def __eq__(self, other):
        return (
            isinstance(other, IndividualTableRow)
            and self.flags == other.flags
            and np.array_equal(self.location, other.location)
            and np.array_equal(self.parents, other.parents)
            and self.metadata == other.metadata
        )


@metadata.lazy_decode()
@dataclass(**dataclass_options)
class NodeTableRow(util.Dataclass):
    """
    A row in a :class:`NodeTable`.
    """

    __slots__ = ["flags", "time", "population", "individual", "metadata"]
    flags: int
    """
    See :attr:`Node.flags`
    """
    time: float
    """
    See :attr:`Node.time`
    """
    population: int
    """
    See :attr:`Node.population`
    """
    individual: int
    """
    See :attr:`Node.individual`
    """
    metadata: Optional[Union[bytes, dict]]
    """
    See :attr:`Node.metadata`
    """


@metadata.lazy_decode()
@dataclass(**dataclass_options)
class EdgeTableRow(util.Dataclass):
    """
    A row in an :class:`EdgeTable`.
    """

    __slots__ = ["left", "right", "parent", "child", "metadata"]
    left: float
    """
    See :attr:`Edge.left`
    """
    right: float
    """
    See :attr:`Edge.right`
    """
    parent: int
    """
    See :attr:`Edge.parent`
    """
    child: int
    """
    See :attr:`Edge.child`
    """
    metadata: Optional[Union[bytes, dict]]
    """
    See :attr:`Edge.metadata`
    """


@metadata.lazy_decode()
@dataclass(**dataclass_options)
class MigrationTableRow(util.Dataclass):
    """
    A row in a :class:`MigrationTable`.
    """

    __slots__ = ["left", "right", "node", "source", "dest", "time", "metadata"]
    left: float
    """
    See :attr:`Migration.left`
    """
    right: float
    """
    See :attr:`Migration.right`
    """
    node: int
    """
    See :attr:`Migration.node`
    """
    source: int
    """
    See :attr:`Migration.source`
    """
    dest: int
    """
    See :attr:`Migration.dest`
    """
    time: float
    """
    See :attr:`Migration.time`
    """
    metadata: Optional[Union[bytes, dict]]
    """
    See :attr:`Migration.metadata`
    """


@metadata.lazy_decode()
@dataclass(**dataclass_options)
class SiteTableRow(util.Dataclass):
    """
    A row in a :class:`SiteTable`.
    """

    __slots__ = ["position", "ancestral_state", "metadata"]
    position: float
    """
    See :attr:`Site.position`
    """
    ancestral_state: str
    """
    See :attr:`Site.ancestral_state`
    """
    metadata: Optional[Union[bytes, dict]]
    """
    See :attr:`Site.metadata`
    """


@metadata.lazy_decode()
@dataclass(**dataclass_options)
class MutationTableRow(util.Dataclass):
    """
    A row in a :class:`MutationTable`.
    """

    __slots__ = ["site", "node", "derived_state", "parent", "metadata", "time"]
    site: int
    """
    See :attr:`Mutation.site`
    """
    node: int
    """
    See :attr:`Mutation.node`
    """
    derived_state: str
    """
    See :attr:`Mutation.derived_state`
    """
    parent: int
    """
    See :attr:`Mutation.parent`
    """
    metadata: Optional[Union[bytes, dict]]
    """
    See :attr:`Mutation.metadata`
    """
    time: float
    """
    See :attr:`Mutation.time`
    """

    # We need a custom eq here as we have unknown times (nans) to check
    def __eq__(self, other):
        return (
            isinstance(other, MutationTableRow)
            and self.site == other.site
            and self.node == other.node
            and self.derived_state == other.derived_state
            and self.parent == other.parent
            and self.metadata == other.metadata
            and (
                self.time == other.time
                or (
                    util.is_unknown_time(self.time) and util.is_unknown_time(other.time)
                )
            )
        )


@metadata.lazy_decode()
@dataclass(**dataclass_options)
class PopulationTableRow(util.Dataclass):
    """
    A row in a :class:`PopulationTable`.
    """

    __slots__ = ["metadata"]
    metadata: Optional[Union[bytes, dict]]
    """
    See :attr:`Population.metadata`
    """


@dataclass(**dataclass_options)
class ProvenanceTableRow(util.Dataclass):
    """
    A row in a :class:`ProvenanceTable`.
    """

    __slots__ = ["timestamp", "record"]
    timestamp: str
    """
    See :attr:`Provenance.timestamp`
    """
    record: str
    """
    See :attr:`Provenance.record`
    """


@dataclass(**dataclass_options)
class TableCollectionIndexes(util.Dataclass):
    """
    A class encapsulating the indexes of a :class:`TableCollection`
    """

    edge_insertion_order: np.ndarray = None
    edge_removal_order: np.ndarray = None

    def asdict(self):
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    @property
    def nbytes(self) -> int:
        """
        The number of bytes taken by the indexes
        """
        total = 0
        if self.edge_removal_order is not None:
            total += self.edge_removal_order.nbytes
        if self.edge_insertion_order is not None:
            total += self.edge_insertion_order.nbytes
        return total


def keep_with_offset(keep, data, offset):
    """
    Used when filtering _offset columns in tables
    """
    # We need the astype here for 32 bit machines
    lens = np.diff(offset).astype(np.int32)
    return (
        data[np.repeat(keep, lens)],
        np.concatenate(
            [
                np.array([0], dtype=offset.dtype),
                np.cumsum(lens[keep], dtype=offset.dtype),
            ]
        ),
    )


class BaseTable:
    """
    Superclass of high-level tables. Not intended for direct instantiation.
    """

    # The list of columns in the table. Must be set by subclasses.
    column_names = []

    def __init__(self, ll_table, row_class):
        self.ll_table = ll_table
        self.row_class = row_class

    def _check_required_args(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                raise TypeError(f"{k} is required")

    @property
    def num_rows(self) -> int:
        return self.ll_table.num_rows

    @property
    def max_rows(self) -> int:
        return self.ll_table.max_rows

    @property
    def max_rows_increment(self) -> int:
        return self.ll_table.max_rows_increment

    @property
    def nbytes(self) -> int:
        """
        Returns the total number of bytes required to store the data
        in this table. Note that this may not be equal to
        the actual memory footprint.
        """
        # It's not ideal that we run asdict() here to do this as we're
        # currently creating copies of the column arrays, so it would
        # be more efficient to have dedicated low-level methods. However,
        # if we do have read-only views on the underlying memory for the
        # column arrays then this will be a perfectly good way of
        # computing the nbytes values and the overhead minimal.
        d = self.asdict()
        nbytes = 0
        # Some tables don't have a metadata_schema
        metadata_schema = d.pop("metadata_schema", None)
        if metadata_schema is not None:
            nbytes += len(metadata_schema.encode())
        nbytes += sum(col.nbytes for col in d.values())
        return nbytes

    def equals(self, other, ignore_metadata=False):
        """
        Returns True if  `self` and `other` are equal. By default, two tables
        are considered equal if their columns and metadata schemas are
        byte-for-byte identical.

        :param other: Another table instance
        :param bool ignore_metadata: If True exclude metadata and metadata schemas
            from the comparison.
        :return: True if other is equal to this table; False otherwise.
        :rtype: bool
        """
        # Note: most tables support ignore_metadata, we can override for those that don't
        ret = False
        if type(other) is type(self):
            ret = bool(
                self.ll_table.equals(other.ll_table, ignore_metadata=ignore_metadata)
            )
        return ret

    def assert_equals(self, other, *, ignore_metadata=False):
        """
        Raise an AssertionError for the first found difference between
        this and another table of the same type.

        :param other: Another table instance
        :param bool ignore_metadata: If True exclude metadata and metadata schemas
            from the comparison.
        """
        if type(other) is not type(self):
            raise AssertionError(f"Types differ: self={type(self)} other={type(other)}")

        # Check using the low-level method to avoid slowly going through everything
        if self.equals(other, ignore_metadata=ignore_metadata):
            return

        if not ignore_metadata and self.metadata_schema != other.metadata_schema:
            raise AssertionError(
                f"{type(self).__name__} metadata schemas differ: "
                f"self={self.metadata_schema} "
                f"other={other.metadata_schema}"
            )

        for n, (row_self, row_other) in enumerate(zip(self, other)):
            if ignore_metadata:
                row_self = dataclasses.replace(row_self, metadata=None)
                row_other = dataclasses.replace(row_other, metadata=None)
            if row_self != row_other:
                self_dict = dataclasses.asdict(self[n])
                other_dict = dataclasses.asdict(other[n])
                diff_string = []
                for col in self_dict.keys():
                    if isinstance(self_dict[col], np.ndarray):
                        equal = np.array_equal(self_dict[col], other_dict[col])
                    else:
                        equal = self_dict[col] == other_dict[col]
                    if not equal:
                        diff_string.append(
                            f"self.{col}={self_dict[col]} other.{col}={other_dict[col]}"
                        )
                diff_string = "\n".join(diff_string)
                raise AssertionError(
                    f"{type(self).__name__} row {n} differs:\n{diff_string}"
                )

        if self.num_rows != other.num_rows:
            raise AssertionError(
                f"{type(self).__name__} number of rows differ: self={self.num_rows} "
                f"other={other.num_rows}"
            )

        raise AssertionError(
            "Tables differ in an undetected way - "
            "this is a bug, please report an issue on gitub"
        )  # pragma: no cover

    def __eq__(self, other):
        return self.equals(other)

    def __len__(self):
        return self.num_rows

    def __getattr__(self, name):
        if name in self.column_names:
            return getattr(self.ll_table, name)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )

    def __setattr__(self, name, value):
        if name in self.column_names:
            d = self.asdict()
            d[name] = value
            self.set_columns(**d)
        else:
            object.__setattr__(self, name, value)

    def _make_row(self, *args):
        return self.row_class(*args)

    def __getitem__(self, index):
        """
        If passed an integer, return the specified row of this table, decoding metadata
        if it is present. Supports negative indexing, e.g. ``table[-5]``.
        If passed a slice, iterable or array return a new table containing the specified
        rows. Similar to numpy fancy indexing, if the array or iterables contains
        booleans then the index acts as a mask, returning those rows for which the mask
        is True. Note that as the result is a new table, the row ids will change as tskit
        row ids are row indexes.

        :param index: the index of a desired row, a slice of the desired rows, an
            iterable or array of the desired row numbers, or a boolean array to use as
            a mask.
        """

        if isinstance(index, numbers.Integral):
            # Single row by integer
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("Index out of bounds")
            return self._make_row(*self.ll_table.get_row(index))
        elif isinstance(index, numbers.Number):
            raise TypeError("Index must be integer, slice or iterable")
        elif isinstance(index, slice):
            index = range(*index.indices(len(self)))
        else:
            index = np.asarray(index)
            if index.dtype == np.bool_:
                if len(index) != len(self):
                    raise IndexError("Boolean index must be same length as table")
                index = np.flatnonzero(index)
            index = util.safe_np_int_cast(index, np.int32)

        ret = self.__class__()
        ret.metadata_schema = self.metadata_schema
        ret.ll_table.extend(self.ll_table, row_indexes=index)

        return ret

    def __setitem__(self, index, new_row):
        """
        Replaces a row of this table at the specified index with information from a
        row-like object. Metadata, will be validated and encoded according to the table's
        :attr:`metadata_schema<tskit.IndividualTable.metadata_schema>`.

        :param index: the index of the row to change
        :param row-like new_row: An object that has attributes corresponding to the
            properties of the new row. Both the objects returned from ``table[i]`` and
            e.g. ``ts.individual(i)`` work for this purpose, along with any other
            object with the correct attributes.
        """
        if isinstance(index, numbers.Integral):
            # Single row by integer
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("Index out of bounds")
        else:
            raise TypeError("Index must be integer")

        row_data = {
            column: getattr(new_row, column)
            for column in self.column_names
            if "_offset" not in column
        }

        # Encode the metadata - note that if this becomes a perf bottleneck it is
        # possible to use the cached, encoded metadata in the row object, rather than
        # decode and reencode
        if "metadata" in row_data:
            row_data["metadata"] = self.metadata_schema.validate_and_encode_row(
                row_data["metadata"]
            )

        self.ll_table.update_row(row_index=index, **row_data)

    def append(self, row):
        """
        Adds a new row to this table and returns the ID of the new row. Metadata, if
        specified, will be validated and encoded according to the table's
        :attr:`metadata_schema<tskit.IndividualTable.metadata_schema>`.

        :param row-like row: An object that has attributes corresponding to the
            properties of the new row. Both the objects returned from ``table[i]`` and
            e.g. ``ts.individual(i)`` work for this purpose, along with any other
            object with the correct attributes.
        :return: The index of the newly added row.
        :rtype: int
        """
        return self.add_row(
            **{
                column: getattr(row, column)
                for column in self.column_names
                if "_offset" not in column
            }
        )

    def replace_with(self, other):
        # Overwrite the contents of this table with a copy of the other table
        self.set_columns(**other.asdict())

    def clear(self):
        """
        Deletes all rows in this table.
        """
        self.ll_table.clear()

    def reset(self):
        # Deprecated alias for clear
        self.clear()

    def truncate(self, num_rows):
        """
        Truncates this table so that the only the first ``num_rows`` are retained.

        :param int num_rows: The number of rows to retain in this table.
        """
        return self.ll_table.truncate(num_rows)

    def keep_rows(self, keep):
        """
        .. include:: substitutions/table_keep_rows_main.rst

        :param array-like keep: The rows to keep as a boolean array. Must
            be the same length as the table, and convertible to a numpy
            array of dtype bool.
        :return: The mapping between old and new row IDs as a numpy
            array (dtype int32).
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        # We do this check here rather than in the C code because calling
        # len() on the input will cause a more readable exception to be
        # raised than the inscrutable errors we get from numpy when
        # converting arguments of the wrong type.
        if len(keep) != len(self):
            msg = (
                "Argument for keep_rows must be a boolean array of "
                "the same length as the table. "
                f"(need:{len(self)}, got:{len(keep)})"
            )
            raise ValueError(msg)
        return self.ll_table.keep_rows(keep)

    # Pickle support
    def __getstate__(self):
        return self.asdict()

    # Unpickle support
    def __setstate__(self, state):
        self.__init__()
        self.set_columns(**state)

    def copy(self):
        """
        Returns a deep copy of this table
        """
        copy = self.__class__()
        copy.set_columns(**self.asdict())
        return copy

    def asdict(self):
        """
        Returns a dictionary mapping the names of the columns in this table
        to the corresponding numpy arrays.
        """
        ret = {col: getattr(self, col) for col in self.column_names}
        # Not all tables have metadata
        try:
            ret["metadata_schema"] = repr(self.metadata_schema)
        except AttributeError:
            pass
        return ret

    def set_columns(self, **kwargs):
        """
        Sets the values for each column in this :class:`Table` using values
        provided in numpy arrays. Overwrites existing data in all the table columns.
        """
        raise NotImplementedError()

    def __str__(self):
        headers, rows = self._text_header_and_rows(
            limit=tskit._print_options["max_lines"]
        )
        return util.unicode_table(rows, header=headers, row_separator=False)

    def _repr_html_(self):
        """
        Called e.g. by jupyter notebooks to render tables
        """
        headers, rows = self._text_header_and_rows(
            limit=tskit._print_options["max_lines"]
        )
        return util.html_table(rows, header=headers)

    def _columns_all_integer(self, *colnames):
        # For displaying floating point values without loads of decimal places
        return all(
            np.all(getattr(self, col) == np.floor(getattr(self, col)))
            for col in colnames
        )


class MetadataTable(BaseTable):
    """
    Base class for tables that have a metadata column.
    """

    # TODO this class has some overlap with the MetadataProvider base class
    # and also the TreeSequence class. These all have methods to deal with
    # schemas and essentially do the same thing (provide a facade for the
    # low-level get/set metadata schemas functionality). We should refactor
    # this so we're only doing it in one place.
    # https://github.com/tskit-dev/tskit/issues/1957
    def __init__(self, ll_table, row_class):
        super().__init__(ll_table, row_class)

    def _make_row(self, *args):
        return self.row_class(*args, metadata_decoder=self.metadata_schema.decode_row)

    def packset_metadata(self, metadatas):
        """
        Packs the specified list of metadata values and updates the ``metadata``
        and ``metadata_offset`` columns. The length of the metadatas array
        must be equal to the number of rows in the table.

        :param list metadatas: A list of metadata bytes values.
        """
        packed, offset = util.pack_bytes(metadatas)
        d = self.asdict()
        d["metadata"] = packed
        d["metadata_offset"] = offset
        self.set_columns(**d)

    @property
    def metadata_schema(self) -> metadata.MetadataSchema:
        """
        The :class:`tskit.MetadataSchema` for this table.
        """
        # This isn't as inefficient as it looks because we're using an LRU cache on
        # the parse_metadata_schema function. Thus, we're really only incurring the
        # cost of creating the unicode string from the low-level schema and looking
        # up the functools cache.
        return metadata.parse_metadata_schema(self.ll_table.metadata_schema)

    @metadata_schema.setter
    def metadata_schema(self, schema: metadata.MetadataSchema) -> None:
        if not isinstance(schema, metadata.MetadataSchema):
            raise TypeError(
                "Only instances of tskit.MetadataSchema can be assigned to "
                f"metadata_schema, not {type(schema)}"
            )
        self.ll_table.metadata_schema = repr(schema)

    def metadata_vector(self, key, *, dtype=None, default_value=NOTSET):
        """
        Returns a numpy array of metadata values obtained by extracting ``key``
        from each metadata entry, and using ``default_value`` if the key is
        not present. ``key`` may be a list, in which case nested values are returned.
        For instance, ``key = ["a", "x"]`` will return an array of
        ``row.metadata["a"]["x"]`` values, iterated over rows in this table.

        :param str key: The name, or a list of names, of metadata entries.
        :param str dtype: The dtype of the result (can usually be omitted).
        :param object default_value: The value to be inserted if the metadata key
            is not present. Note that for numeric columns, a default value of None
            will result in a non-numeric array. The default behaviour is to raise
            ``KeyError`` on missing entries.
        """

        if default_value == NOTSET:

            def getter(d, k):
                return d[k]

        else:

            def getter(d, k):
                return (
                    d.get(k, default_value) if isinstance(d, Mapping) else default_value
                )

        if isinstance(key, list):
            out = np.array(
                [
                    reduce(
                        getter,
                        key,
                        row.metadata,
                    )
                    for row in self
                ],
                dtype=dtype,
            )
        else:
            out = np.array(
                [getter(row.metadata, key) for row in self],
                dtype=dtype,
            )
        return out

    def drop_metadata(self, *, keep_schema=False):
        """
        Drops all metadata in this table. By default, the schema is also cleared,
        except if ``keep_schema`` is True.

        :param bool keep_schema: True if the current schema should be kept intact.
        """
        data = self.asdict()
        data["metadata"] = []
        data["metadata_offset"][:] = 0
        self.set_columns(**data)
        if not keep_schema:
            self.metadata_schema = metadata.MetadataSchema.null()


class IndividualTable(MetadataTable):
    """
    A table defining the individuals in a tree sequence. Note that although
    each Individual has associated nodes, reference to these is not stored in
    the individual table, but rather reference to the individual is stored for
    each node in the :class:`NodeTable`.  This is similar to the way in which
    the relationship between sites and mutations is modelled.

    .. include:: substitutions/table_edit_warning.rst

    :ivar flags: The array of flags values.
    :vartype flags: numpy.ndarray, dtype=np.uint32
    :ivar location: The flattened array of floating point location values. See
        :ref:`sec_encoding_ragged_columns` for more details.
    :vartype location: numpy.ndarray, dtype=np.float64
    :ivar location_offset: The array of offsets into the location column. See
        :ref:`sec_encoding_ragged_columns` for more details.
    :vartype location_offset: numpy.ndarray, dtype=np.uint32
    :ivar parents: The flattened array of parent individual ids. See
        :ref:`sec_encoding_ragged_columns` for more details.
    :vartype parents: numpy.ndarray, dtype=np.int32
    :ivar parents_offset: The array of offsets into the parents column. See
        :ref:`sec_encoding_ragged_columns` for more details.
    :vartype parents_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata: The flattened array of binary metadata values. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata: numpy.ndarray, dtype=np.int8
    :ivar metadata_offset: The array of offsets into the metadata column. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata_schema: The metadata schema for this table's metadata column
    :vartype metadata_schema: tskit.MetadataSchema
    """

    column_names = [
        "flags",
        "location",
        "location_offset",
        "parents",
        "parents_offset",
        "metadata",
        "metadata_offset",
    ]

    def __init__(self, max_rows_increment=0, ll_table=None):
        if ll_table is None:
            ll_table = _tskit.IndividualTable(max_rows_increment=max_rows_increment)
        super().__init__(ll_table, IndividualTableRow)

    def _text_header_and_rows(self, limit=None):
        headers = ("id", "flags", "location", "parents", "metadata")
        rows = []
        row_indexes = util.truncate_rows(self.num_rows, limit)
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                row = self[j]
                location_str = ", ".join(map(str, row.location))
                parents_str = ", ".join(map(str, row.parents))
                rows.append(
                    "{}\t{}\t{}\t{}\t{}".format(
                        j,
                        row.flags,
                        location_str,
                        parents_str,
                        util.render_metadata(row.metadata),
                    ).split("\t")
                )
        return headers, rows

    def add_row(self, flags=0, location=None, parents=None, metadata=None):
        """
        Adds a new row to this :class:`IndividualTable` and returns the ID of the
        corresponding individual. Metadata, if specified, will be validated and encoded
        according to the table's
        :attr:`metadata_schema<tskit.IndividualTable.metadata_schema>`.

        :param int flags: The bitwise flags for the new node.
        :param array-like location: A list of numeric values or one-dimensional numpy
            array describing the location of this individual. If not specified
            or None, a zero-dimensional location is stored.
        :param array-like parents: A list or array of ids of parent individuals. If not
            specified an empty array is stored.
        :param object metadata: Any object that is valid metadata for the table's schema.
            Defaults to the default metadata value for the table's schema. This is
            typically ``{}``. For no schema, ``None``.
        :return: The ID of the newly added individual.
        :rtype: int
        """
        if metadata is None:
            metadata = self.metadata_schema.empty_value
        metadata = self.metadata_schema.validate_and_encode_row(metadata)
        return self.ll_table.add_row(
            flags=flags, location=location, parents=parents, metadata=metadata
        )

    def set_columns(
        self,
        flags=None,
        location=None,
        location_offset=None,
        parents=None,
        parents_offset=None,
        metadata=None,
        metadata_offset=None,
        metadata_schema=None,
    ):
        """
        Sets the values for each column in this :class:`IndividualTable` using the
        values in the specified arrays. Overwrites existing data in all the table
        columns.

        The ``flags`` array is mandatory and defines the number of individuals
        the table will contain.
        The ``location`` and ``location_offset`` parameters must be supplied
        together, and meet the requirements for :ref:`sec_encoding_ragged_columns`.
        The ``parents`` and ``parents_offset`` parameters must be supplied
        together, and meet the requirements for :ref:`sec_encoding_ragged_columns`.
        The ``metadata`` and ``metadata_offset`` parameters must be supplied
        together, and meet the requirements for :ref:`sec_encoding_ragged_columns`.
        See :ref:`sec_tables_api_binary_columns` for more information and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param flags: The bitwise flags for each individual. Required.
        :type flags: numpy.ndarray, dtype=np.uint32
        :param location: The flattened location array. Must be specified along
            with ``location_offset``. If not specified or None, an empty location
            value is stored for each individual.
        :type location: numpy.ndarray, dtype=np.float64
        :param location_offset: The offsets into the ``location`` array.
        :type location_offset: numpy.ndarray, dtype=np.uint32.
        :param parents: The flattened parents array. Must be specified along
            with ``parents_offset``. If not specified or None, an empty parents array
            is stored for each individual.
        :type parents: numpy.ndarray, dtype=np.int32
        :param parents_offset: The offsets into the ``parents`` array.
        :type parents_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each individual.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata_schema: The encoded metadata schema. If None (default)
            do not overwrite the exising schema. Note that a schema will need to be
            encoded as a string, e.g. via ``repr(new_metadata_schema)``.
        :type metadata_schema: str

        """
        self._check_required_args(flags=flags)
        self.ll_table.set_columns(
            dict(
                flags=flags,
                location=location,
                location_offset=location_offset,
                parents=parents,
                parents_offset=parents_offset,
                metadata=metadata,
                metadata_offset=metadata_offset,
                metadata_schema=metadata_schema,
            )
        )

    def append_columns(
        self,
        flags=None,
        location=None,
        location_offset=None,
        parents=None,
        parents_offset=None,
        metadata=None,
        metadata_offset=None,
    ):
        """
        Appends the specified arrays to the end of the columns in this
        :class:`IndividualTable`. This allows many new rows to be added at once.

        The ``flags`` array is mandatory and defines the number of
        extra individuals to add to the table.
        The ``parents`` and ``parents_offset`` parameters must be supplied
        together, and meet the requirements for :ref:`sec_encoding_ragged_columns`.
        The ``location`` and ``location_offset`` parameters must be supplied
        together, and meet the requirements for :ref:`sec_encoding_ragged_columns`.
        The ``metadata`` and ``metadata_offset`` parameters must be supplied
        together, and meet the requirements for :ref:`sec_encoding_ragged_columns`.
        See :ref:`sec_tables_api_binary_columns` for more information and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param flags: The bitwise flags for each individual. Required.
        :type flags: numpy.ndarray, dtype=np.uint32
        :param location: The flattened location array. Must be specified along
            with ``location_offset``. If not specified or None, an empty location
            value is stored for each individual.
        :type location: numpy.ndarray, dtype=np.float64
        :param location_offset: The offsets into the ``location`` array.
        :type location_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each individual.
        :param parents: The flattened parents array. Must be specified along
            with ``parents_offset``. If not specified or None, an empty parents array
            is stored for each individual.
        :type parents: numpy.ndarray, dtype=np.int32
        :param parents_offset: The offsets into the ``parents`` array.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        """
        self._check_required_args(flags=flags)
        self.ll_table.append_columns(
            dict(
                flags=flags,
                location=location,
                location_offset=location_offset,
                parents=parents,
                parents_offset=parents_offset,
                metadata=metadata,
                metadata_offset=metadata_offset,
            )
        )

    def packset_location(self, locations):
        """
        Packs the specified list of location values and updates the ``location``
        and ``location_offset`` columns. The length of the locations array
        must be equal to the number of rows in the table.

        :param list locations: A list of locations interpreted as numpy float64
            arrays.
        """
        packed, offset = util.pack_arrays(locations)
        d = self.asdict()
        d["location"] = packed
        d["location_offset"] = offset
        self.set_columns(**d)

    def packset_parents(self, parents):
        """
        Packs the specified list of parent values and updates the ``parent``
        and ``parent_offset`` columns. The length of the parents array
        must be equal to the number of rows in the table.

        :param list parents: A list of list of parent ids, interpreted as numpy int32
            arrays.
        """
        packed, offset = util.pack_arrays(parents, np.int32)
        d = self.asdict()
        d["parents"] = packed
        d["parents_offset"] = offset
        self.set_columns(**d)

    def keep_rows(self, keep):
        """
        .. include:: substitutions/table_keep_rows_main.rst

        The values in the ``parents`` column are updated according to this
        map, so that reference integrity within the table is maintained.
        As a consequence of this, the values in the ``parents`` column
        for kept rows are bounds-checked and an error raised if they
        are not valid. Rows that are deleted are not checked for
        parent ID integrity.

        If an attempt is made to delete rows that are referred to by
        the ``parents`` column of rows that are retained, an error
        is raised.

        These error conditions are checked before any alterations to
        the table are made.

        :param array-like keep: The rows to keep as a boolean array. Must
            be the same length as the table, and convertible to a numpy
            array of dtype bool.
        :return: The mapping between old and new row IDs as a numpy
            array (dtype int32).
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        return super().keep_rows(keep)


class NodeTable(MetadataTable):
    """
    A table defining the nodes in a tree sequence. See the
    :ref:`definitions <sec_node_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for a node table to be a part of a valid tree sequence.

    .. include:: substitutions/table_edit_warning.rst

    :ivar time: The array of time values.
    :vartype time: numpy.ndarray, dtype=np.float64
    :ivar flags: The array of flags values.
    :vartype flags: numpy.ndarray, dtype=np.uint32
    :ivar population: The array of population IDs.
    :vartype population: numpy.ndarray, dtype=np.int32
    :ivar individual: The array of individual IDs that each node belongs to.
    :vartype individual: numpy.ndarray, dtype=np.int32
    :ivar metadata: The flattened array of binary metadata values. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata: numpy.ndarray, dtype=np.int8
    :ivar metadata_offset: The array of offsets into the metadata column. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata_schema: The metadata schema for this table's metadata column
    :vartype metadata_schema: tskit.MetadataSchema
    """

    column_names = [
        "time",
        "flags",
        "population",
        "individual",
        "metadata",
        "metadata_offset",
    ]

    def __init__(self, max_rows_increment=0, ll_table=None):
        if ll_table is None:
            ll_table = _tskit.NodeTable(max_rows_increment=max_rows_increment)
        super().__init__(ll_table, NodeTableRow)

    def _text_header_and_rows(self, limit=None):
        headers = ("id", "flags", "population", "individual", "time", "metadata")
        rows = []
        row_indexes = util.truncate_rows(self.num_rows, limit)
        decimal_places_times = 0 if self._columns_all_integer("time") else 8
        for j in row_indexes:
            row = self[j]
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                rows.append(
                    "{}\t{}\t{}\t{}\t{:.{dp}f}\t{}".format(
                        j,
                        row.flags,
                        row.population,
                        row.individual,
                        row.time,
                        util.render_metadata(row.metadata),
                        dp=decimal_places_times,
                    ).split("\t")
                )
        return headers, rows

    def add_row(self, flags=0, time=0, population=-1, individual=-1, metadata=None):
        """
        Adds a new row to this :class:`NodeTable` and returns the ID of the
        corresponding node. Metadata, if specified, will be validated and encoded
        according to the table's
        :attr:`metadata_schema<tskit.NodeTable.metadata_schema>`.

        :param int flags: The bitwise flags for the new node.
        :param float time: The birth time for the new node.
        :param int population: The ID of the population in which the new node was born.
            Defaults to :data:`tskit.NULL`.
        :param int individual: The ID of the individual in which the new node was born.
            Defaults to :data:`tskit.NULL`.
        :param object metadata: Any object that is valid metadata for the table's schema.
            Defaults to the default metadata value for the table's schema. This is
            typically ``{}``. For no schema, ``None``.
        :return: The ID of the newly added node.
        :rtype: int
        """
        if metadata is None:
            metadata = self.metadata_schema.empty_value
        metadata = self.metadata_schema.validate_and_encode_row(metadata)
        return self.ll_table.add_row(flags, time, population, individual, metadata)

    def set_columns(
        self,
        flags=None,
        time=None,
        population=None,
        individual=None,
        metadata=None,
        metadata_offset=None,
        metadata_schema=None,
    ):
        """
        Sets the values for each column in this :class:`NodeTable` using the values in
        the specified arrays. Overwrites existing data in all the table columns.

        The ``flags``, ``time`` and ``population`` arrays must all be of the same length,
        which is equal to the number of nodes the table will contain. The
        ``metadata`` and ``metadata_offset`` parameters must be supplied together, and
        meet the requirements for :ref:`sec_encoding_ragged_columns`.
        See :ref:`sec_tables_api_binary_columns` for more information and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param flags: The bitwise flags for each node. Required.
        :type flags: numpy.ndarray, dtype=np.uint32
        :param time: The time values for each node. Required.
        :type time: numpy.ndarray, dtype=np.float64
        :param population: The population values for each node. If not specified
            or None, the :data:`tskit.NULL` value is stored for each node.
        :type population: numpy.ndarray, dtype=np.int32
        :param individual: The individual values for each node. If not specified
            or None, the :data:`tskit.NULL` value is stored for each node.
        :type individual: numpy.ndarray, dtype=np.int32
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata_schema: The encoded metadata schema. If None (default)
            do not overwrite the exising schema. Note that a schema will need to be
            encoded as a string, e.g. via ``repr(new_metadata_schema)``.
        :type metadata_schema: str
        """
        self._check_required_args(flags=flags, time=time)
        self.ll_table.set_columns(
            dict(
                flags=flags,
                time=time,
                population=population,
                individual=individual,
                metadata=metadata,
                metadata_offset=metadata_offset,
                metadata_schema=metadata_schema,
            )
        )

    def append_columns(
        self,
        flags=None,
        time=None,
        population=None,
        individual=None,
        metadata=None,
        metadata_offset=None,
    ):
        """
        Appends the specified arrays to the end of the columns in this
        :class:`NodeTable`. This allows many new rows to be added at once.

        The ``flags``, ``time`` and ``population`` arrays must all be of the same length,
        which is equal to the number of nodes that will be added to the table. The
        ``metadata`` and ``metadata_offset`` parameters must be supplied together, and
        meet the requirements for :ref:`sec_encoding_ragged_columns`.
        See :ref:`sec_tables_api_binary_columns` for more information and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param flags: The bitwise flags for each node. Required.
        :type flags: numpy.ndarray, dtype=np.uint32
        :param time: The time values for each node. Required.
        :type time: numpy.ndarray, dtype=np.float64
        :param population: The population values for each node. If not specified
            or None, the :data:`tskit.NULL` value is stored for each node.
        :type population: numpy.ndarray, dtype=np.int32
        :param individual: The individual values for each node. If not specified
            or None, the :data:`tskit.NULL` value is stored for each node.
        :type individual: numpy.ndarray, dtype=np.int32
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        """
        self._check_required_args(flags=flags, time=time)
        self.ll_table.append_columns(
            dict(
                flags=flags,
                time=time,
                population=population,
                individual=individual,
                metadata=metadata,
                metadata_offset=metadata_offset,
            )
        )


class EdgeTable(MetadataTable):
    """
    A table defining the edges in a tree sequence. See the
    :ref:`definitions <sec_edge_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for an edge table to be a part of a valid tree sequence.

    .. include:: substitutions/table_edit_warning.rst

    :ivar left: The array of left coordinates.
    :vartype left: numpy.ndarray, dtype=np.float64
    :ivar right: The array of right coordinates.
    :vartype right: numpy.ndarray, dtype=np.float64
    :ivar parent: The array of parent node IDs.
    :vartype parent: numpy.ndarray, dtype=np.int32
    :ivar child: The array of child node IDs.
    :vartype child: numpy.ndarray, dtype=np.int32
    :ivar metadata: The flattened array of binary metadata values. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata: numpy.ndarray, dtype=np.int8
    :ivar metadata_offset: The array of offsets into the metadata column. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata_schema: The metadata schema for this table's metadata column
    :vartype metadata_schema: tskit.MetadataSchema
    """

    column_names = [
        "left",
        "right",
        "parent",
        "child",
        "metadata",
        "metadata_offset",
    ]

    def __init__(self, max_rows_increment=0, ll_table=None):
        if ll_table is None:
            ll_table = _tskit.EdgeTable(max_rows_increment=max_rows_increment)
        super().__init__(ll_table, EdgeTableRow)

    def _text_header_and_rows(self, limit=None):
        headers = ("id", "left", "right", "parent", "child", "metadata")
        rows = []
        row_indexes = util.truncate_rows(self.num_rows, limit)
        decimal_places = 0 if self._columns_all_integer("left", "right") else 8
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                row = self[j]
                rows.append(
                    "{}\t{:.{dp}f}\t{:.{dp}f}\t{}\t{}\t{}".format(
                        j,
                        row.left,
                        row.right,
                        row.parent,
                        row.child,
                        util.render_metadata(row.metadata),
                        dp=decimal_places,
                    ).split("\t")
                )
        return headers, rows

    def add_row(self, left, right, parent, child, metadata=None):
        """
        Adds a new row to this :class:`EdgeTable` and returns the ID of the
        corresponding edge. Metadata, if specified, will be validated and encoded
        according to the table's
        :attr:`metadata_schema<tskit.EdgeTable.metadata_schema>`.

        :param float left: The left coordinate (inclusive).
        :param float right: The right coordinate (exclusive).
        :param int parent: The ID of parent node.
        :param int child: The ID of child node.
        :param object metadata: Any object that is valid metadata for the table's schema.
            Defaults to the default metadata value for the table's schema. This is
            typically ``{}``. For no schema, ``None``.
        :return: The ID of the newly added edge.
        :rtype: int
        """
        if metadata is None:
            metadata = self.metadata_schema.empty_value
        metadata = self.metadata_schema.validate_and_encode_row(metadata)
        return self.ll_table.add_row(left, right, parent, child, metadata)

    def set_columns(
        self,
        left=None,
        right=None,
        parent=None,
        child=None,
        metadata=None,
        metadata_offset=None,
        metadata_schema=None,
    ):
        """
        Sets the values for each column in this :class:`EdgeTable` using the values
        in the specified arrays. Overwrites existing data in all the table columns.

        The ``left``, ``right``, ``parent`` and ``child`` parameters are mandatory,
        and must be numpy arrays of the same length (which is equal to the number of
        edges the table will contain).
        The ``metadata`` and ``metadata_offset`` parameters must be supplied together,
        and meet the requirements for :ref:`sec_encoding_ragged_columns`.
        See :ref:`sec_tables_api_binary_columns` for more information and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.


        :param left: The left coordinates (inclusive).
        :type left: numpy.ndarray, dtype=np.float64
        :param right: The right coordinates (exclusive).
        :type right: numpy.ndarray, dtype=np.float64
        :param parent: The parent node IDs.
        :type parent: numpy.ndarray, dtype=np.int32
        :param child: The child node IDs.
        :type child: numpy.ndarray, dtype=np.int32
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata_schema: The encoded metadata schema. If None (default)
            do not overwrite the exising schema. Note that a schema will need to be
            encoded as a string, e.g. via ``repr(new_metadata_schema)``.
        :type metadata_schema: str
        """
        self._check_required_args(left=left, right=right, parent=parent, child=child)
        self.ll_table.set_columns(
            dict(
                left=left,
                right=right,
                parent=parent,
                child=child,
                metadata=metadata,
                metadata_offset=metadata_offset,
                metadata_schema=metadata_schema,
            )
        )

    def append_columns(
        self, left, right, parent, child, metadata=None, metadata_offset=None
    ):
        """
        Appends the specified arrays to the end of the columns of this
        :class:`EdgeTable`. This allows many new rows to be added at once.

        The ``left``, ``right``, ``parent`` and ``child`` parameters are mandatory,
        and must be numpy arrays of the same length (which is equal to the number of
        additional edges to add to the table). The ``metadata`` and
        ``metadata_offset`` parameters must be supplied together, and
        meet the requirements for :ref:`sec_encoding_ragged_columns`.
        See :ref:`sec_tables_api_binary_columns` for more information and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.


        :param left: The left coordinates (inclusive).
        :type left: numpy.ndarray, dtype=np.float64
        :param right: The right coordinates (exclusive).
        :type right: numpy.ndarray, dtype=np.float64
        :param parent: The parent node IDs.
        :type parent: numpy.ndarray, dtype=np.int32
        :param child: The child node IDs.
        :type child: numpy.ndarray, dtype=np.int32
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        """
        self.ll_table.append_columns(
            dict(
                left=left,
                right=right,
                parent=parent,
                child=child,
                metadata=metadata,
                metadata_offset=metadata_offset,
            )
        )

    def squash(self):
        """
        Sorts, then condenses the table into the smallest possible number of rows by
        combining any adjacent edges.
        A pair of edges is said to be `adjacent` if they have the same parent and child
        nodes, and if the left coordinate of one of the edges is equal to the right
        coordinate of the other edge.
        The ``squash`` method modifies an :class:`EdgeTable` in place so that any set of
        adjacent edges is replaced by a single edge.
        The new edge will have the same parent and child node, a left coordinate
        equal to the smallest left coordinate in the set, and a right coordinate
        equal to the largest right coordinate in the set.
        The new edge table will be sorted in the order (P, C, L, R): if the node table
        is ordered by increasing node time, as is common, this order will meet the
        :ref:`sec_edge_requirements` for a valid tree sequence, otherwise you will need
        to call :meth:`.sort` on the entire :class:`TableCollection`.

        .. note::
            Note that this method will fail if any edges have non-empty metadata.

        """
        self.ll_table.squash()


class MigrationTable(MetadataTable):
    """
    A table defining the migrations in a tree sequence. See the
    :ref:`definitions <sec_migration_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for a migration table to be a part of a valid tree
    sequence.

    .. include:: substitutions/table_edit_warning.rst

    :ivar left: The array of left coordinates.
    :vartype left: numpy.ndarray, dtype=np.float64
    :ivar right: The array of right coordinates.
    :vartype right: numpy.ndarray, dtype=np.float64
    :ivar node: The array of node IDs.
    :vartype node: numpy.ndarray, dtype=np.int32
    :ivar source: The array of source population IDs.
    :vartype source: numpy.ndarray, dtype=np.int32
    :ivar dest: The array of destination population IDs.
    :vartype dest: numpy.ndarray, dtype=np.int32
    :ivar time: The array of time values.
    :vartype time: numpy.ndarray, dtype=np.float64
    :ivar metadata: The flattened array of binary metadata values. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata: numpy.ndarray, dtype=np.int8
    :ivar metadata_offset: The array of offsets into the metadata column. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata_schema: The metadata schema for this table's metadata column
    :vartype metadata_schema: tskit.MetadataSchema
    """

    column_names = [
        "left",
        "right",
        "node",
        "source",
        "dest",
        "time",
        "metadata",
        "metadata_offset",
    ]

    def __init__(self, max_rows_increment=0, ll_table=None):
        if ll_table is None:
            ll_table = _tskit.MigrationTable(max_rows_increment=max_rows_increment)
        super().__init__(ll_table, MigrationTableRow)

    def _text_header_and_rows(self, limit=None):
        headers = ("id", "left", "right", "node", "source", "dest", "time", "metadata")
        rows = []
        row_indexes = util.truncate_rows(self.num_rows, limit)
        decimal_places_coords = 0 if self._columns_all_integer("left", "right") else 8
        decimal_places_times = 0 if self._columns_all_integer("time") else 8
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                row = self[j]
                rows.append(
                    "{}\t{:.{dp_c}f}\t{:.{dp_c}f}\t{}\t{}\t{}\t{:.{dp_t}f}\t{}".format(
                        j,
                        row.left,
                        row.right,
                        row.node,
                        row.source,
                        row.dest,
                        row.time,
                        util.render_metadata(row.metadata),
                        dp_c=decimal_places_coords,
                        dp_t=decimal_places_times,
                    ).split("\t")
                )
        return headers, rows

    def add_row(self, left, right, node, source, dest, time, metadata=None):
        """
        Adds a new row to this :class:`MigrationTable` and returns the ID of the
        corresponding migration. Metadata, if specified, will be validated and encoded
        according to the table's
        :attr:`metadata_schema<tskit.MigrationTable.metadata_schema>`.

        :param float left: The left coordinate (inclusive).
        :param float right: The right coordinate (exclusive).
        :param int node: The node ID.
        :param int source: The ID of the source population.
        :param int dest: The ID of the destination population.
        :param float time: The time of the migration event.
        :param object metadata: Any object that is valid metadata for the table's schema.
            Defaults to the default metadata value for the table's schema. This is
            typically ``{}``. For no schema, ``None``.
        :return: The ID of the newly added migration.
        :rtype: int
        """
        if metadata is None:
            metadata = self.metadata_schema.empty_value
        metadata = self.metadata_schema.validate_and_encode_row(metadata)
        return self.ll_table.add_row(left, right, node, source, dest, time, metadata)

    def set_columns(
        self,
        left=None,
        right=None,
        node=None,
        source=None,
        dest=None,
        time=None,
        metadata=None,
        metadata_offset=None,
        metadata_schema=None,
    ):
        """
        Sets the values for each column in this :class:`MigrationTable` using the values
        in the specified arrays. Overwrites existing data in all the table columns.

        All parameters except ``metadata`` and ``metadata_offset`` and are mandatory,
        and must be numpy arrays of the same length (which is equal to the number of
        migrations the table will contain).
        The ``metadata`` and ``metadata_offset`` parameters must be supplied together,
        and meet the requirements for :ref:`sec_encoding_ragged_columns`.
        See :ref:`sec_tables_api_binary_columns` for more information and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param left: The left coordinates (inclusive).
        :type left: numpy.ndarray, dtype=np.float64
        :param right: The right coordinates (exclusive).
        :type right: numpy.ndarray, dtype=np.float64
        :param node: The node IDs.
        :type node: numpy.ndarray, dtype=np.int32
        :param source: The source population IDs.
        :type source: numpy.ndarray, dtype=np.int32
        :param dest: The destination population IDs.
        :type dest: numpy.ndarray, dtype=np.int32
        :param time: The time of each migration.
        :type time: numpy.ndarray, dtype=np.int64
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each migration.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata_schema: The encoded metadata schema. If None (default)
            do not overwrite the exising schema. Note that a schema will need to be
            encoded as a string, e.g. via ``repr(new_metadata_schema)``.
        :type metadata_schema: str
        """
        self._check_required_args(
            left=left, right=right, node=node, source=source, dest=dest, time=time
        )
        self.ll_table.set_columns(
            dict(
                left=left,
                right=right,
                node=node,
                source=source,
                dest=dest,
                time=time,
                metadata=metadata,
                metadata_offset=metadata_offset,
                metadata_schema=metadata_schema,
            )
        )

    def append_columns(
        self,
        left,
        right,
        node,
        source,
        dest,
        time,
        metadata=None,
        metadata_offset=None,
    ):
        """
        Appends the specified arrays to the end of the columns of this
        :class:`MigrationTable`. This allows many new rows to be added at once.

        All parameters except ``metadata`` and ``metadata_offset`` and are mandatory,
        and must be numpy arrays of the same length (which is equal to the number of
        additional migrations to add to the table). The ``metadata`` and
        ``metadata_offset`` parameters must be supplied together, and
        meet the requirements for :ref:`sec_encoding_ragged_columns`.
        See :ref:`sec_tables_api_binary_columns` for more information and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param left: The left coordinates (inclusive).
        :type left: numpy.ndarray, dtype=np.float64
        :param right: The right coordinates (exclusive).
        :type right: numpy.ndarray, dtype=np.float64
        :param node: The node IDs.
        :type node: numpy.ndarray, dtype=np.int32
        :param source: The source population IDs.
        :type source: numpy.ndarray, dtype=np.int32
        :param dest: The destination population IDs.
        :type dest: numpy.ndarray, dtype=np.int32
        :param time: The time of each migration.
        :type time: numpy.ndarray, dtype=np.int64
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each migration.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        """
        self.ll_table.append_columns(
            dict(
                left=left,
                right=right,
                node=node,
                source=source,
                dest=dest,
                time=time,
                metadata=metadata,
                metadata_offset=metadata_offset,
            )
        )


class SiteTable(MetadataTable):
    """
    A table defining the sites in a tree sequence. See the
    :ref:`definitions <sec_site_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for a site table to be a part of a valid tree
    sequence.

    .. include:: substitutions/table_edit_warning.rst

    :ivar position: The array of site position coordinates.
    :vartype position: numpy.ndarray, dtype=np.float64
    :ivar ancestral_state: The flattened array of ancestral state strings.
        See :ref:`sec_tables_api_text_columns` for more details.
    :vartype ancestral_state: numpy.ndarray, dtype=np.int8
    :ivar ancestral_state_offset: The offsets of rows in the ancestral_state
        array. See :ref:`sec_tables_api_text_columns` for more details.
    :vartype ancestral_state_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata: The flattened array of binary metadata values. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata: numpy.ndarray, dtype=np.int8
    :ivar metadata_offset: The array of offsets into the metadata column. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata_schema: The metadata schema for this table's metadata column
    :vartype metadata_schema: tskit.MetadataSchema
    """

    column_names = [
        "position",
        "ancestral_state",
        "ancestral_state_offset",
        "metadata",
        "metadata_offset",
    ]

    def __init__(self, max_rows_increment=0, ll_table=None):
        if ll_table is None:
            ll_table = _tskit.SiteTable(max_rows_increment=max_rows_increment)
        super().__init__(ll_table, SiteTableRow)

    def _text_header_and_rows(self, limit=None):
        headers = ("id", "position", "ancestral_state", "metadata")
        rows = []
        row_indexes = util.truncate_rows(self.num_rows, limit)
        decimal_places = 0 if self._columns_all_integer("position") else 8
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                row = self[j]
                rows.append(
                    "{}\t{:.{dp}f}\t{}\t{}".format(
                        j,
                        row.position,
                        row.ancestral_state,
                        util.render_metadata(row.metadata),
                        dp=decimal_places,
                    ).split("\t")
                )
        return headers, rows

    def add_row(self, position, ancestral_state, metadata=None):
        """
        Adds a new row to this :class:`SiteTable` and returns the ID of the
        corresponding site. Metadata, if specified, will be validated and encoded
        according to the table's
        :attr:`metadata_schema<tskit.SiteTable.metadata_schema>`.

        :param float position: The position of this site in genome coordinates.
        :param str ancestral_state: The state of this site at the root of the tree.
        :param object metadata: Any object that is valid metadata for the table's schema.
            Defaults to the default metadata value for the table's schema. This is
            typically ``{}``. For no schema, ``None``.
        :return: The ID of the newly added site.
        :rtype: int
        """
        if metadata is None:
            metadata = self.metadata_schema.empty_value
        metadata = self.metadata_schema.validate_and_encode_row(metadata)
        return self.ll_table.add_row(position, ancestral_state, metadata)

    def set_columns(
        self,
        position=None,
        ancestral_state=None,
        ancestral_state_offset=None,
        metadata=None,
        metadata_offset=None,
        metadata_schema=None,
    ):
        """
        Sets the values for each column in this :class:`SiteTable` using the values
        in the specified arrays. Overwrites existing data in all the table columns.

        The ``position``, ``ancestral_state`` and ``ancestral_state_offset``
        parameters are mandatory, and must be 1D numpy arrays. The length
        of the ``position`` array determines the number of rows in table.
        The ``ancestral_state`` and ``ancestral_state_offset`` parameters must
        be supplied together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_text_columns` for more information). The
        ``metadata`` and ``metadata_offset`` parameters must be supplied
        together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_binary_columns` for more information) and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param position: The position of each site in genome coordinates.
        :type position: numpy.ndarray, dtype=np.float64
        :param ancestral_state: The flattened ancestral_state array. Required.
        :type ancestral_state: numpy.ndarray, dtype=np.int8
        :param ancestral_state_offset: The offsets into the ``ancestral_state`` array.
        :type ancestral_state_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata_schema: The encoded metadata schema. If None (default)
            do not overwrite the exising schema. Note that a schema will need to be
            encoded as a string, e.g. via ``repr(new_metadata_schema)``.
        :type metadata_schema: str
        """
        self._check_required_args(
            position=position,
            ancestral_state=ancestral_state,
            ancestral_state_offset=ancestral_state_offset,
        )
        self.ll_table.set_columns(
            dict(
                position=position,
                ancestral_state=ancestral_state,
                ancestral_state_offset=ancestral_state_offset,
                metadata=metadata,
                metadata_offset=metadata_offset,
                metadata_schema=metadata_schema,
            )
        )

    def append_columns(
        self,
        position,
        ancestral_state,
        ancestral_state_offset,
        metadata=None,
        metadata_offset=None,
    ):
        """
        Appends the specified arrays to the end of the columns of this
        :class:`SiteTable`. This allows many new rows to be added at once.

        The ``position``, ``ancestral_state`` and ``ancestral_state_offset``
        parameters are mandatory, and must be 1D numpy arrays. The length
        of the ``position`` array determines the number of additional rows
        to add the table.
        The ``ancestral_state`` and ``ancestral_state_offset`` parameters must
        be supplied together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_text_columns` for more information). The
        ``metadata`` and ``metadata_offset`` parameters must be supplied
        together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_binary_columns` for more information) and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param position: The position of each site in genome coordinates.
        :type position: numpy.ndarray, dtype=np.float64
        :param ancestral_state: The flattened ancestral_state array. Required.
        :type ancestral_state: numpy.ndarray, dtype=np.int8
        :param ancestral_state_offset: The offsets into the ``ancestral_state`` array.
        :type ancestral_state_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        """
        self.ll_table.append_columns(
            dict(
                position=position,
                ancestral_state=ancestral_state,
                ancestral_state_offset=ancestral_state_offset,
                metadata=metadata,
                metadata_offset=metadata_offset,
            )
        )

    def packset_ancestral_state(self, ancestral_states):
        """
        Packs the specified list of ancestral_state values and updates the
        ``ancestral_state`` and ``ancestral_state_offset`` columns. The length
        of the ancestral_states array must be equal to the number of rows in
        the table.

        :param list(str) ancestral_states: A list of string ancestral state values.
        """
        packed, offset = util.pack_strings(ancestral_states)
        d = self.asdict()
        d["ancestral_state"] = packed
        d["ancestral_state_offset"] = offset
        self.set_columns(**d)


class MutationTable(MetadataTable):
    """
    A table defining the mutations in a tree sequence. See the
    :ref:`definitions <sec_mutation_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for a mutation table to be a part of a valid tree
    sequence.

    .. include:: substitutions/table_edit_warning.rst

    :ivar site: The array of site IDs.
    :vartype site: numpy.ndarray, dtype=np.int32
    :ivar node: The array of node IDs.
    :vartype node: numpy.ndarray, dtype=np.int32
    :ivar time: The array of time values.
    :vartype time: numpy.ndarray, dtype=np.float64
    :ivar derived_state: The flattened array of derived state strings.
        See :ref:`sec_tables_api_text_columns` for more details.
    :vartype derived_state: numpy.ndarray, dtype=np.int8
    :ivar derived_state_offset: The offsets of rows in the derived_state
        array. See :ref:`sec_tables_api_text_columns` for more details.
    :vartype derived_state_offset: numpy.ndarray, dtype=np.uint32
    :ivar parent: The array of parent mutation IDs.
    :vartype parent: numpy.ndarray, dtype=np.int32
    :ivar metadata: The flattened array of binary metadata values. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata: numpy.ndarray, dtype=np.int8
    :ivar metadata_offset: The array of offsets into the metadata column. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata_schema: The metadata schema for this table's metadata column
    :vartype metadata_schema: tskit.MetadataSchema
    """

    column_names = [
        "site",
        "node",
        "time",
        "derived_state",
        "derived_state_offset",
        "parent",
        "metadata",
        "metadata_offset",
    ]

    def __init__(self, max_rows_increment=0, ll_table=None):
        if ll_table is None:
            ll_table = _tskit.MutationTable(max_rows_increment=max_rows_increment)
        super().__init__(ll_table, MutationTableRow)

    def _text_header_and_rows(self, limit=None):
        headers = ("id", "site", "node", "time", "derived_state", "parent", "metadata")
        rows = []
        row_indexes = util.truncate_rows(self.num_rows, limit)
        # Currently mutations do not have discretised times: this for consistency
        decimal_places_times = 0 if self._columns_all_integer("time") else 8
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                row = self[j]
                rows.append(
                    "{}\t{}\t{}\t{:.{dp}f}\t{}\t{}\t{}".format(
                        j,
                        row.site,
                        row.node,
                        row.time,
                        row.derived_state,
                        row.parent,
                        util.render_metadata(row.metadata),
                        dp=decimal_places_times,
                    ).split("\t")
                )
        return headers, rows

    def add_row(self, site, node, derived_state, parent=-1, metadata=None, time=None):
        """
        Adds a new row to this :class:`MutationTable` and returns the ID of the
        corresponding mutation. Metadata, if specified, will be validated and encoded
        according to the table's
        :attr:`metadata_schema<tskit.MutationTable.metadata_schema>`.

        :param int site: The ID of the site that this mutation occurs at.
        :param int node: The ID of the first node inheriting this mutation.
        :param str derived_state: The state of the site at this mutation's node.
        :param int parent: The ID of the parent mutation. If not specified,
            defaults to :attr:`NULL`.
        :param object metadata: Any object that is valid metadata for the table's schema.
            Defaults to the default metadata value for the table's schema. This is
            typically ``{}``. For no schema, ``None``.
        :return: The ID of the newly added mutation.
        :param float time: The occurrence time for the new mutation. If not specified,
            defaults to ``UNKNOWN_TIME``, indicating the time is unknown.
        :rtype: int
        """
        if metadata is None:
            metadata = self.metadata_schema.empty_value
        metadata = self.metadata_schema.validate_and_encode_row(metadata)
        return self.ll_table.add_row(
            site,
            node,
            derived_state,
            parent,
            metadata,
            UNKNOWN_TIME if time is None else time,
        )

    def set_columns(
        self,
        site=None,
        node=None,
        time=None,
        derived_state=None,
        derived_state_offset=None,
        parent=None,
        metadata=None,
        metadata_offset=None,
        metadata_schema=None,
    ):
        """
        Sets the values for each column in this :class:`MutationTable` using the values
        in the specified arrays. Overwrites existing data in all the the table columns.

        The ``site``, ``node``, ``derived_state`` and ``derived_state_offset``
        parameters are mandatory, and must be 1D numpy arrays. The
        ``site`` and ``node`` (also ``parent`` and ``time``, if supplied) arrays
        must be of equal length, and determine the number of rows in the table.
        The ``derived_state`` and ``derived_state_offset`` parameters must
        be supplied together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_text_columns` for more information). The
        ``metadata`` and ``metadata_offset`` parameters must be supplied
        together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_binary_columns` for more information) and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param site: The ID of the site each mutation occurs at.
        :type site: numpy.ndarray, dtype=np.int32
        :param node: The ID of the node each mutation is associated with.
        :type node: numpy.ndarray, dtype=np.int32
        :param time: The time values for each mutation.
        :type time: numpy.ndarray, dtype=np.float64
        :param derived_state: The flattened derived_state array. Required.
        :type derived_state: numpy.ndarray, dtype=np.int8
        :param derived_state_offset: The offsets into the ``derived_state`` array.
        :type derived_state_offset: numpy.ndarray, dtype=np.uint32.
        :param parent: The ID of the parent mutation for each mutation.
        :type parent: numpy.ndarray, dtype=np.int32
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata_schema: The encoded metadata schema. If None (default)
            do not overwrite the exising schema. Note that a schema will need to be
            encoded as a string, e.g. via ``repr(new_metadata_schema)``.
        :type metadata_schema: str
        """
        self._check_required_args(
            site=site,
            node=node,
            derived_state=derived_state,
            derived_state_offset=derived_state_offset,
        )
        self.ll_table.set_columns(
            dict(
                site=site,
                node=node,
                parent=parent,
                time=time,
                derived_state=derived_state,
                derived_state_offset=derived_state_offset,
                metadata=metadata,
                metadata_offset=metadata_offset,
                metadata_schema=metadata_schema,
            )
        )

    def append_columns(
        self,
        site,
        node,
        derived_state,
        derived_state_offset,
        parent=None,
        time=None,
        metadata=None,
        metadata_offset=None,
    ):
        """
        Appends the specified arrays to the end of the columns of this
        :class:`MutationTable`. This allows many new rows to be added at once.

        The ``site``, ``node``, ``derived_state`` and ``derived_state_offset``
        parameters are mandatory, and must be 1D numpy arrays. The
        ``site`` and ``node`` (also ``time`` and ``parent``, if supplied) arrays
        must be of equal length, and determine the number of additional
        rows to add to the table.
        The ``derived_state`` and ``derived_state_offset`` parameters must
        be supplied together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_text_columns` for more information). The
        ``metadata`` and ``metadata_offset`` parameters must be supplied
        together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_binary_columns` for more information) and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param site: The ID of the site each mutation occurs at.
        :type site: numpy.ndarray, dtype=np.int32
        :param node: The ID of the node each mutation is associated with.
        :type node: numpy.ndarray, dtype=np.int32
        :param time: The time values for each mutation.
        :type time: numpy.ndarray, dtype=np.float64
        :param derived_state: The flattened derived_state array. Required.
        :type derived_state: numpy.ndarray, dtype=np.int8
        :param derived_state_offset: The offsets into the ``derived_state`` array.
        :type derived_state_offset: numpy.ndarray, dtype=np.uint32.
        :param parent: The ID of the parent mutation for each mutation.
        :type parent: numpy.ndarray, dtype=np.int32
        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        """
        self.ll_table.append_columns(
            dict(
                site=site,
                node=node,
                time=time,
                parent=parent,
                derived_state=derived_state,
                derived_state_offset=derived_state_offset,
                metadata=metadata,
                metadata_offset=metadata_offset,
            )
        )

    def packset_derived_state(self, derived_states):
        """
        Packs the specified list of derived_state values and updates the
        ``derived_state`` and ``derived_state_offset`` columns. The length
        of the derived_states array must be equal to the number of rows in
        the table.

        :param list(str) derived_states: A list of string derived state values.
        """
        packed, offset = util.pack_strings(derived_states)
        d = self.asdict()
        d["derived_state"] = packed
        d["derived_state_offset"] = offset
        self.set_columns(**d)

    def keep_rows(self, keep):
        """
        .. include:: substitutions/table_keep_rows_main.rst

        The values in the ``parent`` column are updated according to this
        map, so that reference integrity within the table is maintained.
        As a consequence of this, the values in the ``parent`` column
        for kept rows are bounds-checked and an error raised if they
        are not valid. Rows that are deleted are not checked for
        parent ID integrity.

        If an attempt is made to delete rows that are referred to by
        the ``parent`` column of rows that are retained, an error
        is raised.

        These error conditions are checked before any alterations to
        the table are made.

        :param array-like keep: The rows to keep as a boolean array. Must
            be the same length as the table, and convertible to a numpy
            array of dtype bool.
        :return: The mapping between old and new row IDs as a numpy
            array (dtype int32).
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        return super().keep_rows(keep)


class PopulationTable(MetadataTable):
    """
    A table defining the populations referred to in a tree sequence.
    The PopulationTable stores metadata for populations that may be referred to
    in the NodeTable and MigrationTable".  Note that although nodes
    may be associated with populations, this association is stored in
    the :class:`NodeTable`: only metadata on each population is stored
    in the population table.

    .. include:: substitutions/table_edit_warning.rst

    :ivar metadata: The flattened array of binary metadata values. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata: numpy.ndarray, dtype=np.int8
    :ivar metadata_offset: The array of offsets into the metadata column. See
        :ref:`sec_tables_api_binary_columns` for more details.
    :vartype metadata_offset: numpy.ndarray, dtype=np.uint32
    :ivar metadata_schema: The metadata schema for this table's metadata column
    :vartype metadata_schema: tskit.MetadataSchema
    """

    column_names = ["metadata", "metadata_offset"]

    def __init__(self, max_rows_increment=0, ll_table=None):
        if ll_table is None:
            ll_table = _tskit.PopulationTable(max_rows_increment=max_rows_increment)
        super().__init__(ll_table, PopulationTableRow)

    def add_row(self, metadata=None):
        """
        Adds a new row to this :class:`PopulationTable` and returns the ID of the
        corresponding population. Metadata, if specified, will be validated and encoded
        according to the table's
        :attr:`metadata_schema<tskit.PopulationTable.metadata_schema>`.

        :param object metadata: Any object that is valid metadata for the table's schema.
            Defaults to the default metadata value for the table's schema. This is
            typically ``{}``. For no schema, ``None``.
        :return: The ID of the newly added population.
        :rtype: int
        """
        if metadata is None:
            metadata = self.metadata_schema.empty_value
        metadata = self.metadata_schema.validate_and_encode_row(metadata)
        return self.ll_table.add_row(metadata=metadata)

    def _text_header_and_rows(self, limit=None):
        headers = ("id", "metadata")
        rows = []
        row_indexes = util.truncate_rows(self.num_rows, limit)
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                rows.append((str(j), util.render_metadata(self[j].metadata, length=70)))
        return headers, rows

    def set_columns(self, metadata=None, metadata_offset=None, metadata_schema=None):
        """
        Sets the values for each column in this :class:`PopulationTable` using the
        values in the specified arrays. Overwrites existing data in all the table
        columns.

        The ``metadata`` and ``metadata_offset`` parameters must be supplied
        together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_binary_columns` for more information) and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        :param metadata_schema: The encoded metadata schema. If None (default)
            do not overwrite the exising schema. Note that a schema will need to be
            encoded as a string, e.g. via ``repr(new_metadata_schema)``.
        :type metadata_schema: str
        """
        self.ll_table.set_columns(
            dict(
                metadata=metadata,
                metadata_offset=metadata_offset,
                metadata_schema=metadata_schema,
            )
        )

    def append_columns(self, metadata=None, metadata_offset=None):
        """
        Appends the specified arrays to the end of the columns of this
        :class:`PopulationTable`. This allows many new rows to be added at once.

        The ``metadata`` and ``metadata_offset`` parameters must be supplied
        together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_binary_columns` for more information) and
        :ref:`sec_tutorial_metadata_bulk` for an example of how to prepare metadata.

        :param metadata: The flattened metadata array. Must be specified along
            with ``metadata_offset``. If not specified or None, an empty metadata
            value is stored for each node.
        :type metadata: numpy.ndarray, dtype=np.int8
        :param metadata_offset: The offsets into the ``metadata`` array.
        :type metadata_offset: numpy.ndarray, dtype=np.uint32.
        """
        self.ll_table.append_columns(
            dict(metadata=metadata, metadata_offset=metadata_offset)
        )


class ProvenanceTable(BaseTable):
    """
    A table recording the provenance (i.e., history) of this table, so that the
    origin of the underlying data and sequence of subsequent operations can be
    traced. Each row contains a "record" string (recommended format: JSON) and
    a timestamp.

    .. todo::
        The format of the `record` field will be more precisely specified in
        the future.

    :ivar record: The flattened array containing the record strings.
        :ref:`sec_tables_api_text_columns` for more details.
    :vartype record: numpy.ndarray, dtype=np.int8
    :ivar record_offset: The array of offsets into the record column. See
        :ref:`sec_tables_api_text_columns` for more details.
    :vartype record_offset: numpy.ndarray, dtype=np.uint32
    :ivar timestamp: The flattened array containing the timestamp strings.
        :ref:`sec_tables_api_text_columns` for more details.
    :vartype timestamp: numpy.ndarray, dtype=np.int8
    :ivar timestamp_offset: The array of offsets into the timestamp column. See
        :ref:`sec_tables_api_text_columns` for more details.
    :vartype timestamp_offset: numpy.ndarray, dtype=np.uint32
    """

    column_names = ["record", "record_offset", "timestamp", "timestamp_offset"]

    def __init__(self, max_rows_increment=0, ll_table=None):
        if ll_table is None:
            ll_table = _tskit.ProvenanceTable(max_rows_increment=max_rows_increment)
        super().__init__(ll_table, ProvenanceTableRow)

    def equals(self, other, ignore_timestamps=False):
        """
        Returns True if  `self` and `other` are equal. By default, two provenance
        tables are considered equal if their columns are byte-for-byte identical.

        :param other: Another provenance table instance
        :param bool ignore_timestamps: If True exclude the timestamp column
            from the comparison.
        :return: True if other is equal to this provenance table; False otherwise.
        :rtype: bool
        """
        ret = False
        if type(other) is type(self):
            ret = bool(
                self.ll_table.equals(
                    other.ll_table, ignore_timestamps=ignore_timestamps
                )
            )
        return ret

    def assert_equals(self, other, *, ignore_timestamps=False):
        """
        Raise an AssertionError for the first found difference between
        this and another provenance table.

        :param other: Another provenance table instance
        :param bool ignore_timestamps: If True exclude the timestamp column
            from the comparison.
        """
        if type(other) is not type(self):
            raise AssertionError(f"Types differ: self={type(self)} other={type(other)}")

        # Check using the low-level method to avoid slowly going through everything
        if self.equals(other, ignore_timestamps=ignore_timestamps):
            return

        for n, (row_self, row_other) in enumerate(zip(self, other)):
            if ignore_timestamps:
                row_self = dataclasses.replace(row_self, timestamp=None)
                row_other = dataclasses.replace(row_other, timestamp=None)
            if row_self != row_other:
                self_dict = dataclasses.asdict(self[n])
                other_dict = dataclasses.asdict(other[n])
                diff_string = []
                for col in self_dict.keys():
                    if self_dict[col] != other_dict[col]:
                        diff_string.append(
                            f"self.{col}={self_dict[col]} other.{col}={other_dict[col]}"
                        )
                diff_string = "\n".join(diff_string)
                raise AssertionError(
                    f"{type(self).__name__} row {n} differs:\n{diff_string}"
                )

        if self.num_rows != other.num_rows:
            raise AssertionError(
                f"{type(self).__name__} number of rows differ: self={self.num_rows} "
                f"other={other.num_rows}"
            )

        raise AssertionError(
            "Tables differ in an undetected way - "
            "this is a bug, please report an issue on gitub"
        )  # pragma: no cover

    def add_row(self, record, timestamp=None):
        """
        Adds a new row to this ProvenanceTable consisting of the specified record and
        timestamp. If timestamp is not specified, it is automatically generated from
        the current time.

        :param str record: A provenance record, describing the parameters and
            environment used to generate the current set of tables.
        :param str timestamp: A string timestamp. This should be in ISO8601 form.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        # Note that the order of the positional arguments has been reversed
        # from the low-level module, which is a bit confusing. However, we
        # want the default behaviour here to be to add a row to the table at
        # the current time as simply as possible.
        return self.ll_table.add_row(record=record, timestamp=timestamp)

    def set_columns(
        self, timestamp=None, timestamp_offset=None, record=None, record_offset=None
    ):
        """
        Sets the values for each column in this :class:`ProvenanceTable` using the
        values in the specified arrays. Overwrites existing data in all the table
        columns.

        The ``timestamp`` and ``timestamp_offset`` parameters must be supplied
        together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_binary_columns` for more information). Likewise
        for the ``record`` and ``record_offset`` columns

        :param timestamp: The flattened timestamp array. Must be specified along
            with ``timestamp_offset``. If not specified or None, an empty timestamp
            value is stored for each node.
        :type timestamp: numpy.ndarray, dtype=np.int8
        :param timestamp_offset: The offsets into the ``timestamp`` array.
        :type timestamp_offset: numpy.ndarray, dtype=np.uint32.
        :param record: The flattened record array. Must be specified along
            with ``record_offset``. If not specified or None, an empty record
            value is stored for each node.
        :type record: numpy.ndarray, dtype=np.int8
        :param record_offset: The offsets into the ``record`` array.
        :type record_offset: numpy.ndarray, dtype=np.uint32.
        """
        self.ll_table.set_columns(
            dict(
                timestamp=timestamp,
                timestamp_offset=timestamp_offset,
                record=record,
                record_offset=record_offset,
            )
        )

    def append_columns(
        self, timestamp=None, timestamp_offset=None, record=None, record_offset=None
    ):
        """
        Appends the specified arrays to the end of the columns of this
        :class:`ProvenanceTable`. This allows many new rows to be added at once.

        The ``timestamp`` and ``timestamp_offset`` parameters must be supplied
        together, and meet the requirements for
        :ref:`sec_encoding_ragged_columns` (see
        :ref:`sec_tables_api_binary_columns` for more information). Likewise
        for the ``record`` and ``record_offset`` columns

        :param timestamp: The flattened timestamp array. Must be specified along
            with ``timestamp_offset``. If not specified or None, an empty timestamp
            value is stored for each node.
        :type timestamp: numpy.ndarray, dtype=np.int8
        :param timestamp_offset: The offsets into the ``timestamp`` array.
        :type timestamp_offset: numpy.ndarray, dtype=np.uint32.
        :param record: The flattened record array. Must be specified along
            with ``record_offset``. If not specified or None, an empty record
            value is stored for each node.
        :type record: numpy.ndarray, dtype=np.int8
        :param record_offset: The offsets into the ``record`` array.
        :type record_offset: numpy.ndarray, dtype=np.uint32.
        """
        self.ll_table.append_columns(
            dict(
                timestamp=timestamp,
                timestamp_offset=timestamp_offset,
                record=record,
                record_offset=record_offset,
            )
        )

    def _text_header_and_rows(self, limit=None):
        headers = ("id", "timestamp", "record")
        rows = []
        row_indexes = util.truncate_rows(self.num_rows, limit)
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                row = self[j]
                rows.append(
                    (
                        str(j),
                        str(row.timestamp),
                        util.truncate_string_end(str(row.record), length=60),
                    )
                )
        return headers, rows

    def packset_record(self, records):
        """
        Packs the specified list of record values and updates the
        ``record`` and ``record_offset`` columns. The length
        of the records array must be equal to the number of rows in
        the table.

        :param list(str) records: A list of string record values.
        """
        packed, offset = util.pack_strings(records)
        d = self.asdict()
        d["record"] = packed
        d["record_offset"] = offset
        self.set_columns(**d)

    def packset_timestamp(self, timestamps):
        """
        Packs the specified list of timestamp values and updates the
        ``timestamp`` and ``timestamp_offset`` columns. The length
        of the timestamps array must be equal to the number of rows in
        the table.

        :param list(str) timestamps: A list of string timestamp values.
        """
        packed, offset = util.pack_strings(timestamps)
        d = self.asdict()
        d["timestamp"] = packed
        d["timestamp_offset"] = offset
        self.set_columns(**d)


# We define segment ordering by (left, right, node) tuples
@dataclasses.dataclass(eq=True, order=True)
class IdentitySegment:
    """
    A single segment of identity spanning a genomic interval for a
    a specific ancestor node.
    """

    left: float
    """The left genomic coordinate (inclusive)."""
    right: float
    """The right genomic coordinate (exclusive)."""
    node: int
    """The ID of the most recent common ancestor node."""

    @property
    def span(self) -> float:
        """
        The length of the genomic region spanned by this identity segment.
        """
        return self.right - self.left


class IdentitySegmentList(collections.abc.Iterable, collections.abc.Sized):
    """
    A summary of identity segments for some pair of samples in a
    :class:`.IdentitySegments` result. If the ``store_segments`` argument
    has been specified to :meth:`.TreeSequence.ibd_segments`, this class
    can be treated as a sequence of :class:`.IdentitySegment` objects.

    Access to the segment data via numpy arrays is also available via
    the :attr:`.IdentitySegmentList.left`, :attr:`.IdentitySegmentList.right`
    and :attr:`.IdentitySegmentList.node` attributes.

    If ``store_segments`` is False, only the overall summary values
    such as :attr:`.IdentitySegmentList.total_span` and ``len()`` are
    available.

    .. warning:: The order of segments within an IdentitySegmentList is
        arbitrary and may change in the future

    """

    def __init__(self, ll_segment_list):
        self._ll_segment_list = ll_segment_list

    def __iter__(self):
        for left, right, node in zip(self.left, self.right, self.node):
            yield IdentitySegment(float(left), float(right), int(node))

    def __len__(self):
        return self._ll_segment_list.num_segments

    def __str__(self):
        return (
            f"IdentitySegmentList(num_segments={len(self)}, "
            f"total_span={self.total_span})"
        )

    def __repr__(self):
        return f"IdentitySegmentList({repr(list(self))})"

    def __eq__(self, other):
        if not isinstance(other, IdentitySegmentList):
            return False
        return list(self) == list(other)

    @property
    def total_span(self):
        """
        The total genomic span covered by segments in this list. Equal to
        ``sum(seg.span for seg in seglst)``.
        """
        return self._ll_segment_list.total_span

    @property
    def left(self):
        """
        A numpy array (dtype=np.float64) of the ``left`` coordinates of segments.
        """
        return self._ll_segment_list.left

    @property
    def right(self):
        """
        A numpy array (dtype=np.float64) of the ``right`` coordinates of segments.
        """
        return self._ll_segment_list.right

    @property
    def node(self):
        """
        A numpy array (dtype=np.int32) of the MRCA node IDs in segments.
        """
        return self._ll_segment_list.node


class IdentitySegments(collections.abc.Mapping):
    """
    A class summarising and optionally storing the segments of identity
    by state returned by :meth:`.TreeSequence.ibd_segments`. See the
    :ref:`sec_identity` for more information and examples.

    Along with the documented methods and attributes, the class supports
    the Python mapping protocol, and can be regarded as a dictionary
    mapping sample node pair tuples to the corresponding
    :class:`.IdentitySegmentList`.

    .. note:: It is important to note that the facilities available
       for a given instance of this class are determined by the
       ``store_pairs`` and ``store_segments`` arguments provided to
       :meth:`.TreeSequence.ibd_segments`. For example, attempting
       to access per-sample pair information if ``store_pairs``
       is False will result in a (hopefully informative) error being
       raised.

    .. warning:: This class should not be instantiated directly.
    """

    def __init__(self, ll_result, *, max_time, min_span, store_segments, store_pairs):
        self._ll_identity_segments = ll_result
        self.max_time = max_time
        self.min_span = min_span
        self.store_segments = store_segments
        self.store_pairs = store_pairs

    @property
    def num_segments(self):
        """
        The total number of identity segments found.
        """
        return self._ll_identity_segments.num_segments

    @property
    def num_pairs(self):
        """
        The total number of distinct sample pairs for which identity
        segments were found. (Only available when ``store_pairs`` or
        ``store_segments`` is specified).
        """
        return self._ll_identity_segments.num_pairs

    @property
    def total_span(self):
        """
        The total genomic sequence length spanned by all identity
        segments that were found.
        """
        return self._ll_identity_segments.total_span

    @property
    def pairs(self):
        """
        A numpy array with shape ``(segs.num_pairs, 2)`` and dtype=np.int32
        containing the sample pairs for which IBD segments were found.
        """
        return self._ll_identity_segments.get_keys()

    # We have two different versions of repr - one where we list out the segments
    # for debugging, and the other that just shows the standard representation.
    # We could have repr fail if store_segments isn't true, but then printing,
    # e.g., a list of IdentitySegments objects would fail unexpectedly.
    def __repr__(self):
        if self.store_segments:
            return f"IdentitySegments({dict(self)})"
        return super().__repr__()

    def __str__(self):
        # TODO it would be nice to add horizontal lines as
        # table separators to distinguish the two parts of the
        # table like suggested here:
        # https://github.com/tskit-dev/tskit/pull/1902#issuecomment-989943424
        rows = [
            ["Parameters:", ""],
            ["max_time", str(self.max_time)],
            ["min_span", str(self.min_span)],
            ["store_pairs", str(self.store_pairs)],
            ["store_segments", str(self.store_segments)],
            ["Results:", ""],
            ["num_segments", str(self.num_segments)],
            ["total_span", str(self.total_span)],
        ]
        if self.store_pairs:
            rows.append(["num_pairs", str(len(self))])
        return util.unicode_table(rows, title="IdentitySegments", row_separator=False)

    def __getitem__(self, key):
        sample_a, sample_b = key
        return IdentitySegmentList(self._ll_identity_segments.get(sample_a, sample_b))

    def __iter__(self):
        return map(tuple, self._ll_identity_segments.get_keys())

    def __len__(self):
        return self.num_pairs


# TODO move to reference_sequence.py when we start adding more functionality.
class ReferenceSequence(metadata.MetadataProvider):
    """
    The :ref:`reference sequence<sec_data_model_reference_sequence>` associated
    with a given :class:`.TableCollection` or :class:`.TreeSequence`.

    Metadata concerning reference sequences can be described using the
    :attr:`.metadata_schema` and stored in the :attr:`.metadata` attribute.
    See the :ref:`examples<sec_metadata_examples_reference_sequence>` for
    idiomatic usage.

    .. warning:: This API is preliminary and currently only supports accessing
       reference sequence information via the ``.data`` attribute. Future versions
       will also enable transparent fetching of known reference sequences
       from a URL (see https://github.com/tskit-dev/tskit/issues/2022).
    """

    def __init__(self, ll_reference_sequence):
        super().__init__(ll_reference_sequence)
        self._ll_reference_sequence = ll_reference_sequence

    def is_null(self) -> bool:
        """
        Returns True if this :class:`.ReferenceSequence` is null, i.e.,
        all fields are empty.
        """
        return bool(self._ll_reference_sequence.is_null())

    def clear(self):
        self.data = ""
        self.url = ""
        self.metadata_schema = tskit.MetadataSchema(None)
        self.metadata = b""

    # https://github.com/tskit-dev/tskit/issues/1984
    # TODO add a __str__ method
    # TODO add a _repr_html_
    # FIXME This is a shortcut, we want to put the values in explicitly
    # here to get more control over how they are displayed.
    def __repr__(self):
        return f"ReferenceSequence({repr(self.asdict())})"

    @property
    def data(self) -> str:
        """
        The string encoding of the reference sequence such that ``data[j]``
        represents the reference nucleotide at base ``j``. If this reference
        sequence is writable, the value can be assigned, e.g.
        ``tables.reference_sequence.data = "ACGT"``
        """
        return self._ll_reference_sequence.data

    @data.setter
    def data(self, value):
        self._ll_reference_sequence.data = value

    @property
    def url(self) -> str:
        return self._ll_reference_sequence.url

    @url.setter
    def url(self, value):
        self._ll_reference_sequence.url = value

    def asdict(self) -> dict:
        return {
            "metadata_schema": repr(self.metadata_schema),
            "metadata": self.metadata_bytes,
            "data": self.data,
            "url": self.url,
        }

    def __eq__(self, other):
        return self.equals(other)

    def equals(self, other, ignore_metadata=False):
        try:
            self.assert_equals(other, ignore_metadata)
            return True
        except AssertionError:
            return False

    def assert_equals(self, other, ignore_metadata=False):
        if not ignore_metadata:
            super().assert_equals(other)

        if self.data != other.data:
            raise AssertionError(
                f"Reference sequence data differs: self={self.data} "
                f"other={other.data}"
            )
        if self.url != other.url:
            raise AssertionError(
                f"Reference sequence url differs: self={self.url} " f"other={other.url}"
            )

    @property
    def nbytes(self):
        # TODO this will be inefficient when we work with large references.
        # Make a dedicated low-level method for getting the length of data.
        return super().nbytes + len(self.url) + len(self.data)


class TableCollection(metadata.MetadataProvider):
    """
    A collection of mutable tables defining a tree sequence. See the
    :ref:`sec_data_model` section for definition on the various tables
    and how they together define a :class:`TreeSequence`. Arbitrary
    data can be stored in a TableCollection, but there are certain
    :ref:`requirements <sec_valid_tree_sequence_requirements>` that must be
    satisfied for these tables to be interpreted as a tree sequence.

    To obtain an immutable :class:`TreeSequence` instance corresponding to the
    current state of a ``TableCollection``, please use the :meth:`.tree_sequence`
    method.
    """

    set_err_text = (
        "Cannot set tables in a table collection: use table.replace_with() instead."
    )

    def __init__(self, sequence_length=0, *, ll_tables=None):
        self._ll_tables = ll_tables
        if ll_tables is None:
            self._ll_tables = _tskit.TableCollection(sequence_length)
        super().__init__(self._ll_tables)
        self._individuals = IndividualTable(ll_table=self._ll_tables.individuals)
        self._nodes = NodeTable(ll_table=self._ll_tables.nodes)
        self._edges = EdgeTable(ll_table=self._ll_tables.edges)
        self._migrations = MigrationTable(ll_table=self._ll_tables.migrations)
        self._sites = SiteTable(ll_table=self._ll_tables.sites)
        self._mutations = MutationTable(ll_table=self._ll_tables.mutations)
        self._populations = PopulationTable(ll_table=self._ll_tables.populations)
        self._provenances = ProvenanceTable(ll_table=self._ll_tables.provenances)

    @property
    def individuals(self) -> IndividualTable:
        """
        The :ref:`sec_individual_table_definition` in this collection.
        """
        return self._individuals

    @individuals.setter
    def individuals(self, value):
        raise AttributeError(self.set_err_text)

    @property
    def nodes(self) -> NodeTable:
        """
        The :ref:`sec_node_table_definition` in this collection.
        """
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        raise AttributeError(self.set_err_text)

    @property
    def edges(self) -> EdgeTable:
        """
        The :ref:`sec_edge_table_definition` in this collection.
        """
        return self._edges

    @edges.setter
    def edges(self, value):
        raise AttributeError(self.set_err_text)

    @property
    def migrations(self) -> MigrationTable:
        """
        The :ref:`sec_migration_table_definition` in this collection
        """
        return self._migrations

    @migrations.setter
    def migrations(self, value):
        raise AttributeError(self.set_err_text)

    @property
    def sites(self) -> SiteTable:
        """
        The :ref:`sec_site_table_definition` in this collection.
        """
        return self._sites

    @sites.setter
    def sites(self, value):
        raise AttributeError(self.set_err_text)

    @property
    def mutations(self) -> MutationTable:
        """
        The :ref:`sec_mutation_table_definition` in this collection.
        """
        return self._mutations

    @mutations.setter
    def mutations(self, value):
        raise AttributeError(self.set_err_text)

    @property
    def populations(self) -> PopulationTable:
        """
        The :ref:`sec_population_table_definition` in this collection.
        """
        return self._populations

    @populations.setter
    def populations(self, value):
        raise AttributeError(self.set_err_text)

    @property
    def provenances(self) -> ProvenanceTable:
        """
        The :ref:`sec_provenance_table_definition` in this collection.
        """
        return self._provenances

    @provenances.setter
    def provenances(self, value):
        raise AttributeError(self.set_err_text)

    @property
    def indexes(self) -> TableCollectionIndexes:
        """
        The edge insertion and removal indexes.
        """
        indexes = self._ll_tables.indexes
        return TableCollectionIndexes(**indexes)

    @indexes.setter
    def indexes(self, indexes):
        self._ll_tables.indexes = indexes.asdict()

    @property
    def sequence_length(self) -> float:
        """
        The sequence length defining the coordinate space.
        """
        return self._ll_tables.sequence_length

    @sequence_length.setter
    def sequence_length(self, sequence_length):
        self._ll_tables.sequence_length = sequence_length

    @property
    def file_uuid(self) -> str:
        """
        The UUID for the file this TableCollection is derived
        from, or None if not derived from a file.
        """
        return self._ll_tables.file_uuid

    @property
    def time_units(self) -> str:
        """
        The units used for the time dimension of this TableCollection
        """
        return self._ll_tables.time_units

    @time_units.setter
    def time_units(self, time_units: str) -> None:
        self._ll_tables.time_units = time_units

    def has_reference_sequence(self):
        """
        Returns True if this :class:`.TableCollection` has an associated
        :ref:`reference sequence<sec_data_model_reference_sequence>`.
        """
        return bool(self._ll_tables.has_reference_sequence())

    @property
    def reference_sequence(self):
        """
        The :class:`.ReferenceSequence` associated with this :class:`.TableCollection`.

        .. note:: Note that the behaviour of this attribute differs from
            :attr:`.TreeSequence.reference_sequence` in that we return a valid
            instance of :class:`.ReferenceSequence` even when
            :attr:`.TableCollection.has_reference_sequence` is False. This is
            to allow us to update the state of the reference sequence.
        """
        # NOTE: arguably we should cache the reference to this object
        # during init, rather than creating a new instance each time.
        # However, following the pattern of the Table classes for now
        # for consistency.
        return ReferenceSequence(self._ll_tables.reference_sequence)

    @reference_sequence.setter
    def reference_sequence(self, value: ReferenceSequence):
        self.reference_sequence.metadata_schema = value.metadata_schema
        self.reference_sequence.metadata = value.metadata
        self.reference_sequence.data = value.data
        self.reference_sequence.url = value.url

    def asdict(self, force_offset_64=False):
        """
        Returns the nested dictionary representation of this TableCollection
        used for interchange.

        Note: the semantics of this method changed at tskit 0.1.0. Previously a
        map of table names to the tables themselves was returned.

        :param bool force_offset_64: If True, all offset columns will have dtype
            np.uint64. If False (the default) the offset array columns will have
            a dtype of either np.uint32 or np.uint64, depending on the size of the
            corresponding data array.
        :return: The dictionary representation of this table collection.
        :rtype: dict
        """
        return self._ll_tables.asdict(force_offset_64)

    @property
    def table_name_map(self) -> Dict:
        """
        Returns a dictionary mapping table names to the corresponding
        table instances. For example, the returned dictionary will contain the
        key "edges" that maps to an :class:`.EdgeTable` instance.
        """
        return {
            "edges": self.edges,
            "individuals": self.individuals,
            "migrations": self.migrations,
            "mutations": self.mutations,
            "nodes": self.nodes,
            "populations": self.populations,
            "provenances": self.provenances,
            "sites": self.sites,
        }

    @property
    def name_map(self) -> Dict:
        # Deprecated in 0.4.1
        warnings.warn(
            "name_map is deprecated; use table_name_map instead",
            FutureWarning,
            stacklevel=4,
        )
        return self.table_name_map

    @property
    def nbytes(self) -> int:
        """
        Returns the total number of bytes required to store the data
        in this table collection. Note that this may not be equal to
        the actual memory footprint.
        """
        return sum(
            (
                8,  # sequence_length takes 8 bytes
                super().nbytes,  # metadata
                len(self.time_units.encode()),
                self.indexes.nbytes,
                self.reference_sequence.nbytes,
                sum(table.nbytes for table in self.table_name_map.values()),
            )
        )

    def __str__(self):
        """
        Return a plain text summary of this TableCollection
        """
        return "\n".join(
            [
                "TableCollection",
                "",
                f"Sequence Length: {self.sequence_length}",
                f"Time units: {self.time_units}",
                f"Metadata: {self.metadata}",
                "",
                "Individuals",
                str(self.individuals),
                "Nodes",
                str(self.nodes),
                "Edges",
                str(self.edges),
                "Sites",
                str(self.sites),
                "Mutations",
                str(self.mutations),
                "Migrations",
                str(self.migrations),
                "Populations",
                str(self.populations),
                "Provenances",
                str(self.provenances),
            ]
        )

    def equals(
        self,
        other,
        *,
        ignore_metadata=False,
        ignore_ts_metadata=False,
        ignore_provenance=False,
        ignore_timestamps=False,
        ignore_tables=False,
        ignore_reference_sequence=False,
    ):
        """
        Returns True if  `self` and `other` are equal. By default, two table
        collections are considered equal if their

        - ``sequence_length`` properties are identical;
        - top-level tree sequence metadata and metadata schemas are
          byte-wise identical;
        - constituent tables are byte-wise identical.

        Some of the requirements in this definition can be relaxed using the
        parameters, which can be used to remove certain parts of the data model
        from the comparison.

        Table indexes are not considered in the equality comparison.

        :param TableCollection other: Another table collection.
        :param bool ignore_metadata: If True *all* metadata and metadata schemas
            will be excluded from the comparison. This includes the top-level
            tree sequence and constituent table metadata (default=False).
        :param bool ignore_ts_metadata: If True the top-level tree sequence
            metadata and metadata schemas will be excluded from the comparison.
            If ``ignore_metadata`` is True, this parameter has no effect.
        :param bool ignore_provenance: If True the provenance tables are
            not included in the comparison.
        :param bool ignore_timestamps: If True the provenance timestamp column
            is ignored in the comparison. If ``ignore_provenance`` is True, this
            parameter has no effect.
        :param bool ignore_tables: If True no tables are included in the
            comparison, thus comparing only the top-level information.
        :param bool ignore_reference_sequence: If True the reference sequence
            is not included in the comparison.
        :return: True if other is equal to this table collection; False otherwise.
        :rtype: bool
        """
        ret = False
        if type(other) is type(self):
            ret = bool(
                self._ll_tables.equals(
                    other._ll_tables,
                    ignore_metadata=bool(ignore_metadata),
                    ignore_ts_metadata=bool(ignore_ts_metadata),
                    ignore_provenance=bool(ignore_provenance),
                    ignore_timestamps=bool(ignore_timestamps),
                    ignore_tables=bool(ignore_tables),
                    ignore_reference_sequence=bool(ignore_reference_sequence),
                )
            )
        return ret

    def assert_equals(
        self,
        other,
        *,
        ignore_metadata=False,
        ignore_ts_metadata=False,
        ignore_provenance=False,
        ignore_timestamps=False,
        ignore_tables=False,
        ignore_reference_sequence=False,
    ):
        """
        Raise an AssertionError for the first found difference between
        this and another table collection. Note that table indexes are not checked.

        :param TableCollection other: Another table collection.
        :param bool ignore_metadata: If True *all* metadata and metadata schemas
            will be excluded from the comparison. This includes the top-level
            tree sequence and constituent table metadata (default=False).
        :param bool ignore_ts_metadata: If True the top-level tree sequence
            metadata and metadata schemas will be excluded from the comparison.
            If ``ignore_metadata`` is True, this parameter has no effect.
        :param bool ignore_provenance: If True the provenance tables are
            not included in the comparison.
        :param bool ignore_timestamps: If True the provenance timestamp column
            is ignored in the comparison. If ``ignore_provenance`` is True, this
            parameter has no effect.
        :param bool ignore_tables: If True no tables are included in the
            comparison, thus comparing only the top-level information.
        :param bool ignore_reference_sequence: If True the reference sequence
            is not included in the comparison.
        """
        if type(other) is not type(self):
            raise AssertionError(f"Types differ: self={type(self)} other={type(other)}")

        # Check using the low-level method to avoid slowly going through everything
        if self.equals(
            other,
            ignore_metadata=ignore_metadata,
            ignore_ts_metadata=ignore_ts_metadata,
            ignore_provenance=ignore_provenance,
            ignore_timestamps=ignore_timestamps,
            ignore_tables=ignore_tables,
            ignore_reference_sequence=ignore_reference_sequence,
        ):
            return

        if not ignore_metadata or ignore_ts_metadata:
            super().assert_equals(other)

        if not ignore_reference_sequence:
            self.reference_sequence.assert_equals(
                other.reference_sequence, ignore_metadata=ignore_metadata
            )

        if self.time_units != other.time_units:
            raise AssertionError(
                f"Time units differs: self={self.time_units} "
                f"other={other.time_units}"
            )

        if self.sequence_length != other.sequence_length:
            raise AssertionError(
                f"Sequence Length"
                f" differs: self={self.sequence_length} other={other.sequence_length}"
            )

        for table_name, table in self.table_name_map.items():
            if table_name != "provenances":
                table.assert_equals(
                    getattr(other, table_name), ignore_metadata=ignore_metadata
                )

        if not ignore_provenance:
            self.provenances.assert_equals(
                other.provenances, ignore_timestamps=ignore_timestamps
            )

        raise AssertionError(
            "TableCollections differ in an undetected way - "
            "this is a bug, please report an issue on gitub"
        )  # pragma: no cover

    def __eq__(self, other):
        return self.equals(other)

    def __getstate__(self):
        return self.asdict()

    @classmethod
    def load(cls, file_or_path, *, skip_tables=False, skip_reference_sequence=False):
        file, local_file = util.convert_file_like_to_open_file(file_or_path, "rb")
        ll_tc = _tskit.TableCollection()
        try:
            ll_tc.load(
                file,
                skip_tables=skip_tables,
                skip_reference_sequence=skip_reference_sequence,
            )
            return TableCollection(ll_tables=ll_tc)
        except tskit.FileFormatError as e:
            util.raise_known_file_format_errors(file, e)
        finally:
            if local_file:
                file.close()

    def dump(self, file_or_path):
        """
        Writes the table collection to the specified path or file object.

        :param str file_or_path: The file object or path to write the TreeSequence to.
        """
        file, local_file = util.convert_file_like_to_open_file(file_or_path, "wb")
        try:
            self._ll_tables.dump(file)
        finally:
            if local_file:
                file.close()

    # Unpickle support
    def __setstate__(self, state):
        self.__init__()
        self._ll_tables.fromdict(state)

    @classmethod
    def fromdict(self, tables_dict):
        ll_tc = _tskit.TableCollection()
        ll_tc.fromdict(tables_dict)
        return TableCollection(ll_tables=ll_tc)

    def copy(self):
        """
        Returns a deep copy of this TableCollection.

        :return: A deep copy of this TableCollection.
        :rtype: tskit.TableCollection
        """
        return TableCollection.fromdict(self.asdict())

    def tree_sequence(self):
        """
        Returns a :class:`TreeSequence` instance from the tables defined in this
        :class:`TableCollection`, building the required indexes if they have not yet
        been created by :meth:`.build_index`. If the table collection does not meet
        the :ref:`sec_valid_tree_sequence_requirements`, for example if the tables
        are not correctly sorted or if they cannot be interpreted as a tree sequence,
        an exception is raised. Note that in the former case, the :meth:`.sort`
        method may be used to ensure that sorting requirements are met.

        :return: A :class:`TreeSequence` instance reflecting the structures
            defined in this set of tables.
        :rtype: tskit.TreeSequence
        """
        if not self.has_index():
            self.build_index()
        return tskit.TreeSequence.load_tables(self)

    def simplify(
        self,
        samples=None,
        *,
        reduce_to_site_topology=False,
        filter_populations=None,
        filter_individuals=None,
        filter_sites=None,
        filter_nodes=None,
        update_sample_flags=None,
        keep_unary=False,
        keep_unary_in_individuals=None,
        keep_input_roots=False,
        record_provenance=True,
        filter_zero_mutation_sites=None,  # Deprecated alias for filter_sites
    ):
        """
        Simplifies the tables in place to retain only the information necessary
        to reconstruct the tree sequence describing the given ``samples``.
        If ``filter_nodes`` is True (the default), this can change the ID of
        the nodes, so that the node ``samples[k]`` will have ID ``k`` in the
        result, resulting in a NodeTable where only the first ``len(samples)``
        nodes are marked as samples. The mapping from node IDs in the current
        set of tables to their equivalent values in the simplified tables is
        returned as a numpy array. If an array ``a`` is returned by this
        function and ``u`` is the ID of a node in the input table, then
        ``a[u]`` is the ID of this node in the output table. For any node ``u``
        that is not mapped into the output tables, this mapping will equal
        ``tskit.NULL`` (``-1``).

        Tables operated on by this function must: be sorted (see
        :meth:`TableCollection.sort`), have children be born strictly after their
        parents, and the intervals on which any node is a child must be
        disjoint. Other than this the tables need not satisfy remaining
        requirements to specify a valid tree sequence (but the resulting tables
        will).

        .. note::
            To invert the returned ``node_map``, that is, to obtain a reverse
            mapping from the node ID in the output table to the node ID in
            the input table, you can use::

                rev_map = np.zeros_like(node_map, shape=simplified_ts.num_nodes)
                kept = node_map != tskit.NULL
                rev_map[node_map[kept]] = np.arange(len(node_map))[kept]

            In this case, no elements of the ``rev_map`` array will be set to
            ``tskit.NULL``.

        .. seealso::
            This is identical to :meth:`TreeSequence.simplify` but acts *in place* to
            alter the data in this :class:`TableCollection`. Please see the
            :meth:`TreeSequence.simplify` method for a description of the remaining
            parameters.

        :param list[int] samples: A list of node IDs to retain as samples. They
            need not be nodes marked as samples in the original tree sequence, but
            will constitute the entire set of samples in the returned tree sequence.
            If not specified or None, use all nodes marked with the IS_SAMPLE flag.
            The list may be provided as a numpy array (or array-like) object
            (dtype=np.int32).
        :param bool reduce_to_site_topology: Whether to reduce the topology down
            to the trees that are present at sites. (Default: False).
        :param bool filter_populations: If True, remove any populations that are
            not referenced by nodes after simplification; new population IDs are
            allocated sequentially from zero. If False, the population table will
            not be altered in any way. (Default: None, treated as True)
        :param bool filter_individuals: If True, remove any individuals that are
            not referenced by nodes after simplification; new individual IDs are
            allocated sequentially from zero. If False, the individual table will
            not be altered in any way. (Default: None, treated as True)
        :param bool filter_sites: If True, remove any sites that are
            not referenced by mutations after simplification; new site IDs are
            allocated sequentially from zero. If False, the site table will not
            be altered in any way. (Default: None, treated as True)
        :param bool filter_nodes: If True, remove any nodes that are
            not referenced by edges after simplification. If False, the only
            potential change to the node table may be to change the node flags
            (if ``samples`` is specified and different from the existing samples).
            (Default: None, treated as True)
        :param bool update_sample_flags: If True, update node flags to so that
            nodes in the specified list of samples have the NODE_IS_SAMPLE
            flag after simplification, and nodes that are not in this list
            do not. (Default: None, treated as True)
        :param bool keep_unary: If True, preserve unary nodes (i.e. nodes with
            exactly one child) that exist on the path from samples to root.
            (Default: False)
        :param bool keep_unary_in_individuals: If True, preserve unary nodes
            that exist on the path from samples to root, but only if they are
            associated with an individual in the individuals table. Cannot be
            specified at the same time as ``keep_unary``. (Default: ``None``,
            equivalent to False)
        :param bool keep_input_roots: Whether to retain history ancestral to the
            MRCA of the samples. If ``False``, no topology older than the MRCAs of the
            samples will be included. If ``True`` the roots of all trees in the returned
            tree sequence will be the same roots as in the original tree sequence.
            (Default: False)
        :param bool record_provenance: If True, record details of this call to
            simplify in the returned tree sequence's provenance information
            (Default: True).
        :param bool filter_zero_mutation_sites: Deprecated alias for ``filter_sites``.
        :return: A numpy array mapping node IDs in the input tables to their
            corresponding node IDs in the output tables.
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        if filter_zero_mutation_sites is not None:
            # Deprecated in msprime 0.6.1.
            warnings.warn(
                "filter_zero_mutation_sites is deprecated; use filter_sites instead",
                FutureWarning,
                stacklevel=4,
            )
            filter_sites = filter_zero_mutation_sites
        if samples is None:
            flags = self.nodes.flags
            samples = np.where(np.bitwise_and(flags, _tskit.NODE_IS_SAMPLE) != 0)[
                0
            ].astype(np.int32)
        else:
            samples = util.safe_np_int_cast(samples, np.int32)
        if filter_populations is None:
            filter_populations = True
        if filter_individuals is None:
            filter_individuals = True
        if filter_sites is None:
            filter_sites = True
        if filter_nodes is None:
            filter_nodes = True
        if update_sample_flags is None:
            update_sample_flags = True
        if keep_unary_in_individuals is None:
            keep_unary_in_individuals = False

        node_map = self._ll_tables.simplify(
            samples,
            filter_sites=filter_sites,
            filter_individuals=filter_individuals,
            filter_populations=filter_populations,
            filter_nodes=filter_nodes,
            update_sample_flags=update_sample_flags,
            reduce_to_site_topology=reduce_to_site_topology,
            keep_unary=keep_unary,
            keep_unary_in_individuals=keep_unary_in_individuals,
            keep_input_roots=keep_input_roots,
        )
        if record_provenance:
            # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
            # TODO also make sure we convert all the arguments so that they are
            # definitely JSON encodable.
            parameters = {"command": "simplify", "TODO": "add simplify parameters"}
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )
        return node_map

    def link_ancestors(self, samples, ancestors):
        """
        Returns an :class:`EdgeTable` instance describing a subset of the genealogical
        relationships between the nodes in ``samples`` and ``ancestors``.

        Each row ``parent, child, left, right`` in the output table indicates that
        ``child`` has inherited the segment ``[left, right)`` from ``parent`` more
        recently than from any other node in these lists.

        In particular, suppose ``samples`` is a list of nodes such that ``time`` is 0
        for each node, and ``ancestors`` is a list of nodes such that ``time`` is
        greater than 0.0 for each node. Then each row of the output table will show
        an interval ``[left, right)`` over which a node in ``samples`` has inherited
        most recently from a node in ``ancestors``, or an interval over which one of
        these ``ancestors`` has inherited most recently from another node in
        ``ancestors``.

        The following table shows which ``parent->child`` pairs will be shown in the
        output of ``link_ancestors``.
        A node is a relevant descendant on a given interval if it also appears somewhere
        in the ``parent`` column of the outputted table.

        ========================  ===============================================
        Type of relationship      Shown in output of ``link_ancestors``
        ------------------------  -----------------------------------------------
        ``ancestor->sample``      Always
        ``ancestor1->ancestor2``  Only if ``ancestor2`` has a relevant descendant
        ``sample1->sample2``      Always
        ``sample->ancestor``      Only if ``ancestor`` has a relevant descendant
        ========================  ===============================================

        The difference between ``samples`` and ``ancestors`` is that information about
        the ancestors of a node in ``ancestors`` will only be retained if it also has a
        relevant descendant, while information about the ancestors of a node in
        ``samples`` will always be retained.
        The node IDs in ``parent`` and ``child`` refer to the IDs in the node table
        of the inputted tree sequence.

        The supplied nodes must be non-empty lists of the node IDs in the tree sequence:
        in particular, they do not have to be *samples* of the tree sequence. The lists
        of ``samples`` and ``ancestors`` may overlap, although adding a node from
        ``samples`` to ``ancestors`` will not change the output. So, setting ``samples``
        and ``ancestors`` to the same list of nodes will find all genealogical
        relationships within this list.

        If none of the nodes in ``ancestors`` or ``samples`` are ancestral to ``samples``
        anywhere in the tree sequence, an empty table will be returned.

        :param list[int] samples: A list of node IDs to retain as samples.
        :param list[int] ancestors: A list of node IDs to use as ancestors.
        :return: An :class:`EdgeTable` instance displaying relationships between
            the `samples` and `ancestors`.
        """
        samples = util.safe_np_int_cast(samples, np.int32)
        ancestors = util.safe_np_int_cast(ancestors, np.int32)
        ll_edge_table = self._ll_tables.link_ancestors(samples, ancestors)
        return EdgeTable(ll_table=ll_edge_table)

    def map_ancestors(self, *args, **kwargs):
        # A deprecated alias for link_ancestors()
        return self.link_ancestors(*args, **kwargs)

    def sort(self, edge_start=0, *, site_start=0, mutation_start=0):
        """
        Sorts the tables in place. This ensures that all tree sequence ordering
        requirements listed in the
        :ref:`sec_valid_tree_sequence_requirements` section are met, as long
        as each site has at most one mutation (see below).

        If the ``edge_start`` parameter is provided, this specifies the index
        in the edge table where sorting should start. Only rows with index
        greater than or equal to ``edge_start`` are sorted; rows before this index
        are not affected. This parameter is provided to allow for efficient sorting
        when the user knows that the edges up to a given index are already sorted.

        If both ``site_start`` and ``mutation_start`` are equal to the number of rows
        in their retrospective tables then neither is sorted. Note that a partial
        non-sorting is not possible, and both or neither must be skipped.

        The node, individual, population and provenance tables are not affected
        by this method.

        Edges are sorted as follows:

        - time of parent, then
        - parent node ID, then
        - child node ID, then
        - left endpoint.

        Note that this sorting order exceeds the
        :ref:`edge sorting requirements <sec_edge_requirements>` for a valid
        tree sequence. For a valid tree sequence, we require that all edges for a
        given parent ID are adjacent, but we do not require that they be listed in
        sorted order.

        Sites are sorted by position, and sites with the same position retain
        their relative ordering.

        Mutations are sorted by site ID, and within the same site are sorted by time.
        Those with equal or unknown time retain their relative ordering. This does not
        currently rearrange tables so that mutations occur after their mutation parents,
        which is a requirement for valid tree sequences.

        Migrations are sorted by ``time``, ``source``, ``dest``, ``left`` and
        ``node`` values. This defines a total sort order, such that any permutation
        of a valid migration table will be sorted into the same output order.
        Note that this sorting order exceeds the
        :ref:`migration sorting requirements <sec_migration_requirements>` for a
        valid tree sequence, which only requires that migrations are sorted by
        time value.

        :param int edge_start: The index in the edge table where sorting starts
            (default=0; must be <= len(edges)).
        :param int site_start: The index in the site table where sorting starts
            (default=0; must be one of [0, len(sites)]).
        :param int mutation_start: The index in the mutation table where sorting starts
            (default=0; must be one of [0, len(mutations)]).
        """
        self._ll_tables.sort(edge_start, site_start, mutation_start)
        # TODO add provenance

    def sort_individuals(self):
        """
        Sorts the individual table in place, so that parents come before children,
        and the parent column is remapped as required. Node references to individuals
        are also updated.
        """
        self._ll_tables.sort_individuals()
        # TODO add provenance

    def canonicalise(self, remove_unreferenced=None):
        """
        This puts the tables in *canonical* form, imposing a stricter order on the
        tables than :ref:`required <sec_valid_tree_sequence_requirements>` for
        a valid tree sequence. In particular, the individual
        and population tables are sorted by the first node that refers to each
        (see :meth:`TreeSequence.subset`). Then, the remaining tables are sorted
        as in :meth:`.sort`, with the modification that mutations are sorted by
        site, then time, then number of descendant mutations (ensuring that
        parent mutations occur before children), then node, then original order
        in the tables. This ensures that any two tables with the same information
        and node order should be identical after canonical sorting (note
        that no canonical order exists for the node table).

        By default, the method removes sites, individuals, and populations that
        are not referenced (by mutations and nodes, respectively). If you wish
        to keep these, pass ``remove_unreferenced=False``, but note that
        unreferenced individuals and populations are put at the end of the tables
        in their original order.

        .. seealso::

            :meth:`.sort` for sorting edges, mutations, and sites, and
            :meth:`.subset` for reordering nodes, individuals, and populations.

        :param bool remove_unreferenced: Whether to remove unreferenced sites,
            individuals, and populations (default=True).
        """
        remove_unreferenced = (
            True if remove_unreferenced is None else remove_unreferenced
        )
        self._ll_tables.canonicalise(remove_unreferenced=remove_unreferenced)
        # TODO add provenance

    def compute_mutation_parents(self):
        """
        Modifies the tables in place, computing the ``parent`` column of the
        mutation table. For this to work, the node and edge tables must be
        valid, and the site and mutation tables must be sorted (see
        :meth:`TableCollection.sort`).  This will produce an error if mutations
        are not sorted (i.e., if a mutation appears before its mutation parent)
        *unless* the two mutations occur on the same branch, in which case
        there is no way to detect the error.

        The ``parent`` of a given mutation is the ID of the next mutation
        encountered traversing the tree upwards from that mutation, or
        ``NULL`` if there is no such mutation.
        """
        self._ll_tables.compute_mutation_parents()
        # TODO add provenance

    def compute_mutation_times(self):
        """
        Modifies the tables in place, computing valid values for the ``time`` column of
        the mutation table. For this to work, the node and edge tables must be
        valid, and the site and mutation tables must be sorted and indexed(see
        :meth:`TableCollection.sort` and :meth:`TableCollection.build_index`).

        For a single mutation on an edge at a site, the ``time`` assigned to a mutation
        by this method is the mid-point between the times of the nodes above and below
        the mutation. In the case where there is more than one mutation on an edge for
        a site, the times are evenly spread along the edge. For mutations that are
        above a root node, the time of the root node is assigned.

        The mutation table will be sorted if the new times mean that the original order
        is no longer valid.

        """
        self._ll_tables.compute_mutation_times()
        # TODO add provenance

    def deduplicate_sites(self):
        """
        Modifies the tables in place, removing entries in the site table with
        duplicate ``position`` (and keeping only the *first* entry for each
        site), and renumbering the ``site`` column of the mutation table
        appropriately.  This requires the site table to be sorted by position.

        .. warning:: This method does not sort the tables afterwards, so
            mutations may no longer be sorted by time.
        """
        self._ll_tables.deduplicate_sites()
        # TODO add provenance

    def delete_sites(self, site_ids, record_provenance=True):
        """
        Remove the specified sites entirely from the sites and mutations tables in this
        collection. This is identical to :meth:`TreeSequence.delete_sites` but acts
        *in place* to alter the data in this :class:`TableCollection`.

        :param list[int] site_ids: A list of site IDs specifying the sites to remove.
        :param bool record_provenance: If ``True``, add details of this operation
            to the provenance table in this TableCollection. (Default: ``True``).
        """
        keep_sites = np.ones(len(self.sites), dtype=bool)
        site_ids = util.safe_np_int_cast(site_ids, np.int32)
        if np.any(site_ids < 0) or np.any(site_ids >= len(self.sites)):
            raise ValueError("Site ID out of bounds")
        keep_sites[site_ids] = 0
        new_as, new_as_offset = keep_with_offset(
            keep_sites, self.sites.ancestral_state, self.sites.ancestral_state_offset
        )
        new_md, new_md_offset = keep_with_offset(
            keep_sites, self.sites.metadata, self.sites.metadata_offset
        )
        self.sites.set_columns(
            position=self.sites.position[keep_sites],
            ancestral_state=new_as,
            ancestral_state_offset=new_as_offset,
            metadata=new_md,
            metadata_offset=new_md_offset,
        )
        # We also need to adjust the mutations table, as it references into sites
        keep_mutations = keep_sites[self.mutations.site]
        new_ds, new_ds_offset = keep_with_offset(
            keep_mutations,
            self.mutations.derived_state,
            self.mutations.derived_state_offset,
        )
        new_md, new_md_offset = keep_with_offset(
            keep_mutations, self.mutations.metadata, self.mutations.metadata_offset
        )
        # Site numbers will have changed
        site_map = np.cumsum(keep_sites, dtype=self.mutations.site.dtype) - 1
        # Mutation numbers will change, so the parent references need altering
        mutation_map = np.cumsum(keep_mutations, dtype=self.mutations.parent.dtype) - 1
        # Map parent == -1 to -1, and check this has worked (assumes tskit.NULL == -1)
        mutation_map = np.append(mutation_map, -1).astype(self.mutations.parent.dtype)
        assert mutation_map[tskit.NULL] == tskit.NULL
        self.mutations.set_columns(
            site=site_map[self.mutations.site[keep_mutations]],
            node=self.mutations.node[keep_mutations],
            time=self.mutations.time[keep_mutations],
            derived_state=new_ds,
            derived_state_offset=new_ds_offset,
            parent=mutation_map[self.mutations.parent[keep_mutations]],
            metadata=new_md,
            metadata_offset=new_md_offset,
        )
        if record_provenance:
            # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
            parameters = {"command": "delete_sites", "TODO": "add parameters"}
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

    def delete_intervals(self, intervals, simplify=True, record_provenance=True):
        """
        Delete all information from this set of tables which lies *within* the
        specified list of genomic intervals. This is identical to
        :meth:`TreeSequence.delete_intervals` but acts *in place* to alter
        the data in this :class:`TableCollection`.

        :param array_like intervals: A list (start, end) pairs describing the
            genomic intervals to delete. Intervals must be non-overlapping and
            in increasing order. The list of intervals must be interpretable as a
            2D numpy array with shape (N, 2), where N is the number of intervals.
        :param bool simplify: If True, run simplify on the tables so that nodes
            no longer used are discarded. (Default: True).
        :param bool record_provenance: If ``True``, add details of this operation
            to the provenance table in this TableCollection. (Default: ``True``).
        """
        self.keep_intervals(
            util.negate_intervals(intervals, 0, self.sequence_length),
            simplify=simplify,
            record_provenance=False,
        )
        if record_provenance:
            parameters = {"command": "delete_intervals", "TODO": "add parameters"}
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

    def keep_intervals(self, intervals, simplify=True, record_provenance=True):
        """
        Delete all information from this set of tables which lies *outside* the
        specified list of genomic intervals. This is identical to
        :meth:`TreeSequence.keep_intervals` but acts *in place* to alter
        the data in this :class:`TableCollection`.

        :param array_like intervals: A list (start, end) pairs describing the
            genomic intervals to keep. Intervals must be non-overlapping and
            in increasing order. The list of intervals must be interpretable as a
            2D numpy array with shape (N, 2), where N is the number of intervals.
        :param bool simplify: If True, run simplify on the tables so that nodes
            no longer used are discarded. Must be ``False`` if input tree sequence
            includes migrations. (Default: True).
        :param bool record_provenance: If ``True``, add details of this operation
            to the provenance table in this TableCollection. (Default: ``True``).
        """
        intervals = util.intervals_to_np_array(intervals, 0, self.sequence_length)

        edges = self.edges.copy()
        self.edges.clear()
        migrations = self.migrations.copy()
        self.migrations.clear()
        keep_sites = np.repeat(False, self.sites.num_rows)
        for s, e in intervals:
            curr_keep_sites = np.logical_and(
                self.sites.position >= s, self.sites.position < e
            )
            keep_sites = np.logical_or(keep_sites, curr_keep_sites)
            keep_edges = np.logical_not(
                np.logical_or(edges.right <= s, edges.left >= e)
            )
            metadata, metadata_offset = keep_with_offset(
                keep_edges, edges.metadata, edges.metadata_offset
            )
            self.edges.append_columns(
                left=np.fmax(s, edges.left[keep_edges]),
                right=np.fmin(e, edges.right[keep_edges]),
                parent=edges.parent[keep_edges],
                child=edges.child[keep_edges],
                metadata=metadata,
                metadata_offset=metadata_offset,
            )
            keep_migrations = np.logical_not(
                np.logical_or(migrations.right <= s, migrations.left >= e)
            )
            metadata, metadata_offset = keep_with_offset(
                keep_migrations, migrations.metadata, migrations.metadata_offset
            )
            self.migrations.append_columns(
                left=np.fmax(s, migrations.left[keep_migrations]),
                right=np.fmin(e, migrations.right[keep_migrations]),
                node=migrations.node[keep_migrations],
                source=migrations.source[keep_migrations],
                dest=migrations.dest[keep_migrations],
                time=migrations.time[keep_migrations],
                metadata=metadata,
                metadata_offset=metadata_offset,
            )
        self.delete_sites(
            np.where(np.logical_not(keep_sites))[0], record_provenance=False
        )

        self.sort()
        if simplify:
            self.simplify(record_provenance=False)
        if record_provenance:
            parameters = {"command": "keep_intervals", "TODO": "add parameters"}
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

    def _check_trim_conditions(self):
        if self.migrations.num_rows > 0:
            if (np.min(self.migrations.left) < np.min(self.edges.left)) and (
                np.max(self.migrations.right) > np.max(self.edges.right)
            ):
                raise ValueError(
                    "Cannot trim a tree sequence with migrations which exist to the"
                    "left of the leftmost edge or to the right of the rightmost edge."
                )
        if self.edges.num_rows == 0:
            raise ValueError(
                "Trimming a tree sequence with no edges would reduce the sequence length"
                " to zero, which is not allowed"
            )

    def ltrim(self, record_provenance=True):
        """
        Reset the coordinate system used in these tables, changing the left and right
        genomic positions in the edge table such that the leftmost edge now starts at
        position 0. This is identical to :meth:`TreeSequence.ltrim` but acts *in place*
        to alter the data in this :class:`TableCollection`.

        :param bool record_provenance: If ``True``, add details of this operation
            to the provenance table in this TableCollection. (Default: ``True``).
        """
        self._check_trim_conditions()
        leftmost = np.min(self.edges.left)
        self.delete_sites(
            np.where(self.sites.position < leftmost), record_provenance=False
        )
        self.edges.set_columns(
            left=self.edges.left - leftmost,
            right=self.edges.right - leftmost,
            parent=self.edges.parent,
            child=self.edges.child,
        )
        self.sites.set_columns(
            position=self.sites.position - leftmost,
            ancestral_state=self.sites.ancestral_state,
            ancestral_state_offset=self.sites.ancestral_state_offset,
            metadata=self.sites.metadata,
            metadata_offset=self.sites.metadata_offset,
        )
        self.migrations.set_columns(
            left=self.migrations.left - leftmost,
            right=self.migrations.right - leftmost,
            time=self.migrations.time,
            node=self.migrations.node,
            source=self.migrations.source,
            dest=self.migrations.dest,
        )
        self.sequence_length = self.sequence_length - leftmost
        if record_provenance:
            # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
            parameters = {
                "command": "ltrim",
            }
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

    def rtrim(self, record_provenance=True):
        """
        Reset the ``sequence_length`` property so that the sequence ends at the end of
        the last edge. This is identical to :meth:`TreeSequence.rtrim` but acts
        *in place* to alter the data in this :class:`TableCollection`.

        :param bool record_provenance: If ``True``, add details of this operation
            to the provenance table in this TableCollection. (Default: ``True``).
        """
        self._check_trim_conditions()
        rightmost = np.max(self.edges.right)
        self.delete_sites(
            np.where(self.sites.position >= rightmost), record_provenance=False
        )
        self.sequence_length = rightmost
        if record_provenance:
            # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
            parameters = {
                "command": "rtrim",
            }
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

    def trim(self, record_provenance=True):
        """
        Trim away any empty regions on the right and left of the tree sequence encoded by
        these tables. This is identical to :meth:`TreeSequence.trim` but acts *in place*
        to alter the data in this :class:`TableCollection`.

        :param bool record_provenance: If ``True``, add details of this operation
            to the provenance table in this TableCollection. (Default: ``True``).
        """
        self.rtrim(record_provenance=False)
        self.ltrim(record_provenance=False)
        if record_provenance:
            # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
            parameters = {
                "command": "trim",
            }
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

    def delete_older(self, time):
        """
        Deletes edge, mutation and migration information at least as old as
        the specified time.

        .. seealso:: This method is similar to the higher-level
            :meth:`TreeSequence.decapitate` method, which also splits
            edges that intersect with the given time.
            :meth:`TreeSequence.decapitate`
            is more useful for most purposes, and may be what
            you need instead of this method!

        For the purposes of this method, an edge covers the times from the
        child node up until the *parent* node, so that any any edge with parent
        node time > ``time`` will be removed.

        Any mutation whose time is >= ``time`` will be removed. A mutation's time
        is its associated ``time`` value, or the time of its node if the
        mutation's time was marked as unknown (:data:`UNKNOWN_TIME`).

        Any migration with time >= ``time`` will be removed.

        The node table is not affected by this operation.

        .. note:: This method does not have any specific sorting requirements
            and will maintain mutation parent mappings.

        :param float time: The cutoff time.
        """
        self._ll_tables.delete_older(time)

    def clear(
        self,
        clear_provenance=False,
        clear_metadata_schemas=False,
        clear_ts_metadata_and_schema=False,
    ):
        """
        Remove all rows of the data tables, optionally remove provenance, metadata
        schemas and ts-level metadata.

        :param bool clear_provenance: If ``True``, remove all rows of the provenance
            table. (Default: ``False``).
        :param bool clear_metadata_schemas: If ``True``, clear the table metadata
            schemas. (Default: ``False``).
        :param bool clear_ts_metadata_and_schema: If ``True``, clear the tree-sequence
            level metadata and schema (Default: ``False``).
        """
        self._ll_tables.clear(
            clear_provenance=clear_provenance,
            clear_metadata_schemas=clear_metadata_schemas,
            clear_ts_metadata_and_schema=clear_ts_metadata_and_schema,
        )

    def has_index(self):
        """
        Returns True if this TableCollection is indexed. See :ref:`sec_table_indexes`
        for information on indexes.
        """
        return bool(self._ll_tables.has_index())

    def build_index(self):
        """
        Builds an index on this TableCollection. Any existing indexes are automatically
        dropped.  See :ref:`sec_table_indexes` for information on indexes.
        """
        self._ll_tables.build_index()

    def drop_index(self):
        """
        Drops any indexes present on this table collection. If the tables are not
        currently indexed this method has no effect.  See :ref:`sec_table_indexes`
        for information on indexes.
        """
        self._ll_tables.drop_index()

    def subset(
        self,
        nodes,
        record_provenance=True,
        *,
        reorder_populations=None,
        remove_unreferenced=None,
    ):
        """
        Modifies the tables in place to contain only the entries referring to
        the provided list of node IDs, with nodes reordered according to the
        order they appear in the list. Other tables are :meth:`sorted <sort>`
        to conform to the :ref:`sec_valid_tree_sequence_requirements`, and
        additionally sorted as described in the documentation for the equivalent
        tree sequence method :meth:`TreeSequence.subset`: please see this for more
        detail.

        :param list nodes: The list of nodes for which to retain information. This
            may be a numpy array (or array-like) object (dtype=np.int32).
        :param bool record_provenance: Whether to record a provenance entry
            in the provenance table for this operation.
        :param bool reorder_populations: Whether to reorder the population table
            (default: True).  If False, the population table will not be altered
            in any way.
        :param bool remove_unreferenced: Whether sites, individuals, and populations
            that are not referred to by any retained entries in the tables should
            be removed (default: True). See the description for details.
        """
        reorder_populations = (
            True if reorder_populations is None else reorder_populations
        )
        remove_unreferenced = (
            True if remove_unreferenced is None else remove_unreferenced
        )
        nodes = util.safe_np_int_cast(nodes, np.int32)
        self._ll_tables.subset(
            nodes,
            reorder_populations=reorder_populations,
            remove_unreferenced=remove_unreferenced,
        )
        self.sort()
        if record_provenance:
            parameters = {"command": "subset", "nodes": nodes.tolist()}
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

    def union(
        self,
        other,
        node_mapping,
        check_shared_equality=True,
        add_populations=True,
        record_provenance=True,
    ):
        """
        Modifies the table collection in place by adding the non-shared
        portions of ``other`` to itself. To perform the node-wise union,
        the method relies on a ``node_mapping`` array, that maps nodes in
        ``other`` to its equivalent node in ``self`` or ``tskit.NULL`` if
        the node is exclusive to ``other``. See :meth:`TreeSequence.union` for a more
        detailed description.

        :param TableCollection other: Another table collection.
        :param list node_mapping: An array of node IDs that relate nodes in
            ``other`` to nodes in ``self``: the k-th element of ``node_mapping``
            should be the index of the equivalent node in ``self``, or
            ``tskit.NULL`` if the node is not present in ``self`` (in which case it
            will be added to self).
        :param bool check_shared_equality: If True, the shared portions of the
            table collections will be checked for equality.
        :param bool add_populations: If True, nodes new to ``self`` will be
            assigned new population IDs.
        :param bool record_provenance: Whether to record a provenance entry
            in the provenance table for this operation.
        """
        node_mapping = util.safe_np_int_cast(node_mapping, np.int32)
        self._ll_tables.union(
            other._ll_tables,
            node_mapping,
            check_shared_equality=check_shared_equality,
            add_populations=add_populations,
        )
        if record_provenance:
            other_records = [prov.record for prov in other.provenances]
            other_timestamps = [prov.timestamp for prov in other.provenances]
            parameters = {
                "command": "union",
                "other": {"timestamp": other_timestamps, "record": other_records},
                "node_mapping": node_mapping.tolist(),
            }
            self.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

    def ibd_segments(
        self,
        *,
        within=None,
        between=None,
        max_time=None,
        min_span=None,
        store_pairs=None,
        store_segments=None,
    ):
        """
        Equivalent to the :meth:`TreeSequence.ibd_segments` method; please see its
        documentation for more details, and use this method only if you specifically need
        to work with a :class:`TableCollection` object.

        This method has the same data requirements as
        :meth:`TableCollection.simplify`. In particular, the tables in the collection
        have :ref:`required <sec_valid_tree_sequence_requirements>` sorting orders.
        To enforce this, you can call :meth:`TableCollection.sort` before using this
        method. If the edge table contains any edges with identical
        parents and children over adjacent genomic intervals, any IBD intervals
        underneath the edges will also be split across the breakpoint(s). To prevent this
        behaviour in this situation, use :meth:`EdgeTable.squash` beforehand.

        :param list within: As for the :meth:`TreeSequence.ibd_segments` method.
        :param list[list] between: As for the :meth:`TreeSequence.ibd_segments` method.
        :param float max_time: As for the :meth:`TreeSequence.ibd_segments` method.
        :param float min_span: As for the :meth:`TreeSequence.ibd_segments` method.
        :param bool store_pairs: As for the :meth:`TreeSequence.ibd_segments` method.
        :param bool store_segments: As for the :meth:`TreeSequence.ibd_segments` method.
        :return: An :class:`.IdentitySegments` object containing the recorded
            IBD information.
        :rtype: IdentitySegments
        """
        max_time = np.inf if max_time is None else max_time
        min_span = 0 if min_span is None else min_span
        store_pairs = False if store_pairs is None else store_pairs
        store_segments = False if store_segments is None else store_segments
        if within is not None and between is not None:
            raise ValueError(
                "The ``within`` and ``between`` arguments are mutually exclusive"
            )
        if between is not None:
            sample_set_sizes = np.array(
                [len(sample_set) for sample_set in between], dtype=np.uint64
            )
            # hstack has some annoying quirks around its handling of empty
            # lists which we need to work around. In a way it would be more
            # convenient to detect these conditions as errors, but then we
            # end up having to workaround edge cases in the tests and its
            # mathematically neater this way.
            pre_flattened = [lst for lst in between if len(lst) > 0]
            if len(pre_flattened) == 0:
                flattened = []
            else:
                flattened = util.safe_np_int_cast(np.hstack(pre_flattened), np.int32)
            ll_result = self._ll_tables.ibd_segments_between(
                sample_set_sizes=sample_set_sizes,
                sample_sets=flattened,
                max_time=max_time,
                min_span=min_span,
                store_pairs=store_pairs,
                store_segments=store_segments,
            )
        else:
            if within is not None:
                within = util.safe_np_int_cast(within, np.int32)
            ll_result = self._ll_tables.ibd_segments_within(
                samples=within,
                max_time=max_time,
                min_span=min_span,
                store_pairs=store_pairs,
                store_segments=store_segments,
            )
        return IdentitySegments(
            ll_result,
            max_time=max_time,
            min_span=min_span,
            store_pairs=store_pairs,
            store_segments=store_segments,
        )
