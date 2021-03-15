# MIT License
#
# Copyright (c) 2018-2020 Tskit Developers
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
import base64
import datetime
import itertools
import json
import sys
import warnings
from typing import Any
from typing import Tuple

import attr
import numpy as np

import _tskit
import tskit
import tskit.metadata as metadata
import tskit.provenance as provenance
import tskit.util as util
from tskit import UNKNOWN_TIME

attr_options = {"slots": True, "frozen": True, "auto_attribs": True}


@attr.s(eq=False, **attr_options)
class IndividualTableRow:
    flags: int
    location: np.ndarray
    parents: np.ndarray
    metadata: bytes

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return all(
                (
                    self.flags == other.flags,
                    np.array_equal(self.location, other.location),
                    np.array_equal(self.parents, other.parents),
                    self.metadata == other.metadata,
                )
            )

    def __neq__(self, other):
        return not self.__eq__(other)


@attr.s(**attr_options)
class NodeTableRow:
    flags: int
    time: float
    population: int
    individual: int
    metadata: bytes


@attr.s(**attr_options)
class EdgeTableRow:
    left: float
    right: float
    parent: int
    child: int
    metadata: bytes


@attr.s(**attr_options)
class MigrationTableRow:
    left: float
    right: float
    node: int
    source: int
    dest: int
    time: float
    metadata: bytes


@attr.s(**attr_options)
class SiteTableRow:
    position: float
    ancestral_state: str
    metadata: bytes


@attr.s(eq=False, **attr_options)
class MutationTableRow:
    site: int
    node: int
    derived_state: str
    parent: int
    metadata: bytes
    time: float

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


@attr.s(**attr_options)
class PopulationTableRow:
    metadata: bytes


@attr.s(**attr_options)
class ProvenanceTableRow:
    timestamp: str
    record: str


@attr.s(**attr_options)
class TableCollectionIndexes:
    edge_insertion_order: np.ndarray = attr.ib(default=None)
    edge_removal_order: np.ndarray = attr.ib(default=None)

    def asdict(self):
        return attr.asdict(self, filter=lambda k, v: v is not None)

    @property
    def nbytes(self):
        return self.edge_insertion_order.nbytes + self.edge_removal_order.nbytes


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

    def __init__(self, ll_table, row_class, **kwargs):
        self.ll_table = ll_table
        self.row_class = row_class
        super().__init__(**kwargs)

    def _check_required_args(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                raise TypeError(f"{k} is required")

    @property
    def num_rows(self):
        return self.ll_table.num_rows

    @property
    def max_rows(self):
        return self.ll_table.max_rows

    @property
    def max_rows_increment(self):
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

    def __getitem__(self, index):
        """
        Return the specifed row of this table, decoding metadata if it is present.
        Supports negative indexing, e.g. ``table[-5]``.

        :param int index: the zero-index of the desired row
        """
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds")
        row = self.ll_table.get_row(index)
        try:
            row = self.decode_row(row)
        except AttributeError:
            # This means the class returns the low-level row unchanged.
            pass
        return self.row_class(*row)

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
        Sets the values for each column in this :class:`Table` using
        values provided in numpy arrays. Overwrites any data currently stored in
        the table.
        """
        raise NotImplementedError()

    def __str__(self):
        headers, rows = self._text_header_and_rows()
        return "\n".join("\t".join(row) for row in [headers] + rows)

    def _repr_html_(self):
        """
        Called by jupyter notebooks to render tables
        """
        headers, rows = self._text_header_and_rows(limit=40)
        headers = "".join(f"<th>{header}</th>" for header in headers)
        rows = (
            f"<td><em>... skipped {row[11:]} rows ...</em></td>"
            if "__skipped__" in row
            else "".join(f"<td>{cell}</td>" for cell in row)
            for row in rows
        )
        rows = "".join(f"<tr>{row}</tr>\n" for row in rows)
        return f"""
            <div>
                <style scoped="">
                    .tskit-table tbody tr th:only-of-type {{vertical-align: middle;}}
                    .tskit-table tbody tr th {{vertical-align: top;}}
                    .tskit-table tbody td {{text-align: right;padding: 0.5em 0.5em;}}
                    .tskit-table tbody th {{padding: 0.5em 0.5em;}}
                </style>
                <table border="1" class="tskit-table">
                    <thead>
                        <tr>
                            {headers}
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
        """


class MetadataMixin:
    """
    Mixin class for tables that have a metadata column.
    """

    def __init__(self):
        self.metadata_column_index = list(
            attr.fields_dict(self.row_class).keys()
        ).index("metadata")
        self._update_metadata_schema_cache_from_ll()

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
        return self._metadata_schema_cache

    @metadata_schema.setter
    def metadata_schema(self, schema: metadata.MetadataSchema) -> None:
        if not isinstance(schema, metadata.MetadataSchema):
            raise TypeError(
                "Only instances of tskit.MetadataSchema can be assigned to "
                f"metadata_schema, not {type(schema)}"
            )
        self.ll_table.metadata_schema = repr(schema)
        self._update_metadata_schema_cache_from_ll()

    def decode_row(self, row: Tuple[Any]) -> Tuple:
        return (
            row[: self.metadata_column_index]
            + (self._metadata_schema_cache.decode_row(row[self.metadata_column_index]),)
            + row[self.metadata_column_index + 1 :]
        )

    def _update_metadata_schema_cache_from_ll(self) -> None:
        self._metadata_schema_cache = metadata.parse_metadata_schema(
            self.ll_table.metadata_schema
        )


class IndividualTable(BaseTable, MetadataMixin):
    """
    A table defining the individuals in a tree sequence. Note that although
    each Individual has associated nodes, reference to these is not stored in
    the individual table, but rather reference to the individual is stored for
    each node in the :class:`NodeTable`.  This is similar to the way in which
    the relationship between sites and mutations is modelled.

    :warning: The numpy arrays returned by table attribute accesses are **copies**
        of the underlying data. In particular, this means that you cannot edit
        the values in the columns by updating the attribute arrays.

        **NOTE:** this behaviour may change in future.

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
        flags = self.flags
        location = util.unpack_arrays(self.location, self.location_offset)
        parents = util.unpack_arrays(self.parents, self.parents_offset)
        metadata = util.unpack_bytes(self.metadata, self.metadata_offset)
        headers = ("id", "flags", "location", "parents", "metadata")
        rows = []
        if limit is None or self.num_rows <= limit:
            indexes = range(self.num_rows)
        else:
            indexes = itertools.chain(
                range(limit // 2),
                [-1],
                range(self.num_rows - (limit - (limit // 2)), self.num_rows),
            )
        for j in indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                md = base64.b64encode(metadata[j]).decode("utf8")
                location_str = ",".join(map(str, location[j]))
                parents_str = ",".join(map(str, parents[j]))
                rows.append(
                    "{}\t{}\t{}\t{}\t{}".format(
                        j, flags[j], location_str, parents_str, md
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
        :return: The ID of the newly added node.
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
        values in the specified arrays. Overwrites any data currently stored in
        the table.

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
        :param metadata_schema: The encoded metadata schema.
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


class NodeTable(BaseTable, MetadataMixin):
    """
    A table defining the nodes in a tree sequence. See the
    :ref:`definitions <sec_node_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for a node table to be a part of a valid tree sequence.

    :warning: The numpy arrays returned by table attribute accesses are **copies**
        of the underlying data. In particular, this means that you cannot edit
        the values in the columns by updating the attribute arrays.

        **NOTE:** this behaviour may change in future.

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
        time = self.time
        flags = self.flags
        population = self.population
        individual = self.individual
        metadata = util.unpack_bytes(self.metadata, self.metadata_offset)
        headers = ("id", "flags", "population", "individual", "time", "metadata")
        rows = []
        if limit is None or self.num_rows <= limit:
            indexes = range(self.num_rows)
        else:
            indexes = itertools.chain(
                range(limit // 2),
                [-1],
                range(self.num_rows - (limit - (limit // 2)), self.num_rows),
            )
        for j in indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                md = base64.b64encode(metadata[j]).decode("utf8")
                rows.append(
                    "{}\t{}\t{}\t{}\t{:.14f}\t{}".format(
                        j, flags[j], population[j], individual[j], time[j], md
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
        the specified arrays. Overwrites any data currently stored in the table.

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
        :param metadata_schema: The encoded metadata schema.
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
                metadata_schema=None,
            )
        )


class EdgeTable(BaseTable, MetadataMixin):
    """
    A table defining the edges in a tree sequence. See the
    :ref:`definitions <sec_edge_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for an edge table to be a part of a valid tree sequence.

    :warning: The numpy arrays returned by table attribute accesses are **copies**
        of the underlying data. In particular, this means that you cannot edit
        the values in the columns by updating the attribute arrays.

        **NOTE:** this behaviour may change in future.

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
        left = self.left
        right = self.right
        parent = self.parent
        child = self.child
        metadata = util.unpack_bytes(self.metadata, self.metadata_offset)
        headers = ("id", "left\t", "right\t", "parent", "child", "metadata")
        rows = []
        if limit is None or self.num_rows <= limit:
            indexes = range(self.num_rows)
        else:
            indexes = itertools.chain(
                range(limit // 2),
                [-1],
                range(self.num_rows - (limit - (limit // 2)), self.num_rows),
            )
        for j in indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                md = base64.b64encode(metadata[j]).decode("utf8")
                rows.append(
                    "{}\t{:.8f}\t{:.8f}\t{}\t{}\t{}".format(
                        j, left[j], right[j], parent[j], child[j], md
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
        in the specified arrays. Overwrites any data currently stored in the table.

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
        :param metadata_schema: The encoded metadata schema.
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
        The new edge table will be sorted in the canonical order (P, C, L, R).
        """
        self.ll_table.squash()


class MigrationTable(BaseTable, MetadataMixin):
    """
    A table defining the migrations in a tree sequence. See the
    :ref:`definitions <sec_migration_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for a migration table to be a part of a valid tree
    sequence.

    :warning: The numpy arrays returned by table attribute accesses are **copies**
        of the underlying data. In particular, this means that you cannot edit
        the values in the columns by updating the attribute arrays.

        **NOTE:** this behaviour may change in future.

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
        left = self.left
        right = self.right
        node = self.node
        source = self.source
        dest = self.dest
        time = self.time
        metadata = util.unpack_bytes(self.metadata, self.metadata_offset)
        headers = ("id", "left", "right", "node", "source", "dest", "time", "metadata")
        rows = []
        if limit is None or self.num_rows <= limit:
            indexes = range(self.num_rows)
        else:
            indexes = itertools.chain(
                range(limit // 2),
                [-1],
                range(self.num_rows - (limit - (limit // 2)), self.num_rows),
            )
        for j in indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                md = base64.b64encode(metadata[j]).decode("utf8")
                rows.append(
                    "{}\t{:.8f}\t{:.8f}\t{}\t{}\t{}\t{:.8f}\t{}".format(
                        j, left[j], right[j], node[j], source[j], dest[j], time[j], md
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
        in the specified arrays. Overwrites any data currently stored in the table.

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
        :param metadata_schema: The encoded metadata schema.
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


class SiteTable(BaseTable, MetadataMixin):
    """
    A table defining the sites in a tree sequence. See the
    :ref:`definitions <sec_site_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for a site table to be a part of a valid tree
    sequence.

    :warning: The numpy arrays returned by table attribute accesses are **copies**
        of the underlying data. In particular, this means that you cannot edit
        the values in the columns by updating the attribute arrays.

        **NOTE:** this behaviour may change in future.

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
        position = self.position
        ancestral_state = util.unpack_strings(
            self.ancestral_state, self.ancestral_state_offset
        )
        metadata = util.unpack_bytes(self.metadata, self.metadata_offset)
        headers = ("id", "position", "ancestral_state", "metadata")
        rows = []
        if limit is None or self.num_rows <= limit:
            indexes = range(self.num_rows)
        else:
            indexes = itertools.chain(
                range(limit // 2),
                [-1],
                range(self.num_rows - (limit - (limit // 2)), self.num_rows),
            )
        for j in indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                md = base64.b64encode(metadata[j]).decode("utf8")
                rows.append(
                    "{}\t{:.8f}\t{}\t{}".format(
                        j, position[j], ancestral_state[j], md
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
        in the specified arrays. Overwrites any data currently stored in the table.

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
        :param metadata_schema: The encoded metadata schema.
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


class MutationTable(BaseTable, MetadataMixin):
    """
    A table defining the mutations in a tree sequence. See the
    :ref:`definitions <sec_mutation_table_definition>` for details on the columns
    in this table and the
    :ref:`tree sequence requirements <sec_valid_tree_sequence_requirements>` section
    for the properties needed for a mutation table to be a part of a valid tree
    sequence.

    :warning: The numpy arrays returned by table attribute accesses are **copies**
        of the underlying data. In particular, this means that you cannot edit
        the values in the columns by updating the attribute arrays.

        **NOTE:** this behaviour may change in future.

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
        site = self.site
        node = self.node
        parent = self.parent
        time = self.time
        derived_state = util.unpack_strings(
            self.derived_state, self.derived_state_offset
        )
        metadata = util.unpack_bytes(self.metadata, self.metadata_offset)
        headers = ("id", "site", "node", "time", "derived_state", "parent", "metadata")
        rows = []
        if limit is None or self.num_rows <= limit:
            indexes = range(self.num_rows)
        else:
            indexes = itertools.chain(
                range(limit // 2),
                [-1],
                range(self.num_rows - (limit - (limit // 2)), self.num_rows),
            )
        for j in indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                md = base64.b64encode(metadata[j]).decode("utf8")
                rows.append(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        j, site[j], node[j], time[j], derived_state[j], parent[j], md
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
        in the specified arrays. Overwrites any data currently stored in the table.

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
        :param metadata_schema: The encoded metadata schema.
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


class PopulationTable(BaseTable, MetadataMixin):
    """
    A table defining the populations referred to in a tree sequence.
    The PopulationTable stores metadata for populations that may be referred to
    in the NodeTable and MigrationTable".  Note that although nodes
    may be associated with populations, this association is stored in
    the :class:`NodeTable`: only metadata on each population is stored
    in the population table.

    :warning: The numpy arrays returned by table attribute accesses are **copies**
        of the underlying data. In particular, this means that you cannot edit
        the values in the columns by updating the attribute arrays.

        **NOTE:** this behaviour may change in future.

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
        metadata = util.unpack_bytes(self.metadata, self.metadata_offset)
        headers = ("id", "metadata")
        rows = []
        if limit is None or self.num_rows <= limit:
            indexes = range(self.num_rows)
        else:
            indexes = itertools.chain(
                range(limit // 2),
                [-1],
                range(self.num_rows - (limit - (limit // 2)), self.num_rows),
            )
        for j in indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                md = base64.b64encode(metadata[j]).decode("utf8")
                rows.append((str(j), str(md)))
        return headers, rows

    def set_columns(self, metadata=None, metadata_offset=None, metadata_schema=None):
        """
        Sets the values for each column in this :class:`PopulationTable` using the
        values in the specified arrays. Overwrites any data currently stored in the
        table.

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
        :param metadata_schema: The encoded metadata schema.
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
        values in the specified arrays. Overwrites any data currently stored in the
        table.

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
        timestamp = util.unpack_strings(self.timestamp, self.timestamp_offset)
        record = util.unpack_strings(self.record, self.record_offset)
        headers = ("id", "timestamp", "record")
        rows = []
        if limit is None or self.num_rows <= limit:
            indexes = range(self.num_rows)
        else:
            indexes = itertools.chain(
                range(limit // 2),
                [-1],
                range(self.num_rows - (limit - (limit // 2)), self.num_rows),
            )
        for j in indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                rows.append((str(j), str(timestamp[j]), str(record[j])))
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


class TableCollection:
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

    :ivar individuals: The individual table.
    :vartype individuals: IndividualTable
    :ivar nodes: The node table.
    :vartype nodes: NodeTable
    :ivar edges: The edge table.
    :vartype edges: EdgeTable
    :ivar migrations: The migration table.
    :vartype migrations: MigrationTable
    :ivar sites: The site table.
    :vartype sites: SiteTable
    :ivar mutations: The mutation table.
    :vartype mutations: MutationTable
    :ivar populations: The population table.
    :vartype populations: PopulationTable
    :ivar provenances: The provenance table.
    :vartype provenances: ProvenanceTable
    :ivar index: The edge insertion and removal index.
    :ivar sequence_length: The sequence length defining the coordinate
        space.
    :vartype sequence_length: float
    :ivar file_uuid: The UUID for the file this TableCollection is derived
        from, or None if not derived from a file.
    :vartype file_uuid: str
    """

    def __init__(self, sequence_length=0):
        self._ll_tables = _tskit.TableCollection(sequence_length)

    @property
    def individuals(self):
        return IndividualTable(ll_table=self._ll_tables.individuals)

    @property
    def nodes(self):
        return NodeTable(ll_table=self._ll_tables.nodes)

    @property
    def edges(self):
        return EdgeTable(ll_table=self._ll_tables.edges)

    @property
    def migrations(self):
        return MigrationTable(ll_table=self._ll_tables.migrations)

    @property
    def sites(self):
        return SiteTable(ll_table=self._ll_tables.sites)

    @property
    def mutations(self):
        return MutationTable(ll_table=self._ll_tables.mutations)

    @property
    def populations(self):
        return PopulationTable(ll_table=self._ll_tables.populations)

    @property
    def provenances(self):
        return ProvenanceTable(ll_table=self._ll_tables.provenances)

    @property
    def indexes(self):
        indexes = self._ll_tables.indexes
        return TableCollectionIndexes(**indexes)

    @indexes.setter
    def indexes(self, indexes):
        self._ll_tables.indexes = indexes.asdict()

    @property
    def sequence_length(self):
        return self._ll_tables.sequence_length

    @sequence_length.setter
    def sequence_length(self, sequence_length):
        self._ll_tables.sequence_length = sequence_length

    @property
    def file_uuid(self):
        return self._ll_tables.file_uuid

    @property
    def metadata_schema(self) -> metadata.MetadataSchema:
        """
        The :class:`tskit.MetadataSchema` for this TableCollection.
        """
        return metadata.parse_metadata_schema(self._ll_tables.metadata_schema)

    @metadata_schema.setter
    def metadata_schema(self, schema: metadata.MetadataSchema) -> None:
        # Check the schema is a valid schema instance by roundtripping it.
        metadata.parse_metadata_schema(repr(schema))
        self._ll_tables.metadata_schema = repr(schema)

    @property
    def metadata(self) -> Any:
        """
        The decoded metadata for this TableCollection.
        """
        return self.metadata_schema.decode_row(self._ll_tables.metadata)

    @metadata.setter
    def metadata(self, metadata: Any) -> None:
        self._ll_tables.metadata = self.metadata_schema.validate_and_encode_row(
            metadata
        )

    @property
    def metadata_bytes(self) -> Any:
        """
        The raw bytes of metadata for this TableCollection
        """
        return self._ll_tables.metadata

    def asdict(self):
        """
        Returns a dictionary representation of this TableCollection.

        Note: the semantics of this method changed at tskit 0.1.0. Previously a
        map of table names to the tables themselves was returned.
        """
        ret = {
            "encoding_version": (1, 3),
            "sequence_length": self.sequence_length,
            "metadata_schema": repr(self.metadata_schema),
            "metadata": self.metadata_schema.encode_row(self.metadata),
            "individuals": self.individuals.asdict(),
            "nodes": self.nodes.asdict(),
            "edges": self.edges.asdict(),
            "migrations": self.migrations.asdict(),
            "sites": self.sites.asdict(),
            "mutations": self.mutations.asdict(),
            "populations": self.populations.asdict(),
            "provenances": self.provenances.asdict(),
            "indexes": self.indexes.asdict(),
        }
        return ret

    @property
    def name_map(self):
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
    def nbytes(self) -> int:
        """
        Returns the total number of bytes required to store the data
        in this table collection. Note that this may not be equal to
        the actual memory footprint.
        """
        return sum(
            (
                8,  # sequence_length takes 8 bytes
                len(self.metadata_bytes),
                len(repr(self.metadata_schema).encode()),
                self.indexes.nbytes,
                sum(table.nbytes for table in self.name_map.values()),
            )
        )

    def __banner(self, title):
        width = 60
        line = "#" * width
        title_line = f"#   {title}"
        title_line += " " * (width - len(title_line) - 1)
        title_line += "#"
        return line + "\n" + title_line + "\n" + line + "\n"

    def __str__(self):
        s = self.__banner("Individuals")
        s += str(self.individuals) + "\n"
        s += self.__banner("Nodes")
        s += str(self.nodes) + "\n"
        s += self.__banner("Edges")
        s += str(self.edges) + "\n"
        s += self.__banner("Sites")
        s += str(self.sites) + "\n"
        s += self.__banner("Mutations")
        s += str(self.mutations) + "\n"
        s += self.__banner("Migrations")
        s += str(self.migrations) + "\n"
        s += self.__banner("Populations")
        s += str(self.populations) + "\n"
        s += self.__banner("Provenances")
        s += str(self.provenances)
        return s

    def equals(
        self,
        other,
        *,
        ignore_metadata=False,
        ignore_ts_metadata=False,
        ignore_provenance=False,
        ignore_timestamps=False,
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
                )
            )
        return ret

    def __eq__(self, other):
        return self.equals(other)

    def __getstate__(self):
        return self.asdict()

    @classmethod
    def load(cls, file_or_path):
        file, local_file = util.convert_file_like_to_open_file(file_or_path, "rb")
        ll_tc = _tskit.TableCollection(1)
        ll_tc.load(file)
        tc = TableCollection(1)
        tc._ll_tables = ll_tc
        return tc

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
        self.__init__(state["sequence_length"])
        self.metadata_schema = tskit.parse_metadata_schema(state["metadata_schema"])
        self.metadata = self.metadata_schema.decode_row(state["metadata"])
        self.individuals.set_columns(**state["individuals"])
        self.nodes.set_columns(**state["nodes"])
        self.edges.set_columns(**state["edges"])
        self.migrations.set_columns(**state["migrations"])
        self.sites.set_columns(**state["sites"])
        self.mutations.set_columns(**state["mutations"])
        self.populations.set_columns(**state["populations"])
        self.provenances.set_columns(**state["provenances"])

    @classmethod
    def fromdict(self, tables_dict):
        tables = TableCollection(tables_dict["sequence_length"])
        try:
            tables.metadata_schema = tskit.parse_metadata_schema(
                tables_dict["metadata_schema"]
            )
        except KeyError:
            pass
        try:
            tables.metadata = tables.metadata_schema.decode_row(tables_dict["metadata"])
        except KeyError:
            pass
        tables.individuals.set_columns(**tables_dict["individuals"])
        tables.nodes.set_columns(**tables_dict["nodes"])
        tables.edges.set_columns(**tables_dict["edges"])
        tables.migrations.set_columns(**tables_dict["migrations"])
        tables.sites.set_columns(**tables_dict["sites"])
        tables.mutations.set_columns(**tables_dict["mutations"])
        tables.populations.set_columns(**tables_dict["populations"])
        tables.provenances.set_columns(**tables_dict["provenances"])

        # Indexes must be last as other wise the check for their consistency will fail
        try:
            tables.indexes = TableCollectionIndexes(**tables_dict["indexes"])
        except KeyError:
            pass
        return tables

    def copy(self):
        """
        Returns a deep copy of this TableCollection.

        :return: A deep copy of this TableCollection.
        :rtype: .TableCollection
        """
        return TableCollection.fromdict(self.asdict())

    def tree_sequence(self):
        """
        Returns a :class:`TreeSequence` instance with the structure defined by the
        tables in this :class:`TableCollection`. If the table collection is not
        in canonical form (i.e., does not meet sorting requirements) or cannot be
        interpreted as a tree sequence an exception is raised. The
        :meth:`.sort` method may be used to ensure that input sorting requirements
        are met. If the table collection does not have indexes they will be
        built.

        :return: A :class:`TreeSequence` instance reflecting the structures
            defined in this set of tables.
        :rtype: .TreeSequence
        """
        if not self.has_index():
            self.build_index()
        return tskit.TreeSequence.load_tables(self)

    def simplify(
        self,
        samples=None,
        *,
        reduce_to_site_topology=False,
        filter_populations=True,
        filter_individuals=True,
        filter_sites=True,
        keep_unary=False,
        keep_unary_in_individuals=None,
        keep_input_roots=False,
        record_provenance=True,
        filter_zero_mutation_sites=None,  # Deprecated alias for filter_sites
    ):
        """
        Simplifies the tables in place to retain only the information necessary
        to reconstruct the tree sequence describing the given ``samples``.
        This will change the ID of the nodes, so that the node
        ``samples[k]`` will have ID ``k`` in the result. The resulting
        NodeTable will have only the first ``len(samples)`` nodes marked
        as samples. The mapping from node IDs in the current set of tables to
        their equivalent values in the simplified tables is also returned as a
        numpy array. If an array ``a`` is returned by this function and ``u``
        is the ID of a node in the input table, then ``a[u]`` is the ID of this
        node in the output table. For any node ``u`` that is not mapped into
        the output tables, this mapping will equal ``-1``.

        Tables operated on by this function must: be sorted (see
        :meth:`TableCollection.sort`), have children be born strictly after their
        parents, and the intervals on which any node is a child must be
        disjoint. Other than this the tables need not satisfy remaining
        requirements to specify a valid tree sequence (but the resulting tables
        will).

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
            not be altered in any way. (Default: True)
        :param bool filter_individuals: If True, remove any individuals that are
            not referenced by nodes after simplification; new individual IDs are
            allocated sequentially from zero. If False, the individual table will
            not be altered in any way. (Default: True)
        :param bool filter_sites: If True, remove any sites that are
            not referenced by mutations after simplification; new site IDs are
            allocated sequentially from zero. If False, the site table will not
            be altered in any way. (Default: True)
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
            )
            filter_sites = filter_zero_mutation_sites
        if samples is None:
            flags = self.nodes.flags
            samples = np.where(np.bitwise_and(flags, _tskit.NODE_IS_SAMPLE) != 0)[
                0
            ].astype(np.int32)
        else:
            samples = util.safe_np_int_cast(samples, np.int32)
        if keep_unary_in_individuals is None:
            keep_unary_in_individuals = False

        node_map = self._ll_tables.simplify(
            samples,
            filter_sites=filter_sites,
            filter_individuals=filter_individuals,
            filter_populations=filter_populations,
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

    def sort(self, edge_start=0):
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

        The individual, node, population and provenance tables are not affected
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
        """
        self._ll_tables.sort(edge_start)
        # TODO add provenance

    def canonicalise(self, remove_unreferenced=None):
        """
        This puts the tables in *canonical* form - to do this, the individual
        and population tables are sorted by the first node that refers to each
        (see :meth:`TreeSequence.subset`) Then, the remaining tables are sorted
        as in :meth:`.sort`, with the modification that mutations are sorted by
        site, then time, then number of descendant mutations (ensuring that
        parent mutations occur before children), then node, then original order
        in the tables. This ensures that any two tables with the same
        information should be identical after canonical sorting.

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

        .. note:: note: This method does not check that all mutations result
            in a change of state, as required; see :ref:`sec_mutation_requirements`.

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
        Returns True if this TableCollection is indexed.
        """
        return bool(self._ll_tables.has_index())

    def build_index(self):
        """
        Builds an index on this TableCollection. Any existing indexes are automatically
        dropped.
        """
        self._ll_tables.build_index()

    def drop_index(self):
        """
        Drops any indexes present on this table collection. If the tables are not
        currently indexed this method has no effect.
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
        order they appear in the list. See :meth:`TreeSequence.subset` for a
        more detailed description.

        Note: there are no sortedness requirements on the tables.

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

    def find_ibd(self, samples, max_time=None, min_length=None):
        max_time = sys.float_info.max if max_time is None else max_time
        min_length = 0 if min_length is None else min_length
        return self._ll_tables.find_ibd(
            samples, max_time=max_time, min_length=min_length
        )
