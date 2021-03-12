# MIT License
#
# Copyright (c) 2018-2021 Tskit Developers
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
Test cases for the low-level tables used to transfer information
between simulations and the tree sequence.
"""
import io
import json
import math
import pathlib
import pickle
import platform
import random
import struct
import time
import unittest
import warnings

import attr
import kastore
import msprime
import numpy as np
import pytest

import _tskit
import tests.test_wright_fisher as wf
import tests.tsutil as tsutil
import tskit
import tskit.exceptions as exceptions
import tskit.metadata as metadata


class Column:
    def __init__(self, name):
        self.name = name


class Int32Column(Column):
    def get_input(self, n):
        return 1 + np.arange(n, dtype=np.int32)


class UInt8Column(Column):
    def get_input(self, n):
        return 2 + np.arange(n, dtype=np.uint8)


class UInt32Column(Column):
    def get_input(self, n):
        return 3 + np.arange(n, dtype=np.uint32)


class CharColumn(Column):
    def get_input(self, n):
        return np.zeros(n, dtype=np.int8)


class DoubleColumn(Column):
    def get_input(self, n):
        return 4 + np.arange(n, dtype=np.float64)


class CommonTestsMixin:
    """
    Abstract base class for common table tests. Because of the design of unittest,
    we have to make this a mixin.
    """

    def make_input_data(self, num_rows):
        input_data = {col.name: col.get_input(num_rows) for col in self.columns}
        for list_col, offset_col in self.ragged_list_columns:
            value = list_col.get_input(num_rows)
            input_data[list_col.name] = value
            input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
        return input_data

    def test_max_rows_increment(self):
        for bad_value in [-1, -(2 ** 10)]:
            with pytest.raises(ValueError):
                self.table_class(max_rows_increment=bad_value)
        for v in [1, 100, 256]:
            table = self.table_class(max_rows_increment=v)
            assert table.max_rows_increment == v
        # Setting zero or not argument both denote the default.
        table = self.table_class()
        assert table.max_rows_increment == 1024
        table = self.table_class(max_rows_increment=0)
        assert table.max_rows_increment == 1024

    def test_low_level_get_row(self):
        # Tests the low-level get_row interface to ensure we're getting coverage.
        t = self.table_class()
        with pytest.raises(TypeError):
            t.ll_table.get_row()
        with pytest.raises(TypeError):
            t.ll_table.get_row("row")
        with pytest.raises(_tskit.LibraryError):
            t.ll_table.get_row(1)

    def test_low_level_equals(self):
        # Tests the low-level equals interface to ensure we're getting coverage.
        t = self.table_class()
        with pytest.raises(TypeError):
            t.ll_table.equals()
        with pytest.raises(TypeError):
            t.ll_table.equals(None)

    def test_low_level_set_columns(self):
        t = self.table_class()
        with pytest.raises(TypeError):
            t.ll_table.set_columns(None)
        with pytest.raises(TypeError):
            t.ll_table.append_columns(None)

    def test_input_parameters_errors(self):
        assert len(self.input_parameters) > 0
        for param, _ in self.input_parameters:
            for bad_value in [-1, -(2 ** 10)]:
                with pytest.raises(ValueError):
                    self.table_class(**{param: bad_value})
            for bad_type in [None, ValueError, "ser"]:
                with pytest.raises(TypeError):
                    self.table_class(**{param: bad_type})

    def test_input_parameter_values(self):
        assert len(self.input_parameters) > 0
        for param, _ in self.input_parameters:
            for v in [1, 100, 256]:
                table = self.table_class(**{param: v})
                assert getattr(table, param) == v

    def test_set_columns_string_errors(self):
        inputs = {c.name: c.get_input(1) for c in self.columns}
        for list_col, offset_col in self.ragged_list_columns:
            value = list_col.get_input(1)
            inputs[list_col.name] = value
            inputs[offset_col.name] = [0, 1]
        # Make sure this works.
        table = self.table_class()
        table.set_columns(**inputs)
        for list_col, offset_col in self.ragged_list_columns:
            kwargs = dict(inputs)
            del kwargs[list_col.name]
            with pytest.raises(TypeError):
                table.set_columns(**kwargs)
            kwargs = dict(inputs)
            del kwargs[offset_col.name]
            with pytest.raises(TypeError):
                table.set_columns(**kwargs)

    def test_set_columns_interface(self):
        kwargs = self.make_input_data(1)
        # Make sure this works.
        table = self.table_class()
        table.set_columns(**kwargs)
        table.append_columns(**kwargs)
        for focal_col in self.columns:
            table = self.table_class()
            for bad_type in [Exception, tskit]:
                error_kwargs = dict(kwargs)
                error_kwargs[focal_col.name] = bad_type
                with pytest.raises(ValueError):
                    table.set_columns(**error_kwargs)
                with pytest.raises(ValueError):
                    table.append_columns(**error_kwargs)
            for bad_value in ["qwer", [0, "sd"]]:
                error_kwargs = dict(kwargs)
                error_kwargs[focal_col.name] = bad_value
                with pytest.raises(ValueError):
                    table.set_columns(**error_kwargs)
                with pytest.raises(ValueError):
                    table.append_columns(**error_kwargs)

    def test_set_columns_from_dict(self):
        kwargs = self.make_input_data(1)
        # Make sure this works.
        t1 = self.table_class()
        t1.set_columns(**kwargs)
        t2 = self.table_class()
        t2.set_columns(**t1.asdict())
        assert t1 == t2

    def test_set_columns_dimension(self):
        kwargs = self.make_input_data(1)
        table = self.table_class()
        table.set_columns(**kwargs)
        table.append_columns(**kwargs)
        for focal_col in self.columns:
            table = self.table_class()
            for bad_dims in [5, [[1], [1]], np.zeros((2, 2))]:
                error_kwargs = dict(kwargs)
                error_kwargs[focal_col.name] = bad_dims
                with pytest.raises(ValueError):
                    table.set_columns(**error_kwargs)
                with pytest.raises(ValueError):
                    table.append_columns(**error_kwargs)
        for _, offset_col in self.ragged_list_columns:
            error_kwargs = dict(kwargs)
            for bad_dims in [5, [[1], [1]], np.zeros((2, 2))]:
                error_kwargs[offset_col.name] = bad_dims
                with pytest.raises(ValueError):
                    table.set_columns(**error_kwargs)
                with pytest.raises(ValueError):
                    table.append_columns(**error_kwargs)
            # Empty offset columns are caught also
            error_kwargs[offset_col.name] = []
            with pytest.raises(ValueError):
                table.set_columns(**error_kwargs)

    def test_set_columns_input_sizes(self):
        input_data = self.make_input_data(100)
        col_map = {col.name: col for col in self.columns}
        for list_col, offset_col in self.ragged_list_columns:
            col_map[list_col.name] = list_col
            col_map[offset_col.name] = offset_col
        table = self.table_class()
        table.set_columns(**input_data)
        table.append_columns(**input_data)
        for equal_len_col_set in self.equal_len_columns:
            if len(equal_len_col_set) > 1:
                for col in equal_len_col_set:
                    kwargs = dict(input_data)
                    kwargs[col] = col_map[col].get_input(1)
                    with pytest.raises(ValueError):
                        table.set_columns(**kwargs)
                    with pytest.raises(ValueError):
                        table.append_columns(**kwargs)

    def test_set_read_only_attributes(self):
        table = self.table_class()
        with pytest.raises(AttributeError):
            table.num_rows = 10
        with pytest.raises(AttributeError):
            table.max_rows = 10
        for param, _default in self.input_parameters:
            with pytest.raises(AttributeError):
                setattr(table, param, 2)
        assert table.num_rows == 0
        assert len(table) == 0

    def test_set_column_attributes_empty(self):
        table = self.table_class()
        input_data = {col.name: col.get_input(0) for col in self.columns}
        for col, data in input_data.items():
            setattr(table, col, data)
            assert len(getattr(table, col)) == 0

    def test_set_column_attributes_data(self):
        table = self.table_class()
        for num_rows in [1, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table.set_columns(**input_data)

            for list_col, offset_col in self.ragged_list_columns:
                list_data = input_data[list_col.name]
                assert np.array_equal(getattr(table, list_col.name), list_data)
                list_data += 1
                assert not np.array_equal(getattr(table, list_col.name), list_data)
                setattr(table, list_col.name, list_data)
                assert np.array_equal(getattr(table, list_col.name), list_data)
                list_value = getattr(table[0], list_col.name)
                assert len(list_value) == 1

                # Reset the offsets so that all the full array is associated with the
                # first element.
                offset_data = np.zeros(num_rows + 1, dtype=np.uint32) + num_rows
                offset_data[0] = 0
                setattr(table, offset_col.name, offset_data)
                list_value = getattr(table[0], list_col.name)
                assert len(list_value) == num_rows

                del input_data[list_col.name]
                del input_data[offset_col.name]

            for col, data in input_data.items():
                assert np.array_equal(getattr(table, col), data)
                data += 1
                assert not np.array_equal(getattr(table, col), data)
                setattr(table, col, data)
                assert np.array_equal(getattr(table, col), data)

    def test_set_column_attributes_errors(self):
        table = self.table_class()
        num_rows = 10
        input_data = self.make_input_data(num_rows)
        table.set_columns(**input_data)

        for list_col, offset_col in self.ragged_list_columns:
            for bad_list_col in [[], input_data[list_col.name][:-1]]:
                with pytest.raises(ValueError):
                    setattr(table, list_col.name, bad_list_col)
            for bad_offset_col in [[], np.arange(num_rows + 2, dtype=np.uint32)]:
                with pytest.raises(ValueError):
                    setattr(table, offset_col.name, bad_offset_col)

            del input_data[list_col.name]
            del input_data[offset_col.name]

        for col, data in input_data.items():
            for bad_data in [[], data[:-1]]:
                with pytest.raises(ValueError):
                    setattr(table, col, bad_data)

        # Try to read a column that isn't there. (We can always write to new attributes
        # in Python, so there's nothing to test in that case.)
        with pytest.raises(AttributeError):
            _ = table.no_such_column

    def test_defaults(self):
        table = self.table_class()
        assert table.num_rows == 0
        assert len(table) == 0
        for param, default in self.input_parameters:
            assert getattr(table, param) == default
        for col in self.columns:
            array = getattr(table, col.name)
            assert array.shape == (0,)

    def test_add_row_data(self):
        for num_rows in [0, 10, 100]:
            input_data = {col.name: col.get_input(num_rows) for col in self.columns}
            table = self.table_class()
            for j in range(num_rows):
                kwargs = {col: data[j] for col, data in input_data.items()}
                for col in self.string_colnames:
                    kwargs[col] = "x"
                for col in self.binary_colnames:
                    kwargs[col] = b"x"
                k = table.add_row(**kwargs)
                assert k == j
            for colname, input_array in input_data.items():
                output_array = getattr(table, colname)
                assert input_array.shape == output_array.shape
                assert np.all(input_array == output_array)
            table.clear()
            assert table.num_rows == 0
            assert len(table) == 0

    def test_add_row_round_trip(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            t1 = self.table_class()
            t1.set_columns(**input_data)
            for colname, input_array in input_data.items():
                output_array = getattr(t1, colname)
                assert input_array.shape == output_array.shape
                assert np.all(input_array == output_array)
            t2 = self.table_class()
            for row in list(t1):
                t2.add_row(**attr.asdict(row))
            assert t1 == t2

    def test_set_columns_data(self):
        for num_rows in [0, 10, 100, 1000]:
            input_data = {col.name: col.get_input(num_rows) for col in self.columns}
            offset_cols = set()
            for list_col, offset_col in self.ragged_list_columns:
                value = list_col.get_input(num_rows)
                input_data[list_col.name] = value
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                offset_cols.add(offset_col.name)
            table = self.table_class()
            for _ in range(5):
                table.set_columns(**input_data)
                for colname, input_array in input_data.items():
                    output_array = getattr(table, colname)
                    assert input_array.shape == output_array.shape
                    assert np.all(input_array == output_array)
                table.clear()
                assert table.num_rows == 0
                assert len(table) == 0
                for colname in input_data.keys():
                    if colname in offset_cols:
                        assert list(getattr(table, colname)) == [0]
                    else:
                        assert list(getattr(table, colname)) == []

    def test_truncate(self):
        num_rows = 100
        input_data = {col.name: col.get_input(num_rows) for col in self.columns}
        for list_col, offset_col in self.ragged_list_columns:
            value = list_col.get_input(2 * num_rows)
            input_data[list_col.name] = value
            input_data[offset_col.name] = 2 * np.arange(num_rows + 1, dtype=np.uint32)
        table = self.table_class()
        table.set_columns(**input_data)

        copy = table.copy()
        table.truncate(num_rows)
        assert copy == table

        for num_rows in [100, 10, 1]:
            table.truncate(num_rows)
            assert table.num_rows == num_rows
            assert len(table) == num_rows
            used = set()
            for list_col, offset_col in self.ragged_list_columns:
                offset = getattr(table, offset_col.name)
                assert offset.shape == (num_rows + 1,)
                assert np.array_equal(
                    input_data[offset_col.name][: num_rows + 1], offset
                )
                list_data = getattr(table, list_col.name)
                assert np.array_equal(
                    list_data, input_data[list_col.name][: offset[-1]]
                )
                used.add(offset_col.name)
                used.add(list_col.name)
            for name, data in input_data.items():
                if name not in used:
                    assert np.array_equal(data[:num_rows], getattr(table, name))

    def test_truncate_errors(self):
        num_rows = 10
        input_data = {col.name: col.get_input(num_rows) for col in self.columns}
        for list_col, offset_col in self.ragged_list_columns:
            value = list_col.get_input(2 * num_rows)
            input_data[list_col.name] = value
            input_data[offset_col.name] = 2 * np.arange(num_rows + 1, dtype=np.uint32)
        table = self.table_class()
        table.set_columns(**input_data)
        for bad_type in [None, 0.001, {}]:
            with pytest.raises(TypeError):
                table.truncate(bad_type)
        for bad_num_rows in [-1, num_rows + 1, 10 ** 6]:
            with pytest.raises(ValueError):
                table.truncate(bad_num_rows)

    def test_append_columns_data(self):
        for num_rows in [0, 10, 100, 1000]:
            input_data = self.make_input_data(num_rows)
            offset_cols = set()
            for _, offset_col in self.ragged_list_columns:
                offset_cols.add(offset_col.name)
            table = self.table_class()
            for j in range(1, 10):
                table.append_columns(**input_data)
                for colname, values in input_data.items():
                    output_array = getattr(table, colname)
                    if colname in offset_cols:
                        input_array = np.zeros(j * num_rows + 1, dtype=np.uint32)
                        for k in range(j):
                            input_array[k * num_rows : (k + 1) * num_rows + 1] = (
                                k * values[-1]
                            ) + values
                        assert input_array.shape == output_array.shape
                    else:
                        input_array = np.hstack([values for _ in range(j)])
                        assert input_array.shape == output_array.shape
                    assert np.array_equal(input_array, output_array)
                assert table.num_rows == j * num_rows
                assert len(table) == j * num_rows

    def test_append_columns_max_rows(self):
        for num_rows in [0, 10, 100, 1000]:
            input_data = self.make_input_data(num_rows)
            for max_rows in [0, 1, 8192]:
                table = self.table_class(max_rows_increment=max_rows)
                for j in range(1, 10):
                    table.append_columns(**input_data)
                    assert table.num_rows == j * num_rows
                    assert len(table) == j * num_rows
                    assert table.max_rows > table.num_rows

    def test_str(self):
        for num_rows in [0, 10]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            s = str(table)
            assert len(s.splitlines()) == num_rows + 1

    def test_repr_html(self):
        for num_rows in [0, 10, 40, 50]:
            input_data = {col.name: col.get_input(num_rows) for col in self.columns}
            for list_col, offset_col in self.ragged_list_columns:
                value = list_col.get_input(num_rows)
                input_data[list_col.name] = value
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
            table = self.table_class()
            table.set_columns(**input_data)
            html = table._repr_html_()
            if num_rows == 50:
                assert len(html.splitlines()) == num_rows + 11
                assert (
                    html.split("</tr>")[21]
                    == "\n<tr><td><em>... skipped 10 rows ...</em></td>"
                )
            else:
                assert len(html.splitlines()) == num_rows + 20

    def test_copy(self):
        for num_rows in [0, 10]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            for _ in range(10):
                copy = table.copy()
                assert id(copy) != id(table)
                assert isinstance(copy, self.table_class)
                assert copy == table
                table = copy

    def test_pickle(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            pkl = pickle.dumps(table)
            new_table = pickle.loads(pkl)
            assert table == new_table
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                pkl = pickle.dumps(table, protocol=protocol)
                new_table = pickle.loads(pkl)
                assert table == new_table

    def test_equality(self):
        for num_rows in [1, 10, 100]:
            input_data = self.make_input_data(num_rows)
            t1 = self.table_class()
            t2 = self.table_class()
            assert t1 == t1
            assert t1 == t2
            assert t1 == t2
            assert not (t1 != t2)
            t1.set_columns(**input_data)
            assert t1 == t1
            assert t1 != t2
            assert t2 != t1
            t2.set_columns(**input_data)
            assert t1 == t2
            assert t2 == t2
            t2.clear()
            assert t1 != t2
            assert t2 != t1
            # Check each column in turn to see if we are correctly checking values.
            for col in self.columns:
                col_copy = np.copy(input_data[col.name])
                input_data_copy = dict(input_data)
                input_data_copy[col.name] = col_copy
                t2.set_columns(**input_data_copy)
                assert t1 == t2
                assert not (t1 != t2)
                assert t1[0] == t2[0]
                col_copy += 1
                t2.set_columns(**input_data_copy)
                assert t1 != t2
                assert t2 != t1
                assert t1[0] != t2[0]
                assert t1[0] != t2[0]
                assert t1[0] != []
            for list_col, offset_col in self.ragged_list_columns:
                value = list_col.get_input(num_rows)
                input_data_copy = dict(input_data)
                input_data_copy[list_col.name] = value + 1
                t2.set_columns(**input_data_copy)
                assert t1 != t2
                assert t1[0] != t2[0]
                value = list_col.get_input(num_rows + 1)
                input_data_copy = dict(input_data)
                input_data_copy[list_col.name] = value
                input_data_copy[offset_col.name] = np.arange(
                    num_rows + 1, dtype=np.uint32
                )
                input_data_copy[offset_col.name][-1] = num_rows + 1
                t2.set_columns(**input_data_copy)
                assert t1 != t2
                assert t2 != t1
                assert t1[-1] != t2[-1]
            # Different types should always be unequal.
            assert t1 is not None
            assert t1 != []

    def test_nbytes(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            # We don't have any metadata_schema here, so we can sum over the
            # columns directly.
            assert sum(col.nbytes for col in input_data.values()) == table.nbytes

    def test_bad_offsets(self):
        for num_rows in [10, 100]:
            input_data = self.make_input_data(num_rows)
            t = self.table_class()
            t.set_columns(**input_data)

            for _list_col, offset_col in self.ragged_list_columns:
                input_data[offset_col.name][0] = -1
                with pytest.raises(ValueError):
                    t.set_columns(**input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                t.set_columns(**input_data)
                input_data[offset_col.name][-1] = 0
                with pytest.raises(ValueError):
                    t.set_columns(**input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                t.set_columns(**input_data)
                input_data[offset_col.name][num_rows // 2] = 2 ** 31
                with pytest.raises(ValueError):
                    t.set_columns(**input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)

                input_data[offset_col.name][0] = -1
                with pytest.raises(ValueError):
                    t.append_columns(**input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                t.append_columns(**input_data)
                input_data[offset_col.name][-1] = 0
                with pytest.raises(ValueError):
                    t.append_columns(**input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                t.append_columns(**input_data)
                input_data[offset_col.name][num_rows // 2] = 2 ** 31
                with pytest.raises(ValueError):
                    t.append_columns(**input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)


class MetadataTestsMixin:
    """
    Tests for column that have metadata columns.
    """

    metadata_schema = metadata.MetadataSchema(
        {
            "codec": "json",
            "title": "Example Metadata",
            "type": "object",
            "properties": {
                "one": {"type": "string"},
                "two": {"type": "number"},
                "three": {"type": "array"},
                "four": {"type": "boolean"},
            },
            "required": ["one", "two", "three", "four"],
            "additionalProperties": False,
        },
    )

    def metadata_example_data(self):
        try:
            self.val += 1
        except AttributeError:
            self.val = 0
        return {
            "one": "val one",
            "two": self.val,
            "three": list(range(self.val, self.val + 10)),
            "four": True,
        }

    def input_data_for_add_row(self):
        input_data = {col.name: col.get_input(1) for col in self.columns}
        kwargs = {col: data[0] for col, data in input_data.items()}
        for col in self.string_colnames:
            kwargs[col] = "x"
        for col in self.binary_colnames:
            kwargs[col] = b"x"
        return kwargs

    def test_random_metadata(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            metadatas = [tsutil.random_bytes(10) for _ in range(num_rows)]
            metadata, metadata_offset = tskit.pack_bytes(metadatas)
            input_data["metadata"] = metadata
            input_data["metadata_offset"] = metadata_offset
            table.set_columns(**input_data)
            unpacked_metadatas = tskit.unpack_bytes(
                table.metadata, table.metadata_offset
            )
            assert metadatas == unpacked_metadatas

    def test_optional_metadata(self):
        if not getattr(self, "metadata_mandatory", False):
            for num_rows in [0, 10, 100]:
                input_data = self.make_input_data(num_rows)
                table = self.table_class()
                del input_data["metadata"]
                del input_data["metadata_offset"]
                table.set_columns(**input_data)
                assert len(list(table.metadata)) == 0
                assert list(table.metadata_offset) == [0 for _ in range(num_rows + 1)]
                # Supplying None is the same not providing the column.
                input_data["metadata"] = None
                input_data["metadata_offset"] = None
                table.set_columns(**input_data)
                assert len(list(table.metadata)) == 0
                assert list(table.metadata_offset) == [0 for _ in range(num_rows + 1)]

    def test_packset_metadata(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            metadatas = [tsutil.random_bytes(10) for _ in range(num_rows)]
            metadata, metadata_offset = tskit.pack_bytes(metadatas)
            table.packset_metadata(metadatas)
            assert np.array_equal(table.metadata, metadata)
            assert np.array_equal(table.metadata_offset, metadata_offset)

    def test_set_metadata_schema(self):
        metadata_schema2 = metadata.MetadataSchema({"codec": "json"})
        table = self.table_class()
        # Default is no-op metadata codec
        assert repr(table.metadata_schema) == repr(metadata.MetadataSchema(None))
        # Set
        table.metadata_schema = self.metadata_schema
        assert repr(table.metadata_schema) == repr(self.metadata_schema)
        # Overwrite
        table.metadata_schema = metadata_schema2
        assert repr(table.metadata_schema) == repr(metadata_schema2)
        # Remove
        table.metadata_schema = metadata.MetadataSchema(None)
        assert repr(table.metadata_schema) == repr(metadata.MetadataSchema(None))
        # Set after remove
        table.metadata_schema = self.metadata_schema
        assert repr(table.metadata_schema) == repr(self.metadata_schema)
        # Del should fail
        with pytest.raises(AttributeError, match="can't delete attribute"):
            del table.metadata_schema
        # None should fail
        with pytest.raises(
            TypeError,
            match="Only instances of tskit.MetadataSchema can be assigned to "
            "metadata_schema, not <class 'NoneType'>",
        ):
            table.metadata_schema = None
        # And dict
        with pytest.raises(
            TypeError,
            match="Only instances of tskit.MetadataSchema can be assigned to "
            "metadata_schema, not <class 'dict'>",
        ):
            table.metadata_schema = {}

    def test_default_metadata_schema(self):
        # Default should allow bytes as in pre-exisiting code
        table = self.table_class()
        table.add_row(
            **{**self.input_data_for_add_row(), "metadata": b"acceptable bytes"}
        )
        # Adding non-bytes metadata should error
        with pytest.raises(TypeError):
            table.add_row(
                **{
                    **self.input_data_for_add_row(),
                    "metadata": self.metadata_example_data(),
                }
            )

    def test_default_metadata_add_row(self):
        row_data = self.input_data_for_add_row()
        del row_data["metadata"]

        table = self.table_class()
        table.add_row(**row_data)
        assert table[0].metadata == b""
        assert table[0].metadata == table.metadata_schema.empty_value

        table = self.table_class()
        table.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        table.add_row(**row_data)
        assert table[0].metadata == {}
        assert table[0].metadata == table.metadata_schema.empty_value

    def test_row_round_trip_metadata_schema(self):
        data = self.metadata_example_data()
        table = self.table_class()
        table.metadata_schema = self.metadata_schema
        table.add_row(**{**self.input_data_for_add_row(), "metadata": data})
        assert table[0].metadata == data

    def test_bad_row_metadata_schema(self):
        metadata = self.metadata_example_data()
        metadata["I really shouldn't be here"] = 6
        table = self.table_class()
        table.metadata_schema = self.metadata_schema
        with pytest.raises(exceptions.MetadataValidationError):
            table.add_row(**{**self.input_data_for_add_row(), "metadata": metadata})
        assert len(table) == 0

    def test_absent_metadata_with_required_schema(self):
        table = self.table_class()
        table.metadata_schema = self.metadata_schema
        input_data = self.input_data_for_add_row()
        del input_data["metadata"]
        with pytest.raises(exceptions.MetadataValidationError):
            table.add_row(**{**input_data})

    def test_unsupported_type(self):
        table = self.table_class()
        table.metadata_schema = metadata.MetadataSchema(
            {
                "codec": "json",
                "type": "object",
                "properties": {"an_array": {"type": "array"}},
            }
        )
        input_data = self.input_data_for_add_row()
        # Numpy is not a JSONSchema array
        input_data["metadata"] = {"an_array": np.arange(10)}
        with pytest.raises(exceptions.MetadataValidationError):
            table.add_row(**{**input_data})

    def test_round_trip_set_columns(self):
        for num_rows in [0, 10, 100]:
            table = self.table_class()
            table.metadata_schema = self.metadata_schema
            input_data = self.make_input_data(num_rows)
            del input_data["metadata"]
            del input_data["metadata_offset"]
            metadata_column = [self.metadata_example_data() for _ in range(num_rows)]
            encoded_metadata_column = [
                table.metadata_schema.validate_and_encode_row(r)
                for r in metadata_column
            ]
            packed_metadata, metadata_offset = tskit.util.pack_bytes(
                encoded_metadata_column
            )
            table.set_columns(
                metadata=packed_metadata, metadata_offset=metadata_offset, **input_data
            )
            table.append_columns(
                metadata=packed_metadata, metadata_offset=metadata_offset, **input_data
            )
            for j in range(num_rows):
                assert table[j].metadata == metadata_column[j]
                assert table[j + num_rows].metadata == metadata_column[j]


class TestIndividualTable(CommonTestsMixin, MetadataTestsMixin):
    columns = [UInt32Column("flags")]
    ragged_list_columns = [
        (DoubleColumn("location"), UInt32Column("location_offset")),
        (Int32Column("parents"), UInt32Column("parents_offset")),
        (CharColumn("metadata"), UInt32Column("metadata_offset")),
    ]
    string_colnames = []
    binary_colnames = ["metadata"]
    input_parameters = [("max_rows_increment", 1024)]
    equal_len_columns = [["flags"]]
    table_class = tskit.IndividualTable

    def test_simple_example(self):
        t = tskit.IndividualTable()
        t.add_row(flags=0, location=[], parents=[], metadata=b"123")
        t.add_row(
            flags=1, location=(0, 1, 2, 3), parents=(4, 5, 6, 7), metadata=b"\xf0"
        )
        s = str(t)
        assert len(s) > 0
        assert len(t) == 2
        assert t[0].flags == 0
        assert list(t[0].location) == []
        assert list(t[0].parents) == []
        assert t[0].metadata == b"123"
        assert t[1].flags == 1
        assert list(t[1].location) == [0, 1, 2, 3]
        assert list(t[1].parents) == [4, 5, 6, 7]
        assert t[1].metadata == b"\xf0"
        with pytest.raises(IndexError):
            t.__getitem__(-4)

    def test_add_row_defaults(self):
        t = tskit.IndividualTable()
        assert t.add_row() == 0
        assert t.flags[0] == 0
        assert len(t.location) == 0
        assert t.location_offset[0] == 0
        assert len(t.parents) == 0
        assert t.parents_offset[0] == 0
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

    def test_add_row_bad_data(self):
        t = tskit.IndividualTable()
        with pytest.raises(TypeError):
            t.add_row(flags="x")
        with pytest.raises(TypeError):
            t.add_row(metadata=123)
        with pytest.raises(ValueError):
            t.add_row(location="1234")
        with pytest.raises(ValueError):
            t.add_row(parents="forty-two")

    def test_packset_location(self):
        t = tskit.IndividualTable()
        t.add_row(flags=0)
        t.packset_location([[0.125, 2]])
        assert list(t[0].location) == [0.125, 2]
        t.add_row(flags=1)
        assert list(t[1].location) == []
        t.packset_location([[0], [1, 2, 3]])
        assert list(t[0].location) == [0]
        assert list(t[1].location) == [1, 2, 3]

    def test_packset_parents(self):
        t = tskit.IndividualTable()
        t.add_row(flags=0)
        t.packset_parents([[0, 2]])
        assert list(t[0].parents) == [0, 2]
        t.add_row(flags=1)
        assert list(t[1].parents) == []
        t.packset_parents([[0], [1, 2, 3]])
        assert list(t[0].parents) == [0]
        assert list(t[1].parents) == [1, 2, 3]

    def test_missing_time_equal_to_self(self):
        t = tskit.TableCollection(sequence_length=10)
        t.sites.add_row(position=1, ancestral_state="0")
        t.mutations.add_row(site=0, node=0, derived_state="1", time=tskit.UNKNOWN_TIME)
        assert t.mutations[0] == t.mutations[0]

    def test_various_not_equals(self):
        args = {
            "site": 0,
            "node": 0,
            "derived_state": "a",
            "parent": 0,
            "metadata": b"abc",
            "time": 0,
        }
        a = tskit.MutationTableRow(**args)
        assert a != []
        assert a != 12
        assert a is not None
        b = tskit.MutationTableRow(**args)
        assert a == b
        args["site"] = 2
        b = tskit.MutationTableRow(**args)
        assert a != b
        args["site"] = 0
        args["node"] = 2
        b = tskit.MutationTableRow(**args)
        assert a != b
        args["node"] = 0
        args["derived_state"] = "b"
        b = tskit.MutationTableRow(**args)
        assert a != b
        args["derived_state"] = "a"
        args["parent"] = 2
        b = tskit.MutationTableRow(**args)
        assert a != b
        args["parent"] = 0
        args["metadata"] = b""
        b = tskit.MutationTableRow(**args)
        assert a != b
        args["metadata"] = b"abc"
        args["time"] = 1
        b = tskit.MutationTableRow(**args)
        assert a != b
        args["time"] = 0
        args["time"] = tskit.UNKNOWN_TIME
        b = tskit.MutationTableRow(**args)
        assert a != b
        a = tskit.MutationTableRow(**args)
        assert a == b


class TestNodeTable(CommonTestsMixin, MetadataTestsMixin):

    columns = [
        UInt32Column("flags"),
        DoubleColumn("time"),
        Int32Column("individual"),
        Int32Column("population"),
    ]
    ragged_list_columns = [(CharColumn("metadata"), UInt32Column("metadata_offset"))]
    string_colnames = []
    binary_colnames = ["metadata"]
    input_parameters = [("max_rows_increment", 1024)]
    equal_len_columns = [["time", "flags", "population"]]
    table_class = tskit.NodeTable

    def test_simple_example(self):
        t = tskit.NodeTable()
        t.add_row(flags=0, time=1, population=2, individual=0, metadata=b"123")
        t.add_row(flags=1, time=2, population=3, individual=1, metadata=b"\xf0")
        s = str(t)
        assert len(s) > 0
        assert len(t) == 2
        assert attr.astuple(t[0]) == (0, 1, 2, 0, b"123")
        assert attr.astuple(t[1]) == (1, 2, 3, 1, b"\xf0")
        assert t[0].flags == 0
        assert t[0].time == 1
        assert t[0].population == 2
        assert t[0].individual == 0
        assert t[0].metadata == b"123"
        assert t[0] == t[-2]
        assert t[1] == t[-1]
        with pytest.raises(IndexError):
            t.__getitem__(-3)

    def test_add_row_defaults(self):
        t = tskit.NodeTable()
        assert t.add_row() == 0
        assert t.time[0] == 0
        assert t.flags[0] == 0
        assert t.population[0] == tskit.NULL
        assert t.individual[0] == tskit.NULL
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

    def test_optional_population(self):
        for num_rows in [0, 10, 100]:
            metadatas = [str(j) for j in range(num_rows)]
            metadata, metadata_offset = tskit.pack_strings(metadatas)
            flags = list(range(num_rows))
            time = list(range(num_rows))
            table = tskit.NodeTable()
            table.set_columns(
                metadata=metadata,
                metadata_offset=metadata_offset,
                flags=flags,
                time=time,
            )
            assert list(table.population) == [-1 for _ in range(num_rows)]
            assert list(table.flags) == flags
            assert list(table.time) == time
            assert list(table.metadata) == list(metadata)
            assert list(table.metadata_offset) == list(metadata_offset)
            table.set_columns(flags=flags, time=time, population=None)
            assert list(table.population) == [-1 for _ in range(num_rows)]
            assert list(table.flags) == flags
            assert list(table.time) == time

    def test_add_row_bad_data(self):
        t = tskit.NodeTable()
        with pytest.raises(TypeError):
            t.add_row(flags="x")
        with pytest.raises(TypeError):
            t.add_row(time="x")
        with pytest.raises(TypeError):
            t.add_row(individual="x")
        with pytest.raises(TypeError):
            t.add_row(population="x")
        with pytest.raises(TypeError):
            t.add_row(metadata=123)


class TestEdgeTable(CommonTestsMixin, MetadataTestsMixin):

    columns = [
        DoubleColumn("left"),
        DoubleColumn("right"),
        Int32Column("parent"),
        Int32Column("child"),
    ]
    equal_len_columns = [["left", "right", "parent", "child"]]
    string_colnames = []
    binary_colnames = ["metadata"]
    ragged_list_columns = [(CharColumn("metadata"), UInt32Column("metadata_offset"))]
    input_parameters = [("max_rows_increment", 1024)]
    table_class = tskit.EdgeTable

    def test_simple_example(self):
        t = tskit.EdgeTable()
        t.add_row(left=0, right=1, parent=2, child=3, metadata=b"123")
        t.add_row(1, 2, 3, 4, b"\xf0")
        assert len(t) == 2
        assert attr.astuple(t[0]) == (0, 1, 2, 3, b"123")
        assert attr.astuple(t[1]) == (1, 2, 3, 4, b"\xf0")
        assert t[0].left == 0
        assert t[0].right == 1
        assert t[0].parent == 2
        assert t[0].child == 3
        assert t[0].metadata == b"123"
        assert t[0] == t[-2]
        assert t[1] == t[-1]
        with pytest.raises(IndexError):
            t.__getitem__(-3)

    def test_add_row_defaults(self):
        t = tskit.EdgeTable()
        assert t.add_row(0, 0, 0, 0) == 0
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

    def test_add_row_bad_data(self):
        t = tskit.EdgeTable()
        with pytest.raises(TypeError):
            t.add_row(left="x", right=0, parent=0, child=0)
        with pytest.raises(TypeError):
            t.add_row()
        with pytest.raises(TypeError):
            t.add_row(0, 0, 0, 0, metadata=123)


class TestSiteTable(CommonTestsMixin, MetadataTestsMixin):
    columns = [DoubleColumn("position")]
    ragged_list_columns = [
        (CharColumn("ancestral_state"), UInt32Column("ancestral_state_offset")),
        (CharColumn("metadata"), UInt32Column("metadata_offset")),
    ]
    equal_len_columns = [["position"]]
    string_colnames = ["ancestral_state"]
    binary_colnames = ["metadata"]
    input_parameters = [("max_rows_increment", 1024)]
    table_class = tskit.SiteTable

    def test_simple_example(self):
        t = tskit.SiteTable()
        t.add_row(position=0, ancestral_state="1", metadata=b"2")
        t.add_row(1, "2", b"\xf0")
        s = str(t)
        assert len(s) > 0
        assert len(t) == 2
        assert attr.astuple(t[0]) == (0, "1", b"2")
        assert attr.astuple(t[1]) == (1, "2", b"\xf0")
        assert t[0].position == 0
        assert t[0].ancestral_state == "1"
        assert t[0].metadata == b"2"
        assert t[0] == t[-2]
        assert t[1] == t[-1]
        with pytest.raises(IndexError):
            t.__getitem__(2)
        with pytest.raises(IndexError):
            t.__getitem__(-3)

    def test_add_row_bad_data(self):
        t = tskit.SiteTable()
        t.add_row(0, "A")
        with pytest.raises(TypeError):
            t.add_row("x", "A")
        with pytest.raises(TypeError):
            t.add_row(0, 0)
        with pytest.raises(TypeError):
            t.add_row(0, "A", metadata=[0, 1, 2])

    def test_packset_ancestral_state(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            ancestral_states = [tsutil.random_strings(10) for _ in range(num_rows)]
            ancestral_state, ancestral_state_offset = tskit.pack_strings(
                ancestral_states
            )
            table.packset_ancestral_state(ancestral_states)
            assert np.array_equal(table.ancestral_state, ancestral_state)
            assert np.array_equal(table.ancestral_state_offset, ancestral_state_offset)


class TestMutationTable(CommonTestsMixin, MetadataTestsMixin):
    columns = [
        Int32Column("site"),
        Int32Column("node"),
        DoubleColumn("time"),
        Int32Column("parent"),
    ]
    ragged_list_columns = [
        (CharColumn("derived_state"), UInt32Column("derived_state_offset")),
        (CharColumn("metadata"), UInt32Column("metadata_offset")),
    ]
    equal_len_columns = [["site", "node", "time"]]
    string_colnames = ["derived_state"]
    binary_colnames = ["metadata"]
    input_parameters = [("max_rows_increment", 1024)]
    table_class = tskit.MutationTable

    def test_simple_example(self):
        t = tskit.MutationTable()
        t.add_row(site=0, node=1, derived_state="2", parent=3, metadata=b"4", time=5)
        t.add_row(1, 2, "3", 4, b"\xf0", 6)
        t.add_row(
            site=0,
            node=1,
            derived_state="2",
            parent=3,
            metadata=b"4",
            time=tskit.UNKNOWN_TIME,
        )
        s = str(t)
        assert len(s) > 0
        assert len(t) == 3
        assert attr.astuple(t[0]) == (0, 1, "2", 3, b"4", 5)
        assert attr.astuple(t[1]) == (1, 2, "3", 4, b"\xf0", 6)
        assert t[0].site == 0
        assert t[0].node == 1
        assert t[0].derived_state == "2"
        assert t[0].parent == 3
        assert t[0].metadata == b"4"
        assert t[0].time == 5
        assert t[0] == t[-3]
        assert t[1] == t[-2]
        assert t[2] == t[-1]
        with pytest.raises(IndexError):
            t.__getitem__(-4)

    def test_add_row_bad_data(self):
        t = tskit.MutationTable()
        t.add_row(0, 0, "A")
        with pytest.raises(TypeError):
            t.add_row("0", 0, "A")
        with pytest.raises(TypeError):
            t.add_row(0, "0", "A")
        with pytest.raises(TypeError):
            t.add_row(0, 0, "A", parent=None)
        with pytest.raises(TypeError):
            t.add_row(0, 0, "A", metadata=[0])
        with pytest.raises(TypeError):
            t.add_row(0, 0, "A", time="A")

    def test_packset_derived_state(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            derived_states = [tsutil.random_strings(10) for _ in range(num_rows)]
            derived_state, derived_state_offset = tskit.pack_strings(derived_states)
            table.packset_derived_state(derived_states)
            assert np.array_equal(table.derived_state, derived_state)
            assert np.array_equal(table.derived_state_offset, derived_state_offset)


class TestMigrationTable(CommonTestsMixin, MetadataTestsMixin):
    columns = [
        DoubleColumn("left"),
        DoubleColumn("right"),
        Int32Column("node"),
        Int32Column("source"),
        Int32Column("dest"),
        DoubleColumn("time"),
    ]
    ragged_list_columns = [(CharColumn("metadata"), UInt32Column("metadata_offset"))]
    string_colnames = []
    binary_colnames = ["metadata"]
    input_parameters = [("max_rows_increment", 1024)]
    equal_len_columns = [["left", "right", "node", "source", "dest", "time"]]
    table_class = tskit.MigrationTable

    def test_simple_example(self):
        t = tskit.MigrationTable()
        t.add_row(left=0, right=1, node=2, source=3, dest=4, time=5, metadata=b"123")
        t.add_row(1, 2, 3, 4, 5, 6, b"\xf0")
        assert len(t) == 2
        assert attr.astuple(t[0]) == (0, 1, 2, 3, 4, 5, b"123")
        assert attr.astuple(t[1]) == (1, 2, 3, 4, 5, 6, b"\xf0")
        assert t[0].left == 0
        assert t[0].right == 1
        assert t[0].node == 2
        assert t[0].source == 3
        assert t[0].dest == 4
        assert t[0].time == 5
        assert t[0].metadata == b"123"
        assert t[0] == t[-2]
        assert t[1] == t[-1]
        with pytest.raises(IndexError):
            t.__getitem__(-3)

    def test_add_row_defaults(self):
        t = tskit.MigrationTable()
        assert t.add_row(0, 0, 0, 0, 0, 0) == 0
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

    def test_add_row_bad_data(self):
        t = tskit.MigrationTable()
        with pytest.raises(TypeError):
            t.add_row(left="x", right=0, node=0, source=0, dest=0, time=0)
        with pytest.raises(TypeError):
            t.add_row()
        with pytest.raises(TypeError):
            t.add_row(0, 0, 0, 0, 0, 0, metadata=123)


class TestProvenanceTable(CommonTestsMixin):
    columns = []
    ragged_list_columns = [
        (CharColumn("timestamp"), UInt32Column("timestamp_offset")),
        (CharColumn("record"), UInt32Column("record_offset")),
    ]
    equal_len_columns = [[]]
    string_colnames = ["record", "timestamp"]
    binary_colnames = []
    input_parameters = [("max_rows_increment", 1024)]
    table_class = tskit.ProvenanceTable

    def test_simple_example(self):
        t = tskit.ProvenanceTable()
        t.add_row(timestamp="0", record="1")
        t.add_row("2", "1")  # The orders are reversed for default timestamp.
        assert len(t) == 2
        assert attr.astuple(t[0]) == ("0", "1")
        assert attr.astuple(t[1]) == ("1", "2")
        assert t[0].timestamp == "0"
        assert t[0].record == "1"
        assert t[0] == t[-2]
        assert t[1] == t[-1]
        with pytest.raises(IndexError):
            t.__getitem__(-3)

    def test_add_row_bad_data(self):
        t = tskit.ProvenanceTable()
        t.add_row("a", "b")
        with pytest.raises(TypeError):
            t.add_row(0, "b")
        with pytest.raises(TypeError):
            t.add_row("a", 0)

    def test_packset_timestamp(self):
        t = tskit.ProvenanceTable()
        t.add_row(timestamp="0", record="1")
        t.add_row(timestamp="1", record="2")
        t.packset_timestamp(["AAAA", "BBBB"])
        assert t[0].timestamp == "AAAA"
        assert t[1].timestamp == "BBBB"

    def test_packset_record(self):
        t = tskit.ProvenanceTable()
        t.add_row(timestamp="0", record="1")
        t.add_row(timestamp="1", record="2")
        t.packset_record(["AAAA", "BBBB"])
        assert t[0].record == "AAAA"
        assert t[1].record == "BBBB"


class TestPopulationTable(CommonTestsMixin, MetadataTestsMixin):
    metadata_mandatory = True
    columns = []
    ragged_list_columns = [(CharColumn("metadata"), UInt32Column("metadata_offset"))]
    equal_len_columns = [[]]
    string_colnames = []
    binary_colnames = ["metadata"]
    input_parameters = [("max_rows_increment", 1024)]
    table_class = tskit.PopulationTable

    def test_simple_example(self):
        t = tskit.PopulationTable()
        t.add_row(metadata=b"\xf0")
        t.add_row(b"1")
        s = str(t)
        assert len(s) > 0
        assert len(t) == 2
        assert attr.astuple(t[0]) == (b"\xf0",)
        assert t[0].metadata == b"\xf0"
        assert attr.astuple(t[1]) == (b"1",)
        with pytest.raises(IndexError):
            t.__getitem__(-3)

    def test_add_row_bad_data(self):
        t = tskit.PopulationTable()
        t.add_row()
        with pytest.raises(TypeError):
            t.add_row(metadata=[0])


class TestTableCollectionIndexes:
    def test_index(self):
        i = np.arange(20)
        r = np.arange(20)[::-1]
        index = tskit.TableCollectionIndexes(
            edge_insertion_order=i, edge_removal_order=r
        )
        assert np.array_equal(index.edge_insertion_order, i)
        assert np.array_equal(index.edge_removal_order, r)
        d = index.asdict()
        assert np.array_equal(d["edge_insertion_order"], i)
        assert np.array_equal(d["edge_removal_order"], r)

        index = tskit.TableCollectionIndexes()
        assert index.edge_insertion_order is None
        assert index.edge_removal_order is None
        assert index.asdict() == {}


class TestSortTables:
    """
    Tests for the TableCollection.sort() and TableCollection.canonicalise() methods.
    """

    random_seed = 12345

    def verify_sort_equality(self, tables, seed):
        tables1 = tables.copy()
        tsutil.shuffle_tables(
            tables1,
            seed,
            shuffle_populations=False,
        )
        tables1.individuals.packset_metadata(
            [bytes(str(i), "utf-8") for i in range(tables1.individuals.num_rows)]
        )
        tables2 = tables1.copy()
        tables1.sort()
        tsutil.py_sort(tables2)

        # TODO - Check the sorted tables are valid ts, currently fails due to mutations
        # tables1.tree_sequence()
        # tables2.tree_sequence()

        tsutil.assert_table_collections_equal(tables1, tables2)

    def verify_canonical_equality(self, tables, seed):
        # Migrations not supported
        tables.migrations.clear()

        for ru in [True, False]:
            tables1 = tables.copy()
            tsutil.shuffle_tables(
                tables1,
                seed,
            )
            tables2 = tables1.copy()
            tables1.canonicalise(remove_unreferenced=ru)
            tsutil.py_canonicalise(tables2, remove_unreferenced=ru)
            tsutil.assert_table_collections_equal(tables1, tables2)

    def verify_sort_mutation_consistency(self, orig_tables, seed):
        tables = orig_tables.copy()
        mut_map = {s.position: [] for s in tables.sites}
        for mut in tables.mutations:
            mut_map[tables.sites[mut.site].position].append(
                (mut.node, mut.derived_state, mut.metadata)
            )
        tsutil.shuffle_tables(tables, seed, shuffle_populations=False)
        for mut in tables.mutations:
            site = tables.sites[mut.site]
            assert (mut.node, mut.derived_state, mut.metadata) in mut_map[site.position]
        tables.sort()
        for mut in tables.mutations:
            site = tables.sites[mut.site]
            assert (mut.node, mut.derived_state, mut.metadata) in mut_map[site.position]

    def verify_randomise_tables(self, orig_tables, seed):
        # Check we can shuffle everything and then put it back in canonical form
        tables = orig_tables.copy()
        tables.sort()
        sorted_tables = tables.copy()

        # First randomize only edges: this should work without canonical sorting.
        tsutil.shuffle_tables(
            tables,
            seed=seed,
            shuffle_edges=True,
            shuffle_populations=False,
            shuffle_individuals=False,
            shuffle_sites=False,
            shuffle_mutations=False,
        )
        tables.sort()
        tsutil.assert_table_collections_equal(tables, sorted_tables)

        # Now also randomize sites, mutations and individuals
        tables.canonicalise(remove_unreferenced=False)
        sorted_tables = tables.copy()
        tsutil.shuffle_tables(
            tables,
            seed=1234,
            shuffle_populations=False,
        )
        tables.canonicalise(remove_unreferenced=False)
        tsutil.assert_table_collections_equal(tables, sorted_tables)

        # Finally, randomize everything else
        tsutil.shuffle_tables(tables, seed=1234)
        tables.canonicalise(remove_unreferenced=False)
        tsutil.assert_table_collections_equal(tables, sorted_tables)

        # Check the canonicalised form meets the tree sequence requirements
        tables.tree_sequence()

    def verify_sort(self, tables, seed):
        self.verify_sort_equality(tables, seed)
        self.verify_canonical_equality(tables, seed)
        self.verify_sort_mutation_consistency(tables, seed)
        self.verify_randomise_tables(tables, seed)

    def verify_edge_sort_offset(self, ts):
        """
        Verifies the behaviour of the edge_start offset value.
        """
        tables = ts.dump_tables()
        edges = tables.edges.copy()
        starts = [0]
        if len(edges) > 2:
            starts = [0, 1, len(edges) // 2, len(edges) - 2]
        random.seed(self.random_seed)
        for start in starts:
            # Unsort the edges starting from index start
            all_edges = list(ts.edges())
            keep = all_edges[:start]
            reversed_edges = all_edges[start:][::-1]
            all_edges = keep + reversed_edges
            tables.edges.clear()
            for e in all_edges:
                tables.edges.add_row(e.left, e.right, e.parent, e.child)
            # Verify that import fails for randomised edges
            with pytest.raises(_tskit.LibraryError):
                tables.tree_sequence()
            # If we sort after the start value we should still fail.
            tables.sort(edge_start=start + 1)
            with pytest.raises(_tskit.LibraryError):
                tables.tree_sequence()
            # Sorting from the correct index should give us back the original table.
            tables.edges.clear()
            for e in all_edges:
                tables.edges.add_row(e.left, e.right, e.parent, e.child)
            tables.sort(edge_start=start)
            # Verify the new and old edges are equal.
            assert edges == tables.edges

    def get_wf_example(self, seed):
        tables = wf.wf_sim(
            6,
            3,
            num_pops=2,
            seed=seed,
            num_loci=3,
            record_migrations=True,
            record_individuals=True,
        )
        tables.sort()
        ts = tables.tree_sequence()
        return ts

    def test_wf_example(self):
        tables = wf.wf_sim(
            N=6,
            ngens=3,
            num_pops=2,
            mig_rate=1.0,
            deep_history=False,
            seed=42,
            record_migrations=True,
            record_individuals=True,
        )
        self.verify_sort(tables, 42)

    def test_single_tree_no_mutations(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        self.verify_edge_sort_offset(ts)
        self.verify_sort(ts.tables, 432)

    def test_single_tree_no_mutations_metadata(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts = tsutil.add_random_metadata(ts, self.random_seed)
        self.verify_sort(ts.tables, 12)

    def test_many_trees_no_mutations(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=self.random_seed)
        assert ts.num_trees > 2
        self.verify_edge_sort_offset(ts)
        self.verify_sort(ts.tables, 31)

    def test_single_tree_mutations(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=self.random_seed)
        assert ts.num_sites > 2
        self.verify_edge_sort_offset(ts)
        self.verify_sort(ts.tables, 83)

    def test_single_tree_mutations_metadata(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=self.random_seed)
        assert ts.num_sites > 2
        ts = tsutil.add_random_metadata(ts, self.random_seed)
        self.verify_sort(ts.tables, 384)

    def test_single_tree_multichar_mutations(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts = tsutil.insert_multichar_mutations(ts, self.random_seed)
        self.verify_sort(ts.tables, 185)

    def test_single_tree_multichar_mutations_metadata(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts = tsutil.insert_multichar_mutations(ts, self.random_seed)
        ts = tsutil.add_random_metadata(ts, self.random_seed)
        self.verify_sort(ts.tables, 2175)

    def test_many_trees_mutations(self):
        ts = msprime.simulate(
            10, recombination_rate=2, mutation_rate=2, random_seed=self.random_seed
        )
        assert ts.num_trees > 2
        assert ts.num_sites > 2
        self.verify_edge_sort_offset(ts)
        self.verify_sort(ts.tables, 173)

    def test_many_trees_multichar_mutations(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=self.random_seed)
        assert ts.num_trees > 2
        ts = tsutil.insert_multichar_mutations(ts, self.random_seed)
        self.verify_sort(ts.tables, 16)

    def test_many_trees_multichar_mutations_metadata(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=self.random_seed)
        assert ts.num_trees > 2
        ts = tsutil.insert_multichar_mutations(ts, self.random_seed)
        ts = tsutil.add_random_metadata(ts, self.random_seed)
        self.verify_sort(ts.tables, 91)

    def get_nonbinary_example(self, mutation_rate):
        ts = msprime.simulate(
            sample_size=20,
            recombination_rate=10,
            random_seed=self.random_seed,
            mutation_rate=mutation_rate,
            demographic_events=[
                msprime.SimpleBottleneck(time=0.5, population=0, proportion=1)
            ],
        )
        # Make sure this really has some non-binary nodes
        found = False
        for e in ts.edgesets():
            if len(e.children) > 2:
                found = True
                break
        assert found
        assert ts.num_trees > 2
        return ts

    def test_nonbinary_trees(self):
        ts = self.get_nonbinary_example(mutation_rate=0)
        self.verify_edge_sort_offset(ts)
        self.verify_sort(ts.tables, 9182)

    def test_nonbinary_trees_mutations(self):
        ts = self.get_nonbinary_example(mutation_rate=2)
        assert ts.num_trees > 2
        assert ts.num_sites > 2
        self.verify_edge_sort_offset(ts)
        self.verify_sort(ts.tables, 44)

    def test_unknown_times(self):
        ts = self.get_wf_example(seed=486)
        ts = tsutil.insert_branch_mutations(ts, mutations_per_branch=2)
        ts = tsutil.remove_mutation_times(ts)
        self.verify_sort(ts.tables, 9182)

    def test_no_mutation_parents(self):
        # we should maintain relative order of mutations when all else fails:
        # these tables are not canonicalizable (since we don't sort on derived state)
        rng = random.Random(7000)
        alleles = ["A", "B", "C", "D", "E", "F", "G"]
        for t in [0.5, None]:
            rng.shuffle(alleles)
            tables = tskit.TableCollection(sequence_length=1)
            tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
            tables.sites.add_row(position=0, ancestral_state="")
            for a in alleles:
                tables.mutations.add_row(site=0, node=0, derived_state=a, time=t)
            tables_canonical = tables.copy()
            tables_canonical.canonicalise()
            tables.sort()
            for t in (tables, tables_canonical):
                for a, mut in zip(alleles, t.mutations):
                    assert a == mut.derived_state
                self.verify_sort_equality(t, 985)
                self.verify_sort_mutation_consistency(t, 985)

    def test_discrete_times(self):
        ts = self.get_wf_example(seed=623)
        ts = tsutil.insert_discrete_time_mutations(ts)
        self.verify_sort(ts.tables, 9183)

    def test_incompatible_edges(self):
        ts1 = msprime.simulate(10, random_seed=self.random_seed)
        ts2 = msprime.simulate(20, random_seed=self.random_seed)
        tables1 = ts1.dump_tables()
        tables2 = ts2.dump_tables()
        tables2.edges.set_columns(**tables1.edges.asdict())
        # The edges in tables2 will refer to nodes that don't exist.
        with pytest.raises(_tskit.LibraryError):
            tables2.sort()
        with pytest.raises(_tskit.LibraryError):
            tables2.canonicalise()

    def test_incompatible_sites(self):
        ts1 = msprime.simulate(10, random_seed=self.random_seed)
        ts2 = msprime.simulate(10, mutation_rate=2, random_seed=self.random_seed)
        assert ts2.num_sites > 1
        tables1 = ts1.dump_tables()
        tables2 = ts2.dump_tables()
        # The mutations in tables2 will refer to sites that don't exist.
        tables1.mutations.set_columns(**tables2.mutations.asdict())
        with pytest.raises(_tskit.LibraryError):
            tables1.sort()
        with pytest.raises(_tskit.LibraryError):
            tables1.canonicalise()

    def test_incompatible_mutation_nodes(self):
        ts1 = msprime.simulate(2, random_seed=self.random_seed)
        ts2 = msprime.simulate(10, mutation_rate=2, random_seed=self.random_seed)
        assert ts2.num_sites > 1
        tables1 = ts1.dump_tables()
        tables2 = ts2.dump_tables()
        # The mutations in tables2 will refer to nodes that don't exist.
        tables1.sites.set_columns(**tables2.sites.asdict())
        tables1.mutations.set_columns(**tables2.mutations.asdict())
        with pytest.raises(_tskit.LibraryError):
            tables1.sort()
        with pytest.raises(_tskit.LibraryError):
            tables1.canonicalise()

    def test_empty_tables(self):
        tables = tskit.TableCollection(1)
        tables.sort()
        assert tables.nodes.num_rows == 0
        assert tables.edges.num_rows == 0
        assert tables.sites.num_rows == 0
        assert tables.mutations.num_rows == 0
        assert tables.migrations.num_rows == 0


class TestSortMigrations:
    """
    Tests that migrations are correctly ordered when sorting tables.
    """

    def test_msprime_output_unmodified(self):
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(3)]
        migration_matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        ts = msprime.simulate(
            recombination_rate=0.1,
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            record_migrations=True,
            random_seed=1,
        )
        assert ts.num_migrations > 100
        tables = ts.tables.copy()
        tables.sort()
        assert tables.equals(ts.tables, ignore_provenance=True)

    def test_full_sort_order(self):
        tables = tskit.TableCollection(1)
        for _ in range(3):
            tables.nodes.add_row()
            tables.populations.add_row()
        for left in [0, 0.5]:
            for a_time in range(3):
                for node in range(2):
                    tables.migrations.add_row(
                        left=left, right=1, node=node, source=0, dest=1, time=a_time
                    )
                    tables.migrations.add_row(
                        left=left, right=1, node=node, source=1, dest=2, time=a_time
                    )

        sorted_list = sorted(
            tables.migrations, key=lambda m: (m.time, m.source, m.dest, m.left, m.node)
        )
        assert sorted_list != list(tables.migrations)
        tables.sort()
        assert sorted_list == list(tables.migrations)


class TestSortMutations:
    """
    Tests that mutations are correctly ordered when sorting tables.
    """

    def test_sort_mutations_stability(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.1     0
        0.2     0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state   parent
        1       0       1               -1
        1       1       1               -1
        0       1       1               -1
        0       0       1               -1
        """
        )
        ts = tskit.load_text(
            nodes=nodes,
            edges=edges,
            sites=sites,
            mutations=mutations,
            sequence_length=1,
            strict=False,
        )
        # Load text automatically calls tables.sort(), so we can test the
        # output directly.
        sites = ts.tables.sites
        mutations = ts.tables.mutations
        assert len(sites) == 2
        assert len(mutations) == 4
        assert list(mutations.site) == [0, 0, 1, 1]
        assert list(mutations.node) == [1, 0, 0, 1]

    def test_sort_mutations_remap_parent_id(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.1     0
        0.2     0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    time    derived_state   parent
        1       0       0.5     1               -1
        1       0       0.25    0               0
        1       0       0       1               1
        0       0       0.5     1               -1
        0       0       0.125   0               3
        0       0       0       1               4
        """
        )
        ts = tskit.load_text(
            nodes=nodes,
            edges=edges,
            sites=sites,
            mutations=mutations,
            sequence_length=1,
            strict=False,
        )
        # Load text automatically calls sort tables, so we can test the
        # output directly.
        sites = ts.tables.sites
        mutations = ts.tables.mutations
        assert len(sites) == 2
        assert len(mutations) == 6
        assert list(mutations.site) == [0, 0, 0, 1, 1, 1]
        assert list(mutations.node) == [0, 0, 0, 0, 0, 0]
        assert list(mutations.time) == [0.5, 0.125, 0.0, 0.5, 0.25, 0.0]
        assert list(mutations.parent) == [-1, 0, 1, -1, 3, 4]

    def test_sort_mutations_bad_parent_id(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.1     0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state   parent
        1       0       1               -2
        """
        )
        with pytest.raises(_tskit.LibraryError):
            tskit.load_text(
                nodes=nodes,
                edges=edges,
                sites=sites,
                mutations=mutations,
                sequence_length=1,
                strict=False,
            )

    def test_sort_mutations_time(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           -6
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.1     0
        0.2     0
        0.3     0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    time    derived_state   parent
        2       0       4       a              -1
        2       0       -5      b              -1
        2       0       6       c              -1
        1       0       0.5     d              -1
        1       0       0.5     e              -1
        1       0       0.5     f              -1
        0       0       1       g              -1
        0       0       2       h              -1
        0       0       3       i              -1
        """
        )
        ts = tskit.load_text(
            nodes=nodes,
            edges=edges,
            sites=sites,
            mutations=mutations,
            sequence_length=1,
            strict=False,
        )
        # Load text automatically calls sort tables, so we can test the
        # output directly.
        sites = ts.tables.sites
        mutations = ts.tables.mutations
        assert len(sites) == 3
        assert len(mutations) == 9
        assert list(mutations.site) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert list(mutations.node) == [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Nans are not equal so swap in -1
        times = mutations.time
        times[np.isnan(times)] = -1
        assert list(times) == [3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 6.0, 4.0, -5.0]
        assert list(mutations.derived_state) == list(
            map(ord, ["i", "h", "g", "d", "e", "f", "c", "a", "b"])
        )
        assert list(mutations.parent) == [-1, -1, -1, -1, -1, -1, -1, -1, -1]


class TestTablesToTreeSequence:
    """
    Tests for the .tree_sequence() method of a TableCollection.
    """

    def test_round_trip(self):
        a = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = a.dump_tables()
        b = tables.tree_sequence()
        assert a.tables == b.tables


class TestMutationTimeErrors:
    def test_younger_than_node_below(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        tables.mutations.time = np.zeros(len(tables.mutations.time), dtype=np.float64)
        with pytest.raises(
            _tskit.LibraryError,
            match="A mutation's time must be >= the node time, or be marked as"
            " 'unknown'",
        ):
            tables.tree_sequence()

    def test_older_than_node_above(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        tables.mutations.time = (
            np.ones(len(tables.mutations.time), dtype=np.float64) * 42
        )
        with pytest.raises(
            _tskit.LibraryError,
            match="A mutation's time must be < the parent node of the edge on which it"
            " occurs, or be marked as 'unknown'",
        ):
            tables.tree_sequence()

    def test_older_than_parent_node(self):
        ts = msprime.simulate(
            10, random_seed=42, mutation_rate=0.0, recombination_rate=1.0
        )
        ts = tsutil.jukes_cantor(
            ts, num_sites=10, mu=1, multiple_per_node=False, seed=42
        )
        tables = ts.dump_tables()
        assert sum(tables.mutations.parent != -1) != 0
        # Make all times the node time
        times = tables.nodes.time[tables.mutations.node]
        # Then make mutations without a parent really old
        times[tables.mutations.parent == -1] = 64.0
        tables.mutations.time = times
        tables.sort()
        with pytest.raises(
            _tskit.LibraryError,
            match="A mutation's time must be < the parent node of the edge on which it"
            " occurs, or be marked as 'unknown'",
        ):
            tables.tree_sequence()

    def test_older_than_parent_mutation(self):
        ts = msprime.simulate(
            10, random_seed=42, mutation_rate=0.0, recombination_rate=1.0
        )
        ts = tsutil.jukes_cantor(
            ts, num_sites=10, mu=1, multiple_per_node=False, seed=42
        )
        tables = ts.dump_tables()
        tables.compute_mutation_times()
        assert sum(tables.mutations.parent != -1) != 0
        times = tables.mutations.time
        # Then make mutations without a parent really old
        times[tables.mutations.parent != -1] = 64.0
        tables.mutations.time = times
        with pytest.raises(_tskit.LibraryError):
            tables.tree_sequence()

    def test_unsorted_times(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.1     0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    time    derived_state   parent
        0       0       1       0              -1
        0       0       2       0              -1
        0       0       3       0              -1
        """
        )
        ts = tskit.load_text(
            nodes=nodes,
            edges=edges,
            sites=sites,
            mutations=mutations,
            sequence_length=1,
            strict=False,
        )
        tables = ts.dump_tables()
        tables.mutations.time = tables.mutations.time[::-1]
        with pytest.raises(
            _tskit.LibraryError,
            match="Mutations must be provided in non-decreasing site order and"
            " non-increasing"
            " time order within each site",
        ):
            tables.tree_sequence()

    def test_mixed_known_and_unknown(self):
        ts = msprime.simulate(
            10, random_seed=42, mutation_rate=0.0, recombination_rate=1.0
        )
        ts = tsutil.jukes_cantor(
            ts, num_sites=10, mu=1, multiple_per_node=False, seed=42
        )
        tables = ts.dump_tables()
        tables.compute_mutation_times()
        tables.sort()
        times = tables.mutations.time
        # Unknown times on diff sites pass
        times[(tables.mutations.site % 2) == 0] = tskit.UNKNOWN_TIME
        tables.mutations.time = times
        tables.tree_sequence()
        # Mixed known/unknown times on sites fail
        times[::2] = tskit.UNKNOWN_TIME
        tables.mutations.time = times
        with pytest.raises(
            _tskit.LibraryError,
            match="Mutation times must either be all marked 'unknown', or all be known "
            "values for any single site.",
        ):
            tables.tree_sequence()


class TestNanDoubleValues:
    """
    In some tables we need to guard against NaN/infinite values in the input.
    """

    def test_edge_coords(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)

        tables = ts.dump_tables()
        bad_coords = tables.edges.left + float("inf")
        tables.edges.left = bad_coords
        with pytest.raises(_tskit.LibraryError):
            tables.tree_sequence()

        tables = ts.dump_tables()
        bad_coords = tables.edges.right + float("nan")
        tables.edges.right = bad_coords
        with pytest.raises(_tskit.LibraryError):
            tables.tree_sequence()

    def test_migrations(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)

        tables = ts.dump_tables()
        tables.populations.add_row()
        tables.migrations.add_row(float("inf"), 1, time=0, node=0, source=0, dest=1)
        with pytest.raises(_tskit.LibraryError):
            tables.tree_sequence()

        tables = ts.dump_tables()
        tables.populations.add_row()
        tables.migrations.add_row(0, float("nan"), time=0, node=0, source=0, dest=1)
        with pytest.raises(_tskit.LibraryError):
            tables.tree_sequence()

        tables = ts.dump_tables()
        tables.populations.add_row()
        tables.migrations.add_row(0, 1, time=float("nan"), node=0, source=0, dest=1)
        with pytest.raises(_tskit.LibraryError):
            tables.tree_sequence()

    def test_site_positions(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        bad_pos = tables.sites.position.copy()
        bad_pos[-1] = np.inf
        tables.sites.position = bad_pos
        with pytest.raises(_tskit.LibraryError):
            tables.tree_sequence()

    def test_node_times(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        bad_times = tables.nodes.time.copy()
        bad_times[-1] = np.inf
        tables.nodes.time = bad_times
        with pytest.raises(_tskit.LibraryError, match="Times must be finite"):
            tables.tree_sequence()
        bad_times[-1] = math.nan
        tables.nodes.time = bad_times
        with pytest.raises(_tskit.LibraryError, match="Times must be finite"):
            tables.tree_sequence()

    def test_mutation_times(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        bad_times = tables.mutations.time.copy()
        bad_times[-1] = np.inf
        tables.mutations.time = bad_times
        with pytest.raises(_tskit.LibraryError, match="Times must be finite"):
            tables.tree_sequence()
        bad_times = tables.mutations.time.copy()
        bad_times[-1] = math.nan
        tables.mutations.time = bad_times
        with pytest.raises(_tskit.LibraryError, match="Times must be finite"):
            tables.tree_sequence()

    def test_individual(self):
        ts = msprime.simulate(12, mutation_rate=1, random_seed=42)
        ts = tsutil.insert_random_ploidy_individuals(ts, seed=42)
        assert ts.num_individuals > 1
        tables = ts.dump_tables()
        bad_locations = tables.individuals.location.copy()
        bad_locations[0] = np.inf
        tables.individuals.location = bad_locations
        ts = tables.tree_sequence()


class TestSimplifyTables:
    """
    Tests for the simplify_tables function.
    """

    random_seed = 42

    def test_deprecated_zero_mutation_sites(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=self.random_seed)
        tables = ts.dump_tables()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tables.simplify(ts.samples(), filter_zero_mutation_sites=True)
            assert len(w) == 1
            assert issubclass(w[-1].category, FutureWarning)

    def test_zero_mutation_sites(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=self.random_seed)
        for filter_sites in [True, False]:
            t1 = ts.dump_tables()
            with pytest.warns(FutureWarning):
                t1.simplify([0, 1], filter_zero_mutation_sites=filter_sites)
            t2 = ts.dump_tables()
            t2.simplify([0, 1], filter_sites=filter_sites)
            t1.provenances.clear()
            t2.provenances.clear()
            assert t1 == t2
            if filter_sites:
                assert ts.num_sites > len(t1.sites)

    def test_full_samples(self):
        for n in [2, 10, 100, 1000]:
            ts = msprime.simulate(
                n, recombination_rate=1, mutation_rate=1, random_seed=self.random_seed
            )
            tables = ts.dump_tables()
            nodes_before = tables.nodes.copy()
            edges_before = tables.edges.copy()
            sites_before = tables.sites.copy()
            mutations_before = tables.mutations.copy()
            for samples in [None, list(ts.samples()), ts.samples()]:
                node_map = tables.simplify(samples=samples)
                assert node_map.shape == (len(nodes_before),)
                assert nodes_before == tables.nodes
                assert edges_before == tables.edges
                assert sites_before == tables.sites
                assert mutations_before == tables.mutations

    def test_bad_samples(self):
        n = 10
        ts = msprime.simulate(n, random_seed=self.random_seed)
        tables = ts.dump_tables()
        for bad_node in [-1, n, n + 1, ts.num_nodes - 1, ts.num_nodes, 2 ** 31 - 1]:
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, bad_node])

    def test_bad_edge_ordering(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        tables = ts.dump_tables()
        edges = tables.edges
        # Reversing the edges violates the ordering constraints.
        edges.set_columns(
            left=edges.left[::-1],
            right=edges.right[::-1],
            parent=edges.parent[::-1],
            child=edges.child[::-1],
        )
        with pytest.raises(_tskit.LibraryError):
            tables.simplify(samples=[0, 1])

    def test_bad_edges(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        for bad_node in [-1, ts.num_nodes, ts.num_nodes + 1, 2 ** 31 - 1]:
            # Bad parent node
            tables = ts.dump_tables()
            edges = tables.edges
            parent = edges.parent
            parent[0] = bad_node
            edges.set_columns(
                left=edges.left, right=edges.right, parent=parent, child=edges.child
            )
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, 1])
            # Bad child node
            tables = ts.dump_tables()
            edges = tables.edges
            child = edges.child
            child[0] = bad_node
            edges.set_columns(
                left=edges.left, right=edges.right, parent=edges.parent, child=child
            )
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, 1])
            # child == parent
            tables = ts.dump_tables()
            edges = tables.edges
            child = edges.child
            child[0] = edges.parent[0]
            edges.set_columns(
                left=edges.left, right=edges.right, parent=edges.parent, child=child
            )
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, 1])
            # left == right
            tables = ts.dump_tables()
            edges = tables.edges
            left = edges.left
            left[0] = edges.right[0]
            edges.set_columns(
                left=left, right=edges.right, parent=edges.parent, child=edges.child
            )
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, 1])
            # left > right
            tables = ts.dump_tables()
            edges = tables.edges
            left = edges.left
            left[0] = edges.right[0] + 1
            edges.set_columns(
                left=left, right=edges.right, parent=edges.parent, child=edges.child
            )
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, 1])

    def test_bad_mutation_nodes(self):
        ts = msprime.simulate(10, random_seed=self.random_seed, mutation_rate=1)
        assert ts.num_mutations > 0
        for bad_node in [-1, ts.num_nodes, 2 ** 31 - 1]:
            tables = ts.dump_tables()
            mutations = tables.mutations
            node = mutations.node
            node[0] = bad_node
            mutations.set_columns(
                site=mutations.site,
                node=node,
                derived_state=mutations.derived_state,
                derived_state_offset=mutations.derived_state_offset,
            )
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, 1])

    def test_bad_mutation_sites(self):
        ts = msprime.simulate(10, random_seed=self.random_seed, mutation_rate=1)
        assert ts.num_mutations > 0
        for bad_site in [-1, ts.num_sites, 2 ** 31 - 1]:
            tables = ts.dump_tables()
            mutations = tables.mutations
            site = mutations.site
            site[0] = bad_site
            mutations.set_columns(
                site=site,
                node=mutations.node,
                derived_state=mutations.derived_state,
                derived_state_offset=mutations.derived_state_offset,
            )
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, 1])

    def test_bad_site_positions(self):
        ts = msprime.simulate(10, random_seed=self.random_seed, mutation_rate=1)
        assert ts.num_mutations > 0
        # Positions > sequence_length are valid, as we can have gaps at the end of
        # a tree sequence.
        for bad_position in [-1, -1e-6]:
            tables = ts.dump_tables()
            sites = tables.sites
            position = sites.position
            position[0] = bad_position
            sites.set_columns(
                position=position,
                ancestral_state=sites.ancestral_state,
                ancestral_state_offset=sites.ancestral_state_offset,
            )
            with pytest.raises(_tskit.LibraryError):
                tables.simplify(samples=[0, 1])

    def test_duplicate_positions(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.sites.add_row(0, ancestral_state="0")
        tables.sites.add_row(0, ancestral_state="0")
        with pytest.raises(_tskit.LibraryError):
            tables.simplify([])

    def test_samples_interface(self):
        ts = msprime.simulate(50, random_seed=1)
        for good_form in [[], [0, 1], (0, 1), np.array([0, 1], dtype=np.int32)]:
            tables = ts.dump_tables()
            tables.simplify(good_form)
        tables = ts.dump_tables()
        for bad_values in [[[[]]], np.array([[0, 1], [2, 3]], dtype=np.int32)]:
            with pytest.raises(ValueError):
                tables.simplify(bad_values)
        for bad_type in [[0.1], ["string"], {}, [{}]]:
            with pytest.raises(TypeError):
                tables.simplify(bad_type)
        # We only convert to int if we don't overflow
        for bad_node in [np.iinfo(np.int32).min - 1, np.iinfo(np.int32).max + 1]:
            with pytest.raises(OverflowError):
                tables.simplify(samples=np.array([0, bad_node]))

    @pytest.fixture(scope="session")
    def wf_sim_with_individual_metadata(self):
        tables = wf.wf_sim(
            9,
            10,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=5,
            record_individuals=True,
        )
        assert tables.individuals.num_rows > 50
        assert np.all(tables.nodes.individual >= 0)
        individuals_copy = tables.copy().individuals
        tables.individuals.clear()
        tables.individuals.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        for i, individual in enumerate(individuals_copy):
            tables.individuals.add_row(
                flags=individual.flags,
                location=individual.location,
                parents=individual.parents,
                metadata={
                    "original_id": i,
                    "original_parents": [int(p) for p in individual.parents],
                },
            )
        tables.sort()
        return tables

    def test_individual_parent_mapping(self, wf_sim_with_individual_metadata):
        tables = wf_sim_with_individual_metadata.copy()
        tables.simplify()
        ts = tables.tree_sequence()
        for individual in tables.individuals:
            for parent, original_parent in zip(
                individual.parents, individual.metadata["original_parents"]
            ):
                if parent != tskit.NULL:
                    assert (
                        ts.individual(parent).metadata["original_id"] == original_parent
                    )
        assert set(tables.individuals.parents) != {tskit.NULL}

    def verify_complete_genetic_pedigree(self, tables):
        ts = tables.tree_sequence()
        for edge in ts.edges():
            child = ts.individual(ts.node(edge.child).individual)
            parent = ts.individual(ts.node(edge.parent).individual)
            assert parent.id in child.parents
            assert parent.metadata["original_id"] in child.metadata["original_parents"]

    def test_no_complete_genetic_pedigree(self, wf_sim_with_individual_metadata):
        tables = wf_sim_with_individual_metadata.copy()
        tables.simplify()  # Will remove intermediate individuals
        with pytest.raises(AssertionError):
            self.verify_complete_genetic_pedigree(tables)

    def test_complete_genetic_pedigree(self, wf_sim_with_individual_metadata):
        for params in [{"keep_unary": True}, {"keep_unary_in_individuals": True}]:
            tables = wf_sim_with_individual_metadata.copy()
            tables.simplify(**params)  # Keep intermediate individuals
            self.verify_complete_genetic_pedigree(tables)

    def test_shuffled_individual_parent_mapping(self, wf_sim_with_individual_metadata):
        tables = wf_sim_with_individual_metadata.copy()
        tsutil.shuffle_tables(
            tables,
            42,
            shuffle_edges=False,
            shuffle_populations=False,
            shuffle_individuals=True,
            shuffle_sites=False,
            shuffle_mutations=False,
            shuffle_migrations=False,
        )
        # Check we have a mixed up order
        with pytest.raises(
            tskit.LibraryError,
            match="Individuals must be provided in an order where"
            " children are after their parent individuals",
        ):
            tables.tree_sequence()

        tables.simplify()
        metadata = [
            tables.individuals.metadata_schema.decode_row(m)
            for m in tskit.unpack_bytes(
                tables.individuals.metadata, tables.individuals.metadata_offset
            )
        ]
        for individual in tables.individuals:
            for parent, original_parent in zip(
                individual.parents, individual.metadata["original_parents"]
            ):
                if parent != tskit.NULL:
                    assert metadata[parent]["original_id"] == original_parent
        assert set(tables.individuals.parents) != {tskit.NULL}

    def test_individual_mapping(self):
        tables = wf.wf_sim(
            9,
            10,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=5,
            record_individuals=True,
        )
        assert tables.individuals.num_rows > 50
        node_md = []
        individual_md = [b""] * tables.individuals.num_rows
        for i, node in enumerate(tables.nodes):
            node_md.append(struct.pack("i", i))
            individual_md[node.individual] = struct.pack("i", i)
        tables.nodes.packset_metadata(node_md)
        tables.individuals.packset_metadata(individual_md)
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        for node in tables.nodes:
            if node.individual != tskit.NULL:
                assert ts.individual(node.individual).metadata == node.metadata
        assert set(tables.individuals.parents) != {tskit.NULL}

    def test_bad_individuals(self, simple_ts_fixture):
        tables = simple_ts_fixture.dump_tables()
        tables.individuals.clear()
        tables.individuals.add_row(parents=[-2])
        with pytest.raises(tskit.LibraryError, match="Individual out of bounds"):
            tables.simplify()
        tables.individuals.clear()
        tables.individuals.add_row(parents=[0])
        with pytest.raises(
            tskit.LibraryError, match="Individuals cannot be their own parents"
        ):
            tables.simplify()

    def test_unsorted_individuals_ok(self, simple_ts_fixture):
        tables = simple_ts_fixture.dump_tables()
        tables.individuals.clear()
        tables.individuals.clear()
        tables.individuals.add_row(parents=[1])
        tables.individuals.add_row(parents=[-1])
        # we really just want to check that no error is thrown here
        tables.simplify()
        assert tables.individuals.num_rows == 0


class TestTableCollection:
    """
    Tests for the convenience wrapper around a collection of related tables.
    """

    def add_metadata(self, tc):
        tc.metadata_schema = tskit.MetadataSchema(
            {
                "codec": "struct",
                "type": "object",
                "properties": {"top-level": {"type": "string", "binaryFormat": "50p"}},
            }
        )
        tc.metadata = {"top-level": "top-level-metadata"}
        for table in [
            "individuals",
            "nodes",
            "edges",
            "migrations",
            "sites",
            "mutations",
            "populations",
        ]:
            t = getattr(tc, table)
            t.packset_metadata([f"{table}-{i}".encode() for i in range(t.num_rows)])
            t.metadata_schema = tskit.MetadataSchema(
                {
                    "codec": "struct",
                    "type": "object",
                    "properties": {table: {"type": "string", "binaryFormat": "50p"}},
                }
            )

    def test_table_references(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=1)
        tables = ts.tables
        before_individuals = str(tables.individuals)
        individuals = tables.individuals
        before_nodes = str(tables.nodes)
        nodes = tables.nodes
        before_edges = str(tables.edges)
        edges = tables.edges
        before_migrations = str(tables.migrations)
        migrations = tables.migrations
        before_sites = str(tables.sites)
        sites = tables.sites
        before_mutations = str(tables.mutations)
        mutations = tables.mutations
        before_populations = str(tables.populations)
        populations = tables.populations
        before_provenances = str(tables.provenances)
        provenances = tables.provenances
        del tables
        assert str(individuals) == before_individuals
        assert str(nodes) == before_nodes
        assert str(edges) == before_edges
        assert str(migrations) == before_migrations
        assert str(sites) == before_sites
        assert str(mutations) == before_mutations
        assert str(populations) == before_populations
        assert str(provenances) == before_provenances

    def test_str(self):
        ts = msprime.simulate(10, random_seed=1)
        tables = ts.tables
        s = str(tables)
        assert len(s) > 0

    def test_nbytes(self, tmp_path, ts_fixture):
        tables = ts_fixture.dump_tables()
        tables.dump(tmp_path / "tables")
        store = kastore.load(tmp_path / "tables")
        for v in store.values():
            # Check we really have data in every field
            assert v.nbytes > 0
        nbytes = sum(
            array.nbytes
            for name, array in store.items()
            # nbytes is the size of asdict, so exclude file format items
            if name not in ["format/version", "format/name", "uuid"]
        )
        assert nbytes == tables.nbytes

    def test_asdict(self, ts_fixture):
        t = ts_fixture.dump_tables()
        d1 = {
            "encoding_version": (1, 2),
            "sequence_length": t.sequence_length,
            "metadata_schema": repr(t.metadata_schema),
            "metadata": t.metadata_schema.encode_row(t.metadata),
            "individuals": t.individuals.asdict(),
            "populations": t.populations.asdict(),
            "nodes": t.nodes.asdict(),
            "edges": t.edges.asdict(),
            "sites": t.sites.asdict(),
            "mutations": t.mutations.asdict(),
            "migrations": t.migrations.asdict(),
            "provenances": t.provenances.asdict(),
            "indexes": t.indexes.asdict(),
        }
        d2 = t.asdict()
        assert set(d1.keys()) == set(d2.keys())
        t1 = tskit.TableCollection.fromdict(d1)
        t2 = tskit.TableCollection.fromdict(d2)
        assert t1 == t2
        assert t1.has_index()
        assert t2.has_index()

    def test_asdict_lifecycle(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        tables_dict = tables.asdict()
        del tables
        assert tskit.TableCollection.fromdict(tables_dict) == ts_fixture.dump_tables()

    def test_from_dict(self, ts_fixture):
        t1 = ts_fixture.tables
        d = {
            "encoding_version": (1, 1),
            "sequence_length": t1.sequence_length,
            "metadata_schema": repr(t1.metadata_schema),
            "metadata": t1.metadata_schema.encode_row(t1.metadata),
            "individuals": t1.individuals.asdict(),
            "populations": t1.populations.asdict(),
            "nodes": t1.nodes.asdict(),
            "edges": t1.edges.asdict(),
            "sites": t1.sites.asdict(),
            "mutations": t1.mutations.asdict(),
            "migrations": t1.migrations.asdict(),
            "provenances": t1.provenances.asdict(),
            "indexes": t1.indexes.asdict(),
        }
        t2 = tskit.TableCollection.fromdict(d)
        assert t1 == t2

    def test_roundtrip_dict(self, ts_fixture):
        t1 = ts_fixture.tables
        t2 = tskit.TableCollection.fromdict(t1.asdict())
        assert t1 == t2

    def test_name_map(self, ts_fixture):
        tables = ts_fixture.tables
        td1 = {
            "individuals": tables.individuals,
            "populations": tables.populations,
            "nodes": tables.nodes,
            "edges": tables.edges,
            "sites": tables.sites,
            "mutations": tables.mutations,
            "migrations": tables.migrations,
            "provenances": tables.provenances,
        }
        td2 = tables.name_map
        assert isinstance(td2, dict)
        assert set(td1.keys()) == set(td2.keys())
        for name in td2.keys():
            assert td1[name] == td2[name]
        assert td1 == td2

    def test_equals_empty(self):
        assert tskit.TableCollection() == tskit.TableCollection()

    def test_equals_sequence_length(self):
        assert tskit.TableCollection(sequence_length=1) != tskit.TableCollection(
            sequence_length=2
        )

    def test_copy(self, ts_fixture):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()
        assert t1 is not t2
        assert t1 == t2
        t1.edges.clear()
        assert t1 != t2

    def test_clear_table(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        tables.clear()
        data_tables = [
            "individuals",
            "nodes",
            "edges",
            "migrations",
            "sites",
            "mutations",
            "populations",
        ]
        for table in data_tables:
            assert getattr(tables, f"{table}").num_rows == 0
            assert repr(getattr(tables, f"{table}").metadata_schema) != ""
        assert tables.provenances.num_rows > 0
        assert len(tables.metadata) > 0
        assert repr(tables.metadata_schema) != ""

        tables.clear(clear_provenance=True)
        assert tables.provenances.num_rows == 0
        for table in data_tables:
            assert repr(getattr(tables, f"{table}").metadata_schema) != ""
        assert len(tables.metadata) > 0
        assert repr(tables.metadata_schema) != ""

        tables.clear(clear_metadata_schemas=True)
        for table in data_tables:
            assert repr(getattr(tables, f"{table}").metadata_schema) == ""
        assert len(tables.metadata) > 0
        assert repr(tables.metadata_schema) != 0

        tables.clear(clear_ts_metadata_and_schema=True)
        assert len(tables.metadata) == 0
        assert repr(tables.metadata_schema) == ""

    def test_equals(self):
        # Here we don't use the fixture as we'd like to run the same sim twice
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        t1 = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        ).dump_tables()
        t2 = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        ).dump_tables()
        assert t1 == t1
        assert t1 == t1.copy()
        assert t1.copy() == t1

        # The provenances may or may not be equal depending on the clock
        # precision for record. So clear them first.
        t1.provenances.clear()
        t2.provenances.clear()
        assert t1 == t2
        assert t1 == t2
        assert not (t1 != t2)

        t1.nodes.clear()
        assert t1 != t2
        t2.nodes.clear()
        assert t1 == t2

        t1.edges.clear()
        assert t1 != t2
        t2.edges.clear()
        assert t1 == t2

        t1.migrations.clear()
        assert t1 != t2
        t2.migrations.clear()
        assert t1 == t2

        t1.sites.clear()
        assert t1 != t2
        t2.sites.clear()
        assert t1 == t2

        t1.mutations.clear()
        assert t1 != t2
        t2.mutations.clear()
        assert t1 == t2

        t1.populations.clear()
        assert t1 != t2
        t2.populations.clear()
        assert t1 == t2

    def test_equals_options(self, ts_fixture):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()

        t1.provenances.add_row("random stuff")
        assert not (t1 == t2)
        assert t1.equals(t2, ignore_provenance=True)
        assert t2.equals(t1, ignore_provenance=True)
        assert not (t1.equals(t2))
        assert not (t2.equals(t1))
        t1.provenances.clear()
        t2.provenances.clear()
        assert t1.equals(t2)
        assert t2.equals(t1)

        t1.metadata_schema = tskit.MetadataSchema({"codec": "json", "type": "object"})
        t1.metadata = {"hello": "world"}
        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_ts_metadata=True)
        assert not t2.equals(t1)
        assert t2.equals(t1, ignore_ts_metadata=True)
        t2.metadata_schema = t1.metadata_schema
        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_ts_metadata=True)
        assert not t2.equals(t1)
        assert t2.equals(t1, ignore_ts_metadata=True)

        t1.provenances.add_row("random stuff")
        assert not t1.equals(t2)
        assert not t1.equals(t2, ignore_ts_metadata=True)
        assert not t1.equals(t2, ignore_provenance=True)
        assert t1.equals(t2, ignore_ts_metadata=True, ignore_provenance=True)

        t1.provenances.clear()
        t2.metadata = t1.metadata
        assert t1.equals(t2)
        assert t2.equals(t1)

        with pytest.raises(TypeError):
            t1.equals(t2, True)

    def test_sequence_length(self):
        for sequence_length in [0, 1, 100.1234]:
            tables = tskit.TableCollection(sequence_length=sequence_length)
            assert tables.sequence_length == sequence_length

    def test_uuid_simulation(self, ts_fixture):
        tables = ts_fixture.tables
        assert tables.file_uuid is None, None

    def test_uuid_empty(self):
        tables = tskit.TableCollection(sequence_length=1)
        assert tables.file_uuid is None, None

    def test_empty_indexes(self):
        tables = tskit.TableCollection(sequence_length=1)
        assert not tables.has_index()
        tables.build_index()
        assert tables.has_index()
        tables.drop_index()
        assert not tables.has_index()

    def test_index_unsorted(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=0, time=2)
        tables.edges.add_row(0, 1, 3, 0)
        tables.edges.add_row(0, 1, 3, 1)
        tables.edges.add_row(0, 1, 4, 3)
        tables.edges.add_row(0, 1, 4, 2)

        assert not tables.has_index()
        with pytest.raises(tskit.LibraryError):
            tables.build_index()
        assert not tables.has_index()
        tables.sort()
        tables.build_index()
        assert tables.has_index()
        ts = tables.tree_sequence()
        assert ts.tables == tables

    def test_index_from_ts(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        assert tables.has_index()
        tables.drop_index()
        assert not tables.has_index()
        ts = tables.tree_sequence()
        assert ts.tables == tables
        assert tables.has_index()

    def test_set_sequence_length_errors(self):
        tables = tskit.TableCollection(1)
        with pytest.raises(AttributeError):
            del tables.sequence_length
        for bad_value in ["asdf", None, []]:
            with pytest.raises(TypeError):
                tables.sequence_length = bad_value

    def test_set_sequence_length(self):
        tables = tskit.TableCollection(1)
        for value in [-1, 100, 2 ** 32, 1e-6]:
            tables.sequence_length = value
            assert tables.sequence_length == value

    def test_bad_sequence_length(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        assert tables.sequence_length == 1
        for value in [-1, 0, -0.99, 0.9999]:
            tables.sequence_length = value
            with pytest.raises(tskit.LibraryError):
                tables.tree_sequence()
            with pytest.raises(tskit.LibraryError):
                tables.sort()
            with pytest.raises(tskit.LibraryError):
                tables.build_index()
            with pytest.raises(tskit.LibraryError):
                tables.compute_mutation_parents()
            with pytest.raises(tskit.LibraryError):
                tables.simplify()
            assert tables.sequence_length == value

    def test_sequence_length_longer_than_edges(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        tables.sequence_length = 2
        ts = tables.tree_sequence()
        assert ts.sequence_length == 2
        assert ts.num_trees == 2
        trees = ts.trees()
        tree = next(trees)
        assert len(tree.parent_dict) > 0
        tree = next(trees)
        assert len(tree.parent_dict) == 0

    def test_indexes(self, simple_ts_fixture):
        tc = tskit.TableCollection(sequence_length=1)
        assert tc.indexes == tskit.TableCollectionIndexes()
        tc = simple_ts_fixture.tables
        assert np.array_equal(
            tc.indexes.edge_insertion_order, np.arange(18, dtype=np.int32)
        )
        assert np.array_equal(
            tc.indexes.edge_removal_order, np.arange(18, dtype=np.int32)[::-1]
        )
        tc.drop_index()
        assert tc.indexes == tskit.TableCollectionIndexes()
        tc.build_index()
        assert np.array_equal(
            tc.indexes.edge_insertion_order, np.arange(18, dtype=np.int32)
        )
        assert np.array_equal(
            tc.indexes.edge_removal_order, np.arange(18, dtype=np.int32)[::-1]
        )

        modify_indexes = tskit.TableCollectionIndexes(
            edge_insertion_order=np.arange(42, 42 + 18, dtype=np.int32),
            edge_removal_order=np.arange(4242, 4242 + 18, dtype=np.int32),
        )
        tc.indexes = modify_indexes
        assert np.array_equal(
            tc.indexes.edge_insertion_order, np.arange(42, 42 + 18, dtype=np.int32)
        )
        assert np.array_equal(
            tc.indexes.edge_removal_order, np.arange(4242, 4242 + 18, dtype=np.int32)
        )

    def test_indexes_roundtrip(self, simple_ts_fixture):
        # Indexes shouldn't be made by roundtripping
        tables = tskit.TableCollection(sequence_length=1)
        assert not tables.has_index()
        assert not tskit.TableCollection.fromdict(tables.asdict()).has_index()

        tables = simple_ts_fixture.dump_tables()
        tables.drop_index()
        assert not tskit.TableCollection.fromdict(tables.asdict()).has_index()

    def test_asdict_lwt_concordance(self, ts_fixture):
        def check_concordance(d1, d2):
            assert set(d1.keys()) == set(d2.keys())
            for k1, v1 in d1.items():
                v2 = d2[k1]
                assert type(v1) == type(v2)
                if type(v1) == dict:
                    assert set(v1.keys()) == set(v2.keys())
                    for sk1, sv1 in v1.items():
                        sv2 = v2[sk1]
                        assert type(sv1) == type(sv2)
                        if type(sv1) == np.ndarray:
                            assert np.array_equal(sv1, sv2) or (
                                np.all(tskit.is_unknown_time(sv1))
                                and np.all(tskit.is_unknown_time(sv2))
                            )
                        elif type(sv1) in [bytes, str]:
                            assert sv1 == sv2
                        else:
                            raise AssertionError()

                else:
                    assert v1 == v2

        tables = ts_fixture.dump_tables()
        assert tables.has_index()
        lwt = _tskit.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        check_concordance(lwt.asdict(), tables.asdict())

        tables.drop_index()
        lwt = _tskit.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        check_concordance(lwt.asdict(), tables.asdict())

    def test_dump_pathlib(self, ts_fixture, tmp_path):
        path = pathlib.Path(tmp_path) / "tmp.trees"
        assert path.exists
        assert path.is_file
        tc = ts_fixture.dump_tables()
        tc.dump(path)
        other_tc = tskit.TableCollection.load(path)
        assert tc == other_tc

    @pytest.mark.skipif(platform.system() == "Windows", reason="Windows doesn't raise")
    def test_dump_load_errors(self, ts_fixture):
        tc = ts_fixture.dump_tables()
        # Try to dump/load files we don't have access to or don't exist.
        for func in [tc.dump, tskit.TableCollection.load]:
            for f in ["/", "/test.trees", "/dir_does_not_exist/x.trees"]:
                with pytest.raises(OSError):
                    func(f)
                try:
                    func(f)
                except OSError as e:
                    message = str(e)
                    assert len(message) > 0
            f = "/" + 4000 * "x"
            with pytest.raises(OSError):
                func(f)
            try:
                func(f)
            except OSError as e:
                message = str(e)
            assert "File name too long" in message
            for bad_filename in [[], None, {}]:
                with pytest.raises(TypeError):
                    func(bad_filename)


class TestEqualityOptions:
    def test_equals_provenance(self):
        t1 = msprime.simulate(10, random_seed=42).tables
        time.sleep(0.1)
        t2 = msprime.simulate(10, random_seed=42).tables
        # Timestamps should differ
        assert t1.provenances[-1].timestamp != t2.provenances[-1].timestamp
        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_timestamps=True)
        assert t1.equals(t2, ignore_provenance=True)
        assert t1.equals(t2, ignore_provenance=True, ignore_timestamps=True)

    def test_equals_node_metadata(self, ts_fixture):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()
        assert t1.equals(t2)
        t1.nodes.add_row(time=0, metadata={"a": "a"})
        t2.nodes.add_row(time=0, metadata={"a": "b"})
        assert not t1.nodes.equals(t2.nodes)
        assert not t1.equals(t2)
        assert t1.nodes.equals(t2.nodes, ignore_metadata=True)

    def test_equals_edge_metadata(self, ts_fixture):
        t1 = ts_fixture.dump_tables()
        child = t1.nodes.add_row(time=0)
        parent = t1.nodes.add_row(time=1)
        t2 = t1.copy()
        assert t1.equals(t2)
        t1.edges.add_row(0, 1, parent, child, metadata={"a": "a"})
        t2.edges.add_row(0, 1, parent, child, metadata={"a": "b"})
        assert not t1.edges.equals(t2.edges)
        assert not t1.equals(t2)
        assert t1.edges.equals(t2.edges, ignore_metadata=True)
        assert t1.equals(t2, ignore_metadata=True)

    def test_equals_migration_metadata(self, ts_fixture):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()
        assert t1.equals(t2)
        t1.migrations.add_row(
            0, 1, source=0, dest=1, node=0, time=0, metadata={"a": "a"}
        )
        t2.migrations.add_row(
            0, 1, source=0, dest=1, node=0, time=0, metadata={"a": "b"}
        )
        assert not t1.migrations.equals(t2.migrations)
        assert not t1.equals(t2)
        assert t1.migrations.equals(t2.migrations, ignore_metadata=True)
        assert t1.equals(t2, ignore_metadata=True)

    def test_equals_site_metadata(self, ts_fixture):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()
        assert t1.equals(t2)
        t1.sites.add_row(0, "A", metadata={"a": "a"})
        t2.sites.add_row(0, "A", metadata={"a": "b"})
        assert not t1.sites.equals(t2.sites)
        assert not t1.equals(t2)
        assert t1.sites.equals(t2.sites, ignore_metadata=True)
        assert t1.equals(t2, ignore_metadata=True)

    def test_equals_mutation_metadata(self, ts_fixture):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()
        assert t1.equals(t2)
        t1.mutations.add_row(0, 0, "A", metadata={"a": "a"})
        t2.mutations.add_row(0, 0, "A", metadata={"a": "b"})
        assert not t1.mutations.equals(t2.mutations)
        assert not t1.equals(t2)
        assert t1.mutations.equals(t2.mutations, ignore_metadata=True)
        assert t1.equals(t2, ignore_metadata=True)

    def test_equals_population_metadata(self, ts_fixture):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()
        assert t1.equals(t2)
        t1.populations.add_row({"a": "a"})
        t2.populations.add_row({"a": "b"})
        assert not t1.populations.equals(t2.populations)
        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_metadata=True)


class TestTableCollectionMethodSignatures:
    tc = msprime.simulate(10, random_seed=1234).dump_tables()

    def test_kwargs_only(self):
        with pytest.raises(TypeError):
            self.tc.simplify([], True)


class TestTableCollectionMetadata:

    metadata_schema = metadata.MetadataSchema(
        {
            "codec": "json",
            "title": "Example Metadata",
            "type": "object",
            "properties": {
                "one": {"type": "string"},
                "two": {"type": "number"},
                "three": {"type": "array"},
                "four": {"type": "boolean"},
            },
            "required": ["one", "two", "three", "four"],
            "additionalProperties": False,
        },
    )

    def metadata_example_data(self, val=0):
        return {
            "one": "val one",
            "two": val,
            "three": list(range(val, val + 10)),
            "four": True,
        }

    def test_set_metadata_schema(self):
        tc = tskit.TableCollection(1)
        metadata_schema2 = metadata.MetadataSchema({"codec": "json"})
        # Default is no-op metadata codec
        assert repr(tc.metadata_schema) == repr(metadata.MetadataSchema(None))
        # Set
        tc.metadata_schema = self.metadata_schema
        assert repr(tc.metadata_schema) == repr(self.metadata_schema)
        # Overwrite
        tc.metadata_schema = metadata_schema2
        assert repr(tc.metadata_schema) == repr(metadata_schema2)
        # Remove
        tc.metadata_schema = metadata.MetadataSchema(None)
        assert repr(tc.metadata_schema) == repr(metadata.MetadataSchema(None))
        # Set after remove
        tc.metadata_schema = self.metadata_schema
        assert repr(tc.metadata_schema) == repr(self.metadata_schema)
        # Del should fail
        with pytest.raises(AttributeError):
            del tc.metadata_schema
        # None should fail
        with pytest.raises(ValueError):
            tc.metadata_schema = None

    def test_set_metadata(self):
        tc = tskit.TableCollection(1)
        # Default is empty bytes
        assert tc.metadata == b""
        assert tc.metadata_bytes == b""

        tc.metadata_schema = self.metadata_schema
        md1 = self.metadata_example_data()
        md2 = self.metadata_example_data(val=2)
        # Set
        tc.metadata = md1
        assert tc.metadata == md1
        assert tc.metadata_bytes == tskit.canonical_json(md1).encode()
        # Overwrite
        tc.metadata = md2
        assert tc.metadata == md2
        assert tc.metadata_bytes == tskit.canonical_json(md2).encode()
        # Del should fail
        with pytest.raises(AttributeError):
            del tc.metadata
        with pytest.raises(AttributeError):
            del tc.metadata_bytes
        # None should fail
        with pytest.raises(exceptions.MetadataValidationError):
            tc.metadata = None
        # Setting bytes should fail
        with pytest.raises(AttributeError):
            tc.metadata_bytes = b"123"

    def test_default_metadata_schema(self):
        # Default should allow bytes
        tc = tskit.TableCollection(1)
        tc.metadata = b"acceptable bytes"
        assert tc.metadata == b"acceptable bytes"
        # Adding non-bytes metadata should error
        with pytest.raises(TypeError):
            tc.metadata = self.metadata_example_data()

    def test_round_trip_metadata(self):
        data = self.metadata_example_data()
        tc = tskit.TableCollection(1)
        tc.metadata_schema = self.metadata_schema
        tc.metadata = data
        assert tc.metadata == data
        assert tc.metadata_bytes == tskit.canonical_json(data).encode()

    def test_bad_metadata(self):
        metadata = self.metadata_example_data()
        metadata["I really shouldn't be here"] = 6
        tc = tskit.TableCollection(1)
        tc.metadata_schema = self.metadata_schema
        with pytest.raises(exceptions.MetadataValidationError):
            tc.metadata = metadata
        assert tc._ll_tables.metadata == b""


class TestTableCollectionPickle(TestTableCollection):
    """
    Tests that we can round-trip table collections through pickle.
    """

    def verify(self, tables):
        self.add_metadata(tables)
        other_tables = pickle.loads(pickle.dumps(tables))
        assert tables == other_tables

    def test_simple_simulation(self):
        ts = msprime.simulate(2, random_seed=1)
        self.verify(ts.dump_tables())

    def test_simulation_populations(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(10),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            record_migrations=True,
            random_seed=1,
        )
        self.verify(ts.dump_tables())

    def test_simulation_sites(self):
        ts = msprime.simulate(12, random_seed=1, mutation_rate=5)
        assert ts.num_sites > 1
        self.verify(ts.dump_tables())

    def test_simulation_individuals(self):
        ts = msprime.simulate(100, random_seed=1)
        ts = tsutil.insert_random_ploidy_individuals(ts, seed=1)
        assert ts.num_individuals > 1
        self.verify(ts.dump_tables())

    def test_empty_tables(self):
        self.verify(tskit.TableCollection())


class TestDeduplicateSites:
    """
    Tests for the TableCollection.deduplicate_sites method.
    """

    def test_empty(self):
        tables = tskit.TableCollection(1)
        tables.deduplicate_sites()
        assert tables == tskit.TableCollection(1)

    def test_unsorted(self):
        tables = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        assert len(tables.sites) > 0
        position = tables.sites.position
        for _ in range(len(position) - 1):
            position = np.roll(position, 1)
            tables.sites.set_columns(
                position=position,
                ancestral_state=tables.sites.ancestral_state,
                ancestral_state_offset=tables.sites.ancestral_state_offset,
            )
            with pytest.raises(_tskit.LibraryError):
                tables.deduplicate_sites()

    def test_bad_position(self):
        for bad_position in [-1, -0.001]:
            tables = tskit.TableCollection()
            tables.sites.add_row(bad_position, "0")
            with pytest.raises(_tskit.LibraryError):
                tables.deduplicate_sites()

    def test_no_effect(self):
        t1 = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        t2 = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        assert len(t1.sites) > 0
        t1.deduplicate_sites()
        t1.provenances.clear()
        t2.provenances.clear()
        assert t1 == t2

    def test_same_sites(self):
        t1 = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        t2 = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        assert len(t1.sites) > 0
        t1.sites.append_columns(
            position=t1.sites.position,
            ancestral_state=t1.sites.ancestral_state,
            ancestral_state_offset=t1.sites.ancestral_state_offset,
        )
        assert len(t1.sites) == 2 * len(t2.sites)
        t1.sort()
        t1.deduplicate_sites()
        t1.provenances.clear()
        t2.provenances.clear()
        assert t1 == t2

    def test_order_maintained(self):
        t1 = tskit.TableCollection(1)
        t1.sites.add_row(position=0, ancestral_state="first")
        t1.sites.add_row(position=0, ancestral_state="second")
        t1.deduplicate_sites()
        assert len(t1.sites) == 1
        assert t1.sites.ancestral_state.tobytes() == b"first"

    def test_multichar_ancestral_state(self):
        ts = msprime.simulate(8, random_seed=3, mutation_rate=1)
        assert ts.num_sites > 2
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        for site in ts.sites():
            site_id = tables.sites.add_row(
                position=site.position, ancestral_state="A" * site.id
            )
            tables.sites.add_row(position=site.position, ancestral_state="0")
            for mutation in site.mutations:
                tables.mutations.add_row(
                    site=site_id, node=mutation.node, derived_state="T" * site.id
                )
        tables.deduplicate_sites()
        new_ts = tables.tree_sequence()
        assert new_ts.num_sites == ts.num_sites
        for site in new_ts.sites():
            assert site.ancestral_state == site.id * "A"

    def test_multichar_metadata(self):
        ts = msprime.simulate(8, random_seed=3, mutation_rate=1)
        assert ts.num_sites > 2
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        for site in ts.sites():
            site_id = tables.sites.add_row(
                position=site.position, ancestral_state="0", metadata=b"A" * site.id
            )
            tables.sites.add_row(position=site.position, ancestral_state="0")
            for mutation in site.mutations:
                tables.mutations.add_row(
                    site=site_id,
                    node=mutation.node,
                    derived_state="1",
                    metadata=b"T" * site.id,
                )
        tables.deduplicate_sites()
        new_ts = tables.tree_sequence()
        assert new_ts.num_sites == ts.num_sites
        for site in new_ts.sites():
            assert site.metadata == site.id * b"A"


class TestBaseTable:
    """
    Tests of the table superclass.
    """

    def test_set_columns_not_implemented(self):
        t = tskit.BaseTable(None, None)
        with pytest.raises(NotImplementedError):
            t.set_columns()


class TestSubsetTables:
    """
    Tests for the TableCollection.subset method.
    """

    def get_msprime_example(self, sample_size=10, seed=1234):
        M = [[0.0, 0.1], [1.0, 0.0]]
        population_configurations = [
            msprime.PopulationConfiguration(sample_size=sample_size),
            msprime.PopulationConfiguration(sample_size=sample_size),
        ]
        ts = msprime.simulate(
            population_configurations=population_configurations,
            migration_matrix=M,
            length=2e5,
            recombination_rate=1e-8,
            mutation_rate=1e-7,
            record_migrations=False,
            random_seed=seed,
        )
        # adding metadata and locations
        ts = tsutil.add_random_metadata(ts, seed)
        ts = tsutil.insert_random_ploidy_individuals(ts, max_ploidy=1)
        return ts.tables

    def get_wf_example(self, N=5, ngens=2, seed=1249):
        tables = wf.wf_sim(N, N, num_pops=2, seed=seed)
        tables.sort()
        ts = tables.tree_sequence()
        # adding muts
        ts = tsutil.jukes_cantor(ts, 1, 10, seed=seed)
        ts = tsutil.add_random_metadata(ts, seed)
        ts = tsutil.insert_random_ploidy_individuals(ts, max_ploidy=2)
        return ts.tables

    def get_examples(self, seed):
        yield self.get_msprime_example(seed=seed)
        yield self.get_wf_example(seed=seed)

    def verify_subset_equality(self, tables, nodes):
        for rp in [True, False]:
            for ru in [True, False]:
                sub1 = tables.copy()
                sub2 = tables.copy()
                tsutil.py_subset(
                    sub1,
                    nodes,
                    record_provenance=False,
                    reorder_populations=rp,
                    remove_unreferenced=ru,
                )
                sub2.subset(
                    nodes,
                    record_provenance=False,
                    reorder_populations=rp,
                    remove_unreferenced=ru,
                )
                tsutil.assert_table_collections_equal(sub1, sub2)

    def verify_subset(self, tables, nodes):
        self.verify_subset_equality(tables, nodes)
        subset = tables.copy()
        subset.subset(nodes, record_provenance=False)
        # adding one so the last element always maps to NULL (-1 -> -1)
        node_map = np.repeat(tskit.NULL, tables.nodes.num_rows + 1)
        indivs = []
        pops = []
        for k, n in enumerate(nodes):
            node_map[n] = k
            ind = tables.nodes[n].individual
            pop = tables.nodes[n].population
            if ind not in indivs and ind != tskit.NULL:
                indivs.append(ind)
            if pop not in pops and pop != tskit.NULL:
                pops.append(pop)
        ind_map = np.repeat(tskit.NULL, tables.individuals.num_rows + 1)
        ind_map[indivs] = np.arange(len(indivs), dtype="int32")
        pop_map = np.repeat(tskit.NULL, tables.populations.num_rows + 1)
        pop_map[pops] = np.arange(len(pops), dtype="int32")
        assert subset.nodes.num_rows == len(nodes)
        for k, n in zip(nodes, subset.nodes):
            nn = tables.nodes[k]
            assert nn.time == n.time
            assert nn.flags == n.flags
            assert nn.metadata == n.metadata
            assert ind_map[nn.individual] == n.individual
            assert pop_map[nn.population] == n.population
        assert subset.individuals.num_rows == len(indivs)
        for k, i in zip(indivs, subset.individuals):
            ii = tables.individuals[k]
            assert ii == i
        assert subset.populations.num_rows == len(pops)
        for k, p in zip(pops, subset.populations):
            pp = tables.populations[k]
            assert pp == p
        edges = [
            i
            for i, e in enumerate(tables.edges)
            if e.parent in nodes and e.child in nodes
        ]
        assert subset.edges.num_rows == len(edges)
        for k, e in zip(edges, subset.edges):
            ee = tables.edges[k]
            assert ee.left == e.left
            assert ee.right == e.right
            assert node_map[ee.parent] == e.parent
            assert node_map[ee.child] == e.child
            assert ee.metadata == e.metadata
        muts = []
        sites = []
        for k, m in enumerate(tables.mutations):
            if m.node in nodes:
                muts.append(k)
                if m.site not in sites:
                    sites.append(m.site)
        site_map = np.repeat(-1, tables.sites.num_rows)
        site_map[sites] = np.arange(len(sites), dtype="int32")
        mutation_map = np.repeat(tskit.NULL, tables.mutations.num_rows + 1)
        mutation_map[muts] = np.arange(len(muts), dtype="int32")
        assert subset.sites.num_rows == len(sites)
        for k, s in zip(sites, subset.sites):
            ss = tables.sites[k]
            assert ss == s
        assert subset.mutations.num_rows == len(muts)
        for k, m in zip(muts, subset.mutations):
            mm = tables.mutations[k]
            assert mutation_map[mm.parent] == m.parent
            assert site_map[mm.site] == m.site
            assert node_map[mm.node] == m.node
            assert mm.derived_state == m.derived_state
            assert mm.metadata == m.metadata
        assert tables.migrations == subset.migrations
        assert tables.provenances == subset.provenances

    def test_ts_subset(self):
        nodes = np.array([0, 1])
        for tables in self.get_examples(83592):
            ts = tables.tree_sequence()
            tables2 = ts.subset(nodes, record_provenance=False).dump_tables()
            tables.subset(nodes, record_provenance=False)
            assert tables == tables2

    def test_subset_all(self):
        # subsetting to everything shouldn't change things except the
        # individual and population ids in the node tables if there are gaps
        for tables in self.get_examples(123583):
            tables2 = tables.copy()
            tables2.subset(np.arange(tables.nodes.num_rows))
            tables.individuals.clear()
            tables2.individuals.clear()
            assert np.all(tables.nodes.time == tables2.nodes.time)
            assert np.all(tables.nodes.flags == tables2.nodes.flags)
            assert np.all(tables.nodes.population == tables2.nodes.population)
            assert np.all(tables.nodes.metadata == tables2.nodes.metadata)
            tables.nodes.clear()
            tables2.nodes.clear()
            tsutil.assert_table_collections_equal(
                tables, tables2, ignore_provenance=True
            )

    def test_shuffled_tables(self):
        # subset should work on even unsorted tables
        # (tested more thoroughly in TestSortTables)
        for tables in self.get_examples(95521):
            tables2 = tables.copy()
            tsutil.shuffle_tables(tables2, 7000)
            tables2.subset(
                np.arange(tables.nodes.num_rows),
                remove_unreferenced=False,
            )
            assert tables.nodes.num_rows == tables2.nodes.num_rows
            assert tables.individuals.num_rows == tables2.individuals.num_rows
            assert tables.populations.num_rows == tables2.populations.num_rows
            assert tables.edges.num_rows == tables2.edges.num_rows
            assert tables.sites.num_rows == tables2.sites.num_rows
            assert tables.mutations.num_rows == tables2.mutations.num_rows
            tables2 = tables.copy()
            tsutil.shuffle_tables(tables2, 7001)
            tables2.subset([])
            assert tables2.nodes.num_rows == 0
            assert tables2.individuals.num_rows == 0
            assert tables2.populations.num_rows == 0
            assert tables2.edges.num_rows == 0
            assert tables2.sites.num_rows == 0
            assert tables2.mutations.num_rows == 0

    def test_random_subsets(self):
        rng = np.random.default_rng(1542)
        for tables in self.get_examples(9412):
            for n in [2, tables.nodes.num_rows - 10]:
                nodes = rng.choice(np.arange(tables.nodes.num_rows), n, replace=False)
                self.verify_subset(tables, nodes)

    def test_empty_nodes(self):
        for tables in self.get_examples(8724):
            subset = tables.copy()
            subset.subset(np.array([]), record_provenance=False)
            assert subset.nodes.num_rows == 0
            assert subset.edges.num_rows == 0
            assert subset.populations.num_rows == 0
            assert subset.individuals.num_rows == 0
            assert subset.migrations.num_rows == 0
            assert subset.sites.num_rows == 0
            assert subset.mutations.num_rows == 0
            assert subset.provenances == tables.provenances

    def test_no_remove_unreferenced(self):
        tables = tskit.TableCollection(sequence_length=10)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables.nodes.add_row(time=1)
        tables.edges.add_row(parent=1, child=0, left=0, right=10)
        for k in range(5):
            tables.sites.add_row(position=k, ancestral_state=str(k))
        # these are all unused, so should remain unchanged
        for k in range(5):
            tables.populations.add_row(metadata=str(k).encode())
        for k in range(5):
            tables.individuals.add_row(metadata=str(k).encode())
        sub_tables = tables.copy()
        sub_tables.subset([], remove_unreferenced=False)
        assert tables.sites == sub_tables.sites
        assert tables.populations == sub_tables.populations
        assert tables.individuals == sub_tables.individuals
        ts = tables.tree_sequence()
        sub_tables = ts.subset([], remove_unreferenced=False).tables
        assert tables.sites == sub_tables.sites
        assert tables.populations == sub_tables.populations
        assert tables.individuals == sub_tables.individuals


class TestUnionTables(unittest.TestCase):
    """
    Tests for the TableCollection.union method
    """

    def get_msprime_example(self, sample_size, T, seed):
        # we assume after the split the ts are completely independent
        M = [[0, 0], [0, 0]]
        population_configurations = [
            msprime.PopulationConfiguration(sample_size=sample_size),
            msprime.PopulationConfiguration(sample_size=sample_size),
        ]
        demographic_events = [
            msprime.CensusEvent(time=T),
            msprime.MassMigration(T, source=1, dest=0, proportion=1),
        ]
        ts = msprime.simulate(
            population_configurations=population_configurations,
            demographic_events=demographic_events,
            migration_matrix=M,
            length=2e5,
            recombination_rate=1e-8,
            mutation_rate=1e-7,
            record_migrations=False,
            random_seed=seed,
        )
        ts = tsutil.add_random_metadata(ts, seed)
        ts = tsutil.insert_random_ploidy_individuals(ts, max_ploidy=1)
        return ts

    def get_wf_example(self, N, T, seed):
        twopop_tables = wf.wf_sim(N, T, num_pops=2, seed=seed, deep_history=True)
        twopop_tables.sort()
        ts = twopop_tables.tree_sequence()
        ts = ts.simplify()
        ts = tsutil.jukes_cantor(ts, 1, 10, seed=seed)
        ts = tsutil.add_random_metadata(ts, seed)
        ts = tsutil.insert_random_ploidy_individuals(ts, max_ploidy=2)
        return ts

    def split_example(self, ts, T):
        # splitting two pop ts *with no migration* into disjoint ts
        shared_nodes = [n.id for n in ts.nodes() if n.time >= T]
        pop1 = list(ts.samples(population=0))
        pop2 = list(ts.samples(population=1))
        tables1 = ts.simplify(shared_nodes + pop1, record_provenance=False).tables
        tables2 = ts.simplify(shared_nodes + pop2, record_provenance=False).tables
        node_mapping = [
            i if i < len(shared_nodes) else tskit.NULL
            for i in range(tables2.nodes.num_rows)
        ]
        # adding some metadata to one of the tables
        # union should disregard differences in metadata
        tables1.metadata_schema = tskit.MetadataSchema(
            {"codec": "json", "type": "object"}
        )
        tables1.metadata = {"hello": "world"}
        return tables1, tables2, node_mapping

    def verify_union_equality(self, tables, other, node_mapping, add_populations=True):
        # verifying against py impl
        uni1 = tables.copy()
        uni2 = tables.copy()
        uni1.union(
            other,
            node_mapping,
            record_provenance=True,
            add_populations=add_populations,
        )
        tsutil.py_union(
            uni2,
            other,
            node_mapping,
            record_provenance=False,
            add_populations=add_populations,
        )
        tsutil.assert_table_collections_equal(uni1, uni2, ignore_provenance=True)
        # verifying that subsetting to original nodes return the same table
        orig_nodes = [j for i, j in enumerate(node_mapping) if j != tskit.NULL]
        uni1.subset(orig_nodes)
        # subsetting tables just to make sure order is the same
        tables.subset(orig_nodes)
        tsutil.assert_table_collections_equal(uni1, tables, ignore_provenance=True)

    def test_union_empty(self):
        ts1 = self.get_msprime_example(sample_size=3, T=2, seed=9328)
        ts2 = tskit.TableCollection(sequence_length=ts1.sequence_length).tree_sequence()
        uni = ts1.union(ts2, [])
        tsutil.assert_table_collections_equal(
            ts1.tables, uni.tables, ignore_provenance=True
        )

    def test_noshared_example(self):
        ts1 = self.get_msprime_example(sample_size=3, T=2, seed=9328)
        ts2 = self.get_msprime_example(sample_size=3, T=2, seed=2125)
        node_mapping = np.full(ts2.num_nodes, tskit.NULL, dtype="int32")
        uni1 = ts1.union(ts2, node_mapping, record_provenance=False)
        uni2_tables = ts1.dump_tables()
        tsutil.py_union(uni2_tables, ts2.tables, node_mapping, record_provenance=False)
        assert uni1.tables == uni2_tables

    def test_all_shared_example(self):
        tables = self.get_wf_example(N=5, T=5, seed=11349).dump_tables()
        uni = tables.copy()
        node_mapping = np.arange(tables.nodes.num_rows)
        uni.union(tables, node_mapping, record_provenance=False)
        assert uni == tables

    def test_no_add_pop(self):
        self.verify_union_equality(
            *self.split_example(self.get_msprime_example(10, 10, seed=135), 10),
            add_populations=False,
        )
        self.verify_union_equality(
            *self.split_example(self.get_wf_example(10, 10, seed=157), 10),
            add_populations=False,
        )

    def test_provenance(self):
        tables, other, node_mapping = self.split_example(
            self.get_msprime_example(5, T=2, seed=928), 2
        )
        tables_copy = tables.copy()
        tables.union(other, node_mapping)
        uni_other_dict = json.loads(tables.provenances[-1].record)["parameters"][
            "other"
        ]
        recovered_prov_table = tskit.ProvenanceTable()
        assert len(uni_other_dict["timestamp"]) == len(uni_other_dict["record"])
        for timestamp, record in zip(
            uni_other_dict["timestamp"], uni_other_dict["record"]
        ):
            recovered_prov_table.add_row(record, timestamp)
        assert recovered_prov_table == other.provenances
        tables.provenances.truncate(tables.provenances.num_rows - 1)
        assert tables.provenances == tables_copy.provenances

    def test_examples(self):
        for N in [2, 4, 5]:
            for T in [2, 5, 20]:
                for mut_times in [True, False]:
                    with self.subTest(N=N, T=T):
                        ts = self.get_msprime_example(N, T=T, seed=888)
                        if mut_times:
                            tables = ts.tables
                            tables.compute_mutation_times()
                            ts = tables.tree_sequence()
                        self.verify_union_equality(*self.split_example(ts, T))
                        ts = self.get_wf_example(N=N, T=T, seed=827)
                        if mut_times:
                            tables = ts.tables
                            tables.compute_mutation_times()
                            ts = tables.tree_sequence()
                        self.verify_union_equality(*self.split_example(ts, T))


class TestSubsetUnion:
    # Check that we can remove a single individual from a tree sequence and
    # then use union to put it back in again. This is a good test of union,
    # subset, and various sort options.

    def verify_subset_union(self, ts):
        tables = ts.tables
        # remove unused individuals and populations since if there's more than
        # one of these then it can't be canonically sorted below
        tables.subset(np.arange(tables.nodes.num_rows))
        has_unknown_times = np.any(tskit.is_unknown_time(tables.mutations.time))
        if has_unknown_times:
            tsutil.shuffle_tables(tables, seed=2545, keep_mutation_parent_order=True)
        else:
            tsutil.shuffle_tables(tables, seed=2545, keep_mutation_parent_order=False)
        rng = random.Random(2345)
        target = np.array([rng.choice(list(range(tables.nodes.num_rows)))])
        if ts.node(target[0]).individual != tskit.NULL:
            ind = ts.individual(ts.node(target[0]).individual)
            target = ind.nodes
        target_relatives = np.concatenate(
            [
                tables.edges.parent[np.isin(tables.edges.child, target)],
                tables.edges.child[np.isin(tables.edges.parent, target)],
            ]
        )
        not_target = [a for a in range(tables.nodes.num_rows) if a not in target]
        # relationships to target
        target_plus = np.concatenate([target, target_relatives])
        sub1 = tables.copy()
        sub1.subset(target_plus, reorder_populations=False)
        # everything else
        new_tables = tables.copy()
        new_tables.subset(not_target, reorder_populations=False)
        # and union back together
        mapping12 = [tskit.NULL for _ in target] + [
            not_target.index(n) for n in target_relatives
        ]
        # union needs tables to be sorted
        new_tables.sort()
        sub1.sort()
        # double-check we're doing the mapping right
        new_tables.union(sub1, node_mapping=mapping12, add_populations=False)
        reorder = np.arange(new_tables.nodes.num_rows)
        for j, t in enumerate(target):
            reorder[t:] -= 1
            reorder[t] = new_tables.nodes.num_rows - len(target) + j
        new_tables.subset(reorder)
        new_tables.canonicalise()
        tables.canonicalise()
        tsutil.assert_table_collections_equal(
            new_tables, tables, ignore_provenance=True
        )

    def get_wf_example(self, N, T, seed):
        twopop_tables = wf.wf_sim(N, T, num_pops=2, seed=seed, deep_history=True)
        twopop_tables.sort()
        ts = twopop_tables.tree_sequence()
        ts = ts.simplify()
        ts = tsutil.jukes_cantor(ts, 1, 10, seed=seed)
        ts = tsutil.add_random_metadata(ts, seed)
        ts = tsutil.insert_random_ploidy_individuals(ts, max_ploidy=2)
        return ts

    @pytest.mark.skip("Fails due to #1225")
    def test_no_mutation_times(self):
        ts = self.get_wf_example(10, 4, seed=925)
        self.verify_subset_union(ts)

    def test_with_mutation_times(self):
        ts = self.get_wf_example(10, 4, seed=61)
        tables = ts.tables
        tables.compute_mutation_times()
        ts = tables.tree_sequence()
        self.verify_subset_union(ts)
