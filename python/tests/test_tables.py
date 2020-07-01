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
Test cases for the low-level tables used to transfer information
between simulations and the tree sequence.
"""
import io
import itertools
import math
import pickle
import random
import unittest
import warnings

import attr
import msprime
import numpy as np

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
            self.assertRaises(
                ValueError, self.table_class, max_rows_increment=bad_value
            )
        for v in [1, 100, 256]:
            table = self.table_class(max_rows_increment=v)
            self.assertEqual(table.max_rows_increment, v)
        # Setting zero or not argument both denote the default.
        table = self.table_class()
        self.assertEqual(table.max_rows_increment, 1024)
        table = self.table_class(max_rows_increment=0)
        self.assertEqual(table.max_rows_increment, 1024)

    def test_low_level_get_row(self):
        # Tests the low-level get_row interface to ensure we're getting coverage.
        t = self.table_class()
        with self.assertRaises(TypeError):
            t.ll_table.get_row()
        with self.assertRaises(TypeError):
            t.ll_table.get_row("row")
        with self.assertRaises(_tskit.LibraryError):
            t.ll_table.get_row(1)

    def test_low_level_equals(self):
        # Tests the low-level equals interface to ensure we're getting coverage.
        t = self.table_class()
        with self.assertRaises(TypeError):
            t.ll_table.equals()
        with self.assertRaises(TypeError):
            t.ll_table.equals(None)

    def test_low_level_set_columns(self):
        t = self.table_class()
        with self.assertRaises(TypeError):
            t.ll_table.set_columns(None)
        with self.assertRaises(TypeError):
            t.ll_table.append_columns(None)

    def test_input_parameters_errors(self):
        self.assertGreater(len(self.input_parameters), 0)
        for param, _ in self.input_parameters:
            for bad_value in [-1, -(2 ** 10)]:
                self.assertRaises(ValueError, self.table_class, **{param: bad_value})
            for bad_type in [None, ValueError, "ser"]:
                self.assertRaises(TypeError, self.table_class, **{param: bad_type})

    def test_input_parameter_values(self):
        self.assertGreater(len(self.input_parameters), 0)
        for param, _ in self.input_parameters:
            for v in [1, 100, 256]:
                table = self.table_class(**{param: v})
                self.assertEqual(getattr(table, param), v)

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
            self.assertRaises(TypeError, table.set_columns, **kwargs)
            kwargs = dict(inputs)
            del kwargs[offset_col.name]
            self.assertRaises(TypeError, table.set_columns, **kwargs)

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
                self.assertRaises(ValueError, table.set_columns, **error_kwargs)
                self.assertRaises(ValueError, table.append_columns, **error_kwargs)
            for bad_value in ["qwer", [0, "sd"]]:
                error_kwargs = dict(kwargs)
                error_kwargs[focal_col.name] = bad_value
                self.assertRaises(ValueError, table.set_columns, **error_kwargs)
                self.assertRaises(ValueError, table.append_columns, **error_kwargs)

    def test_set_columns_from_dict(self):
        kwargs = self.make_input_data(1)
        # Make sure this works.
        t1 = self.table_class()
        t1.set_columns(**kwargs)
        t2 = self.table_class()
        t2.set_columns(**t1.asdict())
        self.assertEqual(t1, t2)

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
                self.assertRaises(ValueError, table.set_columns, **error_kwargs)
                self.assertRaises(ValueError, table.append_columns, **error_kwargs)
        for _, offset_col in self.ragged_list_columns:
            error_kwargs = dict(kwargs)
            for bad_dims in [5, [[1], [1]], np.zeros((2, 2))]:
                error_kwargs[offset_col.name] = bad_dims
                self.assertRaises(ValueError, table.set_columns, **error_kwargs)
                self.assertRaises(ValueError, table.append_columns, **error_kwargs)
            # Empty offset columns are caught also
            error_kwargs[offset_col.name] = []
            self.assertRaises(ValueError, table.set_columns, **error_kwargs)

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
                    self.assertRaises(ValueError, table.set_columns, **kwargs)
                    self.assertRaises(ValueError, table.append_columns, **kwargs)

    def test_set_read_only_attributes(self):
        table = self.table_class()
        with self.assertRaises(AttributeError):
            table.num_rows = 10
        with self.assertRaises(AttributeError):
            table.max_rows = 10
        for param, _default in self.input_parameters:
            with self.assertRaises(AttributeError):
                setattr(table, param, 2)
        self.assertEqual(table.num_rows, 0)
        self.assertEqual(len(table), 0)

    def test_set_column_attributes_empty(self):
        table = self.table_class()
        input_data = {col.name: col.get_input(0) for col in self.columns}
        for col, data in input_data.items():
            setattr(table, col, data)
            self.assertEqual(len(getattr(table, col)), 0)

    def test_set_column_attributes_data(self):
        table = self.table_class()
        for num_rows in [1, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table.set_columns(**input_data)

            for list_col, offset_col in self.ragged_list_columns:
                list_data = input_data[list_col.name]
                self.assertTrue(
                    np.array_equal(getattr(table, list_col.name), list_data)
                )
                list_data += 1
                self.assertFalse(
                    np.array_equal(getattr(table, list_col.name), list_data)
                )
                setattr(table, list_col.name, list_data)
                self.assertTrue(
                    np.array_equal(getattr(table, list_col.name), list_data)
                )
                list_value = getattr(table[0], list_col.name)
                self.assertEqual(len(list_value), 1)

                # Reset the offsets so that all the full array is associated with the
                # first element.
                offset_data = np.zeros(num_rows + 1, dtype=np.uint32) + num_rows
                offset_data[0] = 0
                setattr(table, offset_col.name, offset_data)
                list_value = getattr(table[0], list_col.name)
                self.assertEqual(len(list_value), num_rows)

                del input_data[list_col.name]
                del input_data[offset_col.name]

            for col, data in input_data.items():
                self.assertTrue(np.array_equal(getattr(table, col), data))
                data += 1
                self.assertFalse(np.array_equal(getattr(table, col), data))
                setattr(table, col, data)
                self.assertTrue(np.array_equal(getattr(table, col), data))

    def test_set_column_attributes_errors(self):
        table = self.table_class()
        num_rows = 10
        input_data = self.make_input_data(num_rows)
        table.set_columns(**input_data)

        for list_col, offset_col in self.ragged_list_columns:
            for bad_list_col in [[], input_data[list_col.name][:-1]]:
                with self.assertRaises(ValueError):
                    setattr(table, list_col.name, bad_list_col)
            for bad_offset_col in [[], np.arange(num_rows + 2, dtype=np.uint32)]:
                with self.assertRaises(ValueError):
                    setattr(table, offset_col.name, bad_offset_col)

            del input_data[list_col.name]
            del input_data[offset_col.name]

        for col, data in input_data.items():
            for bad_data in [[], data[:-1]]:
                with self.assertRaises(ValueError):
                    setattr(table, col, bad_data)

        # Try to read a column that isn't there. (We can always write to new attributes
        # in Python, so there's nothing to test in that case.)
        with self.assertRaises(AttributeError):
            _ = table.no_such_column

    def test_defaults(self):
        table = self.table_class()
        self.assertEqual(table.num_rows, 0)
        self.assertEqual(len(table), 0)
        for param, default in self.input_parameters:
            self.assertEqual(getattr(table, param), default)
        for col in self.columns:
            array = getattr(table, col.name)
            self.assertEqual(array.shape, (0,))

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
                self.assertEqual(k, j)
            for colname, input_array in input_data.items():
                output_array = getattr(table, colname)
                self.assertEqual(input_array.shape, output_array.shape)
                self.assertTrue(np.all(input_array == output_array))
            table.clear()
            self.assertEqual(table.num_rows, 0)
            self.assertEqual(len(table), 0)

    def test_add_row_round_trip(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            t1 = self.table_class()
            t1.set_columns(**input_data)
            for colname, input_array in input_data.items():
                output_array = getattr(t1, colname)
                self.assertEqual(input_array.shape, output_array.shape)
                self.assertTrue(np.all(input_array == output_array))
            t2 = self.table_class()
            for row in list(t1):
                t2.add_row(**attr.asdict(row))
            self.assertEqual(t1, t2)

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
                    self.assertEqual(input_array.shape, output_array.shape)
                    self.assertTrue(np.all(input_array == output_array))
                table.clear()
                self.assertEqual(table.num_rows, 0)
                self.assertEqual(len(table), 0)
                for colname in input_data.keys():
                    if colname in offset_cols:
                        self.assertEqual(list(getattr(table, colname)), [0])
                    else:
                        self.assertEqual(list(getattr(table, colname)), [])

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
        self.assertEqual(copy, table)

        for num_rows in [100, 10, 1]:
            table.truncate(num_rows)
            self.assertEqual(table.num_rows, num_rows)
            self.assertEqual(len(table), num_rows)
            used = set()
            for list_col, offset_col in self.ragged_list_columns:
                offset = getattr(table, offset_col.name)
                self.assertEqual(offset.shape, (num_rows + 1,))
                self.assertTrue(
                    np.array_equal(input_data[offset_col.name][: num_rows + 1], offset)
                )
                list_data = getattr(table, list_col.name)
                self.assertTrue(
                    np.array_equal(list_data, input_data[list_col.name][: offset[-1]])
                )
                used.add(offset_col.name)
                used.add(list_col.name)
            for name, data in input_data.items():
                if name not in used:
                    self.assertTrue(
                        np.array_equal(data[:num_rows], getattr(table, name))
                    )

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
            self.assertRaises(TypeError, table.truncate, bad_type)
        for bad_num_rows in [-1, num_rows + 1, 10 ** 6]:
            self.assertRaises(ValueError, table.truncate, bad_num_rows)

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
                        self.assertEqual(input_array.shape, output_array.shape)
                    else:
                        input_array = np.hstack([values for _ in range(j)])
                        self.assertEqual(input_array.shape, output_array.shape)
                    self.assertTrue(np.array_equal(input_array, output_array))
                self.assertEqual(table.num_rows, j * num_rows)
                self.assertEqual(len(table), j * num_rows)

    def test_append_columns_max_rows(self):
        for num_rows in [0, 10, 100, 1000]:
            input_data = self.make_input_data(num_rows)
            for max_rows in [0, 1, 8192]:
                table = self.table_class(max_rows_increment=max_rows)
                for j in range(1, 10):
                    table.append_columns(**input_data)
                    self.assertEqual(table.num_rows, j * num_rows)
                    self.assertEqual(len(table), j * num_rows)
                    self.assertGreater(table.max_rows, table.num_rows)

    def test_str(self):
        for num_rows in [0, 10]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            s = str(table)
            self.assertEqual(len(s.splitlines()), num_rows + 1)

    def test_repr_html(self):
        for num_rows in [0, 10]:
            input_data = {col.name: col.get_input(num_rows) for col in self.columns}
            for list_col, offset_col in self.ragged_list_columns:
                value = list_col.get_input(num_rows)
                input_data[list_col.name] = value
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
            table = self.table_class()
            table.set_columns(**input_data)
            html = table._repr_html_()
            self.assertEqual(len(html.splitlines()), num_rows + 19)

    def test_copy(self):
        for num_rows in [0, 10]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            for _ in range(10):
                copy = table.copy()
                self.assertNotEqual(id(copy), id(table))
                self.assertIsInstance(copy, self.table_class)
                self.assertEqual(copy, table)
                table = copy

    def test_pickle(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            pkl = pickle.dumps(table)
            new_table = pickle.loads(pkl)
            self.assertEqual(table, new_table)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                pkl = pickle.dumps(table, protocol=protocol)
                new_table = pickle.loads(pkl)
                self.assertEqual(table, new_table)

    def test_equality(self):
        for num_rows in [1, 10, 100]:
            input_data = self.make_input_data(num_rows)
            t1 = self.table_class()
            t2 = self.table_class()
            self.assertEqual(t1, t1)
            self.assertEqual(t1, t2)
            self.assertTrue(t1 == t2)
            self.assertFalse(t1 != t2)
            t1.set_columns(**input_data)
            self.assertEqual(t1, t1)
            self.assertNotEqual(t1, t2)
            self.assertNotEqual(t2, t1)
            t2.set_columns(**input_data)
            self.assertEqual(t1, t2)
            self.assertEqual(t2, t2)
            t2.clear()
            self.assertNotEqual(t1, t2)
            self.assertNotEqual(t2, t1)
            # Check each column in turn to see if we are correctly checking values.
            for col in self.columns:
                col_copy = np.copy(input_data[col.name])
                input_data_copy = dict(input_data)
                input_data_copy[col.name] = col_copy
                t2.set_columns(**input_data_copy)
                self.assertEqual(t1, t2)
                self.assertFalse(t1 != t2)
                self.assertEqual(t1[0], t2[0])
                col_copy += 1
                t2.set_columns(**input_data_copy)
                self.assertNotEqual(t1, t2)
                self.assertNotEqual(t2, t1)
                self.assertNotEqual(t1[0], t2[0])
                self.assertTrue(t1[0] != t2[0])
                self.assertTrue(t1[0] != [])
            for list_col, offset_col in self.ragged_list_columns:
                value = list_col.get_input(num_rows)
                input_data_copy = dict(input_data)
                input_data_copy[list_col.name] = value + 1
                t2.set_columns(**input_data_copy)
                self.assertNotEqual(t1, t2)
                self.assertNotEqual(t1[0], t2[0])
                value = list_col.get_input(num_rows + 1)
                input_data_copy = dict(input_data)
                input_data_copy[list_col.name] = value
                input_data_copy[offset_col.name] = np.arange(
                    num_rows + 1, dtype=np.uint32
                )
                input_data_copy[offset_col.name][-1] = num_rows + 1
                t2.set_columns(**input_data_copy)
                self.assertNotEqual(t1, t2)
                self.assertNotEqual(t2, t1)
                self.assertNotEqual(t1[-1], t2[-1])
            # Different types should always be unequal.
            self.assertNotEqual(t1, None)
            self.assertNotEqual(t1, [])

    def test_bad_offsets(self):
        for num_rows in [10, 100]:
            input_data = self.make_input_data(num_rows)
            t = self.table_class()
            t.set_columns(**input_data)

            for _list_col, offset_col in self.ragged_list_columns:
                input_data[offset_col.name][0] = -1
                self.assertRaises(_tskit.LibraryError, t.set_columns, **input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                t.set_columns(**input_data)
                input_data[offset_col.name][-1] = 0
                self.assertRaises(ValueError, t.set_columns, **input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                t.set_columns(**input_data)
                input_data[offset_col.name][num_rows // 2] = 2 ** 31
                self.assertRaises(_tskit.LibraryError, t.set_columns, **input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)

                input_data[offset_col.name][0] = -1
                self.assertRaises(_tskit.LibraryError, t.append_columns, **input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                t.append_columns(**input_data)
                input_data[offset_col.name][-1] = 0
                self.assertRaises(ValueError, t.append_columns, **input_data)
                input_data[offset_col.name] = np.arange(num_rows + 1, dtype=np.uint32)
                t.append_columns(**input_data)
                input_data[offset_col.name][num_rows // 2] = 2 ** 31
                self.assertRaises(_tskit.LibraryError, t.append_columns, **input_data)
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
            self.assertEqual(metadatas, unpacked_metadatas)

    def test_optional_metadata(self):
        if not getattr(self, "metadata_mandatory", False):
            for num_rows in [0, 10, 100]:
                input_data = self.make_input_data(num_rows)
                table = self.table_class()
                del input_data["metadata"]
                del input_data["metadata_offset"]
                table.set_columns(**input_data)
                self.assertEqual(len(list(table.metadata)), 0)
                self.assertEqual(
                    list(table.metadata_offset), [0 for _ in range(num_rows + 1)]
                )
                # Supplying None is the same not providing the column.
                input_data["metadata"] = None
                input_data["metadata_offset"] = None
                table.set_columns(**input_data)
                self.assertEqual(len(list(table.metadata)), 0)
                self.assertEqual(
                    list(table.metadata_offset), [0 for _ in range(num_rows + 1)]
                )

    def test_packset_metadata(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            metadatas = [tsutil.random_bytes(10) for _ in range(num_rows)]
            metadata, metadata_offset = tskit.pack_bytes(metadatas)
            table.packset_metadata(metadatas)
            self.assertTrue(np.array_equal(table.metadata, metadata))
            self.assertTrue(np.array_equal(table.metadata_offset, metadata_offset))

    def test_set_metadata_schema(self):
        metadata_schema2 = metadata.MetadataSchema({"codec": "json"})
        table = self.table_class()
        # Default is no-op metadata codec
        self.assertEqual(str(table.metadata_schema), str(metadata.MetadataSchema(None)))
        # Set
        table.metadata_schema = self.metadata_schema
        self.assertEqual(str(table.metadata_schema), str(self.metadata_schema))
        # Overwrite
        table.metadata_schema = metadata_schema2
        self.assertEqual(str(table.metadata_schema), str(metadata_schema2))
        # Remove
        table.metadata_schema = ""
        self.assertEqual(str(table.metadata_schema), str(metadata.MetadataSchema(None)))
        # Set after remove
        table.metadata_schema = self.metadata_schema
        self.assertEqual(str(table.metadata_schema), str(self.metadata_schema))
        # Del should fail
        with self.assertRaises(AttributeError):
            del table.metadata_schema
        # None should fail
        with self.assertRaises(ValueError):
            table.metadata_schema = None

    def test_default_metadata_schema(self):
        # Default should allow bytes as in pre-exisiting code
        table = self.table_class()
        table.add_row(
            **{**self.input_data_for_add_row(), "metadata": b"acceptable bytes"}
        )
        # Adding non-bytes metadata should error
        with self.assertRaises(TypeError):
            table.add_row(
                **{
                    **self.input_data_for_add_row(),
                    "metadata": self.metadata_example_data(),
                }
            )

    def test_row_round_trip_metadata_schema(self):
        data = self.metadata_example_data()
        table = self.table_class()
        table.metadata_schema = self.metadata_schema
        table.add_row(**{**self.input_data_for_add_row(), "metadata": data})
        self.assertDictEqual(table[0].metadata, data)

    def test_bad_row_metadata_schema(self):
        metadata = self.metadata_example_data()
        metadata["I really shouldn't be here"] = 6
        table = self.table_class()
        table.metadata_schema = self.metadata_schema
        with self.assertRaises(exceptions.MetadataValidationError):
            table.add_row(**{**self.input_data_for_add_row(), "metadata": metadata})
        self.assertEqual(len(table), 0)

    def test_absent_metadata_with_required_schema(self):
        table = self.table_class()
        table.metadata_schema = self.metadata_schema
        input_data = self.input_data_for_add_row()
        del input_data["metadata"]
        with self.assertRaises(exceptions.MetadataValidationError):
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
        with self.assertRaises(exceptions.MetadataValidationError):
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
                self.assertEqual(table[j].metadata, metadata_column[j])
                self.assertEqual(table[j + num_rows].metadata, metadata_column[j])


class TestIndividualTable(unittest.TestCase, CommonTestsMixin, MetadataTestsMixin):
    columns = [UInt32Column("flags")]
    ragged_list_columns = [
        (DoubleColumn("location"), UInt32Column("location_offset")),
        (CharColumn("metadata"), UInt32Column("metadata_offset")),
    ]
    string_colnames = []
    binary_colnames = ["metadata"]
    input_parameters = [("max_rows_increment", 1024)]
    equal_len_columns = [["flags"]]
    table_class = tskit.IndividualTable

    def test_simple_example(self):
        t = tskit.IndividualTable()
        t.add_row(flags=0, location=[], metadata=b"123")
        t.add_row(flags=1, location=(0, 1, 2, 3), metadata=b"\xf0")
        s = str(t)
        self.assertGreater(len(s), 0)
        self.assertEqual(len(t), 2)
        self.assertEqual(t[0].flags, 0)
        self.assertEqual(list(t[0].location), [])
        self.assertEqual(t[0].metadata, b"123")
        self.assertEqual(t[1].flags, 1)
        self.assertEqual(list(t[1].location), [0, 1, 2, 3])
        self.assertEqual(t[1].metadata, b"\xf0")
        self.assertRaises(IndexError, t.__getitem__, -3)

    def test_add_row_defaults(self):
        t = tskit.IndividualTable()
        self.assertEqual(t.add_row(), 0)
        self.assertEqual(t.flags[0], 0)
        self.assertEqual(len(t.location), 0)
        self.assertEqual(t.location_offset[0], 0)
        self.assertEqual(len(t.metadata), 0)
        self.assertEqual(t.metadata_offset[0], 0)

    def test_add_row_bad_data(self):
        t = tskit.IndividualTable()
        with self.assertRaises(TypeError):
            t.add_row(flags="x")
        with self.assertRaises(TypeError):
            t.add_row(metadata=123)
        with self.assertRaises(ValueError):
            t.add_row(location="1234")

    def test_packset_location(self):
        t = tskit.IndividualTable()
        t.add_row(flags=0)
        t.packset_location([[0.125, 2]])
        self.assertEqual(list(t[0].location), [0.125, 2])
        t.add_row(flags=1)
        self.assertEqual(list(t[1].location), [])
        t.packset_location([[0], [1, 2, 3]])
        self.assertEqual(list(t[0].location), [0])
        self.assertEqual(list(t[1].location), [1, 2, 3])


class TestNodeTable(unittest.TestCase, CommonTestsMixin, MetadataTestsMixin):

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
        self.assertGreater(len(s), 0)
        self.assertEqual(len(t), 2)
        self.assertEqual(attr.astuple(t[0]), (0, 1, 2, 0, b"123"))
        self.assertEqual(attr.astuple(t[1]), (1, 2, 3, 1, b"\xf0"))
        self.assertEqual(t[0].flags, 0)
        self.assertEqual(t[0].time, 1)
        self.assertEqual(t[0].population, 2)
        self.assertEqual(t[0].individual, 0)
        self.assertEqual(t[0].metadata, b"123")
        self.assertEqual(t[0], t[-2])
        self.assertEqual(t[1], t[-1])
        self.assertRaises(IndexError, t.__getitem__, -3)

    def test_add_row_defaults(self):
        t = tskit.NodeTable()
        self.assertEqual(t.add_row(), 0)
        self.assertEqual(t.time[0], 0)
        self.assertEqual(t.flags[0], 0)
        self.assertEqual(t.population[0], tskit.NULL)
        self.assertEqual(t.individual[0], tskit.NULL)
        self.assertEqual(len(t.metadata), 0)
        self.assertEqual(t.metadata_offset[0], 0)

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
            self.assertEqual(list(table.population), [-1 for _ in range(num_rows)])
            self.assertEqual(list(table.flags), flags)
            self.assertEqual(list(table.time), time)
            self.assertEqual(list(table.metadata), list(metadata))
            self.assertEqual(list(table.metadata_offset), list(metadata_offset))
            table.set_columns(flags=flags, time=time, population=None)
            self.assertEqual(list(table.population), [-1 for _ in range(num_rows)])
            self.assertEqual(list(table.flags), flags)
            self.assertEqual(list(table.time), time)

    def test_add_row_bad_data(self):
        t = tskit.NodeTable()
        with self.assertRaises(TypeError):
            t.add_row(flags="x")
        with self.assertRaises(TypeError):
            t.add_row(time="x")
        with self.assertRaises(TypeError):
            t.add_row(individual="x")
        with self.assertRaises(TypeError):
            t.add_row(population="x")
        with self.assertRaises(TypeError):
            t.add_row(metadata=123)


class TestEdgeTable(unittest.TestCase, CommonTestsMixin, MetadataTestsMixin):

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
        self.assertEqual(len(t), 2)
        self.assertEqual(attr.astuple(t[0]), (0, 1, 2, 3, b"123"))
        self.assertEqual(attr.astuple(t[1]), (1, 2, 3, 4, b"\xf0"))
        self.assertEqual(t[0].left, 0)
        self.assertEqual(t[0].right, 1)
        self.assertEqual(t[0].parent, 2)
        self.assertEqual(t[0].child, 3)
        self.assertEqual(t[0].metadata, b"123")
        self.assertEqual(t[0], t[-2])
        self.assertEqual(t[1], t[-1])
        self.assertRaises(IndexError, t.__getitem__, -3)

    def test_add_row_defaults(self):
        t = tskit.EdgeTable()
        self.assertEqual(t.add_row(0, 0, 0, 0), 0)
        self.assertEqual(len(t.metadata), 0)
        self.assertEqual(t.metadata_offset[0], 0)

    def test_add_row_bad_data(self):
        t = tskit.EdgeTable()
        with self.assertRaises(TypeError):
            t.add_row(left="x", right=0, parent=0, child=0)
        with self.assertRaises(TypeError):
            t.add_row()
        with self.assertRaises(TypeError):
            t.add_row(0, 0, 0, 0, metadata=123)


class TestSiteTable(unittest.TestCase, CommonTestsMixin, MetadataTestsMixin):
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
        self.assertGreater(len(s), 0)
        self.assertEqual(len(t), 2)
        self.assertEqual(attr.astuple(t[0]), (0, "1", b"2"))
        self.assertEqual(attr.astuple(t[1]), (1, "2", b"\xf0"))
        self.assertEqual(t[0].position, 0)
        self.assertEqual(t[0].ancestral_state, "1")
        self.assertEqual(t[0].metadata, b"2")
        self.assertEqual(t[0], t[-2])
        self.assertEqual(t[1], t[-1])
        self.assertRaises(IndexError, t.__getitem__, 2)
        self.assertRaises(IndexError, t.__getitem__, -3)

    def test_add_row_bad_data(self):
        t = tskit.SiteTable()
        t.add_row(0, "A")
        with self.assertRaises(TypeError):
            t.add_row("x", "A")
        with self.assertRaises(TypeError):
            t.add_row(0, 0)
        with self.assertRaises(TypeError):
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
            self.assertTrue(np.array_equal(table.ancestral_state, ancestral_state))
            self.assertTrue(
                np.array_equal(table.ancestral_state_offset, ancestral_state_offset)
            )


class TestMutationTable(unittest.TestCase, CommonTestsMixin, MetadataTestsMixin):
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
        s = str(t)
        self.assertGreater(len(s), 0)
        self.assertEqual(len(t), 2)
        self.assertEqual(attr.astuple(t[0]), (0, 1, "2", 3, b"4", 5))
        self.assertEqual(attr.astuple(t[1]), (1, 2, "3", 4, b"\xf0", 6))
        self.assertEqual(t[0].site, 0)
        self.assertEqual(t[0].node, 1)
        self.assertEqual(t[0].derived_state, "2")
        self.assertEqual(t[0].parent, 3)
        self.assertEqual(t[0].metadata, b"4")
        self.assertEqual(t[0].time, 5)
        self.assertEqual(t[0], t[-2])
        self.assertEqual(t[1], t[-1])
        self.assertRaises(IndexError, t.__getitem__, -3)

    def test_add_row_bad_data(self):
        t = tskit.MutationTable()
        t.add_row(0, 0, "A")
        with self.assertRaises(TypeError):
            t.add_row("0", 0, "A")
        with self.assertRaises(TypeError):
            t.add_row(0, "0", "A")
        with self.assertRaises(TypeError):
            t.add_row(0, 0, "A", parent=None)
        with self.assertRaises(TypeError):
            t.add_row(0, 0, "A", metadata=[0])
        with self.assertRaises(TypeError):
            t.add_row(0, 0, "A", time="A")

    def test_packset_derived_state(self):
        for num_rows in [0, 10, 100]:
            input_data = self.make_input_data(num_rows)
            table = self.table_class()
            table.set_columns(**input_data)
            derived_states = [tsutil.random_strings(10) for _ in range(num_rows)]
            derived_state, derived_state_offset = tskit.pack_strings(derived_states)
            table.packset_derived_state(derived_states)
            self.assertTrue(np.array_equal(table.derived_state, derived_state))
            self.assertTrue(
                np.array_equal(table.derived_state_offset, derived_state_offset)
            )


class TestMigrationTable(unittest.TestCase, CommonTestsMixin, MetadataTestsMixin):
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
        self.assertEqual(len(t), 2)
        self.assertEqual(attr.astuple(t[0]), (0, 1, 2, 3, 4, 5, b"123"))
        self.assertEqual(attr.astuple(t[1]), (1, 2, 3, 4, 5, 6, b"\xf0"))
        self.assertEqual(t[0].left, 0)
        self.assertEqual(t[0].right, 1)
        self.assertEqual(t[0].node, 2)
        self.assertEqual(t[0].source, 3)
        self.assertEqual(t[0].dest, 4)
        self.assertEqual(t[0].time, 5)
        self.assertEqual(t[0].metadata, b"123")
        self.assertEqual(t[0], t[-2])
        self.assertEqual(t[1], t[-1])
        self.assertRaises(IndexError, t.__getitem__, -3)

    def test_add_row_defaults(self):
        t = tskit.MigrationTable()
        self.assertEqual(t.add_row(0, 0, 0, 0, 0, 0), 0)
        self.assertEqual(len(t.metadata), 0)
        self.assertEqual(t.metadata_offset[0], 0)

    def test_add_row_bad_data(self):
        t = tskit.MigrationTable()
        with self.assertRaises(TypeError):
            t.add_row(left="x", right=0, node=0, source=0, dest=0, time=0)
        with self.assertRaises(TypeError):
            t.add_row()
        with self.assertRaises(TypeError):
            t.add_row(0, 0, 0, 0, 0, 0, metadata=123)


class TestProvenanceTable(unittest.TestCase, CommonTestsMixin):
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
        self.assertEqual(len(t), 2)
        self.assertEqual(attr.astuple(t[0]), ("0", "1"))
        self.assertEqual(attr.astuple(t[1]), ("1", "2"))
        self.assertEqual(t[0].timestamp, "0")
        self.assertEqual(t[0].record, "1")
        self.assertEqual(t[0], t[-2])
        self.assertEqual(t[1], t[-1])
        self.assertRaises(IndexError, t.__getitem__, -3)

    def test_add_row_bad_data(self):
        t = tskit.ProvenanceTable()
        t.add_row("a", "b")
        with self.assertRaises(TypeError):
            t.add_row(0, "b")
        with self.assertRaises(TypeError):
            t.add_row("a", 0)

    def test_packset_timestamp(self):
        t = tskit.ProvenanceTable()
        t.add_row(timestamp="0", record="1")
        t.add_row(timestamp="1", record="2")
        t.packset_timestamp(["AAAA", "BBBB"])
        self.assertEqual(t[0].timestamp, "AAAA")
        self.assertEqual(t[1].timestamp, "BBBB")

    def test_packset_record(self):
        t = tskit.ProvenanceTable()
        t.add_row(timestamp="0", record="1")
        t.add_row(timestamp="1", record="2")
        t.packset_record(["AAAA", "BBBB"])
        self.assertEqual(t[0].record, "AAAA")
        self.assertEqual(t[1].record, "BBBB")


class TestPopulationTable(unittest.TestCase, CommonTestsMixin, MetadataTestsMixin):
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
        self.assertGreater(len(s), 0)
        self.assertEqual(len(t), 2)
        self.assertEqual(attr.astuple(t[0]), (b"\xf0",))
        self.assertEqual(t[0].metadata, b"\xf0")
        self.assertEqual(attr.astuple(t[1]), (b"1",))
        self.assertRaises(IndexError, t.__getitem__, -3)

    def test_add_row_bad_data(self):
        t = tskit.PopulationTable()
        t.add_row()
        with self.assertRaises(TypeError):
            t.add_row(metadata=[0])


class TestSortTables(unittest.TestCase):
    """
    Tests for the TableCollection.sort() method.
    """

    random_seed = 12345

    def verify_randomise_tables(self, ts):
        tables = ts.dump_tables()

        # Randomise the tables.
        random.seed(self.random_seed)
        randomised_edges = list(ts.edges())
        random.shuffle(randomised_edges)
        tables.edges.clear()
        for e in randomised_edges:
            tables.edges.add_row(e.left, e.right, e.parent, e.child)
        # Verify that import fails for randomised edges
        self.assertRaises(_tskit.LibraryError, tables.tree_sequence)
        tables.sort()
        self.assertEqual(tables, ts.dump_tables())

        tables.sites.clear()
        tables.mutations.clear()
        randomised_sites = list(ts.sites())
        random.shuffle(randomised_sites)
        # Maps original IDs into their indexes in the randomised table.
        site_id_map = {}
        randomised_mutations = []
        for s in randomised_sites:
            site_id_map[s.id] = tables.sites.add_row(
                s.position, ancestral_state=s.ancestral_state, metadata=s.metadata
            )
            randomised_mutations.extend(s.mutations)
        random.shuffle(randomised_mutations)
        for m in randomised_mutations:
            tables.mutations.add_row(
                site=site_id_map[m.site],
                node=m.node,
                derived_state=m.derived_state,
                parent=m.parent,
                metadata=m.metadata,
                time=m.time,
            )
        if ts.num_sites > 1:
            # Verify that import fails for randomised sites
            self.assertRaises(_tskit.LibraryError, tables.tree_sequence)
        tables.sort()
        self.assertEqual(tables, ts.dump_tables())

        ts_new = tables.tree_sequence()
        self.assertEqual(ts_new.num_edges, ts.num_edges)
        self.assertEqual(ts_new.num_trees, ts.num_trees)
        self.assertEqual(ts_new.num_sites, ts.num_sites)
        self.assertEqual(ts_new.num_mutations, ts.num_mutations)

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
            self.assertRaises(_tskit.LibraryError, tables.tree_sequence)
            # If we sort after the start value we should still fail.
            tables.sort(edge_start=start + 1)
            self.assertRaises(_tskit.LibraryError, tables.tree_sequence)
            # Sorting from the correct index should give us back the original table.
            tables.edges.clear()
            for e in all_edges:
                tables.edges.add_row(e.left, e.right, e.parent, e.child)
            tables.sort(edge_start=start)
            # Verify the new and old edges are equal.
            self.assertEqual(edges, tables.edges)

    def test_single_tree_no_mutations(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        self.verify_randomise_tables(ts)
        self.verify_edge_sort_offset(ts)

    def test_single_tree_no_mutations_metadata(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts = tsutil.add_random_metadata(ts, self.random_seed)
        self.verify_randomise_tables(ts)

    def test_many_trees_no_mutations(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=self.random_seed)
        self.assertGreater(ts.num_trees, 2)
        self.verify_randomise_tables(ts)
        self.verify_edge_sort_offset(ts)

    def test_single_tree_mutations(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=self.random_seed)
        self.assertGreater(ts.num_sites, 2)
        self.verify_randomise_tables(ts)
        self.verify_edge_sort_offset(ts)

    def test_single_tree_mutations_metadata(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=self.random_seed)
        self.assertGreater(ts.num_sites, 2)
        ts = tsutil.add_random_metadata(ts, self.random_seed)
        self.verify_randomise_tables(ts)

    def test_single_tree_multichar_mutations(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts = tsutil.insert_multichar_mutations(ts, self.random_seed)
        self.verify_randomise_tables(ts)

    def test_single_tree_multichar_mutations_metadata(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts = tsutil.insert_multichar_mutations(ts, self.random_seed)
        ts = tsutil.add_random_metadata(ts, self.random_seed)
        self.verify_randomise_tables(ts)

    def test_many_trees_mutations(self):
        ts = msprime.simulate(
            10, recombination_rate=2, mutation_rate=2, random_seed=self.random_seed
        )
        self.assertGreater(ts.num_trees, 2)
        self.assertGreater(ts.num_sites, 2)
        self.verify_randomise_tables(ts)
        self.verify_edge_sort_offset(ts)

    def test_many_trees_multichar_mutations(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=self.random_seed)
        self.assertGreater(ts.num_trees, 2)
        ts = tsutil.insert_multichar_mutations(ts, self.random_seed)
        self.verify_randomise_tables(ts)

    def test_many_trees_multichar_mutations_metadata(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=self.random_seed)
        self.assertGreater(ts.num_trees, 2)
        ts = tsutil.insert_multichar_mutations(ts, self.random_seed)
        ts = tsutil.add_random_metadata(ts, self.random_seed)
        self.verify_randomise_tables(ts)

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
        self.assertTrue(found)
        return ts

    def test_nonbinary_trees(self):
        ts = self.get_nonbinary_example(mutation_rate=0)
        self.assertGreater(ts.num_trees, 2)
        self.verify_randomise_tables(ts)
        self.verify_edge_sort_offset(ts)

    def test_nonbinary_trees_mutations(self):
        ts = self.get_nonbinary_example(mutation_rate=2)
        self.assertGreater(ts.num_trees, 2)
        self.assertGreater(ts.num_sites, 2)
        self.verify_randomise_tables(ts)
        self.verify_edge_sort_offset(ts)

    def test_incompatible_edges(self):
        ts1 = msprime.simulate(10, random_seed=self.random_seed)
        ts2 = msprime.simulate(20, random_seed=self.random_seed)
        tables1 = ts1.dump_tables()
        tables2 = ts2.dump_tables()
        tables2.edges.set_columns(**tables1.edges.asdict())
        # The edges in tables2 will refer to nodes that don't exist.
        self.assertRaises(_tskit.LibraryError, tables2.sort)

    def test_incompatible_sites(self):
        ts1 = msprime.simulate(10, random_seed=self.random_seed)
        ts2 = msprime.simulate(10, mutation_rate=2, random_seed=self.random_seed)
        self.assertGreater(ts2.num_sites, 1)
        tables1 = ts1.dump_tables()
        tables2 = ts2.dump_tables()
        # The mutations in tables2 will refer to sites that don't exist.
        tables1.mutations.set_columns(**tables2.mutations.asdict())
        self.assertRaises(_tskit.LibraryError, tables1.sort)

    def test_incompatible_mutation_nodes(self):
        ts1 = msprime.simulate(2, random_seed=self.random_seed)
        ts2 = msprime.simulate(10, mutation_rate=2, random_seed=self.random_seed)
        self.assertGreater(ts2.num_sites, 1)
        tables1 = ts1.dump_tables()
        tables2 = ts2.dump_tables()
        # The mutations in tables2 will refer to nodes that don't exist.
        # print(tables2.sites.asdict())
        tables1.sites.set_columns(**tables2.sites.asdict())
        tables1.mutations.set_columns(**tables2.mutations.asdict())
        self.assertRaises(_tskit.LibraryError, tables1.sort)

    def test_empty_tables(self):
        tables = tskit.TableCollection(1)
        tables.sort()
        self.assertEqual(tables.nodes.num_rows, 0)
        self.assertEqual(tables.edges.num_rows, 0)
        self.assertEqual(tables.sites.num_rows, 0)
        self.assertEqual(tables.mutations.num_rows, 0)
        self.assertEqual(tables.migrations.num_rows, 0)


class TestSortMutations(unittest.TestCase):
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
        self.assertEqual(len(sites), 2)
        self.assertEqual(len(mutations), 4)
        self.assertEqual(list(mutations.site), [0, 0, 1, 1])
        self.assertEqual(list(mutations.node), [1, 0, 0, 1])

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
        self.assertEqual(len(sites), 2)
        self.assertEqual(len(mutations), 6)
        self.assertEqual(list(mutations.site), [0, 0, 0, 1, 1, 1])
        self.assertEqual(list(mutations.node), [0, 0, 0, 0, 0, 0])
        self.assertEqual(list(mutations.time), [0.5, 0.125, 0.0, 0.5, 0.25, 0.0])
        self.assertEqual(list(mutations.parent), [-1, 0, 1, -1, 3, 4])

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
        self.assertRaises(
            _tskit.LibraryError,
            tskit.load_text,
            nodes=nodes,
            edges=edges,
            sites=sites,
            mutations=mutations,
            sequence_length=1,
            strict=False,
        )


class TestTablesToTreeSequence(unittest.TestCase):
    """
    Tests for the .tree_sequence() method of a TableCollection.
    """

    random_seed = 42

    def test_round_trip(self):
        a = msprime.simulate(5, mutation_rate=1, random_seed=self.random_seed)
        tables = a.dump_tables()
        b = tables.tree_sequence()
        self.assertTrue(
            all(a == b for a, b in zip(a.tables, b.tables) if a[0] != "provenances")
        )


class TestMutationTimeErrors(unittest.TestCase):
    def test_younger_than_node_below(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        tables.mutations.time = np.zeros(len(tables.mutations.time), dtype=np.float64)
        with self.assertRaisesRegex(
            _tskit.LibraryError,
            "A mutation's time must be >= the node time, or be marked as 'unknown'",
        ):
            tables.tree_sequence()

    def test_older_than_node_above(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        tables.mutations.time = (
            np.ones(len(tables.mutations.time), dtype=np.float64) * 42
        )
        with self.assertRaisesRegex(
            _tskit.LibraryError,
            "A mutation's time must be < the parent node of the edge on which it"
            " occurs, or be marked as 'unknown'",
        ):
            tables.tree_sequence()

    def test_older_than_parent(self):
        ts = msprime.simulate(
            10, random_seed=42, mutation_rate=0.0, recombination_rate=1.0
        )
        ts = tsutil.jukes_cantor(
            ts, num_sites=10, mu=1, multiple_per_node=False, seed=42
        )
        tables = ts.dump_tables()
        self.assertNotEqual(sum(tables.mutations.parent != -1), 0)
        as_dict = tables.mutations.asdict()
        as_dict["time"][tables.mutations.parent != -1] = 64
        tables.mutations.set_columns(**as_dict)
        with self.assertRaisesRegex(
            _tskit.LibraryError,
            "A mutation's time must be < the parent node of the edge on which it"
            " occurs, or be marked as 'unknown'",
        ):
            tables.tree_sequence()


class TestNanDoubleValues(unittest.TestCase):
    """
    In some tables we need to guard against NaN/infinite values in the input.
    """

    def test_edge_coords(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)

        tables = ts.dump_tables()
        bad_coords = tables.edges.left + float("inf")
        tables.edges.left = bad_coords
        self.assertRaises(_tskit.LibraryError, tables.tree_sequence)

        tables = ts.dump_tables()
        bad_coords = tables.edges.right + float("nan")
        tables.edges.right = bad_coords
        self.assertRaises(_tskit.LibraryError, tables.tree_sequence)

    def test_migrations(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)

        tables = ts.dump_tables()
        tables.populations.add_row()
        tables.migrations.add_row(float("inf"), 1, time=0, node=0, source=0, dest=1)
        self.assertRaises(_tskit.LibraryError, tables.tree_sequence)

        tables = ts.dump_tables()
        tables.populations.add_row()
        tables.migrations.add_row(0, float("nan"), time=0, node=0, source=0, dest=1)
        self.assertRaises(_tskit.LibraryError, tables.tree_sequence)

        tables = ts.dump_tables()
        tables.populations.add_row()
        tables.migrations.add_row(0, 1, time=float("nan"), node=0, source=0, dest=1)
        self.assertRaises(_tskit.LibraryError, tables.tree_sequence)

    def test_site_positions(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        bad_pos = tables.sites.position.copy()
        bad_pos[-1] = np.inf
        tables.sites.position = bad_pos
        self.assertRaises(_tskit.LibraryError, tables.tree_sequence)

    def test_node_times(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        bad_times = tables.nodes.time.copy()
        bad_times[-1] = np.inf
        tables.nodes.time = bad_times
        with self.assertRaisesRegex(_tskit.LibraryError, "Times must be finite"):
            tables.tree_sequence()
        bad_times[-1] = math.nan
        tables.nodes.time = bad_times
        with self.assertRaisesRegex(_tskit.LibraryError, "Times must be finite"):
            tables.tree_sequence()

    def test_mutation_times(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=42)
        tables = ts.dump_tables()
        bad_times = tables.mutations.time.copy()
        bad_times[-1] = np.inf
        tables.mutations.time = bad_times
        with self.assertRaisesRegex(_tskit.LibraryError, "Times must be finite"):
            tables.tree_sequence()
        bad_times = tables.mutations.time.copy()
        bad_times[-1] = math.nan
        tables.mutations.time = bad_times
        with self.assertRaisesRegex(_tskit.LibraryError, "Times must be finite"):
            tables.tree_sequence()

    def test_individual(self):
        ts = msprime.simulate(12, mutation_rate=1, random_seed=42)
        ts = tsutil.insert_random_ploidy_individuals(ts, seed=42)
        self.assertGreater(ts.num_individuals, 1)
        tables = ts.dump_tables()
        bad_locations = tables.individuals.location.copy()
        bad_locations[0] = np.inf
        tables.individuals.location = bad_locations
        ts = tables.tree_sequence()


class TestSimplifyTables(unittest.TestCase):
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
            assert issubclass(w[-1].category, DeprecationWarning)

    def test_zero_mutation_sites(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=self.random_seed)
        for filter_sites in [True, False]:
            t1 = ts.dump_tables()
            t1.simplify([0, 1], filter_zero_mutation_sites=filter_sites)
            t2 = ts.dump_tables()
            t2.simplify([0, 1], filter_sites=filter_sites)
            t1.provenances.clear()
            t2.provenances.clear()
            self.assertEqual(t1, t2)
            if filter_sites:
                self.assertGreater(ts.num_sites, len(t1.sites))

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
                self.assertEqual(node_map.shape, (len(nodes_before),))
                self.assertEqual(nodes_before, tables.nodes)
                self.assertEqual(edges_before, tables.edges)
                self.assertEqual(sites_before, tables.sites)
                self.assertEqual(mutations_before, tables.mutations)

    def test_bad_samples(self):
        n = 10
        ts = msprime.simulate(n, random_seed=self.random_seed)
        tables = ts.dump_tables()
        for bad_node in [-1, n, n + 1, ts.num_nodes - 1, ts.num_nodes, 2 ** 31 - 1]:
            self.assertRaises(
                _tskit.LibraryError, tables.simplify, samples=[0, bad_node]
            )

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
        self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])

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
            self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])
            # Bad child node
            tables = ts.dump_tables()
            edges = tables.edges
            child = edges.child
            child[0] = bad_node
            edges.set_columns(
                left=edges.left, right=edges.right, parent=edges.parent, child=child
            )
            self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])
            # child == parent
            tables = ts.dump_tables()
            edges = tables.edges
            child = edges.child
            child[0] = edges.parent[0]
            edges.set_columns(
                left=edges.left, right=edges.right, parent=edges.parent, child=child
            )
            self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])
            # left == right
            tables = ts.dump_tables()
            edges = tables.edges
            left = edges.left
            left[0] = edges.right[0]
            edges.set_columns(
                left=left, right=edges.right, parent=edges.parent, child=edges.child
            )
            self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])
            # left > right
            tables = ts.dump_tables()
            edges = tables.edges
            left = edges.left
            left[0] = edges.right[0] + 1
            edges.set_columns(
                left=left, right=edges.right, parent=edges.parent, child=edges.child
            )
            self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])

    def test_bad_mutation_nodes(self):
        ts = msprime.simulate(10, random_seed=self.random_seed, mutation_rate=1)
        self.assertGreater(ts.num_mutations, 0)
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
            self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])

    def test_bad_mutation_sites(self):
        ts = msprime.simulate(10, random_seed=self.random_seed, mutation_rate=1)
        self.assertGreater(ts.num_mutations, 0)
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
            self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])

    def test_bad_site_positions(self):
        ts = msprime.simulate(10, random_seed=self.random_seed, mutation_rate=1)
        self.assertGreater(ts.num_mutations, 0)
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
            self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])

    def test_duplicate_positions(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.sites.add_row(0, ancestral_state="0")
        tables.sites.add_row(0, ancestral_state="0")
        self.assertRaises(_tskit.LibraryError, tables.simplify, [])

    def test_samples_interface(self):
        ts = msprime.simulate(50, random_seed=1)
        for good_form in [[], [0, 1], (0, 1), np.array([0, 1], dtype=np.int32)]:
            tables = ts.dump_tables()
            tables.simplify(good_form)
        tables = ts.dump_tables()
        for bad_values in [[[[]]], np.array([[0, 1], [2, 3]], dtype=np.int32)]:
            self.assertRaises(ValueError, tables.simplify, bad_values)
        for bad_type in [[0.1], ["string"], {}, [{}]]:
            self.assertRaises(TypeError, tables.simplify, bad_type)
        # We only convert to int if we don't overflow
        for bad_node in [np.iinfo(np.int32).min - 1, np.iinfo(np.int32).max + 1]:
            self.assertRaises(
                OverflowError, tables.simplify, samples=np.array([0, bad_node])
            )


class TestTableCollection(unittest.TestCase):
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
        self.assertEqual(str(individuals), before_individuals)
        self.assertEqual(str(nodes), before_nodes)
        self.assertEqual(str(edges), before_edges)
        self.assertEqual(str(migrations), before_migrations)
        self.assertEqual(str(sites), before_sites)
        self.assertEqual(str(mutations), before_mutations)
        self.assertEqual(str(populations), before_populations)
        self.assertEqual(str(provenances), before_provenances)

    def test_str(self):
        ts = msprime.simulate(10, random_seed=1)
        tables = ts.tables
        s = str(tables)
        self.assertGreater(len(s), 0)

    def test_asdict(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        t = ts.tables
        self.add_metadata(t)
        d1 = {
            "encoding_version": (1, 1),
            "sequence_length": t.sequence_length,
            "metadata_schema": str(t.metadata_schema),
            "metadata": t.metadata_schema.encode_row(t.metadata),
            "individuals": t.individuals.asdict(),
            "populations": t.populations.asdict(),
            "nodes": t.nodes.asdict(),
            "edges": t.edges.asdict(),
            "sites": t.sites.asdict(),
            "mutations": t.mutations.asdict(),
            "migrations": t.migrations.asdict(),
            "provenances": t.provenances.asdict(),
        }
        d2 = t.asdict()
        self.assertEqual(set(d1.keys()), set(d2.keys()))
        t1 = tskit.TableCollection.fromdict(d1)
        t2 = tskit.TableCollection.fromdict(d2)
        self.assertEqual(t1, t2)

    def test_from_dict(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        t1 = ts.tables
        self.add_metadata(t1)
        d = {
            "encoding_version": (1, 1),
            "sequence_length": t1.sequence_length,
            "metadata_schema": str(t1.metadata_schema),
            "metadata": t1.metadata_schema.encode_row(t1.metadata),
            "individuals": t1.individuals.asdict(),
            "populations": t1.populations.asdict(),
            "nodes": t1.nodes.asdict(),
            "edges": t1.edges.asdict(),
            "sites": t1.sites.asdict(),
            "mutations": t1.mutations.asdict(),
            "migrations": t1.migrations.asdict(),
            "provenances": t1.provenances.asdict(),
        }
        t2 = tskit.TableCollection.fromdict(d)
        self.assertEquals(t1, t2)

    def test_roundtrip_dict(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        t1 = ts.tables
        t2 = tskit.TableCollection.fromdict(t1.asdict())
        self.assertEqual(t1, t2)

        self.add_metadata(t1)
        t2 = tskit.TableCollection.fromdict(t1.asdict())
        self.assertEqual(t1, t2)

    def test_iter(self):
        def test_iter(table_collection):
            table_names = [
                attr_name
                for attr_name in sorted(dir(table_collection))
                if isinstance(getattr(table_collection, attr_name), tskit.BaseTable)
            ]
            for n in table_names:
                yield n, getattr(table_collection, n)

        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        for t1, t2 in itertools.zip_longest(test_iter(ts.tables), ts.tables):
            self.assertEquals(t1, t2)

    def test_equals_empty(self):
        self.assertEqual(tskit.TableCollection(), tskit.TableCollection())

    def test_equals_sequence_length(self):
        self.assertNotEqual(
            tskit.TableCollection(sequence_length=1),
            tskit.TableCollection(sequence_length=2),
        )

    def test_copy(self):
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        t1 = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=100,
        ).dump_tables()
        t2 = t1.copy()
        self.assertIsNot(t1, t2)
        self.assertEqual(t1, t2)
        t1.edges.clear()
        self.assertNotEqual(t1, t2)

    def test_equals(self):
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
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t1.copy())
        self.assertEqual(t1.copy(), t1)

        # The provenances may or may not be equal depending on the clock
        # precision for record. So clear them first.
        t1.provenances.clear()
        t2.provenances.clear()
        self.assertEqual(t1, t2)
        self.assertTrue(t1 == t2)
        self.assertFalse(t1 != t2)

        t1.nodes.clear()
        self.assertNotEqual(t1, t2)
        t2.nodes.clear()
        self.assertEqual(t1, t2)

        t1.edges.clear()
        self.assertNotEqual(t1, t2)
        t2.edges.clear()
        self.assertEqual(t1, t2)

        t1.migrations.clear()
        self.assertNotEqual(t1, t2)
        t2.migrations.clear()
        self.assertEqual(t1, t2)

        t1.sites.clear()
        self.assertNotEqual(t1, t2)
        t2.sites.clear()
        self.assertEqual(t1, t2)

        t1.mutations.clear()
        self.assertNotEqual(t1, t2)
        t2.mutations.clear()
        self.assertEqual(t1, t2)

        t1.populations.clear()
        self.assertNotEqual(t1, t2)
        t2.populations.clear()
        self.assertEqual(t1, t2)

    def test_sequence_length(self):
        for sequence_length in [0, 1, 100.1234]:
            tables = tskit.TableCollection(sequence_length=sequence_length)
            self.assertEqual(tables.sequence_length, sequence_length)

    def test_uuid_simulation(self):
        ts = msprime.simulate(10, random_seed=1)
        tables = ts.tables
        self.assertIsNone(tables.file_uuid, None)

    def test_uuid_empty(self):
        tables = tskit.TableCollection(sequence_length=1)
        self.assertIsNone(tables.file_uuid, None)

    def test_empty_indexes(self):
        tables = tskit.TableCollection(sequence_length=1)
        self.assertFalse(tables.has_index())
        tables.build_index()
        self.assertTrue(tables.has_index())
        tables.drop_index()
        self.assertFalse(tables.has_index())

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

        self.assertFalse(tables.has_index())
        with self.assertRaises(tskit.LibraryError):
            tables.build_index()
        self.assertFalse(tables.has_index())
        tables.sort()
        tables.build_index()
        self.assertTrue(tables.has_index())
        ts = tables.tree_sequence()
        self.assertEqual(ts.tables, tables)

    def test_index_from_ts(self):
        ts = msprime.simulate(10, random_seed=1)
        tables = ts.dump_tables()
        self.assertTrue(tables.has_index())
        tables.drop_index()
        self.assertFalse(tables.has_index())
        ts = tables.tree_sequence()
        self.assertEqual(ts.tables, tables)
        self.assertFalse(tables.has_index())

    def test_set_sequence_length_errors(self):
        tables = tskit.TableCollection(1)
        with self.assertRaises(AttributeError):
            del tables.sequence_length
        for bad_value in ["asdf", None, []]:
            with self.assertRaises(TypeError):
                tables.sequence_length = bad_value

    def test_set_sequence_length(self):
        tables = tskit.TableCollection(1)
        for value in [-1, 100, 2 ** 32, 1e-6]:
            tables.sequence_length = value
            self.assertEqual(tables.sequence_length, value)

    def test_bad_sequence_length(self):
        tables = msprime.simulate(10, random_seed=1).dump_tables()
        self.assertEqual(tables.sequence_length, 1)
        for value in [-1, 0, -0.99, 0.9999]:
            tables.sequence_length = value
            with self.assertRaises(tskit.LibraryError):
                tables.tree_sequence()
            with self.assertRaises(tskit.LibraryError):
                tables.sort()
            with self.assertRaises(tskit.LibraryError):
                tables.build_index()
            with self.assertRaises(tskit.LibraryError):
                tables.compute_mutation_parents()
            with self.assertRaises(tskit.LibraryError):
                tables.simplify()
            self.assertEqual(tables.sequence_length, value)

    def test_sequence_length_longer_than_edges(self):
        tables = msprime.simulate(10, random_seed=1).dump_tables()
        tables.sequence_length = 2
        ts = tables.tree_sequence()
        self.assertEqual(ts.sequence_length, 2)
        self.assertEqual(ts.num_trees, 2)
        trees = ts.trees()
        tree = next(trees)
        self.assertGreater(len(tree.parent_dict), 0)
        tree = next(trees)
        self.assertEqual(len(tree.parent_dict), 0)


class TestTableCollectionMetadata(unittest.TestCase):

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
        self.assertEqual(str(tc.metadata_schema), str(metadata.MetadataSchema(None)))
        # Set
        tc.metadata_schema = self.metadata_schema
        self.assertEqual(str(tc.metadata_schema), str(self.metadata_schema))
        # Overwrite
        tc.metadata_schema = metadata_schema2
        self.assertEqual(str(tc.metadata_schema), str(metadata_schema2))
        # Remove
        tc.metadata_schema = ""
        self.assertEqual(str(tc.metadata_schema), str(metadata.MetadataSchema(None)))
        # Set after remove
        tc.metadata_schema = self.metadata_schema
        self.assertEqual(str(tc.metadata_schema), str(self.metadata_schema))
        # Del should fail
        with self.assertRaises(AttributeError):
            del tc.metadata_schema
        # None should fail
        with self.assertRaises(ValueError):
            tc.metadata_schema = None

    def test_set_metadata(self):
        tc = tskit.TableCollection(1)
        # Default is empty bytes
        self.assertEqual(tc.metadata, b"")

        tc.metadata_schema = self.metadata_schema
        md1 = self.metadata_example_data()
        md2 = self.metadata_example_data(val=2)
        # Set
        tc.metadata = md1
        self.assertEqual(tc.metadata, md1)
        # Overwrite
        tc.metadata = md2
        self.assertEqual(tc.metadata, md2)
        # Del should fail
        with self.assertRaises(AttributeError):
            del tc.metadata
        # None should fail
        with self.assertRaises(exceptions.MetadataValidationError):
            tc.metadata = None

    def test_default_metadata_schema(self):
        # Default should allow bytes
        tc = tskit.TableCollection(1)
        tc.metadata = b"acceptable bytes"
        self.assertEqual(tc.metadata, b"acceptable bytes")
        # Adding non-bytes metadata should error
        with self.assertRaises(TypeError):
            tc.metadata = self.metadata_example_data()

    def test_round_trip_metadata(self):
        data = self.metadata_example_data()
        tc = tskit.TableCollection(1)
        tc.metadata_schema = self.metadata_schema
        tc.metadata = data
        self.assertDictEqual(tc.metadata, data)

    def test_bad_metadata(self):
        metadata = self.metadata_example_data()
        metadata["I really shouldn't be here"] = 6
        tc = tskit.TableCollection(1)
        tc.metadata_schema = self.metadata_schema
        with self.assertRaises(exceptions.MetadataValidationError):
            tc.metadata = metadata
        self.assertEqual(tc.ll_tables.metadata, b"")


class TestTableCollectionPickle(TestTableCollection):
    """
    Tests that we can round-trip table collections through pickle.
    """

    def verify(self, tables):
        self.add_metadata(tables)
        other_tables = pickle.loads(pickle.dumps(tables))
        self.assertEqual(tables, other_tables)

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
        self.assertGreater(ts.num_sites, 1)
        self.verify(ts.dump_tables())

    def test_simulation_individuals(self):
        ts = msprime.simulate(100, random_seed=1)
        ts = tsutil.insert_random_ploidy_individuals(ts, seed=1)
        self.assertGreater(ts.num_individuals, 1)
        self.verify(ts.dump_tables())

    def test_empty_tables(self):
        self.verify(tskit.TableCollection())


class TestDeduplicateSites(unittest.TestCase):
    """
    Tests for the TableCollection.deduplicate_sites method.
    """

    def test_empty(self):
        tables = tskit.TableCollection(1)
        tables.deduplicate_sites()
        self.assertEqual(tables, tskit.TableCollection(1))

    def test_unsorted(self):
        tables = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        self.assertGreater(len(tables.sites), 0)
        position = tables.sites.position
        for _ in range(len(position) - 1):
            position = np.roll(position, 1)
            tables.sites.set_columns(
                position=position,
                ancestral_state=tables.sites.ancestral_state,
                ancestral_state_offset=tables.sites.ancestral_state_offset,
            )
            self.assertRaises(_tskit.LibraryError, tables.deduplicate_sites)

    def test_bad_position(self):
        for bad_position in [-1, -0.001]:
            tables = tskit.TableCollection()
            tables.sites.add_row(bad_position, "0")
            self.assertRaises(_tskit.LibraryError, tables.deduplicate_sites)

    def test_no_effect(self):
        t1 = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        t2 = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        self.assertGreater(len(t1.sites), 0)
        t1.deduplicate_sites()
        t1.provenances.clear()
        t2.provenances.clear()
        self.assertEqual(t1, t2)

    def test_same_sites(self):
        t1 = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        t2 = msprime.simulate(10, mutation_rate=1, random_seed=1).dump_tables()
        self.assertGreater(len(t1.sites), 0)
        t1.sites.append_columns(
            position=t1.sites.position,
            ancestral_state=t1.sites.ancestral_state,
            ancestral_state_offset=t1.sites.ancestral_state_offset,
        )
        self.assertEqual(len(t1.sites), 2 * len(t2.sites))
        t1.sort()
        t1.deduplicate_sites()
        t1.provenances.clear()
        t2.provenances.clear()
        self.assertEqual(t1, t2)

    def test_order_maintained(self):
        t1 = tskit.TableCollection(1)
        t1.sites.add_row(position=0, ancestral_state="first")
        t1.sites.add_row(position=0, ancestral_state="second")
        t1.deduplicate_sites()
        self.assertEqual(len(t1.sites), 1)
        self.assertEqual(t1.sites.ancestral_state.tobytes(), b"first")

    def test_multichar_ancestral_state(self):
        ts = msprime.simulate(8, random_seed=3, mutation_rate=1)
        self.assertGreater(ts.num_sites, 2)
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
                    site=site_id, node=mutation.node, derived_state="T" * site.id,
                )
        tables.deduplicate_sites()
        new_ts = tables.tree_sequence()
        self.assertEqual(new_ts.num_sites, ts.num_sites)
        for site in new_ts.sites():
            self.assertEqual(site.ancestral_state, site.id * "A")

    def test_multichar_metadata(self):
        ts = msprime.simulate(8, random_seed=3, mutation_rate=1)
        self.assertGreater(ts.num_sites, 2)
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
        self.assertEqual(new_ts.num_sites, ts.num_sites)
        for site in new_ts.sites():
            self.assertEqual(site.metadata, site.id * b"A")


class TestBaseTable(unittest.TestCase):
    """
    Tests of the table superclass.
    """

    def test_set_columns_not_implemented(self):
        t = tskit.BaseTable(None, None)
        with self.assertRaises(NotImplementedError):
            t.set_columns()


class TestSubsetTables(unittest.TestCase):
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
        tables = wf.wf_sim(N, N, seed=seed)
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
        sub1 = tables.copy()
        sub2 = tables.copy()
        tsutil.py_subset(sub1, nodes, record_provenance=False)
        sub2.subset(nodes, record_provenance=False)
        self.assertEqual(sub1, sub2)

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
        self.assertEqual(subset.nodes.num_rows, len(nodes))
        for k, n in zip(nodes, subset.nodes):
            nn = tables.nodes[k]
            self.assertEqual(nn.time, n.time)
            self.assertEqual(nn.flags, n.flags)
            self.assertEqual(nn.metadata, n.metadata)
            self.assertEqual(ind_map[nn.individual], n.individual)
            self.assertEqual(pop_map[nn.population], n.population)
        self.assertEqual(subset.individuals.num_rows, len(indivs))
        for k, i in zip(indivs, subset.individuals):
            ii = tables.individuals[k]
            self.assertEqual(ii, i)
        self.assertEqual(subset.populations.num_rows, len(pops))
        for k, p in zip(pops, subset.populations):
            pp = tables.populations[k]
            self.assertEqual(pp, p)
        edges = [
            i
            for i, e in enumerate(tables.edges)
            if e.parent in nodes and e.child in nodes
        ]
        self.assertEqual(subset.edges.num_rows, len(edges))
        for k, e in zip(edges, subset.edges):
            ee = tables.edges[k]
            self.assertEqual(ee.left, e.left)
            self.assertEqual(ee.right, e.right)
            self.assertEqual(node_map[ee.parent], e.parent)
            self.assertEqual(node_map[ee.child], e.child)
            self.assertEqual(ee.metadata, e.metadata)
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
        self.assertEqual(subset.sites.num_rows, len(sites))
        for k, s in zip(sites, subset.sites):
            ss = tables.sites[k]
            self.assertEqual(ss, s)
        self.assertEqual(subset.mutations.num_rows, len(muts))
        for k, m in zip(muts, subset.mutations):
            mm = tables.mutations[k]
            self.assertEqual(mutation_map[mm.parent], m.parent)
            self.assertEqual(site_map[mm.site], m.site)
            self.assertEqual(node_map[mm.node], m.node)
            self.assertEqual(mm.derived_state, m.derived_state)
            self.assertEqual(mm.metadata, m.metadata)
        self.assertEqual(tables.migrations, subset.migrations)
        self.assertEqual(tables.provenances, subset.provenances)

    def test_ts_subset(self):
        nodes = np.array([0, 1])
        for tables in self.get_examples(83592):
            ts = tables.tree_sequence()
            tables2 = ts.subset(nodes, record_provenance=False).dump_tables()
            tables.subset(nodes, record_provenance=False)
            self.assertEqual(tables, tables2)

    def test_subset_all(self):
        # subsetting to everything shouldn't change things
        # except the individual ids in the node tables if
        # there are gaps
        for tables in self.get_examples(123583):
            tables2 = tables.copy()
            tables2.subset(np.arange(tables.nodes.num_rows))
            tables.provenances.clear()
            tables2.provenances.clear()
            tables.individuals.clear()
            tables2.individuals.clear()
            tables.nodes.clear()
            tables2.nodes.clear()
            self.assertEqual(tables, tables2)

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
            self.assertEqual(subset.nodes.num_rows, 0)
            self.assertEqual(subset.edges.num_rows, 0)
            self.assertEqual(subset.populations.num_rows, 0)
            self.assertEqual(subset.individuals.num_rows, 0)
            self.assertEqual(subset.migrations.num_rows, 0)
            self.assertEqual(subset.sites.num_rows, 0)
            self.assertEqual(subset.mutations.num_rows, 0)
            self.assertEqual(subset.provenances, tables.provenances)
