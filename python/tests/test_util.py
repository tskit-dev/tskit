# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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
Tests for functions in util.py
"""
import itertools
import pickle
import unittest

import numpy as np

import tests.tsutil as tsutil
import tskit.util as util


class TestNumpyArrayCasting(unittest.TestCase):
    """
    Tests that the safe_np_int_cast() function works.
    """

    dtypes_to_test = [np.int32, np.uint32, np.int8, np.uint8]

    def test_basic_arrays(self):
        # Simple array
        for dtype in self.dtypes_to_test:
            target = np.array([0, 1], dtype=dtype)
            for test_array in [[0, 1], (0, 1), np.array([0, 1]), target]:
                converted = util.safe_np_int_cast(test_array, dtype=dtype)
                # Use pickle to test exact equality including dtype
                self.assertEqual(pickle.dumps(converted), pickle.dumps(target))
            # Nested array
            target = np.array([[0, 1], [2, 3]], dtype=dtype)
            for test_array in [[[0, 1], [2, 3]], np.array([[0, 1], [2, 3]]), target]:
                converted = util.safe_np_int_cast(test_array, dtype=dtype)
                self.assertEqual(pickle.dumps(converted), pickle.dumps(target))

    def test_copy(self):
        # Check that a copy is not returned if copy=False & the original matches
        # the specs
        for dtype in self.dtypes_to_test:
            for orig in (np.array([0, 1], dtype=dtype), np.array([], dtype=dtype)):
                converted = util.safe_np_int_cast(orig, dtype=dtype, copy=True)
                self.assertNotEqual(id(orig), id(converted))
                converted = util.safe_np_int_cast(orig, dtype=dtype, copy=False)
                self.assertEqual(id(orig), id(converted))
        for dtype in [d for d in self.dtypes_to_test if d != np.int64]:
            # non numpy arrays, or arrays of a different dtype don't get converted
            for orig in ([0, 1], np.array([0, 1], dtype=np.int64)):
                converted = util.safe_np_int_cast(orig, dtype=dtype, copy=False)
                self.assertNotEqual(id(orig), id(converted))

    def test_empty_arrays(self):
        # Empty arrays of any type (including float) should be allowed
        for dtype in self.dtypes_to_test:
            target = np.array([], dtype=dtype)
            converted = util.safe_np_int_cast([], dtype=dtype)
            self.assertEqual(pickle.dumps(converted), pickle.dumps(target))
            target = np.array([[]], dtype=dtype)
            converted = util.safe_np_int_cast([[]], dtype=dtype)
            self.assertEqual(pickle.dumps(converted), pickle.dumps(target))

    def test_bad_types(self):
        # Shouldn't be able to convert a float (possibility of rounding error)
        for dtype in self.dtypes_to_test:
            for bad_type in [
                [0.1],
                ["str"],
                {},
                [{}],
                np.array([0, 1], dtype=np.float),
            ]:
                self.assertRaises(TypeError, util.safe_np_int_cast, bad_type, dtype)

    def test_overflow(self):
        for dtype in self.dtypes_to_test:
            for bad_node in [np.iinfo(dtype).min - 1, np.iinfo(dtype).max + 1]:
                self.assertRaises(  # Test plain array
                    OverflowError, util.safe_np_int_cast, [0, bad_node], dtype
                )
                self.assertRaises(  # Test numpy array
                    OverflowError, util.safe_np_int_cast, np.array([0, bad_node]), dtype
                )
            for good_node in [np.iinfo(dtype).min, np.iinfo(dtype).max]:
                target = np.array([good_node], dtype=dtype)
                self.assertEqual(  # Test plain array
                    pickle.dumps(target),
                    pickle.dumps(util.safe_np_int_cast([good_node], dtype)),
                )
                self.assertEqual(  # Test numpy array
                    pickle.dumps(target),
                    pickle.dumps(util.safe_np_int_cast(np.array([good_node]), dtype)),
                )

    def test_nonrectangular_input(self):
        bad_inputs = [
            [0, 1, [2]],
            [[0, 1, 2], []],
            [(0, 1, 2), [2, 3]],
            [(0, 1, 2), tuple()],
            [(0, 1, 2), (2,)],
            [(0, 1, 2), [2, 3]],
        ]
        for dtype in self.dtypes_to_test:
            for bad_input in bad_inputs:
                with self.assertRaises(TypeError):
                    util.safe_np_int_cast(bad_input, dtype)


class TestIntervalOps(unittest.TestCase):
    """
    Test cases for the interval operations used in masks and slicing operations.
    """

    def test_bad_intervals(self):
        for bad_type in [{}, Exception]:
            with self.assertRaises(TypeError):
                util.intervals_to_np_array(bad_type, 0, 1)
        for bad_depth in [[[[]]]]:
            with self.assertRaises(ValueError):
                util.intervals_to_np_array(bad_depth, 0, 1)
        for bad_shape in [[[0], [0]], [[[0, 1, 2], [0, 1]]]]:
            with self.assertRaises(ValueError):
                util.intervals_to_np_array(bad_shape, 0, 1)

        # Out of bounds
        with self.assertRaises(ValueError):
            util.intervals_to_np_array([[-1, 0]], 0, 1)
        with self.assertRaises(ValueError):
            util.intervals_to_np_array([[0, 1]], 1, 2)
        with self.assertRaises(ValueError):
            util.intervals_to_np_array([[0, 1]], 0, 0.5)

        # Overlapping intervals
        with self.assertRaises(ValueError):
            util.intervals_to_np_array([[0, 1], [0.9, 2.0]], 0, 10)

        # Empty intervals
        for bad_interval in [[0, 0], [1, 0]]:
            with self.assertRaises(ValueError):
                util.intervals_to_np_array([bad_interval], 0, 10)

    def test_empty_interval_list(self):
        intervals = util.intervals_to_np_array([], 0, 10)
        self.assertEqual(len(intervals), 0)

    def test_negate_intervals(self):
        L = 10
        cases = [
            ([], [[0, L]]),
            ([[0, 5], [6, L]], [[5, 6]]),
            ([[0, 5]], [[5, L]]),
            ([[5, L]], [[0, 5]]),
            ([[0, 1], [2, 3], [3, 4], [5, 6]], [[1, 2], [4, 5], [6, L]]),
        ]
        for source, dest in cases:
            self.assertTrue(np.array_equal(util.negate_intervals(source, 0, L), dest))


class TestStringPacking(unittest.TestCase):
    """
    Tests the code for packing and unpacking unicode string data into numpy arrays.
    """

    def test_simple_string_case(self):
        strings = ["hello", "world"]
        packed, offset = util.pack_strings(strings)
        self.assertEqual(list(offset), [0, 5, 10])
        self.assertEqual(packed.shape, (10,))
        returned = util.unpack_strings(packed, offset)
        self.assertEqual(returned, strings)

    def verify_packing(self, strings):
        packed, offset = util.pack_strings(strings)
        self.assertEqual(packed.dtype, np.int8)
        self.assertEqual(offset.dtype, np.uint32)
        self.assertEqual(packed.shape[0], offset[-1])
        returned = util.unpack_strings(packed, offset)
        self.assertEqual(strings, returned)

    def test_regular_cases(self):
        for n in range(10):
            strings = ["a" * j for j in range(n)]
            self.verify_packing(strings)

    def test_random_cases(self):
        for n in range(100):
            strings = [tsutil.random_strings(10) for _ in range(n)]
            self.verify_packing(strings)

    def test_unicode(self):
        self.verify_packing(["abcdé", "€"])


class TestBytePacking(unittest.TestCase):
    """
    Tests the code for packing and unpacking binary data into numpy arrays.
    """

    def test_simple_string_case(self):
        strings = [b"hello", b"world"]
        packed, offset = util.pack_bytes(strings)
        self.assertEqual(list(offset), [0, 5, 10])
        self.assertEqual(packed.shape, (10,))
        returned = util.unpack_bytes(packed, offset)
        self.assertEqual(returned, strings)

    def verify_packing(self, data):
        packed, offset = util.pack_bytes(data)
        self.assertEqual(packed.dtype, np.int8)
        self.assertEqual(offset.dtype, np.uint32)
        self.assertEqual(packed.shape[0], offset[-1])
        returned = util.unpack_bytes(packed, offset)
        self.assertEqual(data, returned)
        return returned

    def test_random_cases(self):
        for n in range(100):
            data = [tsutil.random_bytes(10) for _ in range(n)]
            self.verify_packing(data)

    def test_pickle_packing(self):
        data = [list(range(j)) for j in range(10)]
        # Pickle each of these in turn
        pickled = [pickle.dumps(d) for d in data]
        unpacked = self.verify_packing(pickled)
        unpickled = [pickle.loads(p) for p in unpacked]
        self.assertEqual(data, unpickled)


class TestArrayPacking(unittest.TestCase):
    """
    Tests the code for packing and unpacking numpy data into numpy arrays.
    """

    def test_simple_case(self):
        lists = [[0], [1.125, 1.25]]
        packed, offset = util.pack_arrays(lists)
        self.assertEqual(list(offset), [0, 1, 3])
        self.assertEqual(list(packed), [0, 1.125, 1.25])
        returned = util.unpack_arrays(packed, offset)
        for a1, a2 in itertools.zip_longest(lists, returned):
            self.assertEqual(a1, list(a2))

    def verify_packing(self, data):
        packed, offset = util.pack_arrays(data)
        self.assertEqual(packed.dtype, np.float64)
        self.assertEqual(offset.dtype, np.uint32)
        self.assertEqual(packed.shape[0], offset[-1])
        returned = util.unpack_arrays(packed, offset)
        for a1, a2 in itertools.zip_longest(data, returned):
            self.assertTrue(np.array_equal(a1, a2))
        return returned

    def test_regular_cases(self):
        for n in range(100):
            data = [np.arange(n) for _ in range(n)]
            self.verify_packing(data)
            data = [1 / (1 + np.arange(n)) for _ in range(n)]
            self.verify_packing(data)
