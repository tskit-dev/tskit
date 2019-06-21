# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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
Tests for functions in util.py
"""
import unittest
import pickle

import numpy as np
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
        """
        Check that a copy is not returned if copy=False & the original matches the specs
        """
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
        """
        Empty arrays of any type (including float) should be allowed
        """
        for dtype in self.dtypes_to_test:
            target = np.array([], dtype=dtype)
            converted = util.safe_np_int_cast([], dtype=dtype)
            self.assertEqual(pickle.dumps(converted), pickle.dumps(target))
            target = np.array([[]], dtype=dtype)
            converted = util.safe_np_int_cast([[]], dtype=dtype)
            self.assertEqual(pickle.dumps(converted), pickle.dumps(target))

    def test_bad_types(self):
        """
        Shouldn't be able to convert a float (possibility of rounding error)
        """
        for dtype in self.dtypes_to_test:
            for bad_type in [[0.1], ['str'], {}, [{}], np.array([0, 1], dtype=np.float)]:
                self.assertRaises(TypeError, util.safe_np_int_cast, bad_type, dtype)

    def test_overflow(self):
        for dtype in self.dtypes_to_test:
            for bad_node in [np.iinfo(dtype).min - 1, np.iinfo(dtype).max + 1]:
                self.assertRaises(  # Test plain array
                    OverflowError, util.safe_np_int_cast, [0, bad_node], dtype)
                self.assertRaises(  # Test numpy array
                    OverflowError, util.safe_np_int_cast, np.array([0, bad_node]), dtype)
            for good_node in [np.iinfo(dtype).min, np.iinfo(dtype).max]:
                target = np.array([good_node], dtype=dtype)
                self.assertEqual(  # Test plain array
                    pickle.dumps(target),
                    pickle.dumps(util.safe_np_int_cast([good_node], dtype)))
                self.assertEqual(  # Test numpy array
                    pickle.dumps(target),
                    pickle.dumps(util.safe_np_int_cast(np.array([good_node]), dtype)))
