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
Module responsible for various utility functions used in other modules.
"""

import numpy as np


def safe_np_int_cast(int_array, dtype, copy=False):
    """
    A few functions require arrays of certain dtypes (e.g. node indices are np.int32,
    genotypes are np.int8, etc. Standard numpy integer arrays are of (dtype=np.int64),
    so need casting. This function casts but checks bounds to avoid wrap-around
    conversion errors. Strangely, numpy seems not to have this functionality built-in.

    If copy=False, and the original array is a numpy array of exactly the same dtype
    required, simply return the original rather than making a copy (same as the numpy
    .astype(copy=...) function)
    """
    if not isinstance(int_array, np.ndarray):
        int_array = np.array(int_array)
        # Since this is a new numpy array anyway, it's always a copy, so economize by
        # setting copy=False
        copy = False
    if int_array.size == 0:
        return int_array.astype(dtype, copy=copy)  # Allow empty arrays of any type
    try:
        return int_array.astype(dtype, casting='safe', copy=copy)
    except TypeError:
        bounds = np.iinfo(dtype)
        if np.any(int_array < bounds.min) or np.any(int_array > bounds.max):
            raise OverflowError("Cannot convert safely to {} type".format(dtype))
        if int_array.dtype.kind == 'i' and np.dtype(dtype).kind == 'u':
            # Allow casting from int to unsigned int, since we have checked bounds
            casting = 'unsafe'
        else:
            # Raise a TypeError when we try to convert from, e.g., a float.
            casting = 'same_kind'
        return int_array.astype(dtype, casting=casting, copy=copy)
