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
        if int_array.dtype == np.dtype('O'):
            # this occurs e.g. if we're passed a list of lists of different lengths
            raise TypeError("Cannot convert to a rectangular array.")
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


#
# Pack/unpack lists of data into flattened numpy arrays.
#

def pack_bytes(data):
    """
    Packs the specified list of bytes into a flattened numpy array of 8 bit integers
    and corresponding offsets. See :ref:`sec_encoding_ragged_columns` for details
    of this encoding.

    :param list[bytes] data: The list of bytes values to encode.
    :return: The tuple (packed, offset) of numpy arrays representing the flattened
        input data and offsets.
    :rtype: numpy.ndarray (dtype=np.int8), numpy.ndarray (dtype=np.uint32)
    """
    n = len(data)
    offsets = np.zeros(n + 1, dtype=np.uint32)
    for j in range(n):
        offsets[j + 1] = offsets[j] + len(data[j])
    column = np.zeros(offsets[-1], dtype=np.int8)
    for j, value in enumerate(data):
        column[offsets[j]: offsets[j + 1]] = bytearray(value)
    return column, offsets


def unpack_bytes(packed, offset):
    """
    Unpacks a list of bytes from the specified numpy arrays of packed byte
    data and corresponding offsets. See :ref:`sec_encoding_ragged_columns` for details
    of this encoding.

    :param numpy.ndarray packed: The flattened array of byte values.
    :param numpy.ndarray offset: The array of offsets into the ``packed`` array.
    :return: The list of bytes values unpacked from the parameter arrays.
    :rtype: list[bytes]
    """
    # This could be done a lot more efficiently...
    ret = []
    for j in range(offset.shape[0] - 1):
        raw = packed[offset[j]: offset[j + 1]].tobytes()
        ret.append(raw)
    return ret


def pack_strings(strings, encoding="utf8"):
    """
    Packs the specified list of strings into a flattened numpy array of 8 bit integers
    and corresponding offsets using the specified text encoding.
    See :ref:`sec_encoding_ragged_columns` for details of this encoding of
    columns of variable length data.

    :param list[str] data: The list of strings to encode.
    :param str encoding: The text encoding to use when converting string data
        to bytes. See the :mod:`codecs` module for information on available
        string encodings.
    :return: The tuple (packed, offset) of numpy arrays representing the flattened
        input data and offsets.
    :rtype: numpy.ndarray (dtype=np.int8), numpy.ndarray (dtype=np.uint32)
    """
    return pack_bytes([bytearray(s.encode(encoding)) for s in strings])


def unpack_strings(packed, offset, encoding="utf8"):
    """
    Unpacks a list of strings from the specified numpy arrays of packed byte
    data and corresponding offsets using the specified text encoding.
    See :ref:`sec_encoding_ragged_columns` for details of this encoding of
    columns of variable length data.

    :param numpy.ndarray packed: The flattened array of byte values.
    :param numpy.ndarray offset: The array of offsets into the ``packed`` array.
    :param str encoding: The text encoding to use when converting string data
        to bytes. See the :mod:`codecs` module for information on available
        string encodings.
    :return: The list of strings unpacked from the parameter arrays.
    :rtype: list[str]
    """
    return [b.decode(encoding) for b in unpack_bytes(packed, offset)]


def pack_arrays(list_of_lists):
    """
    Packs the specified list of numberic lists into a flattened numpy array of
    numpy float64 and corresponding offsets. See
    :ref:`sec_encoding_ragged_columns` for details of this encoding of columns
    of variable length data.

    :param list[list] list_of_lists: The list of numeric lists to encode.
    :return: The tuple (packed, offset) of numpy arrays representing the flattened
        input data and offsets.
    :rtype: numpy.array (dtype=np.float64), numpy.array (dtype=np.uint32)
    """
    # TODO must be possible to do this more efficiently with numpy
    n = len(list_of_lists)
    offset = np.zeros(n + 1, dtype=np.uint32)
    for j in range(n):
        offset[j + 1] = offset[j] + len(list_of_lists[j])
    data = np.empty(offset[-1], dtype=np.float64)
    for j in range(n):
        data[offset[j]: offset[j + 1]] = list_of_lists[j]
    return data, offset


def unpack_arrays(packed, offset):
    """
    Unpacks a list of arrays from the specified numpy arrays of packed
    data and corresponding offsets. See
    :ref:`sec_encoding_ragged_columns` for details of this encoding of columns
    of variable length data.

    :param numpy.ndarray packed: The flattened array of data.
    :param numpy.ndarray offset: The array of offsets into the ``packed`` array.
    :return: The list numpy arrays unpacked from the parameter arrays.
    :rtype: list[numpy.ndarray]
    """
    ret = []
    for j in range(offset.shape[0] - 1):
        ret.append(packed[offset[j]: offset[j + 1]])
    return ret


#
# Interval utilities
#

def intervals_to_np_array(intervals, start, end):
    """
    Converts the specified intervals to a numpy array and checks for
    errors.
    """
    intervals = np.array(intervals, dtype=np.float64)
    # Special case the empty list of intervals
    if len(intervals) == 0:
        intervals = np.zeros((0, 2), dtype=np.float64)
    if len(intervals.shape) != 2:
        raise ValueError("Intervals must be a 2D numpy array")
    if intervals.shape[1] != 2:
        raise ValueError("Intervals array shape must be (N, 2)")
    # TODO do this with numpy operations.
    last_right = start
    for left, right in intervals:
        if left < start or right > end:
            raise ValueError(
                "Intervals must be within {} and {}".format(start, end))
        if right <= left:
            raise ValueError("Bad interval: right <= left")
        if left < last_right:
            raise ValueError("Intervals must be disjoint.")
        last_right = right
    return intervals


def negate_intervals(intervals, start, end):
    """
    Returns the set of intervals *not* covered by the specified set of
    disjoint intervals in the specfied range.
    """
    intervals = intervals_to_np_array(intervals, start, end)
    other_intervals = []
    last_right = start
    for left, right in intervals:
        if left != last_right:
            other_intervals.append((last_right, left))
        last_right = right
    if last_right != end:
        other_intervals.append((last_right, end))
    return np.array(other_intervals)
