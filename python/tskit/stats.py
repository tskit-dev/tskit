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
Module responsible for computing various statistics on tree sequences.
"""
import struct
import sys
import threading

import numpy as np

import _tskit


class LdCalculator:
    """
    Class for calculating `linkage disequilibrium
    <https://en.wikipedia.org/wiki/Linkage_disequilibrium>`_ coefficients
    between pairs of mutations in a :class:`TreeSequence`. This class requires
    the `numpy <http://www.numpy.org/>`_ library.

    This class supports multithreaded access using the Python :mod:`threading`
    module. Separate instances of :class:`LdCalculator` referencing the
    same tree sequence can operate in parallel in multiple threads.

    .. note:: This class does not currently support sites that have more than one
        mutation. Using it on such a tree sequence will raise a LibraryError with
        an "Unsupported operation" message.

    :param TreeSequence tree_sequence: The tree sequence containing the
        mutations we are interested in.
    """

    def __init__(self, tree_sequence):
        self._tree_sequence = tree_sequence
        self._ll_ld_calculator = _tskit.LdCalculator(
            tree_sequence.get_ll_tree_sequence()
        )
        # To protect low-level C code, only one method may execute on the
        # low-level objects at one time.
        self._instance_lock = threading.Lock()

    def get_r2(self, a, b):
        # Deprecated alias for r2(a, b)
        return self.r2(a, b)

    def r2(self, a, b):
        """
        Returns the value of the :math:`r^2` statistic between the pair of
        mutations at the specified indexes. This method is *not* an efficient
        method for computing large numbers of pairwise; please use either
        :meth:`.r2_array` or :meth:`.r2_matrix` for this purpose.

        :param int a: The index of the first mutation.
        :param int b: The index of the second mutation.
        :return: The value of :math:`r^2` between the mutations at indexes
            ``a`` and ``b``.
        :rtype: float
        """
        with self._instance_lock:
            return self._ll_ld_calculator.get_r2(a, b)

    def get_r2_array(self, a, direction=1, max_mutations=None, max_distance=None):
        # Deprecated alias for r2_array
        return self.r2_array(a, direction, max_mutations, max_distance)

    def r2_array(self, a, direction=1, max_mutations=None, max_distance=None):
        """
        Returns the value of the :math:`r^2` statistic between the focal
        mutation at index :math:`a` and a set of other mutations. The method
        operates by starting at the focal mutation and iterating over adjacent
        mutations (in either the forward or backwards direction) until either a
        maximum number of other mutations have been considered (using the
        ``max_mutations`` parameter), a maximum distance in sequence
        coordinates has been reached (using the ``max_distance`` parameter) or
        the start/end of the sequence has been reached. For every mutation
        :math:`b` considered, we then insert the value of :math:`r^2` between
        :math:`a` and :math:`b` at the corresponding index in an array, and
        return the entire array. If the returned array is :math:`x` and
        ``direction`` is :data:`tskit.FORWARD` then :math:`x[0]` is the
        value of the statistic for :math:`a` and :math:`a + 1`, :math:`x[1]`
        the value for :math:`a` and :math:`a + 2`, etc. Similarly, if
        ``direction`` is :data:`tskit.REVERSE` then :math:`x[0]` is the
        value of the statistic for :math:`a` and :math:`a - 1`, :math:`x[1]`
        the value for :math:`a` and :math:`a - 2`, etc.

        :param int a: The index of the focal mutation.
        :param int direction: The direction in which to travel when
            examining other mutations. Must be either
            :data:`tskit.FORWARD` or :data:`tskit.REVERSE`. Defaults
            to :data:`tskit.FORWARD`.
        :param int max_mutations: The maximum number of mutations to return
            :math:`r^2` values for. Defaults to as many mutations as
            possible.
        :param float max_distance: The maximum absolute distance between
            the focal mutation and those for which :math:`r^2` values
            are returned.
        :return: An array of double precision floating point values
            representing the :math:`r^2` values for mutations in the
            specified direction.
        :rtype: numpy.ndarray
        :warning: For efficiency reasons, the underlying memory used to
            store the returned array is shared between calls. Therefore,
            if you wish to store the results of a single call to
            ``get_r2_array()`` for later processing you **must** take a
            copy of the array!
        """
        if max_mutations is None:
            max_mutations = -1
        if max_distance is None:
            max_distance = sys.float_info.max
        item_size = struct.calcsize("d")
        buffer = bytearray(self._tree_sequence.get_num_mutations() * item_size)
        with self._instance_lock:
            num_values = self._ll_ld_calculator.get_r2_array(
                buffer,
                a,
                direction=direction,
                max_mutations=max_mutations,
                max_distance=max_distance,
            )
        return np.frombuffer(buffer, "d", num_values)

    def get_r2_matrix(self):
        # Deprecated alias for r2_matrix
        return self.r2_matrix()

    def r2_matrix(self):
        """
        Returns the complete :math:`m \\times m` matrix of pairwise
        :math:`r^2` values in a tree sequence with :math:`m` mutations.

        :return: An 2 dimensional square array of double precision
            floating point values representing the :math:`r^2` values for
            all pairs of mutations.
        :rtype: numpy.ndarray
        """
        m = self._tree_sequence.get_num_mutations()
        A = np.ones((m, m), dtype=float)
        for j in range(m - 1):
            a = self.get_r2_array(j)
            A[j, j + 1 :] = a
            A[j + 1 :, j] = a
        return A
