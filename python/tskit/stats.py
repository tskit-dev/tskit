# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
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
import sys
import threading

import numpy as np

import _tskit


class LdCalculator:
    """
    Class for calculating `linkage disequilibrium
    <https://en.wikipedia.org/wiki/Linkage_disequilibrium>`_ coefficients
    between pairs of sites in a :class:`TreeSequence`.

    .. note:: This interface is deprecated and a replacement is planned.
        Please see https://github.com/tskit-dev/tskit/issues/1900 for
        more information. Note also that the current implementation is
        quite limited (see warning below).

    .. warning:: This class does not currently support sites that have more than one
        mutation. Using it on such a tree sequence will raise a LibraryError with
        an "Only infinite sites mutations supported" message.

        Silent mutations are also not supported and will result in a LibraryError.

    :param TreeSequence tree_sequence: The tree sequence of interest.
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
        sites at the specified indexes. This method is *not* an efficient
        method for computing large numbers of pairwise LD values; please use either
        :meth:`.r2_array` or :meth:`.r2_matrix` for this purpose.

        :param int a: The index of the first site.
        :param int b: The index of the second site.
        :return: The value of :math:`r^2` between the sites at indexes
            ``a`` and ``b``.
        :rtype: float
        """
        with self._instance_lock:
            return self._ll_ld_calculator.get_r2(a, b)

    def get_r2_array(self, a, direction=1, max_mutations=None, max_distance=None):
        # Deprecated alias for r2_array
        return self.r2_array(
            a,
            direction=direction,
            max_mutations=max_mutations,
            max_distance=max_distance,
        )

    def r2_array(
        self, a, direction=1, max_mutations=None, max_distance=None, max_sites=None
    ):
        """
        Returns the value of the :math:`r^2` statistic between the focal
        site at index :math:`a` and a set of other sites. The method
        operates by starting at the focal site and iterating over adjacent
        sites (in either the forward or backwards direction) until either a
        maximum number of other sites have been considered (using the
        ``max_sites`` parameter), a maximum distance in sequence
        coordinates has been reached (using the ``max_distance`` parameter) or
        the start/end of the sequence has been reached. For every site
        :math:`b` considered, we then insert the value of :math:`r^2` between
        :math:`a` and :math:`b` at the corresponding index in an array, and
        return the entire array. If the returned array is :math:`x` and
        ``direction`` is :data:`tskit.FORWARD` then :math:`x[0]` is the
        value of the statistic for :math:`a` and :math:`a + 1`, :math:`x[1]`
        the value for :math:`a` and :math:`a + 2`, etc. Similarly, if
        ``direction`` is :data:`tskit.REVERSE` then :math:`x[0]` is the
        value of the statistic for :math:`a` and :math:`a - 1`, :math:`x[1]`
        the value for :math:`a` and :math:`a - 2`, etc.

        :param int a: The index of the focal sites.
        :param int direction: The direction in which to travel when
            examining other sites. Must be either
            :data:`tskit.FORWARD` or :data:`tskit.REVERSE`. Defaults
            to :data:`tskit.FORWARD`.
        :param int max_sites: The maximum number of sites to return
            :math:`r^2` values for. Defaults to as many sites as
            possible.
        :param int max_mutations: Deprecated synonym for max_sites.
        :param float max_distance: The maximum absolute distance between
            the focal sites and those for which :math:`r^2` values
            are returned.
        :return: An array of double precision floating point values
            representing the :math:`r^2` values for sites in the
            specified direction.
        :rtype: numpy.ndarray
        """
        if max_mutations is not None and max_sites is not None:
            raise ValueError("max_mutations is a deprecated synonym for max_sites")
        if max_mutations is not None:
            max_sites = max_mutations
        max_sites = -1 if max_sites is None else max_sites
        if max_distance is None:
            max_distance = sys.float_info.max
        with self._instance_lock:
            return self._ll_ld_calculator.get_r2_array(
                a,
                direction=direction,
                max_sites=max_sites,
                max_distance=max_distance,
            )

    def get_r2_matrix(self):
        # Deprecated alias for r2_matrix
        return self.r2_matrix()

    def r2_matrix(self):
        """
        Returns the complete :math:`m \\times m` matrix of pairwise
        :math:`r^2` values in a tree sequence with :math:`m` sites.

        :return: An 2 dimensional square array of double precision
            floating point values representing the :math:`r^2` values for
            all pairs of sites.
        :rtype: numpy.ndarray
        """
        m = self._tree_sequence.num_sites
        A = np.ones((m, m), dtype=float)
        for j in range(m - 1):
            a = self.get_r2_array(j)
            A[j, j + 1 :] = a
            A[j + 1 :, j] = a
        return A
