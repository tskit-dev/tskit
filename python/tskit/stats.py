# MIT License
#
# Copyright (c) 2018-2022 Tskit Developers
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
import copy
import itertools
import sys
import threading

import numpy as np

import _tskit
import tskit
import tskit.trees as trees


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


class CoalescenceTimeTable:
    """
    Container for sorted coalescence times, weights, and block assignments.
    """

    def __init__(self, time, block, weights):
        assert time.shape[0] == weights.shape[0] == block.shape[0]
        assert time.ndim == block.ndim == 1
        assert weights.ndim == 2
        self.num_weights = weights.shape[1]
        # remove empty records
        not_empty = np.sum(weights, 1) > 0
        self.num_records = sum(not_empty)
        self.time = time[not_empty]
        self.block = block[not_empty]
        self.weights = weights[not_empty, :]
        # add left boundary at time 0
        self.num_records += 1
        self.time = np.pad(self.time, (0, 1))
        self.block = np.pad(self.block, (0, 1))
        self.weights = np.pad(self.weights, ((0, 1), (0, 0)))
        # sort by node time
        time_order = np.argsort(self.time)
        self.time = self.time[time_order]
        self.block = self.block[time_order]
        self.weights = self.weights[time_order, :]
        # calculate quantiles
        self.num_blocks = 1 + np.max(self.block) if self.num_records > 0 else 0
        self.block_multiplier = np.ones(self.num_blocks)
        self.cum_weights = np.cumsum(self.weights, 0)
        self.quantile = np.empty((self.num_records, self.num_weights))
        self.quantile[:] = np.nan
        for i in range(self.num_weights):
            if self.cum_weights[-1, i] > 0:
                self.quantile[:, i] = self.cum_weights[:, i] / self.cum_weights[-1, i]

    def resample_blocks(self, block_multiplier):
        assert block_multiplier.shape[0] == self.num_blocks
        assert np.sum(block_multiplier) == self.num_blocks
        self.block_multiplier = block_multiplier
        for i in range(self.num_weights):
            self.cum_weights[:, i] = np.cumsum(
                self.weights[:, i] * self.block_multiplier[self.block], 0
            )
            if self.cum_weights[-1, i] > 0:
                self.quantile[:, i] = self.cum_weights[:, i] / self.cum_weights[-1, i]


class CoalescenceTimeDistribution:
    """
    Class to precompute a table of sorted/weighted node times, from which to calculate
    the empirical distribution function and estimate coalescence rates in time windows.
    """

    @staticmethod
    def _count_coalescence_events(node, tree, sample_sets):
        # TODO this will count unary nodes: should it count nodes
        # with >1 child instead?
        return np.array([1], dtype=np.int32)

    @staticmethod
    def _count_pair_coalescence_events(node, tree, sample_sets):
        """
        Count the number of pairs that coalesce in node, within and between the
        sets of samples in ``sample_sets``. The count of pairs with members that
        belong to sets :math:`a` and :math:`b` is:

        .. math:

            \\sum_{i \\neq j} (C_i(a) C_j(b) + C_i(b) C_j(a))/(1 - \\mathbb{I}[a = b])

        where :math:`C_i(a)` is the number of samples from set :math:`a`
        descended from child :math:`i`.  The values in the output are ordered
        canonically; e.g. if ``len(sample_sets) == 2`` then the values would
        correspond to counts of pairs with set labels ``[(0,0), (0,1), (1,1)]``.
        """

        # TODO needs to be optimized, use np.intersect1d
        children = tree.children(node)
        samples_per_child = [set(list(tree.samples(c))) for c in children]
        sample_counts = np.zeros((len(sample_sets), len(children)), dtype=np.int32)
        for i, s1 in enumerate(samples_per_child):
            for a, s2 in enumerate([set(s) for s in sample_sets]):
                sample_counts[a, i] = len(s1 & s2)

        pair_counts = []
        for a, b in itertools.combinations_with_replacement(
            range(sample_counts.shape[0]), 2
        ):
            count = 0
            for i, j in itertools.combinations(range(sample_counts.shape[1]), 2):
                count += (
                    sample_counts[a, i] * sample_counts[b, j]
                    + sample_counts[a, j] * sample_counts[b, i]
                ) / (1 + int(a == b))
            pair_counts.append(count)

        return np.array(pair_counts, dtype=np.int32)

    @staticmethod
    def _count_trio_first_coalescence_events(node, tree, sample_sets):
        """
        Count the number of pairs that coalesce in node with an outgroup,
        within and between the sets of samples in ``sample_sets``. In other
        words, count topologies of the form ``((A,B):node,C)`` where ``A,B,C``
        are labels and `node` is the node ID.  The count of pairs with members
        that belong to sets :math:`a` and :math:`b` with outgroup :math:`c` is:

        .. math:

            \\sum_{i \\neq j} (C_i(a) C_j(b) + C_i(b) C_j(a)) \\times
            O(c) / (1 - \\mathbb{I}[a = b])

        where :math:`C_i(a)` is the number of samples from set :math:`a`
        descended from child :math:`i` of the node, and :math:`O(c)` is the
        number of samples from set :math:`c` that are *not* descended from the
        node.  The values in the output are ordered canonically by pair then
        outgroup; e.g. if ``len(sample_sets) == 2`` then the values would
        correspond to counts of pairs with set labels,
        ``[((0,0),0), ((0,0),1), ..., ((0,1),0), ((0,1),1), ...]``.
        """
        samples = list(tree.samples(node))
        outg_counts = [len(s) - len(np.intersect1d(samples, s)) for s in sample_sets]
        pair_counts = CoalescenceTimeDistribution._count_pair_coalescence_events(
            node, tree, sample_sets
        )
        trio_counts = []
        for i in pair_counts:
            for j in outg_counts:
                trio_counts.append(i * j)
        return np.array(trio_counts, dtype=np.int32)

    def _update_weights_by_edge_diff(self, tree, edge_diff, running_weights):
        """
        Update ``running_weights`` to reflect ``tree`` using edge differences
        ``edge_diff`` with the previous tree.
        """

        assert edge_diff.interval == tree.interval

        # nodes that have been removed from tree
        removed = {i.child for i in edge_diff.edges_out if tree.is_isolated(i.child)}
        # TODO: What if sample is removed from tree? In that case should all
        # nodes be updated for trio first coalescences?

        # nodes where descendant subtree has been altered
        modified = {i.parent for i in edge_diff.edges_in}
        for i in copy.deepcopy(modified):
            while tree.parent(i) != tskit.NULL and not tree.parent(i) in modified:
                i = tree.parent(i)
                modified.add(i)

        # recalculate weights for current tree
        for i in removed:
            running_weights[i, :] = 0
        for i in modified:
            running_weights[i, :] = self.weight_func(i, tree, self.sample_sets)
        self.weight_func_evals += len(modified)

    def _build_ecdf_table_for_window(
        self, left, right, tree, edge_diffs, running_weights
    ):
        """
        Construct ECDF table for genomic interval [left, right]. Update ``tree``,
        ``edge_diffs``, and ``running_weights`` for input for next window. Trees are
        counted as belonging to any interval with which they overlap, and thus
        can be used in several intervals. Thus, the concatenation of ECDF
        tables across multiple intervals is not the same as the ECDF table
        for the union of those intervals. Trees within intervals are chunked
        into roughly equal-sized blocks for bootstrapping.
        """

        assert tree.interval.left <= left and right > left

        # assign trees in window to equal-sized blocks with unique id
        other_tree = tree.copy()
        # TODO: is a full copy of the tree needed, given that the original is
        # mutated below?
        if right >= other_tree.tree_sequence.sequence_length:
            other_tree.last()
        else:
            # other_tree.seek(right) won't work if `right` is recomb breakpoint
            while other_tree.interval.right < right:
                other_tree.next()
        tree_idx = np.arange(tree.index, other_tree.index + 1) - tree.index
        tree_offset = tree.index
        num_blocks = min(self.num_blocks, len(tree_idx))
        tree_blocks = np.floor_divide(num_blocks * tree_idx, len(tree_idx))

        # calculate span weights
        # TODO: if bootstrapping, does block span need to be tracked
        # and used to renormalise each replicate?
        other_tree.seek(tree.interval.left)
        tree_span = [
            min(other_tree.interval.right, right) - max(other_tree.interval.left, left)
        ]
        while other_tree.index < tree_offset + tree_idx[-1]:
            other_tree.next()
            tree_span.append(
                min(other_tree.interval.right, right)
                - max(other_tree.interval.left, left)
            )
        tree_span = np.array(tree_span) / sum(tree_span)

        # storage if using single window, block for entire tree sequence
        buffer_size = self.buffer_size
        table_size = buffer_size
        time = np.zeros(table_size)
        block = np.zeros(table_size, dtype=np.int32)
        weights = np.zeros((table_size, self.num_weights))

        # assemble table of coalescence times in window
        indices = np.zeros(tree.tree_sequence.num_nodes, dtype=np.int32) - 1
        last_block = np.zeros(tree.tree_sequence.num_nodes, dtype=np.int32) - 1
        num_record = 0
        while tree.index != tskit.NULL:
            if tree.interval.right > left:
                current_block = tree_blocks[tree.index - tree_offset]
                if self.span_normalise:
                    span_weight = tree_span[tree.index - tree_offset]
                else:
                    span_weight = 1.0
                nodes_in_tree = np.array(
                    [i for i in tree.nodes() if tree.is_internal(i)]
                )
                # TODO this will fail if all nodes are isolated (masked tree)
                nodes_to_add = nodes_in_tree[
                    np.where(last_block[nodes_in_tree] != current_block)
                ]
                if len(nodes_to_add) > 0:
                    idx = np.arange(num_record, num_record + len(nodes_to_add))
                    last_block[nodes_to_add] = current_block
                    indices[nodes_to_add] = idx
                    if table_size < num_record + len(nodes_to_add):
                        table_size += buffer_size
                        time = np.pad(time, (0, buffer_size))
                        block = np.pad(block, (0, buffer_size))
                        weights = np.pad(weights, ((0, buffer_size), (0, 0)))
                    time[idx] = [tree.time(i) for i in nodes_to_add]
                    block[idx] = current_block
                    num_record += len(nodes_to_add)
                weights[indices[nodes_in_tree], :] += (
                    span_weight * running_weights[nodes_in_tree, :]
                )

            if tree.interval.right < right:
                # if current tree does not cross window boundary, move to next
                tree.next()
                self._update_weights_by_edge_diff(
                    tree, next(edge_diffs), running_weights
                )
            else:
                # use current tree as initial tree for next window
                break

        return CoalescenceTimeTable(time, block, weights)

    def _generate_ecdf_tables(self, ts, window_breaks):
        """
        Return generator for ECDF tables across genomic windows defined by
        ``window_breaks``.

        ..note:: This could be used in methods in place of loops over
            pre-assembled tables.
        """

        tree = ts.first()
        edge_diffs = ts.edge_diffs()
        running_weights = np.zeros((ts.num_nodes, self.num_weights))
        self._update_weights_by_edge_diff(tree, next(edge_diffs), running_weights)
        for left, right in zip(window_breaks[:-1], window_breaks[1:]):
            yield self._build_ecdf_table_for_window(
                left, right, tree, edge_diffs, running_weights
            )

    def __init__(
        self,
        ts,
        sample_sets=None,
        weight_func=None,
        window_breaks=None,
        blocks_per_window=None,
        span_normalise=True,
    ):

        assert isinstance(ts, trees.TreeSequence)

        if sample_sets is None:
            sample_sets = [list(ts.samples())]
        assert all([isinstance(i, list) for i in sample_sets])
        assert all([i in ts.samples() for j in sample_sets for i in j])
        self.sample_sets = sample_sets

        if weight_func is None or weight_func == "coalescence_events":
            self.weight_func = self._count_coalescence_events
        elif weight_func == "pair_coalescence_events":
            self.weight_func = self._count_pair_coalescence_events
        elif weight_func == "trio_first_coalescence_events":
            self.weight_func = self._count_trio_first_coalescence_events
        else:
            assert callable(weight_func)
            self.weight_func = weight_func
        _weight_func_eval = self.weight_func(0, ts.first(), self.sample_sets)
        assert isinstance(_weight_func_eval, np.ndarray)
        assert _weight_func_eval.ndim == 1
        self.num_weights = len(_weight_func_eval)

        if window_breaks is None:
            window_breaks = np.array([0.0, ts.sequence_length])
        assert isinstance(window_breaks, np.ndarray)
        assert window_breaks.ndim == 1
        assert np.min(window_breaks) >= 0.0
        assert np.max(window_breaks) <= ts.sequence_length
        window_breaks = np.sort(np.unique(window_breaks))
        self.windows = [
            trees.Interval(left, right)
            for left, right in zip(window_breaks[:-1], window_breaks[1:])
        ]
        self.num_windows = len(self.windows)

        if blocks_per_window is None:
            blocks_per_window = 1
        assert isinstance(blocks_per_window, int)
        assert blocks_per_window > 0
        self.num_blocks = blocks_per_window

        assert isinstance(span_normalise, bool)
        self.span_normalise = span_normalise

        self.buffer_size = ts.num_nodes
        self.weight_func_evals = 0
        self.tables = [table for table in self._generate_ecdf_tables(ts, window_breaks)]

    # TODO
    #
    # def __str__(self):
    #    return self.useful_text_summary()
    #
    # def __repr_html__(self):
    #    return self.useful_html_summary()

    def copy(self):
        return copy.deepcopy(self)

    def ecdf(self, times):
        """
        Returns the empirical distribution function evaluated at the time
        points in ``times``.

        The output array has shape ``(self.num_weights, len(times),
        self.num_windows)``.
        """

        assert isinstance(times, np.ndarray)
        assert times.ndim == 1

        values = np.empty((self.num_weights, len(times), self.num_windows))
        values[:] = np.nan
        for k, table in enumerate(self.tables):
            indices = np.searchsorted(table.time, times, side="right") - 1
            assert all([0 <= i < table.num_records for i in indices])
            values[:, :, k] = table.quantile[indices, :].T
        return values

    # TODO
    #
    # def quantile(self, times):
    #   """
    #   Return interpolated quantiles of coalescence times, using the same
    #   approach as numpy.quantile(..., method="linear")
    #   """

    def num_coalesced(self, times):
        """
        Returns number of coalescence events that have occured by the time
        points in ``times``.

        The output array has shape ``(self.num_weights, len(times),
        self.num_windows)``.
        """

        assert isinstance(times, np.ndarray)
        assert times.ndim == 1

        values = self.ecdf(times)
        for k, table in enumerate(self.tables):
            weight_totals = table.cum_weights[-1, :].reshape(values.shape[0], 1)
            values[:, :, k] *= np.tile(weight_totals, (1, values.shape[1]))
        return values

    def num_uncoalesced(self, times):
        """
        Returns the number of coalescence events remaining by the time points
        in ``times``.

        The output array has shape ``(self.num_weights, len(times),
        self.num_windows)``.
        """

        values = 1.0 - self.ecdf(times)
        for k, table in enumerate(self.tables):
            weight_totals = table.cum_weights[-1, :].reshape(values.shape[0], 1)
            values[:, :, k] *= np.tile(weight_totals, (1, values.shape[1]))
        return values

    def mean(self, since=0.0):
        """
        Returns the average time between ``since`` and the coalescence events
        that occurred after ``since``.

        Note that ``1/self.mean(left)`` is an estimate of the coalescence rate
        over the interval (left, infinity).

        The output array has shape ``(self.num_weights, self.num_windows)``.

        ..note:: Check for overflow in ``np.average``.
        """

        assert isinstance(since, float) and since >= 0.0

        values = np.empty((self.num_weights, self.num_windows))
        values[:] = np.nan
        for k, table in enumerate(self.tables):
            index = np.searchsorted(table.time, since, side="right")
            if index == table.num_records:
                values[:, k] = np.nan
            else:
                for i in range(self.num_weights):
                    if table.cum_weights[-1, i] > 0:
                        multiplier = table.block_multiplier[table.block[index:]]
                        values[i, k] = np.average(
                            table.time[index:] - since,
                            weights=table.weights[index:, i] * multiplier,
                        )
        return values

    def coalescence_probability_in_intervals(self, time_breaks):
        """
        Returns the proportion of coalescence events occurring in the time
        intervals defined by ``time_breaks``, out of events that have not
        yet occurred by the intervals' left boundaries.

        The output array has shape ``(self.num_weights, len(time_breaks)-1,
        self.num_windows)``.
        """

        assert isinstance(time_breaks, np.ndarray)

        time_breaks = np.sort(np.unique(time_breaks))
        num_coalesced = self.num_coalesced(time_breaks)
        num_uncoalesced = self.num_uncoalesced(time_breaks)
        numer = num_coalesced[:, 1:, :] - num_coalesced[:, :-1, :]
        denom = num_uncoalesced[:, :-1, :]
        return numer / np.where(np.isclose(denom, 0.0), np.nan, denom)

    def coalescence_rate_in_intervals(self, time_breaks):
        """
        Returns the interval-censored Kaplan-Meier estimate of the hazard rate for
        coalesence events within the time intervals defined by ``time_breaks``. The
        estimator is,

        .. math::
            \\hat{c}_{l,r} = \\begin{cases}
              \\log(1 - x_{l,r}/k_{l})/(l - r) & \\mathrm{if~} x_{l,r} < k_{l} \\\\
              \\hat{c}_{l,r} = k_{l} / t_{l,r} & \\mathrm{if~} x_{l,r} = k_{l}
            \\end{cases}

        and is undefined where :math:`k_{l} = 0`. Here, :math:`x_{l,r}` is the
        number of events occuring in time interval :math:`(l, r]`,
        :math:`k_{l}` is the number of events remaining at time :math:`l`, and
        :math:`t_{l,r}` is the sum of event times occurring in the interval
        :math:`(l, r]`.

        The output array has shape ``(self.num_weights, len(time_breaks)-1,
        self.num_windows)``.
        """

        assert isinstance(time_breaks, np.ndarray)

        time_breaks = np.sort(np.unique(time_breaks))
        phi = self.coalescence_probability_in_intervals(time_breaks)
        duration = np.reshape(time_breaks[1:] - time_breaks[:-1], (1, phi.shape[1], 1))
        numer = -np.log(1.0 - np.where(np.isclose(phi, 1.0), np.nan, phi))
        denom = np.tile(duration, (self.num_weights, 1, self.num_windows))
        for i, j, k in np.argwhere(np.isclose(phi, 1.0)):
            numer[i, j, k] = 1.0
            denom[i, j, k] = self.mean(time_breaks[j])[i, k]
        return numer / denom

    def block_bootstrap(self, num_replicates=1, random_seed=None):
        """
        Return a generator that produces ``num_replicates`` copies of the
        object where blocks within genomic windows are randomly resampled.

        ..note:: Copying could be expensive.
        """

        rng = np.random.default_rng(random_seed)
        for _i in range(num_replicates):
            replicate = self.copy()
            for table in replicate.tables:
                block_multiplier = rng.multinomial(
                    table.num_blocks, [1.0 / table.num_blocks] * table.num_blocks
                )
                table.resample_blocks(block_multiplier)
            yield replicate
