# MIT License
#
# Copyright (c) 2024 Tskit Developers
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
Test cases for coalescence rate calculation in tskit.
"""
import itertools

import msprime
import numpy as np
import pytest

import tests
import tskit
from tests import tsutil

# --- prototype --- #


def _pair_coalescence_counts_identity(
    coalescing_pairs,
    nodes_time,
    nodes_order,
):
    return coalescing_pairs


def _pair_coalescence_quantiles(
    coalescing_pairs,
    nodes_time,
    nodes_order,
    quantiles,
):
    """
    Estimate `quantiles` of the distribution of `nodes_time` weighted by
    `coalescing_pairs`, by inverting the empirical CDF. `nodes_order`
    should sort nodes into ascending time order.
    """
    assert nodes_time.size == coalescing_pairs.size == nodes_order.size
    assert np.all(np.diff(quantiles) > 0)
    assert np.all(np.logical_and(0 <= quantiles, quantiles <= 1))
    num_nodes = coalescing_pairs.size
    num_quantiles = quantiles.size
    total_mass = coalescing_pairs.sum()
    output = np.full(num_quantiles, np.nan)
    i, j = 0, 0
    ecdf = 0.0
    while i < num_nodes:
        mass = coalescing_pairs[nodes_order[i]] / total_mass
        if mass > 0:
            ecdf += mass
            time = nodes_time[nodes_order[i]]
            while j < num_quantiles and quantiles[j] <= ecdf:
                output[j] = time
                j += 1
        i += 1
    if quantiles[-1] == 1.0:
        output[-1] = time
    return output


def _pair_coalescence_stat(
    ts,
    summary_func,
    summary_func_dim,
    summary_func_kwargs,
    sample_sets=None,
    indexes=None,
    windows=None,
    time_windows=None,
    span_normalise=True,
):
    """
    Apply `summary_func(node_weights, node_times, node_order, **summary_func_kwargs)` to
    the empirical distribution of pair coalescence times for each index / window.
    """

    # TODO: the logic with time_windows and summary_func is not quite worked out

    if sample_sets is None:
        sample_sets = [list(ts.samples())]
    for s in sample_sets:
        if len(s) == 0:
            raise ValueError("Sample sets must contain at least one element")
        if not (min(s) >= 0 and max(s) < ts.num_nodes):
            raise ValueError("Sample is out of bounds")

    drop_right_dimension = False
    if indexes is None:
        drop_right_dimension = True
        if len(sample_sets) == 1:
            indexes = [(0, 0)]
        elif len(sample_sets) == 2:
            indexes = [(0, 1)]
        else:
            raise ValueError(
                "Must specify indexes if there are more than two sample sets"
            )
    for i in indexes:
        if not len(i) == 2:
            raise ValueError("Sample set indexes must be length two")
        if not (min(i) >= 0 and max(i) < len(sample_sets)):
            raise ValueError("Sample set index is out of bounds")

    drop_left_dimension = False
    if windows is None:
        drop_left_dimension = True
        windows = np.array([0.0, ts.sequence_length])
    if not (isinstance(windows, np.ndarray) and windows.size > 1):
        raise ValueError("Windows must be an array of breakpoints")
    if not (windows[0] == 0.0 and windows[-1] == ts.sequence_length):
        raise ValueError("First and last window breaks must be sequence boundary")
    if not np.all(np.diff(windows) > 0):
        raise ValueError("Window breaks must be strictly increasing")

    if isinstance(time_windows, str) and time_windows == "nodes":
        nodes_map = np.arange(ts.num_nodes)
        num_time_windows = ts.num_nodes
    else:
        if not (isinstance(time_windows, np.ndarray) and time_windows.size > 1):
            raise ValueError("Time windows must be an array of breakpoints")
        if not np.all(np.diff(time_windows) > 0):
            raise ValueError("Time windows must be strictly increasing")
        if ts.time_units == tskit.TIME_UNITS_UNCALIBRATED:
            raise ValueError("Time windows require calibrated node times")
        nodes_map = np.searchsorted(time_windows, ts.nodes_time, side="right") - 1
        nodes_oob = np.logical_or(nodes_map < 0, nodes_map >= time_windows.size)
        nodes_map[nodes_oob] = tskit.NULL
        num_time_windows = time_windows.size - 1

    num_nodes = ts.num_nodes
    num_windows = windows.size - 1
    num_sample_sets = len(sample_sets)
    num_indexes = len(indexes)

    edges_child = ts.edges_child
    edges_parent = ts.edges_parent
    nodes_time = ts.nodes_time
    nodes_order = np.argsort(nodes_time)
    sequence_length = ts.sequence_length
    output_size = summary_func_dim
    samples = np.concatenate(sample_sets)

    nodes_parent = np.full(num_nodes, tskit.NULL)
    nodes_sample = np.zeros((num_nodes, num_sample_sets))
    nodes_weight = np.zeros((num_time_windows, num_indexes))
    coalescing_pairs = np.zeros((num_time_windows, num_indexes))
    output = np.zeros((num_windows, output_size, num_indexes))
    visited = np.full(num_nodes, False)

    for i, s in enumerate(sample_sets):  # initialize
        nodes_sample[s, i] = 1
    sample_counts = nodes_sample.copy()

    w = 0
    position = tsutil.TreePosition(ts)
    while position.interval.right < sequence_length:
        position.next()
        left, right = position.interval.left, position.interval.right
        out_range, in_range = position.out_range, position.in_range
        remainder = sequence_length - left

        for b in range(out_range.start, out_range.stop):  # edges_out
            e = out_range.order[b]
            p = edges_parent[e]
            c = edges_child[e]
            nodes_parent[c] = tskit.NULL
            inside = sample_counts[c]
            while p != tskit.NULL:
                u = nodes_map[p]
                if u != tskit.NULL:
                    outside = sample_counts[p] - sample_counts[c] - nodes_sample[p]
                    for i, (j, k) in enumerate(indexes):
                        weight = inside[j] * outside[k]
                        if j != k:
                            weight += inside[k] * outside[j]
                        coalescing_pairs[u, i] -= weight * remainder
                c, p = p, nodes_parent[p]
            p = edges_parent[e]
            while p != tskit.NULL:
                sample_counts[p] -= inside
                p = nodes_parent[p]

        for b in range(in_range.start, in_range.stop):  # edges_in
            e = in_range.order[b]
            p = edges_parent[e]
            c = edges_child[e]
            nodes_parent[c] = p
            inside = sample_counts[c]
            while p != tskit.NULL:
                sample_counts[p] += inside
                p = nodes_parent[p]
            p = edges_parent[e]
            while p != tskit.NULL:
                u = nodes_map[p]
                if u != tskit.NULL:
                    outside = sample_counts[p] - sample_counts[c] - nodes_sample[p]
                    for i, (j, k) in enumerate(indexes):
                        weight = inside[j] * outside[k]
                        if j != k:
                            weight += inside[k] * outside[j]
                        coalescing_pairs[u, i] += weight * remainder
                c, p = p, nodes_parent[p]

        while w < num_windows and windows[w + 1] <= right:  # flush window
            remainder = sequence_length - windows[w + 1]
            nodes_weight[:] = coalescing_pairs[:]
            coalescing_pairs[:] = 0.0
            for c in samples:
                p = nodes_parent[c]
                while not visited[c] and p != tskit.NULL:
                    u = nodes_map[p]
                    if u != tskit.NULL:
                        inside = sample_counts[c]
                        outside = sample_counts[p] - sample_counts[c] - nodes_sample[p]
                        for i, (j, k) in enumerate(indexes):
                            weight = inside[j] * outside[k]
                            if j != k:
                                weight += inside[k] * outside[j]
                            nodes_weight[u, i] -= weight * remainder / 2
                            coalescing_pairs[u, i] += weight * remainder / 2
                    visited[c] = True
                    p, c = nodes_parent[p], p
            for c in samples:
                p = nodes_parent[c]
                while visited[c] and p != tskit.NULL:
                    visited[c] = False
                    p, c = nodes_parent[p], p
            if span_normalise:
                nodes_weight /= windows[w + 1] - windows[w]
            for i in range(num_indexes):  # apply function to empirical distribution
                output[w, :, i] = summary_func(
                    nodes_weight[:, i],
                    nodes_time,
                    nodes_order,
                    **summary_func_kwargs,
                )
            w += 1

    if drop_right_dimension:
        output = output[..., 0]
    if drop_left_dimension:
        output = output[0]

    return output


def proto_pair_coalescence_counts(
    ts,
    sample_sets=None,
    indexes=None,
    windows=None,
    span_normalise=True,
    time_windows="nodes",
):
    """
    Prototype for ts.pair_coalescence_counts.

    Calculate the number of coalescing sample pairs per node, summed over
    trees and weighted by tree span.

    The number of coalescing pairs may be calculated within or between the
    non-overlapping lists of samples contained in `sample_sets`. In the
    latter case, pairs are counted if they have exactly one member in each
    of two sample sets. If `sample_sets` is omitted, a single group
    containing all samples is assumed.

    The argument `indexes` may be used to specify which pairs of sample
    sets to compute the statistic between, and in what order. If
    `indexes=None`, then `indexes` is assumed to equal `[(0,0)]` for a
    single sample set and `[(0,1)]` for two sample sets. For more than two
    sample sets, `indexes` must be explicitly passed.

    The argument `time_windows` may be used to count coalescence
    events within time intervals (if an array of breakpoints is supplied)
    rather than for individual nodes (the default).

    The output array has dimension `(windows, nodes, indexes)` with
    dimensions dropped when the corresponding argument is set to None.

    :param list sample_sets: A list of lists of Node IDs, specifying the
        groups of nodes to compute the statistic with, or None.
    :param list indexes: A list of 2-tuples, or None.
    :param list windows: An increasing list of breakpoints between the
        sequence windows to compute the statistic in, or None.
    :param bool span_normalise: Whether to divide the result by the span of
        the window (defaults to True).
    :param time_windows: Either a string "nodes" or an increasing
        list of breakpoints between time intervals.
    """

    summary_func = _pair_coalescence_counts_identity
    summary_func_kwargs = {}
    if isinstance(time_windows, str) and time_windows == "nodes":
        summary_func_dim = ts.num_nodes
    else:
        if not (isinstance(time_windows, np.ndarray) and time_windows.size > 1):
            raise ValueError("Time windows must be an array of breakpoints")
        if not np.all(np.diff(time_windows) > 0):
            raise ValueError("Time windows must be strictly increasing")
        if ts.time_units == tskit.TIME_UNITS_UNCALIBRATED:
            raise ValueError("Time windows require calibrated node times")
        summary_func_dim = time_windows.size - 1

    return _pair_coalescence_stat(
        ts,
        summary_func=summary_func,
        summary_func_dim=summary_func_dim,
        summary_func_kwargs=summary_func_kwargs,
        sample_sets=sample_sets,
        indexes=indexes,
        windows=windows,
        time_windows=time_windows,
        span_normalise=span_normalise,
    )


def proto_pair_coalescence_quantiles(
    ts,
    quantiles,
    sample_sets=None,
    indexes=None,
    windows=None,
):
    """
    Prototype for ts.pair_coalescence_quantiles.

    Estimate quantiles of pair coalescence times by inverting the empirical
    CDF. This is equivalent to the "inverted_cdf" method of `numpy.quantile`
    applied to node times, with weights proportional to the number of
    coalescing pairs per node (averaged over trees).

    Quantiles of pair coalescence times may be calculated within or
    between the non-overlapping lists of samples contained in `sample_sets`. In
    the latter case, pairs are counted if they have exactly one member in each
    of two sample sets. If `sample_sets` is omitted, a single group containing
    all samples is assumed.

    The argument `indexes` may be used to specify which pairs of sample sets to
    compute coalescences between, and in what order. If `indexes=None`, then
    `indexes` is assumed to equal `[(0,0)]` for a single sample set and
    `[(0,1)]` for two sample sets. For more than two sample sets, `indexes`
    must be explicitly passed.

    The output array has dimension `(quantiles, nodes, indexes)` with
    dimensions dropped when the corresponding argument is set to None.

    :param quantiles: A list of breakpoints between [0, 1].
    :param list sample_sets: A list of lists of Node IDs, specifying the
        groups of nodes to compute the statistic with, or None.
    :param list indexes: A list of 2-tuples, or None.
    :param list windows: An increasing list of breakpoints between the
        sequence windows to compute the statistic in, or None.
    """

    if not isinstance(quantiles, np.ndarray):
        raise ValueError("Quantiles must be an array of breakpoints")
    if not np.all(np.logical_and(quantiles >= 0, quantiles <= 1.0)):
        raise ValueError("Quantiles must be in [0, 1]")

    summary_func = _pair_coalescence_quantiles
    summary_func_dim = quantiles.size
    summary_func_kwargs = {"quantiles": quantiles}

    return _pair_coalescence_stat(
        ts,
        summary_func=summary_func,
        summary_func_dim=summary_func_dim,
        summary_func_kwargs=summary_func_kwargs,
        sample_sets=sample_sets,
        indexes=indexes,
        windows=windows,
        time_windows="nodes",
        span_normalise=True,
    )


# --- testing --- #


def naive_pair_coalescence_counts(ts, sample_set_0, sample_set_1):
    """
    Naive implementation of ts.pair_coalescence_counts.

    Count pairwise coalescences tree by tree, by enumerating nodes in each
    tree. For a binary node, the number of pairs of samples that coalesce in a
    given node is the product of the number of samples subtended by the left
    and right child. For higher arities, the count is summed over all possible
    pairs of children.
    """
    output = np.zeros(ts.num_nodes)
    for t in ts.trees():
        sample_counts = np.zeros((ts.num_nodes, 2), dtype=np.int32)
        pair_counts = np.zeros(ts.num_nodes)
        for p in t.postorder():
            samples = list(t.samples(p))
            sample_counts[p, 0] = np.intersect1d(samples, sample_set_0).size
            sample_counts[p, 1] = np.intersect1d(samples, sample_set_1).size
            for i, j in itertools.combinations(t.children(p), 2):
                pair_counts[p] += sample_counts[i, 0] * sample_counts[j, 1]
                pair_counts[p] += sample_counts[i, 1] * sample_counts[j, 0]
        output += pair_counts * t.span
    return output


def _numpy_weighted_quantile(atoms, weights, quantiles):
    """
    Requires numpy 2.0. Enforcing `weights > 0` avoids odd behaviour where
    numpy assigns the 0th quantile to the sample minimum, even if this minimum
    has zero weight.
    """
    assert np.all(weights >= 0.0)
    return np.quantile(
        atoms[weights > 0],
        quantiles,
        weights=weights[weights > 0] / weights.sum(),
        method="inverted_cdf",
    )


def convert_to_nonsuccinct(ts):
    """
    Give the edges and internal nodes in each tree distinct IDs
    """
    tables = tskit.TableCollection(sequence_length=ts.sequence_length)
    for _ in range(ts.num_populations):
        tables.populations.add_row()
    nodes_count = 0
    for n in ts.samples():
        tables.nodes.add_row(
            time=ts.nodes_time[n],
            flags=ts.nodes_flags[n],
            population=ts.nodes_population[n],
        )
        nodes_count += 1
    for t in ts.trees():
        nodes_map = {n: n for n in ts.samples()}
        for n in t.nodes():
            if t.num_samples(n) > 1:
                tables.nodes.add_row(
                    time=ts.nodes_time[n],
                    flags=ts.nodes_flags[n],
                    population=ts.nodes_population[n],
                )
                nodes_map[n] = nodes_count
                nodes_count += 1
        for n in t.nodes():
            if t.edge(n) != tskit.NULL:
                tables.edges.add_row(
                    parent=nodes_map[t.parent(n)],
                    child=nodes_map[n],
                    left=t.interval.left,
                    right=t.interval.right,
                )
    tables.sort()
    ts_unroll = tables.tree_sequence()
    assert nodes_count == ts_unroll.num_nodes
    return ts_unroll


class TestCoalescingPairsOneTree:
    """
    Test against worked example (single tree)
    """

    def example_ts(self):
        """
        10.0┊         13      ┊
            ┊       ┏━━┻━━┓   ┊
         8.0┊      12     ┃   ┊
            ┊     ┏━┻━┓   ┃   ┊
         6.0┊    11   ┃   ┃   ┊
            ┊  ┏━━╋━┓ ┃   ┃   ┊
         2.0┊ 10  ┃ ┃ ┃   9   ┊
            ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓ ┊
         1.0┊ ┃ ┃ ┃ ┃ ┃  8  ┃ ┊
            ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃ ┊
         0.0┊ 0 7 4 5 6 1 2 3 ┊
            ┊ A A A A B B B B ┊
        """
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.set_columns(
            time=np.array([0] * 8 + [1, 2, 2, 6, 8, 10]),
            flags=np.repeat([1, 0], [8, 6]).astype("uint32"),
        )
        tables.edges.set_columns(
            left=np.repeat([0], 13),
            right=np.repeat([100], 13),
            parent=np.array(
                [8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13], dtype="int32"
            ),
            child=np.array([1, 2, 3, 8, 0, 7, 4, 5, 10, 6, 11, 9, 12], dtype="int32"),
        )
        tables.populations.add_row()
        tables.populations.add_row()
        tables.nodes.population = np.array(
            [0, 1, 1, 1, 0, 0, 1, 0] + [tskit.NULL] * 6, dtype="int32"
        )
        return tables.tree_sequence()

    def test_total_pairs(self):
        """
        ┊         15 pairs ┊
        ┊       ┏━━┻━━┓    ┊
        ┊       4     ┃    ┊
        ┊     ┏━┻━┓   ┃    ┊
        ┊     5   ┃   ┃    ┊
        ┊  ┏━━╋━┓ ┃   ┃    ┊
        ┊  1  ┃ ┃ ┃   2    ┊
        ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓  ┊
        ┊ ┃ ┃ ┃ ┃ ┃  1  ┃  ┊
        ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃  ┊
        ┊ 0 0 0 0 0 0 0 0  ┊
        """
        ts = self.example_ts()
        check = np.array([0.0] * 8 + [1, 2, 1, 5, 4, 15])
        implm = ts.pair_coalescence_counts()
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(ts)
        np.testing.assert_allclose(proto, check)

    def test_population_pairs(self):
        """
        ┊ AA       0 pairs ┊ AB      12 pairs ┊ BB       3 pairs ┊
        ┊       ┏━━┻━━┓    ┊       ┏━━┻━━┓    ┊       ┏━━┻━━┓    ┊
        ┊       0     ┃    ┊       4     ┃    ┊       0     ┃    ┊
        ┊     ┏━┻━┓   ┃    ┊     ┏━┻━┓   ┃    ┊     ┏━┻━┓   ┃    ┊
        ┊     5   ┃   ┃    ┊     0   ┃   ┃    ┊     0   ┃   ┃    ┊
        ┊  ┏━━╋━┓ ┃   ┃    ┊  ┏━━╋━┓ ┃   ┃    ┊  ┏━━╋━┓ ┃   ┃    ┊
        ┊  1  ┃ ┃ ┃   0    ┊  0  ┃ ┃ ┃   0    ┊  0  ┃ ┃ ┃   2    ┊
        ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓  ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓  ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓  ┊
        ┊ ┃ ┃ ┃ ┃ ┃  0  ┃  ┊ ┃ ┃ ┃ ┃ ┃  0  ┃  ┊ ┃ ┃ ┃ ┃ ┃  1  ┃  ┊
        ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃  ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃  ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃  ┊
        ┊ A A A A B B B B  ┊ A A A A B B B B  ┊ A A A A B B B B  ┊
        """
        ts = self.example_ts()
        ss0 = np.flatnonzero(ts.nodes_population == 0)
        ss1 = np.flatnonzero(ts.nodes_population == 1)
        indexes = [(0, 0), (0, 1), (1, 1)]
        implm = ts.pair_coalescence_counts(sample_sets=[ss0, ss1], indexes=indexes)
        check = np.full(implm.shape, np.nan)
        check[:, 0] = np.array([0.0] * 8 + [0, 0, 1, 5, 0, 0])
        check[:, 1] = np.array([0.0] * 8 + [0, 0, 0, 0, 4, 12])
        check[:, 2] = np.array([0.0] * 8 + [1, 2, 0, 0, 0, 3])
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts, sample_sets=[ss0, ss1], indexes=indexes
        )
        np.testing.assert_allclose(proto, check)

    def test_internal_samples(self):
        """
        ┊          Not     ┊         24 pairs ┊
        ┊       ┏━━┻━━┓    ┊       ┏━━┻━━┓    ┊
        ┊       N     ┃    ┊       5     ┃    ┊
        ┊     ┏━┻━┓   ┃    ┊     ┏━┻━┓   ┃    ┊
        ┊     S   ┃   ┃    ┊     5   ┃   ┃    ┊
        ┊  ┏━━╋━┓ ┃   ┃    ┊  ┏━━╋━┓ ┃   ┃    ┊
        ┊  N  ┃ ┃ ┃   Samp ┊  1  ┃ ┃ ┃   2    ┊
        ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓  ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓  ┊
        ┊ ┃ ┃ ┃ ┃ ┃  N  ┃  ┊ ┃ ┃ ┃ ┃ ┃  1  ┃  ┊
        ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃  ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃  ┊
        ┊ S S S S S S S S  ┊ 0 0 0 0 0 0 0 0  ┊
        """
        ts = self.example_ts()
        tables = ts.dump_tables()
        nodes_flags = tables.nodes.flags.copy()
        nodes_flags[9] = tskit.NODE_IS_SAMPLE
        nodes_flags[11] = tskit.NODE_IS_SAMPLE
        tables.nodes.flags = nodes_flags
        ts = tables.tree_sequence()
        assert ts.num_samples == 10
        implm = ts.pair_coalescence_counts(span_normalise=False)
        check = np.array([0] * 8 + [1, 2, 1, 5, 5, 24]) * ts.sequence_length
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(ts, span_normalise=False)
        np.testing.assert_allclose(proto, check)

    def test_windows(self):
        ts = self.example_ts()
        check = np.array([0.0] * 8 + [1, 2, 1, 5, 4, 15]) * ts.sequence_length / 2
        implm = ts.pair_coalescence_counts(
            windows=np.linspace(0, ts.sequence_length, 3), span_normalise=False
        )
        np.testing.assert_allclose(implm[0], check)
        np.testing.assert_allclose(implm[1], check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts, windows=np.linspace(0, ts.sequence_length, 3), span_normalise=False
        )
        np.testing.assert_allclose(proto[0], check)
        np.testing.assert_allclose(proto[1], check)

    def test_time_windows(self):
        """
           ┊         15 pairs ┊
           ┊       ┏━━┻━━┓    ┊
           ┊       4     ┃    ┊
        7.0┊-----┏━┻━┓---┃----┊
           ┊     5   ┃   ┃    ┊
        5.0┊--┏━━╋━┓-┃---┃----┊
           ┊  1  ┃ ┃ ┃   2    ┊
           ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓  ┊
           ┊ ┃ ┃ ┃ ┃ ┃  1  ┃  ┊
           ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃  ┊
        0.0┊ 0 0 0 0 0 0 0 0  ┊
        """
        ts = self.example_ts()
        time_windows = np.array([0.0, 5.0, 7.0, np.inf])
        check = np.array([4, 5, 19]) * ts.sequence_length
        implm = ts.pair_coalescence_counts(
            span_normalise=False, time_windows=time_windows
        )
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts, span_normalise=False, time_windows=time_windows
        )
        np.testing.assert_allclose(proto, check)


class TestCoalescingPairsTwoTree:
    """
    Test against worked example (two trees)
    """

    def example_ts(self, S, L):
        """
           0         S         L
        4.0┊   7     ┊   7     ┊
           ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊
        3.0┊ ┃   6   ┊ ┃   ┃   ┊
           ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊
        2.0┊ ┃ ┃  5  ┊ ┃   5   ┊
           ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊
        1.0┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊
           ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
        0.0┊ 0 1 2 3 ┊ 0 1 2 3 ┊
             A A B B   A A B B
        """
        tables = tskit.TableCollection(sequence_length=L)
        tables.nodes.set_columns(
            time=np.array([0, 0, 0, 0, 1.0, 2.0, 3.0, 4.0]),
            flags=np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype="uint32"),
        )
        tables.edges.set_columns(
            left=np.array([S, S, 0, 0, S, 0, 0, 0, S, 0]),
            right=np.array([L, L, S, L, L, S, S, L, L, S]),
            parent=np.array([4, 4, 5, 5, 5, 6, 6, 7, 7, 7], dtype="int32"),
            child=np.array([1, 2, 2, 3, 4, 1, 5, 0, 5, 6], dtype="int32"),
        )
        return tables.tree_sequence()

    def test_total_pairs(self):
        """
        ┊   3 pairs   3     ┊
        ┊ ┏━┻━┓     ┏━┻━┓   ┊
        ┊ ┃   2     ┃   ┃   ┊
        ┊ ┃ ┏━┻┓    ┃   ┃   ┊
        ┊ ┃ ┃  1    ┃   2   ┊
        ┊ ┃ ┃ ┏┻┓   ┃  ┏┻━┓ ┊
        ┊ ┃ ┃ ┃ ┃   ┃  1  ┃ ┊
        ┊ ┃ ┃ ┃ ┃   ┃ ┏┻┓ ┃ ┊
        ┊ 0 0 0 0   0 0 0 0 ┊
        0         S         L
        """
        L, S = 1e8, 1.0
        ts = self.example_ts(S, L)
        implm = ts.pair_coalescence_counts(span_normalise=False)
        check = np.array([0] * 4 + [1 * (L - S), 2 * (L - S) + 1 * S, 2 * S, 3 * L])
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(ts, span_normalise=False)
        np.testing.assert_allclose(proto, check)

    def test_population_pairs(self):
        """
        ┊AA                 ┊AB                 ┊BB                 ┊
        ┊   1 pairs   1     ┊   2 pairs   2     ┊   0 pairs   0     ┊
        ┊ ┏━┻━┓     ┏━┻━┓   ┊ ┏━┻━┓     ┏━┻━┓   ┊ ┏━┻━┓     ┏━┻━┓   ┊
        ┊ ┃   0     ┃   ┃   ┊ ┃   2     ┃   ┃   ┊ ┃   0     ┃   ┃   ┊
        ┊ ┃ ┏━┻┓    ┃   ┃   ┊ ┃ ┏━┻┓    ┃   ┃   ┊ ┃ ┏━┻┓    ┃   ┃   ┊
        ┊ ┃ ┃  0    ┃   0   ┊ ┃ ┃  0    ┃   1   ┊ ┃ ┃  1    ┃   1   ┊
        ┊ ┃ ┃ ┏┻┓   ┃  ┏┻━┓ ┊ ┃ ┃ ┏┻┓   ┃  ┏┻━┓ ┊ ┃ ┃ ┏┻┓   ┃  ┏┻━┓ ┊
        ┊ ┃ ┃ ┃ ┃   ┃  0  ┃ ┊ ┃ ┃ ┃ ┃   ┃  1  ┃ ┊ ┃ ┃ ┃ ┃   ┃  0  ┃ ┊
        ┊ ┃ ┃ ┃ ┃   ┃ ┏┻┓ ┃ ┊ ┃ ┃ ┃ ┃   ┃ ┏┻┓ ┃ ┊ ┃ ┃ ┃ ┃   ┃ ┏┻┓ ┃ ┊
        ┊ A A B B   A A B B ┊ A A B B   A A B B ┊ A A B B   A A B B ┊
        0         S         L         S         L         S         L
        """
        L, S = 1e8, 1.0
        ts = self.example_ts(S, L)
        indexes = [(0, 0), (0, 1), (1, 1)]
        implm = ts.pair_coalescence_counts(
            sample_sets=[[0, 1], [2, 3]], indexes=indexes, span_normalise=False
        )
        check = np.empty(implm.shape)
        check[:, 0] = np.array([0] * 4 + [0, 0, 0, 1 * L])
        check[:, 1] = np.array([0] * 4 + [1 * (L - S), 1 * (L - S), 2 * S, 2 * L])
        check[:, 2] = np.array([0] * 4 + [0, 1 * L, 0, 0])
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts, sample_sets=[[0, 1], [2, 3]], indexes=indexes, span_normalise=False
        )
        np.testing.assert_allclose(proto, check)

    def test_internal_samples(self):
        """
        ┊   Not       N     ┊   4 pairs   4     ┊
        ┊ ┏━┻━┓     ┏━┻━┓   ┊ ┏━┻━┓     ┏━┻━┓   ┊
        ┊ ┃   N     ┃   ┃   ┊ ┃   3     ┃   ┃   ┊
        ┊ ┃ ┏━┻┓    ┃   ┃   ┊ ┃ ┏━┻┓    ┃   ┃   ┊
        ┊ ┃ ┃  Samp ┃   S   ┊ ┃ ┃  1    ┃   2   ┊
        ┊ ┃ ┃ ┏┻┓   ┃  ┏┻━┓ ┊ ┃ ┃ ┏┻┓   ┃  ┏┻━┓ ┊
        ┊ ┃ ┃ ┃ ┃   ┃  N  ┃ ┊ ┃ ┃ ┃ ┃   ┃  1  ┃ ┊
        ┊ ┃ ┃ ┃ ┃   ┃ ┏┻┓ ┃ ┊ ┃ ┃ ┃ ┃   ┃ ┏┻┓ ┃ ┊
        ┊ S S S S   S S S S ┊ 0 0 0 0   0 0 0 0 ┊
        """
        L, S = 200, 100
        ts = self.example_ts(S, L)
        tables = ts.dump_tables()
        nodes_flags = tables.nodes.flags.copy()
        nodes_flags[5] = tskit.NODE_IS_SAMPLE
        tables.nodes.flags = nodes_flags
        ts = tables.tree_sequence()
        assert ts.num_samples == 5
        implm = ts.pair_coalescence_counts(span_normalise=False)
        check = np.array([0.0] * 4 + [(L - S), S + 2 * (L - S), 3 * S, 4 * L])
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(ts, span_normalise=False)
        np.testing.assert_allclose(proto, check)

    def test_windows(self):
        """
        ┊   3 pairs   3     ┊
        ┊ ┏━┻━┓     ┏━┻━┓   ┊
        ┊ ┃   2     ┃   ┃   ┊
        ┊ ┃ ┏━┻┓    ┃   ┃   ┊
        ┊ ┃ ┃  1    ┃   2   ┊
        ┊ ┃ ┃ ┏┻┓   ┃  ┏┻━┓ ┊
        ┊ ┃ ┃ ┃ ┃   ┃  1  ┃ ┊
        ┊ ┃ ┃ ┃ ┃   ┃ ┏┻┓ ┃ ┊
        ┊ 0 0 0 0   0 0 0 0 ┊
        0         S         L
        """
        L, S = 200, 100
        ts = self.example_ts(S, L)
        windows = np.array(list(ts.breakpoints()))
        check_0 = np.array([0.0] * 4 + [0, 1, 2, 3]) * S
        check_1 = np.array([0.0] * 4 + [1, 2, 0, 3]) * (L - S)
        implm = ts.pair_coalescence_counts(windows=windows, span_normalise=False)
        np.testing.assert_allclose(implm[0], check_0)
        np.testing.assert_allclose(implm[1], check_1)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(ts, windows=windows, span_normalise=False)
        np.testing.assert_allclose(proto[0], check_0)
        np.testing.assert_allclose(proto[1], check_1)

    def test_time_windows(self):
        """
           ┊   3 pairs   3     ┊
        3.5┊-┏━┻━┓---┊-┏━┻━┓---┊
           ┊ ┃   2   ┊ ┃   ┃   ┊
           ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊
           ┊ ┃ ┃  1  ┊ ┃   2   ┊
        1.5┊-┃-┃-┏┻┓-┊-┃--┏┻━┓-┊
           ┊ ┃ ┃ ┃ ┃ ┊ ┃  1  ┃ ┊
           ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
        0.0┊ 0 0 0 0 ┊ 0 0 0 0 ┊
           0         S         L
        """
        L, S = 200, 100
        ts = self.example_ts(S, L)
        time_windows = np.array([0.0, 1.5, 3.5, np.inf])
        windows = np.array(list(ts.breakpoints()))
        check_0 = np.array([0.0, 3.0, 3.0]) * S
        check_1 = np.array([1.0, 2.0, 3.0]) * (L - S)
        implm = ts.pair_coalescence_counts(
            span_normalise=False,
            windows=windows,
            time_windows=time_windows,
        )
        np.testing.assert_allclose(implm[0], check_0)
        np.testing.assert_allclose(implm[1], check_1)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts,
            span_normalise=False,
            windows=windows,
            time_windows=time_windows,
        )
        np.testing.assert_allclose(proto[0], check_0)
        np.testing.assert_allclose(proto[1], check_1)


class TestCoalescingPairsSimulated:
    """
    Test against a naive implementation on simulated data.
    """

    @tests.cached_example
    def example_ts(self):
        n = 10
        model = msprime.BetaCoalescent(alpha=1.5)  # polytomies
        tables = msprime.sim_ancestry(
            samples=n,
            recombination_rate=1e-8,
            sequence_length=1e6,
            population_size=1e4,
            random_seed=1024,
            model=model,
        ).dump_tables()
        tables.populations.add_row(metadata={"name": "foo", "description": "bar"})
        tables.populations.add_row(metadata={"name": "bar", "description": "foo"})
        tables.nodes.population = np.repeat(
            [0, 1, 2, tskit.NULL],
            [n, n // 2, n - n // 2, tables.nodes.num_rows - 2 * n],
        ).astype("int32")
        ts = tables.tree_sequence()
        assert ts.num_trees > 1
        return ts

    @staticmethod
    def _check_total_pairs(ts, windows):
        samples = list(ts.samples())
        implm = ts.pair_coalescence_counts(windows=windows, span_normalise=False)
        dim = (windows.size - 1, ts.num_nodes)
        check = np.full(dim, np.nan)
        for w, (a, b) in enumerate(zip(windows[:-1], windows[1:])):
            tsw = ts.keep_intervals(np.array([[a, b]]), simplify=False)
            check[w] = naive_pair_coalescence_counts(tsw, samples, samples) / 2
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(ts, windows=windows, span_normalise=False)
        np.testing.assert_allclose(proto, check)

    @staticmethod
    def _check_subset_pairs(ts, windows):
        ss0 = np.flatnonzero(ts.nodes_population == 0)
        ss1 = np.flatnonzero(ts.nodes_population == 1)
        idx = [(0, 1), (1, 1), (0, 0)]
        implm = ts.pair_coalescence_counts(
            sample_sets=[ss0, ss1], indexes=idx, windows=windows, span_normalise=False
        )
        dim = (windows.size - 1, ts.num_nodes, len(idx))
        check = np.full(dim, np.nan)
        for w, (a, b) in enumerate(zip(windows[:-1], windows[1:])):
            tsw = ts.keep_intervals(np.array([[a, b]]), simplify=False)
            check[w, :, 0] = naive_pair_coalescence_counts(tsw, ss0, ss1)
            check[w, :, 1] = naive_pair_coalescence_counts(tsw, ss1, ss1) / 2
            check[w, :, 2] = naive_pair_coalescence_counts(tsw, ss0, ss0) / 2
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts,
            sample_sets=[ss0, ss1],
            indexes=idx,
            windows=windows,
            span_normalise=False,
        )
        np.testing.assert_allclose(proto, check)

    def test_sequence(self):
        ts = self.example_ts()
        windows = np.array([0.0, ts.sequence_length])
        self._check_total_pairs(ts, windows)
        self._check_subset_pairs(ts, windows)

    def test_missing_interval(self):
        """
        test case where three segments have all samples missing
        """
        ts = self.example_ts()
        windows = np.array([0.0, ts.sequence_length])
        intervals = np.array([[0.0, 0.1], [0.4, 0.6], [0.9, 1.0]]) * ts.sequence_length
        ts = ts.delete_intervals(intervals)
        self._check_total_pairs(ts, windows)
        self._check_subset_pairs(ts, windows)

    def test_missing_leaves(self):
        """
        test case where 1/2 of samples are missing
        """
        t = self.example_ts().dump_tables()
        ss0 = np.flatnonzero(t.nodes.population == 0)
        remove = np.isin(t.edges.child, ss0)
        assert np.any(remove)
        t.edges.set_columns(
            left=t.edges.left[~remove],
            right=t.edges.right[~remove],
            parent=t.edges.parent[~remove],
            child=t.edges.child[~remove],
        )
        t.sort()
        ts = t.tree_sequence()
        windows = np.array([0.0, ts.sequence_length])
        self._check_total_pairs(ts, windows)
        self._check_subset_pairs(ts, windows)

    def test_missing_roots(self):
        """
        test case where all trees have multiple roots
        """
        ts = self.example_ts()
        ts = ts.decapitate(np.quantile(ts.nodes_time, 0.75))
        windows = np.array([0.0, ts.sequence_length])
        self._check_total_pairs(ts, windows)
        self._check_subset_pairs(ts, windows)

    def test_windows(self):
        ts = self.example_ts()
        windows = np.linspace(0.0, ts.sequence_length, 9)
        self._check_total_pairs(ts, windows)
        self._check_subset_pairs(ts, windows)

    def test_windows_are_trees(self):
        """
        test case where window breakpoints coincide with tree breakpoints
        """
        ts = self.example_ts()
        windows = np.array(list(ts.breakpoints()))
        self._check_total_pairs(ts, windows)
        self._check_subset_pairs(ts, windows)

    def test_windows_inside_trees(self):
        """
        test case where windows are nested within trees
        """
        ts = self.example_ts()
        windows = np.array(list(ts.breakpoints()))
        windows = np.sort(np.append(windows[:-1] / 2 + windows[1:] / 2, windows))
        self._check_total_pairs(ts, windows)
        self._check_subset_pairs(ts, windows)

    def test_nonsuccinct_sequence(self):
        """
        test case where each tree has distinct nodes
        """
        ts = convert_to_nonsuccinct(self.example_ts())
        windows = np.linspace(0, ts.sequence_length, 9)
        self._check_total_pairs(ts, windows)
        self._check_subset_pairs(ts, windows)

    def test_span_normalise(self):
        """
        test case where span is normalised
        """
        ts = self.example_ts()
        windows = np.array([0.0, 0.33, 1.0]) * ts.sequence_length
        window_size = np.diff(windows)
        implm = ts.pair_coalescence_counts(windows=windows, span_normalise=False)
        check = ts.pair_coalescence_counts(windows=windows) * window_size[:, np.newaxis]
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(ts, windows=windows, span_normalise=False)
        np.testing.assert_allclose(proto, check)

    def test_internal_nodes_are_samples(self):
        """
        test case where some samples are descendants of other samples
        """
        ts = self.example_ts()
        tables = ts.dump_tables()
        nodes_flags = tables.nodes.flags.copy()
        nodes_sample = np.arange(ts.num_samples, ts.num_nodes, 10)
        nodes_flags[nodes_sample] = tskit.NODE_IS_SAMPLE
        tables.nodes.flags = nodes_flags
        ts_modified = tables.tree_sequence()
        assert ts_modified.num_samples > ts.num_samples
        windows = np.linspace(0.0, 1.0, 9) * ts_modified.sequence_length
        self._check_total_pairs(ts_modified, windows)
        self._check_subset_pairs(ts_modified, windows)

    def test_time_windows(self):
        ts = self.example_ts()
        total_pair_count = ts.pair_coalescence_counts(
            time_windows=np.array([0.0, np.inf]),
            span_normalise=False,
        )[0]
        samples = list(ts.samples())
        time_windows = np.quantile(ts.nodes_time, [0.0, 0.25, 0.5, 0.75])
        time_windows = np.append(time_windows, np.inf)
        implm = ts.pair_coalescence_counts(
            span_normalise=False, time_windows=time_windows
        )
        assert np.isclose(np.sum(implm), total_pair_count)
        check = naive_pair_coalescence_counts(ts, samples, samples).squeeze() / 2
        nodes_map = np.searchsorted(time_windows, ts.nodes_time, side="right") - 1
        check = np.bincount(nodes_map, weights=check)
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts, span_normalise=False, time_windows=time_windows
        )
        assert np.isclose(np.sum(proto), total_pair_count)
        np.testing.assert_allclose(proto, check)

    def test_time_windows_truncated(self):
        """
        test case where some nodes fall outside of time bins
        """
        ts = self.example_ts()
        total_pair_count = ts.pair_coalescence_counts(
            time_windows=np.array([0.0, np.inf]),
            span_normalise=False,
        )[0]
        samples = list(ts.samples())
        time_windows = np.quantile(ts.nodes_time, [0.5, 0.75])
        assert time_windows[0] > 0.0
        time_windows = np.append(time_windows, np.inf)
        implm = ts.pair_coalescence_counts(
            span_normalise=False, time_windows=time_windows
        )
        assert np.sum(implm) < total_pair_count
        check = naive_pair_coalescence_counts(ts, samples, samples).squeeze() / 2
        nodes_map = np.searchsorted(time_windows, ts.nodes_time, side="right") - 1
        oob = np.logical_or(nodes_map < 0, nodes_map >= time_windows.size)
        check = np.bincount(nodes_map[~oob], weights=check[~oob])
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts, span_normalise=False, time_windows=time_windows
        )
        assert np.sum(proto) < total_pair_count
        np.testing.assert_allclose(proto, check)

    def test_time_windows_unique(self):
        ts = self.example_ts()
        total_pair_count = ts.pair_coalescence_counts(
            time_windows=np.array([0.0, np.inf]),
            span_normalise=False,
        )[0]
        samples = list(ts.samples())
        time_windows = np.unique(ts.nodes_time)
        time_windows = np.append(time_windows, np.inf)
        implm = ts.pair_coalescence_counts(
            span_normalise=False, time_windows=time_windows
        )
        assert np.isclose(np.sum(implm), total_pair_count)
        check = naive_pair_coalescence_counts(ts, samples, samples).squeeze() / 2
        nodes_map = np.searchsorted(time_windows, ts.nodes_time, side="right") - 1
        check = np.bincount(nodes_map, weights=check)
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts, span_normalise=False, time_windows=time_windows
        )
        assert np.isclose(np.sum(proto), total_pair_count)
        np.testing.assert_allclose(proto, check)

    def test_diversity(self):
        """
        test that weighted mean of node times equals branch diversity
        """
        ts = self.example_ts()
        windows = np.linspace(0.0, ts.sequence_length, 9)
        check = ts.diversity(mode="branch", windows=windows)
        implm = ts.pair_coalescence_counts(windows=windows)
        implm = 2 * (implm @ ts.nodes_time) / implm.sum(axis=1)
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(ts, windows=windows)
        proto = 2 * (proto @ ts.nodes_time) / proto.sum(axis=1)
        np.testing.assert_allclose(proto, check)

    def test_divergence(self):
        """
        test that weighted mean of node times equals branch divergence
        """
        ts = self.example_ts()
        ss0 = np.flatnonzero(ts.nodes_population == 0)
        ss1 = np.flatnonzero(ts.nodes_population == 1)
        windows = np.linspace(0.0, ts.sequence_length, 9)
        check = ts.divergence(sample_sets=[ss0, ss1], mode="branch", windows=windows)
        implm = ts.pair_coalescence_counts(sample_sets=[ss0, ss1], windows=windows)
        implm = 2 * (implm @ ts.nodes_time) / implm.sum(axis=1)
        np.testing.assert_allclose(implm, check)
        # TODO: remove with prototype
        proto = proto_pair_coalescence_counts(
            ts, sample_sets=[ss0, ss1], windows=windows
        )
        proto = 2 * (proto @ ts.nodes_time) / proto.sum(axis=1)
        np.testing.assert_allclose(proto, check)


@pytest.mark.skip("Not implemented")
class TestQuantileReduction:
    """
    Test quantile reduction
    """

    @tests.cached_example
    def example_ts(self):
        n = 10
        model = msprime.BetaCoalescent(alpha=1.5)  # polytomies
        tables = msprime.sim_ancestry(
            samples=n,
            recombination_rate=1e-8,
            sequence_length=1e6,
            population_size=1e4,
            random_seed=1024,
            model=model,
        ).dump_tables()
        tables.populations.add_row(metadata={"name": "foo", "description": "bar"})
        tables.nodes.population = np.repeat(
            [0, 1, tskit.NULL], [n, n, tables.nodes.num_rows - 2 * n]
        ).astype("int32")
        ts = tables.tree_sequence()
        assert ts.num_trees > 1
        return ts

    def test_quantiles(self):
        ts = self.example_ts()
        quantiles = np.linspace(0, 1, 10)
        weights = ts.pair_coalescence_counts()
        check = _numpy_weighted_quantile(ts.nodes_time, weights, quantiles)
        # TODO: switch to TreeSequence method
        implm = proto_pair_coalescence_quantiles(ts, quantiles=quantiles)
        np.testing.assert_allclose(implm, check)

    def test_boundary_quantiles(self):
        ts = self.example_ts()
        weights = ts.pair_coalescence_counts()
        quantiles = np.unique(weights / np.sum(weights))
        check = _numpy_weighted_quantile(ts.nodes_time, weights, quantiles)
        # TODO: switch to TreeSequence method
        implm = proto_pair_coalescence_quantiles(ts, quantiles=quantiles)
        np.testing.assert_allclose(implm, check)


class TestCoalescingPairsUsage:
    """
    Test invalid inputs
    """

    @tests.cached_example
    def example_ts(self):
        return msprime.sim_ancestry(
            samples=10,
            recombination_rate=1e-8,
            sequence_length=1e5,
            population_size=1e4,
            random_seed=1024,
        )

    def test_oor_windows(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="must be sequence boundary"):
            ts.pair_coalescence_counts(
                windows=np.array([0.0, 2.0]) * ts.sequence_length
            )

    def test_unsorted_windows(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="must be strictly increasing"):
            ts.pair_coalescence_counts(
                windows=np.array([0.0, 0.3, 0.2, 1.0]) * ts.sequence_length
            )

    def test_bad_windows(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="must be an array of breakpoints"):
            ts.pair_coalescence_counts(windows="whatever")
        with pytest.raises(ValueError, match="must be an array of breakpoints"):
            ts.pair_coalescence_counts(windows=np.array([0.0]))

    def test_empty_sample_sets(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="contain at least one element"):
            ts.pair_coalescence_counts(sample_sets=[[0, 1, 2], []])

    def test_oob_sample_sets(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="is out of bounds"):
            ts.pair_coalescence_counts(sample_sets=[[0, ts.num_nodes]])

    def test_nonbinary_indexes(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="must be length two"):
            ts.pair_coalescence_counts(indexes=[(0, 0, 0)])

    def test_oob_indexes(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="is out of bounds"):
            ts.pair_coalescence_counts(indexes=[(0, 1)])

    def test_no_indexes(self):
        ts = self.example_ts()
        ss = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        with pytest.raises(ValueError, match="more than two sample sets"):
            ts.pair_coalescence_counts(sample_sets=ss)

    def test_oob_samples(self):
        ts = self.example_ts()
        sample_sets = [np.arange(ts.num_samples + 1)]
        with pytest.raises(tskit.LibraryError, match="are not samples"):
            ts.pair_coalescence_counts(sample_sets=sample_sets)

    def test_uncalibrated_time(self):
        tables = self.example_ts().dump_tables()
        tables.time_units = tskit.TIME_UNITS_UNCALIBRATED
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="require calibrated node times"):
            ts.pair_coalescence_counts(time_windows=np.array([0.0, np.inf]))

    def test_bad_time_windows(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="must be an array of breakpoints"):
            ts.pair_coalescence_counts(time_windows="whatever")
        with pytest.raises(ValueError, match="must be an array of breakpoints"):
            ts.pair_coalescence_counts(time_windows=np.array([0.0]))

    def test_unsorted_time_windows(self):
        ts = self.example_ts()
        time_windows = np.array([0.0, 12.0, 6.0, np.inf])
        with pytest.raises(ValueError, match="must be strictly increasing"):
            ts.pair_coalescence_counts(time_windows=time_windows)

    def test_empty_time_windows(self):
        ts = self.example_ts()
        time_windows = [np.max(ts.nodes_time) + 1, np.max(ts.nodes_time) + 2]
        time_windows = np.append(time_windows, np.inf)
        with pytest.raises(ValueError, match="has null values for all nodes"):
            ts.pair_coalescence_counts(time_windows=time_windows)

    def test_output_dim(self):
        """
        test that output dimensions corresponding to None arguments are dropped
        """
        ts = self.example_ts()
        ss = [[0, 1, 2], [3, 4, 5]]
        implm = ts.pair_coalescence_counts(sample_sets=ss, windows=None, indexes=None)
        assert implm.shape == (ts.num_nodes,)
        windows = np.linspace(0.0, ts.sequence_length, 2)
        implm = ts.pair_coalescence_counts(
            sample_sets=ss, windows=windows, indexes=None
        )
        assert implm.shape == (1, ts.num_nodes)
        indexes = [(0, 1)]
        implm = ts.pair_coalescence_counts(
            sample_sets=ss, windows=windows, indexes=indexes
        )
        assert implm.shape == (1, ts.num_nodes, 1)
        implm = ts.pair_coalescence_counts(
            sample_sets=ss, windows=None, indexes=indexes
        )
        assert implm.shape == (ts.num_nodes, 1)
