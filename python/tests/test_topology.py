# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (c) 2016-2017 University of Oxford
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
Test cases for the supported topological variations and operations.
"""
import io
import unittest
import itertools
import random
import json
import sys
import math

import numpy as np
import msprime

import tskit
import _tskit
import tskit.provenance as provenance
import tests as tests
import tests.tsutil as tsutil
import tests.test_wright_fisher as wf


def ts_equal(ts_1, ts_2, compare_provenances=True):
    """
    Check equality of tree sequences, ignoring provenance timestamps (but not contents)
    """
    return tables_equal(ts_1.tables, ts_2.tables, compare_provenances)


def tables_equal(table_collection_1, table_collection_2, compare_provenances=True):
    """
    Check equality of tables, ignoring provenance timestamps (but not contents)
    """
    for (_, table_1), (_, table_2) in zip(table_collection_1, table_collection_2):
        if isinstance(table_1, tskit.ProvenanceTable):
            if compare_provenances:
                if np.any(table_1.record != table_2.record):
                    return False
                if np.any(table_1.record_offset != table_2.record_offset):
                    return False
        else:
            if table_1 != table_2:
                return False
    return True


def simple_keep_intervals(tables, intervals, simplify=True, record_provenance=True):
    """
    Simple Python implementation of keep_intervals.
    """
    ts = tables.tree_sequence()
    last_stop = 0
    for start, stop in intervals:
        if start < 0 or stop > ts.sequence_length:
            raise ValueError("Slice bounds must be within the existing tree sequence")
        if start >= stop:
            raise ValueError("Interval error: start must be < stop")
        if start < last_stop:
            raise ValueError("Intervals must be disjoint")
        last_stop = stop
    tables.edges.clear()
    tables.sites.clear()
    tables.mutations.clear()
    for edge in ts.edges():
        for interval_left, interval_right in intervals:
            if not (edge.right <= interval_left or edge.left >= interval_right):
                left = max(interval_left, edge.left)
                right = min(interval_right, edge.right)
                tables.edges.add_row(left, right, edge.parent, edge.child)
    for site in ts.sites():
        for interval_left, interval_right in intervals:
            if interval_left <= site.position < interval_right:
                site_id = tables.sites.add_row(
                    site.position, site.ancestral_state, site.metadata)
                for m in site.mutations:
                    tables.mutations.add_row(
                        site_id, m.node, m.derived_state, tskit.NULL, m.metadata)
    tables.build_index()
    tables.compute_mutation_parents()
    if simplify:
        tables.simplify()
    if record_provenance:
        parameters = {
            "command": "keep_intervals",
            "TODO": "add parameters"
        }
        tables.provenances.add_row(record=json.dumps(
            provenance.get_provenance_dict(parameters)))


def generate_segments(n, sequence_length=100, seed=None):
    rng = random.Random(seed)
    segs = []
    for j in range(n):
        left = rng.randint(0, sequence_length - 1)
        right = rng.randint(left + 1, sequence_length)
        assert left < right
        segs.append(tests.Segment(left, right, j))
    return segs


def kc_distance(tree1, tree2, lambda_=0):
    """
    Returns the Kendall-Colijn distance between the specified pair of trees.
    lambda_ determines weight of topology vs branch lengths in calculating
    the distance. Set lambda_ at 0 to only consider topology, set at 1 to
    only consider branch lengths. See Kendall & Colijn (2016):
    https://academic.oup.com/mbe/article/33/10/2735/2925548
    """
    samples = tree1.tree_sequence.samples()
    if not np.array_equal(samples, tree2.tree_sequence.samples()):
        raise ValueError("Trees must have the same samples")
    if not len(tree1.roots) == len(tree2.roots) == 1:
        raise ValueError("Trees must have one root")
    sample_index_map = np.zeros(tree1.tree_sequence.num_nodes, dtype=int) - 1
    for j, u in enumerate(samples):
        sample_index_map[u] = j
        if not tree1.is_leaf(u) or not tree2.is_leaf(u):
            raise ValueError("Internal samples not supported")

    k = samples.shape[0]
    n = (k * (k - 1)) // 2
    m = [np.ones(n + k), np.ones(n + k)]
    M = [np.zeros(n + k), np.zeros(n + k)]
    for tree_index, tree in enumerate([tree1, tree2]):
        stack = [(tree.root, 0, tree.time(tree.root))]
        while len(stack) > 0:
            node, depth, time = stack.pop()
            children = tree.children(node)
            for child in children:
                stack.append((child, depth + 1, tree.time(child)))
            for c1, c2 in itertools.combinations(children, 2):
                for v1 in tree.samples(c1):
                    index1 = sample_index_map[v1]
                    for v2 in tree.samples(c2):
                        index2 = sample_index_map[v2]
                        a = min(index1, index2)
                        b = max(index1, index2)
                        pair_index = a * (a - 2 * k + 1) // -2 + b - a - 1
                        assert m[tree_index][pair_index] == 1
                        m[tree_index][pair_index] = depth
                        M[tree_index][pair_index] = tree.time(tree.root) - time
            if len(tree.children(node)) == 0:
                index = sample_index_map[node]
                M[tree_index][index + n] = tree.branch_length(node)
    return np.linalg.norm((1 - lambda_) *
                          (m[0] - m[1]) + lambda_ * (M[0] - M[1]))


def kc_distance_simple(tree1, tree2, lambda_=0):
    """
    Simplified version of the kc_distance() function above.
    Written without Python features to aid writing C implementation.
    """
    samples = tree1.tree_sequence.samples()
    for sample1, sample2 in zip(samples, tree2.tree_sequence.samples()):
        if sample1 != sample2:
            raise ValueError("Trees must have the same samples")
    if not len(tree1.roots) == len(tree2.roots) == 1:
        raise ValueError("Trees must have one root")
    sample_index_map = np.zeros(tree1.tree_sequence.num_nodes, dtype=int) - 1
    for j, u in enumerate(samples):
        sample_index_map[u] = j
        if not tree1.is_leaf(u) or not tree2.is_leaf(u):
            raise ValueError("Internal samples not supported")

    n = samples.shape[0]
    N = (n * (n - 1)) // 2
    m = [np.ones(N + n), np.ones(N + n)]
    M = [np.zeros(N + n), np.zeros(N + n)]
    path_distance = [np.zeros(tree1.num_nodes), np.zeros(tree2.num_nodes)]
    time_distance = [np.zeros(tree1.num_nodes), np.zeros(tree2.num_nodes)]
    for tree_index, tree in enumerate([tree1, tree2]):
        stack = [(tree.root, 0, tree.time(tree.root))]
        while len(stack) > 0:
            u, depth, time = stack.pop()
            children = tree.children(u)
            for v in children:
                stack.append((v, depth + 1, tree.time(v)))
            path_distance[tree_index][u] = depth
            time_distance[tree_index][u] = tree.time(tree.root) - time
            if len(tree.children(u)) == 0:
                u_index = sample_index_map[u]
                M[tree_index][u_index + N] = tree.branch_length(u)

        for n1 in range(n):
            for n2 in range(n1 + 1, n):
                mrca = tree.mrca(samples[n1], samples[n2])
                pair_index = n1 * (n1 - 2 * n + 1) // -2 + n2 - n1 - 1
                assert m[tree_index][pair_index] == 1
                m[tree_index][pair_index] = path_distance[tree_index][mrca]
                M[tree_index][pair_index] = time_distance[tree_index][mrca]

    vT1 = 0
    vT2 = 0
    distance_sum = 0
    for i in range(N + n):
        vT1 = (m[0][i] * (1 - lambda_)) + (lambda_ * M[0][i])
        vT2 = (m[1][i] * (1 - lambda_)) + (lambda_ * M[1][i])
        distance_sum += (vT1 - vT2) ** 2

    return math.sqrt(distance_sum)


class TestKCMetric(unittest.TestCase):
    """
    Tests on the KC metric distances.
    """
    def test_same_tree_zero_distance(self):
        for n in range(2, 10):
            for seed in range(1, 10):
                ts = msprime.simulate(n, random_seed=seed)
                tree = ts.first()
                self.assertEqual(kc_distance(tree, tree), 0)
                self.assertEqual(kc_distance_simple(tree, tree), 0)
                self.assertEqual(tree.kc_distance(tree), 0)
                ts = msprime.simulate(n, random_seed=seed)
                tree2 = ts.first()
                self.assertEqual(kc_distance(tree, tree2), 0)
                self.assertEqual(kc_distance_simple(tree, tree2), 0)
                self.assertEqual(tree.kc_distance(tree2), 0)

    def test_sample_2_zero_distance(self):
        # All trees with 2 leaves must be equal distance from each other.
        for seed in range(1, 10):
            tree1 = msprime.simulate(2, random_seed=seed).first()
            tree2 = msprime.simulate(2, random_seed=seed + 1).first()
            self.assertEqual(kc_distance(tree1, tree2, 0), 0)
            self.assertEqual(kc_distance_simple(tree1, tree2, 0), 0)
            self.assertEqual(tree1.kc_distance(tree2, 0), 0)

    def test_different_samples_error(self):
        tree1 = msprime.simulate(10, random_seed=1).first()
        tree2 = msprime.simulate(2, random_seed=1).first()
        self.assertRaises(ValueError, kc_distance, tree1, tree2)
        self.assertRaises(ValueError, kc_distance_simple, tree1, tree2)
        self.assertRaises(_tskit.LibraryError, tree1.kc_distance, tree2)

        ts1 = msprime.simulate(10, random_seed=1)
        nmap = np.arange(0, ts1.num_nodes)[::-1]
        ts2 = tsutil.permute_nodes(ts1, nmap)
        tree1 = ts1.first()
        tree2 = ts2.first()
        self.assertRaises(ValueError, kc_distance, tree1, tree2)
        self.assertRaises(ValueError, kc_distance_simple, tree1, tree2)
        self.assertRaises(_tskit.LibraryError, tree1.kc_distance, tree2)

    def validate_trees(self, n):
        for seed in range(1, 10):
            tree1 = msprime.simulate(n, random_seed=seed).first()
            tree2 = msprime.simulate(n, random_seed=seed + 1).first()
            kc1 = kc_distance(tree1, tree2)
            kc2 = kc_distance_simple(tree1, tree2)
            kc3 = tree1.kc_distance(tree2)
            self.assertAlmostEqual(kc1, kc2)
            self.assertAlmostEqual(kc1, kc3)
            self.assertAlmostEqual(kc1, kc_distance(tree2, tree1))
            self.assertAlmostEqual(
                kc2, kc_distance_simple(tree2, tree1))
            self.assertAlmostEqual(kc3, tree2.kc_distance(tree1))

    def test_sample_3(self):
        self.validate_trees(3)

    def test_sample_4(self):
        self.validate_trees(4)

    def test_sample_10(self):
        self.validate_trees(10)

    def test_sample_20(self):
        self.validate_trees(20)

    def validate_nonbinary_trees(self, n):
        demographic_events = [
            msprime.SimpleBottleneck(0.02, 0, proportion=0.25),
            msprime.SimpleBottleneck(0.2, 0, proportion=1)]

        for seed in range(1, 10):
            ts = msprime.simulate(
                n, random_seed=seed, demographic_events=demographic_events)
            # Check if this is really nonbinary
            found = False
            for edgeset in ts.edgesets():
                if len(edgeset.children) > 2:
                    found = True
                    break
            self.assertTrue(found)
            tree1 = ts.first()

            ts = msprime.simulate(
                n, random_seed=seed + 1, demographic_events=demographic_events)
            tree2 = ts.first()
            self.assertAlmostEqual(
                kc_distance(tree1, tree2), kc_distance(tree1, tree2))
            self.assertAlmostEqual(
                kc_distance(tree2, tree1), kc_distance(tree2, tree1))
            self.assertAlmostEqual(
                kc_distance_simple(tree1, tree2),
                kc_distance_simple(tree1, tree2))
            self.assertAlmostEqual(
                kc_distance_simple(tree2, tree1),
                kc_distance_simple(tree2, tree1))
            # compare to a binary tree also
            tree2 = msprime.simulate(n, random_seed=seed + 1).first()
            self.assertAlmostEqual(
                kc_distance(tree1, tree2), kc_distance(tree1, tree2))
            self.assertAlmostEqual(
                kc_distance(tree2, tree1), kc_distance(tree2, tree1))
            self.assertAlmostEqual(
                kc_distance_simple(tree1, tree2),
                kc_distance_simple(tree1, tree2))
            self.assertAlmostEqual(
                kc_distance_simple(tree2, tree1),
                kc_distance_simple(tree2, tree1))

    def test_non_binary_sample_10(self):
        self.validate_nonbinary_trees(10)

    def test_non_binary_sample_20(self):
        self.validate_nonbinary_trees(20)

    def test_non_binary_sample_30(self):
        self.validate_nonbinary_trees(30)

    def verify_result(self, tree1, tree2, lambda_, result, places=None):
        kc1 = kc_distance(tree1, tree2, lambda_)
        kc2 = kc_distance_simple(tree1, tree2, lambda_)
        kc3 = tree1.kc_distance(tree2, lambda_)
        self.assertAlmostEqual(kc1, result, places=places)
        self.assertAlmostEqual(kc2, result, places=places)
        self.assertAlmostEqual(kc3, result, places=places)

        kc1 = kc_distance(tree2, tree1, lambda_)
        kc2 = kc_distance_simple(tree2, tree1, lambda_)
        kc3 = tree2.kc_distance(tree1, lambda_)
        self.assertAlmostEqual(kc1, result, places=places)
        self.assertAlmostEqual(kc2, result, places=places)
        self.assertAlmostEqual(kc3, result, places=places)

    def test_known_kc_sample_3(self):
        # Test with hardcoded known values
        tables_1 = tskit.TableCollection(sequence_length=1.0)
        tables_2 = tskit.TableCollection(sequence_length=1.0)

        # Nodes
        sv = [True, True, True, False, False]
        tv_1 = [0.0, 0.0, 0.0, 2.0, 3.0]
        tv_2 = [0.0, 0.0, 0.0, 4.0, 6.0]

        for is_sample, t1, t2 in zip(sv, tv_1, tv_2):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables_1.nodes.add_row(flags=flags, time=t1)
            tables_2.nodes.add_row(flags=flags, time=t2)

        # Edges
        lv = [0.0, 0.0, 0.0, 0.0]
        rv = [1.0, 1.0, 1.0, 1.0]
        pv = [3, 3, 4, 4]
        cv = [0, 1, 2, 3]

        for l, r, p, c in zip(lv, rv, pv, cv):
            tables_1.edges.add_row(left=l, right=r, parent=p, child=c)
            tables_2.edges.add_row(left=l, right=r, parent=p, child=c)

        tree_1 = tables_1.tree_sequence().first()
        tree_2 = tables_2.tree_sequence().first()
        self.verify_result(tree_1, tree_2, 0, 0)
        self.verify_result(tree_1, tree_2, 1, 4.243, places=3)

    def test_10_samples(self):
        nodes_1 = io.StringIO("""\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    0   -1  b''
        1   1   0.000000    0   -1  b''
        2   1   0.000000    0   -1  b''
        3   1   0.000000    0   -1  b''
        4   1   0.000000    0   -1  b''
        5   1   0.000000    0   -1  b''
        6   1   0.000000    0   -1  b''
        7   1   0.000000    0   -1  b''
        8   1   0.000000    0   -1  b''
        9   1   0.000000    0   -1  b''
        10  0   0.047734    0   -1  b''
        11  0   0.061603    0   -1  b''
        12  0   0.189503    0   -1  b''
        13  0   0.275885    0   -1  b''
        14  0   0.518301    0   -1  b''
        15  0   0.543143    0   -1  b''
        16  0   0.865193    0   -1  b''
        17  0   1.643658    0   -1  b''
        18  0   2.942350    0   -1  b''
        """)
        edges_1 = io.StringIO("""\
        left    right   parent  child
        0.000000    10000.000000    10  0
        0.000000    10000.000000    10  2
        0.000000    10000.000000    11  9
        0.000000    10000.000000    11  10
        0.000000    10000.000000    12  3
        0.000000    10000.000000    12  7
        0.000000    10000.000000    13  5
        0.000000    10000.000000    13  11
        0.000000    10000.000000    14  1
        0.000000    10000.000000    14  8
        0.000000    10000.000000    15  4
        0.000000    10000.000000    15  14
        0.000000    10000.000000    16  13
        0.000000    10000.000000    16  15
        0.000000    10000.000000    17  6
        0.000000    10000.000000    17  12
        0.000000    10000.000000    18  16
        0.000000    10000.000000    18  17
        """)
        ts_1 = tskit.load_text(nodes_1, edges_1, sequence_length=10000,
                               strict=False, base64_metadata=False)
        nodes_2 = io.StringIO("""\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    0   -1  b''
        1   1   0.000000    0   -1  b''
        2   1   0.000000    0   -1  b''
        3   1   0.000000    0   -1  b''
        4   1   0.000000    0   -1  b''
        5   1   0.000000    0   -1  b''
        6   1   0.000000    0   -1  b''
        7   1   0.000000    0   -1  b''
        8   1   0.000000    0   -1  b''
        9   1   0.000000    0   -1  b''
        10  0   0.210194    0   -1  b''
        11  0   0.212217    0   -1  b''
        12  0   0.223341    0   -1  b''
        13  0   0.272703    0   -1  b''
        14  0   0.443553    0   -1  b''
        15  0   0.491653    0   -1  b''
        16  0   0.729369    0   -1  b''
        17  0   1.604113    0   -1  b''
        18  0   1.896332    0   -1  b''
        """)
        edges_2 = io.StringIO("""\
        left    right   parent  child
        0.000000    10000.000000    10  5
        0.000000    10000.000000    10  7
        0.000000    10000.000000    11  3
        0.000000    10000.000000    11  4
        0.000000    10000.000000    12  6
        0.000000    10000.000000    12  9
        0.000000    10000.000000    13  10
        0.000000    10000.000000    13  12
        0.000000    10000.000000    14  8
        0.000000    10000.000000    14  11
        0.000000    10000.000000    15  1
        0.000000    10000.000000    15  2
        0.000000    10000.000000    16  13
        0.000000    10000.000000    16  14
        0.000000    10000.000000    17  0
        0.000000    10000.000000    17  16
        0.000000    10000.000000    18  15
        0.000000    10000.000000    18  17
        """)
        ts_2 = tskit.load_text(nodes_2, edges_2, sequence_length=10000,
                               strict=False, base64_metadata=False)

        tree_1 = ts_1.first()
        tree_2 = ts_2.first()
        self.verify_result(tree_1, tree_2, 0, 12.85, places=2)
        self.verify_result(tree_1, tree_2, 1, 10.64, places=2)

    def test_15_samples(self):
        nodes_1 = io.StringIO("""\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    0   -1
        1   1   0.000000    0   -1
        2   1   0.000000    0   -1
        3   1   0.000000    0   -1
        4   1   0.000000    0   -1
        5   1   0.000000    0   -1
        6   1   0.000000    0   -1
        7   1   0.000000    0   -1
        8   1   0.000000    0   -1
        9   1   0.000000    0   -1
        10  1   0.000000    0   -1
        11  1   0.000000    0   -1
        12  1   0.000000    0   -1
        13  1   0.000000    0   -1
        14  1   0.000000    0   -1
        15  0   0.026043    0   -1
        16  0   0.032662    0   -1
        17  0   0.072032    0   -1
        18  0   0.086792    0   -1
        19  0   0.130699    0   -1
        20  0   0.177640    0   -1
        21  0   0.199800    0   -1
        22  0   0.236391    0   -1
        23  0   0.342445    0   -1
        24  0   0.380356    0   -1
        25  0   0.438502    0   -1
        26  0   0.525632    0   -1
        27  0   1.180078    0   -1
        28  0   2.548099    0   -1
        """)
        edges_1 = io.StringIO("""\
        left    right   parent  child
        0.000000    10000.000000    15  6
        0.000000    10000.000000    15  13
        0.000000    10000.000000    16  1
        0.000000    10000.000000    16  4
        0.000000    10000.000000    17  0
        0.000000    10000.000000    17  7
        0.000000    10000.000000    18  2
        0.000000    10000.000000    18  17
        0.000000    10000.000000    19  5
        0.000000    10000.000000    19  9
        0.000000    10000.000000    20  12
        0.000000    10000.000000    20  15
        0.000000    10000.000000    21  8
        0.000000    10000.000000    21  20
        0.000000    10000.000000    22  11
        0.000000    10000.000000    22  21
        0.000000    10000.000000    23  10
        0.000000    10000.000000    23  22
        0.000000    10000.000000    24  14
        0.000000    10000.000000    24  16
        0.000000    10000.000000    25  18
        0.000000    10000.000000    25  19
        0.000000    10000.000000    26  23
        0.000000    10000.000000    26  24
        0.000000    10000.000000    27  25
        0.000000    10000.000000    27  26
        0.000000    10000.000000    28  3
        0.000000    10000.000000    28  27
        """)
        ts_1 = tskit.load_text(nodes_1, edges_1, sequence_length=10000,
                               strict=False, base64_metadata=False)

        nodes_2 = io.StringIO("""\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    0   -1
        1   1   0.000000    0   -1
        2   1   0.000000    0   -1
        3   1   0.000000    0   -1
        4   1   0.000000    0   -1
        5   1   0.000000    0   -1
        6   1   0.000000    0   -1
        7   1   0.000000    0   -1
        8   1   0.000000    0   -1
        9   1   0.000000    0   -1
        10  1   0.000000    0   -1
        11  1   0.000000    0   -1
        12  1   0.000000    0   -1
        13  1   0.000000    0   -1
        14  1   0.000000    0   -1
        15  0   0.011443    0   -1
        16  0   0.055694    0   -1
        17  0   0.061677    0   -1
        18  0   0.063416    0   -1
        19  0   0.163014    0   -1
        20  0   0.223445    0   -1
        21  0   0.251724    0   -1
        22  0   0.268749    0   -1
        23  0   0.352039    0   -1
        24  0   0.356134    0   -1
        25  0   0.399454    0   -1
        26  0   0.409174    0   -1
        27  0   2.090839    0   -1
        28  0   3.772716    0   -1
        """)
        edges_2 = io.StringIO("""\
        left    right   parent  child
        0.000000    10000.000000    15  6
        0.000000    10000.000000    15  8
        0.000000    10000.000000    16  9
        0.000000    10000.000000    16  12
        0.000000    10000.000000    17  3
        0.000000    10000.000000    17  4
        0.000000    10000.000000    18  13
        0.000000    10000.000000    18  16
        0.000000    10000.000000    19  2
        0.000000    10000.000000    19  11
        0.000000    10000.000000    20  1
        0.000000    10000.000000    20  17
        0.000000    10000.000000    21  0
        0.000000    10000.000000    21  18
        0.000000    10000.000000    22  10
        0.000000    10000.000000    22  15
        0.000000    10000.000000    23  14
        0.000000    10000.000000    23  21
        0.000000    10000.000000    24  5
        0.000000    10000.000000    24  7
        0.000000    10000.000000    25  19
        0.000000    10000.000000    25  22
        0.000000    10000.000000    26  24
        0.000000    10000.000000    26  25
        0.000000    10000.000000    27  20
        0.000000    10000.000000    27  23
        0.000000    10000.000000    28  26
        0.000000    10000.000000    28  27
        """)
        ts_2 = tskit.load_text(nodes_2, edges_2, sequence_length=10000,
                               strict=False, base64_metadata=False)

        tree_1 = ts_1.first()
        tree_2 = ts_2.first()

        self.verify_result(tree_1, tree_2, 0, 19.95, places=2)
        self.verify_result(tree_1, tree_2, 1, 17.74, places=2)

    def test_nobinary_trees(self):
        nodes_1 = io.StringIO("""\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    -1  -1   e30=
        1   1   0.000000    -1  -1   e30=
        2   1   0.000000    -1  -1   e30=
        3   1   0.000000    -1  -1   e30=
        4   1   0.000000    -1  -1   e30=
        5   1   0.000000    -1  -1   e30=
        6   1   0.000000    -1  -1   e30=
        7   1   0.000000    -1  -1   e30=
        8   1   0.000000    -1  -1   e30=
        9   1   0.000000    -1  -1
        10  1   0.000000    -1  -1
        11  1   0.000000    -1  -1
        12  1   0.000000    -1  -1
        13  1   0.000000    -1  -1
        14  1   0.000000    -1  -1
        15  0   2.000000    -1  -1
        16  0   4.000000    -1  -1
        17  0   11.000000   -1  -1
        18  0   12.000000   -1  -1
        """)
        edges_1 = io.StringIO("""\
        left    right   parent  child
        0.000000    10000.000000    15  8
        0.000000    10000.000000    15  10
        0.000000    10000.000000    16  6
        0.000000    10000.000000    16  12
        0.000000    10000.000000    16  15
        0.000000    10000.000000    17  0
        0.000000    10000.000000    17  1
        0.000000    10000.000000    17  2
        0.000000    10000.000000    17  3
        0.000000    10000.000000    17  4
        0.000000    10000.000000    17  5
        0.000000    10000.000000    17  7
        0.000000    10000.000000    17  9
        0.000000    10000.000000    17  11
        0.000000    10000.000000    17  13
        0.000000    10000.000000    17  14
        0.000000    10000.000000    18  16
        0.000000    10000.000000    18  17
        """)
        ts_1 = tskit.load_text(nodes_1, edges_1, sequence_length=10000,
                               strict=False, base64_metadata=False)

        nodes_2 = io.StringIO("""\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    -1  -1   e30=
        1   1   0.000000    -1  -1   e30=
        2   1   0.000000    -1  -1   e30=
        3   1   0.000000    -1  -1   e30=
        4   1   0.000000    -1  -1   e30=
        5   1   0.000000    -1  -1   e30=
        6   1   0.000000    -1  -1   e30=
        7   1   0.000000    -1  -1   e30=
        8   1   0.000000    -1  -1   e30=
        9   1   0.000000    -1  -1   e30=
        10  1   0.000000    -1  -1  e30=
        11  1   0.000000    -1  -1  e30=
        12  1   0.000000    -1  -1  e30=
        13  1   0.000000    -1  -1  e30=
        14  1   0.000000    -1  -1  e30=
        15  0   2.000000    -1  -1
        16  0   2.000000    -1  -1
        17  0   3.000000    -1  -1
        18  0   3.000000    -1  -1
        19  0   4.000000    -1  -1
        20  0   4.000000    -1  -1
        21  0   11.000000   -1  -1
        22  0   12.000000   -1  -1
        """)
        edges_2 = io.StringIO("""\
        left    right   parent  child
        0.000000    10000.000000    15  12
        0.000000    10000.000000    15  14
        0.000000    10000.000000    16  0
        0.000000    10000.000000    16  7
        0.000000    10000.000000    17  6
        0.000000    10000.000000    17  15
        0.000000    10000.000000    18  4
        0.000000    10000.000000    18  8
        0.000000    10000.000000    18  13
        0.000000    10000.000000    19  11
        0.000000    10000.000000    19  18
        0.000000    10000.000000    20  1
        0.000000    10000.000000    20  5
        0.000000    10000.000000    20  9
        0.000000    10000.000000    20  10
        0.000000    10000.000000    21  2
        0.000000    10000.000000    21  3
        0.000000    10000.000000    21  16
        0.000000    10000.000000    21  17
        0.000000    10000.000000    21  20
        0.000000    10000.000000    22  19
        0.000000    10000.000000    22  21
        """)
        ts_2 = tskit.load_text(nodes_2, edges_2, sequence_length=10000,
                               strict=False, base64_metadata=False)
        tree_1 = ts_1.first()
        tree_2 = ts_2.first()
        self.verify_result(tree_1, tree_2, 0, 9.434, places=3)
        self.verify_result(tree_1, tree_2, 1, 44, places=1)

    def test_multiple_roots(self):
        tables = tskit.TableCollection(sequence_length=1.0)

        # Nodes
        sv = [True, True]
        tv = [0.0, 0.0]

        for is_sample, t in zip(sv, tv):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables.nodes.add_row(flags=flags, time=t)

        ts = tables.tree_sequence()

        with self.assertRaises(ValueError):
            kc_distance(ts.first(), ts.first(), 0)
        with self.assertRaises(ValueError):
            kc_distance_simple(ts.first(), ts.first(), 0)
        with self.assertRaises(_tskit.LibraryError):
            ts.first().kc_distance(ts.first(), 0)

    def do_kc_distance(self, t1, t2, lambda_=0):
        kc1 = kc_distance(t1, t2, lambda_)
        kc2 = kc_distance_simple(t1, t2, lambda_)
        kc3 = t1.kc_distance(t2, lambda_)
        self.assertAlmostEqual(kc1, kc2)
        self.assertAlmostEqual(kc1, kc3)

        kc1 = kc_distance(t2, t1, lambda_)
        kc2 = kc_distance_simple(t1, t1, lambda_)
        kc3 = t2.kc_distance(t1, lambda_)
        self.assertAlmostEqual(kc1, kc2)
        self.assertAlmostEqual(kc1, kc3)

    def test_non_initial_samples(self):
        ts1 = msprime.simulate(10, random_seed=1)
        nmap = np.arange(0, ts1.num_nodes)[::-1]
        ts2 = tsutil.permute_nodes(ts1, nmap)
        t1 = ts2.first()
        t2 = ts2.first()
        self.do_kc_distance(t1, t2)

    def test_internal_samples(self):
        ts1 = msprime.simulate(10, random_seed=1)
        ts2 = tsutil.jiggle_samples(ts1)
        t1 = ts2.first()
        t2 = ts2.first()

        with self.assertRaises(ValueError):
            kc_distance(t1, t2)
        with self.assertRaises(ValueError):
            kc_distance_simple(t1, t2)
        with self.assertRaises(_tskit.LibraryError):
            t1.kc_distance(t2)


class TestOverlappingSegments(unittest.TestCase):
    """
    Tests for the overlapping segments algorithm required for simplify.
    This test probably belongs somewhere else.
    """

    def test_random(self):
        segs = generate_segments(10, 20, 1)
        for left, right, X in tests.overlapping_segments(segs):
            self.assertGreater(right, left)
            self.assertGreater(len(X), 0)

    def test_empty(self):
        ret = list(tests.overlapping_segments([]))
        self.assertEqual(len(ret), 0)

    def test_single_interval(self):
        for j in range(1, 10):
            segs = [tests.Segment(0, 1, j) for _ in range(j)]
            ret = list(tests.overlapping_segments(segs))
            self.assertEqual(len(ret), 1)
            left, right, X = ret[0]
            self.assertEqual(left, 0)
            self.assertEqual(right, 1)
            self.assertEqual(sorted(segs), sorted(X))

    def test_stairs_down(self):
        segs = [
            tests.Segment(0, 1, 0),
            tests.Segment(0, 2, 1),
            tests.Segment(0, 3, 2)]
        ret = list(tests.overlapping_segments(segs))
        self.assertEqual(len(ret), 3)

        left, right, X = ret[0]
        self.assertEqual(left, 0)
        self.assertEqual(right, 1)
        self.assertEqual(sorted(X), sorted(segs))

        left, right, X = ret[1]
        self.assertEqual(left, 1)
        self.assertEqual(right, 2)
        self.assertEqual(sorted(X), sorted(segs[1:]))

        left, right, X = ret[2]
        self.assertEqual(left, 2)
        self.assertEqual(right, 3)
        self.assertEqual(sorted(X), sorted(segs[2:]))

    def test_stairs_up(self):
        segs = [
            tests.Segment(0, 3, 0),
            tests.Segment(1, 3, 1),
            tests.Segment(2, 3, 2)]
        ret = list(tests.overlapping_segments(segs))
        self.assertEqual(len(ret), 3)

        left, right, X = ret[0]
        self.assertEqual(left, 0)
        self.assertEqual(right, 1)
        self.assertEqual(X, segs[:1])

        left, right, X = ret[1]
        self.assertEqual(left, 1)
        self.assertEqual(right, 2)
        self.assertEqual(sorted(X), sorted(segs[:2]))

        left, right, X = ret[2]
        self.assertEqual(left, 2)
        self.assertEqual(right, 3)
        self.assertEqual(sorted(X), sorted(segs))

    def test_pyramid(self):
        segs = [
            tests.Segment(0, 5, 0),
            tests.Segment(1, 4, 1),
            tests.Segment(2, 3, 2)]
        ret = list(tests.overlapping_segments(segs))
        self.assertEqual(len(ret), 5)

        left, right, X = ret[0]
        self.assertEqual(left, 0)
        self.assertEqual(right, 1)
        self.assertEqual(X, segs[:1])

        left, right, X = ret[1]
        self.assertEqual(left, 1)
        self.assertEqual(right, 2)
        self.assertEqual(sorted(X), sorted(segs[:2]))

        left, right, X = ret[2]
        self.assertEqual(left, 2)
        self.assertEqual(right, 3)
        self.assertEqual(sorted(X), sorted(segs))

        left, right, X = ret[3]
        self.assertEqual(left, 3)
        self.assertEqual(right, 4)
        self.assertEqual(sorted(X), sorted(segs[:2]))

        left, right, X = ret[4]
        self.assertEqual(left, 4)
        self.assertEqual(right, 5)
        self.assertEqual(sorted(X), sorted(segs[:1]))

    def test_gap(self):
        segs = [
            tests.Segment(0, 2, 0),
            tests.Segment(3, 4, 1)]
        ret = list(tests.overlapping_segments(segs))
        self.assertEqual(len(ret), 2)

        left, right, X = ret[0]
        self.assertEqual(left, 0)
        self.assertEqual(right, 2)
        self.assertEqual(X, segs[:1])

        left, right, X = ret[1]
        self.assertEqual(left, 3)
        self.assertEqual(right, 4)
        self.assertEqual(X, segs[1:])


class TopologyTestCase(unittest.TestCase):
    """
    Superclass of test cases containing common utilities.
    """
    random_seed = 123456

    def assert_haplotypes_equal(self, ts1, ts2):
        h1 = list(ts1.haplotypes())
        h2 = list(ts2.haplotypes())
        self.assertEqual(h1, h2)

    def assert_variants_equal(self, ts1, ts2):
        v1 = list(ts1.variants(as_bytes=True))
        v2 = list(ts2.variants(as_bytes=True))
        self.assertEqual(v1, v2)

    def check_num_samples(self, ts, x):
        """
        Compare against x, a list of tuples of the form
        `(tree number, parent, number of samples)`.
        """
        k = 0
        tss = ts.trees(sample_counts=True)
        t = next(tss)
        for j, node, nl in x:
            while k < j:
                t = next(tss)
                k += 1
            self.assertEqual(nl, t.num_samples(node))

    def check_num_tracked_samples(self, ts, tracked_samples, x):
        k = 0
        tss = ts.trees(sample_counts=True, tracked_samples=tracked_samples)
        t = next(tss)
        for j, node, nl in x:
            while k < j:
                t = next(tss)
                k += 1
            self.assertEqual(nl, t.num_tracked_samples(node))

    def check_sample_iterator(self, ts, x):
        """
        Compare against x, a list of tuples of the form
        `(tree number, node, sample ID list)`.
        """
        k = 0
        tss = ts.trees(sample_lists=True)
        t = next(tss)
        for j, node, samples in x:
            while k < j:
                t = next(tss)
                k += 1
            for u, v in zip(samples, t.samples(node)):
                self.assertEqual(u, v)


class TestZeroRoots(unittest.TestCase):
    """
    Tests that for the case in which we have zero samples and therefore
    zero roots in our trees.
    """
    def remove_samples(self, ts):
        tables = ts.dump_tables()
        tables.nodes.flags = np.zeros_like(tables.nodes.flags)
        return tables.tree_sequence()

    def verify(self, ts, no_root_ts):
        self.assertEqual(ts.num_trees, no_root_ts.num_trees)
        for tree, no_root in zip(ts.trees(), no_root_ts.trees()):
            self.assertEqual(no_root.num_roots, 0)
            self.assertEqual(no_root.left_root, tskit.NULL)
            self.assertEqual(no_root.roots, [])
            self.assertEqual(tree.parent_dict, no_root.parent_dict)

    def test_single_tree(self):
        ts = msprime.simulate(10, random_seed=1)
        no_root_ts = self.remove_samples(ts)
        self.assertEqual(ts.num_trees, 1)
        self.verify(ts, no_root_ts)

    def test_multiple_trees(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=1)
        no_root_ts = self.remove_samples(ts)
        self.assertGreater(ts.num_trees, 1)
        self.verify(ts, no_root_ts)


class TestEmptyTreeSequences(TopologyTestCase):
    """
    Tests covering tree sequences that have zero edges.
    """
    def test_zero_nodes(self):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        self.assertEqual(ts.sequence_length, 1)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_nodes, 0)
        self.assertEqual(ts.num_edges, 0)
        t = next(ts.trees())
        self.assertEqual(t.index, 0)
        self.assertEqual(t.left_root, tskit.NULL)
        self.assertEqual(t.interval, (0, 1))
        self.assertEqual(t.roots, [])
        self.assertEqual(t.root, tskit.NULL)
        self.assertEqual(t.parent_dict, {})
        self.assertEqual(list(t.nodes()), [])
        self.assertEqual(list(ts.haplotypes()), [])
        self.assertEqual(list(ts.variants()), [])
        methods = [t.parent, t.left_child, t.right_child, t.left_sib, t.right_sib]
        for method in methods:
            for u in [-1, 0, 1, 100]:
                self.assertRaises(ValueError, method, u)
        tsp = ts.simplify()
        self.assertEqual(tsp.num_nodes, 0)
        self.assertEqual(tsp.num_edges, 0)

    def test_one_node_zero_samples(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=0)
        # Without a sequence length this should fail.
        ts = tables.tree_sequence()
        self.assertEqual(ts.sequence_length, 1)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_nodes, 1)
        self.assertEqual(ts.sample_size, 0)
        self.assertEqual(ts.num_edges, 0)
        self.assertEqual(ts.num_sites, 0)
        self.assertEqual(ts.num_mutations, 0)
        t = next(ts.trees())
        self.assertEqual(t.index, 0)
        self.assertEqual(t.left_root, tskit.NULL)
        self.assertEqual(t.interval, (0, 1))
        self.assertEqual(t.roots, [])
        self.assertEqual(t.root, tskit.NULL)
        self.assertEqual(t.parent_dict, {})
        self.assertEqual(list(t.nodes()), [])
        self.assertEqual(list(ts.haplotypes()), [])
        self.assertEqual(list(ts.variants()), [])
        methods = [t.parent, t.left_child, t.right_child, t.left_sib, t.right_sib]
        for method in methods:
            self.assertEqual(method(0), tskit.NULL)
            for u in [-1, 1, 100]:
                self.assertRaises(ValueError, method, u)

    def test_one_node_zero_samples_sites(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=0)
        tables.sites.add_row(position=0.5, ancestral_state='0')
        tables.mutations.add_row(site=0, derived_state='1', node=0)
        ts = tables.tree_sequence()
        self.assertEqual(ts.sequence_length, 1)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_nodes, 1)
        self.assertEqual(ts.sample_size, 0)
        self.assertEqual(ts.num_edges, 0)
        self.assertEqual(ts.num_sites, 1)
        self.assertEqual(ts.num_mutations, 1)
        t = next(ts.trees())
        self.assertEqual(t.index, 0)
        self.assertEqual(t.left_root, tskit.NULL)
        self.assertEqual(t.interval, (0, 1))
        self.assertEqual(t.roots, [])
        self.assertEqual(t.root, tskit.NULL)
        self.assertEqual(t.parent_dict, {})
        self.assertEqual(len(list(t.sites())), 1)
        self.assertEqual(list(t.nodes()), [])
        self.assertEqual(list(ts.haplotypes()), [])
        self.assertEqual(len(list(ts.variants())), 1)
        tsp = ts.simplify()
        self.assertEqual(tsp.num_nodes, 0)
        self.assertEqual(tsp.num_edges, 0)

    def test_one_node_one_sample(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        ts = tables.tree_sequence()
        self.assertEqual(ts.sequence_length, 1)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_nodes, 1)
        self.assertEqual(ts.sample_size, 1)
        self.assertEqual(ts.num_edges, 0)
        t = next(ts.trees())
        self.assertEqual(t.index, 0)
        self.assertEqual(t.left_root, 0)
        self.assertEqual(t.interval, (0, 1))
        self.assertEqual(t.roots, [0])
        self.assertEqual(t.root, 0)
        self.assertEqual(t.parent_dict, {})
        self.assertEqual(list(t.nodes()), [0])
        self.assertEqual(list(ts.haplotypes(impute_missing_data=True)), [""])
        self.assertEqual(list(ts.variants()), [])
        methods = [t.parent, t.left_child, t.right_child, t.left_sib, t.right_sib]
        for method in methods:
            self.assertEqual(method(0), tskit.NULL)
            for u in [-1, 1, 100]:
                self.assertRaises(ValueError, method, u)
        tsp = ts.simplify()
        self.assertEqual(tsp.num_nodes, 1)
        self.assertEqual(tsp.num_edges, 0)

    def test_one_node_one_sample_sites(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables.sites.add_row(position=0.5, ancestral_state='0')
        tables.mutations.add_row(site=0, derived_state='1', node=0)
        ts = tables.tree_sequence()
        self.assertEqual(ts.sequence_length, 1)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_nodes, 1)
        self.assertEqual(ts.sample_size, 1)
        self.assertEqual(ts.num_edges, 0)
        self.assertEqual(ts.num_sites, 1)
        self.assertEqual(ts.num_mutations, 1)
        t = next(ts.trees())
        self.assertEqual(t.index, 0)
        self.assertEqual(t.left_root, 0)
        self.assertEqual(t.interval, (0, 1))
        self.assertEqual(t.roots, [0])
        self.assertEqual(t.root, 0)
        self.assertEqual(t.parent_dict, {})
        self.assertEqual(list(t.nodes()), [0])
        self.assertEqual(list(ts.haplotypes(impute_missing_data=True)), ["1"])
        self.assertEqual(len(list(ts.variants())), 1)
        methods = [t.parent, t.left_child, t.right_child, t.left_sib, t.right_sib]
        for method in methods:
            self.assertEqual(method(0), tskit.NULL)
            for u in [-1, 1, 100]:
                self.assertRaises(ValueError, method, u)
        tsp = ts.simplify(filter_sites=False)
        self.assertEqual(tsp.num_nodes, 1)
        self.assertEqual(tsp.num_edges, 0)
        self.assertEqual(tsp.num_sites, 1)


class TestHoleyTreeSequences(TopologyTestCase):
    """
    Tests for tree sequences in which we have partial (or no) trees defined
    over some of the sequence.
    """
    def verify_trees(self, ts, expected):
        observed = []
        for t in ts.trees():
            observed.append((t.interval, t.parent_dict))
        self.assertEqual(expected, observed)
        # Test simple algorithm also.
        observed = []
        for interval, parent in tsutil.algorithm_T(ts):
            parent_dict = {j: parent[j] for j in range(ts.num_nodes) if parent[j] >= 0}
            observed.append((interval, parent_dict))
        self.assertEqual(expected, observed)

    def verify_zero_roots(self, ts):
        for tree in ts.trees():
            self.assertEqual(tree.num_roots, 0)
            self.assertEqual(tree.left_root, tskit.NULL)
            self.assertEqual(tree.roots, [])

    def test_simple_hole(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       2       0
        2       3       2       0
        0       1       2       1
        2       3       2       1
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        expected = [
            ((0, 1), {0: 2, 1: 2}),
            ((1, 2), {}),
            ((2, 3), {0: 2, 1: 2})]
        self.verify_trees(ts, expected)

    def test_simple_hole_zero_roots(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   0           0
        1   0           0
        2   0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       2       0
        2       3       2       0
        0       1       2       1
        2       3       2       1
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        expected = [
            ((0, 1), {0: 2, 1: 2}),
            ((1, 2), {}),
            ((2, 3), {0: 2, 1: 2})]
        self.verify_trees(ts, expected)
        self.verify_zero_roots(ts)

    def test_initial_gap(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        1       2       2       0,1
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        expected = [
            ((0, 1), {}),
            ((1, 2), {0: 2, 1: 2})]
        self.verify_trees(ts, expected)

    def test_initial_gap_zero_roots(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   0           0
        1   0           0
        2   0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        1       2       2       0,1
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        expected = [
            ((0, 1), {}),
            ((1, 2), {0: 2, 1: 2})]
        self.verify_trees(ts, expected)
        self.verify_zero_roots(ts)

    def test_final_gap(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       2       2       0,1
        """)
        ts = tskit.load_text(nodes, edges, sequence_length=3, strict=False)
        expected = [
            ((0, 2), {0: 2, 1: 2}),
            ((2, 3), {})]
        self.verify_trees(ts, expected)

    def test_final_gap_zero_roots(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   0           0
        1   0           0
        2   0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       2       2       0,1
        """)
        ts = tskit.load_text(nodes, edges, sequence_length=3, strict=False)
        expected = [
            ((0, 2), {0: 2, 1: 2}),
            ((2, 3), {})]
        self.verify_trees(ts, expected)
        self.verify_zero_roots(ts)

    def test_initial_and_final_gap(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        1       2       2       0,1
        """)
        ts = tskit.load_text(nodes, edges, sequence_length=3, strict=False)
        expected = [
            ((0, 1), {}),
            ((1, 2), {0: 2, 1: 2}),
            ((2, 3), {})]
        self.verify_trees(ts, expected)

    def test_initial_and_final_gap_zero_roots(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   0           0
        1   0           0
        2   0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        1       2       2       0,1
        """)
        ts = tskit.load_text(nodes, edges, sequence_length=3, strict=False)
        expected = [
            ((0, 1), {}),
            ((1, 2), {0: 2, 1: 2}),
            ((2, 3), {})]
        self.verify_trees(ts, expected)
        self.verify_zero_roots(ts)


class TestTsinferExamples(TopologyTestCase):
    """
    Test cases on troublesome topology examples that arose from tsinfer.
    """
    def test_no_last_tree(self):
        # The last tree was not being generated here because of a bug in
        # the low-level tree generation code.
        nodes = io.StringIO("""\
        id      is_sample   population      time
        0       1       -1              3.00000000000000
        1       1       -1              2.00000000000000
        2       1       -1              2.00000000000000
        3       1       -1              2.00000000000000
        4       1       -1              2.00000000000000
        5       1       -1              1.00000000000000
        6       1       -1              1.00000000000000
        7       1       -1              1.00000000000000
        8       1       -1              1.00000000000000
        9       1       -1              1.00000000000000
        10      1       -1              1.00000000000000
        """)
        edges = io.StringIO("""\
        id      left            right           parent  child
        0       62291.41659631  79679.17408763  1       5
        1       62291.41659631  62374.60889677  1       6
        2       122179.36037089 138345.43104411 1       7
        3       67608.32330402  79679.17408763  1       8
        4       122179.36037089 138345.43104411 1       8
        5       62291.41659631  79679.17408763  1       9
        6       126684.47550333 138345.43104411 1       10
        7       23972.05905068  62291.41659631  2       5
        8       79679.17408763  82278.53390076  2       5
        9       23972.05905068  62291.41659631  2       6
        10      79679.17408763  110914.43816806 2       7
        11      145458.28890561 189765.31932273 2       7
        12      79679.17408763  110914.43816806 2       8
        13      145458.28890561 200000.00000000 2       8
        14      23972.05905068  62291.41659631  2       9
        15      79679.17408763  110914.43816806 2       9
        16      145458.28890561 145581.18329797 2       10
        17      4331.62138785   23972.05905068  3       6
        18      4331.62138785   23972.05905068  3       9
        19      110914.43816806 122179.36037089 4       7
        20      138345.43104411 145458.28890561 4       7
        21      110914.43816806 122179.36037089 4       8
        22      138345.43104411 145458.28890561 4       8
        23      110914.43816806 112039.30503475 4       9
        24      138345.43104411 145458.28890561 4       10
        25      0.00000000      200000.00000000 0       1
        26      0.00000000      200000.00000000 0       2
        27      0.00000000      200000.00000000 0       3
        28      0.00000000      200000.00000000 0       4
        """)
        ts = tskit.load_text(nodes, edges, sequence_length=200000, strict=False)
        pts = tests.PythonTreeSequence(ts.get_ll_tree_sequence())
        num_trees = 0
        for t in pts.trees():
            num_trees += 1
        self.assertEqual(num_trees, ts.num_trees)
        n = 0
        for pt, t in zip(pts.trees(), ts.trees()):
            self.assertEqual((pt.left, pt.right), t.interval)
            for j in range(ts.num_nodes):
                self.assertEqual(pt.parent[j], t.parent(j))
                self.assertEqual(pt.left_child[j], t.left_child(j))
                self.assertEqual(pt.right_child[j], t.right_child(j))
                self.assertEqual(pt.left_sib[j], t.left_sib(j))
                self.assertEqual(pt.right_sib[j], t.right_sib(j))
            n += 1
        self.assertEqual(n, num_trees)
        intervals = [t.interval for t in ts.trees()]
        self.assertEqual(intervals[0][0], 0)
        self.assertEqual(intervals[-1][-1], ts.sequence_length)


class TestRecordSquashing(TopologyTestCase):
    """
    Tests that we correctly squash adjacent equal records together.
    """
    def test_single_record(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       1       0
        1       2       1       0
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        tss, node_map = ts.simplify(map_nodes=True)
        self.assertEqual(list(node_map), [0, 1])
        self.assertEqual(tss.dump_tables().nodes, ts.dump_tables().nodes)
        simplified_edges = list(tss.edges())
        self.assertEqual(len(simplified_edges), 1)
        e = simplified_edges[0]
        self.assertEqual(e.left, 0)
        self.assertEqual(e.right, 2)

    def test_single_tree(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts_redundant = tsutil.insert_redundant_breakpoints(ts)
        tss = ts_redundant.simplify()
        self.assertEqual(tss.dump_tables().nodes, ts.dump_tables().nodes)
        self.assertEqual(tss.dump_tables().edges, ts.dump_tables().edges)

    def test_many_trees(self):
        ts = msprime.simulate(
            20, recombination_rate=5, random_seed=self.random_seed)
        self.assertGreater(ts.num_trees, 2)
        ts_redundant = tsutil.insert_redundant_breakpoints(ts)
        tss = ts_redundant.simplify()
        self.assertEqual(tss.dump_tables().nodes, ts.dump_tables().nodes)
        self.assertEqual(tss.dump_tables().edges, ts.dump_tables().edges)


class TestRedundantBreakpoints(TopologyTestCase):
    """
    Tests for dealing with redundant breakpoints within the tree sequence.
    These are records that may be squashed together into a single record.
    """
    def test_single_tree(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts_redundant = tsutil.insert_redundant_breakpoints(ts)
        self.assertEqual(ts.sample_size, ts_redundant.sample_size)
        self.assertEqual(ts.sequence_length, ts_redundant.sequence_length)
        self.assertEqual(ts_redundant.num_trees, 2)
        trees = [t.parent_dict for t in ts_redundant.trees()]
        self.assertEqual(len(trees), 2)
        self.assertEqual(trees[0], trees[1])
        self.assertEqual([t.parent_dict for t in ts.trees()][0], trees[0])

    def test_many_trees(self):
        ts = msprime.simulate(
            20, recombination_rate=5, random_seed=self.random_seed)
        self.assertGreater(ts.num_trees, 2)
        ts_redundant = tsutil.insert_redundant_breakpoints(ts)
        self.assertEqual(ts.sample_size, ts_redundant.sample_size)
        self.assertEqual(ts.sequence_length, ts_redundant.sequence_length)
        self.assertGreater(ts_redundant.num_trees, ts.num_trees)
        self.assertGreater(ts_redundant.num_edges, ts.num_edges)
        redundant_trees = ts_redundant.trees()
        redundant_t = next(redundant_trees)
        comparisons = 0
        for t in ts.trees():
            while redundant_t is not None and redundant_t.interval[1] <= t.interval[1]:
                self.assertEqual(t.parent_dict, redundant_t.parent_dict)
                comparisons += 1
                redundant_t = next(redundant_trees, None)
        self.assertEqual(comparisons, ts_redundant.num_trees)


class TestUnaryNodes(TopologyTestCase):
    """
    Tests for situations in which we have unary nodes in the tree sequence.
    """
    def test_simple_case(self):
        # Simple case where we have n = 2 and some unary nodes.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           1
        4       0           2
        5       0           3
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       2       0
        0       1       3       1
        0       1       4       2,3
        0       1       5       4
        """)
        sites = "position    ancestral_state\n"
        mutations = "site    node    derived_state\n"
        for j in range(5):
            position = j * 1 / 5
            sites += "{} 0\n".format(position)
            mutations += "{} {} 1\n".format(j, j)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=io.StringIO(sites),
            mutations=io.StringIO(mutations), strict=False)

        self.assertEqual(ts.sample_size, 2)
        self.assertEqual(ts.num_nodes, 6)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_sites, 5)
        self.assertEqual(ts.num_mutations, 5)
        self.assertEqual(len(list(ts.edge_diffs())), ts.num_trees)
        t = next(ts.trees())
        self.assertEqual(
            t.parent_dict, {0: 2, 1: 3, 2: 4, 3: 4, 4: 5})
        self.assertEqual(t.mrca(0, 1), 4)
        self.assertEqual(t.mrca(0, 2), 2)
        self.assertEqual(t.mrca(0, 4), 4)
        self.assertEqual(t.mrca(0, 5), 5)
        self.assertEqual(t.mrca(0, 3), 4)
        H = list(ts.haplotypes())
        self.assertEqual(H[0], "10101")
        self.assertEqual(H[1], "01011")

    def test_ladder_tree(self):
        # We have a single tree with a long ladder of unary nodes along a path
        num_unary_nodes = 30
        n = 2
        nodes = """\
            is_sample   time
            1           0
            1           0
        """
        edges = """\
            left right parent child
            0    1     2      0
        """
        for j in range(num_unary_nodes + 2):
            nodes += "0 {}\n".format(j + 2)
        for j in range(num_unary_nodes):
            edges += "0 1 {} {}\n".format(n + j + 1, n + j)
        root = num_unary_nodes + 3
        root_time = num_unary_nodes + 3
        edges += "0    1     {}      1,{}\n".format(root, num_unary_nodes + 2)
        ts = tskit.load_text(io.StringIO(nodes), io.StringIO(edges), strict=False)
        t = ts.first()
        self.assertEqual(t.mrca(0, 1), root)
        self.assertEqual(t.tmrca(0, 1), root_time)
        ts_simplified, node_map = ts.simplify(map_nodes=True)
        test_map = [tskit.NULL for _ in range(ts.num_nodes)]
        test_map[0] = 0
        test_map[1] = 1
        test_map[root] = 2
        self.assertEqual(list(node_map), test_map)
        self.assertEqual(ts_simplified.num_edges, 2)
        t = ts_simplified.first()
        self.assertEqual(t.mrca(0, 1), 2)
        self.assertEqual(t.tmrca(0, 1), root_time)
        ts_simplified = ts.simplify(keep_unary=True, record_provenance=False)
        self.assertEqual(ts_simplified.tables, ts.tables)

    def verify_unary_tree_sequence(self, ts):
        """
        Take the specified tree sequence and produce an equivalent in which
        unary records have been interspersed.
        """
        self.assertGreater(ts.num_trees, 2)
        self.assertGreater(ts.num_mutations, 2)
        tables = ts.dump_tables()
        next_node = ts.num_nodes
        node_times = {j: node.time for j, node in enumerate(ts.nodes())}
        edges = []
        for e in ts.edges():
            node = ts.node(e.parent)
            t = node.time - 1e-14  # Arbitrary small value.
            next_node = len(tables.nodes)
            tables.nodes.add_row(time=t, population=node.population)
            edges.append(tskit.Edge(
                left=e.left, right=e.right, parent=next_node, child=e.child))
            node_times[next_node] = t
            edges.append(tskit.Edge(
                left=e.left, right=e.right, parent=e.parent, child=next_node))
        edges.sort(key=lambda e: node_times[e.parent])
        tables.edges.reset()
        for e in edges:
            tables.edges.add_row(
                left=e.left, right=e.right, child=e.child, parent=e.parent)
        ts_new = tables.tree_sequence()
        self.assertGreater(ts_new.num_edges, ts.num_edges)
        self.assert_haplotypes_equal(ts, ts_new)
        self.assert_variants_equal(ts, ts_new)
        ts_simplified = ts_new.simplify()
        self.assertEqual(list(ts_simplified.records()), list(ts.records()))
        self.assert_haplotypes_equal(ts, ts_simplified)
        self.assert_variants_equal(ts, ts_simplified)
        self.assertEqual(len(list(ts.edge_diffs())), ts.num_trees)

        for keep_unary in [True, False]:
            s = tests.Simplifier(ts, ts.samples(), keep_unary=keep_unary)
            py_ts, py_node_map = s.simplify()
            lib_ts, lib_node_map = ts.simplify(keep_unary=keep_unary, map_nodes=True)
            py_tables = py_ts.dump_tables()
            py_tables.provenances.clear()
            lib_tables = lib_ts.dump_tables()
            lib_tables.provenances.clear()
            self.assertEqual(lib_tables, py_tables)
            self.assertTrue(np.all(lib_node_map == py_node_map))

    def test_binary_tree_sequence_unary_nodes(self):
        ts = msprime.simulate(
            20, recombination_rate=5, mutation_rate=5, random_seed=self.random_seed)
        self.verify_unary_tree_sequence(ts)

    def test_nonbinary_tree_sequence_unary_nodes(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=1.0, population=0, proportion=0.95)]
        ts = msprime.simulate(
            20, recombination_rate=10, mutation_rate=5,
            demographic_events=demographic_events, random_seed=self.random_seed)
        found = False
        for r in ts.edgesets():
            if len(r.children) > 2:
                found = True
        self.assertTrue(found)
        self.verify_unary_tree_sequence(ts)


class TestGeneralSamples(TopologyTestCase):
    """
    Test cases in which we have samples at arbitrary nodes (i.e., not at
    {0,...,n - 1}).
    """
    def test_simple_case(self):
        # Simple case where we have n = 3 and samples starting at n.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       0           2
        1       0           1
        2       1           0
        3       1           0
        4       1           0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       1       2,3
        0       1       0       1,4
        """)
        sites = io.StringIO("""\
        position    ancestral_state
        0.1     0
        0.2     0
        0.3     0
        0.4     0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state
        0       2       1
        1       3       1
        2       4       1
        3       1       1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)

        self.assertEqual(ts.sample_size, 3)
        self.assertEqual(list(ts.samples()), [2, 3, 4])
        self.assertEqual(ts.num_nodes, 5)
        self.assertEqual(ts.num_nodes, 5)
        self.assertEqual(ts.num_sites, 4)
        self.assertEqual(ts.num_mutations, 4)
        self.assertEqual(len(list(ts.edge_diffs())), ts.num_trees)
        t = next(ts.trees())
        self.assertEqual(t.root, 0)
        self.assertEqual(t.parent_dict, {1: 0, 2: 1, 3: 1, 4: 0})
        H = list(ts.haplotypes())
        self.assertEqual(H[0], "1001")
        self.assertEqual(H[1], "0101")
        self.assertEqual(H[2], "0010")

        tss, node_map = ts.simplify(map_nodes=True)
        self.assertEqual(list(node_map), [4, 3, 0, 1, 2])
        # We should have the same tree sequence just with canonicalised nodes.
        self.assertEqual(tss.sample_size, 3)
        self.assertEqual(list(tss.samples()), [0, 1, 2])
        self.assertEqual(tss.num_nodes, 5)
        self.assertEqual(tss.num_trees, 1)
        self.assertEqual(tss.num_sites, 4)
        self.assertEqual(tss.num_mutations, 4)
        self.assertEqual(len(list(ts.edge_diffs())), ts.num_trees)
        t = next(tss.trees())
        self.assertEqual(t.root, 4)
        self.assertEqual(t.parent_dict, {0: 3, 1: 3, 2: 4, 3: 4})
        H = list(tss.haplotypes())
        self.assertEqual(H[0], "1001")
        self.assertEqual(H[1], "0101")
        self.assertEqual(H[2], "0010")

    def verify_permuted_nodes(self, ts):
        """
        Take the specified tree sequence and permute the nodes, verifying that we
        get back a tree sequence with the correct properties.
        """
        # Mapping from the original nodes into nodes in the new tree sequence.
        node_map = list(range(ts.num_nodes))
        random.shuffle(node_map)
        # Change the permutation so that the relative order of samples is maintained.
        # Then, we should get back exactly the same tree sequence after simplify
        # and haplotypes and variants are also equal.
        samples = sorted(node_map[:ts.sample_size])
        node_map = samples + node_map[ts.sample_size:]
        permuted = tsutil.permute_nodes(ts, node_map)
        self.assertEqual(ts.sequence_length, permuted.sequence_length)
        self.assertEqual(list(permuted.samples()), samples)
        self.assertEqual(list(permuted.haplotypes()), list(ts.haplotypes()))
        self.assertEqual(
            [v.genotypes for v in permuted.variants(as_bytes=True)],
            [v.genotypes for v in ts.variants(as_bytes=True)])
        self.assertEqual(ts.num_trees, permuted.num_trees)
        j = 0
        for t1, t2 in zip(ts.trees(), permuted.trees()):
            t1_dict = {node_map[k]: node_map[v] for k, v in t1.parent_dict.items()}
            self.assertEqual(node_map[t1.root], t2.root)
            self.assertEqual(t1_dict, t2.parent_dict)
            for u1 in t1.nodes():
                u2 = node_map[u1]
                self.assertEqual(
                    sorted([node_map[v] for v in t1.samples(u1)]),
                    sorted(list(t2.samples(u2))))
            j += 1
        self.assertEqual(j, ts.num_trees)

        # The simplified version of the permuted tree sequence should be in canonical
        # form, and identical to the original.
        simplified, s_node_map = permuted.simplify(map_nodes=True)

        original_tables = ts.dump_tables()
        simplified_tables = simplified.dump_tables()
        original_tables.provenances.clear()
        simplified_tables.provenances.clear()

        self.assertEqual(
            original_tables.sequence_length, simplified_tables.sequence_length)
        self.assertEqual(original_tables.nodes, simplified_tables.nodes)
        self.assertEqual(original_tables.edges, simplified_tables.edges)
        self.assertEqual(original_tables.sites, simplified_tables.sites)
        self.assertEqual(original_tables.mutations, simplified_tables.mutations)
        self.assertEqual(original_tables.individuals, simplified_tables.individuals)
        self.assertEqual(original_tables.populations, simplified_tables.populations)

        self.assertEqual(original_tables, simplified_tables)
        self.assertEqual(ts.sequence_length, simplified.sequence_length)
        for tree in simplified.trees():
            pass

        for u, v in enumerate(node_map):
            self.assertEqual(s_node_map[v], u)
        self.assertTrue(np.array_equal(simplified.samples(), ts.samples()))
        self.assertEqual(list(simplified.nodes()), list(ts.nodes()))
        self.assertEqual(list(simplified.edges()), list(ts.edges()))
        self.assertEqual(list(simplified.sites()), list(ts.sites()))
        self.assertEqual(list(simplified.haplotypes()), list(ts.haplotypes()))
        self.assertEqual(
            list(simplified.variants(as_bytes=True)), list(ts.variants(as_bytes=True)))

    def test_single_tree_permuted_nodes(self):
        ts = msprime.simulate(10,  mutation_rate=5, random_seed=self.random_seed)
        self.verify_permuted_nodes(ts)

    def test_binary_tree_sequence_permuted_nodes(self):
        ts = msprime.simulate(
            20, recombination_rate=5, mutation_rate=5, random_seed=self.random_seed)
        self.verify_permuted_nodes(ts)

    def test_nonbinary_tree_sequence_permuted_nodes(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=1.0, population=0, proportion=0.95)]
        ts = msprime.simulate(
            20, recombination_rate=10, mutation_rate=5,
            demographic_events=demographic_events, random_seed=self.random_seed)
        found = False
        for e in ts.edgesets():
            if len(e.children) > 2:
                found = True
        self.assertTrue(found)
        self.verify_permuted_nodes(ts)


class TestSimplifyExamples(TopologyTestCase):
    """
    Tests for simplify where we write out the input and expected output
    or we detect expected errors.
    """
    def verify_simplify(
            self, samples, filter_sites=True,
            nodes_before=None, edges_before=None, sites_before=None,
            mutations_before=None, nodes_after=None, edges_after=None,
            sites_after=None, mutations_after=None, debug=False):
        """
        Verifies that if we run simplify on the specified input we get the
        required output.
        """
        ts = tskit.load_text(
            nodes=io.StringIO(nodes_before),
            edges=io.StringIO(edges_before),
            sites=io.StringIO(sites_before) if sites_before is not None else None,
            mutations=(
                io.StringIO(mutations_before)
                if mutations_before is not None else None),
            strict=False)
        before = ts.dump_tables()

        ts = tskit.load_text(
            nodes=io.StringIO(nodes_after),
            edges=io.StringIO(edges_after),
            sites=io.StringIO(sites_after) if sites_after is not None else None,
            mutations=(
                io.StringIO(mutations_after)
                if mutations_after is not None else None),
            strict=False,
            sequence_length=before.sequence_length)
        after = ts.dump_tables()

        ts = before.tree_sequence()
        # Make sure it's a valid topology. We want to be sure we evaluate the
        # whole iterator
        for t in ts.trees():
            self.assertTrue(t is not None)
        before.simplify(samples=samples, filter_sites=filter_sites)
        if debug:
            print("before")
            print(before)
            print("after")
            print(after)
        self.assertEqual(before, after)

    def test_unsorted_edges(self):
        # We have two nodes at the same time and interleave edges for
        # these nodes together. This is an error because all edges for
        # a given parent must be contigous.
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           1
        """
        edges_before = """\
        left    right   parent  child
        0       1       2       0,1
        0       1       3       0,1
        1       2       2       0,1
        1       2       3       0,1
        """
        nodes = tskit.parse_nodes(io.StringIO(nodes_before), strict=False)
        edges = tskit.parse_edges(io.StringIO(edges_before), strict=False)
        # Cannot use load_text here because it calls sort()
        tables = tskit.TableCollection(sequence_length=2)
        tables.nodes.set_columns(**nodes.asdict())
        tables.edges.set_columns(**edges.asdict())
        self.assertRaises(_tskit.LibraryError, tables.simplify, samples=[0, 1])

    def test_single_binary_tree(self):
        #
        # 2        4
        #         / \
        # 1      3   \
        #       / \   \
        # 0   (0)(1)  (2)
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        # We sample 0 and 2, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           2
        """
        edges_after = """\
        left    right   parent  child
        0       1       2       0,1
        """
        self.verify_simplify(
            samples=[0, 2],
            nodes_before=nodes_before, edges_before=edges_before,
            nodes_after=nodes_after, edges_after=edges_after)

    def test_single_binary_tree_no_sample_nodes(self):
        #
        # 2        4
        #         / \
        # 1      3   \
        #       / \   \
        # 0   (0)(1)  (2)
        nodes_before = """\
        id      is_sample   time
        0       0           0
        1       0           0
        2       0           0
        3       0           1
        4       0           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        # We sample 0 and 2, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           2
        """
        edges_after = """\
        left    right   parent  child
        0       1       2       0,1
        """
        self.verify_simplify(
            samples=[0, 2],
            nodes_before=nodes_before, edges_before=edges_before,
            nodes_after=nodes_after, edges_after=edges_after)

    def test_single_binary_tree_internal_sample(self):
        #
        # 2        4
        #         / \
        # 1     (3)  \
        #       / \   \
        # 0   (0)  1  (2)
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0
        3       1           1
        4       0           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        # We sample 0 and 3, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           1
        """
        edges_after = """\
        left    right   parent  child
        0       1       1       0
        """
        self.verify_simplify(
            samples=[0, 3],
            nodes_before=nodes_before, edges_before=edges_before,
            nodes_after=nodes_after, edges_after=edges_after)

    def test_single_binary_tree_internal_sample_meet_at_root(self):
        # 3          5
        #           / \
        # 2        4  (6)
        #         / \
        # 1     (3)  \
        #       / \   \
        # 0   (0)  1   2
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0
        3       1           1
        4       0           2
        5       0           3
        6       1           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        0       1       5       4,6
        """
        # We sample 0 and 3 and 6, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           1
        2       1           2
        3       0           3
        """
        edges_after = """\
        left    right   parent  child
        0       1       1       0
        0       1       3       1,2
        """
        self.verify_simplify(
            samples=[0, 3, 6],
            nodes_before=nodes_before, edges_before=edges_before,
            nodes_after=nodes_after, edges_after=edges_after)

    def test_single_binary_tree_simple_mutations(self):
        # 3          5
        #           / \
        # 2        4   \
        #         / \   s0
        # 1      3   s1  \
        #       / \   \   \
        # 0   (0) (1)  2  (6)
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0
        3       0           1
        4       0           2
        5       0           3
        6       1           0
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        0       1       5       4,6
        """
        sites_before = """\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        """
        mutations_before = """\
        site    node    derived_state
        0       6       1
        1       2       1
        """

        # We sample 0 and 2 and 6, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        3       0           3
        """
        edges_after = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        sites_after = """\
        id  position    ancestral_state
        0   0.1         0
        """
        mutations_after = """\
        site    node    derived_state
        0       2       1
        """
        self.verify_simplify(
            samples=[0, 1, 6],
            nodes_before=nodes_before, edges_before=edges_before,
            sites_before=sites_before, mutations_before=mutations_before,
            nodes_after=nodes_after, edges_after=edges_after,
            sites_after=sites_after, mutations_after=mutations_after)
        # If we don't filter the fixed sites, we should get the same
        # mutations and the original sites table back.
        self.verify_simplify(
            samples=[0, 1, 6], filter_sites=False,
            nodes_before=nodes_before, edges_before=edges_before,
            sites_before=sites_before, mutations_before=mutations_before,
            nodes_after=nodes_after, edges_after=edges_after,
            sites_after=sites_before, mutations_after=mutations_after)

    def test_overlapping_edges(self):
        nodes = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        """
        edges_before = """\
        left    right   parent  child
        0       2       2       0
        1       3       2       1
        """
        # We resolve the overlapping edges here. Since the flanking regions
        # have no interesting edges, these are left out of the output.
        edges_after = """\
        left    right   parent  child
        1       2       2       0,1
        """
        self.verify_simplify(
            samples=[0, 1],
            nodes_before=nodes, edges_before=edges_before,
            nodes_after=nodes, edges_after=edges_after)

    def test_overlapping_edges_internal_samples(self):
        nodes = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           1
        """
        edges = """\
        left    right   parent  child
        0       2       2       0
        1       3       2       1
        """
        self.verify_simplify(
            samples=[0, 1, 2],
            nodes_before=nodes, edges_before=edges, nodes_after=nodes, edges_after=edges)

    def test_unary_edges_no_overlap(self):
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        """
        edges_before = """\
        left    right   parent  child
        0       2       2       0
        2       3       2       1
        """
        # Because there is no overlap between the samples, we just get an
        # empty set of output edges.
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        """
        edges_after = """\
        left    right   parent  child
        """
        self.verify_simplify(
            samples=[0, 1],
            nodes_before=nodes_before, edges_before=edges_before,
            nodes_after=nodes_after, edges_after=edges_after)

    def test_unary_edges_no_overlap_internal_sample(self):
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           1
        """
        edges_before = """\
        left    right   parent  child
        0       1       2       0
        1       2       2       1
        """
        self.verify_simplify(
            samples=[0, 1, 2],
            nodes_before=nodes_before, edges_before=edges_before,
            nodes_after=nodes_before, edges_after=edges_before)


class TestNonSampleExternalNodes(TopologyTestCase):
    """
    Tests for situations in which we have tips that are not samples.
    """
    def test_simple_case(self):
        # Simplest case where we have n = 2 and external non-sample nodes.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           0
        4       0           0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       2       0,1,3,4
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        2   0.3         0
        3   0.4         0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state
        0       0       1
        1       1       1
        2       3       1
        3       4       1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        self.assertEqual(ts.sample_size, 2)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_nodes, 5)
        self.assertEqual(ts.num_sites, 4)
        self.assertEqual(ts.num_mutations, 4)
        t = next(ts.trees())
        self.assertEqual(t.parent_dict, {0: 2, 1: 2, 3: 2, 4: 2})
        self.assertEqual(t.root, 2)
        ts_simplified, node_map = ts.simplify(map_nodes=True)
        self.assertEqual(list(node_map), [0, 1, 2, -1, -1])
        self.assertEqual(ts_simplified.num_nodes, 3)
        self.assertEqual(ts_simplified.num_trees, 1)
        t = next(ts_simplified.trees())
        self.assertEqual(t.parent_dict, {0: 2, 1: 2})
        self.assertEqual(t.root, 2)
        # We should have removed the two non-sample mutations.
        self.assertEqual([s.position for s in t.sites()], [0.1, 0.2])

    def test_unary_non_sample_external_nodes(self):
        # Take an ordinary tree sequence and put a bunch of external non
        # sample nodes on it.
        ts = msprime.simulate(
            15, recombination_rate=5, random_seed=self.random_seed, mutation_rate=5)
        self.assertGreater(ts.num_trees, 2)
        self.assertGreater(ts.num_mutations, 2)
        tables = ts.dump_tables()
        next_node = ts.num_nodes
        tables.edges.reset()
        for e in ts.edges():
            tables.edges.add_row(e.left, e.right, e.parent, e.child)
            tables.edges.add_row(e.left, e.right, e.parent, next_node)
            tables.nodes.add_row(time=0)
            next_node += 1
        tables.sort()
        ts_new = tables.tree_sequence()
        self.assertEqual(ts_new.num_nodes, next_node)
        self.assertEqual(ts_new.sample_size, ts.sample_size)
        self.assert_haplotypes_equal(ts, ts_new)
        self.assert_variants_equal(ts, ts_new)
        ts_simplified = ts_new.simplify()
        self.assertEqual(ts_simplified.num_nodes, ts.num_nodes)
        self.assertEqual(ts_simplified.sample_size, ts.sample_size)
        self.assertEqual(list(ts_simplified.records()), list(ts.records()))
        self.assert_haplotypes_equal(ts, ts_simplified)
        self.assert_variants_equal(ts, ts_simplified)


class TestMultipleRoots(TopologyTestCase):
    """
    Tests for situations where we have multiple roots for the samples.
    """

    def test_simplest_degenerate_case(self):
        # Simplest case where we have n = 2 and no edges.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state
        0       0         1
        1       1         1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            sequence_length=1, strict=False)
        self.assertEqual(ts.num_nodes, 2)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_sites, 2)
        self.assertEqual(ts.num_mutations, 2)
        t = next(ts.trees())
        self.assertEqual(t.parent_dict, {})
        self.assertEqual(sorted(t.roots), [0, 1])
        self.assertEqual(list(ts.haplotypes(impute_missing_data=True)), ["10", "01"])
        self.assertEqual(
            [v.genotypes for v in ts.variants(as_bytes=True, impute_missing_data=True)],
            [b"10", b"01"])
        simplified = ts.simplify()
        t1 = ts.dump_tables()
        t2 = simplified.dump_tables()
        self.assertEqual(t1.nodes, t2.nodes)
        self.assertEqual(t1.edges, t2.edges)

    def test_simplest_non_degenerate_case(self):
        # Simplest case where we have n = 4 and two trees.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       0           1
        5       0           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       4       0,1
        0       1       5       2,3
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        2   0.3         0
        3   0.4         0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state
        0       0       1
        1       1       1
        2       2       1
        3       3       1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        self.assertEqual(ts.num_nodes, 6)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_sites, 4)
        self.assertEqual(ts.num_mutations, 4)
        t = next(ts.trees())
        self.assertEqual(t.parent_dict, {0: 4, 1: 4, 2: 5, 3: 5})
        self.assertEqual(list(ts.haplotypes()), ["1000", "0100", "0010", "0001"])
        self.assertEqual(
            [v.genotypes for v in ts.variants(as_bytes=True)],
            [b"1000", b"0100", b"0010", b"0001"])
        self.assertEqual(t.mrca(0, 1), 4)
        self.assertEqual(t.mrca(0, 4), 4)
        self.assertEqual(t.mrca(2, 3), 5)
        self.assertEqual(t.mrca(0, 2), tskit.NULL)
        self.assertEqual(t.mrca(0, 3), tskit.NULL)
        self.assertEqual(t.mrca(2, 4), tskit.NULL)
        ts_simplified, node_map = ts.simplify(map_nodes=True)
        for j in range(4):
            self.assertEqual(node_map[j], j)
        self.assertEqual(ts_simplified.num_nodes, 6)
        self.assertEqual(ts_simplified.num_trees, 1)
        self.assertEqual(ts_simplified.num_sites, 4)
        self.assertEqual(ts_simplified.num_mutations, 4)
        t = next(ts_simplified.trees())
        self.assertEqual(t.parent_dict, {0: 4, 1: 4, 2: 5, 3: 5})

    def test_two_reducable_trees(self):
        # We have n = 4 and two trees, with some unary nodes and non-sample leaves
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       0           1
        5       0           1
        6       0           2
        7       0           3
        8       0           0   # Non sample leaf
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1      4         0
        0       1      5         1
        0       1      6         4,5
        0       1      7         2,3,8
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        2   0.3         0
        3   0.4         0
        4   0.5         0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state
        0       0       1
        1       1       1
        2       2       1
        3       3       1
        4       8       1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        self.assertEqual(ts.num_nodes, 9)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_sites, 5)
        self.assertEqual(ts.num_mutations, 5)
        t = next(ts.trees())
        self.assertEqual(t.parent_dict, {0: 4, 1: 5, 2: 7, 3: 7, 4: 6, 5: 6, 8: 7})
        self.assertEqual(list(ts.haplotypes()), ["10000", "01000", "00100", "00010"])
        self.assertEqual(
            [v.genotypes for v in ts.variants(as_bytes=True)],
            [b"1000", b"0100", b"0010", b"0001", b"0000"])
        self.assertEqual(t.mrca(0, 1), 6)
        self.assertEqual(t.mrca(2, 3), 7)
        self.assertEqual(t.mrca(2, 8), 7)
        self.assertEqual(t.mrca(0, 2), tskit.NULL)
        self.assertEqual(t.mrca(0, 3), tskit.NULL)
        self.assertEqual(t.mrca(0, 8), tskit.NULL)
        ts_simplified, node_map = ts.simplify(map_nodes=True)
        for j in range(4):
            self.assertEqual(node_map[j], j)
        self.assertEqual(ts_simplified.num_nodes, 6)
        self.assertEqual(ts_simplified.num_trees, 1)
        t = next(ts_simplified.trees())
        # print(ts_simplified.tables)
        self.assertEqual(
            list(ts_simplified.haplotypes()), ["1000", "0100", "0010", "0001"])
        self.assertEqual(
            [v.genotypes for v in ts_simplified.variants(as_bytes=True)],
            [b"1000", b"0100", b"0010", b"0001"])
        # The site over the non-sample external node should have been discarded.
        sites = list(t.sites())
        self.assertEqual(sites[-1].position, 0.4)
        self.assertEqual(t.parent_dict, {0: 4, 1: 4, 2: 5, 3: 5})

    def test_one_reducable_tree(self):
        # We have n = 4 and two trees. One tree is reducable and the other isn't.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       0           1
        5       0           1
        6       0           2
        7       0           3
        8       0           0   # Non sample leaf
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1      4         0
        0       1      5         1
        0       1      6         4,5
        0       1      7         2,3,8
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        self.assertEqual(ts.num_nodes, 9)
        self.assertEqual(ts.num_trees, 1)
        t = next(ts.trees())
        self.assertEqual(t.parent_dict, {0: 4, 1: 5, 2: 7, 3: 7, 4: 6, 5: 6, 8: 7})
        self.assertEqual(t.mrca(0, 1), 6)
        self.assertEqual(t.mrca(2, 3), 7)
        self.assertEqual(t.mrca(2, 8), 7)
        self.assertEqual(t.mrca(0, 2), tskit.NULL)
        self.assertEqual(t.mrca(0, 3), tskit.NULL)
        self.assertEqual(t.mrca(0, 8), tskit.NULL)
        ts_simplified = ts.simplify()
        self.assertEqual(ts_simplified.num_nodes, 6)
        self.assertEqual(ts_simplified.num_trees, 1)
        t = next(ts_simplified.trees())
        self.assertEqual(t.parent_dict, {0: 4, 1: 4, 2: 5, 3: 5})

    # NOTE: This test has not been checked since updating to the text representation
    # so there might be other problems with it.
    def test_mutations_over_roots(self):
        # Mutations over root nodes should be ok when we have multiple roots.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        5       0           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       3
        0       1       5       2
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        2   0.3         0
        3   0.4         0
        4   0.5         0
        5   0.6         0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state
        0       0       1
        1       1       1
        2       3       1
        3       4       1
        4       2       1
        5       5       1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        self.assertEqual(ts.num_nodes, 6)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_sites, 6)
        self.assertEqual(ts.num_mutations, 6)
        t = next(ts.trees())
        self.assertEqual(len(list(t.sites())), 6)
        haplotypes = ["101100", "011100", "000011"]
        variants = [b"100", b"010", b"110", b"110", b"001", b"001"]
        self.assertEqual(list(ts.haplotypes()), haplotypes)
        self.assertEqual([v.genotypes for v in ts.variants(as_bytes=True)], variants)
        ts_simplified = ts.simplify(filter_sites=False)
        self.assertEqual(
            list(ts_simplified.haplotypes(impute_missing_data=True)), haplotypes)
        self.assertEqual(
            variants, [
                v.genotypes for v in
                ts_simplified.variants(as_bytes=True, impute_missing_data=True)])

    def test_break_single_tree(self):
        # Take a single largish tree from tskit, and remove the oldest record.
        # This breaks it into two subtrees.
        ts = msprime.simulate(20, random_seed=self.random_seed, mutation_rate=4)
        self.assertGreater(ts.num_mutations, 5)
        tables = ts.dump_tables()
        tables.edges.set_columns(
            left=tables.edges.left[:-1],
            right=tables.edges.right[:-1],
            parent=tables.edges.parent[:-1],
            child=tables.edges.child[:-1])
        ts_new = tables.tree_sequence()
        self.assertEqual(ts.sample_size, ts_new.sample_size)
        self.assertEqual(ts.num_edges, ts_new.num_edges + 1)
        self.assertEqual(ts.num_trees, ts_new.num_trees)
        self.assert_haplotypes_equal(ts, ts_new)
        self.assert_variants_equal(ts, ts_new)
        roots = set()
        t_new = next(ts_new.trees())
        for u in ts_new.samples():
            while t_new.parent(u) != tskit.NULL:
                u = t_new.parent(u)
            roots.add(u)
        self.assertEqual(len(roots), 2)
        self.assertEqual(sorted(roots), sorted(t_new.roots))


class TestWithVisuals(TopologyTestCase):
    """
    Some pedantic tests with ascii depictions of what's supposed to happen.
    """

    def verify_simplify_topology(self, ts, sample, haplotypes=False):
        # copies from test_highlevel.py
        new_ts, node_map = ts.simplify(sample, map_nodes=True)
        old_trees = ts.trees()
        old_tree = next(old_trees)
        self.assertGreaterEqual(ts.get_num_trees(), new_ts.get_num_trees())
        for new_tree in new_ts.trees():
            new_left, new_right = new_tree.get_interval()
            old_left, old_right = old_tree.get_interval()
            # Skip ahead on the old tree until new_left is within its interval
            while old_right <= new_left:
                old_tree = next(old_trees)
                old_left, old_right = old_tree.get_interval()
            # If the TMRCA of all pairs of samples is the same, then we have the
            # same information. We limit this to at most 500 pairs
            pairs = itertools.islice(itertools.combinations(sample, 2), 500)
            for pair in pairs:
                mapped_pair = [node_map[u] for u in pair]
                mrca1 = old_tree.get_mrca(*pair)
                mrca2 = new_tree.get_mrca(*mapped_pair)
                self.assertEqual(mrca2, node_map[mrca1])
        if haplotypes:
            orig_haps = list(ts.haplotypes())
            simp_haps = list(new_ts.haplotypes())
            for i, j in enumerate(sample):
                self.assertEqual(orig_haps[j], simp_haps[i])

    def test_partial_non_sample_external_nodes(self):
        # A somewhat more complicated test case with a partially specified,
        # non-sampled tip.
        #
        # Here is the situation:
        #
        # 1.0             7
        # 0.7            / \                                            6
        #               /   \                                          / \
        # 0.5          /     5                      5                 /   5
        #             /     / \                    / \               /   / \
        # 0.4        /     /   4                  /   4             /   /   4
        #           /     /   / \                /   / \           /   /   / \
        #          /     /   3   \              /   /   \         /   /   3   \
        #         /     /         \            /   /     \       /   /         \
        # 0.0    0     1           2          1   0       2     0   1           2
        #
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)

        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           0.2  # Non sample leaf
        4       0           0.4
        5       0           0.5
        6       0           0.7
        7       0           1.0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.0     0.2     7       0,5
        """)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 7, 6: -1, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: -1, 4: 5, 5: -1, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1}]
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tree_dicts = [t.parent_dict for t in ts.trees()]
        self.assertEqual(ts.sample_size, 3)
        self.assertEqual(ts.num_trees, 3)
        self.assertEqual(ts.num_nodes, 8)
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        # check .simplify() works here
        self.verify_simplify_topology(ts, [0, 1, 2])

    def test_partial_non_sample_external_nodes_2(self):
        # The same situation as above, but partial tip is labeled '7' not '3':
        #
        # 1.0          6
        # 0.7         / \                                       5
        #            /   \                                     / \
        # 0.5       /     4                 4                 /   4
        #          /     / \               / \               /   / \
        # 0.4     /     /   3             /   3             /   /   3
        #        /     /   / \           /   / \           /   /   / \
        #       /     /   7   \         /   /   \         /   /   7   \
        #      /     /         \       /   /     \       /   /         \
        # 0.0 0     1           2     1   0       2     0   1           2
        #
        #          (0.0, 0.2),         (0.2, 0.8),         (0.8, 1.0)
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           0.4
        4       0           0.5
        5       0           0.7
        6       0           1.0
        7       0           0    # Non sample leaf
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     0.2     3       2,7
        0.2     0.8     3       0,2
        0.8     1.0     3       2,7
        0.0     0.2     4       1,3
        0.2     0.8     4       1,3
        0.8     1.0     4       1,3
        0.8     1.0     5       0,4
        0.0     0.2     6       0,4
        """)
        true_trees = [
            {0: 6, 1: 4, 2: 3, 3: 4, 4: 6, 5: -1, 6: -1, 7: 3},
            {0: 3, 1: 4, 2: 3, 3: 4, 4: -1, 5: -1, 6: -1, 7: -1},
            {0: 5, 1: 4, 2: 3, 3: 4, 4: 5, 5: -1, 6: -1, 7: 3}]
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tree_dicts = [t.parent_dict for t in ts.trees()]
        # sample size check works here since 7 > 3
        self.assertEqual(ts.sample_size, 3)
        self.assertEqual(ts.num_trees, 3)
        self.assertEqual(ts.num_nodes, 8)
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        self.verify_simplify_topology(ts, [0, 1, 2])

    def test_single_offspring_records(self):
        # Here we have inserted a single-offspring record
        # (for 6 on the left segment):
        #
        # 1.0             7
        # 0.7            / 6                                                  6
        #               /   \                                                / \
        # 0.5          /     5                       5                      /   5
        #             /     / \                     / \                    /   / \
        # 0.4        /     /   4                   /   4                  /   /   4
        # 0.3       /     /   / \                 /   / \                /   /   / \
        #          /     /   3   \               /   /   \              /   /   3   \
        #         /     /         \             /   /     \            /   /         \
        # 0.0    0     1           2           1   0       2          0   1           2
        #
        #          (0.0, 0.2),               (0.2, 0.8),              (0.8, 1.0)
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           0
        3   0           0       # Non sample leaf
        4   0           0.4
        5   0           0.5
        6   0           0.7
        7   0           1.0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.0     0.2     6       5
        0.0     0.2     7       0,6
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: 7, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: -1, 4: 5, 5: -1, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1}]
        tree_dicts = [t.parent_dict for t in ts.trees()]
        self.assertEqual(ts.sample_size, 3)
        self.assertEqual(ts.num_trees, 3)
        self.assertEqual(ts.num_nodes, 8)
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        self.verify_simplify_topology(ts, [0, 1, 2])

    def test_many_single_offspring(self):
        # a more complex test with single offspring
        # With `(i,j,x)->k` denoting that individual `k` inherits from `i` on `[0,x)`
        #    and from `j` on `[x,1)`:
        # 1. Begin with an individual `3` (and another anonymous one) at `t=0`.
        # 2. `(3,?,1.0)->4` and `(3,?,1.0)->5` at `t=1`
        # 3. `(4,3,0.9)->6` and `(3,5,0.1)->7` and then `3` dies at `t=2`
        # 4. `(6,7,0.7)->8` at `t=3`
        # 5. `(8,6,0.8)->9` and `(7,8,0.2)->10` at `t=4`.
        # 6. `(3,9,0.6)->0` and `(9,10,0.5)->1` and `(10,4,0.4)->2` at `t=5`.
        # 7. We sample `0`, `1`, and `2`.
        # Here are the trees:
        # t                  |              |              |             |
        #
        # 0       --3--      |     --3--    |     --3--    |    --3--    |    --3--
        #        /  |  \     |    /  |  \   |    /     \   |   /     \   |   /     \
        # 1     4   |   5    |   4   *   5  |   4       5  |  4       5  |  4       5
        #       |\ / \ /|    |   |\   \     |   |\     /   |  |\     /   |  |\     /|
        # 2     | 6   7 |    |   | 6   7    |   | 6   7    |  | 6   7    |  | 6   7 |
        #       | |\ /| |    |   |  \  *    |   |  \  |    |  |  *       |  |  *    | ...
        # 3     | | 8 | |    |   |   8 |    |   *   8 *    |  |   8      |  |   8   |
        #       | |/ \| |    |   |  /  |    |   |  /  |    |  |  * *     |  |  / \  |
        # 4     | 9  10 |    |   | 9  10    |   | 9  10    |  | 9  10    |  | 9  10 |
        #       |/ \ / \|    |   |  \   *   |   |  \   \   |  |  \   *   |  |  \    |
        # 5     0   1   2    |   0   1   2  |   0   1   2  |  0   1   2  |  0   1   2
        #
        #                    |   0.0 - 0.1  |   0.1 - 0.2  |  0.2 - 0.4  |  0.4 - 0.5
        # ... continued:
        # t                  |             |             |             |
        #
        # 0         --3--    |    --3--    |    --3--    |    --3--    |    --3--
        #          /     \   |   /     \   |   /     \   |   /     \   |   /  |  \
        # 1       4       5  |  4       5  |  4       5  |  4       5  |  4   |   5
        #         |\     /|  |   \     /|  |   \     /|  |   \     /|  |     /   /|
        # 2       | 6   7 |  |    6   7 |  |    6   7 |  |    6   7 |  |    6   7 |
        #         |  \    |  |     \    |  |       /  |  |    |  /  |  |    |  /  |
        # 3  ...  |   8   |  |      8   |  |      8   |  |    | 8   |  |    | 8   |
        #         |  / \  |  |     / \  |  |     / \  |  |    |  \  |  |    |  \  |
        # 4       | 9  10 |  |    9  10 |  |    9  10 |  |    9  10 |  |    9  10 |
        #         |    /  |  |   /   /  |  |   /   /  |  |   /   /  |  |   /   /  |
        # 5       0   1   2  |  0   1   2  |  0   1   2  |  0   1   2  |  0   1   2
        #
        #         0.5 - 0.6  |  0.6 - 0.7  |  0.7 - 0.8  |  0.8 - 0.9  |  0.9 - 1.0

        true_trees = [
            {0: 4, 1: 9, 2: 10, 3: -1, 4: 3, 5: 3, 6: 4, 7: 3, 8: 6, 9: 8, 10: 7},
            {0: 4, 1: 9, 2: 10, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 7},
            {0: 4, 1: 9, 2: 10, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 8},
            {0: 4, 1: 9,  2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 8},
            {0: 4, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 8},
            {0: 9, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 8},
            {0: 9, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 7, 9: 8, 10: 8},
            {0: 9, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 7, 9: 6, 10: 8},
            {0: 9, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 3, 7: 5, 8: 7, 9: 6, 10: 8}
        ]
        true_haplotypes = ['0100', '0001', '1110']
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           5
        4       0           4
        5       0           4
        6       0           3
        7       0           3
        8       0           2
        9       0           1
        10      0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.5     1.0     10      1
        0.0     0.4     10      2
        0.6     1.0     9       0
        0.0     0.5     9       1
        0.8     1.0     8       10
        0.2     0.8     8       9,10
        0.0     0.2     8       9
        0.7     1.0     7       8
        0.0     0.2     7       10
        0.8     1.0     6       9
        0.0     0.7     6       8
        0.4     1.0     5       2,7
        0.1     0.4     5       7
        0.6     0.9     4       6
        0.0     0.6     4       0,6
        0.9     1.0     3       4,5,6
        0.1     0.9     3       4,5
        0.0     0.1     3       4,5,7
        """)
        sites = io.StringIO("""\
        position    ancestral_state
        0.05        0
        0.15        0
        0.25        0
        0.4         0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       7       1               -1
        0      10       0               0
        0       2       1               1
        1       0       1               -1
        1      10       1               -1
        2       8       1               -1
        2       9       0               5
        2      10       0               5
        2       2       1               7
        3       8       1               -1
        """)
        ts = tskit.load_text(nodes, edges, sites, mutations, strict=False)
        tree_dicts = [t.parent_dict for t in ts.trees()]
        self.assertEqual(ts.sample_size, 3)
        self.assertEqual(ts.num_trees, len(true_trees))
        self.assertEqual(ts.num_nodes, 11)
        self.assertEqual(len(list(ts.edge_diffs())), ts.num_trees)
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        for j, x in enumerate(ts.haplotypes()):
            self.assertEqual(x, true_haplotypes[j])
        self.verify_simplify_topology(ts, [0, 1, 2], haplotypes=True)
        self.verify_simplify_topology(ts, [1, 0, 2], haplotypes=True)
        self.verify_simplify_topology(ts, [0, 1], haplotypes=False)
        self.verify_simplify_topology(ts, [1, 2], haplotypes=False)
        self.verify_simplify_topology(ts, [2, 0], haplotypes=False)

    def test_tricky_switches(self):
        # suppose the topology has:
        # left right parent child
        #  0.0   0.5      6      0,1
        #  0.5   1.0      6      4,5
        #  0.0   0.4      7      2,3
        #
        # --------------------------
        #
        #        12         .        12         .        12         .
        #       /  \        .       /  \        .       /  \        .
        #     11    \       .      /    \       .      /    \       .
        #     / \    \      .     /     10      .     /     10      .
        #    /   \    \     .    /     /  \     .    /     /  \     .
        #   6     7    8    .   6     9    8    .   6     9    8    .
        #  / \   / \   /\   .  / \   / \   /\   .  / \   / \   /\   .
        # 0   1 2   3 4  5  . 0   1 2   3 4  5  . 4   5 2   3 0  1  .
        #                   .                   .                   .
        # 0.0              0.4                 0.5                 1.0
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       1           0
        5       1           0
        6       0           1
        7       0           1
        8       0           1
        9       0           1
        10      0           2
        11      0           3
        12      0           4
        """)
        edges = io.StringIO("""\
        left right parent child
        0.0  0.5   6      0
        0.0  0.5   6      1
        0.5  1.0   6      4
        0.5  1.0   6      5
        0.0  0.4   7      2,3
        0.5  1.0   8      0
        0.5  1.0   8      1
        0.0  0.5   8      4
        0.0  0.5   8      5
        0.4  1.0   9      2,3
        0.4  1.0   10     8,9
        0.0  0.4   11     6,7
        0.4  1.0   12     6
        0.0  0.4   12     8
        0.4  1.0   12     10
        0.0  0.4   12     11
        """)
        true_trees = [
            {0: 6, 1: 6, 2: 7, 3: 7, 4: 8, 5: 8, 6: 11,
                7: 11, 8: 12, 9: -1, 10: -1, 11: 12, 12: -1},
            {0: 6, 1: 6, 2: 9, 3: 9, 4: 8, 5: 8, 6: 12,
                7: -1, 8: 10, 9: 10, 10: 12, 11: -1, 12: -1},
            {0: 8, 1: 8, 2: 9, 3: 9, 4: 6, 5: 6, 6: 12,
                7: -1, 8: 10, 9: 10, 10: 12, 11: -1, 12: -1}
        ]
        ts = tskit.load_text(nodes, edges, strict=False)
        tree_dicts = [t.parent_dict for t in ts.trees()]
        self.assertEqual(ts.sample_size, 6)
        self.assertEqual(ts.num_trees, len(true_trees))
        self.assertEqual(ts.num_nodes, 13)
        self.assertEqual(len(list(ts.edge_diffs())), ts.num_trees)
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        self.verify_simplify_topology(ts, [0, 2])
        self.verify_simplify_topology(ts, [0, 4])
        self.verify_simplify_topology(ts, [2, 4])

    def test_tricky_simplify(self):
        # Continue as above but invoke simplfy:
        #
        #         12         .          12         .
        #        /  \        .         /  \        .
        #      11    \       .       11    \       .
        #      / \    \      .       / \    \      .
        #    13   \    \     .      /  15    \     .
        #    / \   \    \    .     /   / \    \    .
        #   6  14   7    8   .    6  14   7    8   .
        #  / \     / \   /\  .   / \     / \   /\  .
        # 0   1   2   3 4  5 .  0   1   2   3 4  5 .
        #                    .                     .
        # 0.0               0.1                   0.4
        #
        #  .        12         .        12         .
        #  .       /  \        .       /  \        .
        #  .      /    \       .      /    \       .
        #  .     /     10      .     /     10      .
        #  .    /     /  \     .    /     /  \     .
        #  .   6     9    8    .   6     9    8    .
        #  .  / \   / \   /\   .  / \   / \   /\   .
        #  . 0   1 2   3 4  5  . 4   5 2   3 0  1  .
        #  .                   .                   .
        # 0.4                 0.5                 1.0
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       1           0
        5       1           0
        6       0           1
        7       0           1
        8       0           1
        9       0           1
        10      0           2
        11      0           3
        12      0           4
        13      0           2
        14      0           1
        15      0           2
        """)
        edges = io.StringIO("""\
        left right parent child
        0.0  0.5   6      0,1
        0.5  1.0   6      4,5
        0.0  0.4   7      2,3
        0.0  0.5   8      4,5
        0.5  1.0   8      0,1
        0.4  1.0   9      2,3
        0.4  1.0   10     8,9
        0.0  0.1   13     6,14
        0.1  0.4   15     7,14
        0.0  0.1   11     7,13
        0.1  0.4   11     6,15
        0.0  0.4   12     8,11
        0.4  1.0   12     6,10
        """)
        true_trees = [
            {0: 6, 1: 6, 2: 7, 3: 7, 4: 8, 5: 8, 6: 11,
                7: 11, 8: 12, 9: -1, 10: -1, 11: 12, 12: -1},
            {0: 6, 1: 6, 2: 9, 3: 9, 4: 8, 5: 8, 6: 12,
                7: -1, 8: 10, 9: 10, 10: 12, 11: -1, 12: -1},
            {0: 8, 1: 8, 2: 9, 3: 9, 4: 6, 5: 6, 6: 12,
                7: -1, 8: 10, 9: 10, 10: 12, 11: -1, 12: -1}
        ]
        big_ts = tskit.load_text(nodes, edges, strict=False)
        self.assertEqual(big_ts.num_trees, 1 + len(true_trees))
        self.assertEqual(big_ts.num_nodes, 16)
        ts, node_map = big_ts.simplify(map_nodes=True)
        self.assertEqual(list(node_map[:6]), list(range(6)))
        self.assertEqual(ts.sample_size, 6)
        self.assertEqual(ts.num_nodes, 13)

    def test_ancestral_samples(self):
        # Check that specifying samples to be not at time 0.0 works.
        #
        # 1.0             7
        # 0.7            / \                      8                     6
        #               /   \                    / \                   / \
        # 0.5          /     5                  /   5                 /   5
        #             /     / \                /   / \               /   / \
        # 0.4        /     /   4              /   /   4             /   /   4
        #           /     /   / \            /   /   / \           /   /   / \
        # 0.2      /     /   3   \          3   /   /   \         /   /   3   \
        #         /     /    *    \         *  /   /     \       /   /    *    \
        # 0.0    0     1           2          1   0       2     0   1           2
        #              *           *          *           *         *           *
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)
        #
        # Simplified, keeping [1,2,3]
        #
        # 1.0
        # 0.7                                     5
        #                                        / \
        # 0.5                4                  /   4                     4
        #                   / \                /   / \                   / \
        # 0.4              /   3              /   /   3                 /   3
        #                 /   / \            /   /     \               /   / \
        # 0.2            /   2   \          2   /       \             /   2   \
        #               /    *    \         *  /         \           /    *    \
        # 0.0          0           1          0           1         0           1
        #              *           *          *           *         *           *
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)

        nodes = io.StringIO("""\
        id      is_sample   time
        0       0           0
        1       1           0
        2       1           0
        3       1           0.2
        4       0           0.4
        5       0           0.5
        6       0           0.7
        7       0           1.0
        8       0           0.8
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.2     0.8     8       3,5
        0.0     0.2     7       0,5
        """)
        first_ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        ts, node_map = first_ts.simplify(map_nodes=True)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 7, 6: -1, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: 8, 4: 5, 5: 8, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1}]
        # maps [1,2,3] -> [0,1,2]
        self.assertEqual(node_map[1], 0)
        self.assertEqual(node_map[2], 1)
        self.assertEqual(node_map[3], 2)
        true_simplified_trees = [
            {0: 4, 1: 3, 2: 3, 3: 4},
            {0: 4, 1: 4, 2: 5, 4: 5},
            {0: 4, 1: 3, 2: 3, 3: 4}]
        self.assertEqual(first_ts.sample_size, 3)
        self.assertEqual(ts.sample_size, 3)
        self.assertEqual(first_ts.num_trees, 3)
        self.assertEqual(ts.num_trees, 3)
        self.assertEqual(first_ts.num_nodes, 9)
        self.assertEqual(ts.num_nodes, 6)
        self.assertEqual(first_ts.node(3).time, 0.2)
        self.assertEqual(ts.node(2).time, 0.2)
        # check topologies agree:
        tree_dicts = [t.parent_dict for t in first_ts.trees()]
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        tree_simplified_dicts = [t.parent_dict for t in ts.trees()]
        for a, t in zip(true_simplified_trees, tree_simplified_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        # check .simplify() works here
        self.verify_simplify_topology(first_ts, [1, 2, 3])

    def test_all_ancestral_samples(self):
        # Check that specifying samples all to be not at time 0.0 works.
        #
        # 1.0             7
        # 0.7            / \                      8                     6
        #               /   \                    / \                   / \
        # 0.5          /     5                  /   5                 /   5
        #             /     / \                /   / \               /   / \
        # 0.4        /     /   4              /   /   4             /   /   4
        #           /     /   / \            /   /   / \           /   /   / \
        # 0.2      /     /   3   \          3   /   /   \         /   /   3   \
        #         /     1    *    2         *  1   /     2       /   1    *    2
        # 0.0    0      *         *            *  0      *      0    *         *
        #
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)

        nodes = io.StringIO("""\
        id      is_sample   time
        0       0           0
        1       1           0.1
        2       1           0.1
        3       1           0.2
        4       0           0.4
        5       0           0.5
        6       0           0.7
        7       0           1.0
        8       0           0.8
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.2     0.8     8       3,5
        0.0     0.2     7       0,5
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 7, 6: -1, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: 8, 4: 5, 5: 8, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1}]
        self.assertEqual(ts.sample_size, 3)
        self.assertEqual(ts.num_trees, 3)
        self.assertEqual(ts.num_nodes, 9)
        self.assertEqual(ts.node(0).time, 0.0)
        self.assertEqual(ts.node(1).time, 0.1)
        self.assertEqual(ts.node(2).time, 0.1)
        self.assertEqual(ts.node(3).time, 0.2)
        # check topologies agree:
        tree_dicts = [t.parent_dict for t in ts.trees()]
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        # check .simplify() works here
        self.verify_simplify_topology(ts, [1, 2, 3])

    def test_internal_sampled_node(self):
        # 1.0             7
        # 0.7            / \                      8                     6
        #               /   \                    / \                   / \
        # 0.5          /     5                  /   5                 /   5
        #             /     /*\                /   /*\               /   /*\
        # 0.4        /     /   4              /   /   4             /   /   4
        #           /     /   / \            /   /   / \           /   /   / \
        # 0.2      /     /   3   \          3   /   /   \         /   /   3   \
        #         /     1    *    2         *  1   /     2       /   1    *    2
        # 0.0    0      *         *            *  0      *      0    *         *
        #
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)
        nodes = io.StringIO("""\
        id      is_sample   time
        0       0           0
        1       1           0.1
        2       1           0.1
        3       1           0.2
        4       0           0.4
        5       1           0.5
        6       0           0.7
        7       0           1.0
        8       0           0.8
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.2     0.8     8       3,5
        0.0     0.2     7       0,5
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 7, 6: -1, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: 8, 4: 5, 5: 8, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1}]
        self.assertEqual(ts.sample_size, 4)
        self.assertEqual(ts.num_trees, 3)
        self.assertEqual(ts.num_nodes, 9)
        self.assertEqual(ts.node(0).time, 0.0)
        self.assertEqual(ts.node(1).time, 0.1)
        self.assertEqual(ts.node(2).time, 0.1)
        self.assertEqual(ts.node(3).time, 0.2)
        # check topologies agree:
        tree_dicts = [t.parent_dict for t in ts.trees()]
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    self.assertEqual(t[k], a[k])
                else:
                    self.assertEqual(a[k], tskit.NULL)
        # check .simplify() works here
        self.verify_simplify_topology(ts, [1, 2, 3])
        self.check_num_samples(
            ts,
            [(0, 5, 4), (0, 2, 1), (0, 7, 4), (0, 4, 2),
             (1, 4, 1), (1, 5, 3), (1, 8, 4), (1, 0, 0),
             (2, 5, 4), (2, 1, 1)])
        self.check_num_tracked_samples(
            ts, [1, 2, 5],
            [(0, 5, 3), (0, 2, 1), (0, 7, 3), (0, 4, 1),
             (1, 4, 1), (1, 5, 3), (1, 8, 3), (1, 0, 0),
             (2, 5, 3), (2, 1, 1)])
        self.check_sample_iterator(
            ts,
            [(0, 0, []), (0, 5, [5, 1, 2, 3]), (0, 4, [2, 3]),
             (1, 5, [5, 1, 2]), (2, 4, [2, 3])])
        # pedantically check the Tree methods on the second tree
        tst = ts.trees()
        t = next(tst)
        t = next(tst)
        self.assertEqual(t.branch_length(1), 0.4)
        self.assertEqual(t.is_internal(0), False)
        self.assertEqual(t.is_leaf(0), True)
        self.assertEqual(t.is_sample(0), False)
        self.assertEqual(t.is_internal(1), False)
        self.assertEqual(t.is_leaf(1), True)
        self.assertEqual(t.is_sample(1), True)
        self.assertEqual(t.is_internal(5), True)
        self.assertEqual(t.is_leaf(5), False)
        self.assertEqual(t.is_sample(5), True)
        self.assertEqual(t.is_internal(4), True)
        self.assertEqual(t.is_leaf(4), False)
        self.assertEqual(t.is_sample(4), False)
        self.assertEqual(t.root, 8)
        self.assertEqual(t.mrca(0, 1), 5)
        self.assertEqual(t.sample_size, 4)


class TestBadTrees(unittest.TestCase):
    """
    Tests for bad tree sequence topologies that can only be detected when we
    try to create trees.
    """
    def test_simplest_contradictory_children(self):
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     1.0     2       0
        0.0     1.0     3       0
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        self.assertRaises(_tskit.LibraryError, list, ts.trees())

    def test_partial_overlap_contradictory_children(self):
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     1.0     2       0,1
        0.5     1.0     3       0
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        self.assertRaises(_tskit.LibraryError, list, ts.trees())


class TestSimplify(unittest.TestCase):
    """
    Tests that the implementations of simplify() do what they are supposed to.
    """
    random_seed = 23
    #
    #          8
    #         / \
    #        /   \
    #       /     \
    #      7       \
    #     / \       6
    #    /   5     / \
    #   /   / \   /   \
    #  4   0   1 2     3
    small_tree_ex_nodes = """\
    id      is_sample   population      time
    0       1       0               0.00000000000000
    1       1       0               0.00000000000000
    2       1       0               0.00000000000000
    3       1       0               0.00000000000000
    4       1       0               0.00000000000000
    5       0       0               0.14567111023387
    6       0       0               0.21385545626353
    7       0       0               0.43508024345063
    8       0       0               1.60156352971203
    """
    small_tree_ex_edges = """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      5       0,1
    1       0.00000000      1.00000000      6       2,3
    2       0.00000000      1.00000000      7       4,5
    3       0.00000000      1.00000000      8       6,7
    """

    def do_simplify(
            self, ts, samples=None, compare_lib=True, filter_sites=True,
            filter_populations=True, filter_individuals=True, keep_unary=False):
        """
        Runs the Python test implementation of simplify.
        """
        if samples is None:
            samples = ts.samples()
        s = tests.Simplifier(
            ts, samples, filter_sites=filter_sites,
            filter_populations=filter_populations, filter_individuals=filter_individuals,
            keep_unary=keep_unary)
        new_ts, node_map = s.simplify()
        if compare_lib:
            sts, lib_node_map1 = ts.simplify(
                samples,
                filter_sites=filter_sites,
                filter_individuals=filter_individuals,
                filter_populations=filter_populations,
                keep_unary=keep_unary,
                map_nodes=True)
            lib_tables1 = sts.dump_tables()

            lib_tables2 = ts.dump_tables()
            lib_node_map2 = lib_tables2.simplify(
                samples,
                filter_sites=filter_sites,
                keep_unary=keep_unary,
                filter_individuals=filter_individuals,
                filter_populations=filter_populations)

            py_tables = new_ts.dump_tables()
            for lib_tables, lib_node_map in [
                    (lib_tables1, lib_node_map1), (lib_tables2, lib_node_map2)]:

                self.assertEqual(lib_tables.nodes, py_tables.nodes)
                self.assertEqual(lib_tables.edges, py_tables.edges)
                self.assertEqual(lib_tables.migrations, py_tables.migrations)
                self.assertEqual(lib_tables.sites, py_tables.sites)
                self.assertEqual(lib_tables.mutations, py_tables.mutations)
                self.assertEqual(lib_tables.individuals, py_tables.individuals)
                self.assertEqual(lib_tables.populations, py_tables.populations)
                self.assertTrue(all(node_map == lib_node_map))
        return new_ts, node_map

    def verify_no_samples(self, ts, keep_unary=False):
        """
        Zero out the flags column and verify that we get back the correct
        tree sequence when we run simplify.
        """
        t1 = ts.dump_tables()
        t1.nodes.flags = np.zeros_like(t1.nodes.flags)
        ts1, node_map1 = self.do_simplify(
            ts, samples=ts.samples(), keep_unary=keep_unary)
        t1 = ts1.dump_tables()
        ts2, node_map2 = self.do_simplify(ts, keep_unary=keep_unary)
        t2 = ts2.dump_tables()
        self.assertEqual(t1, t2)

    def verify_single_childified(self, ts, keep_unary=False):
        """
        Modify the specified tree sequence so that it has lots of unary
        nodes. Run simplify and verify we get the same tree sequence back
        if keep_unary is False. If keep_unary is True, the simplication
        won't do anything to the original treeSequence.
        """
        ts_single = tsutil.single_childify(ts)

        tss, node_map = self.do_simplify(ts_single, keep_unary=keep_unary)
        # All original nodes should still be present.
        for u in range(ts.num_samples):
            self.assertEqual(u, node_map[u])
        # All introduced nodes should be mapped to null.
        for u in range(ts.num_samples, ts_single.num_samples):
            self.assertEqual(node_map[u], tskit.NULL)
        t1 = ts.dump_tables()
        t2 = tss.dump_tables()
        t3 = ts_single.dump_tables()
        if keep_unary:
            self.assertEqual(set(t3.nodes.time), set(t2.nodes.time))
            self.assertEqual(len(t3.edges), len(t2.edges))
            self.assertEqual(t3.sites, t2.sites)
            self.assertEqual(len(t3.mutations), len(t2.mutations))
        else:
            self.assertEqual(t1.nodes, t2.nodes)
            self.assertEqual(t1.edges, t2.edges)
            self.assertEqual(t1.sites, t2.sites)
            self.assertEqual(t1.mutations, t2.mutations)

    def verify_multiroot_internal_samples(self, ts, keep_unary=False):
        ts_multiroot = tsutil.decapitate(ts, ts.num_edges // 2)
        ts1 = tsutil.jiggle_samples(ts_multiroot)
        ts2, node_map = self.do_simplify(ts1, keep_unary=keep_unary)
        self.assertGreaterEqual(ts1.num_trees, ts2.num_trees)
        trees2 = ts2.trees()
        t2 = next(trees2)
        for t1 in ts1.trees():
            self.assertTrue(t2.interval[0] <= t1.interval[0])
            self.assertTrue(t2.interval[1] >= t1.interval[1])
            pairs = itertools.combinations(ts1.samples(), 2)
            for pair in pairs:
                mapped_pair = [node_map[u] for u in pair]
                mrca1 = t1.get_mrca(*pair)
                mrca2 = t2.get_mrca(*mapped_pair)
                if mrca1 == tskit.NULL:
                    assert mrca2 == tskit.NULL
                else:
                    self.assertEqual(node_map[mrca1], mrca2)
            if t2.interval[1] == t1.interval[1]:
                t2 = next(trees2, None)

    def test_single_tree(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        self.verify_no_samples(ts)
        self.verify_single_childified(ts)
        self.verify_multiroot_internal_samples(ts)
        # Now with keep_unary=True.
        self.verify_no_samples(ts, keep_unary=True)
        self.verify_single_childified(ts, keep_unary=True)
        self.verify_multiroot_internal_samples(ts, keep_unary=True)

    def test_single_tree_mutations(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=self.random_seed)
        self.assertGreater(ts.num_sites, 1)
        self.do_simplify(ts)
        self.verify_single_childified(ts)
        # Also with keep_unary == True.
        self.do_simplify(ts, keep_unary=True)
        self.verify_single_childified(ts, keep_unary=True)

    def test_many_trees_mutations(self):
        ts = msprime.simulate(
            10, recombination_rate=1, mutation_rate=10, random_seed=self.random_seed)
        self.assertGreater(ts.num_trees, 2)
        self.assertGreater(ts.num_sites, 2)
        self.verify_no_samples(ts)
        self.do_simplify(ts)
        self.verify_single_childified(ts)
        # Also with keep_unary == True.
        self.do_simplify(ts, keep_unary=True)
        self.verify_single_childified(ts, keep_unary=True)

    def test_many_trees(self):
        ts = msprime.simulate(5, recombination_rate=4, random_seed=self.random_seed)
        self.assertGreater(ts.num_trees, 2)
        self.verify_no_samples(ts)
        self.verify_single_childified(ts)
        self.verify_multiroot_internal_samples(ts)
        # Also with keep_unary == True.
        self.verify_no_samples(ts, keep_unary=True)
        self.verify_single_childified(ts, keep_unary=True)
        self.verify_multiroot_internal_samples(ts, keep_unary=True)

    def test_small_tree_internal_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # The parent of samples 0 and 1 is 5. Change this to an internal sample
        # and set 0 and 1 to be unsampled.
        flags[0] = 0
        flags[0] = 0
        flags[5] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        ts = tables.tree_sequence()
        self.assertEqual(ts.sample_size, 5)
        tss, node_map = self.do_simplify(ts, [3, 5])
        self.assertEqual(node_map[3], 0)
        self.assertEqual(node_map[5], 1)
        self.assertEqual(tss.num_nodes, 3)
        self.assertEqual(tss.num_edges, 2)
        self.verify_no_samples(ts)
        # with keep_unary == True
        tss, node_map = self.do_simplify(ts, [3, 5], keep_unary=True)
        self.assertEqual(node_map[3], 0)
        self.assertEqual(node_map[5], 1)
        self.assertEqual(tss.num_nodes, 5)
        self.assertEqual(tss.num_edges, 4)
        self.verify_no_samples(ts, keep_unary=True)

    def test_small_tree_linear_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # 7 is above 0. These are the only two samples
        flags[:] = 0
        flags[0] = tskit.NODE_IS_SAMPLE
        flags[7] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        ts = tables.tree_sequence()
        self.assertEqual(ts.sample_size, 2)
        tss, node_map = self.do_simplify(ts, [0, 7])
        self.assertEqual(node_map[0], 0)
        self.assertEqual(node_map[7], 1)
        self.assertEqual(tss.num_nodes, 2)
        self.assertEqual(tss.num_edges, 1)
        t = next(tss.trees())
        self.assertEqual(t.parent_dict, {0: 1})
        # with keep_unary == True
        tss, node_map = self.do_simplify(ts, [0, 7], keep_unary=True)
        self.assertEqual(node_map[0], 0)
        self.assertEqual(node_map[7], 1)
        self.assertEqual(tss.num_nodes, 4)
        self.assertEqual(tss.num_edges, 3)
        t = next(tss.trees())

    def test_small_tree_internal_and_external_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # 7 is above 0 and 1.
        flags[:] = 0
        flags[0] = tskit.NODE_IS_SAMPLE
        flags[1] = tskit.NODE_IS_SAMPLE
        flags[7] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        ts = tables.tree_sequence()
        self.assertEqual(ts.sample_size, 3)
        tss, node_map = self.do_simplify(ts, [0, 1, 7])
        self.assertEqual(node_map[0], 0)
        self.assertEqual(node_map[1], 1)
        self.assertEqual(node_map[7], 2)
        self.assertEqual(tss.num_nodes, 4)
        self.assertEqual(tss.num_edges, 3)
        t = next(tss.trees())
        self.assertEqual(t.parent_dict, {0: 3, 1: 3, 3: 2})
        # with keep_unary == True
        tss, node_map = self.do_simplify(ts, [0, 1, 7], keep_unary=True)
        self.assertEqual(node_map[0], 0)
        self.assertEqual(node_map[1], 1)
        self.assertEqual(node_map[7], 2)
        self.assertEqual(tss.num_nodes, 5)
        self.assertEqual(tss.num_edges, 4)
        t = next(tss.trees())
        self.assertEqual(t.parent_dict, {0: 3, 1: 3, 3: 2, 2: 4})

    def test_small_tree_mutations(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        tables = ts.dump_tables()
        # Add some simple mutations here above the nodes we're keeping.
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.sites.add_row(position=0.75, ancestral_state="0")
        tables.sites.add_row(position=0.8, ancestral_state="0")
        tables.mutations.add_row(site=0, node=0, derived_state="1")
        tables.mutations.add_row(site=1, node=2, derived_state="1")
        tables.mutations.add_row(site=2, node=7, derived_state="1")
        tables.mutations.add_row(site=3, node=0, derived_state="1")
        ts = tables.tree_sequence()
        self.assertEqual(ts.num_sites, 4)
        self.assertEqual(ts.num_mutations, 4)
        for keep in [True, False]:
            tss = self.do_simplify(ts, [0, 2], keep_unary=keep)[0]
            self.assertEqual(tss.sample_size, 2)
            self.assertEqual(tss.num_mutations, 4)
            self.assertEqual(list(tss.haplotypes()), ["1011", "0100"])

    def test_small_tree_filter_zero_mutations(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        ts = tsutil.insert_branch_sites(ts)
        self.assertEqual(ts.num_sites, 8)
        self.assertEqual(ts.num_mutations, 8)
        for keep in [True, False]:
            tss, _ = self.do_simplify(
                ts, [4, 0, 1], filter_sites=True, keep_unary=keep)
            self.assertEqual(tss.num_sites, 5)
            self.assertEqual(tss.num_mutations, 5)
            tss, _ = self.do_simplify(
                ts, [4, 0, 1], filter_sites=False, keep_unary=keep)
            self.assertEqual(tss.num_sites, 8)
            self.assertEqual(tss.num_mutations, 5)

    def test_small_tree_fixed_sites(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        tables = ts.dump_tables()
        # Add some simple mutations that will be fixed after simplify
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.sites.add_row(position=0.75, ancestral_state="0")
        tables.mutations.add_row(site=0, node=2, derived_state="1")
        tables.mutations.add_row(site=1, node=3, derived_state="1")
        tables.mutations.add_row(site=2, node=6, derived_state="1")
        ts = tables.tree_sequence()
        self.assertEqual(ts.num_sites, 3)
        self.assertEqual(ts.num_mutations, 3)
        for keep in [True, False]:
            tss, _ = self.do_simplify(ts, [4, 1], keep_unary=keep)
            self.assertEqual(tss.sample_size, 2)
            self.assertEqual(tss.num_mutations, 0)
            self.assertEqual(list(tss.haplotypes()), ["", ""])

    def test_small_tree_mutations_over_root(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        tables = ts.dump_tables()
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.mutations.add_row(site=0, node=8, derived_state="1")
        ts = tables.tree_sequence()
        self.assertEqual(ts.num_sites, 1)
        self.assertEqual(ts.num_mutations, 1)
        for keep_unary, filter_sites in itertools.product([True, False], repeat=2):
            tss, _ = self.do_simplify(
                ts, [0, 1], filter_sites=filter_sites, keep_unary=keep_unary)
            self.assertEqual(tss.num_sites, 1)
            self.assertEqual(tss.num_mutations, 1)

    def test_small_tree_recurrent_mutations(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        tables = ts.dump_tables()
        # Add recurrent mutation on the root branches
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.mutations.add_row(site=0, node=6, derived_state="1")
        tables.mutations.add_row(site=0, node=7, derived_state="1")
        ts = tables.tree_sequence()
        self.assertEqual(ts.num_sites, 1)
        self.assertEqual(ts.num_mutations, 2)
        for keep in [True, False]:
            tss = self.do_simplify(ts, [4, 3], keep_unary=keep)[0]
            self.assertEqual(tss.sample_size, 2)
            self.assertEqual(tss.num_sites, 1)
            self.assertEqual(tss.num_mutations, 2)
            self.assertEqual(list(tss.haplotypes()), ["1", "1"])

    def test_small_tree_back_mutations(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges), strict=False)
        tables = ts.dump_tables()
        # Add a chain of mutations
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.mutations.add_row(site=0, node=7, derived_state="1")
        tables.mutations.add_row(site=0, node=5, derived_state="0")
        tables.mutations.add_row(site=0, node=1, derived_state="1")
        ts = tables.tree_sequence()
        self.assertEqual(ts.num_sites, 1)
        self.assertEqual(ts.num_mutations, 3)
        self.assertEqual(list(ts.haplotypes()), ["0", "1", "0", "0", "1"])
        # First check if we simplify for all samples and keep original state.
        for keep in [True, False]:
            tss = self.do_simplify(ts, [0, 1, 2, 3, 4], keep_unary=keep)[0]
            self.assertEqual(tss.sample_size, 5)
            self.assertEqual(tss.num_sites, 1)
            self.assertEqual(tss.num_mutations, 3)
            self.assertEqual(list(tss.haplotypes()), ["0", "1", "0", "0", "1"])

        # The ancestral state above 5 should be 0.
        for keep in [True, False]:
            tss = self.do_simplify(ts, [0, 1], keep_unary=keep)[0]
            self.assertEqual(tss.sample_size, 2)
            self.assertEqual(tss.num_sites, 1)
            self.assertEqual(tss.num_mutations, 3)
            self.assertEqual(list(tss.haplotypes()), ["0", "1"])

        # The ancestral state above 7 should be 1.
        for keep in [True, False]:
            tss = self.do_simplify(ts, [4, 0, 1], keep_unary=keep)[0]
            self.assertEqual(tss.sample_size, 3)
            self.assertEqual(tss.num_sites, 1)
            self.assertEqual(tss.num_mutations, 3)
            self.assertEqual(list(tss.haplotypes()), ["1", "0", "1"])

    def test_overlapping_unary_edges(self):
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       2       2       0
        1       3       2       1
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        self.assertEqual(ts.sample_size, 2)
        self.assertEqual(ts.num_trees, 3)
        self.assertEqual(ts.sequence_length, 3)
        for keep in [True, False]:
            tss, node_map = self.do_simplify(ts, samples=[0, 1, 2], keep_unary=keep)
            self.assertEqual(list(node_map), [0, 1, 2])
            trees = [{0: 2}, {0: 2, 1: 2}, {1: 2}]
            for t in tss.trees():
                self.assertEqual(t.parent_dict, trees[t.index])

    def test_overlapping_unary_edges_internal_samples(self):
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       2       2       0
        1       3       2       1
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        self.assertEqual(ts.sample_size, 3)
        self.assertEqual(ts.num_trees, 3)
        trees = [{0: 2}, {0: 2, 1: 2}, {1: 2}]
        for t in ts.trees():
            self.assertEqual(t.parent_dict, trees[t.index])
        for keep in [True, False]:
            tss, node_map = self.do_simplify(ts)
            self.assertEqual(list(node_map), [0, 1, 2])

    def test_isolated_samples(self):
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           1
        2       1           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        """)
        ts = tskit.load_text(nodes, edges, sequence_length=1, strict=False)
        self.assertEqual(ts.num_samples, 3)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_nodes, 3)
        for keep in [True, False]:
            tss, node_map = self.do_simplify(ts, keep_unary=keep)
            self.assertEqual(ts.tables.nodes, tss.tables.nodes)
            self.assertEqual(ts.tables.edges, tss.tables.edges)
            self.assertEqual(list(node_map), [0, 1, 2])

    def test_internal_samples(self):
        nodes = io.StringIO("""\
        id      is_sample   population      time
        0       1       -1              1.00000000000000
        1       0       -1              1.00000000000000
        2       1       -1              1.00000000000000
        3       0       -1              1.31203521181726
        4       0       -1              2.26776380586006
        5       1       -1              0.00000000000000
        6       0       -1              0.50000000000000
        7       0       -1              1.50000000000000

        """)
        edges = io.StringIO("""\
        id      left            right           parent  child
        0       0.62185118      1.00000000      1       6
        1       0.00000000      0.62185118      2       6
        2       0.00000000      1.00000000      3       0,2
        3       0.00000000      1.00000000      4       7,3
        4       0.00000000      1.00000000      6       5
        5       0.00000000      1.00000000      7       1
        """)

        ts = tskit.load_text(nodes, edges, strict=False)
        tss, node_map = self.do_simplify(ts, [5, 2, 0])
        self.assertEqual(node_map[0], 2)
        self.assertEqual(node_map[1], -1)
        self.assertEqual(node_map[2], 1)
        self.assertEqual(node_map[3], 3)
        self.assertEqual(node_map[4], 4)
        self.assertEqual(node_map[5], 0)
        self.assertEqual(node_map[6], -1)
        self.assertEqual(node_map[7], -1)
        self.assertEqual(tss.sample_size, 3)
        self.assertEqual(tss.num_trees, 2)
        trees = [{0: 1, 1: 3, 2: 3}, {0: 4, 1: 3, 2: 3, 3: 4}]
        for t in tss.trees():
            self.assertEqual(t.parent_dict, trees[t.index])
        # with keep_unary == True
        tss, node_map = self.do_simplify(ts, [5, 2, 0], keep_unary=True)
        self.assertEqual(node_map[0], 2)
        self.assertEqual(node_map[1], 4)
        self.assertEqual(node_map[2], 1)
        self.assertEqual(node_map[3], 5)
        self.assertEqual(node_map[4], 7)
        self.assertEqual(node_map[5], 0)
        self.assertEqual(node_map[6], 3)
        self.assertEqual(node_map[7], 6)
        self.assertEqual(tss.sample_size, 3)
        self.assertEqual(tss.num_trees, 2)
        trees = [{0: 3, 1: 5, 2: 5, 3: 1, 5: 7},
                 {0: 3, 1: 5, 2: 5, 3: 4, 4: 6, 5: 7, 6: 7}]
        for t in tss.trees():
            self.assertEqual(t.parent_dict, trees[t.index])

    def test_many_mutations_over_single_sample_ancestral_state(self):
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       1       0
        """)
        sites = io.StringIO("""\
        position    ancestral_state
        0           0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       0       1               -1
        0       0       0               0
        """)
        ts = tskit.load_text(
            nodes, edges, sites=sites, mutations=mutations, strict=False)
        self.assertEqual(ts.sample_size, 1)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_sites, 1)
        self.assertEqual(ts.num_mutations, 2)
        for keep in [True, False]:
            tss, node_map = self.do_simplify(ts, keep_unary=keep)
            self.assertEqual(tss.num_sites, 1)
            self.assertEqual(tss.num_mutations, 2)
            self.assertEqual(list(tss.haplotypes(impute_missing_data=True)), ["0"])

    def test_many_mutations_over_single_sample_derived_state(self):
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       0           1
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       1       0
        """)
        sites = io.StringIO("""\
        position    ancestral_state
        0           0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       0       1               -1
        0       0       0               0
        0       0       1               1
        """)
        ts = tskit.load_text(
            nodes, edges, sites=sites, mutations=mutations, strict=False)
        self.assertEqual(ts.sample_size, 1)
        self.assertEqual(ts.num_trees, 1)
        self.assertEqual(ts.num_sites, 1)
        self.assertEqual(ts.num_mutations, 3)
        for keep in [True, False]:
            tss, node_map = self.do_simplify(ts, keep_unary=keep)
            self.assertEqual(tss.num_sites, 1)
            self.assertEqual(tss.num_mutations, 3)
            self.assertEqual(list(tss.haplotypes(impute_missing_data=True)), ["1"])

    def test_many_trees_filter_zero_mutations(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        self.assertGreater(ts.num_trees, 3)
        ts = tsutil.insert_branch_sites(ts)
        self.assertEqual(ts.num_sites, ts.num_mutations)
        self.assertGreater(ts.num_sites, ts.num_trees)
        for keep in [True, False]:
            for filter_sites in [True, False]:
                tss, _ = self.do_simplify(
                    ts, samples=None, filter_sites=filter_sites, keep_unary=keep)
                self.assertEqual(ts.num_sites, tss.num_sites)
                self.assertEqual(ts.num_mutations, tss.num_mutations)

    def test_many_trees_filter_zero_multichar_mutations(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        self.assertGreater(ts.num_trees, 3)
        ts = tsutil.insert_multichar_mutations(ts)
        self.assertEqual(ts.num_sites, ts.num_trees)
        self.assertEqual(ts.num_mutations, ts.num_trees)
        for keep in [True, False]:
            for filter_sites in [True, False]:
                tss, _ = self.do_simplify(
                    ts, samples=None, filter_sites=filter_sites, keep_unary=keep)
                self.assertEqual(ts.num_sites, tss.num_sites)
                self.assertEqual(ts.num_mutations, tss.num_mutations)

    def test_simple_population_filter(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.dump_tables()
        tables.populations.add_row(metadata=b"unreferenced")
        self.assertEqual(len(tables.populations), 2)
        for keep in [True, False]:
            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_populations=True, keep_unary=keep)
            self.assertEqual(tss.num_populations, 1)
            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_populations=False, keep_unary=keep)
            self.assertEqual(tss.num_populations, 2)

    def test_interleaved_populations_filter(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(),
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(),
                msprime.PopulationConfiguration()],
            random_seed=2)
        self.assertEqual(ts.num_populations, 4)
        tables = ts.dump_tables()
        # Edit the populations so we can identify the rows.
        tables.populations.clear()
        for j in range(4):
            tables.populations.add_row(metadata=bytes([j]))
        ts = tables.tree_sequence()
        id_map = np.array([-1, 0, -1, -1], dtype=np.int32)
        for keep in [True, False]:
            tss, _ = self.do_simplify(
                ts, filter_populations=True, keep_unary=keep)
            self.assertEqual(tss.num_populations, 1)
            population = tss.population(0)
            self.assertEqual(population.metadata, bytes([1]))
            self.assertTrue(np.array_equal(
                id_map[ts.tables.nodes.population], tss.tables.nodes.population))
            tss, _ = self.do_simplify(
                ts, filter_populations=False, keep_unary=keep)
            self.assertEqual(tss.num_populations, 4)

    def test_removed_node_population_filter(self):
        tables = tskit.TableCollection(1)
        tables.populations.add_row(metadata=bytes(0))
        tables.populations.add_row(metadata=bytes(1))
        tables.populations.add_row(metadata=bytes(2))
        tables.nodes.add_row(flags=1, population=0)
        # Because flags=0 here, this node will be simplified out and the node
        # will disappear.
        tables.nodes.add_row(flags=0, population=1)
        tables.nodes.add_row(flags=1, population=2)
        for keep in [True, False]:
            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_populations=True, keep_unary=keep)
            self.assertEqual(tss.num_nodes, 2)
            self.assertEqual(tss.num_populations, 2)
            self.assertEqual(tss.population(0).metadata, bytes(0))
            self.assertEqual(tss.population(1).metadata, bytes(2))
            self.assertEqual(tss.node(0).population, 0)
            self.assertEqual(tss.node(1).population, 1)

            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_populations=False, keep_unary=keep)
            self.assertEqual(tss.tables.populations, tables.populations)

    def test_simple_individual_filter(self):
        tables = tskit.TableCollection(1)
        tables.individuals.add_row(flags=0)
        tables.individuals.add_row(flags=1)
        tables.nodes.add_row(flags=1, individual=0)
        tables.nodes.add_row(flags=1, individual=0)
        for keep in [True, False]:
            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_individuals=True, keep_unary=keep)
            self.assertEqual(tss.num_nodes, 2)
            self.assertEqual(tss.num_individuals, 1)
            self.assertEqual(tss.individual(0).flags, 0)

        tss, _ = self.do_simplify(tables.tree_sequence(), filter_individuals=False)
        self.assertEqual(tss.tables.individuals, tables.individuals)

    def test_interleaved_individual_filter(self):
        tables = tskit.TableCollection(1)
        tables.individuals.add_row(flags=0)
        tables.individuals.add_row(flags=1)
        tables.individuals.add_row(flags=2)
        tables.nodes.add_row(flags=1, individual=1)
        tables.nodes.add_row(flags=1, individual=-1)
        tables.nodes.add_row(flags=1, individual=1)
        for keep in [True, False]:
            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_individuals=True, keep_unary=keep)
            self.assertEqual(tss.num_nodes, 3)
            self.assertEqual(tss.num_individuals, 1)
            self.assertEqual(tss.individual(0).flags, 1)

            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_individuals=False, keep_unary=keep)
            self.assertEqual(tss.tables.individuals, tables.individuals)

    def test_removed_node_individual_filter(self):
        tables = tskit.TableCollection(1)
        tables.individuals.add_row(flags=0)
        tables.individuals.add_row(flags=1)
        tables.individuals.add_row(flags=2)
        tables.nodes.add_row(flags=1, individual=0)
        # Because flags=0 here, this node will be simplified out and the node
        # will disappear.
        tables.nodes.add_row(flags=0, individual=1)
        tables.nodes.add_row(flags=1, individual=2)
        for keep in [True, False]:
            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_individuals=True, keep_unary=keep)
            self.assertEqual(tss.num_nodes, 2)
            self.assertEqual(tss.num_individuals, 2)
            self.assertEqual(tss.individual(0).flags, 0)
            self.assertEqual(tss.individual(1).flags, 2)
            self.assertEqual(tss.node(0).individual, 0)
            self.assertEqual(tss.node(1).individual, 1)

            tss, _ = self.do_simplify(
                tables.tree_sequence(), filter_individuals=False, keep_unary=keep)
            self.assertEqual(tss.tables.individuals, tables.individuals)

    def verify_simplify_haplotypes(self, ts, samples, keep_unary=False):
        sub_ts, node_map = self.do_simplify(
            ts, samples, filter_sites=False, keep_unary=keep_unary)
        self.assertEqual(ts.num_sites, sub_ts.num_sites)
        sub_haplotypes = list(sub_ts.haplotypes(impute_missing_data=True))
        all_samples = list(ts.samples())
        k = 0
        for j, h in enumerate(ts.haplotypes(impute_missing_data=True)):
            if k == len(samples):
                break
            if samples[k] == all_samples[j]:
                self.assertEqual(h, sub_haplotypes[k])
                k += 1

    def test_single_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_many_trees_recurrent_mutations(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        self.assertGreater(ts.num_trees, 3)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_single_multiroot_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_many_multiroot_trees_recurrent_mutations(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        self.assertGreater(ts.num_trees, 3)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_single_tree_recurrent_mutations_internal_samples(self):
        ts = msprime.simulate(6, random_seed=10)
        ts = tsutil.jiggle_samples(ts)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_many_trees_recurrent_mutations_internal_samples(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        ts = tsutil.jiggle_samples(ts)
        self.assertGreater(ts.num_trees, 3)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)


class TestMapToAncestors(unittest.TestCase):
    """
    Tests the AncestorMap class.
    """
    random_seed = 13
    #
    #          8
    #         / \
    #        /   \
    #       /     \
    #      7       \
    #     / \       6
    #    /   5     / \
    #   /   / \   /   \
    #  4   0   1 2     3
    nodes = """\
    id      is_sample   population      time
    0       1       0               0.00000000000000
    1       1       0               0.00000000000000
    2       1       0               0.00000000000000
    3       1       0               0.00000000000000
    4       1       0               0.00000000000000
    5       0       0               0.14567111023387
    6       0       0               0.21385545626353
    7       0       0               0.43508024345063
    8       0       0               1.60156352971203
    """
    edges = """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      5       0,1
    1       0.00000000      1.00000000      6       2,3
    2       0.00000000      1.00000000      7       4,5
    3       0.00000000      1.00000000      8       6,7
    """
    #
    #          9                        10
    #         / \                      / \
    #        /   \                    /   8
    #       /     \                  /   / \
    #      7       \                /   /   \
    #     / \       6              /   /     6
    #    /   5     / \            /   5     / \
    #   /   / \   /   \          /   / \   /   \
    #  4   0   1 2     3        4   0   1 2     3
    #
    # 0 ------------------ 0.5 ------------------ 1.0
    nodes0 = """\
    id      is_sample   population      time
    0       1       0               0.00000000000000
    1       1       0               0.00000000000000
    2       1       0               0.00000000000000
    3       1       0               0.00000000000000
    4       1       0               0.00000000000000
    5       0       0               0.14567111023387
    6       0       0               0.21385545626353
    7       0       0               0.43508024345063
    8       0       0               0.60156352971203
    9       0       0               0.90000000000000
    10      0       0               1.20000000000000
    """
    edges0 = """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      5       0,1
    1       0.00000000      1.00000000      6       2,3
    2       0.00000000      0.50000000      7       4,5
    3       0.50000000      1.00000000      8       5,6
    4       0.00000000      0.50000000      9       6,7
    5       0.50000000      1.00000000      10      4,8
    """
    nodes1 = """\
    id      is_sample   population      time
    0       0           0           1.0
    1       1           0           0.0
    2       1           0           0.0
    """
    edges1 = """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      0       1,2
    """

    def do_map(self, ts, ancestors, samples=None, compare_lib=True):
        """
        Runs the Python test implementation of link_ancestors.
        """
        if samples is None:
            samples = ts.samples()
        s = tests.AncestorMap(ts, samples, ancestors)
        ancestor_table = s.link_ancestors()
        if compare_lib:
            lib_result = ts.tables.link_ancestors(samples, ancestors)
            self.assertEqual(ancestor_table, lib_result)
        return ancestor_table

    def test_deprecated_name(self):
        # copied from test_single_tree_one_ancestor below
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        samples = ts.samples()
        ancestors = [8]
        s = tests.AncestorMap(ts, samples, ancestors)
        tss = s.link_ancestors()
        lib_result = ts.tables.map_ancestors(samples, ancestors)
        self.assertEqual(tss, lib_result)
        self.assertEqual(list(tss.parent), [8, 8, 8, 8, 8])
        self.assertEqual(list(tss.child), [0, 1, 2, 3, 4])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_one_ancestor(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[8])
        self.assertEqual(list(tss.parent), [8, 8, 8, 8, 8])
        self.assertEqual(list(tss.child), [0, 1, 2, 3, 4])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_unordered_nodes(self):
        nodes = io.StringIO(self.nodes1)
        edges = io.StringIO(self.edges1)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[0])
        self.assertEqual(list(tss.parent), [0, 0])
        self.assertEqual(list(tss.child), [1, 2])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_two_ancestors(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[6, 7])
        self.assertEqual(list(tss.parent), [6, 6, 7, 7, 7])
        self.assertEqual(list(tss.child), [2, 3, 0, 1, 4])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_no_ancestors(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[2, 3], ancestors=[7])
        self.assertEqual(tss.num_rows, 0)

    def test_single_tree_samples_or_ancestors_not_in_tree(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        with self.assertRaises(AssertionError):
            self.do_map(ts, samples=[-1, 3], ancestors=[5])
        with self.assertRaises(AssertionError):
            self.do_map(ts, samples=[2, 3], ancestors=[10])

    def test_single_tree_ancestors_descend_from_other_ancestors(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[7, 8])
        self.assertEqual(list(tss.parent), [7, 7, 7, 8, 8, 8])
        self.assertEqual(list(tss.child), [0, 1, 4, 2, 3, 7])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_internal_samples(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[2, 3, 4, 5], ancestors=[7, 8])
        self.assertEqual(list(tss.parent), [7, 7, 8, 8, 8])
        self.assertEqual(list(tss.child), [4, 5, 2, 3, 7])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_samples_and_ancestors_overlap(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[1, 2, 3, 5], ancestors=[5, 6, 7])
        self.assertEqual(list(tss.parent), [5, 6, 6, 7])
        self.assertEqual(list(tss.child), [1, 2, 3, 5])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_unary_ancestor(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[1, 2, 4], ancestors=[5, 7, 8])
        self.assertEqual(list(tss.parent), [5, 7, 7, 8, 8])
        self.assertEqual(list(tss.child), [1, 4, 5, 2, 7])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_ancestors_descend_from_samples(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[1, 7], ancestors=[5, 8])
        self.assertEqual(list(tss.parent), [5, 7, 8])
        self.assertEqual(list(tss.child), [1, 5, 7])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_single_tree_samples_descend_from_samples(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[3, 6], ancestors=[8])
        self.assertEqual(list(tss.parent), [6, 8])
        self.assertEqual(list(tss.child), [3, 6])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_multiple_trees_to_single_tree(self):
        nodes = io.StringIO(self.nodes0)
        edges = io.StringIO(self.edges0)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[5, 6])
        self.assertEqual(list(tss.parent), [5, 5, 6, 6])
        self.assertEqual(list(tss.child), [0, 1, 2, 3])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def test_multiple_trees_one_ancestor(self):
        nodes = io.StringIO(self.nodes0)
        edges = io.StringIO(self.edges0)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[9, 10])
        self.assertEqual(list(tss.parent), [9, 9, 9, 9, 9, 10, 10, 10, 10, 10])
        self.assertEqual(list(tss.child), [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        self.assertEqual(all(tss.left), 0)
        self.assertEqual(all(tss.right), 1)

    def verify(self, ts, sample_nodes, ancestral_nodes):
        tss = self.do_map(ts, ancestors=ancestral_nodes, samples=sample_nodes)
        # ancestors = list(set(tss.parent))
        # Loop through the rows of the ancestral branch table.
        current_ancestor = tss.parent[0]
        current_descendants = [tss.child[0]]
        current_left = tss.left[0]
        current_right = tss.right[0]
        for ind, row in enumerate(tss):
            if row.parent != current_ancestor or\
                    row.left != current_left or\
                    row.right != current_right:
                # Loop through trees.
                for tree in ts.trees():
                    if tree.interval[0] >= current_right:
                        break
                    while tree.interval[1] <= current_left:
                        tree.next()
                    # Check that the most recent ancestor of the descendants is the
                    # current_ancestor.
                    current_descendants = list(set(current_descendants))
                    for des in current_descendants:
                        par = tree.get_parent(des)
                        while par not in ancestral_nodes and par not in sample_nodes:
                            par = tree.get_parent(par)
                        self.assertEqual(par, current_ancestor)
                # Reset the current ancestor and descendants, left and right coords.
                current_ancestor = row.parent
                current_descendants = [row.child]
                current_left = row.left
                current_right = row.right
            else:
                # Collate a list of children corresponding to each ancestral node.
                current_descendants.append(row.child)

    def test_sim_single_coalescent_tree(self):
        ts = msprime.simulate(30, random_seed=1, length=10)
        ancestors = [3*n for n in np.arange(0, ts.num_nodes // 3)]
        self.verify(ts, ts.samples(), ancestors)
        random_samples = [4*n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, random_samples, ancestors)

    def test_sim_coalescent_trees(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=1, length=2)
        ancestors = [3*n for n in np.arange(0, ts.num_nodes // 3)]
        self.verify(ts, ts.samples(), ancestors)
        random_samples = [4*n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, random_samples, ancestors)

    def test_sim_coalescent_trees_internal_samples(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=10, length=2)
        self.assertGreater(ts.num_trees, 2)
        ancestors = [4*n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(tsutil.jiggle_samples(ts), ts.samples(), ancestors)
        random_samples = [4*n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(tsutil.jiggle_samples(ts), random_samples, ancestors)

    def test_sim_many_multiroot_trees(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        self.assertGreater(ts.num_trees, 3)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        ancestors = [4*n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, ts.samples(), ancestors)
        random_samples = [4*n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, random_samples, ancestors)

    def test_sim_wright_fisher_generations(self):
        number_of_gens = 5
        tables = wf.wf_sim(10, number_of_gens, deep_history=False, seed=2)
        tables.sort()
        ts = tables.tree_sequence()
        ancestors = [4*n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, ts.samples(), ancestors)
        for gen in range(1, number_of_gens):
            ancestors = [u.id for u in ts.nodes() if u.time == gen]
            self.verify(ts, ts.samples(), ancestors)

        random_samples = [4*n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, random_samples, ancestors)
        for gen in range(1, number_of_gens):
            ancestors = [u.id for u in ts.nodes() if u.time == gen]
            self.verify(ts, random_samples, ancestors)


class TestMutationParent(unittest.TestCase):
    """
    Tests that mutation parent is correctly specified, and that we correctly
    recompute it with compute_mutation_parent.
    """
    seed = 42

    def verify_parents(self, ts):
        parent = tsutil.compute_mutation_parent(ts)
        tables = ts.tables
        self.assertTrue(np.array_equal(parent, tables.mutations.parent))
        tables.mutations.parent = np.zeros_like(tables.mutations.parent) - 1
        self.assertTrue(np.all(tables.mutations.parent == tskit.NULL))
        tables.compute_mutation_parents()
        self.assertTrue(np.array_equal(parent, tables.mutations.parent))

    def test_example(self):
        nodes = io.StringIO("""\
        id      is_sample   time
        0       0           2.0
        1       0           1.0
        2       0           1.0
        3       1           0
        4       1           0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0    0.5   2  3
        0.0    0.8   2  4
        0.5    1.0   1  3
        0.0    1.0   0  1
        0.0    1.0   0  2
        0.8    1.0   0  4
        """)
        sites = io.StringIO("""\
        position    ancestral_state
        0.1     0
        0.5     0
        0.9     0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       1       1               -1
        0       2       1               -1
        0       3       2               1
        1       0       1               -1
        1       1       1               3
        1       3       2               4
        1       2       1               3
        1       4       2               6
        2       0       1               -1
        2       1       1               8
        2       2       1               8
        2       4       1               8
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        self.verify_parents(ts)

    def test_single_muts(self):
        ts = msprime.simulate(10, random_seed=self.seed, mutation_rate=3.0,
                              recombination_rate=1.0)
        self.verify_parents(ts)

    def test_with_jukes_cantor(self):
        ts = msprime.simulate(10, random_seed=self.seed, mutation_rate=0.0,
                              recombination_rate=1.0)
        # make *lots* of recurrent mutations
        mut_ts = tsutil.jukes_cantor(ts, num_sites=10, mu=1,
                                     multiple_per_node=False, seed=self.seed)
        self.verify_parents(mut_ts)

    def test_with_jukes_cantor_multiple_per_node(self):
        ts = msprime.simulate(10, random_seed=self.seed, mutation_rate=0.0,
                              recombination_rate=1.0)
        # make *lots* of recurrent mutations
        mut_ts = tsutil.jukes_cantor(ts, num_sites=10, mu=1,
                                     multiple_per_node=True, seed=self.seed)
        self.verify_parents(mut_ts)

    def verify_branch_mutations(self, ts, mutations_per_branch):
        ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
        self.assertGreater(ts.num_mutations, 1)
        self.verify_parents(ts)

    def test_single_tree_one_mutation_per_branch(self):
        ts = msprime.simulate(6, random_seed=10)
        self.verify_branch_mutations(ts, 1)

    def test_single_tree_two_mutations_per_branch(self):
        ts = msprime.simulate(10, random_seed=9)
        self.verify_branch_mutations(ts, 2)

    def test_single_tree_three_mutations_per_branch(self):
        ts = msprime.simulate(8, random_seed=9)
        self.verify_branch_mutations(ts, 3)

    def test_single_multiroot_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        for mutations_per_branch in [1, 2, 3]:
            self.verify_branch_mutations(ts, mutations_per_branch)

    def test_many_multiroot_trees_recurrent_mutations(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        self.assertGreater(ts.num_trees, 3)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        for mutations_per_branch in [1, 2, 3]:
            self.verify_branch_mutations(ts, mutations_per_branch)


class TestSimpleTreeAlgorithm(unittest.TestCase):
    """
    Tests for the direct implementation of Algorithm T in tsutil.py.

    See TestHoleyTreeSequences above for further tests on wacky topologies.
    """
    def test_zero_nodes(self):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        self.assertEqual(ts.sequence_length, 1)
        self.assertEqual(ts.num_trees, 1)
        # Test the simple tree iterator.
        trees = list(tsutil.algorithm_T(ts))
        self.assertEqual(len(trees), 1)
        (left, right), parent = trees[0]
        self.assertEqual(left, 0)
        self.assertEqual(right, 1)
        self.assertEqual(parent, [])

    def test_one_node(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row()
        ts = tables.tree_sequence()
        self.assertEqual(ts.sequence_length, 1)
        self.assertEqual(ts.num_trees, 1)
        # Test the simple tree iterator.
        trees = list(tsutil.algorithm_T(ts))
        self.assertEqual(len(trees), 1)
        (left, right), parent = trees[0]
        self.assertEqual(left, 0)
        self.assertEqual(right, 1)
        self.assertEqual(parent, [-1])

    def test_single_coalescent_tree(self):
        ts = msprime.simulate(10, random_seed=1, length=10)
        tree = ts.first()
        p1 = [tree.parent(j) for j in range(ts.num_nodes)]
        interval, p2 = next(tsutil.algorithm_T(ts))
        self.assertEqual(interval, tree.interval)
        self.assertEqual(p1, p2)

    def test_coalescent_trees(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=1, length=2)
        self.assertGreater(ts.num_trees, 2)
        new_trees = tsutil.algorithm_T(ts)
        for tree in ts.trees():
            interval, p2 = next(new_trees)
            p1 = [tree.parent(j) for j in range(ts.num_nodes)]
            self.assertEqual(interval, tree.interval)
            self.assertEqual(p1, p2)
        self.assertRaises(StopIteration, next, new_trees)


class TestSampleLists(unittest.TestCase):
    """
    Tests for the sample lists algorithm.
    """
    def verify(self, ts):
        tree1 = tsutil.LinkedTree(ts)
        s = str(tree1)
        self.assertIsNotNone(s)
        trees = ts.trees(sample_lists=True)
        for left, right in tree1.sample_lists():
            tree2 = next(trees)
            assert (left, right) == tree2.interval
            for u in tree2.nodes():
                self.assertEqual(tree1.left_sample[u], tree2.left_sample(u))
                self.assertEqual(tree1.right_sample[u], tree2.right_sample(u))
            for j in range(ts.num_samples):
                self.assertEqual(tree1.next_sample[j], tree2.next_sample(j))
        assert right == ts.sequence_length

        tree1 = tsutil.LinkedTree(ts)
        trees = ts.trees(sample_lists=False)
        sample_index_map = ts.samples()
        for left, right in tree1.sample_lists():
            tree2 = next(trees)
            for u in range(ts.num_nodes):
                samples2 = list(tree2.samples(u))
                samples1 = []
                index = tree1.left_sample[u]
                if index != tskit.NULL:
                    self.assertEqual(
                        sample_index_map[tree1.left_sample[u]], samples2[0])
                    self.assertEqual(
                        sample_index_map[tree1.right_sample[u]], samples2[-1])
                    stop = tree1.right_sample[u]
                    while True:
                        assert index != -1
                        samples1.append(sample_index_map[index])
                        if index == stop:
                            break
                        index = tree1.next_sample[index]
                self.assertEqual(samples1, samples2)
        assert right == ts.sequence_length

    def test_single_coalescent_tree(self):
        ts = msprime.simulate(10, random_seed=1, length=10)
        self.verify(ts)

    def test_coalescent_trees(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=1, length=2)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts)

    def test_coalescent_trees_internal_samples(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=10, length=2)
        self.assertGreater(ts.num_trees, 2)
        self.verify(tsutil.jiggle_samples(ts))

    def test_coalescent_trees_all_samples(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=10, length=2)
        self.assertGreater(ts.num_trees, 2)
        tables = ts.dump_tables()
        flags = np.zeros_like(tables.nodes.flags) + tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        self.verify(tables.tree_sequence())

    def test_wright_fisher_trees_unsimplified(self):
        tables = wf.wf_sim(10, 5, deep_history=False, seed=2)
        tables.sort()
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_wright_fisher_trees_simplified(self):
        tables = wf.wf_sim(10, 5, deep_history=False, seed=1)
        tables.sort()
        ts = tables.tree_sequence()
        ts = ts.simplify()
        self.verify(ts)

    def test_wright_fisher_trees_simplified_one_gen(self):
        tables = wf.wf_sim(10, 1, deep_history=False, seed=1)
        tables.sort()
        ts = tables.tree_sequence()
        ts = ts.simplify()
        self.verify(ts)

    def test_nonbinary_trees(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=1.0, population=0, proportion=0.95)]
        ts = msprime.simulate(
            20, recombination_rate=10, mutation_rate=5,
            demographic_events=demographic_events, random_seed=7)
        found = False
        for e in ts.edgesets():
            if len(e.children) > 2:
                found = True
        self.assertTrue(found)
        self.verify(ts)

    def test_many_multiroot_trees(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        self.assertGreater(ts.num_trees, 3)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        self.verify(ts)


class TestSquashEdges(unittest.TestCase):
    """
    Tests of the squash_edges function.
    """
    def do_squash(self, ts, compare_lib=True):
        squashed = ts.tables.edges
        squashed.squash()
        if compare_lib:
            squashed_list = squash_edges(ts)
            squashed_py = tskit.EdgeTable()
            for e in squashed_list:
                squashed_py.add_row(e.left, e.right, e.parent, e.child)
            # Check the Python and C implementations produce the same output.
            self.assertEqual(squashed_py, squashed)
        return squashed

    def test_simple_case(self):
        #   2
        #  / \
        # 0   1
        nodes = io.StringIO("""\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       0       0               1.00000000000000
        """)
        edges = io.StringIO("""\
        id      left            right           parent  child
        0       0.00000000      0.50000000      2       0
        1       0.00000000      0.50000000      2       1
        2       0.50000000      1.00000000      2       0
        3       0.50000000      1.00000000      2       1
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        self.assertEqual(all(edges.left), 0)
        self.assertEqual(all(edges.right), 1)
        self.assertEqual(list(edges.parent), [2, 2])
        self.assertEqual(list(edges.child), [0, 1])

    def test_simple_case_unordered_intervals(self):
        # 1
        # |
        # 0
        nodes = io.StringIO("""\
        id      is_sample   population      time
        0       1           0               0.0
        1       0           0               1.0
        """)
        edges = io.StringIO("""\
        id      left            right           parent  child
        0       0.40            1.0             1       0
        0       0.00            0.40            1       0
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        self.assertEqual(edges.left[0], 0)
        self.assertEqual(edges.right[0], 1)
        self.assertEqual(edges.parent[0], 1)
        self.assertEqual(edges.child[0], 0)

    def test_simple_case_unordered_children(self):
        #   2
        #  / \
        # 0   1
        nodes = io.StringIO("""\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       0       0               1.00000000000000
        """)
        edges = io.StringIO("""\
        id      left            right           parent  child
        0       0.50000000      1.00000000      2       1
        1       0.50000000      1.00000000      2       0
        2       0.00000000      0.50000000      2       1
        3       0.00000000      0.50000000      2       0
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        self.assertEqual(all(edges.left), 0)
        self.assertEqual(all(edges.right), 1)
        self.assertEqual(list(edges.parent), [2, 2])
        self.assertEqual(list(edges.child), [0, 1])

    def test_simple_case_unordered_children_and_intervals(self):
        #   2
        #  / \
        # 0   1
        nodes = io.StringIO("""\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       0       0               1.00000000000000
        """)
        edges = io.StringIO("""\
        id      left            right           parent  child
        0       0.50000000      1.00000000      2       1
        2       0.00000000      0.50000000      2       1
        3       0.00000000      0.50000000      2       0
        1       0.50000000      1.00000000      2       0
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        self.assertEqual(all(edges.left), 0)
        self.assertEqual(all(edges.right), 1)
        self.assertEqual(list(edges.parent), [2, 2])
        self.assertEqual(list(edges.child), [0, 1])

    def test_squash_multiple_parents_and_children(self):
        #   4       5
        #  / \     / \
        # 0   1   2   3
        nodes = io.StringIO("""\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       1       0               0.00000000000000
        3       1       0               0.00000000000000
        4       0       0               1.00000000000000
        5       0       0               1.00000000000000
        """)
        edges = io.StringIO("""\
        id      left            right           parent  child
        5       0.50000000      1.00000000      5       3
        6       0.50000000      1.00000000      5       2
        7       0.00000000      0.50000000      5       3
        8       0.00000000      0.50000000      5       2
        9       0.40000000      1.00000000      4       1
        10      0.00000000      0.40000000      4       1
        11      0.40000000      1.00000000      4       0
        12      0.00000000      0.40000000      4       0
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        self.assertEqual(all(edges.left), 0)
        self.assertEqual(all(edges.right), 1)
        self.assertEqual(list(edges.parent), [4, 4, 5, 5])
        self.assertEqual(list(edges.child), [0, 1, 2, 3])

    def test_squash_overlapping_intervals(self):
        nodes = io.StringIO("""\
        id      is_sample   population      time
        0       1           0               0.0
        1       0           0               1.0
        """)
        edges = io.StringIO("""\
        id      left            right           parent  child
        0       0.00            0.50            1       0
        1       0.40            0.80            1       0
        2       0.60            1.00            1       0
        """)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

        with self.assertRaises(tskit.LibraryError):
            self.do_squash(ts)

    def verify_slice_and_squash(self, ts):
        """
        Slices a tree sequence so that there are edge endpoints at
        all integer locations, then squashes these edges and verifies
        that the resulting edge table is the same as the input edge table.
        """
        sliced_edges = []
        # Create new sliced edge table.
        for e in ts.edges():
            left = e.left
            right = e.right

            if left == np.floor(left):
                r_left = np.ceil(left) + 1
            else:
                r_left = np.ceil(left)
            if right == np.floor(right):
                r_right = np.floor(right)
            else:
                r_right = np.floor(right) + 1

            new_range = [left]
            for r in np.arange(r_left, r_right):
                new_range.append((r))
            new_range.append(right)
            assert len(new_range) > 1

            # Add new edges to the list.
            for r in range(1, len(new_range)):
                new = tskit.Edge(new_range[r-1], new_range[r], e.parent, e.child)
                sliced_edges.append(new)

        # Shuffle the edges and create a new edge table.
        random.shuffle(sliced_edges)
        sliced_table = tskit.EdgeTable()
        for e in sliced_edges:
            sliced_table.add_row(e.left, e.right, e.parent, e.child)

        # Squash the edges and check against input table.
        sliced_table.squash()
        self.assertEqual(sliced_table, ts.tables.edges)

    def test_sim_single_coalescent_tree(self):
        ts = msprime.simulate(20, random_seed=4, length=10)
        self.assertEqual(ts.num_trees, 1)
        self.verify_slice_and_squash(ts)

    def test_sim_big_coalescent_trees(self):
        ts = msprime.simulate(20, recombination_rate=5, random_seed=4, length=10)
        self.assertGreater(ts.num_trees, 2)
        self.verify_slice_and_squash(ts)


def squash_edges(ts):
    """
    Returns the edges in the tree sequence squashed.
    """
    t = ts.tables.nodes.time
    edges = list(ts.edges())
    edges.sort(key=lambda e: (t[e.parent], e.parent, e.child, e.left))
    if len(edges) == 0:
        return []

    squashed = []
    last_e = edges[0]
    for e in edges[1:]:
        condition = (
            e.parent != last_e.parent or
            e.child != last_e.child or
            e.left != last_e.right)
        if condition:
            squashed.append(last_e)
            last_e = e
        last_e.right = e.right
    squashed.append(last_e)
    return squashed


def reduce_topology(ts):
    """
    Returns a tree sequence with the minimal information required to represent
    the tree topologies at its sites. Uses a left-to-right algorithm.
    """
    tables = ts.dump_tables()
    edge_map = {}

    def add_edge(left, right, parent, child):
        new_edge = tskit.Edge(left, right, parent, child)
        if child not in edge_map:
            edge_map[child] = new_edge
        else:
            edge = edge_map[child]
            if edge.right == left and edge.parent == parent:
                # Squash
                edge.right = right
            else:
                tables.edges.add_row(edge.left, edge.right, edge.parent, edge.child)
                edge_map[child] = new_edge

    tables.edges.clear()

    edge_buffer = []
    first_site = True
    for tree in ts.trees():
        # print(tree.interval)
        # print(tree.draw(format="unicode"))
        if tree.num_sites > 0:
            sites = list(tree.sites())
            if first_site:
                x = 0
                # print("First site", sites)
                first_site = False
            else:
                x = sites[0].position
            # Flush the edge buffer.
            for left, parent, child in edge_buffer:
                add_edge(left, x, parent, child)
            # Add edges for each node in the tree.
            edge_buffer = []
            for root in tree.roots:
                for u in tree.nodes(root):
                    if u != root:
                        edge_buffer.append((x, tree.parent(u), u))
    # Add the final edges.
    for left, parent, child in edge_buffer:
        add_edge(left, tables.sequence_length, parent, child)
    # Flush the remaining edges to the table
    for edge in edge_map.values():
        tables.edges.add_row(edge.left, edge.right, edge.parent, edge.child)
    tables.sort()
    ts = tables.tree_sequence()
    # Now simplify to remove redundant nodes.
    return ts.simplify(map_nodes=True, filter_sites=False)


class TestReduceTopology(unittest.TestCase):
    """
    Tests to ensure that reduce topology in simplify is equivalent to the
    reduce_topology function above.
    """

    def verify(self, ts):
        source_tables = ts.tables
        X = source_tables.sites.position
        position_count = {x: 0 for x in X}
        position_count[0] = 0
        position_count[ts.sequence_length] = 0
        mts, node_map = reduce_topology(ts)
        for edge in mts.edges():
            self.assertIn(edge.left, position_count)
            self.assertIn(edge.right, position_count)
            position_count[edge.left] += 1
            position_count[edge.right] += 1
        if ts.num_sites == 0:
            # We should have zero edges output.
            self.assertEqual(mts.num_edges, 0)
        elif X[0] != 0:
            # The first site (if it's not zero) should be mapped to zero so
            # this never occurs in edges.
            self.assertEqual(position_count[X[0]], 0)

        minimised_trees = mts.trees()
        minimised_tree = next(minimised_trees)
        minimised_tree_sites = minimised_tree.sites()
        for tree in ts.trees():
            for site in tree.sites():
                minimised_site = next(minimised_tree_sites, None)
                if minimised_site is None:
                    minimised_tree = next(minimised_trees)
                    minimised_tree_sites = minimised_tree.sites()
                    minimised_site = next(minimised_tree_sites)
                self.assertEqual(site.position, minimised_site.position)
                self.assertEqual(site.ancestral_state, minimised_site.ancestral_state)
                self.assertEqual(site.metadata, minimised_site.metadata)
                self.assertEqual(len(site.mutations), len(minimised_site.mutations))

                for mutation, minimised_mutation in zip(
                        site.mutations, minimised_site.mutations):
                    self.assertEqual(
                        mutation.derived_state, minimised_mutation.derived_state)
                    self.assertEqual(mutation.metadata, minimised_mutation.metadata)
                    self.assertEqual(mutation.parent, minimised_mutation.parent)
                    self.assertEqual(node_map[mutation.node], minimised_mutation.node)
            if tree.num_sites > 0:
                mapped_dict = {
                    node_map[u]: node_map[v] for u, v in tree.parent_dict.items()}
                self.assertEqual(mapped_dict, minimised_tree.parent_dict)
        self.assertTrue(np.array_equal(ts.genotype_matrix(), mts.genotype_matrix()))

        edges = list(mts.edges())
        squashed = squash_edges(mts)
        self.assertEqual(len(edges), len(squashed))
        self.assertEqual(edges, squashed)

        # Verify against simplify implementations.
        s = tests.Simplifier(
            ts, ts.samples(), reduce_to_site_topology=True, filter_sites=False)
        sts1, _ = s.simplify()
        sts2 = ts.simplify(reduce_to_site_topology=True, filter_sites=False)
        t1 = mts.tables
        for sts in [sts2, sts2]:
            t2 = sts.tables
            self.assertEqual(t1.nodes,  t2.nodes)
            self.assertEqual(t1.edges, t2.edges)
            self.assertEqual(t1.sites, t2.sites)
            self.assertEqual(t1.mutations, t2.mutations)
            self.assertEqual(t1.populations, t2.populations)
            self.assertEqual(t1.individuals, t2.individuals)
        return mts

    def test_no_recombination_one_site(self):
        ts = msprime.simulate(15, random_seed=1)
        tables = ts.dump_tables()
        tables.sites.add_row(position=0.25, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        self.assertEqual(mts.num_trees, 1)

    def test_simple_recombination_one_site(self):
        ts = msprime.simulate(15, random_seed=1, recombination_rate=2)
        tables = ts.dump_tables()
        tables.sites.add_row(position=0.25, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        self.assertEqual(mts.num_trees, 1)

    def test_simple_recombination_fixed_sites(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2)
        tables = ts.dump_tables()
        for x in [0.25, 0.5, 0.75]:
            tables.sites.add_row(position=x, ancestral_state="0")
        self.verify(tables.tree_sequence())

    def get_integer_edge_ts(self, n, m):
        recombination_map = msprime.RecombinationMap.uniform_map(m, 1, num_loci=m)
        ts = msprime.simulate(n, random_seed=1, recombination_map=recombination_map)
        self.assertGreater(ts.num_trees, 1)
        for edge in ts.edges():
            self.assertEqual(int(edge.left), edge.left)
            self.assertEqual(int(edge.right), edge.right)
        return ts

    def test_integer_edges_one_site(self):
        ts = self.get_integer_edge_ts(5, 10)
        tables = ts.dump_tables()
        tables.sites.add_row(position=1, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        self.assertEqual(mts.num_trees, 1)

    def test_integer_edges_all_sites(self):
        ts = self.get_integer_edge_ts(5, 10)
        tables = ts.dump_tables()
        for x in range(10):
            tables.sites.add_row(position=x, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        self.assertEqual(mts.num_trees, ts.num_trees)

    def test_simple_recombination_site_at_zero(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2)
        tables = ts.dump_tables()
        tables.sites.add_row(position=0, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        self.assertEqual(mts.num_trees, 1)

    def test_simple_recombination(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        self.verify(ts)

    def test_large_recombination(self):
        ts = msprime.simulate(25, random_seed=12, recombination_rate=5, mutation_rate=15)
        self.verify(ts)

    def test_no_recombination(self):
        ts = msprime.simulate(5, random_seed=1, mutation_rate=2)
        self.verify(ts)

    def test_no_mutation(self):
        ts = msprime.simulate(5, random_seed=1)
        self.verify(ts)

    def test_zero_sites(self):
        ts = msprime.simulate(5, random_seed=2)
        self.assertEqual(ts.num_sites, 0)
        mts = ts.simplify(reduce_to_site_topology=True)
        self.assertEqual(mts.num_trees, 1)
        self.assertEqual(mts.num_edges, 0)

    def test_many_roots(self):
        ts = msprime.simulate(25, random_seed=12, recombination_rate=2, length=10)
        tables = tsutil.decapitate(ts, ts.num_edges // 2).dump_tables()
        for x in range(10):
            tables.sites.add_row(x, "0")
        self.verify(tables.tree_sequence())

    def test_branch_sites(self):
        ts = msprime.simulate(15, random_seed=12, recombination_rate=2, length=10)
        ts = tsutil.insert_branch_sites(ts)
        self.verify(ts)

    def test_jiggled_samples(self):
        ts = msprime.simulate(8, random_seed=13, recombination_rate=2, length=10)
        ts = tsutil.jiggle_samples(ts)
        self.verify(ts)


def search_sorted(a, v):
    """
    Implementation of searchsorted based on binary search with the same
    semantics as numpy's searchsorted. Used as the basis of the C
    implementation which we use in the simplify algorithm.
    """
    upper = len(a)
    if upper == 0:
        return 0
    lower = 0
    while upper - lower > 1:
        mid = (upper + lower) // 2
        if (v >= a[mid]):
            lower = mid
        else:
            upper = mid
    offset = 0
    if a[lower] < v:
        offset = 1
    return lower + offset


class TestSearchSorted(unittest.TestCase):
    """
    Tests for the basic implementation of search_sorted.
    """
    def verify(self, a):
        a = np.array(a)
        start, end = a[0], a[-1]
        # Check random values.
        np.random.seed(43)
        for v in np.random.uniform(start, end, 10):
            self.assertEqual(search_sorted(a, v), np.searchsorted(a, v))
        # Check equal values.
        for v in a:
            self.assertEqual(search_sorted(a, v), np.searchsorted(a, v))
        # Check values outside bounds.
        for v in [start - 2, start - 1, end, end + 1, end + 2]:
            self.assertEqual(search_sorted(a, v), np.searchsorted(a, v))

    def test_range(self):
        for j in range(1, 20):
            self.verify(range(j))

    def test_negative_range(self):
        for j in range(1, 20):
            self.verify(-1 * np.arange(j)[::-1])

    def test_random_unit_interval(self):
        np.random.seed(143)
        for size in range(1, 100):
            a = np.random.random(size=size)
            a.sort()
            self.verify(a)

    def test_random_interval(self):
        np.random.seed(143)
        for _ in range(10):
            interval = np.random.random(2) * 10
            interval.sort()
            a = np.random.uniform(*interval, size=100)
            a.sort()
            self.verify(a)

    def test_random_negative(self):
        np.random.seed(143)
        for _ in range(10):
            interval = np.random.random(2) * 5
            interval.sort()
            a = -1 * np.random.uniform(*interval, size=100)
            a.sort()
            self.verify(a)

    def test_edge_cases(self):
        for v in [0, 1]:
            self.assertEqual(search_sorted([], v), np.searchsorted([], v))
            self.assertEqual(search_sorted([1], v), np.searchsorted([1], v))


class TestDeleteSites(unittest.TestCase):
    """
    Tests for the TreeSequence.delete_sites method
    """
    def ts_with_4_sites(self):
        ts = msprime.simulate(8, random_seed=3)
        tables = ts.dump_tables()
        tables.sites.set_columns(np.arange(0, 1, 0.25), *tskit.pack_strings(['G'] * 4))
        tables.mutations.add_row(site=1, node=ts.first().parent(0), derived_state='C')
        tables.mutations.add_row(site=1, node=0, derived_state='T', parent=0)
        tables.mutations.add_row(site=2, node=1, derived_state='A')
        return tables.tree_sequence()

    def test_remove_by_index(self):
        ts = self.ts_with_4_sites().delete_sites([])
        self.assertEquals(ts.num_sites, 4)
        self.assertEquals(ts.num_mutations, 3)
        ts = ts.delete_sites(2)
        self.assertEquals(ts.num_sites, 3)
        self.assertEquals(ts.num_mutations, 2)
        ts = ts.delete_sites([1, 2])
        self.assertEquals(ts.num_sites, 1)
        self.assertEquals(ts.num_mutations, 0)

    def test_remove_all(self):
        ts = self.ts_with_4_sites().delete_sites(range(4))
        self.assertEquals(ts.num_sites, 0)
        self.assertEquals(ts.num_mutations, 0)
        # should be OK to run on a siteless tree seq as no sites specified
        ts.delete_sites([])

    def test_remove_repeated_sites(self):
        ts = self.ts_with_4_sites()
        t1 = ts.delete_sites([0, 1], record_provenance=False)
        t2 = ts.delete_sites([0, 0, 1], record_provenance=False)
        t3 = ts.delete_sites([0, 0, 0, 1], record_provenance=False)
        self.assertEquals(t1.tables, t2.tables)
        self.assertEquals(t1.tables, t3.tables)

    def test_remove_different_orders(self):
        ts = self.ts_with_4_sites()
        t1 = ts.delete_sites([0, 1, 3], record_provenance=False)
        t2 = ts.delete_sites([0, 3, 1], record_provenance=False)
        t3 = ts.delete_sites([3, 0, 1], record_provenance=False)
        self.assertEquals(t1.tables, t2.tables)
        self.assertEquals(t1.tables, t3.tables)

    def test_remove_bad(self):
        ts = self.ts_with_4_sites()
        self.assertRaises(TypeError, ts.delete_sites, ["1"])
        self.assertRaises(ValueError, ts.delete_sites, 4)
        self.assertRaises(ValueError, ts.delete_sites, -5)

    def verify_removal(self, ts, remove_sites):
        tables = ts.dump_tables()
        tables.delete_sites(remove_sites)

        # Make sure we've computed the mutation parents properly.
        mutation_parent = tables.mutations.parent
        tables.compute_mutation_parents()
        self.assertTrue(np.array_equal(mutation_parent, tables.mutations.parent))

        tsd = tables.tree_sequence()
        self.assertEqual(tsd.num_sites, ts.num_sites - len(remove_sites))
        source_sites = [site for site in ts.sites() if site.id not in remove_sites]
        self.assertEqual(len(source_sites), tsd.num_sites)
        for s1, s2 in zip(source_sites, tsd.sites()):
            self.assertEqual(s1.position, s2.position)
            self.assertEqual(s1.ancestral_state, s2.ancestral_state)
            self.assertEqual(s1.metadata, s2.metadata)
            self.assertEqual(len(s1.mutations), len(s2.mutations))
            for m1, m2 in zip(s1.mutations, s2.mutations):
                self.assertEqual(m1.node, m2.node)
                self.assertEqual(m1.derived_state, m2.derived_state)
                self.assertEqual(m1.metadata, m2.metadata)

        # Check we get the same genotype_matrix
        G1 = ts.genotype_matrix()
        G2 = tsd.genotype_matrix()
        keep = np.ones(ts.num_sites, dtype=bool)
        keep[remove_sites] = 0
        self.assertTrue(np.array_equal(G1[keep], G2))

    def test_simple_random_metadata(self):
        ts = msprime.simulate(10, mutation_rate=10, random_seed=2)
        ts = tsutil.add_random_metadata(ts)
        self.assertGreater(ts.num_mutations, 5)
        self.verify_removal(ts, [1, 3])

    def test_simple_mixed_length_states(self):
        ts = msprime.simulate(10, random_seed=2, length=10)
        tables = ts.dump_tables()
        for j in range(10):
            tables.sites.add_row(j, "X" * j)
            tables.mutations.add_row(site=j, node=j, derived_state="X" * (j + 1))
        ts = tables.tree_sequence()
        self.verify_removal(ts, [9])

    def test_jukes_cantor_random_metadata(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 10, 1, seed=2)
        ts = tsutil.add_random_metadata(ts)
        self.assertGreater(ts.num_mutations, 10)
        self.verify_removal(ts, [])
        self.verify_removal(ts, [0, 2, 4, 8])
        self.verify_removal(ts, range(5))

    def test_jukes_cantor_many_mutations(self):
        ts = msprime.simulate(2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 10, mu=10, seed=2)
        self.assertGreater(ts.num_mutations, 100)
        self.verify_removal(ts, [1, 3, 5, 7])
        self.verify_removal(ts, [1])
        self.verify_removal(ts, [9])

    def test_jukes_cantor_one_site(self):
        ts = msprime.simulate(5, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 1, mu=10, seed=2)
        self.assertGreater(ts.num_mutations, 10)
        self.verify_removal(ts, [])
        self.verify_removal(ts, [0])


class TestKeepSingleInterval(unittest.TestCase):
    """
    Tests for cutting up tree sequences along the genome.
    """
    def test_slice_by_tree_positions(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        breakpoints = list(ts.breakpoints())

        # Keep the last 3 trees (from 4th last breakpoint onwards)
        ts_sliced = ts.keep_intervals([[breakpoints[-4], ts.sequence_length]])
        self.assertEqual(ts_sliced.num_trees, 4)
        self.assertLess(ts_sliced.num_edges, ts.num_edges)
        self.assertAlmostEqual(ts_sliced.sequence_length, 1.0)
        last_3_mutations = 0
        for tree_index in range(-3, 0):
            last_3_mutations += ts.at_index(tree_index).num_mutations
        self.assertEqual(ts_sliced.num_mutations, last_3_mutations)

        # Keep the first 3 trees
        ts_sliced = ts.keep_intervals([[0, breakpoints[3]]])
        self.assertEqual(ts_sliced.num_trees, 4)
        self.assertLess(ts_sliced.num_edges, ts.num_edges)
        self.assertAlmostEqual(ts_sliced.sequence_length, 1)
        first_3_mutations = 0
        for tree_index in range(0, 3):
            first_3_mutations += ts.at_index(tree_index).num_mutations
        self.assertEqual(ts_sliced.num_mutations, first_3_mutations)

        # Slice out the middle
        ts_sliced = ts.keep_intervals([[breakpoints[3], breakpoints[-4]]])
        self.assertEqual(ts_sliced.num_trees, ts.num_trees - 4)
        self.assertLess(ts_sliced.num_edges, ts.num_edges)
        self.assertAlmostEqual(ts_sliced.sequence_length, 1.0)
        self.assertEqual(
            ts_sliced.num_mutations,
            ts.num_mutations - first_3_mutations - last_3_mutations)

    def test_slice_by_position(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        ts_sliced = ts.keep_intervals([[0.4, 0.6]])
        positions = ts.tables.sites.position
        self.assertEqual(
            ts_sliced.num_sites, np.sum((positions >= 0.4) & (positions < 0.6)))

    def test_slice_unsimplified(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        ts_sliced = ts.keep_intervals([[0.4, 0.6]], simplify=True)
        self.assertNotEqual(ts.num_nodes, ts_sliced.num_nodes)
        self.assertAlmostEqual(ts_sliced.sequence_length, 1.0)
        ts_sliced = ts.keep_intervals([[0.4, 0.6]], simplify=False)
        self.assertEqual(ts.num_nodes, ts_sliced.num_nodes)
        self.assertAlmostEqual(ts_sliced.sequence_length, 1.0)

    def test_slice_coordinates(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        ts_sliced = ts.keep_intervals([[0.4, 0.6]])
        self.assertAlmostEqual(ts_sliced.sequence_length, 1)
        self.assertNotEqual(ts_sliced.num_trees, ts.num_trees)
        self.assertEqual(ts_sliced.at_index(0).total_branch_length, 0)
        self.assertEqual(ts_sliced.at(0).total_branch_length, 0)
        self.assertEqual(ts_sliced.at(0.399).total_branch_length, 0)
        self.assertNotEqual(ts_sliced.at(0.4).total_branch_length, 0)
        self.assertNotEqual(ts_sliced.at(0.5).total_branch_length, 0)
        self.assertNotEqual(ts_sliced.at(0.599).total_branch_length, 0)
        self.assertEqual(ts_sliced.at(0.6).total_branch_length, 0)
        self.assertEqual(ts_sliced.at(0.999).total_branch_length, 0)
        self.assertEqual(ts_sliced.at_index(-1).total_branch_length, 0)


class TestKeepIntervals(TopologyTestCase):
    """
    Tests for keep_intervals operation, where we slice out multiple disjoint
    intervals concurrently.
    """
    def example_intervals(self, tables):
        L = tables.sequence_length
        yield []
        yield [(0, L)]
        yield [(0, L / 2), (L / 2, L)]
        yield [(0, 0.25 * L), (0.75 * L, L)]
        yield [(0.25 * L, L)]
        yield [(0.25 * L, 0.5 * L)]
        yield [(0.25 * L, 0.5 * L), (0.75 * L, 0.8 * L)]

    def do_keep_intervals(
            self, tables, intervals, simplify=True, record_provenance=True):
        t1 = tables.copy()
        simple_keep_intervals(t1, intervals, simplify, record_provenance)
        t2 = tables.copy()
        t2.keep_intervals(intervals, simplify, record_provenance)
        self.assertTrue(tables_equal(t1, t2))
        return t2

    def test_migration_error(self):
        tables = tskit.TableCollection(1)
        tables.migrations.add_row(0, 1, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            tables.keep_intervals([[0, 1]])

    def test_bad_intervals(self):
        tables = tskit.TableCollection(10)
        bad_intervals = [
            [[1, 1]],
            [[-1, 0]],
            [[0, 11]],
            [[0, 5], [4, 6]]
        ]
        for intervals in bad_intervals:
            with self.assertRaises(ValueError):
                tables.keep_intervals(intervals)
            with self.assertRaises(ValueError):
                tables.delete_intervals(intervals)

    def test_one_interval(self):
        ts = msprime.simulate(
            10, random_seed=self.random_seed, recombination_rate=2, mutation_rate=2)
        tables = ts.tables
        intervals = [(0.3, 0.7)]
        for simplify in (True, False):
            for rec_prov in (True, False):
                self.do_keep_intervals(tables, intervals, simplify, rec_prov)

    def test_two_intervals(self):
        ts = msprime.simulate(
            10, random_seed=self.random_seed, recombination_rate=2, mutation_rate=2)
        tables = ts.tables
        intervals = [(0.1, 0.2), (0.8, 0.9)]
        for simplify in (True, False):
            for rec_prov in (True, False):
                self.do_keep_intervals(tables, intervals, simplify, rec_prov)

    def test_ten_intervals(self):
        ts = msprime.simulate(
            10, random_seed=self.random_seed, recombination_rate=2, mutation_rate=2)
        tables = ts.tables
        intervals = [(x, x + 0.05) for x in np.arange(0.0, 1.0, 0.1)]
        for simplify in (True, False):
            for rec_prov in (True, False):
                self.do_keep_intervals(tables, intervals, simplify, rec_prov)

    def test_hundred_intervals(self):
        ts = msprime.simulate(
            10, random_seed=self.random_seed, recombination_rate=2, mutation_rate=2)
        tables = ts.tables
        intervals = [(x, x + 0.005) for x in np.arange(0.0, 1.0, 0.01)]
        for simplify in (True, False):
            for rec_prov in (True, False):
                self.do_keep_intervals(tables, intervals, simplify, rec_prov)

    def test_regular_intervals(self):
        ts = msprime.simulate(
            3, random_seed=1234, recombination_rate=2, mutation_rate=2)
        tables = ts.tables
        eps = 0.0125
        for num_intervals in range(2, 10):
            breaks = np.linspace(0, ts.sequence_length, num=num_intervals)
            intervals = [(x, x + eps) for x in breaks[:-1]]
            self.do_keep_intervals(tables, intervals)

    def test_no_edges_sites(self):
        tables = tskit.TableCollection(1.0)
        tables.sites.add_row(0.1, "A")
        tables.sites.add_row(0.2, "T")
        for intervals in self.example_intervals(tables):
            self.assertEqual(len(tables.sites), 2)
            diced = self.do_keep_intervals(tables, intervals)
            self.assertEqual(diced.sequence_length, 1)
            self.assertEqual(len(diced.edges), 0)
            self.assertEqual(len(diced.sites), 0)

    def verify(self, tables):
        for intervals in self.example_intervals(tables):
            for simplify in [True, False]:
                self.do_keep_intervals(tables, intervals, simplify=simplify)

    def test_empty_tables(self):
        tables = tskit.TableCollection(1.0)
        self.verify(tables)

    def test_single_tree_jukes_cantor(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        self.verify(ts.tables)

    def test_single_tree_multichar_mutations(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.insert_multichar_mutations(ts)
        self.verify(ts.tables)

    def test_many_trees_infinite_sites(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=2, random_seed=1)
        self.assertGreater(ts.num_sites, 0)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts.tables)

    def test_many_trees_sequence_length_infinite_sites(self):
        for L in [0.5, 1.5, 3.3333]:
            ts = msprime.simulate(
                6, length=L, recombination_rate=2, mutation_rate=1, random_seed=1)
            self.verify(ts.tables)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            4, 5, seed=1, deep_history=True, initial_generation_samples=False,
            num_loci=10)
        tables.sort()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.05, random_seed=234)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts.tables)

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            6, 5, seed=3, deep_history=True, initial_generation_samples=True,
            num_loci=2)
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.08, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts.tables)

    def test_wright_fisher_initial_generation_no_deep_history(self):
        tables = wf.wf_sim(
            7, 15, seed=202, deep_history=False, initial_generation_samples=True,
            num_loci=5)
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.01, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts.tables)

    def test_wright_fisher_unsimplified_multiple_roots(self):
        tables = wf.wf_sim(
            8, 15, seed=1, deep_history=False, initial_generation_samples=False,
            num_loci=20)
        tables.sort()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.006, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts.tables)

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            9, 10, seed=1, deep_history=True, initial_generation_samples=False,
            num_loci=5)
        tables.sort()
        ts = tables.tree_sequence().simplify()
        ts = msprime.mutate(ts, rate=0.01, random_seed=1234)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts.tables)


class TestKeepDeleteIntervalsExamples(unittest.TestCase):
    """
    Simple examples of keep/delete intervals at work.
    """

    def test_tables_single_tree_keep_middle(self):
        ts = msprime.simulate(10, random_seed=2)
        t_keep = ts.dump_tables()
        t_keep.keep_intervals([[0.25, 0.5]], record_provenance=False)
        t_delete = ts.dump_tables()
        t_delete.delete_intervals([[0, 0.25], [0.5, 1.0]], record_provenance=False)
        self.assertEqual(t_keep, t_delete)

    def test_tables_single_tree_delete_middle(self):
        ts = msprime.simulate(10, random_seed=2)
        t_keep = ts.dump_tables()
        t_keep.delete_intervals([[0.25, 0.5]], record_provenance=False)
        t_delete = ts.dump_tables()
        t_delete.keep_intervals([[0, 0.25], [0.5, 1.0]], record_provenance=False)
        self.assertEqual(t_keep, t_delete)

    def test_ts_single_tree_keep_middle(self):
        ts = msprime.simulate(10, random_seed=2)
        ts_keep = ts.keep_intervals([[0.25, 0.5]], record_provenance=False)
        ts_delete = ts.delete_intervals([[0, 0.25], [0.5, 1.0]], record_provenance=False)
        self.assertTrue(ts_equal(ts_keep, ts_delete))

    def test_ts_single_tree_delete_middle(self):
        ts = msprime.simulate(10, random_seed=2)
        ts_keep = ts.delete_intervals([[0.25, 0.5]], record_provenance=False)
        ts_delete = ts.keep_intervals([[0, 0.25], [0.5, 1.0]], record_provenance=False)
        self.assertTrue(ts_equal(ts_keep, ts_delete))


class TestTrim(unittest.TestCase):
    """
    Test the trimming functionality
    """
    def add_mutations(self, ts, position, ancestral_state, derived_states, nodes):
        """
        Create a site at the specified position and assign mutations to the specified
        nodes (could be sequential mutations)
        """
        tables = ts.dump_tables()
        site = tables.sites.add_row(position, ancestral_state)
        for state, node in zip(derived_states, nodes):
            tables.mutations.add_row(site, node, state)
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        return tables.tree_sequence()

    def verify_sites(self, source_tree, trimmed_tree, position_offset):
        source_sites = list(source_tree.sites())
        trimmed_sites = list(trimmed_tree.sites())
        self.assertEqual(len(source_sites), len(trimmed_sites))
        for source_site, trimmed_site in zip(source_sites, trimmed_sites):
            self.assertAlmostEqual(
                source_site.position, position_offset + trimmed_site.position)
            self.assertEqual(
                source_site.ancestral_state, trimmed_site.ancestral_state)
            self.assertEqual(source_site.metadata, trimmed_site.metadata)
            self.assertEqual(
                len(source_site.mutations), len(trimmed_site.mutations))
            for source_mut, trimmed_mut in zip(
                    source_site.mutations, trimmed_site.mutations):
                self.assertEqual(source_mut.node, trimmed_mut.node)
                self.assertEqual(
                    source_mut.derived_state, trimmed_mut.derived_state)
                self.assertEqual(
                    source_mut.metadata, trimmed_mut.metadata)
                # mutation.parent id may have changed after deleting redundant mutations
                if source_mut.parent == trimmed_mut.parent == tskit.NULL:
                    pass
                else:
                    self.assertEqual(
                        source_tree.tree_sequence.mutation(source_mut.parent).node,
                        trimmed_tree.tree_sequence.mutation(trimmed_mut.parent).node)

    def verify_ltrim(self, source_ts, trimmed_ts):
        deleted_span = source_ts.first().span
        self.assertAlmostEqual(
            source_ts.sequence_length, trimmed_ts.sequence_length + deleted_span)
        self.assertEqual(source_ts.num_trees, trimmed_ts.num_trees + 1)
        for j in range(trimmed_ts.num_trees):
            source_tree = source_ts.at_index(j + 1)
            trimmed_tree = trimmed_ts.at_index(j)
            self.assertEqual(source_tree.parent_dict, trimmed_tree.parent_dict)
            self.assertAlmostEqual(source_tree.span, trimmed_tree.span)
            self.assertAlmostEqual(
                source_tree.interval[0], trimmed_tree.interval[0] + deleted_span)
            self.verify_sites(source_tree, trimmed_tree, deleted_span)

    def verify_rtrim(self, source_ts, trimmed_ts):
        deleted_span = source_ts.last().span
        self.assertAlmostEqual(
            source_ts.sequence_length, trimmed_ts.sequence_length + deleted_span)
        self.assertEqual(source_ts.num_trees, trimmed_ts.num_trees + 1)
        for j in range(trimmed_ts.num_trees):
            source_tree = source_ts.at_index(j)
            trimmed_tree = trimmed_ts.at_index(j)
            self.assertEqual(source_tree.parent_dict, trimmed_tree.parent_dict)
            self.assertEqual(source_tree.interval, trimmed_tree.interval)
            self.verify_sites(source_tree, trimmed_tree, 0)

    def clear_left_mutate(self, ts, left, num_sites):
        """
        Clear the edges from a tree sequence left of the specified coordinate
        and add in num_sites regularly spaced sites into the cleared region.
        """
        new_ts = ts.delete_intervals([[0.0, left]])
        for j, x in enumerate(np.linspace(0, left, num_sites, endpoint=False)):
            new_ts = self.add_mutations(new_ts, x, 'A' * j, ['T'] * j, range(j+1))
        return new_ts

    def clear_right_mutate(self, ts, right, num_sites):
        """
        Clear the edges from a tree sequence right of the specified coordinate
        and add in num_sites regularly spaced sites into the cleared region.
        """
        new_ts = ts.delete_intervals([[right, ts.sequence_length]])
        for j, x in enumerate(
                np.linspace(right, ts.sequence_length, num_sites, endpoint=False)):
            new_ts = self.add_mutations(new_ts, x, 'A' * j, ['T'] * j, range(j+1))
        return new_ts

    def clear_left_right_234(self, left, right):
        """
        Clear edges to left and right and add 2 mutations at the same site into the left
        cleared region, 3 at the same site into the untouched region, and 4 into the
        right cleared region.
        """
        assert 0.0 < left < right < 1.0
        ts = msprime.simulate(10, recombination_rate=10, random_seed=2)
        left_pos = np.mean([0.0, left])
        left_root = ts.at(left_pos).root
        mid_pos = np.mean([left, right])
        mid_root = ts.at(mid_pos).root
        right_pos = np.mean([right, ts.sequence_length])
        right_root = ts.at(right_pos).root
        # Clear
        ts = ts.keep_intervals([[left, right]], simplify=False)
        ts = self.add_mutations(ts, left_pos, 'A', ['T', 'C'], [left_root, 0])
        ts = self.add_mutations(ts, mid_pos, 'T', ['A', 'C', 'G'], [mid_root, 0, 1])
        ts = self.add_mutations(
            ts, right_pos, 'X', ['T', 'C', 'G', 'A'], [right_root, 0, 1, 2])
        self.assertNotEqual(np.min(ts.tables.edges.left), 0)
        self.assertEqual(ts.num_mutations, 9)
        self.assertEqual(ts.num_sites, 3)
        return ts

    def test_ltrim_single_tree(self):
        ts = msprime.simulate(10, mutation_rate=12, random_seed=2)
        ts = self.clear_left_mutate(ts, 0.5, 10)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_single_tree_no_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = self.clear_left_mutate(ts, 0.5, 0)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_single_tree_tiny_left(self):
        ts = msprime.simulate(10, mutation_rate=12, random_seed=2)
        ts = self.clear_left_mutate(ts, 1e-200, 10)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_many_trees(self):
        ts = msprime.simulate(10, recombination_rate=10, mutation_rate=12, random_seed=2)
        ts = self.clear_left_mutate(ts, 0.5, 10)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_many_trees_left_min(self):
        ts = msprime.simulate(10, recombination_rate=10, mutation_rate=12, random_seed=2)
        ts = self.clear_left_mutate(ts, sys.float_info.min, 10)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_many_trees_left_epsilon(self):
        ts = msprime.simulate(10, recombination_rate=10, mutation_rate=12, random_seed=2)
        ts = self.clear_left_mutate(ts, sys.float_info.epsilon, 0)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_empty(self):
        ts = msprime.simulate(2, random_seed=2)
        ts = ts.delete_intervals([[0, 1]])
        self.assertRaises(ValueError, ts.ltrim)

    def test_ltrim_multiple_mutations(self):
        ts = self.clear_left_right_234(0.1, 0.5)
        trimmed_ts = ts.ltrim()
        self.assertAlmostEqual(trimmed_ts.sequence_length, 0.9)
        self.assertEqual(trimmed_ts.num_sites, 2)
        self.assertEqual(trimmed_ts.num_mutations, 7)  # We should have deleted 2
        self.assertEqual(np.min(trimmed_ts.tables.edges.left), 0)
        self.verify_ltrim(ts, trimmed_ts)

    def test_rtrim_single_tree(self):
        ts = msprime.simulate(10, mutation_rate=12, random_seed=2)
        ts = self.clear_right_mutate(ts, 0.5, 10)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_single_tree_no_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = self.clear_right_mutate(ts, 0.5, 0)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_single_tree_tiny_left(self):
        ts = msprime.simulate(10, mutation_rate=12, random_seed=2)
        ts = self.clear_right_mutate(ts, 1e-200, 10)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_many_trees(self):
        ts = msprime.simulate(10, recombination_rate=10, mutation_rate=12, random_seed=2)
        ts = self.clear_right_mutate(ts, 0.5, 10)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_many_trees_left_min(self):
        ts = msprime.simulate(10, recombination_rate=10, mutation_rate=12, random_seed=2)
        ts = self.clear_right_mutate(ts, sys.float_info.min, 10)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_many_trees_left_epsilon(self):
        ts = msprime.simulate(10, recombination_rate=10, mutation_rate=12, random_seed=2)
        ts = self.clear_right_mutate(ts, sys.float_info.epsilon, 0)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_empty(self):
        ts = msprime.simulate(2, random_seed=2)
        ts = ts.delete_intervals([[0, 1]])
        self.assertRaises(ValueError, ts.rtrim)

    def test_rtrim_multiple_mutations(self):
        ts = self.clear_left_right_234(0.1, 0.5)
        trimmed_ts = ts.rtrim()
        self.assertAlmostEqual(trimmed_ts.sequence_length, 0.5)
        self.assertEqual(trimmed_ts.num_sites, 2)
        self.assertEqual(trimmed_ts.num_mutations, 5)  # We should have deleted 4
        self.assertEqual(
            np.max(trimmed_ts.tables.edges.right), trimmed_ts.tables.sequence_length)
        self.verify_rtrim(ts, trimmed_ts)

    def test_trim_multiple_mutations(self):
        ts = self.clear_left_right_234(0.1, 0.5)
        trimmed_ts = ts.trim()
        self.assertAlmostEqual(trimmed_ts.sequence_length, 0.4)
        self.assertEqual(trimmed_ts.num_mutations, 3)
        self.assertEqual(trimmed_ts.num_sites, 1)
        self.assertEqual(np.min(trimmed_ts.tables.edges.left), 0)
        self.assertEqual(
            np.max(trimmed_ts.tables.edges.right), trimmed_ts.tables.sequence_length)

    def test_trims_no_effect(self):
        # Deleting from middle should have no effect on any trim function
        ts = msprime.simulate(10, recombination_rate=2, mutation_rate=50, random_seed=2)
        ts = ts.delete_intervals([[0.1, 0.5]])
        trimmed_ts = ts.ltrim(record_provenance=False)
        self.assertTrue(ts_equal(ts, trimmed_ts))
        trimmed_ts = ts.rtrim(record_provenance=False)
        self.assertTrue(ts_equal(ts, trimmed_ts))
        trimmed_ts = ts.trim(record_provenance=False)
        self.assertTrue(ts_equal(ts, trimmed_ts))

    def test_failure_with_migrations(self):
        # All trim functions fail if migrations present
        ts = msprime.simulate(10, recombination_rate=2, random_seed=2)
        ts = ts.keep_intervals([[0.1, 0.5]])
        tables = ts.dump_tables()
        tables.migrations.add_row(0, 1, 0, 0, 0, 0)
        ts = tables.tree_sequence()
        self.assertRaises(ValueError, ts.ltrim)
        self.assertRaises(ValueError, ts.rtrim)
        self.assertRaises(ValueError, ts.trim)
