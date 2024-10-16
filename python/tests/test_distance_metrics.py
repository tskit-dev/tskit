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
Tests for tree distance metrics.
"""
import io
import itertools
import math
import unittest

import dendropy
import msprime
import numpy as np
import pytest
from dendropy.calculate import treecompare

import _tskit
import tests
import tests.tsutil as tsutil
import tskit


def c_kc_distance(tree1, tree2, lambda_=0):
    """
    Simplified version of the naive_kc_distance() function above.
    Written without Python features to aid writing C implementation.
    """
    samples = tree1.tree_sequence.samples()
    if tree1.tree_sequence.num_samples != tree2.tree_sequence.num_samples:
        raise ValueError("Trees must have the same samples")
    for sample1, sample2 in zip(samples, tree2.tree_sequence.samples()):
        if sample1 != sample2:
            raise ValueError("Trees must have the same samples")
    if not len(tree1.roots) == len(tree2.roots) == 1:
        raise ValueError("Trees must have one root")
    for tree in [tree1, tree2]:
        for u in range(tree.tree_sequence.num_nodes):
            if tree.num_children(u) == 1:
                raise ValueError("Unary nodes are not supported")

    n = tree1.tree_sequence.num_samples
    vecs1 = KCVectors(n)
    fill_kc_vectors(tree1, vecs1)
    vecs2 = KCVectors(n)
    fill_kc_vectors(tree2, vecs2)
    return norm_kc_vectors(vecs1, vecs2, lambda_)


def naive_kc_distance(tree1, tree2, lambda_=0):
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
    for tree in [tree1, tree2]:
        for u in tree.nodes():
            if tree.num_children(u) == 1:
                raise ValueError("Unary nodes are not supported")

    n = samples.shape[0]
    N = (n * (n - 1)) // 2
    m = [np.zeros(N + n), np.zeros(N + n)]
    M = [np.zeros(N + n), np.zeros(N + n)]
    for tree_index, tree in enumerate([tree1, tree2]):
        for sample in range(n):
            m[tree_index][N + sample] = 1
            M[tree_index][N + sample] = tree.branch_length(sample)

        for n1, n2 in itertools.combinations(range(n), 2):
            mrca = tree.mrca(samples[n1], samples[n2])
            depth = 0
            u = tree.parent(mrca)
            while u != tskit.NULL:
                depth += 1
                u = tree.parent(u)
            pair_index = n1 * (n1 - 2 * n + 1) // -2 + n2 - n1 - 1
            m[tree_index][pair_index] = depth
            M[tree_index][pair_index] = tree.time(tree.root) - tree.time(mrca)

    return np.linalg.norm((1 - lambda_) * (m[0] - m[1]) + lambda_ * (M[0] - M[1]))


class KCVectors:
    """
    Manages the two vectors (m and M) of a tree used to compute the
    KC distance between trees. For any two samples, u and v,
    m and M capture the distance of mrca(u, v) to the root in
    number of edges and time, respectively.

    See Kendall & Colijn (2016):
    https://academic.oup.com/mbe/article/33/10/2735/2925548
    """

    def __init__(self, n):
        self.n = n
        self.N = (self.n * (self.n - 1)) // 2
        self.m = np.zeros(self.N + self.n)
        self.M = np.zeros(self.N + self.n)


def fill_kc_vectors(tree, kc_vecs):
    sample_index_map = np.zeros(tree.tree_sequence.num_nodes)
    for j, u in enumerate(tree.tree_sequence.samples()):
        sample_index_map[u] = j
    for root in tree.roots:
        stack = [(tree.root, 0)]
        while len(stack) > 0:
            u, depth = stack.pop()
            if tree.is_sample(u):
                time = tree.branch_length(u)
                update_kc_vectors_single_leaf(kc_vecs, u, time, sample_index_map)

            c1 = tree.left_child(u)
            while c1 != tskit.NULL:
                stack.append((c1, depth + 1))
                c2 = tree.right_sib(c1)
                while c2 != tskit.NULL:
                    update_kc_vectors_all_pairs(
                        tree, kc_vecs, c1, c2, depth, tree.time(root) - tree.time(u)
                    )
                    c2 = tree.right_sib(c2)
                c1 = tree.right_sib(c1)


def update_kc_vectors_single_leaf(kc_vecs, u, time, sample_index_map):
    u_index = int(sample_index_map[u])
    kc_vecs.m[kc_vecs.N + u_index] = 1
    kc_vecs.M[kc_vecs.N + u_index] = time


def update_kc_vectors_all_pairs(tree, kc_vecs, c1, c2, depth, time):
    s1_index = tree.left_sample(c1)
    while True:
        s2_index = tree.left_sample(c2)
        while True:
            update_kc_vectors_pair(kc_vecs, s1_index, s2_index, depth, time)
            if s2_index == tree.right_sample(c2):
                break
            s2_index = tree.next_sample(s2_index)
        if s1_index == tree.right_sample(c1):
            break
        s1_index = tree.next_sample(s1_index)


def update_kc_vectors_pair(kc_vecs, n1, n2, depth, time):
    if n1 > n2:
        n1, n2 = n2, n1
    pair_index = n2 - n1 - 1 + (-1 * n1 * (n1 - 2 * kc_vecs.n + 1)) // 2

    kc_vecs.m[pair_index] = depth
    kc_vecs.M[pair_index] = time


def norm_kc_vectors(kc_vecs1, kc_vecs2, lambda_):
    vT1 = 0
    vT2 = 0
    distance_sum = 0
    for i in range(kc_vecs1.n + kc_vecs1.N):
        vT1 = (kc_vecs1.m[i] * (1 - lambda_)) + (lambda_ * kc_vecs1.M[i])
        vT2 = (kc_vecs2.m[i] * (1 - lambda_)) + (lambda_ * kc_vecs2.M[i])
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
                tree = next(ts.trees(sample_lists=True))
                assert naive_kc_distance(tree, tree) == 0
                assert c_kc_distance(tree, tree) == 0
                assert tree.kc_distance(tree) == 0
                ts = msprime.simulate(n, random_seed=seed)
                tree2 = next(ts.trees(sample_lists=True))
                assert naive_kc_distance(tree, tree2) == 0
                assert c_kc_distance(tree, tree2) == 0
                assert tree.kc_distance(tree2) == 0

    def test_sample_2_zero_distance(self):
        # All trees with 2 leaves must be equal distance from each other.
        for seed in range(1, 10):
            ts1 = msprime.simulate(2, random_seed=seed)
            tree1 = next(ts1.trees(sample_lists=True))
            ts2 = msprime.simulate(2, random_seed=seed + 1)
            tree2 = next(ts2.trees(sample_lists=True))
            assert naive_kc_distance(tree1, tree2, 0) == 0
            assert c_kc_distance(tree1, tree2, 0) == 0
            assert tree1.kc_distance(tree2, 0) == 0

    def test_different_samples_error(self):
        tree1 = next(msprime.simulate(10, random_seed=1).trees(sample_lists=True))
        tree2 = next(msprime.simulate(2, random_seed=1).trees(sample_lists=True))
        with pytest.raises(ValueError):
            naive_kc_distance(tree1, tree2)
        with pytest.raises(ValueError):
            c_kc_distance(tree1, tree2)
        with pytest.raises(_tskit.LibraryError):
            tree1.kc_distance(tree2)

        ts1 = msprime.simulate(10, random_seed=1)
        nmap = np.arange(0, ts1.num_nodes)[::-1]
        ts2 = tsutil.permute_nodes(ts1, nmap)
        tree1 = next(ts1.trees(sample_lists=True))
        tree2 = next(ts2.trees(sample_lists=True))
        with pytest.raises(ValueError):
            naive_kc_distance(tree1, tree2)
        with pytest.raises(ValueError):
            c_kc_distance(tree1, tree2)
        with pytest.raises(_tskit.LibraryError):
            tree1.kc_distance(tree2)

        unsimplified_ts = msprime.simulate(
            10, random_seed=1, recombination_rate=10, record_full_arg=True
        )
        trees = unsimplified_ts.trees(sample_lists=True)
        tree1 = next(trees)
        tree2 = next(trees)
        with pytest.raises(ValueError):
            naive_kc_distance(tree1, tree2)
        with pytest.raises(ValueError):
            c_kc_distance(tree1, tree2)
        with pytest.raises(_tskit.LibraryError):
            tree1.kc_distance(tree2)

    def validate_trees(self, n):
        for seed in range(1, 10):
            ts1 = msprime.simulate(n, random_seed=seed)
            ts2 = msprime.simulate(n, random_seed=seed + 1)
            tree1 = next(ts1.trees(sample_lists=True))
            tree2 = next(ts2.trees(sample_lists=True))
            kc1 = naive_kc_distance(tree1, tree2)
            kc2 = c_kc_distance(tree1, tree2)
            kc3 = tree1.kc_distance(tree2)
            self.assertAlmostEqual(kc1, kc2)
            self.assertAlmostEqual(kc1, kc3)
            self.assertAlmostEqual(kc1, naive_kc_distance(tree2, tree1))
            self.assertAlmostEqual(kc2, c_kc_distance(tree2, tree1))
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
            msprime.SimpleBottleneck(0.2, 0, proportion=1),
        ]

        for seed in range(1, 10):
            ts = msprime.simulate(
                n, random_seed=seed, demographic_events=demographic_events
            )
            # Check if this is really nonbinary
            found = False
            for edgeset in ts.edgesets():
                if len(edgeset.children) > 2:
                    found = True
                    break
            assert found
            tree1 = next(ts.trees(sample_lists=True))

            ts = msprime.simulate(
                n, random_seed=seed + 1, demographic_events=demographic_events
            )
            tree2 = next(ts.trees(sample_lists=True))
            self.do_kc_distance(tree1, tree2)
            # compare to a binary tree also

            ts = msprime.simulate(n, random_seed=seed + 1)
            tree2 = next(ts.trees(sample_lists=True))
            self.do_kc_distance(tree1, tree2)

    def test_non_binary_sample_10(self):
        self.validate_nonbinary_trees(10)

    def test_non_binary_sample_20(self):
        self.validate_nonbinary_trees(20)

    def test_non_binary_sample_30(self):
        self.validate_nonbinary_trees(30)

    def verify_result(self, tree1, tree2, lambda_, result, places=None):
        kc1 = naive_kc_distance(tree1, tree2, lambda_)
        kc2 = c_kc_distance(tree1, tree2, lambda_)
        kc3 = tree1.kc_distance(tree2, lambda_)
        self.assertAlmostEqual(kc1, result, places=places)
        self.assertAlmostEqual(kc2, result, places=places)
        self.assertAlmostEqual(kc3, result, places=places)

        kc1 = naive_kc_distance(tree2, tree1, lambda_)
        kc2 = c_kc_distance(tree2, tree1, lambda_)
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

        for left, right, p, c in zip(lv, rv, pv, cv):
            tables_1.edges.add_row(left=left, right=right, parent=p, child=c)
            tables_2.edges.add_row(left=left, right=right, parent=p, child=c)

        tree_1 = next(tables_1.tree_sequence().trees(sample_lists=True))
        tree_2 = next(tables_2.tree_sequence().trees(sample_lists=True))
        self.verify_result(tree_1, tree_2, 0, 0)
        self.verify_result(tree_1, tree_2, 1, 4.243, places=3)

    def test_10_samples(self):
        nodes_1 = io.StringIO(
            """\
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
        """
        )
        edges_1 = io.StringIO(
            """\
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
        """
        )
        ts_1 = tskit.load_text(
            nodes_1, edges_1, sequence_length=10000, strict=False, base64_metadata=False
        )
        nodes_2 = io.StringIO(
            """\
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
        """
        )
        edges_2 = io.StringIO(
            """\
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
        """
        )
        ts_2 = tskit.load_text(
            nodes_2, edges_2, sequence_length=10000, strict=False, base64_metadata=False
        )

        tree_1 = next(ts_1.trees(sample_lists=True))
        tree_2 = next(ts_2.trees(sample_lists=True))
        self.verify_result(tree_1, tree_2, 0, 12.85, places=2)
        self.verify_result(tree_1, tree_2, 1, 10.64, places=2)

    def test_15_samples(self):
        nodes_1 = io.StringIO(
            """\
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
        """
        )
        edges_1 = io.StringIO(
            """\
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
        """
        )
        ts_1 = tskit.load_text(
            nodes_1, edges_1, sequence_length=10000, strict=False, base64_metadata=False
        )

        nodes_2 = io.StringIO(
            """\
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
        """
        )
        edges_2 = io.StringIO(
            """\
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
        """
        )
        ts_2 = tskit.load_text(
            nodes_2, edges_2, sequence_length=10000, strict=False, base64_metadata=False
        )

        tree_1 = next(ts_1.trees(sample_lists=True))
        tree_2 = next(ts_2.trees(sample_lists=True))

        self.verify_result(tree_1, tree_2, 0, 19.95, places=2)
        self.verify_result(tree_1, tree_2, 1, 17.74, places=2)

    def test_nobinary_trees(self):
        nodes_1 = io.StringIO(
            """\
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
        """
        )
        edges_1 = io.StringIO(
            """\
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
        """
        )
        ts_1 = tskit.load_text(
            nodes_1, edges_1, sequence_length=10000, strict=False, base64_metadata=False
        )

        nodes_2 = io.StringIO(
            """\
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
        """
        )
        edges_2 = io.StringIO(
            """\
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
        """
        )
        ts_2 = tskit.load_text(
            nodes_2, edges_2, sequence_length=10000, strict=False, base64_metadata=False
        )
        tree_1 = next(ts_1.trees(sample_lists=True))
        tree_2 = next(ts_2.trees(sample_lists=True))
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

        with pytest.raises(ValueError):
            naive_kc_distance(ts.first(), ts.first(), 0)
        with pytest.raises(ValueError):
            c_kc_distance(ts.first(), ts.first(), 0)
        with pytest.raises(_tskit.LibraryError):
            ts.first().kc_distance(ts.first(), 0)

    def do_kc_distance(self, t1, t2, lambda_=0):
        kc1 = naive_kc_distance(t1, t2, lambda_)
        kc2 = c_kc_distance(t1, t2, lambda_)
        kc3 = t1.kc_distance(t2, lambda_)
        self.assertAlmostEqual(kc1, kc2)
        self.assertAlmostEqual(kc1, kc3)

        kc1 = naive_kc_distance(t2, t1, lambda_)
        kc2 = c_kc_distance(t2, t1, lambda_)
        kc3 = t2.kc_distance(t1, lambda_)
        self.assertAlmostEqual(kc1, kc2)
        self.assertAlmostEqual(kc1, kc3)

    def test_non_initial_samples(self):
        ts1 = msprime.simulate(10, random_seed=1)
        nmap = np.arange(0, ts1.num_nodes)[::-1]
        ts2 = tsutil.permute_nodes(ts1, nmap)
        t1 = next(ts2.trees(sample_lists=True))
        t2 = next(ts2.trees(sample_lists=True))
        self.do_kc_distance(t1, t2)

    def test_internal_samples(self):
        ts1 = msprime.simulate(10, random_seed=1)
        ts2 = tsutil.jiggle_samples(ts1)
        t1 = next(ts2.trees(sample_lists=True))
        t2 = next(ts2.trees(sample_lists=True))

        naive_kc_distance(t1, t2)
        c_kc_distance(t1, t2)
        t1.kc_distance(t2)

    def test_root_sample(self):
        tables1 = tskit.TableCollection(sequence_length=1.0)
        tables1.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        only_root = next(tables1.tree_sequence().trees(sample_lists=True))
        assert only_root.kc_distance(only_root) == 0
        assert only_root.kc_distance(only_root, lambda_=1) == 0

    def test_non_sample_leaf(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        c1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        c2 = tables.nodes.add_row(time=0)
        p = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c2)
        ts = tables.tree_sequence()
        tree = next(ts.trees(sample_lists=True))
        assert ts.kc_distance(ts) == 0
        assert tree.kc_distance(tree) == 0

        # mirrored
        tables = tskit.TableCollection(sequence_length=1.0)
        c1 = tables.nodes.add_row(time=0)
        c2 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        p = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c2)
        ts = tables.tree_sequence()
        tree = next(ts.trees(sample_lists=True))
        assert ts.kc_distance(ts) == 0
        assert tree.kc_distance(tree) == 0

    def test_ignores_subtrees_with_no_samples(self):
        nodes_1 = io.StringIO(
            """\
        id  is_sample   time    population  individual  metadata
        0   0   0.000000    0   -1
        1   0   0.000000    0   -1
        2   0   0.000000    0   -1
        3   1   0.000000    0   -1
        4   0   0.000000    0   -1
        5   0   0.000000    0   -1
        6   1   1.000000    0   -1
        7   1   2.000000    0   -1
        8   0   2.000000    0   -1
        9   0   3.000000    0   -1
        """
        )
        edges_1 = io.StringIO(
            """\
        left    right   parent  child
        0.000000    1.000000    6  0
        0.000000    1.000000    6  1
        0.000000    1.000000    7  2
        0.000000    1.000000    7  6
        0.000000    1.000000    8  4
        0.000000    1.000000    8  5
        0.000000    1.000000    9  3
        0.000000    1.000000    9  7
        0.000000    1.000000    9  8
        """
        )
        redundant = tskit.load_text(
            nodes_1, edges_1, sequence_length=1, strict=False, base64_metadata=False
        )

        nodes_2 = io.StringIO(
            """\
        id  is_sample   time    population  individual  metadata
        0   0   0.000000    0   -1
        1   0   0.000000    0   -1
        2   0   0.000000    0   -1
        3   1   0.000000    0   -1
        4   0   0.000000    0   -1
        5   0   0.000000    0   -1
        6   1   1.000000    0   -1
        7   1   2.000000    0   -1
        8   0   2.000000    0   -1
        9   0   3.000000    0   -1
        """
        )
        edges_2 = io.StringIO(
            """\
        left    right   parent  child
        0.000000    1.000000    7  2
        0.000000    1.000000    7  6
        0.000000    1.000000    9  3
        0.000000    1.000000    9  7
        """
        )
        simplified = tskit.load_text(
            nodes_2, edges_2, sequence_length=1, strict=False, base64_metadata=False
        )
        assert redundant.kc_distance(simplified, 0) == 0
        assert redundant.kc_distance(simplified, 1) == 0


def ts_kc_distance(ts1, ts2, lambda_=0):
    check_kc_tree_sequence_inputs(ts1, ts2)

    total = 0
    left = 0
    tree1_iter = ts1.trees(sample_lists=True)
    tree1 = next(tree1_iter)
    for tree2 in ts2.trees(sample_lists=True):
        while tree1.interval.right < tree2.interval.right:
            span = tree1.interval.right - left
            total += tree1.kc_distance(tree2, lambda_) * span

            left = tree1.interval.right
            tree1 = next(tree1_iter)
        span = tree2.interval.right - left
        left = tree2.interval.right
        total += tree1.kc_distance(tree2, lambda_) * span

    return total / ts1.sequence_length


def ts_kc_distance_incremental(ts1, ts2, lambda_=0):
    check_kc_tree_sequence_inputs(ts1, ts2)

    sample_maps = [dict(), dict()]
    for i, ts in enumerate([ts1, ts2]):
        for j, u in enumerate(ts.samples()):
            sample_maps[i][u] = j

    total = 0
    left = 0

    t1_vecs = KCVectors(ts1.num_samples)
    t2_vecs = KCVectors(ts2.num_samples)

    t1_depths = np.zeros(ts1.num_nodes)
    t2_depths = np.zeros(ts2.num_nodes)

    edge_diffs_iter_1 = ts1.edge_diffs()
    tree_iter_1 = ts1.trees(sample_lists=True)
    t1, t1_diffs = next(tree_iter_1), next(edge_diffs_iter_1)
    update_kc_incremental(t1, t1_vecs, t1_diffs, sample_maps[0], t1_depths)
    for t2, t2_diffs in zip(ts2.trees(sample_lists=True), ts2.edge_diffs()):
        update_kc_incremental(t2, t2_vecs, t2_diffs, sample_maps[1], t2_depths)
        while t1_diffs[0][1] < t2_diffs[0][1]:
            span = t1_diffs[0][1] - left
            total += norm_kc_vectors(t1_vecs, t2_vecs, lambda_) * span

            left = t1_diffs[0][1]
            t1, t1_diffs = next(tree_iter_1), next(edge_diffs_iter_1)
            update_kc_incremental(t1, t1_vecs, t1_diffs, sample_maps[0], t1_depths)
        span = t2_diffs[0][1] - left
        left = t2_diffs[0][1]
        total += norm_kc_vectors(t1_vecs, t2_vecs, lambda_) * span

    return total / ts1.sequence_length


# tree is the result of removing/inserting the edges in edge_diffs
def update_kc_incremental(tree, kc, edge_diffs, sample_index_map, depths):
    _, edges_out, edges_in = edge_diffs

    # Update state of detached subtrees.
    for e in reversed(edges_out):
        u = e.child
        depths[u] = 0

        # Only update detached subtrees that remain detached. Otherwise,
        # they must be reattached by an incoming edge and will be
        # updated below. We're looking into the future here by seeing
        # that u remains detached after all the incoming edges are
        # inserted into `tree`.
        if tree.parent(u) == tskit.NULL:
            update_kc_subtree_state(tree, kc, u, sample_index_map, depths)

    # Propagate state change down into reattached subtrees.
    for e in reversed(edges_in):
        u = e.child
        assert depths[u] == 0
        depths[u] = depths[e.parent] + 1
        update_kc_subtree_state(tree, kc, u, sample_index_map, depths)

        # The per-leaf elements of KC only change when the edge directly
        # above the leaf changes, so are handled separately from the
        # propagated state used for leaf-pair elements.
        if tree.is_leaf(u):
            time = tree.branch_length(u)
            update_kc_vectors_single_leaf(kc, u, time, sample_index_map)


def update_kc_subtree_state(tree, kc, u, sample_index_map, depths):
    """
    Update the depths of the nodes in this subtree. When a leaf is hit,
    update the KC vector elements associated with that leaf.
    """
    stack = [u]
    while len(stack) > 0:
        v = stack.pop()
        if tree.is_leaf(v):
            update_kc_pairs_with_leaf(tree, kc, v, sample_index_map, depths)
        else:
            c = tree.left_child(v)
            while c != -1:
                # Terminate iteration at nodes that are currently considered
                # roots by the edge diffs. Nodes with a depth of 0 are
                # temporary root nodes made by breaking an outgoing edge
                # that have yet to be inserted by a later incoming edge.
                if depths[c] != 0:
                    depths[c] = depths[v] + 1
                    stack.append(c)
                c = tree.right_sib(c)


def update_kc_pairs_with_leaf(tree, kc, leaf, sample_index_map, depths):
    """
    Perform an upward traversal from `leaf` to the root, updating the KC
    vector elements for pairs of `leaf` with every other leaf in the tree.
    """
    root_time = tree.time(tree.root)
    p = tree.parent(leaf)
    c = leaf
    while p != -1:
        time = root_time - tree.time(p)
        depth = depths[p]
        for sibling in tree.children(p):
            if sibling != c:
                update_kc_vectors_all_pairs(tree, kc, leaf, sibling, depth, time)
        c, p = p, tree.parent(p)


def check_kc_tree_sequence_inputs(ts1, ts2):
    if not np.array_equal(ts1.samples(), ts2.samples()):
        raise ValueError("Trees must have the same samples")
    if ts1.sequence_length != ts2.sequence_length:
        raise ValueError("Can't compare with sequences of different lengths")

    tree1_iter = ts1.trees(sample_lists=True)
    tree1 = next(tree1_iter)
    for tree2 in ts2.trees(sample_lists=True):
        while tree1.interval.right < tree2.interval.right:
            check_kc_tree_inputs(tree1, tree2)
            tree1 = next(tree1_iter)
        check_kc_tree_inputs(tree1, tree2)


def check_kc_tree_inputs(tree1, tree2):
    if not len(tree1.roots) == len(tree2.roots) == 1:
        raise ValueError("Trees must have one root")
    for tree in [tree1, tree2]:
        for u in tree.nodes():
            if tree.num_children(u) == 1:
                raise ValueError("Unary nodes are not supported")


class TestKCSequenceMetric(unittest.TestCase):
    """
    Tests the KC Metric on a tree sequence.
    """

    def test_0_distance_from_self(self):
        ts = msprime.simulate(10)
        assert ts_kc_distance(ts, ts) == 0

    def verify_errors(self, ts1, ts2):
        with pytest.raises(ValueError):
            ts_kc_distance(ts1, ts2)
        with pytest.raises(ValueError):
            ts_kc_distance_incremental(ts1, ts2)
        with pytest.raises(_tskit.LibraryError):
            ts1.kc_distance(ts2)

    def test_errors_diff_seq_length(self):
        ts1 = msprime.simulate(10, length=1)
        ts2 = msprime.simulate(10, length=2)
        self.verify_errors(ts1, ts2)

    def test_errors_diff_num_samples(self):
        ts1 = msprime.simulate(10, length=1)
        ts2 = msprime.simulate(12, length=2)
        self.verify_errors(ts1, ts2)

    def test_errors_different_sample_lists(self):
        tables_1 = tskit.TableCollection(sequence_length=2.0)
        tables_2 = tskit.TableCollection(sequence_length=2.0)

        sv1 = [True, True, True, False, False]
        tv1 = [0.0, 0.0, 0.0, 1.0, 2.0]
        sv2 = [True, True, False, False, True]
        tv2 = [0.0, 0.0, 1.0, 2.0, 0.0]
        for is_sample, t in zip(sv1, tv1):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables_1.nodes.add_row(flags=flags, time=t)
        for is_sample, t in zip(sv2, tv2):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables_2.nodes.add_row(flags=flags, time=t)

        lv = [0.0, 0.0, 0.0, 0.0]
        rv = [1.0, 1.0, 1.0, 1.0]
        pv1 = [3, 3, 4, 4]
        cv1 = [0, 1, 2, 3]
        for left, right, p, c in zip(lv, rv, pv1, cv1):
            tables_1.edges.add_row(left=left, right=right, parent=p, child=c)

        pv2 = [2, 2, 3, 3]
        cv2 = [0, 1, 2, 4]
        for left, right, p, c in zip(lv, rv, pv2, cv2):
            tables_2.edges.add_row(left=left, right=right, parent=p, child=c)

        ts1 = tables_1.tree_sequence()
        ts2 = tables_2.tree_sequence()
        self.verify_errors(ts1, ts2)

        unsimplified_ts = msprime.simulate(
            10, random_seed=1, recombination_rate=10, record_full_arg=True
        )
        self.verify_errors(unsimplified_ts, unsimplified_ts)

    def test_errors_unary_nodes(self):
        tables = tskit.TableCollection(sequence_length=2.0)

        sv = [True, False, False]
        tv = [0.0, 1.0, 2.0]
        for is_sample, t in zip(sv, tv):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables.nodes.add_row(flags=flags, time=t)

        lv = [0.0, 0.0, 0.0]
        rv = [1.0, 1.0, 1.0]
        pv = [1, 2]
        cv = [0, 1]
        for left, right, p, c in zip(lv, rv, pv, cv):
            tables.edges.add_row(left=left, right=right, parent=p, child=c)

        ts = tables.tree_sequence()
        self.verify_errors(ts, ts)

    def test_errors_different_samples(self):
        ts1 = msprime.simulate(10, random_seed=1)
        ts2 = tsutil.jiggle_samples(ts1)
        self.verify_errors(ts1, ts2)

    def verify_result(self, ts1, ts2, lambda_, result, places=None):
        kc1 = ts_kc_distance(ts1, ts2, lambda_)
        kc2 = ts_kc_distance_incremental(ts1, ts2, lambda_)
        kc3 = ts1.kc_distance(ts2, lambda_)
        self.assertAlmostEqual(kc1, result, places=places)
        self.assertAlmostEqual(kc2, result, places=places)
        self.assertAlmostEqual(kc3, result, places=places)

        kc1 = ts_kc_distance(ts2, ts1, lambda_)
        kc2 = ts_kc_distance_incremental(ts2, ts1, lambda_)
        kc3 = ts2.kc_distance(ts1, lambda_)
        self.assertAlmostEqual(kc1, result, places=places)
        self.assertAlmostEqual(kc2, result, places=places)
        self.assertAlmostEqual(kc3, result, places=places)

    def verify_same_kc(self, ts1, ts2, lambda_=0):
        kc1 = ts_kc_distance(ts1, ts2, lambda_)
        kc2 = ts_kc_distance_incremental(ts1, ts2, lambda_)
        kc3 = ts1.kc_distance(ts2, lambda_)
        self.assertAlmostEqual(kc1, kc2)
        self.assertAlmostEqual(kc2, kc3)

        kc1 = ts_kc_distance(ts2, ts1, lambda_)
        kc2 = ts_kc_distance_incremental(ts2, ts1, lambda_)
        kc3 = ts2.kc_distance(ts1, lambda_)
        self.assertAlmostEqual(kc1, kc2)
        self.assertAlmostEqual(kc2, kc3)

    def validate_trees(self, n):
        for seed in range(1, 10):
            ts1 = msprime.simulate(n, random_seed=seed, recombination_rate=1)
            ts2 = msprime.simulate(n, random_seed=seed + 1, recombination_rate=1)
            self.verify_same_kc(ts2, ts1)
            self.verify_same_kc(ts1, ts2)
            self.verify_same_kc(ts1, ts1)  # Test sequences with equal breakpoints

    def test_sample_5(self):
        self.validate_trees(5)

    def test_sample_10(self):
        self.validate_trees(10)

    def test_sample_20(self):
        self.validate_trees(20)

    def validate_nonbinary_trees(self, n):
        demographic_events = [
            msprime.SimpleBottleneck(0.02, 0, proportion=0.25),
            msprime.SimpleBottleneck(0.2, 0, proportion=1),
        ]

        for seed in range(1, 10):
            ts1 = msprime.simulate(
                n,
                random_seed=seed,
                demographic_events=demographic_events,
                recombination_rate=1,
            )
            # Check if this is really nonbinary
            found = False
            for edgeset in ts1.edgesets():
                if len(edgeset.children) > 2:
                    found = True
                    break
            assert found

            ts2 = msprime.simulate(
                n,
                random_seed=seed + 1,
                demographic_events=demographic_events,
                recombination_rate=1,
            )
            self.verify_same_kc(ts1, ts2)

            # compare to a binary tree also
            ts2 = msprime.simulate(n, recombination_rate=1, random_seed=seed + 1)
            self.verify_same_kc(ts1, ts2)

    def test_non_binary_sample_10(self):
        self.validate_nonbinary_trees(10)

    def test_non_binary_sample_20(self):
        self.validate_nonbinary_trees(20)

    def test_permit_internal_samples(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(flags=1)
        tables.nodes.add_row(flags=1)
        tables.nodes.add_row(flags=1, time=1)
        tables.edges.add_row(0, 1, 2, 0)
        tables.edges.add_row(0, 1, 2, 1)
        ts = tables.tree_sequence()
        assert ts.kc_distance(ts) == 0
        assert ts_kc_distance_incremental(ts, ts) == 0

    def test_known_kc_sample_trees_different_shapes(self):
        tables_1 = tskit.TableCollection(sequence_length=2.0)
        tables_2 = tskit.TableCollection(sequence_length=2.0)

        # Nodes
        sv = [True, True, True, True, False, False, False]
        tv = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
        for is_sample, t in zip(sv, tv):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables_1.nodes.add_row(flags=flags, time=t)
            tables_2.nodes.add_row(flags=flags, time=t)

        # First tree edges
        pv1 = [4, 4, 5, 5, 6, 6, 5, 6]
        cv1 = [2, 3, 1, 4, 0, 5, 0, 4]
        lv1 = [0, 0, 0, 0, 0, 0, 1, 1]
        rv1 = [2, 2, 2, 1, 1, 2, 2, 2]

        # Second tree edges
        pv2 = [4, 4, 5, 5, 6, 6, 5, 6]
        cv2 = [2, 3, 0, 1, 4, 5, 4, 0]
        lv2 = [0, 0, 0, 0, 0, 0, 1, 1]
        rv2 = [2, 2, 1, 2, 1, 2, 2, 2]

        for left, right, p, c in zip(lv1, rv1, pv1, cv1):
            tables_1.edges.add_row(left=left, right=right, parent=p, child=c)
        for left, right, p, c in zip(lv2, rv2, pv2, cv2):
            tables_2.edges.add_row(left=left, right=right, parent=p, child=c)

        tables_1.sort()
        tables_2.sort()
        ts_1 = tables_1.tree_sequence()
        ts_2 = tables_2.tree_sequence()
        self.verify_result(ts_1, ts_2, 0, 2.0)

    def test_known_kc_sample_trees_same_shape_different_times(self):
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

        for left, right, p, c in zip(lv, rv, pv, cv):
            tables_1.edges.add_row(left=left, right=right, parent=p, child=c)
            tables_2.edges.add_row(left=left, right=right, parent=p, child=c)

        ts_1 = tables_1.tree_sequence()
        ts_2 = tables_2.tree_sequence()

        self.verify_result(ts_1, ts_2, 0, 0)
        self.verify_result(ts_1, ts_2, 1, 4.243, places=3)

    def test_known_kc_same_tree_twice_same_metric(self):
        tables_1 = tskit.TableCollection(sequence_length=2.0)
        tables_2 = tskit.TableCollection(sequence_length=2.0)

        # Nodes
        sv = [True, True, True, False, False]
        tv_1 = [0.0, 0.0, 0.0, 2.0, 3.0]
        tv_2 = [0.0, 0.0, 0.0, 4.0, 6.0]

        for is_sample, t1, t2 in zip(sv, tv_1, tv_2):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables_1.nodes.add_row(flags=flags, time=t1)
            tables_2.nodes.add_row(flags=flags, time=t2)

        # Edges
        pv = [3, 3, 4, 4]
        cv = [0, 1, 2, 3]

        for p, c in zip(pv, cv):
            tables_1.edges.add_row(left=0, right=1, parent=p, child=c)
            tables_1.edges.add_row(left=1, right=2, parent=p, child=c)
            tables_2.edges.add_row(left=0, right=0.5, parent=p, child=c)
            tables_2.edges.add_row(left=0.5, right=2, parent=p, child=c)

        ts_1 = tables_1.tree_sequence()
        ts_2 = tables_2.tree_sequence()
        self.verify_result(ts_1, ts_2, 0, 0)
        self.verify_result(ts_1, ts_2, 1, 4.243, places=3)

    def test_remove_root(self):
        tables_1 = tskit.TableCollection(sequence_length=10.0)
        tables_2 = tskit.TableCollection(sequence_length=10.0)

        # Nodes
        sv1 = [True, True, True, True, True, False, False, False, False, False]
        tv1 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        sv2 = [True, True, True, True, True, False, False, False, False]
        tv2 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]

        for is_sample, t in zip(sv1, tv1):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables_1.nodes.add_row(flags=flags, time=t)
        for is_sample, t in zip(sv2, tv2):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables_2.nodes.add_row(flags=flags, time=t)

        # Edges
        pv1 = [5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9]
        cv1 = [0, 1, 3, 4, 2, 5, 2, 6, 7, 5, 8]
        lv1 = [0, 0, 0, 0, 5, 5, 0, 0, 5, 0, 0]
        rv1 = [10, 10, 10, 10, 10, 10, 5, 10, 10, 5, 5]

        pv2 = [5, 5, 6, 6, 7, 7, 8, 8]
        cv2 = [0, 1, 2, 3, 4, 5, 6, 7]
        lv2 = [0, 0, 0, 0, 0, 0, 0, 0]
        rv2 = [10, 10, 10, 10, 10, 10, 10, 10]

        for p, c, l, r in zip(pv1, cv1, lv1, rv1):
            tables_1.edges.add_row(left=l, right=r, parent=p, child=c)

        for p, c, l, r in zip(pv2, cv2, lv2, rv2):
            tables_2.edges.add_row(left=l, right=r, parent=p, child=c)

        ts_1 = tables_1.tree_sequence()
        ts_2 = tables_2.tree_sequence()
        distance = (math.sqrt(8) * 5 + math.sqrt(6) * 5) / 10
        self.verify_result(ts_1, ts_2, 0, distance)

    def test_ignores_subtrees_with_no_samples(self):
        nodes_1 = io.StringIO(
            """\
        id  is_sample   time    population  individual  metadata
        0   0   0.000000    0   -1
        1   0   0.000000    0   -1
        2   0   0.000000    0   -1
        3   1   0.000000    0   -1
        4   0   0.000000    0   -1
        5   0   0.000000    0   -1
        6   1   1.000000    0   -1
        7   1   2.000000    0   -1
        8   0   2.000000    0   -1
        9   0   3.000000    0   -1
        """
        )
        edges_1 = io.StringIO(
            """\
        left    right   parent  child
        0.000000    1.000000    6  0
        0.000000    1.000000    6  1
        0.000000    1.000000    7  2
        0.000000    1.000000    7  6
        0.000000    1.000000    8  4
        0.000000    1.000000    8  5
        0.000000    1.000000    9  3
        0.000000    1.000000    9  7
        0.000000    1.000000    9  8
        """
        )
        redundant = tskit.load_text(
            nodes_1, edges_1, sequence_length=1, strict=False, base64_metadata=False
        )

        nodes_2 = io.StringIO(
            """\
        id  is_sample   time    population  individual  metadata
        0   0   0.000000    0   -1
        1   0   0.000000    0   -1
        2   0   0.000000    0   -1
        3   1   0.000000    0   -1
        4   0   0.000000    0   -1
        5   0   0.000000    0   -1
        6   1   1.000000    0   -1
        7   1   2.000000    0   -1
        8   0   2.000000    0   -1
        9   0   3.000000    0   -1
        """
        )
        edges_2 = io.StringIO(
            """\
        left    right   parent  child
        0.000000    1.000000    7  2
        0.000000    1.000000    7  6
        0.000000    1.000000    9  3
        0.000000    1.000000    9  7
        """
        )
        simplified = tskit.load_text(
            nodes_2, edges_2, sequence_length=1, strict=False, base64_metadata=False
        )
        t1 = next(redundant.trees(sample_lists=True))
        t2 = next(simplified.trees(sample_lists=True))
        assert t1.kc_distance(t2, 0) == 0
        assert t1.kc_distance(t2, 1) == 0


# Test the RF distance metrics:
# TODO: integrate with the KC tests


class TestTreeSameSamples:
    # Tree1
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1
    #
    # Tree2
    # 3.00┊   6     ┊
    #     ┊ ┏━┻━┓   ┊
    # 2.00┊ ┃   5   ┊
    #     ┊ ┃ ┏━┻┓  ┊
    # 1.00┊ ┃ ┃  4  ┊
    #     ┊ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1

    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(4)

    @tests.cached_example
    def tree_other(self):
        return tskit.Tree.generate_comb(4)

    def test_rf_distance(self):
        assert self.tree().rf_distance(self.tree_other()) == 2


class TestTreeDifferentSamples:
    # Tree1
    # 2.00┊     6     ┊
    #     ┊   ┏━┻━┓   ┊
    # 1.00┊   4   5   ┊
    #     ┊  ┏┻┓ ┏┻┓  ┊
    # 0.00┊  0 1 2 3  ┊
    #     0           1
    #
    # Tree2
    # 4.00┊   8       ┊
    #     ┊ ┏━┻━┓     ┊
    # 3.00┊ ┃   7     ┊
    #     ┊ ┃ ┏━┻━┓   ┊
    # 2.00┊ ┃ ┃   6   ┊
    #     ┊ ┃ ┃ ┏━┻┓  ┊
    # 1.00┊ ┃ ┃ ┃  5  ┊
    #     ┊ ┃ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 4 ┊
    #     0           1

    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(4)

    @tests.cached_example
    def tree_other(self):
        return tskit.Tree.generate_comb(5)

    def test_rf_distance(self):
        assert self.tree().rf_distance(self.tree_other()) == 8


class TestTreeMultiRoots:
    # Tree1
    # 4.00┊        15             ┊
    #     ┊     ┏━━━┻━━━┓         ┊
    # 3.00┊     ┃      14         ┊
    #     ┊     ┃     ┏━┻━┓       ┊
    # 2.00┊    12     ┃  13       ┊
    #     ┊   ┏━┻━┓   ┃  ┏┻┓      ┊
    # 1.00┊   9  10   ┃  ┃ 11     ┊
    #     ┊  ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓    ┊
    # 0.00┊  0 1 2 3 4 5 6 7 8    ┊
    #     0                       1
    #
    # Tree2
    # 3.00┊              15       ┊
    #     ┊            ┏━━┻━┓     ┊
    # 2.00┊     11     ┃   14     ┊
    #     ┊    ┏━┻━┓   ┃  ┏━┻┓    ┊
    # 1.00┊    9  10  12  ┃ 13    ┊
    #     ┊   ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓   ┊
    # 0.00┊   0 1 2 3 4 5 6 7 8   ┊
    #     0                       1

    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(9)

    @tests.cached_example
    def tree_other(self):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            self.tree().rf_distance(self.tree_other())


class TestEmpty:
    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    @tests.cached_example
    def tree_other(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            self.tree().rf_distance(self.tree_other())


class TestTreeInNullState:
    @tests.cached_example
    def tsk_tree1(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    @tests.cached_example
    def tree_other(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            self.tsk_tree1().rf_distance(self.tree_other())


class TestAllRootsN5:
    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def test_rf_distance(self):
        with pytest.raises(ValueError, match="single root"):
            self.tree().rf_distance(self.tree())


class TestWithPackages:
    def to_dendropy(self, newick_data, tns):
        return dendropy.Tree.get(
            data=newick_data,
            schema="newick",
            rooting="force-rooted",
            taxon_namespace=tns,
        )

    def dendropy_rf_distance(self, tree1, tree2, weighted=False):
        tns = dendropy.TaxonNamespace()
        tree1 = self.to_dendropy(tree1.as_newick(), tns)
        tree2 = self.to_dendropy(tree2.as_newick(), tns)
        tree1.encode_bipartitions()
        tree2.encode_bipartitions()
        if weighted:
            return treecompare.weighted_robinson_foulds_distance(tree1, tree2)
        else:
            return treecompare.unweighted_robinson_foulds_distance(tree1, tree2)

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 20])
    def test_rf_distance_against_dendropy(self, n):
        trees = []
        for seed in [42, 43]:
            ts = msprime.sim_ancestry(n, ploidy=1, random_seed=seed)
            trees.append(ts.first())
        rf1 = trees[0].rf_distance(trees[1])
        rf2 = self.dendropy_rf_distance(trees[0], trees[1])
        assert rf1 == rf2


class TestDistanceBetween:
    @pytest.mark.parametrize(
        ("u", "v"),
        itertools.combinations([0, 1, 2, 3], 2),
    )
    def test_distance_between_sample(self, u, v):
        ts = msprime.sim_ancestry(
            2, sequence_length=10, recombination_rate=0.1, random_seed=42
        )
        test_tree = ts.first()
        assert test_tree.distance_between(u, v) == pytest.approx(
            ts.diversity([u, v], mode="branch", windows="trees")[0]
        )

    def test_distance_between_same_node(self):
        ts = msprime.sim_ancestry(
            2, sequence_length=10, recombination_rate=0.1, random_seed=42
        )
        test_tree = ts.first()
        assert test_tree.distance_between(0, 0) == 0

    def test_distance_between_nodes(self):
        # 4.00┊   8       ┊
        #     ┊ ┏━┻━┓     ┊
        # 3.00┊ ┃   7     ┊
        #     ┊ ┃ ┏━┻━┓   ┊
        # 2.00┊ ┃ ┃   6   ┊
        #     ┊ ┃ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃ ┃  5  ┊
        #     ┊ ┃ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 4 ┊
        #     0           1
        ts = tskit.Tree.generate_comb(5)
        assert ts.distance_between(1, 7) == 3.0
        assert ts.distance_between(6, 8) == 2.0

    def test_distance_between_invalid_nodes(self):
        ts = tskit.Tree.generate_comb(5)
        with pytest.raises(ValueError):
            ts.distance_between(0, 100)
