# MIT License
#
# Copyright (c) 2018-2020 Tskit Developers
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
Test cases for covariance computation.
"""
import io
import itertools
import unittest

import msprime
import numpy as np
import pytest

import tests.tsutil as tsutil
import tskit


def check_cov_tree_inputs(tree):
    if not len(tree.roots) == 1:
        raise ValueError("Trees must have one root")
    for u in tree.nodes():
        if tree.num_children(u) == 1:
            raise ValueError("Unary nodes are not supported")


def naive_tree_covariance(tree):
    """
    Returns the (branch) covariance matrix for the sample nodes in a tree. The
    covariance between a pair of nodes is the distance from the root to their
    most recent common ancestor.
    """
    samples = tree.tree_sequence.samples()
    check_cov_tree_inputs(tree)
    n = samples.shape[0]
    cov = np.zeros((n, n))
    for n1, n2 in itertools.combinations_with_replacement(range(n), 2):
        mrca = tree.mrca(samples[n1], samples[n2])
        cov[n1, n2] = tree.time(tree.root) - tree.time(mrca)
        cov[n2, n1] = cov[n1, n2]
    return cov


def naive_ts_covariance(ts):
    """
    Returns the (branch) covariance matrix for the sample nodes in a tree
    sequence. The covariance between a pair of nodes is the weighted sum of the
    tree covariance, with the weights given by the spans of the trees.
    """
    samples = ts.samples()
    n = samples.shape[0]
    cov = np.zeros((n, n))
    for tree in ts.trees():
        cov += naive_tree_covariance(tree) * tree.span
    return cov / ts.sequence_length


def genetic_relatedness(ts):
    # NOTE: I'm outputting a matrix here just for convenience; the proposal
    # is that the tskit method *not* output a matrix, and use the indices argument
    sample_sets = [[u] for u in ts.samples()]
    # sample_sets = [[0], [1]]
    n = len(sample_sets)
    num_samples = sum(map(len, sample_sets))

    def f(x):
        # x[i] gives the number of descendants in sample set i below the branch
        return np.array(
            [x[i] * x[j] * (sum(x) != num_samples) for i in range(n) for j in range(n)]
        )

    return ts.sample_count_stat(
        sample_sets,
        f,
        output_dim=n * n,
        mode="branch",
        span_normalise=True,
        polarised=True,
    ).reshape((n, n))


def c_genetic_relatedness(ts, sample_sets, indexes):
    n = len(indexes)
    state_dim = len(sample_sets)

    def f(x):
        # x[i] gives the number of descendants in sample set i below the branch
        sumx = 0
        num = 0
        c = 0
        for k in range(state_dim):
            sumx += x[k]

        for k in range(state_dim):
            num += len(sample_sets[k])

        if num != sumx:
            c = 1

        result = np.zeros(n)
        for k in range(n):
            i = indexes[k][0]
            j = indexes[k][1]
            result[k] = x[i] * x[j] * c

        return result

    return ts.sample_count_stat(
        sample_sets, f, output_dim=n, mode="branch", span_normalise=True, polarised=True
    )


class TestCovariance(unittest.TestCase):
    """
    Tests on covariance matrix computation
    """

    def verify(self, ts):
        cov1 = naive_ts_covariance(ts)
        cov2 = genetic_relatedness(ts)
        sample_sets = [[u] for u in ts.samples()]
        n = len(sample_sets)
        indexes = [
            (n1, n2) for n1, n2 in itertools.combinations_with_replacement(range(n), 2)
        ]
        cov3 = np.zeros((n, n))
        cov4 = np.zeros((n, n))
        i_upper = np.triu_indices(n)
        cov3[i_upper] = ts.genetic_relatedness(
            sample_sets, indexes, mode="branch", span_normalise=True
        )
        cov4[i_upper] = c_genetic_relatedness(ts, sample_sets, indexes)
        cov3 = np.maximum(cov3, cov3.transpose())
        cov4 = np.maximum(cov4, cov4.transpose())
        assert np.allclose(cov1, cov2)
        assert np.allclose(cov1, cov4)
        assert np.allclose(cov1, cov3)

    def verify_errors(self, ts):
        with pytest.raises(ValueError):
            naive_ts_covariance(ts)

    # def test_errors_unary_nodes(self):
    #     tables = tskit.TableCollection(sequence_length=2.0)

    #     sv = [True, False, False]
    #     tv = [0.0, 1.0, 2.0]
    #     for is_sample, t in zip(sv, tv):
    #         flags = tskit.NODE_IS_SAMPLE if is_sample else 0
    #         tables.nodes.add_row(flags=flags, time=t)

    #     lv = [0.0, 0.0, 0.0]
    #     rv = [1.0, 1.0, 1.0]
    #     pv = [1, 2]
    #     cv = [0, 1]
    #     for left, right, p, c in zip(lv, rv, pv, cv):
    #         tables.edges.add_row(left=left, right=right, parent=p, child=c)

    #     ts = tables.tree_sequence()
    #     self.verify_errors(ts)

    def test_errors_multiroot_tree(self):
        ts = msprime.simulate(15, random_seed=10)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        self.verify_errors(ts)

    def test_single_coalescent_tree(self):
        ts = msprime.simulate(10, random_seed=1, length=10)
        self.verify(ts)

    def test_coalescent_trees(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=1, length=2)
        assert ts.num_trees > 2
        self.verify(ts)

    @pytest.mark.skip(reason="internal samples not working")
    def test_internal_samples(self):
        nodes = io.StringIO(
            """\
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
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.2     0.8     8       3,5
        0.0     0.2     7       0,5
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        self.verify(ts)

    def validate_trees(self, n):
        for seed in range(1, 10):
            ts = msprime.simulate(n, random_seed=seed, recombination_rate=1)
            self.verify(ts)

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
            ts = msprime.simulate(
                n,
                random_seed=seed,
                demographic_events=demographic_events,
                recombination_rate=1,
            )
            # Check if this is really nonbinary
            found = False
            for edgeset in ts.edgesets():
                if len(edgeset.children) > 2:
                    found = True
                    break
            assert found

            self.verify(ts)

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
        self.verify(ts)
