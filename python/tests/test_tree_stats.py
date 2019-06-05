# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (C) 2016 University of Oxford
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
Test cases for generalized statistic computation.
"""
import io
import unittest
import random
import collections
import functools

import numpy as np
import numpy.testing as nt

import msprime

import tskit
import tests.tsutil as tsutil


def naive_general_branch_stats(ts, W, f, windows=None, polarised=False):
    n, K = W.shape
    if n != ts.num_samples:
        raise ValueError("First dimension of W must be number of samples")
    # Hack to determine M
    M = len(f(W[0]))
    total = np.sum(W, axis=0)

    sigma = np.zeros((ts.num_trees, M))
    for tree in ts.trees():
        X = np.zeros((ts.num_nodes, K))
        X[ts.samples()] = W
        for u in tree.nodes(order="postorder"):
            for v in tree.children(u):
                X[u] += X[v]
        if polarised:
            s = sum(tree.branch_length(u) * f(X[u]) for u in tree.nodes())
        else:
            s = sum(
                tree.branch_length(u) * (f(X[u]) + f(total - X[u]))
                for u in tree.nodes())
        sigma[tree.index] = s * tree.span
    if windows is None:
        return sigma
    else:
        bsc = tskit.BranchLengthStatCalculator(ts)
        return bsc.windowed_tree_stat(sigma, windows)


def general_branch_stats(ts, W, f, windows=None, polarised=False):
    # moved code over to tskit/stats.py
    bsc = tskit.BranchLengthStatCalculator(ts)
    return bsc.general_stat(W, f, windows=windows, polarised=polarised)


def naive_general_site_stats(ts, W, f, windows=None, polarised=False):
    n, K = W.shape
    if n != ts.num_samples:
        raise ValueError("First dimension of W must be number of samples")
    # Hack to determine M
    M = len(f(W[0]))
    sigma = np.zeros((ts.num_sites, M))
    for tree in ts.trees():
        X = np.zeros((ts.num_nodes, K))
        X[ts.samples()] = W
        for u in tree.nodes(order="postorder"):
            for v in tree.children(u):
                X[u] += X[v]
        for site in tree.sites():
            state_map = collections.defaultdict(functools.partial(np.zeros, K))
            state_map[site.ancestral_state] = sum(X[root] for root in tree.roots)
            for mutation in site.mutations:
                state_map[mutation.derived_state] += X[mutation.node]
                if mutation.parent != tskit.NULL:
                    parent = site.mutations[mutation.parent - site.mutations[0].id]
                    state_map[parent.derived_state] -= X[mutation.node]
                else:
                    state_map[site.ancestral_state] -= X[mutation.node]
            if polarised:
                del state_map[site.ancestral_state]
            sigma[site.id] += sum(map(f, state_map.values()))
    if windows is None:
        return sigma
    else:
        ssc = tskit.SiteStatCalculator(ts)
        return ssc.windowed_sitewise_stat(sigma, windows)


def general_site_stats(ts, W, f, windows=None, polarised=False):
    # moved code over to tskit/stats.py
    ssc = tskit.SiteStatCalculator(ts)
    return ssc.general_stat(W, f, windows=windows, polarised=polarised)


class TestGeneralBranchStats(unittest.TestCase):
    """
    Tests for general tree stats.
    """
    def run_stats(self, ts, W, f, windows=None, polarised=False):
        sigma2 = general_branch_stats(ts, W, f, windows, polarised=polarised)
        sigma1 = naive_general_branch_stats(ts, W, f, windows, polarised=polarised)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertTrue(np.allclose(sigma1, sigma2))
        return sigma1

    def test_simple_identity_f_w_zeros(self):
        ts = msprime.simulate(20, recombination_rate=3, random_seed=2)
        W = np.zeros((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: x, polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_simple_identity_f_w_ones(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        sigma = self.run_stats(ts, W, lambda x: x, polarised=True)
        self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))
        # A W of 1 for every node and identity f counts the samples in the subtree
        # if polarised is True.
        # print(sigma)
        for tree in ts.trees():
            s = tree.span * sum(
                tree.num_samples(u) * tree.branch_length(u) for u in tree.nodes())
            print(s)
            self.assertTrue(np.allclose(sigma[tree.index], s))

    def test_single_tree_cumsum_f_w_arange(self):
        ts = msprime.simulate(20, random_seed=7)
        W = np.arange(ts.num_samples * 2).reshape((ts.num_samples, 2))

        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: np.cumsum(x), polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))

    def test_simple_cumsum_f_w_ones(self):
        ts = msprime.simulate(20, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 8))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: np.cumsum(x), polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))

    def test_windows_equal_to_ts_breakpoints(self):
        ts = msprime.simulate(40, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 1))
        for polarised in [True, False]:
            sigma_no_windows = self.run_stats(
                ts, W, lambda x: np.cumsum(x), polarised=polarised)
            self.assertEqual(sigma_no_windows.shape, (ts.num_trees, W.shape[1]))
            sigma_windows = self.run_stats(
                ts, W, lambda x: np.cumsum(x), windows=np.array(list(ts.breakpoints())),
                polarised=polarised)
            self.assertEqual(sigma_windows.shape, sigma_no_windows.shape)
            self.assertTrue(np.allclose(sigma_windows.shape, sigma_no_windows.shape))

    def test_simple_identity_f_w_zeros_windows(self):
        ts = msprime.simulate(35, recombination_rate=3, random_seed=2)
        W = np.zeros((ts.num_samples, 3))
        windows = np.linspace(0, 1, num=11)
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: x, windows, polarised=polarised)
            self.assertEqual(sigma.shape, (10, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))


class TestGeneralSiteStats(unittest.TestCase):

    def run_stats(self, ts, W, f, windows=None, polarised=False):
        sigma1 = naive_general_site_stats(ts, W, f, windows, polarised=polarised)
        sigma2 = general_site_stats(ts, W, f, windows, polarised=polarised)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertTrue(np.allclose(sigma1, sigma2))
        return sigma1

    def test_identity_f_W_0_multiple_alleles(self):
        ts = msprime.simulate(20, recombination_rate=0, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.zeros((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: x, polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_identity_f_W_0_multiple_alleles_windows(self):
        ts = msprime.simulate(34, recombination_rate=0, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.zeros((ts.num_samples, 3))
        windows = np.linspace(0, 1, num=11)
        for polarised in [True, False]:
            sigma = self.run_stats(
                ts, W, lambda x: x, windows=windows, polarised=polarised)
            self.assertEqual(sigma.shape, (windows.shape[0] - 1, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_cumsum_f_W_1_multiple_alleles(self):
        ts = msprime.simulate(32, recombination_rate=2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.ones((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: np.cumsum(x), polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))

    def test_cumsum_f_W_1_two_alleles(self):
        ts = msprime.simulate(42, recombination_rate=2, mutation_rate=2, random_seed=1)
        W = np.ones((ts.num_samples, 5))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: np.cumsum(x))
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))


def path_length(tr, x, y):
    L = 0
    mrca = tr.mrca(x, y)
    for u in x, y:
        while u != mrca:
            L += tr.branch_length(u)
            u = tr.parent(u)
    return L


def windowed_tree_stat(ts, stat, windows):
    A = np.zeros((len(windows) - 1, stat.shape[1]))
    tree_breakpoints = np.array(list(ts.breakpoints()))
    tree_index = 0
    for j in range(len(windows) - 1):
        w_left = windows[j]
        w_right = windows[j + 1]
        while True:
            t_left = tree_breakpoints[tree_index]
            t_right = tree_breakpoints[tree_index + 1]
            left = max(t_left, w_left)
            right = min(t_right, w_right)
            weight = max(0.0, (right - left) / (t_right - t_left))
            A[j] += stat[tree_index] * weight
            assert left != right
            if t_right <= w_right:
                tree_index += 1
                # TODO This is inelegant - should include this in the case below
                if t_right == w_right:
                    break
            else:
                break
    # re-normalize by window lengths
    window_lengths = np.diff(windows)
    for j in range(len(windows) - 1):
        A[j] /= window_lengths[j]
    return A


def branch_general_stat(ts, sample_weights, summary_func, windows=None, polarised=False):
    """
    Efficient implementation of the algorithm used as the basis for the
    underlying C version.
    """
    n, state_dim = sample_weights.shape
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1

    # Determine result_dim
    result_dim = len(summary_func(sample_weights[0]))
    result = np.zeros((num_windows, result_dim))
    state = np.zeros((ts.num_nodes, state_dim))
    state[ts.samples()] = sample_weights
    total_weight = np.sum(sample_weights, axis=0)

    def area_weighted_summary(u):
        v = parent[u]
        branch_length = 0
        if v != -1:
            branch_length = time[v] - time[u]
        s = summary_func(state[u])
        if not polarised:
            s += summary_func(total_weight - state[u])
        return branch_length * s

    tree_index = 0
    window_index = 0
    time = ts.tables.nodes.time
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    running_sum = np.zeros(result_dim)
    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            u = edge.child
            running_sum -= area_weighted_summary(u)
            u = edge.parent
            while u != -1:
                running_sum -= area_weighted_summary(u)
                state[u] -= state[edge.child]
                running_sum += area_weighted_summary(u)
                u = parent[u]
            parent[edge.child] = -1

        for edge in edges_in:
            parent[edge.child] = edge.parent
            u = edge.child
            running_sum += area_weighted_summary(u)
            u = edge.parent
            while u != -1:
                running_sum -= area_weighted_summary(u)
                state[u] += state[edge.child]
                running_sum += area_weighted_summary(u)
                u = parent[u]

        # Update the windows
        assert window_index < num_windows
        while windows[window_index] < t_right:
            w_left = windows[window_index]
            w_right = windows[window_index + 1]
            left = max(t_left, w_left)
            right = min(t_right, w_right)
            weight = right - left
            assert weight > 0
            result[window_index] += running_sum * weight
            if w_right <= t_right:
                window_index += 1
            else:
                # This interval crosses a tree boundary, so we update it again in the
                # for the next tree
                break

        tree_index += 1

    # print("window_index:", window_index, windows.shape)
    assert window_index == windows.shape[0] - 1
    for j in range(num_windows):
        result[j] /= windows[j + 1] - windows[j]
    return result


class PythonNodeStatCalculator(object):
    """
    Python implementations of various "node" statistics --
    inefficient but more clear what they are doing.
    """

    def __init__(self, tree_sequence):
        self.tree_sequence = tree_sequence

    def divergence(self, X, Y, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros((len(windows) - 1, self.tree_sequence.num_nodes))
        tX = len(X)
        tY = len(Y)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = np.zeros(self.tree_sequence.num_nodes)
            for t1, t2 in zip(self.tree_sequence.trees(tracked_samples=X),
                              self.tree_sequence.trees(tracked_samples=Y)):
                if t1.interval[1] <= begin:
                    continue
                if t1.interval[0] >= end:
                    break
                SS = np.zeros(self.tree_sequence.num_nodes)
                for u in t1.nodes():
                    # count number of pairwise paths going through u
                    nX = t1.num_tracked_samples(u)
                    nY = t2.num_tracked_samples(u)
                    SS[u] += nX * (tY - nY) + (tX - nX) * nY
                S += SS*(min(end, t1.interval[1]) - max(begin, t1.interval[0]))
            out[j] = S/((end-begin)*len(X)*len(Y))
        return out

    def diversity(self, X, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros((len(windows) - 1, self.tree_sequence.num_nodes))
        tX = len(X)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = np.zeros(self.tree_sequence.num_nodes)
            for tr in self.tree_sequence.trees(tracked_samples=X):
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = np.zeros(self.tree_sequence.num_nodes)
                for u in tr.nodes():
                    # count number of pairwise paths going through u
                    n = tr.num_tracked_samples(u)
                    SS[u] += n * (tX - n)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            out[j] = 2 * S/((end-begin) * len(X) * (len(X) - 1))
        return out

    def naive_general_stat(self, W, f, windows=None, polarised=False):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        n, K = W.shape
        if n != self.tree_sequence.num_samples:
            raise ValueError("First dimension of W must be number of samples")
        M = self.tree_sequence.num_nodes
        total = np.sum(W, axis=0)

        sigma = np.zeros((self.tree_sequence.num_trees, M))
        for tree in self.tree_sequence.trees():
            X = np.zeros((self.tree_sequence.num_nodes, K))
            X[self.tree_sequence.samples()] = W
            for u in tree.nodes(order="postorder"):
                for v in tree.children(u):
                    X[u] += X[v]
            if polarised:
                s = np.array([f(X[u])
                              for u in range(self.tree_sequence.num_nodes)])
            else:
                s = np.array([f(X[u]) + f(total - X[u])
                              for u in range(self.tree_sequence.num_nodes)])
            sigma[tree.index] = s * tree.span
        if isinstance(windows, str) and windows == "treewise":
            # need to average across the windows
            for j, tree in enumerate(self.tree_sequence.trees()):
                sigma[j] /= tree.span
            return sigma
        else:
            return windowed_tree_stat(self.tree_sequence, sigma, windows)


class PythonBranchStatCalculator(object):
    """
    Python implementations of various ("tree") branch-length statistics -
    inefficient but more clear what they are doing.
    """

    def __init__(self, tree_sequence):
        self.tree_sequence = tree_sequence

    def divergence(self, X, Y, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = 0
            for tr in self.tree_sequence.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = 0
                for x in X:
                    for y in Y:
                        SS += path_length(tr, x, y)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            out[j] = S/((end-begin)*len(X)*len(Y))
        return out

    def diversity(self, X, windows=None):
        '''
        Computes average pairwise diversity between two random choices from x
        over the window specified.
        '''
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = 0
            for tr in self.tree_sequence.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = 0
                for x in X:
                    for y in set(X) - set([x]):
                        SS += path_length(tr, x, y)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            out[j] = S/((end - begin) * len(X)*(len(X) - 1))
        return out

    def Y3(self, X, Y, Z, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = 0
            for tr in self.tree_sequence.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                for x in X:
                    for y in Y:
                        for z in Z:
                            xy_mrca = tr.mrca(x, y)
                            xz_mrca = tr.mrca(x, z)
                            yz_mrca = tr.mrca(y, z)
                            if xy_mrca == xz_mrca:
                                #   /\
                                #  / /\
                                # x y  z
                                S += path_length(tr, x, yz_mrca) * this_length
                            elif xy_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # y x  z
                                S += path_length(tr, x, xz_mrca) * this_length
                            elif xz_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # z x  y
                                S += path_length(tr, x, xy_mrca) * this_length
            out[j] = S/((end - begin) * len(X) * len(Y) * len(Z))
        return out

    def Y2(self, X, Y, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = 0
            for tr in self.tree_sequence.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                for x in X:
                    for y in Y:
                        for z in set(Y) - {y}:
                            xy_mrca = tr.mrca(x, y)
                            xz_mrca = tr.mrca(x, z)
                            yz_mrca = tr.mrca(y, z)
                            if xy_mrca == xz_mrca:
                                #   /\
                                #  / /\
                                # x y  z
                                S += path_length(tr, x, yz_mrca) * this_length
                            elif xy_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # y x  z
                                S += path_length(tr, x, xz_mrca) * this_length
                            elif xz_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # z x  y
                                S += path_length(tr, x, xy_mrca) * this_length
            out[j] = S/((end - begin) * len(X) * len(Y) * (len(Y)-1))
        return out

    def Y1(self, X, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = 0
            for tr in self.tree_sequence.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                for x in X:
                    for y in set(X) - {x}:
                        for z in set(X) - {x, y}:
                            xy_mrca = tr.mrca(x, y)
                            xz_mrca = tr.mrca(x, z)
                            yz_mrca = tr.mrca(y, z)
                            if xy_mrca == xz_mrca:
                                #   /\
                                #  / /\
                                # x y  z
                                S += path_length(tr, x, yz_mrca) * this_length
                            elif xy_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # y x  z
                                S += path_length(tr, x, xz_mrca) * this_length
                            elif xz_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # z x  y
                                S += path_length(tr, x, xy_mrca) * this_length
            out[j] = S/((end - begin) * len(X) * (len(X)-1) * (len(X)-2))
        return out

    def f4(self, A, B, C, D, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            for U in A, B, C, D:
                if max([U.count(x) for x in set(U)]) > 1:
                    raise ValueError("A,B,C, and D cannot contain repeated elements.")
            S = 0
            for tr in self.tree_sequence.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                SS = 0
                for a in A:
                    for b in B:
                        for c in C:
                            for d in D:
                                SS += path_length(tr, tr.mrca(a, c), tr.mrca(b, d))
                                SS -= path_length(tr, tr.mrca(a, d), tr.mrca(b, c))
                S += SS * this_length
            out[j] = S / ((end - begin) * len(A) * len(B) * len(C) * len(D))
        return out

    def f3(self, A, B, C, windows=None):
        # this is f4(A,B;A,C) but drawing distinct samples from A
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            assert(len(A) > 1)
            for U in A, B, C:
                if max([U.count(x) for x in set(U)]) > 1:
                    raise ValueError("A, B and C cannot contain repeated elements.")
            S = 0
            for tr in self.tree_sequence.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                SS = 0
                for a in A:
                    for b in B:
                        for c in set(A) - {a}:
                            for d in C:
                                SS += path_length(tr, tr.mrca(a, c), tr.mrca(b, d))
                                SS -= path_length(tr, tr.mrca(a, d), tr.mrca(b, c))
                S += SS * this_length
            out[j] = S / ((end - begin) * len(A) * (len(A) - 1) * len(B) * len(C))
        return out

    def f2(self, A, B, windows=None):
        # this is f4(A,B;A,B) but drawing distinct samples from A and B
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            assert(len(A) > 1)
            for U in A, B:
                if max([U.count(x) for x in set(U)]) > 1:
                    raise ValueError("A and B cannot contain repeated elements.")
            S = 0
            for tr in self.tree_sequence.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                SS = 0
                for a in A:
                    for b in B:
                        for c in set(A) - {a}:
                            for d in set(B) - {b}:
                                SS += path_length(tr, tr.mrca(a, c), tr.mrca(b, d))
                                SS -= path_length(tr, tr.mrca(a, d), tr.mrca(b, c))
                S += SS * this_length
            out[j] = S / ((end - begin) * len(A) * (len(A) - 1) * len(B) * (len(B) - 1))
        return out

    def sample_count_stats(self, sample_sets, f, windows=None, polarised=False):
        '''
        Here sample_sets is a list of lists of samples, and f is a function
        whose argument is a list of integers of the same length as sample_sets
        that returns a list of numbers; there will be one output for each element.
        For each value, each branch in a tree is weighted by f(x),
        where x[i] is the number of samples in sample_sets[i] below that
        branch.  This finds the sum of all counted branches for each tree,
        and averages this across the tree sequence ts, weighted by genomic length.

        This version is inefficient as it iterates over all nodes in each tree.
        '''
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        for U in sample_sets:
            if max([U.count(x) for x in set(U)]) > 1:
                raise ValueError("elements of sample_sets",
                                 "cannot contain repeated elements.")
        W = np.array([[float(u in A) for A in sample_sets]
                      for u in self.tree_sequence.samples()])
        S = self.naive_general_stat(W, f, windows=windows)
        return S

    def naive_general_stat(self, W, f, windows=None, polarised=False):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        n, K = W.shape
        if n != self.tree_sequence.num_samples:
            raise ValueError("First dimension of W must be number of samples")
        # Hack to determine M
        M = len(f(W[0]))
        total = np.sum(W, axis=0)

        sigma = np.zeros((self.tree_sequence.num_trees, M))
        for tree in self.tree_sequence.trees():
            X = np.zeros((self.tree_sequence.num_nodes, K))
            X[self.tree_sequence.samples()] = W
            for u in tree.nodes(order="postorder"):
                for v in tree.children(u):
                    X[u] += X[v]
            if polarised:
                s = sum(tree.branch_length(u) * f(X[u]) for u in tree.nodes())
            else:
                s = sum(
                    tree.branch_length(u) * (f(X[u]) + f(total - X[u]))
                    for u in tree.nodes())
            sigma[tree.index] = s * tree.span
        if isinstance(windows, str) and windows == "treewise":
            # need to average across the windows
            for j, tree in enumerate(self.tree_sequence.trees()):
                sigma[j] /= tree.span
            return sigma
        else:
            return windowed_tree_stat(self.tree_sequence, sigma, windows)

    def site_frequency_spectrum(self, sample_set, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        n_out = len(sample_set)
        out = np.zeros((n_out, len(windows) - 1))
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = [0.0 for j in range(n_out)]
            for t in self.tree_sequence.trees(tracked_samples=sample_set,
                                              sample_counts=True):
                root = t.root
                tr_len = min(end, t.interval[1]) - max(begin, t.interval[0])
                if tr_len > 0:
                    for node in t.nodes():
                        if node != root:
                            x = t.num_tracked_samples(node)
                            if x > 0:
                                S[x - 1] += t.branch_length(node) * tr_len
            for j in range(n_out):
                S[j] /= (end-begin)
            out[j] = S
        return(out)


class PythonSiteStatCalculator(object):
    """
    Python implementations of various single-site statistics -
    inefficient but more clear what they are doing.
    """

    def __init__(self, tree_sequence):
        self.tree_sequence = tree_sequence

    def divergence(self, X, Y, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            haps = list(self.tree_sequence.haplotypes())
            site_positions = [x.position for x in self.tree_sequence.sites()]
            S = 0
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    for x in X:
                        for y in Y:
                            if (haps[x][k] != haps[y][k]):
                                # x|y
                                S += 1
            out[j] = S/((end - begin) * len(X) * len(Y))
        return out

    def Y3(self, X, Y, Z, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            haps = list(self.tree_sequence.haplotypes())
            site_positions = [x.position for x in self.tree_sequence.sites()]
            S = 0
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    for x in X:
                        for y in Y:
                            for z in Z:
                                if ((haps[x][k] != haps[y][k])
                                   and (haps[x][k] != haps[z][k])):
                                    # x|yz
                                    S += 1
            out[j] = S/((end - begin) * len(X) * len(Y) * len(Z))
        return out

    def Y2(self, X, Y, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            haps = list(self.tree_sequence.haplotypes())
            site_positions = [x.position for x in self.tree_sequence.sites()]
            S = 0
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    for x in X:
                        for y in Y:
                            for z in set(Y) - {y}:
                                if ((haps[x][k] != haps[y][k])
                                   and (haps[x][k] != haps[z][k])):
                                    # x|yz
                                    S += 1
            out[j] = S/((end - begin) * len(X) * len(Y) * (len(Y) - 1))
        return out

    def Y1(self, X, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            haps = list(self.tree_sequence.haplotypes())
            site_positions = [x.position for x in self.tree_sequence.sites()]
            S = 0
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    for x in X:
                        for y in set(X) - {x}:
                            for z in set(X) - {x, y}:
                                if ((haps[x][k] != haps[y][k])
                                   and (haps[x][k] != haps[z][k])):
                                    # x|yz
                                    S += 1
            out[j] = S/((end - begin) * len(X) * (len(X) - 1) * (len(X) - 2))
        return out

    def f4(self, A, B, C, D, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            for U in A, B, C, D:
                if max([U.count(x) for x in set(U)]) > 1:
                    raise ValueError("A,B,C, and D cannot contain repeated elements.")
            haps = list(self.tree_sequence.haplotypes())
            site_positions = [x.position for x in self.tree_sequence.sites()]
            S = 0
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    for a in A:
                        for b in B:
                            for c in C:
                                for d in D:
                                    if ((haps[a][k] == haps[c][k])
                                       and (haps[a][k] != haps[d][k])
                                       and (haps[a][k] != haps[b][k])):
                                        # ac|bd
                                        S += 1
                                    elif ((haps[a][k] == haps[d][k])
                                          and (haps[a][k] != haps[c][k])
                                          and (haps[a][k] != haps[b][k])):
                                        # ad|bc
                                        S -= 1
            out[j] = S / ((end - begin) * len(A) * len(B) * len(C) * len(D))
        return out

    def f3(self, A, B, C, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            for U in A, B, C:
                if max([U.count(x) for x in set(U)]) > 1:
                    raise ValueError("A,B,and C cannot contain repeated elements.")
            haps = list(self.tree_sequence.haplotypes())
            site_positions = [x.position for x in self.tree_sequence.sites()]
            S = 0
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    for a in A:
                        for b in B:
                            for c in set(A) - {a}:
                                for d in C:
                                    if ((haps[a][k] == haps[c][k])
                                       and (haps[a][k] != haps[d][k])
                                       and (haps[a][k] != haps[b][k])):
                                        # ac|bd
                                        S += 1
                                    elif ((haps[a][k] == haps[d][k])
                                          and (haps[a][k] != haps[c][k])
                                          and (haps[a][k] != haps[b][k])):
                                        # ad|bc
                                        S -= 1
            out[j] = S / ((end - begin) * len(A) * len(B) * len(C) * (len(A) - 1))
        return out

    def f2(self, A, B, windows=None):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        out = np.zeros(len(windows) - 1)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            for U in A, B:
                if max([U.count(x) for x in set(U)]) > 1:
                    raise ValueError("A,and B cannot contain repeated elements.")
            haps = list(self.tree_sequence.haplotypes())
            site_positions = [x.position for x in self.tree_sequence.sites()]
            S = 0
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    for a in A:
                        for b in B:
                            for c in set(A) - {a}:
                                for d in set(B) - {b}:
                                    if ((haps[a][k] == haps[c][k])
                                       and (haps[a][k] != haps[d][k])
                                       and (haps[a][k] != haps[b][k])):
                                        # ac|bd
                                        S += 1
                                    elif ((haps[a][k] == haps[d][k])
                                          and (haps[a][k] != haps[c][k])
                                          and (haps[a][k] != haps[b][k])):
                                        # ad|bc
                                        S -= 1
            out[j] = S / ((end - begin) * len(A) * len(B)
                          * (len(A) - 1) * (len(B) - 1))
        return out

    def sample_count_stats(self, sample_sets, f, windows=None, polarised=False):
        '''
        Here sample_sets is a list of lists of samples, and f is a function
        whose argument is a list of integers of the same length as sample_sets
        that returns a list of numbers; there will be one output for each element.
        For each value, each allele in a tree is weighted by f(x), where
        x[i] is the number of samples in sample_sets[i] that inherit that allele.
        This finds the sum of this value for all alleles at all polymorphic sites,
        and across the tree sequence ts, weighted by genomic length.

        This version is inefficient as it works directly with haplotypes.
        '''
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        for U in sample_sets:
            if max([U.count(x) for x in set(U)]) > 1:
                raise ValueError("elements of sample_sets",
                                 "cannot contain repeated elements.")
        haps = list(self.tree_sequence.haplotypes())
        n_out = len(f([0 for a in sample_sets]))
        out = np.zeros((n_out, len(windows) - 1))
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            site_positions = [x.position for x in self.tree_sequence.sites()]
            S = [0.0 for j in range(n_out)]
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    all_g = [haps[j][k] for j in range(self.tree_sequence.num_samples)]
                    g = [[haps[j][k] for j in u] for u in sample_sets]
                    for a in set(all_g):
                        x = [h.count(a) for h in g]
                        w = f(x)
                        for j in range(n_out):
                            S[j] += w[j]
            for j in range(n_out):
                S[j] /= (end - begin)
            out[j] = np.array([S])
        return out

    def naive_general_stat(self, W, f, windows=None, polarised=False):
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        n, K = W.shape
        if n != self.tree_sequence.num_samples:
            raise ValueError("First dimension of W must be number of samples")
        # Hack to determine M
        M = len(f(W[0]))
        sigma = np.zeros((self.tree_sequence.num_sites, M))
        for tree in self.tree_sequence.trees():
            X = np.zeros((self.tree_sequence.num_nodes, K))
            X[self.tree_sequence.samples()] = W
            for u in tree.nodes(order="postorder"):
                for v in tree.children(u):
                    X[u] += X[v]
            for site in tree.sites():
                state_map = collections.defaultdict(functools.partial(np.zeros, K))
                state_map[site.ancestral_state] = sum(X[root] for root in tree.roots)
                for mutation in site.mutations:
                    state_map[mutation.derived_state] += X[mutation.node]
                    if mutation.parent != tskit.NULL:
                        parent = site.mutations[mutation.parent - site.mutations[0].id]
                        state_map[parent.derived_state] -= X[mutation.node]
                    else:
                        state_map[site.ancestral_state] -= X[mutation.node]
                if polarised:
                    del state_map[site.ancestral_state]
                sigma[site.id] += sum(map(f, state_map.values()))
        if isinstance(windows, str) and windows == "sitewise":
            return sigma
        else:
            return self.tree_sequence.windowed_sitewise_stat(sigma, windows)

    def site_frequency_spectrum(self, sample_set, windows=None):
        '''
        '''
        if windows is None:
            windows = [0.0, self.tree_sequence.sequence_length]
        haps = list(self.tree_sequence.haplotypes())
        site_positions = [x.position for x in self.tree_sequence.sites()]
        n_out = len(sample_set)
        out = np.zeros((n_out, len(windows) - 1))
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = [0.0 for j in range(n_out)]
            for k in range(self.tree_sequence.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    all_g = [haps[j][k] for j in range(self.tree_sequence.num_samples)]
                    g = [haps[j][k] for j in sample_set]
                    for a in set(all_g):
                        x = g.count(a)
                        if x > 0:
                            S[x - 1] += 1.0
            for j in range(n_out):
                S[j] /= (end - begin)
            out[j] = S
        return out


def upper_tri_to_matrix(x):
    """
    Given x, a vector of entries of the upper triangle of a matrix
    in row-major order, including the diagonal, return the corresponding matrix.
    """
    # n^2 + n = 2 u => n = (-1 + sqrt(1 + 8*u))/2
    n = int((np.sqrt(1 + 8 * len(x)) - 1)/2.0)
    out = np.ones((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            out[i, j] = out[j, i] = x[k]
            k += 1
    return out


class StatsTestCase(unittest.TestCase):
    """
    Provides convenience functions.
    """

    def assertListAlmostEqual(self, x, y):
        self.assertEqual(len(x), len(y))
        for a, b in zip(x, y):
            self.assertAlmostEqual(a, b)

    def assertArrayEqual(self, x, y):
        nt.assert_equal(x, y)

    def assertArrayAlmostEqual(self, x, y):
        nt.assert_array_almost_equal(x, y)


class TestWindowedTreeStat(StatsTestCase):
    """
    Tests that the treewise windowing function defined here has the correct
    behaviour.
    """
    # TODO add more tests here covering the various windowing possibilities.
    def get_tree_sequence(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=1)
        self.assertGreater(ts.num_trees, 3)
        return ts

    def test_all_trees(self):
        ts = self.get_tree_sequence()
        A1 = np.ones((ts.num_trees, 1))
        windows = np.array(list(ts.breakpoints()))
        A2 = windowed_tree_stat(ts, A1, windows)
        # print("breakpoints = ", windows)
        # print(A2)
        self.assertEqual(A1.shape, A2.shape)
        # JK: I don't understand what we're computing here, this normalisation
        # seems pretty weird.
        # for tree in ts.trees():
        #     self.assertAlmostEqual(A2[tree.index, 0], tree.span / ts.sequence_length)

    def test_single_interval(self):
        ts = self.get_tree_sequence()
        A1 = np.ones((ts.num_trees, 1))
        windows = np.array([0, ts.sequence_length])
        A2 = windowed_tree_stat(ts, A1, windows)
        self.assertEqual(A2.shape, (1, 1))
        # TODO: Test output


class TestGeneralBranchStats(StatsTestCase):
    """
    Tests for general tree stats.
    """
    def run_stats(self, ts, W, f, windows=None, polarised=False):
        py_bsc = PythonBranchStatCalculator(ts)
        sigma1 = py_bsc.naive_general_stat(W, f, windows, polarised=polarised)
        sigma2 = ts.branch_general_stat(W, f, windows, polarised=polarised)
        sigma3 = branch_general_stat(ts, W, f, windows, polarised=polarised)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertEqual(sigma1.shape, sigma3.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)
        self.assertArrayAlmostEqual(sigma1, sigma3)
        return sigma1

    def test_simple_identity_f_w_zeros(self):
        ts = msprime.simulate(12, recombination_rate=3, random_seed=2)
        W = np.zeros((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: x, windows="treewise",
                                   polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_simple_identity_f_w_ones(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        sigma = self.run_stats(ts, W, lambda x: x, windows="treewise", polarised=True)
        self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))
        # A W of 1 for every node and identity f counts the samples in the subtree
        # if polarised is True.
        for tree in ts.trees():
            s = sum(tree.num_samples(u) * tree.branch_length(u) for u in tree.nodes())
            self.assertTrue(np.allclose(sigma[tree.index], s))

    def test_simple_cumsum_f_w_ones(self):
        ts = msprime.simulate(13, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 8))
        for polarised in [True, False]:
            sigma = self.run_stats(
                ts, W, lambda x: np.cumsum(x), windows="treewise", polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))

    def test_simple_cumsum_f_w_ones_many_windows(self):
        ts = msprime.simulate(15, recombination_rate=3, random_seed=3)
        self.assertGreater(ts.num_trees, 3)
        windows = np.linspace(0, ts.sequence_length, num=ts.num_trees * 10)
        W = np.ones((ts.num_samples, 3))
        sigma = self.run_stats(ts, W, lambda x: np.cumsum(x), windows=windows)
        self.assertEqual(sigma.shape, (windows.shape[0] - 1, W.shape[1]))

    def test_windows_equal_to_ts_breakpoints(self):
        ts = msprime.simulate(14, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 1))
        for polarised in [True, False]:
            sigma_no_windows = self.run_stats(
                ts, W, lambda x: np.cumsum(x), windows="treewise", polarised=polarised)
            self.assertEqual(sigma_no_windows.shape, (ts.num_trees, W.shape[1]))
            sigma_windows = self.run_stats(
                ts, W, lambda x: np.cumsum(x), windows=ts.breakpoints(as_array=True),
                polarised=polarised)
            self.assertEqual(sigma_windows.shape, sigma_no_windows.shape)
            self.assertTrue(np.allclose(sigma_windows.shape, sigma_no_windows.shape))

    def test_single_tree_windows(self):
        ts = msprime.simulate(15, random_seed=2, length=100)
        W = np.ones((ts.num_samples, 2))
        # for num_windows in range(1, 10):
        for num_windows in [2]:
            windows = np.linspace(0, ts.sequence_length, num=num_windows + 1)
            sigma = self.run_stats(ts, W, lambda x: np.array([np.sum(x)]), windows)
            self.assertEqual(sigma.shape, (num_windows, 1))

    def test_simple_identity_f_w_zeros_windows(self):
        ts = msprime.simulate(15, recombination_rate=3, random_seed=2)
        W = np.zeros((ts.num_samples, 3))
        windows = np.linspace(0, ts.sequence_length, num=11)
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: x, windows, polarised=polarised)
            self.assertEqual(sigma.shape, (10, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))


class TestGeneralSiteStats(StatsTestCase):

    def run_stats(self, ts, W, f, windows=None, polarised=False):
        py_ssc = PythonSiteStatCalculator(ts)
        sigma1 = py_ssc.naive_general_stat(W, f, windows, polarised=polarised)
        sigma2 = ts.general_stat("site", W, f, windows, polarised=polarised)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)
        return sigma1

    def test_identity_f_W_0_multiple_alleles(self):
        ts = msprime.simulate(20, recombination_rate=0, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.zeros((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: x, windows="sitewise",
                                   polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_identity_f_W_0_multiple_alleles_windows(self):
        ts = msprime.simulate(34, recombination_rate=0, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.zeros((ts.num_samples, 3))
        windows = np.linspace(0, 1, num=11)
        for polarised in [True, False]:
            sigma = self.run_stats(
                ts, W, lambda x: x, windows=windows, polarised=polarised)
            self.assertEqual(sigma.shape, (windows.shape[0] - 1, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_cumsum_f_W_1_multiple_alleles(self):
        ts = msprime.simulate(3, recombination_rate=2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.ones((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: np.cumsum(x),
                                   windows="sitewise", polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))

    def test_cumsum_f_W_1_two_alleles(self):
        ts = msprime.simulate(42, recombination_rate=2, mutation_rate=2, random_seed=1)
        W = np.ones((ts.num_samples, 5))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: np.cumsum(x), windows="sitewise")
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))


class TestGeneralNodeStats(StatsTestCase):

    def run_stats(self, ts, W, f, windows=None, polarised=False):
        py_ssc = PythonNodeStatCalculator(ts)
        sigma1 = py_ssc.naive_general_stat(W, f, windows, polarised=polarised)
        sigma2 = ts.general_stat("node", W, f, windows, polarised=polarised)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)
        return sigma1

    def test_simple_sum_f_w_zeros(self):
        ts = msprime.simulate(12, recombination_rate=3, random_seed=2)
        W = np.zeros((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: sum(x), windows="treewise",
                                   polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, ts.num_nodes))
            self.assertTrue(np.all(sigma == 0))

    def test_simple_sum_f_w_ones(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        sigma = self.run_stats(ts, W, lambda x: sum(x),
                               windows="treewise", polarised=True)
        self.assertEqual(sigma.shape, (ts.num_trees, ts.num_nodes))
        # A W of 1 for every node and f(x)=sum(x) counts the samples in the subtree
        # times 2 if polarised is True.
        for tree in ts.trees():
            s = np.array([tree.num_samples(u) for u in range(ts.num_nodes)])
            self.assertArrayAlmostEqual(sigma[tree.index], 2*s)


class GeneralStatsTestCase(StatsTestCase):
    """
    Tests of statistic computation.  Derived classes should have an attribute
    `stat_type` and `rng`.
    """

    random_seed = 123456

    def compare_stats(self, ts, py_fn, ts_fn,
                      sample_sets, index_length):
        # will compare py_fn() to ts_fn(..., stat_type)
        assert(len(sample_sets) >= index_length)
        windows = [k * ts.sequence_length / 20 for k in
                   [0] + sorted(self.rng.sample(range(1, 20), 4)) + [20]]
        indices = [self.rng.sample(range(len(sample_sets)), max(1, index_length))
                   for _ in range(5)]
        ts_vals = ts_fn(sample_sets, indices, windows, stat_type=self.stat_type)
        self.assertEqual((len(windows) - 1, len(indices)), ts_vals.shape)
        leafset_args = [[sample_sets[i] for i in ii] for ii in indices]
        py_vals = np.column_stack([py_fn(*a, windows) for a in leafset_args])
        self.assertEqual(py_vals.shape, ts_vals.shape)
        self.assertArrayAlmostEqual(py_vals, ts_vals)

    def compare_sfs(self, ts, tree_fn, sample_sets, tsc_fn):
        for sample_set in sample_sets:
            windows = [k * ts.sequence_length / 20 for k in
                       [0] + sorted(self.rng.sample(range(1, 20), 4)) + [20]]
            win_args = [{'begin': windows[i], 'end': windows[i+1]}
                        for i in range(len(windows)-1)]
            tree_vals = [tree_fn(sample_set, **b) for b in win_args]

            tsc_vals = tsc_fn(sample_set, windows)
            self.assertEqual(len(tsc_vals), len(windows) - 1)
            for i in range(len(windows) - 1):
                self.assertListAlmostEqual(tsc_vals[i], tree_vals[i])

    def check_tree_stat_interface(self, ts):
        samples = list(ts.samples())

        def f(x):
            return [1]

        # empty sample sets will raise an error
        self.assertRaises(ValueError, ts.sample_count_stats,
                          self.stat_type, samples[0:2] + [], f)
        # sample_sets must be lists without repeated elements
        self.assertRaises(ValueError, ts.sample_count_stats,
                          self.stat_type, samples[0:2], f)
        self.assertRaises(ValueError, ts.sample_count_stats,
                          self.stat_type, [samples[0:2], [samples[2], samples[2]]], f)
        # and must all be samples
        self.assertRaises(ValueError, ts.sample_count_stats,
                          self.stat_type, [samples[0:2], [max(samples)+1]], f)
        # windows must start at 0.0, be increasing, and extend to the end
        self.assertRaises(ValueError, ts.sample_count_stats,
                          self.stat_type, [samples[0:2], samples[2:4]], f,
                          windows=[0.1, ts.sequence_length])
        self.assertRaises(ValueError, ts.sample_count_stats,
                          self.stat_type, [samples[0:2], samples[2:4]], f,
                          windows=[0.0, 0.8*ts.sequence_length])
        self.assertRaises(ValueError, ts.sample_count_stats,
                          self.stat_type, [samples[0:2], samples[2:4]], f,
                          windows=[0.0, 0.8*ts.sequence_length,
                                   0.4*ts.sequence_length, ts.sequence_length])

    def check_sfs_interface(self, ts):
        samples = ts.samples()

        # empty sample sets will raise an error
        self.assertRaises(ValueError, ts.site_frequency_spectrum, [],
                          self.stat_type)
        # sample_sets must be lists without repeated elements
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          [samples[2], samples[2]], self.stat_type)
        # and must all be samples
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          [samples[0], max(samples)+1], self.stat_type)
        # windows must start at 0.0, be increasing, and extend to the end
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          samples[0:2], [0.1, ts.sequence_length],
                          self.stat_type)
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          samples[0:2], [0.0, 0.8*ts.sequence_length],
                          self.stat_type)
        self.assertRaises(ValueError, ts.site_frequency_spectrum, samples[0:2],
                          [0.0, 0.8*ts.sequence_length, 0.4*ts.sequence_length,
                           ts.sequence_length], self.stat_type)

    def check_sfs(self, ts):
        # check site frequency spectrum
        self.check_sfs_interface(ts)
        A = [self.rng.sample(list(ts.samples()), 2),
             self.rng.sample(list(ts.samples()), 4),
             self.rng.sample(list(ts.samples()), 8),
             self.rng.sample(list(ts.samples()), 10),
             self.rng.sample(list(ts.samples()), 12)]
        py_tsc = self.py_stat_class(ts)

        self.compare_sfs(ts, py_tsc.site_frequency_spectrum, A,
                         ts.site_frequency_spectrum)

    def check_f_interface(self, ts):
        # sample sets must have at least two samples
        self.assertRaises(ValueError, ts.f2,
                          [[0, 1], [3]], [(0, 1)], [0, ts.sequence_length],
                          self.stat_type)

    def check_f_stats(self, ts):
        self.check_f_interface(ts)
        samples = self.rng.sample(list(ts.samples()), 12)
        A = [[samples[0], samples[1], samples[2]],
             [samples[3], samples[4]],
             [samples[5], samples[6]],
             [samples[7], samples[8]],
             [samples[9], samples[10], samples[11]]]
        py_tsc = self.py_stat_class(ts)
        self.compare_stats(ts, py_tsc.f2, ts.f2,
                           self.stat_type, A, 2)
        self.compare_stats(ts, py_tsc.f3, ts.f3,
                           self.stat_type, A, 3)
        self.compare_stats(ts, py_tsc.f4, ts.f4,
                           self.stat_type, A, 4)

    def check_Y_stat(self, ts):
        samples = self.rng.sample(list(ts.samples()), 12)
        A = [[samples[0], samples[1], samples[6]],
             [samples[2], samples[3], samples[7]],
             [samples[4], samples[5], samples[8]],
             [samples[9], samples[10], samples[11]]]
        py_tsc = self.py_stat_class(ts)
        self.compare_stats(ts, py_tsc.Y3, ts.Y3,
                           self.stat_type, A, 3)
        self.compare_stats(ts, py_tsc.Y2, ts.Y2,
                           self.stat_type, A, 2)
        self.compare_stats(ts, py_tsc.Y1, ts.Y1,
                           self.stat_type, A, 0)

    def check_divergence(self, ts):
        samples = self.rng.sample(list(ts.samples()), 12)
        A = [[samples[0], samples[1], samples[6]],
             [samples[2], samples[3], samples[7]],
             [samples[4], samples[5], samples[8]],
             [samples[9], samples[10], samples[11]]]
        py_tsc = self.py_stat_class(ts)
        self.compare_stats(ts, py_tsc.divergence, ts.divergence, A, 2)
        self.compare_stats(ts, py_tsc.diversity, ts.diversity, A, 1)


class SpecificTreesTestCase(GeneralStatsTestCase):
    seed = 21

    def test_case_1(self):
        # With mutations:
        #
        # 1.0          6
        # 0.7         / \                                    5
        #            /   X                                  / \
        # 0.5       X     4                4               /   4
        #          /     / \              / \             /   X X
        # 0.4     X     X   \            X   3           X   /   \
        #        /     /     X          /   / X         /   /     \
        # 0.0   0     1       2        1   0   2       0   1       2
        #          (0.0, 0.2),        (0.2, 0.8),       (0.8, 1.0)
        #
        branch_true_diversity_01 = 2*(1 * (0.2-0) +
                                      0.5 * (0.8-0.2) + 0.7 * (1.0-0.8))
        branch_true_diversity_02 = 2*(1 * (0.2-0) +
                                      0.4 * (0.8-0.2) + 0.7 * (1.0-0.8))
        branch_true_diversity_12 = 2*(0.5 * (0.2-0) +
                                      0.5 * (0.8-0.2) + 0.5 * (1.0-0.8))
        branch_true_Y = 0.2*(1 + 0.5) + 0.6*(0.4) + 0.2*(0.7+0.2)
        site_true_Y = 3 + 0 + 1
        node_true_diversity_012 = np.array([
                0.2 * np.array([2, 2, 2, 0, 2, 0, 0]) +
                0.6 * np.array([2, 2, 2, 2, 0, 0, 0]) +
                0.2 * np.array([2, 2, 2, 0, 2, 0, 0])]) / 3
        node_true_divergence_0_12 = np.array([
                0.2 * np.array([2, 1, 1, 0, 2, 0, 0]) +
                0.6 * np.array([2, 1, 1, 1, 0, 0, 0]) +
                0.2 * np.array([2, 1, 1, 0, 2, 0, 0])]) / 2

        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           0.4
        4       0           0.5
        5       0           0.7
        6       0           1.0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.2     0.8     3       0,2
        0.0     0.2     4       1,2
        0.2     0.8     4       1,3
        0.8     1.0     4       1,2
        0.8     1.0     5       0,4
        0.0     0.2     6       0,4
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.05        0
        1   0.1         0
        2   0.11        0
        3   0.15        0
        4   0.151       0
        5   0.3         0
        6   0.6         0
        7   0.9         0
        8   0.95        0
        9   0.951       0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state
        0       4       1
        1       0       1
        2       2       1
        3       0       1
        4       1       1
        5       1       1
        6       2       1
        7       0       1
        8       1       1
        9       2       1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            strict=False)
        py_bsc = PythonBranchStatCalculator(ts)
        py_ssc = PythonSiteStatCalculator(ts)
        py_nsc = PythonNodeStatCalculator(ts)

        # diversity between 0 and 1
        A = [[0], [1]]
        n = [len(a) for a in A]

        def f(x):
            return np.array([float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])/(2*n[0]*n[1])])

        # tree lengths:
        self.assertAlmostEqual(py_bsc.divergence([0], [1]),
                               branch_true_diversity_01)
        self.assertAlmostEqual(py_bsc.sample_count_stats(A, f)[0][0],
                               branch_true_diversity_01)
        self.assertAlmostEqual(ts.sample_count_stats("branch", A, f)[0][0],
                               branch_true_diversity_01)
        self.assertAlmostEqual(
                ts.diversity([[0, 1]], stat_type="branch")[0][0],
                branch_true_diversity_01)

        # mean diversity between [0, 1] and [0, 2]:
        branch_true_mean_diversity = (0 + branch_true_diversity_02
                                      + branch_true_diversity_01
                                      + branch_true_diversity_12)/4
        A = [[0, 1], [0, 2]]
        n = [len(a) for a in A]

        def f(x):
            return np.array([float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])/8.0])

        # tree lengths:
        self.assertAlmostEqual(py_bsc.divergence(A[0], A[1]),
                               branch_true_mean_diversity)
        self.assertAlmostEqual(py_bsc.sample_count_stats(A, f)[0][0],
                               branch_true_mean_diversity)
        self.assertAlmostEqual(ts.sample_count_stats("branch", A, f)[0][0],
                               branch_true_mean_diversity)

        # Y-statistic for (0/12)
        A = [[0], [1, 2]]

        def f(x):
            return np.array([float(((x[0] == 1) and (x[1] == 0))
                                   or ((x[0] == 0) and (x[1] == 2)))/2.0])

        # tree lengths:
        bts_Y = ts.Y3([[0], [1], [2]], windows=[0.0, 1.0],
                      stat_type="branch")[0][0]
        py_bsc_Y = py_bsc.Y3([0], [1], [2], windows=[0.0, 1.0])
        self.assertArrayAlmostEqual(bts_Y, branch_true_Y)
        self.assertArrayAlmostEqual(py_bsc_Y, branch_true_Y)
        self.assertArrayAlmostEqual(ts.sample_count_stats("branch", A, f)[0][0],
                                    branch_true_Y)
        self.assertArrayAlmostEqual(py_bsc.sample_count_stats(A, f)[0][0],
                                    branch_true_Y)

        # sites, Y:
        sts_Y = ts.Y3([[0], [1], [2]], windows=[0.0, 1.0],
                      stat_type="site")[0][0]
        py_ssc_Y = py_ssc.Y3([0], [1], [2], windows=[0.0, 1.0])
        self.assertArrayAlmostEqual(sts_Y, site_true_Y)
        self.assertArrayAlmostEqual(py_ssc_Y, site_true_Y)
        self.assertArrayAlmostEqual(ts.sample_count_stats("site", A, f)[0][0],
                                    site_true_Y)
        self.assertArrayAlmostEqual(py_ssc.sample_count_stats(A, f)[0][0],
                                    site_true_Y)

        # nodes, diversity in [0,1,2]
        nodes_div_012 = ts.diversity([[0, 1, 2]], stat_type="node")
        py_nodes_div_012 = py_nsc.diversity([0, 1, 2])
        self.assertArrayAlmostEqual(nodes_div_012, node_true_diversity_012)
        self.assertArrayAlmostEqual(py_nodes_div_012, node_true_diversity_012)
        # nodes, divergence [0] to [1,2]
        nodes_div_0_12 = ts.divergence([[0], [1, 2]], indices=[(0, 1)], stat_type="node")
        py_nodes_div_0_12 = py_nsc.divergence([0], [1, 2])
        self.assertArrayAlmostEqual(nodes_div_0_12, node_true_divergence_0_12)
        self.assertArrayAlmostEqual(py_nodes_div_0_12, node_true_divergence_0_12)

    @unittest.skip("Skip divergence")
    def test_case_odds_and_ends(self):
        # Tests having (a) the first site after the first window, and
        # (b) no samples having the ancestral state.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0.5
        3       0           1.0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     0.5     2       0,1
        0.5     1.0     3       0,1
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.65        0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       0       1               -1
        0       1       2               -1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            strict=False)
        py_ssc = PythonSiteStatCalculator(ts)

        py_div = np.array(
                  [[np.nan, py_ssc.divergence([0], [1], windows=[0.0, 0.5])[0], np.nan],
                   [np.nan, py_ssc.divergence([0], [1], windows=[0.5, 1.0])[0], np.nan]])
        div = ts.divergence([[0], [1]], indices=[(0, 0), (0, 1), (1, 1)],
                            windows=[0.0, 0.5, 1.0], stat_type="site")
        self.assertArrayEqual(py_div[0], div[0])
        self.assertArrayEqual(py_div[1], div[1])

    def test_case_four_taxa(self):
        #
        # 1.0          7
        # 0.7         / \                                    6
        #            /   \                                  / \
        # 0.5       /     5              5                 /   5
        #          /     / \            / \__             /   / \
        # 0.4     /     8   \          8     4           /   8   \
        #        /     / \   \        / \   / \         /   / \   \
        # 0.0   0     1   3   2      1   3 0   2       0   1   3   2
        #          (0.0, 0.2),        (0.2, 0.8),       (0.8, 2.5)

        # f4(0, 1, 2, 3): (0 -> 1)(2 -> 3)
        branch_true_f4_0123 = (0.1 * 0.2 + (0.1 + 0.1) * 0.6 + 0.1 * 1.7) / 2.5
        windows = [0.0, 0.4, 2.5]
        branch_true_f4_0123_windowed = np.array([(0.1 * 0.2 + (0.1 + 0.1) * 0.2) / 0.4,
                                                 ((0.1 + 0.1) * 0.4 + 0.1 * 1.7) / 2.1])
        # f4(0, 3, 2, 1): (0 -> 3)(2 -> 1)
        branch_true_f4_0321 = (0.1 * 0.2 + (0.1 + 0.1) * 0.6 + 0.1 * 1.7) / 2.5
        # f2([0,2], [1,3]) = (1/2) (f4(0,1,2,3) + f4(0,3,2,1))
        branch_true_f2_02_13 = (branch_true_f4_0123 + branch_true_f4_0321) / 2

        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       0           0.4
        5       0           0.5
        6       0           0.7
        7       0           1.0
        8       0           0.4
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     2.5     8       1,3
        0.2     0.8     4       0,2
        0.0     0.2     5       8,2
        0.2     0.8     5       8,4
        0.8     2.5     5       8,2
        0.8     2.5     6       0,5
        0.0     0.2     7       0,5
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            strict=False)
        py_bsc = PythonBranchStatCalculator(ts)

        A = [[0], [1], [2], [3]]
        self.assertAlmostEqual(branch_true_f4_0123, py_bsc.f4(*A)[0])
        self.assertAlmostEqual(branch_true_f4_0123, ts.f4(A, stat_type="branch")[0][0])
        self.assertArrayAlmostEqual(branch_true_f4_0123_windowed,
                                    ts.f4(A, windows=windows, stat_type="branch").flatten())
        A = [[0], [3], [2], [1]]
        self.assertAlmostEqual(branch_true_f4_0321, py_bsc.f4(*A)[0])
        self.assertAlmostEqual(branch_true_f4_0321, ts.f4(A, stat_type="branch")[0][0])
        A = [[0], [2], [1], [3]]
        self.assertAlmostEqual(0.0, py_bsc.f4(*A)[0])
        self.assertAlmostEqual(0.0, ts.f4(A, stat_type="branch")[0][0])
        A = [[0, 2], [1, 3]]
        self.assertAlmostEqual(branch_true_f2_02_13, py_bsc.f2(*A)[0])
        self.assertAlmostEqual(branch_true_f2_02_13, ts.f2(A, stat_type="branch")[0][0])

    def test_case_recurrent_muts(self):
        # With mutations:
        #
        # 1.0          6
        # 0.7         / \                                    5
        #           (0)  \                                  /(6)
        # 0.5      (1)    4                4               /   4
        #          /     / \              / \             /  (7|8)
        # 0.4    (2)   (3)  \           (4)  3           /   /   \
        #        /     /     \          /   /(5)        /   /     \
        # 0.0   0     1       2        1   0   2       0   1       2
        #          (0.0, 0.2),        (0.2, 0.8),       (0.8, 1.0)
        # genotypes:
        #       0     2       0        1   0   1       0   2       3
        site_true_Y = 0 + 1 + 1

        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           0.4
        4       0           0.5
        5       0           0.7
        6       0           1.0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.2     0.8     3       0,2
        0.0     0.2     4       1,2
        0.2     0.8     4       1,3
        0.8     1.0     4       1,2
        0.8     1.0     5       0,4
        0.0     0.2     6       0,4
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.05        0
        1   0.3         0
        2   0.9         0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       0       1               -1
        0       0       2               0
        0       0       0               1
        0       1       2               -1
        1       1       1               -1
        1       2       1               -1
        2       4       1               -1
        2       1       2               6
        2       2       3               6
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        py_ssc = PythonSiteStatCalculator(ts)

        # Y3:
        site_tsc_Y = ts.Y3([[0], [1], [2]], windows=[0.0, 1.0], stat_type="site")[0][0]
        py_ssc_Y = py_ssc.Y3([0], [1], [2], windows=[0.0, 1.0])
        self.assertAlmostEqual(site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(py_ssc_Y, site_true_Y)

    def test_case_2(self):
        # Here are the trees:
        # t                  |              |              |             |
        #
        # 0       --3--      |     --3--    |     --3--    |    --3--    |
        #        /  |  \     |    /  |  \   |    /     \   |   /     \   |
        # 1     4   |   5    |   4   |   5  |   4       5  |  4       5  |
        #       |\ / \ /|    |   |\   \     |   |\     /   |  |\     /   |
        # 2     | 6   7 |    |   | 6   7    |   | 6   7    |  | 6   7    |
        #       | |\ /| |    |   *  \  |    |   |  \  |    |  |  \       | ...
        # 3     | | 8 | |    |   |   8 *    |   |   8 |    |  |   8      |
        #       | |/ \| |    |   |  /  |    |   |  /  |    |  |  / \     |
        # 4     | 9  10 |    |   * 9  10    |   | 9  10    |  | 9  10    |
        #       |/ \ / \|    |   |  \   \   |   |  \   \   |  |  \   \   |
        # 5     0   1   2    |   0   1   2  |   0   1   2  |  0   1   2  |
        #
        #                    |   0.0 - 0.1  |   0.1 - 0.2  |  0.2 - 0.4  |
        # ... continued:
        # t                 |             |             |
        #
        # 0         --3--   |    --3--    |    --3--    |
        #          /     \  |   /     \   |   /     \   |
        # 1       4       5 |  4       5  |  4       5  |
        #         |\     /| |  |\     /|  |   \     /|  |
        # 2       | 6   7 | |  | 6   7 |  |    6   7 |  |
        #         |  \    | |  |  *    *  |     \    |  |
        # 3 ...   |   8   | |  |   8   |  |      8   |  | ...
        #         |  / \  | |  |  / \  |  |     / \  |  |
        # 4       | 9  10 | |  | 9  10 |  |    9  10 |  |
        #         |  \    | |  |    /  |  |   /   /  |  |
        # 5       0   1   2 |  0   1   2  |  0   1   2  |
        #
        #         0.4 - 0.5 |  0.5 - 0.6  |  0.6 - 0.7  |
        # ... continued:
        # t                   |             |
        #
        # 0          --3--    |    --3--    |    --3--
        #           /     \   |   /     \   |   /  |  \
        # 1        4       5  |  4       5  |  4   |   5
        #           \     /|  |   \     /|  |     /   /|
        # 2          6   7 |  |    6   7 |  |    6   7 |
        #               *  |  |    |  /  |  |    |  /  |
        # 3  ...       8   |  |    | 8   |  |    | 8   |
        #             * \  |  |    |  \  |  |    |  \  |
        # 4          9  10 |  |    9  10 |  |    9  10 |
        #           /   /  |  |   /   /  |  |   /   /  |
        # 5        0   1   2  |  0   1   2  |  0   1   2
        #
        #          0.7 - 0.8  |  0.8 - 0.9  |  0.9 - 1.0
        #
        # Above, subsequent mutations are backmutations.

        # divergence betw 0 and 1
        branch_true_diversity_01 = 2*(0.6*4 + 0.2*2 + 0.2*5)
        # # divergence betw 1 and 2
        # branch_true_diversity_12 = 2*(0.2*5 + 0.2*2 + 0.3*5 + 0.3*4)
        # # divergence betw 0 and 2
        # branch_true_diversity_02 = 2*(0.2*5 + 0.2*4 + 0.3*5 + 0.1*4 + 0.2*5)
        # Y(0;1, 2)
        branch_true_Y = 0.2*4 + 0.2*(4+2) + 0.2*4 + 0.2*2 + 0.2*(5+1)

        # site stats
        # Y(0;1, 2)
        site_true_Y = 1

        nodes = io.StringIO("""\
        is_sample       time    population
        1       0.000000        0
        1       0.000000        0
        1       0.000000        0
        0       5.000000        0
        0       4.000000        0
        0       4.000000        0
        0       3.000000        0
        0       3.000000        0
        0       2.000000        0
        0       1.000000        0
        0       1.000000        0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.500000        1.000000        10      1
        0.000000        0.400000        10      2
        0.600000        1.000000        9       0
        0.000000        0.500000        9       1
        0.800000        1.000000        8       10
        0.200000        0.800000        8       9,10
        0.000000        0.200000        8       9
        0.700000        1.000000        7       8
        0.000000        0.200000        7       10
        0.800000        1.000000        6       9
        0.000000        0.700000        6       8
        0.400000        1.000000        5       2,7
        0.100000        0.400000        5       7
        0.600000        0.900000        4       6
        0.000000        0.600000        4       0,6
        0.900000        1.000000        3       4,5,6
        0.100000        0.900000        3       4,5
        0.000000        0.100000        3       4,5,7
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.0         0
        1   0.55        0
        2   0.75        0
        3   0.85        0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       0       1               -1
        0       10      1               -1
        0       0       0               0
        1       8       1               -1
        1       2       1               -1
        2       8       1               -1
        2       9       0               5
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            strict=False)
        py_bsc = PythonBranchStatCalculator(ts)
        py_ssc = PythonSiteStatCalculator(ts)

        # divergence between 0 and 1
        A = [[0], [1]]

        def f(x):
            return np.array([float((x[0] > 0) != (x[1] > 0))/2.0])

        # tree lengths:
        self.assertArrayAlmostEqual(py_bsc.diversity([0], [1]),
                                    branch_true_diversity_01)
        self.assertArrayAlmostEqual(
                ts.sample_count_stats("branch", A, f)[0][0],
                branch_true_diversity_01)
        self.assertArrayAlmostEqual(
                py_bsc.sample_count_stats(A, f)[0][0],
                branch_true_diversity_01)

        # Y-statistic for (0/12)
        A = [[0], [1, 2]]

        def f(x):
            return np.array([float(((x[0] == 1) and (x[1] == 0))
                                   or ((x[0] == 0) and (x[1] == 2)))/2.0])

        # tree lengths:
        self.assertArrayAlmostEqual(py_bsc.Y3([0], [1], [2]), branch_true_Y)
        self.assertArrayAlmostEqual(ts.sample_count_stats("branch", A, f)[0][0],
                                    branch_true_Y)
        self.assertArrayAlmostEqual(py_bsc.sample_count_stats(A, f)[0][0],
                                    branch_true_Y)

        # sites:
        site_tsc_Y = ts.Y3([[0], [1], [2]], windows=[0.0, 1.0],
                           stat_type="site")[0][0]
        py_ssc_Y = py_ssc.Y3([0], [1], [2], windows=[0.0, 1.0])
        self.assertAlmostEqual(site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(py_ssc_Y, site_true_Y)
        self.assertAlmostEqual(ts.sample_count_stats("site", A, f)[0][0],
                               site_true_Y)
        self.assertAlmostEqual(py_ssc.sample_count_stats(A, f)[0][0],
                               site_true_Y)

    def test_small_sim(self):
        orig_ts = msprime.simulate(4, random_seed=self.random_seed,
                                   mutation_rate=0.0,
                                   recombination_rate=3.0)
        ts = tsutil.jukes_cantor(orig_ts, num_sites=3, mu=3,
                                 multiple_per_node=True, seed=self.seed)
        py_bsc = PythonBranchStatCalculator(ts)
        py_ssc = PythonSiteStatCalculator(ts)

        A = [[0], [1], [2]]
        self.assertAlmostEqual(
                ts.Y3(A, windows=[0.0, 1.0], stat_type="branch")[0][0],
                py_bsc.Y3(*A))
        self.assertAlmostEqual(
                ts.Y3(A, windows=[0.0, 1.0], stat_type="site")[0][0],
                py_ssc.Y3(*A))

        A = [[0], [1, 2]]
        self.assertAlmostEqual(
                ts.Y2(A, windows=[0.0, 1.0], stat_type="branch")[0][0],
                py_bsc.Y2(*A))
        self.assertAlmostEqual(
                ts.Y2(A, windows=[0.0, 1.0], stat_type="site")[0][0],
                py_ssc.Y2(*A))


class BranchStatsTestCase(GeneralStatsTestCase):
    """
    Tests of tree statistic computation.
    """

    def setUp(self):
        self.rng = random.Random(self.random_seed)
        self.stat_type = "branch"
        self.py_stat_class = PythonBranchStatCalculator

    def get_ts(self):
        for N in [12, 15, 20]:
            yield msprime.simulate(N, random_seed=self.random_seed,
                                   recombination_rate=10)

    def check_divergence_matrix(self, ts):
        # nonoverlapping samples
        samples = self.rng.sample(list(ts.samples()), 6)
        py_tsc = PythonBranchStatCalculator(ts)
        A = [samples[0:3], samples[3:5], samples[5:6]]
        windows = [0.0, ts.sequence_length/2, ts.sequence_length]
        indices = [(i, j) for i in range(len(A)) for j in range(i, len(A))]
        ts_values = ts.divergence(A, indices=indices, windows=windows,
                                  stat_type="branch")
        ts_matrix_values = ts.divergence_matrix(A, windows, stat_type="branch")
        self.assertListEqual([len(x) for x in ts_values],
                             [len(samples), len(samples)])
        assert(len(A[2]) == 1)
        for x in ts_values:
            self.assertTrue(np.isnan(x[5]))
        self.assertEqual(len(ts_values), len(ts_matrix_values))
        for w in range(len(ts_values)):
            self.assertArrayEqual(
                ts_matrix_values[w, :, :], upper_tri_to_matrix(ts_values[w]))
        here_values = np.array([[py_tsc.divergence(A[i], A[j], windows=windows)
                                 for i in range(len(A))]
                                for j in range(len(A))])
        for k in range(len(windows)-1):
            for i in range(len(A)):
                for j in range(len(A)):
                    if i == j:
                        if len(A[i]) == 1:
                            here_values[k, i, i] = np.nan
                        else:
                            here_values[k, i, i] /= (len(A[i])-1)/len(A[i])
                    else:
                        here_values[k, j, i]
        for k in range(len(windows)-1):
            self.assertArrayAlmostEqual(here_values[k], ts_matrix_values[k])

    @unittest.skip("Skipping errors")
    def test_errors(self):
        ts = msprime.simulate(10, random_seed=self.random_seed,
                              recombination_rate=10)
        self.assertRaises(ValueError,
                          ts.divergence, [[0], [11]],
                          windows=[0, ts.sequence_length], stat_type="branch")
        self.assertRaises(ValueError,
                          ts.divergence, [[0], [1]],
                          windows=[0.0, 2.0, 1.0, ts.sequence_length],
                          stat_type="branch")
        # errors if indices aren't of the right length
        self.assertRaises(ValueError,
                          ts.Y3, [[0], [1], [2]], indices=[(0, 1)],
                          windows=[0, ts.sequence_length], stat_type="branch")
        self.assertRaises(ValueError,
                          ts.f4, [[0], [1], [2], [3]], indices=[(0, 1)],
                          windows=[0, ts.sequence_length], stat_type="branch")
        self.assertRaises(ValueError,
                          ts.f3, [[0], [1], [2], [3]], indices=[(0, 1)],
                          windows=[0, ts.sequence_length], stat_type="branch")
        self.assertRaises(ValueError,
                          ts.f2, [[0], [1], [2], [3]], indices=[(0, 1, 2)],
                          windows=[0, ts.sequence_length], stat_type="branch")

    @unittest.skip("Skipping SFS.")
    def test_sfs_interface(self):
        ts = msprime.simulate(10)
        tsc = tskit.BranchStatCalculator(ts)

        # Duplicated samples raise an error
        self.assertRaises(ValueError, tsc.site_frequency_spectrum, [1, 1])
        self.assertRaises(ValueError, tsc.site_frequency_spectrum, [])
        self.assertRaises(ValueError, tsc.site_frequency_spectrum, [0, 11])
        # Check for bad windows
        for bad_start in [-1, 1, 1e-7]:
            self.assertRaises(
                ValueError, tsc.site_frequency_spectrum, [1, 2],
                [bad_start, ts.sequence_length])
        for bad_end in [0, ts.sequence_length - 1, ts.sequence_length + 1]:
            self.assertRaises(
                ValueError, tsc.site_frequency_spectrum, [1, 2],
                [0, bad_end])
        # Windows must be increasing.
        self.assertRaises(
            ValueError, tsc.site_frequency_spectrum, [1, 2], [0, 1, 1])

    def test_branch_f_stats(self):
        for ts in self.get_ts():
            self.check_f_stats(ts)

    def test_branch_Y_stats(self):
        for ts in self.get_ts():
            self.check_Y_stat(ts)

    @unittest.skip("do we return a divergence matrix?")
    def test_diversity(self):
        for ts in self.get_ts():
            self.check_divergence_matrix(ts)

    @unittest.skip("No SFS.")
    def test_branch_sfs(self):
        for ts in self.get_ts():
            self.check_sfs(ts)


class SiteStatsTestCase(GeneralStatsTestCase):
    """
    Tests of site statistic computation.
    """

    def setUp(self):
        self.rng = random.Random(self.random_seed)
        self.stat_type = "site"
        self.py_stat_class = PythonSiteStatCalculator

    def get_ts(self):
        for mut in [0.0, 3.0]:
            yield msprime.simulate(20, random_seed=self.random_seed,
                                   mutation_rate=mut,
                                   recombination_rate=3.0)
        ts = msprime.simulate(20, random_seed=self.random_seed,
                              mutation_rate=0.0,
                              recombination_rate=3.0)
        for mpn in [False, True]:
            for num_sites in [10, 100]:
                mut_ts = tsutil.jukes_cantor(ts, num_sites=num_sites, mu=3,
                                             multiple_per_node=mpn,
                                             seed=self.random_seed)
                yield mut_ts

    def check_pairwise_diversity_mutations(self, ts):
        py_tsc = PythonSiteStatCalculator(ts)
        samples = random.sample(list(ts.samples()), 2)
        A = [[samples[0]], [samples[1]]]
        n = [len(a) for a in A]

        def f(x):
            return np.array([float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])
                             / (2*n[0]*n[1])])

        self.assertAlmostEqual(
            py_tsc.sample_count_stats(A, f).flatten(),
            ts.pairwise_diversity(samples=samples))

    def test_pairwise_diversity(self):
        ts = msprime.simulate(20, random_seed=self.random_seed,
                              recombination_rate=100)
        self.check_pairwise_diversity_mutations(ts)

    def test_site_f_stats(self):
        for ts in self.get_ts():
            self.check_f_stats(ts)

    def test_site_Y_stats(self):
        for ts in self.get_ts():
            self.check_Y_stat(ts)

    @unittest.skip("No sfs.")
    def test_site_sfs(self):
        for ts in self.get_ts():
            self.check_sfs(ts)


class NodeStatsTestCase(GeneralStatsTestCase):
    """
    Tests of node statistic computation.
    """

    def setUp(self):
        self.rng = random.Random(self.random_seed)
        self.stat_type = "node"
        self.py_stat_class = PythonNodeStatCalculator

    def get_ts(self):
        for N in [12, 15, 20]:
            yield msprime.simulate(N, random_seed=self.random_seed,
                                   recombination_rate=10)

    def compare_stats(self, ts, py_fn, ts_fn,
                      sample_sets, index_length):
        # will compare py_fn() to ts_fn(..., stat_type)
        assert(len(sample_sets) >= index_length)
        windows = [k * ts.sequence_length / 20 for k in
                   [0] + sorted(self.rng.sample(range(1, 20), 4)) + [20]]
        indices = [self.rng.sample(range(len(sample_sets)), max(1, index_length))
                   for _ in range(5)]
        for idx in indices:
            ssets = [sample_sets[i] for i in idx]
            ts_vals = ts_fn(ssets, windows=windows, stat_type=self.stat_type)
            self.assertEqual((len(windows) - 1, ts.num_nodes), ts_vals.shape)
            py_vals = py_fn(*ssets, windows)
            self.assertArrayAlmostEqual(py_vals, ts_vals)

    def test_node_diversity(self):
        for ts in self.get_ts():
            self.check_divergence(ts)
