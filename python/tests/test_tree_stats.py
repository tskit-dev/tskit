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


def path_length(tr, x, y):
    L = 0
    mrca = tr.mrca(x, y)
    for u in x, y:
        while u != mrca:
            L += tr.branch_length(u)
            u = tr.parent(u)
    return L


class PythonBranchLengthStatCalculator(object):
    """
    Python implementations of various ("tree") branch-length statistics -
    inefficient but more clear what they are doing.
    """

    def __init__(self, tree_sequence):
        self.tree_sequence = tree_sequence

    def divergence(self, X, Y, begin=0.0, end=None):
        '''
        Computes average pairwise diversity between a random choice from x
        and a random choice from y over the window specified.
        '''
        if end is None:
            end = self.tree_sequence.sequence_length
        S = 0
        for tr in self.tree_sequence.trees():
            if tr.interval[1] <= begin:
                continue
            if tr.interval[0] >= end:
                break
            SS = 0
            for x in X:
                for y in Y:
                    SS += path_length(tr, x, y) / 2.0
            S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
        return S/((end-begin)*len(X)*len(Y))

    def tree_length_diversity(self, X, Y, begin=0.0, end=None):
        '''
        Computes average pairwise diversity between a random choice from x
        and a random choice from y over the window specified.
        '''
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S/((end-begin)*len(X)*len(Y))

    def Y3(self, X, Y, Z, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S/((end - begin) * len(X) * len(Y) * len(Z))

    def Y2(self, X, Y, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S/((end - begin) * len(X) * len(Y) * (len(Y)-1))

    def Y1(self, X, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S/((end - begin) * len(X) * (len(X)-1) * (len(X)-2))

    def f4(self, A, B, C, D, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S / ((end - begin) * len(A) * len(B) * len(C) * len(D))

    def f3(self, A, B, C, begin=0.0, end=None):
        # this is f4(A,B;A,C) but drawing distinct samples from A
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S / ((end - begin) * len(A) * (len(A) - 1) * len(B) * len(C))

    def f2(self, A, B, begin=0.0, end=None):
        # this is f4(A,B;A,B) but drawing distinct samples from A and B
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S / ((end - begin) * len(A) * (len(A) - 1) * len(B) * (len(B) - 1))

    def tree_stat_vector(self, sample_sets, weight_fun, begin=0.0, end=None):
        '''
        Here sample_sets is a list of lists of samples, and weight_fun is a function
        whose argument is a list of integers of the same length as sample_sets
        that returns a list of numbers; there will be one output for each element.
        For each value, each branch in a tree is weighted by weight_fun(x),
        where x[i] is the number of samples in sample_sets[i] below that
        branch.  This finds the sum of all counted branches for each tree,
        and averages this across the tree sequence ts, weighted by genomic length.

        This version is inefficient as it iterates over all nodes in each tree.
        '''
        for U in sample_sets:
            if max([U.count(x) for x in set(U)]) > 1:
                raise ValueError("elements of sample_sets",
                                 "cannot contain repeated elements.")
        if end is None:
            end = self.tree_sequence.sequence_length
        W = np.array([[float(u in A) for A in sample_sets] for u in self.tree_sequence.samples()])
        S = naive_general_branch_stats(self.tree_sequence, W, weight_fun, windows=[begin, end])
        return S

    def naive_general_branch_stats(self, W, f, windows=None, polarised=False):
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
        if windows is None:
            return sigma
        else:
            return self.tree_sequence.windowed_tree_stat(sigma, windows)

    def site_frequency_spectrum(self, sample_set, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
        n_out = len(sample_set)
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
        return S


class PythonSiteStatCalculator(object):
    """
    Python implementations of various single-site statistics -
    inefficient but more clear what they are doing.
    """

    def __init__(self, tree_sequence):
        self.tree_sequence = tree_sequence

    def divergence(self, X, Y, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S/((end - begin) * len(X) * len(Y))

    def Y3(self, X, Y, Z, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S/((end - begin) * len(X) * len(Y) * len(Z))

    def Y2(self, X, Y, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S/((end - begin) * len(X) * len(Y) * (len(Y) - 1))

    def Y1(self, X, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S/((end - begin) * len(X) * (len(X) - 1) * (len(X) - 2))

    def f4(self, A, B, C, D, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S / ((end - begin) * len(A) * len(B) * len(C) * len(D))

    def f3(self, A, B, C, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S / ((end - begin) * len(A) * len(B) * len(C) * (len(A) - 1))

    def f2(self, A, B, begin=0.0, end=None):
        if end is None:
            end = self.tree_sequence.sequence_length
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
        return S / ((end - begin) * len(A) * len(B)
                    * (len(A) - 1) * (len(B) - 1))

    def tree_stat_vector(self, sample_sets, weight_fun, begin=0.0, end=None):
        '''
        Here sample_sets is a list of lists of samples, and weight_fun is a function
        whose argument is a list of integers of the same length as sample_sets
        that returns a list of numbers; there will be one output for each element.
        For each value, each allele in a tree is weighted by weight_fun(x), where
        x[i] is the number of samples in sample_sets[i] that inherit that allele.
        This finds the sum of this value for all alleles at all polymorphic sites,
        and across the tree sequence ts, weighted by genomic length.

        This version is inefficient as it works directly with haplotypes.
        '''
        for U in sample_sets:
            if max([U.count(x) for x in set(U)]) > 1:
                raise ValueError("elements of sample_sets",
                                 "cannot contain repeated elements.")
        if end is None:
            end = self.tree_sequence.sequence_length
        haps = list(self.tree_sequence.haplotypes())
        n_out = len(weight_fun([0 for a in sample_sets]))
        site_positions = [x.position for x in self.tree_sequence.sites()]
        S = [0.0 for j in range(n_out)]
        for k in range(self.tree_sequence.num_sites):
            if (site_positions[k] >= begin) and (site_positions[k] < end):
                all_g = [haps[j][k] for j in range(self.tree_sequence.num_samples)]
                g = [[haps[j][k] for j in u] for u in sample_sets]
                for a in set(all_g):
                    x = [h.count(a) for h in g]
                    w = weight_fun(x)
                    for j in range(n_out):
                        S[j] += w[j]
        for j in range(n_out):
            S[j] /= (end - begin)
        return np.array([S])

    def naive_general_site_stats(self, W, f, windows=None, polarised=False):
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
        if windows is None:
            return sigma
        else:
            return self.tree_sequence.windowed_sitewise_stat(sigma, windows)

    def site_frequency_spectrum(self, sample_set, begin=0.0, end=None):
        '''
        '''
        if end is None:
            end = self.tree_sequence.sequence_length
        haps = list(self.tree_sequence.haplotypes())
        n_out = len(sample_set)
        site_positions = [x.position for x in self.tree_sequence.sites()]
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
        return S


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


class TestGeneralBranchStats(unittest.TestCase):
    """
    Tests for general tree stats.
    """
    def run_stats(self, ts, W, f, windows=None, polarised=False):
        py_bsc = PythonBranchLengthStatCalculator(ts)
        sigma1 = py_bsc.naive_general_branch_stats(W, f, windows, polarised=polarised)
        sigma2 = ts.general_stat("branch", W, f, windows, polarised=polarised)
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
        ts = msprime.simulate(30, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        sigma = self.run_stats(ts, W, lambda x: x, polarised=True)
        self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))
        # A W of 1 for every node and identity f counts the samples in the subtree
        # if polarised is True.
        # print(sigma)
        for tree in ts.trees():
            s = tree.span * sum(
                tree.num_samples(u) * tree.branch_length(u) for u in tree.nodes())
            self.assertTrue(np.allclose(sigma[tree.index], s))

    def test_simple_cumsum_f_w_ones(self):
        ts = msprime.simulate(20, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 8))
        for polarised in [True, False]:
            sigma = self.run_stats(ts, W, lambda x: np.cumsum(x))
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
        py_ssc = PythonSiteStatCalculator(ts)
        sigma1 = py_ssc.naive_general_site_stats(W, f, windows, polarised=polarised)
        sigma2 = ts.general_stat("site", W, f, windows, polarised=polarised)
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


class GeneralStatsTestCase(unittest.TestCase):
    """
    Tests of statistic computation.  Derived classes should have attributes
    `stat_type` and `py_stat_class`.
    """

    random_seed = 123456

    def setUp(self):
        self.rng = random.Random(self.random_seed)

    def assertListAlmostEqual(self, x, y):
        self.assertEqual(len(x), len(y))
        for a, b in zip(x, y):
            self.assertAlmostEqual(a, b)

    def assertArrayEqual(self, x, y):
        nt.assert_equal(x, y)

    def assertArrayAlmostEqual(self, x, y):
        nt.assert_array_almost_equal(x, y)

    def compare_stats(self, ts,
                      py_fn, ts_fn, stat_type,
                      sample_sets, index_length)
        """
        Use to compare a tree sequence method tsc_vector_fn to a single-window-based
        implementation tree_fn that takes index_length leaf sets at once.  Pass
        index_length=0 to signal that tsc_fn does not take an 'indices' argument;
        otherwise, gives the length of each of the tuples.

        Here are the arguments these functions will get:
            py_fn(sample_set[i], ... , sample_set[k], begin=left, end=right)
            ts_fn(sample_sets, windows, indices)
            ... or tsc_vector_fn(sample_sets, windows)
        """
        assert(len(sample_sets) >= index_length)
        nl = len(sample_sets)
        windows = [k * ts.sequence_length / 20 for k in
                   [0] + sorted(self.rng.sample(range(1, 20), 4)) + [20]]
        indices = [self.rng.sample(range(nl), max(1, index_length)) for _ in range(5)]
        leafset_args = [[sample_sets[i] for i in ii] for ii in indices]
        win_args = [{'begin': windows[i], 'end': windows[i+1]}
                    for i in range(len(windows)-1)]
        tree_vals = np.array([[py_fn(*a, **b) for a in leafset_args] for b in win_args])
        if index_length > 0:
            tsc_vector_vals = ts_fn(stat_type, sample_sets, windows, indices, **kwargs)
        else:
            tsc_vector_vals = ts_fn(stat_type, [sample_sets[i[0]] for i in indices],
                                            windows, **kwargs)
        self.assertEqual(len(tree_vals), len(windows)-1)
        self.assertEqual(len(tsc_vector_vals), len(windows)-1)
        self.assertArrayAlmostEqual(tsc_vector_vals, tree_vals)

    def compare_sfs(self, ts, tree_fn, sample_sets, tsc_fn):
        """
        """
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

        def wfn(x):
            return [1]

        # empty sample sets will raise an error
        self.assertRaises(ValueError, ts.tree_stat_vector,
                          self.stat_type, samples[0:2] + [], wfn)
        # sample_sets must be lists without repeated elements
        self.assertRaises(ValueError, ts.tree_stat_vector,
                          self.stat_type, samples[0:2], wfn)
        self.assertRaises(ValueError, ts.tree_stat_vector,
                          self.stat_type, [samples[0:2], [samples[2], samples[2]]], wfn)
        # and must all be samples
        self.assertRaises(ValueError, ts.tree_stat_vector,
                          self.stat_type, [samples[0:2], [max(samples)+1]], wfn)
        # windows must start at 0.0, be increasing, and extend to the end
        self.assertRaises(ValueError, ts.tree_stat_vector,
                          self.stat_type, [samples[0:2], samples[2:4]], wfn,
                          [0.1, ts.sequence_length])
        self.assertRaises(ValueError, ts.tree_stat_vector,
                          self.stat_type, [samples[0:2], samples[2:4]], wfn,
                          [0.0, 0.8*ts.sequence_length])
        self.assertRaises(ValueError, ts.tree_stat_vector,
                          self.stat_type, [samples[0:2], samples[2:4]], wfn,
                          [0.0, 0.8*ts.sequence_length, 0.4*ts.sequence_length,
                           ts.sequence_length])

    def check_sfs_interface(self, ts):
        samples = ts.samples()

        # empty sample sets will raise an error
        self.assertRaises(ValueError, ts.site_frequency_spectrum, self.stat_type, [])
        # sample_sets must be lists without repeated elements
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          self.stat_type, [samples[2], samples[2]])
        # and must all be samples
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          self.stat_type, [samples[0], max(samples)+1])
        # windows must start at 0.0, be increasing, and extend to the end
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          self.stat_type, samples[0:2], [0.1, ts.sequence_length])
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          self.stat_type, samples[0:2], [0.0, 0.8*ts.sequence_length])
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          self.stat_type, samples[0:2],
                          [0.0, 0.8*ts.sequence_length, 0.4*ts.sequence_length,
                           ts.sequence_length])

    def check_tree_stat_vector(self, ts):
        # test the general tree_stat_vector() machinery
        self.check_tree_stat_interface(ts)
        samples = self.rng.sample(list(ts.samples()), 12)
        A = [[samples[0], samples[1], samples[6]],
             [samples[2], samples[3], samples[7]],
             [samples[4], samples[5], samples[8]],
             [samples[9], samples[10], samples[11]]]
        py_tsc = self.py_stat_class(ts)

        # a made-up example
        def tsf(sample_sets, windows, indices):
            def f(x):
                return np.array([x[i] + 2.0 * x[j] + 3.5 * x[k] for i, j, k in indices])
            return ts.tree_stat_vector(self.stat_type, sample_sets, weight_fun=f, windows=windows)

        def py_tsf(X, Y, Z, begin, end):
            def f(x):
                return np.array([x[0] + 2.0 * x[1] + 3.5 * x[2]])
            return py_tsc.tree_stat_vector([X, Y, Z], weight_fun=f,
                                           begin=begin, end=end)[0][0]

        self.compare_stats(ts, py_tsf, tsf, A, 3)

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
        self.assertRaises(ValueError, ts.f2_vector, self.stat_type,
                          [[0, 1], [3]], [0, ts.sequence_length], [(0, 1)])

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
        branch_true_diversity_01 = 2*(1 * (0.2-0) + 0.5 * (0.8-0.2) + 0.7 * (1.0-0.8))
        branch_true_diversity_02 = 2*(1 * (0.2-0) + 0.4 * (0.8-0.2) + 0.7 * (1.0-0.8))
        branch_true_diversity_12 = 2*(0.5 * (0.2-0) + 0.5 * (0.8-0.2) + 0.5 * (1.0-0.8))
        branch_true_Y = 0.2*(1 + 0.5) + 0.6*(0.4) + 0.2*(0.7+0.2)
        site_true_Y = 3 + 0 + 1

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
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        py_branch_tsc = PythonBranchLengthStatCalculator(ts)
        py_site_tsc = PythonSiteStatCalculator(ts)

        # diversity between 0 and 1
        A = [[0], [1]]
        n = [len(a) for a in A]

        def f(x):
            return np.array([float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])/float(2*n[0]*n[1])])

        # tree lengths:
        self.assertAlmostEqual(py_branch_tsc.tree_length_diversity([0], [1]),
                               branch_true_diversity_01)
        self.assertAlmostEqual(ts.tree_stat_vector("branch", A, f)[0][0],
                               branch_true_diversity_01)
        self.assertAlmostEqual(py_branch_tsc.tree_stat_vector(A, f)[0][0],
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
        self.assertAlmostEqual(py_branch_tsc.tree_length_diversity(A[0], A[1]),
                               branch_true_mean_diversity)
        self.assertAlmostEqual(ts.tree_stat_vector("branch", A, f)[0][0],
                               branch_true_mean_diversity)
        self.assertAlmostEqual(py_branch_tsc.tree_stat_vector(A, f)[0][0],
                               branch_true_mean_diversity)

        # Y-statistic for (0/12)
        A = [[0], [1, 2]]

        def f(x):
            return np.array([float(((x[0] == 1) and (x[1] == 0))
                                   or ((x[0] == 0) and (x[1] == 2)))/2.0])

        # tree lengths:
        branch_tsc_Y = ts.Y3("branch", [[0], [1], [2]], [0.0, 1.0])[0][0]
        py_branch_tsc_Y = py_branch_tsc.Y3([0], [1], [2], 0.0, 1.0)
        self.assertAlmostEqual(branch_tsc_Y, branch_true_Y)
        self.assertAlmostEqual(py_branch_tsc_Y, branch_true_Y)
        self.assertAlmostEqual(ts.tree_stat_vector("branch", A, f)[0][0], branch_true_Y)
        self.assertAlmostEqual(py_branch_tsc.tree_stat_vector(A, f)[0][0], branch_true_Y)

        # sites, Y:
        site_tsc_Y = ts.Y3("site", [[0], [1], [2]], [0.0, 1.0])[0][0]
        py_site_tsc_Y = py_site_tsc.Y3([0], [1], [2], 0.0, 1.0)
        self.assertAlmostEqual(site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(py_site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(ts.tree_stat_vector("site", A, f)[0][0], site_true_Y)
        self.assertAlmostEqual(py_site_tsc.tree_stat_vector(A, f)[0][0], site_true_Y)

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
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        py_site_tsc = PythonSiteStatCalculator(ts)

        # recall that divergence returns the upper triangle
        # with nans on the diag in this case
        py_div = np.array([[np.nan, py_site_tsc.divergence([0], [1], 0.0, 0.5), np.nan],
                          [np.nan, py_site_tsc.divergence([0], [1], 0.5, 1.0), np.nan]])
        div = ts.divergence("site", [[0], [1]], [0.0, 0.5, 1.0])
        self.assertArrayEqual(py_div[0], div[0])
        self.assertArrayEqual(py_div[1], div[1])

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
        py_site_tsc = PythonSiteStatCalculator(ts)

        # Y3:
        site_tsc_Y = ts.Y3("site", [[0], [1], [2]], [0.0, 1.0])[0][0]
        py_site_tsc_Y = py_site_tsc.Y3([0], [1], [2], 0.0, 1.0)
        self.assertAlmostEqual(site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(py_site_tsc_Y, site_true_Y)

    def test_case_2(self):
        # Here are the trees:
        # t                  |              |              |             |
        #
        # 0       --3--      |     --3--    |     --3--    |    --3--    |    --3--
        #        /  |  \     |    /  |  \   |    /     \   |   /     \   |   /     \
        # 1     4   |   5    |   4   |   5  |   4       5  |  4       5  |  4       5
        #       |\ / \ /|    |   |\   \     |   |\     /   |  |\     /   |  |\     /|
        # 2     | 6   7 |    |   | 6   7    |   | 6   7    |  | 6   7    |  | 6   7 |
        #       | |\ /| |    |   *  \  |    |   |  \  |    |  |  \       |  |  \    | ...
        # 3     | | 8 | |    |   |   8 *    |   |   8 |    |  |   8      |  |   8   |
        #       | |/ \| |    |   |  /  |    |   |  /  |    |  |  / \     |  |  / \  |
        # 4     | 9  10 |    |   * 9  10    |   | 9  10    |  | 9  10    |  | 9  10 |
        #       |/ \ / \|    |   |  \   \   |   |  \   \   |  |  \   \   |  |  \    |
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
        #         |  *    *  |     \    |  |       *  |  |    |  /  |  |    |  /  |
        # 3  ...  |   8   |  |      8   |  |      8   |  |    | 8   |  |    | 8   |
        #         |  / \  |  |     / \  |  |     * \  |  |    |  \  |  |    |  \  |
        # 4       | 9  10 |  |    9  10 |  |    9  10 |  |    9  10 |  |    9  10 |
        #         |    /  |  |   /   /  |  |   /   /  |  |   /   /  |  |   /   /  |
        # 5       0   1   2  |  0   1   2  |  0   1   2  |  0   1   2  |  0   1   2
        #
        #         0.5 - 0.6  |  0.6 - 0.7  |  0.7 - 0.8  |  0.8 - 0.9  |  0.9 - 1.0
        #
        # Above, subsequent mutations are backmutations.

        # divergence betw 0 and 1
        branch_true_diversity_01 = 2*(0.6*4 + 0.2*2 + 0.2*5)
        # divergence betw 1 and 2
        branch_true_diversity_12 = 2*(0.2*5 + 0.2*2 + 0.3*5 + 0.3*4)
        # divergence betw 0 and 2
        branch_true_diversity_02 = 2*(0.2*5 + 0.2*4 + 0.3*5 + 0.1*4 + 0.2*5)
        # mean divergence between 0, 1 and 0, 2
        branch_true_mean_diversity = (
            0 + branch_true_diversity_02 + branch_true_diversity_01
            + branch_true_diversity_12) / 4
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
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)
        py_branch_tsc = PythonBranchLengthStatCalculator(ts)
        py_site_tsc = PythonSiteStatCalculator(ts)

        # divergence between 0 and 1
        A = [[0], [1]]

        def f(x):
            return np.array([float((x[0] > 0) != (x[1] > 0))/2.0])

        # tree lengths:
        self.assertAlmostEqual(py_branch_tsc.tree_length_diversity([0], [1]),
                               branch_true_diversity_01)
        self.assertAlmostEqual(ts.tree_stat_vector("branch", A, f)[0][0],
                               branch_true_diversity_01)
        self.assertAlmostEqual(py_branch_tsc.tree_stat_vector(A, f)[0][0],
                               branch_true_diversity_01)

        # mean divergence between 0, 1 and 0, 2
        A = [[0, 1], [0, 2]]
        n = [len(a) for a in A]

        def f(x):
            return np.array([float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])/8.0])

        # tree lengths:
        self.assertAlmostEqual(py_branch_tsc.tree_length_diversity(A[0], A[1]),
                               branch_true_mean_diversity)
        self.assertAlmostEqual(ts.tree_stat_vector("branch", A, f)[0][0],
                               branch_true_mean_diversity)
        self.assertAlmostEqual(py_branch_tsc.tree_stat_vector(A, f)[0][0],
                               branch_true_mean_diversity)

        # Y-statistic for (0/12)
        A = [[0], [1, 2]]

        def f(x):
            return np.array([float(((x[0] == 1) and (x[1] == 0))
                                   or ((x[0] == 0) and (x[1] == 2)))/2.0])

        # tree lengths:
        self.assertAlmostEqual(py_branch_tsc.Y3([0], [1], [2]), branch_true_Y)
        self.assertAlmostEqual(ts.tree_stat_vector("branch", A, f)[0][0], branch_true_Y)
        self.assertAlmostEqual(py_branch_tsc.tree_stat_vector(A, f)[0][0], branch_true_Y)

        # sites:
        site_tsc_Y = ts.Y3("site", [[0], [1], [2]], [0.0, 1.0])[0][0]
        py_site_tsc_Y = py_site_tsc.Y3([0], [1], [2], 0.0, 1.0)
        self.assertAlmostEqual(site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(py_site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(ts.tree_stat_vector("site", A, f)[0][0], site_true_Y)
        self.assertAlmostEqual(py_site_tsc.tree_stat_vector(A, f)[0][0], site_true_Y)

    def test_small_sim(self):
        orig_ts = msprime.simulate(4, random_seed=self.random_seed,
                                   mutation_rate=0.0,
                                   recombination_rate=3.0)
        ts = tsutil.jukes_cantor(orig_ts, num_sites=3, mu=3,
                                 multiple_per_node=True, seed=self.seed)
        py_branch_tsc = PythonBranchLengthStatCalculator(ts)
        py_site_tsc = PythonSiteStatCalculator(ts)

        A = [[0], [1], [2]]
        self.assertAlmostEqual(ts.Y3("branch", A, [0.0, 1.0])[0][0],
                               py_branch_tsc.Y3(*A))
        self.assertAlmostEqual(ts.Y3("site", A, [0.0, 1.0])[0][0],
                               py_site_tsc.Y3(*A))

        A = [[0], [1, 2]]
        self.assertAlmostEqual(ts.Y2("branch", A, [0.0, 1.0])[0][0],
                               py_branch_tsc.Y2(*A))
        self.assertAlmostEqual(ts.Y2("site", A, [0.0, 1.0])[0][0],
                               py_site_tsc.Y2(*A))


class BranchLengthStatsTestCase(GeneralStatsTestCase):
    """
    Tests of tree statistic computation.
    """
    stat_type = "branch"
    py_stat_class = PythonBranchLengthStatCalculator

    def get_ts(self):
        for N in [12, 15, 20]:
            yield msprime.simulate(N, random_seed=self.random_seed,
                                   recombination_rate=10)

    def check_pairwise_diversity(self, ts):
        samples = self.rng.sample(list(ts.samples()), 2)
        py_tsc = PythonBranchLengthStatCalculator(ts)
        A_one = [[samples[0]], [samples[1]]]
        A_many = [self.rng.sample(list(ts.samples()), 2),
                  self.rng.sample(list(ts.samples()), 2)]
        for A in (A_one, A_many):
            n = [len(a) for a in A]

            def f(x):
                return np.array([float(x[0]*(n[1]-x[1]))/float(n[0]*n[1])])

            self.assertAlmostEqual(
                py_tsc.tree_stat_vector(A, f)[0][0],
                py_tsc.tree_length_diversity(A[0], A[1]))
            self.assertAlmostEqual(
                ts.tree_stat_vector("branch", A, f)[0][0],
                py_tsc.tree_length_diversity(A[0], A[1]))

    def check_divergence_matrix(self, ts):
        # nonoverlapping samples
        samples = self.rng.sample(list(ts.samples()), 6)
        py_tsc = PythonBranchLengthStatCalculator(ts)
        A = [samples[0:3], samples[3:5], samples[5:6]]
        windows = [0.0, ts.sequence_length/2, ts.sequence_length]
        ts_values = ts.divergence("branch", A, windows)
        ts_matrix_values = ts.divergence_matrix("branch", A, windows)
        self.assertListEqual([len(x) for x in ts_values], [len(samples), len(samples)])
        assert(len(A[2]) == 1)
        for x in ts_values:
            self.assertTrue(np.isnan(x[5]))
        self.assertEqual(len(ts_values), len(ts_matrix_values))
        for w in range(len(ts_values)):
            self.assertArrayEqual(
                ts_matrix_values[w, :, :], upper_tri_to_matrix(ts_values[w]))
        here_values = np.array([[[py_tsc.tree_length_diversity(A[i], A[j],
                                                               begin=windows[k],
                                                               end=windows[k+1])
                                  for i in range(len(A))]
                                 for j in range(len(A))]
                                for k in range(len(windows)-1)])
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

    def test_errors(self):
        ts = msprime.simulate(10, random_seed=self.random_seed, recombination_rate=10)
        self.assertRaises(ValueError,
                          ts.divergence, "branch", [[0], [11]], [0, ts.sequence_length])
        self.assertRaises(ValueError,
                          ts.divergence, "branch", [[0], [1]], [0, ts.sequence_length/2])
        self.assertRaises(ValueError,
                          ts.divergence, "branch", [[0], [1]], [ts.sequence_length/2,
                                                       ts.sequence_length])
        self.assertRaises(ValueError,
                          ts.divergence, "branch", [[0], [1]], [0.0, 2.0, 1.0,
                                                       ts.sequence_length])
        # errors if indices aren't of the right length
        self.assertRaises(ValueError,
                          ts.Y3, "branch", [[0], [1], [2]], [0, ts.sequence_length],
                          [[0, 1]])
        self.assertRaises(ValueError,
                          ts.f4, "branch", [[0], [1], [2], [3]], [0, ts.sequence_length],
                          [[0, 1]])
        self.assertRaises(ValueError,
                          ts.f3, "branch", [[0], [1], [2], [3]], [0, ts.sequence_length],
                          [[0, 1]])
        self.assertRaises(ValueError,
                          ts.f2, "branch", [[0], [1], [2], [3]], [0, ts.sequence_length],
                          [[0, 1, 2]])

    def test_windowization(self):
        ts = msprime.simulate(10, random_seed=self.random_seed, recombination_rate=100)
        samples = self.rng.sample(list(ts.samples()), 2)
        py_tsc = PythonBranchLengthStatCalculator(ts)
        A_one = [[samples[0]], [samples[1]]]
        A_many = [self.rng.sample(list(ts.samples()), 2),
                  self.rng.sample(list(ts.samples()), 2)]
        some_breaks = list(set([0.0, ts.sequence_length/2, ts.sequence_length] +
                               self.rng.sample(list(ts.breakpoints()), 5)))
        some_breaks.sort()
        tiny_breaks = ([(k / 4) * list(ts.breakpoints())[1] for k in range(4)] +
                       [ts.sequence_length])
        wins = [[0.0, ts.sequence_length],
                [0.0, ts.sequence_length/2, ts.sequence_length],
                tiny_breaks,
                some_breaks]

        with self.assertRaises(ValueError):
            ts.tree_stat_vector("branch", A_one, lambda x: 1.0,
                                 windows=[0.0, 1.0, ts.sequence_length+1.1])

        for A in (A_one, A_many):
            for windows in wins:
                n = [len(a) for a in A]

                def f(x):
                    return float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])/float(2*n[0]*n[1])

                def g(x):
                    return np.array([f(x)])

                tsdiv_v = ts.tree_stat_vector("branch", A, g, windows)
                tsdiv_vx = [x[0] for x in tsdiv_v]
                pydiv = np.array([py_tsc.tree_length_diversity(A[0], A[1], windows[k],
                                                               windows[k+1])
                                  for k in range(len(windows)-1)])
                self.assertListAlmostEqual(tsdiv_vx, pydiv)

    def test_tree_stat_vector_interface(self):
        ts = msprime.simulate(10)

        def f(x):
            return np.array([1.0])

        # Duplicated samples raise an error
        self.assertRaises(ValueError, ts.tree_stat_vector, "branch", [[1, 1]], f)
        self.assertRaises(ValueError, ts.tree_stat_vector, "branch", [[1], [2, 2]], f)
        # Make sure the basic call doesn't throw an exception
        ts.tree_stat_vector("branch", [[1, 2]], f)
        # Check for bad windows
        for bad_start in [-1, 1, 1e-7]:
            self.assertRaises(
                ValueError, ts.tree_stat_vector, "branch", [[1, 2]], f,
                [bad_start, ts.sequence_length])
        for bad_end in [0, ts.sequence_length - 1, ts.sequence_length + 1]:
            self.assertRaises(
                ValueError, ts.tree_stat_vector, "branch", [[1, 2]], f,
                [0, bad_end])
        # Windows must be increasing.
        self.assertRaises(
            ValueError, ts.tree_stat_vector, "branch", [[1, 2]], f, [0, 1, 1])

    @unittest.skip("Skipping SFS.")
    def test_sfs_interface(self):
        ts = msprime.simulate(10)
        tsc = tskit.BranchLengthStatCalculator(ts)

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

    def test_branch_general_stats(self):
        for ts in self.get_ts():
            self.check_tree_stat_vector(ts)

    def test_branch_f_stats(self):
        for ts in self.get_ts():
            self.check_f_stats(ts)

    def test_branch_Y_stats(self):
        for ts in self.get_ts():
            self.check_Y_stat(ts)

    def test_diversity(self):
        for ts in self.get_ts():
            self.check_pairwise_diversity(ts)
            self.check_divergence_matrix(ts)

    @unittest.skip("No SFS.")
    def test_branch_sfs(self):
        for ts in self.get_ts():
            self.check_sfs(ts)


class SiteStatsTestCase(GeneralStatsTestCase):
    """
    Tests of site statistic computation.
    """
    stat_class = "site"
    py_stat_class = PythonSiteStatCalculator
    seed = 23

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
                                             multiple_per_node=mpn, seed=self.seed)
                yield mut_ts

    def check_pairwise_diversity_mutations(self, ts):
        py_tsc = PythonSiteStatCalculator(ts)
        samples = random.sample(list(ts.samples()), 2)
        A = [[samples[0]], [samples[1]]]
        n = [len(a) for a in A]

        def f(x):
            return np.array([float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])/float(2*n[0]*n[1])])

        self.assertAlmostEqual(
            py_tsc.tree_stat_vector(A, f).flatten(), ts.pairwise_diversity(samples=samples))

    def test_pairwise_diversity(self):
        ts = msprime.simulate(20, random_seed=self.random_seed, recombination_rate=100)
        self.check_pairwise_diversity_mutations(ts)

    def test_site_general_stats(self):
        for ts in self.get_ts():
            self.check_tree_stat_vector(ts)

    def test_site_f_stats(self):
        for ts in self.get_ts():
            self.check_f_stats(ts)

    def test_site_Y_stats(self):
        for ts in self.get_ts():
            self.check_Y_stat(ts)

    def test_site_sfs(self):
        for ts in self.get_ts():
            self.check_sfs(ts)
