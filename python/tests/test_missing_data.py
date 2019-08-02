# MIT License
#
# Copyright (c) 2019 Tskit Developers
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
Test cases for missing data.
"""
import bisect
import unittest

import numpy as np
import msprime


import tskit


def get_ancestral_haplotypes(ts):
    """
    Returns a numpy array of the haplotypes of the ancestors in the
    specified tree sequence.
    """
    tables = ts.dump_tables()
    nodes = tables.nodes
    flags = nodes.flags[:]
    flags[:] = 1
    nodes.set_columns(time=nodes.time, flags=flags)

    sites = tables.sites.position
    tsp = tables.tree_sequence()
    B = tsp.genotype_matrix().T

    A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.int8)
    A[:] = tskit.MISSING_DATA
    for edge in ts.edges():
        start = bisect.bisect_left(sites, edge.left)
        end = bisect.bisect_right(sites, edge.right)
        if sites[end - 1] == edge.right:
            end -= 1
        A[edge.parent, start:end] = B[edge.parent, start:end]
    A[:ts.num_samples] = B[:ts.num_samples]
    return A



class TestGetAncestralHaplotypes(unittest.TestCase):
    """
    Tests for the engine to the actual ancestors from a simulation.
    """
    def get_matrix(self, ts):
        """
        Simple implementation using tree traversals.
        """
        A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.int8)
        A[:] = tskit.MISSING_DATA
        for t in ts.trees():
            for site in t.sites():
                for u in t.nodes():
                    A[u, site.id] = 0
                for mutation in site.mutations:
                    # Every node underneath this node will have the value set
                    # at this site.
                    for u in t.nodes(mutation.node):
                        A[u, site.id] = 1
        return A

    def verify_samples(self, ts, A):
        # Samples should be nodes rows 0 to n - 1, and should be equal to
        # the genotypes.
        G = ts.genotype_matrix()
        self.assertTrue(np.array_equal(G.T, A[:ts.num_samples]))

    def verify_haplotypes(self, ts, A):
        self.verify_samples(ts, A)
        for tree in ts.trees():
            for site in tree.sites():
                self.assertEqual(len(site.mutations), 1)
                mutation = site.mutations[0]
                below = np.array(list(tree.nodes(mutation.node)), dtype=int)
                self.assertTrue(np.all(A[below, site.id] == 1))
                above = np.array(list(
                    set(tree.nodes()) - set(tree.nodes(mutation.node))), dtype=int)
                self.assertTrue(np.all(A[above, site.id] == 0))
                outside = np.array(list(
                    set(range(ts.num_nodes)) - set(tree.nodes())), dtype=int)
                self.assertTrue(np.all(A[outside, site.id] == tskit.MISSING_DATA))

    def verify(self, ts):
        A = get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        # Build the matrix using the variants iterator
        nodes = np.arange(ts.num_nodes, dtype=np.int32)
        # C = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.int8)
        # for var in ts.variants(samples=nodes):
        #     C[:, var.index] = var.genotypes
        # print(C)
        self.assertTrue(np.array_equal(A, B))
        # self.assertTrue(np.array_equal(A, C))
        self.verify_haplotypes(ts, A)

    def test_single_tree(self):
        ts = msprime.simulate(5, mutation_rate=10, random_seed=234)
        self.verify(ts)

    def test_many_trees(self):
        ts = msprime.simulate(
            8, recombination_rate=10, mutation_rate=10, random_seed=234)
        self.assertGreater(ts.num_trees, 1)
        self.assertGreater(ts.num_sites, 1)
        self.verify(ts)
