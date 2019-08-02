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
import unittest

import numpy as np
import msprime

import tskit
import tests.tsutil as tsutil
import tests.test_wright_fisher as wf


def naive_get_ancestral_haplotypes(ts):
    """
    Simple implementation using tree traversals. Note that this definition
    won't work when we have topology that's not reachable from a root,
    but this seems more trouble than it's worth dealing with.
    """
    A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.int8)
    A[:] = tskit.MISSING_DATA
    for t in ts.trees():
        for site in t.sites():
            alleles = {site.ancestral_state: 0}
            for u in t.nodes():
                A[u, site.id] = 0
            j = 1
            for mutation in site.mutations:
                if mutation.derived_state not in alleles:
                    alleles[mutation.derived_state] = j
                    j += 1
                for u in t.nodes(mutation.node):
                    A[u, site.id] = alleles[mutation.derived_state]
    return A


class TestGetAncestralHaplotypes(unittest.TestCase):
    """
    Tests for the engine to the actual ancestors from a simulation.
    """

    def verify(self, ts):
        A = naive_get_ancestral_haplotypes(ts)
        # To detect missing data in ancestors we must set all nodes
        # to be samples
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags[:]
        flags[:] = 1
        nodes.set_columns(time=nodes.time, flags=flags)
        ts = tables.tree_sequence()
        B = ts.genotype_matrix().T
        self.assertTrue(np.array_equal(A, B))

    def test_single_tree(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=234)
        self.verify(ts)

    def test_many_trees(self):
        ts = msprime.simulate(
            8, recombination_rate=10, mutation_rate=10, random_seed=234)
        self.assertGreater(ts.num_trees, 1)
        self.assertGreater(ts.num_sites, 1)
        self.verify(ts)

    def test_single_tree_jukes_cantor(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        self.verify(ts)

    def test_single_tree_multichar_mutations(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.insert_multichar_mutations(ts)
        self.verify(ts)

    def test_many_trees_infinite_sites(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=2, random_seed=1)
        self.assertGreater(ts.num_sites, 0)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts)

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            6, 5, seed=3, deep_history=True, initial_generation_samples=True,
            num_loci=2)
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.08, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            9, 10, seed=1, deep_history=True, initial_generation_samples=False,
            num_loci=5)
        tables.sort()
        ts = tables.tree_sequence().simplify()
        ts = msprime.mutate(ts, rate=0.01, random_seed=1234)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_empty_ts(self):
        tables = tskit.TableCollection(1.0)
        for _ in range(10):
            tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        ts = tables.tree_sequence()
        self.verify(ts)
