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
Tests for the various testing utilities.
"""
import unittest

import msprime

import tests.tsutil as tsutil


class TestJukesCantor(unittest.TestCase):
    """
    Check that the we get useable tree sequences.
    """
    def verify(self, ts):
        tables = ts.dump_tables()
        tables.compute_mutation_parents()
        self.assertEqual(tables, ts.tables)
        # This will catch inconsistent mutations.
        self.assertIsNotNone(ts.genotype_matrix())

    def test_n10_multiroot(self):
        ts = msprime.simulate(10, random_seed=1)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        ts = tsutil.jukes_cantor(ts, 1, 2, seed=7)
        self.verify(ts)

    def test_n50_multiroot(self):
        ts = msprime.simulate(50, random_seed=1)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=2)
        self.verify(ts)


class TestInsertIndividuals(unittest.TestCase):
    """
    Test that we insert individuals correctly.
    """
    def test_ploidy_1(self):
        ts = msprime.simulate(10, random_seed=1)
        self.assertEqual(ts.num_individuals, 0)
        ts = tsutil.insert_individuals(ts, ploidy=1)
        self.assertEqual(ts.num_individuals, 10)
        for j, ind in enumerate(ts.individuals()):
            self.assertEqual(list(ind.nodes), [j])

    def test_ploidy_2(self):
        ts = msprime.simulate(10, random_seed=1)
        self.assertEqual(ts.num_individuals, 0)
        ts = tsutil.insert_individuals(ts, ploidy=2)
        self.assertEqual(ts.num_individuals, 5)
        for j, ind in enumerate(ts.individuals()):
            self.assertEqual(list(ind.nodes), [2 * j, 2 * j + 1])

    def test_ploidy_2_reversed(self):
        ts = msprime.simulate(10, random_seed=1)
        self.assertEqual(ts.num_individuals, 0)
        samples = ts.samples()[::-1]
        ts = tsutil.insert_individuals(ts, samples=samples, ploidy=2)
        self.assertEqual(ts.num_individuals, 5)
        for j, ind in enumerate(ts.individuals()):
            self.assertEqual(list(ind.nodes), [samples[2 * j + 1], samples[2 * j]])
