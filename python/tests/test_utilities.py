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
import msprime
import pytest

import tests.tsutil as tsutil


class TestJukesCantor:
    """
    Check that the we get useable tree sequences.
    """

    def verify(self, ts):
        tables = ts.dump_tables()
        tables.compute_mutation_parents()
        assert tables == ts.tables
        # This will catch inconsistent mutations.
        assert ts.genotype_matrix() is not None

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


class TestCaterpillarTree:
    """
    Tests for the caterpillar tree method.
    """

    def verify(self, ts, n):
        assert ts.num_trees == 1
        assert ts.num_nodes == ts.num_samples * 2 - 1
        tree = ts.first()
        for j in range(1, n):
            assert tree.parent(j) == n + j - 1
        # This will catch inconsistent mutations.
        assert ts.genotype_matrix() is not None

    def test_n_2(self):
        ts = tsutil.caterpillar_tree(2)
        self.verify(ts, 2)

    def test_n_3(self):
        ts = tsutil.caterpillar_tree(3)
        self.verify(ts, 3)

    def test_n_50(self):
        ts = tsutil.caterpillar_tree(50)
        self.verify(ts, 50)

    def test_n_5_sites(self):
        ts = tsutil.caterpillar_tree(5, num_sites=4)
        self.verify(ts, 5)
        assert ts.num_sites == 4
        assert ts.num_mutations == 4
        assert list(ts.tables.sites.position) == [0.2, 0.4, 0.6, 0.8]
        ts = tsutil.caterpillar_tree(5, num_sites=1, num_mutations=1)
        assert ts.num_sites == 1
        assert ts.num_mutations == 1
        site = ts.site(0)
        assert site.mutations[0].node == 7

    def test_n_5_mutations(self):
        ts = tsutil.caterpillar_tree(5, num_sites=1, num_mutations=3)
        self.verify(ts, 5)
        assert ts.num_sites == 1
        assert ts.num_mutations == 3
        node = ts.tables.mutations.node
        assert list(node) == [7, 6, 5]

    def test_n_many_mutations(self):
        for n in range(10, 15):
            for num_mutations in range(0, n - 1):
                ts = tsutil.caterpillar_tree(
                    n, num_sites=1, num_mutations=num_mutations
                )
                self.verify(ts, n)
                assert ts.num_sites == 1
                assert ts.num_mutations == num_mutations
            for num_mutations in range(n - 1, n + 2):
                with pytest.raises(ValueError):
                    tsutil.caterpillar_tree(n, num_sites=1, num_mutations=num_mutations)


class TestInsertIndividuals:
    """
    Test that we insert individuals correctly.
    """

    def test_ploidy_1(self):
        ts = msprime.simulate(10, random_seed=1)
        assert ts.num_individuals == 0
        ts = tsutil.insert_individuals(ts, ploidy=1)
        assert ts.num_individuals == 10
        for j, ind in enumerate(ts.individuals()):
            assert list(ind.nodes) == [j]

    def test_ploidy_2(self):
        ts = msprime.simulate(10, random_seed=1)
        assert ts.num_individuals == 0
        ts = tsutil.insert_individuals(ts, ploidy=2)
        assert ts.num_individuals == 5
        for j, ind in enumerate(ts.individuals()):
            assert list(ind.nodes) == [2 * j, 2 * j + 1]

    def test_ploidy_2_reversed(self):
        ts = msprime.simulate(10, random_seed=1)
        assert ts.num_individuals == 0
        samples = ts.samples()[::-1]
        ts = tsutil.insert_individuals(ts, samples=samples, ploidy=2)
        assert ts.num_individuals == 5
        for j, ind in enumerate(ts.individuals()):
            assert list(ind.nodes) == [samples[2 * j + 1], samples[2 * j]]
