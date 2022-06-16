# MIT License
#
# Copyright (c) 2019-2022 Tskit Developers
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
import numpy as np
import pytest

import tests.tsutil as tsutil
import tskit


class TestJukesCantor:
    """
    Check that the we get useable tree sequences.
    """

    def verify(self, ts):
        tables = ts.dump_tables()
        tables.compute_mutation_parents()
        tables.assert_equals(ts.tables)
        # This will catch inconsistent mutations.
        assert ts.genotype_matrix() is not None

    def test_n10_multiroot(self):
        ts = msprime.simulate(10, random_seed=1)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        ts = tsutil.jukes_cantor(ts, 1, 2, seed=7)
        self.verify(ts)

    def test_n50_multiroot(self):
        ts = msprime.simulate(50, random_seed=1)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=2)
        self.verify(ts)

    def test_silent_mutations(self):
        ts = msprime.simulate(50, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=2)
        num_silent = 0
        for m in ts.mutations():
            if (
                m.parent != -1
                and ts.mutation(m.parent).derived_state == m.derived_state
            ):
                num_silent += 1
        assert num_silent > 20


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
        ts = tsutil.insert_individuals(ts, nodes=samples, ploidy=2)
        assert ts.num_individuals == 5
        for j, ind in enumerate(ts.individuals()):
            assert list(ind.nodes) == [samples[2 * j + 1], samples[2 * j]]


class TestSortIndividuals:
    def test_sort_individuals(self):
        tables = tskit.TableCollection()
        tables.individuals.add_row(parents=[1], metadata=b"0")
        tables.individuals.add_row(parents=[-1], metadata=b"1")
        tsutil.sort_individual_table(tables)
        assert tables.individuals.metadata.tobytes() == b"10"

        tables = tskit.TableCollection()
        tables.individuals.add_row(parents=[2, 3], metadata=b"0")
        tables.individuals.add_row(parents=[5], metadata=b"1")
        tables.individuals.add_row(parents=[-1], metadata=b"2")
        tables.individuals.add_row(parents=[-1], metadata=b"3")
        tables.individuals.add_row(parents=[3], metadata=b"4")
        tables.individuals.add_row(parents=[4], metadata=b"5")

        tsutil.sort_individual_table(tables)
        assert tables.individuals.metadata.tobytes() == b"342501"

        tables = tskit.TableCollection()
        tables.individuals.add_row(parents=[1], metadata=b"0")
        tables.individuals.add_row(parents=[0], metadata=b"1")
        with pytest.raises(ValueError, match="Individual pedigree has cycles"):
            tsutil.sort_individual_table(tables)


class TestQuintuplyLinkedTrees:
    def test_branch_operations_num_children(self):
        qlt = tsutil.QuintuplyLinkedTree(3)
        assert np.sum(qlt.num_children) == 0
        qlt.insert_branch(2, 0)
        assert qlt.num_children[2] == 1
        assert np.sum(qlt.num_children) == 1

        qlt.remove_branch(2, 0)
        assert qlt.num_children[2] == 0

    def test_edge_operations(self):
        tt = tskit.Tree.generate_balanced(3)
        tts = tt.tree_sequence

        for _, qlt in tsutil.algorithm_R(tts):
            assert np.sum(qlt.edge != -1) == tt.num_edges
            self.verify_tree_edges(qlt, tts)

    def verify_tree_edges(self, quintuply_linked_tree, tts):
        for edge in tts.edges():
            assert quintuply_linked_tree.edge[edge.child] == edge.id
            assert quintuply_linked_tree.parent[edge.child] == edge.parent
