#
# MIT License
#
# Copyright (c) 2020 Tskit Developers
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
Test cases for combinatorial algorithms.
"""
import collections
import io
import itertools
import random

import msprime
import numpy as np
import pytest

import tests.test_wright_fisher as wf
import tskit
import tskit.combinatorics as comb
from tests import test_stats
from tskit.combinatorics import RankTree


class TestCombination:
    def test_combination_with_replacement_rank_unrank(self):
        for n in range(9):
            for k in range(n):
                nums = list(range(n))
                combs = itertools.combinations_with_replacement(nums, k)
                for exp_rank, c in enumerate(combs):
                    c = list(c)
                    actual_rank = comb.Combination.with_replacement_rank(c, n)
                    assert actual_rank == exp_rank
                    unranked = comb.Combination.with_replacement_unrank(exp_rank, n, k)
                    assert unranked == c

    def test_combination_rank_unrank(self):
        for n in range(11):
            for k in range(n):
                nums = list(range(n))
                for rank, c in enumerate(itertools.combinations(nums, k)):
                    c = list(c)
                    assert comb.Combination.rank(c, nums) == rank
                    assert comb.Combination.unrank(rank, nums, k) == c

    def test_combination_unrank_errors(self):
        self.verify_unrank_errors(1, 1, 1)
        self.verify_unrank_errors(2, 0, 1)

    def verify_unrank_errors(self, rank, n, k):
        with pytest.raises(ValueError):
            comb.Combination.unrank(rank, list(range(n)), k)


class TestPartition:
    def test_rule_asc(self):
        self.verify_rule_asc(1, [[1]])
        self.verify_rule_asc(2, [[1, 1], [2]])
        self.verify_rule_asc(3, [[1, 1, 1], [1, 2], [3]])
        self.verify_rule_asc(4, [[1, 1, 1, 1], [1, 1, 2], [1, 3], [2, 2], [4]])
        self.verify_rule_asc(
            5,
            [[1, 1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 3], [1, 2, 2], [1, 4], [2, 3], [5]],
        )
        self.verify_rule_asc(
            6,
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 1, 1, 3],
                [1, 1, 2, 2],
                [1, 1, 4],
                [1, 2, 3],
                [1, 5],
                [2, 2, 2],
                [2, 4],
                [3, 3],
                [6],
            ],
        )

    def verify_rule_asc(self, n, partitions):
        assert list(comb.rule_asc(n)) == partitions

    def test_partitions(self):
        assert list(comb.partitions(0)) == []
        for n in range(1, 7):
            assert list(comb.partitions(n)) == list(comb.rule_asc(n))[:-1]

    def test_group_partition(self):
        assert comb.group_partition([1]) == [[1]]
        assert comb.group_partition([1, 2]) == [[1], [2]]
        assert comb.group_partition([1, 1, 1]) == [[1, 1, 1]]
        assert comb.group_partition([1, 1, 2, 3, 3]) == [[1, 1], [2], [3, 3]]


class TestRankTree:
    def test_num_shapes(self):
        for i in range(11):
            all_trees = RankTree.all_unlabelled_trees(i)
            assert len(list(all_trees)) == comb.num_shapes(i)

    def test_num_labellings(self):
        for n in range(2, 8):
            for tree in RankTree.all_unlabelled_trees(n):
                tree = tree.label_unrank(0)
                tree2 = tree.to_tsk_tree()
                n_labellings = sum(1 for _ in RankTree.all_labellings(tree))
                assert n_labellings == RankTree.from_tsk_tree(tree2).num_labellings()

    def test_num_labelled_trees(self):
        # Number of leaf-labelled trees with n leaves on OEIS
        n_trees = [0, 1, 1, 4, 26, 236, 2752, 39208]
        for i, expected in zip(range(len(n_trees)), n_trees):
            actual = sum(1 for _ in RankTree.all_labelled_trees(i))
            assert actual == expected

    def test_all_labelled_trees_3(self):
        expected = ["(0,1,2)", "(0,(1,2))", "(1,(0,2))", "(2,(0,1))"]
        actual = [t.newick() for t in RankTree.all_labelled_trees(3)]
        assert expected == actual

    def test_all_labelled_trees_4(self):
        expected = [
            # 1 + 1 + 1 + 1 (partition of num leaves)
            "(0,1,2,3)",
            # 1 + 1 + 2
            "(0,1,(2,3))",
            "(0,2,(1,3))",
            "(0,3,(1,2))",
            "(1,2,(0,3))",
            "(1,3,(0,2))",
            "(2,3,(0,1))",
            # 1 + 3
            # partition of 3 = 1 + 1 + 1
            "(0,(1,2,3))",
            "(1,(0,2,3))",
            "(2,(0,1,3))",
            "(3,(0,1,2))",
            # partition of 3 = 1 + 2
            "(0,(1,(2,3)))",
            "(0,(2,(1,3)))",
            "(0,(3,(1,2)))",
            "(1,(0,(2,3)))",
            "(1,(2,(0,3)))",
            "(1,(3,(0,2)))",
            "(2,(0,(1,3)))",
            "(2,(1,(0,3)))",
            "(2,(3,(0,1)))",
            "(3,(0,(1,2)))",
            "(3,(1,(0,2)))",
            "(3,(2,(0,1)))",
            # 2 + 2
            "((0,1),(2,3))",
            "((0,2),(1,3))",
            "((0,3),(1,2))",
        ]
        actual = [t.newick() for t in RankTree.all_labelled_trees(4)]
        assert expected == actual

    def test_generate_trees_roundtrip(self):
        n = 5
        all_rank_trees = RankTree.all_labelled_trees(n)
        all_tsk_trees = tskit.all_trees(n)
        for rank_tree, tsk_tree in zip(all_rank_trees, all_tsk_trees):
            assert rank_tree == RankTree.from_tsk_tree(tsk_tree)

    def test_generate_treeseq_roundtrip(self):
        n = 5
        span = 9
        all_rank_trees = RankTree.all_labelled_trees(n)
        all_tsk_trees = tskit.all_trees(n, span=span)
        for rank_tree, tsk_tree in zip(all_rank_trees, all_tsk_trees):
            ts1 = tsk_tree.tree_sequence
            ts2 = rank_tree.to_tsk_tree(span=span).tree_sequence
            assert ts1.tables.equals(ts2.tables, ignore_provenance=True)

    def test_all_shapes_roundtrip(self):
        n = 5
        all_rank_tree_shapes = RankTree.all_unlabelled_trees(n)
        all_tsk_tree_shapes = tskit.all_tree_shapes(n)
        for rank_tree, tsk_tree in zip(all_rank_tree_shapes, all_tsk_tree_shapes):
            assert rank_tree.shape_equal(RankTree.from_tsk_tree(tsk_tree))

    def test_all_labellings_roundtrip(self):
        n = 5
        rank_tree = RankTree.unrank(n, (comb.num_shapes(n) - 1, 0))
        tsk_tree = rank_tree.to_tsk_tree()
        rank_tree_labellings = RankTree.all_labellings(rank_tree)
        tsk_tree_labellings = tskit.all_tree_labellings(tsk_tree)
        for rank_t, tsk_t in zip(rank_tree_labellings, tsk_tree_labellings):
            assert rank_t == RankTree.from_tsk_tree(tsk_t)

    def test_unrank(self):
        for n in range(6):
            for shape_rank, t in enumerate(RankTree.all_unlabelled_trees(n)):
                for label_rank, labelled_tree in enumerate(RankTree.all_labellings(t)):
                    unranked = RankTree.unrank(n, (shape_rank, label_rank))
                    assert labelled_tree == unranked

        # The number of labelled trees gets very big quickly
        for n in range(6, 10):
            for shape_rank in range(comb.num_shapes(n)):
                rank = (shape_rank, 0)
                unranked = RankTree.unrank(n, rank)
                assert rank, unranked.rank()

                rank = (shape_rank, comb.num_labellings(n, shape_rank) - 1)
                unranked = RankTree.unrank(n, rank)
                assert rank, unranked.rank()

    def test_unrank_errors(self):
        self.verify_unrank_errors((-1, 0), 1)
        self.verify_unrank_errors((0, -1), 1)
        self.verify_unrank_errors((-1, 0), 2)
        self.verify_unrank_errors((0, -1), 2)
        self.verify_unrank_errors((-1, 0), 10)
        self.verify_unrank_errors((0, -1), 10)

        self.verify_unrank_errors((0, 1), 1)
        self.verify_unrank_errors((1, 0), 2)
        self.verify_unrank_errors((0, 1), 2)
        self.verify_unrank_errors((2, 0), 3)
        self.verify_unrank_errors((0, 1), 3)
        self.verify_unrank_errors((1, 3), 3)

        invalid_shape = (comb.num_shapes(10), 0)
        self.verify_unrank_errors(invalid_shape, 10)
        invalid_labelling = (0, comb.num_labellings(10, 0))
        self.verify_unrank_errors(invalid_labelling, 10)

    def verify_unrank_errors(self, rank, n):
        with pytest.raises(ValueError):
            RankTree.unrank(n, rank)
        with pytest.raises(ValueError):
            tskit.Tree.unrank(n, rank)

    def test_shape_rank(self):
        for n in range(10):
            for rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
                assert tree.shape_rank() == rank

    def test_shape_unrank(self):
        for n in range(6):
            for rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
                t = RankTree.shape_unrank(n, rank)
                assert tree.shape_equal(t)

        for n in range(2, 9):
            for shape_rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
                tsk_tree = tskit.Tree.unrank(n, (shape_rank, 0))
                assert shape_rank == tree.shape_rank()
                shape_rank, _ = tsk_tree.rank()
                assert shape_rank == tree.shape_rank()

    def test_label_rank(self):
        for n in range(7):
            for tree in RankTree.all_unlabelled_trees(n):
                for rank, labelled_tree in enumerate(RankTree.all_labellings(tree)):
                    assert labelled_tree.label_rank() == rank

    def test_label_unrank(self):
        for n in range(7):
            for shape_rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
                for label_rank, labelled_tree in enumerate(
                    RankTree.all_labellings(tree)
                ):
                    rank = (shape_rank, label_rank)
                    unranked = tree.label_unrank(label_rank)
                    assert labelled_tree.rank() == rank
                    assert unranked.rank() == rank

    def test_unrank_rank_round_trip(self):
        for n in range(6):  # Can do more but gets slow pretty quickly after 6
            for shape_rank in range(comb.num_shapes(n)):
                tree = RankTree.shape_unrank(n, shape_rank)
                tree = tree.label_unrank(0)
                assert tree.shape_rank() == shape_rank
                for label_rank in range(tree.num_labellings()):
                    tree = tree.label_unrank(label_rank)
                    assert tree.label_rank() == label_rank
                    tsk_tree = tree.label_unrank(label_rank).to_tsk_tree()
                    _, tsk_label_rank = tsk_tree.rank()
                    assert tsk_label_rank == label_rank

    def test_is_canonical(self):
        for n in range(7):
            for tree in RankTree.all_labelled_trees(n):
                assert tree.is_canonical()

        shape_not_canonical = RankTree(
            children=[
                RankTree(children=[], label=0),
                RankTree(
                    children=[
                        RankTree(
                            children=[
                                RankTree(children=[], label=1),
                                RankTree(children=[], label=2),
                            ]
                        ),
                        RankTree(children=[], label=3),
                    ]
                ),
            ]
        )
        assert not shape_not_canonical.is_canonical()

        labels_not_canonical = RankTree(
            children=[
                RankTree(children=[], label=0),
                RankTree(
                    children=[
                        RankTree(
                            children=[
                                RankTree(children=[], label=2),
                                RankTree(children=[], label=3),
                            ]
                        ),
                        RankTree(
                            children=[
                                RankTree(children=[], label=1),
                                RankTree(children=[], label=4),
                            ]
                        ),
                    ]
                ),
            ]
        )
        assert not labels_not_canonical.is_canonical()

    def test_unranking_is_canonical(self):
        for n in range(7):
            for shape_rank in range(comb.num_shapes(n)):
                for label_rank in range(comb.num_labellings(n, shape_rank)):
                    t = RankTree.shape_unrank(n, shape_rank)
                    assert t.is_canonical()
                    t = t.label_unrank(label_rank)
                    assert t.is_canonical()
                    t = tskit.Tree.unrank(n, (shape_rank, label_rank))
                    assert RankTree.from_tsk_tree(t).is_canonical()

    def test_to_from_tsk_tree(self):
        for n in range(5):
            for tree in RankTree.all_labelled_trees(n):
                assert tree.is_canonical()
                tsk_tree = tree.to_tsk_tree()
                reconstructed = RankTree.from_tsk_tree(tsk_tree)
                assert tree.is_canonical()
                assert tree == reconstructed

    def test_from_unary_tree(self):
        tables = tskit.TableCollection(sequence_length=1)
        c = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        p = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c)

        t = tables.tree_sequence().first()
        with pytest.raises(ValueError):
            RankTree.from_tsk_tree(t)

    def test_to_tsk_tree_errors(self):
        alpha_tree = RankTree.unrank(3, (0, 0), ["A", "B", "C"])
        out_of_bounds_tree = RankTree.unrank(3, (0, 0), [2, 3, 4])
        with pytest.raises(ValueError):
            alpha_tree.to_tsk_tree()
        with pytest.raises(ValueError):
            out_of_bounds_tree.to_tsk_tree()

    def test_rank_errors_multiple_roots(self):
        tables = tskit.TableCollection(sequence_length=1.0)

        # Nodes
        sv = [True, True]
        tv = [0.0, 0.0]

        for is_sample, t in zip(sv, tv):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables.nodes.add_row(flags=flags, time=t)

        ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            ts.first().rank()

    def test_span(self):
        n = 5
        span = 8
        # Create a start tree, with a single root
        tsk_tree = tskit.Tree.unrank(n, (0, 0), span=span)
        assert tsk_tree.num_nodes == n + 1
        assert tsk_tree.interval.left == 0
        assert tsk_tree.interval.right == span
        assert tsk_tree.tree_sequence.sequence_length == span

    def test_big_trees(self):
        n = 14
        shape = 22
        labelling = 0
        tree = RankTree.unrank(n, (shape, labelling))
        tsk_tree = tskit.Tree.unrank(n, (shape, labelling))
        assert tree.rank() == tsk_tree.rank()

        n = 10
        shape = 95
        labelling = comb.num_labellings(n, shape) // 2
        tree = RankTree.unrank(n, (shape, labelling))
        tsk_tree = tskit.Tree.unrank(n, (shape, labelling))
        assert tree.rank() == tsk_tree.rank()

    def test_symmetrical_trees(self):
        for n in range(2, 18, 2):
            last_rank = comb.num_shapes(n) - 1
            t = RankTree.shape_unrank(n, last_rank)
            assert t.is_symmetrical()

    def test_equal(self):
        unlabelled_leaf = RankTree(children=[])
        assert unlabelled_leaf == unlabelled_leaf
        assert unlabelled_leaf.shape_equal(unlabelled_leaf)

        leaf_zero = RankTree(children=[], label=0)
        leaf_one = RankTree(children=[], label=1)
        leaf_two = RankTree(children=[], label=2)
        assert leaf_zero == leaf_zero
        assert leaf_zero != leaf_one
        assert leaf_zero.shape_equal(leaf_one)

        tree1 = RankTree(children=[leaf_zero, leaf_one])
        assert tree1 == tree1
        assert tree1 != unlabelled_leaf
        assert not tree1.shape_equal(unlabelled_leaf)

        tree2 = RankTree(children=[leaf_two, leaf_one])
        assert tree1 != tree2
        assert tree1.shape_equal(tree2)

    def test_is_symmetrical(self):
        unlabelled_leaf = RankTree(children=[])
        assert unlabelled_leaf.is_symmetrical()
        three_leaf_asym = RankTree(
            children=[
                unlabelled_leaf,
                RankTree(children=[unlabelled_leaf, unlabelled_leaf]),
            ]
        )
        assert not three_leaf_asym.is_symmetrical()
        six_leaf_sym = RankTree(children=[three_leaf_asym, three_leaf_asym])
        assert six_leaf_sym.is_symmetrical()


class TestPartialTopologyCounter:
    def test_add_sibling_topologies_simple(self):
        a = RankTree(children=[], label="A")
        b = RankTree(children=[], label="B")
        ab = RankTree(children=[a, b])

        a_counter = comb.TopologyCounter()
        a_counter["A"][a.rank()] = 1
        assert a_counter == comb.TopologyCounter.from_sample("A")

        b_counter = comb.TopologyCounter()
        b_counter["B"][b.rank()] = 1
        assert b_counter == comb.TopologyCounter.from_sample("B")

        partial_counter = comb.PartialTopologyCounter()
        partial_counter.add_sibling_topologies(a_counter)
        partial_counter.add_sibling_topologies(b_counter)

        expected = comb.TopologyCounter()
        expected["A"][a.rank()] = 1
        expected["B"][b.rank()] = 1
        expected["A", "B"][ab.rank()] = 1
        joined_counter = partial_counter.join_all_combinations()
        assert joined_counter == expected

    def test_add_sibling_topologies_polytomy(self):
        """
        Goes through the topology-merging step at the root
        of this tree:
                    |
                    |
            +----+-----+----+
            |    |     |    |
            |    |     |    |
            |    |     |  +---+
            |    |     |  |   |
            |    |     |  |   |
            A    A     B  A   C
        """
        partial_counter = comb.PartialTopologyCounter()
        a = RankTree(children=[], label="A")
        c = RankTree(children=[], label="C")
        ac = RankTree(children=[a, c])

        expected = collections.defaultdict(collections.Counter)

        a_counter = comb.TopologyCounter.from_sample("A")
        b_counter = comb.TopologyCounter.from_sample("B")
        ac_counter = comb.TopologyCounter()
        ac_counter["A"][a.rank()] = 1
        ac_counter["C"][c.rank()] = 1
        ac_counter["A", "C"][ac.rank()] = 1

        partial_counter.add_sibling_topologies(a_counter)
        expected[("A",)] = collections.Counter({((("A",), (0, 0)),): 1})
        assert partial_counter.partials == expected

        partial_counter.add_sibling_topologies(a_counter)
        expected[("A",)][((("A",), (0, 0)),)] += 1
        assert partial_counter.partials == expected

        partial_counter.add_sibling_topologies(b_counter)
        expected[("B",)][((("B",), (0, 0)),)] = 1
        expected[("A", "B")][((("A",), (0, 0)), (("B",), (0, 0)))] = 2
        assert partial_counter.partials == expected

        partial_counter.add_sibling_topologies(ac_counter)
        expected[("A",)][((("A",), (0, 0)),)] += 1
        expected[("C",)][((("C",), (0, 0)),)] = 1
        expected[("A", "B")][((("A",), (0, 0)), (("B",), (0, 0)))] += 1
        expected[("A", "C")][((("A",), (0, 0)), (("C",), (0, 0)))] = 2
        expected[("A", "C")][((("A", "C"), (0, 0)),)] = 1
        expected[("B", "C")][((("B",), (0, 0)), (("C",), (0, 0)))] = 1
        expected[("A", "B", "C")][
            ((("A",), (0, 0)), (("B",), (0, 0)), (("C",), (0, 0)))
        ] = 2
        expected[("A", "B", "C")][((("A", "C"), (0, 0)), (("B",), (0, 0)))] = 1
        assert partial_counter.partials == expected

        expected_topologies = comb.TopologyCounter()
        expected_topologies["A"][(0, 0)] = 3
        expected_topologies["B"][(0, 0)] = 1
        expected_topologies["C"][(0, 0)] = 1
        expected_topologies["A", "B"][(0, 0)] = 3
        expected_topologies["A", "C"][(0, 0)] = 3
        expected_topologies["B", "C"][(0, 0)] = 1
        expected_topologies["A", "B", "C"][(0, 0)] = 2
        expected_topologies["A", "B", "C"][(1, 1)] = 1
        joined_topologies = partial_counter.join_all_combinations()
        assert joined_topologies == expected_topologies

    def test_join_topologies(self):
        a = RankTree(children=[], label="A")
        b = RankTree(children=[], label="B")
        c = RankTree(children=[], label="C")
        a_tuple = (("A"), a.rank())
        b_tuple = (("B"), b.rank())
        c_tuple = (("C"), c.rank())
        ab_tuple = (("A", "B"), RankTree(children=[a, b]).rank())
        ac_tuple = (("A", "C"), RankTree(children=[a, c]).rank())
        bc_tuple = (("B", "C"), RankTree(children=[b, c]).rank())

        self.verify_join_topologies((a_tuple, b_tuple), (0, 0))
        self.verify_join_topologies((b_tuple, a_tuple), (0, 0))
        self.verify_join_topologies((b_tuple, c_tuple), (0, 0))

        self.verify_join_topologies((a_tuple, b_tuple, c_tuple), (0, 0))
        self.verify_join_topologies((a_tuple, bc_tuple), (1, 0))
        self.verify_join_topologies((b_tuple, ac_tuple), (1, 1))
        self.verify_join_topologies((c_tuple, ab_tuple), (1, 2))

    def verify_join_topologies(self, topologies, expected_topology):
        actual_topology = comb.PartialTopologyCounter.join_topologies(topologies)
        assert actual_topology == expected_topology


class TestCountTopologies:
    def verify_topologies(self, ts, sample_sets=None, expected=None):
        if sample_sets is None:
            sample_sets = [ts.samples(population=pop.id) for pop in ts.populations()]
        topologies = [t.count_topologies(sample_sets) for t in ts.trees()]
        inc_topologies = list(ts.count_topologies(sample_sets))
        # count_topologies calculates the embedded topologies for every
        # combination of populations, so we need to check the results
        # of subsampling for every combination.
        for num_sample_sets in range(1, len(sample_sets) + 1):
            for i, t in enumerate(ts.trees()):
                just_t = ts.keep_intervals([t.interval], simplify=False)
                for sample_set_indexes in itertools.combinations(
                    range(len(sample_sets)), num_sample_sets
                ):
                    actual_topologies = topologies[i][sample_set_indexes]
                    actual_inc_topologies = inc_topologies[i][sample_set_indexes]
                    if len(t.roots) == 1:
                        subsampled_topologies = self.subsample_topologies(
                            just_t, sample_sets, sample_set_indexes
                        )
                        assert actual_topologies == subsampled_topologies
                    if expected is not None:
                        assert actual_topologies == expected[i][sample_set_indexes]
                    assert actual_topologies == actual_inc_topologies

    def subsample_topologies(self, ts, sample_sets, sample_set_indexes):
        subsample_sets = [sample_sets[i] for i in sample_set_indexes]
        topologies = collections.Counter()
        for subsample in itertools.product(*subsample_sets):
            for pop_tree in ts.simplify(samples=subsample).trees():
                # regions before and after keep interval have all samples as roots
                # so don't count those
                # The single tree of interest should have one root
                if len(pop_tree.roots) == 1:
                    topologies[pop_tree.rank()] += 1
        return topologies

    def test_single_population(self):
        n = 10
        ts = msprime.simulate(n, recombination_rate=10)
        expected = comb.TopologyCounter()
        expected[0] = collections.Counter({(0, 0): n})
        self.verify_topologies(ts, expected=[expected] * ts.num_trees)

    def test_three_populations(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    0   -1
        1   1   0.000000    1   -1
        2   1   0.000000    1   -1
        3   1   0.000000    2   -1
        4   1   0.000000    2   -1
        5   1   0.000000    0   -1
        6   0   1.000000    0   -1
        7   0   2.000000    0   -1
        8   0   2.000000    0   -1
        9   0   3.000000    0   -1
        10  0   4.000000    0   -1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.000000    1.000000    6  4
        0.000000    1.000000    6  5
        0.000000    1.000000    7  1
        0.000000    1.000000    7  2
        0.000000    1.000000    8  3
        0.000000    1.000000    8  6
        0.000000    1.000000    9  7
        0.000000    1.000000    9  8
        0.000000    1.000000    10  0
        0.000000    1.000000    10  9
        """
        )
        ts = tskit.load_text(
            nodes, edges, sequence_length=1, strict=False, base64_metadata=False
        )

        expected = comb.TopologyCounter()
        expected[0] = collections.Counter({(0, 0): 2})
        expected[1] = collections.Counter({(0, 0): 2})
        expected[2] = collections.Counter({(0, 0): 2})
        expected[0, 1] = collections.Counter({(0, 0): 4})
        expected[0, 2] = collections.Counter({(0, 0): 4})
        expected[1, 2] = collections.Counter({(0, 0): 4})
        expected[0, 1, 2] = collections.Counter({(1, 0): 4, (1, 1): 4})
        self.verify_topologies(ts, expected=[expected])

    def test_multiple_roots(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        tables.populations.add_row()
        tables.populations.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=1)

        # Not samples so they are ignored
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=1, population=1)

        expected = comb.TopologyCounter()
        expected[0] = collections.Counter({(0, 0): 1})
        expected[1] = collections.Counter({(0, 0): 1})
        self.verify_topologies(tables.tree_sequence(), expected=[expected])

    def test_no_sample_subtrees(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        c1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        c2 = tables.nodes.add_row(time=0)
        c3 = tables.nodes.add_row(time=0)
        p1 = tables.nodes.add_row(time=1)
        p2 = tables.nodes.add_row(time=1)

        tables.edges.add_row(left=0, right=1, parent=p1, child=c2)
        tables.edges.add_row(left=0, right=1, parent=p1, child=c3)
        tables.edges.add_row(left=0, right=1, parent=p2, child=c1)

        expected = comb.TopologyCounter()
        expected[0] = collections.Counter({(0, 0): 1})
        self.verify_topologies(tables.tree_sequence(), expected=[expected])

    def test_no_full_topology(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        tables.populations.add_row()
        tables.populations.add_row()
        tables.populations.add_row()
        child1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=0)
        child2 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=1)
        parent = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=0, right=1, parent=parent, child=child1)
        tables.edges.add_row(left=0, right=1, parent=parent, child=child2)

        # Left as root so there is no topology with all three populations
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=2)

        expected = comb.TopologyCounter()
        for pop_combo in [(0,), (1,), (2,), (0, 1)]:
            expected[pop_combo] = collections.Counter({(0, 0): 1})
        self.verify_topologies(tables.tree_sequence(), expected=[expected])

    def test_polytomies(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        tables.populations.add_row()
        tables.populations.add_row()
        tables.populations.add_row()
        c1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=0)
        c2 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=1)
        c3 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=2)
        p = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c2)
        tables.edges.add_row(left=0, right=1, parent=p, child=c3)

        expected = comb.TopologyCounter()
        for pop_combos in [0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
            expected[pop_combos] = collections.Counter({(0, 0): 1})
        self.verify_topologies(tables.tree_sequence(), expected=[expected])

    def test_custom_key(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    0   -1
        1   1   0.000000    0   -1
        2   1   0.000000    0   -1
        3   1   0.000000    0   -1
        4   1   0.000000    0   -1
        5   0   1.000000    0   -1
        6   0   1.000000    0   -1
        7   0   2.000000    0   -1
        8   0   3.000000    0   -1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.000000    1.000000    5  0
        0.000000    1.000000    5  1
        0.000000    1.000000    6  2
        0.000000    1.000000    6  3
        0.000000    1.000000    7  5
        0.000000    1.000000    7  6
        0.000000    1.000000    8  4
        0.000000    1.000000    8  7
        """
        )
        ts = tskit.load_text(
            nodes, edges, sequence_length=1, strict=False, base64_metadata=False
        )

        sample_sets = [[0, 1], [2, 3], [4]]

        expected = comb.TopologyCounter()
        expected[0] = collections.Counter({(0, 0): 2})
        expected[1] = collections.Counter({(0, 0): 2})
        expected[2] = collections.Counter({(0, 0): 1})
        expected[0, 1] = collections.Counter({(0, 0): 4})
        expected[0, 2] = collections.Counter({(0, 0): 2})
        expected[1, 2] = collections.Counter({(0, 0): 2})
        expected[0, 1, 2] = collections.Counter({(1, 2): 4})

        tree_topologies = ts.first().count_topologies(sample_sets)
        treeseq_topologies = list(ts.count_topologies(sample_sets))
        assert tree_topologies == expected
        assert treeseq_topologies == [expected]

    def test_ignores_non_sample_leaves(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    0   -1
        1   0   0.000000    0   -1
        2   1   0.000000    0   -1
        3   0   0.000000    0   -1
        4   1   0.000000    0   -1
        5   0   1.000000    0   -1
        6   0   1.000000    0   -1
        7   0   2.000000    0   -1
        8   0   3.000000    0   -1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.000000    1.000000    5  0
        0.000000    1.000000    5  1
        0.000000    1.000000    6  2
        0.000000    1.000000    6  3
        0.000000    1.000000    7  5
        0.000000    1.000000    7  6
        0.000000    1.000000    8  4
        0.000000    1.000000    8  7
        """
        )
        ts = tskit.load_text(
            nodes, edges, sequence_length=1, strict=False, base64_metadata=False
        )

        sample_sets = [[0], [2], [4]]

        expected = comb.TopologyCounter()
        expected[0] = collections.Counter({(0, 0): 1})
        expected[1] = collections.Counter({(0, 0): 1})
        expected[2] = collections.Counter({(0, 0): 1})
        expected[0, 1] = collections.Counter({(0, 0): 1})
        expected[0, 2] = collections.Counter({(0, 0): 1})
        expected[1, 2] = collections.Counter({(0, 0): 1})
        expected[0, 1, 2] = collections.Counter({(1, 2): 1})

        tree_topologies = ts.first().count_topologies(sample_sets)
        treeseq_topologies = list(ts.count_topologies(sample_sets))
        assert tree_topologies == expected
        assert treeseq_topologies == [expected]

    def test_internal_samples_errors(self):
        tables = tskit.TableCollection(sequence_length=1.0)

        c1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        c2 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        p = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=1)

        tables.edges.add_row(left=0, right=1, parent=p, child=c1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c2)

        self.verify_value_error(tables.tree_sequence())

    def test_non_sample_nodes_errors(self):
        tables = tskit.TableCollection(sequence_length=1.0)

        c1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        c2 = tables.nodes.add_row(time=0)
        p = tables.nodes.add_row(time=1)

        tables.edges.add_row(left=0, right=1, parent=p, child=c1)
        tables.edges.add_row(left=0, right=1, parent=p, child=c2)

        sample_sets = [[0], [1]]
        self.verify_value_error(tables.tree_sequence(), sample_sets)

        sample_sets = [[0], [-1]]
        self.verify_node_out_of_bounds_error(tables.tree_sequence(), sample_sets)

    def verify_value_error(self, ts, sample_sets=None):
        with pytest.raises(ValueError):
            ts.first().count_topologies(sample_sets)
        with pytest.raises(ValueError):
            list(ts.count_topologies(sample_sets))

    def verify_node_out_of_bounds_error(self, ts, sample_sets=None):
        with pytest.raises(ValueError):
            ts.first().count_topologies(sample_sets)
        with pytest.raises(IndexError):
            list(ts.count_topologies(sample_sets))

    def test_standard_msprime_migrations(self):
        for num_populations in range(2, 5):
            samples = [5] * num_populations
            ts = self.simulate_multiple_populations(samples)
            self.verify_topologies(ts)

    def simulate_multiple_populations(self, sample_sizes):
        d = len(sample_sizes)
        M = 0.2
        m = M / (2 * (d - 1))

        migration_matrix = [
            [m if k < d and k == i + 1 else 0 for k in range(d)] for i in range(d)
        ]

        pop_configurations = [
            msprime.PopulationConfiguration(sample_size=size) for size in sample_sizes
        ]
        return msprime.simulate(
            population_configurations=pop_configurations,
            migration_matrix=migration_matrix,
            recombination_rate=0.1,
        )

    def test_msprime_dtwf(self):
        migration_matrix = np.zeros((4, 4))
        population_configurations = [
            msprime.PopulationConfiguration(
                sample_size=10, initial_size=10, growth_rate=0
            ),
            msprime.PopulationConfiguration(
                sample_size=10, initial_size=10, growth_rate=0
            ),
            msprime.PopulationConfiguration(
                sample_size=10, initial_size=10, growth_rate=0
            ),
            msprime.PopulationConfiguration(
                sample_size=0, initial_size=10, growth_rate=0
            ),
        ]
        demographic_events = [
            msprime.PopulationParametersChange(population=1, time=0.1, initial_size=5),
            msprime.PopulationParametersChange(population=0, time=0.2, initial_size=5),
            msprime.MassMigration(time=1.1, source=0, dest=2),
            msprime.MassMigration(time=1.2, source=1, dest=3),
            msprime.MigrationRateChange(time=2.1, rate=0.3, matrix_index=(2, 3)),
            msprime.MigrationRateChange(time=2.2, rate=0.3, matrix_index=(3, 2)),
        ]
        ts = msprime.simulate(
            migration_matrix=migration_matrix,
            population_configurations=population_configurations,
            demographic_events=demographic_events,
            random_seed=2,
            model="dtwf",
        )

        self.verify_topologies(ts)

    def test_forward_time_wright_fisher_unsimplified_all_sample_sets(self):
        tables = wf.wf_sim(
            4,
            5,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=10,
        )
        tables.sort()
        ts = tables.tree_sequence()
        for S in test_stats.set_partitions(list(ts.samples())):
            self.verify_topologies(ts, sample_sets=S)

    def test_forward_time_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            20,
            15,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=20,
        )
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify_topologies(ts, sample_sets=[samples[:10], samples[10:]])

    def test_forward_time_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            30,
            10,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=5,
        )
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify_topologies(ts, sample_sets=[samples[:10], samples[10:]])


class TestTreeNode:
    """
    Tests for the TreeNode class used to build simple trees in memory.
    """

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_random_binary_tree(self, n):
        # Note this doesn't check any statistical properties of the returned
        # trees, just that a single instance returned in a valid binary tree.
        rng = random.Random(32)
        labels = range(n)
        root = comb.TreeNode.random_binary_tree(labels, rng)

        stack = [root]
        num_nodes = 0
        recovered_labels = []
        while len(stack) > 0:
            node = stack.pop()
            num_nodes += 1
            if node.label is not None:
                assert len(node.children) == 0
                recovered_labels.append(node.label)
            for child in node.children:
                assert child.parent == node
                stack.append(child)
        assert len(recovered_labels) == n
        assert sorted(recovered_labels) == list(labels)
