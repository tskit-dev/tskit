#
# MIT License
#
# Copyright (c) 2020-2023 Tskit Developers
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
import json
import math
import random

import msprime
import numpy as np
import pytest

import tests.test_wright_fisher as wf
import tskit
import tskit.combinatorics as comb
from tests import test_stats
from tskit.combinatorics import Rank
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
    @pytest.mark.parametrize("n", range(11))
    def test_num_shapes(self, n):
        all_trees = RankTree.all_unlabelled_trees(n)
        assert len(list(all_trees)) == comb.num_shapes(n)

    @pytest.mark.parametrize("n", range(2, 8))
    def test_num_labellings(self, n):
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
        expected = ["(0,1,2);", "(0,(1,2));", "(1,(0,2));", "(2,(0,1));"]
        actual = [t.newick() for t in RankTree.all_labelled_trees(3)]
        assert expected == actual

    def test_all_labelled_trees_4(self):
        expected = [
            # 1 + 1 + 1 + 1 (partition of num leaves)
            "(0,1,2,3);",
            # 1 + 1 + 2
            "(0,1,(2,3));",
            "(0,2,(1,3));",
            "(0,3,(1,2));",
            "(1,2,(0,3));",
            "(1,3,(0,2));",
            "(2,3,(0,1));",
            # 1 + 3
            # partition of 3 = 1 + 1 + 1
            "(0,(1,2,3));",
            "(1,(0,2,3));",
            "(2,(0,1,3));",
            "(3,(0,1,2));",
            # partition of 3 = 1 + 2
            "(0,(1,(2,3)));",
            "(0,(2,(1,3)));",
            "(0,(3,(1,2)));",
            "(1,(0,(2,3)));",
            "(1,(2,(0,3)));",
            "(1,(3,(0,2)));",
            "(2,(0,(1,3)));",
            "(2,(1,(0,3)));",
            "(2,(3,(0,1)));",
            "(3,(0,(1,2)));",
            "(3,(1,(0,2)));",
            "(3,(2,(0,1)));",
            # 2 + 2
            "((0,1),(2,3));",
            "((0,2),(1,3));",
            "((0,3),(1,2));",
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

    @pytest.mark.parametrize("n", range(6))
    def test_unrank_labelled(self, n):
        for shape_rank, t in enumerate(RankTree.all_unlabelled_trees(n)):
            for label_rank, labelled_tree in enumerate(RankTree.all_labellings(t)):
                unranked = RankTree.unrank(n, (shape_rank, label_rank))
                assert labelled_tree == unranked

    @pytest.mark.parametrize("n", range(10))
    def test_unrank_unlabelled(self, n):
        for shape_rank in range(comb.num_shapes(n)):
            rank = Rank(shape_rank, 0)
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

    @pytest.mark.parametrize("n", range(6))
    def test_shape_rank(self, n):
        for rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
            assert tree.shape_rank() == rank

    @pytest.mark.parametrize("n", range(6))
    def test_shape_unrank(self, n):
        for rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
            t = RankTree.shape_unrank(n, rank)
            assert tree.shape_equal(t)

    @pytest.mark.parametrize("n", range(2, 9))
    def test_shape_unrank_tsk_tree(self, n):
        for shape_rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
            tsk_tree = tskit.Tree.unrank(n, (shape_rank, 0))
            assert shape_rank == tree.shape_rank()
            shape_rank, _ = tsk_tree.rank()
            assert shape_rank == tree.shape_rank()

    @pytest.mark.parametrize("n", range(7))
    def test_label_rank(self, n):
        for tree in RankTree.all_unlabelled_trees(n):
            for rank, labelled_tree in enumerate(RankTree.all_labellings(tree)):
                assert labelled_tree.label_rank() == rank

    @pytest.mark.parametrize("n", range(7))
    def test_label_unrank(self, n):
        for shape_rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
            for label_rank, labelled_tree in enumerate(RankTree.all_labellings(tree)):
                rank = (shape_rank, label_rank)
                unranked = tree.label_unrank(label_rank)
                assert labelled_tree.rank() == rank
                assert unranked.rank() == rank

    def test_rank_names(self):
        shape = 1
        label = 0
        n = 3
        tree = tskit.Tree.unrank(n, (shape, label))
        rank = tree.rank()
        assert rank.shape == shape
        assert rank.label == label

    @pytest.mark.parametrize("n", range(6))
    def test_unrank_rank_round_trip(self, n):
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

    @pytest.mark.parametrize("n", range(7))
    def test_unranking_is_canonical(self, n):
        for shape_rank in range(comb.num_shapes(n)):
            for label_rank in range(comb.num_labellings(n, shape_rank)):
                t = RankTree.shape_unrank(n, shape_rank)
                assert t.is_canonical()
                t = t.label_unrank(label_rank)
                assert t.is_canonical()
                t = tskit.Tree.unrank(n, (shape_rank, label_rank))
                assert RankTree.from_tsk_tree(t).is_canonical()

    @pytest.mark.parametrize("n", range(5))
    def test_to_from_tsk_tree(self, n):
        for tree in RankTree.all_labelled_trees(n):
            assert tree.is_canonical()
            tsk_tree = tree.to_tsk_tree()
            reconstructed = RankTree.from_tsk_tree(tsk_tree)
            assert tree.is_canonical()
            assert tree == reconstructed

    @pytest.mark.parametrize("n", range(6))
    def test_to_tsk_tree_internal_nodes(self, n):
        branch_length = 1234
        for tree in RankTree.all_labelled_trees(n):
            tsk_tree = tree.to_tsk_tree(branch_length=branch_length)
            internal_nodes = [
                u for u in tsk_tree.nodes(order="postorder") if tsk_tree.is_internal(u)
            ]
            assert np.all(internal_nodes == n + np.arange(len(internal_nodes)))
            for u in tsk_tree.nodes():
                if tsk_tree.is_internal(u):
                    max_child_time = max(tsk_tree.time(v) for v in tsk_tree.children(u))
                    assert tsk_tree.time(u) == max_child_time + branch_length
                else:
                    assert tsk_tree.time(u) == 0

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
        assert tsk_tree.tree_sequence.num_nodes == n + 1
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

        sample_sets = [[0], [tables.nodes.num_rows]]
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

    def verify_tree(self, root, labels):
        # Note this doesn't check any statistical properties of the returned
        # trees, just that a single instance returned in a valid binary tree.
        # Structural properties are best verified using the tskit API, and so
        # we test these properties elsewhere.
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
        assert sorted(recovered_labels) == list(labels)

    @pytest.mark.parametrize("n", range(1, 16))
    def test_random_binary_tree(self, n):
        rng = random.Random(32)
        labels = range(n)
        root = comb.TreeNode.random_binary_tree(labels, rng)
        self.verify_tree(root, range(n))

    @pytest.mark.parametrize("n", range(1, 16))
    def test_balanced_binary(self, n):
        root = comb.TreeNode.balanced_tree(range(n), 2)
        self.verify_tree(root, range(n))

    @pytest.mark.parametrize("arity", range(2, 8))
    def test_balanced_arity(self, arity):
        labels = range(30)
        root = comb.TreeNode.balanced_tree(labels, arity)
        self.verify_tree(root, labels)


def num_leaf_labelled_binary_trees(n):
    """
    Returns the number of leaf labelled binary trees with n leaves.

    TODO: this would probably be helpful to have in the combinatorics
    module.

    https://oeis.org/A005373/
    """
    return int(math.factorial(2 * n - 3) / (2 ** (n - 2) * math.factorial(n - 2)))


class TestPolytomySplitting:
    """
    Test the ability to randomly split polytomies
    """

    # A complex ts with polytomies
    #
    # 1.00┊    6      ┊      6    ┊       6   ┊           ┊      6    ┊
    #     ┊ ┏━┳┻┳━┓   ┊   ┏━┳┻┳━┓ ┊    ┏━━╋━┓ ┊           ┊   ┏━┳┻┳━┓ ┊
    # 0.50┊ 5 ┃ ┃ ┃   ┊   5 ┃ ┃ ┃ ┊    5  ┃ ┃ ┊      5    ┊   ┃ ┃ ┃ ┃ ┊
    #     ┊ ┃ ┃ ┃ ┃ . ┊   ┃ ┃ ┃ ┃ ┊ . ┏┻┓ ┃ ┃ ┊ . ┏━┳┻┳━┓ ┊ . ┃ ┃ ┃ ┃ ┊
    # 0.00┊ 0 2 3 4 1 ┊ 0 1 2 3 4 ┊ 0 1 2 3 4 ┊ 0 1 2 3 4 ┊ 0 1 2 3 4 ┊
    #   0.00        0.20        0.40        0.60        0.80        1.00
    nodes_polytomy_44344 = """\
    id      is_sample   population      time
    0       1           0               0.0
    1       1           0               0.0
    2       1           0               0.0
    3       1           0               0.0
    4       1           0               0.0
    5       0           0               0.5
    6       0           0               1.0
    """
    edges_polytomy_44344 = """\
    id      left     right    parent  child
    0       0.0      0.2      5       0
    1       0.0      0.8      5       1
    2       0.0      0.4      6       2
    3       0.4      0.8      5       2
    4       0.0      0.6      6       3,4
    5       0.0      0.6      6       5
    6       0.6      0.8      5       3,4
    7       0.8      1.0      6       1,2,3,4
    """

    def ts_polytomy_44344(self):
        return tskit.load_text(
            nodes=io.StringIO(self.nodes_polytomy_44344),
            edges=io.StringIO(self.edges_polytomy_44344),
            strict=False,
        )

    def verify_trees(self, source_tree, split_tree, epsilon=None):
        N = 0
        for u in split_tree.nodes():
            assert split_tree.num_children(u) < 3
            N += 1
            if u >= source_tree.tree_sequence.num_nodes:
                # This is a new node
                branch_length = split_tree.branch_length(u)
                if epsilon is not None:
                    assert epsilon == pytest.approx(branch_length)
                else:
                    assert branch_length > 0
                    assert 0 == pytest.approx(branch_length)

        assert N == len(list(split_tree.leaves())) * 2 - 1
        for u in source_tree.nodes():
            if source_tree.num_children(u) <= 2:
                assert source_tree.children(u) == split_tree.children(u)
            else:
                assert len(split_tree.children(u)) == 2

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_resolve_star(self, n):
        tree = tskit.Tree.generate_star(n)
        self.verify_trees(tree, tree.split_polytomies(random_seed=12))

    def test_large_epsilon(self):
        tree = tskit.Tree.generate_star(10, branch_length=100)
        eps = 10
        split = tree.split_polytomies(random_seed=12234, epsilon=eps)
        self.verify_trees(tree, split, epsilon=eps)

    def test_small_epsilon(self):
        tree = tskit.Tree.generate_star(10, branch_length=1e-20)
        eps = 1e-22
        split = tree.split_polytomies(random_seed=12234, epsilon=eps)
        self.verify_trees(tree, split, epsilon=eps)

    def test_nextafter_near_zero(self):
        tree = tskit.Tree.generate_star(3, branch_length=np.finfo(float).tiny)
        split = tree.split_polytomies(random_seed=234)
        self.verify_trees(tree, split)

    def test_nextafter_large_tree(self):
        tree = tskit.Tree.generate_star(100)
        split = tree.split_polytomies(random_seed=32)
        self.verify_trees(tree, split)
        for u in tree.nodes():
            if tree.parent(u) != tskit.NULL and not tree.is_leaf(u):
                parent_time = tree.time(tree.parent(u))
                child_time = tree.time(u)
                assert child_time == np.nextafter(parent_time, 0)
            if tree.is_leaf(u):
                assert tree.branch_length(u) == pytest.approx(1)

    def test_epsilon_near_one(self):
        tree = tskit.Tree.generate_star(3, branch_length=1)
        split = tree.split_polytomies(random_seed=234, epsilon=np.finfo(float).eps)
        self.verify_trees(tree, split)

    def verify_tree_sequence_splits(self, ts):
        n_poly = 0
        for e in ts.edgesets():
            if len(e.children) > 2:
                n_poly += 1
        assert n_poly > 3
        assert ts.num_trees > 3
        for tree in ts.trees():
            binary_tree = tree.split_polytomies(random_seed=11)
            assert binary_tree.interval == tree.interval
            for u in binary_tree.nodes():
                assert binary_tree.num_children(u) < 3
            for u in tree.nodes():
                assert binary_tree.time(u) == tree.time(u)
            resolved_ts = binary_tree.tree_sequence
            assert resolved_ts.sequence_length == ts.sequence_length
            assert resolved_ts.num_trees <= 3
            if tree.interval.left == 0:
                assert resolved_ts.num_trees == 2
                null_tree = resolved_ts.last()
                assert null_tree.num_roots == ts.num_samples
            elif tree.interval.right == ts.sequence_length:
                assert resolved_ts.num_trees == 2
                null_tree = resolved_ts.first()
                assert null_tree.num_roots == ts.num_samples
            else:
                null_tree = resolved_ts.first()
                assert null_tree.num_roots == ts.num_samples
                null_tree.next()
                assert null_tree.num_roots == tree.num_roots
                null_tree.next()
                assert null_tree.num_roots == ts.num_samples

    def test_complex_examples(self):
        self.verify_tree_sequence_splits(self.ts_polytomy_44344())

    def test_nonbinary_simulation(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=1.0, population=0, proportion=0.95)
        ]
        ts = msprime.simulate(
            20,
            recombination_rate=10,
            mutation_rate=5,
            demographic_events=demographic_events,
            random_seed=7,
        )
        self.verify_tree_sequence_splits(ts)

    def test_seeds(self):
        base = tskit.Tree.generate_star(5)
        t1 = base.split_polytomies(random_seed=1234)
        t2 = base.split_polytomies(random_seed=1234)
        assert t1.tree_sequence.tables.equals(
            t2.tree_sequence.tables, ignore_timestamps=True
        )
        t2 = base.split_polytomies(random_seed=1)
        assert not t1.tree_sequence.tables.equals(
            t2.tree_sequence.tables, ignore_provenance=True
        )

    def test_internal_polytomy(self):
        #       9
        # ┏━┳━━━┻┳━━━━┓
        # ┃ ┃    8    ┃
        # ┃ ┃ ┏━━╋━━┓ ┃
        # ┃ ┃ ┃  7  ┃ ┃
        # ┃ ┃ ┃ ┏┻┓ ┃ ┃
        # 0 1 2 3 5 4 6
        t1 = tskit.Tree.unrank(7, (6, 25))
        t2 = t1.split_polytomies(random_seed=1234)
        assert t2.parent(3) == 7
        assert t2.parent(5) == 7
        assert t2.root == 9
        for u in t2.nodes():
            assert t2.num_children(u) in [0, 2]

    def test_binary_tree(self):
        t1 = msprime.simulate(10, random_seed=1234).first()
        t2 = t1.split_polytomies(random_seed=1234)
        tables = t1.tree_sequence.dump_tables()
        tables.assert_equals(t2.tree_sequence.tables, ignore_provenance=True)

    def test_bad_method(self):
        tree = tskit.Tree.generate_star(3)
        with pytest.raises(ValueError, match="Method"):
            tree.split_polytomies(method="something_else")

    @pytest.mark.parametrize("epsilon", [10, 1.1, 1.0])
    def test_epsilon_too_large(self, epsilon):
        tree = tskit.Tree.generate_star(3)
        msg = (
            "Cannot resolve the degree 3 polytomy rooted at node 3 "
            "with minimum time difference of 1.0 to the resolved leaves. "
            f"The fixed epsilon value of {epsilon} is too large, resulting in the "
            "parent time being less than the child time."
        )
        with pytest.raises(
            tskit.LibraryError,
            match=msg,
        ):
            tree.split_polytomies(epsilon=epsilon, random_seed=12)

    def test_epsilon_too_small(self):
        tree = tskit.Tree.generate_star(3)
        msg = (
            "Cannot resolve the degree 3 polytomy rooted at node 3 "
            "with minimum time difference of 1.0 to the resolved leaves. "
            "The fixed epsilon value of 0 is too small, resulting in the "
            "parent and child times being equal within the limits of "
            "numerical precision."
        )
        with pytest.raises(
            tskit.LibraryError,
            match=msg,
        ):
            tree.split_polytomies(epsilon=0, random_seed=12)

    def test_unsplittable_branch(self):
        branch_length = np.nextafter(0, 1)
        tree = tskit.Tree.generate_star(3, branch_length=branch_length)
        msg = (
            "Cannot resolve the degree 3 polytomy rooted at node 3 with "
            "minimum time difference of 5e-324 to the resolved leaves. "
            "The time difference between nodes is so small that more nodes "
            "cannot be inserted between within the limits of floating point "
            "precision."
        )
        with pytest.raises(
            tskit.LibraryError,
            match=msg,
        ):
            tree.split_polytomies(random_seed=12)

    def test_epsilon_for_mutations(self):
        tables = tskit.Tree.generate_star(3).tree_sequence.dump_tables()
        root_time = tables.nodes.time[-1]
        assert root_time == 1
        site = tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.mutations.add_row(site=site, time=0.9, node=0, derived_state="1")
        tables.mutations.add_row(site=site, time=0.9, node=1, derived_state="1")
        tree = tables.tree_sequence().first()
        with pytest.raises(
            tskit.LibraryError,
            match="not small enough to create new nodes below a polytomy",
        ):
            tree.split_polytomies(epsilon=0.5, random_seed=123)

    def test_mutation_within_eps_parent(self):
        tables = tskit.Tree.generate_star(3).tree_sequence.dump_tables()
        site = tables.sites.add_row(position=0.5, ancestral_state="0")
        branch_length = np.nextafter(1, 0)
        tables.mutations.add_row(
            site=site, time=branch_length, node=0, derived_state="1"
        )
        tables.mutations.add_row(
            site=site, time=branch_length, node=1, derived_state="1"
        )
        tree = tables.tree_sequence().first()
        with pytest.raises(
            tskit.LibraryError,
            match="Cannot split polytomy: mutation with numerical precision",
        ):
            tree.split_polytomies(random_seed=123)

    def test_provenance(self):
        tree = tskit.Tree.generate_star(4)
        ts_split = tree.split_polytomies(random_seed=14).tree_sequence
        record = json.loads(ts_split.provenance(ts_split.num_provenances - 1).record)
        assert record["parameters"]["command"] == "split_polytomies"
        ts_split = tree.split_polytomies(
            random_seed=12, record_provenance=False
        ).tree_sequence
        record = json.loads(ts_split.provenance(ts_split.num_provenances - 1).record)
        assert record["parameters"]["command"] != "split_polytomies"

    def test_kwargs(self):
        tree = tskit.Tree.generate_star(4)
        split_tree = tree.split_polytomies(random_seed=14, tracked_samples=[0, 1])
        assert split_tree.num_tracked_samples() == 2

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_all_topologies(self, n):
        N = num_leaf_labelled_binary_trees(n)
        ranks = collections.Counter()
        for seed in range(20 * N):
            star = tskit.Tree.generate_star(n)
            random_tree = star.split_polytomies(random_seed=seed)
            ranks[random_tree.rank()] += 1
        # There are N possible binary trees here, we should have seen them
        # all with high probability after 20 N attempts.
        assert len(ranks) == N


class TreeGeneratorTestBase:
    """
    Abstract superclass of tree generator test methods.

    Concrete subclasses should defined "method_name" class variable.
    """

    def method(self, n, **kwargs):
        return getattr(tskit.Tree, self.method_name)(n, **kwargs)

    @pytest.mark.parametrize("n", range(2, 10))
    def test_leaves(self, n):
        tree = self.method(n)
        assert list(tree.leaves()) == list(range(n))

    def test_bad_n(self):
        for n in [-1, 0, np.array([1, 2])]:
            with pytest.raises(ValueError):
                self.method(n)
        for n in [None, "", []]:
            with pytest.raises(TypeError):
                self.method(n)

    def test_bad_span(self):
        with pytest.raises(tskit.LibraryError):
            self.method(2, span=0)

    def test_bad_branch_length(self):
        with pytest.raises(tskit.LibraryError):
            self.method(2, branch_length=0)

    @pytest.mark.parametrize("span", [0.1, 1, 100])
    def test_span(self, span):
        tree = self.method(5, span=span)
        assert tree.tree_sequence.sequence_length == span

    @pytest.mark.parametrize("branch_length", [0.25, 1, 100])
    def test_branch_length(self, branch_length):
        tree = self.method(5, branch_length=branch_length)
        for u in tree.nodes():
            if u != tree.root:
                assert tree.branch_length(u) >= branch_length

    def test_provenance(self):
        ts = self.method(2).tree_sequence
        assert ts.num_provenances == 1
        record = json.loads(ts.provenance(0).record)
        assert record["parameters"]["command"] == self.method_name
        ts = self.method(2, record_provenance=False).tree_sequence
        assert ts.num_provenances == 0

    @pytest.mark.parametrize("n", range(2, 10))
    def test_rank_unrank_round_trip(self, n):
        tree1 = self.method(n)
        rank = tree1.rank()
        tree2 = tskit.Tree.unrank(n, rank)
        tables1 = tree1.tree_sequence.tables
        tables2 = tree2.tree_sequence.tables
        tables1.assert_equals(tables2, ignore_provenance=True)

    def test_kwargs(self):
        tree = self.method(3, tracked_samples=[0, 1])
        assert tree.num_tracked_samples() == 2


class TestGenerateStar(TreeGeneratorTestBase):
    method_name = "generate_star"

    @pytest.mark.parametrize("n", range(2, 10))
    def test_unrank_equal(self, n):
        for extra_params in [{}, {"span": 2.5}, {"branch_length": 3}]:
            ts = tskit.Tree.generate_star(n, **extra_params).tree_sequence
            equiv_ts = tskit.Tree.unrank(n, (0, 0), **extra_params).tree_sequence
            assert ts.tables.equals(equiv_ts.tables, ignore_provenance=True)

    def test_branch_length_semantics(self):
        branch_length = 10
        ts = tskit.Tree.generate_star(7, branch_length=branch_length).tree_sequence
        time = ts.tables.nodes.time
        edges = ts.tables.edges
        length = time[edges.parent] - time[edges.child]
        assert np.all(length == branch_length)


class TestGenerateBalanced(TreeGeneratorTestBase):
    method_name = "generate_balanced"

    @pytest.mark.parametrize("arity", range(2, 10))
    def test_arity_leaves(self, arity):
        n = 20
        tree = tskit.Tree.generate_balanced(n, arity=arity)
        assert list(tree.leaves()) == list(range(n))

    @pytest.mark.parametrize("n", range(1, 13))
    def test_binary_unrank_equal(self, n):
        for extra_params in [{}, {"span": 2.5}, {"branch_length": 3}]:
            ts = tskit.Tree.generate_balanced(n, **extra_params).tree_sequence
            N = tskit.combinatorics.num_shapes(n)
            equiv_ts = tskit.Tree.unrank(n, (N - 1, 0), **extra_params).tree_sequence
            assert ts.tables.equals(equiv_ts.tables, ignore_provenance=True)

    @pytest.mark.parametrize(
        ("n", "arity"), [(2, 2), (8, 2), (27, 3), (29, 3), (11, 5), (5, 10)]
    )
    def test_rank_unrank_round_trip_arity(self, n, arity):
        tree1 = tskit.Tree.generate_balanced(n, arity=arity)
        rank = tree1.rank()
        tree2 = tskit.Tree.unrank(n, rank)
        tables1 = tree1.tree_sequence.tables
        tables2 = tree2.tree_sequence.tables
        tables1.assert_equals(tables2, ignore_provenance=True)

    def test_bad_arity(self):
        for arity in [-1, 0, 1]:
            with pytest.raises(ValueError):
                tskit.Tree.generate_balanced(10, arity=arity)

    def test_branch_length_semantics(self):
        branch_length = 10
        tree = tskit.Tree.generate_balanced(8, branch_length=branch_length)
        for u in tree.nodes():
            for v in tree.children(u):
                # Special case cause n is a power of 2
                assert tree.time(u) == tree.time(v) + branch_length


class TestGenerateRandomBinary(TreeGeneratorTestBase):
    method_name = "generate_random_binary"

    def method(self, n, **kwargs):
        return tskit.Tree.generate_random_binary(n, random_seed=53, **kwargs)

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_all_topologies(self, n):
        N = num_leaf_labelled_binary_trees(n)
        ranks = collections.Counter()
        for seed in range(20 * N):
            random_tree = tskit.Tree.generate_random_binary(n, random_seed=seed)
            ranks[random_tree.rank()] += 1
        # There are N possible binary trees here, we should have seen them
        # all with high probability after 20 N attempts.
        assert len(ranks) == N

    @pytest.mark.parametrize("n", range(2, 10))
    def test_leaves(self, n):
        tree = tskit.Tree.generate_random_binary(n, random_seed=1234)
        # The leaves should be a permutation of range(n)
        assert list(sorted(tree.leaves())) == list(range(n))

    @pytest.mark.parametrize("seed", range(1, 20))
    def test_rank_unrank_round_trip_seeds(self, seed):
        n = 10
        tree1 = tskit.Tree.generate_random_binary(n, random_seed=seed)
        rank = tree1.rank()
        tree2 = tskit.Tree.unrank(n, rank)
        tables1 = tree1.tree_sequence.tables
        tables2 = tree2.tree_sequence.tables
        tables1.assert_equals(tables2, ignore_provenance=True)


class TestGenerateComb(TreeGeneratorTestBase):
    method_name = "generate_comb"

    # Hard-code in some pre-computed ranks for the comb(n) tree.
    @pytest.mark.parametrize(["n", "rank"], [(2, 0), (3, 1), (4, 3), (5, 8), (6, 20)])
    def test_unrank_equal(self, n, rank):
        for extra_params in [{}, {"span": 2.5}, {"branch_length": 3}]:
            ts = tskit.Tree.generate_comb(n, **extra_params).tree_sequence
            equiv_ts = tskit.Tree.unrank(n, (rank, 0), **extra_params).tree_sequence
            assert ts.tables.equals(equiv_ts.tables, ignore_provenance=True)

    def test_branch_length_semantics(self):
        branch_length = 10
        tree = tskit.Tree.generate_comb(2, branch_length=branch_length)
        assert tree.time(tree.root) == branch_length


class TestEqualChunks:
    @pytest.mark.parametrize(("n", "k"), [(2, 1), (4, 2), (9, 3), (100, 10)])
    def test_evenly_divisible(self, n, k):
        lst = range(n)
        chunks = list(comb.equal_chunks(lst, k))
        assert len(chunks) == k
        for chunk in chunks:
            assert len(chunk) == n // k
        assert list(itertools.chain(*chunks)) == list(range(n))

    @pytest.mark.parametrize("n", range(1, 5))
    def test_one_chunk(self, n):
        lst = list(range(n))
        chunks = list(comb.equal_chunks(lst, 1))
        assert chunks == [lst]

    @pytest.mark.parametrize(("n", "k"), [(1, 2), (5, 6), (10, 20), (5, 100)])
    def test_empty_chunks(self, n, k):
        lst = range(n)
        chunks = list(comb.equal_chunks(lst, k))
        assert len(chunks) == n
        for chunk in chunks:
            assert len(chunk) == 1
        assert list(itertools.chain(*chunks)) == list(range(n))

    @pytest.mark.parametrize(("n", "k"), [(3, 2), (10, 3), (11, 5), (13, 10)])
    def test_trailing_chunk(self, n, k):
        lst = range(n)
        chunks = list(comb.equal_chunks(lst, k))
        assert len(chunks) == k
        assert list(itertools.chain(*chunks)) == list(range(n))

    def test_empty_list(self):
        assert len(list(comb.equal_chunks([], 1))) == 0
        assert len(list(comb.equal_chunks([], 2))) == 0

    def test_bad_num_chunks(self):
        for bad_num_chunks in [0, -1, 0.1]:
            with pytest.raises(ValueError):
                list(comb.equal_chunks([1], bad_num_chunks))
