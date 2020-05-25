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
import itertools
import unittest

import tskit
import tskit.combinatorics as comb
from tskit.combinatorics import RankTree


class TestCombination(unittest.TestCase):
    def test_combination_with_replacement_rank_unrank(self):
        for n in range(9):
            for k in range(n):
                nums = list(range(n))
                combs = itertools.combinations_with_replacement(nums, k)
                for exp_rank, c in enumerate(combs):
                    c = list(c)
                    actual_rank = comb.Combination.with_replacement_rank(c, n)
                    self.assertEqual(actual_rank, exp_rank)
                    unranked = comb.Combination.with_replacement_unrank(exp_rank, n, k)
                    self.assertEqual(unranked, c)

    def test_combination_rank_unrank(self):
        for n in range(11):
            for k in range(n):
                nums = list(range(n))
                for rank, c in enumerate(itertools.combinations(nums, k)):
                    c = list(c)
                    self.assertEqual(comb.Combination.rank(c, nums), rank)
                    self.assertEqual(comb.Combination.unrank(rank, nums, k), c)

    def test_combination_unrank_errors(self):
        self.verify_unrank_errors(1, 1, 1)
        self.verify_unrank_errors(2, 0, 1)

    def verify_unrank_errors(self, rank, n, k):
        with self.assertRaises(ValueError):
            comb.Combination.unrank(rank, list(range(n)), k)


class TestPartition(unittest.TestCase):
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
        self.assertEqual(list(comb.rule_asc(n)), partitions)

    def test_partitions(self):
        self.assertEqual(list(comb.partitions(0)), [])
        for n in range(1, 7):
            self.assertEqual(list(comb.partitions(n)), list(comb.rule_asc(n))[:-1])

    def test_group_partition(self):
        self.assertEqual(comb.group_partition([1]), [[1]])
        self.assertEqual(comb.group_partition([1, 2]), [[1], [2]])
        self.assertEqual(comb.group_partition([1, 1, 1]), [[1, 1, 1]])
        self.assertEqual(comb.group_partition([1, 1, 2, 3, 3]), [[1, 1], [2], [3, 3]])


class TestRankTree(unittest.TestCase):
    def test_num_shapes(self):
        for i in range(11):
            all_trees = RankTree.all_unlabelled_trees(i)
            self.assertEqual(len(list(all_trees)), comb.num_shapes(i))

    def test_num_labellings(self):
        for n in range(2, 8):
            for tree in RankTree.all_unlabelled_trees(n):
                tree = tree.label_unrank(0)
                tree2 = tree.to_tsk_tree()
                n_labellings = sum(1 for _ in RankTree.all_labellings(tree))
                self.assertEqual(
                    n_labellings, RankTree.from_tsk_tree(tree2).num_labellings()
                )

    def test_num_labelled_trees(self):
        # Number of leaf-labelled trees with n leaves on OEIS
        n_trees = [0, 1, 1, 4, 26, 236, 2752, 39208]
        for i, expected in zip(range(len(n_trees)), n_trees):
            actual = sum(1 for _ in RankTree.all_labelled_trees(i))
            self.assertEqual(actual, expected)

    def test_all_labelled_trees_3(self):
        expected = ["(0,1,2)", "(0,(1,2))", "(1,(0,2))", "(2,(0,1))"]
        actual = [t.newick() for t in RankTree.all_labelled_trees(3)]
        self.assertEqual(expected, actual)

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
        self.assertEqual(expected, actual)

    def test_generate_trees_roundtrip(self):
        n = 5
        all_rank_trees = RankTree.all_labelled_trees(n)
        all_tsk_trees = tskit.all_trees(n)
        for rank_tree, tsk_tree in zip(all_rank_trees, all_tsk_trees):
            self.assertEqual(rank_tree, RankTree.from_tsk_tree(tsk_tree))

    def test_all_shapes_roundtrip(self):
        n = 5
        all_rank_tree_shapes = RankTree.all_unlabelled_trees(n)
        all_tsk_tree_shapes = tskit.all_tree_shapes(n)
        for rank_tree, tsk_tree in zip(all_rank_tree_shapes, all_tsk_tree_shapes):
            self.assertTrue(rank_tree.shape_equal(RankTree.from_tsk_tree(tsk_tree)))

    def test_all_labellings_roundtrip(self):
        n = 5
        rank_tree = RankTree.unrank((comb.num_shapes(n) - 1, 0), n)
        tsk_tree = rank_tree.to_tsk_tree()
        rank_tree_labellings = RankTree.all_labellings(rank_tree)
        tsk_tree_labellings = tskit.all_tree_labellings(tsk_tree)
        for rank_t, tsk_t in zip(rank_tree_labellings, tsk_tree_labellings):
            self.assertEqual(rank_t, RankTree.from_tsk_tree(tsk_t))

    def test_unrank(self):
        for n in range(6):
            for shape_rank, t in enumerate(RankTree.all_unlabelled_trees(n)):
                for label_rank, labelled_tree in enumerate(RankTree.all_labellings(t)):
                    unranked = RankTree.unrank((shape_rank, label_rank), n)
                    self.assertTrue(labelled_tree == unranked)

        # The number of labelled trees gets very big quickly
        for n in range(6, 10):
            for shape_rank in range(comb.num_shapes(n)):
                rank = (shape_rank, 0)
                unranked = RankTree.unrank(rank, n)
                self.assertTrue(rank, unranked.rank())

                rank = (shape_rank, comb.num_labellings(shape_rank, n) - 1)
                unranked = RankTree.unrank(rank, n)
                self.assertTrue(rank, unranked.rank())

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
        invalid_labelling = (0, comb.num_labellings(0, 10))
        self.verify_unrank_errors(invalid_labelling, 10)

    def verify_unrank_errors(self, rank, n):
        with self.assertRaises(ValueError):
            RankTree.unrank(rank, n)
        with self.assertRaises(ValueError):
            tskit.Tree.unrank(rank, n)

    def test_shape_rank(self):
        for n in range(10):
            for rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
                self.assertEqual(tree.shape_rank(), rank)

    def test_shape_unrank(self):
        for n in range(6):
            for rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
                t = RankTree.shape_unrank(rank, n)
                self.assertTrue(tree.shape_equal(t))

        for n in range(2, 9):
            for shape_rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
                tsk_tree = tskit.Tree.unrank((shape_rank, 0), n)
                self.assertEqual(shape_rank, tree.shape_rank())
                shape_rank, _ = tsk_tree.rank()
                self.assertEqual(shape_rank, tree.shape_rank())

    def test_label_rank(self):
        for n in range(7):
            for tree in RankTree.all_unlabelled_trees(n):
                for rank, labelled_tree in enumerate(RankTree.all_labellings(tree)):
                    self.assertEqual(labelled_tree.label_rank(), rank)

    def test_label_unrank(self):
        for n in range(7):
            for shape_rank, tree in enumerate(RankTree.all_unlabelled_trees(n)):
                for label_rank, labelled_tree in enumerate(
                    RankTree.all_labellings(tree)
                ):
                    rank = (shape_rank, label_rank)
                    unranked = tree.label_unrank(label_rank)
                    self.assertEqual(labelled_tree.rank(), rank)
                    self.assertEqual(unranked.rank(), rank)

    def test_unrank_rank_round_trip(self):
        for n in range(6):  # Can do more but gets slow pretty quickly after 6
            for shape_rank in range(comb.num_shapes(n)):
                tree = RankTree.shape_unrank(shape_rank, n)
                tree = tree.label_unrank(0)
                self.assertEqual(tree.shape_rank(), shape_rank)
                for label_rank in range(tree.num_labellings()):
                    tree = tree.label_unrank(label_rank)
                    self.assertEqual(tree.label_rank(), label_rank)
                    tsk_tree = tree.label_unrank(label_rank).to_tsk_tree()
                    _, tsk_label_rank = tsk_tree.rank()
                    self.assertEqual(tsk_label_rank, label_rank)

    def test_is_canonical(self):
        for n in range(7):
            for tree in RankTree.all_labelled_trees(n):
                self.assertTrue(tree.is_canonical())

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
        self.assertFalse(shape_not_canonical.is_canonical())

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
        self.assertFalse(labels_not_canonical.is_canonical())

    def test_unranking_is_canonical(self):
        for n in range(7):
            for shape_rank in range(comb.num_shapes(n)):
                for label_rank in range(comb.num_labellings(shape_rank, n)):
                    t = RankTree.shape_unrank(shape_rank, n)
                    self.assertTrue(t.is_canonical())
                    t = t.label_unrank(label_rank)
                    self.assertTrue(t.is_canonical())
                    t = tskit.Tree.unrank((shape_rank, label_rank), n)
                    self.assertTrue(RankTree.from_tsk_tree(t).is_canonical())

    def test_to_from_tsk_tree(self):
        for n in range(5):
            for tree in RankTree.all_labelled_trees(n):
                self.assertTrue(tree.is_canonical())
                tsk_tree = tree.to_tsk_tree()
                reconstructed = RankTree.from_tsk_tree(tsk_tree)
                self.assertTrue(tree.is_canonical())
                self.assertEqual(tree, reconstructed)

    def test_rank_errors_multiple_roots(self):
        tables = tskit.TableCollection(sequence_length=1.0)

        # Nodes
        sv = [True, True]
        tv = [0.0, 0.0]

        for is_sample, t in zip(sv, tv):
            flags = tskit.NODE_IS_SAMPLE if is_sample else 0
            tables.nodes.add_row(flags=flags, time=t)

        ts = tables.tree_sequence()
        with self.assertRaises(ValueError):
            ts.first().rank()

    def test_big_trees(self):
        n = 14
        shape = 22
        labelling = 0
        tree = RankTree.unrank((shape, labelling), n)
        tsk_tree = tskit.Tree.unrank((shape, labelling), n)
        self.assertEqual(tree.rank(), tsk_tree.rank())

        n = 10
        shape = 95
        labelling = comb.num_labellings(shape, n) // 2
        tree = RankTree.unrank((shape, labelling), n)
        tsk_tree = tskit.Tree.unrank((shape, labelling), n)
        self.assertEqual(tree.rank(), tsk_tree.rank())

    def test_symmetrical_trees(self):
        for n in range(2, 18, 2):
            last_rank = comb.num_shapes(n) - 1
            t = RankTree.shape_unrank(last_rank, n)
            self.assertTrue(t.is_symmetrical())

    def test_equal(self):
        unlabelled_leaf = RankTree(children=[])
        self.assertEqual(unlabelled_leaf, unlabelled_leaf)
        self.assertTrue(unlabelled_leaf.shape_equal(unlabelled_leaf))

        leaf_zero = RankTree(children=[], label=0)
        leaf_one = RankTree(children=[], label=1)
        leaf_two = RankTree(children=[], label=2)
        self.assertEqual(leaf_zero, leaf_zero)
        self.assertNotEqual(leaf_zero, leaf_one)
        self.assertTrue(leaf_zero.shape_equal(leaf_one))

        tree1 = RankTree(children=[leaf_zero, leaf_one])
        self.assertEqual(tree1, tree1)
        self.assertNotEqual(tree1, unlabelled_leaf)
        self.assertFalse(tree1.shape_equal(unlabelled_leaf))

        tree2 = RankTree(children=[leaf_two, leaf_one])
        self.assertNotEqual(tree1, tree2)
        self.assertTrue(tree1.shape_equal(tree2))

    def test_is_symmetrical(self):
        unlabelled_leaf = RankTree(children=[])
        self.assertTrue(unlabelled_leaf.is_symmetrical())
        three_leaf_asym = RankTree(
            children=[
                unlabelled_leaf,
                RankTree(children=[unlabelled_leaf, unlabelled_leaf]),
            ]
        )
        self.assertFalse(three_leaf_asym.is_symmetrical())
        six_leaf_sym = RankTree(children=[three_leaf_asym, three_leaf_asym])
        self.assertTrue(six_leaf_sym.is_symmetrical())
