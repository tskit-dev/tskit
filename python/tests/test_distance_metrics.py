# MIT License
#
# Copyright (c) 2024 Tskit Developers
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
Tests for tree distance metrics.
"""
import itertools

import dendropy
import msprime
import pytest
from dendropy.calculate import treecompare

import tests
import tskit


class TestTreeSameSamples:
    # Tree1
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1
    #
    # Tree2
    # 3.00┊   6     ┊
    #     ┊ ┏━┻━┓   ┊
    # 2.00┊ ┃   5   ┊
    #     ┊ ┃ ┏━┻┓  ┊
    # 1.00┊ ┃ ┃  4  ┊
    #     ┊ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1

    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(4)

    @tests.cached_example
    def tree_other(self):
        return tskit.Tree.generate_comb(4)

    def test_rf_distance(self):
        assert self.tree().rf_distance(self.tree_other()) == 2


class TestTreeDifferentSamples:
    # Tree1
    # 2.00┊     6     ┊
    #     ┊   ┏━┻━┓   ┊
    # 1.00┊   4   5   ┊
    #     ┊  ┏┻┓ ┏┻┓  ┊
    # 0.00┊  0 1 2 3  ┊
    #     0           1
    #
    # Tree2
    # 4.00┊   8       ┊
    #     ┊ ┏━┻━┓     ┊
    # 3.00┊ ┃   7     ┊
    #     ┊ ┃ ┏━┻━┓   ┊
    # 2.00┊ ┃ ┃   6   ┊
    #     ┊ ┃ ┃ ┏━┻┓  ┊
    # 1.00┊ ┃ ┃ ┃  5  ┊
    #     ┊ ┃ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 4 ┊
    #     0           1

    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(4)

    @tests.cached_example
    def tree_other(self):
        return tskit.Tree.generate_comb(5)

    def test_rf_distance(self):
        assert self.tree().rf_distance(self.tree_other()) == 8


class TestTreeMultiRoots:
    # Tree1
    # 4.00┊        15             ┊
    #     ┊     ┏━━━┻━━━┓         ┊
    # 3.00┊     ┃      14         ┊
    #     ┊     ┃     ┏━┻━┓       ┊
    # 2.00┊    12     ┃  13       ┊
    #     ┊   ┏━┻━┓   ┃  ┏┻┓      ┊
    # 1.00┊   9  10   ┃  ┃ 11     ┊
    #     ┊  ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓    ┊
    # 0.00┊  0 1 2 3 4 5 6 7 8    ┊
    #     0                       1
    #
    # Tree2
    # 3.00┊              15       ┊
    #     ┊            ┏━━┻━┓     ┊
    # 2.00┊     11     ┃   14     ┊
    #     ┊    ┏━┻━┓   ┃  ┏━┻┓    ┊
    # 1.00┊    9  10  12  ┃ 13    ┊
    #     ┊   ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓   ┊
    # 0.00┊   0 1 2 3 4 5 6 7 8   ┊
    #     0                       1

    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(9)

    @tests.cached_example
    def tree_other(self):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            self.tree().rf_distance(self.tree_other())


class TestEmpty:
    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    @tests.cached_example
    def tree_other(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            self.tree().rf_distance(self.tree_other())


class TestTreeInNullState:
    @tests.cached_example
    def tsk_tree1(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    @tests.cached_example
    def tree_other(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            self.tsk_tree1().rf_distance(self.tree_other())


class TestAllRootsN5:
    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            self.tree().rf_distance(self.tree())


class TestWithPackages:
    def to_dendropy(self, newick_data, tns):
        return dendropy.Tree.get(
            data=newick_data,
            schema="newick",
            rooting="force-rooted",
            taxon_namespace=tns,
        )

    def dendropy_rf_distance(self, tree1, tree2, weighted=False):
        tns = dendropy.TaxonNamespace()
        tree1 = self.to_dendropy(tree1.as_newick(), tns)
        tree2 = self.to_dendropy(tree2.as_newick(), tns)
        tree1.encode_bipartitions()
        tree2.encode_bipartitions()
        if weighted:
            return treecompare.weighted_robinson_foulds_distance(tree1, tree2)
        else:
            return treecompare.unweighted_robinson_foulds_distance(tree1, tree2)

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 20])
    def test_rf_distance_against_dendropy(self, n):
        trees = []
        for seed in [42, 43]:
            ts = msprime.sim_ancestry(n, ploidy=1, random_seed=seed)
            trees.append(ts.first())
        rf1 = trees[0].rf_distance(trees[1])
        rf2 = self.dendropy_rf_distance(trees[0], trees[1])
        assert rf1 == rf2


class TestDistanceBetween:
    @pytest.mark.parametrize(
        ("u", "v"),
        itertools.combinations([0, 1, 2, 3], 2),
    )
    def test_distance_between_sample(self, u, v):
        ts = msprime.sim_ancestry(
            2, sequence_length=10, recombination_rate=0.1, random_seed=42
        )
        test_tree = ts.first()
        assert test_tree.distance_between(u, v) == pytest.approx(
            ts.diversity([u, v], mode="branch", windows="trees")[0]
        )

    def test_distance_between_same_node(self):
        ts = msprime.sim_ancestry(
            2, sequence_length=10, recombination_rate=0.1, random_seed=42
        )
        test_tree = ts.first()
        assert test_tree.distance_between(0, 0) == 0

    def test_distance_between_nodes(self):
        # 4.00┊   8       ┊
        #     ┊ ┏━┻━┓     ┊
        # 3.00┊ ┃   7     ┊
        #     ┊ ┃ ┏━┻━┓   ┊
        # 2.00┊ ┃ ┃   6   ┊
        #     ┊ ┃ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃ ┃  5  ┊
        #     ┊ ┃ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 4 ┊
        #     0           1
        ts = tskit.Tree.generate_comb(5)
        assert ts.distance_between(1, 7) == 3.0
        assert ts.distance_between(6, 8) == 2.0

    def test_distance_between_invalid_nodes(self):
        ts = tskit.Tree.generate_comb(5)
        with pytest.raises(ValueError):
            ts.distance_between(0, 100)
