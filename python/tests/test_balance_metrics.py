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
Tests for tree balance/imbalance metrics.
"""
import math

import numpy as np
import pytest

import tests
import tskit
from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


def node_path(tree, u):
    path = []
    u = tree.parent(u)
    while u != tskit.NULL:
        path.append(u)
        u = tree.parent(u)
    return path


def sackin_index_definition(tree):
    return sum(tree.depth(u) for u in tree.leaves())


def colless_index_definition(tree):
    is_binary = all(
        tree.num_children(u) == 2 for u in tree.nodes() if tree.is_internal(u)
    )
    if tree.num_roots != 1:
        raise ValueError("Colless index not defined for multiroot trees")
    if not is_binary:
        raise ValueError("Colless index not defined for nonbinary trees")

    return sum(
        abs(
            len(list(tree.leaves(tree.left_child(u))))
            - len(list(tree.leaves(tree.right_child(u))))
        )
        for u in tree.nodes()
        if tree.is_internal(u)
    )


def b1_index_definition(tree):
    return sum(
        1 / max(tree.path_length(n, leaf) for leaf in tree.leaves(n))
        for n in tree.nodes()
        if tree.parent(n) != tskit.NULL and tree.is_internal(n)
    )


def b2_index_definition(tree, base=10):
    if tree.num_roots != 1:
        raise ValueError("B2 index is only defined for trees with one root")
    proba = [
        np.prod([1 / tree.num_children(u) for u in node_path(tree, leaf)])
        for leaf in tree.leaves()
    ]
    return -sum(p * math.log(p, base) for p in proba)


class TestDefinitions:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_sackin(self, ts):
        for tree in ts.trees():
            assert tree.sackin_index() == sackin_index_definition(tree)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_colless(self, ts):
        for tree in ts.trees():
            is_binary = all(
                tree.num_children(u) == 2 for u in tree.nodes() if tree.is_internal(u)
            )
            if tree.num_roots != 1 or not is_binary:
                with pytest.raises(tskit.LibraryError):
                    tree.colless_index()
                with pytest.raises(ValueError):
                    colless_index_definition(tree)
            else:
                assert tree.colless_index() == colless_index_definition(tree)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_b1(self, ts):
        for tree in ts.trees():
            assert tree.b1_index() == pytest.approx(b1_index_definition(tree))

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_b2(self, ts):
        for tree in ts.trees():
            if tree.num_roots != 1:
                with pytest.raises(tskit.LibraryError, match="MULTIROOT"):
                    tree.b2_index()
                with pytest.raises(ValueError):
                    b2_index_definition(tree)
            else:
                assert tree.b2_index() == pytest.approx(b2_index_definition(tree))

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    @pytest.mark.parametrize("base", [0.1, 1.1, 2, 10, math.e, np.array([3])[0]])
    def test_b2_base(self, ts, base):
        for tree in ts.trees():
            if tree.num_roots != 1:
                with pytest.raises(tskit.LibraryError, match="MULTIROOT"):
                    tree.b2_index(base)
                with pytest.raises(ValueError):
                    b2_index_definition(tree, base)
            else:
                assert tree.b2_index(base) == pytest.approx(
                    b2_index_definition(tree, base)
                )


class TestBalancedBinaryOdd:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0      1
    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(3)

    def test_sackin(self):
        assert self.tree().sackin_index() == 5

    def test_colless(self):
        assert self.tree().colless_index() == 1

    def test_b1(self):
        assert self.tree().b1_index() == 1

    def test_b2(self):
        assert self.tree().b2_index(base=10) == pytest.approx(0.4515, rel=1e-3)


class TestBalancedBinaryEven:
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1
    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(4)

    def test_sackin(self):
        assert self.tree().sackin_index() == 8

    def test_colless(self):
        assert self.tree().colless_index() == 0

    def test_b1(self):
        assert self.tree().b1_index() == 2

    def test_b2(self):
        assert self.tree().b2_index() == pytest.approx(0.602, rel=1e-3)

    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            (2, 2),
            (3, 1.2618595071429148),
            (4, 1.0),
            (5, 0.8613531161467861),
            (10, 0.6020599913279623),
            (100, 0.30102999566398114),
            (1000000, 0.10034333188799373),
            (2.718281828459045, 1.3862943611198906),
        ],
    )
    def test_b2_base(self, base, expected):
        assert self.tree().b2_index(base) == expected

    @pytest.mark.parametrize("base", [0, -0.001, -1, -1e-6, -1e200])
    def test_b2_bad_base(self, base):
        with pytest.raises(ValueError, match="math domain"):
            self.tree().b2_index(base=base)

    def test_b2_base1(self):
        with pytest.raises(ZeroDivisionError):
            self.tree().b2_index(base=1)


class TestBalancedTernary:
    # 2.00┊        12         ┊
    #     ┊   ┏━━━━━╋━━━━━┓   ┊
    # 1.00┊   9    10    11   ┊
    #     ┊ ┏━╋━┓ ┏━╋━┓ ┏━╋━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(9, arity=3)

    def test_sackin(self):
        assert self.tree().sackin_index() == 18

    def test_colless(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_NONBINARY"):
            self.tree().colless_index()

    def test_b1(self):
        assert self.tree().b1_index() == 3

    def test_b2(self):
        assert self.tree().b2_index() == pytest.approx(0.954, rel=1e-3)


class TestStarN10:
    # 1.00┊         10          ┊
    #     ┊ ┏━┳━┳━┳━┳┻┳━┳━┳━┳━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 9 ┊
    #     0                     1
    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_star(10)

    def test_sackin(self):
        assert self.tree().sackin_index() == 10

    def test_colless(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_NONBINARY"):
            self.tree().colless_index()

    def test_b1(self):
        assert self.tree().b1_index() == 0

    def test_b2(self):
        assert self.tree().b2_index() == pytest.approx(0.9999, rel=1e-3)


class TestCombN5:
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
        return tskit.Tree.generate_comb(5)

    def test_sackin(self):
        assert self.tree().sackin_index() == 14

    def test_colless(self):
        assert self.tree().colless_index() == 6

    def test_b1(self):
        assert self.tree().b1_index() == pytest.approx(1.833, rel=1e-3)

    def test_b2(self):
        assert self.tree().b2_index() == pytest.approx(0.564, rel=1e-3)


class TestMultiRootBinary:
    # 3.00┊            15     ┊
    #     ┊          ┏━━┻━┓   ┊
    # 2.00┊   11     ┃   14   ┊
    #     ┊  ┏━┻━┓   ┃  ┏━┻┓  ┊
    # 1.00┊  9  10  12  ┃ 13  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    @tests.cached_example
    def tree(self):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert self.tree().sackin_index() == 20

    def test_colless(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_MULTIROOT"):
            self.tree().colless_index()

    def test_b1(self):
        assert self.tree().b1_index() == 4.5

    def test_b2(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_MULTIROOT"):
            self.tree().b2_index()


class TestEmpty:
    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert self.tree().sackin_index() == 0

    def test_colless(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_MULTIROOT"):
            self.tree().colless_index()

    def test_b1(self):
        assert self.tree().b1_index() == 0

    def test_b2(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_MULTIROOT"):
            self.tree().b2_index()


class TestTreeInNullState:
    @tests.cached_example
    def tree(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def test_sackin(self):
        assert self.tree().sackin_index() == 0

    def test_colless(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_MULTIROOT"):
            self.tree().colless_index()

    def test_b1(self):
        assert self.tree().b1_index() == 0

    def test_b2(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_MULTIROOT"):
            self.tree().b2_index()


class TestAllRootsN5:
    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert self.tree().sackin_index() == 0

    def test_colless(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_MULTIROOT"):
            self.tree().colless_index()

    def test_b1(self):
        assert self.tree().b1_index() == 0

    def test_b2(self):
        with pytest.raises(tskit.LibraryError, match="UNDEFINED_MULTIROOT"):
            self.tree().b2_index()
