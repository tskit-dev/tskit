# MIT License
#
# Copyright (c) 2022 Tskit Developers
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
import pytest

import tests
import tskit
from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


def sackin_index_definition(tree):
    return sum(tree.depth(u) for u in tree.leaves())


class TestDefinitions:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_sackin(self, ts):
        for tree in ts.trees():
            assert tree.sackin_index() == sackin_index_definition(tree)


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


class TestMultiRoot:
    #
    # 1.00┊   9    10    11   ┊
    #     ┊ ┏━╋━┓ ┏━╋━┓ ┏━╋━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    @tests.cached_example
    def tree(self):
        tables = tskit.Tree.generate_balanced(9, arity=3).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 12:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert self.tree().sackin_index() == 9


class TestEmpty:
    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert self.tree().sackin_index() == 0


class TestTreeInNullState:
    @tests.cached_example
    def tree(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def test_sackin(self):
        assert self.tree().sackin_index() == 0


class TestAllRootsN5:
    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert self.tree().sackin_index() == 0
