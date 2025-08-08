# MIT License
#
# Copyright (c) 2023 Tskit Developers
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
Tests for tree iterator schemes. Mostly used to develop the incremental
iterator infrastructure.
"""
import msprime
import numpy as np
import pytest

import tests
import tskit
from tests import tsutil
from tests.tsutil import get_example_tree_sequences


class StatefulTree:
    """
    Just enough functionality to mimic the low-level tree implementation
    for testing of forward/backward moving.
    """

    def __init__(self, ts):
        self.ts = ts
        self.tree_pos = tsutil.TreeIndexes(ts)
        self.parent = [-1 for _ in range(ts.num_nodes)]

    def __str__(self):
        s = f"parent: {self.parent}\nposition:\n"
        for line in str(self.tree_pos).splitlines():
            s += f"\t{line}\n"
        return s

    def assert_equal(self, other):
        assert self.parent == other.parent
        assert self.tree_pos.index == other.tree_pos.index
        assert self.tree_pos.interval == other.tree_pos.interval

    def next(self):  # NOQA: A003
        valid = self.tree_pos.next()
        if valid:
            for j in range(self.tree_pos.out_range.start, self.tree_pos.out_range.stop):
                e = self.tree_pos.out_range.order[j]
                c = self.ts.edges_child[e]
                self.parent[c] = -1
            for j in range(self.tree_pos.in_range.start, self.tree_pos.in_range.stop):
                e = self.tree_pos.in_range.order[j]
                c = self.ts.edges_child[e]
                p = self.ts.edges_parent[e]
                self.parent[c] = p
        return valid

    def prev(self):
        valid = self.tree_pos.prev()
        if valid:
            for j in range(
                self.tree_pos.out_range.start, self.tree_pos.out_range.stop, -1
            ):
                e = self.tree_pos.out_range.order[j]
                c = self.ts.edges_child[e]
                self.parent[c] = -1
            for j in range(
                self.tree_pos.in_range.start, self.tree_pos.in_range.stop, -1
            ):
                e = self.tree_pos.in_range.order[j]
                c = self.ts.edges_child[e]
                p = self.ts.edges_parent[e]
                self.parent[c] = p
        return valid

    def iter_forward(self, index):
        while self.tree_pos.index != index:
            self.next()

    def seek_forward(self, index):
        old_left, old_right = self.tree_pos.interval
        self.tree_pos.seek_forward(index)
        left, right = self.tree_pos.interval
        for j in range(self.tree_pos.out_range.start, self.tree_pos.out_range.stop):
            e = self.tree_pos.out_range.order[j]
            e_left = self.ts.edges_left[e]
            # We only need to remove an edge if it's in the current tree, which
            # can only happen if the edge's left coord is < the current tree's
            # right coordinate.
            if e_left < old_right:
                c = self.ts.edges_child[e]
                assert self.parent[c] != -1
                self.parent[c] = -1
            assert e_left < left
        for j in range(self.tree_pos.in_range.start, self.tree_pos.in_range.stop):
            e = self.tree_pos.in_range.order[j]
            if self.ts.edges_left[e] <= left < self.ts.edges_right[e]:
                c = self.ts.edges_child[e]
                p = self.ts.edges_parent[e]
                self.parent[c] = p
            else:
                a = self.tree_pos.in_range.start
                b = self.tree_pos.in_range.stop
                # The first and last indexes in the range should always be valid
                # for the tree.
                assert a < j < b - 1

    def seek_backward(self, index):
        old_left, old_right = self.tree_pos.interval
        self.tree_pos.seek_backward(index)
        left, right = self.tree_pos.interval
        for j in range(self.tree_pos.out_range.start, self.tree_pos.out_range.stop, -1):
            e = self.tree_pos.out_range.order[j]
            e_right = self.ts.edges_right[e]
            # We only need to remove an edge if it's in the current tree, which
            # can only happen if the edge's right coord is >= the current tree's
            # right coordinate.
            if e_right >= old_right:
                c = self.ts.edges_child[e]
                assert self.parent[c] != -1
                self.parent[c] = -1
            assert e_right > right
        for j in range(self.tree_pos.in_range.start, self.tree_pos.in_range.stop, -1):
            e = self.tree_pos.in_range.order[j]
            if self.ts.edges_right[e] >= right > self.ts.edges_left[e]:
                c = self.ts.edges_child[e]
                p = self.ts.edges_parent[e]
                self.parent[c] = p
            else:
                a = self.tree_pos.in_range.start
                b = self.tree_pos.in_range.stop
                # The first and last indexes in the range should always be valid
                # for the tree.
                assert a > j > b + 1

    def iter_backward(self, index):
        while self.tree_pos.index != index:
            self.prev()


def check_iters_forward(ts):
    alg_t_output = tsutil.algorithm_T(ts)
    lib_tree = tskit.Tree(ts)
    tree_pos = tsutil.TreeIndexes(ts)
    sample_count = np.zeros(ts.num_nodes, dtype=int)
    sample_count[ts.samples()] = 1
    parent1 = [-1 for _ in range(ts.num_nodes)]
    i = 0
    lib_tree.next()
    while tree_pos.next():
        out_times = []
        for j in range(tree_pos.out_range.start, tree_pos.out_range.stop):
            e = tree_pos.out_range.order[j]
            c = ts.edges_child[e]
            p = ts.edges_parent[e]
            out_times.append(ts.nodes_time[p])
            parent1[c] = -1
        in_times = []
        for j in range(tree_pos.in_range.start, tree_pos.in_range.stop):
            e = tree_pos.in_range.order[j]
            c = ts.edges_child[e]
            p = ts.edges_parent[e]
            in_times.append(ts.nodes_time[p])
            parent1[c] = p
        # We must visit the edges in *increasing* time order on the way in,
        # and *decreasing* order on the way out. Otherwise we get quadratic
        # behaviour for algorithms that need to propagate changes up to the
        # root.
        assert out_times == sorted(out_times, reverse=True)
        assert in_times == sorted(in_times)

        interval, parent2 = next(alg_t_output)
        assert list(interval) == list(tree_pos.interval)
        assert parent1 == parent2

        assert lib_tree.index == i
        assert list(lib_tree.interval) == list(interval)
        assert list(lib_tree.parent_array[:-1]) == parent1

        lib_tree.next()
        i += 1
    assert i == ts.num_trees
    assert lib_tree.index == -1
    assert next(alg_t_output, None) is None


def check_iters_back(ts):
    alg_t_output = [
        (list(interval), list(parent)) for interval, parent in tsutil.algorithm_T(ts)
    ]
    i = len(alg_t_output) - 1

    lib_tree = tskit.Tree(ts)
    tree_pos = tsutil.TreeIndexes(ts)
    parent1 = [-1 for _ in range(ts.num_nodes)]

    lib_tree.last()

    while tree_pos.prev():
        # print(tree_pos.out_range)
        out_times = []
        for j in range(tree_pos.out_range.start, tree_pos.out_range.stop, -1):
            e = tree_pos.out_range.order[j]
            c = ts.edges_child[e]
            p = ts.edges_parent[e]
            out_times.append(ts.nodes_time[p])
            parent1[c] = -1
        in_times = []
        for j in range(tree_pos.in_range.start, tree_pos.in_range.stop, -1):
            e = tree_pos.in_range.order[j]
            c = ts.edges_child[e]
            p = ts.edges_parent[e]
            in_times.append(ts.nodes_time[p])
            parent1[c] = p

        # We must visit the edges in *increasing* time order on the way in,
        # and *decreasing* order on the way out. Otherwise we get quadratic
        # behaviour for algorithms that need to propagate changes up to the
        # root.
        assert out_times == sorted(out_times, reverse=True)
        assert in_times == sorted(in_times)

        interval, parent2 = alg_t_output[i]
        assert list(interval) == list(tree_pos.interval)
        assert parent1 == parent2

        assert lib_tree.index == i
        assert list(lib_tree.interval) == list(interval)
        assert list(lib_tree.parent_array[:-1]) == parent1

        lib_tree.prev()
        i -= 1

    assert lib_tree.index == -1
    assert i == -1


def check_forward_back_sweep(ts):
    alg_t_output = [
        (list(interval), list(parent)) for interval, parent in tsutil.algorithm_T(ts)
    ]
    for j in range(ts.num_trees - 1):
        tree = StatefulTree(ts)
        # Seek forward to j
        k = 0
        while k <= j:
            tree.next()
            interval, parent = alg_t_output[k]
            assert tree.tree_pos.index == k
            assert list(tree.tree_pos.interval) == interval
            assert parent == tree.parent
            k += 1
        k = j
        # And back to zero
        while k >= 0:
            interval, parent = alg_t_output[k]
            assert tree.tree_pos.index == k
            assert list(tree.tree_pos.interval) == interval
            assert parent == tree.parent
            tree.prev()
            k -= 1


def check_seek_forward_out_range_is_empty(ts, index):
    tree = StatefulTree(ts)
    tree.seek_forward(index)
    assert tree.tree_pos.out_range.start == tree.tree_pos.out_range.stop
    tree.iter_backward(-1)
    tree.seek_forward(index)
    assert tree.tree_pos.out_range.start == tree.tree_pos.out_range.stop


def check_seek_backward_out_range_is_empty(ts, index):
    tree = StatefulTree(ts)
    tree.seek_backward(index)
    assert tree.tree_pos.out_range.start == tree.tree_pos.out_range.stop
    tree.iter_forward(-1)
    tree.seek_backward(index)
    assert tree.tree_pos.out_range.start == tree.tree_pos.out_range.stop


def check_seek_forward_from_null(ts, index):
    tree1 = StatefulTree(ts)
    tree1.seek_forward(index)
    tree2 = StatefulTree(ts)
    tree2.iter_forward(index)
    tree1.assert_equal(tree2)


def check_seek_backward_from_null(ts, index):
    tree1 = StatefulTree(ts)
    tree1.seek_backward(index)
    tree2 = StatefulTree(ts)
    tree2.iter_backward(index)
    tree1.assert_equal(tree2)


def check_seek_forward_from_first(ts, index):
    tree1 = StatefulTree(ts)
    tree1.next()
    tree1.seek_forward(index)
    tree2 = StatefulTree(ts)
    tree2.iter_forward(index)
    tree1.assert_equal(tree2)


def check_seek_backward_from_last(ts, index):
    tree1 = StatefulTree(ts)
    tree1.prev()
    tree1.seek_backward(index)
    tree2 = StatefulTree(ts)
    tree2.iter_backward(index)


class TestDirectionSwitching:
    # 2.00┊       ┊   4   ┊   4   ┊   4   ┊
    #     ┊       ┊ ┏━┻┓  ┊  ┏┻━┓ ┊  ┏┻━┓ ┊
    # 1.00┊   3   ┊ ┃  3  ┊  3  ┃ ┊  3  ┃ ┊
    #     ┊ ┏━╋━┓ ┊ ┃ ┏┻┓ ┊ ┏┻┓ ┃ ┊ ┏┻┓ ┃ ┊
    # 0.00┊ 0 1 2 ┊ 0 1 2 ┊ 0 2 1 ┊ 0 1 2 ┊
    #     0       1       2       3       4
    # index   0       1       2       3
    def ts(self):
        return tsutil.all_trees_ts(3)

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_iter_backward_matches_iter_forward(self, index):
        ts = self.ts()
        tree1 = StatefulTree(ts)
        tree1.iter_forward(index)
        tree2 = StatefulTree(ts)
        tree2.iter_backward(index)
        tree1.assert_equal(tree2)

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_prev_from_seek_forward(self, index):
        tree1 = StatefulTree(self.ts())
        tree1.seek_forward(index)
        tree1.prev()
        tree2 = StatefulTree(self.ts())
        tree2.seek_forward(index - 1)
        tree1.assert_equal(tree2)

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_seek_forward_from_prev(self, index):
        tree1 = StatefulTree(self.ts())
        tree1.iter_forward(index)
        tree1.prev()
        tree1.seek_forward(index)
        tree2 = StatefulTree(self.ts())
        tree2.iter_forward(index)
        tree1.assert_equal(tree2)

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_seek_forward_from_null(self, index):
        ts = self.ts()
        check_seek_forward_from_null(ts, index)

    def test_seek_forward_next_null(self):
        tree1 = StatefulTree(self.ts())
        tree1.seek_forward(3)
        tree1.next()
        assert tree1.tree_pos.index == -1
        assert list(tree1.tree_pos.interval) == [0, 0]

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_next_from_seek_backward(self, index):
        tree1 = StatefulTree(self.ts())
        tree1.seek_backward(index)
        tree1.next()
        tree2 = StatefulTree(self.ts())
        tree2.seek_backward(index + 1)
        tree1.assert_equal(tree2)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_seek_backward_from_next(self, index):
        tree1 = StatefulTree(self.ts())
        tree1.iter_backward(index)
        tree1.next()
        tree1.seek_backward(index)
        tree2 = StatefulTree(self.ts())
        tree2.iter_backward(index)
        tree1.assert_equal(tree2)

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_seek_backward_from_null(self, index):
        ts = self.ts()
        check_seek_backward_from_null(ts, index)

    def test_seek_backward_prev_null(self):
        tree1 = StatefulTree(self.ts())
        tree1.seek_backward(0)
        tree1.prev()
        assert tree1.tree_pos.index == -1
        assert list(tree1.tree_pos.interval) == [0, 0]

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_seek_forward_out_range_is_empty(self, index):
        ts = self.ts()
        check_seek_forward_out_range_is_empty(ts, index)

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_seek_backward_out_range_is_empty(self, index):
        ts = self.ts()
        check_seek_backward_out_range_is_empty(ts, index)


class TestTreeIndexesStep:
    def ts(self):
        return tsutil.all_trees_ts(3)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_tree_position_step_forward(self, index):
        ts = self.ts()
        tree1_pos = tsutil.TreeIndexes(ts)
        tree1_pos.seek_forward(index)
        tree1_pos.step(direction=1)
        tree2_pos = tsutil.TreeIndexes(ts)
        tree2_pos.seek_forward(index + 1)
        tree1_pos.assert_equal(tree2_pos)

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_tree_position_step_backward(self, index):
        ts = self.ts()
        tree1_pos = tsutil.TreeIndexes(ts)
        tree1_pos.seek_backward(index)
        tree1_pos.step(direction=-1)
        tree2_pos = tsutil.TreeIndexes(ts)
        tree2_pos.seek_backward(index - 1)
        tree1_pos.assert_equal(tree2_pos)

    def test_tree_position_step_invalid_direction(self):
        ts = self.ts()
        # Test for unallowed direction
        with pytest.raises(ValueError, match="Direction must be FORWARD"):
            tsutil.TreeIndexes(ts).step(direction="foo")


class TestSeeking:
    @tests.cached_example
    def ts(self):
        ts = tsutil.all_trees_ts(4)
        assert ts.num_trees == 26
        return ts

    @pytest.mark.parametrize("index", range(26))
    def test_seek_forward_from_null(self, index):
        ts = self.ts()
        check_seek_forward_from_null(ts, index)

    @pytest.mark.parametrize("index", range(1, 26))
    def test_seek_forward_from_first(self, index):
        ts = self.ts()
        check_seek_forward_from_first(ts, index)

    @pytest.mark.parametrize("index", range(1, 26))
    def test_seek_last_from_index(self, index):
        ts = self.ts()
        tree1 = StatefulTree(ts)
        tree1.iter_forward(index)
        tree1.seek_forward(ts.num_trees - 1)
        tree2 = StatefulTree(ts)
        tree2.prev()
        tree1.assert_equal(tree2)

    @pytest.mark.parametrize("index", range(26))
    def test_seek_backward_from_null(self, index):
        ts = self.ts()
        check_seek_backward_from_null(ts, index)

    @pytest.mark.parametrize("index", range(0, 25))
    def test_seek_backward_from_last(self, index):
        ts = self.ts()
        check_seek_backward_from_last(ts, index)

    @pytest.mark.parametrize("index", range(0, 25))
    def test_seek_first_from_index(self, index):
        ts = self.ts()
        tree1 = StatefulTree(ts)
        tree1.iter_backward(index)
        tree1.seek_backward(0)
        tree2 = StatefulTree(ts)
        tree2.next()
        tree1.assert_equal(tree2)

    @pytest.mark.parametrize("index", range(26))
    def test_seek_forward_out_range_is_empty(self, index):
        ts = self.ts()
        check_seek_forward_out_range_is_empty(ts, index)

    @pytest.mark.parametrize("index", range(26))
    def test_seek_backward_out_range_is_empty(self, index):
        ts = self.ts()
        check_seek_backward_out_range_is_empty(ts, index)


class TestAllTreesTs:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_forward_full(self, n):
        ts = tsutil.all_trees_ts(n)
        check_iters_forward(ts)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_back_full(self, n):
        ts = tsutil.all_trees_ts(n)
        check_iters_back(ts)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_forward_back(self, n):
        ts = tsutil.all_trees_ts(n)
        check_forward_back_sweep(ts)


class TestManyTreesSimulationExample:
    @tests.cached_example
    def ts(self):
        ts = msprime.sim_ancestry(
            10, sequence_length=1000, recombination_rate=0.1, random_seed=1234
        )
        assert ts.num_trees > 250
        return ts

    @pytest.mark.parametrize("index", [1, 5, 10, 50, 100])
    def test_seek_forward_from_null(self, index):
        ts = self.ts()
        check_seek_forward_from_null(ts, index)

    @pytest.mark.parametrize("num_trees", [1, 5, 10, 50, 100])
    def test_seek_forward_from_mid(self, num_trees):
        ts = self.ts()
        start_index = ts.num_trees // 2
        dest_index = min(start_index + num_trees, ts.num_trees - 1)
        tree1 = StatefulTree(ts)
        tree1.iter_forward(start_index)
        tree1.seek_forward(dest_index)
        tree2 = StatefulTree(ts)
        tree2.iter_forward(dest_index)
        tree1.assert_equal(tree2)

    @pytest.mark.parametrize("index", [1, 5, 10, 50, 100])
    def test_seek_backward_from_null(self, index):
        ts = self.ts()
        check_seek_backward_from_null(ts, index)

    @pytest.mark.parametrize("num_trees", [1, 5, 10, 50, 100])
    def test_seek_backward_from_mid(self, num_trees):
        ts = self.ts()
        start_index = ts.num_trees // 2
        dest_index = max(start_index - num_trees, 0)
        tree1 = StatefulTree(ts)
        tree1.iter_backward(start_index)
        tree1.seek_backward(dest_index)
        tree2 = StatefulTree(ts)
        tree2.iter_backward(dest_index)

    @pytest.mark.parametrize("index", [1, 5, 10, 50, 100])
    def test_seek_forward_out_range_is_empty(self, index):
        ts = self.ts()
        check_seek_forward_out_range_is_empty(ts, index)

    @pytest.mark.parametrize("index", [1, 5, 10, 50, 100])
    def test_seek_backward_out_range_is_empty(self, index):
        ts = self.ts()
        check_seek_backward_out_range_is_empty(ts, index)

    def test_forward_full(self):
        check_iters_forward(self.ts())

    def test_back_full(self):
        check_iters_back(self.ts())


class TestSuiteExamples:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_forward_full(self, ts):
        check_iters_forward(ts)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_back_full(self, ts):
        check_iters_back(ts)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_seek_forward_from_null(self, ts):
        index = ts.num_trees // 2
        check_seek_forward_from_null(ts, index)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_seek_forward_from_first(self, ts):
        index = ts.num_trees - 1
        check_seek_forward_from_first(ts, index)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_seek_backward_from_null(self, ts):
        index = ts.num_trees // 2
        check_seek_backward_from_null(ts, index)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_seek_backward_from_last(self, ts):
        index = 0
        check_seek_backward_from_last(ts, index)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_seek_forward_out_range_is_empty(self, ts):
        index = ts.num_trees // 2
        check_seek_forward_out_range_is_empty(ts, index)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_seek_backward_out_range_is_empty(self, ts):
        index = ts.num_trees // 2
        check_seek_backward_out_range_is_empty(ts, index)
