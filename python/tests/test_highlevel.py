# MIT License
#
# Copyright (c) 2018-2025 Tskit Developers
# Copyright (c) 2015-2018 University of Oxford
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
Test cases for the high level interface to tskit.
"""
import collections
import dataclasses
import decimal
import inspect
import io
import itertools
import json
import math
import os
import pathlib
import pickle
import platform
import random
import re
import tempfile
import unittest
import uuid as _uuid
import warnings
from xml.etree import ElementTree

import kastore
import msprime
import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import _tskit
import tests as tests
import tests.simplify as simplify
import tests.tsutil as tsutil
import tskit
import tskit.metadata as metadata
import tskit.util as util
from tskit import UNKNOWN_TIME


def traversal_preorder(tree, root=None):
    roots = tree.roots if root is None else [root]
    for node in roots:
        yield node
        for child in tree.children(node):
            yield from traversal_preorder(tree, child)


def traversal_postorder(tree, root=None):
    roots = tree.roots if root is None else [root]
    for node in roots:
        for child in tree.children(node):
            yield from traversal_postorder(tree, child)
        yield node


def traversal_inorder(tree, root=None):
    roots = tree.roots if root is None else [root]
    for node in roots:
        children = list(tree.children(node))
        half = len(children) // 2
        for child in children[:half]:
            yield from traversal_inorder(tree, child)
        yield node
        for child in children[half:]:
            yield from traversal_inorder(tree, child)


def traversal_levelorder(tree, root=None):
    yield from sorted(list(tree.nodes(root)), key=lambda u: tree.depth(u))


def _traversal_minlex_postorder(tree, u):
    """
    For a given input ID u, this function returns a tuple whose first value
    is the minimum leaf node ID under node u, and whose second value is
    a list containing the minlex postorder for the subtree rooted at node u.
    The first value is needed for sorting, and the second value is what
    finally gets returned.
    """
    children = tree.children(u)
    if len(children) > 0:
        children_return = [_traversal_minlex_postorder(tree, c) for c in children]
        # sorts by first value, which is the minimum leaf node ID
        children_return.sort(key=lambda x: x[0])
        minlex_postorder = []
        for _, child_minlex_postorder in children_return:
            minlex_postorder.extend(child_minlex_postorder)
        minlex_postorder.extend([u])
        return (children_return[0][0], minlex_postorder)
    else:
        return (u, [u])


def traversal_minlex_postorder(tree, root=None):
    roots = tree.roots if root is None else [root]
    root_lists = [_traversal_minlex_postorder(tree, node) for node in roots]
    for _, node_list in sorted(root_lists, key=lambda x: x[0]):
        yield from node_list


def traversal_timeasc(tree, root=None):
    yield from sorted(tree.nodes(root), key=lambda u: (tree.time(u), u))


def traversal_timedesc(tree, root=None):
    yield from sorted(tree.nodes(root), key=lambda u: (tree.time(u), u), reverse=True)


traversal_map = {
    "preorder": traversal_preorder,
    "postorder": traversal_postorder,
    "inorder": traversal_inorder,
    "levelorder": traversal_levelorder,
    "breadthfirst": traversal_levelorder,
    "minlex_postorder": traversal_minlex_postorder,
    "timeasc": traversal_timeasc,
    "timedesc": traversal_timedesc,
}


def simple_get_pairwise_diversity(haplotypes):
    """
    Returns the value of pi for the specified haplotypes.
    """
    # Very simplistic algorithm...
    n = len(haplotypes)
    pi = 0
    for k in range(n):
        for j in range(k):
            for u, v in zip(haplotypes[j], haplotypes[k]):
                pi += u != v
    return 2 * pi / (n * (n - 1))


def simplify_tree_sequence(ts, samples, filter_sites=True):
    """
    Simple tree-by-tree algorithm to get a simplify of a tree sequence.
    """
    s = simplify.Simplifier(ts, samples, filter_sites=filter_sites)
    return s.simplify()


def oriented_forests(n):
    """
    Implementation of Algorithm O from TAOCP section 7.2.1.6.
    Generates all canonical n-node oriented forests.
    """
    p = [k - 1 for k in range(0, n + 1)]
    k = 1
    while k != 0:
        yield p
        if p[n] > 0:
            p[n] = p[p[n]]
            yield p
        k = n
        while k > 0 and p[k] == 0:
            k -= 1
        if k != 0:
            j = p[k]
            d = k - j
            not_done = True
            while not_done:
                if p[k - d] == p[j]:
                    p[k] = p[j]
                else:
                    p[k] = p[k - d] + d
                if k == n:
                    not_done = False
                else:
                    k += 1


def get_mrca(pi, x, y):
    """
    Returns the most recent common ancestor of nodes x and y in the
    oriented forest pi.
    """
    x_parents = [x]
    j = x
    while j != 0:
        j = pi[j]
        x_parents.append(j)
    y_parents = {y: None}
    j = y
    while j != 0:
        j = pi[j]
        y_parents[j] = None
    # We have the complete list of parents for x and y back to root.
    mrca = 0
    j = 0
    while x_parents[j] not in y_parents:
        j += 1
    mrca = x_parents[j]
    return mrca


def get_samples(ts, time=None, population=None):
    samples = []
    for node in ts.nodes():
        keep = bool(node.is_sample())
        if time is not None:
            if isinstance(time, (int, float)):
                keep &= np.isclose(node.time, time)
            if isinstance(time, (tuple, list, np.ndarray)):
                keep &= node.time >= time[0]
                keep &= node.time < time[1]
        if population is not None:
            keep &= node.population == population
        if keep:
            samples.append(node.id)
    return np.array(samples)


class TestTreeTraversals:
    def test_bad_traversal_order(self, simple_degree2_ts_fixture):
        tree = simple_degree2_ts_fixture.first()
        for bad_order in ["pre", "post", "preorderorder", ("x",), b"preorder"]:
            with pytest.raises(ValueError, match="Traversal order"):
                tree.nodes(order=bad_order)

    @pytest.mark.parametrize("order", list(traversal_map.keys()))
    def test_returned_types(self, order):
        ts = msprime.sim_ancestry(2, random_seed=234)
        tree = ts.first()
        iterator = tree.nodes(order=order)
        assert isinstance(iterator, collections.abc.Iterable)
        lst = list(iterator)
        assert len(lst) > 0
        for u in lst:
            assert isinstance(u, int)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    @pytest.mark.parametrize("order", list(traversal_map.keys()))
    def test_traversals_virtual_root(self, ts, order):
        tree = ts.first()
        node_list2 = list(traversal_map[order](tree, tree.virtual_root))
        node_list1 = list(tree.nodes(tree.virtual_root, order=order))
        assert tree.virtual_root in node_list1
        assert node_list1 == node_list2

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    @pytest.mark.parametrize("order", list(traversal_map.keys()))
    def test_traversals(self, ts, order):
        tree = next(ts.trees())
        traverser = traversal_map[order]
        node_list1 = list(tree.nodes(order=order))
        node_list2 = list(traverser(tree))
        assert node_list1 == node_list2

    def test_binary_example(self):
        t = tskit.Tree.generate_balanced(5)
        #     8
        #  ┏━━┻━┓
        #  ┃    7
        #  ┃  ┏━┻┓
        #  5  ┃  6
        # ┏┻┓ ┃ ┏┻┓
        # 0 1 2 3 4

        def f(node=None, order=None):
            return list(t.nodes(node, order))

        assert f(order="preorder") == [8, 5, 0, 1, 7, 2, 6, 3, 4]
        assert f(order="postorder") == [0, 1, 5, 2, 3, 4, 6, 7, 8]
        assert f(order="inorder") == [0, 5, 1, 8, 2, 7, 3, 6, 4]
        assert f(order="levelorder") == [8, 5, 7, 0, 1, 2, 6, 3, 4]
        assert f(order="breadthfirst") == [8, 5, 7, 0, 1, 2, 6, 3, 4]
        assert f(order="timeasc") == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert f(order="timedesc") == [8, 7, 6, 5, 4, 3, 2, 1, 0]
        assert f(order="minlex_postorder") == [0, 1, 5, 2, 3, 4, 6, 7, 8]

        q = t.virtual_root
        assert f(q, order="preorder") == [q, 8, 5, 0, 1, 7, 2, 6, 3, 4]
        assert f(q, order="postorder") == [0, 1, 5, 2, 3, 4, 6, 7, 8, q]
        assert f(q, order="inorder") == [q, 0, 5, 1, 8, 2, 7, 3, 6, 4]
        assert f(q, order="levelorder") == [q, 8, 5, 7, 0, 1, 2, 6, 3, 4]
        assert f(q, order="breadthfirst") == [q, 8, 5, 7, 0, 1, 2, 6, 3, 4]
        assert f(q, order="timeasc") == [0, 1, 2, 3, 4, 5, 6, 7, 8, q]
        assert f(q, order="timedesc") == [q, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        assert f(q, order="minlex_postorder") == [0, 1, 5, 2, 3, 4, 6, 7, 8, q]

        assert f(7, order="preorder") == [7, 2, 6, 3, 4]
        assert f(7, order="postorder") == [2, 3, 4, 6, 7]
        assert f(7, order="inorder") == [2, 7, 3, 6, 4]
        assert f(7, order="levelorder") == [7, 2, 6, 3, 4]
        assert f(7, order="breadthfirst") == [7, 2, 6, 3, 4]
        assert f(7, order="timeasc") == [2, 3, 4, 6, 7]
        assert f(7, order="timedesc") == [7, 6, 4, 3, 2]
        assert f(7, order="minlex_postorder") == [2, 3, 4, 6, 7]

    def test_ternary_example(self):
        t = tskit.Tree.generate_balanced(7, arity=3)
        #      10
        #  ┏━━━┳┻━━━┓
        #  7   8    9
        # ┏┻┓ ┏┻┓ ┏━╋━┓
        # 0 1 2 3 4 5 6

        def f(node=None, order=None):
            return list(t.nodes(node, order))

        assert f(order="preorder") == [10, 7, 0, 1, 8, 2, 3, 9, 4, 5, 6]
        assert f(order="postorder") == [0, 1, 7, 2, 3, 8, 4, 5, 6, 9, 10]
        assert f(order="inorder") == [0, 7, 1, 10, 2, 8, 3, 4, 9, 5, 6]
        assert f(order="levelorder") == [10, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert f(order="breadthfirst") == [10, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert f(order="timeasc") == list(range(11))
        assert f(order="timedesc") == list(reversed(range(11)))
        assert f(order="minlex_postorder") == [0, 1, 7, 2, 3, 8, 4, 5, 6, 9, 10]

        q = t.virtual_root
        assert f(q, order="preorder") == [q, 10, 7, 0, 1, 8, 2, 3, 9, 4, 5, 6]
        assert f(q, order="postorder") == [0, 1, 7, 2, 3, 8, 4, 5, 6, 9, 10, q]
        assert f(q, order="inorder") == [q, 0, 7, 1, 10, 2, 8, 3, 4, 9, 5, 6]
        assert f(q, order="levelorder") == [q, 10, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert f(q, order="breadthfirst") == [q, 10, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert f(q, order="timeasc") == list(range(12))
        assert f(q, order="timedesc") == list(reversed(range(12)))
        assert f(q, order="minlex_postorder") == [0, 1, 7, 2, 3, 8, 4, 5, 6, 9, 10, q]

        assert f(9, order="preorder") == [9, 4, 5, 6]
        assert f(9, order="postorder") == [4, 5, 6, 9]
        assert f(9, order="inorder") == [4, 9, 5, 6]
        assert f(9, order="levelorder") == [9, 4, 5, 6]
        assert f(9, order="breadthfirst") == [9, 4, 5, 6]
        assert f(9, order="timeasc") == [4, 5, 6, 9]
        assert f(9, order="timedesc") == [9, 6, 5, 4]
        assert f(9, order="minlex_postorder") == [4, 5, 6, 9]

    def test_multiroot_example(self):
        tables = tskit.Tree.generate_balanced(7, arity=3).tree_sequence.dump_tables()
        tables.edges.truncate(len(tables.edges) - 3)
        t = tables.tree_sequence().first()

        #  7   8    9
        # ┏┻┓ ┏┻┓ ┏━╋━┓
        # 0 1 2 3 4 5 6
        def f(node=None, order=None):
            return list(t.nodes(node, order))

        assert f(order="preorder") == [7, 0, 1, 8, 2, 3, 9, 4, 5, 6]
        assert f(order="postorder") == [0, 1, 7, 2, 3, 8, 4, 5, 6, 9]
        assert f(order="inorder") == [0, 7, 1, 2, 8, 3, 4, 9, 5, 6]
        assert f(order="levelorder") == [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert f(order="breadthfirst") == [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert f(order="timeasc") == list(range(10))
        assert f(order="timedesc") == list(reversed(range(10)))
        assert f(order="minlex_postorder") == [0, 1, 7, 2, 3, 8, 4, 5, 6, 9]

        q = t.virtual_root
        assert f(q, order="preorder") == [q, 7, 0, 1, 8, 2, 3, 9, 4, 5, 6]
        assert f(q, order="postorder") == [0, 1, 7, 2, 3, 8, 4, 5, 6, 9, q]
        assert f(q, order="inorder") == [0, 7, 1, q, 2, 8, 3, 4, 9, 5, 6]
        assert f(q, order="levelorder") == [q, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert f(q, order="breadthfirst") == [q, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert f(q, order="timeasc") == list(range(10)) + [q]
        assert f(q, order="timedesc") == [q] + list(reversed(range(10)))
        assert f(q, order="minlex_postorder") == [0, 1, 7, 2, 3, 8, 4, 5, 6, 9, q]

        assert f(9, order="preorder") == [9, 4, 5, 6]
        assert f(9, order="postorder") == [4, 5, 6, 9]
        assert f(9, order="inorder") == [4, 9, 5, 6]
        assert f(9, order="levelorder") == [9, 4, 5, 6]
        assert f(9, order="breadthfirst") == [9, 4, 5, 6]
        assert f(9, order="minlex_postorder") == [4, 5, 6, 9]
        assert f(9, order="timeasc") == [4, 5, 6, 9]
        assert f(9, order="timedesc") == [9, 6, 5, 4]

    def test_multiroot_non_lexical_example(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time    population  individual  metadata
        0   1   0.000000    0   -1  b''
        1   1   0.000000    0   -1  b''
        2   1   0.000000    0   -1  b''
        3   1   0.000000    0   -1  b''
        4   1   0.000000    0   -1  b''
        5   1   0.000000    0   -1  b''
        6   1   0.000000    0   -1  b''
        7   1   0.000000    0   -1  b''
        8   1   0.000000    0   -1  b''
        9   1   0.000000    0   -1  b''
        10  0   0.047734    0   -1  b''
        11  0   0.061603    0   -1  b''
        12  0   0.189503    0   -1  b''
        13  0   0.275885    0   -1  b''
        14  0   0.518301    0   -1  b''
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.000000    10000.000000    10  0
        0.000000    10000.000000    10  2
        0.000000    10000.000000    11  9
        0.000000    10000.000000    11  10
        0.000000    10000.000000    12  3
        0.000000    10000.000000    12  7
        0.000000    10000.000000    13  5
        0.000000    10000.000000    13  11
        0.000000    10000.000000    14  1
        0.000000    10000.000000    14  8
        """
        )
        ts = tskit.load_text(
            nodes, edges, sequence_length=10000, strict=False, base64_metadata=False
        )
        t = ts.first()

        # Note: this is drawn out in "tree" order.
        #                 14
        #                 ┏┻┓
        #          13     ┃ ┃
        #         ┏━┻━┓   ┃ ┃
        #     12  ┃   ┃   ┃ ┃
        #     ┏┻┓ ┃   ┃   ┃ ┃
        #     ┃ ┃ ┃  11   ┃ ┃
        #     ┃ ┃ ┃ ┏━┻┓  ┃ ┃
        #     ┃ ┃ ┃ ┃ 10  ┃ ┃
        #     ┃ ┃ ┃ ┃ ┏┻┓ ┃ ┃
        # 4 6 3 7 5 9 0 2 1 8

        def f(node=None, order=None):
            return list(t.nodes(node, order))

        pre = f(order="preorder")
        post = f(order="postorder")
        inord = f(order="inorder")
        level = f(order="levelorder")
        breadth = f(order="breadthfirst")
        timeasc = f(order="timeasc")
        timedesc = f(order="timedesc")
        minlex = f(order="minlex_postorder")
        assert pre == [4, 6, 12, 3, 7, 13, 5, 11, 9, 10, 0, 2, 14, 1, 8]
        assert post == [4, 6, 3, 7, 12, 5, 9, 0, 2, 10, 11, 13, 1, 8, 14]
        assert inord == [4, 6, 3, 12, 7, 5, 13, 9, 11, 0, 10, 2, 1, 14, 8]
        assert level == [4, 6, 12, 13, 14, 3, 7, 5, 11, 1, 8, 9, 10, 0, 2]
        assert breadth == [4, 6, 12, 13, 14, 3, 7, 5, 11, 1, 8, 9, 10, 0, 2]
        assert timeasc == list(range(15))
        assert timedesc == list(reversed(range(15)))

        # And the minlex tree:
        #         14
        #         ┏┻┓
        #    13   ┃ ┃
        #   ┏━┻━┓ ┃ ┃
        #   ┃   ┃ ┃ ┃ 12
        #   ┃   ┃ ┃ ┃ ┏┻┓
        #  11   ┃ ┃ ┃ ┃ ┃
        #  ┏┻━┓ ┃ ┃ ┃ ┃ ┃
        # 10  ┃ ┃ ┃ ┃ ┃ ┃
        # ┏┻┓ ┃ ┃ ┃ ┃ ┃ ┃
        # 0 2 9 5 1 8 3 7 4 6
        assert minlex == [0, 2, 10, 9, 11, 5, 13, 1, 8, 14, 3, 7, 12, 4, 6]

    @pytest.mark.parametrize(
        ["order", "expected"],
        [
            ("preorder", [[9, 6, 2, 3, 7, 4, 5, 0, 1], [10, 4, 8, 5, 0, 1, 6, 2, 3]]),
            ("inorder", [[2, 6, 3, 9, 4, 7, 0, 5, 1], [4, 10, 0, 5, 1, 8, 2, 6, 3]]),
            ("postorder", [[2, 3, 6, 4, 0, 1, 5, 7, 9], [4, 0, 1, 5, 2, 3, 6, 8, 10]]),
            ("levelorder", [[9, 6, 7, 2, 3, 4, 5, 0, 1], [10, 4, 8, 5, 6, 0, 1, 2, 3]]),
            (
                "breadthfirst",
                [[9, 6, 7, 2, 3, 4, 5, 0, 1], [10, 4, 8, 5, 6, 0, 1, 2, 3]],
            ),
            ("timeasc", [[0, 1, 2, 3, 4, 5, 6, 7, 9], [0, 1, 2, 3, 4, 5, 6, 8, 10]]),
            ("timedesc", [[9, 7, 6, 5, 4, 3, 2, 1, 0], [10, 8, 6, 5, 4, 3, 2, 1, 0]]),
            (
                "minlex_postorder",
                [[0, 1, 5, 4, 7, 2, 3, 6, 9], [0, 1, 5, 2, 3, 6, 8, 4, 10]],
            ),
        ],
    )
    def test_ts_example(self, order, expected):
        # 1.20┊           ┊  10       ┊
        #     ┊           ┊ ┏━┻━━┓    ┊
        # 0.90┊     9     ┊ ┃    ┃    ┊
        #     ┊  ┏━━┻━┓   ┊ ┃    ┃    ┊
        # 0.60┊  ┃    ┃   ┊ ┃    8    ┊
        #     ┊  ┃    ┃   ┊ ┃  ┏━┻━┓  ┊
        # 0.44┊  ┃    7   ┊ ┃  ┃   ┃  ┊
        #     ┊  ┃  ┏━┻┓  ┊ ┃  ┃   ┃  ┊
        # 0.21┊  6  ┃  ┃  ┊ ┃  ┃   6  ┊
        #     ┊ ┏┻┓ ┃  ┃  ┊ ┃  ┃  ┏┻┓ ┊
        # 0.15┊ ┃ ┃ ┃  5  ┊ ┃  5  ┃ ┃ ┊
        #     ┊ ┃ ┃ ┃ ┏┻┓ ┊ ┃ ┏┻┓ ┃ ┃ ┊
        # 0.00┊ 2 3 4 0 1 ┊ 4 0 1 2 3 ┊
        #   0.00        0.50        1.00
        nodes = """\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       1       0               0.00000000000000
        3       1       0               0.00000000000000
        4       1       0               0.00000000000000
        5       0       0               0.14567111023387
        6       0       0               0.21385545626353
        7       0       0               0.43508024345063
        8       0       0               0.60156352971203
        9       0       0               0.90000000000000
        10      0       0               1.20000000000000
        """
        edges = """\
        id      left            right           parent  child
        0       0.00000000      1.00000000      5       0,1
        1       0.00000000      1.00000000      6       2,3
        2       0.00000000      0.50000000      7       4,5
        3       0.50000000      1.00000000      8       5,6
        4       0.00000000      0.50000000      9       6,7
        5       0.50000000      1.00000000      10      4,8
        """
        ts = tskit.load_text(
            nodes=io.StringIO(nodes), edges=io.StringIO(edges), strict=False
        )
        tree_orders = [list(tree.nodes(order=order)) for tree in ts.trees()]
        assert tree_orders == expected

    def test_polytomy_inorder(self):
        """
        If there are N children, current inorder traversal first visits
        floor(N/2) children, then the parent, then the remaining children.
        Here we explicitly test that behaviour.
        """
        #
        #    __4__
        #   / / \ \
        #  0 1   2 3
        #
        nodes_polytomy_4 = """\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       1       0               0.00000000000000
        3       1       0               0.00000000000000
        4       0       0               1.00000000000000
        """
        edges_polytomy_4 = """\
        id      left            right           parent  child
        0       0.00000000      1.00000000      4       0,1,2,3
        """
        #
        #    __5__
        #   / /|\ \
        #  0 1 2 3 4
        #
        nodes_polytomy_5 = """\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       1       0               0.00000000000000
        3       1       0               0.00000000000000
        4       1       0               0.00000000000000
        5       0       0               1.00000000000000
        """
        edges_polytomy_5 = """\
        id      left            right           parent  child
        0       0.00000000      1.00000000      5       0,1,2,3,4
        """
        for nodes_string, edges_string, expected_result in [
            [nodes_polytomy_4, edges_polytomy_4, [[0, 1, 4, 2, 3]]],
            [nodes_polytomy_5, edges_polytomy_5, [[0, 1, 5, 2, 3, 4]]],
        ]:
            ts = tskit.load_text(
                nodes=io.StringIO(nodes_string),
                edges=io.StringIO(edges_string),
                strict=False,
            )
            tree_orders = []
            for tree in ts.trees():
                tree_orders.append(list(tree.nodes(order="inorder")))
            assert tree_orders == expected_result

    def test_minlex_postorder_multiple_roots(self):
        #
        #    10    8     9     11
        #   / \   / \   / \   / \
        #  5   3 2   4 6   7 1   0
        #
        nodes_string = """\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       1       0               0.00000000000000
        3       1       0               0.00000000000000
        4       1       0               0.00000000000000
        5       1       0               0.00000000000000
        6       1       0               0.00000000000000
        7       1       0               0.00000000000000
        8       0       0               1.00000000000000
        9       0       0               1.00000000000000
        10      0       0               1.00000000000000
        11      0       0               1.00000000000000
        """
        edges_string = """\
        id      left            right           parent  child
        0       0.00000000      1.00000000      8       2,4
        1       0.00000000      1.00000000      9       6,7
        2       0.00000000      1.00000000      10      5,3
        3       0.00000000      1.00000000      11      1,0
        """
        expected_result = [[0, 1, 11, 2, 4, 8, 3, 5, 10, 6, 7, 9]]
        ts = tskit.load_text(
            nodes=io.StringIO(nodes_string),
            edges=io.StringIO(edges_string),
            strict=False,
        )
        tree_orders = []
        for tree in ts.trees():
            tree_orders.append(list(tree.nodes(order="minlex_postorder")))
        assert tree_orders == expected_result


class TestMRCA:
    """
    Test both the tree.mrca and tree.tmrca methods.
    """

    t = tskit.Tree.generate_balanced(3)
    #  4
    # ┏━┻┓
    # ┃  3
    # ┃ ┏┻┓
    # 0 1 2

    @pytest.mark.parametrize("args, expected", [((2, 1), 3), ((0, 1, 2), 4)])
    def test_two_or_more_args(self, args, expected):
        assert self.t.mrca(*args) == expected
        assert self.t.tmrca(*args) == self.t.tree_sequence.nodes_time[expected]

    def test_less_than_two_args(self):
        with pytest.raises(ValueError):
            self.t.mrca(1)
        with pytest.raises(ValueError):
            self.t.tmrca(1)

    def test_no_args(self):
        with pytest.raises(ValueError):
            self.t.mrca()
        with pytest.raises(ValueError):
            self.t.tmrca()

    def test_same_args(self):
        assert self.t.mrca(0, 0, 0, 0) == 0
        assert self.t.tmrca(0, 0, 0, 0) == self.t.tree_sequence.nodes_time[0]

    def test_different_tree_levels(self):
        assert self.t.mrca(0, 3) == 4
        assert self.t.tmrca(0, 3) == self.t.tree_sequence.nodes_time[4]

    def test_out_of_bounds_args(self):
        with pytest.raises(ValueError):
            self.t.mrca(0, 6)
        with pytest.raises(ValueError):
            self.t.tmrca(0, 6)

    def test_virtual_root_arg(self):
        assert self.t.mrca(0, 5) == 5
        assert np.isposinf(self.t.tmrca(0, 5))

    def test_multiple_roots(self):
        ts = tskit.Tree.generate_balanced(10).tree_sequence
        ts = ts.delete_intervals([ts.first().interval])
        assert ts.first().mrca(*ts.samples()) == tskit.NULL
        # We decided to raise an error for tmrca here, rather than report inf
        # see https://github.com/tskit-dev/tskit/issues/2801
        with pytest.raises(ValueError, match="do not share a common ancestor"):
            ts.first().tmrca(0, 6)


class TestPathLength:
    t = tskit.Tree.generate_balanced(9)
    #         16
    #    ┏━━━━┻━━━┓
    #    ┃       15
    #    ┃     ┏━━┻━┓
    #   11     ┃   14
    #  ┏━┻━┓   ┃  ┏━┻┓
    #  9  10  12  ┃ 13
    # ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓
    # 0 1 2 3 4 5 6 7 8

    def test_tmrca_leaf(self):
        assert self.t.path_length(0, 16) == 3
        assert self.t.path_length(16, 0) == 3
        assert self.t.path_length(7, 16) == 4

    def test_equal_depth(self):
        assert self.t.path_length(5, 16) == self.t.depth(5)

    def test_two_leaves(self):
        assert self.t.path_length(0, 8) == 7

    def test_two_leaves_depth(self):
        assert self.t.path_length(0, 8) == self.t.depth(0) + self.t.depth(8)

    @pytest.mark.parametrize("args", [[], [1], [1, 2, 3]])
    def test_bad_num_args(self, args):
        with pytest.raises(TypeError):
            self.t.path_length(*args)

    @pytest.mark.parametrize("bad_arg", [[], "1"])
    def test_bad_arg_type(self, bad_arg):
        with pytest.raises(TypeError):
            self.t.path_length(0, bad_arg)
        with pytest.raises(TypeError):
            self.t.path_length(bad_arg, 0)

    def test_same_args(self):
        assert self.t.path_length(10, 10) == 0

    def test_different_tree_levels(self):
        assert self.t.path_length(1, 10) == 3

    def test_out_of_bounds_args(self):
        with pytest.raises(ValueError):
            self.t.path_length(0, 20)

    @pytest.mark.parametrize("u", range(17))
    def test_virtual_root_arg(self, u):
        assert self.t.path_length(u, self.t.virtual_root) == self.t.depth(u) + 1
        assert self.t.path_length(self.t.virtual_root, u) == self.t.depth(u) + 1

    def test_both_args_virtual_root(self):
        assert self.t.path_length(self.t.virtual_root, self.t.virtual_root) == 0

    def test_no_mrca(self):
        tree = self.t.copy()
        tree.clear()
        assert math.isinf(tree.path_length(0, 1))


class TestMRCACalculator:
    """
    Class to test the Schieber-Vishkin algorithm.

    These tests are included here as we use the MRCA calculator below in
    our tests.
    """

    def test_all_oriented_forests(self):
        # Runs through all possible oriented forests and checks all possible
        # node pairs using an inferior algorithm.
        for n in range(2, 9):
            for pi in oriented_forests(n):
                sv = tests.MRCACalculator(pi)
                for j in range(1, n + 1):
                    for k in range(1, j + 1):
                        mrca = get_mrca(pi, j, k)
                        assert mrca == sv.get_mrca(j, k)


class HighLevelTestCase:
    """
    Superclass of tests on the high level interface.
    """

    def verify_tree_mrcas(self, st):
        # Check the mrcas
        oriented_forest = [st.get_parent(j) for j in range(st.tree_sequence.num_nodes)]
        mrca_calc = tests.MRCACalculator(oriented_forest)
        # We've done exhaustive tests elsewhere, no need to go
        # through the combinations.
        for j in range(st.tree_sequence.num_nodes):
            mrca = st.get_mrca(0, j)
            assert mrca == mrca_calc.get_mrca(0, j)
            if mrca != tskit.NULL:
                assert st.get_time(mrca) == st.get_tmrca(0, j)

    def verify_tree_branch_lengths(self, tree):
        for u in tree.tree_sequence.samples():
            while tree.parent(u) != tskit.NULL:
                length = tree.time(tree.parent(u)) - tree.time(u)
                assert length > 0.0
                assert tree.branch_length(u) == length
                u = tree.parent(u)
            assert tree.parent(u) == tskit.NULL
            assert tree.branch_length(u) == 0

    def verify_tree_structure(self, st):
        roots = set()
        for u in st.samples():
            # verify the path to root
            assert st.is_sample(u)
            times = []
            while st.get_parent(u) != tskit.NULL:
                v = st.get_parent(u)
                times.append(st.get_time(v))
                assert st.get_time(v) >= 0.0
                assert u in st.get_children(v)
                u = v
            roots.add(u)
            assert times == sorted(times)
        assert sorted(list(roots)) == sorted(st.roots)
        assert len(st.roots) == st.num_roots
        u = st.left_root
        roots = []
        while u != tskit.NULL:
            roots.append(u)
            u = st.right_sib(u)
        assert roots == st.roots
        # To a top-down traversal, and make sure we meet all the samples.
        samples = []
        for root in st.roots:
            stack = [root]
            while len(stack) > 0:
                u = stack.pop()
                assert u != tskit.NULL
                if st.is_sample(u):
                    samples.append(u)
                if st.is_leaf(u):
                    assert len(st.get_children(u)) == 0
                else:
                    for c in reversed(st.get_children(u)):
                        stack.append(c)
                # Check that we get the correct number of samples at each
                # node.
                assert st.get_num_samples(u) == len(list(st.samples(u)))
                assert st.get_num_tracked_samples(u) == 0
        assert sorted(samples) == sorted(st.samples())
        # Check the parent dict
        pi = st.get_parent_dict()
        for root in st.roots:
            assert root not in pi
        for k, v in pi.items():
            assert st.get_parent(k) == v
        assert st.num_samples() == len(samples)
        assert sorted(st.samples()) == sorted(samples)

    def verify_tree_depths(self, st):
        for root in st.roots:
            stack = [(root, 0)]
            while len(stack) > 0:
                u, depth = stack.pop()
                assert st.depth(u) == depth
                for c in st.children(u):
                    stack.append((c, depth + 1))

    def verify_tree(self, st):
        self.verify_tree_mrcas(st)
        self.verify_tree_branch_lengths(st)
        self.verify_tree_structure(st)
        self.verify_tree_depths(st)

    def verify_trees(self, ts):
        pts = tests.PythonTreeSequence(ts)
        iter1 = ts.trees()
        iter2 = pts.trees()
        length = 0
        num_trees = 0
        breakpoints = [0]
        for st1, st2 in zip(iter1, iter2):
            assert st1.get_sample_size() == ts.get_sample_size()
            roots = set()
            for u in ts.samples():
                root = u
                while st1.get_parent(root) != tskit.NULL:
                    root = st1.get_parent(root)
                roots.add(root)
            assert st1.left_root == st2.left_root
            assert sorted(list(roots)) == sorted(st1.roots)
            assert st1.roots == st2.roots
            if len(roots) == 0:
                assert st1.root == tskit.NULL
            elif len(roots) == 1:
                assert st1.root == list(roots)[0]
            else:
                with pytest.raises(ValueError):
                    st1.root
            assert st2 == st1
            assert not (st2 != st1)
            left, right = st1.get_interval()
            breakpoints.append(right)
            assert left == pytest.approx(length)
            assert left >= 0
            assert right > left
            assert right <= ts.get_sequence_length()
            length += right - left
            self.verify_tree(st1)
            num_trees += 1
        with pytest.raises(StopIteration):
            next(iter1)
        with pytest.raises(StopIteration):
            next(iter2)
        assert ts.get_num_trees() == num_trees
        assert breakpoints == list(ts.breakpoints())
        assert length == pytest.approx(ts.get_sequence_length())


class TestNumpySamples:
    """
    Tests that we correctly handle samples as numpy arrays when passed to
    various methods.
    """

    def get_tree_sequence(self, num_demes=4, times=None, n=40):
        if times is None:
            times = [0]
        return msprime.simulate(
            samples=[
                msprime.Sample(time=t, population=j % num_demes)
                for j in range(n)
                for t in times
            ],
            population_configurations=[
                msprime.PopulationConfiguration() for _ in range(num_demes)
            ],
            migration_matrix=[
                [int(j != k) for j in range(num_demes)] for k in range(num_demes)
            ],
            random_seed=1,
            mutation_rate=10,
        )

    def test_samples(self):
        d = 4
        ts = self.get_tree_sequence(d)
        assert np.array_equal(ts.samples(), np.arange(ts.num_samples, dtype=np.int32))
        total = 0
        for pop in range(d):
            subsample = ts.samples(pop)
            total += subsample.shape[0]
            assert np.array_equal(subsample, ts.samples(population=pop))
            assert list(subsample) == [
                node.id
                for node in ts.nodes()
                if node.population == pop and node.is_sample()
            ]
        assert total == ts.num_samples

    @pytest.mark.parametrize("time", [0, 0.1, 1 / 3, 1 / 4, 5 / 7])
    def test_samples_time(self, time):
        ts = self.get_tree_sequence(num_demes=2, n=20, times=[time, 0.2, 1, 15])
        assert np.array_equal(get_samples(ts, time=time), ts.samples(time=time))
        for population in (None, 0):
            assert np.array_equal(
                get_samples(ts, time=time, population=population),
                ts.samples(time=time, population=population),
            )

    @pytest.mark.parametrize(
        "time_interval",
        [
            [0, 0.1],
            (0, 1 / 3),
            np.array([1 / 4, 2 / 3]),
            (0.345, 5 / 7),
            (-1, 1),
        ],
    )
    def test_samples_time_interval(self, time_interval):
        rng = np.random.default_rng(seed=931)
        times = rng.uniform(low=time_interval[0], high=2 * time_interval[1], size=20)
        ts = self.get_tree_sequence(num_demes=2, n=1, times=times)
        assert np.array_equal(
            get_samples(ts, time=time_interval),
            ts.samples(time=time_interval),
        )
        for population in (None, 0):
            assert np.array_equal(
                get_samples(ts, time=time_interval, population=population),
                ts.samples(time=time_interval, population=population),
            )

    def test_samples_example(self):
        tables = tskit.TableCollection(sequence_length=10)
        time = [0, 0, 1, 1, 1, 3, 3.00001, 3.0 - 0.0001, 1 / 3]
        pops = [1, 3, 1, 2, 1, 1, 1, 3, 1]
        for _ in range(max(pops) + 1):
            tables.populations.add_row()
        for t, p in zip(time, pops):
            tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE,
                time=t,
                population=p,
            )
        # add not-samples also
        for t, p in zip(time, pops):
            tables.nodes.add_row(
                flags=0,
                time=t,
                population=p,
            )
        ts = tables.tree_sequence()
        assert np.array_equal(
            ts.samples(),
            np.arange(len(time)),
        )
        assert np.array_equal(
            ts.samples(time=[0, np.inf]),
            np.arange(len(time)),
        )
        assert np.array_equal(
            ts.samples(time=0),
            [0, 1],
        )
        # default tolerance is 1e-5
        assert np.array_equal(
            ts.samples(time=0.3333333),
            [8],
        )
        assert np.array_equal(
            ts.samples(time=3),
            [5, 6],
        )
        assert np.array_equal(
            ts.samples(time=1),
            [2, 3, 4],
        )
        assert np.array_equal(
            ts.samples(time=1, population=2),
            [3],
        )
        assert np.array_equal(
            ts.samples(population=0),
            [],
        )
        assert np.array_equal(
            ts.samples(population=1),
            [0, 2, 4, 5, 6, 8],
        )
        assert np.array_equal(
            ts.samples(population=2),
            [3],
        )
        assert np.array_equal(
            ts.samples(time=[0, 3]),
            [0, 1, 2, 3, 4, 7, 8],
        )
        # note tuple instead of array
        assert np.array_equal(
            ts.samples(time=(1, 3)),
            [2, 3, 4, 7],
        )
        assert np.array_equal(
            ts.samples(time=[0, 3], population=1),
            [0, 2, 4, 8],
        )
        assert np.array_equal(
            ts.samples(time=[0.333333, 3]),
            [2, 3, 4, 7, 8],
        )
        assert np.array_equal(
            ts.samples(time=[100, np.inf]),
            [],
        )
        assert np.array_equal(
            ts.samples(time=-1),
            [],
        )
        assert np.array_equal(
            ts.samples(time=[-100, 100]),
            np.arange(len(time)),
        )
        assert np.array_equal(
            ts.samples(time=[-100, -1]),
            [],
        )

    def test_samples_time_errors(self):
        ts = self.get_tree_sequence(4)
        # error incorrect types
        with pytest.raises(ValueError):
            ts.samples(time="s")
        with pytest.raises(ValueError):
            ts.samples(time=[])
        with pytest.raises(ValueError):
            ts.samples(time=np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            ts.samples(time=(1, 2, 3))
        # error using min and max switched
        with pytest.raises(ValueError):
            ts.samples(time=(2.4, 1))

    def test_samples_args(self, ts_fixture):
        ts_fixture.samples(1)
        with pytest.raises(TypeError, match="takes from 1 to 2 positional arguments"):
            ts_fixture.samples(1, 2)

    def test_genotype_matrix_indexing(self):
        num_demes = 4
        ts = self.get_tree_sequence(num_demes)
        G = ts.genotype_matrix()
        for d in range(num_demes):
            samples = ts.samples(population=d)
            total = 0
            for tree in ts.trees(tracked_samples=samples):
                for mutation in tree.mutations():
                    total += tree.num_tracked_samples(mutation.node)
            assert total == np.sum(G[:, samples])

    def test_genotype_indexing(self):
        num_demes = 6
        ts = self.get_tree_sequence(num_demes)
        for d in range(num_demes):
            samples = ts.samples(population=d)
            total = 0
            for tree in ts.trees(tracked_samples=samples):
                for mutation in tree.mutations():
                    total += tree.num_tracked_samples(mutation.node)
            other_total = 0
            for variant in ts.variants():
                other_total += np.sum(variant.genotypes[samples])
            assert total == other_total

    def test_pairwise_diversity(self):
        num_demes = 6
        ts = self.get_tree_sequence(num_demes)
        pi1 = ts.pairwise_diversity(ts.samples())
        pi2 = ts.pairwise_diversity()
        assert pi1 == pi2
        for d in range(num_demes):
            samples = ts.samples(population=d)
            pi1 = ts.pairwise_diversity(samples)
            pi2 = ts.pairwise_diversity(list(samples))
            assert pi1 == pi2

    def test_simplify(self):
        num_demes = 3
        ts = self.get_tree_sequence(num_demes)
        sts = ts.simplify(samples=ts.samples())
        assert ts.num_samples == sts.num_samples
        for d in range(num_demes):
            samples = ts.samples(population=d)
            sts = ts.simplify(samples=samples)
            assert sts.num_samples == samples.shape[0]


class TestTreeSequence(HighLevelTestCase):
    """
    Tests for the tree sequence object.
    """

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_row_getter(self, ts):
        for table_name, table in ts.tables_dict.items():
            sequence = getattr(ts, table_name)()
            element_name = table_name[:-1]  # cut off the "s": "edges" -> "edge"
            element_accessor = getattr(ts, element_name)
            for i, n in enumerate(sequence):
                assert element_accessor(i) == n
                assert element_accessor(-(table.num_rows - i)) == n
            with pytest.raises(IndexError):
                element_accessor(table.num_rows)
            with pytest.raises(IndexError):
                element_accessor(-(table.num_rows + 1))

    @pytest.mark.parametrize("index", [0.1, float(0), None, np.array([0, 1]), np.inf])
    def test_bad_row_getter(self, index, simple_degree2_ts_fixture):
        for table_name in simple_degree2_ts_fixture.tables_dict.keys():
            element_name = table_name[:-1]  # cut off the "s": "edges" -> "edge"
            element_accessor = getattr(simple_degree2_ts_fixture, element_name)
            if element_name == "site" and index is None:
                # special case
                match = "id or position must be provided"
            else:
                match = "integer type"
            with pytest.raises(TypeError, match=match):
                element_accessor(index)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_discrete_genome(self, ts):
        def is_discrete(a):
            return np.all(np.floor(a) == a)

        tables = ts.tables
        discrete_genome = (
            is_discrete([tables.sequence_length])
            and is_discrete(tables.edges.left)
            and is_discrete(tables.edges.right)
            and is_discrete(tables.sites.position)
            and is_discrete(tables.migrations.left)
            and is_discrete(tables.migrations.right)
        )
        assert ts.discrete_genome == discrete_genome

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_discrete_time(self, ts):
        def is_discrete(a):
            return np.all(np.logical_or(np.floor(a) == a, tskit.is_unknown_time(a)))

        tables = ts.tables
        discrete_time = (
            is_discrete(tables.nodes.time)
            and is_discrete(tables.mutations.time)
            and is_discrete(tables.migrations.time)
        )
        assert ts.discrete_time == discrete_time

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_trees(self, ts):
        self.verify_trees(ts)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_mutations(self, ts):
        self.verify_mutations(ts)

    @pytest.mark.skipif(not _tskit.HAS_NUMPY_2, reason="Requires NumPy 2.0 or higher")
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_mutation_inherited_state_property(self, ts):
        inherited_states = ts.mutations_inherited_state
        for mut in ts.mutations():
            expected = inherited_states[mut.id]
            actual = mut.inherited_state
            assert actual == expected

            if mut.parent == tskit.NULL:
                expected_direct = ts.site(mut.site).ancestral_state
            else:
                expected_direct = ts.mutation(mut.parent).derived_state
            assert actual == expected_direct

    def verify_pairwise_diversity(self, ts):
        haplotypes = ts.genotype_matrix(isolated_as_missing=False).T
        if ts.num_samples == 0:
            with pytest.raises(ValueError, match="at least one element"):
                ts.get_pairwise_diversity()
            return
        pi1 = ts.get_pairwise_diversity()
        pi2 = simple_get_pairwise_diversity(haplotypes)
        assert pi1 == pytest.approx(pi2)
        assert pi1 >= 0.0
        assert not math.isnan(pi1)
        # Check for a subsample.
        num_samples = ts.get_sample_size() // 2 + 1
        samples = list(ts.samples())[:num_samples]
        pi1 = ts.get_pairwise_diversity(samples)
        pi2 = simple_get_pairwise_diversity([haplotypes[j] for j in range(num_samples)])
        assert pi1 == pytest.approx(pi2)
        assert pi1 >= 0.0
        assert not math.isnan(pi1)

    @pytest.mark.slow
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_pairwise_diversity(self, ts):
        self.verify_pairwise_diversity(ts)

    @pytest.mark.parametrize("order", ["abc", 0, 1, False])
    def test_bad_node_iteration_order(self, order):
        ts = tskit.TableCollection(1).tree_sequence()
        with pytest.raises(ValueError, match="order"):
            ts.nodes(order=order)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_node_iteration_order(self, ts):
        order = [n.id for n in ts.nodes()]
        assert order == list(range(ts.num_nodes))
        order = [n.id for n in ts.nodes(order="id")]
        assert order == list(range(ts.num_nodes))
        order = np.array([n.id for n in ts.nodes(order="timeasc")], dtype=int)
        assert np.all(ts.nodes_time[order] == np.sort(ts.nodes_time))
        # Check it conforms to the order of parents in the edge table
        parent_only_order = order[np.isin(order, ts.edges_parent)]
        edge_parents = np.concatenate(
            (ts.edges_parent[:-1][np.diff(ts.edges_parent) != 0], ts.edges_parent[-1:])
        )
        assert np.all(parent_only_order == edge_parents)

    def verify_edgesets(self, ts):
        """
        Verifies that the edgesets we return are equivalent to the original edges.
        """
        new_edges = []
        for edgeset in ts.edgesets():
            assert edgeset.children == sorted(edgeset.children)
            assert len(edgeset.children) > 0
            for child in edgeset.children:
                new_edges.append(
                    tskit.Edge(edgeset.left, edgeset.right, edgeset.parent, child)
                )
        # squash the edges.
        t = ts.tables.nodes.time
        new_edges.sort(key=lambda e: (t[e.parent], e.parent, e.child, e.left))

        squashed = []
        if len(new_edges) > 0:
            last_e = new_edges[0]
            for e in new_edges[1:]:
                condition = (
                    e.parent != last_e.parent
                    or e.child != last_e.child
                    or e.left != last_e.right
                )
                if condition:
                    squashed.append(last_e)
                    last_e = e
                last_e.right = e.right
            squashed.append(last_e)
            # reset the IDs
            for i, e in enumerate(squashed):
                e.id = i
        edges = list(ts.edges())
        assert len(squashed) == len(edges)
        assert edges == squashed

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_edge_ids(self, ts):
        for index, edge in enumerate(ts.edges()):
            assert edge.id == index

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_edge_span_property(self, ts):
        for edge in ts.edges():
            assert edge.span == edge.right - edge.left

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_edge_interval_property(self, ts):
        for edge in ts.edges():
            assert edge.interval == (edge.left, edge.right)
        if ts.num_trees == 1 and ts.num_edges > 0:
            for edge in ts.edges():
                assert edge.interval == ts.first().interval

    def test_edgesets(self):
        tested = False
        # We manual loop in this test to test the example tree sequences are working
        for ts in tsutil.get_example_tree_sequences(pytest_params=False):
            # Can't get edgesets with metadata
            if ts.tables.edges.metadata_schema == tskit.MetadataSchema(None):
                self.verify_edgesets(ts)
                tested = True
        assert tested

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_breakpoints(self, ts):
        breakpoints = ts.breakpoints(as_array=True)
        assert breakpoints.shape == (ts.num_trees + 1,)
        other = np.fromiter(iter([0] + [t.interval.right for t in ts.trees()]), float)
        assert np.array_equal(other, breakpoints)
        # in case downstream code has
        for j, x in enumerate(ts.breakpoints()):
            assert breakpoints[j] == x
            assert isinstance(x, float)
        assert j == ts.num_trees

    def verify_coalescence_records(self, ts):
        """
        Checks that the coalescence records we output are correct.
        """
        edgesets = list(ts.edgesets())
        records = list(ts.records())
        assert len(edgesets) == len(records)
        for edgeset, record in zip(edgesets, records):
            assert edgeset.left == record.left
            assert edgeset.right == record.right
            assert edgeset.parent == record.node
            assert edgeset.children == record.children
            parent = ts.node(edgeset.parent)
            assert parent.time == record.time
            assert parent.population == record.population

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_coalescence_records(self, ts):
        self.verify_coalescence_records(ts)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_compute_mutation_parent(self, ts):
        tables = ts.dump_tables()
        before = tables.mutations.parent[:]
        tables.compute_mutation_parents()
        parent = ts.tables.mutations.parent
        assert np.array_equal(parent, before)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_compute_mutation_time(self, ts):
        tables = ts.dump_tables()
        python_time = tsutil.compute_mutation_times(ts)
        tables.compute_mutation_times()
        assert np.allclose(python_time, tables.mutations.time, rtol=1e-10, atol=1e-10)
        # Check we have valid times
        tables.tree_sequence()

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_tracked_samples(self, ts):
        # Should be empty list by default.
        for tree in ts.trees():
            assert tree.num_tracked_samples() == 0
            for u in tree.nodes():
                assert tree.num_tracked_samples(u) == 0
        samples = list(ts.samples())
        tracked_samples = samples[:2]
        for tree in ts.trees(tracked_samples=tracked_samples):
            nu = [0 for j in range(ts.num_nodes)]
            assert tree.num_tracked_samples() == len(tracked_samples)
            for j in tracked_samples:
                u = j
                while u != tskit.NULL:
                    nu[u] += 1
                    u = tree.parent(u)
            for u, count in enumerate(nu):
                assert tree.num_tracked_samples(u) == count
            assert tree.num_tracked_samples(tree.virtual_root) == len(tracked_samples)

    def test_tracked_samples_is_first_arg(self):
        ts = tskit.Tree.generate_balanced(6).tree_sequence
        samples = [0, 1, 2]
        tree = next(ts.trees(samples))
        assert tree.num_tracked_samples() == 3

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_deprecated_sample_aliases(self, ts):
        # Ensure that we get the same results from the various combinations
        # of leaf_lists, sample_lists etc.
        samples = list(ts.samples())[:2]
        # tracked leaves/samples
        trees_new = ts.trees(tracked_samples=samples)
        trees_old = ts.trees(tracked_leaves=samples)
        for t_new, t_old in zip(trees_new, trees_old):
            for u in t_new.nodes():
                assert t_new.num_tracked_samples(u) == t_old.get_num_tracked_leaves(u)
        trees_new = ts.trees()
        trees_old = ts.trees()
        for t_new, t_old in zip(trees_new, trees_old):
            for u in t_new.nodes():
                assert t_new.num_samples(u) == t_old.get_num_leaves(u)
                assert list(t_new.samples(u)) == list(t_old.get_leaves(u))
        for on in [True, False]:
            trees_new = ts.trees(sample_lists=on)
            trees_old = ts.trees(leaf_lists=on)
            for t_new, t_old in zip(trees_new, trees_old):
                for u in t_new.nodes():
                    assert t_new.num_samples(u) == t_old.get_num_leaves(u)
                    assert list(t_new.samples(u)) == list(t_old.get_leaves(u))

    def verify_samples(self, ts):
        # We should get the same list of samples if we use the low-level
        # sample lists or a simple traversal.
        samples1 = []
        for t in ts.trees(sample_lists=False):
            samples1.append(list(t.samples()))
        samples2 = []
        for t in ts.trees(sample_lists=True):
            samples2.append(list(t.samples()))
        assert samples1 == samples2

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_samples(self, ts):
        self.verify_samples(ts)
        pops = {node.population for node in ts.nodes()}
        for pop in pops:
            subsample = ts.samples(pop)
            assert np.array_equal(subsample, ts.samples(population=pop))
            assert np.array_equal(subsample, ts.samples(population_id=pop))
            assert list(subsample) == [
                node.id
                for node in ts.nodes()
                if node.population == pop and node.is_sample()
            ]
        with pytest.raises(ValueError):
            ts.samples(population=0, population_id=0)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_first_last(self, ts):
        for kwargs in [{}, {"tracked_samples": ts.samples()}]:
            t1 = ts.first(**kwargs)
            t2 = next(ts.trees())
            assert not (t1 is t2)
            assert t1.parent_dict == t2.parent_dict
            assert t1.index == 0
            if "tracked_samples" in kwargs:
                assert t1.num_tracked_samples() == ts.num_samples
            else:
                assert t1.num_tracked_samples() == 0

            t1 = ts.last(**kwargs)
            t2 = next(reversed(ts.trees()))
            assert not (t1 is t2)
            assert t1.parent_dict == t2.parent_dict
            assert t1.index == ts.num_trees - 1
            if "tracked_samples" in kwargs:
                assert t1.num_tracked_samples() == ts.num_samples
            else:
                assert t1.num_tracked_samples() == 0

    def test_trees_interface(self):
        # Use a tree sequence guaranteed to have node 0 as the first sample node
        ts = tskit.Tree.generate_balanced(10).tree_sequence
        for t in ts.trees():
            assert t.get_num_samples(0) == 1
            assert t.get_num_tracked_samples(0) == 0
            assert list(t.samples(0)) == [0]
            assert t.tree_sequence is ts

        for t in ts.trees(tracked_samples=[0]):
            assert t.get_num_samples(0) == 1
            assert t.get_num_tracked_samples(0) == 1
            assert list(t.samples(0)) == [0]

        for t in ts.trees(sample_lists=True):
            assert t.get_num_samples(0) == 1
            assert t.get_num_tracked_samples(0) == 0
            assert list(t.samples(0)) == [0]

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_get_pairwise_diversity(self, ts):
        with pytest.raises(ValueError, match="at least one element"):
            ts.get_pairwise_diversity([])
        samples = list(ts.samples())
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="Sample sets must contain at least one element"
            ):
                ts.get_pairwise_diversity()
        else:
            assert ts.get_pairwise_diversity() == ts.get_pairwise_diversity(samples)
            assert ts.get_pairwise_diversity(samples[:2]) == ts.get_pairwise_diversity(
                list(reversed(samples[:2]))
            )

    def test_populations(self):
        more_than_zero = False
        for ts in tsutil.get_example_tree_sequences(pytest_params=False):
            N = ts.num_populations
            if N > 0:
                more_than_zero = True
            pops = list(ts.populations())
            assert len(pops) == N
            for j in range(N):
                assert pops[j] == ts.population(j)
                assert pops[j].id == j
        assert more_than_zero

    def test_individuals(self):
        more_than_zero = False
        mapped_to_nodes = False
        for ts in tsutil.get_example_tree_sequences(pytest_params=False):
            ind_node_map = collections.defaultdict(list)
            for node in ts.nodes():
                if node.individual != tskit.NULL:
                    ind_node_map[node.individual].append(node.id)
            if len(ind_node_map) > 0:
                mapped_to_nodes = True
            N = ts.num_individuals
            if N > 0:
                more_than_zero = True
            inds = list(ts.individuals())
            assert len(inds) == N
            for j in range(N):
                assert inds[j] == ts.individual(j)
                assert inds[j].id == j
                assert isinstance(inds[j].parents, np.ndarray)
                assert isinstance(inds[j].location, np.ndarray)
                assert isinstance(inds[j].nodes, np.ndarray)
                assert ind_node_map[j] == list(inds[j].nodes)

        assert more_than_zero
        assert mapped_to_nodes

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_get_population(self, ts):
        # Deprecated interface for ts.node(id).population
        N = ts.get_num_nodes()
        with pytest.raises(ValueError):
            ts.get_population(-1)
        with pytest.raises(ValueError):
            ts.get_population(N)
        with pytest.raises(ValueError):
            ts.get_population(N + 1)
        for node in range(N):
            assert ts.get_population(node) == ts.node(node).population

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_get_time(self, ts):
        # Deprecated interface for ts.node(id).time
        N = ts.get_num_nodes()
        with pytest.raises(ValueError):
            ts.get_time(-1)
        with pytest.raises(ValueError):
            ts.get_time(N)
        with pytest.raises(ValueError):
            ts.get_time(N + 1)
        for u in range(N):
            assert ts.get_time(u) == ts.node(u).time

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_max_root_time(self, ts):
        oldest = None
        for tree in ts.trees():
            for root in tree.roots:
                oldest = (
                    tree.time(root) if oldest is None else max(oldest, tree.time(root))
                )
        if oldest is None:
            assert pytest.raises(ValueError, match="max()")
        else:
            assert oldest == ts.max_root_time

    def test_max_root_time_corner_cases(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=2)
        tables.nodes.add_row(flags=0, time=3)
        assert tables.tree_sequence().max_root_time == 2
        tables.edges.add_row(0, 1, 1, 0)
        assert tables.tree_sequence().max_root_time == 2
        tables.edges.add_row(0, 1, 3, 1)
        assert tables.tree_sequence().max_root_time == 3

    def test_subset_reverse_all_nodes(self):
        ts = tskit.Tree.generate_comb(5).tree_sequence
        assert np.all(ts.samples() == np.arange(ts.num_samples))
        flipped_ids = np.flip(np.arange(ts.num_nodes))
        new_ts = ts.subset(flipped_ids)
        assert set(new_ts.samples()) == set(flipped_ids[np.arange(ts.num_samples)])
        r1 = ts.first().rank()
        r2 = new_ts.first().rank()
        assert r1.shape == r2.shape
        assert r1.label != r2.label

    def test_subset_reverse_internal_nodes(self):
        ts = tskit.Tree.generate_balanced(5).tree_sequence
        internal_nodes = np.ones(ts.num_nodes, dtype=bool)
        internal_nodes[ts.samples()] = False
        node_ids = np.arange(ts.num_nodes)
        node_ids[internal_nodes] = np.flip(node_ids[internal_nodes])
        new_ts = ts.subset(node_ids)
        assert np.any(new_ts.nodes_time != ts.nodes_time)
        assert new_ts.first().rank() == ts.first().rank()

    def test_deprecated_apis(self):
        ts = msprime.simulate(10, random_seed=1)
        assert ts.get_ll_tree_sequence() == ts.ll_tree_sequence
        assert ts.get_sample_size() == ts.sample_size
        assert ts.get_sample_size() == ts.num_samples
        assert ts.get_sequence_length() == ts.sequence_length
        assert ts.get_num_trees() == ts.num_trees
        assert ts.get_num_mutations() == ts.num_mutations
        assert ts.get_num_nodes() == ts.num_nodes
        assert ts.get_pairwise_diversity() == ts.pairwise_diversity()
        samples = ts.samples()
        assert ts.get_pairwise_diversity(samples) == ts.pairwise_diversity(samples)
        assert np.array_equal(ts.get_samples(), ts.samples())

    def test_sites(self):
        some_sites = False
        for ts in tsutil.get_example_tree_sequences(pytest_params=False):
            tables = ts.dump_tables()
            sites = tables.sites
            mutations = tables.mutations
            assert ts.num_sites == len(sites)
            assert ts.num_mutations == len(mutations)
            previous_pos = -1
            mutation_index = 0
            ancestral_state = tskit.unpack_strings(
                sites.ancestral_state, sites.ancestral_state_offset
            )
            derived_state = tskit.unpack_strings(
                mutations.derived_state, mutations.derived_state_offset
            )

            for index, site in enumerate(ts.sites()):
                s2 = ts.site(site.id)
                assert s2 == site
                s3 = ts.site(position=site.position)
                assert s3 == site
                assert site.position == sites.position[index]
                assert site.position > previous_pos
                previous_pos = site.position
                assert ancestral_state[index] == site.ancestral_state
                assert site.id == index
                for mutation in site.mutations:
                    m2 = ts.mutation(mutation.id)
                    assert m2 == mutation
                    assert mutation.site == site.id
                    assert mutation.site == mutations.site[mutation_index]
                    assert mutation.node == mutations.node[mutation_index]
                    assert mutation.parent == mutations.parent[mutation_index]
                    assert mutation.id == mutation_index
                    assert derived_state[mutation_index] == mutation.derived_state
                    mutation_index += 1
                some_sites = True
            total_sites = 0
            for tree in ts.trees():
                assert len(list(tree.sites())) == tree.num_sites
                total_sites += tree.num_sites
            assert ts.num_sites == total_sites
            assert mutation_index == len(mutations)
        assert some_sites

    def verify_mutations(self, ts):
        other_mutations = []
        for site in ts.sites():
            for mutation in site.mutations:
                other_mutations.append(mutation)
        mutations = list(ts.mutations())
        assert ts.num_mutations == len(other_mutations)
        assert ts.num_mutations == len(mutations)
        for mut, other_mut in zip(mutations, other_mutations):
            assert mut == other_mut

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_sites_mutations(self, ts):
        # Check that the mutations iterator returns the correct values.
        self.verify_mutations(ts)

    def test_removed_methods(self):
        ts = tskit.TableCollection(1).tree_sequence()
        with pytest.raises(NotImplementedError):
            ts.get_num_records()
        with pytest.raises(NotImplementedError):
            ts.diffs()
        with pytest.raises(NotImplementedError):
            ts.newick_trees()
        with pytest.raises(NotImplementedError):
            ts.to_nexus()

    def test_dump_pathlib(self, ts_fixture, tmp_path):
        path = tmp_path / "tmp.trees"
        assert path.exists
        assert path.is_file
        ts_fixture.dump(path)
        other_ts = tskit.load(path)
        assert ts_fixture.tables == other_ts.tables

    @pytest.mark.skipif(platform.system() == "Windows", reason="Windows doesn't raise")
    def test_dump_load_errors(self):
        ts = msprime.simulate(5, random_seed=1)
        # Try to dump/load files we don't have access to or don't exist.
        for func in [ts.dump, tskit.load]:
            for f in ["/", "/test.trees", "/dir_does_not_exist/x.trees"]:
                with pytest.raises(OSError):
                    func(f)
                try:
                    func(f)
                except OSError as e:
                    message = str(e)
                    assert len(message) > 0
            f = "/" + 4000 * "x"
            with pytest.raises(OSError):
                func(f)
            try:
                func(f)
            except OSError as e:
                message = str(e)
            assert "File name too long" in message
            for bad_filename in [[], None, {}]:
                with pytest.raises(TypeError):
                    func(bad_filename)

    def test_zlib_compression_warning(self, ts_fixture, tmp_path):
        temp_file = tmp_path / "tmp.trees"
        with warnings.catch_warnings(record=True) as w:
            ts_fixture.dump(temp_file, zlib_compression=True)
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
        with warnings.catch_warnings(record=True) as w:
            ts_fixture.dump(temp_file, zlib_compression=False)
            assert len(w) == 0

    def test_tables_sequence_length_round_trip(self):
        for sequence_length in [0.1, 1, 10, 100]:
            ts = msprime.simulate(5, length=sequence_length, random_seed=1)
            assert ts.sequence_length == sequence_length
            tables = ts.tables
            assert tables.sequence_length == sequence_length
            new_ts = tables.tree_sequence()
            assert new_ts.sequence_length == sequence_length

    def test_migrations(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(10),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            random_seed=2,
            record_migrations=True,
        )
        assert ts.num_migrations > 0
        migrations = list(ts.migrations())
        assert len(migrations) == ts.num_migrations
        for migration in migrations:
            assert migration.source in [0, 1]
            assert migration.dest in [0, 1]
            assert migration.time > 0
            assert migration.left == 0
            assert migration.right == 1
            assert 0 <= migration.node < ts.num_nodes

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_len_trees(self, ts):
        tree_iter = ts.trees()
        assert len(tree_iter) == ts.num_trees

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_list(self, ts):
        for kwargs in [{}, {"tracked_samples": ts.samples()}]:
            tree_list = ts.aslist(**kwargs)
            assert len(tree_list) == ts.num_trees
            assert len(set(map(id, tree_list))) == ts.num_trees
            for index, tree in enumerate(tree_list):
                assert index == tree.index
            for t1, t2 in zip(tree_list, ts.trees(**kwargs)):
                assert t1 == t2
                assert t1.parent_dict == t2.parent_dict
                if "tracked_samples" in kwargs:
                    assert t1.num_tracked_samples() == ts.num_samples
                    assert t2.num_tracked_samples() == ts.num_samples
                else:
                    assert t1.num_tracked_samples() == 0
                    assert t2.num_tracked_samples() == 0

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_reversed_trees(self, ts):
        index = ts.num_trees - 1
        tree_list = ts.aslist()
        for tree in reversed(ts.trees()):
            assert tree.index == index
            t2 = tree_list[index]
            assert tree.interval == t2.interval
            assert tree.parent_dict == t2.parent_dict
            index -= 1

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_at_index(self, ts):
        for kwargs in [{}, {"tracked_samples": ts.samples()}]:
            tree_list = ts.aslist(**kwargs)
            for index in list(range(ts.num_trees)) + [-1]:
                t1 = tree_list[index]
                t2 = ts.at_index(index, **kwargs)
                assert t1 == t2
                assert t1.interval == t2.interval
                assert t1.parent_dict == t2.parent_dict
                if "tracked_samples" in kwargs:
                    assert t2.num_tracked_samples() == ts.num_samples
                else:
                    assert t2.num_tracked_samples() == 0

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_at(self, ts):
        for kwargs in [{}, {"tracked_samples": ts.samples()}]:
            tree_list = ts.aslist(**kwargs)
            for t1 in tree_list:
                left, right = t1.interval
                mid = left + (right - left) / 2
                for pos in [left, left + 1e-9, mid, right - 1e-9]:
                    t2 = ts.at(pos, **kwargs)
                    assert t1 == t2
                    assert t1.interval == t2.interval
                    assert t1.parent_dict == t2.parent_dict
                if right < ts.sequence_length:
                    t2 = ts.at(right, **kwargs)
                    t3 = tree_list[t1.index + 1]
                    assert t3 == t2
                    assert t3.interval == t2.interval
                    assert t3.parent_dict == t2.parent_dict
                if "tracked_samples" in kwargs:
                    assert t2.num_tracked_samples() == ts.num_samples
                else:
                    assert t2.num_tracked_samples() == 0

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_sequence_iteration(self, ts):
        for table_name in ts.tables_dict.keys():
            sequence = getattr(ts, table_name)()
            length = getattr(ts, "num_" + table_name)
            # Test __iter__
            i = None
            for i, n in enumerate(sequence):
                assert i == n.id
            if i is not None:
                assert n.id == (length - 1 if length else 0)
            if table_name == "mutations":
                # Mutations are not currently sequences, so have no len or idx access
                with pytest.raises(TypeError):
                    len(sequence)
                if length != 0:
                    with pytest.raises(TypeError):
                        sequence[0]
            else:
                # Test __len__
                assert len(sequence) == length
                # Test __getitem__ on the last item in the sequence
                if length != 0:
                    assert sequence[length - 1] == n  # +ive indexing
                    assert sequence[-1] == n  # -ive indexing
                with pytest.raises(IndexError):
                    sequence[length]
                # Test reverse
                i = None
                for i, n in enumerate(reversed(sequence)):
                    assert i == length - 1 - n.id
                if i is not None:
                    assert n.id == 0

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_load_tables(self, ts):
        tables = ts.dump_tables()
        tables.drop_index()

        # Tables not in tc not rebuilt as per default, so error
        with pytest.raises(
            _tskit.LibraryError, match="Table collection must be indexed"
        ):
            assert tskit.TreeSequence.load_tables(tables).tables.has_index()

        # Tables not in tc, but rebuilt
        assert tskit.TreeSequence.load_tables(
            tables, build_indexes=True
        ).tables.has_index()

        tables.build_index()
        # Tables in tc, not rebuilt
        assert tskit.TreeSequence.load_tables(
            tables, build_indexes=False
        ).tables.has_index()
        # Tables in tc, and rebuilt
        assert tskit.TreeSequence.load_tables(tables).tables.has_index()

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_html_repr(self, ts):
        html = ts._repr_html_()
        # Parse to check valid
        ElementTree.fromstring(html)
        assert len(html) > 5000
        assert f"<tr><td>Trees</td><td>{ts.num_trees:,}</td></tr>" in html
        assert f"<tr><td>Time Units</td><td>{ts.time_units}</td></tr>" in html
        for table in ts.tables.table_name_map:
            assert f"<td>{table.capitalize()}</td>" in html
        if ts.num_provenances > 0:
            assert (
                f"<td>{json.loads(ts.provenance(0).record)['software']['name']}</td>"
                in html
            )

    def test_bad_provenance(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        tables.provenances.add_row("bad", "bad")
        ts = tables.tree_sequence()
        assert "Could not parse provenance" in ts._repr_html_()

    def test_provenance_summary_html(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        for _ in range(20):
            # Add a row with isotimestamp
            tables.provenances.add_row("foo", "bar")
        assert "... 15 more" in tables.tree_sequence()._repr_html_()

    def test_html_repr_limit(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        d = {n: n for n in range(50)}
        d[0] = "N" * 200
        tables.metadata = d
        ts = tables.tree_sequence()
        assert "... and 20 more" in ts._repr_html_()
        assert "NN..." in ts._repr_html_()

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_str(self, ts):
        s = str(ts)
        assert len(s) > 999
        assert re.search(rf"║Trees *│ *{ts.num_trees}║", s)
        assert re.search(rf"║Time Units *│ *{ts.time_units}║", s)
        for table in ts.tables.table_name_map:
            assert re.search(rf"║{table.capitalize()} *│", s)

    @pytest.mark.skip("FIXME nbytes")
    def test_nbytes(self, tmp_path, ts_fixture):
        ts_fixture.dump(tmp_path / "tables")
        store = kastore.load(tmp_path / "tables")
        for v in store.values():
            # Check we really have data in every field
            assert v.nbytes > 0
        nbytes = sum(
            array.nbytes
            for name, array in store.items()
            # nbytes is the size of asdict, so exclude file format items
            if name not in ["format/version", "format/name", "uuid"]
        )
        assert nbytes == ts_fixture.nbytes

    def test_equals(self):
        # Here we don't use the fixture as we'd like to run the same sim twice
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        t1 = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        )
        t2 = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        )

        assert t1 == t1
        assert t1 == t1.dump_tables().tree_sequence()
        assert t1.dump_tables().tree_sequence() == t1

        # The provenances may or may not be equal depending on the clock
        # precision for record. So clear them first.
        tb1 = t1.dump_tables()
        tb2 = t2.dump_tables()
        tb1.provenances.clear()
        tb2.provenances.clear()
        t1 = tb1.tree_sequence()
        t2 = tb2.tree_sequence()

        assert t1 == t2
        assert t1 == t2
        assert not (t1 != t2)
        # We don't do more as this is the same code path as TableCollection.__eq__

    def test_equals_options(self, ts_fixture):
        t1 = ts_fixture
        # Take a copy
        t2 = ts_fixture.dump_tables().tree_sequence()

        def modify(ts, func):
            tc = ts.dump_tables()
            func(tc)
            return tc.tree_sequence()

        t1 = modify(t1, lambda tc: tc.provenances.add_row("random stuff"))
        assert not (t1 == t2)
        assert t1.equals(t2, ignore_provenance=True)
        assert t2.equals(t1, ignore_provenance=True)
        assert not (t1.equals(t2))
        assert not (t2.equals(t1))
        t1 = modify(t1, lambda tc: tc.provenances.clear())
        t2 = modify(t2, lambda tc: tc.provenances.clear())
        assert t1.equals(t2)
        assert t2.equals(t1)

        tc = t1.dump_tables()
        tc.metadata_schema = tskit.MetadataSchema({"codec": "json", "type": "object"})
        t1 = tc.tree_sequence()
        tc = t1.dump_tables()
        tc.metadata = {"hello": "world"}
        t1 = tc.tree_sequence()

        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_ts_metadata=True)
        assert not t2.equals(t1)
        assert t2.equals(t1, ignore_ts_metadata=True)
        tc = t2.dump_tables()
        tc.metadata_schema = t1.metadata_schema
        t2 = tc.tree_sequence()
        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_ts_metadata=True)
        assert not t2.equals(t1)
        assert t2.equals(t1, ignore_ts_metadata=True)

        t1 = modify(t1, lambda tc: tc.provenances.add_row("random stuff"))
        assert not t1.equals(t2)
        assert not t1.equals(t2, ignore_ts_metadata=True)
        assert not t1.equals(t2, ignore_provenance=True)
        assert t1.equals(t2, ignore_ts_metadata=True, ignore_provenance=True)

        t1 = modify(t1, lambda tc: tc.provenances.clear())
        t2 = modify(t2, lambda tc: setattr(tc, "metadata", t1.metadata))  # noqa: B010
        assert t1.equals(t2)
        assert t2.equals(t1)

        # Empty out tables to test ignore_tables flag
        tc = t2.dump_tables()
        tc.individuals.truncate(0)
        tc.nodes.truncate(0)
        tc.edges.truncate(0)
        tc.migrations.truncate(0)
        tc.sites.truncate(0)
        tc.mutations.truncate(0)
        tc.populations.truncate(0)
        t2 = tc.tree_sequence()
        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_tables=True)
        # Empty out reference to test ignore_reference_sequence flag
        tc = t1.dump_tables()
        tc.reference_sequence.clear()
        t2 = tc.tree_sequence()
        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_reference_sequence=True)
        # Make t1 and t2 equal again
        t2 = t1.dump_tables().tree_sequence()
        assert t1.equals(t2)
        assert t2.equals(t1)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_tree_node_edges(self, ts):
        edge_visited = np.zeros(ts.num_edges, dtype=bool)
        for tree in ts.trees():
            mapping = tree.edge_array
            node_mapped = mapping >= 0
            edge_visited[mapping[node_mapped]] = True
            # Note that tree.nodes() does not necessarily list all the nodes
            # in the tree topology, only the ones that descend from a root.
            # Therefore if not all the topological trees in a single `Tree` have
            # a root, we can have edges above nodes that are not listed. This
            # happens, for example, in a tree with no sample nodes.
            assert np.sum(node_mapped) >= len(list(tree.nodes())) - tree.num_roots
            for u in tree.nodes():
                if tree.parent(u) == tskit.NULL:
                    assert mapping[u] == tskit.NULL
                else:
                    edge = ts.edge(mapping[u])
                    assert edge.child == u
                    assert edge.left <= tree.interval.left
                    assert edge.right >= tree.interval.right
        assert np.all(edge_visited)

    def verify_individual_vectors(self, ts):
        verify_times = np.repeat(np.nan, ts.num_individuals)
        verify_populations = np.repeat(tskit.NULL, ts.num_individuals)
        for ind in ts.individuals():
            if len(ind.nodes) > 0:
                t = {ts.node(n).time for n in ind.nodes}
                p = {ts.node(n).population for n in ind.nodes}
                assert len(t) <= 1
                assert len(p) <= 1
                verify_times[ind.id] = t.pop()
                verify_populations[ind.id] = p.pop()

        times = ts.individuals_time
        populations = ts.individuals_population
        assert np.array_equal(times, verify_times, equal_nan=True)
        assert np.array_equal(populations, verify_populations, equal_nan=True)
        times2 = ts.individuals_time
        populations2 = ts.individuals_population
        assert np.array_equal(times, times2, equal_nan=True)
        assert np.array_equal(populations, populations2, equal_nan=True)
        # check aliases also
        times3 = ts.individual_times
        populations3 = ts.individual_populations
        assert np.array_equal(times, times3, equal_nan=True)
        assert np.array_equal(populations, populations3, equal_nan=True)

    def test_individuals_population_errors(self):
        t = tskit.TableCollection(sequence_length=1)
        t.individuals.add_row()
        t.individuals.add_row()
        for j in range(2):
            t.populations.add_row()
            t.nodes.add_row(time=0, population=j, individual=0)
        ts = t.tree_sequence()
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INDIVIDUAL_POPULATION_MISMATCH"
        ):
            _ = ts.individuals_population
        # inconsistent but NULL populations are also an error
        t.nodes.clear()
        t.nodes.add_row(time=0, population=1, individual=0)
        t.nodes.add_row(time=0, population=tskit.NULL, individual=0)
        ts = t.tree_sequence()
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INDIVIDUAL_POPULATION_MISMATCH"
        ):
            _ = ts.individuals_population
        t.nodes.clear()
        t.nodes.add_row(time=0, population=tskit.NULL, individual=1)
        t.nodes.add_row(time=0, population=0, individual=1)
        ts = t.tree_sequence()
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INDIVIDUAL_POPULATION_MISMATCH"
        ):
            _ = ts.individuals_population

    def test_individuals_time_errors(self):
        t = tskit.TableCollection(sequence_length=1)
        t.individuals.add_row()
        for j in range(2):
            t.nodes.add_row(time=j, individual=0)
        ts = t.tree_sequence()
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INDIVIDUAL_TIME_MISMATCH"
        ):
            _ = ts.individuals_time

    @pytest.mark.parametrize("n", [1, 10])
    def test_individual_vectors(self, n):
        d = msprime.Demography.island_model([10] * n, 0.1)
        ts = msprime.sim_ancestry(
            {pop.name: 10 for pop in d.populations},
            demography=d,
            random_seed=100 + n,
            model="dtwf",
        )
        ts = tsutil.insert_random_consistent_individuals(ts, seed=100 + n)
        assert ts.num_individuals > 10
        self.verify_individual_vectors(ts)

    def test_individuals_location_errors(self):
        t = tskit.TableCollection(sequence_length=1)
        t.individuals.add_row(location=[1.0, 2.0])
        t.individuals.add_row(location=[0.0])
        ts = t.tree_sequence()
        with pytest.raises(ValueError, match="locations"):
            _ = ts.individuals_location

        t.clear()
        t.individuals.add_row(location=[1.0, 2.0])
        t.individuals.add_row(location=[])
        t.individuals.add_row(location=[1.0, 2.0])
        t.individuals.add_row(location=[])
        ts = t.tree_sequence()
        with pytest.raises(ValueError, match="locations"):
            _ = ts.individuals_location

    @pytest.mark.parametrize("nlocs", [0, 1, 4])
    @pytest.mark.parametrize("num_indivs", [0, 3])
    def test_individuals_location(self, nlocs, num_indivs):
        t = tskit.TableCollection(sequence_length=1)
        locs = np.array([j + np.arange(nlocs) for j in range(num_indivs)])
        if len(locs) == 0:
            locs = locs.reshape((num_indivs, 0))
        for j in range(num_indivs):
            t.individuals.add_row(location=locs[j])
        ts = t.tree_sequence()
        ts_locs = ts.individuals_location
        assert locs.shape == ts_locs.shape
        assert np.array_equal(locs, ts_locs)
        locs2 = ts.individuals_location
        assert np.array_equal(ts_locs, locs2)
        # test alias
        locs3 = ts.individual_locations
        assert np.array_equal(ts_locs, locs3)

    def verify_individual_properties(self, ts):
        for ind in ts.individuals():
            times = [ts.node(n).time for n in ind.nodes]
            if len(set(times)) > 1:
                with pytest.raises(ValueError, match="mis-matched times"):
                    _ = ind.time
            elif len(times) == 0:
                assert tskit.is_unknown_time(ind.time)
            else:
                assert len(set(times)) == 1
                assert times[0] == ind.time
                # test accessing more than once in case we mess up with {}.pop()
                assert times[0] == ind.time
            pops = [ts.node(n).population for n in ind.nodes]
            if len(set(pops)) > 1:
                with pytest.raises(ValueError, match="mis-matched populations"):
                    _ = ind.population
            elif len(pops) == 0:
                assert ind.population is tskit.NULL
            else:
                assert len(set(pops)) == 1
                assert ind.population == pops[0]
                # test accessing more than once in case we mess up with {}.pop()
                assert ind.population == pops[0]

    def test_individual_getter_population(self):
        tables = tskit.TableCollection(sequence_length=1)
        for _ in range(2):
            tables.populations.add_row()
        pop_list = [
            ((), tskit.NULL),
            ((tskit.NULL,), tskit.NULL),
            ((1,), 1),
            ((1, 1, 1), 1),
            ((tskit.NULL, 1), "ERR"),
            ((0, tskit.NULL), "ERR"),
            ((0, 1), "ERR"),
        ]
        for pops, _ in pop_list:
            j = tables.individuals.add_row()
            for p in pops:
                tables.nodes.add_row(time=0, population=p, individual=j)
        ts = tables.tree_sequence()
        for ind, (_, p) in zip(ts.individuals(), pop_list):
            if p == "ERR":
                with pytest.raises(ValueError, match="mis-matched populations"):
                    _ = ind.population
            else:
                assert p == ind.population

    def test_individual_getter_time(self):
        tables = tskit.TableCollection(sequence_length=1)
        time_list = [
            ((), tskit.UNKNOWN_TIME),
            ((0.0,), 0.0),
            ((1, 1, 1), 1),
            ((4.0, 1), "ERR"),
            ((0, 4.0), "ERR"),
        ]
        for times, _ in time_list:
            j = tables.individuals.add_row()
            for t in times:
                tables.nodes.add_row(time=t, individual=j)
        ts = tables.tree_sequence()
        for ind, (_, t) in zip(ts.individuals(), time_list):
            if t == "ERR":
                with pytest.raises(ValueError, match="mis-matched times"):
                    _ = ind.time
            elif tskit.is_unknown_time(t):
                assert tskit.is_unknown_time(ind.time)
            else:
                assert t == ind.time

    @pytest.mark.parametrize("n", [1, 10])
    def test_individual_properties(self, n):
        # tests for the .time and .population attributes of
        # the Individual class
        d = msprime.Demography.island_model([10] * n, 0.1)
        ts = msprime.sim_ancestry(
            {pop.name: int(150 / n) for pop in d.populations},
            demography=d,
            random_seed=100 + n,
            model="dtwf",
        )
        ts = tsutil.insert_random_consistent_individuals(ts, seed=100 + n)
        assert ts.num_individuals > 10
        self.verify_individual_properties(ts)
        ts = tsutil.insert_random_ploidy_individuals(ts, seed=100 + n)
        assert ts.num_individuals > 10
        self.verify_individual_properties(ts)

    @pytest.mark.parametrize(
        "array",
        [
            "individuals_flags",
            "nodes_time",
            "nodes_flags",
            "nodes_population",
            "nodes_individual",
            "edges_left",
            "edges_right",
            "edges_parent",
            "edges_child",
            "sites_position",
            "mutations_site",
            "mutations_node",
            "mutations_parent",
            "mutations_time",
            "migrations_left",
            "migrations_right",
            "migrations_node",
            "migrations_source",
            "migrations_dest",
            "migrations_time",
            "indexes_edge_insertion_order",
            "indexes_edge_removal_order",
        ],
    )
    def test_array_attr_properties(self, ts_fixture, array):
        ts = ts_fixture
        a = getattr(ts, array)
        assert isinstance(a, np.ndarray)
        with pytest.raises(AttributeError):
            setattr(ts, array, None)
        with pytest.raises(AttributeError):
            delattr(ts, array)
        with pytest.raises(ValueError, match="read-only"):
            a[:] = 1

    def test_arrays_equal_to_tables(self, ts_fixture):
        ts = ts_fixture
        tables = ts.tables

        assert_array_equal(ts.individuals_flags, tables.individuals.flags)

        assert_array_equal(ts.nodes_flags, tables.nodes.flags)
        assert_array_equal(ts.nodes_population, tables.nodes.population)
        assert_array_equal(ts.nodes_time, tables.nodes.time)
        assert_array_equal(ts.nodes_individual, tables.nodes.individual)

        assert_array_equal(ts.edges_left, tables.edges.left)
        assert_array_equal(ts.edges_right, tables.edges.right)
        assert_array_equal(ts.edges_parent, tables.edges.parent)
        assert_array_equal(ts.edges_child, tables.edges.child)

        assert_array_equal(ts.sites_position, tables.sites.position)

        assert_array_equal(ts.mutations_site, tables.mutations.site)
        assert_array_equal(ts.mutations_node, tables.mutations.node)
        assert_array_equal(ts.mutations_parent, tables.mutations.parent)
        assert_array_equal(ts.mutations_time, tables.mutations.time)

        assert_array_equal(ts.migrations_left, tables.migrations.left)
        assert_array_equal(ts.migrations_right, tables.migrations.right)
        assert_array_equal(ts.migrations_node, tables.migrations.node)
        assert_array_equal(ts.migrations_source, tables.migrations.source)
        assert_array_equal(ts.migrations_dest, tables.migrations.dest)
        assert_array_equal(ts.migrations_time, tables.migrations.time)

        assert_array_equal(
            ts.indexes_edge_insertion_order, tables.indexes.edge_insertion_order
        )
        assert_array_equal(
            ts.indexes_edge_removal_order, tables.indexes.edge_removal_order
        )

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_impute_unknown_mutations_time(self, ts):
        # Tests for method='min'
        imputed_time = ts.impute_unknown_mutations_time(method="min")
        mutations = ts.tables.mutations
        nodes_time = ts.nodes_time
        table_time = np.zeros(len(mutations))

        for mut_idx, mut in enumerate(mutations):
            if tskit.is_unknown_time(mut.time):
                node_time = nodes_time[mut.node]
                table_time[mut_idx] = node_time
            else:
                table_time[mut_idx] = mut.time

        assert np.allclose(imputed_time, table_time, rtol=1e-10, atol=1e-10)

        # Check we have valid times
        tables = ts.dump_tables()
        tables.mutations.time = imputed_time
        tables.sort()
        tables.tree_sequence()

        # Test for unallowed methods
        with pytest.raises(
            ValueError, match="Mutations time imputation method must be chosen"
        ):
            ts.impute_unknown_mutations_time(method="foobar")

    @pytest.mark.parametrize(
        "mutations, error",
        [
            ([], None),
            (
                [{"node": 0, "parent": -1}, {"node": 1, "parent": -1}],
                None,
            ),  # On parallel branches, no parents
            (
                [
                    {"node": 4, "parent": -1},
                    {"node": 0, "parent": 0},
                    {"node": 1, "parent": 0},
                ],
                None,
            ),  # On parallel branches, legal parent
            (
                [{"node": 0, "parent": -1}, {"node": 0, "parent": 0}],
                None,
            ),  # On same node
            (
                [{"node": 0, "parent": -1}, {"node": 0, "parent": -1}],
                "not consistent with the topology",
            ),  # On same node without parents
            (
                [
                    {"node": 3, "parent": -1},
                    {"node": 0, "parent": 0},
                    {"node": 1, "parent": 0},
                ],
                "not consistent with the topology",
            ),  # On parallel branches, parent on parallel branches
            (
                [
                    {"node": 5, "parent": -1},
                    {"node": 0, "parent": 0},
                    {"node": 1, "parent": 0},
                ],
                "not consistent with the topology",
            ),  # On parallel branches, parent high on parallel
            (
                [
                    {"node": 3, "parent": -1},
                    {"node": 0, "parent": 0},
                    {"node": 7, "parent": 0},
                ],
                "not consistent with the topology",
            ),  # On parallel branches, parent on different root
            (
                [
                    {"node": 0, "parent": -1},
                    {"node": 1, "parent": 0},
                ],
                "not consistent with the topology",
            ),  # parent on parallel branch
            (
                [
                    {"node": 6, "parent": -1},
                    {"node": 6, "parent": 0},
                ],
                None,
            ),  # parent above root
            (
                [
                    {"node": 6, "parent": -1},
                    {"node": 6, "parent": -1},
                ],
                "not consistent with the topology",
            ),  # parent above root, no parents
        ],
    )
    def test_mutation_parent_errors(self, mutations, error):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=2)
        tables.nodes.add_row(time=3)
        tables.edges.add_row(left=0, right=1, parent=4, child=0)
        tables.edges.add_row(left=0, right=1, parent=4, child=1)
        tables.edges.add_row(left=0, right=1, parent=5, child=2)
        tables.edges.add_row(left=0, right=1, parent=5, child=3)
        tables.edges.add_row(left=0, right=1, parent=6, child=4)
        tables.edges.add_row(left=0, right=1, parent=6, child=5)
        tables.sites.add_row(position=0.5, ancestral_state="A")

        for mut in mutations:
            tables.mutations.add_row(**{"derived_state": "G", "site": 0, **mut})

        if error is not None:
            with pytest.raises(_tskit.LibraryError, match=error):
                tables.tree_sequence()
        else:
            tables.tree_sequence()


class TestSimplify:
    # This class was factored out of the old TestHighlevel class 2022-12-13,
    # and is a mishmash of different testing paradigms. There is some valuable
    # testing done here, so it would be good to fully bring it up to date.

    def verify_simplify_provenance(self, ts):
        new_ts = ts.simplify()
        assert new_ts.num_provenances == ts.num_provenances + 1
        old = list(ts.provenances())
        new = list(new_ts.provenances())
        assert old == new[:-1]
        # TODO call verify_provenance on this.
        assert len(new[-1].timestamp) > 0
        assert len(new[-1].record) > 0

        new_ts = ts.simplify(record_provenance=False)
        assert new_ts.tables.provenances == ts.tables.provenances

    def verify_simplify_topology(self, ts, sample):
        new_ts, node_map = ts.simplify(sample, map_nodes=True)
        if len(sample) == 0:
            assert new_ts.num_nodes == 0
            assert new_ts.num_edges == 0
            assert new_ts.num_sites == 0
            assert new_ts.num_mutations == 0
        elif len(sample) == 1:
            assert new_ts.num_nodes == 1
            assert new_ts.num_edges == 0
        # The output samples should be 0...n
        assert new_ts.num_samples == len(sample)
        assert list(range(len(sample))) == list(new_ts.samples())
        for j in range(new_ts.num_samples):
            assert node_map[sample[j]] == j
        for u in range(ts.num_nodes):
            old_node = ts.node(u)
            if node_map[u] != tskit.NULL:
                new_node = new_ts.node(node_map[u])
                assert old_node.time == new_node.time
                assert old_node.population == new_node.population
                assert old_node.metadata == new_node.metadata
        for u in sample:
            old_node = ts.node(u)
            new_node = new_ts.node(node_map[u])
            assert old_node.flags == new_node.flags
            assert old_node.time == new_node.time
            assert old_node.population == new_node.population
            assert old_node.metadata == new_node.metadata
        old_trees = ts.trees()
        old_tree = next(old_trees)
        assert ts.get_num_trees() >= new_ts.get_num_trees()
        for new_tree in new_ts.trees():
            new_left, new_right = new_tree.get_interval()
            old_left, old_right = old_tree.get_interval()
            # Skip ahead on the old tree until new_left is within its interval
            while old_right <= new_left:
                old_tree = next(old_trees)
                old_left, old_right = old_tree.get_interval()
            # If the MRCA of all pairs of samples is the same, then we have the
            # same information. We limit this to at most 500 pairs
            pairs = itertools.islice(itertools.combinations(sample, 2), 500)
            for pair in pairs:
                mapped_pair = [node_map[u] for u in pair]
                mrca1 = old_tree.get_mrca(*pair)
                mrca2 = new_tree.get_mrca(*mapped_pair)
                if mrca1 == tskit.NULL:
                    assert mrca2 == mrca1
                else:
                    assert mrca2 == node_map[mrca1]
                    assert old_tree.get_time(mrca1) == new_tree.get_time(mrca2)
                    assert old_tree.get_population(mrca1) == new_tree.get_population(
                        mrca2
                    )

    def verify_simplify_equality(self, ts, sample):
        for filter_sites in [False, True]:
            s1, node_map1 = ts.simplify(
                sample, map_nodes=True, filter_sites=filter_sites
            )
            t1 = s1.dump_tables()
            s2, node_map2 = simplify_tree_sequence(
                ts, sample, filter_sites=filter_sites
            )
            t2 = s2.dump_tables()
            assert s1.num_samples == len(sample)
            assert s2.num_samples == len(sample)
            assert all(node_map1 == node_map2)
            assert t1.individuals == t2.individuals
            assert t1.nodes == t2.nodes
            assert t1.edges == t2.edges
            assert t1.migrations == t2.migrations
            assert t1.sites == t2.sites
            assert t1.mutations == t2.mutations
            assert t1.populations == t2.populations

    def verify_simplify_variants(self, ts, sample):
        subset = ts.simplify(sample)
        sample_map = {u: j for j, u in enumerate(ts.samples())}
        # Need to map IDs back to their sample indexes
        s = np.array([sample_map[u] for u in sample])
        # Build a map of genotypes by position
        full_genotypes = {}
        for variant in ts.variants(isolated_as_missing=False):
            alleles = [variant.alleles[g] for g in variant.genotypes]
            full_genotypes[variant.position] = alleles
        for variant in subset.variants(isolated_as_missing=False):
            if variant.position in full_genotypes:
                a1 = [full_genotypes[variant.position][u] for u in s]
                a2 = [variant.alleles[g] for g in variant.genotypes]
                assert a1 == a2

    def verify_tables_api_equality(self, ts):
        for samples in [None, list(ts.samples()), ts.samples()]:
            tables = ts.dump_tables()
            tables.simplify(samples=samples)
            tables.assert_equals(
                ts.simplify(samples=samples).dump_tables(),
                ignore_timestamps=True,
            )

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_simplify_tables_equality(self, ts):
        # Can't simplify edges with metadata
        if ts.tables.edges.metadata_schema == tskit.MetadataSchema(schema=None):
            self.verify_tables_api_equality(ts)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_simplify_provenance(self, ts):
        # Can't simplify edges with metadata
        if ts.tables.edges.metadata_schema == tskit.MetadataSchema(schema=None):
            self.verify_simplify_provenance(ts)

    # TODO this test needs to be broken up into discrete bits, so that we can
    # test them independently. A way of getting a random-ish subset of samples
    # from the pytest param would be useful.
    @pytest.mark.slow
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_simplify(self, ts):
        # Can't simplify edges with metadata
        if ts.tables.edges.metadata_schema == tskit.MetadataSchema(schema=None):
            n = ts.num_samples
            sample_sizes = {0}
            if n > 1:
                sample_sizes |= {1}
            if n > 2:
                sample_sizes |= {2, max(2, n // 2), n - 1}
            for k in sample_sizes:
                subset = random.sample(list(ts.samples()), k)
                self.verify_simplify_topology(ts, subset)
                self.verify_simplify_equality(ts, subset)
                self.verify_simplify_variants(ts, subset)

    def test_simplify_bugs(self):
        prefix = os.path.join(os.path.dirname(__file__), "data", "simplify-bugs")
        j = 1
        while True:
            nodes_file = os.path.join(prefix, f"{j:02d}-nodes.txt")
            if not os.path.exists(nodes_file):
                break
            edges_file = os.path.join(prefix, f"{j:02d}-edges.txt")
            sites_file = os.path.join(prefix, f"{j:02d}-sites.txt")
            mutations_file = os.path.join(prefix, f"{j:02d}-mutations.txt")
            with open(nodes_file) as nodes, open(edges_file) as edges, open(
                sites_file
            ) as sites, open(mutations_file) as mutations:
                ts = tskit.load_text(
                    nodes=nodes,
                    edges=edges,
                    sites=sites,
                    mutations=mutations,
                    strict=False,
                )
            samples = list(ts.samples())
            self.verify_simplify_equality(ts, samples)
            j += 1
        assert j > 1

    def test_simplify_migrations_fails(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(10),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            random_seed=2,
            record_migrations=True,
        )
        assert ts.num_migrations > 0
        # We don't support simplify with migrations, so should fail.
        with pytest.raises(_tskit.LibraryError):
            ts.simplify()

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_no_update_sample_flags_no_filter_nodes(self, ts):
        # Can't simplify edges with metadata
        if ts.tables.edges.metadata_schema == tskit.MetadataSchema(schema=None):
            k = min(ts.num_samples, 3)
            subset = ts.samples()[:k]
            ts1 = ts.simplify(subset)
            ts2 = ts.simplify(subset, update_sample_flags=False, filter_nodes=False)
            assert ts1.num_samples == len(subset)
            assert ts2.num_samples == ts.num_samples
            assert ts1.num_edges == ts2.num_edges
            assert ts2.tables.nodes == ts.tables.nodes


class TestMinMaxTime:
    def get_example_tree_sequence(self, use_unknown_time):
        """
        Min time is set to 0.1.
        Max time is set to 2.0.
        """
        tables = tskit.TableCollection(sequence_length=2)
        tables.nodes.add_row(flags=1, time=0.1)
        tables.nodes.add_row(flags=1, time=0.1)
        tables.nodes.add_row(flags=1, time=0.1)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=0, time=2)
        tables.edges.add_row(left=0, right=2, parent=3, child=0)
        tables.edges.add_row(left=0, right=2, parent=3, child=1)
        tables.edges.add_row(left=0, right=2, parent=4, child=2)
        tables.edges.add_row(left=0, right=2, parent=4, child=3)
        tables.sites.add_row(position=0, ancestral_state="0")
        tables.sites.add_row(position=1, ancestral_state="0")
        if use_unknown_time:
            tables.mutations.add_row(
                site=0, node=2, derived_state="1", time=tskit.UNKNOWN_TIME
            )
            tables.mutations.add_row(
                site=1, node=3, derived_state="1", time=tskit.UNKNOWN_TIME
            )
        else:
            tables.mutations.add_row(site=0, node=2, derived_state="1", time=0.5)
            tables.mutations.add_row(site=1, node=3, derived_state="1", time=1.5)
        ts = tables.tree_sequence()
        return ts

    def get_empty_tree_sequence(self):
        """
        Min time is initialised to positive infinity.
        Max time is initialised to negative infinity.
        """
        tables = tskit.TableCollection(sequence_length=2)
        ts = tables.tree_sequence()
        return ts

    def test_example(self):
        ts = self.get_example_tree_sequence(use_unknown_time=False)
        expected_min_time = min(ts.nodes_time.min(), ts.mutations_time.min())
        expected_max_time = max(ts.nodes_time.max(), ts.mutations_time.max())
        assert ts.min_time == expected_min_time
        assert ts.max_time == expected_max_time

    def test_example_unknown_mutation_times(self):
        ts = self.get_example_tree_sequence(use_unknown_time=True)
        expected_min_time = ts.nodes_time.min()
        expected_max_time = ts.nodes_time.max()
        assert ts.min_time == expected_min_time
        assert ts.max_time == expected_max_time

    def test_empty(self):
        ts = self.get_empty_tree_sequence()
        assert ts.min_time == np.inf
        assert ts.max_time == -np.inf


class TestSiteAlleles:
    def test_no_mutations(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.sites.add_row(0, ancestral_state="")
        site = tables.tree_sequence().site(0)
        assert site.alleles == {""}

    @pytest.mark.parametrize("k", range(5))
    def test_k_mutations(self, k):
        tables = tskit.TableCollection(sequence_length=1)
        tables.sites.add_row(0, ancestral_state="ABC")
        tables.nodes.add_row(1, 0)
        tables.nodes.add_row(1, 0)  # will not have any mutations => missing
        for j in range(k):
            tables.mutations.add_row(site=0, node=0, derived_state=str(j))
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        variant = next(ts.variants())
        assert variant.has_missing_data
        assert len(variant.site.alleles) == k + 1
        assert "ABC" in variant.site.alleles
        assert variant.site.alleles == set(variant.alleles[:-1])


class TestEdgeDiffs:
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_correct_trees_forward(self, ts):
        parent = np.full(ts.num_nodes + 1, tskit.NULL, dtype=np.int32)
        for edge_diff, tree in itertools.zip_longest(ts.edge_diffs(), ts.trees()):
            assert edge_diff.interval == tree.interval
            for edge in edge_diff.edges_out:
                parent[edge.child] = tskit.NULL
            for edge in edge_diff.edges_in:
                parent[edge.child] = edge.parent
            assert_array_equal(parent, tree.parent_array)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_correct_trees_reverse(self, ts):
        parent = np.full(ts.num_nodes + 1, tskit.NULL, dtype=np.int32)
        iterator = itertools.zip_longest(
            ts.edge_diffs(direction=tskit.REVERSE), reversed(ts.trees())
        )
        for edge_diff, tree in iterator:
            assert edge_diff.interval == tree.interval
            for edge in edge_diff.edges_out:
                parent[edge.child] = tskit.NULL
            for edge in edge_diff.edges_in:
                parent[edge.child] = edge.parent
            assert_array_equal(parent, tree.parent_array)

    def test_elements_are_like_named_tuple(self, simple_degree2_ts_fixture):
        for val in simple_degree2_ts_fixture.edge_diffs():
            assert len(val) == 3
            assert val[0] == val.interval
            assert val[1] == val.edges_out
            assert val[2] == val.edges_in

    @pytest.mark.parametrize("direction", [-6, "forward", None])
    def test_bad_direction(self, direction, simple_degree2_ts_fixture):
        ts = simple_degree2_ts_fixture
        with pytest.raises(ValueError, match="direction must be"):
            ts.edge_diffs(direction=direction)

    @pytest.mark.parametrize("direction", [tskit.FORWARD, tskit.REVERSE])
    def test_edge_properties(self, direction, simple_degree2_ts_fixture):
        ts = simple_degree2_ts_fixture
        edge_ids = set()
        for _, e_out, e_in in ts.edge_diffs(direction=direction):
            for edge in e_in:
                assert edge.id not in edge_ids
                edge_ids.add(edge.id)
                assert ts.edge(edge.id) == edge
            for edge in e_out:
                assert ts.edge(edge.id) == edge
        assert edge_ids == set(range(ts.num_edges))

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    @pytest.mark.parametrize("direction", [tskit.FORWARD, tskit.REVERSE])
    def test_include_terminal(self, ts, direction):
        edges = set()
        i = 0
        diffs = ts.edge_diffs(include_terminal=True, direction=direction)
        parent = np.full(ts.num_nodes + 1, tskit.NULL, dtype=np.int32)
        for (left, right), e_out, e_in in diffs:  # noqa: B007
            for e in e_out:
                edges.remove(e.id)
                parent[e.child] = tskit.NULL
            for e in e_in:
                edges.add(e.id)
                parent[e.child] = e.parent
            i += 1
        assert np.all(parent == tskit.NULL)
        assert i == ts.num_trees + 1
        assert len(edges) == 0
        # On last iteration, interval is empty
        if direction == tskit.FORWARD:
            assert left == ts.sequence_length
            assert right == ts.sequence_length
        else:
            assert left == 0
            assert right == 0


class TestTreeSequenceMethodSignatures:
    ts = msprime.simulate(10, random_seed=1234)

    def test_kwargs_only(self):
        with pytest.raises(TypeError, match="argument"):
            tskit.Tree(self.ts, [], True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.trees([], True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.haplotypes(True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.variants(True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.genotype_matrix(True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.simplify([], True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.draw_svg("filename", True)
        with pytest.raises(TypeError, match="argument"):
            tskit.TreeSequence.load_tables(tskit.TableCollection(1), True)

    def test_trees_params(self):
        """
        The initial .trees() iterator parameters should match those in Tree.__init__()
        """
        tree_class_params = list(inspect.signature(tskit.Tree).parameters.items())
        trees_iter_params = list(
            inspect.signature(tskit.TreeSequence.trees).parameters.items()
        )
        # Skip the first param, which is `tree_sequence` and `self` respectively
        tree_class_params = tree_class_params[1:]
        # The trees iterator has some extra (deprecated) aliases
        trees_iter_params = trees_iter_params[1:-3]
        assert trees_iter_params == tree_class_params


class TestTreeSequenceMetadata:
    metadata_tables = [
        "node",
        "edge",
        "site",
        "mutation",
        "migration",
        "individual",
        "population",
    ]
    metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "title": "Example Metadata",
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "string_prop": {"type": "string"},
                "num_prop": {"type": "number"},
            },
            "required": ["table", "string_prop", "num_prop"],
            "additionalProperties": False,
        },
    )

    def test_tree_sequence_metadata_schema(self):
        tc = tskit.TableCollection(1)
        ts = tc.tree_sequence()
        assert repr(ts.metadata_schema) == repr(tskit.MetadataSchema(None))
        tc.metadata_schema = self.metadata_schema
        ts = tc.tree_sequence()
        assert repr(ts.metadata_schema) == repr(self.metadata_schema)
        with pytest.raises(AttributeError):
            del ts.metadata_schema
        with pytest.raises(AttributeError):
            ts.metadata_schema = tskit.MetadataSchema(None)

    def test_tree_sequence_metadata(self):
        tc = tskit.TableCollection(1)
        ts = tc.tree_sequence()
        assert ts.metadata == b""
        tc.metadata_schema = self.metadata_schema
        data = {
            "table": "tree-sequence",
            "string_prop": "stringy",
            "num_prop": 42,
        }
        tc.metadata = data
        ts = tc.tree_sequence()
        assert ts.metadata == data
        with pytest.raises(AttributeError):
            ts.metadata = {"should": "fail"}
        with pytest.raises(AttributeError):
            del ts.metadata

    def test_tree_sequence_time_units(self):
        tc = tskit.TableCollection(1)
        ts = tc.tree_sequence()
        assert ts.time_units == tskit.TIME_UNITS_UNKNOWN
        tc.time_units = "something else"
        ts = tc.tree_sequence()
        assert ts.time_units == "something else"
        with pytest.raises(AttributeError):
            del ts.time_units
        with pytest.raises(AttributeError):
            ts.time_units = "readonly"
        assert tskit.TIME_UNITS_UNKNOWN == "unknown"
        assert tskit.TIME_UNITS_UNCALIBRATED == "uncalibrated"

    def test_table_metadata_schemas(self):
        ts = msprime.simulate(5)
        for table in self.metadata_tables:
            tables = ts.dump_tables()
            # Set and read back a unique schema for each table
            schema = tskit.MetadataSchema({"codec": "json", "TEST": f"{table}-SCHEMA"})
            # Check via table API
            getattr(tables, f"{table}s").metadata_schema = schema
            assert repr(getattr(tables, f"{table}s").metadata_schema) == repr(schema)
            for other_table in self.metadata_tables:
                if other_table != table:
                    assert (
                        repr(getattr(tables, f"{other_table}s").metadata_schema) == ""
                    )
            # Check via tree-sequence API
            new_ts = tskit.TreeSequence.load_tables(tables)
            assert repr(getattr(new_ts.table_metadata_schemas, table)) == repr(schema)
            for other_table in self.metadata_tables:
                if other_table != table:
                    assert (
                        repr(getattr(new_ts.table_metadata_schemas, other_table)) == ""
                    )
            # Can't set schema via this API
            with pytest.raises(AttributeError):
                new_ts.table_metadata_schemas = {}
                # or modify the schema tuple return object
                with pytest.raises(dataclasses.exceptions.FrozenInstanceError):
                    setattr(
                        new_ts.table_metadata_schemas,
                        table,
                        tskit.MetadataSchema({"codec": "json"}),
                    )

    def test_table_metadata_round_trip_via_row_getters(self):
        # A tree sequence with all entities
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        )
        tables = ts.dump_tables()
        tables.individuals.add_row(location=[1, 2, 3])
        tables.individuals.add_row(location=[4, 5, 6])
        ts = tables.tree_sequence()

        for table in self.metadata_tables:
            new_tables = ts.dump_tables()
            tables_copy = ts.dump_tables()
            table_obj = getattr(new_tables, f"{table}s")
            table_obj.metadata_schema = self.metadata_schema
            table_obj.clear()
            # Write back the rows, but adding unique metadata
            for j, row in enumerate(getattr(tables_copy, f"{table}s")):
                row_data = dataclasses.asdict(row)
                row_data["metadata"] = {
                    "table": table,
                    "string_prop": f"Row number{j}",
                    "num_prop": j,
                }
                table_obj.add_row(**row_data)
            new_ts = new_tables.tree_sequence()
            # Check that all tables have data otherwise we'll silently not check one
            assert getattr(new_ts, f"num_{table}s") > 0
            assert getattr(new_ts, f"num_{table}s") == getattr(ts, f"num_{table}s")
            for j, row in enumerate(getattr(new_ts, f"{table}s")()):
                assert row.metadata == {
                    "table": table,
                    "string_prop": f"Row number{row.id}",
                    "num_prop": row.id,
                }
                assert getattr(new_ts, f"{table}")(j).metadata == {
                    "table": table,
                    "string_prop": f"Row number{row.id}",
                    "num_prop": row.id,
                }


def test_pickle_round_trip(ts_fixture):
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        ts = pickle.loads(pickle.dumps(ts_fixture, protocol=protocol))
        assert ts.tables == ts_fixture.tables
        # Do some thing to check the ts is init'd properly
        ts.draw_text()


class TestFileUuid(HighLevelTestCase):
    """
    Tests that the file UUID attribute is handled correctly.
    """

    def validate(self, ts):
        with tempfile.TemporaryDirectory() as tempdir:
            temp_file = pathlib.Path(tempdir) / "tmp.trees"
            assert ts.file_uuid is None
            ts.dump(temp_file)
            other_ts = tskit.load(temp_file)
            assert other_ts.file_uuid is not None
            assert len(other_ts.file_uuid), 36
            uuid = other_ts.file_uuid
            other_ts = tskit.load(temp_file)
            assert other_ts.file_uuid == uuid
            assert ts.tables == other_ts.tables

            # Check that the UUID is well-formed.
            parsed = _uuid.UUID("{" + uuid + "}")
            assert str(parsed) == uuid

            # Save the same tree sequence to the file. We should get a different UUID.
            ts.dump(temp_file)
            other_ts = tskit.load(temp_file)
            assert other_ts.file_uuid is not None
            assert other_ts.file_uuid != uuid

            # Even saving a ts that has a UUID to another file changes the UUID
            old_uuid = other_ts.file_uuid
            other_ts.dump(temp_file)
            assert other_ts.file_uuid == old_uuid
            other_ts = tskit.load(temp_file)
            assert other_ts.file_uuid is not None
            assert other_ts.file_uuid != old_uuid

            # Tables dumped from this ts are a deep copy, so they don't have
            # the file_uuid.
            tables = other_ts.dump_tables()
            assert tables.file_uuid is None

            # For now, ts.tables also returns a deep copy. This will hopefully
            # change in the future though.
            assert ts.tables.file_uuid is None

    def test_simple_simulation(self):
        ts = msprime.simulate(2, random_seed=1)
        self.validate(ts)

    def test_empty_tables(self):
        tables = tskit.TableCollection(1)
        self.validate(tables.tree_sequence())


class TestTreeSequenceTextIO(HighLevelTestCase):
    """
    Tests for the tree sequence text IO.
    """

    def verify_nodes_format(self, ts, nodes_file, precision, base64_metadata):
        """
        Verifies that the nodes we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_nodes = nodes_file.read().splitlines()
        assert len(output_nodes) - 1 == ts.num_nodes
        assert list(output_nodes[0].split()) == [
            "id",
            "is_sample",
            "time",
            "population",
            "individual",
            "metadata",
        ]
        for node, line in zip(ts.nodes(), output_nodes[1:]):
            splits = line.split("\t")
            assert str(node.id) == splits[0]
            assert str(node.is_sample()) == splits[1]
            assert convert(node.time) == splits[2]
            assert str(node.population) == splits[3]
            assert str(node.individual) == splits[4]
            if isinstance(node.metadata, bytes) and base64_metadata:
                assert tests.base64_encode(node.metadata) == splits[5]
            else:
                assert repr(node.metadata) == splits[5]

    def verify_edges_format(self, ts, edges_file, precision, base64_metadata):
        """
        Verifies that the edges we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_edges = edges_file.read().splitlines()
        assert len(output_edges) - 1 == ts.num_edges
        assert list(output_edges[0].split()) == [
            "left",
            "right",
            "parent",
            "child",
            "metadata",
        ]
        for edge, line in zip(ts.edges(), output_edges[1:]):
            splits = line.split("\t")
            assert convert(edge.left) == splits[0]
            assert convert(edge.right) == splits[1]
            assert str(edge.parent) == splits[2]
            assert str(edge.child) == splits[3]
            if isinstance(edge.metadata, bytes) and base64_metadata:
                assert tests.base64_encode(edge.metadata) == splits[4]
            else:
                assert repr(edge.metadata) == splits[4]

    def verify_sites_format(self, ts, sites_file, precision, base64_metadata):
        """
        Verifies that the sites we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_sites = sites_file.read().splitlines()
        assert len(output_sites) - 1 == ts.num_sites
        assert list(output_sites[0].split()) == [
            "position",
            "ancestral_state",
            "metadata",
        ]
        for site, line in zip(ts.sites(), output_sites[1:]):
            splits = line.split("\t")
            assert convert(site.position) == splits[0]
            assert site.ancestral_state == splits[1]
            if isinstance(site.metadata, bytes) and base64_metadata:
                assert tests.base64_encode(site.metadata) == splits[2]
            else:
                assert repr(site.metadata) == splits[2]

    def verify_mutations_format(self, ts, mutations_file, precision, base64_metadata):
        """
        Verifies that the mutations we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_mutations = mutations_file.read().splitlines()
        assert len(output_mutations) - 1 == ts.num_mutations
        assert list(output_mutations[0].split()) == [
            "site",
            "node",
            "time",
            "derived_state",
            "parent",
            "metadata",
        ]
        mutations = [mut for site in ts.sites() for mut in site.mutations]
        for mutation, line in zip(mutations, output_mutations[1:]):
            splits = line.split("\t")
            assert str(mutation.site) == splits[0]
            assert str(mutation.node) == splits[1]
            assert (
                "unknown" if util.is_unknown_time(mutation.time) else str(mutation.time)
            ) == splits[2]
            assert str(mutation.derived_state) == splits[3]
            assert str(mutation.parent) == splits[4]
            if isinstance(mutation.metadata, bytes) and base64_metadata:
                assert tests.base64_encode(mutation.metadata) == splits[5]
            else:
                assert repr(mutation.metadata) == splits[5]

    def verify_individuals_format(
        self, ts, individuals_file, precision, base64_metadata
    ):
        """
        Verifies that the individuals we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_individuals = individuals_file.read().splitlines()
        assert len(output_individuals) - 1 == ts.num_individuals
        assert list(output_individuals[0].split()) == [
            "id",
            "flags",
            "location",
            "parents",
            "metadata",
        ]
        for individual, line in zip(ts.individuals(), output_individuals[1:]):
            splits = line.split("\t")
            assert str(individual.id) == splits[0]
            assert str(individual.flags) == splits[1]
            assert ",".join(map(str, individual.location)) == splits[2]
            assert ",".join(map(str, individual.parents)) == splits[3]
            if isinstance(individual.metadata, bytes) and base64_metadata:
                assert tests.base64_encode(individual.metadata) == splits[4]
            else:
                assert repr(individual.metadata) == splits[4]

    def verify_populations_format(
        self, ts, populations_file, precision, base64_metadata
    ):
        """
        Verifies that the populations we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_populations = populations_file.read().splitlines()
        assert len(output_populations) - 1 == ts.num_populations
        assert list(output_populations[0].split()) == [
            "id",
            "metadata",
        ]
        for population, line in zip(ts.populations(), output_populations[1:]):
            splits = line.split("\t")
            assert str(population.id) == splits[0]
            if isinstance(population.metadata, bytes) and base64_metadata:
                assert tests.base64_encode(population.metadata) == splits[1]
            else:
                assert repr(population.metadata) == splits[1]

    def verify_migrations_format(self, ts, migrations_file, precision, base64_metadata):
        """
        Verifies that the migrations we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_migrations = migrations_file.read().splitlines()
        assert len(output_migrations) - 1 == ts.num_migrations
        assert list(output_migrations[0].split()) == [
            "left",
            "right",
            "node",
            "source",
            "dest",
            "time",
            "metadata",
        ]
        for migration, line in zip(ts.migrations(), output_migrations[1:]):
            splits = line.split("\t")
            assert str(migration.left) == splits[0]
            assert str(migration.right) == splits[1]
            assert str(migration.node) == splits[2]
            assert str(migration.source) == splits[3]
            assert str(migration.dest) == splits[4]
            assert str(migration.time) == splits[5]
            if isinstance(migration.metadata, bytes) and base64_metadata:
                assert tests.base64_encode(migration.metadata) == splits[6]
            else:
                assert repr(migration.metadata) == splits[6]

    @pytest.mark.parametrize(("precision", "base64_metadata"), [(2, True), (7, False)])
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_output_format(self, precision, base64_metadata, ts):
        nodes_file = io.StringIO()
        edges_file = io.StringIO()
        sites_file = io.StringIO()
        mutations_file = io.StringIO()
        individuals_file = io.StringIO()
        populations_file = io.StringIO()
        migrations_file = io.StringIO()
        provenances_file = io.StringIO()
        ts.dump_text(
            nodes=nodes_file,
            edges=edges_file,
            sites=sites_file,
            mutations=mutations_file,
            individuals=individuals_file,
            populations=populations_file,
            migrations=migrations_file,
            provenances=provenances_file,
            precision=precision,
            base64_metadata=base64_metadata,
        )
        nodes_file.seek(0)
        edges_file.seek(0)
        sites_file.seek(0)
        mutations_file.seek(0)
        individuals_file.seek(0)
        populations_file.seek(0)
        migrations_file.seek(0)
        self.verify_nodes_format(ts, nodes_file, precision, base64_metadata)
        self.verify_edges_format(ts, edges_file, precision, base64_metadata)
        self.verify_sites_format(ts, sites_file, precision, base64_metadata)
        self.verify_mutations_format(ts, mutations_file, precision, base64_metadata)
        self.verify_individuals_format(ts, individuals_file, precision, base64_metadata)
        self.verify_populations_format(ts, populations_file, precision, base64_metadata)
        self.verify_migrations_format(ts, migrations_file, precision, base64_metadata)

    def verify_approximate_equality(self, ts1, ts2):
        """
        Verifies that the specified tree sequences are approximately
        equal, taking into account the error incurred in exporting to text.
        """
        assert ts1.sample_size == ts2.sample_size
        assert ts1.sequence_length == ts2.sequence_length
        assert ts1.num_nodes == ts2.num_nodes
        assert ts1.num_edges == ts2.num_edges
        assert ts1.num_sites == ts2.num_sites
        assert ts1.num_mutations == ts2.num_mutations
        assert ts1.num_populations == ts2.num_populations
        assert ts1.num_migrations == ts2.num_migrations

        checked = 0
        for n1, n2 in zip(ts1.nodes(), ts2.nodes()):
            assert n1.population == n2.population
            assert n1.metadata == n2.metadata
            assert n1.time == pytest.approx(n2.time)
            checked += 1
        assert checked == ts1.num_nodes

        checked = 0
        for r1, r2 in zip(ts1.edges(), ts2.edges()):
            checked += 1
            assert r1.left == pytest.approx(r2.left)
            assert r1.right == pytest.approx(r2.right)
            assert r1.parent == r2.parent
            assert r1.child == r2.child
        assert ts1.num_edges == checked

        checked = 0
        for s1, s2 in zip(ts1.sites(), ts2.sites()):
            checked += 1
            assert s1.position == pytest.approx(s2.position)
            assert s1.ancestral_state == s2.ancestral_state
            assert s1.metadata == s2.metadata
            assert s1.mutations == s2.mutations
        assert ts1.num_sites == checked

        checked = 0
        for s1, s2 in zip(ts1.mutations(), ts2.mutations()):
            checked += 1
            assert s1.site == s2.site
            assert s1.node == s2.node
            if not (math.isnan(s1.time) and math.isnan(s2.time)):
                assert s1.time == pytest.approx(s2.time)
            assert s1.derived_state == s2.derived_state
            assert s1.parent == s2.parent
            assert s1.metadata == s2.metadata
        assert ts1.num_mutations == checked

        checked = 0
        for s1, s2 in zip(ts1.migrations(), ts2.migrations()):
            checked += 1
            assert s1.left == s2.left
            assert s1.right == s2.right
            assert s1.node == s2.node
            assert s1.source == s2.source
            assert s1.dest == s2.dest
            assert s1.time == s2.time
            assert s1.metadata == s2.metadata
        assert ts1.num_migrations == checked

        # Check the trees
        check = 0
        for t1, t2 in zip(ts1.trees(), ts2.trees()):
            assert list(t1.nodes()) == list(t2.nodes())
            check += 1
        assert check == ts1.get_num_trees()

    @pytest.mark.parametrize("ts1", tsutil.get_example_tree_sequences())
    def test_text_record_round_trip(self, ts1):
        # Can't round trip without the schema
        if ts1.tables.nodes.metadata_schema == tskit.MetadataSchema(None):
            nodes_file = io.StringIO()
            edges_file = io.StringIO()
            sites_file = io.StringIO()
            mutations_file = io.StringIO()
            individuals_file = io.StringIO()
            populations_file = io.StringIO()
            migrations_file = io.StringIO()
            ts1.dump_text(
                nodes=nodes_file,
                edges=edges_file,
                sites=sites_file,
                mutations=mutations_file,
                individuals=individuals_file,
                populations=populations_file,
                migrations=migrations_file,
                precision=16,
            )
            nodes_file.seek(0)
            edges_file.seek(0)
            sites_file.seek(0)
            mutations_file.seek(0)
            individuals_file.seek(0)
            populations_file.seek(0)
            migrations_file.seek(0)
            ts2 = tskit.load_text(
                nodes=nodes_file,
                edges=edges_file,
                sites=sites_file,
                mutations=mutations_file,
                individuals=individuals_file,
                populations=populations_file,
                migrations=migrations_file,
                sequence_length=ts1.sequence_length,
                strict=True,
            )
            tables1 = ts1.tables.copy()
            # load_text performs a `sort`, which changes the order relative to
            # the original tree sequence
            tables1.sort()
            ts1_sorted = tables1.tree_sequence()
            self.verify_approximate_equality(ts1_sorted, ts2)

    def test_empty_files(self):
        nodes_file = io.StringIO("is_sample\ttime\n")
        edges_file = io.StringIO("left\tright\tparent\tchild\n")
        sites_file = io.StringIO("position\tancestral_state\n")
        mutations_file = io.StringIO("site\tnode\tderived_state\n")
        individuals_file = io.StringIO("flags\n")
        migrations_file = io.StringIO("left\tright\tnode\tsource\tdest\ttime\n")
        with pytest.raises(_tskit.LibraryError):
            tskit.load_text(
                nodes=nodes_file,
                edges=edges_file,
                sites=sites_file,
                mutations=mutations_file,
                individuals=individuals_file,
                migrations=migrations_file,
            )

    def test_empty_files_sequence_length(self):
        nodes_file = io.StringIO("is_sample\ttime\n")
        edges_file = io.StringIO("left\tright\tparent\tchild\n")
        sites_file = io.StringIO("position\tancestral_state\n")
        mutations_file = io.StringIO("site\tnode\tderived_state\n")
        individuals_file = io.StringIO("flags\n")
        migrations_file = io.StringIO("left\tright\tnode\tsource\tdest\ttime\n")
        ts = tskit.load_text(
            nodes=nodes_file,
            edges=edges_file,
            sites=sites_file,
            mutations=mutations_file,
            individuals=individuals_file,
            migrations=migrations_file,
            sequence_length=100,
        )
        assert ts.sequence_length == 100
        assert ts.num_nodes == 0
        assert ts.num_edges == 0
        assert ts.num_sites == 0
        assert ts.num_mutations == 0
        assert ts.num_individuals == 0
        assert ts.num_migrations == 0

    def test_load_text_no_populations(self):
        nodes_file = io.StringIO("is_sample\ttime\tpopulation\n1\t0\t2\n")
        edges_file = io.StringIO("left\tright\tparent\tchild\n")
        ts = tskit.load_text(nodes_file, edges_file, sequence_length=100)
        assert ts.num_nodes == 1
        assert ts.num_populations == 3

    def test_load_text_populations(self):
        nodes_file = io.StringIO("is_sample\ttime\tpopulation\n")
        edges_file = io.StringIO("left\tright\tparent\tchild\n")
        populations_file = io.StringIO("metadata\nmetadata_1\nmetadata_2\n")
        ts = tskit.load_text(
            nodes_file,
            edges_file,
            populations=populations_file,
            sequence_length=100,
            base64_metadata=False,
        )
        assert ts.num_populations == 2
        assert ts.tables.populations[0].metadata == b"metadata_1"
        assert ts.tables.populations[1].metadata == b"metadata_2"


class TestTree(HighLevelTestCase):
    """
    Some simple tests on the tree API.
    """

    def get_tree(self, sample_lists=False):
        ts = msprime.simulate(10, random_seed=1, mutation_rate=1, record_full_arg=True)
        return next(ts.trees(sample_lists=sample_lists))

    def verify_mutations(self, tree):
        assert tree.num_mutations > 0
        other_mutations = []
        for site in tree.sites():
            for mutation in site.mutations:
                other_mutations.append(mutation)
        mutations = list(tree.mutations())
        assert tree.num_mutations == len(other_mutations)
        assert tree.num_mutations == len(mutations)
        for mut, other_mut in zip(mutations, other_mutations):
            assert mut == other_mut

    def test_simple_mutations(self):
        tree = self.get_tree()
        self.verify_mutations(tree)

    def test_complex_mutations(self):
        ts = tsutil.insert_branch_mutations(msprime.simulate(10, random_seed=1))
        self.verify_mutations(ts.first())

    def test_str(self, ts_fixture):
        t = ts_fixture.first()
        assert isinstance(str(t), str)
        pattern = re.compile(
            r"""
            ╔═+╗\s*
            ║Tree.*?║\s*
            ╠═+╤═+╣\s*
            ║Index.*?│\s*[\d\u2009,]+║\s*
            ╟─+┼─+╢\s*
            ║Interval.*?│\s*[\d\u2009,]+-[\d\u2009,]+\s*\([\d\u2009,]+\)║\s*
            ╟─+┼─+╢\s*
            ║Roots.*?│\s*[\d\u2009,]+║\s*
            ╟─+┼─+╢\s*
            ║Nodes.*?│\s*[\d\u2009,]+║\s*
            ╟─+┼─+╢\s*
            ║Sites.*?│\s*[\d\u2009,]+║\s*
            ╟─+┼─+╢\s*
            ║Mutations.*?│\s*[\d\u2009,]+║\s*
            ╟─+┼─+╢\s*
            ║Total\s*Branch\s*Length.*?│\s*[\d\u2009,]+\.\d+║\s*
            ╚═+╧═+╝\s*
            """,
            re.VERBOSE | re.DOTALL,
        )
        assert pattern.search(str(t))

    def test_html_repr(self, ts_fixture):
        html = ts_fixture.first()._repr_html_()
        # Parse to check valid
        ElementTree.fromstring(html)
        assert len(html) > 1900
        assert "<tr><td>Total Branch Length</td><td>" in html

    def test_samples(self):
        for sample_lists in [True, False]:
            t = self.get_tree(sample_lists)
            n = t.get_sample_size()
            all_samples = list(t.samples(t.get_root()))
            assert sorted(all_samples) == list(range(n))
            for j in range(n):
                assert list(t.samples(j)) == [j]

            def test_func(t, u):
                """
                Simple test definition of the traversal.
                """
                stack = [u]
                while len(stack) > 0:
                    v = stack.pop()
                    if t.is_sample(v):
                        yield v
                    if t.is_internal(v):
                        for c in reversed(t.get_children(v)):
                            stack.append(c)

            for u in t.nodes():
                l1 = list(t.samples(u))
                l2 = list(test_func(t, u))
                assert l1 == l2
                assert t.get_num_samples(u) == len(l1)

    def test_num_children(self):
        tree = self.get_tree()
        for u in tree.nodes():
            assert tree.num_children(u) == len(tree.children(u))

    def test_ancestors(self):
        tree = tskit.Tree.generate_balanced(10, arity=3)
        ancestors_arrays = {u: [] for u in np.arange(tree.tree_sequence.num_nodes)}
        ancestors_arrays[-1] = []
        for u in tree.nodes(order="preorder"):
            parent = tree.parent(u)
            if parent != tskit.NULL:
                ancestors_arrays[u] = [parent] + ancestors_arrays[tree.parent(u)]
        for u in tree.nodes():
            assert list(tree.ancestors(u)) == ancestors_arrays[u]

    def test_ancestors_empty(self):
        ts = tskit.Tree.generate_comb(10).tree_sequence
        tree = ts.delete_intervals([[0, 1]]).first()
        for u in ts.samples():
            assert len(list(tree.ancestors(u))) == 0

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_virtual_root_semantics(self, ts):
        for tree in ts.trees():
            assert math.isinf(tree.time(tree.virtual_root))
            assert tree.depth(tree.virtual_root) == -1
            assert tree.parent(tree.virtual_root) == -1
            assert list(tree.children(tree.virtual_root)) == tree.roots
            with pytest.raises(tskit.LibraryError, match="bounds"):
                tree.population(tree.virtual_root)

    def test_root_properties(self):
        tested = set()
        for ts in tsutil.get_example_tree_sequences(pytest_params=False):
            for tree in ts.trees():
                if tree.has_single_root:
                    tested.add("single")
                    assert tree.num_roots == 1
                    assert tree.num_roots == 1
                    assert tree.root != tskit.NULL
                elif tree.has_multiple_roots:
                    tested.add("multiple")
                    assert tree.num_roots > 1
                    with pytest.raises(ValueError, match="More than one root exists"):
                        _ = tree.root
                else:
                    tested.add("zero")
                    assert tree.num_roots == 0
                    assert tree.root == tskit.NULL
        assert len(tested) == 3

    def test_as_dict_of_dicts(self):
        for ts in tsutil.get_example_tree_sequences(pytest_params=False):
            tree = next(ts.trees())
            adj_dod = tree.as_dict_of_dicts()
            g = nx.DiGraph(adj_dod)

            self.verify_nx_graph_topology(tree, g)
            self.verify_nx_algorithm_equivalence(tree, g)
            self.verify_nx_for_tutorial_algorithms(tree, g)
        self.verify_nx_nearest_neighbor_search()

    def verify_nx_graph_topology(self, tree, g):
        assert set(tree.nodes()) == set(g.nodes)

        assert set(tree.roots) == {n for n in g.nodes if g.in_degree(n) == 0}

        assert set(tree.leaves()) == {n for n in g.nodes if g.out_degree(n) == 0}

        # test if tree has no in-degrees > 1
        if len(g) > 0:
            assert nx.is_branching(g)

    def verify_nx_algorithm_equivalence(self, tree, g):
        for root in tree.roots:
            assert nx.is_directed_acyclic_graph(g)

            # test descendants
            assert {u for u in tree.nodes() if tree.is_descendant(u, root)} == set(
                nx.descendants(g, root)
            ) | {root}

            # test MRCA
            if tree.tree_sequence.num_nodes < 20:
                for u, v in itertools.combinations(tree.nodes(), 2):
                    mrca = nx.lowest_common_ancestor(g, u, v)
                    if mrca is None:
                        mrca = -1
                    assert tree.mrca(u, v) == mrca

            # test node traversal modes
            assert list(tree.nodes(root=root, order="breadthfirst")) == [root] + [
                v for u, v in nx.bfs_edges(g, root)
            ]
            assert list(tree.nodes(root=root, order="preorder")) == list(
                nx.dfs_preorder_nodes(g, root)
            )

    def verify_nx_for_tutorial_algorithms(self, tree, g):
        # traversing upwards
        for u in tree.leaves():
            path = []
            v = u
            while v != tskit.NULL:
                path.append(v)
                v = tree.parent(v)

            assert set(path) == {u} | nx.ancestors(g, u)
            assert path == [u] + [
                n1 for n1, n2, _ in nx.edge_dfs(g, u, orientation="reverse")
            ]

        # traversals with information
        def preorder_dist(tree, root):
            stack = [(root, 0)]
            while len(stack) > 0:
                u, distance = stack.pop()
                yield u, distance
                for v in tree.children(u):
                    stack.append((v, distance + 1))

        for root in tree.roots:
            assert {
                k: v for k, v in preorder_dist(tree, root)
            } == nx.shortest_path_length(g, source=root)

        for root in tree.roots:
            # new traversal: measuring time between root and MRCA
            for u, v in itertools.combinations(nx.descendants(g, root), 2):
                mrca = tree.mrca(u, v)
                tmrca = tree.time(mrca)
                assert tree.time(root) - tmrca == pytest.approx(
                    nx.shortest_path_length(
                        g, source=root, target=mrca, weight="branch_length"
                    )
                )

    def verify_nx_nearest_neighbor_search(self):
        samples = [
            msprime.Sample(0, 0),
            msprime.Sample(0, 1),
            msprime.Sample(0, 20),
        ]
        ts = msprime.simulate(
            Ne=1e6,
            samples=samples,
            demographic_events=[
                msprime.PopulationParametersChange(
                    time=10, growth_rate=2, population_id=0
                ),
            ],
            random_seed=42,
        )

        tree = ts.first()
        g = nx.Graph(tree.as_dict_of_dicts())

        dist_dod = collections.defaultdict(dict)
        for source, target in itertools.combinations(tree.samples(), 2):
            dist_dod[source][target] = nx.shortest_path_length(
                g, source=source, target=target, weight="branch_length"
            )
            dist_dod[target][source] = dist_dod[source][target]

        nearest_neighbor_of = [min(dist_dod[u], key=dist_dod[u].get) for u in range(3)]
        assert [2, 2, 1] == [nearest_neighbor_of[u] for u in range(3)]

    def test_total_branch_length(self):
        # Note: this definition works when we have no non-sample branches.
        t1 = self.get_tree()
        bl = 0
        root = t1.get_root()
        for node in t1.nodes():
            if node != root:
                bl += t1.get_branch_length(node)
        assert bl > 0
        assert t1.get_total_branch_length() == pytest.approx(bl)

    def test_branch_length_empty_tree(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        ts = tables.tree_sequence()
        assert ts.num_trees == 1
        tree = ts.first()
        assert tree.branch_length(0) == 0
        assert tree.branch_length(1) == 0
        assert tree.total_branch_length == 0

    @pytest.mark.parametrize("r_threshold", [0, -1])
    def test_bad_val_root_threshold(self, r_threshold):
        with pytest.raises(ValueError, match="greater than 0"):
            tskit.Tree.generate_balanced(2, root_threshold=r_threshold)

    @pytest.mark.parametrize("r_threshold", [None, 0.5, 1.5, np.inf])
    def test_bad_type_root_threshold(self, r_threshold):
        with pytest.raises(TypeError):
            tskit.Tree.generate_balanced(2, root_threshold=r_threshold)

    def test_simple_root_threshold(self):
        tree = tskit.Tree.generate_balanced(3, root_threshold=3)
        assert tree.num_roots == 1
        tree = tskit.Tree.generate_balanced(3, root_threshold=4)
        assert tree.num_roots == 0

    @pytest.mark.parametrize("root_threshold", [1, 2, 3])
    def test_is_root(self, root_threshold):
        # Make a tree with multiple roots with different numbers of samples under each
        ts = tskit.Tree.generate_balanced(5).tree_sequence
        ts = ts.decapitate(ts.max_root_time - 0.1)
        tables = ts.dump_tables()
        tables.nodes.add_row(flags=0)  # Isolated non-sample
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE)  # Isolated sample
        ts = tables.tree_sequence()
        assert {ts.first().num_samples(u) for u in ts.first().roots} == {1, 2, 3}
        tree = ts.first(root_threshold=root_threshold)
        roots = set(tree.roots)
        for u in range(ts.num_nodes):  # Will also test isolated nodes
            assert tree.is_root(u) == (u in roots)

    def test_is_descendant(self):
        def is_descendant(tree, u, v):
            path = []
            while u != tskit.NULL:
                path.append(u)
                u = tree.parent(u)
            return v in path

        tree = self.get_tree()
        for u, v in itertools.product(range(tree.tree_sequence.num_nodes), repeat=2):
            assert is_descendant(tree, u, v) == tree.is_descendant(u, v)
        # All nodes are descendents of themselves
        for u in range(tree.tree_sequence.num_nodes + 1):
            assert tree.is_descendant(u, u)
        for bad_node in [-1, -2, tree.tree_sequence.num_nodes + 1]:
            with pytest.raises(ValueError):
                tree.is_descendant(0, bad_node)
            with pytest.raises(ValueError):
                tree.is_descendant(bad_node, 0)
            with pytest.raises(ValueError):
                tree.is_descendant(bad_node, bad_node)

    def test_apis(self):
        # tree properties
        t1 = self.get_tree()
        assert t1.get_root() == t1.root
        assert t1.get_index() == t1.index
        assert t1.get_interval() == t1.interval
        assert t1.get_sample_size() == t1.sample_size
        assert t1.get_num_mutations() == t1.num_mutations
        assert t1.get_parent_dict() == t1.parent_dict
        assert t1.get_total_branch_length() == t1.total_branch_length
        assert t1.span == t1.interval.right - t1.interval.left
        assert t1.mid == t1.interval.left + (t1.interval.right - t1.interval.left) / 2
        # node properties
        root = t1.get_root()
        for node in t1.nodes():
            if node != root:
                assert t1.get_time(node) == t1.time(node)
                assert t1.get_parent(node) == t1.parent(node)
                assert t1.get_children(node) == t1.children(node)
                assert t1.get_population(node) == t1.population(node)
                assert t1.get_num_samples(node) == t1.num_samples(node)
                assert t1.get_branch_length(node) == t1.branch_length(node)
                assert t1.get_num_tracked_samples(node) == t1.num_tracked_samples(node)

        pairs = itertools.islice(itertools.combinations(t1.nodes(), 2), 50)
        for pair in pairs:
            assert t1.get_mrca(*pair) == t1.mrca(*pair)
            assert t1.get_tmrca(*pair) == t1.tmrca(*pair)

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_deprecated_apis(self):
        t1 = self.get_tree()
        assert t1.get_length() == t1.span
        assert t1.length == t1.span
        assert t1.num_nodes == t1.tree_sequence.num_nodes

    def test_deprecated_api_warnings(self):
        # Deprecated and will be removed
        t1 = self.get_tree()
        with pytest.warns(FutureWarning, match="Tree.tree_sequence.num_nodes"):
            t1.num_nodes

    @pytest.mark.parametrize("skip", [False, True])
    def test_seek_index(self, skip):
        ts = msprime.simulate(10, recombination_rate=3, length=5, random_seed=42)
        N = ts.num_trees
        assert ts.num_trees > 3
        tree = tskit.Tree(ts)
        for index in [0, N // 2, N - 1, 1]:
            fresh_tree = tskit.Tree(ts)
            assert fresh_tree.index == -1
            fresh_tree.seek_index(index)
            assert fresh_tree.index == index
            tree.seek_index(index, skip)
            assert_trees_equivalent(fresh_tree, tree)

        tree = tskit.Tree(ts)
        for index in [-1, -2, -N + 2, -N + 1, -N]:
            fresh_tree = tskit.Tree(ts)
            assert fresh_tree.index == -1
            fresh_tree.seek_index(index)
            tree.seek_index(index, skip)
            assert fresh_tree.index == index + N
            assert tree.index == index + N
            assert_trees_equivalent(fresh_tree, tree)

    def test_seek_index_errors(self):
        tree = self.get_tree()
        N = tree.tree_sequence.num_trees
        with pytest.raises(IndexError):
            tree.seek_index(N)
        with pytest.raises(IndexError):
            tree.seek_index(N + 1)
        with pytest.raises(IndexError):
            tree.seek_index(-N - 1)
        with pytest.raises(IndexError):
            tree.seek_index(-N - 2)

    def test_first_last(self):
        ts = msprime.simulate(10, recombination_rate=3, length=2, random_seed=42)
        assert ts.num_trees > 3
        tree = tskit.Tree(ts)
        tree.first()
        assert tree.index == 0
        tree = tskit.Tree(ts)
        tree.last()
        assert tree.index == ts.num_trees - 1
        tree = tskit.Tree(ts)
        for _ in range(3):
            tree.last()
            assert tree.index == ts.num_trees - 1
            tree.first()
            assert tree.index == 0

    def test_eq_different_tree_sequence(self):
        ts = msprime.simulate(4, recombination_rate=1, length=2, random_seed=42)
        copy = ts.dump_tables().tree_sequence()
        for tree1, tree2 in zip(ts.aslist(), copy.aslist()):
            assert tree1 != tree2

    def test_next_prev(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        assert ts.num_trees > 5
        for index, tree in enumerate(ts.aslist()):
            assert tree.index == index
            j = index
            while tree.next():
                j += 1
                assert tree.index == j
            assert tree.index == -1
            assert j + 1 == ts.num_trees
        for index, tree in enumerate(ts.aslist()):
            assert tree.index == index
            j = index
            while tree.prev():
                j -= 1
                assert tree.index == j
            assert tree.index == -1
            assert j == 0
        tree.first()
        tree.prev()
        assert tree.index == -1
        tree.last()
        tree.next()
        assert tree.index == -1

    def test_interval(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=1)
        assert ts.num_trees > 1
        breakpoints = list(ts.breakpoints())
        assert breakpoints[0] == 0
        assert breakpoints[-1] == ts.sequence_length
        for i, tree in enumerate(ts.trees()):
            assert tree.interval.left == pytest.approx(breakpoints[i])
            assert tree.interval.left == pytest.approx(breakpoints[i])
            assert tree.interval.right == pytest.approx(breakpoints[i + 1])
            assert tree.interval.right == pytest.approx(breakpoints[i + 1])
            assert tree.interval.span == pytest.approx(
                breakpoints[i + 1] - breakpoints[i]
            )
            assert tree.interval.mid == pytest.approx(
                breakpoints[i] + (breakpoints[i + 1] - breakpoints[i]) / 2
            )

    def verify_tree_arrays(self, tree):
        ts = tree.tree_sequence
        N = ts.num_nodes + 1
        assert tree.parent_array.shape == (N,)
        assert tree.left_child_array.shape == (N,)
        assert tree.right_child_array.shape == (N,)
        assert tree.left_sib_array.shape == (N,)
        assert tree.right_sib_array.shape == (N,)
        assert tree.num_children_array.shape == (N,)
        assert tree.edge_array.shape == (N,)
        for u in range(N):
            assert tree.parent(u) == tree.parent_array[u]
            assert tree.left_child(u) == tree.left_child_array[u]
            assert tree.right_child(u) == tree.right_child_array[u]
            assert tree.left_sib(u) == tree.left_sib_array[u]
            assert tree.right_sib(u) == tree.right_sib_array[u]
            assert tree.num_children(u) == tree.num_children_array[u]
            assert tree.edge(u) == tree.edge_array[u]

    def verify_tree_arrays_python_ts(self, ts):
        pts = tests.PythonTreeSequence(ts)
        iter1 = ts.trees()
        iter2 = pts.trees()
        for st1, st2 in zip(iter1, iter2):
            assert np.all(st1.parent_array == st2.parent)
            assert np.all(st1.left_child_array == st2.left_child)
            assert np.all(st1.right_child_array == st2.right_child)
            assert np.all(st1.left_sib_array == st2.left_sib)
            assert np.all(st1.right_sib_array == st2.right_sib)
            assert np.all(st1.num_children_array == st2.num_children)
            assert np.all(st1.edge_array == st2.edge)

    def test_tree_arrays(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=1)
        assert ts.num_trees > 1
        self.verify_tree_arrays_python_ts(ts)
        for tree in ts.trees():
            self.verify_tree_arrays(tree)

    @pytest.mark.parametrize(
        "array",
        [
            "parent",
            "left_child",
            "right_child",
            "left_sib",
            "right_sib",
            "num_children",
            "edge",
        ],
    )
    def test_tree_array_properties(self, array):
        name = array + "_array"
        ts = msprime.simulate(10, random_seed=1)
        tree = ts.first()
        a = getattr(tree, name)
        assert getattr(tree, name) is a
        assert a.base is tree._ll_tree
        with pytest.raises(AttributeError):
            setattr(tree, name, None)
        with pytest.raises(AttributeError):
            delattr(tree, name)

    def verify_empty_tree(self, tree):
        ts = tree.tree_sequence
        assert tree.index == -1
        assert tree.parent_dict == {}
        for u in range(ts.num_nodes):
            assert tree.parent(u) == tskit.NULL
            assert tree.left_child(u) == tskit.NULL
            assert tree.right_child(u) == tskit.NULL
            assert tree.num_children(u) == 0
            assert tree.edge(u) == tskit.NULL
            if not ts.node(u).is_sample():
                assert tree.left_sib(u) == tskit.NULL
                assert tree.right_sib(u) == tskit.NULL
        # Samples should have left-sib right-sibs set
        samples = ts.samples()
        assert tree.left_root == samples[0]
        for j in range(ts.num_samples):
            if j > 0:
                assert tree.left_sib(samples[j]) == samples[j - 1]
            if j < ts.num_samples - 1:
                assert tree.right_sib(samples[j]) == samples[j + 1]
        self.verify_tree_arrays(tree)

    def test_empty_tree(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        assert ts.num_trees > 5
        tree = tskit.Tree(ts)
        self.verify_empty_tree(tree)
        while tree.next():
            pass
        self.verify_empty_tree(tree)
        while tree.prev():
            pass
        self.verify_empty_tree(tree)

    def test_clear(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        assert ts.num_trees > 5
        tree = tskit.Tree(ts)
        tree.first()
        tree.clear()
        self.verify_empty_tree(tree)
        tree.last()
        tree.clear()
        self.verify_empty_tree(tree)
        tree.seek_index(ts.num_trees // 2)
        tree.clear()
        self.verify_empty_tree(tree)

    def verify_trees_identical(self, t1, t2):
        assert t1.tree_sequence is t2.tree_sequence
        assert np.all(t1.parent_array == t2.parent_array)
        assert np.all(t1.left_child_array == t2.left_child_array)
        assert np.all(t1.right_child_array == t2.right_child_array)
        assert np.all(t1.left_sib_array == t2.left_sib_array)
        assert np.all(t1.right_sib_array == t2.right_sib_array)
        assert np.all(t1.num_children_array == t2.num_children_array)
        assert np.all(t1.edge_array == t2.edge_array)
        assert list(t1.sites()) == list(t2.sites())

    def test_copy_seek(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        assert ts.num_trees > 5
        tree = tskit.Tree(ts)
        copy = tree.copy()
        self.verify_empty_tree(copy)
        while tree.next():
            copy = tree.copy()
            self.verify_trees_identical(tree, copy)
        while tree.prev():
            copy = tree.copy()
            self.verify_trees_identical(tree, copy)
        tree.clear()
        copy = tree.copy()
        tree.first()
        # Make sure the underlying arrays are different
        assert np.any(tree.parent_array != copy.parent_array)
        copy.first()
        while tree.index != -1:
            self.verify_trees_identical(tree, copy)
            assert tree.next() == copy.next()
        tree.last()
        copy.last()
        while tree.index != -1:
            self.verify_trees_identical(tree, copy)
            assert tree.prev() == copy.prev()
        # Seek to middle and two independent trees.
        tree.seek_index(ts.num_trees // 2)
        left_copy = tree.copy()
        right_copy = tree.copy()
        self.verify_trees_identical(tree, left_copy)
        self.verify_trees_identical(tree, right_copy)
        left_copy.prev()
        assert left_copy.index == tree.index - 1
        right_copy.next()
        assert right_copy.index == tree.index + 1

    def test_copy_tracked_samples(self):
        ts = msprime.simulate(10, recombination_rate=2, length=3, random_seed=42)
        tree = tskit.Tree(ts, tracked_samples=[0, 1])
        while tree.next():
            copy = tree.copy()
            for j in range(ts.num_nodes):
                assert tree.num_tracked_samples(j) == copy.num_tracked_samples(j)
        copy = tree.copy()
        while tree.next():
            copy.next()
            for j in range(ts.num_nodes):
                assert tree.num_tracked_samples(j) == copy.num_tracked_samples(j)

    def test_copy_multiple_roots(self):
        ts = msprime.simulate(20, recombination_rate=2, length=3, random_seed=42)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for root_threshold in [1, 2, 100]:
            tree = tskit.Tree(ts, root_threshold=root_threshold)
            copy = tree.copy()
            assert copy.roots == tree.roots
            assert copy.root_threshold == root_threshold
            while tree.next():
                copy = tree.copy()
                assert copy.roots == tree.roots
                assert copy.root_threshold == root_threshold
            copy = tree.copy()
            assert copy.roots == tree.roots
            assert copy.root_threshold == root_threshold

    def test_map_mutations(self):
        ts = msprime.simulate(5, random_seed=42)
        tree = ts.first()
        genotypes = np.zeros(5, dtype=np.int8)
        alleles = [str(j) for j in range(64)]
        ancestral_state, transitions = tree.map_mutations(genotypes, alleles)
        assert ancestral_state == "0"
        assert len(transitions) == 0
        for j in range(1, 64):
            genotypes[0] = j
            ancestral_state, transitions = tree.map_mutations(genotypes, alleles)
            assert ancestral_state == "0"
            assert len(transitions) == 1
        for j in range(64, 67):
            genotypes[0] = j
            with pytest.raises(ValueError):
                tree.map_mutations(genotypes, alleles)
        tree.map_mutations([0] * 5, alleles)
        tree.map_mutations(np.zeros(5, dtype=int), alleles)

    def test_sample_count_deprecated(self):
        ts = msprime.simulate(5, random_seed=42)
        with warnings.catch_warnings(record=True) as w:
            ts.trees(sample_counts=True)
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)

        with warnings.catch_warnings(record=True) as w:
            tskit.Tree(ts, sample_counts=False)
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)

    def test_node_edges(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=42)
        assert ts.num_trees > 2
        edge_table = ts.tables.edges
        for tree in ts.trees():
            nodes = set(tree.nodes())
            midpoint = sum(tree.interval) / 2
            # mapping = tree._node_edges()
            mapping = tree.edge_array
            for node, edge in enumerate(mapping):
                if node in nodes and tree.parent(node) != tskit.NULL:
                    edge_above_node = np.where(
                        np.logical_and.reduce(
                            (
                                edge_table.child == node,
                                edge_table.left < midpoint,
                                edge_table.right > midpoint,
                            )
                        )
                    )[0]
                    assert len(edge_above_node) == 1
                    assert edge_above_node[0] == edge
                else:
                    assert edge == tskit.NULL


class TestSiblings:
    def test_balanced_binary_tree(self):
        t = tskit.Tree.generate_balanced(num_leaves=3)
        assert t.has_single_root
        # Nodes 0 to 2 are leaves
        for u in range(2):
            assert t.is_leaf(u)
        assert t.siblings(0) == (3,)
        assert t.siblings(1) == (2,)
        assert t.siblings(2) == (1,)
        # Node 3 is the internal node
        assert t.is_internal(3)
        assert t.siblings(3) == (0,)
        # Node 4 is the root
        assert 4 == t.root
        assert t.siblings(4) == tuple()
        # Node 5 is the virtual root
        assert 5 == t.virtual_root
        assert t.siblings(5) == tuple()

    def test_star(self):
        t = tskit.Tree.generate_star(num_leaves=3)
        assert t.has_single_root
        # Nodes 0 to 2 are leaves
        for u in range(2):
            assert t.is_leaf(u)
        assert t.siblings(0) == (1, 2)
        assert t.siblings(1) == (0, 2)
        assert t.siblings(2) == (0, 1)
        # Node 3 is the root
        assert 3 == t.root
        assert t.siblings(3) == tuple()
        # Node 4 is the virtual root
        assert 4 == t.virtual_root
        assert t.siblings(4) == tuple()

    def test_multiroot_tree(self):
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        t = ts.decapitate(ts.node(5).time).first()
        assert t.has_multiple_roots
        # Nodes 0 to 3 are leaves
        assert t.siblings(0) == (1,)
        assert t.siblings(1) == (0,)
        assert t.siblings(2) == (3,)
        assert t.siblings(3) == (2,)
        # Nodes 4 and 5 are both roots
        assert 4 in t.roots
        assert t.siblings(4) == (5,)
        assert 5 in t.roots
        assert t.siblings(5) == (4,)
        # Node 7 is the virtual root
        assert 7 == t.virtual_root
        assert t.siblings(7) == tuple()

    @pytest.mark.parametrize("flag,expected", [(0, ()), (1, (2,))])
    def test_isolated_node(self, flag, expected):
        tables = tskit.Tree.generate_balanced(2, arity=2).tree_sequence.dump_tables()
        tables.nodes.add_row(flags=flag)  # Add node 3
        t = tables.tree_sequence().first()
        assert t.is_isolated(3)
        assert t.siblings(3) == expected


class TestNodeOrdering(HighLevelTestCase):
    """
    Verify that we can use any node ordering for internal nodes
    and get the same topologies.
    """

    num_random_permutations = 10

    def verify_tree_sequences_equal(self, ts1, ts2, approximate=False):
        assert ts1.get_num_trees() == ts2.get_num_trees()
        assert ts1.get_sample_size() == ts2.get_sample_size()
        assert ts1.get_num_nodes() == ts2.get_num_nodes()
        j = 0
        for r1, r2 in zip(ts1.edges(), ts2.edges()):
            assert r1.parent == r2.parent
            assert r1.child == r2.child
            if approximate:
                assert r1.left == pytest.approx(r2.left)
                assert r1.right == pytest.approx(r2.right)
            else:
                assert r1.left == r2.left
                assert r1.right == r2.right
            j += 1
        assert ts1.num_edges == j
        j = 0
        for n1, n2 in zip(ts1.nodes(), ts2.nodes()):
            assert n1.metadata == n2.metadata
            assert n1.population == n2.population
            if approximate:
                assert n1.time == pytest.approx(n2.time)
            else:
                assert n1.time == n2.time
            j += 1
        assert ts1.num_nodes == j

    def verify_random_permutation(self, ts):
        n = ts.sample_size
        node_map = {}
        for j in range(n):
            node_map[j] = j
        internal_nodes = list(range(n, ts.num_nodes))
        random.shuffle(internal_nodes)
        for j, node in enumerate(internal_nodes):
            node_map[n + j] = node
        other_tables = tskit.TableCollection(ts.sequence_length)
        # Insert the new nodes into the table.
        inv_node_map = {v: k for k, v in node_map.items()}
        for j in range(ts.num_nodes):
            node = ts.node(inv_node_map[j])
            other_tables.nodes.append(node)
        for e in ts.edges():
            other_tables.edges.append(
                e.replace(parent=node_map[e.parent], child=node_map[e.child])
            )
        for _ in range(ts.num_populations):
            other_tables.populations.add_row()
        other_tables.sort()
        other_ts = other_tables.tree_sequence()

        assert ts.get_num_trees() == other_ts.get_num_trees()
        assert ts.get_sample_size() == other_ts.get_sample_size()
        assert ts.get_num_nodes() == other_ts.get_num_nodes()
        j = 0
        for t1, t2 in zip(ts.trees(), other_ts.trees()):
            # Verify the topologies are identical. We do this by traversing
            # upwards to the root for every sample and checking if we map to
            # the correct node and time.
            for u in range(n):
                v_orig = u
                v_map = u
                while v_orig != tskit.NULL:
                    assert node_map[v_orig] == v_map
                    assert t1.get_time(v_orig) == t2.get_time(v_map)
                    v_orig = t1.get_parent(v_orig)
                    v_map = t2.get_parent(v_map)
                assert v_orig == tskit.NULL
                assert v_map == tskit.NULL
            j += 1
        assert j == ts.get_num_trees()
        # Verify we can dump this new tree sequence OK.
        with tempfile.TemporaryDirectory() as tempdir:
            temp_file = pathlib.Path(tempdir) / "tmp.trees"
            other_ts.dump(temp_file)
            ts3 = tskit.load(temp_file)
        self.verify_tree_sequences_equal(other_ts, ts3)
        nodes_file = io.StringIO()
        edges_file = io.StringIO()
        # Also verify we can read the text version.
        other_ts.dump_text(nodes=nodes_file, edges=edges_file, precision=14)
        nodes_file.seek(0)
        edges_file.seek(0)
        ts3 = tskit.load_text(nodes_file, edges_file)
        self.verify_tree_sequences_equal(other_ts, ts3, True)

    def test_single_locus(self):
        ts = msprime.simulate(7)
        for _ in range(self.num_random_permutations):
            self.verify_random_permutation(ts)

    def test_multi_locus(self):
        ts = msprime.simulate(20, recombination_rate=10)
        for _ in range(self.num_random_permutations):
            self.verify_random_permutation(ts)

    def test_nonbinary(self):
        ts = msprime.simulate(
            sample_size=20,
            recombination_rate=10,
            demographic_events=[
                msprime.SimpleBottleneck(time=0.5, population=0, proportion=1)
            ],
        )
        # Make sure this really has some non-binary nodes
        found = False
        for t in ts.trees():
            for u in t.nodes():
                if len(t.children(u)) > 2:
                    found = True
                    break
            if found:
                break
        assert found
        for _ in range(self.num_random_permutations):
            self.verify_random_permutation(ts)


def assert_trees_identical(t1, t2):
    assert t1.tree_sequence == t2.tree_sequence
    assert t1.index == t2.index
    assert_array_equal(t1.parent_array, t2.parent_array)
    assert_array_equal(t1.left_child_array, t2.left_child_array)
    assert_array_equal(t1.left_sib_array, t2.left_sib_array)
    assert_array_equal(t1.right_child_array, t2.right_child_array)
    assert_array_equal(t1.right_sib_array, t2.right_sib_array)


def assert_trees_equivalent(t1, t2):
    assert t1.tree_sequence == t2.tree_sequence
    assert t1.index == t2.index
    assert_array_equal(t1.parent_array, t2.parent_array)
    assert_array_equal(t1.edge_array, t2.edge_array)
    for u in range(t1.tree_sequence.num_nodes):
        # this isn't fully testing the data model, but that's done elsewhere
        assert sorted(t1.children(u)) == sorted(t2.children(u))


def assert_same_tree_different_order(t1, t2):
    assert t1.tree_sequence == t2.tree_sequence
    assert t1.index == t2.index
    assert np.all(t1.parent_array == t2.parent_array)
    assert not np.all(t1.left_child_array == t2.left_child_array)


def seek(tree, x):
    """
    Python implementation of the seek algorithm. Useful for developing
    tests.
    """
    L = tree.tree_sequence.sequence_length
    t_l, t_r = tree.interval
    if x < t_l:
        # |-----|-----|========|---------|
        # 0     x    t_l      t_r        L
        distance_left = t_l - x
        distance_right = L - t_r + x
    else:
        # |------|========|------|-------|
        # 0     t_l      t_r     x       L
        distance_right = x - t_r
        distance_left = t_l + L - x
    if distance_right <= distance_left:
        while not (tree.interval.left <= x < tree.interval.right):
            tree.next()
    else:
        while not (tree.interval.left <= x < tree.interval.right):
            tree.prev()


class TestSeekDirection:
    """
    Test if we seek in the correct direction according to our hueristics.
    """

    # 2.00┊       ┊   4   ┊   4   ┊   4   ┊
    #     ┊       ┊ ┏━┻┓  ┊  ┏┻━┓ ┊  ┏┻━┓ ┊
    # 1.00┊   3   ┊ ┃  3  ┊  3  ┃ ┊  3  ┃ ┊
    #     ┊ ┏━╋━┓ ┊ ┃ ┏┻┓ ┊ ┏┻┓ ┃ ┊ ┏┻┓ ┃ ┊
    # 0.00┊ 0 1 2 ┊ 0 1 2 ┊ 0 2 1 ┊ 0 1 2 ┊
    #     0       1       2       3       4
    @tests.cached_example
    def ts(self):
        return tsutil.all_trees_ts(3)

    def get_tree_pair(self):
        ts = self.ts()
        t1 = tskit.Tree(ts)
        t2 = tskit.Tree(ts)
        # # Note: for development we can monkeypatch in the Python implementation
        # # above like this:
        # import functools
        # t2.seek = functools.partial(seek, t2)
        return t1, t2

    @pytest.mark.parametrize("index", range(4))
    def test_index_from_different_directions(self, index):
        # Check that we get different orderings of the children arrays
        # for all trees when we go in different directions.
        t1, t2 = self.get_tree_pair()
        while t1.index != index:
            t1.next()
        while t2.index != index:
            t2.prev()
        assert_same_tree_different_order(t1, t2)

    @pytest.mark.parametrize("position", [0, 1, 2, 3])
    def test_seek_from_null(self, position):
        t1, t2 = self.get_tree_pair()
        t1.clear()
        t1.seek(position)
        t2.first()
        t2.seek(position, skip=False)
        assert_trees_identical(t1, t2)

    @pytest.mark.parametrize("position", [0, 1, 2, 3])
    def test_skip_from_null(self, position):
        t1, t2 = self.get_tree_pair()
        t1.clear()
        t1.seek(position)
        t2.first()
        t2.seek(position, skip=True)
        assert_trees_equivalent(t1, t2)

    @pytest.mark.parametrize("index", range(3))
    @pytest.mark.parametrize("skip", [False, True])
    def test_seek_next_tree(self, index, skip):
        t1, t2 = self.get_tree_pair()
        while t1.index != index:
            t1.next()
            t2.next()
        t1.next()
        t2.seek(index + 1, skip=skip)
        assert_trees_identical(t1, t2)

    @pytest.mark.parametrize("index", [3, 2, 1])
    @pytest.mark.parametrize("skip", [False, True])
    def test_seek_prev_tree(self, index, skip):
        t1, t2 = self.get_tree_pair()
        while t1.index != index:
            t1.prev()
            t2.prev()
        t1.prev()
        t2.seek(index - 1, skip=skip)
        assert_trees_identical(t1, t2)

    @pytest.mark.parametrize("skip", [False, True])
    def test_seek_1_from_0(self, skip):
        t1, t2 = self.get_tree_pair()
        t1.first()
        t1.next()
        t2.first()
        t2.seek(1, skip)
        assert_trees_identical(t1, t2)

    @pytest.mark.parametrize("skip", [False, True])
    def test_seek_1_5_from_0(self, skip):
        t1, t2 = self.get_tree_pair()
        t1.first()
        t1.next()
        t2.first()
        t2.seek(1.5, skip)
        assert_trees_identical(t1, t2)

    @pytest.mark.parametrize("skip", [False, True])
    def test_seek_1_5_from_1(self, skip):
        t1, t2 = self.get_tree_pair()
        for _ in range(2):
            t1.next()
            t2.next()
        t2.seek(1.5, skip)
        assert_trees_identical(t1, t2)

    @pytest.mark.parametrize("skip", [False, True])
    def test_seek_3_from_null(self, skip):
        t1, t2 = self.get_tree_pair()
        t1.last()
        t2.seek(3, skip)
        assert_trees_identical(t1, t2)

    @pytest.mark.parametrize("skip", [False, True])
    def test_seek_3_from_null_prev(self, skip):
        t1, t2 = self.get_tree_pair()
        t1.last()
        t1.prev()
        t2.seek(3, skip)
        t2.prev()
        assert_trees_identical(t1, t2)

    def test_seek_3_from_0(self):
        t1, t2 = self.get_tree_pair()
        t1.last()
        t2.first()
        t2.seek(3)
        assert_trees_identical(t1, t2)

    def test_skip_3_from_0(self):
        t1, t2 = self.get_tree_pair()
        t1.last()
        t2.first()
        t2.seek(3, True)
        assert_trees_equivalent(t1, t2)

    def test_skip_0_from_3(self):
        t1, t2 = self.get_tree_pair()
        t1.last()
        t1.first()
        t2.last()
        t2.seek(0, True)
        assert_trees_equivalent(t1, t2)

    def test_seek_0_from_3(self):
        t1, t2 = self.get_tree_pair()
        t1.last()
        t1.first()
        t2.last()
        t2.seek(0)
        assert_trees_identical(t1, t2)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_seek_mid_null_and_middle(self, ts):
        breakpoints = ts.breakpoints(as_array=True)
        mid = breakpoints[:-1] + np.diff(breakpoints) / 2
        for index, x in enumerate(mid[:-1]):
            t1 = tskit.Tree(ts)
            t1.seek(x)
            # Also seek to this point manually to make sure we're not
            # reusing the seek from null under the hood.
            t2 = tskit.Tree(ts)
            if index <= ts.num_trees / 2:
                while t2.index != index:
                    t2.next()
            else:
                while t2.index != index:
                    t2.prev()
            assert_trees_equivalent(t1, t2)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_seek_skip_middle(self, ts):
        breakpoints = ts.breakpoints(as_array=True)
        mid = breakpoints[:-1] + np.diff(breakpoints) / 2
        for _, x in enumerate(mid[:-1]):
            t1 = tskit.Tree(ts)
            t1.seek(x, skip=False)
            t2 = tskit.Tree(ts)
            t2.seek(x, skip=True)
            assert_trees_equivalent(t1, t2)

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_seek_last_then_prev(self, ts):
        t1 = tskit.Tree(ts)
        t1.seek(ts.sequence_length - 0.00001)
        assert t1.index == ts.num_trees - 1
        t2 = tskit.Tree(ts)
        t2.prev()
        assert_trees_identical(t1, t2)
        t1.prev()
        t2.prev()
        assert_trees_identical(t1, t2)


class TestSeek:
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_new_seek_breakpoints(self, ts):
        breakpoints = ts.breakpoints(as_array=True)
        for index, left in enumerate(breakpoints[:-1]):
            tree = tskit.Tree(ts)
            tree.seek(left)
            assert tree.index == index

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_new_seek_mid(self, ts):
        breakpoints = ts.breakpoints(as_array=True)
        mid = breakpoints[:-1] + np.diff(breakpoints) / 2
        for index, left in enumerate(mid[:-1]):
            tree = tskit.Tree(ts)
            tree.seek(left)
            assert tree.index == index

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_same_seek_breakpoints(self, ts):
        breakpoints = ts.breakpoints(as_array=True)
        tree = tskit.Tree(ts)
        for index, left in enumerate(breakpoints[:-1]):
            tree.seek(left)
            assert tree.index == index

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_new_seek_breakpoints_reversed(self, ts):
        breakpoints = ts.breakpoints(as_array=True)
        for index, left in reversed(list(enumerate(breakpoints[:-1]))):
            tree = tskit.Tree(ts)
            tree.seek(left)
            assert tree.index == index

    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_same_seek_breakpoints_reversed(self, ts):
        breakpoints = ts.breakpoints(as_array=True)
        tree = tskit.Tree(ts)
        for index, left in reversed(list(enumerate(breakpoints[:-1]))):
            tree.seek(left)
            assert tree.index == index

    def test_example(self):
        L = 10
        ts = msprime.simulate(10, recombination_rate=3, length=L, random_seed=42)
        assert ts.num_trees > 5
        same_tree = tskit.Tree(ts)
        for j in range(L):
            for tree in [same_tree, tskit.Tree(ts)]:
                tree.seek(j)
                index = tree.index
                assert tree.interval.left <= j < tree.interval.right
                tree.seek(tree.interval.left)
                assert tree.index == index
                if tree.interval.right < L:
                    tree.seek(tree.interval.right)
                    assert tree.index == index + 1
        for j in reversed(range(L)):
            for tree in [same_tree, tskit.Tree(ts)]:
                tree.seek(j)
                assert tree.interval.left <= j < tree.interval.right

    def test_errors(self, ts_fixture):
        L = ts_fixture.sequence_length
        tree = tskit.Tree(ts_fixture)
        for bad_position in [-1, L, L + 1, -L]:
            with pytest.raises(ValueError):
                tree.seek(bad_position)


class SimpleContainersMixin:
    """
    Tests for the SimpleContainer classes.
    """

    def test_equality(self):
        c1, c2 = self.get_instances(2)
        assert c1 == c1
        assert not (c1 == c2)
        assert not (c1 != c1)
        assert c1 != c2
        (c3,) = self.get_instances(1)
        assert c1 == c3
        assert not (c1 != c3)

    def test_repr(self):
        (c,) = self.get_instances(1)
        assert len(repr(c)) > 0


class SimpleContainersWithMetadataMixin:
    """
    Tests for the SimpleContainerWithMetadata classes.
    """

    def test_metadata(self):
        # Test decoding
        instances = self.get_instances(5)
        for j, inst in enumerate(instances):
            assert inst.metadata == ("x" * j) + "decoded"

        # Decoder doesn't effect equality
        (inst,) = self.get_instances(1)
        (inst2,) = self.get_instances(1)
        assert inst == inst2
        inst._metadata = "different"
        assert inst != inst2

    def test_decoder_run_once(self):
        # For a given instance, the decoded metadata should be cached, with the decoder
        # called once
        (inst,) = self.get_instances(1)
        times_run = 0

        # Hack in a tracing decoder
        def decoder(m):
            nonlocal times_run
            times_run += 1
            return m.decode() + "decoded"

        inst._metadata_decoder = decoder
        assert times_run == 0
        _ = inst.metadata
        assert times_run == 1
        _ = inst.metadata
        assert times_run == 1


class TestIndividualContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Individual(
                id=j,
                flags=j,
                location=[j],
                parents=[j],
                nodes=[j],
                metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestNodeContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Node(
                id=j,
                flags=j,
                time=j,
                population=j,
                individual=j,
                metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestEdgeContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Edge(
                left=j,
                right=j,
                parent=j,
                child=j,
                metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
                id=j,
            )
            for j in range(n)
        ]


class TestSiteContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Site(
                id=j,
                position=j,
                ancestral_state="A" * j,
                mutations=TestMutationContainer().get_instances(j),
                metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestMutationContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Mutation(
                id=j,
                site=j,
                node=j,
                time=j,
                derived_state="A" * j,
                parent=j,
                metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]

    def test_nan_equality(self):
        a = tskit.Mutation(
            id=42,
            site=42,
            node=42,
            time=UNKNOWN_TIME,
            derived_state="A" * 42,
            parent=42,
            metadata=b"x" * 42,
            metadata_decoder=lambda m: m.decode() + "decoded",
        )
        b = tskit.Mutation(
            id=42,
            site=42,
            node=42,
            derived_state="A" * 42,
            parent=42,
            metadata=b"x" * 42,
            metadata_decoder=lambda m: m.decode() + "decoded",
        )
        c = tskit.Mutation(
            id=42,
            site=42,
            node=42,
            time=math.nan,
            derived_state="A" * 42,
            parent=42,
            metadata=b"x" * 42,
            metadata_decoder=lambda m: m.decode() + "decoded",
        )
        assert a == a
        assert a == b
        assert not (a == c)
        assert not (b == c)
        assert not (a != a)
        assert not (a != b)
        assert a != c
        assert c != c
        assert not (c == c)


class TestMigrationContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Migration(
                id=j,
                left=j,
                right=j,
                node=j,
                source=j,
                dest=j,
                time=j,
                metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestPopulationContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Population(
                id=j,
                metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestProvenanceContainer(SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Provenance(id=j, timestamp="x" * j, record="y" * j) for j in range(n)
        ]


class TestEdgesetContainer(SimpleContainersMixin):
    def get_instances(self, n):
        return [tskit.Edgeset(left=j, right=j, parent=j, children=j) for j in range(n)]


class TestContainersAppend:
    def test_containers_append(self, ts_fixture):
        """
        Test that the containers work with `Table.append`
        """
        tables = ts_fixture.dump_tables()
        tables.clear(clear_provenance=True)
        for table_name in tskit.TABLE_NAMES:
            table = getattr(tables, table_name)
            for i in range(len(getattr(ts_fixture.tables, table_name))):
                table.append(getattr(ts_fixture, table_name[:-1])(i))
        ts_fixture.tables.assert_equals(tables)


class TestTskitConversionOutput(unittest.TestCase):
    """
    Tests conversion output to ensure it is correct.
    """

    @classmethod
    def setUpClass(cls):
        ts = msprime.simulate(
            length=1,
            recombination_rate=2,
            mutation_rate=2,
            random_seed=1,
            migration_matrix=[[0, 1], [1, 0]],
            population_configurations=[
                msprime.PopulationConfiguration(5) for _ in range(2)
            ],
            record_migrations=True,
        )
        assert ts.num_migrations > 0
        cls._tree_sequence = tsutil.insert_random_ploidy_individuals(ts)

    def test_macs(self):
        output = self._tree_sequence.to_macs().splitlines()
        assert output[0].startswith("COMMAND:")
        assert output[1].startswith("SEED:")
        assert len(output) == 2 + self._tree_sequence.get_num_mutations()
        n = self._tree_sequence.get_sample_size()
        m = self._tree_sequence.get_sequence_length()
        sites = list(self._tree_sequence.sites())
        haplotypes = list(self._tree_sequence.haplotypes())
        for site_id, line in enumerate(output[2:]):
            splits = line.split()
            assert splits[0] == "SITE:"
            assert int(splits[1]) == site_id
            position = sites[site_id].position / m
            self.assertAlmostEqual(float(splits[2]), position)
            col = splits[4]
            assert len(col) == n
            for j in range(n):
                assert col[j] == haplotypes[j][site_id]

    def test_macs_error(self):
        tables = tskit.TableCollection(1)
        tables.sites.add_row(position=0.5, ancestral_state="A")
        tables.nodes.add_row(time=1, flags=tskit.NODE_IS_SAMPLE)
        tables.mutations.add_row(node=0, site=0, derived_state="FOO")
        ts = tables.tree_sequence()
        with pytest.raises(
            ValueError, match="macs output only supports single letter alleles"
        ):
            ts.to_macs()


class TestTreeSequenceGetSite:
    """
    Tests for getting Site objects from a TreeSequence object
    by specifying the position.
    """

    def get_example_ts_discrete_coordinates(self):
        tables = tskit.TableCollection(sequence_length=10)
        tables.sites.add_row(position=3, ancestral_state="A")
        tables.sites.add_row(position=5, ancestral_state="C")
        tables.sites.add_row(position=7, ancestral_state="G")
        return tables.tree_sequence()

    def get_example_ts_continuous_coordinates(self):
        tables = tskit.TableCollection(sequence_length=10)
        tables.sites.add_row(position=0.5, ancestral_state="A")
        tables.sites.add_row(position=6.2, ancestral_state="C")
        tables.sites.add_row(position=8.3, ancestral_state="T")
        return tables.tree_sequence()

    def get_example_ts_without_sites(self):
        tables = tskit.TableCollection(sequence_length=10)
        return tables.tree_sequence()

    @pytest.mark.parametrize("id_", [0, 1, 2])
    def test_site_id(self, id_):
        ts = self.get_example_ts_discrete_coordinates()
        site = ts.site(id_)
        assert site.id == id_

    @pytest.mark.parametrize("position", [3, 5, 7])
    def test_position_discrete_coordinates(self, position):
        ts = self.get_example_ts_discrete_coordinates()
        site = ts.site(position=position)
        assert site.position == position

    @pytest.mark.parametrize("position", [0.5, 6.2, 8.3])
    def test_position_continuous_coordinates(self, position):
        ts = self.get_example_ts_continuous_coordinates()
        site = ts.site(position=position)
        assert site.position == position

    @pytest.mark.parametrize("position", [0, 2.999999999, 5.000000001, 9])
    def test_position_not_found(self, position):
        with pytest.raises(ValueError, match=r"There is no site at position"):
            ts = self.get_example_ts_discrete_coordinates()
            ts.site(position=position)

    @pytest.mark.parametrize(
        "position",
        [
            np.array([3], dtype=float)[0],
            np.array([3], dtype=int)[0],
            decimal.Decimal(3),
        ],
    )
    def test_position_good_type(self, position):
        ts = self.get_example_ts_discrete_coordinates()
        ts.site(position=position)

    def test_position_not_scalar(self):
        with pytest.raises(
            ValueError, match="Position must be provided as a scalar value."
        ):
            ts = self.get_example_ts_discrete_coordinates()
            ts.site(position=[1, 4, 8])

    @pytest.mark.parametrize("position", [-1, 10, 11])
    def test_position_out_of_bounds(self, position):
        with pytest.raises(
            ValueError,
            match="Position is beyond the coordinates defined by sequence length.",
        ):
            ts = self.get_example_ts_discrete_coordinates()
            ts.site(position=position)

    def test_query_position_siteless_ts(self):
        with pytest.raises(ValueError, match=r"There is no site at position"):
            ts = self.get_example_ts_without_sites()
            ts.site(position=1)

    def test_site_id_and_position_are_none(self):
        with pytest.raises(TypeError, match="Site id or position must be provided."):
            ts = self.get_example_ts_discrete_coordinates()
            ts.site(None, position=None)

    def test_site_id_and_position_are_specified(self):
        with pytest.raises(
            TypeError, match="Only one of site id or position needs to be provided."
        ):
            ts = self.get_example_ts_discrete_coordinates()
            ts.site(0, position=3)


def num_lineages_definition(tree, t):
    lineages = 0
    for u in tree.nodes():
        v = tree.parent(u)
        if v != tskit.NULL:
            if tree.time(u) <= t < tree.time(v):
                lineages += 1
    return lineages


class TestNumLineages:
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_tree_midpoint_definition(self, ts):
        t = 0
        if ts.num_nodes > 0:
            t = np.max(ts.tables.nodes.time) / 2
        tree = ts.first()
        assert tree.num_lineages(t) == num_lineages_definition(tree, t)

    @pytest.mark.parametrize("t", [-np.inf, np.inf, np.nan])
    def test_nonfinite_time(self, t):
        tree = tskit.Tree.generate_balanced(2)
        with pytest.raises(tskit.LibraryError, match="NONFINITE"):
            tree.num_lineages(t)

    @pytest.mark.parametrize("t", [1, 1.0, np.array([1.0])[0]])
    def test_number_types(self, t):
        tree = tskit.Tree.generate_balanced(2)
        assert tree.num_lineages(t) == 0

    # 2.00┊        12         ┊
    #     ┊   ┏━━━━━╋━━━━━┓   ┊
    # 1.00┊   9    10    11   ┊
    #     ┊ ┏━╋━┓ ┏━╋━┓ ┏━╋━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    @pytest.mark.parametrize(
        ["t", "expected"],
        [
            (-0.00001, 0),
            (0, 9),
            (0.0000001, 9),
            (0.99999, 9),
            (1, 3),
            (1.999999, 3),
            (2, 0),
            (2.000001, 0),
        ],
    )
    def test_balanced_ternary(self, t, expected):
        tree = tskit.Tree.generate_balanced(9, arity=3)
        assert tree.num_lineages(t) == expected

    # 3.00┊            15     ┊
    #     ┊          ┏━━┻━┓   ┊
    # 2.00┊   11     ┃   14   ┊
    #     ┊  ┏━┻━┓   ┃  ┏━┻┓  ┊
    # 1.00┊  9  10  12  ┃ 13  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    @pytest.mark.parametrize(
        ["t", "expected"],
        [
            (-0.00001, 0),
            (0, 9),
            (0.0000001, 9),
            (0.99999, 9),
            (1, 5),
            (1.999999, 5),
            (2, 2),
            (2.000001, 2),
            (3.00000, 0),
            (5.00000, 0),
        ],
    )
    def test_multiroot_different_times(self, t, expected):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        ts = tables.tree_sequence()
        tree = ts.first()
        assert tree.num_lineages(t) == expected

    # 4.00┊   8       ┊
    #     ┊ ┏━┻━┓     ┊
    # 3.00┊ 0   7     ┊
    #     ┊   ┏━┻━┓   ┊
    # 2.00┊   1   6   ┊
    #     ┊     ┏━┻┓  ┊
    # 1.00┊     2  5  ┊
    #     ┊       ┏┻┓ ┊
    # 0.00┊       3 4 ┊
    #     0           1
    @pytest.mark.parametrize(
        ["t", "expected"],
        [
            (-0.00001, 0),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (3, 2),
            (4, 0),
        ],
    )
    def test_comb_different_leaf_times(self, t, expected):
        tables = tskit.Tree.generate_comb(5).tree_sequence.dump_tables()
        time = tables.nodes.time
        time[2] = 1
        time[1] = 2
        time[0] = 3
        tables.nodes.time = time
        ts = tables.tree_sequence()
        tree = ts.first()
        assert tree.num_lineages(t) == expected

    @pytest.mark.parametrize(
        ["t", "expected"],
        [
            (-0.00001, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
        ],
    )
    def test_missing_data_different_times(self, t, expected):
        tables = tskit.TableCollection(1)
        for j in range(3):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=j)
        ts = tables.tree_sequence()
        tree = ts.first()
        assert tree.num_lineages(t) == expected


@pytest.fixture
def struct_metadata_ts(ts_fixture):
    schema = metadata.MetadataSchema(
        {
            "codec": "struct",
            "type": "object",
            "properties": {
                "id": {"type": "integer", "binaryFormat": "i"},
                "name": {"type": "string", "binaryFormat": "10s"},
                "value": {"type": "number", "binaryFormat": "d"},
                "active": {"type": "boolean", "binaryFormat": "?"},
            },
        }
    )
    tables = ts_fixture.dump_tables()
    for table_name in TestStructuredNumpyMetadata.metadata_tables:
        table = getattr(tables, table_name)
        table.metadata_schema = schema
        table_copy = table.copy()
        table.clear()
        for j, row in enumerate(table_copy):
            table.append(
                row.replace(
                    metadata={"id": j, "name": "name", "value": 1.0, "active": True}
                )
            )
    return tables.tree_sequence()


class TestStructuredNumpyMetadata:
    metadata_tables = [
        "nodes",
        "edges",
        "sites",
        "mutations",
        "migrations",
        "individuals",
        "populations",
    ]

    @pytest.mark.parametrize("table_name", metadata_tables)
    def test_not_implemented_json(self, table_name, ts_fixture):
        with pytest.raises(NotImplementedError):
            getattr(ts_fixture, f"{table_name}_metadata")

    @pytest.mark.parametrize("table_name", metadata_tables)
    def test_array_attr_properties(self, struct_metadata_ts, table_name):
        ts = struct_metadata_ts
        attr_name = f"{table_name}_metadata"
        a = getattr(ts, attr_name)
        assert isinstance(a, np.ndarray)
        with pytest.raises(AttributeError):
            setattr(ts, attr_name, None)
        with pytest.raises(AttributeError):
            delattr(ts, attr_name)
        with pytest.raises(ValueError, match="read-only"):
            a[:] = 1

    @pytest.mark.parametrize("table_name", metadata_tables)
    def test_array_contents(self, struct_metadata_ts, table_name):
        ts = struct_metadata_ts
        attr_name = f"{table_name}_metadata"
        a = getattr(ts, attr_name)
        assert len(a) == getattr(ts, f"num_{table_name}")
        for j, row in enumerate(a):
            assert row["id"] == j
            assert row["name"] == b"name"
            assert row["value"] == 1.0
            assert row["active"]

    @pytest.mark.parametrize("table_name", metadata_tables)
    def test_error_if_no_schema(self, table_name):
        ts = msprime.simulate(10)
        with pytest.raises(NotImplementedError):
            getattr(ts, f"{table_name}_metadata")


class TestIndividualsNodes:
    def test_basic_individuals_nodes(self, tmp_path):
        # Create a basic tree sequence with two individuals
        tables = tskit.TableCollection(sequence_length=100)
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        ts = tables.tree_sequence()

        result = ts.individuals_nodes
        assert result.shape == (2, 2)
        assert_array_equal(result, [[0, 1], [2, 3]])

    def test_variable_ploidy(self, tmp_path):
        tables = tskit.TableCollection(sequence_length=100)
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")  # Diploid
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")  # Haploid
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")  # Triploid

        # Diploid individual
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)

        # Haploid individual
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)

        # Triploid individual
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=2)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=2)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=2)

        ts = tables.tree_sequence()

        result = ts.individuals_nodes

        assert result.shape == (3, 3)

        expected = np.array(
            [[0, 1, -1], [2, -1, -1], [3, 4, 5]]  # Diploid  # Haploid  # Triploid
        )
        assert_array_equal(result, expected)

    def test_no_individuals(self):
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.individuals_nodes
        expected = np.array([], dtype=np.int32).reshape(0, 0)
        assert result.shape == (0, 0)
        assert_array_equal(result, expected)

    def test_no_nodes_with_individuals(self):
        tables = tskit.TableCollection(sequence_length=100)
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        # Node without individual reference
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.individuals_nodes
        expected = np.array([[]])
        assert result.shape == (1, 0)
        assert_array_equal(result, expected)

    def test_individual_with_no_nodes(self):
        tables = tskit.TableCollection(sequence_length=100)
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        # Only add nodes for first individual
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        ts = tables.tree_sequence()

        result = ts.individuals_nodes
        expected = np.array([[0], [-1]])
        assert result.shape == (2, 1)
        assert_array_equal(result, expected)

    def test_mixed_sample_status(self):
        tables = tskit.TableCollection(sequence_length=100)
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=0, time=0, individual=0)
        ts = tables.tree_sequence()

        result = ts.individuals_nodes
        expected = np.array([[0, 1]])
        assert result.shape == (1, 2)
        assert_array_equal(result, expected)


class TestRaggedArrays:
    @pytest.mark.skipif(not _tskit.HAS_NUMPY_2, reason="Requires NumPy 2.0 or higher")
    @pytest.mark.parametrize("num_rows", [0, 1, 100])
    @pytest.mark.parametrize("column", ["ancestral_state", "derived_state"])
    def test_site_ancestral_state(self, num_rows, column):
        tables = tskit.TableCollection(sequence_length=100)
        rng = random.Random(42)
        for i in range(num_rows):
            state_length = rng.randint(0, 10)
            state = "".join(
                chr(rng.randint(0x1F300, 0x1F6FF)) for _ in range(state_length)
            )
            if column == "ancestral_state":
                tables.sites.add_row(position=i, ancestral_state=state)
            elif column == "derived_state":
                tables.nodes.add_row()
                tables.sites.add_row(position=i, ancestral_state="A")
                tables.mutations.add_row(site=i, node=0, derived_state=state)
        ts = tables.tree_sequence()
        a = getattr(
            ts,
            (
                "sites_ancestral_state"
                if column == "ancestral_state"
                else "mutations_derived_state"
            ),
        )
        assert isinstance(a, np.ndarray)
        assert a.shape == (num_rows,)
        assert a.dtype == np.dtype("T")
        assert a.size == num_rows

        # Check that the value is cached
        assert a is getattr(
            ts,
            (
                "sites_ancestral_state"
                if column == "ancestral_state"
                else "mutations_derived_state"
            ),
        )

        for state, row in itertools.zip_longest(
            a, ts.sites() if column == "ancestral_state" else ts.mutations()
        ):
            assert state == getattr(row, column)

    @pytest.mark.skipif(not _tskit.HAS_NUMPY_2, reason="Requires NumPy 2.0 or higher")
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_equality_sites_ancestral_state(self, ts):
        assert_array_equal(
            ts.sites_ancestral_state, [site.ancestral_state for site in ts.sites()]
        )

    @pytest.mark.skipif(not _tskit.HAS_NUMPY_2, reason="Requires NumPy 2.0 or higher")
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_equality_mutations_derived_state(self, ts):
        assert_array_equal(
            ts.mutations_derived_state,
            [mutation.derived_state for mutation in ts.mutations()],
        )

    @pytest.mark.skipif(not _tskit.HAS_NUMPY_2, reason="Requires NumPy 2.0 or higher")
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_equality_mutations_inherited_state(self, ts):
        assert_array_equal(
            ts.mutations_inherited_state,
            [mutation.inherited_state for mutation in ts.mutations()],
        )

    @pytest.mark.skipif(not _tskit.HAS_NUMPY_2, reason="Requires NumPy 2.0 or higher")
    @pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
    def test_mutations_inherited_state(self, ts):
        inherited_state = ts.mutations_inherited_state
        assert len(inherited_state) == ts.num_mutations
        assert isinstance(inherited_state, np.ndarray)
        assert inherited_state.shape == (ts.num_mutations,)
        assert inherited_state.dtype == np.dtype("T")
        assert inherited_state.size == ts.num_mutations

        for mut in ts.mutations():
            state0 = ts.site(mut.site).ancestral_state
            if mut.parent != -1:
                state0 = ts.mutation(mut.parent).derived_state
            assert state0 == inherited_state[mut.id]

        # Test caching - second access should return the same object
        inherited_state2 = ts.mutations_inherited_state
        assert inherited_state is inherited_state2

    @pytest.mark.skipif(_tskit.HAS_NUMPY_2, reason="Test only on Numpy 1.X")
    @pytest.mark.parametrize(
        "column",
        [
            "sites_ancestral_state",
            "mutations_derived_state",
            "mutations_inherited_state",
        ],
    )
    def test_ragged_array_not_supported(self, column):
        tables = tskit.TableCollection(sequence_length=100)
        ts = tables.tree_sequence()

        with pytest.raises(
            RuntimeError,
            match="requires numpy 2.0",
        ):
            getattr(ts, column)

    @pytest.mark.skipif(_tskit.HAS_NUMPY_2, reason="Test only on Numpy 1.X")
    def test_tables_emits_warning(self):
        tables = tskit.TableCollection(sequence_length=1)
        ts = tables.tree_sequence()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            result = ts.tables

        assert isinstance(result, tskit.TableCollection)
        assert len(caught) == 1
        warning = caught[0]
        assert warning.category is UserWarning
        assert "Immutable table views require tskit" in str(warning.message)


class TestSampleNodesByPloidy:
    @pytest.mark.parametrize(
        "n_samples,ploidy,expected",
        [
            (6, 2, np.array([[0, 1], [2, 3], [4, 5]])),  # Basic diploid
            (9, 3, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),  # Triploid
            (5, 1, np.array([[0], [1], [2], [3], [4]])),  # Ploidy of 1
            (4, 4, np.array([[0, 1, 2, 3]])),  # Ploidy equals number of samples
        ],
    )
    def test_various_ploidy_scenarios(self, n_samples, ploidy, expected):
        tables = tskit.TableCollection(sequence_length=100)
        for _ in range(n_samples):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.sample_nodes_by_ploidy(ploidy)
        expected_shape = (n_samples // ploidy, ploidy)
        assert result.shape == expected_shape
        assert_array_equal(result, expected)

    def test_mixed_sample_status(self):
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=0, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=0, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.sample_nodes_by_ploidy(2)
        assert result.shape == (2, 2)
        expected = np.array([[0, 2], [4, 5]])
        assert_array_equal(result, expected)

    def test_no_sample_nodes(self):
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.add_row(flags=0, time=0)
        tables.nodes.add_row(flags=0, time=0)
        ts = tables.tree_sequence()

        with pytest.raises(ValueError, match="No sample nodes in tree sequence"):
            ts.sample_nodes_by_ploidy(2)

    def test_not_multiple_of_ploidy(self):
        tables = tskit.TableCollection(sequence_length=100)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        with pytest.raises(ValueError, match="not a multiple of ploidy"):
            ts.sample_nodes_by_ploidy(2)

    def test_with_existing_individuals(self):
        tables = tskit.TableCollection(sequence_length=100)
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        # Add nodes with individual references but in a different order
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)

        ts = tables.tree_sequence()
        result = ts.sample_nodes_by_ploidy(2)
        expected = np.array([[0, 1], [2, 3]])
        assert_array_equal(result, expected)
        ind_nodes = ts.individuals_nodes
        assert not np.array_equal(result, ind_nodes)

    def test_different_node_flags(self):
        tables = tskit.TableCollection(sequence_length=100)
        OTHER_FLAG1 = 1 << 1
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=OTHER_FLAG1, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE | OTHER_FLAG1, time=0)
        tables.nodes.add_row()
        ts = tables.tree_sequence()
        result = ts.sample_nodes_by_ploidy(2)
        assert result.shape == (1, 2)
        assert_array_equal(result, np.array([[0, 2]]))


class TestMapToVcfModel:
    def test_no_individuals_default_ploidy(self):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        assert ts.num_individuals == 0

        # Default ploidy should be 1
        result = ts.map_to_vcf_model()
        assert isinstance(result, tskit.VcfModelMapping)
        assert result.individuals_nodes.shape == (4, 1)
        for i in range(4):
            assert result.individuals_nodes[i, 0] == i
        assert result.individuals_name.shape == (4,)
        for i in range(4):
            assert result.individuals_name[i] == f"tsk_{i}"

        with pytest.raises(
            ValueError,
            match="Cannot include non-sample nodes when individuals are not present",
        ):
            ts.map_to_vcf_model(include_non_sample_nodes=True)

    def test_no_individuals_custom_ploidy(self):
        ts = tskit.Tree.generate_balanced(6).tree_sequence
        assert ts.num_individuals == 0

        # Use ploidy = 2
        result = ts.map_to_vcf_model(ploidy=2)
        assert isinstance(result, tskit.VcfModelMapping)
        assert result.individuals_nodes.shape == (3, 2)
        for i in range(3):
            assert result.individuals_nodes[i, 0] == i * 2
            assert result.individuals_nodes[i, 1] == i * 2 + 1
        assert result.individuals_name.shape == (3,)
        for i in range(3):
            assert result.individuals_name[i] == f"tsk_{i}"

    def test_no_individuals_uneven_ploidy(self):
        ts = tskit.Tree.generate_balanced(5).tree_sequence
        # This tree sequence has no individuals
        assert ts.num_individuals == 0

        # 5 samples cannot be evenly divided into ploidy=2
        with pytest.raises(ValueError, match="not a multiple"):
            ts.map_to_vcf_model(ploidy=2)

    def test_with_individuals(self):
        ts = msprime.sim_ancestry(
            5,
            random_seed=42,
        )
        result = ts.map_to_vcf_model()
        assert isinstance(result, tskit.VcfModelMapping)
        assert result.individuals_nodes.shape == (5, 2)
        assert np.array_equal(
            result.individuals_nodes,
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        )
        assert result.individuals_name.shape == (5,)
        for i in range(5):
            assert result.individuals_name[i] == f"tsk_{i}"

    def test_with_individuals_and_ploidy_error(self):
        tables = tskit.TableCollection(1.0)
        tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        ts = tables.tree_sequence()

        with pytest.raises(ValueError, match="Cannot specify ploidy when individuals"):
            ts.map_to_vcf_model(ploidy=2)

    def test_specific_individuals(self):
        tables = tskit.TableCollection(1.0)
        # Create 5 individuals with varying ploidy
        for i in range(5):
            tables.individuals.add_row()
            # Individuals have ploidy i+1
            for _ in range(i + 1):
                tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=i)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model(individuals=[1, 3])
        assert isinstance(result, tskit.VcfModelMapping)
        # Individual 1 has ploidy 2, individual 3 has ploidy 4
        assert result.individuals_nodes.shape == (2, 5)
        assert np.array_equal(result.individuals_nodes[0], [1, 2, -1, -1, -1])
        assert np.array_equal(result.individuals_nodes[1], [6, 7, 8, 9, -1])

        assert result.individuals_name.shape == (2,)
        assert result.individuals_name[0] == "tsk_1"
        assert result.individuals_name[1] == "tsk_3"

    def test_individual_with_no_nodes(self):
        tables = tskit.TableCollection(1.0)
        # Individual with no nodes
        tables.individuals.add_row()
        # Individual with nodes
        tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model()
        assert result.individuals_nodes.shape == (2, 1)
        assert np.array_equal(result.individuals_nodes, [[-1], [0]])

    def test_individual_with_no_nodes_only(self):
        tables = tskit.TableCollection(1.0)
        # Individual with no nodes
        tables.individuals.add_row()
        # Individual with nodes
        tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model(individuals=[0])
        assert result.individuals_nodes.shape == (1, 1)
        assert np.array_equal(result.individuals_nodes, [[-1]])

    def test_invalid_individual_id(self):
        tables = tskit.TableCollection(1.0)
        tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        ts = tables.tree_sequence()

        with pytest.raises(ValueError, match="Invalid individual ID"):
            ts.map_to_vcf_model(individuals=[-1])

        with pytest.raises(ValueError, match="Invalid individual ID"):
            ts.map_to_vcf_model(individuals=[1])

    def test_mixed_sample_non_sample_ordering(self):
        tables = tskit.TableCollection(1.0)
        tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=0, time=0, individual=0)  # Non-sample node
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=0, time=0, individual=0)  # Non-sample node
        tables.individuals.add_row()
        tables.nodes.add_row(flags=0, time=0, individual=1)  # Non-sample node
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model()
        assert result.individuals_nodes.shape == (2, 4)
        assert np.array_equal(
            result.individuals_nodes,
            np.array([[0, 2, -1, -1], [5, -1, -1, -1]]),
        )

        result = ts.map_to_vcf_model(include_non_sample_nodes=True)
        assert result.individuals_nodes.shape == (2, 4)
        assert np.array_equal(
            result.individuals_nodes,
            np.array([[0, 1, 2, 3], [4, 5, -1, -1]]),
        )

    def test_samples_without_individuals_warning(self):
        tables = tskit.TableCollection(1.0)
        tables.individuals.add_row()
        # Node with individual
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        # Node without individual
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=tskit.NULL)
        ts = tables.tree_sequence()

        with warnings.catch_warnings(record=True) as w:
            ts.map_to_vcf_model()
            assert len(w) == 1
            assert "At least one sample node does not have an individual ID" in str(
                w[0].message
            )

    def test_metadata_key_for_names(self):
        tables = tskit.TableCollection(1.0)

        # Add individuals with metadata
        tables.individuals.metadata_schema = tskit.MetadataSchema(
            {
                "codec": "json",
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        )
        tables.individuals.add_row(metadata={"name": "ind1"})
        tables.individuals.add_row(metadata={"name": "ind2"})

        # Add nodes
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model(name_metadata_key="name")
        assert result.individuals_name.shape == (2,)
        assert result.individuals_name[0] == "ind1"
        assert result.individuals_name[1] == "ind2"

    def test_custom_individual_names(self):
        tables = tskit.TableCollection(1.0)
        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        ts = tables.tree_sequence()

        custom_names = ["individual_A", "individual_B"]
        result = ts.map_to_vcf_model(individual_names=custom_names)
        assert result.individuals_name.shape == (2,)
        assert result.individuals_name[0] == "individual_A"
        assert result.individuals_name[1] == "individual_B"

    def test_name_conflict_error(self):
        tables = tskit.TableCollection(1.0)
        ts = tables.tree_sequence()
        with pytest.raises(
            ValueError,
            match="Cannot specify both name_metadata_key and individual_names",
        ):
            ts.map_to_vcf_model(
                name_metadata_key="name", individual_names=["custom_name"]
            )

    def test_name_count_mismatch_error(self):
        tables = tskit.TableCollection(1.0)
        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        ts = tables.tree_sequence()

        with pytest.raises(
            ValueError, match="number of individuals does not match the number of names"
        ):
            ts.map_to_vcf_model(individual_names=["only_one_name"])

    def test_all_individuals_no_nodes(self):
        tables = tskit.TableCollection(1.0)
        tables.individuals.add_row()
        tables.individuals.add_row()
        ts = tables.tree_sequence()
        result = ts.map_to_vcf_model()
        assert result.individuals_nodes.shape == (2, 0)

    def test_position_transform_default_and_custom(self):
        tables = tskit.TableCollection(10.6)
        tables.sites.add_row(position=1.3, ancestral_state="A")
        tables.sites.add_row(position=5.7, ancestral_state="T")
        tables.sites.add_row(position=9.9, ancestral_state="C")
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model()
        assert np.array_equal(result.transformed_positions, [1, 6, 10])
        assert result.contig_length == 11

        def floor_transform(positions):
            return np.floor(positions).astype(int)

        result = ts.map_to_vcf_model(position_transform=floor_transform)
        assert np.array_equal(result.transformed_positions, [1, 5, 9])
        assert result.contig_length == 10

    def test_legacy_position_transform(self):
        # Test legacy transform with duplicate positions
        tables = tskit.TableCollection(10.0)
        tables.sites.add_row(position=1.4, ancestral_state="A")
        tables.sites.add_row(position=1.6, ancestral_state="T")
        tables.sites.add_row(position=1.7, ancestral_state="T")
        tables.sites.add_row(position=3.2, ancestral_state="C")
        tables.sites.add_row(position=3.8, ancestral_state="G")
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model(position_transform="legacy")
        assert np.array_equal(result.transformed_positions, [1, 2, 3, 4, 5])
        assert result.contig_length == 10

    def test_position_transform_no_sites(self):
        tables = tskit.TableCollection(5.5)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model()
        assert result.transformed_positions.shape == (0,)
        assert result.contig_length == 6

    def test_invalid_position_transform_return_shape(self):
        tables = tskit.TableCollection(10.0)
        tables.sites.add_row(position=1.0, ancestral_state="A")
        tables.sites.add_row(position=5.0, ancestral_state="T")
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        def bad_transform(positions):
            return np.array([1])  # Wrong length

        with pytest.raises(
            ValueError,
            match="Position transform must return an array of the same length",
        ):
            ts.map_to_vcf_model(position_transform=bad_transform)

    def test_contig_id(self):
        tables = tskit.TableCollection(10.0)
        tables.sites.add_row(position=1.0, ancestral_state="A")
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model(contig_id="chr1")
        assert result.contig_id == "chr1"

        result = ts.map_to_vcf_model()
        assert result.contig_id == "1"

    def test_isolated_as_missing(self):
        tables = tskit.TableCollection(10.0)
        tables.sites.add_row(position=1.0, ancestral_state="A")
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()

        result = ts.map_to_vcf_model(isolated_as_missing=False)
        assert result.isolated_as_missing is False

        result = ts.map_to_vcf_model()
        assert result.isolated_as_missing is True


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_mutations_edge(ts):
    for mut, mut_edge in itertools.zip_longest(ts.mutations(), ts.mutations_edge):
        assert mut.edge == mut_edge
