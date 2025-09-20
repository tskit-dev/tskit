# MIT License
#
# Copyright (c) 2018-2025 Tskit Developers
# Copyright (C) 2017 University of Oxford
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
Test cases for visualisation in tskit.
"""
import collections
import io
import logging
import math
import os
import pathlib
import platform
import re
import xml.etree

import msprime
import numpy as np
import pytest
import xmlunittest

import tests.test_wright_fisher as wf
import tests.tsutil as tsutil
import tskit
from tskit import drawing


IS_WINDOWS = platform.system() == "Windows"


class TestTreeDraw:
    """
    Tests for the tree drawing functionality.
    TODO - the get_XXX_tree() functions should probably be placed in fixtures
    """

    def get_binary_tree(self):
        ts = msprime.simulate(10, random_seed=1, mutation_rate=1)
        return next(ts.trees())

    def get_nonbinary_ts(self):
        tables = wf.wf_sim(
            8,
            4,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=2,
        )
        tables.sort()
        ts = tables.tree_sequence().simplify()
        return tsutil.jukes_cantor(ts, 10, 0.025, seed=1)

    def get_nonbinary_tree(self):
        for t in self.get_nonbinary_ts().trees():
            for u in t.nodes():
                if len(t.children(u)) > 2:
                    return t
        raise AssertionError()

    def get_zero_edge_tree(self):
        tables = tskit.TableCollection(sequence_length=2)
        # These must be samples or we will have zero roots.
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=0, ancestral_state="0")
        tables.mutations.add_row(site=0, node=0, derived_state="1")
        tables.mutations.add_row(site=0, node=1, derived_state="1")
        return tables.tree_sequence().first()

    def get_zero_roots_tree(self):
        tables = tskit.TableCollection(sequence_length=2)
        # If we have no samples we have zero roots
        tables.nodes.add_row(time=0)
        tables.nodes.add_row(time=0)
        tables.nodes.add_row(time=1)
        tables.edges.add_row(0, 2, 2, 0)
        tables.edges.add_row(0, 2, 2, 1)
        tree = tables.tree_sequence().first()
        assert tree.num_roots == 0
        return tree

    def get_multiroot_tree(self):
        ts = msprime.simulate(15, random_seed=1)
        # Take off the top quarter of edges
        tables = ts.dump_tables()
        edges = tables.edges
        n = len(edges) - len(edges) // 4
        edges.set_columns(
            left=edges.left[:n],
            right=edges.right[:n],
            parent=edges.parent[:n],
            child=edges.child[:n],
        )
        ts = tables.tree_sequence()
        for t in ts.trees():
            if t.num_roots > 1:
                return t
        raise AssertionError()

    def get_mutations_over_roots_tree(self):
        ts = msprime.simulate(15, random_seed=1)
        ts = ts.decapitate(ts.tables.nodes.time[-1] / 2)
        tables = ts.dump_tables()
        delta = 1.0 / (ts.num_nodes + 1)
        x = 0
        for node in range(ts.num_nodes):
            site_id = tables.sites.add_row(x, ancestral_state="0")
            x += delta
            tables.mutations.add_row(site_id, node=node, derived_state="1")
        ts = tables.tree_sequence()
        tree = ts.first()
        assert any(tree.parent(mut.node) == tskit.NULL for mut in tree.mutations())
        return tree

    def get_unary_node_tree(self):
        ts = msprime.simulate(2, random_seed=1)
        tables = ts.dump_tables()
        edges = tables.edges
        # Take out all the edges except 1
        n = 1
        edges.set_columns(
            left=edges.left[:n],
            right=edges.right[:n],
            parent=edges.parent[:n],
            child=edges.child[:n],
        )
        ts = tables.tree_sequence()
        for t in ts.trees():
            for u in t.nodes():
                if len(t.children(u)) == 1:
                    return t
        raise AssertionError()

    def get_empty_tree(self):
        tables = tskit.TableCollection(sequence_length=1)
        ts = tables.tree_sequence()
        return next(ts.trees())

    def get_simple_ts(self, use_mutation_times=False):
        """
        return a simple tree seq that does not depend on msprime
        """
        nodes = io.StringIO(
            """\
        id      is_sample   population      individual      time    metadata
        0       1       0       -1      0
        1       1       0       -1      0
        2       1       0       -1      0
        3       1       0       -1      0
        4       0       0       -1      0.1145014598813
        5       0       0       -1      1.11067965364865
        6       0       0       -1      1.75005250750382
        7       0       0       -1      5.31067154311640
        8       0       0       -1      6.57331354884652
        9       0       0       -1      9.08308317451295
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.00000000      1.00000000      4       0
        1       0.00000000      1.00000000      4       1
        2       0.00000000      1.00000000      5       2
        3       0.00000000      1.00000000      5       3
        4       0.79258618      0.90634460      6       4
        5       0.79258618      0.90634460      6       5
        6       0.05975243      0.79258618      7       4
        7       0.90634460      0.91029435      7       4
        8       0.05975243      0.79258618      7       5
        9       0.90634460      0.91029435      7       5
        10      0.91029435      1.00000000      8       4
        11      0.91029435      1.00000000      8       5
        12      0.00000000      0.05975243      9       4
        13      0.00000000      0.05975243      9       5
        """
        )
        sites = io.StringIO(
            """\
        position      ancestral_state
        0.05          A
        0.06          0
        0.3           Empty
        0.5           XXX
        0.91          T
        """
        )
        muts = io.StringIO(
            """\
        site   node    derived_state    parent    time
        0      9       T                -1        15
        0      9       G                0         9.1
        0      5       1                1         9
        1      4       C                -1        1.6
        1      4       G                3         1.5
        2      7       G                -1        10
        2      3       C                5         1
        4      3       G                -1        1
        """
        )
        ts = tskit.load_text(nodes, edges, sites=sites, mutations=muts, strict=False)
        if use_mutation_times:
            return ts
        tables = ts.dump_tables()
        tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)
        return tables.tree_sequence()

    def get_ts_varying_min_times(self, *args, **kwargs):
        """
        Like get_simple_ts but return a tree sequence with negative times, and some trees
        with different min times (i.e. with dangling nonsample nodes at negative times)
        """
        ts = self.get_simple_ts(*args, **kwargs)
        tables = ts.dump_tables()
        time = tables.nodes.time
        time[time == 0] = 0.1
        time[3] = -9.99
        tables.nodes.time = time
        # set node 3 to be non-sample node lower than the rest
        flags = tables.nodes.flags
        flags[3] = 0
        tables.nodes.flags = flags
        edges = tables.edges
        assert edges[3].child == 3 and edges[3].parent == 5
        edges[3] = edges[3].replace(left=ts.breakpoints(True)[1])
        tables.sort()
        tables.nodes.flags = flags
        return tables.tree_sequence()

    def fail(self, *args, **kwargs):
        """
        Required for xmlunittest.XmlTestMixin to work with pytest not unittest
        """
        pytest.fail(*args, **kwargs)


def closest_left_node(tree, u):
    """
    Returns the node that is closest to u in a left-to-right sense.
    """
    ret = tskit.NULL
    while u != tskit.NULL and ret == tskit.NULL:
        ret = tree.left_sib(u)
        u = tree.parent(u)
    return ret


def get_left_neighbour(tree, traversal_order):
    """
    This is a less efficient version of the get_left_neighbour function in
    drawing.py.
    """
    # Note: roots are the children of -1 here.
    children = collections.defaultdict(list)
    for u in tree.nodes(order=traversal_order):
        parent = tree.parent(u)
        children[parent].append(u)

    left_neighbour = np.full(tree.tree_sequence.num_nodes, tskit.NULL, dtype=int)
    for u in tree.nodes():
        next_left = tskit.NULL
        child = u
        while child != tskit.NULL and next_left == tskit.NULL:
            parent = tree.parent(child)
            child_index = children[parent].index(child)
            if child_index > 0:
                next_left = children[parent][child_index - 1]
            child = parent
        left_neighbour[u] = next_left
    return left_neighbour


class TestClosestLeftNode(TestTreeDraw):
    """
    Tests the code for finding the closest left node in a tree.
    """

    def verify(self, tree):
        m1 = drawing.get_left_neighbour(tree, "postorder")
        m2 = get_left_neighbour(tree, "postorder")
        np.testing.assert_array_equal(m1, m2)
        for u in tree.nodes():
            assert m1[u] == closest_left_node(tree, u)

        m1 = drawing.get_left_neighbour(tree, "minlex_postorder")
        m2 = get_left_neighbour(tree, "minlex_postorder")
        np.testing.assert_array_equal(m1, m2)

    def test_2_binary(self):
        ts = msprime.simulate(2, random_seed=2)
        self.verify(ts.first())

    def test_5_binary(self):
        ts = msprime.simulate(5, random_seed=2)
        self.verify(ts.first())

    def test_10_binary(self):
        ts = msprime.simulate(10, random_seed=2)
        self.verify(ts.first())

    def test_20_binary(self):
        ts = msprime.simulate(20, random_seed=3)
        self.verify(ts.first())

    def test_nonbinary(self):
        self.verify(self.get_nonbinary_tree())

    def test_zero_edge(self):
        self.verify(self.get_zero_edge_tree())

    def test_zero_roots(self):
        self.verify(self.get_zero_roots_tree())

    def test_multiroot(self):
        self.verify(self.get_multiroot_tree())

    def test_left_child(self):
        t = self.get_nonbinary_tree()
        left_child = drawing.get_left_child(t, t.postorder())
        for u in t.nodes(order="postorder"):
            if t.num_children(u) > 0:
                assert left_child[u] == t.children(u)[0]

    def test_null_node_left_child(self):
        t = self.get_nonbinary_tree()
        arr = list(t.nodes(order="minlex_postorder"))
        left_child = drawing.get_left_child(t, arr)
        assert left_child[tskit.NULL] == tskit.NULL

    def test_leaf_node_left_child(self):
        t = self.get_nonbinary_tree()
        arr = list(t.nodes(order="minlex_postorder"))
        left_child = drawing.get_left_child(t, arr)
        for u in t.samples():
            assert left_child[u] == tskit.NULL


class TestOrder(TestTreeDraw):
    """
    Tests for using the different node orderings.
    """

    def test_bad_order(self):
        for bad_order in [("sdf"), "sdf", 1234, ""]:
            with pytest.raises(ValueError):
                drawing.check_order(bad_order)

    def test_default_order(self):
        traversal_order = drawing.check_order(None)
        assert traversal_order == "minlex_postorder"

    def test_order_mapping(self):
        assert drawing.check_order("tree") == "postorder"
        assert drawing.check_order("minlex") == "minlex_postorder"

    def test_tree_svg_variants(self):
        t = self.get_binary_tree()
        output1 = t.draw(format="svg")
        output2 = t.draw(format="svg", order="minlex")
        output3 = t.draw(format="svg", order="tree")
        # Default is minlex
        assert output1 == output2
        # tree is at least different to minlex
        assert output1 != output3
        # draw_svg gets the same results
        assert t.draw_svg() == output1
        assert t.draw_svg(order="minlex") == output1
        assert t.draw_svg(order="tree") == output3

    def test_tree_text_variants(self):
        t = self.get_binary_tree()
        output1 = t.draw(format="unicode")
        output2 = t.draw(format="unicode", order="minlex")
        output3 = t.draw(format="unicode", order="tree")
        # Default is minlex
        assert output1 == output2
        # tree is at least different to minlex
        assert output1 != output3
        # draw_text gets the same results
        assert t.draw_text() == output1
        assert t.draw_text(order="minlex") == output1
        assert t.draw_text(order="tree") == output3

    def test_tree_sequence_text_variants(self):
        ts = msprime.simulate(10, random_seed=2)
        output1 = ts.draw_text()
        output2 = ts.draw_text(order="minlex")
        output3 = ts.draw_text(order="tree")

        # Default is minlex
        assert output1 == output2
        # tree is at least different to minlex
        assert output1 != output3

    def test_tree_sequence_svg_variants(self):
        ts = msprime.simulate(10, random_seed=2)
        output1 = ts.draw_svg()
        output2 = ts.draw_svg(order="minlex")
        output3 = ts.draw_svg(order="tree")

        # Default is minlex
        assert output1 == output2
        # tree is at least different to minlex
        assert output1 != output3


class TestFormats(TestTreeDraw):
    """
    Tests that formats are recognised correctly.
    """

    def test_svg_variants(self):
        t = self.get_binary_tree()
        for svg in ["svg", "SVG", "sVg"]:
            output = t.draw(format=svg)
            root = xml.etree.ElementTree.fromstring(output)
            assert root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_default(self):
        # Default is SVG
        t = self.get_binary_tree()
        output = t.draw(format=None)
        root = xml.etree.ElementTree.fromstring(output)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        output = t.draw()
        root = xml.etree.ElementTree.fromstring(output)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_ascii_variants(self):
        t = self.get_binary_tree()
        for fmt in ["ascii", "ASCII", "AScii"]:
            output = t.draw(format=fmt)
            with pytest.raises(xml.etree.ElementTree.ParseError):
                xml.etree.ElementTree.fromstring(
                    output,
                )

    def test_unicode_variants(self):
        t = self.get_binary_tree()
        for fmt in ["unicode", "UNICODE", "uniCODE"]:
            output = t.draw(format=fmt)
            with pytest.raises(xml.etree.ElementTree.ParseError):
                xml.etree.ElementTree.fromstring(
                    output,
                )

    def test_bad_formats(self):
        t = self.get_binary_tree()
        for bad_format in ["", "ASC", "SV", "jpeg"]:
            with pytest.raises(ValueError):
                t.draw(format=bad_format)


class TestDrawText(TestTreeDraw):
    """
    Tests the ASCII tree drawing method.
    """

    drawing_format = "ascii"
    example_label = "XXX"

    def verify_basic_text(self, text):
        assert isinstance(text, str)
        # TODO surely something else we can verify about this...

    def test_draw_defaults(self):
        t = self.get_binary_tree()
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_draw_nonbinary(self):
        t = self.get_nonbinary_tree()
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_draw_multiroot(self):
        t = self.get_multiroot_tree()
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_draw_mutations_over_roots(self):
        t = self.get_mutations_over_roots_tree()
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_draw_unary(self):
        t = self.get_unary_node_tree()
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_draw_empty_tree(self):
        t = self.get_empty_tree()
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format)

    def test_draw_zero_roots_tree(self):
        t = self.get_zero_roots_tree()
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format)

    def test_draw_zero_edge_tree(self):
        t = self.get_zero_edge_tree()
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_even_num_children_tree(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           1
        2   1           2
        3   1           1
        4   1           4
        5   1           5
        6   1           7
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       6       0
        0       1       6       1
        0       1       6       2
        0       1       6       3
        0       1       6       4
        0       1       6       5
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_odd_num_children_tree(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           1
        2   1           2
        3   1           1
        4   1           4
        5   1           5
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       5       0
        0       1       5       1
        0       1       5       2
        0       1       5       3
        0       1       5       4
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_node_labels(self):
        t = self.get_binary_tree()
        labels = {u: self.example_label for u in t.nodes()}
        text = t.draw(format=self.drawing_format, node_labels=labels)
        self.verify_basic_text(text)
        j = 0
        for _ in t.nodes():
            j = text[j:].find(self.example_label)
            assert j != -1

    def test_long_internal_labels(self):
        t = self.get_binary_tree()
        labels = {u: "X" * 10 for u in t.nodes() if t.is_internal(u)}
        text = t.draw(format=self.drawing_format, node_labels=labels)
        self.verify_basic_text(text)

    def test_no_node_labels(self):
        t = self.get_binary_tree()
        labels = {}
        text = t.draw(format=self.drawing_format, node_labels=labels)
        self.verify_basic_text(text)
        for u in t.nodes():
            assert text.find(str(u)) == -1

    def test_unused_args(self):
        t = self.get_binary_tree()
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, width=300)
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, height=300)
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, mutation_labels={})
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, mutation_colours={})
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, edge_colours={})
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, node_colours={})
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, max_time=1234)
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, min_time=1234)
        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                t.draw(format=self.drawing_format, max_tree_height=1234)
        with pytest.raises(ValueError):
            t.draw(format=self.drawing_format, time_scale="time")
        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                t.draw(format=self.drawing_format, tree_height_scale="time")


class TestDrawUnicode(TestDrawText):
    """
    Tests the Unicode tree drawing method
    """

    drawing_format = "unicode"
    example_label = "\u20ac" * 10  # euro symbol


class TestDrawTextErrors:
    """
    Tests for errors occuring in tree drawing code.
    """

    def test_bad_orientation(self):
        t = msprime.simulate(5, mutation_rate=0.1, random_seed=2).first()
        for bad_orientation in ["", "leftright", "sdf"]:
            with pytest.raises(ValueError):
                t.draw_text(orientation=bad_orientation)


class TestDrawTextExamples(TestTreeDraw):
    """
    Verify that we get the correct rendering for some examples.
    """

    def verify_text_rendering(self, drawn, drawn_tree, debug=False):
        if debug:
            print("Drawn:")
            print(drawn)
            print("Expected:")
            print(drawn_tree)
        tree_lines = drawn_tree.splitlines()
        drawn_lines = drawn.splitlines()
        assert len(tree_lines) == len(drawn_lines)
        for l1, l2 in zip(tree_lines, drawn_lines):
            # Trailing white space isn't significant.
            assert l1.rstrip() == l2.rstrip()

    def test_simple_tree(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       2       0
        0       1       2       1
        """
        )
        tree = (
            # fmt: off
            " 2 \n"
            "┏┻┓\n"
            "0 1"
            # fmt: on
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode", order="tree")
        self.verify_text_rendering(drawn, tree)
        drawn = t.draw_text()
        self.verify_text_rendering(drawn, tree)

        tree = (
            # fmt: off
            " 2 \n"
            "+++\n"
            "0 1\n"
            # fmt: on
        )
        drawn = t.draw_text(use_ascii=True, order="tree")
        self.verify_text_rendering(drawn, tree)

        tree = (
            # fmt: off
            " ┏0\n"
            "2┫  \n"
            " ┗1\n"
            # fmt: on
        )
        drawn = t.draw_text(orientation="left", order="tree")
        self.verify_text_rendering(drawn, tree)
        tree = (
            # fmt: off
            " +0\n"
            "2+  \n"
            " +1\n"
            # fmt: on
        )
        drawn = t.draw_text(orientation="left", use_ascii=True, order="tree")
        self.verify_text_rendering(drawn, tree)

    def test_simple_tree_long_label(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       2       0
        0       1       2       1
        """
        )
        tree = (
            # fmt: off
            "ABCDEF\n"
            "┏┻┓   \n"
            "0 1   \n"
            # fmt: on
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw_text(node_labels={0: "0", 1: "1", 2: "ABCDEF"}, order="tree")
        self.verify_text_rendering(drawn, tree)

        tree = (
            # fmt: off
            "0┓      \n"
            " ┣ABCDEF\n"
            "1┛      \n"
            # fmt: on
        )
        drawn = t.draw_text(
            node_labels={0: "0", 1: "1", 2: "ABCDEF"}, orientation="right", order="tree"
        )
        self.verify_text_rendering(drawn, tree)

        drawn = t.draw_text(
            node_labels={0: "ABCDEF", 1: "1", 2: "2"}, orientation="right", order="tree"
        )
        tree = (
            # fmt: off
            "ABCDEF┓ \n"
            "      ┣2\n"
            "1━━━━━┛ \n"
            # fmt: on
        )
        self.verify_text_rendering(drawn, tree)

        tree = (
            # fmt: off
            "      ┏0\n"
            "ABCDEF┫ \n"
            "      ┗1\n"
            # fmt: on
        )
        drawn = t.draw_text(
            node_labels={0: "0", 1: "1", 2: "ABCDEF"}, orientation="left", order="tree"
        )
        self.verify_text_rendering(drawn, tree)

    def test_four_leaves(self):
        nodes = io.StringIO(
            """\
        id      is_sample   population      individual      time    metadata
        0       1       0       -1      0.00000000000000
        1       1       0       -1      0.00000000000000
        2       1       0       -1      0.00000000000000
        3       1       0       -1      0.00000000000000
        4       0       0       -1      0.26676079696421
        5       0       0       -1      1.48826948286480
        6       0       0       -1      2.91835007758007
        """
        )
        edges = io.StringIO(
            """\
        left            right           parent  child
        0.00000000      1.00000000      4       0
        0.00000000      1.00000000      4       3
        0.00000000      1.00000000      5       2
        0.00000000      1.00000000      5       4
        0.00000000      1.00000000      6       1
        0.00000000      1.00000000      6       5
        """
        )
        tree = (
            "  6     \n"
            "┏━┻━┓   \n"
            "┃   5   \n"
            "┃ ┏━┻┓  \n"
            "┃ ┃  4  \n"
            "┃ ┃ ┏┻┓ \n"
            "1 2 0 3 \n"
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = ts.first()
        drawn = t.draw(format="unicode", order="tree")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(order="tree"), tree)

        drawn = t.draw_text(orientation="bottom", order="tree")
        tree = (
            "1 2 0 3\n"
            "┃ ┃ ┗┳┛\n"
            "┃ ┃  4 \n"
            "┃ ┗━┳┛ \n"
            "┃   5  \n"
            "┗━┳━┛  \n"
            "  6    \n"
        )
        self.verify_text_rendering(drawn, tree)

        tree = (
            " ┏━━━━1\n"
            " ┃     \n"
            "6┫ ┏━━2\n"
            " ┃ ┃   \n"
            " ┗5┫ ┏0\n"
            "   ┗4┫  \n"
            "     ┗3\n"
        )
        self.verify_text_rendering(t.draw_text(orientation="left", order="tree"), tree)

        tree = (
            "2.92┊   6     ┊\n"
            "    ┊ ┏━┻━┓   ┊\n"
            "1.49┊ ┃   5   ┊\n"
            "    ┊ ┃ ┏━┻┓  ┊\n"
            "0.27┊ ┃ ┃  4  ┊\n"
            "    ┊ ┃ ┃ ┏┻┓ ┊\n"
            "0.00┊ 1 2 0 3 ┊\n"
            "    0         1\n"
        )
        self.verify_text_rendering(ts.draw_text(order="tree"), tree)

        tree = (
            "  6    \n"
            "+-+-+  \n"
            "|   5  \n"
            "| +-++ \n"
            "| |  4 \n"
            "| | +++\n"
            "1 2 0 3\n"
        )
        drawn = t.draw(format="ascii", order="tree")
        self.verify_text_rendering(drawn, tree)

        tree = (
            "  6     \n"
            "┏━┻━┓   \n"
            "┃xxxxxxxxxx\n"
            "┃ ┏━┻┓  \n"
            "┃ ┃  4  \n"
            "┃ ┃ ┏┻┓ \n"
            "1 2 0 3 \n"
        )
        labels = {u: str(u) for u in t.nodes()}
        labels[5] = "xxxxxxxxxx"
        drawn = t.draw_text(node_labels=labels, order="tree")
        self.verify_text_rendering(drawn, tree)

        tree = (
            " ┏━━━━━━━━━━━━━1\n"
            " ┃              \n"
            "6┫          ┏━━2\n"
            " ┃          ┃   \n"
            " ┗xxxxxxxxxx┫ ┏0\n"
            "            ┗4┫ \n"
            "              ┗3\n"
        )
        drawn = t.draw_text(node_labels=labels, orientation="left", order="tree")
        self.verify_text_rendering(drawn, tree)

        tree = (
            "2.92┊   6         ┊\n"
            "    ┊ ┏━┻━┓       ┊\n"
            "1.49┊ ┃xxxxxxxxxx ┊\n"
            "    ┊ ┃ ┏━┻┓      ┊\n"
            "0.27┊ ┃ ┃  4      ┊\n"
            "    ┊ ┃ ┃ ┏┻┓     ┊\n"
            "0.00┊ 1 2 0 3     ┊\n"
            "    0             1\n"
        )
        drawn = ts.draw_text(node_labels=labels, order="tree")
        self.verify_text_rendering(drawn, tree)

    def test_trident_tree(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           0
        3   1           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       3       0
        0       1       3       1
        0       1       3       2
        """
        )
        tree = (
            # fmt: off
            "  3  \n"
            "┏━╋━┓\n"
            "0 1 2\n"
            # fmt: on
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode", order="tree")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

        tree = (
            # fmt: off
            " ┏0\n"
            " ┃\n"
            "3╋1\n"
            " ┃\n"
            " ┗2\n"
            # fmt: on
        )
        drawn = t.draw_text(orientation="left")
        self.verify_text_rendering(drawn, tree)

        tree = (
            # fmt: off
            "0┓\n"
            " ┃\n"
            "1╋3\n"
            " ┃\n"
            "2┛\n"
            # fmt: on
        )
        drawn = t.draw_text(orientation="right")
        self.verify_text_rendering(drawn, tree)

    def test_pitchfork_tree(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           0
        3   1           0
        4   1           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       4       0
        0       1       4       1
        0       1       4       2
        0       1       4       3
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        tree = (
            # fmt: off
            "   4   \n"
            "┏━┳┻┳━┓\n"
            "0 1 2 3\n"
            # fmt: on
        )
        drawn = t.draw(format="unicode", order="tree")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

        # No labels
        tree = (
            # fmt: off
            "   ┃   \n"
            "┏━┳┻┳━┓\n"
            "┃ ┃ ┃ ┃\n"
            # fmt: on
        )
        drawn = t.draw(format="unicode", node_labels={}, order="tree")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(node_labels={}), tree)
        # Some labels
        tree = (
            # fmt: off
            "   ┃   \n"
            "┏━┳┻┳━┓\n"
            "0 ┃ ┃ 3\n"
            # fmt: on
        )
        labels = {0: "0", 3: "3"}
        drawn = t.draw(format="unicode", node_labels=labels, order="tree")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(node_labels=labels), tree)

        tree = (
            # fmt: off
            " ┏0\n"
            " ┃\n"
            " ┣1\n"
            "4┫\n"
            " ┣2\n"
            " ┃\n"
            " ┗3\n"
            # fmt: on
        )
        drawn = t.draw_text(orientation="left")
        self.verify_text_rendering(drawn, tree)

        tree = (
            # fmt: off
            "0┓\n"
            " ┃\n"
            "1┫\n"
            " ┣4\n"
            "2┫\n"
            " ┃\n"
            "3┛\n"
            # fmt: on
        )
        drawn = t.draw_text(orientation="right")
        self.verify_text_rendering(drawn, tree)

    def test_stick_tree(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           1
        2   1           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       1       0
        0       1       2       1
        """
        )
        tree = (
            # fmt: off
            "2\n"
            "┃\n"
            "1\n"
            "┃\n"
            "0\n"
            # fmt: on
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode", order="tree")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

        tree = (
            # fmt: off
            "0\n"
            "┃\n"
            "1\n"
            "┃\n"
            "2\n"
            # fmt: on
        )
        drawn = t.draw_text(orientation="bottom")
        self.verify_text_rendering(drawn, tree)

        tree = "2━1━0\n"
        drawn = t.draw_text(orientation="left")
        self.verify_text_rendering(drawn, tree)

        tree = "0━1━2\n"
        drawn = t.draw_text(orientation="right")
        self.verify_text_rendering(drawn, tree)

    def test_draw_forky_tree(self):
        tree = (
            "      14            \n"
            "  ┏━━━━┻━━━━┓       \n"
            "  ┃        13       \n"
            "  ┃   ┏━┳━┳━╋━┳━━┓  \n"
            "  ┃   ┃ ┃ ┃ ┃ ┃ 12  \n"
            "  ┃   ┃ ┃ ┃ ┃ ┃ ┏┻┓ \n"
            " 11   ┃ ┃ ┃ ┃ ┃ ┃ ┃ \n"
            "┏━┻┓  ┃ ┃ ┃ ┃ ┃ ┃ ┃ \n"
            "┃ 10  ┃ ┃ ┃ ┃ ┃ ┃ ┃ \n"
            "┃ ┏┻┓ ┃ ┃ ┃ ┃ ┃ ┃ ┃ \n"
            "8 0 3 2 4 5 6 9 1 7 \n"
        )

        nodes = io.StringIO(
            """\
            id      is_sample   population      individual      time    metadata
            0       1       0       -1      0.00000000000000
            1       1       0       -1      0.00000000000000
            2       1       0       -1      0.00000000000000
            3       1       0       -1      0.00000000000000
            4       1       0       -1      0.00000000000000
            5       1       0       -1      0.00000000000000
            6       1       0       -1      0.00000000000000
            7       1       0       -1      0.00000000000000
            8       1       0       -1      0.00000000000000
            9       1       0       -1      0.00000000000000
            10      0       0       -1      0.02398248117831
            11      0       0       -1      0.17378680550869
            12      0       0       -1      0.19950200178411
            13      0       0       -1      0.20000000000000
            14      0       0       -1      5.68339203134457
        """
        )
        edges = io.StringIO(
            """\
            left            right           parent  child
            0.00000000      1.00000000      10      0
            0.00000000      1.00000000      10      3
            0.00000000      1.00000000      11      8
            0.00000000      1.00000000      11      10
            0.00000000      1.00000000      12      1
            0.00000000      1.00000000      12      7
            0.00000000      1.00000000      13      2
            0.00000000      1.00000000      13      4
            0.00000000      1.00000000      13      5
            0.00000000      1.00000000      13      6
            0.00000000      1.00000000      13      9
            0.00000000      1.00000000      13      12
            0.00000000      1.00000000      14      11
            0.00000000      1.00000000      14      13
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode", order="tree")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(order="tree"), tree)

        tree = (
            "        14              \n"
            "  ┏━━━━━━┻━━━━━━┓       \n"
            "  ┃            13       \n"
            "  ┃        ┏━┳━┳┻┳━┳━━┓ \n"
            "  ┃        ┃ ┃ ┃ ┃ ┃ 12 \n"
            "  ┃        ┃ ┃ ┃ ┃ ┃ ┏┻┓\n"
            "x11xxxxxxx ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "┏━┻┓       ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "┃ 10       ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "┃ ┏┻┓      ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "8 0 3      2 4 5 6 9 1 7\n"
        )
        labels = {u: str(u) for u in t.nodes()}
        labels[11] = "x11xxxxxxx"
        self.verify_text_rendering(t.draw_text(node_labels=labels, order="tree"), tree)

        tree = (
            "      14           \n"
            "  ┏━━━━┻━━━━┓      \n"
            "  ┃        13      \n"
            "  ┃    ┏━━┳━╋━┳━┳━┓\n"
            "  ┃   12  ┃ ┃ ┃ ┃ ┃\n"
            "  ┃   ┏┻┓ ┃ ┃ ┃ ┃ ┃\n"
            " 11   ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            " ┏┻━┓ ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "10  ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "┏┻┓ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "0 3 8 1 7 2 4 5 6 9\n"
        )
        self.verify_text_rendering(t.draw_text(order="minlex"), tree)

    def test_draw_multiroot_forky_tree(self):
        tree = (
            "           13      \n"
            "      ┏━┳━┳━╋━┳━━┓ \n"
            "      ┃ ┃ ┃ ┃ ┃ 12 \n"
            "      ┃ ┃ ┃ ┃ ┃ ┏┻┓\n"
            " 11   ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "┏━┻┓  ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "┃ 10  ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "┃ ┏┻┓ ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "8 0 3 2 4 5 6 9 1 7\n"
        )
        nodes = io.StringIO(
            """\
            id      is_sample   population      individual      time    metadata
            0       1       0       -1      0.00000000000000
            1       1       0       -1      0.00000000000000
            2       1       0       -1      0.00000000000000
            3       1       0       -1      0.00000000000000
            4       1       0       -1      0.00000000000000
            5       1       0       -1      0.00000000000000
            6       1       0       -1      0.00000000000000
            7       1       0       -1      0.00000000000000
            8       1       0       -1      0.00000000000000
            9       1       0       -1      0.00000000000000
            10      0       0       -1      0.02398248117831
            11      0       0       -1      0.17378680550869
            12      0       0       -1      0.19950200178411
            13      0       0       -1      0.20000000000000
            14      0       0       -1      5.68339203134457
        """
        )
        edges = io.StringIO(
            """\
            left            right           parent  child
            0.00000000      1.00000000      10      0
            0.00000000      1.00000000      10      3
            0.00000000      1.00000000      11      8
            0.00000000      1.00000000      11      10
            0.00000000      1.00000000      12      1
            0.00000000      1.00000000      12      7
            0.00000000      1.00000000      13      2
            0.00000000      1.00000000      13      4
            0.00000000      1.00000000      13      5
            0.00000000      1.00000000      13      6
            0.00000000      1.00000000      13      9
            0.00000000      1.00000000      13      12
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode", order="tree")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(order="tree"), tree)

        tree = (
            "           13      \n"
            "       ┏━━┳━╋━┳━┳━┓\n"
            "      12  ┃ ┃ ┃ ┃ ┃\n"
            "      ┏┻┓ ┃ ┃ ┃ ┃ ┃\n"
            " 11   ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            " ┏┻━┓ ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "10  ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "┏┻┓ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃\n"
            "0 3 8 1 7 2 4 5 6 9\n"
        )
        drawn = t.draw(format="unicode")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)
        self.verify_text_rendering(t.draw_text(order="minlex"), tree)

    def test_simple_tree_sequence(self):
        ts = self.get_simple_ts()
        ts_drawing = (
            "9.08┊    9    ┊         ┊         ┊         ┊         ┊\n"
            "    ┊  ┏━┻━┓  ┊         ┊         ┊         ┊         ┊\n"
            "6.57┊  ┃   ┃  ┊         ┊         ┊         ┊    8    ┊\n"
            "    ┊  ┃   ┃  ┊         ┊         ┊         ┊  ┏━┻━┓  ┊\n"
            "5.31┊  ┃   ┃  ┊    7    ┊         ┊    7    ┊  ┃   ┃  ┊\n"
            "    ┊  ┃   ┃  ┊  ┏━┻━┓  ┊         ┊  ┏━┻━┓  ┊  ┃   ┃  ┊\n"
            "1.75┊  ┃   ┃  ┊  ┃   ┃  ┊    6    ┊  ┃   ┃  ┊  ┃   ┃  ┊\n"
            "    ┊  ┃   ┃  ┊  ┃   ┃  ┊  ┏━┻━┓  ┊  ┃   ┃  ┊  ┃   ┃  ┊\n"
            "1.11┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊\n"
            "    ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊\n"
            "0.11┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊\n"
            "    ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊\n"
            "0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊\n"
            "  0.00      0.06      0.79      0.91      0.91      1.00\n"
        )
        self.verify_text_rendering(ts.draw_text(), ts_drawing)

        ts_drawing = (
            "9.08|    9    |         |         |         |         |\n"
            "    |  +-+-+  |         |         |         |         |\n"
            "6.57|  |   |  |         |         |         |    8    |\n"
            "    |  |   |  |         |         |         |  +-+-+  |\n"
            "5.31|  |   |  |    7    |         |    7    |  |   |  |\n"
            "    |  |   |  |  +-+-+  |         |  +-+-+  |  |   |  |\n"
            "1.75|  |   |  |  |   |  |    6    |  |   |  |  |   |  |\n"
            "    |  |   |  |  |   |  |  +-+-+  |  |   |  |  |   |  |\n"
            "1.11|  |   5  |  |   5  |  |   5  |  |   5  |  |   5  |\n"
            "    |  |  +++ |  |  +++ |  |  +++ |  |  +++ |  |  +++ |\n"
            "0.11|  4  | | |  4  | | |  4  | | |  4  | | |  4  | | |\n"
            "    | +++ | | | +++ | | | +++ | | | +++ | | | +++ | | |\n"
            "0.00| 0 1 2 3 | 0 1 2 3 | 0 1 2 3 | 0 1 2 3 | 0 1 2 3 |\n"
            "  0.00      0.06      0.79      0.91      0.91      1.00\n"
        )
        self.verify_text_rendering(ts.draw_text(use_ascii=True), ts_drawing)

        ts_drawing = (
            "┊    9    ┊         ┊         ┊         ┊         ┊\n"
            "┊  ┏━┻━┓  ┊         ┊         ┊         ┊         ┊\n"
            "┊  ┃   ┃  ┊         ┊         ┊         ┊    8    ┊\n"
            "┊  ┃   ┃  ┊         ┊         ┊         ┊  ┏━┻━┓  ┊\n"
            "┊  ┃   ┃  ┊    7    ┊         ┊    7    ┊  ┃   ┃  ┊\n"
            "┊  ┃   ┃  ┊  ┏━┻━┓  ┊         ┊  ┏━┻━┓  ┊  ┃   ┃  ┊\n"
            "┊  ┃   ┃  ┊  ┃   ┃  ┊    6    ┊  ┃   ┃  ┊  ┃   ┃  ┊\n"
            "┊  ┃   ┃  ┊  ┃   ┃  ┊  ┏━┻━┓  ┊  ┃   ┃  ┊  ┃   ┃  ┊\n"
            "┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊\n"
            "┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊\n"
            "┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊\n"
            "┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊\n"
            "┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊\n"
            "0.00    0.06      0.79      0.91      0.91      1.00\n"
        )
        self.verify_text_rendering(ts.draw_text(time_label_format=""), ts_drawing)

        ts_drawing = (
            "┊    9    ┊         ┊         ┊         ┊         ┊\n"
            "┊  ┏━┻━┓  ┊         ┊         ┊         ┊         ┊\n"
            "┊  ┃   ┃  ┊         ┊         ┊         ┊    8    ┊\n"
            "┊  ┃   ┃  ┊         ┊         ┊         ┊  ┏━┻━┓  ┊\n"
            "┊  ┃   ┃  ┊    7    ┊         ┊    7    ┊  ┃   ┃  ┊\n"
            "┊  ┃   ┃  ┊  ┏━┻━┓  ┊         ┊  ┏━┻━┓  ┊  ┃   ┃  ┊\n"
            "┊  ┃   ┃  ┊  ┃   ┃  ┊    6    ┊  ┃   ┃  ┊  ┃   ┃  ┊\n"
            "┊  ┃   ┃  ┊  ┃   ┃  ┊  ┏━┻━┓  ┊  ┃   ┃  ┊  ┃   ┃  ┊\n"
            "┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊\n"
            "┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊\n"
            "┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊\n"
            "┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊\n"
            "┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊\n"
            "┊         ┊         ┊         ┊         ┊         ┊\n"
        )
        self.verify_text_rendering(
            ts.draw_text(time_label_format="", position_label_format=""), ts_drawing
        )

    def test_tree_sequence_non_minlex(self):
        nodes = io.StringIO(
            """\
            id      is_sample       time    population      individual      metadata
            0       1       0.000000        0       -1
            1       1       0.000000        0       -1
            2       1       0.000000        0       -1
            3       1       0.000000        0       -1
            4       1       0.000000        0       -1
            5       0       1.174545        0       -1
            6       0       1.207717        0       -1
            7       0       1.276422        0       -1
            8       0       1.613390        0       -1
            9       0       2.700069        0       -1
        """
        )
        edges = io.StringIO(
            """\
            left    right   parent  child
            0.000000        1.000000        5       0
            0.000000        1.000000        5       1
            0.000000        0.209330        6       4
            0.000000        0.209330        6       5
            0.000000        1.000000        7       2
            0.209330        1.000000        7       5
            0.000000        0.209330        7       6
            0.209330        1.000000        8       3
            0.209330        1.000000        8       4
            0.000000        0.209330        9       3
            0.000000        1.000000        9       7
            0.209330        1.000000        9       8
        """
        )

        ts = tskit.load_text(nodes, edges, strict=False)

        drawn_minlex = (
            "2.70┊       9   ┊     9     ┊\n"
            "    ┊     ┏━┻━┓ ┊   ┏━┻━━┓  ┊\n"
            "1.61┊     ┃   ┃ ┊   ┃    8  ┊\n"
            "    ┊     ┃   ┃ ┊   ┃   ┏┻┓ ┊\n"
            "1.28┊     7   ┃ ┊   7   ┃ ┃ ┊\n"
            "    ┊   ┏━┻━┓ ┃ ┊  ┏┻━┓ ┃ ┃ ┊\n"
            "1.21┊   6   ┃ ┃ ┊  ┃  ┃ ┃ ┃ ┊\n"
            "    ┊  ┏┻━┓ ┃ ┃ ┊  ┃  ┃ ┃ ┃ ┊\n"
            "1.17┊  5  ┃ ┃ ┃ ┊  5  ┃ ┃ ┃ ┊\n"
            "    ┊ ┏┻┓ ┃ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┃ ┊\n"
            "0.00┊ 0 1 4 2 3 ┊ 0 1 2 3 4 ┊\n"
            "  0.00        0.21        1.00\n"
        )
        self.verify_text_rendering(ts.draw_text(order="minlex"), drawn_minlex)
        self.verify_text_rendering(ts.draw_text(), drawn_minlex)

        drawn_tree = (
            "2.70┊   9       ┊     9     ┊\n"
            "    ┊ ┏━┻━┓     ┊   ┏━┻━━┓  ┊\n"
            "1.61┊ ┃   ┃     ┊   ┃    8  ┊\n"
            "    ┊ ┃   ┃     ┊   ┃   ┏┻┓ ┊\n"
            "1.28┊ ┃   7     ┊   7   ┃ ┃ ┊\n"
            "    ┊ ┃ ┏━┻━┓   ┊ ┏━┻┓  ┃ ┃ ┊\n"
            "1.21┊ ┃ ┃   6   ┊ ┃  ┃  ┃ ┃ ┊\n"
            "    ┊ ┃ ┃ ┏━┻┓  ┊ ┃  ┃  ┃ ┃ ┊\n"
            "1.17┊ ┃ ┃ ┃  5  ┊ ┃  5  ┃ ┃ ┊\n"
            "    ┊ ┃ ┃ ┃ ┏┻┓ ┊ ┃ ┏┻┓ ┃ ┃ ┊\n"
            "0.00┊ 3 2 4 0 1 ┊ 2 0 1 3 4 ┊\n"
            "  0.00        0.21        1.00\n"
        )
        self.verify_text_rendering(ts.draw_text(order="tree"), drawn_tree)

    def test_max_time(self):
        ts = self.get_simple_ts()
        tree = (
            "   9   \n"
            " ┏━┻━┓ \n"
            " ┃   ┃ \n"
            " ┃   ┃ \n"
            " ┃   ┃ \n"
            " ┃   ┃ \n"
            " ┃   ┃ \n"
            " ┃   ┃ \n"
            " ┃   5 \n"
            " ┃  ┏┻┓\n"
            " 4  ┃ ┃\n"
            "┏┻┓ ┃ ┃\n"
            "0 1 2 3\n"
        )
        t = ts.first()
        self.verify_text_rendering(t.draw_text(max_time="ts"), tree)

        tree = (
            "   9   \n"
            " ┏━┻━┓ \n"
            " ┃   5 \n"
            " ┃  ┏┻┓\n"
            " 4  ┃ ┃\n"
            "┏┻┓ ┃ ┃\n"
            "0 1 2 3\n"
        )
        t = ts.first()
        self.verify_text_rendering(t.draw_text(max_time="tree"), tree)
        for bad_max_time in [1, "sdfr", ""]:
            with pytest.raises(ValueError):
                t.draw_text(max_time=bad_max_time)

    def test_no_repr_svg(self):
        tree = self.get_simple_ts().first()
        output = tree.draw(format="unicode")
        with pytest.raises(AttributeError, match="no attribute"):
            output._repr_svg_()


class TestDrawSvgBase(TestTreeDraw, xmlunittest.XmlTestMixin):
    """
    Base class for testing the SVG tree drawing method
    """

    def verify_basic_svg(self, svg, width=200, height=200, num_trees=1, has_root=True):
        prefix = "{http://www.w3.org/2000/svg}"
        root = xml.etree.ElementTree.fromstring(svg)
        assert root.tag == prefix + "svg"
        assert width * num_trees == int(root.attrib["width"])
        assert height == int(root.attrib["height"])

        # Verify the class structure of the svg
        root_group = root.find(prefix + "g")
        assert "class" in root_group.attrib
        assert re.search(r"\b(tree|tree-sequence)\b", root_group.attrib["class"])
        first_plotbox = None
        if "tree-sequence" in root_group.attrib["class"]:
            trees = None
            for g in root_group.findall(prefix + "g"):
                if "trees" in g.attrib.get("class", ""):
                    trees = g
                    break
            assert trees is not None  # Must have found a trees group
            first_tree = trees.find(prefix + "g")
            assert "class" in first_tree.attrib
            assert re.search(r"\btree\b", first_tree.attrib["class"])
            for g in first_tree.findall(prefix + "g"):
                if "class" in g.attrib and re.search(r"\bplotbox\b", g.attrib["class"]):
                    first_plotbox = g
        else:
            for g in root_group.findall(prefix + "g"):
                if "class" in g.attrib and re.search(r"\bplotbox\b", g.attrib["class"]):
                    first_plotbox = g
        assert first_plotbox is not None
        # Check that we have edges, symbols, and labels groups
        groups = first_plotbox.findall(prefix + "g")
        assert len(groups) > 0
        for group in groups:
            assert "class" in group.attrib
            cls = group.attrib["class"]
            # if a subtree plot, the top of the displayed topology is not a local root
            if has_root:
                assert re.search(r"\broot\b", cls)
            else:
                assert not re.search(r"\broot\b", cls)


class TestDrawSvg(TestDrawSvgBase):
    """
    Simple testing for the draw_svg method
    """

    def test_repr_svg(self):
        ts = self.get_simple_ts()
        svg = ts.draw_svg()
        assert str(svg) == svg._repr_svg_()
        svg = ts.first().draw_svg()
        assert str(svg) == svg._repr_svg_()
        svg = ts.first().draw(format="svg")
        assert str(svg) == svg._repr_svg_()

    def test_draw_to_file(self, tmp_path):
        # NB: to view output files for testing changes to drawing code, it is possible
        # to save to a fixed directory using e.g. `pytest --basetemp=/tmp/svgtest ...`
        t = self.get_binary_tree()
        filename = tmp_path / "tree-draw.svg"
        svg = t.draw(path=filename)
        assert os.path.getsize(filename) > 0
        with open(filename) as tmp:
            other_svg = tmp.read()
        assert svg == other_svg

        filename = tmp_path / "tree-draw_svg.svg"
        svg = t.draw_svg(path=filename)
        assert os.path.getsize(filename) > 0
        with open(filename) as tmp:
            other_svg = tmp.read()
        self.verify_basic_svg(svg)
        self.verify_basic_svg(other_svg)

        filename = tmp_path / "ts-draw_svg.svg"
        ts = self.get_simple_ts()
        svg = ts.draw_svg(path=filename)
        assert os.path.getsize(filename) > 0
        with open(filename) as tmp:
            other_svg = tmp.read()
        self.verify_basic_svg(svg, num_trees=ts.num_trees)
        self.verify_basic_svg(other_svg, num_trees=ts.num_trees)

    def test_nonimplemented_base_class(self):
        ts = self.get_simple_ts()
        plot = drawing.SvgAxisPlot(
            ts, (100, 100), {}, "", "dummy-class", None, True, True
        )
        plot.set_spacing()
        with pytest.raises(NotImplementedError):
            plot.draw_x_axis(tick_positions=ts.breakpoints(as_array=True))

    def test_bad_tick_spacing(self):
        # Integer y_ticks to give auto-generated tick locs is not currently implemented
        t = self.get_binary_tree()
        with pytest.raises(TypeError):
            t.draw_svg(y_axis=True, y_ticks=6)
        ts = self.get_simple_ts()
        with pytest.raises(TypeError):
            ts.draw_svg(y_axis=True, y_ticks=6)

    def test_no_mixed_yscales(self):
        ts = self.get_simple_ts()
        with pytest.raises(ValueError, match="vary in timescale"):
            ts.draw_svg(y_axis=True, max_time="tree")

    def test_draw_defaults(self):
        t = self.get_binary_tree()
        svg = t.draw()
        self.verify_basic_svg(svg)
        svg = t.draw_svg()
        self.verify_basic_svg(svg)

    @pytest.mark.parametrize("y_axis", ("left", "right", True, False))
    @pytest.mark.parametrize("y_label", (True, False))
    @pytest.mark.parametrize(
        "time_scale",
        (
            "rank",
            "time",
        ),
    )
    @pytest.mark.parametrize("y_ticks", ([], [0, 1], None))
    @pytest.mark.parametrize("y_gridlines", (True, False))
    def test_draw_svg_y_axis_parameter_combos(
        self, y_axis, y_label, time_scale, y_ticks, y_gridlines
    ):
        t = self.get_binary_tree()
        svg = t.draw_svg(
            y_axis=y_axis,
            y_label=y_label,
            y_ticks=y_ticks,
            y_gridlines=y_gridlines,
            time_scale=time_scale,
        )
        self.verify_basic_svg(svg)
        ts = self.get_simple_ts()
        svg = ts.draw_svg(
            y_axis=y_axis,
            y_label=y_label,
            y_ticks=y_ticks,
            y_gridlines=y_gridlines,
            time_scale=time_scale,
        )
        self.verify_basic_svg(svg, width=200 * ts.num_trees)

    def test_draw_multiroot(self):
        t = self.get_multiroot_tree()
        svg = t.draw()
        self.verify_basic_svg(svg)
        svg = t.draw_svg()
        self.verify_basic_svg(svg)

    def test_draw_mutations_over_roots(self):
        t = self.get_mutations_over_roots_tree()
        with pytest.warns(UserWarning, match="nodes which are not present"):
            svg = t.draw()
            self.verify_basic_svg(svg)
        with pytest.warns(UserWarning, match="nodes which are not present"):
            svg = t.draw_svg()
            self.verify_basic_svg(svg)

    def test_draw_unary(self):
        t = self.get_unary_node_tree()
        svg = t.draw()
        self.verify_basic_svg(svg)
        svg = t.draw_svg()
        self.verify_basic_svg(svg)

    def test_draw_empty(self):
        t = self.get_empty_tree()
        with pytest.raises(ValueError):
            t.draw()
        with pytest.raises(ValueError):
            t.draw_svg()

    def test_draw_zero_roots(self):
        t = self.get_zero_roots_tree()
        with pytest.raises(ValueError):
            t.draw()
        with pytest.raises(ValueError):
            t.draw_svg()

    def test_draw_zero_edge(self):
        t = self.get_zero_edge_tree()
        svg = t.draw()
        self.verify_basic_svg(svg)
        svg = t.draw_svg()
        self.verify_basic_svg(svg)

    def test_mutations_present(self):
        t = self.get_binary_tree()
        assert t.tree_sequence.num_mutations > 0
        svg = t.draw()
        self.verify_basic_svg(svg)
        assert svg.count('class="mut') == t.tree_sequence.num_mutations
        svg = t.draw_svg()
        self.verify_basic_svg(svg)
        assert svg.count('class="mut') == t.tree_sequence.num_mutations
        svg = t.tree_sequence.draw_svg()
        self.verify_basic_svg(svg)
        assert 'class="site' in svg
        assert svg.count('class="site') == t.tree_sequence.num_sites

    def test_sites_omitted(self):
        t = self.get_binary_tree()
        assert t.tree_sequence.num_mutations > 0
        svg = t.draw(omit_sites=True)
        self.verify_basic_svg(svg)
        assert svg.count('class="mut') == 0
        svg = t.draw_svg(omit_sites=True)
        self.verify_basic_svg(svg)
        assert svg.count('class="mut') == 0
        svg = t.tree_sequence.draw_svg(omit_sites=True)
        self.verify_basic_svg(svg)
        assert svg.count('class="mut') == 0
        assert svg.count('class="site') == 0

    def test_width_height(self):
        t = self.get_binary_tree()
        w = 123
        h = 456
        svg = t.draw(width=w, height=h)
        self.verify_basic_svg(svg, w, h)
        svg = t.draw_svg(size=(w, h))
        self.verify_basic_svg(svg, w, h)

    def test_node_labels(self):
        t = self.get_binary_tree()
        labels = {u: "XXX" for u in t.nodes()}
        svg = t.draw(format="svg", node_labels=labels)
        self.verify_basic_svg(svg)
        assert svg.count("XXX") == t.tree_sequence.num_nodes
        svg = t.draw_svg(node_label_attrs={u: {"text": labels[u]} for u in t.nodes()})
        self.verify_basic_svg(svg)
        assert svg.count("XXX") == t.tree_sequence.num_nodes

    def test_one_node_label(self):
        t = self.get_binary_tree()
        labels = {0: "XXX"}
        svg = t.draw(format="svg", node_labels=labels)
        self.verify_basic_svg(svg)
        assert svg.count("XXX") == 1
        svg = t.draw_svg(node_label_attrs={0: {"text": "XXX"}})
        self.verify_basic_svg(svg)
        assert svg.count("XXX") == 1

    def test_no_node_labels(self):
        t = self.get_binary_tree()
        labels = {}
        svg = t.draw(format="svg", node_labels=labels)
        self.verify_basic_svg(svg)
        # Can't really test for much here if we don't understand the SVG

    def test_one_node_colour(self):
        t = self.get_binary_tree()
        colour = "rgb(0, 1, 2)"
        colours = {0: colour}
        svg = t.draw(format="svg", node_colours=colours)
        self.verify_basic_svg(svg)
        assert svg.count(f"fill:{colour}") == 1
        svg = t.draw_svg(node_attrs={0: {"fill": colour}})
        self.verify_basic_svg(svg)
        assert svg.count(f'fill="{colour}"') == 1

    def test_all_nodes_colour(self):
        t = self.get_binary_tree()
        colours = {u: f"rgb({u}, {u}, {u})" for u in t.nodes()}
        svg = t.draw(format="svg", node_colours=colours)
        self.verify_basic_svg(svg)
        for colour in colours.values():
            assert svg.count(f"fill:{colour}") == 1

        svg = t.draw_svg(node_attrs={u: {"fill": colours[u]} for u in t.nodes()})
        self.verify_basic_svg(svg)
        assert svg.count(f'fill="{colour}"') == 1
        for colour in colours.values():
            assert svg.count(f'fill="{colour}"') == 1

    def test_unplotted_node(self):
        t = self.get_binary_tree()
        colour = None
        colours = {0: colour}
        svg = t.draw(format="svg", node_colours=colours)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("opacity:0") == 1

    def test_one_edge_colour(self):
        t = self.get_binary_tree()
        colour = "rgb(0, 1, 2)"
        colours = {0: colour}
        svg = t.draw(format="svg", edge_colours=colours)
        self.verify_basic_svg(svg)
        assert svg.count(f"stroke:{colour}") > 0
        svg = t.draw_svg(edge_attrs={0: {"stroke": colour}})
        self.verify_basic_svg(svg)
        assert svg.count(f'stroke="{colour}"') == 1

    def test_one_mutation_label_colour(self):
        t = self.get_binary_tree()
        colour = "rgb(0, 1, 2)"
        svg = t.draw_svg(mutation_label_attrs={0: {"stroke": colour}})
        self.verify_basic_svg(svg)
        assert svg.count(f'stroke="{colour}"') == 1

    def test_bad_y_axis(self):
        t = self.get_binary_tree()
        for bad_axis in ["te", "asdf", "", [], b"23"]:
            with pytest.raises(ValueError):
                t.draw_svg(y_axis=bad_axis)

    def test_bad_time_scale(self):
        t = self.get_binary_tree()
        for bad_scale in ["te", "asdf", "", [], b"23"]:
            with pytest.raises(ValueError):
                t.draw_svg(time_scale=bad_scale)
            with pytest.raises(ValueError):
                with pytest.warns(FutureWarning):
                    t.draw_svg(tree_height_scale=bad_scale)

    def test_bad_max_time(self):
        t = self.get_binary_tree()
        for bad_height in ["te", "asdf", "", [], b"23"]:
            with pytest.raises(ValueError):
                t.draw_svg(max_time=bad_height)
            with pytest.raises(ValueError):
                with pytest.warns(FutureWarning):
                    t.draw_svg(max_tree_height=bad_height)

    def test_bad_min_time(self):
        t = self.get_binary_tree()
        for bad_min in ["te", "asdf", "", [], b"23"]:
            with pytest.raises(ValueError):
                t.draw_svg(min_time=bad_min)
            with pytest.raises(ValueError):
                with pytest.warns(FutureWarning):
                    t.draw_svg(max_tree_height=bad_min)

    def test_bad_neg_log_time(self):
        t = self.get_ts_varying_min_times().at_index(1)
        assert min(t.time(u) for u in t.nodes()) < 0
        with pytest.raises(ValueError, match="negative times"):
            with np.errstate(invalid="ignore"):
                t.draw_svg(t.draw_svg(time_scale="log_time"))

    def test_time_scale_time_and_max_time(self):
        ts = msprime.simulate(5, recombination_rate=2, random_seed=2)
        t = ts.first()
        # The default should be the same as tree.
        svg1 = t.draw_svg(max_time="tree")
        self.verify_basic_svg(svg1)
        svg2 = t.draw_svg()
        assert svg1 == svg2
        svg3 = t.draw_svg(max_time="ts")
        assert svg1 != svg3
        svg4 = t.draw_svg(max_time=max(ts.tables.nodes.time))
        assert svg3 == svg4
        with pytest.warns(FutureWarning):
            svg5 = t.draw_svg(max_tree_height="tree")
        assert svg5 == svg1
        svg6 = t.draw_svg(max_time="tree", max_tree_height="i should be ignored")
        assert svg6 == svg1

    def test_time_scale_rank_and_max_time(self):
        # Make sure the rank height scale and max_time interact properly.
        ts = msprime.simulate(5, recombination_rate=2, random_seed=2)
        t = ts.first()
        # The default should be the same as tree.
        svg1 = t.draw_svg(max_time="tree", time_scale="rank", y_axis=True)
        self.verify_basic_svg(svg1)
        svg2 = t.draw_svg(time_scale="rank", y_axis=True)
        assert svg1 == svg2
        svg3 = t.draw_svg(max_time="ts", time_scale="rank", y_axis=True)
        assert svg1 != svg3
        self.verify_basic_svg(svg3)
        # Numeric max time not supported for rank scale.
        with pytest.raises(ValueError):
            t.draw_svg(max_time=2, time_scale="rank", y_axis=True)

    def test_min_tree_time(self):
        ts = self.get_ts_varying_min_times()
        t = ts.first()
        # The default should be the same as tree.
        svg1 = t.draw_svg(min_time="tree", y_axis=True)
        self.verify_basic_svg(svg1)
        svg2 = t.draw_svg(y_axis=True)
        assert svg1 == svg2
        svg3 = t.draw_svg(min_time="ts", y_axis=True)
        assert svg1 != svg3
        svg4 = t.draw_svg(min_time=min(ts.tables.nodes.time), y_axis=True)
        assert svg3 == svg4

    def test_min_ts_time(self):
        ts = self.get_ts_varying_min_times()
        svg1 = ts.draw_svg(y_axis=True)
        self.verify_basic_svg(svg1, width=200 * ts.num_trees)
        svg2 = ts.draw_svg(min_time="ts", y_axis=True)
        assert svg1 == svg2
        with pytest.raises(ValueError, match="vary in timescale"):
            ts.draw_svg(min_time="tree", y_axis=True)
        svg3 = ts.draw_svg(min_time=min(ts.tables.nodes.time), y_axis=True)
        assert svg2 == svg3

    def test_numeric_max_time_with_mutations_over_roots(self):
        max_time_value = 0.1  # Use a numeric max_time value
        params = {"y_ticks": [1.23], "y_axis": True}
        test_draw = {
            "svg_nomin": {},
            "svg_min": {"max_time": max_time_value},
            "svg_log_min": {"max_time": max_time_value, "time_scale": "log_time"},
        }

        t = self.get_mutations_over_roots_tree()
        assert t.tree_sequence.max_time > max_time_value

        for name, extra in test_draw.items():
            with pytest.warns(
                UserWarning, match="Mutations .* are above nodes which are not present"
            ):
                svg = t.draw_svg(**{**params, **extra})
            assert svg.count('class="tick"') == 1
            m = re.search(r'<g class="tick" transform="translate\((.*?)\)">', svg)
            assert m is not None
            translate_coords = [float(x) for x in m.group(1).split()]
            if name == "svg_nomin":
                # single tick within the plot region
                assert translate_coords[1] > 0
            else:
                assert translate_coords[1] < 0

    #
    # TODO: update the tests below here to check the new SVG based interface.
    #
    def test_all_edges_colour(self):
        t = self.get_binary_tree()
        colours = {u: "rgb({u},255,{u})".format(u=u) for u in t.nodes() if u != t.root}
        svg = t.draw(format="svg", edge_colours=colours)
        self.verify_basic_svg(svg)
        for colour in colours.values():
            assert svg.count(f"stroke:{colour}") > 0

    def test_unplotted_edge(self):
        t = self.get_binary_tree()
        colour = None
        colours = {0: colour}
        svg = t.draw(format="svg", edge_colours=colours)
        self.verify_basic_svg(svg)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("opacity:0") == 1

    def test_mutations_unknown_time(self):
        ts = self.get_simple_ts(use_mutation_times=True)
        svg = ts.draw_svg()
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        assert "unknown_time" not in svg
        ts = self.get_simple_ts(use_mutation_times=False)
        svg = ts.draw_svg()
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        assert svg.count("unknown_time") == ts.num_mutations

    def test_mutation_labels(self):
        t = self.get_binary_tree()
        labels = {u.id: "XXX" for u in t.mutations()}
        svg = t.draw(format="svg", mutation_labels=labels)
        self.verify_basic_svg(svg)
        assert svg.count("XXX") == t.num_mutations

    def test_one_mutation_label(self):
        t = self.get_binary_tree()
        labels = {0: "XXX"}
        svg = t.draw(format="svg", mutation_labels=labels)
        self.verify_basic_svg(svg)
        assert svg.count("XXX") == 1

    def test_no_mutation_labels(self):
        t = self.get_binary_tree()
        labels = {}
        svg = t.draw(format="svg", mutation_labels=labels)
        self.verify_basic_svg(svg)
        # Can't really test for much here if we don't understand the SVG

    def test_one_mutation_colour(self):
        t = self.get_binary_tree()
        colour = "rgb(0, 1, 2)"
        colours = {0: colour}
        svg = t.draw(format="svg", mutation_colours=colours)
        self.verify_basic_svg(svg)
        assert svg.count(f"fill:{colour}") == 1

    def test_all_mutations_colour(self):
        t = self.get_binary_tree()
        colours = {
            mut.id: f"rgb({mut.id}, {mut.id}, {mut.id})" for mut in t.mutations()
        }
        svg = t.draw(format="svg", mutation_colours=colours)
        self.verify_basic_svg(svg)
        for colour in colours.values():
            assert svg.count(f"fill:{colour}") == 1

    def test_unplotted_mutation(self):
        t = self.get_binary_tree()
        colour = None
        colours = {0: colour}
        svg = t.draw(format="svg", mutation_colours=colours)
        self.verify_basic_svg(svg)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("fill-opacity:0") == 1

    @pytest.mark.parametrize("all_muts", [False, True])
    @pytest.mark.parametrize("x_axis", [False, True])
    def test_extra_mutations(self, all_muts, x_axis):
        # The simple_ts has 2 mutations on an edge which spans the whole ts
        # One mut is within tree 1, the other within tree 3
        ts = self.get_simple_ts()
        extra_mut_copies = 0
        if all_muts:
            extra_mut_copies = 2 if x_axis else 1
        extra_right = ts.at_index(1)
        svg = extra_right.draw_svg(all_edge_mutations=all_muts, x_axis=x_axis)
        self.verify_basic_svg(svg)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("extra") == 1 * extra_mut_copies

        extra_right_and_left = ts.at_index(2)
        svg = extra_right_and_left.draw_svg(all_edge_mutations=all_muts, x_axis=x_axis)
        self.verify_basic_svg(svg)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("extra") == 2 * extra_mut_copies

        extra_left = ts.at_index(3)
        svg = extra_left.draw_svg(all_edge_mutations=all_muts, x_axis=x_axis)
        self.verify_basic_svg(svg)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("extra") == 1 * extra_mut_copies

    def test_max_time(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           0
        3   0           1
        4   0           2
        5   0           3
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       5       2
        0       1       5       3
        1       2       4       2
        1       2       4       3
        0       2       3       0
        0       2       3       1
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)

        svg1 = ts.at_index(0).draw()
        svg2 = ts.at_index(1).draw()
        # if not scaled to ts, the edge above node 0 is of a different length in both
        # trees, because the root is at a different height. We expect a group like
        # <path class="edge" d="M 0 0 V -46 H 22.5" /><text>0</text>
        str_pos = svg1.find(">0<")
        snippet1 = svg1[svg1.rfind("edge", 0, str_pos) : str_pos]
        str_pos = svg2.find(">0<")
        snippet2 = svg2[svg2.rfind("edge", 0, str_pos) : str_pos]
        assert snippet1 != snippet2

        svg1 = ts.at_index(0).draw(max_time="ts")
        svg2 = ts.at_index(1).draw(max_time="ts")
        with pytest.warns(FutureWarning):
            svg3 = ts.at_index(1).draw(max_tree_height="ts")
        assert svg3 == svg2
        # when scaled, node 3 should be at the *same* height in both trees, so the edge
        # definition should be the same
        self.verify_basic_svg(svg1)
        self.verify_basic_svg(svg2)
        str_pos = svg1.find(">0<")
        snippet1 = svg1[svg1.rfind("edge", 0, str_pos) : str_pos]
        str_pos = svg2.find(">0<")
        snippet2 = svg2[svg2.rfind("edge", 0, str_pos) : str_pos]
        assert snippet1 == snippet2

    def test_min_time(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   0           -1.11
        1   1           2.22
        2   1           2.22
        3   0           3.33
        4   0           4.44
        5   0           5.55
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       5       2
        0       1       5       3
        1       2       4       2
        1       2       4       3
        0       1       3       0
        0       2       3       1
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        svg1a = ts.at_index(0).draw_svg(y_axis=True)
        svg1b = ts.at_index(0).draw_svg(y_axis=True, min_time="ts")
        svg2a = ts.at_index(1).draw_svg(y_axis=True)
        svg2b = ts.at_index(1).draw_svg(y_axis=True, min_time="ts")
        # axis should start at -1.11
        assert svg1a == svg1b
        assert ">-1.11<" in svg1a
        # 2nd tree should be different depending on whether min_time is "tree" or "ts"
        assert svg2a != svg2b
        assert ">-1.11<" not in svg2a
        assert ">-1.11<" not in svg2b

    def test_draw_sized_tree(self):
        tree = self.get_binary_tree()
        svg = tree.draw_svg(size=(600, 400))
        self.verify_basic_svg(svg, width=600, height=400)

    def test_canvas_size_tree(self):
        tree = self.get_binary_tree()
        svg1 = tree.draw_svg(size=(200, 200))
        svg2 = tree.draw_svg(size=(200, 200), canvas_size=(700, 500))
        self.verify_basic_svg(svg1, width=200, height=200)
        self.verify_basic_svg(svg2, width=700, height=500)
        # height and width are specified in the starting <svg> tag
        assert svg1.startswith("<svg")
        assert svg2.startswith("<svg")
        # after the close of the tag, the two strings should be the same
        assert svg1[svg1.find(">") :] == svg2[svg2.find(">") :]

    def test_draw_bad_sized_treebox(self):
        tree = self.get_binary_tree()
        with pytest.raises(ValueError, match="too small to fit"):
            # Too small for plotbox
            tree.draw_svg(size=(20, 20))

    def test_draw_bad_sized_tree(self):
        tree = self.get_binary_tree()
        with pytest.raises(ValueError, match="too small to allow space"):
            # Too small for standard-sized labels on tree
            tree.draw_svg(size=(50, 50))

    def test_draw_simple_ts(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=1)
        svg = ts.draw_svg()
        self.verify_basic_svg(svg, width=200 * ts.num_trees)

    def test_draw_integer_breaks_ts(self):
        ts = msprime.sim_ancestry(
            5, sequence_length=10, recombination_rate=0.05, random_seed=1
        )
        assert ts.num_trees > 2
        svg = ts.draw_svg()
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        axis_pos = svg.find('class="x-axis"')
        for b in ts.breakpoints():
            assert b == round(b)
            assert svg.find(f">{b:.0f}<", axis_pos) != -1

    def test_draw_integer_times_ts(self):
        ts = msprime.sim_ancestry(
            5, population_size=5, sequence_length=10, model="dtwf", random_seed=1
        )
        svg = ts.draw_svg(y_axis=True)
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        axis_pos = svg.find('class="y-axis"')
        for t in ts.tables.nodes.time:
            assert t == round(t)
            assert svg.find(f">{t:.0f}<", axis_pos) != -1

    def test_draw_integer_times_tree(self):
        ts = msprime.sim_ancestry(
            5, population_size=5, sequence_length=10, model="dtwf", random_seed=1
        )
        svg = ts.first().draw_svg(y_axis=True)
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        axis_pos = svg.find('class="y-axis"')
        for t in ts.tables.nodes.time:
            assert t == round(t)
            assert svg.find(f">{t:.0f}<", axis_pos) != -1

    def test_draw_even_height_ts(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=1)
        svg = ts.draw_svg(max_time="tree")
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        with pytest.warns(FutureWarning):
            svg = ts.draw_svg(max_tree_height="tree")
        self.verify_basic_svg(svg, width=200 * ts.num_trees)

    def test_draw_sized_ts(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=1)
        svg = ts.draw_svg(size=(600, 400))
        self.verify_basic_svg(svg, width=600, height=400)

    def test_canvas_size_ts(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=1)
        svg1 = ts.draw_svg(size=(600, 400))
        svg2 = ts.draw_svg(size=(600, 400), canvas_size=(1000, 500))
        self.verify_basic_svg(svg1, width=600, height=400)
        self.verify_basic_svg(svg2, width=1000, height=500)
        # height and width are specified in the starting <svg> tag
        assert svg1.startswith("<svg")
        assert svg2.startswith("<svg")
        # after the close of the tag, the two strings should be the same
        assert svg1[svg1.find(">") :] == svg2[svg2.find(">") :]

    def test_time_scale(self):
        ts = msprime.simulate(4, random_seed=2)
        svg = ts.draw_svg(time_scale="time")
        self.verify_basic_svg(svg)
        svg = ts.draw_svg(time_scale="log_time")
        self.verify_basic_svg(svg)
        with pytest.warns(FutureWarning):
            svg2 = ts.draw_svg(tree_height_scale="log_time")
        assert svg2 == svg
        svg = ts.draw_svg(time_scale="rank")
        self.verify_basic_svg(svg)
        svg3 = ts.draw_svg(time_scale="rank", tree_height_scale="ignore me please")
        assert svg3 == svg
        for bad_scale in [0, "", "NOT A SCALE"]:
            with pytest.raises(ValueError):
                ts.draw_svg(time_scale=bad_scale)
            with pytest.raises(ValueError):
                with pytest.warns(FutureWarning):
                    ts.draw_svg(tree_height_scale=bad_scale)

    def test_x_scale(self):
        ts = msprime.simulate(4, random_seed=2)
        svg = ts.draw_svg(x_scale="physical")
        self.verify_basic_svg(svg)
        svg = ts.draw_svg(x_scale="treewise")
        self.verify_basic_svg(svg)

    def test_bad_x_scale(self):
        ts = msprime.simulate(4, random_seed=2)
        for bad_x_scale in ["te", "asdf", "", [], b"23"]:
            with pytest.raises(ValueError):
                ts.draw_svg(x_scale=bad_x_scale)

    def test_x_axis(self):
        tree = msprime.simulate(4, random_seed=2).first()
        svg = tree.draw_svg(x_axis=True)
        svg_no_css = svg[svg.find("</style>") :]
        assert "Genome position" in svg_no_css
        assert svg_no_css.count("axes") == 1
        assert svg_no_css.count("x-axis") == 1
        assert svg_no_css.count("y-axis") == 0

    def test_y_axis(self):
        tree = self.get_simple_ts().first()
        for hscale, label in [
            (None, "Time"),
            ("time", "Time"),
            ("log_time", "Time"),
            ("rank", "Node time"),
        ]:
            svg = tree.draw_svg(y_axis=True, time_scale=hscale)
            if hscale is not None:
                with pytest.warns(FutureWarning):
                    svg2 = tree.draw_svg(y_axis=True, tree_height_scale=hscale)
                assert svg2 == svg
                svg3 = tree.draw_svg(
                    y_axis=True, time_scale=hscale, tree_height_scale="ignore me please"
                )
                assert svg3 == svg
            svg_no_css = svg[svg.find("</style>") :]
            assert label in svg_no_css
            assert svg_no_css.count("axes") == 1
            assert svg_no_css.count("x-axis") == 0
            assert svg_no_css.count("y-axis") == 1
            assert svg_no_css.count("ticks") == 1
            assert svg_no_css.count('class="tick"') == len(
                {tree.time(u) for u in tree.nodes()}
            )

    def test_y_axis_noticks(self):
        tree = msprime.simulate(4, random_seed=2).first()
        svg = tree.draw_svg(y_axis=True, y_label="Time", y_ticks=[])
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("axes") == 1
        assert svg_no_css.count("x-axis") == 0
        assert svg_no_css.count("y-axis") == 1
        assert svg_no_css.count('"tick"') == 0

    def test_y_axis_tick_warning(sefl, caplog):
        tree = msprime.simulate(4, random_seed=2).first()
        upper = int(tree.time(tree.root))
        with caplog.at_level(logging.WARNING):
            tree.draw_svg(
                y_axis=True,
                y_label="Time",
                y_ticks={upper + 100: "above", upper / 3: "inside"},
            )
            assert (
                f"Ticks {{{upper + 100}: 'above'}} lie outside the plotted axis"
                in caplog.text
            )
        with caplog.at_level(logging.WARNING):
            tree.draw_svg(
                y_axis=True, y_label="Time", y_ticks={upper / 2: "inside", -1: "below"}
            )
            assert "Ticks {-1: 'below'} lie outside the plotted axis" in caplog.text

    def test_symbol_size(self):
        tree = msprime.simulate(4, random_seed=2, mutation_rate=8).first()
        sz = 24
        svg = tree.draw_svg(symbol_size=sz)
        svg_no_css = svg[svg.find("</style>") :]
        num_mutations = len([_ for _ in tree.mutations()])
        num_nodes = len([_ for _ in tree.nodes()])
        # Squares have 'height="sz" width="sz"'
        assert svg_no_css.count(f'"{sz}"') == tree.num_samples() * 2
        # Circles define a radius like 'r="sz/2"'
        assert svg_no_css.count(f'r="{sz / 2:g}"') == num_nodes - tree.num_samples()
        # Mutations draw a line on the cross using 'l sz,sz'
        assert svg_no_css.count(f"l {sz},{sz} ") == num_mutations

    def test_no_edges_invalid(self):
        full_ts = msprime.simulate(10, random_seed=2)
        tables = full_ts.dump_tables()
        tables.edges.clear()
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="To plot an empty tree sequence"):
            ts.draw_svg()
        with pytest.raises(ValueError, match="To plot an empty tree sequence"):
            ts.draw_svg(x_lim=[None, 1])
        with pytest.raises(ValueError, match="To plot an empty tree sequence"):
            ts.draw_svg(x_lim=[0, None])

    def test_no_edges_show_empty(self):
        # Should be possible to print empty trees if xlim=[0, seq_len]
        full_ts = msprime.simulate(10, random_seed=2)
        tables = full_ts.dump_tables()
        tables.edges.clear()
        ts = tables.tree_sequence()
        for time_scale in ("time", "log_time", "rank"):
            # SVG should just be a row of 10 sample nodes
            svg = ts.draw_svg(time_scale=time_scale, x_lim=[0, ts.sequence_length])
            self.verify_basic_svg(svg)
            assert svg.count("<rect") == 10  # Sample nodes are rectangles
            assert svg.count('<path class="edge') == 0
        svg = ts.draw_svg(force_root_branch=True, x_lim=[0, ts.sequence_length])
        self.verify_basic_svg(svg)
        assert svg.count("<rect") == 10
        assert svg.count('<path class="edge') == 10

    def test_no_edges_with_muts(self):
        # If there is a mutation above a sample, the root branches should be there too
        # And we should be able to plot the "empty" tree because the region still has
        # mutations
        full_ts = msprime.simulate(10, mutation_rate=1, random_seed=2)
        tables = full_ts.dump_tables()
        tables.edges.clear()
        ts = tables.tree_sequence().simplify()
        assert ts.num_mutations > 0  # Should have some singletons
        svg = ts.draw_svg()
        self.verify_basic_svg(svg)
        assert svg.count("<rect") == 10
        assert svg.count('<path class="edge') == 10
        assert svg.count('<path class="sym"') == ts.num_mutations
        assert svg.count('<line class="sym"') == ts.num_sites

    def test_empty_flanks(self):
        ts = msprime.simulate(10, random_seed=2, recombination_rate=0.1)
        assert ts.num_trees == 2
        assert 0.2 < ts.first().interval.right < 0.8
        degree_2_ts = ts.keep_intervals([[0.2, 0.8]])
        svg = degree_2_ts.draw_svg(y_axis=False)
        assert svg.count('class="tick"') == 3
        assert svg.count('<text class="lab">0.2') == 1
        assert svg.count('<text class="lab">0.8') == 1
        degree_1_ts = ts.keep_intervals([[0.05, 0.15]])
        svg = degree_1_ts.draw_svg(y_axis=False)
        assert svg.count('class="tick"') == 2
        assert svg.count('<text class="lab">0.05') == 1
        assert svg.count('<text class="lab">0.15') == 1

    def test_bad_xlim(self):
        ts = msprime.simulate(10, random_seed=2)
        svg = ts.draw_svg(x_lim=[None, None])
        self.verify_basic_svg(svg)
        with pytest.raises(ValueError, match="must be a list of length 2"):
            ts.draw_svg(x_lim=[0])
        with pytest.raises(TypeError, match="must be numeric"):
            ts.draw_svg(x_lim=[0, "a"])
        with pytest.raises(ValueError, match="must be less than"):
            ts.draw_svg(x_lim=[0.5, 0.5])
        with pytest.raises(ValueError, match="cannot be negative"):
            ts.draw_svg(x_lim=[-1, 0])
        with pytest.raises(ValueError, match="cannot be greater than"):
            ts.draw_svg(x_lim=[0, ts.sequence_length * 2])

    def test_xlim_on_empty(self):
        full_ts = msprime.simulate(10, random_seed=2)
        tables = full_ts.dump_tables()
        tables.edges.clear()
        ts = tables.tree_sequence()
        ts.draw_svg(x_lim=[0, ts.sequence_length])
        with pytest.raises(ValueError, match="whole region is empty"):
            ts.draw_svg(x_lim=[0, 0.9])

    def test_xlim_edge_cases(self):
        tables = msprime.simulate(10, random_seed=2, mutation_rate=10).dump_tables()
        # Delete edges but keep mutations
        old_sites = tables.sites.copy()
        tables.keep_intervals([[0.4, 0.6]], simplify=False)
        tables.sites.set_columns(**old_sites.asdict())
        ts = tables.tree_sequence().simplify(filter_sites=False)
        assert np.any(ts.tables.sites.position < 0.4)
        assert np.any(ts.tables.sites.position > 0.6)
        for x_lim in [None, (0, 1), (None, 1), (0, None)]:
            # All have sites in the deleted region, so should have all trees
            svg = ts.draw_svg(x_lim=x_lim)
            self.verify_basic_svg(svg, width=200 * 3)
            assert svg.count('class="tree ') == 3
        tables.sites.clear()
        tables.mutations.clear()
        ts = tables.tree_sequence().simplify()
        for x_lim, n_trees in {None: 1, (0, 1): 3, (None, 1): 2, (0, None): 2}.items():
            # No sites in the deleted region, so x_lim determines # plotted trees
            svg = ts.draw_svg(x_lim=x_lim)
            self.verify_basic_svg(svg, width=200 * n_trees)
            assert svg.count('class="tree ') == n_trees

    def test_xlim_maintains_tree_ids(self):
        ts = self.get_simple_ts()
        breaks = ts.breakpoints(as_array=True)
        svg = ts.draw_svg(x_lim=[breaks[1], breaks[4]])
        assert "t0" not in svg
        assert "t4" not in svg
        svg = ts.draw_svg(
            x_lim=[np.nextafter(breaks[1], 0), np.nextafter(breaks[4], 1)]
        )
        assert "t0" in svg
        assert "t4" in svg

    def test_xlim_maintains_site_and_mutation_ids(self):
        ts = self.get_simple_ts()
        breaks = ts.breakpoints(as_array=True)
        tree_svg = ts.at_index(1).draw_svg(x_axis=True)

        ts_svg = ts.draw_svg(x_lim=[breaks[1], breaks[2]])
        assert re.findall(r">\d+<", tree_svg) == re.findall(r">\d+<", ts_svg)  # labels
        for identifier in ["s", "m"]:
            tree_ids = re.findall(rf"{identifier}\d+", tree_svg)
            assert len(tree_ids) > 0
            ts_ids = re.findall(rf"{identifier}\d+", ts_svg)
            assert tree_ids == ts_ids

        site_pos0_in_tree1 = next(ts.at_index(1).sites()).position
        ts_svg = ts.draw_svg(x_lim=[site_pos0_in_tree1, breaks[2]])
        assert re.findall(r">\d+<", tree_svg) == re.findall(r">\d+<", ts_svg)  # labels
        for identifier in ["s", "m"]:
            tree_ids = re.findall(rf"{identifier}\d+", tree_svg)
            ts_ids = re.findall(rf"{identifier}\d+", ts_svg)
            assert tree_ids == ts_ids

        ts_svg = ts.draw_svg(x_lim=[np.nextafter(site_pos0_in_tree1, 1), breaks[2]])
        assert re.findall(r">\d+<", tree_svg) != re.findall(r">\d+<", ts_svg)  # labels
        for identifier in ["s", "m"]:
            tree_ids = re.findall(rf"{identifier}\d+", tree_svg)
            ts_ids = re.findall(rf"{identifier}\d+", ts_svg)
            assert tree_ids != ts_ids

    def test_xlim_with_ranks(self):
        ts = self.get_simple_ts()
        xlim = ts.breakpoints(as_array=True)[:2]  # plot first tree only
        svg = ts.draw_svg(x_lim=xlim, time_scale="rank", y_axis=True, y_gridlines=True)
        # excluding ".grid" in the stylesheet, there should be only 4 y-axis steps
        # for a 4 tip tree with all samples at 0: simplest check is to count gridlines
        assert len(re.findall(r"[^.]grid", svg)) == 4

    def test_half_truncated(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = ts.delete_intervals([[0.4, 0.6]])
        svg = ts.draw_svg(x_lim=(0.5, 0.7), y_axis=False)
        # Only one tree and one tick shown (leftmost is an empty region)
        assert svg.count('class="tree ') == 1
        assert svg.count('class="tick"') == 1

    def test_tree_root_branch(self):
        # in the simple_ts, there are root mutations in the first tree but not the last
        ts = self.get_simple_ts()
        tree_with_root_mutations = ts.at_index(0)
        root1 = tree_with_root_mutations.root
        tree_without_root_mutations = ts.at_index(-1)
        root2 = tree_without_root_mutations.root
        svg1 = tree_with_root_mutations.draw_svg()
        svg2a = tree_without_root_mutations.draw_svg()
        svg2b = tree_without_root_mutations.draw_svg(force_root_branch=True)
        self.verify_basic_svg(svg1)
        self.verify_basic_svg(svg2a)
        self.verify_basic_svg(svg2b)
        # Last <path> should be the root branch, if it exists
        edge_str = '<path class="edge root" d='
        str_pos1 = svg1.rfind(edge_str, 0, svg1.find(f">{root1}<"))
        assert edge_str not in svg2a
        str_pos2b = svg2b.rfind(edge_str, 0, svg2b.find(f">{root2}<"))
        snippet1 = svg1[str_pos1 + len(edge_str) : svg1.find(">", str_pos1)]
        snippet2b = svg2b[str_pos2b + len(edge_str) : svg2b.find(">", str_pos2b)]
        assert snippet1.startswith('"M 0 0')
        assert snippet2b.startswith('"M 0 0')
        assert "H 0" in snippet1
        assert "H 0" in snippet2b

    def test_debug_box(self):
        ts = self.get_simple_ts()
        svg = ts.first().draw_svg(debug_box=True)
        self.verify_basic_svg(svg)
        assert svg.count("outer_plotbox") == 1
        assert svg.count("inner_plotbox") == 1
        svg = ts.draw_svg(debug_box=True)
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        assert svg.count("outer_plotbox") == ts.num_trees + 1
        assert svg.count("inner_plotbox") == ts.num_trees + 1

    @pytest.mark.parametrize("max_trees", [-1, 0, 1])
    def test_bad_max_num_trees(self, max_trees):
        ts = self.get_simple_ts()
        with pytest.raises(ValueError, match="at least 2"):
            ts.draw_svg(max_num_trees=max_trees)

    @pytest.mark.parametrize("max_trees", [2, 4, 9])
    def test_max_num_trees(self, max_trees):
        ts = msprime.sim_ancestry(
            3, sequence_length=100, recombination_rate=0.1, random_seed=1
        )
        ts = msprime.sim_mutations(ts, rate=0.1, random_seed=1)
        assert ts.num_trees > 10
        num_sites = 0
        num_unplotted_sites = 0
        svg = ts.draw_svg(max_num_trees=max_trees)
        for tree in ts.trees():
            if (
                tree.index < (max_trees + 1) // 2
                or ts.num_trees - tree.index <= max_trees // 2
            ):
                num_sites += tree.num_sites
                assert re.search(rf"t{tree.index}[^\d]", svg) is not None
            else:
                assert re.search(rf"t{tree.index}[^\d]", svg) is None
                num_unplotted_sites += tree.num_sites
        assert num_unplotted_sites > 0
        site_strings_in_stylesheet = svg.count(".site")
        assert svg.count("site") - site_strings_in_stylesheet == num_sites
        self.verify_basic_svg(svg, width=200 * (max_trees + 1))

    def test_edge_ids(self):
        ts = self.get_simple_ts()
        for tree in ts.trees():
            svg = tree.draw_svg()
            mut_nodes = {m.node for m in tree.mutations()}
            assert svg.count('"edge root"') == (1 if tree.root in mut_nodes else 0)
            edges = {tree.edge(u) for u in tree.nodes()}
            for e in range(tree.num_edges):
                assert svg.count(f'"edge e{e}"') == (1 if e in edges else 0)

    def test_draw_tree_symbol_titles(self):
        tree = self.get_binary_tree()
        assert tree.tree_sequence.num_mutations > 0
        svg = tree.draw_svg(
            node_titles={u: f"NODE{u}$" for u in tree.nodes()},
            mutation_titles={m.id: f"MUT{m.id}$" for m in tree.mutations()},
        )
        for u in tree.nodes():
            assert svg.count(f"<title>NODE{u}$</title>") == 1
        for m in tree.mutations():
            assert svg.count(f"<title>MUT{m.id}$</title>") == 1
        self.verify_basic_svg(svg)

    def test_nodraw_x_axis(self):
        ts = msprime.sim_ancestry(
            1, sequence_length=100, recombination_rate=0.1, random_seed=1
        )
        svg = ts.first().draw_svg(x_axis=False, y_axis=False)
        assert 'class="x-axis"' not in svg

    def test_x_regions_ts(self):
        ts = msprime.sim_ancestry(
            3, sequence_length=100, recombination_rate=0.1, random_seed=1
        )
        regions = [(0, 10), (9, 20), (50, 90)]
        svg = ts.draw_svg(
            x_regions={r: f"reg{'ABC'[i]}" for i, r in enumerate(regions)}
        )
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        assert svg.count("x-regions") == 2  # one in stylesheet, one in svg
        assert svg.count("r0") == 1
        assert svg.count("r1") == 1
        assert svg.count("r2") == 1
        assert svg.count("r3") == 0
        assert svg.count("regA") == 1
        assert svg.count("regB") == 1
        assert svg.count("regC") == 1
        # "rect" string present for 6 samples in each tree + 3 regions + 1 in stylesheet
        assert svg.count("rect") == 6 * ts.num_trees + 3 + 1

    def test_x_regions_tree(self):
        ts = msprime.sim_ancestry(
            3, sequence_length=100, recombination_rate=0.1, random_seed=1
        )
        svg = ts.first().draw_svg(x_regions={(0, 10): "💩"})
        assert svg.count("💩") == 0
        svg = ts.first().draw_svg(x_axis=True, x_regions={(0, 10): "💩"})
        assert svg.count("💩") == 1

    def test_unsupported_x_regions(self):
        ts = msprime.sim_ancestry(
            1, sequence_length=100, recombination_rate=0.1, random_seed=1
        )
        ts.draw_svg(x_scale="treewise")
        with pytest.raises(ValueError, match="not supported for treewise"):
            ts.draw_svg(x_scale="treewise", x_regions={(0, 1): "bad"})

    def test_bad_x_regions(self):
        ts = msprime.sim_ancestry(
            1, sequence_length=100, recombination_rate=0.1, random_seed=1
        )
        with pytest.raises(ValueError, match="Invalid coordinates"):
            ts.draw_svg(x_regions={(-1, 1): "bad"})
        with pytest.raises(ValueError, match="Invalid coordinates"):
            ts.draw_svg(x_regions={(0, ts.sequence_length + 1): "bad"})
        with pytest.raises(ValueError, match="Invalid coordinates"):
            ts.draw_svg(x_regions={(1, 0): "bad"})

    def test_title(self):
        ts = msprime.sim_ancestry(1, sequence_length=100, random_seed=1)
        svg = ts.draw_svg(title="This is a title")
        assert "This is a title" in svg
        svg = ts.first().draw_svg(title="This is another title")
        assert "This is another title" in svg

    def test_bad_ts_order(self):
        ts = msprime.sim_ancestry(1, sequence_length=100, random_seed=1)
        with pytest.raises(ValueError, match="Unknown display order"):
            ts.draw_svg(order=(ts.first().nodes(order="minlex_postorder")))

    def test_good_tree_order(self):
        ts = msprime.sim_ancestry(1, sequence_length=100, random_seed=1)
        ts.first().draw_svg(order=(ts.first().nodes(order="minlex_postorder")))

    def test_nonpostorder_tree_order(self):
        tree = tskit.Tree.generate_balanced(10)
        with pytest.raises(ValueError, match="must be passed in postorder"):
            tree.draw_svg(order=(tree.nodes(order="preorder")))

    def test_only_subset_nodes_in_rank(self, caplog):
        tree = tskit.Tree.generate_comb(100)
        # Only show the last few tips of the comb. We should only use the ranks
        # from those tip times, so ticks > 5 should raise a warning
        with caplog.at_level(logging.WARNING):
            tree.draw_svg(
                order=tree.nodes(root=105, order="minlex_postorder"),
                time_scale="rank",
                y_axis=True,
                y_ticks=[0, 1, 6],
            )
            assert "lie outside the plotted axis" not in caplog.text
        with caplog.at_level(logging.WARNING):
            tree.draw_svg(
                order=tree.nodes(root=105, order="minlex_postorder"),
                time_scale="rank",
                y_axis=True,
                y_ticks=[0, 1, 10],
            )
            assert "Ticks {10: '10'} lie outside the plotted axis" in caplog.text

    def test_polytomy_collapsing(self):
        tree = tskit.Tree.generate_balanced(
            20, arity=4, tracked_samples=np.arange(2, 8)
        )
        svg = tree.draw_svg(pack_untracked_polytomies=True)
        # Should have one collapsed node (untracked samples 8 and 9)
        # and two "polytomy lines" (from nodes 21 and 28 (the root))
        assert svg.count('class="polytomy"') == 2  # poolytomy lines
        collapsed_symbol = re.search("<polygon[^>]*>", svg)
        assert collapsed_symbol is not None
        assert collapsed_symbol.group(0).count("sym") == 1
        assert collapsed_symbol.group(0).count("multi") == 1

    @pytest.mark.parametrize(
        "tree_or_ts",
        [tskit.Tree.generate_comb(3), tskit.Tree.generate_comb(3).tree_sequence],
    )
    def test_preamble(self, tree_or_ts):
        embed = tskit.Tree.generate_comb(4)  # svg string to embed
        svg = tree_or_ts.draw_svg(
            size=(200, 200),
            canvas_size=(400, 200),
            preamble=embed.draw_svg(
                root_svg_attributes={"x": 200, "class": "embedded"}
            ),
        )
        self.verify_basic_svg(svg, width=400, height=200)
        assert svg.count("<svg") == 2
        assert svg.count('class="embedded"') == 1

    @pytest.mark.parametrize(
        "tree_or_ts",
        [tskit.Tree.generate_comb(3), tskit.Tree.generate_comb(3).tree_sequence],
    )
    def test_non_svg_preamble(self, tree_or_ts):
        svg = tree_or_ts.draw_svg(
            size=(200, 200), canvas_size=(400, 200), preamble="<UnbalancedTag>"
        )
        with pytest.raises(xml.etree.ElementTree.ParseError):
            self.verify_basic_svg(svg, width=400, height=200)


class TestDrawKnownSvg(TestDrawSvgBase):
    """
    Compare against known files
    """

    def verify_known_svg(self, svg, filename, save=False, **kwargs):
        # expected SVG files can be inspected in tests/data/svg/*.svg
        svg = xml.dom.minidom.parseString(
            svg
        ).toprettyxml()  # Prettify for easy viewing
        self.verify_basic_svg(svg, **kwargs)
        svg_fn = pathlib.Path(__file__).parent / "data" / "svg" / filename
        if save:
            logging.warning(f"Overwriting SVG file `{svg_fn}` with new version")
            with open(svg_fn, "w") as file:
                file.write(svg)
        with open(svg_fn, "rb") as file:
            expected_svg = file.read()
        self.assertXmlEquivalentOutputs(svg, expected_svg)

    def test_known_svg_tree_no_mut(self, overwrite_viz, draw_plotbox):
        tree = self.get_simple_ts().at_index(-1)
        svg = tree.draw_svg(
            root_svg_attributes={"id": "XYZ"},
            style=".edge {stroke: blue}",
            debug_box=draw_plotbox,
        )
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("axes") == 0
        assert svg_no_css.count("x-axis") == 0
        assert svg_no_css.count("y-axis") == 0
        self.verify_known_svg(svg, "tree.svg", overwrite_viz)

    def test_known_svg_tree_x_axis(self, overwrite_viz, draw_plotbox):
        tree = self.get_simple_ts().at_index(1)
        svg = tree.draw_svg(
            x_axis=True,
            x_label="pos on genome",
            size=(400, 200),
            debug_box=draw_plotbox,
        )
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("axes") == 1
        assert svg_no_css.count("x-axis") == 1
        assert svg_no_css.count("title") == 1
        assert svg_no_css.count("y-axis") == 0
        self.verify_known_svg(svg, "tree_x_axis.svg", overwrite_viz, width=400)

    def test_known_svg_tree_y_axis_rank(self, overwrite_viz, draw_plotbox):
        tree = self.get_simple_ts().at_index(1)
        label = "Time (relative steps)"
        svg = tree.draw_svg(
            y_axis=True,
            y_label=label,
            y_gridlines=True,
            time_scale="rank",
            style=".y-axis line.grid {stroke: #CCCCCC}",
            debug_box=draw_plotbox,
        )
        svg_no_css = svg[svg.find("</style>") :]
        node_times = [tree.time(u) for u in tree.nodes()]
        assert label in svg_no_css
        assert svg_no_css.count('class="grid"') == len(set(node_times))
        assert svg_no_css.count("axes") == 1
        assert svg_no_css.count("x-axis") == 0
        assert svg_no_css.count("y-axis") == 1
        assert svg_no_css.count("title") == 1
        self.verify_known_svg(svg, "tree_y_axis_rank.svg", overwrite_viz)

    def test_known_svg_tree_both_axes(self, overwrite_viz, draw_plotbox):
        tree = self.get_simple_ts().at_index(-1)
        svg = tree.draw_svg(x_axis=True, y_axis=True, debug_box=draw_plotbox)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("axes") == 1
        assert svg_no_css.count("x-axis") == 1
        assert svg_no_css.count("y-axis") == 1
        assert svg_no_css.count("title") == 2
        self.verify_known_svg(svg, "tree_both_axes.svg", overwrite_viz)

    def test_known_svg_tree_root_mut(self, overwrite_viz, draw_plotbox):
        tree = self.get_simple_ts().at_index(0)  # Tree 0 has a few mutations above root
        svg = tree.draw_svg(debug_box=draw_plotbox)
        self.verify_known_svg(svg, "tree_muts.svg", overwrite_viz)

    def test_known_svg_tree_mut_all_edge(self, overwrite_viz, draw_plotbox):
        tree = self.get_simple_ts().at_index(1)
        size = (300, 400)
        svg = tree.draw_svg(
            size=size,
            debug_box=draw_plotbox,
            all_edge_mutations=True,
            x_axis=True,
            title="All mutations tree: background shading shown",
        )
        self.verify_known_svg(
            svg, "tree_muts_all_edge.svg", overwrite_viz, width=size[0], height=size[1]
        )

    def test_known_svg_tree_timed_root_mut(self, overwrite_viz, draw_plotbox):
        tree = self.get_simple_ts(use_mutation_times=True).at_index(0)
        # Also look at y_axis=right
        svg = tree.draw_svg(debug_box=draw_plotbox, y_axis="right")
        self.verify_known_svg(svg, "tree_timed_muts.svg", overwrite_viz)

    def test_known_svg_ts(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts()
        svg = ts.draw_svg(debug_box=draw_plotbox)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("axes") == 1
        assert svg_no_css.count("x-axis") == 1
        assert svg_no_css.count("y-axis") == 0
        assert svg_no_css.count('class="site ') == ts.num_sites
        assert svg_no_css.count('class="mut ') == ts.num_mutations * 2
        self.verify_known_svg(svg, "ts.svg", overwrite_viz, width=200 * ts.num_trees)

    def test_known_svg_ts_title(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts()
        svg = ts.draw_svg(title="The main plot title", debug_box=draw_plotbox)
        self.verify_known_svg(
            svg, "ts_title.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_no_axes(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts()
        svg = ts.draw_svg(x_axis=False, debug_box=draw_plotbox)
        svg_no_css = svg[svg.find("</style>") :]
        assert svg_no_css.count("axes") == 0
        assert svg_no_css.count("x-axis") == 0
        assert svg_no_css.count("y-axis") == 0
        assert 'class="site ' not in svg_no_css
        assert svg_no_css.count('class="mut ') == ts.num_mutations
        self.verify_known_svg(
            svg, "ts_no_axes.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_internal_sample(self, overwrite_viz, draw_plotbox):
        ts = tsutil.jiggle_samples(self.get_simple_ts())
        svg = ts.draw_svg(
            root_svg_attributes={"id": "XYZ"},
            style="#XYZ .leaf .sym {fill: magenta} #XYZ .sample > .sym {fill: cyan}",
            debug_box=draw_plotbox,
        )
        self.verify_known_svg(
            svg, "internal_sample_ts.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_highlighted_mut(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts()
        style = (
            ".edge {stroke: grey}"
            ".mut .sym{stroke:pink} .mut text{fill:pink}"
            ".mut.m2 .sym,.m2>line, .m2>.node .edge{stroke:red} .mut.m2 text{fill:red}"
            ".mut.m3 .sym,.m3>line, .m3>.node .edge{stroke:cyan} .mut.m3 text{fill:cyan}"
            ".mut.m4 .sym,.m4>line, .m4>.node .edge{stroke:blue} .mut.m4 text{fill:blue}"
        )
        svg = ts.draw_svg(style=style, debug_box=draw_plotbox)
        self.verify_known_svg(
            svg, "ts_mut_highlight.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_rank(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts()
        svg1 = ts.draw_svg(time_scale="rank", y_axis=True, debug_box=draw_plotbox)
        ts = self.get_simple_ts(use_mutation_times=True)
        svg2 = ts.draw_svg(time_scale="rank", y_axis=True, debug_box=draw_plotbox)
        assert svg1.count('class="site ') == ts.num_sites
        assert svg1.count('class="mut ') == ts.num_mutations * 2
        assert svg1.replace(" unknown_time", "") == svg2  # Trim the unknown_time class
        self.verify_known_svg(
            svg1, "ts_rank.svg", overwrite_viz, width=200 * ts.num_trees
        )

    @pytest.mark.skip(reason="Fails on CI as OSX gives different random numbers")
    def test_known_svg_nonbinary_ts(self, overwrite_viz, draw_plotbox):
        ts = self.get_nonbinary_ts()
        svg = ts.draw_svg(time_scale="log_time", debug_box=draw_plotbox)
        assert svg.count('class="site ') == ts.num_sites
        assert svg.count('class="mut ') == ts.num_mutations * 2
        self.verify_known_svg(
            svg, "ts_nonbinary.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_plain(self, overwrite_viz, draw_plotbox):
        """
        Plain style: no background shading and a variable scale X axis with no sites
        """
        ts = self.get_simple_ts()
        svg = ts.draw_svg(x_scale="treewise", debug_box=draw_plotbox)
        assert svg.count('class="site ') == 0
        assert svg.count('class="mut ') == ts.num_mutations
        self.verify_known_svg(
            svg, "ts_plain.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_plain_no_xlab(self, overwrite_viz, draw_plotbox):
        """
        Plain style: no background shading and a variable scale X axis with no sites
        """
        ts = self.get_simple_ts()
        svg = ts.draw_svg(x_scale="treewise", x_label="", debug_box=draw_plotbox)
        assert "Genome position" not in svg
        self.verify_known_svg(
            svg, "ts_plain_no_xlab.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_plain_y(self, overwrite_viz, draw_plotbox):
        """
        Plain style: no background shading and a variable scale X axis with no sites
        """
        ts = self.get_simple_ts()
        ticks = [0, 5, 10]
        svg = ts.draw_svg(
            x_scale="treewise",
            y_axis=True,
            y_ticks=ticks,
            y_gridlines=True,
            style=".y-axis line.grid {stroke: #CCCCCC}",
            debug_box=draw_plotbox,
        )
        self.verify_known_svg(
            svg, "ts_plain_y.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_with_xlabel(self, overwrite_viz, draw_plotbox):
        """
        Style with X axis label
        """
        ts = self.get_simple_ts()
        x_label = "genomic position (bp)"
        svg = ts.draw_svg(x_label=x_label, debug_box=draw_plotbox)
        assert x_label in svg
        self.verify_known_svg(
            svg, "ts_xlabel.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_y_axis(self, overwrite_viz, draw_plotbox):
        tables = self.get_simple_ts().dump_tables()
        # set units
        tables.time_units = "generations"
        ts = tables.tree_sequence()
        svg = ts.draw_svg(y_axis=True, debug_box=draw_plotbox)
        assert "Time ago (generations)" in svg
        self.verify_known_svg(
            svg, "ts_y_axis.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_y_axis_regular(self, overwrite_viz, draw_plotbox):
        # This should have gridlines
        ts = self.get_simple_ts()
        ticks = np.arange(0, max(ts.tables.nodes.time), 1)
        svg = ts.draw_svg(
            y_axis=True, y_ticks=ticks, y_gridlines=True, debug_box=draw_plotbox
        )
        assert svg.count('class="grid"') == len(ticks)
        self.verify_known_svg(
            svg, "ts_y_axis_regular.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_y_axis_log(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts()
        svg = ts.draw_svg(
            y_axis=True,
            y_label="Time (log scale)",
            time_scale="log_time",
            debug_box=draw_plotbox,
        )
        self.verify_known_svg(
            svg, "ts_y_axis_log.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_mutation_times(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts(use_mutation_times=True)
        # also look at y_axis="right"
        svg = ts.draw_svg(debug_box=draw_plotbox, y_axis="right")
        assert svg.count('class="site ') == ts.num_sites
        assert svg.count('class="mut ') == ts.num_mutations * 2
        self.verify_known_svg(
            svg, "ts_mut_times.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_titles(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts(use_mutation_times=True)
        svg = ts.draw_svg(
            node_titles={nd.id: f"NoDe{nd.id}!" for nd in ts.nodes()},
            mutation_titles={m.id: f"MuT{m.id}!" for m in ts.mutations()},
            debug_box=draw_plotbox,
        )
        for nd in ts.nodes():
            if nd.is_sample():
                assert svg.count(f"<title>NoDe{nd.id}!</title>") == ts.num_trees
            else:
                assert f"<title>NoDe{nd.id}!</title>" in svg
        for m in ts.mutations():
            assert (
                svg.count(f"<title>MuT{m.id}!</title>") == 2
            )  # 1 on tree, 1 on x-axis
        self.verify_known_svg(
            svg, "ts_mut_times_titles.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_mutation_times_logscale(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts(use_mutation_times=True)
        svg = ts.draw_svg(time_scale="log_time", debug_box=draw_plotbox)
        assert svg.count('class="site ') == ts.num_sites
        assert svg.count('class="mut ') == ts.num_mutations * 2
        self.verify_known_svg(
            svg, "ts_mut_times_logscale.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_mut_no_edges(self, overwrite_viz, draw_plotbox):
        # An example with some muts on axis but not on a visible node
        ts = msprime.simulate(10, random_seed=2, mutation_rate=1)
        tables = ts.dump_tables()
        tables.edges.clear()
        tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)
        ts_no_edges = tables.tree_sequence()
        with pytest.warns(UserWarning, match="nodes which are not present"):
            svg = ts_no_edges.draw_svg(debug_box=draw_plotbox)
            self.verify_known_svg(
                svg,
                "ts_mutations_no_edges.svg",
                overwrite_viz,
                width=200 * ts.num_trees,
            )

    def test_known_svg_ts_timed_mut_no_edges(self, overwrite_viz, draw_plotbox):
        # An example with some muts on axis but not on a visible node
        ts = msprime.simulate(10, random_seed=2, mutation_rate=1)
        tables = ts.dump_tables()
        tables.edges.clear()
        tables.mutations.time = np.arange(
            ts.num_mutations, dtype=tables.mutations.time.dtype
        )
        ts_no_edges = tables.tree_sequence()

        with pytest.warns(UserWarning, match="nodes which are not present"):
            svg = ts_no_edges.draw_svg(debug_box=draw_plotbox)
            self.verify_known_svg(
                svg,
                "ts_mutations_timed_no_edges.svg",
                overwrite_viz,
                width=200 * ts.num_trees,
            )

    def test_known_svg_ts_multiroot(self, overwrite_viz, draw_plotbox):
        tables = wf.wf_sim(
            6,
            5,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=8,
        )
        tables.sort()
        ts = tables.tree_sequence().simplify()
        tables = tsutil.jukes_cantor(ts, 10, mu=0.1, seed=123).dump_tables()
        # Set unknown times, so we are msprime 0.7.4 and 1.0.0 compatible
        tables.mutations.time = np.full(tables.mutations.num_rows, tskit.UNKNOWN_TIME)
        svg = tables.tree_sequence().draw_svg(
            y_axis=True, y_gridlines=True, debug_box=draw_plotbox
        )
        self.verify_known_svg(
            svg, "ts_multiroot.svg", overwrite_viz, width=200 * ts.num_trees
        )
        assert "Time ago (generations)" in svg

    def test_known_svg_ts_xlim(self, overwrite_viz, draw_plotbox):
        ts = self.get_simple_ts()
        svg = ts.draw_svg(x_lim=[0.051, 0.9], debug_box=draw_plotbox)
        num_trees = sum(1 for b in ts.breakpoints() if 0.051 <= b < 0.9) + 1
        self.verify_known_svg(svg, "ts_x_lim.svg", overwrite_viz, width=200 * num_trees)

    @pytest.mark.skipif(IS_WINDOWS, reason="Msprime gives different result on Windows")
    def test_known_max_num_trees(self, overwrite_viz, draw_plotbox):
        max_trees = 5
        ts = msprime.sim_ancestry(
            3, sequence_length=100, recombination_rate=0.1, random_seed=1
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        assert ts.num_trees > 10
        first_break = next(ts.trees()).interval.right
        # limit to just past the first tree
        svg = ts.draw_svg(
            max_num_trees=max_trees,
            x_lim=(first_break + 0.1, ts.sequence_length - 0.1),
            y_axis=True,
            time_scale="log_time",
            debug_box=draw_plotbox,
        )
        self.verify_known_svg(
            svg, "ts_max_trees.svg", overwrite_viz, width=200 * (max_trees + 1)
        )

    @pytest.mark.skipif(IS_WINDOWS, reason="Msprime gives different result on Windows")
    def test_known_max_num_trees_treewise(self, overwrite_viz, draw_plotbox):
        max_trees = 5
        ts = msprime.sim_ancestry(
            3, sequence_length=100, recombination_rate=0.1, random_seed=1
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        assert ts.num_trees > 10
        first_break = next(ts.trees()).interval.right
        svg = ts.draw_svg(
            max_num_trees=max_trees,
            x_lim=(first_break + 0.1, ts.sequence_length - 0.1),
            y_axis=True,
            x_scale="treewise",
            debug_box=draw_plotbox,
        )
        self.verify_known_svg(
            svg, "ts_max_trees_treewise.svg", overwrite_viz, width=200 * (max_trees + 1)
        )

    def test_known_svg_tree_collapsed(self, overwrite_viz, draw_plotbox):
        tree = tskit.Tree.generate_balanced(8)
        remove_nodes = set()
        remove_nodes_below = {8, 13}
        for u in remove_nodes_below:
            subtree_nodes = set(tree.nodes(root=u)) - {u}
            remove_nodes.update(subtree_nodes)
        order = [
            u for u in tree.nodes(order="minlex_postorder") if u not in remove_nodes
        ]
        svg = tree.draw_svg(order=order, debug_box=draw_plotbox)
        assert svg.count("multi") == len(remove_nodes_below)
        assert svg.count(">+2<") == 1  # One tip has 2 samples below it
        assert svg.count(">+4<") == 1  # Another tip has 4 samples below it
        for u in order:
            assert f'n{u}"' in svg or f"n{u} " in svg
        for u in remove_nodes:
            assert f'n{u}"' not in svg and f"n{u} " not in svg
        self.verify_known_svg(svg, "tree_simple_collapsed.svg", overwrite_viz)

    def test_known_svg_tree_subtree(self, overwrite_viz, draw_plotbox):
        tree = tskit.Tree.generate_balanced(8)
        order = [u for u in tree.nodes(root=10, order="minlex_postorder")]
        # The balanced tree has all descendants of nodes 10 with IDs < 10
        assert np.all(np.array(order) <= 10)
        svg = tree.draw_svg(order=order, debug_box=draw_plotbox)
        for u in order:
            assert f'n{u}"' in svg or f"n{u} " in svg
        for u in set(tree.nodes()) - set(order):
            assert f'n{u}"' not in svg and f"n{u} " not in svg
        self.verify_known_svg(svg, "tree_subtree.svg", overwrite_viz, has_root=False)

    def test_known_svg_tree_subtrees_with_collapsed(self, overwrite_viz, draw_plotbox):
        # Two subtrees, one with a collapsed node below node 16
        tree = tskit.Tree.generate_balanced(16)
        roots = [22, 25]
        order = []
        remove_nodes_below = 16
        remove_nodes = set(tree.nodes(root=remove_nodes_below)) - {remove_nodes_below}
        for root in roots:
            order += [
                u
                for u in tree.nodes(root=root, order="minlex_postorder")
                if u not in remove_nodes
            ]
        svg = tree.draw_svg(order=order, debug_box=draw_plotbox)
        assert svg.count("multi") == 1  # One tip representing multiple nodes
        for u in order:
            assert f'n{u}"' in svg or f"n{u} " in svg
        for u in remove_nodes:
            assert f'n{u}"' not in svg and f"n{u} " not in svg
        self.verify_known_svg(
            svg, "tree_subtrees_with_collapsed.svg", overwrite_viz, has_root=False
        )

    def test_known_svg_tree_polytomy(self, overwrite_viz, draw_plotbox):
        tracked_nodes = [20, 24, 25, 27, 28, 29]
        tree = tskit.Tree.generate_balanced(30, arity=4)
        svg = tree.draw_svg(
            time_scale="rank",
            debug_box=draw_plotbox,
            size=(600, 200),
            style="".join(f".n{u} > .sym {{fill: cyan}}" for u in tracked_nodes + [39]),
        )
        self.verify_known_svg(
            svg, "tree_poly.svg", overwrite_viz, width=600, height=200
        )

    def test_known_svg_tree_polytomy_tracked(self, overwrite_viz, draw_plotbox):
        tracked_nodes = [20, 24, 25, 27, 28, 29]
        tree = tskit.Tree.generate_balanced(30, arity=4, tracked_samples=tracked_nodes)
        svg = tree.draw_svg(
            time_scale="rank",
            order=drawing._postorder_tracked_minlex_traversal(tree),
            debug_box=draw_plotbox,
            pack_untracked_polytomies=True,
            size=(600, 200),
            style="".join(f".n{u} > .sym {{fill: cyan}}" for u in tracked_nodes + [39]),
        )
        self.verify_known_svg(
            svg, "tree_poly_tracked.svg", overwrite_viz, width=600, height=200
        )

    def test_known_svg_tree_polytomy_tracked_collapse(
        self, overwrite_viz, draw_plotbox
    ):
        tracked_nodes = [20, 24, 25, 27, 28, 29]
        tree = tskit.Tree.generate_balanced(30, arity=4, tracked_samples=tracked_nodes)
        svg = tree.draw_svg(
            time_scale="rank",
            order=drawing._postorder_tracked_minlex_traversal(
                tree, collapse_tracked=True
            ),
            debug_box=draw_plotbox,
            size=(600, 200),
            pack_untracked_polytomies=True,
            style="".join(f".n{u} > .sym {{fill: cyan}}" for u in tracked_nodes + [39]),
        )
        self.verify_known_svg(
            svg, "tree_poly_tracked_collapse.svg", overwrite_viz, width=600, height=200
        )


class TestRounding:
    def test_rnd(self):
        assert 0 == drawing.rnd(0)
        assert math.inf == drawing.rnd(math.inf)
        assert 1 == drawing.rnd(1)
        assert 1.1 == drawing.rnd(1.1)
        assert 1.11111 == drawing.rnd(1.111111)
        assert 1111110 == drawing.rnd(1111111)
        assert 123.457 == drawing.rnd(123.4567)
        assert 123.456 == drawing.rnd(123.4564)


class TestDrawingTraversals:
    # TODO: test drawing._postorder_tracked_minlex_traversal and
    # drawing._postorder_tracked_node_traversal
    pass
