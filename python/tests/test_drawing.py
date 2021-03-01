# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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
import re
import xml.dom.minidom
import xml.etree

import msprime
import numpy as np
import pytest
import xmlunittest

import tests.tsutil as tsutil
import tskit
from tskit import drawing


class TestTreeDraw:
    """
    Tests for the tree drawing functionality.
    TODO - the get_XXX_tree() functions should probably be placed in fixtures
    """

    def get_binary_tree(self):
        ts = msprime.simulate(10, random_seed=1, mutation_rate=1)
        return next(ts.trees())

    def get_nonbinary_ts(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=0.1, population=0, proportion=0.5)
        ]
        return msprime.simulate(
            10,
            recombination_rate=5,
            mutation_rate=10,
            demographic_events=demographic_events,
            random_seed=1,
        )

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
        ts = tsutil.decapitate(ts, 20)
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

    def get_simple_ts(self):
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
        4       0       0       -1      0.02445014598813
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
        """
        )
        mutations = io.StringIO(
            """\
        site   node    derived_state    parent
        0      9       T                -1
        0      9       G                0
        0      5       1                1
        1      4       C                -1
        1      4       G                3
        2      7       G                -1
        """
        )
        return tskit.load_text(
            nodes, edges, sites=sites, mutations=mutations, strict=False
        )

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

    left_neighbour = np.full(tree.num_nodes, tskit.NULL, dtype=int)
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
        left_child = drawing.get_left_child(t, "postorder")
        for u in t.nodes(order="postorder"):
            if t.num_children(u) > 0:
                assert left_child[u] == t.children(u)[0]

    def test_null_node_left_child(self):
        t = self.get_nonbinary_tree()
        left_child = drawing.get_left_child(t, "minlex_postorder")
        assert left_child[tskit.NULL] == tskit.NULL

    def test_leaf_node_left_child(self):
        t = self.get_nonbinary_tree()
        left_child = drawing.get_left_child(t, "minlex_postorder")
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
            t.draw(format=self.drawing_format, max_tree_height=1234)
        with pytest.raises(ValueError):
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
            "  0.00      1.00\n"
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
            "  0.00          1.00\n"
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
            "     13             \n"
            "┏━┳━┳━╋━┳━━┓        \n"
            "┃ ┃ ┃ ┃ ┃ 12        \n"
            "┃ ┃ ┃ ┃ ┃ ┏┻┓       \n"
            "┃ ┃ ┃ ┃ ┃ ┃ ┃  11   \n"
            "┃ ┃ ┃ ┃ ┃ ┃ ┃ ┏━┻┓  \n"
            "┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ 10  \n"
            "┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┏┻┓ \n"
            "2 4 5 6 9 1 7 8 0 3 \n"
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
        print(ts.draw_text())
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
            "0.02┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊\n"
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
            "0.02|  4  | | |  4  | | |  4  | | |  4  | | |  4  | | |\n"
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

    def test_max_tree_height(self):
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
        self.verify_text_rendering(t.draw_text(max_tree_height="ts"), tree)

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
        self.verify_text_rendering(t.draw_text(max_tree_height="tree"), tree)
        for bad_max_tree_height in [1, "sdfr", ""]:
            with pytest.raises(ValueError):
                t.draw_text(max_tree_height=bad_max_tree_height)


class TestDrawSvg(TestTreeDraw, xmlunittest.XmlTestMixin):
    """
    Tests the SVG tree drawing.
    """

    def verify_basic_svg(self, svg, width=200, height=200, num_trees=1):
        prefix = "{http://www.w3.org/2000/svg}"
        root = xml.etree.ElementTree.fromstring(svg)
        assert root.tag == prefix + "svg"
        assert width * num_trees == int(root.attrib["width"])
        assert height == int(root.attrib["height"])

        # Verify the class structure of the svg
        root_group = root.find(prefix + "g")
        assert "class" in root_group.attrib
        assert re.search(r"\b(tree|tree-sequence)\b", root_group.attrib["class"])
        if "tree-sequence" in root_group.attrib["class"]:
            trees = None
            for g in root_group.findall(prefix + "g"):
                if "trees" in g.attrib.get("class", ""):
                    trees = g
                    break
            assert trees is not None  # Must have found a trees group
            first_treebox = trees.find(prefix + "g")
            assert "class" in first_treebox.attrib
            assert re.search(r"\btreebox\b", first_treebox.attrib["class"])
            first_tree = first_treebox.find(prefix + "g")
            assert "class" in first_tree.attrib
            assert re.search(r"\btree\b", first_tree.attrib["class"])
        else:
            first_tree = root_group
        # Check that we have edges, symbols, and labels groups
        groups = first_tree.findall(prefix + "g")
        assert len(groups) > 0
        for group in groups:
            assert "class" in group.attrib
            cls = group.attrib["class"]
            assert re.search(r"\broot\b", cls)

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

    def test_draw_defaults(self):
        t = self.get_binary_tree()
        svg = t.draw()
        self.verify_basic_svg(svg)
        svg = t.draw_svg()
        self.verify_basic_svg(svg)

    def test_draw_nonbinary(self):
        t = self.get_nonbinary_tree()
        svg = t.draw()
        self.verify_basic_svg(svg)
        svg = t.draw_svg()
        self.verify_basic_svg(svg)

    def test_draw_multiroot(self):
        t = self.get_multiroot_tree()
        svg = t.draw()
        self.verify_basic_svg(svg)
        svg = t.draw_svg()
        self.verify_basic_svg(svg)

    def test_draw_mutations_over_roots(self):
        t = self.get_mutations_over_roots_tree()
        svg = t.draw()
        self.verify_basic_svg(svg)
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
        assert svg.count("XXX") == t.num_nodes
        svg = t.draw_svg(node_label_attrs={u: {"text": labels[u]} for u in t.nodes()})
        self.verify_basic_svg(svg)
        assert svg.count("XXX") == t.num_nodes

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
        assert svg.count("opacity:0") == 1

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

    def test_bad_tree_height_scale(self):
        t = self.get_binary_tree()
        for bad_scale in ["te", "asdf", "", [], b"23"]:
            with pytest.raises(ValueError):
                t.draw_svg(tree_height_scale=bad_scale)

    def test_bad_max_tree_height(self):
        t = self.get_binary_tree()
        for bad_height in ["te", "asdf", "", [], b"23"]:
            with pytest.raises(ValueError):
                t.draw_svg(max_tree_height=bad_height)

    def test_height_scale_time_and_max_tree_height(self):
        ts = msprime.simulate(5, recombination_rate=2, random_seed=2)
        t = ts.first()
        # The default should be the same as tree.
        svg1 = t.draw_svg(max_tree_height="tree")
        self.verify_basic_svg(svg1)
        svg2 = t.draw_svg()
        assert svg1 == svg2
        svg3 = t.draw_svg(max_tree_height="ts")
        assert svg1 != svg3
        svg4 = t.draw_svg(max_tree_height=max(ts.tables.nodes.time))
        assert svg3 == svg4

    def test_height_scale_rank_and_max_tree_height(self):
        # Make sure the rank height scale and max_tree_height interact properly.
        ts = msprime.simulate(5, recombination_rate=2, random_seed=2)
        t = ts.first()
        # The default should be the same as tree.
        svg1 = t.draw_svg(max_tree_height="tree", tree_height_scale="rank")
        self.verify_basic_svg(svg1)
        svg2 = t.draw_svg(tree_height_scale="rank")
        assert svg1 == svg2
        svg3 = t.draw_svg(max_tree_height="ts", tree_height_scale="rank")
        assert svg1 != svg3
        self.verify_basic_svg(svg3)
        # Numeric max tree height not supported for rank scale.
        with pytest.raises(ValueError):
            t.draw_svg(max_tree_height=2, tree_height_scale="rank")

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
        assert svg.count("opacity:0") == 1

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
        assert svg.count("fill-opacity:0") == 1

    def test_max_tree_height(self):
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

        svg1 = ts.at_index(0).draw(max_tree_height="ts")
        svg2 = ts.at_index(1).draw(max_tree_height="ts")
        # when scaled, node 3 should be at the *same* height in both trees, so the edge
        # definition should be the same
        self.verify_basic_svg(svg1)
        self.verify_basic_svg(svg2)
        str_pos = svg1.find(">0<")
        snippet1 = svg1[svg1.rfind("edge", 0, str_pos) : str_pos]
        str_pos = svg2.find(">0<")
        snippet2 = svg2[svg2.rfind("edge", 0, str_pos) : str_pos]
        assert snippet1 == snippet2

    def test_draw_sized_tree(self):
        tree = self.get_binary_tree()
        svg = tree.draw_svg(size=(600, 400))
        self.verify_basic_svg(svg, width=600, height=400)

    def test_draw_simple_ts(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=1)
        svg = ts.draw_svg()
        self.verify_basic_svg(svg, width=200 * ts.num_trees)

    def test_draw_integer_breaks_ts(self):
        # TODO update this to use the msprime 1.0 API. Then we'll need to
        # change to make the floating point breaks the exception.
        recomb_map = msprime.RecombinationMap.uniform_map(
            length=1000, rate=0.005, num_loci=1000
        )
        ts = msprime.simulate(5, recombination_map=recomb_map, random_seed=1)
        assert ts.num_trees > 2
        svg = ts.draw_svg()
        self.verify_basic_svg(svg, width=200 * ts.num_trees)
        axis_pos = svg.find('class="axis"')
        for b in ts.breakpoints():
            assert b == round(b)
            assert svg.find(f">{b:.0f}<", axis_pos) != -1

    def test_draw_even_height_ts(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=1)
        svg = ts.draw_svg(max_tree_height="tree")
        self.verify_basic_svg(svg, width=200 * ts.num_trees)

    def test_draw_sized_ts(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=1)
        svg = ts.draw_svg(size=(600, 400))
        self.verify_basic_svg(svg, width=600, height=400)

    def test_tree_height_scale(self):
        ts = msprime.simulate(4, random_seed=2)
        svg = ts.draw_svg(tree_height_scale="time")
        self.verify_basic_svg(svg)
        svg = ts.draw_svg(tree_height_scale="log_time")
        self.verify_basic_svg(svg)
        svg = ts.draw_svg(tree_height_scale="rank")
        self.verify_basic_svg(svg)
        for bad_scale in [0, "", "NOT A SCALE"]:
            with pytest.raises(ValueError):
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

    def test_no_edges(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.dump_tables()
        tables.edges.clear()
        ts_no_edges = tables.tree_sequence()
        svg = ts_no_edges.draw_svg()  # This should just be a row of 10 sample nodes
        self.verify_basic_svg(svg)
        assert svg.count("rect") == 10  # Sample nodes are rectangles
        assert svg.count('path class="edge"') == 0

        svg = ts_no_edges.draw_svg(force_root_branch=True)
        self.verify_basic_svg(svg)
        assert svg.count("rect") == 10
        assert svg.count('path class="edge"') == 10

        # If there is a mutation, the root branches should be there too
        ts = msprime.mutate(ts, rate=1, random_seed=1)
        tables = ts.dump_tables()
        tables.edges.clear()
        ts_no_edges = tables.tree_sequence().simplify()
        assert ts_no_edges.num_mutations > 0  # Should have some singletons
        svg = ts_no_edges.draw_svg()
        self.verify_basic_svg(svg)
        assert svg.count("rect") == 10
        assert svg.count('path class="edge"') == 10
        assert svg.count('path class="sym"') == (
            ts_no_edges.num_mutations + ts_no_edges.num_sites
        )

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
        edge_str = '<path class="edge" d='
        str_pos1 = svg1.rfind(edge_str, 0, svg1.find(f">{root1}<"))
        str_pos2a = svg2a.rfind(edge_str, 0, svg2a.find(f">{root2}<"))
        str_pos2b = svg2b.rfind(edge_str, 0, svg2b.find(f">{root2}<"))
        snippet1 = svg1[str_pos1 + len(edge_str) : svg1.find(">", str_pos1)]
        snippet2a = svg2a[str_pos2a + len(edge_str) : svg2a.find(">", str_pos2a)]
        snippet2b = svg2b[str_pos2b + len(edge_str) : svg2b.find(">", str_pos2b)]
        assert snippet1.startswith('"M 0 0')
        assert snippet2a.startswith('"M 0 0')
        assert snippet2b.startswith('"M 0 0')
        assert "H 0" in snippet1
        assert not ("H 0" in snippet2a)  # No root branch
        assert "H 0" in snippet2b

    def verify_known_svg(self, svg, filename, save=False, **kwargs):
        # expected SVG files can be inspected in tests/data/svg/*.svg
        svg = xml.dom.minidom.parseString(
            svg
        ).toprettyxml()  # Prettify for easy viewing
        self.verify_basic_svg(svg, **kwargs)
        svg_fn = pathlib.Path(__file__).parent / "data" / "svg" / filename
        if save:
            logging.warning(f"Overwriting SVG file `{svg_fn}` with new version")
            with open(svg_fn, "wt") as file:
                file.write(svg)
        with open(svg_fn, "rb") as file:
            expected_svg = file.read()
        self.assertXmlEquivalentOutputs(svg, expected_svg)

    def test_known_svg_tree_no_mut(self, overwrite_viz):
        tree = self.get_simple_ts().at_index(-1)
        svg = tree.draw_svg(
            root_svg_attributes={"id": "XYZ"}, style=".edge {stroke: blue}"
        )
        self.verify_known_svg(svg, "tree.svg", overwrite_viz)

    def test_known_svg_tree_root_mut(self, overwrite_viz):
        tree = self.get_simple_ts().at_index(0)  # Tree 0 has a few mutations above root
        svg = tree.draw_svg(
            root_svg_attributes={"id": "XYZ"}, style=".edge {stroke: blue}"
        )
        self.verify_known_svg(svg, "mut_tree.svg", overwrite_viz)

    def test_known_svg_ts(self, overwrite_viz):
        ts = self.get_simple_ts()
        svg = ts.draw_svg()
        assert svg.count('class="site ') == ts.num_sites
        assert svg.count('class="mut ') == ts.num_mutations * 2
        self.verify_known_svg(svg, "ts.svg", overwrite_viz, width=200 * ts.num_trees)

    def test_known_svg_ts_highlighted_mut(self, overwrite_viz):
        ts = self.get_simple_ts()
        style = (
            ".edge {stroke: grey}"
            ".mut .sym{stroke:pink} .mut text{fill:pink}"
            ".mut.m2 .sym,.m2>line, .m2>.node .edge{stroke:red} .mut.m2 text{fill:red}"
            ".mut.m3 .sym,.m3>line, .m3>.node .edge{stroke:cyan} .mut.m3 text{fill:cyan}"
            ".mut.m4 .sym,.m4>line, .m4>.node .edge{stroke:blue} .mut.m4 text{fill:blue}"
        )
        svg = ts.draw_svg(style=style)
        self.verify_known_svg(
            svg, "ts_mut_highlight.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_nonbinary_ts(self, overwrite_viz):
        ts = self.get_nonbinary_ts()
        svg = ts.draw_svg(tree_height_scale="log_time")
        assert svg.count('class="site ') == ts.num_sites
        assert svg.count('class="mut ') == ts.num_mutations * 2
        self.verify_known_svg(
            svg, "ts_nonbinary.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_plain(self, overwrite_viz):
        """
        Plain style: no background shading and a variable scale X axis with no sites
        """
        ts = self.get_simple_ts()
        svg = ts.draw_svg(x_scale="treewise")
        assert svg.count('class="site ') == 0
        assert svg.count('class="mut ') == ts.num_mutations
        self.verify_known_svg(
            svg, "ts_plain.svg", overwrite_viz, width=200 * ts.num_trees
        )

    def test_known_svg_ts_with_xlabel(self, overwrite_viz):
        """
        Style with X axis label
        """
        ts = self.get_simple_ts()
        x_label = "genomic position (bp)"
        svg = ts.draw_svg(x_label=x_label)
        assert x_label in svg
        self.verify_known_svg(
            svg, "ts_xlabel.svg", overwrite_viz, width=200 * ts.num_trees
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
