# -*- coding: utf-8 -*-
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
import io
import os
import tempfile
import unittest
import xml.etree

import msprime
import tskit
import tests.tsutil as tsutil


class TestTreeDraw(unittest.TestCase):
    """
    Tests for the tree drawing functionality.
    """
    def get_binary_tree(self):
        ts = msprime.simulate(10, random_seed=1, mutation_rate=1)
        return next(ts.trees())

    def get_nonbinary_tree(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=0.1, population=0, proportion=0.5)]
        ts = msprime.simulate(
            10, recombination_rate=5, mutation_rate=10,
            demographic_events=demographic_events, random_seed=1)
        for t in ts.trees():
            for u in t.nodes():
                if len(t.children(u)) > 2:
                    return t
        assert False

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
        self.assertEqual(tree.num_roots, 0)
        return tree

    def get_multiroot_tree(self):
        ts = msprime.simulate(15, random_seed=1)
        # Take off the top quarter of edges
        tables = ts.dump_tables()
        edges = tables.edges
        n = len(edges) - len(edges) // 4
        edges.set_columns(
            left=edges.left[:n], right=edges.right[:n],
            parent=edges.parent[:n], child=edges.child[:n])
        ts = tables.tree_sequence()
        for t in ts.trees():
            if t.num_roots > 1:
                return t
        assert False

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
        assert any(
            tree.parent(mut.node) == tskit.NULL
            for mut in tree.mutations())
        return tree

    def get_unary_node_tree(self):
        ts = msprime.simulate(2, random_seed=1)
        tables = ts.dump_tables()
        edges = tables.edges
        # Take out all the edges except 1
        n = 1
        edges.set_columns(
            left=edges.left[:n], right=edges.right[:n],
            parent=edges.parent[:n], child=edges.child[:n])
        ts = tables.tree_sequence()
        for t in ts.trees():
            for u in t.nodes():
                if len(t.children(u)) == 1:
                    return t
        assert False

    def get_empty_tree(self):
        tables = tskit.TableCollection(sequence_length=1)
        ts = tables.tree_sequence()
        return next(ts.trees())


class TestFormats(TestTreeDraw):
    """
    Tests that formats are recognised correctly.
    """
    def test_svg_variants(self):
        t = self.get_binary_tree()
        for svg in ["svg", "SVG", "sVg"]:
            output = t.draw(format=svg)
            root = xml.etree.ElementTree.fromstring(output)
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")

    def test_default(self):
        # Default is SVG
        t = self.get_binary_tree()
        output = t.draw(format=None)
        root = xml.etree.ElementTree.fromstring(output)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        output = t.draw()
        root = xml.etree.ElementTree.fromstring(output)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")

    def test_ascii_variants(self):
        t = self.get_binary_tree()
        for fmt in ["ascii", "ASCII", "AScii"]:
            output = t.draw(format=fmt)
            self.assertRaises(
                xml.etree.ElementTree.ParseError, xml.etree.ElementTree.fromstring,
                output)

    def test_unicode_variants(self):
        t = self.get_binary_tree()
        for fmt in ["unicode", "UNICODE", "uniCODE"]:
            output = t.draw(format=fmt)
            self.assertRaises(
                xml.etree.ElementTree.ParseError, xml.etree.ElementTree.fromstring,
                output)

    def test_bad_formats(self):
        t = self.get_binary_tree()
        for bad_format in ["", "ASC", "SV", "jpeg"]:
            self.assertRaises(ValueError, t.draw, format=bad_format)


# TODO we should gather some of these tests into a superclass as they are
# very similar for SVG and ASCII.

class TestDrawText(TestTreeDraw):
    """
    Tests the ASCII tree drawing method.
    """
    drawing_format = "ascii"
    example_label = "XXX"

    def verify_basic_text(self, text):
        self.assertTrue(isinstance(text, str))
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
        self.assertRaises(ValueError, t.draw, format=self.drawing_format)

    def test_draw_zero_roots_tree(self):
        t = self.get_zero_roots_tree()
        self.assertRaises(ValueError, t.draw, format=self.drawing_format)

    def test_draw_zero_edge_tree(self):
        t = self.get_zero_edge_tree()
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_even_num_children_tree(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           1
        2   1           2
        3   1           1
        4   1           4
        5   1           5
        6   1           7
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       6       0
        0       1       6       1
        0       1       6       2
        0       1       6       3
        0       1       6       4
        0       1       6       5
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        text = t.draw(format=self.drawing_format)
        self.verify_basic_text(text)

    def test_odd_num_children_tree(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           1
        2   1           2
        3   1           1
        4   1           4
        5   1           5
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       5       0
        0       1       5       1
        0       1       5       2
        0       1       5       3
        0       1       5       4
        """)
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
            self.assertNotEqual(j, -1)

    def test_no_node_labels(self):
        t = self.get_binary_tree()
        labels = {}
        text = t.draw(format=self.drawing_format, node_labels=labels)
        self.verify_basic_text(text)
        for u in t.nodes():
            self.assertEqual(text.find(str(u)), -1)


class TestDrawUnicode(TestDrawText):
    """
    Tests the Unicode tree drawing method
    """
    drawing_format = "unicode"
    example_label = "\u20ac" * 10  # euro symbol

    def verify_text_rendering(self, drawn, drawn_tree, debug=False):
        if debug:
            print("Drawn:")
            print(drawn)
            print("Expected:")
            print(drawn_tree)
        tree_lines = drawn_tree.splitlines()
        drawn_lines = drawn.splitlines()
        self.assertEqual(len(tree_lines), len(drawn_lines))
        for l1, l2 in zip(tree_lines, drawn_lines):
            # Trailing white space isn't significant.
            self.assertEqual(l1.rstrip(), l2.rstrip())

    def test_simple_tree(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       2       0
        0       1       2       1
        """)
        tree = (
            " 2 \n"
            "┏┻┓\n"
            "0 1")
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode")
        self.verify_text_rendering(drawn, tree)
        drawn = t.draw_text()
        self.verify_text_rendering(drawn, tree)

    def test_four_leaves(self):
        nodes = io.StringIO("""\
        id      is_sample   population      individual      time    metadata
        0       1       0       -1      0.00000000000000
        1       1       0       -1      0.00000000000000
        2       1       0       -1      0.00000000000000
        3       1       0       -1      0.00000000000000
        4       0       0       -1      0.26676079696421
        5       0       0       -1      1.48826948286480
        6       0       0       -1      2.91835007758007
        """)
        edges = io.StringIO("""\
        left            right           parent  child
        0.00000000      1.00000000      4       0
        0.00000000      1.00000000      4       3
        0.00000000      1.00000000      5       2
        0.00000000      1.00000000      5       4
        0.00000000      1.00000000      6       1
        0.00000000      1.00000000      6       5
        """)
        tree = (
            "  6     \n"
            "┏━┻━┓   \n"
            "┃   5   \n"
            "┃ ┏━┻┓  \n"
            "┃ ┃  4  \n"
            "┃ ┃ ┏┻┓ \n"
            "1 2 0 3 \n")
        ts = tskit.load_text(nodes, edges, strict=False)
        t = ts.first()
        drawn = t.draw(format="unicode")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

    def test_trident_tree(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           0
        3   1           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       3       0
        0       1       3       1
        0       1       3       2
        """)
        tree = (
            "  3  \n"
            "┏━╋━┓\n"
            "0 1 2\n")
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

    def test_pitchfork_tree(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           0
        3   1           0
        4   1           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       4       0
        0       1       4       1
        0       1       4       2
        0       1       4       3
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        tree = (
            "   4   \n"
            "┏━┳┻┳━┓\n"
            "0 1 2 3\n")
        drawn = t.draw(format="unicode")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

        # No labels
        tree = (
            "   ┃   \n"
            "┏━┳┻┳━┓\n"
            "┃ ┃ ┃ ┃\n")
        drawn = t.draw(format="unicode", node_labels={})
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(node_labels={}), tree)
        # Some lables
        tree = (
            "   ┃   \n"
            "┏━┳┻┳━┓\n"
            "0 ┃ ┃ 3\n")
        labels = {0: "0", 3: "3"}
        drawn = t.draw(format="unicode", node_labels=labels)
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(node_labels=labels), tree)

    def test_stick_tree(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           1
        2   1           2
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       1       0
        0       1       2       1
        """)
        tree = (
            "2\n"
            "┃\n"
            "1\n"
            "┃\n"
            "0\n")
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

    def test_draw_forky_tree(self):
        tree = (
           "       14           \n"
           "  ┏━━━━┻━━━━┓       \n"
           "  ┃         13      \n"
           "  ┃   ┏━┳━┳━╋━┳━━┓  \n"
           "  ┃   ┃ ┃ ┃ ┃ ┃  12 \n"
           "  ┃   ┃ ┃ ┃ ┃ ┃ ┏┻┓ \n"
           "  11  ┃ ┃ ┃ ┃ ┃ ┃ ┃ \n"
           "┏━┻┓  ┃ ┃ ┃ ┃ ┃ ┃ ┃ \n"
           "┃  10 ┃ ┃ ┃ ┃ ┃ ┃ ┃ \n"
           "┃ ┏┻┓ ┃ ┃ ┃ ┃ ┃ ┃ ┃ \n"
           "8 0 3 2 4 5 6 9 1 7 \n")

        nodes = io.StringIO("""\
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
        """)
        edges = io.StringIO("""\
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
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

    def test_draw_multiroot_forky_tree(self):
        tree = (
           "      13             \n"
           "┏━┳━┳━╋━┳━━┓         \n"
           "┃ ┃ ┃ ┃ ┃  12        \n"
           "┃ ┃ ┃ ┃ ┃ ┏┻┓        \n"
           "┃ ┃ ┃ ┃ ┃ ┃ ┃   11   \n"
           "┃ ┃ ┃ ┃ ┃ ┃ ┃  ┏┻━┓  \n"
           "┃ ┃ ┃ ┃ ┃ ┃ ┃  ┃  10 \n"
           "┃ ┃ ┃ ┃ ┃ ┃ ┃  ┃ ┏┻┓ \n"
           "2 4 5 6 9 1 7  8 0 3 \n")
        nodes = io.StringIO("""\
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
        """)
        edges = io.StringIO("""\
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
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        t = next(ts.trees())
        drawn = t.draw(format="unicode")
        self.verify_text_rendering(drawn, tree)
        self.verify_text_rendering(t.draw_text(), tree)

    def test_simple_tree_sequence(self):
        ts_drawing = (
           "   9    ┊         ┊         ┊         ┊        \n"
           " ┏━┻━┓  ┊         ┊         ┊         ┊        \n"
           " ┃   ┃  ┊         ┊         ┊         ┊    8   \n"
           " ┃   ┃  ┊         ┊         ┊         ┊  ┏━┻━┓ \n"
           " ┃   ┃  ┊    7    ┊         ┊    7    ┊  ┃   ┃ \n"
           " ┃   ┃  ┊  ┏━┻━┓  ┊         ┊  ┏━┻━┓  ┊  ┃   ┃ \n"
           " ┃   ┃  ┊  ┃   ┃  ┊    6    ┊  ┃   ┃  ┊  ┃   ┃ \n"
           " ┃   ┃  ┊  ┃   ┃  ┊  ┏━┻━┓  ┊  ┃   ┃  ┊  ┃   ┃ \n"
           " ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5 \n"
           " ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓\n"
           " 4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃\n"
           "┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃\n"
           "0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3\n")
        nodes = io.StringIO("""\
            id      is_sample   population      individual      time    metadata
            0       1       0       -1      0.00000000000000
            1       1       0       -1      0.00000000000000
            2       1       0       -1      0.00000000000000
            3       1       0       -1      0.00000000000000
            4       0       0       -1      0.02445014598813
            5       0       0       -1      0.11067965364865
            6       0       0       -1      1.75005250750382
            7       0       0       -1      2.31067154311640
            8       0       0       -1      3.57331354884652
            9       0       0       -1      9.08308317451295
        """)
        edges = io.StringIO("""\
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
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
        self.verify_text_rendering(ts.draw_text(), ts_drawing)

    def test_tree_height_scale(self):
        tree = msprime.simulate(4, random_seed=2).first()
        with self.assertRaises(ValueError):
            tree.draw_text(tree_height_scale="time")

        t1 = tree.draw_text(tree_height_scale="rank")
        t2 = tree.draw_text()
        self.assertEqual(t1, t2)

        for bad_scale in [0, "", "NOT A SCALE"]:
            with self.assertRaises(ValueError):
                tree.draw_text(tree_height_scale=bad_scale)

    def test_max_tree_height(self):
        nodes = io.StringIO("""\
            id      is_sample   population      individual      time    metadata
            0       1       0       -1      0.00000000000000
            1       1       0       -1      0.00000000000000
            2       1       0       -1      0.00000000000000
            3       1       0       -1      0.00000000000000
            4       0       0       -1      0.02445014598813
            5       0       0       -1      0.11067965364865
            6       0       0       -1      1.75005250750382
            7       0       0       -1      2.31067154311640
            8       0       0       -1      3.57331354884652
            9       0       0       -1      9.08308317451295
        """)
        edges = io.StringIO("""\
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
        """)
        ts = tskit.load_text(nodes, edges, strict=False)
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
           "0 1 2 3\n")
        t = ts.first()
        self.verify_text_rendering(t.draw_text(max_tree_height="ts"), tree)


class TestDrawSvg(TestTreeDraw):
    """
    Tests the SVG tree drawing.
    """
    def verify_basic_svg(self, svg, width=200, height=200):
        root = xml.etree.ElementTree.fromstring(svg)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        self.assertEqual(width, int(root.attrib["width"]))
        self.assertEqual(height, int(root.attrib["height"]))

    def test_draw_file(self):
        t = self.get_binary_tree()
        fd, filename = tempfile.mkstemp(prefix="tskit_viz_")
        try:
            os.close(fd)
            svg = t.draw(path=filename)
            self.assertGreater(os.path.getsize(filename), 0)
            with open(filename) as tmp:
                other_svg = tmp.read()
            self.assertEqual(svg, other_svg)
            os.unlink(filename)

            svg = t.draw_svg(path=filename)
            self.assertGreater(os.path.getsize(filename), 0)
            with open(filename) as tmp:
                other_svg = tmp.read()
            self.verify_basic_svg(svg)
            self.verify_basic_svg(other_svg)

            ts = t.tree_sequence
            svg = ts.draw_svg(path=filename)
            self.assertGreater(os.path.getsize(filename), 0)
            with open(filename) as tmp:
                other_svg = tmp.read()
            self.verify_basic_svg(svg)
            self.verify_basic_svg(other_svg)
        finally:
            os.unlink(filename)

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
        self.assertRaises(ValueError, t.draw)
        self.assertRaises(ValueError, t.draw_svg)

    def test_draw_zero_roots(self):
        t = self.get_zero_roots_tree()
        self.assertRaises(ValueError, t.draw)
        self.assertRaises(ValueError, t.draw_svg)

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
        self.assertEqual(svg.count("XXX"), t.num_nodes)
        svg = t.draw_svg(node_label_attrs={u: {"text": labels[u]} for u in t.nodes()})
        self.verify_basic_svg(svg)
        self.assertEqual(svg.count("XXX"), t.num_nodes)

    def test_one_node_label(self):
        t = self.get_binary_tree()
        labels = {0: "XXX"}
        svg = t.draw(format="svg", node_labels=labels)
        self.verify_basic_svg(svg)
        self.assertEqual(svg.count("XXX"), 1)
        svg = t.draw_svg(node_label_attrs={0: {"text": "XXX"}})
        self.verify_basic_svg(svg)
        self.assertEqual(svg.count("XXX"), 1)

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
        self.assertEqual(svg.count('fill="{}"'.format(colour)), 1)
        svg = t.draw_svg(node_attrs={0: {'fill': colour}})
        self.verify_basic_svg(svg)
        self.assertEqual(svg.count('fill="{}"'.format(colour)), 1)

    def test_all_nodes_colour(self):
        t = self.get_binary_tree()
        colours = {u: "rgb({}, {}, {})".format(u, u, u) for u in t.nodes()}
        svg = t.draw(format="svg", node_colours=colours)
        self.verify_basic_svg(svg)
        for colour in colours.values():
            self.assertEqual(svg.count('fill="{}"'.format(colour)), 1)

        svg = t.draw_svg(node_attrs={u: {'fill': colours[u]} for u in t.nodes()})
        self.verify_basic_svg(svg)
        self.assertEqual(svg.count('fill="{}"'.format(colour)), 1)
        for colour in colours.values():
            self.assertEqual(svg.count('fill="{}"'.format(colour)), 1)

    def test_unplotted_node(self):
        t = self.get_binary_tree()
        colour = None
        colours = {0: colour}
        svg = t.draw(format="svg", node_colours=colours)
        self.verify_basic_svg(svg)
        nodes_in_tree = list(t.nodes())
        self.assertEqual(svg.count('<circle'.format(colour)), len(nodes_in_tree)-1)

    def test_one_edge_colour(self):
        t = self.get_binary_tree()
        colour = "rgb(0, 1, 2)"
        colours = {0: colour}
        svg = t.draw(format="svg", edge_colours=colours)
        self.verify_basic_svg(svg)
        self.assertEqual(svg.count('stroke="{}"'.format(colour)), 2)
        svg = t.draw_svg(edge_attrs={0: {"stroke": colour}})
        self.verify_basic_svg(svg)
        # We're mapping to a path here, so only see it once. The old code
        # drew two lines.
        self.assertEqual(svg.count('stroke="{}"'.format(colour)), 1)

    #
    # TODO: update the tests below here to check the new SVG based interface.
    #
    def test_all_edges_colour(self):
        t = self.get_binary_tree()
        colours = {u: "rgb({u}, {u}, {u})".format(u=u) for u in t.nodes() if u != t.root}
        svg = t.draw(format="svg", edge_colours=colours)
        self.verify_basic_svg(svg)
        for colour in colours.values():
            self.assertEqual(svg.count('stroke="{}"'.format(colour)), 2)

    def test_unplotted_edge(self):
        t = self.get_binary_tree()
        colour = None
        colours = {0: colour}
        svg = t.draw(format="svg", edge_colours=colours)
        self.verify_basic_svg(svg)
        nodes_in_tree = set(t.nodes())
        non_root_nodes = nodes_in_tree - set([t.root])
        self.assertEqual(svg.count('<line'), (len(non_root_nodes) - 1) * 2)

    def test_mutation_labels(self):
        t = self.get_binary_tree()
        labels = {u.id: "XXX" for u in t.mutations()}
        svg = t.draw(format="svg", mutation_labels=labels)
        self.verify_basic_svg(svg)
        self.assertEqual(svg.count("XXX"), t.num_mutations)

    def test_one_mutation_label(self):
        t = self.get_binary_tree()
        labels = {0: "XXX"}
        svg = t.draw(format="svg", mutation_labels=labels)
        self.verify_basic_svg(svg)
        self.assertEqual(svg.count("XXX"), 1)

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
        self.assertEqual(svg.count('fill="{}"'.format(colour)), 1)

    def test_all_mutations_colour(self):
        t = self.get_binary_tree()
        colours = {
            mut.id: "rgb({}, {}, {})".format(mut.id, mut.id, mut.id)
            for mut in t.mutations()}
        svg = t.draw(format="svg", mutation_colours=colours)
        self.verify_basic_svg(svg)
        for colour in colours.values():
            self.assertEqual(svg.count('fill="{}"'.format(colour)), 1)

    def test_unplotted_mutation(self):
        t = self.get_binary_tree()
        colour = None
        colours = {0: colour}
        svg = t.draw(format="svg", mutation_colours=colours)
        self.verify_basic_svg(svg)
        mutations_in_tree = list(t.mutations())
        self.assertEqual(svg.count('<rect'), len(mutations_in_tree) - 1)

    def test_max_tree_height(self):
        nodes = io.StringIO("""\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           0
        3   0           1
        4   0           2
        5   0           3
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0       1       5       2
        0       1       5       3
        1       2       4       2
        1       2       4       3
        0       2       3       0
        0       2       3       1
        """)
        ts = tskit.load_text(nodes, edges, strict=False)

        svg1 = ts.at_index(0).draw()
        svg2 = ts.at_index(1).draw()
        # if not scaled to ts, node 3 is at a different height in both trees, because the
        # root is at a different height. We expect a label looking something like
        # <text x="10.0" y="XXXX">3</text> where XXXX is different
        str_pos = svg1.find('>3<')
        snippet1 = svg1[svg1.rfind("<", 0, str_pos):str_pos]
        str_pos = svg2.find('>3<')
        snippet2 = svg2[svg2.rfind("<", 0, str_pos):str_pos]
        self.assertNotEqual(snippet1, snippet2)

        svg1 = ts.at_index(0).draw(max_tree_height="ts")
        svg2 = ts.at_index(1).draw(max_tree_height="ts")
        # when scaled, node 3 should be at the *same* height in both trees, so the label
        # should be the same
        self.verify_basic_svg(svg1)
        self.verify_basic_svg(svg2)
        str_pos = svg1.find('>3<')
        snippet1 = svg1[svg1.rfind("<", 0, str_pos):str_pos]
        str_pos = svg2.find('>3<')
        snippet2 = svg2[svg2.rfind("<", 0, str_pos):str_pos]
        self.assertEqual(snippet1, snippet2)

    def test_draw_simple_ts(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=1)
        svg = ts.draw_svg()
        self.verify_basic_svg(svg, width=200 * ts.num_trees)

    def test_tree_height_scale(self):
        ts = msprime.simulate(4, random_seed=2)
        svg = ts.draw_svg(tree_height_scale="time")
        self.verify_basic_svg(svg)
        svg = ts.draw_svg(tree_height_scale="rank")
        self.verify_basic_svg(svg)
        for bad_scale in [0, "", "NOT A SCALE"]:
            with self.assertRaises(ValueError):
                ts.draw_svg(tree_height_scale=bad_scale)
