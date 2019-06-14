# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (c) 2015-2017 University of Oxford
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
Module responsible for visualisations.
"""
import array
import collections
from _tskit import NULL

import svgwrite
import numpy as np

# NOTE: The code on the top of this module is marked for removal, to be replaced
# by the new SVG and Unicode tree drawing methods on the bottom. The aim is to
# make the current tree.draw() method use these functions, giving a simple high
# level interface. The draw_svg method on the other hand then gives a much
# more flexible way of drawing trees, with direct access to the SVG drawing
# primitives.


def check_format(format):
    if format is None:
        format = "SVG"
    fmt = format.lower()
    supported_formats = ["svg", "ascii", "unicode"]
    if fmt not in supported_formats:
        raise ValueError("Unknown format '{}'. Supported formats are {}".format(
            format, supported_formats))
    return fmt


def draw_tree(
        tree, width=None, height=None, node_labels=None, node_colours=None,
        mutation_labels=None, mutation_colours=None, format=None, edge_colours=None,
        tree_height_scale=None, max_tree_height=None):
    # See tree.draw() for documentation on these arguments.
    fmt = check_format(format)
    if fmt == "svg":
        if width is None:
            width = 200
        if height is None:
            height = 200
        cls = SvgTreeDrawer
    elif fmt == "ascii":
        cls = AsciiTreeDrawer
    elif fmt == "unicode":
        cls = UnicodeTreeDrawer

    # We can't draw trees with zero roots.
    if tree.num_roots == 0:
        raise ValueError("Cannot draw a tree with zero roots")

    td = cls(
        tree, width=width, height=height,
        node_labels=node_labels, node_colours=node_colours,
        mutation_labels=mutation_labels, mutation_colours=mutation_colours,
        edge_colours=edge_colours, tree_height_scale=tree_height_scale,
        max_tree_height=max_tree_height)
    return td.draw()


# NOTE The design of these classes is pretty poor. Could badly do with a rewrite.

class TreeDrawer(object):
    """
    A class to draw sparse trees in SVG format.
    """
    # NOTE: This was introduced as a way to centralise the SVG and text drawing
    # code, but isn't actually used now. Probably the right thing to do is to
    # have a more abstract 'canvas' idea which the backends draw onto, and the
    # coordinates get transformed as required.
    discretise_coordinates = False

    def _discretise(self, x):
        """
        Discetises the specified value, if necessary.
        """
        ret = x
        if self.discretise_coordinates:
            ret = int(round(x))
        return ret

    def __init__(
            self, tree, width=None, height=None, node_labels=None, node_colours=None,
            mutation_labels=None, mutation_colours=None, edge_colours=None,
            tree_height_scale=None, max_tree_height=None):
        self._tree = tree
        self._num_leaves = len(list(tree.leaves()))
        self._width = width
        self._height = height
        self._x_coords = {}
        self._y_coords = {}
        self._node_labels = {}
        self._node_colours = {}
        self._mutation_labels = {}
        self._mutation_colours = {}
        self._edge_colours = {}
        self._tree_height_scale = tree_height_scale
        self._max_tree_height = max_tree_height

        if tree_height_scale not in [None, "time", "rank"]:
            raise ValueError("tree_height_scale must be one of 'time' or 'rank'")
        numeric_max_tree_height = max_tree_height not in [None, "tree", "ts"]
        if tree_height_scale == "rank" and numeric_max_tree_height:
            raise ValueError("Cannot specify numeric max_tree_height with rank scale")

        # Set the node labels and colours.
        for u in tree.nodes():
            if node_labels is None:
                self._node_labels[u] = str(u)
            else:
                self._node_labels[u] = None
        if node_labels is not None:
            for node, label in node_labels.items():
                self._node_labels[node] = label
        if node_colours is not None:
            for node, colour in node_colours.items():
                self._node_colours[node] = colour
        if edge_colours is not None:
            for node, colour in edge_colours.items():
                self._edge_colours[node] = colour

        # Set the mutation labels.
        for site in tree.sites():
            for mutation in site.mutations:
                if mutation_labels is None:
                    self._mutation_labels[mutation.id] = str(mutation.id)
                else:
                    self._mutation_labels[mutation.id] = None
        if mutation_labels is not None:
            for mutation, label in mutation_labels.items():
                self._mutation_labels[mutation] = label
        if mutation_colours is not None:
            for mutation, colour in mutation_colours.items():
                self._mutation_colours[mutation] = colour

        self._assign_y_coordinates()
        self._assign_x_coordinates()


class SvgTreeDrawer(TreeDrawer):
    """
    Draws trees in SVG format using the svgwrite library.
    """
    def _assign_y_coordinates(self):
        tree = self._tree
        ts = tree.tree_sequence
        if self._tree_height_scale in [None, "time"]:
            if self._max_tree_height in [None, "tree"]:
                max_tree_height = max(tree.time(root) for root in tree.roots)
            elif self._max_tree_height == "ts":
                max_tree_height = ts.max_root_time
            else:
                # Use the numeric tree height value directly.
                max_tree_height = self._max_tree_height
            node_height = {u: tree.time(u) for u in tree.nodes()}
        else:
            assert self._tree_height_scale == "rank"
            assert self._max_tree_height in [None, "tree", "ts"]
            if self._max_tree_height in [None, "tree"]:
                times = {tree.time(u) for u in tree.nodes()}
            elif self._max_tree_height == "ts":
                times = {node.time for node in ts.nodes()}
            depth = {t: 2 * j for j, t in enumerate(sorted(times))}
            node_height = {u: depth[tree.time(u)] for u in tree.nodes()}
            max_tree_height = max(depth.values())
        # In pathological cases, all the roots are at 0
        if max_tree_height == 0:
            max_tree_height = 1

        y_padding = 20
        mutations_over_root = any(
            tree.parent(mut.node) == NULL for mut in tree.mutations())
        root_branch_length = 0
        if mutations_over_root:
            # Allocate a fixed about of space to show the mutations on the
            # 'root branch'
            root_branch_length = self._height / 10
        self._y_scale = (
            self._height - root_branch_length - 2 * y_padding) / max_tree_height
        self._y_coords[-1] = y_padding
        for u in tree.nodes():
            scaled_h = node_height[u] * self._y_scale
            self._y_coords[u] = self._height - scaled_h - y_padding

    def _assign_x_coordinates(self):
        self._x_scale = self._width / (self._num_leaves + 2)
        self._leaf_x = 1
        for root in self._tree.roots:
            self._assign_x_coordinates_node(root)
        self._mutations = []
        node_mutations = collections.defaultdict(list)
        for site in self._tree.sites():
            for mutation in site.mutations:
                node_mutations[mutation.node].append(mutation)
        for child, mutations in node_mutations.items():
            n = len(mutations)
            parent = self._tree.parent(child)
            # Ignore any mutations that are above non-roots that are
            # not in the current tree.
            if child in self._x_coords:
                x = self._x_coords[child]
                y1 = self._y_coords[child]
                y2 = self._y_coords[parent]
                chunk = (y2 - y1) / (n + 1)
                for k, mutation in enumerate(mutations):
                    z = x, self._discretise(y1 + (k + 1) * chunk)
                    self._mutations.append((z, mutation))

    def _assign_x_coordinates_node(self, node):
        """
        Assign x coordinates to all nodes underneath this node.
        """
        if self._tree.is_internal(node):
            children = self._tree.children(node)
            for c in children:
                self._assign_x_coordinates_node(c)
            coords = [self._x_coords[c] for c in children]
            a = min(coords)
            b = max(coords)
            self._x_coords[node] = self._discretise(a + (b - a) / 2)
        else:
            self._x_coords[node] = self._discretise(self._leaf_x * self._x_scale)
            self._leaf_x += 1

    def draw(self):
        """
        Writes the SVG description of this tree and returns the resulting XML
        code as text.
        """
        dwg = svgwrite.Drawing(size=(self._width, self._height), debug=True)
        default_edge_colour = "black"
        default_node_colour = "black"
        default_mutation_colour = "red"
        lines = dwg.add(dwg.g(id='lines', stroke=default_edge_colour))
        nodes = dwg.add(dwg.g(id='nodes', fill=default_node_colour))
        mutations = dwg.add(dwg.g(id='mutations', fill=default_mutation_colour))
        left_labels = dwg.add(dwg.g(font_size=14, text_anchor="start"))
        right_labels = dwg.add(dwg.g(font_size=14, text_anchor="end"))
        mid_labels = dwg.add(dwg.g(font_size=14, text_anchor="middle"))
        for u in self._tree.nodes():
            v = self._tree.get_parent(u)
            x = self._x_coords[u], self._y_coords[u]
            fill = self._node_colours.get(u, default_node_colour)
            if fill is not None:
                # Keep SVG small and clean by only adding node markers if required,
                # and only specifying a fill colour if not the default
                params = {} if fill == default_node_colour else {'fill': fill}
                nodes.add(dwg.circle(center=x, r=3, **params))
            dx = 0
            dy = -5
            labels = mid_labels
            if self._tree.is_leaf(u):
                dy = 20
            elif self._tree.parent(u) != NULL:
                dx = 5
                if self._tree.left_sib(u) == NULL:
                    dx *= -1
                    labels = right_labels
                else:
                    labels = left_labels
            if self._node_labels[u] is not None:
                labels.add(dwg.text(self._node_labels[u], (x[0] + dx, x[1] + dy)))
            if self._tree.parent(u) != NULL:
                y = self._x_coords[v], self._y_coords[v]
                stroke = self._edge_colours.get(u, default_edge_colour)
                if stroke is not None:
                    # Keep SVG small and clean
                    params = {} if stroke == default_edge_colour else {'stroke': stroke}
                    lines.add(dwg.line(x, (x[0], y[1]), **params))
                    lines.add(dwg.line((x[0], y[1]), y, **params))

        # Experimental stuff to render the mutation labels. Not working very
        # well at the moment.
        left_labels = dwg.add(dwg.g(
            font_size=14, text_anchor="start", font_style="italic",
            alignment_baseline="middle"))
        right_labels = dwg.add(dwg.g(
            font_size=14, text_anchor="end", font_style="italic",
            alignment_baseline="middle"))
        for x, mutation in self._mutations:
            r = 3
            fill = self._mutation_colours.get(mutation.id, default_mutation_colour)
            if fill is not None:
                # Keep SVG small and clean
                params = {} if fill == default_mutation_colour else {'fill': fill}
                mutations.add(dwg.rect(
                    insert=(x[0] - r, x[1] - r), size=(2 * r, 2 * r), **params))
            dx = 5
            if self._tree.left_sib(mutation.node) == NULL:
                dx *= -1
                labels = right_labels
            else:
                labels = left_labels
            if self._mutation_labels[mutation.id] is not None:
                dy = 1.5 * r
                labels.add(dwg.text(
                    self._mutation_labels[mutation.id], (x[0] + dx, x[1] + dy)))
        return dwg.tostring()
        # return dwg


class TextTreeDrawer(TreeDrawer):
    """
    Abstract superclass of TreeDrawers that draw trees in a text buffer.
    """
    discretise_coordinates = False

    array_type = None  # the type used for the array.array canvas
    background_char = None  # The fill char
    eol_char = None  # End of line
    left_down_char = None  # left corner of a horizontal line
    right_down_char = None  # right corner of a horizontal line
    horizontal_line_char = None  # horizontal line fill
    vertical_line_char = None  # vertial line fill
    mid_up_char = None  # char in a horizontal line going up
    mid_down_char = None  # char in a horizontal line going down
    mid_up_down_char = None  # char in a horizontal line going down and up

    def _convert_text(self, text):
        """
        Converts the specified string into an array representation that can be
        filled into the text buffer.
        """
        raise NotImplementedError()

    def _assign_y_coordinates(self):
        if self._tree_height_scale == "time":
            raise ValueError("time scaling not currently supported in text trees")
        assert self._tree_height_scale in [None, "rank"]
        assert self._max_tree_height in [None, "tree", "ts"]
        tree = self._tree
        if self._max_tree_height in [None, "tree"]:
            times = {tree.time(u) for u in tree.nodes()}
        elif self._max_tree_height == "ts":
            times = {node.time for node in tree.tree_sequence.nodes()}
        # NOTE the only real difference here between the y coordinates here
        # and rank coordinates in SVG is that we're reversing. This is because
        # the y-axis is measured in different directions. We could resolve this
        # with a generic canvas that we draw on.
        depth = {t: 2 * j for j, t in enumerate(sorted(times, reverse=True))}
        max_tree_height = max(depth.values())
        for u in self._tree.nodes():
            self._y_coords[u] = depth[self._tree.time(u)]
        # TODO This should only be set if height is None, ie., the default
        # strategy is to set the height to the minimum required.
        self._height = max_tree_height + 1

    def _assign_x_coordinates(self):
        # Get the overall width and assign x coordinates.
        x = 0
        for root in self._tree.roots:
            for u in self._tree.nodes(root, order="postorder"):
                if self._tree.is_leaf(u):
                    label_size = 1
                    if self._node_labels[u] is not None:
                        label_size = len(self._node_labels[u])
                    self._x_coords[u] = x
                    x += label_size + 1
                else:
                    coords = [self._x_coords[c] for c in self._tree.children(u)]
                    if len(coords) == 1:
                        self._x_coords[u] = coords[0]
                    else:
                        a = min(coords)
                        b = max(coords)
                        assert b - a > 1
                        self._x_coords[u] = int(round((a + (b - a) / 2)))
            x += 1
        self._width = x + 1

    def _draw(self):
        w = self._width
        h = self._height

        # Create a width * height canvas of spaces.
        canvas = array.array(self.array_type, (w * h) * [self.background_char])
        for u in self._tree.nodes():
            col = self._x_coords[u]
            row = self._y_coords[u]
            j = row * w + col
            label = self._convert_text(self._node_labels[u])
            n = len(label)
            canvas[j: j + n] = label
            if self._tree.is_internal(u):
                children = self._tree.children(u)
                row += 1
                left = min(self._x_coords[v] for v in children)
                right = max(self._x_coords[v] for v in children)
                for col in range(left + 1, right):
                    canvas[row * w + col] = self.horizontal_line_char
                if len(self._tree.children(u)) == 1:
                    canvas[row * w + self._x_coords[u]] = self.vertical_line_char
                else:
                    canvas[row * w + self._x_coords[u]] = self.mid_up_char
                for v in children:
                    col = self._x_coords[v]
                    canvas[row * w + col] = self.mid_down_char
                    if col == self._x_coords[u]:
                        canvas[row * w + col] = self.mid_up_down_char
                    for j in range(row + 1, self._y_coords[v]):
                        canvas[j * w + col] = self.vertical_line_char
                if left == right:
                    canvas[row * w + left] = self.vertical_line_char
                else:
                    canvas[row * w + left] = self.left_down_char
                    canvas[row * w + right] = self.right_down_char

        # Put in the EOLs last so that if we can't overwrite them.
        for row in range(h):
            canvas[row * w + w - 1] = self.eol_char
        return canvas


# NOTE: hopefully this can be dropped soon. See
# https://github.com/tskit-dev/tskit/issues/174.
class AsciiTreeDrawer(TextTreeDrawer):
    """
    Draws an ASCII rendering of a tree.
    """
    array_type = 'b'
    background_char = ord(' ')
    eol_char = ord('\n')
    left_down_char = ord('+')
    right_down_char = ord('+')
    horizontal_line_char = ord('-')
    vertical_line_char = ord('|')
    mid_up_char = ord('+')
    mid_down_char = ord('+')
    mid_up_down_char = ord('+')

    def _convert_text(self, text):
        if text is None:
            text = "|"  # vertical line char
        return array.array(self.array_type, text.encode())

    def draw(self):
        s = self._draw().tostring().decode()
        return s


class UnicodeTreeDrawer(TextTreeDrawer):
    """
    Draws an Unicode rendering of a tree using box drawing characters.
    """
    array_type = 'u'
    background_char = ' '
    eol_char = '\n'
    left_down_char = "\u250F"
    right_down_char = "\u2513"
    horizontal_line_char = "\u2501"
    vertical_line_char = "\u2503"
    mid_up_char = "\u253b"
    mid_down_char = "\u2533"
    mid_up_down_char = "\u254b"

    def _convert_text(self, text):
        if text is None:
            text = self.vertical_line_char
        return array.array(self.array_type, text)

    def draw(self):
        return self._draw().tounicode()


#
# New API - separate classes for drawing text and SVG
#

class SvgTreeSequence(object):
    """
    TODO: this is badly structured right now. What we should do is
    Move all of the tree drawing logic into the SvgTree class, and then
    this class should combine these linearly. Each tree will be an
    independant SVG entity, so that we can manipulate it.
    """

    def __init__(
            self, ts, size=None, tree_height_scale=None, max_tree_height=None,
            node_attrs=None, edge_attrs=None, node_label_attrs=None):
        self.ts = ts
        if size is None:
            size = (200 * ts.num_trees, 200)
        self.image_size = size
        self.drawing = svgwrite.Drawing(size=self.image_size, debug=True)
        self.node_labels = {u: str(u) for u in range(ts.num_nodes)}
        # TODO add general padding arguments following matplotlib's terminology.
        self.axes_x_offset = 15
        self.axes_y_offset = 10
        self.treebox_x_offset = self.axes_x_offset + 5
        self.treebox_y_offset = self.axes_y_offset + 5
        x = self.treebox_x_offset
        treebox_width = size[0] - 2 * self.treebox_x_offset
        treebox_height = size[1] - 2 * self.treebox_y_offset
        tree_width = treebox_width / ts.num_trees
        svg_trees = [
            SvgTree(
                tree, (tree_width, treebox_height),
                tree_height_scale=tree_height_scale,
                node_attrs=node_attrs, edge_attrs=edge_attrs,
                node_label_attrs=node_label_attrs)
            for tree in ts.trees()]

        ticks = []
        y = self.treebox_y_offset
        defs = self.drawing.defs

        for tree, svg_tree in zip(ts.trees(), svg_trees):
            defs.add(svg_tree.root_group)

        for tree in ts.trees():
            tree_id = "#tree_{}".format(tree.index)
            use = self.drawing.use(tree_id, (x, y))
            self.drawing.add(use)
            ticks.append((x, tree.interval[0]))
            x += tree_width
        ticks.append((x, ts.sequence_length))

        dwg = self.drawing

        # # Debug --- draw the tree and axes boxes
        # w = self.image_size[0] - 2 * self.treebox_x_offset
        # h = self.image_size[1] - 2 * self.treebox_y_offset
        # dwg.add(dwg.rect((self.treebox_x_offset, self.treebox_y_offset), (w, h),
        #     fill="white", fill_opacity=0, stroke="black", stroke_dasharray="15,15"))
        # w = self.image_size[0] - 2 * self.axes_x_offset
        # h = self.image_size[1] - 2 * self.axes_y_offset
        # dwg.add(dwg.rect((self.axes_x_offset, self.axes_y_offset), (w, h),
        #     fill="white", fill_opacity=0, stroke="black", stroke_dasharray="5,5"))

        axes_left = self.treebox_x_offset
        axes_right = self.image_size[0] - self.treebox_x_offset
        y = self.image_size[1] - 2 * self.axes_y_offset
        dwg.add(dwg.line((axes_left, y), (axes_right, y), stroke="black"))
        for x, genome_coord in ticks:
            delta = 5
            dwg.add(dwg.line((x, y - delta), (x, y + delta), stroke="black"))
            dwg.add(dwg.text(
                "{:.2f}".format(genome_coord), (x, y + 20),
                font_size=14, text_anchor="middle", font_weight="bold"))


class SvgTree(object):
    """
    An SVG representation of a single tree.

    TODO should provide much more SVG structure which we document fully
    to that the SVG elements can be manipulated directly by the user.
    For example, every edge should be given an SVG ID so that it can
    be referred to and modified.

    """
    def __init__(
            self, tree, size=None, tree_height_scale=None, max_tree_height=None,
            node_attrs=None, edge_attrs=None, node_label_attrs=None):
        self.tree = tree
        if size is None:
            size = (200, 200)
        self.image_size = size
        self.setup_drawing()
        self.node_labels = {u: str(u) for u in tree.nodes()}
        self.treebox_x_offset = 10
        self.treebox_y_offset = 10
        self.treebox_width = size[0] - 2 * self.treebox_x_offset
        self.assign_y_coordinates(tree_height_scale, max_tree_height)
        self.node_x_coord_map = self.assign_x_coordinates(
            tree, self.treebox_x_offset, self.treebox_width)
        self.edge_attrs = {}
        self.node_attrs = {}
        self.node_label_attrs = {}
        for u in tree.nodes():
            self.edge_attrs[u] = {}
            if edge_attrs is not None and u in edge_attrs:
                self.edge_attrs[u].update(edge_attrs[u])
            self.node_attrs[u] = {"r": 3}
            if node_attrs is not None and u in node_attrs:
                self.node_attrs[u].update(node_attrs[u])
            self.node_label_attrs[u] = {"text": "{}".format(u)}
            if node_label_attrs is not None and u in node_label_attrs:
                self.node_label_attrs[u].update(node_label_attrs[u])
        self.draw()

    def setup_drawing(self):
        self.drawing = svgwrite.Drawing(size=self.image_size, debug=True)
        dwg = self.drawing

        self.root_group = dwg.add(dwg.g(id='tree_{}'.format(self.tree.index)))
        self.edges = self.root_group.add(dwg.g(id='edges',  stroke="black", fill="none"))
        self.nodes = self.root_group.add(dwg.g(id='nodes'))
        self.mutations = self.root_group.add(dwg.g(id='mutations', fill="red"))
        self.left_labels = self.root_group.add(dwg.g(font_size=14, text_anchor="start"))
        self.right_labels = self.root_group.add(dwg.g(font_size=14, text_anchor="end"))
        self.mid_labels = self.root_group.add(dwg.g(font_size=14, text_anchor="middle"))
        self.mutation_left_labels = self.root_group.add(dwg.g(
            font_size=14, text_anchor="start", font_style="italic",
            alignment_baseline="middle"))
        self.mutation_right_labels = self.root_group.add(dwg.g(
            font_size=14, text_anchor="end", font_style="italic",
            alignment_baseline="middle"))

    def assign_y_coordinates(self, tree_height_scale, max_tree_height):
        ts = self.tree.tree_sequence
        node_time = ts.tables.nodes.time
        if tree_height_scale in [None, "time"]:
            node_height = node_time
            if max_tree_height is None:
                max_tree_height = ts.max_root_time
        else:
            if tree_height_scale != "rank":
                raise ValueError(
                    "Only 'time' and 'rank' are supported for tree_height_scale")
            depth = {t: 2 * j for j, t in enumerate(np.unique(node_time))}
            node_height = [depth[node_time[u]] for u in range(ts.num_nodes)]
            if max_tree_height is None:
                max_tree_height = max(depth.values())
        # In pathological cases, all the roots are at 0
        if max_tree_height == 0:
            max_tree_height = 1

        # TODO should make this a parameter somewhere. This is padding to keep the
        # node labels within the treebox
        label_padding = 10
        y_padding = self.treebox_y_offset + 2 * label_padding
        mutations_over_root = any(
            any(tree.parent(mut.node) == NULL for mut in tree.mutations())
            for tree in ts.trees())
        root_branch_length = 0
        height = self.image_size[1]
        if mutations_over_root:
            # Allocate a fixed about of space to show the mutations on the
            # 'root branch'
            root_branch_length = height / 10  # FIXME just draw branch??
        y_scale = (height - root_branch_length - 2 * y_padding) / max_tree_height
        self.node_y_coord_map = [
                height - y_scale * node_height[u] - y_padding
                for u in range(ts.num_nodes)]

    def assign_x_coordinates(self, tree, x_start, width):
        num_leaves = len(list(tree.leaves()))
        x_scale = width / (num_leaves + 1)
        node_x_coord_map = {}
        leaf_x = x_start
        for root in tree.roots:
            for u in tree.nodes(root, order="postorder"):
                if tree.is_leaf(u):
                    leaf_x += x_scale
                    node_x_coord_map[u] = leaf_x
                else:
                    child_coords = [node_x_coord_map[c] for c in tree.children(u)]
                    if len(child_coords) == 1:
                        node_x_coord_map[u] = child_coords[0]
                    else:
                        a = min(child_coords)
                        b = max(child_coords)
                        assert b - a > 1
                        node_x_coord_map[u] = a + (b - a) / 2
        return node_x_coord_map

    def draw(self):
        dwg = self.drawing
        node_x_coord_map = self.node_x_coord_map
        node_y_coord_map = self.node_y_coord_map
        tree = self.tree

        node_mutations = collections.defaultdict(list)
        for site in tree.sites():
            for mutation in site.mutations:
                node_mutations[mutation.node].append(mutation)

        for u in tree.nodes():
            pu = node_x_coord_map[u], node_y_coord_map[u]
            node_id = "node_{}_{}".format(tree.index, u)
            self.nodes.add(dwg.circle(id=node_id, center=pu, **self.node_attrs[u]))
            dx = 0
            dy = -5
            labels = self.mid_labels
            if tree.is_leaf(u):
                dy = 20
            elif tree.parent(u) != NULL:
                dx = 5
                if tree.left_sib(u) == NULL:
                    dx *= -1
                    labels = self.right_labels
                else:
                    labels = self.left_labels
            # TODO add ID to node label text.
            labels.add(dwg.text(
                insert=(pu[0] + dx, pu[1] + dy), **self.node_label_attrs[u]))
            v = tree.parent(u)
            if v != NULL:
                edge_id = "edge_{}_{}".format(tree.index, u)
                pv = node_x_coord_map[v], node_y_coord_map[v]
                path = dwg.path(
                    [("M", pu), ("V", pv[1]), ("H", pv[0])], id=edge_id,
                    **self.edge_attrs[u])
                self.edges.add(path)

                # TODO do something with mutations over the root
                # Draw the mutations
                num_mutations = len(node_mutations[u])
                delta = (pv[1] - pu[1]) / (num_mutations + 1)
                x = pu[0]
                y = pv[1] - delta
                r = 3
                # TODO add support for manipulating mutation properties and IDs
                for mutation in node_mutations[u]:
                    self.mutations.add(dwg.rect(
                        insert=(x - r, y - r), size=(2 * r, 2 * r)))
                    dx = 5
                    if tree.left_sib(mutation.node) == NULL:
                        dx *= -1
                        labels = self.mutation_right_labels
                    else:
                        labels = self.mutation_left_labels
                    dy = 1.5 * r
                    labels.add(dwg.text(str(mutation.id), (x + dx, y + dy)))
                    y -= delta


class TextTreeSequence(object):
    """
    Draw a tree sequence as horizontal line of trees.
    """
    def __init__(self, ts):
        self.ts = ts
        self.canvas = None
        self.draw()

    def draw(self):
        trees = [
            TextTree(tree, tree_height_scale="rank", max_tree_height="ts")
            for tree in self.ts.trees()]
        self.width = sum(tree.width + 2 for tree in trees) - 1
        self.height = max(tree.height for tree in trees)
        self.canvas = np.zeros((self.height, self.width), dtype=str)
        self.canvas[:] = " "

        x = 0
        for j, tree in enumerate(trees):
            h, w = tree.canvas.shape
            self.canvas[-h:, x: x + w - 1] = tree.canvas[:, :-1]
            x += w
            self.canvas[:, x] = "┊"
            x += 2
        self.canvas[:, -1] = "\n"

    def __str__(self):
        return "".join(self.canvas.reshape(self.width * self.height))


# # TODO not actually done anything with this yet. See if it we can use it.
# LEFT = "left"
# TOP = "top"


# def check_orientation(orientation):
#     if orientation is None:
#         orientation = LEFT
#     else:
#         orientation = orientation.lower()
#         orientations = [LEFT, TOP]
#         if orientation not in orientations:
#             raise ValueError(
#                 "Unknown orientiation: choose from {}".format(orientations))
#     return orientation


def to_np_unicode(string):
    """
    Converts the specified string to a numpy unicode array.
    """
    # TODO: what's the clean of doing this with numpy?
    # It really wants to create a zero-d Un array here
    # which breaks the assignment below and we end up
    # with n copies of the first char.
    n = len(string)
    np_string = np.zeros(n, dtype="U")
    for j in range(n):
        np_string[j] = string[j]
    return np_string


class TextTree(object):
    """
    Draws a reprentation of a tree using unicode drawing characters written
    to a 2D array.
    """
    # TODO make this an option for to allow using ASCII chars.
    left_down_char = "\u250F"
    right_down_char = "\u2513"
    horizontal_line_char = "\u2501"
    vertical_line_char = "\u2503"
    mid_up_char = "\u253b"
    mid_down_char = "\u2533"
    mid_up_down_char = "\u254b"

    def __init__(
            self, tree, node_labels=None, tree_height_scale=None, max_tree_height=None):
        self.tree = tree
        self.tree_height_scale = tree_height_scale
        self.max_tree_height = max_tree_height
        self.num_leaves = len(list(tree.leaves()))
        # self.orientation = check_orientation(orientation)
        # TODO Change to size tuple
        self.width = None
        self.height = None
        self.canvas = None
        # Placement of nodes in the 2D space. Nodes are positioned in one
        # dimension based on traversal ordering and by their time in the
        # other dimension. These are mapped to x and y coordinates according
        # to the orientation.
        self.traversal_position = {}  # Position of nodes in traversal space
        self.time_position = {}
        # Labels for nodes
        self.node_labels = {}

        # TODO Clear up the logic here. What do we actually support?
        if tree_height_scale not in [None, "time", "rank"]:
            raise ValueError("tree_height_scale must be one of 'time' or 'rank'")
        numeric_max_tree_height = max_tree_height not in [None, "tree", "ts"]
        if tree_height_scale == "rank" and numeric_max_tree_height:
            raise ValueError("Cannot specify numeric max_tree_height with rank scale")

        # Set the node labels and colours.
        for u in tree.nodes():
            if node_labels is None:
                # If we don't specify node_labels, default to node ID
                self.node_labels[u] = str(u)
            else:
                # If we do specify node_labels, default an empty line
                self.node_labels[u] = self.vertical_line_char
        if node_labels is not None:
            for node, label in node_labels.items():
                self.node_labels[node] = label

        self.assign_time_positions()
        self.assign_traversal_positions()

        # TODO This should only be set if height is None, ie., the default
        # strategy is to set the height to the minimum required.
        self.height = max(self.time_position.values()) + 1
        self.width = max(self.traversal_position.values()) + 2
        self.draw()

    def assign_time_positions(self):
        if self.tree_height_scale == "time":
            raise ValueError("time scaling not currently supported in text trees")
        assert self.tree_height_scale in [None, "rank"]
        assert self.max_tree_height in [None, "tree", "ts"]
        tree = self.tree
        if self.max_tree_height in [None, "tree"]:
            times = {tree.time(u) for u in tree.nodes()}
        elif self.max_tree_height == "ts":
            times = {node.time for node in tree.tree_sequence.nodes()}
        depth = {t: 2 * j for j, t in enumerate(sorted(times, reverse=True))}
        for u in self.tree.nodes():
            self.time_position[u] = depth[self.tree.time(u)]

    def assign_traversal_positions(self):
        # Get the overall width and assign x coordinates.
        x = 0
        for root in self.tree.roots:
            for u in self.tree.nodes(root, order="postorder"):
                if self.tree.is_leaf(u):
                    label_size = len(self.node_labels[u])
                    self.traversal_position[u] = x
                    x += label_size + 1
                else:
                    coords = [self.traversal_position[c] for c in self.tree.children(u)]
                    if len(coords) == 1:
                        self.traversal_position[u] = coords[0]
                    else:
                        a = min(coords)
                        b = max(coords)
                        assert b - a > 1
                        self.traversal_position[u] = int(round((a + (b - a) / 2)))
            x += 1

    def draw(self):
        # Create a width * height canvas of spaces.
        self.canvas = np.zeros((self.height, self.width), dtype=str)
        self.canvas[:] = " "
        for u in self.tree.nodes():
            xu = self.traversal_position[u]
            yu = self.time_position[u]
            label = to_np_unicode(self.node_labels[u])
            self.canvas[yu, xu: xu + label.shape[0]] = label
            children = self.tree.children(u)
            # Not quite right, we're still getting leaves wrong.
            if len(children) > 0:
                if len(children) == 1:
                    yv = self.time_position[children[0]]
                    self.canvas[yu + 1: yv, xu] = self.vertical_line_char
                else:
                    left = min(self.traversal_position[v] for v in children)
                    right = max(self.traversal_position[v] for v in children)
                    y = yu + 1
                    self.canvas[y, left + 1: right] = self.horizontal_line_char
                    self.canvas[y, xu] = self.mid_up_char
                    for v in children:
                        xv = self.traversal_position[v]
                        yv = self.time_position[v]
                        self.canvas[yu + 2: yv, xv] = self.vertical_line_char
                        mid_char = (
                            self.mid_up_down_char if xv == xu else self.mid_down_char)
                        self.canvas[yu + 1, xv] = mid_char
                    self.canvas[y, left] = self.left_down_char
                    self.canvas[y, right] = self.right_down_char

        # Put in the EOLs last so that if we can't overwrite them.
        self.canvas[:, -1] = "\n"
        # print(self.canvas)

    def __str__(self):
        return "".join(self.canvas.reshape(self.width * self.height))
