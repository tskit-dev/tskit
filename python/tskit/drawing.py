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
import collections
import itertools
import logging
import math
import numbers
import operator
from dataclasses import dataclass

import numpy as np
import svgwrite

import tskit.util as util
from _tskit import NULL

LEFT = "left"
RIGHT = "right"
TOP = "top"
BOTTOM = "bottom"


def check_orientation(orientation):
    if orientation is None:
        orientation = TOP
    else:
        orientation = orientation.lower()
        orientations = [LEFT, RIGHT, TOP, BOTTOM]
        if orientation not in orientations:
            raise ValueError(f"Unknown orientiation: choose from {orientations}")
    return orientation


def check_max_tree_height(max_tree_height, allow_numeric=True):
    if max_tree_height is None:
        max_tree_height = "tree"
    is_numeric = isinstance(max_tree_height, numbers.Real)
    if max_tree_height not in ["tree", "ts"] and not allow_numeric:
        raise ValueError("max_tree_height must be 'tree' or 'ts'")
    if max_tree_height not in ["tree", "ts"] and (allow_numeric and not is_numeric):
        raise ValueError(
            "max_tree_height must be a numeric value or one of 'tree' or 'ts'"
        )
    return max_tree_height


def check_tree_height_scale(tree_height_scale):
    if tree_height_scale is None:
        tree_height_scale = "time"
    if tree_height_scale not in ["time", "log_time", "rank"]:
        raise ValueError("tree_height_scale must be 'time', 'log_time' or 'rank'")
    return tree_height_scale


def check_format(format):  # noqa A002
    if format is None:
        format = "SVG"  # noqa A001
    fmt = format.lower()
    supported_formats = ["svg", "ascii", "unicode"]
    if fmt not in supported_formats:
        raise ValueError(
            "Unknown format '{}'. Supported formats are {}".format(
                format, supported_formats
            )
        )
    return fmt


def check_order(order):
    """
    Checks the specified drawing order is valid and returns the corresponding
    tree traversal order.
    """
    if order is None:
        order = "minlex"
    traversal_orders = {
        "minlex": "minlex_postorder",
        "tree": "postorder",
    }
    if order not in traversal_orders:
        raise ValueError(
            f"Unknown display order '{order}'. "
            f"Supported orders are {list(traversal_orders.keys())}"
        )
    return traversal_orders[order]


def check_x_scale(x_scale):
    """
    Checks the specified x_scale is valid and sets default if None
    """
    if x_scale is None:
        x_scale = "physical"
    x_scales = ["physical", "treewise"]
    if x_scale not in x_scales:
        raise ValueError(
            f"Unknown display x_scale '{x_scale}'. " f"Supported orders are {x_scales}"
        )
    return x_scale


def check_ticks(ticks, default_iterable):
    """
    This is trivial, but implemented as a function so that later we can implement a tick
    locator function, such that e.g. ticks=5 selects ~5 nicely spaced tick locations
    (ideally with sensible behaviour for log scales)
    """
    if ticks is None:
        return default_iterable
    return ticks


def rnd(x):
    """
    Round a number so that the output SVG doesn't have unneeded precision
    """
    digits = 6
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    x = round(x, digits)
    if int(x) == x:
        return int(x)
    return x


def draw_tree(
    tree,
    width=None,
    height=None,
    node_labels=None,
    node_colours=None,
    mutation_labels=None,
    mutation_colours=None,
    format=None,  # noqa A002
    edge_colours=None,
    tree_height_scale=None,
    max_tree_height=None,
    order=None,
):

    # See tree.draw() for documentation on these arguments.
    fmt = check_format(format)
    if fmt == "svg":
        if width is None:
            width = 200
        if height is None:
            height = 200

        def remap_style(original_map, new_key, none_value):
            if original_map is None:
                return None
            new_map = {}
            for key, value in original_map.items():
                if value is None:
                    new_map[key] = {"style": none_value}
                else:
                    new_map[key] = {"style": f"{new_key}:{value};"}
            return new_map

        # Set style rather than fill & stroke directly to override top stylesheet
        # Old semantics were to not draw the node if colour is None.
        # Setting opacity to zero has the same effect.
        node_attrs = remap_style(node_colours, "fill", "fill-opacity:0;")
        edge_attrs = remap_style(edge_colours, "stroke", "stroke-opacity:0;")
        mutation_attrs = remap_style(mutation_colours, "fill", "fill-opacity:0;")

        node_label_attrs = None
        tree = SvgTree(
            tree,
            (width, height),
            node_labels=node_labels,
            mutation_labels=mutation_labels,
            tree_height_scale=tree_height_scale,
            max_tree_height=max_tree_height,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            node_label_attrs=node_label_attrs,
            mutation_attrs=mutation_attrs,
            order=order,
        )
        return tree.drawing.tostring()

    else:
        if width is not None:
            raise ValueError("Text trees do not support width")
        if height is not None:
            raise ValueError("Text trees do not support height")
        if mutation_labels is not None:
            raise ValueError("Text trees do not support mutation_labels")
        if mutation_colours is not None:
            raise ValueError("Text trees do not support mutation_colours")
        if node_colours is not None:
            raise ValueError("Text trees do not support node_colours")
        if edge_colours is not None:
            raise ValueError("Text trees do not support edge_colours")
        if tree_height_scale is not None:
            raise ValueError("Text trees do not support tree_height_scale")

        use_ascii = fmt == "ascii"
        text_tree = VerticalTextTree(
            tree,
            node_labels=node_labels,
            max_tree_height=max_tree_height,
            use_ascii=use_ascii,
            orientation=TOP,
            order=order,
        )
        return str(text_tree)


def add_class(attrs_dict, classes_str):
    """Adds the classes_str to the 'class' key in attrs_dict, or creates it"""
    try:
        attrs_dict["class"] += " " + classes_str
    except KeyError:
        attrs_dict["class"] = classes_str


@dataclass
class Plotbox:
    total_size: list
    pad_top: float
    pad_left: float
    pad_bottom: float
    pad_right: float

    @property
    def max_x(self):
        return self.total_size[0]

    @property
    def max_y(self):
        return self.total_size[1]

    @property
    def top(self):  # Alias for consistency with top & bottom
        return self.pad_top

    @property
    def left(self):  # Alias for consistency with top & bottom
        return self.pad_left

    @property
    def bottom(self):
        return self.max_y - self.pad_bottom

    @property
    def right(self):
        return self.max_x - self.pad_right

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    def __post_init__(self):
        if self.width < 1 or self.height < 1:
            raise ValueError("Image size too small to fit")

    def draw(self, dwg, add_to, colour="grey"):
        # used for debugging
        add_to.add(
            dwg.rect(
                (0, 0),
                (self.max_x, self.max_y),
                fill="white",
                fill_opacity=0,
                stroke=colour,
                stroke_dasharray="15,15",
                class_="outer_plotbox",
            )
        )
        add_to.add(
            dwg.rect(
                (self.left, self.top),
                (self.width, self.height),
                fill="white",
                fill_opacity=0,
                stroke=colour,
                stroke_dasharray="5,5",
                class_="inner_plotbox",
            )
        )


class SvgPlot:
    """ The base class for plotting either a tree or a tree sequence as an SVG file"""

    standard_style = (
        ".tree-sequence .background path {fill: #808080; fill-opacity:0}"
        ".tree-sequence .background path:nth-child(odd) {fill-opacity:.1}"
        ".axes {font-size: 14px}"
        ".x-axis .tick .lab {font-weight: bold}"
        ".axes, .tree {font-size: 14px; text-anchor:middle}"
        ".y-axis line.grid {stroke: #FAFAFA}"
        ".y-axis > .lab text {transform: translateX(0.8em) rotate(-90deg)}"
        ".x-axis .tick g {transform: translateY(0.9em)}"
        ".x-axis > .lab text {transform: translateY(-0.8em)}"
        ".axes line, .edge {stroke:black; fill:none}"
        ".node > .sym {fill: black; stroke: none}"
        ".site > .sym {stroke: black}"
        ".mut text {fill: red; font-style: italic}"
        ".mut line {fill: none; stroke: none}"  # Default hide mut line to expose edges
        ".mut .sym {fill: none; stroke: red}"
        ".node .mut .sym {stroke-width: 1.5px}"
        ".tree text, .tree-sequence text {dominant-baseline: central}"
        ".plotbox .lab.lft {text-anchor: end}"
        ".plotbox .lab.rgt {text-anchor: start}"
    )

    # TODO: we may want to make some of the constants below into parameters
    text_height = 14  # May want to calculate this based on a font size
    line_height = text_height * 1.2  # allowing padding above and below a line
    root_branch_fraction = 1 / 8  # Rel root branch len, unless it has a timed mutation
    default_tick_length = 5
    default_tick_length_site = 10
    # Placement of the axes lines within the padding - not used unless axis is plotted
    default_x_axis_offset = 20
    default_y_axis_offset = 40

    def __init__(
        self,
        ts,
        size,
        root_svg_attributes,
        style,
        svg_class,
        tree_height_scale,
        x_axis=None,
        y_axis=None,
        x_label=None,
        y_label=None,
        debug_box=None,
    ):
        """
        Creates self.drawing, an svgwrite.Drawing object for further use, and populates
        it with a stylesheet and base group. The root_groups will be populated with
        items that can be accessed from the ourside, such as the plotbox, axes, etc.
        """
        self.ts = ts
        self.image_size = size
        self.svg_class = svg_class
        if root_svg_attributes is None:
            root_svg_attributes = {}
        self.root_svg_attributes = root_svg_attributes
        dwg = svgwrite.Drawing(size=size, debug=True, **root_svg_attributes)
        # Put all styles in a single stylesheet (required for Inkscape 0.92)
        style = self.standard_style + ("" if style is None else style)
        dwg.defs.add(dwg.style(style))
        self.dwg_base = dwg.add(dwg.g(class_=svg_class))
        self.root_groups = {}
        self.debug_box = debug_box
        self.drawing = dwg
        self.tree_height_scale = check_tree_height_scale(tree_height_scale)
        self.y_axis = y_axis
        self.x_axis = x_axis
        if x_label is None and x_axis:
            x_label = "Genome position"
        if y_label is None and y_axis:
            if tree_height_scale == "rank":
                y_label = "Ranked node time"
            else:
                y_label = "Time"
        self.x_label = x_label
        self.y_label = y_label

    def get_plotbox(self):
        """
        Get the svgwrite plotbox (contains the tree(s) but not axes etc), creating it
        if necessary.
        """
        if "plotbox" not in self.root_groups:
            dwg = self.drawing
            self.root_groups["plotbox"] = self.dwg_base.add(dwg.g(class_="plotbox"))
        return self.root_groups["plotbox"]

    def add_text_in_group(self, text, add_to, pos, group_class=None, **kwargs):
        """
        Add the text to the elem within a group; allows text rotations to work smoothly,
        otherwise, if x & y parameters are used to position text, rotations applied to
        the text tag occur around the (0,0) point of the containing group
        """
        dwg = self.drawing
        group_attributes = {"transform": f"translate({rnd(pos[0])},{rnd(pos[1])})"}
        if group_class is not None:
            group_attributes["class_"] = group_class
        grp = add_to.add(dwg.g(**group_attributes))
        grp.add(dwg.text(text, **kwargs))

    def set_spacing(self, top=0, left=0, bottom=0, right=0):
        """
        Set edges, but allow space for axes etc
        """
        self.x_axis_offset = self.default_x_axis_offset
        self.y_axis_offset = self.default_y_axis_offset
        if self.x_label:
            self.x_axis_offset += self.line_height
        if self.y_label:
            self.y_axis_offset += self.line_height
        if self.x_axis:
            bottom += self.x_axis_offset
        if self.y_axis:
            left = self.y_axis_offset  # Override user-provided, so y-axis is at x=0
        self.plotbox = Plotbox(self.image_size, top, left, bottom, right)
        if self.debug_box:
            self.root_groups["debug"] = self.dwg_base.add(
                self.drawing.g(class_="debug")
            )
            self.plotbox.draw(self.drawing, self.root_groups["debug"])

    def get_axes(self):
        if "axes" not in self.root_groups:
            self.root_groups["axes"] = self.dwg_base.add(self.drawing.g(class_="axes"))
        return self.root_groups["axes"]

    def draw_x_axis(
        self,
        tick_positions=None,  # np.array of ax ticks below (+ above if sites is None)
        tick_labels=None,  # Tick labels below axis. If None, use the position value
        tick_length_lower=default_tick_length,
        tick_length_upper=None,  # If None, use the same as tick_length_lower
        sites=None,  # An iterator over site objects to plot as ticks above the x axis
    ):
        if not self.x_axis and not self.x_label:
            return
        dwg = self.drawing
        axes = self.get_axes()
        x_axis = axes.add(dwg.g(class_="x-axis"))
        if self.x_label:
            self.add_text_in_group(
                self.x_label,
                x_axis,
                pos=((self.plotbox.left + self.plotbox.right) / 2, self.plotbox.max_y),
                group_class="lab",
                text_anchor="middle",
            )
        if self.x_axis:
            if tick_length_upper is None:
                tick_length_upper = tick_length_lower
            y = rnd(self.plotbox.max_y - self.x_axis_offset)
            x_axis.add(dwg.line((self.plotbox.left, y), (self.plotbox.right, y)))
            if tick_positions is not None:
                if tick_labels is None or isinstance(tick_labels, np.ndarray):
                    if tick_labels is None:
                        tick_labels = tick_positions
                    integer_ticks = np.all(np.round(tick_labels) == tick_labels)
                    label_precision = 0 if integer_ticks else 2
                    tick_labels = [f"{lab:.{label_precision}f}" for lab in tick_labels]

                upper_length = -tick_length_upper if sites is None else 0
                for pos, lab in itertools.zip_longest(tick_positions, tick_labels):
                    tick = x_axis.add(
                        dwg.g(
                            class_="tick",
                            transform=f"translate({rnd(self.x_transform(pos))} {y})",
                        )
                    )
                    tick.add(
                        dwg.line((0, rnd(upper_length)), (0, rnd(tick_length_lower)))
                    )
                    self.add_text_in_group(
                        lab, tick, pos=(0, tick_length_lower), group_class="lab"
                    )
            if sites is not None:
                # Add sites as upper chevrons
                for s in sites:
                    x = self.x_transform(s.position)
                    site = x_axis.add(
                        dwg.g(
                            class_=f"site s{s.id}", transform=f"translate({rnd(x)} {y})"
                        )
                    )
                    site.add(
                        dwg.line((0, 0), (0, rnd(-tick_length_upper)), class_="sym")
                    )
                    for i, m in enumerate(reversed(s.mutations)):
                        mut = dwg.g(class_=f"mut m{m.id}")
                        h = -i * 4 - 1.5
                        w = tick_length_upper / 4
                        mut.add(
                            dwg.polyline(
                                [
                                    (rnd(w), rnd(h - 2 * w)),
                                    (0, rnd(h)),
                                    (rnd(-w), rnd(h - 2 * w)),
                                ],
                                class_="sym",
                            )
                        )
                        site.add(mut)

    def draw_y_axis(
        self,
        upper=None,  # In plot coords
        lower=None,  # In plot coords
        tick_positions=None,
        tick_length_left=default_tick_length,
        gridlines=None,
    ):
        if not self.y_axis and not self.y_label:
            return
        if upper is None:
            upper = self.plotbox.top
        if lower is None:
            lower = self.plotbox.bottom
        dwg = self.drawing
        x = rnd(self.y_axis_offset)
        axes = self.get_axes()
        y_axis = axes.add(dwg.g(class_="y-axis"))
        if self.y_label:
            self.add_text_in_group(
                self.y_label,
                y_axis,
                pos=(0, (upper + lower) / 2),
                group_class="lab",
                text_anchor="middle",
            )
        if self.y_axis:
            y_axis.add(dwg.line((x, rnd(lower)), (x, rnd(upper))))
            if tick_positions is not None:
                for pos in tick_positions:
                    tick = y_axis.add(
                        dwg.g(
                            class_="tick",
                            transform=f"translate({x} {rnd(self.y_transform(pos))})",
                        )
                    )
                    if gridlines:
                        tick.add(
                            dwg.line(
                                (0, 0), (rnd(self.plotbox.right - x), 0), class_="grid"
                            )
                        )
                    tick.add(dwg.line((0, 0), (rnd(-tick_length_left), 0)))
                    self.add_text_in_group(
                        f"{pos:.2f}",
                        tick,
                        pos=(rnd(-tick_length_left), 0),
                        group_class="lab",
                        text_anchor="end",
                    )

    def x_transform(self, x):
        raise NotImplementedError(
            "No transform func defined for genome pos -> plot coords"
        )

    def y_transform(self, y):
        raise NotImplementedError(
            "No transform func defined for tree height -> plot pos"
        )


class SvgTreeSequence(SvgPlot):
    """
    A class to draw a tree sequence in SVG format.

    See :meth:`TreeSequence.draw_svg` for a description of usage and parameters.
    """

    def __init__(
        self,
        ts,
        size,
        x_scale,
        tree_height_scale,
        node_labels,
        mutation_labels,
        root_svg_attributes,
        style,
        order,
        force_root_branch,
        symbol_size,
        x_axis,
        y_axis,
        x_label,
        y_label,
        y_ticks,
        y_gridlines,
        max_tree_height=None,
        node_attrs=None,
        mutation_attrs=None,
        edge_attrs=None,
        node_label_attrs=None,
        mutation_label_attrs=None,
        **kwargs,
    ):
        if size is None:
            size = (200 * ts.num_trees, 200)
        if max_tree_height is None:
            max_tree_height = "ts"
        # X axis shown by default
        if x_axis is None:
            x_axis = True
        super().__init__(
            ts,
            size,
            root_svg_attributes,
            style,
            svg_class="tree-sequence",
            tree_height_scale=tree_height_scale,
            x_axis=x_axis,
            y_axis=y_axis,
            x_label=x_label,
            y_label=y_label,
            **kwargs,
        )
        x_scale = check_x_scale(x_scale)
        if node_labels is None:
            node_labels = {u: str(u) for u in range(ts.num_nodes)}
        if force_root_branch is None:
            force_root_branch = any(
                any(tree.parent(mut.node) == NULL for mut in tree.mutations())
                for tree in ts.trees()
            )
        # TODO add general padding arguments following matplotlib's terminology.
        self.set_spacing(top=0, left=20, bottom=15, right=20)
        svg_trees = [
            SvgTree(
                tree,
                (self.plotbox.width / ts.num_trees, self.plotbox.height),
                tree_height_scale=tree_height_scale,
                node_labels=node_labels,
                mutation_labels=mutation_labels,
                order=order,
                force_root_branch=force_root_branch,
                symbol_size=symbol_size,
                max_tree_height=max_tree_height,
                node_attrs=node_attrs,
                mutation_attrs=mutation_attrs,
                edge_attrs=edge_attrs,
                node_label_attrs=node_label_attrs,
                mutation_label_attrs=mutation_label_attrs,
                # Do not plot axes on these subplots
                **kwargs,  # pass though e.g. debug boxes
            )
            for tree in ts.trees()
        ]
        y = self.plotbox.top
        self.tree_plotbox = svg_trees[0].plotbox
        self.draw_x_axis(
            x_scale,
            tick_length_lower=self.default_tick_length,  # TODO - parameterize
            tick_length_upper=self.default_tick_length_site,  # TODO - parameterize
        )
        y_low = self.tree_plotbox.bottom
        if y_axis is not None:
            self.y_transform = lambda x: svg_trees[0].y_transform(x) + y
            for svg_tree in svg_trees:
                if self.y_transform(1.234) != svg_tree.y_transform(1.234) + y:
                    # Slight hack: check an arbitrary value is transformed identically
                    raise ValueError(
                        "Can't draw a tree sequence Y axis for trees of varying yscales"
                    )
            y_low = self.y_transform(
                0
            )  # if poss use the zero point for lowest axis pos
            ytimes = np.unique(ts.tables.nodes.time)
            if self.tree_height_scale == "rank":
                ytimes = np.arange(len(ytimes))
            y_ticks = check_ticks(y_ticks, ytimes)
        self.draw_y_axis(
            upper=self.tree_plotbox.top,
            lower=y_low,
            tick_positions=y_ticks,
            tick_length_left=self.default_tick_length,
            gridlines=y_gridlines,
        )

        tree_x = self.plotbox.left
        trees = self.get_plotbox()  # Top-level TS plotbox contains all trees
        trees["class"] = trees["class"] + " trees"
        for svg_tree in svg_trees:
            tree = trees.add(
                self.drawing.g(
                    class_=svg_tree.svg_class, transform=f"translate({rnd(tree_x)} {y})"
                )
            )
            for svg_items in svg_tree.root_groups.values():
                tree.add(svg_items)
            tree_x += svg_tree.image_size[0]
            assert self.tree_plotbox == svg_tree.plotbox

    def draw_x_axis(
        self,
        x_scale,
        tick_length_lower=SvgPlot.default_tick_length,
        tick_length_upper=SvgPlot.default_tick_length_site,
    ):
        """
        Add extra functionality to the original draw_x_axis method in SvgPlot, mainly
        to account for the background shading that is displayed in a tree sequence
        """
        if not self.x_axis and not self.x_label:
            return
        if x_scale == "physical":
            breaks = self.ts.breakpoints(as_array=True)
            if self.x_axis:
                # Assume the trees are simply concatenated end-to-end
                self.x_transform = (
                    lambda x: self.plotbox.left
                    + x / self.ts.sequence_length * self.plotbox.width
                )
                plot_breaks = self.x_transform(breaks)
                dwg = self.drawing

                # For tree sequences, we need to add on the background shaded regions
                self.root_groups["background"] = self.dwg_base.add(
                    dwg.g(class_="background")
                )
                # plotbox_bottom_padding += 10  # extra space for the diagonal lines
                y = self.image_size[1] - self.x_axis_offset
                for i in range(1, len(breaks)):
                    break_x = plot_breaks[i]
                    prev_break_x = plot_breaks[i - 1]
                    tree_x = i * self.tree_plotbox.max_x + self.plotbox.left
                    prev_tree_x = (i - 1) * self.tree_plotbox.max_x + self.plotbox.left
                    # Shift diagonal lines between tree & axis into the treebox a little
                    diag_height = y - (
                        self.plotbox.bottom - self.tree_plotbox.pad_bottom
                    )
                    self.root_groups["background"].add(
                        dwg.path(
                            f"M{rnd(prev_tree_x):g},0 "
                            f"l{rnd(tree_x-prev_tree_x):g},0 "
                            f"l0,{rnd(y - diag_height):g} "
                            f"l{rnd(break_x-tree_x):g},{rnd(diag_height):g} "
                            # NB for curves try "c0,{1} {0},0 {0},{1}" instead of above
                            f"l0,{rnd(tick_length_lower):g} "
                            f"l{rnd(prev_break_x-break_x):g},0 "
                            f"l0,{rnd(-tick_length_lower):g} "
                            f"l{rnd(prev_tree_x-prev_break_x):g},{rnd(-diag_height):g} "
                            # NB for curves try "c0,{1} {0},0 {0},{1}" instead of above
                            f"l0,{rnd(diag_height - y):g}z",
                        )
                    )
            super().draw_x_axis(
                tick_positions=breaks,
                tick_length_lower=tick_length_lower,
                tick_length_upper=tick_length_upper,
                sites=self.ts.sites(),
            )

        else:
            # No background shading needed if x_scale is "treewise"
            n = self.ts.num_trees
            self.x_transform = lambda x: self.plotbox.left + x * self.plotbox.width / n
            super().draw_x_axis(
                tick_positions=np.arange(n + 1),
                tick_labels=self.ts.breakpoints(as_array=True),
                tick_length_lower=tick_length_lower,
            )


class SvgTree(SvgPlot):
    """
    A class to draw a tree in SVG format.

    See :meth:`Tree.draw_svg` for a description of usage and frequently used parameters.
    """

    def __init__(
        self,
        tree,
        size=None,
        max_tree_height=None,
        node_labels=None,
        mutation_labels=None,
        root_svg_attributes=None,
        style=None,
        order=None,
        force_root_branch=None,
        symbol_size=None,
        x_axis=None,
        y_axis=None,
        x_label=None,
        y_label=None,
        y_ticks=None,
        y_gridlines=None,
        tree_height_scale=None,
        node_attrs=None,
        mutation_attrs=None,
        edge_attrs=None,
        node_label_attrs=None,
        mutation_label_attrs=None,
        **kwargs,
    ):
        if size is None:
            size = (200, 200)
        if symbol_size is None:
            symbol_size = 6
        self.symbol_size = symbol_size
        super().__init__(
            tree.tree_sequence,
            size,
            root_svg_attributes,
            style,
            svg_class=f"tree t{tree.index}",
            tree_height_scale=tree_height_scale,
            x_axis=x_axis,
            y_axis=y_axis,
            x_label=x_label,
            y_label=y_label,
            **kwargs,
        )
        self.tree = tree
        self.traversal_order = check_order(order)

        # Create some instance variables for later use in plotting
        self.node_mutations = collections.defaultdict(list)
        self.edge_attrs = {}
        self.node_attrs = {}
        self.node_label_attrs = {}
        self.mutation_attrs = {}
        self.mutation_label_attrs = {}
        self.mutations_over_roots = False
        # mutations collected per node
        nodes = set(tree.nodes())
        unplotted = []
        for site in tree.sites():
            for mutation in site.mutations:
                if mutation.node in nodes:
                    self.node_mutations[mutation.node].append(mutation)
                    if tree.parent(mutation.node) == NULL:
                        self.mutations_over_roots = True
                else:
                    unplotted.append(mutation.id)
        if len(unplotted) > 0:
            logging.warning(
                f"Mutations {unplotted} are above nodes which are not present in the "
                "displayed tree, so are not plotted on the topology."
            )
        # attributes for symbols
        half_symbol_size = "{:g}".format(rnd(symbol_size / 2))
        symbol_size = "{:g}".format(rnd(symbol_size))
        for u in tree.nodes():
            self.edge_attrs[u] = {}
            if edge_attrs is not None and u in edge_attrs:
                self.edge_attrs[u].update(edge_attrs[u])
            if tree.is_sample(u):
                # a square: set bespoke svgwrite params
                self.node_attrs[u] = {
                    "size": (symbol_size,) * 2,
                    "insert": ("-" + half_symbol_size,) * 2,
                }
            else:
                # a circle: set bespoke svgwrite param `centre` and default radius
                self.node_attrs[u] = {"center": (0, 0), "r": half_symbol_size}
            if node_attrs is not None and u in node_attrs:
                self.node_attrs[u].update(node_attrs[u])
            add_class(self.node_attrs[u], "sym")  # class 'sym' for symbol
            label = ""
            if node_labels is None:
                label = str(u)
            elif u in node_labels:
                label = str(node_labels[u])
            self.node_label_attrs[u] = {"text": label}
            add_class(self.node_label_attrs[u], "lab")  # class 'lab' for label
            if node_label_attrs is not None and u in node_label_attrs:
                self.node_label_attrs[u].update(node_label_attrs[u])
        for site in tree.sites():
            for mutation in site.mutations:
                m = mutation.id
                # We need to offset the rectangle so that it's centred
                self.mutation_attrs[m] = {
                    "d": "M -{0},-{0} l {1},{1} M -{0},{0} l {1},-{1}".format(
                        half_symbol_size, symbol_size
                    )
                }
                if mutation_attrs is not None and m in mutation_attrs:
                    self.mutation_attrs[m].update(mutation_attrs[m])
                add_class(self.mutation_attrs[m], "sym")  # class 'sym' for symbol
                label = ""
                if mutation_labels is None:
                    label = str(m)
                elif mutation.id in mutation_labels:
                    label = str(mutation_labels[m])
                self.mutation_label_attrs[m] = {"text": label}
                if mutation_label_attrs is not None and m in mutation_label_attrs:
                    self.mutation_label_attrs[m].update(mutation_label_attrs[m])
                add_class(self.mutation_label_attrs[m], "lab")

        self.set_spacing(top=10, left=20, bottom=10, right=20)
        self.assign_y_coordinates(max_tree_height, force_root_branch)
        self.assign_x_coordinates()
        self.draw_x_axis(
            tick_positions=np.array(tree.interval),
            tick_length_lower=self.default_tick_length,  # TODO - parameterize
            tick_length_upper=self.default_tick_length_site,  # TODO - parameterize
            sites=tree.sites(),
        )
        self.draw_y_axis(
            lower=self.y_transform(0),
            tick_positions=check_ticks(y_ticks, set(self.node_height.values())),
            tick_length_left=self.default_tick_length,
            gridlines=y_gridlines,
        )
        self.draw_tree()

    def process_mutations_over_node(self, u, low_bound, high_bound, ignore_times=False):
        """
        Sort the self.node_mutations array for a given node ``u`` in reverse time order.
        The main complication is with UNKNOWN_TIME values: we replace these with times
        spaced between the low & high bounds (this is always done if ignore_times=True).
        We do not currently allow a mix of known & unknown mutation times in a tree
        sequence, which makes the logic easy. If we were to allow it, more complex
        logic can be neatly encapsulated in this method.
        """
        mutations = self.node_mutations[u]
        time_unknown = [util.is_unknown_time(m.time) for m in mutations]
        if all(time_unknown) or ignore_times is True:
            # sort by site then within site by parent: will end up with oldest first
            mutations.sort(key=operator.attrgetter("site", "parent"))
            diff = high_bound - low_bound
            for i in range(len(mutations)):
                mutations[i].time = high_bound - diff * (i + 1) / (len(mutations) + 1)
        else:
            assert not any(time_unknown)
            mutations.sort(key=operator.attrgetter("time"), reverse=True)

    def assign_y_coordinates(
        self,
        max_tree_height,
        force_root_branch,
        bottom_space=SvgPlot.line_height,
        top_space=SvgPlot.line_height,
    ):
        """
        Create a self.node_height dict, a self.y_transform func and
        self.min_root_branch_plot_length for use in plotting. Allow extra space within
        the plotbox, at the bottom for leaf labels, and  (potentially, if no root
        branches are plotted) above the topmost root node for root labels.
        """
        max_tree_height = check_max_tree_height(
            max_tree_height, self.tree_height_scale != "rank"
        )
        node_time = self.ts.tables.nodes.time
        mut_time = self.ts.tables.mutations.time
        root_branch_length = 0
        if self.tree_height_scale == "rank":
            if max_tree_height == "tree":
                # We only rank the times within the tree in this case.
                t = np.zeros_like(node_time)
                for u in self.tree.nodes():
                    t[u] = node_time[u]
                node_time = t
            times = np.unique(node_time[node_time <= self.ts.max_root_time])
            max_node_height = len(times)
            depth = {t: j for j, t in enumerate(times)}
            if self.mutations_over_roots or force_root_branch:
                root_branch_length = 1  # Will get scaled later
            max_tree_height = max(depth.values()) + root_branch_length
            # In pathological cases, all the roots are at 0
            if max_tree_height == 0:
                max_tree_height = 1
            self.node_height = {u: depth[node_time[u]] for u in self.tree.nodes()}
            for u in self.node_mutations.keys():
                parent = self.tree.parent(u)
                if parent == NULL:
                    top = self.node_height[u] + root_branch_length
                else:
                    top = self.node_height[parent]
                self.process_mutations_over_node(
                    u, self.node_height[u], top, ignore_times=True
                )
        else:
            assert self.tree_height_scale in ["time", "log_time"]
            self.node_height = {u: node_time[u] for u in self.tree.nodes()}
            if max_tree_height == "tree":
                max_node_height = max(self.node_height.values())
                max_mut_height = np.nanmax(
                    [0] + [mut.time for m in self.node_mutations.values() for mut in m]
                )
            else:
                max_node_height = self.ts.max_root_time
                max_mut_height = np.nanmax(np.append(mut_time, 0))
            max_tree_height = max(max_node_height, max_mut_height)  # Reuse variable
            # In pathological cases, all the roots are at 0
            if max_tree_height == 0:
                max_tree_height = 1

            if self.mutations_over_roots or force_root_branch:
                # Define a minimum root branch length, after transformation if necessary
                if self.tree_height_scale != "log_time":
                    root_branch_length = max_tree_height * self.root_branch_fraction
                else:
                    log_height = np.log(max_tree_height + 1)
                    root_branch_length = (
                        np.exp(log_height * (1 + self.root_branch_fraction))
                        - 1
                        - max_tree_height
                    )
                # If necessary, allow for this extra branch in max_tree_height
                if max_node_height + root_branch_length > max_tree_height:
                    max_tree_height = max_node_height + root_branch_length
            for u in self.node_mutations.keys():
                parent = self.tree.parent(u)
                if parent == NULL:
                    # This is a root: if muts have no times we must specify an upper time
                    top = self.node_height[u] + root_branch_length
                else:
                    top = self.node_height[parent]
                self.process_mutations_over_node(u, self.node_height[u], top)

        assert float(max_tree_height) == max_tree_height

        # Add extra space above the top and below the bottom of the tree to keep the
        # node labels within the plotbox (but top label space not needed if the
        # existence of a root branch pushes the whole tree + labels downwards anyway)
        top_space = 0 if root_branch_length > 0 else top_space
        zero_pos = self.plotbox.height + self.plotbox.top - bottom_space
        padding_numerator = self.plotbox.height - top_space - bottom_space
        if padding_numerator < 0:
            raise ValueError("Image size too small to allow space to plot tree")
        # Transform the y values into plot space (inverted y with 0 at the top of screen)
        if self.tree_height_scale == "log_time":
            # add 1 so that don't reach log(0) = -inf error.
            # just shifts entire timeset by 1 unit so shouldn't affect anything
            y_scale = padding_numerator / np.log(max_tree_height + 1)
            self.y_transform = lambda y: zero_pos - np.log(y + 1) * y_scale
        else:
            y_scale = padding_numerator / max_tree_height
            self.y_transform = lambda y: zero_pos - y * y_scale

        # Calculate default root branch length to use (in plot coords). This is a
        # minimum, as branches with deep root mutations could be longer
        self.min_root_branch_plot_length = self.y_transform(
            max_tree_height
        ) - self.y_transform(max_tree_height + root_branch_length)

    def assign_x_coordinates(self):
        num_leaves = len(list(self.tree.leaves()))
        x_scale = self.plotbox.width / num_leaves
        node_x_coord_map = {}
        leaf_x = self.plotbox.left + x_scale / 2
        for root in self.tree.roots:
            for u in self.tree.nodes(root, order=self.traversal_order):
                if self.tree.is_leaf(u):
                    node_x_coord_map[u] = leaf_x
                    leaf_x += x_scale
                else:
                    child_coords = [node_x_coord_map[c] for c in self.tree.children(u)]
                    if len(child_coords) == 1:
                        node_x_coord_map[u] = child_coords[0]
                    else:
                        a = min(child_coords)
                        b = max(child_coords)
                        node_x_coord_map[u] = a + (b - a) / 2
        self.node_x_coord_map = node_x_coord_map
        # Transform is not for nodes but for genome positions
        self.x_transform = lambda x: (
            (x - self.tree.interval.left) / self.tree.interval.span * self.plotbox.width
            + self.plotbox.left
        )

    def info_classes(self, focal_node_id):
        """
        For a focal node id, return a set of classes that encode this useful information:
            "a<X>" or "root": where <X> == id of immediate ancestor (parent) node
            "n<Y>":           where <Y> == focal node id
            "m<A>":           where <A> == mutation id
            "s<B>":           where <B> == site id of all mutations
        """
        # Add a new group for each node, and give it classes for css targetting
        focal_node = self.ts.node(focal_node_id)
        classes = set()
        classes.add(f"node n{focal_node_id}")
        if focal_node.individual != NULL:
            classes.add(f"i{focal_node.individual}")
        if focal_node.population != NULL:
            classes.add(f"p{focal_node.population}")
        v = self.tree.parent(focal_node_id)
        if v == NULL:
            classes.add("root")
        else:
            classes.add(f"a{v}")
        if self.tree.is_sample(focal_node_id):
            classes.add("sample")
        if self.tree.is_leaf(focal_node_id):
            classes.add("leaf")
        for mutation in self.node_mutations[focal_node_id]:
            # Adding mutations and sites above this node allows identification
            # of the tree under any specific mutation
            classes.add(f"m{mutation.id}")
            classes.add(f"s{mutation.site}")
        return sorted(classes)

    def draw_tree(self):
        dwg = self.drawing
        node_x_coord_map = self.node_x_coord_map
        node_y_coord_map = {u: self.y_transform(h) for u, h in self.node_height.items()}
        tree = self.tree
        left_child = get_left_child(tree, self.traversal_order)

        # Iterate over nodes, adding groups to reflect the tree heirarchy
        stack = []
        for u in tree.roots:
            grp = dwg.g(
                class_=" ".join(self.info_classes(u)),
                transform=f"translate({rnd(node_x_coord_map[u])} "
                f"{rnd(node_y_coord_map[u])})",
            )
            stack.append((u, self.get_plotbox().add(grp)))
        while len(stack) > 0:
            u, curr_svg_group = stack.pop()
            pu = node_x_coord_map[u], node_y_coord_map[u]
            for focal in tree.children(u):
                fx = node_x_coord_map[focal] - pu[0]
                fy = node_y_coord_map[focal] - pu[1]
                new_svg_group = curr_svg_group.add(
                    dwg.g(
                        class_=" ".join(self.info_classes(focal)),
                        transform=f"translate({rnd(fx)} {rnd(fy)})",
                    )
                )
                stack.append((focal, new_svg_group))

            o = (0, 0)
            v = tree.parent(u)

            # Add edge first => on layer underneath anything else
            if v != NULL:
                add_class(self.edge_attrs[u], "edge")
                pv = node_x_coord_map[v], node_y_coord_map[v]
                dx = pv[0] - pu[0]
                dy = pv[1] - pu[1]
                path = dwg.path(
                    [("M", o), ("V", rnd(dy)), ("H", rnd(dx))], **self.edge_attrs[u]
                )
                curr_svg_group.add(path)
            else:
                root_branch_l = self.min_root_branch_plot_length
                if root_branch_l > 0:
                    add_class(self.edge_attrs[u], "edge")
                    if len(self.node_mutations[u]) > 0:
                        mutation = self.node_mutations[u][0]  # Oldest on this branch
                        root_branch_l = max(
                            root_branch_l,
                            node_y_coord_map[u] - self.y_transform(mutation.time),
                        )
                    path = dwg.path(
                        [("M", o), ("V", rnd(-root_branch_l)), ("H", 0)],
                        **self.edge_attrs[u],
                    )
                    curr_svg_group.add(path)
                pv = (pu[0], pu[1] - root_branch_l)

            # Add mutation symbols + labels
            for mutation in self.node_mutations[u]:
                # TODO get rid of these manual positioning tweaks and add them
                # as offsets the user can access via a transform or something.
                dy = self.y_transform(mutation.time) - pu[1]
                mutation_class = f"mut m{mutation.id} s{mutation.site}"
                if util.is_unknown_time(self.ts.mutation(mutation.id).time):
                    mutation_class += " unknown_time"
                mut_group = curr_svg_group.add(
                    dwg.g(class_=mutation_class, transform=f"translate(0 {rnd(dy)})")
                )
                # A line from the mutation to the node below, normally hidden, but
                # revealable if we want to flag the path below a mutation
                mut_group.add(dwg.line(end=(0, -rnd(dy))))
                # Symbols
                mut_group.add(dwg.path(**self.mutation_attrs[mutation.id]))
                # Labels
                if u == left_child[tree.parent(u)]:
                    mut_label_class = "lft"
                    transform = f"translate(-{rnd(2+self.symbol_size/2)} 0)"
                else:
                    mut_label_class = "rgt"
                    transform = f"translate({rnd(2+self.symbol_size/2)} 0)"
                add_class(self.mutation_label_attrs[mutation.id], mut_label_class)
                self.mutation_label_attrs[mutation.id]["transform"] = transform
                mut_group.add(dwg.text(**self.mutation_label_attrs[mutation.id]))

            # Add node symbol + label next (visually above the edge subtending this node)
            # Symbols
            if self.tree.is_sample(u):
                curr_svg_group.add(dwg.rect(**self.node_attrs[u]))
            else:
                curr_svg_group.add(dwg.circle(**self.node_attrs[u]))
            # Labels
            node_lab_attr = self.node_label_attrs[u]
            if tree.is_leaf(u):
                node_lab_attr["transform"] = f"translate(0 {self.text_height - 3})"
            elif tree.parent(u) == NULL and self.min_root_branch_plot_length == 0:
                node_lab_attr["transform"] = f"translate(0 -{self.text_height - 3})"
            else:
                if u == left_child[tree.parent(u)]:
                    add_class(node_lab_attr, "lft")
                    node_lab_attr["transform"] = f"translate(-3 -{self.text_height/2})"
                else:
                    add_class(node_lab_attr, "rgt")
                    node_lab_attr["transform"] = f"translate(3 -{self.text_height/2})"
            curr_svg_group.add(dwg.text(**node_lab_attr))


class TextTreeSequence:
    """
    Draw a tree sequence as horizontal line of trees.
    """

    def __init__(
        self,
        ts,
        node_labels=None,
        use_ascii=False,
        time_label_format=None,
        position_label_format=None,
        order=None,
    ):
        self.ts = ts

        time_label_format = "{:.2f}" if time_label_format is None else time_label_format
        position_label_format = (
            "{:.2f}" if position_label_format is None else position_label_format
        )

        time = ts.tables.nodes.time
        time_scale_labels = [
            time_label_format.format(time[u]) for u in range(ts.num_nodes)
        ]
        position_scale_labels = [
            position_label_format.format(x) for x in ts.breakpoints()
        ]
        trees = [
            VerticalTextTree(
                tree,
                max_tree_height="ts",
                node_labels=node_labels,
                use_ascii=use_ascii,
                order=order,
            )
            for tree in self.ts.trees()
        ]

        self.height = 1 + max(tree.height for tree in trees)
        self.width = sum(tree.width + 2 for tree in trees) - 1
        max_time_scale_label_len = max(map(len, time_scale_labels))
        self.width += 3 + max_time_scale_label_len + len(position_scale_labels[-1]) // 2

        self.canvas = np.zeros((self.height, self.width), dtype=str)
        self.canvas[:] = " "

        vertical_sep = "|" if use_ascii else "┊"
        x = 0
        time_position = trees[0].time_position
        for u, label in enumerate(map(to_np_unicode, time_scale_labels)):
            y = time_position[u]
            self.canvas[y, 0 : label.shape[0]] = label
        self.canvas[:, max_time_scale_label_len] = vertical_sep
        x = 2 + max_time_scale_label_len

        for j, tree in enumerate(trees):
            pos_label = to_np_unicode(position_scale_labels[j])
            k = len(pos_label)
            label_x = max(x - k // 2 - 2, 0)
            self.canvas[-1, label_x : label_x + k] = pos_label
            h, w = tree.canvas.shape
            self.canvas[-h - 1 : -1, x : x + w - 1] = tree.canvas[:, :-1]
            x += w
            self.canvas[:, x] = vertical_sep
            x += 2

        pos_label = to_np_unicode(position_scale_labels[-1])
        k = len(pos_label)
        label_x = max(x - k // 2 - 2, 0)
        self.canvas[-1, label_x : label_x + k] = pos_label
        self.canvas[:, -1] = "\n"

    def __str__(self):
        return "".join(self.canvas.reshape(self.width * self.height))


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


def get_left_neighbour(tree, traversal_order):
    """
    Returns the left-most neighbour of each node in the tree according to the
    specified traversal order. The left neighbour is the closest node in terms
    of path distance to the left of a given node.
    """
    # The traversal order will define the order of children and roots.
    # Root order is defined by this traversal, and the roots are
    # the children of -1
    children = collections.defaultdict(list)
    for u in tree.nodes(order=traversal_order):
        children[tree.parent(u)].append(u)

    left_neighbour = np.full(tree.num_nodes + 1, NULL, dtype=int)

    def find_neighbours(u, neighbour):
        left_neighbour[u] = neighbour
        for v in children[u]:
            find_neighbours(v, neighbour)
            neighbour = v

    # The children of -1 are the roots and the neighbour of all left-most
    # nodes in the tree is also -1 (NULL)
    find_neighbours(-1, -1)

    return left_neighbour[:-1]


def get_left_child(tree, traversal_order):
    """
    Returns the left-most child of each node in the tree according to the
    specified traversal order. If a node has no children or NULL is passed
    in, return NULL.
    """
    left_child = np.full(tree.num_nodes + 1, NULL, dtype=int)
    for u in tree.nodes(order=traversal_order):
        parent = tree.parent(u)
        if parent != NULL and left_child[parent] == NULL:
            left_child[parent] = u
    return left_child


def node_time_depth(tree, min_branch_length=None, max_tree_height="tree"):
    """
    Returns a dictionary mapping nodes in the specified tree to their depth
    in the specified tree (from the root direction). If min_branch_len is
    provided, it specifies the minimum length of each branch. If not specified,
    default to 1.
    """
    if min_branch_length is None:
        min_branch_length = {u: 1 for u in range(tree.tree_sequence.num_nodes)}
    time_node_map = collections.defaultdict(list)
    current_depth = 0
    depth = {}
    # TODO this is basically the same code for the two cases. Refactor so that
    # we use the same code.
    if max_tree_height == "tree":
        for u in tree.nodes():
            time_node_map[tree.time(u)].append(u)
        for t in sorted(time_node_map.keys()):
            for u in time_node_map[t]:
                for v in tree.children(u):
                    current_depth = max(current_depth, depth[v] + min_branch_length[v])
            for u in time_node_map[t]:
                depth[u] = current_depth
            current_depth += 2
        for root in tree.roots:
            current_depth = max(current_depth, depth[root] + min_branch_length[root])
    else:
        assert max_tree_height == "ts"
        ts = tree.tree_sequence
        for node in ts.nodes():
            time_node_map[node.time].append(node.id)
        node_edges = collections.defaultdict(list)
        for edge in ts.edges():
            node_edges[edge.parent].append(edge)

        for t in sorted(time_node_map.keys()):
            for u in time_node_map[t]:
                for edge in node_edges[u]:
                    v = edge.child
                    current_depth = max(current_depth, depth[v] + min_branch_length[v])
            for u in time_node_map[t]:
                depth[u] = current_depth
            current_depth += 2

    return depth, current_depth


class TextTree:
    """
    Draws a reprentation of a tree using unicode drawing characters written
    to a 2D array.
    """

    def __init__(
        self,
        tree,
        node_labels=None,
        max_tree_height=None,
        use_ascii=False,
        orientation=None,
        order=None,
    ):
        self.tree = tree
        self.traversal_order = check_order(order)
        self.max_tree_height = check_max_tree_height(
            max_tree_height, allow_numeric=False
        )
        self.use_ascii = use_ascii
        self.orientation = check_orientation(orientation)
        self.horizontal_line_char = "━"
        self.vertical_line_char = "┃"
        if use_ascii:
            self.horizontal_line_char = "-"
            self.vertical_line_char = "|"
        # These are set below by the placement algorithms.
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

        # Set the node labels
        for u in tree.nodes():
            if node_labels is None:
                # If we don't specify node_labels, default to node ID
                self.node_labels[u] = str(u)
            else:
                # If we do specify node_labels, default to an empty line
                self.node_labels[u] = self.default_node_label
        if node_labels is not None:
            for node, label in node_labels.items():
                self.node_labels[node] = label

        self._assign_time_positions()
        self._assign_traversal_positions()
        self.canvas = np.zeros((self.height, self.width), dtype=str)
        self.canvas[:] = " "
        self._draw()
        self.canvas[:, -1] = "\n"

    def __str__(self):
        return "".join(self.canvas.reshape(self.width * self.height))


class VerticalTextTree(TextTree):
    """
    Text tree rendering where root nodes are at the top and time goes downwards
    into the present.
    """

    @property
    def default_node_label(self):
        return self.vertical_line_char

    def _assign_time_positions(self):
        tree = self.tree
        # TODO when we add mutations to the text tree we'll need to take it into
        # account here. Presumably we need to get the maximum number of mutations
        # per branch.
        self.time_position, total_depth = node_time_depth(
            tree, max_tree_height=self.max_tree_height
        )
        self.height = total_depth - 1

    def _assign_traversal_positions(self):
        self.label_x = {}
        left_neighbour = get_left_neighbour(self.tree, self.traversal_order)
        x = 0
        for u in self.tree.nodes(order=self.traversal_order):
            label_size = len(self.node_labels[u])
            if self.tree.is_leaf(u):
                self.traversal_position[u] = x + label_size // 2
                self.label_x[u] = x
                x += label_size + 1
            else:
                coords = [self.traversal_position[c] for c in self.tree.children(u)]
                if len(coords) == 1:
                    self.traversal_position[u] = coords[0]
                else:
                    a = min(coords)
                    b = max(coords)
                    child_mid = int(round(a + (b - a) / 2))
                    self.traversal_position[u] = child_mid
                self.label_x[u] = self.traversal_position[u] - label_size // 2
                neighbour_x = -1
                neighbour = left_neighbour[u]
                if neighbour != NULL:
                    neighbour_x = self.traversal_position[neighbour]
                self.label_x[u] = max(neighbour_x + 1, self.label_x[u])
                x = max(x, self.label_x[u] + label_size + 1)
            assert self.label_x[u] >= 0
        self.width = x

    def _draw(self):
        if self.use_ascii:
            left_child = "+"
            right_child = "+"
            mid_parent = "+"
            mid_parent_child = "+"
            mid_child = "+"
        elif self.orientation == TOP:
            left_child = "┏"
            right_child = "┓"
            mid_parent = "┻"
            mid_parent_child = "╋"
            mid_child = "┳"
        else:
            left_child = "┗"
            right_child = "┛"
            mid_parent = "┳"
            mid_parent_child = "╋"
            mid_child = "┻"

        for u in self.tree.nodes():
            xu = self.traversal_position[u]
            yu = self.time_position[u]
            label = to_np_unicode(self.node_labels[u])
            label_len = label.shape[0]
            label_x = self.label_x[u]
            assert label_x >= 0
            self.canvas[yu, label_x : label_x + label_len] = label
            children = self.tree.children(u)
            if len(children) > 0:
                if len(children) == 1:
                    yv = self.time_position[children[0]]
                    self.canvas[yv:yu, xu] = self.vertical_line_char
                else:
                    left = min(self.traversal_position[v] for v in children)
                    right = max(self.traversal_position[v] for v in children)
                    y = yu - 1
                    self.canvas[y, left + 1 : right] = self.horizontal_line_char
                    self.canvas[y, xu] = mid_parent
                    for v in children:
                        xv = self.traversal_position[v]
                        yv = self.time_position[v]
                        self.canvas[yv:yu, xv] = self.vertical_line_char
                        mid_char = mid_parent_child if xv == xu else mid_child
                        self.canvas[y, xv] = mid_char
                    self.canvas[y, left] = left_child
                    self.canvas[y, right] = right_child
        # print(self.canvas)
        if self.orientation == TOP:
            self.canvas = np.flip(self.canvas, axis=0)
            # Reverse the time positions so that we can use them in the tree
            # sequence drawing as well.
            flipped_time_position = {
                u: self.height - y - 1 for u, y in self.time_position.items()
            }
            self.time_position = flipped_time_position


class HorizontalTextTree(TextTree):
    """
    Text tree rendering where root nodes are at the left and time goes
    rightwards into the present.
    """

    @property
    def default_node_label(self):
        return self.horizontal_line_char

    def _assign_time_positions(self):
        # TODO when we add mutations to the text tree we'll need to take it into
        # account here. Presumably we need to get the maximum number of mutations
        # per branch.
        self.time_position, total_depth = node_time_depth(
            self.tree, {u: 1 + len(self.node_labels[u]) for u in self.tree.nodes()}
        )
        self.width = total_depth

    def _assign_traversal_positions(self):
        y = 0
        for root in self.tree.roots:
            for u in self.tree.nodes(root, order=self.traversal_order):
                if self.tree.is_leaf(u):
                    self.traversal_position[u] = y
                    y += 2
                else:
                    coords = [self.traversal_position[c] for c in self.tree.children(u)]
                    if len(coords) == 1:
                        self.traversal_position[u] = coords[0]
                    else:
                        a = min(coords)
                        b = max(coords)
                        child_mid = int(round(a + (b - a) / 2))
                        self.traversal_position[u] = child_mid
            y += 1
        self.height = y - 2

    def _draw(self):
        if self.use_ascii:
            top_across = "+"
            bot_across = "+"
            mid_parent = "+"
            mid_parent_child = "+"
            mid_child = "+"
        elif self.orientation == LEFT:
            top_across = "┏"
            bot_across = "┗"
            mid_parent = "┫"
            mid_parent_child = "╋"
            mid_child = "┣"
        else:
            top_across = "┓"
            bot_across = "┛"
            mid_parent = "┣"
            mid_parent_child = "╋"
            mid_child = "┫"

        # Draw in root-right mode as the coordinates go in the expected direction.
        for u in self.tree.nodes():
            yu = self.traversal_position[u]
            xu = self.time_position[u]
            label = to_np_unicode(self.node_labels[u])
            if self.orientation == LEFT:
                # We flip the array at the end so need to reverse the label.
                label = label[::-1]
            label_len = label.shape[0]
            self.canvas[yu, xu : xu + label_len] = label
            children = self.tree.children(u)
            if len(children) > 0:
                if len(children) == 1:
                    xv = self.time_position[children[0]]
                    self.canvas[yu, xv:xu] = self.horizontal_line_char
                else:
                    bot = min(self.traversal_position[v] for v in children)
                    top = max(self.traversal_position[v] for v in children)
                    x = xu - 1
                    self.canvas[bot + 1 : top, x] = self.vertical_line_char
                    self.canvas[yu, x] = mid_parent
                    for v in children:
                        yv = self.traversal_position[v]
                        xv = self.time_position[v]
                        self.canvas[yv, xv:x] = self.horizontal_line_char
                        mid_char = mid_parent_child if yv == yu else mid_child
                        self.canvas[yv, x] = mid_char
                    self.canvas[bot, x] = top_across
                    self.canvas[top, x] = bot_across
        if self.orientation == LEFT:
            self.canvas = np.flip(self.canvas, axis=1)
            # Move the padding to the left.
            self.canvas[:, :-1] = self.canvas[:, 1:]
            self.canvas[:, -1] = " "
        # print(self.canvas)
