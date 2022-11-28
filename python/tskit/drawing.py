# MIT License
#
# Copyright (c) 2018-2022 Tskit Developers
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
import warnings
from dataclasses import dataclass
from typing import List
from typing import Mapping
from typing import Union

import numpy as np
import svgwrite

import tskit
import tskit.util as util
from _tskit import NODE_IS_SAMPLE
from _tskit import NULL

LEFT = "left"
RIGHT = "right"
TOP = "top"
BOTTOM = "bottom"

# constants for whether to plot a tree in a tree sequence
OMIT = 1
LEFT_CLIP = 2
RIGHT_CLIP = 4
OMIT_MIDDLE = 8


@dataclass
class Offsets:
    "Used when x_lim set, and displayed ts has been cut down by keep_intervals"
    tree: int = 0
    site: int = 0
    mutation: int = 0


@dataclass(frozen=True)
class Timescaling:
    "Class used to transform the time axis"
    max_time: float
    min_time: float
    plot_min: float
    plot_range: float
    use_log_transform: bool

    def __post_init__(self):
        if self.plot_range < 0:
            raise ValueError("Image size too small to allow space to plot tree")
        if self.use_log_transform:
            if self.min_time < 0:
                raise ValueError("Cannot use a log scale if there are negative times")
            super().__setattr__("transform", self.log_transform)
        else:
            super().__setattr__("transform", self.linear_transform)

    def log_transform(self, y):
        "Standard log transform but allowing for values of 0 by adding 1"
        delta = 1 if self.min_time == 0 else 0
        log_max = np.log(self.max_time + delta)
        log_min = np.log(self.min_time + delta)
        y_scale = self.plot_range / (log_max - log_min)
        return self.plot_min - (np.log(y + delta) - log_min) * y_scale

    def linear_transform(self, y):
        y_scale = self.plot_range / (self.max_time - self.min_time)
        return self.plot_min - (y - self.min_time) * y_scale


class SVGString(str):
    "A string containing an SVG representation"

    def _repr_svg_(self):
        """
        Simply return the SVG string: called by jupyter notebooks to render trees.
        """
        return self


def check_orientation(orientation):
    if orientation is None:
        orientation = TOP
    else:
        orientation = orientation.lower()
        orientations = [LEFT, RIGHT, TOP, BOTTOM]
        if orientation not in orientations:
            raise ValueError(f"Unknown orientiation: choose from {orientations}")
    return orientation


def check_max_time(max_time, allow_numeric=True):
    if max_time is None:
        max_time = "tree"
    is_numeric = isinstance(max_time, numbers.Real)
    if max_time not in ["tree", "ts"] and not allow_numeric:
        raise ValueError("max_time must be 'tree' or 'ts'")
    if max_time not in ["tree", "ts"] and (allow_numeric and not is_numeric):
        raise ValueError("max_time must be a numeric value or one of 'tree' or 'ts'")
    return max_time


def check_min_time(min_time, allow_numeric=True):
    if min_time is None:
        min_time = "tree"
    if allow_numeric:
        is_numeric = isinstance(min_time, numbers.Real)
        if min_time not in ["tree", "ts"] and not is_numeric:
            raise ValueError(
                "min_time must be a numeric value or one of 'tree' or 'ts'"
            )
    else:
        if min_time not in ["tree", "ts"]:
            raise ValueError("min_time must be 'tree' or 'ts'")
    return min_time


def check_time_scale(time_scale):
    if time_scale is None:
        time_scale = "time"
    if time_scale not in ["time", "log_time", "rank"]:
        raise ValueError("time_scale must be 'time', 'log_time' or 'rank'")
    return time_scale


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


def check_x_lim(x_lim, max_x):
    """
    Checks the specified x_limits are valid and sets default if None.
    """
    if x_lim is None:
        x_lim = (None, None)
    if len(x_lim) != 2:
        raise ValueError("The x_lim parameter must be a list of length 2, or None")
    try:
        if x_lim[0] is not None and x_lim[0] < 0:
            raise ValueError("x_lim[0] cannot be negative")
        if x_lim[1] is not None and x_lim[1] > max_x:
            raise ValueError("x_lim[1] cannot be greater than the sequence length")
        if x_lim[0] is not None and x_lim[1] is not None and x_lim[0] >= x_lim[1]:
            raise ValueError("x_lim[0] must be less than x_lim[1]")
    except TypeError:
        raise TypeError("x_lim parameters must be numeric")
    return x_lim


def create_tick_labels(tick_values, decimal_places=2):
    """
    If tick_values are numeric, round the labels to X decimal_places, but do not print
    decimals if all values are integers
    """
    try:
        integer_ticks = np.all(np.round(tick_values) == tick_values)
    except TypeError:
        return tick_values
    label_precision = 0 if integer_ticks else decimal_places
    return [f"{lab:.{label_precision}f}" for lab in tick_values]


def clip_ts(ts, x_min, x_max, max_num_trees=None):
    """
    Culls the edges of the tree sequence outside the limits of x_min and x_max if
    necessary, and flags internal trees for omission if there are more than
    max_num_trees in the tree sequence

    Returns the new tree sequence using the same genomic scale, and an
    array specifying which trees to actually plot from it. This array contains
    information about whether a plotted tree was clipped, because clipping can
    cause the rightmost and leftmost tree in this new TS to have reduced spans, and
    should be displayed by omitting the appropriate breakpoint.

    If x_min is None, we take it to be 0 if the first tree has edges or sites, or
    ``min(edges.left)`` if the first tree represents an empty region.
    Similarly, if x_max is None we take it to be ``ts.sequence_length`` if the last tree
    has edges or mutations, or ``ts.last().interval.left`` if the last tree represents
    an empty region.

    To plot the full ts, including empty flanking regions, specify x_limits of
    [0, seq_len].

    """
    edges = ts.tables.edges
    sites = ts.tables.sites
    offsets = Offsets()
    if x_min is None:
        if ts.num_edges == 0:
            if ts.num_sites == 0:
                raise ValueError(
                    "To plot an empty tree sequence, specify x_lim=[0, sequence_length]"
                )
            x_min = 0
        else:
            x_min = np.min(edges.left)
            if ts.num_sites > 0 and np.min(sites.position) < x_min:
                x_min = 0  # First region has no edges, but does have sites => keep
    if x_max is None:
        if ts.num_edges == 0:
            if ts.num_sites == 0:
                raise ValueError(
                    "To plot an empty tree sequence, specify x_lim=[0, sequence_length]"
                )
            x_max = ts.sequence_length
        else:
            x_max = np.max(edges.right)
            if ts.num_sites > 0 and np.max(sites.position) > x_max:
                x_max = ts.sequence_length  # Last region has sites but no edges => keep

    if max_num_trees is None:
        max_num_trees = np.inf

    if max_num_trees < 2:
        raise ValueError("Must show at least 2 trees when clipping a tree sequence")

    if (x_min > 0) or (x_max < ts.sequence_length):
        old_breaks = ts.breakpoints(as_array=True)
        offsets.tree = np.searchsorted(old_breaks, x_min, "right") - 2
        offsets.site = np.searchsorted(sites.position, x_min)
        offsets.mutation = np.searchsorted(ts.tables.mutations.site, offsets.site)
        ts = ts.keep_intervals([[x_min, x_max]], simplify=False)
        if ts.num_edges == 0:
            raise ValueError(
                f"Can't limit plotting from {x_min} to {x_max} as whole region is empty"
            )
        edges = ts.tables.edges
        sites = ts.tables.sites
        trees_start = np.min(edges.left)
        trees_end = np.max(edges.right)
        tree_status = np.zeros(ts.num_trees, dtype=np.uint8)
        # Are the leftmost/rightmost regions completely empty - if so, don't plot them
        if 0 < x_min <= trees_start and (
            ts.num_sites == 0 or trees_start <= np.min(sites.position)
        ):
            tree_status[0] = OMIT
        if trees_end <= x_max < ts.sequence_length and (
            ts.num_sites == 0 or trees_end >= np.max(sites.position)
        ):
            tree_status[-1] = OMIT

        # Which breakpoints are new ones, as a result of clipping
        new_breaks = np.logical_not(np.isin(ts.breakpoints(as_array=True), old_breaks))
        tree_status[new_breaks[:-1]] |= LEFT_CLIP
        tree_status[new_breaks[1:]] |= RIGHT_CLIP
    else:
        tree_status = np.zeros(ts.num_trees, dtype=np.uint8)

    first_tree = 1 if tree_status[0] & OMIT else 0
    last_tree = ts.num_trees - 2 if tree_status[-1] & OMIT else ts.num_trees - 1
    num_shown_trees = last_tree - first_tree + 1
    if num_shown_trees > max_num_trees:
        num_start_trees = max_num_trees // 2 + (1 if max_num_trees % 2 else 0)
        num_end_trees = max_num_trees // 2
        assert num_start_trees + num_end_trees == max_num_trees
        tree_status[
            (first_tree + num_start_trees) : (last_tree - num_end_trees + 1)
        ] = (OMIT | OMIT_MIDDLE)

    return ts, tree_status, offsets


def check_y_ticks(ticks: Union[List, Mapping, None]) -> Mapping:
    """
    Later we might want to implement a tick locator function, such that e.g. ticks=5
    selects ~5 nicely spaced tick locations (with sensible behaviour for log scales)
    """
    if ticks is None:
        return {}
    if isinstance(ticks, Mapping):
        return dict(zip(ticks, create_tick_labels(list(ticks.values()))))
    return dict(zip(ticks, create_tick_labels(ticks)))


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


def edge_and_sample_nodes(ts, omit_regions=None):
    """
    Return ids of nodes which are mentioned in an edge in this tree sequence or which
    are samples: nodes not connected to an edge are often found if x_lim is specified.
    """
    if omit_regions is None or len(omit_regions) == 0:
        ids = np.concatenate((ts.edges_child, ts.edges_parent))
    else:
        ids = np.array([], dtype=ts.edges_child.dtype)
        edges = ts.tables.edges
        assert omit_regions.shape[1] == 2
        omit_regions = omit_regions.flatten()
        assert np.all(omit_regions == np.unique(omit_regions))  # Check they're in order
        use_regions = np.concatenate(([0.0], omit_regions, [ts.sequence_length]))
        use_regions = use_regions.reshape(-1, 2)
        for left, right in use_regions:
            used_edges = edges[np.logical_and(edges.left >= left, edges.right < right)]
            ids = np.concatenate((ids, used_edges.child, used_edges.parent))
    return np.unique(
        np.concatenate((ids, np.where(ts.nodes_flags & NODE_IS_SAMPLE)[0]))
    )


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
    time_scale=None,
    tree_height_scale=None,
    max_time=None,
    min_time=None,
    max_tree_height=None,
    order=None,
    omit_sites=None,
):
    if time_scale is None and tree_height_scale is not None:
        time_scale = tree_height_scale
        # Deprecated in 0.3.6
        warnings.warn(
            "tree_height_scale is deprecated; use time_scale instead",
            FutureWarning,
        )
    if max_time is None and max_tree_height is not None:
        max_time = max_tree_height
        # Deprecated in 0.3.6
        warnings.warn(
            "max_tree_height is deprecated; use max_time instead",
            FutureWarning,
        )

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
            time_scale=time_scale,
            max_time=max_time,
            min_time=min_time,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            node_label_attrs=node_label_attrs,
            mutation_attrs=mutation_attrs,
            order=order,
            omit_sites=omit_sites,
        )
        return SVGString(tree.drawing.tostring())

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
        if time_scale is not None:
            raise ValueError("Text trees do not support time_scale")

        use_ascii = fmt == "ascii"
        text_tree = VerticalTextTree(
            tree,
            node_labels=node_labels,
            max_time=max_time,
            min_time=min_time,
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
    pad_top: float = 0
    pad_left: float = 0
    pad_bottom: float = 0
    pad_right: float = 0

    def set_padding(self, top, left, bottom, right):
        self.pad_top = top
        self.pad_left = left
        self.pad_bottom = bottom
        self.pad_right = right
        self._check()

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
        self._check()

    def _check(self):
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
    """
    The base class for plotting any box to canvas
    """

    text_height = 14  # May want to calculate this based on a font size
    line_height = text_height * 1.2  # allowing padding above and below a line

    def __init__(
        self,
        size,
        svg_class,
        root_svg_attributes=None,
        canvas_size=None,
    ):
        """
        Creates self.drawing, an svgwrite.Drawing object for further use, and populates
        it with a base group. The root_groups will be populated with
        items that can be accessed from the outside, such as the plotbox, axes, etc.
        """

        if root_svg_attributes is None:
            root_svg_attributes = {}
        if canvas_size is None:
            canvas_size = size
        dwg = svgwrite.Drawing(size=canvas_size, debug=True, **root_svg_attributes)

        self.image_size = size
        self.plotbox = Plotbox(size)
        self.root_groups = {}
        self.svg_class = svg_class
        self.timescaling = None
        self.root_svg_attributes = root_svg_attributes
        self.dwg_base = dwg.add(dwg.g(class_=svg_class))
        self.drawing = dwg

    def get_plotbox(self):
        """
        Get the svgwrite plotbox, creating it if necessary.
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
        group_attributes = {"transform": f"translate({rnd(pos[0])} {rnd(pos[1])})"}
        if group_class is not None:
            group_attributes["class_"] = group_class
        grp = add_to.add(dwg.g(**group_attributes))
        grp.add(dwg.text(text, **kwargs))


class SvgSkippedPlot(SvgPlot):
    def __init__(
        self,
        size,
        num_skipped,
    ):
        super().__init__(
            size,
            svg_class="skipped",
        )
        container = self.get_plotbox()
        x = self.plotbox.width / 2
        y = self.plotbox.height / 2
        self.add_text_in_group(
            f"{num_skipped} trees",
            container,
            (x, y - self.line_height / 2),
            text_anchor="middle",
        )
        self.add_text_in_group(
            "skipped", container, (x, y + self.line_height / 2), text_anchor="middle"
        )


class SvgAxisPlot(SvgPlot):
    """
    The class used for plotting either a tree or a tree sequence as an SVG file
    """

    standard_style = (
        ".background path {fill: #808080; fill-opacity: 0}"
        ".background path:nth-child(odd) {fill-opacity: .1}"
        ".axes {font-size: 14px}"
        ".x-axis .tick .lab {font-weight: bold; dominant-baseline: hanging}"
        ".axes, .tree {font-size: 14px; text-anchor: middle}"
        ".axes line, .edge {stroke: black; fill: none}"
        ".axes .ax-skip {stroke-dasharray: 4}"
        ".y-axis .grid {stroke: #FAFAFA}"
        ".node > .sym {fill: black; stroke: none}"
        ".site > .sym {stroke: black}"
        ".mut text {fill: red; font-style: italic}"
        ".mut.extra text {fill: hotpink}"
        ".mut line {fill: none; stroke: none}"  # Default hide mut line to expose edges
        ".mut .sym {fill: none; stroke: red}"
        ".mut.extra .sym {stroke: hotpink}"
        ".node .mut .sym {stroke-width: 1.5px}"
        ".tree text, .tree-sequence text {dominant-baseline: central}"
        ".plotbox .lab.lft {text-anchor: end}"
        ".plotbox .lab.rgt {text-anchor: start}"
    )

    # TODO: we may want to make some of the constants below into parameters
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
        time_scale,
        x_axis=None,
        y_axis=None,
        x_label=None,
        y_label=None,
        offsets=None,
        debug_box=None,
        omit_sites=None,
        canvas_size=None,
    ):
        super().__init__(
            size,
            svg_class,
            root_svg_attributes,
            canvas_size,
        )
        self.ts = ts
        dwg = self.drawing
        # Put all styles in a single stylesheet (required for Inkscape 0.92)
        style = self.standard_style + ("" if style is None else style)
        dwg.defs.add(dwg.style(style))
        self.debug_box = debug_box
        self.time_scale = check_time_scale(time_scale)
        self.y_axis = y_axis
        self.x_axis = x_axis
        if x_label is None and x_axis:
            x_label = "Genome position"
        if y_label is None and y_axis:
            if time_scale == "rank":
                y_label = "Node time"
            else:
                y_label = "Time"
            if ts.time_units != tskit.TIME_UNITS_UNKNOWN:
                y_label += f" ({ts.time_units})"
        self.x_label = x_label
        self.y_label = y_label
        self.offsets = Offsets() if offsets is None else offsets
        self.omit_sites = omit_sites
        self.mutations_outside_tree = set()  # mutations in here get an additional class

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
        self.plotbox.set_padding(top, left, bottom, right)
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
        site_muts=None,  # A dict of site id => mutation to plot as ticks on the x axis
        alternate_dash_positions=None,  # Where to alternate the axis from solid to dash
    ):
        if not self.x_axis and not self.x_label:
            return
        if alternate_dash_positions is None:
            alternate_dash_positions = np.array([])
        dwg = self.drawing
        axes = self.get_axes()
        x_axis = axes.add(dwg.g(class_="x-axis"))
        if self.x_label:
            self.add_text_in_group(
                self.x_label,
                x_axis,
                pos=((self.plotbox.left + self.plotbox.right) / 2, self.plotbox.max_y),
                group_class="title",
                class_="lab",
                transform="translate(0 -11)",
                text_anchor="middle",
            )
        if self.x_axis:
            if tick_length_upper is None:
                tick_length_upper = tick_length_lower
            y = rnd(self.plotbox.max_y - self.x_axis_offset)
            dash_locs = np.concatenate(
                (
                    [self.plotbox.left],
                    self.x_transform(alternate_dash_positions),
                    [self.plotbox.right],
                )
            )
            for i, (x1, x2) in enumerate(zip(dash_locs[:-1], dash_locs[1:])):
                x_axis.add(
                    dwg.line(
                        (rnd(x1), y),
                        (rnd(x2), y),
                        class_="ax-skip" if i % 2 else "ax-line",
                    )
                )
            if tick_positions is not None:
                if tick_labels is None or isinstance(tick_labels, np.ndarray):
                    if tick_labels is None:
                        tick_labels = tick_positions
                    tick_labels = create_tick_labels(tick_labels)  # format integers

                upper_length = -tick_length_upper if site_muts is None else 0
                ticks_group = x_axis.add(dwg.g(class_="ticks"))
                for pos, lab in itertools.zip_longest(tick_positions, tick_labels):
                    tick = ticks_group.add(
                        dwg.g(
                            class_="tick",
                            transform=f"translate({rnd(self.x_transform(pos))} {y})",
                        )
                    )
                    tick.add(
                        dwg.line((0, rnd(upper_length)), (0, rnd(tick_length_lower)))
                    )
                    self.add_text_in_group(
                        lab,
                        tick,
                        class_="lab",
                        # place origin at the bottom of the tick plus a single px space
                        pos=(0, tick_length_lower + 1),
                    )
            if not self.omit_sites and site_muts is not None:
                # Add sites as vertical lines with overlaid mutations as upper chevrons
                for s_id, mutations in site_muts.items():
                    s = self.ts.site(s_id)
                    x = self.x_transform(s.position)
                    site = x_axis.add(
                        dwg.g(
                            class_=f"site s{s.id + self.offsets.site}",
                            transform=f"translate({rnd(x)} {y})",
                        )
                    )
                    site.add(
                        dwg.line((0, 0), (0, rnd(-tick_length_upper)), class_="sym")
                    )
                    for i, m in enumerate(reversed(mutations)):
                        mutation_class = f"mut m{m.id + self.offsets.mutation}"
                        if m.id in self.mutations_outside_tree:
                            mutation_class += " extra"
                        mut = dwg.g(class_=mutation_class)
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
        ticks,  # A dict of pos->label
        upper=None,  # In plot coords
        lower=None,  # In plot coords
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
                group_class="title",
                class_="lab",
                text_anchor="middle",
                transform="translate(11) rotate(-90)",
            )
        if self.y_axis:
            y_axis.add(dwg.line((x, rnd(lower)), (x, rnd(upper)), class_="ax-line"))
            ticks_group = y_axis.add(dwg.g(class_="ticks"))
            for y, label in ticks.items():
                tick = ticks_group.add(
                    dwg.g(
                        class_="tick",
                        transform=f"translate({x} {rnd(self.timescaling.transform(y))})",
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
                    # place the origin at the left of the tickmark plus a single px space
                    label,
                    tick,
                    pos=(rnd(-tick_length_left - 1), 0),
                    class_="lab",
                    text_anchor="end",
                )

    def shade_background(
        self,
        breaks,
        tick_length_lower,
        tree_width=None,
        bottom_padding=None,
    ):
        if not self.x_axis:
            return
        if tree_width is None:
            tree_width = self.plotbox.width
        if bottom_padding is None:
            bottom_padding = self.plotbox.pad_bottom
        plot_breaks = self.x_transform(np.array(breaks))
        dwg = self.drawing

        # For tree sequences, we need to add on the background shaded regions
        self.root_groups["background"] = self.dwg_base.add(dwg.g(class_="background"))
        y = self.image_size[1] - self.x_axis_offset
        for i in range(1, len(breaks)):
            break_x = plot_breaks[i]
            prev_break_x = plot_breaks[i - 1]
            tree_x = i * tree_width + self.plotbox.left
            prev_tree_x = (i - 1) * tree_width + self.plotbox.left
            # Shift diagonal lines between tree & axis into the treebox a little
            diag_height = y - (self.image_size[1] - bottom_padding)
            self.root_groups["background"].add(
                # NB: the path below draws straight diagonal lines between the tree boxes
                # and the X axis. An alternative implementation using bezier curves could
                # substitute the following for lines 2 and 4 of the path spec string
                # "l0,{box_h:g} c0,{diag_h} {rdiag_x},0 {rdiag_x},{diag_h} "
                # "c0,-{diag_h} {ldiag_x},0 {ldiag_x},-{diag_h} l0,-{box_h:g}z"
                dwg.path(
                    "M{start_x:g},0 l{box_w:g},0 "  # Top left to top right of tree box
                    "l0,{box_h:g} l{rdiag_x:g},{diag_h:g} "  # Down to axis
                    "l0,{tick_h:g} l{ax_x:g},0 l0,-{tick_h:g} "  # Between axis ticks
                    "l{ldiag_x:g},-{diag_h:g} l0,-{box_h:g}z".format(  # Up from axis
                        start_x=rnd(prev_tree_x),
                        box_w=rnd(tree_x - prev_tree_x),
                        box_h=rnd(y - diag_height),
                        rdiag_x=rnd(break_x - tree_x),
                        diag_h=rnd(diag_height),
                        tick_h=rnd(tick_length_lower),
                        ax_x=rnd(prev_break_x - break_x),
                        ldiag_x=rnd(rnd(prev_tree_x) - rnd(prev_break_x)),
                    )
                )
            )

    def x_transform(self, x):
        raise NotImplementedError(
            "No transform func defined for genome pos -> plot coords"
        )


class SvgTreeSequence(SvgAxisPlot):
    """
    A class to draw a tree sequence in SVG format.

    See :meth:`TreeSequence.draw_svg` for a description of usage and parameters.
    """

    def __init__(
        self,
        ts,
        size,
        x_scale,
        time_scale,
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
        x_lim=None,
        max_time=None,
        min_time=None,
        node_attrs=None,
        mutation_attrs=None,
        edge_attrs=None,
        node_label_attrs=None,
        mutation_label_attrs=None,
        tree_height_scale=None,
        max_tree_height=None,
        max_num_trees=None,
        **kwargs,
    ):
        if max_time is None and max_tree_height is not None:
            max_time = max_tree_height
            # Deprecated in 0.3.6
            warnings.warn(
                "max_tree_height is deprecated; use max_time instead",
                FutureWarning,
            )
        if time_scale is None and tree_height_scale is not None:
            time_scale = tree_height_scale
            # Deprecated in 0.3.6
            warnings.warn(
                "tree_height_scale is deprecated; use time_scale instead",
                FutureWarning,
            )
        x_lim = check_x_lim(x_lim, max_x=ts.sequence_length)
        ts, self.tree_status, offsets = clip_ts(ts, x_lim[0], x_lim[1], max_num_trees)

        use_tree = self.tree_status & OMIT == 0
        use_skipped = np.append(np.diff(self.tree_status & OMIT_MIDDLE == 0) == 1, 0)
        num_plotboxes = np.sum(np.logical_or(use_tree, use_skipped))
        if size is None:
            size = (200 * int(num_plotboxes), 200)
        if max_time is None:
            max_time = "ts"
        if min_time is None:
            min_time = "ts"
        # X axis shown by default
        if x_axis is None:
            x_axis = True
        super().__init__(
            ts,
            size,
            root_svg_attributes,
            style,
            svg_class="tree-sequence",
            time_scale=time_scale,
            x_axis=x_axis,
            y_axis=y_axis,
            x_label=x_label,
            y_label=y_label,
            offsets=offsets,
            **kwargs,
        )
        x_scale = check_x_scale(x_scale)
        if node_labels is None:
            node_labels = {u: str(u) for u in range(ts.num_nodes)}
        if force_root_branch is None:
            force_root_branch = any(
                any(tree.parent(mut.node) == NULL for mut in tree.mutations())
                for tree, use in zip(ts.trees(), use_tree)
                if use
            )

        # TODO add general padding arguments following matplotlib's terminology.
        self.set_spacing(top=0, left=20, bottom=10, right=20)
        subplot_size = (self.plotbox.width / num_plotboxes, self.plotbox.height)
        subplots = []
        for tree, use, summary in zip(ts.trees(), use_tree, use_skipped):
            if use:
                subplots.append(
                    SvgTree(
                        tree,
                        size=subplot_size,
                        time_scale=time_scale,
                        node_labels=node_labels,
                        mutation_labels=mutation_labels,
                        order=order,
                        force_root_branch=force_root_branch,
                        symbol_size=symbol_size,
                        max_time=max_time,
                        min_time=min_time,
                        node_attrs=node_attrs,
                        mutation_attrs=mutation_attrs,
                        edge_attrs=edge_attrs,
                        node_label_attrs=node_label_attrs,
                        mutation_label_attrs=mutation_label_attrs,
                        offsets=offsets,
                        # Do not plot axes on these subplots
                        **kwargs,  # pass though e.g. debug boxes
                    )
                )
                last_used_index = tree.index
            elif summary:
                subplots.append(
                    SvgSkippedPlot(
                        size=subplot_size, num_skipped=tree.index - last_used_index
                    )
                )
        y = self.plotbox.top
        self.tree_plotbox = subplots[0].plotbox
        tree_is_used, breaks, skipbreaks = self.find_used_trees()
        self.draw_x_axis(
            x_scale,
            tree_is_used,
            breaks,
            skipbreaks,
            tick_length_lower=self.default_tick_length,  # TODO - parameterize
            tick_length_upper=self.default_tick_length_site,  # TODO - parameterize
        )
        y_low = self.tree_plotbox.bottom
        if y_axis is not None:
            tscales = {s.timescaling for s in subplots if s.timescaling}
            if len(tscales) > 1:
                raise ValueError(
                    "Can't draw a tree sequence Y axis if trees vary in timescale"
                )
            self.timescaling = tscales.pop()
            y_low = self.timescaling.transform(self.timescaling.min_time)
            if y_ticks is None:
                used_nodes = edge_and_sample_nodes(ts, breaks[skipbreaks])
                y_ticks = np.unique(ts.nodes_time[used_nodes])
                if self.time_scale == "rank":
                    # Ticks labelled by time not rank
                    y_ticks = dict(enumerate(y_ticks))

        self.draw_y_axis(
            ticks=check_y_ticks(y_ticks),
            upper=self.tree_plotbox.top,
            lower=y_low,
            tick_length_left=self.default_tick_length,
            gridlines=y_gridlines,
        )

        subplot_x = self.plotbox.left
        container = self.get_plotbox()  # Top-level TS plotbox contains all trees
        container["class"] = container["class"] + " trees"
        for subplot in subplots:
            svg_subplot = container.add(
                self.drawing.g(
                    class_=subplot.svg_class,
                    transform=f"translate({rnd(subplot_x)} {y})",
                )
            )
            for svg_items in subplot.root_groups.values():
                svg_subplot.add(svg_items)
            subplot_x += subplot.image_size[0]

    def find_used_trees(self):
        """
        Return a boolean array of which trees are actually plotted,
        a list of which breakpoints are used to transition between plotted trees,
        and a 2 x n array (often n=0) of indexes into these breakpoints delimiting
        the regions that should be plotted as "skipped"
        """
        tree_is_used = (self.tree_status & OMIT) != OMIT
        break_used_as_tree_left = np.append(tree_is_used, False)
        break_used_as_tree_right = np.insert(tree_is_used, 0, False)
        break_used = np.logical_or(break_used_as_tree_left, break_used_as_tree_right)
        all_breaks = self.ts.breakpoints(True)
        used_breaks = all_breaks[break_used]
        mark_skip_transitions = np.concatenate(
            ([False], np.diff(self.tree_status & OMIT_MIDDLE) != 0, [False])
        )
        skipregion_indexes = np.where(mark_skip_transitions[break_used])[0]
        assert len(skipregion_indexes) % 2 == 0  # all skipped regions have start, end
        return tree_is_used, used_breaks, skipregion_indexes.reshape((-1, 2))

    def draw_x_axis(
        self,
        x_scale,
        tree_is_used,
        breaks,
        skipbreaks,
        tick_length_lower=SvgAxisPlot.default_tick_length,
        tick_length_upper=SvgAxisPlot.default_tick_length_site,
    ):
        """
        Add extra functionality to the original draw_x_axis method in SvgAxisPlot,
        to account for the background shading that is displayed in a tree sequence
        and in case trees are omitted from the middle of the tree sequence
        """
        if not self.x_axis and not self.x_label:
            return
        if x_scale == "physical":
            # In a tree sequence plot, the x_transform is used for the ticks, background
            # shading positions, and sites along the x-axis. Each tree will have its own
            # separate x_transform function for node positions within the tree.

            # For a plot with a break on the x-axis (representing "skipped" trees), the
            # x_transform is a piecewise function. We need to identify the breakpoints
            # where the x-scale transitions from the standard scale to the scale(s) used
            # within a skipped region

            skipregion_plot_width = self.tree_plotbox.width
            skipregion_span = np.diff(breaks[skipbreaks]).T[0]
            std_scale = (
                self.plotbox.width - skipregion_plot_width * len(skipregion_span)
            ) / (breaks[-1] - breaks[0] - np.sum(skipregion_span))
            skipregion_pos = breaks[skipbreaks].flatten()
            genome_pos = np.concatenate(([breaks[0]], skipregion_pos, [breaks[-1]]))
            plot_step = np.full(len(genome_pos) - 1, skipregion_plot_width)
            plot_step[::2] = std_scale * np.diff(genome_pos)[::2]
            plot_pos = np.cumsum(np.insert(plot_step, 0, self.plotbox.left))
            # Convert to slope + intercept form
            slope = np.diff(plot_pos) / np.diff(genome_pos)
            intercept = plot_pos[1:] - slope * genome_pos[1:]
            self.x_transform = lambda y: (
                y * slope[np.searchsorted(skipregion_pos, y)]
                + intercept[np.searchsorted(skipregion_pos, y)]
            )
            tick_positions = breaks
            site_muts = {
                s.id: s.mutations
                for tree, use in zip(self.ts.trees(), tree_is_used)
                for s in tree.sites()
                if use
            }

            self.shade_background(
                breaks,
                tick_length_lower,
                self.tree_plotbox.max_x,
                self.plotbox.pad_bottom + self.tree_plotbox.pad_bottom,
            )
        else:

            # For a treewise plot, the only time the x_transform is used is to apply
            # to tick positions, so simply use positions 0..num_used_breaks for the
            # positions, and a simple transform
            self.x_transform = (
                lambda x: self.plotbox.left + x / (len(breaks) - 1) * self.plotbox.width
            )
            tick_positions = np.arange(len(breaks))

            site_muts = None  # It doesn't make sense to plot sites for "treewise" plots
            tick_length_upper = None  # No sites plotted, so use the default upper tick

            # NB: no background shading needed if x_scale is "treewise"

            skipregion_pos = skipbreaks.flatten()

        first_tick = 1 if np.any(self.tree_status[tree_is_used] & LEFT_CLIP) else 0
        last_tick = -1 if np.any(self.tree_status[tree_is_used] & RIGHT_CLIP) else None

        super().draw_x_axis(
            tick_positions=tick_positions[first_tick:last_tick],
            tick_labels=breaks[first_tick:last_tick],
            tick_length_lower=tick_length_lower,
            tick_length_upper=tick_length_upper,
            site_muts=site_muts,
            alternate_dash_positions=skipregion_pos,
        )


class SvgTree(SvgAxisPlot):
    """
    A class to draw a tree in SVG format.

    See :meth:`Tree.draw_svg` for a description of usage and frequently used parameters.
    """

    def __init__(
        self,
        tree,
        size=None,
        max_time=None,
        min_time=None,
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
        all_edge_mutations=None,
        time_scale=None,
        tree_height_scale=None,
        node_attrs=None,
        mutation_attrs=None,
        edge_attrs=None,
        node_label_attrs=None,
        mutation_label_attrs=None,
        offsets=None,
        omit_sites=None,
        **kwargs,
    ):
        if max_time is None and max_tree_height is not None:
            max_time = max_tree_height
            # Deprecated in 0.3.6
            warnings.warn(
                "max_tree_height is deprecated; use max_time instead",
                FutureWarning,
            )
        if time_scale is None and tree_height_scale is not None:
            time_scale = tree_height_scale
            # Deprecated in 0.3.6
            warnings.warn(
                "tree_height_scale is deprecated; use time_scale instead",
                FutureWarning,
            )
        if size is None:
            size = (200, 200)
        if symbol_size is None:
            symbol_size = 6
        self.symbol_size = symbol_size
        ts = tree.tree_sequence
        tree_index = tree.index
        if offsets is not None:
            tree_index += offsets.tree
        super().__init__(
            ts,
            size,
            root_svg_attributes,
            style,
            svg_class=f"tree t{tree_index}",
            time_scale=time_scale,
            x_axis=x_axis,
            y_axis=y_axis,
            x_label=x_label,
            y_label=y_label,
            offsets=offsets,
            omit_sites=omit_sites,
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
        if not omit_sites:
            for site in tree.sites():
                for mutation in site.mutations:
                    if mutation.node in nodes:
                        self.node_mutations[mutation.node].append(mutation)
                        if tree.parent(mutation.node) == NULL:
                            self.mutations_over_roots = True
                    else:
                        unplotted.append(mutation.id + self.offsets.mutation)
        if len(unplotted) > 0:
            logging.warning(
                f"Mutations {unplotted} are above nodes which are not present in the "
                "displayed tree, so are not plotted on the topology."
            )
        self.left_extent = tree.interval.left
        self.right_extent = tree.interval.right
        if not omit_sites and all_edge_mutations:
            tree_left = tree.interval.left
            tree_right = tree.interval.right
            edge_left = ts.tables.edges.left
            edge_right = ts.tables.edges.right
            node_edges = tree.edge_array
            # whittle mutations down so we only need look at those above the tree nodes
            mut_t = ts.tables.mutations
            focal_mutations = np.isin(mut_t.node, np.fromiter(nodes, mut_t.node.dtype))
            mutation_nodes = mut_t.node[focal_mutations]
            mutation_positions = ts.tables.sites.position[mut_t.site][focal_mutations]
            mutation_ids = np.arange(ts.num_mutations, dtype=int)[focal_mutations]
            for m_id, node, pos in zip(
                mutation_ids, mutation_nodes, mutation_positions
            ):
                curr_edge = node_edges[node]
                if curr_edge >= 0:
                    if (
                        edge_left[curr_edge] <= pos < tree_left
                    ):  # Mutation on this edge but to left of plotted tree
                        self.node_mutations[node].append(ts.mutation(m_id))
                        self.mutations_outside_tree.add(m_id)
                        self.left_extent = min(self.left_extent, pos)
                    elif (
                        tree_right <= pos < edge_right[curr_edge]
                    ):  # Mutation on this edge but to right of plotted tree
                        self.node_mutations[node].append(ts.mutation(m_id))
                        self.mutations_outside_tree.add(m_id)
                        self.right_extent = max(self.right_extent, pos)
            if self.right_extent != tree.interval.right:
                # Use nextafter so extent of plotting incorporates the mutation
                self.right_extent = np.nextafter(
                    self.right_extent, self.right_extent + 1
                )
        # attributes for symbols
        half_symbol_size = f"{rnd(symbol_size / 2):g}"
        symbol_size = f"{rnd(symbol_size):g}"
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
        for _, mutations in self.node_mutations.items():
            for mutation in mutations:
                m = mutation.id + self.offsets.mutation
                # We need to offset the mutation symbol so that it's centred
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
                elif m in mutation_labels:
                    label = str(mutation_labels[m])
                self.mutation_label_attrs[m] = {"text": label}
                if mutation_label_attrs is not None and m in mutation_label_attrs:
                    self.mutation_label_attrs[m].update(mutation_label_attrs[m])
                add_class(self.mutation_label_attrs[m], "lab")

        self.set_spacing(top=10, left=20, bottom=15, right=20)
        self.assign_y_coordinates(max_time, min_time, force_root_branch)
        self.assign_x_coordinates()
        tick_length_lower = self.default_tick_length  # TODO - parameterize
        tick_length_upper = self.default_tick_length_site  # TODO - parameterize
        if all_edge_mutations:
            self.shade_background(tree.interval, tick_length_lower)

        first_site, last_site = np.searchsorted(
            self.ts.tables.sites.position, [self.left_extent, self.right_extent]
        )
        site_muts = {site_id: [] for site_id in range(first_site, last_site)}
        # Only use mutations plotted on the tree (not necessarily all at the site)
        for muts in self.node_mutations.values():
            for mut in muts:
                site_muts[mut.site].append(mut)

        self.draw_x_axis(
            tick_positions=np.array(tree.interval),
            tick_length_lower=tick_length_lower,
            tick_length_upper=tick_length_upper,
            site_muts=site_muts,
        )
        if y_ticks is None:
            y_ticks = {h: ts.node(u).time for u, h in self.node_height.items()}

        self.draw_y_axis(
            ticks=check_y_ticks(y_ticks),
            lower=self.timescaling.transform(self.timescaling.min_time),
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
        max_time,
        min_time,
        force_root_branch,
        bottom_space=SvgAxisPlot.line_height,
        top_space=SvgAxisPlot.line_height,
    ):
        """
        Create a self.node_height dict, a self.timescaling instance and
        self.min_root_branch_plot_length for use in plotting. Allow extra space within
        the plotbox, at the bottom for leaf labels, and  (potentially, if no root
        branches are plotted) above the topmost root node for root labels.
        """
        max_time = check_max_time(max_time, self.time_scale != "rank")
        min_time = check_min_time(min_time, self.time_scale != "rank")
        node_time = self.ts.nodes_time
        mut_time = self.ts.mutations_time
        root_branch_len = 0
        if self.time_scale == "rank":
            t = np.zeros_like(node_time)
            if max_time == "tree":
                # We only rank the times within the tree in this case.
                for u in self.tree.nodes():
                    t[u] = node_time[u]
            else:
                # only rank the nodes that are actually referenced in the edge table
                # (non-referenced nodes could occur if the user specifies x_lim values)
                # However, we do include nodes in trees that have been skipped
                use_time = edge_and_sample_nodes(self.ts)
                t[use_time] = node_time[use_time]
            node_time = t
            times = np.unique(node_time[node_time <= self.ts.max_root_time])
            max_node_height = len(times)
            depth = {t: j for j, t in enumerate(times)}
            if self.mutations_over_roots or force_root_branch:
                root_branch_len = 1  # Will get scaled later
            max_time = max(depth.values()) + root_branch_len
            if min_time in (None, "tree", "ts"):
                assert min(depth.values()) == 0
                min_time = 0
            # In pathological cases, all the nodes are at the same time
            if max_time == min_time:
                max_time = min_time + 1
            self.node_height = {u: depth[node_time[u]] for u in self.tree.nodes()}
            for u in self.node_mutations.keys():
                parent = self.tree.parent(u)
                if parent == NULL:
                    top = self.node_height[u] + root_branch_len
                else:
                    top = self.node_height[parent]
                self.process_mutations_over_node(
                    u, self.node_height[u], top, ignore_times=True
                )
        else:
            assert self.time_scale in ["time", "log_time"]
            self.node_height = {u: node_time[u] for u in self.tree.nodes()}
            if max_time == "tree":
                max_node_height = max(self.node_height.values())
                max_mut_height = np.nanmax(
                    [0] + [mut.time for m in self.node_mutations.values() for mut in m]
                )
                max_time = max(max_node_height, max_mut_height)  # Reuse variable
            elif max_time == "ts":
                max_node_height = self.ts.max_root_time
                max_mut_height = np.nanmax(np.append(mut_time, 0))
                max_time = max(max_node_height, max_mut_height)  # Reuse variable
            if min_time == "tree":
                min_time = min(self.node_height.values())
                # don't need to check mutation times, as they must be above a node
            elif min_time == "ts":
                min_time = np.min(self.ts.nodes_time[edge_and_sample_nodes(self.ts)])
            # In pathological cases, all the nodes are at the same time
            if min_time == max_time:
                max_time = min_time + 1
            if self.mutations_over_roots or force_root_branch:
                # Define a minimum root branch length, after transformation if necessary
                if self.time_scale != "log_time":
                    root_branch_len = (max_time - min_time) * self.root_branch_fraction
                else:
                    max_plot_y = np.log(max_time + 1)
                    diff_plot_y = max_plot_y - np.log(min_time + 1)
                    root_plot_y = max_plot_y + diff_plot_y * self.root_branch_fraction
                    root_branch_len = np.exp(root_plot_y) - 1 - max_time
                # If necessary, allow for this extra branch in max_time
                if max_node_height + root_branch_len > max_time:
                    max_time = max_node_height + root_branch_len
            for u in self.node_mutations.keys():
                parent = self.tree.parent(u)
                if parent == NULL:
                    # This is a root: if muts have no times we must specify an upper time
                    top = self.node_height[u] + root_branch_len
                else:
                    top = self.node_height[parent]
                self.process_mutations_over_node(u, self.node_height[u], top)

        assert float(max_time) == max_time
        assert float(min_time) == min_time
        # Add extra space above the top and below the bottom of the tree to keep the
        # node labels within the plotbox (but top label space not needed if the
        # existence of a root branch pushes the whole tree + labels downwards anyway)
        top_space = 0 if root_branch_len > 0 else top_space
        self.timescaling = Timescaling(
            max_time=max_time,
            min_time=min_time,
            plot_min=self.plotbox.height + self.plotbox.top - bottom_space,
            plot_range=self.plotbox.height - top_space - bottom_space,
            use_log_transform=(self.time_scale == "log_time"),
        )

        # Calculate default root branch length to use (in plot coords). This is a
        # minimum, as branches with deep root mutations could be longer
        self.min_root_branch_plot_length = self.timescaling.transform(
            self.timescaling.max_time
        ) - self.timescaling.transform(self.timescaling.max_time + root_branch_len)

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
            (x - self.left_extent)
            / (self.right_extent - self.left_extent)
            * self.plotbox.width
            + self.plotbox.left
        )

    def info_classes(self, focal_node_id):
        """
        For a focal node id, return a set of classes that encode this useful information:
            "a<X>" or "root": where <X> == id of immediate ancestor (parent) node
            "i<I>":           where <I> == individual id
            "p<P>":           where <P> == population id
            "n<Y>":           where <Y> == focal node id
            "m<A>":           where <A> == mutation id
            "s<B>":           where <B> == site id of all mutations
            "c<N>" or "leaf": where <N> == number of direct children of this node
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
        else:
            classes.add(f"c{self.tree.num_children(focal_node_id)}")
        for mutation in self.node_mutations[focal_node_id]:
            # Adding mutations and sites above this node allows identification
            # of the tree under any specific mutation
            classes.add(f"m{mutation.id + self.offsets.mutation}")
            classes.add(f"s{mutation.site+ self.offsets.site}")
        return sorted(classes)

    def draw_tree(self):
        dwg = self.drawing
        node_x_coord_map = self.node_x_coord_map
        node_y_coord_map = {
            u: self.timescaling.transform(h) for u, h in self.node_height.items()
        }
        tree = self.tree
        left_child = get_left_child(tree, self.traversal_order)

        # Iterate over nodes, adding groups to reflect the tree hierarchy
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
                            node_y_coord_map[u]
                            - self.timescaling.transform(mutation.time),
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
                dy = self.timescaling.transform(mutation.time) - pu[1]
                mutation_id = mutation.id + self.offsets.mutation
                mutation_class = (
                    f"mut m{mutation_id} " f"s{mutation.site+ self.offsets.site}"
                )
                # Use the real mutation ID here, since we are referencing into the ts
                if util.is_unknown_time(self.ts.mutation(mutation.id).time):
                    mutation_class += " unknown_time"
                if mutation_id in self.mutations_outside_tree:
                    mutation_class += " extra"
                mut_group = curr_svg_group.add(
                    dwg.g(class_=mutation_class, transform=f"translate(0 {rnd(dy)})")
                )
                # A line from the mutation to the node below, normally hidden, but
                # revealable if we want to flag the path below a mutation
                mut_group.add(dwg.line(end=(0, -rnd(dy))))
                # Symbols
                mut_group.add(dwg.path(**self.mutation_attrs[mutation_id]))
                # Labels
                if u == left_child[tree.parent(u)]:
                    mut_label_class = "lft"
                    transform = f"translate(-{rnd(2+self.symbol_size/2)} 0)"
                else:
                    mut_label_class = "rgt"
                    transform = f"translate({rnd(2+self.symbol_size/2)} 0)"
                add_class(self.mutation_label_attrs[mutation_id], mut_label_class)
                self.mutation_label_attrs[mutation_id]["transform"] = transform
                mut_group.add(dwg.text(**self.mutation_label_attrs[mutation_id]))

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
        tick_labels = ts.breakpoints(as_array=True)
        if position_label_format is None:
            position_scale_labels = create_tick_labels(tick_labels)
        else:
            position_scale_labels = [
                position_label_format.format(x) for x in tick_labels
            ]

        time = ts.tables.nodes.time
        time_scale_labels = [
            time_label_format.format(time[u]) for u in range(ts.num_nodes)
        ]

        trees = [
            VerticalTextTree(
                tree,
                max_time="ts",
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

    left_neighbour = np.full(tree.tree_sequence.num_nodes + 1, NULL, dtype=int)

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
    left_child = np.full(tree.tree_sequence.num_nodes + 1, NULL, dtype=int)
    for u in tree.nodes(order=traversal_order):
        parent = tree.parent(u)
        if parent != NULL and left_child[parent] == NULL:
            left_child[parent] = u
    return left_child


def node_time_depth(tree, min_branch_length=None, max_time="tree"):
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
    if max_time == "tree":
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
        assert max_time == "ts"
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
        max_time=None,
        min_time=None,
        use_ascii=False,
        orientation=None,
        order=None,
    ):
        self.tree = tree
        self.traversal_order = check_order(order)
        self.max_time = check_max_time(max_time, allow_numeric=False)
        self.min_time = check_min_time(min_time, allow_numeric=False)
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
        self.time_position, total_depth = node_time_depth(tree, max_time=self.max_time)
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
