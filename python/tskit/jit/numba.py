import numpy as np

try:
    import numba
except ImportError:
    raise ImportError(
        "Numba is not installed. Please install it with `pip install numba` "
        "or `conda install numba` to use the tskit.jit.numba module."
    )


FORWARD = 1
REVERSE = -1


tree_sequence_spec = [
    ("num_trees", numba.int32),
    ("num_edges", numba.int32),
    ("sequence_length", numba.float64),
    ("edges_left", numba.float64[:]),
    ("edges_right", numba.float64[:]),
    ("indexes_edge_insertion_order", numba.int32[:]),
    ("indexes_edge_removal_order", numba.int32[:]),
    ("individuals_flags", numba.uint32[:]),
    ("nodes_time", numba.float64[:]),
    ("nodes_flags", numba.uint32[:]),
    ("nodes_population", numba.int32[:]),
    ("nodes_individual", numba.int32[:]),
    ("edges_parent", numba.int32[:]),
    ("edges_child", numba.int32[:]),
    ("sites_position", numba.float64[:]),
    ("mutations_site", numba.int32[:]),
    ("mutations_node", numba.int32[:]),
    ("mutations_parent", numba.int32[:]),
    ("mutations_time", numba.float64[:]),
    ("breakpoints", numba.float64[:]),
]


@numba.experimental.jitclass(tree_sequence_spec)
class NumbaTreeSequence:
    def __init__(
        self,
        num_trees,
        num_edges,
        sequence_length,
        edges_left,
        edges_right,
        indexes_edge_insertion_order,
        indexes_edge_removal_order,
        individuals_flags,
        nodes_time,
        nodes_flags,
        nodes_population,
        nodes_individual,
        edges_parent,
        edges_child,
        sites_position,
        mutations_site,
        mutations_node,
        mutations_parent,
        mutations_time,
        breakpoints,
    ):
        self.num_trees = num_trees
        self.num_edges = num_edges
        self.sequence_length = sequence_length
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.indexes_edge_insertion_order = indexes_edge_insertion_order
        self.indexes_edge_removal_order = indexes_edge_removal_order
        self.individuals_flags = individuals_flags
        self.nodes_time = nodes_time
        self.nodes_flags = nodes_flags
        self.nodes_population = nodes_population
        self.nodes_individual = nodes_individual
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.sites_position = sites_position
        self.mutations_site = mutations_site
        self.mutations_node = mutations_node
        self.mutations_parent = mutations_parent
        self.mutations_time = mutations_time
        self.breakpoints = breakpoints

    def tree_position(self):
        return NumbaTreePosition(self)


edge_range_spec = [
    ("start", numba.int32),
    ("stop", numba.int32),
    ("order", numba.int32[:]),
]


@numba.experimental.jitclass(edge_range_spec)
class NumbaEdgeRange:
    def __init__(self, start, stop, order):
        self.start = start
        self.stop = stop
        self.order = order


tree_position_spec = [
    ("ts", NumbaTreeSequence.class_type.instance_type),
    ("index", numba.int32),
    ("direction", numba.int32),
    ("interval", numba.types.UniTuple(numba.float64, 2)),
    ("in_range", NumbaEdgeRange.class_type.instance_type),
    ("out_range", NumbaEdgeRange.class_type.instance_type),
]


@numba.experimental.jitclass(tree_position_spec)
class NumbaTreePosition:
    def __init__(self, ts):
        self.ts = ts
        self.index = -1
        self.direction = 0
        self.interval = (0, 0)
        self.in_range = NumbaEdgeRange(0, 0, np.zeros(0, dtype=numba.int32))
        self.out_range = NumbaEdgeRange(0, 0, np.zeros(0, dtype=numba.int32))

    def set_null(self):
        self.index = -1
        self.interval = (0, 0)

    def next(self):  # noqa: A003
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order

        if self.index == -1:
            self.interval = (self.interval[0], 0)
            self.out_range.stop = 0
            self.in_range.stop = 0
            self.direction = FORWARD

        if self.direction == FORWARD:
            left_current_index = self.in_range.stop
            right_current_index = self.out_range.stop
        else:
            left_current_index = self.out_range.stop + 1
            right_current_index = self.in_range.stop + 1

        left = self.interval[1]

        j = right_current_index
        self.out_range.start = j
        while j < M and right_coords[right_order[j]] == left:
            j += 1
        self.out_range.stop = j
        self.out_range.order = right_order

        j = left_current_index
        self.in_range.start = j
        while j < M and left_coords[left_order[j]] == left:
            j += 1
        self.in_range.stop = j
        self.in_range.order = left_order

        self.direction = FORWARD
        self.index += 1
        if self.index == self.ts.num_trees:
            self.set_null()
        else:
            self.interval = (left, breakpoints[self.index + 1])
        return self.index != -1

    def prev(self):
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order

        if self.index == -1:
            self.index = self.ts.num_trees
            self.interval = (self.ts.sequence_length, self.interval[1])
            self.in_range.stop = M - 1
            self.out_range.stop = M - 1
            self.direction = REVERSE

        if self.direction == REVERSE:
            left_current_index = self.out_range.stop
            right_current_index = self.in_range.stop
        else:
            left_current_index = self.in_range.stop - 1
            right_current_index = self.out_range.stop - 1

        right = self.interval[0]

        j = left_current_index
        self.out_range.start = j
        while j >= 0 and left_coords[left_order[j]] == right:
            j -= 1
        self.out_range.stop = j
        self.out_range.order = left_order

        j = right_current_index
        self.in_range.start = j
        while j >= 0 and right_coords[right_order[j]] == right:
            j -= 1
        self.in_range.stop = j
        self.in_range.order = right_order

        self.direction = REVERSE
        self.index -= 1
        if self.index == -1:
            self.set_null()
        else:
            self.interval = (breakpoints[self.index], right)
        return self.index != -1


def numba_tree_sequence(ts):
    return NumbaTreeSequence(
        num_trees=ts.num_trees,
        num_edges=ts.num_edges,
        sequence_length=ts.sequence_length,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        indexes_edge_insertion_order=ts.indexes_edge_insertion_order,
        indexes_edge_removal_order=ts.indexes_edge_removal_order,
        individuals_flags=ts.individuals_flags,
        nodes_time=ts.nodes_time,
        nodes_flags=ts.nodes_flags,
        nodes_population=ts.nodes_population,
        nodes_individual=ts.nodes_individual,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        sites_position=ts.sites_position,
        mutations_site=ts.mutations_site,
        mutations_node=ts.mutations_node,
        mutations_parent=ts.mutations_parent,
        mutations_time=ts.mutations_time,
        breakpoints=ts.breakpoints(as_array=True),
    )
