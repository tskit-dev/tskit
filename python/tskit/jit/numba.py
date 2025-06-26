try:
    import numba
except ImportError:
    raise ImportError(
        "Numba is not installed. Please install it with `pip install numba` "
        "or `conda install numba` to use the tskit.jit.numba module."
    )


tree_sequence_spec = [
    ("num_edges", numba.int64),
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
        return NumbaTreePosition(self, (0, 0), (0, 0), (0, 0))


tree_position_spec = [
    ("ts", NumbaTreeSequence.class_type.instance_type),
    ("interval", numba.types.UniTuple(numba.float64, 2)),
    ("edges_in_index_range", numba.types.UniTuple(numba.int32, 2)),
    ("edges_out_index_range", numba.types.UniTuple(numba.int32, 2)),
]


@numba.experimental.jitclass(tree_position_spec)
class NumbaTreePosition:
    def __init__(self, ts, interval, edges_in_index_range, edges_out_index_range):
        self.ts = ts
        self.interval = interval
        self.edges_in_index_range = edges_in_index_range
        self.edges_out_index_range = edges_out_index_range

    def next(self):  # noqa: A003
        M = self.ts.num_edges
        edges_left = self.ts.edges_left
        edges_right = self.ts.edges_right
        in_order = self.ts.indexes_edge_insertion_order
        out_order = self.ts.indexes_edge_removal_order

        left = self.interval[1]
        j = self.edges_in_index_range[1]
        k = self.edges_out_index_range[1]

        while k < M and edges_right[out_order[k]] == left:
            k += 1
        while j < M and edges_left[in_order[j]] == left:
            j += 1

        self.edges_in_index_range = (self.edges_in_index_range[1], j)
        self.edges_out_index_range = (self.edges_out_index_range[1], k)

        right = self.ts.sequence_length
        if j < M:
            right = min(right, edges_left[in_order[j]])
        if k < M:
            right = min(right, edges_right[out_order[k]])

        self.interval = (left, right)
        return j < M or left < self.ts.sequence_length


def numba_tree_sequence(ts):
    return NumbaTreeSequence(
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
