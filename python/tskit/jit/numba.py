import functools

import numpy as np

try:
    import numba
except ImportError:
    raise ImportError(
        "Numba is not installed. Please install it with `pip install numba` "
        "or `conda install numba` to use the tskit.jit.numba module."
    )


FORWARD = 1  #: Direction constant for forward tree traversal
REVERSE = -1  #: Direction constant for reverse tree traversal


edge_range_spec = [
    ("start", numba.int32),
    ("stop", numba.int32),
    ("order", numba.int32[:]),
]


@numba.experimental.jitclass(edge_range_spec)
class NumbaEdgeRange:
    """
    Represents a range of edges during tree traversal.

    This class encapsulates information about a contiguous range of edges
    that are either being removed or added to step from one tree to another
    The ``start`` and ``stop`` indices, when applied to the order array,
    define the ids of edges to process.

    Attributes
    ----------
    start : int32
        Starting index of the edge range (inclusive).
    stop : int32
        Stopping index of the edge range (exclusive).
    order : int32[]
        Array containing edge IDs in the order they should be processed.
        The edge ids in this range are order[start:stop].
    """

    def __init__(self, start, stop, order):
        self.start = start
        self.stop = stop
        self.order = order


# We cache these as otherwise we'll do a numba re-compile
# on every new call to `numba_tree_sequence`
@functools.lru_cache(None)
def _numba_tree_sequence(max_ancestral_length, max_derived_length):
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
        ("sites_ancestral_state", numba.types.UnicodeCharSeq(max_ancestral_length)[:]),
        ("mutations_site", numba.int32[:]),
        ("mutations_node", numba.int32[:]),
        ("mutations_parent", numba.int32[:]),
        ("mutations_time", numba.float64[:]),
        ("mutations_derived_state", numba.types.UnicodeCharSeq(max_derived_length)[:]),
        ("breakpoints", numba.float64[:]),
    ]

    @numba.experimental.jitclass(tree_sequence_spec)
    class NumbaTreeSequence:
        """
        A Numba-compatible representation of a tree sequence.

        This class provides access a tree sequence class that can be used
        from within Numba "njit" compiled functions, as it is a Numba
        "jitclass". :meth:`numba_tree_sequence` should be used to
        create this class from a :class:`tskit.TreeSequence` object,
        before it is passed to a Numba function.

        Attributes
        ----------
        num_trees : int32
            Number of trees in the tree sequence.
        num_edges : int32
            Number of edges in the tree sequence.
        sequence_length : float64
            Total sequence length of the tree sequence.
        edges_left : float64[]
            Left coordinates of edges.
        edges_right : float64[]
            Right coordinates of edges.
        edges_parent : int32[]
            Parent node IDs for each edge.
        edges_child : int32[]
            Child node IDs for each edge.
        nodes_time : float64[]
            Time values for each node.
        nodes_flags : uint32[]
            Flag values for each node.
        nodes_population : int32[]
            Population IDs for each node.
        nodes_individual : int32[]
            Individual IDs for each node.
        individuals_flags : uint32[]
            Flag values for each individual.
        sites_position : float64[]
            Positions of sites along the sequence.
        mutations_site : int32[]
            Site IDs for each mutation.
        mutations_node : int32[]
            Node IDs for each mutation.
        mutations_parent : int32[]
            Parent mutation IDs.
        mutations_time : float64[]
            Time values for each mutation.
        breakpoints : float64[]
            Genomic positions where trees change.
        indexes_edge_insertion_order : int32[]
            Order in which edges are inserted during tree building.
        indexes_edge_removal_order : int32[]
            Order in which edges are removed during tree building.

        """

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
            sites_ancestral_state,
            mutations_site,
            mutations_node,
            mutations_parent,
            mutations_time,
            mutations_derived_state,
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
            self.sites_ancestral_state = sites_ancestral_state
            self.mutations_site = mutations_site
            self.mutations_node = mutations_node
            self.mutations_parent = mutations_parent
            self.mutations_time = mutations_time
            self.mutations_derived_state = mutations_derived_state
            self.breakpoints = breakpoints

        def tree_position(self):
            """
            Create a :class:`NumbaTreePosition` for traversing this tree sequence.

            Returns
            -------
            NumbaTreePosition
                A new tree position initialized to the null tree.
                Use next() or prev() to move to actual tree positions.

            Examples
            --------
            >>> tree_pos = numba_ts.tree_position()
            >>> while tree_pos.next():
            ...     # Process current tree at tree_pos.index
            ...     print(f"Tree {tree_pos.index}: {tree_pos.interval}")
            """
            return NumbaTreePosition(self)

    tree_position_spec = [
        ("ts", NumbaTreeSequence.class_type.instance_type),
        ("index", numba.int32),
        ("direction", numba.int32),
        ("interval", numba.types.UniTuple(numba.float64, 2)),
        ("in_range", NumbaEdgeRange.class_type.instance_type),
        ("out_range", NumbaEdgeRange.class_type.instance_type),
        ("site_range", numba.types.UniTuple(numba.int32, 2)),
        ("mutation_range", numba.types.UniTuple(numba.int32, 2)),
    ]

    @numba.experimental.jitclass(tree_position_spec)
    class NumbaTreePosition:
        """
        Traverse trees in a numba compatible tree sequence.

        This class provides efficient forward and backward iteration through
        the trees in a tree sequence. It tracks the current position and interval,
        providing edge changes between trees.


        Attributes
        ----------
        ts : NumbaTreeSequence
            Reference to the tree sequence being traversed.
        index : int32
            Current tree index. -1 indicates no current tree (null state).
        direction : int32
            Traversal direction: tskit.FORWARD or tskit.REVERSE. tskit.NULL
            if uninitialised.
        interval : tuple of float64
            Genomic interval (left, right) covered by the current tree.
        in_range : NumbaEdgeRange
            Edges being added to form this current tree, relative to the last state
        out_range : NumbaEdgeRange
            Edges being removed to form this current tree, relative to the last state
        site_range : tuple of int32
            Range of sites in the current tree (start, stop).
        mutation_range : tuple of int32

        Example
        --------
        >>> tree_pos = numba_ts.tree_position()
        >>> num_edges
        >>> while tree_pos.next():
                num_edges += (tree_pos.in_range.stop - tree_pos.in_range.start)
                num_edges -= (tree_pos.out_range.stop - tree_pos.out_range.start)
                print(f"Tree {tree_pos.index}: {num_edges} edges")
        """

        def __init__(self, ts):
            self.ts = ts
            self.index = -1
            self.direction = 0
            self.interval = (0, 0)
            self.in_range = NumbaEdgeRange(0, 0, np.zeros(0, dtype=numba.int32))
            self.out_range = NumbaEdgeRange(0, 0, np.zeros(0, dtype=numba.int32))
            self.site_range = (0, 0)
            self.mutation_range = (0, 0)

        def set_null(self):
            """
            Reset the tree position to null state.
            """
            self.index = -1
            self.interval = (0, 0)
            self.site_range = (0, 0)
            self.mutation_range = (0, 0)

        def next(self):  # noqa: A003
            """
            Move to the next tree in forward direction.

            Updates the tree position to the next tree in the sequence,
            computing the edges that need to be added and removed to
            transform from the previous tree to the current tree, storing
            them in self.in_range and self.out_range.

            Returns
            -------
            bool
                True if successfully moved to next tree, False if the end
                of the tree sequence is reached.
                When False is returned, the iterator is in null state (index=-1).

            Notes
            -----
            On the first call, this initializes the iterator and moves to tree 0.
            The in_range and out_range attributes are updated to reflect the
            edge changes needed for the current tree.
            """
            M = self.ts.num_edges
            NS = self.ts.num_sites
            NM = self.ts.num_mutations
            breakpoints = self.ts.breakpoints
            left_coords = self.ts.edges_left
            left_order = self.ts.indexes_edge_insertion_order
            right_coords = self.ts.edges_right
            right_order = self.ts.indexes_edge_removal_order
            sites_position = self.ts.sites_position
            mutations_site = self.ts.mutations_site

            if self.index == -1:
                self.interval = (self.interval[0], 0)
                self.out_range.stop = 0
                self.in_range.stop = 0
                self.direction = FORWARD
                self.site_range = (0, 0)
                self.mutation_range = (0, 0)

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
                old_site_left, old_site_right = self.site_range
                j = old_site_right
                while j < NS and sites_position[j] < left:
                    j += 1
                self.site_range = (old_site_right, j)
                old_mutation_left, old_mutation_right = self.mutation_range
                k = old_mutation_right
                while k < NM and mutations_site[k] < j:
                    k += 1
                self.mutation_range = (old_mutation_right, k)

            return self.index != -1

        def prev(self):
            """
            Move to the previous tree in reverse direction.

            Updates the tree position to the previous tree in the sequence,
            computing the edges that need to be added and removed to
            transform from the next tree to the current tree, storing them
            in self.in_range and self.out_range

            Returns
            -------
            bool
                True if successfully moved to previous tree, False if the beginning
                of the tree sequence is reached.
                When False is returned, the iterator is in null state (index=-1).

            Notes
            -----
            On the first call, this initializes the iterator and moves to the most
            rightward tree.
            The in_range and out_range attributes are updated to reflect the
            edge changes needed for the current tree when traversing backward.
            """
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

    return NumbaTreeSequence


def numba_tree_sequence(ts):
    """
    Convert a TreeSequence to a Numba-compatible format.

    Creates a NumbaTreeSequence object that can be used within
    Numba-compiled functions.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to convert.

    Returns
    -------
    NumbaTreeSequence
        A Numba-compatible representation of the input tree sequence.
        Contains all necessary data arrays and metadata for tree traversal.
    """
    max_ancestral_length = max(1, max(map(len, ts.sites_ancestral_state), default=1))
    max_derived_length = max(1, max(map(len, ts.mutations_derived_state), default=1))

    return _numba_tree_sequence(max_ancestral_length, max_derived_length)(
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
        sites_ancestral_state=ts.sites_ancestral_state.astype(
            f"U{max_ancestral_length}"
        ),
        mutations_site=ts.mutations_site,
        mutations_node=ts.mutations_node,
        mutations_parent=ts.mutations_parent,
        mutations_time=ts.mutations_time,
        mutations_derived_state=ts.mutations_derived_state.astype(
            f"U{max_derived_length}"
        ),
        breakpoints=ts.breakpoints(as_array=True),
    )
