import functools
import os

import numpy as np

import tskit

try:
    import numba

except ImportError:
    raise ImportError(
        "Numba is not installed. Please install it with `pip install numba` "
        "or `conda install numba` to use the tskit.jit.numba module."
    )


FORWARD = 1  #: Direction constant for forward tree traversal
REVERSE = -1  #: Direction constant for reverse tree traversal

# Retrieve these here to avoid lookups in tight loops
NODE_IS_SAMPLE = tskit.NODE_IS_SAMPLE
NULL = tskit.NULL

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


class NumbaTreeIndex:
    """
    Traverse trees in a numba compatible tree sequence.

    This class provides efficient forward and backward iteration through
    the trees in a tree sequence. It provides the tree interval,
    edge changes to create the current tree, along with its sites and mutations.

    It should not be instantiated directly, but is returned by the `tree_index` method
    of `NumbaTreeSequence`.


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
        Range of mutations in the current tree (start, stop).

    Example
    --------
    >>> tree_index = numba_ts.tree_index()
    >>> num_edges = 0
    >>> while tree_index.next():
            num_edges += (tree_index.in_range.stop - tree_index.in_range.start)
            num_edges -= (tree_index.out_range.stop - tree_index.out_range.start)
            print(f"Tree {tree_index.index}: {num_edges} edges")
    """

    def __init__(self, ts):
        self.ts = ts
        self.index = -1
        self.direction = NULL
        self.interval = (0, 0)
        self.in_range = NumbaEdgeRange(0, 0, np.zeros(0, dtype=np.int32))
        self.out_range = NumbaEdgeRange(0, 0, np.zeros(0, dtype=np.int32))
        self.site_range = (0, 0)
        self.mutation_range = (0, 0)

    def set_null(self):
        """
        Reset the tree index to null state.
        """
        self.index = -1
        self.interval = (0, 0)
        self.site_range = (0, 0)
        self.mutation_range = (0, 0)

    def next(self):  # noqa: A003
        """
        Move to the next tree in forward direction.

        Updates the tree index to the next tree in the sequence,
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
            right = breakpoints[self.index + 1]
            self.interval = (left, right)

            # Find sites in current tree interval [left, right)
            old_site_left, old_site_right = self.site_range
            j = old_site_right
            while j < NS and sites_position[j] < right:
                j += 1
            self.site_range = (old_site_right, j)

            # Find mutations for sites in this interval
            old_mutation_left, old_mutation_right = self.mutation_range
            k = old_mutation_right
            while k < NM and mutations_site[k] < j:
                k += 1
            self.mutation_range = (old_mutation_right, k)

        return self.index != -1

    def prev(self):
        """
        Move to the previous tree in reverse direction.

        Updates the tree index to the previous tree in the sequence,
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
        NS = self.ts.num_sites
        NM = self.ts.num_mutations
        breakpoints = self.ts.breakpoints
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order
        sites_position = self.ts.sites_position
        mutations_site = self.ts.mutations_site

        if self.index == -1:
            self.index = self.ts.num_trees
            self.interval = (self.ts.sequence_length, self.interval[1])
            self.in_range.stop = M - 1
            self.out_range.stop = M - 1
            self.direction = REVERSE
            self.site_range = (NS, NS)
            self.mutation_range = (NM, NM)

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
            left = breakpoints[self.index]
            self.interval = (left, right)

            # Find sites in current tree interval [left, right) going backward
            old_site_left, old_site_right = self.site_range
            j = old_site_left - 1
            while j >= 0 and sites_position[j] >= left:
                j -= 1
            self.site_range = (j + 1, old_site_left)

            # Find mutations for sites in this interval going backward
            old_mutation_left, old_mutation_right = self.mutation_range
            k = old_mutation_left - 1
            while k >= 0 and mutations_site[k] >= self.site_range[0]:
                k -= 1
            self.mutation_range = (k + 1, old_mutation_left)

        return self.index != -1


class NumbaTreeSequence:
    """
    A Numba-compatible representation of a tree sequence.

    This class provides access a tree sequence class that can be used
    from within Numba "njit" compiled functions. :meth:`jitwrap` should
    be used to JIT compile this class from a :class:`tskit.TreeSequence` object,
    before it is passed to a Numba function.

    Attributes
    ----------
    num_trees : int32
        Number of trees in the tree sequence.
    num_nodes : int32
        Number of nodes in the tree sequence.
    num_samples : int32
        Number of samples in the tree sequence.
    num_edges : int32
        Number of edges in the tree sequence.
    num_sites : int32
        Number of sites in the tree sequence.
    num_mutations : int32
        Number of mutations in the tree sequence.
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
        num_nodes,
        num_samples,
        num_edges,
        num_sites,
        num_mutations,
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
        max_ancestral_length,
        max_derived_length,
    ):
        self.num_trees = num_trees
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.num_edges = num_edges
        self.num_sites = num_sites
        self.num_mutations = num_mutations
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
        self.max_ancestral_length = max_ancestral_length
        self.max_derived_length = max_derived_length

    def tree_index(self):
        """
        Create a :class:`NumbaTreeIndex` for traversing this tree sequence.

        Returns
        -------
        NumbaTreeIndex
            A new tree index initialized to the null tree.
            Use next() or prev() to move to an actual tree.

        Examples
        --------
        >>> tree_index = numba_ts.tree_index()
        >>> while tree_index.next():
        ...     # Process current tree at tree_index.index
        ...     print(f"Tree {tree_index.index}: {tree_index.interval}")
        """
        # This method will be overriden when the concrete JIT class NumbaTreeIndex
        # is defined in `jitwrap`.
        return NumbaTreeIndex(self)  # pragma: no cover


# We cache these classes to avoid repeated JIT compilation
@functools.lru_cache(None)
def _jitwrap(max_ancestral_length, max_derived_length):
    # We have a circular dependency in JIT compilation between NumbaTreeSequence
    # and NumbaTreeIndex so we used a deferred type to break it
    tree_sequence_type = numba.deferred_type()

    # We run this code on CI with this env var set so we can get coverage
    # of the jitted functions. NumbaEdgeRange doesn't have a class_type
    # in this case, so we skip the spec entirely.
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        tree_index_spec = []
    else:
        tree_index_spec = [
            ("ts", tree_sequence_type),
            ("index", numba.int32),
            ("direction", numba.int32),
            ("interval", numba.types.UniTuple(numba.float64, 2)),
            ("in_range", NumbaEdgeRange.class_type.instance_type),
            ("out_range", NumbaEdgeRange.class_type.instance_type),
            ("site_range", numba.types.UniTuple(numba.int32, 2)),
            ("mutation_range", numba.types.UniTuple(numba.int32, 2)),
        ]
    JittedTreeIndex = numba.experimental.jitclass(tree_index_spec)(NumbaTreeIndex)

    tree_sequence_spec = [
        ("num_trees", numba.int32),
        ("num_nodes", numba.int32),
        ("num_samples", numba.int32),
        ("num_edges", numba.int32),
        ("num_sites", numba.int32),
        ("num_mutations", numba.int32),
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
        ("max_ancestral_length", numba.int32),
        ("max_derived_length", numba.int32),
    ]

    # The `tree_index` method on NumbaTreeSequence uses NumbaTreeIndex
    # which is the uncompiled version of the class. The compiled version isn't
    # known till now, so replace the method with this definition.

    class _NumbaTreeSequence(NumbaTreeSequence):
        def tree_index(self):
            return JittedTreeIndex(self)

    JittedTreeSequence = numba.experimental.jitclass(tree_sequence_spec)(
        _NumbaTreeSequence
    )

    # Now both classes are setup we can resolve the deferred type
    if os.environ.get("NUMBA_DISABLE_JIT") != "1":
        tree_sequence_type.define(JittedTreeSequence.class_type.instance_type)

    return JittedTreeSequence


def jitwrap(ts):
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

    JittedTreeSequence = _jitwrap(max_ancestral_length, max_derived_length)

    # Create the tree sequence instance
    numba_ts = JittedTreeSequence(
        num_trees=ts.num_trees,
        num_nodes=ts.num_nodes,
        num_samples=ts.num_samples,
        num_edges=ts.num_edges,
        num_sites=ts.num_sites,
        num_mutations=ts.num_mutations,
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
        max_ancestral_length=max_ancestral_length,
        max_derived_length=max_derived_length,
    )

    return numba_ts
