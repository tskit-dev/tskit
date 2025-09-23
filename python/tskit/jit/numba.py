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

parent_index_spec = [
    ("edge_index", numba.int32[:]),
    ("index_range", numba.int32[:, :]),
]


@numba.experimental.jitclass(edge_range_spec)
class EdgeRange:
    """
    Represents a range of edges during tree traversal.

    This class encapsulates information about a contiguous range of edges
    that are either being removed or added to step from one tree to another.
    The ``start`` and ``stop`` indices, when applied to the order array,
    define the ids of edges to process.

    Attributes
    ----------
    start : int
        Starting index of the edge range (inclusive).
    stop : int
        Stopping index of the edge range (exclusive).
    order : numpy.ndarray
        Array (dtype=np.int32) containing edge IDs in the order they should be processed.
        The edge ids in this range are order[start:stop].
    """

    def __init__(self, start, stop, order):
        self.start = start
        self.stop = stop
        self.order = order


@numba.experimental.jitclass(parent_index_spec)
class ParentIndex:
    """
    Simple data container for parent index information.

    This class provides access to all edges where a given node is the child.
    Since edges are not sorted by child in the tskit edge table, a custom index
    (edge_index) is built that sorts edge IDs by child node. `index_range`
    then contains the [start, stop) range of edges for each child node in `edge_index`.

    Attributes
    ----------
    edge_index : numpy.ndarray
        Array (dtype=np.int32) of edge IDs sorted by child node and left coordinate.
    index_range : numpy.ndarray
        Array (dtype=np.int32, shape=(num_nodes, 2)) where each row contains the
        [start, stop) range in edge_index where this node is the child.
    """

    def __init__(self, edge_index, index_range):
        self.edge_index = edge_index
        self.index_range = index_range


class TreeIndex:
    """
    Traverse trees in a numba compatible tree sequence.

    This class provides efficient forward and backward iteration through
    the trees in a tree sequence. It provides the tree interval,
    edge changes to create the current tree, along with its sites and mutations.
    A full pass over the trees using repeated `next` or `prev` requires O(E + M + S) time
    complexity.

    It should not be instantiated directly, but is returned by the `tree_index` method
    of `NumbaTreeSequence`.


    Attributes
    ----------
    ts : NumbaTreeSequence
        Reference to the tree sequence being traversed.
    index : int
        Current tree index. -1 indicates no current tree (null state).
    direction : int
        Traversal direction: tskit.FORWARD or tskit.REVERSE. tskit.NULL
        if uninitialised.
    interval : tuple
        Genomic interval (left, right) covered by the current tree.
    in_range : EdgeRange
        Edges being added to form this current tree, relative to the last state
    out_range : EdgeRange
        Edges being removed to form this current tree, relative to the last state
    site_range : tuple
        Range of sites in the current tree (start, stop).
    mutation_range : tuple
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
        self.in_range = EdgeRange(0, 0, np.zeros(0, dtype=np.int32))
        self.out_range = EdgeRange(0, 0, np.zeros(0, dtype=np.int32))
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
        transform from the previous tree to the current tree.
        On the first call, this initializes the iterator and moves to tree 0.

        :return: True if successfully moved to next tree, False if the end
            of the tree sequence is reached. When False is returned, the iterator
            is in null state (index=-1).
        :rtype: bool
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
        transform from the next tree to the current tree.
        On the first call, this initializes the iterator and moves to the most
        rightward tree.

        :return: True if successfully moved to previous tree, False if the beginning
            of the tree sequence is reached. When False is returned, the iterator
            is in null state (index=-1).
        :rtype: bool
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
    num_trees : int
        Number of trees in the tree sequence.
    num_nodes : int
        Number of nodes in the tree sequence.
    num_samples : int
        Number of samples in the tree sequence.
    num_edges : int
        Number of edges in the tree sequence.
    num_sites : int
        Number of sites in the tree sequence.
    num_mutations : int
        Number of mutations in the tree sequence.
    sequence_length : float
        Total sequence length of the tree sequence.
    edges_left : numpy.ndarray
        Array (dtype=np.float64) of left coordinates of edges.
    edges_right : numpy.ndarray
        Array (dtype=np.float64) of right coordinates of edges.
    edges_parent : numpy.ndarray
        Array (dtype=np.int32) of parent node IDs for each edge.
    edges_child : numpy.ndarray
        Array (dtype=np.int32) of child node IDs for each edge.
    nodes_time : numpy.ndarray
        Array (dtype=np.float64) of time values for each node.
    nodes_flags : numpy.ndarray
        Array (dtype=np.uint32) of flag values for each node.
    nodes_population : numpy.ndarray
        Array (dtype=np.int32) of population IDs for each node.
    nodes_individual : numpy.ndarray
        Array (dtype=np.int32) of individual IDs for each node.
    individuals_flags : numpy.ndarray
        Array (dtype=np.uint32) of flag values for each individual.
    sites_position : numpy.ndarray
        Array (dtype=np.float64) of positions of sites along the sequence.
    mutations_site : numpy.ndarray
        Array (dtype=np.int32) of site IDs for each mutation.
    mutations_node : numpy.ndarray
        Array (dtype=np.int32) of node IDs for each mutation.
    mutations_parent : numpy.ndarray
        Array (dtype=np.int32) of parent mutation IDs.
    mutations_time : numpy.ndarray
        Array (dtype=np.float64) of time values for each mutation.
    breakpoints : numpy.ndarray
        Array (dtype=np.float64) of genomic positions where trees change.
    indexes_edge_insertion_order : numpy.ndarray
        Array (dtype=np.int32) specifying the order in which edges are inserted
        during tree building.
    indexes_edge_removal_order : numpy.ndarray
        Array (dtype=np.int32) specifying the order in which edges are removed
        during tree building.

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
        mutations_inherited_state,
        breakpoints,
        max_ancestral_length,
        max_derived_length,
        max_inherited_length,
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
        self.mutations_inherited_state = mutations_inherited_state
        self.breakpoints = breakpoints
        self.max_ancestral_length = max_ancestral_length
        self.max_derived_length = max_derived_length
        self.max_inherited_length = max_inherited_length

    def tree_index(self):
        """
        Create a :class:`TreeIndex` for traversing this tree sequence.

        :return: A new tree index initialized to the null tree.
            Use next() or prev() to move to an actual tree.
        :rtype: TreeIndex
        """
        # This method will be overriden when the concrete JIT class TreeIndex
        # is defined in `jitwrap`.
        return TreeIndex(self)  # pragma: no cover

    def child_index(self):
        """
        Create child index array for finding child edges of nodes. This operation
        requires a linear pass over the edge table and therefore has a time
        complexity of O(E).

        :return: A numpy array (dtype=np.int32, shape=(num_nodes, 2)) where each row
            contains the [start, stop) range of edges where this node is the parent.
        :rtype: numpy.ndarray
        """
        child_range = np.full((self.num_nodes, 2), -1, dtype=np.int32)
        edges_parent = self.edges_parent
        if self.num_edges == 0:
            return child_range

        # Find ranges in tskit edge ordering
        last_parent = -1
        for edge_id in range(self.num_edges):
            parent = edges_parent[edge_id]
            if parent != last_parent:
                child_range[parent, 0] = edge_id
            if last_parent != -1:
                child_range[last_parent, 1] = edge_id
            last_parent = parent

        if last_parent != -1:
            child_range[last_parent, 1] = self.num_edges

        return child_range

    def parent_index(self):
        """
        Create a :class:`ParentIndex` for finding parent edges of nodes.

        Edges within each child's group are not guaranteed to be in any
        specific order. This operation uses a two-pass algorithm with
        O(N + E) time complexity and O(N) auxiliary space.

        :return: A new parent index container that can be used to
            efficiently find all edges where a given node is the child.
        :rtype: ParentIndex
        """
        num_nodes = self.num_nodes
        num_edges = self.num_edges
        edges_child = self.edges_child

        child_counts = np.zeros(num_nodes, dtype=np.int32)
        edge_index = np.zeros(num_edges, dtype=np.int32)
        index_range = np.zeros((num_nodes, 2), dtype=np.int32)

        if num_edges == 0:
            return ParentIndex(edge_index, index_range)

        # Count how many children each node has
        for child_node in edges_child:
            child_counts[child_node] += 1

        # From the counts build the index ranges, we set both the start and the
        # end index to the start - this lets us use the end index as a tracker
        # for where we should insert the next edge for that node - when all
        # edges are done these values will be the correct end values!
        current_start = 0
        for i in range(num_nodes):
            index_range[i, :] = current_start
            current_start += child_counts[i]

        # Now go over the edges, inserting them at the index pointed to
        # by the node's current end value, then increment.
        for edge_id in range(num_edges):
            child = edges_child[edge_id]
            pos = index_range[child, 1]
            edge_index[pos] = edge_id
            index_range[child, 1] += 1

        return ParentIndex(edge_index, index_range)


# We cache these classes to avoid repeated JIT compilation
@functools.lru_cache(None)
def _jitwrap(max_ancestral_length, max_derived_length, max_inherited_length):
    # We have a circular dependency in JIT compilation between NumbaTreeSequence
    # and NumbaTreeIndex so we used a deferred type to break it
    tree_sequence_type = numba.deferred_type()

    # We run this code on CI with this env var set so we can get coverage
    # of the jitted functions. EdgeRange doesn't have a class_type
    # in this case, so we skip the spec entirely.
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        tree_index_spec = []
    else:
        tree_index_spec = [
            ("ts", tree_sequence_type),
            ("index", numba.int32),
            ("direction", numba.int32),
            ("interval", numba.types.UniTuple(numba.float64, 2)),
            ("in_range", EdgeRange.class_type.instance_type),
            ("out_range", EdgeRange.class_type.instance_type),
            ("site_range", numba.types.UniTuple(numba.int32, 2)),
            ("mutation_range", numba.types.UniTuple(numba.int32, 2)),
        ]

    JittedTreeIndex = numba.experimental.jitclass(tree_index_spec)(TreeIndex)

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
        (
            "mutations_inherited_state",
            numba.types.UnicodeCharSeq(max_inherited_length)[:],
        ),
        ("breakpoints", numba.float64[:]),
        ("max_ancestral_length", numba.int32),
        ("max_derived_length", numba.int32),
        ("max_inherited_length", numba.int32),
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

    :param tskit.TreeSequence ts: The tree sequence to convert.
    :return: A Numba-compatible representation of the input tree sequence.
        Contains all necessary data arrays and metadata for tree traversal.
    :rtype: NumbaTreeSequence
    """
    max_ancestral_length = max(1, max(map(len, ts.sites_ancestral_state), default=1))
    max_derived_length = max(1, max(map(len, ts.mutations_derived_state), default=1))
    max_inherited_length = max(
        1, max(map(len, ts.mutations_inherited_state), default=1)
    )

    JittedTreeSequence = _jitwrap(
        max_ancestral_length, max_derived_length, max_inherited_length
    )

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
        mutations_inherited_state=ts.mutations_inherited_state.astype(
            f"U{max_inherited_length}"
        ),
        breakpoints=ts.breakpoints(as_array=True),
        max_ancestral_length=max_ancestral_length,
        max_derived_length=max_derived_length,
        max_inherited_length=max_inherited_length,
    )

    return numba_ts
