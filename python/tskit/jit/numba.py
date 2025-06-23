from dataclasses import dataclass


try:
    import numba
except ImportError:
    raise ImportError(
        "Numba is not installed. Please install it with `pip install numba` "
        "or `conda install numba` to use the tskit.jit.numba module."
    )


# Decorator that makes a jited dataclass by removing certain methods
# that are not compatible with Numba's JIT compilation.
def jitdataclass(cls):
    dc_cls = dataclass(cls, eq=False)
    del dc_cls.__dataclass_params__
    del dc_cls.__dataclass_fields__
    del dc_cls.__repr__
    try:
        del dc_cls.__replace__
    except AttributeError:
        # __replace__ is not available in Python < 3.10
        pass
    try:
        del dc_cls.__match_args__
    except AttributeError:
        # __match_args__ is not available in Python < 3.10
        pass
    return numba.experimental.jitclass(dc_cls)


@jitdataclass
class NumbaTreeSequence:
    num_edges: numba.int64
    sequence_length: numba.float64
    edges_left: numba.float64[:]
    edges_right: numba.float64[:]
    indexes_edge_insertion_order: numba.int32[:]
    indexes_edge_removal_order: numba.int32[:]

    def tree_position(self):
        return NumbaTreePosition(self, (0, 0), (0, 0), (0, 0))


@jitdataclass
class NumbaTreePosition:
    ts: NumbaTreeSequence
    interval: numba.types.UniTuple(numba.float64, 2)
    edges_in_index_range: numba.types.UniTuple(numba.int32, 2)
    edges_out_index_range: numba.types.UniTuple(numba.int32, 2)

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
    )
