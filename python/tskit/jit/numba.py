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
class NumbaEdgeDiff:
    interval: numba.types.UniTuple(numba.float64, 2)
    edges_in_index_range: numba.types.UniTuple(numba.int32, 2)
    edges_out_index_range: numba.types.UniTuple(numba.int32, 2)


@jitdataclass
class NumbaTreeSequence:
    num_edges: numba.int64
    sequence_length: numba.float64
    edges_left: numba.float64[:]
    edges_right: numba.float64[:]
    indexes_edge_insertion_order: numba.int32[:]
    indexes_edge_removal_order: numba.int32[:]

    def edge_diffs(self, include_terminal=False):
        left = 0.0
        j = 0
        k = 0
        edges_left = self.edges_left
        edges_right = self.edges_right
        in_order = self.indexes_edge_insertion_order
        out_order = self.indexes_edge_removal_order

        while j < self.num_edges or left < self.sequence_length:
            in_start = j
            out_start = k

            while k < self.num_edges and edges_right[out_order[k]] == left:
                k += 1
            while j < self.num_edges and edges_left[in_order[j]] == left:
                j += 1
            in_end = j
            out_end = k

            right = self.sequence_length
            if j < self.num_edges:
                right = min(right, edges_left[in_order[j]])
            if k < self.num_edges:
                right = min(right, edges_right[out_order[k]])

            yield NumbaEdgeDiff((left, right), (in_start, in_end), (out_start, out_end))

            left = right

        # Handle remaining edges that haven't been processed
        if include_terminal:
            yield NumbaEdgeDiff((left, right), (j, j), (k, self.num_edges))


def numba_tree_sequence(ts):
    return NumbaTreeSequence(
        num_edges=ts.num_edges,
        sequence_length=ts.sequence_length,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        indexes_edge_insertion_order=ts.indexes_edge_insertion_order,
        indexes_edge_removal_order=ts.indexes_edge_removal_order,
    )
