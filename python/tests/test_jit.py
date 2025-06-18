import itertools
import sys
from unittest.mock import patch

import msprime
import numba
import numpy as np
import pytest

import tests.tsutil as tsutil


def test_numba_import_error():
    # Mock numba as not available
    with patch.dict(sys.modules, {"numba": None}):
        with pytest.raises(ImportError, match="pip install numba"):
            import tskit.jit.numba  # noqa: F401


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_correct_trees_forward(ts):
    import tskit.jit.numba as jit_numba

    numba_ts = jit_numba.numba_tree_sequence(ts)
    in_index = ts.indexes_edge_insertion_order
    out_index = ts.indexes_edge_removal_order
    for numba_edge_diff, edge_diff in itertools.zip_longest(
        numba_ts.edge_diffs(), ts.edge_diffs()
    ):
        assert edge_diff.interval == numba_edge_diff.interval
        for edge_in_index, edge in itertools.zip_longest(
            range(*numba_edge_diff.edges_in_index_range), edge_diff.edges_in
        ):
            assert edge.id == in_index[edge_in_index]
        for edge_out_index, edge in itertools.zip_longest(
            range(*numba_edge_diff.edges_out_index_range), edge_diff.edges_out
        ):
            assert edge.id == out_index[edge_out_index]


def test_using_from_jit_function():
    """
    Test that we can use the numba jit function from the tskit.jit module.
    """
    import tskit.jit.numba as jit_numba

    ts = msprime.sim_ancestry(
        samples=10, sequence_length=100, recombination_rate=1, random_seed=42
    )

    @numba.njit
    def _coalescent_nodes_numba(numba_ts, num_nodes, edges_parent):
        is_coalescent = np.zeros(num_nodes, dtype=np.int8)
        num_children = np.zeros(num_nodes, dtype=np.int64)
        for tree_pos in numba_ts.edge_diffs():
            for j in range(*tree_pos.edges_out_index_range):
                e = numba_ts.indexes_edge_removal_order[j]
                num_children[edges_parent[e]] -= 1
            for j in range(*tree_pos.edges_in_index_range):
                e = numba_ts.indexes_edge_insertion_order[j]
                p = edges_parent[e]
                num_children[p] += 1
                if num_children[p] == 2:
                    is_coalescent[p] = True
        return is_coalescent

    def coalescent_nodes_python(ts):
        is_coalescent = np.zeros(ts.num_nodes, dtype=bool)
        num_children = np.zeros(ts.num_nodes, dtype=int)
        for _, edges_out, edges_in in ts.edge_diffs():
            for e in edges_out:
                num_children[e.parent] -= 1
            for e in edges_in:
                num_children[e.parent] += 1
                if num_children[e.parent] == 2:
                    # Num_children will always be exactly two once, even arity is greater
                    is_coalescent[e.parent] = True
        return is_coalescent

    numba_ts = jit_numba.numba_tree_sequence(ts)
    C1 = coalescent_nodes_python(ts)
    C2 = _coalescent_nodes_numba(numba_ts, ts.num_nodes, ts.edges_parent)

    np.testing.assert_array_equal(C1, C2)
