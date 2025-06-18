import itertools
import sys
from unittest.mock import patch

import pytest

import tests.tsutil as tsutil


def test_numba_import_error():
    # Mock numba as not available
    with patch.dict(sys.modules, {"numba": None}):
        with pytest.raises(ImportError, match="pip install numba"):
            import tskit.jit.numba  # noqa: F401


@pytest.mark.numba
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
