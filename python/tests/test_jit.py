import itertools
import sys

import msprime
import numba
import numpy as np
import numpy.testing as nt
import pytest

import tests.tsutil as tsutil
import tskit
import tskit.jit.numba as jit_numba


def test_numba_import_error():
    # Make the modules unavailable temporarily
    original_numba = sys.modules.get("numba")
    original_jit_numba = sys.modules.get("tskit.jit.numba")
    try:
        if "numba" in sys.modules:
            del sys.modules["numba"]
        if "tskit.jit.numba" in sys.modules:
            del sys.modules["tskit.jit.numba"]

        # Mock numba as not available at all
        sys.modules["numba"] = None
        with pytest.raises(ImportError, match="pip install numba"):
            import tskit.jit.numba  # noqa: F401
    finally:
        # Restore original modules
        sys.modules["numba"] = original_numba
        sys.modules["tskit.jit.numba"] = original_jit_numba


def _verify_tree_index_state(tree_index, edge_diff, tree, reverse=False):
    assert edge_diff.interval == tree_index.interval

    if reverse:
        edge_range = range(tree_index.in_range.start, tree_index.in_range.stop, -1)
    else:
        edge_range = range(tree_index.in_range.start, tree_index.in_range.stop)

    for edge_in_index, edge in itertools.zip_longest(edge_range, edge_diff.edges_in):
        assert edge.id == tree_index.in_range.order[edge_in_index]

    if reverse:
        edge_range = range(tree_index.out_range.start, tree_index.out_range.stop, -1)
    else:
        edge_range = range(tree_index.out_range.start, tree_index.out_range.stop)

    for edge_out_index, edge in itertools.zip_longest(edge_range, edge_diff.edges_out):
        assert edge.id == tree_index.out_range.order[edge_out_index]

    sites = [s.id for s in tree.sites()]
    if len(sites) > 0:
        assert tree_index.site_range == (min(sites), max(sites) + 1)
    else:
        assert tree_index.site_range[0] == tree_index.site_range[1]

    muts = [m.id for m in tree.mutations()]
    if len(muts) > 0:
        assert tree_index.mutation_range == (min(muts), max(muts) + 1)
    else:
        assert tree_index.mutation_range[0] == tree_index.mutation_range[1]


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_correct_trees_forward(ts):
    numba_ts = jit_numba.jitwrap(ts)
    tree_index = numba_ts.tree_index()
    ts_edge_diffs = ts.edge_diffs()
    tree = ts.first()
    while tree_index.next():
        edge_diff = next(ts_edge_diffs)
        _verify_tree_index_state(tree_index, edge_diff, tree, reverse=False)
        last_tree = not tree.next()
    assert last_tree


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_correct_trees_backwards(ts):
    numba_ts = jit_numba.jitwrap(ts)
    tree_index = numba_ts.tree_index()
    ts_edge_diffs = ts.edge_diffs(direction=tskit.REVERSE)
    tree = ts.last()
    while tree_index.prev():
        edge_diff = next(ts_edge_diffs)
        _verify_tree_index_state(tree_index, edge_diff, tree, reverse=True)
        last_tree = not tree.prev()
    assert last_tree


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_correct_trees_backwards_and_forwards(ts):
    numba_ts = jit_numba.jitwrap(ts)
    tree_index = numba_ts.tree_index()
    ts_edge_diffs = ts.edge_diffs(direction=tskit.REVERSE)
    tree = ts.last()
    while tree_index.prev():
        edge_diff = next(ts_edge_diffs)
        _verify_tree_index_state(tree_index, edge_diff, tree, reverse=True)
        last_tree = not tree.prev()
    assert last_tree
    tree = ts.first()
    ts_edge_diffs = ts.edge_diffs()
    while tree_index.next():
        edge_diff = next(ts_edge_diffs)
        _verify_tree_index_state(tree_index, edge_diff, tree, reverse=False)
        last_tree = not tree.next()
    assert last_tree
    tree = ts.last()
    ts_edge_diffs = ts.edge_diffs(direction=tskit.REVERSE)
    while tree_index.prev():
        edge_diff = next(ts_edge_diffs)
        _verify_tree_index_state(tree_index, edge_diff, tree, reverse=True)
        last_tree = not tree.prev()
    assert last_tree


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_child_index_correctness(ts):
    numba_ts = jit_numba.jitwrap(ts)
    child_index = numba_ts.child_index()
    for node in range(ts.num_nodes):
        start, stop = child_index[node]

        expected_children = []
        for edge_id in range(ts.num_edges):
            if ts.edges_parent[edge_id] == node:
                expected_children.append(edge_id)

        if len(expected_children) == 0:
            assert start == -1 and stop == -1
        else:
            assert stop > start
            actual_children = list(range(start, stop))
            for edge_id in actual_children:
                assert ts.edges_parent[edge_id] == node
            assert actual_children == expected_children


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_parent_index_correctness(ts):
    numba_ts = jit_numba.jitwrap(ts)
    parent_index = numba_ts.parent_index()
    for node in range(ts.num_nodes):
        start, stop = parent_index.index_range[node]

        expected_parents = []
        for edge_id in range(ts.num_edges):
            if ts.edges_child[edge_id] == node:
                expected_parents.append(edge_id)

        if len(expected_parents) == 0:
            assert start == stop
        else:
            assert stop > start
            actual_parent_edge_ids = []
            for j in range(start, stop):
                edge_id = parent_index.edge_index[j]
                actual_parent_edge_ids.append(edge_id)
                assert ts.edges_child[edge_id] == node
            assert set(actual_parent_edge_ids) == set(expected_parents)


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_parent_index_tree_reconstruction(ts):
    numba_ts = jit_numba.jitwrap(ts)
    parent_index = numba_ts.parent_index()

    # Test tree reconstruction at all breakpoints
    for tree in ts.trees():
        position = tree.interval.left + 0.5 * tree.span
        reconstructed_parent = np.full(ts.num_nodes, -1, dtype=np.int32)
        for node in range(ts.num_nodes):
            start, stop = parent_index.index_range[node]
            if start != -1:
                for j in range(start, stop):
                    edge_id = parent_index.edge_index[j]
                    if ts.edges_left[edge_id] <= position < ts.edges_right[edge_id]:
                        reconstructed_parent[node] = ts.edges_parent[edge_id]
                        break
        expected_parent = tree.parent_array

        # Compare parent arrays (excluding virtual root)
        nt.assert_array_equal(
            reconstructed_parent,
            expected_parent[:-1],
        )


def test_child_parent_index_from_jit_function():
    ts = msprime.sim_ancestry(
        samples=10, sequence_length=100, recombination_rate=1, random_seed=42
    )

    @numba.njit
    def _count_children_parents_numba(numba_ts):
        child_index = numba_ts.child_index()
        parent_index = numba_ts.parent_index()

        total_child_edges = 0
        total_parent_edges = 0

        for node in range(numba_ts.num_nodes):
            # Count child edges
            child_start, child_stop = child_index[node]
            if child_start != -1:
                total_child_edges += child_stop - child_start

            # Count parent edges
            parent_start, parent_stop = parent_index.index_range[node]
            if parent_start != -1:
                total_parent_edges += parent_stop - parent_start

        return total_child_edges, total_parent_edges

    def count_children_parents_python(ts):
        total_child_edges = 0
        total_parent_edges = 0

        for node in range(ts.num_nodes):
            # Count child edges
            for edge in ts.edges():
                if edge.parent == node:
                    total_child_edges += 1
                if edge.child == node:
                    total_parent_edges += 1

        return total_child_edges, total_parent_edges

    numba_ts = jit_numba.jitwrap(ts)
    numba_result = _count_children_parents_numba(numba_ts)
    python_result = count_children_parents_python(ts)

    assert numba_result == python_result


def test_using_tree_index_from_jit_function():
    # Test we can use from a numba jitted function

    ts = msprime.sim_ancestry(
        samples=10, sequence_length=100, recombination_rate=1, random_seed=42
    )

    @numba.njit
    def _coalescent_nodes_numba(numba_ts, num_nodes, edges_parent):
        is_coalescent = np.zeros(num_nodes, dtype=np.int8)
        num_children = np.zeros(num_nodes, dtype=np.int64)
        tree_index = numba_ts.tree_index()
        while tree_index.next():
            for j in range(tree_index.out_range.start, tree_index.out_range.stop):
                e = tree_index.out_range.order[j]
                num_children[edges_parent[e]] -= 1
            for j in range(tree_index.in_range.start, tree_index.in_range.stop):
                e = tree_index.in_range.order[j]
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

    numba_ts = jit_numba.jitwrap(ts)
    C1 = coalescent_nodes_python(ts)
    C2 = _coalescent_nodes_numba(numba_ts, ts.num_nodes, ts.edges_parent)

    nt.assert_array_equal(C1, C2)


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_jit_diversity(ts):
    if ts.num_samples < 1:
        pytest.skip(
            "Tree sequence must have at least one sample for diversity calculation"
        )

    @numba.njit
    def diversity(numba_ts):
        # Cache arrays to avoid repeated attribute access in
        # tight loops
        edge_child = numba_ts.edges_child
        edge_parent = numba_ts.edges_parent
        node_times = numba_ts.nodes_time
        node_flags = numba_ts.nodes_flags

        if numba_ts.num_samples <= 1:
            return 0.0

        parent = np.full(numba_ts.num_nodes, -1, dtype=np.int32)
        branch_length = np.zeros(numba_ts.num_nodes, dtype=np.float64)
        state = np.zeros(numba_ts.num_nodes, dtype=np.int32)
        summary = np.zeros(numba_ts.num_nodes, dtype=np.float64)

        n = float(numba_ts.num_samples)
        two_over_denom = 2.0 / (n * (n - 1.0))
        sample_summary = 2.0 / n

        # Retrieve this constant outside the loop
        # to avoid repeated attribute access
        NODE_IS_SAMPLE = tskit.NODE_IS_SAMPLE
        # Find the sample nodes and initialize their states
        for node in range(numba_ts.num_nodes):
            if node_flags[node] & NODE_IS_SAMPLE:
                state[node] = 1.0
                summary[node] = sample_summary

        result = 0.0
        running_sum = 0.0
        tree_index = numba_ts.tree_index()

        # Now iterate through the trees
        while tree_index.next():
            # Process the outgoing edges
            for j in range(tree_index.out_range.start, tree_index.out_range.stop):
                h = tree_index.out_range.order[j]
                child = edge_child[h]
                child_parent = edge_parent[h]

                running_sum -= branch_length[child] * summary[child]
                parent[child] = -1
                branch_length[child] = 0.0

                u = child_parent
                parent_u = parent[u]
                while u != -1:
                    running_sum -= branch_length[u] * summary[u]
                    state[u] -= state[child]
                    summary[u] = state[u] * (n - state[u]) * two_over_denom
                    running_sum += branch_length[u] * summary[u]
                    u = parent_u
                    if u != -1:
                        parent_u = parent[u]

            # Process the incoming edges
            for j in range(tree_index.in_range.start, tree_index.in_range.stop):
                h = tree_index.in_range.order[j]
                child = edge_child[h]
                child_parent = edge_parent[h]

                parent[child] = child_parent
                branch_length[child] = node_times[child_parent] - node_times[child]
                running_sum += branch_length[child] * summary[child]

                u = child_parent
                parent_u = parent[u]
                while u != -1:
                    running_sum -= branch_length[u] * summary[u]
                    state[u] += state[child]
                    summary[u] = state[u] * (n - state[u]) * two_over_denom
                    running_sum += branch_length[u] * summary[u]
                    u = parent_u
                    if u != -1:
                        parent_u = parent[u]

            result += running_sum * (tree_index.interval[1] - tree_index.interval[0])

        return result / numba_ts.sequence_length

    numba_ts = jit_numba.jitwrap(ts)
    diversity_numba = diversity(numba_ts)
    diversity_python = ts.diversity(mode="branch")

    assert diversity_numba == pytest.approx(diversity_python, rel=1e-5)


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_jitwrap_properties(ts):
    numba_ts = jit_numba.jitwrap(ts)

    assert numba_ts.num_trees == ts.num_trees
    assert numba_ts.num_edges == ts.num_edges
    assert numba_ts.sequence_length == ts.sequence_length
    assert numba_ts.num_nodes == ts.num_nodes
    assert numba_ts.num_samples == ts.num_samples
    assert numba_ts.num_sites == ts.num_sites
    assert numba_ts.num_mutations == ts.num_mutations

    nt.assert_array_equal(numba_ts.edges_left, ts.edges_left)
    nt.assert_array_equal(numba_ts.edges_right, ts.edges_right)
    nt.assert_array_equal(numba_ts.edges_parent, ts.edges_parent)
    nt.assert_array_equal(numba_ts.edges_child, ts.edges_child)
    assert numba_ts.edges_left.dtype == np.float64
    assert numba_ts.edges_right.dtype == np.float64
    assert numba_ts.edges_parent.dtype == np.int32
    assert numba_ts.edges_child.dtype == np.int32
    nt.assert_array_equal(numba_ts.nodes_time, ts.nodes_time)
    nt.assert_array_equal(numba_ts.nodes_flags, ts.nodes_flags)
    nt.assert_array_equal(numba_ts.nodes_population, ts.nodes_population)
    nt.assert_array_equal(numba_ts.nodes_individual, ts.nodes_individual)
    assert numba_ts.nodes_time.dtype == np.float64
    assert numba_ts.nodes_flags.dtype == np.uint32
    assert numba_ts.nodes_population.dtype == np.int32
    assert numba_ts.nodes_individual.dtype == np.int32
    nt.assert_array_equal(numba_ts.individuals_flags, ts.individuals_flags)
    assert numba_ts.individuals_flags.dtype == np.uint32
    nt.assert_array_equal(numba_ts.sites_position, ts.sites_position)
    assert numba_ts.sites_position.dtype == np.float64
    nt.assert_array_equal(numba_ts.sites_ancestral_state, ts.sites_ancestral_state)
    assert numba_ts.sites_ancestral_state.dtype.kind == "U"  # Unicode string
    nt.assert_array_equal(numba_ts.mutations_site, ts.mutations_site)
    nt.assert_array_equal(numba_ts.mutations_node, ts.mutations_node)
    nt.assert_array_equal(numba_ts.mutations_parent, ts.mutations_parent)
    nt.assert_array_equal(numba_ts.mutations_time, ts.mutations_time)
    assert numba_ts.mutations_site.dtype == np.int32
    assert numba_ts.mutations_node.dtype == np.int32
    assert numba_ts.mutations_parent.dtype == np.int32
    assert numba_ts.mutations_time.dtype == np.float64
    nt.assert_array_equal(numba_ts.mutations_derived_state, ts.mutations_derived_state)
    assert numba_ts.mutations_derived_state.dtype.kind == "U"  # Unicode string
    nt.assert_array_equal(
        numba_ts.mutations_inherited_state, ts.mutations_inherited_state
    )
    assert numba_ts.mutations_inherited_state.dtype.kind == "U"  # Unicode string
    nt.assert_array_equal(
        numba_ts.indexes_edge_insertion_order, ts.indexes_edge_insertion_order
    )
    nt.assert_array_equal(
        numba_ts.indexes_edge_removal_order, ts.indexes_edge_removal_order
    )
    assert numba_ts.indexes_edge_insertion_order.dtype == np.int32
    assert numba_ts.indexes_edge_removal_order.dtype == np.int32
    assert numba_ts.breakpoints.dtype == np.float64
    nt.assert_array_equal(numba_ts.breakpoints, ts.breakpoints(as_array=True))


def test_numba_edge_range():

    order = np.array([1, 3, 2, 0], dtype=np.int32)
    edge_range = jit_numba.EdgeRange(start=1, stop=3, order=order)

    assert edge_range.start == 1
    assert edge_range.stop == 3
    nt.assert_array_equal(edge_range.order, order)


def test_numba_tree_index_set_null(ts_fixture):

    numba_ts = jit_numba.jitwrap(ts_fixture)
    tree_index = numba_ts.tree_index()

    # Move to a valid position first
    tree_index.next()
    initial_interval = tree_index.interval
    assert tree_index.index != -1
    assert initial_interval != (0, 0)

    # Test set_null
    tree_index.set_null()
    assert tree_index.index == -1
    assert tree_index.interval == (0, 0)


def test_numba_tree_index_constants(ts_fixture):

    numba_ts = jit_numba.jitwrap(ts_fixture)
    tree_index = numba_ts.tree_index()

    # Initial direction should be 0
    assert tree_index.direction == tskit.NULL

    # After next(), direction should be FORWARD
    tree_index.next()
    assert tree_index.direction == jit_numba.FORWARD
    assert tree_index.direction == 1

    # After prev(), direction should be REVERSE
    tree_index.prev()
    assert tree_index.direction == jit_numba.REVERSE
    assert tree_index.direction == -1

    # Test mixed direction
    tree_index.set_null()
    tree_index.prev()
    assert tree_index.direction == jit_numba.REVERSE
    tree_index.next()
    assert tree_index.direction == jit_numba.FORWARD


def test_numba_tree_index_edge_cases():

    # Test with empty tree sequence
    tables = tskit.TableCollection(sequence_length=1.0)
    empty_ts = tables.tree_sequence()
    numba_ts = jit_numba.jitwrap(empty_ts)
    tree_index = numba_ts.tree_index()

    # Should have exactly one tree
    assert tree_index.next()
    assert tree_index.index == 0
    assert tree_index.interval == (0.0, 1.0)
    assert not tree_index.next()  # No more trees
    assert tree_index.index == -1

    # Test with single tree (with edges)
    ts = msprime.sim_ancestry(samples=2, random_seed=42)  # No recombination
    numba_ts = jit_numba.jitwrap(ts)
    tree_index = numba_ts.tree_index()

    # Should have exactly one tree
    assert tree_index.next()
    assert tree_index.index == 0
    assert not tree_index.next()  # No more trees
    assert tree_index.index == -1


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_jit_descendant_span(ts):
    if ts.num_nodes == 0:
        pytest.skip("Tree sequence must have at least one node")

    @numba.njit
    def descendant_span(numba_ts, u):
        child_index = numba_ts.child_index()
        edges_left = numba_ts.edges_left
        edges_right = numba_ts.edges_right
        edges_child = numba_ts.edges_child

        total_descending = np.zeros(numba_ts.num_nodes)
        stack = [(u, 0.0, numba_ts.sequence_length)]

        while len(stack) > 0:
            node, left, right = stack.pop()
            total_descending[node] += right - left

            # Find all child edges for this node
            for e in range(child_index[node, 0], child_index[node, 1]):
                e_left = edges_left[e]
                e_right = edges_right[e]

                # Check if edge overlaps with current interval
                if e_right > left and right > e_left:
                    inter_left = max(e_left, left)
                    inter_right = min(e_right, right)
                    e_child = edges_child[e]
                    stack.append((e_child, inter_left, inter_right))

        return total_descending

    def descendant_span_tree(ts, u):
        total_descending = np.zeros(ts.num_nodes)
        for tree in ts.trees():
            descendants = tree.preorder(u)
            total_descending[descendants] += tree.span
        return total_descending

    numba_ts = jit_numba.jitwrap(ts)
    for u in range(ts.num_nodes):
        d1 = descendant_span(numba_ts, u)
        d2 = descendant_span_tree(ts, u)
        nt.assert_array_almost_equal(d1, d2, decimal=10)


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_jit_descendant_edges(ts):
    if ts.num_nodes == 0:
        pytest.skip("Tree sequence must have at least one node")

    @numba.njit
    def descendant_edges(numba_ts, u):
        """
        Returns a boolean array which is only True for edges that
        are descendants of node u.
        """
        edge_select = np.zeros(numba_ts.num_edges, dtype=np.bool_)
        child_index = numba_ts.child_index()
        edges_left = numba_ts.edges_left
        edges_right = numba_ts.edges_right
        edges_child = numba_ts.edges_child

        # The stack stores (node_id, left_coord, right_coord)
        stack = [(u, 0.0, numba_ts.sequence_length)]

        while len(stack) > 0:
            node, left, right = stack.pop()

            start, stop = child_index[node]
            for e in range(start, stop):
                e_left = edges_left[e]
                e_right = edges_right[e]

                if e_right > left and right > e_left:
                    edge_select[e] = True
                    inter_left = max(e_left, left)
                    inter_right = min(e_right, right)
                    e_child = edges_child[e]
                    stack.append((e_child, inter_left, inter_right))

        return edge_select

    def descendant_edges_tskit(ts, start_node):
        D = np.zeros(ts.num_edges, dtype=bool)
        for tree in ts.trees():
            for v in tree.preorder(start_node):
                # We want the edges *below* the start_node, so we skip the node itself.
                if v != start_node:
                    D[tree.edge(v)] = True
        return D

    numba_ts = jit_numba.jitwrap(ts)
    for u in range(ts.num_nodes):
        d1 = descendant_edges(numba_ts, u)
        d2 = descendant_edges_tskit(ts, u)
        nt.assert_array_equal(d1, d2)


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
def test_jit_ancestral_edges(ts):
    if ts.num_nodes == 0:
        pytest.skip("Tree sequence must have at least one node")

    @numba.njit
    def ancestral_edges(numba_ts, u):
        """
        Returns a boolean array which is only True for edges that are
        ancestors of node u.
        """
        edge_select = np.zeros(numba_ts.num_edges, dtype=np.bool_)
        parent_index = numba_ts.parent_index()
        edges_left = numba_ts.edges_left
        edges_right = numba_ts.edges_right
        edges_parent = numba_ts.edges_parent

        # The stack stores (node_id, left_coord, right_coord)
        stack = [(u, 0.0, numba_ts.sequence_length)]

        while len(stack) > 0:
            node, left, right = stack.pop()

            start, stop = parent_index.index_range[node]
            for i in range(start, stop):
                e = parent_index.edge_index[i]
                e_left = edges_left[e]
                e_right = edges_right[e]

                if e_right > left and right > e_left:
                    edge_select[e] = True
                    inter_left = max(e_left, left)
                    inter_right = min(e_right, right)
                    e_parent = edges_parent[e]
                    stack.append((e_parent, inter_left, inter_right))

        return edge_select

    def ancestral_edges_tskit(ts, start_node):
        A = np.zeros(ts.num_edges, dtype=bool)
        for tree in ts.trees():
            curr_node = start_node
            parent = tree.parent(curr_node)
            while parent != tskit.NULL:
                edge_id = tree.edge(curr_node)
                A[edge_id] = True
                curr_node = parent
                parent = tree.parent(curr_node)
        return A

    numba_ts = jit_numba.jitwrap(ts)
    for u in range(ts.num_nodes):
        a1 = ancestral_edges(numba_ts, u)
        a2 = ancestral_edges_tskit(ts, u)
        nt.assert_array_equal(a1, a2)
