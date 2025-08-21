---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{currentmodule} tskit.jit.numba
```

(sec_numba)=

# Numba Integration

The `tskit.jit.numba` module provides classes for working with tree sequences
from [Numba](https://numba.pydata.org/) jit-compiled Python code. Such code can run
up to hundreds of times faster than normal Python, yet avoids the difficulties of writing
C or other low-level code.

:::{note}
Numba is not a direct dependency of tskit, so will not be available unless installed:

```bash
pip install numba
```

or

```bash
conda install numba
```
:::

## Overview

The numba integration provides a {class}`tskit.TreeSequence` wrapper class {class}`NumbaTreeSequence`.
This class can be used directly in `numba.njit` compiled functions, and provides several efficient
methods for tree traversal:

- **{meth}`~NumbaTreeSequence.tree_index`**: For efficient iteration through the trees in the sequence
- **{meth}`~NumbaTreeSequence.parent_index`**: For efficient access to parent edge information, to
traverse upwards through the ARG.
- **{meth}`~NumbaTreeSequence.child_index`**: For efficient access to child edge information, to
traverse downwards through the ARG.

These methods are optimised to work within Numba's `@njit` decorated functions,
allowing you to write high-performance tree sequence analysis code in a plain Python style.

## Basic Usage

The ``tskit.jit.numba`` module is not imported with normal `tskit` so must be imported explicitly:
```{code-cell} python
import numpy as np
import tskit
import tskit.jit.numba as tskit_numba
```

Normal third-party classes such as {class}`tskit.TreeSequence` can't be used in `numba.njit` compiled
functions so the {class}`tskit.TreeSequence` must be wrapped in a {class}`NumbaTreeSequence` by 
{meth}`jitwrap`. This must be done outside `njit` code:

```{code-cell} python
import msprime

ts = msprime.sim_ancestry(
    samples=50000,
    sequence_length=100000,
    recombination_rate=0.1,
    random_seed=42
)
numba_ts = tskit_numba.jitwrap(ts)
print(type(numba_ts))
```

## Tree Iteration

Tree iteration can be performed in `numba.njit` compiled functions using the {class}`TreeIndex` class.
This class provides `next()` and `prev()` methods for forward and backward iteration through the trees in a tree sequence. Its `in_range` and `out_range` attributes provide the edges that must be added or removed to form the current
tree from the previous tree, along with the current tree `interval` and its sites and mutations through `site_range` and `mutation_range`.

A `TreeIndex` instance can be obtained from a {class}`NumbaTreeSequence` using the {meth}`~NumbaTreeSequence.tree_index` method. The initial state of this is of a "null" tree outside the range of the tree sequence, the first call to `next()` or `prev()` will be to the first, or last tree sequence tree respectively. After that, the `in_range` and `out_range` attributes will provide the edges that must be added or removed to form the current tree from the previous tree. For example
`tree_index.in_range.order[in_range.start:in_range.stop]` will give the edge ids that are new in the current tree, and `tree_index.out_range.order[out_range.start:out_range.stop]` will give the edge ids that are no longer present in the current tree. `tree_index.site_range` and
`tree_index.mutation_range` give the indexes into the tree sequences site and mutation arrays.

As a simple example we can calculate the number of edges in each tree in a tree sequence:

```{code-cell} python
import numba

@numba.njit
def edges_per_tree(numba_ts):
    tree_index = numba_ts.tree_index()
    current_num_edges = 0
    num_edges = []
    
    # Move forward through the trees
    while tree_index.next():
        # Access current tree information
        in_range = tree_index.in_range
        out_range = tree_index.out_range
        
        current_num_edges -= (out_range.stop - out_range.start)
        current_num_edges += (in_range.stop - in_range.start)
        num_edges.append(current_num_edges)
    return num_edges
```

```{code-cell} python
:tags: [hide-cell]
# Warm up the JIT compiler
edges = edges_per_tree(numba_ts)
```


```{code-cell} python
import time

t = time.time()
jit_num_edges = edges_per_tree(numba_ts)
print(f"JIT Time taken: {time.time() - t:.4f} seconds")
```

Doing the same thing with the normal `tskit` API would be much slower:

```{code-cell} python
t = time.time()
python_num_edges = []
for tree in ts.trees():
    python_num_edges.append(tree.num_edges)
print(f"Normal Time taken: {time.time() - t:.4f} seconds")

assert jit_num_edges == python_num_edges, "JIT and normal results do not match!"
```

### Example - diversity calculation

As a more interesting example we can calculate genetic diversity (also known as pi).
For this example we'll be calculating based on the distance in the tree between samples.
(`mode="branch"` in the tskit API.)

This example also shows the style of Python code that gives best performance under `numba`
JIT compilation - using simple loops and fixed-size arrays with minimal object attribute access.

```{code-cell} python
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

            result += running_sum * (
                tree_index.interval[1] - tree_index.interval[0]
            )

        return result / numba_ts.sequence_length
```

```{code-cell} python
:tags: [hide-cell]
# Warm up the JIT
d = diversity(numba_ts)
```

```{code-cell} python
t = time.time()
d = diversity(numba_ts)
print("Diversity:", d)
print("Time taken:", time.time() - t)
```

As this code is written for this specific diversity calculation it is even faster
than the tskit C implementation, called here from Python:

```{code-cell} python
t = time.time()
d_tskit = ts.diversity(mode="branch")
print("Diversity (tskit):", d_tskit)
print("Time taken:", time.time() - t)
```

## ARG Traversal

Beyond iterating through trees, you may need to traverse the ARG vertically. The {meth}`~NumbaTreeSequence.child_index` and {meth}`~NumbaTreeSequence.parent_index` methods provide efficient access to parent-child relationships in the edge table within `numba.njit` functions.

The {meth}`~NumbaTreeSequence.child_index` method returns an array that allows you to efficiently find all edges where a given node is the parent. Since edges are already sorted by parent in the tskit data model, this is implemented using simple range indexing. For any node `u`, the returned array `child_index[u]` gives a tuple of the start and stop indices in the tskit edge table where node `u` is the parent. The index is calculated on each call to `child_index()` so should be called once.

The {meth}`~NumbaTreeSequence.parent_index` method creates a {class}`ParentIndex` that allows you to efficiently find all edges where a given node is the child. Since edges are not sorted by child in the edge table, the returned class contains a custom index that sorts edge IDs by child node (and then by left coordinate). For any node `u`, `parent_index.index_range[u]` gives a tuple of the start and stop indices in the `parent_index.edge_index` array, and `parent_index.edge_index[start:stop]` gives the actual tskit edge IDs.

Both can be obtained from a {class}`NumbaTreeSequence`:

```{code-cell} python
# Get the indexes
child_index = numba_ts.child_index()
parent_index = numba_ts.parent_index()

# Example: find all left coordinates of edges where node 5 is the parent
start, stop = child_index[5]
left_coords = numba_ts.edges_left[start:stop]
print(left_coords)

# Example: find all right coordinates of edges where node 3 is the child
start, stop = parent_index.index_range[3]
right_coords = numba_ts.edges_right[start:stop]
print(right_coords)
```

These indexes enable efficient algorithms that need to traverse parent-child relationships in the ARG, such as computing descendant sets, ancestral paths, or subtree properties.

### Example - descendant span calculation

Here's an example of using the ARG traversal indexes to calculate the total sequence length over which each node descends from a specified node:

```{code-cell} python
@numba.njit
def descendant_span(numba_ts, u):
    """
    Calculate the total sequence length over which each node 
    descends from the specified node u.
    """
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
```

```{code-cell} python
:tags: [hide-cell]
# Warm up the JIT
result = descendant_span(numba_ts, 0)
```

```{code-cell} python
# Calculate descendant span for the root node (highest numbered node)
root_node = numba_ts.num_nodes - 1
result = descendant_span(numba_ts, root_node)

# Show nodes that have non-zero descendant span
non_zero = result > 0
print(f"Nodes descended from {root_node}:")
print(f"Node IDs: {np.where(non_zero)[0]}")
print(f"Span lengths: {result[non_zero]}")
```

Comparing performance with using the tskit Python API:

```{code-cell} python
def descendant_span_tskit(ts, u):
    """Reference implementation using tskit trees"""
    total_descending = np.zeros(ts.num_nodes)
    for tree in ts.trees():
        descendants = tree.preorder(u)
        total_descending[descendants] += tree.span
    return total_descending

import time
t = time.time()
numba_result = descendant_span(numba_ts, root_node)
print(f"Numba time: {time.time() - t:.6f} seconds")

t = time.time()
tskit_result = descendant_span_tskit(ts, root_node)
print(f"tskit time: {time.time() - t:.6f} seconds")

np.testing.assert_array_almost_equal(numba_result, tskit_result, decimal=10)
print("Results match!")
```

### Example - ARG descendant and ancestral edges calculation

As we have `child_index` and `parent_index`, we can efficiently find both descendant and ancestral sub-ARGs
for a given node. This first example shows how to find all edges in the ARG that are descendants of a given node. It returns a boolean array indicating which edges are part of the sub-ARG rooted at the specified node:

```{code-cell} python
@numba.njit
def descendant_edges(numba_ts, u):
    """
    Returns a boolean array which is only True for edges that are descendants of node u.
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
        
        # Find all edges where 'node' is the parent
        start, stop = child_index[node]
        for e in range(start, stop):
            e_left = edges_left[e]
            e_right = edges_right[e]
            
            # Check for genomic interval overlap
            if e_right > left and right > e_left:
                # This edge is part of the sub-ARG
                edge_select[e] = True
                
                # Calculate the intersection for the next traversal step
                inter_left = max(e_left, left)
                inter_right = min(e_right, right)
                e_child = edges_child[e]
                stack.append((e_child, inter_left, inter_right))
                
    return edge_select
```

```{code-cell} python
# Find descendant edges for a high-numbered node (likely near root)
test_node = max(0, numba_ts.num_nodes - 5)
edge_select = descendant_edges(numba_ts, test_node)

# Show which edges are descendants
descendant_edge_ids = np.where(edge_select)[0]
print(f"Edges descended from node {test_node}: {descendant_edge_ids[:10]}...")
print(f"Total descendant edges: {np.sum(edge_select)}")
```

```{code-cell} python
:tags: [hide-cell]
# Create a simple hard-coded example for consistent visualization
tables = tskit.TableCollection(sequence_length=10.0)

tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # node 0
tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # node 1
tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # node 2
tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # node 3
tables.nodes.add_row(flags=0, time=1)  # node 4
tables.nodes.add_row(flags=0, time=2)  # node 5
tables.nodes.add_row(flags=0, time=3)  # node 6

tables.edges.add_row(left=0, right=5, parent=4, child=0)
tables.edges.add_row(left=0, right=10, parent=4, child=1)
tables.edges.add_row(left=5, right=10, parent=5, child=0)
tables.edges.add_row(left=0, right=10, parent=5, child=2)
tables.edges.add_row(left=0, right=7, parent=6, child=4)
tables.edges.add_row(left=0, right=10, parent=6, child=5)
tables.edges.add_row(left=7, right=10, parent=6, child=3)

tables.sort()
ts_simple = tables.tree_sequence()
```

A tree sequence is easily made from the descendant edges array:

```{code-cell} python
numba_ts_simple = tskit_numba.jitwrap(ts_simple)
node = 5
E = descendant_edges(numba_ts_simple, node)
tables_sub = ts_simple.dump_tables()
tables_sub.edges.replace_with(tables_sub.edges[E])
ts_sub = tables_sub.tree_sequence()
```

As an example, lets visualise the selection of a sub-ARG. Here is the full ARG
with a highlighted node:

```{code-cell} python
css_style = f".node.n{node} > .sym {{ fill: #c41e3a; }}"
ts_simple.draw_svg(size=(400, 200), node_labels={}, y_axis=True, style=css_style)
```

And the sub-ARG from that node:

```{code-cell} python
ts_sub.draw_svg(size=(400, 200), node_labels={}, y_axis=True, style=css_style)
```

In the other direction, we can similarly find the sub-ARG that is ancestral to a given node:

```{code-cell} python
@numba.njit
def ancestral_edges(numba_ts, u):
    """
    Returns a boolean array which is only True for edges that are ancestors of node u.
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
        
        # Find all edges where 'node' is the child
        start, stop = parent_index.index_range[node]
        for i in range(start, stop):
            e = parent_index.edge_index[i]
            e_left = edges_left[e]
            e_right = edges_right[e]
            
            # Check for genomic interval overlap
            if e_right > left and right > e_left:
                # This edge is part of the sub-ARG
                edge_select[e] = True
                
                # Calculate the intersection for the next traversal step
                inter_left = max(e_left, left)
                inter_right = min(e_right, right)
                e_parent = edges_parent[e]
                stack.append((e_parent, inter_left, inter_right))

    return edge_select
```

```{code-cell} python
# Find ancestral edges for a sample node (low-numbered nodes are usually samples)
test_node = min(5, numba_ts.num_nodes - 1)
edge_select = ancestral_edges(numba_ts, test_node)

# Show which edges are ancestors
ancestral_edge_ids = np.where(edge_select)[0]
print(f"Edges ancestral to node {test_node}: {ancestral_edge_ids[:10]}...")
print(f"Total ancestral edges: {np.sum(edge_select)}")
```

```{code-cell} python
:tags: [hide-cell]
# Warm up the JIT for both functions
_ = descendant_edges(numba_ts, 0)
_ = ancestral_edges(numba_ts, 0)
```

Comparing performance with using the tskit Python API shows significant speedup:

```{code-cell} python
def descendant_edges_tskit(ts, start_node):
    D = np.zeros(ts.num_edges, dtype=bool)
    for tree in ts.trees():
        for v in tree.preorder(start_node):
            if v != start_node:
                D[tree.edge(v)] = True
    return D

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

import time

# Test with root node for descendant edges
root_node = numba_ts.num_nodes - 1
t = time.time()
numba_desc = descendant_edges(numba_ts, root_node)
print(f"Numba descendant edges time: {time.time() - t:.6f} seconds")

t = time.time()
tskit_desc = descendant_edges_tskit(ts, root_node)
print(f"tskit descendant edges time: {time.time() - t:.6f} seconds")

# Test with sample node for ancestral edges  
sample_node = 0
t = time.time()
numba_anc = ancestral_edges(numba_ts, sample_node)
print(f"Numba ancestral edges time: {time.time() - t:.6f} seconds")

t = time.time()
tskit_anc = ancestral_edges_tskit(ts, sample_node)
print(f"tskit ancestral edges time: {time.time() - t:.6f} seconds")

# Verify results match
np.testing.assert_array_equal(numba_desc, tskit_desc)
np.testing.assert_array_equal(numba_anc, tskit_anc)
print("Results match!")
```

## API Reference

```{eval-rst}
.. currentmodule:: tskit.jit.numba

.. autofunction:: jitwrap

.. autoclass:: NumbaTreeSequence
   :members:

.. autoclass:: TreeIndex
   :members:

.. autoclass:: EdgeRange
   :members:

.. autoclass:: ParentIndex
   :members:
```