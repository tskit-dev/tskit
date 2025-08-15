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

The numba integration provides:

- **{class}`NumbaTreeSequence`**: A Numba-compatible representation of tree sequence data
- **{class}`NumbaTreeIndex`**: A class for efficient tree iteration
- **{class}`NumbaEdgeRange`**: Container class for edge ranges during iteration

These classes are designed to work within Numba's `@njit` decorated functions,
allowing you to write high-performance tree sequence analysis code.

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

Tree iteration can be performed using the {class}`NumbaTreeIndex` class.
This class provides `next()` and `prev()` methods for forward and backward iteration through the trees in a tree sequence. Its `in_range` and `out_range` attributes provide the edges that must be added or removed to form the current
tree from the previous tree, along with the current tree `interval` and its sites and mutations through `site_range` and `mutation_range`.

A `NumbaTreeIndex` instance can be obtained from a `NumbaTreeSequence` using the `tree_index()` method. The initial state of this is of a "null" tree outside the range of the tree sequence, the first call to `next()` or `prev()` will be to the first, or last tree sequence tree respectively. After that, the `in_range` and `out_range` attributes will provide the edges that must be added or removed to form the current tree from the previous tree. For example
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

## Example - diversity calculation

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
                u = edge_child[h]

                running_sum -= branch_length[u] * summary[u]
                parent[u] = -1
                branch_length[u] = 0.0

                u = edge_parent[h]
                while u != -1:
                    running_sum -= branch_length[u] * summary[u]
                    state[u] -= state[edge_child[h]]
                    summary[u] = state[u] * (n - state[u]) * two_over_denom
                    running_sum += branch_length[u] * summary[u]
                    u = parent[u]

            # Process the incoming edges
            for j in range(tree_index.in_range.start, tree_index.in_range.stop):
                h = tree_index.in_range.order[j]
                u = edge_child[h]
                v = edge_parent[h]

                parent[u] = v
                branch_length[u] = node_times[v] - node_times[u]
                running_sum += branch_length[u] * summary[u]

                u = v
                while u != -1:
                    running_sum -= branch_length[u] * summary[u]
                    state[u] += state[edge_child[h]]
                    summary[u] = state[u] * (n - state[u]) * two_over_denom
                    running_sum += branch_length[u] * summary[u]
                    u = parent[u]

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




## API Reference

```{eval-rst}
.. currentmodule:: tskit.jit.numba

.. autofunction:: jitwrap

.. autoclass:: NumbaTreeSequence
   :members:

.. autoclass:: NumbaTreeIndex
   :members:

.. autoclass:: NumbaEdgeRange
   :members:
```