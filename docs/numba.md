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
upto hundreds of times faster than normal Python, yet avoids the difficulties of writing
C or other low-level code.

:::{note}
Numba is not a direct dependency of tskit, so will not be avaliable unless installed:

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
- **{class}`NumbaTreePosition`**: An class for efficient tree traversal
- **{class}`NumbaEdgeRange`**: Container class for edge ranges during traversal

These classes are designed to work within Numba's `@njit` decorated functions,
allowing you to write high-performance tree sequence analysis code.

## Basic Usage

The ``tskit.jit.numba`` module is not imported with normal `tskit` so must be imported explicitly:
```{code-cell} python
import tskit
import tskit.jit.numba as tskit_numba
```

Normal third-party classes such as {class}`tskit.TreeSequence` can't be used in `numba.njit` compiled
functions so the {class}`tskit.TreeSequence` must be wrapped in a {class}`NumbaTreeSequence` by 
{meth}`numba_tree_sequence`. This must be done outside `njit` code:

```{code-cell} python
import msprime

ts = msprime.sim_ancestry(
    samples=50000,
    sequence_length=100000,
    recombination_rate=0.1,
    random_seed=42
)
numba_ts = tskit_numba.numba_tree_sequence(ts)
print(type(numba_ts))
```

## Tree Traversal

Tree traversal can be performed using the {class}`NumbaTreePosition` class.
This class provides `next()` and `prev()` methods for forward and backward iteration through the trees in a tree sequence. It's `in_range` and `out_range` attributes provide the edges that must be added or removed to form the current
tree from the previous tree.

A `NumbaTreePosition` instance can be obtained from a `NumbaTreeSequence` using the `tree_position()` method. The initial state of this is of a "null" tree outside the range of the tree sequence, the first call to `next()` or `prev()`will be to the first, or last tree sequence tree respectively. After that, the `in_range` and `out_range` attributes will provide the edges that must be added or removed to form the current tree from the previous tree. For example
`in_range.order[in_range.start:in_range.stop]` will give the edge ids that are new in the current tree, and `out_range.order[out_range.start:out_range.stop]` will give the edge ids that are no longer present in the current tree.

As a simple example we can calulate the number of edges in each tree in a tree sequence:

```{code-cell} python
import numba

@numba.njit
def edges_per_tree(numba_ts):
    tree_pos = numba_ts.tree_position()
    current_num_edges = 0
    num_edges = []
    
    # Traverse trees forward
    while tree_pos.next():
        # Access current tree information
        in_range = tree_pos.in_range
        out_range = tree_pos.out_range
        
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

## API Reference

```{eval-rst}
.. currentmodule:: tskit.jit.numba

.. autofunction:: numba_tree_sequence

.. autoclass:: NumbaTreeSequence
   :members:

.. autoclass:: NumbaTreePosition
   :members:

.. autoclass:: NumbaEdgeRange
   :members:
```