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

```{currentmodule} tskit
```


(sec_identity)=

# Identity by descent

The {meth}`.TreeSequence.ibd_segments` method allows us to compute
segments of identity by descent along a tree sequence.

:::{note}
This documentation page is preliminary
:::

## Examples

Let's take a simple tree sequence to illustrate the {meth}`.TreeSequence.ibd_segments`
method and associated {ref}`sec_python_api_reference_identity`:

```{code-cell}
:tags: [hide-input]

import tskit
import io
from IPython.display import SVG

nodes = io.StringIO(
    """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    """
)
edges = io.StringIO(
    """\
    left    right   parent  child
    2     10     3       0
    2     10     3       2
    0     10     4       1
    0     2      4       2
    2     10     4       3
    0     2      5       0
    0     2      5       4
    """
)
ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

SVG(ts.draw_svg())
```

### Definition

A pair of nodes ``(u, v)`` has an IBD segment with a left and right
coordinate ``[left, right)`` and ancestral node ``a`` iff the most
recent common ancestor of the segment ``[left, right)`` in nodes ``u``
and ``v`` is ``a``, and the segment has been inherited along the same
genealogical path (ie. it has not been broken by recombination). The
definition of a "genealogical path" used here is
the sequence of edges, rather than nodes.
So, for instance, if ``u`` inherits a segment ``[x, z)`` from ``a``,
but that inheritance is represented by two edges,
one spanning ``[x, y)`` and the other spanning ``[y, z)``,
then this represents two genealogical paths,
and any IBD segments would be split at ``y``.
In other words, the method assumes that the end
of an edge represents a recombination,
an assumption that may not reflect how the tree sequence
is used -- see below for more discussion.

This definition is purely genealogical: it depends only on the tree
sequence topology and node times, and does not inspect allelic
states or mutations. In particular, if we compute the MRCA of ``(u, v)``
in each tree along the sequence, then (up to the additional refinement
by genealogical path) the IBD segments are those
that share the same ancestor and paths to that
ancestor. Intervals in which ``u`` and ``v`` lie in different roots
have no MRCA and therefore do not contribute IBD segments.

Consider the IBD segments that we get from our example tree sequence:

```{code-cell}
segments = ts.ibd_segments(store_segments=True)
for pair, segment_list in segments.items():
    print(pair, list(segment_list))
```

Each of the sample pairs (0, 1), (0, 2) and (1, 2) is associated with
two IBD segments, representing the different paths from these sample
pairs to their common ancestor. Note in particular that (1, 2) has
**two** IBD segments rather than one: even though the MRCA is
4 in both cases, the paths from the samples to the MRCA are different
in the left and right trees.


### Data structures

The result of calling {meth}`.TreeSequence.ibd_segments` is an
{class}`.IdentitySegments` class:

```{code-cell}
segments = ts.ibd_segments()
print(segments)
```

By default this class only stores the high-level summaries of the
IBD segments discovered. As we can see in this example, we have a
total of six segments and
the total span (i.e., the sum lengths of the genomic intervals spanned
by IBD segments) is 30. In this default mode the object does not
store information about individual sample pairs, and methods that
inspect per-pair information (such as indexing with ``[(a, b)]`` or
iterating over the mapping) will raise an
``IdentityPairsNotStoredError``.

If required, we can get more detailed information about particular
segment pairs and the actual segments using the ``store_pairs``
and ``store_segments`` arguments.

:::{warning}
Only use the ``store_pairs`` and ``store_segments`` arguments if you
really need this information! The number of IBD segments can be
very large and storing them all requires a lot of memory. It is
also much faster to just compute the overall summaries, without
needing to store the actual lists.
:::


```{code-cell}
segments = ts.ibd_segments(store_pairs=True)
for pair, value in segments.items():
    print(pair, "::", value)
```

Now we can see the more detailed breakdown of how the identity segments
are distributed among the sample pairs. The {class}`.IdentitySegments`
class behaves like a dictionary, such that ``segments[(a, b)]`` will return
the {class}`.IdentitySegmentList` instance for that pair of samples:

```{code-cell}
seglist = segments[(0, 1)]
print(seglist)
```

If we want to access the detailed information about the actual
identity segments, we must use the ``store_segments`` argument:

```{code-cell}
segments = ts.ibd_segments(store_pairs=True, store_segments=True)
segments[(0, 1)]
```

When ``store_segments=True``, the {class}`.IdentitySegmentList` behaves
like a Python list, where each element is an instance of
{class}`.IdentitySegment`. When only ``store_pairs=True`` is specified,
the number of segments and their total span are still available, but
attempting to iterate over the list or access the per-segment arrays
will raise an ``IdentitySegmentsNotStoredError``.

:::{warning}
The order of segments in an {class}`.IdentitySegmentList`
is arbitrary, and may change in future versions.
:::


```{eval-rst}
.. todo:: More examples using the other bits of the IdentitySegments
    API here
```

### Controlling the sample sets

By default we get the IBD segments between all pairs of
{ref}`sample<sec_data_model_definitions_sample>` nodes.

#### IBD within a sample set

We can reduce this to pairs within a specific set using the
``within`` argument:

```{code-cell}
segments = ts.ibd_segments(within=[0, 2], store_pairs=True)
print(list(segments.keys()))
```

Here we have restricted attention to the samples with node IDs 0 and 2,
so only the pair ``(0, 2)`` appears in the result. In general:

- ``within`` should be a one-dimensional array-like of node IDs
  (typically sample nodes). All unordered pairs from this set are
  considered.
- If ``within`` is omitted (the default), all nodes flagged as samples
  in the node table are used.

#### IBD between sample sets

We can also compute IBD **between** sample sets:

```{code-cell}
segments = ts.ibd_segments(between=[[0,1], [2]], store_pairs=True)
print(list(segments.keys()))
```

In this example we have two sample sets, ``[0, 1]`` and ``[2]``, so the
identity segments are computed only for pairs in which one sample lies
in the first set and the other lies in the second. More generally:

- ``between`` should be a list of non-overlapping lists of node IDs.
- All pairs ``(u, v)`` are considered such that ``u`` and ``v`` belong
  to different sample sets.

The ``within`` and ``between`` arguments are mutually exclusive: passing
both at the same time raises a :class:`ValueError`.

:::{seealso}
See the {meth}`.TreeSequence.ibd_segments` documentation for
more details.
:::

### Constraints on the segments

The ``max_time`` and ``min_span`` arguments allow us to constrain the
segments that we consider.

The ``max_time`` argument specifies an upper bound on the time of the
common ancestor node: only IBD segments whose MRCA node has a time
no greater than ``max_time`` are returned.

The ``min_span`` argument filters by genomic length: only segments with
span strictly greater than ``min_span`` are included.

For example, working with ``ts2`` as the following tree sequence:

```{code-cell}
:tags: [hide-input]

import io

nodes = io.StringIO(
    """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       0           1
    3       0           3
    """
)
edges = io.StringIO(
    """\
    left    right   parent  child
    0      4     2       0,1
    4     10     3       0,1
    """
)
ts2 = tskit.load_text(nodes=nodes, edges=edges, strict=False)
SVG(ts2.draw_svg())
```

There are two segments:
```{code-cell}
segments = ts2.ibd_segments(store_segments=True)
print("all segments:", list(segments.values())[0])
```
... but only the left-hand one is more recent than 2 time units ago:
```{code-cell}
segments_recent = ts2.ibd_segments(max_time=2, store_segments=True)
print("max_time=1.2:", list(segments_recent.values())[0])
```
... and only the right-hand one is longer than 5 units.
```{code-cell}

segments_long = ts2.ibd_segments(min_span=5, store_segments=True)
print("min_span=0.5:", list(segments_long.values())[0])
```

So: the full result contains two IBD segments for the single sample
pair, one inherited via ancestor 2 over ``[0, 4)`` and one via
ancestor 3 over ``[4, 10)``. The ``max_time`` constraint removes the
segment inherited from the older ancestor (time 3), while the
``min_span`` constraint keeps only the longer of the two segments.

### More on the "pathwise" definition of IBD segments

We said above that the definition of IBD used by
{meth}`.TreeSequence.ibd_segments` says that a given segment
must be inherited from the MRCA along a single genealogical path,
and that "genealogical paths" are defined *edgewise*.
This can lead to surprising consequences.

Returning to our example above:
```{code-cell}
:tags: [hide-input]

SVG(ts.draw_svg())
```
there are two IBD segments between ``1`` and ``2``:
```{code-cell}
segments = ts.ibd_segments(within=[1, 2], store_pairs=True)
for pair, value in segments.items():
    print(pair, "::", value)
```
This might be surprising, because the MRCA of ``1`` and ``2``
is node ``4`` over the entire sequence.
In fact, some definitions of IBD segments
would have this as a single segment,
because the MRCA does not change,
even if there are distinct genealogical paths.

The reason this is split into two segments
is because the path from ``4`` to ``2`` changes:
on the left-hand segment ``[0, 2)``, the node ``2``
inherits from node ``4``
via node ``3``, while on the right-hand segment ``[2, 10)``
it inherits from node ``4`` directly.
The tree sequence doesn't say directly whether node ``2``
also inherits from node ``3`` on the right-hand segment,
so whether or not this should be one IBD segment or two
depends on our interpretation
of what's stored in the tree sequence.
As discussed in 
[Fritze et al](https://doi.org/10.1093/genetics/iyaf198),
most tree sequence simulators (at time of writing)
will produce this tree sequence even if node ``2``
does in fact inherit from ``3`` over the entire sequence.
Using {meth}`.TreeSequence.extend_haplotypes` will
"put the unary nodes back":
```{code-cell}
ets = ts.extend_haplotypes()
SVG(ets.draw_svg())
```
and once this is done, there is only a single IBD segment:
```{code-cell}
segments = ets.ibd_segments(within=[1, 2], store_pairs=True)
for pair, value in segments.items():
    print(pair, "::", value)
```
So, extending haplotypes may produce IBD segments
more in line with theory, if the desired definition if IBD
is the "pathwise" definition.
However, this will also probably introduce erroneous
portions of IBD segments,
so caution is needed.
Another approach would be to merge adjacent segments of IBD
that have the same MRCA.

Summarizing this section --
there is a confusing array of possible definitions
of what it means to be "an IBD segment";
and these may be extracted from a tree sequence
in subtly different ways.
How much of a problem is this?
The answer depends on the precise situation,
but it seems likely that in practice,
differences due to definition are small
relative to errors due to tree sequence inference.
Indeed, empirical haplotype-matching methods
for identifying IBD segments can differ substantially
depending on the values of various hyperparameters.
More work is needed to develop a complete picture.
