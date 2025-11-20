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
segments returned are the longest possible ones: for a fixed pair
``(u, v)`` we follow the ancestral paths from ``u`` and ``v`` up the
trees and merge together adjacent genomic intervals whenever both the
MRCA ``a`` and the full ancestral paths from ``u`` and ``v`` to ``a``
are identical.

This definition is purely genealogical: it depends only on the tree
sequence topology and node times, and does not inspect allelic
states or mutations. In particular, if we compute the MRCA of ``(u, v)``
in each tree along the sequence, then (up to the additional refinement
by genealogical path) the IBD segments are obtained by merging together
adjacent MRCA intervals that share the same ancestor and paths to that
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
- Passing an empty list, e.g. ``within=[]``, is allowed and simply
  yields a result with zero pairs and zero segments.

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
- Empty sample sets are permitted (e.g., ``between=[[0, 1], []]``) and
  simply do not contribute any pairs.

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
no greater than ``max_time`` are returned. The time is measured in
the same units as the node times in the tree sequence (e.g., generations).

The ``min_span`` argument filters by genomic length: only segments with
span strictly greater than ``min_span`` are included. This threshold is
measured in the same units as the ``sequence_length`` (for example,
base pairs).

For example:

```{code-cell}
import io

nodes = io.StringIO(
    """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       0           1
    3       0           1.5
    """
)
edges = io.StringIO(
    """\
    left    right   parent  child
    0.0     0.4     2       0,1
    0.4     1.0     3       0,1
    """
)
ts2 = tskit.load_text(nodes=nodes, edges=edges, strict=False)

segments = ts2.ibd_segments(store_segments=True)
print("all segments:", list(segments.values())[0])

segments_recent = ts2.ibd_segments(max_time=1.2, store_segments=True)
print("max_time=1.2:", list(segments_recent.values())[0])

segments_long = ts2.ibd_segments(min_span=0.5, store_segments=True)
print("min_span=0.5:", list(segments_long.values())[0])
```

Here the full result contains two IBD segments for the single sample
pair, one inherited via ancestor 2 over ``[0.0, 0.4)`` and one via
ancestor 3 over ``[0.4, 1.0)``. The ``max_time`` constraint removes the
segment inherited from the older ancestor (time 1.5), while the
``min_span`` constraint keeps only the longer of the two segments.
