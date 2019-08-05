import msprime
import numpy as np
import tskit

def find_interval_left(x, breaks):
    """
    Given a location x and an increasing vector of breakpoints breaks,
    return the index k such that breaks[k] <= x < breaks[k+1],
    returning -1 if x < breaks[0] and len(breaks)-1 if x >= breaks[-1].
    """
    if x < breaks[0]:
        return -1
    if x >= breaks[-1]:
        return len(breaks) - 1
    i = 0
    j = len(breaks) - 1
    while i + 1 < j:
        k = int((i + j)/2)
        if breaks[k] <= x:
            i = k
        else:
            j = k
    return i

breaks = np.array([0.0, 1.0, 3.5, 5.0])
for x, y in [(0.0, 0),
             (0.5, 0),
             (1.0, 1),
             (4.0, 2),
             (5.0, 3),
             (100, 3)]:
    assert find_interval_left(x, breaks) == y


def find_interval_right(x, breaks):
    """
    Given a location x and an increasing vector of breakpoints breaks,
    return the index k such that breaks[k] < x <= breaks[k+1].
    returning -1 if x <= breaks[0] and len(breaks)-1 if x > breaks[-1].
    """
    if x <= breaks[0]:
        return -1
    if x > breaks[-1]:
        return len(breaks) - 1
    i = 0
    j = len(breaks) - 1
    while i + 1 < j:
        k = int((i + j)/2)
        if breaks[k] < x:
            i = k
        else:
            j = k
    return i

breaks = np.array([0.0, 1.0, 3.5, 5.0])
for x, y in [(0.0, -1),
             (0.5, 0),
             (1.0, 0),
             (1.2, 1),
             (4.0, 2)]:
    assert find_interval_right(x, breaks) == y

def interval_index(x, starts, ends):
    """
    Returns the index of the interval that the position x lies in,
    or -1 if it does not lie in an interval.
    """
    i = find_interval_left(x, starts)
    j = find_interval_left(x, ends)
    if j >= i:
        out = -1
    else:
        out = i
    return out

starts = np.array([0.0, 1.0, 3.5, 5.0])
ends = np.array([0.5, 2.0, 4.0, 6.0])
for x, y in [(-1.0, -1),
             (0.0, 0),
             (0.25, 0),
             (0.5, -1),
             (1.0, 1),
             (1.2, 1),
             (5.2, 3),
             (8.0, -1)]:
    assert interval_index(x, starts, ends) == y

def do_overlap(segment, starts, ends):
    """
    Given a segment = [left, right), yield the segments
    found by intersecting it with the intervals described by starts, ends,
    which should be sorted and nonoverlapping.
    """
    assert(len(starts) == len(ends))
    assert(np.all(starts < ends))
    assert(np.all(ends[:-1] < starts[1:]))
    left, right = segment
    # the index of the first interval that ends at or before `left`
    a = find_interval_left(left, ends)
    # the index of the first interval that starts before `right`
    b = find_interval_right(right, starts)
    for k in range(a+1, b+1):
        yield (max(left, starts[k]), min(right, ends[k]))

starts = np.array([0.0, 1.0, 3.5, 5.0])
ends = np.array([0.5, 2.0, 4.0, 6.0])
for x, y in [((0.0, 0.5), [(0.0, 0.5)]),
             ((0.0, 0.7), [(0.0, 0.5)]),
             ((0.0, 1.2), [(0.0, 0.5), (1.0, 1.2)]),
             ((0.4, 1.2), [(0.4, 0.5), (1.0, 1.2)]),
             ((0.5, 1.0), []),
             ((0.5, 1.2), [(1.0, 1.2)]),
             ((0.6, 4.2), [(1.0, 2.0), (3.5, 4.0)]),
             ((-1.0, 6.2), [(0.0, 0.5), (1.0, 2.0), (3.5, 4.0), (5.0, 6.0)])]:
    out = list(do_overlap(x, starts, ends))
    assert len(y) == len(out)
    for a, b in zip(y, out):
        assert a == b


def dice(ts, starts, ends):
    """
    Remove edges and sites of the tree sequence that do *not* lie in the collection
    of half-open intervals [s, e) given by starts, ends.
    """
    assert(len(starts) == len(ends))
    assert(np.all(starts < ends))
    assert(np.all(ends[:-1] < starts[1:]))
    tables = ts.tables
    tables.edges.clear()
    for e in ts.edges():
        for l,r in do_overlap((e.left, e.right), starts, ends):
            tables.edges.add_row(left=l, right=r, parent=e.parent, child=e.child)
    tables.sites.clear()
    site_map = np.repeat(-1, ts.num_sites)
    for i, s in enumerate(ts.sites()):
        if interval_index(s.position, starts, ends) >= 0:
            j = tables.sites.add_row(s.position, s.ancestral_state, s.metadata)
            site_map[i] = j
    tables.mutations.clear()
    for m in ts.mutations():
        s = site_map[m.site]
        if s >= 0:
            tables.mutations.add_row(
                    site=s, node=m.node,
                    derived_state=m.derived_state,
                    parent=-1,
                    metadata=m.metadata)
    tables.build_index()
    tables.compute_mutation_parents()
    return tables
    

ts = msprime.simulate(10, recombination_rate=2, mutation_rate=1, length=10, random_seed=23)

sub_tables = dice(ts, starts=[1.0, 5.0], ends=[2.0, 10.0])
sub_ts = sub_tables.tree_sequence()
