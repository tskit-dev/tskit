# MIT License
#
# Copyright (c) 2020-2021 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Python implementation of the IBD-finding algorithms.
"""
from __future__ import annotations

import collections
import dataclasses
import itertools
from typing import List
from typing import Union

import intervaltree
import numpy as np

import tskit


class Segment:
    """
    A class representing a single segment. Each segment has a left and right,
    denoting the loci over which it spans, a node and a next, giving the next
    in the chain.

    The node it records is the *output* node ID.
    """

    def __init__(self, left=None, right=None, node=None, next_seg=None):
        self.left = left
        self.right = right
        self.node = node
        self.next = next_seg

    def __str__(self):
        s = "({}-{}->{}:next={})".format(
            self.left, self.right, self.node, repr(self.next)
        )
        return s

    def __repr__(self):
        return repr((self.left, self.right, self.node))

    def __eq__(self, other):
        # NOTE: to simplify tests, we DON'T check for equality of 'next'.
        return (
            self.left == other.left
            and self.right == other.right
            and self.node == other.node
        )

    def __lt__(self, other):
        return (self.node, self.left, self.right) < (
            other.node,
            other.left,
            other.right,
        )


class SegmentList:
    """
    A class representing a list of segments that are descended from a given ancestral
    node via a particular child of the ancestor.
    Each SegmentList keeps track of the first and last segment in the list, head and
    tail.
    """

    def __init__(self, head=None, tail=None):
        self.head = head
        self.tail = tail

    def __str__(self):
        return repr(self)

    def __repr__(self):
        tuple_segs = []
        seg = self.head
        while seg is not None:
            tuple_segs.append((seg.left, seg.right, seg.node))
            seg = seg.next
        return repr(tuple_segs)

    def extend(self, seglist):
        """
        Extends this segment list with the segments in the specified list.
        """
        assert isinstance(seglist, SegmentList)
        if seglist.head is not None:
            if self.head is None:
                self.head = seglist.head
                self.tail = seglist.tail
            else:
                self.tail.next = seglist.head
                self.tail = seglist.tail

    def append(self, segment):
        """
        Append the specified segment to the end of this list.
        """
        assert isinstance(segment, Segment)
        if self.head is None:
            self.head = segment
            self.tail = segment
        else:
            self.tail.next = segment
            self.tail = segment


class IbdResult:
    """
    Class representing the IBD segments in a tree sequence for a given
    set of sample pairs.
    """

    def __init__(self):
        self.segments = collections.defaultdict(list)

    def __repr__(self):
        return repr(self.segments)

    def __str__(self):
        return repr(self)

    def add_segment(self, a, b, seg):
        key = (a, b) if a < b else (b, a)
        self.segments[key].append(tskit.IbdSegment(seg.left, seg.right, seg.node))


@dataclasses.dataclass
class AncestrySegment:
    left: float
    right: float
    samples: List[int] = dataclasses.field(default_factory=list)
    next: Union[None, AncestrySegment] = None  # noqa: A003
    prev: Union[None, AncestrySegment] = None

    def __repr__(self):
        return repr((self.left, self.right, self.samples))


@dataclasses.dataclass
class AncestrySegmentList:
    head: AncestrySegment
    tail: AncestrySegment

    def __repr__(self):
        lst = []
        u = self.head.next
        while u != self.tail:
            lst.append((u.left, u.right, u.samples))
            u = u.next
        return repr(lst)

    def check(self):
        prev = None
        u = self.head
        while u is not None:
            assert u.prev is prev
            if prev is not None:
                assert prev.right == u.left
            assert u.left < u.right

            prev = u
            u = u.next
        assert prev == self.tail


class IbdFinderBroken:
    """
    Finds all IBD relationships between specified sample pairs in a tree sequence.
    """

    def __init__(self, ts, *, within=None, between=None, min_length=0, max_time=None):
        self.ts = ts
        self.result = IbdResult()
        if within is not None and between is not None:
            raise ValueError("within and between are mutually exclusive")

        self.sample_set_id = np.zeros(ts.num_nodes, dtype=int) - 1
        self.finding_between = False
        if between is not None:
            self.finding_between = True
            for set_id, samples in enumerate(between):
                self.sample_set_id[samples] = set_id
        else:
            if within is None:
                within = ts.samples()
            self.sample_set_id[within] = 0
        self.min_length = min_length
        self.max_time = np.inf if max_time is None else max_time
        self.D = [None for _ in range(ts.num_nodes)]
        for u in range(ts.num_nodes):
            head = AncestrySegment(-1, 0)
            tail = AncestrySegment(ts.sequence_length, ts.sequence_length + 1)
            lst = AncestrySegment(0, ts.sequence_length, prev=head, next=tail)
            head.next = lst
            tail.prev = lst
            self.D[u] = AncestrySegmentList(head, tail)
            if self.sample_set_id[u] != -1:
                lst.samples.append(u)
        self.tables = self.ts.tables

    def print_state(self):
        print("IBD Finder")
        print("min_length = ", self.min_length)
        print("max_time   = ", self.max_time)
        print("finding_between = ", self.finding_between)
        print("u\tset_id\tD = ")
        for u, a in enumerate(self.D):
            print(u, self.sample_set_id[u], a, sep="\t")

        # self.check_state()

    def add_ibd_samples(self, parent, segment, samples):

        for a, b in itertools.product(samples, segment.samples):
            if self.passes_filters(a, b, segment.left, segment.right):
                self.result.add_segment(
                    a, b, Segment(segment.left, segment.right, parent)
                )
        segment.samples.extend(samples)

    def update_ancestry(self, left, right, parent, samples):
        """
        Update the sample lists for the specified interval on the
        specified parent to include the specified list. Also record
        any qualifying IBD segments that result.
        """
        if right - left <= self.min_length:
            return

        u = self.D[parent].head
        while left >= u.right:
            u = u.next
        if u.left < left:
            #             left
            #     u.left   |    u.right
            # ----|--------------|
            #
            # ->
            #         v       u
            # ----|--------|-----|
            v = AncestrySegment(u.left, left, list(u.samples), prev=u.prev, next=u)
            u.left = left
            u.prev.next = v
            u.prev = v
        # Update segments completely contained in the interval
        while u.right <= right:
            # print("updating u", repr(u), left, right)
            self.add_ibd_samples(parent, u, samples)
            u = u.next
        # Update and trim the last segment overlapping the interval
        if right > u.left:
            #             right
            #     u.left   |    u.right
            # ----|--------------|
            #
            # ->
            #         u       v
            # ----|--------|-----|
            v = AncestrySegment(right, u.right, list(u.samples), prev=u, next=u.next)
            u.right = right
            u.next.prev = v
            u.next = v
            self.add_ibd_samples(parent, u, samples)
        # print("DONE:", repr(self))
        # print()

    def run(self):
        node_times = self.tables.nodes.time
        for e in self.ts.edges():
            time = node_times[e.parent]
            if time > self.max_time:
                # Stop looking for IBD segments once the
                # processed nodes are older than the max time.
                break
            # print("processing edge", e)
            s = self.D[e.child].head.next
            while s is not None:
                interval = (
                    max(e.left, s.left),
                    min(e.right, s.right),
                )
                self.update_ancestry(*interval, e.parent, s.samples)
                s = s.next
        # self.check_state()
        return self.result.segments

    def passes_filters(self, a, b, left, right):
        # These are already tested
        # if a == b:
        #     return False
        # if right - left <= self.min_length:
        #     return False
        if self.finding_between:
            return self.sample_set_id[a] != self.sample_set_id[b]
        else:
            return True


class IbdFinderIntervalTrees:
    """"""

    def __init__(self, ts, *, within=None, between=None, min_length=0, max_time=None):
        self.ts = ts

        if within is not None and between is not None:
            raise ValueError("within and between are mutually exclusive")

        self.sample_set_id = np.zeros(ts.num_nodes, dtype=int) - 1
        self.finding_between = False
        if between is not None:
            self.finding_between = True
            for set_id, samples in enumerate(between):
                self.sample_set_id[samples] = set_id
        else:
            if within is None:
                within = ts.samples()
            self.sample_set_id[within] = 0
        self.min_length = min_length
        self.max_time = np.inf if max_time is None else max_time
        self.tables = self.ts.tables
        self.A = [intervaltree.IntervalTree() for _ in range(ts.num_nodes)]

        for u in range(ts.num_nodes):
            if self.sample_set_id[u] != -1:
                self.A[u][0 : ts.sequence_length] = u

    def print_state(self):
        print("IBD Finder")
        print("min_length = ", self.min_length)
        print("max_time   = ", self.max_time)
        print("finding_between = ", self.finding_between)
        print("u\tset_id\tI = ")
        for u, i in enumerate(self.A):
            print(u, self.sample_set_id[u], len(i), i, sep="\t")

    def keep_pair(self, a, b):
        if self.finding_between:
            return self.sample_set_id[a] != self.sample_set_id[b]
        return True

    def run(self):
        result = collections.defaultdict(list)
        node_times = self.tables.nodes.time
        L = self.ts.sequence_length
        for e in self.ts.edges():
            if node_times[e.parent] > self.max_time:
                # Stop looking for IBD segments once the
                # processed nodes are older than the max time.
                break
            t_child = self.A[e.child].copy()
            # print(e)
            # There's no operation to "keep" an interval
            t_child.chop(0, e.left)
            t_child.chop(e.right, L)
            t_parent = self.A[e.parent].copy()
            t_parent.chop(0, e.left)
            t_parent.chop(e.right, L)

            for int1, int2 in itertools.product(t_parent, t_child):
                if int1.overlap_size(int2) > self.min_length:
                    key = tuple(sorted([int1.data, int2.data]))
                    intersection = (
                        max(int1.begin, int2.begin),
                        min(int1.end, int2.end),
                    )
                    if self.keep_pair(*key):
                        result[key].append(tskit.IbdSegment(*intersection, e.parent))

            # Update parent's intervals
            self.A[e.parent] |= t_child
        return result


class IbdFinderOld:
    """
    Finds all IBD relationships between specified sample pairs in a tree sequence.
    """

    def __init__(self, ts, *, within=None, between=None, min_length=0, max_time=None):
        self.ts = ts
        self.result = IbdResult()
        if within is not None and between is not None:
            raise ValueError("within and between are mutually exclusive")

        self.sample_set_id = np.zeros(ts.num_nodes, dtype=int) - 1
        self.finding_between = False
        if between is not None:
            self.finding_between = True
            for set_id, samples in enumerate(between):
                self.sample_set_id[samples] = set_id
        else:
            if within is None:
                within = ts.samples()
            self.sample_set_id[within] = 0
        self.min_length = min_length
        self.max_time = np.inf if max_time is None else max_time
        self.A = [SegmentList() for _ in range(ts.num_nodes)]  # Descendant segments
        for u in range(ts.num_nodes):
            if self.sample_set_id[u] != -1:
                self.A[u].append(Segment(0, ts.sequence_length, u))
        self.tables = self.ts.tables

    def print_state(self):
        print("IBD Finder")
        print("min_length = ", self.min_length)
        print("max_time   = ", self.max_time)
        print("finding_between = ", self.finding_between)
        print("u\tset_id\tA = ")
        for u, a in enumerate(self.A):
            n = 0
            x = self.A[u].head
            while x is not None:
                n += 1
                x = x.next
            print(u, self.sample_set_id[u], n, a, sep="\t")

    def run(self):
        node_times = self.tables.nodes.time
        for e in self.ts.edges():
            time = node_times[e.parent]
            if time > self.max_time:
                # Stop looking for IBD segments once the
                # processed nodes are older than the max time.
                break
            child_segs = SegmentList()
            s = self.A[e.child].head
            while s is not None:
                intvl = (
                    max(e.left, s.left),
                    min(e.right, s.right),
                )
                if intvl[1] - intvl[0] > self.min_length:
                    child_segs.append(Segment(intvl[0], intvl[1], s.node))
                s = s.next
            self.record_ibd(e.parent, child_segs)
            self.A[e.parent].extend(child_segs)
        return self.result.segments

    def record_ibd(self, current_parent, child_segs):
        """
        Given the specified set of child segments for the current parent
        record the IBD segments that will occur as a result of adding these
        new segments into the existing list.
        """
        # Note the implementation here is O(n^2) because we have to compare
        # every segment with every other one. If the segments were stored in
        # left-to-right sorted order, we could avoid and merge them more
        # efficiently. There is some added complexity in doing this, however.
        seg0 = self.A[current_parent].head
        while seg0 is not None:
            seg1 = child_segs.head
            while seg1 is not None:
                left = max(seg0.left, seg1.left)
                right = min(seg0.right, seg1.right)
                # If there are any overlapping segments, record as a new
                # IBD relationship.
                if self.passes_filters(seg0.node, seg1.node, left, right):
                    self.result.add_segment(
                        seg0.node, seg1.node, Segment(left, right, current_parent)
                    )
                seg1 = seg1.next
            seg0 = seg0.next

    def passes_filters(self, a, b, left, right):
        if a == b:
            return False
        if right - left <= self.min_length:
            return False
        if self.finding_between:
            return self.sample_set_id[a] != self.sample_set_id[b]
        else:
            return True
