# MIT License
#
# Copyright (c) 2020-2024 Tskit Developers
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
import argparse
import collections

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
        self.segments[key].append(tskit.IdentitySegment(seg.left, seg.right, seg.node))


class IbdFinder:
    """
    Finds all IBD relationships between specified sample pairs in a tree sequence.
    """

    def __init__(self, ts, *, within=None, between=None, min_span=0, max_time=None):
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
        self.min_span = min_span
        self.max_time = np.inf if max_time is None else max_time
        self.A = [SegmentList() for _ in range(ts.num_nodes)]  # Descendant segments
        for u in range(ts.num_nodes):
            if self.sample_set_id[u] != -1:
                self.A[u].append(Segment(0, ts.sequence_length, u))
        self.tables = self.ts.tables

    def print_state(self):
        print("IBD Finder")
        print("min_span = ", self.min_span)
        print("max_time   = ", self.max_time)
        print("finding_between = ", self.finding_between)
        print("u\tset_id\tA = ")
        for u, a in enumerate(self.A):
            print(u, self.sample_set_id[u], a, sep="\t")

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
                if intvl[1] - intvl[0] > self.min_span:
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
        if right - left <= self.min_span:
            return False
        if self.finding_between:
            return self.sample_set_id[a] != self.sample_set_id[b]
        else:
            return True


if __name__ == "__main__":
    """
    A simple CLI for running IBDFinder on a command line from the `python`
    subdirectory. Basic usage:
    > python3 ./tests/ibd.py --infile test.trees
    """

    parser = argparse.ArgumentParser(
        description="Command line interface for the IBDFinder."
    )

    parser.add_argument(
        "--infile",
        type=str,
        dest="infile",
        nargs=1,
        metavar="IN_FILE",
        help="The tree sequence to be analysed.",
    )

    parser.add_argument(
        "--min-length",
        type=float,
        dest="min_span",
        nargs=1,
        metavar="MIN_LENGTH",
        help="Only segments longer than this cutoff will be returned.",
    )

    parser.add_argument(
        "--max-time",
        type=float,
        dest="max_time",
        nargs=1,
        metavar="MAX_TIME",
        help="Only segments younger this time will be returned.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        dest="samples",
        nargs=2,
        metavar="SAMPLES",
        help="If provided, only this pair's IBD info is returned.",
    )

    args = parser.parse_args()
    ts = tskit.load(args.infile[0])
    if args.min_span is None:
        min_span = 0
    else:
        min_span = args.min_span[0]
    if args.max_time is None:
        max_time = None
    else:
        max_time = args.max_time[0]

    s = IbdFinder(ts, min_span=min_span, max_time=max_time)
    all_segs = s.run()

    if args.samples is None:
        print(all_segs)
    else:
        samples = args.samples
        print(all_segs[(samples[0], samples[1])])
