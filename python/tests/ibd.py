# MIT License
#
# Copyright (c) 2020 Tskit Developers
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
    tail. The next attribute points to another SegmentList, allowing SegmentList
    objects to be 'chained' to one another.
    """

    def __init__(self, head=None, tail=None, next_list=None):
        self.head = head
        self.tail = tail
        self.next = next_list

    def __str__(self):
        s = "head={},tail={},next={}".format(self.head, self.tail, repr(self.next))
        return s

    def __repr__(self):
        if self.head is None:
            s = "[{}]".format(repr(None))
        elif self.head == self.tail:
            s = "[{}]".format(repr(self.head))
        elif self.head.next == self.tail:
            s = "[{}, {}]".format(repr(self.head), repr(self.tail))
        else:
            s = "[{}, ..., {}]".format(repr(self.head), repr(self.tail))
        return s

    def add(self, other):
        """
        Use to append another SegmentList, or a single segment.
        SegmentList1.add(SegmentList2) will modify SegmentList1 so that
        SegmentList1.tail.next = SegmentList2.head
        SegmentList1.add(Segment1) will add Segment1 to the tail of SegmentList1
        """
        assert isinstance(other, SegmentList) or isinstance(other, Segment)

        if isinstance(other, SegmentList):
            if self.head is None:
                self.head = other.head
                self.tail = other.tail
            else:
                self.tail.next = other.head
                self.tail = other.tail
        elif isinstance(other, Segment):
            if self.head is None:
                self.head = other
                self.tail = other
            else:
                self.tail.next = other
                self.tail = other


class IbdFinder:
    """
    Finds all IBD relationships between specified samples in a tree sequence.
    """

    def __init__(self, ts, samples=None, min_length=0, max_time=None):

        self.ts = ts
        # Note: samples *must* be in order of ascending node ID
        if samples is None:
            self.samples = ts.samples()
        else:
            self.samples = samples
        if len(self.samples) == 0:
            raise ValueError("The tree sequence contains no samples.")

        self.sample_id_map = np.zeros(ts.num_nodes, dtype=int) - 1
        for index, u in enumerate(self.samples):
            self.sample_id_map[u] = index
        self.min_length = min_length
        if max_time is None:
            self.max_time = 2 * ts.max_root_time
        else:
            self.max_time = max_time
        self.A = [None for _ in range(ts.num_nodes)]  # Descendant segments
        self.tables = self.ts.tables

        self.oldest_parent = self.get_oldest_parents()

        # Objects below are needed for the IBD segment-holding object.
        self.num_samples = len(self.samples)
        self.sample_pairs = self.get_sample_pairs()

        # Note: in the C code the object below should be a struct array.
        # Each item will be accessed using its index, which corresponds to a particular
        # sample pair. The mapping between index and sample pair is defined in the
        # find_sample_pair_index method further down.

        self.ibd_segments = {}
        for key in self.sample_pairs:
            self.ibd_segments[key] = None

    def get_oldest_parents(self):
        oldest_parents = [-1 for _ in range(self.ts.num_nodes)]
        node_times = self.ts.tables.nodes.time
        for e in self.ts.tables.edges:
            c = e.child
            if (
                oldest_parents[c] == -1
                or node_times[oldest_parents[c]] < node_times[e.parent]
            ):
                oldest_parents[c] = e.parent
        return oldest_parents

    def add_ibd_segments(self, sample0, sample1, seg):
        index = self.find_sample_pair_index(sample0, sample1)

        # Note: the code below is specific to the Python implementation, where the
        # output is a dictionary indexed by sample pairs.
        # In the C implementation, it'll be more like
        # self.ibd_segments[index].add(seg)

        if self.ibd_segments[self.sample_pairs[index]] is None:
            self.ibd_segments[self.sample_pairs[index]] = SegmentList(
                head=seg, tail=seg
            )
        else:
            self.ibd_segments[self.sample_pairs[index]].add(seg)

    def get_sample_pairs(self):
        """
        Returns a list of all pairs of samples. Replaces itertools.combinations.
        Note: they must be sorted
        """
        sample_pairs = []
        for ind, i in enumerate(self.samples):
            for j in self.samples[(ind + 1) :]:
                sample_pairs.append((i, j))

        return sample_pairs

    def find_sample_pair_index(self, sample0, sample1):
        """
        Note: this method isn't strictly necessary for the Python implementation
        but is needed for the C implemention, where the output ibd_segments is a
        struct array.
        This calculates the position of the object corresponding to the inputted
        sample pair in the struct array.
        """

        # Ensure samples are in order.
        if sample0 == sample1:
            raise ValueError("Samples in pair must have different node IDs.")
        elif sample0 > sample1:
            sample0, sample1 = sample1, sample0

        i0 = self.sample_id_map[sample0]
        i1 = self.sample_id_map[sample1]

        # Calculate the position of the sample pair in the vector.
        index = (
            (self.num_samples) * (self.num_samples - 1) / 2
            - (self.num_samples - i0) * (self.num_samples - i0 - 1) / 2
            + i1
            - i0
            - 1
        )

        return int(index)

    def find_ibd_segments(self):
        """
        The wrapper for the procedure that calculates IBD segments.
        """

        # Set up an iterator over the edges in the tree sequence.
        edges_iter = iter(self.ts.edges())
        e = next(edges_iter)
        parent_should_be_added = True
        node_times = self.tables.nodes.time

        # Iterate over the edges.
        while e is not None:

            current_parent = e.parent
            current_time = node_times[current_parent]
            if current_time > self.max_time:
                # Stop looking for IBD segments once the
                # processed nodes are older than the max time.
                break

            seg = Segment(e.left, e.right, e.child)

            # Create a SegmentList() holding all segments that descend from seg.
            list_to_add = SegmentList()
            u = seg.node
            if self.sample_id_map[u] != tskit.NULL:
                list_to_add.add(seg)
            else:
                if self.A[u] is not None:
                    s = self.A[u].head
                    while s is not None:
                        intvl = (
                            max(seg.left, s.left),
                            min(seg.right, s.right),
                        )
                        if intvl[1] - intvl[0] > 0:
                            list_to_add.add(Segment(intvl[0], intvl[1], s.node))
                        s = s.next

            if list_to_add.head is not None:
                self.calculate_ibd_segs(current_parent, list_to_add)

            # For parents that are also samples
            if (
                self.sample_id_map[current_parent] != tskit.NULL
            ) and parent_should_be_added:
                singleton_seg = SegmentList()
                singleton_seg.add(Segment(0, self.ts.sequence_length, current_parent))
                self.calculate_ibd_segs(current_parent, singleton_seg)
                parent_should_be_added = False

            # Move to next edge.
            e = next(edges_iter, None)

            # Remove any processed nodes that are no longer needed.
            # Update parent_should_be_added.
            if e is not None and e.parent != current_parent:
                parent_should_be_added = True
                for i, n in enumerate(self.oldest_parent):
                    if current_parent == n:
                        self.A[i] = None

        return self.ibd_segments

    def calculate_ibd_segs(self, current_parent, list_to_add):
        """
        Write later.
        """

        if list_to_add.head is None:
            return []

        if self.A[current_parent] is None:
            self.A[current_parent] = list_to_add

        else:
            seg0 = self.A[current_parent].head
            while seg0 is not None:
                seg1 = list_to_add.head
                while seg1 is not None:
                    left = max(seg0.left, seg1.left)
                    right = min(seg0.right, seg1.right)
                    if left >= right:
                        seg1 = seg1.next
                        continue
                    nodes = [seg0.node, seg1.node]
                    nodes.sort()

                    # If there are any overlapping segments, record as a new
                    # IBD relationship.
                    if right - left > self.min_length:
                        self.add_ibd_segments(
                            nodes[0], nodes[1], Segment(left, right, current_parent),
                        )
                    seg1 = seg1.next
                seg0 = seg0.next

            # Add list_to_add to A[u].
            self.A[current_parent].add(list_to_add)


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
        dest="min_length",
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
    if args.min_length is None:
        min_length = 0
    else:
        min_length = args.min_length[0]
    if args.max_time is None:
        max_time = None
    else:
        max_time = args.max_time[0]

    s = IbdFinder(ts, samples=ts.samples(), min_length=min_length, max_time=max_time)
    all_segs = s.find_ibd_segments()

    if args.samples is None:
        print(all_segs)
    else:
        samples = args.samples
        print(all_segs[(samples[0], samples[1])])
