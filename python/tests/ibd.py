# MIT License
#
# Copyright (c) 2018-2020 Tskit Developers
# Copyright (c) 2015-2018 University of Oxford
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

import tskit
import numpy as np
import itertools
import sys
import argparse


class Segment(object):
    """
    A class representing a single segment. Each segment has a left and right,
    denoting the loci over which it spans, a node and a next, giving the next
    in the chain.

    The node it records is the *output* node ID.
    """
    def __init__(self, left=None, right=None, node=None, next=None):
        self.left = left
        self.right = right
        self.node = node
        self.next = next

    def __str__(self):
        s = "({}-{}->{}:next={})".format(
            self.left, self.right, self.node, repr(self.next))
        return s

    def __repr__(self):
        return repr((self.left, self.right, self.node))

    def __eq__(self, other):
        return (self.left == other.left and self.right == other.right and self.node == other.node)

    def __lt__(self, other):
        return (self.node, self.left, self.right) < (other.node, other.left, other.right)


class SegmentList(object):
    """
    A class representing a list of segments that are descended from a given ancestral node
    via a particular child of the ancestor.
    """
    def __init__(self, head=None, tail=None, next=None):
        self.head = head
        self.tail = tail
        self.next = next

    def __str__(self):
        s = "head={},tail={},next={}".format(
            self.head, self.right, repr(self.next))
        return s

    def __repr__(self):
        s = "[{}...]".format(repr(self.next.head))
        return s



class IbdFinder(object):
    """
    Finds all IBD relationships between specified samples in a tree sequence.
    """

    def __init__(
        self,
        ts,
        samples=None,
        min_length=0,
        max_time=None):

        self.ts = ts
        if samples is None:
            samples = ts.samples()
        else:
            self.samples = samples
        self.min_length = min_length
        self.max_time = max_time
        self.current_parent = self.ts.tables.edges.parent[0]
        self.current_time = 0
        self.A = [[] for _ in range(ts.num_nodes)] # Descendant segments
        self.tables = self.ts.tables
        # self.ibd_segments = dict.fromkeys(itertools.combinations(self.ts.samples(), 2), [])


    def find_ibd_segments(self, min_length=0, max_time=None):
        
        # 1
        # A = [[] for n in range(0, self.ts.num_nodes)]
        ibd_segments = dict.fromkeys(itertools.combinations(self.ts.samples(), 2), [])
        edges = self.ts.edges()  
        parent_list = self.list_of_parents() ## Needed for memory-pruning step
        
        # 2
        mygen = iter(self.ts.edges())
        e = next(mygen)
            
        # 3
        while e is not None:

            # 3a
            S_head = None
            S_tail = None
            self.current_parent = e.parent
            self.current_time = self.tables.nodes.time[self.current_parent]
            if self.max_time is not None and self.current_time > self.max_time:
                # Stop looking for IBD segments once the
                # processed nodes are older than the max time.
                break
            
            # 3b
            while e is not None and self.current_parent == e.parent:
                # Create the list S of immediate descendants of u.
                seg = Segment(e.left, e.right, e.child)
                if S_head is None:
                    S_head = seg
                    S_tail = seg
                else:
                    S_tail.next = seg
                    S_tail = S_tail.next
                if e.id < self.ts.num_edges - 1:
                    e = next(mygen)
                    continue
                else:
                    e = None

            # 3c
            while S_head is not None:
                # Create A[u] from S.
                # Do we still need to do the initialisation if the below is there??
                u = S_head.node
                if u in self.ts.samples():
                    self.A[self.current_parent].append([S_head])
                else:
                    list_to_add = []
                    for s in self.A[u]:
                        l = (max(S_head.left, s.left), min(S_head.right, s.right))
                        if l[1] - l[0] > 0:
                            list_to_add.append(Segment(l[0], l[1], s.node))
                    self.A[self.current_parent].append(list_to_add)
                S_head = S_head.next

            # d. Squash
            # A[self.current_parent] = self.squash(self.A[self.current_parent])
            
            # e. Process A[self.current_parent]
            if len(self.A[self.current_parent]) > 1:
                new_segs, nodes_to_remove = self.update_A_and_find_ibd_segs(ibd_segments)
                
                # e. Add any new IBD segments discovered.
                for key, val in new_segs.items():
                    for v in val:
                        if len(ibd_segments[key]) == 0:
                            ibd_segments[key] = [v]
                        else:
                            ibd_segments[key].append(v)
                            
                # g. Remove elements of self.A[u] if they are no longer needed.
                ## (memory-pruning step)
                for n in nodes_to_remove:
                    if self.current_parent in parent_list[n]:
                        parent_list[n].remove(self.current_parent)
                    if len(parent_list[n]) == 0:
                        self.A[n] = []
                        
            # Unlist the ancestral segments in self.A.
            self.A[self.current_parent] = list(itertools.chain(*self.A[self.current_parent]))

        # 4
        return ibd_segments


    def update_A_and_find_ibd_segs(self, ibd_segments):
    
        new_segments = dict.fromkeys(itertools.combinations(self.ts.samples(), 2), [])
        num_coalescing_sets = len(self.A[self.current_parent])
        index_pairs = list(itertools.combinations(range(0, num_coalescing_sets), 2))

        for setpair in index_pairs:
            for seg0 in self.A[self.current_parent][setpair[0]]:
                for seg1 in self.A[self.current_parent][setpair[1]]:

                    if seg0.node == seg1.node:
                        continue
                    left = max(seg0.left, seg1.left)
                    right = min(seg0.right, seg1.right)
                    if left >= right:
                        continue
                    nodes = [seg0.node, seg1.node]
                    nodes.sort()

                    # if mrca_ibd:
                        # pass # for now
                        # existing_segs = ibd_segments[(nodes[0], nodes[1])].copy()
                        # if right - left > self.min_length:
                        #     if len(existing_segs) == 0:
                        #         new_segments[(nodes[0], nodes[1])] = [Segment(left, right, self.current_parent)]
                        #         existing_segs.append(Segment(left, right, self.current_parent))
                        #     else:
                        #         for i in existing_segs:
                        #             # no overlap.
                        #             if right <= i.left or left >= i.right:
                        #                 if len(new_segments[(nodes[0], nodes[1])]) == 0:
                        #                     new_segments[(nodes[0], nodes[1])] = [Segment(left, right, self.current_parent)]
                        #                 else:
                        #                     new_segments[(nodes[0], nodes[1])].append(Segment(left, right, self.current_parent))
                        #                 existing_segs.append(Segment(left, right, self.current_parent))
                        #             # partial overlap -- does this even happen?
                        #             elif (left < i.left and right < i.right) or (i.left < left and i.right < right):
                        #                 print('partial overlap')
                                    # Yes, but I think it's okay to leave these segments...
                    # else:
                    if right - left > self.min_length:
                        if len(new_segments[(nodes[0], nodes[1])]) == 0:
                            new_segments[(nodes[0], nodes[1])] = [Segment(left, right, self.current_parent)]
                        else:
                            new_segments[(nodes[0], nodes[1])].append(Segment(left, right, self.current_parent))

        # iv. specify elements of A that can be removed (for memory-pruning step)
        processed_child_nodes = []
        for seglist in self.A[self.current_parent]:
            processed_child_nodes += [seg.node for seg in seglist]
            processed_child_nodes = list(set(processed_child_nodes))

        return new_segments, processed_child_nodes


    def list_of_parents(self):
        parents = [[] for i in range(0, self.ts.num_nodes)]
        edges = self.ts.tables.edges
        for e in edges:
            if len(parents[e.child]) == 0 or e.parent != parents[e.child][-1]:
                parents[e.child].append(e.parent)
        return parents


    # def squash(self, segment_lists):
    
    #     # Concatenate the input lists and record the number of
    #     # segments in each.
    #     A_u = []
    #     num_desc_edges = []
    #     for L in segment_lists:
    #         for l in L:
    #             A_u.append(l)
    #         num_desc_edges.append(len(L))
            
    #     # Sort the list, keeping track of the original order.
    #     sorted_A = sorted(enumerate(A_u), key=lambda i:i[1])
        
    #     # Squash the list.
    #     next_ind = len(sorted_A)
    #     inds_to_remove = []
    #     ind = 1
    #     while ind < len(sorted_A):
    #         if sorted_A[ind][1].node == sorted_A[ind - 1][1].node:
    #             if sorted_A[ind][1].right > sorted_A[ind - 1][1].right and\
    #                 sorted_A[ind][1].left <= sorted_A[ind - 1][1].right:
    #                 # Squash the previous int into the current one.
    #                 sorted_A[ind][1].left = sorted_A[ind - 1][1].left
    #                 # Flag the interval to be removed.
    #                 inds_to_remove.append(ind - 1)
    #                 # Change order index.
    #                 sorted_A[ind] = (next_ind, sorted_A[ind][1])
    #                 next_ind += 1
    #         ind += 1
            
    #     # Remove any unnecessary list items.
    #     for i in reversed(inds_to_remove):
    #         # Needs to be done in reverse order!!
    #         sorted_A.pop(i)

    #     # Restore the original order as lists of lists.
    #     cum_sum = np.cumsum(num_desc_edges)
    #     squashed_sorted_A = [[] for _ in range(0, next_ind)]
    #     for a in sorted_A:
    #         ind = a[0]
    #         if ind < cum_sum[-1]:
    #             s = 0
    #             while s < len(cum_sum):
    #                 if a[0] < cum_sum[s]:
    #                     squashed_sorted_A[s].append(a[1])
    #                     break
    #                 s += 1
                    
    #         else:
    #             squashed_sorted_A[ind].append(a[1])
                
    #     # Remove lists of length 0.
    #     squashed_sorted_A = [_ for _ in squashed_sorted_A if len(_) > 0]
                
    #     return squashed_sorted_A
    

if __name__ == "__main__":
    # Simple CLI for running IBDFinder.

    parser = argparse.ArgumentParser(description="Command line interface for the IBDFinder.")

    parser.add_argument('--infile',
                        type=str,
                        dest='infile',
                        nargs=1,
                        metavar="IN_FILE",
                        help="The tree sequence to be analysed.")

    parser.add_argument('--min-length',
                        type=float,
                        dest='min_length',
                        nargs=1,
                        metavar="MIN_LENGTH",
                        help="Only segments longer than this cutoff will be returned.")

    parser.add_argument('--max-time',
                        type=float,
                        dest='max_time',
                        nargs=1,
                        metavar="MAX_TIME",
                        help="Only segments younger this time will be returned.")

    parser.add_argument('--samples',
                        type=int,
                        dest='samples',
                        nargs=2,
                        metavar="SAMPLES",
                        help="If provided, only IBD relationships between the given node pair are returned."
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

    s = IbdFinder(ts, samples = ts.samples(),
        min_length=min_length, max_time=max_time)
    all_segs = s.find_ibd_segments()


    if args.samples is None:
        print(all_segs)
    else:
        samples = args.samples
        print(all_segs[(samples[0], samples[1])])


