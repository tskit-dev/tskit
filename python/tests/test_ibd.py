
"""
Tests of IBD finding algorithms.
"""
import unittest
import sys
import random
import io
import itertools

import tests as tests
import tests.ibd as ibd

import tskit
import msprime
import numpy as np

# Functions for computing IBD 'naively'.

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

    def __lt__(self, other):
        return (self.node, self.left, self.right) < (other.node, other.left, other.right)


def get_ibd(sample0, sample1, treeSequence, min_length=0, max_time=None,
           path_ibd=True, mrca_ibd=False):
    """
    Returns all IBD segments for a given pair of nodes in a tree
    using a naive algorithm.
    """

    ibd_list = []
    ts, node_map = treeSequence.simplify(samples=[sample0, sample1], keep_unary=True,
                              map_nodes=True)
    node_map = node_map.tolist()
    
    for n in ts.nodes():
        
        if max_time is not None and n.time > max_time:
            break
            
        node_id = n.id
        interval_list = []
        if n.flags == 1:
            continue

        prev_dict = None
        for t in ts.trees():

            if len(list(t.nodes(n.id))) == 1 or t.num_samples(n.id) < 2:
                continue
            if mrca_ibd and n.id != t.mrca(0, 1):
                    continue

            current_int = t.get_interval()
            if len(interval_list) == 0:
                interval_list.append(current_int)
            else:
                prev_int = interval_list[-1]
                if not path_ibd and prev_int[1] == current_int[0]:
                    interval_list[-1] = (prev_int[0], current_int[1])
                elif prev_dict is not None and subtrees_are_equal(t, prev_dict, node_id):
                    interval_list[-1] = (prev_int[0], current_int[1])
                else:
                    interval_list.append(current_int)
                    
            prev_dict = t.get_parent_dict()
                    
        for interval in interval_list:
            if min_length == 0 or interval[1] - interval[0] > min_length:
                orig_id = node_map.index(node_id)
                ibd_list.append(Segment(interval[0], interval[1], orig_id))
        
    return(ibd_list)


def get_ibd_all_pairs(treeSequence, samples=None, min_length=0, max_time=None,
                      path_ibd=True, mrca_ibd=False):
    
    ibd_dict = {}
    
    if samples is None:
        samples = treeSequence.samples().tolist()
    
    pairs = itertools.combinations(samples, 2)
    for pair in pairs:
        ibd_list = get_ibd(pair[0], pair[1], treeSequence,
                           min_length=min_length, max_time=max_time,
                          path_ibd=path_ibd, mrca_ibd=mrca_ibd)
        if len(ibd_list) > 0:
            ibd_dict[pair] = ibd_list
            
    return(ibd_dict)


def subtrees_are_equal(tree1, pdict0, root):
    pdict1 = tree1.get_parent_dict()
    if root not in pdict0.values() or root not in pdict1.values():
        return False
    leaves1 = set(tree1.leaves(root))
    for l in leaves1:
        node = l
        while node != root:
            p1 = pdict1[node]
            if p1 not in pdict0.values():
                return False
            p0 = pdict0[node]
            if  p0 != p1:
                return False  
            node = p1
            
    return True


def verify_equal_ibd(treeSequence):
    """
    Calculates IBD segments using both the 'naive' and sophisticated algorithms, 
    verifies that the same output is produced.
    NB: May be good to expand this in the future so that many different combos
    of IBD options are tested simultaneously (all the MRCA and path-IBD combos),
    for example.
    """
    ts = treeSequence
    ibd0 = ibd.IbdFinder(ts, samples = ts.samples())
    ibd0 = ibd0.find_ibd_segments_of_length()
    ibd1 = get_ibd_all_pairs(ts, path_ibd=True, mrca_ibd=True)

    for key0, val0 in ibd0.items():
        # print(key0)
        assert key0 in ibd1.keys()
        val1 = ibd1[key0]
        val0.sort()
        val1.sort()

        # print('IBD from IBDFinder')
        # print(val0)
        # print('IBD from naive function')
        # print(val1)

        if val0 is None: # Get rid of this later -- don't want empty dict values at all
            assert val1 is None
            continue
        elif val1 is None:
        #     print(val0)
        #     print(val1)
            assert val0 is None
        assert len(val0) == len(val1)
        for i in range(0, len(val0)):
            assert val0[i] == val1[i]


class TestIbdByLength(unittest.TestCase):
    """
    Tests of length-based IBD function. 
    """
    #        11           *                   *                   *
    #       /  \          *                   *                   *        10
    #      /    \         *        9          *                   *       /  \
    #     /      \        *       / \         *         8         *      /    8
    #    |        |       *      /   \        *        / \        *     /    / \
    #    |        7       *     /     7       *       /   7       *    /    /   7 
    #    |       / \      *    |     / \      *      /   / \      *   /    /   / \
    #    |      /   6     *    |    /   6     *     /   /   6     *  |    /   |   |    
    #    5     |   / \    *    5   |   / \    *    5   |   / \    *  |    5   |   |    
    #   / \    |  /   \   *   / \  |  /   \   *   / \  |  /   \   *  |   / \  |   |     
    #  0   4   1 2     3  *  0   4 1 2     3  *  0   4 1 2     3  *  3  0   4 1   2
    # 
    # ------------------------------------------------------------------------------
    # 0                  0.10                 0.50               0.75           1.00 

    small_tree_ex_nodes = """\
    id  flags   population  individual  time
    0   1   0   -1  0.00000000000000    
    1   1   0   -1  0.00000000000000    
    2   1   0   -1  0.00000000000000    
    3   1   0   -1  0.00000000000000    
    4   1   0   -1  0.00000000000000    
    5   0   0   -1  1.00000000000000    
    6   0   0   -1  2.00000000000000   
    7   0   0   -1  3.00000000000000    
    8   0   0   -1  4.00000000000000    
    9   0   0   -1  5.00000000000000    
    10  0   0   -1  6.00000000000000    
    11  0   0   -1  7.00000000000000
    """
    small_tree_ex_edges = """\
    id  left        right       parent  child
    0   0.00000000  1.00000000  5   0
    1   0.00000000  1.00000000  5   4
    2   0.00000000  0.75000000  6   2
    3   0.00000000  0.75000000  6   3
    4   0.00000000  1.00000000  7   1
    5   0.75000000  1.00000000  7   2
    6   0.00000000  0.75000000  7   6
    7   0.50000000  1.00000000  8   5
    8   0.50000000  1.00000000  8   7
    9   0.10000000  0.50000000  9   5
    10  0.10000000  0.50000000  9   7
    11  0.75000000  1.00000000  10  3
    12  0.75000000  1.00000000  10  8
    13  0.00000000  0.10000000  11  5
    14  0.00000000  0.10000000  11  7
    """


    def test_canonical_example1(self):
        ts0 = msprime.simulate(sample_size=5, recombination_rate=.5, random_seed=2)
        verify_equal_ibd(ts0)

    def test_canonical_example2(self):
        ts1 = msprime.simulate(sample_size=5, recombination_rate=.5, random_seed=23)
        verify_equal_ibd(ts1)

    def test_canonical_example3(self):
        ts2 = msprime.simulate(sample_size=5, recombination_rate=.5, random_seed=232)
        verify_equal_ibd(ts2)

    def test_random_example(self):
        ts_r = msprime.simulate(sample_size=10, recombination_rate=.3, random_seed=726)
        verify_equal_ibd(ts_r)
