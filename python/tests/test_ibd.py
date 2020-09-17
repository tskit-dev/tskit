"""
Tests of IBD finding algorithms.
"""
import io
import itertools
import random
import unittest

import msprime

import tests.ibd as ibd
import tests.test_wright_fisher as wf
import tskit

# Functions for computing IBD 'naively'.


def get_ibd(
    sample0,
    sample1,
    treeSequence,
    min_length=0,
    max_time=None,
    path_ibd=True,
    mrca_ibd=True,
):
    """
    Returns all IBD segments for a given pair of nodes in a tree
    using a naive algorithm.
    Note: This function probably looks more complicated than it needs to be --
    This is because it also calculates other 'versions' of IBD (mrca_ibd=False,
    path_ibd=False) that we have't implemented properly yet.
    """

    ibd_list = []
    ts, node_map = treeSequence.simplify(
        samples=[sample0, sample1], keep_unary=True, map_nodes=True
    )
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
                elif prev_dict is not None and subtrees_are_equal(
                    t, prev_dict, node_id
                ):
                    interval_list[-1] = (prev_int[0], current_int[1])
                else:
                    interval_list.append(current_int)

            prev_dict = t.get_parent_dict()

        for interval in interval_list:
            if min_length == 0 or interval[1] - interval[0] > min_length:
                orig_id = node_map.index(node_id)
                ibd_list.append(ibd.Segment(interval[0], interval[1], orig_id))

    return ibd_list


def get_ibd_all_pairs(
    treeSequence,
    samples=None,
    min_length=0,
    max_time=None,
    path_ibd=True,
    mrca_ibd=False,
):

    """
    Returns all IBD segments for all pairs of nodes in a tree sequence
    using the naive algorithm above.
    """

    ibd_dict = {}

    if samples is None:
        samples = treeSequence.samples().tolist()

    pairs = itertools.combinations(samples, 2)
    for pair in pairs:
        ibd_list = get_ibd(
            pair[0],
            pair[1],
            treeSequence,
            min_length=min_length,
            max_time=max_time,
            path_ibd=path_ibd,
            mrca_ibd=mrca_ibd,
        )
        if len(ibd_list) > 0:
            ibd_dict[pair] = ibd_list

    return ibd_dict


def subtrees_are_equal(tree1, pdict0, root):
    """
    Checks for equality of two subtrees beneath a given root node.
    """
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
            if p0 != p1:
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
    ibd0 = ibd.IbdFinder(ts, samples=ts.samples())
    ibd0 = ibd0.find_ibd_segments()
    ibd1 = get_ibd_all_pairs(ts, path_ibd=True, mrca_ibd=True)

    # Convert each SegmentList object into a list of Segment objects.
    ibd0_tolist = {}
    for key, val in ibd0.items():
        if val is not None:
            ibd0_tolist[key] = convert_segmentlist_to_list(val)

    # Check for equality.
    for key0, val0 in ibd0_tolist.items():

        assert key0 in ibd1.keys()
        val1 = ibd1[key0]
        val0.sort()
        val1.sort()


def convert_segmentlist_to_list(seglist):
    """
    Turns a SegmentList object into a list of Segment objects.
    (This makes them easier to compare for testing purposes)
    """
    outlist = []
    if seglist is None:
        return outlist
    else:
        seg = seglist.head
        outlist = [seg]
        seg = seg.next
        while seg is not None:
            outlist.append(seg)
            seg = seg.next

    return outlist


def convert_dict_of_segmentlists(dict0):
    """
    Turns a dictionary of SegmentList objects into a dictionary of lists of
    Segment objects. (makes them easier to compare in tests).
    """
    dict_out = {}
    for key, val in dict0.items():
        dict_out[key] = convert_segmentlist_to_list(val)

    return dict_out


def ibd_is_equal(dict1, dict2):
    """
    Verifies that two dictionaries have the same keys, and that
    the set of items corresponding to each key is identical.
    Used to check identical IBD output.
    NOTE: is there a better/neater way to do this???
    """
    if len(dict1) != len(dict2):
        return False
    for key1, val1 in dict1.items():
        if key1 not in dict2.keys():
            return False
        val2 = dict2[key1]
        if not segment_lists_are_equal(val1, val2):
            return False

    return True


def segment_lists_are_equal(val1, val2):
    """
    Returns True if the two lists hold the same set of segments, otherwise
    returns False.
    """

    if len(val1) != len(val2):
        return False

    val1.sort()
    val2.sort()

    if val1 is None:  # get rid of this later -- we don't any empty dict values!
        if val2 is not None:
            return False
    elif val2 is None:
        if val1 is not None:
            return False
    for i in range(len(val1)):
        if val1[i] != val2[i]:
            return False

    return True


class TestIbdSingleBinaryTree(unittest.TestCase):

    #
    # 2        4
    #         / \
    # 1      3   \
    #       / \   \
    # 0    0   1   2
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    # Basic test
    def test_defaults(self):
        ibd_f = ibd.IbdFinder(self.ts)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {
            (0, 1): [ibd.Segment(0.0, 1.0, 3)],
            (0, 2): [ibd.Segment(0.0, 1.0, 4)],
            (1, 2): [ibd.Segment(0.0, 1.0, 4)],
        }
        assert ibd_is_equal(ibd_segs, true_segs)

    # Max time = 1.5
    def test_time(self):
        ibd_f = ibd.IbdFinder(self.ts, max_time=1.5)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {(0, 1): [ibd.Segment(0.0, 1.0, 3)], (0, 2): [], (1, 2): []}
        assert ibd_is_equal(ibd_segs, true_segs)

    # Min length = 2
    def test_length(self):
        ibd_f = ibd.IbdFinder(self.ts, min_length=2)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {(0, 1): [], (0, 2): [], (1, 2): []}
        assert ibd_is_equal(ibd_segs, true_segs)


class TestIbdTwoSamplesTwoTrees(unittest.TestCase):

    # 2
    #             |     3
    # 1      2    |    / \
    #       / \   |   /   \
    # 0    0   1  |  0     1
    # |------------|----------|
    # 0.0          0.4        1.0
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
    0       0.4     2       0,1
    0.4     1.0     3       0,1
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    # Basic test
    def test_basic(self):
        ibd_f = ibd.IbdFinder(self.ts)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {(0, 1): [ibd.Segment(0.0, 0.4, 2), ibd.Segment(0.4, 1.0, 3)]}
        assert ibd_is_equal(ibd_segs, true_segs)

    # Max time = 1.2
    def test_time(self):
        ibd_f = ibd.IbdFinder(self.ts, max_time=1.2)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {(0, 1): [ibd.Segment(0.0, 0.4, 2)]}
        assert ibd_is_equal(ibd_segs, true_segs)

    # Min length = 0.5
    def test_length(self):
        ibd_f = ibd.IbdFinder(self.ts, min_length=0.5)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {(0, 1): [ibd.Segment(0.4, 1.0, 3)]}
        assert ibd_is_equal(ibd_segs, true_segs)


class TestIbdUnrelatedSamples(unittest.TestCase):

    #
    #    2   3
    #    |   |
    #    0   1

    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       0           1
    3       0           1
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       2       0
    0       1       3       1
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_basic(self):
        ibd_f = ibd.IbdFinder(self.ts)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {(0, 1): []}
        assert ibd_is_equal(ibd_segs, true_segs)

    def test_time(self):
        ibd_f = ibd.IbdFinder(self.ts, max_time=1.2)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {(0, 1): []}
        assert ibd_is_equal(ibd_segs, true_segs)

    def test_length(self):
        ibd_f = ibd.IbdFinder(self.ts, min_length=0.2)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {(0, 1): []}
        assert ibd_is_equal(ibd_segs, true_segs)


class TestIbdNoSamples(unittest.TestCase):
    def test_no_samples(self):
        #
        #     2
        #    / \
        #   /   \
        #  /     \
        # (0)   (1)
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       0           0
        1       0           0
        2       0           1
        3       0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       2       0
        0       1       3       1
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        with self.assertRaises(ValueError):
            ibd.IbdFinder(ts)


class TestIbdSamplesAreDescendants(unittest.TestCase):
    #
    # 4     5
    # |     |
    # 2     3
    # |     |
    # 0     1
    #
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           1
    3       1           1
    4       0           2
    5       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       2       0
    0       1       3       1
    0       1       4       2
    0       1       5       3
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_basic(self):
        ts = self.ts
        ibd_f = ibd.IbdFinder(ts)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {
            (0, 1): [],
            (0, 2): [ibd.Segment(0.0, 1.0, 2)],
            (0, 3): [],
            (1, 2): [],
            (1, 3): [ibd.Segment(0.0, 1.0, 3)],
            (2, 3): [],
        }

        assert ibd_is_equal(ibd_segs, true_segs)


class TestIbdDifferentPaths(unittest.TestCase):
    #
    #        4       |      4       |        4
    #       / \      |     / \      |       / \
    #      /   \     |    /   3     |      /   \
    #     /     \    |   2     \    |     /     \
    #    /       \   |  /       \   |    /       \
    #   0         1  | 0         1  |   0         1
    #                |              |
    #                0.2            0.7

    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       0           1
    3       0           1.5
    4       0           2.5
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0.2     0.7     2       0
    0.2     0.7     3       1
    0.0     0.2     4       0
    0.0     0.2     4       1
    0.7     1.0     4       0
    0.7     1.0     4       1
    0.2     0.7     4       2
    0.2     0.7     4       3
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        ts = self.ts
        ibd_f = ibd.IbdFinder(ts)
        ibd_segs = ibd_f.find_ibd_segments()
        true_segs = {
            (0, 1): [
                ibd.Segment(0.0, 0.2, 4),
                ibd.Segment(0.7, 1.0, 4),
                ibd.Segment(0.2, 0.7, 4),
            ]
        }
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        assert ibd_is_equal(ibd_segs, true_segs)

    def test_time(self):
        ts = self.ts
        ibd_f = ibd.IbdFinder(ts, max_time=1.8)
        ibd_segs = ibd_f.find_ibd_segments()
        true_segs = {(0, 1): []}
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        assert ibd_is_equal(ibd_segs, true_segs)
        ibd_f = ibd.IbdFinder(ts, max_time=2.8)
        ibd_segs = ibd_f.find_ibd_segments()
        true_segs = {
            (0, 1): [
                ibd.Segment(0.0, 0.2, 4),
                ibd.Segment(0.7, 1.0, 4),
                ibd.Segment(0.2, 0.7, 4),
            ]
        }
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        assert ibd_is_equal(ibd_segs, true_segs)

    def test_length(self):
        ts = self.ts
        ibd_f = ibd.IbdFinder(ts, min_length=0.4)
        ibd_segs = ibd_f.find_ibd_segments()
        true_segs = {(0, 1): [ibd.Segment(0.2, 0.7, 4)]}
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        assert ibd_is_equal(ibd_segs, true_segs)


class TestIbdPolytomies(unittest.TestCase):
    #
    #          5         |         5
    #         / \        |        / \
    #        4   \       |       4   \
    #       /|\   \      |      /|\   \
    #      / | \   \     |     / | \   \
    #     /  |  \   \    |    /  |  \   \
    #    /   |   \   \   |   /   |   \   \
    #   0    1    2   3  |  0    1    3   2
    #                    |
    #                   0.3

    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       0           2.5
    5       0           3.5
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0.0     1.0     4       0
    0.0     1.0     4       1
    0.0     0.3     4       2
    0.3     1.0     4       3
    0.3     1.0     5       2
    0.0     0.3     5       3
    0.0     1.0     5       4
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        ts = self.ts
        ibd_f = ibd.IbdFinder(ts)
        ibd_segs = ibd_f.find_ibd_segments()
        # print(ibd_segs[(0,1)])
        true_segs = {
            (0, 1): [ibd.Segment(0, 1, 4)],
            (0, 2): [ibd.Segment(0, 0.3, 4), ibd.Segment(0.3, 1, 5)],
            (0, 3): [ibd.Segment(0, 0.3, 5), ibd.Segment(0.3, 1, 4)],
            (1, 2): [ibd.Segment(0, 0.3, 4), ibd.Segment(0.3, 1, 5)],
            (1, 3): [ibd.Segment(0, 0.3, 5), ibd.Segment(0.3, 1, 4)],
            (2, 3): [ibd.Segment(0.3, 1, 5), ibd.Segment(0, 0.3, 5)],
        }
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        # print(ibd_segs)
        assert ibd_is_equal(ibd_segs, true_segs)

    def test_time(self):
        ts = self.ts
        ibd_f = ibd.IbdFinder(ts, max_time=3)
        ibd_segs = ibd_f.find_ibd_segments()
        true_segs = {
            (0, 1): [ibd.Segment(0, 1, 4)],
            (0, 2): [ibd.Segment(0, 0.3, 4)],
            (0, 3): [ibd.Segment(0.3, 1, 4)],
            (1, 2): [ibd.Segment(0, 0.3, 4)],
            (1, 3): [ibd.Segment(0.3, 1, 4)],
            (2, 3): [],
        }
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        assert ibd_is_equal(ibd_segs, true_segs)

    def test_length(self):
        ts = self.ts
        ibd_f = ibd.IbdFinder(ts, min_length=0.5)
        ibd_segs = ibd_f.find_ibd_segments()
        true_segs = {
            (0, 1): [ibd.Segment(0, 1, 4)],
            (0, 2): [ibd.Segment(0.3, 1, 5)],
            (0, 3): [ibd.Segment(0.3, 1, 4)],
            (1, 2): [ibd.Segment(0.3, 1, 5)],
            (1, 3): [ibd.Segment(0.3, 1, 4)],
            (2, 3): [ibd.Segment(0.3, 1, 5)],
        }
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        assert ibd_is_equal(ibd_segs, true_segs)


class TestIbdInternalSamples(unittest.TestCase):
    #
    #
    #      3
    #     / \
    #    /   2
    #   /     \
    #  0      (1)
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       0           0
    2       1           1
    3       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0.0     1.0     2       1
    0.0     1.0     3       0
    0.0     1.0     3       2
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        ts = self.ts
        ibd_f = ibd.IbdFinder(ts)
        ibd_segs = ibd_f.find_ibd_segments()
        ibd_segs = convert_dict_of_segmentlists(ibd_segs)
        true_segs = {
            (0, 2): [ibd.Segment(0, 1, 3)],
        }
        assert ibd_is_equal(ibd_segs, true_segs)


class TestIbdRandomExamples(unittest.TestCase):
    """
    Randomly generated test cases.
    """

    # Infinite sites, Hudson model.
    def test_random_example1(self):
        ts = msprime.simulate(sample_size=10, recombination_rate=0.5, random_seed=2)
        verify_equal_ibd(ts)

    def test_random_example2(self):
        ts = msprime.simulate(sample_size=10, recombination_rate=0.5, random_seed=23)
        verify_equal_ibd(ts)

    def test_random_example3(self):
        ts = msprime.simulate(sample_size=10, recombination_rate=0.5, random_seed=232)
        verify_equal_ibd(ts)

    def test_random_example4(self):
        ts = msprime.simulate(sample_size=10, recombination_rate=0.3, random_seed=726)
        verify_equal_ibd(ts)

    # Finite sites
    def sim_finite_sites(self, random_seed, dtwf=False):
        seq_length = int(1e5)
        positions = random.sample(range(1, seq_length), 98) + [0, seq_length]
        positions.sort()
        rates = [random.uniform(1e-9, 1e-7) for _ in range(100)]
        r_map = msprime.RecombinationMap(
            positions=positions, rates=rates, num_loci=seq_length
        )
        if dtwf:
            model = "dtwf"
        else:
            model = "hudson"
        ts = msprime.simulate(
            sample_size=10,
            recombination_map=r_map,
            Ne=10,
            random_seed=random_seed,
            model=model,
        )
        return ts

    def test_finite_sites1(self):
        ts = self.sim_finite_sites(9257)
        verify_equal_ibd(ts)

    def test_finite_sites2(self):
        ts = self.sim_finite_sites(835)
        verify_equal_ibd(ts)

    def test_finite_sites3(self):
        ts = self.sim_finite_sites(27278)
        verify_equal_ibd(ts)

    def test_finite_sites4(self):
        ts = self.sim_finite_sites(22446688)
        verify_equal_ibd(ts)

    # DTWF
    def test_dtwf1(self):
        ts = self.sim_finite_sites(84, dtwf=True)
        verify_equal_ibd(ts)

    def test_dtwf2(self):
        ts = self.sim_finite_sites(17482, dtwf=True)
        verify_equal_ibd(ts)

    def test_dtwf3(self):
        ts = self.sim_finite_sites(846, dtwf=True)
        verify_equal_ibd(ts)

    def test_dtwf4(self):
        ts = self.sim_finite_sites(273, dtwf=True)
        verify_equal_ibd(ts)

    def test_sim_wright_fisher_generations(self):
        # Uses the bespoke DTWF forward-time simulator.
        number_of_gens = 3
        tables = wf.wf_sim(4, number_of_gens, deep_history=False, seed=83)
        tables.sort()
        ts = tables.tree_sequence()
        verify_equal_ibd(ts)

    def test_sim_wright_fisher_generations2(self):
        # Uses the bespoke DTWF forward-time simulator.
        number_of_gens = 10
        tables = wf.wf_sim(10, number_of_gens, deep_history=False, seed=837)
        tables.sort()
        ts = tables.tree_sequence()
        verify_equal_ibd(ts)

    def test_sim_wright_fisher_generations3(self):
        # Uses the bespoke DTWF forward-time simulator.
        number_of_gens = 10
        tables = wf.wf_sim(10, number_of_gens, deep_history=False, seed=37)
        tables.sort()
        ts = tables.tree_sequence()
        verify_equal_ibd(ts)
