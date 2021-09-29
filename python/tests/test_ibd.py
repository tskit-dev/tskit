"""
Tests of IBD finding algorithms.
"""
import io
import itertools
import random

import msprime
import numpy as np
import pytest

import tests.ibd as ibd
import tests.test_wright_fisher as wf
import tskit

# Functions for computing IBD 'naively'.


def ibd_segments(
    ts,
    *,
    within=None,
    min_length=0,
    max_time=None,
    compare_lib=True,
    print_c=False,
    print_py=False,
):
    """
    Calculates IBD segments using Python and converts output to lists of segments.
    Also compares result with C library.
    """
    ibd_f = ibd.IbdFinder(ts, within=within, max_time=max_time, min_length=min_length)
    ibd_segs = ibd_f.run()
    # ibd_f.print_state()
    if compare_lib:
        c_out = ts.ibd_segments(within=within, max_time=max_time, min_length=min_length)
        if print_c:
            print("C output:\n")
            print(c_out)
        if print_py:
            print("Python output:\n")
            print(ibd_segs)
        assert_ibd_equal(ibd_segs, c_out)
    return ibd_segs


def get_ibd(
    sample0,
    sample1,
    ts,
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
    ts, node_map = ts.simplify(
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
            if min_length == 0 or interval.right - interval.left > min_length:
                orig_id = node_map.index(node_id)
                ibd_list.append(tskit.IbdSegment(interval[0], interval[1], orig_id))

    return ibd_list


def get_ibd_all_pairs(
    ts,
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
        samples = ts.samples().tolist()

    for pair in itertools.combinations(samples, 2):
        ibd_list = get_ibd(
            pair[0],
            pair[1],
            ts,
            min_length=min_length,
            max_time=max_time,
            path_ibd=path_ibd,
            mrca_ibd=mrca_ibd,
        )
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
    for leaf in leaves1:
        node = leaf
        while node != root:
            p1 = pdict1[node]
            if p1 not in pdict0.values():
                return False
            p0 = pdict0[node]
            if p0 != p1:
                return False
            node = p1

    return True


def verify_equal_ibd(ts, within=None, compare_lib=True, print_c=False, print_py=False):
    """
    Calculates IBD segments using both the 'naive' and sophisticated algorithms,
    verifies that the same output is produced.
    """
    ibd0 = ibd_segments(
        ts,
        within=within,
        compare_lib=compare_lib,
        print_c=print_c,
        print_py=print_py,
    )
    ibd1 = get_ibd_all_pairs(ts, path_ibd=True, mrca_ibd=True)
    assert_ibd_equal(ibd0, ibd1)


def convert_ibd_output_to_seglists(ibd_out):
    """
    Converts the Python mock-up output back into lists of segments.
    This is needed to use the ibd_is_equal function.
    """
    out = {}
    for key, value in ibd_out.items():
        out[key] = list(value)
    return out


def assert_ibd_equal(dict1, dict2):
    """
    Verifies that two dictionaries have the same keys, and that
    the set of items corresponding to each key is identical.
    Used to check identical IBD output.
    """
    assert len(dict1) == len(dict2)
    for key, val in dict1.items():
        assert key in dict2
        assert sorted(list(val)) == sorted(list(dict2[key]))


class TestIbdSingleBinaryTree:

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
        true_segs = {
            (0, 1): [tskit.IbdSegment(0.0, 1.0, 3)],
            (0, 2): [tskit.IbdSegment(0.0, 1.0, 4)],
            (1, 2): [tskit.IbdSegment(0.0, 1.0, 4)],
        }
        ibd_segs = ibd_segments(self.ts, within=[0, 1, 2])
        assert_ibd_equal(ibd_segs, true_segs)
        assert_ibd_equal(ibd_segments(self.ts), true_segs)

    def test_time(self):
        ibd_segs = ibd_segments(
            self.ts,
            max_time=1.5,
            compare_lib=True,
        )
        true_segs = {(0, 1): [tskit.IbdSegment(0.0, 1.0, 3)]}
        assert_ibd_equal(ibd_segs, true_segs)

    # Min length = 2
    def test_length(self):
        ibd_segs = ibd_segments(self.ts, min_length=2)
        assert_ibd_equal(ibd_segs, {})

    def test_input_errors(self):
        with pytest.raises(tskit.LibraryError, match="Node out of bounds"):
            self.ts.ibd_segments(within=[-1])
        with pytest.raises(tskit.LibraryError, match="Duplicate sample value"):
            self.ts.ibd_segments(within=[0, 0])

    # A simple test of the Python wrapper.
    def test_ts(self):
        ibd_tab = self.ts.tables.ibd_segments()
        ibd_ts = self.ts.ibd_segments()
        assert len(ibd_tab) == len(ibd_ts)
        for k in ibd_tab.keys():
            assert ibd_tab[k] == ibd_ts[k]


class TestIbdTwoSamplesTwoTrees:

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
        ibd_segs = ibd_segments(self.ts)
        true_segs = {
            (0, 1): [tskit.IbdSegment(0.0, 0.4, 2), tskit.IbdSegment(0.4, 1.0, 3)]
        }
        assert_ibd_equal(ibd_segs, true_segs)

    # Max time = 1.2
    def test_time(self):
        ibd_segs = ibd_segments(self.ts, max_time=1.2, compare_lib=True)
        true_segs = {(0, 1): [tskit.IbdSegment(0.0, 0.4, 2)]}
        assert_ibd_equal(ibd_segs, true_segs)

    # Min length = 0.5
    def test_length(self):
        ibd_segs = ibd_segments(self.ts, min_length=0.5, compare_lib=True)
        true_segs = {(0, 1): [tskit.IbdSegment(0.4, 1.0, 3)]}
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdUnrelatedSamples:

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
        ibd_segs = ibd_segments(self.ts)
        assert len(ibd_segs) == 0

    def test_time(self):
        ibd_segs = ibd_segments(self.ts, max_time=1.2)
        assert len(ibd_segs) == 0

    def test_length(self):
        ibd_segs = ibd_segments(self.ts, min_length=0.2)
        assert len(ibd_segs) == 0


class TestIbdNoSamples:
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
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       2       0
    0       1       2       1
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        result = ibd_segments(self.ts)
        assert len(result) == 0

    def test_specified_samples(self):
        ibd_segs = ibd_segments(self.ts, within=[0, 1])
        true_segs = {
            (0, 1): [
                tskit.IbdSegment(0.0, 1, 2),
            ]
        }
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdSamplesAreDescendants:
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
        ibd_segs = ibd_segments(self.ts)
        true_segs = {
            (0, 2): [tskit.IbdSegment(0.0, 1.0, 2)],
            (1, 3): [tskit.IbdSegment(0.0, 1.0, 3)],
        }

        assert_ibd_equal(ibd_segs, true_segs)

    def test_input_within(self):
        ibd_segs = ibd_segments(self.ts, within=[0, 2, 3, 5])
        true_segs = {
            (0, 2): [tskit.IbdSegment(0.0, 1.0, 2)],
            (3, 5): [tskit.IbdSegment(0.0, 1.0, 5)],
        }
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdDifferentPaths:
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
        ibd_segs = ibd_segments(self.ts)
        true_segs = {
            (0, 1): [
                tskit.IbdSegment(0.0, 0.2, 4),
                tskit.IbdSegment(0.7, 1.0, 4),
                tskit.IbdSegment(0.2, 0.7, 4),
            ]
        }
        assert_ibd_equal(ibd_segs, true_segs)

    def test_time(self):
        ibd_segs = ibd_segments(self.ts, max_time=1.8)
        assert len(ibd_segs) == 0

    def test_length(self):
        ibd_segs = ibd_segments(self.ts, min_length=0.4)
        true_segs = {(0, 1): [tskit.IbdSegment(0.2, 0.7, 4)]}
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdDifferentPaths2:
    #
    #        5         |
    #       / \        |
    #      /   4       |      4
    #     /   / \      |     / \
    #    /   /   \     |    /   \
    #   /   /     \    |   3     \
    #  /   /       \   |  / \     \
    # 0   1         2  | 0   2     1
    #                  |
    #                  0.2

    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2.5
    5       0           3.5
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0.2     1.0     3       0
    0.2     1.0     3       2
    0.0     1.0     4       1
    0.0     0.2     4       2
    0.2     1.0     4       3
    0.0     0.2     5       0
    0.0     0.2     5       4
    """
    )
    ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        ibd_segs = ibd_segments(self.ts, within=[1, 2])
        true_segs = {
            (1, 2): [tskit.IbdSegment(0.0, 0.2, 4), tskit.IbdSegment(0.2, 1.0, 4)],
        }
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdPolytomies:
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
        ibd_segs = ibd_segments(self.ts)
        true_segs = {
            (0, 1): [tskit.IbdSegment(0, 1, 4)],
            (0, 2): [tskit.IbdSegment(0, 0.3, 4), tskit.IbdSegment(0.3, 1, 5)],
            (0, 3): [tskit.IbdSegment(0, 0.3, 5), tskit.IbdSegment(0.3, 1, 4)],
            (1, 2): [tskit.IbdSegment(0, 0.3, 4), tskit.IbdSegment(0.3, 1, 5)],
            (1, 3): [tskit.IbdSegment(0, 0.3, 5), tskit.IbdSegment(0.3, 1, 4)],
            (2, 3): [tskit.IbdSegment(0.3, 1, 5), tskit.IbdSegment(0, 0.3, 5)],
        }
        assert_ibd_equal(ibd_segs, true_segs)

    def test_time(self):
        ibd_segs = ibd_segments(self.ts, max_time=3)
        true_segs = {
            (0, 1): [tskit.IbdSegment(0, 1, 4)],
            (0, 2): [tskit.IbdSegment(0, 0.3, 4)],
            (0, 3): [tskit.IbdSegment(0.3, 1, 4)],
            (1, 2): [tskit.IbdSegment(0, 0.3, 4)],
            (1, 3): [tskit.IbdSegment(0.3, 1, 4)],
        }
        assert_ibd_equal(ibd_segs, true_segs)

    def test_length(self):
        ibd_segs = ibd_segments(self.ts, min_length=0.5)
        true_segs = {
            (0, 1): [tskit.IbdSegment(0, 1, 4)],
            (0, 2): [tskit.IbdSegment(0.3, 1, 5)],
            (0, 3): [tskit.IbdSegment(0.3, 1, 4)],
            (1, 2): [tskit.IbdSegment(0.3, 1, 5)],
            (1, 3): [tskit.IbdSegment(0.3, 1, 4)],
            (2, 3): [tskit.IbdSegment(0.3, 1, 5)],
        }
        assert_ibd_equal(ibd_segs, true_segs)

    def test_input_within(self):
        ibd_segs = ibd_segments(self.ts, within=[0, 1, 2])
        true_segs = {
            (0, 1): [tskit.IbdSegment(0, 1, 4)],
            (0, 2): [tskit.IbdSegment(0, 0.3, 4), tskit.IbdSegment(0.3, 1, 5)],
            (1, 2): [tskit.IbdSegment(0, 0.3, 4), tskit.IbdSegment(0.3, 1, 5)],
        }
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdInternalSamples:
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
        ibd_segs = ibd_segments(self.ts)
        true_segs = {
            (0, 2): [tskit.IbdSegment(0, 1, 3)],
        }
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdLengthThreshold:
    """
    Tests the behaviour of the min_length argument in niche cases.
    """

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

    def test_length_exceeds_segment(self):
        ibd_segs = ibd_segments(self.ts, min_length=1.1)
        assert_ibd_equal(ibd_segs, {})

    def test_length_is_negative(self):
        with pytest.raises(tskit.LibraryError):
            ibd_segments(self.ts, min_length=-0.1)

    def test_equal_to_length(self):
        ibd_segs = ibd_segments(self.ts, min_length=0.4)
        true_segs = {(0, 1): [tskit.IbdSegment(0.4, 1.0, 3)]}
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdRandomExamples:
    """
    Randomly generated test cases.
    """

    @pytest.mark.parametrize("seed", range(1, 5))
    def test_random_examples(self, seed):
        ts = msprime.simulate(sample_size=10, recombination_rate=0.3, random_seed=seed)
        verify_equal_ibd(ts)

    # Finite sites
    # TODO update this to use msprime 1.0 APIs
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

    @pytest.mark.parametrize("seed", range(1, 5))
    def test_finite_sites(self, seed):
        ts = self.sim_finite_sites(seed)
        verify_equal_ibd(ts)

    @pytest.mark.parametrize("seed", range(1, 5))
    def test_dtwf(self, seed):
        ts = self.sim_finite_sites(seed, dtwf=True)
        verify_equal_ibd(ts)

    @pytest.mark.skip("FIXME: issue 1677")
    @pytest.mark.parametrize("seed", range(1, 5))
    def test_sim_wright_fisher_generations(self, seed):
        # Uses the bespoke DTWF forward-time simulator.
        number_of_gens = 10
        tables = wf.wf_sim(10, number_of_gens, deep_history=False, seed=seed)
        tables.sort()
        ts = tables.tree_sequence()
        verify_equal_ibd(ts)


class TestIbdResult:
    """
    Test the IbdResult class interface.
    """

    def verify_segments(self, ts, ibd_segments):
        samples = set(ts.samples())
        for (a, b), segment_list in ibd_segments.items():
            assert a < b
            assert a in samples
            assert b in samples
            left = segment_list.left
            right = segment_list.right
            node = segment_list.node

            num_segments = 0
            total_span = 0
            for j, seg in enumerate(segment_list):
                assert isinstance(seg, tskit.IbdSegment)
                total_span += seg.span
                num_segments += 1
                assert seg.span == seg.right - seg.left
                assert seg.left == left[j]
                assert seg.right == right[j]
                assert seg.node == node[j]
                assert 0 <= seg.node < ts.num_nodes
            assert total_span == segment_list.total_span
            assert num_segments == len(segment_list)

        total_span = sum(lst.total_span for lst in ibd_segments.values())
        np.testing.assert_allclose(ibd_segments.total_span, total_span)
        num_segments = sum(len(lst) for lst in ibd_segments.values())
        assert num_segments == ibd_segments.num_segments

    def test_str(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments()
        s = str(result)
        assert s.startswith("IBD Result")

        for lst in result.values():
            s = str(lst)
            assert s.startswith("IbdSegmentList(num_segments=")

    def test_repr(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments()
        s = repr(result)
        assert s.startswith("IbdResult({")
        for lst in result.values():
            s = repr(lst)
            assert s.startswith("IbdSegmentList([")

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_pairs_all_samples(self, n):
        ts = msprime.sim_ancestry(n, random_seed=2)
        result = ts.ibd_segments()
        pairs = np.array(list(itertools.combinations(ts.samples(), 2)))
        np.testing.assert_array_equal(pairs, result.pairs)
        self.verify_segments(ts, result)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_pairs_subset(self, n):
        ts = msprime.sim_ancestry(n, random_seed=2)
        pairs = np.array([(0, 1), (0, 2), (1, 2)])
        result = ts.ibd_segments(within=[0, 1, 2])
        np.testing.assert_array_equal(pairs, result.pairs)
        self.verify_segments(ts, result)

    @pytest.mark.parametrize("max_time", [0, 1, 10])
    def test_max_time(self, max_time):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(max_time=max_time)
        assert result.max_time == max_time
        self.verify_segments(ts, result)

    def test_max_time_default(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments()
        assert np.isinf(result.max_time)
        self.verify_segments(ts, result)

    @pytest.mark.parametrize("min_length", [0, 1, 10])
    def test_min_length(self, min_length):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(min_length=min_length)
        assert result.min_length == min_length
        self.verify_segments(ts, result)

    @pytest.mark.parametrize("min_length", [100, 101, 100000])
    def test_min_length_longer_than_seq_length(self, min_length):
        ts = msprime.sim_ancestry(
            100, recombination_rate=0.1, sequence_length=100, random_seed=2
        )
        result = ts.ibd_segments(min_length=min_length)
        assert result.min_length == min_length
        assert result.num_segments == 0
        self.verify_segments(ts, result)

    def test_recombination_discrete(self):
        ts = msprime.sim_ancestry(
            10, sequence_length=100, recombination_rate=0.1, random_seed=2
        )
        assert ts.num_trees > 2
        result = ts.ibd_segments()
        self.verify_segments(ts, result)

    def test_recombination_continuous(self):
        ts = msprime.sim_ancestry(
            10,
            recombination_rate=1,
            random_seed=2,
            discrete_genome=False,
            sequence_length=1,
        )
        assert ts.num_trees > 2
        result = ts.ibd_segments()
        self.verify_segments(ts, result)

    def test_dict_interface(self):
        ts = msprime.sim_ancestry(5, random_seed=2)
        pairs = list(itertools.combinations(ts.samples(), 2))
        result = ts.ibd_segments()
        assert len(result) == len(pairs)
        for pair in pairs:
            assert pair in result
            assert result[pair] is not None
        for k, v in result.items():
            assert k in pairs
            assert isinstance(v, tskit.IbdSegmentList)
