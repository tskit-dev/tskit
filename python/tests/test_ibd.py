import collections
import io
import itertools

import msprime
import numpy as np
import pytest

import tests
import tests.ibd as ibd
import tests.test_wright_fisher as wf
import tskit
from tests.test_highlevel import get_example_tree_sequences

"""
Tests of IBD finding algorithms.
"""


# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this. The example_ts here is intended to be the
# basic tree sequence which should give a meaningful result for
# most operations. Probably rename it to ``examples.simple_ts()``
# or something.


@tests.cached_example
def example_ts():
    return [msprime.sim_ancestry(2, random_seed=1)]


def ibd_segments(
    ts,
    *,
    within=None,
    between=None,
    min_span=0,
    max_time=None,
    compare_lib=True,
    print_c=False,
    print_py=False,
):
    """
    Calculates IBD segments using Python and converts output to lists of segments.
    Also compares result with C library.
    """
    ibd_f = ibd.IbdFinder(
        ts, within=within, between=between, max_time=max_time, min_span=min_span
    )
    ibd_segs = ibd_f.run()
    if print_py:
        print("Python output:\n")
        print(ibd_segs)
    # ibd_f.print_state()
    if compare_lib:
        c_out = ts.ibd_segments(
            within=within,
            between=between,
            max_time=max_time,
            min_span=min_span,
            store_segments=True,
        )
        if print_c:
            print("C output:\n")
            print(c_out)
        assert_ibd_equal(ibd_segs, c_out)
    return ibd_segs


def naive_ibd(ts, a, b):
    """
    Returns the IBD segments along the genome for a and b.
    """

    def path(tree, u, v):
        ret = [u]
        while u != v:
            u = tree.parent(u)
            ret.append(u)
        return ret

    tree = ts.first()
    mrca = tree.mrca(a, b)
    last_paths = [path(tree, a, mrca), path(tree, b, mrca)]
    last_mrca = mrca
    left = 0.0
    segs = []
    while tree.next():
        mrca = tree.mrca(a, b)
        paths = [path(tree, a, mrca), path(tree, b, mrca)]
        if paths != last_paths:
            segs.append(tskit.IdentitySegment(left, tree.interval.left, last_mrca))
            last_paths = paths
            left = tree.interval.left
            last_mrca = mrca

    segs.append(tskit.IdentitySegment(left, ts.sequence_length, last_mrca))
    # Filter out segments with no mrca
    return [seg for seg in segs if seg.node != -1]


def naive_ibd_all_pairs(ts, samples=None):
    samples = ts.samples() if samples is None else samples
    all_pairs_map = {
        (a, b): naive_ibd(ts, a, b) for a, b in itertools.combinations(samples, 2)
    }
    # Filter out pairs with empty segment lists
    return {key: value for key, value in all_pairs_map.items() if len(value) > 0}


class TestIbdDefinition:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_all_pairs(self, ts):
        samples = ts.samples()[:10]
        ibd_lib = ts.ibd_segments(within=samples, store_segments=True)
        ibd_def = naive_ibd_all_pairs(ts, samples=samples)
        assert_ibd_equal(ibd_lib, ibd_def)

    @pytest.mark.parametrize("N", [2, 5, 10])
    @pytest.mark.parametrize("T", [2, 5, 10])
    def test_wright_fisher_examples(self, N, T):
        tables = wf.wf_sim(N, T, deep_history=False, seed=42)
        tables.sort()
        # NB this is essential! We get spurious breakpoints otherwise
        tables.edges.squash()
        tables.sort()
        ts = tables.tree_sequence()
        ibd0 = ibd_segments(ts)
        ibd1 = naive_ibd_all_pairs(ts)
        assert_ibd_equal(ibd0, ibd1)


class TestIbdImplementations:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_all_pairs(self, ts):
        # Automatically compares the two implementations
        ibd_segments(ts)


def assert_ibd_equal(dict1, dict2):
    """
    Verifies that two dictionaries have the same keys, and that
    the set of items corresponding to each key is identical.
    Used to check identical IBD output.
    """
    assert len(dict1) == len(dict2)
    for key, val in dict1.items():
        assert key in dict2
        assert len(val) == len(dict2[key])
        segs1 = list(sorted(val))
        segs2 = list(sorted(dict2[key]))
        assert segs1 == segs2


class TestIbdSingleBinaryTree:
    @tests.cached_example
    def ts(self):
        #
        # 2        4
        #         / \
        # 1      3   \
        #       / \   \
        # 0    0   1   2
        print("evaluating ts")
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    # Basic test
    def test_defaults(self):
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0.0, 1.0, 3)],
            (0, 2): [tskit.IdentitySegment(0.0, 1.0, 4)],
            (1, 2): [tskit.IdentitySegment(0.0, 1.0, 4)],
        }
        ibd_segs = ibd_segments(self.ts(), within=[0, 1, 2])
        assert_ibd_equal(ibd_segs, true_segs)

    def test_within(self):
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0.0, 1.0, 3)],
        }
        ibd_segs = ibd_segments(self.ts(), within=[0, 1])
        assert_ibd_equal(ibd_segs, true_segs)

    def test_between_0_1(self):
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0.0, 1.0, 3)],
        }
        ibd_segs = ibd_segments(self.ts(), between=[[0], [1]])
        assert_ibd_equal(ibd_segs, true_segs)

    def test_between_0_2(self):
        true_segs = {
            (0, 2): [tskit.IdentitySegment(0.0, 1.0, 4)],
        }
        ibd_segs = ibd_segments(self.ts(), between=[[0], [2]])
        assert_ibd_equal(ibd_segs, true_segs)

    def test_between_0_1_2(self):
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0.0, 1.0, 3)],
            (0, 2): [tskit.IdentitySegment(0.0, 1.0, 4)],
            (1, 2): [tskit.IdentitySegment(0.0, 1.0, 4)],
        }
        ibd_segs = ibd_segments(self.ts(), between=[[0], [1], [2]])
        assert_ibd_equal(ibd_segs, true_segs)

    def test_between_0_12(self):
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0.0, 1.0, 3)],
            (0, 2): [tskit.IdentitySegment(0.0, 1.0, 4)],
        }
        ibd_segs = ibd_segments(self.ts(), between=[[0], [1, 2]])
        assert_ibd_equal(ibd_segs, true_segs)

    def test_time(self):
        ibd_segs = ibd_segments(
            self.ts(),
            max_time=1.5,
            compare_lib=True,
        )
        true_segs = {(0, 1): [tskit.IdentitySegment(0.0, 1.0, 3)]}
        assert_ibd_equal(ibd_segs, true_segs)

    def test_length(self):
        ibd_segs = ibd_segments(self.ts(), min_span=2)
        assert_ibd_equal(ibd_segs, {})


class TestIbdInterface:
    @pytest.mark.parametrize("ts", example_ts())
    def test_input_errors_within(self, ts):
        with pytest.raises(tskit.LibraryError, match="Node out of bounds"):
            ts.ibd_segments(within=[-1])
        with pytest.raises(tskit.LibraryError, match="Duplicate sample value"):
            ts.ibd_segments(within=[0, 0])

    @pytest.mark.parametrize("ts", example_ts())
    def test_input_errors_between(self, ts):
        with pytest.raises(tskit.LibraryError, match="Node out of bounds"):
            ts.ibd_segments(between=[[0], [-1]])
        with pytest.raises(tskit.LibraryError, match="Duplicate sample"):
            ts.ibd_segments(between=[[0], [0]])

    @pytest.mark.parametrize("ts", example_ts())
    def test_within_between_mutually_exclusive(self, ts):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ts.ibd_segments(within=[0], between=[1])

    @pytest.mark.parametrize("ts", example_ts())
    def test_tables_interface(self, ts):
        ibd_tab = ts.tables.ibd_segments(store_segments=True)
        ibd_ts = ts.ibd_segments(store_segments=True)
        assert ibd_tab == ibd_ts

    @pytest.mark.parametrize("ts", example_ts())
    def test_empty_within(self, ts):
        ibd = ts.ibd_segments(within=[], store_pairs=True)
        assert len(ibd) == 0

    @pytest.mark.parametrize("ts", example_ts())
    def test_empty_between(self, ts):
        ibd = ts.ibd_segments(between=[], store_pairs=True)
        assert len(ibd) == 0

    @pytest.mark.parametrize("ts", example_ts())
    def test_empty_in_between(self, ts):
        ibd = ts.ibd_segments(between=[[1, 2], []], store_pairs=True)
        assert len(ibd) == 0


class TestIbdTwoSamplesTwoTrees:
    # 2
    #             |     3
    # 1      2    |    / \
    #       / \   |   /   \
    # 0    0   1  |  0     1
    # |------------|----------|
    # 0.0          0.4        1.0

    @tests.cached_example
    def ts(self):
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    # Basic test
    def test_basic(self):
        ibd_segs = ibd_segments(self.ts())
        true_segs = {
            (0, 1): [
                tskit.IdentitySegment(0.0, 0.4, 2),
                tskit.IdentitySegment(0.4, 1.0, 3),
            ]
        }
        assert_ibd_equal(ibd_segs, true_segs)

    # Max time = 1.2
    def test_time(self):
        ibd_segs = ibd_segments(self.ts(), max_time=1.2, compare_lib=True)
        true_segs = {(0, 1): [tskit.IdentitySegment(0.0, 0.4, 2)]}
        assert_ibd_equal(ibd_segs, true_segs)

    # Min length = 0.5
    def test_length(self):
        ibd_segs = ibd_segments(self.ts(), min_span=0.5, compare_lib=True)
        true_segs = {(0, 1): [tskit.IdentitySegment(0.4, 1.0, 3)]}
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdUnrelatedSamples:
    #
    #    2   3
    #    |   |
    #    0   1

    @tests.cached_example
    def ts(self):
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

        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_basic(self):
        ibd_segs = ibd_segments(self.ts())
        assert len(ibd_segs) == 0

    def test_time(self):
        ibd_segs = ibd_segments(self.ts(), max_time=1.2)
        assert len(ibd_segs) == 0

    def test_length(self):
        ibd_segs = ibd_segments(self.ts(), min_span=0.2)
        assert len(ibd_segs) == 0


class TestIbdNoSamples:
    #
    #     2
    #    / \
    #   /   \
    #  /     \
    # (0)   (1)

    @tests.cached_example
    def ts(self):
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        result = ibd_segments(self.ts())
        assert len(result) == 0

    def test_specified_samples(self):
        ibd_segs = ibd_segments(self.ts(), within=[0, 1])
        true_segs = {
            (0, 1): [
                tskit.IdentitySegment(0.0, 1, 2),
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
    @tests.cached_example
    def ts(self):
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_basic(self):
        ibd_segs = ibd_segments(self.ts())
        true_segs = {
            (0, 2): [tskit.IdentitySegment(0.0, 1.0, 2)],
            (1, 3): [tskit.IdentitySegment(0.0, 1.0, 3)],
        }

        assert_ibd_equal(ibd_segs, true_segs)

    def test_input_within(self):
        ibd_segs = ibd_segments(self.ts(), within=[0, 2, 3, 5])
        true_segs = {
            (0, 2): [tskit.IdentitySegment(0.0, 1.0, 2)],
            (3, 5): [tskit.IdentitySegment(0.0, 1.0, 5)],
        }
        assert_ibd_equal(ibd_segs, true_segs)

    def test_all_samples(self):
        # FIXME
        ibd_segs = ibd_segments(self.ts(), within=range(6), compare_lib=False)
        true_segs = {
            (0, 2): [tskit.IdentitySegment(0.0, 1.0, 2)],
            (0, 4): [tskit.IdentitySegment(0.0, 1.0, 4)],
            (2, 4): [tskit.IdentitySegment(0.0, 1.0, 4)],
            (1, 3): [tskit.IdentitySegment(0.0, 1.0, 3)],
            (1, 5): [tskit.IdentitySegment(0.0, 1.0, 5)],
            (3, 5): [tskit.IdentitySegment(0.0, 1.0, 5)],
        }
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdSimpleInternalSampleChain:
    #
    # 2
    # |
    # 1
    # |
    # 0
    #
    @tests.cached_example
    def ts(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           1
        2       1           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       1       0
        0       1       2       1
        """
        )
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_basic(self):
        # FIXME
        ibd_segs = ibd_segments(self.ts(), compare_lib=False)
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0.0, 1.0, 1)],
            (0, 2): [tskit.IdentitySegment(0.0, 1.0, 2)],
            (1, 2): [tskit.IdentitySegment(0.0, 1.0, 2)],
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

    @tests.cached_example
    def ts(self):
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        ibd_segs = ibd_segments(self.ts())
        true_segs = {
            (0, 1): [
                tskit.IdentitySegment(0.0, 0.2, 4),
                tskit.IdentitySegment(0.7, 1.0, 4),
                tskit.IdentitySegment(0.2, 0.7, 4),
            ]
        }
        assert_ibd_equal(ibd_segs, true_segs)

    def test_time(self):
        ibd_segs = ibd_segments(self.ts(), max_time=1.8)
        assert len(ibd_segs) == 0

    def test_length(self):
        ibd_segs = ibd_segments(self.ts(), min_span=0.4)
        true_segs = {(0, 1): [tskit.IdentitySegment(0.2, 0.7, 4)]}
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

    @tests.cached_example
    def ts(self):
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        ibd_segs = ibd_segments(self.ts(), within=[1, 2])
        true_segs = {
            (1, 2): [
                tskit.IdentitySegment(0.0, 0.2, 4),
                tskit.IdentitySegment(0.2, 1.0, 4),
            ],
        }
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdDifferentPaths3:
    # 2.00┊   4   ┊   4   ┊
    #     ┊ ┏━╋━┓ ┊ ┏━╋━┓ ┊
    # 1.00┊ 2 ┃ 3 ┊ 3 ┃ 2 ┊
    #     ┊ ┃ ┃   ┊ ┃ ┃   ┊
    # 0.00┊ 0 1   ┊ 0 1   ┊
    #     0       5      10
    @tests.cached_example
    def ts(self):
        t = tskit.TableCollection(sequence_length=10)
        t.nodes.add_row(flags=1, time=0)
        t.nodes.add_row(flags=1, time=0)
        t.nodes.add_row(flags=0, time=1)
        t.nodes.add_row(flags=0, time=1)
        t.nodes.add_row(flags=0, time=2)
        t.edges.add_row(parent=2, child=0, left=0, right=5)
        t.edges.add_row(parent=3, child=0, left=5, right=10)
        t.edges.add_row(parent=4, child=2, left=0, right=10)
        t.edges.add_row(parent=4, child=3, left=0, right=10)
        t.edges.add_row(parent=4, child=1, left=0, right=10)
        t.sort()
        return t.tree_sequence()

    def test_defaults(self):
        ibd_segs = ibd_segments(self.ts())
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0, 5, 4), tskit.IdentitySegment(5, 10, 4)],
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

    @tests.cached_example
    def ts(self):
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        ibd_segs = ibd_segments(self.ts())
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0, 1, 4)],
            (0, 2): [
                tskit.IdentitySegment(0, 0.3, 4),
                tskit.IdentitySegment(0.3, 1, 5),
            ],
            (0, 3): [
                tskit.IdentitySegment(0, 0.3, 5),
                tskit.IdentitySegment(0.3, 1, 4),
            ],
            (1, 2): [
                tskit.IdentitySegment(0, 0.3, 4),
                tskit.IdentitySegment(0.3, 1, 5),
            ],
            (1, 3): [
                tskit.IdentitySegment(0, 0.3, 5),
                tskit.IdentitySegment(0.3, 1, 4),
            ],
            (2, 3): [
                tskit.IdentitySegment(0.3, 1, 5),
                tskit.IdentitySegment(0, 0.3, 5),
            ],
        }
        assert_ibd_equal(ibd_segs, true_segs)

    def test_time(self):
        ibd_segs = ibd_segments(self.ts(), max_time=3)
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0, 1, 4)],
            (0, 2): [tskit.IdentitySegment(0, 0.3, 4)],
            (0, 3): [tskit.IdentitySegment(0.3, 1, 4)],
            (1, 2): [tskit.IdentitySegment(0, 0.3, 4)],
            (1, 3): [tskit.IdentitySegment(0.3, 1, 4)],
        }
        assert_ibd_equal(ibd_segs, true_segs)

    def test_length(self):
        ibd_segs = ibd_segments(self.ts(), min_span=0.5)
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0, 1, 4)],
            (0, 2): [tskit.IdentitySegment(0.3, 1, 5)],
            (0, 3): [tskit.IdentitySegment(0.3, 1, 4)],
            (1, 2): [tskit.IdentitySegment(0.3, 1, 5)],
            (1, 3): [tskit.IdentitySegment(0.3, 1, 4)],
            (2, 3): [tskit.IdentitySegment(0.3, 1, 5)],
        }
        (ibd_segs, true_segs)

    def test_input_within(self):
        ibd_segs = ibd_segments(self.ts(), within=[0, 1, 2])
        true_segs = {
            (0, 1): [tskit.IdentitySegment(0, 1, 4)],
            (0, 2): [
                tskit.IdentitySegment(0, 0.3, 4),
                tskit.IdentitySegment(0.3, 1, 5),
            ],
            (1, 2): [
                tskit.IdentitySegment(0, 0.3, 4),
                tskit.IdentitySegment(0.3, 1, 5),
            ],
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

    @tests.cached_example
    def ts(self):
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_defaults(self):
        ibd_segs = ibd_segments(self.ts())
        true_segs = {
            (0, 2): [tskit.IdentitySegment(0, 1, 3)],
        }
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdLengthThreshold:
    """
    Tests the behaviour of the min_span argument in niche cases.
    """

    # 2
    #             |     3
    # 1      2    |    / \
    #       / \   |   /   \
    # 0    0   1  |  0     1
    # |------------|----------|
    # 0.0          0.4        1.0

    @tests.cached_example
    def ts(self):
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
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_length_exceeds_segment(self):
        ibd_segs = ibd_segments(self.ts(), min_span=1.1)
        assert_ibd_equal(ibd_segs, {})

    def test_length_is_negative(self):
        with pytest.raises(tskit.LibraryError):
            ibd_segments(self.ts(), min_span=-0.1)

    def test_equal_to_length(self):
        ibd_segs = ibd_segments(self.ts(), min_span=0.4)
        true_segs = {(0, 1): [tskit.IdentitySegment(0.4, 1.0, 3)]}
        assert_ibd_equal(ibd_segs, true_segs)


class TestIbdProperties:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_default_within_all_samples(self, ts):
        segs = ts.ibd_segments(store_pairs=True)
        for a, b in segs.keys():
            assert ts.node(a).is_sample()
            assert ts.node(b).is_sample()

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_within_subset(self, ts):
        samples = ts.samples()
        samples = samples[:3]
        segs = ts.ibd_segments(store_pairs=True, within=samples)
        for a, b in segs.keys():
            assert a in samples
            assert b in samples

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_between_two_subsets(self, ts):
        samples = ts.samples()
        k = len(samples) // 2
        A = samples[:k]
        B = samples[k:]
        segs = ts.ibd_segments(store_pairs=True, between=[A, B])
        for a, b in segs.keys():
            assert a in A
            assert b in B

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_between_same_segments_as_filtered_within_pair(self, ts):
        samples = ts.samples()[:10]
        all_segs = ts.ibd_segments(within=samples, store_segments=True)
        A = samples[1::2]
        B = samples[::2]
        between_segs = ts.ibd_segments(store_segments=True, between=[A, B])
        filtered_segs = collections.defaultdict(list)
        for (u, v), seglist in all_segs.items():
            if (u in A and v in B) or (v in A and u in B):
                filtered_segs[(u, v)] = seglist
        assert_ibd_equal(between_segs, filtered_segs)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_between_same_segments_as_filtered_within_triple(self, ts):
        samples = ts.samples()[:9]  # Limit the number of samples
        all_segs = ts.ibd_segments(within=samples, store_segments=True)
        A = samples[1::3]
        B = samples[2::3]
        C = samples[0::3]
        all_pairs = set()
        for set_pair in itertools.combinations([A, B, C], 2):
            for pair in itertools.product(*set_pair):
                all_pairs.add(tuple(sorted(pair)))
        between_segs = ts.ibd_segments(store_segments=True, between=[A, B, C])
        filtered_segs = collections.defaultdict(list)
        for pair, seglist in all_segs.items():
            if pair in all_pairs:
                filtered_segs[pair] = seglist
        assert_ibd_equal(between_segs, filtered_segs)


class TestIdentitySegments:
    """
    Test the IdentitySegments class interface.
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
                assert isinstance(seg, tskit.IdentitySegment)
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

    @pytest.mark.parametrize("store_segments", [True, False])
    @pytest.mark.parametrize("store_pairs", [True, False])
    def test_str(self, store_segments, store_pairs):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(store_segments=store_segments, store_pairs=store_pairs)
        s = str(result)
        assert "IdentitySegments" in s
        assert "max_time" in s
        assert "min_span" in s

    def test_repr_store_segments(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(store_segments=True)
        s = repr(result)
        assert s.startswith("IdentitySegments({")
        for lst in result.values():
            s = repr(lst)
            assert s.startswith("IdentitySegmentList([")

    def test_repr_without_store_segments(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(store_pairs=True)
        s = repr(result)
        assert s.startswith("<tskit.tables.IdentitySegments")
        result = ts.ibd_segments()
        s = repr(result)
        assert s.startswith("<tskit.tables.IdentitySegments")

    def test_store_segs_implies_store_pairs(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(store_pairs=True)
        assert result.num_pairs == 6
        result = ts.ibd_segments(store_segments=True)
        assert result.num_pairs == 6

    def test_operations_available_by_default(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments()
        assert result.num_segments == 6
        assert result.total_span == 6
        with pytest.raises(tskit.IdentityPairsNotStoredError):
            _ = result.num_pairs
        with pytest.raises(tskit.IdentityPairsNotStoredError):
            _ = len(result)
        with pytest.raises(tskit.IdentityPairsNotStoredError):
            _ = result.pairs
        with pytest.raises(tskit.IdentityPairsNotStoredError):
            _ = result[0, 1]
        with pytest.raises(tskit.IdentityPairsNotStoredError):
            _ = list(result)
        with pytest.raises(tskit.IdentityPairsNotStoredError):
            _ = result == result
        # It's OK to when we compare with another type
        assert result != []

    def test_operations_available_store_pairs(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(store_pairs=True)
        assert result.num_segments == 6
        assert result.total_span == 6
        assert result.num_pairs == 6
        assert len(result) == 6
        assert result.pairs is not None
        seglist = result[0, 1]
        assert seglist.total_span == 1
        assert len(seglist) == 1
        with pytest.raises(tskit.IdentitySegmentsNotStoredError):
            _ = list(seglist)
        with pytest.raises(tskit.IdentitySegmentsNotStoredError):
            _ = seglist.left
        with pytest.raises(tskit.IdentitySegmentsNotStoredError):
            _ = seglist.right
        with pytest.raises(tskit.IdentitySegmentsNotStoredError):
            _ = seglist.node
        with pytest.raises(tskit.IdentitySegmentsNotStoredError):
            _ = seglist == seglist

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_pairs_all_samples(self, n):
        ts = msprime.sim_ancestry(n, random_seed=2)
        result = ts.ibd_segments(store_segments=True)
        pairs = np.array(list(itertools.combinations(ts.samples(), 2)))
        np.testing.assert_array_equal(pairs, result.pairs)
        self.verify_segments(ts, result)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_pairs_subset(self, n):
        ts = msprime.sim_ancestry(n, random_seed=2)
        pairs = np.array([(0, 1), (0, 2), (1, 2)])
        result = ts.ibd_segments(within=[0, 1, 2], store_segments=True)
        np.testing.assert_array_equal(pairs, result.pairs)
        self.verify_segments(ts, result)

    @pytest.mark.parametrize("max_time", [0, 1, 10])
    def test_max_time(self, max_time):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(max_time=max_time, store_segments=True)
        assert result.max_time == max_time
        self.verify_segments(ts, result)

    def test_max_time_default(self):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(store_segments=True)
        assert np.isinf(result.max_time)
        self.verify_segments(ts, result)

    @pytest.mark.parametrize("min_span", [0, 1, 10])
    def test_min_span(self, min_span):
        ts = msprime.sim_ancestry(2, random_seed=2)
        result = ts.ibd_segments(min_span=min_span, store_segments=True)
        assert result.min_span == min_span
        self.verify_segments(ts, result)

    @pytest.mark.parametrize("min_span", [100, 101, 100000])
    def test_min_span_longer_than_seq_length(self, min_span):
        ts = msprime.sim_ancestry(
            100, recombination_rate=0.1, sequence_length=100, random_seed=2
        )
        result = ts.ibd_segments(min_span=min_span, store_segments=True)
        assert result.min_span == min_span
        assert result.num_segments == 0
        self.verify_segments(ts, result)

    def test_recombination_discrete(self):
        ts = msprime.sim_ancestry(
            10, sequence_length=100, recombination_rate=0.1, random_seed=2
        )
        assert ts.num_trees > 2
        result = ts.ibd_segments(store_segments=True)
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
        result = ts.ibd_segments(store_segments=True)
        self.verify_segments(ts, result)

    def test_dict_interface(self):
        ts = msprime.sim_ancestry(5, random_seed=2)
        pairs = list(itertools.combinations(ts.samples(), 2))
        result = ts.ibd_segments(store_segments=True)
        assert len(result) == len(pairs)
        for pair in pairs:
            assert pair in result
            assert result[pair] is not None
        for k, v in result.items():
            assert k in pairs
            assert isinstance(v, tskit.IdentitySegmentList)


class TestIdentitySegmentsList:
    """
    Tests for the IdentitySegmentList class.
    """

    example_ts = msprime.sim_ancestry(
        3, sequence_length=100, recombination_rate=0.1, random_seed=2
    )

    def test_list_semantics(self):
        result = self.example_ts.ibd_segments(store_segments=True)
        assert len(result) > 0
        for seglist in result.values():
            lst = list(seglist)
            assert len(lst) == len(seglist)
            assert lst == list(seglist)

    def test_str(self):
        result = self.example_ts.ibd_segments(store_segments=True)
        seglist = list(result.values())[0]
        assert str(seglist).startswith("IdentitySegmentList")

    def test_repr(self):
        result = self.example_ts.ibd_segments(store_segments=True)
        seglist = list(result.values())[0]
        assert repr(seglist).startswith("IdentitySegmentList([IdentitySegment")

    def test_eq_semantics(self):
        result = self.example_ts.ibd_segments(store_segments=True)
        seglists = list(result.values())
        assert len(result) == len(seglists)
        assert len(seglists) > 1
        for seglist1, seglist2 in zip(result.values(), seglists):
            assert seglist1 == seglist2
            assert not (seglist1 != seglist2)
            assert seglist1 != result
            assert seglist1 != []
        # The chance of getting two identical seglists is miniscule
        for seglist in seglists[1:]:
            assert seglist != seglists[0]

    def test_eq_fails_without_store_segments(self):
        result = self.example_ts.ibd_segments(store_pairs=True)
        for seglist in result.values():
            with pytest.raises(tskit.IdentitySegmentsNotStoredError):
                _ = seglist == seglist
            # But it's OK when comparing to another type, since we know
            # it'll be False regardless
            assert seglist != []

    def test_list_contents(self):
        result = self.example_ts.ibd_segments(store_segments=True)
        assert len(result) > 0
        for seglist in result.values():
            for seg in seglist:
                assert isinstance(seg, tskit.IdentitySegment)
