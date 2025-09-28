# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
# Copyright (c) 2016-2017 University of Oxford
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
Test cases for the supported topological variations and operations.
"""
import functools
import io
import itertools
import json
import random
import sys
import unittest

import msprime
import numpy as np
import pytest

import _tskit
import tests as tests
import tests.test_wright_fisher as wf
import tests.tsutil as tsutil
import tskit
import tskit.provenance as provenance


def simple_keep_intervals(tables, intervals, simplify=True, record_provenance=True):
    """
    Simple Python implementation of keep_intervals.
    """
    ts = tables.tree_sequence()
    last_stop = 0
    for start, stop in intervals:
        if start < 0 or stop > ts.sequence_length:
            raise ValueError("Slice bounds must be within the existing tree sequence")
        if start >= stop:
            raise ValueError("Interval error: start must be < stop")
        if start < last_stop:
            raise ValueError("Intervals must be disjoint")
        last_stop = stop
    tables.edges.clear()
    tables.sites.clear()
    tables.mutations.clear()
    for edge in ts.edges():
        for interval_left, interval_right in intervals:
            if not (edge.right <= interval_left or edge.left >= interval_right):
                left = max(interval_left, edge.left)
                right = min(interval_right, edge.right)
                tables.edges.append(edge.replace(left=left, right=right))
    for site in ts.sites():
        for interval_left, interval_right in intervals:
            if interval_left <= site.position < interval_right:
                site_id = tables.sites.append(site)
                for m in site.mutations:
                    tables.mutations.append(m.replace(site=site_id, parent=tskit.NULL))
    tables.build_index()
    tables.compute_mutation_parents()
    tables.sort()
    if simplify:
        tables.simplify(record_provenance=False)
    if record_provenance:
        parameters = {"command": "keep_intervals", "TODO": "add parameters"}
        tables.provenances.add_row(
            record=json.dumps(provenance.get_provenance_dict(parameters))
        )


def generate_segments(n, sequence_length=100, seed=None):
    rng = random.Random(seed)
    segs = []
    for j in range(n):
        left = rng.randint(0, sequence_length - 1)
        right = rng.randint(left + 1, sequence_length)
        assert left < right
        segs.append(tests.Segment(left, right, j))
    return segs


class ExampleTopologyMixin:
    """
    Some example topologies for tests cases.
    """

    def test_single_coalescent_tree(self):
        ts = msprime.simulate(10, random_seed=1, length=10)
        self.verify(ts)

    def test_coalescent_trees(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=1, length=2)
        assert ts.num_trees > 2
        self.verify(ts)

    def test_coalescent_trees_internal_samples(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=10, length=2)
        assert ts.num_trees > 2
        self.verify(tsutil.jiggle_samples(ts))

    def test_coalescent_trees_all_samples(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=10, length=2)
        assert ts.num_trees > 2
        tables = ts.dump_tables()
        flags = np.zeros_like(tables.nodes.flags) + tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        self.verify(tables.tree_sequence())

    def test_wright_fisher_trees_unsimplified(self):
        tables = wf.wf_sim(10, 5, deep_history=False, seed=2)
        tables.sort()
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_wright_fisher_trees_simplified(self):
        tables = wf.wf_sim(10, 5, deep_history=False, seed=1)
        tables.sort()
        ts = tables.tree_sequence()
        ts = ts.simplify()
        self.verify(ts)

    def test_wright_fisher_trees_simplified_one_gen(self):
        tables = wf.wf_sim(10, 1, deep_history=False, seed=1)
        tables.sort()
        ts = tables.tree_sequence()
        ts = ts.simplify()
        self.verify(ts)

    def test_nonbinary_trees(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=1.0, population=0, proportion=0.95)
        ]
        ts = msprime.simulate(
            20,
            recombination_rate=10,
            mutation_rate=5,
            demographic_events=demographic_events,
            random_seed=7,
        )
        found = False
        for e in ts.edgesets():
            if len(e.children) > 2:
                found = True
        assert found
        self.verify(ts)

    def test_many_multiroot_trees(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        self.verify(ts)

    def test_multiroot_tree(self):
        ts = msprime.simulate(15, random_seed=10)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        self.verify(ts)

    def test_all_missing_data(self):
        tables = tskit.TableCollection(1)
        for _ in range(10):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        self.verify(tables.tree_sequence())


class TestOverlappingSegments:
    """
    Tests for the overlapping segments algorithm required for simplify.
    This test probably belongs somewhere else.
    """

    def test_random(self):
        segs = generate_segments(10, 20, 1)
        for left, right, X in tests.overlapping_segments(segs):
            assert right > left
            assert len(X) > 0

    def test_empty(self):
        ret = list(tests.overlapping_segments([]))
        assert len(ret) == 0

    def test_single_interval(self):
        for j in range(1, 10):
            segs = [tests.Segment(0, 1, j) for _ in range(j)]
            ret = list(tests.overlapping_segments(segs))
            assert len(ret) == 1
            left, right, X = ret[0]
            assert left == 0
            assert right == 1
            assert sorted(segs) == sorted(X)

    def test_stairs_down(self):
        segs = [tests.Segment(0, 1, 0), tests.Segment(0, 2, 1), tests.Segment(0, 3, 2)]
        ret = list(tests.overlapping_segments(segs))
        assert len(ret) == 3

        left, right, X = ret[0]
        assert left == 0
        assert right == 1
        assert sorted(X) == sorted(segs)

        left, right, X = ret[1]
        assert left == 1
        assert right == 2
        assert sorted(X) == sorted(segs[1:])

        left, right, X = ret[2]
        assert left == 2
        assert right == 3
        assert sorted(X) == sorted(segs[2:])

    def test_stairs_up(self):
        segs = [tests.Segment(0, 3, 0), tests.Segment(1, 3, 1), tests.Segment(2, 3, 2)]
        ret = list(tests.overlapping_segments(segs))
        assert len(ret) == 3

        left, right, X = ret[0]
        assert left == 0
        assert right == 1
        assert X == segs[:1]

        left, right, X = ret[1]
        assert left == 1
        assert right == 2
        assert sorted(X) == sorted(segs[:2])

        left, right, X = ret[2]
        assert left == 2
        assert right == 3
        assert sorted(X) == sorted(segs)

    def test_pyramid(self):
        segs = [tests.Segment(0, 5, 0), tests.Segment(1, 4, 1), tests.Segment(2, 3, 2)]
        ret = list(tests.overlapping_segments(segs))
        assert len(ret) == 5

        left, right, X = ret[0]
        assert left == 0
        assert right == 1
        assert X == segs[:1]

        left, right, X = ret[1]
        assert left == 1
        assert right == 2
        assert sorted(X) == sorted(segs[:2])

        left, right, X = ret[2]
        assert left == 2
        assert right == 3
        assert sorted(X) == sorted(segs)

        left, right, X = ret[3]
        assert left == 3
        assert right == 4
        assert sorted(X) == sorted(segs[:2])

        left, right, X = ret[4]
        assert left == 4
        assert right == 5
        assert sorted(X) == sorted(segs[:1])

    def test_gap(self):
        segs = [tests.Segment(0, 2, 0), tests.Segment(3, 4, 1)]
        ret = list(tests.overlapping_segments(segs))
        assert len(ret) == 2

        left, right, X = ret[0]
        assert left == 0
        assert right == 2
        assert X == segs[:1]

        left, right, X = ret[1]
        assert left == 3
        assert right == 4
        assert X == segs[1:]


class TopologyTestCase:
    """
    Superclass of test cases containing common utilities.
    """

    random_seed = 123456

    def assert_haplotypes_equal(self, ts1, ts2):
        h1 = list(ts1.haplotypes())
        h2 = list(ts2.haplotypes())
        assert h1 == h2

    def assert_variants_equal(self, ts1, ts2):
        for v1, v2 in zip(
            ts1.variants(copy=False),
            ts2.variants(copy=False),
        ):
            assert v1.alleles == v2.alleles
            assert np.array_equal(v1.genotypes, v2.genotypes)

    def check_num_samples(self, ts, x):
        """
        Compare against x, a list of tuples of the form
        `(tree number, parent, number of samples)`.
        """
        k = 0
        tss = ts.trees()
        t = next(tss)
        for j, node, nl in x:
            while k < j:
                t = next(tss)
                k += 1
            assert nl == t.num_samples(node)

    def check_num_tracked_samples(self, ts, tracked_samples, x):
        k = 0
        tss = ts.trees(tracked_samples=tracked_samples)
        t = next(tss)
        for j, node, nl in x:
            while k < j:
                t = next(tss)
                k += 1
            assert nl == t.num_tracked_samples(node)

    def check_sample_iterator(self, ts, x):
        """
        Compare against x, a list of tuples of the form
        `(tree number, node, sample ID list)`.
        """
        k = 0
        tss = ts.trees(sample_lists=True)
        t = next(tss)
        for j, node, samples in x:
            while k < j:
                t = next(tss)
                k += 1
            for u, v in zip(samples, t.samples(node)):
                assert u == v


class TestZeroRoots:
    """
    Tests that for the case in which we have zero samples and therefore
    zero roots in our trees.
    """

    def remove_samples(self, ts):
        tables = ts.dump_tables()
        tables.nodes.flags = np.zeros_like(tables.nodes.flags)
        return tables.tree_sequence()

    def verify(self, ts, no_root_ts):
        assert ts.num_trees == no_root_ts.num_trees
        for tree, no_root in zip(ts.trees(), no_root_ts.trees()):
            assert no_root.num_roots == 0
            assert no_root.left_root == tskit.NULL
            assert no_root.roots == []
            assert tree.parent_dict == no_root.parent_dict

    def test_single_tree(self):
        ts = msprime.simulate(10, random_seed=1)
        no_root_ts = self.remove_samples(ts)
        assert ts.num_trees == 1
        self.verify(ts, no_root_ts)

    def test_multiple_trees(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=1)
        no_root_ts = self.remove_samples(ts)
        assert ts.num_trees > 1
        self.verify(ts, no_root_ts)


class TestEmptyTreeSequences(TopologyTestCase):
    """
    Tests covering tree sequences that have zero edges.
    """

    def test_zero_nodes(self):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        assert ts.sequence_length == 1
        assert ts.num_trees == 1
        assert ts.num_nodes == 0
        assert ts.num_edges == 0
        t = next(ts.trees())
        assert t.index == 0
        assert t.left_root == tskit.NULL
        assert t.interval == (0, 1)
        assert t.roots == []
        assert t.root == tskit.NULL
        assert t.parent_dict == {}
        assert t.virtual_root == 0
        assert t.left_child(t.virtual_root) == -1
        assert t.right_child(t.virtual_root) == -1
        assert list(t.nodes()) == []
        assert list(ts.haplotypes()) == []
        assert list(ts.variants()) == []
        methods = [
            t.parent,
            t.left_child,
            t.right_child,
            t.left_sib,
            t.right_sib,
            t.num_children,
        ]
        for method in methods:
            for u in [-1, 1, 100]:
                with pytest.raises(ValueError):
                    method(u)
        tsp = ts.simplify()
        assert tsp.num_nodes == 0
        assert tsp.num_edges == 0

    def test_one_node_zero_samples(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=0)
        # Without a sequence length this should fail.
        ts = tables.tree_sequence()
        assert ts.sequence_length == 1
        assert ts.num_trees == 1
        assert ts.num_nodes == 1
        assert ts.sample_size == 0
        assert ts.num_edges == 0
        assert ts.num_sites == 0
        assert ts.num_mutations == 0
        t = next(ts.trees())
        assert t.index == 0
        assert t.left_root == tskit.NULL
        assert t.interval == (0, 1)
        assert t.roots == []
        assert t.root == tskit.NULL
        assert t.virtual_root == 1
        assert t.parent_dict == {}
        assert list(t.nodes()) == []
        assert list(ts.haplotypes()) == []
        assert list(ts.variants()) == []
        methods = [
            t.parent,
            t.left_child,
            t.right_child,
            t.left_sib,
            t.right_sib,
            t.num_children,
        ]
        for method in methods:
            expected = tskit.NULL if method != t.num_children else 0
            assert method(0) == expected
            for u in [-1, 2, 100]:
                with pytest.raises(ValueError):
                    method(u)

    def test_one_node_zero_samples_sites(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=0)
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.mutations.add_row(site=0, derived_state="1", node=0)
        ts = tables.tree_sequence()
        assert ts.sequence_length == 1
        assert ts.num_trees == 1
        assert ts.num_nodes == 1
        assert ts.sample_size == 0
        assert ts.num_edges == 0
        assert ts.num_sites == 1
        assert ts.num_mutations == 1
        t = next(ts.trees())
        assert t.index == 0
        assert t.left_root == tskit.NULL
        assert t.interval == (0, 1)
        assert t.roots == []
        assert t.root == tskit.NULL
        assert t.parent_dict == {}
        assert len(list(t.sites())) == 1
        assert list(t.nodes()) == []
        assert list(ts.haplotypes()) == []
        assert len(list(ts.variants())) == 1
        tsp = ts.simplify()
        assert tsp.num_nodes == 0
        assert tsp.num_edges == 0

    def test_one_node_one_sample(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        ts = tables.tree_sequence()
        assert ts.sequence_length == 1
        assert ts.num_trees == 1
        assert ts.num_nodes == 1
        assert ts.sample_size == 1
        assert ts.num_edges == 0
        t = next(ts.trees())
        assert t.index == 0
        assert t.left_root == 0
        assert t.interval == (0, 1)
        assert t.roots == [0]
        assert t.root == 0
        assert t.virtual_root == 1
        assert t.parent_dict == {}
        assert list(t.nodes()) == [0]
        assert list(ts.haplotypes(isolated_as_missing=False)) == [""]
        assert list(ts.variants()) == []
        methods = [
            t.parent,
            t.left_child,
            t.right_child,
            t.left_sib,
            t.right_sib,
            t.num_children,
        ]
        for method in methods:
            expected = tskit.NULL if method != t.num_children else 0
            assert method(0) == expected
            for u in [-1, 2, 100]:
                with pytest.raises(ValueError):
                    method(u)
        tsp = ts.simplify()
        assert tsp.num_nodes == 1
        assert tsp.num_edges == 0

    def test_one_node_one_sample_sites(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.mutations.add_row(site=0, derived_state="1", node=0)
        ts = tables.tree_sequence()
        assert ts.sequence_length == 1
        assert ts.num_trees == 1
        assert ts.num_nodes == 1
        assert ts.sample_size == 1
        assert ts.num_edges == 0
        assert ts.num_sites == 1
        assert ts.num_mutations == 1
        t = next(ts.trees())
        assert t.index == 0
        assert t.left_root == 0
        assert t.interval == (0, 1)
        assert t.roots == [0]
        assert t.root == 0
        assert t.virtual_root == 1
        assert t.parent_dict == {}
        assert list(t.nodes()) == [0]
        assert list(ts.haplotypes(isolated_as_missing=False)) == ["1"]
        assert len(list(ts.variants())) == 1
        methods = [
            t.parent,
            t.left_child,
            t.right_child,
            t.left_sib,
            t.right_sib,
            t.num_children,
        ]
        for method in methods:
            expected = tskit.NULL if method != t.num_children else 0
            assert method(0) == expected
            for u in [-1, 2, 100]:
                with pytest.raises(ValueError):
                    method(u)
        tsp = ts.simplify(filter_sites=False)
        assert tsp.num_nodes == 1
        assert tsp.num_edges == 0
        assert tsp.num_sites == 1


class TestHoleyTreeSequences(TopologyTestCase):
    """
    Tests for tree sequences in which we have partial (or no) trees defined
    over some of the sequence.
    """

    def verify_trees(self, ts, expected):
        observed = []
        for t in ts.trees():
            observed.append((t.interval, t.parent_dict))
        assert expected == observed
        # Test simple algorithm also.
        observed = []
        for interval, parent in tsutil.algorithm_T(ts):
            parent_dict = {j: parent[j] for j in range(ts.num_nodes) if parent[j] >= 0}
            observed.append((interval, parent_dict))
        assert expected == observed

    def verify_zero_roots(self, ts):
        for tree in ts.trees():
            assert tree.num_roots == 0
            assert tree.left_root == tskit.NULL
            assert tree.roots == []

    def test_simple_hole(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       2       0
        2       3       2       0
        0       1       2       1
        2       3       2       1
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        expected = [((0, 1), {0: 2, 1: 2}), ((1, 2), {}), ((2, 3), {0: 2, 1: 2})]
        self.verify_trees(ts, expected)

    def test_simple_hole_zero_roots(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   0           0
        1   0           0
        2   0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       2       0
        2       3       2       0
        0       1       2       1
        2       3       2       1
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        expected = [((0, 1), {0: 2, 1: 2}), ((1, 2), {}), ((2, 3), {0: 2, 1: 2})]
        self.verify_trees(ts, expected)
        self.verify_zero_roots(ts)

    def test_initial_gap(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        1       2       2       0,1
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        expected = [((0, 1), {}), ((1, 2), {0: 2, 1: 2})]
        self.verify_trees(ts, expected)

    def test_initial_gap_zero_roots(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   0           0
        1   0           0
        2   0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        1       2       2       0,1
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        expected = [((0, 1), {}), ((1, 2), {0: 2, 1: 2})]
        self.verify_trees(ts, expected)
        self.verify_zero_roots(ts)

    def test_final_gap(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       2       2       0,1
        """
        )
        ts = tskit.load_text(nodes, edges, sequence_length=3, strict=False)
        expected = [((0, 2), {0: 2, 1: 2}), ((2, 3), {})]
        self.verify_trees(ts, expected)

    def test_final_gap_zero_roots(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   0           0
        1   0           0
        2   0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       2       2       0,1
        """
        )
        ts = tskit.load_text(nodes, edges, sequence_length=3, strict=False)
        expected = [((0, 2), {0: 2, 1: 2}), ((2, 3), {})]
        self.verify_trees(ts, expected)
        self.verify_zero_roots(ts)

    def test_initial_and_final_gap(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        1       2       2       0,1
        """
        )
        ts = tskit.load_text(nodes, edges, sequence_length=3, strict=False)
        expected = [((0, 1), {}), ((1, 2), {0: 2, 1: 2}), ((2, 3), {})]
        self.verify_trees(ts, expected)

    def test_initial_and_final_gap_zero_roots(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   0           0
        1   0           0
        2   0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        1       2       2       0,1
        """
        )
        ts = tskit.load_text(nodes, edges, sequence_length=3, strict=False)
        expected = [((0, 1), {}), ((1, 2), {0: 2, 1: 2}), ((2, 3), {})]
        self.verify_trees(ts, expected)
        self.verify_zero_roots(ts)


class TestTsinferExamples(TopologyTestCase):
    """
    Test cases on troublesome topology examples that arose from tsinfer.
    """

    def test_no_last_tree(self):
        # The last tree was not being generated here because of a bug in
        # the low-level tree generation code.
        nodes = io.StringIO(
            """\
        id      is_sample   population      time
        0       1       -1              3.00000000000000
        1       1       -1              2.00000000000000
        2       1       -1              2.00000000000000
        3       1       -1              2.00000000000000
        4       1       -1              2.00000000000000
        5       1       -1              1.00000000000000
        6       1       -1              1.00000000000000
        7       1       -1              1.00000000000000
        8       1       -1              1.00000000000000
        9       1       -1              1.00000000000000
        10      1       -1              1.00000000000000
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       62291.41659631  79679.17408763  1       5
        1       62291.41659631  62374.60889677  1       6
        2       122179.36037089 138345.43104411 1       7
        3       67608.32330402  79679.17408763  1       8
        4       122179.36037089 138345.43104411 1       8
        5       62291.41659631  79679.17408763  1       9
        6       126684.47550333 138345.43104411 1       10
        7       23972.05905068  62291.41659631  2       5
        8       79679.17408763  82278.53390076  2       5
        9       23972.05905068  62291.41659631  2       6
        10      79679.17408763  110914.43816806 2       7
        11      145458.28890561 189765.31932273 2       7
        12      79679.17408763  110914.43816806 2       8
        13      145458.28890561 200000.00000000 2       8
        14      23972.05905068  62291.41659631  2       9
        15      79679.17408763  110914.43816806 2       9
        16      145458.28890561 145581.18329797 2       10
        17      4331.62138785   23972.05905068  3       6
        18      4331.62138785   23972.05905068  3       9
        19      110914.43816806 122179.36037089 4       7
        20      138345.43104411 145458.28890561 4       7
        21      110914.43816806 122179.36037089 4       8
        22      138345.43104411 145458.28890561 4       8
        23      110914.43816806 112039.30503475 4       9
        24      138345.43104411 145458.28890561 4       10
        25      0.00000000      200000.00000000 0       1
        26      0.00000000      200000.00000000 0       2
        27      0.00000000      200000.00000000 0       3
        28      0.00000000      200000.00000000 0       4
        """
        )
        ts = tskit.load_text(nodes, edges, sequence_length=200000, strict=False)
        pts = tests.PythonTreeSequence(ts)
        num_trees = 0
        for _ in pts.trees():
            num_trees += 1
        assert num_trees == ts.num_trees
        n = 0
        for pt, t in zip(pts.trees(), ts.trees()):
            assert (pt.left, pt.right) == t.interval
            for j in range(ts.num_nodes):
                assert pt.parent[j] == t.parent(j)
                assert pt.left_child[j] == t.left_child(j)
                assert pt.right_child[j] == t.right_child(j)
                assert pt.left_sib[j] == t.left_sib(j)
                assert pt.right_sib[j] == t.right_sib(j)
                assert pt.num_children[j] == t.num_children(j)
            n += 1
        assert n == num_trees
        intervals = [t.interval for t in ts.trees()]
        assert intervals[0][0] == 0
        assert intervals[-1][-1] == ts.sequence_length


class TestRecordSquashing(TopologyTestCase):
    """
    Tests that we correctly squash adjacent equal records together.
    """

    def test_single_record(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       1       0
        1       2       1       0
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        tss, node_map = ts.simplify(map_nodes=True)
        assert list(node_map) == [0, 1]
        assert tss.tables.nodes == ts.tables.nodes
        simplified_edges = list(tss.edges())
        assert len(simplified_edges) == 1
        e = simplified_edges[0]
        assert e.left == 0
        assert e.right == 2

    def test_single_tree(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts_redundant = tsutil.insert_redundant_breakpoints(ts)
        tss = ts_redundant.simplify()
        assert tss.tables.nodes == ts.tables.nodes
        assert tss.tables.edges == ts.tables.edges

    def test_many_trees(self):
        ts = msprime.simulate(20, recombination_rate=5, random_seed=self.random_seed)
        assert ts.num_trees > 2
        ts_redundant = tsutil.insert_redundant_breakpoints(ts)
        tss = ts_redundant.simplify()
        assert tss.tables.nodes == ts.tables.nodes
        assert tss.tables.edges == ts.tables.edges


class TestRedundantBreakpoints(TopologyTestCase):
    """
    Tests for dealing with redundant breakpoints within the tree sequence.
    These are records that may be squashed together into a single record.
    """

    def test_single_tree(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        ts_redundant = tsutil.insert_redundant_breakpoints(ts)
        assert ts.sample_size == ts_redundant.sample_size
        assert ts.sequence_length == ts_redundant.sequence_length
        assert ts_redundant.num_trees == 2
        trees = [t.parent_dict for t in ts_redundant.trees()]
        assert len(trees) == 2
        assert trees[0] == trees[1]
        assert [t.parent_dict for t in ts.trees()][0] == trees[0]

    def test_many_trees(self):
        ts = msprime.simulate(20, recombination_rate=5, random_seed=self.random_seed)
        assert ts.num_trees > 2
        ts_redundant = tsutil.insert_redundant_breakpoints(ts)
        assert ts.sample_size == ts_redundant.sample_size
        assert ts.sequence_length == ts_redundant.sequence_length
        assert ts_redundant.num_trees > ts.num_trees
        assert ts_redundant.num_edges > ts.num_edges
        redundant_trees = ts_redundant.trees()
        redundant_t = next(redundant_trees)
        comparisons = 0
        for t in ts.trees():
            while (
                redundant_t is not None
                and redundant_t.interval.right <= t.interval.right
            ):
                assert t.parent_dict == redundant_t.parent_dict
                comparisons += 1
                redundant_t = next(redundant_trees, None)
        assert comparisons == ts_redundant.num_trees


class TestUnaryNodes(TopologyTestCase):
    """
    Tests for situations in which we have unary nodes in the tree sequence.
    """

    def test_simple_case(self):
        # Simple case where we have n = 2 and some unary nodes.
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           1
        4       0           2
        5       0           3
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       2       0
        0       1       3       1
        0       1       4       2,3
        0       1       5       4
        """
        )
        sites = "position    ancestral_state\n"
        mutations = "site    node    derived_state\n"
        for j in range(5):
            position = j * 1 / 5
            sites += f"{position} 0\n"
            mutations += f"{j} {j} 1\n"
        ts = tskit.load_text(
            nodes=nodes,
            edges=edges,
            sites=io.StringIO(sites),
            mutations=io.StringIO(mutations),
            strict=False,
        )

        assert ts.sample_size == 2
        assert ts.num_nodes == 6
        assert ts.num_trees == 1
        assert ts.num_sites == 5
        assert ts.num_mutations == 5
        assert len(list(ts.edge_diffs())) == ts.num_trees
        t = next(ts.trees())
        assert t.parent_dict == {0: 2, 1: 3, 2: 4, 3: 4, 4: 5}
        assert t.mrca(0, 1) == 4
        assert t.mrca(0, 2) == 2
        assert t.mrca(0, 4) == 4
        assert t.mrca(0, 5) == 5
        assert t.mrca(0, 3) == 4
        H = list(ts.haplotypes())
        assert H[0] == "10101"
        assert H[1] == "01011"

    def test_ladder_tree(self):
        # We have a single tree with a long ladder of unary nodes along a path
        num_unary_nodes = 30
        n = 2
        nodes = """\
            is_sample   time
            1           0
            1           0
        """
        edges = """\
            left right parent child
            0    1     2      0
        """
        for j in range(num_unary_nodes + 2):
            nodes += f"0 {j + 2}\n"
        for j in range(num_unary_nodes):
            edges += f"0 1 {n + j + 1} {n + j}\n"
        root = num_unary_nodes + 3
        root_time = num_unary_nodes + 3
        edges += f"0    1     {root}      1,{num_unary_nodes + 2}\n"
        ts = tskit.load_text(io.StringIO(nodes), io.StringIO(edges), strict=False)
        t = ts.first()
        assert t.mrca(0, 1) == root
        assert t.tmrca(0, 1) == root_time
        ts_simplified, node_map = ts.simplify(map_nodes=True)
        test_map = [tskit.NULL for _ in range(ts.num_nodes)]
        test_map[0] = 0
        test_map[1] = 1
        test_map[root] = 2
        assert list(node_map) == test_map
        assert ts_simplified.num_edges == 2
        t = ts_simplified.first()
        assert t.mrca(0, 1) == 2
        assert t.tmrca(0, 1) == root_time
        ts_simplified = ts.simplify(keep_unary=True, record_provenance=False)
        assert ts_simplified.tables == ts.tables

    def verify_unary_tree_sequence(self, ts):
        """
        Take the specified tree sequence and produce an equivalent in which
        unary records have been interspersed, every other with an associated individual
        """
        assert ts.num_trees > 2
        assert ts.num_mutations > 2
        tables = ts.dump_tables()
        next_node = ts.num_nodes
        node_times = {j: node.time for j, node in enumerate(ts.nodes())}
        edges = []
        for i, e in enumerate(ts.edges()):
            node = ts.node(e.parent)
            t = node.time - 1e-14  # Arbitrary small value.
            next_node = len(tables.nodes)
            indiv = tables.individuals.add_row() if i % 2 == 0 else tskit.NULL
            tables.nodes.add_row(time=t, population=node.population, individual=indiv)
            edges.append(
                tskit.Edge(left=e.left, right=e.right, parent=next_node, child=e.child)
            )
            node_times[next_node] = t
            edges.append(
                tskit.Edge(left=e.left, right=e.right, parent=e.parent, child=next_node)
            )
        edges.sort(key=lambda e: node_times[e.parent])
        tables.edges.reset()
        for e in edges:
            tables.edges.append(e)
        ts_new = tables.tree_sequence()
        assert ts_new.num_edges > ts.num_edges
        self.assert_haplotypes_equal(ts, ts_new)
        self.assert_variants_equal(ts, ts_new)
        ts_simplified = ts_new.simplify()
        assert list(ts_simplified.records()) == list(ts.records())
        self.assert_haplotypes_equal(ts, ts_simplified)
        self.assert_variants_equal(ts, ts_simplified)
        assert len(list(ts.edge_diffs())) == ts.num_trees
        assert 0 < ts_new.num_individuals < ts_new.num_nodes

        for params in [
            {"keep_unary": False, "keep_unary_in_individuals": False},
            {"keep_unary": True, "keep_unary_in_individuals": False},
            {"keep_unary": False, "keep_unary_in_individuals": True},
        ]:
            s = tests.Simplifier(ts_new, ts_new.samples(), **params)
            py_ts, py_node_map = s.simplify()
            lib_ts, lib_node_map = ts_new.simplify(map_nodes=True, **params)
            py_tables = py_ts.dump_tables()
            lib_tables = lib_ts.dump_tables()
            lib_tables.assert_equals(py_tables, ignore_provenance=True)
            assert np.all(lib_node_map == py_node_map)

    def test_binary_tree_sequence_unary_nodes(self):
        ts = msprime.simulate(
            20, recombination_rate=5, mutation_rate=5, random_seed=self.random_seed
        )
        self.verify_unary_tree_sequence(ts)

    def test_nonbinary_tree_sequence_unary_nodes(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=1.0, population=0, proportion=0.95)
        ]
        ts = msprime.simulate(
            20,
            recombination_rate=10,
            mutation_rate=5,
            demographic_events=demographic_events,
            random_seed=self.random_seed,
        )
        found = False
        for r in ts.edgesets():
            if len(r.children) > 2:
                found = True
        assert found
        self.verify_unary_tree_sequence(ts)


class TestGeneralSamples(TopologyTestCase):
    """
    Test cases in which we have samples at arbitrary nodes (i.e., not at
    {0,...,n - 1}).
    """

    def test_simple_case(self):
        # Simple case where we have n = 3 and samples starting at n.
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       0           2
        1       0           1
        2       1           0
        3       1           0
        4       1           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       1       2,3
        0       1       0       1,4
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.1     0
        0.2     0
        0.3     0
        0.4     0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state
        0       2       1
        1       3       1
        2       4       1
        3       1       1
        """
        )
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
        )

        assert ts.sample_size == 3
        assert list(ts.samples()) == [2, 3, 4]
        assert ts.num_nodes == 5
        assert ts.num_nodes == 5
        assert ts.num_sites == 4
        assert ts.num_mutations == 4
        assert len(list(ts.edge_diffs())) == ts.num_trees
        t = next(ts.trees())
        assert t.root == 0
        assert t.parent_dict == {1: 0, 2: 1, 3: 1, 4: 0}
        H = list(ts.haplotypes())
        assert H[0] == "1001"
        assert H[1] == "0101"
        assert H[2] == "0010"

        tss, node_map = ts.simplify(map_nodes=True)
        assert list(node_map) == [4, 3, 0, 1, 2]
        # We should have the same tree sequence just with canonicalised nodes.
        assert tss.sample_size == 3
        assert list(tss.samples()) == [0, 1, 2]
        assert tss.num_nodes == 5
        assert tss.num_trees == 1
        assert tss.num_sites == 4
        assert tss.num_mutations == 4
        assert len(list(ts.edge_diffs())) == ts.num_trees
        t = next(tss.trees())
        assert t.root == 4
        assert t.parent_dict == {0: 3, 1: 3, 2: 4, 3: 4}
        H = list(tss.haplotypes())
        assert H[0] == "1001"
        assert H[1] == "0101"
        assert H[2] == "0010"

    def verify_permuted_nodes(self, ts):
        """
        Take the specified tree sequence and permute the nodes, verifying that we
        get back a tree sequence with the correct properties.
        """
        # Mapping from the original nodes into nodes in the new tree sequence.
        node_map = list(range(ts.num_nodes))
        random.shuffle(node_map)
        # Change the permutation so that the relative order of samples is maintained.
        # Then, we should get back exactly the same tree sequence after simplify
        # and haplotypes and variants are also equal.
        samples = sorted(node_map[: ts.sample_size])
        node_map = samples + node_map[ts.sample_size :]
        permuted = tsutil.permute_nodes(ts, node_map)
        assert ts.sequence_length == permuted.sequence_length
        assert list(permuted.samples()) == samples
        assert list(permuted.haplotypes()) == list(ts.haplotypes())
        for v1, v2 in zip(
            permuted.variants(copy=False),
            ts.variants(copy=False),
        ):
            assert np.array_equal(v1.genotypes, v2.genotypes)

        assert ts.num_trees == permuted.num_trees
        j = 0
        for t1, t2 in zip(ts.trees(), permuted.trees()):
            t1_dict = {node_map[k]: node_map[v] for k, v in t1.parent_dict.items()}
            assert node_map[t1.root] == t2.root
            assert t1_dict == t2.parent_dict
            for u1 in t1.nodes():
                u2 = node_map[u1]
                assert sorted(node_map[v] for v in t1.samples(u1)) == sorted(
                    list(t2.samples(u2))
                )
            j += 1
        assert j == ts.num_trees

        # The simplified version of the permuted tree sequence should be in canonical
        # form, and identical to the original.
        simplified, s_node_map = permuted.simplify(map_nodes=True)

        for u, v in enumerate(node_map):
            assert s_node_map[v] == u
        ts.tables.assert_equals(simplified.tables, ignore_provenance=True)

    def test_single_tree_permuted_nodes(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=self.random_seed)
        self.verify_permuted_nodes(ts)

    def test_binary_tree_sequence_permuted_nodes(self):
        ts = msprime.simulate(
            20, recombination_rate=5, mutation_rate=5, random_seed=self.random_seed
        )
        self.verify_permuted_nodes(ts)

    def test_nonbinary_tree_sequence_permuted_nodes(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=1.0, population=0, proportion=0.95)
        ]
        ts = msprime.simulate(
            20,
            recombination_rate=10,
            mutation_rate=5,
            demographic_events=demographic_events,
            random_seed=self.random_seed,
        )
        found = False
        for e in ts.edgesets():
            if len(e.children) > 2:
                found = True
        assert found
        self.verify_permuted_nodes(ts)


class TestSimplifyExamples(TopologyTestCase):
    """
    Tests for simplify where we write out the input and expected output
    or we detect expected errors.
    """

    def verify_simplify(
        self,
        samples,
        *,
        filter_sites=True,
        keep_input_roots=False,
        filter_nodes=True,
        nodes_before=None,
        edges_before=None,
        sites_before=None,
        mutations_before=None,
        nodes_after=None,
        edges_after=None,
        sites_after=None,
        mutations_after=None,
        debug=False,
    ):
        """
        Verifies that if we run simplify on the specified input we get the
        required output.
        """
        before = tskit.load_text(
            nodes=io.StringIO(nodes_before),
            edges=io.StringIO(edges_before),
            sites=io.StringIO(sites_before) if sites_before is not None else None,
            mutations=(
                io.StringIO(mutations_before) if mutations_before is not None else None
            ),
            strict=False,
        )

        after = tskit.load_text(
            nodes=io.StringIO(nodes_after),
            edges=io.StringIO(edges_after),
            sites=io.StringIO(sites_after) if sites_after is not None else None,
            mutations=(
                io.StringIO(mutations_after) if mutations_after is not None else None
            ),
            strict=False,
            sequence_length=before.sequence_length,
        )

        result, _ = do_simplify(
            before,
            samples=samples,
            filter_sites=filter_sites,
            keep_input_roots=keep_input_roots,
            filter_nodes=filter_nodes,
            compare_lib=True,
        )
        if debug:
            print("before")
            print(before)
            print(before.draw_text())
            print("after")
            print(after)
            print(after.draw_text())
            print("result")
            print(result)
            print(result.draw_text())
        after.tables.assert_equals(result.tables)

    def test_unsorted_edges(self):
        # We have two nodes at the same time and interleave edges for
        # these nodes together. This is an error because all edges for
        # a given parent must be contigous.
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           1
        """
        edges_before = """\
        left    right   parent  child
        0       1       2       0,1
        0       1       3       0,1
        1       2       2       0,1
        1       2       3       0,1
        """
        nodes = tskit.parse_nodes(io.StringIO(nodes_before), strict=False)
        edges = tskit.parse_edges(io.StringIO(edges_before), strict=False)
        # Cannot use load_text here because it calls sort()
        tables = tskit.TableCollection(sequence_length=2)
        tables.nodes.set_columns(**nodes.asdict())
        tables.edges.set_columns(**edges.asdict())
        with pytest.raises(_tskit.LibraryError):
            tables.simplify(samples=[0, 1])

    def test_single_binary_tree(self):
        #
        # 2        4
        #         / \
        # 1      3   \
        #       / \   \
        # 0   (0)(1)  (2)
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        # We sample 0 and 2, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           2
        """
        edges_after = """\
        left    right   parent  child
        0       1       2       0,1
        """
        self.verify_simplify(
            samples=[0, 2],
            nodes_before=nodes_before,
            edges_before=edges_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
        )

    def test_single_binary_tree_no_sample_nodes(self):
        #
        # 2        4
        #         / \
        # 1      3   \
        #       / \   \
        # 0   (0)(1)  (2)
        nodes_before = """\
        id      is_sample   time
        0       0           0
        1       0           0
        2       0           0
        3       0           1
        4       0           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        # We sample 0 and 2, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           2
        """
        edges_after = """\
        left    right   parent  child
        0       1       2       0,1
        """
        self.verify_simplify(
            samples=[0, 2],
            nodes_before=nodes_before,
            edges_before=edges_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
        )

    def test_single_binary_tree_keep_input_root(self):
        #
        # 2        4
        #         / \
        # 1      3   \
        #       / \   \
        # 0   (0)(1)  (2)
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           2
        """
        edges_after = """\
        left    right   parent  child
        0       1       2       0,1
        0       1       3       2
        """
        self.verify_simplify(
            samples=[0, 1],
            nodes_before=nodes_before,
            edges_before=edges_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
            keep_input_roots=True,
        )

    def test_single_binary_tree_internal_sample(self):
        #
        # 2        4
        #         / \
        # 1     (3)  \
        #       / \   \
        # 0   (0)  1  (2)
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0
        3       1           1
        4       0           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        # We sample 0 and 3, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           1
        """
        edges_after = """\
        left    right   parent  child
        0       1       1       0
        """
        self.verify_simplify(
            samples=[0, 3],
            nodes_before=nodes_before,
            edges_before=edges_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
        )

    def test_single_binary_tree_internal_sample_meet_at_root(self):
        # 3          5
        #           / \
        # 2        4  (6)
        #         / \
        # 1     (3)  \
        #       / \   \
        # 0   (0)  1   2
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0
        3       1           1
        4       0           2
        5       0           3
        6       1           2
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        0       1       5       4,6
        """
        # We sample 0 and 3 and 6, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           1
        2       1           2
        3       0           3
        """
        edges_after = """\
        left    right   parent  child
        0       1       1       0
        0       1       3       1,2
        """
        self.verify_simplify(
            samples=[0, 3, 6],
            nodes_before=nodes_before,
            edges_before=edges_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
        )

    def test_single_binary_tree_simple_mutations(self):
        # 3          5
        #           / \
        # 2        4   \
        #         / \   s0
        # 1      3   s1  \
        #       / \   \   \
        # 0   (0) (1)  2  (6)
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0
        3       0           1
        4       0           2
        5       0           3
        6       1           0
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        0       1       5       4,6
        """
        sites_before = """\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        """
        mutations_before = """\
        site    node    derived_state
        0       6       1
        1       2       1
        """

        # We sample 0 and 2 and 6, so we get
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        3       0           3
        """
        edges_after = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        """
        sites_after = """\
        id  position    ancestral_state
        0   0.1         0
        """
        mutations_after = """\
        site    node    derived_state
        0       2       1
        """
        self.verify_simplify(
            samples=[0, 1, 6],
            nodes_before=nodes_before,
            edges_before=edges_before,
            sites_before=sites_before,
            mutations_before=mutations_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
            sites_after=sites_after,
            mutations_after=mutations_after,
        )
        # If we don't filter the fixed sites, we should get the same
        # mutations and the original sites table back.
        self.verify_simplify(
            samples=[0, 1, 6],
            filter_sites=False,
            nodes_before=nodes_before,
            edges_before=edges_before,
            sites_before=sites_before,
            mutations_before=mutations_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
            sites_after=sites_before,
            mutations_after=mutations_after,
        )

    def test_single_binary_tree_keep_roots_mutations(self):
        # 3          5
        #        m0 / \
        # 2        4   \
        #      m1 / \   \
        # 1      3   \   \
        #       / \   \   \
        # 0   (0) (1)  2   6
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0
        3       0           1
        4       0           2
        5       0           3
        6       0           0
        """
        edges_before = """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       2,3
        0       1       5       4,6
        """
        sites_before = """\
        id  position    ancestral_state
        0   0.1         0
        """
        mutations_before = """\
        site    node    derived_state parent
        0       4       1             -1
        0       3       2             0
        """

        # We sample 0 and 2
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           3
        """
        edges_after = """\
        left    right   parent  child
        0       1       2       0,1
        0       1       3       2
        """
        sites_after = """\
        id  position    ancestral_state
        0   0.1         0
        """
        mutations_after = """\
        site    node    derived_state parent
        0       2       1             -1
        0       2       2             0
        """
        self.verify_simplify(
            samples=[0, 1],
            nodes_before=nodes_before,
            edges_before=edges_before,
            sites_before=sites_before,
            mutations_before=mutations_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
            sites_after=sites_after,
            mutations_after=mutations_after,
            keep_input_roots=True,
        )

    def test_place_mutations_with_and_without_roots(self):
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       0           1
        2       0           2
        """
        edges_before = """\
        left    right   parent  child
        0       2       1       0
        0       2       2       1
        """
        sites = """\
        id  position    ancestral_state
        0   1.0         0
        """
        mutations_before = """\
        site    node    derived_state time parent
        0       2       3             2    -1
        0       1       1             1    0
        0       0       2             0    1
        """
        # expected result without keep_input_roots
        nodes_after = """\
        id      is_sample   time
        0       1           0
        """
        edges_after = """\
        left    right   parent  child
        """
        mutations_after = """\
        site    node    derived_state time parent
        0       0       3             2    -1
        0       0       1             1    0
        0       0       2             0    1
        """
        # expected result with keep_input_roots
        nodes_after_keep = """\
        id      is_sample   time
        0       1           0
        1       0           2
        """
        edges_after_keep = """\
        left    right   parent  child
        0       2       1       0
        """
        mutations_after_keep = """\
        site    node    derived_state time parent
        0       1       3             2    -1
        0       0       1             1    0
        0       0       2             0    1
        """
        self.verify_simplify(
            samples=[0],
            nodes_before=nodes_before,
            edges_before=edges_before,
            sites_before=sites,
            mutations_before=mutations_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
            sites_after=sites,
            mutations_after=mutations_after,
            keep_input_roots=False,
        )
        self.verify_simplify(
            samples=[0],
            nodes_before=nodes_before,
            edges_before=edges_before,
            sites_before=sites,
            mutations_before=mutations_before,
            nodes_after=nodes_after_keep,
            edges_after=edges_after_keep,
            sites_after=sites,
            mutations_after=mutations_after_keep,
            keep_input_roots=True,
        )

    def test_overlapping_edges(self):
        nodes = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        """
        edges_before = """\
        left    right   parent  child
        0       2       2       0
        1       3       2       1
        """
        # We resolve the overlapping edges here. Since the flanking regions
        # have no interesting edges, these are left out of the output.
        edges_after = """\
        left    right   parent  child
        1       2       2       0,1
        """
        self.verify_simplify(
            samples=[0, 1],
            nodes_before=nodes,
            edges_before=edges_before,
            nodes_after=nodes,
            edges_after=edges_after,
        )

    def test_overlapping_edges_internal_samples(self):
        nodes = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           1
        """
        edges = """\
        left    right   parent  child
        0       2       2       0
        1       3       2       1
        """
        self.verify_simplify(
            samples=[0, 1, 2],
            nodes_before=nodes,
            edges_before=edges,
            nodes_after=nodes,
            edges_after=edges,
        )

    def test_unary_edges_no_overlap(self):
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        """
        edges_before = """\
        left    right   parent  child
        0       2       2       0
        2       3       2       1
        """
        # Because there is no overlap between the samples, we just get an
        # empty set of output edges.
        nodes_after = """\
        id      is_sample   time
        0       1           0
        1       1           0
        """
        edges_after = """\
        left    right   parent  child
        """
        self.verify_simplify(
            samples=[0, 1],
            nodes_before=nodes_before,
            edges_before=edges_before,
            nodes_after=nodes_after,
            edges_after=edges_after,
        )

    def test_unary_edges_no_overlap_internal_sample(self):
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           1
        """
        edges_before = """\
        left    right   parent  child
        0       1       2       0
        1       2       2       1
        """
        self.verify_simplify(
            samples=[0, 1, 2],
            nodes_before=nodes_before,
            edges_before=edges_before,
            nodes_after=nodes_before,
            edges_after=edges_before,
        )

    def test_keep_nodes(self):
        nodes_before = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           2
        4       0           3
        """
        edges_before = """\
        left    right   parent  child
        0       1       2       0
        0       1       2       1
        0       1       3       2
        0       1       4       3
        """
        edges_after = """\
        left    right   parent  child
        0       1       2       0
        0       1       2       1
        0       1       4       2
        """
        self.verify_simplify(
            samples=[0, 1],
            nodes_before=nodes_before,
            edges_before=edges_before,
            nodes_after=nodes_before,
            edges_after=edges_after,
            filter_nodes=False,
            keep_input_roots=True,
        )


class TestNonSampleExternalNodes(TopologyTestCase):
    """
    Tests for situations in which we have tips that are not samples.
    """

    def test_simple_case(self):
        # Simplest case where we have n = 2 and external non-sample nodes.
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           0
        4       0           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       2       0,1,3,4
        """
        )
        sites = io.StringIO(
            """\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        2   0.3         0
        3   0.4         0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state
        0       0       1
        1       1       1
        2       3       1
        3       4       1
        """
        )
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
        )
        assert ts.sample_size == 2
        assert ts.num_trees == 1
        assert ts.num_nodes == 5
        assert ts.num_sites == 4
        assert ts.num_mutations == 4
        t = next(ts.trees())
        assert t.parent_dict == {0: 2, 1: 2, 3: 2, 4: 2}
        assert t.root == 2
        ts_simplified, node_map = ts.simplify(map_nodes=True)
        assert list(node_map) == [0, 1, 2, -1, -1]
        assert ts_simplified.num_nodes == 3
        assert ts_simplified.num_trees == 1
        t = next(ts_simplified.trees())
        assert t.parent_dict == {0: 2, 1: 2}
        assert t.root == 2
        # We should have removed the two non-sample mutations.
        assert [s.position for s in t.sites()] == [0.1, 0.2]

    def test_unary_non_sample_external_nodes(self):
        # Take an ordinary tree sequence and put a bunch of external non
        # sample nodes on it.
        ts = msprime.simulate(
            15, recombination_rate=5, random_seed=self.random_seed, mutation_rate=5
        )
        assert ts.num_trees > 2
        assert ts.num_mutations > 2
        tables = ts.dump_tables()
        next_node = ts.num_nodes
        tables.edges.reset()
        for e in ts.edges():
            tables.edges.append(e)
            tables.edges.append(e.replace(child=next_node))
            tables.nodes.add_row(time=0)
            next_node += 1
        tables.sort()
        ts_new = tables.tree_sequence()
        assert ts_new.num_nodes == next_node
        assert ts_new.sample_size == ts.sample_size
        self.assert_haplotypes_equal(ts, ts_new)
        self.assert_variants_equal(ts, ts_new)
        ts_simplified = ts_new.simplify()
        assert ts_simplified.num_nodes == ts.num_nodes
        assert ts_simplified.sample_size == ts.sample_size
        assert list(ts_simplified.records()) == list(ts.records())
        self.assert_haplotypes_equal(ts, ts_simplified)
        self.assert_variants_equal(ts, ts_simplified)


class TestMultipleRoots(TopologyTestCase):
    """
    Tests for situations where we have multiple roots for the samples.
    """

    def test_simplest_degenerate_case(self):
        # Simplest case where we have n = 2 and no edges.
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        """
        )
        sites = io.StringIO(
            """\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state
        0       0         1
        1       1         1
        """
        )
        ts = tskit.load_text(
            nodes=nodes,
            edges=edges,
            sites=sites,
            mutations=mutations,
            sequence_length=1,
            strict=False,
        )
        assert ts.num_nodes == 2
        assert ts.num_trees == 1
        assert ts.num_sites == 2
        assert ts.num_mutations == 2
        t = next(ts.trees())
        assert t.parent_dict == {}
        assert sorted(t.roots) == [0, 1]
        assert list(ts.haplotypes(isolated_as_missing=False)) == ["10", "01"]
        assert np.array_equal(
            np.stack([v.genotypes for v in ts.variants(isolated_as_missing=False)]),
            [[1, 0], [0, 1]],
        )
        simplified = ts.simplify()
        t1 = ts.dump_tables()
        t2 = simplified.dump_tables()
        assert t1.nodes == t2.nodes
        assert t1.edges == t2.edges

    def test_simplest_non_degenerate_case(self):
        # Simplest case where we have n = 4 and two trees.
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       0           1
        5       0           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       4       0,1
        0       1       5       2,3
        """
        )
        sites = io.StringIO(
            """\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        2   0.3         0
        3   0.4         0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state
        0       0       1
        1       1       1
        2       2       1
        3       3       1
        """
        )
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
        )
        assert ts.num_nodes == 6
        assert ts.num_trees == 1
        assert ts.num_sites == 4
        assert ts.num_mutations == 4
        t = next(ts.trees())
        assert t.parent_dict == {0: 4, 1: 4, 2: 5, 3: 5}
        assert list(ts.haplotypes()) == ["1000", "0100", "0010", "0001"]
        assert np.array_equal(
            np.stack([v.genotypes for v in ts.variants()]),
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        assert t.mrca(0, 1) == 4
        assert t.mrca(0, 4) == 4
        assert t.mrca(2, 3) == 5
        assert t.mrca(0, 2) == tskit.NULL
        assert t.mrca(0, 3) == tskit.NULL
        assert t.mrca(2, 4) == tskit.NULL
        ts_simplified, node_map = ts.simplify(map_nodes=True)
        for j in range(4):
            assert node_map[j] == j
        assert ts_simplified.num_nodes == 6
        assert ts_simplified.num_trees == 1
        assert ts_simplified.num_sites == 4
        assert ts_simplified.num_mutations == 4
        t = next(ts_simplified.trees())
        assert t.parent_dict == {0: 4, 1: 4, 2: 5, 3: 5}

    def test_two_reducible_trees(self):
        # We have n = 4 and two trees, with some unary nodes and non-sample leaves
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       0           1
        5       0           1
        6       0           2
        7       0           3
        8       0           0   # Non sample leaf
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1      4         0
        0       1      5         1
        0       1      6         4,5
        0       1      7         2,3,8
        """
        )
        sites = io.StringIO(
            """\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        2   0.3         0
        3   0.4         0
        4   0.5         0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state
        0       0       1
        1       1       1
        2       2       1
        3       3       1
        4       8       1
        """
        )
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
        )
        assert ts.num_nodes == 9
        assert ts.num_trees == 1
        assert ts.num_sites == 5
        assert ts.num_mutations == 5
        t = next(ts.trees())
        assert t.parent_dict == {0: 4, 1: 5, 2: 7, 3: 7, 4: 6, 5: 6, 8: 7}
        assert list(ts.haplotypes()) == ["10000", "01000", "00100", "00010"]
        assert np.array_equal(
            np.stack([v.genotypes for v in ts.variants()]),
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        )
        assert t.mrca(0, 1) == 6
        assert t.mrca(2, 3) == 7
        assert t.mrca(2, 8) == 7
        assert t.mrca(0, 2) == tskit.NULL
        assert t.mrca(0, 3) == tskit.NULL
        assert t.mrca(0, 8) == tskit.NULL
        ts_simplified, node_map = ts.simplify(map_nodes=True)
        for j in range(4):
            assert node_map[j] == j
        assert ts_simplified.num_nodes == 6
        assert ts_simplified.num_trees == 1
        t = next(ts_simplified.trees())
        assert list(ts_simplified.haplotypes()) == ["1000", "0100", "0010", "0001"]
        assert np.array_equal(
            np.stack([v.genotypes for v in ts_simplified.variants()]),
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        # The site over the non-sample external node should have been discarded.
        sites = list(t.sites())
        assert sites[-1].position == 0.4
        assert t.parent_dict == {0: 4, 1: 4, 2: 5, 3: 5}

    def test_one_reducible_tree(self):
        # We have n = 4 and two trees. One tree is reducible and the other isn't.
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       0           1
        5       0           1
        6       0           2
        7       0           3
        8       0           0   # Non sample leaf
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1      4         0
        0       1      5         1
        0       1      6         4,5
        0       1      7         2,3,8
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        assert ts.num_nodes == 9
        assert ts.num_trees == 1
        t = next(ts.trees())
        assert t.parent_dict == {0: 4, 1: 5, 2: 7, 3: 7, 4: 6, 5: 6, 8: 7}
        assert t.mrca(0, 1) == 6
        assert t.mrca(2, 3) == 7
        assert t.mrca(2, 8) == 7
        assert t.mrca(0, 2) == tskit.NULL
        assert t.mrca(0, 3) == tskit.NULL
        assert t.mrca(0, 8) == tskit.NULL
        ts_simplified = ts.simplify()
        assert ts_simplified.num_nodes == 6
        assert ts_simplified.num_trees == 1
        t = next(ts_simplified.trees())
        assert t.parent_dict == {0: 4, 1: 4, 2: 5, 3: 5}

    # NOTE: This test has not been checked since updating to the text representation
    # so there might be other problems with it.
    def test_mutations_over_roots(self):
        # Mutations over root nodes should be ok when we have multiple roots.
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        5       0           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       3       0,1
        0       1       4       3
        0       1       5       2
        """
        )
        sites = io.StringIO(
            """\
        id  position    ancestral_state
        0   0.1         0
        1   0.2         0
        2   0.3         0
        3   0.4         0
        4   0.5         0
        5   0.6         0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state
        0       0       1
        1       1       1
        2       3       1
        3       4       1
        4       2       1
        5       5       1
        """
        )
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
        )
        assert ts.num_nodes == 6
        assert ts.num_trees == 1
        assert ts.num_sites == 6
        assert ts.num_mutations == 6
        t = next(ts.trees())
        assert len(list(t.sites())) == 6
        haplotypes = ["101100", "011100", "000011"]
        variants = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]]
        assert list(ts.haplotypes()) == haplotypes
        assert np.array_equal(np.stack([v.genotypes for v in ts.variants()]), variants)
        ts_simplified = ts.simplify(filter_sites=False)
        assert list(ts_simplified.haplotypes(isolated_as_missing=False)) == haplotypes
        assert np.array_equal(
            np.stack(
                [v.genotypes for v in ts_simplified.variants(isolated_as_missing=False)]
            ),
            variants,
        )

    def test_break_single_tree(self):
        # Take a single largish tree from tskit, and remove the oldest record.
        # This breaks it into two subtrees.
        ts = msprime.simulate(20, random_seed=self.random_seed, mutation_rate=4)
        assert ts.num_mutations > 5
        tables = ts.dump_tables()
        tables.edges.set_columns(
            left=tables.edges.left[:-1],
            right=tables.edges.right[:-1],
            parent=tables.edges.parent[:-1],
            child=tables.edges.child[:-1],
        )
        ts_new = tables.tree_sequence()
        assert ts.sample_size == ts_new.sample_size
        assert ts.num_edges == ts_new.num_edges + 1
        assert ts.num_trees == ts_new.num_trees
        self.assert_haplotypes_equal(ts, ts_new)
        self.assert_variants_equal(ts, ts_new)
        roots = set()
        t_new = next(ts_new.trees())
        for u in ts_new.samples():
            while t_new.parent(u) != tskit.NULL:
                u = t_new.parent(u)
            roots.add(u)
        assert len(roots) == 2
        assert sorted(roots) == sorted(t_new.roots)


class TestWithVisuals(TopologyTestCase):
    """
    Some pedantic tests with ascii depictions of what's supposed to happen.
    """

    def verify_simplify_topology(self, ts, sample, haplotypes=False):
        # copies from test_highlevel.py
        new_ts, node_map = ts.simplify(sample, map_nodes=True)
        old_trees = ts.trees()
        old_tree = next(old_trees)
        assert ts.get_num_trees() >= new_ts.get_num_trees()
        for new_tree in new_ts.trees():
            new_left, new_right = new_tree.get_interval()
            old_left, old_right = old_tree.get_interval()
            # Skip ahead on the old tree until new_left is within its interval
            while old_right <= new_left:
                old_tree = next(old_trees)
                old_left, old_right = old_tree.get_interval()
            # If the TMRCA of all pairs of samples is the same, then we have the
            # same information. We limit this to at most 500 pairs
            pairs = itertools.islice(itertools.combinations(sample, 2), 500)
            for pair in pairs:
                mapped_pair = [node_map[u] for u in pair]
                mrca1 = old_tree.get_mrca(*pair)
                mrca2 = new_tree.get_mrca(*mapped_pair)
                assert mrca2 == node_map[mrca1]
        if haplotypes:
            orig_haps = list(ts.haplotypes())
            simp_haps = list(new_ts.haplotypes())
            for i, j in enumerate(sample):
                assert orig_haps[j] == simp_haps[i]

    def test_partial_non_sample_external_nodes(self):
        # A somewhat more complicated test case with a partially specified,
        # non-sampled tip.
        #
        # Here is the situation:
        #
        # 1.0             7
        # 0.7            / \                                            6
        #               /   \                                          / \
        # 0.5          /     5                      5                 /   5
        #             /     / \                    / \               /   / \
        # 0.4        /     /   4                  /   4             /   /   4
        #           /     /   / \                /   / \           /   /   / \
        #          /     /   3   \              /   /   \         /   /   3   \
        #         /     /         \            /   /     \       /   /         \
        # 0.0    0     1           2          1   0       2     0   1           2
        #
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)

        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           0.2  # Non sample leaf
        4       0           0.4
        5       0           0.5
        6       0           0.7
        7       0           1.0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.0     0.2     7       0,5
        """
        )
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 7, 6: -1, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: -1, 4: 5, 5: -1, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1},
        ]
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tree_dicts = [t.parent_dict for t in ts.trees()]
        assert ts.sample_size == 3
        assert ts.num_trees == 3
        assert ts.num_nodes == 8
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        # check .simplify() works here
        self.verify_simplify_topology(ts, [0, 1, 2])

    def test_partial_non_sample_external_nodes_2(self):
        # The same situation as above, but partial tip is labeled '7' not '3':
        #
        # 1.0          6
        # 0.7         / \                                       5
        #            /   \                                     / \
        # 0.5       /     4                 4                 /   4
        #          /     / \               / \               /   / \
        # 0.4     /     /   3             /   3             /   /   3
        #        /     /   / \           /   / \           /   /   / \
        #       /     /   7   \         /   /   \         /   /   7   \
        #      /     /         \       /   /     \       /   /         \
        # 0.0 0     1           2     1   0       2     0   1           2
        #
        #          (0.0, 0.2),         (0.2, 0.8),         (0.8, 1.0)
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           0.4
        4       0           0.5
        5       0           0.7
        6       0           1.0
        7       0           0    # Non sample leaf
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     0.2     3       2,7
        0.2     0.8     3       0,2
        0.8     1.0     3       2,7
        0.0     0.2     4       1,3
        0.2     0.8     4       1,3
        0.8     1.0     4       1,3
        0.8     1.0     5       0,4
        0.0     0.2     6       0,4
        """
        )
        true_trees = [
            {0: 6, 1: 4, 2: 3, 3: 4, 4: 6, 5: -1, 6: -1, 7: 3},
            {0: 3, 1: 4, 2: 3, 3: 4, 4: -1, 5: -1, 6: -1, 7: -1},
            {0: 5, 1: 4, 2: 3, 3: 4, 4: 5, 5: -1, 6: -1, 7: 3},
        ]
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tree_dicts = [t.parent_dict for t in ts.trees()]
        # sample size check works here since 7 > 3
        assert ts.sample_size == 3
        assert ts.num_trees == 3
        assert ts.num_nodes == 8
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        self.verify_simplify_topology(ts, [0, 1, 2])

    def test_single_offspring_records(self):
        # Here we have inserted a single-offspring record
        # (for 6 on the left segment):
        #
        # 1.0             7
        # 0.7            / 6                                                  6
        #               /   \                                                / \
        # 0.5          /     5                       5                      /   5
        #             /     / \                     / \                    /   / \
        # 0.4        /     /   4                   /   4                  /   /   4
        # 0.3       /     /   / \                 /   / \                /   /   / \
        #          /     /   3   \               /   /   \              /   /   3   \
        #         /     /         \             /   /     \            /   /         \
        # 0.0    0     1           2           1   0       2          0   1           2
        #
        #          (0.0, 0.2),               (0.2, 0.8),              (0.8, 1.0)
        nodes = io.StringIO(
            """\
        id  is_sample   time
        0   1           0
        1   1           0
        2   1           0
        3   0           0       # Non sample leaf
        4   0           0.4
        5   0           0.5
        6   0           0.7
        7   0           1.0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.0     0.2     6       5
        0.0     0.2     7       0,6
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: 7, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: -1, 4: 5, 5: -1, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1},
        ]
        tree_dicts = [t.parent_dict for t in ts.trees()]
        assert ts.sample_size == 3
        assert ts.num_trees == 3
        assert ts.num_nodes == 8
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        self.verify_simplify_topology(ts, [0, 1, 2])

    def test_many_single_offspring(self):
        # a more complex test with single offspring
        # With `(i,j,x)->k` denoting that individual `k` inherits from `i` on `[0,x)`
        #    and from `j` on `[x,1)`:
        # 1. Begin with an individual `3` (and another anonymous one) at `t=0`.
        # 2. `(3,?,1.0)->4` and `(3,?,1.0)->5` at `t=1`
        # 3. `(4,3,0.9)->6` and `(3,5,0.1)->7` and then `3` dies at `t=2`
        # 4. `(6,7,0.7)->8` at `t=3`
        # 5. `(8,6,0.8)->9` and `(7,8,0.2)->10` at `t=4`.
        # 6. `(3,9,0.6)->0` and `(9,10,0.5)->1` and `(10,4,0.4)->2` at `t=5`.
        # 7. We sample `0`, `1`, and `2`.
        # Here are the trees:
        # t                  |              |              |             |
        #
        # 0       --3--      |     --3--    |     --3--    |    --3--    |    --3--
        #        /  |  \     |    /  |  \   |    /     \   |   /     \   |   /     \
        # 1     4   |   5    |   4   *   5  |   4       5  |  4       5  |  4       5
        #       |\ / \ /|    |   |\   \     |   |\     /   |  |\     /   |  |\     /|
        # 2     | 6   7 |    |   | 6   7    |   | 6   7    |  | 6   7    |  | 6   7 |
        #       | |\ /| |    |   |  \  *    |   |  \  |    |  |  *       |  |  *    | ...
        # 3     | | 8 | |    |   |   8 |    |   *   8 *    |  |   8      |  |   8   |
        #       | |/ \| |    |   |  /  |    |   |  /  |    |  |  * *     |  |  / \  |
        # 4     | 9  10 |    |   | 9  10    |   | 9  10    |  | 9  10    |  | 9  10 |
        #       |/ \ / \|    |   |  \   *   |   |  \   \   |  |  \   *   |  |  \    |
        # 5     0   1   2    |   0   1   2  |   0   1   2  |  0   1   2  |  0   1   2
        #
        #                    |   0.0 - 0.1  |   0.1 - 0.2  |  0.2 - 0.4  |  0.4 - 0.5
        # ... continued:
        # t                  |             |             |             |
        #
        # 0         --3--    |    --3--    |    --3--    |    --3--    |    --3--
        #          /     \   |   /     \   |   /     \   |   /     \   |   /  |  \
        # 1       4       5  |  4       5  |  4       5  |  4       5  |  4   |   5
        #         |\     /|  |   \     /|  |   \     /|  |   \     /|  |     /   /|
        # 2       | 6   7 |  |    6   7 |  |    6   7 |  |    6   7 |  |    6   7 |
        #         |  \    |  |     \    |  |       /  |  |    |  /  |  |    |  /  |
        # 3  ...  |   8   |  |      8   |  |      8   |  |    | 8   |  |    | 8   |
        #         |  / \  |  |     / \  |  |     / \  |  |    |  \  |  |    |  \  |
        # 4       | 9  10 |  |    9  10 |  |    9  10 |  |    9  10 |  |    9  10 |
        #         |    /  |  |   /   /  |  |   /   /  |  |   /   /  |  |   /   /  |
        # 5       0   1   2  |  0   1   2  |  0   1   2  |  0   1   2  |  0   1   2
        #
        #         0.5 - 0.6  |  0.6 - 0.7  |  0.7 - 0.8  |  0.8 - 0.9  |  0.9 - 1.0

        true_trees = [
            {0: 4, 1: 9, 2: 10, 3: -1, 4: 3, 5: 3, 6: 4, 7: 3, 8: 6, 9: 8, 10: 7},
            {0: 4, 1: 9, 2: 10, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 7},
            {0: 4, 1: 9, 2: 10, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 8},
            {0: 4, 1: 9, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 8},
            {0: 4, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 8},
            {0: 9, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 8},
            {0: 9, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 7, 9: 8, 10: 8},
            {0: 9, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 4, 7: 5, 8: 7, 9: 6, 10: 8},
            {0: 9, 1: 10, 2: 5, 3: -1, 4: 3, 5: 3, 6: 3, 7: 5, 8: 7, 9: 6, 10: 8},
        ]
        true_haplotypes = ["0100", "0001", "1110"]
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           5
        4       0           4
        5       0           4
        6       0           3
        7       0           3
        8       0           2
        9       0           1
        10      0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.5     1.0     10      1
        0.0     0.4     10      2
        0.6     1.0     9       0
        0.0     0.5     9       1
        0.8     1.0     8       10
        0.2     0.8     8       9,10
        0.0     0.2     8       9
        0.7     1.0     7       8
        0.0     0.2     7       10
        0.8     1.0     6       9
        0.0     0.7     6       8
        0.4     1.0     5       2,7
        0.1     0.4     5       7
        0.6     0.9     4       6
        0.0     0.6     4       0,6
        0.9     1.0     3       4,5,6
        0.1     0.9     3       4,5
        0.0     0.1     3       4,5,7
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.05        0
        0.15        0
        0.25        0
        0.4         0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state   parent
        0       7       1               -1
        0      10       0               0
        0       2       1               1
        1       0       1               -1
        1      10       1               -1
        2       8       1               -1
        2       9       0               5
        2      10       0               5
        2       2       1               7
        3       8       1               -1
        """
        )
        ts = tskit.load_text(nodes, edges, sites, mutations, strict=False)
        tree_dicts = [t.parent_dict for t in ts.trees()]
        assert ts.sample_size == 3
        assert ts.num_trees == len(true_trees)
        assert ts.num_nodes == 11
        assert len(list(ts.edge_diffs())) == ts.num_trees
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        for j, x in enumerate(ts.haplotypes()):
            assert x == true_haplotypes[j]
        self.verify_simplify_topology(ts, [0, 1, 2], haplotypes=True)
        self.verify_simplify_topology(ts, [1, 0, 2], haplotypes=True)
        self.verify_simplify_topology(ts, [0, 1], haplotypes=False)
        self.verify_simplify_topology(ts, [1, 2], haplotypes=False)
        self.verify_simplify_topology(ts, [2, 0], haplotypes=False)

    def test_tricky_switches(self):
        # suppose the topology has:
        # left right parent child
        #  0.0   0.5      6      0,1
        #  0.5   1.0      6      4,5
        #  0.0   0.4      7      2,3
        #
        # --------------------------
        #
        #        12         .        12         .        12         .
        #       /  \        .       /  \        .       /  \        .
        #     11    \       .      /    \       .      /    \       .
        #     / \    \      .     /     10      .     /     10      .
        #    /   \    \     .    /     /  \     .    /     /  \     .
        #   6     7    8    .   6     9    8    .   6     9    8    .
        #  / \   / \   /\   .  / \   / \   /\   .  / \   / \   /\   .
        # 0   1 2   3 4  5  . 0   1 2   3 4  5  . 4   5 2   3 0  1  .
        #                   .                   .                   .
        # 0.0              0.4                 0.5                 1.0
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       1           0
        5       1           0
        6       0           1
        7       0           1
        8       0           1
        9       0           1
        10      0           2
        11      0           3
        12      0           4
        """
        )
        edges = io.StringIO(
            """\
        left right parent child
        0.0  0.5   6      0
        0.0  0.5   6      1
        0.5  1.0   6      4
        0.5  1.0   6      5
        0.0  0.4   7      2,3
        0.5  1.0   8      0
        0.5  1.0   8      1
        0.0  0.5   8      4
        0.0  0.5   8      5
        0.4  1.0   9      2,3
        0.4  1.0   10     8,9
        0.0  0.4   11     6,7
        0.4  1.0   12     6
        0.0  0.4   12     8
        0.4  1.0   12     10
        0.0  0.4   12     11
        """
        )
        true_trees = [
            {
                0: 6,
                1: 6,
                2: 7,
                3: 7,
                4: 8,
                5: 8,
                6: 11,
                7: 11,
                8: 12,
                9: -1,
                10: -1,
                11: 12,
                12: -1,
            },
            {
                0: 6,
                1: 6,
                2: 9,
                3: 9,
                4: 8,
                5: 8,
                6: 12,
                7: -1,
                8: 10,
                9: 10,
                10: 12,
                11: -1,
                12: -1,
            },
            {
                0: 8,
                1: 8,
                2: 9,
                3: 9,
                4: 6,
                5: 6,
                6: 12,
                7: -1,
                8: 10,
                9: 10,
                10: 12,
                11: -1,
                12: -1,
            },
        ]
        ts = tskit.load_text(nodes, edges, strict=False)
        tree_dicts = [t.parent_dict for t in ts.trees()]
        assert ts.sample_size == 6
        assert ts.num_trees == len(true_trees)
        assert ts.num_nodes == 13
        assert len(list(ts.edge_diffs())) == ts.num_trees
        # check topologies agree:
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        self.verify_simplify_topology(ts, [0, 2])
        self.verify_simplify_topology(ts, [0, 4])
        self.verify_simplify_topology(ts, [2, 4])

    def test_tricky_simplify(self):
        # Continue as above but invoke simplfy:
        #
        #         12         .          12         .
        #        /  \        .         /  \        .
        #      11    \       .       11    \       .
        #      / \    \      .       / \    \      .
        #    13   \    \     .      /  15    \     .
        #    / \   \    \    .     /   / \    \    .
        #   6  14   7    8   .    6  14   7    8   .
        #  / \     / \   /\  .   / \     / \   /\  .
        # 0   1   2   3 4  5 .  0   1   2   3 4  5 .
        #                    .                     .
        # 0.0               0.1                   0.4
        #
        #  .        12         .        12         .
        #  .       /  \        .       /  \        .
        #  .      /    \       .      /    \       .
        #  .     /     10      .     /     10      .
        #  .    /     /  \     .    /     /  \     .
        #  .   6     9    8    .   6     9    8    .
        #  .  / \   / \   /\   .  / \   / \   /\   .
        #  . 0   1 2   3 4  5  . 4   5 2   3 0  1  .
        #  .                   .                   .
        # 0.4                 0.5                 1.0
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       1           0
        5       1           0
        6       0           1
        7       0           1
        8       0           1
        9       0           1
        10      0           2
        11      0           3
        12      0           4
        13      0           2
        14      0           1
        15      0           2
        """
        )
        edges = io.StringIO(
            """\
        left right parent child
        0.0  0.5   6      0,1
        0.5  1.0   6      4,5
        0.0  0.4   7      2,3
        0.0  0.5   8      4,5
        0.5  1.0   8      0,1
        0.4  1.0   9      2,3
        0.4  1.0   10     8,9
        0.0  0.1   13     6,14
        0.1  0.4   15     7,14
        0.0  0.1   11     7,13
        0.1  0.4   11     6,15
        0.0  0.4   12     8,11
        0.4  1.0   12     6,10
        """
        )
        true_trees = [
            {
                0: 6,
                1: 6,
                2: 7,
                3: 7,
                4: 8,
                5: 8,
                6: 11,
                7: 11,
                8: 12,
                9: -1,
                10: -1,
                11: 12,
                12: -1,
            },
            {
                0: 6,
                1: 6,
                2: 9,
                3: 9,
                4: 8,
                5: 8,
                6: 12,
                7: -1,
                8: 10,
                9: 10,
                10: 12,
                11: -1,
                12: -1,
            },
            {
                0: 8,
                1: 8,
                2: 9,
                3: 9,
                4: 6,
                5: 6,
                6: 12,
                7: -1,
                8: 10,
                9: 10,
                10: 12,
                11: -1,
                12: -1,
            },
        ]
        big_ts = tskit.load_text(nodes, edges, strict=False)
        assert big_ts.num_trees == 1 + len(true_trees)
        assert big_ts.num_nodes == 16
        ts, node_map = big_ts.simplify(map_nodes=True)
        assert list(node_map[:6]) == list(range(6))
        assert ts.sample_size == 6
        assert ts.num_nodes == 13

    def test_ancestral_samples(self):
        # Check that specifying samples to be not at time 0.0 works.
        #
        # 1.0             7
        # 0.7            / \                      8                     6
        #               /   \                    / \                   / \
        # 0.5          /     5                  /   5                 /   5
        #             /     / \                /   / \               /   / \
        # 0.4        /     /   4              /   /   4             /   /   4
        #           /     /   / \            /   /   / \           /   /   / \
        # 0.2      /     /   3   \          3   /   /   \         /   /   3   \
        #         /     /    *    \         *  /   /     \       /   /    *    \
        # 0.0    0     1           2          1   0       2     0   1           2
        #              *           *          *           *         *           *
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)
        #
        # Simplified, keeping [1,2,3]
        #
        # 1.0
        # 0.7                                     5
        #                                        / \
        # 0.5                4                  /   4                     4
        #                   / \                /   / \                   / \
        # 0.4              /   3              /   /   3                 /   3
        #                 /   / \            /   /     \               /   / \
        # 0.2            /   2   \          2   /       \             /   2   \
        #               /    *    \         *  /         \           /    *    \
        # 0.0          0           1          0           1         0           1
        #              *           *          *           *         *           *
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)

        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       0           0
        1       1           0
        2       1           0
        3       1           0.2
        4       0           0.4
        5       0           0.5
        6       0           0.7
        7       0           1.0
        8       0           0.8
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.2     0.8     8       3,5
        0.0     0.2     7       0,5
        """
        )
        first_ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        ts, node_map = first_ts.simplify(map_nodes=True)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 7, 6: -1, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: 8, 4: 5, 5: 8, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1},
        ]
        # maps [1,2,3] -> [0,1,2]
        assert node_map[1] == 0
        assert node_map[2] == 1
        assert node_map[3] == 2
        true_simplified_trees = [
            {0: 4, 1: 3, 2: 3, 3: 4},
            {0: 4, 1: 4, 2: 5, 4: 5},
            {0: 4, 1: 3, 2: 3, 3: 4},
        ]
        assert first_ts.sample_size == 3
        assert ts.sample_size == 3
        assert first_ts.num_trees == 3
        assert ts.num_trees == 3
        assert first_ts.num_nodes == 9
        assert ts.num_nodes == 6
        assert first_ts.node(3).time == 0.2
        assert ts.node(2).time == 0.2
        # check topologies agree:
        tree_dicts = [t.parent_dict for t in first_ts.trees()]
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        tree_simplified_dicts = [t.parent_dict for t in ts.trees()]
        for a, t in zip(true_simplified_trees, tree_simplified_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        # check .simplify() works here
        self.verify_simplify_topology(first_ts, [1, 2, 3])

    def test_all_ancestral_samples(self):
        # Check that specifying samples all to be not at time 0.0 works.
        #
        # 1.0             7
        # 0.7            / \                      8                     6
        #               /   \                    / \                   / \
        # 0.5          /     5                  /   5                 /   5
        #             /     / \                /   / \               /   / \
        # 0.4        /     /   4              /   /   4             /   /   4
        #           /     /   / \            /   /   / \           /   /   / \
        # 0.2      /     /   3   \          3   /   /   \         /   /   3   \
        #         /     1    *    2         *  1   /     2       /   1    *    2
        # 0.0    0      *         *            *  0      *      0    *         *
        #
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)

        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       0           0
        1       1           0.1
        2       1           0.1
        3       1           0.2
        4       0           0.4
        5       0           0.5
        6       0           0.7
        7       0           1.0
        8       0           0.8
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.2     0.8     8       3,5
        0.0     0.2     7       0,5
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 7, 6: -1, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: 8, 4: 5, 5: 8, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1},
        ]
        assert ts.sample_size == 3
        assert ts.num_trees == 3
        assert ts.num_nodes == 9
        assert ts.node(0).time == 0.0
        assert ts.node(1).time == 0.1
        assert ts.node(2).time == 0.1
        assert ts.node(3).time == 0.2
        # check topologies agree:
        tree_dicts = [t.parent_dict for t in ts.trees()]
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        # check .simplify() works here
        self.verify_simplify_topology(ts, [1, 2, 3])

    def test_internal_sampled_node(self):
        # 1.0             7
        # 0.7            / \                      8                     6
        #               /   \                    / \                   / \
        # 0.5          /     5                  /   5                 /   5
        #             /     /*\                /   /*\               /   /*\
        # 0.4        /     /   4              /   /   4             /   /   4
        #           /     /   / \            /   /   / \           /   /   / \
        # 0.2      /     /   3   \          3   /   /   \         /   /   3   \
        #         /     1    *    2         *  1   /     2       /   1    *    2
        # 0.0    0      *         *            *  0      *      0    *         *
        #
        #          (0.0, 0.2),                 (0.2, 0.8),         (0.8, 1.0)
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       0           0
        1       1           0.1
        2       1           0.1
        3       1           0.2
        4       0           0.4
        5       1           0.5
        6       0           0.7
        7       0           1.0
        8       0           0.8
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     0.2     4       2,3
        0.2     0.8     4       0,2
        0.8     1.0     4       2,3
        0.0     1.0     5       1,4
        0.8     1.0     6       0,5
        0.2     0.8     8       3,5
        0.0     0.2     7       0,5
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        true_trees = [
            {0: 7, 1: 5, 2: 4, 3: 4, 4: 5, 5: 7, 6: -1, 7: -1},
            {0: 4, 1: 5, 2: 4, 3: 8, 4: 5, 5: 8, 6: -1, 7: -1},
            {0: 6, 1: 5, 2: 4, 3: 4, 4: 5, 5: 6, 6: -1, 7: -1},
        ]
        assert ts.sample_size == 4
        assert ts.num_trees == 3
        assert ts.num_nodes == 9
        assert ts.node(0).time == 0.0
        assert ts.node(1).time == 0.1
        assert ts.node(2).time == 0.1
        assert ts.node(3).time == 0.2
        # check topologies agree:
        tree_dicts = [t.parent_dict for t in ts.trees()]
        for a, t in zip(true_trees, tree_dicts):
            for k in a.keys():
                if k in t.keys():
                    assert t[k] == a[k]
                else:
                    assert a[k] == tskit.NULL
        # check .simplify() works here
        self.verify_simplify_topology(ts, [1, 2, 3])
        self.check_num_samples(
            ts,
            [
                (0, 5, 4),
                (0, 2, 1),
                (0, 7, 4),
                (0, 4, 2),
                (1, 4, 1),
                (1, 5, 3),
                (1, 8, 4),
                (1, 0, 0),
                (2, 5, 4),
                (2, 1, 1),
            ],
        )
        self.check_num_tracked_samples(
            ts,
            [1, 2, 5],
            [
                (0, 5, 3),
                (0, 2, 1),
                (0, 7, 3),
                (0, 4, 1),
                (1, 4, 1),
                (1, 5, 3),
                (1, 8, 3),
                (1, 0, 0),
                (2, 5, 3),
                (2, 1, 1),
            ],
        )
        self.check_sample_iterator(
            ts,
            [
                (0, 0, []),
                (0, 5, [5, 1, 2, 3]),
                (0, 4, [2, 3]),
                (1, 5, [5, 1, 2]),
                (2, 4, [2, 3]),
            ],
        )
        # pedantically check the Tree methods on the second tree
        tst = ts.trees()
        t = next(tst)
        t = next(tst)
        assert t.branch_length(1) == 0.4
        assert not t.is_internal(0)
        assert t.is_leaf(0)
        assert not t.is_sample(0)
        assert not t.is_internal(1)
        assert t.is_leaf(1)
        assert t.is_sample(1)
        assert t.is_internal(5)
        assert not t.is_leaf(5)
        assert t.is_sample(5)
        assert t.is_internal(4)
        assert not t.is_leaf(4)
        assert not t.is_sample(4)
        assert t.root == 8
        assert t.mrca(0, 1) == 5
        assert t.sample_size == 4


class TestBadTrees:
    """
    Tests for bad tree sequence topologies that can only be detected when we
    try to create trees.
    """

    def test_simplest_contradictory_children(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     1.0     2       0
        0.0     1.0     3       0
        """
        )
        with pytest.raises(_tskit.LibraryError):
            tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_partial_overlap_contradictory_children(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        3       0           2
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0     1.0     2       0,1
        0.5     1.0     3       0
        """
        )
        with pytest.raises(_tskit.LibraryError):
            tskit.load_text(nodes=nodes, edges=edges, strict=False)


class TestCoiteration:
    """
    Test ability to iterate over multiple (currently 2) tree sequences simultaneously
    """

    def test_identical_ts(self):
        ts = msprime.simulate(4, recombination_rate=1, random_seed=123)
        assert ts.num_trees > 1
        total_iterations = 0
        for tree, (_, t1, t2) in zip(ts.trees(), ts.coiterate(ts)):
            total_iterations += 1
            assert tree == t1 == t2
        assert ts.num_trees == total_iterations

    def test_intervals(self):
        ts1 = msprime.simulate(4, recombination_rate=1, random_seed=1)
        assert ts1.num_trees > 1
        one_tree_ts = msprime.simulate(5, random_seed=2)
        multi_tree_ts = msprime.simulate(5, recombination_rate=1, random_seed=2)
        assert multi_tree_ts.num_trees > 1
        for ts2 in (one_tree_ts, multi_tree_ts):
            bp1 = set(ts1.breakpoints())
            bp2 = set(ts2.breakpoints())
            assert bp1 != bp2
            breaks = set()
            for interval, t1, t2 in ts1.coiterate(ts2):
                assert set(interval) <= set(t1.interval) | set(t2.interval)
                breaks.add(interval.left)
                breaks.add(interval.right)
                assert t1.tree_sequence == ts1
                assert t2.tree_sequence == ts2
            assert breaks == bp1 | bp2

    def test_simple_ts(self):
        nodes = """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        """
        edges1 = """\
        left    right   parent  child
        0       0.2       3       0,1
        0       0.2       4       2,3
        0.2     1         3       2,1
        0.2     1         4       0,3
        """
        edges2 = """\
        left    right   parent  child
        0       0.8       3       2,1
        0       0.8       4       0,3
        0.8     1         3       0,1
        0.8     1         4       2,3
        """
        ts1 = tskit.load_text(io.StringIO(nodes), io.StringIO(edges1), strict=False)
        ts2 = tskit.load_text(io.StringIO(nodes), io.StringIO(edges2), strict=False)
        coiterator = ts1.coiterate(ts2)
        interval, tree1, tree2 = next(coiterator)
        assert interval.left == 0
        assert interval.right == 0.2
        assert tree1 == ts1.at_index(0)
        assert tree2 == ts2.at_index(0)
        interval, tree1, tree2 = next(coiterator)
        assert interval.left == 0.2
        assert interval.right == 0.8
        assert tree1 == ts1.at_index(1)
        assert tree2 == ts2.at_index(0)
        interval, tree1, tree2 = next(coiterator)
        assert interval.left == 0.8
        assert interval.right == 1
        assert tree1 == ts1.at_index(1)
        assert tree2 == ts2.at_index(1)

    def test_nonequal_lengths(self):
        ts1 = msprime.simulate(4, random_seed=1, length=2)
        ts2 = msprime.simulate(4, random_seed=1)
        with pytest.raises(ValueError, match="equal sequence length"):
            next(ts1.coiterate(ts2))

    def test_kwargs(self):
        ts = msprime.simulate(4, recombination_rate=1, random_seed=123)
        for _, t1, t2 in ts.coiterate(ts):
            assert t1.num_tracked_samples() == t2.num_tracked_samples() == 0
        for _, t1, t2 in ts.coiterate(ts, tracked_samples=ts.samples()):
            assert t1.num_tracked_samples() == t2.num_tracked_samples() == 4


def do_simplify(
    ts,
    samples=None,
    compare_lib=True,
    filter_sites=True,
    filter_populations=True,
    filter_individuals=True,
    filter_nodes=True,
    keep_unary=False,
    keep_input_roots=False,
    update_sample_flags=True,
):
    """
    Runs the Python test implementation of simplify.
    """
    if samples is None:
        samples = ts.samples()
    s = tests.Simplifier(
        ts,
        samples,
        filter_sites=filter_sites,
        filter_populations=filter_populations,
        filter_individuals=filter_individuals,
        filter_nodes=filter_nodes,
        keep_unary=keep_unary,
        keep_input_roots=keep_input_roots,
        update_sample_flags=update_sample_flags,
    )
    new_ts, node_map = s.simplify()
    if compare_lib:
        sts, lib_node_map1 = ts.simplify(
            samples,
            filter_sites=filter_sites,
            filter_individuals=filter_individuals,
            filter_populations=filter_populations,
            filter_nodes=filter_nodes,
            update_sample_flags=update_sample_flags,
            keep_unary=keep_unary,
            keep_input_roots=keep_input_roots,
            map_nodes=True,
        )
        lib_tables1 = sts.dump_tables()

        py_tables = new_ts.dump_tables()
        # Compare all tables except mutations
        py_tables_no_mut = py_tables.copy()
        lib_tables1_no_mut = lib_tables1.copy()
        py_tables_no_mut.mutations.clear()
        lib_tables1_no_mut.mutations.clear()
        py_tables_no_mut.assert_equals(lib_tables1_no_mut, ignore_provenance=True)

        # For mutations, check functional equivalence by comparing mutation properties
        # but handling parent relationships that may differ due to reordering
        def normalize_time(time):
            return -42.0 if tskit.is_unknown_time(time) else time

        def mutation_signature(m, mutations):
            # Create a signature that identifies a mutation by its properties
            # and its parent's properties (to handle parent ID remapping)
            def make_hashable(metadata):
                # Convert unhashable metadata (like dicts) to hashable form
                if isinstance(metadata, dict):
                    return tuple(sorted(metadata.items()))
                elif isinstance(metadata, list):
                    return tuple(metadata)
                else:
                    return metadata

            parent_sig = None
            if m.parent != -1 and m.parent < len(mutations):
                parent = mutations[m.parent]
                parent_sig = (
                    parent.site,
                    parent.node,
                    parent.derived_state,
                    make_hashable(parent.metadata),
                    normalize_time(parent.time),
                )
            return (
                m.site,
                m.node,
                m.derived_state,
                make_hashable(m.metadata),
                normalize_time(m.time),
                parent_sig,
            )

        py_mut_sigs = {
            mutation_signature(m, py_tables.mutations) for m in py_tables.mutations
        }
        lib_mut_sigs = {
            mutation_signature(m, lib_tables1.mutations) for m in lib_tables1.mutations
        }

        assert py_mut_sigs == lib_mut_sigs
        assert all(node_map == lib_node_map1)
    return new_ts, node_map


class SimplifyTestBase:
    """
    Base class for simplify tests.
    """


class TestSimplify(SimplifyTestBase):
    """
    Tests that the implementations of simplify() do what they are supposed to.
    """

    random_seed = 23
    #
    #          8
    #         / \
    #        /   \
    #       /     \
    #      7       \
    #     / \       6
    #    /   5     / \
    #   /   / \   /   \
    #  4   0   1 2     3
    small_tree_ex_nodes = """\
    id      is_sample   population      time
    0       1       0               0.00000000000000
    1       1       0               0.00000000000000
    2       1       0               0.00000000000000
    3       1       0               0.00000000000000
    4       1       0               0.00000000000000
    5       0       0               0.14567111023387
    6       0       0               0.21385545626353
    7       0       0               0.43508024345063
    8       0       0               1.60156352971203
    """
    small_tree_ex_edges = """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      5       0,1
    1       0.00000000      1.00000000      6       2,3
    2       0.00000000      1.00000000      7       4,5
    3       0.00000000      1.00000000      8       6,7
    """

    def verify_no_samples(self, ts, keep_unary=False):
        """
        Zero out the flags column and verify that we get back the correct
        tree sequence when we run simplify.
        """
        t1 = ts.dump_tables()
        t1.nodes.flags = np.zeros_like(t1.nodes.flags)
        ts1, node_map1 = do_simplify(ts, samples=ts.samples(), keep_unary=keep_unary)
        t1 = ts1.dump_tables()
        ts2, node_map2 = do_simplify(ts, keep_unary=keep_unary)
        t2 = ts2.dump_tables()
        t1.assert_equals(t2)

    def verify_single_childified(self, ts, keep_unary=False):
        """
        Modify the specified tree sequence so that it has lots of unary
        nodes. Run simplify and verify we get the same tree sequence back
        if keep_unary is False. If keep_unary is True, the simplication
        won't do anything to the original treeSequence.
        """
        ts_single = tsutil.single_childify(ts)

        tss, node_map = do_simplify(ts_single, keep_unary=keep_unary)
        # All original nodes should still be present.
        for u in range(ts.num_samples):
            assert u == node_map[u]
        # All introduced nodes should be mapped to null.
        for u in range(ts.num_samples, ts_single.num_samples):
            assert node_map[u] == tskit.NULL
        t1 = ts.dump_tables()
        t2 = tss.dump_tables()
        t3 = ts_single.dump_tables()
        if keep_unary:
            assert set(t3.nodes.time) == set(t2.nodes.time)
            assert len(t3.edges) == len(t2.edges)
            assert t3.sites == t2.sites
            assert len(t3.mutations) == len(t2.mutations)
        else:
            assert t1.nodes == t2.nodes
            assert t1.edges == t2.edges
            assert t1.sites == t2.sites
            assert t1.mutations == t2.mutations

    def verify_multiroot_internal_samples(self, ts, keep_unary=False):
        ts_multiroot = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        ts1 = tsutil.jiggle_samples(ts_multiroot)
        ts2, node_map = do_simplify(ts1, keep_unary=keep_unary)
        assert ts1.num_trees >= ts2.num_trees
        trees2 = ts2.trees()
        t2 = next(trees2)
        for t1 in ts1.trees():
            assert t2.interval.left <= t1.interval.left
            assert t2.interval.right >= t1.interval.right
            pairs = itertools.combinations(ts1.samples(), 2)
            for pair in pairs:
                mapped_pair = [node_map[u] for u in pair]
                mrca1 = t1.get_mrca(*pair)
                mrca2 = t2.get_mrca(*mapped_pair)
                if mrca1 == tskit.NULL:
                    assert mrca2 == tskit.NULL
                else:
                    assert node_map[mrca1] == mrca2
            if t2.interval.right == t1.interval.right:
                t2 = next(trees2, None)

    def test_single_tree(self):
        ts = msprime.simulate(10, random_seed=self.random_seed)
        self.verify_no_samples(ts)
        self.verify_single_childified(ts)
        self.verify_multiroot_internal_samples(ts)
        # Now with keep_unary=True.
        self.verify_no_samples(ts, keep_unary=True)
        self.verify_single_childified(ts, keep_unary=True)
        self.verify_multiroot_internal_samples(ts, keep_unary=True)

    def test_single_tree_mutations(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=self.random_seed)
        assert ts.num_sites > 1
        do_simplify(ts)
        self.verify_single_childified(ts)
        # Also with keep_unary == True.
        do_simplify(ts, keep_unary=True)
        self.verify_single_childified(ts, keep_unary=True)

    def test_many_trees_mutations(self):
        ts = msprime.simulate(
            10, recombination_rate=1, mutation_rate=10, random_seed=self.random_seed
        )
        assert ts.num_trees > 2
        assert ts.num_sites > 2
        self.verify_no_samples(ts)
        do_simplify(ts)
        self.verify_single_childified(ts)
        # Also with keep_unary == True.
        do_simplify(ts, keep_unary=True)
        self.verify_single_childified(ts, keep_unary=True)

    def test_many_trees(self):
        ts = msprime.simulate(5, recombination_rate=4, random_seed=self.random_seed)
        assert ts.num_trees > 2
        self.verify_no_samples(ts)
        self.verify_single_childified(ts)
        self.verify_multiroot_internal_samples(ts)
        # Also with keep_unary == True.
        self.verify_no_samples(ts, keep_unary=True)
        self.verify_single_childified(ts, keep_unary=True)
        self.verify_multiroot_internal_samples(ts, keep_unary=True)

    def test_small_tree_internal_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # The parent of samples 0 and 1 is 5. Change this to an internal sample
        # and set 0 and 1 to be unsampled.
        flags[0] = 0
        flags[0] = 0
        flags[5] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        ts = tables.tree_sequence()
        assert ts.sample_size == 5
        tss, node_map = do_simplify(ts, [3, 5])
        assert node_map[3] == 0
        assert node_map[5] == 1
        assert tss.num_nodes == 3
        assert tss.num_edges == 2
        self.verify_no_samples(ts)
        # with keep_unary == True
        tss, node_map = do_simplify(ts, [3, 5], keep_unary=True)
        assert node_map[3] == 0
        assert node_map[5] == 1
        assert tss.num_nodes == 5
        assert tss.num_edges == 4
        self.verify_no_samples(ts, keep_unary=True)

    def test_small_tree_linear_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # 7 is above 0. These are the only two samples
        flags[:] = 0
        flags[0] = tskit.NODE_IS_SAMPLE
        flags[7] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        ts = tables.tree_sequence()
        assert ts.sample_size == 2
        tss, node_map = do_simplify(ts, [0, 7])
        assert node_map[0] == 0
        assert node_map[7] == 1
        assert tss.num_nodes == 2
        assert tss.num_edges == 1
        t = next(tss.trees())
        assert t.parent_dict == {0: 1}
        # with keep_unary == True
        tss, node_map = do_simplify(ts, [0, 7], keep_unary=True)
        assert node_map[0] == 0
        assert node_map[7] == 1
        assert tss.num_nodes == 4
        assert tss.num_edges == 3
        t = next(tss.trees())

    def test_small_tree_internal_and_external_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # 7 is above 0 and 1.
        flags[:] = 0
        flags[0] = tskit.NODE_IS_SAMPLE
        flags[1] = tskit.NODE_IS_SAMPLE
        flags[7] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        ts = tables.tree_sequence()
        assert ts.sample_size == 3
        tss, node_map = do_simplify(ts, [0, 1, 7])
        assert node_map[0] == 0
        assert node_map[1] == 1
        assert node_map[7] == 2
        assert tss.num_nodes == 4
        assert tss.num_edges == 3
        t = next(tss.trees())
        assert t.parent_dict == {0: 3, 1: 3, 3: 2}
        # with keep_unary == True
        tss, node_map = do_simplify(ts, [0, 1, 7], keep_unary=True)
        assert node_map[0] == 0
        assert node_map[1] == 1
        assert node_map[7] == 2
        assert tss.num_nodes == 5
        assert tss.num_edges == 4
        t = next(tss.trees())
        assert t.parent_dict == {0: 3, 1: 3, 3: 2, 2: 4}

    def test_small_tree_mutations(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        tables = ts.dump_tables()
        # Add some simple mutations here above the nodes we're keeping.
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.sites.add_row(position=0.75, ancestral_state="0")
        tables.sites.add_row(position=0.8, ancestral_state="0")
        tables.mutations.add_row(site=0, node=0, derived_state="1")
        tables.mutations.add_row(site=1, node=2, derived_state="1")
        tables.mutations.add_row(site=2, node=7, derived_state="1")
        tables.mutations.add_row(site=3, node=0, derived_state="1")
        ts = tables.tree_sequence()
        assert ts.num_sites == 4
        assert ts.num_mutations == 4
        for keep in [True, False]:
            tss = do_simplify(ts, [0, 2], keep_unary=keep)[0]
            assert tss.sample_size == 2
            assert tss.num_mutations == 4
            assert list(tss.haplotypes()) == ["1011", "0100"]

    def test_small_tree_filter_zero_mutations(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        ts = tsutil.insert_branch_sites(ts)
        assert ts.num_sites == 8
        assert ts.num_mutations == 8
        for keep in [True, False]:
            tss, _ = do_simplify(ts, [4, 0, 1], filter_sites=True, keep_unary=keep)
            assert tss.num_sites == 5
            assert tss.num_mutations == 5
            tss, _ = do_simplify(ts, [4, 0, 1], filter_sites=False, keep_unary=keep)
            assert tss.num_sites == 8
            assert tss.num_mutations == 5

    def test_small_tree_fixed_sites(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        tables = ts.dump_tables()
        # Add some simple mutations that will be fixed after simplify
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.sites.add_row(position=0.75, ancestral_state="0")
        tables.mutations.add_row(site=0, node=2, derived_state="1")
        tables.mutations.add_row(site=1, node=3, derived_state="1")
        tables.mutations.add_row(site=2, node=6, derived_state="1")
        ts = tables.tree_sequence()
        assert ts.num_sites == 3
        assert ts.num_mutations == 3
        for keep in [True, False]:
            tss, _ = do_simplify(ts, [4, 1], keep_unary=keep)
            assert tss.sample_size == 2
            assert tss.num_mutations == 0
            assert list(tss.haplotypes()) == ["", ""]

    def test_small_tree_mutations_over_root(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        tables = ts.dump_tables()
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.mutations.add_row(site=0, node=8, derived_state="1")
        ts = tables.tree_sequence()
        assert ts.num_sites == 1
        assert ts.num_mutations == 1
        for keep_unary, filter_sites in itertools.product([True, False], repeat=2):
            tss, _ = do_simplify(
                ts, [0, 1], filter_sites=filter_sites, keep_unary=keep_unary
            )
            assert tss.num_sites == 1
            assert tss.num_mutations == 1

    def test_small_tree_recurrent_mutations(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        tables = ts.dump_tables()
        # Add recurrent mutation on the root branches
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.mutations.add_row(site=0, node=6, derived_state="1")
        tables.mutations.add_row(site=0, node=7, derived_state="1")
        ts = tables.tree_sequence()
        assert ts.num_sites == 1
        assert ts.num_mutations == 2
        for keep in [True, False]:
            tss = do_simplify(ts, [4, 3], keep_unary=keep)[0]
            assert tss.sample_size == 2
            assert tss.num_sites == 1
            assert tss.num_mutations == 2
            assert list(tss.haplotypes()) == ["1", "1"]

    def test_small_tree_back_mutations(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        tables = ts.dump_tables()
        # Add a chain of mutations
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.mutations.add_row(site=0, node=7, derived_state="1")
        tables.mutations.add_row(site=0, node=5, derived_state="0")
        tables.mutations.add_row(site=0, node=1, derived_state="1")
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        assert ts.num_sites == 1
        assert ts.num_mutations == 3
        assert list(ts.haplotypes()) == ["0", "1", "0", "0", "1"]
        # First check if we simplify for all samples and keep original state.
        for keep in [True, False]:
            tss = do_simplify(ts, [0, 1, 2, 3, 4], keep_unary=keep)[0]
            assert tss.sample_size == 5
            assert tss.num_sites == 1
            assert tss.num_mutations == 3
            assert list(tss.haplotypes()) == ["0", "1", "0", "0", "1"]

        # The ancestral state above 5 should be 0.
        for keep in [True, False]:
            tss = do_simplify(ts, [0, 1], keep_unary=keep)[0]
            assert tss.sample_size == 2
            assert tss.num_sites == 1
            assert tss.num_mutations == 3
            assert list(tss.haplotypes()) == ["0", "1"]

        # The ancestral state above 7 should be 1.
        for keep in [True, False]:
            tss = do_simplify(ts, [4, 0, 1], keep_unary=keep)[0]
            assert tss.sample_size == 3
            assert tss.num_sites == 1
            assert tss.num_mutations == 3
            assert list(tss.haplotypes()) == ["1", "0", "1"]

    def test_overlapping_unary_edges(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       2       2       0
        1       3       2       1
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        assert ts.sample_size == 2
        assert ts.num_trees == 3
        assert ts.sequence_length == 3
        for keep in [True, False]:
            tss, node_map = do_simplify(ts, samples=[0, 1, 2], keep_unary=keep)
            assert list(node_map) == [0, 1, 2]
            trees = [{0: 2}, {0: 2, 1: 2}, {1: 2}]
            for t in tss.trees():
                assert t.parent_dict == trees[t.index]

    def test_overlapping_unary_edges_internal_samples(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       2       2       0
        1       3       2       1
        """
        )
        ts = tskit.load_text(nodes, edges, strict=False)
        assert ts.sample_size == 3
        assert ts.num_trees == 3
        trees = [{0: 2}, {0: 2, 1: 2}, {1: 2}]
        for t in ts.trees():
            assert t.parent_dict == trees[t.index]
        tss, node_map = do_simplify(ts)
        assert list(node_map) == [0, 1, 2]

    def test_isolated_samples(self):
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
        """
        )
        ts = tskit.load_text(nodes, edges, sequence_length=1, strict=False)
        assert ts.num_samples == 3
        assert ts.num_trees == 1
        assert ts.num_nodes == 3
        for keep in [True, False]:
            tss, node_map = do_simplify(ts, keep_unary=keep)
            assert ts.tables.nodes == tss.tables.nodes
            assert ts.tables.edges == tss.tables.edges
            assert list(node_map) == [0, 1, 2]

    def test_internal_samples(self):
        nodes = io.StringIO(
            """\
        id      is_sample   population      time
        0       1       -1              1.00000000000000
        1       0       -1              1.00000000000000
        2       1       -1              1.00000000000000
        3       0       -1              1.31203521181726
        4       0       -1              2.26776380586006
        5       1       -1              0.00000000000000
        6       0       -1              0.50000000000000
        7       0       -1              1.50000000000000

        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.62185118      1.00000000      1       6
        1       0.00000000      0.62185118      2       6
        2       0.00000000      1.00000000      3       0,2
        3       0.00000000      1.00000000      4       7,3
        4       0.00000000      1.00000000      6       5
        5       0.00000000      1.00000000      7       1
        """
        )

        ts = tskit.load_text(nodes, edges, strict=False)
        tss, node_map = do_simplify(ts, [5, 2, 0])
        assert node_map[0] == 2
        assert node_map[1] == -1
        assert node_map[2] == 1
        assert node_map[3] == 3
        assert node_map[4] == 4
        assert node_map[5] == 0
        assert node_map[6] == -1
        assert node_map[7] == -1
        assert tss.sample_size == 3
        assert tss.num_trees == 2
        trees = [{0: 1, 1: 3, 2: 3}, {0: 4, 1: 3, 2: 3, 3: 4}]
        for t in tss.trees():
            assert t.parent_dict == trees[t.index]
        # with keep_unary == True
        tss, node_map = do_simplify(ts, [5, 2, 0], keep_unary=True)
        assert node_map[0] == 2
        assert node_map[1] == 4
        assert node_map[2] == 1
        assert node_map[3] == 5
        assert node_map[4] == 7
        assert node_map[5] == 0
        assert node_map[6] == 3
        assert node_map[7] == 6
        assert tss.sample_size == 3
        assert tss.num_trees == 2
        trees = [
            {0: 3, 1: 5, 2: 5, 3: 1, 5: 7},
            {0: 3, 1: 5, 2: 5, 3: 4, 4: 6, 5: 7, 6: 7},
        ]
        for t in tss.trees():
            assert t.parent_dict == trees[t.index]

    def test_many_mutations_over_single_sample_ancestral_state(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       1       0
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0           0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state   parent
        0       0       1               -1
        0       0       0               0
        """
        )
        ts = tskit.load_text(
            nodes, edges, sites=sites, mutations=mutations, strict=False
        )
        assert ts.sample_size == 1
        assert ts.num_trees == 1
        assert ts.num_sites == 1
        assert ts.num_mutations == 2
        for keep in [True, False]:
            tss, node_map = do_simplify(ts, keep_unary=keep)
            assert tss.num_sites == 1
            assert tss.num_mutations == 2
            assert list(tss.haplotypes(isolated_as_missing=False)) == ["0"]

    def test_many_mutations_over_single_sample_derived_state(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       0           1
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0       1       1       0
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0           0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state   parent
        0       0       1               -1
        0       0       0               0
        0       0       1               1
        """
        )
        ts = tskit.load_text(
            nodes, edges, sites=sites, mutations=mutations, strict=False
        )
        assert ts.sample_size == 1
        assert ts.num_trees == 1
        assert ts.num_sites == 1
        assert ts.num_mutations == 3
        for keep in [True, False]:
            tss, node_map = do_simplify(ts, keep_unary=keep)
            assert tss.num_sites == 1
            assert tss.num_mutations == 3
            assert list(tss.haplotypes(isolated_as_missing=False)) == ["1"]

    def test_many_trees_filter_zero_mutations(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = tsutil.insert_branch_sites(ts)
        assert ts.num_sites == ts.num_mutations
        assert ts.num_sites > ts.num_trees
        for keep in [True, False]:
            for filter_sites in [True, False]:
                tss, _ = do_simplify(
                    ts, samples=None, filter_sites=filter_sites, keep_unary=keep
                )
                assert ts.num_sites == tss.num_sites
                assert ts.num_mutations == tss.num_mutations

    def test_many_trees_filter_zero_multichar_mutations(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = tsutil.insert_multichar_mutations(ts)
        assert ts.num_sites == ts.num_trees
        assert ts.num_mutations == ts.num_trees
        for keep in [True, False]:
            for filter_sites in [True, False]:
                tss, _ = do_simplify(
                    ts, samples=None, filter_sites=filter_sites, keep_unary=keep
                )
                assert ts.num_sites == tss.num_sites
                assert ts.num_mutations == tss.num_mutations

    def test_simple_population_filter(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.dump_tables()
        tables.populations.add_row(metadata=b"unreferenced")
        assert len(tables.populations) == 2
        for keep in [True, False]:
            tss, _ = do_simplify(
                tables.tree_sequence(), filter_populations=True, keep_unary=keep
            )
            assert tss.num_populations == 1
            tss, _ = do_simplify(
                tables.tree_sequence(), filter_populations=False, keep_unary=keep
            )
            assert tss.num_populations == 2

    def test_interleaved_populations_filter(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(),
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(),
                msprime.PopulationConfiguration(),
            ],
            random_seed=2,
        )
        assert ts.num_populations == 4
        tables = ts.dump_tables()
        # Edit the populations so we can identify the rows.
        tables.populations.clear()
        for j in range(4):
            tables.populations.add_row(metadata=bytes([j]))
        ts = tables.tree_sequence()
        id_map = np.array([-1, 0, -1, -1], dtype=np.int32)
        for keep in [True, False]:
            tss, _ = do_simplify(ts, filter_populations=True, keep_unary=keep)
            assert tss.num_populations == 1
            population = tss.population(0)
            assert population.metadata == bytes([1])
            assert np.array_equal(
                id_map[ts.tables.nodes.population], tss.tables.nodes.population
            )
            tss, _ = do_simplify(ts, filter_populations=False, keep_unary=keep)
            assert tss.num_populations == 4

    def test_removed_node_population_filter(self):
        tables = tskit.TableCollection(1)
        tables.populations.add_row(metadata=bytes(0))
        tables.populations.add_row(metadata=bytes(1))
        tables.populations.add_row(metadata=bytes(2))
        tables.nodes.add_row(flags=1, population=0)
        # Because flags=0 here, this node will be simplified out and the node
        # will disappear.
        tables.nodes.add_row(flags=0, population=1)
        tables.nodes.add_row(flags=1, population=2)
        for keep in [True, False]:
            tss, _ = do_simplify(
                tables.tree_sequence(), filter_populations=True, keep_unary=keep
            )
            assert tss.num_nodes == 2
            assert tss.num_populations == 2
            assert tss.population(0).metadata == bytes(0)
            assert tss.population(1).metadata == bytes(2)
            assert tss.node(0).population == 0
            assert tss.node(1).population == 1

            tss, _ = do_simplify(
                tables.tree_sequence(), filter_populations=False, keep_unary=keep
            )
            assert tss.tables.populations == tables.populations

    def test_simple_individual_filter(self):
        tables = tskit.TableCollection(1)
        tables.individuals.add_row(flags=0)
        tables.individuals.add_row(flags=1)
        tables.nodes.add_row(flags=1, individual=0)
        tables.nodes.add_row(flags=1, individual=0)
        for keep in [True, False]:
            tss, _ = do_simplify(
                tables.tree_sequence(), filter_individuals=True, keep_unary=keep
            )
            assert tss.num_nodes == 2
            assert tss.num_individuals == 1
            assert tss.individual(0).flags == 0

        tss, _ = do_simplify(tables.tree_sequence(), filter_individuals=False)
        assert tss.tables.individuals == tables.individuals

    def test_interleaved_individual_filter(self):
        tables = tskit.TableCollection(1)
        tables.individuals.add_row(flags=0)
        tables.individuals.add_row(flags=1)
        tables.individuals.add_row(flags=2)
        tables.nodes.add_row(flags=1, individual=1)
        tables.nodes.add_row(flags=1, individual=-1)
        tables.nodes.add_row(flags=1, individual=1)
        for keep in [True, False]:
            tss, _ = do_simplify(
                tables.tree_sequence(), filter_individuals=True, keep_unary=keep
            )
            assert tss.num_nodes == 3
            assert tss.num_individuals == 1
            assert tss.individual(0).flags == 1

            tss, _ = do_simplify(
                tables.tree_sequence(), filter_individuals=False, keep_unary=keep
            )
            assert tss.tables.individuals == tables.individuals

    def test_removed_node_individual_filter(self):
        tables = tskit.TableCollection(1)
        tables.individuals.add_row(flags=0)
        tables.individuals.add_row(flags=1)
        tables.individuals.add_row(flags=2)
        tables.nodes.add_row(flags=1, individual=0)
        # Because flags=0 here, this node will be simplified out and the node
        # will disappear.
        tables.nodes.add_row(flags=0, individual=1)
        tables.nodes.add_row(flags=1, individual=2)
        for keep in [True, False]:
            tss, _ = do_simplify(
                tables.tree_sequence(), filter_individuals=True, keep_unary=keep
            )
            assert tss.num_nodes == 2
            assert tss.num_individuals == 2
            assert tss.individual(0).flags == 0
            assert tss.individual(1).flags == 2
            assert tss.node(0).individual == 0
            assert tss.node(1).individual == 1

            tss, _ = do_simplify(
                tables.tree_sequence(), filter_individuals=False, keep_unary=keep
            )
            assert tss.tables.individuals == tables.individuals

    def verify_simplify_haplotypes(self, ts, samples, keep_unary=False):
        sub_ts, node_map = do_simplify(
            ts, samples, filter_sites=False, keep_unary=keep_unary
        )
        assert ts.num_sites == sub_ts.num_sites
        sub_haplotypes = list(sub_ts.haplotypes(isolated_as_missing=False))
        all_samples = list(ts.samples())
        k = 0
        for j, h in enumerate(ts.haplotypes(isolated_as_missing=False)):
            if k == len(samples):
                break
            if samples[k] == all_samples[j]:
                assert h == sub_haplotypes[k]
                k += 1

    def test_single_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_many_trees_recurrent_mutations(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_single_multiroot_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    @pytest.mark.slow
    def test_many_multiroot_trees_recurrent_mutations(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_single_tree_recurrent_mutations_internal_samples(self):
        ts = msprime.simulate(6, random_seed=10)
        ts = tsutil.jiggle_samples(ts)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)

    def test_many_trees_recurrent_mutations_internal_samples(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        ts = tsutil.jiggle_samples(ts)
        assert ts.num_trees > 3
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    for keep in [True, False]:
                        self.verify_simplify_haplotypes(ts, samples, keep_unary=keep)


class TestSimplifyUnreferencedPopulations:
    def example(self):
        tables = tskit.TableCollection(1)
        tables.populations.add_row()
        tables.populations.add_row()
        # No references to population 0
        tables.nodes.add_row(time=0, population=1, flags=1)
        tables.nodes.add_row(time=0, population=1, flags=1)
        tables.nodes.add_row(time=1, population=1, flags=0)
        # Unreference node
        tables.nodes.add_row(time=1, population=1, flags=0)
        tables.edges.add_row(0, 1, parent=2, child=0)
        tables.edges.add_row(0, 1, parent=2, child=1)
        tables.sort()
        return tables

    def test_no_filter_populations(self):
        tables = self.example()
        tables.simplify(filter_populations=False)
        assert len(tables.populations) == 2
        assert len(tables.nodes) == 3
        assert np.all(tables.nodes.population == 1)

    def test_no_filter_populations_nodes(self):
        tables = self.example()
        tables.simplify(filter_populations=False, filter_nodes=False)
        assert len(tables.populations) == 2
        assert len(tables.nodes) == 4
        assert np.all(tables.nodes.population == 1)

    def test_filter_populations_no_filter_nodes(self):
        tables = self.example()
        tables.simplify(filter_populations=True, filter_nodes=False)
        assert len(tables.populations) == 1
        assert len(tables.nodes) == 4
        assert np.all(tables.nodes.population == 0)

    def test_remapped_default(self):
        tables = self.example()
        tables.simplify()
        assert len(tables.populations) == 1
        assert len(tables.nodes) == 3
        assert np.all(tables.nodes.population == 0)


class TestSimplifyUnreferencedIndividuals:
    def example(self):
        tables = tskit.TableCollection(1)
        tables.individuals.add_row()
        tables.individuals.add_row()
        # No references to individual 0
        tables.nodes.add_row(time=0, individual=1, flags=1)
        tables.nodes.add_row(time=0, individual=1, flags=1)
        tables.nodes.add_row(time=1, individual=1, flags=0)
        # Unreference node
        tables.nodes.add_row(time=1, individual=1, flags=0)
        tables.edges.add_row(0, 1, parent=2, child=0)
        tables.edges.add_row(0, 1, parent=2, child=1)
        tables.sort()
        return tables

    def test_no_filter_individuals(self):
        tables = self.example()
        tables.simplify(filter_individuals=False)
        assert len(tables.individuals) == 2
        assert len(tables.nodes) == 3
        assert np.all(tables.nodes.individual == 1)

    def test_no_filter_individuals_nodes(self):
        tables = self.example()
        tables.simplify(filter_individuals=False, filter_nodes=False)
        assert len(tables.individuals) == 2
        assert len(tables.nodes) == 4
        assert np.all(tables.nodes.individual == 1)

    def test_filter_individuals_no_filter_nodes(self):
        tables = self.example()
        tables.simplify(filter_individuals=True, filter_nodes=False)
        assert len(tables.individuals) == 1
        assert len(tables.nodes) == 4
        assert np.all(tables.nodes.individual == 0)

    def test_remapped_default(self):
        tables = self.example()
        tables.simplify()
        assert len(tables.individuals) == 1
        assert len(tables.nodes) == 3
        assert np.all(tables.nodes.individual == 0)


class TestSimplifyKeepInputRoots(SimplifyTestBase, ExampleTopologyMixin):
    """
    Tests for the keep_input_roots option to simplify.
    """

    def verify(self, ts):
        # Called by the examples in ExampleTopologyMixin
        samples = ts.samples()
        self.verify_keep_input_roots(ts, samples[:2])
        self.verify_keep_input_roots(ts, samples[:3])
        self.verify_keep_input_roots(ts, samples[:-1])
        self.verify_keep_input_roots(ts, samples)

    def verify_keep_input_roots(self, ts, samples):
        ts = tsutil.insert_unique_metadata(ts, ["individuals"])
        ts_with_roots, node_map = do_simplify(
            ts, samples, keep_input_roots=True, filter_sites=False, compare_lib=True
        )
        new_to_input_map = {
            value: key for key, value in enumerate(node_map) if value != tskit.NULL
        }
        for (left, right), input_tree, tree_with_roots in ts.coiterate(ts_with_roots):
            input_roots = input_tree.roots
            assert len(tree_with_roots.roots) > 0
            for root in tree_with_roots.roots:
                # Check that the roots in the current
                input_root = new_to_input_map[root]
                assert input_root in input_roots
                input_node = ts.node(input_root)
                new_node = ts_with_roots.node(root)
                assert new_node.time == input_node.time
                assert new_node.population == input_node.population
                if new_node.individual == tskit.NULL:
                    assert new_node.individual == input_node.individual
                else:
                    assert (
                        ts_with_roots.individual(new_node.individual).metadata
                        == ts.individual(input_node.individual).metadata
                    )
                assert new_node.metadata == input_node.metadata
                # This should only be marked as a sample if it's an
                # element of the samples list.
                assert new_node.is_sample() == (input_root in samples)
                # Find the MRCA of the samples below this root.
                root_samples = list(tree_with_roots.samples(root))
                mrca = functools.reduce(tree_with_roots.mrca, root_samples)
                if mrca != root:
                    # If the MRCA is not equal to the root, then there should
                    # be a unary branch joining them.
                    assert tree_with_roots.parent(mrca) == root
                    assert tree_with_roots.children(root) == (mrca,)

                    # Any mutations that were on the path from the old MRCA
                    # to the root should be mapped to this node, and any mutations
                    # above the root should still be there.
                    u = new_to_input_map[mrca]
                    root_path = []
                    while u != tskit.NULL:
                        root_path.append(u)
                        u = input_tree.parent(u)
                    input_sites = {
                        site.position: site
                        for site in input_tree.sites()
                        if site.position >= left and site.position < right
                    }
                    new_sites = {
                        site.position: site
                        for site in tree_with_roots.sites()
                        if site.position >= left and site.position < right
                    }
                    assert set(input_sites.keys()) == set(new_sites.keys())
                    positions = input_sites.keys()
                    for position in positions:
                        assert left <= position < right
                        new_site = new_sites[position]
                        # We assume the metadata contains a unique key for each mutation.
                        new_mutations = {
                            mut.metadata: mut for mut in new_site.mutations
                        }
                        # Just make sure the metadata is actually unique.
                        assert len(new_mutations) == len(new_site.mutations)
                        input_site = input_sites[position]
                        for input_mutation in input_site.mutations:
                            if input_mutation.node in root_path:
                                new_node = (
                                    mrca if input_mutation.node != input_root else root
                                )
                                # The same mutation should exist and be mapped to
                                # new_node
                                new_mutation = new_mutations[input_mutation.metadata]
                                # We have turned filter sites off, so sites should
                                # be comparable
                                assert new_mutation.site == input_mutation.site
                                assert (
                                    new_mutation.derived_state
                                    == input_mutation.derived_state
                                )
                                assert new_mutation.node == new_node

        return ts_with_roots

    def test_many_trees(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        for num_samples in range(1, ts.num_samples):
            for samples in itertools.combinations(ts.samples(), num_samples):
                self.verify_keep_input_roots(ts, samples)

    def test_many_trees_internal_samples(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=10)
        ts = tsutil.jiggle_samples(ts)
        assert ts.num_trees > 3
        for num_samples in range(1, ts.num_samples):
            for samples in itertools.combinations(ts.samples(), num_samples):
                self.verify_keep_input_roots(ts, samples)

    def test_many_multiroot_trees(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for num_samples in range(1, ts.num_samples):
            for samples in itertools.combinations(ts.samples(), num_samples):
                self.verify_keep_input_roots(ts, samples)

    def test_wright_fisher_unsimplified(self):
        num_generations = 10
        tables = wf.wf_sim(10, num_generations, deep_history=False, seed=2)
        tables.sort()
        ts = tables.tree_sequence()
        simplified = self.verify_keep_input_roots(ts, ts.samples())
        roots = set()
        for tree in simplified.trees():
            for root in tree.roots:
                roots.add(root)
                assert tree.time(root) == num_generations
        init_nodes = np.where(simplified.tables.nodes.time == num_generations)[0]
        assert set(init_nodes) == roots

    def test_single_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    self.verify_keep_input_roots(ts, samples)

    def test_many_trees_recurrent_mutations(self):
        ts = msprime.simulate(5, recombination_rate=1, random_seed=8)
        assert ts.num_trees > 2
        for mutations_per_branch in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
            for num_samples in range(1, ts.num_samples):
                for samples in itertools.combinations(ts.samples(), num_samples):
                    self.verify_keep_input_roots(ts, samples)


class TestSimplifyFilterNodes:
    """
    Tests simplify when nodes are kept in the ts with filter_nodes=False
    """

    def reverse_node_indexes(self, ts):
        tables = ts.dump_tables()
        nodes = tables.nodes
        edges = tables.edges
        mutations = tables.mutations
        nodes.replace_with(nodes[::-1])
        edges.parent = ts.num_nodes - edges.parent - 1
        edges.child = ts.num_nodes - edges.child - 1
        mutations.node = ts.num_nodes - mutations.node - 1
        tables.sort()
        return tables.tree_sequence()

    def verify_nodes_unchanged(self, ts_in, resample_size=None, **kwargs):
        if resample_size is None:
            samples = None
        else:
            np.random.seed(42)
            samples = np.sort(
                np.random.choice(ts_in.num_nodes, resample_size, replace=False)
            )

        for ts in (ts_in, self.reverse_node_indexes(ts_in)):
            filtered, n_map = do_simplify(
                ts, samples=samples, filter_nodes=False, compare_lib=True, **kwargs
            )
            assert np.array_equal(n_map, np.arange(ts.num_nodes, dtype=n_map.dtype))
            referenced_nodes = set(filtered.samples())
            referenced_nodes.update(filtered.edges_parent)
            referenced_nodes.update(filtered.edges_child)
            for n1, n2 in zip(ts.nodes(), filtered.nodes()):
                # Ignore the tskit.NODE_IS_SAMPLE flag which can be changed by simplify
                n1 = n1.replace(flags=n1.flags | tskit.NODE_IS_SAMPLE)
                n2 = n2.replace(flags=n2.flags | tskit.NODE_IS_SAMPLE)
                assert n1 == n2

            # Check that edges are identical to the normal simplify(),
            # with the normal "simplify" having altered IDs
            simplified, node_map = ts.simplify(
                samples=samples, map_nodes=True, **kwargs
            )
            simplified_edges = {e for e in simplified.tables.edges}
            filtered_edges = {
                e.replace(parent=node_map[e.parent], child=node_map[e.child])
                for e in filtered.tables.edges
            }
            assert filtered_edges == simplified_edges

    def test_empty(self):
        ts = tskit.TableCollection(1).tree_sequence()
        self.verify_nodes_unchanged(ts)

    def test_all_samples(self):
        ts = tskit.Tree.generate_comb(5).tree_sequence
        tables = ts.dump_tables()
        flags = tables.nodes.flags
        flags |= tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        ts = tables.tree_sequence()
        assert ts.num_samples == ts.num_nodes
        self.verify_nodes_unchanged(ts)

    @pytest.mark.parametrize("resample_size", [None, 4])
    def test_no_topology(self, resample_size):
        ts = tskit.Tree.generate_comb(5).tree_sequence
        ts = ts.keep_intervals([], simplify=False)
        assert ts.num_nodes > 5  # has unreferenced nodes
        self.verify_nodes_unchanged(ts, resample_size=resample_size)

    @pytest.mark.parametrize("resample_size", [None, 2])
    def test_stick_tree(self, resample_size):
        ts = tskit.Tree.generate_comb(2).tree_sequence
        ts = ts.simplify([0], keep_unary=True)
        assert ts.first().parent(0) != tskit.NULL
        self.verify_nodes_unchanged(ts, resample_size=resample_size)

        # switch to an internal sample
        tables = ts.dump_tables()
        flags = tables.nodes.flags
        flags[0] = 0
        flags[1] = tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        self.verify_nodes_unchanged(tables.tree_sequence(), resample_size=resample_size)

    @pytest.mark.parametrize("resample_size", [None, 4])
    def test_internal_samples(self, resample_size):
        ts = tskit.Tree.generate_comb(4).tree_sequence
        tables = ts.dump_tables()
        flags = tables.nodes.flags
        flags ^= tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        ts = tables.tree_sequence()
        assert np.all(ts.samples() >= ts.num_samples)
        self.verify_nodes_unchanged(ts, resample_size=resample_size)

    @pytest.mark.parametrize("resample_size", [None, 4])
    def test_blank_flanks(self, resample_size):
        ts = tskit.Tree.generate_comb(4).tree_sequence
        ts = ts.keep_intervals([[0.25, 0.75]], simplify=False)
        self.verify_nodes_unchanged(ts, resample_size=resample_size)

    @pytest.mark.parametrize("resample_size", [None, 4])
    def test_multiroot(self, resample_size):
        ts = tskit.Tree.generate_balanced(6).tree_sequence
        ts = ts.decapitate(2.5)
        self.verify_nodes_unchanged(ts, resample_size=resample_size)

    @pytest.mark.parametrize("resample_size", [None, 10])
    def test_with_metadata(self, ts_fixture_for_simplify, resample_size):
        assert ts_fixture_for_simplify.num_nodes > 10
        self.verify_nodes_unchanged(
            ts_fixture_for_simplify, resample_size=resample_size
        )

    @pytest.mark.parametrize("resample_size", [None, 7])
    def test_complex_ts_with_unary(self, resample_size):
        ts = msprime.sim_ancestry(
            3,
            sequence_length=10,
            recombination_rate=1,
            record_full_arg=True,
            random_seed=123,
        )
        assert ts.num_trees > 2
        ts = msprime.sim_mutations(ts, rate=1, random_seed=123)
        # Add some unreferenced nodes
        tables = ts.dump_tables()
        tables.nodes.add_row(flags=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE)
        ts = tables.tree_sequence()
        self.verify_nodes_unchanged(ts, resample_size=resample_size)

    def test_keeping_unary(self):
        # Test interaction with keeping unary nodes
        n_samples = 6
        ts = tskit.Tree.generate_comb(n_samples).tree_sequence
        num_nodes = ts.num_nodes
        reduced_n_samples = [2, n_samples - 1]  # last sample is most deeply nested
        ts_with_unary = ts.simplify(reduced_n_samples, keep_unary=True)
        assert ts_with_unary.num_nodes == num_nodes - n_samples + len(reduced_n_samples)
        tree = ts_with_unary.first()
        assert any([tree.num_children(u) == 1 for u in tree.nodes()])
        self.verify_nodes_unchanged(ts_with_unary, keep_unary=True)
        self.verify_nodes_unchanged(ts_with_unary, keep_unary=False)

    def test_find_unreferenced_nodes(self):
        # Simple test to show we can find unreferenced nodes easily.
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4).tree_sequence
        ts2, node_map = do_simplify(
            ts1,
            [0, 1, 2],
            filter_nodes=False,
        )
        assert np.array_equal(node_map, np.arange(ts1.num_nodes))
        node_references = np.zeros(ts1.num_nodes, dtype=np.int32)
        node_references[ts2.edges_parent] += 1
        node_references[ts2.edges_child] += 1
        # Simplifying for [0, 1, 2] should remove references to node 3 and 5
        assert list(node_references) == [1, 1, 1, 0, 2, 0, 1]

    def test_mutations_on_removed_branches(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
        # A mutation on a removed branch should get removed
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(0, node=3, derived_state="T")
        ts2, node_map = do_simplify(
            tables.tree_sequence(),
            [0, 1, 2],
            filter_nodes=False,
        )
        assert ts2.num_sites == 0
        assert ts2.num_mutations == 0


class TestSimplifyNoUpdateSampleFlags:
    """
    Tests for simplify when we don't update the sample flags.
    """

    def test_simple_case_filter_nodes(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4).tree_sequence
        ts2, node_map = do_simplify(
            ts1,
            [0, 1, 6],
            update_sample_flags=False,
        )
        # Because we don't retain 2 and 3 here, they don't stay as
        # samples. But, we specified 6 as a sample, so it's coming
        # through where it would ordinarily be dropped.

        # 2.00┊  2  ┊
        #     ┊  ┃  ┊
        # 1.00┊  3  ┊
        #     ┊ ┏┻┓ ┊
        # 0.00┊ 0 1 ┊
        #     0     1
        assert list(ts2.nodes_flags) == [1, 1, 0, 0]
        tree = ts2.first()
        assert list(tree.parent_array) == [3, 3, -1, 2, -1]

    def test_simple_case_no_filter_nodes(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4).tree_sequence
        ts2, node_map = do_simplify(
            ts1,
            [0, 1, 6],
            update_sample_flags=False,
            filter_nodes=False,
        )

        # 2.00┊  6      ┊
        #     ┊  ┃      ┊
        # 1.00┊  4      ┊
        #     ┊ ┏┻┓     ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        assert list(ts2.nodes_flags) == list(ts1.nodes_flags)
        tree = ts2.first()
        assert list(tree.parent_array) == [4, 4, -1, -1, 6, -1, -1, -1]


class TestMapToAncestors:
    """
    Tests the AncestorMap class.
    """

    random_seed = 13
    #
    #          8
    #         / \
    #        /   \
    #       /     \
    #      7       \
    #     / \       6
    #    /   5     / \
    #   /   / \   /   \
    #  4   0   1 2     3
    nodes = """\
    id      is_sample   population      time
    0       1       0               0.00000000000000
    1       1       0               0.00000000000000
    2       1       0               0.00000000000000
    3       1       0               0.00000000000000
    4       1       0               0.00000000000000
    5       0       0               0.14567111023387
    6       0       0               0.21385545626353
    7       0       0               0.43508024345063
    8       0       0               1.60156352971203
    """
    edges = """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      5       0,1
    1       0.00000000      1.00000000      6       2,3
    2       0.00000000      1.00000000      7       4,5
    3       0.00000000      1.00000000      8       6,7
    """
    #
    #          9                        10
    #         / \                      / \
    #        /   \                    /   8
    #       /     \                  /   / \
    #      7       \                /   /   \
    #     / \       6              /   /     6
    #    /   5     / \            /   5     / \
    #   /   / \   /   \          /   / \   /   \
    #  4   0   1 2     3        4   0   1 2     3
    #
    # 0 ------------------ 0.5 ------------------ 1.0
    nodes0 = """\
    id      is_sample   population      time
    0       1       0               0.00000000000000
    1       1       0               0.00000000000000
    2       1       0               0.00000000000000
    3       1       0               0.00000000000000
    4       1       0               0.00000000000000
    5       0       0               0.14567111023387
    6       0       0               0.21385545626353
    7       0       0               0.43508024345063
    8       0       0               0.60156352971203
    9       0       0               0.90000000000000
    10      0       0               1.20000000000000
    """
    edges0 = """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      5       0,1
    1       0.00000000      1.00000000      6       2,3
    2       0.00000000      0.50000000      7       4,5
    3       0.50000000      1.00000000      8       5,6
    4       0.00000000      0.50000000      9       6,7
    5       0.50000000      1.00000000      10      4,8
    """
    nodes1 = """\
    id      is_sample   population      time
    0       0           0           1.0
    1       1           0           0.0
    2       1           0           0.0
    """
    edges1 = """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      0       1,2
    """

    def do_map(self, ts, ancestors, samples=None, compare_lib=True):
        """
        Runs the Python test implementation of link_ancestors.
        """
        if samples is None:
            samples = ts.samples()
        s = tests.AncestorMap(ts, samples, ancestors)
        ancestor_table = s.link_ancestors()
        if compare_lib:
            lib_result = ts.dump_tables().link_ancestors(samples, ancestors)
            assert ancestor_table == lib_result
        return ancestor_table

    def test_deprecated_name(self):
        # copied from test_single_tree_one_ancestor below
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        samples = ts.samples()
        ancestors = [8]
        s = tests.AncestorMap(ts, samples, ancestors)
        tss = s.link_ancestors()
        lib_result = ts.dump_tables().map_ancestors(samples, ancestors)
        assert tss == lib_result
        assert list(tss.parent) == [8, 8, 8, 8, 8]
        assert list(tss.child) == [0, 1, 2, 3, 4]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_one_ancestor(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[8])
        assert list(tss.parent) == [8, 8, 8, 8, 8]
        assert list(tss.child) == [0, 1, 2, 3, 4]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_unordered_nodes(self):
        nodes = io.StringIO(self.nodes1)
        edges = io.StringIO(self.edges1)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[0])
        assert list(tss.parent) == [0, 0]
        assert list(tss.child) == [1, 2]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_two_ancestors(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[6, 7])
        assert list(tss.parent) == [6, 6, 7, 7, 7]
        assert list(tss.child) == [2, 3, 0, 1, 4]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_no_ancestors(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[2, 3], ancestors=[7])
        assert tss.num_rows == 0

    def test_single_tree_samples_or_ancestors_not_in_tree(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        with pytest.raises(AssertionError):
            self.do_map(ts, samples=[-1, 3], ancestors=[5])
        with pytest.raises(AssertionError):
            self.do_map(ts, samples=[2, 3], ancestors=[10])

    def test_single_tree_ancestors_descend_from_other_ancestors(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[7, 8])
        assert list(tss.parent) == [7, 7, 7, 8, 8, 8]
        assert list(tss.child) == [0, 1, 4, 2, 3, 7]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_internal_samples(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[2, 3, 4, 5], ancestors=[7, 8])
        assert list(tss.parent) == [7, 7, 8, 8, 8]
        assert list(tss.child) == [4, 5, 2, 3, 7]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_samples_and_ancestors_overlap(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[1, 2, 3, 5], ancestors=[5, 6, 7])
        assert list(tss.parent) == [5, 6, 6, 7]
        assert list(tss.child) == [1, 2, 3, 5]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_unary_ancestor(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[1, 2, 4], ancestors=[5, 7, 8])
        assert list(tss.parent) == [5, 7, 7, 8, 8]
        assert list(tss.child) == [1, 4, 5, 2, 7]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_ancestors_descend_from_samples(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[1, 7], ancestors=[5, 8])
        assert list(tss.parent) == [5, 7, 8]
        assert list(tss.child) == [1, 5, 7]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_single_tree_samples_descend_from_samples(self):
        nodes = io.StringIO(self.nodes)
        edges = io.StringIO(self.edges)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, samples=[3, 6], ancestors=[8])
        assert list(tss.parent) == [6, 8]
        assert list(tss.child) == [3, 6]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_multiple_trees_to_single_tree(self):
        nodes = io.StringIO(self.nodes0)
        edges = io.StringIO(self.edges0)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[5, 6])
        assert list(tss.parent) == [5, 5, 6, 6]
        assert list(tss.child) == [0, 1, 2, 3]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def test_multiple_trees_one_ancestor(self):
        nodes = io.StringIO(self.nodes0)
        edges = io.StringIO(self.edges0)
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        tss = self.do_map(ts, ancestors=[9, 10])
        assert list(tss.parent) == [9, 9, 9, 9, 9, 10, 10, 10, 10, 10]
        assert list(tss.child) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        assert all(tss.left) == 0
        assert all(tss.right) == 1

    def verify(self, ts, sample_nodes, ancestral_nodes):
        tss = self.do_map(ts, ancestors=ancestral_nodes, samples=sample_nodes)
        # ancestors = list(set(tss.parent))
        # Loop through the rows of the ancestral branch table.
        current_ancestor = tss.parent[0]
        current_descendants = [tss.child[0]]
        current_left = tss.left[0]
        current_right = tss.right[0]
        for _, row in enumerate(tss):
            if (
                row.parent != current_ancestor
                or row.left != current_left
                or row.right != current_right
            ):
                # Loop through trees.
                for tree in ts.trees():
                    if tree.interval.left >= current_right:
                        break
                    while tree.interval.right <= current_left:
                        tree.next()
                    # Check that the most recent ancestor of the descendants is the
                    # current_ancestor.
                    current_descendants = list(set(current_descendants))
                    for des in current_descendants:
                        par = tree.get_parent(des)
                        while par not in ancestral_nodes and par not in sample_nodes:
                            par = tree.get_parent(par)
                        assert par == current_ancestor
                # Reset the current ancestor and descendants, left and right coords.
                current_ancestor = row.parent
                current_descendants = [row.child]
                current_left = row.left
                current_right = row.right
            else:
                # Collate a list of children corresponding to each ancestral node.
                current_descendants.append(row.child)

    def test_sim_single_coalescent_tree(self):
        ts = msprime.simulate(30, random_seed=1, length=10)
        ancestors = [3 * n for n in np.arange(0, ts.num_nodes // 3)]
        self.verify(ts, ts.samples(), ancestors)
        random_samples = [4 * n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, random_samples, ancestors)

    def test_sim_coalescent_trees(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=1, length=2)
        ancestors = [3 * n for n in np.arange(0, ts.num_nodes // 3)]
        self.verify(ts, ts.samples(), ancestors)
        random_samples = [4 * n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, random_samples, ancestors)

    def test_sim_coalescent_trees_internal_samples(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=10, length=2)
        assert ts.num_trees > 2
        ancestors = [4 * n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(tsutil.jiggle_samples(ts), ts.samples(), ancestors)
        random_samples = [4 * n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(tsutil.jiggle_samples(ts), random_samples, ancestors)

    def test_sim_many_multiroot_trees(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        ancestors = [4 * n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, ts.samples(), ancestors)
        random_samples = [4 * n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, random_samples, ancestors)

    def test_sim_wright_fisher_generations(self):
        number_of_gens = 5
        tables = wf.wf_sim(10, number_of_gens, deep_history=False, seed=2)
        tables.sort()
        ts = tables.tree_sequence()
        ancestors = [4 * n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, ts.samples(), ancestors)
        for gen in range(1, number_of_gens):
            ancestors = [u.id for u in ts.nodes() if u.time == gen]
            self.verify(ts, ts.samples(), ancestors)

        random_samples = [4 * n for n in np.arange(0, ts.num_nodes // 4)]
        self.verify(ts, random_samples, ancestors)
        for gen in range(1, number_of_gens):
            ancestors = [u.id for u in ts.nodes() if u.time == gen]
            self.verify(ts, random_samples, ancestors)


class TestMutationParent:
    """
    Tests that mutation parent is correctly specified, and that we correctly
    recompute it with compute_mutation_parent.
    """

    seed = 42

    def verify_parents(self, ts):
        parent = tsutil.compute_mutation_parent(ts)
        tables = ts.dump_tables()
        assert np.array_equal(parent, tables.mutations.parent)
        tables.mutations.parent = np.zeros_like(tables.mutations.parent) - 1
        assert np.all(tables.mutations.parent == tskit.NULL)
        tables.compute_mutation_parents()
        assert np.array_equal(parent, tables.mutations.parent)

    def test_example(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       0           2.0
        1       0           1.0
        2       0           1.0
        3       1           0
        4       1           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0    0.5   2  3
        0.0    0.8   2  4
        0.5    1.0   1  3
        0.0    1.0   0  1
        0.0    1.0   0  2
        0.8    1.0   0  4
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.1     0
        0.5     0
        0.9     0
        """
        )
        mutations = io.StringIO(
            """\
        site    node    derived_state   parent
        0       1       1               -1
        0       2       1               -1
        0       3       2               1
        1       0       1               -1
        1       1       1               3
        1       3       2               4
        1       2       1               3
        1       4       2               6
        2       0       1               -1
        2       1       1               8
        2       2       1               8
        2       4       1               8
        """
        )
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
        )
        self.verify_parents(ts)

    def test_single_muts(self):
        ts = msprime.simulate(
            10, random_seed=self.seed, mutation_rate=3.0, recombination_rate=1.0
        )
        self.verify_parents(ts)

    def test_with_jukes_cantor(self):
        ts = msprime.simulate(
            10, random_seed=self.seed, mutation_rate=0.0, recombination_rate=1.0
        )
        # make *lots* of recurrent mutations
        mut_ts = tsutil.jukes_cantor(
            ts, num_sites=10, mu=1, multiple_per_node=False, seed=self.seed
        )
        self.verify_parents(mut_ts)

    def test_with_jukes_cantor_multiple_per_node(self):
        ts = msprime.simulate(
            10, random_seed=self.seed, mutation_rate=0.0, recombination_rate=1.0
        )
        # make *lots* of recurrent mutations
        mut_ts = tsutil.jukes_cantor(
            ts, num_sites=10, mu=1, multiple_per_node=True, seed=self.seed
        )
        self.verify_parents(mut_ts)

    def verify_branch_mutations(self, ts, mutations_per_branch):
        ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
        assert ts.num_mutations > 1
        self.verify_parents(ts)

    def test_single_tree_one_mutation_per_branch(self):
        ts = msprime.simulate(6, random_seed=10)
        self.verify_branch_mutations(ts, 1)

    def test_single_tree_two_mutations_per_branch(self):
        ts = msprime.simulate(10, random_seed=9)
        self.verify_branch_mutations(ts, 2)

    def test_single_tree_three_mutations_per_branch(self):
        ts = msprime.simulate(8, random_seed=9)
        self.verify_branch_mutations(ts, 3)

    def test_single_multiroot_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for mutations_per_branch in [1, 2, 3]:
            self.verify_branch_mutations(ts, mutations_per_branch)

    def test_many_multiroot_trees_recurrent_mutations(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for mutations_per_branch in [1, 2, 3]:
            self.verify_branch_mutations(ts, mutations_per_branch)


class TestMutationEdge:
    def verify_mutation_edge(self, ts):
        # print(ts.tables)
        for mutation in ts.mutations():
            site = ts.site(mutation.site)
            if mutation.edge == tskit.NULL:
                edges = [
                    edge
                    for edge in ts.edges()
                    if edge.left <= site.position < edge.right
                    and mutation.node == edge.child
                ]
                assert len(edges) == 0
            else:
                edge = ts.edge(mutation.edge)
                assert edge.left <= site.position < edge.right
                assert edge.child == mutation.node

        for tree in ts.trees():
            for site in tree.sites():
                for mutation in site.mutations:
                    assert mutation.edge == ts.mutation(mutation.id).edge
                    if mutation.edge == tskit.NULL:
                        assert tree.parent(mutation.node) == tskit.NULL

    def verify_branch_mutations(self, ts, mutations_per_branch):
        ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
        assert ts.num_mutations > 1
        self.verify_mutation_edge(ts)

    def test_single_tree_one_mutation_per_branch(self):
        ts = msprime.simulate(6, random_seed=10)
        self.verify_branch_mutations(ts, 1)

    def test_single_tree_two_mutations_per_branch(self):
        ts = msprime.simulate(10, random_seed=9)
        self.verify_branch_mutations(ts, 2)

    def test_single_tree_three_mutations_per_branch(self):
        ts = msprime.simulate(8, random_seed=9)
        self.verify_branch_mutations(ts, 3)

    def test_single_multiroot_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for mutations_per_branch in [1, 2, 3]:
            self.verify_branch_mutations(ts, mutations_per_branch)

    def test_many_multiroot_trees_recurrent_mutations(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for mutations_per_branch in [1, 2, 3]:
            self.verify_branch_mutations(ts, mutations_per_branch)

    @pytest.mark.parametrize("n", range(2, 5))
    @pytest.mark.parametrize("mutations_per_branch", range(3))
    def test_balanced_binary_tree(self, n, mutations_per_branch):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        # These trees have a handy property
        assert all(edge.id == edge.child for edge in ts.edges())
        for mutation in ts.mutations():
            assert mutation.edge == mutation.node
        for site in ts.first().sites():
            for mutation in site.mutations:
                assert mutation.edge == mutation.node


class TestMutationTime:
    """
    Tests that mutation time is correctly specified, and that we correctly
    recompute it with compute_mutation_times.
    """

    seed = 42

    def verify_times(self, ts):
        tables = ts.dump_tables()
        # Clear out the existing mutations as they come from msprime
        tables.mutations.time = np.full(
            tables.mutations.time.shape, -1, dtype=np.float64
        )
        assert np.all(tables.mutations.time == -1)
        # Compute times with C method and dumb python method
        tables.compute_mutation_times()
        python_time = tsutil.compute_mutation_times(ts)
        assert np.allclose(python_time, tables.mutations.time, rtol=1e-15, atol=1e-15)

    def test_example(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       0           2.0
        1       0           1.0
        2       0           1.0
        3       1           0
        4       1           0
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        0.0    0.5   2  3
        0.0    0.8   2  4
        0.5    1.0   1  3
        0.0    1.0   0  1
        0.0    1.0   0  2
        0.8    1.0   0  4
        """
        )
        sites = io.StringIO(
            """\
        position    ancestral_state
        0.1     0
        0.5     0
        0.9     0
        """
        )
        mutations = io.StringIO(
            """\
        site	node	time	derived_state	parent
        0       1       1.5     1               -1
        0       2       1.5     1               -1
        0       3       0.5     2               1
        1       0       2.0     1               -1
        1       1       1.5     1               3
        1       3       0.5     2               4
        1       2       1.5     1               3
        1       4       0.5     2               6
        2       0       2.0     1               -1
        2       1       1.5     1               8
        2       2       1.5     1               8
        2       4       1.0     1               8
        """
        )
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
        )
        # ts.dump_text(mutations=sys.stdout)
        # self.assertFalse(True)
        tables = ts.dump_tables()
        python_time = tsutil.compute_mutation_times(ts)
        assert np.allclose(python_time, tables.mutations.time, rtol=1e-15, atol=1e-15)
        tables.mutations.time = np.full(
            tables.mutations.time.shape, -1, dtype=np.float64
        )
        assert np.all(tables.mutations.time == -1)
        tables.compute_mutation_times()
        assert np.allclose(python_time, tables.mutations.time, rtol=1e-15, atol=1e-15)

    def test_single_muts(self):
        ts = msprime.simulate(
            10, random_seed=self.seed, mutation_rate=3.0, recombination_rate=1.0
        )
        self.verify_times(ts)

    def test_with_jukes_cantor(self):
        ts = msprime.simulate(
            10, random_seed=self.seed, mutation_rate=0.0, recombination_rate=1.0
        )
        # make *lots* of recurrent mutations
        mut_ts = tsutil.jukes_cantor(
            ts, num_sites=10, mu=1, multiple_per_node=False, seed=self.seed
        )
        self.verify_times(mut_ts)

    def test_with_jukes_cantor_multiple_per_node(self):
        ts = msprime.simulate(
            10, random_seed=self.seed, mutation_rate=0.0, recombination_rate=1.0
        )
        # make *lots* of recurrent mutations
        mut_ts = tsutil.jukes_cantor(
            ts, num_sites=10, mu=1, multiple_per_node=True, seed=self.seed
        )
        self.verify_times(mut_ts)

    def verify_branch_mutations(self, ts, mutations_per_branch):
        ts = tsutil.insert_branch_mutations(ts, mutations_per_branch)
        assert ts.num_mutations > 1
        self.verify_times(ts)

    def test_single_tree_one_mutation_per_branch(self):
        ts = msprime.simulate(6, random_seed=10)
        self.verify_branch_mutations(ts, 1)

    def test_single_tree_two_mutations_per_branch(self):
        ts = msprime.simulate(10, random_seed=9)
        self.verify_branch_mutations(ts, 2)

    def test_single_tree_three_mutations_per_branch(self):
        ts = msprime.simulate(8, random_seed=9)
        self.verify_branch_mutations(ts, 3)

    def test_single_multiroot_tree_recurrent_mutations(self):
        ts = msprime.simulate(6, random_seed=10)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for mutations_per_branch in [1, 2, 3]:
            self.verify_branch_mutations(ts, mutations_per_branch)

    def test_many_multiroot_trees_recurrent_mutations(self):
        ts = msprime.simulate(7, recombination_rate=1, random_seed=10)
        assert ts.num_trees > 3
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        for mutations_per_branch in [1, 2, 3]:
            self.verify_branch_mutations(ts, mutations_per_branch)


class TestSimpleTreeAlgorithm:
    """
    Tests for the direct implementation of Algorithm T in tsutil.py.

    See TestHoleyTreeSequences above for further tests on wacky topologies.
    """

    def test_zero_nodes(self):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        assert ts.sequence_length == 1
        assert ts.num_trees == 1
        # Test the simple tree iterator.
        trees = list(tsutil.algorithm_T(ts))
        assert len(trees) == 1
        (left, right), parent = trees[0]
        assert left == 0
        assert right == 1
        assert parent == []

    def test_one_node(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row()
        ts = tables.tree_sequence()
        assert ts.sequence_length == 1
        assert ts.num_trees == 1
        # Test the simple tree iterator.
        trees = list(tsutil.algorithm_T(ts))
        assert len(trees) == 1
        (left, right), parent = trees[0]
        assert left == 0
        assert right == 1
        assert parent == [-1]

    def test_single_coalescent_tree(self):
        ts = msprime.simulate(10, random_seed=1, length=10)
        tree = ts.first()
        p1 = [tree.parent(j) for j in range(ts.num_nodes)]
        interval, p2 = next(tsutil.algorithm_T(ts))
        assert interval == tree.interval
        assert p1 == p2

    def test_coalescent_trees(self):
        ts = msprime.simulate(8, recombination_rate=5, random_seed=1, length=2)
        assert ts.num_trees > 2
        new_trees = tsutil.algorithm_T(ts)
        for tree in ts.trees():
            interval, p2 = next(new_trees)
            p1 = [tree.parent(j) for j in range(ts.num_nodes)]
            assert interval == tree.interval
            assert p1 == p2
        with pytest.raises(StopIteration):
            next(new_trees)


class TestVirtualRootAPIs(ExampleTopologyMixin):
    """
    Tests the APIs based on getting roots.
    """

    def verify(self, ts):
        for tree in ts.trees():
            left_child = tree.left_child_array
            right_child = tree.right_child_array
            assert tree.virtual_root == ts.num_nodes
            assert tree.left_root == tree.left_child(tree.virtual_root)
            assert tree.right_root == tree.right_child(tree.virtual_root)
            assert tree.left_root == left_child[-1]
            assert tree.right_root == right_child[-1]
            assert tree.parent(tree.virtual_root) == tskit.NULL
            assert tree.left_sib(tree.virtual_root) == tskit.NULL
            assert tree.right_sib(tree.virtual_root) == tskit.NULL
            assert tree.num_children(tree.virtual_root) == tree.num_roots

            u = tree.left_root
            roots = []
            while u != tskit.NULL:
                roots.append(u)
                u = tree.right_sib(u)
            assert roots == list(tree.roots)

            # The branch_length for roots is defined as 0, and it's consistent
            # to have the same for the virtual root.
            assert tree.branch_length(tree.virtual_root) == 0
            # The virtual root has depth -1 from the root
            assert tree.depth(tree.virtual_root) == -1
            assert tree.num_children(tree.virtual_root) == tree.num_roots
            assert tree.num_samples(tree.virtual_root) == tree.num_samples()
            # We're not using tracked samples here.
            assert tree.num_tracked_samples(tree.virtual_root) == 0
            # The virtual_root is internal because it has children (the roots)
            assert tree.is_internal(tree.virtual_root)
            assert not tree.is_leaf(tree.virtual_root)
            assert not tree.is_sample(tree.virtual_root)
            # The mrca of the virtual_root and anything is itself
            assert tree.mrca(0, tree.virtual_root) == tree.virtual_root
            assert tree.mrca(tree.virtual_root, 0) == tree.virtual_root
            assert tree.mrca(tree.virtual_root, tree.virtual_root) == tree.virtual_root
            # The virtual_root is a descendant of nothing other than itself
            assert not tree.is_descendant(0, tree.virtual_root)
            assert tree.is_descendant(tree.virtual_root, tree.virtual_root)

            assert list(tree.leaves(tree.virtual_root)) == list(tree.leaves())
            assert list(tree.samples(tree.virtual_root)) == list(tree.samples())

            orders = [
                "preorder",
                "inorder",
                "levelorder",
                "breadthfirst",
                "postorder",
                "timeasc",
                "timedesc",
                "minlex_postorder",
            ]
            for order in orders:
                l_vr = list(tree.nodes(tree.virtual_root, order=order))
                l_standard = list(tree.nodes(order=order))
                assert len(l_vr) == 1 + len(l_standard)
                assert tree.virtual_root in l_vr

            # For pre-order, virtual_root should be first node visited:
            assert next(tree.nodes(tree.virtual_root)) == tree.virtual_root

            # Methods that imply looking up tree sequence properties of the
            # node raise an error
            # Some methods don't apply
            for method in [tree.population]:
                with pytest.raises(tskit.LibraryError, match="Node out of bounds"):
                    method(tree.virtual_root)


class TestSampleLists(ExampleTopologyMixin):
    """
    Tests for the sample lists algorithm.
    """

    def verify(self, ts):
        tree1 = tsutil.SampleListTree(ts)
        s = str(tree1)
        assert s is not None
        trees = ts.trees(sample_lists=True)
        for left, right in tree1.sample_lists():
            tree2 = next(trees)
            assert (left, right) == tree2.interval
            for u in tree2.nodes():
                assert tree1.left_sample[u] == tree2.left_sample(u)
                assert tree1.right_sample[u] == tree2.right_sample(u)
            for j in range(ts.num_samples):
                assert tree1.next_sample[j] == tree2.next_sample(j)
        assert right == ts.sequence_length

        tree1 = tsutil.SampleListTree(ts)
        trees = ts.trees(sample_lists=False)
        sample_index_map = ts.samples()
        for _, _ in tree1.sample_lists():
            tree2 = next(trees)
            for u in range(ts.num_nodes):
                samples2 = list(tree2.samples(u))
                samples1 = []
                index = tree1.left_sample[u]
                if index != tskit.NULL:
                    assert sample_index_map[tree1.left_sample[u]] == samples2[0]
                    assert sample_index_map[tree1.right_sample[u]] == samples2[-1]
                    stop = tree1.right_sample[u]
                    while True:
                        assert index != -1
                        samples1.append(sample_index_map[index])
                        if index == stop:
                            break
                        index = tree1.next_sample[index]
                assert samples1 == samples2
            # The python implementation here doesn't maintain roots
            np.testing.assert_array_equal(tree1.parent, tree2.parent_array[:-1])
            np.testing.assert_array_equal(tree1.left_child, tree2.left_child_array[:-1])
            np.testing.assert_array_equal(
                tree1.right_child, tree2.right_child_array[:-1]
            )
        assert right == ts.sequence_length


class TestOneSampleRoot(ExampleTopologyMixin):
    """
    Tests for the standard root threshold of subtending at least
    one sample.
    """

    def verify(self, ts):
        tree2 = tskit.Tree(ts)
        tree2.first()
        for interval, tree1 in tsutil.algorithm_R(ts, root_threshold=1):
            root_reachable_nodes = len(tree2.preorder())
            size_bound = tree1.num_edges + ts.num_samples
            assert size_bound >= root_reachable_nodes
            assert interval == tree2.interval
            assert tree1.roots() == tree2.roots
            # Definition here is the set unique path ends from samples
            roots = set()
            for u in ts.samples():
                while u != tskit.NULL:
                    path_end = u
                    u = tree2.parent(u)
                roots.add(path_end)
            assert set(tree1.roots()) == roots
            np.testing.assert_array_equal(tree1.parent, tree2.parent_array)
            np.testing.assert_array_equal(tree1.left_child, tree2.left_child_array)
            np.testing.assert_array_equal(tree1.right_child, tree2.right_child_array)
            np.testing.assert_array_equal(tree1.left_sib, tree2.left_sib_array)
            np.testing.assert_array_equal(tree1.right_sib, tree2.right_sib_array)
            np.testing.assert_array_equal(tree1.num_children, tree2.num_children_array)
            tree2.next()
        assert tree2.index == -1


class RootThreshold(ExampleTopologyMixin):
    """
    Tests for the root criteria of subtending at least k samples.
    """

    def verify(self, ts):
        k = self.root_threshold
        trees_py = tsutil.algorithm_R(ts, root_threshold=k)
        tree_lib = tskit.Tree(ts, root_threshold=k)
        tree_lib.first()
        tree_leg = tsutil.LegacyRootThresholdTree(ts, root_threshold=k)
        for (interval_py, tree_py), interval_leg in itertools.zip_longest(
            trees_py, tree_leg.iterate()
        ):
            assert interval_py == tree_lib.interval
            assert interval_leg == tree_lib.interval

            root_reachable_nodes = len(tree_lib.preorder())
            size_bound = tree_py.num_edges + ts.num_samples
            assert size_bound >= root_reachable_nodes
            assert tree_py.num_edges == tree_lib.num_edges

            # Definition here is the set unique path ends from samples
            # that subtend at least k samples
            roots = set()
            for u in ts.samples():
                while u != tskit.NULL:
                    path_end = u
                    u = tree_lib.parent(u)
                if tree_lib.num_samples(path_end) >= k:
                    roots.add(path_end)
            assert set(tree_py.roots()) == roots
            assert set(tree_lib.roots) == roots
            assert set(tree_leg.roots()) == roots
            assert len(tree_leg.roots()) == tree_lib.num_roots
            assert tree_py.roots() == tree_lib.roots

            # # The python class has identical behaviour to the lib version
            assert tree_py.left_child[-1] == tree_lib.left_root
            np.testing.assert_array_equal(tree_py.parent, tree_lib.parent_array)
            np.testing.assert_array_equal(tree_py.left_child, tree_lib.left_child_array)
            np.testing.assert_array_equal(
                tree_py.right_child, tree_lib.right_child_array
            )
            np.testing.assert_array_equal(tree_py.left_sib, tree_lib.left_sib_array)
            np.testing.assert_array_equal(tree_py.right_sib, tree_lib.right_sib_array)
            np.testing.assert_array_equal(
                tree_py.num_children, tree_lib.num_children_array
            )

            # NOTE: the legacy left_root value is *not* necessarily the same as the
            # new left_root.
            # assert tree_leg.left_root == tree_py.left_child[-1]

            # The virtual root version is identical to the legacy tree
            # except for the extra node and the details of the sib arrays.
            np.testing.assert_array_equal(tree_py.parent[:-1], tree_leg.parent)
            np.testing.assert_array_equal(tree_py.left_child[:-1], tree_leg.left_child)
            np.testing.assert_array_equal(
                tree_py.right_child[:-1], tree_leg.right_child
            )
            # The sib arrays are identical except for root nodes.
            for u in range(ts.num_nodes):
                if u not in roots:
                    assert tree_py.left_sib[u] == tree_leg.left_sib[u]
                    assert tree_py.right_sib[u] == tree_leg.right_sib[u]

            tree_lib.next()
        assert tree_lib.index == -1


class TestRootThreshold1(RootThreshold):
    root_threshold = 1


class TestRootThreshold2(RootThreshold):
    root_threshold = 2


class TestRootThreshold3(RootThreshold):
    root_threshold = 3


class TestRootThreshold4(RootThreshold):
    root_threshold = 4


class TestRootThreshold10(RootThreshold):
    root_threshold = 10


class TestSquashEdges:
    """
    Tests of the squash_edges function.
    """

    def do_squash(self, ts, compare_lib=True):
        squashed = ts.dump_tables().edges
        squashed.squash()
        if compare_lib:
            squashed_list = squash_edges(ts)
            squashed_py = tskit.EdgeTable()
            for e in squashed_list:
                squashed_py.append(e)
            # Check the Python and C implementations produce the same output.
            assert squashed_py == squashed
        return squashed

    def test_simple_case(self):
        #   2
        #  / \
        # 0   1
        nodes = io.StringIO(
            """\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       0       0               1.00000000000000
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.00000000      0.50000000      2       0
        1       0.00000000      0.50000000      2       1
        2       0.50000000      1.00000000      2       0
        3       0.50000000      1.00000000      2       1
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        assert all(edges.left) == 0
        assert all(edges.right) == 1
        assert list(edges.parent) == [2, 2]
        assert list(edges.child) == [0, 1]

    def test_simple_case_unordered_intervals(self):
        # 1
        # |
        # 0
        nodes = io.StringIO(
            """\
        id      is_sample   population      time
        0       1           0               0.0
        1       0           0               1.0
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.40            1.0             1       0
        0       0.00            0.40            1       0
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        assert edges.left[0] == 0
        assert edges.right[0] == 1
        assert edges.parent[0] == 1
        assert edges.child[0] == 0

    def test_simple_case_unordered_children(self):
        #   2
        #  / \
        # 0   1
        nodes = io.StringIO(
            """\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       0       0               1.00000000000000
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.50000000      1.00000000      2       1
        1       0.50000000      1.00000000      2       0
        2       0.00000000      0.50000000      2       1
        3       0.00000000      0.50000000      2       0
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        assert all(edges.left) == 0
        assert all(edges.right) == 1
        assert list(edges.parent) == [2, 2]
        assert list(edges.child) == [0, 1]

    def test_simple_case_unordered_children_and_intervals(self):
        #   2
        #  / \
        # 0   1
        nodes = io.StringIO(
            """\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       0       0               1.00000000000000
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.50000000      1.00000000      2       1
        2       0.00000000      0.50000000      2       1
        3       0.00000000      0.50000000      2       0
        1       0.50000000      1.00000000      2       0
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        assert all(edges.left) == 0
        assert all(edges.right) == 1
        assert list(edges.parent) == [2, 2]
        assert list(edges.child) == [0, 1]

    def test_squash_multiple_parents_and_children(self):
        #   4       5
        #  / \     / \
        # 0   1   2   3
        nodes = io.StringIO(
            """\
        id      is_sample   population      time
        0       1       0               0.00000000000000
        1       1       0               0.00000000000000
        2       1       0               0.00000000000000
        3       1       0               0.00000000000000
        4       0       0               1.00000000000000
        5       0       0               1.00000000000000
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        5       0.50000000      1.00000000      5       3
        6       0.50000000      1.00000000      5       2
        7       0.00000000      0.50000000      5       3
        8       0.00000000      0.50000000      5       2
        9       0.40000000      1.00000000      4       1
        10      0.00000000      0.40000000      4       1
        11      0.40000000      1.00000000      4       0
        12      0.00000000      0.40000000      4       0
        """
        )
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        edges = self.do_squash(ts)
        assert all(edges.left) == 0
        assert all(edges.right) == 1
        assert list(edges.parent) == [4, 4, 5, 5]
        assert list(edges.child) == [0, 1, 2, 3]

    def test_squash_overlapping_intervals(self):
        nodes = io.StringIO(
            """\
        id      is_sample   population      time
        0       1           0               0.0
        1       0           0               1.0
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.00            0.50            1       0
        1       0.40            0.80            1       0
        2       0.60            1.00            1       0
        """
        )
        with pytest.raises(tskit.LibraryError):
            tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def verify_slice_and_squash(self, ts):
        """
        Slices a tree sequence so that there are edge endpoints at
        all integer locations, then squashes these edges and verifies
        that the resulting edge table is the same as the input edge table.
        """
        sliced_edges = []
        # Create new sliced edge table.
        for e in ts.edges():
            left = e.left
            right = e.right

            if left == np.floor(left):
                r_left = np.ceil(left) + 1
            else:
                r_left = np.ceil(left)
            if right == np.floor(right):
                r_right = np.floor(right)
            else:
                r_right = np.floor(right) + 1

            new_range = [left]
            for r in np.arange(r_left, r_right):
                new_range.append(r)
            new_range.append(right)
            assert len(new_range) > 1

            # Add new edges to the list.
            for r in range(1, len(new_range)):
                new = tskit.Edge(new_range[r - 1], new_range[r], e.parent, e.child)
                sliced_edges.append(new)

        # Shuffle the edges and create a new edge table.
        random.shuffle(sliced_edges)
        sliced_table = tskit.EdgeTable()
        for e in sliced_edges:
            sliced_table.append(e)

        # Squash the edges and check against input table.
        sliced_table.squash()
        assert sliced_table == ts.tables.edges

    def test_sim_single_coalescent_tree(self):
        ts = msprime.simulate(20, random_seed=4, length=10)
        assert ts.num_trees == 1
        self.verify_slice_and_squash(ts)

    def test_sim_big_coalescent_trees(self):
        ts = msprime.simulate(20, recombination_rate=5, random_seed=4, length=10)
        assert ts.num_trees > 2
        self.verify_slice_and_squash(ts)


def squash_edges(ts):
    """
    Returns the edges in the tree sequence squashed.
    """
    t = ts.tables.nodes.time
    edges = list(ts.edges())
    edges.sort(key=lambda e: (t[e.parent], e.parent, e.child, e.left))
    if len(edges) == 0:
        return []

    squashed = []
    last_e = edges[0]
    for e in edges[1:]:
        condition = (
            e.parent != last_e.parent
            or e.child != last_e.child
            or e.left != last_e.right
        )
        if condition:
            squashed.append(last_e)
            last_e = e
        last_e.right = e.right
    squashed.append(last_e)
    return squashed


def reduce_topology(ts):
    """
    Returns a tree sequence with the minimal information required to represent
    the tree topologies at its sites. Uses a left-to-right algorithm.
    """
    tables = ts.dump_tables()
    edge_map = {}

    def add_edge(left, right, parent, child):
        new_edge = tskit.Edge(left, right, parent, child)
        if child not in edge_map:
            edge_map[child] = new_edge
        else:
            edge = edge_map[child]
            if edge.right == left and edge.parent == parent:
                # Squash
                edge.right = right
            else:
                tables.edges.append(edge)
                edge_map[child] = new_edge

    tables.edges.clear()

    edge_buffer = []
    first_site = True
    for tree in ts.trees():
        # print(tree.interval)
        # print(tree.draw(format="unicode"))
        if tree.num_sites > 0:
            sites = list(tree.sites())
            if first_site:
                x = 0
                # print("First site", sites)
                first_site = False
            else:
                x = sites[0].position
            # Flush the edge buffer.
            for left, parent, child in edge_buffer:
                add_edge(left, x, parent, child)
            # Add edges for each node in the tree.
            edge_buffer = []
            for root in tree.roots:
                for u in tree.nodes(root):
                    if u != root:
                        edge_buffer.append((x, tree.parent(u), u))
    # Add the final edges.
    for left, parent, child in edge_buffer:
        add_edge(left, tables.sequence_length, parent, child)
    # Flush the remaining edges to the table
    for edge in edge_map.values():
        tables.edges.append(edge)
    tables.sort()
    ts = tables.tree_sequence()
    # Now simplify to remove redundant nodes.
    return ts.simplify(map_nodes=True, filter_sites=False)


class TestReduceTopology:
    """
    Tests to ensure that reduce topology in simplify is equivalent to the
    reduce_topology function above.
    """

    def verify(self, ts):
        source_tables = ts.tables
        X = source_tables.sites.position
        position_count = {x: 0 for x in X}
        position_count[0] = 0
        position_count[ts.sequence_length] = 0
        mts, node_map = reduce_topology(ts)
        for edge in mts.edges():
            assert edge.left in position_count
            assert edge.right in position_count
            position_count[edge.left] += 1
            position_count[edge.right] += 1
        if ts.num_sites == 0:
            # We should have zero edges output.
            assert mts.num_edges == 0
        elif X[0] != 0:
            # The first site (if it's not zero) should be mapped to zero so
            # this never occurs in edges.
            assert position_count[X[0]] == 0

        minimised_trees = mts.trees()
        minimised_tree = next(minimised_trees)
        minimised_tree_sites = minimised_tree.sites()
        for tree in ts.trees():
            for site in tree.sites():
                minimised_site = next(minimised_tree_sites, None)
                if minimised_site is None:
                    minimised_tree = next(minimised_trees)
                    minimised_tree_sites = minimised_tree.sites()
                    minimised_site = next(minimised_tree_sites)
                assert site.position == minimised_site.position
                assert site.ancestral_state == minimised_site.ancestral_state
                assert site.metadata == minimised_site.metadata
                assert len(site.mutations) == len(minimised_site.mutations)

                for mutation, minimised_mutation in zip(
                    site.mutations, minimised_site.mutations
                ):
                    assert mutation.derived_state == minimised_mutation.derived_state
                    assert mutation.metadata == minimised_mutation.metadata
                    assert mutation.parent == minimised_mutation.parent
                    assert node_map[mutation.node] == minimised_mutation.node
            if tree.num_sites > 0:
                mapped_dict = {
                    node_map[u]: node_map[v] for u, v in tree.parent_dict.items()
                }
                assert mapped_dict == minimised_tree.parent_dict
        assert np.array_equal(ts.genotype_matrix(), mts.genotype_matrix())

        edges = list(mts.edges())
        squashed = squash_edges(mts)
        assert len(edges) == len(squashed)
        assert edges == squashed

        # Verify against simplify implementations.
        s = tests.Simplifier(
            ts, ts.samples(), reduce_to_site_topology=True, filter_sites=False
        )
        sts1, _ = s.simplify()
        sts2 = ts.simplify(reduce_to_site_topology=True, filter_sites=False)
        t1 = mts.tables
        for sts in [sts2, sts2]:
            t2 = sts.tables
            assert t1.nodes == t2.nodes
            assert t1.edges == t2.edges
            assert t1.sites == t2.sites
            assert t1.mutations == t2.mutations
            assert t1.populations == t2.populations
            assert t1.individuals == t2.individuals
        return mts

    def test_no_recombination_one_site(self):
        ts = msprime.simulate(15, random_seed=1)
        tables = ts.dump_tables()
        tables.sites.add_row(position=0.25, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        assert mts.num_trees == 1

    def test_simple_recombination_one_site(self):
        ts = msprime.simulate(15, random_seed=1, recombination_rate=2)
        tables = ts.dump_tables()
        tables.sites.add_row(position=0.25, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        assert mts.num_trees == 1

    def test_simple_recombination_fixed_sites(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2)
        tables = ts.dump_tables()
        for x in [0.25, 0.5, 0.75]:
            tables.sites.add_row(position=x, ancestral_state="0")
        self.verify(tables.tree_sequence())

    def get_integer_edge_ts(self, n, m):
        recombination_map = msprime.RecombinationMap.uniform_map(m, 1, num_loci=m)
        ts = msprime.simulate(n, random_seed=1, recombination_map=recombination_map)
        assert ts.num_trees > 1
        for edge in ts.edges():
            assert int(edge.left) == edge.left
            assert int(edge.right) == edge.right
        return ts

    def test_integer_edges_one_site(self):
        ts = self.get_integer_edge_ts(5, 10)
        tables = ts.dump_tables()
        tables.sites.add_row(position=1, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        assert mts.num_trees == 1

    def test_integer_edges_all_sites(self):
        ts = self.get_integer_edge_ts(5, 10)
        tables = ts.dump_tables()
        for x in range(10):
            tables.sites.add_row(position=x, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        assert mts.num_trees == ts.num_trees

    def test_simple_recombination_site_at_zero(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2)
        tables = ts.dump_tables()
        tables.sites.add_row(position=0, ancestral_state="0")
        mts = self.verify(tables.tree_sequence())
        assert mts.num_trees == 1

    def test_simple_recombination(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        self.verify(ts)

    def test_large_recombination(self):
        ts = msprime.simulate(
            25, random_seed=12, recombination_rate=5, mutation_rate=15
        )
        self.verify(ts)

    def test_no_recombination(self):
        ts = msprime.simulate(5, random_seed=1, mutation_rate=2)
        self.verify(ts)

    def test_no_mutation(self):
        ts = msprime.simulate(5, random_seed=1)
        self.verify(ts)

    def test_zero_sites(self):
        ts = msprime.simulate(5, random_seed=2)
        assert ts.num_sites == 0
        mts = ts.simplify(reduce_to_site_topology=True)
        assert mts.num_trees == 1
        assert mts.num_edges == 0

    def test_branch_sites(self):
        ts = msprime.simulate(15, random_seed=12, recombination_rate=2, length=10)
        ts = tsutil.insert_branch_sites(ts)
        self.verify(ts)

    def test_jiggled_samples(self):
        ts = msprime.simulate(8, random_seed=13, recombination_rate=2, length=10)
        ts = tsutil.jiggle_samples(ts)
        self.verify(ts)


def search_sorted(a, v):
    """
    Implementation of searchsorted based on binary search with the same
    semantics as numpy's searchsorted. Used as the basis of the C
    implementation which we use in the simplify algorithm.
    """
    upper = len(a)
    if upper == 0:
        return 0
    lower = 0
    while upper - lower > 1:
        mid = (upper + lower) // 2
        if v >= a[mid]:
            lower = mid
        else:
            upper = mid
    offset = 0
    if a[lower] < v:
        offset = 1
    return lower + offset


class TestSearchSorted:
    """
    Tests for the basic implementation of search_sorted.
    """

    def verify(self, a):
        a = np.array(a)
        start, end = a[0], a[-1]
        # Check random values.
        np.random.seed(43)
        for v in np.random.uniform(start, end, 10):
            assert search_sorted(a, v) == np.searchsorted(a, v)
        # Check equal values.
        for v in a:
            assert search_sorted(a, v) == np.searchsorted(a, v)
        # Check values outside bounds.
        for v in [start - 2, start - 1, end, end + 1, end + 2]:
            assert search_sorted(a, v) == np.searchsorted(a, v)

    def test_range(self):
        for j in range(1, 20):
            self.verify(range(j))

    def test_negative_range(self):
        for j in range(1, 20):
            self.verify(-1 * np.arange(j)[::-1])

    def test_random_unit_interval(self):
        np.random.seed(143)
        for size in range(1, 100):
            a = np.random.random(size=size)
            a.sort()
            self.verify(a)

    def test_random_interval(self):
        np.random.seed(143)
        for _ in range(10):
            interval = np.random.random(2) * 10
            interval.sort()
            a = np.random.uniform(*interval, size=100)
            a.sort()
            self.verify(a)

    def test_random_negative(self):
        np.random.seed(143)
        for _ in range(10):
            interval = np.random.random(2) * 5
            interval.sort()
            a = -1 * np.random.uniform(*interval, size=100)
            a.sort()
            self.verify(a)

    def test_edge_cases(self):
        for v in [0, 1]:
            assert search_sorted([], v) == np.searchsorted([], v)
            assert search_sorted([1], v) == np.searchsorted([1], v)


class TestDeleteSites:
    """
    Tests for the TreeSequence.delete_sites method
    """

    def ts_with_4_sites(self):
        ts = msprime.simulate(8, random_seed=3)
        tables = ts.dump_tables()
        tables.sites.set_columns(np.arange(0, 1, 0.25), *tskit.pack_strings(["G"] * 4))
        tables.mutations.add_row(site=1, node=ts.first().parent(0), derived_state="C")
        tables.mutations.add_row(site=1, node=0, derived_state="T", parent=0)
        tables.mutations.add_row(site=2, node=1, derived_state="A")
        return tables.tree_sequence()

    def test_remove_by_index(self):
        ts = self.ts_with_4_sites().delete_sites([])
        assert ts.num_sites == 4
        assert ts.num_mutations == 3
        ts = ts.delete_sites(2)
        assert ts.num_sites == 3
        assert ts.num_mutations == 2
        ts = ts.delete_sites([1, 2])
        assert ts.num_sites == 1
        assert ts.num_mutations == 0

    def test_remove_all(self):
        ts = self.ts_with_4_sites().delete_sites(range(4))
        assert ts.num_sites == 0
        assert ts.num_mutations == 0
        # should be OK to run on a siteless tree seq as no sites specified
        ts.delete_sites([])

    def test_remove_repeated_sites(self):
        ts = self.ts_with_4_sites()
        t1 = ts.delete_sites([0, 1], record_provenance=False)
        t2 = ts.delete_sites([0, 0, 1], record_provenance=False)
        t3 = ts.delete_sites([0, 0, 0, 1], record_provenance=False)
        assert t1.tables == t2.tables
        assert t1.tables == t3.tables

    def test_remove_different_orders(self):
        ts = self.ts_with_4_sites()
        t1 = ts.delete_sites([0, 1, 3], record_provenance=False)
        t2 = ts.delete_sites([0, 3, 1], record_provenance=False)
        t3 = ts.delete_sites([3, 0, 1], record_provenance=False)
        assert t1.tables == t2.tables
        assert t1.tables == t3.tables

    def test_remove_bad(self):
        ts = self.ts_with_4_sites()
        with pytest.raises(TypeError):
            ts.delete_sites(["1"])
        with pytest.raises(ValueError):
            ts.delete_sites(4)
        with pytest.raises(ValueError):
            ts.delete_sites(-5)

    def verify_removal(self, ts, remove_sites):
        tables = ts.dump_tables()
        tables.delete_sites(remove_sites)

        # Make sure we've computed the mutation parents properly.
        mutation_parent = tables.mutations.parent
        tables.compute_mutation_parents()
        assert np.array_equal(mutation_parent, tables.mutations.parent)

        tsd = tables.tree_sequence()
        assert tsd.num_sites == ts.num_sites - len(remove_sites)
        source_sites = [site for site in ts.sites() if site.id not in remove_sites]
        assert len(source_sites) == tsd.num_sites
        for s1, s2 in zip(source_sites, tsd.sites()):
            assert s1.position == s2.position
            assert s1.ancestral_state == s2.ancestral_state
            assert s1.metadata == s2.metadata
            assert len(s1.mutations) == len(s2.mutations)
            for m1, m2 in zip(s1.mutations, s2.mutations):
                assert m1.node == m2.node
                assert m1.derived_state == m2.derived_state
                assert m1.metadata == m2.metadata

        # Check we get the same genotype_matrix
        G1 = ts.genotype_matrix()
        G2 = tsd.genotype_matrix()
        keep = np.ones(ts.num_sites, dtype=bool)
        keep[remove_sites] = 0
        assert np.array_equal(G1[keep], G2)

    def test_simple_random_metadata(self):
        ts = msprime.simulate(10, mutation_rate=10, random_seed=2)
        ts = tsutil.add_random_metadata(ts)
        assert ts.num_mutations > 5
        self.verify_removal(ts, [1, 3])

    def test_simple_mixed_length_states(self):
        ts = msprime.simulate(10, random_seed=2, length=10)
        tables = ts.dump_tables()
        for j in range(10):
            tables.sites.add_row(j, "X" * j)
            tables.mutations.add_row(site=j, node=j, derived_state="X" * (j + 1))
        ts = tables.tree_sequence()
        self.verify_removal(ts, [9])

    def test_jukes_cantor_random_metadata(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 10, 1, seed=2)
        ts = tsutil.add_random_metadata(ts)
        assert ts.num_mutations > 10
        self.verify_removal(ts, [])
        self.verify_removal(ts, [0, 2, 4, 8])
        self.verify_removal(ts, range(5))

    def test_jukes_cantor_many_mutations(self):
        ts = msprime.simulate(2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 10, mu=10, seed=2)
        assert ts.num_mutations > 100
        self.verify_removal(ts, [1, 3, 5, 7])
        self.verify_removal(ts, [1])
        self.verify_removal(ts, [9])

    def test_jukes_cantor_one_site(self):
        ts = msprime.simulate(5, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 1, mu=10, seed=2)
        assert ts.num_mutations > 10
        self.verify_removal(ts, [])
        self.verify_removal(ts, [0])


class TestKeepSingleInterval(unittest.TestCase):
    """
    Tests for cutting up tree sequences along the genome.
    """

    def test_slice_unchanged(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        tables = ts.dump_tables()
        tables.edges.packset_metadata([b"edge {i}" for i in range(ts.num_edges)])
        ts1 = tables.tree_sequence()
        ts2 = ts1.keep_intervals([[0, 1]], simplify=False, record_provenance=False)
        ts1.tables.assert_equals(ts2.tables)

    def test_slice_by_tree_positions(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        breakpoints = list(ts.breakpoints())

        # Keep the last 3 trees (from 4th last breakpoint onwards)
        ts_sliced = ts.keep_intervals([[breakpoints[-4], ts.sequence_length]])
        assert ts_sliced.num_trees == 4
        assert ts_sliced.num_edges < ts.num_edges
        self.assertAlmostEqual(ts_sliced.sequence_length, 1.0)
        last_3_mutations = 0
        for tree_index in range(-3, 0):
            last_3_mutations += ts.at_index(tree_index).num_mutations
        assert ts_sliced.num_mutations == last_3_mutations

        # Keep the first 3 trees
        ts_sliced = ts.keep_intervals([[0, breakpoints[3]]])
        assert ts_sliced.num_trees == 4
        assert ts_sliced.num_edges < ts.num_edges
        self.assertAlmostEqual(ts_sliced.sequence_length, 1)
        first_3_mutations = 0
        for tree_index in range(0, 3):
            first_3_mutations += ts.at_index(tree_index).num_mutations
        assert ts_sliced.num_mutations == first_3_mutations

        # Slice out the middle
        ts_sliced = ts.keep_intervals([[breakpoints[3], breakpoints[-4]]])
        assert ts_sliced.num_trees == ts.num_trees - 4
        assert ts_sliced.num_edges < ts.num_edges
        self.assertAlmostEqual(ts_sliced.sequence_length, 1.0)
        assert (
            ts_sliced.num_mutations
            == ts.num_mutations - first_3_mutations - last_3_mutations
        )

    def test_slice_by_position(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        ts_sliced = ts.keep_intervals([[0.4, 0.6]])
        positions = ts.tables.sites.position
        assert ts_sliced.num_sites == np.sum((positions >= 0.4) & (positions < 0.6))

    def test_slice_unsimplified(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        ts_sliced = ts.keep_intervals([[0.4, 0.6]], simplify=True)
        assert ts.num_nodes != ts_sliced.num_nodes
        self.assertAlmostEqual(ts_sliced.sequence_length, 1.0)
        ts_sliced = ts.keep_intervals([[0.4, 0.6]], simplify=False)
        assert ts.num_nodes == ts_sliced.num_nodes
        self.assertAlmostEqual(ts_sliced.sequence_length, 1.0)

    def test_slice_coordinates(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        ts_sliced = ts.keep_intervals([[0.4, 0.6]])
        self.assertAlmostEqual(ts_sliced.sequence_length, 1)
        assert ts_sliced.num_trees != ts.num_trees
        assert ts_sliced.at_index(0).total_branch_length == 0
        assert ts_sliced.at(0).total_branch_length == 0
        assert ts_sliced.at(0.399).total_branch_length == 0
        assert ts_sliced.at(0.4).total_branch_length != 0
        assert ts_sliced.at(0.5).total_branch_length != 0
        assert ts_sliced.at(0.599).total_branch_length != 0
        assert ts_sliced.at(0.6).total_branch_length == 0
        assert ts_sliced.at(0.999).total_branch_length == 0
        assert ts_sliced.at_index(-1).total_branch_length == 0

    def test_slice_migrations(self):
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 0.05], [0.05, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            record_migrations=True,
            recombination_rate=2,
            random_seed=1,
        )
        tables = ts.dump_tables()
        tables.migrations.packset_metadata(
            [b"migration {i}" for i in range(ts.num_migrations)]
        )
        ts = tables.tree_sequence()

        ts_sliced = ts.keep_intervals([[0, 1]], simplify=False)
        assert ts.tables.migrations == ts_sliced.tables.migrations
        ts_sliced = ts.keep_intervals([[0, 0.5]], simplify=False)
        assert np.max(ts_sliced.tables.migrations.right) <= 0.5
        assert ts.num_migrations > ts_sliced.num_migrations

        ts_sliced = ts.keep_intervals([[0.5, 1]], simplify=False)
        assert np.max(ts_sliced.tables.migrations.left) >= 0.5
        assert ts.num_migrations > ts_sliced.num_migrations

        ts_sliced = ts.keep_intervals([[0.4, 0.6]], simplify=False)
        assert np.max(ts_sliced.tables.migrations.right) <= 0.6
        assert np.max(ts_sliced.tables.migrations.left) >= 0.4
        assert ts.num_migrations > ts_sliced.num_migrations


class TestKeepIntervals(TopologyTestCase):
    """
    Tests for keep_intervals operation, where we slice out multiple disjoint
    intervals concurrently.
    """

    def example_intervals(self, tables):
        L = tables.sequence_length
        yield []
        yield [(0, L)]
        yield [(0, L / 2), (L / 2, L)]
        yield [(0, 0.25 * L), (0.75 * L, L)]
        yield [(0.25 * L, L)]
        yield [(0.25 * L, 0.5 * L)]
        yield [(0.25 * L, 0.5 * L), (0.75 * L, 0.8 * L)]

    def do_keep_intervals(
        self, tables, intervals, simplify=True, record_provenance=True
    ):
        t1 = tables.copy()
        simple_keep_intervals(t1, intervals, simplify, record_provenance)
        t2 = tables.copy()
        t2.keep_intervals(intervals, simplify, record_provenance)
        t1.assert_equals(t2, ignore_timestamps=True)
        return t2

    def test_migration_error(self):
        # keep_intervals should fail if simplify=True (default)
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 0.05], [0.05, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            record_migrations=True,
            recombination_rate=2,
            random_seed=1,
        )
        with pytest.raises(tskit.LibraryError):
            ts.dump_tables().keep_intervals([[0, 1]])

    def test_bad_intervals(self):
        tables = tskit.TableCollection(10)
        bad_intervals = [[[1, 1]], [[-1, 0]], [[0, 11]], [[0, 5], [4, 6]]]
        for intervals in bad_intervals:
            with pytest.raises(ValueError):
                tables.keep_intervals(intervals)
            with pytest.raises(ValueError):
                tables.delete_intervals(intervals)

    def test_one_interval(self):
        ts = msprime.simulate(
            10, random_seed=self.random_seed, recombination_rate=2, mutation_rate=2
        )
        tables = ts.dump_tables()
        intervals = [(0.3, 0.7)]
        for simplify in (True, False):
            for rec_prov in (True, False):
                self.do_keep_intervals(tables, intervals, simplify, rec_prov)

    def test_two_intervals(self):
        ts = msprime.simulate(
            10, random_seed=self.random_seed, recombination_rate=2, mutation_rate=2
        )
        tables = ts.dump_tables()
        intervals = [(0.1, 0.2), (0.8, 0.9)]
        for simplify in (True, False):
            for rec_prov in (True, False):
                self.do_keep_intervals(tables, intervals, simplify, rec_prov)

    def test_ten_intervals(self):
        ts = msprime.simulate(
            10, random_seed=self.random_seed, recombination_rate=2, mutation_rate=2
        )
        tables = ts.dump_tables()
        intervals = [(x, x + 0.05) for x in np.arange(0.0, 1.0, 0.1)]
        for simplify in (True, False):
            for rec_prov in (True, False):
                self.do_keep_intervals(tables, intervals, simplify, rec_prov)

    def test_hundred_intervals(self):
        ts = msprime.simulate(
            10, random_seed=self.random_seed, recombination_rate=2, mutation_rate=2
        )
        tables = ts.dump_tables()
        intervals = [(x, x + 0.005) for x in np.arange(0.0, 1.0, 0.01)]
        for simplify in (True, False):
            for rec_prov in (True, False):
                self.do_keep_intervals(tables, intervals, simplify, rec_prov)

    def test_regular_intervals(self):
        ts = msprime.simulate(
            3, random_seed=1234, recombination_rate=2, mutation_rate=2
        )
        tables = ts.dump_tables()
        eps = 0.0125
        for num_intervals in range(2, 10):
            breaks = np.linspace(0, ts.sequence_length, num=num_intervals)
            intervals = [(x, x + eps) for x in breaks[:-1]]
            self.do_keep_intervals(tables, intervals)

    def test_no_edges_sites(self):
        tables = tskit.TableCollection(1.0)
        tables.sites.add_row(0.1, "A")
        tables.sites.add_row(0.2, "T")
        for intervals in self.example_intervals(tables):
            assert len(tables.sites) == 2
            diced = self.do_keep_intervals(tables, intervals)
            assert diced.sequence_length == 1
            assert len(diced.edges) == 0
            assert len(diced.sites) == 0

    def verify(self, tables):
        for intervals in self.example_intervals(tables):
            for simplify in [True, False]:
                self.do_keep_intervals(tables, intervals, simplify=simplify)

    def test_empty_tables(self):
        tables = tskit.TableCollection(1.0)
        self.verify(tables)

    def test_single_tree_jukes_cantor(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        self.verify(ts.tables)

    def test_single_tree_multichar_mutations(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.insert_multichar_mutations(ts)
        self.verify(ts.tables)

    def test_many_trees_infinite_sites(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        assert ts.num_trees > 2
        self.verify(ts.tables)

    def test_many_trees_sequence_length_infinite_sites(self):
        for L in [0.5, 1.5, 3.3333]:
            ts = msprime.simulate(
                6, length=L, recombination_rate=2, mutation_rate=1, random_seed=1
            )
            self.verify(ts.tables)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            4,
            5,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=10,
        )
        tables.sort()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.05, random_seed=234)
        assert ts.num_sites > 0
        self.verify(ts.tables)

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            6, 5, seed=3, deep_history=True, initial_generation_samples=True, num_loci=2
        )
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.08, random_seed=2)
        assert ts.num_sites > 0
        self.verify(ts.tables)

    def test_wright_fisher_initial_generation_no_deep_history(self):
        tables = wf.wf_sim(
            7,
            15,
            seed=202,
            deep_history=False,
            initial_generation_samples=True,
            num_loci=5,
        )
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.2, random_seed=2)
        assert ts.num_sites > 0
        self.verify(ts.tables)

    def test_wright_fisher_unsimplified_multiple_roots(self):
        tables = wf.wf_sim(
            8,
            15,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=20,
        )
        tables.sort()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.006, random_seed=2)
        assert ts.num_sites > 0
        self.verify(ts.tables)

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            9,
            10,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=5,
        )
        tables.sort()
        ts = tables.tree_sequence().simplify()
        ts = msprime.mutate(ts, rate=0.2, random_seed=1234)
        assert ts.num_sites > 0
        self.verify(ts.tables)


class TestKeepDeleteIntervalsExamples:
    """
    Simple examples of keep/delete intervals at work.
    """

    def test_tables_single_tree_keep_middle(self):
        ts = msprime.simulate(10, random_seed=2)
        t_keep = ts.dump_tables()
        t_keep.keep_intervals([[0.25, 0.5]], record_provenance=False)
        t_delete = ts.dump_tables()
        t_delete.delete_intervals([[0, 0.25], [0.5, 1.0]], record_provenance=False)
        assert t_keep == t_delete

    def test_tables_single_tree_delete_middle(self):
        ts = msprime.simulate(10, random_seed=2)
        t_keep = ts.dump_tables()
        t_keep.delete_intervals([[0.25, 0.5]], record_provenance=False)
        t_delete = ts.dump_tables()
        t_delete.keep_intervals([[0, 0.25], [0.5, 1.0]], record_provenance=False)
        assert t_keep == t_delete

    def test_ts_single_tree_keep_middle(self):
        ts = msprime.simulate(10, random_seed=2)
        ts_keep = ts.keep_intervals([[0.25, 0.5]], record_provenance=False)
        ts_delete = ts.delete_intervals(
            [[0, 0.25], [0.5, 1.0]], record_provenance=False
        )
        assert ts_keep == ts_delete

    def test_ts_single_tree_delete_middle(self):
        ts = msprime.simulate(10, random_seed=2)
        ts_keep = ts.delete_intervals([[0.25, 0.5]], record_provenance=False)
        ts_delete = ts.keep_intervals([[0, 0.25], [0.5, 1.0]], record_provenance=False)
        assert ts_keep == ts_delete

    def test_ts_migrations(self):
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 0.05], [0.05, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            record_migrations=True,
            recombination_rate=2,
            random_seed=1,
        )
        ts_keep = ts.delete_intervals(
            [[0.25, 0.5]], record_provenance=False, simplify=False
        )
        ts_delete = ts.keep_intervals(
            [[0, 0.25], [0.5, 1.0]], record_provenance=False, simplify=False
        )
        assert ts_keep == ts_delete


class TestTrim(unittest.TestCase):
    """
    Test the trimming functionality
    """

    def add_mutations(self, ts, position, ancestral_state, derived_states, nodes):
        """
        Create a site at the specified position and assign mutations to the specified
        nodes (could be sequential mutations)
        """
        tables = ts.dump_tables()
        site = tables.sites.add_row(position, ancestral_state)
        for state, node in zip(derived_states, nodes):
            tables.mutations.add_row(site, node, state)
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        return tables.tree_sequence()

    def verify_sites(self, source_tree, trimmed_tree, position_offset):
        source_sites = list(source_tree.sites())
        trimmed_sites = list(trimmed_tree.sites())
        assert len(source_sites) == len(trimmed_sites)
        for source_site, trimmed_site in zip(source_sites, trimmed_sites):
            self.assertAlmostEqual(
                source_site.position, position_offset + trimmed_site.position
            )
            assert source_site.ancestral_state == trimmed_site.ancestral_state
            assert source_site.metadata == trimmed_site.metadata
            assert len(source_site.mutations) == len(trimmed_site.mutations)
            for source_mut, trimmed_mut in zip(
                source_site.mutations, trimmed_site.mutations
            ):
                assert source_mut.node == trimmed_mut.node
                assert source_mut.derived_state == trimmed_mut.derived_state
                assert source_mut.metadata == trimmed_mut.metadata
                # mutation.parent id may have changed after deleting redundant mutations
                if source_mut.parent == trimmed_mut.parent == tskit.NULL:
                    pass
                else:
                    assert (
                        source_tree.tree_sequence.mutation(source_mut.parent).node
                        == trimmed_tree.tree_sequence.mutation(trimmed_mut.parent).node
                    )

    def verify_ltrim(self, source_ts, trimmed_ts):
        deleted_span = source_ts.first().span
        self.assertAlmostEqual(
            source_ts.sequence_length, trimmed_ts.sequence_length + deleted_span
        )
        assert source_ts.num_trees == trimmed_ts.num_trees + 1
        for j in range(trimmed_ts.num_trees):
            source_tree = source_ts.at_index(j + 1)
            trimmed_tree = trimmed_ts.at_index(j)
            assert source_tree.parent_dict == trimmed_tree.parent_dict
            self.assertAlmostEqual(source_tree.span, trimmed_tree.span)
            self.assertAlmostEqual(
                source_tree.interval.left, trimmed_tree.interval.left + deleted_span
            )
            self.verify_sites(source_tree, trimmed_tree, deleted_span)

    def verify_rtrim(self, source_ts, trimmed_ts):
        deleted_span = source_ts.last().span
        self.assertAlmostEqual(
            source_ts.sequence_length, trimmed_ts.sequence_length + deleted_span
        )
        assert source_ts.num_trees == trimmed_ts.num_trees + 1
        for j in range(trimmed_ts.num_trees):
            source_tree = source_ts.at_index(j)
            trimmed_tree = trimmed_ts.at_index(j)
            assert source_tree.parent_dict == trimmed_tree.parent_dict
            assert source_tree.interval == trimmed_tree.interval
            self.verify_sites(source_tree, trimmed_tree, 0)

    def clear_left_mutate(self, ts, left, num_sites):
        """
        Clear the edges from a tree sequence left of the specified coordinate
        and add in num_sites regularly spaced sites into the cleared region.
        """
        new_ts = ts.delete_intervals([[0.0, left]])
        for j, x in enumerate(np.linspace(0, left, num_sites, endpoint=False)):
            new_ts = self.add_mutations(new_ts, x, "A" * j, ["T"] * j, range(j + 1))
        return new_ts

    def clear_right_mutate(self, ts, right, num_sites):
        """
        Clear the edges from a tree sequence right of the specified coordinate
        and add in num_sites regularly spaced sites into the cleared region.
        """
        new_ts = ts.delete_intervals([[right, ts.sequence_length]])
        for j, x in enumerate(
            np.linspace(right, ts.sequence_length, num_sites, endpoint=False)
        ):
            new_ts = self.add_mutations(new_ts, x, "A" * j, ["T"] * j, range(j + 1))
        return new_ts

    def clear_left_right_234(self, left, right):
        """
        Clear edges to left and right and add 2 mutations at the same site into the left
        cleared region, 3 at the same site into the untouched region, and 4 into the
        right cleared region.
        """
        assert 0.0 < left < right < 1.0
        ts = msprime.simulate(10, recombination_rate=10, random_seed=2)
        left_pos = np.mean([0.0, left])
        left_root = ts.at(left_pos).root
        mid_pos = np.mean([left, right])
        mid_root = ts.at(mid_pos).root
        right_pos = np.mean([right, ts.sequence_length])
        right_root = ts.at(right_pos).root
        # Clear
        ts = ts.keep_intervals([[left, right]], simplify=False)
        ts = self.add_mutations(ts, left_pos, "A", ["T", "C"], [left_root, 0])
        ts = self.add_mutations(ts, mid_pos, "T", ["A", "C", "G"], [mid_root, 0, 1])
        ts = self.add_mutations(
            ts, right_pos, "X", ["T", "C", "G", "A"], [right_root, 0, 1, 2]
        )
        assert np.min(ts.tables.edges.left) != 0
        assert ts.num_mutations == 9
        assert ts.num_sites == 3
        return ts

    def migration_sim(self):
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 0.05], [0.05, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            record_migrations=True,
            recombination_rate=2,
            random_seed=1,
        )
        return ts

    def test_ltrim_single_tree(self):
        ts = msprime.simulate(10, mutation_rate=12, random_seed=2)
        ts = self.clear_left_mutate(ts, 0.5, 10)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_single_tree_no_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = self.clear_left_mutate(ts, 0.5, 0)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_single_tree_tiny_left(self):
        ts = msprime.simulate(10, mutation_rate=12, random_seed=2)
        ts = self.clear_left_mutate(ts, 1e-200, 10)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_many_trees(self):
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=12, random_seed=2
        )
        ts = self.clear_left_mutate(ts, 0.5, 10)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_many_trees_left_min(self):
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=12, random_seed=2
        )
        ts = self.clear_left_mutate(ts, sys.float_info.min, 10)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_many_trees_left_epsilon(self):
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=12, random_seed=2
        )
        ts = self.clear_left_mutate(ts, sys.float_info.epsilon, 0)
        self.verify_ltrim(ts, ts.ltrim())

    def test_ltrim_empty(self):
        ts = msprime.simulate(2, random_seed=2)
        ts = ts.delete_intervals([[0, 1]])
        with pytest.raises(ValueError):
            ts.ltrim()

    def test_ltrim_multiple_mutations(self):
        ts = self.clear_left_right_234(0.1, 0.5)
        trimmed_ts = ts.ltrim()
        self.assertAlmostEqual(trimmed_ts.sequence_length, 0.9)
        assert trimmed_ts.num_sites == 2
        assert trimmed_ts.num_mutations == 7  # We should have deleted 2
        assert np.min(trimmed_ts.tables.edges.left) == 0
        self.verify_ltrim(ts, trimmed_ts)

    def test_ltrim_migrations(self):
        ts = self.migration_sim()
        ts = ts.delete_intervals([[0, 0.1]], simplify=False)
        trimmed_ts = ts.ltrim()
        assert np.array_equal(
            trimmed_ts.tables.migrations.left, ts.tables.migrations.left - 0.1
        )
        assert np.array_equal(
            trimmed_ts.tables.migrations.right, ts.tables.migrations.right - 0.1
        )

    def test_rtrim_single_tree(self):
        ts = msprime.simulate(10, mutation_rate=12, random_seed=2)
        ts = self.clear_right_mutate(ts, 0.5, 10)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_single_tree_no_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = self.clear_right_mutate(ts, 0.5, 0)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_single_tree_tiny_left(self):
        ts = msprime.simulate(10, mutation_rate=12, random_seed=2)
        ts = self.clear_right_mutate(ts, 1e-200, 10)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_many_trees(self):
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=12, random_seed=2
        )
        ts = self.clear_right_mutate(ts, 0.5, 10)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_many_trees_left_min(self):
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=12, random_seed=2
        )
        ts = self.clear_right_mutate(ts, sys.float_info.min, 10)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_many_trees_left_epsilon(self):
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=12, random_seed=2
        )
        ts = self.clear_right_mutate(ts, sys.float_info.epsilon, 0)
        self.verify_rtrim(ts, ts.rtrim())

    def test_rtrim_empty(self):
        ts = msprime.simulate(2, random_seed=2)
        ts = ts.delete_intervals([[0, 1]])
        with pytest.raises(ValueError):
            ts.rtrim()

    def test_rtrim_multiple_mutations(self):
        ts = self.clear_left_right_234(0.1, 0.5)
        trimmed_ts = ts.rtrim()
        self.assertAlmostEqual(trimmed_ts.sequence_length, 0.5)
        assert trimmed_ts.num_sites == 2
        assert trimmed_ts.num_mutations == 5  # We should have deleted 4
        assert (
            np.max(trimmed_ts.tables.edges.right) == trimmed_ts.tables.sequence_length
        )
        self.verify_rtrim(ts, trimmed_ts)

    def test_rtrim_migrations(self):
        ts = self.migration_sim()
        ts = ts.delete_intervals([[0.9, 1]], simplify=False)
        trimmed_ts = ts.rtrim()
        trimmed_rights = trimmed_ts.tables.migrations.right
        assert np.max(trimmed_rights) == 0.9

    def test_trim_multiple_mutations(self):
        ts = self.clear_left_right_234(0.1, 0.5)
        trimmed_ts = ts.trim()
        self.assertAlmostEqual(trimmed_ts.sequence_length, 0.4)
        assert trimmed_ts.num_mutations == 3
        assert trimmed_ts.num_sites == 1
        assert np.min(trimmed_ts.tables.edges.left) == 0
        assert (
            np.max(trimmed_ts.tables.edges.right) == trimmed_ts.tables.sequence_length
        )

    def test_trims_no_effect(self):
        # Deleting from middle should have no effect on any trim function
        ts = msprime.simulate(10, recombination_rate=2, mutation_rate=50, random_seed=2)
        ts = ts.delete_intervals([[0.1, 0.5]])
        trimmed_ts = ts.ltrim(record_provenance=False)
        assert ts == trimmed_ts
        trimmed_ts = ts.rtrim(record_provenance=False)
        assert ts == trimmed_ts
        trimmed_ts = ts.trim(record_provenance=False)
        assert ts == trimmed_ts

    def test_failure_with_migrations(self):
        # All trim functions fail if migrations extend further than rightmost or
        # leftmost edges
        ts = msprime.simulate(10, recombination_rate=2, random_seed=2)
        ts = ts.keep_intervals([[0.1, 0.5]])
        tables = ts.dump_tables()
        tables.migrations.add_row(0, 1, 0, 0, 0, 0)
        ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            ts.ltrim()
        with pytest.raises(ValueError):
            ts.rtrim()
        with pytest.raises(ValueError):
            ts.trim()

    def test_reference_sequence(self):
        # Test that we fail if there is a reference sequence
        tables = tskit.TableCollection(3.0)
        tables.reference_sequence.data = "ABC"
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="reference sequence"):
            ts.ltrim()
        with pytest.raises(ValueError, match="reference sequence"):
            ts.rtrim()
        with pytest.raises(ValueError, match="reference sequence"):
            ts.trim()


class TestShift:
    """
    Test the shift functionality
    """

    @pytest.mark.parametrize("shift", [-0.5, 0, 0.5])
    def test_shift(self, shift):
        ts = tskit.Tree.generate_comb(2, span=2).tree_sequence
        tables = ts.dump_tables()
        tables.delete_intervals([[0, 1]], simplify=False)
        tables.sites.add_row(1.5, "A")
        ts = tables.tree_sequence()
        ts = ts.shift(shift)
        assert ts.sequence_length == 2 + shift
        assert np.min(ts.tables.edges.left) == 1 + shift
        assert np.max(ts.tables.edges.right) == 2 + shift
        assert np.all(ts.tables.sites.position == 1.5 + shift)
        assert len(list(ts.trees())) == ts.num_trees

    def test_sequence_length(self):
        ts = tskit.Tree.generate_comb(2).tree_sequence
        ts = ts.shift(1, sequence_length=3)
        assert ts.sequence_length == 3
        ts = ts.shift(-1, sequence_length=1)
        assert ts.sequence_length == 1

    def test_empty(self):
        empty_ts = tskit.TableCollection(1.0).tree_sequence()
        empty_ts = empty_ts.shift(1)
        assert empty_ts.sequence_length == 2
        empty_ts = empty_ts.shift(-1.5)
        assert empty_ts.sequence_length == 0.5
        assert empty_ts.num_nodes == 0

    def test_migrations(self):
        tables = tskit.Tree.generate_comb(2, span=2).tree_sequence.dump_tables()
        tables.populations.add_row()
        tables.migrations.add_row(0, 1, 0, 0, 0, 0)
        ts = tables.tree_sequence().shift(10)
        assert np.all(ts.tables.migrations.left == 10)
        assert np.all(ts.tables.migrations.right == 11)

    def test_provenance(self):
        ts = tskit.Tree.generate_comb(2).tree_sequence
        ts = ts.shift(1, record_provenance=False)
        params = json.loads(ts.provenance(-1).record)["parameters"]
        assert params["command"] != "shift"
        ts = ts.shift(1, sequence_length=9)
        params = json.loads(ts.provenance(-1).record)["parameters"]
        assert params["command"] == "shift"
        assert params["value"] == 1
        assert params["sequence_length"] == 9

    def test_too_negative(self):
        ts = tskit.Tree.generate_comb(2).tree_sequence
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_BAD_SEQUENCE_LENGTH"):
            ts.shift(-1)

    def test_bad_seq_len(self):
        ts = tskit.Tree.generate_comb(2).tree_sequence
        with pytest.raises(
            tskit.LibraryError, match="TSK_ERR_RIGHT_GREATER_SEQ_LENGTH"
        ):
            ts.shift(1, sequence_length=1)

    def test_reference_sequence(self):
        # Test that we fail if there is a reference sequence
        tables = tskit.TableCollection(3.0)
        tables.reference_sequence.data = "ABC"
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="reference sequence"):
            ts.shift(1)


class TestConcatenate:
    def test_simple(self):
        ts1 = tskit.Tree.generate_comb(5, span=2).tree_sequence
        ts2 = tskit.Tree.generate_balanced(5, arity=3, span=3).tree_sequence
        assert ts1.num_samples == ts2.num_samples
        assert ts1.num_nodes != ts2.num_nodes
        joint_ts = ts1.concatenate(ts2)
        assert joint_ts.num_nodes == ts1.num_nodes + ts2.num_nodes - 5
        assert joint_ts.sequence_length == ts1.sequence_length + ts2.sequence_length
        assert joint_ts.num_samples == ts1.num_samples
        ts3 = joint_ts.delete_intervals([[2, 5]]).rtrim()
        # Have to simplify here, to remove the redundant nodes
        assert ts3.equals(ts1.simplify(), ignore_provenance=True)
        ts4 = joint_ts.delete_intervals([[0, 2]]).ltrim()
        assert ts4.equals(ts2.simplify(), ignore_provenance=True)

    def test_multiple(self):
        np.random.seed(42)
        ts3 = [
            tskit.Tree.generate_comb(5, span=2).tree_sequence,
            tskit.Tree.generate_balanced(5, arity=3, span=3).tree_sequence,
            tskit.Tree.generate_star(5, span=5).tree_sequence,
        ]
        for i in range(1, len(ts3)):
            # shuffle the sample nodes so they don't have the same IDs
            ts3[i] = ts3[i].subset(np.random.permutation(ts3[i].num_nodes))
        assert not np.all(ts3[0].samples() == ts3[1].samples())
        assert not np.all(ts3[0].samples() == ts3[2].samples())
        assert not np.all(ts3[1].samples() == ts3[2].samples())
        ts = ts3[0].concatenate(*ts3[1:])
        assert ts.sequence_length == sum([t.sequence_length for t in ts3])
        assert ts.num_nodes - ts.num_samples == sum(
            [t.num_nodes - t.num_samples for t in ts3]
        )
        assert np.all(ts.samples() == ts3[0].samples())

    def test_empty(self):
        empty_ts = tskit.TableCollection(10).tree_sequence()
        ts = empty_ts.concatenate(empty_ts, empty_ts, empty_ts)
        assert ts.num_nodes == 0
        assert ts.sequence_length == 40

    def test_samples_at_end(self):
        ts1 = tskit.Tree.generate_comb(5, span=2).tree_sequence
        ts2 = tskit.Tree.generate_balanced(5, arity=3, span=3).tree_sequence
        # reverse the node order
        ts1 = ts1.subset(np.arange(ts1.num_nodes)[::-1])
        assert ts1.num_samples == ts2.num_samples
        assert np.all(ts1.samples() != ts2.samples())
        joint_ts = ts1.concatenate(ts2)
        assert joint_ts.num_samples == ts1.num_samples
        assert np.all(joint_ts.samples() == ts1.samples())

    def test_internal_samples(self):
        tables = tskit.Tree.generate_comb(4, span=2).tree_sequence.dump_tables()
        nodes_flags = tables.nodes.flags
        nodes_flags[:] = tskit.NODE_IS_SAMPLE
        nodes_flags[-1] = 0  # Only root is not a sample
        tables.nodes.flags = nodes_flags
        ts = tables.tree_sequence()
        joint_ts = ts.concatenate(ts)
        assert joint_ts.num_samples == ts.num_samples
        assert joint_ts.num_nodes == ts.num_nodes + 1
        assert joint_ts.sequence_length == ts.sequence_length * 2

    def test_some_shared_samples(self):
        ts1 = tskit.Tree.generate_comb(4, span=2).tree_sequence
        ts2 = tskit.Tree.generate_balanced(8, arity=3, span=3).tree_sequence
        shared = np.full(ts2.num_nodes, tskit.NULL)
        shared[0] = 1
        shared[1] = 0
        joint_ts = ts1.concatenate(ts2, node_mappings=[shared])
        assert joint_ts.sequence_length == ts1.sequence_length + ts2.sequence_length
        assert joint_ts.num_samples == ts1.num_samples + ts2.num_samples - 2
        assert joint_ts.num_nodes == ts1.num_nodes + ts2.num_nodes - 2

    def test_provenance(self):
        ts = tskit.Tree.generate_comb(2).tree_sequence
        ts = ts.concatenate(ts, record_provenance=False)
        params = json.loads(ts.provenance(-1).record)["parameters"]
        assert params["command"] != "concatenate"

        ts = ts.concatenate(ts)
        params = json.loads(ts.provenance(-1).record)["parameters"]
        assert params["command"] == "concatenate"

    def test_unequal_samples(self):
        ts1 = tskit.Tree.generate_comb(5, span=2).tree_sequence
        ts2 = tskit.Tree.generate_balanced(4, arity=3, span=3).tree_sequence
        with pytest.raises(ValueError, match="must have the same number of samples"):
            ts1.concatenate(ts2)

    @pytest.mark.skip(
        reason="union bug: https://github.com/tskit-dev/tskit/issues/3168"
    )
    def test_duplicate_ts(self):
        ts1 = tskit.Tree.generate_comb(3, span=4).tree_sequence
        ts = ts1.keep_intervals([[0, 1]]).trim()  # a quarter of the original
        nm = np.arange(ts.num_nodes)  # all nodes identical
        ts2 = ts.concatenate(ts, ts, ts, node_mappings=[nm] * 3, add_populations=False)
        ts2 = ts2.simplify()  # squash the edges
        assert ts1.equals(ts2, ignore_provenance=True)

    def test_node_mappings_bad_len(self):
        ts = tskit.Tree.generate_comb(3, span=2).tree_sequence
        nm = np.arange(ts.num_nodes)
        with pytest.raises(ValueError, match="same number of node_mappings"):
            ts.concatenate(ts, ts, ts, node_mappings=[nm, nm])


class TestMissingData:
    """
    Test various aspects of missing data functionality
    """

    # TODO tests for missing data currently sparse: more tests should go here

    def ts_missing_middle(self):
        # Simple ts with sample 0 missing a middle section
        ts = msprime.simulate(4, mutation_rate=1, recombination_rate=4, random_seed=2)
        tables = ts.dump_tables()
        tables.edges.clear()
        # mark the middle as missing
        for e in ts.tables.edges:
            if e.child == 0:
                if e.left == 0.0:
                    missing_from = e.right
                elif e.right == 1.0:
                    missing_to = e.left
                else:
                    continue  # omit this edge => node is isolated
            tables.edges.append(e)
        # Check we have non-missing to L & R
        assert 0.0 < missing_from < 1.0
        assert 0.0 < missing_to < 1.0
        return tables.tree_sequence(), missing_from, missing_to

    def test_is_isolated(self):
        ts, missing_from, missing_to = self.ts_missing_middle()
        for tree in ts.trees():
            if tree.interval.right > missing_from and tree.interval.left < missing_to:
                assert tree.is_isolated(0)
                assert not tree.is_isolated(1)
            else:
                assert not tree.is_isolated(0)
                assert not tree.is_isolated(1)
            # A non-sample node is isolated if not in the tree
            tree_nodes = set(tree.nodes())
            for nonsample_node in np.setdiff1d(np.arange(ts.num_nodes), ts.samples()):
                if nonsample_node in tree_nodes:
                    assert not tree.is_isolated(nonsample_node)
                else:
                    assert tree.is_isolated(nonsample_node)

    def test_is_isolated_bad(self):
        ts, missing_from, missing_to = self.ts_missing_middle()
        for tree in ts.trees():
            with pytest.raises(ValueError):
                tree.is_isolated(tskit.NULL)
            with pytest.raises(ValueError):
                tree.is_isolated(ts.num_nodes + 1)
            with pytest.raises(ValueError):
                tree.is_isolated(-2)
            with pytest.raises(TypeError):
                tree.is_isolated(None)
            with pytest.raises(TypeError):
                tree.is_isolated("abc")
            with pytest.raises(TypeError):
                tree.is_isolated(1.1)
