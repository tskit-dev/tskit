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
Test cases for the high level interface to tskit.
"""
import collections
import inspect
import io
import itertools
import json
import math
import os
import pathlib
import pickle
import platform
import random
import re
import tempfile
import unittest
import uuid as _uuid
import warnings

import attr
import kastore
import msprime
import networkx as nx
import numpy as np
import pytest

import _tskit
import tests as tests
import tests.simplify as simplify
import tests.tsutil as tsutil
import tskit
import tskit.util as util
from tskit import UNKNOWN_TIME


def insert_uniform_mutations(tables, num_mutations, nodes):
    """
    Returns n evenly mutations over the specified list of nodes.
    """
    for j in range(num_mutations):
        tables.sites.add_row(
            position=j * (tables.sequence_length / num_mutations),
            ancestral_state="0",
            metadata=json.dumps({"index": j}).encode(),
        )
        tables.mutations.add_row(
            site=j,
            derived_state="1",
            node=nodes[j % len(nodes)],
            metadata=json.dumps({"index": j}).encode(),
        )


def get_table_collection_copy(tables, sequence_length):
    """
    Returns a copy of the specified table collection with the specified
    sequence length.
    """
    table_dict = tables.asdict()
    table_dict["sequence_length"] = sequence_length
    return tskit.TableCollection.fromdict(table_dict)


def insert_gap(ts, position, length):
    """
    Inserts a gap of the specified size into the specified tree sequence.
    This involves: (1) breaking all edges that intersect with this point;
    and (2) shifting all coordinates greater than this value up by the
    gap length.
    """
    new_edges = []
    for e in ts.edges():
        if e.left < position < e.right:
            new_edges.append([e.left, position, e.parent, e.child])
            new_edges.append([position, e.right, e.parent, e.child])
        else:
            new_edges.append([e.left, e.right, e.parent, e.child])

    # Now shift up all coordinates.
    for e in new_edges:
        # Left coordinates == position get shifted
        if e[0] >= position:
            e[0] += length
        # Right coordinates == position do not get shifted
        if e[1] > position:
            e[1] += length
    tables = ts.dump_tables()
    L = ts.sequence_length + length
    tables = get_table_collection_copy(tables, L)
    tables.edges.clear()
    tables.sites.clear()
    tables.mutations.clear()
    for left, right, parent, child in new_edges:
        tables.edges.add_row(left, right, parent, child)
    tables.sort()
    # Throw in a bunch of mutations over the whole sequence on the samples.
    insert_uniform_mutations(tables, 100, list(ts.samples()))
    return tables.tree_sequence()


def get_gap_examples():
    """
    Returns example tree sequences that contain gaps within the list of
    edges.
    """
    ts = msprime.simulate(20, random_seed=56, recombination_rate=1)

    assert ts.num_trees > 1

    gap = 0.0125
    for x in [0, 0.1, 0.5, 0.75]:
        ts = insert_gap(ts, x, gap)
        found = False
        for t in ts.trees():
            if t.interval[0] == x:
                assert t.interval[1] == x + gap
                assert len(t.parent_dict) == 0
                found = True
        assert found
        yield ts
    # Give an example with a gap at the end.
    ts = msprime.simulate(10, random_seed=5, recombination_rate=1)
    tables = get_table_collection_copy(ts.dump_tables(), 2)
    tables.sites.clear()
    tables.mutations.clear()
    insert_uniform_mutations(tables, 100, list(ts.samples()))
    yield tables.tree_sequence()


def get_internal_samples_examples():
    """
    Returns example tree sequences with internal samples.
    """
    n = 5
    ts = msprime.simulate(n, random_seed=10, mutation_rate=5)
    assert ts.num_mutations > 0
    tables = ts.dump_tables()
    nodes = tables.nodes
    flags = nodes.flags
    # Set all nodes to be samples.
    flags[:] = tskit.NODE_IS_SAMPLE
    nodes.flags = flags
    yield tables.tree_sequence()

    # Set just internal nodes to be samples.
    flags[:] = 0
    flags[n:] = tskit.NODE_IS_SAMPLE
    nodes.flags = flags
    yield tables.tree_sequence()

    # Set a mixture of internal and leaf samples.
    flags[:] = 0
    flags[n // 2 : n + n // 2] = tskit.NODE_IS_SAMPLE
    nodes.flags = flags
    yield tables.tree_sequence()


def get_decapitated_examples():
    """
    Returns example tree sequences in which the oldest edges have been removed.
    """
    ts = msprime.simulate(10, random_seed=1234)
    yield tsutil.decapitate(ts, ts.num_edges // 2)

    ts = msprime.simulate(20, recombination_rate=1, random_seed=1234)
    assert ts.num_trees > 2
    yield tsutil.decapitate(ts, ts.num_edges // 4)


def get_example_tree_sequences(back_mutations=True, gaps=True, internal_samples=True):
    if gaps:
        yield from get_decapitated_examples()
        yield from get_gap_examples()
    if internal_samples:
        yield from get_internal_samples_examples()
    seed = 1
    for n in [2, 3, 10, 100]:
        for m in [1, 2, 32]:
            for rho in [0, 0.1, 0.5]:
                recomb_map = msprime.RecombinationMap.uniform_map(m, rho, num_loci=m)
                ts = msprime.simulate(
                    recombination_map=recomb_map,
                    mutation_rate=0.1,
                    random_seed=seed,
                    population_configurations=[
                        msprime.PopulationConfiguration(n),
                        msprime.PopulationConfiguration(0),
                    ],
                    migration_matrix=[[0, 1], [1, 0]],
                )
                ts = tsutil.insert_random_ploidy_individuals(ts, 4, seed=seed)
                yield tsutil.add_random_metadata(ts, seed=seed)
                seed += 1
    for ts in get_bottleneck_examples():
        yield msprime.mutate(
            ts,
            rate=0.1,
            random_seed=seed,
            model=msprime.InfiniteSites(msprime.NUCLEOTIDES),
        )
    ts = msprime.simulate(15, length=4, recombination_rate=1)
    assert ts.num_trees > 1
    if back_mutations:
        yield tsutil.insert_branch_mutations(ts, mutations_per_branch=2)
    ts = tsutil.insert_multichar_mutations(ts)
    yield ts
    yield tsutil.add_random_metadata(ts)
    tables = ts.dump_tables()
    tables.edges.clear()
    yield tables.tree_sequence()  # empty tree sequence


def get_bottleneck_examples():
    """
    Returns an iterator of example tree sequences with nonbinary trees.
    """
    bottlenecks = [
        msprime.SimpleBottleneck(0.01, 0, proportion=0.05),
        msprime.SimpleBottleneck(0.02, 0, proportion=0.25),
        msprime.SimpleBottleneck(0.03, 0, proportion=1),
    ]
    for n in [3, 10, 100]:
        ts = msprime.simulate(
            n,
            length=100,
            recombination_rate=1,
            demographic_events=bottlenecks,
            random_seed=n,
        )
        yield ts


def get_back_mutation_examples():
    """
    Returns an iterator of example tree sequences with nonbinary trees.
    """
    ts = msprime.simulate(10, random_seed=1)
    for j in [1, 2, 3]:
        yield tsutil.insert_branch_mutations(ts, mutations_per_branch=j)
    for ts in get_bottleneck_examples():
        yield tsutil.insert_branch_mutations(ts)


def simple_get_pairwise_diversity(haplotypes):
    """
    Returns the value of pi for the specified haplotypes.
    """
    # Very simplistic algorithm...
    n = len(haplotypes)
    pi = 0
    for k in range(n):
        for j in range(k):
            for u, v in zip(haplotypes[j], haplotypes[k]):
                pi += u != v
    return 2 * pi / (n * (n - 1))


def simplify_tree_sequence(ts, samples, filter_sites=True):
    """
    Simple tree-by-tree algorithm to get a simplify of a tree sequence.
    """
    s = simplify.Simplifier(ts, samples, filter_sites=filter_sites)
    return s.simplify()


def oriented_forests(n):
    """
    Implementation of Algorithm O from TAOCP section 7.2.1.6.
    Generates all canonical n-node oriented forests.
    """
    p = [k - 1 for k in range(0, n + 1)]
    k = 1
    while k != 0:
        yield p
        if p[n] > 0:
            p[n] = p[p[n]]
            yield p
        k = n
        while k > 0 and p[k] == 0:
            k -= 1
        if k != 0:
            j = p[k]
            d = k - j
            not_done = True
            while not_done:
                if p[k - d] == p[j]:
                    p[k] = p[j]
                else:
                    p[k] = p[k - d] + d
                if k == n:
                    not_done = False
                else:
                    k += 1


def get_mrca(pi, x, y):
    """
    Returns the most recent common ancestor of nodes x and y in the
    oriented forest pi.
    """
    x_parents = [x]
    j = x
    while j != 0:
        j = pi[j]
        x_parents.append(j)
    y_parents = {y: None}
    j = y
    while j != 0:
        j = pi[j]
        y_parents[j] = None
    # We have the complete list of parents for x and y back to root.
    mrca = 0
    j = 0
    while x_parents[j] not in y_parents:
        j += 1
    mrca = x_parents[j]
    return mrca


class TestMRCACalculator:
    """
    Class to test the Schieber-Vishkin algorithm.

    These tests are included here as we use the MRCA calculator below in
    our tests.
    """

    def test_all_oriented_forests(self):
        # Runs through all possible oriented forests and checks all possible
        # node pairs using an inferior algorithm.
        for n in range(2, 9):
            for pi in oriented_forests(n):
                sv = tests.MRCACalculator(pi)
                for j in range(1, n + 1):
                    for k in range(1, j + 1):
                        mrca = get_mrca(pi, j, k)
                        assert mrca == sv.get_mrca(j, k)


class HighLevelTestCase:
    """
    Superclass of tests on the high level interface.
    """

    def verify_tree_mrcas(self, st):
        # Check the mrcas
        oriented_forest = [st.get_parent(j) for j in range(st.num_nodes)]
        mrca_calc = tests.MRCACalculator(oriented_forest)
        # We've done exhaustive tests elsewhere, no need to go
        # through the combinations.
        for j in range(st.num_nodes):
            mrca = st.get_mrca(0, j)
            assert mrca == mrca_calc.get_mrca(0, j)
            if mrca != tskit.NULL:
                assert st.get_time(mrca) == st.get_tmrca(0, j)

    def verify_tree_branch_lengths(self, tree):
        for u in tree.tree_sequence.samples():
            while tree.parent(u) != tskit.NULL:
                length = tree.time(tree.parent(u)) - tree.time(u)
                assert length > 0.0
                assert tree.branch_length(u) == length
                u = tree.parent(u)
            assert tree.parent(u) == tskit.NULL
            assert tree.branch_length(u) == 0

    def verify_tree_structure(self, st):
        roots = set()
        for u in st.samples():
            # verify the path to root
            assert st.is_sample(u)
            times = []
            while st.get_parent(u) != tskit.NULL:
                v = st.get_parent(u)
                times.append(st.get_time(v))
                assert st.get_time(v) >= 0.0
                assert u in st.get_children(v)
                u = v
            roots.add(u)
            assert times == sorted(times)
        assert sorted(list(roots)) == sorted(st.roots)
        assert len(st.roots) == st.num_roots
        u = st.left_root
        roots = []
        while u != tskit.NULL:
            roots.append(u)
            u = st.right_sib(u)
        assert roots == st.roots
        # To a top-down traversal, and make sure we meet all the samples.
        samples = []
        for root in st.roots:
            stack = [root]
            while len(stack) > 0:
                u = stack.pop()
                assert u != tskit.NULL
                if st.is_sample(u):
                    samples.append(u)
                if st.is_leaf(u):
                    assert len(st.get_children(u)) == 0
                else:
                    for c in reversed(st.get_children(u)):
                        stack.append(c)
                # Check that we get the correct number of samples at each
                # node.
                assert st.get_num_samples(u) == len(list(st.samples(u)))
                assert st.get_num_tracked_samples(u) == 0
        assert sorted(samples) == sorted(st.samples())
        # Check the parent dict
        pi = st.get_parent_dict()
        for root in st.roots:
            assert root not in pi
        for k, v in pi.items():
            assert st.get_parent(k) == v
        assert st.num_samples() == len(samples)
        assert sorted(st.samples()) == sorted(samples)

    def verify_tree_depths(self, st):
        for root in st.roots:
            stack = [(root, 0)]
            while len(stack) > 0:
                u, depth = stack.pop()
                assert st.depth(u) == depth
                for c in st.children(u):
                    stack.append((c, depth + 1))

    def verify_tree(self, st):
        self.verify_tree_mrcas(st)
        self.verify_tree_branch_lengths(st)
        self.verify_tree_structure(st)
        self.verify_tree_depths(st)

    def verify_trees(self, ts):
        pts = tests.PythonTreeSequence(ts)
        iter1 = ts.trees()
        iter2 = pts.trees()
        length = 0
        num_trees = 0
        breakpoints = [0]
        for st1, st2 in zip(iter1, iter2):
            assert st1.get_sample_size() == ts.get_sample_size()
            roots = set()
            for u in ts.samples():
                root = u
                while st1.get_parent(root) != tskit.NULL:
                    root = st1.get_parent(root)
                roots.add(root)
            assert st1.left_root == st2.left_root
            assert sorted(list(roots)) == sorted(st1.roots)
            assert st1.roots == st2.roots
            if len(roots) > 1:
                with pytest.raises(ValueError):
                    st1.root
            else:
                assert st1.root == list(roots)[0]
            assert st2 == st1
            assert not (st2 != st1)
            left, right = st1.get_interval()
            breakpoints.append(right)
            assert left == pytest.approx(length)
            assert left >= 0
            assert right > left
            assert right <= ts.get_sequence_length()
            length += right - left
            self.verify_tree(st1)
            num_trees += 1
        with pytest.raises(StopIteration):
            next(iter1)
        with pytest.raises(StopIteration):
            next(iter2)
        assert ts.get_num_trees() == num_trees
        assert breakpoints == list(ts.breakpoints())
        assert length == ts.get_sequence_length()


class TestNumpySamples:
    """
    Tests that we correctly handle samples as numpy arrays when passed to
    various methods.
    """

    def get_tree_sequence(self, num_demes=4):
        n = 40
        return msprime.simulate(
            samples=[
                msprime.Sample(time=0, population=j % num_demes) for j in range(n)
            ],
            population_configurations=[
                msprime.PopulationConfiguration() for _ in range(num_demes)
            ],
            migration_matrix=[
                [int(j != k) for j in range(num_demes)] for k in range(num_demes)
            ],
            random_seed=1,
            mutation_rate=10,
        )

    def test_samples(self):
        d = 4
        ts = self.get_tree_sequence(d)
        assert np.array_equal(ts.samples(), np.arange(ts.num_samples, dtype=np.int32))
        total = 0
        for pop in range(d):
            subsample = ts.samples(pop)
            total += subsample.shape[0]
            assert np.array_equal(subsample, ts.samples(population=pop))
            assert list(subsample) == [
                node.id
                for node in ts.nodes()
                if node.population == pop and node.is_sample()
            ]
        assert total == ts.num_samples

    def test_genotype_matrix_indexing(self):
        num_demes = 4
        ts = self.get_tree_sequence(num_demes)
        G = ts.genotype_matrix()
        for d in range(num_demes):
            samples = ts.samples(population=d)
            total = 0
            for tree in ts.trees(tracked_samples=samples):
                for mutation in tree.mutations():
                    total += tree.num_tracked_samples(mutation.node)
            assert total == np.sum(G[:, samples])

    def test_genotype_indexing(self):
        num_demes = 6
        ts = self.get_tree_sequence(num_demes)
        for d in range(num_demes):
            samples = ts.samples(population=d)
            total = 0
            for tree in ts.trees(tracked_samples=samples):
                for mutation in tree.mutations():
                    total += tree.num_tracked_samples(mutation.node)
            other_total = 0
            for variant in ts.variants():
                other_total += np.sum(variant.genotypes[samples])
            assert total == other_total

    def test_pairwise_diversity(self):
        num_demes = 6
        ts = self.get_tree_sequence(num_demes)
        pi1 = ts.pairwise_diversity(ts.samples())
        pi2 = ts.pairwise_diversity()
        assert pi1 == pi2
        for d in range(num_demes):
            samples = ts.samples(population=d)
            pi1 = ts.pairwise_diversity(samples)
            pi2 = ts.pairwise_diversity(list(samples))
            assert pi1 == pi2

    def test_simplify(self):
        num_demes = 3
        ts = self.get_tree_sequence(num_demes)
        sts = ts.simplify(samples=ts.samples())
        assert ts.num_samples == sts.num_samples
        for d in range(num_demes):
            samples = ts.samples(population=d)
            sts = ts.simplify(samples=samples)
            assert sts.num_samples == samples.shape[0]


class TestTreeSequence(HighLevelTestCase):
    """
    Tests for the tree sequence object.
    """

    def test_trees(self):
        for ts in get_example_tree_sequences():
            self.verify_trees(ts)

    def test_mutations(self):
        for ts in get_example_tree_sequences():
            self.verify_mutations(ts)

    def verify_pairwise_diversity(self, ts):
        haplotypes = ts.genotype_matrix(isolated_as_missing=False).T
        pi1 = ts.get_pairwise_diversity()
        pi2 = simple_get_pairwise_diversity(haplotypes)
        assert pi1 == pytest.approx(pi2)
        assert pi1 >= 0.0
        assert not math.isnan(pi1)
        # Check for a subsample.
        num_samples = ts.get_sample_size() // 2 + 1
        samples = list(ts.samples())[:num_samples]
        pi1 = ts.get_pairwise_diversity(samples)
        pi2 = simple_get_pairwise_diversity([haplotypes[j] for j in range(num_samples)])
        assert pi1 == pytest.approx(pi2)
        assert pi1 >= 0.0
        assert not math.isnan(pi1)

    @pytest.mark.slow
    def test_pairwise_diversity(self):
        for ts in get_example_tree_sequences():
            self.verify_pairwise_diversity(ts)

    def verify_edge_diffs(self, ts):
        pts = tests.PythonTreeSequence(ts)
        d1 = list(ts.edge_diffs())
        d2 = list(pts.edge_diffs())
        assert d1 == d2

        # check that we have the correct set of children at all nodes.
        children = collections.defaultdict(set)
        trees = iter(ts.trees())
        tree = next(trees)
        edge_ids = []
        last_right = 0
        for (left, right), edges_out, edges_in in ts.edge_diffs():
            assert left == last_right
            last_right = right
            for edge in edges_out:
                assert edge == ts.edge(edge.id)
                children[edge.parent].remove(edge.child)
            for edge in edges_in:
                edge_ids.append(edge.id)
                assert edge == ts.edge(edge.id)
                children[edge.parent].add(edge.child)
            while tree.interval[1] <= left:
                tree = next(trees)
            assert left >= tree.interval.left
            assert right <= tree.interval.right
            for u in tree.nodes():
                if tree.is_internal(u):
                    assert u in children
                    assert children[u] == set(tree.children(u))
        # check that we have seen all the edge ids
        assert np.array_equal(np.unique(edge_ids), np.arange(0, ts.num_edges))

    def test_edge_diffs(self):
        for ts in get_example_tree_sequences():
            self.verify_edge_diffs(ts)

    def test_edge_diffs_include_terminal(self):
        for ts in get_example_tree_sequences():
            edges = set()
            i = 0
            breakpoints = list(ts.breakpoints())
            for (left, right), e_out, e_in in ts.edge_diffs(include_terminal=True):
                assert left == breakpoints[i]
                if i == ts.num_trees:
                    # Last iteration, right==left==sequence_length
                    assert left == ts.sequence_length
                    assert right == ts.sequence_length
                else:
                    assert right == breakpoints[i + 1]
                for e in e_out:
                    edges.remove(e.id)
                for e in e_in:
                    edges.add(e.id)
                i += 1
            assert i == ts.num_trees + 1
            assert len(edges) == 0

    def verify_edgesets(self, ts):
        """
        Verifies that the edgesets we return are equivalent to the original edges.
        """
        new_edges = []
        for edgeset in ts.edgesets():
            assert edgeset.children == sorted(edgeset.children)
            assert len(edgeset.children) > 0
            for child in edgeset.children:
                new_edges.append(
                    tskit.Edge(edgeset.left, edgeset.right, edgeset.parent, child)
                )
        # squash the edges.
        t = ts.dump_tables().nodes.time
        new_edges.sort(key=lambda e: (t[e.parent], e.parent, e.child, e.left))

        squashed = []
        if len(new_edges) > 0:
            last_e = new_edges[0]
            for e in new_edges[1:]:
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
            # reset the IDs
            for i, e in enumerate(squashed):
                e.id = i
        edges = list(ts.edges())
        assert len(squashed) == len(edges)
        assert edges == squashed

    def test_edge_ids(self):
        for ts in get_example_tree_sequences():
            for index, edge in enumerate(ts.edges()):
                assert edge.id == index

    def test_edge_span_property(self):
        for ts in get_example_tree_sequences():
            for edge in ts.edges():
                assert edge.span == edge.right - edge.left

    def test_edgesets(self):
        for ts in get_example_tree_sequences():
            self.verify_edgesets(ts)

    def test_breakpoints(self):
        for ts in get_example_tree_sequences():
            breakpoints = ts.breakpoints(as_array=True)
            assert breakpoints.shape == (ts.num_trees + 1,)
            other = np.fromiter(iter([0] + [t.interval[1] for t in ts.trees()]), float)
            assert np.array_equal(other, breakpoints)
            # in case downstream code has
            for j, x in enumerate(ts.breakpoints()):
                assert breakpoints[j] == x
                assert isinstance(x, float)
            assert j == ts.num_trees

    def verify_coalescence_records(self, ts):
        """
        Checks that the coalescence records we output are correct.
        """
        edgesets = list(ts.edgesets())
        records = list(ts.records())
        assert len(edgesets) == len(records)
        for edgeset, record in zip(edgesets, records):
            assert edgeset.left == record.left
            assert edgeset.right == record.right
            assert edgeset.parent == record.node
            assert edgeset.children == record.children
            parent = ts.node(edgeset.parent)
            assert parent.time == record.time
            assert parent.population == record.population

    def test_coalescence_records(self):
        for ts in get_example_tree_sequences():
            self.verify_coalescence_records(ts)

    def test_compute_mutation_parent(self):
        for ts in get_example_tree_sequences():
            tables = ts.dump_tables()
            before = tables.mutations.parent[:]
            tables.compute_mutation_parents()
            parent = ts.tables.mutations.parent
            assert np.array_equal(parent, before)

    def test_compute_mutation_time(self):
        for ts in get_example_tree_sequences():
            tables = ts.dump_tables()
            python_time = tsutil.compute_mutation_times(ts)
            tables.compute_mutation_times()
            assert np.allclose(
                python_time, tables.mutations.time, rtol=1e-15, atol=1e-15
            )
            # Check we have valid times
            tables.tree_sequence()

    def verify_tracked_samples(self, ts):
        # Should be empty list by default.
        for tree in ts.trees():
            assert tree.get_num_tracked_samples() == 0
            for u in tree.nodes():
                assert tree.get_num_tracked_samples(u) == 0
        samples = list(ts.samples())
        tracked_samples = samples[:2]
        for tree in ts.trees(tracked_samples=tracked_samples):
            if len(tree.parent_dict) == 0:
                # This is a crude way of checking if we have multiple roots.
                # We'll need to fix this code up properly when we support multiple
                # roots and remove this check
                break
            nu = [0 for j in range(ts.get_num_nodes())]
            assert tree.get_num_tracked_samples() == len(tracked_samples)
            for j in tracked_samples:
                u = j
                while u != tskit.NULL:
                    nu[u] += 1
                    u = tree.get_parent(u)
            for u, count in enumerate(nu):
                assert tree.get_num_tracked_samples(u) == count

    def test_tracked_samples(self):
        for ts in get_example_tree_sequences():
            self.verify_tracked_samples(ts)

    def test_tracked_samples_is_first_arg(self):
        for ts in get_example_tree_sequences():
            samples = list(ts.samples())[:2]
            for a, b in zip(ts.trees(samples), ts.trees(tracked_samples=samples)):
                assert a.get_num_tracked_samples() == b.get_num_tracked_samples()

    def test_deprecated_sample_aliases(self):
        for ts in get_example_tree_sequences():
            # Ensure that we get the same results from the various combinations
            # of leaf_lists, sample_lists etc.
            samples = list(ts.samples())[:2]
            # tracked leaves/samples
            trees_new = ts.trees(tracked_samples=samples)
            trees_old = ts.trees(tracked_leaves=samples)
            for t_new, t_old in zip(trees_new, trees_old):
                for u in t_new.nodes():
                    assert t_new.num_tracked_samples(u) == t_old.get_num_tracked_leaves(
                        u
                    )
            trees_new = ts.trees()
            trees_old = ts.trees()
            for t_new, t_old in zip(trees_new, trees_old):
                for u in t_new.nodes():
                    assert t_new.num_samples(u) == t_old.get_num_leaves(u)
                    assert list(t_new.samples(u)) == list(t_old.get_leaves(u))
            for on in [True, False]:
                trees_new = ts.trees(sample_lists=on)
                trees_old = ts.trees(leaf_lists=on)
                for t_new, t_old in zip(trees_new, trees_old):
                    for u in t_new.nodes():
                        assert t_new.num_samples(u) == t_old.get_num_leaves(u)
                        assert list(t_new.samples(u)) == list(t_old.get_leaves(u))

    def verify_samples(self, ts):
        # We should get the same list of samples if we use the low-level
        # sample lists or a simple traversal.
        samples1 = []
        for t in ts.trees(sample_lists=False):
            samples1.append(list(t.samples()))
        samples2 = []
        for t in ts.trees(sample_lists=True):
            samples2.append(list(t.samples()))
        assert samples1 == samples2

    def test_samples(self):
        for ts in get_example_tree_sequences():
            self.verify_samples(ts)
            pops = {node.population for node in ts.nodes()}
            for pop in pops:
                subsample = ts.samples(pop)
                assert np.array_equal(subsample, ts.samples(population=pop))
                assert np.array_equal(subsample, ts.samples(population_id=pop))
                assert list(subsample) == [
                    node.id
                    for node in ts.nodes()
                    if node.population == pop and node.is_sample()
                ]
            with pytest.raises(ValueError):
                ts.samples(population=0, population_id=0)

    def test_first_last(self):
        for ts in get_example_tree_sequences():
            for kwargs in [{}, {"tracked_samples": ts.samples()}]:
                t1 = ts.first(**kwargs)
                t2 = next(ts.trees())
                assert not (t1 is t2)
                assert t1.parent_dict == t2.parent_dict
                assert t1.index == 0
                if "tracked_samples" in kwargs:
                    assert t1.num_tracked_samples() != 0
                else:
                    assert t1.num_tracked_samples() == 0

                t1 = ts.last(**kwargs)
                t2 = next(reversed(ts.trees()))
                assert not (t1 is t2)
                assert t1.parent_dict == t2.parent_dict
                assert t1.index == ts.num_trees - 1
                if "tracked_samples" in kwargs:
                    assert t1.num_tracked_samples() != 0
                else:
                    assert t1.num_tracked_samples() == 0

    def test_trees_interface(self):
        ts = list(get_example_tree_sequences())[0]
        for t in ts.trees():
            assert t.get_num_samples(0) == 1
            assert t.get_num_tracked_samples(0) == 0
            assert list(t.samples(0)) == [0]
            assert t.tree_sequence is ts

        for t in ts.trees(tracked_samples=[0]):
            assert t.get_num_samples(0) == 1
            assert t.get_num_tracked_samples(0) == 1
            assert list(t.samples(0)) == [0]

        for t in ts.trees(sample_lists=True):
            assert t.get_num_samples(0) == 1
            assert t.get_num_tracked_samples(0) == 0
            assert list(t.samples(0)) == [0]

    def test_get_pairwise_diversity(self):
        for ts in get_example_tree_sequences():
            with pytest.raises(ValueError):
                ts.get_pairwise_diversity([])
            samples = list(ts.samples())
            assert ts.get_pairwise_diversity() == ts.get_pairwise_diversity(samples)
            assert ts.get_pairwise_diversity(samples[:2]) == ts.get_pairwise_diversity(
                list(reversed(samples[:2]))
            )

    def test_populations(self):
        more_than_zero = False
        for ts in get_example_tree_sequences():
            N = ts.num_populations
            if N > 0:
                more_than_zero = True
            pops = list(ts.populations())
            assert len(pops) == N
            for j in range(N):
                assert pops[j] == ts.population(j)
                assert pops[j].id == j
                assert isinstance(pops[j].metadata, bytes)
        assert more_than_zero

    def test_individuals(self):
        more_than_zero = False
        mapped_to_nodes = False
        for ts in get_example_tree_sequences():
            ind_node_map = collections.defaultdict(list)
            for node in ts.nodes():
                if node.individual != tskit.NULL:
                    ind_node_map[node.individual].append(node.id)
            if len(ind_node_map) > 0:
                mapped_to_nodes = True
            N = ts.num_individuals
            if N > 0:
                more_than_zero = True
            inds = list(ts.individuals())
            assert len(inds) == N
            for j in range(N):
                assert inds[j] == ts.individual(j)
                assert inds[j].id == j
                assert isinstance(inds[j].metadata, bytes)
                assert isinstance(inds[j].location, np.ndarray)
                assert isinstance(inds[j].nodes, np.ndarray)
                assert ind_node_map[j] == list(inds[j].nodes)

        assert more_than_zero
        assert mapped_to_nodes

    def test_get_population(self):
        # Deprecated interface for ts.node(id).population
        for ts in get_example_tree_sequences():
            N = ts.get_num_nodes()
            with pytest.raises(ValueError):
                ts.get_population(-1)
            with pytest.raises(ValueError):
                ts.get_population(N)
            with pytest.raises(ValueError):
                ts.get_population(N + 1)
            for node in [0, N - 1]:
                assert ts.get_population(node) == ts.node(node).population

    def test_get_time(self):
        # Deprecated interface for ts.node(id).time
        for ts in get_example_tree_sequences():
            N = ts.get_num_nodes()
            with pytest.raises(ValueError):
                ts.get_time(-1)
            with pytest.raises(ValueError):
                ts.get_time(N)
            with pytest.raises(ValueError):
                ts.get_time(N + 1)
            for u in range(N):
                assert ts.get_time(u) == ts.node(u).time

    def test_max_root_time(self):
        for ts in get_example_tree_sequences():
            oldest = max(
                max(tree.time(root) for root in tree.roots) for tree in ts.trees()
            )
            assert oldest == ts.max_root_time

    def test_max_root_time_corner_cases(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=2)
        tables.nodes.add_row(flags=0, time=3)
        assert tables.tree_sequence().max_root_time == 2
        tables.edges.add_row(0, 1, 1, 0)
        assert tables.tree_sequence().max_root_time == 2
        tables.edges.add_row(0, 1, 3, 1)
        assert tables.tree_sequence().max_root_time == 3

    def verify_simplify_provenance(self, ts):
        new_ts = ts.simplify()
        assert new_ts.num_provenances == ts.num_provenances + 1
        old = list(ts.provenances())
        new = list(new_ts.provenances())
        assert old == new[:-1]
        # TODO call verify_provenance on this.
        assert len(new[-1].timestamp) > 0
        assert len(new[-1].record) > 0

        new_ts = ts.simplify(record_provenance=False)
        assert new_ts.tables.provenances == ts.tables.provenances

    def verify_simplify_topology(self, ts, sample):
        new_ts, node_map = ts.simplify(sample, map_nodes=True)
        if len(sample) == 0:
            assert new_ts.num_nodes == 0
            assert new_ts.num_edges == 0
            assert new_ts.num_sites == 0
            assert new_ts.num_mutations == 0
        elif len(sample) == 1:
            assert new_ts.num_nodes == 1
            assert new_ts.num_edges == 0
        # The output samples should be 0...n
        assert new_ts.num_samples == len(sample)
        assert list(range(len(sample))) == list(new_ts.samples())
        for j in range(new_ts.num_samples):
            assert node_map[sample[j]] == j
        for u in range(ts.num_nodes):
            old_node = ts.node(u)
            if node_map[u] != tskit.NULL:
                new_node = new_ts.node(node_map[u])
                assert old_node.time == new_node.time
                assert old_node.population == new_node.population
                assert old_node.metadata == new_node.metadata
        for u in sample:
            old_node = ts.node(u)
            new_node = new_ts.node(node_map[u])
            assert old_node.flags == new_node.flags
            assert old_node.time == new_node.time
            assert old_node.population == new_node.population
            assert old_node.metadata == new_node.metadata
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
            # If the MRCA of all pairs of samples is the same, then we have the
            # same information. We limit this to at most 500 pairs
            pairs = itertools.islice(itertools.combinations(sample, 2), 500)
            for pair in pairs:
                mapped_pair = [node_map[u] for u in pair]
                mrca1 = old_tree.get_mrca(*pair)
                mrca2 = new_tree.get_mrca(*mapped_pair)
                if mrca1 == tskit.NULL:
                    assert mrca2 == mrca1
                else:
                    assert mrca2 == node_map[mrca1]
                    assert old_tree.get_time(mrca1) == new_tree.get_time(mrca2)
                    assert old_tree.get_population(mrca1) == new_tree.get_population(
                        mrca2
                    )

    def verify_simplify_equality(self, ts, sample):
        for filter_sites in [False, True]:
            s1, node_map1 = ts.simplify(
                sample, map_nodes=True, filter_sites=filter_sites
            )
            t1 = s1.dump_tables()
            s2, node_map2 = simplify_tree_sequence(
                ts, sample, filter_sites=filter_sites
            )
            t2 = s2.dump_tables()
            assert s1.num_samples == len(sample)
            assert s2.num_samples == len(sample)
            assert all(node_map1 == node_map2)
            assert t1.individuals == t2.individuals
            assert t1.nodes == t2.nodes
            assert t1.edges == t2.edges
            assert t1.migrations == t2.migrations
            assert t1.sites == t2.sites
            assert t1.mutations == t2.mutations
            assert t1.populations == t2.populations

    def verify_simplify_variants(self, ts, sample):
        subset = ts.simplify(sample)
        sample_map = {u: j for j, u in enumerate(ts.samples())}
        # Need to map IDs back to their sample indexes
        s = np.array([sample_map[u] for u in sample])
        # Build a map of genotypes by position
        full_genotypes = {}
        for variant in ts.variants(isolated_as_missing=False):
            alleles = [variant.alleles[g] for g in variant.genotypes]
            full_genotypes[variant.position] = alleles
        for variant in subset.variants(isolated_as_missing=False):
            if variant.position in full_genotypes:
                a1 = [full_genotypes[variant.position][u] for u in s]
                a2 = [variant.alleles[g] for g in variant.genotypes]
                assert a1 == a2

    @pytest.mark.slow
    def test_simplify(self):
        num_mutations = 0
        for ts in get_example_tree_sequences():
            self.verify_simplify_provenance(ts)
            n = ts.get_sample_size()
            num_mutations += ts.get_num_mutations()
            sample_sizes = {0, 1}
            if n > 2:
                sample_sizes |= {2, max(2, n // 2), n - 1}
            for k in sample_sizes:
                subset = random.sample(list(ts.samples()), k)
                self.verify_simplify_topology(ts, subset)
                self.verify_simplify_equality(ts, subset)
                self.verify_simplify_variants(ts, subset)
        assert num_mutations > 0

    def test_simplify_bugs(self):
        prefix = os.path.join(os.path.dirname(__file__), "data", "simplify-bugs")
        j = 1
        while True:
            nodes_file = os.path.join(prefix, f"{j:02d}-nodes.txt")
            if not os.path.exists(nodes_file):
                break
            edges_file = os.path.join(prefix, f"{j:02d}-edges.txt")
            sites_file = os.path.join(prefix, f"{j:02d}-sites.txt")
            mutations_file = os.path.join(prefix, f"{j:02d}-mutations.txt")
            with open(nodes_file) as nodes, open(edges_file) as edges, open(
                sites_file
            ) as sites, open(mutations_file) as mutations:
                ts = tskit.load_text(
                    nodes=nodes,
                    edges=edges,
                    sites=sites,
                    mutations=mutations,
                    strict=False,
                )
            samples = list(ts.samples())
            self.verify_simplify_equality(ts, samples)
            j += 1
        assert j > 1

    def test_simplify_migrations_fails(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(10),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            random_seed=2,
            record_migrations=True,
        )
        assert ts.num_migrations > 0
        # We don't support simplify with migrations, so should fail.
        with pytest.raises(_tskit.LibraryError):
            ts.simplify()

    def test_deprecated_apis(self):
        ts = msprime.simulate(10, random_seed=1)
        assert ts.get_ll_tree_sequence() == ts.ll_tree_sequence
        assert ts.get_sample_size() == ts.sample_size
        assert ts.get_sample_size() == ts.num_samples
        assert ts.get_sequence_length() == ts.sequence_length
        assert ts.get_num_trees() == ts.num_trees
        assert ts.get_num_mutations() == ts.num_mutations
        assert ts.get_num_nodes() == ts.num_nodes
        assert ts.get_pairwise_diversity() == ts.pairwise_diversity()
        samples = ts.samples()
        assert ts.get_pairwise_diversity(samples) == ts.pairwise_diversity(samples)
        assert np.array_equal(ts.get_samples(), ts.samples())

    def test_sites(self):
        some_sites = False
        for ts in get_example_tree_sequences():
            tables = ts.dump_tables()
            sites = tables.sites
            mutations = tables.mutations
            assert ts.num_sites == len(sites)
            assert ts.num_mutations == len(mutations)
            previous_pos = -1
            mutation_index = 0
            ancestral_state = tskit.unpack_strings(
                sites.ancestral_state, sites.ancestral_state_offset
            )
            derived_state = tskit.unpack_strings(
                mutations.derived_state, mutations.derived_state_offset
            )

            for index, site in enumerate(ts.sites()):
                s2 = ts.site(site.id)
                assert s2 == site
                assert site.position == sites.position[index]
                assert site.position > previous_pos
                previous_pos = site.position
                assert ancestral_state[index] == site.ancestral_state
                assert site.id == index
                for mutation in site.mutations:
                    m2 = ts.mutation(mutation.id)
                    assert m2 == mutation
                    assert mutation.site == site.id
                    assert mutation.site == mutations.site[mutation_index]
                    assert mutation.node == mutations.node[mutation_index]
                    assert mutation.parent == mutations.parent[mutation_index]
                    assert mutation.id == mutation_index
                    assert derived_state[mutation_index] == mutation.derived_state
                    mutation_index += 1
                some_sites = True
            total_sites = 0
            for tree in ts.trees():
                assert len(list(tree.sites())) == tree.num_sites
                total_sites += tree.num_sites
            assert ts.num_sites == total_sites
            assert mutation_index == len(mutations)
        assert some_sites

    def verify_mutations(self, ts):
        other_mutations = []
        for site in ts.sites():
            for mutation in site.mutations:
                other_mutations.append(mutation)
        mutations = list(ts.mutations())
        assert ts.num_mutations == len(other_mutations)
        assert ts.num_mutations == len(mutations)
        for mut, other_mut in zip(mutations, other_mutations):
            # We cannot compare these directly as the mutations obtained
            # from the mutations iterator will have extra deprecated
            # attributes.
            assert mut.id == other_mut.id
            assert mut.site == other_mut.site
            assert mut.parent == other_mut.parent
            assert mut.node == other_mut.node
            assert mut.metadata == other_mut.metadata
            # Check the deprecated attrs.
            assert mut.position == ts.site(mut.site).position
            assert mut.index == mut.site

    def test_sites_mutations(self):
        # Check that the mutations iterator returns the correct values.
        for ts in get_example_tree_sequences():
            self.verify_mutations(ts)

    def test_removed_methods(self):
        ts = next(get_example_tree_sequences())
        with pytest.raises(NotImplementedError):
            ts.get_num_records()
        with pytest.raises(NotImplementedError):
            ts.diffs()
        with pytest.raises(NotImplementedError):
            ts.newick_trees()

    def test_dump_pathlib(self, ts_fixture, tmp_path):
        path = tmp_path / "tmp.trees"
        assert path.exists
        assert path.is_file
        ts_fixture.dump(path)
        other_ts = tskit.load(path)
        assert ts_fixture.tables == other_ts.tables

    @pytest.mark.skipif(platform.system() == "Windows", reason="Windows doesn't raise")
    def test_dump_load_errors(self):
        ts = msprime.simulate(5, random_seed=1)
        # Try to dump/load files we don't have access to or don't exist.
        for func in [ts.dump, tskit.load]:
            for f in ["/", "/test.trees", "/dir_does_not_exist/x.trees"]:
                with pytest.raises(OSError):
                    func(f)
                try:
                    func(f)
                except OSError as e:
                    message = str(e)
                    assert len(message) > 0
            f = "/" + 4000 * "x"
            with pytest.raises(OSError):
                func(f)
            try:
                func(f)
            except OSError as e:
                message = str(e)
            assert "File name too long" in message
            for bad_filename in [[], None, {}]:
                with pytest.raises(TypeError):
                    func(bad_filename)

    def test_tables_sequence_length_round_trip(self):
        for sequence_length in [0.1, 1, 10, 100]:
            ts = msprime.simulate(5, length=sequence_length, random_seed=1)
            assert ts.sequence_length == sequence_length
            tables = ts.tables
            assert tables.sequence_length == sequence_length
            new_ts = tables.tree_sequence()
            assert new_ts.sequence_length == sequence_length

    def test_migrations(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(10),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            random_seed=2,
            record_migrations=True,
        )
        assert ts.num_migrations > 0
        migrations = list(ts.migrations())
        assert len(migrations) == ts.num_migrations
        for migration in migrations:
            assert migration.source in [0, 1]
            assert migration.dest in [0, 1]
            assert migration.time > 0
            assert migration.left == 0
            assert migration.right == 1
            assert 0 <= migration.node < ts.num_nodes

    def test_len_trees(self):
        for ts in get_example_tree_sequences():
            tree_iter = ts.trees()
            assert len(tree_iter) == ts.num_trees

    def test_list(self):
        for ts in get_example_tree_sequences():
            for kwargs in [{}, {"tracked_samples": ts.samples()}]:
                tree_list = ts.aslist(**kwargs)
                assert len(tree_list) == ts.num_trees
                assert len(set(map(id, tree_list))) == ts.num_trees
                for index, tree in enumerate(tree_list):
                    assert index == tree.index
                for t1, t2 in zip(tree_list, ts.trees(**kwargs)):
                    assert t1 == t2
                    assert t1.parent_dict == t2.parent_dict
                    if "tracked_samples" in kwargs:
                        assert t1.num_tracked_samples() != 0
                        assert t2.num_tracked_samples() != 0
                    else:
                        assert t1.num_tracked_samples() == 0
                        assert t2.num_tracked_samples() == 0

    def test_reversed_trees(self):
        for ts in get_example_tree_sequences():
            index = ts.num_trees - 1
            tree_list = ts.aslist()
            for tree in reversed(ts.trees()):
                assert tree.index == index
                t2 = tree_list[index]
                assert tree.interval == t2.interval
                assert tree.parent_dict == t2.parent_dict
                index -= 1

    def test_at_index(self):
        for ts in get_example_tree_sequences():
            for kwargs in [{}, {"tracked_samples": ts.samples()}]:
                tree_list = ts.aslist(**kwargs)
                for index in list(range(ts.num_trees)) + [-1]:
                    t1 = tree_list[index]
                    t2 = ts.at_index(index, **kwargs)
                    assert t1 == t2
                    assert t1.interval == t2.interval
                    assert t1.parent_dict == t2.parent_dict
                    if "tracked_samples" in kwargs:
                        assert t2.num_tracked_samples() != 0
                    else:
                        assert t2.num_tracked_samples() == 0

    def test_at(self):
        for ts in get_example_tree_sequences():
            for kwargs in [{}, {"tracked_samples": ts.samples()}]:
                tree_list = ts.aslist(**kwargs)
                for t1 in tree_list:
                    left, right = t1.interval
                    mid = left + (right - left) / 2
                    for pos in [left, left + 1e-9, mid, right - 1e-9]:
                        t2 = ts.at(pos, **kwargs)
                        assert t1 == t2
                        assert t1.interval == t2.interval
                        assert t1.parent_dict == t2.parent_dict
                    if right < ts.sequence_length:
                        t2 = ts.at(right, **kwargs)
                        t3 = tree_list[t1.index + 1]
                        assert t3 == t2
                        assert t3.interval == t2.interval
                        assert t3.parent_dict == t2.parent_dict
                    if "tracked_samples" in kwargs:
                        assert t2.num_tracked_samples() != 0
                    else:
                        assert t2.num_tracked_samples() == 0

    def test_sequence_iteration(self):
        for ts in get_example_tree_sequences():
            for table_name in ts.tables_dict.keys():
                sequence = getattr(ts, table_name)()
                length = getattr(ts, "num_" + table_name)
                # Test __iter__
                for i, n in enumerate(sequence):
                    assert i == n.id
                assert n.id == (length - 1 if length else 0)
                if table_name == "mutations":
                    # Mutations are not currently sequences, so have no len or idx access
                    with pytest.raises(TypeError):
                        len(sequence)
                    if length != 0:
                        with pytest.raises(TypeError):
                            sequence[0]
                else:
                    # Test __len__
                    assert len(sequence) == length
                    # Test __getitem__ on the last item in the sequence
                    if length != 0:
                        assert sequence[length - 1] == n  # +ive indexing
                        assert sequence[-1] == n  # -ive indexing
                    with pytest.raises(IndexError):
                        sequence[length]
                    # Test reverse
                    for i, n in enumerate(reversed(sequence)):
                        assert i == length - 1 - n.id
                    assert n.id == 0

    def test_load_tables(self):
        for ts in get_example_tree_sequences():
            tables = ts.dump_tables()
            tables.drop_index()

            # Tables not in tc not rebuilt as per default, so error
            with pytest.raises(
                _tskit.LibraryError, match="^Table collection must be indexed$"
            ):
                assert tskit.TreeSequence.load_tables(tables).dump_tables().has_index()

            # Tables not in tc, but rebuilt
            assert (
                tskit.TreeSequence.load_tables(tables, build_indexes=True)
                .dump_tables()
                .has_index()
            )

            tables.build_index()
            # Tables in tc, not rebuilt
            assert (
                tskit.TreeSequence.load_tables(tables, build_indexes=False)
                .dump_tables()
                .has_index()
            )
            # Tables in tc, and rebuilt
            assert tskit.TreeSequence.load_tables(tables).dump_tables().has_index()

    def test_html_repr(self):
        for ts in get_example_tree_sequences():
            html = ts._repr_html_()
            assert len(html) > 4300
            assert f"<tr><td>Trees</td><td>{ts.num_trees}</td></tr>" in html
            for table in ts.tables.name_map:
                assert f"<td>{table.capitalize()}</td>" in html

    def test_repr(self):
        for ts in get_example_tree_sequences():
            s = repr(ts)
            assert len(s) > 999
            assert re.search(rf"║Trees *│ *{ts.num_trees}║", s)
            for table in ts.tables.name_map:
                assert re.search(rf"║{table.capitalize()} *│", s)

    def test_nbytes(self, tmp_path, ts_fixture):
        ts_fixture.dump(tmp_path / "tables")
        store = kastore.load(tmp_path / "tables")
        for v in store.values():
            # Check we really have data in every field
            assert v.nbytes > 0
        nbytes = sum(
            array.nbytes
            for name, array in store.items()
            # nbytes is the size of asdict, so exclude file format items
            if name not in ["format/version", "format/name", "uuid"]
        )
        assert nbytes == ts_fixture.nbytes

    def test_equals(self):
        # Here we don't use the fixture as we'd like to run the same sim twice
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        t1 = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        )
        t2 = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        )

        assert t1 == t1
        assert t1 == t1.dump_tables().tree_sequence()
        assert t1.dump_tables().tree_sequence() == t1

        # The provenances may or may not be equal depending on the clock
        # precision for record. So clear them first.
        tb1 = t1.dump_tables()
        tb2 = t2.dump_tables()
        tb1.provenances.clear()
        tb2.provenances.clear()
        t1 = tb1.tree_sequence()
        t2 = tb2.tree_sequence()

        assert t1 == t2
        assert t1 == t2
        assert not (t1 != t2)
        # We don't do more as this is the same code path as TableCollection.__eq__

    def test_equals_options(self, ts_fixture):
        t1 = ts_fixture
        # Take a copy
        t2 = ts_fixture.dump_tables().tree_sequence()

        def modify(ts, func):
            tc = ts.dump_tables()
            func(tc)
            return tc.tree_sequence()

        t1 = modify(t1, lambda tc: tc.provenances.add_row("random stuff"))
        assert not (t1 == t2)
        assert t1.equals(t2, ignore_provenance=True)
        assert t2.equals(t1, ignore_provenance=True)
        assert not (t1.equals(t2))
        assert not (t2.equals(t1))
        t1 = modify(t1, lambda tc: tc.provenances.clear())
        t2 = modify(t2, lambda tc: tc.provenances.clear())
        assert t1.equals(t2)
        assert t2.equals(t1)

        tc = t1.dump_tables()
        tc.metadata_schema = tskit.MetadataSchema({"codec": "json", "type": "object"})
        t1 = tc.tree_sequence()
        tc = t1.dump_tables()
        tc.metadata = {"hello": "world"}
        t1 = tc.tree_sequence()

        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_ts_metadata=True)
        assert not t2.equals(t1)
        assert t2.equals(t1, ignore_ts_metadata=True)
        tc = t2.dump_tables()
        tc.metadata_schema = t1.metadata_schema
        t2 = tc.tree_sequence()
        assert not t1.equals(t2)
        assert t1.equals(t2, ignore_ts_metadata=True)
        assert not t2.equals(t1)
        assert t2.equals(t1, ignore_ts_metadata=True)

        t1 = modify(t1, lambda tc: tc.provenances.add_row("random stuff"))
        assert not t1.equals(t2)
        assert not t1.equals(t2, ignore_ts_metadata=True)
        assert not t1.equals(t2, ignore_provenance=True)
        assert t1.equals(t2, ignore_ts_metadata=True, ignore_provenance=True)

        t1 = modify(t1, lambda tc: tc.provenances.clear())
        t2 = modify(t2, lambda tc: setattr(tc, "metadata", t1.metadata))  # noqa: B010
        assert t1.equals(t2)
        assert t2.equals(t1)


class TestTreeSequenceMethodSignatures:
    ts = msprime.simulate(10, random_seed=1234)

    def test_kwargs_only(self):
        with pytest.raises(TypeError, match="argument"):
            tskit.Tree(self.ts, [], True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.trees([], True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.haplotypes(True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.variants(True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.genotype_matrix(True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.simplify([], True)
        with pytest.raises(TypeError, match="argument"):
            self.ts.draw_svg("filename", True)
        with pytest.raises(TypeError, match="argument"):
            tskit.TreeSequence.load_tables(tskit.TableCollection(1), True)

    def test_trees_params(self):
        """
        The initial .trees() iterator parameters should match those in Tree.__init__()
        """
        tree_class_params = list(inspect.signature(tskit.Tree).parameters.items())
        trees_iter_params = list(
            inspect.signature(tskit.TreeSequence.trees).parameters.items()
        )
        # Skip the first param, which is `tree_sequence` and `self` respectively
        tree_class_params = tree_class_params[1:]
        # The trees iterator has some extra (deprecated) aliases
        trees_iter_params = trees_iter_params[1:-3]
        assert trees_iter_params == tree_class_params


class TestTreeSequenceMetadata:
    metadata_tables = [
        "node",
        "edge",
        "site",
        "mutation",
        "migration",
        "individual",
        "population",
    ]
    metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "title": "Example Metadata",
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "string_prop": {"type": "string"},
                "num_prop": {"type": "number"},
            },
            "required": ["table", "string_prop", "num_prop"],
            "additionalProperties": False,
        },
    )

    def test_tree_sequence_metadata_schema(self):
        tc = tskit.TableCollection(1)
        ts = tc.tree_sequence()
        assert str(ts.metadata_schema) == str(tskit.MetadataSchema(None))
        tc.metadata_schema = self.metadata_schema
        ts = tc.tree_sequence()
        assert str(ts.metadata_schema) == str(self.metadata_schema)
        with pytest.raises(AttributeError):
            del ts.metadata_schema
        with pytest.raises(AttributeError):
            ts.metadata_schema = tskit.MetadataSchema(None)

    def test_tree_sequence_metadata(self):
        tc = tskit.TableCollection(1)
        ts = tc.tree_sequence()
        assert ts.metadata == b""
        tc.metadata_schema = self.metadata_schema
        data = {
            "table": "tree-sequence",
            "string_prop": "stringy",
            "num_prop": 42,
        }
        tc.metadata = data
        ts = tc.tree_sequence()
        assert ts.metadata == data
        with pytest.raises(AttributeError):
            ts.metadata = {"should": "fail"}
        with pytest.raises(AttributeError):
            del ts.metadata

    def test_table_metadata_schemas(self):
        ts = msprime.simulate(5)
        for table in self.metadata_tables:
            tables = ts.dump_tables()
            # Set and read back a unique schema for each table
            schema = tskit.MetadataSchema({"codec": "json", "TEST": f"{table}-SCHEMA"})
            # Check via table API
            getattr(tables, f"{table}s").metadata_schema = schema
            assert str(getattr(tables, f"{table}s").metadata_schema) == str(schema)
            for other_table in self.metadata_tables:
                if other_table != table:
                    assert str(getattr(tables, f"{other_table}s").metadata_schema) == ""
            # Check via tree-sequence API
            new_ts = tskit.TreeSequence.load_tables(tables)
            assert str(getattr(new_ts.table_metadata_schemas, table)) == str(schema)
            for other_table in self.metadata_tables:
                if other_table != table:
                    assert (
                        str(getattr(new_ts.table_metadata_schemas, other_table)) == ""
                    )
            # Can't set schema via this API
            with pytest.raises(AttributeError):
                new_ts.table_metadata_schemas = {}
                # or modify the schema tuple return object
                with pytest.raises(attr.exceptions.FrozenInstanceError):
                    setattr(
                        new_ts.table_metadata_schemas,
                        table,
                        tskit.MetadataSchema({"codec": "json"}),
                    )

    def test_table_metadata_round_trip_via_row_getters(self):
        # A tree sequence with all entities
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        )
        tables = ts.dump_tables()
        tables.individuals.add_row(location=[1, 2, 3])
        tables.individuals.add_row(location=[4, 5, 6])
        ts = tables.tree_sequence()

        for table in self.metadata_tables:
            new_tables = ts.dump_tables()
            tables_copy = ts.dump_tables()
            table_obj = getattr(new_tables, f"{table}s")
            table_obj.metadata_schema = self.metadata_schema
            table_obj.clear()
            # Write back the rows, but adding unique metadata
            for j, row in enumerate(getattr(tables_copy, f"{table}s")):
                row_data = attr.asdict(row)
                row_data["metadata"] = {
                    "table": table,
                    "string_prop": f"Row number{j}",
                    "num_prop": j,
                }
                table_obj.add_row(**row_data)
            new_ts = new_tables.tree_sequence()
            # Check that all tables have data otherwise we'll silently not check one
            assert getattr(new_ts, f"num_{table}s") > 0
            assert getattr(new_ts, f"num_{table}s") == getattr(ts, f"num_{table}s")
            for j, row in enumerate(getattr(new_ts, f"{table}s")()):
                assert row.metadata == {
                    "table": table,
                    "string_prop": f"Row number{row.id}",
                    "num_prop": row.id,
                }
                assert getattr(new_ts, f"{table}")(j).metadata == {
                    "table": table,
                    "string_prop": f"Row number{row.id}",
                    "num_prop": row.id,
                }


def test_pickle_round_trip(ts_fixture):
    assert ts_fixture.tables == pickle.loads(pickle.dumps(ts_fixture)).tables


class TestFileUuid(HighLevelTestCase):
    """
    Tests that the file UUID attribute is handled correctly.
    """

    def validate(self, ts):
        with tempfile.TemporaryDirectory() as tempdir:
            temp_file = pathlib.Path(tempdir) / "tmp.trees"
            assert ts.file_uuid is None
            ts.dump(temp_file)
            other_ts = tskit.load(temp_file)
            assert other_ts.file_uuid is not None
            assert len(other_ts.file_uuid), 36
            uuid = other_ts.file_uuid
            other_ts = tskit.load(temp_file)
            assert other_ts.file_uuid == uuid
            assert ts.tables == other_ts.tables

            # Check that the UUID is well-formed.
            parsed = _uuid.UUID("{" + uuid + "}")
            assert str(parsed) == uuid

            # Save the same tree sequence to the file. We should get a different UUID.
            ts.dump(temp_file)
            other_ts = tskit.load(temp_file)
            assert other_ts.file_uuid is not None
            assert other_ts.file_uuid != uuid

            # Even saving a ts that has a UUID to another file changes the UUID
            old_uuid = other_ts.file_uuid
            other_ts.dump(temp_file)
            assert other_ts.file_uuid == old_uuid
            other_ts = tskit.load(temp_file)
            assert other_ts.file_uuid is not None
            assert other_ts.file_uuid != old_uuid

            # Tables dumped from this ts are a deep copy, so they don't have
            # the file_uuid.
            tables = other_ts.dump_tables()
            assert tables.file_uuid is None

            # For now, ts.tables also returns a deep copy. This will hopefully
            # change in the future though.
            assert ts.tables.file_uuid is None

    def test_simple_simulation(self):
        ts = msprime.simulate(2, random_seed=1)
        self.validate(ts)

    def test_empty_tables(self):
        tables = tskit.TableCollection(1)
        self.validate(tables.tree_sequence())


class TestTreeSequenceTextIO(HighLevelTestCase):
    """
    Tests for the tree sequence text IO.
    """

    def verify_nodes_format(self, ts, nodes_file, precision):
        """
        Verifies that the nodes we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_nodes = nodes_file.read().splitlines()
        assert len(output_nodes) - 1 == ts.num_nodes
        assert list(output_nodes[0].split()) == [
            "id",
            "is_sample",
            "time",
            "population",
            "individual",
            "metadata",
        ]
        for node, line in zip(ts.nodes(), output_nodes[1:]):
            splits = line.split("\t")
            assert str(node.id) == splits[0]
            assert str(node.is_sample()) == splits[1]
            assert convert(node.time) == splits[2]
            assert str(node.population) == splits[3]
            assert str(node.individual) == splits[4]
            assert tests.base64_encode(node.metadata) == splits[5]

    def verify_edges_format(self, ts, edges_file, precision):
        """
        Verifies that the edges we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_edges = edges_file.read().splitlines()
        assert len(output_edges) - 1 == ts.num_edges
        assert list(output_edges[0].split()) == ["left", "right", "parent", "child"]
        for edge, line in zip(ts.edges(), output_edges[1:]):
            splits = line.split("\t")
            assert convert(edge.left) == splits[0]
            assert convert(edge.right) == splits[1]
            assert str(edge.parent) == splits[2]
            assert str(edge.child) == splits[3]

    def verify_sites_format(self, ts, sites_file, precision):
        """
        Verifies that the sites we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_sites = sites_file.read().splitlines()
        assert len(output_sites) - 1 == ts.num_sites
        assert list(output_sites[0].split()) == [
            "position",
            "ancestral_state",
            "metadata",
        ]
        for site, line in zip(ts.sites(), output_sites[1:]):
            splits = line.split("\t")
            assert convert(site.position) == splits[0]
            assert site.ancestral_state == splits[1]
            assert tests.base64_encode(site.metadata) == splits[2]

    def verify_mutations_format(self, ts, mutations_file, precision):
        """
        Verifies that the mutations we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_mutations = mutations_file.read().splitlines()
        assert len(output_mutations) - 1 == ts.num_mutations
        assert list(output_mutations[0].split()) == [
            "site",
            "node",
            "time",
            "derived_state",
            "parent",
            "metadata",
        ]
        mutations = [mut for site in ts.sites() for mut in site.mutations]
        for mutation, line in zip(mutations, output_mutations[1:]):
            splits = line.split("\t")
            assert str(mutation.site) == splits[0]
            assert str(mutation.node) == splits[1]
            assert (
                "unknown" if util.is_unknown_time(mutation.time) else str(mutation.time)
            ) == splits[2]
            assert str(mutation.derived_state) == splits[3]
            assert str(mutation.parent) == splits[4]
            assert tests.base64_encode(mutation.metadata) == splits[5]

    def test_output_format(self):
        for ts in get_example_tree_sequences():
            for precision in [2, 7]:
                nodes_file = io.StringIO()
                edges_file = io.StringIO()
                sites_file = io.StringIO()
                mutations_file = io.StringIO()
                ts.dump_text(
                    nodes=nodes_file,
                    edges=edges_file,
                    sites=sites_file,
                    mutations=mutations_file,
                    precision=precision,
                )
                nodes_file.seek(0)
                edges_file.seek(0)
                sites_file.seek(0)
                mutations_file.seek(0)
                self.verify_nodes_format(ts, nodes_file, precision)
                self.verify_edges_format(ts, edges_file, precision)
                self.verify_sites_format(ts, sites_file, precision)
                self.verify_mutations_format(ts, mutations_file, precision)

    def verify_approximate_equality(self, ts1, ts2):
        """
        Verifies that the specified tree sequences are approximately
        equal, taking into account the error incurred in exporting to text.
        """
        assert ts1.sample_size == ts2.sample_size
        assert ts1.sequence_length == ts2.sequence_length
        assert ts1.num_nodes == ts2.num_nodes
        assert ts1.num_edges == ts2.num_edges
        assert ts1.num_sites == ts2.num_sites
        assert ts1.num_mutations == ts2.num_mutations

        checked = 0
        for n1, n2 in zip(ts1.nodes(), ts2.nodes()):
            assert n1.population == n2.population
            assert n1.metadata == n2.metadata
            assert n1.time == pytest.approx(n2.time)
            checked += 1
        assert checked == ts1.num_nodes

        checked = 0
        for r1, r2 in zip(ts1.edges(), ts2.edges()):
            checked += 1
            assert r1.left == pytest.approx(r2.left)
            assert r1.right == pytest.approx(r2.right)
            assert r1.parent == r2.parent
            assert r1.child == r2.child
        assert ts1.num_edges == checked

        checked = 0
        for s1, s2 in zip(ts1.sites(), ts2.sites()):
            checked += 1
            assert s1.position == pytest.approx(s2.position)
            assert s1.ancestral_state == s2.ancestral_state
            assert s1.metadata == s2.metadata
            assert s1.mutations == s2.mutations
        assert ts1.num_sites == checked

        checked = 0
        for s1, s2 in zip(ts1.mutations(), ts2.mutations()):
            checked += 1
            assert s1.site == s2.site
            assert s1.node == s2.node
            if not (math.isnan(s1.time) and math.isnan(s2.time)):
                assert s1.time == pytest.approx(s2.time)
            assert s1.derived_state == s2.derived_state
            assert s1.parent == s2.parent
            assert s1.metadata == s2.metadata
        assert ts1.num_mutations == checked

        # Check the trees
        check = 0
        for t1, t2 in zip(ts1.trees(), ts2.trees()):
            assert list(t1.nodes()) == list(t2.nodes())
            check += 1
        assert check == ts1.get_num_trees()

    def test_text_record_round_trip(self):
        for ts1 in get_example_tree_sequences():
            nodes_file = io.StringIO()
            edges_file = io.StringIO()
            sites_file = io.StringIO()
            mutations_file = io.StringIO()
            individuals_file = io.StringIO()
            populations_file = io.StringIO()
            ts1.dump_text(
                nodes=nodes_file,
                edges=edges_file,
                sites=sites_file,
                mutations=mutations_file,
                individuals=individuals_file,
                populations=populations_file,
                precision=16,
            )
            nodes_file.seek(0)
            edges_file.seek(0)
            sites_file.seek(0)
            mutations_file.seek(0)
            individuals_file.seek(0)
            populations_file.seek(0)
            ts2 = tskit.load_text(
                nodes=nodes_file,
                edges=edges_file,
                sites=sites_file,
                mutations=mutations_file,
                individuals=individuals_file,
                populations=populations_file,
                sequence_length=ts1.sequence_length,
                strict=True,
            )
            self.verify_approximate_equality(ts1, ts2)

    def test_empty_files(self):
        nodes_file = io.StringIO("is_sample\ttime\n")
        edges_file = io.StringIO("left\tright\tparent\tchild\n")
        sites_file = io.StringIO("position\tancestral_state\n")
        mutations_file = io.StringIO("site\tnode\tderived_state\n")
        with pytest.raises(_tskit.LibraryError):
            tskit.load_text(
                nodes=nodes_file,
                edges=edges_file,
                sites=sites_file,
                mutations=mutations_file,
            )

    def test_empty_files_sequence_length(self):
        nodes_file = io.StringIO("is_sample\ttime\n")
        edges_file = io.StringIO("left\tright\tparent\tchild\n")
        sites_file = io.StringIO("position\tancestral_state\n")
        mutations_file = io.StringIO("site\tnode\tderived_state\n")
        ts = tskit.load_text(
            nodes=nodes_file,
            edges=edges_file,
            sites=sites_file,
            mutations=mutations_file,
            sequence_length=100,
        )
        assert ts.sequence_length == 100
        assert ts.num_nodes == 0
        assert ts.num_edges == 0
        assert ts.num_sites == 0
        assert ts.num_edges == 0


class TestTree(HighLevelTestCase):
    """
    Some simple tests on the tree API.
    """

    def get_tree(self, sample_lists=False):
        ts = msprime.simulate(10, random_seed=1, mutation_rate=1, record_full_arg=True)
        return next(ts.trees(sample_lists=sample_lists))

    def verify_mutations(self, tree):
        assert tree.num_mutations > 0
        other_mutations = []
        for site in tree.sites():
            for mutation in site.mutations:
                other_mutations.append(mutation)
        mutations = list(tree.mutations())
        assert tree.num_mutations == len(other_mutations)
        assert tree.num_mutations == len(mutations)
        for mut, other_mut in zip(mutations, other_mutations):
            # We cannot compare these directly as the mutations obtained
            # from the mutations iterator will have extra deprecated
            # attributes.
            assert mut.id == other_mut.id
            assert mut.site == other_mut.site
            assert mut.parent == other_mut.parent
            assert mut.node == other_mut.node
            assert mut.metadata == other_mut.metadata
            # Check the deprecated attrs.
            assert mut.position == tree.tree_sequence.site(mut.site).position
            assert mut.index == mut.site

    def test_simple_mutations(self):
        tree = self.get_tree()
        self.verify_mutations(tree)

    def test_complex_mutations(self):
        ts = tsutil.insert_branch_mutations(msprime.simulate(10, random_seed=1))
        self.verify_mutations(ts.first())

    def test_str(self):
        t = self.get_tree()
        assert isinstance(str(t), str)
        assert str(t) == str(t.get_parent_dict())

    def test_samples(self):
        for sample_lists in [True, False]:
            t = self.get_tree(sample_lists)
            n = t.get_sample_size()
            all_samples = list(t.samples(t.get_root()))
            assert sorted(all_samples) == list(range(n))
            for j in range(n):
                assert list(t.samples(j)) == [j]

            def test_func(t, u):
                """
                Simple test definition of the traversal.
                """
                stack = [u]
                while len(stack) > 0:
                    v = stack.pop()
                    if t.is_sample(v):
                        yield v
                    if t.is_internal(v):
                        for c in reversed(t.get_children(v)):
                            stack.append(c)

            for u in t.nodes():
                l1 = list(t.samples(u))
                l2 = list(test_func(t, u))
                assert l1 == l2
                assert t.get_num_samples(u) == len(l1)

    def test_num_children(self):
        tree = self.get_tree()
        for u in tree.nodes():
            assert tree.num_children(u) == len(tree.children(u))

    def verify_newick(self, tree):
        """
        Verifies that we output the newick tree as expected.
        """
        # TODO to make this work we may need to clamp the precision of node
        # times because Python and C float printing algorithms work slightly
        # differently. Seems to work OK now, so leaving alone.
        if tree.num_roots == 1:
            py_tree = tests.PythonTree.from_tree(tree)
            newick1 = tree.newick(precision=16)
            newick2 = py_tree.newick()
            assert newick1 == newick2

            # Make sure we get the same results for a leaf root.
            newick1 = tree.newick(root=0, precision=16)
            newick2 = py_tree.newick(root=0)
            assert newick1 == newick2

            # When we specify the node_labels we should get precisely the
            # same result as we are using Python code now.
            for precision in [0, 3, 19]:
                newick1 = tree.newick(precision=precision, node_labels={})
                newick2 = py_tree.newick(precision=precision, node_labels={})
                assert newick1 == newick2
        else:
            with pytest.raises(ValueError):
                tree.newick()
            for root in tree.roots:
                py_tree = tests.PythonTree.from_tree(tree)
                newick1 = tree.newick(precision=16, root=root)
                newick2 = py_tree.newick(root=root)
                assert newick1 == newick2

    def test_newick(self):
        for ts in get_example_tree_sequences():
            for tree in ts.trees():
                self.verify_newick(tree)

    def test_newick_large_times(self):
        for n in [2, 10, 20, 100]:
            ts = msprime.simulate(n, Ne=100e6, random_seed=1)
            tree = ts.first()
            for precision in [0, 1, 16]:
                newick_py = tree.newick(
                    node_labels={u: str(u + 1) for u in ts.samples()},
                    precision=precision,
                )
                newick_c = tree.newick(precision=precision)
                assert newick_c == newick_py

    def test_bifurcating_newick(self):
        for n_tips in range(2, 6):
            ts = msprime.simulate(n_tips, random_seed=1)  # msprime trees are binary
            for tree in ts.trees():
                base_newick = tree.newick(include_branch_lengths=False).strip(";")
                for i in range(n_tips):
                    # Each tip number (i+1) mentioned once
                    assert base_newick.count(str(i + 1)) == 1
                # Binary newick trees have 3 chars per extra tip: "(,)"
                assert len(base_newick) == n_tips + 3 * (n_tips - 1)

    def test_newick_topology_equiv(self):
        replace_numeric = {ord(x): None for x in "1234567890:."}
        for ts in get_example_tree_sequences():
            for tree in ts.trees():
                if tree.num_roots > 1:
                    continue
                plain_newick = tree.newick(node_labels={}, include_branch_lengths=False)
                newick1 = tree.newick().translate(replace_numeric)
                newick2 = tree.newick(node_labels={}).translate(replace_numeric)
                assert newick1 == newick2 == plain_newick

    def test_newick_buffer_too_small_bug(self):
        nodes = io.StringIO(
            """\
        id  is_sample   population individual time
        0       1       0       -1      0.00000000000000
        1       1       0       -1      0.00000000000000
        2       1       0       -1      0.00000000000000
        3       1       0       -1      0.00000000000000
        4       0       0       -1      0.21204940078588
        5       0       0       -1      0.38445004304611
        6       0       0       -1      0.83130278081275
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.00000000      1.00000000      4       0
        1       0.00000000      1.00000000      4       2
        2       0.00000000      1.00000000      5       1
        3       0.00000000      1.00000000      5       3
        4       0.00000000      1.00000000      6       4
        5       0.00000000      1.00000000      6       5
        """
        )
        ts = tskit.load_text(nodes, edges, sequence_length=1, strict=False)
        tree = ts.first()
        for precision in range(17):
            newick_c = tree.newick(precision=precision)
            node_labels = {u: str(u + 1) for u in ts.samples()}
            newick_py = tree.newick(precision=precision, node_labels=node_labels)
            assert newick_c == newick_py

    def test_as_dict_of_dicts(self):
        for ts in get_example_tree_sequences():
            tree = next(ts.trees())
            adj_dod = tree.as_dict_of_dicts()
            g = nx.DiGraph(adj_dod)

            self.verify_nx_graph_topology(tree, g)
            self.verify_nx_algorithm_equivalence(tree, g)
            self.verify_nx_for_tutorial_algorithms(tree, g)
        self.verify_nx_nearest_neighbor_search()

    def verify_nx_graph_topology(self, tree, g):
        assert set(tree.nodes()) == set(g.nodes)

        assert set(tree.roots) == {n for n in g.nodes if g.in_degree(n) == 0}

        assert set(tree.leaves()) == {n for n in g.nodes if g.out_degree(n) == 0}

        # test if tree has no in-degrees > 1
        assert nx.is_branching(g)

    def verify_nx_algorithm_equivalence(self, tree, g):
        for root in tree.roots:
            assert nx.is_directed_acyclic_graph(g)

            # test descendants
            assert {u for u in tree.nodes() if tree.is_descendant(u, root)} == set(
                nx.descendants(g, root)
            ) | {root}

            # test MRCA
            if tree.num_nodes < 20:
                for u, v in itertools.combinations(tree.nodes(), 2):
                    mrca = nx.lowest_common_ancestor(g, u, v)
                    if mrca is None:
                        mrca = -1
                    assert tree.mrca(u, v) == mrca

            # test node traversal modes
            assert list(tree.nodes(root=root, order="breadthfirst")) == [root] + [
                v for u, v in nx.bfs_edges(g, root)
            ]
            assert list(tree.nodes(root=root, order="preorder")) == list(
                nx.dfs_preorder_nodes(g, root)
            )

    def verify_nx_for_tutorial_algorithms(self, tree, g):
        # traversing upwards
        for u in tree.leaves():
            path = []
            v = u
            while v != tskit.NULL:
                path.append(v)
                v = tree.parent(v)

            assert set(path) == {u} | nx.ancestors(g, u)
            assert path == [u] + [
                n1 for n1, n2, _ in nx.edge_dfs(g, u, orientation="reverse")
            ]

        # traversals with information
        def preorder_dist(tree, root):
            stack = [(root, 0)]
            while len(stack) > 0:
                u, distance = stack.pop()
                yield u, distance
                for v in tree.children(u):
                    stack.append((v, distance + 1))

        for root in tree.roots:
            assert {
                k: v for k, v in preorder_dist(tree, root)
            } == nx.shortest_path_length(g, source=root)

        for root in tree.roots:
            # new traversal: measuring time between root and MRCA
            for u, v in itertools.combinations(nx.descendants(g, root), 2):
                mrca = tree.mrca(u, v)
                tmrca = tree.time(mrca)
                assert tree.time(root) - tmrca == pytest.approx(
                    nx.shortest_path_length(
                        g, source=root, target=mrca, weight="branch_length"
                    )
                )

    def verify_nx_nearest_neighbor_search(self):
        samples = [
            msprime.Sample(0, 0),
            msprime.Sample(0, 1),
            msprime.Sample(0, 20),
        ]
        ts = msprime.simulate(
            Ne=1e6,
            samples=samples,
            demographic_events=[
                msprime.PopulationParametersChange(
                    time=10, growth_rate=2, population_id=0
                ),
            ],
            random_seed=42,
        )

        tree = ts.first()
        g = nx.Graph(tree.as_dict_of_dicts())

        dist_dod = collections.defaultdict(dict)
        for source, target in itertools.combinations(tree.samples(), 2):
            dist_dod[source][target] = nx.shortest_path_length(
                g, source=source, target=target, weight="branch_length"
            )
            dist_dod[target][source] = dist_dod[source][target]

        nearest_neighbor_of = [min(dist_dod[u], key=dist_dod[u].get) for u in range(3)]
        assert [2, 2, 1] == [nearest_neighbor_of[u] for u in range(3)]

    def test_traversals(self):
        for ts in get_example_tree_sequences():
            tree = next(ts.trees())
            self.verify_traversals(tree)

            # Verify time-ordered traversals separately, because the PythonTree
            # class does not contain time information at the moment
            for root in tree.roots:
                time_ordered = tree.nodes(root, order="timeasc")
                t = tree.time(next(time_ordered))
                for u in time_ordered:
                    next_t = tree.time(u)
                    assert next_t >= t
                    t = next_t
                time_ordered = tree.nodes(root, order="timedesc")
                t = tree.time(next(time_ordered))
                for u in time_ordered:
                    next_t = tree.time(u)
                    assert next_t <= t
                    t = next_t

    def verify_traversals(self, tree):
        t1 = tree
        t2 = tests.PythonTree.from_tree(t1)
        assert list(t1.nodes()) == list(t2.nodes())
        orders = [
            "inorder",
            "postorder",
            "levelorder",
            "breadthfirst",
            "minlex_postorder",
        ]
        if tree.num_roots == 1:
            with pytest.raises(ValueError):
                list(t1.nodes(order="bad order"))
            assert list(t1.nodes()) == list(t1.nodes(t1.get_root()))
            assert list(t1.nodes()) == list(t1.nodes(t1.get_root(), "preorder"))
            for u in t1.nodes():
                assert list(t1.nodes(u)) == list(t2.nodes(u))
            for test_order in orders:
                assert sorted(list(t1.nodes())) == sorted(
                    list(t1.nodes(order=test_order))
                )
                assert list(t1.nodes(order=test_order)) == list(
                    t1.nodes(t1.get_root(), order=test_order)
                )
                assert list(t1.nodes(order=test_order)) == list(
                    t1.nodes(t1.get_root(), test_order)
                )
                assert list(t1.nodes(order=test_order)) == list(
                    t2.nodes(order=test_order)
                )
                for u in t1.nodes():
                    assert list(t1.nodes(u, test_order)) == list(
                        t2.nodes(u, test_order)
                    )
        else:
            for test_order in orders:
                all_nodes = []
                for root in t1.roots:
                    assert list(t1.nodes(root, order=test_order)) == list(
                        t2.nodes(root, order=test_order)
                    )
                    all_nodes.extend(t1.nodes(root, order=test_order))
                # minlex_postorder reorders the roots, so this last test is
                # not appropriate
                if test_order != "minlex_postorder":
                    assert all_nodes == list(t1.nodes(order=test_order))

    def test_total_branch_length(self):
        # Note: this definition works when we have no non-sample branches.
        t1 = self.get_tree()
        bl = 0
        root = t1.get_root()
        for node in t1.nodes():
            if node != root:
                bl += t1.get_branch_length(node)
        assert bl > 0
        assert t1.get_total_branch_length() == pytest.approx(bl)

    def test_branch_length_empty_tree(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        ts = tables.tree_sequence()
        assert ts.num_trees == 1
        tree = ts.first()
        assert tree.branch_length(0) == 0
        assert tree.branch_length(1) == 0
        assert tree.total_branch_length == 0

    def test_is_descendant(self):
        def is_descendant(tree, u, v):
            path = []
            while u != tskit.NULL:
                path.append(u)
                u = tree.parent(u)
            return v in path

        tree = self.get_tree()
        for u, v in itertools.product(range(tree.num_nodes), repeat=2):
            assert is_descendant(tree, u, v) == tree.is_descendant(u, v)
        for bad_node in [-1, -2, tree.num_nodes, tree.num_nodes + 1]:
            with pytest.raises(ValueError):
                tree.is_descendant(0, bad_node)
            with pytest.raises(ValueError):
                tree.is_descendant(bad_node, 0)
            with pytest.raises(ValueError):
                tree.is_descendant(bad_node, bad_node)

    def test_apis(self):
        # tree properties
        t1 = self.get_tree()
        assert t1.get_root() == t1.root
        assert t1.get_index() == t1.index
        assert t1.get_interval() == t1.interval
        assert t1.get_sample_size() == t1.sample_size
        assert t1.get_num_mutations() == t1.num_mutations
        assert t1.get_parent_dict() == t1.parent_dict
        assert t1.get_total_branch_length() == t1.total_branch_length
        assert t1.span == t1.interval[1] - t1.interval[0]
        # node properties
        root = t1.get_root()
        for node in t1.nodes():
            if node != root:
                assert t1.get_time(node) == t1.time(node)
                assert t1.get_parent(node) == t1.parent(node)
                assert t1.get_children(node) == t1.children(node)
                assert t1.get_population(node) == t1.population(node)
                assert t1.get_num_samples(node) == t1.num_samples(node)
                assert t1.get_branch_length(node) == t1.branch_length(node)
                assert t1.get_num_tracked_samples(node) == t1.num_tracked_samples(node)

        pairs = itertools.islice(itertools.combinations(t1.nodes(), 2), 50)
        for pair in pairs:
            assert t1.get_mrca(*pair) == t1.mrca(*pair)
            assert t1.get_tmrca(*pair) == t1.tmrca(*pair)

    def test_deprecated_apis(self):
        t1 = self.get_tree()
        assert t1.get_length() == t1.span
        assert t1.length == t1.span

    def test_seek_index(self):
        ts = msprime.simulate(10, recombination_rate=3, length=5, random_seed=42)
        N = ts.num_trees
        assert ts.num_trees > 3
        tree = tskit.Tree(ts)
        for index in [0, N // 2, N - 1, 1]:
            fresh_tree = tskit.Tree(ts)
            assert fresh_tree.index == -1
            fresh_tree.seek_index(index)
            tree.seek_index(index)
            assert fresh_tree.index == index
            assert tree.index == index

        tree = tskit.Tree(ts)
        for index in [-1, -2, -N + 2, -N + 1, -N]:
            fresh_tree = tskit.Tree(ts)
            assert fresh_tree.index == -1
            fresh_tree.seek_index(index)
            tree.seek_index(index)
            assert fresh_tree.index == index + N
            assert tree.index == index + N
        with pytest.raises(IndexError):
            tree.seek_index(N)
        with pytest.raises(IndexError):
            tree.seek_index(N + 1)
        with pytest.raises(IndexError):
            tree.seek_index(-N - 1)
        with pytest.raises(IndexError):
            tree.seek_index(-N - 2)

    def test_first_last(self):
        ts = msprime.simulate(10, recombination_rate=3, length=2, random_seed=42)
        assert ts.num_trees > 3
        tree = tskit.Tree(ts)
        tree.first()
        assert tree.index == 0
        tree = tskit.Tree(ts)
        tree.last()
        assert tree.index == ts.num_trees - 1
        tree = tskit.Tree(ts)
        for _ in range(3):
            tree.last()
            assert tree.index == ts.num_trees - 1
            tree.first()
            assert tree.index == 0

    def test_eq_different_tree_sequence(self):
        ts = msprime.simulate(4, recombination_rate=1, length=2, random_seed=42)
        copy = ts.tables.tree_sequence()
        for tree1, tree2 in zip(ts.aslist(), copy.aslist()):
            assert tree1 != tree2

    def test_next_prev(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        assert ts.num_trees > 5
        for index, tree in enumerate(ts.aslist()):
            assert tree.index == index
            j = index
            while tree.next():
                j += 1
                assert tree.index == j
            assert tree.index == -1
            assert j + 1 == ts.num_trees
        for index, tree in enumerate(ts.aslist()):
            assert tree.index == index
            j = index
            while tree.prev():
                j -= 1
                assert tree.index == j
            assert tree.index == -1
            assert j == 0
        tree.first()
        tree.prev()
        assert tree.index == -1
        tree.last()
        tree.next()
        assert tree.index == -1

    def test_seek(self):
        L = 10
        ts = msprime.simulate(10, recombination_rate=3, length=L, random_seed=42)
        assert ts.num_trees > 5
        same_tree = tskit.Tree(ts)
        for tree in [same_tree, tskit.Tree(ts)]:
            for j in range(L):
                tree.seek(j)
                index = tree.index
                assert tree.interval[0] <= j < tree.interval[1]
                tree.seek(tree.interval[0])
                assert tree.index == index
                if tree.interval[1] < L:
                    tree.seek(tree.interval[1])
                    assert tree.index == index + 1
            for j in reversed(range(L)):
                tree.seek(j)
                assert tree.interval[0] <= j < tree.interval[1]
        for bad_position in [-1, L, L + 1, -L]:
            with pytest.raises(ValueError):
                tree.seek(bad_position)

    def test_interval(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=1)
        assert ts.num_trees > 1
        breakpoints = list(ts.breakpoints())
        assert breakpoints[0] == 0
        assert breakpoints[-1] == ts.sequence_length
        for i, tree in enumerate(ts.trees()):
            assert tree.interval[0] == pytest.approx(breakpoints[i])
            assert tree.interval.left == pytest.approx(breakpoints[i])
            assert tree.interval[1] == pytest.approx(breakpoints[i + 1])
            assert tree.interval.right == pytest.approx(breakpoints[i + 1])
            assert tree.interval.span == pytest.approx(
                breakpoints[i + 1] - breakpoints[i]
            )

    def verify_empty_tree(self, tree):
        ts = tree.tree_sequence
        assert tree.index == -1
        assert tree.parent_dict == {}
        for u in range(ts.num_nodes):
            assert tree.parent(u) == tskit.NULL
            assert tree.left_child(u) == tskit.NULL
            assert tree.right_child(u) == tskit.NULL
            if not ts.node(u).is_sample():
                assert tree.left_sib(u) == tskit.NULL
                assert tree.right_sib(u) == tskit.NULL
        # Samples should have left-sib right-sibs set
        samples = ts.samples()
        assert tree.left_root == samples[0]
        for j in range(ts.num_samples):
            if j > 0:
                assert tree.left_sib(samples[j]) == samples[j - 1]
            if j < ts.num_samples - 1:
                assert tree.right_sib(samples[j]) == samples[j + 1]

    def test_empty_tree(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        assert ts.num_trees > 5
        tree = tskit.Tree(ts)
        self.verify_empty_tree(tree)
        while tree.next():
            pass
        self.verify_empty_tree(tree)
        while tree.prev():
            pass
        self.verify_empty_tree(tree)

    def test_clear(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        assert ts.num_trees > 5
        tree = tskit.Tree(ts)
        tree.first()
        tree.clear()
        self.verify_empty_tree(tree)
        tree.last()
        tree.clear()
        self.verify_empty_tree(tree)
        tree.seek_index(ts.num_trees // 2)
        tree.clear()
        self.verify_empty_tree(tree)

    def verify_trees_identical(self, t1, t2):
        assert t1.tree_sequence is t2.tree_sequence
        assert t1.num_nodes is t2.num_nodes
        assert [t1.parent(u) for u in range(t1.num_nodes)] == [
            t2.parent(u) for u in range(t2.num_nodes)
        ]
        assert [t1.left_child(u) for u in range(t1.num_nodes)] == [
            t2.left_child(u) for u in range(t2.num_nodes)
        ]
        assert [t1.right_child(u) for u in range(t1.num_nodes)] == [
            t2.right_child(u) for u in range(t2.num_nodes)
        ]
        assert [t1.left_sib(u) for u in range(t1.num_nodes)] == [
            t2.left_sib(u) for u in range(t2.num_nodes)
        ]
        assert [t1.right_sib(u) for u in range(t1.num_nodes)] == [
            t2.right_sib(u) for u in range(t2.num_nodes)
        ]
        assert list(t1.sites()) == list(t2.sites())

    def test_copy_seek(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        assert ts.num_trees > 5
        tree = tskit.Tree(ts)
        copy = tree.copy()
        self.verify_empty_tree(copy)
        while tree.next():
            copy = tree.copy()
            self.verify_trees_identical(tree, copy)
        while tree.prev():
            copy = tree.copy()
            self.verify_trees_identical(tree, copy)
        tree.clear()
        copy = tree.copy()
        tree.first()
        copy.first()
        while tree.index != -1:
            self.verify_trees_identical(tree, copy)
            assert tree.next() == copy.next()
        tree.last()
        copy.last()
        while tree.index != -1:
            self.verify_trees_identical(tree, copy)
            assert tree.prev() == copy.prev()
        # Seek to middle and two independent trees.
        tree.seek_index(ts.num_trees // 2)
        left_copy = tree.copy()
        right_copy = tree.copy()
        self.verify_trees_identical(tree, left_copy)
        self.verify_trees_identical(tree, right_copy)
        left_copy.prev()
        assert left_copy.index == tree.index - 1
        right_copy.next()
        assert right_copy.index == tree.index + 1

    def test_copy_tracked_samples(self):
        ts = msprime.simulate(10, recombination_rate=2, length=3, random_seed=42)
        tree = tskit.Tree(ts, tracked_samples=[0, 1])
        while tree.next():
            copy = tree.copy()
            for j in range(ts.num_nodes):
                assert tree.num_tracked_samples(j) == copy.num_tracked_samples(j)
        copy = tree.copy()
        while tree.next():
            copy.next()
            for j in range(ts.num_nodes):
                assert tree.num_tracked_samples(j) == copy.num_tracked_samples(j)

    def test_copy_multiple_roots(self):
        ts = msprime.simulate(20, recombination_rate=2, length=3, random_seed=42)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        for root_threshold in [1, 2, 100]:
            tree = tskit.Tree(ts, root_threshold=root_threshold)
            copy = tree.copy()
            assert copy.roots == tree.roots
            assert copy.root_threshold == root_threshold
            while tree.next():
                copy = tree.copy()
                assert copy.roots == tree.roots
                assert copy.root_threshold == root_threshold
            copy = tree.copy()
            assert copy.roots == tree.roots
            assert copy.root_threshold == root_threshold

    def test_map_mutations(self):
        ts = msprime.simulate(5, random_seed=42)
        tree = ts.first()
        genotypes = np.zeros(5, dtype=np.int8)
        alleles = [str(j) for j in range(64)]
        ancestral_state, transitions = tree.map_mutations(genotypes, alleles)
        assert ancestral_state == "0"
        assert len(transitions) == 0
        for j in range(1, 64):
            genotypes[0] = j
            ancestral_state, transitions = tree.map_mutations(genotypes, alleles)
            assert ancestral_state == "0"
            assert len(transitions) == 1
        for j in range(64, 67):
            genotypes[0] = j
            with pytest.raises(ValueError):
                tree.map_mutations(genotypes, alleles)
        tree.map_mutations([0] * 5, alleles)
        tree.map_mutations(np.zeros(5, dtype=int), alleles)

    def test_sample_count_deprecated(self):
        ts = msprime.simulate(5, random_seed=42)
        with warnings.catch_warnings(record=True) as w:
            ts.trees(sample_counts=True)
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)

        with warnings.catch_warnings(record=True) as w:
            tskit.Tree(ts, sample_counts=False)
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)


class TestNodeOrdering(HighLevelTestCase):
    """
    Verify that we can use any node ordering for internal nodes
    and get the same topologies.
    """

    num_random_permutations = 10

    def verify_tree_sequences_equal(self, ts1, ts2, approximate=False):
        assert ts1.get_num_trees() == ts2.get_num_trees()
        assert ts1.get_sample_size() == ts2.get_sample_size()
        assert ts1.get_num_nodes() == ts2.get_num_nodes()
        j = 0
        for r1, r2 in zip(ts1.edges(), ts2.edges()):
            assert r1.parent == r2.parent
            assert r1.child == r2.child
            if approximate:
                assert r1.left == pytest.approx(r2.left)
                assert r1.right == pytest.approx(r2.right)
            else:
                assert r1.left == r2.left
                assert r1.right == r2.right
            j += 1
        assert ts1.num_edges == j
        j = 0
        for n1, n2 in zip(ts1.nodes(), ts2.nodes()):
            assert n1.metadata == n2.metadata
            assert n1.population == n2.population
            if approximate:
                assert n1.time == pytest.approx(n2.time)
            else:
                assert n1.time == n2.time
            j += 1
        assert ts1.num_nodes == j

    def verify_random_permutation(self, ts):
        n = ts.sample_size
        node_map = {}
        for j in range(n):
            node_map[j] = j
        internal_nodes = list(range(n, ts.num_nodes))
        random.shuffle(internal_nodes)
        for j, node in enumerate(internal_nodes):
            node_map[n + j] = node
        other_tables = tskit.TableCollection(ts.sequence_length)
        # Insert the new nodes into the table.
        inv_node_map = {v: k for k, v in node_map.items()}
        for j in range(ts.num_nodes):
            node = ts.node(inv_node_map[j])
            other_tables.nodes.add_row(
                flags=node.flags, time=node.time, population=node.population
            )
        for e in ts.edges():
            other_tables.edges.add_row(
                left=e.left,
                right=e.right,
                parent=node_map[e.parent],
                child=node_map[e.child],
            )
        for _ in range(ts.num_populations):
            other_tables.populations.add_row()
        other_tables.sort()
        other_ts = other_tables.tree_sequence()

        assert ts.get_num_trees() == other_ts.get_num_trees()
        assert ts.get_sample_size() == other_ts.get_sample_size()
        assert ts.get_num_nodes() == other_ts.get_num_nodes()
        j = 0
        for t1, t2 in zip(ts.trees(), other_ts.trees()):
            # Verify the topologies are identical. We do this by traversing
            # upwards to the root for every sample and checking if we map to
            # the correct node and time.
            for u in range(n):
                v_orig = u
                v_map = u
                while v_orig != tskit.NULL:
                    assert node_map[v_orig] == v_map
                    assert t1.get_time(v_orig) == t2.get_time(v_map)
                    v_orig = t1.get_parent(v_orig)
                    v_map = t2.get_parent(v_map)
                assert v_orig == tskit.NULL
                assert v_map == tskit.NULL
            j += 1
        assert j == ts.get_num_trees()
        # Verify we can dump this new tree sequence OK.
        with tempfile.TemporaryDirectory() as tempdir:
            temp_file = pathlib.Path(tempdir) / "tmp.trees"
            other_ts.dump(temp_file)
            ts3 = tskit.load(temp_file)
        self.verify_tree_sequences_equal(other_ts, ts3)
        nodes_file = io.StringIO()
        edges_file = io.StringIO()
        # Also verify we can read the text version.
        other_ts.dump_text(nodes=nodes_file, edges=edges_file, precision=14)
        nodes_file.seek(0)
        edges_file.seek(0)
        ts3 = tskit.load_text(nodes_file, edges_file)
        self.verify_tree_sequences_equal(other_ts, ts3, True)

    def test_single_locus(self):
        ts = msprime.simulate(7)
        for _ in range(self.num_random_permutations):
            self.verify_random_permutation(ts)

    def test_multi_locus(self):
        ts = msprime.simulate(20, recombination_rate=10)
        for _ in range(self.num_random_permutations):
            self.verify_random_permutation(ts)

    def test_nonbinary(self):
        ts = msprime.simulate(
            sample_size=20,
            recombination_rate=10,
            demographic_events=[
                msprime.SimpleBottleneck(time=0.5, population=0, proportion=1)
            ],
        )
        # Make sure this really has some non-binary nodes
        found = False
        for t in ts.trees():
            for u in t.nodes():
                if len(t.children(u)) > 2:
                    found = True
                    break
            if found:
                break
        assert found
        for _ in range(self.num_random_permutations):
            self.verify_random_permutation(ts)


class SimpleContainersMixin:
    """
    Tests for the SimpleContainer classes.
    """

    def test_equality(self):
        c1, c2 = self.get_instances(2)
        assert c1 == c1
        assert not (c1 == c2)
        assert not (c1 != c1)
        assert c1 != c2
        (c3,) = self.get_instances(1)
        assert c1 == c3
        assert not (c1 != c3)

    def test_repr(self):
        (c,) = self.get_instances(1)
        assert len(repr(c)) > 0


class SimpleContainersWithMetadataMixin:
    """
    Tests for the SimpleContainerWithMetadata classes.
    """

    def test_metadata(self):
        # Test decoding
        instances = self.get_instances(5)
        for j, inst in enumerate(instances):
            assert inst.metadata == ("x" * j) + "decoded"

        # Decoder doesn't effect equality
        (inst,) = self.get_instances(1)
        (inst2,) = self.get_instances(1)
        assert inst == inst2
        inst._metadata_decoder = lambda m: "different decoder"
        assert inst == inst2
        inst._encoded_metadata = b"different"
        assert not (inst == inst2)

    def test_decoder_run_once(self):
        # For a given instance, the decoded metadata should be cached, with the decoder
        # called once
        (inst,) = self.get_instances(1)
        times_run = 0

        def decoder(m):
            nonlocal times_run
            times_run += 1
            return m.decode() + "decoded"

        inst._metadata_decoder = decoder
        assert times_run == 0
        _ = inst.metadata
        assert times_run == 1
        _ = inst.metadata
        assert times_run == 1


class TestIndividualContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Individual(
                id_=j,
                flags=j,
                location=[j],
                nodes=[j],
                encoded_metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestNodeContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Node(
                id_=j,
                flags=j,
                time=j,
                population=j,
                individual=j,
                encoded_metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestEdgeContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Edge(
                left=j,
                right=j,
                parent=j,
                child=j,
                encoded_metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
                id_=j,
            )
            for j in range(n)
        ]


class TestSiteContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Site(
                id_=j,
                position=j,
                ancestral_state="A" * j,
                mutations=TestMutationContainer().get_instances(j),
                encoded_metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestMutationContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Mutation(
                id_=j,
                site=j,
                node=j,
                time=j,
                derived_state="A" * j,
                parent=j,
                encoded_metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]

    def test_nan_equality(self):
        a = tskit.Mutation(
            id_=42,
            site=42,
            node=42,
            time=UNKNOWN_TIME,
            derived_state="A" * 42,
            parent=42,
            encoded_metadata=b"x" * 42,
            metadata_decoder=lambda m: m.decode() + "decoded",
        )
        b = tskit.Mutation(
            id_=42,
            site=42,
            node=42,
            derived_state="A" * 42,
            parent=42,
            encoded_metadata=b"x" * 42,
            metadata_decoder=lambda m: m.decode() + "decoded",
        )
        c = tskit.Mutation(
            id_=42,
            site=42,
            node=42,
            time=math.nan,
            derived_state="A" * 42,
            parent=42,
            encoded_metadata=b"x" * 42,
            metadata_decoder=lambda m: m.decode() + "decoded",
        )
        assert a == a
        assert a == b
        assert not (a == c)
        assert not (b == c)
        assert not (a != a)
        assert not (a != b)
        assert a != c
        assert c != c
        assert not (c == c)


class TestMigrationContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Migration(
                left=j,
                right=j,
                node=j,
                source=j,
                dest=j,
                time=j,
                encoded_metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestPopulationContainer(SimpleContainersMixin, SimpleContainersWithMetadataMixin):
    def get_instances(self, n):
        return [
            tskit.Population(
                id_=j,
                encoded_metadata=b"x" * j,
                metadata_decoder=lambda m: m.decode() + "decoded",
            )
            for j in range(n)
        ]


class TestProvenanceContainer(SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Provenance(id_=j, timestamp="x" * j, record="y" * j) for j in range(n)
        ]


class TestEdgesetContainer(SimpleContainersMixin):
    def get_instances(self, n):
        return [tskit.Edgeset(left=j, right=j, parent=j, children=j) for j in range(n)]


class TestVariantContainer(SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Variant(
                site=TestSiteContainer().get_instances(1)[0],
                alleles=["A" * j, "T"],
                genotypes=np.zeros(j, dtype=np.int8),
            )
            for j in range(n)
        ]


class TestTskitConversionOutput(unittest.TestCase):
    """
    Tests conversion output to ensure it is correct.
    """

    @classmethod
    def setUpClass(cls):
        ts = msprime.simulate(
            length=1,
            recombination_rate=2,
            mutation_rate=2,
            random_seed=1,
            migration_matrix=[[0, 1], [1, 0]],
            population_configurations=[
                msprime.PopulationConfiguration(5) for _ in range(2)
            ],
            record_migrations=True,
        )
        assert ts.num_migrations > 0
        cls._tree_sequence = tsutil.insert_random_ploidy_individuals(ts)

    def test_macs(self):
        output = self._tree_sequence.to_macs().splitlines()
        assert output[0].startswith("COMMAND:")
        assert output[1].startswith("SEED:")
        assert len(output) == 2 + self._tree_sequence.get_num_mutations()
        n = self._tree_sequence.get_sample_size()
        m = self._tree_sequence.get_sequence_length()
        sites = list(self._tree_sequence.sites())
        haplotypes = list(self._tree_sequence.haplotypes())
        for site_id, line in enumerate(output[2:]):
            splits = line.split()
            assert splits[0] == "SITE:"
            assert int(splits[1]) == site_id
            position = sites[site_id].position / m
            self.assertAlmostEqual(float(splits[2]), position)
            col = splits[4]
            assert len(col) == n
            for j in range(n):
                assert col[j] == haplotypes[j][site_id]
