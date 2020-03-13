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
import io
import itertools
import json
import math
import os
import pathlib
import pickle
import random
import shutil
import tempfile
import unittest
import uuid as _uuid
import warnings

import msprime
import networkx as nx
import numpy as np

import _tskit
import tests as tests
import tests.simplify as simplify
import tests.tsutil as tsutil
import tskit


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
        for ts in get_decapitated_examples():
            yield ts
        for ts in get_gap_examples():
            yield ts
    if internal_samples:
        for ts in get_internal_samples_examples():
            yield ts
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


class TestMRCACalculator(unittest.TestCase):
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
                        self.assertEqual(mrca, sv.get_mrca(j, k))


class HighLevelTestCase(unittest.TestCase):
    """
    Superclass of tests on the high level interface.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tsk_hl_testcase_")
        self.temp_file = os.path.join(self.temp_dir, "generic")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def verify_tree_mrcas(self, st):
        # Check the mrcas
        oriented_forest = [st.get_parent(j) for j in range(st.num_nodes)]
        mrca_calc = tests.MRCACalculator(oriented_forest)
        # We've done exhaustive tests elsewhere, no need to go
        # through the combinations.
        for j in range(st.num_nodes):
            mrca = st.get_mrca(0, j)
            self.assertEqual(mrca, mrca_calc.get_mrca(0, j))
            if mrca != tskit.NULL:
                self.assertEqual(st.get_time(mrca), st.get_tmrca(0, j))

    def verify_tree_branch_lengths(self, tree):
        for u in tree.tree_sequence.samples():
            while tree.parent(u) != tskit.NULL:
                length = tree.time(tree.parent(u)) - tree.time(u)
                self.assertGreater(length, 0.0)
                self.assertEqual(tree.branch_length(u), length)
                u = tree.parent(u)
            self.assertEqual(tree.parent(u), tskit.NULL)
            self.assertEqual(tree.branch_length(u), 0)

    def verify_tree_structure(self, st):
        roots = set()
        for u in st.samples():
            # verify the path to root
            self.assertTrue(st.is_sample(u))
            times = []
            while st.get_parent(u) != tskit.NULL:
                v = st.get_parent(u)
                times.append(st.get_time(v))
                self.assertGreaterEqual(st.get_time(v), 0.0)
                self.assertIn(u, st.get_children(v))
                u = v
            roots.add(u)
            self.assertEqual(times, sorted(times))
        self.assertEqual(sorted(list(roots)), sorted(st.roots))
        self.assertEqual(len(st.roots), st.num_roots)
        u = st.left_root
        roots = []
        while u != tskit.NULL:
            roots.append(u)
            u = st.right_sib(u)
        self.assertEqual(roots, st.roots)
        # To a top-down traversal, and make sure we meet all the samples.
        samples = []
        for root in st.roots:
            stack = [root]
            while len(stack) > 0:
                u = stack.pop()
                self.assertNotEqual(u, tskit.NULL)
                if st.is_sample(u):
                    samples.append(u)
                if st.is_leaf(u):
                    self.assertEqual(len(st.get_children(u)), 0)
                else:
                    for c in reversed(st.get_children(u)):
                        stack.append(c)
                # Check that we get the correct number of samples at each
                # node.
                self.assertEqual(st.get_num_samples(u), len(list(st.samples(u))))
                self.assertEqual(st.get_num_tracked_samples(u), 0)
        self.assertEqual(sorted(samples), sorted(st.samples()))
        # Check the parent dict
        pi = st.get_parent_dict()
        for root in st.roots:
            self.assertNotIn(root, pi)
        for k, v in pi.items():
            self.assertEqual(st.get_parent(k), v)
        self.assertEqual(st.num_samples(), len(samples))
        self.assertEqual(sorted(st.samples()), sorted(samples))

    def verify_tree(self, st):
        self.verify_tree_mrcas(st)
        self.verify_tree_branch_lengths(st)
        self.verify_tree_structure(st)

    def verify_trees(self, ts):
        pts = tests.PythonTreeSequence(ts)
        iter1 = ts.trees()
        iter2 = pts.trees()
        length = 0
        num_trees = 0
        breakpoints = [0]
        for st1, st2 in zip(iter1, iter2):
            self.assertEqual(st1.get_sample_size(), ts.get_sample_size())
            roots = set()
            for u in ts.samples():
                root = u
                while st1.get_parent(root) != tskit.NULL:
                    root = st1.get_parent(root)
                roots.add(root)
            self.assertEqual(st1.left_root, st2.left_root)
            self.assertEqual(sorted(list(roots)), sorted(st1.roots))
            self.assertEqual(st1.roots, st2.roots)
            if len(roots) > 1:
                with self.assertRaises(ValueError):
                    st1.root
            else:
                self.assertEqual(st1.root, list(roots)[0])
            self.assertEqual(st2, st1)
            self.assertFalse(st2 != st1)
            l, r = st1.get_interval()
            breakpoints.append(r)
            self.assertAlmostEqual(l, length)
            self.assertGreaterEqual(l, 0)
            self.assertGreater(r, l)
            self.assertLessEqual(r, ts.get_sequence_length())
            length += r - l
            self.verify_tree(st1)
            num_trees += 1
        self.assertRaises(StopIteration, next, iter1)
        self.assertRaises(StopIteration, next, iter2)
        self.assertEqual(ts.get_num_trees(), num_trees)
        self.assertEqual(breakpoints, list(ts.breakpoints()))
        self.assertAlmostEqual(length, ts.get_sequence_length())


class TestNumpySamples(unittest.TestCase):
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
        self.assertTrue(
            np.array_equal(ts.samples(), np.arange(ts.num_samples, dtype=np.int32))
        )
        total = 0
        for pop in range(d):
            subsample = ts.samples(pop)
            total += subsample.shape[0]
            self.assertTrue(np.array_equal(subsample, ts.samples(population=pop)))
            self.assertEqual(
                list(subsample),
                [
                    node.id
                    for node in ts.nodes()
                    if node.population == pop and node.is_sample()
                ],
            )
        self.assertEqual(total, ts.num_samples)

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
            self.assertEqual(total, np.sum(G[:, samples]))

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
            self.assertEqual(total, other_total)

    def test_pairwise_diversity(self):
        num_demes = 6
        ts = self.get_tree_sequence(num_demes)
        pi1 = ts.pairwise_diversity(ts.samples())
        pi2 = ts.pairwise_diversity()
        self.assertEqual(pi1, pi2)
        for d in range(num_demes):
            samples = ts.samples(population=d)
            pi1 = ts.pairwise_diversity(samples)
            pi2 = ts.pairwise_diversity(list(samples))
            self.assertEqual(pi1, pi2)

    def test_simplify(self):
        num_demes = 3
        ts = self.get_tree_sequence(num_demes)
        sts = ts.simplify(samples=ts.samples())
        self.assertEqual(ts.num_samples, sts.num_samples)
        for d in range(num_demes):
            samples = ts.samples(population=d)
            sts = ts.simplify(samples=samples)
            self.assertEqual(sts.num_samples, samples.shape[0])


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
        haplotypes = ts.genotype_matrix(impute_missing_data=True).T
        pi1 = ts.get_pairwise_diversity()
        pi2 = simple_get_pairwise_diversity(haplotypes)
        self.assertAlmostEqual(pi1, pi2)
        self.assertGreaterEqual(pi1, 0.0)
        self.assertFalse(math.isnan(pi1))
        # Check for a subsample.
        num_samples = ts.get_sample_size() // 2 + 1
        samples = list(ts.samples())[:num_samples]
        pi1 = ts.get_pairwise_diversity(samples)
        pi2 = simple_get_pairwise_diversity([haplotypes[j] for j in range(num_samples)])
        self.assertAlmostEqual(pi1, pi2)
        self.assertGreaterEqual(pi1, 0.0)
        self.assertFalse(math.isnan(pi1))

    def test_pairwise_diversity(self):
        for ts in get_example_tree_sequences():
            self.verify_pairwise_diversity(ts)

    def verify_edge_diffs(self, ts):
        pts = tests.PythonTreeSequence(ts)
        d1 = list(ts.edge_diffs())
        d2 = list(pts.edge_diffs())
        self.assertEqual(d1, d2)

        # check that we have the correct set of children at all nodes.
        children = collections.defaultdict(set)
        trees = iter(ts.trees())
        tree = next(trees)
        edge_ids = []
        last_right = 0
        for (left, right), edges_out, edges_in in ts.edge_diffs():
            self.assertEqual(left, last_right)
            last_right = right
            for edge in edges_out:
                self.assertEqual(edge, ts.edge(edge.id))
                children[edge.parent].remove(edge.child)
            for edge in edges_in:
                edge_ids.append(edge.id)
                self.assertEqual(edge, ts.edge(edge.id))
                children[edge.parent].add(edge.child)
            while tree.interval[1] <= left:
                tree = next(trees)
            self.assertTrue(left >= tree.interval[0])
            self.assertTrue(right <= tree.interval[1])
            for u in tree.nodes():
                if tree.is_internal(u):
                    self.assertIn(u, children)
                    self.assertEqual(children[u], set(tree.children(u)))
        # check that we have seen all the edge ids
        self.assertTrue(np.array_equal(np.unique(edge_ids), np.arange(0, ts.num_edges)))

    def test_edge_diffs(self):
        for ts in get_example_tree_sequences():
            self.verify_edge_diffs(ts)

    def verify_edgesets(self, ts):
        """
        Verifies that the edgesets we return are equivalent to the original edges.
        """
        new_edges = []
        for edgeset in ts.edgesets():
            self.assertEqual(edgeset.children, sorted(edgeset.children))
            self.assertGreater(len(edgeset.children), 0)
            for child in edgeset.children:
                new_edges.append(
                    tskit.Edge(edgeset.left, edgeset.right, edgeset.parent, child)
                )
        # squash the edges.
        t = ts.dump_tables().nodes.time
        new_edges.sort(key=lambda e: (t[e.parent], e.parent, e.child, e.left))

        squashed = []
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
        self.assertEqual(len(squashed), len(edges))
        self.assertEqual(edges, squashed)

    def test_edge_ids(self):
        for ts in get_example_tree_sequences():
            for index, edge in enumerate(ts.edges()):
                self.assertEqual(edge.id, index)

    def test_edge_span_property(self):
        for ts in get_example_tree_sequences():
            for edge in ts.edges():
                self.assertEqual(edge.span, edge.right - edge.left)

    def test_edgesets(self):
        for ts in get_example_tree_sequences():
            self.verify_edgesets(ts)

    def test_breakpoints(self):
        for ts in get_example_tree_sequences():
            breakpoints = ts.breakpoints(as_array=True)
            self.assertEqual(breakpoints.shape, (ts.num_trees + 1,))
            other = np.fromiter(iter([0] + [t.interval[1] for t in ts.trees()]), float)
            self.assertTrue(np.array_equal(other, breakpoints))
            # in case downstream code has
            for j, x in enumerate(ts.breakpoints()):
                self.assertEqual(breakpoints[j], x)
                self.assertIsInstance(x, float)
            self.assertEqual(j, ts.num_trees)

    def verify_coalescence_records(self, ts):
        """
        Checks that the coalescence records we output are correct.
        """
        edgesets = list(ts.edgesets())
        records = list(ts.records())
        self.assertEqual(len(edgesets), len(records))
        for edgeset, record in zip(edgesets, records):
            self.assertEqual(edgeset.left, record.left)
            self.assertEqual(edgeset.right, record.right)
            self.assertEqual(edgeset.parent, record.node)
            self.assertEqual(edgeset.children, record.children)
            parent = ts.node(edgeset.parent)
            self.assertEqual(parent.time, record.time)
            self.assertEqual(parent.population, record.population)

    def test_coalescence_records(self):
        for ts in get_example_tree_sequences():
            self.verify_coalescence_records(ts)

    def test_compute_mutation_parent(self):
        for ts in get_example_tree_sequences():
            tables = ts.dump_tables()
            before = tables.mutations.parent[:]
            tables.compute_mutation_parents()
            parent = ts.tables.mutations.parent
            self.assertTrue(np.array_equal(parent, before))

    def verify_tracked_samples(self, ts):
        # Should be empty list by default.
        for tree in ts.trees():
            self.assertEqual(tree.get_num_tracked_samples(), 0)
            for u in tree.nodes():
                self.assertEqual(tree.get_num_tracked_samples(u), 0)
        samples = list(ts.samples())
        tracked_samples = samples[:2]
        for tree in ts.trees(tracked_samples):
            if len(tree.parent_dict) == 0:
                # This is a crude way of checking if we have multiple roots.
                # We'll need to fix this code up properly when we support multiple
                # roots and remove this check
                break
            nu = [0 for j in range(ts.get_num_nodes())]
            self.assertEqual(tree.get_num_tracked_samples(), len(tracked_samples))
            for j in tracked_samples:
                u = j
                while u != tskit.NULL:
                    nu[u] += 1
                    u = tree.get_parent(u)
            for u, count in enumerate(nu):
                self.assertEqual(tree.get_num_tracked_samples(u), count)

    def test_tracked_samples(self):
        for ts in get_example_tree_sequences():
            self.verify_tracked_samples(ts)

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
                    self.assertEqual(
                        t_new.num_tracked_samples(u), t_old.get_num_tracked_leaves(u)
                    )
            trees_new = ts.trees()
            trees_old = ts.trees()
            for t_new, t_old in zip(trees_new, trees_old):
                for u in t_new.nodes():
                    self.assertEqual(t_new.num_samples(u), t_old.get_num_leaves(u))
                    self.assertEqual(list(t_new.samples(u)), list(t_old.get_leaves(u)))
            for on in [True, False]:
                trees_new = ts.trees(sample_lists=on)
                trees_old = ts.trees(leaf_lists=on)
                for t_new, t_old in zip(trees_new, trees_old):
                    for u in t_new.nodes():
                        self.assertEqual(t_new.num_samples(u), t_old.get_num_leaves(u))
                        self.assertEqual(
                            list(t_new.samples(u)), list(t_old.get_leaves(u))
                        )

    def verify_samples(self, ts):
        # We should get the same list of samples if we use the low-level
        # sample lists or a simple traversal.
        samples1 = []
        for t in ts.trees(sample_lists=False):
            samples1.append(list(t.samples()))
        samples2 = []
        for t in ts.trees(sample_lists=True):
            samples2.append(list(t.samples()))
        self.assertEqual(samples1, samples2)

    def test_samples(self):
        for ts in get_example_tree_sequences():
            self.verify_samples(ts)
            pops = {node.population for node in ts.nodes()}
            for pop in pops:
                subsample = ts.samples(pop)
                self.assertTrue(np.array_equal(subsample, ts.samples(population=pop)))
                self.assertTrue(
                    np.array_equal(subsample, ts.samples(population_id=pop))
                )
                self.assertEqual(
                    list(subsample),
                    [
                        node.id
                        for node in ts.nodes()
                        if node.population == pop and node.is_sample()
                    ],
                )
            self.assertRaises(ValueError, ts.samples, population=0, population_id=0)

    def test_first_last(self):
        for ts in get_example_tree_sequences():
            t1 = ts.first()
            t2 = next(ts.trees())
            self.assertFalse(t1 is t2)
            self.assertEqual(t1.parent_dict, t2.parent_dict)
            self.assertEqual(t1.index, 0)

            t1 = ts.last()
            t2 = next(reversed(ts.trees()))
            self.assertFalse(t1 is t2)
            self.assertEqual(t1.parent_dict, t2.parent_dict)
            self.assertEqual(t1.index, ts.num_trees - 1)

    def test_trees_interface(self):
        ts = list(get_example_tree_sequences())[0]
        for t in ts.trees():
            self.assertEqual(t.get_num_samples(0), 1)
            self.assertEqual(t.get_num_tracked_samples(0), 0)
            self.assertEqual(list(t.samples(0)), [0])
            self.assertIs(t.tree_sequence, ts)

        for t in ts.trees(tracked_samples=[0]):
            self.assertEqual(t.get_num_samples(0), 1)
            self.assertEqual(t.get_num_tracked_samples(0), 1)
            self.assertEqual(list(t.samples(0)), [0])

        for t in ts.trees(sample_lists=True):
            self.assertEqual(t.get_num_samples(0), 1)
            self.assertEqual(t.get_num_tracked_samples(0), 0)
            self.assertEqual(list(t.samples(0)), [0])

    def test_get_pairwise_diversity(self):
        for ts in get_example_tree_sequences():
            self.assertRaises(ValueError, ts.get_pairwise_diversity, [])
            samples = list(ts.samples())
            self.assertEqual(
                ts.get_pairwise_diversity(), ts.get_pairwise_diversity(samples)
            )
            self.assertEqual(
                ts.get_pairwise_diversity(samples[:2]),
                ts.get_pairwise_diversity(list(reversed(samples[:2]))),
            )

    def test_populations(self):
        more_than_zero = False
        for ts in get_example_tree_sequences():
            N = ts.num_populations
            if N > 0:
                more_than_zero = True
            pops = list(ts.populations())
            self.assertEqual(len(pops), N)
            for j in range(N):
                self.assertEqual(pops[j], ts.population(j))
                self.assertEqual(pops[j].id, j)
                self.assertTrue(isinstance(pops[j].metadata, bytes))
        self.assertTrue(more_than_zero)

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
            self.assertEqual(len(inds), N)
            for j in range(N):
                self.assertEqual(inds[j], ts.individual(j))
                self.assertEqual(inds[j].id, j)
                self.assertTrue(isinstance(inds[j].metadata, bytes))
                self.assertTrue(isinstance(inds[j].location, np.ndarray))
                self.assertTrue(isinstance(inds[j].nodes, np.ndarray))
                self.assertEqual(ind_node_map[j], list(inds[j].nodes))

        self.assertTrue(more_than_zero)
        self.assertTrue(mapped_to_nodes)

    def test_get_population(self):
        # Deprecated interface for ts.node(id).population
        for ts in get_example_tree_sequences():
            N = ts.get_num_nodes()
            self.assertRaises(ValueError, ts.get_population, -1)
            self.assertRaises(ValueError, ts.get_population, N)
            self.assertRaises(ValueError, ts.get_population, N + 1)
            for node in [0, N - 1]:
                self.assertEqual(ts.get_population(node), ts.node(node).population)

    def test_get_time(self):
        # Deprecated interface for ts.node(id).time
        for ts in get_example_tree_sequences():
            N = ts.get_num_nodes()
            self.assertRaises(ValueError, ts.get_time, -1)
            self.assertRaises(ValueError, ts.get_time, N)
            self.assertRaises(ValueError, ts.get_time, N + 1)
            for u in range(N):
                self.assertEqual(ts.get_time(u), ts.node(u).time)

    def test_max_root_time(self):
        for ts in get_example_tree_sequences():
            oldest = max(
                max(tree.time(root) for root in tree.roots) for tree in ts.trees()
            )
            self.assertEqual(oldest, ts.max_root_time)

    def test_max_root_time_corner_cases(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=2)
        tables.nodes.add_row(flags=0, time=3)
        self.assertEqual(tables.tree_sequence().max_root_time, 2)
        tables.edges.add_row(0, 1, 1, 0)
        self.assertEqual(tables.tree_sequence().max_root_time, 2)
        tables.edges.add_row(0, 1, 3, 1)
        self.assertEqual(tables.tree_sequence().max_root_time, 3)

    def verify_simplify_provenance(self, ts):
        new_ts = ts.simplify()
        self.assertEqual(new_ts.num_provenances, ts.num_provenances + 1)
        old = list(ts.provenances())
        new = list(new_ts.provenances())
        self.assertEqual(old, new[:-1])
        # TODO call verify_provenance on this.
        self.assertGreater(len(new[-1].timestamp), 0)
        self.assertGreater(len(new[-1].record), 0)

        new_ts = ts.simplify(record_provenance=False)
        self.assertEqual(new_ts.tables.provenances, ts.tables.provenances)

    def verify_simplify_topology(self, ts, sample):
        new_ts, node_map = ts.simplify(sample, map_nodes=True)
        if len(sample) == 0:
            self.assertEqual(new_ts.num_nodes, 0)
            self.assertEqual(new_ts.num_edges, 0)
            self.assertEqual(new_ts.num_sites, 0)
            self.assertEqual(new_ts.num_mutations, 0)
        elif len(sample) == 1:
            self.assertEqual(new_ts.num_nodes, 1)
            self.assertEqual(new_ts.num_edges, 0)
        # The output samples should be 0...n
        self.assertEqual(new_ts.num_samples, len(sample))
        self.assertEqual(list(range(len(sample))), list(new_ts.samples()))
        for j in range(new_ts.num_samples):
            self.assertEqual(node_map[sample[j]], j)
        for u in range(ts.num_nodes):
            old_node = ts.node(u)
            if node_map[u] != tskit.NULL:
                new_node = new_ts.node(node_map[u])
                self.assertEqual(old_node.time, new_node.time)
                self.assertEqual(old_node.population, new_node.population)
                self.assertEqual(old_node.metadata, new_node.metadata)
        for u in sample:
            old_node = ts.node(u)
            new_node = new_ts.node(node_map[u])
            self.assertEqual(old_node.flags, new_node.flags)
            self.assertEqual(old_node.time, new_node.time)
            self.assertEqual(old_node.population, new_node.population)
            self.assertEqual(old_node.metadata, new_node.metadata)
        old_trees = ts.trees()
        old_tree = next(old_trees)
        self.assertGreaterEqual(ts.get_num_trees(), new_ts.get_num_trees())
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
                    self.assertEqual(mrca2, mrca1)
                else:
                    self.assertEqual(mrca2, node_map[mrca1])
                    self.assertEqual(old_tree.get_time(mrca1), new_tree.get_time(mrca2))
                    self.assertEqual(
                        old_tree.get_population(mrca1), new_tree.get_population(mrca2)
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
            self.assertEqual(s1.num_samples, len(sample))
            self.assertEqual(s2.num_samples, len(sample))
            self.assertTrue(all(node_map1 == node_map2))
            self.assertEqual(t1.individuals, t2.individuals)
            self.assertEqual(t1.nodes, t2.nodes)
            self.assertEqual(t1.edges, t2.edges)
            self.assertEqual(t1.migrations, t2.migrations)
            self.assertEqual(t1.sites, t2.sites)
            self.assertEqual(t1.mutations, t2.mutations)
            self.assertEqual(t1.populations, t2.populations)

    def verify_simplify_variants(self, ts, sample):
        subset = ts.simplify(sample)
        sample_map = {u: j for j, u in enumerate(ts.samples())}
        # Need to map IDs back to their sample indexes
        s = np.array([sample_map[u] for u in sample])
        # Build a map of genotypes by position
        full_genotypes = {}
        for variant in ts.variants(impute_missing_data=True):
            alleles = [variant.alleles[g] for g in variant.genotypes]
            full_genotypes[variant.position] = alleles
        for variant in subset.variants(impute_missing_data=True):
            if variant.position in full_genotypes:
                a1 = [full_genotypes[variant.position][u] for u in s]
                a2 = [variant.alleles[g] for g in variant.genotypes]
                self.assertEqual(a1, a2)

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
        self.assertGreater(num_mutations, 0)

    def test_simplify_bugs(self):
        prefix = os.path.join(os.path.dirname(__file__), "data", "simplify-bugs")
        j = 1
        while True:
            nodes_file = os.path.join(prefix, "{:02d}-nodes.txt".format(j))
            if not os.path.exists(nodes_file):
                break
            edges_file = os.path.join(prefix, "{:02d}-edges.txt".format(j))
            sites_file = os.path.join(prefix, "{:02d}-sites.txt".format(j))
            mutations_file = os.path.join(prefix, "{:02d}-mutations.txt".format(j))
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
        self.assertGreater(j, 1)

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
        self.assertGreater(ts.num_migrations, 0)
        # We don't support simplify with migrations, so should fail.
        with self.assertRaises(_tskit.LibraryError):
            ts.simplify()

    def test_deprecated_apis(self):
        ts = msprime.simulate(10, random_seed=1)
        self.assertEqual(ts.get_ll_tree_sequence(), ts.ll_tree_sequence)
        self.assertEqual(ts.get_sample_size(), ts.sample_size)
        self.assertEqual(ts.get_sample_size(), ts.num_samples)
        self.assertEqual(ts.get_sequence_length(), ts.sequence_length)
        self.assertEqual(ts.get_num_trees(), ts.num_trees)
        self.assertEqual(ts.get_num_mutations(), ts.num_mutations)
        self.assertEqual(ts.get_num_nodes(), ts.num_nodes)
        self.assertEqual(ts.get_pairwise_diversity(), ts.pairwise_diversity())
        samples = ts.samples()
        self.assertEqual(
            ts.get_pairwise_diversity(samples), ts.pairwise_diversity(samples)
        )
        self.assertTrue(np.array_equal(ts.get_samples(), ts.samples()))

    def test_sites(self):
        some_sites = False
        for ts in get_example_tree_sequences():
            tables = ts.dump_tables()
            sites = tables.sites
            mutations = tables.mutations
            self.assertEqual(ts.num_sites, len(sites))
            self.assertEqual(ts.num_mutations, len(mutations))
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
                self.assertEqual(s2, site)
                self.assertEqual(site.position, sites.position[index])
                self.assertGreater(site.position, previous_pos)
                previous_pos = site.position
                self.assertEqual(ancestral_state[index], site.ancestral_state)
                self.assertEqual(site.id, index)
                for mutation in site.mutations:
                    m2 = ts.mutation(mutation.id)
                    self.assertEqual(m2, mutation)
                    self.assertEqual(mutation.site, site.id)
                    self.assertEqual(mutation.site, mutations.site[mutation_index])
                    self.assertEqual(mutation.node, mutations.node[mutation_index])
                    self.assertEqual(mutation.parent, mutations.parent[mutation_index])
                    self.assertEqual(mutation.id, mutation_index)
                    self.assertEqual(
                        derived_state[mutation_index], mutation.derived_state
                    )
                    mutation_index += 1
                some_sites = True
            total_sites = 0
            for tree in ts.trees():
                self.assertEqual(len(list(tree.sites())), tree.num_sites)
                total_sites += tree.num_sites
            self.assertEqual(ts.num_sites, total_sites)
            self.assertEqual(mutation_index, len(mutations))
        self.assertTrue(some_sites)

    def verify_mutations(self, ts):
        other_mutations = []
        for site in ts.sites():
            for mutation in site.mutations:
                other_mutations.append(mutation)
        mutations = list(ts.mutations())
        self.assertEqual(ts.num_mutations, len(other_mutations))
        self.assertEqual(ts.num_mutations, len(mutations))
        for mut, other_mut in zip(mutations, other_mutations):
            # We cannot compare these directly as the mutations obtained
            # from the mutations iterator will have extra deprecated
            # attributes.
            self.assertEqual(mut.id, other_mut.id)
            self.assertEqual(mut.site, other_mut.site)
            self.assertEqual(mut.parent, other_mut.parent)
            self.assertEqual(mut.node, other_mut.node)
            self.assertEqual(mut.metadata, other_mut.metadata)
            # Check the deprecated attrs.
            self.assertEqual(mut.position, ts.site(mut.site).position)
            self.assertEqual(mut.index, mut.site)

    def test_sites_mutations(self):
        # Check that the mutations iterator returns the correct values.
        for ts in get_example_tree_sequences():
            self.verify_mutations(ts)

    def test_removed_methods(self):
        ts = next(get_example_tree_sequences())
        self.assertRaises(NotImplementedError, ts.get_num_records)
        self.assertRaises(NotImplementedError, ts.diffs)
        self.assertRaises(NotImplementedError, ts.newick_trees)

    def test_dump_pathlib(self):
        ts = msprime.simulate(5, random_seed=1)
        path = pathlib.Path(self.temp_dir) / "tmp.trees"
        self.assertTrue(path.exists)
        self.assertTrue(path.is_file)
        ts.dump(path)
        other_ts = tskit.load(path)
        self.assertEqual(ts.tables, other_ts.tables)

    def test_zlib_compression_warning(self):
        ts = msprime.simulate(5, random_seed=1)
        with warnings.catch_warnings(record=True) as w:
            ts.dump(self.temp_file, zlib_compression=True)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, RuntimeWarning))
        with warnings.catch_warnings(record=True) as w:
            ts.dump(self.temp_file, zlib_compression=False)
            self.assertEqual(len(w), 0)

    def test_tables_sequence_length_round_trip(self):
        for sequence_length in [0.1, 1, 10, 100]:
            ts = msprime.simulate(5, length=sequence_length, random_seed=1)
            self.assertEqual(ts.sequence_length, sequence_length)
            tables = ts.tables
            self.assertEqual(tables.sequence_length, sequence_length)
            new_ts = tables.tree_sequence()
            self.assertEqual(new_ts.sequence_length, sequence_length)

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
        self.assertGreater(ts.num_migrations, 0)
        migrations = list(ts.migrations())
        self.assertEqual(len(migrations), ts.num_migrations)
        for migration in migrations:
            self.assertIn(migration.source, [0, 1])
            self.assertIn(migration.dest, [0, 1])
            self.assertGreater(migration.time, 0)
            self.assertEqual(migration.left, 0)
            self.assertEqual(migration.right, 1)
            self.assertTrue(0 <= migration.node < ts.num_nodes)

    def test_len_trees(self):
        for ts in get_example_tree_sequences():
            tree_iter = ts.trees()
            self.assertEqual(len(tree_iter), ts.num_trees)

    def test_list(self):
        for ts in get_example_tree_sequences():
            tree_list = ts.aslist()
            self.assertEqual(len(tree_list), ts.num_trees)
            self.assertEqual(len(set(map(id, tree_list))), ts.num_trees)
            for index, tree in enumerate(tree_list):
                self.assertEqual(index, tree.index)
            for t1, t2 in zip(tree_list, ts.trees()):
                self.assertEqual(t1, t2)
                self.assertEqual(t1.parent_dict, t2.parent_dict)

    def test_reversed_trees(self):
        for ts in get_example_tree_sequences():
            index = ts.num_trees - 1
            tree_list = ts.aslist()
            for tree in reversed(ts.trees()):
                self.assertEqual(tree.index, index)
                t2 = tree_list[index]
                self.assertEqual(tree.interval, t2.interval)
                self.assertEqual(tree.parent_dict, t2.parent_dict)
                index -= 1

    def test_at_index(self):
        for ts in get_example_tree_sequences():
            tree_list = ts.aslist()
            for index in list(range(ts.num_trees)) + [-1]:
                t1 = tree_list[index]
                t2 = ts.at_index(index)
                self.assertEqual(t1, t2)
                self.assertEqual(t1.interval, t2.interval)
                self.assertEqual(t1.parent_dict, t2.parent_dict)

    def test_at(self):
        for ts in get_example_tree_sequences():
            tree_list = ts.aslist()
            for t1 in tree_list:
                left, right = t1.interval
                mid = left + (right - left) / 2
                for pos in [left, left + 1e-9, mid, right - 1e-9]:
                    t2 = ts.at(pos)
                    self.assertEqual(t1, t2)
                    self.assertEqual(t1.interval, t2.interval)
                    self.assertEqual(t1.parent_dict, t2.parent_dict)
                if right < ts.sequence_length:
                    t2 = ts.at(right)
                    t3 = tree_list[t1.index + 1]
                    self.assertEqual(t3, t2)
                    self.assertEqual(t3.interval, t2.interval)
                    self.assertEqual(t3.parent_dict, t2.parent_dict)

    def test_sequence_iteration(self):
        for ts in get_example_tree_sequences():
            for table_name, _ in ts.tables:
                sequence = getattr(ts, table_name)()
                length = getattr(ts, "num_" + table_name)
                # Test __iter__
                for i, n in enumerate(sequence):
                    self.assertEqual(i, n.id)
                self.assertEqual(n.id, length - 1 if length else 0)
                if table_name == "mutations":
                    # Mutations are not currently sequences, so have no len or idx access
                    self.assertRaises(TypeError, len, sequence)
                    if length != 0:
                        with self.assertRaises(TypeError):
                            sequence[0]
                else:
                    # Test __len__
                    self.assertEqual(len(sequence), length)
                    # Test __getitem__ on the last item in the sequence
                    if length != 0:
                        self.assertEqual(sequence[length - 1], n)  # +ive indexing
                        self.assertEqual(sequence[-1], n)  # -ive indexing
                    with self.assertRaises(IndexError):
                        sequence[length]
                    # Test reverse
                    for i, n in enumerate(reversed(sequence)):
                        self.assertEqual(i, length - 1 - n.id)
                    self.assertEqual(n.id, 0)


class TestPickle(HighLevelTestCase):
    """
    Test pickling of a TreeSequence.
    """

    def verify_round_trip(self, ts):
        pkl = pickle.dumps(ts)
        tsp = pickle.loads(pkl)
        self.assertEqual(ts.tables, tsp.tables)

    def test_simple(self):
        self.verify_round_trip(msprime.simulate(10, random_seed=2))

    def test_recombination(self):
        self.verify_round_trip(
            msprime.simulate(10, recombination_rate=1, random_seed=2)
        )

    def test_mutations(self):
        self.verify_round_trip(msprime.simulate(10, mutation_rate=1, random_seed=2))

    def test_migrations(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(10),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            record_migrations=True,
            random_seed=2,
        )
        self.verify_round_trip(ts)


class TestFileUuid(HighLevelTestCase):
    """
    Tests that the file UUID attribute is handled correctly.
    """

    def validate(self, ts):
        self.assertIsNone(ts.file_uuid)
        ts.dump(self.temp_file)
        other_ts = tskit.load(self.temp_file)
        self.assertIsNotNone(other_ts.file_uuid)
        self.assertTrue(len(other_ts.file_uuid), 36)
        uuid = other_ts.file_uuid
        other_ts = tskit.load(self.temp_file)
        self.assertEqual(other_ts.file_uuid, uuid)
        self.assertEqual(ts.tables, other_ts.tables)

        # Check that the UUID is well-formed.
        parsed = _uuid.UUID("{" + uuid + "}")
        self.assertEqual(str(parsed), uuid)

        # Save the same tree sequence to the file. We should get a different UUID.
        ts.dump(self.temp_file)
        other_ts = tskit.load(self.temp_file)
        self.assertIsNotNone(other_ts.file_uuid)
        self.assertNotEqual(other_ts.file_uuid, uuid)

        # Even saving a ts that has a UUID to another file changes the UUID
        old_uuid = other_ts.file_uuid
        other_ts.dump(self.temp_file)
        self.assertEqual(other_ts.file_uuid, old_uuid)
        other_ts = tskit.load(self.temp_file)
        self.assertIsNotNone(other_ts.file_uuid)
        self.assertNotEqual(other_ts.file_uuid, old_uuid)

        # Tables dumped from this ts are a deep copy, so they don't have
        # the file_uuid.
        tables = other_ts.dump_tables()
        self.assertIsNone(tables.file_uuid)

        # For now, ts.tables also returns a deep copy. This will hopefully
        # change in the future thoug.
        self.assertIsNone(ts.tables.file_uuid)

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
        self.assertEqual(len(output_nodes) - 1, ts.num_nodes)
        self.assertEqual(
            list(output_nodes[0].split()),
            ["id", "is_sample", "time", "population", "individual", "metadata"],
        )
        for node, line in zip(ts.nodes(), output_nodes[1:]):
            splits = line.split("\t")
            self.assertEqual(str(node.id), splits[0])
            self.assertEqual(str(node.is_sample()), splits[1])
            self.assertEqual(convert(node.time), splits[2])
            self.assertEqual(str(node.population), splits[3])
            self.assertEqual(str(node.individual), splits[4])
            self.assertEqual(tests.base64_encode(node.metadata), splits[5])

    def verify_edges_format(self, ts, edges_file, precision):
        """
        Verifies that the edges we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_edges = edges_file.read().splitlines()
        self.assertEqual(len(output_edges) - 1, ts.num_edges)
        self.assertEqual(
            list(output_edges[0].split()), ["left", "right", "parent", "child"]
        )
        for edge, line in zip(ts.edges(), output_edges[1:]):
            splits = line.split("\t")
            self.assertEqual(convert(edge.left), splits[0])
            self.assertEqual(convert(edge.right), splits[1])
            self.assertEqual(str(edge.parent), splits[2])
            self.assertEqual(str(edge.child), splits[3])

    def verify_sites_format(self, ts, sites_file, precision):
        """
        Verifies that the sites we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_sites = sites_file.read().splitlines()
        self.assertEqual(len(output_sites) - 1, ts.num_sites)
        self.assertEqual(
            list(output_sites[0].split()), ["position", "ancestral_state", "metadata"]
        )
        for site, line in zip(ts.sites(), output_sites[1:]):
            splits = line.split("\t")
            self.assertEqual(convert(site.position), splits[0])
            self.assertEqual(site.ancestral_state, splits[1])
            self.assertEqual(tests.base64_encode(site.metadata), splits[2])

    def verify_mutations_format(self, ts, mutations_file, precision):
        """
        Verifies that the mutations we output have the correct form.
        """

        def convert(v):
            return "{:.{}f}".format(v, precision)

        output_mutations = mutations_file.read().splitlines()
        self.assertEqual(len(output_mutations) - 1, ts.num_mutations)
        self.assertEqual(
            list(output_mutations[0].split()),
            ["site", "node", "derived_state", "parent", "metadata"],
        )
        mutations = [mut for site in ts.sites() for mut in site.mutations]
        for mutation, line in zip(mutations, output_mutations[1:]):
            splits = line.split("\t")
            self.assertEqual(str(mutation.site), splits[0])
            self.assertEqual(str(mutation.node), splits[1])
            self.assertEqual(str(mutation.derived_state), splits[2])
            self.assertEqual(str(mutation.parent), splits[3])
            self.assertEqual(tests.base64_encode(mutation.metadata), splits[4])

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
        self.assertEqual(ts1.sample_size, ts2.sample_size)
        self.assertAlmostEqual(ts1.sequence_length, ts2.sequence_length)
        self.assertEqual(ts1.num_nodes, ts2.num_nodes)
        self.assertEqual(ts1.num_edges, ts2.num_edges)
        self.assertEqual(ts1.num_sites, ts2.num_sites)
        self.assertEqual(ts1.num_mutations, ts2.num_mutations)

        checked = 0
        for n1, n2 in zip(ts1.nodes(), ts2.nodes()):
            self.assertEqual(n1.population, n2.population)
            self.assertEqual(n1.metadata, n2.metadata)
            self.assertAlmostEqual(n1.time, n2.time)
            checked += 1
        self.assertEqual(checked, ts1.num_nodes)

        checked = 0
        for r1, r2 in zip(ts1.edges(), ts2.edges()):
            checked += 1
            self.assertAlmostEqual(r1.left, r2.left)
            self.assertAlmostEqual(r1.right, r2.right)
            self.assertEqual(r1.parent, r2.parent)
            self.assertEqual(r1.child, r2.child)
        self.assertEqual(ts1.num_edges, checked)

        checked = 0
        for s1, s2 in zip(ts1.sites(), ts2.sites()):
            checked += 1
            self.assertAlmostEqual(s1.position, s2.position)
            self.assertAlmostEqual(s1.ancestral_state, s2.ancestral_state)
            self.assertEqual(s1.metadata, s2.metadata)
            self.assertEqual(s1.mutations, s2.mutations)
        self.assertEqual(ts1.num_sites, checked)

        # Check the trees
        check = 0
        for t1, t2 in zip(ts1.trees(), ts2.trees()):
            self.assertEqual(list(t1.nodes()), list(t2.nodes()))
            check += 1
        self.assertEqual(check, ts1.get_num_trees())

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
        self.assertRaises(
            _tskit.LibraryError,
            tskit.load_text,
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
        self.assertEqual(ts.sequence_length, 100)
        self.assertEqual(ts.num_nodes, 0)
        self.assertEqual(ts.num_edges, 0)
        self.assertEqual(ts.num_sites, 0)
        self.assertEqual(ts.num_edges, 0)


class TestTree(HighLevelTestCase):
    """
    Some simple tests on the tree API.
    """

    def get_tree(self, sample_lists=False):
        ts = msprime.simulate(10, random_seed=1, mutation_rate=1, record_full_arg=True)
        return next(ts.trees(sample_lists=sample_lists))

    def verify_mutations(self, tree):
        self.assertGreater(tree.num_mutations, 0)
        other_mutations = []
        for site in tree.sites():
            for mutation in site.mutations:
                other_mutations.append(mutation)
        mutations = list(tree.mutations())
        self.assertEqual(tree.num_mutations, len(other_mutations))
        self.assertEqual(tree.num_mutations, len(mutations))
        for mut, other_mut in zip(mutations, other_mutations):
            # We cannot compare these directly as the mutations obtained
            # from the mutations iterator will have extra deprecated
            # attributes.
            self.assertEqual(mut.id, other_mut.id)
            self.assertEqual(mut.site, other_mut.site)
            self.assertEqual(mut.parent, other_mut.parent)
            self.assertEqual(mut.node, other_mut.node)
            self.assertEqual(mut.metadata, other_mut.metadata)
            # Check the deprecated attrs.
            self.assertEqual(mut.position, tree.tree_sequence.site(mut.site).position)
            self.assertEqual(mut.index, mut.site)

    def test_simple_mutations(self):
        tree = self.get_tree()
        self.verify_mutations(tree)

    def test_complex_mutations(self):
        ts = tsutil.insert_branch_mutations(msprime.simulate(10, random_seed=1))
        self.verify_mutations(ts.first())

    def test_str(self):
        t = self.get_tree()
        self.assertIsInstance(str(t), str)
        self.assertEqual(str(t), str(t.get_parent_dict()))

    def test_samples(self):
        for sample_lists in [True, False]:
            t = self.get_tree(sample_lists)
            n = t.get_sample_size()
            all_samples = list(t.samples(t.get_root()))
            self.assertEqual(sorted(all_samples), list(range(n)))
            for j in range(n):
                self.assertEqual(list(t.samples(j)), [j])

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
                self.assertEqual(l1, l2)
                self.assertEqual(t.get_num_samples(u), len(l1))

    def test_num_children(self):
        tree = self.get_tree()
        for u in tree.nodes():
            self.assertEqual(tree.num_children(u), len(tree.children(u)))

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
            self.assertEqual(newick1, newick2)

            # Make sure we get the same results for a leaf root.
            newick1 = tree.newick(root=0, precision=16)
            newick2 = py_tree.newick(root=0)
            self.assertEqual(newick1, newick2)

            # When we specify the node_labels we should get precisely the
            # same result as we are using Python code now.
            for precision in [0, 3, 19]:
                newick1 = tree.newick(precision=precision, node_labels={})
                newick2 = py_tree.newick(precision=precision, node_labels={})
                self.assertEqual(newick1, newick2)
        else:
            self.assertRaises(ValueError, tree.newick)
            for root in tree.roots:
                py_tree = tests.PythonTree.from_tree(tree)
                newick1 = tree.newick(precision=16, root=root)
                newick2 = py_tree.newick(root=root)
                self.assertEqual(newick1, newick2)

    def test_newick(self):
        for ts in get_example_tree_sequences():
            for tree in ts.trees():
                self.verify_newick(tree)

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
        self.assertSetEqual(set(tree.nodes()), set(g.nodes))

        self.assertSetEqual(
            set(tree.roots), {n for n in g.nodes if g.in_degree(n) == 0}
        )

        self.assertSetEqual(
            set(tree.leaves()), {n for n in g.nodes if g.out_degree(n) == 0}
        )

        # test if tree has no in-degrees > 1
        self.assertTrue(nx.is_branching(g))

    def verify_nx_algorithm_equivalence(self, tree, g):
        for root in tree.roots:
            self.assertTrue(nx.is_directed_acyclic_graph(g))

            # test descendants
            self.assertSetEqual(
                set(u for u in tree.nodes() if tree.is_descendant(u, root)),
                set(nx.descendants(g, root)) | {root},
            )

            # test MRCA
            if tree.num_nodes < 20:
                for u, v in itertools.combinations(tree.nodes(), 2):
                    mrca = nx.lowest_common_ancestor(g, u, v)
                    if mrca is None:
                        mrca = -1
                    self.assertEqual(tree.mrca(u, v), mrca)

            # test node traversal modes
            self.assertEqual(
                list(tree.nodes(root=root, order="breadthfirst")),
                [root] + [v for u, v in nx.bfs_edges(g, root)],
            )
            self.assertEqual(
                list(tree.nodes(root=root, order="preorder")),
                list(nx.dfs_preorder_nodes(g, root)),
            )

    def verify_nx_for_tutorial_algorithms(self, tree, g):
        # traversing upwards
        for u in tree.leaves():
            path = []
            v = u
            while v != tskit.NULL:
                path.append(v)
                v = tree.parent(v)

            self.assertSetEqual(set(path), {u} | nx.ancestors(g, u))
            self.assertEqual(
                path,
                [u] + [n1 for n1, n2, _ in nx.edge_dfs(g, u, orientation="reverse")],
            )

        # traversals with information
        def preorder_dist(tree, root):
            stack = [(root, 0)]
            while len(stack) > 0:
                u, distance = stack.pop()
                yield u, distance
                for v in tree.children(u):
                    stack.append((v, distance + 1))

        for root in tree.roots:
            self.assertDictEqual(
                {k: v for k, v in preorder_dist(tree, root)},
                nx.shortest_path_length(g, source=root),
            )

        for root in tree.roots:
            # new traversal: measuring time between root and MRCA
            for u, v in itertools.combinations(nx.descendants(g, root), 2):
                mrca = tree.mrca(u, v)
                tmrca = tree.time(mrca)
                self.assertAlmostEqual(
                    tree.time(root) - tmrca,
                    nx.shortest_path_length(
                        g, source=root, target=mrca, weight="branch_length"
                    ),
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
        self.assertEqual([2, 2, 1], [nearest_neighbor_of[u] for u in range(3)])

    def test_traversals(self):
        for ts in get_example_tree_sequences():
            tree = next(ts.trees())
            self.verify_traversals(tree)

            # To verify time-ordered traversal we can't use the method used for the
            # other traversals above, it checks for one-to-one correspondence.
            # As more than one ordering is valid for time, we do it separately here
            for root in tree.roots:
                time_ordered = tree.nodes(root, order="timeasc")
                t = tree.time(next(time_ordered))
                for u in time_ordered:
                    next_t = tree.time(u)
                    self.assertGreaterEqual(next_t, t)
                    t = next_t
                time_ordered = tree.nodes(root, order="timedesc")
                t = tree.time(next(time_ordered))
                for u in time_ordered:
                    next_t = tree.time(u)
                    self.assertLessEqual(next_t, t)
                    t = next_t

    def verify_traversals(self, tree):
        t1 = tree
        t2 = tests.PythonTree.from_tree(t1)
        self.assertEqual(list(t1.nodes()), list(t2.nodes()))
        orders = ["inorder", "postorder", "levelorder", "breadthfirst"]
        if tree.num_roots == 1:
            self.assertRaises(ValueError, list, t1.nodes(order="bad order"))
            self.assertEqual(list(t1.nodes()), list(t1.nodes(t1.get_root())))
            self.assertEqual(
                list(t1.nodes()), list(t1.nodes(t1.get_root(), "preorder"))
            )
            for u in t1.nodes():
                self.assertEqual(list(t1.nodes(u)), list(t2.nodes(u)))
            for test_order in orders:
                self.assertEqual(
                    sorted(list(t1.nodes())), sorted(list(t1.nodes(order=test_order)))
                )
                self.assertEqual(
                    list(t1.nodes(order=test_order)),
                    list(t1.nodes(t1.get_root(), order=test_order)),
                )
                self.assertEqual(
                    list(t1.nodes(order=test_order)),
                    list(t1.nodes(t1.get_root(), test_order)),
                )
                self.assertEqual(
                    list(t1.nodes(order=test_order)), list(t2.nodes(order=test_order))
                )
                for u in t1.nodes():
                    self.assertEqual(
                        list(t1.nodes(u, test_order)), list(t2.nodes(u, test_order))
                    )
        else:
            for test_order in orders:
                all_nodes = []
                for root in t1.roots:
                    self.assertEqual(
                        list(t1.nodes(root, order=test_order)),
                        list(t2.nodes(root, order=test_order)),
                    )
                    all_nodes.extend(t1.nodes(root, order=test_order))
                self.assertEqual(all_nodes, list(t1.nodes(order=test_order)))

    def test_total_branch_length(self):
        # Note: this definition works when we have no non-sample branches.
        t1 = self.get_tree()
        bl = 0
        root = t1.get_root()
        for node in t1.nodes():
            if node != root:
                bl += t1.get_branch_length(node)
        self.assertGreater(bl, 0)
        self.assertAlmostEqual(t1.get_total_branch_length(), bl)

    def test_branch_length_empty_tree(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        ts = tables.tree_sequence()
        self.assertEqual(ts.num_trees, 1)
        tree = ts.first()
        self.assertEqual(tree.branch_length(0), 0)
        self.assertEqual(tree.branch_length(1), 0)
        self.assertEqual(tree.total_branch_length, 0)

    def test_is_descendant(self):
        def is_descendant(tree, u, v):
            path = []
            while u != tskit.NULL:
                path.append(u)
                u = tree.parent(u)
            return v in path

        tree = self.get_tree()
        for u, v in itertools.product(range(tree.num_nodes), repeat=2):
            self.assertEqual(is_descendant(tree, u, v), tree.is_descendant(u, v))
        for bad_node in [-1, -2, tree.num_nodes, tree.num_nodes + 1]:
            self.assertRaises(ValueError, tree.is_descendant, 0, bad_node)
            self.assertRaises(ValueError, tree.is_descendant, bad_node, 0)
            self.assertRaises(ValueError, tree.is_descendant, bad_node, bad_node)

    def test_apis(self):
        # tree properties
        t1 = self.get_tree()
        self.assertEqual(t1.get_root(), t1.root)
        self.assertEqual(t1.get_index(), t1.index)
        self.assertEqual(t1.get_interval(), t1.interval)
        self.assertEqual(t1.get_sample_size(), t1.sample_size)
        self.assertEqual(t1.get_num_mutations(), t1.num_mutations)
        self.assertEqual(t1.get_parent_dict(), t1.parent_dict)
        self.assertEqual(t1.get_total_branch_length(), t1.total_branch_length)
        self.assertEqual(t1.span, t1.interval[1] - t1.interval[0])
        # node properties
        root = t1.get_root()
        for node in t1.nodes():
            if node != root:
                self.assertEqual(t1.get_time(node), t1.time(node))
                self.assertEqual(t1.get_parent(node), t1.parent(node))
                self.assertEqual(t1.get_children(node), t1.children(node))
                self.assertEqual(t1.get_population(node), t1.population(node))
                self.assertEqual(t1.get_num_samples(node), t1.num_samples(node))
                self.assertEqual(t1.get_branch_length(node), t1.branch_length(node))
                self.assertEqual(
                    t1.get_num_tracked_samples(node), t1.num_tracked_samples(node)
                )

        pairs = itertools.islice(itertools.combinations(t1.nodes(), 2), 50)
        for pair in pairs:
            self.assertEqual(t1.get_mrca(*pair), t1.mrca(*pair))
            self.assertEqual(t1.get_tmrca(*pair), t1.tmrca(*pair))

    def test_deprecated_apis(self):
        t1 = self.get_tree()
        self.assertEqual(t1.get_length(), t1.span)
        self.assertEqual(t1.length, t1.span)

    def test_seek_index(self):
        ts = msprime.simulate(10, recombination_rate=3, length=5, random_seed=42)
        N = ts.num_trees
        self.assertGreater(ts.num_trees, 3)
        tree = tskit.Tree(ts)
        for index in [0, N // 2, N - 1, 1]:
            fresh_tree = tskit.Tree(ts)
            self.assertEqual(fresh_tree.index, -1)
            fresh_tree.seek_index(index)
            tree.seek_index(index)
            self.assertEqual(fresh_tree.index, index)
            self.assertEqual(tree.index, index)

        tree = tskit.Tree(ts)
        for index in [-1, -2, -N + 2, -N + 1, -N]:
            fresh_tree = tskit.Tree(ts)
            self.assertEqual(fresh_tree.index, -1)
            fresh_tree.seek_index(index)
            tree.seek_index(index)
            self.assertEqual(fresh_tree.index, index + N)
            self.assertEqual(tree.index, index + N)
        self.assertRaises(IndexError, tree.seek_index, N)
        self.assertRaises(IndexError, tree.seek_index, N + 1)
        self.assertRaises(IndexError, tree.seek_index, -N - 1)
        self.assertRaises(IndexError, tree.seek_index, -N - 2)

    def test_first_last(self):
        ts = msprime.simulate(10, recombination_rate=3, length=2, random_seed=42)
        self.assertGreater(ts.num_trees, 3)
        tree = tskit.Tree(ts)
        tree.first()
        self.assertEqual(tree.index, 0)
        tree = tskit.Tree(ts)
        tree.last()
        self.assertEqual(tree.index, ts.num_trees - 1)
        tree = tskit.Tree(ts)
        for _ in range(3):
            tree.last()
            self.assertEqual(tree.index, ts.num_trees - 1)
            tree.first()
            self.assertEqual(tree.index, 0)

    def test_eq_different_tree_sequence(self):
        ts = msprime.simulate(4, recombination_rate=1, length=2, random_seed=42)
        copy = ts.tables.tree_sequence()
        for tree1, tree2 in zip(ts.aslist(), copy.aslist()):
            self.assertNotEqual(tree1, tree2)

    def test_next_prev(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        self.assertGreater(ts.num_trees, 5)
        for index, tree in enumerate(ts.aslist()):
            self.assertEqual(tree.index, index)
            j = index
            while tree.next():
                j += 1
                self.assertEqual(tree.index, j)
            self.assertEqual(tree.index, -1)
            self.assertEqual(j + 1, ts.num_trees)
        for index, tree in enumerate(ts.aslist()):
            self.assertEqual(tree.index, index)
            j = index
            while tree.prev():
                j -= 1
                self.assertEqual(tree.index, j)
            self.assertEqual(tree.index, -1)
            self.assertEqual(j, 0)
        tree.first()
        tree.prev()
        self.assertEqual(tree.index, -1)
        tree.last()
        tree.next()
        self.assertEqual(tree.index, -1)

    def test_seek(self):
        L = 10
        ts = msprime.simulate(10, recombination_rate=3, length=L, random_seed=42)
        self.assertGreater(ts.num_trees, 5)
        same_tree = tskit.Tree(ts)
        for tree in [same_tree, tskit.Tree(ts)]:
            for j in range(L):
                tree.seek(j)
                index = tree.index
                self.assertTrue(tree.interval[0] <= j < tree.interval[1])
                tree.seek(tree.interval[0])
                self.assertEqual(tree.index, index)
                if tree.interval[1] < L:
                    tree.seek(tree.interval[1])
                    self.assertEqual(tree.index, index + 1)
            for j in reversed(range(L)):
                tree.seek(j)
                self.assertTrue(tree.interval[0] <= j < tree.interval[1])
        for bad_position in [-1, L, L + 1, -L]:
            self.assertRaises(ValueError, tree.seek, bad_position)

    def verify_empty_tree(self, tree):
        ts = tree.tree_sequence
        self.assertEqual(tree.index, -1)
        self.assertEqual(tree.parent_dict, {})
        for u in range(ts.num_nodes):
            self.assertEqual(tree.parent(u), tskit.NULL)
            self.assertEqual(tree.left_child(u), tskit.NULL)
            self.assertEqual(tree.right_child(u), tskit.NULL)
            if not ts.node(u).is_sample():
                self.assertEqual(tree.left_sib(u), tskit.NULL)
                self.assertEqual(tree.right_sib(u), tskit.NULL)
        # Samples should have left-sib right-sibs set
        samples = ts.samples()
        self.assertEqual(tree.left_root, samples[0])
        for j in range(ts.num_samples):
            if j > 0:
                self.assertEqual(tree.left_sib(samples[j]), samples[j - 1])
            if j < ts.num_samples - 1:
                self.assertEqual(tree.right_sib(samples[j]), samples[j + 1])

    def test_empty_tree(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        self.assertGreater(ts.num_trees, 5)
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
        self.assertGreater(ts.num_trees, 5)
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
        self.assertIs(t1.tree_sequence, t2.tree_sequence)
        self.assertIs(t1.num_nodes, t2.num_nodes)
        self.assertEqual(
            [t1.parent(u) for u in range(t1.num_nodes)],
            [t2.parent(u) for u in range(t2.num_nodes)],
        )
        self.assertEqual(
            [t1.left_child(u) for u in range(t1.num_nodes)],
            [t2.left_child(u) for u in range(t2.num_nodes)],
        )
        self.assertEqual(
            [t1.right_child(u) for u in range(t1.num_nodes)],
            [t2.right_child(u) for u in range(t2.num_nodes)],
        )
        self.assertEqual(
            [t1.left_sib(u) for u in range(t1.num_nodes)],
            [t2.left_sib(u) for u in range(t2.num_nodes)],
        )
        self.assertEqual(
            [t1.right_sib(u) for u in range(t1.num_nodes)],
            [t2.right_sib(u) for u in range(t2.num_nodes)],
        )
        self.assertEqual(list(t1.sites()), list(t2.sites()))

    def test_copy_seek(self):
        ts = msprime.simulate(10, recombination_rate=3, length=3, random_seed=42)
        self.assertGreater(ts.num_trees, 5)
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
            self.assertEqual(tree.next(), copy.next())
        tree.last()
        copy.last()
        while tree.index != -1:
            self.verify_trees_identical(tree, copy)
            self.assertEqual(tree.prev(), copy.prev())
        # Seek to middle and two independent trees.
        tree.seek_index(ts.num_trees // 2)
        left_copy = tree.copy()
        right_copy = tree.copy()
        self.verify_trees_identical(tree, left_copy)
        self.verify_trees_identical(tree, right_copy)
        left_copy.prev()
        self.assertEqual(left_copy.index, tree.index - 1)
        right_copy.next()
        self.assertEqual(right_copy.index, tree.index + 1)

    def test_copy_tracked_samples(self):
        ts = msprime.simulate(10, recombination_rate=2, length=3, random_seed=42)
        tree = tskit.Tree(ts, tracked_samples=[0, 1])
        while tree.next():
            copy = tree.copy()
            for j in range(ts.num_nodes):
                self.assertEqual(
                    tree.num_tracked_samples(j), copy.num_tracked_samples(j)
                )
        copy = tree.copy()
        while tree.next():
            copy.next()
            for j in range(ts.num_nodes):
                self.assertEqual(
                    tree.num_tracked_samples(j), copy.num_tracked_samples(j)
                )

    def test_copy_multiple_roots(self):
        ts = msprime.simulate(20, recombination_rate=2, length=3, random_seed=42)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        for root_threshold in [1, 2, 100]:
            tree = tskit.Tree(ts, root_threshold=root_threshold)
            copy = tree.copy()
            self.assertEqual(copy.roots, tree.roots)
            self.assertEqual(copy.root_threshold, root_threshold)
            while tree.next():
                copy = tree.copy()
                self.assertEqual(copy.roots, tree.roots)
                self.assertEqual(copy.root_threshold, root_threshold)
            copy = tree.copy()
            self.assertEqual(copy.roots, tree.roots)
            self.assertEqual(copy.root_threshold, root_threshold)

    def test_map_mutations(self):
        ts = msprime.simulate(5, random_seed=42)
        tree = ts.first()
        genotypes = np.zeros(5, dtype=np.int8)
        alleles = [str(j) for j in range(64)]
        ancestral_state, transitions = tree.map_mutations(genotypes, alleles)
        self.assertEqual(ancestral_state, "0")
        self.assertEqual(len(transitions), 0)
        for j in range(1, 64):
            genotypes[0] = j
            ancestral_state, transitions = tree.map_mutations(genotypes, alleles)
            self.assertEqual(ancestral_state, "0")
            self.assertEqual(len(transitions), 1)
        for j in range(64, 67):
            genotypes[0] = j
            with self.assertRaises(ValueError):
                tree.map_mutations(genotypes, alleles)
        tree.map_mutations([0] * 5, alleles)
        tree.map_mutations(np.zeros(5, dtype=int), alleles)

    def test_sample_count_deprecated(self):
        ts = msprime.simulate(5, random_seed=42)
        with warnings.catch_warnings(record=True) as w:
            ts.trees(sample_counts=True)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, RuntimeWarning))

        with warnings.catch_warnings(record=True) as w:
            tskit.Tree(ts, sample_counts=False)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, RuntimeWarning))


class TestNodeOrdering(HighLevelTestCase):
    """
    Verify that we can use any node ordering for internal nodes
    and get the same topologies.
    """

    num_random_permutations = 10

    def verify_tree_sequences_equal(self, ts1, ts2, approx=False):
        self.assertEqual(ts1.get_num_trees(), ts2.get_num_trees())
        self.assertEqual(ts1.get_sample_size(), ts2.get_sample_size())
        self.assertEqual(ts1.get_num_nodes(), ts2.get_num_nodes())
        j = 0
        for r1, r2 in zip(ts1.edges(), ts2.edges()):
            self.assertEqual(r1.parent, r2.parent)
            self.assertEqual(r1.child, r2.child)
            if approx:
                self.assertAlmostEqual(r1.left, r2.left)
                self.assertAlmostEqual(r1.right, r2.right)
            else:
                self.assertEqual(r1.left, r2.left)
                self.assertEqual(r1.right, r2.right)
            j += 1
        self.assertEqual(ts1.num_edges, j)
        j = 0
        for n1, n2 in zip(ts1.nodes(), ts2.nodes()):
            self.assertEqual(n1.metadata, n2.metadata)
            self.assertEqual(n1.population, n2.population)
            if approx:
                self.assertAlmostEqual(n1.time, n2.time)
            else:
                self.assertEqual(n1.time, n2.time)
            j += 1
        self.assertEqual(ts1.num_nodes, j)

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

        self.assertEqual(ts.get_num_trees(), other_ts.get_num_trees())
        self.assertEqual(ts.get_sample_size(), other_ts.get_sample_size())
        self.assertEqual(ts.get_num_nodes(), other_ts.get_num_nodes())
        j = 0
        for t1, t2 in zip(ts.trees(), other_ts.trees()):
            # Verify the topologies are identical. We do this by traversing
            # upwards to the root for every sample and checking if we map to
            # the correct node and time.
            for u in range(n):
                v_orig = u
                v_map = u
                while v_orig != tskit.NULL:
                    self.assertEqual(node_map[v_orig], v_map)
                    self.assertEqual(t1.get_time(v_orig), t2.get_time(v_map))
                    v_orig = t1.get_parent(v_orig)
                    v_map = t2.get_parent(v_map)
                self.assertEqual(v_orig, tskit.NULL)
                self.assertEqual(v_map, tskit.NULL)
            j += 1
        self.assertEqual(j, ts.get_num_trees())
        # Verify we can dump this new tree sequence OK.
        other_ts.dump(self.temp_file)
        ts3 = tskit.load(self.temp_file)
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
        self.assertTrue(found)
        for _ in range(self.num_random_permutations):
            self.verify_random_permutation(ts)


class SimpleContainersMixin(object):
    """
    Tests for the SimpleContainer classes.
    """

    def test_equality(self):
        c1, c2 = self.get_instances(2)
        self.assertTrue(c1 == c1)
        self.assertFalse(c1 == c2)
        self.assertFalse(c1 != c1)
        self.assertTrue(c1 != c2)
        (c3,) = self.get_instances(1)
        self.assertTrue(c1 == c3)
        self.assertFalse(c1 != c3)

    def test_repr(self):
        (c,) = self.get_instances(1)
        self.assertGreater(len(repr(c)), 0)


class TestIndividualContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Individual(id_=j, flags=j, location=[j], nodes=[j], metadata=b"x" * j)
            for j in range(n)
        ]


class TestNodeContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Node(
                id_=j, flags=j, time=j, population=j, individual=j, metadata=b"x" * j
            )
            for j in range(n)
        ]


class TestEdgeContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [tskit.Edge(left=j, right=j, parent=j, child=j, id_=j) for j in range(n)]


class TestSiteContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Site(
                id_=j,
                position=j,
                ancestral_state="A" * j,
                mutations=TestMutationContainer().get_instances(j),
                metadata=b"x" * j,
            )
            for j in range(n)
        ]


class TestMutationContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Mutation(
                id_=j,
                site=j,
                node=j,
                derived_state="A" * j,
                parent=j,
                metadata=b"x" * j,
            )
            for j in range(n)
        ]


class TestMigrationContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Migration(left=j, right=j, node=j, source=j, dest=j, time=j)
            for j in range(n)
        ]


class TestPopulationContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [tskit.Population(id_=j, metadata="x" * j) for j in range(n)]


class TestProvenanceContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Provenance(id_=j, timestamp="x" * j, record="y" * j) for j in range(n)
        ]


class TestEdgesetContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [tskit.Edgeset(left=j, right=j, parent=j, children=j) for j in range(n)]


class TestVariantContainer(unittest.TestCase, SimpleContainersMixin):
    def get_instances(self, n):
        return [
            tskit.Variant(
                site=TestSiteContainer().get_instances(1)[0],
                alleles=["A" * j, "T"],
                genotypes=np.zeros(j, dtype=np.int8),
            )
            for j in range(n)
        ]
