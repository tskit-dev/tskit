# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (C) 2016 University of Oxford
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
Test cases for stats calculations in tskit.
"""
import io
import unittest

import msprime
import numpy as np

import _tskit
import tests.test_wright_fisher as wf
import tests.tsutil as tsutil
import tskit


def get_r2_matrix(ts):
    """
    Returns the matrix for the specified tree sequence. This is computed
    via a straightforward Python algorithm.
    """
    n = ts.get_sample_size()
    m = ts.get_num_mutations()
    A = np.zeros((m, m), dtype=float)
    for t1 in ts.trees():
        for sA in t1.sites():
            assert len(sA.mutations) == 1
            mA = sA.mutations[0]
            A[sA.id, sA.id] = 1
            fA = t1.get_num_samples(mA.node) / n
            samples = list(t1.samples(mA.node))
            for t2 in ts.trees(tracked_samples=samples):
                for sB in t2.sites():
                    assert len(sB.mutations) == 1
                    mB = sB.mutations[0]
                    if sB.position > sA.position:
                        fB = t2.get_num_samples(mB.node) / n
                        fAB = t2.get_num_tracked_samples(mB.node) / n
                        D = fAB - fA * fB
                        r2 = D * D / (fA * fB * (1 - fA) * (1 - fB))
                        A[sA.id, sB.id] = r2
                        A[sB.id, sA.id] = r2
    return A


class TestLdCalculator(unittest.TestCase):
    """
    Tests for the LdCalculator class.
    """

    num_test_sites = 50

    def verify_matrix(self, ts):
        m = ts.get_num_sites()
        ldc = tskit.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        self.assertEqual(A.shape, (m, m))
        B = get_r2_matrix(ts)
        self.assertTrue(np.allclose(A, B))

        # Now look at each row in turn, and verify it's the same
        # when we use get_r2 directly.
        for j in range(m):
            a = ldc.get_r2_array(j, direction=tskit.FORWARD)
            b = A[j, j + 1 :]
            self.assertEqual(a.shape[0], m - j - 1)
            self.assertEqual(b.shape[0], m - j - 1)
            self.assertTrue(np.allclose(a, b))
            a = ldc.get_r2_array(j, direction=tskit.REVERSE)
            b = A[j, :j]
            self.assertEqual(a.shape[0], j)
            self.assertEqual(b.shape[0], j)
            self.assertTrue(np.allclose(a[::-1], b))

        # Now check every cell in the matrix in turn.
        for j in range(m):
            for k in range(m):
                self.assertAlmostEqual(ldc.get_r2(j, k), A[j, k])

    def verify_max_distance(self, ts):
        """
        Verifies that the max_distance parameter works as expected.
        """
        mutations = list(ts.mutations())
        ldc = tskit.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        j = len(mutations) // 2
        for k in range(j):
            x = mutations[j + k].position - mutations[j].position
            a = ldc.get_r2_array(j, max_distance=x)
            self.assertEqual(a.shape[0], k)
            self.assertTrue(np.allclose(A[j, j + 1 : j + 1 + k], a))
            x = mutations[j].position - mutations[j - k].position
            a = ldc.get_r2_array(j, max_distance=x, direction=tskit.REVERSE)
            self.assertEqual(a.shape[0], k)
            self.assertTrue(np.allclose(A[j, j - k : j], a[::-1]))
        L = ts.get_sequence_length()
        m = len(mutations)
        a = ldc.get_r2_array(0, max_distance=L)
        self.assertEqual(a.shape[0], m - 1)
        self.assertTrue(np.allclose(A[0, 1:], a))
        a = ldc.get_r2_array(m - 1, max_distance=L, direction=tskit.REVERSE)
        self.assertEqual(a.shape[0], m - 1)
        self.assertTrue(np.allclose(A[m - 1, :-1], a[::-1]))

    def verify_max_mutations(self, ts):
        """
        Verifies that the max mutations parameter works as expected.
        """
        mutations = list(ts.mutations())
        ldc = tskit.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        j = len(mutations) // 2
        for k in range(j):
            a = ldc.get_r2_array(j, max_mutations=k)
            self.assertEqual(a.shape[0], k)
            self.assertTrue(np.allclose(A[j, j + 1 : j + 1 + k], a))
            a = ldc.get_r2_array(j, max_mutations=k, direction=tskit.REVERSE)
            self.assertEqual(a.shape[0], k)
            self.assertTrue(np.allclose(A[j, j - k : j], a[::-1]))

    def test_single_tree_simulated_mutations(self):
        ts = msprime.simulate(20, mutation_rate=10, random_seed=15)
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        self.verify_matrix(ts)
        self.verify_max_distance(ts)

    def test_deprecated_aliases(self):
        ts = msprime.simulate(20, mutation_rate=10, random_seed=15)
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        ldc = tskit.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        B = ldc.r2_matrix()
        self.assertTrue(np.array_equal(A, B))
        a = ldc.get_r2_array(0)
        b = ldc.r2_array(0)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(ldc.get_r2(0, 1), ldc.r2(0, 1))

    def test_single_tree_regular_mutations(self):
        ts = msprime.simulate(self.num_test_sites, length=self.num_test_sites)
        ts = tsutil.insert_branch_mutations(ts)
        # We don't support back mutations, so this should fail.
        self.assertRaises(_tskit.LibraryError, self.verify_matrix, ts)
        self.assertRaises(_tskit.LibraryError, self.verify_max_distance, ts)

    def test_tree_sequence_regular_mutations(self):
        ts = msprime.simulate(
            self.num_test_sites, recombination_rate=1, length=self.num_test_sites
        )
        self.assertGreater(ts.get_num_trees(), 10)
        t = ts.dump_tables()
        t.sites.reset()
        t.mutations.reset()
        for j in range(self.num_test_sites):
            site_id = len(t.sites)
            t.sites.add_row(position=j, ancestral_state="0")
            t.mutations.add_row(site=site_id, derived_state="1", node=j)
        ts = t.tree_sequence()
        self.verify_matrix(ts)
        self.verify_max_distance(ts)

    def test_tree_sequence_simulated_mutations(self):
        ts = msprime.simulate(20, mutation_rate=10, recombination_rate=10)
        self.assertGreater(ts.get_num_trees(), 10)
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        self.verify_matrix(ts)
        self.verify_max_distance(ts)
        self.verify_max_mutations(ts)


def set_partitions(collection):
    """
    Returns an ierator over all partitions of the specified set.

    From https://stackoverflow.com/questions/19368375/set-partitions-in-python
    """
    if len(collection) == 1:
        yield [collection]
    else:
        first = collection[0]
        for smaller in set_partitions(collection[1:]):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
            yield [[first]] + smaller


def naive_mean_descendants(ts, reference_sets):
    """
    Straightforward implementation of mean sample ancestry by iterating
    over the trees and nodes in each tree.
    """
    # TODO generalise this to allow arbitrary nodes, not just samples.
    C = np.zeros((ts.num_nodes, len(reference_sets)))
    T = np.zeros(ts.num_nodes)
    tree_iters = [ts.trees(tracked_samples=sample_set) for sample_set in reference_sets]
    for _ in range(ts.num_trees):
        trees = [next(tree_iter) for tree_iter in tree_iters]
        span = trees[0].span
        for node in trees[0].nodes():
            num_samples = trees[0].num_samples(node)
            if num_samples > 0:
                for j, tree in enumerate(trees):
                    C[node, j] += span * tree.num_tracked_samples(node)
                T[node] += span
    for node in range(ts.num_nodes):
        if T[node] > 0:
            C[node] /= T[node]
    return C


class TestMeanDescendants(unittest.TestCase):
    """
    Tests the TreeSequence.mean_descendants method.
    """

    def verify(self, ts, reference_sets):
        C1 = naive_mean_descendants(ts, reference_sets)
        C2 = tsutil.mean_descendants(ts, reference_sets)
        C3 = ts.mean_descendants(reference_sets)
        self.assertEqual(C1.shape, C2.shape)
        self.assertTrue(np.allclose(C1, C2))
        self.assertTrue(np.allclose(C1, C3))
        return C1

    def test_two_populations_high_migration(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(8),
                msprime.PopulationConfiguration(8),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            recombination_rate=3,
            random_seed=5,
        )
        self.assertGreater(ts.num_trees, 1)
        self.verify(ts, [ts.samples(0), ts.samples(1)])

    def test_single_tree(self):
        ts = msprime.simulate(6, random_seed=1)
        S = [range(3), range(3, 6)]
        C = self.verify(ts, S)
        for j, samples in enumerate(S):
            tree = next(ts.trees(tracked_samples=samples))
            for u in tree.nodes():
                self.assertEqual(tree.num_tracked_samples(u), C[u, j])

    def test_single_tree_partial_samples(self):
        ts = msprime.simulate(6, random_seed=1)
        S = [range(3), range(3, 4)]
        C = self.verify(ts, S)
        for j, samples in enumerate(S):
            tree = next(ts.trees(tracked_samples=samples))
            for u in tree.nodes():
                self.assertEqual(tree.num_tracked_samples(u), C[u, j])

    def test_single_tree_all_sample_sets(self):
        ts = msprime.simulate(6, random_seed=1)
        for S in set_partitions(list(range(ts.num_samples))):
            C = self.verify(ts, S)
            for j, samples in enumerate(S):
                tree = next(ts.trees(tracked_samples=samples))
                for u in tree.nodes():
                    self.assertEqual(tree.num_tracked_samples(u), C[u, j])

    def test_many_trees_all_sample_sets(self):
        ts = msprime.simulate(6, recombination_rate=2, random_seed=1)
        self.assertGreater(ts.num_trees, 2)
        for S in set_partitions(list(range(ts.num_samples))):
            self.verify(ts, S)

    def test_wright_fisher_unsimplified_all_sample_sets(self):
        tables = wf.wf_sim(
            4,
            5,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=10,
        )
        tables.sort()
        ts = tables.tree_sequence()
        for S in set_partitions(list(ts.samples())):
            self.verify(ts, S)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            20,
            15,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=20,
        )
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify(ts, [samples[:10], samples[10:]])

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            30,
            10,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=5,
        )
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify(ts, [samples[:10], samples[10:]])


def naive_genealogical_nearest_neighbours(ts, focal, reference_sets):
    # Make sure everything is a sample so we can use the tracked_samples option.
    # This is a limitation of the current API.
    tables = ts.dump_tables()
    tables.nodes.set_columns(
        flags=np.ones_like(tables.nodes.flags), time=tables.nodes.time
    )
    ts = tables.tree_sequence()

    A = np.zeros((len(focal), len(reference_sets)))
    L = np.zeros(len(focal))
    reference_set_map = np.zeros(ts.num_nodes, dtype=int) - 1
    for k, ref_set in enumerate(reference_sets):
        for u in ref_set:
            reference_set_map[u] = k
    tree_iters = [
        ts.trees(tracked_samples=reference_nodes) for reference_nodes in reference_sets
    ]
    for _ in range(ts.num_trees):
        trees = list(map(next, tree_iters))
        length = trees[0].interval[1] - trees[0].interval[0]
        for j, u in enumerate(focal):
            focal_node_set = reference_set_map[u]
            # delta(u) = 1 if u exists in any of the reference sets; 0 otherwise
            delta = int(focal_node_set != -1)
            v = u
            while v != tskit.NULL:
                total = sum(tree.num_tracked_samples(v) for tree in trees)
                if total > delta:
                    break
                v = trees[0].parent(v)
            if v != tskit.NULL:
                for k, tree in enumerate(trees):
                    # If the focal node is in the current set, we subtract its
                    # contribution from the numerator
                    n = tree.num_tracked_samples(v) - (k == focal_node_set)
                    # If the focal node is in *any* reference set, we subtract its
                    # contribution from the demoninator.
                    A[j, k] += length * n / (total - delta)
                L[j] += length
    # Normalise by the accumulated value for each focal node.
    index = L > 0
    L = L[index]
    L = L.reshape((L.shape[0], 1))
    A[index, :] /= L
    return A


class TestGenealogicalNearestNeighbours(unittest.TestCase):
    """
    Tests the TreeSequence.genealogical_nearest_neighbours method.
    """

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

    def verify(self, ts, reference_sets, focal=None):
        if focal is None:
            focal = [u for refset in reference_sets for u in refset]
        A1 = naive_genealogical_nearest_neighbours(ts, focal, reference_sets)
        A2 = tsutil.genealogical_nearest_neighbours(ts, focal, reference_sets)
        A3 = ts.genealogical_nearest_neighbours(focal, reference_sets)
        A4 = ts.genealogical_nearest_neighbours(focal, reference_sets, num_threads=3)
        self.assertTrue(np.array_equal(A3, A4))
        self.assertEqual(A1.shape, A2.shape)
        self.assertEqual(A1.shape, A3.shape)
        self.assertTrue(np.allclose(A1, A2))
        self.assertTrue(np.allclose(A1, A3))
        if ts.num_edges > 0 and all(ts.node(u).is_sample() for u in focal):
            # When the focal nodes are samples, we can assert some stronger properties.
            self.assertTrue(np.allclose(np.sum(A1, axis=1), 1))
        return A1

    def test_simple_example_all_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0])
        self.assertEqual(list(A[0]), [1, 0])
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [4])
        self.assertEqual(list(A[0]), [1, 0])
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [2])
        self.assertEqual(list(A[0]), [0, 1])
        A = self.verify(ts, [[0, 2], [1, 3, 4]], [0])
        self.assertEqual(list(A[0]), [0, 1])
        A = self.verify(ts, [[0, 2], [1, 3, 4]], [4])
        self.assertEqual(list(A[0]), [0.5, 0.5])

    def test_simple_example_missing_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        A = self.verify(ts, [[0, 1], [2, 4]], [3])
        self.assertEqual(list(A[0]), [0, 1])
        A = self.verify(ts, [[0, 1], [2, 4]], [2])
        self.assertTrue(np.allclose(A[0], [2 / 3, 1 / 3]))

    def test_simple_example_internal_focal_node(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        focal = [7]  # An internal node
        reference_sets = [[4, 0, 1], [2, 3]]
        GNN = naive_genealogical_nearest_neighbours(ts, focal, reference_sets)
        self.assertTrue(np.allclose(GNN[0], np.array([1.0, 0.0])))
        GNN = tsutil.genealogical_nearest_neighbours(ts, focal, reference_sets)
        self.assertTrue(np.allclose(GNN[0], np.array([1.0, 0.0])))
        GNN = ts.genealogical_nearest_neighbours(focal, reference_sets)
        self.assertTrue(np.allclose(GNN[0], np.array([1.0, 0.0])))
        focal = [8]  # The root
        GNN = naive_genealogical_nearest_neighbours(ts, focal, reference_sets)
        self.assertTrue(np.allclose(GNN[0], np.array([0.6, 0.4])))
        GNN = tsutil.genealogical_nearest_neighbours(ts, focal, reference_sets)
        self.assertTrue(np.allclose(GNN[0], np.array([0.6, 0.4])))
        GNN = ts.genealogical_nearest_neighbours(focal, reference_sets)
        self.assertTrue(np.allclose(GNN[0], np.array([0.6, 0.4])))

    def test_two_populations_high_migration(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(18),
                msprime.PopulationConfiguration(18),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            recombination_rate=8,
            random_seed=5,
        )
        self.assertGreater(ts.num_trees, 1)
        self.verify(ts, [ts.samples(0), ts.samples(1)])

    def test_single_tree(self):
        ts = msprime.simulate(6, random_seed=1)
        S = [range(3), range(3, 6)]
        self.verify(ts, S)

    def test_single_tree_internal_reference_sets(self):
        ts = msprime.simulate(10, random_seed=1)
        tree = ts.first()
        S = [[u] for u in tree.children(tree.root)]
        self.verify(ts, S, ts.samples())

    def test_single_tree_all_nodes(self):
        ts = msprime.simulate(10, random_seed=1)
        S = [np.arange(ts.num_nodes, dtype=np.int32)]
        self.verify(ts, S, np.arange(ts.num_nodes, dtype=np.int32))

    def test_single_tree_partial_samples(self):
        ts = msprime.simulate(6, random_seed=1)
        S = [range(3), range(3, 4)]
        self.verify(ts, S)

    def test_single_tree_all_sample_sets(self):
        ts = msprime.simulate(6, random_seed=1)
        for S in set_partitions(list(range(ts.num_samples))):
            self.verify(ts, S)

    def test_many_trees_all_sample_sets(self):
        ts = msprime.simulate(6, recombination_rate=2, random_seed=1)
        self.assertGreater(ts.num_trees, 2)
        for S in set_partitions(list(range(ts.num_samples))):
            self.verify(ts, S)

    def test_many_trees_sequence_length(self):
        for L in [0.5, 1.5, 3.3333]:
            ts = msprime.simulate(6, length=L, recombination_rate=2, random_seed=1)
            self.verify(ts, [range(3), range(3, 6)])

    def test_many_trees_all_nodes(self):
        ts = msprime.simulate(6, length=4, recombination_rate=2, random_seed=1)
        S = [np.arange(ts.num_nodes, dtype=np.int32)]
        self.verify(ts, S, np.arange(ts.num_nodes, dtype=np.int32))

    def test_wright_fisher_unsimplified_all_sample_sets(self):
        tables = wf.wf_sim(
            4,
            5,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=10,
        )
        tables.sort()
        ts = tables.tree_sequence()
        for S in set_partitions(list(ts.samples())):
            self.verify(ts, S)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            20,
            15,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=20,
        )
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify(ts, [samples[:10], samples[10:]])

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            20,
            15,
            seed=1,
            deep_history=True,
            initial_generation_samples=True,
            num_loci=20,
        )
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        samples = ts.samples()
        founders = [u for u in samples if ts.node(u).time > 0]
        samples = [u for u in samples if ts.node(u).time == 0]
        self.verify(ts, [founders[:10], founders[10:]], samples)

    def test_wright_fisher_initial_generation_no_deep_history(self):
        tables = wf.wf_sim(
            20,
            15,
            seed=2,
            deep_history=False,
            initial_generation_samples=True,
            num_loci=20,
        )
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        samples = ts.samples()
        founders = [u for u in samples if ts.node(u).time > 0]
        samples = [u for u in samples if ts.node(u).time == 0]
        self.verify(ts, [founders[:10], founders[10:]], samples)

    def test_wright_fisher_unsimplified_multiple_roots(self):
        tables = wf.wf_sim(
            20,
            15,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=20,
        )
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify(ts, [samples[:10], samples[10:]])

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            31,
            10,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=5,
        )
        tables.sort()
        ts = tables.tree_sequence().simplify()
        samples = ts.samples()
        self.verify(ts, [samples[:10], samples[10:]])

    def test_wright_fisher_simplified_multiple_roots(self):
        tables = wf.wf_sim(
            31,
            10,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=5,
        )
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify(ts, [samples[:10], samples[10:]])

    def test_empty_ts(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(1, 0)
        tables.nodes.add_row(1, 0)
        ts = tables.tree_sequence()
        self.verify(ts, [[0], [1]])


def exact_genealogical_nearest_neighbours(ts, focal, reference_sets):
    # Same as above, except we return the per-tree value for a single node.

    # Make sure everyhing is a sample so we can use the tracked_samples option.
    # This is a limitation of the current API.
    tables = ts.dump_tables()
    tables.nodes.set_columns(
        flags=np.ones_like(tables.nodes.flags), time=tables.nodes.time
    )
    ts = tables.tree_sequence()

    A = np.zeros((len(reference_sets), ts.num_trees))
    L = np.zeros(ts.num_trees)
    reference_set_map = np.zeros(ts.num_nodes, dtype=int) - 1
    for k, ref_set in enumerate(reference_sets):
        for u in ref_set:
            reference_set_map[u] = k
    tree_iters = [
        ts.trees(tracked_samples=reference_nodes) for reference_nodes in reference_sets
    ]
    u = focal
    focal_node_set = reference_set_map[u]
    # delta(u) = 1 if u exists in any of the reference sets; 0 otherwise
    delta = int(focal_node_set != -1)
    for _ in range(ts.num_trees):
        trees = list(map(next, tree_iters))
        v = trees[0].parent(u)
        while v != tskit.NULL:
            total = sum(tree.num_tracked_samples(v) for tree in trees)
            if total > delta:
                break
            v = trees[0].parent(v)
        if v != tskit.NULL:
            # The length is only reported where the statistic is defined.
            L[trees[0].index] = trees[0].interval[1] - trees[0].interval[0]
            for k, tree in enumerate(trees):
                # If the focal node is in the current set, we subtract its
                # contribution from the numerator
                n = tree.num_tracked_samples(v) - (k == focal_node_set)
                # If the focal node is in *any* reference set, we subtract its
                # contribution from the demoninator.
                A[k, tree.index] = n / (total - delta)
    return A, L


def local_gnn(ts, focal, reference_sets):
    # Temporary implementation of the treewise GNN.
    reference_set_map = np.zeros(ts.num_nodes, dtype=int) - 1
    for k, reference_set in enumerate(reference_sets):
        for u in reference_set:
            if reference_set_map[u] != -1:
                raise ValueError("Duplicate value in reference sets")
            reference_set_map[u] = k

    K = len(reference_sets)
    A = np.zeros((len(focal), ts.num_trees, K))
    lefts = np.zeros(ts.num_trees, dtype=float)
    rights = np.zeros(ts.num_trees, dtype=float)
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    sample_count = np.zeros((ts.num_nodes, K), dtype=int)

    # Set the intitial conditions.
    for j in range(K):
        sample_count[reference_sets[j], j] = 1

    for t, ((left, right), edges_out, edges_in) in enumerate(ts.edge_diffs()):
        for edge in edges_out:
            parent[edge.child] = -1
            v = edge.parent
            while v != -1:
                sample_count[v] -= sample_count[edge.child]
                v = parent[v]
        for edge in edges_in:
            parent[edge.child] = edge.parent
            v = edge.parent
            while v != -1:
                sample_count[v] += sample_count[edge.child]
                v = parent[v]

        # Process this tree.
        for j, u in enumerate(focal):
            focal_reference_set = reference_set_map[u]
            delta = int(focal_reference_set != -1)
            p = parent[u]
            lefts[t] = left
            rights[t] = right
            while p != tskit.NULL:
                total = np.sum(sample_count[p])
                if total > delta:
                    break
                p = parent[p]
            if p != tskit.NULL:
                scale = 1 / (total - delta)
                for k, reference_set in enumerate(reference_sets):
                    n = sample_count[p, k] - int(focal_reference_set == k)
                    A[j, t, k] = n * scale
    return (A, lefts, rights)


class TestExactGenealogicalNearestNeighbours(TestGenealogicalNearestNeighbours):
    # This is a work in progress - these tests will be adapted to use the
    # treewise GNN when it's implemented.

    def verify(self, ts, reference_sets, focal=None):
        if focal is None:
            focal = [u for refset in reference_sets for u in refset]
        A = ts.genealogical_nearest_neighbours(focal, reference_sets)

        G, lefts, rights = local_gnn(ts, focal, reference_sets)
        for tree in ts.trees():
            self.assertEqual(lefts[tree.index], tree.interval[0])
            self.assertEqual(rights[tree.index], tree.interval[1])

        for j, u in enumerate(focal):
            T, L = exact_genealogical_nearest_neighbours(ts, u, reference_sets)
            self.assertTrue(np.allclose(G[j], T.T))
            # Ignore the cases where the node has no GNNs
            if np.sum(L) > 0:
                mean = np.sum(T * L, axis=1) / np.sum(L)
                self.assertTrue(np.allclose(mean, A[j]))
        return A
