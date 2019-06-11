# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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
Test cases for the low level C interface to tskit.
"""
import collections
import itertools
import os
import platform
import random
import tempfile
import unittest

import numpy as np
import msprime

import _tskit

IS_WINDOWS = platform.system() == "Windows"


def get_tracked_sample_counts(st, tracked_samples):
    """
    Returns a list giving the number of samples in the specified list
    that are in the subtree rooted at each node.
    """
    nu = [0 for j in range(st.get_num_nodes())]
    for j in tracked_samples:
        # Duplicates not permitted.
        assert nu[j] == 0
        u = j
        while u != _tskit.NULL:
            nu[u] += 1
            u = st.get_parent(u)
    return nu


def get_sample_counts(tree_sequence, st):
    """
    Returns a list of the sample node counts for the specfied sparse tree.
    """
    nu = [0 for j in range(st.get_num_nodes())]
    for j in range(tree_sequence.get_num_samples()):
        u = j
        while u != _tskit.NULL:
            nu[u] += 1
            u = st.get_parent(u)
    return nu


class LowLevelTestCase(unittest.TestCase):
    """
    Superclass of tests for the low-level interface.
    """
    def verify_tree_dict(self, n, pi):
        """
        Verifies that the specified sparse tree in dict format is a
        consistent coalescent history for a sample of size n.
        """
        self.assertLessEqual(len(pi), 2 * n - 1)
        # _tskit.NULL should not be a node
        self.assertNotIn(_tskit.NULL, pi)
        # verify the root is equal for all samples
        root = 0
        while pi[root] != _tskit.NULL:
            root = pi[root]
        for j in range(n):
            k = j
            while pi[k] != _tskit.NULL:
                k = pi[k]
            self.assertEqual(k, root)
        # 0 to n - 1 inclusive should always be nodes
        for j in range(n):
            self.assertIn(j, pi)
        num_children = collections.defaultdict(int)
        for j in pi.keys():
            num_children[pi[j]] += 1
        # nodes 0 to n are samples.
        for j in range(n):
            self.assertNotEqual(pi[j], 0)
            self.assertEqual(num_children[j], 0)
        # All non-sample nodes should be binary
        for j in pi.keys():
            if j > n:
                self.assertGreaterEqual(num_children[j], 2)

    def get_example_tree_sequence(self, sample_size=10, length=1, random_seed=1):
        ts = msprime.simulate(
            sample_size, recombination_rate=0.1, mutation_rate=1,
            random_seed=random_seed, length=length)
        return ts.ll_tree_sequence

    def get_example_tree_sequences(self):
        yield self.get_example_tree_sequence()
        yield self.get_example_tree_sequence(2, 10)
        yield self.get_example_tree_sequence(20, 10)
        yield self.get_example_migration_tree_sequence()

    def get_example_migration_tree_sequence(self):
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1)
        return ts.ll_tree_sequence

    def verify_iterator(self, iterator):
        """
        Checks that the specified non-empty iterator implements the
        iterator protocol correctly.
        """
        list_ = list(iterator)
        self.assertGreater(len(list_), 0)
        for j in range(10):
            self.assertRaises(StopIteration, next, iterator)


class TestTableCollection(LowLevelTestCase):
    """
    Tests for the low-level TableCollection class
    """

    def test_reference_deletion(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        tc = ts.tables.ll_tables
        # Get references to all the tables
        tables = [
            tc.individuals, tc.nodes, tc.edges, tc.migrations, tc.sites, tc.mutations,
            tc.populations, tc.provenances]
        del tc
        for _ in range(10):
            for table in tables:
                self.assertGreater(len(str(table)), 0)

    def test_set_sequence_length_errors(self):
        tables = _tskit.TableCollection(1)
        with self.assertRaises(TypeError):
            del tables.sequence_length
        for bad_value in ["sdf", None, []]:
            with self.assertRaises(TypeError):
                tables.sequence_length = bad_value

    def test_set_sequence_length(self):
        tables = _tskit.TableCollection(1)
        self.assertEqual(tables.sequence_length, 1)
        for value in [-1, 1e6, 1e-22, 1000, 2**32, -10000]:
            tables.sequence_length = value
            self.assertEqual(tables.sequence_length, value)


class TestTreeSequence(LowLevelTestCase):
    """
    Tests for the low-level interface for the TreeSequence.
    """
    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(prefix="msp_ll_ts_")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)

    @unittest.skipIf(IS_WINDOWS, "File permissions on Windows")
    def test_file_errors(self):
        ts1 = self.get_example_tree_sequence()

        def loader(*args):
            ts2 = _tskit.TreeSequence()
            ts2.load(*args)

        for func in [ts1.dump, loader]:
            self.assertRaises(TypeError, func)
            for bad_type in [1, None, [], {}]:
                self.assertRaises(TypeError, func, bad_type)
            # Try to dump/load files we don't have access to or don't exist.
            for f in ["/", "/test.trees", "/dir_does_not_exist/x.trees"]:
                self.assertRaises(_tskit.FileFormatError, func, f)
                try:
                    func(f)
                except _tskit.FileFormatError as e:
                    message = str(e)
                    self.assertGreater(len(message), 0)
            # use a long filename and make sure we don't overflow error
            # buffers
            f = "/" + 4000 * "x"
            self.assertRaises(_tskit.FileFormatError, func, f)
            try:
                func(f)
            except _tskit.FileFormatError as e:
                message = str(e)
                self.assertLess(len(message), 1024)

    def test_initial_state(self):
        # Check the initial state to make sure that it is empty.
        ts = _tskit.TreeSequence()
        self.assertRaises(ValueError, ts.get_num_samples)
        self.assertRaises(ValueError, ts.get_sequence_length)
        self.assertRaises(ValueError, ts.get_num_trees)
        self.assertRaises(ValueError, ts.get_num_edges)
        self.assertRaises(ValueError, ts.get_num_mutations)
        self.assertRaises(ValueError, ts.get_num_migrations)
        self.assertRaises(ValueError, ts.get_num_migrations)
        self.assertRaises(ValueError, ts.get_genotype_matrix)
        self.assertRaises(ValueError, ts.dump)

    def test_num_nodes(self):
        for ts in self.get_example_tree_sequences():
            max_node = 0
            for j in range(ts.get_num_edges()):
                _, _, parent, child = ts.get_edge(j)
                for node in [parent, child]:
                    if node > max_node:
                        max_node = node
            self.assertEqual(max_node + 1, ts.get_num_nodes())

    def verify_dump_equality(self, ts):
        """
        Verifies that we can dump a copy of the specified tree sequence
        to the specified file, and load an identical copy.
        """
        ts.dump(self.temp_file)
        ts2 = _tskit.TreeSequence()
        ts2.load(self.temp_file)
        self.assertEqual(ts.get_num_samples(), ts2.get_num_samples())
        self.assertEqual(ts.get_sequence_length(), ts2.get_sequence_length())
        self.assertEqual(ts.get_num_mutations(), ts2.get_num_mutations())
        self.assertEqual(ts.get_num_nodes(), ts2.get_num_nodes())
        records1 = [ts.get_edge(j) for j in range(ts.get_num_edges())]
        records2 = [ts2.get_edge(j) for j in range(ts2.get_num_edges())]
        self.assertEqual(records1, records2)
        mutations1 = [ts.get_mutation(j) for j in range(ts.get_num_mutations())]
        mutations2 = [ts2.get_mutation(j) for j in range(ts2.get_num_mutations())]
        self.assertEqual(mutations1, mutations2)
        provenances1 = [ts.get_provenance(j) for j in range(ts.get_num_provenances())]
        provenances2 = [ts2.get_provenance(j) for j in range(ts2.get_num_provenances())]
        self.assertEqual(provenances1, provenances2)

    def test_dump_equality(self):
        for ts in self.get_example_tree_sequences():
            self.verify_dump_equality(ts)

    def verify_mutations(self, ts):
        mutations = [ts.get_mutation(j) for j in range(ts.get_num_mutations())]
        self.assertGreater(ts.get_num_mutations(), 0)
        self.assertEqual(len(mutations), ts.get_num_mutations())
        # Check the form of the mutations
        for j, (position, nodes, index) in enumerate(mutations):
            self.assertEqual(j, index)
            for node in nodes:
                self.assertIsInstance(node, int)
                self.assertGreaterEqual(node, 0)
                self.assertLessEqual(node, ts.get_num_nodes())
            self.assertIsInstance(position, float)
            self.assertGreater(position, 0)
            self.assertLess(position, ts.get_sequence_length())
        # mutations must be sorted by position order.
        self.assertEqual(mutations, sorted(mutations))

    def test_get_edge_interface(self):
        for ts in self.get_example_tree_sequences():
            num_edges = ts.get_num_edges()
            # We don't accept Python negative indexes here.
            self.assertRaises(IndexError, ts.get_edge, -1)
            for j in [0, 10, 10**6]:
                self.assertRaises(IndexError, ts.get_edge, num_edges + j)
            for x in [None, "", {}, []]:
                self.assertRaises(TypeError, ts.get_edge, x)

    def test_get_node_interface(self):
        for ts in self.get_example_tree_sequences():
            num_nodes = ts.get_num_nodes()
            # We don't accept Python negative indexes here.
            self.assertRaises(IndexError, ts.get_node, -1)
            for j in [0, 10, 10**6]:
                self.assertRaises(IndexError, ts.get_node, num_nodes + j)
            for x in [None, "", {}, []]:
                self.assertRaises(TypeError, ts.get_node, x)

    def test_get_genotype_matrix_interface(self):
        for ts in self.get_example_tree_sequences():
            num_samples = ts.get_num_samples()
            num_sites = ts.get_num_sites()
            G = ts.get_genotype_matrix()
            self.assertEqual(G.shape, (num_sites, num_samples))

    def test_get_migration_interface(self):
        ts = self.get_example_migration_tree_sequence()
        for bad_type in ["", None, {}]:
            self.assertRaises(TypeError, ts.get_migration, bad_type)
        num_records = ts.get_num_migrations()
        # We don't accept Python negative indexes here.
        self.assertRaises(IndexError, ts.get_migration, -1)
        for j in [0, 10, 10**6]:
            self.assertRaises(IndexError, ts.get_migration, num_records + j)

    def test_get_samples(self):
        ts = self.get_example_migration_tree_sequence()
        # get_samples takes no arguments.
        self.assertRaises(TypeError, ts.get_samples, 0)
        self.assertEqual(list(range(ts.get_num_samples())), ts.get_samples())

    def test_pairwise_diversity(self):
        for ts in self.get_example_tree_sequences():
            for bad_type in ["", None, {}]:
                self.assertRaises(TypeError, ts.get_pairwise_diversity, bad_type)
                self.assertRaises(TypeError, ts.get_pairwise_diversity, [0, bad_type])
            self.assertRaises(
                ValueError, ts.get_pairwise_diversity, [])
            self.assertRaises(
                ValueError, ts.get_pairwise_diversity, [0])
            self.assertRaises(
                ValueError, ts.get_pairwise_diversity,
                [0, ts.get_num_samples()])
            self.assertRaises(
                _tskit.LibraryError, ts.get_pairwise_diversity, [0, 0])
            samples = list(range(ts.get_num_samples()))
            pi1 = ts.get_pairwise_diversity(samples)
            self.assertGreaterEqual(pi1, 0)

    def test_genealogical_nearest_neighbours(self):
        for ts in self.get_example_tree_sequences():
            self.assertRaises(TypeError, ts.genealogical_nearest_neighbours)
            self.assertRaises(
                TypeError, ts.genealogical_nearest_neighbours, focal=None)
            self.assertRaises(
                TypeError, ts.genealogical_nearest_neighbours, focal=ts.get_samples(),
                reference_sets={})
            self.assertRaises(
                ValueError, ts.genealogical_nearest_neighbours, focal=ts.get_samples(),
                reference_sets=[])

            bad_array_values = ["", {}, "x", [[[0], [1, 2]]]]
            for bad_array_value in bad_array_values:
                self.assertRaises(
                    ValueError, ts.genealogical_nearest_neighbours,
                    focal=bad_array_value, reference_sets=[[0], [1]])
                self.assertRaises(
                    ValueError, ts.genealogical_nearest_neighbours,
                    focal=ts.get_samples(), reference_sets=[[0], bad_array_value])
                self.assertRaises(
                    ValueError, ts.genealogical_nearest_neighbours,
                    focal=ts.get_samples(), reference_sets=[bad_array_value])
            focal = ts.get_samples()
            A = ts.genealogical_nearest_neighbours(focal, [focal[2:], focal[:2]])
            self.assertEqual(A.shape, (len(focal), 2))

    def test_mean_descendants(self):
        for ts in self.get_example_tree_sequences():
            self.assertRaises(TypeError, ts.mean_descendants)
            self.assertRaises(TypeError, ts.mean_descendants, reference_sets={})
            self.assertRaises(ValueError, ts.mean_descendants, reference_sets=[])

            bad_array_values = ["", {}, "x", [[[0], [1, 2]]]]
            for bad_array_value in bad_array_values:
                self.assertRaises(
                    ValueError, ts.mean_descendants,
                    reference_sets=[[0], bad_array_value])
                self.assertRaises(
                    ValueError, ts.mean_descendants, reference_sets=[bad_array_value])
            focal = ts.get_samples()
            A = ts.mean_descendants([focal[2:], focal[:2]])
            self.assertEqual(A.shape, (ts.get_num_nodes(), 2))


class StatsInterfaceMixin(object):
    """
    Tests for the interface on specific stats.
    """

    def test_mode_errors(self):
        _, f, params = self.get_example()
        for bad_mode in ["", "not a mode", "SITE", "x" * 8192]:
            with self.assertRaises(ValueError):
                f(mode=bad_mode, **params)

        for bad_type in [123, {}, None, [[]]]:
            with self.assertRaises(TypeError):
                f(mode=bad_type, **params)

    def test_window_errors(self):
        ts, f, params = self.get_example()
        del params["windows"]
        for bad_array in ["asdf", None, [[[[]], [[]]]], np.zeros((10, 3, 4))]:
            with self.assertRaises(ValueError):
                f(windows=bad_array, **params)

        for bad_windows in [[], [0]]:
            with self.assertRaises(ValueError):
                f(windows=bad_windows, **params)
        L = ts.get_sequence_length()
        bad_windows = [
            [L, 0], [0.1, L], [-1, L], [0, L + 0.1], [0, 0.1, 0.1, L],
            [0, -1, L], [0, 0.1, 0.05, 0.2, L]]
        for bad_window in bad_windows:
            with self.assertRaises(_tskit.LibraryError):
                f(windows=bad_window, **params)

    def test_windows_output(self):
        ts, f, params = self.get_example()
        del params["windows"]
        for num_windows in range(1, 10):
            windows = np.linspace(0, ts.get_sequence_length(), num=num_windows + 1)
            self.assertEqual(windows.shape[0], num_windows + 1)
            sigma = f(windows=windows, **params)
            self.assertEqual(sigma.shape[0], num_windows)


class SampleSetMixin(StatsInterfaceMixin):

    def test_bad_sample_sets(self):
        ts, f, params = self.get_example()
        del params["sample_set_sizes"]
        del params["sample_sets"]

        with self.assertRaises(_tskit.LibraryError):
            f(sample_sets=[], sample_set_sizes=[], **params)

        n = ts.get_num_samples()
        samples = ts.get_samples()
        for bad_set_sizes in [[], [1], [n - 1], [n + 1], [n - 3, 1, 1], [1, n - 2]]:
            with self.assertRaises(ValueError):
                f(sample_set_sizes=bad_set_sizes, sample_sets=samples, **params)

        N = ts.get_num_nodes()
        for bad_node in [-1, N, N + 1, -N]:
            with self.assertRaises(_tskit.LibraryError):
                f(sample_set_sizes=[2], sample_sets=[0, bad_node], **params)

        for bad_sample in [n, n + 1, N - 1]:
            with self.assertRaises(_tskit.LibraryError):
                f(sample_set_sizes=[2], sample_sets=[0, bad_sample], **params)


class OneWaySampleStatsMixin(SampleSetMixin):
    """
    Tests for one-way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [ts.get_num_samples()],
            "sample_sets": ts.get_samples(),
            "windows": [0, ts.get_sequence_length()]}
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        result = method(
            [ts.get_num_samples()], ts.get_samples(), [0, ts.get_sequence_length()])
        self.assertEqual(result.shape, (1, 1))
        result = method(
            [ts.get_num_samples()], ts.get_samples(), [0, ts.get_sequence_length()],
            mode="node")
        self.assertEqual(result.shape, (1, ts.get_num_nodes(), 1))
        result = method(
            [ts.get_num_samples()], ts.get_samples(), ts.get_breakpoints(),
            mode="node")
        self.assertEqual(result.shape, (ts.get_num_trees(), ts.get_num_nodes(), 1))

    def test_output_dims(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)

        for mode in ["site", "branch"]:
            pi = method([n], samples, windows, mode=mode)
            self.assertEqual(pi.shape, (1, 1))
            pi = method([2, n - 2], samples, windows, mode=mode)
            self.assertEqual(pi.shape, (1, 2))
            pi = method([2, 2, n - 4], samples, windows, mode=mode)
            self.assertEqual(pi.shape, (1, 3))
            pi = method(np.ones(n).astype(np.uint32), samples, windows, mode=mode)
            self.assertEqual(pi.shape, (1, n))
        mode = "node"
        N = ts.get_num_nodes()
        pi = method([n], samples, windows, mode=mode)
        self.assertEqual(pi.shape, (1, N, 1))
        pi = method([2, n - 2], samples, windows, mode=mode)
        self.assertEqual(pi.shape, (1, N, 2))
        pi = method([2, 2, n - 4], samples, windows, mode=mode)
        self.assertEqual(pi.shape, (1, N, 3))
        pi = method(np.ones(n).astype(np.uint32), samples, windows, mode=mode)
        self.assertEqual(pi.shape, (1, N, n))


class TestDiversity(LowLevelTestCase, OneWaySampleStatsMixin):
    """
    Tests for the diversity method.
    """
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.diversity


class TestY1(LowLevelTestCase, OneWaySampleStatsMixin):
    """
    Tests for the diversity method.
    """
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.Y1


class TwoWaySampleStatsMixin(SampleSetMixin):
    """
    Tests for the two way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [2, ts.get_num_samples() - 2],
            "sample_sets": ts.get_samples(),
            "indexes": [[0, 1]],
            "windows": [0, ts.get_sequence_length()]}
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        div = method(
            [2, ts.get_num_samples() - 2],
            ts.get_samples(),
            [[0, 1]],
            windows=[0, ts.get_sequence_length()])
        self.assertEqual(div.shape, (1, 1))

    def test_output_dims(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)
        for mode in ["site", "branch"]:
            div = method([2, 2, n - 4], samples, [[0, 1]], windows, mode=mode)
            self.assertEqual(div.shape, (1, 1))
            div = method(
                [2, 2, n - 4], samples, [[0, 1], [1, 2]], windows, mode=mode)
            self.assertEqual(div.shape, (1, 2))
            div = method(
                [2, 2, n - 4], samples, [[0, 1], [1, 2], [0, 1]], windows, mode=mode)
            self.assertEqual(div.shape, (1, 3))

        N = ts.get_num_nodes()
        mode = "node"
        div = method([2, 2, n - 4], samples, [[0, 1]], windows, mode=mode)
        self.assertEqual(div.shape, (1, N, 1))
        div = method([2, 2, n - 4], samples, [[0, 1], [1, 2]], windows, mode=mode)
        self.assertEqual(div.shape, (1, N, 2))
        div = method(
            [2, 2, n - 4], samples, [[0, 1], [1, 2], [0, 1]], windows, mode=mode)
        self.assertEqual(div.shape, (1, N, 3))

    def test_set_index_errors(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)

        def f(indexes):
            method([2, 2, n - 4], samples, indexes, windows)

        for bad_array in ["wer", {}, [[[], []], [[], []]]]:
            with self.assertRaises(ValueError):
                f(bad_array)
        for bad_dim in [[[]], [[1], [1]]]:
            with self.assertRaises(ValueError):
                f(bad_dim)


class ThreeWaySampleStatsMixin(SampleSetMixin):
    """
    Tests for the two way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [1, 1, ts.get_num_samples() - 2],
            "sample_sets": ts.get_samples(),
            "indexes": [[0, 1, 2]],
            "windows": [0, ts.get_sequence_length()]}
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        div = method(
            [1, 1, ts.get_num_samples() - 2],
            ts.get_samples(),
            [[0, 1, 2]],
            windows=[0, ts.get_sequence_length()])
        self.assertEqual(div.shape, (1, 1))

    def test_output_dims(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)
        for mode in ["site", "branch"]:
            div = method([2, 2, n - 4], samples, [[0, 1, 2]], windows, mode=mode)
            self.assertEqual(div.shape, (1, 1))
            div = method(
                [1, 1, 2, n - 4], samples, [[0, 1, 2], [1, 2, 3]], windows, mode=mode)
            self.assertEqual(div.shape, (1, 2))
            div = method(
                [1, 1, 2, n - 4], samples, [[0, 1, 2], [1, 2, 3], [0, 1, 2]],
                windows, mode=mode)
            self.assertEqual(div.shape, (1, 3))

        N = ts.get_num_nodes()
        mode = "node"
        div = method([2, 2, n - 4], samples, [[0, 1, 2]], windows, mode=mode)
        self.assertEqual(div.shape, (1, N, 1))
        div = method(
            [1, 1, 2, n - 4], samples, [[0, 1, 2], [1, 2, 3]], windows, mode=mode)
        self.assertEqual(div.shape, (1, N, 2))
        div = method(
            [1, 1, 2, n - 4], samples, [[0, 1, 2], [1, 2, 3], [0, 1, 2]],
            windows, mode=mode)
        self.assertEqual(div.shape, (1, N, 3))

    def test_set_index_errors(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)

        def f(indexes):
            method([2, 2, n - 4], samples, indexes, windows)

        for bad_array in ["wer", {}, [[[], []], [[], []]]]:
            with self.assertRaises(ValueError):
                f(bad_array)
        for bad_dim in [[[]], [[1], [1]], [(0, 1)], [(0, 1, 2, 3)]]:
            with self.assertRaises(ValueError):
                f(bad_dim)


class FourWaySampleStatsMixin(SampleSetMixin):
    """
    Tests for the four way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [1, 1, 1, ts.get_num_samples() - 3],
            "sample_sets": ts.get_samples(),
            "indexes": [[0, 1, 2, 3]],
            "windows": [0, ts.get_sequence_length()]}
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        div = method(
            [1, 1, 1, ts.get_num_samples() - 3],
            ts.get_samples(),
            [[0, 1, 2, 3]],
            windows=[0, ts.get_sequence_length()])
        self.assertEqual(div.shape, (1, 1))

    def test_output_dims(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)
        for mode in ["site", "branch"]:
            div = method([2, 1, 1, n - 4], samples, [[0, 1, 2, 3]], windows, mode=mode)
            self.assertEqual(div.shape, (1, 1))
            div = method(
                [1, 1, 1, 1, n - 4], samples, [[0, 1, 2, 3], [1, 2, 3, 4]],
                windows, mode=mode)
            self.assertEqual(div.shape, (1, 2))
            div = method(
                [1, 1, 1, 1, n - 4], samples, [[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 4]],
                windows, mode=mode)
            self.assertEqual(div.shape, (1, 3))

        N = ts.get_num_nodes()
        mode = "node"
        div = method([2, 1, 1, n - 4], samples, [[0, 1, 2, 3]], windows, mode=mode)
        self.assertEqual(div.shape, (1, N, 1))
        div = method(
            [1, 1, 1, 1, n - 4], samples, [[0, 1, 2, 3], [1, 2, 3, 4]], windows,
            mode=mode)
        self.assertEqual(div.shape, (1, N, 2))
        div = method(
            [1, 1, 1, 1, n - 4], samples, [[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 4]],
            windows, mode=mode)
        self.assertEqual(div.shape, (1, N, 3))

    def test_set_index_errors(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)

        def f(indexes):
            method([2, 1, 1, n - 4], samples, indexes, windows)

        for bad_array in ["wer", {}, [[[], []], [[], []]]]:
            with self.assertRaises(ValueError):
                f(bad_array)
        for bad_dim in [[[]], [[1], [1]], [(0, 1)], [(0, 1, 2, 3, 4)]]:
            with self.assertRaises(ValueError):
                f(bad_dim)


class TestDivergence(LowLevelTestCase, TwoWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.divergence


class TestY2(LowLevelTestCase, TwoWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.Y2


class Testf2(LowLevelTestCase, TwoWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.f2


class TestY3(LowLevelTestCase, ThreeWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.Y3


class Testf3(LowLevelTestCase, ThreeWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.f3


class Testf4(LowLevelTestCase, FourWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.f4


class TestGeneralStatsInterface(LowLevelTestCase, StatsInterfaceMixin):
    """
    Tests for the general stats interface.
    """
    def get_example(self):
        ts = self.get_example_tree_sequence()
        W = np.zeros((ts.get_num_samples(), 1))
        params = {
            "weights": W,
            "summary_func": lambda x: np.cumsum(x),
            "output_dim": 1,
            "windows": ts.get_breakpoints()
        }
        return ts, ts.general_stat, params

    def test_basic_example(self):
        ts = self.get_example_tree_sequence()
        W = np.zeros((ts.get_num_samples(), 1))
        sigma = ts.general_stat(
            W, lambda x: np.cumsum(x), 1, ts.get_breakpoints(), mode="branch")
        self.assertEqual(sigma.shape, (ts.get_num_trees(), 1))

    def test_non_numpy_return(self):
        ts = self.get_example_tree_sequence()
        W = np.ones((ts.get_num_samples(), 3))
        sigma = ts.general_stat(
            W, lambda x: [sum(x)], 1, ts.get_breakpoints(), mode="branch")
        self.assertEqual(sigma.shape, (ts.get_num_trees(), 1))
        sigma = ts.general_stat(
            W, lambda x: [2, 2], 2, ts.get_breakpoints(), mode="branch")
        self.assertEqual(sigma.shape, (ts.get_num_trees(), 2))

    def test_complicated_numpy_function(self):
        ts = self.get_example_tree_sequence(sample_size=20, length=30, random_seed=325)
        W = np.zeros((ts.get_num_samples(), 4))

        def f(x):
            y = np.sum(x * x), np.prod(x + np.arange(x.shape[0]))
            return y
        sigma = ts.general_stat(W, f, 2, ts.get_breakpoints(), mode="branch")
        self.assertEqual(sigma.shape, (ts.get_num_trees(), 2))

    def test_input_dims(self):
        ts = self.get_example_tree_sequence()
        for k in range(1, 20):
            W = np.zeros((ts.get_num_samples(), k))
            sigma = ts.general_stat(
                W, lambda x: np.cumsum(x), k, ts.get_breakpoints(), mode="branch")
            self.assertEqual(sigma.shape, (ts.get_num_trees(), k))
            sigma = ts.general_stat(
                W, lambda x: [np.sum(x)], 1, ts.get_breakpoints(), mode="branch")
            self.assertEqual(sigma.shape, (ts.get_num_trees(), 1))

    def test_W_errors(self):
        ts = self.get_example_tree_sequence()
        n = ts.get_num_samples()
        for bad_array in [[], [0, 1], [[[[]], [[]]]], np.zeros((10, 3, 4))]:
            with self.assertRaises(ValueError):
                ts.general_stat(bad_array, lambda x: x, 1, ts.get_breakpoints())

        for bad_size in [n - 1, n + 1, 0]:
            W = np.zeros((bad_size, 1))
            with self.assertRaises(ValueError):
                ts.general_stat(W, lambda x: x, 1, ts.get_breakpoints())

    def test_summary_func_errors(self):
        ts = self.get_example_tree_sequence()
        W = np.zeros((ts.get_num_samples(), 1))
        for bad_type in ["sdf", 1, {}]:
            with self.assertRaises(TypeError):
                ts.general_stat(W, bad_type, 1, ts.get_breakpoints())

        # Wrong numbers of arguments to f
        with self.assertRaises(TypeError):
            ts.general_stat(W, lambda: 0, 1, ts.get_breakpoints())
        with self.assertRaises(TypeError):
            ts.general_stat(W, lambda x, y: None, 1, ts.get_breakpoints())

        # Exceptions within f are correctly raised.
        for exception in [ValueError, TypeError]:
            def f(x):
                raise exception("test")
            with self.assertRaises(exception):
                ts.general_stat(W, f, 1, ts.get_breakpoints())

        # Wrong output dimensions
        for bad_array in [[1, 1], range(10)]:
            with self.assertRaises(ValueError):
                ts.general_stat(W, lambda x: bad_array, 1, ts.get_breakpoints())
        with self.assertRaises(ValueError):
            ts.general_stat(W, lambda x: [1], 2, ts.get_breakpoints())

        # Bad arrays returned from f
        for bad_array in [["sdf"], 0, "w4", None]:
            with self.assertRaises(ValueError):
                ts.general_stat(W, lambda x: bad_array, 1, ts.get_breakpoints())


class TestTreeDiffIterator(LowLevelTestCase):
    """
    Tests for the low-level tree diff iterator.
    """
    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        self.assertRaises(ValueError, _tskit.TreeDiffIterator, ts)

    def test_constructor(self):
        self.assertRaises(TypeError, _tskit.TreeDiffIterator)
        self.assertRaises(TypeError, _tskit.TreeDiffIterator, None)
        ts = self.get_example_tree_sequence()
        before = list(_tskit.TreeDiffIterator(ts))
        iterator = _tskit.TreeDiffIterator(ts)
        del ts
        # We should keep a reference to the tree sequence.
        after = list(iterator)
        self.assertEqual(before, after)

    def test_iterator(self):
        ts = self.get_example_tree_sequence()
        self.verify_iterator(_tskit.TreeDiffIterator(ts))


class TestVcfConverter(LowLevelTestCase):
    """
    Tests for the VcfConverter class.
    """
    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        self.assertRaises(ValueError, _tskit.VcfConverter, ts)

    def test_constructor(self):
        self.assertRaises(TypeError, _tskit.VcfConverter)
        self.assertRaises(TypeError, _tskit.VcfConverter, None)
        ts = self.get_example_tree_sequence()
        self.assertRaises(TypeError, _tskit.VcfConverter, ts, ploidy="xyt")
        self.assertRaises(ValueError, _tskit.VcfConverter, ts, ploidy=0)
        self.assertRaises(TypeError, _tskit.VcfConverter, ts, contig_id=None)
        self.assertRaises(ValueError, _tskit.VcfConverter, ts, contig_id="")

    def test_iterator(self):
        ts = self.get_example_tree_sequence()
        self.verify_iterator(_tskit.VcfConverter(ts))


class TestVariantGenerator(LowLevelTestCase):
    """
    Tests for the VariantGenerator class.
    """
    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        self.assertRaises(ValueError, _tskit.VariantGenerator, ts)

    def test_constructor(self):
        self.assertRaises(TypeError, _tskit.VariantGenerator)
        self.assertRaises(TypeError, _tskit.VariantGenerator, None)
        ts = self.get_example_tree_sequence()
        self.assertRaises(ValueError, _tskit.VariantGenerator, ts, samples={})
        self.assertRaises(
            _tskit.LibraryError, _tskit.VariantGenerator, ts, samples=[-1, 2])

    def test_iterator(self):
        ts = self.get_example_tree_sequence()
        self.verify_iterator(_tskit.VariantGenerator(ts))


class TestHaplotypeGenerator(LowLevelTestCase):
    """
    Tests for the HaplotypeGenerator class.
    """
    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        self.assertRaises(ValueError, _tskit.HaplotypeGenerator, ts)

    def test_constructor(self):
        self.assertRaises(TypeError, _tskit.HaplotypeGenerator)
        self.assertRaises(TypeError, _tskit.HaplotypeGenerator, None)

    def test_bad_id(self):
        ts = self.get_example_tree_sequence()
        hg = _tskit.HaplotypeGenerator(ts)
        n = ts.get_num_samples()
        for bad_id in [-1, n, n + 1]:
            with self.assertRaises(_tskit.LibraryError):
                hg.get_haplotype(bad_id)


class TestLdCalculator(LowLevelTestCase):
    """
    Tests for the LdCalculator class.
    """
    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        self.assertRaises(ValueError, _tskit.LdCalculator, ts)

    def test_constructor(self):
        self.assertRaises(TypeError, _tskit.LdCalculator)
        self.assertRaises(TypeError, _tskit.LdCalculator, None)

    def test_get_r2(self):
        ts = self.get_example_tree_sequence()
        calc = _tskit.LdCalculator(ts)
        n = ts.get_num_sites()
        for bad_id in [-1, n, n + 1]:
            with self.assertRaises(_tskit.LibraryError):
                calc.get_r2(0, bad_id)
            with self.assertRaises(_tskit.LibraryError):
                calc.get_r2(bad_id, 0)

    def test_get_r2_array(self):
        ts = self.get_example_tree_sequence()
        calc = _tskit.LdCalculator(ts)

        self.assertRaises(TypeError, calc.get_r2_array)
        self.assertRaises(TypeError, calc.get_r2_array, None)
        # Doesn't support buffer protocol, so raises typeerror
        self.assertRaises(TypeError, calc.get_r2_array, None, 0)

        n = ts.get_num_sites()
        self.assertGreater(n, 2)
        with self.assertRaises(BufferError):
            calc.get_r2_array(bytes(100), 0)

        buff = bytearray(1024)
        with self.assertRaises(ValueError):
            calc.get_r2_array(buff, 0, max_distance=-1)
        with self.assertRaises(ValueError):
            calc.get_r2_array(buff, 0, direction=1000)
        # TODO this API is poor, we should explicitly catch these negative
        # size errors.
        for bad_max_mutations in [-2, -3, -2**32]:
            with self.assertRaises(BufferError):
                calc.get_r2_array(buff, 0, max_mutations=bad_max_mutations)
        for bad_start_pos in [-1, n, n + 1]:
            with self.assertRaises(_tskit.LibraryError):
                calc.get_r2_array(buff, bad_start_pos)


class TestTree(LowLevelTestCase):
    """
    Tests on the low-level sparse tree interface.
    """

    def test_options(self):
        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        self.assertEqual(st.get_options(), 0)
        # We should still be able to count the samples, just inefficiently.
        self.assertEqual(st.get_num_samples(0), 1)
        self.assertRaises(_tskit.LibraryError, st.get_num_tracked_samples, 0)
        all_options = [
            0, _tskit.SAMPLE_COUNTS, _tskit.SAMPLE_LISTS,
            _tskit.SAMPLE_COUNTS | _tskit.SAMPLE_LISTS]
        for options in all_options:
            tree = _tskit.Tree(ts, options=options)
            copy = tree.copy()
            for st in [tree, copy]:
                self.assertEqual(st.get_options(), options)
                self.assertEqual(st.get_num_samples(0), 1)
                if options & _tskit.SAMPLE_COUNTS:
                    self.assertEqual(st.get_num_tracked_samples(0), 0)
                else:
                    self.assertRaises(_tskit.LibraryError, st.get_num_tracked_samples, 0)
                if options & _tskit.SAMPLE_LISTS:
                    self.assertEqual(0, st.get_left_sample(0))
                    self.assertEqual(0, st.get_right_sample(0))
                else:
                    self.assertRaises(ValueError, st.get_left_sample, 0)
                    self.assertRaises(ValueError, st.get_right_sample, 0)
                    self.assertRaises(ValueError, st.get_next_sample, 0)

    def test_site_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_sites(), ts.get_num_sites() + 1]:
            self.assertRaises(IndexError, ts.get_site, bad_index)

    def test_mutation_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_mutations(), ts.get_num_mutations() + 1]:
            self.assertRaises(IndexError, ts.get_mutation, bad_index)

    def test_individual_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_individuals(), ts.get_num_individuals() + 1]:
            self.assertRaises(IndexError, ts.get_individual, bad_index)

    def test_population_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_populations(), ts.get_num_populations() + 1]:
            self.assertRaises(IndexError, ts.get_population, bad_index)

    def test_provenance_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_provenances(), ts.get_num_provenances() + 1]:
            self.assertRaises(IndexError, ts.get_provenance, bad_index)

    def test_sites(self):
        for ts in self.get_example_tree_sequences():
            st = _tskit.Tree(ts)
            all_sites = [ts.get_site(j) for j in range(ts.get_num_sites())]
            all_tree_sites = []
            j = 0
            mutation_id = 0
            while st.next():
                tree_sites = st.get_sites()
                self.assertEqual(st.get_num_sites(), len(tree_sites))
                all_tree_sites.extend(tree_sites)
                for position, ancestral_state, mutations, index, metadata in tree_sites:
                    self.assertTrue(st.get_left() <= position < st.get_right())
                    self.assertEqual(index, j)
                    self.assertEqual(metadata, b"")
                    for mut_id in mutations:
                        site, node, derived_state, parent, metadata = \
                            ts.get_mutation(mut_id)
                        self.assertEqual(site, index)
                        self.assertEqual(mutation_id, mut_id)
                        self.assertNotEqual(st.get_parent(node), _tskit.NULL)
                        self.assertEqual(metadata, b"")
                        mutation_id += 1
                    j += 1
            self.assertEqual(all_tree_sites, all_sites)

    def test_constructor(self):
        self.assertRaises(TypeError, _tskit.Tree)
        for bad_type in ["", {}, [], None, 0]:
            self.assertRaises(
                TypeError, _tskit.Tree, bad_type)
        ts = self.get_example_tree_sequence()
        for bad_type in ["", {}, True, 1, None]:
            self.assertRaises(
                TypeError, _tskit.Tree, ts, tracked_samples=bad_type)
        for bad_type in ["", {}, None, []]:
            self.assertRaises(
                TypeError, _tskit.Tree, ts, options=bad_type)
        for ts in self.get_example_tree_sequences():
            st = _tskit.Tree(ts)
            self.assertEqual(st.get_num_nodes(), ts.get_num_nodes())
            # An uninitialised sparse tree should always be zero.
            self.assertEqual(st.get_left_root(), 0)
            self.assertEqual(st.get_left(), 0)
            self.assertEqual(st.get_right(), 0)
            for j in range(ts.get_num_samples()):
                self.assertEqual(st.get_parent(j), _tskit.NULL)
                self.assertEqual(st.get_children(j), tuple())
                self.assertEqual(st.get_time(j), 0)

    def test_bad_tracked_samples(self):
        ts = self.get_example_tree_sequence()
        options = _tskit.SAMPLE_COUNTS
        for bad_type in ["", {}, [], None]:
            self.assertRaises(
                TypeError, _tskit.Tree, ts, options=options,
                tracked_samples=[bad_type])
            self.assertRaises(
                TypeError, _tskit.Tree, ts, options=options,
                tracked_samples=[1, bad_type])
        for bad_sample in [10**6, -1e6]:
            self.assertRaises(
                ValueError, _tskit.Tree, ts, options=options,
                tracked_samples=[bad_sample])
            self.assertRaises(
                ValueError, _tskit.Tree, ts, options=options,
                tracked_samples=[1, bad_sample])
            self.assertRaises(
                ValueError, _tskit.Tree, ts,
                tracked_samples=[1, bad_sample, 1])

    def test_while_loop_semantics(self):
        for ts in self.get_example_tree_sequences():
            tree = _tskit.Tree(ts)
            # Any mixture of prev and next is OK and gives a valid iteration.
            for _ in range(2):
                j = 0
                while tree.next():
                    self.assertEqual(tree.get_index(), j)
                    j += 1
                self.assertEqual(j, ts.get_num_trees())
            for _ in range(2):
                j = ts.get_num_trees()
                while tree.prev():
                    self.assertEqual(tree.get_index(), j - 1)
                    j -= 1
                self.assertEqual(j, 0)
            j = 0
            while tree.next():
                self.assertEqual(tree.get_index(), j)
                j += 1
            self.assertEqual(j, ts.get_num_trees())

    def test_count_all_samples(self):
        for ts in self.get_example_tree_sequences():
            self.verify_iterator(_tskit.TreeDiffIterator(ts))
            st = _tskit.Tree(ts, options=_tskit.SAMPLE_COUNTS)
            # Without initialisation we should be 0 samples for every node
            # that is not a sample.
            for j in range(st.get_num_nodes()):
                count = 1 if j < ts.get_num_samples() else 0
                self.assertEqual(st.get_num_samples(j), count)
                self.assertEqual(st.get_num_tracked_samples(j), 0)
            while st.next():
                nu = get_sample_counts(ts, st)
                nu_prime = [st.get_num_samples(j) for j in range(st.get_num_nodes())]
                self.assertEqual(nu, nu_prime)
                # For tracked samples, this should be all zeros.
                nu = [st.get_num_tracked_samples(j) for j in range(st.get_num_nodes())]
                self.assertEqual(nu, list([0 for _ in nu]))

    def test_count_tracked_samples(self):
        # Ensure that there are some non-binary nodes.
        non_binary = False
        for ts in self.get_example_tree_sequences():
            st = _tskit.Tree(ts)
            while st.next():
                for u in range(ts.get_num_nodes()):
                    if len(st.get_children(u)) > 1:
                        non_binary = True
            samples = [j for j in range(ts.get_num_samples())]
            powerset = itertools.chain.from_iterable(
                itertools.combinations(samples, r) for r in range(len(samples) + 1))
            max_sets = 100
            for _, subset in zip(range(max_sets), map(list, powerset)):
                # Ordering shouldn't make any difference.
                random.shuffle(subset)
                st = _tskit.Tree(
                    ts, options=_tskit.SAMPLE_COUNTS, tracked_samples=subset)
                while st.next():
                    nu = get_tracked_sample_counts(st, subset)
                    nu_prime = [
                        st.get_num_tracked_samples(j) for j in
                        range(st.get_num_nodes())]
                    self.assertEqual(nu, nu_prime)
            # Passing duplicated values should raise an error
            sample = 1
            for j in range(2, 20):
                tracked_samples = [sample for _ in range(j)]
                self.assertRaises(
                    _tskit.LibraryError, _tskit.Tree,
                    ts, options=_tskit.SAMPLE_COUNTS,
                    tracked_samples=tracked_samples)
        self.assertTrue(non_binary)

    def test_bounds_checking(self):
        for ts in self.get_example_tree_sequences():
            n = ts.get_num_nodes()
            st = _tskit.Tree(ts, options=_tskit.SAMPLE_COUNTS | _tskit.SAMPLE_LISTS)
            for v in [-100, -1, n + 1, n + 100, n * 100]:
                self.assertRaises(ValueError, st.get_parent, v)
                self.assertRaises(ValueError, st.get_children, v)
                self.assertRaises(ValueError, st.get_time, v)
                self.assertRaises(ValueError, st.get_left_sample, v)
                self.assertRaises(ValueError, st.get_right_sample, v)
                self.assertRaises(ValueError, st.is_descendant, v, 0)
                self.assertRaises(ValueError, st.is_descendant, 0, v)
            n = ts.get_num_samples()
            for v in [-100, -1, n + 1, n + 100, n * 100]:
                self.assertRaises(ValueError, st.get_next_sample, v)

    def test_mrca_interface(self):
        for ts in self.get_example_tree_sequences():
            num_nodes = ts.get_num_nodes()
            st = _tskit.Tree(ts)
            for v in [num_nodes, 10**6, _tskit.NULL]:
                self.assertRaises(ValueError, st.get_mrca, v, v)
                self.assertRaises(ValueError, st.get_mrca, v, 1)
                self.assertRaises(ValueError, st.get_mrca, 1, v)
            # All the mrcas for an uninitialised tree should be _tskit.NULL
            for u, v in itertools.combinations(range(num_nodes), 2):
                self.assertEqual(st.get_mrca(u, v), _tskit.NULL)

    def test_newick_precision(self):

        def get_times(tree):
            """
            Returns the time strings from the specified newick tree.
            """
            ret = []
            current_time = None
            for c in tree:
                if c == ":":
                    current_time = ""
                elif c in [",", ")"]:
                    ret.append(current_time)
                    current_time = None
                elif current_time is not None:
                    current_time += c
            return ret

        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        while st.next():
            self.assertRaises(ValueError, st.get_newick, root=0, precision=-1)
            self.assertRaises(ValueError, st.get_newick, root=0, precision=17)
            self.assertRaises(ValueError, st.get_newick, root=0, precision=100)
            for precision in range(17):
                tree = st.get_newick(
                    root=st.get_left_root(), precision=precision).decode()
                times = get_times(tree)
                self.assertGreater(len(times), ts.get_num_samples())
                for t in times:
                    if precision == 0:
                        self.assertNotIn(".", t)
                    else:
                        point = t.find(".")
                        self.assertEqual(precision, len(t) - point - 1)

    def test_cleared_tree(self):
        ts = self.get_example_tree_sequence()
        samples = ts.get_samples()

        def check_tree(tree):
            self.assertEqual(tree.get_index(), -1)
            self.assertEqual(tree.get_left_root(), samples[0])
            self.assertEqual(tree.get_mrca(0, 1), _tskit.NULL)
            for u in range(ts.get_num_nodes()):
                self.assertEqual(tree.get_parent(u), _tskit.NULL)
                self.assertEqual(tree.get_left_child(u), _tskit.NULL)
                self.assertEqual(tree.get_right_child(u), _tskit.NULL)

        tree = _tskit.Tree(ts)
        check_tree(tree)
        while tree.next():
            pass
        check_tree(tree)
        while tree.prev():
            pass
        check_tree(tree)

    def test_newick_interface(self):
        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        # TODO this will break when we correctly handle multiple roots.
        self.assertEqual(st.get_newick(0), b"1;")
        for bad_type in [None, "", [], {}]:
            self.assertRaises(TypeError, st.get_newick, precision=bad_type)
            self.assertRaises(TypeError, st.get_newick, ts, time_scale=bad_type)
        while st.next():
            newick = st.get_newick(st.get_left_root())
            self.assertTrue(newick.endswith(b";"))

    def test_index(self):
        for ts in self.get_example_tree_sequences():
            st = _tskit.Tree(ts)
            index = 0
            while st.next():
                self.assertEqual(index, st.get_index())
                index += 1

    def test_bad_mutations(self):
        ts = self.get_example_tree_sequence()
        tables = _tskit.TableCollection()
        ts.dump_tables(tables)

        def f(mutations):
            position = []
            node = []
            site = []
            ancestral_state = []
            ancestral_state_offset = [0]
            derived_state = []
            derived_state_offset = [0]
            for j, (p, n) in enumerate(mutations):
                site.append(j)
                position.append(p)
                ancestral_state.append("0")
                ancestral_state_offset.append(ancestral_state_offset[-1] + 1)
                derived_state.append("1")
                derived_state_offset.append(derived_state_offset[-1] + 1)
                node.append(n)
            tables.sites.set_columns(dict(
                position=position, ancestral_state=ancestral_state,
                ancestral_state_offset=ancestral_state_offset,
                metadata=None, metadata_offset=None))
            tables.mutations.set_columns(dict(
                site=site, node=node, derived_state=derived_state,
                derived_state_offset=derived_state_offset,
                parent=None, metadata=None, metadata_offset=None))
            ts2 = _tskit.TreeSequence()
            ts2.load_tables(tables)
        self.assertRaises(_tskit.LibraryError, f, [(0.1, -1)])
        length = ts.get_sequence_length()
        u = ts.get_num_nodes()
        for bad_node in [u, u + 1, 2 * u]:
            self.assertRaises(_tskit.LibraryError, f, [(0.1, bad_node)])
        for bad_pos in [-1, length, length + 1]:
            self.assertRaises(_tskit.LibraryError, f, [(length, 0)])

    def test_sample_list(self):
        options = _tskit.SAMPLE_COUNTS | _tskit.SAMPLE_LISTS
        # Note: we're assuming that samples are 0-n here.
        for ts in self.get_example_tree_sequences():
            t = _tskit.Tree(ts, options=options)
            while t.next():
                # All sample nodes should have themselves.
                for j in range(ts.get_num_samples()):
                    self.assertEqual(t.get_left_sample(j), j)
                    self.assertEqual(t.get_right_sample(j), j)

                # All non-tree nodes should have 0
                for j in range(t.get_num_nodes()):
                    if t.get_parent(j) == _tskit.NULL \
                            and t.get_left_child(j) == _tskit.NULL:
                        self.assertEqual(t.get_left_sample(j), _tskit.NULL)
                        self.assertEqual(t.get_right_sample(j), _tskit.NULL)
                # The roots should have all samples.
                u = t.get_left_root()
                samples = []
                while u != _tskit.NULL:
                    sample = t.get_left_sample(u)
                    end = t.get_right_sample(u)
                    while True:
                        samples.append(sample)
                        if sample == end:
                            break
                        sample = t.get_next_sample(sample)
                    u = t.get_right_sib(u)
                self.assertEqual(sorted(samples), list(range(ts.get_num_samples())))

    def test_equality(self):
        last_ts = None
        for ts in self.get_example_tree_sequences():
            t1 = _tskit.Tree(ts)
            t2 = _tskit.Tree(ts)
            self.assertTrue(t1.equals(t2))
            self.assertTrue(t2.equals(t1))
            while True:
                self.assertTrue(t1.equals(t2))
                self.assertTrue(t2.equals(t1))
                n1 = t1.next()
                self.assertFalse(t1.equals(t2))
                self.assertFalse(t2.equals(t1))
                n2 = t2.next()
                self.assertEqual(n1, n2)
                if not n1:
                    break
            if last_ts is not None:
                t2 = _tskit.Tree(last_ts)
                self.assertFalse(t1.equals(t2))
                self.assertFalse(t2.equals(t1))
            last_ts = ts

    def test_copy(self):
        for ts in self.get_example_tree_sequences():
            t1 = _tskit.Tree(ts)
            t2 = t1.copy()
            self.assertEqual(t1.get_index(), t2.get_index())
            self.assertIsNot(t1, t2)
            while t1.next():
                t2 = t1.copy()
                self.assertEqual(t1.get_index(), t2.get_index())


class TestModuleFunctions(unittest.TestCase):
    """
    Tests for the module level functions.
    """
    def test_kastore_version(self):
        version = _tskit.get_kastore_version()
        self.assertEqual(version, (1, 0, 0))

    def test_tskit_version(self):
        version = _tskit.get_tskit_version()
        self.assertEqual(version, (0, 99, 2))
