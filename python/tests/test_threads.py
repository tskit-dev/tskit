# MIT License
#
# Copyright (c) 2018-2021 Tskit Developers
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
Test cases for threading enabled aspects of the API.
"""
import platform
import threading

import msprime
import numpy as np
import pytest

import tests.tsutil as tsutil
import tskit

IS_WINDOWS = platform.system() == "Windows"
IS_OSX = platform.system() == "Darwin"


def run_threads(worker, num_threads):
    results = [None for _ in range(num_threads)]
    threads = [
        threading.Thread(target=worker, args=(j, results)) for j in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


class TestLdCalculatorReplicates:
    """
    Tests the LdCalculator object to ensure we get correct results
    when using threads.
    """

    num_test_sites = 25

    def get_tree_sequence(self):
        ts = msprime.simulate(
            20, mutation_rate=10, recombination_rate=10, random_seed=8
        )
        return tsutil.subsample_sites(ts, self.num_test_sites)

    def test_get_r2_multiple_instances(self):
        # This is the nominal case where we have a separate LdCalculator
        # instance in each thread.
        ts = self.get_tree_sequence()
        ld_calc = tskit.LdCalculator(ts)
        A = ld_calc.get_r2_matrix()
        del ld_calc
        m = A.shape[0]

        def worker(thread_index, results):
            ld_calc = tskit.LdCalculator(ts)
            row = np.zeros(m)
            results[thread_index] = row
            for j in range(m):
                row[j] = ld_calc.get_r2(thread_index, j)

        results = run_threads(worker, m)
        for j in range(m):
            assert np.allclose(results[j], A[j])

    def test_get_r2_single_instance(self):
        # This is the degenerate case where we have a single LdCalculator
        # instance shared by the threads. We should have only one thread
        # actually executing get_r2() at one time.
        ts = self.get_tree_sequence()
        ld_calc = tskit.LdCalculator(ts)
        A = ld_calc.get_r2_matrix()
        m = A.shape[0]

        def worker(thread_index, results):
            row = np.zeros(m)
            results[thread_index] = row
            for j in range(m):
                row[j] = ld_calc.get_r2(thread_index, j)

        results = run_threads(worker, m)
        for j in range(m):
            assert np.allclose(results[j], A[j])

    def test_get_r2_array_multiple_instances(self):
        # This is the nominal case where we have a separate LdCalculator
        # instance in each thread.
        ts = self.get_tree_sequence()
        ld_calc = tskit.LdCalculator(ts)
        A = ld_calc.get_r2_matrix()
        m = A.shape[0]
        del ld_calc

        def worker(thread_index, results):
            ld_calc = tskit.LdCalculator(ts)
            results[thread_index] = np.array(ld_calc.get_r2_array(thread_index))

        results = run_threads(worker, m)
        for j in range(m):
            assert np.allclose(results[j], A[j, j + 1 :])

    def test_get_r2_array_single_instance(self):
        # This is the degenerate case where we have a single LdCalculator
        # instance shared by the threads. We should have only one thread
        # actually executing get_r2_array() at one time. Because the buffer
        # is shared by many different instances, we can't make any assertions
        # about the returned values --- they are essentially gibberish.
        # However, we shouldn't crash and burn, which is what this test
        # is here to check for.
        ts = self.get_tree_sequence()
        ld_calc = tskit.LdCalculator(ts)
        m = ts.get_num_mutations()

        def worker(thread_index, results):
            results[thread_index] = ld_calc.get_r2_array(thread_index).shape

        results = run_threads(worker, m)
        for j in range(m):
            assert results[j][0] == m - j - 1


# Temporarily skipping these on Windows and OSX See
# https://github.com/tskit-dev/tskit/issues/344
# https://github.com/tskit-dev/tskit/issues/1041
@pytest.mark.skipif(
    IS_WINDOWS or IS_OSX, reason="Can't test thread support on Windows."
)
class TestTables:
    """
    Tests to ensure that attempts to access tables in threads correctly
    raise an exception.
    """

    def get_tables(self):
        # TODO include migrations here.
        ts = msprime.simulate(
            100, mutation_rate=10, recombination_rate=10, random_seed=8
        )
        return ts.tables

    def run_multiple_writers(self, writer, num_writers=32):
        barrier = threading.Barrier(num_writers)

        def writer_proxy(thread_index, results):
            barrier.wait()
            # Attempts to operate on a table while locked should raise a RuntimeError
            try:
                writer(thread_index, results)
                results[thread_index] = 0
            except RuntimeError:
                results[thread_index] = 1

        results = run_threads(writer_proxy, num_writers)
        failures = sum(results)
        successes = num_writers - failures
        # Note: we would like to insist that #failures is > 0, but this is too
        # stochastic to guarantee for test purposes.
        assert failures >= 0
        assert successes > 0

    def run_failing_reader(self, writer, reader, num_readers=32):
        """
        Runs a test in which a single writer acceses some tables
        and a bunch of other threads try to read the data.
        """
        barrier = threading.Barrier(num_readers + 1)

        def writer_proxy():
            barrier.wait()
            writer()

        def reader_proxy(thread_index, results):
            barrier.wait()
            # Attempts to operate on a table while locked should raise a RuntimeError
            results[thread_index] = 0
            try:
                reader(thread_index, results)
            except RuntimeError:
                results[thread_index] = 1

        writer_thread = threading.Thread(target=writer_proxy)
        writer_thread.start()
        results = run_threads(reader_proxy, num_readers)
        writer_thread.join()

        failures = sum(results)
        successes = num_readers - failures
        # Note: we would like to insist that #failures is > 0, but this is too
        # stochastic to guarantee for test purposes.
        assert failures >= 0
        assert successes > 0

    def test_many_simplify_all_tables(self):
        tables = self.get_tables()

        def writer(thread_index, results):
            tables.simplify([0, 1])

        self.run_multiple_writers(writer)

    def test_many_sort(self):
        tables = self.get_tables()

        def writer(thread_index, results):
            tables.sort()

        self.run_multiple_writers(writer)

    def run_simplify_access_table(self, table_name, col_name):
        tables = self.get_tables()

        def writer():
            tables.simplify([0, 1])

        table = getattr(tables, table_name)

        def reader(thread_index, results):
            for _ in range(100):
                x = getattr(table, col_name)
                assert x.shape[0] == len(table)

        self.run_failing_reader(writer, reader)

    def run_sort_access_table(self, table_name, col_name):
        tables = self.get_tables()

        def writer():
            tables.sort()

        table = getattr(tables, table_name)

        def reader(thread_index, results):
            for _ in range(100):
                x = getattr(table, col_name)
                assert x.shape[0] == len(table)

        self.run_failing_reader(writer, reader)

    def test_simplify_access_nodes(self):
        self.run_simplify_access_table("nodes", "time")

    def test_simplify_access_edges(self):
        self.run_simplify_access_table("edges", "left")

    def test_simplify_access_sites(self):
        self.run_simplify_access_table("sites", "position")

    def test_simplify_access_mutations(self):
        self.run_simplify_access_table("mutations", "site")

    def test_sort_access_nodes(self):
        self.run_sort_access_table("nodes", "time")

    def test_sort_access_edges(self):
        self.run_sort_access_table("edges", "left")

    def test_sort_access_sites(self):
        self.run_sort_access_table("sites", "position")

    def test_sort_access_mutations(self):
        self.run_sort_access_table("mutations", "site")
