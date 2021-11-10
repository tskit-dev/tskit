# MIT License
#
# Copyright (c) 2018-2021 Tskit Developers
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
import contextlib
import io

import msprime
import numpy as np
import pytest

import _tskit
import tests
import tests.test_wright_fisher as wf
import tests.tsutil as tsutil
import tskit


@contextlib.contextmanager
def suppress_division_by_zero_warning():
    with np.errstate(invalid="ignore", divide="ignore"):
        yield


def get_r2_matrix(ts):
    """
    Simple site-based version assuming biallic sites.
    """
    A = np.zeros((ts.num_sites, ts.num_sites))
    G = ts.genotype_matrix()
    n = ts.num_samples
    for a in range(ts.num_sites):
        A[a, a] = 1
        fA = np.sum(G[a] != 0) / n
        for b in range(a + 1, ts.num_sites):
            fB = np.sum(G[b] != 0) / n
            nAB = np.sum(np.logical_and(G[a] != 0, G[b] != 0))
            fAB = nAB / n
            D = fAB - fA * fB
            denom = fA * fB * (1 - fA) * (1 - fB)
            A[a, b] = D * D
            with suppress_division_by_zero_warning():
                A[a, b] /= denom
            A[b, a] = A[a, b]
    return A


def _compute_r2(tree, n, f_a, site_b):
    assert len(site_b.mutations) == 1
    assert site_b.ancestral_state != site_b.mutations[0].derived_state
    f_b = tree.num_samples(site_b.mutations[0].node) / n
    f_ab = tree.num_tracked_samples(site_b.mutations[0].node) / n
    D2 = (f_ab - f_a * f_b) ** 2
    denom = f_a * f_b * (1 - f_a) * (1 - f_b)
    if denom == 0:
        return np.nan
    return D2 / denom


def ts_r2(ts, a, b):
    """
    Returns the r2 value between sites a and b in the specified tree sequence.
    """
    a, b = (a, b) if a < b else (b, a)
    site_a = ts.site(a)
    site_b = ts.site(b)
    assert len(site_a.mutations) == 1
    assert len(site_b.mutations) == 1
    n = ts.num_samples
    tree = ts.at(site_a.position)
    a_samples = list(tree.samples(site_a.mutations[0].node))
    f_a = len(a_samples) / n
    tree = ts.at(site_b.position, tracked_samples=a_samples)
    return _compute_r2(tree, n, f_a, site_b)


class LdArrayCalculator:
    """
    Utility class to help organise the state required when tracking all
    the different termination conditions.
    """

    def __init__(self, ts, focal_site_id, direction, max_sites, max_distance):
        self.ts = ts
        self.focal_site = ts.site(focal_site_id)
        self.direction = direction
        self.max_sites = max_sites
        self.max_distance = max_distance
        self.result = []
        self.tree = None

    def _check_site(self, site):
        assert len(site.mutations) == 1
        assert site.ancestral_state != site.mutations[0].derived_state

    def _compute_and_append(self, target_site):
        self._check_site(target_site)

        distance = abs(target_site.position - self.focal_site.position)
        if distance > self.max_distance or len(self.result) >= self.max_sites:
            return True
        r2 = _compute_r2(
            self.tree, self.ts.num_samples, self.focal_frequency, target_site
        )
        self.result.append(r2)
        return False

    def _compute_forward(self):
        done = False
        for site in self.tree.sites():
            if site.id > self.focal_site.id:
                done = self._compute_and_append(site)
                if done:
                    break
        while self.tree.next() and not done:
            for site in self.tree.sites():
                done = self._compute_and_append(site)
                if done:
                    break

    def _compute_backward(self):
        done = False
        for site in reversed(list(self.tree.sites())):
            if site.id < self.focal_site.id:
                done = self._compute_and_append(site)
                if done:
                    break
        while self.tree.prev() and not done:
            for site in reversed(list(self.tree.sites())):
                done = self._compute_and_append(site)
                if done:
                    break

    def run(self):
        self._check_site(self.focal_site)
        self.tree = self.ts.at(self.focal_site.position)
        a_samples = list(self.tree.samples(self.focal_site.mutations[0].node))
        self.focal_frequency = len(a_samples) / self.ts.num_samples

        # Now set the tracked samples on the tree. We don't have a python
        # API for doing this, so we just create a new tree.
        self.tree = self.ts.at(self.focal_site.position, tracked_samples=a_samples)
        if self.direction == 1:
            self._compute_forward()
        else:
            self._compute_backward()
        return np.array(self.result)


def ts_r2_array(ts, a, *, direction=1, max_sites=None, max_distance=None):
    max_sites = ts.num_sites if max_sites is None else max_sites
    max_distance = np.inf if max_distance is None else max_distance
    calc = LdArrayCalculator(ts, a, direction, max_sites, max_distance)
    return calc.run()


class TestLdSingleTree:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0      10
    #      | |  |
    #  pos 2 4  9
    # node 0 1  0
    @tests.cached_example
    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(2, ancestral_state="A")
        tables.sites.add_row(4, ancestral_state="A")
        tables.sites.add_row(9, ancestral_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="G")
        tables.mutations.add_row(site=1, node=3, derived_state="C")
        tables.mutations.add_row(site=2, node=0, derived_state="G")
        return tables.tree_sequence()

    @pytest.mark.parametrize(["a", "b", "expected"], [(0, 0, 1), (0, 1, 1), (0, 2, 1)])
    def test_r2(self, a, b, expected):
        ts = self.ts()
        A = get_r2_matrix(ts)
        ldc = tskit.LdCalculator(ts)
        assert ldc.r2(a, b) == pytest.approx(expected)
        assert ts_r2(ts, a, b) == pytest.approx(expected)
        assert A[a, b] == pytest.approx(expected)
        assert ldc.r2(b, a) == pytest.approx(expected)
        assert ts_r2(ts, b, a) == pytest.approx(expected)
        assert A[b, a] == pytest.approx(expected)

    @pytest.mark.parametrize("a", [0, 1, 2])
    @pytest.mark.parametrize("direction", [1, -1])
    def test_r2_array(self, a, direction):
        ts = self.ts()
        ldc = tskit.LdCalculator(ts)
        lib_a = ldc.r2_array(a, direction=direction)
        py_a = ts_r2_array(ts, a, direction=direction)
        np.testing.assert_array_almost_equal(lib_a, py_a)


class TestLdFixedSites:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0      10
    #      | |  |
    #  pos 2 4  9
    # node 0 1  0
    @tests.cached_example
    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        tables = ts.dump_tables()
        # First and last mutations are over the root
        tables.sites.add_row(2, ancestral_state="A")
        tables.sites.add_row(4, ancestral_state="A")
        tables.sites.add_row(9, ancestral_state="T")
        tables.mutations.add_row(site=0, node=4, derived_state="G")
        tables.mutations.add_row(site=1, node=3, derived_state="C")
        tables.mutations.add_row(site=2, node=4, derived_state="G")
        return tables.tree_sequence()

    def test_r2_fixed_fixed(self):
        ts = self.ts()
        A = get_r2_matrix(ts)
        ldc = tskit.LdCalculator(ts)
        assert np.isnan(ldc.r2(0, 2))
        assert np.isnan(ts_r2(ts, 0, 2))
        assert np.isnan(A[0, 2])

    def test_r2_fixed_non_fixed(self):
        ts = self.ts()
        A = get_r2_matrix(ts)
        ldc = tskit.LdCalculator(ts)
        assert np.isnan(ldc.r2(0, 1))
        assert np.isnan(ts_r2(ts, 0, 1))
        assert np.isnan(A[0, 1])

    def test_r2_non_fixed_fixed(self):
        ts = self.ts()
        A = get_r2_matrix(ts)
        ldc = tskit.LdCalculator(ts)
        assert np.isnan(ldc.r2(1, 0))
        assert np.isnan(ts_r2(ts, 1, 0))
        assert np.isnan(A[1, 0])


class BaseTestLd:
    """
    Define a set of tests for LD calculations. Subclasses should be
    concrete examples with at least two sites which implement a
    method ts() which returns the tree sequence and the full LD
    matrix.
    """

    def test_r2_all_pairs(self):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        for j in range(ts.num_sites):
            for k in range(ts.num_sites):
                r2 = A[j, k]
                assert ldc.r2(j, k) == pytest.approx(r2)
                assert ts_r2(ts, j, k) == pytest.approx(r2)

    def test_r2_array_first_site_forward(self):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        A1 = ldc.r2_array(0, direction=1)
        A2 = ts_r2_array(ts, 0, direction=1)
        np.testing.assert_array_almost_equal(A2, A[0, 1:])
        np.testing.assert_array_almost_equal(A1, A2)

    def test_r2_array_mid_forward(self):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        site = ts.num_sites // 2
        A1 = ldc.r2_array(site, direction=1)
        A2 = ts_r2_array(ts, site, direction=1)
        np.testing.assert_array_almost_equal(A2, A[site, site + 1 :])
        np.testing.assert_array_almost_equal(A1, A2)

    def test_r2_array_first_site_forward_max_sites(self):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        A1 = ldc.r2_array(0, direction=1, max_sites=2)
        A2 = ts_r2_array(ts, 0, direction=1, max_sites=2)
        np.testing.assert_array_almost_equal(A2, A[0, 1:3])
        np.testing.assert_array_almost_equal(A1, A2)

    def test_r2_array_first_site_forward_max_distance(self):
        ts, _ = self.ts()
        ldc = tskit.LdCalculator(ts)
        A1 = ldc.r2_array(0, direction=1, max_distance=3)
        A2 = ts_r2_array(ts, 0, direction=1, max_distance=3)
        np.testing.assert_array_almost_equal(A1, A2)

    def test_r2_array_last_site_backward(self):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        a = ts.num_sites - 1
        A1 = ldc.r2_array(a, direction=-1)
        A2 = ts_r2_array(ts, a, direction=-1)
        np.testing.assert_array_almost_equal(A2, A[-1, :-1][::-1])
        np.testing.assert_array_almost_equal(A1, A2)

    def test_r2_array_mid_backward(self):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        site = ts.num_sites // 2
        A1 = ldc.r2_array(site, direction=-1)
        A2 = ts_r2_array(ts, site, direction=-1)
        np.testing.assert_array_almost_equal(A2, A[site, :site][::-1])
        np.testing.assert_array_almost_equal(A1, A2)

    def test_r2_array_last_site_backward_max_sites(self):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        a = ts.num_sites - 1
        A1 = ldc.r2_array(a, direction=-1, max_sites=2)
        A2 = ts_r2_array(ts, a, direction=-1, max_sites=2)
        np.testing.assert_array_almost_equal(A2, A[-1, -3:-1][::-1])
        np.testing.assert_array_almost_equal(A1, A2)

    def test_r2_array_last_site_backward_max_distance(self):
        ts, _ = self.ts()
        ldc = tskit.LdCalculator(ts)
        a = ts.num_sites - 1
        A1 = ldc.r2_array(a, direction=-1, max_distance=3)
        A2 = ts_r2_array(ts, a, direction=-1, max_distance=3)
        np.testing.assert_array_almost_equal(A1, A2)

    @pytest.mark.parametrize("max_sites", [0, 1, 2])
    def test_r2_array_forward_max_sites_zero(self, max_sites):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        site = ts.num_sites // 2
        A1 = ldc.r2_array(site, direction=1, max_sites=max_sites)
        assert A1.shape[0] == max_sites
        A2 = ts_r2_array(ts, site, direction=1, max_sites=max_sites)
        assert A2.shape[0] == max_sites

    @pytest.mark.parametrize("max_sites", [0, 1, 2])
    def test_r2_array_backward_max_sites_zero(self, max_sites):
        ts, A = self.ts()
        ldc = tskit.LdCalculator(ts)
        site = ts.num_sites // 2
        A1 = ldc.r2_array(site, direction=-1, max_sites=max_sites)
        assert A1.shape[0] == max_sites
        A2 = ts_r2_array(ts, site, direction=-1, max_sites=max_sites)
        assert A2.shape[0] == max_sites


class TestLdOneSitePerTree(BaseTestLd):
    @tests.cached_example
    def ts(self):
        ts = msprime.sim_ancestry(
            5, sequence_length=10, recombination_rate=0.1, random_seed=1234
        )
        assert ts.num_trees > 3

        tables = ts.dump_tables()
        for tree in ts.trees():
            site = tables.sites.add_row(tree.interval[0], ancestral_state="A")
            # Put the mutation somewhere deep in the tree
            node = tree.preorder()[2]
            tables.mutations.add_row(site=site, node=node, derived_state="B")
        ts = tables.tree_sequence()
        # Return the full f2 matrix also
        return ts, get_r2_matrix(ts)


class TestLdAllSitesOneTree(BaseTestLd):
    @tests.cached_example
    def ts(self):
        ts = msprime.sim_ancestry(
            5, sequence_length=10, recombination_rate=0.1, random_seed=1234
        )
        assert ts.num_trees > 3

        tables = ts.dump_tables()
        tree = ts.at(5)
        pos = np.linspace(tree.interval[0], tree.interval[1], num=10, endpoint=False)
        for x, node in zip(pos, tree.preorder()[1:]):
            site = tables.sites.add_row(x, ancestral_state="A")
            tables.mutations.add_row(site=site, node=node, derived_state="B")
        ts = tables.tree_sequence()
        return ts, get_r2_matrix(ts)


class TestLdSitesEveryOtherTree(BaseTestLd):
    @tests.cached_example
    def ts(self):
        ts = msprime.sim_ancestry(
            5, sequence_length=20, recombination_rate=0.1, random_seed=1234
        )
        assert ts.num_trees > 5

        tables = ts.dump_tables()
        for tree in ts.trees():
            if tree.index % 2 == 0:
                pos = np.linspace(*tree.interval, num=2, endpoint=False)
                for x, node in zip(pos, tree.preorder()[1:]):
                    site = tables.sites.add_row(x, ancestral_state="A")
                    tables.mutations.add_row(site=site, node=node, derived_state="B")
        ts = tables.tree_sequence()
        return ts, get_r2_matrix(ts)


class TestLdErrors:
    def test_multi_mutations(self):
        tables = tskit.TableCollection(2)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=0, ancestral_state="A")
        tables.sites.add_row(position=1, ancestral_state="A")
        tables.mutations.add_row(site=0, node=0, derived_state="C")
        tables.mutations.add_row(site=0, node=0, derived_state="T", parent=0)
        tables.mutations.add_row(site=1, node=0, derived_state="C")
        ts = tables.tree_sequence()
        ldc = tskit.LdCalculator(ts)
        with pytest.raises(tskit.LibraryError, match="Only infinite sites mutations"):
            ldc.r2(0, 1)
        with pytest.raises(tskit.LibraryError, match="Only infinite sites mutations"):
            ldc.r2(1, 0)

    @pytest.mark.parametrize("state", ["", "A", "AAAA", "💩"])
    def test_silent_mutations(self, state):
        tables = tskit.TableCollection(2)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=0, ancestral_state=state)
        tables.sites.add_row(position=1, ancestral_state="A")
        tables.mutations.add_row(site=0, node=0, derived_state=state)
        tables.mutations.add_row(site=1, node=0, derived_state="C")
        ts = tables.tree_sequence()
        ldc = tskit.LdCalculator(ts)
        with pytest.raises(tskit.LibraryError, match="Silent mutations not supported"):
            ldc.r2(0, 1)
        with pytest.raises(tskit.LibraryError, match="Silent mutations not supported"):
            ldc.r2(1, 0)


class TestLdCalculator:
    """
    Tests for the LdCalculator class.
    """

    num_test_sites = 50

    def verify_matrix(self, ts):
        m = ts.get_num_sites()
        ldc = tskit.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        assert A.shape == (m, m)
        B = get_r2_matrix(ts)
        assert np.allclose(A, B)

        # Now look at each row in turn, and verify it's the same
        # when we use get_r2 directly.
        for j in range(m):
            a = ldc.get_r2_array(j, direction=tskit.FORWARD)
            b = A[j, j + 1 :]
            assert a.shape[0] == m - j - 1
            assert b.shape[0] == m - j - 1
            assert np.allclose(a, b)
            a = ldc.get_r2_array(j, direction=tskit.REVERSE)
            b = A[j, :j]
            assert a.shape[0] == j
            assert b.shape[0] == j
            assert np.allclose(a[::-1], b)

        # Now check every cell in the matrix in turn.
        for j in range(m):
            for k in range(m):
                assert ldc.get_r2(j, k) == pytest.approx(A[j, k])

    def verify_max_distance(self, ts):
        """
        Verifies that the max_distance parameter works as expected.
        """
        mutations = list(ts.mutations())
        ldc = tskit.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        j = len(mutations) // 2
        for k in range(j):
            x = (
                ts.site(mutations[j + k].site).position
                - ts.site(mutations[j].site).position
            )
            a = ldc.get_r2_array(j, max_distance=x)
            assert a.shape[0] == k
            assert np.allclose(A[j, j + 1 : j + 1 + k], a)
            x = (
                ts.site(mutations[j].site).position
                - ts.site(mutations[j - k].site).position
            )
            a = ldc.get_r2_array(j, max_distance=x, direction=tskit.REVERSE)
            assert a.shape[0] == k
            assert np.allclose(A[j, j - k : j], a[::-1])
        L = ts.get_sequence_length()
        m = len(mutations)
        a = ldc.get_r2_array(0, max_distance=L)
        assert a.shape[0] == m - 1
        assert np.allclose(A[0, 1:], a)
        a = ldc.get_r2_array(m - 1, max_distance=L, direction=tskit.REVERSE)
        assert a.shape[0] == m - 1
        assert np.allclose(A[m - 1, :-1], a[::-1])

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
            assert a.shape[0] == k
            assert np.allclose(A[j, j + 1 : j + 1 + k], a)
            a = ldc.get_r2_array(j, max_mutations=k, direction=tskit.REVERSE)
            assert a.shape[0] == k
            assert np.allclose(A[j, j - k : j], a[::-1])

    def test_single_tree_simulated_mutations(self):
        ts = msprime.simulate(20, mutation_rate=10, random_seed=15)
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        self.verify_matrix(ts)
        self.verify_max_distance(ts)

    def test_deprecated_get_aliases(self):
        ts = msprime.simulate(20, mutation_rate=10, random_seed=15)
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        ldc = tskit.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        B = ldc.r2_matrix()
        assert np.array_equal(A, B)
        a = ldc.get_r2_array(0)
        b = ldc.r2_array(0)
        assert np.array_equal(a, b)
        assert ldc.get_r2(0, 1) == ldc.r2(0, 1)

    def test_deprecated_max_mutations_alias(self):
        ts = msprime.simulate(2, mutation_rate=0.1, random_seed=15)
        ldc = tskit.LdCalculator(ts)
        with pytest.raises(ValueError, match="deprecated synonym"):
            ldc.r2_array(0, max_sites=1, max_mutations=1)

    def test_single_tree_regular_mutations(self):
        ts = msprime.simulate(self.num_test_sites, length=self.num_test_sites)
        ts = tsutil.insert_branch_mutations(ts)
        # We don't support back mutations, so this should fail.
        with pytest.raises(_tskit.LibraryError):
            self.verify_matrix(ts)
        with pytest.raises(_tskit.LibraryError):
            self.verify_max_distance(ts)

    def test_tree_sequence_regular_mutations(self):
        ts = msprime.simulate(
            self.num_test_sites, recombination_rate=1, length=self.num_test_sites
        )
        assert ts.get_num_trees() > 10
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
        assert ts.get_num_trees() > 10
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        self.verify_matrix(ts)
        self.verify_max_distance(ts)
        self.verify_max_mutations(ts)


def set_partitions(collection):
    """
    Returns an iterator over all partitions of the specified set.

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


class TestMeanDescendants:
    """
    Tests the TreeSequence.mean_descendants method.
    """

    def verify(self, ts, reference_sets):
        C1 = naive_mean_descendants(ts, reference_sets)
        C2 = tsutil.mean_descendants(ts, reference_sets)
        C3 = ts.mean_descendants(reference_sets)
        assert C1.shape == C2.shape
        assert np.allclose(C1, C2)
        assert np.allclose(C1, C3)
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
        assert ts.num_trees > 1
        self.verify(ts, [ts.samples(0), ts.samples(1)])

    def test_single_tree(self):
        ts = msprime.simulate(6, random_seed=1)
        S = [range(3), range(3, 6)]
        C = self.verify(ts, S)
        for j, samples in enumerate(S):
            tree = next(ts.trees(tracked_samples=samples))
            for u in tree.nodes():
                assert tree.num_tracked_samples(u) == C[u, j]

    def test_single_tree_partial_samples(self):
        ts = msprime.simulate(6, random_seed=1)
        S = [range(3), range(3, 4)]
        C = self.verify(ts, S)
        for j, samples in enumerate(S):
            tree = next(ts.trees(tracked_samples=samples))
            for u in tree.nodes():
                assert tree.num_tracked_samples(u) == C[u, j]

    def test_single_tree_all_sample_sets(self):
        ts = msprime.simulate(6, random_seed=1)
        for S in set_partitions(list(range(ts.num_samples))):
            C = self.verify(ts, S)
            for j, samples in enumerate(S):
                tree = next(ts.trees(tracked_samples=samples))
                for u in tree.nodes():
                    assert tree.num_tracked_samples(u) == C[u, j]

    def test_many_trees_all_sample_sets(self):
        ts = msprime.simulate(6, recombination_rate=2, random_seed=1)
        assert ts.num_trees > 2
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
        length = trees[0].interval.right - trees[0].interval.left
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


def parse_time_windows(ts, time_windows):
    if time_windows is None:
        time_windows = [0.0, ts.max_root_time]
    return np.array(time_windows)


def windowed_genealogical_nearest_neighbours(
    ts,
    focal,
    reference_sets,
    windows=None,
    time_windows=None,
    span_normalise=True,
    time_normalise=True,
):
    """
    genealogical_nearest_neighbours with support for span- and time-based windows
    """
    reference_set_map = np.full(ts.num_nodes, tskit.NULL, dtype=int)
    for k, reference_set in enumerate(reference_sets):
        for u in reference_set:
            if reference_set_map[u] != tskit.NULL:
                raise ValueError("Duplicate value in reference sets")
            reference_set_map[u] = k

    windows_used = windows is not None
    time_windows_used = time_windows is not None
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    time_windows = parse_time_windows(ts, time_windows)
    num_time_windows = time_windows.shape[0] - 1
    A = np.zeros((num_windows, num_time_windows, len(focal), len(reference_sets)))
    K = len(reference_sets)
    parent = np.full(ts.num_nodes, tskit.NULL, dtype=int)
    sample_count = np.zeros((ts.num_nodes, K), dtype=int)
    time = ts.tables.nodes.time
    norm = np.zeros((num_windows, num_time_windows, len(focal)))

    # Set the initial conditions.
    for j in range(K):
        sample_count[reference_sets[j], j] = 1

    window_index = 0
    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            parent[edge.child] = tskit.NULL
            v = edge.parent
            while v != tskit.NULL:
                sample_count[v] -= sample_count[edge.child]
                v = parent[v]
        for edge in edges_in:
            parent[edge.child] = edge.parent
            v = edge.parent
            while v != tskit.NULL:
                sample_count[v] += sample_count[edge.child]
                v = parent[v]

        # Update the windows
        assert window_index < num_windows
        while windows[window_index] < t_right and window_index + 1 <= num_windows:
            w_left = windows[window_index]
            w_right = windows[window_index + 1]
            left = max(t_left, w_left)
            right = min(t_right, w_right)
            span = right - left
            # Process this tree.
            for j, u in enumerate(focal):
                focal_reference_set = reference_set_map[u]
                delta = int(focal_reference_set != tskit.NULL)
                p = u
                while p != tskit.NULL:
                    total = np.sum(sample_count[p])
                    if total > delta:
                        break
                    p = parent[p]
                if p != tskit.NULL:
                    scale = span / (total - delta)
                    time_index = np.searchsorted(time_windows, time[p]) - 1
                    if 0 <= time_index < num_time_windows:
                        for k in range(len(reference_sets)):
                            n = sample_count[p, k] - int(focal_reference_set == k)
                            A[window_index, time_index, j, k] += n * scale
                        norm[window_index, time_index, j] += span
            assert span > 0
            if w_right <= t_right:
                window_index += 1
            else:
                # This interval crosses a tree boundary, so we update it again
                # in the next tree
                break

    # Reshape norm depending on normalization selected
    # Return NaN when normalisation value is 0
    if span_normalise and time_normalise:
        reshaped_norm = norm.reshape((num_windows, num_time_windows, len(focal), 1))
    elif span_normalise and not time_normalise:
        norm = np.sum(norm, axis=1)
        reshaped_norm = norm.reshape((num_windows, 1, len(focal), 1))
    elif time_normalise and not span_normalise:
        norm = np.sum(norm, axis=0)
        reshaped_norm = norm.reshape((1, num_time_windows, len(focal), 1))

    with np.errstate(invalid="ignore", divide="ignore"):
        A /= reshaped_norm
    A[np.all(A == 0, axis=3)] = np.nan

    # Remove dimension for windows and/or time_windows if parameter is None
    if not windows_used and time_windows_used:
        A = A.reshape((num_time_windows, len(focal), len(reference_sets)))
    elif not time_windows_used and windows_used:
        A = A.reshape((num_windows, len(focal), len(reference_sets)))
    elif not windows_used and not time_windows_used:
        A = A.reshape((len(focal), len(reference_sets)))
    return A


class TestGenealogicalNearestNeighbours:
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
        A5 = windowed_genealogical_nearest_neighbours(ts, focal, reference_sets)
        assert np.array_equal(A3, A4)
        assert A1.shape == A2.shape
        assert A1.shape == A3.shape
        assert np.allclose(A1, A2)
        assert np.allclose(A1, A3)
        mask = ~np.isnan(A5)
        assert np.sum(mask) > 0 or ts.num_edges == 0
        assert np.allclose(A1[mask], A5[mask])
        assert np.allclose(A5[mask], A2[mask])
        assert np.allclose(A5[mask], A3[mask])

        if ts.num_edges > 0 and all(ts.node(u).is_sample() for u in focal):
            # When the focal nodes are samples, we can assert some stronger properties.
            assert np.allclose(np.sum(A1, axis=1), 1)
            assert np.allclose(np.sum(A5, axis=1), 1)
        return A1

    def test_simple_example_all_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0])
        assert list(A[0]) == [1, 0]
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [4])
        assert list(A[0]) == [1, 0]
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [2])
        assert list(A[0]) == [0, 1]
        A = self.verify(ts, [[0, 2], [1, 3, 4]], [0])
        assert list(A[0]) == [0, 1]
        A = self.verify(ts, [[0, 2], [1, 3, 4]], [4])
        assert list(A[0]) == [0.5, 0.5]

    def test_simple_example_missing_samples(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        A = self.verify(ts, [[0, 1], [2, 4]], [3])
        assert list(A[0]) == [0, 1]
        A = self.verify(ts, [[0, 1], [2, 4]], [2])
        assert np.allclose(A[0], [2 / 3, 1 / 3])

    def test_simple_example_internal_focal_node(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        focal = [7]  # An internal node
        reference_sets = [[4, 0, 1], [2, 3]]
        GNN = naive_genealogical_nearest_neighbours(ts, focal, reference_sets)
        assert np.allclose(GNN[0], np.array([1.0, 0.0]))
        GNN = tsutil.genealogical_nearest_neighbours(ts, focal, reference_sets)
        assert np.allclose(GNN[0], np.array([1.0, 0.0]))
        GNN = ts.genealogical_nearest_neighbours(focal, reference_sets)
        assert np.allclose(GNN[0], np.array([1.0, 0.0]))
        focal = [8]  # The root
        GNN = naive_genealogical_nearest_neighbours(ts, focal, reference_sets)
        assert np.allclose(GNN[0], np.array([0.6, 0.4]))
        GNN = tsutil.genealogical_nearest_neighbours(ts, focal, reference_sets)
        assert np.allclose(GNN[0], np.array([0.6, 0.4]))
        GNN = ts.genealogical_nearest_neighbours(focal, reference_sets)
        assert np.allclose(GNN[0], np.array([0.6, 0.4]))

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
        assert ts.num_trees > 1
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
        assert ts.num_trees > 2
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


class TestWindowedGenealogicalNearestNeighbours(TestGenealogicalNearestNeighbours):
    """
    Tests the TreeSequence.genealogical_nearest_neighbours method.
    """

    #               .    5
    #               .   / \
    #        4      .  |   4
    #       / \     .  |   |\
    #      3   \    .  |   | \
    #     / \   \   .  |   |  \
    #   [0] [1] [2] . [0] [1] [2]
    #
    two_tree_nodes = """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    """
    two_tree_edges = """\
    left    right   parent  child
    0       0.2     3       0,1
    0       1       4       2
    0       0.2     4       3
    0.2     1       4       1
    0.2     1       5       0,4
    """

    def get_two_tree_ts(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.two_tree_nodes),
            edges=io.StringIO(self.two_tree_edges),
            strict=False,
        )
        return ts

    def verify(self, ts, reference_sets, focal=None, windows=None, time_windows=None):
        if focal is None:
            focal = [u for refset in reference_sets for u in refset]
        gnn = windowed_genealogical_nearest_neighbours(
            ts, focal, reference_sets, windows, time_windows
        )
        if windows is not None:
            windows_len = len(windows) - 1
        if time_windows is not None:
            time_windows_len = len(time_windows) - 1
        if windows is None and time_windows is None:
            assert np.array_equal(gnn.shape, [len(focal), len(reference_sets)])
        elif windows is None and time_windows is not None:
            assert np.array_equal(
                gnn.shape, [time_windows_len, len(focal), len(reference_sets)]
            )
        elif windows is not None and time_windows is None:
            assert np.array_equal(
                gnn.shape, [windows_len, len(focal), len(reference_sets)]
            )
        else:
            assert np.array_equal(
                gnn.shape,
                [windows_len, time_windows_len, len(focal), len(reference_sets)],
            )

        return gnn

    def test_one_tree_windows(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], [0, 1])
        assert np.allclose(A, [[[[1, 0]]]])
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], [0, 0.5, 1])
        assert np.allclose(A, [[[[1.0, 0.0]]], [[[1.0, 0.0]]]])
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], [0, 0.5, 0.6, 1])
        assert np.allclose(A, [[[[1.0, 0.0]]], [[[1.0, 0.0]]], [[[1.0, 0.0]]]])

    def test_two_tree_windows(self):
        ts = self.get_two_tree_ts()
        A = self.verify(ts, [[0, 1], [2]], [0], [0, 1])
        assert np.allclose(A, [[[0.6, 0.4]]])
        A = self.verify(ts, [[0, 1], [2]], [0], [0, 0.2, 1])
        assert np.allclose(A, [[[1.0, 0.0]], [[0.5, 0.5]]])
        A = self.verify(ts, [[0, 1], [2]], [0], [0, 0.2, 0.5, 1])
        assert np.allclose(A, [[[1.0, 0.0]], [[0.5, 0.5]], [[0.5, 0.5]]])

    def test_one_tree_time_windows(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], None, [0, ts.max_root_time])
        assert np.allclose(A, [[[1, 0]]])
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], None, [1, 2])
        assert np.allclose(A, [[[np.nan, np.nan]]], equal_nan=True)
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], None, [0, 0.1])
        assert np.allclose(A, [[[np.nan, np.nan]]], equal_nan=True)

    def test_two_tree_time_windows(self):
        ts = self.get_two_tree_ts()
        A = self.verify(ts, [[0, 1], [2]], [0], None, [0, ts.max_root_time])
        assert np.allclose(A, [[[0.6, 0.4]]])
        A = self.verify(ts, [[0, 1], [2]], [0], None, [0, 1.1, ts.max_root_time])
        assert np.allclose(A, [[[1.0, 0.0]], [[0.5, 0.5]]])
        A = self.verify(ts, [[0, 1], [2]], [0], None, [0, 0.5, 1])
        assert np.allclose(A, [[[np.nan, np.nan]], [[1.0, 0.0]]], equal_nan=True)
        A = self.verify(ts, [[0, 1], [2]], [0], None, [1, ts.max_root_time, 10])
        assert np.allclose(A, [[[0.5, 0.5]], [[np.nan, np.nan]]], equal_nan=True)

    def test_one_tree_windows_time_windows(self):
        ts = tskit.load_text(
            nodes=io.StringIO(self.small_tree_ex_nodes),
            edges=io.StringIO(self.small_tree_ex_edges),
            strict=False,
        )
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], [0, 1], [0, ts.max_root_time])
        assert np.allclose(A, [[[[1, 0]]]])
        A = self.verify(
            ts, [[0, 1], [2, 3, 4]], [0], [0, 0.2, 1], [0, 1.1, ts.max_root_time]
        )
        assert np.allclose(
            A,
            [
                [[[1.0, 0.0]], [[np.nan, np.nan]]],
                [[[1.0, 0.0]], [[np.nan, np.nan]]],
            ],
            equal_nan=True,
        )
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], [0, 0.2], [0, 0.5, 1])
        assert np.allclose(A, [[[[1.0, 0.0]], [[np.nan, np.nan]]]], equal_nan=True)
        A = self.verify(
            ts, [[0, 1], [2, 3, 4]], [0], [0, 0.2, 1, 1.5], [0, ts.max_root_time, 10]
        )
        assert np.allclose(
            A,
            [
                [[[1.0, 0.0]], [[np.nan, np.nan]]],
                [[[1.0, 0.0]], [[np.nan, np.nan]]],
                [[[np.nan, np.nan]], [[np.nan, np.nan]]],
            ],
            equal_nan=True,
        )

    def test_two_tree_windows_time_windows(self):
        ts = self.get_two_tree_ts()
        A = self.verify(ts, [[0, 1], [2]], [0], [0, 1], [0, ts.max_root_time])
        assert np.allclose(A, [[[[0.6, 0.4]]]])
        A = self.verify(ts, [[0, 1], [2]], [0], [0, 0.2, 1], [0, 1.1, ts.max_root_time])
        assert np.allclose(
            A,
            [
                [[[1.0, 0.0]], [[np.nan, np.nan]]],
                [[[np.nan, np.nan]], [[0.5, 0.5]]],
            ],
            equal_nan=True,
        )
        A = self.verify(ts, [[0, 1], [2, 3, 4]], [0], [0, 0.2, 1], [0, 0.5, 1])
        assert np.allclose(
            A,
            [
                [[[np.nan, np.nan]], [[0.5, 0.5]]],
                [[[np.nan, np.nan]], [[np.nan, np.nan]]],
            ],
            equal_nan=True,
        )

    def test_span_normalise(self):
        ts = self.get_two_tree_ts()
        sample_sets = [[0, 1], [2]]
        focal = [0]
        np.random.seed(5)
        windows = ts.sequence_length * np.array([0.2, 0.4, 0.6, 0.8, 1])
        windows.sort()
        windows[0] = 0.0
        windows[-1] = ts.sequence_length

        result1 = windowed_genealogical_nearest_neighbours(
            ts, focal, sample_sets, windows
        )
        result2 = windowed_genealogical_nearest_neighbours(
            ts, focal, sample_sets, windows, span_normalise=True
        )
        result3 = windowed_genealogical_nearest_neighbours(
            ts, focal, sample_sets, windows, span_normalise=False
        )
        denom = np.diff(windows)[:, np.newaxis, np.newaxis]

        # Test the dimensions are correct
        assert np.array_equal(result1.shape, result2.shape)
        assert np.array_equal(result2.shape, result3.shape)

        # Test normalisation is correct
        assert np.allclose(result1, result2)
        assert np.allclose(result1, result3 / denom)

        # If span_normalised, then sum over all reference sets should be 1
        assert np.allclose(np.sum(result1, axis=2), 1)
        assert np.allclose(np.sum(result2, axis=2), 1)
        # If not span_normalised, then sum over all value is 1
        assert np.allclose(result3.sum(), 1)

    def test_time_normalise(self):
        """
        Testing time_normalise is trickier than span_normalise, as the norm
        depends on the span of the nearest neighbours found in each time grid.
        In this small example, we check which grid nodes 3 and 5 fall in, and use their
        spans to check the time_normalisation.
        """
        ts = self.get_two_tree_ts()
        sample_sets = [[0, 1], [2]]
        focal = [0]
        oldest_node = ts.max_root_time
        time_windows = oldest_node * np.array([0.2, 0.4, 0.6, 0.8, 1])
        time_windows.sort()
        time_windows[0] = 0.0
        time_windows[-1] = oldest_node

        # Determine output_dim of the function
        result1 = windowed_genealogical_nearest_neighbours(
            ts, focal, sample_sets, windows=None, time_windows=time_windows
        )
        result2 = windowed_genealogical_nearest_neighbours(
            ts,
            focal,
            sample_sets,
            windows=None,
            time_windows=time_windows,
            time_normalise=True,
        )
        result3 = windowed_genealogical_nearest_neighbours(
            ts,
            focal,
            sample_sets,
            windows=None,
            time_windows=time_windows,
            time_normalise=False,
        )
        denom = np.zeros(len(time_windows) - 1)
        time_index_3 = np.searchsorted(time_windows, ts.tables.nodes.time[3]) - 1
        time_index_5 = np.searchsorted(time_windows, ts.tables.nodes.time[5]) - 1
        denom[time_index_3] += 0.2
        denom[time_index_5] += 0.8

        # Avoid division by zero
        denom[denom == 0] = 1
        denom = denom[:, np.newaxis, np.newaxis]

        # Test the dimensions are correct
        assert np.array_equal(result1.shape, result2.shape)
        assert np.array_equal(result2.shape, result3.shape)

        # Test normalisation is correct
        assert np.allclose(result1, result2, equal_nan=True)
        assert np.allclose(result1, result3 / denom, equal_nan=True)

        # If time_normalised, then sum over all reference sets should be 1
        # Mask out time intervals that sum to 0
        result1_dim_sum = np.sum(result1, axis=2)
        mask = ~(np.isnan(result1_dim_sum))
        assert np.allclose(
            result1_dim_sum[mask],
            np.ones((len(result1_dim_sum), len(focal)))[mask],
            equal_nan=True,
        )
        result2_dim_sum = np.sum(result2, axis=2)
        mask = ~(np.isnan(result2_dim_sum))
        assert np.allclose(
            result2_dim_sum[mask],
            np.ones((len(result2_dim_sum), len(focal)))[mask],
            equal_nan=True,
        )
        # If not span_normalised, then sum over all value is 1
        assert np.allclose(np.nansum(result3), 1)


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
            L[trees[0].index] = trees[0].interval.right - trees[0].interval.left
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
                for k, _reference_set in enumerate(reference_sets):
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
            assert lefts[tree.index] == tree.interval.left
            assert rights[tree.index] == tree.interval.right

        for j, u in enumerate(focal):
            T, L = exact_genealogical_nearest_neighbours(ts, u, reference_sets)
            assert np.allclose(G[j], T.T)
            # Ignore the cases where the node has no GNNs
            if np.sum(L) > 0:
                mean = np.sum(T * L, axis=1) / np.sum(L)
                assert np.allclose(mean, A[j])
        return A
