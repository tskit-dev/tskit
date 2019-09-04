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
Test cases for generalized statistic computation.
"""
import io
import unittest
import random
import collections
import itertools
import functools
import contextlib

import numpy as np
import numpy.testing as nt

import msprime

import tskit
import tskit.exceptions as exceptions
import tests.tsutil as tsutil
import tests.test_wright_fisher as wf

np.random.seed(5)


def subset_combos(*args, p=0.5, min_tests=3):
    # We have too many tests, combinatorially; so we will run a random subset
    # of them, using this function, below. If we don't set a seed, a different
    # random set is run each time. Ensures that at least min_tests are run.
    # Uncomment this line to run all tests (takes about an hour):
    # p = 1.0
    num_tests = 0
    skipped_tests = []
    # total_tests = 0
    for x in itertools.product(*args):
        # total_tests = total_tests + 1
        if np.random.uniform() < p:
            num_tests += num_tests + 1
            yield x
        elif len(skipped_tests) < min_tests:
            skipped_tests.append(x)
        elif np.random.uniform() < 0.1:
            skipped_tests[np.random.randint(min_tests)] = x
    while num_tests < min_tests:
        yield skipped_tests.pop()
        num_tests = num_tests + 1
    # print("tests", num_tests)
    assert num_tests >= min_tests


def path_length(tr, x, y):
    L = 0
    if x >= 0 and y >= 0:
        mrca = tr.mrca(x, y)
    else:
        mrca = -1
    for u in x, y:
        while u != mrca:
            L += tr.branch_length(u)
            u = tr.parent(u)
    return L


@contextlib.contextmanager
def suppress_division_by_zero_warning():
    with np.errstate(invalid='ignore', divide='ignore'):
        yield


##############################
# Branch general stat algorithms
##############################

def windowed_tree_stat(ts, stat, windows, span_normalise=True):
    shape = list(stat.shape)
    shape[0] = len(windows) - 1
    A = np.zeros(shape)

    tree_breakpoints = np.array(list(ts.breakpoints()))
    tree_index = 0
    for j in range(len(windows) - 1):
        w_left = windows[j]
        w_right = windows[j + 1]
        while True:
            t_left = tree_breakpoints[tree_index]
            t_right = tree_breakpoints[tree_index + 1]
            left = max(t_left, w_left)
            right = min(t_right, w_right)
            weight = max(0.0, (right - left) / (t_right - t_left))
            A[j] += stat[tree_index] * weight
            assert left != right
            if t_right <= w_right:
                tree_index += 1
                # TODO This is inelegant - should include this in the case below
                if t_right == w_right:
                    break
            else:
                break
    if span_normalise:
        # re-normalize by window lengths
        window_lengths = np.diff(windows)
        for j in range(len(windows) - 1):
            A[j] /= window_lengths[j]
    return A


def naive_branch_general_stat(ts, w, f, windows=None, polarised=False,
                              span_normalise=True):
    if windows is None:
        windows = [0.0, ts.sequence_length]
    n, k = w.shape
    # hack to determine m
    m = len(f(w[0]))
    total = np.sum(w, axis=0)

    sigma = np.zeros((ts.num_trees, m))
    for tree in ts.trees():
        x = np.zeros((ts.num_nodes, k))
        x[ts.samples()] = w
        for u in tree.nodes(order="postorder"):
            for v in tree.children(u):
                x[u] += x[v]
        if polarised:
            s = sum(tree.branch_length(u) * f(x[u]) for u in tree.nodes())
        else:
            s = sum(
                tree.branch_length(u) * (f(x[u]) + f(total - x[u]))
                for u in tree.nodes())
        sigma[tree.index] = s * tree.span
    if isinstance(windows, str) and windows == "trees":
        # need to average across the windows
        if span_normalise:
            for j, tree in enumerate(ts.trees()):
                sigma[j] /= tree.span
        return sigma
    else:
        return windowed_tree_stat(ts, sigma, windows, span_normalise=span_normalise)


def branch_general_stat(ts, sample_weights, summary_func, windows=None,
                        polarised=False, span_normalise=True):
    """
    Efficient implementation of the algorithm used as the basis for the
    underlying C version.
    """
    n, state_dim = sample_weights.shape
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1

    # Determine result_dim
    result_dim = len(summary_func(sample_weights[0]))
    result = np.zeros((num_windows, result_dim))
    state = np.zeros((ts.num_nodes, state_dim))
    state[ts.samples()] = sample_weights
    total_weight = np.sum(sample_weights, axis=0)

    time = ts.tables.nodes.time
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    branch_length = np.zeros(ts.num_nodes)
    # The value of summary_func(u) for every node.
    summary = np.zeros((ts.num_nodes, result_dim))
    # The result for the current tree *not* weighted by span.
    running_sum = np.zeros(result_dim)

    def polarised_summary(u):
        s = summary_func(state[u])
        if not polarised:
            s += summary_func(total_weight - state[u])
        return s

    for u in ts.samples():
        summary[u] = polarised_summary(u)

    window_index = 0
    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            u = edge.child
            running_sum -= branch_length[u] * summary[u]
            u = edge.parent
            while u != -1:
                running_sum -= branch_length[u] * summary[u]
                state[u] -= state[edge.child]
                summary[u] = polarised_summary(u)
                running_sum += branch_length[u] * summary[u]
                u = parent[u]
            parent[edge.child] = -1
            branch_length[edge.child] = 0

        for edge in edges_in:
            parent[edge.child] = edge.parent
            branch_length[edge.child] = time[edge.parent] - time[edge.child]
            u = edge.child
            running_sum += branch_length[u] * summary[u]
            u = edge.parent
            while u != -1:
                running_sum -= branch_length[u] * summary[u]
                state[u] += state[edge.child]
                summary[u] = polarised_summary(u)
                running_sum += branch_length[u] * summary[u]
                u = parent[u]

        # Update the windows
        assert window_index < num_windows
        while windows[window_index] < t_right:
            w_left = windows[window_index]
            w_right = windows[window_index + 1]
            left = max(t_left, w_left)
            right = min(t_right, w_right)
            span = right - left
            assert span > 0
            result[window_index] += running_sum * span
            if w_right <= t_right:
                window_index += 1
            else:
                # This interval crosses a tree boundary, so we update it again in the
                # for the next tree
                break

    # print("window_index:", window_index, windows.shape)
    assert window_index == windows.shape[0] - 1
    if span_normalise:
        for j in range(num_windows):
            result[j] /= windows[j + 1] - windows[j]
    return result


##############################
# Site general stat algorithms
##############################

def windowed_sitewise_stat(ts, sigma, windows, span_normalise=True):
    M = sigma.shape[1]
    A = np.zeros((len(windows) - 1, M))
    window = 0
    for site in ts.sites():
        while windows[window + 1] <= site.position:
            window += 1
        assert windows[window] <= site.position < windows[window + 1]
        A[window] += sigma[site.id]
    if span_normalise:
        diff = np.zeros((A.shape[0], 1))
        diff[:, 0] = np.diff(windows).T
        A /= diff
    return A


def naive_site_general_stat(ts, W, f, windows=None, polarised=False,
                            span_normalise=True):
    n, K = W.shape
    # Hack to determine M
    M = len(f(W[0]))
    sigma = np.zeros((ts.num_sites, M))
    for tree in ts.trees():
        X = np.zeros((ts.num_nodes, K))
        X[ts.samples()] = W
        for u in tree.nodes(order="postorder"):
            for v in tree.children(u):
                X[u] += X[v]
        for site in tree.sites():
            state_map = collections.defaultdict(functools.partial(np.zeros, K))
            state_map[site.ancestral_state] = sum(X[root] for root in tree.roots)
            for mutation in site.mutations:
                state_map[mutation.derived_state] += X[mutation.node]
                if mutation.parent != tskit.NULL:
                    parent = site.mutations[mutation.parent - site.mutations[0].id]
                    state_map[parent.derived_state] -= X[mutation.node]
                else:
                    state_map[site.ancestral_state] -= X[mutation.node]
            if polarised:
                del state_map[site.ancestral_state]
            sigma[site.id] += sum(map(f, state_map.values()))
    return windowed_sitewise_stat(
        ts, sigma, ts.parse_windows(windows),
        span_normalise=span_normalise)


def site_general_stat(ts, sample_weights, summary_func, windows=None, polarised=False,
                      span_normalise=True):
    """
    Problem: 'sites' is different that the other windowing options
    because if we output by site we don't want to normalize by length of the window.
    Solution: we pass an argument "normalize", to the windowing function.
    """
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    n, state_dim = sample_weights.shape
    # Determine result_dim
    result_dim, = summary_func(sample_weights[0]).shape
    result = np.zeros((num_windows, result_dim))
    state = np.zeros((ts.num_nodes, state_dim))
    state[ts.samples()] = sample_weights
    total_weight = np.sum(sample_weights, axis=0)

    site_index = 0
    mutation_index = 0
    window_index = 0
    sites = ts.tables.sites
    mutations = ts.tables.mutations
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    for (left, right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            u = edge.parent
            while u != -1:
                state[u] -= state[edge.child]
                u = parent[u]
            parent[edge.child] = -1
        for edge in edges_in:
            parent[edge.child] = edge.parent
            u = edge.parent
            while u != -1:
                state[u] += state[edge.child]
                u = parent[u]
        while site_index < len(sites) and sites.position[site_index] < right:
            assert left <= sites.position[site_index]
            ancestral_state = sites[site_index].ancestral_state
            allele_state = collections.defaultdict(
                functools.partial(np.zeros, state_dim))
            allele_state[ancestral_state][:] = total_weight
            while (
                    mutation_index < len(mutations)
                    and mutations[mutation_index].site == site_index):
                mutation = mutations[mutation_index]
                allele_state[mutation.derived_state] += state[mutation.node]
                if mutation.parent != -1:
                    parent_allele = mutations[mutation.parent].derived_state
                    allele_state[parent_allele] -= state[mutation.node]
                else:
                    allele_state[ancestral_state] -= state[mutation.node]
                mutation_index += 1
            if polarised:
                del allele_state[ancestral_state]

            pos = sites.position[site_index]
            while windows[window_index + 1] <= pos:
                window_index += 1
            assert windows[window_index] <= pos < windows[window_index + 1]
            site_result = result[window_index]

            for allele, value in allele_state.items():
                site_result += summary_func(value)
            site_index += 1
    if span_normalise:
        for j in range(num_windows):
            span = windows[j + 1] - windows[j]
            result[j] /= span
    return result


##############################
# Node general stat algorithms
##############################


def naive_node_general_stat(ts, W, f, windows=None, polarised=False,
                            span_normalise=True):
    windows = ts.parse_windows(windows)
    n, K = W.shape
    M = f(W[0]).shape[0]
    total = np.sum(W, axis=0)
    sigma = np.zeros((ts.num_trees, ts.num_nodes, M))
    for tree in ts.trees():
        X = np.zeros((ts.num_nodes, K))
        X[ts.samples()] = W
        for u in tree.nodes(order="postorder"):
            for v in tree.children(u):
                X[u] += X[v]
        s = np.zeros((ts.num_nodes, M))
        for u in range(ts.num_nodes):
            s[u] = f(X[u])
            if not polarised:
                s[u] += f(total - X[u])
        sigma[tree.index] = s * tree.span
    return windowed_tree_stat(ts, sigma, windows, span_normalise=span_normalise)


def node_general_stat(ts, sample_weights, summary_func, windows=None, polarised=False,
                      span_normalise=True):
    """
    Efficient implementation of the algorithm used as the basis for the
    underlying C version.
    """
    n, state_dim = sample_weights.shape
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    result_dim = summary_func(sample_weights[0]).shape[0]
    result = np.zeros((num_windows, ts.num_nodes, result_dim))
    state = np.zeros((ts.num_nodes, state_dim))
    state[ts.samples()] = sample_weights
    total_weight = np.sum(sample_weights, axis=0)

    def node_summary(u):
        s = summary_func(state[u])
        if not polarised:
            s += summary_func(total_weight - state[u])
        return s

    window_index = 0
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    # contains summary_func(state[u]) for each node
    current_values = np.zeros((ts.num_nodes, result_dim))
    for u in range(ts.num_nodes):
        current_values[u] = node_summary(u)
    # contains the location of the last time we updated the output for a node.
    last_update = np.zeros((ts.num_nodes, 1))
    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():

        for edge in edges_out:
            u = edge.child
            v = edge.parent
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] -= state[u]
                current_values[v] = node_summary(v)
                v = parent[v]
            parent[u] = -1

        for edge in edges_in:
            u = edge.child
            v = edge.parent
            parent[u] = v
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] += state[u]
                current_values[v] = node_summary(v)
                v = parent[v]

        # Update the windows
        while window_index < num_windows and windows[window_index + 1] <= t_right:
            w_right = windows[window_index + 1]
            # Flush the contribution of all nodes to the current window.
            for u in range(ts.num_nodes):
                result[window_index, u] += (w_right - last_update[u]) * current_values[u]
                last_update[u] = w_right
            window_index += 1

    assert window_index == windows.shape[0] - 1
    if span_normalise:
        for j in range(num_windows):
            result[j] /= windows[j + 1] - windows[j]
    return result


def general_stat(
        ts, sample_weights, summary_func, windows=None, polarised=False,
        mode="site", span_normalise=True):
    """
    General iterface for algorithms above. Directly corresponds to the interface
    for TreeSequence.general_stat.
    """
    method_map = {
        "site": site_general_stat,
        "node": node_general_stat,
        "branch": branch_general_stat}
    return method_map[mode](
        ts, sample_weights, summary_func, windows=windows, polarised=polarised,
        span_normalise=span_normalise)


def upper_tri_to_matrix(x):
    """
    Given x, a vector of entries of the upper triangle of a matrix
    in row-major order, including the diagonal, return the corresponding matrix.
    """
    # n^2 + n = 2 u => n = (-1 + sqrt(1 + 8*u))/2
    n = int((np.sqrt(1 + 8 * len(x)) - 1)/2.0)
    out = np.ones((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            out[i, j] = out[j, i] = x[k]
            k += 1
    return out


##################################
# Test cases
##################################


class StatsTestCase(unittest.TestCase):
    """
    Provides convenience functions.
    """
    def assertListAlmostEqual(self, x, y):
        self.assertEqual(len(x), len(y))
        for a, b in zip(x, y):
            self.assertAlmostEqual(a, b)

    def assertArrayEqual(self, x, y):
        nt.assert_equal(x, y)

    def assertArrayAlmostEqual(self, x, y, atol=1e-6, rtol=1e-7):
        nt.assert_allclose(x, y, atol=atol, rtol=rtol)

    def identity_f(self, ts):
        return lambda x: x * (x < ts.num_samples)

    def cumsum_f(self, ts):
        return lambda x: np.cumsum(x) * (x < ts.num_samples)

    def sum_f(self, ts, k=1):
        return lambda x: np.array([sum(x) * (sum(x) < 2 * ts.num_samples)] * k)


class TopologyExamplesMixin(object):
    """
    Defines a set of test cases on different example tree sequence topologies.
    Derived classes need to define a 'verify' function which will perform the
    actual tests.
    """
    def test_single_tree(self):
        ts = msprime.simulate(6, random_seed=1)
        self.verify(ts)

    def test_single_tree_sequence_length(self):
        ts = msprime.simulate(6, length=10, random_seed=1)
        self.verify(ts)

    def test_single_tree_multiple_roots(self):
        ts = msprime.simulate(8, random_seed=1)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        self.verify(ts)

    def test_many_trees(self):
        ts = msprime.simulate(6, recombination_rate=2, random_seed=1)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts)

    def test_many_trees_sequence_length(self):
        for L in [0.5, 3.3333]:
            ts = msprime.simulate(6, length=L, recombination_rate=2, random_seed=1)
            self.verify(ts)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            4, 5, seed=1, deep_history=True, initial_generation_samples=False,
            num_loci=5)
        tables.sort()
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            6, 5, seed=3, deep_history=True, initial_generation_samples=True,
            num_loci=2)
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_wright_fisher_initial_generation_no_deep_history(self):
        tables = wf.wf_sim(
            6, 15, seed=202, deep_history=False, initial_generation_samples=True,
            num_loci=5)
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_wright_fisher_unsimplified_multiple_roots(self):
        tables = wf.wf_sim(
            6, 5, seed=1, deep_history=False, initial_generation_samples=False,
            num_loci=4)
        tables.sort()
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            5, 8, seed=1, deep_history=True, initial_generation_samples=False,
            num_loci=5)
        tables.sort()
        ts = tables.tree_sequence().simplify()
        self.verify(ts)

    def test_wright_fisher_simplified_multiple_roots(self):
        tables = wf.wf_sim(
            6, 8, seed=1, deep_history=False, initial_generation_samples=False,
            num_loci=3)
        tables.sort()
        ts = tables.tree_sequence().simplify()
        self.verify(ts)

    def test_empty_ts(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(1, 0)
        tables.nodes.add_row(1, 0)
        tables.nodes.add_row(1, 0)
        tables.nodes.add_row(1, 0)
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_non_sample_ancestry(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(1, 0)
        tables.nodes.add_row(1, 0)
        tables.nodes.add_row(0, 1)
        tables.nodes.add_row(0, 0)  # 3 is a leaf but not a sample.
        # Make sure we have 4 samples for the tests.
        tables.nodes.add_row(1, 1)
        tables.nodes.add_row(1, 1)
        tables.edges.add_row(0, 1, 2, 0)
        tables.edges.add_row(0, 1, 2, 1)
        tables.edges.add_row(0, 1, 4, 3)
        ts = tables.tree_sequence()
        self.verify(ts)


class MutatedTopologyExamplesMixin(object):
    """
    Defines a set of test cases on different example tree sequence topologies.
    Derived classes need to define a 'verify' function which will perform the
    actual tests.
    """
    def test_single_tree_no_sites(self):
        ts = msprime.simulate(6, random_seed=1)
        self.assertEqual(ts.num_sites, 0)
        self.verify(ts)

    def test_ghost_allele(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=0, time=1)
        tables.edges.add_row(0, 1, 2, 0)
        tables.edges.add_row(0, 1, 2, 1)
        tables.sites.add_row(position=0.5, ancestral_state="A")
        # Make sure there's 4 samples
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        # The ghost mutation that's never seen in the genotypes
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="G", parent=0)
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_ghost_allele_all_ancestral(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=0, time=1)
        # Make sure there's 4 samples
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.edges.add_row(0, 1, 2, 0)
        tables.edges.add_row(0, 1, 2, 1)
        tables.sites.add_row(position=0.5, ancestral_state="A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        # Mutate back to the ancestral state so that all genotypes are zero
        tables.mutations.add_row(site=0, node=0, derived_state="A", parent=0)
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_non_sample_ancestry(self):
        # 2.00┊       5   ┊
        #     ┊    ┏━━┻━┓ ┊
        # 1.00┊    4    ┃ ┊
        #     ┊ ┏━┳┻┳━┓ ┃ ┊
        # 0.00┊ 0 1 2 3 6 ┊
        #    0.00        1.00
        tables = tskit.TableCollection(1)
        # Four sample nodes
        for j in range(4):
            tables.nodes.add_row(flags=1, time=0)
            tables.edges.add_row(0, 1, 4, j)
        # Their MRCA, 4, joins to older ancestor 5
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=0, time=2)
        tables.edges.add_row(0, 1, 5, 4)
        # Which has non-sample leaf at time 0
        tables.nodes.add_row(flags=0, time=0)
        tables.edges.add_row(0, 1, 5, 6)
        # Two sites with mutations. One over the MRCA of the
        # samples so it's fixed at 1 and one over the non sample
        # leaf so that samples are fixed at zero.
        tables.sites.add_row(position=0.25, ancestral_state="0")
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.mutations.add_row(site=0, node=4, derived_state="1")
        tables.mutations.add_row(site=1, node=6, derived_state="1")
        ts = tables.tree_sequence()
        self.verify(ts)

    def test_single_tree_infinite_sites(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_single_tree_sites_no_mutations(self):
        ts = msprime.simulate(6, random_seed=1)
        tables = ts.dump_tables()
        tables.sites.add_row(0.1, "a")
        tables.sites.add_row(0.2, "aaa")
        self.verify(tables.tree_sequence())

    def test_single_tree_jukes_cantor(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        self.verify(ts)

    def test_single_tree_multichar_mutations(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.insert_multichar_mutations(ts)
        self.verify(ts)

    def test_many_trees_infinite_sites(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=2, random_seed=1)
        self.assertGreater(ts.num_sites, 0)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts)

    def test_many_trees_sequence_length_infinite_sites(self):
        for L in [0.5, 1.5, 3.3333]:
            ts = msprime.simulate(
                6, length=L, recombination_rate=2, mutation_rate=1, random_seed=1)
            self.verify(ts)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            4, 5, seed=1, deep_history=True, initial_generation_samples=False,
            num_loci=10)
        tables.sort()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.05, random_seed=234)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            6, 5, seed=3, deep_history=True, initial_generation_samples=True,
            num_loci=2)
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.08, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_wright_fisher_initial_generation_no_deep_history(self):
        tables = wf.wf_sim(
            7, 15, seed=202, deep_history=False, initial_generation_samples=True,
            num_loci=5)
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.01, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_wright_fisher_unsimplified_multiple_roots(self):
        tables = wf.wf_sim(
            8, 15, seed=1, deep_history=False, initial_generation_samples=False,
            num_loci=20)
        tables.sort()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.006, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            9, 10, seed=1, deep_history=True, initial_generation_samples=False,
            num_loci=5)
        tables.sort()
        ts = tables.tree_sequence().simplify()
        ts = msprime.mutate(ts, rate=0.01, random_seed=1234)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_empty_ts(self):
        tables = tskit.TableCollection(1.0)
        for _ in range(10):
            tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        ts = tables.tree_sequence()
        self.verify(ts)


def example_sample_sets(ts, min_size=1):
    """
    Generate a series of example sample sets from the specfied tree sequence. The
    number of sample sets returned in each example must be at least min_size
    """
    samples = ts.samples()
    np.random.shuffle(samples)
    splits = np.array_split(samples, min_size)
    yield splits
    yield [[s] for s in samples]
    if min_size == 1:
        yield [samples[:1]]
    if ts.num_samples > 2 and min_size <= 2:
        yield [samples[:2], samples[2:]]
    if ts.num_samples > 7 and min_size <= 4:
        yield [samples[:2], samples[2:4], samples[4:6], samples[6:]]


def example_sample_set_index_pairs(sample_sets):
    assert len(sample_sets) >= 2
    yield [(0, 1)]
    yield [(1, 0), (0, 1)]
    if len(sample_sets) > 2:
        yield [(0, 1), (1, 2), (0, 2)]


def example_sample_set_index_triples(sample_sets):
    assert len(sample_sets) >= 3
    yield [(0, 1, 2)]
    yield [(0, 2, 1), (2, 1, 0)]
    if len(sample_sets) > 3:
        yield [(3, 0, 1), (0, 2, 3), (1, 2, 3)]


def example_sample_set_index_quads(sample_sets):
    assert len(sample_sets) >= 4
    yield [(0, 1, 2, 3)]
    yield [(0, 1, 2, 3), (3, 2, 1, 0)]
    yield [(0, 1, 2, 3), (3, 2, 1, 0), (1, 2, 3, 0)]


def example_windows(ts):
    """
    Generate a series of example windows for the specified tree sequence.
    """
    L = ts.sequence_length
    yield "sites"
    yield "trees"
    yield [0, L]
    yield ts.breakpoints(as_array=True)
    yield np.linspace(0, L, num=10)


class WeightStatsMixin(object):
    """
    Implements the verify method and dispatches it to verify_weighted_stat
    for a representative set of sample sets and windows.
    """

    def example_weights(self, ts, min_size=1):
        """
        Generate a series of example weights from the specfied tree sequence.
        """
        for k in [min_size, min_size + 1, min_size + 10]:
            W = 1.0 + np.zeros((ts.num_samples, k))
            W[0, :] = 2.0
            yield W
            for j in range(k):
                W[:, j] = np.random.exponential(1, ts.num_samples)
            yield W
            for j in range(k):
                W[:, j] = np.random.normal(0, 1, ts.num_samples)
            yield W

    def transform_weights(self, W):
        """
        Specific methods will need to transform weights
        before passing them to general_stat.
        """
        return W

    def verify(self, ts):
        for W, windows in subset_combos(
                self.example_weights(ts), example_windows(ts), p=0.1):
            self.verify_weighted_stat(ts, W, windows=windows)

    def verify_definition(
            self, ts, W, windows, summary_func, ts_method, definition):

        # general_stat will need an extra column for p
        gW = self.transform_weights(W)

        def wrapped_summary_func(x):
            with suppress_division_by_zero_warning():
                return summary_func(x)

        # Determine output_dim of the function
        M = len(wrapped_summary_func(gW[0]))
        for sn in [True, False]:
            sigma1 = ts.general_stat(gW, wrapped_summary_func, M,
                                     windows, mode=self.mode,
                                     span_normalise=sn)
            sigma2 = general_stat(ts, gW, wrapped_summary_func, windows, mode=self.mode,
                                  span_normalise=sn)
            sigma3 = ts_method(W, windows=windows, mode=self.mode,
                               span_normalise=sn)
            sigma4 = definition(ts, W, windows=windows, mode=self.mode,
                                span_normalise=sn)

            self.assertEqual(sigma1.shape, sigma2.shape)
            self.assertEqual(sigma1.shape, sigma3.shape)
            self.assertEqual(sigma1.shape, sigma4.shape)
            self.assertArrayAlmostEqual(sigma1, sigma2)
            self.assertArrayAlmostEqual(sigma1, sigma3)
            self.assertArrayAlmostEqual(sigma1, sigma4)


class SampleSetStatsMixin(object):
    """
    Implements the verify method and dispatches it to verify_sample_sets
    for a representative set of sample sets and windows.
    """
    def verify(self, ts):
        for sample_sets, windows in subset_combos(
                example_sample_sets(ts), example_windows(ts)):
            self.verify_sample_sets(ts, sample_sets, windows=windows)

    def verify_definition(
            self, ts, sample_sets, windows, summary_func, ts_method, definition):

        W = np.array(
            [[u in A for A in sample_sets] for u in ts.samples()], dtype=float)

        def wrapped_summary_func(x):
            with suppress_division_by_zero_warning():
                return summary_func(x)

        for sn in [True, False]:
            # Determine output_dim of the function
            M = len(summary_func(W[0]))
            sigma1 = ts.general_stat(W, wrapped_summary_func, M, windows, mode=self.mode,
                                     span_normalise=sn)
            sigma2 = general_stat(ts, W, wrapped_summary_func, windows, mode=self.mode,
                                  span_normalise=sn)
            sigma3 = ts_method(sample_sets, windows=windows, mode=self.mode,
                               span_normalise=sn)
            sigma4 = definition(ts, sample_sets, windows=windows, mode=self.mode,
                                span_normalise=sn)

            self.assertEqual(sigma1.shape, sigma2.shape)
            self.assertEqual(sigma1.shape, sigma3.shape)
            self.assertEqual(sigma1.shape, sigma4.shape)
            self.assertArrayAlmostEqual(sigma1, sigma2)
            self.assertArrayAlmostEqual(sigma1, sigma3)
            self.assertArrayAlmostEqual(sigma1, sigma4)


class KWaySampleSetStatsMixin(SampleSetStatsMixin):
    """
    Defines the verify definition method, which comparse the results from
    several different ways of defining and computing the same statistic.
    """
    def verify_definition(
            self, ts, sample_sets, indexes, windows, summary_func, ts_method,
            definition):

        def wrapped_summary_func(x):
            with suppress_division_by_zero_warning():
                return summary_func(x)

        W = np.array(
            [[u in A for A in sample_sets] for u in ts.samples()], dtype=float)
        # Determine output_dim of the function
        M = len(wrapped_summary_func(W[0]))
        sigma1 = ts.general_stat(W, wrapped_summary_func, M, windows, mode=self.mode)
        sigma2 = general_stat(ts, W, wrapped_summary_func, windows, mode=self.mode)
        sigma3 = ts_method(
            sample_sets, indexes=indexes, windows=windows, mode=self.mode)
        sigma4 = definition(
            ts, sample_sets, indexes=indexes, windows=windows, mode=self.mode)

        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertEqual(sigma1.shape, sigma3.shape)
        self.assertEqual(sigma1.shape, sigma4.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)
        self.assertArrayAlmostEqual(sigma1, sigma3)
        self.assertArrayAlmostEqual(sigma1, sigma4)


class TwoWaySampleSetStatsMixin(KWaySampleSetStatsMixin):
    """
    Implements the verify method and dispatches it to verify_sample_sets_indexes,
    which gives a representative sample of sample set indexes.
    """

    def verify(self, ts):
        for sample_sets, windows in subset_combos(
                example_sample_sets(ts, min_size=2), example_windows(ts)):
            for indexes in example_sample_set_index_pairs(sample_sets):
                self.verify_sample_sets_indexes(ts, sample_sets, indexes, windows)


class ThreeWaySampleSetStatsMixin(KWaySampleSetStatsMixin):
    """
    Implements the verify method and dispatches it to verify_sample_sets_indexes,
    which gives a representative sample of sample set indexes.
    """
    def verify(self, ts):
        for sample_sets, windows in subset_combos(
                example_sample_sets(ts, min_size=3), example_windows(ts)):
            for indexes in example_sample_set_index_triples(sample_sets):
                self.verify_sample_sets_indexes(ts, sample_sets, indexes, windows)


class FourWaySampleSetStatsMixin(KWaySampleSetStatsMixin):
    """
    Implements the verify method and dispatches it to verify_sample_sets_indexes,
    which gives a representative sample of sample set indexes.
    """
    def verify(self, ts):
        for sample_sets, windows in subset_combos(
                example_sample_sets(ts, min_size=4), example_windows(ts)):
            for indexes in example_sample_set_index_quads(sample_sets):
                self.verify_sample_sets_indexes(ts, sample_sets, indexes, windows)


############################################
# Diversity
############################################


def site_diversity(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(sample_sets)))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True).T
        site_positions = [x.position for x in ts.sites()]
        for i, X in enumerate(sample_sets):
            S = 0
            site_in_window = False
            denom = np.float64(len(X) * (len(X) - 1))
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    for x in X:
                        for y in set(X) - set([x]):
                            x_index = np.where(samples == x)[0][0]
                            y_index = np.where(samples == y)[0][0]
                            if haps[x_index][k] != haps[y_index][k]:
                                # x|y
                                S += 1
            if site_in_window:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def branch_diversity(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(sample_sets)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, X in enumerate(sample_sets):
            S = 0
            denom = np.float64(len(X) * (len(X) - 1))
            has_trees = False
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                SS = 0
                for x in X:
                    for y in set(X) - set([x]):
                        SS += path_length(tr, x, y)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            if has_trees:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def node_diversity(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    K = len(sample_sets)
    out = np.zeros((len(windows) - 1, ts.num_nodes, K))
    for k in range(K):
        X = sample_sets[k]
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            tX = len(X)
            denom = np.float64(len(X) * (len(X) - 1))
            S = np.zeros(ts.num_nodes)
            for tr in ts.trees(tracked_samples=X):
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in tr.nodes():
                    # count number of pairwise paths going through u
                    n = tr.num_tracked_samples(u)
                    SS[u] += 2 * n * (tX - n)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            with suppress_division_by_zero_warning():
                out[j, :, k] = S / denom
            if span_normalise:
                out[j, :, k] /= (end - begin)
    return out


def diversity(ts, sample_sets, windows=None, mode="site", span_normalise=True):
    """
    Computes average pairwise diversity between two random choices from x
    over the window specified.
    """
    method_map = {
        "site": site_diversity,
        "node": node_diversity,
        "branch": branch_diversity}
    return method_map[mode](ts, sample_sets, windows=windows,
                            span_normalise=span_normalise)


class TestDiversity(StatsTestCase, SampleSetStatsMixin):
    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets(self, ts, sample_sets, windows):
        n = np.array([len(x) for x in sample_sets])

        def f(x):
            with np.errstate(invalid='ignore', divide='ignore'):
                return x * (n - x) / (n * (n - 1))

        self.verify_definition(
            ts, sample_sets, windows, f, ts.diversity, diversity)


class TestBranchDiversity(TestDiversity, TopologyExamplesMixin):
    mode = "branch"


class TestNodeDiversity(TestDiversity, TopologyExamplesMixin):
    mode = "node"


class TestSiteDiversity(TestDiversity, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# Segregating sites
############################################

def site_segregating_sites(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(sample_sets)))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True)
        site_positions = [x.position for x in ts.sites()]
        for i, X in enumerate(sample_sets):
            X_index = np.where(np.in1d(samples, X))[0]
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    num_alleles = len(set(haps[k, X_index]))
                    out[j][i] += (num_alleles - 1)
            if span_normalise:
                out[j][i] /= (end - begin)
    return out


def branch_segregating_sites(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(sample_sets)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, X in enumerate(sample_sets):
            tX = len(X)
            for tr in ts.trees(tracked_samples=X):
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = 0
                for u in tr.nodes():
                    nX = tr.num_tracked_samples(u)
                    if nX > 0 and nX < tX:
                        SS += tr.branch_length(u)
                out[j][i] += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            if span_normalise:
                out[j][i] /= (end - begin)
    return out


def node_segregating_sites(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    K = len(sample_sets)
    out = np.zeros((len(windows) - 1, ts.num_nodes, K))
    for k in range(K):
        X = sample_sets[k]
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            tX = len(X)
            S = np.zeros(ts.num_nodes)
            for tr in ts.trees(tracked_samples=X):
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in tr.nodes():
                    nX = tr.num_tracked_samples(u)
                    SS[u] = (nX > 0) and (nX < tX)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            out[j, :, k] = S
            if span_normalise:
                out[j, :, k] /= (end - begin)
    return out


def segregating_sites(ts, sample_sets, windows=None, mode="site", span_normalise=True):
    """
    Computes the density of segregating sites over the window specified.
    """
    method_map = {
        "site": site_segregating_sites,
        "node": node_segregating_sites,
        "branch": branch_segregating_sites}
    return method_map[mode](ts, sample_sets, windows=windows,
                            span_normalise=span_normalise)


class TestSegregatingSites(StatsTestCase, SampleSetStatsMixin):
    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets(self, ts, sample_sets, windows):
        n = np.array([len(x) for x in sample_sets])

        # this works because sum_{i=1}^k (1-p_i) = k-1
        def f(x):
            return (x > 0) * (1 - x / n)

        self.verify_definition(
            ts, sample_sets, windows, f, ts.segregating_sites, segregating_sites)


class TestBranchSegregatingSites(TestSegregatingSites, TopologyExamplesMixin):
    mode = "branch"


class TestNodeSegregatingSites(TestSegregatingSites, TopologyExamplesMixin):
    mode = "node"


class TestSiteSegregatingSites(TestSegregatingSites, MutatedTopologyExamplesMixin):
    mode = "site"


class TestBranchSegregatingSitesProperties(StatsTestCase, TopologyExamplesMixin):

    def verify(self, ts):
        windows = ts.breakpoints(as_array=True)
        # If we split by tree, this should always be equal to the total
        # branch length. The definition of total_branch_length here is slightly
        # tricky: it's the sum of all branch lengths that subtend between 0
        # and n samples. This differs from the built-in total_branch_length
        # function, which just sums that total branch length reachable from
        # roots.
        tbl_tree = [
            sum(
                tree.branch_length(u) for u in tree.nodes()
                if 0 < tree.num_samples(u) < ts.num_samples)
            for tree in ts.trees()]
        # We must span_normalise, because these values are always weighted
        # by the span, so we're effectively cancelling out this contribution
        tbl = ts.segregating_sites([ts.samples()], windows=windows, mode="branch")
        tbl = tbl.reshape(tbl.shape[:-1])
        self.assertArrayAlmostEqual(tbl_tree, tbl)


############################################
# Tajima's D
############################################

def site_tajimas_d(ts, sample_sets, windows=None):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(sample_sets)))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True)
        site_positions = [x.position for x in ts.sites()]
        n = np.array([len(X) for X in sample_sets])
        for i, X in enumerate(sample_sets):
            nn = n[i]
            S = 0
            T = 0
            X_index = np.where(np.in1d(samples, X))[0]
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    hX = haps[k, X_index]
                    alleles = set(hX)
                    num_alleles = len(alleles)
                    n_alleles = [np.sum(hX == a) for a in alleles]
                    S += (num_alleles - 1)
                    for k in n_alleles:
                        with suppress_division_by_zero_warning():
                            T += k * (nn - k) / (nn * (nn - 1))
            with suppress_division_by_zero_warning():
                a1 = np.sum(1/np.arange(1, nn))  # this is h in the main version
                a2 = np.sum(1/np.arange(1, nn)**2)  # this is g
                b1 = (nn+1)/(3*(nn-1))
                b2 = 2 * (nn**2 + nn + 3) / (9 * nn * (nn-1))
                c1 = b1 - 1/a1
                c2 = b2 - (nn + 2)/(a1 * nn) + a2 / a1**2
                e1 = c1 / a1  # this is a
                e2 = c2 / (a1**2 + a2)  # this is b
                out[j][i] = (T - S/a1) / np.sqrt(e1*S + e2*S*(S-1))
    return out


def tajimas_d(ts, sample_sets, windows=None, mode="site", span_normalise=True):
    method_map = {
        "site": site_tajimas_d}
    return method_map[mode](ts, sample_sets, windows=windows,
                            span_normalise=span_normalise)


class TestTajimasD(StatsTestCase, SampleSetStatsMixin):
    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify(self, ts):
        # only check per-site
        for sample_sets in example_sample_sets(ts, min_size=1):
            self.verify_persite_tajimas_d(ts, sample_sets)

    def get_windows(self, ts):
        yield "sites"
        yield [0, ts.sequence_length]
        yield np.arange(0, 1.1, 0.1) * ts.sequence_length

    def verify_persite_tajimas_d(self, ts, sample_sets):
        for windows in self.get_windows(ts):
            sigma1 = ts.Tajimas_D(sample_sets, windows=windows, mode=self.mode)
            sigma2 = site_tajimas_d(ts, sample_sets, windows=windows)
            self.assertEqual(sigma1.shape, sigma2.shape)
            self.assertArrayAlmostEqual(sigma1, sigma2)


class TestSiteTajimasD(TestTajimasD, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# Y1
############################################

def branch_Y1(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(sample_sets)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, X in enumerate(sample_sets):
            S = 0
            denom = np.float64(len(X) * (len(X)-1) * (len(X)-2))
            has_trees = False
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                for x in X:
                    for y in set(X) - {x}:
                        for z in set(X) - {x, y}:
                            xy_mrca = tr.mrca(x, y)
                            xz_mrca = tr.mrca(x, z)
                            yz_mrca = tr.mrca(y, z)
                            if xy_mrca == xz_mrca:
                                #   /\
                                #  / /\
                                # x y  z
                                S += path_length(tr, x, yz_mrca) * this_length
                            elif xy_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # y x  z
                                S += path_length(tr, x, xz_mrca) * this_length
                            elif xz_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # z x  y
                                S += path_length(tr, x, xy_mrca) * this_length
            if has_trees:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def site_Y1(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(sample_sets)))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True).T
        site_positions = [x.position for x in ts.sites()]
        for i, X in enumerate(sample_sets):
            S = 0
            denom = np.float64(len(X) * (len(X)-1) * (len(X)-2))
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    for x in X:
                        x_index = np.where(samples == x)[0][0]
                        for y in set(X) - {x}:
                            y_index = np.where(samples == y)[0][0]
                            for z in set(X) - {x, y}:
                                z_index = np.where(samples == z)[0][0]
                                condition = (
                                    haps[x_index, k] != haps[y_index, k] and
                                    haps[x_index, k] != haps[z_index, k])
                                if condition:
                                    # x|yz
                                    S += 1
            if site_in_window:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def node_Y1(ts, sample_sets, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    K = len(sample_sets)
    out = np.zeros((len(windows) - 1, ts.num_nodes, K))
    for k in range(K):
        X = sample_sets[k]
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            tX = len(X)
            denom = np.float64(tX * (tX - 1) * (tX - 2))
            S = np.zeros(ts.num_nodes)
            for tr in ts.trees(tracked_samples=X):
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in tr.nodes():
                    # count number of paths above a but not b,c
                    n = tr.num_tracked_samples(u)
                    SS[u] += (n * (tX - n) * (tX - n - 1)
                              + (tX - n) * n * (n - 1))
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            with suppress_division_by_zero_warning():
                out[j, :, k] = S / denom
            if span_normalise:
                out[j, :, k] /= (end - begin)
    return out


def Y1(ts, sample_sets, windows=None, mode="site", span_normalise=True):
    windows = ts.parse_windows(windows)
    method_map = {
        "site": site_Y1,
        "node": node_Y1,
        "branch": branch_Y1}
    return method_map[mode](ts, sample_sets, windows=windows,
                            span_normalise=span_normalise)


class TestY1(StatsTestCase, SampleSetStatsMixin):
    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets(self, ts, sample_sets, windows):
        n = np.array([len(x) for x in sample_sets])
        denom = n * (n - 1) * (n - 2)

        def f(x):
            with np.errstate(invalid='ignore', divide='ignore'):
                return x * (n - x) * (n - x - 1) / denom

        self.verify_definition(ts, sample_sets, windows, f, ts.Y1, Y1)


class TestBranchY1(TestY1, TopologyExamplesMixin):
    mode = "branch"


class TestNodeY1(TestY1, TopologyExamplesMixin):
    mode = "node"


class TestSiteY1(TestY1, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# Divergence
############################################

def site_divergence(ts, sample_sets, indexes, windows=None, span_normalise=True):
    out = np.zeros((len(windows) - 1, len(indexes)))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True).T
        site_positions = [x.position for x in ts.sites()]
        for i, (ix, iy) in enumerate(indexes):
            X = sample_sets[ix]
            Y = sample_sets[iy]
            denom = np.float64(len(X) * len(Y))
            site_in_window = False
            S = 0
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    for x in X:
                        x_index = np.where(samples == x)[0][0]
                        for y in Y:
                            y_index = np.where(samples == y)[0][0]
                            if haps[x_index][k] != haps[y_index][k]:
                                # x|y
                                S += 1
            if site_in_window:
                with np.errstate(invalid='ignore', divide='ignore'):
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def branch_divergence(ts, sample_sets, indexes, windows=None, span_normalise=True):
    out = np.zeros((len(windows) - 1, len(indexes)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (ix, iy) in enumerate(indexes):
            X = sample_sets[ix]
            Y = sample_sets[iy]
            denom = np.float64(len(X) * len(Y))
            has_trees = False
            S = 0
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                SS = 0
                for x in X:
                    for y in Y:
                        SS += path_length(tr, x, y)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            if has_trees:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def node_divergence(ts, sample_sets, indexes, windows=None, span_normalise=True):
    out = np.zeros((len(windows) - 1, ts.num_nodes, len(indexes)))
    for i, (ix, iy) in enumerate(indexes):
        X = sample_sets[ix]
        Y = sample_sets[iy]
        tX = len(X)
        tY = len(Y)
        denom = np.float64(len(X) * len(Y))
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = np.zeros(ts.num_nodes)
            for t1, t2 in zip(ts.trees(tracked_samples=X),
                              ts.trees(tracked_samples=Y)):
                if t1.interval[1] <= begin:
                    continue
                if t1.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in t1.nodes():
                    # count number of pairwise paths going through u
                    nX = t1.num_tracked_samples(u)
                    nY = t2.num_tracked_samples(u)
                    SS[u] += (nX * (tY - nY) + (tX - nX) * nY)
                S += SS*(min(end, t1.interval[1]) - max(begin, t1.interval[0]))
            with suppress_division_by_zero_warning():
                out[j, :, i] = S / denom
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def divergence(ts, sample_sets, indexes=None, windows=None, mode="site",
               span_normalise=True):
    """
    Computes average pairwise divergence between two random choices from x
    over the window specified.
    """
    windows = ts.parse_windows(windows)
    if indexes is None:
        indexes = [(0, 1)]
    method_map = {
        "site": site_divergence,
        "node": node_divergence,
        "branch": branch_divergence}
    return method_map[mode](ts, sample_sets, indexes=indexes, windows=windows,
                            span_normalise=span_normalise)


class TestDivergence(StatsTestCase, TwoWaySampleSetStatsMixin):

    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets_indexes(self, ts, sample_sets, indexes, windows):
        n = np.array([len(x) for x in sample_sets])

        denom = np.array([n[i] * (n[j] - (i == j)) for i, j in indexes])

        def f(x):
            numer = np.array([(x[i] * (n[j] - x[j])) for i, j in indexes])
            return numer / denom

        self.verify_definition(
            ts, sample_sets, indexes, windows, f, ts.divergence, divergence)


class TestBranchDivergence(TestDivergence, TopologyExamplesMixin):
    mode = "branch"


class TestNodeDivergence(TestDivergence, TopologyExamplesMixin):
    mode = "node"


class TestSiteDivergence(TestDivergence, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# Fst
############################################

def single_site_Fst(ts, sample_sets, indexes):
    """
    Compute single-site Fst, which between two groups with frequencies p and q is
      1 - 2 * (p (1-p) + q(1-q)) / ( p(1-p) + q(1-q) + p(1-q) + q(1-p) )
    or in the multiallelic case, replacing p(1-p) with the sum over alleles of p(1-p),
    and adjusted for sampling without replacement.
    """
    # TODO: what to do in this case?
    if ts.num_sites == 0:
        out = np.array([np.repeat(np.nan, len(indexes))])
        return out
    out = np.zeros((ts.num_sites, len(indexes)))
    samples = ts.samples()
    # TODO deal with missing data properly.
    for j, v in enumerate(ts.variants(impute_missing_data=True)):
        for i, (ix, iy) in enumerate(indexes):
            g = v.genotypes
            X = sample_sets[ix]
            Y = sample_sets[iy]
            gX = [a for k, a in zip(samples, g) if k in X]
            gY = [a for k, a in zip(samples, g) if k in Y]
            nX = len(X)
            nY = len(Y)
            dX = dY = dXY = 0
            for a in set(g):
                fX = np.sum(gX == a)
                fY = np.sum(gY == a)
                with suppress_division_by_zero_warning():
                    dX += fX * (nX - fX) / (nX * (nX - 1))
                    dY += fY * (nY - fY) / (nY * (nY - 1))
                    dXY += (fX * (nY - fY) + (nX - fX) * fY) / (2 * nX * nY)
            with suppress_division_by_zero_warning():
                out[j][i] = 1 - 2 * (dX + dY) / (dX + dY + 2 * dXY)
    return out


class TestFst(StatsTestCase, TwoWaySampleSetStatsMixin):

    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify(self, ts):
        # only check per-site
        for sample_sets in example_sample_sets(ts, min_size=2):
            for indexes in example_sample_set_index_pairs(sample_sets):
                self.verify_persite_Fst(ts, sample_sets, indexes)

    def verify_persite_Fst(self, ts, sample_sets, indexes):
        sigma1 = ts.Fst(sample_sets, indexes=indexes, windows="sites",
                        mode=self.mode, span_normalise=False)
        sigma2 = single_site_Fst(ts, sample_sets, indexes)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)


class FstInterfaceMixin(object):

    def test_interface(self):
        ts = msprime.simulate(10, mutation_rate=0.0)
        sample_sets = [[0, 1, 2], [6, 7], [4]]
        with self.assertRaises(ValueError):
            ts.Fst(sample_sets, mode=self.mode)
        with self.assertRaises(ValueError):
            ts.Fst(sample_sets, indexes=[(0, 1, 2), (3, 4, 5)], mode=self.mode)
        with self.assertRaises(tskit.LibraryError):
            ts.Fst(sample_sets, indexes=[(0, 1), (0, 20)])
        sigma1 = ts.Fst(sample_sets, indexes=[(0, 1)], mode=self.mode)
        sigma2 = ts.Fst(sample_sets, indexes=[(0, 1), (0, 2), (1, 2)], mode=self.mode)
        self.assertArrayAlmostEqual(sigma1[..., 0], sigma2[..., 0])


class TestSiteFst(TestFst, MutatedTopologyExamplesMixin, FstInterfaceMixin):
    mode = "site"


# Since Fst is defined using diversity and divergence, we don't seriously
# test it for correctness for node and branch, and only test the interface.

class TestNodeFst(StatsTestCase, FstInterfaceMixin):
    mode = "node"


class TestBranchFst(StatsTestCase, FstInterfaceMixin):
    mode = "node"


############################################
# Y2
############################################

def branch_Y2(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(indexes)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (ix, iy) in enumerate(indexes):
            X = sample_sets[ix]
            Y = sample_sets[iy]
            denom = np.float64(len(X) * len(Y) * (len(Y)-1))
            has_trees = False
            S = 0
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                for x in X:
                    for y in Y:
                        for z in set(Y) - {y}:
                            xy_mrca = tr.mrca(x, y)
                            xz_mrca = tr.mrca(x, z)
                            yz_mrca = tr.mrca(y, z)
                            if xy_mrca == xz_mrca:
                                #   /\
                                #  / /\
                                # x y  z
                                S += path_length(tr, x, yz_mrca) * this_length
                            elif xy_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # y x  z
                                S += path_length(tr, x, xz_mrca) * this_length
                            elif xz_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # z x  y
                                S += path_length(tr, x, xy_mrca) * this_length
            if has_trees:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def site_Y2(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    samples = ts.samples()
    out = np.zeros((len(windows) - 1, len(indexes)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True).T
        site_positions = [x.position for x in ts.sites()]
        for i, (ix, iy) in enumerate(indexes):
            X = sample_sets[ix]
            Y = sample_sets[iy]
            denom = np.float64(len(X) * len(Y) * (len(Y)-1))
            S = 0
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    for x in X:
                        x_index = np.where(samples == x)[0][0]
                        for y in Y:
                            y_index = np.where(samples == y)[0][0]
                            for z in set(Y) - {y}:
                                z_index = np.where(samples == z)[0][0]
                                condition = (
                                    haps[x_index, k] != haps[y_index, k] and
                                    haps[x_index, k] != haps[z_index, k])
                                if condition:
                                    # x|yz
                                    S += 1
            if site_in_window:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def node_Y2(ts, sample_sets, indexes, windows=None, span_normalise=True):
    out = np.zeros((len(windows) - 1, ts.num_nodes, len(indexes)))
    for i, (ix, iy) in enumerate(indexes):
        X = sample_sets[ix]
        Y = sample_sets[iy]
        tX = len(X)
        tY = len(Y)
        denom = np.float64(tX * tY * (tY - 1))
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = np.zeros(ts.num_nodes)
            for t1, t2 in zip(ts.trees(tracked_samples=X),
                              ts.trees(tracked_samples=Y)):
                if t1.interval[1] <= begin:
                    continue
                if t1.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in t1.nodes():
                    # count number of pairwise paths going through u
                    nX = t1.num_tracked_samples(u)
                    nY = t2.num_tracked_samples(u)
                    SS[u] += (nX * (tY - nY) * (tY - nY - 1)
                              + (tX - nX) * nY * (nY - 1))
                S += SS*(min(end, t1.interval[1]) - max(begin, t1.interval[0]))
            with suppress_division_by_zero_warning():
                out[j, :, i] = S / denom
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def Y2(ts, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True):
    windows = ts.parse_windows(windows)
    if indexes is None:
        indexes = [(0, 1)]
    method_map = {
        "site": site_Y2,
        "node": node_Y2,
        "branch": branch_Y2}
    return method_map[mode](ts, sample_sets, indexes=indexes, windows=windows,
                            span_normalise=span_normalise)


class TestY2(StatsTestCase, TwoWaySampleSetStatsMixin):

    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets_indexes(self, ts, sample_sets, indexes, windows):
        n = np.array([len(x) for x in sample_sets])

        denom = np.array([n[i] * n[j] * (n[j] - 1) for i, j in indexes])

        def f(x):
            numer = np.array([
                (x[i] * (n[j] - x[j]) * (n[j] - x[j] - 1)) for i, j in indexes])
            return numer / denom

        self.verify_definition(ts, sample_sets, indexes, windows, f, ts.Y2, Y2)


class TestBranchY2(TestY2, TopologyExamplesMixin):
    mode = "branch"


class TestNodeY2(TestY2, TopologyExamplesMixin):
    mode = "node"


class TestSiteY2(TestY2, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# Y3
############################################

def branch_Y3(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(indexes)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (ix, iy, iz) in enumerate(indexes):
            S = 0
            X = sample_sets[ix]
            Y = sample_sets[iy]
            Z = sample_sets[iz]
            denom = np.float64(len(X) * len(Y) * len(Z))
            has_trees = False
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                for x in X:
                    for y in Y:
                        for z in Z:
                            xy_mrca = tr.mrca(x, y)
                            xz_mrca = tr.mrca(x, z)
                            yz_mrca = tr.mrca(y, z)
                            if xy_mrca == xz_mrca:
                                #   /\
                                #  / /\
                                # x y  z
                                S += path_length(tr, x, yz_mrca) * this_length
                            elif xy_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # y x  z
                                S += path_length(tr, x, xz_mrca) * this_length
                            elif xz_mrca == yz_mrca:
                                #   /\
                                #  / /\
                                # z x  y
                                S += path_length(tr, x, xy_mrca) * this_length
            if has_trees:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def site_Y3(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(indexes)))
    haps = ts.genotype_matrix(impute_missing_data=True).T
    site_positions = ts.tables.sites.position
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (ix, iy, iz) in enumerate(indexes):
            X = sample_sets[ix]
            Y = sample_sets[iy]
            Z = sample_sets[iz]
            denom = np.float64(len(X) * len(Y) * len(Z))
            S = 0
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    for x in X:
                        x_index = np.where(samples == x)[0][0]
                        for y in Y:
                            y_index = np.where(samples == y)[0][0]
                            for z in Z:
                                z_index = np.where(samples == z)[0][0]
                                if ((haps[x_index][k] != haps[y_index][k])
                                   and (haps[x_index][k] != haps[z_index][k])):
                                    # x|yz
                                    with suppress_division_by_zero_warning():
                                        S += 1
            if site_in_window:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def node_Y3(ts, sample_sets, indexes, windows=None, span_normalise=True):
    out = np.zeros((len(windows) - 1, ts.num_nodes, len(indexes)))
    for i, (ix, iy, iz) in enumerate(indexes):
        X = sample_sets[ix]
        Y = sample_sets[iy]
        Z = sample_sets[iz]
        tX = len(X)
        tY = len(Y)
        tZ = len(Z)
        denom = np.float64(tX * tY * tZ)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = np.zeros(ts.num_nodes)
            for t1, t2, t3 in zip(ts.trees(tracked_samples=X),
                                  ts.trees(tracked_samples=Y),
                                  ts.trees(tracked_samples=Z)):
                if t1.interval[1] <= begin:
                    continue
                if t1.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in t1.nodes():
                    # count number of pairwise paths going through u
                    nX = t1.num_tracked_samples(u)
                    nY = t2.num_tracked_samples(u)
                    nZ = t3.num_tracked_samples(u)
                    SS[u] += (nX * (tY - nY) * (tZ - nZ)
                              + (tX - nX) * nY * nZ)
                S += SS*(min(end, t1.interval[1]) - max(begin, t1.interval[0]))
            with suppress_division_by_zero_warning():
                out[j, :, i] = S / denom
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def Y3(ts, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True):
    windows = ts.parse_windows(windows)
    if indexes is None:
        indexes = [(0, 1, 2)]
    method_map = {
        "site": site_Y3,
        "node": node_Y3,
        "branch": branch_Y3}
    return method_map[mode](ts, sample_sets, indexes=indexes, windows=windows,
                            span_normalise=span_normalise)


class TestY3(StatsTestCase, ThreeWaySampleSetStatsMixin):

    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets_indexes(self, ts, sample_sets, indexes, windows):
        n = np.array([len(x) for x in sample_sets])
        denom = np.array([n[i] * n[j] * n[k] for i, j, k in indexes])

        def f(x):
            numer = np.array(
                [x[i] * (n[j] - x[j]) * (n[k] - x[k]) for i, j, k in indexes])
            return numer / denom

        self.verify_definition(ts, sample_sets, indexes, windows, f, ts.Y3, Y3)


class TestBranchY3(TestY3, TopologyExamplesMixin):
    mode = "branch"


class TestNodeY3(TestY3, TopologyExamplesMixin):
    mode = "node"


class TestSiteY3(TestY3, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# f2
############################################

def branch_f2(ts, sample_sets, indexes, windows=None, span_normalise=True):
    # this is f4(A,B;A,B) but drawing distinct samples from A and B
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(indexes)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (ia, ib) in enumerate(indexes):
            A = sample_sets[ia]
            B = sample_sets[ib]
            denom = np.float64(len(A) * (len(A) - 1) * len(B) * (len(B) - 1))
            has_trees = False
            S = 0
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                SS = 0
                for a in A:
                    for b in B:
                        for c in set(A) - {a}:
                            for d in set(B) - {b}:
                                with suppress_division_by_zero_warning():
                                    SS += path_length(tr, tr.mrca(a, c), tr.mrca(b, d))
                                    SS -= path_length(tr, tr.mrca(a, d), tr.mrca(b, c))
                S += SS * this_length
            if has_trees:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def site_f2(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(indexes)))
    samples = ts.samples()
    haps = ts.genotype_matrix(impute_missing_data=True).T
    site_positions = ts.tables.sites.position
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (iA, iB) in enumerate(indexes):
            A = sample_sets[iA]
            B = sample_sets[iB]
            denom = np.float64(len(A) * (len(A) - 1) * len(B) * (len(B) - 1))
            S = 0
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    for a in A:
                        a_index = np.where(samples == a)[0][0]
                        for b in B:
                            b_index = np.where(samples == b)[0][0]
                            for c in set(A) - {a}:
                                c_index = np.where(samples == c)[0][0]
                                for d in set(B) - {b}:
                                    d_index = np.where(samples == d)[0][0]
                                    if ((haps[a_index][k] == haps[c_index][k])
                                       and (haps[a_index][k] != haps[d_index][k])
                                       and (haps[a_index][k] != haps[b_index][k])):
                                        # ac|bd
                                        S += 1
                                    elif ((haps[a_index][k] == haps[d_index][k])
                                          and (haps[a_index][k] != haps[c_index][k])
                                          and (haps[a_index][k] != haps[b_index][k])):
                                        # ad|bc
                                        S -= 1
            if site_in_window:
                with np.errstate(invalid='ignore', divide='ignore'):
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def node_f2(ts, sample_sets, indexes, windows=None, span_normalise=True):
    out = np.zeros((len(windows) - 1, ts.num_nodes, len(indexes)))
    for i, (ia, ib) in enumerate(indexes):
        A = sample_sets[ia]
        B = sample_sets[ib]
        tA = len(A)
        tB = len(B)
        denom = np.float64(tA * (tA - 1) * tB * (tB - 1))
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = np.zeros(ts.num_nodes)
            for t1, t2 in zip(ts.trees(tracked_samples=A),
                              ts.trees(tracked_samples=B)):
                if t1.interval[1] <= begin:
                    continue
                if t1.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in t1.nodes():
                    # count number of pairwise paths going through u
                    nA = t1.num_tracked_samples(u)
                    nB = t2.num_tracked_samples(u)
                    # xy|uv - xv|uy with x,y in A, u, v in B
                    SS[u] += (nA * (nA - 1) * (tB - nB) * (tB - nB - 1)
                              + (tA - nA) * (tA - nA - 1) * nB * (nB - 1))
                    SS[u] -= 2 * nA * nB * (tA - nA) * (tB - nB)
                S += SS*(min(end, t1.interval[1]) - max(begin, t1.interval[0]))
            with suppress_division_by_zero_warning():
                out[j, :, i] = S / denom
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def f2(ts, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True):
    """
    Patterson's f2 statistic definitions.
    """
    windows = ts.parse_windows(windows)
    if indexes is None:
        indexes = [(0, 1)]
    method_map = {
        "site": site_f2,
        "node": node_f2,
        "branch": branch_f2}
    return method_map[mode](ts, sample_sets, indexes=indexes, windows=windows,
                            span_normalise=span_normalise)


class Testf2(StatsTestCase, TwoWaySampleSetStatsMixin):

    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets_indexes(self, ts, sample_sets, indexes, windows):
        n = np.array([len(x) for x in sample_sets])

        denom = np.array([n[i] * (n[i] - 1) * n[j] * (n[j] - 1) for i, j in indexes])

        def f(x):
            numer = np.array([
                x[i] * (x[i] - 1) * (n[j] - x[j]) * (n[j] - x[j] - 1)
                - x[i] * (n[i] - x[i]) * (n[j] - x[j]) * x[j]
                for i, j in indexes])
            return numer / denom

        self.verify_definition(ts, sample_sets, indexes, windows, f, ts.f2, f2)


class TestBranchf2(Testf2, TopologyExamplesMixin):
    mode = "branch"


class TestNodef2(Testf2, TopologyExamplesMixin):
    mode = "node"


class TestSitef2(Testf2, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# f3
############################################

def branch_f3(ts, sample_sets, indexes, windows=None, span_normalise=True):
    # this is f4(A,B;A,C) but drawing distinct samples from A
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(indexes)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (ia, ib, ic) in enumerate(indexes):
            A = sample_sets[ia]
            B = sample_sets[ib]
            C = sample_sets[ic]
            denom = np.float64(len(A) * (len(A) - 1) * len(B) * len(C))
            has_trees = False
            S = 0
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                SS = 0
                for a in A:
                    for b in B:
                        for c in set(A) - {a}:
                            for d in C:
                                SS += path_length(tr, tr.mrca(a, c), tr.mrca(b, d))
                                SS -= path_length(tr, tr.mrca(a, d), tr.mrca(b, c))
                S += SS * this_length
            if has_trees:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def site_f3(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(indexes)))
    samples = ts.samples()
    haps = ts.genotype_matrix(impute_missing_data=True).T
    site_positions = ts.tables.sites.position
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (iA, iB, iC) in enumerate(indexes):
            A = sample_sets[iA]
            B = sample_sets[iB]
            C = sample_sets[iC]
            denom = np.float64(len(A) * (len(A) - 1) * len(B) * len(C))
            S = 0
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    for a in A:
                        a_index = np.where(samples == a)[0][0]
                        for b in B:
                            b_index = np.where(samples == b)[0][0]
                            for c in set(A) - {a}:
                                c_index = np.where(samples == c)[0][0]
                                for d in C:
                                    d_index = np.where(samples == d)[0][0]
                                    if ((haps[a_index][k] == haps[c_index][k])
                                       and (haps[a_index][k] != haps[d_index][k])
                                       and (haps[a_index][k] != haps[b_index][k])):
                                        # ac|bd
                                        S += 1
                                    elif ((haps[a_index][k] == haps[d_index][k])
                                          and (haps[a_index][k] != haps[c_index][k])
                                          and (haps[a_index][k] != haps[b_index][k])):
                                        # ad|bc
                                        S -= 1
            if site_in_window:
                with np.errstate(invalid='ignore', divide='ignore'):
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def node_f3(ts, sample_sets, indexes, windows=None, span_normalise=True):
    out = np.zeros((len(windows) - 1, ts.num_nodes, len(indexes)))
    for i, (iA, iB, iC) in enumerate(indexes):
        A = sample_sets[iA]
        B = sample_sets[iB]
        C = sample_sets[iC]
        tA = len(A)
        tB = len(B)
        tC = len(C)
        denom = np.float64(tA * (tA - 1) * tB * tC)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = np.zeros(ts.num_nodes)
            for t1, t2, t3 in zip(ts.trees(tracked_samples=A),
                                  ts.trees(tracked_samples=B),
                                  ts.trees(tracked_samples=C)):
                if t1.interval[1] <= begin:
                    continue
                if t1.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in t1.nodes():
                    # count number of pairwise paths going through u
                    nA = t1.num_tracked_samples(u)
                    nB = t2.num_tracked_samples(u)
                    nC = t3.num_tracked_samples(u)
                    # xy|uv - xv|uy with x,y in A, u in B and v in C
                    SS[u] += (nA * (nA - 1) * (tB - nB) * (tC - nC)
                              + (tA - nA) * (tA - nA - 1) * nB * nC)
                    SS[u] -= (nA * nC * (tA - nA) * (tB - nB)
                              + (tA - nA) * (tC - nC) * nA * nB)
                S += SS*(min(end, t1.interval[1]) - max(begin, t1.interval[0]))
            with suppress_division_by_zero_warning():
                out[j, :, i] = S / denom
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def f3(ts, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True):
    """
    Patterson's f3 statistic definitions.
    """
    windows = ts.parse_windows(windows)
    if indexes is None:
        indexes = [(0, 1, 2)]
    method_map = {
        "site": site_f3,
        "node": node_f3,
        "branch": branch_f3}
    return method_map[mode](ts, sample_sets, indexes=indexes, windows=windows,
                            span_normalise=span_normalise)


class Testf3(StatsTestCase, ThreeWaySampleSetStatsMixin):

    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets_indexes(self, ts, sample_sets, indexes, windows):
        n = np.array([len(x) for x in sample_sets])
        denom = np.array([n[i] * (n[i] - 1) * n[j] * n[k] for i, j, k in indexes])

        def f(x):
            numer = np.array([
                x[i] * (x[i] - 1) * (n[j] - x[j]) * (n[k] - x[k])
                - x[i] * (n[i] - x[i]) * (n[j] - x[j]) * x[k] for i, j, k in indexes])
            return numer / denom
        self.verify_definition(ts, sample_sets, indexes, windows, f, ts.f3, f3)


class TestBranchf3(Testf3, TopologyExamplesMixin):
    mode = "branch"


class TestNodef3(Testf3, TopologyExamplesMixin):
    mode = "node"


class TestSitef3(Testf3, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# f4
############################################

def branch_f4(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, len(indexes)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (iA, iB, iC, iD) in enumerate(indexes):
            A = sample_sets[iA]
            B = sample_sets[iB]
            C = sample_sets[iC]
            D = sample_sets[iD]
            denom = np.float64(len(A) * len(B) * len(C) * len(D))
            has_trees = False
            S = 0
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                this_length = min(end, tr.interval[1]) - max(begin, tr.interval[0])
                SS = 0
                for a in A:
                    for b in B:
                        for c in C:
                            for d in D:
                                with suppress_division_by_zero_warning():
                                    SS += path_length(tr, tr.mrca(a, c), tr.mrca(b, d))
                                    SS -= path_length(tr, tr.mrca(a, d), tr.mrca(b, c))
                S += SS * this_length
            if has_trees:
                with suppress_division_by_zero_warning():
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def site_f4(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    samples = ts.samples()
    haps = ts.genotype_matrix(impute_missing_data=True).T
    site_positions = ts.tables.sites.position
    out = np.zeros((len(windows) - 1, len(indexes)))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i, (iA, iB, iC, iD) in enumerate(indexes):
            A = sample_sets[iA]
            B = sample_sets[iB]
            C = sample_sets[iC]
            D = sample_sets[iD]
            denom = np.float64(len(A) * len(B) * len(C) * len(D))
            S = 0
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    for a in A:
                        a_index = np.where(samples == a)[0][0]
                        for b in B:
                            b_index = np.where(samples == b)[0][0]
                            for c in C:
                                c_index = np.where(samples == c)[0][0]
                                for d in D:
                                    d_index = np.where(samples == d)[0][0]
                                    if ((haps[a_index][k] == haps[c_index][k])
                                       and (haps[a_index][k] != haps[d_index][k])
                                       and (haps[a_index][k] != haps[b_index][k])):
                                        # ac|bd
                                        S += 1
                                    elif ((haps[a_index][k] == haps[d_index][k])
                                          and (haps[a_index][k] != haps[c_index][k])
                                          and (haps[a_index][k] != haps[b_index][k])):
                                        # ad|bc
                                        S -= 1
            if site_in_window:
                with np.errstate(invalid='ignore', divide='ignore'):
                    out[j][i] = S / denom
                if span_normalise:
                    out[j][i] /= (end - begin)
    return out


def node_f4(ts, sample_sets, indexes, windows=None, span_normalise=True):
    windows = ts.parse_windows(windows)
    out = np.zeros((len(windows) - 1, ts.num_nodes, len(indexes)))
    for i, (iA, iB, iC, iD) in enumerate(indexes):
        A = sample_sets[iA]
        B = sample_sets[iB]
        C = sample_sets[iC]
        D = sample_sets[iD]
        tA = len(A)
        tB = len(B)
        tC = len(C)
        tD = len(D)
        denom = np.float64(tA * tB * tC * tD)
        for j in range(len(windows) - 1):
            begin = windows[j]
            end = windows[j + 1]
            S = np.zeros(ts.num_nodes)
            for t1, t2, t3, t4 in zip(ts.trees(tracked_samples=A),
                                      ts.trees(tracked_samples=B),
                                      ts.trees(tracked_samples=C),
                                      ts.trees(tracked_samples=D)):
                if t1.interval[1] <= begin:
                    continue
                if t1.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in t1.nodes():
                    # count number of pairwise paths going through u
                    nA = t1.num_tracked_samples(u)
                    nB = t2.num_tracked_samples(u)
                    nC = t3.num_tracked_samples(u)
                    nD = t4.num_tracked_samples(u)
                    # ac|bd - ad|bc
                    SS[u] += (nA * nC * (tB - nB) * (tD - nD)
                              + (tA - nA) * (tC - nC) * nB * nD)
                    SS[u] -= (nA * nD * (tB - nB) * (tC - nC)
                              + (tA - nA) * (tD - nD) * nB * nC)
                S += SS*(min(end, t1.interval[1]) - max(begin, t1.interval[0]))
            with suppress_division_by_zero_warning():
                out[j, :, i] = S / denom
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def f4(ts, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True):
    """
    Patterson's f4 statistic definitions.
    """
    if indexes is None:
        indexes = [(0, 1, 2, 3)]
    method_map = {
        "site": site_f4,
        "node": node_f4,
        "branch": branch_f4}
    return method_map[mode](ts, sample_sets, indexes=indexes, windows=windows,
                            span_normalise=span_normalise)


class Testf4(StatsTestCase, FourWaySampleSetStatsMixin):

    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_sample_sets_indexes(self, ts, sample_sets, indexes, windows):
        n = np.array([len(x) for x in sample_sets])
        denom = np.array([n[i] * n[j] * n[k] * n[l] for i, j, k, l in indexes])

        def f(x):
            numer = np.array([
                x[i] * x[k] * (n[j] - x[j]) * (n[l] - x[l])
                - x[i] * x[l] * (n[j] - x[j]) * (n[k] - x[k]) for i, j, k, l in indexes])
            return numer / denom
        self.verify_definition(ts, sample_sets, indexes, windows, f, ts.f4, f4)

    def verify_interface(self, ts):
        self.verify_interface_method(ts.f4)


class TestBranchf4(Testf4, TopologyExamplesMixin):
    mode = "branch"


class TestNodef4(Testf4, TopologyExamplesMixin):
    mode = "node"


class TestSitef4(Testf4, MutatedTopologyExamplesMixin):
    mode = "site"


############################################
# Allele frequency spectrum
############################################

def fold(x, dims):
    """
    Folds the specified coordinates.
    """
    x = np.array(x, dtype=int)
    dims = np.array(dims, dtype=int)
    k = len(dims)
    n = np.sum(dims - 1) / 2
    s = np.sum(x)
    while s == n and k > 0:
        k -= 1
        assert k >= 0
        n -= (dims[k] - 1) / 2
        s -= x[k]
    if s > n:
        x = dims - 1 - x
    assert np.all(x >= 0)
    return tuple(x)


def foldit(A):
    B = np.zeros(A.shape)
    dims = A.shape
    inds = [range(k) for k in dims]
    for ij in itertools.product(*inds):
        nij = fold(ij, dims)
        B[nij] += A[ij]
    return B


class TestFold(unittest.TestCase):
    """
    Tests for the fold operation used in the AFS.
    """

    def test_examples(self):
        A = np.arange(12)
        Af = np.array(
            [11., 11., 11., 11., 11., 11.,  0.,  0.,  0.,  0.,  0.,  0.])

        self.assertTrue(np.all(foldit(A) == Af))

        B = A.copy().reshape(3, 4)
        Bf = np.array([[11., 11., 11.,  0.],
                       [11., 11.,  0.,  0.],
                       [11.,  0.,  0.,  0.]])
        self.assertTrue(np.all(foldit(B) == Bf))

        C = A.copy().reshape(3, 2, 2)
        Cf = np.array([[[11., 11.],
                        [11., 11.]],
                       [[11., 11.],
                        [0.,  0.]],
                       [[0.,  0.],
                        [0.,  0.]]])
        self.assertTrue(np.all(foldit(C) == Cf))

        D = np.arange(9).reshape((3, 3))
        Df = np.array([[8., 8., 8.],
                       [8., 4., 0.],
                       [0., 0., 0.]])
        self.assertTrue(np.all(foldit(D) == Df))

        E = np.arange(9)
        Ef = np.array([8., 8., 8., 8., 4., 0., 0., 0., 0.])
        self.assertTrue(np.all(foldit(E) == Ef))


def naive_site_allele_frequency_spectrum(
        ts, sample_sets, windows=None, polarised=False, span_normalise=True):
    """
    The joint allele frequency spectrum for sites.
    """
    windows = ts.parse_windows(windows)
    num_windows = len(windows) - 1
    out_dim = [1 + len(sample_set) for sample_set in sample_sets]
    out = np.zeros([num_windows] + out_dim)
    G = ts.genotype_matrix(impute_missing_data=True)
    samples = ts.samples()
    # Indexes of the samples within the sample sets into the samples array.
    sample_set_indexes = [
        np.array([np.where(x == samples)[0][0] for x in sample_set])
        for sample_set in sample_sets]
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for site in ts.sites():
            S = np.zeros(out_dim)
            if begin <= site.position < end:
                g = G[site.id]
                alleles = np.unique(g)

                # Any site monomorphic across all samples does not contribute
                if len(alleles) == 1:
                    continue

                # For each allele, count the number present in each sample set.
                count = {
                    allele: np.zeros(len(sample_sets), dtype=int)
                    for allele in alleles}
                for k, sample_set in enumerate(sample_set_indexes):
                    allele_counts = zip(*np.unique(g[sample_set], return_counts=True))
                    for allele, c in allele_counts:
                        count[allele][k] = c
                increment = 0.5
                if polarised:
                    increment = 1
                    # Remove the contribution of the ancestral state
                    if 0 in count:
                        del count[0]
                for allele_count in count.values():
                    x = tuple(allele_count)
                    if not polarised:
                        x = fold(x, out_dim)
                    S[x] += increment
            if span_normalise:
                S /= (end - begin)
            out[j, :] += S
    return out


def naive_branch_allele_frequency_spectrum(
        ts, sample_sets, windows=None, polarised=False, span_normalise=True):
    """
    The joint allele frequency spectrum for branches.
    """
    windows = ts.parse_windows(windows)
    num_windows = len(windows) - 1
    out_dim = [1 + len(sample_set) for sample_set in sample_sets]
    out = np.zeros([num_windows] + out_dim)
    for j in range(num_windows):
        begin = windows[j]
        end = windows[j + 1]
        for set_index, sample_set in enumerate(sample_sets):
            S = np.zeros(out_dim)
            trees = [
                next(ts.trees(tracked_samples=sample_set, sample_counts=True))
                for sample_set in sample_sets]
            t = trees[0]
            while True:
                tr_len = min(end, t.interval[1]) - max(begin, t.interval[0])
                if tr_len > 0:
                    for node in t.nodes():
                        if 0 < t.num_samples(node) < ts.num_samples:
                            x = [tree.num_tracked_samples(node) for tree in trees]
                            # Note x must be a tuple for indexing to work
                            if polarised:
                                S[tuple(x)] += t.branch_length(node) * tr_len
                            else:
                                x = fold(x, out_dim)
                                S[tuple(x)] += 0.5 * t.branch_length(node) * tr_len

                # Advance the trees
                more = [tree.next() for tree in trees]
                assert len(set(more)) == 1
                if not more[0]:
                    break
            if span_normalise:
                S /= (end - begin)
            out[j, :] = S
    return out


def naive_allele_frequency_spectrum(
        ts, sample_sets, windows=None, polarised=False, mode="site",
        span_normalise=True):
    """
    Naive definition of the generalised site frequency spectrum.
    """
    method_map = {
        "site": naive_site_allele_frequency_spectrum,
        "branch": naive_branch_allele_frequency_spectrum}
    return method_map[mode](
        ts, sample_sets, windows=windows, polarised=polarised,
        span_normalise=span_normalise)


def branch_allele_frequency_spectrum(
        ts, sample_sets, windows, polarised=False, span_normalise=True):
    """
    Efficient implementation of the algorithm used as the basis for the
    underlying C version.
    """
    num_sample_sets = len(sample_sets)
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    out_dim = [1 + len(sample_set) for sample_set in sample_sets]
    time = ts.tables.nodes.time

    result = np.zeros([num_windows] + out_dim)
    # Number of nodes in sample_set j ancestral to each node u.
    count = np.zeros((ts.num_nodes, num_sample_sets + 1), dtype=np.uint32)
    for j in range(num_sample_sets):
        count[sample_sets[j], j] = 1
    # The last column counts across all samples
    count[ts.samples(), -1] = 1
    # contains the location of the last time we updated the output for a node.
    last_update = np.zeros((ts.num_nodes))
    window_index = 0
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    branch_length = np.zeros(ts.num_nodes)
    tree_index = 0

    def update_result(window_index, u, right):
        if 0 < count[u, -1] < ts.num_samples:
            x = (right - last_update[u]) * branch_length[u]
            c = count[u, :num_sample_sets]
            if not polarised:
                c = fold(c, out_dim)
                x *= 0.5
            index = tuple([window_index] + list(c))
            result[index] += x
        last_update[u] = right

    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():

        for edge in edges_out:
            u = edge.child
            v = edge.parent
            update_result(window_index, u, t_left)
            while v != -1:
                update_result(window_index, v, t_left)
                count[v] -= count[u]
                v = parent[v]
            parent[u] = -1
            branch_length[u] = 0

        for edge in edges_in:
            u = edge.child
            v = edge.parent
            parent[u] = v
            branch_length[u] = time[v] - time[u]
            while v != -1:
                update_result(window_index, v, t_left)
                count[v] += count[u]
                v = parent[v]

        # Update the windows
        while window_index < num_windows and windows[window_index + 1] <= t_right:
            w_right = windows[window_index + 1]
            # This seems like a bad idea as we incur a O(N) cost for each window,
            # where N is the number of nodes.  It might be hard to do much better
            # though, since we can't help but incur O(|sample_set|) cost at each window
            # which we'll assume is O(n), and for large n, N isn't much larger than n.
            # For K > 1 dimensions, the cost of the scan through the nodes is much
            # less than the O(n^K) required to copy (if n is large and K is small).
            # We could keep track of the roots and do a tree traversal, bringing this
            # down to O(n), but this adds a lot of complexity and memory and I'm
            # fairly confident would be slower overall. We could keep a set of
            # non-zero branches, but this would add a O(log n) cost to each edge
            # insertion and removal and a lot of complexity to the C implementation.
            for u in range(ts.num_nodes):
                update_result(window_index, u, w_right)
            window_index += 1
        tree_index += 1

    assert window_index == windows.shape[0] - 1
    if span_normalise:
        for j in range(num_windows):
            result[j] /= windows[j + 1] - windows[j]
    return result


def site_allele_frequency_spectrum(
        ts, sample_sets, windows, polarised=False, span_normalise=True):
    """
    Efficient implementation of the algorithm used as the basis for the
    underlying C version.
    """
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    out_dim = [1 + len(sample_set) for sample_set in sample_sets]

    result = np.zeros([num_windows] + out_dim)
    # Add an extra sample set to count across all samples
    sample_sets = list(sample_sets) + [ts.samples()]
    # Number of nodes in sample_set j ancestral to each node u.
    count = np.zeros((ts.num_nodes, len(sample_sets)), dtype=np.uint32)
    for j in range(len(sample_sets)):
        count[sample_sets[j], j] = 1

    site_index = 0
    mutation_index = 0
    window_index = 0
    sites = ts.tables.sites
    mutations = ts.tables.mutations
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            u = edge.child
            v = edge.parent
            while v != -1:
                count[v] -= count[u]
                v = parent[v]
            parent[u] = -1

        for edge in edges_in:
            u = edge.child
            v = edge.parent
            parent[u] = v
            while v != -1:
                count[v] += count[u]
                v = parent[v]

        while site_index < len(sites) and sites.position[site_index] < t_right:
            assert t_left <= sites.position[site_index]
            ancestral_state = sites[site_index].ancestral_state
            allele_count = collections.defaultdict(
                functools.partial(np.zeros, len(sample_sets), dtype=int))
            allele_count[ancestral_state][:] = [
                len(sample_set) for sample_set in sample_sets]
            while (
                    mutation_index < len(mutations)
                    and mutations[mutation_index].site == site_index):
                mutation = mutations[mutation_index]
                allele_count[mutation.derived_state] += count[mutation.node]
                if mutation.parent != -1:
                    parent_allele = mutations[mutation.parent].derived_state
                    allele_count[parent_allele] -= count[mutation.node]
                else:
                    allele_count[ancestral_state] -= count[mutation.node]
                mutation_index += 1

            pos = sites.position[site_index]
            while windows[window_index + 1] <= pos:
                window_index += 1
            assert windows[window_index] <= pos < windows[window_index + 1]
            site_result = result[window_index]

            for allele, c in dict(allele_count).items():
                # Any allele monomorphic across all samples does not
                # contribute to the AFS
                if 0 == c[-1] or c[-1] == ts.num_samples:
                    del allele_count[allele]
            if polarised and ancestral_state in allele_count:
                del allele_count[ancestral_state]

            increment = 1 if polarised else 0.5
            for allele, c in allele_count.items():
                x = tuple(c[:-1])
                if not polarised:
                    x = fold(x, out_dim)
                site_result[x] += increment
            site_index += 1

    if span_normalise:
        for j in range(num_windows):
            span = windows[j + 1] - windows[j]
            result[j] /= span
    return result


def allele_frequency_spectrum(
        ts, sample_sets, windows=None, polarised=False, mode="site",
        span_normalise=True):
    """
    Generalised site frequency spectrum.
    """
    method_map = {
        "site": site_allele_frequency_spectrum,
        "branch": branch_allele_frequency_spectrum}
    return method_map[mode](
        ts, sample_sets, windows=windows, polarised=polarised,
        span_normalise=span_normalise)


class TestAlleleFrequencySpectrum(StatsTestCase, SampleSetStatsMixin):

    # Derived classes define this to get a specific stats mode.
    mode = None

    def verify_single_sample_set(self, ts):
        L = ts.sequence_length
        samples = ts.samples()
        a1 = ts.allele_frequency_spectrum(mode=self.mode)
        a2 = ts.allele_frequency_spectrum([samples], mode=self.mode)
        self.assertArrayEqual(a1, a2)
        for windows in [None, (0, L), (0, L / 2, L)]:
            a1 = ts.allele_frequency_spectrum(mode=self.mode, windows=windows)
            a2 = ts.allele_frequency_spectrum(
                [samples], mode=self.mode, windows=windows)
            self.assertArrayEqual(a1, a2)
        for polarised in [True, False]:
            a1 = ts.allele_frequency_spectrum(mode=self.mode, polarised=polarised)
            a2 = ts.allele_frequency_spectrum(
                [samples], mode=self.mode, polarised=polarised)
            self.assertArrayEqual(a1, a2)
        for span_normalise in [True, False]:
            a1 = ts.allele_frequency_spectrum(
                mode=self.mode, span_normalise=span_normalise)
            a2 = ts.allele_frequency_spectrum(
                [samples], mode=self.mode, span_normalise=span_normalise)
            self.assertArrayEqual(a1, a2)

    def verify_sample_sets(self, ts, sample_sets, windows):
        # print(ts.genotype_matrix())
        # print(ts.draw_text())
        # print("sample_sets = ", sample_sets)
        windows = ts.parse_windows(windows)
        for span_normalise, polarised in itertools.product([True, False], [True, False]):
            sfs1 = naive_allele_frequency_spectrum(
                ts, sample_sets, windows, mode=self.mode, polarised=polarised,
                span_normalise=span_normalise)
            sfs2 = allele_frequency_spectrum(
                ts, sample_sets, windows, mode=self.mode, polarised=polarised,
                span_normalise=span_normalise)
            sfs3 = ts.allele_frequency_spectrum(
                sample_sets, windows, mode=self.mode, polarised=polarised,
                span_normalise=span_normalise)
            self.assertEqual(sfs1.shape[0], len(windows) - 1)
            self.assertEqual(len(sfs1.shape), len(sample_sets) + 1)
            for j, sample_set in enumerate(sample_sets):
                n = 1 + len(sample_set)
                self.assertEqual(sfs1.shape[j + 1], n)

            self.assertEqual(len(sfs1.shape), len(sample_sets) + 1)
            self.assertEqual(sfs1.shape, sfs2.shape)
            self.assertEqual(sfs1.shape, sfs3.shape)
            if not np.allclose(sfs1, sfs3):
                print()
                print("sample sets", sample_sets)
                print("simple", sfs1)
                print("effic ", sfs2)
                print("ts    ", sfs3)
            self.assertArrayAlmostEqual(sfs1, sfs2)
            self.assertArrayAlmostEqual(sfs1, sfs3)


class TestBranchAlleleFrequencySpectrum(
        TestAlleleFrequencySpectrum, TopologyExamplesMixin):
    mode = "branch"

    def test_simple_example(self):
        ts = msprime.simulate(6, recombination_rate=0.1, random_seed=1)
        self.verify_single_sample_set(ts)

        self.verify_sample_sets(ts, [range(6)], [0, 1])
        self.verify_sample_sets(ts, [[0, 1]], [0, 1])
        self.verify_sample_sets(ts, [[0, 1], [2, 3]], [0, 1])
        self.verify_sample_sets(ts, [[0, 1, 2, 3, 4, 5]], [0, 1])
        self.verify_sample_sets(ts, [[0, 1, 2], [3, 4, 5]], [0, 1])
        self.verify_sample_sets(ts, [[0, 1], [2, 3], [4, 5]], [0, 1])


class TestSiteAlleleFrequencySpectrum(
        TestAlleleFrequencySpectrum, MutatedTopologyExamplesMixin):
    mode = "site"

    def test_simple_example(self):
        ts = msprime.simulate(6, mutation_rate=0.2, random_seed=1)
        self.verify_single_sample_set(ts)

        self.verify_sample_sets(ts, [[0]], [0, 1])
        self.verify_sample_sets(ts, [[0, 1, 2, 3, 4, 5]], [0, 1])
        self.verify_sample_sets(ts, [[0, 1, 2], [3, 4, 5]], [0, 1])
        self.verify_sample_sets(ts, [[0, 1], [2, 3], [4, 5]], [0, 1])


class TestBranchAlleleFrequencySpectrumProperties(StatsTestCase, TopologyExamplesMixin):

    def verify(self, ts):
        # If we split by tree, the sum of the AFS should be equal to the
        # tree total branch length in each window
        windows = ts.breakpoints(as_array=True)
        S = ts.samples()
        examples = [
            [S], [S[:1]], [S[:-1]],
            [S[:1], S[1:]], [S[:1], S[:-1]],
        ]
        if len(S) > 2:
            examples += [
                [S[:1], S[2:], S[:3]]
            ]
        # This is the same definition that we use for segregating_sites
        tbl = [
            sum(
                tree.branch_length(u) for u in tree.nodes()
                if 0 < tree.num_samples(u) < ts.num_samples)
            for tree in ts.trees()]
        for polarised in [True, False]:
            for sample_sets in examples:
                afs = ts.allele_frequency_spectrum(
                    sample_sets,  windows=windows, mode="branch", polarised=polarised,
                    span_normalise=True)
                if not polarised:
                    afs *= 2
                afs_sum = [np.sum(window) for window in afs]
                self.assertArrayAlmostEqual(afs_sum, tbl)


############################################
# End of specific stats tests.
############################################


class TestWindowedTreeStat(StatsTestCase):
    """
    Tests that the treewise windowing function defined here has the correct
    behaviour.
    """
    # TODO add more tests here covering the various windowing possibilities.
    def get_tree_sequence(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=1)
        self.assertGreater(ts.num_trees, 3)
        return ts

    def test_all_trees(self):
        ts = self.get_tree_sequence()
        A1 = np.ones((ts.num_trees, 1))
        windows = np.array(list(ts.breakpoints()))
        A2 = windowed_tree_stat(ts, A1, windows)
        # print("breakpoints = ", windows)
        # print(A2)
        self.assertEqual(A1.shape, A2.shape)
        # JK: I don't understand what we're computing here, this normalisation
        # seems pretty weird.
        # for tree in ts.trees():
        #     self.assertAlmostEqual(A2[tree.index, 0], tree.span / ts.sequence_length)

    def test_single_interval(self):
        ts = self.get_tree_sequence()
        A1 = np.ones((ts.num_trees, 1))
        windows = np.array([0, ts.sequence_length])
        A2 = windowed_tree_stat(ts, A1, windows)
        self.assertEqual(A2.shape, (1, 1))
        # TODO: Test output


class TestSampleSets(StatsTestCase):
    """
    Tests that passing sample sets in various ways gets interpreted correctly.
    """

    def get_example_ts(self):
        ts = msprime.simulate(10, mutation_rate=1, recombination_rate=1, random_seed=2)
        assert ts.num_mutations > 0
        return ts

    def test_duplicate_samples(self):
        ts = self.get_example_ts()
        for bad_set in [[1, 1], [1, 2, 1], list(range(10)) + [9]]:
            with self.assertRaises(exceptions.LibraryError):
                ts.diversity([bad_set])
            with self.assertRaises(exceptions.LibraryError):
                ts.divergence([[0, 1], bad_set])
            with self.assertRaises(ValueError):
                ts.sample_count_stat([bad_set], self.identity_f(ts), 1)

    def test_empty_sample_set(self):
        ts = self.get_example_ts()
        with self.assertRaises(ValueError):
            ts.diversity([[]])
        for bad_sample_sets in [[[], []], [[1], []], [[1, 2], [1], []]]:
            with self.assertRaises(ValueError):
                ts.diversity(bad_sample_sets)
            with self.assertRaises(ValueError):
                ts.divergence(bad_sample_sets)
            with self.assertRaises(ValueError):
                ts.sample_count_stat(bad_sample_sets, self.identity_f(ts), 1)

    def test_non_samples(self):
        ts = self.get_example_ts()
        with self.assertRaises(exceptions.LibraryError):
            ts.diversity([[ts.num_samples]])

        with self.assertRaises(exceptions.LibraryError):
            ts.divergence([[ts.num_samples], [1, 2]])

        with self.assertRaises(ValueError):
            ts.sample_count_stat([[ts.num_samples]], self.identity_f(ts), 1)

    def test_span_normalise(self):
        ts = self.get_example_ts()
        sample_sets = [[0, 1], [2, 3, 4], [5, 6]]
        windows = ts.sequence_length * np.random.uniform(size=10)
        windows.sort()
        windows[0] = 0.0
        windows[-1] = ts.sequence_length
        n = np.array([len(u) for u in sample_sets])

        def f(x):
            return x * (x < n)

        # Determine output_dim of the function
        for mode in ('site', 'branch', 'node'):
            sigma1 = ts.sample_count_stat(sample_sets, f, 3, windows=windows)
            sigma2 = ts.sample_count_stat(sample_sets, f, 3, windows=windows,
                                          span_normalise=True)
            sigma3 = ts.sample_count_stat(sample_sets, f, 3, windows=windows,
                                          span_normalise=False)
            denom = np.diff(windows)[:, np.newaxis]
            self.assertEqual(sigma1.shape, sigma2.shape)
            self.assertEqual(sigma1.shape, sigma3.shape)
            self.assertArrayAlmostEqual(sigma1, sigma2)
            self.assertArrayAlmostEqual(sigma1, sigma3 / denom)


class TestSampleSetIndexes(StatsTestCase):
    """
    Tests that we get the correct behaviour from the indexes argument to
    k-way stats functions.
    """
    def get_example_ts(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        self.assertGreater(ts.num_mutations, 0)
        return ts

    def test_2_way_default(self):
        ts = self.get_example_ts()
        sample_sets = np.array_split(ts.samples(), 2)
        S1 = ts.divergence(sample_sets)
        S2 = divergence(ts, sample_sets)[0, 0]
        S3 = ts.divergence(sample_sets, [0, 1])
        self.assertEqual(S1.shape, S2.shape)
        self.assertArrayAlmostEqual(S1, S2)
        self.assertArrayAlmostEqual(S1, S3)
        sample_sets = np.array_split(ts.samples(), 3)
        with self.assertRaises(ValueError):
            _ = ts.divergence(sample_sets)
        with self.assertRaises(ValueError):
            _ = ts.divergence([sample_sets[0]])

    def test_3_way_default(self):
        ts = self.get_example_ts()
        sample_sets = np.array_split(ts.samples(), 3)
        S1 = ts.f3(sample_sets)
        S2 = f3(ts, sample_sets)[0, 0]
        S3 = ts.f3(sample_sets, [0, 1, 2])
        self.assertEqual(S1.shape, S2.shape)
        self.assertArrayAlmostEqual(S1, S2)
        self.assertArrayAlmostEqual(S1, S3)
        sample_sets = np.array_split(ts.samples(), 4)
        with self.assertRaises(ValueError):
            _ = ts.f3(sample_sets)

    def test_4_way_default(self):
        ts = self.get_example_ts()
        sample_sets = np.array_split(ts.samples(), 4)
        S1 = ts.f4(sample_sets)
        S2 = f4(ts, sample_sets)
        S3 = ts.f4(sample_sets, [0, 1, 2, 3])
        self.assertEqual(S1.shape, S3.shape)
        self.assertArrayAlmostEqual(S1, S2)
        self.assertArrayAlmostEqual(S1, S3)
        sample_sets = np.array_split(ts.samples(), 5)
        with self.assertRaises(ValueError):
            _ = ts.f4(sample_sets)

    def test_2_way_combinations(self):
        ts = self.get_example_ts()
        sample_sets = np.array_split(ts.samples(), 4)
        pairs = list(itertools.combinations(range(4), 2))
        for k in range(1, len(pairs)):
            S1 = ts.divergence(sample_sets, pairs[:k])
            S2 = divergence(ts, sample_sets, pairs[:k])[0]
            self.assertEqual(S1.shape[-1], k)
            self.assertEqual(S1.shape, S2.shape)
            self.assertArrayAlmostEqual(S1, S2)

    def test_3_way_combinations(self):
        ts = self.get_example_ts()
        sample_sets = np.array_split(ts.samples(), 5)
        triples = list(itertools.combinations(range(5), 3))
        for k in range(1, len(triples)):
            S1 = ts.Y3(sample_sets, triples[:k])
            S2 = Y3(ts, sample_sets, triples[:k])[0]
            self.assertEqual(S1.shape[-1], k)
            self.assertEqual(S1.shape, S2.shape)
            self.assertArrayAlmostEqual(S1, S2)

    def test_4_way_combinations(self):
        ts = self.get_example_ts()
        sample_sets = np.array_split(ts.samples(), 5)
        quads = list(itertools.combinations(range(5), 4))
        for k in range(1, len(quads)):
            S1 = ts.f4(sample_sets, quads[:k], windows=[0, ts.sequence_length])
            S2 = f4(ts, sample_sets, quads[:k])
            self.assertEqual(S1.shape[-1], k)
            self.assertEqual(S2.shape, S2.shape)
            self.assertArrayAlmostEqual(S1, S2)

    def test_errors(self):
        ts = self.get_example_ts()
        sample_sets = np.array_split(ts.samples(), 2)
        with self.assertRaises(ValueError):
            ts.divergence(sample_sets, indexes=[])
        with self.assertRaises(ValueError):
            ts.divergence(sample_sets, indexes=[(1, 1, 1)])
        with self.assertRaises(exceptions.LibraryError):
            ts.divergence(sample_sets, indexes=[(1, 2)])


class TestGeneralStatInterface(StatsTestCase):
    """
    Tests for the basic interface for general_stats.
    """

    def get_tree_sequence(self):
        ts = msprime.simulate(10, recombination_rate=2,
                              mutation_rate=2, random_seed=1)
        return ts

    def test_function_cannot_update_state(self):
        ts = self.get_tree_sequence()

        def f(x):
            out = x.copy()
            x[:] = 0.0
            return out

        def g(x):
            return x

        x = ts.sample_count_stat(
            [ts.samples()], f, output_dim=1, strict=False, mode="node",
            span_normalise=False)
        y = ts.sample_count_stat(
            [ts.samples()], g, output_dim=1, strict=False, mode="node",
            span_normalise=False)
        self.assertArrayEqual(x, y)

    def test_default_mode(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        sigma1 = ts.general_stat(W, self.identity_f(ts), W.shape[1])
        sigma2 = ts.general_stat(W, self.identity_f(ts), W.shape[1], mode="site")
        self.assertArrayEqual(sigma1, sigma2)

    def test_bad_mode(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        for bad_mode in ["", "MODE", "x" * 8192]:
            with self.assertRaises(ValueError):
                ts.general_stat(W, self.identity_f(ts), W.shape[1], mode=bad_mode)

    def test_bad_window_strings(self):
        ts = self.get_tree_sequence()
        with self.assertRaises(ValueError):
            ts.diversity([ts.samples()], mode="site", windows="abc")
        with self.assertRaises(ValueError):
            ts.diversity([ts.samples()], mode="site", windows="")
        with self.assertRaises(ValueError):
            ts.diversity([ts.samples()], mode="tree", windows="abc")

    def test_bad_summary_function(self):
        ts = self.get_tree_sequence()
        W = np.ones((ts.num_samples, 3))
        with self.assertRaises(ValueError):
            ts.general_stat(W, lambda x: x, 3, windows="sites")
        with self.assertRaises(ValueError):
            ts.general_stat(W, lambda x: np.array([1.0]), 1, windows="sites")

    def test_nonnumpy_summary_function(self):
        ts = self.get_tree_sequence()
        W = np.ones((ts.num_samples, 3))
        sigma1 = ts.general_stat(W, lambda x: [0.0], 1)
        sigma2 = ts.general_stat(W, lambda x: np.array([0.0]), 1)
        self.assertArrayEqual(sigma1, sigma2)


class TestGeneralBranchStats(StatsTestCase):
    """
    Tests for general branch stats (using functions and arbitrary weights)
    """
    def compare_general_stat(self, ts, W, f, windows=None, polarised=False):
        # Determine output_dim of the function
        M = len(f(W[0]))
        sigma1 = naive_branch_general_stat(ts, W, f, windows, polarised=polarised)
        sigma2 = ts.general_stat(W, f, M, windows, polarised=polarised, mode="branch")
        sigma3 = branch_general_stat(ts, W, f, windows, polarised=polarised)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertEqual(sigma1.shape, sigma3.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)
        self.assertArrayAlmostEqual(sigma1, sigma3)
        return sigma1

    def test_simple_identity_f_w_zeros(self):
        ts = msprime.simulate(12, recombination_rate=3, random_seed=2)
        W = np.zeros((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.compare_general_stat(
                        ts, W, self.identity_f(ts),
                        windows="trees", polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_simple_identity_f_w_ones(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        sigma = self.compare_general_stat(ts, W, self.identity_f(ts), windows="trees",
                                          polarised=True)
        self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))
        # A W of 1 for every node and identity f counts the samples in the subtree
        # if polarised is True.
        for tree in ts.trees():
            s = sum(tree.num_samples(u) * tree.branch_length(u) for u in tree.nodes())
            self.assertTrue(np.allclose(sigma[tree.index], s))

    def test_simple_cumsum_f_w_ones(self):
        ts = msprime.simulate(13, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 8))
        for polarised in [True, False]:
            sigma = self.compare_general_stat(
                ts, W, self.cumsum_f(ts), windows="trees", polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, W.shape[1]))

    def test_simple_cumsum_f_w_ones_many_windows(self):
        ts = msprime.simulate(15, recombination_rate=3, random_seed=3)
        self.assertGreater(ts.num_trees, 3)
        windows = np.linspace(0, ts.sequence_length, num=ts.num_trees * 10)
        W = np.ones((ts.num_samples, 3))
        sigma = self.compare_general_stat(ts, W, self.cumsum_f(ts), windows=windows)
        self.assertEqual(sigma.shape, (windows.shape[0] - 1, W.shape[1]))

    def test_windows_equal_to_ts_breakpoints(self):
        ts = msprime.simulate(14, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 1))
        for polarised in [True, False]:
            sigma_no_windows = self.compare_general_stat(
                ts, W, self.cumsum_f(ts), windows="trees", polarised=polarised)
            self.assertEqual(sigma_no_windows.shape, (ts.num_trees, W.shape[1]))
            sigma_windows = self.compare_general_stat(
                ts, W, self.cumsum_f(ts), windows=ts.breakpoints(as_array=True),
                polarised=polarised)
            self.assertEqual(sigma_windows.shape, sigma_no_windows.shape)
            self.assertTrue(np.allclose(sigma_windows.shape, sigma_no_windows.shape))

    def test_single_tree_windows(self):
        ts = msprime.simulate(15, random_seed=2, length=100)
        W = np.ones((ts.num_samples, 2))
        f = self.sum_f(ts)
        # for num_windows in range(1, 10):
        for num_windows in [2]:
            windows = np.linspace(0, ts.sequence_length, num=num_windows + 1)
            sigma = self.compare_general_stat(ts, W, f, windows)
            self.assertEqual(sigma.shape, (num_windows, 1))

    def test_simple_identity_f_w_zeros_windows(self):
        ts = msprime.simulate(15, recombination_rate=3, random_seed=2)
        W = np.zeros((ts.num_samples, 3))
        f = self.identity_f(ts)
        windows = np.linspace(0, ts.sequence_length, num=11)
        for polarised in [True, False]:
            sigma = self.compare_general_stat(ts, W, f, windows,
                                              polarised=polarised)
            self.assertEqual(sigma.shape, (10, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))


class TestGeneralSiteStats(StatsTestCase):
    """
    Tests for general site stats (using functions and arbitrary weights)
    """
    def compare_general_stat(self, ts, W, f, windows=None, polarised=False):
        # Determine output_dim of the function
        M = len(f(W[0]))
        sigma1 = naive_site_general_stat(ts, W, f, windows, polarised=polarised)
        sigma2 = ts.general_stat(W, f, M, windows, polarised=polarised, mode="site")
        sigma3 = site_general_stat(ts, W, f, windows, polarised=polarised)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertEqual(sigma1.shape, sigma3.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)
        self.assertArrayAlmostEqual(sigma1, sigma3)
        return sigma1

    def test_identity_f_W_0_multiple_alleles(self):
        ts = msprime.simulate(20, recombination_rate=0, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.zeros((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.compare_general_stat(
                        ts, W, self.identity_f(ts),
                        windows="sites", polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_identity_f_W_0_multiple_alleles_windows(self):
        ts = msprime.simulate(34, recombination_rate=0, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.zeros((ts.num_samples, 3))
        windows = np.linspace(0, 1, num=11)
        for polarised in [True, False]:
            sigma = self.compare_general_stat(
                ts, W, self.identity_f(ts), windows=windows, polarised=polarised)
            self.assertEqual(sigma.shape, (windows.shape[0] - 1, W.shape[1]))
            self.assertTrue(np.all(sigma == 0))

    def test_cumsum_f_W_1_multiple_alleles(self):
        ts = msprime.simulate(3, recombination_rate=2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        W = np.ones((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.compare_general_stat(ts, W, self.cumsum_f(ts),
                                              windows="sites", polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))

    def test_cumsum_f_W_1_two_alleles(self):
        ts = msprime.simulate(33, recombination_rate=1, mutation_rate=2, random_seed=1)
        W = np.ones((ts.num_samples, 5))
        for polarised in [True, False]:
            sigma = self.compare_general_stat(
                ts, W, self.cumsum_f(ts), windows="sites", polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_sites, W.shape[1]))


class TestGeneralNodeStats(StatsTestCase):
    """
    Tests for general node stats (using functions and arbitrary weights)
    """
    def compare_general_stat(self, ts, W, f, windows=None, polarised=False):
        # Determine output_dim of the function
        M = len(f(W[0]))
        sigma1 = naive_node_general_stat(ts, W, f, windows, polarised=polarised)
        sigma2 = ts.general_stat(W, f, M, windows, polarised=polarised, mode="node")
        sigma3 = node_general_stat(ts, W, f, windows, polarised=polarised)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertEqual(sigma1.shape, sigma3.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)
        self.assertArrayAlmostEqual(sigma1, sigma3)
        return sigma1

    def test_simple_sum_f_w_zeros(self):
        ts = msprime.simulate(12, recombination_rate=3, random_seed=2)
        W = np.zeros((ts.num_samples, 3))
        for polarised in [True, False]:
            sigma = self.compare_general_stat(
                ts, W, self.identity_f(ts), windows="trees", polarised=polarised)
            self.assertEqual(sigma.shape, (ts.num_trees, ts.num_nodes, 3))
            self.assertTrue(np.all(sigma == 0))

    def test_simple_sum_f_w_ones(self):
        ts = msprime.simulate(44, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        f = self.sum_f(ts)
        sigma = self.compare_general_stat(
            ts, W, f, windows="trees", polarised=True)
        self.assertEqual(sigma.shape, (ts.num_trees, ts.num_nodes, 1))
        # Drop the last dimension
        sigma = sigma.reshape((ts.num_trees, ts.num_nodes))
        # A W of 1 for every node and f(x)=sum(x) counts the samples in the subtree
        # times 2 if polarised is True.
        for tree in ts.trees():
            s = np.array([tree.num_samples(u) if tree.num_samples(u) < ts.num_samples
                          else 0 for u in range(ts.num_nodes)])
            self.assertArrayAlmostEqual(sigma[tree.index], 2*s)

    def test_simple_sum_f_w_ones_notstrict(self):
        ts = msprime.simulate(44, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        sigma = ts.general_stat(W, lambda x: np.array([np.sum(x)]), 1, windows="trees",
                                polarised=True, mode="node", strict=False)
        self.assertEqual(sigma.shape, (ts.num_trees, ts.num_nodes, 1))
        # Drop the last dimension
        sigma = sigma.reshape((ts.num_trees, ts.num_nodes))
        # A W of 1 for every node and f(x)=sum(x) counts the samples in the subtree
        # times 2 if polarised is True.
        for tree in ts.trees():
            s = np.array([tree.num_samples(u) for u in range(ts.num_nodes)])
            self.assertArrayAlmostEqual(sigma[tree.index], 2*s)

    def test_small_tree_windows_polarised(self):
        ts = msprime.simulate(4, recombination_rate=0.5, random_seed=2)
        self.assertGreater(ts.num_trees, 1)
        W = np.ones((ts.num_samples, 1))
        sigma = self.compare_general_stat(
            ts, W, self.cumsum_f(ts), windows=ts.breakpoints(as_array=True),
            polarised=True)
        self.assertEqual(sigma.shape, (ts.num_trees, ts.num_nodes, 1))

    def test_one_window_polarised(self):
        ts = msprime.simulate(4, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 1))
        sigma = self.compare_general_stat(
            ts, W, self.cumsum_f(ts), windows=[0, ts.sequence_length],
            polarised=True)
        self.assertEqual(sigma.shape, (1, ts.num_nodes, W.shape[1]))

    def test_one_window_unpolarised(self):
        ts = msprime.simulate(4, recombination_rate=1, random_seed=2)
        W = np.ones((ts.num_samples, 2))
        sigma = self.compare_general_stat(
            ts, W, self.cumsum_f(ts), windows=[0, ts.sequence_length],
            polarised=False)
        self.assertEqual(sigma.shape, (1, ts.num_nodes, 2))

    def test_many_windows(self):
        ts = msprime.simulate(24, recombination_rate=3, random_seed=2)
        W = np.ones((ts.num_samples, 3))
        for k in [1, ts.num_trees // 2, ts.num_trees, ts.num_trees * 2]:
            windows = np.linspace(0, 1, num=k + 1)
            for polarised in [True]:
                sigma = self.compare_general_stat(
                    ts, W, self.cumsum_f(ts), windows=windows, polarised=polarised)
            self.assertEqual(sigma.shape, (k, ts.num_nodes, 3))

    def test_one_tree(self):
        ts = msprime.simulate(10, random_seed=3)
        W = np.ones((ts.num_samples, 2))
        f = self.sum_f(ts, k=2)
        sigma = self.compare_general_stat(
            ts, W, f, windows=[0, 1], polarised=True)
        self.assertEqual(sigma.shape, (1, ts.num_nodes, 2))
        # A W of 1 for every node and f(x)=sum(x) counts the samples in the subtree
        # times 2 if polarised is True.
        tree = ts.first()
        s = np.array([tree.num_samples(u) if tree.num_samples(u) < ts.num_samples else 0
                      for u in range(ts.num_nodes)])
        self.assertArrayAlmostEqual(sigma[tree.index, :, 0], 2 * s)
        self.assertArrayAlmostEqual(sigma[tree.index, :, 1], 2 * s)


##############################
# Trait covariance
##############################

def covsq(x, y):
    cov = np.dot(x - np.mean(x), y - np.mean(y)) / (len(x) - 1)
    return cov * cov


def corsq(x, y):
    vx = covsq(x, x)
    vy = covsq(y, y)
    # sqrt is because vx and vy are *squared* variances
    return covsq(x, y) / np.sqrt(vx * vy)


def site_trait_covariance(ts, W, windows=None, span_normalise=True):
    """
    For each site, computes the covariance between the columns of W and the genotypes.
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, K))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True)
        site_positions = [x.position for x in ts.sites()]
        for i in range(K):
            w = W[:, i].copy()
            w -= np.mean(w)
            S = 0
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    hX = haps[k]
                    alleles = set(hX)
                    for a in alleles:
                        S += covsq(w, hX == a) / 2
            if site_in_window:
                out[j, i] = S
                if span_normalise:
                    out[j, i] /= (end - begin)
    return out


def branch_trait_covariance(ts, W, windows=None, span_normalise=True):
    """
    For each branch, computes the covariance between the columns of W and the split
    induced by the branch, multiplied by the length of the branch.
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, K))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i in range(K):
            w = W[:, i].copy()
            w -= np.mean(w)
            S = 0
            has_trees = False
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                SS = 0
                for u in range(ts.num_nodes):
                    below = np.in1d(samples, list(tr.samples(u)))
                    branch_length = tr.branch_length(u)
                    SS += covsq(w, below) * branch_length
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            if has_trees:
                out[j, i] = S
                if span_normalise:
                    out[j, i] /= (end - begin)
    return out


def node_trait_covariance(ts, W, windows=None, span_normalise=True):
    """
    For each node, computes the covariance between the columns of W and the split
    induced by above/below the node.
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, ts.num_nodes, K))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i in range(K):
            w = W[:, i].copy()
            w -= np.mean(w)
            S = np.zeros(ts.num_nodes)
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in range(ts.num_nodes):
                    below = np.in1d(samples, list(tr.samples(u)))
                    SS[u] += covsq(w, below)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            out[j, :, i] = S
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def trait_covariance(ts, W, windows=None, mode="site", span_normalise=True):
    method_map = {
        "site": site_trait_covariance,
        "node": node_trait_covariance,
        "branch": branch_trait_covariance}
    return method_map[mode](ts, W, windows=windows,
                            span_normalise=span_normalise)


class TestTraitCovariance(StatsTestCase, WeightStatsMixin):
    # Derived classes define this to get a specific stats mode.
    mode = None

    def get_example_ts(self):
        ts = msprime.simulate(10, mutation_rate=1, recombination_rate=2, random_seed=1)
        self.assertGreater(ts.num_mutations, 0)
        return ts

    def transform_weights(self, W):
        """
        Need centered weights to compare to general stats.
        """
        W -= np.mean(W, axis=0)
        return W

    def verify_weighted_stat(self, ts, W, windows):
        n = W.shape[0]

        def f(x):
            return (x ** 2) / (2 * (n - 1) * (n - 1))

        self.verify_definition(
            ts, W, windows, f, ts.trait_covariance, trait_covariance)

    def verify_interface(self, ts, ts_method):
        W = np.array([np.arange(ts.num_samples)]).T
        sigma1 = ts_method(W, mode=self.mode)
        sigma2 = ts_method(W, windows=None, mode=self.mode)
        sigma3 = ts_method(W, windows=[0.0, ts.sequence_length], mode=self.mode)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertArrayAlmostEqual(sigma1, sigma2)
        self.assertArrayAlmostEqual(sigma1, sigma3[0])

    def verify_centering(self, ts, method, ts_method):
        # Since weights are mean-centered, adding a constant shouldn't change anything.
        ts = self.get_example_ts()
        for W, windows in subset_combos(
                self.example_weights(ts), example_windows(ts), p=0.1):
            shift = np.arange(1, W.shape[1] + 1)
            sigma1 = ts_method(W, windows=windows, mode=self.mode)
            sigma2 = ts_method(W + shift, windows=windows, mode=self.mode)
            sigma3 = method(ts, W, windows=windows, mode=self.mode)
            sigma4 = method(ts, W + shift, windows=windows, mode=self.mode)
            self.assertEqual(sigma1.shape, sigma2.shape)
            self.assertEqual(sigma1.shape, sigma3.shape)
            self.assertEqual(sigma1.shape, sigma4.shape)
            self.assertArrayAlmostEqual(sigma1, sigma2)
            self.assertArrayAlmostEqual(sigma1, sigma3)
            self.assertArrayAlmostEqual(sigma1, sigma4)


class TraitCovarianceMixin(object):

    def test_interface(self):
        ts = self.get_example_ts()
        self.verify_interface(ts, ts.trait_covariance)

    def test_normalisation(self):
        ts = self.get_example_ts()
        self.verify_centering(ts, trait_covariance, ts.trait_covariance)

    def test_errors(self):
        ts = self.get_example_ts()
        W = np.ones((ts.num_samples, 2))
        # W must have the right number of rows
        self.assertRaises(ValueError, ts.trait_correlation, W[1:, :])


class TestBranchTraitCovariance(
        TestTraitCovariance, TopologyExamplesMixin, TraitCovarianceMixin):
    mode = "branch"


class TestNodeTraitCovariance(
        TestTraitCovariance, TopologyExamplesMixin, TraitCovarianceMixin):
    mode = "node"


class TestSiteTraitCovariance(
        TestTraitCovariance, MutatedTopologyExamplesMixin,
        TraitCovarianceMixin):
    mode = "site"


##############################
# Trait correlation
##############################


def site_trait_correlation(ts, W, windows=None, span_normalise=True):
    """
    For each site, computes the correlation between the columns of W and the genotypes.
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, K))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True)
        site_positions = [x.position for x in ts.sites()]
        for i in range(K):
            w = W[:, i].copy()
            w -= np.mean(w)
            w /= np.std(w) * np.sqrt(len(w) / (len(w) - 1))
            S = 0
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    hX = haps[k]
                    alleles = set(hX)
                    for a in alleles:
                        p = np.mean(hX == a)
                        if p > 0 and p < 1:
                            # S += sum(w[hX == a])**2 / (2 * (p * (1 - p)))
                            S += corsq(w, hX == a) / 2
            if site_in_window:
                out[j, i] = S
                if span_normalise:
                    out[j, i] /= (end - begin)
    return out


def branch_trait_correlation(ts, W, windows=None, span_normalise=True):
    """
    For each branch, computes the correlation between the columns of W and the split
    induced by the branch, multiplied by the length of the branch.
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, K))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i in range(K):
            w = W[:, i].copy()
            w -= np.mean(w)
            w /= np.std(w) * np.sqrt(len(w) / (len(w) - 1))
            S = 0
            has_trees = False
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                SS = 0
                for u in range(ts.num_nodes):
                    below = np.in1d(samples, list(tr.samples(u)))
                    p = np.mean(below)
                    if p > 0 and p < 1:
                        branch_length = tr.branch_length(u)
                        # SS += ((sum(w[below])**2 +
                        #         sum(w[np.logical_not(below)])**2) * branch_length
                        #        / (2 * (p * (1 - p))))
                        SS += corsq(w, below) * branch_length
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            if has_trees:
                out[j, i] = S
                if span_normalise:
                    out[j, i] /= (end - begin)
    return out


def node_trait_correlation(ts, W, windows=None, span_normalise=True):
    """
    For each node, computes the correlation between the columns of W and the split
    induced by above/below the node.
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, ts.num_nodes, K))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i in range(K):
            w = W[:, i].copy()
            w -= np.mean(w)
            w /= np.std(w) * np.sqrt(len(w) / (len(w) - 1))
            S = np.zeros(ts.num_nodes)
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in range(ts.num_nodes):
                    below = np.in1d(samples, list(tr.samples(u)))
                    p = np.mean(below)
                    if p > 0 and p < 1:
                        # SS[u] += sum(w[below])**2 / 2
                        # SS[u] += sum(w[np.logical_not(below)])**2 / 2
                        # SS[u] /= (p * (1 - p))
                        SS[u] += corsq(w, below)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            out[j, :, i] = S
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def trait_correlation(ts, W, windows=None, mode="site", span_normalise=True):
    method_map = {
        "site": site_trait_correlation,
        "node": node_trait_correlation,
        "branch": branch_trait_correlation}
    return method_map[mode](ts, W, windows=windows,
                            span_normalise=span_normalise)


class TestTraitCorrelation(TestTraitCovariance):
    # Derived classes define this to get a specific stats mode.
    mode = None

    def transform_weights(self, W):
        """
        Need standardised weights to compare to general stats,
        and also an extra column to compute allele frequencies.
        """
        W -= np.mean(W, axis=0)
        n = W.shape[0]
        with suppress_division_by_zero_warning():
            W /= np.std(W, axis=0) * np.sqrt(n / (n - 1))
        return np.column_stack((W, np.ones(W.shape[0])/W.shape[0]))

    def verify_weighted_stat(self, ts, W, windows):
        n = W.shape[0]

        def f(x):
            p = x[-1]
            if p > 0 and p < 1:
                return (x[:-1] ** 2) / (2 * (p * (1 - p)) * n * (n - 1))
            else:
                return x[:-1] * 0.0

        self.verify_definition(
            ts, W, windows, f, ts.trait_correlation, trait_correlation)

    def test_errors(self):
        ts = self.get_example_ts()
        # columns of W must have positive SD
        W = np.ones((ts.num_samples, 2))
        self.assertRaises(ValueError, ts.trait_correlation, W)
        # W must have the right number of rows
        self.assertRaises(ValueError, ts.trait_correlation, W[1:, :])

    def verify_standardising(self, ts, method, ts_method):
        """
        Since weights are standardised, multiplying by a constant shouldn't
        change anything.
        """
        for W, windows in subset_combos(
                self.example_weights(ts), example_windows(ts), p=0.1):
            scale = np.arange(1, W.shape[1] + 1)
            sigma1 = ts_method(W, windows=windows, mode=self.mode)
            sigma2 = ts_method(W * scale, windows=windows, mode=self.mode)
            sigma3 = method(ts, W, windows=windows, mode=self.mode)
            sigma4 = method(ts, W * scale, windows=windows, mode=self.mode)
            self.assertEqual(sigma1.shape, sigma2.shape)
            self.assertArrayAlmostEqual(sigma1, sigma2)
            self.assertArrayAlmostEqual(sigma1, sigma3)
            self.assertArrayAlmostEqual(sigma1, sigma4)


class TraitCorrelationMixin(object):

    def test_interface(self):
        ts = self.get_example_ts()
        self.verify_interface(ts, ts.trait_correlation)

    def test_normalisation(self):
        ts = self.get_example_ts()
        self.verify_centering(ts, trait_correlation, ts.trait_correlation)
        self.verify_standardising(
                ts, trait_correlation, ts.trait_correlation)


class TestBranchTraitCorrelation(
        TestTraitCorrelation, TopologyExamplesMixin, TraitCorrelationMixin):
    mode = "branch"


class TestNodeTraitCorrelation(
        TestTraitCorrelation, TopologyExamplesMixin, TraitCorrelationMixin):
    mode = "node"


class TestSiteTraitCorrelation(
        TestTraitCorrelation, MutatedTopologyExamplesMixin,
        TraitCorrelationMixin):
    mode = "site"


##############################
# Trait regression
##############################


def regression(y, x, z):
    """
    Returns the squared coefficient of x in the least-squares linear regression
    :   y ~ x + z
    where x and y are vectors and z is a matrix.
    Note that if z is None then the output is
      cor(x, y) * sd(y) / sd(x) = cov(x, y) / (sd(x) ** 2) .
    """
    # add the constant vector to z
    if z is None:
        z = np.ones((len(x), 1))
    else:
        xz = np.column_stack([z, np.ones((len(x), 1))])
        if np.linalg.matrix_rank(xz) == xz.shape[1]:
            z = xz
    xz = np.column_stack([x, z])
    if np.linalg.matrix_rank(xz) < xz.shape[1]:
        return 0.0
    else:
        coefs, _, _, _ = np.linalg.lstsq(xz, y, rcond=None)
        return coefs[0] * coefs[0]


def site_trait_regression(ts, W, Z, windows=None, span_normalise=True):
    """
    For each site, and for each trait w (column of W), computes the coefficient
    of site in the linear regression:
      w ~ site + Z
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, K))
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        haps = ts.genotype_matrix(impute_missing_data=True)
        site_positions = [x.position for x in ts.sites()]
        for i in range(K):
            w = W[:, i]
            S = 0
            site_in_window = False
            for k in range(ts.num_sites):
                if (site_positions[k] >= begin) and (site_positions[k] < end):
                    site_in_window = True
                    hX = haps[k]
                    alleles = set(hX)
                    for a in alleles:
                        p = np.mean(hX == a)
                        if p > 0 and p < 1:
                            S += regression(w, hX == a, Z) / 2
            if site_in_window:
                out[j, i] = S
                if span_normalise:
                    out[j, i] /= (end - begin)
    return out


def branch_trait_regression(ts, W, Z, windows=None, span_normalise=True):
    """
    For each branch, computes the regression of each column of W onto the split
    induced by the branch and the covariates Z, multiplied by the length of the branch,
    returning the squared coefficient of the column of W.
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, K))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i in range(K):
            w = W[:, i]
            S = 0
            has_trees = False
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                if tr.total_branch_length > 0:
                    has_trees = True
                SS = 0
                for u in range(ts.num_nodes):
                    below = np.in1d(samples, list(tr.samples(u)))
                    branch_length = tr.branch_length(u)
                    SS += regression(w, below, Z) * branch_length
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            if has_trees:
                out[j, i] = S
                if span_normalise:
                    out[j, i] /= (end - begin)
    return out


def node_trait_regression(ts, W, Z, windows=None, span_normalise=True):
    """
    For each node, computes the regression of each columns of W on the split
    induced by above/below the node and the covariates Z, returning the squared
    coefficient of the column of W.
    """
    windows = ts.parse_windows(windows)
    n, K = W.shape
    assert(n == ts.num_samples)
    out = np.zeros((len(windows) - 1, ts.num_nodes, K))
    samples = ts.samples()
    for j in range(len(windows) - 1):
        begin = windows[j]
        end = windows[j + 1]
        for i in range(K):
            w = W[:, i]
            S = np.zeros(ts.num_nodes)
            for tr in ts.trees():
                if tr.interval[1] <= begin:
                    continue
                if tr.interval[0] >= end:
                    break
                SS = np.zeros(ts.num_nodes)
                for u in range(ts.num_nodes):
                    below = np.in1d(samples, list(tr.samples(u)))
                    SS[u] += regression(w, below, Z)
                S += SS*(min(end, tr.interval[1]) - max(begin, tr.interval[0]))
            out[j, :, i] = S
            if span_normalise:
                out[j, :, i] /= (end - begin)
    return out


def trait_regression(ts, W, Z, windows=None, mode="site", span_normalise=True):
    method_map = {
        "site": site_trait_regression,
        "node": node_trait_regression,
        "branch": branch_trait_regression}
    return method_map[mode](ts, W, Z, windows=windows,
                            span_normalise=span_normalise)


class TestTraitRegression(StatsTestCase, WeightStatsMixin):
    # Derived classes define this to get a specific stats mode.
    mode = None

    def get_example_ts(self):
        ts = msprime.simulate(10, mutation_rate=1, recombination_rate=2, random_seed=1)
        self.assertGreater(ts.num_mutations, 0)
        return ts

    def example_covariates(self, ts):
        N = ts.num_samples
        for k in [1, 2, 5]:
            k = min(k, ts.num_samples)
            Z = np.ones((N, k))
            Z[1, :] = np.arange(k, 2*k)
            yield Z
            for j in range(k):
                Z[:, j] = np.random.normal(0, 1, N)
            yield Z

    def transform_weights(self, W, Z):
        n = W.shape[0]
        return np.column_stack([W, Z, np.ones((n, 1))])

    def transform_covariates(self, Z):
        tZ = np.column_stack([Z, np.ones((Z.shape[0], 1))])
        if np.linalg.matrix_rank(tZ) == tZ.shape[1]:
            Z = tZ
        assert(np.linalg.matrix_rank(Z) == Z.shape[1])
        K = np.linalg.cholesky(np.matmul(Z.T, Z)).T
        Z = np.matmul(Z, np.linalg.inv(K))
        return Z

    def verify(self, ts):
        for W, Z, windows in subset_combos(
                self.example_weights(ts),
                self.example_covariates(ts),
                example_windows(ts), p=0.04):
            self.verify_trait_regression(
                    ts, W, Z, windows=windows)

    def verify_trait_regression(self, ts, W, Z, windows):
        n, result_dim = W.shape
        tZ = self.transform_covariates(Z)
        n, k = tZ.shape
        V = np.matmul(W.T, tZ)

        def f(x):
            m = x[-1]
            a = np.zeros(result_dim)
            for i in range(result_dim):
                # print("i=", i, "result_dim=", result_dim, "m=", m, "x=", x)
                # print("V=", V)
                if m > 0 and m < ts.num_samples:
                    v = V[i, :]
                    a[i] = x[i]
                    denom = m
                    for j in range(k):
                        xx = x[result_dim + j]
                        a[i] -= xx * v[j]
                        denom -= xx * xx
                    if abs(denom) < 1e-8:
                        a[i] = 0.0
                    else:
                        a[i] /= denom
                else:
                    a[i] = 0.0
            # print("out", a*a/2)
            return a * a / 2

        # general_stat will need Z added, and an extra column for m
        gW = self.transform_weights(W, tZ)

        def wrapped_summary_func(x):
            with suppress_division_by_zero_warning():
                return f(x)

        # Determine output_dim of the function
        M = len(wrapped_summary_func(gW[0]))
        for sn in [True, False]:
            sigma1 = ts.general_stat(gW, wrapped_summary_func, M,
                                     windows, mode=self.mode,
                                     span_normalise=sn)
            sigma2 = general_stat(ts, gW, wrapped_summary_func, windows, mode=self.mode,
                                  span_normalise=sn)
            sigma3 = ts.trait_regression(W, Z, windows=windows, mode=self.mode,
                                         span_normalise=sn)
            sigma4 = trait_regression(ts, W, Z, windows=windows, mode=self.mode,
                                      span_normalise=sn)

            self.assertEqual(sigma1.shape, sigma2.shape)
            self.assertEqual(sigma1.shape, sigma3.shape)
            self.assertEqual(sigma1.shape, sigma4.shape)
            self.assertArrayAlmostEqual(sigma1, sigma2)
            self.assertArrayAlmostEqual(sigma1, sigma3)
            self.assertArrayAlmostEqual(sigma1, sigma4)


class TraitRegressionMixin(object):

    def test_interface(self):
        ts = self.get_example_ts()
        W = np.array([np.arange(ts.num_samples)]).T
        Z = np.ones((ts.num_samples, 1))
        sigma1 = ts.trait_regression(W, Z=Z, mode=self.mode)
        sigma2 = ts.trait_regression(W, Z=Z, windows=None, mode=self.mode)
        sigma3 = ts.trait_regression(W, Z=Z, windows=[0.0, ts.sequence_length],
                                     mode=self.mode)
        sigma4 = ts.trait_regression(W, Z=None, windows=[0.0, ts.sequence_length],
                                     mode=self.mode)
        self.assertEqual(sigma1.shape, sigma2.shape)
        self.assertEqual(sigma3.shape[0], 1)
        self.assertEqual(sigma1.shape, sigma3.shape[1:])
        self.assertEqual(sigma1.shape, sigma4.shape[1:])
        self.assertArrayAlmostEqual(sigma1, sigma2)
        self.assertArrayAlmostEqual(sigma1, sigma3[0])
        self.assertArrayAlmostEqual(sigma1, sigma4[0])

    def test_errors(self):
        ts = self.get_example_ts()
        W = np.array([np.arange(ts.num_samples)]).T
        Z = np.ones((ts.num_samples, 1))
        # singular covariates
        self.assertRaises(ValueError, ts.trait_regression, W,
                          np.ones((ts.num_samples, 2)), mode=self.mode)
        # wrong dimensions of W
        self.assertRaises(ValueError, ts.trait_regression, W[1:, :], Z, mode=self.mode)
        # wrong dimensions of Z
        self.assertRaises(ValueError, ts.trait_regression, W, Z[1:, :], mode=self.mode)


class TestBranchTraitRegression(
        TestTraitRegression, TopologyExamplesMixin, TraitRegressionMixin):
    mode = "branch"


class TestNodeTraitRegression(
        TestTraitRegression, TopologyExamplesMixin, TraitRegressionMixin):
    mode = "node"


class TestSiteTraitRegression(
        TestTraitRegression, MutatedTopologyExamplesMixin,
        TraitRegressionMixin):
    mode = "site"

##############################
# Sample set statistics
##############################


@unittest.skip("Broken - need to port tests")
class SampleSetStatTestCase(StatsTestCase):
    """
    Provides checks for testing of sample set-based statistics.  Actual testing
    is done by derived classes, which should have attributes `stat_type` and `rng`.
    This works by using parallel structure between different statistic "modes",
    in tree sequence methods (with stat_type=X) and python stat calculators as
    implemented here (with StatCalculator.X).
    """

    random_seed = 123456

    def compare_sfs(self, ts, tree_fn, sample_sets, tsc_fn):
        for sample_set in sample_sets:
            windows = [k * ts.sequence_length / 20 for k in
                       [0] + sorted(self.rng.sample(range(1, 20), 4)) + [20]]
            win_args = [{'begin': windows[i], 'end': windows[i+1]}
                        for i in range(len(windows)-1)]
            tree_vals = [tree_fn(sample_set, **b) for b in win_args]

            tsc_vals = tsc_fn(sample_set, windows)
            self.assertEqual(len(tsc_vals), len(windows) - 1)
            for i in range(len(windows) - 1):
                self.assertListAlmostEqual(tsc_vals[i], tree_vals[i])

    def check_sfs_interface(self, ts):
        samples = ts.samples()

        # empty sample sets will raise an error
        self.assertRaises(ValueError, ts.site_frequency_spectrum, [],
                          self.stat_type)
        # sample_sets must be lists without repeated elements
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          [samples[2], samples[2]], self.stat_type)
        # and must all be samples
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          [samples[0], max(samples)+1], self.stat_type)
        # windows must start at 0.0, be increasing, and extend to the end
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          samples[0:2], [0.1, ts.sequence_length],
                          self.stat_type)
        self.assertRaises(ValueError, ts.site_frequency_spectrum,
                          samples[0:2], [0.0, 0.8*ts.sequence_length],
                          self.stat_type)
        self.assertRaises(ValueError, ts.site_frequency_spectrum, samples[0:2],
                          [0.0, 0.8*ts.sequence_length, 0.4*ts.sequence_length,
                           ts.sequence_length], self.stat_type)

    def check_sfs(self, ts):
        # check site frequency spectrum
        self.check_sfs_interface(ts)
        A = [self.rng.sample(list(ts.samples()), 2),
             self.rng.sample(list(ts.samples()), 4),
             self.rng.sample(list(ts.samples()), 8),
             self.rng.sample(list(ts.samples()), 10),
             self.rng.sample(list(ts.samples()), 12)]
        py_tsc = self.py_stat_class(ts)

        self.compare_sfs(ts, py_tsc.site_frequency_spectrum, A,
                         ts.site_frequency_spectrum)


class BranchSampleSetStatsTestCase(SampleSetStatTestCase):
    """
    Tests of branch statistic computation with sample sets,
    mostly running the checks in SampleSetStatTestCase.
    """

    def setUp(self):
        self.rng = random.Random(self.random_seed)
        self.stat_type = "branch"

    def get_ts(self):
        for N in [12, 15, 20]:
            yield msprime.simulate(N, random_seed=self.random_seed,
                                   recombination_rate=10)

    @unittest.skip("Skipping SFS.")
    def test_sfs_interface(self):
        ts = msprime.simulate(10)
        tsc = tskit.BranchStatCalculator(ts)

        # Duplicated samples raise an error
        self.assertRaises(ValueError, tsc.site_frequency_spectrum, [1, 1])
        self.assertRaises(ValueError, tsc.site_frequency_spectrum, [])
        self.assertRaises(ValueError, tsc.site_frequency_spectrum, [0, 11])
        # Check for bad windows
        for bad_start in [-1, 1, 1e-7]:
            self.assertRaises(
                ValueError, tsc.site_frequency_spectrum, [1, 2],
                [bad_start, ts.sequence_length])
        for bad_end in [0, ts.sequence_length - 1, ts.sequence_length + 1]:
            self.assertRaises(
                ValueError, tsc.site_frequency_spectrum, [1, 2],
                [0, bad_end])
        # Windows must be increasing.
        self.assertRaises(
            ValueError, tsc.site_frequency_spectrum, [1, 2], [0, 1, 1])

    @unittest.skip("No SFS.")
    def test_branch_sfs(self):
        for ts in self.get_ts():
            self.check_sfs(ts)


class SpecificTreesTestCase(StatsTestCase):
    """
    Some particular cases, that are easy to see and debug.
    """
    seed = 21

    def test_case_1(self):
        # With mutations:
        #
        # 1.0          6
        # 0.7         / \                                    5
        #            /   X                                  / \
        # 0.5       X     4                4               /   4
        #          /     / \              / \             /   X X
        # 0.4     X     X   \            X   3           X   /   \
        #        /     /     X          /   / X         /   /     \
        # 0.0   0     1       2        1   0   2       0   1       2
        #          (0.0, 0.2),        (0.2, 0.8),       (0.8, 1.0)
        #
        # haplotypes:
        # site:  0   1 2  3 4         5    6              7   8 9
        # 0:     0   1 0  1 0         0    0              1   0 0
        # 1:     1   0 0  0 1         1    0              0   1 0
        # 2:     1   0 1  0 0         0    1              0   0 1
        branch_true_diversity_01 = 2*(1 * (0.2-0) +
                                      0.5 * (0.8-0.2) + 0.7 * (1.0-0.8))
        branch_true_diversity_02 = 2*(1 * (0.2-0) +
                                      0.4 * (0.8-0.2) + 0.7 * (1.0-0.8))
        branch_true_diversity_12 = 2*(0.5 * (0.2-0) +
                                      0.5 * (0.8-0.2) + 0.5 * (1.0-0.8))
        branch_true_Y = 0.2*(1 + 0.5) + 0.6*(0.4) + 0.2*(0.7+0.2)
        site_true_Y = 3 + 0 + 1
        node_true_diversity_012 = np.array([
                0.2 * np.array([2, 2, 2, 0, 2, 0, 0]) +
                0.6 * np.array([2, 2, 2, 2, 0, 0, 0]) +
                0.2 * np.array([2, 2, 2, 0, 2, 0, 0])]) / 3
        node_true_divergence_0_12 = np.array([
                0.2 * np.array([2, 1, 1, 0, 2, 0, 0]) +
                0.6 * np.array([2, 1, 1, 1, 0, 0, 0]) +
                0.2 * np.array([2, 1, 1, 0, 2, 0, 0])]) / 2
        haplotypes = np.array([[0, 1, 1],
                               [1, 0, 0],
                               [0, 0, 1],
                               [1, 0, 0],
                               [0, 1, 0],
                               [0, 1, 0],
                               [0, 0, 1],
                               [1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
        traits = np.array([[1, 2, 3, 0],
                           [-5, 0, 1, 1],
                           [3, 4, 1.2, 2]])
        # nb: verified the following with R
        true_cov = np.cov(
                haplotypes,
                traits.T)[:haplotypes.shape[0], haplotypes.shape[0]:] ** 2
        true_cor = np.corrcoef(
                haplotypes,
                traits.T)[:haplotypes.shape[0], haplotypes.shape[0]:] ** 2
        cov02 = np.cov(np.array([1, 0, 1]), traits.T)[:1, 1:] ** 2
        true_branch_cov = (true_cov[1, :] * 1.0 * 0.2 +  # branch 0, tree 0
                           true_cov[4, :] * 0.5 * 0.2 +  # branch 1, tree 0
                           true_cov[2, :] * 0.5 * 0.2 +  # branch 2, tree 0
                           true_cov[0, :] * 0.5 * 0.2 +  # branch 4, tree 0
                           true_cov[1, :] * 0.4 * 0.6 +  # branch 0, tree 1
                           true_cov[4, :] * 0.5 * 0.6 +  # branch 1, tree 1
                           true_cov[2, :] * 0.4 * 0.6 +  # branch 2, tree 1
                           cov02 * 0.1 * 0.6 +  # branch 3, tree 1
                           true_cov[1, :] * 0.7 * 0.2 +  # branch 0, tree 2
                           true_cov[4, :] * 0.5 * 0.2 +  # branch 1, tree 2
                           true_cov[2, :] * 0.5 * 0.2 +  # branch 2, tree 2
                           true_cov[0, :] * 0.2 * 0.2)  # branch 4, tree 2
        cor02 = np.corrcoef(np.array([1, 0, 1]), traits.T)[:1, 1:] ** 2
        true_branch_cor = (true_cor[1, :] * 1.0 * 0.2 +  # branch 0, tree 0
                           true_cor[4, :] * 0.5 * 0.2 +  # branch 1, tree 0
                           true_cor[2, :] * 0.5 * 0.2 +  # branch 2, tree 0
                           true_cor[0, :] * 0.5 * 0.2 +  # branch 4, tree 0
                           true_cor[1, :] * 0.4 * 0.6 +  # branch 0, tree 1
                           true_cor[4, :] * 0.5 * 0.6 +  # branch 1, tree 1
                           true_cor[2, :] * 0.4 * 0.6 +  # branch 2, tree 1
                           cor02 * 0.1 * 0.6 +  # branch 3, tree 1
                           true_cor[1, :] * 0.7 * 0.2 +  # branch 0, tree 2
                           true_cor[4, :] * 0.5 * 0.2 +  # branch 1, tree 2
                           true_cor[2, :] * 0.5 * 0.2 +  # branch 2, tree 2
                           true_cor[0, :] * 0.2 * 0.2)  # branch 4, tree 2

        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           0.4
        4       0           0.5
        5       0           0.7
        6       0           1.0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.2     0.8     3       0,2
        0.0     0.2     4       1,2
        0.2     0.8     4       1,3
        0.8     1.0     4       1,2
        0.8     1.0     5       0,4
        0.0     0.2     6       0,4
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.05        0
        1   0.1         0
        2   0.11        0
        3   0.15        0
        4   0.151       0
        5   0.3         0
        6   0.6         0
        7   0.9         0
        8   0.95        0
        9   0.951       0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state
        0       4       1
        1       0       1
        2       2       1
        3       0       1
        4       1       1
        5       1       1
        6       2       1
        7       0       1
        8       1       1
        9       2       1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            strict=False)

        # diversity between 0 and 1
        A = [[0], [1]]
        n = [len(a) for a in A]

        def f(x):
            return np.array([float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])/(2*n[0]*n[1])])

        # tree lengths:
        mode = "branch"
        self.assertAlmostEqual(divergence(ts, [[0], [1]], [(0, 1)], mode=mode),
                               branch_true_diversity_01)
        self.assertAlmostEqual(ts.divergence([[0], [1]], [(0, 1)], mode=mode),
                               branch_true_diversity_01)
        self.assertAlmostEqual(ts.sample_count_stat(A, f, 1, mode=mode)[0],
                               branch_true_diversity_01)
        self.assertAlmostEqual(ts.diversity([[0, 1]], mode=mode)[0],
                               branch_true_diversity_01)

        # mean diversity between [0, 1] and [0, 2]:
        branch_true_mean_diversity = (0 + branch_true_diversity_02
                                      + branch_true_diversity_01
                                      + branch_true_diversity_12)/4
        A = [[0, 1], [0, 2]]
        n = [len(a) for a in A]

        def f(x):
            return np.array([float(x[0]*(n[1]-x[1]) + (n[0]-x[0])*x[1])/8.0])

        # tree lengths:
        self.assertAlmostEqual(divergence(ts, [A[0], A[1]], [(0, 1)], mode=mode),
                               branch_true_mean_diversity)
        self.assertAlmostEqual(ts.divergence([A[0], A[1]], [(0, 1)], mode=mode),
                               branch_true_mean_diversity)
        self.assertAlmostEqual(ts.sample_count_stat(A, f, 1, mode=mode)[0],
                               branch_true_mean_diversity)

        # Y-statistic for (0/12)
        A = [[0], [1, 2]]

        def f(x):
            return np.array([float(((x[0] == 1) and (x[1] == 0))
                                   or ((x[0] == 0) and (x[1] == 2)))/2.0])

        # tree lengths:
        bts_Y = ts.Y3([[0], [1], [2]], mode=mode)
        py_bsc_Y = Y3(ts, [[0], [1], [2]], [(0, 1, 2)], windows=[0.0, 1.0], mode=mode)
        self.assertArrayAlmostEqual(bts_Y, branch_true_Y)
        self.assertArrayAlmostEqual(py_bsc_Y, branch_true_Y)
        self.assertArrayAlmostEqual(ts.sample_count_stat(A, f, 1, mode=mode)[0],
                                    branch_true_Y)

        mode = "site"
        # sites, Y:
        sts_Y = ts.Y3([[0], [1], [2]], mode=mode)
        py_ssc_Y = Y3(ts, [[0], [1], [2]], [(0, 1, 2)], windows=[0.0, 1.0], mode=mode)
        self.assertArrayAlmostEqual(sts_Y, site_true_Y)
        self.assertArrayAlmostEqual(py_ssc_Y, site_true_Y)
        self.assertArrayAlmostEqual(ts.sample_count_stat(A, f, 1, mode=mode)[0],
                                    site_true_Y)

        A = [[0, 1, 2]]
        n = 3
        W = np.array([[u in A[0]] for u in ts.samples()], dtype=float)

        def f(x):
            return np.array([x[0]*(n-x[0])/(n * (n - 1))])

        mode = "node"
        # nodes, diversity in [0,1,2]
        nodes_div_012 = ts.diversity([[0, 1, 2]], mode=mode).reshape((1, 7))
        py_nodes_div_012 = diversity(ts, [[0, 1, 2]], mode=mode).reshape((1, 7))
        py_general_nodes_div_012 = general_stat(ts, W, f, mode=mode).reshape((1, 7))
        self.assertArrayAlmostEqual(py_nodes_div_012, node_true_diversity_012)
        self.assertArrayAlmostEqual(py_general_nodes_div_012, node_true_diversity_012)
        self.assertArrayAlmostEqual(nodes_div_012, node_true_diversity_012)

        # nodes, divergence [0] to [1,2]
        nodes_div_0_12 = ts.divergence([[0], [1, 2]], mode=mode).reshape((1, 7))
        py_nodes_div_0_12 = divergence(ts, [[0], [1, 2]], mode=mode).reshape((1, 7))
        self.assertArrayAlmostEqual(nodes_div_0_12, node_true_divergence_0_12)
        self.assertArrayAlmostEqual(py_nodes_div_0_12, node_true_divergence_0_12)

        # covariance and correlation
        ts_sitewise_cov = ts.trait_covariance(
                traits, mode="site", windows="sites", span_normalise=False)
        py_sitewise_cov = site_trait_covariance(
                ts, traits, windows="sites", span_normalise=False)
        self.assertArrayAlmostEqual(py_sitewise_cov, true_cov)
        self.assertArrayAlmostEqual(ts_sitewise_cov, true_cov)
        ts_sitewise_cor = ts.trait_correlation(
                traits, mode="site", windows="sites", span_normalise=False)
        py_sitewise_cor = site_trait_correlation(
                ts, traits, windows="sites", span_normalise=False)
        self.assertArrayAlmostEqual(py_sitewise_cor, true_cor)
        self.assertArrayAlmostEqual(ts_sitewise_cor, true_cor)
        # mean
        ts_mean_cov = ts.trait_covariance(traits, mode="site",
                                          windows=[0, ts.sequence_length])
        py_mean_cov = site_trait_covariance(ts, traits)
        self.assertArrayAlmostEqual(ts_mean_cov,
                                    np.array([np.sum(true_cov, axis=0)]))
        self.assertArrayAlmostEqual(ts_mean_cov, py_mean_cov)
        ts_mean_cor = ts.trait_correlation(traits, mode="site",
                                           windows=[0, ts.sequence_length])
        py_mean_cor = site_trait_correlation(ts, traits)
        self.assertArrayAlmostEqual(ts_mean_cor,
                                    np.array([np.sum(true_cor, axis=0)]))
        self.assertArrayAlmostEqual(ts_mean_cor, py_mean_cor)
        # mode = 'branch'
        ts_mean_cov = ts.trait_covariance(traits, mode="branch",
                                          windows=[0, ts.sequence_length])
        py_mean_cov = branch_trait_covariance(ts, traits)
        self.assertArrayAlmostEqual(ts_mean_cov, true_branch_cov)
        self.assertArrayAlmostEqual(ts_mean_cov, py_mean_cov)
        ts_mean_cor = ts.trait_correlation(traits, mode="branch",
                                           windows=[0, ts.sequence_length])
        py_mean_cor = branch_trait_correlation(ts, traits)
        self.assertArrayAlmostEqual(ts_mean_cor, true_branch_cor)
        self.assertArrayAlmostEqual(ts_mean_cor, py_mean_cor)

        # trait regression:
        # r = cor * sd(y) / sd(x) = cov / var(x)
        # geno_var = allele_freqs * (1 - allele_freqs) * (3 / (3 - 1))
        geno_var = np.var(haplotypes, axis=1) * (3 / (3 - 1))
        trait_var = np.var(traits, axis=0) * (3 / (3 - 1))
        py_r = trait_regression(
                ts, traits, None, mode="site", windows="sites", span_normalise=False)
        ts_r = ts.trait_regression(
                traits, None, mode="site", windows="sites", span_normalise=False)
        self.assertArrayAlmostEqual(py_r, ts_r)
        self.assertArrayAlmostEqual(true_cov, py_r * (geno_var[:, np.newaxis] ** 2))
        self.assertArrayAlmostEqual(
                true_cor,
                ts_r * geno_var[:, np.newaxis] / trait_var)

    def test_case_odds_and_ends(self):
        # Tests having (a) the first site after the first window, and
        # (b) no samples having the ancestral state.
        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       0           0.5
        3       0           1.0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     0.5     2       0,1
        0.5     1.0     3       0,1
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.65        0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       0       1               -1
        0       1       2               -1
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            strict=False)

        mode = "site"
        py_div = divergence(
            ts, [[0], [1]], indexes=[(0, 1)], windows=[0.0, 0.5, 1.0], mode=mode)
        div = ts.divergence(
            [[0], [1]], indexes=[(0, 1)], windows=[0.0, 0.5, 1.0], mode=mode)
        self.assertArrayEqual(py_div, div)

    def test_case_four_taxa(self):
        #
        # 1.0          7
        # 0.7         / \                                    6
        #            /   \                                  / \
        # 0.5       /     5              5                 /   5
        #          /     / \            / \__             /   / \
        # 0.4     /     8   \          8     4           /   8   \
        #        /     / \   \        / \   / \         /   / \   \
        # 0.0   0     1   3   2      1   3 0   2       0   1   3   2
        #          (0.0, 0.2),        (0.2, 0.8),       (0.8, 2.5)

        # f4(0, 1, 2, 3): (0 -> 1)(2 -> 3)
        branch_true_f4_0123 = (0.1 * 0.2 + (0.1 + 0.1) * 0.6 + 0.1 * 1.7) / 2.5
        windows = [0.0, 0.4, 2.5]
        branch_true_f4_0123_windowed = np.array([(0.1 * 0.2 + (0.1 + 0.1) * 0.2) / 0.4,
                                                 ((0.1 + 0.1) * 0.4 + 0.1 * 1.7) / 2.1])
        # f4(0, 3, 2, 1): (0 -> 3)(2 -> 1)
        branch_true_f4_0321 = (0.1 * 0.2 + (0.1 + 0.1) * 0.6 + 0.1 * 1.7) / 2.5
        # f2([0,2], [1,3]) = (1/2) (f4(0,1,2,3) + f4(0,3,2,1))
        branch_true_f2_02_13 = (branch_true_f4_0123 + branch_true_f4_0321) / 2
        # diversity([0,1,2,3])
        branch_true_diversity_windowed = (2 / 6) * np.array([
                [(0.2 * (1 + 1 + 1 + 0.5 + 0.4 + 0.5) +
                  (0.4 - 0.2) * (0.5 + 0.4 + 0.5 + 0.5 + 0.4 + 0.5)) /
                 0.4],
                [((0.8 - 0.4) * (0.5 + 0.4 + 0.5 + 0.5 + 0.4 + 0.5) +
                  (2.5 - 0.8) * (0.7 + 0.7 + 0.7 + 0.5 + 0.4 + 0.5)) /
                 (2.5 - 0.4)]])

        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       1           0
        4       0           0.4
        5       0           0.5
        6       0           0.7
        7       0           1.0
        8       0           0.4
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.0     2.5     8       1,3
        0.2     0.8     4       0,2
        0.0     0.2     5       8,2
        0.2     0.8     5       8,4
        0.8     2.5     5       8,2
        0.8     2.5     6       0,5
        0.0     0.2     7       0,5
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            strict=False)

        mode = "branch"
        A = [[0], [1], [2], [3]]
        self.assertAlmostEqual(branch_true_f4_0123, f4(ts, A, mode=mode)[0][0])
        self.assertAlmostEqual(branch_true_f4_0123, ts.f4(A, mode=mode))
        self.assertArrayAlmostEqual(
            branch_true_f4_0123_windowed,
            ts.f4(A, windows=windows, mode=mode).flatten())
        A = [[0], [3], [2], [1]]
        self.assertAlmostEqual(
            branch_true_f4_0321,
            f4(ts, A, [(0, 1, 2, 3)], mode=mode)[0][0])
        self.assertAlmostEqual(branch_true_f4_0321, ts.f4(A, mode=mode))
        A = [[0], [2], [1], [3]]
        self.assertAlmostEqual(0.0, f4(ts, A, [(0, 1, 2, 3)], mode=mode)[0])
        self.assertAlmostEqual(0.0, ts.f4(A, mode=mode))
        A = [[0, 2], [1, 3]]
        self.assertAlmostEqual(
            branch_true_f2_02_13, f2(ts, A, [(0, 1)], mode=mode)[0][0])
        self.assertAlmostEqual(branch_true_f2_02_13, ts.f2(A, mode=mode))

        # diversity
        A = [[0, 1, 2, 3]]
        self.assertArrayAlmostEqual(
            branch_true_diversity_windowed,
            diversity(ts, A, windows=windows, mode=mode))
        self.assertArrayAlmostEqual(
            branch_true_diversity_windowed,
            ts.diversity(A, windows=windows, mode=mode))

    def test_case_recurrent_muts(self):
        # With mutations:
        #
        # 1.0          6
        # 0.7         / \                                    5
        #           (0)  \                                  /(6)
        # 0.5      (1)    4                4               /   4
        #          /     / \              / \             /  (7|8)
        # 0.4    (2)   (3)  \           (4)  3           /   /   \
        #        /     /     \          /   /(5)        /   /     \
        # 0.0   0     1       2        1   0   2       0   1       2
        #          (0.0, 0.2),        (0.2, 0.8),       (0.8, 1.0)
        # genotypes:
        #       0     2       0        1   0   1       0   2       3
        site_true_Y = 0 + 1 + 1

        nodes = io.StringIO("""\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           0.4
        4       0           0.5
        5       0           0.7
        6       0           1.0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.2     0.8     3       0,2
        0.0     0.2     4       1,2
        0.2     0.8     4       1,3
        0.8     1.0     4       1,2
        0.8     1.0     5       0,4
        0.0     0.2     6       0,4
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.05        0
        1   0.3         0
        2   0.9         0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       0       1               -1
        0       0       2               0
        0       0       0               1
        0       1       2               -1
        1       1       1               -1
        1       2       1               -1
        2       4       1               -1
        2       1       2               6
        2       2       3               6
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False)

        # Y3:
        site_tsc_Y = ts.Y3([[0], [1], [2]], mode="site")
        py_ssc_Y = Y3(ts, [[0], [1], [2]], [(0, 1, 2)], windows=[0.0, 1.0], mode="site")
        self.assertAlmostEqual(site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(py_ssc_Y, site_true_Y)

    def test_case_2(self):
        # Here are the trees:
        # t                  |              |              |             |            |
        #
        # 0       --3--      |     --3--    |     --3--    |    --3--    |    --3--   |
        #        /  |  \     |    /  |  \   |    /     \   |   /     \   |   /     \  |
        # 1     4   |   5    |   4   |   5  |   4       5  |  4       5  |  4       5 |
        #       |\ / \ /|    |   |\   \     |   |\     /   |  |\     /   |  |\     /| |
        # 2     | 6   7 |    |   | 6   7    |   | 6   7    |  | 6   7    |  | 6   7 | |
        #       | |\ /| |    |   *  \  |    |   |  \  |    |  |  \       |  |  \    | |
        # 3     | | 8 | |    |   |   8 *    |   |   8 |    |  |   8      |  |   8   | |
        #       | |/ \| |    |   |  /  |    |   |  /  |    |  |  / \     |  |  / \  | |
        # 4     | 9  10 |    |   * 9  10    |   | 9  10    |  | 9  10    |  | 9  10 | |
        #       |/ \ / \|    |   |  \   \   |   |  \   \   |  |  \   \   |  |  \    | |
        # 5     0   1   2    |   0   1   2  |   0   1   2  |  0   1   2  |  0   1   2 |
        #
        #                    |   0.0 - 0.1  |   0.1 - 0.2  |  0.2 - 0.4  |  0.4 - 0.5 |
        # ... continued:
        # t                  |             |             |             |
        #
        # 0         --3--    |    --3--    |    --3--    |    --3--    |    --3--
        #          /     \   |   /     \   |   /     \   |   /     \   |   /  |  \
        # 1       4       5  |  4       5  |  4       5  |  4       5  |  4   |   5
        #         |\     /|  |   \     /|  |   \     /|  |   \     /|  |     /   /|
        # 2       | 6   7 |  |    6   7 |  |    6   7 |  |    6   7 |  |    6   7 |
        #         |  *    *  |     \    |  |       *  |  |    |  /  |  |    |  /  |
        # 3  ...  |   8   |  |      8   |  |      8   |  |    | 8   |  |    | 8   |
        #         |  / \  |  |     / \  |  |     * \  |  |    |  \  |  |    |  \  |
        # 4       | 9  10 |  |    9  10 |  |    9  10 |  |    9  10 |  |    9  10 |
        #         |    /  |  |   /   /  |  |   /   /  |  |   /   /  |  |   /   /  |
        # 5       0   1   2  |  0   1   2  |  0   1   2  |  0   1   2  |  0   1   2
        #
        #         0.5 - 0.6  |  0.6 - 0.7  |  0.7 - 0.8  |  0.8 - 0.9  |  0.9 - 1.0
        #
        # Above, subsequent mutations are backmutations.

        # divergence betw 0 and 1
        branch_true_diversity_01 = 2*(0.6*4 + 0.2*2 + 0.2*5)
        # divergence betw 1 and 2
        branch_true_diversity_12 = 2*(0.2*5 + 0.2*2 + 0.3*5 + 0.3*4)
        # divergence betw 0 and 2
        branch_true_diversity_02 = 2*(0.2*5 + 0.2*4 + 0.3*5 + 0.1*4 + 0.2*5)
        # Y(0;1, 2)
        branch_true_Y = 0.2*4 + 0.2*(4+2) + 0.2*4 + 0.2*2 + 0.2*(5+1)

        # site stats
        # Y(0;1, 2)
        site_true_Y = 1

        nodes = io.StringIO("""\
        is_sample       time    population
        1       0.000000        0
        1       0.000000        0
        1       0.000000        0
        0       5.000000        0
        0       4.000000        0
        0       4.000000        0
        0       3.000000        0
        0       3.000000        0
        0       2.000000        0
        0       1.000000        0
        0       1.000000        0
        """)
        edges = io.StringIO("""\
        left    right   parent  child
        0.500000        1.000000        10      1
        0.000000        0.400000        10      2
        0.600000        1.000000        9       0
        0.000000        0.500000        9       1
        0.800000        1.000000        8       10
        0.200000        0.800000        8       9,10
        0.000000        0.200000        8       9
        0.700000        1.000000        7       8
        0.000000        0.200000        7       10
        0.800000        1.000000        6       9
        0.000000        0.700000        6       8
        0.400000        1.000000        5       2,7
        0.100000        0.400000        5       7
        0.600000        0.900000        4       6
        0.000000        0.600000        4       0,6
        0.900000        1.000000        3       4,5,6
        0.100000        0.900000        3       4,5
        0.000000        0.100000        3       4,5,7
        """)
        sites = io.StringIO("""\
        id  position    ancestral_state
        0   0.0         0
        1   0.55        0
        2   0.75        0
        3   0.85        0
        """)
        mutations = io.StringIO("""\
        site    node    derived_state   parent
        0       0       1               -1
        0       10      1               -1
        0       0       0               0
        1       8       1               -1
        1       2       1               -1
        2       8       1               -1
        2       9       0               5
        """)
        ts = tskit.load_text(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            strict=False)

        def f(x):
            return np.array([float(x[0] == 1)/2.0])

        # divergence between 0 and 1
        mode = "branch"
        for A, truth in zip(
                [[[0, 1]], [[1, 2]], [[0, 2]]],
                [branch_true_diversity_01,
                 branch_true_diversity_12,
                 branch_true_diversity_02]):

            self.assertAlmostEqual(diversity(ts, A, mode=mode)[0][0], truth)
            self.assertAlmostEqual(ts.sample_count_stat(A, f, 1, mode=mode)[0], truth)
            self.assertAlmostEqual(ts.diversity(A, mode="branch")[0], truth)

        # Y-statistic for (0/12)
        A = [[0], [1, 2]]

        def f(x):
            return np.array([float(((x[0] == 1) and (x[1] == 0))
                                   or ((x[0] == 0) and (x[1] == 2)))/2.0])

        # tree lengths:
        self.assertArrayAlmostEqual(Y3(ts, [[0], [1], [2]], [(0, 1, 2)], mode=mode),
                                    branch_true_Y)
        self.assertArrayAlmostEqual(ts.Y3([[0], [1], [2]], [(0, 1, 2)], mode=mode),
                                    branch_true_Y)
        self.assertArrayAlmostEqual(ts.sample_count_stat(A, f, 1, mode=mode)[0],
                                    branch_true_Y)

        # sites:
        mode = "site"
        site_tsc_Y = ts.Y3([[0], [1], [2]], mode=mode)
        py_ssc_Y = Y3(ts, [[0], [1], [2]], [(0, 1, 2)], windows=[0.0, 1.0])
        self.assertAlmostEqual(site_tsc_Y, site_true_Y)
        self.assertAlmostEqual(py_ssc_Y, site_true_Y)
        self.assertAlmostEqual(ts.sample_count_stat(A, f, 1, mode=mode)[0],
                               site_true_Y)


class TestOutputDimensions(StatsTestCase):
    """
    Tests for the dimension stripping behaviour of the stats functions.
    """
    def get_example_ts(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        return ts

    def test_afs_default_windows(self):
        ts = self.get_example_ts()
        n = ts.num_samples
        A = ts.samples()[:4]
        B = ts.samples()[6:]
        for mode in ["site", "branch"]:
            x = ts.allele_frequency_spectrum(mode=mode)
            # x is a 1D numpy array with n + 1 values
            self.assertEqual(x.shape, (n + 1,))
            self.assertArrayEqual(
                x, ts.allele_frequency_spectrum([ts.samples()], mode=mode))
            x = ts.allele_frequency_spectrum([A, B], mode=mode)
            self.assertEqual(x.shape, (len(A) + 1, len(B) + 1))

    def test_afs_windows(self):
        ts = self.get_example_ts()
        L = ts.sequence_length

        windows = [0, L / 4, L / 2, L]
        A = ts.samples()[:4]
        B = ts.samples()[6:]
        for mode in ["site", "branch"]:
            x = ts.allele_frequency_spectrum([A, B], windows=windows, mode=mode)
            self.assertEqual(x.shape, (3, len(A) + 1, len(B) + 1))

            x = ts.allele_frequency_spectrum([A], windows=windows, mode=mode)
            self.assertEqual(x.shape, (3, len(A) + 1))

            x = ts.allele_frequency_spectrum(windows=windows, mode=mode)
            # Default returns this for all samples
            self.assertEqual(x.shape, (3, ts.num_samples + 1))
            y = ts.allele_frequency_spectrum([ts.samples()], windows=windows, mode=mode)
            self.assertArrayEqual(x, y)

    def test_one_way_stat_default_windows(self):
        ts = self.get_example_ts()
        # Use diversity as the example one-way stat.
        for mode in ["site", "branch"]:
            x = ts.diversity(mode=mode)
            # x is a zero-d numpy value
            self.assertEqual(np.shape(x), tuple())
            self.assertEqual(x, float(x))
            self.assertEqual(x, ts.diversity(ts.samples(), mode=mode))
            self.assertArrayEqual([x], ts.diversity([ts.samples()], mode=mode))

        mode = "node"
        x = ts.diversity(mode=mode)
        # x is a 1D numpy array with N values
        self.assertEqual(x.shape, (ts.num_nodes,))
        self.assertArrayEqual(x, ts.diversity(ts.samples(), mode=mode))
        y = ts.diversity([ts.samples()], mode=mode)
        # We're adding on the *last* dimension, so must reshape
        self.assertArrayEqual(x.reshape(ts.num_nodes, 1), y)

    def verify_one_way_stat_windows(self, ts, method):
        L = ts.sequence_length
        N = ts.num_nodes

        windows = [0, L / 4, L / 2, 0.75 * L, L]
        A = ts.samples()[:6]
        B = ts.samples()[6:]
        for mode in ["site", "branch"]:
            x = method([A, B], windows=windows, mode=mode)
            # Four windows, 2 sets.
            self.assertEqual(x.shape, (4, 2))

            x = method([A], windows=windows, mode=mode)
            # Four windows, 1 sets.
            self.assertEqual(x.shape, (4, 1))

            x = method(A, windows=windows, mode=mode)
            # Dropping the outer list removes the last dimension
            self.assertEqual(x.shape, (4, ))

            x = method(windows=windows, mode=mode)
            # Default returns this for all samples
            self.assertEqual(x.shape, (4, ))
            y = method(ts.samples(), windows=windows, mode=mode)
            self.assertArrayEqual(x, y)

        mode = "node"
        x = method([A, B], windows=windows, mode=mode)
        # Four windows, N nodes and 2 sets.
        self.assertEqual(x.shape, (4, N, 2))

        x = method([A], windows=windows, mode=mode)
        # Four windows, N nodes and 1 set.
        self.assertEqual(x.shape, (4, N, 1))

        x = method(A, windows=windows, mode=mode)
        # Drop the outer list, so we lose the last dimension
        self.assertEqual(x.shape, (4, N))

        x = method(windows=windows, mode=mode)
        # The default sample sets also drops the last dimension
        self.assertEqual(x.shape, (4, N))

        self.assertEqual(ts.num_trees, 1)
        # In this example, we know that the trees are all the same so check this
        # for sanity.
        self.assertArrayEqual(x[0], x[1])
        self.assertArrayEqual(x[0], x[2])

    def test_diversity_windows(self):
        ts = self.get_example_ts()
        self.verify_one_way_stat_windows(ts, ts.diversity)

    def test_Tajimas_D_windows(self):
        ts = self.get_example_ts()
        self.verify_one_way_stat_windows(ts, ts.Tajimas_D)

    def test_segregating_sites_windows(self):
        ts = self.get_example_ts()
        self.verify_one_way_stat_windows(ts, ts.segregating_sites)

    def test_two_way_stat_default_windows(self):
        ts = self.get_example_ts()
        # Use divergence as the example one-way stat.
        A = ts.samples()[:6]
        B = ts.samples()[6:]
        for mode in ["site", "branch"]:
            x = ts.divergence([A, B], mode=mode)
            # x is a zero-d numpy value
            self.assertEqual(np.shape(x), tuple())
            self.assertEqual(x, float(x))
            # If indexes is a 1D array, we also drop the outer dimension
            self.assertEqual(x, ts.divergence([A, B, A], indexes=[0, 1], mode=mode))
            # But, if it's a 2D array we keep the outer dimension
            self.assertEqual([x], ts.divergence([A, B], indexes=[[0, 1]], mode=mode))

        mode = "node"
        x = ts.divergence([A, B], mode=mode)
        # x is a 1D numpy array with N values
        self.assertEqual(x.shape, (ts.num_nodes,))
        self.assertArrayEqual(x, ts.divergence([A, B], indexes=[0, 1], mode=mode))
        y = ts.divergence([A, B], indexes=[[0, 1]], mode=mode)
        # We're adding on the *last* dimension, so must reshape
        self.assertArrayEqual(x.reshape(ts.num_nodes, 1), y)

    def verify_two_way_stat_windows(self, ts, method):
        L = ts.sequence_length
        N = ts.num_nodes

        windows = [0, L / 4, L / 2, L]
        A = ts.samples()[:7]
        B = ts.samples()[7:]
        for mode in ["site", "branch"]:
            x = method(
                [A, B, A], indexes=[[0, 1], [0, 2]], windows=windows, mode=mode)
            # Three windows, 2 pairs
            self.assertEqual(x.shape, (3, 2))

            x = method([A, B], indexes=[[0, 1]], windows=windows, mode=mode)
            # Three windows, 1 pair
            self.assertEqual(x.shape, (3, 1))

            x = method([A, B], indexes=[0, 1], windows=windows, mode=mode)
            # Dropping the outer list removes the last dimension
            self.assertEqual(x.shape, (3, ))

            y = method([A, B], windows=windows, mode=mode)
            self.assertEqual(y.shape, (3, ))
            self.assertArrayEqual(x, y)

        mode = "node"
        x = method([A, B], indexes=[[0, 1], [0, 1]], windows=windows, mode=mode)
        # Three windows, N nodes and 2 pairs
        self.assertEqual(x.shape, (3, N, 2))

        x = method([A, B], indexes=[[0, 1]], windows=windows, mode=mode)
        # Three windows, N nodes and 1 pairs
        self.assertEqual(x.shape, (3, N, 1))

        x = method([A, B], indexes=[0, 1], windows=windows, mode=mode)
        # Drop the outer list, so we lose the last dimension
        self.assertEqual(x.shape, (3, N))

        x = method([A, B], windows=windows, mode=mode)
        # The default sample sets also drops the last dimension
        self.assertEqual(x.shape, (3, N))

        self.assertEqual(ts.num_trees, 1)
        # In this example, we know that the trees are all the same so check this
        # for sanity.
        self.assertArrayEqual(x[0], x[1])
        self.assertArrayEqual(x[0], x[2])

    def test_divergence_windows(self):
        ts = self.get_example_ts()
        self.verify_two_way_stat_windows(ts, ts.divergence)

    def test_Fst_windows(self):
        ts = self.get_example_ts()
        self.verify_two_way_stat_windows(ts, ts.Fst)

    def test_f2_windows(self):
        ts = self.get_example_ts()
        self.verify_two_way_stat_windows(ts, ts.f2)

    def verify_three_way_stat_windows(self, ts, method):
        L = ts.sequence_length
        N = ts.num_nodes

        windows = [0, L / 4, L / 2, L]
        A = ts.samples()[:2]
        B = ts.samples()[2: 4]
        C = ts.samples()[4:]
        for mode in ["site", "branch"]:
            x = method(
                [A, B, C], indexes=[[0, 1, 2], [0, 2, 1]], windows=windows, mode=mode)
            # Three windows, 2 triple
            self.assertEqual(x.shape, (3, 2))

            x = method([A, B, C], indexes=[[0, 1, 2]], windows=windows, mode=mode)
            # Three windows, 1 triple
            self.assertEqual(x.shape, (3, 1))

            x = method([A, B, C], indexes=[0, 1, 2], windows=windows, mode=mode)
            # Dropping the outer list removes the last dimension
            self.assertEqual(x.shape, (3, ))

            y = method([A, B, C], windows=windows, mode=mode)
            self.assertEqual(y.shape, (3, ))
            self.assertArrayEqual(x, y)

        mode = "node"
        x = method([A, B, C], indexes=[[0, 1, 2], [0, 2, 1]], windows=windows, mode=mode)
        # Three windows, N nodes and 2 triples
        self.assertEqual(x.shape, (3, N, 2))

        x = method([A, B, C], indexes=[[0, 1, 2]], windows=windows, mode=mode)
        # Three windows, N nodes and 1 triples
        self.assertEqual(x.shape, (3, N, 1))

        x = method([A, B, C], indexes=[0, 1, 2], windows=windows, mode=mode)
        # Drop the outer list, so we lose the last dimension
        self.assertEqual(x.shape, (3, N))

        x = method([A, B, C], windows=windows, mode=mode)
        # The default sample sets also drops the last dimension
        self.assertEqual(x.shape, (3, N))

        self.assertEqual(ts.num_trees, 1)
        # In this example, we know that the trees are all the same so check this
        # for sanity.
        self.assertArrayEqual(x[0], x[1])
        self.assertArrayEqual(x[0], x[2])

    def test_Y3_windows(self):
        ts = self.get_example_ts()
        self.verify_three_way_stat_windows(ts, ts.Y3)

    def test_f3_windows(self):
        ts = self.get_example_ts()
        self.verify_three_way_stat_windows(ts, ts.f3)
