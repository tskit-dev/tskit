# MIT License
#
# Copyright (c) 2023 Tskit Developers
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
Test cases for divergence matrix based pairwise stats
"""
import itertools

import msprime
import numpy as np
import pytest

import tskit
from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


def assert_dicts_close(d1, d2):
    if not set(d1.keys()) == set(d2.keys()):
        print("d1:", set(d1.keys()) - set(d2.keys()))
        print("d2:", set(d2.keys()) - set(d1.keys()))
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        np.testing.assert_allclose(d1[k], d2[k])


# Implementation note: the class structure here, where we pass in all the
# needed arrays through the constructor was determined by an older version
# in which we used numba acceleration. We could just pass in a reference to
# the tree sequence now, but it is useful to keep track of exactly what we
# require, so leaving it as it is for now.
class DivergenceMatrix:
    def __init__(
        self,
        num_nodes,
        samples,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length,
        verbosity=0,
        internal_checks=False,
    ):
        # virtual root is at num_nodes; virtual samples are beyond that
        N = num_nodes + 1 + len(samples)
        # Quintuply linked tree
        self.parent = np.full(N, -1, dtype=np.int32)
        self.left_sib = np.full(N, -1, dtype=np.int32)
        self.right_sib = np.full(N, -1, dtype=np.int32)
        self.left_child = np.full(N, -1, dtype=np.int32)
        self.right_child = np.full(N, -1, dtype=np.int32)
        # Sample lists refer to sample *index*
        self.num_samples = np.full(N, 0, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples
        self.position = 0
        self.virtual_root = num_nodes
        self.x = np.zeros(N, dtype=np.float64)
        self.stack = [{} for _ in range(N)]
        self.verbosity = verbosity
        self.internal_checks = internal_checks

        for j, u in enumerate(samples):
            self.num_samples[u] = 1
            self.insert_root(u)
            # Add branch to the virtual sample
            v = num_nodes + 1 + j
            self.insert_branch(u, v)
            self.num_samples[v] = 1

    def print_state(self, msg=""):
        num_nodes = len(self.parent)
        print(f"..........{msg}................")
        print(f"position = {self.position}")
        for j in range(num_nodes):
            if j <= self.virtual_root:
                st = "NaN" if j >= self.virtual_root else f"{self.nodes_time[j]}"
                pt = (
                    "NaN"
                    if self.parent[j] == tskit.NULL
                    else f"{self.nodes_time[self.parent[j]]}"
                )
                print(
                    f"node {j} -> {self.parent[j]}: "
                    f"ns = {self.num_samples[j]}, "
                    f"z = ({pt} - {st})"
                    f" * ({self.position} - {self.x[j]})"
                    f" = {self.get_z(j)}"
                )
            else:
                sample = self.samples[j - self.virtual_root - 1]
                print(f"node {j} -> virtual sample for : {sample}")
            for u, z in self.stack[j].items():
                print(f"   {(j, u)}: {z}")
        print(f"Virtual root: {self.virtual_root}")
        roots = []
        u = self.left_child[self.virtual_root]
        while u != tskit.NULL:
            roots.append(u)
            u = self.right_sib[u]
        print("Roots:", roots)
        print("Current state:")
        state = self.current_state()
        for k in state:
            print(f"   {k}: {state[k]}")

    def remove_branch(self, p, c):
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == -1:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == -1:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = -1
        self.left_sib[c] = -1
        self.right_sib[c] = -1

    def insert_branch(self, p, c):
        self.parent[c] = p
        u = self.right_child[p]
        if u == -1:
            self.left_child[p] = c
            self.left_sib[c] = -1
            self.right_sib[c] = -1
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = -1
        self.right_child[p] = c

    def remove_root(self, root):
        self.remove_branch(self.virtual_root, root)

    def insert_root(self, root):
        self.insert_branch(self.virtual_root, root)
        self.parent[root] = -1

    def remove_edge(self, p, c):
        assert p != -1
        self.remove_branch(p, c)
        # check for root changes
        u = p
        while u != tskit.NULL:
            path_end = u
            path_end_was_root = self.num_samples[u] > 0
            self.num_samples[u] -= self.num_samples[c]
            u = self.parent[u]
        if path_end_was_root and (self.num_samples[path_end] == 0):
            self.remove_root(path_end)
        if self.num_samples[c] > 0:
            self.insert_root(c)

    def insert_edge(self, p, c):
        assert p != -1
        assert self.parent[c] == -1, "contradictory edges"
        # check for root changes
        u = p
        while u != tskit.NULL:
            path_end = u
            path_end_was_root = self.num_samples[u] > 0
            self.num_samples[u] += self.num_samples[c]
            u = self.parent[u]
        if self.num_samples[c] > 0:
            self.remove_root(c)
        if (self.num_samples[path_end] > 0) and not path_end_was_root:
            self.insert_root(path_end)
        self.insert_branch(p, c)

    # begin stack stuff

    def add_to_stack(self, u, v, z):
        if z > 0 and self.num_samples[u] > 0 and self.num_samples[v] > 0:
            if v not in self.stack[u]:
                self.stack[u][v] = 0.0
                assert u not in self.stack[v]
                self.stack[v][u] = 0.0
            self.stack[u][v] += z
            self.stack[v][u] += z
        # pedantic error checking:
        if self.internal_checks:
            p = u
            while p != tskit.NULL:
                assert p != v
                p = self.parent[p]
            p = v
            while p != tskit.NULL:
                assert p != u
                p = self.parent[p]

    def empty_stack(self, u):
        for w in self.stack[u]:
            assert u in self.stack[w]
            del self.stack[w][u]
        self.stack[u].clear()

    def verify_zero_root_path(self, u):
        """
        Verify that there are no contributions along the path
        from u up to the root. (should be true after flush_root_path)
        """
        for v, z in self.stack[u].items():
            if z != 0:
                print(f"Uh-oh!: [{u}] : ({v}, {z})")
            assert z == 0
        while u != tskit.NULL:
            assert self.parent[u] == tskit.NULL or self.x[u] == self.position
            u = self.parent[u]

    def get_z(self, u):
        p = self.parent[u]
        if p == tskit.NULL or u >= self.virtual_root:
            return 0
        time = self.nodes_time[p] - self.nodes_time[u]
        span = self.position - self.x[u]
        return time * span

    def mrca(self, a, b):
        # just used for `current_state`
        aa = [a]
        while a != tskit.NULL:
            a = self.parent[a]
            aa.append(a)
        while b not in aa:
            b = self.parent[b]
        return b

    def current_state(self):
        """
        Compute the current output, for debugging.
        NOTE that the path back to the roots of disconnected trees
        *still counts* for divergence *between* those trees!
        (In other words, disconnected trees act as if they are
        connected to a virtual root by a branch of length zero.)
        """
        if self.verbosity > 1:
            print("---------------")
        n = len(self.samples)
        virtual_samples = [j + self.virtual_root + 1 for j in range(n)]
        out = {}
        for a, b in itertools.combinations(virtual_samples, 2):
            k = (a, b)
            out[k] = 0
            m = self.mrca(a, b)
            # edges on the path up from a
            pa = a
            while pa != m:
                if self.verbosity > 1:
                    print("edge:", k, pa, self.get_z(pa))
                out[k] += self.get_z(pa)
                pa = self.parent[pa]
            # edges on the path up from b
            pb = b
            while pb != m:
                if self.verbosity > 1:
                    print("edge:", k, pb, self.get_z(pb))
                out[k] += self.get_z(pb)
                pb = self.parent[pb]
            # pairwise stack references along the way
            pa = a
            while pa != m:
                pb = b
                while pb != m:
                    for w, z in self.stack[pa].items():
                        if w == pb:
                            if self.verbosity > 1:
                                print("stack:", k, (pa, pb), z)
                            out[k] += z
                    pb = self.parent[pb]
                pa = self.parent[pa]
        if self.verbosity > 1:
            print("---------------")
        return out

    def flush_branch(self, root_path):
        # Flush the contributions of the branch at the start of this
        # path to the total divergance to all of the sibs on the path.
        if self.verbosity > 0:
            print(f"flush_branch({root_path})")
        if self.internal_checks:
            # this operation should not change the current output
            before_state = self.current_state()

        u = root_path[0]
        z = self.get_z(u)
        self.x[u] = self.position
        assert self.get_z(u) == 0

        # iterate over the siblings of the path to the virtual root
        for j in range(len(root_path) - 1):
            c = root_path[j]
            p = root_path[j + 1]
            s = self.left_child[p]
            while s != tskit.NULL:
                if s != c:
                    if self.verbosity > 1:
                        print(f"adding {z} to {(u, s)}")
                    self.add_to_stack(u, s, z)
                s = self.right_sib[s]

        if self.internal_checks:
            after_state = self.current_state()
            assert_dicts_close(before_state, after_state)

    def push_down(self, u):
        """
        Push down references in the stack from u to other nodes
        to the children of u.
        """
        if self.internal_checks:
            # this operation should not change the current output
            before_state = self.current_state()
        if self.verbosity > 0:
            print(f"push_down({u})")

        for w, z in self.stack[u].items():
            c = self.left_child[u]
            while c != tskit.NULL:
                if self.verbosity > 1:
                    print(f"adding {z} to {(w, c)}")
                self.add_to_stack(w, c, z)
                c = self.right_sib[c]
        self.empty_stack(u)

        assert len(self.stack[u]) == 0
        if self.internal_checks:
            after_state = self.current_state()
            assert_dicts_close(before_state, after_state)

    def get_root_path(self, u):
        """
        Returns the list of nodes back to the virtual root.
        """
        root_path = []
        p = u
        while p != tskit.NULL:
            root_path.append(p)
            p = self.parent[p]
        root_path.append(self.virtual_root)
        return root_path

    def flush_root_path(self, root_path):
        """
        Clears all nodes on the path from the virtual root down to u
        by pushing the contributions of all their branches to the stack
        and pushing all stack references to their children.
        """
        if self.verbosity > 0:
            print(f"flush_root_path({root_path})")
        if self.internal_checks:
            # this operation should not change the current output
            before_state = self.current_state()

        # NOTE: we have a quadratic complexity here in the length of the
        # root path, because each iteration of this loop goes over the root
        # path in flush_branch
        j = len(root_path) - 1
        while j >= 0:
            p = root_path[j]
            self.flush_branch(root_path[j:])
            self.push_down(p)
            self.verify_zero_root_path(p)
            j -= 1

        u = root_path[0]
        self.verify_zero_root_path(u)
        if self.internal_checks:
            after_state = self.current_state()
            assert_dicts_close(before_state, after_state)

    def run(self):
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        # TODO: self.position is redundant with left
        left = 0
        self.position = left

        while k < M and left <= self.sequence_length:
            while k < M and edges_right[out_order[k]] == left:
                p = edges_parent[out_order[k]]
                c = edges_child[out_order[k]]
                root_path = self.get_root_path(c)
                self.flush_root_path(root_path)
                assert self.x[c] == self.position
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.remove_edge(p, c)
                k += 1
                if self.verbosity > 1:
                    self.print_state(f"remove {p, c}")
            while j < M and edges_left[in_order[j]] == left:
                p = edges_parent[in_order[j]]
                c = edges_child[in_order[j]]
                # QUESTION: how does this deal with us having bits missing
                # out of the trees? What happens when we have a missing
                # left flank?
                if self.position > 0:
                    root_path = self.get_root_path(p)
                    self.flush_root_path(root_path)
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.insert_edge(p, c)
                self.x[c] = self.position
                j += 1
                if self.verbosity > 1:
                    self.print_state(f"add {p, c}")
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            left = right
            self.position = left

        # self.print_state()

        # clear remaining things down to virtual samples
        for j, u in enumerate(self.samples):
            self.push_down(u)
            v = self.virtual_root + 1 + j
            self.remove_edge(u, v)
        # self.print_state()
        out = np.zeros((len(self.samples), len(self.samples)))
        for out_i in range(len(self.samples)):
            i = out_i + self.virtual_root + 1
            for j, z in self.stack[i].items():
                assert j > self.virtual_root
                assert j <= self.virtual_root + len(self.samples)
                out_j = j - self.virtual_root - 1
                out[out_i, out_j] = z
        return out


def divergence_matrix(ts, **kwargs):
    dm = DivergenceMatrix(
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        **kwargs,
    )
    return dm.run()


def lib_divergence_matrix(ts, mode="branch"):
    if ts.num_samples > 0:
        # FIXME: the code general stat code doesn't seem to handle zero samples
        # case, need to identify MWE and file issue.
        out = ts.divergence(
            [[u] for u in ts.samples()],
            [(i, j) for i in range(ts.num_samples) for j in range(ts.num_samples)],
            mode=mode,
            span_normalise=False,
        ).reshape((ts.num_samples, ts.num_samples))
        for i in range(ts.num_samples):
            out[i, i] = 0
    else:
        out = np.zeros(shape=(0, 0))
    return out


def check_divmat(ts, *, internal_checks=False, verbosity=0):
    if verbosity > 1:
        print(ts.draw_text())
    D1 = lib_divergence_matrix(ts, mode="branch")
    D2 = divergence_matrix(ts, internal_checks=internal_checks, verbosity=verbosity)
    np.testing.assert_allclose(D1, D2)
    # D3 = ts.divergence_matrix()
    # np.testing.assert_allclose(D1, D3)
    return D1


class TestExamples:
    @pytest.mark.parametrize("n", [2, 3, 5])
    @pytest.mark.parametrize("seed", range(1, 4))
    def test_small_internal_checks(self, n, seed):
        ts = msprime.sim_ancestry(
            n,
            ploidy=1,
            sequence_length=1000,
            recombination_rate=0.01,
            random_seed=seed,
        )
        assert ts.num_trees >= 2
        check_divmat(ts, verbosity=0, internal_checks=True)

    @pytest.mark.parametrize("n", [2, 3, 5, 15])
    @pytest.mark.parametrize("seed", range(1, 5))
    def test_simple_sims(self, n, seed):
        ts = msprime.sim_ancestry(
            n,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=seed,
        )
        assert ts.num_trees >= 2
        check_divmat(ts)

    @pytest.mark.parametrize("n", [2, 3, 5, 15])
    def test_single_balanced_tree(self, n):
        ts = tskit.Tree.generate_balanced(n).tree_sequence
        check_divmat(ts, verbosity=0)

    @pytest.mark.parametrize("seed", range(1, 5))
    def test_one_internal_sample_sims(self, seed):
        ts = msprime.sim_ancestry(
            10,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=seed,
        )
        t = ts.dump_tables()
        # Add a new sample directly below another sample
        u = t.nodes.add_row(time=-1, flags=tskit.NODE_IS_SAMPLE)
        t.edges.add_row(parent=0, child=u, left=0, right=ts.sequence_length)
        t.sort()
        t.build_index()
        ts = t.tree_sequence()
        check_divmat(ts)

    def test_missing_flanks(self):
        ts = msprime.sim_ancestry(
            20,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=1234,
        )
        assert ts.num_trees >= 2
        ts = ts.keep_intervals([[20, 80]])
        assert ts.first().interval == (0, 20)
        check_divmat(ts)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_suite_examples(self, ts):
        check_divmat(ts)

    @pytest.mark.parametrize("n", [2, 3, 10])
    def test_dangling_on_samples(self, n):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(n).tree_sequence
        D1 = check_divmat(ts1)
        tables = ts1.dump_tables()
        for u in ts1.samples():
            v = tables.nodes.add_row(time=-1)
            tables.edges.add_row(left=0, right=ts1.sequence_length, parent=u, child=v)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_divmat(ts2, internal_checks=True)
        np.testing.assert_array_almost_equal(D1, D2)

    @pytest.mark.parametrize("n", [2, 3, 10])
    def test_dangling_on_all(self, n):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(n).tree_sequence
        D1 = check_divmat(ts1)
        tables = ts1.dump_tables()
        for u in range(ts1.num_nodes):
            v = tables.nodes.add_row(time=-1)
            tables.edges.add_row(left=0, right=ts1.sequence_length, parent=u, child=v)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_divmat(ts2, internal_checks=True)
        np.testing.assert_array_almost_equal(D1, D2)

    def test_disconnected_non_sample_topology(self):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(5).tree_sequence
        D1 = check_divmat(ts1)
        tables = ts1.dump_tables()
        # Add an extra bit of disconnected non-sample topology
        u = tables.nodes.add_row(time=0)
        v = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=0, right=ts1.sequence_length, parent=v, child=u)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_divmat(ts2, internal_checks=True)
        np.testing.assert_array_almost_equal(D1, D2)
