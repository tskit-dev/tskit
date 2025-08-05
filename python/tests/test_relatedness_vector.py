# MIT License
#
# Copyright (c) 2025 Tskit Developers
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
Test cases for matrix-vector product stats
"""
import msprime
import numpy as np
import pytest

import tskit
from tests import tsutil
from tests.tsutil import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


# Implementation note: the class structure here, where we pass in all the
# needed arrays through the constructor was determined by an older version
# in which we used numba acceleration. We could just pass in a reference to
# the tree sequence now, but it is useful to keep track of exactly what we
# require, so leaving it as it is for now.
class RelatednessVector:
    def __init__(
        self,
        sample_weights,
        windows,
        num_nodes,
        samples,
        focal_nodes,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        sequence_length,
        tree_pos,
        verbosity=0,
        internal_checks=False,
        centre=True,
    ):
        self.sample_weights = np.asarray(sample_weights, dtype=np.float64)
        self.num_weights = self.sample_weights.shape[1]
        self.windows = windows
        N = num_nodes
        self.parent = np.full(N, -1, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples
        self.focal_nodes = focal_nodes
        self.tree_pos = tree_pos
        self.position = windows[0]
        self.x = np.zeros(N, dtype=np.float64)
        self.w = np.zeros((N, self.num_weights), dtype=np.float64)
        self.v = np.zeros((N, self.num_weights), dtype=np.float64)
        self.verbosity = verbosity
        self.internal_checks = internal_checks
        self.centre = centre

        if self.centre:
            self.sample_weights -= np.mean(self.sample_weights, axis=0)

        for j, u in enumerate(samples):
            self.w[u] = self.sample_weights[j]

        if self.verbosity > 0:
            self.print_state("init")

    def print_state(self, msg=""):
        num_nodes = len(self.parent)
        print(f"..........{msg}................")
        print("tree_pos:")
        print(self.tree_pos)
        print(f"position = {self.position}")
        for j in range(num_nodes):
            st = f"{self.nodes_time[j]}"
            pt = (
                "NaN"
                if self.parent[j] == tskit.NULL
                else f"{self.nodes_time[self.parent[j]]}"
            )
            print(
                f"node {j} -> {self.parent[j]}: "
                f"z = ({pt} - {st})"
                f" * ({self.position} - {self.x[j]:.2})"
                f" * {','.join(map(str, self.w[j].round(2)))}"
                f" = {','.join(map(str, self.get_z(j).round(2)))}"
            )
            print(f"         value: {','.join(map(str, self.v[j].round(2)))}")
        roots = []
        fmt = "{:<6}{:>8}\t{}\t{}\t{}"
        s = f"roots = {roots}\n"
        s += (
            fmt.format(
                "node",
                "parent",
                "value",
                "weight",
                "z",
            )
            + "\n"
        )
        for u in range(num_nodes):
            u_str = f"{u}"
            s += (
                fmt.format(
                    u_str,
                    self.parent[u],
                    ",".join(map(str, self.v[u].round(2))),
                    ",".join(map(str, self.w[u].round(2))),
                    ",".join(map(str, self.get_z(u).round(2))),
                )
                + "\n"
            )
        print(s)

        print("Current state:")
        state = self.current_state()
        for j, x in enumerate(state):
            print(f"   {j}: {x}")
        print("..........................")

    def remove_edge(self, p, c):
        if self.verbosity > 0:
            self.print_state(f"remove {int(p), int(c)}")
        assert p != -1
        self.v[c] += self.get_z(c)
        self.x[c] = self.position
        self.parent[c] = -1
        self.adjust_path_up(p, c, -1)

    def insert_edge(self, p, c):
        if self.verbosity > 0:
            self.print_state(f"insert {int(p), int(c)}")
        assert p != -1
        assert self.parent[c] == -1, "contradictory edges"
        self.adjust_path_up(p, c, +1)
        self.x[c] = self.position
        self.parent[c] = p

    def adjust_path_up(self, p, c, sign):
        # sign = -1 for removing edges, +1 for adding
        while p != tskit.NULL:
            self.v[p] += self.get_z(p)
            self.x[p] = self.position
            self.v[c] -= sign * self.v[p]
            self.w[p] += sign * self.w[c]
            p = self.parent[p]

    def get_z(self, u):
        p = self.parent[u]
        if p == tskit.NULL:
            return np.zeros(self.num_weights, dtype=np.float64)
        time = self.nodes_time[p] - self.nodes_time[u]
        span = self.position - self.x[u]
        return time * span * self.w[u]

    def mrca(self, a, b):
        # just used for `current_state`
        aa = [a]
        while a != tskit.NULL:
            a = self.parent[a]
            aa.append(a)
        while b not in aa:
            b = self.parent[b]
        return b

    def write_output(self):
        """
        Compute and return the current state, zero-ing out
        all contributions (used for switching between windows).
        """
        n = len(self.focal_nodes)
        out = np.zeros((n, self.num_weights))
        for j, c in enumerate(self.focal_nodes):
            while c != tskit.NULL:
                if self.x[c] != self.position:
                    self.v[c] += self.get_z(c)
                    self.x[c] = self.position
                out[j] += self.v[c]
                c = self.parent[c]
        self.v *= 0.0
        return out

    def current_state(self):
        """
        Compute the current output, for debugging.
        """
        if self.verbosity > 2:
            print("---------------")
        n = len(self.focal_nodes)
        out = np.zeros((n, self.num_weights))
        for j, a in enumerate(self.focal_nodes):
            # edges on the path up from a
            pa = a
            while pa != tskit.NULL:
                if self.verbosity > 2:
                    print("edge:", pa, self.get_z(pa))
                out[j] += self.get_z(pa) + self.v[pa]
                pa = self.parent[pa]
        if self.verbosity > 2:
            print("---------------")
        return out

    def run(self):
        M = self.edges_left.shape[0]
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child
        tree_pos = self.tree_pos
        in_order = tree_pos.in_range.order
        out_order = tree_pos.out_range.order
        num_windows = len(self.windows) - 1
        out = np.zeros(
            (num_windows, len(self.focal_nodes), self.sample_weights.shape[1])
        )

        m = 0
        self.position = self.windows[0]

        # seek to first window
        for j in range(tree_pos.in_range.start, tree_pos.in_range.stop, 1):
            e = in_order[j]
            if edges_left[e] <= self.position and self.position < edges_right[e]:
                p = edges_parent[e]
                c = edges_child[e]
                self.insert_edge(p, c)

        valid = tree_pos.next()
        j = tree_pos.in_range.start - 1
        k = tree_pos.out_range.start - 1
        while m < num_windows:
            if valid and self.position == tree_pos.interval.left:
                for k in range(tree_pos.out_range.start, tree_pos.out_range.stop, 1):
                    e = out_order[k]
                    p = edges_parent[e]
                    c = edges_child[e]
                    self.remove_edge(p, c)
                for j in range(tree_pos.in_range.start, tree_pos.in_range.stop, 1):
                    e = in_order[j]
                    p = edges_parent[e]
                    c = edges_child[e]
                    self.insert_edge(p, c)
                    assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                valid = tree_pos.next()
            next_position = self.windows[m + 1]
            if j + 1 < M:
                next_position = min(next_position, edges_left[in_order[j + 1]])
            if k + 1 < M:
                next_position = min(next_position, edges_right[out_order[k + 1]])
            assert self.position < next_position
            self.position = next_position
            if self.position == self.windows[m + 1]:
                out[m] = self.write_output()
                m = m + 1

        if self.verbosity > 1:
            self.print_state()

        if self.centre:
            for m in range(num_windows):
                out[m] -= np.mean(out[m], axis=0)
        return out


def relatedness_vector(ts, sample_weights, windows=None, nodes=None, **kwargs):
    if len(sample_weights.shape) == 1:
        sample_weights = sample_weights[:, np.newaxis]
    if nodes is None:
        nodes = np.fromiter(ts.samples(), dtype=np.int32)
    drop_dimension = windows is None
    if drop_dimension:
        windows = [0, ts.sequence_length]

    tree_pos = tsutil.TreeIndexes(ts)
    breakpoints = np.fromiter(ts.breakpoints(), dtype="float")
    index = np.searchsorted(breakpoints, windows[0])
    if breakpoints[index] > windows[0]:
        index -= 1
    tree_pos.seek_forward(index)

    rv = RelatednessVector(
        sample_weights,
        windows,
        ts.num_nodes,
        samples=ts.samples(),
        focal_nodes=nodes,
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        sequence_length=ts.sequence_length,
        tree_pos=tree_pos,
        **kwargs,
    )
    out = rv.run()
    if drop_dimension:
        assert len(out.shape) == 3 and out.shape[0] == 1
        out = out[0]
    return out


def relatedness_matrix(ts, windows, centre, nodes=None):
    if nodes is None:
        keep_rows = np.arange(ts.num_samples)
        keep_cols = np.arange(ts.num_samples)
    else:
        orig_samples = list(ts.samples())
        extra_nodes = set(nodes).difference(set(orig_samples))
        tables = ts.dump_tables()
        tables.nodes.clear()
        for n in ts.nodes():
            if n.id in extra_nodes:
                n = n.replace(flags=n.flags | tskit.NODE_IS_SAMPLE)
            tables.nodes.append(n)
        ts = tables.tree_sequence()
        all_samples = list(ts.samples())
        keep_rows = np.array([all_samples.index(i) for i in nodes])
        keep_cols = np.array([all_samples.index(i) for i in orig_samples])

    use_windows = windows
    drop_first = windows is not None and windows[0] > 0
    if drop_first:
        use_windows = np.concatenate([[0], np.array(use_windows)])
    drop_last = windows is not None and windows[-1] < ts.sequence_length
    if drop_last:
        use_windows = np.concatenate([np.array(use_windows), [ts.sequence_length]])
    Sigma = ts.genetic_relatedness(
        sample_sets=[[i] for i in ts.samples()],
        indexes=[(i, j) for i in range(ts.num_samples) for j in range(ts.num_samples)],
        windows=use_windows,
        mode="branch",
        span_normalise=False,
        proportion=False,
        centre=centre,
    )
    if windows is not None:
        if drop_first:
            Sigma = Sigma[1:]
        if drop_last:
            Sigma = Sigma[:-1]
    nwin = 1 if windows is None else len(windows) - 1
    shape = (nwin, ts.num_samples, ts.num_samples)
    Sigma = Sigma.reshape(shape)
    out = np.array([x[np.ix_(keep_rows, keep_cols)] for x in Sigma])
    if windows is None:
        out = out[0]
    return out


def verify_relatedness_vector(
    ts, w, windows, *, internal_checks=False, verbosity=0, centre=True, nodes=None
):
    R1 = relatedness_vector(
        ts,
        sample_weights=w,
        windows=windows,
        internal_checks=internal_checks,
        verbosity=verbosity,
        centre=centre,
        nodes=nodes,
    )
    nrows = ts.num_samples if nodes is None else len(nodes)
    wvec = w if len(w.shape) > 1 else w[:, np.newaxis]
    Sigma = relatedness_matrix(ts, windows=windows, centre=centre, nodes=nodes)
    if windows is None:
        R2 = Sigma.dot(wvec)
    else:
        R2 = np.zeros((len(windows) - 1, nrows, wvec.shape[1]))
        for k in range(len(windows) - 1):
            R2[k] = Sigma[k].dot(wvec)
    R3 = ts.genetic_relatedness_vector(
        w, windows=windows, mode="branch", centre=centre, nodes=nodes
    )
    if verbosity > 0:
        print(ts.draw_text())
        print("weights:", w)
        print("windows:", windows)
        print("centre:", centre)
        print("here:", R1)
        print("with ts:", R2)
        print("with lib:", R3)
        print("Sigma:", Sigma)
    if windows is None:
        assert R1.shape == (nrows, wvec.shape[1])
    else:
        assert R1.shape == (len(windows) - 1, nrows, wvec.shape[1])
    np.testing.assert_allclose(R1, R2, atol=1e-10)
    np.testing.assert_allclose(R1, R3, atol=1e-10)
    return R1


def check_relatedness_vector(
    ts,
    n=2,
    num_windows=0,
    *,
    internal_checks=False,
    verbosity=0,
    seed=123,
    centre=True,
    do_nodes=True,
):
    rng = np.random.default_rng(seed=seed)
    if num_windows == 0:
        windows = None
    elif num_windows % 2 == 0:
        windows = np.linspace(
            0.2 * ts.sequence_length, 0.8 * ts.sequence_length, num_windows + 1
        )
    else:
        windows = np.linspace(0, ts.sequence_length, num_windows + 1)
    num_nodes_list = (0,) if (centre or not do_nodes) else (0, 3)
    for num_nodes in num_nodes_list:
        if num_nodes == 0:
            nodes = None
        else:
            nodes = rng.choice(ts.num_nodes, num_nodes, replace=False)
        for k in range(n):
            if k == 0:
                w = rng.normal(size=ts.num_samples)
            else:
                w = rng.normal(size=ts.num_samples * k).reshape((ts.num_samples, k))
            w = np.round(len(w) * w)
            R = verify_relatedness_vector(
                ts,
                w,
                windows,
                internal_checks=internal_checks,
                verbosity=verbosity,
                centre=centre,
                nodes=nodes,
            )
    return R


class TestRelatednessVector:

    def test_bad_weights(self):
        n = 5
        ts = msprime.sim_ancestry(
            n,
            ploidy=2,
            sequence_length=10,
            random_seed=123,
        )
        for bad_W in (None, [1], np.ones((3 * n, 2)), np.ones((n - 1, 2))):
            with pytest.raises(ValueError, match="number of samples"):
                ts.genetic_relatedness_vector(bad_W, mode="branch")

    def test_bad_windows(self):
        n = 5
        ts = msprime.sim_ancestry(
            n,
            ploidy=2,
            sequence_length=10,
            random_seed=123,
        )
        for bad_w in ([1], []):
            with pytest.raises(ValueError, match="Windows array"):
                ts.genetic_relatedness_vector(
                    np.ones(ts.num_samples), windows=bad_w, mode="branch"
                )

    def test_nodes_centred_error(self):
        ts = msprime.sim_ancestry(
            5,
            ploidy=2,
            sequence_length=10,
            random_seed=123,
        )
        with pytest.raises(ValueError, match="must have centre"):
            ts.genetic_relatedness_vector(
                np.ones(ts.num_samples), mode="branch", centre=True, nodes=[0, 1]
            )

    def test_bad_nodes(self):
        n = 5
        ts = msprime.sim_ancestry(
            n,
            ploidy=2,
            sequence_length=10,
            random_seed=123,
        )
        for bad_nodes in ([[]], "foo"):
            with pytest.raises(ValueError):
                ts.genetic_relatedness_vector(
                    np.ones(ts.num_samples),
                    mode="branch",
                    centre=False,
                    nodes=bad_nodes,
                )
        for bad_nodes in ([-1, 10], [3, 2 * ts.num_nodes]):
            with pytest.raises(tskit.LibraryError, match="TSK_ERR_NODE_OUT_OF_BOUNDS"):
                ts.genetic_relatedness_vector(
                    np.ones(ts.num_samples),
                    mode="branch",
                    centre=False,
                    nodes=bad_nodes,
                )

    def test_good_nodes(self):
        n = 5
        ts = msprime.sim_ancestry(
            n,
            ploidy=2,
            sequence_length=10,
            random_seed=123,
        )
        V0 = ts.genetic_relatedness_vector(
            np.ones(ts.num_samples), mode="branch", centre=False
        )
        V = ts.genetic_relatedness_vector(
            np.ones(ts.num_samples),
            mode="branch",
            centre=False,
            nodes=list(ts.samples()),
        )
        np.testing.assert_allclose(V0, V, atol=1e-13)
        V = ts.genetic_relatedness_vector(
            np.ones(ts.num_samples),
            mode="branch",
            centre=False,
            nodes=np.fromiter(ts.samples(), dtype=np.int32),
        )
        np.testing.assert_allclose(V0, V, atol=1e-13)
        V = ts.genetic_relatedness_vector(
            np.ones(ts.num_samples),
            mode="branch",
            centre=False,
            nodes=np.fromiter(ts.samples(), dtype=np.int64),
        )
        np.testing.assert_allclose(V0, V, atol=1e-13)
        V = ts.genetic_relatedness_vector(
            np.ones(ts.num_samples),
            mode="branch",
            centre=False,
            nodes=list(ts.samples())[:2],
        )
        np.testing.assert_allclose(V0[:2], V, atol=1e-13)

    @pytest.mark.parametrize("n", [2, 3, 5])
    @pytest.mark.parametrize("seed", range(1, 4))
    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 1, 2, 3))
    def test_small_internal_checks(self, n, seed, centre, num_windows):
        ts = msprime.sim_ancestry(
            n,
            ploidy=1,
            sequence_length=1000,
            recombination_rate=0.01,
            random_seed=seed,
        )
        assert ts.num_trees >= 2
        check_relatedness_vector(
            ts, num_windows=num_windows, internal_checks=True, centre=centre
        )

    @pytest.mark.parametrize("n", [2, 3, 5, 15])
    @pytest.mark.parametrize("seed", range(1, 5))
    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 1, 2, 3))
    def test_simple_sims(self, n, seed, centre, num_windows):
        ts = msprime.sim_ancestry(
            n,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=seed,
        )
        assert ts.num_trees >= 2
        check_relatedness_vector(
            ts, num_windows=num_windows, centre=centre, verbosity=0
        )

    def test_simple_sims_windows(self):
        L = 100
        ts = msprime.sim_ancestry(
            5,
            ploidy=1,
            population_size=20,
            sequence_length=L,
            recombination_rate=0.01,
            random_seed=345,
        )
        assert ts.num_trees >= 2
        W = np.linspace(0, 1, 2 * ts.num_samples).reshape((ts.num_samples, 2))
        kwargs = {"centre": False, "mode": "branch"}
        total = ts.genetic_relatedness_vector(W, **kwargs)
        for windows in [[0, L], [0, L / 3, L / 2, L]]:
            pieces = ts.genetic_relatedness_vector(W, windows=windows, **kwargs)
            np.testing.assert_allclose(total, pieces.sum(axis=0), atol=1e-13)
            assert len(pieces) == len(windows) - 1
            for k in range(len(pieces)):
                piece = ts.genetic_relatedness_vector(
                    W, windows=windows[k : k + 2], **kwargs
                )
                assert piece.shape[0] == 1
                np.testing.assert_allclose(piece[0], pieces[k], atol=1e-13)

    @pytest.mark.parametrize("n", [2, 3, 5, 15])
    @pytest.mark.parametrize("centre", (True, False))
    def test_single_balanced_tree(self, n, centre):
        ts = tskit.Tree.generate_balanced(n).tree_sequence
        check_relatedness_vector(ts, internal_checks=True, centre=centre)

    @pytest.mark.parametrize("centre", (True, False))
    def test_internal_sample(self, centre):
        tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
        flags = tables.nodes.flags
        flags[3] = 0
        flags[5] = tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        ts = tables.tree_sequence()
        check_relatedness_vector(ts, centre=centre)

    @pytest.mark.parametrize("seed", range(1, 5))
    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 1, 2, 3))
    def test_one_internal_sample_sims(self, seed, centre, num_windows):
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
        check_relatedness_vector(ts, num_windows=num_windows, centre=centre)

    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 1, 2, 3))
    def test_missing_flanks(self, centre, num_windows):
        ts = msprime.sim_ancestry(
            2,
            ploidy=1,
            population_size=10,
            sequence_length=100,
            recombination_rate=0.001,
            random_seed=1234,
        )
        assert ts.num_trees >= 2
        ts = ts.keep_intervals([[20, 80]])
        assert ts.first().interval == (0, 20)
        check_relatedness_vector(ts, num_windows=num_windows, centre=centre)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    @pytest.mark.parametrize("centre", (True, False))
    def test_suite_examples(self, ts, centre):
        if ts.num_samples > 0:
            check_relatedness_vector(ts, centre=centre)

    @pytest.mark.parametrize("n", [2, 3, 10])
    def test_dangling_on_samples(self, n):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(n).tree_sequence
        D1 = check_relatedness_vector(ts1, do_nodes=False)
        tables = ts1.dump_tables()
        for u in ts1.samples():
            v = tables.nodes.add_row(time=-1)
            tables.edges.add_row(left=0, right=ts1.sequence_length, parent=u, child=v)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_relatedness_vector(ts2, internal_checks=True, do_nodes=False)
        np.testing.assert_array_almost_equal(D1, D2)

    @pytest.mark.parametrize("n", [2, 3, 10])
    @pytest.mark.parametrize("centre", (True, False))
    def test_dangling_on_all(self, n, centre):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(n).tree_sequence
        D1 = check_relatedness_vector(ts1, centre=centre, do_nodes=False)
        tables = ts1.dump_tables()
        for u in range(ts1.num_nodes):
            v = tables.nodes.add_row(time=-1)
            tables.edges.add_row(left=0, right=ts1.sequence_length, parent=u, child=v)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_relatedness_vector(
            ts2, internal_checks=True, centre=centre, do_nodes=False
        )
        np.testing.assert_array_almost_equal(D1, D2)

    @pytest.mark.parametrize("centre", (True, False))
    def test_disconnected_non_sample_topology(self, centre):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(5).tree_sequence
        D1 = check_relatedness_vector(ts1, centre=centre, do_nodes=False)
        tables = ts1.dump_tables()
        # Add an extra bit of disconnected non-sample topology
        u = tables.nodes.add_row(time=0)
        v = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=0, right=ts1.sequence_length, parent=v, child=u)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_relatedness_vector(
            ts2, internal_checks=True, centre=centre, do_nodes=False
        )
        np.testing.assert_array_almost_equal(D1, D2)


def pca(ts, windows, centre, samples=None, individuals=None, time_windows=None):
    assert samples is None or individuals is None
    if samples is None:
        ii = np.arange(ts.num_samples)
    else:
        all_samples = ts.samples()
        ii = np.searchsorted(all_samples, samples)
    drop_dimension = windows is None
    if drop_dimension:
        windows = [0, ts.sequence_length]
    if time_windows is None:
        Sigma = relatedness_matrix(ts=ts, windows=windows, centre=False)[:, ii, :][
            :, :, ii
        ]
    else:
        assert time_windows[0] < time_windows[1]
        ts_low, ts_high = (
            ts.decapitate(time_windows[0]),
            ts.decapitate(time_windows[1]),
        )
        Sigma_low = relatedness_matrix(ts=ts_low, windows=windows, centre=False)[
            :, ii, :
        ][:, :, ii]
        Sigma_high = relatedness_matrix(ts=ts_high, windows=windows, centre=False)[
            :, ii, :
        ][:, :, ii]
        Sigma = Sigma_high - Sigma_low
    if individuals is not None:
        ni = len(individuals)
        J = np.zeros((ts.num_samples, ni))
        for k, i in enumerate(individuals):
            nn = ts.individual(i).nodes
            for j in nn:
                J[j, k] = 1 / len(nn)
        Sigma = np.matmul(J.T, np.matmul(Sigma, J))
    if centre:
        n = Sigma.shape[-1]
        P = np.eye(n) - 1 / n
        Sigma = np.matmul(P, np.matmul(Sigma, P))
    U, S, _ = np.linalg.svd(Sigma, hermitian=True)
    if drop_dimension:
        U = U[0]
        S = S[0]
    return U, S


def allclose_up_to_sign(x, y, **kwargs):
    # check if two vectors are the same up to sign
    x_const = np.isclose(np.std(x), 0)
    y_const = np.isclose(np.std(y), 0)
    if x_const or y_const:
        if np.allclose(x, 0):
            r = 1.0
        else:
            r = np.mean(x / y)
    else:
        r = np.sign(np.corrcoef(x, y)[0, 1])
    return np.allclose(x, r * y, **kwargs)


def assert_pcs_equal(U, D, U_full, D_full, rtol=1e-5, atol=1e-8):
    # check that the PCs in U, D occur in U_full, D_full
    # accounting for sign and ordering
    assert len(D) <= len(D_full)
    assert U.shape[0] == U_full.shape[0]
    assert U.shape[1] == len(D)
    for k in range(len(D)):
        u = U[:, k]
        d = D[k]
        (ii,) = np.where(np.isclose(D_full, d, rtol=rtol, atol=atol))
        assert len(ii) > 0, f"{k}th singular value {d} not found in {D_full}."
        found_it = False
        for i in ii:
            if allclose_up_to_sign(u, U_full[:, i], rtol=rtol, atol=atol):
                found_it = True
                break
        assert found_it, f"{k}th singular vector {u} not found in {U_full}."


def assert_errors_bound(pca_res, D, U, w=None):
    # Bounds on the error are from equation 1.11 in https://arxiv.org/pdf/0909.4061 -
    # this gives a bound on reconstruction error (i.e., operator norm between the GRM
    # and the low-diml approx). But since the (L2) operator norm is
    # |X| = sup_v |Xv|/|v|,
    # this implies bounds on singular values and vectors also:
    # If G v = lambda v, and we've got estimated singular vectors U and values diag(L),
    # then let v = \sum_i b_i u_i + delta be the projection of v into U,
    # and we have that
    #  |lambda v - U L U* v|^2
    #   = \sum_i b_i^2 (lambda - L_i)^2 + lambda^2 |delta|^2
    #   < \epsilon^2   (where epsilon is the spectral norm bound error_bound)
    # so
    #  |delta| < epsilon / lambda
    # since this is the amount by which the eigenvector v isn't hit by the columns of U.
    # Then also for each i that if b_i is not small then
    #  |lambda - L_i| < epsilon
    # and there must be at least one b_i that is big (since sum_i b_i^2 = 1 - |delta|^2).
    # More concretely, let m = min_i |lambda - L_i|^2,
    # so that
    #  epsilon^2 > \sum_i (lambda - L_i)^2 b_i^2 + lambda^2 |delta|^2
    #   >= m * \sum_i b_i^2 + lambda^2 |delta|^2
    #   = m * (1-|delta|^2) + lambda^2 |delta|^2.
    # Hence,
    # min_i |lambda-L_i|^2 = m < (epsilon^2 - lambda^2  |delta|^2) / (1- |delta|^2).
    # In summary: roughly, epsilon should be the bound on error in eigenvalues,
    # and epsilon / sigma[k+1] the L2 bound for eigenvectors
    # Below, the 'roughly/should be' translates into the factor of 5.

    f = pca_res.factors
    ev = pca_res.eigenvalues
    rs = pca_res.range_sketch
    eps = pca_res.error_bound
    if w is not None:
        D, U = D[w], U[w]
        f, ev, rs, eps = f[w], ev[w], rs[w], eps[w]
    n = ev.shape[0]
    Sigma = U @ np.diag(D) @ U.T
    Q = rs[:, :n]
    err = np.linalg.svd(Sigma - Q @ Q.T @ Sigma).S[0]
    assert (
        err <= 5 * eps**2
    ), "Reconstruction error should be smaller than the bound squared."
    assert (
        np.max(np.abs(ev - D[:n])) < 5 * eps
    ), "Eigenvalue error should be smaller than error bound."
    for k in range(n):
        assert (
            np.sum((f[:, k] - U[:, k]) ** 2) < 5 * eps**2 / ev[-1]
        ), "Factor error should be smaller than the bound squared."


class TestPCA:

    def verify_error_est(
        self,
        ts,
        num_windows,
        num_components,
        centre,
        samples=None,
        individuals=None,
        time_windows=None,
        **kwargs,
    ):
        assert samples is None or individuals is None
        if num_windows == 0:
            windows = None
        elif num_windows % 2 == 0:
            windows = np.linspace(
                0.2 * ts.sequence_length, 0.8 * ts.sequence_length, num_windows + 1
            )
        else:
            windows = np.linspace(0, ts.sequence_length, num_windows + 1)
        if samples is not None:
            num_rows = len(samples)
        elif individuals is not None:
            num_rows = len(individuals)
        else:
            num_rows = ts.num_samples
        num_oversamples = kwargs.get(
            "num_oversamples", min(num_rows - num_components, 10)
        )
        pca_res = ts.pca(
            windows=windows,
            samples=samples,
            individuals=individuals,
            num_components=num_components,
            centre=centre,
            time_windows=time_windows,
            random_seed=1238,
            **kwargs,
        )
        if windows is None:
            assert pca_res.factors.shape == (num_rows, num_components)
            assert pca_res.eigenvalues.shape == (num_components,)
            assert pca_res.range_sketch.shape == (
                num_rows,
                num_components + num_oversamples,
            )
            assert pca_res.error_bound.shape == ()
        else:
            assert pca_res.factors.shape == (num_windows, num_rows, num_components)
            assert pca_res.eigenvalues.shape == (num_windows, num_components)
            assert pca_res.range_sketch.shape == (
                num_windows,
                num_rows,
                num_components + num_oversamples,
            )
            assert pca_res.error_bound.shape == (num_windows,)
        U, D = pca(
            ts=ts,
            windows=windows,
            centre=centre,
            samples=samples,
            individuals=individuals,
            time_windows=time_windows,
        )
        if windows is None:
            assert_errors_bound(pca_res, D, U)
        else:
            for w in range(num_windows):
                assert_errors_bound(pca_res, D, U, w=w)

    def verify_pca(
        self,
        ts,
        num_windows,
        num_components,
        centre,
        samples=None,
        individuals=None,
        time_windows=None,
        **kwargs,
    ):
        assert samples is None or individuals is None
        if num_windows == 0:
            windows = None
        elif num_windows % 2 == 0:
            windows = np.linspace(
                0.2 * ts.sequence_length, 0.8 * ts.sequence_length, num_windows + 1
            )
        else:
            windows = np.linspace(0, ts.sequence_length, num_windows + 1)
        if samples is not None:
            num_rows = len(samples)
        elif individuals is not None:
            num_rows = len(individuals)
        else:
            num_rows = ts.num_samples
        num_oversamples = kwargs.get(
            "num_oversamples", min(num_rows - num_components, 10)
        )
        pca_res = ts.pca(
            windows=windows,
            samples=samples,
            individuals=individuals,
            num_components=num_components,
            centre=centre,
            time_windows=time_windows,
            random_seed=1238,
            **kwargs,
        )
        if windows is None:
            assert pca_res.factors.shape == (num_rows, num_components)
            assert pca_res.eigenvalues.shape == (num_components,)
            assert pca_res.range_sketch.shape == (
                num_rows,
                num_components + num_oversamples,
            )
            assert pca_res.error_bound.shape == ()
        else:
            assert pca_res.factors.shape == (num_windows, num_rows, num_components)
            assert pca_res.eigenvalues.shape == (num_windows, num_components)
            assert pca_res.range_sketch.shape == (
                num_windows,
                num_rows,
                num_components + num_oversamples,
            )
            assert pca_res.error_bound.shape == (num_windows,)
        U, D = pca(
            ts=ts,
            windows=windows,
            centre=centre,
            samples=samples,
            individuals=individuals,
            time_windows=time_windows,
        )
        if windows is None:
            np.testing.assert_allclose(
                pca_res.eigenvalues, D[:num_components], atol=1e-8
            )
            assert_pcs_equal(pca_res.factors, pca_res.eigenvalues, U, D)
        else:
            for w in range(num_windows):
                np.testing.assert_allclose(
                    pca_res.eigenvalues[w], D[w, :num_components], atol=1e-8
                )
                assert_pcs_equal(pca_res.factors[w], pca_res.eigenvalues[w], U[w], D[w])

    def test_bad_windows(self):
        ts = msprime.sim_ancestry(
            3,
            sequence_length=10,
            random_seed=123,
        )
        for bad_w in ([], [1]):
            with pytest.raises(ValueError, match="at least one window"):
                ts.pca(num_components=2, windows=bad_w)
        for bad_w in ([1, 0], [-3, 10]):
            with pytest.raises(tskit.LibraryError, match="TSK_ERR_BAD_WINDOWS"):
                ts.pca(num_components=2, windows=bad_w)

    def test_bad_params(self):
        ts = msprime.sim_ancestry(
            3,
            sequence_length=10,
            random_seed=123,
        )
        _ = ts.pca(num_components=3)
        with pytest.raises(ValueError, match="Number of components"):
            ts.pca(num_components=ts.num_samples + 1)
        with pytest.raises(ValueError, match="Number of components"):
            ts.pca(num_components=4, samples=[0, 1, 2])
        with pytest.raises(ValueError, match="Number of components"):
            ts.pca(num_components=4, individuals=[0, 1])
        with pytest.raises(ValueError, match="num_components \\+ num_oversamples"):
            ts.pca(num_components=2, num_oversamples=ts.num_samples)
        with pytest.raises(ValueError, match="Cannot specify both num_over"):
            ts.pca(
                num_components=2,
                num_oversamples=2,
                range_sketch=np.zeros((ts.num_samples, 4)),
            )
        with pytest.raises(ValueError, match="num_iterations should be"):
            ts.pca(num_components=3, num_iterations=-1)
        with pytest.raises(ValueError, match="num_iterations should be"):
            ts.pca(num_components=3, num_iterations=0)
        with pytest.raises(ValueError, match="num_iterations should be"):
            ts.pca(num_components=3, num_iterations="bac")
        with pytest.raises(ValueError, match="num_iterations should be"):
            ts.pca(num_components=3, num_iterations=[])

    def test_bad_range_sketch(self):
        ts = msprime.sim_ancestry(
            3,
            sequence_length=10,
            random_seed=123,
        )
        nc = 2
        # too few rows
        Q = np.zeros((ts.num_samples - 1, ts.num_samples))
        with pytest.raises(ValueError, match="Incorrect shape of range"):
            ts.pca(num_components=nc, range_sketch=Q)
        # too many rows
        Q = np.zeros((ts.num_samples + 1, ts.num_samples))
        with pytest.raises(ValueError, match="Incorrect shape of range"):
            ts.pca(num_components=nc, range_sketch=Q)
        # too few columns
        Q = np.zeros((ts.num_samples, nc - 1))
        with pytest.raises(ValueError, match="must have at least as many"):
            ts.pca(num_components=nc, range_sketch=Q)
        # too many columns
        Q = np.zeros((ts.num_samples, nc + ts.num_samples))
        with pytest.raises(ValueError, match="must be less than"):
            ts.pca(num_components=nc, range_sketch=Q)
        # not enough dimensions
        Q = np.zeros((ts.num_samples,))
        with pytest.raises(ValueError, match="Incorrect shape of range"):
            ts.pca(num_components=nc, range_sketch=Q)
        # not enough dimensions, with windows
        Q = np.zeros((ts.num_samples, nc + 2))
        with pytest.raises(ValueError, match="Incorrect shape of range"):
            ts.pca(num_components=nc, windows=[0, 10], range_sketch=Q)
        # not enough windows
        Q = np.zeros((ts.num_samples, 1, nc + 2))
        with pytest.raises(ValueError, match="Incorrect shape of range"):
            ts.pca(num_components=nc, windows=[0, 5, 10], range_sketch=Q)

    def test_indivs_and_samples(self):
        ts = msprime.sim_ancestry(
            3,
            ploidy=2,
            sequence_length=10,
            random_seed=123,
        )
        with pytest.raises(ValueError, match="Samples and individuals"):
            ts.pca(num_components=2, samples=[0, 1, 2, 3], individuals=[0, 1, 2])

    def test_modes(self):
        ts = msprime.sim_ancestry(
            3,
            sequence_length=10,
            random_seed=123,
        )
        for bad_mode in ("site", "node"):
            with pytest.raises(
                tskit.LibraryError, match="TSK_ERR_UNSUPPORTED_STAT_MODE"
            ):
                ts.pca(num_components=2, mode=bad_mode)

    @pytest.mark.parametrize("n", [2, 3, 5, 15])
    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 1, 2, 3))
    @pytest.mark.parametrize("num_components", (1, 3))
    def test_simple_sims(self, n, centre, num_windows, num_components):
        ploidy = 2
        nc = min(num_components, n * ploidy)
        ts = msprime.sim_ancestry(
            n,
            ploidy=ploidy,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=12345,
        )
        kwargs = {}
        # with n=15 and the default of 5 iterations, the relative tolerance on
        # the eigenvectors is only 1e-4; so, up this:
        if n > 10:
            kwargs["num_iterations"] = 10
        self.verify_pca(
            ts, num_windows=num_windows, num_components=nc, centre=centre, **kwargs
        )

    def test_range_sketch(self):
        n = 10
        ploidy = 2
        ts = msprime.sim_ancestry(
            n,
            ploidy=2,
            sequence_length=100,
            random_seed=123,
        )
        nc, no = 2, 3
        # should work as long as columns are linearly independent
        range_sketch = np.linspace(0, 1, n * ploidy * (nc + no)).reshape(
            (n * ploidy, nc + no)
        )
        pca_res0 = ts.pca(num_components=nc)
        pca_res1 = ts.pca(
            num_components=nc, range_sketch=range_sketch, num_iterations=20
        )
        assert_pcs_equal(
            pca_res0.factors,
            pca_res0.eigenvalues,
            pca_res1.factors,
            pca_res1.eigenvalues,
        )
        # check we can recycle previously returned sketches
        pca_res_1 = ts.pca(num_components=nc, range_sketch=None)
        for _ in range(20):
            pca_res_1 = ts.pca(num_components=nc, range_sketch=pca_res_1.range_sketch)
        assert_pcs_equal(
            pca_res0.factors,
            pca_res0.eigenvalues,
            pca_res1.factors,
            pca_res1.eigenvalues,
        )

    def test_num_iterations(self):
        n = 10
        ploidy = 2
        ts = msprime.sim_ancestry(
            n,
            ploidy=2,
            sequence_length=100,
            random_seed=123,
        )
        nc, no = 2, 3
        range_sketch = np.linspace(0, 1, n * ploidy * (nc + no)).reshape(
            (n * ploidy, nc + no)
        )
        pca_res0 = ts.pca(
            num_components=nc, range_sketch=range_sketch, num_iterations=5
        )
        pca_res1 = ts.pca(
            num_components=nc, range_sketch=range_sketch, num_iterations=1
        )
        for _ in range(4):
            pca_res1 = ts.pca(
                num_components=nc, range_sketch=pca_res1.range_sketch, num_iterations=1
            )
        assert_pcs_equal(
            pca_res0.factors,
            pca_res0.eigenvalues,
            pca_res1.factors,
            pca_res1.eigenvalues,
        )

    def test_seed(self):
        ts = msprime.sim_ancestry(
            4,
            ploidy=2,
            sequence_length=100,
            random_seed=345,
        )
        pc1 = ts.pca(num_components=3, random_seed=123)
        pc2 = ts.pca(num_components=3, random_seed=123)
        assert np.all(pc1.factors == pc2.factors)
        assert np.all(pc1.eigenvalues == pc2.eigenvalues)
        assert np.all(pc1.range_sketch == pc2.range_sketch)
        assert np.all(pc1.error_bound == pc2.error_bound)

    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 2))
    def test_samples(self, centre, num_windows):
        ploidy = 2
        ts = msprime.sim_ancestry(
            20,
            ploidy=ploidy,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=12345,
        )
        samples = [3, 0, 2, 5, 6, 15, 12, 17, 7, 9, 11]
        time_low, time_high = (ts.nodes_time.max() / 4, ts.nodes_time.max() / 2)
        self.verify_pca(
            ts,
            num_windows=num_windows,
            num_components=5,
            centre=centre,
            samples=samples,
            time_windows=[time_low, time_high],
        )

    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 2))
    def test_err_samples(self, centre, num_windows):
        ploidy = 2
        ts = msprime.sim_ancestry(
            20,
            ploidy=ploidy,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=12345,
        )
        samples = [3, 0, 2, 5, 6, 15, 12, 17, 7, 9, 11]
        time_low, time_high = (ts.nodes_time.max() / 4, ts.nodes_time.max() / 2)
        self.verify_error_est(
            ts,
            num_windows=num_windows,
            num_components=4,
            centre=centre,
            samples=samples,
            time_windows=[time_low, time_high],
        )

    @pytest.mark.parametrize("centre", (True, False))
    def test_individuals_matches_samples(self, centre):
        # ploidy 1 individuals should be the same as samples
        ploidy = 1
        ts = msprime.sim_ancestry(
            20,
            ploidy=ploidy,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=12345,
        )
        individuals = [3, 0, 2, 5, 6, 15, 12]
        for i in individuals:
            assert ts.individual(i).nodes == [
                i,
            ]
        pci = pca(
            ts, windows=[0, ts.sequence_length], centre=centre, samples=individuals
        )
        pcs = pca(
            ts, windows=[0, ts.sequence_length], centre=centre, individuals=individuals
        )
        tspci = ts.pca(
            num_components=5, centre=centre, samples=individuals, random_seed=456
        )
        tspcs = ts.pca(
            num_components=5, centre=centre, individuals=individuals, random_seed=456
        )
        assert np.all(pci[0] == pcs[0])
        assert np.all(pci[1] == pcs[1])
        assert np.all(tspci.factors == tspcs.factors)
        assert np.all(tspci.eigenvalues == tspcs.eigenvalues)
        pci = ts.pca(
            num_components=5,
            windows=[0, 50, 100],
            centre=centre,
            samples=individuals,
            random_seed=456,
        )
        pcs = ts.pca(
            num_components=5,
            windows=[0, 50, 100],
            centre=centre,
            individuals=individuals,
            random_seed=456,
        )
        assert np.all(pci.factors == pcs.factors)
        assert np.all(pci.eigenvalues == pcs.eigenvalues)

    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 2))
    @pytest.mark.parametrize("ploidy", (1, 2, 3))
    def test_individuals(self, centre, num_windows, ploidy):
        ts = msprime.sim_ancestry(
            20,
            ploidy=ploidy,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=12345,
        )
        individuals = [3, 0, 2, 5, 6, 15, 12, 11, 7, 17]
        time_low, time_high = (ts.nodes_time.max() / 4, ts.nodes_time.max() / 2)
        self.verify_pca(
            ts,
            num_windows=num_windows,
            num_components=5,
            centre=centre,
            individuals=individuals,
            time_windows=[time_low, time_high],
        )

    @pytest.mark.parametrize("centre", (True, False))
    @pytest.mark.parametrize("num_windows", (0, 2))
    @pytest.mark.parametrize("ploidy", (1, 2, 3))
    def test_err_individuals(self, centre, num_windows, ploidy):
        # NOTE: this is a randomized test, so if things change under the
        # hood it might start to fail for perfectly normal (ie unlucky) reasons.
        # If so, it's probably better to replace the test with a simpler test,
        # e.g., that error_bound is roughly the right order of magnitude.
        ts = msprime.sim_ancestry(
            30,
            ploidy=ploidy,
            population_size=30,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=12346,
        )
        individuals = np.arange(30)
        time_low, time_high = (ts.nodes_time.max() / 4, ts.nodes_time.max() / 2)
        self.verify_error_est(
            ts,
            num_windows=num_windows,
            num_components=5,
            centre=centre,
            individuals=individuals,
            time_windows=[time_low, time_high],
        )
