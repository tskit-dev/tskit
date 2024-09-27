# MIT License
#
# Copyright (c) 2024 Tskit Developers
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
from tests.test_highlevel import get_example_tree_sequences

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

    tree_pos = tsutil.TreePosition(ts)
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


class TestExamples:

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
