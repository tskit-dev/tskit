import msprime
import numpy as np
import pytest

import _tskit
import tests.test_wright_fisher as wf
import tskit
from tests import tsutil
from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


def extend_edges(ts, max_iter=10):
    tables = ts.dump_tables()
    mutations = tables.mutations.copy()
    tables.mutations.clear()

    last_num_edges = ts.num_edges
    for _ in range(max_iter):
        for forwards in [True, False]:
            edges = _extend(ts, forwards=forwards)
            tables.edges.replace_with(edges)
            tables.build_index()
            ts = tables.tree_sequence()
        if ts.num_edges == last_num_edges:
            break
        else:
            last_num_edges = ts.num_edges

    tables = ts.dump_tables()
    mutations = _slide_mutation_nodes_up(ts, mutations)
    tables.mutations.replace_with(mutations)
    ts = tables.tree_sequence()

    return ts


def _slide_mutation_nodes_up(ts, mutations):
    # adjusts mutations' nodes to place each mutation on the correct edge given
    # their time; requires mutation times be nonmissing and the mutation times
    # be >= their nodes' times.

    assert np.all(~tskit.is_unknown_time(mutations.time)), "times must be known"
    new_nodes = mutations.node.copy()

    mut = 0
    for tree in ts.trees():
        _, right = tree.interval
        while (
            mut < mutations.num_rows and ts.sites_position[mutations.site[mut]] < right
        ):
            t = mutations.time[mut]
            c = mutations.node[mut]
            p = tree.parent(c)
            assert ts.nodes_time[c] <= t
            while p != -1 and ts.nodes_time[p] <= t:
                c = p
                p = tree.parent(c)
            assert ts.nodes_time[c] <= t
            if p != -1:
                assert t < ts.nodes_time[p]
            new_nodes[mut] = c
            mut += 1

    # in C the node column can be edited in place
    new_mutations = mutations.copy()
    new_mutations.clear()
    for mut, n in zip(mutations, new_nodes):
        new_mutations.append(mut.replace(node=n))

    return new_mutations


def _extend(ts, forwards=True):
    # `degree` will record the degree of each node in the tree we'd get if
    # we removed all `out` edges and added all `in` edges
    degree = np.full(ts.num_nodes, 0, dtype="int")
    # `out_parent` will record the sub-forest of edges-to-be-removed
    out_parent = np.full(ts.num_nodes, -1, dtype="int")
    keep = np.full(ts.num_edges, True, dtype=bool)
    not_sample = [not n.is_sample() for n in ts.nodes()]

    edges = ts.tables.edges.copy()

    # "here" will be left if fowards else right;
    # and "there" is the other
    new_left = edges.left.copy()
    new_right = edges.right.copy()
    if forwards:
        direction = 1
        # in C we can just modify these in place, but in
        # python they are (silently) immutable
        near_side = new_left
        far_side = new_right
    else:
        direction = -1
        near_side = new_right
        far_side = new_left
    edges_out = []
    edges_in = []

    tree_pos = tsutil.TreePosition(ts)
    if forwards:
        valid = tree_pos.next()
    else:
        valid = tree_pos.prev()
    while valid:
        left, right = tree_pos.interval
        there = right if forwards else left

        # Clear out non-extended or postponed edges:
        # Note: maintaining out_parent is a bit tricky, because
        # if an edge from p->c has been extended, entirely replacing
        # another edge from p'->c, then both edges may be in edges_out,
        # and we only want to include the *first* one.
        for e, _ in edges_out:
            out_parent[edges.child[e]] = -1
        tmp = []
        for e, x in edges_out:
            if x:
                tmp.append([e, False])
        edges_out = tmp
        tmp = []
        for e, x in edges_in:
            if x:
                tmp.append([e, False])
        edges_in = tmp

        for e, _ in edges_out:
            out_parent[edges.child[e]] = edges.parent[e]

        for j in range(tree_pos.out_range.start, tree_pos.out_range.stop, direction):
            e = tree_pos.out_range.order[j]
            if out_parent[edges.child[e]] == -1:
                edges_out.append([e, False])
                out_parent[edges.child[e]] = edges.parent[e]

        for j in range(tree_pos.in_range.start, tree_pos.in_range.stop, direction):
            e = tree_pos.in_range.order[j]
            edges_in.append([e, False])

        for e, _ in edges_out:
            degree[edges.parent[e]] -= 1
            degree[edges.child[e]] -= 1
        for e, _ in edges_in:
            degree[edges.parent[e]] += 1
            degree[edges.child[e]] += 1

        # validate out_parent array
        for c, p in enumerate(out_parent):
            foundit = False
            for e, _ in edges_out:
                if edges.child[e] == c:
                    assert edges.parent[e] == p
                    foundit = True
                    break
            assert foundit == (p != -1)

        assert np.all(degree >= 0)
        for ex_in in edges_in:
            e_in = ex_in[0]
            # check whether the parent-child relationship exists
            # in the sub-forest of edges to be removed:
            # out_parent[p] != -1 only when it is the bottom of
            # an edge to be removed,
            # and degree[p] == 0 only if it is not in the new tree
            c = edges.child[e_in]
            p = out_parent[c]
            p_in = edges.parent[e_in]
            while p != tskit.NULL and degree[p] == 0 and p != p_in and not_sample[p]:
                p = out_parent[p]
            if p == p_in:
                # we might have passed the interval that a
                # postponed edge in covers, in which case
                # we should skip it
                if far_side[e_in] != there:
                    ex_in[1] = True
                near_side[e_in] = there
                while c != p:
                    # just loop over the edges out until we find the right entry
                    for ex_out in edges_out:
                        e_out = ex_out[0]
                        if edges.child[e_out] == c:
                            break
                    assert edges.child[e_out] == c
                    ex_out[1] = True
                    far_side[e_out] = there
                    # amend degree: the intermediate
                    # nodes have 2 edges instead of 0
                    assert degree[c] == 0 or c == edges.child[e_in]
                    if degree[c] == 0:
                        degree[c] = 2
                    c = out_parent[c]

        # end of loop, next tree
        if forwards:
            valid = tree_pos.next()
        else:
            valid = tree_pos.prev()

    for j in range(edges.num_rows):
        left = new_left[j]
        right = new_right[j]
        if left < right:
            edges[j] = edges[j].replace(left=left, right=right)
        else:
            keep[j] = False
    edges.keep_rows(keep)
    return edges


class TestExtendEdges:
    """
    Test the 'extend edges' method
    """

    def verify_extend_edges(self, ts, max_iter=10):
        # This can still fail for various weird examples:
        # for instance, if adjacent trees have
        # a <- b <- c <- d and a <- d (where say b was
        # inserted in an earlier pass), then b and c
        # won't be extended

        ets = ts.extend_edges(max_iter=max_iter)
        assert np.all(ts.genotype_matrix() == ets.genotype_matrix())
        assert ts.num_samples == ets.num_samples
        assert ts.num_nodes == ets.num_nodes
        assert ts.num_edges >= ets.num_edges
        t = ts.simplify().tables
        et = ets.simplify().tables
        t.assert_equals(et, ignore_provenance=True)
        old_edges = {}
        for e in ts.edges():
            k = (e.parent, e.child)
            if k not in old_edges:
                old_edges[k] = []
            old_edges[k].append((e.left, e.right))

        for e in ets.edges():
            # e should be in old_edges,
            # but with modified limits:
            # USUALLY overlapping limits, but
            # not necessarily after more than one pass
            k = (e.parent, e.child)
            assert k in old_edges
            if max_iter == 1:
                overlaps = False
                for left, right in old_edges[k]:
                    if (left <= e.right) and (right >= e.left):
                        overlaps = True
                assert overlaps

        if max_iter > 1:
            chains = []
            for _, tt, ett in ts.coiterate(ets):
                this_chains = []
                for a in tt.nodes():
                    assert a in ett.nodes()
                    b = tt.parent(a)
                    if b != tskit.NULL:
                        c = tt.parent(b)
                        if c != tskit.NULL:
                            this_chains.append((a, b, c))
                        assert b in ett.nodes()
                        # the relationship a <- b should still be in the tree
                        p = a
                        while p != tskit.NULL and p != b:
                            p = ett.parent(p)
                        assert p == b
                chains.append(this_chains)

            extended_ac = {}
            not_extended_ac = {}
            extended_ab = {}
            not_extended_ab = {}
            for k, (interval, tt, ett) in enumerate(ts.coiterate(ets)):
                for j in (k - 1, k + 1):
                    if j < 0 or j >= len(chains):
                        continue
                    else:
                        this_chains = chains[j]
                    for a, b, c in this_chains:
                        if (
                            a in tt.nodes()
                            and tt.parent(a) == c
                            and b not in tt.nodes()
                        ):
                            # the relationship a <- b <- c should still be in the tree,
                            # although maybe they aren't direct parent-offspring
                            # UNLESS we've got an ambiguous case, where on the opposite
                            # side of the interval a chain a <- b' <- c got extended
                            # into the region OR b got inserted into another chain
                            assert a in ett.nodes()
                            assert c in ett.nodes()
                            if b not in ett.nodes():
                                if (a, c) not in not_extended_ac:
                                    not_extended_ac[(a, c)] = []
                                not_extended_ac[(a, c)].append(interval)
                            else:
                                if (a, c) not in extended_ac:
                                    extended_ac[(a, c)] = []
                                extended_ac[(a, c)].append(interval)
                                p = a
                                while p != tskit.NULL and p != b:
                                    p = ett.parent(p)
                                if p != b:
                                    if (a, b) not in not_extended_ab:
                                        not_extended_ab[(a, b)] = []
                                    not_extended_ab[(a, b)].append(interval)
                                else:
                                    if (a, b) not in extended_ab:
                                        extended_ab[(a, b)] = []
                                    extended_ab[(a, b)].append(interval)
                                    while p != tskit.NULL and p != c:
                                        p = ett.parent(p)
                                    assert p == c
            for a, c in not_extended_ac:
                # check that a <- ... <- c has been extended somewhere
                # although not necessarily from an adjacent segment
                assert (a, c) in extended_ac
                for interval in not_extended_ac[(a, c)]:
                    ett = ets.at(interval.left)
                    assert ett.parent(a) != c
            for k in not_extended_ab:
                assert k in extended_ab
                for interval in not_extended_ab[k]:
                    assert interval in extended_ab[k]

        # finally, compare C version to python version
        py_ts = extend_edges(ts, max_iter=max_iter)
        py_et = py_ts.dump_tables()
        et = ets.dump_tables()
        et.assert_equals(py_et)

    def test_runs(self):
        ts = msprime.simulate(5, mutation_rate=1.0, random_seed=126)
        self.verify_extend_edges(ts)

    def test_migrations_disallowed(self):
        ts = msprime.simulate(5, mutation_rate=1.0, random_seed=126)
        tables = ts.dump_tables()
        tables.populations.add_row()
        tables.populations.add_row()
        tables.migrations.add_row(0, 1, 0, 0, 1, 0)
        ts = tables.tree_sequence()
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_MIGRATIONS_NOT_SUPPORTED"
        ):
            _ = ts.extend_edges()

    def test_unknown_times(self):
        ts = msprime.simulate(5, mutation_rate=1.0, random_seed=126)
        tables = ts.dump_tables()
        tables.mutations.clear()
        for mut in ts.mutations():
            tables.mutations.append(mut.replace(time=tskit.UNKNOWN_TIME))
        ts = tables.tree_sequence()
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_DISALLOWED_UNKNOWN_MUTATION_TIME"
        ):
            _ = ts.extend_edges()

    def test_max_iter(self):
        ts = msprime.simulate(5, random_seed=126)
        with pytest.raises(_tskit.LibraryError, match="positive"):
            ets = ts.extend_edges(max_iter=0)
        with pytest.raises(_tskit.LibraryError, match="positive"):
            ets = ts.extend_edges(max_iter=-1)
        ets = ts.extend_edges(max_iter=1)
        et = ets.extend_edges(max_iter=1).dump_tables()
        eet = ets.extend_edges(max_iter=2).dump_tables()
        eet.assert_equals(et)

    def get_simple_ex(self, samples=None):
        # An example where you need to go forwards *and* backwards:
        # 7 and 8 should be extended to the whole sequence
        #
        #    6          6      6         6
        #  +-+-+      +-+-+  +-+-+     +-+-+
        #  |   |      7   |  |   8     |   |
        #  |   |     ++-+ |  | +-++    |   |
        #  4   5     4  | |  4 |  5    4   5
        # +++ +++   +++ | |  | | +++  +++ +++
        # 0 1 2 3   0 1 2 3  0 1 2 3  0 1 2 3
        #
        # Result:
        #
        #    6         6      6         6
        #  +-+-+     +-+-+  +-+-+     +-+-+
        #  7   8     7   8  7   8     7   8
        #  |   |    ++-+ |  | +-++    |   |
        #  4   5    4  | 5  4 |  5    4   5
        # +++ +++  +++ | |  | | +++  +++ +++
        # 0 1 2 3  0 1 2 3  0 1 2 3  0 1 2 3

        node_times = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 1.0,
            5: 1.0,
            6: 3.0,
            7: 2.0,
            8: 2.0,
        }
        # (p, c, l, r)
        edges = [
            (4, 0, 0, 10),
            (4, 1, 0, 5),
            (4, 1, 7, 10),
            (5, 2, 0, 2),
            (5, 2, 5, 10),
            (5, 3, 0, 2),
            (5, 3, 5, 10),
            (7, 2, 2, 5),
            (7, 4, 2, 5),
            (8, 1, 5, 7),
            (8, 5, 5, 7),
            (6, 3, 2, 5),
            (6, 4, 0, 2),
            (6, 4, 5, 10),
            (6, 5, 0, 2),
            (6, 5, 7, 10),
            (6, 7, 2, 5),
            (6, 8, 5, 7),
        ]
        # here is the 'right answer' (but note only with the default args)
        extended_edges = [
            (4, 0, 0, 10),
            (4, 1, 0, 5),
            (4, 1, 7, 10),
            (5, 2, 0, 2),
            (5, 2, 5, 10),
            (5, 3, 0, 10),
            (7, 2, 2, 5),
            (7, 4, 0, 10),
            (8, 1, 5, 7),
            (8, 5, 0, 10),
            (6, 7, 0, 10),
            (6, 8, 0, 10),
        ]
        tables = tskit.TableCollection(sequence_length=10)
        if samples is None:
            samples = [0, 1, 2, 3]
        for n, t in node_times.items():
            flags = tskit.NODE_IS_SAMPLE if n in samples else 0
            tables.nodes.add_row(time=t, flags=flags)
        for p, c, l, r in edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ts = tables.tree_sequence()
        tables.edges.clear()
        for p, c, l, r in extended_edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ets = tables.tree_sequence()
        assert ts.num_edges == 18
        assert ets.num_edges == 12
        return ts, ets

    def test_simple_ex(self):
        ts, right_ets = self.get_simple_ex()
        ets = ts.extend_edges()
        ets.tables.assert_equals(right_ets.tables)
        self.verify_extend_edges(ts)

    def test_internal_samples(self):
        # Now we should have the same but not extend 5 (where * is):
        #
        #    6         6      6         6
        #  +-+-+     +-+-+  +-+-+     +-+-+
        #  7   *     7   *  7   8     7   8
        #  |   |    ++-+ |  | +-++    |   |
        #  4   5    4  | *  4 |  5    4   5
        # +++ +++  +++ | |  | | +++  +++ +++
        # 0 1 2 3  0 1 2 3  0 1 2 3  0 1 2 3
        #
        # (p, c, l, r)
        edges = [
            (4, 0, 0, 10),
            (4, 1, 0, 5),
            (4, 1, 7, 10),
            (5, 2, 0, 2),
            (5, 2, 5, 10),
            (5, 3, 0, 2),
            (5, 3, 5, 10),
            (7, 2, 2, 5),
            (7, 4, 0, 10),
            (8, 1, 5, 7),
            (8, 5, 5, 10),
            (6, 3, 2, 5),
            (6, 5, 0, 2),
            (6, 7, 0, 10),
            (6, 8, 5, 10),
        ]
        ts, _ = self.get_simple_ex(samples=[0, 1, 2, 3, 5])
        tables = ts.dump_tables()
        tables.edges.clear()
        for p, c, l, r in edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ets = ts.extend_edges()
        ets.tables.assert_equals(tables)
        # validation doesn't work with internal, incomplete samples
        # (and it would be a pain to make it work)
        # self.verify_extend_edges(ts)

    def test_iterative_example(self):
        # Here is the full tree; extend edges should be able to
        # recover all unary nodes after simplification:
        #
        #       9         9         9          9
        #     +-+-+    +--+--+  +---+---+  +-+-+--+
        #     8   |    8     |  8   |   |  8 | |  |
        #     |   |  +-+-+   |  |   |   |  | | |  |
        #     7   |  |   7   |  |   7   |  | | |  7
        #   +-+-+ |  | +-++  |  | +-++  |  | | |  |
        #   6   | |  | |  6  |  | |  6  |  | | |  6
        # +-++  | |  | |  |  |  | |  |  |  | | |  |
        # 1  0  2 3  1 2  0  3  1 2  0  3  1 2 3  0
        #   +++          +++        +++          +++
        #   4 5          4 5        4 5          4 5
        #
        samples = [0, 1, 2, 3, 4, 5]
        node_times = [1, 1, 1, 1, 0, 0, 2, 3, 4, 5]
        # (p, c, l, r)
        edges = [
            (0, 4, 0, 10),
            (0, 5, 0, 10),
            (6, 0, 0, 10),
            (6, 1, 0, 3),
            (7, 2, 0, 7),
            (7, 6, 0, 10),
            (8, 1, 3, 10),
            (8, 7, 0, 5),
            (9, 2, 7, 10),
            (9, 3, 0, 10),
            (9, 7, 5, 10),
            (9, 8, 0, 10),
        ]
        tables = tskit.TableCollection(sequence_length=10)
        for n, t in enumerate(node_times):
            flags = tskit.NODE_IS_SAMPLE if n in samples else 0
            tables.nodes.add_row(time=t, flags=flags)
        for p, c, l, r in edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ts = tables.tree_sequence()
        sts = ts.simplify()
        assert ts.num_edges == 12
        assert sts.num_edges == 16
        tables.assert_equals(sts.extend_edges().tables, ignore_provenance=True)

    def test_very_simple(self):
        samples = [0]
        node_times = [0, 1, 2, 3]
        # (p, c, l, r)
        edges = [
            (1, 0, 0, 1),
            (2, 0, 1, 2),
            (2, 1, 0, 1),
            (3, 0, 2, 3),
            (3, 2, 0, 2),
        ]
        correct_edges = [
            (1, 0, 0, 3),
            (2, 1, 0, 3),
            (3, 2, 0, 3),
        ]
        tables = tskit.TableCollection(sequence_length=3)
        for n, t in enumerate(node_times):
            flags = tskit.NODE_IS_SAMPLE if n in samples else 0
            tables.nodes.add_row(time=t, flags=flags)
        for p, c, l, r in edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ts = tables.tree_sequence()
        ets = ts.extend_edges()
        for _, t, et in ts.coiterate(ets):
            print("----")
            print(t.draw(format="ascii"))
            print(et.draw(format="ascii"))
        etables = ets.tables
        correct_tables = etables.copy()
        etables.edges.clear()
        for p, c, l, r in correct_edges:
            etables.edges.add_row(parent=p, child=c, left=l, right=r)
        etables.assert_equals(correct_tables, ignore_provenance=True)

    def test_wright_fisher(self):
        tables = wf.wf_sim(N=5, ngens=20, num_loci=100, deep_history=False, seed=3)
        tables.sort()
        tables.simplify()
        ts = msprime.sim_mutations(tables.tree_sequence(), rate=0.01, random_seed=888)
        self.verify_extend_edges(ts, max_iter=1)
        self.verify_extend_edges(ts)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(N=6, ngens=22, num_loci=100, deep_history=False, seed=4)
        tables.sort()
        ts = msprime.sim_mutations(tables.tree_sequence(), rate=0.01, random_seed=888)
        self.verify_extend_edges(ts, max_iter=1)
        self.verify_extend_edges(ts)

    def test_wright_fisher_with_history(self):
        tables = wf.wf_sim(N=8, ngens=15, num_loci=100, deep_history=True, seed=5)
        tables.sort()
        tables.simplify()
        ts = msprime.sim_mutations(tables.tree_sequence(), rate=0.01, random_seed=888)
        self.verify_extend_edges(ts, max_iter=1)
        self.verify_extend_edges(ts)

    # This one fails sometimes but just because our verification can't handle
    # figuring out what exactly should be the right answer in complex cases.
    #
    # def test_bigger_wright_fisher(self):
    #     tables = wf.wf_sim(N=50, ngens=15, deep_history=True, seed=6)
    #     tables.sort()
    #     tables.simplify()
    #     ts = tables.tree_sequence()
    #     self.verify_extend_edges(ts, max_iter=1)
    #     self.verify_extend_edges(ts, max_iter=200)


class TestExamples:
    """
    Compare the ts method with local implementation.
    """

    def check(self, ts):
        if np.any(tskit.is_unknown_time(ts.mutations_time)):
            tables = ts.dump_tables()
            tables.compute_mutation_times()
            ts = tables.tree_sequence()
        py_ts = extend_edges(ts)
        lib_ts = ts.extend_edges()
        lib_ts.tables.assert_equals(py_ts.tables)
        assert np.all(ts.genotype_matrix() == lib_ts.genotype_matrix())
        sts = ts.simplify()
        lib_sts = lib_ts.simplify()
        lib_sts.tables.assert_equals(sts.tables, ignore_provenance=True)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_suite_examples_defaults(self, ts):
        if ts.num_migrations == 0:
            self.check(ts)
        else:
            with pytest.raises(
                _tskit.LibraryError, match="TSK_ERR_MIGRATIONS_NOT_SUPPORTED"
            ):
                _ = ts.extend_edges()

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_all_trees_ts(self, n):
        ts = tsutil.all_trees_ts(n)
        self.check(ts)
