# import msprime
import numpy as np
import pytest

import tests.test_wright_fisher as wf
import tskit
from tests import tsutil

# from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


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


"""
Here is our new `extend_paths` algorithm.
This handles some other tricker cases we want the
`extend_edges` algorithm to succeed on.
This algorithm can also extend edges, however
its convergence is now not monotonically decreasing;
this makes convergence INCREDIBLY SLOW.
We think that we can combine `extend_paths` and `extend_edges`
in a piece-meal way to speed up this convergence, but this requires
further testing.
"""


def _build_degree(edges, nodes_edge):
    degree = np.zeros(nodes_edge.size)
    for n, e in enumerate(nodes_edge):
        if e == tskit.NULL:
            continue
        p, c = edges.parent[e], edges.child[e]
        assert n == c
        degree[p] += 1
        degree[c] += 1
    return degree


def _build_degree_from_parent(parent, num_nodes):
    degree = np.zeros(num_nodes, dtype="int")
    for c, p in enumerate(parent):
        degree[c] += 1
        degree[p] += 1
    return degree


def _build_parent(edges, edge_ids, num_nodes):
    parent = np.full(num_nodes, -1, dtype="int")
    for j in edge_ids:
        c = edges.child[j]
        p = edges.parent[j]
        parent[c] = p
    return parent


def _check_parent(parent, edges, edge_ids, num_nodes):
    check_parent = _build_parent(edges, edge_ids, num_nodes)
    assert np.all(check_parent == parent)


def _check_state(pos, edges, degree, nodes_edge, near_side, far_side, num_nodes):
    check_degree = np.zeros(num_nodes, dtype="int")
    check_nodes_edge = np.full(num_nodes, -1, dtype="int")
    assert len(near_side) == edges.num_rows
    assert len(far_side) == edges.num_rows
    for j, (e, l, r) in enumerate(zip(edges, near_side, far_side)):
        if (pos - l) * (r - pos) > 0:
            check_degree[e.child] += 1
            check_degree[e.parent] += 1
            check_nodes_edge[e.child] = j
    assert np.all(check_degree == degree)
    assert np.all(check_nodes_edge == nodes_edge)


def merge_edge_paths(
    edges_in, in_parent, out_parent, in_degree, out_degree, not_sample, ts, edges
):
    # We want a list of all longest edge paths with shared endpoints but
    # disjoint intermediate nodes from in_parent and out_parent.
    # Does NOT modify its arguments.
    paths = list()
    not_checked = np.full(ts.num_nodes, True, dtype=bool)
    for e_in in edges_in:
        c = edges[e_in].child
        if not_checked[c] is False:
            continue
        p_in = edges[e_in].parent
        p_out = out_parent[c]
        ipp = [c]
        opp = [c]
        while (
            p_out != tskit.NULL
            and in_degree[p_out] == 0
            and not_sample[p_out]
            and not_checked[p_out]
        ):
            opp.append(p_out)
            p_out = out_parent[p_out]
        if p_out != tskit.NULL and not_checked[p_out]:
            opp.append(p_out)
        while (
            p_in != tskit.NULL
            and in_degree[p_in] < 3
            and not_sample[p_in]
            and not_checked[p_in]
        ):
            ipp.append(p_in)
            p_in = in_parent[p_in]
        if p_in != tskit.NULL and not_checked[p_in]:
            ipp.append(p_in)
        # build the path list:
        if ipp[-1] == opp[-1]:
            path = list(set(ipp + opp))
            path.sort(key=lambda x: ts.nodes_time[x])
            not_checked[path[:-1]] = False
            paths.append(path)
        if ipp[-1] != opp[-1]:
            common_nodes, ipp_ind, opp_ind = np.intersect1d(
                ipp, opp, return_indices=True
            )
            common_nodes, ipp_ind, opp_ind = (
                list(common_nodes),
                list(ipp_ind),
                list(opp_ind),
            )
            if len(common_nodes) > 1:
                common_nodes.sort(key=lambda x: ts.nodes_time[x])
                ipp_ind.sort(key=lambda x: ts.nodes_time[ipp[x]])
                opp_ind.sort(key=lambda x: ts.nodes_time[opp[x]])
                ipp_last_ind = ipp_ind[-1]
                opp_last_ind = opp_ind[-1]
                ipp = ipp[: ipp_last_ind + 1]
                opp = opp[: opp_last_ind + 1]
                path = list(set(ipp + opp))
                path.sort(key=lambda x: ts.nodes_time[x])
                not_checked[path[:-1]] = False
                paths.append(path)
                # print('pathcheck', not_checked[path])
                # print('out path', opp)
                # print('in path', ipp)
                # print('path', path)
    return paths


def _add_edge(edges, near_side, far_side, forwards, parent, child, left, right):
    new_id = edges.add_row(parent=parent, child=child, left=left, right=right)
    if forwards:
        near_side.append(left)
        far_side.append(right)
    if not forwards:
        near_side.append(right)
        far_side.append(left)
    return new_id


def _extend_paths(ts, forwards=True):
    """
    Below we will iterate through the trees, either to the left or the right,
    keeping the following state consistent:
    - we are moving from a previous tree, out_tree, to new one, in_tree
    - here: the position that separates the out_tree from the in_tree
    - (here, there): the segment covered by in_tree
    - edges_out: edges to be removed from out_tree to get in_tree
    - out_parent: the forest induced by edges_out, a subset of out_tree
    - edges_in: edges to be added to out_tree to get in_tree
    - in_parent: the forest induced by edges_in, a subset of in_tree
    - in_degree: the negree of each node in in_tree
    - in_nodes_edge: for each node, the edge above it in in_tree
    - out_degree: the negree of each node in out_tree
    - out_nodes_edge: for each node, the edge above it in out_tree
    Except: each of edges_in and edges_out is of the form e, x, and the
    label x=True if the edge is postponed to the next segment: in other words,
    these edges don't count as beign part of edges_in and edges_out.
    """
    out_degree = np.full(ts.num_nodes, 0, dtype="int")
    in_degree = np.full(ts.num_nodes, 0, dtype="int")
    out_parent = np.full(ts.num_nodes, -1, dtype="int")
    in_parent = np.full(ts.num_nodes, -1, dtype="int")
    not_sample = [not n.is_sample() for n in ts.nodes()]
    edges = ts.tables.edges.copy()
    in_nodes_edge = np.full(ts.num_nodes, -1, dtype="int")
    out_nodes_edge = np.full(ts.num_nodes, -1, dtype="int")
    new_left = edges.left.copy()
    new_right = edges.right.copy()
    if forwards:
        direction = 1
        # in C we can just modify these in place, but in
        # python they are (silently) immutable
        near_side = list(new_left)
        far_side = list(new_right)
    else:
        direction = -1
        near_side = list(new_right)
        far_side = list(new_left)
    edges_out = []
    edges_in = []
    next_edges_out = []
    next_edges_in = []

    tree_pos = tsutil.TreePosition(ts)
    if forwards:
        valid = tree_pos.next()
    else:
        valid = tree_pos.prev()
    while valid:
        # print(f'--------{forwards}----------')
        left, right = tree_pos.interval
        # print('-----------',left, right,'----------')
        there = right if forwards else left
        here = left if forwards else right
        # Clear out non-extended or postponed edges:
        # Note: maintaining out_parent is a bit tricky, because
        # if an edge from p->c has been extended, entirely replacing
        # another edge from p'->c, then both edges may be in edges_out,
        # and we only want to include the *first* one.
        tmp = []
        for e in edges_out:
            out_parent[edges.child[e]] = -1
            out_nodes_edge[edges.child[e]] = tskit.NULL
        for e in next_edges_out:
            out_nodes_edge[edges.child[e]] = e
            far_side[e] = here
            if near_side[e] != here:
                tmp.append(e)
        edges_out = tmp
        tmp = []
        for e in edges_in:
            in_parent[edges.child[e]] = -1
            out_nodes_edge[edges.child[e]] = e
        for e in next_edges_in:
            out_nodes_edge[edges.child[e]] = tskit.NULL
            near_side[e] = here
            if far_side[e] != here:
                tmp.append(e)
        edges_in = tmp

        for e in edges_out:
            out_parent[edges.child[e]] = edges.parent[e]
            in_nodes_edge[edges.child[e]] = tskit.NULL

        for j in range(tree_pos.out_range.start, tree_pos.out_range.stop, direction):
            e = tree_pos.out_range.order[j]
            if out_parent[edges.child[e]] == -1:
                edges_out.append(e)
                out_parent[edges.child[e]] = edges.parent[e]
                in_nodes_edge[edges.child[e]] = tskit.NULL

        for j in range(tree_pos.in_range.start, tree_pos.in_range.stop, direction):
            e = tree_pos.in_range.order[j]
            edges_in.append(e)

        for e in edges_in:
            in_parent[edges.child[e]] = edges.parent[e]
            in_nodes_edge[edges.child[e]] = e

        in_degree = _build_degree(edges, in_nodes_edge)
        # TODO: do we use out_nodes_edge for anything except building out_degree?
        # If not, we can just update out_degree directly above... EXCEPT
        # for duplicated edges appended to edges_in.
        out_degree = _build_degree(edges, out_nodes_edge)
        _check_state(
            (here + there) / 2,
            edges,
            in_degree,
            in_nodes_edge,
            near_side,
            far_side,
            ts.num_nodes,
        )
        _check_parent(in_parent, edges, edges_in, ts.num_nodes)
        _check_parent(out_parent, edges, edges_out, ts.num_nodes)
        edge_paths = merge_edge_paths(
            edges_in,
            in_parent,
            out_parent,
            in_degree,
            out_degree,
            not_sample,
            ts,
            edges,
        )
        next_edges_out = []
        next_edges_in = []
        for path in edge_paths:
            # For each node in the path
            # Consider edge (j+1, j)
            for j in range(len(path) - 1):
                child = path[j]
                new_parent = path[j + 1]
                old_edge = in_nodes_edge[child]
                if old_edge != tskit.NULL:
                    old_parent = edges[old_edge].parent
                if old_edge == tskit.NULL:
                    old_parent = tskit.NULL
                # Do nothing if (j+1,j) exists in both trees
                if new_parent == old_parent:
                    # this is an edge already in the tree; do nothing
                    continue
                # If (j+1, j) not in previous tree
                # Determine if we should extend edges
                # or create a new edge
                if new_parent != old_parent:
                    # check if our new edge is in edges_out
                    # hence it should be extended
                    ### found_it = (out_parent[child] == new_parent)
                    # find the edge
                    for e_out in edges_out:
                        found_it = False
                        if (
                            edges[e_out].child == child
                            and edges[e_out].parent == new_parent
                        ):
                            found_it = True
                            break
                    # Extend edge_out into current tree;
                    # since we are not adding a new edge we know that if there is
                    # already an edge above this node, it must have been in edges in,
                    # so needs to be postponed coming in until the next tree
                    if found_it:
                        ### e_out = out_nodes_edge[child]
                        assert edges.child[e_out] == child
                        assert edges.parent[e_out] == new_parent
                        for eo in edges_out:
                            if e_out == eo:
                                break
                        assert e_out == eo
                        ## ex_out[1] = True
                        next_edges_out.append(e_out)
                        if old_edge != tskit.NULL:
                            next_edges_in.append(old_edge)
                        # if old_edge == tskit.NULL:
                        #     in_degree[child] += 2
                        in_nodes_edge[child] = e_out

                    # if edge is not in edges_out
                    # it is new and should be added to the edge table
                    if not found_it:
                        new_id = _add_edge(
                            edges,
                            near_side,
                            far_side,
                            forwards,
                            new_parent,
                            child,
                            left,
                            right,
                        )
                        in_nodes_edge[child] = new_id
                        next_edges_out.append(new_id)
                        if old_edge != tskit.NULL:
                            next_edges_in.append(old_edge)
                            # in_degree[child] -= 1
                            # in_degree[old_parent] -= 1
        # end of loop, next tree
        if forwards:
            valid = tree_pos.next()
        else:
            valid = tree_pos.prev()

    # clean-up from last interval
    for e in next_edges_out:
        far_side[e] = there
    for e in next_edges_in:
        near_side[e] = there

    if forwards:
        new_left = np.array(near_side)
        new_right = np.array(far_side)
    else:
        new_right = np.array(near_side)
        new_left = np.array(far_side)
    keep = np.full(edges.num_rows, True, dtype=bool)
    for j in range(edges.num_rows):
        left = new_left[j]
        right = new_right[j]
        if left < right:
            edges[j] = edges[j].replace(left=left, right=right)
        else:
            keep[j] = False
    edges.keep_rows(keep)
    return edges


def extend_paths(ts, max_iter=10):
    tables = ts.dump_tables()
    mutations = tables.mutations.copy()
    tables.mutations.clear()

    last_num_edges = ts.num_edges
    for _ in range(max_iter):
        for forwards in [True, False]:
            edges = _extend_paths(ts, forwards=forwards)
            tables.edges.replace_with(edges)
            tables.sort()
            tables.build_index()
            ts = tables.tree_sequence()
        if ts.num_edges == last_num_edges:
            break
        else:
            last_num_edges = ts.num_edges

    tables = ts.dump_tables()
    mutations = _slide_mutation_nodes_up(ts, mutations)
    tables.mutations.replace_with(mutations)
    tables.edges.squash()
    tables.sort()
    ts = tables.tree_sequence()

    return ts


def _path_pairs(tree):
    for c in tree.postorder():
        p = tree.parent(c)
        while p != tskit.NULL:
            yield (c, p)
            p = tree.parent(p)


def _path_up(c, p, tree, include_parent=False):
    # path from c up to p in tree, not including c or p
    c = tree.parent(c)
    while c != p and c != tskit.NULL:
        yield c
        c = tree.parent(c)
    assert c == p
    if include_parent:
        yield p


def _path_up_pairs(c, p, tree, others):
    # others should be a list of nodes
    otherdict = {tree.time(n): n for n in others}
    ot = min(otherdict)
    for n in _path_up(c, p, tree, include_parent=True):
        nt = tree.time(n)
        while ot < nt:
            on = otherdict.pop(ot)
            yield c, on
            c = on
            if len(otherdict) > 0:
                ot = min(otherdict)
            else:
                ot = np.inf
        yield c, n
        c = n
    assert n == p
    assert len(otherdict) == 0


def _path_overlaps(c, p, tree1, tree2):
    for n in _path_up(c, p, tree1):
        if n in tree2.nodes():
            return True
    return False


def _paths_mergeable(c, p, tree1, tree2):
    # checks that the nodes between c and p in each tree
    # are not present in the other tree
    # and their sets of times are disjoint
    nodes1 = set(tree1.nodes())
    nodes2 = set(tree2.nodes())
    assert c in nodes1, f"child node {c} not in tree1"
    assert p in nodes1, f"parent node {p} not in tree1"
    assert c in nodes2, f"child node {c} not in tree2"
    assert p in nodes2, f"parent node {p} not in tree2"
    path1 = set(_path_up(c, p, tree1))
    path2 = set(_path_up(c, p, tree2))
    times1 = {tree1.time(n) for n in path1}
    times2 = {tree2.time(n) for n in path2}
    return (
        (not _path_overlaps(c, p, tree1, tree2))
        and (not _path_overlaps(c, p, tree2, tree1))
        and len(times1.intersection(times2)) == 0
    )


def assert_not_extendable(ts):
    right_tree = ts.first()
    for tree in ts.trees():
        if tree.index + 1 >= ts.num_trees:
            break
        right_tree.seek_index(tree.index + 1)
        for c, p in _path_pairs(tree):
            extendable = (
                p != tree.parent(c)
                and c in right_tree.nodes(p)
                and p in right_tree.nodes()
                and _paths_mergeable(c, p, tree, right_tree)
            )
            if extendable:
                print("------------>", c, p)
                print(tree.draw_text())
                print(right_tree.draw_text())
            assert not extendable


def _extend_nodes(ts, interval, extendable):
    tables = ts.dump_tables()
    tables.edges.clear()
    mutations = tables.mutations.copy()
    tables.mutations.clear()
    left, right = interval
    # print("=================")
    # print("extending", left, right)
    extend_above = {}  # gives the new child->parent mapping
    todo_edges = np.repeat(True, ts.num_edges)
    tree = ts.at(left)
    for c, p, others in extendable:
        print("c:", c, "p:", p, "others:", others)
        others_not_done_yet = set(others) - set(extend_above)
        if len(others_not_done_yet) > 0:
            for cn, pn in _path_up_pairs(c, p, tree, others_not_done_yet):
                if cn not in extend_above:
                    assert cn not in extend_above
                    extend_above[cn] = pn
    for c, p in extend_above.items():
        e = tree.edge(c)
        if e == tskit.NULL or ts.edge(e).parent != p:
            # print("adding", c, p)
            tables.edges.add_row(child=c, parent=p, left=left, right=right)
            if e != tskit.NULL:
                edge = ts.edge(e)
                # adjust endpoints on existing edge
                for el, er in [
                    (max(edge.left, right), edge.right),
                    (edge.left, min(edge.right, left)),
                ]:
                    if el < er:
                        # print("replacing", edge, el, er)
                        tables.edges.append(edge.replace(left=el, right=er))
                todo_edges[e] = False
    for todo, edge in zip(todo_edges, ts.edges()):
        if todo:
            # print("retaining", edge)
            tables.edges.append(edge)
    tables.sort()
    ts = tables.tree_sequence()
    mutations = _slide_mutation_nodes_up(ts, mutations)
    tables.mutations.replace_with(mutations)
    tables.sort()
    return tables.tree_sequence()


def _naive_pass(ts, direction):
    assert direction in (-1, +1)
    num_trees = ts.num_trees
    if direction == +1:
        indexes = range(0, num_trees - 1, 1)
    else:
        indexes = range(num_trees - 1, 0, -1)
    for tj in indexes:
        extendable = []
        this_tree = ts.at_index(tj)
        next_tree = ts.at_index(tj + direction)
        # print("-----------", this_tree.index)
        # print(this_tree.draw_text())
        # print(next_tree.draw_text())
        for c, p in _path_pairs(this_tree):
            if (
                p != this_tree.parent(c)
                and p in next_tree.nodes()
                and c in next_tree.nodes(p)
            ):
                # print(c, p, " and ", list(next_tree.nodes(p)))
                if _paths_mergeable(c, p, this_tree, next_tree):
                    extendable.append((c, p, list(_path_up(c, p, this_tree))))
        # print("extending to", extendable)
        ts = _extend_nodes(ts, next_tree.interval, extendable)
        assert num_trees == ts.num_trees
    return ts


def naive_extend_paths(ts, max_iter=20):
    for _ in range(max_iter):
        ets = _naive_pass(ts, +1)
        ets = _naive_pass(ets, -1)
        if ets == ts:
            break
        ts = ets
    return ts


class TestExtendThings:
    """
    Common utilities in the two classes below.
    """

    def verify_simplify_equality(self, ts, ets):
        assert np.all(ts.genotype_matrix() == ets.genotype_matrix())
        assert ts.num_nodes == ets.num_nodes
        assert ts.num_samples == ets.num_samples
        t = ts.simplify().tables
        et = ets.simplify().tables
        et.assert_equals(t, ignore_provenance=True)

    def naive_verify(self, ts):
        ets = naive_extend_paths(ts)
        for i, t, et in ts.coiterate(ets):
            print("---------------", i)
            print(t.draw_text())
            print(et.draw_text())
        self.verify_simplify_equality(ts, ets)


class TestExtendPaths(TestExtendThings):
    """
    Test the 'extend_paths' method.
    """

    def get_example1(self):
        # 15.00|         |   13    |         |
        #      |         |    |    |         |
        # 12.00|   10    |   10    |    10   |
        #      |  +-+-+  |  +-+-+  |   +-+-+ |
        # 10.00|  8   |  |  |   |  |   8   | |
        #      |  |   |  |  |   |  |  ++-+ | |
        # 8.00 |  |   |  | 11  12  |  |  | | |
        #      |  |   |  |  |   |  |  |  | | |
        # 6.00 |  |   |  |  7   |  |  |  | | |
        #      |  |   |  |  |   |  |  |  | | |
        # 4.00 |  6   9  |  |   |  |  |  | | |
        #      |  |   |  |  |   |  |  |  | | |
        # 1.00 |  4   5  |  4   5  |  4  | 5 |
        #      | +++ +++ | +++ +++ | +++ | | |
        # 0.00 | 0 1 2 3 | 0 1 2 3 | 0 1 2 3 |
        #      0         3         6         9
        node_times = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1,
            6: 4,
            7: 6,
            8: 10,
            9: 4,
            10: 12,
            11: 8,
            12: 8,
            13: 15,
        }
        # (p,c,l,r)
        edges = [
            (4, 0, 0, 9),
            (4, 1, 0, 9),
            (5, 2, 0, 6),
            (5, 3, 0, 9),
            (6, 4, 0, 3),
            (9, 5, 0, 3),
            (7, 4, 3, 6),
            (11, 7, 3, 6),
            (12, 5, 3, 6),
            (8, 2, 6, 9),
            (8, 4, 6, 9),
            (8, 6, 0, 3),
            (10, 5, 6, 9),
            (10, 8, 0, 3),
            (10, 8, 6, 9),
            (10, 9, 0, 3),
            (10, 11, 3, 6),
            (10, 12, 3, 6),
            (13, 10, 3, 6),
        ]
        extended_path_edges = [
            (4, 0, 0.0, 9.0),
            (4, 1, 0.0, 9.0),
            (5, 2, 0.0, 6.0),
            (5, 3, 0.0, 9.0),
            (6, 4, 0.0, 9.0),
            (9, 5, 0.0, 9.0),
            (7, 6, 0.0, 9.0),
            (11, 7, 0.0, 9.0),
            (12, 9, 0.0, 9.0),
            (8, 2, 6.0, 9.0),
            (8, 11, 0.0, 9.0),
            (10, 8, 0.0, 9.0),
            (10, 12, 0.0, 9.0),
            (13, 10, 3.0, 6.0),
        ]
        samples = list(np.arange(4))
        tables = tskit.TableCollection(sequence_length=9)
        for (
            n,
            t,
        ) in node_times.items():
            flags = tskit.NODE_IS_SAMPLE if n in samples else 0
            tables.nodes.add_row(time=t, flags=flags)
        for p, c, l, r in edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ts = tables.tree_sequence()
        tables.edges.clear()
        for p, c, l, r in extended_path_edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ets = tables.tree_sequence()
        assert ts.num_edges == 19
        assert ets.num_edges == 14
        return ts, ets

    def get_example2(self):
        # 12.00|                     |          21         |                     |
        #      |                     |      +----+-----+   |                     |
        # 11.00|            20       |      |          |   |            20       |
        #      |        +----+---+   |      |          |   |        +----+---+   |
        # 10.00|        |       19   |      |         19   |        |       19   |
        #      |        |       ++-+ |      |        +-+-+ |        |       ++-+ |
        # 9.00 |       18       |  | |     18        |   | |       18       |  | |
        #      |     +--+--+    |  | |   +--+--+     |   | |     +--+--+    |  | |
        # 8.00 |     |     |    |  | |   |     |     |   | |    17     |    |  | |
        #      |     |     |    |  | |   |     |     |   | |   +-+-+   |    |  | |
        # 7.00 |     |     |   16  | |   |     |    16   | |   |   |   |    |  | |
        #      |     |     |   +++ | |   |     |   +-++  | |   |   |   |    |  | |
        # 6.00 |    15     |   | | | |   |     |   |  |  | |   |   |   |    |  | |
        #      |   +-+-+   |   | | | |   |     |   |  |  | |   |   |   |    |  | |
        # 5.00 |   |   |  14   | | | |   |    14   |  |  | |   |   |  14    |  | |
        #      |   |   |  ++-+ | | | |   |    ++-+ |  |  | |   |   |  ++-+  |  | |
        # 4.00 |  13   |  |  | | | | |  13    |  | |  |  | |  13   |  |  |  |  | |
        #      |  ++-+ |  |  | | | | |  ++-+  |  | |  |  | |  ++-+ |  |  |  |  | |
        # 3.00 |  |  | |  |  | | | | |  |  |  |  | | 12  | |  |  | |  |  | 12  | |
        #      |  |  | |  |  | | | | |  |  |  |  | | +++ | |  |  | |  |  | +++ | |
        # 2.00 | 11  | |  |  | | | | | 11  |  |  | | | | | | 11  | |  |  | | | | |
        #      | +++ | |  |  | | | | | +++ |  |  | | | | | | +++ | |  |  | | | | |
        # 1.00 | | | | | 10  | | | | | | | | 10  | | | | | | | | | | 10  | | | | |
        #      | | | | | +++ | | | | | | | | +++ | | | | | | | | | | +++ | | | | |
        # 0.00 | 0 7 4 9 2 5 6 1 3 8 | 0 7 4 2 5 6 1 3 9 8 | 0 7 4 1 2 5 6 3 9 8 |
        #      0                     3                     6                     9
        node_times = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 1,
            11: 2,
            12: 3,
            13: 4,
            14: 5,
            15: 6,
            16: 7,
            17: 8,
            18: 9,
            19: 10,
            20: 11,
            21: 12,
        }
        # (p,c,l,r)
        edges = [
            (10, 2, 0, 9),
            (10, 5, 0, 9),
            (11, 0, 0, 9),
            (11, 7, 0, 9),
            (12, 3, 3, 9),
            (12, 9, 3, 9),
            (13, 4, 0, 9),
            (13, 11, 0, 9),
            (14, 6, 0, 9),
            (14, 10, 0, 9),
            (15, 9, 0, 3),
            (15, 13, 0, 3),
            (16, 1, 0, 6),
            (16, 3, 0, 3),
            (16, 12, 3, 6),
            (17, 1, 6, 9),
            (17, 13, 6, 9),
            (18, 13, 3, 6),
            (18, 14, 0, 9),
            (18, 15, 0, 3),
            (18, 17, 6, 9),
            (19, 8, 0, 9),
            (19, 12, 6, 9),
            (19, 16, 0, 6),
            (20, 18, 0, 3),
            (20, 18, 6, 9),
            (20, 19, 0, 3),
            (20, 19, 6, 9),
            (21, 18, 3, 6),
            (21, 19, 3, 6),
        ]
        extended_path_edges = [
            (10, 2, 0.0, 9.0),
            (10, 5, 0.0, 9.0),
            (11, 0, 0.0, 9.0),
            (11, 7, 0.0, 9.0),
            (12, 3, 0.0, 9.0),
            (12, 9, 3.0, 9.0),
            (13, 4, 0.0, 9.0),
            (13, 11, 0.0, 9.0),
            (14, 6, 0.0, 9.0),
            (14, 10, 0.0, 9.0),
            (15, 9, 0.0, 3.0),
            (15, 13, 0.0, 9.0),
            (16, 1, 0.0, 6.0),
            (16, 12, 0.0, 9.0),
            (17, 1, 6.0, 9.0),
            (17, 15, 0.0, 9.0),
            (18, 14, 0.0, 9.0),
            (18, 17, 0.0, 9.0),
            (19, 8, 0.0, 9.0),
            (19, 16, 0.0, 9.0),
            (20, 18, 0.0, 3.0),
            (20, 18, 6.0, 9.0),
            (20, 19, 0.0, 3.0),
            (20, 19, 6.0, 9.0),
            (21, 18, 3.0, 6.0),
            (21, 19, 3.0, 6.0),
        ]
        samples = list(np.arange(10))
        tables = tskit.TableCollection(sequence_length=9)
        for (
            n,
            t,
        ) in node_times.items():
            flags = tskit.NODE_IS_SAMPLE if n in samples else 0
            tables.nodes.add_row(time=t, flags=flags)
        for p, c, l, r in edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ts = tables.tree_sequence()
        tables.edges.clear()
        for p, c, l, r in extended_path_edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ets = tables.tree_sequence()
        assert ts.num_edges == 30
        assert ets.num_edges == 26
        return ts, ets

    def get_example3(self):
        # 12.00|         |         |         |         |         |         |         |         |         |
        #      |         |         |         |         |         |         |         |         |         |
        # 11.00|  12     |   12    |    12   |    12   |   12    |         |         |         |         |
        #      | +-+-+   |  +-+-+  |   +-+-+ |     |   |    |    |         |         |         |         |
        # 10.00| |   |   |  |   |  |   |   | |     |   |    |    |   11    |  11     |  11     |  11     |
        #      | |   |   |  |   |  |   |   | |     |   |    |    |    |    |   |     | +-+-+   | +-+-+   |
        # 9.00 | |  10   |  |  10  |  10   | |    10   |   10    |   10    |  10     | |  10   | |  10   |
        #      | |   |   |  |   |  |   |   | |   +-+-+ |  +-+-+  |  +-+-+  | +-+-+   | |   |   | |   |   |
        # 8.00 | |   |   |  |   |  |   |   | |   |   | |  |   |  |  |   |  | |   |   | |   |   | |   |   |
        #      | |   |   |  |   |  |   |   | |   |   | |  |   |  |  |   |  | |   |   | |   |   | |   |   |
        # 7.00 | 8   |   |  8   |  |   |   8 |   |   8 |  |   8  |  |   8  | |   8   | |   8   | |   8   |
        #      | |   |   |  |   |  |   |   | |   |   | |  |  +++ |  |  +++ | |  ++-+ | |  ++-+ | |   |   |
        # 6.00 | |   7   |  |   7  |   7   | |   7   | |  7  | | |  7  | | | 7  |  | | 7  |  | | 7   |   |
        #      | | +-+-+ |  |  +++ | +-+-+ | | +-+-+ | | +++ | | | +++ | | | |  |  | | |  |  | | |   |   |
        # 5.00 | | | | 6 |  |  | 6 | | | 6 | | | | 6 | | | | | 6 | | | | 6 | |  |  6 | |  |  6 | |   6   |
        #      | | | | | |  |  | | | | | | | | | | | | | | | | | | | | | | | |  |  | | |  |  | | |  ++-+ |
        # 4.00 | 5 | | | |  5  | | | | | | 5 | | | | 5 | | | 5 | | | | 5 | | |  5  | | |  5  | | |  |  | |
        #      | | | | | | +++ | | | | | | | | | | | | | | | | | | | | | | | |  |  | | |  |  | | |  |  | |
        # 3.00 | | | | | | | 4 | | | | | | 4 | | | | 4 | | | 4 | | | | 4 | | |  4  | | |  4  | | |  4  | |
        #      | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | +++ | | | +++ | | | +++ | |
        # 0.00 | 0 1 2 3 | 0 1 2 3 | 0 2 3 1 | 0 2 3 1 | 0 2 1 3 | 0 2 1 3 | 0 1 2 3 | 0 1 2 3 | 0 1 2 3 |
        #      0         1         2         3         4         5         6         7         8        10
        (x1, x2, x3, x4, x5, x6, x7, x8) = (1, 2, 3, 4, 5, 6, 7, 8)
        edges = [  # c, p, l, r
            (0, 5, 0, x2),
            (0, 7, x2, x6),
            (0, 10, x6, x8),
            (1, 7, 0, x1),
            (1, 4, x1, x8),
            (2, 4, x5, x8),
            (2, 7, 0, x5),
            (3, 6, 0, x8),
            (4, 5, x1, x7),
            (4, 6, x7, x8),
            (5, 8, 0, x7),
            (6, 7, 0, x4),
            (6, 8, x4, x8),
            (7, 9, 0, x6),
            (8, 9, x3, x8),
            (8, 11, 0, x3),
            (9, 10, x6, x8),
            (9, 11, 0, x3),
        ]
        tables = tskit.TableCollection(sequence_length=x8)
        for _ in range(4):
            n = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)

        while n < 11:
            n = tables.nodes.add_row(time=n)

        for c, p, l, r in edges:
            tables.edges.add_row(child=c, parent=p, left=l, right=r)

        tables.sort()
        ets = tables.tree_sequence()
        ts = ets.simplify()
        assert ts.num_nodes == ets.num_nodes
        return ts, ets

    def verify_extend_paths(self, ts, max_iter=10):
        ets = extend_paths(ts, max_iter=max_iter)
        self.verify_simplify_equality(ts, ets)

    def test_example1(self):
        ts, ets = self.get_example1()
        test_ets = extend_paths(ts)
        test_ets.tables.assert_equals(ets.tables, ignore_provenance=True)
        self.verify_extend_paths(ts)
        self.naive_verify(ts)

    def test_example2(self):
        ts, ets = self.get_example2()
        test_ets = extend_paths(ts)
        test_ets.tables.assert_equals(ets.tables, ignore_provenance=True)
        self.verify_extend_paths(ts)
        self.naive_verify(ts)

    @pytest.mark.skip("FIXME: too much un-inferrable stuff still")
    def test_example3(self):
        ts, ets = self.get_example3()
        test_ets = naive_extend_paths(ts)
        for (
            x,
            t,
            et,
        ) in ets.coiterate(test_ets):
            print("--------------", x)
            print(t.draw(format="ascii"))
            print(et.draw(format="ascii"))
        test_ets.tables.assert_equals(ets.tables, ignore_provenance=True)
        self.verify_extend_paths(ts)
        self.naive_verify(ts)

    # TODO: include seed 5; it has time conflicts
    @pytest.mark.parametrize("seed", [3, 4, 6])
    def test_wf(self, seed):
        tables = wf.wf_sim(N=6, ngens=9, num_loci=100, deep_history=False, seed=seed)
        tables.sort()
        ts = tables.tree_sequence().simplify()
        self.verify_extend_paths(ts)
        self.naive_verify(ts)
