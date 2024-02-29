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
Python implementation of the low-level supporting code for forward simulations.
"""
import itertools
import random

import numpy as np
import pytest

import tskit
from tests import simplify


class BirthBuffer:
    def __init__(self):
        self.edges = {}
        self.parents = []

    def add_edge(self, left, right, parent, child):
        if parent not in self.edges:
            self.parents.append(parent)
            self.edges[parent] = []
        self.edges[parent].append((child, left, right))

    def clear(self):
        self.edges = {}
        self.parents = []

    def __str__(self):
        s = ""
        for parent in self.parents:
            for child, left, right in self.edges[parent]:
                s += f"{parent}\t{child}\t{left:0.3f}\t{right:0.3f}\n"
        return s


def add_younger_edges_to_simplifier(simplifier, t, tables, edge_offset):
    parent_edges = []
    while (
        edge_offset < len(tables.edges)
        and tables.nodes.time[tables.edges.parent[edge_offset]] <= t
    ):
        print("edge offset = ", edge_offset)
        if len(parent_edges) == 0:
            last_parent = tables.edges.parent[edge_offset]
        else:
            last_parent = parent_edges[-1].parent
        if last_parent == tables.edges.parent[edge_offset]:
            parent_edges.append(tables.edges[edge_offset])
        else:
            print(
                "Flush ", tables.nodes.time[parent_edges[-1].parent], len(parent_edges)
            )
            simplifier.process_parent_edges(parent_edges)
            parent_edges = []
        edge_offset += 1
    if len(parent_edges) > 0:
        print("Flush ", tables.nodes.time[parent_edges[-1].parent], len(parent_edges))
        simplifier.process_parent_edges(parent_edges)
    return edge_offset


def simplify_with_births(tables, births, alive, verbose):
    total_edges = len(tables.edges)
    for edges in births.edges.values():
        total_edges += len(edges)
    if verbose > 0:
        print("Simplify with births")
        # print(births)
        print("total_input edges = ", total_edges)
        print("alive = ", alive)
        print("\ttable edges:", len(tables.edges))
        print("\ttable nodes:", len(tables.nodes))

    simplifier = simplify.Simplifier(tables.tree_sequence(), alive)
    nodes_time = tables.nodes.time
    # This should be almost sorted, because
    parent_time = nodes_time[births.parents]
    index = np.argsort(parent_time)
    print(index)
    offset = 0
    for parent in np.array(births.parents)[index]:
        offset = add_younger_edges_to_simplifier(
            simplifier, nodes_time[parent], tables, offset
        )
        edges = [
            tskit.Edge(left, right, parent, child)
            for child, left, right in sorted(births.edges[parent])
        ]
        # print("Adding parent from time", nodes_time[parent], len(edges))
        # print("edges = ", edges)
        simplifier.process_parent_edges(edges)
        # simplifier.print_state()

    # FIXME should probably reuse the add_younger_edges_to_simplifier function
    # for this - doesn't quite seem to work though
    for _, edges in itertools.groupby(tables.edges[offset:], lambda e: e.parent):
        edges = list(edges)
        simplifier.process_parent_edges(edges)

    simplifier.check_state()
    assert simplifier.parent_edges_processed == total_edges
    # if simplifier.parent_edges_processed != total_edges:
    #     print("HERE!!!!", total_edges)
    simplifier.finalise()

    tables.nodes.replace_with(simplifier.tables.nodes)
    tables.edges.replace_with(simplifier.tables.edges)

    # This is needed because we call .tree_sequence here and later.
    # Can be removed is we change the Simplifier to take a set of
    # tables which it modifies, like the C version.
    tables.drop_index()
    # Just to check
    tables.tree_sequence()

    births.clear()
    # Add back all the edges with an alive parent to the buffer, so that
    # we store them contiguously
    keep = np.ones(len(tables.edges), dtype=bool)
    for u in alive:
        u = simplifier.node_id_map[u]
        for e in np.where(tables.edges.parent == u)[0]:
            keep[e] = False
            edge = tables.edges[e]
            # print(edge)
            births.add_edge(edge.left, edge.right, edge.parent, edge.child)

    if verbose > 0:
        print("Done")
        print(births)
        print("\ttable edges:", len(tables.edges))
        print("\ttable nodes:", len(tables.nodes))


def simplify_with_births_easy(tables, births, alive, verbose):
    for parent, edges in births.edges.items():
        for child, left, right in edges:
            tables.edges.add_row(left, right, parent, child)
    tables.sort()
    tables.simplify(alive)
    births.clear()

    # print(tables.nodes.time[tables.edges.parent])


def wright_fisher(
    N, *, death_proba=1, L=1, T=10, simplify_interval=1, seed=42, verbose=0
):
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    alive = [tables.nodes.add_row(time=T) for _ in range(N)]
    births = BirthBuffer()

    t = T
    while t > 0:
        t -= 1
        next_alive = list(alive)
        for j in range(N):
            if rng.random() < death_proba:
                # alive[j] is dead - replace it.
                u = tables.nodes.add_row(time=t)
                next_alive[j] = u
                a = rng.randint(0, N - 1)
                b = rng.randint(0, N - 1)
                x = rng.uniform(0, L)
                # TODO Possibly more natural do this like
                # births.add(u, parents=[a, b], breaks=[0, x, L])
                births.add_edge(0, x, alive[a], u)
                births.add_edge(x, L, alive[b], u)
        alive = next_alive
        if t % simplify_interval == 0 or t == 0:
            simplify_with_births(tables, births, alive, verbose=verbose)
            # simplify_with_births_easy(tables, births, alive, verbose=verbose)
            alive = list(range(N))
    # print(tables.tree_sequence())
    return tables.tree_sequence()


class TestSimulationBasics:
    """
    Check that the basic simulation algorithm roughly works, so we're not building
    on sand.
    """

    @pytest.mark.parametrize("N", [1, 10, 100])
    def test_pop_size(self, N):
        ts = wright_fisher(N, simplify_interval=100)
        assert ts.num_samples == N

    @pytest.mark.parametrize("T", [1, 10, 100])
    def test_time(self, T):
        N = 10
        ts = wright_fisher(N=N, T=T, simplify_interval=1000)
        assert np.all(ts.nodes_time[ts.samples()] == 0)
        # Can't really assert anything much stronger, not really trying to
        # do anything particularly rigorous here
        assert np.max(ts.nodes_time) > 0

    def test_death_proba_0(self):
        N = 10
        T = 5
        ts = wright_fisher(N=N, T=T, death_proba=0, simplify_interval=1000)
        assert ts.num_nodes == N

    @pytest.mark.parametrize("seed", [1, 5, 1234])
    def test_seed_identical(self, seed):
        N = 10
        T = 5
        ts1 = wright_fisher(N=N, T=T, simplify_interval=1000, seed=seed)
        ts2 = wright_fisher(N=N, T=T, simplify_interval=1000, seed=seed)
        ts1.tables.assert_equals(ts2.tables, ignore_provenance=True)
        ts3 = wright_fisher(N=N, T=T, simplify_interval=1000, seed=seed - 1)
        assert not ts3.tables.equals(ts2.tables, ignore_provenance=True)

    def test_full_simulation(self):
        ts = wright_fisher(N=5, T=500, death_proba=0.9, simplify_interval=1000)
        for tree in ts.trees():
            assert tree.num_roots == 1


class TestSimplifyIntervals:
    @pytest.mark.parametrize("interval", [1, 10, 33, 100])
    def test_non_overlapping_generations(self, interval):
        N = 10
        ts = wright_fisher(N, T=100, death_proba=1, simplify_interval=interval)
        assert ts.num_samples == N

    @pytest.mark.parametrize("interval", [1, 10, 33, 100])
    @pytest.mark.parametrize("death_proba", [0.33, 0.5, 0.9])
    def test_overlapping_generations(self, interval, death_proba):
        N = 4
        ts = wright_fisher(
            N, T=20, death_proba=death_proba, simplify_interval=interval, verbose=1
        )
        assert ts.num_samples == N
        print()
        print(ts.draw_text())
