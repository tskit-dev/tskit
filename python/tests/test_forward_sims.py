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
import collections
import random

import numpy as np
import pytest

import tskit


def simplify_with_buffer(tables, parent_buffer, samples, verbose):
    # Pretend this was done efficiently internally without any sorting
    # by creating a simplifier object and adding the ancstry for the
    # new parents appropriately before flushing through the rest of the
    # edges.
    for parent, edges in parent_buffer.items():
        for left, right, child in edges:
            tables.edges.add_row(left, right, parent, child)
    tables.sort()
    tables.simplify(samples)
    # We've exhausted the parent buffer, so clear it out. In reality we'd
    # do this more carefully, like KT does in the post_simplify step.
    parent_buffer.clear()


def wright_fisher(
    N, *, death_proba=1, L=1, T=10, simplify_interval=1, seed=42, verbose=0
):
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    alive = [tables.nodes.add_row(time=T) for _ in range(N)]
    parent_buffer = collections.defaultdict(list)

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
                parent_buffer[alive[a]].append((0, x, u))
                parent_buffer[alive[b]].append((x, L, u))
        alive = next_alive
        if t % simplify_interval == 0 or t == 0:
            simplify_with_buffer(tables, parent_buffer, alive, verbose=verbose)
            alive = list(range(N))
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
