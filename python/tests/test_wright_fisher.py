# MIT License
#
# Copyright (c) 2018-2021 Tskit Developers
# Copyright (C) 2017 University of Oxford
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
Test various functions using messy tables output by a forwards-time simulator.
"""
import itertools
import random

import msprime
import numpy as np
import numpy.testing as nt
import pytest

import tests as tests
import tests.tsutil as tsutil
import tskit


class WrightFisherSimulator:
    """
    SIMPLE simulation of `num_pops` bisexual, haploid Wright-Fisher populations
    of size `N` for `ngens` generations, in which each individual survives with
    probability `survival` and only those who die are replaced. If `num_pops` is
    greater than 1, the individual to be replaced has a chance `mig_rate` of
    being the offspring of nodes from a different and randomly chosen
    population. If `num_loci` is None, the chromosome is 1.0 Morgans long. If
    `num_loci` not None, a discrete recombination model is used where
    breakpoints are chosen uniformly from 1 to `num_loci` - 1. If
    `deep_history` is True, a history to coalescence of just one population of
    `self.N` samples is added at the beginning.
    """

    def __init__(
        self,
        N,
        survival=0.0,
        seed=None,
        deep_history=True,
        debug=False,
        initial_generation_samples=False,
        num_loci=None,
        num_pops=1,
        mig_rate=0.0,
        record_migrations=False,
        record_individuals=True,
    ):
        self.N = N
        self.num_pops = num_pops
        self.num_loci = num_loci
        self.survival = survival
        self.mig_rate = mig_rate
        self.record_migrations = record_migrations
        self.record_individuals = record_individuals
        self.deep_history = deep_history
        self.debug = debug
        self.initial_generation_samples = initial_generation_samples
        self.seed = seed
        self.rng = random.Random(seed)

    def random_breakpoint(self):
        if self.num_loci is None:
            return min(1.0, max(0.0, 2 * self.rng.random() - 0.5))
        else:
            return self.rng.randint(1, self.num_loci - 1)

    def run(self, ngens):
        L = 1
        if self.num_loci is not None:
            L = self.num_loci
        tables = tskit.TableCollection(sequence_length=L)
        for _ in range(self.num_pops):
            tables.populations.add_row()
        if self.deep_history:
            # initial population
            population_configurations = [
                msprime.PopulationConfiguration(sample_size=self.N)
            ]
            init_ts = msprime.simulate(
                population_configurations=population_configurations,
                recombination_rate=1.0,
                length=L,
                random_seed=self.seed,
            )
            init_tables = init_ts.dump_tables()
            flags = init_tables.nodes.flags
            if not self.initial_generation_samples:
                flags = np.zeros_like(init_tables.nodes.flags)
            tables.nodes.set_columns(time=init_tables.nodes.time + ngens, flags=flags)
            tables.edges.set_columns(
                left=init_tables.edges.left,
                right=init_tables.edges.right,
                parent=init_tables.edges.parent,
                child=init_tables.edges.child,
            )
        else:
            flags = 0
            if self.initial_generation_samples:
                flags = tskit.NODE_IS_SAMPLE
            for p in range(self.num_pops):
                for _ in range(self.N):
                    individual = -1
                    if self.record_individuals:
                        individual = tables.individuals.add_row(parents=[-1, -1])
                    tables.nodes.add_row(
                        flags=flags, time=ngens, population=p, individual=individual
                    )

        pops = [
            list(range(p * self.N, (p * self.N) + self.N)) for p in range(self.num_pops)
        ]
        pop_ids = list(range(self.num_pops))
        for t in range(ngens - 1, -1, -1):
            if self.debug:
                print("t:", t)
                print("pops:", pops)
            dead = [[self.rng.random() > self.survival for _ in pop] for pop in pops]
            # sample these first so that all parents are from the previous gen
            parent_pop = []
            new_parents = []
            for p in pop_ids:
                w = [
                    1 - self.mig_rate if i == p else self.mig_rate / (self.num_pops - 1)
                    for i in pop_ids
                ]
                parent_pop.append(self.rng.choices(pop_ids, w, k=sum(dead[p])))
                new_parents.append(
                    [
                        self.rng.choices(pops[parent_pop[p][k]], k=2)
                        for k in range(sum(dead[p]))
                    ]
                )

            if self.debug:
                for p in pop_ids:
                    print("Replacing", sum(dead[p]), "individuals from pop", p)
            for p in pop_ids:
                k = 0
                for j in range(self.N):
                    if dead[p][j]:
                        lparent, rparent = new_parents[p][k]
                        individual = -1
                        if self.record_individuals:
                            individual = tables.individuals.add_row(
                                parents=[
                                    tables.nodes[lparent].individual,
                                    tables.nodes[rparent].individual,
                                ]
                            )
                        offspring = tables.nodes.add_row(
                            time=t, population=p, individual=individual
                        )
                        if parent_pop[p][k] != p and self.record_migrations:
                            tables.migrations.add_row(
                                left=0.0,
                                right=L,
                                node=offspring,
                                source=parent_pop[p][k],
                                dest=p,
                                time=t,
                            )
                        k += 1
                        bp = self.random_breakpoint()
                        if self.debug:
                            print("--->", offspring, lparent, rparent, bp)
                        pops[p][j] = offspring
                        if bp > 0.0:
                            tables.edges.add_row(
                                left=0.0, right=bp, parent=lparent, child=offspring
                            )
                        if bp < L:
                            tables.edges.add_row(
                                left=bp, right=L, parent=rparent, child=offspring
                            )

        if self.debug:
            print("Done! Final pop:")
            print(pops)
        flags = tables.nodes.flags
        flattened = [n for pop in pops for n in pop]
        flags[flattened] = tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        tables.time_units = "generations"
        return tables


def wf_sim(
    N,
    ngens,
    survival=0.0,
    deep_history=True,
    debug=False,
    seed=None,
    initial_generation_samples=False,
    num_loci=None,
    num_pops=1,
    mig_rate=0.0,
    record_migrations=False,
    record_individuals=True,
):
    sim = WrightFisherSimulator(
        N,
        survival=survival,
        deep_history=deep_history,
        debug=debug,
        seed=seed,
        initial_generation_samples=initial_generation_samples,
        num_loci=num_loci,
        num_pops=num_pops,
        mig_rate=mig_rate,
        record_migrations=record_migrations,
        record_individuals=record_individuals,
    )
    return sim.run(ngens)


class TestSimulation:
    """
    Tests that the simulations produce the output we expect.
    """

    random_seed = 5678

    def test_one_gen_multipop_mig_no_deep(self):
        tables = wf_sim(
            N=5,
            ngens=1,
            num_pops=4,
            mig_rate=1.0,
            deep_history=False,
            seed=self.random_seed,
            record_migrations=True,
        )
        assert tables.time_units == "generations"
        assert tables.nodes.num_rows == 5 * 4 * (1 + 1)
        assert tables.edges.num_rows > 0
        assert tables.migrations.num_rows == 5 * 4
        assert tables.individuals.num_rows == tables.nodes.num_rows

    def test_multipop_mig_deep(self):
        N = 10
        ngens = 20
        num_pops = 3
        tables = wf_sim(
            N=N,
            ngens=ngens,
            num_pops=num_pops,
            mig_rate=1.0,
            seed=self.random_seed,
            record_migrations=True,
        )
        assert tables.nodes.num_rows > (num_pops * N * ngens) + N
        assert tables.edges.num_rows > 0
        assert tables.sites.num_rows == 0
        assert tables.mutations.num_rows == 0
        assert tables.migrations.num_rows >= N * num_pops * ngens
        assert tables.populations.num_rows == num_pops
        assert tables.individuals.num_rows >= num_pops * N * ngens

        # sort does not support mig
        tables.migrations.clear()
        # making sure trees are valid
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        sample_pops = tables.nodes.population[ts.samples()]
        assert np.unique(sample_pops).size == num_pops

    def test_multipop_mig_no_deep(self):
        N = 5
        ngens = 5
        num_pops = 2
        tables = wf_sim(
            N=N,
            ngens=ngens,
            num_pops=num_pops,
            mig_rate=1.0,
            deep_history=False,
            seed=self.random_seed,
            record_migrations=True,
        )
        assert tables.nodes.num_rows == num_pops * N * (ngens + 1)
        assert tables.edges.num_rows > 0
        assert tables.sites.num_rows == 0
        assert tables.mutations.num_rows == 0
        assert tables.migrations.num_rows == N * num_pops * ngens
        assert tables.populations.num_rows == num_pops
        assert tables.individuals.num_rows == tables.nodes.num_rows
        # FIXME this is no longer needed.
        # sort does not support mig
        tables.migrations.clear()
        # making sure trees are valid
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        sample_pops = tables.nodes.population[ts.samples()]
        assert np.unique(sample_pops).size == num_pops

    def test_non_overlapping_generations(self):
        tables = wf_sim(N=10, ngens=10, survival=0.0, seed=self.random_seed)
        assert tables.nodes.num_rows > 0
        assert tables.edges.num_rows > 0
        assert tables.sites.num_rows == 0
        assert tables.mutations.num_rows == 0
        assert tables.migrations.num_rows == 0
        assert tables.individuals.num_rows > 0
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        # All trees should have exactly one root and all internal nodes should
        # have arity > 1
        for tree in ts.trees():
            assert tree.num_roots == 1
            leaves = set(tree.leaves(tree.root))
            assert leaves == set(ts.samples())
            for u in tree.nodes():
                if tree.is_internal(u):
                    assert len(tree.children(u)) > 1

    def test_overlapping_generations(self):
        tables = wf_sim(N=30, ngens=10, survival=0.85, seed=self.random_seed)
        assert tables.nodes.num_rows > 0
        assert tables.edges.num_rows > 0
        assert tables.sites.num_rows == 0
        assert tables.mutations.num_rows == 0
        assert tables.migrations.num_rows == 0
        assert tables.individuals.num_rows > 0
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        for tree in ts.trees():
            assert tree.num_roots == 1

    def test_one_generation_no_deep_history(self):
        N = 20
        tables = wf_sim(N=N, ngens=1, deep_history=False, seed=self.random_seed)
        assert tables.nodes.num_rows == 2 * N
        assert tables.edges.num_rows > 0
        assert tables.sites.num_rows == 0
        assert tables.mutations.num_rows == 0
        assert tables.migrations.num_rows == 0
        assert tables.individuals.num_rows > 0
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        assert tables.nodes.num_rows > 0
        assert tables.edges.num_rows > 0
        ts = tables.tree_sequence()
        for tree in ts.trees():
            all_samples = set()
            for root in tree.roots:
                root_samples = set(tree.samples(root))
                assert len(root_samples & all_samples) == 0
                all_samples |= root_samples
            assert all_samples == set(ts.samples())

    def test_many_generations_no_deep_history(self):
        N = 10
        ngens = 100
        tables = wf_sim(N=N, ngens=ngens, deep_history=False, seed=self.random_seed)
        assert tables.nodes.num_rows == N * (ngens + 1)
        assert tables.edges.num_rows > 0
        assert tables.sites.num_rows == 0
        assert tables.mutations.num_rows == 0
        assert tables.migrations.num_rows == 0
        assert tables.individuals.num_rows > 0
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        assert tables.nodes.num_rows > 0
        assert tables.edges.num_rows > 0
        ts = tables.tree_sequence()
        # We are assuming that everything has coalesced and we have single-root trees
        for tree in ts.trees():
            assert tree.num_roots == 1

    def test_with_mutations(self):
        N = 10
        ngens = 100
        tables = wf_sim(N=N, ngens=ngens, deep_history=False, seed=self.random_seed)
        tables.sort()
        ts = tables.tree_sequence()
        ts = tsutil.jukes_cantor(ts, 10, 0.1, seed=self.random_seed)
        tables = ts.dump_tables()
        assert tables.sites.num_rows > 0
        assert tables.mutations.num_rows > 0
        samples = np.where(tables.nodes.flags == tskit.NODE_IS_SAMPLE)[0].astype(
            np.int32
        )
        tables.sort()
        tables.simplify(samples)
        assert tables.nodes.num_rows > 0
        assert tables.edges.num_rows > 0
        assert tables.nodes.num_rows > 0
        assert tables.edges.num_rows > 0
        assert tables.sites.num_rows > 0
        assert tables.mutations.num_rows > 0
        ts = tables.tree_sequence()
        assert ts.sample_size == N
        for hap in ts.haplotypes():
            assert len(hap) == ts.num_sites

    def test_with_recurrent_mutations(self):
        # actually with only ONE site, at 0.0
        N = 10
        ngens = 100
        tables = wf_sim(N=N, ngens=ngens, deep_history=False, seed=self.random_seed)
        tables.sort()
        ts = tables.tree_sequence()
        ts = tsutil.jukes_cantor(ts, 1, 10, seed=self.random_seed)
        tables = ts.dump_tables()
        assert tables.sites.num_rows == 1
        assert tables.mutations.num_rows > 0
        # before simplify
        for h in ts.haplotypes():
            assert len(h) == 1
        # after simplify
        tables.sort()
        tables.simplify()
        assert tables.nodes.num_rows > 0
        assert tables.edges.num_rows > 0
        assert tables.sites.num_rows == 1
        assert tables.mutations.num_rows > 0
        ts = tables.tree_sequence()
        assert ts.sample_size == N
        for hap in ts.haplotypes():
            assert len(hap) == ts.num_sites

    def test_record_individuals_initial_state(self):
        N = 10
        tables = wf_sim(N=N, ngens=0, seed=12345, deep_history=False)
        tables.sort()
        assert len(tables.individuals) == N
        assert len(tables.nodes) == N
        for individual in list(tables.individuals)[:N]:
            assert list(individual.parents) == [-1, -1]
        for j, node in enumerate(tables.nodes):
            assert node.individual == j

    def test_record_individuals(self):
        N = 10
        tables = wf_sim(N=N, ngens=10, seed=12345, deep_history=False)
        assert len(tables.individuals) == len(tables.nodes)
        for node_id, individual in enumerate(tables.nodes.individual):
            assert node_id == individual
        tables.sort()
        ts = tables.tree_sequence()
        for tree in ts.trees():
            for u in tree.nodes():
                individual = ts.individual(ts.node(u).individual)
                parent_node = tree.parent(u)
                if parent_node != tskit.NULL:
                    parent_individual = ts.individual(ts.node(parent_node).individual)
                    assert parent_individual.id in individual.parents


def get_wf_sims(seed):
    wf_sims = []
    for N in [5, 10, 20]:
        for surv in [0.0, 0.5, 0.9]:
            for mut in [0.01, 1.0]:
                for nloci in [1, 2, 3]:
                    tables = wf_sim(N=N, ngens=N, survival=surv, seed=seed)
                    tables.sort()
                    ts = tables.tree_sequence()
                    ts = tsutil.jukes_cantor(ts, num_sites=nloci, mu=mut, seed=seed)
                    wf_sims.append(ts)
    return wf_sims


# List of simulations used to parametrize tests.
wf_sims = get_wf_sims(1234)


class TestSimplify:
    """
    Tests for simplify on cases generated by the Wright-Fisher simulator.
    """

    def verify_simplify(self, ts, new_ts, samples, node_map):
        """
        Check that trees in `ts` match `new_ts` using the specified node_map.
        Modified from `verify_simplify_topology`.  Also check that the `parent`
        and `time` column in the MutationTable is correct.
        """
        # check trees agree at these points
        locs = [random.random() for _ in range(20)]
        locs += random.sample(list(ts.breakpoints())[:-1], min(20, ts.num_trees))
        locs.sort()
        old_trees = ts.trees()
        new_trees = new_ts.trees()
        old_right = -1
        new_right = -1
        for loc in locs:
            while old_right <= loc:
                old_tree = next(old_trees)
                old_left, old_right = old_tree.get_interval()
            assert old_left <= loc < old_right
            while new_right <= loc:
                new_tree = next(new_trees)
                new_left, new_right = new_tree.get_interval()
            assert new_left <= loc < new_right
            # print("comparing trees")
            # print("interval:", old_tree.interval)
            # print(old_tree.draw(format="unicode"))
            # print("interval:", new_tree.interval)
            # print(new_tree.draw(format="unicode"))
            pairs = itertools.islice(itertools.combinations(samples, 2), 500)
            for pair in pairs:
                mapped_pair = [node_map[u] for u in pair]
                mrca1 = old_tree.get_mrca(*pair)
                assert mrca1 != tskit.NULL
                mrca2 = new_tree.get_mrca(*mapped_pair)
                assert mrca2 != tskit.NULL
                assert node_map[mrca1] == mrca2
        mut_parent = tsutil.compute_mutation_parent(ts=ts)
        nt.assert_equal(mut_parent, ts.tables.mutations.parent)

    def verify_haplotypes(self, ts, samples):
        """
        Check that haplotypes are unchanged by simplify.
        """
        sub_ts, node_map = ts.simplify(samples, map_nodes=True, filter_sites=False)
        # Sites tables should be equal
        assert ts.tables.sites == sub_ts.tables.sites
        sub_haplotypes = dict(zip(sub_ts.samples(), sub_ts.haplotypes()))
        all_haplotypes = dict(zip(ts.samples(), ts.haplotypes()))
        mapped_ids = []
        for node_id, h in all_haplotypes.items():
            mapped_node_id = node_map[node_id]
            if mapped_node_id in sub_haplotypes:
                assert h == sub_haplotypes[mapped_node_id]
                mapped_ids.append(mapped_node_id)
        assert sorted(mapped_ids) == sorted(sub_ts.samples())

    @pytest.mark.parametrize("ts", wf_sims)
    def test_python_simplify_all_samples(self, ts):
        s = tests.Simplifier(ts, ts.samples())
        py_full_ts, py_full_map = s.simplify()
        full_ts, full_map = ts.simplify(ts.samples(), map_nodes=True)
        assert all(py_full_map == full_map)
        full_ts.tables.assert_equals(py_full_ts.tables, ignore_provenance=True)

    @pytest.mark.parametrize("ts", wf_sims)
    @pytest.mark.parametrize("nsamples", [2, 5, 10])
    def test_python_simplify_sample_subset(self, ts, nsamples):
        sub_samples = random.sample(list(ts.samples()), min(nsamples, ts.sample_size))
        s = tests.Simplifier(ts, sub_samples)
        py_small_ts, py_small_map = s.simplify()
        small_ts, small_map = ts.simplify(samples=sub_samples, map_nodes=True)
        small_ts.tables.assert_equals(py_small_ts.tables, ignore_provenance=True)
        self.verify_simplify(ts, small_ts, sub_samples, small_map)
        self.verify_haplotypes(ts, samples=sub_samples)

    @pytest.mark.parametrize("ts", wf_sims)
    @pytest.mark.parametrize("nsamples", [2, 5, 10])
    def test_simplify_tables(self, ts, nsamples):
        tables = ts.dump_tables()
        sub_samples = random.sample(list(ts.samples()), min(nsamples, ts.num_samples))
        node_map = tables.simplify(samples=sub_samples)
        small_ts = tables.tree_sequence()
        other_tables = small_ts.dump_tables()
        tables.assert_equals(other_tables, ignore_provenance=True)
        self.verify_simplify(ts, small_ts, sub_samples, node_map)

    @pytest.mark.parametrize("ts", wf_sims)
    @pytest.mark.parametrize("nsamples", [2, 5])
    def test_simplify_keep_unary(self, ts, nsamples):
        np.random.seed(123)
        ts = tsutil.mark_metadata(ts, "nodes")
        sub_samples = random.sample(list(ts.samples()), min(nsamples, ts.num_samples))
        random_nodes = np.random.choice(ts.num_nodes, ts.num_nodes // 2)
        ts = tsutil.insert_individuals(ts, random_nodes)
        ts = tsutil.mark_metadata(ts, "individuals")

        for params in [{}, {"keep_unary": True}, {"keep_unary_in_individuals": True}]:
            sts = ts.simplify(sub_samples, **params)
            # check samples match
            assert sts.num_samples == len(sub_samples)
            for n, sn in zip(sub_samples, sts.samples()):
                assert ts.node(n).metadata == sts.node(sn).metadata

            # check that nodes are correctly retained: only nodes ancestral to
            # retained samples, and: by default, only coalescent events; if
            # keep_unary_in_individuals then also nodes in individuals; if
            # keep_unary then all such nodes.
            for t in ts.trees(tracked_samples=sub_samples):
                st = sts.at(t.interval.left)
                visited = [False for _ in sts.nodes()]
                for n, sn in zip(sub_samples, sts.samples()):
                    last_n = t.num_tracked_samples(n)
                    while n != tskit.NULL:
                        ind = ts.node(n).individual
                        keep = False
                        if t.num_tracked_samples(n) > last_n:
                            # a coalescent node
                            keep = True
                        if "keep_unary_in_individuals" in params and ind != tskit.NULL:
                            keep = True
                        if "keep_unary" in params:
                            keep = True
                        if (n in sub_samples) or keep:
                            visited[sn] = True
                            assert sn != tskit.NULL
                            assert ts.node(n).metadata == sts.node(sn).metadata
                            assert t.num_tracked_samples(n) == st.num_samples(sn)
                            if ind != tskit.NULL:
                                sind = sts.node(sn).individual
                                assert sind != tskit.NULL
                                assert (
                                    ts.individual(ind).metadata
                                    == sts.individual(sind).metadata
                                )
                            sn = st.parent(sn)
                        last_n = t.num_tracked_samples(n)
                        n = t.parent(n)
                st_nodes = list(st.nodes())
                for k, v in enumerate(visited):
                    assert v == (k in st_nodes)
