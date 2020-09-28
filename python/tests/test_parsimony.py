# MIT License
#
# Copyright (c) 2019 Tskit Developers
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
Tests for the tree parsimony methods.
"""
import io
import itertools

import Bio.Phylo.TreeConstruction
import msprime
import numpy as np
import pytest

import tests.tsutil as tsutil
import tskit


INF = np.inf


def bp_sankoff_score(tree, genotypes, cost_matrix):
    """
    Returns the sankoff score matrix computed by BioPython.
    """
    ts = tree.tree_sequence
    bp_tree = Bio.Phylo.read(io.StringIO(tree.newick()), "newick")
    records = [
        Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(str(genotypes[j])), id=str(j + 1))
        for j in range(ts.num_samples)
    ]
    alignment = Bio.Align.MultipleSeqAlignment(records)
    lower_triangular = []
    for j in range(cost_matrix.shape[0]):
        lower_triangular.append(list(cost_matrix[j, : j + 1]))
    bp_matrix = Bio.Phylo.TreeConstruction._Matrix(
        list(map(str, range(cost_matrix.shape[0]))), lower_triangular
    )
    ps = Bio.Phylo.TreeConstruction.ParsimonyScorer(bp_matrix)
    return ps.get_score(bp_tree, alignment)


def bp_fitch_score(tree, genotypes):
    """
    Returns the Fitch parsimony score computed by BioPython.
    """
    ts = tree.tree_sequence
    bp_tree = Bio.Phylo.read(io.StringIO(tree.newick()), "newick")
    records = [
        Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(str(genotypes[j])), id=str(j + 1))
        for j in range(ts.num_samples)
    ]
    alignment = Bio.Align.MultipleSeqAlignment(records)
    ps = Bio.Phylo.TreeConstruction.ParsimonyScorer()
    return ps.get_score(bp_tree, alignment)


def sankoff_score(tree, genotypes, cost_matrix):
    """
    Returns a num_nodes * num_alleles numpy array giving the minimum cost
    scores for the specified genotypes on the specified tree. If a cost
    matrix is provided, it must be a num_alleles * num_alleles array giving
    the cost of transitioning from each allele to every other allele.
    """
    num_alleles = cost_matrix.shape[0]
    S = np.zeros((tree.num_nodes, num_alleles))
    for allele, u in zip(genotypes, tree.tree_sequence.samples()):
        S[u, :] = INF
        S[u, allele] = 0
    for parent in tree.nodes(order="postorder"):
        for child in tree.children(parent):
            for j in range(num_alleles):
                S[parent, j] += np.min(cost_matrix[:, j] + S[child])
    return S


def fitch_score(tree, genotypes):
    """
    Returns the Fitch parsimony score for the specified set of genotypes.
    """
    # Use the simplest set operation encoding of the set operations.
    A = {}
    for allele, u in zip(genotypes, tree.tree_sequence.samples()):
        A[u] = {allele}
    score = 0
    for u in tree.nodes(order="postorder"):
        if tree.is_internal(u):
            A[u] = set.intersection(*[A[v] for v in tree.children(u)])
            if len(A[u]) == 0:
                A[u] = set.union(*[A[v] for v in tree.children(u)])
                score += 1
    return score


def fitch_map_mutations(tree, genotypes, alleles):
    """
    Returns the Fitch parsimony reconstruction for the specified set of genotypes.
    The reconstruction is specified by returning the ancestral state and a
    list of mutations on the tree. Each mutation is a (node, parent, state)
    triple, where node is the node over which the transition occurs, the
    parent is the index of the parent transition above it on the tree (or -1
    if there is none) and state is the new state.
    """
    genotypes = np.array(genotypes)
    # Encode the set operations using a numpy array.
    not_missing = genotypes != -1
    if np.sum(not_missing) == 0:
        raise ValueError("Must have at least one non-missing genotype")
    num_alleles = np.max(genotypes[not_missing]) + 1
    A = np.zeros((tree.num_nodes, num_alleles), dtype=np.int8)
    for allele, u in zip(genotypes, tree.tree_sequence.samples()):
        if allele != -1:
            A[u, allele] = 1
        else:
            A[u] = 1
    for u in tree.nodes(order="postorder"):
        if not tree.is_sample(u):
            A[u] = 1
            for v in tree.children(u):
                A[u] = np.logical_and(A[u], A[v])
            if np.sum(A[u]) == 0:
                for v in tree.children(u):
                    A[u] = np.logical_or(A[u], A[v])

    root_states = np.zeros_like(A[0])
    for root in tree.roots:
        root_states = np.logical_or(root_states, A[root])
    ancestral_state = np.where(root_states == 1)[0][0]

    mutations = []
    state = {}
    for root in tree.roots:
        state[root] = ancestral_state
        parent = tskit.NULL
        if A[root, ancestral_state] != 1:
            state[root] = np.where(A[root] == 1)[0][0]
            mutations.append(
                tskit.Mutation(
                    node=root, parent=tskit.NULL, derived_state=alleles[state[root]]
                )
            )
            parent = len(mutations) - 1
        stack = [(root, parent)]
        while len(stack) > 0:
            u, parent_mutation = stack.pop()
            for v in tree.children(u):
                state[v] = state[u]
                if A[v, state[u]] != 1:
                    state[v] = np.where(A[v] == 1)[0][0]
                    mutations.append(
                        tskit.Mutation(
                            node=v,
                            parent=parent_mutation,
                            derived_state=alleles[state[v]],
                        )
                    )
                    stack.append((v, len(mutations) - 1))
                else:
                    stack.append((v, parent_mutation))
    return alleles[ancestral_state], mutations


def reconstruct_states(tree, genotypes, S, cost_matrix):
    """
    Given the specified observations for the samples and tree score
    matrix computed by sankoff_score and the transition cost matrix,
    return the ancestral_state and state transitions on the tree.
    """
    root_cost = np.zeros_like(S[0])
    for root in tree.roots:
        for j in range(S.shape[1]):
            root_cost[j] += np.min(cost_matrix[:, j] + S[root])
    ancestral_state = np.argmin(root_cost)

    transitions = {}
    A = {}
    for root in tree.roots:
        A[root] = ancestral_state
        for u in tree.nodes(order="preorder"):
            for v in tree.children(u):
                cost = cost_matrix[A[u]] + S[v]
                A[v] = np.argmin(cost)
                if A[u] != A[v]:
                    transitions[v] = A[v]

    return ancestral_state, transitions


def sankoff_map_mutations(tree, genotypes, cost_matrix=None):
    """
    Returns the recontructed minimal state transitions for the specified set of
    genotypes on the specified (optional) cost matrix.

    NOTE: we don't consider complications of multiple roots and internal samples
    here.

    TODO: update this to take the alleles as input like the other methods.
    """
    if cost_matrix is None:
        num_alleles = np.max(genotypes) + 1
        cost_matrix = np.ones((num_alleles, num_alleles))
        np.fill_diagonal(cost_matrix, 0)
    S = sankoff_score(tree, genotypes, cost_matrix)
    return reconstruct_states(tree, genotypes, S, cost_matrix)


def felsenstein_tables():
    """
    Return tables for the example tree.
    """
    #
    #     8
    #   ┏━┻━━┓
    #   ┃    7
    #   ┃   ┏┻┓
    #   6   ┃ ┃
    # ┏━┻┓  ┃ ┃
    # ┃  5  ┃ ┃
    # ┃ ┏┻┓ ┃ ┃
    # 2 3 4 0 1
    #
    tables = tskit.TableCollection(1)
    for _ in range(5):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for j in range(4):
        tables.nodes.add_row(flags=0, time=j + 1)
    tables.edges.add_row(0, 1, 7, 0)
    tables.edges.add_row(0, 1, 7, 1)
    tables.edges.add_row(0, 1, 6, 2)
    tables.edges.add_row(0, 1, 5, 3)
    tables.edges.add_row(0, 1, 5, 4)
    tables.edges.add_row(0, 1, 6, 5)
    tables.edges.add_row(0, 1, 8, 6)
    tables.edges.add_row(0, 1, 8, 7)
    tables.sort()
    return tables


def felsenstein_example():
    """
    Returns the tree used in Felsenstein's book, pg.15.
    """
    ts = felsenstein_tables().tree_sequence()
    return ts.first()


class TestSankoff:
    """
    Tests for the Sankoff algorithm.
    """

    def test_felsenstein_example_score(self):
        tree = felsenstein_example()
        genotypes = [1, 0, 1, 0, 2]
        cost_matrix = np.array(
            [[0, 2.5, 1, 2.5], [2.5, 0, 2.5, 1], [1, 2.5, 0, 2.5], [2.5, 1, 2.5, 0]]
        )
        S = sankoff_score(tree, genotypes, cost_matrix)
        S2 = [
            [INF, 0, INF, INF],
            [0, INF, INF, INF],
            [INF, 0, INF, INF],
            [0, INF, INF, INF],
            [INF, INF, 0, INF],
            [1, 5, 1, 5],
            [3.5, 3.5, 3.5, 4.5],
            [2.5, 2.5, 3.5, 3.5],
            [6, 6, 7, 8],
        ]
        assert np.array_equal(S, np.array(S2))

    def test_felsenstein_example_reconstruct(self):
        tree = felsenstein_example()
        genotypes = [1, 0, 1, 0, 2]
        cost_matrix = np.array(
            [[0, 2.5, 1, 2.5], [2.5, 0, 2.5, 1], [1, 2.5, 0, 2.5], [2.5, 1, 2.5, 0]]
        )
        S = sankoff_score(tree, genotypes, cost_matrix)
        ancestral_state, transitions = reconstruct_states(
            tree, genotypes, S, cost_matrix
        )
        assert {2: 1, 4: 2, 0: 1} == transitions
        assert 0 == ancestral_state

    def verify_infinite_sites(self, ts):
        assert ts.num_trees == 1
        assert ts.num_sites > 5
        tree = ts.first()
        for variant in ts.variants():
            ancestral_state, transitions = sankoff_map_mutations(
                tree, variant.genotypes
            )
            assert len(transitions) == 1
            assert ancestral_state == 0
            assert transitions[variant.site.mutations[0].node] == 1

    def test_infinite_sites_binary_n2(self):
        ts = msprime.simulate(2, mutation_rate=10, random_seed=1)
        self.verify_infinite_sites(ts)

    def test_infinite_sites_binary_n50(self):
        ts = msprime.simulate(50, mutation_rate=2, random_seed=1)
        self.verify_infinite_sites(ts)

    def test_infinite_sites_acgt_n2(self):
        ts = msprime.simulate(2, random_seed=1)
        ts = msprime.mutate(
            ts, rate=3, model=msprime.InfiniteSites(msprime.NUCLEOTIDES), random_seed=1
        )
        self.verify_infinite_sites(ts)

    def test_infinite_sites_acgt_n15(self):
        ts = msprime.simulate(2, random_seed=1)
        ts = msprime.mutate(
            ts, rate=3, model=msprime.InfiniteSites(msprime.NUCLEOTIDES), random_seed=1
        )
        self.verify_infinite_sites(ts)

    def verify_jukes_cantor(self, ts, cost_matrix):
        assert ts.num_trees == 1
        assert ts.num_mutations > ts.num_sites
        tree = ts.first()
        for variant in ts.variants():
            single_score = bp_sankoff_score(tree, variant.genotypes, cost_matrix)
            score_matrix = sankoff_score(tree, variant.genotypes, cost_matrix)
            score = np.min(score_matrix[tree.root])
            assert single_score == score

    def test_jukes_cantor_n2_simple_matrix(self):
        cost_matrix = np.ones((4, 4))
        np.fill_diagonal(cost_matrix, 0)
        ts = msprime.simulate(2, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=1)
        self.verify_jukes_cantor(ts, cost_matrix)

    def test_jukes_cantor_n20_simple_matrix(self):
        cost_matrix = np.ones((4, 4))
        np.fill_diagonal(cost_matrix, 0)
        ts = msprime.simulate(20, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=1)
        self.verify_jukes_cantor(ts, cost_matrix)

    def test_jukes_cantor_n2_felsenstein_matrix(self):
        cost_matrix = np.array(
            [[0, 2.5, 1, 2.5], [2.5, 0, 2.5, 1], [1, 2.5, 0, 2.5], [2.5, 1, 2.5, 0]]
        )
        ts = msprime.simulate(2, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=1)
        self.verify_jukes_cantor(ts, cost_matrix)

    def test_jukes_cantor_n20_felsenstein_matrix(self):
        cost_matrix = np.array(
            [[0, 2.5, 1, 2.5], [2.5, 0, 2.5, 1], [1, 2.5, 0, 2.5], [2.5, 1, 2.5, 0]]
        )
        ts = msprime.simulate(20, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=1)
        self.verify_jukes_cantor(ts, cost_matrix)


class TestFitchParsimonyDistance:
    """
    Tests for the Fitch parsimony algorithm.
    """

    def verify(self, ts):
        assert ts.num_trees == 1
        assert ts.num_sites > 3
        tree = ts.first()
        for variant in ts.variants(isolated_as_missing=False):
            score = fitch_score(tree, variant.genotypes)
            bp_score = bp_fitch_score(tree, variant.genotypes)
            assert bp_score == score
            ancestral_state1, transitions1 = fitch_map_mutations(
                tree, variant.genotypes, variant.alleles
            )
            ancestral_state2, transitions2 = tree.map_mutations(
                variant.genotypes, variant.alleles
            )
            assert ancestral_state1 == ancestral_state2
            assert transitions1 == transitions2
            # The Sankoff algorithm doesn't recontruct the state in the same way.
            # Just a limitation of the implementation.
            ancestral_state3, transitions3 = sankoff_map_mutations(
                tree, variant.genotypes
            )
            assert ancestral_state1 == variant.alleles[ancestral_state3]
            # The algorithms will make slightly different choices on where to put
            # the transitions, but they are equally parsimonious.
            assert len(transitions1) == len(transitions3)

    def test_infinite_sites_binary_n2(self):
        ts = msprime.simulate(2, mutation_rate=10, random_seed=1)
        self.verify(ts)

    def test_infinite_sites_binary_n50(self):
        ts = msprime.simulate(50, mutation_rate=2, random_seed=1)
        self.verify(ts)

    def test_infinite_sites_acgt_n2(self):
        ts = msprime.simulate(2, random_seed=1)
        ts = msprime.mutate(
            ts, rate=3, model=msprime.InfiniteSites(msprime.NUCLEOTIDES), random_seed=1
        )
        self.verify(ts)

    def test_jukes_cantor_n2(self):
        ts = msprime.simulate(2, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=1)
        self.verify(ts)

    def test_jukes_cantor_n5(self):
        ts = msprime.simulate(5, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 5, 1, seed=0)
        self.verify(ts)

    def test_jukes_cantor_n20(self):
        ts = msprime.simulate(20, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=11)
        self.verify(ts)

    def test_jukes_cantor_n50(self):
        ts = msprime.simulate(50, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=1)
        self.verify(ts)


class TestParsimonyBase:
    """
    Base class for tests of the map_mutations parsimony method.
    """

    def do_map_mutations(self, tree, genotypes, alleles=None, compare_lib=True):
        if alleles is None:
            alleles = [str(j) for j in range(max(genotypes) + 1)]
        ancestral_state, transitions = fitch_map_mutations(tree, genotypes, alleles)
        if compare_lib:
            ancestral_state1, transitions1 = tree.map_mutations(genotypes, alleles)
            assert ancestral_state == ancestral_state1
            assert transitions == transitions1
        return ancestral_state, transitions


class TestParsimonyRoundTrip(TestParsimonyBase):
    """
    Tests that we can reproduce the genotypes for set of tree sequences by
    inferring the locations of mutations.
    """

    def verify(self, ts):
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        G = ts.genotype_matrix(isolated_as_missing=False)
        alleles = [v.alleles for v in ts.variants()]
        for tree in ts.trees():
            for site in tree.sites():
                ancestral_state, mutations = self.do_map_mutations(
                    tree, G[site.id], alleles[site.id]
                )
                site_id = tables.sites.add_row(site.position, ancestral_state)
                parent_offset = len(tables.mutations)
                for mutation in mutations:
                    parent = mutation.parent
                    if parent != tskit.NULL:
                        parent += parent_offset
                    tables.mutations.add_row(
                        site_id,
                        node=mutation.node,
                        time=mutation.time,
                        parent=parent,
                        derived_state=mutation.derived_state,
                    )
        other_ts = tables.tree_sequence()
        for h1, h2 in zip(
            ts.haplotypes(isolated_as_missing=False),
            other_ts.haplotypes(isolated_as_missing=False),
        ):
            assert h1 == h2

        # Run again without the mutation parent to make sure we're doing it
        # properly.
        tables2 = ts.dump_tables()
        tables2.sites.clear()
        tables2.mutations.clear()
        G = ts.genotype_matrix(isolated_as_missing=False)
        alleles = [v.alleles for v in ts.variants()]
        for tree in ts.trees():
            for site in tree.sites():
                ancestral_state, mutations = tree.map_mutations(
                    G[site.id], alleles[site.id]
                )
                site_id = tables2.sites.add_row(site.position, ancestral_state)
                for mutation in mutations:
                    tables2.mutations.add_row(
                        site_id,
                        node=mutation.node,
                        time=mutation.time,
                        derived_state=mutation.derived_state,
                    )
        tables2.sort()
        tables2.build_index()
        tables2.compute_mutation_parents()
        assert tables.sites == tables2.sites
        assert tables.mutations == tables2.mutations

    def test_infinite_sites_n3(self):
        ts = msprime.simulate(3, mutation_rate=3, random_seed=3)
        self.verify(ts)

    def test_infinite_sites_n20(self):
        ts = msprime.simulate(20, mutation_rate=3, random_seed=3)
        self.verify(ts)

    def test_infinite_sites_n20_recombination(self):
        ts = msprime.simulate(20, mutation_rate=3, recombination_rate=2, random_seed=3)
        assert ts.num_trees > 2
        self.verify(ts)

    def test_infinite_sites_n5_internal_samples(self):
        ts = msprime.simulate(5, mutation_rate=3, random_seed=3)
        self.verify(tsutil.jiggle_samples(ts))

    def test_infinite_sites_n20_internal_samples(self):
        ts = msprime.simulate(20, mutation_rate=3, random_seed=3)
        self.verify(tsutil.jiggle_samples(ts))

    def test_jukes_cantor_n5(self):
        ts = msprime.simulate(5, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 1, seed=1)
        self.verify(ts)

    def test_jukes_cantor_n20(self):
        ts = msprime.simulate(20, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=1)
        self.verify(ts)

    def test_jukes_cantor_n50(self):
        ts = msprime.simulate(50, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=2)
        self.verify(ts)

    def test_jukes_cantor_n5_internal_samples(self):
        ts = msprime.simulate(5, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 1, 1, seed=1)
        ts = tsutil.jiggle_samples(ts)
        self.verify(ts)

    def test_jukes_cantor_n20_internal_samples(self):
        ts = msprime.simulate(20, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=1)
        self.verify(tsutil.jiggle_samples(ts))

    def test_jukes_cantor_n50_internal_samples(self):
        ts = msprime.simulate(50, random_seed=1)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=2)
        self.verify(tsutil.jiggle_samples(ts))

    def test_infinite_sites_n20_multiroot(self):
        ts = msprime.simulate(20, mutation_rate=3, random_seed=3)
        self.verify(tsutil.decapitate(ts, ts.num_edges // 2))

    def test_jukes_cantor_n15_multiroot(self):
        ts = msprime.simulate(15, random_seed=1)
        ts = tsutil.decapitate(ts, ts.num_edges // 3)
        ts = tsutil.jukes_cantor(ts, 15, 2, seed=3)
        self.verify(ts)

    def test_jukes_cantor_n50_multiroot(self):
        ts = msprime.simulate(50, random_seed=1)
        ts = tsutil.decapitate(ts, ts.num_edges // 2)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=2)
        self.verify(ts)


class TestParsimonyRoundTripMissingData(TestParsimonyRoundTrip):
    """
    Tests that we can reproduce the genotypes for set of tree sequences by
    inferring the locations of mutations.
    """

    def verify(self, ts):
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        G = ts.genotype_matrix(isolated_as_missing=False)
        # Set the first sample to missing data everywhere
        G[:, 0] = -1
        alleles = [v.alleles for v in ts.variants()]
        for tree in ts.trees():
            for site in tree.sites():
                ancestral_state, mutations = self.do_map_mutations(
                    tree, G[site.id], alleles[site.id]
                )
                site_id = tables.sites.add_row(site.position, ancestral_state)
                parent_offset = len(tables.mutations)
                for m in mutations:
                    parent = m.parent
                    if m.parent != tskit.NULL:
                        parent = m.parent + parent_offset
                    tables.mutations.add_row(
                        site_id,
                        node=m.node,
                        time=m.time,
                        parent=parent,
                        derived_state=m.derived_state,
                    )
        other_ts = tables.tree_sequence()
        assert ts.num_samples == other_ts.num_samples
        H1 = list(ts.haplotypes(isolated_as_missing=False))
        H2 = list(other_ts.haplotypes(isolated_as_missing=False))
        # All samples except 0 should be reproduced exactly.
        assert H1[1:] == H2[1:]


class TestParsimonyMissingData(TestParsimonyBase):
    """
    Tests that we correctly map_mutations when we have missing data.
    """

    def test_all_missing(self):
        for n in range(2, 10):
            ts = msprime.simulate(n, random_seed=2)
            tree = ts.first()
            genotypes = np.zeros(n, dtype=np.int8) - 1
            alleles = ["0", "1"]
            with pytest.raises(ValueError):
                fitch_map_mutations(tree, genotypes, alleles)
            with pytest.raises(tskit.LibraryError):
                tree.map_mutations(genotypes, alleles)

    def test_one_non_missing(self):
        for n in range(2, 10):
            ts = msprime.simulate(n, random_seed=2)
            tree = ts.first()
            for j in range(n):
                genotypes = np.zeros(n, dtype=np.int8) - 1
                genotypes[j] = 0
                ancestral_state, transitions = self.do_map_mutations(
                    tree, genotypes, ["0", "1"]
                )
                assert ancestral_state == "0"
                assert len(transitions) == 0

    def test_many_states_half_missing(self):
        for n in range(2, 20):
            ts = msprime.simulate(n, random_seed=2)
            tree = ts.first()
            genotypes = np.zeros(n, dtype=np.int8) - 1
            genotypes[0 : n // 2] = np.arange(n // 2, dtype=int)
            alleles = [str(j) for j in range(n)]
            ancestral_state, transitions = self.do_map_mutations(
                tree, genotypes, alleles
            )
            assert ancestral_state == "0"
            assert len(transitions) == max(0, n // 2 - 1)

    def test_one_missing(self):
        for n in range(2, 10):
            ts = msprime.simulate(n, random_seed=2)
            tree = ts.first()
            alleles = [str(j) for j in range(2)]
            for j in range(n):
                genotypes = np.zeros(n, dtype=np.int8) - 1
                genotypes[j] = 0
                ancestral_state, transitions = self.do_map_mutations(
                    tree, genotypes, alleles
                )
                assert ancestral_state == "0"
                assert len(transitions) == 0

    def test_one_missing_derived_state(self):
        tables = felsenstein_tables()
        ts = tables.tree_sequence()
        genotypes = np.zeros(5, dtype=np.int8)
        genotypes[0] = -1
        genotypes[1] = 1
        alleles = [str(j) for j in range(2)]
        ancestral_state, transitions = self.do_map_mutations(
            ts.first(), genotypes, alleles
        )
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0].node == 7
        assert transitions[0].parent == -1
        assert transitions[0].derived_state == "1"


class TestParsimonyExamples(TestParsimonyBase):
    """
    Some examples on a given tree.
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
    tree = tskit.load_text(
        nodes=io.StringIO(small_tree_ex_nodes),
        edges=io.StringIO(small_tree_ex_edges),
        strict=False,
    ).first()

    def test_mutation_over_0(self):
        genotypes = [1, 0, 0, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=0, parent=-1, derived_state="1")

    def test_mutation_over_5(self):
        genotypes = [1, 1, 0, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=5, parent=-1, derived_state="1")

    def test_mutation_over_7(self):
        genotypes = [1, 1, 0, 0, 1]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=7, parent=-1, derived_state="1")

    def test_mutation_over_7_0(self):
        genotypes = [2, 1, 0, 0, 1]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=7, parent=-1, derived_state="1")
        assert transitions[1] == tskit.Mutation(node=0, parent=0, derived_state="2")

    def test_mutation_over_7_0_alleles(self):
        genotypes = [2, 1, 0, 0, 1]
        alleles = ["ANC", "ONE", "TWO"]
        ancestral_state, transitions = self.do_map_mutations(
            self.tree, genotypes, alleles
        )
        assert ancestral_state == "ANC"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=7, parent=-1, derived_state="ONE")
        assert transitions[1] == tskit.Mutation(node=0, parent=0, derived_state="TWO")

    def test_mutation_over_7_missing_data_0(self):
        genotypes = [-1, 1, 0, 0, 1]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=7, parent=-1, derived_state="1")

    def test_mutation_over_leaf_sibling_missing(self):
        genotypes = [0, 0, 1, -1, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        # We assume that the mutation is over the parent of 2 and the missing data
        # so we impute that 3 also has allele 1. This suprising behaviour to me:
        # I would have thought it was more parsimonious to assume that the missing
        # data had the ancestral state. However, the number of *state changes*
        # is the same, which is what the algorithm is minimising.
        assert transitions[0] == tskit.Mutation(node=6, parent=-1, derived_state="1")

        # Reverse is the same
        genotypes = [0, 0, -1, 1, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=6, parent=-1, derived_state="1")

    def test_mutation_over_6_missing_data_0(self):
        genotypes = [-1, 0, 1, 1, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=6, parent=-1, derived_state="1")

    def test_mutation_over_0_missing_data_4(self):
        genotypes = [1, 0, 0, 0, -1]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=0, parent=-1, derived_state="1")

    def test_multi_mutation_missing_data(self):
        genotypes = [1, 2, -1, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=5, parent=-1, derived_state="1")
        assert transitions[1] == tskit.Mutation(node=1, parent=0, derived_state="2")


class TestReconstructAllTuples:
    """
    Tests that the parsimony algorithm correctly round-trips all possible
    states.
    """

    def verify(self, ts, k):
        tables = ts.dump_tables()
        assert ts.num_trees == 1
        tree = ts.first()
        n = ts.num_samples
        m = k ** n
        tables.sequence_length = m + 1
        tables.edges.set_columns(
            left=tables.edges.left,
            right=np.zeros_like(tables.edges.right) + tables.sequence_length,
            parent=tables.edges.parent,
            child=tables.edges.child,
        )
        G1 = np.zeros((m, n), dtype=np.int8)
        alleles = [str(j) for j in range(k)]
        for j, genotypes in enumerate(itertools.product(range(k), repeat=n)):
            G1[j] = genotypes
            ancestral_state, mutations = tree.map_mutations(G1[j], alleles)
            tables.sites.add_row(j, ancestral_state=ancestral_state)
            parent_offset = len(tables.mutations)
            for mutation in mutations:
                parent = mutation.parent
                if parent != tskit.NULL:
                    parent += parent_offset
                tables.mutations.add_row(
                    j,
                    node=mutation.node,
                    time=mutation.time,
                    parent=parent,
                    derived_state=mutation.derived_state,
                )

        ts2 = tables.tree_sequence()
        G2 = np.zeros((m, n), dtype=np.int8)
        for j, variant in enumerate(ts2.variants()):
            alleles = np.array(list(map(int, variant.alleles)), dtype=np.int8)
            G2[j] = alleles[variant.genotypes]
        assert np.array_equal(G1, G2)

    def test_simple_n3_k2(self):
        ts = msprime.simulate(3, random_seed=4)
        self.verify(ts, 2)

    def test_simple_n3_k4(self):
        ts = msprime.simulate(3, random_seed=4)
        self.verify(ts, 4)

    def test_simple_n4_k2(self):
        ts = msprime.simulate(4, random_seed=4)
        self.verify(ts, 2)

    def test_simple_n4_k4(self):
        ts = msprime.simulate(4, random_seed=4)
        self.verify(ts, 4)

    def test_simple_n4_k5(self):
        ts = msprime.simulate(4, random_seed=4)
        self.verify(ts, 5)

    def test_simple_n5_k4(self):
        ts = msprime.simulate(5, random_seed=4)
        self.verify(ts, 4)

    def test_simple_n6_k3(self):
        ts = msprime.simulate(6, random_seed=4)
        self.verify(ts, 3)
