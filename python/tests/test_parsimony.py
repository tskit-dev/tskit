# MIT License
#
# Copyright (c) 2019-2022 Tskit Developers
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
import dataclasses
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
    bp_tree = Bio.Phylo.read(io.StringIO(tree.as_newick()), "newick")
    records = [
        Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(str(genotypes[j])), id=f"n{u}")
        for j, u in enumerate(ts.samples())
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
    bp_tree = Bio.Phylo.read(io.StringIO(tree.as_newick()), "newick")
    records = [
        Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(str(genotypes[j])), id=f"n{u}")
        for j, u in enumerate(ts.samples())
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
    S = np.zeros((tree.tree_sequence.num_nodes, num_alleles))
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
    A = np.zeros((tree.tree_sequence.num_nodes, num_alleles), dtype=np.int8)
    for allele, u in zip(genotypes, tree.tree_sequence.samples()):
        if allele != -1:
            A[u, allele] = 1
        else:
            A[u] = 1
    for u in tree.nodes(order="postorder"):
        if tree.num_children(u) > 2:
            raise ValueError("Fitch parsimony is for binary trees only")
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


def hartigan_map_mutations(tree, genotypes, alleles, ancestral_state=None):
    """
    Returns a Hartigan parsimony reconstruction for the specified set of genotypes.
    The reconstruction is specified by returning the ancestral state and a
    list of mutations on the tree. Each mutation is a (node, parent, state)
    triple, where node is the node over which the transition occurs, the
    parent is the index of the parent transition above it on the tree (or -1
    if there is none) and state is the new state.
    """
    # The python version of map_mutations allows the ancestral_state to be a string
    # from the alleles list, so we implement this at the top of this function although
    # it doesn't need to be in the C equivalent of this function
    if isinstance(ancestral_state, str):
        ancestral_state = alleles.index(ancestral_state)

    # equivalent C implementation can start here
    genotypes = np.array(genotypes)
    not_missing = genotypes != -1
    if np.sum(not_missing) == 0:
        raise ValueError("Must have at least one non-missing genotype")
    num_alleles = np.max(genotypes[not_missing]) + 1
    if ancestral_state is not None:
        if ancestral_state < 0 or ancestral_state >= len(alleles):
            raise ValueError("ancestral_state must be a number from 0..(num_alleles-1)")
        if ancestral_state >= num_alleles:
            num_alleles = ancestral_state + 1
    num_nodes = tree.tree_sequence.num_nodes

    # use a numpy array of 0/1 values to represent the set of states
    # to make the code as similar as possible to the C implementation.
    optimal_set = np.zeros((num_nodes + 1, num_alleles), dtype=np.int8)
    for allele, u in zip(genotypes, tree.tree_sequence.samples()):
        if allele != -1:
            optimal_set[u, allele] = 1
        else:
            optimal_set[u] = 1

    allele_count = np.zeros(num_alleles, dtype=int)
    for u in tree.nodes(tree.virtual_root, order="postorder"):
        allele_count[:] = 0
        for v in tree.children(u):
            for j in range(num_alleles):
                allele_count[j] += optimal_set[v, j]
        if not tree.is_sample(u):
            max_allele_count = np.max(allele_count)
            optimal_set[u, allele_count == max_allele_count] = 1

    if ancestral_state is None:
        ancestral_state = np.argmax(optimal_set[tree.virtual_root])
    else:
        optimal_set[tree.virtual_root] = 1

    @dataclasses.dataclass
    class StackElement:
        node: int
        state: int
        mutation_parent: int

    mutations = []
    stack = [StackElement(tree.virtual_root, ancestral_state, -1)]
    while len(stack) > 0:
        s = stack.pop()
        if optimal_set[s.node, s.state] == 0:
            s.state = np.argmax(optimal_set[s.node])
            mutation = tskit.Mutation(
                node=s.node,
                derived_state=alleles[s.state],
                parent=s.mutation_parent,
            )
            s.mutation_parent = len(mutations)
            mutations.append(mutation)
        for v in tree.children(s.node):
            stack.append(StackElement(v, s.state, s.mutation_parent))
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
            assert len(transitions1) == len(transitions2)
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

    def do_map_mutations(
        self, tree, genotypes, alleles=None, ancestral_state=None, compare_lib=True
    ):
        if alleles is None:
            alleles = [str(j) for j in range(max(genotypes) + 1)]
        ancestral_state1, transitions1 = tree.map_mutations(
            genotypes, alleles, ancestral_state
        )
        if compare_lib:
            ancestral_state2, transitions2 = hartigan_map_mutations(
                tree, genotypes, alleles, ancestral_state
            )
            assert ancestral_state1 == ancestral_state2
            assert len(transitions1) == len(transitions2)
            sorted_t1 = sorted((m.node, m.derived_state) for m in transitions1)
            sorted_t2 = sorted((m.node, m.derived_state) for m in transitions2)
            assert sorted_t1 == sorted_t2
            assert transitions1 == transitions2
        return ancestral_state1, transitions1


class TestParsimonyBadAlleles(TestParsimonyBase):
    tree = tskit.Tree.generate_comb(3)

    def test_too_many_alleles(self):
        genotypes = [0, 0, 64]
        alleles = [str(j) for j in range(max(genotypes) + 1)]
        with pytest.raises(ValueError, match="maximum of 64"):
            # Only a limitation in the C version of map_mutations
            self.tree.map_mutations(genotypes, alleles)

    def test_ancestral_state_too_big(self):
        genotypes = [0, 0, 1]
        alleles = [str(x) for x in range(2**8)]  # exceeds HARTIGAN_MAX_ALLELES
        with pytest.raises(ValueError, match="maximum of 64"):
            # Only a limitation in the C version of map_mutations
            self.tree.map_mutations(
                genotypes, alleles=alleles, ancestral_state=alleles[-1]
            )


class TestParsimonyRoundTrip(TestParsimonyBase):
    """
    Tests that we can reproduce the genotypes for set of tree sequences by
    inferring the locations of mutations.
    """

    def verify(self, ts):
        G = ts.genotype_matrix(isolated_as_missing=False)
        alleles = [v.alleles for v in ts.variants()]
        for randomize_ancestral_states in [False, True]:
            tables = ts.dump_tables()
            tables.sites.clear()
            tables.mutations.clear()
            fixed_anc_state = None
            for tree in ts.trees():
                for site in tree.sites():
                    if randomize_ancestral_states:
                        num_alleles = len(alleles[site.id])
                        if alleles[site.id][-1] is None:
                            num_alleles -= 1
                        fixed_anc_state = np.random.randint(num_alleles)
                    ancestral_state, mutations = self.do_map_mutations(
                        tree,
                        G[site.id],
                        alleles[site.id],
                        ancestral_state=fixed_anc_state,
                    )
                    site_id = tables.sites.append(
                        site.replace(ancestral_state=ancestral_state)
                    )
                    parent_offset = len(tables.mutations)
                    for mutation in mutations:
                        parent = mutation.parent
                        if parent != tskit.NULL:
                            parent += parent_offset
                        tables.mutations.append(
                            mutation.replace(site=site_id, parent=parent)
                        )
            other_ts = tables.tree_sequence()
            for h1, h2 in zip(
                ts.haplotypes(isolated_as_missing=False),
                other_ts.haplotypes(isolated_as_missing=False),
            ):
                assert h1 == h2

            # Make sure we're computing the parent correctly.
            tables2 = tables.copy()
            nulled = np.zeros_like(tables.mutations.parent) - 1
            tables2.mutations.parent = nulled
            assert np.array_equal(tables.mutations.parent, tables.mutations.parent)

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

    def test_jukes_cantor_balanced_ternary_internal_samples(self):
        tree = tskit.Tree.generate_balanced(27, arity=3)
        ts = tsutil.jukes_cantor(tree.tree_sequence, 5, 2, seed=1)
        assert ts.num_sites > 1
        self.verify(tsutil.jiggle_samples(ts))

    def test_infinite_sites_n20_multiroot(self):
        ts = msprime.simulate(20, mutation_rate=3, random_seed=3)
        self.verify(ts.decapitate(np.max(ts.tables.nodes.time) / 2))

    def test_jukes_cantor_n15_multiroot(self):
        ts = msprime.simulate(15, random_seed=1)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 5)
        ts = tsutil.jukes_cantor(ts, 15, 2, seed=3)
        self.verify(ts)

    def test_jukes_cantor_balanced_ternary_multiroot(self):
        ts = tskit.Tree.generate_balanced(50, arity=3).tree_sequence
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 3)
        ts = tsutil.jukes_cantor(ts, 15, 2, seed=3)
        self.verify(ts)
        assert ts.num_sites > 1
        self.verify(tsutil.jiggle_samples(ts))

    def test_jukes_cantor_n50_multiroot(self):
        ts = msprime.simulate(50, random_seed=1)
        ts = ts.decapitate(np.max(ts.tables.nodes.time) / 2)
        ts = tsutil.jukes_cantor(ts, 5, 2, seed=2)
        self.verify(ts)

    def test_jukes_cantor_root_polytomy_n5(self):
        tree = tskit.Tree.unrank(5, (1, 0))
        ts = tsutil.jukes_cantor(tree.tree_sequence, 5, 2, seed=1)
        assert ts.num_sites > 2
        self.verify(ts)

    def test_jukes_cantor_leaf_polytomy_n5(self):
        tree = tskit.Tree.unrank(5, (7, 0))
        ts = tsutil.jukes_cantor(tree.tree_sequence, 5, 2, seed=1)
        assert ts.num_sites > 2
        self.verify(ts)

    @pytest.mark.parametrize(
        "tree_builder", [tskit.Tree.generate_balanced, tskit.Tree.generate_comb]
    )
    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_many_states_binary(self, tree_builder, n):
        tree = tree_builder(n)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0.5, "0")
        for j in range(1, n):
            tables.mutations.add_row(0, derived_state=str(j), node=j)
        ts = tables.tree_sequence()
        assert np.array_equal(ts.genotype_matrix(), [np.arange(n, dtype=np.int8)])
        self.verify(tables.tree_sequence())

    @pytest.mark.parametrize("arity", [2, 3, 4])
    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_many_states_arity(self, n, arity):
        tree = tskit.Tree.generate_balanced(n, arity=arity)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0.5, "0")
        for j in range(1, n):
            tables.mutations.add_row(0, derived_state=str(j), node=j)
        ts = tables.tree_sequence()
        assert np.array_equal(ts.genotype_matrix(), [np.arange(n, dtype=np.int8)])
        self.verify(tables.tree_sequence())


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
                site_id = tables.sites.append(
                    site.replace(ancestral_state=ancestral_state)
                )
                parent_offset = len(tables.mutations)
                for m in mutations:
                    parent = m.parent
                    if m.parent != tskit.NULL:
                        parent = m.parent + parent_offset
                    tables.mutations.append(m.replace(site=site_id, parent=parent))
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

    @pytest.mark.parametrize("n", range(2, 10))
    def test_all_missing(self, n):
        ts = msprime.simulate(n, random_seed=2)
        tree = ts.first()
        genotypes = np.zeros(n, dtype=np.int8) - 1
        alleles = ["0", "1"]
        with pytest.raises(ValueError):
            fitch_map_mutations(tree, genotypes, alleles)
        with pytest.raises(ValueError):
            hartigan_map_mutations(tree, genotypes, alleles)
        with pytest.raises(tskit.LibraryError):
            tree.map_mutations(genotypes, alleles)

    @pytest.mark.parametrize("n", range(2, 10))
    def test_one_non_missing(self, n):
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

    @pytest.mark.parametrize("arity", range(2, 10))
    def test_one_non_missing_balanced(self, arity):
        n = 40
        tree = tskit.Tree.generate_balanced(n, arity=arity)
        for j in range(n):
            genotypes = np.zeros(n, dtype=np.int8) - 1
            genotypes[j] = 0
            ancestral_state, transitions = self.do_map_mutations(
                tree, genotypes, ["0", "1"]
            )
            assert ancestral_state == "0"
            assert len(transitions) == 0

    @pytest.mark.parametrize("n", range(2, 10))
    def test_many_states_half_missing(self, n):
        ts = msprime.simulate(n, random_seed=2)
        tree = ts.first()
        genotypes = np.zeros(n, dtype=np.int8) - 1
        genotypes[0 : n // 2] = np.arange(n // 2, dtype=int)
        alleles = [str(j) for j in range(n)]
        ancestral_state, transitions = self.do_map_mutations(tree, genotypes, alleles)
        assert ancestral_state == "0"
        assert len(transitions) == max(0, n // 2 - 1)

    @pytest.mark.parametrize("n", range(2, 10))
    def test_one_missing(self, n):
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

    @pytest.mark.parametrize("arity", range(2, 10))
    def test_one_missing_balanced(self, arity):
        n = 40
        tree = tskit.Tree.generate_balanced(n, arity=arity)
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


class TestParsimonyExamplesPolytomy(TestParsimonyBase):
    """
    Some examples on a given non-binary tree.
    """

    #         9
    #       ┏━┻━━┓
    #       7    8
    #     ┏━┻━┓ ┏┻┓
    #     6   ┃ ┃ ┃
    #   ┏━╋━┓ ┃ ┃ ┃
    #   0 2 4 5 1 3

    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       1           0
    5       1           0
    6       0           1
    7       0           2
    8       0           2
    9       0           3
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       6       0,2,4
    0       1       7       6,5
    0       1       8       1,3
    0       1       9       7,8
    """
    )

    tree = tskit.load_text(
        nodes=nodes,
        edges=edges,
        strict=False,
    ).first()

    def test_all_zeros(self):
        genotypes = [0, 0, 0, 0, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 0

    def test_mutation_over_8(self):
        genotypes = [0, 1, 0, 1, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=8, derived_state="1")

    def test_mutation_over_6(self):
        genotypes = [1, 0, 1, 0, 1, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=6, derived_state="1")

    def test_mutation_over_0_5(self):
        # Bug reported in https://github.com/tskit-dev/tskit/issues/987
        genotypes = [1, 0, 0, 0, 0, 1]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=0, derived_state="1")
        assert transitions[1] == tskit.Mutation(node=5, derived_state="1")

    def test_mutation_over_7_back_mutation_4(self):
        genotypes = [1, 0, 1, 0, 0, 1]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=7, derived_state="1")
        assert transitions[1] == tskit.Mutation(node=4, derived_state="0", parent=0)


class TestParsimonyExamplesStar(TestParsimonyBase):
    """
    Some examples on star topologies.
    """

    @pytest.mark.parametrize("n", range(3, 8))
    def test_two_states_freq_n_minus_1(self, n):
        tree = tskit.Tree.generate_star(n)
        genotypes = np.zeros(n, dtype=np.int8)
        genotypes[0] = 1
        ancestral_state, transitions = self.do_map_mutations(tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=0, derived_state="1")

        genotypes[:] = 1
        genotypes[0] = 0
        ancestral_state, transitions = self.do_map_mutations(tree, genotypes)
        assert ancestral_state == "1"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=0, derived_state="0")

    @pytest.mark.parametrize("n", range(5, 10))
    def test_two_states_freq_n_minus_2(self, n):
        tree = tskit.Tree.generate_star(n)
        genotypes = np.zeros(n, dtype=np.int8)
        genotypes[0:2] = 1
        ancestral_state, transitions = self.do_map_mutations(tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=1, derived_state="1")
        assert transitions[1] == tskit.Mutation(node=0, derived_state="1")

        genotypes[:] = 1
        genotypes[0:2] = 0
        ancestral_state, transitions = self.do_map_mutations(tree, genotypes)
        assert ancestral_state == "1"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=1, derived_state="0")
        assert transitions[1] == tskit.Mutation(node=0, derived_state="0")

    @pytest.mark.parametrize("n", range(5, 10))
    def test_three_states_freq_n_minus_2(self, n):
        tree = tskit.Tree.generate_star(n)
        genotypes = np.zeros(n, dtype=np.int8)
        genotypes[0] = 1
        genotypes[1] = 2
        ancestral_state, transitions = self.do_map_mutations(tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=1, derived_state="2")
        assert transitions[1] == tskit.Mutation(node=0, derived_state="1")

    @pytest.mark.parametrize("n", range(2, 10))
    def test_n_states(self, n):
        tree = tskit.Tree.generate_star(n)
        genotypes = np.arange(n, dtype=np.int8)
        ancestral_state, transitions = self.do_map_mutations(tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == n - 1

    @pytest.mark.parametrize("n", range(3, 10))
    def test_missing_data(self, n):
        tree = tskit.Tree.generate_star(n)
        genotypes = np.zeros(n, dtype=np.int8)
        genotypes[0] = tskit.MISSING_DATA
        genotypes[1] = 1
        ancestral_state, transitions = self.do_map_mutations(tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=1, derived_state="1")


class TestParsimonyExamplesBalancedTernary(TestParsimonyBase):
    """
    Some examples on a given non-binary tree.
    """

    tree = tskit.Tree.generate_balanced(27, arity=3)
    #                                39
    #         ┏━━━━━━━━━━━━━━━━━━━━━┳━┻━━━━━━━━━━━━━━━━━━━━━━━━┓
    #        30                    34                         38
    #   ┏━━━━━╋━━━━━┓      ┏━━━━━━━━╋━━━━━━━━┓        ┏━━━━━━━━╋━━━━━━━━┓
    #  27    28    29     31       32       33       35       36       37
    # ┏━╋━┓ ┏━╋━┓ ┏━╋━┓ ┏━━╋━━┓  ┏━━╋━━┓  ┏━━╋━━┓  ┏━━╋━━┓  ┏━━╋━━┓  ┏━━╋━━┓
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26

    def test_mutation_over_27_29(self):
        genotypes = np.zeros(27, dtype=int)
        genotypes[0:3] = 1
        genotypes[6:9] = 1
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 2
        # the algorithm chooses a back mutation instead
        assert transitions[0] == tskit.Mutation(node=30, derived_state="1")
        assert transitions[1] == tskit.Mutation(node=28, derived_state="0", parent=0)

    def test_three_clades(self):
        genotypes = np.zeros(27, dtype=int)
        genotypes[9:18] = 1
        genotypes[18:27] = 2
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=38, derived_state="2")
        assert transitions[1] == tskit.Mutation(node=34, derived_state="1")

    def test_nonzero_ancestral_state(self):
        genotypes = np.ones(27, dtype=int)
        genotypes[0] = 0
        genotypes[26] = 0
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "1"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=26, derived_state="0")
        assert transitions[1] == tskit.Mutation(node=0, derived_state="0")

    def test_many_states(self):
        genotypes = np.arange(27, dtype=int)
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 26

    def test_least_parsimonious(self):
        genotypes = [0, 1, 2] * 9
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 18


class TestParsimonyExamplesUnary(TestParsimonyBase):
    """
    Some examples on a tree with unary nodes. The mutation should be placed
    on the highest node along the lineage compatible with the parsimonious placement
    """

    #        9
    #      ┏━┻━┓
    #      8   ┃
    #    ┏━┻━┓ ┃
    #    6   7 ┃
    #    ┃   ┃ ┃
    #    5   ┃ ┃
    #  ┏━╋━┓ ┃ ┃
    #  0 2 3 1 4

    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       1           0
    5       0           1
    6       0           2
    7       0           2
    8       0           3
    9       0           4
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       5       0,2,3
    0       1       6       5
    0       1       7       1
    0       1       8       6
    0       1       8       7
    0       1       9       8
    0       1       9       4
    """
    )

    tree = tskit.load_text(
        nodes=nodes,
        edges=edges,
        strict=False,
    ).first()

    def test_all_zeros(self):
        genotypes = [0, 0, 0, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 0

    def test_mutation_over_6(self):
        genotypes = [1, 0, 1, 1, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=6, derived_state="1")

    def test_mutation_over_7(self):
        genotypes = [0, 1, 0, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=7, derived_state="1")

    def test_reversed_mutation_over_7(self):
        genotypes = [1, 0, 1, 1, 1]
        ancestral_state, transitions = self.do_map_mutations(self.tree, genotypes)
        assert ancestral_state == "1"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=7, derived_state="0")


class TestParsimonyExamplesAncestralState(TestParsimonyBase):
    """
    Test fixing the ancestral state. Note that a mutation can occur above node 10
    to switch the ancestral state
    """

    #     10
    #    ┏━┻━┓
    #    ┃   9
    #    ┃ ┏━┻━┓
    #    ┃ ┃   8
    #    ┃ ┃ ┏━┻━┓
    #    ┃ ┃ ┃   7
    #    ┃ ┃ ┃ ┏━┻┓
    #    ┃ ┃ ┃ ┃  6
    #    ┃ ┃ ┃ ┃ ┏┻┓
    #    0 1 2 3 4 5
    tree = tskit.Tree.generate_comb(6)

    def test_mutation_over_0(self):
        genotypes = [1, 0, 0, 0, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(
            self.tree, genotypes, ancestral_state=0
        )
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=0, parent=-1, derived_state="1")

        ancestral_state, transitions = self.do_map_mutations(
            self.tree, genotypes, ancestral_state=1
        )
        assert ancestral_state == "1"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=9, parent=-1, derived_state="0")

    def test_mutation_over_3(self):
        genotypes = [0, 0, 0, 1, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(
            self.tree, genotypes, ancestral_state=None
        )
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=3, parent=-1, derived_state="1")

        ancestral_state, transitions = self.do_map_mutations(
            self.tree, genotypes, ancestral_state=0
        )
        assert ancestral_state == "0"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=3, parent=-1, derived_state="1")
        ancestral_state, transitions = self.do_map_mutations(
            self.tree, genotypes, ancestral_state=1
        )
        assert ancestral_state == "1"
        assert len(transitions) == 2
        assert transitions[0] == tskit.Mutation(node=10, parent=-1, derived_state="0")
        assert transitions[1] == tskit.Mutation(node=3, parent=0, derived_state="1")

    def test_novel_ancestral_state(self):
        # should put a single mutation above the root
        genotypes = [0, 0, 0, 0, 0, 0]
        for alleles in (["0", "1", "2", "3"], ["0", "1", "2", "3", None]):
            ancestral_state, transitions = self.do_map_mutations(
                self.tree, genotypes, alleles=alleles, ancestral_state=3
            )
            assert len(transitions) == 1
            assert transitions[0] == tskit.Mutation(node=10, derived_state="0")

    def test_mutations_over_root(self):
        tree = tskit.Tree.generate_star(6)
        # Mutations on root children
        genotypes = [0, 0, 0, 1, 1, 1]
        ancestral_state, transitions = self.do_map_mutations(
            tree, genotypes, ancestral_state=1
        )
        assert ancestral_state == "1"
        assert len(transitions) == 3
        assert all(m.derived_state == "0" for m in transitions)
        assert set(range(3)) == {m.node for m in transitions}

        # Should now switch to a mutation over the root
        genotypes = [0, 0, 0, 0, 1, 1]
        ancestral_state, transitions = self.do_map_mutations(
            tree, genotypes, ancestral_state=1
        )
        assert ancestral_state == "1"
        assert len(transitions) == 3
        assert transitions[0] == tskit.Mutation(node=tree.root, derived_state="0")
        assert all(m.derived_state == "1" for m in transitions[1:])
        assert all(m.parent == 0 for m in transitions[1:])
        assert {4, 5} == {m.node for m in transitions[1:]}

    def test_all_isolated_different_from_ancestral(self):
        ts = tskit.Tree.generate_star(6).tree_sequence
        ts = ts.decapitate(0)
        tree = ts.first()
        genotypes = [0, 0, 0, 1, 1, 1]
        ancestral_state, transitions = self.do_map_mutations(
            tree, genotypes, alleles=["A", "T", "G", "C"], ancestral_state=2
        )
        assert len(transitions) == 6
        assert all(m.parent == -1 for m in transitions)
        derived_states = [m.derived_state for m in transitions]
        assert derived_states.count("A") == 3
        assert derived_states.count("T") == 3
        assert {m.node for m in transitions if m.derived_state == "A"} == {0, 1, 2}
        assert {m.node for m in transitions if m.derived_state == "T"} == {3, 4, 5}

    def test_ancestral_as_string(self):
        genotypes = [1, 0, 0, 0, 0, 0]
        ancestral_state, transitions = self.do_map_mutations(
            self.tree, genotypes, alleles=["A", "T", "G", "C"], ancestral_state="A"
        )
        assert ancestral_state == "A"
        assert len(transitions) == 1
        assert transitions[0] == tskit.Mutation(node=0, parent=-1, derived_state="T")

    def test_bad_ancestral_state(self):
        genotypes = [0, 0, 0, 1, 0, 0]
        alleles = [str(j) for j in range(max(genotypes) + 1)]
        for bad, err in {
            2: "ancestral_state",
            -1: "ancestral_state",
            "A": "not in list",
        }.items():
            with pytest.raises(ValueError, match=err):
                hartigan_map_mutations(
                    self.tree, genotypes, alleles=alleles, ancestral_state=bad
                )
            with pytest.raises(ValueError, match=err):
                self.tree.map_mutations(genotypes, alleles=alleles, ancestral_state=bad)


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
        m = k**n
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
                tables.mutations.append(mutation.replace(site=j, parent=parent))

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

    def test_root_polytomy_n5_k4(self):
        tree = tskit.Tree.unrank(5, (1, 0))
        self.verify(tree.tree_sequence, 4)

    def test_leaf_polytomy_n5_k4(self):
        tree = tskit.Tree.unrank(5, (7, 0))
        self.verify(tree.tree_sequence, 4)

    def test_leaf_polytomy_n5_k5(self):
        tree = tskit.Tree.unrank(5, (7, 0))
        self.verify(tree.tree_sequence, 5)
