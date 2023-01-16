# MIT License
#
# Copyright (c) 2019-2023 Tskit Developers
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
Test cases for generating genotypes/haplotypes.
"""
import itertools
import logging
import random
import re
import textwrap
from xml.etree import ElementTree

import msprime
import numpy as np
import pytest

import tests
import tests.test_wright_fisher as wf
import tests.tsutil as tsutil
import tskit
from tests.test_highlevel import get_example_tree_sequences
from tskit import exceptions
from tskit.genotypes import allele_remap

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.

# TODO replace this with a call to
# example_tree_sequences(discrete_genome=True, snps_only=True)


@tests.cached_example
def get_example_discrete_genome_tree_sequences():
    ret = []
    for ts in get_example_tree_sequences(pytest_params=False):
        if ts.discrete_genome:
            snps = all(len(site.ancestral_state) == 1 for site in ts.sites()) and all(
                len(mut.derived_state) == 1 for mut in ts.mutations()
            )
            if snps:
                ret.append(ts)
    return ret


def naive_get_ancestral_haplotypes(ts):
    """
    Simple implementation using tree traversals. Note that this definition
    won't work when we have topology that's not reachable from a root,
    but this seems more trouble than it's worth dealing with.
    """
    A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.int8)
    A[:] = tskit.MISSING_DATA
    for t in ts.trees():
        for site in t.sites():
            alleles = {site.ancestral_state: 0}
            for u in t.nodes():
                A[u, site.id] = 0
            j = 1
            for mutation in site.mutations:
                if mutation.derived_state not in alleles:
                    alleles[mutation.derived_state] = j
                    j += 1
                for u in t.nodes(mutation.node):
                    A[u, site.id] = alleles[mutation.derived_state]
    return A


class TestGetAncestralHaplotypes:
    """
    Tests for the engine to the actual ancestors from a simulation.
    """

    def verify(self, ts):
        A = naive_get_ancestral_haplotypes(ts)
        # To detect missing data in ancestors we must set all nodes
        # to be samples
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags[:]
        flags[:] = 1
        nodes.set_columns(time=nodes.time, flags=flags)
        ts = tables.tree_sequence()
        B = ts.genotype_matrix().T
        assert np.array_equal(A, B)

    def test_single_tree(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=234)
        self.verify(ts)

    def test_many_trees(self):
        ts = msprime.simulate(
            8, recombination_rate=10, mutation_rate=10, random_seed=234
        )
        assert ts.num_trees > 1
        assert ts.num_sites > 1
        self.verify(ts)

    def test_single_tree_jukes_cantor(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        self.verify(ts)

    def test_single_tree_multichar_mutations(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.insert_multichar_mutations(ts)
        self.verify(ts)

    def test_many_trees_infinite_sites(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        assert ts.num_trees > 2
        self.verify(ts)

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            6, 5, seed=3, deep_history=True, initial_generation_samples=True, num_loci=2
        )
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.08, random_seed=2)
        assert ts.num_sites > 0
        self.verify(ts)

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            9,
            10,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=5,
        )
        tables.sort()
        ts = tables.tree_sequence().simplify()
        ts = msprime.mutate(ts, rate=0.2, random_seed=1234)
        assert ts.num_sites > 0
        self.verify(ts)

    def test_empty_ts(self):
        tables = tskit.TableCollection(1.0)
        for _ in range(10):
            tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        ts = tables.tree_sequence()
        self.verify(ts)


def isolated_samples_genotype_matrix(ts):
    """
    Returns the genotype matrix for the specified tree sequence
    where isolated samples are marked with MISSING_DATA.
    """
    G = ts.genotype_matrix()
    samples = ts.samples()
    sample_index_map = np.zeros(ts.num_nodes, dtype=int) - 1
    for index, sample in enumerate(samples):
        sample_index_map[sample] = index
    for tree in ts.trees():
        for site in tree.sites():
            for root in tree.roots:
                # An isolated sample is any root that has no children.
                if tree.left_child(root) == -1:
                    assert sample_index_map[root] != -1
                    G[site.id, sample_index_map[root]] = -1
    return G


class TestVariantGenerator:
    """
    Tests the variants() method to ensure the output is consistent.
    """

    def get_tree_sequence(self):
        ts = msprime.simulate(
            10, length=10, recombination_rate=1, mutation_rate=10, random_seed=3
        )
        assert ts.get_num_mutations() > 10
        return ts

    def test_dtype(self):
        ts = self.get_tree_sequence()
        for var in ts.variants():
            assert var.genotypes.dtype == np.int32

    def test_dtype_conversion(self):
        # Check if we hit any issues if we assume the variants are uint8
        # as they were prior to version 0.2.0
        ts = self.get_tree_sequence()
        G = ts.genotype_matrix().astype(np.uint8)
        assert G.dtype == np.uint8
        for var in ts.variants():
            assert np.array_equal(G[var.index], var.genotypes)
            assert np.all(G[var.index] == var.genotypes)
            assert [var.alleles[g] for g in var.genotypes] == [
                var.alleles[g] for g in G[var.index]
            ]
            G[var.index, :] = var.genotypes
            assert np.array_equal(G[var.index], var.genotypes)

    def test_multichar_alleles(self):
        ts = tsutil.insert_multichar_mutations(self.get_tree_sequence())
        for var in ts.variants():
            assert len(var.alleles) == 2
            assert var.site.ancestral_state == var.alleles[0]
            assert var.site.mutations[0].derived_state == var.alleles[1]
            assert all(0 <= var.genotypes)
            assert all(var.genotypes <= 1)

    def test_many_alleles(self):
        ts = self.get_tree_sequence()
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        # This gives us a total of 360 permutations.
        alleles = list(map("".join, itertools.permutations("ABCDEF", 4)))
        assert len(alleles) > 127
        tables.sites.add_row(0, alleles[0])
        parent = -1
        num_alleles = 1
        for allele in alleles[1:]:
            ts = tables.tree_sequence()
            var = next(ts.variants())
            assert not var.has_missing_data
            assert var.num_alleles == num_alleles
            assert len(var.alleles) == num_alleles
            assert list(var.alleles) == alleles[:num_alleles]
            assert var.alleles[var.genotypes[0]] == alleles[num_alleles - 1]
            for u in ts.samples():
                if u != 0:
                    assert var.alleles[var.genotypes[u]] == alleles[0]
            tables.mutations.add_row(0, 0, allele, parent=parent)
            parent += 1
            num_alleles += 1

    def test_many_alleles_missing_data(self):
        ts = self.get_tree_sequence()
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        # Add an isolated sample
        tables.nodes.add_row(flags=1, time=0)
        # This gives us a total of 360 permutations.
        alleles = list(map("".join, itertools.permutations("ABCDEF", 4)))
        assert len(alleles) > 127
        tables.sites.add_row(0, alleles[0])
        parent = -1
        num_alleles = 1
        for allele in alleles[1:]:
            ts = tables.tree_sequence()

            var = next(ts.variants())
            assert var.has_missing_data
            assert var.num_alleles == num_alleles
            assert len(var.alleles) == num_alleles + 1
            assert list(var.alleles)[:-1] == alleles[:num_alleles]
            assert var.alleles[-1] is None
            assert var.alleles[var.genotypes[0]] == alleles[num_alleles - 1]
            assert var.genotypes[-1] == -1
            samples = ts.samples()
            for u in samples[:-1]:
                if u != 0:
                    assert var.alleles[var.genotypes[u]] == alleles[0]
            tables.mutations.add_row(0, 0, allele, parent=parent)
            parent += 1
            num_alleles += 1

    def test_site_information(self):
        ts = self.get_tree_sequence()
        for site, variant in zip(ts.sites(), ts.variants()):
            assert site.position == variant.position
            assert site == variant.site

    def test_no_mutations(self):
        ts = msprime.simulate(10)
        assert ts.get_num_mutations() == 0
        variants = list(ts.variants())
        assert len(variants) == 0

    @pytest.mark.parametrize("samples", [None, [1, 2], [2, 4], []])
    def test_genotype_matrix(self, samples):
        ts = self.get_tree_sequence()
        num_samples = ts.num_samples if samples is None else len(samples)
        G = np.empty((ts.num_sites, num_samples), dtype=np.int32)
        for v in ts.variants(samples=samples):
            G[v.index, :] = v.genotypes
        if samples is None:
            G2 = ts.genotype_matrix()
        else:
            G2 = ts.genotype_matrix(samples=samples)
        assert np.array_equal(G, G2)
        assert G2.dtype == np.int32

    def test_recurrent_mutations_over_samples(self):
        ts = self.get_tree_sequence()
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        num_sites = 5
        for j in range(num_sites):
            tables.sites.add_row(
                position=j * ts.sequence_length / num_sites, ancestral_state="0"
            )
            for u in range(ts.sample_size):
                tables.mutations.add_row(site=j, node=u, derived_state="1")
        ts = tables.tree_sequence()
        variants = list(ts.variants())
        assert len(variants) == num_sites
        for site, variant in zip(ts.sites(), variants):
            assert site.position == variant.position
            assert site == variant.site
            assert site.id == variant.index
            assert variant.alleles == ("0", "1")
            assert np.all(variant.genotypes == np.ones(ts.sample_size))

    def test_silent_mutations(self):
        ts = self.get_tree_sequence()
        tree = next(ts.trees())
        tables = ts.dump_tables()
        for u in tree.nodes():
            for sample in tree.samples(u):
                if sample != u:
                    tables.sites.clear()
                    tables.mutations.clear()
                    site = tables.sites.add_row(position=0, ancestral_state="0")
                    tables.mutations.add_row(site=site, node=u, derived_state="1")
                    tables.mutations.add_row(site=site, node=sample, derived_state="1")
                    ts_new = tables.tree_sequence()
                    assert all([v.genotypes[sample] == 1 for v in ts_new.variants()])

    def test_zero_samples(self):
        ts = self.get_tree_sequence()
        for var1, var2 in zip(ts.variants(), ts.variants(samples=[])):
            assert var1.site == var2.site
            assert var1.alleles == var2.alleles
            assert var2.genotypes.shape[0] == 0

    def test_samples(self):
        n = 4
        ts = msprime.simulate(
            n, length=5, recombination_rate=1, mutation_rate=5, random_seed=2
        )
        assert ts.num_sites > 1
        samples = list(range(n))
        # Generate all possible sample lists.
        for j in range(n + 1):
            for s in itertools.permutations(samples, j):
                s = np.array(s, dtype=np.int32)
                count = 0
                for var1, var2 in zip(ts.variants(), ts.variants(samples=s)):
                    assert var1.site == var2.site
                    assert var1.alleles == var2.alleles
                    assert np.array_equal(var1.samples, ts.samples())
                    assert np.array_equal(var2.samples, s)
                    assert var2.genotypes.shape == (len(s),)
                    assert np.array_equal(var1.genotypes[s], var2.genotypes)
                    count += 1
                assert count == ts.num_sites

    def test_samples_64bit(self):
        ts = msprime.simulate(4, length=5, mutation_rate=5, random_seed=2)
        s = np.where(ts.nodes_time == 0)[0]  # normally returns 64 bit ints
        next(ts.variants(samples=s))
        s = np.array(s, dtype=np.int64)  # cast just to make sure
        next(ts.variants(samples=s))

    def test_samples_missing_data(self):
        n = 4
        ts = msprime.simulate(
            n, length=5, recombination_rate=1, mutation_rate=5, random_seed=2
        )
        assert ts.num_sites > 1
        tables = ts.dump_tables()
        tables.delete_intervals([[0.5, 0.6]])
        tables.sites.add_row(0.5, ancestral_state="0")
        tables.sort()
        ts = tables.tree_sequence()
        samples = list(range(n))
        # Generate all possible sample lists.
        for j in range(1, n + 1):
            for s in itertools.permutations(samples, j):
                s = np.array(s, dtype=np.int32)
                count = 0
                for var1, var2 in zip(ts.variants(), ts.variants(samples=s)):
                    assert var1.site == var2.site
                    assert var1.alleles == var2.alleles
                    assert var2.genotypes.shape == (len(s),)
                    assert np.array_equal(var1.genotypes[s], var2.genotypes)
                    count += 1
                assert count == ts.num_sites

    def test_non_sample_samples(self):
        # We don't have to use sample nodes. This does make the terminology confusing
        # but it's probably still the best option.
        ts = msprime.simulate(
            10, length=5, recombination_rate=1, mutation_rate=5, random_seed=2
        )
        tables = ts.dump_tables()
        tables.nodes.set_columns(
            flags=np.zeros_like(tables.nodes.flags) + tskit.NODE_IS_SAMPLE,
            time=tables.nodes.time,
        )
        all_samples_ts = tables.tree_sequence()
        assert all_samples_ts.num_samples == ts.num_nodes

        count = 0
        samples = range(ts.num_nodes)
        for var1, var2 in zip(
            all_samples_ts.variants(isolated_as_missing=False),
            ts.variants(samples=samples, isolated_as_missing=False),
        ):
            assert var1.site == var2.site
            assert var1.alleles == var2.alleles
            assert var2.genotypes.shape == (len(samples),)
            assert np.array_equal(var1.genotypes, var2.genotypes)
            count += 1
        assert count == ts.num_sites

    def verify_jukes_cantor(self, ts):
        assert np.array_equal(ts.genotype_matrix(), ts.genotype_matrix())
        tree = ts.first()
        for variant in ts.variants():
            assert not variant.has_missing_data
            mutations = {
                mutation.node: mutation.derived_state
                for mutation in variant.site.mutations
            }
            for sample_index, u in enumerate(ts.samples()):
                while u not in mutations and u != tskit.NULL:
                    u = tree.parent(u)
                state1 = mutations.get(u, variant.site.ancestral_state)
                state2 = variant.alleles[variant.genotypes[sample_index]]
                assert state1 == state2

    def test_jukes_cantor_n5(self):
        ts = msprime.simulate(5, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 5, 1, seed=2)
        self.verify_jukes_cantor(ts)

    def test_jukes_cantor_n20(self):
        ts = msprime.simulate(20, random_seed=2)
        ts = tsutil.jukes_cantor(ts, 5, 1, seed=2)
        self.verify_jukes_cantor(ts)

    def test_zero_edge_missing_data(self):
        ts = msprime.simulate(10, random_seed=2, mutation_rate=2)
        tables = ts.dump_tables()
        tables.keep_intervals([[0.25, 0.75]])
        # add some sites in the deleted regions
        tables.sites.add_row(0.1, "A")
        tables.sites.add_row(0.2, "A")
        tables.sites.add_row(0.8, "A")
        tables.sites.add_row(0.9, "A")
        tables.sort()
        ts = tables.tree_sequence()
        Gnm = ts.genotype_matrix(isolated_as_missing=False)
        assert np.all(Gnm[0] == 0)
        assert np.all(Gnm[1] == 0)
        assert np.all(Gnm[-1] == 0)
        assert np.all(Gnm[-2] == 0)
        Gm = isolated_samples_genotype_matrix(ts)
        assert np.all(Gm[0] == -1)
        assert np.all(Gm[1] == -1)
        assert np.all(Gm[-1] == -1)
        assert np.all(Gm[-2] == -1)
        Gm2 = ts.genotype_matrix(isolated_as_missing=True)
        assert np.array_equal(Gm, Gm2)

        # Test deprecated param

        with pytest.warns(FutureWarning):
            Gi = ts.genotype_matrix(impute_missing_data=True)
        assert np.array_equal(Gnm, Gi)
        with pytest.warns(FutureWarning):
            Gni = ts.genotype_matrix(impute_missing_data=False)
        assert np.array_equal(Gm, Gni)

        with pytest.warns(FutureWarning):
            G = ts.genotype_matrix(isolated_as_missing=False, impute_missing_data=True)
        assert np.array_equal(Gnm, G)
        with pytest.warns(FutureWarning):
            G = ts.genotype_matrix(isolated_as_missing=True, impute_missing_data=False)
        assert np.array_equal(Gm, G)

    def test_empty_ts_missing_data(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        ts = tables.tree_sequence()
        variants = list(ts.variants())
        assert len(variants) == 1
        var = variants[0]
        assert var.alleles == ("A", None)
        assert var.num_alleles == 1
        assert np.all(var.genotypes == -1)

    def test_empty_ts_incomplete_samples(self):
        # https://github.com/tskit-dev/tskit/issues/776
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        ts = tables.tree_sequence()
        variants = list(ts.variants(samples=[0]))
        assert list(variants[0].genotypes) == [-1]
        variants = list(ts.variants(samples=[1]))
        assert list(variants[0].genotypes) == [-1]

    def test_missing_data_samples(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(0, 0, "T")
        ts = tables.tree_sequence()

        # If we have no samples we still get a list of variants.
        variants = list(ts.variants(samples=[]))
        assert len(variants[0].genotypes) == 0
        assert not variants[0].has_missing_data
        assert variants[0].alleles == ("A", "T")

        # If we have a single sample that's not missing, there's no
        # missing data.
        variants = list(ts.variants(samples=[0]))
        assert len(variants[0].genotypes) == 1
        assert variants[0].genotypes[0] == 1
        assert not variants[0].has_missing_data
        assert variants[0].alleles == ("A", "T")

        # If we have a single sample that is missing, there is
        # missing data.
        variants = list(ts.variants(samples=[1]))
        assert len(variants[0].genotypes) == 1
        assert variants[0].genotypes[0] == -1
        assert variants[0].has_missing_data
        assert variants[0].alleles == ("A", "T", None)

    def test_mutation_over_isolated_sample_not_missing(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(0, 0, "T")
        ts = tables.tree_sequence()
        variants = list(ts.variants())
        assert len(variants) == 1
        var = variants[0]
        assert var.alleles == ("A", "T", None)
        assert var.num_alleles == 2
        assert list(var.genotypes) == [1, -1]

    def test_multiple_mutations_over_isolated_sample(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(0, 0, "T")
        tables.mutations.add_row(0, 0, "G", parent=0)
        ts = tables.tree_sequence()
        variants = list(ts.variants())
        assert len(variants) == 1
        var = variants[0]
        assert var.alleles == ("A", "T", "G", None)
        assert var.num_alleles == 3
        assert len(var.site.mutations) == 2
        assert list(var.genotypes) == [2, -1]

    def test_snipped_tree_sequence_missing_data(self):
        ts = msprime.simulate(
            10, length=10, recombination_rate=0.1, mutation_rate=10, random_seed=3
        )
        tables = ts.dump_tables()
        tables.delete_intervals([[4, 6]], simplify=False)
        tables.sites.add_row(4, ancestral_state="0")
        tables.sites.add_row(5, ancestral_state="0")
        tables.sites.add_row(5.999999, ancestral_state="0")
        tables.sort()
        ts = tables.tree_sequence()
        G = ts.genotype_matrix()
        num_missing = 0
        for var in ts.variants():
            if 4 <= var.site.position < 6:
                assert var.has_missing_data
                assert np.all(var.genotypes == tskit.MISSING_DATA)
                num_missing += 1
            else:
                assert not var.has_missing_data
                assert np.all(var.genotypes != tskit.MISSING_DATA)
            assert np.array_equal(var.genotypes, G[var.site.id])
        assert num_missing == 3

        G = ts.genotype_matrix(isolated_as_missing=False)
        for var in ts.variants(isolated_as_missing=False):
            if 4 <= var.site.position < 6:
                assert not var.has_missing_data
                assert np.all(var.genotypes == 0)
            else:
                assert not var.has_missing_data
                assert np.all(var.genotypes != tskit.MISSING_DATA)
            assert np.array_equal(var.genotypes, G[var.site.id])

    def test_snipped_tree_sequence_mutations_over_isolated(self):
        ts = msprime.simulate(
            10, length=10, recombination_rate=0.1, mutation_rate=10, random_seed=3
        )
        tables = ts.dump_tables()
        tables.delete_intervals([[4, 6]], simplify=False)
        missing_site = tables.sites.add_row(4, ancestral_state="0")
        tables.mutations.add_row(missing_site, node=0, derived_state="1")
        # Add another site in which all the samples are marked with a mutation
        # to the ancestral state. Note: this would normally not be allowed because
        # there's not state change. However, this allows us to mark a sample
        # as not-missing, so it's an important feature.
        missing_site = tables.sites.add_row(5, ancestral_state="0")
        for u in range(10):
            tables.mutations.add_row(missing_site, node=u, derived_state="0")
        tables.sort()
        ts = tables.tree_sequence()
        G = ts.genotype_matrix()
        missing_found = False
        non_missing_found = False
        for var in ts.variants():
            if var.site.position == 4:
                assert var.has_missing_data
                assert var.genotypes[0] == 1
                assert np.all(var.genotypes[1:] == tskit.MISSING_DATA)
                missing_found += 1
            elif var.site.position == 5:
                assert not var.has_missing_data
                assert np.all(var.genotypes == 0)
                non_missing_found = 1
            else:
                assert not var.has_missing_data
                assert np.all(var.genotypes != tskit.MISSING_DATA)
            assert np.array_equal(var.genotypes, G[var.site.id])
        assert non_missing_found
        assert missing_found


class TestLimitInterval:
    def test_simple_case(self, ts_fixture):
        ts = ts_fixture
        test_variant = tskit.Variant(ts)
        test_variant.decode(1)
        for v in ts.variants(left=ts.site(1).position, right=ts.site(2).position):
            # should only decode the first variant
            assert v.site.id == 1
            assert np.all(v.genotypes == test_variant.genotypes)
            assert v.alleles == test_variant.alleles

    @pytest.mark.parametrize(
        ["left", "expected"],
        [
            (None, [0, 1, 2, 3, 4]),
            (0, [0, 1, 2, 3, 4]),
            (0.999, [1, 2, 3, 4]),
            (1, [1, 2, 3, 4]),
            (3.999, [4]),
            (4, [4]),
            (4.00001, []),
            (4.99999, []),
            (np.array([4.99999])[0], []),
        ],
    )
    def test_left(self, left, expected):
        tables = tskit.TableCollection(5)
        for x in range(int(tables.sequence_length)):
            tables.sites.add_row(position=x, ancestral_state="A")
        ts = tables.tree_sequence()
        positions = [var.site.position for var in ts.variants(left=left)]
        assert positions == expected

    @pytest.mark.parametrize(
        ["right", "expected"],
        [
            (None, [0, 1, 2, 3, 4]),
            (5, [0, 1, 2, 3, 4]),
            (4.00001, [0, 1, 2, 3, 4]),
            (4.0, [0, 1, 2, 3]),
            (3.9999, [0, 1, 2, 3]),
            (0.00001, [0]),
            (np.array([1e-200])[0], [0]),
        ],
    )
    def test_right(self, right, expected):
        tables = tskit.TableCollection(5)
        for x in range(int(tables.sequence_length)):
            tables.sites.add_row(position=x, ancestral_state="A")
        ts = tables.tree_sequence()
        positions = [var.site.position for var in ts.variants(right=right)]
        assert positions == expected

    @pytest.mark.parametrize("bad_left", [-1, 10, 100, np.nan, np.inf, -np.inf])
    def test_bad_left(self, bad_left):
        ts = tskit.TableCollection(10).tree_sequence()
        with pytest.raises(ValueError, match="`left` not between"):
            list(ts.variants(left=bad_left))

    @pytest.mark.parametrize("bad_right", [-1, 0, 100, np.nan, np.inf, -np.inf])
    def test_bad_right(self, bad_right):
        ts = tskit.TableCollection(10).tree_sequence()
        with pytest.raises(ValueError, match="`right` not between"):
            list(ts.variants(right=bad_right))

    def test_bad_left_right(self):
        ts = tskit.TableCollection(10).tree_sequence()
        with pytest.raises(ValueError, match="must be less than"):
            list(ts.variants(left=1, right=1))


class TestHaplotypeGenerator:
    """
    Tests the haplotype generation code.
    """

    def verify_haplotypes(self, n, haplotypes):
        """
        Verify that the specified set of haplotypes is consistent.
        """
        assert len(haplotypes) == n
        m = len(haplotypes[0])
        for h in haplotypes:
            assert len(h) == m
        # Examine each column in H; we must have a mixture of 0s and 1s
        for k in range(m):
            zeros = 0
            ones = 0
            col = ""
            for j in range(n):
                b = haplotypes[j][k]
                zeros += b == "0"
                ones += b == "1"
                col += b
            assert zeros + ones == n

    def verify_tree_sequence(self, tree_sequence):
        n = tree_sequence.sample_size
        m = tree_sequence.num_sites
        haplotypes = list(tree_sequence.haplotypes())
        A = np.zeros((n, m), dtype="u1")
        B = np.zeros((n, m), dtype="u1")
        for j, h in enumerate(haplotypes):
            assert len(h) == m
            A[j] = np.frombuffer(h.encode("ascii"), np.uint8) - ord("0")
        for variant in tree_sequence.variants():
            B[:, variant.index] = variant.genotypes
        assert np.all(A == B)
        self.verify_haplotypes(n, haplotypes)

    def verify_simulation(self, n, m, r, theta):
        """
        Verifies a simulation for the specified parameters.
        """
        recomb_map = msprime.RecombinationMap.uniform_map(m, r, m)
        tree_sequence = msprime.simulate(
            n, recombination_map=recomb_map, mutation_rate=theta
        )
        self.verify_tree_sequence(tree_sequence)

    def test_random_parameters(self):
        num_random_sims = 10
        for _ in range(num_random_sims):
            n = random.randint(2, 50)
            m = random.randint(10, 200)
            r = random.random()
            theta = random.uniform(0, 2)
            self.verify_simulation(n, m, r, theta)

    def test_nonbinary_trees(self):
        bottlenecks = [
            msprime.SimpleBottleneck(0.01, 0, proportion=0.05),
            msprime.SimpleBottleneck(0.02, 0, proportion=0.25),
            msprime.SimpleBottleneck(0.03, 0, proportion=1),
        ]
        ts = msprime.simulate(
            10,
            length=100,
            recombination_rate=1,
            demographic_events=bottlenecks,
            random_seed=1,
        )
        self.verify_tree_sequence(ts)

    def test_acgt_mutations(self):
        ts = msprime.simulate(10, mutation_rate=10)
        assert ts.num_sites > 0
        tables = ts.tables
        sites = tables.sites
        mutations = tables.mutations
        sites.set_columns(
            position=sites.position,
            ancestral_state=np.zeros(ts.num_sites, dtype=np.int8) + ord("A"),
            ancestral_state_offset=np.arange(ts.num_sites + 1, dtype=np.uint32),
        )
        mutations.set_columns(
            site=mutations.site,
            node=mutations.node,
            derived_state=np.zeros(ts.num_sites, dtype=np.int8) + ord("T"),
            derived_state_offset=np.arange(ts.num_sites + 1, dtype=np.uint32),
        )
        tsp = tables.tree_sequence()
        H = [h.replace("0", "A").replace("1", "T") for h in ts.haplotypes()]
        assert H == list(tsp.haplotypes())

    def test_fails_multiletter_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.tables
        tables.sites.add_row(0, "ACTG")
        tsp = tables.tree_sequence()
        with pytest.raises(TypeError):
            list(tsp.haplotypes())

    def test_fails_deletion_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.tables
        tables.sites.add_row(0, "")
        tsp = tables.tree_sequence()
        with pytest.raises(TypeError):
            list(tsp.haplotypes())

    def test_nonascii_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.tables
        tables.sites.add_row(0, chr(169))  # Copyright symbol
        tsp = tables.tree_sequence()
        with pytest.raises(TypeError):
            list(tsp.haplotypes())

    def test_recurrent_mutations_over_samples(self):
        ts = msprime.simulate(10, random_seed=2)
        num_sites = 5
        tables = ts.dump_tables()
        for j in range(num_sites):
            tables.sites.add_row(
                position=j * ts.sequence_length / num_sites, ancestral_state="0"
            )
            for u in range(ts.sample_size):
                tables.mutations.add_row(site=j, node=u, derived_state="1")
        ts_new = tables.tree_sequence()
        ones = "1" * num_sites
        for h in ts_new.haplotypes():
            assert ones == h

    def test_silent_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.dump_tables()
        tree = next(ts.trees())
        for u in tree.children(tree.root):
            tables.sites.clear()
            tables.mutations.clear()
            site = tables.sites.add_row(position=0, ancestral_state="0")
            tables.mutations.add_row(site=site, node=u, derived_state="1")
            tables.mutations.add_row(site=site, node=tree.root, derived_state="1")
            ts_new = tables.tree_sequence()
            all(h == 1 for h in ts_new.haplotypes())

    def test_back_mutations(self):
        base_ts = msprime.simulate(10, random_seed=2)
        for j in [1, 2, 3]:
            ts = tsutil.insert_branch_mutations(base_ts, mutations_per_branch=j)
            self.verify_tree_sequence(ts)

    def test_missing_data(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            list(ts.haplotypes(missing_data_character="A"))
        for c in ("-", ".", "a"):
            h = list(ts.haplotypes(missing_data_character=c))
            assert h == [c, c]
        h = list(ts.haplotypes(isolated_as_missing=True))
        assert h == ["N", "N"]
        h = list(ts.haplotypes(isolated_as_missing=False))
        assert h == ["A", "A"]
        h = list(ts.haplotypes())
        assert h == ["N", "N"]
        # Test deprecated method
        with pytest.warns(FutureWarning):
            h = list(ts.haplotypes(impute_missing_data=True))
        assert h == ["A", "A"]
        with pytest.warns(FutureWarning):
            h = list(ts.haplotypes(impute_missing_data=False))
        assert h == ["N", "N"]
        with pytest.warns(FutureWarning):
            h = list(ts.haplotypes(isolated_as_missing=True, impute_missing_data=True))
        assert h == ["N", "N"]
        with pytest.warns(FutureWarning):
            h = list(ts.haplotypes(isolated_as_missing=True, impute_missing_data=False))
        assert h == ["N", "N"]
        with pytest.warns(FutureWarning):
            h = list(ts.haplotypes(isolated_as_missing=False, impute_missing_data=True))
        assert h == ["A", "A"]
        with pytest.warns(FutureWarning):
            h = list(
                ts.haplotypes(isolated_as_missing=False, impute_missing_data=False)
            )
        assert h == ["A", "A"]

    def test_restrict_samples(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(0, 0, derived_state="B")
        ts = tables.tree_sequence()
        haplotypes = list(ts.haplotypes(samples=[0], isolated_as_missing=False))
        assert haplotypes == ["B"]
        haplotypes = list(ts.haplotypes(samples=[1], isolated_as_missing=False))
        assert haplotypes == ["A"]

    def test_restrict_positions(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.1, "A")
        tables.sites.add_row(0.2, "B")
        tables.sites.add_row(0.3, "C")
        tables.sites.add_row(0.4, "D")
        ts = tables.tree_sequence()
        haplotypes = list(ts.haplotypes(left=0.2, right=0.4, isolated_as_missing=False))
        assert haplotypes == ["BC"]


class TestUserAlleles:
    """
    Tests the functionality of providing a user-specified allele mapping.
    """

    def test_simple_01(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        assert ts.num_sites > 2
        G1 = ts.genotype_matrix()
        G2 = ts.genotype_matrix(alleles=("0", "1"))
        assert np.array_equal(G1, G2)
        for v1, v2 in itertools.zip_longest(
            ts.variants(), ts.variants(alleles=("0", "1"))
        ):
            assert v1.alleles == v2.alleles
            assert v1.site == v2.site
            assert np.array_equal(v1.genotypes, v2.genotypes)

    def test_simple_01_trailing_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        assert ts.num_sites > 2
        G1 = ts.genotype_matrix()
        alleles = ("0", "1", "2", "xxxxx")
        G2 = ts.genotype_matrix(alleles=alleles)
        assert np.array_equal(G1, G2)
        for v1, v2 in itertools.zip_longest(
            ts.variants(), ts.variants(alleles=alleles)
        ):
            assert v2.alleles == alleles
            assert v1.site == v2.site
            assert np.array_equal(v1.genotypes, v2.genotypes)

    def test_simple_01_leading_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        assert ts.num_sites > 2
        G1 = ts.genotype_matrix()
        alleles = ("A", "B", "C", "0", "1")
        G2 = ts.genotype_matrix(alleles=alleles)
        assert np.array_equal(G1 + 3, G2)
        for v1, v2 in itertools.zip_longest(
            ts.variants(), ts.variants(alleles=alleles)
        ):
            assert v2.alleles == alleles
            assert v1.site == v2.site
            assert np.array_equal(v1.genotypes + 3, v2.genotypes)

    def test_simple_01_duplicate_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        assert ts.num_sites > 2
        G1 = ts.genotype_matrix()
        alleles = ("0", "0", "1")
        G2 = ts.genotype_matrix(alleles=alleles)
        index = np.where(G1 == 1)
        G1[index] = 2
        assert np.array_equal(G1, G2)
        for v1, v2 in itertools.zip_longest(
            ts.variants(), ts.variants(alleles=alleles)
        ):
            assert v2.alleles == alleles
            assert v1.site == v2.site
            g = np.array(v1.genotypes)
            index = np.where(g == 1)
            g[index] = 2
            assert np.array_equal(g, v2.genotypes)

    def test_simple_acgt(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = msprime.mutate(
            ts, rate=4, random_seed=2, model=msprime.InfiniteSites(msprime.NUCLEOTIDES)
        )
        assert ts.num_sites > 2
        alleles = tskit.ALLELES_ACGT
        G = ts.genotype_matrix(alleles=alleles)
        for v1, v2 in itertools.zip_longest(
            ts.variants(), ts.variants(alleles=alleles)
        ):
            assert v2.alleles == alleles
            assert v1.site == v2.site
            h1 = "".join(v1.alleles[g] for g in v1.genotypes)
            h2 = "".join(v2.alleles[g] for g in v2.genotypes)
            assert h1 == h2
            assert np.array_equal(v2.genotypes, G[v1.site.id])

    def test_missing_alleles(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = msprime.mutate(
            ts, rate=4, random_seed=2, model=msprime.InfiniteSites(msprime.NUCLEOTIDES)
        )
        assert ts.num_sites > 2
        bad_allele_examples = [
            tskit.ALLELES_01,
            tuple(["A"]),
            ("C", "T", "G"),
            ("AA", "C", "T", "G"),
            tuple(["ACTG"]),
        ]
        for bad_alleles in bad_allele_examples:
            with pytest.raises(exceptions.LibraryError):
                ts.genotype_matrix(alleles=bad_alleles)
            with pytest.raises(exceptions.LibraryError):
                list(ts.variants(alleles=bad_alleles))

    def test_too_many_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        for n in range(128, 138):
            bad_alleles = tuple("0" for _ in range(n))
            with pytest.raises(exceptions.LibraryError):
                ts.genotype_matrix(alleles=bad_alleles)
            with pytest.raises(exceptions.LibraryError):
                list(ts.variants(alleles=bad_alleles))

    def test_zero_allele(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        with pytest.raises(ValueError):
            ts.genotype_matrix(alleles=tuple())
        with pytest.raises(ValueError):
            list(ts.variants(alleles=tuple()))

    def test_missing_data(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(0.5, "0")
        tables.mutations.add_row(0, 0, "1")

        ts = tables.tree_sequence()
        for isolated_as_missing in [True, False]:
            G1 = ts.genotype_matrix(isolated_as_missing=isolated_as_missing)
            G2 = ts.genotype_matrix(
                isolated_as_missing=isolated_as_missing, alleles=tskit.ALLELES_01
            )
            assert np.array_equal(G1, G2)
            vars1 = ts.variants(isolated_as_missing=isolated_as_missing)
            vars2 = ts.variants(
                isolated_as_missing=isolated_as_missing, alleles=tskit.ALLELES_01
            )
            for v1, v2 in itertools.zip_longest(vars1, vars2):
                assert v2.alleles == v1.alleles
                assert v1.site == v2.site
                assert np.array_equal(v1.genotypes, v2.genotypes)


class TestUserAllelesRoundTrip:
    """
    Tests that we correctly produce haplotypes in a variety of situations for
    the user specified allele map encoding.
    """

    def verify(self, ts, alleles):
        for v1, v2 in itertools.zip_longest(
            ts.variants(), ts.variants(alleles=alleles)
        ):
            h1 = [v1.alleles[g] for g in v1.genotypes]
            h2 = [v2.alleles[g] for g in v2.genotypes]
            assert h1 == h2

    def test_simple_01(self):
        ts = msprime.simulate(5, mutation_rate=2, random_seed=3)
        assert ts.num_sites > 3
        valid_alleles = [
            tskit.ALLELES_01,
            ("0", "1", "xry"),
            ("xry", "0", "1", "xry"),
            tuple(str(j) for j in range(127)),
            tuple(["0" for j in range(126)] + ["1"]),
        ]
        for alleles in valid_alleles:
            self.verify(ts, alleles)

    def test_simple_acgt(self):
        ts = msprime.simulate(5, random_seed=3)
        ts = msprime.mutate(
            ts, rate=4, random_seed=3, model=msprime.InfiniteSites(msprime.NUCLEOTIDES)
        )
        assert ts.num_sites > 3
        valid_alleles = [
            tskit.ALLELES_ACGT,
            ("A", "C", "T", "G", "AAAAAAAAAAAAAA"),
            ("AA", "CC", "TT", "GG", "A", "C", "T", "G"),
        ]
        for alleles in valid_alleles:
            self.verify(ts, alleles)

    def test_jukes_cantor(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.jukes_cantor(ts, 20, 1, seed=10)
        valid_alleles = [
            tskit.ALLELES_ACGT,
            ("A", "C", "T", "G", "AAAAAAAAAAAAAA"),
            ("AA", "CC", "TT", "GG", "A", "C", "T", "G"),
        ]
        for alleles in valid_alleles:
            self.verify(ts, alleles)

    def test_multichar_mutations(self):
        ts = msprime.simulate(6, random_seed=1, recombination_rate=2)
        ts = tsutil.insert_multichar_mutations(ts)
        assert ts.num_sites > 5
        all_alleles = set()
        for var in ts.variants():
            all_alleles.update(var.alleles)
        all_alleles = tuple(all_alleles)
        self.verify(ts, all_alleles)
        self.verify(ts, all_alleles[::-1])

    def test_simple_01_missing_data(self):
        ts = msprime.simulate(6, mutation_rate=2, random_seed=3)
        tables = ts.dump_tables()
        # Add another sample node. This will be missing data everywhere.
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()
        assert ts.num_sites > 3
        valid_alleles = [
            tskit.ALLELES_01,
            ("0", "1", "xry"),
            ("xry", "0", "1", "xry"),
            tuple(str(j) for j in range(127)),
            tuple(["0" for j in range(126)] + ["1"]),
        ]
        for alleles in valid_alleles:
            self.verify(ts, alleles)


class TestBinaryTreeExample:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0      10
    #      |    |
    #  pos 2    9
    #  anc A    T
    @tests.cached_example
    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(2, ancestral_state="A")
        tables.sites.add_row(9, ancestral_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="G")
        tables.mutations.add_row(site=1, node=3, derived_state="C")
        return tables.tree_sequence()

    def test_haplotypes(self):
        H = list(self.ts().haplotypes())
        assert H[0] == "GT"
        assert H[1] == "AC"
        assert H[2] == "AC"

    def test_haplotypes_empty_interval(self):
        ts = self.ts()
        H = list(ts.haplotypes(left=4, right=5))
        assert H == ["", "", ""]

    def test_genotypes(self):
        G = self.ts().genotype_matrix()
        Gp = [[1, 0, 0], [0, 1, 1]]
        np.testing.assert_array_equal(G, Gp)

    def test_alignments_default(self):
        A = list(self.ts().alignments())
        assert A[0] == "NNGNNNNNNT"
        assert A[1] == "NNANNNNNNC"
        assert A[2] == "NNANNNNNNC"

    def test_alignments_restricted(self):
        ts = self.ts()
        samples = ts.samples()
        # Take the first 2 in reverse order
        A = list(ts.alignments(left=1, right=9, samples=samples[1::-1]))
        assert A[0] == "NANNNNNN"
        assert A[1] == "NGNNNNNN"

    def test_empty_samples(self):
        ts = self.ts()
        A = list(ts.alignments(samples=[]))
        assert len(A) == 0

    def test_non_sample_samples(self):
        ts = self.ts()
        with pytest.raises(tskit.LibraryError, match="MUST_IMPUTE_NON_SAMPLES"):
            list(ts.alignments(samples=[4]))

    def test_alignments_missing_data_char(self):
        A = list(self.ts().alignments(missing_data_character="x"))
        assert A[0] == "xxGxxxxxxT"
        assert A[1] == "xxAxxxxxxC"
        assert A[2] == "xxAxxxxxxC"

    def test_alignments_reference_sequence(self):
        ref = "0123456789"
        A = list(self.ts().alignments(reference_sequence=ref))
        assert A[0] == "01G345678T"
        assert A[1] == "01A345678C"
        assert A[2] == "01A345678C"

    def test_alignments_reference_sequence_embedded_null(self):
        # This is a total corner case, but just want to make sure
        # we do something sensible.
        ref = "0123" + "\0" + "56789"
        A = list(self.ts().alignments(reference_sequence=ref))
        assert A[0] == "01G3\x005678T"
        assert A[1] == "01A3\x005678C"
        assert A[2] == "01A3\x005678C"

    def test_fasta_default(self):
        expected = textwrap.dedent(
            """\
            >n0
            NNGNNNNNNT
            >n1
            NNANNNNNNC
            >n2
            NNANNNNNNC
            """
        )
        assert expected == self.ts().as_fasta()

    def test_fasta_missing_Q(self):
        expected = textwrap.dedent(
            """\
            >n0
            QQGQQQQQQT
            >n1
            QQAQQQQQQC
            >n2
            QQAQQQQQQC
            """
        )
        assert expected == self.ts().as_fasta(missing_data_character="Q")

    def test_fasta_reference_sequence(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            >n0
            01G345678T
            >n1
            01A345678C
            >n2
            01A345678C
            """
        )
        assert expected == self.ts().as_fasta(reference_sequence=ref)

    def test_nexus_default(self):
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 ??G??????T
                n1 ??A??????C
                n2 ??A??????C
              ;
            END;
            BEGIN TREES;
              TREE t0^10 = [&R] (n0:2,(n1:1,n2:1):1);
            END;
            """
        )
        assert expected == self.ts().as_nexus()

    def test_nexus_missing_N(self):
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=N;
              MATRIX
                n0 NNGNNNNNNT
                n1 NNANNNNNNC
                n2 NNANNNNNNC
              ;
            END;
            BEGIN TREES;
              TREE t0^10 = [&R] (n0:2,(n1:1,n2:1):1);
            END;
            """
        )
        assert expected == self.ts().as_nexus(missing_data_character="N")

    def test_nexus_reference_sequence(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 01G345678T
                n1 01A345678C
                n2 01A345678C
              ;
            END;
            BEGIN TREES;
              TREE t0^10 = [&R] (n0:2,(n1:1,n2:1):1);
            END;
            """
        )
        assert expected == self.ts().as_nexus(reference_sequence=ref)


class TestBinaryTreeWithReferenceExample:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0      10
    #      |    |
    #  pos 2    9
    #  anc A    T
    @tests.cached_example
    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(2, ancestral_state="A")
        tables.sites.add_row(9, ancestral_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="G")
        tables.mutations.add_row(site=1, node=3, derived_state="C")
        tables.reference_sequence.data = "ACGTACGTAC"
        return tables.tree_sequence()

    def test_alignments_default(self):
        A = list(self.ts().alignments())
        assert A[0] == "ACGTACGTAT"
        assert A[1] == "ACATACGTAC"
        assert A[2] == "ACATACGTAC"

    def test_alignments_missing_data_char(self):
        A = list(self.ts().alignments(missing_data_character="x"))
        assert A[0] == "ACGTACGTAT"
        assert A[1] == "ACATACGTAC"
        assert A[2] == "ACATACGTAC"

    def test_alignments_reference_sequence(self):
        ref = "0123456789"
        A = list(self.ts().alignments(reference_sequence=ref))
        assert A[0] == "01G345678T"
        assert A[1] == "01A345678C"
        assert A[2] == "01A345678C"

    def test_fasta_default(self):
        expected = textwrap.dedent(
            """\
            >n0
            ACGTACGTAT
            >n1
            ACATACGTAC
            >n2
            ACATACGTAC
            """
        )
        assert expected == self.ts().as_fasta()

    def test_fasta_reference_sequence(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            >n0
            01G345678T
            >n1
            01A345678C
            >n2
            01A345678C
            """
        )
        assert expected == self.ts().as_fasta(reference_sequence=ref)

    def test_nexus_default(self):
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 ACGTACGTAT
                n1 ACATACGTAC
                n2 ACATACGTAC
              ;
            END;
            BEGIN TREES;
              TREE t0^10 = [&R] (n0:2,(n1:1,n2:1):1);
            END;
            """
        )
        assert expected == self.ts().as_nexus()

    def test_nexus_reference_sequence(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 01G345678T
                n1 01A345678C
                n2 01A345678C
              ;
            END;
            BEGIN TREES;
              TREE t0^10 = [&R] (n0:2,(n1:1,n2:1):1);
            END;
            """
        )
        assert expected == self.ts().as_nexus(reference_sequence=ref)


class TestMissingDataExample:
    # 2.00┊   4     ┊
    #     ┊ ┏━┻┓    ┊
    # 1.00┊ ┃  3    ┊
    #     ┊ ┃ ┏┻┓   ┊
    # 0.00┊ 0 1 2 5 ┊
    #     0        10
    #      |      |
    #  pos 2      9
    #  anc A      T
    @tests.cached_example
    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        tables = ts.dump_tables()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(2, ancestral_state="A")
        tables.sites.add_row(9, ancestral_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="G")
        tables.mutations.add_row(site=1, node=3, derived_state="C")
        return tables.tree_sequence()

    def test_haplotypes(self):
        H = list(self.ts().haplotypes())
        assert H[0] == "GT"
        assert H[1] == "AC"
        assert H[2] == "AC"
        assert H[3] == "NN"

    def test_haplotypes_missing_data_char(self):
        H = list(self.ts().haplotypes(missing_data_character="?"))
        assert H[0] == "GT"
        assert H[1] == "AC"
        assert H[2] == "AC"
        assert H[3] == "??"

    def test_genotypes(self):
        G = self.ts().genotype_matrix()
        Gp = [[1, 0, 0, -1], [0, 1, 1, -1]]
        np.testing.assert_array_equal(G, Gp)

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_alignments_default(self):
        A = list(self.ts().alignments())
        assert A[0] == "NNGNNNNNNT"
        assert A[1] == "NNANNNNNNC"
        assert A[2] == "NNANNNNNNC"
        assert A[3] == "NNNNNNNNNN"

    def test_alignments_fails(self):
        # https://github.com/tskit-dev/tskit/issues/1896
        with pytest.raises(ValueError, match="1896"):
            next(self.ts().alignments())

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_alignments_impute_missing(self):
        ref = "N" * 10
        A = list(
            self.ts().alignments(reference_sequence=ref, isolated_as_missing=False)
        )
        assert A[0] == "NNGNNNNNNT"
        assert A[1] == "NNANNNNNNC"
        assert A[2] == "NNANNNNNNC"
        assert A[3] == "NNANNNNNNT"

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_alignments_missing_char(self):
        A = list(self.ts().alignments(missing_data_character="z"))
        assert A[0] == "zzGzzzzzzT"
        assert A[1] == "zzAzzzzzzC"
        assert A[2] == "zzAzzzzzzC"
        assert A[3] == "zzzzzzzzzz"

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_alignments_missing_char_ref(self):
        A = list(self.ts().alignments(missing_data_character="z"))
        assert A[0] == "NNGNNNNNNT"
        assert A[1] == "NNANNNNNNC"
        assert A[2] == "NNANNNNNNC"
        assert A[3] == "zzzzzzzzzz"

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_alignments_reference_sequence(self):
        ref = "0123456789"
        A = list(self.ts().alignments(reference_sequence=ref))
        assert A[0] == "01G345678T"
        assert A[1] == "01A345678C"
        assert A[2] == "01A345678C"
        assert A[3] == "NNNNNNNNNN"

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_alignments_reference_sequence_missing_data_char(self):
        ref = "0123456789"
        A = list(
            self.ts().alignments(reference_sequence=ref, missing_data_character="Q")
        )
        assert A[0] == "01G345678T"
        assert A[1] == "01A345678C"
        assert A[2] == "01A345678C"
        assert A[3] == "QQQQQQQQQQ"

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_fasta_reference_sequence(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            >n0
            01G345678T
            >n1
            01A345678C
            >n2
            01A345678C
            >n5
            NNNNNNNNNN
            """
        )
        assert expected == self.ts().as_fasta(reference_sequence=ref)

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_fasta_reference_sequence_missing_data_char(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            >n0
            01G345678T
            >n1
            01A345678C
            >n2
            01A345678C
            >n5
            QQQQQQQQQQ
            """
        )
        assert expected == self.ts().as_fasta(
            reference_sequence=ref, missing_data_character="Q"
        )

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_fasta_impute_missing(self):
        ref = "N" * 10
        expected = textwrap.dedent(
            """\
            >n0
            NNGNNNNNNT
            >n1
            NNANNNNNNC
            >n2
            NNANNNNNNC
            >n5
            NNANNNNNNT
            """
        )
        assert expected == self.ts().as_fasta(
            reference_sequence=ref, isolated_as_missing=False
        )

    # Note: the nexus tree output isn't compatible with our representation of
    # missing data as trees with isolated roots (newick parsers won't accept
    # this as valid input), so we set include_trees=False for these examples.
    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_nexus_reference_sequence(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=4;
              TAXLABELS n0 n1 n2 n5;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 01G345678T
                n1 01A345678C
                n2 01A345678C
                n5 ??????????
              ;
            END;
            """
        )
        assert expected == self.ts().as_nexus(
            reference_sequence=ref, include_trees=False
        )

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_nexus_reference_sequence_missing_data_char(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=4;
              TAXLABELS n0 n1 n2 n5;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=Q;
              MATRIX
                n0 01G345678T
                n1 01A345678C
                n2 01A345678C
                n5 QQQQQQQQQQ
              ;
            END;
            """
        )
        assert expected == self.ts().as_nexus(
            reference_sequence=ref,
            missing_data_character="Q",
            include_trees=False,
        )

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_nexus_impute_missing(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=4;
              TAXLABELS n0 n1 n2 n5;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 01G345678T
                n1 01A345678C
                n2 01A345678C
                n5 01A345678T
              ;
            END;
            """
        )
        assert expected == self.ts().as_nexus(
            reference_sequence=ref,
            isolated_as_missing=False,
            include_trees=False,
        )


class TestMultiRootExample:
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0        10
    #       |     |
    #  pos  2     8
    #  anc  G     C
    @tests.cached_example
    def ts(self):
        tree = tskit.Tree.generate_balanced(4, arity=2, span=10)
        tables = tree.tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 6:
                tables.edges.append(edge)
        tables.sites.add_row(2, ancestral_state="G")
        tables.sites.add_row(8, ancestral_state="C")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=1, node=5, derived_state="A")
        return tables.tree_sequence()

    def test_haplotypes(self):
        H = list(self.ts().haplotypes())
        assert H[0] == "TC"
        assert H[1] == "GC"
        assert H[2] == "GA"
        assert H[3] == "GA"

    def test_genotypes(self):
        G = self.ts().genotype_matrix()
        Gp = [[1, 0, 0, 0], [0, 0, 1, 1]]
        np.testing.assert_array_equal(G, Gp)

    def test_alignments_default(self):
        A = list(self.ts().alignments())
        assert A[0] == "NNTNNNNNCN"
        assert A[1] == "NNGNNNNNCN"
        assert A[2] == "NNGNNNNNAN"
        assert A[3] == "NNGNNNNNAN"

    def test_alignments_N_ref(self):
        A = list(self.ts().alignments(reference_sequence="N" * 10))
        assert A[0] == "NNTNNNNNCN"
        assert A[1] == "NNGNNNNNCN"
        assert A[2] == "NNGNNNNNAN"
        assert A[3] == "NNGNNNNNAN"

    def test_fasta_reference_sequence(self):
        ref = "0123456789"
        expected = textwrap.dedent(
            """\
            >n0
            01T34567C9
            >n1
            01G34567C9
            >n2
            01G34567A9
            >n3
            01G34567A9
            """
        )
        assert expected == self.ts().as_fasta(reference_sequence=ref)


class TestAlignmentsErrors:
    @tests.cached_example
    def simplest_ts(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        return tables.tree_sequence()

    def test_non_discrete_genome(self):
        ts = tskit.TableCollection(1.1).tree_sequence()
        assert not ts.discrete_genome
        with pytest.raises(ValueError, match="defined for discrete genomes"):
            list(ts.alignments())

    @pytest.mark.parametrize("ref_length", [1, 9, 11])
    def test_reference_length_mismatch(self, ref_length):
        tables = tskit.TableCollection(10)
        tables.reference_sequence.data = "A" * ref_length
        ts = tables.tree_sequence()
        if ref_length <= tables.sequence_length:
            with pytest.raises(ValueError, match="shorter than"):
                list(ts.alignments())
        else:
            # Longer reference sequences are allowed
            list(ts.alignments())

    @pytest.mark.parametrize("ref", ["", "xy"])
    def test_reference_sequence_length_mismatch(self, ref):
        ts = self.simplest_ts()
        with pytest.raises(ValueError, match="shorter than"):
            list(ts.alignments(reference_sequence=ref))

    @pytest.mark.parametrize("ref", ["À", "┃", "α"])
    def test_non_ascii_references(self, ref):
        ts = self.simplest_ts()
        with pytest.raises(UnicodeEncodeError):
            list(ts.alignments(reference_sequence=ref))

    @pytest.mark.parametrize("ref", ["À", "┃", "α"])
    def test_non_ascii_embedded_references(self, ref):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.reference_sequence.data = ref
        ts = tables.tree_sequence()
        with pytest.raises(UnicodeEncodeError):
            list(ts.alignments())

    @pytest.mark.parametrize("missing_data_char", ["À", "┃", "α"])
    def test_non_ascii_missing_data_char(self, missing_data_char):
        ts = self.simplest_ts()
        with pytest.raises(UnicodeEncodeError):
            list(ts.alignments(missing_data_character=missing_data_char))

    def test_bad_left(self):
        ts = tskit.TableCollection(10).tree_sequence()
        with pytest.raises(ValueError, match="integer"):
            list(ts.alignments(left=0.1))

    def test_bad_right(self):
        ts = tskit.TableCollection(10).tree_sequence()
        with pytest.raises(ValueError, match="integer"):
            list(ts.alignments(right=1.1))

    def test_bad_restricted(self):
        tables = tskit.TableCollection(10)
        tables.reference_sequence.data = "A" * 7
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="sequence ends before"):
            list(ts.alignments(right=8))


class TestAlignmentExamples:
    @pytest.mark.parametrize("ts", get_example_discrete_genome_tree_sequences())
    def test_defaults(self, ts):
        has_missing_data = np.any(ts.genotype_matrix() == -1)
        if has_missing_data:
            with pytest.raises(ValueError, match="1896"):
                list(ts.alignments())
        else:
            A = list(ts.alignments())
            assert len(A) == ts.num_samples
            H = list(ts.haplotypes())
            pos = ts.tables.sites.position.astype(int)
            for a, h in map(np.array, zip(A, H)):
                last = 0
                for j, x in enumerate(pos):
                    assert a[last:x] == "N" * (x - last)
                    assert a[x] == h[j]
                    last = x + 1

    @pytest.mark.parametrize("ts", get_example_discrete_genome_tree_sequences())
    def test_reference_sequence(self, ts):
        ref = tskit.random_nucleotides(ts.sequence_length, seed=1234)
        has_missing_data = np.any(ts.genotype_matrix() == -1)
        if has_missing_data:
            with pytest.raises(ValueError, match="1896"):
                list(ts.alignments(reference_sequence=ref))
        else:
            A = list(ts.alignments(reference_sequence=ref))
            assert len(A) == ts.num_samples
            H = list(ts.haplotypes())
            pos = ts.tables.sites.position.astype(int)
            for a, h in map(np.array, zip(A, H)):
                last = 0
                for j, x in enumerate(pos):
                    assert a[last:x] == ref[last:x]
                    assert a[x] == h[j]
                    last = x + 1
                assert a[last:] == ref[last:]


#
# Tests for allele_remap
#
@pytest.mark.parametrize(
    "alleles_from, alleles_to, allele_map",
    [
        # Case 1: alleles_to is longer than alleles_from.
        (
            ["A", "C", "G", "T"],
            ["G", "C"],
            np.array([2, 1, 0, 3], dtype="uint32"),
        ),
        # Case 2: alleles_to is shorter than alleles_from.
        (
            ["G", "C"],
            ["A", "C", "G", "T"],
            np.array([2, 1], dtype="uint32"),
        ),
        # Case 3: alleles_to is empty.
        (
            ["A", "C", "G", "T"],
            [],
            np.array([0, 1, 2, 3], dtype="uint32"),
        ),
        # Case 4: alleles_from is empty.
        (
            [],
            ["A", "C", "G", "T"],
            np.array([], dtype="uint32"),
        ),
        # Case 5: Both lists are empty.
        (
            [],
            [],
            np.array([], dtype="uint32"),
        ),
        # Case 6: Both lists are tuples.
        (
            ("G", "C"),
            ("A", "C", "G", "T"),
            np.array([2, 1], dtype="uint32"),
        ),
        # Case 7: Both lists are numpy arrays.
        (
            np.array(("G", "C")),
            np.array(("A", "C", "G", "T")),
            np.array([2, 1], dtype="uint32"),
        ),
        # Case 8: Lists are of two different types.
        (
            np.array(("G", "C")),
            ["A", "C", "G", "T"],
            np.array([2, 1], dtype="uint32"),
        ),
        # Case 9: Lists contain elements of arbitrary types.
        (
            ["ABBA", "CDCD"],
            ["ABBA", "CDCD", "EFEF", "GG", 18],
            np.array([0, 1], dtype="uint32"),
        ),
        # Case 10: Lists contain unicode characters.
        (
            ["\u1F1E8", "\u1F1EC"],
            ["\u1F1EC", "\u1F1E8", "\u1F1E6", "\u1F1F3"],
            np.array([1, 0], dtype="uint32"),
        ),
    ],
)
def test_allele_remap(alleles_from, alleles_to, allele_map):
    assert np.array_equal(allele_map, allele_remap(alleles_from, alleles_to))


class TestVariant:
    # Much more in-depth testing of variant decoding is done via the ts.variants
    # method as it existed before this class was publicly creatable.
    def test_variant_init(self, ts_fixture):
        v = tskit.Variant(ts_fixture)
        assert np.array_equal(v.samples, np.array(ts_fixture.samples()))
        assert v.alleles == ()
        assert v.num_alleles == 0
        assert v.isolated_as_missing
        v = tskit.Variant(ts_fixture, samples=[43, 1])
        assert np.array_equal(v.samples, np.array([43, 1]))
        v = tskit.Variant(ts_fixture, alleles=("A", "💩"))
        assert v.alleles == ("A", "💩")
        v = tskit.Variant(ts_fixture, isolated_as_missing=False)
        assert not v.isolated_as_missing

    def test_not_decoded(self, ts_fixture):
        variant = tskit.Variant(ts_fixture)
        assert variant.index == tskit.NULL
        with pytest.raises(ValueError, match="not yet been decoded"):
            variant.site
        assert variant.alleles == ()
        with pytest.raises(ValueError, match="not yet been decoded"):
            assert variant.genotypes
        assert not variant.has_missing_data
        assert variant.num_alleles == 0
        with pytest.raises(ValueError, match="not yet been decoded"):
            variant.position
        assert np.array_equal(variant.samples, np.array(ts_fixture.samples()))

    def test_variant_decode(self, ts_fixture):
        v = tskit.Variant(ts_fixture)
        v.decode(2)
        assert v.index == 2
        assert np.array_equal(v.samples, np.array(ts_fixture.samples()))
        assert v.alleles == ("A", "T", "G", "C", None)
        # No need to check contents as done in other tests
        assert len(v.genotypes) == ts_fixture.num_samples

    def test_variant_num_missing(self, ts_fixture):
        variant = next(ts_fixture.variants())
        assert variant.num_missing > 0
        assert variant.num_missing == np.sum(variant.genotypes == -1)

    def test_variant_counts(self, ts_fixture):
        variant = next(ts_fixture.variants())
        assert len(variant.alleles) > 2
        assert None in variant.alleles
        counts = variant.counts()
        assert len(counts) == len(variant.alleles)
        assert np.sum(list(counts.values())) == ts_fixture.num_samples
        assert counts[None] == variant.num_missing
        assert ts_fixture.num_samples > variant.num_missing
        for i, v in enumerate(variant.alleles):
            if v is not None:
                assert np.sum(variant.genotypes == i) == counts[v]

    def test_variant_counts_empty(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.sites.add_row(0, ancestral_state="💩")
        ts = tables.tree_sequence()
        variant = next(ts.variants())
        assert len(variant.counts()) == 1
        assert variant.counts()["💩"] == 0

    def test_variant_simple_frequencies(self):
        simple_tree = tskit.Tree.generate_balanced(4)
        simple_ts = simple_tree.tree_sequence
        tables = simple_ts.dump_tables()
        tables.sites.add_row(position=0.3, ancestral_state="AS0")
        tables.sites.add_row(position=0.6, ancestral_state="AS1")
        tables.mutations.add_row(site=0, derived_state="DS0_0", node=0)
        tables.mutations.add_row(site=0, derived_state="DS0_3", node=3)
        tables.mutations.add_row(
            site=1, derived_state="DS1", node=simple_tree.parent(0)
        )
        ts = tables.tree_sequence()
        variant_0 = next(ts.variants())
        freqs = variant_0.frequencies()
        assert len(freqs) == 3
        assert np.allclose(freqs["AS0"], 0.5)
        assert np.allclose(freqs["DS0_0"], 0.25)
        assert np.allclose(freqs["DS0_3"], 0.25)
        variant_1 = next(ts.variants(left=0.5))
        freqs = variant_1.frequencies()
        assert len(freqs) == 2
        assert np.allclose(freqs["AS1"], 0.5)
        assert np.allclose(freqs["DS1"], 0.5)

    def test_variant_frequencies(self, ts_fixture):
        variant = next(ts_fixture.variants())
        assert variant.num_missing > 0
        freqs = variant.frequencies()
        assert len(freqs) == len(variant.alleles)
        assert np.allclose(np.sum(list(freqs.values())), 1)
        for i, v in enumerate(variant.alleles):
            if v is None:
                f = np.sum(variant.genotypes == tskit.NULL) / ts_fixture.num_samples
            else:
                f = np.sum(variant.genotypes == i) / ts_fixture.num_samples
            assert np.allclose(f, freqs[v])

        freqs = variant.frequencies(remove_missing=True)
        assert len(freqs) == len(variant.alleles) - 1
        for i, v in enumerate(variant.alleles[:-1]):
            f = np.sum(variant.genotypes == i) / (
                ts_fixture.num_samples - variant.num_missing
            )
            assert np.allclose(f, freqs[v])

    def test_variant_frequencies_limit_samples(self, ts_fixture):
        assert ts_fixture.num_samples > 1
        variant = next(ts_fixture.variants(samples=ts_fixture.samples()[0:1]))
        assert len(variant.genotypes) == 1
        allele = variant.alleles[variant.genotypes[0]]
        freqs = variant.frequencies()
        assert freqs[allele] == 1
        # should be one freq of 1 and all the rest zero
        assert list(freqs.values()).count(0) == len(freqs) - 1

    def test_variant_nonsample_freqs(self):
        simple_tree = tskit.Tree.generate_balanced(4)
        nonsample_node_left = simple_tree.parent(0)
        nonsample_node_right = simple_tree.parent(3)
        assert nonsample_node_left != nonsample_node_right
        simple_ts = simple_tree.tree_sequence
        tables = simple_ts.dump_tables()
        tables.sites.add_row(position=0, ancestral_state="As")
        tables.mutations.add_row(site=0, derived_state="Ds", node=nonsample_node_left)
        ts = tables.tree_sequence()
        samples = [nonsample_node_left, nonsample_node_right]
        samples += list(simple_tree.children(nonsample_node_right))
        variant = next(ts.variants(samples=samples, isolated_as_missing=False))
        freqs = variant.frequencies()
        assert np.allclose(freqs["Ds"], 0.25)  # Just nonsample_node_left has the Ds
        assert np.allclose(freqs["As"], 0.75)

    def test_variant_frequencies_no_samples(self, ts_fixture, caplog):
        tables = ts_fixture.dump_tables()
        tables.nodes.flags = np.zeros_like(tables.nodes.flags)
        ts = tables.tree_sequence()
        variant = next(ts.variants())
        assert ts.num_samples == 0
        with caplog.at_level(logging.WARNING):
            freqs = variant.frequencies()
            assert caplog.text.count("frequencies undefined") == 1
        assert np.all(np.isnan(list(freqs.values())))

    def test_variant_str(self):
        """
        Test using a simple dummy tree sequence for testing.
        It has only one tree and one site, whose variant has the alleles
        ('A', 'T', 'G', '💩', '', 'TAG', None).
        """
        tables = tskit.TableCollection(10)
        for _ in np.arange(6):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=5, ancestral_state="A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=0, node=1, derived_state="G")
        tables.mutations.add_row(site=0, node=2, derived_state="💩")
        tables.mutations.add_row(site=0, node=3, derived_state="")
        tables.mutations.add_row(site=0, node=4, derived_state="TAG")
        ts = tables.tree_sequence()
        v = next(ts.variants())
        assert v.alleles == ("A", "T", "G", "💩", "", "TAG", None)
        assert isinstance(str(v), str)
        assert re.match(
            textwrap.dedent(
                r"""
                ╔═+╗
                ║Variant\s*║
                ╠═+╤═+╣
                ║Site id\s*│\s*0║
                ╟─+┼─+╢
                ║Site position\s*│\s*[0-9\.]+║
                ╟─+┼─+╢
                ║Number of samples\s*│\s*[0-9]+║
                ╟─+┼─+╢
                ║Number of alleles\s*│\s*[0-9]+║
                ╟─+┼─+╢
                ║Samples with allele \'A\'\s*│\s*[0-9]+\s*\([0-9\.]+\%\)║
                ╟─+┼─+╢
                ║Samples with allele \'T\'\s*│\s*[0-9]+\s*\([0-9\.]+\%\)║
                ╟─+┼─+╢
                ║Samples with allele \'G\'\s*│\s*[0-9]+\s*\([0-9\.]+\%\)║
                ╟─+┼─+╢
                ║Samples with allele \'💩\'\s*│\s*[0-9]+\s*\([0-9\.]+\%\)║
                ╟─+┼─+╢
                ║Samples with allele \'\'\s*│\s*[0-9]+\s*\([0-9\.]+\%\)║
                ╟─+┼─+╢
                ║Samples with allele \'TAG\'\s*│\s*[0-9]+\s*\([0-9\.]+\%\)║
                ╟─+┼─+╢
                ║Samples with allele missing\s*│\s*[0-9]+\s*\([0-9\.]+\%\)║
                ╟─+┼─+╢
                ║Has missing data\s*│\s*True║
                ╟─+┼─+╢
                ║Isolated as missing\s*│\s*True║
                ╚═+╧═+╝
                """[
                    1:
                ]
            ),
            str(v),
        )

    def test_variant_str_no_samples(self):
        tables = tskit.TableCollection(10)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=5, ancestral_state="A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        ts = tables.tree_sequence()
        v = next(ts.variants(samples=[]))
        for allele in v.alleles:
            if allele is not None:
                assert (
                    re.search(
                        rf"║Samples with allele '{allele}'\s*│\s*0\s*\(nan\%\)║", str(v)
                    )
                    is not None
                )

    def test_variant_str_no_site(self):
        tables = tskit.TableCollection(10)
        ts = tables.tree_sequence()
        v = tskit.Variant(ts)
        s = str(v)
        assert len(s.splitlines()) == 5
        assert (
            "This variant has not yet been decoded at a specific site, "
            + "call Variant.decode to set the site"
            in s
        )

    def test_variant_html_repr(self, ts_fixture):
        v = next(ts_fixture.variants())
        html = v._repr_html_()
        # Parse to check valid
        ElementTree.fromstring(html)
        assert len(html) > 1900

    def test_variant_html_repr_no_site(self):
        tables = tskit.TableCollection(10)
        ts = tables.tree_sequence()
        v = tskit.Variant(ts)
        html = v._repr_html_()
        ElementTree.fromstring(html)
        assert len(html) > 1600

    def test_variant_repr(self, ts_fixture):
        v = next(ts_fixture.variants())
        str_rep = repr(v)
        assert len(str_rep) > 0 and len(str_rep) < 10000
        assert re.search(r"\AVariant", str_rep)
        assert re.search(rf"\'site\': Site\(id={v.site.id}", str_rep)
        assert re.search(rf"position={v.position}", str_rep)
        alleles = re.escape("'alleles': " + str(v.alleles))
        assert re.search(rf"{alleles}", str_rep)
        assert re.search(r"\'genotypes\': array\(\[", str_rep)
        assert re.search(rf"position={v.position}", str_rep)
        assert re.search(rf"\'has_missing_data\': {v.has_missing_data}", str_rep)
        assert re.search(rf"\'isolated_as_missing\': {v.isolated_as_missing}", str_rep)
