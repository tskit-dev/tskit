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
Test cases for generating genotypes/haplotypes.
"""
import unittest
import itertools
import random

import numpy as np
import msprime

import tskit

from tskit import exceptions
import tests.tsutil as tsutil
import tests.test_wright_fisher as wf


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


class TestGetAncestralHaplotypes(unittest.TestCase):
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
        self.assertTrue(np.array_equal(A, B))

    def test_single_tree(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=234)
        self.verify(ts)

    def test_many_trees(self):
        ts = msprime.simulate(
            8, recombination_rate=10, mutation_rate=10, random_seed=234)
        self.assertGreater(ts.num_trees, 1)
        self.assertGreater(ts.num_sites, 1)
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
        self.assertGreater(ts.num_sites, 0)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts)

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            6, 5, seed=3, deep_history=True, initial_generation_samples=True,
            num_loci=2)
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.08, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify(ts)

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            9, 10, seed=1, deep_history=True, initial_generation_samples=False,
            num_loci=5)
        tables.sort()
        ts = tables.tree_sequence().simplify()
        ts = msprime.mutate(ts, rate=0.01, random_seed=1234)
        self.assertGreater(ts.num_sites, 0)
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


class TestVariantGenerator(unittest.TestCase):
    """
    Tests the variants() method to ensure the output is consistent.
    """
    def get_tree_sequence(self):
        ts = msprime.simulate(
            10, length=10, recombination_rate=1, mutation_rate=10, random_seed=3)
        self.assertGreater(ts.get_num_mutations(), 10)
        return ts

    def test_as_bytes(self):
        ts = self.get_tree_sequence()
        n = ts.get_sample_size()
        m = ts.get_num_mutations()
        A = np.zeros((m, n), dtype='u1')
        B = np.zeros((m, n), dtype='u1')
        for variant in ts.variants():
            A[variant.index] = variant.genotypes
        for variant in ts.variants(as_bytes=True):
            self.assertIsInstance(variant.genotypes, bytes)
            B[variant.index] = np.fromstring(variant.genotypes, np.uint8) - ord('0')
        self.assertTrue(np.all(A == B))
        bytes_variants = list(ts.variants(as_bytes=True))
        for j, variant in enumerate(bytes_variants):
            self.assertEqual(j, variant.index)
            row = np.fromstring(variant.genotypes, np.uint8) - ord('0')
            self.assertTrue(np.all(A[j] == row))

    def test_as_bytes_fails(self):
        ts = tsutil.insert_multichar_mutations(self.get_tree_sequence())
        self.assertRaises(ValueError, list, ts.variants(as_bytes=True))

    def test_dtype(self):
        ts = self.get_tree_sequence()
        for var in ts.variants():
            self.assertEqual(var.genotypes.dtype, np.int8)

    def test_dtype_conversion(self):
        # Check if we hit any issues if we assume the variants are uint8
        # as they were prior to version 0.2.0
        ts = self.get_tree_sequence()
        G = ts.genotype_matrix().astype(np.uint8)
        self.assertEqual(G.dtype, np.uint8)
        for var in ts.variants():
            self.assertTrue(np.array_equal(G[var.index], var.genotypes))
            self.assertTrue(np.all(G[var.index] == var.genotypes))
            self.assertEqual(
                [var.alleles[g] for g in var.genotypes],
                [var.alleles[g] for g in G[var.index]])
            G[var.index, :] = var.genotypes
            self.assertTrue(np.array_equal(G[var.index], var.genotypes))

    def test_multichar_alleles(self):
        ts = tsutil.insert_multichar_mutations(self.get_tree_sequence())
        for var in ts.variants():
            self.assertEqual(len(var.alleles), 2)
            self.assertEqual(var.site.ancestral_state, var.alleles[0])
            self.assertEqual(var.site.mutations[0].derived_state, var.alleles[1])
            self.assertTrue(all(0 <= var.genotypes))
            self.assertTrue(all(var.genotypes <= 1))

    def test_many_alleles(self):
        ts = self.get_tree_sequence()
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        # This gives us a total of 360 permutations.
        alleles = list(map("".join, itertools.permutations('ABCDEF', 4)))
        self.assertGreater(len(alleles), 127)
        tables.sites.add_row(0, alleles[0])
        parent = -1
        num_alleles = 1
        for allele in alleles[1:]:
            ts = tables.tree_sequence()
            if num_alleles > 127:
                self.assertRaises(exceptions.LibraryError, next, ts.variants())
            else:
                var = next(ts.variants())
                self.assertFalse(var.has_missing_data)
                self.assertEqual(var.num_alleles, num_alleles)
                self.assertEqual(len(var.alleles), num_alleles)
                self.assertEqual(list(var.alleles), alleles[:num_alleles])
                self.assertEqual(var.alleles[var.genotypes[0]], alleles[num_alleles - 1])
                for u in ts.samples():
                    if u != 0:
                        self.assertEqual(var.alleles[var.genotypes[u]], alleles[0])
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
        alleles = list(map("".join, itertools.permutations('ABCDEF', 4)))
        self.assertGreater(len(alleles), 127)
        tables.sites.add_row(0, alleles[0])
        parent = -1
        num_alleles = 1
        for allele in alleles[1:]:
            ts = tables.tree_sequence()
            if num_alleles > 127:
                self.assertRaises(exceptions.LibraryError, next, ts.variants())
            else:
                var = next(ts.variants())
                self.assertTrue(var.has_missing_data)
                self.assertEqual(var.num_alleles, num_alleles)
                self.assertEqual(len(var.alleles), num_alleles + 1)
                self.assertEqual(list(var.alleles)[:-1], alleles[:num_alleles])
                self.assertIsNone(var.alleles[-1])
                self.assertEqual(var.alleles[var.genotypes[0]], alleles[num_alleles - 1])
                self.assertEqual(var.genotypes[-1], -1)
                samples = ts.samples()
                for u in samples[:-1]:
                    if u != 0:
                        self.assertEqual(var.alleles[var.genotypes[u]], alleles[0])
            tables.mutations.add_row(0, 0, allele, parent=parent)
            parent += 1
            num_alleles += 1

    def test_site_information(self):
        ts = self.get_tree_sequence()
        for site, variant in zip(ts.sites(), ts.variants()):
            self.assertEqual(site.position, variant.position)
            self.assertEqual(site, variant.site)

    def test_no_mutations(self):
        ts = msprime.simulate(10)
        self.assertEqual(ts.get_num_mutations(), 0)
        variants = list(ts.variants())
        self.assertEqual(len(variants), 0)

    def test_genotype_matrix(self):
        ts = self.get_tree_sequence()
        G = np.empty((ts.num_sites, ts.num_samples), dtype=np.int8)
        for v in ts.variants():
            G[v.index, :] = v.genotypes
        G2 = ts.genotype_matrix()
        self.assertTrue(np.array_equal(G, G2))
        self.assertEqual(G2.dtype, np.int8)

    def test_recurrent_mutations_over_samples(self):
        ts = self.get_tree_sequence()
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        num_sites = 5
        for j in range(num_sites):
            tables.sites.add_row(
                position=j * ts.sequence_length / num_sites,
                ancestral_state="0")
            for u in range(ts.sample_size):
                tables.mutations.add_row(site=j, node=u, derived_state="1")
        ts = tables.tree_sequence()
        variants = list(ts.variants())
        self.assertEqual(len(variants), num_sites)
        for site, variant in zip(ts.sites(), variants):
            self.assertEqual(site.position, variant.position)
            self.assertEqual(site, variant.site)
            self.assertEqual(site.id, variant.index)
            self.assertEqual(variant.alleles, ("0", "1"))
            self.assertTrue(np.all(variant.genotypes == np.ones(ts.sample_size)))

    def test_recurrent_mutations_errors(self):
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
                    self.assertRaises(exceptions.LibraryError, list, ts_new.variants())

    def test_zero_samples(self):
        ts = self.get_tree_sequence()
        for var1, var2 in zip(ts.variants(), ts.variants(samples=[])):
            self.assertEqual(var1.site, var2.site)
            self.assertEqual(var1.alleles, var2.alleles)
            self.assertEqual(var2.genotypes.shape[0], 0)

    def test_samples(self):
        n = 4
        ts = msprime.simulate(
            n, length=5, recombination_rate=1, mutation_rate=5, random_seed=2)
        self.assertGreater(ts.num_sites, 1)
        samples = list(range(n))
        # Generate all possible sample lists.
        for j in range(n + 1):
            for s in itertools.permutations(samples, j):
                s = np.array(s, dtype=np.int32)
                count = 0
                for var1, var2 in zip(ts.variants(), ts.variants(samples=s)):
                    self.assertEqual(var1.site, var2.site)
                    self.assertEqual(var1.alleles, var2.alleles)
                    self.assertEqual(var2.genotypes.shape, (len(s),))
                    self.assertTrue(np.array_equal(var1.genotypes[s], var2.genotypes))
                    count += 1
                self.assertEqual(count, ts.num_sites)

    def test_non_sample_samples(self):
        # We don't have to use sample nodes. This does make the terminology confusing
        # but it's probably still the best option.
        ts = msprime.simulate(
            10, length=5, recombination_rate=1, mutation_rate=5, random_seed=2)
        tables = ts.dump_tables()
        tables.nodes.set_columns(
            flags=np.zeros_like(tables.nodes.flags) + tskit.NODE_IS_SAMPLE,
            time=tables.nodes.time)
        all_samples_ts = tables.tree_sequence()
        self.assertEqual(all_samples_ts.num_samples, ts.num_nodes)

        count = 0
        samples = range(ts.num_nodes)
        for var1, var2 in zip(
                all_samples_ts.variants(impute_missing_data=True),
                ts.variants(samples=samples, impute_missing_data=True)):
            self.assertEqual(var1.site, var2.site)
            self.assertEqual(var1.alleles, var2.alleles)
            self.assertEqual(var2.genotypes.shape, (len(samples),))
            self.assertTrue(np.array_equal(var1.genotypes, var2.genotypes))
            count += 1
        self.assertEqual(count, ts.num_sites)

    def verify_jukes_cantor(self, ts):
        self.assertTrue(np.array_equal(ts.genotype_matrix(), ts.genotype_matrix()))
        tree = ts.first()
        for variant in ts.variants():
            self.assertFalse(variant.has_missing_data)
            mutations = {
                mutation.node: mutation.derived_state
                for mutation in variant.site.mutations}
            for sample_index, u in enumerate(ts.samples()):
                while u not in mutations and u != tskit.NULL:
                    u = tree.parent(u)
                state1 = mutations.get(u, variant.site.ancestral_state)
                state2 = variant.alleles[variant.genotypes[sample_index]]
                self.assertEqual(state1, state2)

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
        G = ts.genotype_matrix(impute_missing_data=True)
        self.assertTrue(np.all(G[0] == 0))
        self.assertTrue(np.all(G[1] == 0))
        self.assertTrue(np.all(G[-1] == 0))
        self.assertTrue(np.all(G[-2] == 0))
        G = isolated_samples_genotype_matrix(ts)
        self.assertTrue(np.all(G[0] == -1))
        self.assertTrue(np.all(G[1] == -1))
        self.assertTrue(np.all(G[-1] == -1))
        self.assertTrue(np.all(G[-2] == -1))
        Gp = ts.genotype_matrix(impute_missing_data=False)
        self.assertTrue(np.array_equal(G, Gp))

    def test_empty_ts_missing_data(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        ts = tables.tree_sequence()
        variants = list(ts.variants())
        self.assertEqual(len(variants), 1)
        var = variants[0]
        self.assertEqual(var.alleles, ("A", None))
        self.assertEqual(var.num_alleles, 1)
        self.assertTrue(np.all(var.genotypes == -1))

    def test_mutation_over_isolated_sample_missing(self):
        # This is a somewhat pathological case: isolated samples are still missing
        # *even if* there is a mutation directly over them, and this mutation will
        # be listed in the alleles.
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(0, 0, "T")
        ts = tables.tree_sequence()
        variants = list(ts.variants())
        self.assertEqual(len(variants), 1)
        var = variants[0]
        self.assertEqual(var.alleles, ("A", "T", None))
        self.assertEqual(var.num_alleles, 2)
        self.assertTrue(np.all(var.genotypes == -1))


class TestHaplotypeGenerator(unittest.TestCase):
    """
    Tests the haplotype generation code.
    """

    def verify_haplotypes(self, n, haplotypes):
        """
        Verify that the specified set of haplotypes is consistent.
        """
        self.assertEqual(len(haplotypes), n)
        m = len(haplotypes[0])
        for h in haplotypes:
            self.assertEqual(len(h), m)
        # Examine each column in H; we must have a mixture of 0s and 1s
        for k in range(m):
            zeros = 0
            ones = 0
            col = ""
            for j in range(n):
                b = haplotypes[j][k]
                zeros += b == '0'
                ones += b == '1'
                col += b
            self.assertEqual(zeros + ones, n)

    def verify_tree_sequence(self, tree_sequence):
        n = tree_sequence.sample_size
        m = tree_sequence.num_sites
        haplotypes = list(tree_sequence.haplotypes())
        A = np.zeros((n, m), dtype='u1')
        B = np.zeros((n, m), dtype='u1')
        for j, h in enumerate(haplotypes):
            self.assertEqual(len(h), m)
            A[j] = np.fromstring(h, np.uint8) - ord('0')
        for variant in tree_sequence.variants():
            B[:, variant.index] = variant.genotypes
        self.assertTrue(np.all(A == B))
        self.verify_haplotypes(n, haplotypes)

    def verify_simulation(self, n, m, r, theta):
        """
        Verifies a simulation for the specified parameters.
        """
        recomb_map = msprime.RecombinationMap.uniform_map(m, r, m)
        tree_sequence = msprime.simulate(
            n, recombination_map=recomb_map, mutation_rate=theta)
        self.verify_tree_sequence(tree_sequence)

    def test_random_parameters(self):
        num_random_sims = 10
        for j in range(num_random_sims):
            n = random.randint(2, 50)
            m = random.randint(10, 200)
            r = random.random()
            theta = random.uniform(0, 2)
            self.verify_simulation(n, m, r, theta)

    def test_nonbinary_trees(self):
        bottlenecks = [
            msprime.SimpleBottleneck(0.01, 0, proportion=0.05),
            msprime.SimpleBottleneck(0.02, 0, proportion=0.25),
            msprime.SimpleBottleneck(0.03, 0, proportion=1)]
        ts = msprime.simulate(
            10, length=100, recombination_rate=1,
            demographic_events=bottlenecks, random_seed=1)
        self.verify_tree_sequence(ts)

    def test_acgt_mutations(self):
        ts = msprime.simulate(10, mutation_rate=10)
        self.assertGreater(ts.num_sites, 0)
        tables = ts.tables
        sites = tables.sites
        mutations = tables.mutations
        sites.set_columns(
            position=sites.position,
            ancestral_state=np.zeros(ts.num_sites, dtype=np.int8) + ord("A"),
            ancestral_state_offset=np.arange(ts.num_sites + 1, dtype=np.uint32))
        mutations.set_columns(
            site=mutations.site,
            node=mutations.node,
            derived_state=np.zeros(ts.num_sites, dtype=np.int8) + ord("T"),
            derived_state_offset=np.arange(ts.num_sites + 1, dtype=np.uint32))
        tsp = tables.tree_sequence()
        H = [h.replace("0", "A").replace("1", "T") for h in ts.haplotypes()]
        self.assertEqual(H, list(tsp.haplotypes()))

    def test_fails_multiletter_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.tables
        tables.sites.add_row(0, "ACTG")
        tsp = tables.tree_sequence()
        self.assertRaises(TypeError, list, tsp.haplotypes())

    def test_fails_deletion_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.tables
        tables.sites.add_row(0, "")
        tsp = tables.tree_sequence()
        self.assertRaises(TypeError, list, tsp.haplotypes())

    def test_nonascii_mutations(self):
        ts = msprime.simulate(10, random_seed=2)
        tables = ts.tables
        tables.sites.add_row(0, chr(169))  # Copyright symbol
        tsp = tables.tree_sequence()
        self.assertRaises(TypeError, list, tsp.haplotypes())

    def test_recurrent_mutations_over_samples(self):
        ts = msprime.simulate(10, random_seed=2)
        num_sites = 5
        tables = ts.dump_tables()
        for j in range(num_sites):
            tables.sites.add_row(
                position=j * ts.sequence_length / num_sites,
                ancestral_state="0")
            for u in range(ts.sample_size):
                tables.mutations.add_row(site=j, node=u, derived_state="1")
        ts_new = tables.tree_sequence()
        ones = "1" * num_sites
        for h in ts_new.haplotypes():
            self.assertEqual(ones, h)

    def test_recurrent_mutations_errors(self):
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
            self.assertRaises(exceptions.LibraryError, list, ts_new.haplotypes())
            ts_new.haplotypes()

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
        self.assertRaises(ValueError, list, ts.haplotypes(missing_data_character="A"))
        for c in ("-", ".", "a"):
            h = list(ts.haplotypes(missing_data_character=c))
            self.assertEqual(h, [c, c])
        h = list(ts.haplotypes(impute_missing_data=True))
        self.assertEqual(h, ["A", "A"])


class TestUserAlleles(unittest.TestCase):
    """
    Tests the functionality of providing a user-specified allele mapping.
    """

    def test_simple_01(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        self.assertGreater(ts.num_sites, 2)
        G1 = ts.genotype_matrix()
        G2 = ts.genotype_matrix(alleles=("0", "1"))
        self.assertTrue(np.array_equal(G1, G2))
        for v1, v2 in itertools.zip_longest(
                ts.variants(), ts.variants(alleles=("0", "1"))):
            self.assertEqual(v1.alleles, v2.alleles)
            self.assertEqual(v1.site, v2.site)
            self.assertTrue(np.array_equal(v1.genotypes, v2.genotypes))

    def test_simple_01_trailing_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        self.assertGreater(ts.num_sites, 2)
        G1 = ts.genotype_matrix()
        alleles = ("0", "1", "2", "xxxxx")
        G2 = ts.genotype_matrix(alleles=alleles)
        self.assertTrue(np.array_equal(G1, G2))
        for v1, v2 in itertools.zip_longest(
                ts.variants(), ts.variants(alleles=alleles)):
            self.assertEqual(v2.alleles, alleles)
            self.assertEqual(v1.site, v2.site)
            self.assertTrue(np.array_equal(v1.genotypes, v2.genotypes))

    def test_simple_01_leading_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        self.assertGreater(ts.num_sites, 2)
        G1 = ts.genotype_matrix()
        alleles = ("A", "B", "C", "0", "1")
        G2 = ts.genotype_matrix(alleles=alleles)
        self.assertTrue(np.array_equal(G1 + 3, G2))
        for v1, v2 in itertools.zip_longest(
                ts.variants(), ts.variants(alleles=alleles)):
            self.assertEqual(v2.alleles, alleles)
            self.assertEqual(v1.site, v2.site)
            self.assertTrue(np.array_equal(v1.genotypes + 3, v2.genotypes))

    def test_simple_01_duplicate_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        self.assertGreater(ts.num_sites, 2)
        G1 = ts.genotype_matrix()
        alleles = ("0", "0", "1")
        G2 = ts.genotype_matrix(alleles=alleles)
        index = np.where(G1 == 1)
        G1[index] = 2
        self.assertTrue(np.array_equal(G1, G2))
        for v1, v2 in itertools.zip_longest(
                ts.variants(), ts.variants(alleles=alleles)):
            self.assertEqual(v2.alleles, alleles)
            self.assertEqual(v1.site, v2.site)
            g = v1.genotypes
            index = np.where(g == 1)
            g[index] = 2
            self.assertTrue(np.array_equal(g, v2.genotypes))

    def test_simple_acgt(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = msprime.mutate(
            ts, rate=4, random_seed=2, model=msprime.InfiniteSites(msprime.NUCLEOTIDES))
        self.assertGreater(ts.num_sites, 2)
        alleles = tskit.ALLELES_ACGT
        G = ts.genotype_matrix(alleles=alleles)
        for v1, v2 in itertools.zip_longest(ts.variants(), ts.variants(alleles=alleles)):
            self.assertEqual(v2.alleles, alleles)
            self.assertEqual(v1.site, v2.site)
            h1 = "".join(v1.alleles[g] for g in v1.genotypes)
            h2 = "".join(v2.alleles[g] for g in v2.genotypes)
            self.assertEqual(h1, h2)
            self.assertTrue(np.array_equal(v2.genotypes, G[v1.site.id]))

    def test_missing_alleles(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = msprime.mutate(
            ts, rate=4, random_seed=2, model=msprime.InfiniteSites(msprime.NUCLEOTIDES))
        self.assertGreater(ts.num_sites, 2)
        bad_allele_examples = [
                tskit.ALLELES_01, tuple(["A"]), ("C", "T", "G"), ("AA", "C", "T", "G"),
                tuple(["ACTG"])]
        for bad_alleles in bad_allele_examples:
            with self.assertRaises(exceptions.LibraryError):
                ts.genotype_matrix(alleles=bad_alleles)
            with self.assertRaises(exceptions.LibraryError):
                list(ts.variants(alleles=bad_alleles))

    def test_too_many_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        for n in range(128, 138):
            bad_alleles = tuple(["0" for _ in range(n)])
            with self.assertRaises(exceptions.LibraryError):
                ts.genotype_matrix(alleles=bad_alleles)
            with self.assertRaises(exceptions.LibraryError):
                list(ts.variants(alleles=bad_alleles))

    def test_zero_allele(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=2)
        with self.assertRaises(ValueError):
            ts.genotype_matrix(alleles=tuple())
        with self.assertRaises(ValueError):
            list(ts.variants(alleles=tuple()))

    def test_missing_data(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(0.5, "0")
        tables.mutations.add_row(0, 0, "1")

        ts = tables.tree_sequence()
        for impute in [True, False]:
            G1 = ts.genotype_matrix(impute_missing_data=impute)
            G2 = ts.genotype_matrix(
                impute_missing_data=impute, alleles=tskit.ALLELES_01)
            self.assertTrue(np.array_equal(G1, G2))
            vars1 = ts.variants(impute_missing_data=impute)
            vars2 = ts.variants(impute_missing_data=impute, alleles=tskit.ALLELES_01)
            for v1, v2 in itertools.zip_longest(vars1, vars2):
                self.assertEqual(v2.alleles, v1.alleles)
                self.assertEqual(v1.site, v2.site)
                self.assertTrue(np.array_equal(v1.genotypes, v2.genotypes))


class TestUserAllelesRoundTrip(unittest.TestCase):
    """
    Tests that we correctly produce haplotypes in a variety of situations for
    the user specified allele map encoding.
    """
    def verify(self, ts, alleles):
        for v1, v2 in itertools.zip_longest(ts.variants(), ts.variants(alleles=alleles)):
            h1 = [v1.alleles[g] for g in v1.genotypes]
            h2 = [v2.alleles[g] for g in v2.genotypes]
            self.assertEqual(h1, h2)

    def test_simple_01(self):
        ts = msprime.simulate(5, mutation_rate=2, random_seed=3)
        self.assertGreater(ts.num_sites, 3)
        valid_alleles = [
            tskit.ALLELES_01,
            ("0", "1", "xry"),
            ("xry", "0", "1", "xry"),
            tuple([str(j) for j in range(127)]),
            tuple(["0" for j in range(126)] + ["1"]),
        ]
        for alleles in valid_alleles:
            self.verify(ts, alleles)

    def test_simple_acgt(self):
        ts = msprime.simulate(5, random_seed=3)
        ts = msprime.mutate(
            ts, rate=4, random_seed=3, model=msprime.InfiniteSites(msprime.NUCLEOTIDES))
        self.assertGreater(ts.num_sites, 3)
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
        self.assertGreater(ts.num_sites, 5)
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
        self.assertGreater(ts.num_sites, 3)
        valid_alleles = [
            tskit.ALLELES_01,
            ("0", "1", "xry"),
            ("xry", "0", "1", "xry"),
            tuple([str(j) for j in range(127)]),
            tuple(["0" for j in range(126)] + ["1"]),
        ]
        for alleles in valid_alleles:
            self.verify(ts, alleles)
