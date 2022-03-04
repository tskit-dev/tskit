# Simulation
import itertools

# Python libraries
import msprime
import numpy as np
import pytest
import tskit
import lshmm as ls

import fb_haploid_variants_samples_tree as fbht

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2

class LSBase:
    """Superclass of Li and Stephens tests."""

    def example_haplotypes(self, ts):

        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0])
        H = H[:, 1:]

        return H, s

    def haplotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 2))
        e[:, 0] = mu
        e[:, 1] = 1 - mu

        return e

    def genotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 8))
        e[:, EQUAL_BOTH_HOM] = (1 - mu) ** 2
        e[:, UNEQUAL_BOTH_HOM] = mu ** 2
        e[:, BOTH_HET] = 1 - mu
        e[:, REF_HOM_OBS_HET] = 2 * mu * (1 - mu)
        e[:, REF_HET_OBS_HOM] = mu * (1 - mu)

        return e

    def example_parameters_haplotypes(self, ts, seed=42):
        """Returns an iterator over combinations of haplotype, recombination and mutation rates."""
        np.random.seed(seed)
        H, s = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.haplotype_emission(mu, m)

        yield n, m, H, s, e, r

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]

        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        e = self.haplotype_emission(mu, m)

        for r, mu in itertools.product(rs, mus):
            r[0] = 0
            yield n, m, H, s, e, r

    def example_parameters_haplotypes_larger(
        self, ts, seed=42, mean_r=1e-5, mean_mu=1e-5
    ):

        np.random.seed(seed)
        H, s = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        r = mean_r * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)
        r[0] = 0

        # Error probability
        mu = mean_mu * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)

        # Define the emission probability matrix
        e = self.haplotype_emission(mu, m)

        return n, m, H, s, e, r

    def example_genotypes(self, ts):

        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0]) + H[:, 1].reshape(1, H.shape[0])
        H = H[:, 2:]

        m = ts.get_num_sites()
        n = H.shape[1]

        G = np.zeros((m, n, n))
        for i in range(m):
            G[i, :, :] = np.add.outer(H[i, :], H[i, :])

        return H, G, s

    def example_parameters_genotypes(self, ts, seed=42):
        np.random.seed(seed)
        H, G, s = self.example_genotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.genotype_emission(mu, m)

        yield n, m, G, s, e, r

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]

        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        e = self.genotype_emission(mu, m)

        for r, mu in itertools.product(rs, mus):
            r[0] = 0
            yield n, m, G, s, e, r

    def example_parameters_genotypes_larger(
        self, ts, seed=42, mean_r=1e-5, mean_mu=1e-5
    ):

        np.random.seed(seed)
        H, G, s = self.example_genotypes(ts)

        m = ts.get_num_sites()
        n = H.shape[1]

        r = mean_r * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)
        r[0] = 0

        # Error probability
        mu = mean_mu * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)

        # Define the emission probability matrix
        e = self.genotype_emission(mu, m)

        return n, m, G, s, e, r

    def assertAllClose(self, A, B):
        """Assert that all entries of two matrices are 'close'"""
        assert np.allclose(A, B, rtol=1e-9, atol=0.0)

    # Define a bunch of very small tree-sequences for testing a collection of parameters on
    def test_simple_n_10_no_recombination(self):
        ts = msprime.simulate(
            10, recombination_rate=0, mutation_rate=0.5, random_seed=42
        )
        assert ts.num_sites > 3
        self.verify(ts)

    def test_simple_n_6(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=7, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8(self):
        ts = msprime.simulate(8, recombination_rate=2, mutation_rate=5, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8_high_recombination(self):
        ts = msprime.simulate(8, recombination_rate=20, mutation_rate=5, random_seed=42)
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_16(self):
        ts = msprime.simulate(16, recombination_rate=2, mutation_rate=5, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    # Test a bigger one.
    def test_large(self, n=50, length=100000, mean_r=1e-5, mean_mu=1e-5, seed=42):
        ts = msprime.simulate(
            n + 1,
            length=length,
            mutation_rate=mean_mu,
            recombination_rate=mean_r,
            random_seed=seed,
        )
        self.verify_larger(ts)

    def verify(self, ts):
        raise NotImplementedError()

    def verify_larger(self, ts):
        pass

class FBAlgorithmBase(LSBase):
    """Base for forwards backwards algorithm tests."""


class TestForwardTree(FBAlgorithmBase):
    """Tests that the tree algorithm computes the same forward matrix as the simple method."""

    def verify(self, ts):
        for n, m, H, s, e, r in self.example_parameters_haplotypes(ts):
            mu = e[:,0]
            F, c, ll = ls.forwards(n, m, H, s, e, r)
            # Note, need to remove the first sample from the ts, and ensure that invariant sites aren't removed.
            ts_check = ts.simplify(range(1,n+1), filter_sites=False)
            cm = fbht.ls_forward_tree(s[0,:], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))
            self.assertAllClose(ll, ll_tree)
            self.assertAllClose(F, cm.decode())

    def verify_larger(self, ts):
        n, m, H, s, e, r = self.example_parameters_haplotypes_larger(ts)
        mu = e[:,0]
        F, c, ll = ls.forwards(n, m, H, s, e, r)
        # Note, need to remove the first sample from the ts, and ensure that invariant sites aren't removed.
        ts_check = ts.simplify(range(1,n+1), filter_sites=False)
        cm = fbht.ls_forward_tree(s[0,:], ts_check, r, mu)
        ll_tree = np.sum(np.log10(cm.normalisation_factor))
        self.assertAllClose(ll, ll_tree)
        self.assertAllClose(F, cm.decode())


class TestMirroring(FBAlgorithmBase):
    """Tests that mirroring the tree sequence and running forwards and backwards algorithms gives
    the same log-likelihood of observing the data."""

    def verify(self, ts):
        for n, m, H, s, e, r in self.example_parameters_haplotypes(ts):
            mu = e[:,0]
            F, c, ll = ls.forwards(n, m, H, s, e, r)
            # Note, need to remove the first sample from the ts, and ensure that invariant sites aren't removed.
            ts_check = ts.simplify(range(1,n+1), filter_sites=False)
            cm = fbht.ls_forward_tree(s[0,:], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))
            ts_check_mirror = fbht.mirror_coordinates(ts_check)
            r_flip = np.insert(np.flip(r)[:-1], 0, 0)
            cm = fbht.ls_forward_tree(np.flip(s[0,:]), ts_check_mirror, r_flip, np.flip(mu))
            ll_mirror_tree = np.sum(np.log10(cm.normalisation_factor))
            self.assertAllClose(ll_tree, ll_mirror_tree)

            # Ensure that the decoded matrices are the same
            F_mirror_matrix, c, ll = ls.forwards(
                n, m, np.flip(H, axis=0), np.flip(s, axis=1), np.flip(e, axis=0), r_flip)
            F_mirror = cm.decode()

            self.assertAllClose(F_mirror_matrix, F_mirror)


class TestForwardBackwardTree(FBAlgorithmBase):
    """Tests that the tree algorithm computes the same forward matrix as the simple method."""

    def verify(self, ts):
        for n, m, H, s, e, r in self.example_parameters_haplotypes(ts):
            mu = e[:,0]
            F, c, ll = ls.forwards(n, m, H, s, e, r)
            B = ls.backwards(n, m, H, s, e, c, r)
            # Note, need to remove the first sample from the ts, and ensure that invariant sites aren't removed.
            ts_check = ts.simplify(range(1,n+1), filter_sites=False)
            cm = fbht.ls_forward_tree(s[0,:], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))

            # Ensure that the backwards matrices agree
            ts_check_mirror = fbht.mirror_coordinates(ts_check)
            r_flip = np.flip(r)
            cm = fbht.ls_backward_tree(np.flip(s[0,:]), ts_check_mirror, r_flip, np.flip(mu), np.flip(cm.normalisation_factor))
            B_tree = np.flip(cm.decode(), axis=0)

            self.assertAllClose(B, B_tree)
