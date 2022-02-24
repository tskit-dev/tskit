# Simulation
import itertools

# Python libraries
import msprime
import numpy as np
import pytest

import fb_diploid_samples_variants as fbd_sv
import fb_diploid_variants_samples as fbd_vs
import fb_haploid_samples_variants as fbh_sv
import fb_haploid_variants_samples as fbh_vs

import vit_diploid_samples_variants as vd_sv
import diploid_variants_samples as vd_vs
import vit_haploid_samples_variants as vh_sv
import vit_haploid_variants_samples as vh_vs

import fb_haploid_variants_samples_tree as fbh_vst

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2

import tskit

def ls_forward_matrix(h, G, r, mu):
    """
    Simple matrix based method for LS forward algorithm using numpy vectorisation.
    """
    assert r[0] == 0
    m, n = G.shape
    F = np.zeros((m, n))
    S = np.zeros(m)
    f = np.zeros(n) + 1 / n
    p_e = np.zeros(n)

    for el in range(0, m):
        p_t = f * (1 - r[el]) + r[el] / n
        eq = G[el] == h[0, el]
        # if h[el] == tskit.MISSING_DATA:
        #     # Missing data is equal to everything
        #     eq[:] = True
        p_e[:] = mu[el]
        p_e[eq] = 1 - mu[el]
        f = p_t * p_e
        S[el] = np.sum(f)
        # TODO need to handle the 0 case.
        assert S[el] > 0
        f /= S[el]
        F[el] = f
    return F, S


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

@pytest.mark.skip(reason="DEV: skip for time being")
class TestNonTreeMethodsHap(FBAlgorithmBase):
    """Test that we compute the sample likelihoods across all implementations."""
    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):
            e_sv = e_vs.T
            H_sv = H_vs.T

            # variants x samples
            F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=False
            )
            B_vs = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.log10(np.sum(F_vs * B_vs, 1)), ll_vs * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh_vs.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=True
            )
            B_tmp = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 1), np.ones(m))

            # samples x variants
            F_sv, c_sv, ll_sv = fbh_sv.forwards_ls_hap(
                n, m, H_sv, s, e_sv, r, norm=False
            )
            B_sv = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_sv, r)
            self.assertAllClose(np.log10(np.sum(F_sv * B_sv, 0)), ll_sv * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh_sv.forwards_ls_hap(
                n, m, H_sv, s, e_sv, r, norm=True
            )
            B_tmp = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 0), np.ones(m))

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)

    def verify_larger(self, ts):
        # variants x samples
        n, m, H_vs, s, e_vs, r = self.example_parameters_haplotypes_larger(ts)

        e_sv = e_vs.T
        H_sv = H_vs.T

        F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(n, m, H_vs, s, e_vs, r, norm=False)
        B_vs = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)
        self.assertAllClose(np.log10(np.sum(F_vs * B_vs, 1)), ll_vs * np.ones(m))
        F_tmp, c_tmp, ll_tmp = fbh_vs.forwards_ls_hap(n, m, H_vs, s, e_vs, r, norm=True)
        B_tmp = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_tmp, r)
        self.assertAllClose(ll_vs, ll_tmp)
        self.assertAllClose(np.sum(F_tmp * B_tmp, 1), np.ones(m))

        # samples x variants
        F_sv, c_sv, ll_sv = fbh_sv.forwards_ls_hap(n, m, H_sv, s, e_sv, r, norm=False)
        B_sv = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_sv, r)
        self.assertAllClose(np.log10(np.sum(F_sv * B_sv, 0)), ll_sv * np.ones(m))
        F_tmp, c_tmp, ll_tmp = fbh_sv.forwards_ls_hap(n, m, H_sv, s, e_sv, r, norm=True)
        B_tmp = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_tmp, r)
        self.assertAllClose(ll_sv, ll_tmp)
        self.assertAllClose(np.sum(F_tmp * B_tmp, 0), np.ones(m))

        # samples x variants agrees with variants x samples
        self.assertAllClose(ll_vs, ll_sv)

class TestNonTreeMethodsDip(FBAlgorithmBase):
    """Test that we compute the sample likelihoods across all implementations."""
    def verify(self, ts):
        for n, m, G_vs, s, e_vs, r in self.example_parameters_genotypes(ts):
            e_sv = e_vs.T
            G_sv = G_vs.T

            # variants x samples
            F_vs, c_vs, ll_vs = fbd_vs.forwards_ls_dip(n, m, G_vs, s, e_vs, r, norm=True)
            B_vs = fbd_vs.backwards_ls_dip(n, m, G_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.sum(F_vs * B_vs, (1, 2)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_vs.forwards_ls_dip(n, m, G_vs, s, e_vs, r, norm=False)
            B_tmp = fbd_vs.backwards_ls_dip(n, m, G_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m))

            F_tmp, ll_tmp = fbd_vs.forward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
            B_tmp = fbd_vs.backward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_vs.forward_ls_dip_loop(n, m, G_vs, s, e_vs, r, norm=False)
            B_tmp = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_vs.forward_ls_dip_loop(n, m, G_vs, s, e_vs, r, norm=True)
            B_tmp = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, (1, 2)), np.ones(m))

            # samples x variants

            F_sv, c_sv, ll_sv = fbd_sv.forwards_ls_dip(n, m, G_sv, s, e_sv, r, norm=True)
            B_sv = fbd_sv.backwards_ls_dip(n, m, G_sv, s, e_sv, c_sv, r)
            self.assertAllClose(np.sum(F_sv * B_sv, (0, 1)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_sv.forwards_ls_dip(n, m, G_sv, s, e_sv, r, norm=False)
            B_tmp = fbd_sv.backwards_ls_dip(n, m, G_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m))
            self.assertAllClose(ll_sv, ll_tmp)

            F_tmp, ll_tmp = fbd_sv.forward_ls_dip_starting_point(n, m, G_sv, s, e_sv, r)
            B_tmp = fbd_sv.backward_ls_dip_starting_point(n, m, G_sv, s, e_sv, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m))

            F_tmp, c_tmp, ll_tmp = fbd_sv.forward_ls_dip_loop(n, m, G_sv, s, e_sv, r, norm=True)
            B_tmp = fbd_sv.backward_ls_dip_loop(n, m, G_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, (0, 1)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_sv.forward_ls_dip_loop(n, m, G_sv, s, e_sv, r, norm=False)
            B_tmp = fbd_sv.backward_ls_dip_loop(n, m, G_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m))

            # compare sample x variants to variants x samples
            self.assertAllClose(ll_vs, ll_sv)

    def verify_larger(self, ts):
        # variants x samples
        n, m, G_vs, s, e_vs, r = self.example_parameters_genotypes_larger(ts)
        
        e_sv = e_vs.T
        G_sv = G_vs.T

        # variants x samples
        F_vs, c_vs, ll_vs = fbd_vs.forwards_ls_dip(n, m, G_vs, s, e_vs, r, norm=True)
        B_vs = fbd_vs.backwards_ls_dip(n, m, G_vs, s, e_vs, c_vs, r)
        self.assertAllClose(np.sum(F_vs * B_vs, (1, 2)), np.ones(m))
        F_tmp, c_tmp, ll_tmp = fbd_vs.forwards_ls_dip(n, m, G_vs, s, e_vs, r, norm=False)
        B_tmp = fbd_vs.backwards_ls_dip(n, m, G_vs, s, e_vs, c_tmp, r)
        self.assertAllClose(ll_vs, ll_tmp)
        self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m))

        F_tmp, ll_tmp = fbd_vs.forward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
        B_tmp = fbd_vs.backward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
        self.assertAllClose(ll_vs, ll_tmp)
        self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m))
        F_tmp, c_tmp, ll_tmp = fbd_vs.forward_ls_dip_loop(n, m, G_vs, s, e_vs, r, norm=False)
        B_tmp = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
        self.assertAllClose(ll_vs, ll_tmp)
        self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m))
        F_tmp, c_tmp, ll_tmp = fbd_vs.forward_ls_dip_loop(n, m, G_vs, s, e_vs, r, norm=True)
        B_tmp = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
        self.assertAllClose(ll_vs, ll_tmp)
        self.assertAllClose(np.sum(F_tmp * B_tmp, (1, 2)), np.ones(m))

        # samples x variants

        F_sv, c_sv, ll_sv = fbd_sv.forwards_ls_dip(n, m, G_sv, s, e_sv, r, norm=True)
        B_sv = fbd_sv.backwards_ls_dip(n, m, G_sv, s, e_sv, c_sv, r)
        self.assertAllClose(np.sum(F_sv * B_sv, (0, 1)), np.ones(m))
        F_tmp, c_tmp, ll_tmp = fbd_sv.forwards_ls_dip(n, m, G_sv, s, e_sv, r, norm=False)
        B_tmp = fbd_sv.backwards_ls_dip(n, m, G_sv, s, e_sv, c_tmp, r)
        self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m))
        self.assertAllClose(ll_sv, ll_tmp)

        F_tmp, ll_tmp = fbd_sv.forward_ls_dip_starting_point(n, m, G_sv, s, e_sv, r)
        B_tmp = fbd_sv.backward_ls_dip_starting_point(n, m, G_sv, s, e_sv, r)
        self.assertAllClose(ll_sv, ll_tmp)
        self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m))

        F_tmp, c_tmp, ll_tmp = fbd_sv.forward_ls_dip_loop(n, m, G_sv, s, e_sv, r, norm=True)
        B_tmp = fbd_sv.backward_ls_dip_loop(n, m, G_sv, s, e_sv, c_tmp, r)
        self.assertAllClose(ll_sv, ll_tmp)
        self.assertAllClose(np.sum(F_tmp * B_tmp, (0, 1)), np.ones(m))
        F_tmp, c_tmp, ll_tmp = fbd_sv.forward_ls_dip_loop(n, m, G_sv, s, e_sv, r, norm=False)
        B_tmp = fbd_sv.backward_ls_dip_loop(n, m, G_sv, s, e_sv, c_tmp, r)
        self.assertAllClose(ll_sv, ll_tmp)
        self.assertAllClose(np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m))

        # compare sample x variants to variants x samples
        self.assertAllClose(ll_vs, ll_sv)


class VitAlgorithmBase(LSBase):
    """Base for viterbi algoritm tests."""

class TestNonTreeViterbiHap(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):
            e_sv = e_vs.T
            H_sv = H_vs.T

            # variants x samples
            V_vs, P_vs, ll_vs = vh_vs.forwards_viterbi_hap_naive(n, m, H_vs, s, e_vs, r)
            path_vs = vh_vs.backwards_viterbi_hap(m, V_vs[m-1, :], P_vs)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, ll_check)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_vec(n, m, H_vs, s, e_vs, r)
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp[m-1, :], P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_low_mem(n, m, H_vs, s, e_vs, r)
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H_vs, s, e_vs, r)
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_low_mem_rescaling(n, m, H_vs, s, e_vs, r)
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_lower_mem_rescaling(n, m, H_vs, s, e_vs, r)
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants
            V_sv, P_sv, ll_sv = vh_sv.forwards_viterbi_hap_naive(n, m, H_sv, s, e_sv, r)
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_sv[:, m-1], P_sv)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_sv, ll_check)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_vec(n, m, H_sv, s, e_sv, r)
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp[:, m-1], P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_low_mem(n, m, H_sv, s, e_sv, r)
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H_sv, s, e_sv, r)
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_low_mem_rescaling(n, m, H_sv, s, e_sv, r)
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_lower_mem_rescaling(n, m, H_sv, s, e_sv, r)
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)

    def verify_larger(self, ts):
        n, m, H_vs, s, e_vs, r = self.example_parameters_haplotypes_larger(ts)
        e_sv = e_vs.T
        H_sv = H_vs.T

        # variants x samples
        V_vs, P_vs, ll_vs = vh_vs.forwards_viterbi_hap_naive(n, m, H_vs, s, e_vs, r)
        path_vs = vh_vs.backwards_viterbi_hap(m, V_vs[m-1, :], P_vs)
        ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_vs, s, e_vs, r)
        self.assertAllClose(ll_vs, ll_check)
        V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_vec(n, m, H_vs, s, e_vs, r)
        path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp[m-1, :], P_tmp)
        ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_vs, ll_tmp)
        V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_low_mem(n, m, H_vs, s, e_vs, r)
        path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
        ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_vs, ll_tmp)
        V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H_vs, s, e_vs, r)
        path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
        ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_vs, ll_tmp)
        V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_low_mem_rescaling(n, m, H_vs, s, e_vs, r)
        path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
        ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_vs, ll_tmp)
        V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_lower_mem_rescaling(n, m, H_vs, s, e_vs, r)
        path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
        ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_vs, ll_tmp)

        # samples x variants
        V_sv, P_sv, ll_sv = vh_sv.forwards_viterbi_hap_naive(n, m, H_sv, s, e_sv, r)
        path_tmp = vh_sv.backwards_viterbi_hap(m, V_sv[:, m-1], P_sv)
        ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
        self.assertAllClose(ll_sv, ll_check)
        V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_vec(n, m, H_sv, s, e_sv, r)
        path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp[:, m-1], P_tmp)
        ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_sv, ll_tmp)
        V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_low_mem(n, m, H_sv, s, e_sv, r)
        path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
        ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_sv, ll_tmp)
        V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H_sv, s, e_sv, r)
        path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
        ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_sv, ll_tmp)
        V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_low_mem_rescaling(n, m, H_sv, s, e_sv, r)
        path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
        ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_sv, ll_tmp)
        V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_lower_mem_rescaling(n, m, H_sv, s, e_sv, r)
        path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
        ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
        self.assertAllClose(ll_tmp, ll_check)
        self.assertAllClose(ll_vs, ll_tmp)

        # samples x variants agrees with variants x samples
        self.assertAllClose(ll_vs, ll_sv)

class TestNonTreeViterbiDip(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, G_vs, s, e_vs, r in self.example_parameters_genotypes(ts):
            e_sv = e_vs.T
            G_sv = G_vs.T

            # variants x samples
            V_vs, P_vs, ll_vs = vd_vs.forwards_viterbi_dip_naive(n, m, G_vs, s, e_vs, r)
            path_vs = vd_vs.backwards_viterbi_dip(m, V_vs[m-1, :, :], P_vs)
            phased_path_vs = vd_vs.get_phased_path(n, path_vs)
            path_ll_vs = vd_vs.path_ll_dip(n, m, G_vs, phased_path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, path_ll_vs)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_naive_low_mem(n, m, G_vs, s, e_vs, r)
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_low_mem(n, m, G_vs, s, e_vs, r)
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_naive_vec(n, m, G_vs, s, e_vs, r)
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp[m-1, :, :], P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants
            V_sv, P_sv, ll_sv = vd_sv.forwards_viterbi_dip_naive(n, m, G_sv, s, e_sv, r)
            path_sv = vd_sv.backwards_viterbi_dip(m, V_sv[:, :, m-1], P_sv)
            phased_path_sv = vd_sv.get_phased_path(n, path_sv)
            path_ll_sv = vd_sv.path_ll_dip(n, m, G_sv, phased_path_sv, s, e_sv, r)
            self.assertAllClose(ll_sv, path_ll_sv)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_naive_low_mem(n, m, G_sv, s, e_sv, r)
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_low_mem(n, m, G_sv, s, e_sv, r)
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_naive_vec(n, m, G_sv, s, e_sv, r)
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp[:, :, m-1], P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)

    def verify_larger(self, ts):
        n, m, G_vs, s, e_vs, r = self.example_parameters_genotypes_larger(ts)
        e_sv = e_vs.T
        G_sv = G_vs.T

        # variants x samples
        V_vs, P_vs, ll_vs = vd_vs.forwards_viterbi_dip_naive(n, m, G_vs, s, e_vs, r)
        path_vs = vd_vs.backwards_viterbi_dip(m, V_vs[m-1, :, :], P_vs)
        phased_path_vs = vd_vs.get_phased_path(n, path_vs)
        path_ll_vs = vd_vs.path_ll_dip(n, m, G_vs, phased_path_vs, s, e_vs, r)
        self.assertAllClose(ll_vs, path_ll_vs)

        V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_naive_low_mem(n, m, G_vs, s, e_vs, r)
        path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp, P_tmp)
        phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
        path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
        self.assertAllClose(ll_tmp, path_ll_tmp)
        self.assertAllClose(ll_vs, ll_tmp)

        V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_low_mem(n, m, G_vs, s, e_vs, r)
        path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp, P_tmp)
        phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
        path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
        self.assertAllClose(ll_tmp, path_ll_tmp)
        self.assertAllClose(ll_vs, ll_tmp)

        V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_naive_vec(n, m, G_vs, s, e_vs, r)
        path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp[m-1, :, :], P_tmp)
        phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
        path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
        self.assertAllClose(ll_tmp, path_ll_tmp)
        self.assertAllClose(ll_vs, ll_tmp)

        # samples x variants
        V_sv, P_sv, ll_sv = vd_sv.forwards_viterbi_dip_naive(n, m, G_sv, s, e_sv, r)
        path_sv = vd_sv.backwards_viterbi_dip(m, V_sv[:, :, m-1], P_sv)
        phased_path_sv = vd_sv.get_phased_path(n, path_sv)
        path_ll_sv = vd_sv.path_ll_dip(n, m, G_sv, phased_path_sv, s, e_sv, r)
        self.assertAllClose(ll_sv, path_ll_sv)

        V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_naive_low_mem(n, m, G_sv, s, e_sv, r)
        path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp, P_tmp)
        phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
        path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
        self.assertAllClose(ll_tmp, path_ll_tmp)
        self.assertAllClose(ll_sv, ll_tmp)

        V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_low_mem(n, m, G_sv, s, e_sv, r)
        path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp, P_tmp)
        phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
        path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
        self.assertAllClose(ll_tmp, path_ll_tmp)
        self.assertAllClose(ll_sv, ll_tmp)

        V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_naive_vec(n, m, G_sv, s, e_sv, r)
        path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp[:, :, m-1], P_tmp)
        phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
        path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
        self.assertAllClose(ll_tmp, path_ll_tmp)
        self.assertAllClose(ll_sv, ll_tmp)

        # samples x variants agrees with variants x samples
        self.assertAllClose(ll_vs, ll_sv)


class TestForwardTree(FBAlgorithmBase):
    """Tests that the tree algorithm computes the same forward matrix as the simple method."""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):
            mu = e_vs[:,0]
            F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(n, m, H_vs, s, e_vs, r, norm=True)
            # Note, need to remove the first sample from the ts, and ensure that invariant sites aren't removed.
            ts_check = ts.simplify(range(1,n+1), filter_sites=False)
            cm = fbh_vst.ls_forward_tree(s[0,:], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))
            self.assertAllClose(ll_vs, ll_tree)

class TestMirroring(FBAlgorithmBase):
    """Tests that mirroring the tree sequence and running forwards and backwards algorithms gives
    the same log-likelihood of observing the data."""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):
            mu = e_vs[:,0]
            F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(n, m, H_vs, s, e_vs, r, norm=True)
            ts_check = ts.simplify(range(1,n+1), filter_sites=False)
            cm = fbh_vst.ls_forward_tree(s[0,:], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))
            ts_check_mirror = fbh_vst.mirror_coordinates(ts_check)
            r_flip = np.insert(np.flip(r)[:-1], 0, 0)
            cm = fbh_vst.ls_forward_tree(np.flip(s[0,:]), ts_check_mirror, r_flip, np.flip(mu))
            ll_mirror_tree = np.sum(np.log10(cm.normalisation_factor))

            self.assertAllClose(ll_tree, ll_mirror_tree)

            # Ensure that the decoded matrices are the same
            F_vs_mirror_matrix, c_vs, ll_vs = fbh_vs.forwards_ls_hap(
                n, m, np.flip(H_vs, axis=0), np.flip(s, axis=1), np.flip(e_vs, axis=0), r_flip, norm=True)
            F_vs_mirror = cm.decode()

            self.assertAllClose(F_vs_mirror_matrix, F_vs_mirror)


class TestForwardBackwardTree(FBAlgorithmBase):
    """Tests that the tree algorithm computes the same forward matrix as the simple method."""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):
            mu = e_vs[:,0]
            F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(n, m, H_vs, s, e_vs, r, norm=True)
            B_vs = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)

            # Note, need to remove the first sample from the ts, and ensure that invariant sites aren't removed.
            ts_check = ts.simplify(range(1,n+1), filter_sites=False)
            cm = fbh_vst.ls_forward_tree(s[0,:], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))
            
            ts_check_mirror = fbh_vst.mirror_coordinates(ts_check)
            r_flip = np.flip(r)
            cm = fbh_vst.ls_backward_tree(np.flip(s[0,:]), ts_check_mirror, r_flip, np.flip(mu), np.flip(cm.normalisation_factor))
            B_vs_tree = np.flip(cm.decode(), axis=0)

            self.assertAllClose(B_vs, B_vs_tree)
