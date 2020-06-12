# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (C) 2016 University of Oxford
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
Test cases for ld matrix calculations.
"""
import unittest
import io

import numpy as np
import msprime

import tskit
import _tskit
#import tests.tsutil as tsutil
#import tests.test_wright_fisher as wf


##############################
# LD matrix calculator
##############################


## notes 6/12:
## Question: Do we want to return the LD matrix over variants or over sites?
##  I think sites, and that's what the naive_ld_matrix returns
##
## Because the returned LD matrix might not include every variant, and variants may be
##  distributed over multiple windows, we probably need to also return information of
##  which index in each LD matrix corresponds to which mutation in the tree sequence.
##  Thus, probably want to also return list of lists of site id indexes.
##
## Should windows be a list of non-overlapping intervals? Previously, I set up so that we
##  just define window edges, so all windows necessarily have zero separation. Now we
##  instead are able to compute LD matrices or LD scores from windows that are
##  not adjacent. So could look like `windows = [(100, 200), (500, 1000), (1000, 2000)]`


def naive_ld_matrix(ts, function, sample_sets=None, windows=None, polarized=True):
    """
    This function returns the LD matrix for a given two-locus stats function and a list
    of site indexes corresponding to the indexes in the LD matrices. This is
    a simple implementation that loops through sites in the tree sequence, collects
    genotypes for observed variants at those sites, tallies two-locus genotype counts,
    and passes those counts to the stats function.

    function: the stats function takes sample_sets, sample_sizes, and polarized as its
    arguments and returns the scalar statistics.

    sample_sets: a list of lists of samples. Typically, we'd think of each list of
    samples as the samples that belong to each population we compute the statistics for.

    windows: a list of window intervals. If windows is None, we compute one matrix that
    spans the entire sequence length. Note that windows does not need to cover the entire
    sequence length, and there can be gaps between the windows.
    For example: windows = [(100, 200), (500, 1000), (1000, 2000)]

    polarized: if True, we compute statistics over derived variants at each site. If
    False, we sum statistics over all variants, derived and ancestral, at each site.
    Note that in the summary function, the standard approach is to return the statistic
    multiplied by 1/4 to maintain consistency between polarized and unpolarized stats.

    Note: we assume that genotypes coded as `0` are the ancestral state, and all other
    genotypes are derived.
    """
    if windows is None:
        windows = [(0, ts.sequence_length)]
    window_edges = np.array(windows)

    assert np.all(window_edges[:,0] < window_edges[:,1]), "windows must specify left/right endpoints"
    assert np.all(window_edges[1:,0] >= window_edges[:-1,1] ), "windows cannot overlap"
    assert window_edges[0][0] >= 0 and window_edges[-1][1] <= ts.sequence_length, "windows must be within sequence length"

    if sample_sets is None:
        sample_sets = [ts.samples()]
    ns = [len(ss) for ss in sample_sets]

    num_window_mutations = [0 for _ in windows]
    site_indexes = {}
    for s in ts.sites():
        # get window index and snp index within that window
        window_index = np.where(
            np.logical_and(
                s.position > window_edges[:,0],
                s.position < window_edges[:,1]
            )
        )[0]
        if len(window_index) == 0:
            continue
        assert len(window_index) == 1
        site_indexes[s.id] = (window_index[0], num_window_mutations[window_index[0]])
        num_window_mutations[window_index[0]] += 1

    A = [np.zeros((num_mutations, num_mutations))
         for num_mutations in num_window_mutations]

    for v1 in ts.variants():
        if v1.site.id not in site_indexes:
            continue
        w1, m1 = site_indexes[v1.site.id]
        for v2 in ts.variants():
            if v2.site.id not in site_indexes:
                continue
            if v1.position >= v2.position:
                w2, m2 = site_indexes[v2.site.id]
                if w1 == w2: # sites fall within the same window
                    all_genotypes1 = set(v1.genotypes)
                    all_genotypes2 = set(v2.genotypes)
                    for g1 in all_genotypes1:
                        if polarized is True and g1 == 0:
                            continue
                        g_array1 = (v1.genotypes == g1).astype(int)
                        for g2 in all_genotypes2:
                            if polarized is True and g2 == 0:
                                continue
                            g_array2 = (v2.genotypes == g2).astype(int)

                            two_locus_genotypes = list(zip(g_array1, g_array2))
                            sample_genotypes = [[two_locus_genotypes[sample_index] for
                                                sample_index in ss]
                                                for ss in sample_sets]
                            two_locus_counts = [(gts.count((1, 1)),
                                                 gts.count((1, 0)),
                                                 gts.count((0, 1)))
                                                for gts in sample_genotypes]
                            A[w1][m1, m2] += function(two_locus_counts, ns, polarized)

    for i, ld_mat in enumerate(A):
        idx_upper = np.triu_indices(len(ld_mat), 1)
        A[i][idx_upper] += A[i].T[idx_upper]

    return A


##############################
# LD score calculator
##############################

## Notes from 6/12
## Question: Should the LD score include `LD` with itself? What is the standard when
##  computing LD scores? Maybe allow as an option: `include_focal=False`, or something?
## Should we take linked variants that fall outside of the focal site's window, but still
##  falls within the focal_window_size?

def naive_ld_scores(ts, function, sample_sets=None, windows=None, focal_window_size=1e5,
                    polarized=True):
    """
    This function, as in the naive_ld_matrix function, computes LD scores by simply
    looping over focal sites, and then considering all linked variants within the
    focal_window_size.

    function: the two-locus statistic function that takes genotype counts, sample sizes,
    and polarized.

    sample_sets: list of lists of sample ids. The length of sample_sets is the number
    of sets that the LD statistic function takes. If None, we assume use all samples
    as a single set.

    windows: if None, compute LD scores for all sites, using all other variants. If
    windows are given, we compute LD scores for focal sites within each window with LD
    scores computed only from variants within the same window.

    focal_window_size: the maximum distance (in units of the sequence length) from the
    focal site to include other variants in each SNP's score.

    polarized: if True, don't compute statistics over ancestral state (0s). If False,
    we sum over two-locus statistics computed between all pairs of variants (both
    derived and ancestral) at the two sites.
    """
    # a lot is copied from naive_ld_matrix here
    if windows is None:
        windows = [(0, ts.sequence_length)]
    window_edges = np.array(windows)

    assert np.all(window_edges[:,0] < window_edges[:,1]), "windows must specify left/right endpoints"
    assert np.all(window_edges[1:,0] >= window_edges[:-1,1] ), "windows cannot overlap"
    assert window_edges[0][0] >= 0 and window_edges[-1][1] <= ts.sequence_length, "windows must be within sequence length"

    if sample_sets is None:
        sample_sets = [ts.samples()]
    ns = [len(ss) for ss in sample_sets]

    num_window_mutations = [0 for _ in windows]
    site_indexes = {}
    for s in ts.sites():
        window_index = np.where(
            np.logical_and(
                s.position > window_edges[:,0],
                s.position < window_edges[:,1]
            )
        )[0]
        if len(window_index) == 0:
            continue
        assert len(window_index) == 1
        site_indexes[s.id] = (window_index[0], num_window_mutations[window_index[0]])
        num_window_mutations[window_index[0]] += 1

    scores = [np.zeros(num_mutations) for num_mutations in num_window_mutations]

    for focal_var in ts.variants():
        if focal_var.site.id not in site_indexes:
            # this site isn't within the windows we track
            continue
        focal_window, focal_index = site_indexes[focal_var.site.id]
        for linked_var in ts.variants():
            if (linked_var.site.id in site_indexes and
                site_indexes[linked_var.site.id][0] == focal_window):
                # currently, we only take linked variants that fall within same
                # focal window - might want to have that as an option
                if abs(focal_var.position-linked_var.position) <= focal_window_size:
                    # a lot of copied logic from the naive ld matrix function
                    all_genotypes1 = set(focal_var.genotypes)
                    all_genotypes2 = set(linked_var.genotypes)
                    for g1 in all_genotypes1:
                        if polarized is True and g1 == 0:
                            continue
                        g_array1 = (focal_var.genotypes == g1).astype(int)
                        for g2 in all_genotypes2:
                            if polarized is True and g2 == 0:
                                continue
                            g_array2 = (linked_var.genotypes == g2).astype(int)

                            two_locus_genotypes = list(zip(g_array1, g_array2))
                            sample_genotypes = [[two_locus_genotypes[sample_index] for
                                                sample_index in ss]
                                                for ss in sample_sets]
                            two_locus_counts = [(gts.count((1, 1)),
                                                 gts.count((1, 0)),
                                                 gts.count((0, 1)))
                                                for gts in sample_genotypes]
                            scores[focal_window][focal_index] += function(
                                two_locus_counts, ns, polarized
                            )

    return scores


##############################
# Common/example summary functions
##############################


####
# One-populations two-locus functions
####


def r2_function(sample_counts, ns, polarized=True):
    """
    To compute r^2, we take the two-locus haplotype counts and sample size to
    compute haplotype frequencies f. Compute r^2 from f's.
    Returns 0 if either A or B mutation is fixed.
    """
    assert len(sample_counts) == 1, "compute r^2 for only a single population"
    assert len(ns) == 1, "compute r^2 for only a single population"
    (nAB, nAb, naB) = sample_counts[0]
    n = ns[0]
    fAB = nAB / n
    fA = (nAB + nAb) / n
    fB = (nAB + naB) / n
    D = fAB - fA * fB
    if fA != 0 and fA != 1 and fB != 0 and fB != 1:
        r2 = D * D / (fA * (1 - fA) * fB * (1 - fB))
    else:
        r2 = 0
    if polarized is True:
        return r2
    else:
        return r2 / 4


def r_function(sample_counts, ns, polarized=True):
    assert len(sample_counts) == 1, "compute r^2 for only a single population"
    assert len(ns) == 1, "compute r^2 for only a single population"
    (nAB, nAb, naB) = sample_counts[0]
    n = ns[0]
    fAB = nAB / n
    fA = (nAB + nAb) / n
    fB = (nAB + naB) / n
    D = fAB - fA * fB
    if fA != 0 and fA != 1 and fB != 0 and fB != 1:
        r = D / np.sqrt(fA * (1 - fA) * fB * (1 - fB))
    else:
        r=0
    if polarized is True:
        return r
    else:
        return r / 4


def D_unbiased_function(sample_counts, ns, polarized=True):
    """
    sample counts: list of length one, with sample counts (nAB, nAb, naB)
    ns: list of length 1 of sample size
    """
    assert len(sample_counts) == 1, "can compute D for only a single population"
    assert len(ns) == 1, "can compute D for only a single population"
    (nAB, nAb, naB) = sample_counts[0]
    n = ns[0]
    nab = n - nAB - nAb - naB
    assert nab >= 0, "sum of sample counts larger than given sample size"
    D = (nAB * nab - nAb * naB) / n / (n-1)
    if polarized is True:
        return D
    else:
        return D / 4

def D2_unbiased_function(sample_counts, ns, polarized=True):
    """
    sample counts: list of length one, with sample counts (nAB, nAb, naB)
    ns: list of length 1 of sample size
    """
    assert len(sample_counts) == 1, "can compute D for only a single population"
    assert len(ns) == 1, "can compute D for only a single population"
    (nAB, nAb, naB) = sample_counts[0]
    n = ns[0]
    nab = n - nAB - nAb - naB
    assert nab >= 0, "sum of sample counts larger than given sample size"
    if polarized is False:
        return ((nAB * (nAB - 1) * nab * (nab - 1) 
                - nAB * nAb * naB * nab ) / (2 * n * (n - 1) * (n - 2) * (n - 3)))
    else:
        return ((nAB * (nAB - 1) * nab * (nab - 1)
                 + nAb * (nAb - 1) * naB * (naB - 1)
                 - 2 * nAB * nAb * naB * nab) / (2 * n * (n - 1) * (n - 2) * (n - 3)))

####
# Two-populations two-locus functions
####


def r1_r2_function(sample_counts, ns, polarized=True):
    """
    The two population analog to r^2: computes r1 times r2, taking haplotype and
    sample size counts in two populations.
    If the mutation is fixed in either population, returns 0
    """
    assert len(sample_counts) == 2, "compute r1.r2 for only a single population"
    assert len(ns) == 2, "compute r1.r2 for only a single population"
    fAB1 = sample_counts[0][0] / ns[0]
    fAB2 = sample_counts[1][0] / ns[1]
    fA1 = sample_counts[0][1] / ns[0]
    fA2 = sample_counts[1][1] / ns[1]
    fB1 = sample_counts[0][2] / ns[0]
    fB2 = sample_counts[1][2] / ns[1]
    D1 = fAB1 - fA1 * fB1
    D2 = fAB2 - fA2 * fB2
    if fA1 != 0 and fA1 != 1 and fB1 != 0 and fB1 != 1:
        r1 = D1 / np.sqrt(fA1 * (1 - fA1) * fB1 * (1 - fB1))
    else:
        r1 = 0
    if fA2 != 0 and fA2 != 1 and fB2 != 0 and fB2 != 1:
        r2 = D2 / np.sqrt(fA2 * (1 - fA2) * fB2 * (1 - fB2))
    else:
        r2 = 0
    if polarized is True:
        return r1 * r2
    else:
        return r1 * r2 / 4



##############################
# Test implementations
##############################


#class TestLdMatrixCalculator(unittest.TestCase):
#    """
#    Tests for the general LD matrix calculator.
#    """
#    
#    num_test_sites = 50
