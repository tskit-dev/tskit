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


## notes 6/11:
## Question: Do we want to return the LD matrix over variants or over sites?
##  I think sites, and that's what the naive_ld_matrix returns
##
## Other notes: the second ld_matrix function is outdated, and cannot handle polarized
##  or multiple derived mutations.
##
## Because the returned LD matrix might not include every variant, and variants may be
##  distributed over multiple windows, we probably need to also return information of
##  which index in each LD matrix corresponds to which mutation in the tree sequence.
##  Thus, probably want to also return list of lists of site id indexes.
##
## Should windows be a list of non-overlapping intervals? Currently, set up so that we
##  just define window edges, so all windows necessarily have zero separation. We might
##  instead want to be able to compute LD matrices or LD scores from windows that are
##  distant. So could look like `windows = [(100, 200), (500, 1000), (1000, 2000)]`


def naive_ld_matrix(ts, function, sample_sets=None, windows=None, polarized=True):
    """
    This function returns the LD matrix for a given two-locus stats function. This is
    a simple implementation that loops through sites in the tree sequence, collects
    genotypes for observed variants at those sites, tallies two-locus genotype counts,
    and passes those counts to the stats function.

    function: the stats function takes sample_sets, sample_sizes, and polarized as its
    arguments and returns the scalar statistics.

    sample_sets: a list of lists of samples. Typically, we'd think of each list of
    samples as the samples that belong to each population we compute the statistics for.

    windows: a list of window break points, so that there are len(windows)-1 total LD
    matrices, if windows is given. If windows is None, we compute one LD matrix that
    spans the entire sequence length. Note that windows does not need to span the entire
    sequence length, but there will not be gaps between consecutive windows if
    len(windows) is greater than two.

    polarized: if True, we compute statistics over derived variants at each site. If
    False, we sum statistics over all variants, derived and ancestral, at each site.
    Note that in the summary function, the standard approach is to return the statistic
    multiplied by 1/4 to maintain consistency between polarized and unpolarized stats.

    Note: we assume that genotypes coded as `0` are the ancestral state, and all other
    genotypes are derived.
    """
    if windows is None:
        windows = [0, ts.sequence_length]
    else:
        assert len(windows) > 1, "need to define at least one window interval"
    windows = np.array(windows)

    assert np.all(windows[1:] - windows[:-1] > 0), "windows list must be increasing"
    assert windows[0] >= 0 and windows[-1] <= ts.sequence_length

    if sample_sets is None:
        sample_sets = [ts.samples()]
    ns = [len(ss) for ss in sample_sets]

    num_window_mutations = [0 for w in range(len(windows)-1)]
    site_indexes = {}
    for s in ts.sites():
        # get window index and snp index within that window
        window_index = min(np.argwhere(s.position < windows)[0]) - 1
        site_indexes[s.id] = (window_index, num_window_mutations[window_index])
        num_window_mutations[window_index] += 1

    A = [np.zeros((num_mutations, num_mutations))
         for num_mutations in num_window_mutations]

    for v1 in ts.variants():
        w1, m1 = site_indexes[v1.site.id]
        for v2 in ts.variants():
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


## old function, probably won't work with multi-allelic sites
def ld_matrix(ts, function, sample_sets=None, windows=None):
    """
    Note from 6/11: this function will not generalize well to the case with more than
    two variants, since I'm getting genotype counts using `tree.samples(node)`.

    Returns the two-locus stats matrix for a given tree sequence and statistic
    function. Optionally, we can specify population ids for multi-pop statistics
    or if we want to compute the statistic for just a subset of samples (say
    if the ts has samples from multiple populations, but we just want r^2 from
    one population). The defaul behavior is to return a single LD matrix for all
    pairwise comparisons across the tree sequence. We can also define windows,
    in which case we return an LD matrix for all pairwise comparisons within
    each window. Windows are non-overlapping and contiguous, but don't
    necessarily need to reach to the tree sequence limits. We could thus specify
    a single window for a subset of the full length if we care about a specific
    region.

    The function's arguments are a list of two-locus haplotype counts
    [(n_AB, n_Ab, n_aB)_0, (n_AB, n_Ab, n_aB)_1, ...] for each population, as
    given in population_ids.

    sample_sets is a list of lists of sample indexes to compute statistics over.
    The length of sample_sets has to match the number of two-locus counts that
    the function takes. For example, if the function computes some LD statistic
    for two populations, sample_sets must have length two.
    If sample_sets is None, we assume we compute a single population statistic and
    take all samples.
    
    This method uses `tree.samples(node_id)` and takes set intersections of samples
    below pairs of nodes to get counts of AB, Ab, and aB. This is not readily
    generalizable to weights that aren't 0/1 indicators or for more than two states
    per site. But it will be useful to compare to the generalized function.
    """
    if windows is None:
        # we take all mutations (careful, could be quite large)
        windows = [0, ts.sequence_length]
        window_arr = np.array(windows)
        ms = [ts.get_num_mutations()]
    else:
        # get the number of mutations within each window
        # there has to be a better way of doing this...
        ms = [0 for w in range(len(windows)-1)]
        window_arr = np.array(windows)
        for mut in ts.mutations():
            if mut.position < window_arr[0]:
                continue
            if mut.position >= window_arr[-1]:
                break
            w_ind = np.argwhere(np.logical_and(mut.position >= window_arr[:-1],
                                               mut.position < window_arr[1:]))
            ms[w_ind[0][0]] += 1

    if sample_sets is None:
        sample_sets = [ts.samples()]

    ns = [len(ss) for ss in sample_sets]

    # list of empty ld matrices over the specified windows (or entire length, if
    # windows are unspecified)
    A = [np.zeros((m, m), dtype=float) for m in ms]

    # need to keep track of how many mutations within each window we visited, so that
    # we fill in the correct entry in the LD matrices
    sites_visited = [0 for w in range(len(ms))]

    # which leaves are in each population, use sets for set intersections
    pop_leaves = [set(ss) for ss in sample_sets]

    for t1 in ts.trees():
        # if tree is outside of window ranges, skip
        if t1.interval[1] < windows[0]:
            continue
        if t1.interval[0] >= windows[-1]:
            break
        # loop over each mutation on this tree, compute statistic with all
        # mutations to the right, within the window
        for sA in t1.sites():
            assert len(sA.mutations) == 1
            # get window for this site
            if sA.position < windows[0] or sA.position >= windows[-1]:
                continue
            w_ind = np.argwhere(np.logical_and(sA.position >= window_arr[:-1],
                                               sA.position < window_arr[1:]))[0][0]

            mA = sA.mutations[0]

            # get the number of leaves carrying this mutation in each pop
            leaves_below_A = set(t1.samples(mA.node))
            nAs = [len(leaves_below_A & pl) for pl in pop_leaves]

            # track the number visited to the right of sA
            sites_visited_to_right = 0

            for t2 in ts.trees(tracked_samples=leaves_below_A):
                if t2.interval[1] < t1.interval[1]:
                    continue
                for sB in t2.sites():
                    assert len(sB.mutations) == 1
                    # check that sB is in the same window
                    if sB.position < windows[w_ind] or sB.position >= windows[w_ind+1]:
                        continue
                    if sB.position < sA.position:
                        continue

                    mB = sB.mutations[0]

                    # get number of leaves carrying B mutation in each pop
                    leaves_below_B = set(t2.samples(mB.node))
                    nBs = [len(leaves_below_B & pl) for pl in pop_leaves]

                    # get the number of samples with AB in each pop
                    nABs = [len(leaves_below_A & leaves_below_B & pl)
                            for pl in pop_leaves]

                    counts = [(nAB, nA-nAB, nB-nAB)
                              for nAB, nA, nB in zip(nABs, nAs, nBs)]

                    # pass counts and sample sizes to two-locus function, then
                    # fill in the matrix for the site in this window
                    val = function(counts, ns)
                    A[w_ind][sites_visited[w_ind],
                             sites_visited[w_ind]+sites_visited_to_right] = val
                    A[w_ind][sites_visited[w_ind]+sites_visited_to_right,
                             sites_visited[w_ind]] = val

                    sites_visited_to_right += 1

            sites_visited[w_ind] += 1

    return A



##############################
# LD score calculator
##############################

## Notes from 6/11
## Question: Should the LD score include `LD` with itself? What is the standard when
##  computing LD scores? Maybe allow as an option: `include_focal=False`, or something?

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
    if windows is None:
        windows = [0, ts.sequence_length]
    else:
        assert len(windows) > 1, "need to define at least one window interval"
    windows = np.array(windows)

    assert np.all(windows[1:] - windows[:-1] > 0), "windows list must be increasing"
    assert windows[0] >= 0 and windows[-1] <= ts.sequence_length

    if sample_sets is None:
        sample_sets = [ts.samples()]
    ns = [len(ss) for ss in sample_sets]

    # get the number of mutations within each window and store window and site indexes
    num_window_mutations = [0 for w in range(len(windows)-1)]
    site_indexes = {}
    for s in ts.sites():
        window_index = min(np.argwhere(s.position < windows)[0]) - 1
        site_indexes[s.id] = (window_index, num_window_mutations[window_index])
        num_window_mutations[window_index] += 1

    scores = [np.zeros(num_mutations) for num_mutations in num_window_mutations]

    for focal_var in ts.variants():
        if focal_var.site.id not in site_indexes:
            # this site isn't within the windows we track
            continue
        focal_window, focal_index = site_indexes[focal_var.site.id]
        for linked_var in ts.variants():
            if (linked_var.site.id in site_indexes and
                site_indexes[linked_var.site.id][0] == focal_window):
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


## also an old function, probably will not work well with multi-allelic sites
def ld_scores(ts, sample_sets, function, window_size=1e5):
    """
    Computes LD scores mutations within the tree sequence. The LD score of a
    SNP is computed as the sum of LD between that focal SNP and all other SNPs
    within a given distance (window_size to left and right).

    A lot of the logic is copied over from the ld_matrix function.
    """
    ns = [len(ss) for ss in sample_sets]

    scores = np.zeros(ts.num_mutations)

    # track current site for indexing the scores
    current_site = 0

    # which leaves are in each population, use sets for set intersections
    pop_leaves = [set(ss) for ss in sample_sets]

    for t1 in ts.trees():
        for sA in t1.sites():
            mA = sA.mutations[0]

            # get the number of leaves carrying this mutation in each pop
            leaves_below_A = set(t1.samples(mA.node))
            nAs = [len(leaves_below_A & pl) for pl in pop_leaves]

            # track the number visited to the right of sA
            sites_visited_to_right = 0

            for t2 in ts.trees(tracked_samples=leaves_below_A):
                if t2.interval[1] < t1.interval[1]:
                    continue
                if t2.interval[0] > t1.interval[1] + window_size:
                    break
                for sB in t2.sites():
                    assert len(sB.mutations) == 1
                    if sB.position < sA.position:
                        continue
                    if sB.position > sA.position + window_size:
                        break

                    mB = sB.mutations[0]

                    # get number of leaves carrying B mutation in each pop
                    leaves_below_B = set(t2.samples(mB.node))
                    nBs = [len(leaves_below_B & pl) for pl in pop_leaves]

                    # get the number of samples with AB in each pop
                    nABs = [len(leaves_below_A & leaves_below_B & pl)
                            for pl in pop_leaves]

                    counts = [(nAB, nA-nAB, nB-nAB) 
                              for nAB, nA, nB in zip(nABs, nAs, nBs)]

                    # pass counts and sample sizes to two-locus function
                    val = function(counts, ns)
                    scores[current_site] += val
                    if sites_visited_to_right > 0:
                        scores[current_site+sites_visited_to_right] += val

                    sites_visited_to_right += 1

            current_site += 1

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
    D2 = 0#(nAB * nab - nAb * naB) / n / (n-1)
    if polarized is True:
        return D2
    else:
        return D2 / 4

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
