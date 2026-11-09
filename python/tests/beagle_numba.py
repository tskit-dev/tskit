"""
Implementation of the BEAGLE 4.1 algorithm to impute alleles by linear interpolation
of state probabilities at ungenotyped positions in the hidden state probability matrix.

This was implemented while closely consulting the BEAGLE 4.1 paper and source code:
* Browning & Browning (2016). Am J Hum Genet 98:116-126. doi:10.1016/j.ajhg.2015.11.020
* Source code: https://faculty.washington.edu/browning/beagle/b4_1.html

These notations are used throughout:
h = number of reference haplotypes.
m = number of genotyped positions.
x = number of ungenotyped positions.

This implementation takes the following inputs:
* Panel of reference haplotypes in a matrix of size (m + x, h).
* Query haplotype in an array of size (m + x).
* Physical positions of all the markers in an array of size (m + x).
* Genetic map.

In the query haplotype:
* Genotyped positions take values of 0, 1, 2, or 3 (i.e. ACGT encoding).
* Ungenotyped positions take -1.

The following objects are computed:
* Forward and backward probability matrices of size (m, h).
* Hidden state probability matrix of size (m, h).
* Interpolated state probability matrix of size (x, h).
* Imputed allele probability matrix of size (x, 4),
* Imputed alleles as the maximum a posteriori alleles.

The following evaluation metrics are produced in VCF format:
* Estimated allelic R-squared (AR2).
* Dosage R-squared (DR2).
* Estimated allele frequency (AF).
* Genotype probabilities of 00, 01/10, and 11 (GP).
* Estimated dosage (DS).

To improve computational efficiency, BEAGLE uses aggregated markers, which are clusters
of markers within a 0.005 cM interval (default). Because the genotypes are phased,
the alleles in the aggregated markers form distinct "allele sequences". Below, we do not
use aggregated markers or allele sequences, which would complicate the implementation.

Rather than exactly replicate the original BEAGLE algorithm, this implementation uses
Equation 1 of BB2016.
"""
import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit

import _tskit
import tskit


__VERSION__ = "0.0.0"
__DATE__ = "20XXXXXX"

_ACGT_ALLELES_INT = [0, 1, 2, 3, tskit.MISSING_DATA]


@dataclass(frozen=True)
class GeneticMap:
    """
    A genetic map containing:
    * Physical positions (bp).
    * Genetic map positions (cM).
    """

    base_pos: np.ndarray
    gen_pos: np.ndarray

    def __post_init__(self):
        assert len(self.base_pos) == len(
            self.gen_pos
        ), "Lengths of physical positions and genetic map positions differ."
        assert np.all(
            self.base_pos[1:] > self.base_pos[:-1]
        ), "Physical positions are not in strict ascending order."


@dataclass(frozen=True)
class ImpData:
    """
    Imputation data containing:
    * Individual names.
    * Physical positions of imputed sites (bp).
    * Designated REF allele at each site.
    * Designated ALT allele at each site.
    * Imputed alleles at each site.
    * Imputed allele probabilities at each site.

    Assume that all the sites are biallelic.

    Let x = number of imputed sites and q = number of query haplotypes.
    Since the query haplotypes are from diploid individuals, q is equal to
    twice the number of individuals.

    Imputed alleles is a matrix of size (q, x).
    Imputed allele probabilities is a matrix of size (q, x).
    """

    individual_names: list
    site_pos: np.ndarray
    refs: np.ndarray
    alts: np.ndarray
    alleles: np.ndarray
    allele_probs: np.ndarray

    def __post_init__(self):
        assert len(self.individual_names) > 0, "There must be at least one individual."
        assert len(self.site_pos) > 0, "There must be at least one site."
        assert self.alleles.shape[0] / 2 == len(
            self.individual_names
        ), "Number of query haplotypes is not equal to twice the number of individuals."
        assert len(self.site_pos) == len(
            self.alts
        ), "Number of sites in refs is not equal to the number of site positions."
        assert len(self.site_pos) == len(
            self.refs
        ), "Number of sites in alts != number of site positions."
        assert (
            len(self.site_pos) == self.alleles.shape[1]
        ), "Number of sites in alleles != number of site positions."
        assert (
            self.alleles.shape == self.allele_probs.shape
        ), "Dimensions of alleles != dimensions of allele probabilities."
        for i in range(self.alleles.shape[1]):
            assert np.all(np.isin(self.alleles[:, i], [self.refs[i], self.alts[i]]))
        assert np.all(
            np.isin(np.unique(self.refs), _ACGT_ALLELES_INT)
        ), "Unrecognized alleles are in REF alleles."
        assert np.all(
            np.isin(np.unique(self.alts), _ACGT_ALLELES_INT)
        ), "Unrecognized alleles are in ALT alleles."
        assert np.all(
            np.isin(np.unique(self.alleles), _ACGT_ALLELES_INT)
        ), "Unrecognized alleles are in alleles."
        assert ~np.array_equal(self.refs, self.alts), "Some REFs are identical to ALTs."

    @property
    def num_sites(self):
        return len(self.site_pos)

    @property
    def num_samples(self):
        return self.alleles.shape[0]

    @property
    def num_individuals(self):
        return len(self.individual_names)

    def get_ref_allele_at_site(self, i):
        return self.refs[i]

    def get_alt_allele_at_site(self, i):
        return self.alts[i]

    def get_alleles_at_site(self, i):
        idx_hap1 = np.arange(0, self.num_samples, 2)
        idx_hap2 = np.arange(1, self.num_samples, 2)
        a1 = self.alleles[idx_hap1, i]
        ap1 = self.allele_probs[idx_hap1, i]
        a2 = self.alleles[idx_hap2, i]
        ap2 = self.allele_probs[idx_hap2, i]
        return a1, ap1, a2, ap2


""" Helper functions. """


def remap_alleles(a):
    """
    Map an array of alleles encoded as characters to integers.

    :param np.ndarray a: Alleles.
    :return: Recoded alleles.
    :rtype: np.ndarray(dtype=np.int8)
    """
    b = np.zeros(len(a), dtype=np.int8) - 1  # Encoded as missing by default.
    for i in range(len(a)):
        if a[i] in [None, ""]:
            continue
        try:
            b[i] = tskit.ALLELES_ACGT.index(a[i])
        except ValueError as err:
            err.args = (f"Allele {a[i]} is not recognised.",)
            raise
    return b


def check_data(ref_h, query_h):
    """
    For each position, check whether the alleles in the query haplotype
    are represented in the reference haplotypes.

    Missing data (i.e. -1) are ignored.

    :param numpy.ndarray ref_h: Reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :return: True if alleles in query are in references at all the positions.
    :rtype: bool
    """
    m = ref_h.shape[0]  # Number of genotyped positions.
    assert m == len(query_h), "Reference and query haplotypes differ in length."
    for i in range(m):
        if query_h[i] != tskit.MISSING_DATA:
            ref_a = np.unique(ref_h[i, :])
            if query_h[i] not in ref_a:
                raise AssertionError(
                    f"Allele {query_h[i]} at the {i}-th position is not in reference."
                )
    return True


def convert_to_genetic_map_positions(pos, *, genetic_map=None):
    """
    Convert physical positions (bp) to genetic map positions (cM).

    In BEAGLE 4.1, when a genetic map is not specified, it is assumed
    that the recombination rate is constant (1 cM / 1 Mb).

    If a genetic map is specified, then the genetic map positions are
    either taken straight from it or interpolated using it. The genetic map
    needs to contain physical positions and corresponding genetic map positions.
    See `PlinkGenMap.java` in the BEAGLE 4.1 source code for details.

    :param numpy.ndarray pos: Physical positions (bp).
    :param GeneticMap genetic_map: Genetic map.
    :return: Genetic map positions (cM).
    :rtype: numpy.ndarray
    """
    # See 'cumPos' in 'ImputationData.java' in BEAGLE 4.1.
    _MIN_CM_DIST = 1e-7
    if genetic_map is None:
        return pos / 1e6  # 1 cM / 1 Mb
    assert np.all(pos >= genetic_map.base_pos[0]) and np.all(
        pos < genetic_map.base_pos[-1]
    ), "Some physical positions are outside of genetic map."
    # Approximate genetic map distances by linear interpolation.
    # Note np.searchsorted(a, v, side='right') returns i s.t. a[i-1] <= v < a[i].
    right_idx = np.searchsorted(genetic_map.base_pos, pos, side="right")
    est_cm = np.zeros(len(pos), dtype=np.float64)  # BEAGLE 4.1 uses double in Java.
    for i in range(len(pos)):
        a = genetic_map.base_pos[right_idx[i] - 1]
        b = genetic_map.base_pos[right_idx[i]]
        fa = genetic_map.gen_pos[right_idx[i] - 1]
        fb = genetic_map.gen_pos[right_idx[i]]
        assert (
            pos[i] >= a
        ), f"Query position is not >= left-bound position: {pos[i]}, {a}."
        assert (
            fb >= fa
        ), f"Genetic map positions are not monotonically ascending: {fb}, {fa}."
        est_cm[i] = fa
        est_cm[i] += (fb - fa) * (pos[i] - a) / (b - a)
        # Ensure that adjacent positions are not identical in cM.
        if i > 0:
            if est_cm[i] - est_cm[i - 1] < _MIN_CM_DIST:
                est_cm[i] = est_cm[i - 1] + _MIN_CM_DIST
    return est_cm


""" Li & Stephens HMM. """


def get_mismatch_probs(num_sites, error_rate):
    """
    Compute mismatch probabilities at genotyped positions.

    Mutation rates should be dominated by the rate of allele error,
    which should be the main source of mismatch between query and
    reference haplotypes.

    In BEAGLE 4.1/5.4, error rate is 1e-4 by default, and capped at 0.5.
    In IMPUTE5, the default value is also 1e-4.

    This corresponds to `mu` in `_tskit.LsHmm`.

    :param numpy.ndarray num_sites: Number of sites.
    :param float error_rate: Allele error rate.
    :return: Mismatch probabilities.
    :rtype: numpy.ndarray
    """
    MAX_ERROR_RATE = 0.50
    if error_rate >= MAX_ERROR_RATE:
        error_rate = MAX_ERROR_RATE
    mismatch_probs = np.zeros(num_sites, dtype=np.float64) + error_rate
    return mismatch_probs


def get_transition_probs(cm, h, ne):
    """
    Compute probabilities of transitioning to a random state at genotyped positions.

    In BEAGLE 4.1, the default value of `ne` is set to 1e6,
    whereas in BEAGLE 5.4, the default value of `ne` is set to 1e5.
    In BEAGLE 4.1/5.4, this value was optimized empirically on datasets
    from large outbred human populations.

    In IMPUTE5, the default value of `ne` is set to 1e6.

    If `h` is small and `ne` is large, the transition probabilities are ~1.
    In such cases, it may be desirable to set `ne` to a small value
    to avoid switching between haplotypes too frequently.

    This corresponds to `rho` in `_tskit.LsHmm`.

    :param numpy.ndarray cm: Genetic map positions (cM).
    :param int h: Number of reference haplotypes.
    :param float ne: Effective population size.
    :return: Transition probabilities.
    :rtype: numpy.ndarray
    """
    # E(number of crossover events) at first site is always 0.
    r = np.zeros(len(cm), dtype=np.float64)
    r[1:] = np.diff(cm)
    c = -0.04 * (ne / h)
    trans_probs = -1 * np.expm1(c * r)
    return trans_probs


@njit
def compute_emission_probability(mismatch_prob, is_match, *, num_alleles=2):
    """
    Compute the emission probability at a site based on whether the alleles
    carried by a query haplotype and a reference haplotype match at the site.

    Emission probability may be scaled according to the number of distinct
    segregating alleles.

    :param float mismatch_prob: Mismatch probability.
    :param bool is_match: True if matched, otherwise False.
    :param int num_alleles: Number of distinct alleles (default = 2).
    :return: Emission probability.
    :rtype: float
    """
    em_prob = mismatch_prob
    if is_match:
        em_prob = 1.0 - (num_alleles - 1) * mismatch_prob
    return em_prob


""" Replication of BEAGLE's implementation of the Li and Stephens HMM. """


@njit
def compute_forward_matrix(
    ref_h, query_h, trans_probs, mismatch_probs, *, num_alleles=2
):
    """
    Implement Li and Stephens forward algorithm.

    Reference haplotypes and query haplotype are subsetted to genotyped positions.
    So, they are a matrix of size (m, h) and an array of size m, respectively.

    This computes a forward probablity matrix of size (m, h).

    :param numpy.ndarray ref_h: Reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :param numpy.ndarray trans_probs: Transition probabilities.
    :param numpy.ndarray mismatch_probs: Mismatch probabilities.
    :param int num_alleles: Number of distinct alleles (default = 2).
    :return: Forward probability matrix.
    :rtype: numpy.ndarray
    """
    h = ref_h.shape[1]  # Number of reference haplotypes.
    m = ref_h.shape[0]  # Number of genotyped positions.
    fwd_mat = np.zeros((m, h), dtype=np.float64)
    last_sum = 1.0  # Normalization factor.
    for i in range(m):
        # Get site-specific parameters.
        shift = trans_probs[i] / h
        scale = (1 - trans_probs[i]) / last_sum
        # Get allele at genotyped position i on query haplotype.
        query_a = query_h[i]
        for j in range(h):
            # Get allele at genotyped position i on reference haplotype j.
            ref_a = ref_h[i, j]
            # Get emission probability.
            em_prob = mismatch_probs[i]
            if query_a == ref_a:
                em_prob = 1.0 - (num_alleles - 1) * mismatch_probs[i]
            fwd_mat[i, j] = em_prob
            if i > 0:
                fwd_mat[i, j] *= scale * fwd_mat[i - 1, j] + shift
        site_sum = np.sum(fwd_mat[i, :])
        # Prior probabilities are multiplied when i = 0.
        last_sum = site_sum / h if i == 0 else site_sum
    return fwd_mat


@njit
def compute_backward_matrix(
    ref_h, query_h, trans_probs, mismatch_probs, *, num_alleles=2
):
    """
    Implement Li and Stephens backward algorithm.

    Reference haplotypes and query haplotype are subsetted to genotyped positions.
    So, they are a matrix of size (m, h) and an array of size m, respectively.

    This computes a backward probablity matrix of size (m, h).

    In BEAGLE 4.1, the values are kept one position at a time. Here, we keep the values
    at all the positions.

    :param numpy.ndarray ref_h: Reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :param numpy.ndarray trans_probs: Transition probabilities.
    :param numpy.ndarray mismatch_probs: Mismatch probabilities.
    :param int num_alleles: Number of distinct alleles (default = 2).
    :return: Backward probability matrix.
    :rtype: numpy.ndarray
    """
    h = ref_h.shape[1]  # Number of reference haplotypes.
    m = ref_h.shape[0]  # Number of genotyped positions.
    bwd_mat = np.zeros((m, h), dtype=np.float64)
    bwd_mat[-1, :] = 1.0 / h  # Initialise the last column.
    for i in range(m - 2, -1, -1):
        iP1 = i + 1
        query_a = query_h[iP1]
        for j in range(h):
            ref_a = ref_h[iP1, j]
            em_prob = mismatch_probs[iP1]
            if query_a == ref_a:
                em_prob = 1.0 - (num_alleles - 1) * mismatch_probs[iP1]
            bwd_mat[iP1, j] *= em_prob
        site_sum = np.sum(bwd_mat[iP1, :])
        scale = (1 - trans_probs[iP1]) / site_sum
        shift = trans_probs[iP1] / h
        bwd_mat[i, :] = scale * bwd_mat[iP1, :] + shift
    return bwd_mat


def compute_state_prob_matrix(fwd_mat, bwd_mat):
    """
    Implement Li and Stephens forward-backward algorithm.

    The forward and backward probability matrices are of size (m, h).

    This computes a hidden state probability matrix of size (m, h),
    in which each element is the probability of copying from
    a reference haplotype at a genotyped position.

    Implementing this is simpler than faithfully reproducing BEAGLE 4.1.

    :param numpy.ndarray fwd_mat: Forward probability matrix.
    :param numpy.ndarray bwd_mat: Backward probability matrix.
    :return: Hidden state probability matrix.
    :rtype: numpy.ndarray
    """
    assert (
        fwd_mat.shape == bwd_mat.shape
    ), "Forward and backward matrices differ in shape."
    state_mat = np.multiply(fwd_mat, bwd_mat)
    # Normalise each column.
    for i in range(len(state_mat)):
        state_mat[i, :] /= np.sum(state_mat[i, :])
    return state_mat


""" Imputation of ungenotyped positions. """


@njit
def get_weights(typed_pos, untyped_pos, typed_cm, untyped_cm):
    """
    Compute weights for the ungenotyped positions in a query haplotype, which are used in
    linear interpolation of hidden state probabilities at the ungenotyped positions.

    In BB2016 (see below Equation 1), a weight between genotyped positions m and (m + 1)
    bounding ungenotyped position x is denoted lambda_m,x.

    lambda_m,x = [g(m + 1) - g(x)] / [g(m + 1) - g(m)],
    where g(i) = genetic map position of marker i.

    This looks for the right-bounding position instead of the left-bounding.

    :param numpy.ndarray typed_pos: Physical positions of genotyped markers (bp).
    :param numpy.ndarray untyped_pos: Physical positions of ungenotyped markers (bp).
    :param numpy.ndarray typed_cm: Genetic map positions of genotyped markers (cM).
    :param numpy.ndarray untyped_cm: Genetic map positions of ungenotyped markers (cM).
    :return: Weights for ungenotyped positions and indices of right-bounding positions.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    m = len(typed_pos)  # Number of genotyped positions.
    x = len(untyped_pos)  # Number of ungenotyped positions.
    # Identify genotype positions (m - 1) and m bounding ungenotyped position i.
    # Note np.searchsorted(a, v, side='right') returns i s.t. a[i-1] <= v < a[i].
    right_idx = np.searchsorted(typed_pos, untyped_pos, side="right")
    # Calculate weights for ungenotyped positions.
    weights = np.zeros(x, dtype=np.float64)
    for i in range(len(right_idx)):
        k = right_idx[i]
        if k == 0:
            # Left of the first genotyped position.
            weights[i] = 1.0
        elif k == m:
            # Right of the last genotyped position.
            weights[i] = 0.0
        else:
            # Between two genotyped positions.
            cm_m2x = typed_cm[k] - untyped_cm[i]
            # Avoid negative weights.
            if cm_m2x < 0:
                cm_m2x = 0
            cm_m2mM1 = typed_cm[k] - typed_cm[k - 1]
            weights[i] = cm_m2x / cm_m2mM1
    return (weights, right_idx)


@njit
def interpolate_allele_probs(
    state_mat,
    ref_h,
    pos_typed,
    pos_untyped,
    cm_typed,
    cm_untyped,
    *,
    use_threshold=False,
    return_weights=False,
):
    """
    Interpolate allele probabilities at the ungenotyped positions of a query haplotype
    following Equation 1 of BB2016.

    The interpolated allele probability matrix is of size (x, a),
    where a is the number of alleles.

    Note that this function takes:
    1. Hidden state probability matrix at genotyped positions of size (m, h).
    2. Reference haplotypes subsetted to ungenotyped positions of size (x, h).

    If thresholding is employed, it replicates BEAGLE's way to approximate calculations.
    See 'setFirstAlleleProbs', 'setAlleleProbs', and 'setLastAlleleProbs'
    in 'LSHapBaum.java' in BEAGLE 4.1 source code.

    :param numpy.ndarray state_mat: State probability matrix at genotyped positions.
    :param numpy.ndarray ref_h: Reference haplotypes subsetted to ungenotyped positions.
    :param numpy.ndarray pos_typed: Physical positions of genotyped markers (bp).
    :param numpy.ndarray pos_untyped: Physical positions of ungenotyped markers (bp).
    :param numpy.ndarray cm_typed: Genetic map positions at genotyped markers (cM).
    :param numpy.ndarray cm_untyped: Genetic map positions at ungenotyped markers (cM).
    :param bool use_threshold: Set trivial probabilities to 0 if True (default = False).
    :param bool return_weights: Return weights if True (default = False).
    :return: Imputed allele probabilities and weights.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    # TODO: Allow for biallelic site matrix. Work with `_tskit.lshmm` properly.
    alleles = np.arange(len(tskit.ALLELES_ACGT))
    m = state_mat.shape[0]  # Number of genotyped positions.
    x = ref_h.shape[0]  # Number of ungenotyped positions.
    # Set threshold to set trivial probabilities to zero.
    _MIN_THRESHOLD = 0
    weights, right_idx = get_weights(pos_typed, pos_untyped, cm_typed, cm_untyped)
    probs = np.zeros((x, len(alleles)), dtype=np.float64)
    for i in range(x):
        k = right_idx[i]
        w = weights[i]
        for a in alleles:
            is_a_in_ref_h = ref_h[i, :] == a
            if np.sum(is_a_in_ref_h) == 0:
                # This avoids division by zero when getting a threshold adaptively below.
                continue
            if use_threshold:
                # TODO: Check whether this is implemented correctly. Not used by default.
                # Threshold based on "the number of subsets in the partition Am of H".
                threshold_Am = 1 / np.sum(is_a_in_ref_h)
                _MIN_THRESHOLD = min(0.005, threshold_Am)
            if k == 0:
                # See 'setFirstAlleleProbs' in 'LSHapBaum.java'.
                assert w == 1.0, "Weight should be 1.0."
                sum_probs_a_k = np.sum(state_mat[k, is_a_in_ref_h])
                if sum_probs_a_k > _MIN_THRESHOLD:
                    probs[i, a] += sum_probs_a_k
            elif k == m:
                # See 'setLastAlleleProbs' in 'LSHapBaum.java'.
                assert w == 0.0, "Weight should be 0.0."
                sum_probs_a_kM1 = np.sum(state_mat[k - 1, is_a_in_ref_h])
                if sum_probs_a_kM1 > _MIN_THRESHOLD:
                    probs[i, a] += sum_probs_a_kM1
            else:
                # See 'setAlleleProbs' in 'LSHapBaum.java'.
                sum_probs_a_k = np.sum(state_mat[k, is_a_in_ref_h])
                sum_probs_a_kM1 = np.sum(state_mat[k - 1, is_a_in_ref_h])
                if max(sum_probs_a_k, sum_probs_a_kM1) > _MIN_THRESHOLD:
                    probs[i, a] += w * sum_probs_a_kM1
                    probs[i, a] += (1 - w) * sum_probs_a_k
    site_sums = np.sum(probs, axis=1)
    assert np.all(site_sums > 0), "Some site sums of allele probabilities is <= 0."
    probs_rescaled = probs / site_sums[:, np.newaxis]
    if return_weights:
        return (probs_rescaled, weights)
    return (probs_rescaled, None)


def get_map_alleles(allele_probs):
    """
    Compute maximum a posteriori (MAP) alleles at the ungenotyped positions
    of a query haplotype, based on posterior marginal allele probabilities.

    The imputed alleles and their probabilities are arrays of size x.

    WARN: If the allele probabilities are equal, then allele 0 is arbitrarily chosen.
    TODO: Investigate how often this happens and the effect of this arbitrary choice.

    :param numpy.ndarray allele_probs: Imputed allele probabilities.
    :return: Imputed alleles and their probabilities.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    imputed_alleles = np.argmax(allele_probs, axis=1)
    max_allele_probs = np.max(allele_probs, axis=1)
    return (imputed_alleles, max_allele_probs)


def run_interpolation_beagle(
    ref_h,
    query_h,
    pos_all,
    *,
    ne=1e6,
    error_rate=1e-4,
    genetic_map=None,
    use_threshold=False,
):
    """
    Perform a simplified version of the procedure of interpolation-style imputation
    based on Equation 1 of BB2016.

    Reference haplotypes and query haplotype are of size (m + x, h) and (m + x).

    The physical positions of all the markers are an array of size (m + x).

    This produces imputed alleles and their probabilities at the ungenotyped positions
    of the query haplotype.

    The default values of `ne` and `error_rate` are taken from BEAGLE 4.1, not 5.4.
    In BEAGLE 5.4, the default value of `ne` is 1e5, and `error_rate` is data-dependent.

    :param numpy.ndarray ref_h: Reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :param numpy.ndarray pos_all: Physical positions of all the markers (bp).
    :param int ne: Effective population size (default = 1e6).
    :param float error_rate: Allele error rate (default = 1e-4).
    :param GeneticMap genetic_map: Genetic map (default = None).
    :param bool use_threshold: Set trivial probabilities to 0 if True (default = False).
    :return: Imputed alleles and their probabilities.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    warnings.warn("This function is experimental and not fully tested.", stacklevel=1)
    warnings.warn(
        "Check the reference and query haplotypes use the same allele encoding.",
        stacklevel=1,
    )
    num_alleles = len(tskit.ALLELES_ACGT)
    h = ref_h.shape[1]  # Number of reference haplotypes.
    # Separate indices of genotyped and ungenotyped positions.
    idx_typed = np.where(query_h != tskit.MISSING_DATA)[0]
    idx_untyped = np.where(query_h == tskit.MISSING_DATA)[0]
    # Get physical positions of of genotyped and ungenotyped markers.
    pos_typed = pos_all[idx_typed]
    pos_untyped = pos_all[idx_untyped]
    # Get genetic map positions of of genotyped and ungenotyped markers.
    cm_typed = convert_to_genetic_map_positions(pos_typed, genetic_map=genetic_map)
    cm_untyped = convert_to_genetic_map_positions(pos_untyped, genetic_map=genetic_map)
    # Get HMM probabilities at genotyped positions.
    trans_probs = get_transition_probs(cm_typed, h=h, ne=ne)
    mismatch_probs = get_mismatch_probs(len(pos_typed), error_rate=error_rate)
    # Subset haplotypes.
    ref_h_typed = ref_h[idx_typed, :]
    ref_h_untyped = ref_h[idx_untyped, :]
    query_h_typed = query_h[idx_typed]
    # Compute matrices at genotyped positions.
    fwd_mat = compute_forward_matrix(
        ref_h_typed,
        query_h_typed,
        trans_probs,
        mismatch_probs,
        num_alleles=num_alleles,
    )
    bwd_mat = compute_backward_matrix(
        ref_h_typed,
        query_h_typed,
        trans_probs,
        mismatch_probs,
        num_alleles=num_alleles,
    )
    state_mat = compute_state_prob_matrix(fwd_mat, bwd_mat)
    # Interpolate allele probabilities.
    imputed_allele_probs, _ = interpolate_allele_probs(
        state_mat=state_mat,
        ref_h=ref_h_untyped,
        pos_typed=pos_typed,
        pos_untyped=pos_untyped,
        cm_typed=cm_typed,
        cm_untyped=cm_untyped,
        use_threshold=use_threshold,
        return_weights=False,
    )
    imputed_alleles, max_allele_probs = get_map_alleles(imputed_allele_probs)
    return (imputed_alleles, max_allele_probs)


def run_tsimpute(
    ref_ts,
    query_h,
    pos_all,
    *,
    ne=1e6,
    error_rate=1e-4,
    precision=10,
    genetic_map=None,
    use_threshold=False,
):
    """
    Perform interpolation-style imputation, except that the forward and backward
    probability matrices are computed from a tree sequence.

    Reference haplotypes and query haplotype are of size (m + x, h) and (m + x).

    The physical positions of all the markers are an array of size (m + x).

    This produces imputed alleles and their probabilities at the ungenotyped positions
    of the query haplotype.

    The default values for `ne` and `error_rate` are taken from BEAGLE 4.1.

    In an analysis comparing imputation accuracy from precision 6 to 24
    using the FinnGen SiSu dataset (~1k genotyped positions in query haplotypes),
    accuracy was highly similar from 8 to 24 and only drastically worsened at 6.
    Also, in an informal benchmark experiment, the runtime per query haplotype
    improved ~8x, going from precision 22 to 8. This indicates that there is
    a large boost in speed with very little loss in accuracy when precision is 8.
    To be on the safe side, the default value of precision is set to 10.

    Note that BEAGLE 4.1 uses Java float (32-bit) when calculating
    the forward, backward, and hidden state probability matrices.

    TODO: Handle `acgt_alleles` properly.

    :param numpy.ndarray ref_ts: Tree sequence with reference haplotypes.
    :param numpy.ndarray query_h: One query haplotype.
    :param numpy.ndarray pos_all: Physical positions of all the markers (bp).
    :param int ne: Effective population size (default = 1e6).
    :param float error_rate: Allelic error rate (default = 1e-4).
    :param int precision: Precision for running LS HMM (default = 10).
    :param GeneticMap genetic_map: Genetic map (default = None).
    :param bool use_threshold: Set trivial probabilities to 0 if True (default = False).
    :return: Imputed alleles and their probabilities.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    warnings.warn(
        "Check the reference and query haplotypes use the same allele encoding.",
        stacklevel=1,
    )
    h = ref_ts.num_samples  # Number of reference haplotypes.
    # Separate indices of genotyped and ungenotyped positions.
    idx_typed = np.where(query_h != tskit.MISSING_DATA)[0]
    idx_untyped = np.where(query_h == tskit.MISSING_DATA)[0]
    # Set physical positions of genotyped and ungenotyped markers.
    pos_typed = pos_all[idx_typed]
    pos_untyped = pos_all[idx_untyped]
    # Get genetic map positions of genotyped and ungenotyped markers.
    cm_typed = convert_to_genetic_map_positions(pos_typed, genetic_map=genetic_map)
    cm_untyped = convert_to_genetic_map_positions(pos_untyped, genetic_map=genetic_map)
    # Get HMM probabilities at genotyped positions.
    trans_probs = get_transition_probs(cm_typed, h=h, ne=ne)
    mismatch_probs = get_mismatch_probs(len(pos_typed), error_rate=error_rate)
    # Subset haplotypes.
    ref_ts_typed = ref_ts.delete_sites(site_ids=idx_untyped)
    ref_ts_untyped = ref_ts.delete_sites(site_ids=idx_typed)
    ref_h_untyped = ref_ts_untyped.genotype_matrix(alleles=tskit.ALLELES_ACGT)
    query_h_typed = query_h[idx_typed]
    # Get matrices at genotyped positions from tree sequence.
    fwd_mat = _tskit.CompressedMatrix(ref_ts_typed._ll_tree_sequence)
    bwd_mat = _tskit.CompressedMatrix(ref_ts_typed._ll_tree_sequence)
    # WARN: Be careful with argument `acgt_alleles`!!!
    ls_hmm = _tskit.LsHmm(
        ref_ts_typed._ll_tree_sequence,
        recombination_rate=trans_probs,  # Transition probabilities.
        mutation_rate=mismatch_probs,  # Mismatch probabilities.
        acgt_alleles=True,  # TODO: Handle allele encoding properly.
        precision=precision,
    )
    ls_hmm.forward_matrix(query_h_typed.T, fwd_mat)
    ls_hmm.backward_matrix(query_h_typed.T, fwd_mat.normalisation_factor, bwd_mat)
    # TODO: Check that these state probabilities align.
    state_mat = state_mat = np.multiply(fwd_mat.decode(), bwd_mat.decode())
    # Interpolate allele probabilities.
    imputed_allele_probs, _ = interpolate_allele_probs(
        state_mat=state_mat,
        ref_h=ref_h_untyped,
        pos_typed=pos_typed,
        pos_untyped=pos_untyped,
        cm_typed=cm_typed,
        cm_untyped=cm_untyped,
        use_threshold=use_threshold,
        return_weights=False,
    )
    imputed_alleles, max_allele_probs = get_map_alleles(imputed_allele_probs)
    return (imputed_alleles, max_allele_probs)


""" Evaluation metrics and printing of results. """


# Individual-level data.
def compute_individual_scores(
    alleles_1, allele_probs_1, alleles_2, allele_probs_2, ref
):
    """
    Compute genotype probabilities and allele dosages of diploid individuals
    at a position based on posterior marginal allele probabilities.

    Assume that all sites are biallelic. Otherwise, the calculation below is incorrect.
    Note 0 refers to the REF allele and 1 the ALT allele.

    Unphased genotype (or dosage) probabilities are: P(RR), P(RA or AR), P(AA).
    Dosages of the ALT allele are: RR = 0, RA or AR = 1, AA = 2.

    In BEAGLE 4.1 output,
    GP: "Estimated Genotype Probability", and
    DS: "Estimated ALT dose [P(RA) + P(AA)]".

    :param numpy.ndarray alleles_1: Imputed alleles for haplotype 1.
    :param numpy.ndarray allele_probs_1: Imputed allele probabilities for haplotype 1.
    :param numpy.ndarray alleles_2: Imputed alleles for haplotype 2.
    :param numpy.ndarray allele_probs_2: Imputed allele probabilities for haplotype 2.
    :param int ref: Specified REF allele (ACGT encoding).
    :return: Dosage probabilities and dosage scores.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    n = len(alleles_1)  # Number of individuals.
    assert len(alleles_2) == n, "Lengths of alleles differ."
    assert n > 0, "There must be at least one individual."
    assert len(allele_probs_1) == n, "Lengths of alleles and probabilities differ."
    assert len(allele_probs_2) == n, "Lengths of alleles and probabilities differ."
    dosage_probs = np.zeros((n, 3), dtype=np.float64)
    dosage_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ap_hap1_ref = (
            allele_probs_1[i] if alleles_1[i] == ref else 1 - allele_probs_1[i]
        )
        ap_hap1_alt = 1 - ap_hap1_ref
        ap_hap2_ref = (
            allele_probs_2[i] if alleles_2[i] == ref else 1 - allele_probs_2[i]
        )
        ap_hap2_alt = 1 - ap_hap2_ref
        dosage_probs[i, 0] = ap_hap1_ref * ap_hap2_ref  # P(RR)
        dosage_probs[i, 1] = ap_hap1_ref * ap_hap2_alt  # P(RA)
        dosage_probs[i, 1] += ap_hap1_alt * ap_hap2_ref  # P(AR)
        dosage_probs[i, 2] = ap_hap1_alt * ap_hap2_alt  # P(AA)
        dosage_scores[i] = dosage_probs[i, 1] + 2 * dosage_probs[i, 2]
    return (dosage_probs, dosage_scores)


# Site-level data.
def compute_allelic_r_squared(dosage_probs):
    """
    Compute the estimated allelic R^2 at a position from the unphased genotype
    (or dosage) probabilities of a set of diploid individuals.

    Assume that site is biallelic. Otherwise, the calculation below is incorrect.
    Note that 0 refers to REF allele and 1 the ALT allele.

    It is not the true allelic R^2, which needs access to true genotypes to compute.
    The true allelic R^s is the squared correlation between true and imputed ALT dosages.
    It has been shown the true and estimated allelic R-squared are highly correlated.

    In BEAGLE 4.1, it is AR2: "Allelic R-Squared: estimated squared correlation
    between most probable REF dose and true REF dose".
    See `allelicR2` in `R2Estimator.java` of the BEAGLE 4.1 source code.

    See the formulation in the Appendix 1 of Browning & Browning (2009).
    Am J Hum Genet. 84(2): 210–223. doi: 10.1016/j.ajhg.2009.01.005.

    :return: Dosage probabilities and dosage scores.
    :return: Estimated allelic R-squared.
    :rtype: float
    """
    _MIN_R2_DEN = 1e-8
    n = len(dosage_probs)  # Number of individuals.
    assert n > 0, "There must be at least one individual."
    assert dosage_probs.shape[1] == 3, "Three genotypes are considered."
    f = 1 / n
    z = np.argmax(dosage_probs, axis=1)  # Most likely imputed dosage.
    u = dosage_probs[:, 1] + 2 * dosage_probs[:, 2]  # E[X | y_i]
    w = dosage_probs[:, 1] + 4 * dosage_probs[:, 2]  # E[X^2 | y_i]
    cov = np.sum(z * u) - np.sum(z) * np.sum(u) * f
    var_best = np.sum(z**2) - np.sum(z) ** 2 * f
    var_exp = np.sum(w) - np.sum(u) ** 2 * f
    den = var_best * var_exp
    # Minimum of allelic R^2 is zero.
    allelic_rsq = 0 if den < _MIN_R2_DEN else cov**2 / den
    return allelic_rsq


def compute_dosage_r_squared(dosage_probs):
    """
    Compute the dosage R^2 for a position from the unphased genotype (or dosage)
    probabilities of a set of diploid individuals.

    Assume that site is biallelic. Otherwise, the calculation below is incorrect.
    Note that 0 refers to REF allele and 1 the ALT allele.

    In BEAGLE 4.1, DR2: "Dosage R-Squared: estimated squared correlation
    between estimated REF dose [P(RA) + 2 * P(RR)] and true REF dose".
    See `doseR2` in `R2Estimator.java` of the BEAGLE 4.1 source code.

    :return: Dosage probabilities and dosage scores.
    :return: Dosage R-squared.
    :rtype: float
    """
    _MIN_R2_DEN = 1e-8
    n = len(dosage_probs)  # Number of individuals.
    assert n > 0, "There must be at least one individual."
    assert dosage_probs.shape[1] == 3, "Three genotypes are considered."
    f = 1 / n
    u = dosage_probs[:, 1] + 2 * dosage_probs[:, 2]  # E[X | y_i].
    w = dosage_probs[:, 1] + 4 * dosage_probs[:, 2]  # E[X^2 | y_i].
    c = np.sum(u) ** 2 * f
    num = np.sum(u**2) - c
    if num < 0:
        num = 0
    den = np.sum(w) - c
    dosage_rsq = 0 if den < _MIN_R2_DEN else num / den
    return dosage_rsq


def compute_allele_frequency(
    alleles_1,
    allele_probs_1,
    alleles_2,
    allele_probs_2,
    allele,
):
    """
    Estimate the frequency of a specified allele at a position from allele probabilities
    of a set of diploid individuals.

    Assume that site is biallelic. Otherwise, the calculation below is incorrect.

    Input are the imputed alleles and their probabilities at a position.

    In BEAGLE 4.1, AF: "Estimated ALT Allele Frequencies".
    See `printInfo` in `VcfRecBuilder.java` of the BEAGLE 4.1 source code.

    See the note in "Standardized Allele-Frequency Error" in Browning & Browning (2009).
    Am J Hum Genet. 84(2): 210–223. doi: 10.1016/j.ajhg.2009.01.005.

    :param numpy.ndarray alleles_1: Imputed alleles for haplotype 1.
    :param numpy.ndarray allele_probs_1: Imputed allele probabilities for haplotype 1.
    :param numpy.ndarray alleles_2: Imputed alleles for haplotype 2.
    :param numpy.ndarray allele_probs_2: Imputed allele probabilities for haplotype 2.
    :param int allele: Specified allele (ACGT encoding).
    :return: Estimated allele frequency.
    :rtype: float
    """
    n = len(alleles_1)  # Number of individuals.
    assert len(alleles_2) == n, "Lengths of alleles differ."
    assert n > 0, "There must be at least one individual."
    assert len(allele_probs_1) == n, "Lengths of alleles and probabilities differ."
    assert len(allele_probs_2) == n, "Lengths of alleles and probabilities differ."
    cum_ap_hap1 = np.sum(allele_probs_1[alleles_1 == allele])
    cum_ap_hap2 = np.sum(allele_probs_2[alleles_2 == allele])
    # See `printInfo` in `VcfRecBuilder.java` in BEAGLE 4.1 source code.
    est_af = (cum_ap_hap1 + cum_ap_hap2) / (2 * n)
    return est_af


def write_vcf(impdata, out_file, *, chr_name="1", print_gp=False, decimals=2):
    """
    Print imputation results in VCF format, following the output of BEAGLE 4.1.

    TODO: Print VCF records for genotyped sites.

    :param ImpData impdata: Object containing imputation data.
    :param str out_file: Path to output VCF file.
    :param str chr_name: Chromosome name (default = "1").
    :param bool print_gp: Print genotype probabilities if True (default = False).
    :param int decimals: Number of decimal places to print (default = 2).
    :return: None
    :rtype: None
    """
    _HEADER = [
        "##fileformat=VCFv4.2",
        f"##filedata={__DATE__}",
        f"##source=tsimpute (version {__VERSION__})",
        "##INFO=<ID=AF,Number=A,Type=Float,"
        + 'Description="Estimated ALT Allele Frequencies">',
        "##INFO=<ID=AR2,Number=1,Type=Float,"
        + 'Description="Allelic R-Squared: estimated squared correlation '
        + 'between most probable REF dose and true REF dose">',
        "##INFO=<ID=DR2,Number=1,Type=Float,"
        + 'Description="Dosage R-Squared: estimated squared correlation '
        + 'between estimated REF dose [P(RA) + 2*P(RR)] and true REF dose">',
        '##INFO=<ID=IMP,Number=0,Type=Flag,Description="Imputed marker">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##FORMAT=<ID=DS,Number=A,Type=Float,"
        + 'Description="estimated ALT dose [P(RA) + P(AA)]">',
        "##FORMAT=<ID=GL,Number=G,Type=Float,"
        + 'Description="Log10-scaled Genotype Likelihood">',
        "##FORMAT=<ID=GP,Number=G,Type=Float,"
        + 'Description="Estimated Genotype Probability">',
    ]
    _COL_NAMES = [
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
    ]
    with open(out_file, "w") as f:
        # Add header with metadata and definitions.
        for line in _HEADER:
            f.write(line + "\n")
        # Add column names.
        col_str = "#"
        col_str += "\t".join(_COL_NAMES)
        col_str += "\t"
        col_str += "\t".join(impdata.individual_names)
        f.write(col_str + "\n")
        # Add VCF records.
        is_imputed = True
        for i in range(impdata.num_sites):
            a1, ap1, a2, ap2 = impdata.get_alleles_at_site(i)
            gt_probs, dosages = compute_individual_scores(a1, ap1, a2, ap2)
            line_str = chr_name
            line_str += "\t"
            line_str += str(int(impdata.site_pos[i]))
            line_str += "\t"
            line_str += str(i)
            line_str += "\t"
            REF = impdata.get_ref_allele_at_site(i)
            ALT = impdata.get_alt_allele_at_site(i)
            line_str += tskit.ALLELES_ACGT[REF]
            line_str += "\t"
            line_str += tskit.ALLELES_ACGT[ALT]
            line_str += "\t"
            # QUAL field
            # '.' denotes missing.
            line_str += "."
            line_str += "\t"
            # FILTER field
            line_str += "PASS"
            line_str += "\t"
            # INFO field
            ar2 = compute_allelic_r_squared(gt_probs)
            dr2 = compute_dosage_r_squared(gt_probs)
            af = compute_allele_frequency(a1, ap1, a2, ap2, allele=1)
            ar2 = round(ar2, decimals)
            dr2 = round(dr2, decimals)
            af = round(af, decimals)
            info_str = f"AR2={ar2};DR2={dr2};AF={af}"
            if is_imputed:
                info_str += ";" + "IMP"
            line_str += info_str
            line_str += "\t"
            # FORMAT field
            line_str += "GT:DS"
            if print_gp:
                line_str += ":" + "GP"
            line_str += "\t"
            # DATA fields
            data_str = ""
            for j in range(impdata.num_individuals):
                gt_a1 = "0" if a1[j] == REF else "1"
                gt_a2 = "0" if a2[j] == REF else "1"
                data_str += gt_a1 + "|" + gt_a2 + ":"
                data_str += str(round(dosages[j], decimals))
                if print_gp:
                    data_str += ":"
                    data_str += ",".join(
                        [str(round(gt_probs[j, k], decimals)) for k in range(3)]
                    )
                if j < impdata.num_individuals - 1:
                    data_str += "\t"
            line_str += data_str
            f.write(line_str + "\n")
