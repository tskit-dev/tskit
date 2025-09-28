# MIT License
#
# Copyright (c) 2023-2024 Tskit Developers
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
Test cases for two-locus statistics
"""
import contextlib
import io
from dataclasses import dataclass
from itertools import combinations_with_replacement
from itertools import permutations
from itertools import product
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple
from typing import Union

import msprime
import numpy as np
import pytest

import tskit
import tskit.util as util
from tests import tsutil
from tests.tsutil import get_example_tree_sequences


@contextlib.contextmanager
def suppress_overflow_div0_warning():
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        yield


class BitSet:
    """BitSet object, which stores values in arrays of unsigned integers.
    The rows represent all possible values a bit can take, and the rows
    represent each item that can be stored in the array.

    :param num_bits: The number of values that a single row can contain.
    :param length: The number of rows.
    """

    DTYPE = np.uint32  # Data type to be stored in the bitset
    CHUNK_SIZE = DTYPE(32)  # Size of integer field to store the data in

    def __init__(self: "BitSet", num_bits: int, length: int) -> None:
        self.row_len = num_bits // self.CHUNK_SIZE
        self.row_len += 1 if num_bits % self.CHUNK_SIZE else 0
        self.row_len = int(self.row_len)
        self.data = np.zeros(self.row_len * length, dtype=self.DTYPE)

    def intersect(
        self: "BitSet", self_row: int, other: "BitSet", other_row: int, out: "BitSet"
    ) -> None:
        """Intersect a row from the current array instance with a row from
        another BitSet and store it in an output bit array of length 1.

        NB: we don't specify the row in the output array, it is expected
        to be length 1.

        :param self_row: Row from the current array instance to be intersected.
        :param other: Other BitSet to intersect with.
        :param other_row: Row from the other BitSet instance.
        :param out: BitArray to store the result.
        """
        self_offset = self_row * self.row_len
        other_offset = other_row * self.row_len

        for i in range(self.row_len):
            out.data[i] = self.data[i + self_offset] & other.data[i + other_offset]

    def difference(
        self: "BitSet", self_row: int, other: "BitSet", other_row: int
    ) -> None:
        """Take the difference between the current array instance and another
        array instance. Store the result in the specified row of the current
        instance.

        :param self_row: Row from the current array from which to subtract.
        :param other: Other BitSet to subtract from the current instance.
        :param other_row: Row from the other BitSet instance.
        """
        self_offset = self_row * self.row_len
        other_offset = other_row * self.row_len

        for i in range(self.row_len):
            self.data[i + self_offset] &= ~(other.data[i + other_offset])

    def union(self: "BitSet", self_row: int, other: "BitSet", other_row: int) -> None:
        """Take the union between the current array instance and another
        array instance. Store the result in the specified row of the current
        instance.

        :param self_row: Row from the current array with which to union.
        :param other: Other BitSet to union with the current instance.
        :param other_row: Row from the other BitSet instance.
        """
        self_offset = self_row * self.row_len
        other_offset = other_row * self.row_len

        for i in range(self.row_len):
            self.data[i + self_offset] |= other.data[i + other_offset]

    def add(self: "BitSet", row: int, bit: int) -> None:
        """Add a single bit to the row of a bit array

        :param row: Row to be modified.
        :param bit: Bit to be added.
        """
        offset = row * self.row_len
        i = bit // self.CHUNK_SIZE
        self.data[i + offset] |= self.DTYPE(1) << (bit - (self.CHUNK_SIZE * i))

    def get_items(self: "BitSet", row: int) -> Generator[int, None, None]:
        """Get the items stored in the row of a bitset
        Uses a de Bruijn sequence lookup table to determine the lowest bit set.
        See the wikipedia article for more info: https://w.wiki/BYiF

        :param row: Row from the array to list from.
        :returns: A generator of integers stored in the array.
        """
        lookup = [0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27,
                  13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9]  # fmt: skip
        m = np.uint32(125613361)
        offset = row * self.row_len
        for i in range(self.row_len):
            v = self.data[i + offset]
            if v == 0:
                continue
            else:
                # v & -v operations rely on integer overflow
                with np.errstate(over="ignore"):
                    lsb = v & -v  # isolate the least significant bit
                    while lsb:  # while there are bits remaining
                        yield lookup[(lsb * m) >> 27] + (i * self.CHUNK_SIZE)
                        v ^= lsb  # unset the lsb
                        lsb = v & -v

    def contains(self: "BitSet", row: int, bit: int) -> bool:
        """Test if a bit is contained within a bit array row

        :param row: Row to test.
        :param bit: Bit to check.
        :returns: True if the bit is set in the row, else false.
        """
        i = bit // self.CHUNK_SIZE
        offset = row * self.row_len
        return bool(
            self.data[i + offset] & (self.DTYPE(1) << (bit - (self.CHUNK_SIZE * i)))
        )

    def count(self: "BitSet", row: int) -> int:
        """Count all of the set bits in a specified row. Uses a SWAR
        algorithm to count in parallel with a constant number (12) of operations.

        NB: we have to cast all values to our unsigned dtype to avoid type promotion

        Details here:
        # https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

        :param row: Row to count.
        :returns: Count of all of the set bits.
        """
        count = 0
        offset = row * self.row_len
        D = self.DTYPE

        for i in range(offset, offset + self.row_len):
            v = self.data[i]
            v = v - ((v >> D(1)) & D(0x55555555))
            v = (v & D(0x33333333)) + ((v >> D(2)) & D(0x33333333))
            # this operation relies on integer overflow
            with np.errstate(over="ignore"):
                count += ((v + (v >> D(4)) & D(0xF0F0F0F)) * D(0x1010101)) >> D(24)

        return count

    def count_naive(self: "BitSet", row: int) -> int:
        """Naive counting algorithm implementing the same functionality as the count
        method. Useful for testing correctness, uses the same number of operations
        as set bits.

        Details here:
        # https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive

        :param row: Row to count.
        :returns: Count of all of the set bits.
        """
        count = 0
        offset = row * self.row_len

        for i in range(offset, offset + self.row_len):
            v = self.data[i]
            while v:
                v &= v - self.DTYPE(1)
                count += self.DTYPE(1)
        return int(count)


def norm_hap_weighted(
    result_dim: int,
    hap_weights: np.ndarray,
    n_a: int,
    n_b: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    """Create a vector of normalizing coefficients, length of the number of
    sample sets. In this normalization strategy, we weight each allele's
    statistic by the proportion of the haplotype present.

    :param result_dim: Number of dimensions in output. Dependent on arity of stat.
    :param hap_weights: Proportion of each two-locus haplotype.
    :param n_a: Number of alleles at the A locus.
    :param n_b: Number of alleles at the B locus.
    :param result: Result vector to store the normalizing coefficients in.
    :param params: Params of summary function.
    """
    del n_a, n_b  # handle unused params
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(result_dim):
        n = sample_set_sizes[k]
        result[k] = hap_weights[0, k] / n


def norm_hap_weighted_ij(
    result_dim: int,
    hap_weights: np.ndarray,
    n_a: int,
    n_b: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    """
    Create a vector of normalizing coefficients, length of the number of
    index tuples. Each allele's statistic will be weighted by the average
    of the proportion of AB haplotypes in each population present in the
    index tuple.

    :param result_dim: Number of dimensions in output. Dependent on arity of stat.
    :param hap_weights: Proportion of each two-locus haplotype.
    :param n_a: Number of alleles at the A locus.
    :param n_b: Number of alleles at the B locus.
    :param result: Result vector to store the normalizing coefficients in.
    :param params: Params of summary function.
    """
    del n_a, n_b  # handle unused params
    sample_set_sizes = params["sample_set_sizes"]
    set_indexes = params["set_indexes"]

    for k in range(result_dim):
        i = set_indexes[k][0]
        j = set_indexes[k][1]
        ni = sample_set_sizes[i]
        nj = sample_set_sizes[j]
        wAB_i = hap_weights[0, i]
        wAB_j = hap_weights[0, j]
        result[k] = (wAB_i + wAB_j) / (ni + nj)
        # result[k] = (wAB_i / ni / 2) + (wAB_j / nj / 2)


def norm_total_weighted(
    result_dim: int,
    hap_weights: np.ndarray,
    n_a: int,
    n_b: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    """Create a vector of normalizing coefficients, length of the number of
    sample sets. In this normalization strategy, we weight each allele's
    statistic by the product of the allele frequencies

    :param result_dim: Number of dimensions in output. Dependent on arity of stat.
    :param hap_weights: Proportion of each two-locus haplotype.
    :param n_a: Number of alleles at the A locus.
    :param n_b: Number of alleles at the B locus.
    :param result: Result vector to store the normalizing coefficients in.
    :param params: Params of summary function.
    """
    del hap_weights, params  # handle unused params
    for k in range(result_dim):
        result[k] = 1 / (n_a * n_b)


def check_order_bounds_dups(values, max_value):
    """Validate the specified values.

    We require that values are:

    1) Within the boundaries of the max value in the tree sequence
    2) Sorted
    3) Non-repeating

    Raises an exception if any error is found.

    :param values: 1d array of values to validate.
    :param max_value: The upper bound for the provided values.
    """
    if len(values) == 0:
        return
    i = 0
    for i in range(len(values) - 1):
        if values[i] < 0 or values[i] >= max_value:
            raise ValueError(f"Value out of bounds: {values[i]}")
        if values[i] >= values[i + 1]:
            raise ValueError(f"Value not sorted: {values[i], values[i + 1]}")
    if values[-1] < 0 or values[-1] >= max_value:
        raise ValueError(f"Value out of bounds: {values[i + 1]}")


def get_site_row_col_indices(
    row_sites: np.ndarray, col_sites: np.ndarray
) -> Tuple[List[int], List[int], List[int]]:
    """Co-iterate over the row and column sites, keeping a sorted union of
    site values and an index into the unique list of sites for both the row
    and column sites. This function produces a list of sites of interest and
    row and column indexes into this list of sites.

    NB: This routine requires that the site lists are sorted and deduplicated.

    :param row_sites: List of sites that will be represented in the output
                      matrix rows.
    :param col_sites: List of sites that will be represented in the output
                      matrix columns.
    :returns: Tuple of lists of sites, row, and column indices.
    """
    r = 0
    c = 0
    s = 0
    sites = []
    col_idx = []
    row_idx = []

    while r < len(row_sites) and c < len(col_sites):
        if row_sites[r] < col_sites[c]:
            sites.append(row_sites[r])
            row_idx.append(s)
            s += 1
            r += 1
        elif row_sites[r] > col_sites[c]:
            sites.append(col_sites[c])
            col_idx.append(s)
            s += 1
            c += 1
        else:
            sites.append(row_sites[r])
            row_idx.append(s)
            col_idx.append(s)
            s += 1
            r += 1
            c += 1
    while r < len(row_sites):
        sites.append(row_sites[r])
        row_idx.append(s)
        s += 1
        r += 1
    while c < len(col_sites):
        sites.append(col_sites[c])
        col_idx.append(s)
        s += 1
        c += 1

    return sites, row_idx, col_idx


def get_all_samples_bits(num_samples: int) -> BitSet:
    """Get the bits for all samples in the tree sequence. This is achieved
    by creating a length 1 bitset and adding every sample's bit to it.

    :param num_samples: Number of samples contained in the tree sequence.
    :returns: Length 1 BitSet containing all samples in the tree sequence.
    """
    all_samples = BitSet(num_samples, 1)
    for i in range(num_samples):
        all_samples.add(0, i)
    return all_samples


def get_allele_samples(
    site: tskit.Site, site_offset: int, mut_samples: BitSet, allele_samples: BitSet
) -> int:
    """Given a BitSet that has been arranged so that we have every sample under
    a given mutation's node, create the final output where we know which samples
    should belong under each mutation, considering the mutation's parentage,
    back mutations, and ancestral state.

    To this end, we iterate over each mutation and store the samples under the
    focal mutation in the output BitSet (allele_samples). Then, we check the
    parent of the focal mutation (either a mutation or the ancestral allele),
    and we subtract the samples in the focal mutation from the parent allele's
    samples.

    :param site: Focal site for which to adjust mutation data.
    :param site_offset: Offset into allele_samples for our focal site.
    :param mut_samples: BitSet containing the samples under each mutation in the
                        focal site.
    :param allele_samples: Output BitSet, initially passed in with all of the
                           tree sequence samples set in the ancestral allele
                           state.
    :returns: number of alleles actually encountered (adjusting for back-mutation).
    """
    alleles = []
    num_alleles = 1
    alleles.append(site.ancestral_state)

    for m, mut in enumerate(site.mutations):
        try:
            allele = alleles.index(mut.derived_state)
        except ValueError:
            allele = len(alleles)
            alleles.append(mut.derived_state)
            num_alleles += 1
        allele_samples.union(allele + site_offset, mut_samples, m)
        # now to find the parent allele from which we must subtract
        alt_allele_state = site.ancestral_state
        if mut.parent != tskit.NULL:
            parent_mut = site.mutations[mut.parent - site.mutations[0].id]
            alt_allele_state = parent_mut.derived_state
        alt_allele = alleles.index(alt_allele_state)
        # subtract focal allele's samples from the alt allele
        allele_samples.difference(
            alt_allele + site_offset, allele_samples, allele + site_offset
        )

    return num_alleles


def get_mutation_samples(
    ts: tskit.TreeSequence, sites: List[int], sample_index_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, BitSet]:
    """For a given set of sites, generate a BitSet of all samples posessing
    each allelic state for each site. This includes the ancestral state, along
    with any mutations contained in the site.

    We achieve this goal by starting at the tree containing the first site in
    our list, then we walk along each tree until we've encountered the last
    tree containing the last site in our list. Along the way, we perform a
    preorder traversal from the node of each mutation in a given site, storing
    the samples under that particular node. After we've stored all of the samples
    for each allele at a site, we adjust each allele's samples by removing
    samples that have a different allele at a child mutation down the tree (see
    get_allele_samples for more details).

    We also gather some ancillary data while we iterate over the sites: the
    number of alleles for each site, and the offset of each site. The number of
    alleles at each site includes the count of mutations + the ancestral allele.
    The offeset for each site indicates how many array entries we must skip (ie
    how many alleles exist before a specific site's entry) in order to address
    the data for a given site.

    :param ts: Tree sequence to gather data from.
    :param sites: Subset of sites to consider when gathering data.
    :param sample_index_map: Mapping from node id to sample id
    :returns: Tuple of the number of alleles per site, site offsets, and the
              BitSet of all samples in each allelic state.
    """
    num_alleles = np.zeros(len(sites), dtype=np.uint64)
    site_offsets = np.zeros(len(sites), dtype=np.uint64)
    all_samples = get_all_samples_bits(ts.num_samples)
    allele_samples = BitSet(
        ts.num_samples, sum(len(ts.site(i).mutations) + 1 for i in sites)
    )

    site_offset = 0
    site_idx = 0
    for site_idx, site_id in enumerate(sites):
        site = ts.site(site_id)
        tree = ts.at(site.position)
        # initialize the ancestral allele with all samples
        allele_samples.union(site_offset, all_samples, 0)
        # store samples for each mutation in mut_samples
        mut_samples = BitSet(ts.num_samples, len(site.mutations))
        for m, mut in enumerate(site.mutations):
            for node in tree.preorder(mut.node):
                if ts.node(node).is_sample():
                    mut_samples.add(m, sample_index_map[node])
        # account for mutation parentage, subtract samples from mutation parents
        num_alleles[site_idx] = get_allele_samples(
            site, site_offset, mut_samples, allele_samples
        )
        # increment the offset for ancestral + mutation alleles
        site_offsets[site_idx] = site_offset
        site_offset += len(site.mutations) + 1

    return num_alleles, site_offsets, allele_samples


SummaryFunc = Callable[[int, np.ndarray, int, np.ndarray, Dict[str, Any]], None]
NormFunc = Callable[[int, np.ndarray, int, int, np.ndarray, Dict[str, Any]], None]


def compute_general_two_site_stat_result(
    row_site_offset: int,
    col_site_offset: int,
    num_row_alleles: int,
    num_col_alleles: int,
    num_samples: int,
    allele_samples: BitSet,
    state_dim: int,
    sample_sets: BitSet,
    result_dim: int,
    func: SummaryFunc,
    norm_func: NormFunc,
    params: Dict[str, Any],
    polarised: bool,
    result: np.ndarray,
) -> None:
    """For a given pair of sites, compute the summary statistic for the allele
    frequencies for each allelic state of the two pairs.

    :param row_site_offset: Offset of the row site's data in the allele_samples.
    :param col_site_offset: Offset of the col site's data in the allele_samples.
    :param num_row_alleles: Number of alleles in the row site.
    :param num_col_alleles: Number of alleles in the col site.
    :param num_samples: Number of samples in tree sequence.
    :param allele_samples: BitSet containing the samples with each allelic state
                           for each site of interest.
    :param state_dim: Number of sample sets.
    :param sample_sets: BitSet of sample sets to be intersected with the samples
                        contained within each allele.
    :param result_dim: Number of dimensions in output. Dependent on arity of stat.
    :param func: Summary function used to compute each two-locus statistic.
    :param norm_func: Function used to generate the normalization coefficients
                      for each statistic.
    :param params: Parameters to pass to the norm and summary function.
    :param polarised: If true, skip the computation of the statistic for the
                      ancestral state.
    :param result: Vector of the results matrix to populate. We will produce one
                   value per sample set, hence the vector of length state_dim.
    """
    ss_A_samples = BitSet(num_samples, 1)
    ss_B_samples = BitSet(num_samples, 1)
    ss_AB_samples = BitSet(num_samples, 1)
    AB_samples = BitSet(num_samples, 1)
    weights = np.zeros((3, state_dim), np.float64)
    norm = np.zeros(result_dim, np.float64)
    result_tmp = np.zeros(result_dim, np.float64)

    polarised_val = 1 if polarised else 0
    for mut_a in range(polarised_val, num_row_alleles):
        a = int(mut_a + row_site_offset)
        for mut_b in range(polarised_val, num_col_alleles):
            b = int(mut_b + col_site_offset)
            allele_samples.intersect(a, allele_samples, b, AB_samples)
            for k in range(state_dim):
                allele_samples.intersect(a, sample_sets, k, ss_A_samples)
                allele_samples.intersect(b, sample_sets, k, ss_B_samples)
                AB_samples.intersect(0, sample_sets, k, ss_AB_samples)

                w_AB = ss_AB_samples.count(0)
                w_A = ss_A_samples.count(0)
                w_B = ss_B_samples.count(0)

                weights[0, k] = w_AB
                weights[1, k] = w_A - w_AB  # w_Ab
                weights[2, k] = w_B - w_AB  # w_aB

            func(state_dim, weights, result_dim, result_tmp, params)

            norm_func(
                result_dim,
                weights,
                num_row_alleles - polarised_val,
                num_col_alleles - polarised_val,
                norm,
                params,
            )

            for k in range(result_dim):
                result[k] += result_tmp[k] * norm[k]


def two_site_count_stat(
    ts: tskit.TreeSequence,
    func: SummaryFunc,
    norm_func: NormFunc,
    result_dim: int,
    num_sample_sets: int,
    sample_set_sizes: np.ndarray,
    sample_sets: BitSet,
    sample_index_map: np.ndarray,
    row_sites: np.ndarray,
    col_sites: np.ndarray,
    indexes: np.ndarray,
    polarised: bool,
) -> np.ndarray:
    """Outer function that generates the high-level intermediates used in the
    computation of our two-locus statistics. First, we compute the row and
    column indices for our unique list of sites, then we get each sample for
    each allele in our list of specified sites.

    With those intermediates in hand, we iterate over the row and column indices
    to compute comparisons between each of the specified lists of sites. We pass
    a vector of results to the computation, which will compute a single result
    for each sample set, inserting that into our result matrix.

    :param ts: Tree sequence to gather data from.
    :param func: Function used to compute each two-locus statistic.
    :param norm_func: Function used to generate the normalization coefficients
                      for each statistic.
    :param result_dim: The dimensions of the output array. For one-way stats,
                       this will be the number of sample sets. For two-way stats,
                       the number of index tuples.
    :param num_sample_sets: Number of sample sets that we will consider.
    :param sample_set_sizes: Number of samples in each sample set.
    :param sample_sets: BitSet of samples to compute stats for. We will only
                        consider these samples in our computations, resulting
                        in stats that are computed on subsets of the samples
                        on the tree sequence.
    :param sample_index_map: Mapping from node id to sample id
    :param row_sites: Sites contained in the rows of the output matrix.
    :param col_sites: Sites contained in the columns of the output matrix.
    :param indexes: List of sample set indexes on which to compute statistics. The
                    arity (and hence the length of each index group) is dictated
                    by the summary function.
    :param polarised: If true, skip the computation of the statistic for the
                      ancestral state.
    :returns: 3D array of results, dimensions (sample_sets, row_sites, col_sites).
    """
    params = {"sample_set_sizes": sample_set_sizes, "set_indexes": indexes}
    result = np.zeros((result_dim, len(row_sites), len(col_sites)), dtype=np.float64)

    state_dim = num_sample_sets

    sites, row_idx, col_idx = get_site_row_col_indices(row_sites, col_sites)
    num_alleles, site_offsets, allele_samples = get_mutation_samples(
        ts, sites, sample_index_map
    )

    for row, row_site in enumerate(row_idx):
        for col, col_site in enumerate(col_idx):
            compute_general_two_site_stat_result(
                site_offsets[row_site],
                site_offsets[col_site],
                num_alleles[row_site],
                num_alleles[col_site],
                ts.num_samples,
                allele_samples,
                state_dim,
                sample_sets,
                result_dim,
                func,
                norm_func,
                params,
                polarised,
                result[:, row, col],
            )

    return result


def get_index_repeats(indices):
    """In a list of indices, find the repeat values. The first value
    is offset by the first index and ranges to the last index.
    For instance, [4, 4, 5, 6, 8] becomes [2, 1, 1, 0, 1].
    The list must be sorted and ordered.

    :param indices: List of indices to count
    :returns: Counts of index repeats
    """
    counts = np.zeros(indices[-1] - indices[0] + 1, dtype=np.int32)
    idx = indices[0]
    count = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1]:
            count += 1
        else:
            counts[idx - indices[0]] = count
            count = 1
            idx = indices[i]
    counts[idx - indices[0]] = count
    return counts


def two_branch_count_stat(
    ts: tskit.TreeSequence,
    func: SummaryFunc,
    norm_func,
    state_dim: int,
    result_dim: int,
    sample_set_sizes: np.ndarray,
    sample_sets: BitSet,
    sample_index_map: np.ndarray,
    row_trees: np.ndarray,
    col_trees: np.ndarray,
    indexes: np.ndarray,
    polarised: bool,
) -> np.ndarray:
    """
    Compute a tree X tree LD matrix by walking along the tree sequence and
    computing haplotype counts. This method incrementally adds and removes
    branches from a tree sequence and updates the stat based on sample additions
    and removals. We bifurcate the tree with a given branch on each locus and
    intersect the samples under each branch to produce haplotype counts. It
    is possible to subset the output matrix with genomic positions. Positions
    lying on the same tree will receive the same LD value in the output matrix.

    :param ts: Tree sequence to gather data from.
    :param func: Function used to compute each two-locus statistic.
    :param norm_func: Not (YET) applicable for branch stats: TODO?
    :param state_dim: Number of sample sets.
    :param result_dim: The dimensions of the output array. For one-way stats,
                       this will be the number of sample sets. For two-way stats,
                       the number of index tuples.
    :param sample_set_sizes: Number of samples in each sample set.
    :param sample_sets: BitSet of samples to compute stats for. We will only
                        consider these samples in our computations, resulting
                        in stats that are computed on subsets of the samples
                        on the tree sequence.
    :param sample_index_map: Mapping from node id to sample id
    :param row_trees: Trees contained in the rows of the output matrix (repeats ok)
    :param col_trees: Trees contained in the rows of the output matrix (repeats ok)
    :param indexes: List of sample set indexes on which to compute statistics. The
                    arity (and hence the length of each index group) is dictated
                    by the summary function.
    :param polarised: If true, skip the computation of the statistic for the
                      ancestral state.
    :returns: 3D array of results, dimensions (sample_sets, row_sites, col_sites).
    """
    params = {"sample_set_sizes": sample_set_sizes, "set_indexes": indexes}
    result = np.zeros((result_dim, len(row_trees), len(col_trees)), dtype=np.float64)
    row_repeats = get_index_repeats(row_trees)
    col_repeats = get_index_repeats(col_trees)

    stat = np.zeros(result_dim, dtype=np.float64)
    # State is initialized at tree -1
    l_state = TreeState(ts, sample_sets, state_dim, sample_index_map)
    r_state = TreeState(ts, sample_sets, state_dim, sample_index_map)

    # Even if we're skipping trees, we must iterate over the range to keep the
    # running total of the statistic consistent.
    row = 0
    for r in range(row_trees[-1] + 1 - row_trees[0]):
        # zero out stat and r_state at the beginning of each row
        stat = np.zeros_like(stat)
        r_state = TreeState(ts, sample_sets, state_dim, sample_index_map)
        l_state.advance(r + row_trees[0])
        # use null TreeState to advance l_state, conveniently we just zerod r_state
        _, l_state = compute_branch_stat(
            ts, func, stat, params, state_dim, result_dim, r_state, l_state
        )
        col = 0
        for c in range(col_trees[-1] + 1 - col_trees[0]):
            r_state.advance(c + col_trees[0])
            stat, r_state = compute_branch_stat(
                ts, func, stat, params, state_dim, result_dim, l_state, r_state
            )
            # Fill in repeated values for all sample sets
            for i in range(row_repeats[r]):
                for j in range(col_repeats[c]):
                    result[:, i + row, j + col] = stat
            col += col_repeats[c]
        row += row_repeats[r]
    return result


def sample_sets_to_bit_array(
    ts: tskit.TreeSequence, sample_sets: Union[List[List[int]], List[np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, BitSet]:
    """Convert the list of sample ids to a bit array. This function takes
    sample identifiers and maps them to their enumerated integer values, then
    stores these values in a bit array. We produce a BitArray and a numpy
    array of integers that specify how many samples there are in each sample set.

    NB: this function's type signature is of type integer, but I believe this
        could be expanded to Any, currently untested so the integer
        specification remains.

    :param ts: Tree sequence to gather data from.
    :param sample_sets: List of sample identifiers to store in bit array.
    :returns: Tuple containing numpy array of sample set sizes and the sample
              set BitSet.
    """
    sample_sets_bits = BitSet(ts.num_samples, len(sample_sets))
    sample_index_map = -np.ones(ts.num_nodes, dtype=np.int32)
    sample_set_sizes = np.zeros(len(sample_sets), dtype=np.uint64)

    sample_count = 0
    for node in ts.nodes():
        if node.flags & tskit.NODE_IS_SAMPLE:
            sample_index_map[node.id] = sample_count
            sample_count += 1

    for k, sample_set in enumerate(sample_sets):
        sample_set_sizes[k] = len(sample_set)
        for sample in sample_set:
            sample_index = sample_index_map[sample]
            if sample_index == tskit.NULL:
                raise ValueError(f"Sample out of bounds: {sample}")
            if sample_sets_bits.contains(k, sample_index):
                raise ValueError(f"Duplicate sample detected: {sample}")
            sample_sets_bits.add(k, sample_index)

    return sample_index_map, sample_set_sizes, sample_sets_bits


def positions_to_tree_indices(bp, positions):
    """Given a set of breakpoints and positions, provide an array of tree
    indices that correspond with positions. We have already validated that the
    bounds of the positions are correct and that they are sorted, and
    deduplicated.

    :param bp: Breakpoints of the tree sequence
    :param positions: Positions to search over
    :returns: Array of tree indices
    """
    tree_idx = 0
    tree_indices = -np.ones_like(positions, dtype=np.int32)

    for i in range(len(positions)):
        while bp[tree_idx + 1] <= positions[i]:
            tree_idx += 1
        tree_indices[i] = tree_idx

    return tree_indices


def two_locus_count_stat(
    ts,
    summary_func,
    norm_func,
    polarised,
    mode,
    sites=None,
    positions=None,
    sample_sets=None,
    indexes=None,
):
    """Outer wrapper for two site general stat functionality. Perform some input
    validation, get the site index and allele state, then compute the LD matrix.

    :param ts: Tree sequence to gather data from.
    :param summary_func: Function used to compute each two-locus statistic.
    :param norm_func: Function used to generate the normalization coefficients
                      for each statistic.
    :param polarised: If true, skip the computation of the statistic for the
                      ancestral state.
    :param mode: Whether or not to compute "site" or "branch" statistics.
    :param sites: List of two lists containing [row_sites, column_sites].
    :param positions: List of two lists containing [row_positions, col_positions],
                      which are genomic positions to compute LD on.
    :param sample_sets: List of lists of samples to compute stats for. We will
                        only consider these samples in our computations,
                        resulting in stats that are computed on subsets of the
                        samples on the tree sequence.
    :param indexes: List of sample set indexes on which to compute statistics. The
                    arity (and hence the length of each index group) is dictated
                    by the summary function.
    :returns: 3d numpy array containing LD for (sample_set,row_site,column_site)
              unless one or no sample sets are specified, then 2d array
              containing LD for (row_site,column_site).
    """
    if sample_sets is None:
        sample_sets = ts.samples()

    drop_dim = False
    if indexes is None:
        try:
            sample_sets = np.array(sample_sets, dtype=np.uint64)
        except ValueError:
            pass
        else:
            if sample_sets.ndim == 1:
                sample_sets = [sample_sets]
                drop_dim = True
        result_dim = len(sample_sets)
    else:
        indexes = util.safe_np_int_cast(indexes, np.int32)
        if len(indexes.shape) == 1:
            indexes = indexes.reshape((1, indexes.shape[0]))
            drop_dim = True
        if len(indexes.shape) != 2 or indexes.shape[1] != 2:
            raise ValueError(
                "Indexes must be convertable to a 2D numpy array with 2 columns"
            )
        result_dim = len(indexes)

    sample_index_map, ss_sizes, ss_bits = sample_sets_to_bit_array(ts, sample_sets)
    num_sample_sets = len(ss_sizes)
    # If indexes are specified, we are using two-way statistics
    if indexes is not None:
        indexes = tskit.util.safe_np_int_cast(indexes, np.int32)
        idx_lens = {len(i) for i in indexes}
        if idx_lens != {2}:
            raise ValueError(
                f"Sample set indexes must be length 2, lengths: {idx_lens}"
            )
        check_sample_stat_inputs(num_sample_sets, 2, result_dim, indexes)
    if mode == "site":
        if positions is not None:
            raise ValueError("Cannot specify positions in site mode")
        if sites is None:
            row_sites = np.arange(ts.num_sites, dtype=np.int32)
            col_sites = np.arange(ts.num_sites, dtype=np.int32)
        elif len(sites) == 2:
            row_sites = np.asarray(sites[0])
            col_sites = np.asarray(sites[1])
        elif len(sites) == 1:
            row_sites = np.asarray(sites[0])
            col_sites = row_sites
        else:
            raise ValueError(
                f"Sites must be a length 1 or 2 list, got a length {len(sites)} list"
            )
        check_order_bounds_dups(row_sites, ts.num_sites)
        check_order_bounds_dups(col_sites, ts.num_sites)
        result = two_site_count_stat(
            ts,
            summary_func,
            norm_func,
            result_dim,
            num_sample_sets,
            ss_sizes,
            ss_bits,
            sample_index_map,
            row_sites,
            col_sites,
            indexes,
            polarised,
        )
    elif mode == "branch":
        if sites is not None:
            raise ValueError("Cannot specify sites in branch mode")
        if positions is None:
            row_trees = np.arange(ts.num_trees, dtype=np.int32)
            col_trees = np.arange(ts.num_trees, dtype=np.int32)
        elif len(positions) == 2:
            breakpoints = ts.breakpoints(as_array=True)
            row_positions = np.asarray(positions[0])
            col_positions = np.asarray(positions[1])
            check_order_bounds_dups(row_positions, breakpoints[-1])
            check_order_bounds_dups(col_positions, breakpoints[-1])
            row_trees = positions_to_tree_indices(breakpoints, row_positions)
            col_trees = positions_to_tree_indices(breakpoints, col_positions)
        elif len(positions) == 1:
            breakpoints = ts.breakpoints(as_array=True)
            row_positions = np.asarray(positions[0])
            col_positions = row_positions
            check_order_bounds_dups(row_positions, breakpoints[-1])
            row_trees = positions_to_tree_indices(breakpoints, row_positions)
            col_trees = row_trees
        else:
            raise ValueError(
                "Positions must be a length 1 or 2 list, "
                f"got a length {len(positions)} list"
            )
        result = two_branch_count_stat(
            ts,
            summary_func,
            None,
            num_sample_sets,
            result_dim,
            ss_sizes,
            ss_bits,
            sample_index_map,
            row_trees,
            col_trees,
            indexes,
            False,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # If there is one result dimension, return a 2d array
    if drop_dim is True:
        return result.reshape(result.shape[1:3])
    return result


def r2_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    """Summary function for the r2 statistic. We first compute the proportion of
    AB, A, and B haplotypes, then we compute the r2 statistic, storing the outputs
    in the result vector, one entry per sample set.

    :param state_dim: Number of sample sets.
    :param state: Counts of 3 haplotype configurations for each sample set.
    :param result_dim: Number of dimensions in output. Dependent on arity of stat.
    :param result: Vector of length state_dim to store the results in.
    :param params: Parameters for the summary function.
    """
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        p_AB = state[0, k] / n
        p_Ab = state[1, k] / n
        p_aB = state[2, k] / n

        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB

        D = p_AB - (p_A * p_B)
        denom = p_A * p_B * (1 - p_A) * (1 - p_B)

        with suppress_overflow_div0_warning():
            result[k] = (D * D) / denom


def r2_ij_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    sample_set_sizes = params["sample_set_sizes"]
    set_indexes = params["set_indexes"]
    for k in range(result_dim):
        i = set_indexes[k][0]
        j = set_indexes[k][1]
        n = sample_set_sizes[i]
        pAB = state[0, i] / n
        pAb = state[1, i] / n
        paB = state[2, i] / n
        pA = pAB + pAb
        pB = pAB + paB
        D_i = pAB - pA * pB
        denom_i = np.sqrt(pA * (1 - pA) * pB * (1 - pB))

        n = sample_set_sizes[j]
        pAB = state[0, j] / n
        pAb = state[1, j] / n
        paB = state[2, j] / n
        pA = pAB + pAb
        pB = pAB + paB
        D_j = pAB - pA * pB
        denom_j = np.sqrt(pA * (1 - pA) * pB * (1 - pB))

        with suppress_overflow_div0_warning():
            result[k] = (D_i * D_j) / (denom_i * denom_j)


def D_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        p_AB = state[0, k] / float(n)
        p_Ab = state[1, k] / float(n)
        p_aB = state[2, k] / float(n)

        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB

        result[k] = p_AB - (p_A * p_B)


def D2_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        p_AB = state[0, k] / float(n)
        p_Ab = state[1, k] / float(n)
        p_aB = state[2, k] / float(n)

        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB

        result[k] = p_AB - (p_A * p_B)
        result[k] = result[k] * result[k]


def D_prime_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        p_AB = state[0, k] / float(n)
        p_Ab = state[1, k] / float(n)
        p_aB = state[2, k] / float(n)

        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB

        D = p_AB - (p_A * p_B)
        with suppress_overflow_div0_warning():
            if D >= 0:
                result[k] = D / min(p_A * (1 - p_B), (1 - p_A) * p_B)
            else:
                result[k] = D / min(p_A * p_B, (1 - p_A) * (1 - p_B))


def r_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        p_AB = state[0, k] / n
        p_Ab = state[1, k] / n
        p_aB = state[2, k] / n

        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB

        D = p_AB - (p_A * p_B)
        denom = p_A * p_B * (1 - p_A) * (1 - p_B)

        with suppress_overflow_div0_warning():
            result[k] = D / np.sqrt(denom)


def Dz_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        p_AB = state[0, k] / n
        p_Ab = state[1, k] / n
        p_aB = state[2, k] / n

        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB

        D = p_AB - (p_A * p_B)

        result[k] = D * (1 - 2 * p_A) * (1 - 2 * p_B)


def pi2_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
) -> None:
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        p_AB = state[0, k] / n
        p_Ab = state[1, k] / n
        p_aB = state[2, k] / n

        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB

        result[k] = p_A * (1 - p_A) * p_B * (1 - p_B)


# Unbiased estimators of pi2, dz, and d2. These are derived in Ragsdale 2019
# (https://doi.org/10.1093/molbev/msz265) and can be used in place of the method
# outlined by McVean 2002. The reason for using haplotype counts in the branch
# methods is that we can compute statistics that cannot be represented by tMRCA
# covariance. With these unbiased estimators, we still reproduce the values
# estimated with tMRCA covariance.

# TODO: update these summary functions to have the same function signature as
#       the summary functions defined above.


def pi2_unbiased_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
):
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        w_AB = state[0, k]
        w_Ab = state[1, k]
        w_aB = state[2, k]
        w_ab = n - (w_AB + w_Ab + w_aB)
        with suppress_overflow_div0_warning():
            result[k] = (1 / (n * (n - 1) * (n - 2) * (n - 3))) * (
                ((w_AB + w_Ab) * (w_aB + w_ab) * (w_AB + w_aB) * (w_Ab + w_ab))
                - ((w_AB * w_ab) * (w_AB + w_ab + (3 * w_Ab) + (3 * w_aB) - 1))
                - ((w_Ab * w_aB) * (w_Ab + w_aB + (3 * w_AB) + (3 * w_ab) - 1))
            )


def Dz_unbiased_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
):
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        w_AB = state[0, k]
        w_Ab = state[1, k]
        w_aB = state[2, k]
        w_ab = n - (w_AB + w_Ab + w_aB)
        with suppress_overflow_div0_warning():
            result[k] = (1 / (n * (n - 1) * (n - 2) * (n - 3))) * (
                (
                    ((w_AB * w_ab) - (w_Ab * w_aB))
                    * (w_aB + w_ab - w_AB - w_Ab)
                    * (w_Ab + w_ab - w_AB - w_aB)
                )
                - ((w_AB * w_ab) * (w_AB + w_ab - w_Ab - w_aB - 2))
                - ((w_Ab * w_aB) * (w_Ab + w_aB - w_AB - w_ab - 2))
            )


def D2_unbiased_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
):
    sample_set_sizes = params["sample_set_sizes"]
    for k in range(state_dim):
        n = sample_set_sizes[k]
        w_AB = state[0, k]
        w_Ab = state[1, k]
        w_aB = state[2, k]
        w_ab = n - (w_AB + w_Ab + w_aB)
        with suppress_overflow_div0_warning():
            result[k] = (1 / (n * (n - 1) * (n - 2) * (n - 3))) * (
                ((w_aB**2) * (w_Ab - 1) * w_Ab)
                + ((w_ab - 1) * w_ab * (w_AB - 1) * w_AB)
                - (w_aB * w_Ab * (w_Ab + (2 * w_ab * w_AB) - 1))
            )


def D2_ij_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
):
    sample_set_sizes = params["sample_set_sizes"]
    set_indexes = params["set_indexes"]
    for k in range(result_dim):
        i = set_indexes[k][0]
        j = set_indexes[k][1]

        n = sample_set_sizes[i]
        p_AB = state[0, i] / n
        p_Ab = state[1, i] / n
        p_aB = state[2, i] / n
        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB
        D_i = p_AB - (p_A * p_B)

        n = sample_set_sizes[j]
        p_AB = state[0, j] / n
        p_Ab = state[1, j] / n
        p_aB = state[2, j] / n
        p_A = p_AB + p_Ab
        p_B = p_AB + p_aB
        D_j = p_AB - (p_A * p_B)

        result[k] = D_i * D_j


def D2_ij_unbiased_summary_func(
    state_dim: int,
    state: np.ndarray,
    result_dim: int,
    result: np.ndarray,
    params: Dict[str, Any],
):
    sample_set_sizes = params["sample_set_sizes"]
    set_indexes = params["set_indexes"]

    for k in range(result_dim):
        i = set_indexes[k][0]
        j = set_indexes[k][1]
        # We require disjoint sample sets because we test equality here
        if i == j:
            n = sample_set_sizes[i]
            w_AB = state[0, i]
            w_Ab = state[1, i]
            w_aB = state[2, i]
            w_ab = n - (w_AB + w_Ab + w_aB)
            with suppress_overflow_div0_warning():
                result[k] = (
                    (
                        w_AB * (w_AB - 1) * w_ab * (w_ab - 1)
                        + w_Ab * (w_Ab - 1) * w_aB * (w_aB - 1)
                        - 2 * w_AB * w_Ab * w_aB * w_ab
                    )
                    / n
                    / (n - 1)
                    / (n - 2)
                    / (n - 3)
                )
        else:
            n_i = sample_set_sizes[i]
            w_AB_i = state[0, i]
            w_Ab_i = state[1, i]
            w_aB_i = state[2, i]
            w_ab_i = n_i - (w_AB_i + w_Ab_i + w_aB_i)

            n_j = sample_set_sizes[j]
            w_AB_j = state[0, j]
            w_Ab_j = state[1, j]
            w_aB_j = state[2, j]
            w_ab_j = n_j - (w_AB_j + w_Ab_j + w_aB_j)

            with suppress_overflow_div0_warning():
                result[k] = (
                    (w_Ab_i * w_aB_i - w_AB_i * w_ab_i)
                    * (w_Ab_j * w_aB_j - w_AB_j * w_ab_j)
                    / n_i
                    / (n_i - 1)
                    / n_j
                    / (n_j - 1)
                )


SUMMARY_FUNCS = {
    "r": r_summary_func,
    "r2": r2_summary_func,
    "D": D_summary_func,
    "D2": D2_summary_func,
    "D_prime": D_prime_summary_func,
    "pi2": pi2_summary_func,
    "Dz": Dz_summary_func,
    "D2_unbiased": D2_unbiased_summary_func,
    "Dz_unbiased": Dz_unbiased_summary_func,
    "pi2_unbiased": pi2_unbiased_summary_func,
}

TWO_WAY_SUMMARY_FUNCS = {
    "r2": r2_ij_summary_func,
    "D2": D2_ij_summary_func,
    "D2_unbiased": D2_ij_unbiased_summary_func,
}

NORM_METHOD = {
    D_summary_func: norm_total_weighted,
    D_prime_summary_func: norm_total_weighted,
    D2_summary_func: norm_total_weighted,
    Dz_summary_func: norm_total_weighted,
    pi2_summary_func: norm_total_weighted,
    r_summary_func: norm_total_weighted,
    r2_summary_func: norm_hap_weighted,
    D2_unbiased_summary_func: norm_total_weighted,
    Dz_unbiased_summary_func: norm_total_weighted,
    pi2_unbiased_summary_func: norm_total_weighted,
    r2_ij_summary_func: norm_hap_weighted_ij,
    D2_ij_summary_func: norm_total_weighted,
    D2_ij_unbiased_summary_func: norm_total_weighted,
}

POLARIZATION = {
    D_summary_func: True,
    D_prime_summary_func: True,
    D2_summary_func: False,
    Dz_summary_func: False,
    pi2_summary_func: False,
    r_summary_func: True,
    r2_summary_func: False,
    D2_unbiased_summary_func: False,
    Dz_unbiased_summary_func: False,
    pi2_unbiased_summary_func: False,
    r2_ij_summary_func: None,
    D2_ij_summary_func: None,
    D2_ij_unbiased_summary_func: None,
}


def check_set_indexes(
    num_sets: int, num_set_indexes: int, tuple_size: int, set_indexes: np.ndarray
):
    for i in range(num_set_indexes):
        for j in range(tuple_size):
            if set_indexes[i, j] < 0 or set_indexes[i, j] >= num_sets:
                raise ValueError(f"Bad sample set index: {set_indexes[i, j]}")


def check_sample_stat_inputs(
    num_sample_sets: int,
    tuple_size: int,
    num_index_tuples: int,
    index_tuples: np.ndarray,
):
    if num_sample_sets < tuple_size:
        raise ValueError(
            "Insufficient number of sample sets: "
            f"num_sample_sets: {num_sample_sets} tuple_size: {tuple_size}"
        )
    if num_index_tuples < 1:
        raise ValueError(f"Insufficient number of index tuples: {num_index_tuples}")
    check_set_indexes(num_sample_sets, num_index_tuples, tuple_size, index_tuples)


def ld_matrix(
    ts,
    sample_sets=None,
    sites=None,
    positions=None,
    stat="r2",
    indexes=None,
    mode="site",
):
    if indexes is not None:
        summary_func = TWO_WAY_SUMMARY_FUNCS[stat]
    else:
        summary_func = SUMMARY_FUNCS[stat]
    return two_locus_count_stat(
        ts,
        summary_func,
        NORM_METHOD[summary_func],
        POLARIZATION[summary_func],
        mode,
        sites=sites,
        positions=positions,
        indexes=indexes,
        sample_sets=sample_sets,
    )


def get_paper_ex_ts():
    """Generate the tree sequence example from the tskit paper

    Data taken from the tests:
    https://github.com/tskit-dev/tskit/blob/61a844a/c/tests/testlib.c#L55-L96

    :returns: Tree sequence
    """
    nodes = """\
    is_sample time population individual
    1  0       -1   0
    1  0       -1   0
    1  0       -1   1
    1  0       -1   1
    0  0.071   -1   -1
    0  0.090   -1   -1
    0  0.170   -1   -1
    0  0.202   -1   -1
    0  0.253   -1   -1
    """

    edges = """\
    left   right   parent  child
    2 10 4 2
    2 10 4 3
    0 10 5 1
    0 2  5 3
    2 10 5 4
    0 7  6 0,5
    7 10 7 0,5
    0 2  8 2,6
    """

    sites = """\
    position ancestral_state
    1      0
    4.5    0
    8.5    0
    """

    mutations = """\
    site node derived_state
    0      2   1
    1      0   1
    2      5   1
    """

    individuals = """\
    flags  location   parents
    0      0.2,1.5    -1,-1
    0      0.0,0.0    -1,-1
    """

    return tskit.load_text(
        nodes=io.StringIO(nodes),
        edges=io.StringIO(edges),
        sites=io.StringIO(sites),
        individuals=io.StringIO(individuals),
        mutations=io.StringIO(mutations),
        strict=False,
    )


# fmt:off
# true r2 values for the tree sequence from the tskit paper
PAPER_EX_TRUTH_MATRIX = np.array(
    [[1.0,        0.11111111, 0.11111111],  # noqa: E241
     [0.11111111, 1.0,        1.0],  # noqa: E241
     [0.11111111, 1.0,        1.0]]  # noqa: E241
)
PAPER_EX_BRANCH_TRUTH_MATRIX = np.array(
    [[ 1.06666667e-03, -1.26666667e-04, -1.26666667e-04],  # noqa: E241,E201
     [-1.26666667e-04,  6.01666667e-05,  6.01666667e-05],  # noqa: E241
     [-1.26666667e-04,  6.01666667e-05,  6.01666667e-05]]  # noqa: E241
)
# fmt:on


def get_matrix_partitions(n):
    """Generate all partitions for square matricies, then combine with replacement
    and return all possible pairs of all partitions.

    TODO: only works for square matricies, would need to generate two lists of
    partitions to get around this

    :param n: length of one dimension of the !square! matrix.
    :returns: combinations of partitions.
    """
    parts = []
    for part in tskit.combinatorics.rule_asc(n):
        for g in set(permutations(part, len(part))):
            p = []
            i = iter(range(n))
            for item in g:
                p.append([next(i) for _ in range(item)])
            parts.append(p)
    combos = []
    for a, b in combinations_with_replacement({tuple(j) for i in parts for j in i}, 2):
        combos.append((a, b))
        combos.append((b, a))
    combos = [[list(a), list(b)] for a, b in set(combos)]
    return combos


# Generate all partitions of the LD matrix, then pass into test_subset
@pytest.mark.parametrize("partition", get_matrix_partitions(len(PAPER_EX_TRUTH_MATRIX)))
def test_subset_sites(partition):
    """Given a partition of the truth matrix, check that we can successfully
    compute the LD matrix for that given partition, effectively ensuring that
    our handling of site subsets is correct.

    :param partition: length 2 list of [row_sites, column_sites].
    """
    a, b = partition
    ts = get_paper_ex_ts()
    np.testing.assert_allclose(
        ld_matrix(ts, sites=partition),
        PAPER_EX_TRUTH_MATRIX[a[0] : a[-1] + 1, b[0] : b[-1] + 1],
    )
    np.testing.assert_equal(
        ld_matrix(ts, sites=partition), ts.ld_matrix(sites=partition)
    )


@pytest.mark.parametrize(
    "partition", get_matrix_partitions(len(PAPER_EX_BRANCH_TRUTH_MATRIX))
)
def test_subset_positions(partition):
    """Given a partition of the truth matrix, check that we can successfully
    compute the LD matrix for that given partition, effectively ensuring that
    our handling of positions is correct. We use the midpoint inside of the
    tree interval as the position for a particular tree.

    :param partition: length 2 list of [row_positions, column_positions].
    """
    a, b = partition
    ts = get_paper_ex_ts()
    bp = ts.breakpoints(as_array=True)
    mid = (bp[1:] + bp[:-1]) / 2
    np.testing.assert_allclose(
        ld_matrix(ts, mode="branch", stat="D2_unbiased", positions=[mid[a], mid[b]]),
        PAPER_EX_BRANCH_TRUTH_MATRIX[a[0] : a[-1] + 1, b[0] : b[-1] + 1],
    )
    np.testing.assert_allclose(
        ts.ld_matrix(mode="branch", stat="D2_unbiased", positions=[mid[a], mid[b]]),
        PAPER_EX_BRANCH_TRUTH_MATRIX[a[0] : a[-1] + 1, b[0] : b[-1] + 1],
    )


@pytest.mark.parametrize(
    "positions,truth",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ([0], [0]),
        ([8], [8]),
        ([1], [1]),
        ([1, 2, 3], [1, 2, 3]),
        ([], []),
    ],
)
def test_positions_to_tree_indices(positions, truth):
    breakpoints = np.arange(10, dtype=np.float64)
    np.testing.assert_equal(positions_to_tree_indices(breakpoints, positions), truth)


def test_bad_positions():
    with pytest.raises(IndexError, match="out of bounds"):
        breakpoints = np.arange(10, dtype=np.float64)
        positions_to_tree_indices(breakpoints, breakpoints)


@pytest.mark.parametrize("sites", [[0, 1, 2], [1, 2], [0, 1], [0], [1]])
def test_subset_sites_one_list(sites):
    """Test the case where we only pass only one list of sites to compute. This
    should return a square matrix comparing the sites to themselves.
    """
    ts = get_paper_ex_ts()
    np.testing.assert_equal(ld_matrix(ts, sites=[sites]), ts.ld_matrix(sites=[sites]))


@pytest.mark.parametrize("tree_index", [[0, 1, 2], [1, 2], [0, 1], [0], [1]])
def test_subset_positions_one_list(tree_index):
    """Test the case where we only pass only one list of positions to compute. This
    should return a square matrix comparing the positions to themselves.
    """
    ts = get_paper_ex_ts()
    bp = ts.breakpoints(as_array=True)
    mid = (bp[1:] + bp[:-1]) / 2
    np.testing.assert_allclose(
        ld_matrix(ts, mode="branch", stat="D2_unbiased", positions=[mid[tree_index]]),
        PAPER_EX_BRANCH_TRUTH_MATRIX[
            tree_index[0] : tree_index[-1] + 1, tree_index[0] : tree_index[-1] + 1
        ],
    )
    np.testing.assert_allclose(
        ts.ld_matrix(mode="branch", stat="D2_unbiased", positions=[mid[tree_index]]),
        PAPER_EX_BRANCH_TRUTH_MATRIX[
            tree_index[0] : tree_index[-1] + 1, tree_index[0] : tree_index[-1] + 1
        ],
    )


@pytest.mark.parametrize(
    "tree_index",
    [
        ([0, 0, 1, 2], [1, 2]),
        ([0, 0, 0, 2], [0, 2]),
        ([1, 1, 1], [1]),
        ([2, 2], [1]),
        ([0, 2, 2, 2], [0, 0, 0]),
        ([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2]),
    ],
)
def test_repeated_position_elements(tree_index):
    """Test that we repeat positions in the LD matrix when we have multiple positions
    that overlap the same tree when specifying positions in branch mode.
    """
    ts = get_paper_ex_ts()
    l, r = tree_index
    bp = ts.breakpoints(as_array=True)
    val, count = np.unique(l, return_counts=True)
    l_pos = np.hstack(
        [np.linspace(bp[v], bp[v + 1], count[i] + 2)[1:-1] for i, v in enumerate(val)]
    )
    val, count = np.unique(r, return_counts=True)
    r_pos = np.hstack(
        [np.linspace(bp[v], bp[v + 1], count[i] + 2)[1:-1] for i, v in enumerate(val)]
    )
    assert (positions_to_tree_indices(bp, l_pos) == l).all()
    assert (positions_to_tree_indices(bp, r_pos) == r).all()

    truth = PAPER_EX_BRANCH_TRUTH_MATRIX[
        [i for i, _ in product(l, r)], [i for _, i in product(l, r)]
    ].reshape(len(l), len(r))

    np.testing.assert_allclose(
        truth,
        ld_matrix(ts, mode="branch", stat="D2_unbiased", positions=[l_pos, r_pos]),
    )
    np.testing.assert_allclose(
        truth,
        ts.ld_matrix(mode="branch", stat="D2_unbiased", positions=[l_pos, r_pos]),
    )


# Generate all partitions of the samples, producing pairs of sample sets
@pytest.mark.parametrize(
    "partition", get_matrix_partitions(get_paper_ex_ts().num_samples)
)
def test_sample_sets(partition):
    """Test all partitions of sample sets, ensuring that we are correctly
    computing stats for various subsets of the samples in a given tree.

    :param partition: length 2 list of [ss_1, ss_2].
    """
    ts = get_paper_ex_ts()
    np.testing.assert_allclose(
        ld_matrix(ts, sample_sets=partition), ts.ld_matrix(sample_sets=partition)
    )


def test_compare_to_ld_calculator():
    ts = msprime.sim_ancestry(
        samples=4, recombination_rate=0.2, sequence_length=10, random_seed=1
    )
    ts = msprime.sim_mutations(ts, rate=0.5, random_seed=1, discrete_genome=False)
    ld_calc = tskit.LdCalculator(ts)
    np.testing.assert_array_almost_equal(ld_calc.get_r2_matrix(), ts.ld_matrix())


@pytest.mark.parametrize(
    "stat",
    sorted(SUMMARY_FUNCS.keys()),
)
def test_multiallelic_with_back_mutation(stat):
    ts = msprime.sim_ancestry(
        samples=4, recombination_rate=0.2, sequence_length=10, random_seed=1
    )
    ts = msprime.sim_mutations(ts, rate=0.5, random_seed=1)
    np.testing.assert_array_almost_equal(
        ld_matrix(ts, stat=stat), ts.ld_matrix(stat=stat)
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "ts",
    [
        ts
        for ts in get_example_tree_sequences()
        if ts.id not in {"no_samples", "empty_ts"}
    ],
)
# TODO: port unbiased summary functions
@pytest.mark.parametrize(
    "stat",
    sorted(SUMMARY_FUNCS.keys()),
)
def test_ld_matrix(ts, stat):
    np.testing.assert_array_almost_equal(
        ld_matrix(ts, stat=stat), ts.ld_matrix(stat=stat)
    )


@pytest.mark.parametrize(
    "ts",
    [ts for ts in get_example_tree_sequences() if ts.id in {"no_samples", "empty_ts"}],
)
def test_ld_empty_examples(ts):
    with pytest.raises(ValueError, match="at least one element"):
        ts.ld_matrix()
    with pytest.raises(ValueError, match="at least one element"):
        ts.ld_matrix(mode="branch")


def test_input_validation():
    ts = get_paper_ex_ts()
    with pytest.raises(ValueError, match="Unknown two-locus statistic"):
        ts.ld_matrix(stat="bad_stat")

    with pytest.raises(ValueError, match="must be a list of"):
        ts.ld_matrix(sites=["abc"])
    with pytest.raises(ValueError, match="must be a list of"):
        ts.ld_matrix(sites=[1, 2, 3])
    with pytest.raises(ValueError, match="must be a length 1 or 2 list"):
        ts.ld_matrix(sites=[[1, 2], [2, 3], [3, 4]])
    with pytest.raises(ValueError, match="must be a length 1 or 2 list"):
        ts.ld_matrix(sites=[[1, 2], [2, 3], [3, 4]])
    with pytest.raises(ValueError, match="must be a length 1 or 2 list"):
        ts.ld_matrix(sites=[])

    with pytest.raises(ValueError, match="must be a list of"):
        ts.ld_matrix(positions=["abc"], mode="branch")
    with pytest.raises(ValueError, match="must be a list of"):
        ts.ld_matrix(positions=[1.0, 2.0, 3.0], mode="branch")
    with pytest.raises(ValueError, match="must be a length 1 or 2 list"):
        ts.ld_matrix(positions=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], mode="branch")
    with pytest.raises(ValueError, match="must be a length 1 or 2 list"):
        ts.ld_matrix(positions=[], mode="branch")

    with pytest.raises(
        ValueError, match="Sample sets must contain at least one element"
    ):
        ts.ld_matrix(sample_sets=[[1, 2, 3], []], indexes=[])
    with pytest.raises(
        ValueError, match="Indexes must be convertable to a 2D numpy array"
    ):
        ts.ld_matrix(
            sample_sets=[ts.samples(), ts.samples()], indexes=[[1, 2, 3], [2, 3, 4]]
        )


@dataclass
class TreeState:
    """
    Class for storing tree state from one iteration to the next. This object
    enables easy copying of the state for computing a matrix.
    """

    pos: tsutil.TreeIndexes  # current position in the tree sequence
    parent: np.ndarray  # parent node of a given node (connected by an edge)
    branch_len: np.ndarray  # length of the branch above a particular child node
    node_samples: BitSet  # samples that exist under a given node, this is a
    # bitset with a row for each node and sample set. Rows are grouped by node,
    # for example:
    # node sample_set
    # 0    0
    # 0    1
    # 1    0
    # 1    1
    edges_out: List[int]  # list of edges removed during iteration
    edges_in: List[int]  # list of edges added during iteration

    def __init__(self, ts, sample_sets, num_sample_sets, sample_index_map):
        self.pos = tsutil.TreeIndexes(ts)
        self.parent = -np.ones(ts.num_nodes, dtype=np.int64)
        self.branch_len = np.zeros(ts.num_nodes, dtype=np.float64)
        self.node_samples = BitSet(ts.num_samples, ts.num_nodes * num_sample_sets)
        # Create a bit array to store all samples under each node for each sample set.
        # We initialize with the samples under the sample nodes.
        for n in range(ts.num_nodes):
            for k in range(num_sample_sets):
                if sample_sets.contains(k, sample_index_map[n]):
                    self.node_samples.add(
                        (num_sample_sets * n) + k, sample_index_map[n]
                    )
        # these are empty for the uninitialized state (index = -1)
        self.edges_in = []
        self.edges_out = []

    def advance(self, index):
        """
        Advance tree to next tree position. If the tree is still uninitialized,
        seeks may be performed to an arbitrary position. Since we need to
        compute stats over contiguous ranges of trees, once we've seeked to a
        position, we step forward by one tree. Finally, we set `edges_in` and
        `edges_out` to be consumed by the downstream stats function.

        :param index: Tree index to advance to
        """

        # if initialized or seeking to the first position from the beginning, jump
        # forward one tree
        if self.pos.index != tskit.NULL or index == 0:
            if index != 0:
                assert index == self.pos.index + 1, "only one step allowed"
            assert self.pos.next(), "out of bounds"
            edges_out = [
                self.pos.out_range.order[j]
                for j in range(self.pos.out_range.start, self.pos.out_range.stop)
            ]
            edges_in = [
                self.pos.in_range.order[j]
                for j in range(self.pos.in_range.start, self.pos.in_range.stop)
            ]
            self.edges_out = edges_out
            self.edges_in = edges_in
            return

        # if uninitialized (no current position), and seeking to an arbitrary point
        # in the tree, use seek_forward
        edges_out, edges_in = [], []
        self.pos.seek_forward(index)
        left = self.pos.interval.left
        # since we're starting from an uninitialized tree, we only add edges
        for j in range(self.pos.in_range.start, self.pos.in_range.stop):
            e = self.pos.in_range.order[j]
            # skip over edges that are not in the current tree
            if self.pos.ts.edges_left[e] <= left < self.pos.ts.edges_right[e]:
                edges_in.append(e)

        self.edges_out = edges_out
        self.edges_in = edges_in
        return


def compute_branch_stat_update(
    c,
    A_state,
    B_state,
    state_dim,
    result_dim,
    sign,
    stat_func,
    num_samples,
    result,
    params,
):
    """Compute an update to the two-locus statistic for a single subset of the
    tree being modified, relative to all subsets of the fixed tree.

    :param c: Child node of the edge we're modifying
    :param A_state: State for the tree contributing to the A samples (fixed)
    :param B_state: State for the tree contributing to the B samples (modified)
    :param state_dim: Number of sample sets.
    :param result_dim: Number of dimensions in output. Dependent on arity of stat.
    :param sign: The sign of the update
    :param stat_func: Function used to compute the two-locus statistic
    :param num_samples: Number of samples in the tree sequence
    :param result: Vector of LD results, length of number of sample sets
    :param params: Params of summary function.
    """
    b_len = B_state.branch_len[c] * sign
    if b_len == 0:
        return result

    AB_samples = BitSet(num_samples, 1)
    weights = np.zeros((3, state_dim), dtype=np.int64)
    result_tmp = np.zeros(result_dim, np.float64)

    for n in np.where(A_state.branch_len > 0)[0]:
        a_len = A_state.branch_len[n]
        for k in range(state_dim):
            row = (state_dim * n) + k
            c_row = (state_dim * c) + k
            # Samples under the modified edge and the current fixed tree node are AB
            A_state.node_samples.intersect(row, B_state.node_samples, c_row, AB_samples)

            w_AB = AB_samples.count(0)
            w_A = A_state.node_samples.count(row)
            w_B = B_state.node_samples.count(c_row)

            weights[0, k] = w_AB
            weights[1, k] = w_A - w_AB  # w_Ab
            weights[2, k] = w_B - w_AB  # w_aB

        stat_func(state_dim, weights, result_dim, result_tmp, params)
        for k in range(result_dim):
            result[k] += result_tmp[k] * a_len * b_len


def compute_branch_stat(
    ts: tskit.TreeSequence,
    stat_func,
    stat,
    params,
    state_dim,
    result_dim,
    l_state: TreeState,
    r_state: TreeState,
):
    """Step between trees in a tree sequence, updating our two-locus statistic
    as we add or remove edges. Since we're computing statistics for two loci, we
    have a focal tree that remains constant, and a tree that is updated to
    represent the tree we're comparing to. The lefthand tree is held constant
    and the righthand tree is modified. The statistic is updated as we add and
    remove branches, and when we reach the point where the righthand tree is
    fully updated, the statistic will have been updated to the two-locus
    statistic between both trees.

    For instance, if we pass in the l_state for tree 0 and the r_state for tree
    0, we will update the r_state until r_state contains the information for
    tree 1. Then, the statistic will represent the LD between tree 1 and tree 2.

    Currenty, iteration happens in the forward direction.

    :param ts: The underlying tree sequence object that we're iterating across.
    :param stat_func: A function that computes the two locus statistic, given
                      haplotype counts.
    :param stat: The two-locus statistic computed between two trees.
    :param params: Params of summary function.
    :param state_dim: Number of sample sets.
    :param result_dim: Number of dimensions in output. Dependent on arity of stat.
    :param l_state: The lefthand constant state
    :param r_state: The righthand state to be updated
    :returns: A tuple containing the statistic between the two trees after
              branch updates and the righthand tree state.
    """
    num_samples = ts.num_samples
    time = ts.tables.nodes.time
    updates = BitSet(ts.num_nodes, 1)

    # Identify modified nodes
    for e in r_state.edges_out + r_state.edges_in:
        p = ts.edges_parent[e]
        c = ts.edges_child[e]
        # identify affected nodes above child
        while p != tskit.NULL:
            updates.add(0, c)
            c = p
            p = r_state.parent[p]

    # Subtract the whole contribution from child node
    for c in updates.get_items(0):
        compute_branch_stat_update(
            c,
            l_state,
            r_state,
            state_dim,
            result_dim,
            -1,
            stat_func,
            num_samples,
            stat,
            params,
        )

    # Sample Removal
    for e in r_state.edges_out:
        p = ts.edges_parent[e]
        ec = ts.edges_child[e]
        # update samples under nodes, propagate upwards
        while p != tskit.NULL:
            for k in range(state_dim):
                r_state.node_samples.difference(
                    state_dim * p + k, r_state.node_samples, state_dim * ec + k
                )
            p = r_state.parent[p]
        # set the parent to prevent upwards iteration
        r_state.branch_len[ec] = 0
        r_state.parent[ec] = tskit.NULL

    # Sample Addition
    for e in r_state.edges_in:
        p = ts.edges_parent[e]
        ec = c = ts.edges_child[e]
        r_state.branch_len[c] = time[p] - time[c]
        r_state.parent[c] = p
        # update samples under nodes, store modified node, propagate upwards
        while p != tskit.NULL:
            updates.add(0, c)
            for k in range(state_dim):
                r_state.node_samples.union(
                    state_dim * p + k, r_state.node_samples, state_dim * ec + k
                )
            c = p
            p = r_state.parent[p]

    # Update all affected child nodes (fully subtracted, deferred from addition)
    for c in updates.get_items(0):
        compute_branch_stat_update(
            c,
            l_state,
            r_state,
            state_dim,
            result_dim,
            +1,
            stat_func,
            num_samples,
            stat,
            params,
        )

    return stat, r_state


@pytest.mark.parametrize(
    "ts",
    [
        ts
        for ts in get_example_tree_sequences()
        if ts.id
        not in {
            "no_samples",
            "empty_ts",
            # We must skip these cases so that tests run in a reasonable
            # amount of time. To get more complete testing, these filters
            # can be commented out. (runtime ~1hr)
            "gap_0",
            "gap_0.1",
            "gap_0.5",
            "gap_0.75",
            "n=2_m=32_rho=0",
            "n=10_m=1_rho=0",
            "n=10_m=1_rho=0.1",
            "n=10_m=2_rho=0",
            "n=10_m=2_rho=0.1",
            "n=10_m=32_rho=0",
            "n=10_m=32_rho=0.1",
            "n=10_m=32_rho=0.5",
            # we keep one n=100 case to ensure bit arrays are working
            "n=100_m=1_rho=0.1",
            "n=100_m=1_rho=0.5",
            "n=100_m=2_rho=0",
            "n=100_m=2_rho=0.1",
            "n=100_m=2_rho=0.5",
            "n=100_m=32_rho=0",
            "n=100_m=32_rho=0.1",
            "n=100_m=32_rho=0.5",
            "all_fields",
            "back_mutations",
            "multichar",
            "multichar_no_metadata",
            "bottleneck_n=100_mutated",
        }
    ],
)
@pytest.mark.parametrize("stat", sorted(SUMMARY_FUNCS.keys()))
def test_branch_ld_matrix(ts, stat):
    np.testing.assert_array_almost_equal(
        ts.ld_matrix(stat=stat, mode="branch"), ld_matrix(ts, stat=stat, mode="branch")
    )


def get_test_branch_sample_set_test_cases():
    p_dict = {ps.id: ps for ps in get_example_tree_sequences()}
    return [
        pytest.param(
            p_dict["n=100_m=1_rho=0"].values[0],
            [[51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]],
            id="n=100_m=1_rho=0",
        ),
        pytest.param(
            p_dict["all_nodes_samples"].values[0],
            [[2, 4, 5, 6]],
            id="all_nodes_samples",
        ),
        pytest.param(
            p_dict["bottleneck_n=10_mutated"].values[0],
            [[1, 2, 4, 9]],
            id="bottleneck_n=10_mutated",
        ),
        pytest.param(p_dict["gap_at_end"].values[0], [[1, 3, 5, 8]], id="gap_at_end"),
    ]


@pytest.mark.parametrize("ts,sample_set", get_test_branch_sample_set_test_cases())
@pytest.mark.parametrize("stat", sorted(SUMMARY_FUNCS.keys()))
def test_branch_ld_matrix_sample_sets(ts, sample_set, stat):
    np.testing.assert_array_almost_equal(
        ld_matrix(ts, stat=stat, mode="branch", sample_sets=sample_set),
        ts.ld_matrix(stat=stat, mode="branch", sample_sets=sample_set),
    )


def get_test_branch_2pop_test_cases():
    p_dict = {ps.id: ps for ps in get_example_tree_sequences()}
    return [
        pytest.param(
            p_dict["n=100_m=1_rho=0"].values[0],
            [
                [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
                [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
            ],
            id="n=100_m=1_rho=0",
        ),
        pytest.param(
            p_dict["all_nodes_samples"].values[0],
            [[2, 4, 5, 6], [2, 4, 5, 6]],
            id="all_nodes_samples",
        ),
        pytest.param(
            p_dict["bottleneck_n=10_mutated"].values[0],
            [[1, 2, 4, 9], [1, 2, 4, 9]],
            id="bottleneck_n=10_mutated",
        ),
        pytest.param(
            p_dict["gap_at_end"].values[0],
            [[1, 3, 5, 8], [1, 3, 5, 8]],
            id="gap_at_end",
        ),
    ]


@pytest.mark.parametrize("ts,sample_set", get_test_branch_2pop_test_cases())
@pytest.mark.parametrize(
    "stat", sorted([f for f in TWO_WAY_SUMMARY_FUNCS.keys() if "unbiased" not in f])
)
def test_branch_ld_matrix_2pop_sample_sets(ts, sample_set, stat):
    oneway_result = ts.ld_matrix(stat=stat, mode="branch", sample_sets=sample_set[0])
    # biased two-way statistics between two identical sample sets are equal to
    # results from the one-way statistic.
    np.testing.assert_array_almost_equal(
        ld_matrix(ts, stat=stat, mode="branch", sample_sets=sample_set, indexes=(0, 1)),
        oneway_result,
    )


@pytest.mark.parametrize("ts,sample_set", get_test_branch_2pop_test_cases())
@pytest.mark.parametrize(
    "stat", sorted([f for f in TWO_WAY_SUMMARY_FUNCS.keys() if "unbiased" in f])
)
def test_branch_ld_matrix_2pop_sample_sets_unbiased(ts, sample_set, stat):
    oneway_result = ts.ld_matrix(stat=stat, mode="branch", sample_sets=sample_set[0])
    # If the indexes are the same between two identical sample sets, we recover
    # the one-way statistic. We do not make any assertions about sample disjointedness
    np.testing.assert_array_almost_equal(
        ld_matrix(ts, stat=stat, mode="branch", sample_sets=sample_set, indexes=(0, 0)),
        oneway_result,
    )


def gen_dims_test_cases(ts, mode):
    ss = ts.samples()
    dim = ts.num_sites if mode == "site" else ts.num_trees
    base = (dim, dim)
    return [
        {"name": f"{mode}_default", "ld_params": {"mode": mode}, "shape": base},
        {
            "name": f"{mode}_dim_drop",
            "ld_params": {"mode": mode, "sample_sets": ss},
            "shape": base,
        },
        {
            "name": f"{mode}_no_dim_drop",
            "ld_params": {"mode": mode, "sample_sets": [ss]},
            "shape": (1, *base),
        },
        {
            "name": f"{mode}_two_sample_sets",
            "ld_params": {"mode": mode, "sample_sets": [ss, ss]},
            "shape": (2, *base),
        },
        {
            "name": f"{mode}_two_way_dim_drop",
            "ld_params": {"mode": mode, "sample_sets": [ss, ss], "indexes": (0, 1)},
            "shape": base,
        },
        {
            "name": f"{mode}_two_way_no_dim_drop",
            "ld_params": {"mode": mode, "sample_sets": [ss, ss], "indexes": [(0, 1)]},
            "shape": (1, *base),
        },
        {
            "name": f"{mode}_two_way_three_set_indexes",
            "ld_params": {
                "mode": mode,
                "sample_sets": [ss, ss],
                "indexes": [(0, 0), (0, 1), (1, 1)],
            },
            "shape": (3, *base),
        },
    ]


def get_test_dims_test_cases():
    test_cases = {
        "empty_tree",
        "all_nodes_samples",
        "n=3_m=32_rho=0.5",
        "rev_node_order",
        "internal_nodes_samples",
        "mixed_internal_leaf_samples",
    }
    for ts_case in [t for t in get_example_tree_sequences() if t.id in test_cases]:
        ts = ts_case.values[0]
        for dim_case in gen_dims_test_cases(ts, "site"):
            name = "_".join([dim_case["name"], ts_case.id])
            yield pytest.param(ts, dim_case["ld_params"], dim_case["shape"], id=name)
        for dim_case in gen_dims_test_cases(ts, "branch"):
            name = "_".join([dim_case["name"], ts_case.id])
            yield pytest.param(ts, dim_case["ld_params"], dim_case["shape"], id=name)


@pytest.mark.parametrize("ts,params,shape", get_test_dims_test_cases())
def test_dims(ts, params, shape):
    assert ts.ld_matrix(**params).shape == ld_matrix(ts, **params).shape == shape


@pytest.mark.parametrize("ts,sample_sets", get_test_branch_2pop_test_cases())
@pytest.mark.parametrize("stat", sorted(TWO_WAY_SUMMARY_FUNCS.keys()))
def test_two_way_branch_ld_matrix(ts, sample_sets, stat):
    np.testing.assert_array_almost_equal(
        ld_matrix(ts, sample_sets=sample_sets, indexes=[(0, 0), (0, 1), (1, 1)]),
        ts.ld_matrix(sample_sets=sample_sets, indexes=[(0, 0), (0, 1), (1, 1)]),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "ts",
    [
        ts
        for ts in get_example_tree_sequences()
        if ts.id not in {"no_samples", "empty_ts"}
    ],
)
@pytest.mark.parametrize(
    "stat",
    sorted(TWO_WAY_SUMMARY_FUNCS.keys()),
)
def test_two_way_site_ld_matrix(ts, stat):
    np.testing.assert_array_almost_equal(
        ld_matrix(ts, stat=stat), ts.ld_matrix(stat=stat)
    )
    ss = [ts.samples()] * 3
    np.testing.assert_array_almost_equal(
        ld_matrix(ts, stat=stat, sample_sets=ss, indexes=[(0, 0), (0, 1), (1, 1)]),
        ts.ld_matrix(stat=stat, sample_sets=ss, indexes=[(0, 0), (0, 1), (1, 1)]),
    )


@pytest.mark.parametrize(
    "genotypes,sample_sets,expected",
    [
        (
            # these genotypes are rows from a genotype matrix (sites x samples)
            correlated := np.array(
                [
                    [0, 1, 1, 0, 2, 2, 1, 0, 2, 0, 1, 2],
                    [1, 2, 2, 1, 0, 0, 2, 1, 0, 1, 2, 0],
                ],
            ),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10, 11])),
            np.float64(1.0),
        ),
        (
            correlated,
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])),
            np.float64(1.0),
        ),
        (
            correlated,
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8, 9])),
            np.float64(1.0),
        ),
        (
            correlated,
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8])),
            np.float64(1.0),
        ),
        (
            correlated,
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
            np.float64(np.nan),
        ),
        (
            correlated,
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6])),
            np.float64(np.nan),
        ),
        (
            anticorrelated := np.array(
                [
                    [0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3],
                    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 2, 2, 2, 2],
                ]
            ),
            (
                np.array([0, 2, 4, 6, 8, 10, 12, 14]),
                np.array([1, 3, 5, 7, 9, 11, 13, 15]),
            ),
            np.float64(1.0),
        ),
        (
            anticorrelated,
            (np.array([0, 2, 4, 6, 8, 10, 12, 14]), np.array([1, 3, 5, 7, 9, 11, 13])),
            np.float64(1.0),
        ),
        (
            anticorrelated,
            (np.array([0, 2, 4, 6, 8, 10, 12, 14]), np.array([1, 3, 5, 7, 9, 11])),
            np.float64(np.nan),
        ),
        (
            anticorrelated,
            (np.array([0, 2, 4, 6, 8, 10, 12, 14]), np.array([1, 3, 5, 7, 9])),
            np.float64(np.nan),
        ),
        (
            anticorrelated,
            (np.array([0, 2, 4, 6, 8, 10, 12, 14]), np.array([1, 3, 5, 7])),
            np.float64(np.nan),
        ),
        (
            anticorrelated,
            (np.array([0, 2, 4, 6, 8, 10, 12, 14]), np.array([1, 3, 5])),
            np.float64(np.nan),
        ),
        (
            anticorrelated,
            (np.array([0, 2, 4, 6, 8, 10, 12, 14]), np.array([1, 3])),
            np.float64(np.nan),
        ),
        (
            anticorrelated,
            (np.array([0, 2, 4, 6, 8, 10, 12, 14]), np.array([1])),
            np.float64(np.nan),
        ),
    ],
)
def test_multipopulation_r2_varying_unequal_set_sizes(genotypes, sample_sets, expected):
    a, b = genotypes
    state_dim = len(sample_sets)
    state = np.zeros((3, state_dim), dtype=int)
    result = np.zeros((max(a) + 1, max(b) + 1, 1))
    norm = np.zeros_like(result)
    params = dict(sample_set_sizes=list(map(len, sample_sets)), set_indexes=[(0, 1)])
    for i, j in np.ndindex(result.shape[:2]):
        for k, ss in enumerate(sample_sets):
            A = a[ss] == i
            B = b[ss] == j
            state[:, k] = (A & B).sum(), (A & ~B).sum(), (~A & B).sum()
        r2_ij_summary_func(state_dim, state, 1, result[i, j], params)
        norm_hap_weighted_ij(1, state, max(a) + 1, max(b) + 1, norm[i, j], params)

    np.testing.assert_allclose((result * norm).sum(), expected)
