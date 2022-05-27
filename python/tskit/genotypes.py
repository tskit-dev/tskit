#
# MIT License
#
# Copyright (c) 2018-2022 Tskit Developers
# Copyright (c) 2015-2018 University of Oxford
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
from __future__ import annotations

import numpy as np

import _tskit
import tskit.trees as trees


class Variant:
    """
    A variant in a tree sequence, describing the observed genetic variation
    among samples for a given site. A variant consists (a) of a reference to
    the :class:`Site` instance in question; (b) the **alleles** that may be
    observed at the samples for this site; and (c) the **genotypes**
    mapping sample IDs to the observed alleles.

    Each element in the ``alleles`` tuple is a string, representing the
    actual observed state for a given sample. The ``alleles`` tuple is
    generated in one of two ways. The first (and default) way is for
    ``tskit`` to generate the encoding on the fly as alleles are encountered
    while generating genotypes. In this case, the first element of this
    tuple is guaranteed to be the same as the site's ``ancestral_state`` value
    and the list of alleles is also guaranteed not to contain any duplicates.
    Note that allelic values may be listed that are not referred to by any
    samples. For example, if we have a site that is fixed for the derived state
    (i.e., we have a mutation over the tree root), all genotypes will be 1, but
    the alleles list will be equal to ``('0', '1')``. Other than the
    ancestral state being the first allele, the alleles are listed in
    no particular order, and the ordering should not be relied upon
    (but see the notes on missing data below).

    The second way is for the user to define the mapping between
    genotype values and allelic state strings using the
    ``alleles`` parameter to the :meth:`TreeSequence.variants` method.
    In this case, there is no indication of which allele is the ancestral state,
    as the ordering is determined by the user.

    The ``genotypes`` represent the observed allelic states for each sample,
    such that ``var.alleles[var.genotypes[j]]`` gives the string allele
    for sample ID ``j``. Thus, the elements of the genotypes array are
    indexes into the ``alleles`` list. The genotypes are provided in this
    way via a numpy array to enable efficient calculations.

    When :ref:`missing data<sec_data_model_missing_data>` is present at a given
    site, the property ``has_missing_data`` will be True, at least one element
    of the ``genotypes`` array will be equal to ``tskit.MISSING_DATA``, and the
    last element of the ``alleles`` array will be ``None``. Note that in this
    case ``variant.num_alleles`` will **not** be equal to
    ``len(variant.alleles)``. The rationale for adding ``None`` to the end of
    the ``alleles`` list is to help code that does not handle missing data
    correctly fail early rather than introducing subtle and hard-to-find bugs.
    As ``tskit.MISSING_DATA`` is equal to -1, code that decodes genotypes into
    allelic values without taking missing data into account would otherwise
    incorrectly output the last allele in the list.
    """

    def __init__(self, tree_sequence, samples, isolated_as_missing, alleles):
        self.tree_sequence = tree_sequence
        self._ll_variant = _tskit.Variant(
            tree_sequence._ll_tree_sequence,
            samples=samples,
            isolated_as_missing=isolated_as_missing,
            alleles=alleles,
        )

    @property
    def site(self) -> trees.Site:
        """
        The site object for this variant.
        """
        return self.tree_sequence.site(self._ll_variant.site_id)

    @property
    def alleles(self) -> tuple:
        """
        A tuple of the allelic values that may be observed at the
        samples at the current site. The first element of this tuple is always
        the site's ancestral state.
        """
        return self._ll_variant.alleles

    @property
    def genotypes(self) -> np.ndarray:
        """
        An array of indexes into the list ``alleles``, giving the
        state of each sample at the current site.
        """
        return self._ll_variant.genotypes

    @property
    def has_missing_data(self) -> bool:
        """
        True if there is missing data for any of the
        samples at the current site.
        """
        return self._ll_variant.alleles[-1] is None

    @property
    def num_alleles(self) -> int:
        """
        The number of distinct alleles at this site. Note that
        this may be greater than the number of distinct values in the genotypes
        array.
        """
        return len(self.alleles) - self.has_missing_data

    # Deprecated alias to avoid breaking existing code.
    @property
    def position(self) -> float:
        return self.site.position

    # Deprecated alias to avoid breaking existing code.
    @property
    def index(self) -> int:
        return self._ll_variant.site_id

    # We need a custom eq for the numpy array
    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Variant)
            and self.tree_sequence == other.tree_sequence
            and self._ll_variant.site_id == other._ll_variant.site_id
            and self._ll_variant.alleles == other._ll_variant.alleles
            and np.array_equal(self._ll_variant.genotypes, other._ll_variant.genotypes)
        )

    def decode(self, site_id) -> None:
        self._ll_variant.decode(site_id)

    def copy(self) -> Variant:
        variant_copy = Variant.__new__(Variant)
        variant_copy.tree_sequence = self.tree_sequence
        variant_copy._ll_variant = self._ll_variant.restricted_copy()
        return variant_copy


#
# Miscellaneous auxiliary methods.
#
def allele_remap(alleles_from, alleles_to):
    # Returns an index map from the elements in one list (alleles_from)
    # to the elements of another list (alleles_to).
    #
    # If some elements in alleles_from are not in alleles_to,
    # then indices outside of alleles_to are used.
    alleles_to = np.array(alleles_to, dtype="U")
    alleles_from = np.array(alleles_from, dtype="U")
    allele_map = np.empty_like(alleles_from, dtype="uint32")
    overflow = len(alleles_to)
    for i, allele in enumerate(alleles_from):
        try:
            # Use the index of the first matching element.
            allele_map[i] = np.where(alleles_to == allele)[0][0]
        except IndexError:
            allele_map[i] = overflow
            overflow += 1
    return allele_map
