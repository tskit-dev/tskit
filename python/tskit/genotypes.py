#
# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
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

import collections
import logging
import typing

import numpy as np

import _tskit
import tskit
import tskit.trees as trees
import tskit.util as util


class Variant:
    """
    A variant in a tree sequence, describing the observed genetic variation
    among samples for a given site. A variant consists of (a) a tuple of
    **alleles** listing the potential allelic states which samples at this site
    can possess; (b) an array of **genotypes** mapping sample IDs to the observed
    alleles (c) a reference to the :class:`Site` at which the Variant has been decoded
    and (d) an array of **samples** giving the node id to which the each element of
    the genotypes array corresponds.

    After creation a Variant is not yet decoded, and has no genotypes.
    To decode a Variant, call the :meth:`decode` method. The Variant class will then
    use a Tree, internal to the Variant, to seek to the position of the site and
    decode the genotypes at that site. It is therefore much more efficient to visit
    sites in sequential genomic order, either in a forwards or backwards direction,
    than to do so randomly.

    Each element in the ``alleles`` tuple is a string, representing an
    observed allelic state that may be seen at this site. The ``alleles`` tuple,
    which is guaranteed not to contain any duplicates, is generated in one of two
    ways. The first (and default) way is for ``tskit`` to generate the encoding on
    the fly while generating genotypes. In this case, the first element of this
    tuple is guaranteed to be the same as the site's ``ancestral_state`` value.
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
    way via a numpy numeric array to enable efficient calculations. To obtain a
    (less efficient) array of allele strings for each sample, you can use e.g.
    ``np.asarray(variant.alleles)[variant.genotypes]``.

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

    :param TreeSequence tree_sequence: The tree sequence to which this variant
        belongs.
    :param array_like samples: An array of node IDs for which to generate
        genotypes, or None for all sample nodes. Default: None.
    :param bool isolated_as_missing: If True, the genotype value assigned to
        missing samples (i.e., isolated samples without mutations) is
        :data:`.MISSING_DATA` (-1). If False, missing samples will be
        assigned the allele index for the ancestral state.
        Default: True.
    :param tuple alleles: A tuple of strings defining the encoding of
        alleles as integer genotype values. At least one allele must be provided.
        If duplicate alleles are provided, output genotypes will always be
        encoded as the first occurrence of the allele. If None (the default),
        the alleles are encoded as they are encountered during genotype
        generation.

    """

    def __init__(
        self, tree_sequence, samples=None, isolated_as_missing=None, alleles=None
    ):
        if isolated_as_missing is None:
            isolated_as_missing = True
        self.tree_sequence = tree_sequence
        if samples is not None:
            samples = util.safe_np_int_cast(samples, np.int32)
        self._ll_variant = _tskit.Variant(
            tree_sequence._ll_tree_sequence,
            samples=samples,
            isolated_as_missing=isolated_as_missing,
            alleles=alleles,
        )

    def _check_decoded(self):
        if self._ll_variant.site_id == tskit.NULL:
            raise ValueError(
                "This variant has not yet been decoded at a specific site, "
                "call Variant.decode to set the site."
            )

    @property
    def site(self) -> trees.Site:
        """
        The Site object for the site at which this variant has been decoded.
        """
        self._check_decoded()
        return self.tree_sequence.site(self._ll_variant.site_id)

    @property
    def alleles(self) -> tuple[str | None, ...]:
        """
        A tuple of the allelic values which samples can possess at the current
        site. Unless an encoding of alleles is specified when creating this
        variant instance, the first element of this tuple is always the site's
        ancestral state.
        """
        return self._ll_variant.alleles

    @property
    def samples(self) -> np.ndarray:
        """
        A numpy array of the node ids whose genotypes will be returned
        by the :meth:`genotypes` method.
        """
        return self._ll_variant.samples

    @property
    def genotypes(self) -> np.ndarray:
        """
        An array of indexes into the list ``alleles``, giving the
        state of each sample at the current site.
        """
        self._check_decoded()
        return self._ll_variant.genotypes

    @property
    def isolated_as_missing(self) -> bool:
        """
        True if isolated samples are decoded to missing data. If False, isolated
        samples are decoded to the ancestral state.
        """
        return self._ll_variant.isolated_as_missing

    @property
    def has_missing_data(self) -> bool:
        """
        True if there is missing data for any of the
        samples at the current site.
        """
        alleles = self._ll_variant.alleles
        return len(alleles) > 0 and alleles[-1] is None

    @property
    def num_missing(self) -> int:
        """
        The number of samples with missing data at this site.
        """
        return np.sum(self.genotypes == tskit.NULL)

    @property
    def num_alleles(self) -> int:
        """
        The number of distinct alleles at this site. Note that this may
        not be the same as the number of distinct values in the genotypes
        array: firstly missing data is not counted as an allele, and secondly,
        the site may contain mutations to alternative allele states (which are
        counted in the number of alleles) without the mutation being inherited
        by any of the samples.
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
            and self._ll_variant.site_id != tskit.NULL
            and self._ll_variant.alleles == other._ll_variant.alleles
            and np.array_equal(self._ll_variant.genotypes, other._ll_variant.genotypes)
        )

    def decode(self, site_id) -> None:
        """
        Decode the variant at the given site, setting the site ID, genotypes and
        alleles to those of the site and samples of this Variant.

        :param int site_id: The ID of the site to decode. This must be a valid site ID.
        """
        self._ll_variant.decode(site_id)

    def copy(self) -> Variant:
        """
        Create a copy of this Variant. Note that calling :meth:`decode` on the
        copy will fail as it does not take a copy of the internal tree.

        :return: The copy of this Variant.
        """
        variant_copy = Variant.__new__(Variant)
        variant_copy.tree_sequence = self.tree_sequence
        variant_copy._ll_variant = self._ll_variant.restricted_copy()
        return variant_copy

    def states(self, missing_data_string=None) -> np.ndarray:
        """
        Returns the allelic states at this site as an array of strings.

        .. warning::
            Using this method is inefficient compared to working with the
            underlying integer representation of genotypes as returned by
            the :attr:`~Variant.genotypes` property.

        :param str missing_data_string: A string that will be used to represent missing
            data. If any normal allele contains this character, an error is raised.
            Default: `None`, treated as `'N'`.
        :return: An numpy array of strings of length ``num_sites``.
        """
        if missing_data_string is None:
            missing_data_string = "N"
        elif not isinstance(missing_data_string, str):
            # Must explicitly test here, otherwise we output a numpy object array
            raise ValueError("Missing data string is not a string")
        alleles = self.alleles
        if alleles[-1] is None:
            if missing_data_string in alleles:
                raise ValueError(
                    "An existing allele is equal to the "
                    f"missing data string '{missing_data_string}'"
                )
            alleles = alleles[:-1] + (missing_data_string,)
        return np.array(alleles)[self.genotypes]

    def counts(self) -> typing.Counter[str | None]:
        """
        Returns a :class:`python:collections.Counter` object providing counts for each
        possible :attr:`allele <Variant.alleles>` at this site: i.e. the number of
        samples possessing that allele among the set of samples specified when creating
        this Variant (by default, this is all the sample nodes in the tree sequence).
        Missing data is represented by an allelic state of ``None``.

        :return: A counter of the number of samples associated with each allele.
        """
        counts = collections.Counter()
        if self.alleles[-1] is None:
            # we have to treat the last element of the genotypes array as special
            counts[None] = np.sum(self.genotypes == tskit.MISSING_DATA)
            for i, allele in enumerate(self.alleles[:-1]):
                counts[allele] = np.sum(self.genotypes == i)
        else:
            bincounts = np.bincount(self.genotypes, minlength=self.num_alleles)
            for i, allele in enumerate(self.alleles):
                counts[allele] = bincounts[i]
        return counts

    def frequencies(self, remove_missing=None) -> dict[str, float]:
        """
        Return a dictionary mapping each possible :attr:`allele <Variant.alleles>`
        at this site to the frequency of that allele: i.e. the number of samples
        with that allele divided by the total number of samples, among the set of
        samples specified when creating this Variant (by default, this is all the
        sample nodes in the tree sequence). Note, therefore, that if a restricted set
        of samples was specified on creation, the allele frequencies returned here
        will *not* be the global allele frequencies in the whole tree sequence.

        :param bool remove_missing: If True, only samples with non-missing data will
            be counted in the total number of samples used to calculate the frequency,
            and no information on the frequency of missing data is returned. Otherwise
            (default), samples with missing data are included when calculating
            frequencies.
        :return: A dictionary mapping allelic states to the frequency of each allele
            among the samples
        """
        if remove_missing is None:
            remove_missing = False
        total = len(self.samples)
        if remove_missing:
            total -= self.num_missing
        if total == 0:
            logging.warning(
                "No non-missing samples at this site, frequencies undefined"
            )
        return {
            allele: count / total if total > 0 else np.nan
            for allele, count in self.counts().items()
            if not (allele is None and remove_missing)
        }

    def __str__(self) -> str:
        """
        Return a plain text summary of the contents of a variant.
        """
        try:
            site_id = util.format_number(self.site.id, sep=",")
            site_position = util.format_number(self.site.position, sep=",")
            counts = self.counts()
            freqs = self.frequencies()
            samples = util.format_number(len(self.samples), sep=",")
            num_alleles = util.format_number(self.num_alleles, sep=",")
            rows = (
                [
                    ["Site id", f"{site_id}"],
                    ["Site position", f"{site_position}"],
                    ["Number of samples", f"{samples}"],
                    ["Number of alleles", f"{num_alleles}"],
                ]
                + [
                    [
                        f"Samples with allele "
                        f"""{'missing' if k is None else "'" + k + "'"}""",
                        f"{util.format_number(counts[k], sep=',')} "
                        f"({util.format_number(freqs[k] * 100, 2, sep=',')}%)",
                    ]
                    for k in self.alleles
                ]
                + [
                    ["Has missing data", str(self.has_missing_data)],
                    ["Isolated as missing", str(bool(self.isolated_as_missing))],
                ]
            )
        except ValueError as err:
            rows = [[str(err), ""]]
        return util.unicode_table(rows, title="Variant")

    def _repr_html_(self) -> str:
        """
        Return an html summary of a variant. Called by Jupyter notebooks
        to render a Variant.
        """
        return util.variant_html(self)

    def __repr__(self):
        d = {
            "site": self.site,
            "samples": self.samples,
            "alleles": self.alleles,
            "genotypes": self.genotypes,
            "has_missing_data": self.has_missing_data,
            "isolated_as_missing": self.isolated_as_missing,
        }
        return f"Variant({repr(d)})"


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
