#
# MIT License
#
# Copyright (c) 2019-2024 Tskit Developers
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
Convert tree sequences to VCF.
"""
import numpy as np

import tskit
from . import provenance


def legacy_position_transform(positions):
    """
    Transforms positions in the tree sequence into VCF coordinates under
    the pre 0.2.0 legacy rule.
    """
    last_pos = 0
    transformed = []
    for pos in positions:
        pos = int(round(pos))
        if pos <= last_pos:
            pos = last_pos + 1
        transformed.append(pos)
        last_pos = pos
    return transformed


class VcfWriter:
    """
    Writes a VCF representation of the genotypes tree sequence to a
    file-like object.
    """

    def __init__(
        self,
        tree_sequence,
        *,
        ploidy,
        contig_id,
        individuals,
        individual_names,
        position_transform,
        site_mask,
        sample_mask,
        isolated_as_missing,
        allow_position_zero,
    ):
        self.tree_sequence = tree_sequence
        self.contig_id = contig_id
        self.isolated_as_missing = isolated_as_missing

        self.__make_sample_mapping(ploidy, individuals)
        if individual_names is None:
            individual_names = [f"tsk_{j}" for j in range(self.num_individuals)]
        self.individual_names = individual_names
        if len(self.individual_names) != self.num_individuals:
            raise ValueError(
                "individual_names must have length equal to the number of individuals"
            )

        # Transform coordinates for VCF
        if position_transform is None:
            position_transform = np.round
        elif position_transform == "legacy":
            position_transform = legacy_position_transform
        self.transformed_positions = np.array(
            position_transform(tree_sequence.tables.sites.position), dtype=int
        )
        if self.transformed_positions.shape != (tree_sequence.num_sites,):
            raise ValueError(
                "Position transform must return an array of the same length"
            )
        self.contig_length = max(
            1, int(position_transform([tree_sequence.sequence_length])[0])
        )
        if len(self.transformed_positions) > 0:
            # Arguably this should be last_pos + 1, but if we hit this
            # condition the coordinate systems are all muddled up anyway
            # so it's simpler to stay with this rule that was inherited
            # from the legacy VCF output code.
            self.contig_length = max(self.transformed_positions[-1], self.contig_length)

        if site_mask is None:
            site_mask = np.zeros(tree_sequence.num_sites, dtype=bool)
        self.site_mask = np.array(site_mask, dtype=bool)
        if self.site_mask.shape != (tree_sequence.num_sites,):
            raise ValueError("Site mask must be 1D a boolean array of length num_sites")

        self.sample_mask = sample_mask
        if sample_mask is not None:
            if not callable(sample_mask):
                sample_mask = np.array(sample_mask, dtype=bool)
                self.sample_mask = lambda _: sample_mask

        # The VCF spec does not allow for positions to be 0, so we error if one of the
        # transformed positions is 0 and allow_position_zero is False.
        if not allow_position_zero and np.any(
            self.transformed_positions[~site_mask] == 0
        ):
            raise ValueError(
                "A variant position of 0 was found in the VCF output, this is not "
                "fully compliant with the VCF spec. If you still wish to write the VCF "
                'please use the "allow_position_zero" argument to write_vcf. '
                "Alternatively, you can increment all the positions by one using "
                '"position_transform = lambda x: 1 + x" or coerce the zero to one with '
                '"position_transform = lambda x: np.fmax(1, x)"'
            )

    def __make_sample_mapping(self, ploidy, individuals):
        """
        Compute the sample IDs for each VCF individual and the template for
        writing out genotypes.
        """
        ts = self.tree_sequence
        self.samples = None
        self.individual_ploidies = []

        # Cannot use "ploidy" when *any* individuals are present.
        if ts.num_individuals > 0 and ploidy is not None:
            raise ValueError(
                "Cannot specify ploidy when individuals are present in tables "
            )

        if individuals is None:
            # Find all sample nodes that reference individuals
            individuals = np.unique(ts.nodes_individual[ts.samples()])
            if len(individuals) == 1 and individuals[0] == tskit.NULL:
                # No samples refer to individuals
                individuals = None
            else:
                # np.unique sorts the argument, so if NULL (-1) is present it
                # will be the first value.
                if individuals[0] == tskit.NULL:
                    raise ValueError(
                        "Sample nodes must either all be associated with individuals "
                        "or not associated with any individuals"
                    )
        else:
            individuals = np.array(individuals, dtype=np.int32)
            if len(individuals) == 0:
                raise ValueError("List of sample individuals empty")

        if individuals is not None:
            self.samples = []
            # FIXME this could probably be done more efficiently.
            for i in individuals:
                if i < 0 or i >= self.tree_sequence.num_individuals:
                    raise ValueError("Invalid individual IDs provided.")
                ind = self.tree_sequence.individual(i)
                if len(ind.nodes) == 0:
                    raise ValueError(f"Individual {i} not associated with a node")
                is_sample = {ts.node(u).is_sample() for u in ind.nodes}
                if len(is_sample) != 1:
                    raise ValueError(
                        f"Individual {ind.id} has nodes that are sample and "
                        "non-samples"
                    )
                self.samples.extend(ind.nodes)
                self.individual_ploidies.append(len(ind.nodes))
        else:
            if ploidy is None:
                ploidy = 1
            if ploidy < 1:
                raise ValueError("Ploidy must be >= 1")
            if ts.num_samples % ploidy != 0:
                raise ValueError("Sample size must be divisible by ploidy")
            self.individual_ploidies = np.full(
                ts.sample_size // ploidy, ploidy, dtype=np.int32
            )
        self.num_individuals = len(self.individual_ploidies)

    def __write_header(self, output):
        print("##fileformat=VCFv4.2", file=output)
        print(f"##source=tskit {provenance.__version__}", file=output)
        print('##FILTER=<ID=PASS,Description="All filters passed">', file=output)
        print(
            f"##contig=<ID={self.contig_id},length={self.contig_length}>", file=output
        )
        print(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">', file=output
        )
        vcf_samples = "\t".join(self.individual_names)
        print(
            "#CHROM",
            "POS",
            "ID",
            "REF",
            "ALT",
            "QUAL",
            "FILTER",
            "INFO",
            "FORMAT",
            vcf_samples,
            sep="\t",
            file=output,
        )

    def write(self, output):
        self.__write_header(output)

        # Build the array for hold the text genotype VCF data and the indexes into
        # this array for when we're updating it.
        gt_array = []
        indexes = []
        for ploidy in self.individual_ploidies:
            for _ in range(ploidy):
                indexes.append(len(gt_array))
                # First element here is a placeholder that we'll write the actual
                # genotypes into when for each variant.
                gt_array.extend([0, ord("|")])
            gt_array[-1] = ord("\t")
        gt_array[-1] = ord("\n")
        gt_array = np.array(gt_array, dtype=np.int8)
        # TODO Unclear here whether using int64 or int32 will be faster for this index
        # array. Test it out.
        indexes = np.array(indexes, dtype=int)

        for variant in self.tree_sequence.variants(
            samples=self.samples, isolated_as_missing=self.isolated_as_missing
        ):
            site_id = variant.site.id
            # We check the mask before we do any checks so we can use this as a
            # way of skipping problematic sites.
            if self.site_mask[site_id]:
                continue

            if variant.num_alleles > 9:
                raise ValueError(
                    "More than 9 alleles not currently supported. Please open an issue "
                    "on GitHub if this limitation affects you."
                )
            pos = self.transformed_positions[variant.index]
            ref = variant.alleles[0]
            alt = "."
            if variant.num_alleles > 1:
                alt = ",".join(variant.alleles[1 : variant.num_alleles])
            print(
                self.contig_id,
                pos,
                site_id,
                ref,
                alt,
                ".",
                "PASS",
                ".",
                "GT",
                sep="\t",
                end="\t",
                file=output,
            )
            genotypes = variant.genotypes
            gt_array[indexes] = genotypes + ord("0")
            if self.sample_mask is not None:
                genotypes = genotypes.copy()
                sample_mask = np.array(self.sample_mask(variant), dtype=bool)
                if sample_mask.shape != genotypes.shape:
                    raise ValueError(
                        "Sample mask must be a numpy array of size num_samples"
                    )
                genotypes[sample_mask] = -1
            if self.sample_mask is not None or variant.has_missing_data:
                missing = genotypes == -1
                gt_array[indexes[missing]] = ord(".")
            g_bytes = memoryview(gt_array).tobytes()
            g_str = g_bytes.decode()
            print(g_str, end="", file=output)
