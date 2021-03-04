#
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
Convert tree sequences to VCF.
"""
import numpy as np

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
        ploidy=None,
        contig_id="1",
        individuals=None,
        individual_names=None,
        position_transform=None,
    ):
        self.tree_sequence = tree_sequence
        self.contig_id = contig_id

        if individuals is None:
            individuals = np.arange(tree_sequence.num_individuals, dtype=int)
        self.individuals = individuals

        self.__make_sample_mapping(ploidy)

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

    def __make_sample_mapping(self, ploidy):
        """
        Compute the sample IDs for each VCF individual and the template for
        writing out genotypes.
        """
        self.samples = None
        self.individual_ploidies = []
        if len(self.individuals) > 0:
            if ploidy is not None:
                raise ValueError("Cannot specify ploidy when individuals present")
            self.samples = []
            for i in self.individuals:
                if i < 0 or i >= self.tree_sequence.num_individuals:
                    raise ValueError("Invalid individual IDs provided.")
                ind = self.tree_sequence.individual(i)
                self.samples.extend(ind.nodes)
                self.individual_ploidies.append(len(ind.nodes))
            if len(self.samples) == 0:
                raise ValueError("The individuals do not map to any sampled nodes.")
        else:
            if ploidy is None:
                ploidy = 1
            if ploidy < 1:
                raise ValueError("Ploidy must be >= 1")
            if self.tree_sequence.num_samples % ploidy != 0:
                raise ValueError("Sample size must be divisible by ploidy")
            self.individual_ploidies = [
                ploidy for _ in range(self.tree_sequence.sample_size // ploidy)
            ]
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

        for variant in self.tree_sequence.variants(samples=self.samples):
            if variant.num_alleles > 9:
                raise ValueError(
                    "More than 9 alleles not currently supported. Please open an issue "
                    "on GitHub if this limitation affects you."
                )
            if variant.has_missing_data:
                raise ValueError(
                    "Missing data is not currently supported. Please open an issue "
                    "on GitHub if this limitation affects you."
                )
            pos = self.transformed_positions[variant.index]
            ref = variant.alleles[0]
            alt = ",".join(variant.alleles[1:]) if len(variant.alleles) > 1 else "."
            print(
                self.contig_id,
                pos,
                ".",
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
            variant.genotypes += ord("0")
            gt_array[indexes] = variant.genotypes
            g_bytes = memoryview(gt_array).tobytes()
            g_str = g_bytes.decode()
            print(g_str, end="", file=output)
