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

from . import provenance


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
        include_non_sample_nodes,
    ):
        self.tree_sequence = tree_sequence

        vcf_model = tree_sequence.map_to_vcf_model(
            individuals=individuals,
            ploidy=ploidy,
            individual_names=individual_names,
            include_non_sample_nodes=include_non_sample_nodes,
            position_transform=position_transform,
            contig_id=contig_id,
            isolated_as_missing=isolated_as_missing,
        )

        # We now make some tweaks to the VCF model required for
        # writing the VCF in text format

        # Remove individuals with zero ploidy as these cannot be
        # represented in VCF.
        to_keep = (vcf_model.individuals_nodes != -1).any(axis=1)
        vcf_model.individuals_nodes = vcf_model.individuals_nodes[to_keep]
        vcf_model.individual_names = vcf_model.individuals_name[to_keep]
        self.individual_ploidies = [
            len(nodes[nodes >= 0]) for nodes in vcf_model.individuals_nodes
        ]
        self.num_individuals = len(vcf_model.individual_names)

        if len(vcf_model.individuals_nodes) == 0:
            raise ValueError("No samples in resulting VCF model")

        if len(vcf_model.transformed_positions) > 0:
            # Arguably this should be last_pos + 1, but if we hit this
            # condition the coordinate systems are all muddled up anyway
            # so it's simpler to stay with this rule that was inherited
            # from the legacy VCF output code.
            vcf_model.contig_length = max(
                vcf_model.transformed_positions[-1], vcf_model.contig_length
            )

        # Flatten the array of node IDs, filtering out the -1 padding values
        self.samples = []
        for row in vcf_model.individuals_nodes:
            for node_id in row:
                if node_id != -1:
                    self.samples.append(node_id)

        if site_mask is None:
            site_mask = np.zeros(tree_sequence.num_sites, dtype=bool)
        self.site_mask = np.array(site_mask, dtype=bool)
        if self.site_mask.shape != (tree_sequence.num_sites,):
            raise ValueError("Site mask must be 1D a boolean array of length num_sites")

        # The VCF spec does not allow for positions to be 0, so we error if one of the
        # transformed positions is 0 and allow_position_zero is False.
        if not allow_position_zero and np.any(
            vcf_model.transformed_positions[~site_mask] == 0
        ):
            raise ValueError(
                "A variant position of 0 was found in the VCF output, this is not "
                "fully compliant with the VCF spec. If you still wish to write the VCF "
                'please use the "allow_position_zero" argument to write_vcf. '
                "Alternatively, you can increment all the positions by one using "
                '"position_transform = lambda x: 1 + x" or coerce the zero to one with '
                '"position_transform = lambda x: np.fmax(1, x)"'
            )

        self.sample_mask = sample_mask
        if sample_mask is not None:
            if not callable(sample_mask):
                sample_mask = np.array(sample_mask, dtype=bool)
                self.sample_mask = lambda _: sample_mask

        self.vcf_model = vcf_model

    def __write_header(self, output):
        print("##fileformat=VCFv4.2", file=output)
        print(f"##source=tskit {provenance.__version__}", file=output)
        print('##FILTER=<ID=PASS,Description="All filters passed">', file=output)
        print(
            f"##contig=<ID={self.vcf_model.contig_id},length={self.vcf_model.contig_length}>",
            file=output,
        )
        print(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">', file=output
        )
        vcf_samples = "\t".join(self.vcf_model.individual_names)
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
            samples=self.samples, isolated_as_missing=self.vcf_model.isolated_as_missing
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
            pos = self.vcf_model.transformed_positions[variant.index]
            ref = variant.alleles[0]
            alt = "."
            if variant.num_alleles > 1:
                alt = ",".join(variant.alleles[1 : variant.num_alleles])
            print(
                self.vcf_model.contig_id,
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
