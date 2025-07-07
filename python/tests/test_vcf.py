# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
# Copyright (c) 2016 University of Oxford
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
Test cases for VCF output in tskit.
"""
import contextlib
import io
import math
import os
import tempfile
import textwrap
import warnings

import msprime
import numpy as np
import pytest

import tests
import tests.test_wright_fisher as wf
import tskit
from tests import tsutil

# Pysam is not available on windows, so we don't make it mandatory here.
_pysam_imported = False
try:
    import pysam

    _pysam_imported = True
except ImportError:
    pass


@contextlib.contextmanager
def ts_to_pysam(ts, *args, **kwargs):
    """
    Returns a pysam VariantFile for the specified tree sequence and arguments.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        vcf_path = os.path.join(temp_dir, "file.vcf")
        with open(vcf_path, "w") as f:
            ts.write_vcf(f, *args, **kwargs, allow_position_zero=True)
        yield pysam.VariantFile(vcf_path)


def example_individuals(ts, ploidy=1):
    if ts.num_individuals == 0:
        yield None, ts.num_samples / ploidy
    else:
        yield None, ts.num_individuals
        yield list(range(ts.num_individuals)), ts.num_individuals
    if ts.num_individuals > 3:
        n = ts.num_individuals - 2
        yield list(range(n)), n
        yield 2 + np.random.choice(np.arange(n), n, replace=False), n


def legacy_write_vcf(tree_sequence, output, ploidy, contig_id):
    """
    Writes a VCF under the legacy conversion rules used in versions before 0.2.0.
    """
    if tree_sequence.get_sample_size() % ploidy != 0:
        raise ValueError("Sample size must a multiple of ploidy")
    n = tree_sequence.get_sample_size() // ploidy
    sample_names = [f"msp_{j}" for j in range(n)]
    last_pos = 0
    positions = []
    for variant in tree_sequence.variants():
        pos = int(round(variant.position))
        if pos <= last_pos:
            pos = last_pos + 1
        positions.append(pos)
        last_pos = pos
    contig_length = int(math.ceil(tree_sequence.get_sequence_length()))
    if len(positions) > 0:
        contig_length = max(positions[-1], contig_length)
    print("##fileformat=VCFv4.2", file=output)
    print(f"##source=tskit {tskit.__version__}", file=output)
    print('##FILTER=<ID=PASS,Description="All filters passed">', file=output)
    print(f"##contig=<ID={contig_id},length={contig_length}>", file=output)
    print('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">', file=output)
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
        sep="\t",
        end="",
        file=output,
    )
    for sample_name in sample_names:
        print("\t", sample_name, sep="", end="", file=output)
    print(file=output)
    for variant in tree_sequence.variants():
        pos = positions[variant.index]
        site_id = variant.site.id
        assert variant.num_alleles == 2
        print(
            contig_id,
            pos,
            site_id,
            variant.alleles[0],
            variant.alleles[1],
            ".",
            "PASS",
            ".",
            "GT",
            sep="\t",
            end="",
            file=output,
        )
        for j in range(n):
            genotype = "|".join(
                str(g) for g in variant.genotypes[j * ploidy : j * ploidy + ploidy]
            )
            print("\t", genotype, end="", sep="", file=output)
        print(file=output)


class TestLegacyOutput:
    """
    Tests if the VCF file produced by the low level code is the
    same as one we generate here.
    """

    def verify(self, ts, ploidy=1, contig_id="1"):
        assert ts.num_sites > 0
        f = io.StringIO()
        legacy_write_vcf(ts, f, ploidy=ploidy, contig_id=contig_id)
        vcf1 = f.getvalue()

        num_individuals = ts.num_samples // ploidy
        individual_names = [f"msp_{j}" for j in range(num_individuals)]
        f = io.StringIO()
        ts.write_vcf(
            f,
            ploidy=ploidy,
            contig_id=contig_id,
            position_transform="legacy",
            individual_names=individual_names,
        )
        vcf2 = f.getvalue()
        assert vcf1 == vcf2

    def test_msprime_length_1(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=666)
        self.verify(ts, ploidy=1)
        self.verify(ts, ploidy=2)
        self.verify(ts, ploidy=5)

    def test_msprime_length_10(self):
        ts = msprime.simulate(9, length=10, mutation_rate=0.1, random_seed=666)
        self.verify(ts, ploidy=1)
        self.verify(ts, ploidy=3)

    def test_contig_id(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=666)
        self.verify(ts, ploidy=1, contig_id="X")
        self.verify(ts, ploidy=2, contig_id="X" * 10)


class ExamplesMixin:
    """
    Mixin defining tests on various example tree sequences.
    """

    def test_simple_infinite_sites_random_ploidy(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=2)
        ts = tsutil.insert_random_ploidy_individuals(
            ts, min_ploidy=1, samples_only=True
        )
        assert ts.num_sites > 2
        self.verify(ts)

    def test_simple_infinite_sites_ploidy_2(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=2)
        ts = tsutil.insert_individuals(ts, ploidy=2)
        assert ts.num_sites > 2
        self.verify(ts)

    def test_simple_infinite_sites_ploidy_2_reversed_samples(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=2)
        samples = ts.samples()[::-1]
        ts = tsutil.insert_individuals(ts, nodes=samples, ploidy=2)
        assert ts.num_sites > 2
        self.verify(ts)

    def test_simple_jukes_cantor_random_ploidy(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = tsutil.jukes_cantor(ts, num_sites=10, mu=1, seed=2)
        ts = tsutil.insert_random_ploidy_individuals(
            ts, min_ploidy=1, samples_only=True
        )
        self.verify(ts)

    def test_single_tree_multichar_mutations(self):
        ts = msprime.simulate(6, random_seed=1, mutation_rate=1)
        ts = tsutil.insert_multichar_mutations(ts)
        ts = tsutil.insert_individuals(ts, ploidy=2)
        self.verify(ts)

    def test_many_trees_infinite_sites(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        assert ts.num_trees > 2
        ts = tsutil.insert_individuals(ts, ploidy=2)
        self.verify(ts)

    def test_many_trees_sequence_length_infinite_sites(self):
        for L in [0.5, 1.5, 3.3333]:
            ts = msprime.simulate(
                6, length=L, recombination_rate=2, mutation_rate=1, random_seed=1
            )
            assert ts.num_sites > 0
            ts = tsutil.insert_individuals(ts, ploidy=2)
            self.verify(ts)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            4,
            5,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=10,
        )
        tables.sort()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.05, random_seed=234)
        assert ts.num_sites > 0
        ts = tsutil.insert_individuals(ts, ploidy=4)
        self.verify(ts)

    def test_wright_fisher_initial_generation(self):
        tables = wf.wf_sim(
            6, 5, seed=3, deep_history=True, initial_generation_samples=True, num_loci=2
        )
        tables.sort()
        tables.simplify()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.08, random_seed=2)
        assert ts.num_sites > 0
        ts = tsutil.insert_individuals(ts, ploidy=3)
        self.verify(ts)

    def test_wright_fisher_unsimplified_multiple_roots(self):
        tables = wf.wf_sim(
            8,
            15,
            seed=1,
            deep_history=False,
            initial_generation_samples=False,
            num_loci=20,
        )
        tables.sort()
        ts = msprime.mutate(tables.tree_sequence(), rate=0.006, random_seed=2)
        assert ts.num_sites > 0
        ts = tsutil.insert_individuals(ts, ploidy=2)
        self.verify(ts)

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            9,
            10,
            seed=1,
            deep_history=True,
            initial_generation_samples=False,
            num_loci=5,
        )
        tables.sort()
        ts = tables.tree_sequence().simplify()
        ts = msprime.mutate(ts, rate=0.2, random_seed=1234)
        assert ts.num_sites > 0
        ts = tsutil.insert_individuals(ts, ploidy=3)
        self.verify(ts)


@pytest.mark.skipif(not _pysam_imported, reason="pysam not available")
class TestParseHeaderPysam(ExamplesMixin):
    """
    Test that pysam can parse the headers correctly.
    """

    def verify(self, ts):
        contig_id = "pysam"
        for indivs, num_indivs in example_individuals(ts):
            with ts_to_pysam(ts, contig_id=contig_id, individuals=indivs) as bcf_file:
                assert bcf_file.format == "VCF"
                assert bcf_file.version == (4, 2)
                header = bcf_file.header
                assert len(header.contigs) == 1
                contig = header.contigs[0]
                assert contig.name == contig_id
                assert contig.length > 0
                assert len(header.filters) == 1
                p = header.filters["PASS"]
                assert p.name == "PASS"
                assert p.description == "All filters passed"
                assert len(header.info) == 0
                assert len(header.formats) == 1
                fmt = header.formats["GT"]
                assert fmt.name == "GT"
                assert fmt.number == 1
                assert fmt.type == "String"
                assert fmt.description == "Genotype"
                assert len(bcf_file.header.samples) == num_indivs


class TestInterface:
    """
    Tests for the interface.
    """

    def test_bad_ploidy(self):
        ts = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        for bad_ploidy in [-1, 0]:
            with pytest.raises(ValueError, match="Ploidy must be a positive integer"):
                ts.write_vcf(io.StringIO, bad_ploidy)
        # Non divisible
        for bad_ploidy in [3, 7]:
            with pytest.raises(
                ValueError,
                match="Number of sample nodes 10 is not a multiple of ploidy",
            ):
                ts.write_vcf(io.StringIO, bad_ploidy)

    def test_individuals_no_nodes_default_args(self):
        ts1 = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        tables = ts1.dump_tables()
        tables.individuals.add_row()
        ts2 = tables.tree_sequence()
        # ts1 should work as it has no individuals
        ts1.as_vcf(allow_position_zero=True)
        # ts2 should fail as it has individuals but no nodes
        with warnings.catch_warnings(record=True) as w:
            with pytest.raises(ValueError, match="No samples in resulting VCF model"):
                ts2.as_vcf(allow_position_zero=True)
            assert len(w) == 1
            assert "At least one sample node does not have an individual ID" in str(
                w[0].message
            )

    def test_individuals_no_nodes_as_argument(self):
        ts1 = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        tables = ts1.dump_tables()
        tables.individuals.add_row()
        ts2 = tables.tree_sequence()
        with warnings.catch_warnings(record=True) as w:
            with pytest.raises(ValueError, match="No samples in resulting VCF model"):
                ts2.as_vcf(individuals=[0])
            assert len(w) == 1
            assert "At least one sample node does not have an individual ID" in str(
                w[0].message
            )

    def test_ploidy_with_sample_individuals(self):
        ts = msprime.sim_ancestry(3, random_seed=2)
        ts = tsutil.insert_branch_sites(ts)
        with pytest.raises(ValueError, match="Cannot specify ploidy when individuals"):
            ts.write_vcf(io.StringIO(), ploidy=2)

    def test_ploidy_with_no_node_individuals(self):
        ts1 = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        tables = ts1.dump_tables()
        tables.individuals.add_row()
        ts2 = tables.tree_sequence()
        with pytest.raises(ValueError, match="Cannot specify ploidy when individuals"):
            ts2.as_vcf(ploidy=2)

    def test_empty_individuals(self):
        ts = msprime.sim_ancestry(3, random_seed=2)
        ts = tsutil.insert_branch_sites(ts)
        with pytest.raises(ValueError, match="No individuals specified"):
            ts.as_vcf(individuals=[])

    def test_duplicate_individuals(self):
        ts = msprime.sim_ancestry(3, random_seed=2)
        ts = tsutil.insert_branch_sites(ts)
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_DUPLICATE_SAMPLE"):
            ts.as_vcf(individuals=[0, 0], allow_position_zero=True)

    def test_samples_with_and_without_individuals(self):
        ts = tskit.Tree.generate_balanced(3).tree_sequence
        tables = ts.dump_tables()
        tables.individuals.add_row()
        # Add a reference to an individual from one sample
        individual = tables.nodes.individual
        individual[0] = 0
        tables.nodes.individual = individual
        ts = tables.tree_sequence()
        ts = tsutil.insert_branch_sites(ts)
        with warnings.catch_warnings(record=True) as w:
            ts.as_vcf(allow_position_zero=True)
            assert len(w) == 1
            assert "At least one sample node does not have an individual ID" in str(
                w[0].message
            )

    def test_bad_individuals(self):
        ts = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        ts = tsutil.insert_individuals(ts, ploidy=2)
        with pytest.raises(ValueError, match="Invalid individual ID"):
            ts.write_vcf(io.StringIO(), individuals=[0, -1])
        with pytest.raises(ValueError, match="Invalid individual ID"):
            ts.write_vcf(io.StringIO(), individuals=[1, 2, ts.num_individuals])

    def test_ploidy_positional(self):
        ts = msprime.simulate(2, mutation_rate=2, random_seed=1)
        assert ts.as_vcf(2, allow_position_zero=True) == ts.as_vcf(
            ploidy=2, allow_position_zero=True
        )

    def test_only_ploidy_positional(self):
        ts = msprime.simulate(2, mutation_rate=2, random_seed=1)
        with pytest.raises(TypeError, match="positional arguments"):
            assert ts.as_vcf(2, "chr2")


class TestLimitations:
    """
    Verify the correct error behaviour in cases we don't support.
    """

    def test_many_alleles(self):
        ts = msprime.simulate(20, random_seed=45)
        tables = ts.dump_tables()
        tables.sites.add_row(0.5, "0")
        # 9 alleles should be fine
        for j in range(8):
            tables.mutations.add_row(0, node=j, derived_state=str(j + 1))
        ts = tables.tree_sequence()
        ts.write_vcf(io.StringIO(), allow_position_zero=True)
        for j in range(9, 15):
            tables.mutations.add_row(0, node=j, derived_state=str(j))
            ts = tables.tree_sequence()
            with pytest.raises(
                ValueError, match="More than 9 alleles not currently supported"
            ):
                ts.write_vcf(io.StringIO(), allow_position_zero=True)


class TestPositionTransformErrors:
    """
    Tests what happens when we provide bad position transforms
    """

    def get_example_ts(self):
        ts = msprime.simulate(11, mutation_rate=1, random_seed=11)
        assert ts.num_sites > 1
        return ts

    def test_wrong_output_dimensions(self):
        ts = self.get_example_ts()
        for bad_func in [np.sum, lambda x: []]:
            with pytest.raises(ValueError):
                ts.write_vcf(io.StringIO(), position_transform=bad_func)

    def test_bad_func(self):
        ts = self.get_example_ts()
        for bad_func in ["", Exception]:
            with pytest.raises(TypeError):
                ts.write_vcf(io.StringIO(), position_transform=bad_func)


class TestZeroPositionErrors:
    """
    Tests for handling zero position sites
    """

    def test_zero_position_error(self):
        ts = msprime.sim_ancestry(3, random_seed=2, sequence_length=10)
        ts = msprime.sim_mutations(ts, rate=1, random_seed=2)
        assert ts.sites_position[0] == 0

        with pytest.raises(ValueError, match="A variant position of 0"):
            ts.write_vcf(io.StringIO())

        # Should succeed if we allow it, or the site is masked or transformed
        ts.write_vcf(io.StringIO(), allow_position_zero=True)
        ts.write_vcf(io.StringIO(), position_transform=lambda pos: [x + 1 for x in pos])
        mask = np.zeros(ts.num_sites, dtype=bool)
        mask[0] = True
        ts.write_vcf(io.StringIO(), site_mask=mask)

    def test_no_position_zero_ok(self):
        ts = msprime.sim_ancestry(3, random_seed=2, sequence_length=10)
        ts = msprime.sim_mutations(ts, rate=0.25, random_seed=4)
        assert ts.num_sites > 0
        assert ts.sites_position[0] != 0
        ts.write_vcf(io.StringIO(), allow_position_zero=True)
        ts.write_vcf(io.StringIO())


class TestIndividualNames:
    """
    Tests for the individual names argument.
    """

    def test_bad_length_individuals(self):
        ts = msprime.simulate(6, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        ts = tsutil.insert_individuals(ts, ploidy=2)
        with pytest.raises(
            ValueError,
            match="The number of individuals does not match the number of names",
        ):
            ts.write_vcf(io.StringIO(), individual_names=[])
        with pytest.raises(
            ValueError,
            match="The number of individuals does not match the number of names",
        ):
            ts.write_vcf(io.StringIO(), individual_names=["x" for _ in range(4)])
        with pytest.raises(
            ValueError,
            match="The number of individuals does not match the number of names",
        ):
            ts.write_vcf(
                io.StringIO(),
                individuals=list(range(ts.num_individuals)),
                individual_names=["x" for _ in range(ts.num_individuals - 1)],
            )
        with pytest.raises(
            ValueError,
            match="The number of individuals does not match the number of names",
        ):
            ts.write_vcf(
                io.StringIO(),
                individuals=list(range(ts.num_individuals - 1)),
                individual_names=["x" for _ in range(ts.num_individuals)],
            )

    def test_bad_length_ploidy(self):
        ts = msprime.simulate(6, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        with pytest.raises(
            ValueError,
            match="The number of individuals does not match the number of names",
        ):
            ts.write_vcf(io.StringIO(), ploidy=2, individual_names=[])
        with pytest.raises(
            ValueError,
            match="The number of individuals does not match the number of names",
        ):
            ts.write_vcf(
                io.StringIO(), ploidy=2, individual_names=["x" for _ in range(4)]
            )

    def test_bad_type(self):
        ts = msprime.simulate(2, mutation_rate=2, random_seed=1)
        with pytest.raises(
            TypeError, match="sequence item 0: expected str instance," " NoneType found"
        ):
            ts.write_vcf(
                io.StringIO(), individual_names=[None, "b"], allow_position_zero=True
            )
        with pytest.raises(
            TypeError, match="sequence item 0: expected str instance," " bytes found"
        ):
            ts.write_vcf(
                io.StringIO(), individual_names=[b"a", "b"], allow_position_zero=True
            )


def drop_header(s):
    return "\n".join(line for line in s.splitlines() if not line.startswith("##"))


class TestMasking:
    @tests.cached_example
    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        ts = tsutil.insert_branch_sites(ts)
        return ts

    @pytest.mark.parametrize("mask", [[True], np.zeros(5, dtype=bool), []])
    def test_site_mask_wrong_size(self, mask):
        with pytest.raises(ValueError, match="Site mask must be"):
            self.ts().as_vcf(site_mask=mask)

    @pytest.mark.parametrize("mask", [[[0, 1], [1, 0]], "abcd"])
    def test_site_mask_bad_type(self, mask):
        # converting to a bool array is pretty lax in what's allows.
        with pytest.raises(ValueError, match="Site mask must be"):
            self.ts().as_vcf(site_mask=mask)

    @pytest.mark.parametrize("mask", [[[0, 1], [1, 0]], "abcd"])
    def test_sample_mask_bad_type(self, mask):
        # converting to a bool array is pretty lax in what's allows.
        with pytest.raises(ValueError, match="Sample mask must be"):
            self.ts().as_vcf(sample_mask=mask, allow_position_zero=True)

    def test_no_masks(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t0\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0\t1\t0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t0\t1"""
        expected = textwrap.dedent(s)
        assert drop_header(self.ts().as_vcf(allow_position_zero=True)) == expected

    def test_no_masks_triploid(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1|0|0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0|1|1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0|1|0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0|0|1"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(self.ts().as_vcf(ploidy=3, allow_position_zero=True))
            == expected
        )

    def test_site_0_masked(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0\t1\t0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t0\t1"""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(
            site_mask=[True, False, False, False], allow_position_zero=True
        )
        assert drop_header(actual) == expected

    def test_site_0_masked_triploid(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0|1|1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0|1|0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0|0|1"""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(
            ploidy=3, site_mask=[True, False, False, False], allow_position_zero=True
        )
        assert drop_header(actual) == expected

    def test_site_1_masked(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t0\t0
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0\t1\t0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t0\t1"""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(
            site_mask=[False, True, False, False], allow_position_zero=True
        )
        assert drop_header(actual) == expected

    def test_all_sites_masked(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2"""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(
            site_mask=[True, True, True, True], allow_position_zero=True
        )
        assert drop_header(actual) == expected

    def test_all_sites_not_masked(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t0\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0\t1\t0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t0\t1"""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(
            site_mask=[False, False, False, False], allow_position_zero=True
        )
        assert drop_header(actual) == expected

    @pytest.mark.parametrize(
        "mask",
        [[False, False, False], [0, 0, 0], lambda _: [False, False, False]],
    )
    def test_all_samples_not_masked(self, mask):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t0\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0\t1\t0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t0\t1"""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(sample_mask=mask, allow_position_zero=True)
        assert drop_header(actual) == expected

    @pytest.mark.parametrize(
        "mask", [[True, False, False], [1, 0, 0], lambda _: [True, False, False]]
    )
    def test_sample_0_masked(self, mask):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t.\t0\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t.\t1\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t.\t1\t0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t.\t0\t1"""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(sample_mask=mask, allow_position_zero=True)
        assert drop_header(actual) == expected

    @pytest.mark.parametrize(
        "mask", [[False, True, False], [0, 1, 0], lambda _: [False, True, False]]
    )
    def test_sample_1_masked(self, mask):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t.\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t.\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0\t.\t0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t.\t1"""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(sample_mask=mask, allow_position_zero=True)
        assert drop_header(actual) == expected

    @pytest.mark.parametrize(
        "mask", [[True, True, True], [1, 1, 1], lambda _: [True, True, True]]
    )
    def test_all_samples_masked(self, mask):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t.\t.\t.
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t.\t.\t.
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t.\t.\t.
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t.\t.\t."""
        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(sample_mask=mask, allow_position_zero=True)
        assert drop_header(actual) == expected

    def test_all_functional_sample_mask(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t.\t0\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t.\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0\t1\t.
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t.\t0\t1"""

        def mask(variant):
            a = [0, 0, 0]
            a[variant.site.id % 3] = 1
            return a

        expected = textwrap.dedent(s)
        actual = self.ts().as_vcf(sample_mask=mask, allow_position_zero=True)
        assert drop_header(actual) == expected

    @pytest.mark.skipif(not _pysam_imported, reason="pysam not available")
    def test_mask_ok_with_pysam(self):
        with ts_to_pysam(self.ts(), sample_mask=[0, 0, 1]) as records:
            variants = list(records)
            assert len(variants) == 4
            samples = ["tsk_0", "tsk_1", "tsk_2"]
            gts = [variants[0].samples[key]["GT"] for key in samples]
            assert gts == [(1,), (0,), (None,)]

            gts = [variants[1].samples[key]["GT"] for key in samples]
            assert gts == [(0,), (1,), (None,)]

            gts = [variants[2].samples[key]["GT"] for key in samples]
            assert gts == [(0,), (1,), (None,)]

            gts = [variants[3].samples[key]["GT"] for key in samples]
            assert gts == [(0,), (0,), (None,)]


class TestMissingData:
    @tests.cached_example
    def ts(self):
        tables = tskit.Tree.generate_balanced(2, span=10).tree_sequence.dump_tables()
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        ts = tsutil.insert_branch_sites(tables.tree_sequence())
        return ts

    def test_defaults(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t0\t.
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1\t."""
        expected = textwrap.dedent(s)
        assert drop_header(self.ts().as_vcf(allow_position_zero=True)) == expected

    def test_isolated_as_missing_true(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t0\t.
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1\t."""
        expected = textwrap.dedent(s)
        assert (
            drop_header(
                self.ts().as_vcf(isolated_as_missing=True, allow_position_zero=True)
            )
            == expected
        )

    def test_isolated_as_missing_false(self):
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t0\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1\t0"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(
                self.ts().as_vcf(isolated_as_missing=False, allow_position_zero=True)
            )
            == expected
        )

    @pytest.mark.skipif(not _pysam_imported, reason="pysam not available")
    def test_ok_with_pysam(self):
        with ts_to_pysam(self.ts(), sample_mask=[0, 0, 1]) as records:
            variants = list(records)
            assert len(variants) == 2
            samples = ["tsk_0", "tsk_1", "tsk_2"]
            gts = [variants[0].samples[key]["GT"] for key in samples]
            assert gts == [(1,), (0,), (None,)]

            gts = [variants[1].samples[key]["GT"] for key in samples]
            assert gts == [(0,), (1,), (None,)]


def drop_individuals(ts):
    tables = ts.dump_tables()
    individual = tables.nodes.individual
    individual[:] = -1
    tables.individuals.clear()
    tables.nodes.individual = individual
    return tables.tree_sequence()


class TestSampleOptions:
    @tests.cached_example
    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        ts = tsutil.insert_branch_sites(ts)
        tables = ts.dump_tables()
        tables.individuals.add_row()
        tables.individuals.add_row()
        individual = tables.nodes.individual
        # One diploid and one haploid, not in adjacent individuals
        individual[0] = 0
        individual[1] = 1
        individual[2] = 0
        tables.nodes.individual = individual
        return tables.tree_sequence()

    def test_no_individuals_defaults(self):
        ts = drop_individuals(self.ts())
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1\ttsk_2
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1\t0\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0\t1\t0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t0\t1"""
        expected = textwrap.dedent(s)
        assert drop_header(ts.as_vcf(allow_position_zero=True)) == expected

    def test_no_individuals_ploidy_3(self):
        ts = drop_individuals(self.ts())
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1|0|0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0|1|1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0|1|0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0|0|1"""
        expected = textwrap.dedent(s)
        assert drop_header(ts.as_vcf(ploidy=3, allow_position_zero=True)) == expected

    def test_no_individuals_ploidy_3_names(self):
        ts = drop_individuals(self.ts())
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1|0|0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0|1|1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0|1|0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0|0|1"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(
                ts.as_vcf(ploidy=3, individual_names=["A"], allow_position_zero=True)
            )
            == expected
        )

    def test_defaults(self):
        ts = self.ts()
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0\ttsk_1
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1|0\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0|1\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0|0\t1
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0|1\t0"""
        expected = textwrap.dedent(s)
        assert drop_header(ts.as_vcf(allow_position_zero=True)) == expected

    def test_individual_0(self):
        ts = self.ts()
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_0
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1|0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t0|1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t0|0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0|1"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(ts.as_vcf(individuals=[0], allow_position_zero=True))
            == expected
        )

    def test_individual_1(self):
        ts = self.ts()
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_1
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t1
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(ts.as_vcf(individuals=[1], allow_position_zero=True))
            == expected
        )

    def test_reversed(self):
        ts = self.ts()
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttsk_1\ttsk_0
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t0\t1|0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t1\t0|1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t1\t0|0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t0|1"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(ts.as_vcf(individuals=[1, 0], allow_position_zero=True))
            == expected
        )

    def test_reversed_names(self):
        ts = self.ts()
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t0\t1|0
        1\t2\t1\t0\t1\t.\tPASS\t.\tGT\t1\t0|1
        1\t4\t2\t0\t1\t.\tPASS\t.\tGT\t1\t0|0
        1\t6\t3\t0\t1\t.\tPASS\t.\tGT\t0\t0|1"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(
                ts.as_vcf(
                    individuals=[1, 0],
                    individual_names=["A", "B"],
                    allow_position_zero=True,
                ),
            )
            == expected
        )


class TestVcfMapping:
    def test_mix_sample_non_sample(self):
        ts = tskit.Tree.generate_balanced(5, span=10).tree_sequence
        ts = tsutil.insert_branch_sites(ts)
        assert ts.num_nodes >= 8
        tables = ts.dump_tables()
        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.individuals.add_row()
        individual = tables.nodes.individual
        assert np.all(individual == -1)
        # First has only non-sample nodes
        individual[7] = 0
        # Second has 2 sample nodes
        individual[0] = 1
        individual[1] = 1
        # Third has 1 non-sample and 1 sample
        individual[5] = 2
        individual[2] = 2
        # Fourth has sandwiched non-sample
        individual[3] = 3
        individual[6] = 3
        individual[4] = 3
        tables.nodes.individual = individual
        ts = tables.tree_sequence()

        # Individual "A" is redacted as has no nodes
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tB\tC\tD
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t1|1\t0\t0|0
        1\t1\t1\t0\t1\t.\tPASS\t.\tGT\t1|0\t0\t0|0
        1\t2\t2\t0\t1\t.\tPASS\t.\tGT\t0|1\t0\t0|0
        1\t3\t3\t0\t1\t.\tPASS\t.\tGT\t0|0\t1\t1|1
        1\t4\t4\t0\t1\t.\tPASS\t.\tGT\t0|0\t1\t0|0
        1\t6\t5\t0\t1\t.\tPASS\t.\tGT\t0|0\t0\t1|1
        1\t7\t6\t0\t1\t.\tPASS\t.\tGT\t0|0\t0\t1|0
        1\t8\t7\t0\t1\t.\tPASS\t.\tGT\t0|0\t0\t0|1"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(
                ts.as_vcf(
                    individual_names=["A", "B", "C", "D"],
                    allow_position_zero=True,
                ),
            )
            == expected
        )

        # Now with non-sample nodes, so A is included, C becomes diploid
        # and D is triploid
        s = """\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\tD
        1\t0\t0\t0\t1\t.\tPASS\t.\tGT\t0\t1|1\t0|1\t0|0|0
        1\t1\t1\t0\t1\t.\tPASS\t.\tGT\t0\t1|0\t0|0\t0|0|0
        1\t2\t2\t0\t1\t.\tPASS\t.\tGT\t0\t0|1\t0|0\t0|0|0
        1\t3\t3\t0\t1\t.\tPASS\t.\tGT\t1\t0|0\t1|0\t1|1|1
        1\t4\t4\t0\t1\t.\tPASS\t.\tGT\t0\t0|0\t1|0\t0|0|0
        1\t6\t5\t0\t1\t.\tPASS\t.\tGT\t0\t0|0\t0|0\t1|1|1
        1\t7\t6\t0\t1\t.\tPASS\t.\tGT\t0\t0|0\t0|0\t1|0|0
        1\t8\t7\t0\t1\t.\tPASS\t.\tGT\t0\t0|0\t0|0\t0|1|0"""
        expected = textwrap.dedent(s)
        assert (
            drop_header(
                ts.as_vcf(
                    individual_names=["A", "B", "C", "D"],
                    allow_position_zero=True,
                    include_non_sample_nodes=True,
                    isolated_as_missing=False,
                ),
            )
            == expected
        )
