# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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
import itertools
import math
import os
import tempfile

import msprime
import numpy as np
import pytest
import vcf

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
def ts_to_pyvcf(ts, *args, **kwargs):
    """
    Returns a PyVCF reader for the specified tree sequence and arguments.
    """
    f = io.StringIO()
    ts.write_vcf(f, *args, **kwargs)
    f.seek(0)
    yield vcf.Reader(f)


@contextlib.contextmanager
def ts_to_pysam(ts, *args, **kwargs):
    """
    Returns a pysam VariantFile for the specified tree sequence and arguments.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        vcf_path = os.path.join(temp_dir, "file.vcf")
        with open(vcf_path, "w") as f:
            ts.write_vcf(f, *args, **kwargs)
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
        assert variant.num_alleles == 2
        print(
            contig_id,
            pos,
            ".",
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
        ts = tsutil.insert_random_ploidy_individuals(ts)
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

    def test_simple_infinite_sites_ploidy_2_even_samples(self):
        ts = msprime.simulate(20, mutation_rate=1, random_seed=2)
        samples = ts.samples()[0::2]
        ts = tsutil.insert_individuals(ts, nodes=samples, ploidy=2)
        assert ts.num_sites > 2
        self.verify(ts)

    def test_simple_jukes_cantor_random_ploidy(self):
        ts = msprime.simulate(10, random_seed=2)
        ts = tsutil.jukes_cantor(ts, num_sites=10, mu=1, seed=2)
        ts = tsutil.insert_random_ploidy_individuals(ts)
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
        ts = msprime.mutate(ts, rate=0.01, random_seed=1234)
        assert ts.num_sites > 0
        ts = tsutil.insert_individuals(ts, ploidy=3)
        self.verify(ts)


class TestParseHeaderPyvcf(ExamplesMixin):
    """
    Test that pyvcf can parse the headers correctly.
    """

    def verify(self, ts):
        contig_id = "pyvcf"
        for indivs, num_indivs in example_individuals(ts):
            with ts_to_pyvcf(ts, contig_id=contig_id, individuals=indivs) as reader:
                assert len(reader.contigs) == 1
                contig = reader.contigs[contig_id]
                assert contig.id == contig_id
                assert contig.length > 0
                assert len(reader.alts) == 0
                assert len(reader.filters) == 1
                p = reader.filters["PASS"]
                assert p.id == "PASS"
                assert len(reader.formats) == 1
                f = reader.formats["GT"]
                assert f.id == "GT"
                assert len(reader.infos) == 0
                assert len(reader.samples) == num_indivs


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


@pytest.mark.skipif(not _pysam_imported, reason="pysam not available")
class TestRecordsEqual(ExamplesMixin):
    """
    Tests where we parse the input using PyVCF and Pysam
    """

    def verify_records(self, pyvcf_records, pysam_records):
        assert len(pyvcf_records) == len(pysam_records)
        for pyvcf_record, pysam_record in zip(pyvcf_records, pysam_records):
            assert pyvcf_record.CHROM == pysam_record.chrom
            assert pyvcf_record.POS == pysam_record.pos
            assert pyvcf_record.ID == pysam_record.id
            if pysam_record.alts:
                assert pyvcf_record.ALT == list(pysam_record.alts)
            else:
                assert pyvcf_record.ALT == [] or pyvcf_record.ALT == [None]
            assert pyvcf_record.REF == pysam_record.ref
            assert pysam_record.filter[0].name == "PASS"
            assert pyvcf_record.FORMAT == "GT"
            pysam_samples = list(pysam_record.samples.keys())
            pyvcf_samples = [sample.sample for sample in pyvcf_record.samples]
            assert pysam_samples == pyvcf_samples
            for index, name in enumerate(pysam_samples):
                pyvcf_sample = pyvcf_record.samples[index]
                pysam_sample = pysam_record.samples[name]
                pyvcf_alleles = pyvcf_sample.gt_bases.split("|")
                assert list(pysam_sample.alleles) == pyvcf_alleles

    def verify(self, ts):
        for indivs, _num_indivs in example_individuals(ts):
            with ts_to_pysam(ts, individuals=indivs) as bcf_file, ts_to_pyvcf(
                ts, individuals=indivs
            ) as vcf_reader:
                pyvcf_records = list(vcf_reader)
                pysam_records = list(bcf_file)
                self.verify_records(pyvcf_records, pysam_records)


class TestContigLengths:
    """
    Tests that we create sensible contig lengths under a variety of conditions.
    """

    def get_contig_length(self, ts):
        with ts_to_pyvcf(ts) as reader:
            contig = reader.contigs["1"]
            return contig.length

    def test_no_mutations(self):
        ts = msprime.simulate(10, length=1)
        assert ts.num_mutations == 0
        contig_length = self.get_contig_length(ts)
        assert contig_length == 1

    def test_long_sequence(self):
        # Nominal case where we expect the positions to map within the original
        # sequence length
        ts = msprime.simulate(10, length=100, mutation_rate=0.01, random_seed=3)
        assert ts.num_mutations > 0
        contig_length = self.get_contig_length(ts)
        assert contig_length == 100

    def test_short_sequence(self):
        # Degenerate case where the positions cannot map into the sequence length
        ts = msprime.simulate(10, length=1, mutation_rate=10)
        assert ts.num_mutations > 1
        contig_length = self.get_contig_length(ts)
        assert contig_length == 1


class TestInterface:
    """
    Tests for the interface.
    """

    def test_bad_ploidy(self):
        ts = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        for bad_ploidy in [-1, 0, 20]:
            with pytest.raises(ValueError):
                ts.write_vcf(io.StringIO, bad_ploidy)
        # Non divisible
        for bad_ploidy in [3, 7]:
            with pytest.raises(ValueError):
                ts.write_vcf(io.StringIO, bad_ploidy)

    def test_individuals_no_nodes(self):
        ts = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        tables = ts.dump_tables()
        tables.individuals.add_row()
        ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            ts.write_vcf(io.StringIO())

    def test_ploidy_with_individuals(self):
        ts = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        tables = ts.dump_tables()
        tables.individuals.add_row()
        ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            ts.write_vcf(io.StringIO(), ploidy=2)

    def test_bad_individuals(self):
        ts = msprime.simulate(10, mutation_rate=0.1, random_seed=2)
        ts = tsutil.insert_individuals(ts, ploidy=2)
        with pytest.raises(ValueError):
            ts.write_vcf(io.StringIO(), individuals=[0, -1])
        with pytest.raises(ValueError):
            ts.write_vcf(io.StringIO(), individuals=[1, 2, ts.num_individuals])


class TestRoundTripIndividuals(ExamplesMixin):
    """
    Tests that we can round-trip genotype data through VCF using pyvcf.
    """

    def verify(self, ts):
        for indivs, _num_indivs in example_individuals(ts):
            with ts_to_pyvcf(ts, individuals=indivs) as vcf_reader:
                samples = []
                if indivs is None:
                    indivs = range(ts.num_individuals)
                for ind in map(ts.individual, indivs):
                    samples.extend(ind.nodes)
                for variant, vcf_row in itertools.zip_longest(
                    ts.variants(samples=samples), vcf_reader
                ):
                    assert vcf_row.POS == np.round(variant.site.position)
                    assert variant.alleles[0] == vcf_row.REF
                    assert list(variant.alleles[1:]) == [
                        allele for allele in vcf_row.ALT if allele is not None
                    ]
                    j = 0
                    for individual, sample in itertools.zip_longest(
                        map(ts.individual, indivs), vcf_row.samples
                    ):
                        calls = sample.data.GT.split("|")
                        allele_calls = sample.gt_bases.split("|")
                        assert len(calls) == len(individual.nodes)
                        for allele_call, call in zip(allele_calls, calls):
                            assert int(call) == variant.genotypes[j]
                            assert allele_call == variant.alleles[variant.genotypes[j]]
                            j += 1


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
        ts.write_vcf(io.StringIO())
        for j in range(9, 15):
            tables.mutations.add_row(0, node=j, derived_state=str(j))
            ts = tables.tree_sequence()
            with pytest.raises(ValueError):
                ts.write_vcf(io.StringIO())

    def test_missing_data(self):
        ts = msprime.simulate(5, mutation_rate=1, random_seed=45)
        tables = ts.dump_tables()
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            ts.write_vcf(io.StringIO())


class TestPositionTransformRoundTrip(ExamplesMixin):
    """
    Tests that the position transform method is working correctly.
    """

    def verify(self, ts):
        for transform in [np.round, np.ceil, lambda x: list(map(int, x))]:
            with ts_to_pyvcf(ts, position_transform=transform) as vcf_reader:
                values = [record.POS for record in vcf_reader]
                assert values == list(transform(ts.tables.sites.position))


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


class TestIndividualNames:
    """
    Tests for the individual names argument.
    """

    def test_bad_length_individuals(self):
        ts = msprime.simulate(6, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        ts = tsutil.insert_individuals(ts, ploidy=2)
        with pytest.raises(ValueError):
            ts.write_vcf(io.StringIO(), individual_names=[])
        with pytest.raises(ValueError):
            ts.write_vcf(io.StringIO(), individual_names=["x" for _ in range(4)])
        with pytest.raises(ValueError):
            ts.write_vcf(
                io.StringIO(),
                individuals=list(range(ts.num_individuals)),
                individual_names=["x" for _ in range(ts.num_individuals - 1)],
            )
        with pytest.raises(ValueError):
            ts.write_vcf(
                io.StringIO(),
                individuals=list(range(ts.num_individuals - 1)),
                individual_names=["x" for _ in range(ts.num_individuals)],
            )

    def test_bad_length_ploidy(self):
        ts = msprime.simulate(6, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        with pytest.raises(ValueError):
            ts.write_vcf(io.StringIO(), ploidy=2, individual_names=[])
        with pytest.raises(ValueError):
            ts.write_vcf(
                io.StringIO(), ploidy=2, individual_names=["x" for _ in range(4)]
            )

    def test_bad_type(self):
        ts = msprime.simulate(2, mutation_rate=2, random_seed=1)
        with pytest.raises(TypeError):
            ts.write_vcf(io.StringIO(), individual_names=[None, "b"])
        with pytest.raises(TypeError):
            ts.write_vcf(io.StringIO(), individual_names=[b"a", "b"])

    def test_round_trip(self):
        ts = msprime.simulate(2, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        with ts_to_pyvcf(ts, individual_names=["a", "b"]) as vcf_reader:
            assert vcf_reader.samples == ["a", "b"]

    def test_defaults(self):
        ts = msprime.simulate(2, mutation_rate=2, random_seed=1)
        assert ts.num_sites > 0
        with ts_to_pyvcf(ts) as vcf_reader:
            assert vcf_reader.samples == ["tsk_0", "tsk_1"]
