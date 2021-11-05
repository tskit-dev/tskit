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
Test cases for fasta output in tskit.
"""
import functools
import io

import dendropy
import msprime
import numpy as np
import pytest
from Bio import SeqIO

import tests
import tskit


# setting up some basic haplotype data for tests
@functools.lru_cache(maxsize=100)
def create_data(sequence_length):
    ts = msprime.sim_ancestry(
        samples=5, sequence_length=sequence_length, random_seed=123
    )
    ts = msprime.sim_mutations(ts, rate=0.1, random_seed=1234)
    assert ts.num_sites > 5
    return ts


@tests.cached_example
def missing_data_example():
    # 2.00┊   4     ┊
    #     ┊ ┏━┻┓    ┊
    # 1.00┊ ┃  3    ┊
    #     ┊ ┃ ┏┻┓   ┊
    # 0.00┊ 0 1 2 5 ┊
    #     0        10
    #      |      |
    #  pos 2      9
    #  anc A      T
    ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.sites.add_row(2, ancestral_state="A")
    tables.sites.add_row(9, ancestral_state="T")
    tables.mutations.add_row(site=0, node=0, derived_state="G")
    tables.mutations.add_row(site=1, node=3, derived_state="C")
    return tables.tree_sequence()


class TestWrapText:
    def test_even_split(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 4))
        assert result == ["ABCD", "EFGH"]

    def test_non_even_split(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 3))
        assert result == ["ABC", "DEF", "GH"]

    def test_width_one(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 1))
        assert result == ["A", "B", "C", "D", "E", "F", "G", "H"]

    def test_width_full_length(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 8))
        assert result == ["ABCDEFGH"]

    def test_width_more_than_length(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 100))
        assert result == ["ABCDEFGH"]

    def test_width_0(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 0))
        assert result == ["ABCDEFGH"]

    @pytest.mark.parametrize("width", [-1, -2, -8, -100])
    def test_width_negative(self, width):
        # Just documenting that the current implementation works for negative
        # values fine.
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, width))
        assert result == ["ABCDEFGH"]


class TestLineLength:
    """
    Tests if the fasta file produced has the correct line lengths for
    default, custom, and no-wrapping options.
    """

    def verify_line_length(self, length, wrap_width=60):
        # set up data
        ts = create_data(length)
        output = io.StringIO()
        ref = "A" * length
        ts.write_fasta(output, wrap_width=wrap_width, reference_sequence=ref)
        output.seek(0)

        # check if length perfectly divisible by wrap_width or not and thus
        # expected line lengths
        no_hanging_line = True
        if wrap_width == 0:
            lines_expect = 1
            # for easier code in testing function, redefine wrap_width as
            # full length, ok as called write already
            wrap_width = length
        elif length % wrap_width == 0:
            lines_expect = length // wrap_width
        else:
            lines_expect = length // wrap_width + 1
            extra_line_length = length % wrap_width
            no_hanging_line = False

        seq_line_counter = 0
        id_lines = 0
        for line in output:
            # testing correct characters per sequence line
            if line[0] != ">":
                seq_line_counter += 1
                line_chars = len(line.strip("\n"))
                # test full default width lines
                if seq_line_counter < lines_expect:
                    assert wrap_width == line_chars
                elif no_hanging_line:
                    assert wrap_width == line_chars
                # test extra line if not perfectly divided by wrap_width
                else:
                    assert extra_line_length == line_chars
            # testing correct number of lines per sequence and correct num sequences
            else:
                id_lines += 1
                if seq_line_counter > 0:
                    assert lines_expect == seq_line_counter
                    seq_line_counter = 0
        assert id_lines == ts.num_samples

    def test_wrap_length_default_easy(self):
        # default wrap width (60) perfectly divides sequence length
        self.verify_line_length(length=300)

    def test_wrap_length_default_harder(self):
        # default wrap_width imperfectly divides sequence length
        self.verify_line_length(length=280)

    def test_wrap_length_custom_easy(self):
        # custom wrap_width, perfectly divides
        self.verify_line_length(length=100, wrap_width=20)

    def test_wrap_length_custom_harder(self):
        # custom wrap_width, imperfectly divides
        self.verify_line_length(length=100, wrap_width=30)

    def test_wrap_length_no_wrap(self):
        # no wrapping set by wrap_width = 0
        self.verify_line_length(length=100, wrap_width=0)

    def test_negative_wrap(self):
        ts = create_data(10)
        ref = "-" * 10
        with pytest.raises(ValueError, match="non-negative integer"):
            ts.as_fasta(reference_sequence=ref, wrap_width=-1)

    def test_floating_wrap(self):
        ts = create_data(10)
        ref = "-" * 10
        with pytest.raises(ValueError):
            ts.as_fasta(reference_sequence=ref, wrap_width=1.1)

    def test_numpy_wrap(self):
        ts = create_data(10)
        ref = "-" * 10
        x1 = ts.as_fasta(reference_sequence=ref, wrap_width=4)
        x2 = ts.as_fasta(reference_sequence=ref, wrap_width=np.array([4.0])[0])
        assert x1 == x2


class TestFileTextOutputEqual:
    @tests.cached_example
    def ts(self):
        return create_data(20)

    def test_defaults(self):
        ts = self.ts()
        buff = io.StringIO()
        ref = "_" * int(ts.sequence_length)
        ts.write_fasta(buff, reference_sequence=ref)
        assert buff.getvalue() == ts.as_fasta(reference_sequence=ref)

    def test_wrap_width(self):
        ts = self.ts()
        buff = io.StringIO()
        ref = "X" * int(ts.sequence_length)
        ts.write_fasta(buff, reference_sequence=ref, wrap_width=4)
        assert buff.getvalue() == ts.as_fasta(reference_sequence=ref, wrap_width=4)


class TestFlexibleFileArg:
    @tests.cached_example
    def ts(self):
        return create_data(20)

    def test_pathlib(self, tmp_path):
        path = tmp_path / "file.fa"
        ts = self.ts()
        ref = "-" * int(ts.sequence_length)
        ts.write_fasta(path, reference_sequence=ref)
        with open(path) as f:
            assert f.read() == ts.as_fasta(reference_sequence=ref)

    def test_path_str(self, tmp_path):
        path = str(tmp_path / "file.fa")
        ts = self.ts()
        ref = "-" * int(ts.sequence_length)
        ts.write_fasta(path, reference_sequence=ref)
        with open(path) as f:
            assert f.read() == ts.as_fasta(reference_sequence=ref)

    def test_fileobj(self, tmp_path):
        path = tmp_path / "file.fa"
        ts = self.ts()
        ref = "-" * int(ts.sequence_length)
        with open(path, "w") as f:
            ts.write_fasta(f, reference_sequence=ref)
        with open(path) as f:
            assert f.read() == ts.as_fasta(reference_sequence=ref)


def get_alignment_map(ts, reference_sequence):
    alignments = ts.alignments(reference_sequence=reference_sequence)
    return {f"n{u}": alignment for u, alignment in zip(ts.samples(), alignments)}


class TestBioPythonRoundTrip:
    """
    Tests that output from our code is read in by available software packages
    Here test for compatability with biopython processing - Bio.SeqIO
    """

    def verify(self, ts, wrap_width=60, reference_sequence=None):
        if reference_sequence is None:
            reference_sequence = "-" * int(ts.sequence_length)
        text = ts.as_fasta(wrap_width=wrap_width, reference_sequence=reference_sequence)
        bio_map = {
            k: v.seq
            for k, v in SeqIO.to_dict(SeqIO.parse(io.StringIO(text), "fasta")).items()
        }
        assert bio_map == get_alignment_map(ts, reference_sequence)

    def test_equal_lines(self):
        # sequence length perfectly divisible by wrap_width
        ts = create_data(300)
        self.verify(ts)

    def test_unequal_lines(self):
        # sequence length not perfectly divisible by wrap_width
        ts = create_data(280)
        self.verify(ts)

    def test_unwrapped(self):
        # sequences not wrapped
        ts = create_data(300)
        self.verify(ts, wrap_width=0)

    def test_A_reference(self):
        ts = create_data(20)
        self.verify(ts, reference_sequence="A" * 20)

    def test_missing_data(self):
        self.verify(missing_data_example())


class TestDendropyRoundTrip:
    def parse(self, fasta):
        d = dendropy.DnaCharacterMatrix.get(data=fasta, schema="fasta")
        return {str(k.label): str(v) for k, v in d.items()}

    def test_wrapped(self):
        ts = create_data(300)
        ref = "-" * int(ts.sequence_length)
        text = ts.as_fasta(reference_sequence=ref)
        alignment_map = self.parse(text)
        assert get_alignment_map(ts, ref) == alignment_map

    def test_unwrapped(self):
        ts = create_data(300)
        ref = "-" * int(ts.sequence_length)
        text = ts.as_fasta(reference_sequence=ref, wrap_width=0)
        alignment_map = self.parse(text)
        assert get_alignment_map(ts, ref) == alignment_map

    def test_missing_data(self):
        ts = missing_data_example()
        ref = "-" * int(ts.sequence_length)
        text = ts.as_fasta(reference_sequence=ref, wrap_width=0)
        alignment_map = self.parse(text)
        assert get_alignment_map(ts, ref) == alignment_map
