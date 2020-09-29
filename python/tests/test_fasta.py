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
import io
import itertools
import os
import tempfile

import msprime
import pytest
from Bio import SeqIO

from tests import tsutil


# setting up some basic haplotype data for tests
def create_data(length):
    ts = msprime.simulate(
        sample_size=10, length=length, mutation_rate=1e-2, random_seed=123
    )
    ts = tsutil.jukes_cantor(ts, length, 1.0, seed=123)
    assert ts.num_sites == length
    return ts


class TestLineLength:
    """
    Tests if the fasta file produced has the correct line lengths for
    default, custom, and no-wrapping options.
    """

    def verify_line_length(self, length, wrap_width=60):
        # set up data
        length = length
        ts = create_data(length)
        output = io.StringIO()
        ts.write_fasta(output, wrap_width=wrap_width)
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

    def test_bad_wrap(self):
        ts = create_data(100)
        with pytest.raises(ValueError):
            ts.write_fasta(io.StringIO(), wrap_width=-1)


class TestSequenceIds:
    """
    Tests that sequence IDs are output correctly, whether default or custom
    and that the length of IDs supplied must equal number of sequences
    """

    def verify_ids(self, ts, seq_ids_in=None):
        seq_ids_read = []
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = os.path.join(temp_dir, "testing_def_fasta.txt")
            with open(fasta_path, "w") as f:
                ts.write_fasta(f, sequence_ids=seq_ids_in)
            with open(fasta_path) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq_ids_read.append(record.id)

        # test default seq ids
        if seq_ids_in in [None]:
            for i, val in enumerate(seq_ids_read):
                assert f"tsk_{i}" == val
        # test custom seq ids
        else:
            for i, j in itertools.zip_longest(seq_ids_in, seq_ids_read):
                assert i == j

    def test_default_ids(self):
        # test that default sequence ids, immediately following '>', are as expected
        ts = create_data(100)
        self.verify_ids(ts)

    def test_custom_ids(self):
        # test that custom sequence ids, immediately following '>', are as expected
        ts = create_data(100)
        seq_ids_in = [f"x_{_}" for _ in range(ts.num_samples)]
        self.verify_ids(ts, seq_ids_in)

    def test_bad_length_ids(self):
        ts = create_data(100)
        with pytest.raises(ValueError):
            seq_ids_in = [f"x_{_}" for _ in range(ts.num_samples - 1)]
            ts.write_fasta(io.StringIO(), sequence_ids=seq_ids_in)
        with pytest.raises(ValueError):
            seq_ids_in = [f"x_{_}" for _ in range(ts.num_samples + 1)]
            ts.write_fasta(io.StringIO(), sequence_ids=seq_ids_in)
        with pytest.raises(ValueError):
            seq_ids_in = []
            ts.write_fasta(io.StringIO(), sequence_ids=seq_ids_in)


class TestRoundTrip:
    """
    Tests that output from our code is read in by available software packages
    Here test for compatability with biopython processing - Bio.SeqIO
    """

    def verify(self, ts, wrap_width=60):
        biopython_fasta_read = []
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = os.path.join(temp_dir, "testing_def_fasta.txt")
            with open(fasta_path, "w") as f:
                ts.write_fasta(f, wrap_width=wrap_width)
            with open(fasta_path) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    biopython_fasta_read.append(record.seq)

        for i, j in itertools.zip_longest(biopython_fasta_read, ts.haplotypes()):
            assert i == j

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
