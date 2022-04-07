# MIT License
#
# Copyright (c) 2018-2022 Tskit Developers
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
Test cases for ms output in tskit. All of these tests have separate versions
for the cases of single replicate and multiple replicates. This is because
msprime.simulate generates a tree_sequence object if the num_replicates argument
is not used but an iterator over tree_sequences if the num_replicates argument
is used.
"""
import collections
import itertools
import os
import tempfile
import unittest

import msprime

import tskit as ts

length = 1e2
mutation_rate = 1e-2
num_replicates = 3


def get_ms_file_quantity(ms_file, quantity):
    quantities = {}
    num_replicates = 0
    num_sites = []
    num_positions = []
    num_haplotypes = []
    genotypes = []
    positions = []
    gens = []
    for line in ms_file:
        if len(line.split()) > 0:
            if line.split()[0] == "//":
                num_replicates = num_replicates + 1
                num_haplotypes.append(0)
                if len(gens) > 0:
                    genotypes.append(gens)
                    gens = []
            if line.split()[0] == "segsites:":
                num_sites.append(int(line.split()[1]))
            if line.split()[0] == "positions:":
                num_positions.append(len(line.split()) - 1)
                positions.append(line[11:].rstrip())
            if (
                line[0:2] == "00"
                or line[0:2] == "01"
                or line[0:2] == "10"
                or line[0:2] == "11"
            ):
                num_haplotypes[-1] = num_haplotypes[-1] + 1
                gens.append(line.rstrip())
    genotypes.append(gens)
    quantities["num_replicates"] = num_replicates
    quantities["num_sites"] = num_sites
    quantities["num_positions"] = num_positions
    quantities["num_haplotypes"] = num_haplotypes
    quantities["genotypes"] = genotypes
    quantities["positions"] = positions

    return quantities[quantity]


class TestNumReplicates(unittest.TestCase):
    """
    Tests that the number of replicates written out is the same as
    the number of replicates simulated
    """

    def verify_num_replicates(self, tree_seq, num_replicates):
        if isinstance(tree_seq, collections.abc.Iterable):
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(
                        tree_seq,
                        f,
                        num_replicates=num_replicates,
                    )
                with open(ms_file_path) as handle:
                    num_replicates_file = get_ms_file_quantity(handle, "num_replicates")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(tree_seq, f)
                with open(ms_file_path) as handle:
                    num_replicates_file = get_ms_file_quantity(handle, "num_replicates")
        self.assertEqual(num_replicates, num_replicates_file)

    def test_num_replicates(self):
        tree_seq = msprime.simulate(
            25, length=length, mutation_rate=mutation_rate, random_seed=123
        )
        self.verify_num_replicates(tree_seq, 1)

    def test_num_replicates_multiple(self):
        tree_seq = msprime.simulate(
            25,
            length=length,
            mutation_rate=mutation_rate,
            random_seed=123,
            num_replicates=num_replicates,
        )
        self.verify_num_replicates(tree_seq, num_replicates)


class TestNumHaplotypes(unittest.TestCase):
    """
    Tests that the number of haplotypes output are the same as the
    number of individuals simulated.
    """

    def verify_num_haplotypes(self, tree_seq, tree_seq2, num_replicates):
        if isinstance(tree_seq, collections.abc.Iterable):
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(
                        tree_seq,
                        f,
                        num_replicates=num_replicates,
                    )
                with open(ms_file_path) as handle:
                    num_haplotypes = get_ms_file_quantity(handle, "num_haplotypes")
            j = 0
            for ts_indv in tree_seq2:
                self.assertEqual(ts_indv.num_samples, num_haplotypes[j])
                j = j + 1
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(tree_seq, f)
                with open(ms_file_path) as handle:
                    num_haplotypes = get_ms_file_quantity(handle, "num_haplotypes")
            self.assertEqual(tree_seq.num_samples, num_haplotypes[0])

    def test_num_haplotypes(self):
        tree_seq = msprime.simulate(
            25, length=length, mutation_rate=mutation_rate, random_seed=123
        )
        self.verify_num_haplotypes(tree_seq, tree_seq, 1)

    def test_num_haplotypes_replicates(self):
        tree_seq = msprime.simulate(
            25,
            length=length,
            mutation_rate=mutation_rate,
            random_seed=123,
            num_replicates=num_replicates,
        )
        tree_seq, tree_seq2 = itertools.tee(tree_seq)
        self.verify_num_haplotypes(tree_seq, tree_seq2, num_replicates)


class TestNumSites(unittest.TestCase):
    """
    Tests that the number of sites written out as well as the length
    of the positions list match the number of variants in the tree sequence
    """

    def verify_num_sites(self, tree_seq, tree_seq2, num_replicates):
        if isinstance(tree_seq, collections.abc.Iterable):
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(
                        tree_seq,
                        f,
                        num_replicates=num_replicates,
                    )
                with open(ms_file_path) as handle:
                    num_sites = get_ms_file_quantity(handle, "num_sites")
                with open(ms_file_path) as handle:
                    num_positions = get_ms_file_quantity(handle, "num_positions")
            j = 0
            for ts_indv in tree_seq2:
                self.assertEqual(ts_indv.num_sites, num_sites[j])
                self.assertEqual(ts_indv.num_sites, num_positions[j])
                j = j + 1
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(tree_seq, f)
                with open(ms_file_path) as handle:
                    num_sites = get_ms_file_quantity(handle, "num_sites")
                with open(ms_file_path) as handle:
                    num_positions = get_ms_file_quantity(handle, "num_positions")
            self.assertEqual(tree_seq.num_sites, num_sites[0])
            self.assertEqual(tree_seq.num_sites, num_positions[0])

    def test_num_sites(self):
        tree_seq = msprime.simulate(
            25, length=length, mutation_rate=mutation_rate, random_seed=123
        )
        self.verify_num_sites(tree_seq, tree_seq, 1)

    def test_num_sites_replicates(self):
        tree_seq = msprime.simulate(
            25,
            length=length,
            mutation_rate=mutation_rate,
            random_seed=123,
            num_replicates=num_replicates,
        )
        tree_seq, tree_seq2 = itertools.tee(tree_seq)
        self.verify_num_sites(tree_seq, tree_seq2, num_replicates)


class TestGenotypes(unittest.TestCase):
    """
    Tests that the haplotypes written out are the same as the haplotypes generated.
    """

    def get_genotypes(self, tree_seq):
        genotypes = tree_seq.genotype_matrix()
        gens_array = []
        for k in range(tree_seq.num_samples):
            tmp_str = "".join(map(str, genotypes[:, k]))
            gens_array.append(tmp_str)
        return gens_array

    def verify_genotypes(self, tree_seq, tree_seq2, num_replicates):
        if isinstance(tree_seq, collections.abc.Iterable):
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(
                        tree_seq,
                        f,
                        num_replicates=num_replicates,
                    )
                with open(ms_file_path) as handle:
                    genotypes = get_ms_file_quantity(handle, "genotypes")
            j = 0
            for ts_indv in tree_seq2:
                self.assertEqual(self.get_genotypes(ts_indv), genotypes[j])
                j = j + 1
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(tree_seq, f)
                with open(ms_file_path) as handle:
                    genotypes = get_ms_file_quantity(handle, "genotypes")
                self.assertEqual(self.get_genotypes(tree_seq), genotypes[0])

    def test_genotypes(self):
        tree_seq = msprime.simulate(
            25, length=length, mutation_rate=mutation_rate, random_seed=123
        )
        self.verify_genotypes(tree_seq, tree_seq, 1)

    def test_genotypes_replicates(self):
        tree_seq = msprime.simulate(
            25,
            length=length,
            mutation_rate=mutation_rate,
            random_seed=123,
            num_replicates=num_replicates,
        )
        tree_seq, tree_seq2 = itertools.tee(tree_seq)
        self.verify_genotypes(tree_seq, tree_seq2, num_replicates)


class TestPositions(unittest.TestCase):
    """
    Tests that the positions for the mutations written out are the same as the
    positions generated.
    """

    def get_positions(self, tree_seq):
        positions = []
        for i in range(tree_seq.num_sites):
            positions.append(
                f"{tree_seq.site(i).position / tree_seq.sequence_length:.4f}"
            )
        positions = " ".join(positions)
        return positions

    def verify_positions(self, tree_seq, tree_seq2, num_replicates):
        if isinstance(tree_seq, collections.abc.Iterable):
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(
                        tree_seq,
                        f,
                        num_replicates=num_replicates,
                    )
                with open(ms_file_path) as handle:
                    positions = get_ms_file_quantity(handle, "positions")
            j = 0
            for ts_indv in tree_seq2:
                self.assertEqual(self.get_positions(ts_indv), positions[j])
                j = j + 1
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                ms_file_path = os.path.join(temp_dir, "testing_ms_file.txt")
                with open(ms_file_path, "w") as f:
                    ts.write_ms(tree_seq, f)
                with open(ms_file_path) as handle:
                    positions = get_ms_file_quantity(handle, "positions")
            self.assertEqual(self.get_positions(tree_seq), positions[0])

    def test_positions(self):
        tree_seq = msprime.simulate(
            25, length=length, mutation_rate=mutation_rate, random_seed=123
        )
        self.verify_positions(tree_seq, tree_seq, 1)

    def test_positions_replicates(self):
        tree_seq = msprime.simulate(
            25,
            length=length,
            mutation_rate=mutation_rate,
            random_seed=123,
            num_replicates=num_replicates,
        )
        tree_seq, tree_seq2 = itertools.tee(tree_seq)
        self.verify_positions(tree_seq, tree_seq2, num_replicates)
