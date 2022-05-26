# MIT License
#
# Copyright (c) 2018-2022 Tskit Developers
# Copyright (c) 2017 University of Oxford
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
Test cases for the command line interfaces to tskit
"""
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import msprime
import pytest

import tskit
import tskit.cli as cli
from . import tsutil


class TestException(Exception):
    __test__ = False
    """
    Custom exception we can throw for testing.
    """


def capture_output(func, *args, **kwargs):
    """
    Runs the specified function and arguments, and returns the
    tuple (stdout, stderr) as strings.
    """
    buffer_class = io.BytesIO
    if sys.version_info[0] == 3:
        buffer_class = io.StringIO
    stdout = sys.stdout
    sys.stdout = buffer_class()
    stderr = sys.stderr
    sys.stderr = buffer_class()

    try:
        # Recent versions of MacOS seem to have issues with us calling signal
        # during tests.
        with mock.patch("signal.signal"):
            func(*args, **kwargs)
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
    finally:
        sys.stdout.close()
        sys.stdout = stdout
        sys.stderr.close()
        sys.stderr = stderr
    return stdout_output, stderr_output


class TestCli(unittest.TestCase):
    """
    Superclass of tests for the CLI needing temp files.
    """

    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(prefix="tsk_cli_testcase_")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)


class TestTskitArgumentParser:
    """
    Tests for the argument parsers in msp.
    """

    def test_individuals_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "individuals"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 6

    def test_individuals_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "individuals"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "8"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 8

    def test_individuals_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "individuals"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--precision", "5"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 5

    def test_nodes_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "nodes"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 6

    def test_nodes_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "nodes"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "8"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 8

    def test_nodes_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "nodes"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--precision", "5"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 5

    def test_edges_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "edges"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 6

    def test_edges_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "edges"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "8"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 8

    def test_edges_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "edges"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--precision", "5"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 5

    def test_sites_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "sites"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 6

    def test_sites_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "sites"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "8"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 8

    def test_sites_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "sites"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--precision", "5"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 5

    def test_mutations_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "mutations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 6

    def test_mutations_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "mutations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "4"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 4

    def test_mutations_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "mutations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--precision", "9"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 9

    def test_provenances_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "provenances"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert not args.human

    def test_provenances_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "provenances"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-H"])
        assert args.tree_sequence == tree_sequence
        assert args.human

    def test_provenances_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "provenances"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--human"])
        assert args.tree_sequence == tree_sequence
        assert args.human

    @pytest.mark.skip(reason="fasta output temporarily disabled")
    def test_fasta_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "fasta"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert args.wrap == 60

    @pytest.mark.skip(reason="fasta output temporarily disabled")
    def test_fasta_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "fasta"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-w", "100"])
        assert args.tree_sequence == tree_sequence
        assert args.wrap == 100

    @pytest.mark.skip(reason="fasta output temporarily disabled")
    def test_fasta_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "fasta"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--wrap", "50"])
        assert args.tree_sequence == tree_sequence
        assert args.wrap == 50

    def test_vcf_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert args.ploidy is None

    def test_vcf_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-P", "2"])
        assert args.tree_sequence == tree_sequence
        assert args.ploidy == 2

    def test_vcf_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--ploidy", "5"])
        assert args.tree_sequence == tree_sequence
        assert args.ploidy == 5

    def test_upgrade_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "upgrade"
        source = "in.trees"
        destination = "out.trees"
        args = parser.parse_args([cmd, source, destination])
        assert args.source == source
        assert args.destination == destination
        assert not args.remove_duplicate_positions

    def test_info_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "info"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence

    def test_populations_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "populations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence

    def test_trees_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "trees"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 6
        assert not args.draw

    def test_trees_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "trees"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-d", "-p", "8"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 8
        assert args.draw

    def test_trees_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "trees"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--precision", "5", "--draw"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 5
        assert args.draw


class TestTskitConversionOutput(unittest.TestCase):
    """
    Tests the output of msp to ensure it's correct.
    """

    @classmethod
    def setUpClass(cls):
        ts = msprime.simulate(
            length=1,
            recombination_rate=2,
            mutation_rate=2,
            random_seed=1,
            migration_matrix=[[0, 1], [1, 0]],
            population_configurations=[
                msprime.PopulationConfiguration(5) for _ in range(2)
            ],
            record_migrations=True,
        )
        assert ts.num_migrations > 0
        cls._tree_sequence = tsutil.insert_random_ploidy_individuals(
            ts, samples_only=True
        )
        fd, cls._tree_sequence_file = tempfile.mkstemp(
            prefix="tsk_cli", suffix=".trees"
        )
        os.close(fd)
        cls._tree_sequence.dump(cls._tree_sequence_file)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._tree_sequence_file)

    def verify_individuals(self, output_individuals, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(individuals=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        assert output == output_individuals

    def test_individuals(self):
        cmd = "individuals"
        precision = 8
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, self._tree_sequence_file, "-p", str(precision)]
        )
        assert len(stderr) == 0
        output_individuals = stdout.splitlines()
        self.verify_individuals(output_individuals, precision)

    def verify_nodes(self, output_nodes, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(nodes=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        assert output == output_nodes

    def test_nodes(self):
        cmd = "nodes"
        precision = 8
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, self._tree_sequence_file, "-p", str(precision)]
        )
        assert len(stderr) == 0
        output_nodes = stdout.splitlines()
        self.verify_nodes(output_nodes, precision)

    def verify_edges(self, output_edges, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(edges=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        assert output == output_edges

    def test_edges(self):
        cmd = "edges"
        precision = 8
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, self._tree_sequence_file, "-p", str(precision)]
        )
        assert len(stderr) == 0
        output_edges = stdout.splitlines()
        self.verify_edges(output_edges, precision)

    def verify_sites(self, output_sites, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(sites=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        assert output == output_sites

    def test_sites(self):
        cmd = "sites"
        precision = 8
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, self._tree_sequence_file, "-p", str(precision)]
        )
        assert len(stderr) == 0
        output_sites = stdout.splitlines()
        self.verify_sites(output_sites, precision)

    def verify_mutations(self, output_mutations, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(mutations=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        assert output == output_mutations

    def test_mutations(self):
        cmd = "mutations"
        precision = 8
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, self._tree_sequence_file, "-p", str(precision)]
        )
        assert len(stderr) == 0
        output_mutations = stdout.splitlines()
        self.verify_mutations(output_mutations, precision)

    def verify_provenances(self, output_provenances):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(provenances=f)
            f.seek(0)
            output = f.read().splitlines()
        assert output == output_provenances

    def test_provenances(self):
        cmd = "provenances"
        stdout, stderr = capture_output(cli.tskit_main, [cmd, self._tree_sequence_file])
        assert len(stderr) == 0
        output_provenances = stdout.splitlines()
        self.verify_provenances(output_provenances)

    def test_provenances_human(self):
        cmd = "provenances"
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, "-H", self._tree_sequence_file]
        )
        assert len(stderr) == 0
        output_provenances = stdout.splitlines()
        # TODO Check the actual output here.
        assert len(output_provenances) > 0

    def verify_fasta(self, output_fasta):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.write_fasta(f)
            f.seek(0)
            fasta = f.read()
        assert output_fasta == fasta

    @pytest.mark.skip(reason="fasta output temporarily disabled")
    def test_fasta(self):
        cmd = "fasta"
        stdout, stderr = capture_output(cli.tskit_main, [cmd, self._tree_sequence_file])
        assert len(stderr) == 0
        self.verify_fasta(stdout)

    def verify_vcf(self, output_vcf):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.write_vcf(f)
            f.seek(0)
            vcf = f.read()
        assert output_vcf == vcf

    def test_vcf(self):
        cmd = "vcf"
        stdout, stderr = capture_output(cli.tskit_main, [cmd, self._tree_sequence_file])
        assert len(stderr) == 0
        self.verify_vcf(stdout)

    def verify_info(self, ts, output_info):
        assert str(ts) == output_info

    def test_info(self):
        cmd = "info"
        stdout, stderr = capture_output(cli.tskit_main, [cmd, self._tree_sequence_file])
        assert len(stderr) == 0
        ts = tskit.load(self._tree_sequence_file)
        self.verify_info(ts, stdout[:-1])

    def test_trees_no_draw(self):
        cmd = "trees"
        stdout, stderr = capture_output(cli.tskit_main, [cmd, self._tree_sequence_file])
        assert len(stderr) == 0
        ts = tskit.load(self._tree_sequence_file)
        assert len(stdout.splitlines()) == 3 * ts.num_trees

    def test_trees_draw(self):
        cmd = "trees"
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, "-d", self._tree_sequence_file]
        )
        assert len(stderr) == 0
        ts = tskit.load(self._tree_sequence_file)
        assert len(stdout.splitlines()) > 3 * ts.num_trees


class TestBadFile:
    """
    Tests that we deal with IO errors appropriately.
    """

    def verify(self, command):
        with mock.patch("sys.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                capture_output(cli.tskit_main, ["info", "/no/such/file"])
            mocked_exit.assert_called_once_with(
                "Load error: [Errno 2] No such file or directory: '/no/such/file'"
            )

    def test_info(self):
        self.verify("info")

    def test_fasta(self):
        self.verify("fasta")

    def test_vcf(self):
        self.verify("vcf")

    def test_nodes(self):
        self.verify("nodes")

    def test_edges(self):
        self.verify("edges")

    def test_sites(self):
        self.verify("sites")

    def test_mutations(self):
        self.verify("mutations")

    def test_provenances(self):
        self.verify("provenances")


class TestUpgrade(TestCli):
    """
    Tests the results of the upgrade operation to ensure they are
    correct.
    """

    def setUp(self):
        fd, self.legacy_file_name = tempfile.mkstemp(prefix="msp_cli", suffix=".trees")
        os.close(fd)
        fd, self.current_file_name = tempfile.mkstemp(prefix="msp_cli", suffix=".trees")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.legacy_file_name)
        os.unlink(self.current_file_name)

    def test_conversion(self):
        ts1 = msprime.simulate(10)
        for version in [2, 3]:
            tskit.dump_legacy(ts1, self.legacy_file_name, version=version)
            stdout, stderr = capture_output(
                cli.tskit_main,
                ["upgrade", self.legacy_file_name, self.current_file_name],
            )
            ts2 = tskit.load(self.current_file_name)
            assert stdout == ""
            assert stderr == ""
            # Quick checks to ensure we have the right tree sequence.
            # More thorough checks are done elsewhere.
            assert ts1.get_sample_size() == ts2.get_sample_size()
            assert ts1.num_edges == ts2.num_edges
            assert ts1.get_num_trees() == ts2.get_num_trees()

    def test_duplicate_positions(self):
        ts = msprime.simulate(10, mutation_rate=10)
        for version in [2, 3]:
            tskit.dump_legacy(ts, self.legacy_file_name, version=version)
            root = h5py.File(self.legacy_file_name, "r+")
            root["mutations/position"][:] = 0
            root.close()
            stdout, stderr = capture_output(
                cli.tskit_main,
                ["upgrade", "-d", self.legacy_file_name, self.current_file_name],
            )
            assert stdout == ""
            tsp = tskit.load(self.current_file_name)
            assert tsp.sample_size == ts.sample_size
            assert tsp.num_sites == 1

    def test_duplicate_positions_error(self):
        ts = msprime.simulate(10, mutation_rate=10)
        for version in [2, 3]:
            tskit.dump_legacy(ts, self.legacy_file_name, version=version)
            root = h5py.File(self.legacy_file_name, "r+")
            root["mutations/position"][:] = 0
            root.close()
            with mock.patch("sys.exit", side_effect=TestException) as mocked_exit:
                with pytest.raises(TestException):
                    capture_output(
                        cli.tskit_main,
                        ["upgrade", self.legacy_file_name, self.current_file_name],
                    )
                assert mocked_exit.call_count == 1
