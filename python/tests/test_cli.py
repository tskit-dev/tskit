# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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

import msprime
import h5py

import tskit
import tskit.cli as cli


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


class TestTskitArgumentParser(unittest.TestCase):
    """
    Tests for the argument parsers in msp.
    """

    def test_nodes_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "nodes"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 6)

    def test_nodes_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "nodes"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "8"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 8)

    def test_nodes_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "nodes"
        tree_sequence = "test.trees"
        args = parser.parse_args([
            cmd, tree_sequence, "--precision", "5"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 5)

    def test_edges_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "edges"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 6)

    def test_edges_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "edges"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "8"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 8)

    def test_edges_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "edges"
        tree_sequence = "test.trees"
        args = parser.parse_args([
            cmd, tree_sequence, "--precision", "5"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 5)

    def test_sites_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "sites"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 6)

    def test_sites_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "sites"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "8"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 8)

    def test_sites_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "sites"
        tree_sequence = "test.trees"
        args = parser.parse_args([
            cmd, tree_sequence, "--precision", "5"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 5)

    def test_mutations_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "mutations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 6)

    def test_mutations_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "mutations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "4"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 4)

    def test_mutations_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "mutations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--precision", "9"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.precision, 9)

    def test_provenances_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "provenances"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.human, False)

    def test_provenances_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "provenances"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-H"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.human, True)

    def test_provenances_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "provenances"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--human"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.human, True)

    def test_vcf_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.ploidy, 1)

    def test_vcf_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([
            cmd, tree_sequence, "-P", "2"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.ploidy, 2)

    def test_vcf_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([
            cmd, tree_sequence, "--ploidy", "5"])
        self.assertEqual(args.tree_sequence, tree_sequence)
        self.assertEqual(args.ploidy, 5)

    def test_upgrade_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "upgrade"
        source = "in.trees"
        destination = "out.trees"
        args = parser.parse_args([cmd, source, destination])
        self.assertEqual(args.source, source)
        self.assertEqual(args.destination, destination)
        self.assertEqual(args.remove_duplicate_positions, False)


class TestMspConversionOutput(unittest.TestCase):
    """
    Tests the output of msp to ensure it's correct.
    """
    @classmethod
    def setUpClass(cls):
        cls._tree_sequence = msprime.simulate(
            10, length=10, recombination_rate=10,
            mutation_rate=10, random_seed=1)
        fd, cls._tree_sequence_file = tempfile.mkstemp(
            prefix="msp_cli", suffix=".trees")
        os.close(fd)
        cls._tree_sequence.dump(cls._tree_sequence_file)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._tree_sequence_file)

    def verify_nodes(self, output_nodes, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(nodes=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        self.assertEqual(output, output_nodes)

    def test_nodes(self):
        cmd = "nodes"
        precision = 8
        stdout, stderr = capture_output(cli.tskit_main, [
            cmd, self._tree_sequence_file, "-p", str(precision)])
        self.assertEqual(len(stderr), 0)
        output_nodes = stdout.splitlines()
        self.verify_nodes(output_nodes, precision)

    def verify_edges(self, output_edges, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(edges=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        self.assertEqual(output, output_edges)

    def test_edges(self):
        cmd = "edges"
        precision = 8
        stdout, stderr = capture_output(cli.tskit_main, [
            cmd, self._tree_sequence_file, "-p", str(precision)])
        self.assertEqual(len(stderr), 0)
        output_edges = stdout.splitlines()
        self.verify_edges(output_edges, precision)

    def verify_sites(self, output_sites, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(sites=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        self.assertEqual(output, output_sites)

    def test_sites(self):
        cmd = "sites"
        precision = 8
        stdout, stderr = capture_output(cli.tskit_main, [
            cmd, self._tree_sequence_file, "-p", str(precision)])
        self.assertEqual(len(stderr), 0)
        output_sites = stdout.splitlines()
        self.verify_sites(output_sites, precision)

    def verify_mutations(self, output_mutations, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(mutations=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        self.assertEqual(output, output_mutations)

    def test_mutations(self):
        cmd = "mutations"
        precision = 8
        stdout, stderr = capture_output(cli.tskit_main, [
            cmd, self._tree_sequence_file, "-p", str(precision)])
        self.assertEqual(len(stderr), 0)
        output_mutations = stdout.splitlines()
        self.verify_mutations(output_mutations, precision)

    def verify_provenances(self, output_provenances):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(provenances=f)
            f.seek(0)
            output = f.read().splitlines()
        self.assertEqual(output, output_provenances)

    def test_provenances(self):
        cmd = "provenances"
        stdout, stderr = capture_output(cli.tskit_main, [cmd, self._tree_sequence_file])
        self.assertEqual(len(stderr), 0)
        output_provenances = stdout.splitlines()
        self.verify_provenances(output_provenances)

    def test_provenances_human(self):
        cmd = "provenances"
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, "-H", self._tree_sequence_file])
        self.assertEqual(len(stderr), 0)
        output_provenances = stdout.splitlines()
        # TODO Check the actual output here.
        self.assertGreater(len(output_provenances), 0)

    def verify_vcf(self, output_vcf):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.write_vcf(f)
            f.seek(0)
            vcf = f.read()
        self.assertEqual(output_vcf, vcf)

    def test_vcf(self):
        cmd = "vcf"
        stdout, stderr = capture_output(cli.tskit_main, [cmd, self._tree_sequence_file])
        self.assertEqual(len(stderr), 0)
        self.verify_vcf(stdout)


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
                cli.tskit_main, [
                    "upgrade", self.legacy_file_name, self.current_file_name])
            ts2 = tskit.load(self.current_file_name)
            self.assertEqual(stdout, "")
            self.assertEqual(stderr, "")
            # Quick checks to ensure we have the right tree sequence.
            # More thorough checks are done elsewhere.
            self.assertEqual(ts1.get_sample_size(), ts2.get_sample_size())
            self.assertEqual(ts1.num_edges, ts2.num_edges)
            self.assertEqual(ts1.get_num_trees(), ts2.get_num_trees())

    def test_duplicate_positions(self):
        ts = msprime.simulate(10, mutation_rate=10)
        for version in [2, 3]:
            tskit.dump_legacy(ts, self.legacy_file_name, version=version)
            root = h5py.File(self.legacy_file_name, "r+")
            root['mutations/position'][:] = 0
            root.close()
            stdout, stderr = capture_output(
                cli.tskit_main,
                ["upgrade", "-d", self.legacy_file_name, self.current_file_name])
            self.assertEqual(stdout, "")
            tsp = tskit.load(self.current_file_name)
            self.assertEqual(tsp.sample_size, ts.sample_size)
            self.assertEqual(tsp.num_sites, 1)

    def test_duplicate_positions_error(self):
        ts = msprime.simulate(10, mutation_rate=10)
        for version in [2, 3]:
            tskit.dump_legacy(ts, self.legacy_file_name, version=version)
            root = h5py.File(self.legacy_file_name, "r+")
            root['mutations/position'][:] = 0
            root.close()
            with mock.patch("tskit.cli.exit") as mocked_exit:
                stdout, stderr = capture_output(
                    cli.tskit_main,
                    ["upgrade",  self.legacy_file_name, self.current_file_name])
                self.assertEqual(mocked_exit.call_count, 1)
                self.assertEqual(stdout, "")
                self.assertEqual(stderr, "")
