# MIT License
#
# Copyright (c) 2018-2025 Tskit Developers
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
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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

    @pytest.mark.parametrize(
        "flags,expected",
        (
            [[], None],
            [["-P", "2"], 2],
            [["--ploidy", "5"], 5],
        ),
    )
    def test_vcf_ploidy(self, flags, expected):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, *flags])
        assert args.tree_sequence == tree_sequence
        assert args.ploidy == expected

    @pytest.mark.parametrize(
        "flags,expected",
        (
            [[], "1"],
            [["-c", "chrX"], "chrX"],
            [["--contig-id", "chr20"], "chr20"],
        ),
    )
    def test_vcf_contig_id(self, flags, expected):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, *flags])
        assert args.tree_sequence == tree_sequence
        assert args.contig_id == expected

    @pytest.mark.parametrize(
        "flags,expected",
        (
            [[], False],
            [["-0"], True],
            [["--allow-position-zero"], True],
        ),
    )
    def test_vcf_allow_position_zero(self, flags, expected):
        parser = cli.get_tskit_parser()
        cmd = "vcf"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, *flags])
        assert args.tree_sequence == tree_sequence
        assert args.allow_position_zero == expected

    @pytest.mark.parametrize(
        "cmd",
        [
            "info",
            "trees",
            "vcf",
            "nodes",
            "edges",
            "sites",
            "mutations",
            "migrations",
            "individuals",
            "populations",
            "provenances",
        ],
    )
    def test_tree_sequence_argument_optional(self, cmd):
        # Omitting the positional argument selects stdin (tree_sequence is None);
        # providing a path stores the path string.
        parser = cli.get_tskit_parser()
        assert parser.parse_args([cmd]).tree_sequence is None
        assert parser.parse_args([cmd, "test.trees"]).tree_sequence == "test.trees"

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

    def test_migrations_default_values(self):
        parser = cli.get_tskit_parser()
        cmd = "migrations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence])
        assert args.tree_sequence == tree_sequence

    def test_migrations_short_args(self):
        parser = cli.get_tskit_parser()
        cmd = "migrations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "-p", "2"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 2

    def test_migrations_long_args(self):
        parser = cli.get_tskit_parser()
        cmd = "migrations"
        tree_sequence = "test.trees"
        args = parser.parse_args([cmd, tree_sequence, "--precision", "5"])
        assert args.tree_sequence == tree_sequence
        assert args.precision == 5

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
    Tests the output of tskit to ensure it's correct.
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
        fd, cls._tree_sequence_file = tempfile.mkstemp(prefix="tsk_cli", suffix=".trees")
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

    def verify_migrations(self, output_migrations, precision):
        with tempfile.TemporaryFile("w+") as f:
            self._tree_sequence.dump_text(migrations=f, precision=precision)
            f.seek(0)
            output = f.read().splitlines()
        assert output == output_migrations

    def test_migrations(self):
        cmd = "migrations"
        precision = 4
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, self._tree_sequence_file, "-p", str(precision)]
        )
        assert len(stderr) == 0
        output_migrations = stdout.splitlines()
        self.verify_migrations(output_migrations, precision)

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
            self._tree_sequence.write_vcf(f, allow_position_zero=True)
            f.seek(0)
            vcf = f.read()
        assert output_vcf == vcf

    def test_vcf(self):
        cmd = "vcf"
        stdout, stderr = capture_output(
            cli.tskit_main, [cmd, "-0", self._tree_sequence_file]
        )
        assert len(stderr) == 0
        self.verify_vcf(stdout)

    def test_vcf_stdin(self):
        # Omitting the path argument reads the tree sequence from stdin. The
        # low-level loader requires a real file descriptor, so sys.stdin must be
        # patched with an actual open binary file (not e.g. an io.BytesIO).
        with open(self._tree_sequence_file, "rb") as f:
            with mock.patch("sys.stdin", f):
                stdout, stderr = capture_output(cli.tskit_main, ["vcf", "-0"])
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


class TestVCFZeroPosition:
    """
    Tests that we can write VCF files with position 0.
    """

    def test_zero_position(self, tmp_path):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        ts.dump(tmp_path / "test.trees")
        with pytest.raises(ValueError):
            capture_output(cli.tskit_main, ["vcf", str(tmp_path / "test.trees")])
        stdout, stderr = capture_output(
            cli.tskit_main, ["vcf", "-0", str(tmp_path / "test.trees")]
        )
        assert len(stderr) == 0


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

    def test_migrations(self):
        self.verify("migrations")

    def test_provenances(self):
        self.verify("provenances")


@pytest.fixture(scope="module")
def treeseq_file(tmp_path_factory):
    """
    A tree sequence dumped to file, containing migrations, mutations and
    individuals so that every loading subcommand has something to output.
    """
    ts = msprime.simulate(
        length=1,
        recombination_rate=2,
        mutation_rate=2,
        random_seed=1,
        migration_matrix=[[0, 1], [1, 0]],
        population_configurations=[msprime.PopulationConfiguration(5) for _ in range(2)],
        record_migrations=True,
    )
    assert ts.num_migrations > 0
    ts = tsutil.insert_random_ploidy_individuals(ts, samples_only=True)
    path = tmp_path_factory.mktemp("tsk_cli_stdin") / "stdin.trees"
    ts.dump(path)
    return str(path)


# The loading subcommands and any extra flags they need to produce output.
STDIN_SUBCOMMANDS = [
    ["info"],
    ["trees"],
    ["vcf", "-0"],
    ["nodes"],
    ["edges"],
    ["sites"],
    ["mutations"],
    ["migrations"],
    ["individuals"],
    ["populations"],
    ["provenances"],
]


class TestStdin:
    """
    Tests that reading from stdin (omitting the path argument) produces the same
    output as loading from a file, for every loading subcommand.
    """

    @pytest.mark.parametrize("subcommand", STDIN_SUBCOMMANDS)
    def test_stdin_matches_file(self, treeseq_file, subcommand):
        file_stdout, file_stderr = capture_output(
            cli.tskit_main, [*subcommand, treeseq_file]
        )
        with open(treeseq_file, "rb") as f:
            with mock.patch("sys.stdin", f):
                stdin_stdout, stdin_stderr = capture_output(cli.tskit_main, subcommand)
        assert file_stderr == ""
        assert stdin_stderr == ""
        assert len(file_stdout) > 0
        assert stdin_stdout == file_stdout


class TestStdinErrors:
    """
    Tests that errors loading from stdin are reported cleanly.
    """

    def run_info_stdin(self, path):
        with mock.patch("sys.exit", side_effect=TestException) as mocked_exit:
            with open(path, "rb") as f:
                with mock.patch("sys.stdin", f):
                    with pytest.raises(TestException):
                        capture_output(cli.tskit_main, ["info"])
        return mocked_exit.call_args[0][0]

    def test_empty_stdin(self, tmp_path):
        path = tmp_path / "empty.trees"
        path.write_bytes(b"")
        assert self.run_info_stdin(path) == "Load error: End of file"

    def test_garbage_stdin(self, tmp_path):
        path = tmp_path / "garbage.trees"
        path.write_bytes(b"not a tree sequence at all")
        message = self.run_info_stdin(path)
        assert message.startswith("Load error: File not in kastore format")

    def test_truncated_stdin(self, tmp_path, treeseq_file):
        path = tmp_path / "truncated.trees"
        with open(treeseq_file, "rb") as f:
            path.write_bytes(f.read(100))
        message = self.run_info_stdin(path)
        assert message.startswith("Load error: File not in kastore format")


class TestStdinSubprocess:
    """
    End-to-end tests that feed the tree sequence through a real OS pipe. Unlike
    the in-process tests (which mock sys.stdin with a seekable file), these
    exercise the genuine non-seekable stdin path.
    """

    def run_cli(self, args, input_bytes):
        return subprocess.run(
            [sys.executable, "-m", "tskit", *args],
            input=input_bytes,
            capture_output=True,
        )

    @pytest.mark.parametrize("subcommand", [["info"], ["vcf", "-0"], ["nodes"]])
    def test_stdin_pipe_matches_file(self, treeseq_file, subcommand):
        with open(treeseq_file, "rb") as f:
            ts_bytes = f.read()
        stdin_result = self.run_cli(subcommand, ts_bytes)
        file_result = self.run_cli([*subcommand, treeseq_file], b"")
        assert stdin_result.returncode == 0
        assert stdin_result.stderr == b""
        assert len(stdin_result.stdout) > 0
        assert stdin_result.stdout == file_result.stdout

    def test_empty_pipe(self):
        result = self.run_cli(["info"], b"")
        assert result.returncode != 0
        assert b"End of file" in result.stderr

    def test_garbage_pipe(self):
        result = self.run_cli(["info"], b"not a tree sequence")
        assert result.returncode != 0
        assert b"not in kastore format" in result.stderr
