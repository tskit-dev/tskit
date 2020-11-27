# MIT License
#
# Copyright (c) 2018-2020 Tskit Developers
# Copyright (c) 2015-2018 University of Oxford
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
Test cases for the low level C interface to tskit.
"""
import collections
import inspect
import itertools
import os
import random
import tempfile

import msprime
import numpy as np
import pytest

import _tskit


def get_tracked_sample_counts(st, tracked_samples):
    """
    Returns a list giving the number of samples in the specified list
    that are in the subtree rooted at each node.
    """
    nu = [0 for j in range(st.get_num_nodes())]
    for j in tracked_samples:
        # Duplicates not permitted.
        assert nu[j] == 0
        u = j
        while u != _tskit.NULL:
            nu[u] += 1
            u = st.get_parent(u)
    return nu


def get_sample_counts(tree_sequence, st):
    """
    Returns a list of the sample node counts for the specified tree.
    """
    nu = [0 for j in range(st.get_num_nodes())]
    for j in range(tree_sequence.get_num_samples()):
        u = j
        while u != _tskit.NULL:
            nu[u] += 1
            u = st.get_parent(u)
    return nu


class LowLevelTestCase:
    """
    Superclass of tests for the low-level interface.
    """

    def verify_tree_dict(self, n, pi):
        """
        Verifies that the specified tree in dict format is a
        consistent coalescent history for a sample of size n.
        """
        assert len(pi) <= 2 * n - 1
        # _tskit.NULL should not be a node
        assert _tskit.NULL not in pi
        # verify the root is equal for all samples
        root = 0
        while pi[root] != _tskit.NULL:
            root = pi[root]
        for j in range(n):
            k = j
            while pi[k] != _tskit.NULL:
                k = pi[k]
            assert k == root
        # 0 to n - 1 inclusive should always be nodes
        for j in range(n):
            assert j in pi
        num_children = collections.defaultdict(int)
        for j in pi.keys():
            num_children[pi[j]] += 1
        # nodes 0 to n are samples.
        for j in range(n):
            assert pi[j] != 0
            assert num_children[j] == 0
        # All non-sample nodes should be binary
        for j in pi.keys():
            if j > n:
                assert num_children[j] >= 2

    def get_example_tree_sequence(
        self, sample_size=10, length=1, mutation_rate=1, random_seed=1
    ):
        ts = msprime.simulate(
            sample_size,
            recombination_rate=0.1,
            mutation_rate=mutation_rate,
            random_seed=random_seed,
            length=length,
        )
        return ts.ll_tree_sequence

    def get_example_tree_sequences(self):
        yield self.get_example_tree_sequence()
        yield self.get_example_tree_sequence(2, 10)
        yield self.get_example_tree_sequence(20, 10)
        yield self.get_example_migration_tree_sequence()

    def get_example_migration_tree_sequence(self):
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        )
        return ts.ll_tree_sequence

    def verify_iterator(self, iterator):
        """
        Checks that the specified non-empty iterator implements the
        iterator protocol correctly.
        """
        list_ = list(iterator)
        assert len(list_) > 0
        for _ in range(10):
            with pytest.raises(StopIteration):
                next(iterator)


class MetadataTestMixin:
    metadata_tables = [
        "node",
        "edge",
        "site",
        "mutation",
        "migration",
        "individual",
        "population",
    ]


class TestTableCollection(LowLevelTestCase):
    """
    Tests for the low-level TableCollection class
    """

    def test_file_errors(self):
        tc1 = _tskit.TableCollection(1)
        self.get_example_tree_sequence().dump_tables(tc1)

        def loader(*args):
            tc = _tskit.TableCollection(1)
            tc.load(*args)

        for func in [tc1.dump, loader]:
            with pytest.raises(TypeError):
                func()
            for bad_type in [None, [], {}]:
                with pytest.raises(TypeError):
                    func(bad_type)

    def test_dump_equality(self, tmp_path):
        for ts in self.get_example_tree_sequences():
            tc = _tskit.TableCollection(sequence_length=ts.get_sequence_length())
            ts.dump_tables(tc)
            with open(tmp_path / "tmp.trees", "wb") as f:
                tc.dump(f)
            with open(tmp_path / "tmp.trees", "rb") as f:
                tc2 = _tskit.TableCollection()
                tc2.load(f)
            assert tc.equals(tc2)

    def test_reference_deletion(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        tc = ts.tables._ll_tables
        # Get references to all the tables
        tables = [
            tc.individuals,
            tc.nodes,
            tc.edges,
            tc.migrations,
            tc.sites,
            tc.mutations,
            tc.populations,
            tc.provenances,
        ]
        del tc
        for _ in range(10):
            for table in tables:
                assert len(str(table)) > 0

    def test_set_sequence_length_errors(self):
        tables = _tskit.TableCollection(1)
        with pytest.raises(TypeError):
            del tables.sequence_length
        for bad_value in ["sdf", None, []]:
            with pytest.raises(TypeError):
                tables.sequence_length = bad_value

    def test_set_sequence_length(self):
        tables = _tskit.TableCollection(1)
        assert tables.sequence_length == 1
        for value in [-1, 1e6, 1e-22, 1000, 2 ** 32, -10000]:
            tables.sequence_length = value
            assert tables.sequence_length == value

    def test_set_metadata_errors(self):
        tables = _tskit.TableCollection(1)
        with pytest.raises(AttributeError):
            del tables.metadata
        for bad_value in ["bytes only", 59, 43.4, None, []]:
            with pytest.raises(TypeError):
                tables.metadata = bad_value

    def test_set_metadata(self):
        tables = _tskit.TableCollection(1)
        assert tables.metadata == b""
        for value in [b"foo", b"", "💩".encode(), b"null char \0 in string"]:
            tables.metadata = value
            tables.metadata_schema = "Test we have two separate fields"
            assert tables.metadata == value

    def test_set_metadata_schema_errors(self):
        tables = _tskit.TableCollection(1)
        with pytest.raises(AttributeError):
            del tables.metadata_schema
        for bad_value in [59, 43.4, None, []]:
            with pytest.raises(TypeError):
                tables.metadata_schema = bad_value

    def test_set_metadata_schema(self):
        tables = _tskit.TableCollection(1)
        assert tables.metadata_schema == ""
        for value in ["foo", "", "💩", "null char \0 in string"]:
            tables.metadata_schema = value
            tables.metadata = b"Test we have two separate fields"
            assert tables.metadata_schema == value

    def test_simplify_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.tables._ll_tables
        with pytest.raises(TypeError):
            tc.simplify()
        with pytest.raises(ValueError):
            tc.simplify("asdf")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], keep_unary="sdf")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], keep_input_roots="sdf")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], filter_populations="x")
        with pytest.raises(_tskit.LibraryError):
            tc.simplify([0, -1])

    def test_link_ancestors_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.tables._ll_tables
        with pytest.raises(TypeError):
            tc.link_ancestors()
        with pytest.raises(TypeError):
            tc.link_ancestors([0, 1])
        with pytest.raises(ValueError):
            tc.link_ancestors(samples=[0, 1], ancestors="sdf")
        with pytest.raises(ValueError):
            tc.link_ancestors(samples="sdf", ancestors=[0, 1])
        with pytest.raises(_tskit.LibraryError):
            tc.link_ancestors(samples=[0, 1], ancestors=[11, -1])
        with pytest.raises(_tskit.LibraryError):
            tc.link_ancestors(samples=[0, -1], ancestors=[11])

    def test_link_ancestors(self):
        ts = msprime.simulate(2, random_seed=1)
        tc = ts.tables._ll_tables
        edges = tc.link_ancestors([0, 1], [3])
        assert isinstance(edges, _tskit.EdgeTable)
        del edges
        assert tc.edges.num_rows == 2

    def test_subset_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.tables._ll_tables
        with pytest.raises(TypeError):
            tc.subset(np.array(["a"]))
        with pytest.raises(ValueError):
            tc.subset(np.array([[1], [2]], dtype="int32"))
        with pytest.raises(TypeError):
            tc.subset()
        with pytest.raises(_tskit.LibraryError):
            tc.subset(np.array([100, 200], dtype="int32"))

    def test_union_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.tables._ll_tables
        tc2 = tc
        with pytest.raises(TypeError):
            tc.union(tc2, np.array(["a"]))
        with pytest.raises(ValueError):
            tc.union(tc2, np.array([0], dtype="int32"))
        with pytest.raises(TypeError):
            tc.union(tc2)
        with pytest.raises(TypeError):
            tc.union()
        node_mapping = np.arange(ts.num_nodes, dtype="int32")
        node_mapping[0] = 1200
        with pytest.raises(_tskit.LibraryError):
            tc.union(tc2, node_mapping)
        node_mapping = np.array(
            [node_mapping.tolist(), node_mapping.tolist()], dtype="int32"
        )
        with pytest.raises(ValueError):
            tc.union(tc2, node_mapping)
        with pytest.raises(ValueError):
            tc.union(tc2, np.array([[1], [2]], dtype="int32"))

    def test_ibd_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.tables._ll_tables
        with pytest.raises(TypeError):
            tc.find_ibd()
        for bad_samples in ["sdf", None, {}]:
            with pytest.raises(ValueError):
                tc.find_ibd(bad_samples)
        for not_enough_samples in [[], [0]]:
            with pytest.raises(ValueError):
                tc.find_ibd(not_enough_samples)
        # input array must be 2D
        with pytest.raises(ValueError):
            tc.find_ibd([[[1], [1]]])
        # Input array must be (n, 2)
        with pytest.raises(ValueError):
            tc.find_ibd([[1, 1, 1]])
        for bad_float in ["sdf", None, {}]:
            with pytest.raises(TypeError):
                tc.find_ibd([(0, 1)], min_length=bad_float)
            with pytest.raises(TypeError):
                tc.find_ibd([(0, 1)], max_time=bad_float)
        with pytest.raises(_tskit.LibraryError):
            tc.find_ibd([(0, 1)], max_time=-1)
        with pytest.raises(_tskit.LibraryError):
            tc.find_ibd([(0, 1)], min_length=-1)

    def test_ibd_output_no_recomb(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.tables._ll_tables
        segs = tc.find_ibd([(0, 1), (2, 3)])
        assert isinstance(segs, dict)
        assert len(segs) > 0
        for key, value in segs.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(value, dict)
            assert len(value) == 3
            assert list(value["left"]) == [0]
            assert list(value["right"]) == [1]
            assert len(value["node"]) == 1

    def test_ibd_output_recomb(self):
        ts = msprime.simulate(10, recombination_rate=1, random_seed=1)
        assert ts.num_trees > 1
        tc = ts.tables._ll_tables
        segs = tc.find_ibd([(0, 1), (2, 3)])
        assert isinstance(segs, dict)
        assert len(segs) > 0
        for key, value in segs.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(value, dict)
            assert len(value) == 3

    def test_equals_bad_args(self):
        ts = msprime.simulate(10, random_seed=1242)
        tc = ts.tables._ll_tables
        with pytest.raises(TypeError):
            tc.equals()
        with pytest.raises(TypeError):
            tc.equals(None)
        assert tc.equals(tc)
        with pytest.raises(TypeError):
            tc.equals(tc, no_such_arg=1)
        bad_bool = "x"
        with pytest.raises(TypeError):
            tc.equals(tc, ignore_metadata=bad_bool)
        with pytest.raises(TypeError):
            tc.equals(tc, ignore_ts_metadata=bad_bool)
        with pytest.raises(TypeError):
            tc.equals(tc, ignore_provenance=bad_bool)
        with pytest.raises(TypeError):
            tc.equals(tc, ignore_timestamps=bad_bool)


class TestTableMethodsErrors:
    """
    Tests for the error handling of errors in the low-level tables.
    """

    def yield_tables(self, ts):
        for table in ts.tables.name_map.values():
            yield table.ll_table

    def test_equals_bad_args(self, ts_fixture):
        for ll_table in self.yield_tables(ts_fixture):
            assert ll_table.equals(ll_table)
            with pytest.raises(TypeError):
                ll_table.equals(None)
            with pytest.raises(TypeError):
                ll_table.equals(ll_table, no_such_arg="")
            uninit_other = type(ll_table).__new__(type(ll_table))
            with pytest.raises(SystemError):
                ll_table.equals(uninit_other)

    def test_get_row_bad_args(self, ts_fixture):
        for ll_table in self.yield_tables(ts_fixture):
            assert ll_table.get_row(0) is not None
            with pytest.raises(TypeError):
                ll_table.get_row(no_such_arg="")

    @pytest.mark.parametrize("table", ["nodes", "individuals"])
    def test_flag_underflow_overflow(self, table):
        tables = _tskit.TableCollection(1)
        table = getattr(tables, table)
        table.add_row(flags=0)
        table.add_row(flags=(1 << 32) - 1)
        with pytest.raises(OverflowError, match="unsigned int32 >= than 2\\^32"):
            table.add_row(flags=1 << 32)
        with pytest.raises(OverflowError, match="int too big to convert"):
            table.add_row(flags=1 << 64)
        with pytest.raises(OverflowError, match="int too big to convert"):
            table.add_row(flags=1 << 256)
        with pytest.raises(
            ValueError, match="Can't convert negative value to unsigned int"
        ):
            table.add_row(flags=-1)

    def test_index(self):
        tc = msprime.simulate(10, random_seed=42).tables._ll_tables
        assert tc.indexes["edge_insertion_order"].dtype == np.int32
        assert tc.indexes["edge_removal_order"].dtype == np.int32
        assert np.array_equal(
            tc.indexes["edge_insertion_order"], np.arange(18, dtype=np.int32)
        )
        assert np.array_equal(
            tc.indexes["edge_removal_order"], np.arange(18, dtype=np.int32)[::-1]
        )
        tc.drop_index()
        assert tc.indexes == {}
        tc.build_index()
        assert np.array_equal(
            tc.indexes["edge_insertion_order"], np.arange(18, dtype=np.int32)
        )
        assert np.array_equal(
            tc.indexes["edge_removal_order"], np.arange(18, dtype=np.int32)[::-1]
        )

        modify_indexes = tc.indexes
        modify_indexes["edge_insertion_order"] = np.arange(42, 42 + 18, dtype=np.int32)
        modify_indexes["edge_removal_order"] = np.arange(
            4242, 4242 + 18, dtype=np.int32
        )
        tc.indexes = modify_indexes
        assert np.array_equal(
            tc.indexes["edge_insertion_order"], np.arange(42, 42 + 18, dtype=np.int32)
        )
        assert np.array_equal(
            tc.indexes["edge_removal_order"], np.arange(4242, 4242 + 18, dtype=np.int32)
        )

    def test_no_indexes(self):
        tc = msprime.simulate(10, random_seed=42).tables._ll_tables
        tc.drop_index()
        assert tc.indexes == {}

    def test_bad_indexes(self):
        tc = msprime.simulate(10, random_seed=42).tables._ll_tables
        for col in ("insertion", "removal"):
            d = tc.indexes
            d[f"edge_{col}_order"] = d[f"edge_{col}_order"][:-1]
            with pytest.raises(
                ValueError,
                match="^edge_insertion_order and"
                " edge_removal_order must be the same"
                " length$",
            ):
                tc.indexes = d
        d = tc.indexes
        for col in ("insertion", "removal"):
            d[f"edge_{col}_order"] = d[f"edge_{col}_order"][:-1]
        with pytest.raises(
            ValueError,
            match="^edge_insertion_order and edge_removal_order must be"
            " the same length as the number of edges$",
        ):
            tc.indexes = d

        # Both columns must be provided, if one is
        for col in ("insertion", "removal"):
            d = tc.indexes
            del d[f"edge_{col}_order"]
            with pytest.raises(
                TypeError,
                match="^edge_insertion_order and "
                "edge_removal_order must be specified "
                "together$",
            ):
                tc.indexes = d

        tc = msprime.simulate(
            10, recombination_rate=10, random_seed=42
        ).tables._ll_tables
        modify_indexes = tc.indexes
        shape = modify_indexes["edge_insertion_order"].shape
        modify_indexes["edge_insertion_order"] = np.zeros(shape, dtype=np.int32)
        modify_indexes["edge_removal_order"] = np.zeros(shape, dtype=np.int32)
        tc.indexes = modify_indexes
        ts = _tskit.TreeSequence()
        with pytest.raises(
            _tskit.LibraryError,
            match="^Bad edges: contradictory children for a given"
            " parent over an interval$",
        ):
            ts.load_tables(tc, build_indexes=False)

        modify_indexes["edge_insertion_order"] = np.full(shape, 2 ** 30, dtype=np.int32)
        modify_indexes["edge_removal_order"] = np.full(shape, 2 ** 30, dtype=np.int32)
        tc.indexes = modify_indexes
        ts = _tskit.TreeSequence()
        with pytest.raises(_tskit.LibraryError, match="^Edge out of bounds$"):
            ts.load_tables(tc, build_indexes=False)


class TestTreeSequence(LowLevelTestCase, MetadataTestMixin):
    """
    Tests for the low-level interface for the TreeSequence.
    """

    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(prefix="msp_ll_ts_")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)

    def test_file_errors(self):
        ts1 = self.get_example_tree_sequence()

        def loader(*args):
            ts2 = _tskit.TreeSequence()
            ts2.load(*args)

        for func in [ts1.dump, loader]:
            with pytest.raises(TypeError):
                func()
            for bad_type in [None, [], {}]:
                with pytest.raises(TypeError):
                    func(bad_type)

    def test_initial_state(self):
        # Check the initial state to make sure that it is empty.
        ts = _tskit.TreeSequence()
        with pytest.raises(ValueError):
            ts.get_num_samples()
        with pytest.raises(ValueError):
            ts.get_sequence_length()
        with pytest.raises(ValueError):
            ts.get_num_trees()
        with pytest.raises(ValueError):
            ts.get_num_edges()
        with pytest.raises(ValueError):
            ts.get_num_mutations()
        with pytest.raises(ValueError):
            ts.get_num_migrations()
        with pytest.raises(ValueError):
            ts.get_num_migrations()
        with pytest.raises(ValueError):
            ts.get_genotype_matrix()
        with pytest.raises(ValueError):
            ts.dump()

    def test_num_nodes(self):
        for ts in self.get_example_tree_sequences():
            max_node = 0
            for j in range(ts.get_num_edges()):
                _, _, parent, child, _ = ts.get_edge(j)
                for node in [parent, child]:
                    if node > max_node:
                        max_node = node
            assert max_node + 1 == ts.get_num_nodes()

    def test_dump_equality(self, tmp_path):
        for ts in self.get_example_tree_sequences():
            tables = _tskit.TableCollection(sequence_length=ts.get_sequence_length())
            ts.dump_tables(tables)
            tables.compute_mutation_times()
            ts = _tskit.TreeSequence()
            ts.load_tables(tables)
            with open(tmp_path / "temp.trees", "wb") as f:
                ts.dump(f)
            with open(tmp_path / "temp.trees", "rb") as f:
                ts2 = _tskit.TreeSequence()
                ts2.load(f)
            tc = _tskit.TableCollection(ts.get_sequence_length())
            ts.dump_tables(tc)
            tc2 = _tskit.TableCollection(ts2.get_sequence_length())
            ts2.dump_tables(tc2)
            assert tc.equals(tc2)

    def verify_mutations(self, ts):
        mutations = [ts.get_mutation(j) for j in range(ts.get_num_mutations())]
        assert ts.get_num_mutations() > 0
        assert len(mutations) == ts.get_num_mutations()
        # Check the form of the mutations
        for j, (position, nodes, index) in enumerate(mutations):
            assert j == index
            for node in nodes:
                assert isinstance(node, int)
                assert node >= 0
                assert node <= ts.get_num_nodes()
            assert isinstance(position, float)
            assert position > 0
            assert position < ts.get_sequence_length()
        # mutations must be sorted by position order.
        assert mutations == sorted(mutations)

    def test_get_edge_interface(self):
        for ts in self.get_example_tree_sequences():
            num_edges = ts.get_num_edges()
            # We don't accept Python negative indexes here.
            with pytest.raises(IndexError):
                ts.get_edge(-1)
            for j in [0, 10, 10 ** 6]:
                with pytest.raises(IndexError):
                    ts.get_edge(num_edges + j)
            for x in [None, "", {}, []]:
                with pytest.raises(TypeError):
                    ts.get_edge(x)

    def test_get_node_interface(self):
        for ts in self.get_example_tree_sequences():
            num_nodes = ts.get_num_nodes()
            # We don't accept Python negative indexes here.
            with pytest.raises(IndexError):
                ts.get_node(-1)
            for j in [0, 10, 10 ** 6]:
                with pytest.raises(IndexError):
                    ts.get_node(num_nodes + j)
            for x in [None, "", {}, []]:
                with pytest.raises(TypeError):
                    ts.get_node(x)

    def test_get_genotype_matrix_interface(self):
        for ts in self.get_example_tree_sequences():
            num_samples = ts.get_num_samples()
            num_sites = ts.get_num_sites()
            G = ts.get_genotype_matrix()
            assert G.shape == (num_sites, num_samples)
            with pytest.raises(TypeError):
                ts.get_genotype_matrix(isolated_as_missing=None)
            with pytest.raises(TypeError):
                ts.get_genotype_matrix(alleles="XYZ")
            with pytest.raises(ValueError):
                ts.get_genotype_matrix(alleles=tuple())
            G = ts.get_genotype_matrix(isolated_as_missing=False)
            assert G.shape == (num_sites, num_samples)

    def test_get_genotype_matrix_missing_data(self):
        tables = _tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.sites.add_row(0.1, "A")
        tables.build_index()
        ts = _tskit.TreeSequence(0)
        ts.load_tables(tables)
        G = ts.get_genotype_matrix(isolated_as_missing=False)
        assert np.all(G == 0)
        G = ts.get_genotype_matrix(isolated_as_missing=True)
        assert np.all(G == -1)
        G = ts.get_genotype_matrix()
        assert np.all(G == -1)

    def test_get_migration_interface(self):
        ts = self.get_example_migration_tree_sequence()
        for bad_type in ["", None, {}]:
            with pytest.raises(TypeError):
                ts.get_migration(bad_type)
        num_records = ts.get_num_migrations()
        # We don't accept Python negative indexes here.
        with pytest.raises(IndexError):
            ts.get_migration(-1)
        for j in [0, 10, 10 ** 6]:
            with pytest.raises(IndexError):
                ts.get_migration(num_records + j)

    def test_get_samples(self):
        for ts in self.get_example_tree_sequences():
            # get_samples takes no arguments.
            with pytest.raises(TypeError):
                ts.get_samples(0)
            assert np.array_equal(
                np.arange(ts.get_num_samples(), dtype=np.int32), ts.get_samples()
            )

    def test_genealogical_nearest_neighbours(self):
        for ts in self.get_example_tree_sequences():
            with pytest.raises(TypeError):
                ts.genealogical_nearest_neighbours()
            with pytest.raises(TypeError):
                ts.genealogical_nearest_neighbours(focal=None)
            with pytest.raises(TypeError):
                ts.genealogical_nearest_neighbours(
                    focal=ts.get_samples(),
                    reference_sets={},
                )
            with pytest.raises(ValueError):
                ts.genealogical_nearest_neighbours(
                    focal=ts.get_samples(),
                    reference_sets=[],
                )

            bad_array_values = ["", {}, "x", [[[0], [1, 2]]]]
            for bad_array_value in bad_array_values:
                with pytest.raises(ValueError):
                    ts.genealogical_nearest_neighbours(
                        focal=bad_array_value,
                        reference_sets=[[0], [1]],
                    )
                with pytest.raises(ValueError):
                    ts.genealogical_nearest_neighbours(
                        focal=ts.get_samples(),
                        reference_sets=[[0], bad_array_value],
                    )
                with pytest.raises(ValueError):
                    ts.genealogical_nearest_neighbours(
                        focal=ts.get_samples(),
                        reference_sets=[bad_array_value],
                    )
            focal = ts.get_samples()
            A = ts.genealogical_nearest_neighbours(focal, [focal[2:], focal[:2]])
            assert A.shape == (len(focal), 2)

    def test_mean_descendants(self):
        for ts in self.get_example_tree_sequences():
            with pytest.raises(TypeError):
                ts.mean_descendants()
            with pytest.raises(TypeError):
                ts.mean_descendants(reference_sets={})
            with pytest.raises(ValueError):
                ts.mean_descendants(reference_sets=[])

            bad_array_values = ["", {}, "x", [[[0], [1, 2]]]]
            for bad_array_value in bad_array_values:
                with pytest.raises(ValueError):
                    ts.mean_descendants(
                        reference_sets=[[0], bad_array_value],
                    )
                with pytest.raises(ValueError):
                    ts.mean_descendants(reference_sets=[bad_array_value])
            focal = ts.get_samples()
            A = ts.mean_descendants([focal[2:], focal[:2]])
            assert A.shape == (ts.get_num_nodes(), 2)

    def test_metadata_schemas(self):
        tables = _tskit.TableCollection(1.0)
        # Set the schema
        for table_name in self.metadata_tables:
            table = getattr(tables, f"{table_name}s")
            table.metadata_schema = f"{table_name} test metadata schema"
        # Read back via ll tree sequence
        tables.build_index()
        ts = _tskit.TreeSequence()
        ts.load_tables(tables)
        schemas = ts.get_table_metadata_schemas()
        for table_name in self.metadata_tables:
            assert getattr(schemas, table_name) == f"{table_name} test metadata schema"
        # Clear and read back again
        for table_name in self.metadata_tables:
            getattr(tables, f"{table_name}s").metadata_schema = ""
        ts = _tskit.TreeSequence()
        ts.load_tables(tables)
        schemas = ts.get_table_metadata_schemas()
        for table_name in self.metadata_tables:
            assert getattr(schemas, table_name) == ""

    def test_metadata(self):
        tables = _tskit.TableCollection(1)
        tables.build_index()
        ts = _tskit.TreeSequence()
        ts.load_tables(tables)
        assert ts.get_metadata() == b""
        for value in [b"foo", b"", "💩".encode(), b"null char \0 in string"]:
            tables.metadata = value
            ts = _tskit.TreeSequence()
            ts.load_tables(tables)
            assert ts.get_metadata() == value

    def test_metadata_schema(self):
        tables = _tskit.TableCollection(1)
        tables.build_index()
        ts = _tskit.TreeSequence()
        ts.load_tables(tables)
        assert ts.get_metadata_schema() == ""
        for value in ["foo", "", "💩", "null char \0 in string"]:
            tables.metadata_schema = value
            ts = _tskit.TreeSequence()
            ts.load_tables(tables)
            assert ts.get_metadata_schema() == value

    def test_kc_distance_errors(self):
        ts1 = self.get_example_tree_sequence(10)
        with pytest.raises(TypeError):
            ts1.get_kc_distance()
        with pytest.raises(TypeError):
            ts1.get_kc_distance(ts1)
        for bad_tree in [None, "tree", 0]:
            with pytest.raises(TypeError):
                ts1.get_kc_distance(bad_tree, lambda_=0)
        for bad_value in ["tree", [], None]:
            with pytest.raises(TypeError):
                ts1.get_kc_distance(ts1, lambda_=bad_value)

        # Different numbers of samples fail.
        ts2 = self.get_example_tree_sequence(11)
        self.verify_kc_library_error(ts1, ts2)

        # Different sequence lengths fail.
        ts2 = self.get_example_tree_sequence(10, length=11)
        self.verify_kc_library_error(ts1, ts2)

    def verify_kc_library_error(self, ts1, ts2):
        with pytest.raises(_tskit.LibraryError):
            ts1.get_kc_distance(ts2, 0)

    def test_kc_distance(self):
        ts1 = self.get_example_tree_sequence(10, random_seed=123456)
        ts2 = self.get_example_tree_sequence(10, random_seed=1234)
        for lambda_ in [-1, 0, 1, 1000, -1e300]:
            x1 = ts1.get_kc_distance(ts2, lambda_)
            x2 = ts2.get_kc_distance(ts1, lambda_)
            assert x1 == x2

    def test_load_tables_build_indexes(self):
        for ts in self.get_example_tree_sequences():
            tables = _tskit.TableCollection(sequence_length=ts.get_sequence_length())
            ts.dump_tables(tables)
            tables.drop_index()

            # Tables not in tc but rebuilt
            ts2 = _tskit.TreeSequence()
            ts2.load_tables(tables, build_indexes=True)
            tables2 = _tskit.TableCollection(sequence_length=ts.get_sequence_length())
            ts2.dump_tables(tables2)
            assert tables2.has_index()

            # Tables not in tc, not rebuilt so error
            ts3 = _tskit.TreeSequence()
            with pytest.raises(
                _tskit.LibraryError, match="Table collection must be indexed"
            ):
                ts3.load_tables(tables)

            # Tables in tc, not rebuilt
            tables.build_index()
            ts4 = _tskit.TreeSequence()
            ts4.load_tables(tables, build_indexes=False)
            tables4 = _tskit.TableCollection(sequence_length=ts.get_sequence_length())
            ts4.dump_tables(tables4)
            assert tables4.has_index()

    def test_clear_table(self, ts_fixture):
        tables = _tskit.TableCollection(
            sequence_length=ts_fixture.get_sequence_length()
        )
        ts_fixture.ll_tree_sequence.dump_tables(tables)
        tables.clear()
        data_tables = [
            "individuals",
            "nodes",
            "edges",
            "migrations",
            "sites",
            "mutations",
            "populations",
        ]
        for table in data_tables:
            assert getattr(tables, f"{table}").num_rows == 0
            assert len(getattr(tables, f"{table}").metadata_schema) != 0
        assert tables.provenances.num_rows > 0
        assert len(tables.metadata) > 0
        assert len(tables.metadata_schema) > 0

        tables.clear(clear_provenance=True)
        assert tables.provenances.num_rows == 0
        for table in data_tables:
            assert len(getattr(tables, f"{table}").metadata_schema) != 0
        assert len(tables.metadata) > 0
        assert len(tables.metadata_schema) > 0

        tables.clear(clear_metadata_schemas=True)
        for table in data_tables:
            assert len(getattr(tables, f"{table}").metadata_schema) == 0
        assert len(tables.metadata) > 0
        assert len(tables.metadata_schema) > 0

        tables.clear(clear_ts_metadata_and_schema=True)
        assert len(tables.metadata) == 0
        assert len(tables.metadata_schema) == 0


class StatsInterfaceMixin:
    """
    Tests for the interface on specific stats.
    """

    def test_mode_errors(self):
        _, f, params = self.get_example()
        for bad_mode in ["", "not a mode", "SITE", "x" * 8192]:
            with pytest.raises(ValueError):
                f(mode=bad_mode, **params)

        for bad_type in [123, {}, None, [[]]]:
            with pytest.raises(TypeError):
                f(mode=bad_type, **params)

    def test_window_errors(self):
        ts, f, params = self.get_example()
        del params["windows"]
        for bad_array in ["asdf", None, [[[[]], [[]]]], np.zeros((10, 3, 4))]:
            with pytest.raises(ValueError):
                f(windows=bad_array, **params)

        for bad_windows in [[], [0]]:
            with pytest.raises(ValueError):
                f(windows=bad_windows, **params)
        L = ts.get_sequence_length()
        bad_windows = [
            [L, 0],
            [0.1, L],
            [-1, L],
            [0, L + 0.1],
            [0, 0.1, 0.1, L],
            [0, -1, L],
            [0, 0.1, 0.05, 0.2, L],
        ]
        for bad_window in bad_windows:
            with pytest.raises(_tskit.LibraryError):
                f(windows=bad_window, **params)

    def test_windows_output(self):
        ts, f, params = self.get_example()
        del params["windows"]
        for num_windows in range(1, 10):
            windows = np.linspace(0, ts.get_sequence_length(), num=num_windows + 1)
            assert windows.shape[0] == num_windows + 1
            sigma = f(windows=windows, **params)
            assert sigma.shape[0] == num_windows


class WeightMixin(StatsInterfaceMixin):
    def get_example(self):
        ts, method = self.get_method()
        params = {
            "weights": np.ones((ts.get_num_samples(), 2)),
            "windows": [0, ts.get_sequence_length()],
        }
        return ts, method, params

    def test_bad_weights(self):
        ts, f, params = self.get_example()
        del params["weights"]
        n = ts.get_num_samples()

        with pytest.raises(_tskit.LibraryError):
            f(weights=np.ones((n, 0)), **params)

        for bad_weight_shape in [(n - 1, 1), (n + 1, 1), (0, 3)]:
            with pytest.raises(ValueError):
                f(weights=np.ones(bad_weight_shape), **params)

    def test_output_dims(self):
        ts, method, params = self.get_example()
        weights = params["weights"]
        nw = weights.shape[1]
        windows = [0, ts.get_sequence_length()]

        for mode in ["site", "branch"]:
            out = method(weights[:, [0]], windows, mode=mode)
            assert out.shape == (1, 1)
            out = method(weights, windows, mode=mode)
            assert out.shape == (1, nw)
            out = method(weights[:, [0, 0, 0]], windows, mode=mode)
            assert out.shape == (1, 3)
        mode = "node"
        N = ts.get_num_nodes()
        out = method(weights[:, [0]], windows, mode=mode)
        assert out.shape == (1, N, 1)
        out = method(weights, windows, mode=mode)
        assert out.shape == (1, N, nw)
        out = method(weights[:, [0, 0, 0]], windows, mode=mode)
        assert out.shape == (1, N, 3)


class WeightCovariateMixin(StatsInterfaceMixin):
    def get_example(self):
        ts, method = self.get_method()
        params = {
            "weights": np.ones((ts.get_num_samples(), 2)),
            "covariates": np.array(
                [np.arange(ts.get_num_samples()), np.arange(ts.get_num_samples()) ** 2]
            ).T,
            "windows": [0, ts.get_sequence_length()],
        }
        return ts, method, params

    def test_output_dims(self):
        ts, method, params = self.get_example()
        weights = params["weights"]
        nw = weights.shape[1]
        windows = [0, ts.get_sequence_length()]
        for covariates in (params["covariates"], params["covariates"][:, :0]):
            for mode in ["site", "branch"]:
                out = method(weights[:, [0]], covariates, windows, mode=mode)
                assert out.shape == (1, 1)
                out = method(weights, covariates, windows, mode=mode)
                assert out.shape == (1, nw)
                out = method(weights[:, [0, 0, 0]], covariates, windows, mode=mode)
                assert out.shape == (1, 3)
            mode = "node"
            N = ts.get_num_nodes()
            out = method(weights[:, [0]], covariates, windows, mode=mode)
            assert out.shape == (1, N, 1)
            out = method(weights, covariates, windows, mode=mode)
            assert out.shape == (1, N, nw)
            out = method(weights[:, [0, 0, 0]], covariates, windows, mode=mode)
            assert out.shape == (1, N, 3)


class SampleSetMixin(StatsInterfaceMixin):
    def test_bad_sample_sets(self):
        ts, f, params = self.get_example()
        del params["sample_set_sizes"]
        del params["sample_sets"]

        with pytest.raises(_tskit.LibraryError):
            f(sample_sets=[], sample_set_sizes=[], **params)

        n = ts.get_num_samples()
        samples = ts.get_samples()
        for bad_set_sizes in [[], [1], [n - 1], [n + 1], [n - 3, 1, 1], [1, n - 2]]:
            with pytest.raises(ValueError):
                f(sample_set_sizes=bad_set_sizes, sample_sets=samples, **params)

        N = ts.get_num_nodes()
        for bad_node in [-1, N, N + 1, -N]:
            with pytest.raises(_tskit.LibraryError):
                f(sample_set_sizes=[2], sample_sets=[0, bad_node], **params)

        for bad_sample in [n, n + 1, N - 1]:
            with pytest.raises(_tskit.LibraryError):
                f(sample_set_sizes=[2], sample_sets=[0, bad_sample], **params)


class OneWaySampleStatsMixin(SampleSetMixin):
    """
    Tests for one-way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [ts.get_num_samples()],
            "sample_sets": ts.get_samples(),
            "windows": [0, ts.get_sequence_length()],
        }
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        result = method(
            [ts.get_num_samples()], ts.get_samples(), [0, ts.get_sequence_length()]
        )
        assert result.shape == (1, 1)
        result = method(
            [ts.get_num_samples()],
            ts.get_samples(),
            [0, ts.get_sequence_length()],
            mode="node",
        )
        assert result.shape == (1, ts.get_num_nodes(), 1)
        result = method(
            [ts.get_num_samples()], ts.get_samples(), ts.get_breakpoints(), mode="node"
        )
        assert result.shape == (ts.get_num_trees(), ts.get_num_nodes(), 1)

    def test_output_dims(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)

        for mode in ["site", "branch"]:
            pi = method([n], samples, windows, mode=mode)
            assert pi.shape == (1, 1)
            pi = method([2, n - 2], samples, windows, mode=mode)
            assert pi.shape == (1, 2)
            pi = method([2, 2, n - 4], samples, windows, mode=mode)
            assert pi.shape == (1, 3)
            pi = method(np.ones(n).astype(np.uint32), samples, windows, mode=mode)
            assert pi.shape == (1, n)
        mode = "node"
        N = ts.get_num_nodes()
        pi = method([n], samples, windows, mode=mode)
        assert pi.shape == (1, N, 1)
        pi = method([2, n - 2], samples, windows, mode=mode)
        assert pi.shape == (1, N, 2)
        pi = method([2, 2, n - 4], samples, windows, mode=mode)
        assert pi.shape == (1, N, 3)
        pi = method(np.ones(n).astype(np.uint32), samples, windows, mode=mode)
        assert pi.shape == (1, N, n)

    def test_polarised(self):
        # TODO move this to the top level.
        ts, method = self.get_method()
        samples = ts.get_samples()
        n = len(samples)
        windows = [0, ts.get_sequence_length()]
        method([n], samples, windows, polarised=True)
        method([n], samples, windows, polarised=False)


class TestDiversity(LowLevelTestCase, OneWaySampleStatsMixin):
    """
    Tests for the diversity method.
    """

    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.diversity


class TestTraitCovariance(LowLevelTestCase, WeightMixin):
    """
    Tests for trait covariance.
    """

    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.trait_covariance


class TestTraitCorrelation(LowLevelTestCase, WeightMixin):
    """
    Tests for trait correlation.
    """

    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.trait_correlation


class TestTraitLinearModel(LowLevelTestCase, WeightCovariateMixin):
    """
    Tests for trait correlation.
    """

    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.trait_linear_model


class TestSegregatingSites(LowLevelTestCase, OneWaySampleStatsMixin):
    """
    Tests for the diversity method.
    """

    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.segregating_sites


class TestY1(LowLevelTestCase, OneWaySampleStatsMixin):
    """
    Tests for the diversity method.
    """

    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.Y1


class TestAlleleFrequencySpectrum(LowLevelTestCase, OneWaySampleStatsMixin):
    """
    Tests for the diversity method.
    """

    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.allele_frequency_spectrum

    def test_basic_example(self):
        ts = self.get_example_tree_sequence()
        n = ts.get_num_samples()
        result = ts.allele_frequency_spectrum(
            [n], ts.get_samples(), [0, ts.get_sequence_length()]
        )
        assert result.shape == (1, n + 1)
        result = ts.allele_frequency_spectrum(
            [n], ts.get_samples(), [0, ts.get_sequence_length()], polarised=True
        )
        assert result.shape == (1, n + 1)

    def test_output_dims(self):
        ts = self.get_example_tree_sequence()
        samples = ts.get_samples()
        L = ts.get_sequence_length()
        n = len(samples)

        for mode in ["site", "branch"]:
            for s in [[n], [n - 2, 2], [n - 4, 2, 2], [1] * n]:
                s = np.array(s, dtype=np.uint32)
                windows = [0, L]
                for windows in [[0, L], [0, L / 2, L], np.linspace(0, L, num=10)]:
                    jafs = ts.allele_frequency_spectrum(
                        s, samples, windows, mode=mode, polarised=True
                    )
                    assert jafs.shape == tuple([len(windows) - 1] + list(s + 1))
                    jafs = ts.allele_frequency_spectrum(
                        s, samples, windows, mode=mode, polarised=False
                    )
                    assert jafs.shape == tuple([len(windows) - 1] + list(s + 1))

    def test_node_mode_not_supported(self):
        ts = self.get_example_tree_sequence()
        with pytest.raises(_tskit.LibraryError):
            ts.allele_frequency_spectrum(
                [ts.get_num_samples()],
                ts.get_samples(),
                [0, ts.get_sequence_length()],
                mode="node",
            )


class TwoWaySampleStatsMixin(SampleSetMixin):
    """
    Tests for the two way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [2, ts.get_num_samples() - 2],
            "sample_sets": ts.get_samples(),
            "indexes": [[0, 1]],
            "windows": [0, ts.get_sequence_length()],
        }
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        div = method(
            [2, ts.get_num_samples() - 2],
            ts.get_samples(),
            [[0, 1]],
            windows=[0, ts.get_sequence_length()],
        )
        assert div.shape == (1, 1)

    def test_output_dims(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)
        for mode in ["site", "branch"]:
            div = method([2, 2, n - 4], samples, [[0, 1]], windows, mode=mode)
            assert div.shape == (1, 1)
            div = method([2, 2, n - 4], samples, [[0, 1], [1, 2]], windows, mode=mode)
            assert div.shape == (1, 2)
            div = method(
                [2, 2, n - 4], samples, [[0, 1], [1, 2], [0, 1]], windows, mode=mode
            )
            assert div.shape == (1, 3)

        N = ts.get_num_nodes()
        mode = "node"
        div = method([2, 2, n - 4], samples, [[0, 1]], windows, mode=mode)
        assert div.shape == (1, N, 1)
        div = method([2, 2, n - 4], samples, [[0, 1], [1, 2]], windows, mode=mode)
        assert div.shape == (1, N, 2)
        div = method(
            [2, 2, n - 4], samples, [[0, 1], [1, 2], [0, 1]], windows, mode=mode
        )
        assert div.shape == (1, N, 3)

    def test_set_index_errors(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)

        def f(indexes):
            method([2, 2, n - 4], samples, indexes, windows)

        for bad_array in ["wer", {}, [[[], []], [[], []]]]:
            with pytest.raises(ValueError):
                f(bad_array)
        for bad_dim in [[[]], [[1], [1]]]:
            with pytest.raises(ValueError):
                f(bad_dim)


class ThreeWaySampleStatsMixin(SampleSetMixin):
    """
    Tests for the two way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [1, 1, ts.get_num_samples() - 2],
            "sample_sets": ts.get_samples(),
            "indexes": [[0, 1, 2]],
            "windows": [0, ts.get_sequence_length()],
        }
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        div = method(
            [1, 1, ts.get_num_samples() - 2],
            ts.get_samples(),
            [[0, 1, 2]],
            windows=[0, ts.get_sequence_length()],
        )
        assert div.shape == (1, 1)

    def test_output_dims(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)
        for mode in ["site", "branch"]:
            div = method([2, 2, n - 4], samples, [[0, 1, 2]], windows, mode=mode)
            assert div.shape == (1, 1)
            div = method(
                [1, 1, 2, n - 4], samples, [[0, 1, 2], [1, 2, 3]], windows, mode=mode
            )
            assert div.shape == (1, 2)
            div = method(
                [1, 1, 2, n - 4],
                samples,
                [[0, 1, 2], [1, 2, 3], [0, 1, 2]],
                windows,
                mode=mode,
            )
            assert div.shape == (1, 3)

        N = ts.get_num_nodes()
        mode = "node"
        div = method([2, 2, n - 4], samples, [[0, 1, 2]], windows, mode=mode)
        assert div.shape == (1, N, 1)
        div = method(
            [1, 1, 2, n - 4], samples, [[0, 1, 2], [1, 2, 3]], windows, mode=mode
        )
        assert div.shape == (1, N, 2)
        div = method(
            [1, 1, 2, n - 4],
            samples,
            [[0, 1, 2], [1, 2, 3], [0, 1, 2]],
            windows,
            mode=mode,
        )
        assert div.shape == (1, N, 3)

    def test_set_index_errors(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)

        def f(indexes):
            method([2, 2, n - 4], samples, indexes, windows)

        for bad_array in ["wer", {}, [[[], []], [[], []]]]:
            with pytest.raises(ValueError):
                f(bad_array)
        for bad_dim in [[[]], [[1], [1]], [(0, 1)], [(0, 1, 2, 3)]]:
            with pytest.raises(ValueError):
                f(bad_dim)


class FourWaySampleStatsMixin(SampleSetMixin):
    """
    Tests for the four way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [1, 1, 1, ts.get_num_samples() - 3],
            "sample_sets": ts.get_samples(),
            "indexes": [[0, 1, 2, 3]],
            "windows": [0, ts.get_sequence_length()],
        }
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        div = method(
            [1, 1, 1, ts.get_num_samples() - 3],
            ts.get_samples(),
            [[0, 1, 2, 3]],
            windows=[0, ts.get_sequence_length()],
        )
        assert div.shape == (1, 1)

    def test_output_dims(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)
        for mode in ["site", "branch"]:
            div = method([2, 1, 1, n - 4], samples, [[0, 1, 2, 3]], windows, mode=mode)
            assert div.shape == (1, 1)
            div = method(
                [1, 1, 1, 1, n - 4],
                samples,
                [[0, 1, 2, 3], [1, 2, 3, 4]],
                windows,
                mode=mode,
            )
            assert div.shape == (1, 2)
            div = method(
                [1, 1, 1, 1, n - 4],
                samples,
                [[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 4]],
                windows,
                mode=mode,
            )
            assert div.shape == (1, 3)

        N = ts.get_num_nodes()
        mode = "node"
        div = method([2, 1, 1, n - 4], samples, [[0, 1, 2, 3]], windows, mode=mode)
        assert div.shape == (1, N, 1)
        div = method(
            [1, 1, 1, 1, n - 4],
            samples,
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            windows,
            mode=mode,
        )
        assert div.shape == (1, N, 2)
        div = method(
            [1, 1, 1, 1, n - 4],
            samples,
            [[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 4]],
            windows,
            mode=mode,
        )
        assert div.shape == (1, N, 3)

    def test_set_index_errors(self):
        ts, method = self.get_method()
        samples = ts.get_samples()
        windows = [0, ts.get_sequence_length()]
        n = len(samples)

        def f(indexes):
            method([2, 1, 1, n - 4], samples, indexes, windows)

        for bad_array in ["wer", {}, [[[], []], [[], []]]]:
            with pytest.raises(ValueError):
                f(bad_array)
        for bad_dim in [[[]], [[1], [1]], [(0, 1)], [(0, 1, 2, 3, 4)]]:
            with pytest.raises(ValueError):
                f(bad_dim)


class TestDivergence(LowLevelTestCase, TwoWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.divergence


class TestY2(LowLevelTestCase, TwoWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.Y2


class Testf2(LowLevelTestCase, TwoWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.f2


class TestY3(LowLevelTestCase, ThreeWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.Y3


class Testf3(LowLevelTestCase, ThreeWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.f3


class Testf4(LowLevelTestCase, FourWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.f4


class TestGeneralStatsInterface(LowLevelTestCase, StatsInterfaceMixin):
    """
    Tests for the general stats interface.
    """

    def get_example(self):
        ts = self.get_example_tree_sequence()
        W = np.zeros((ts.get_num_samples(), 1))
        params = {
            "weights": W,
            "summary_func": lambda x: np.cumsum(x),
            "output_dim": 1,
            "windows": ts.get_breakpoints(),
        }
        return ts, ts.general_stat, params

    def test_basic_example(self):
        ts = self.get_example_tree_sequence()
        W = np.zeros((ts.get_num_samples(), 1))
        sigma = ts.general_stat(
            W, lambda x: np.cumsum(x), 1, ts.get_breakpoints(), mode="branch"
        )
        assert sigma.shape == (ts.get_num_trees(), 1)

    def test_non_numpy_return(self):
        ts = self.get_example_tree_sequence()
        W = np.ones((ts.get_num_samples(), 3))
        sigma = ts.general_stat(
            W, lambda x: [sum(x)], 1, ts.get_breakpoints(), mode="branch"
        )
        assert sigma.shape == (ts.get_num_trees(), 1)
        sigma = ts.general_stat(
            W, lambda x: [2, 2], 2, ts.get_breakpoints(), mode="branch"
        )
        assert sigma.shape == (ts.get_num_trees(), 2)

    def test_complicated_numpy_function(self):
        ts = self.get_example_tree_sequence(sample_size=20, length=30, random_seed=325)
        W = np.zeros((ts.get_num_samples(), 4))

        def f(x):
            y = np.sum(x * x), np.prod(x + np.arange(x.shape[0]))
            return y

        sigma = ts.general_stat(W, f, 2, ts.get_breakpoints(), mode="branch")
        assert sigma.shape == (ts.get_num_trees(), 2)

    def test_input_dims(self):
        ts = self.get_example_tree_sequence()
        for k in range(1, 20):
            W = np.zeros((ts.get_num_samples(), k))
            sigma = ts.general_stat(
                W, lambda x: np.cumsum(x), k, ts.get_breakpoints(), mode="branch"
            )
            assert sigma.shape == (ts.get_num_trees(), k)
            sigma = ts.general_stat(
                W, lambda x: [np.sum(x)], 1, ts.get_breakpoints(), mode="branch"
            )
            assert sigma.shape == (ts.get_num_trees(), 1)

    def test_W_errors(self):
        ts = self.get_example_tree_sequence()
        n = ts.get_num_samples()
        for bad_array in [[], [0, 1], [[[[]], [[]]]], np.zeros((10, 3, 4))]:
            with pytest.raises(ValueError):
                ts.general_stat(bad_array, lambda x: x, 1, ts.get_breakpoints())

        for bad_size in [n - 1, n + 1, 0]:
            W = np.zeros((bad_size, 1))
            with pytest.raises(ValueError):
                ts.general_stat(W, lambda x: x, 1, ts.get_breakpoints())

    def test_summary_func_errors(self):
        ts = self.get_example_tree_sequence()
        W = np.zeros((ts.get_num_samples(), 1))
        for bad_type in ["sdf", 1, {}]:
            with pytest.raises(TypeError):
                ts.general_stat(W, bad_type, 1, ts.get_breakpoints())

        # Wrong numbers of arguments to f
        with pytest.raises(TypeError):
            ts.general_stat(W, lambda: 0, 1, ts.get_breakpoints())
        with pytest.raises(TypeError):
            ts.general_stat(W, lambda x, y: None, 1, ts.get_breakpoints())

        # Exceptions within f are correctly raised.
        for exception in [ValueError, TypeError]:

            def f(x):
                raise exception("test")

            with pytest.raises(exception):
                ts.general_stat(W, f, 1, ts.get_breakpoints())

        # Wrong output dimensions
        for bad_array in [[1, 1], range(10)]:
            with pytest.raises(ValueError):
                ts.general_stat(W, lambda x: bad_array, 1, ts.get_breakpoints())
        with pytest.raises(ValueError):
            ts.general_stat(W, lambda x: [1], 2, ts.get_breakpoints())

        # Bad arrays returned from f
        for bad_array in [["sdf"], 0, "w4", None]:
            with pytest.raises(ValueError):
                ts.general_stat(W, lambda x: bad_array, 1, ts.get_breakpoints())


class TestTreeDiffIterator(LowLevelTestCase):
    """
    Tests for the low-level tree diff iterator.
    """

    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        with pytest.raises(ValueError):
            _tskit.TreeDiffIterator(ts)

    def test_constructor(self):
        with pytest.raises(TypeError):
            _tskit.TreeDiffIterator()
        with pytest.raises(TypeError):
            _tskit.TreeDiffIterator(None)
        ts = self.get_example_tree_sequence()
        before = list(_tskit.TreeDiffIterator(ts))
        iterator = _tskit.TreeDiffIterator(ts)
        del ts
        # We should keep a reference to the tree sequence.
        after = list(iterator)
        assert before == after

    def test_iterator(self):
        ts = self.get_example_tree_sequence()
        self.verify_iterator(_tskit.TreeDiffIterator(ts))


class TestVariantGenerator(LowLevelTestCase):
    """
    Tests for the VariantGenerator class.
    """

    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        with pytest.raises(ValueError):
            _tskit.VariantGenerator(ts)

    def test_constructor(self):
        with pytest.raises(TypeError):
            _tskit.VariantGenerator()
        with pytest.raises(TypeError):
            _tskit.VariantGenerator(None)
        ts = self.get_example_tree_sequence()
        with pytest.raises(ValueError):
            _tskit.VariantGenerator(ts, samples={})
        with pytest.raises(TypeError):
            _tskit.VariantGenerator(ts, impute_missing_data=None)
        with pytest.raises(_tskit.LibraryError):
            _tskit.VariantGenerator(ts, samples=[-1, 2])
        with pytest.raises(TypeError):
            _tskit.VariantGenerator(ts, alleles=1234)

    def test_alleles(self):
        ts = self.get_example_tree_sequence()
        for bad_type in [["a", "b"], "sdf", 234]:
            with pytest.raises(TypeError):
                _tskit.VariantGenerator(ts, samples=[1, 2], alleles=bad_type)
        with pytest.raises(ValueError):
            _tskit.VariantGenerator(ts, samples=[1, 2], alleles=tuple())

        for bad_allele_type in [None, 0, b"x", []]:
            with pytest.raises(TypeError):
                _tskit.VariantGenerator(ts, samples=[1, 2], alleles=(bad_allele_type,))

        too_many_alleles = tuple(str(j) for j in range(128))
        with pytest.raises(_tskit.LibraryError):
            _tskit.VariantGenerator(ts, samples=[1, 2], alleles=too_many_alleles)

    def test_iterator(self):
        ts = self.get_example_tree_sequence()
        self.verify_iterator(_tskit.VariantGenerator(ts))

    def test_missing_data(self):
        tables = _tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.sites.add_row(0.1, "A")
        tables.build_index()
        ts = _tskit.TreeSequence(0)
        ts.load_tables(tables)
        variant = list(_tskit.VariantGenerator(ts))[0]
        _, genotypes, alleles = variant
        assert np.all(genotypes == -1)
        assert alleles == ("A", None)


class TestLdCalculator(LowLevelTestCase):
    """
    Tests for the LdCalculator class.
    """

    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        with pytest.raises(ValueError):
            _tskit.LdCalculator(ts)

    def test_constructor(self):
        with pytest.raises(TypeError):
            _tskit.LdCalculator()
        with pytest.raises(TypeError):
            _tskit.LdCalculator(None)

    def test_get_r2(self):
        ts = self.get_example_tree_sequence()
        calc = _tskit.LdCalculator(ts)
        n = ts.get_num_sites()
        for bad_id in [-1, n, n + 1]:
            with pytest.raises(_tskit.LibraryError):
                calc.get_r2(0, bad_id)
            with pytest.raises(_tskit.LibraryError):
                calc.get_r2(bad_id, 0)

    def test_get_r2_array(self):
        ts = self.get_example_tree_sequence()
        calc = _tskit.LdCalculator(ts)

        with pytest.raises(TypeError):
            calc.get_r2_array()
        with pytest.raises(TypeError):
            calc.get_r2_array(None)
        # Doesn't support buffer protocol, so raises typeerror
        with pytest.raises(TypeError):
            calc.get_r2_array(None, 0)

        n = ts.get_num_sites()
        assert n > 2
        with pytest.raises(BufferError):
            calc.get_r2_array(bytes(100), 0)

        buff = bytearray(1024)
        with pytest.raises(ValueError):
            calc.get_r2_array(buff, 0, max_distance=-1)
        with pytest.raises(ValueError):
            calc.get_r2_array(buff, 0, direction=1000)

        # TODO this API is poor, we should explicitly catch these negative
        # size errors.
        for bad_max_mutations in [-2, -3]:
            with pytest.raises(BufferError):
                calc.get_r2_array(buff, 0, max_mutations=bad_max_mutations)
        for bad_start_pos in [-1, n, n + 1]:
            with pytest.raises(_tskit.LibraryError):
                calc.get_r2_array(buff, bad_start_pos)


class TestLsHmm(LowLevelTestCase):
    """
    Tests for the LsHmm class.
    """

    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        with pytest.raises(ValueError):
            _tskit.LsHmm(ts, None, None)

    def test_constructor(self):
        ts = self.get_example_tree_sequence()
        with pytest.raises(TypeError):
            _tskit.LsHmm()
        with pytest.raises(TypeError):
            _tskit.LsHmm(None)
        values = np.zeros(ts.get_num_sites())
        for bad_array in ["asdf", [[], []], None]:
            with pytest.raises(ValueError):
                _tskit.LsHmm(ts, bad_array, values)
            with pytest.raises(ValueError):
                _tskit.LsHmm(ts, values, bad_array)

    def test_bad_rate_arrays(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        assert m > 0
        values = np.zeros(m)
        for bad_size in [0, m - 1, m + 1, m + 2]:
            bad_array = np.zeros(bad_size)
            with pytest.raises(ValueError):
                _tskit.LsHmm(ts, bad_array, values)
            with pytest.raises(ValueError):
                _tskit.LsHmm(ts, values, bad_array)

    def test_haplotype_input(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        fm = _tskit.CompressedMatrix(ts)
        vm = _tskit.ViterbiMatrix(ts)
        ls_hmm = _tskit.LsHmm(ts, np.zeros(m), np.zeros(m))
        for bad_size in [0, m - 1, m + 1, m + 2]:
            bad_array = np.zeros(bad_size, dtype=np.int8)
            with pytest.raises(ValueError):
                ls_hmm.forward_matrix(bad_array, fm)
            with pytest.raises(ValueError):
                ls_hmm.viterbi_matrix(bad_array, vm)
        for bad_array in [[0.002], [[], []], None]:
            with pytest.raises(ValueError):
                ls_hmm.forward_matrix(bad_array, fm)
            with pytest.raises(ValueError):
                ls_hmm.viterbi_matrix(bad_array, vm)

    def test_output_type_errors(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        h = np.zeros(m, dtype=np.int8)
        ls_hmm = _tskit.LsHmm(ts, np.zeros(m), np.zeros(m))
        for bad_type in [ls_hmm, None, m, []]:
            with pytest.raises(TypeError):
                ls_hmm.forward_matrix(h, bad_type)
            with pytest.raises(TypeError):
                ls_hmm.viterbi_matrix(h, bad_type)

        other_ts = self.get_example_tree_sequence()
        output = _tskit.CompressedMatrix(other_ts)
        with pytest.raises(_tskit.LibraryError):
            ls_hmm.forward_matrix(h, output)
        output = _tskit.ViterbiMatrix(other_ts)
        with pytest.raises(_tskit.LibraryError):
            ls_hmm.viterbi_matrix(h, output)

    def test_empty_forward_matrix(self):
        for mu in [0, 1]:
            ts = self.get_example_tree_sequence(mutation_rate=mu)
            m = ts.get_num_sites()
            fm = _tskit.CompressedMatrix(ts)
            assert fm.num_sites == m
            assert np.array_equal(np.zeros(m), fm.normalisation_factor)
            assert np.array_equal(np.zeros(m, dtype=np.uint32), fm.num_transitions)
            F = fm.decode()
            assert np.all(F >= 0)
            for j in range(m):
                assert fm.get_site(j) == []

    def test_empty_viterbi_matrix(self):
        for mu in [0, 1]:
            ts = self.get_example_tree_sequence(mutation_rate=mu)
            m = ts.get_num_sites()
            vm = _tskit.ViterbiMatrix(ts)
            assert vm.num_sites == m
            # TODO we should have the same semantics for 0 sites
            if m == 0:
                h = vm.traceback()
                assert len(h) == 0
            else:
                with pytest.raises(_tskit.LibraryError):
                    vm.traceback()

    def verify_compressed_matrix(self, ts, output):
        S = output.normalisation_factor
        N = output.num_transitions
        assert np.all(0 < S)
        assert np.all(S < 1)
        assert np.all(N > 0)
        F = output.decode()
        assert F.shape == (ts.get_num_sites(), ts.get_num_samples())
        assert np.all(F >= 0)
        m = ts.get_num_sites()
        for j in range(m):
            site_list = output.get_site(j)
            assert len(site_list) == N[j]
            for item in site_list:
                assert len(item) == 2
                node, value = item
                assert 0 <= node < ts.get_num_nodes()
                assert 0 <= value <= 1
        for site in [m, m + 1, 2 * m]:
            with pytest.raises(ValueError):
                output.get_site(site)

    def test_forward_matrix(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        output = _tskit.CompressedMatrix(ts)
        ls_hmm = _tskit.LsHmm(ts, np.zeros(m) + 0.1, np.zeros(m) + 0.1)
        rv = ls_hmm.forward_matrix([0 for _ in range(m)], output)
        assert rv is None
        self.verify_compressed_matrix(ts, output)

    def test_viterbi_matrix(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        output = _tskit.ViterbiMatrix(ts)
        ls_hmm = _tskit.LsHmm(ts, np.zeros(m) + 0.1, np.zeros(m) + 0.1)
        rv = ls_hmm.viterbi_matrix([0 for _ in range(m)], output)
        assert rv is None
        self.verify_compressed_matrix(ts, output)
        h = output.traceback()
        assert isinstance(h, np.ndarray)


class TestTree(LowLevelTestCase):
    """
    Tests on the low-level tree interface.
    """

    def test_options(self):
        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        assert st.get_options() == 0
        all_options = [
            0,
            _tskit.NO_SAMPLE_COUNTS,
            _tskit.SAMPLE_LISTS,
            _tskit.NO_SAMPLE_COUNTS | _tskit.SAMPLE_LISTS,
        ]
        for options in all_options:
            tree = _tskit.Tree(ts, options=options)
            copy = tree.copy()
            for st in [tree, copy]:
                assert st.get_options() == options
                assert st.get_num_samples(0) == 1
                if options & _tskit.NO_SAMPLE_COUNTS:
                    # We should still be able to count the samples, just inefficiently.
                    assert st.get_num_samples(0) == 1
                    with pytest.raises(_tskit.LibraryError):
                        st.get_num_tracked_samples(0)
                else:
                    assert st.get_num_tracked_samples(0) == 0
                if options & _tskit.SAMPLE_LISTS:
                    assert 0 == st.get_left_sample(0)
                    assert 0 == st.get_right_sample(0)
                else:
                    with pytest.raises(ValueError):
                        st.get_left_sample(0)
                    with pytest.raises(ValueError):
                        st.get_right_sample(0)
                    with pytest.raises(ValueError):
                        st.get_next_sample(0)

    def test_site_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_sites(), ts.get_num_sites() + 1]:
            with pytest.raises(IndexError):
                ts.get_site(bad_index)

    def test_mutation_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_mutations(), ts.get_num_mutations() + 1]:
            with pytest.raises(IndexError):
                ts.get_mutation(bad_index)

    def test_individual_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_individuals(), ts.get_num_individuals() + 1]:
            with pytest.raises(IndexError):
                ts.get_individual(bad_index)

    def test_population_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_populations(), ts.get_num_populations() + 1]:
            with pytest.raises(IndexError):
                ts.get_population(bad_index)

    def test_provenance_errors(self):
        ts = self.get_example_tree_sequence()
        for bad_index in [-1, ts.get_num_provenances(), ts.get_num_provenances() + 1]:
            with pytest.raises(IndexError):
                ts.get_provenance(bad_index)

    def test_sites(self):
        for ts in self.get_example_tree_sequences():
            st = _tskit.Tree(ts)
            all_sites = [ts.get_site(j) for j in range(ts.get_num_sites())]
            all_tree_sites = []
            j = 0
            mutation_id = 0
            while st.next():
                tree_sites = st.get_sites()
                assert st.get_num_sites() == len(tree_sites)
                all_tree_sites.extend(tree_sites)
                for (
                    position,
                    _ancestral_state,
                    mutations,
                    index,
                    metadata,
                ) in tree_sites:
                    assert st.get_left() <= position < st.get_right()
                    assert index == j
                    assert metadata == b""
                    for mut_id in mutations:
                        (
                            site,
                            node,
                            derived_state,
                            parent,
                            metadata,
                            time,
                        ) = ts.get_mutation(mut_id)
                        assert site == index
                        assert mutation_id == mut_id
                        assert st.get_parent(node) != _tskit.NULL
                        assert metadata == b""
                        mutation_id += 1
                    j += 1
            assert all_tree_sites == all_sites

    def test_root_threshold_errors(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        for bad_type in ["", "x", {}]:
            with pytest.raises(TypeError):
                tree.set_root_threshold(bad_type)

        with pytest.raises(_tskit.LibraryError):
            tree.set_root_threshold(0)
        tree.set_root_threshold(2)
        # Setting when not in the null state raises an error
        tree.next()
        with pytest.raises(_tskit.LibraryError):
            tree.set_root_threshold(2)

    def test_root_threshold(self):
        for ts in self.get_example_tree_sequences():
            tree = _tskit.Tree(ts)
            for root_threshold in [1, 2, ts.get_num_samples() * 2]:
                tree.set_root_threshold(root_threshold)
                assert tree.get_root_threshold() == root_threshold
                while tree.next():
                    assert tree.get_root_threshold() == root_threshold
                    with pytest.raises(_tskit.LibraryError):
                        tree.set_root_threshold(2)
                assert tree.get_root_threshold() == root_threshold

    def test_constructor(self):
        with pytest.raises(TypeError):
            _tskit.Tree()
        for bad_type in ["", {}, [], None, 0]:
            with pytest.raises(TypeError):
                _tskit.Tree(bad_type)
        ts = self.get_example_tree_sequence()
        for bad_type in ["", {}, True, 1, None]:
            with pytest.raises(TypeError):
                _tskit.Tree(ts, tracked_samples=bad_type)
        for bad_type in ["", {}, None, []]:
            with pytest.raises(TypeError):
                _tskit.Tree(ts, options=bad_type)
        for ts in self.get_example_tree_sequences():
            st = _tskit.Tree(ts)
            assert st.get_num_nodes() == ts.get_num_nodes()
            # An uninitialised tree should always be zero.
            assert st.get_left_root() == 0
            assert st.get_left() == 0
            assert st.get_right() == 0
            for j in range(ts.get_num_samples()):
                assert st.get_parent(j) == _tskit.NULL
                assert st.get_children(j) == tuple()
                assert st.get_time(j) == 0

    def test_bad_tracked_samples(self):
        ts = self.get_example_tree_sequence()
        options = 0
        for bad_type in ["", {}, [], None]:
            with pytest.raises(TypeError):
                _tskit.Tree(ts, options=options, tracked_samples=[bad_type])
            with pytest.raises(TypeError):
                _tskit.Tree(
                    ts,
                    options=options,
                    tracked_samples=[1, bad_type],
                )
        for bad_sample in [10 ** 6, -1e6]:
            with pytest.raises(ValueError):
                # Implicit conversion to integers using __int__ is deprecated
                with pytest.deprecated_call():
                    _tskit.Tree(
                        ts,
                        options=options,
                        tracked_samples=[bad_sample],
                    )
            with pytest.raises(ValueError):
                with pytest.deprecated_call():
                    _tskit.Tree(
                        ts,
                        options=options,
                        tracked_samples=[1, bad_sample],
                    )
            with pytest.raises(ValueError):
                with pytest.deprecated_call():
                    _tskit.Tree(ts, tracked_samples=[1, bad_sample, 1])

    def test_while_loop_semantics(self):
        for ts in self.get_example_tree_sequences():
            tree = _tskit.Tree(ts)
            # Any mixture of prev and next is OK and gives a valid iteration.
            for _ in range(2):
                j = 0
                while tree.next():
                    assert tree.get_index() == j
                    j += 1
                assert j == ts.get_num_trees()
            for _ in range(2):
                j = ts.get_num_trees()
                while tree.prev():
                    assert tree.get_index() == j - 1
                    j -= 1
                assert j == 0
            j = 0
            while tree.next():
                assert tree.get_index() == j
                j += 1
            assert j == ts.get_num_trees()

    def test_count_all_samples(self):
        for ts in self.get_example_tree_sequences():
            self.verify_iterator(_tskit.TreeDiffIterator(ts))
            st = _tskit.Tree(ts)
            # Without initialisation we should be 0 samples for every node
            # that is not a sample.
            for j in range(st.get_num_nodes()):
                count = 1 if j < ts.get_num_samples() else 0
                assert st.get_num_samples(j) == count
                assert st.get_num_tracked_samples(j) == 0
            while st.next():
                nu = get_sample_counts(ts, st)
                nu_prime = [st.get_num_samples(j) for j in range(st.get_num_nodes())]
                assert nu == nu_prime
                # For tracked samples, this should be all zeros.
                nu = [st.get_num_tracked_samples(j) for j in range(st.get_num_nodes())]
                assert nu == list([0 for _ in nu])

    def test_count_tracked_samples(self):
        # Ensure that there are some non-binary nodes.
        non_binary = False
        for ts in self.get_example_tree_sequences():
            st = _tskit.Tree(ts)
            while st.next():
                for u in range(ts.get_num_nodes()):
                    if len(st.get_children(u)) > 1:
                        non_binary = True
            samples = [j for j in range(ts.get_num_samples())]
            powerset = itertools.chain.from_iterable(
                itertools.combinations(samples, r) for r in range(len(samples) + 1)
            )
            max_sets = 100
            for _, subset in zip(range(max_sets), map(list, powerset)):
                # Ordering shouldn't make any difference.
                random.shuffle(subset)
                st = _tskit.Tree(ts, tracked_samples=subset)
                while st.next():
                    nu = get_tracked_sample_counts(st, subset)
                    nu_prime = [
                        st.get_num_tracked_samples(j) for j in range(st.get_num_nodes())
                    ]
                    assert nu == nu_prime
            # Passing duplicated values should raise an error
            sample = 1
            for j in range(2, 20):
                tracked_samples = [sample for _ in range(j)]
                with pytest.raises(_tskit.LibraryError):
                    _tskit.Tree(
                        ts,
                        tracked_samples=tracked_samples,
                    )
        assert non_binary

    def test_bounds_checking(self):
        for ts in self.get_example_tree_sequences():
            n = ts.get_num_nodes()
            st = _tskit.Tree(ts, options=_tskit.SAMPLE_LISTS)
            for v in [-100, -1, n + 1, n + 100, n * 100]:
                with pytest.raises(ValueError):
                    st.get_parent(v)
                with pytest.raises(ValueError):
                    st.get_children(v)
                with pytest.raises(ValueError):
                    st.get_time(v)
                with pytest.raises(ValueError):
                    st.get_left_sample(v)
                with pytest.raises(ValueError):
                    st.get_right_sample(v)
                with pytest.raises(ValueError):
                    st.is_descendant(v, 0)
                with pytest.raises(ValueError):
                    st.is_descendant(0, v)
                with pytest.raises(ValueError):
                    st.depth(v)
            n = ts.get_num_samples()
            for v in [-100, -1, n + 1, n + 100, n * 100]:
                with pytest.raises(ValueError):
                    st.get_next_sample(v)

    def test_mrca_interface(self):
        for ts in self.get_example_tree_sequences():
            num_nodes = ts.get_num_nodes()
            st = _tskit.Tree(ts)
            for v in [num_nodes, 10 ** 6, _tskit.NULL]:
                with pytest.raises(ValueError):
                    st.get_mrca(v, v)
                with pytest.raises(ValueError):
                    st.get_mrca(v, 1)
                with pytest.raises(ValueError):
                    st.get_mrca(1, v)
            # All the mrcas for an uninitialised tree should be _tskit.NULL
            for u, v in itertools.combinations(range(num_nodes), 2):
                assert st.get_mrca(u, v) == _tskit.NULL

    def test_newick_precision(self):
        def get_times(tree):
            """
            Returns the time strings from the specified newick tree.
            """
            ret = []
            current_time = None
            for c in tree:
                if c == ":":
                    current_time = ""
                elif c in [",", ")"]:
                    ret.append(current_time)
                    current_time = None
                elif current_time is not None:
                    current_time += c
            return ret

        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        while st.next():
            with pytest.raises(ValueError):
                st.get_newick(root=0, precision=-1)
            with pytest.raises(ValueError):
                st.get_newick(root=0, precision=17)
            with pytest.raises(ValueError):
                st.get_newick(root=0, precision=100)
            for precision in range(17):
                tree = st.get_newick(
                    root=st.get_left_root(), precision=precision
                ).decode()
                times = get_times(tree)
                assert len(times) > ts.get_num_samples()
                for t in times:
                    if precision == 0:
                        assert "." not in t
                    else:
                        point = t.find(".")
                        assert precision == len(t) - point - 1

    def test_cleared_tree(self):
        ts = self.get_example_tree_sequence()
        samples = ts.get_samples()

        def check_tree(tree):
            assert tree.get_index() == -1
            assert tree.get_left_root() == samples[0]
            assert tree.get_mrca(0, 1) == _tskit.NULL
            for u in range(ts.get_num_nodes()):
                assert tree.get_parent(u) == _tskit.NULL
                assert tree.get_left_child(u) == _tskit.NULL
                assert tree.get_right_child(u) == _tskit.NULL

        tree = _tskit.Tree(ts)
        check_tree(tree)
        while tree.next():
            pass
        check_tree(tree)
        while tree.prev():
            pass
        check_tree(tree)

    def test_newick_interface(self):
        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        # TODO this will break when we correctly handle multiple roots.
        assert st.get_newick(0) == b"1;"
        for bad_type in [None, "", [], {}]:
            with pytest.raises(TypeError):
                st.get_newick(precision=bad_type)
            with pytest.raises(TypeError):
                st.get_newick(ts, buffer_size=bad_type)
        while st.next():
            u = st.get_left_root()
            newick = st.get_newick(u)
            assert newick.endswith(b";")
            with pytest.raises(ValueError):
                st.get_newick(u, buffer_size=-1)
            with pytest.raises(_tskit.LibraryError):
                st.get_newick(u, buffer_size=1)

    def test_index(self):
        for ts in self.get_example_tree_sequences():
            st = _tskit.Tree(ts)
            index = 0
            while st.next():
                assert index == st.get_index()
                index += 1

    def test_bad_mutations(self):
        ts = self.get_example_tree_sequence()
        tables = _tskit.TableCollection()
        ts.dump_tables(tables)

        def f(mutations):
            position = []
            node = []
            site = []
            ancestral_state = []
            ancestral_state_offset = [0]
            derived_state = []
            derived_state_offset = [0]
            for j, (p, n) in enumerate(mutations):
                site.append(j)
                position.append(p)
                ancestral_state.append("0")
                ancestral_state_offset.append(ancestral_state_offset[-1] + 1)
                derived_state.append("1")
                derived_state_offset.append(derived_state_offset[-1] + 1)
                node.append(n)
            tables.sites.set_columns(
                dict(
                    position=position,
                    ancestral_state=ancestral_state,
                    ancestral_state_offset=ancestral_state_offset,
                    metadata=None,
                    metadata_offset=None,
                )
            )
            tables.mutations.set_columns(
                dict(
                    site=site,
                    node=node,
                    derived_state=derived_state,
                    derived_state_offset=derived_state_offset,
                    parent=None,
                    metadata=None,
                    metadata_offset=None,
                )
            )
            ts2 = _tskit.TreeSequence()
            ts2.load_tables(tables)

        with pytest.raises(_tskit.LibraryError):
            f([(0.1, -1)])
        length = ts.get_sequence_length()
        u = ts.get_num_nodes()
        for bad_node in [u, u + 1, 2 * u]:
            with pytest.raises(_tskit.LibraryError):
                f([(0.1, bad_node)])
        for bad_pos in [-1, length, length + 1]:
            with pytest.raises(_tskit.LibraryError):
                f([(bad_pos, 0)])

    def test_sample_list(self):
        options = _tskit.SAMPLE_LISTS
        # Note: we're assuming that samples are 0-n here.
        for ts in self.get_example_tree_sequences():
            t = _tskit.Tree(ts, options=options)
            while t.next():
                # All sample nodes should have themselves.
                for j in range(ts.get_num_samples()):
                    assert t.get_left_sample(j) == j
                    assert t.get_right_sample(j) == j

                # All non-tree nodes should have 0
                for j in range(t.get_num_nodes()):
                    if (
                        t.get_parent(j) == _tskit.NULL
                        and t.get_left_child(j) == _tskit.NULL
                    ):
                        assert t.get_left_sample(j) == _tskit.NULL
                        assert t.get_right_sample(j) == _tskit.NULL
                # The roots should have all samples.
                u = t.get_left_root()
                samples = []
                while u != _tskit.NULL:
                    sample = t.get_left_sample(u)
                    end = t.get_right_sample(u)
                    while True:
                        samples.append(sample)
                        if sample == end:
                            break
                        sample = t.get_next_sample(sample)
                    u = t.get_right_sib(u)
                assert sorted(samples) == list(range(ts.get_num_samples()))

    def test_equality(self):
        last_ts = None
        for ts in self.get_example_tree_sequences():
            t1 = _tskit.Tree(ts)
            t2 = _tskit.Tree(ts)
            assert t1.equals(t2)
            assert t2.equals(t1)
            while True:
                assert t1.equals(t2)
                assert t2.equals(t1)
                n1 = t1.next()
                assert not t1.equals(t2)
                assert not t2.equals(t1)
                n2 = t2.next()
                assert n1 == n2
                if not n1:
                    break
            if last_ts is not None:
                t2 = _tskit.Tree(last_ts)
                assert not t1.equals(t2)
                assert not t2.equals(t1)
            last_ts = ts

    def test_kc_distance_errors(self):
        ts1 = self.get_example_tree_sequence(10)
        t1 = _tskit.Tree(ts1, options=_tskit.SAMPLE_LISTS)
        t1.first()
        with pytest.raises(TypeError):
            t1.get_kc_distance()
        with pytest.raises(TypeError):
            t1.get_kc_distance(t1)
        for bad_tree in [None, "tree", 0]:
            with pytest.raises(TypeError):
                t1.get_kc_distance(bad_tree, lambda_=0)
        for bad_value in ["tree", [], None]:
            with pytest.raises(TypeError):
                t1.get_kc_distance(t1, lambda_=bad_value)

        t2 = _tskit.Tree(ts1, options=_tskit.SAMPLE_LISTS)
        # If we don't seek to a specific tree, it has multiple roots (i.e., it's
        # in the null state). This fails because we don't accept multiple roots.
        self.verify_kc_library_error(t2, t2)

        # Different numbers of samples fail.
        ts2 = self.get_example_tree_sequence(11)
        t2 = _tskit.Tree(ts2, options=_tskit.SAMPLE_LISTS)
        t2.first()
        self.verify_kc_library_error(t1, t2)

        # Error when tree not initialized with sample lists
        ts2 = self.get_example_tree_sequence(10)
        t2 = _tskit.Tree(ts2)
        t2.first()
        self.verify_kc_library_error(t1, t2)

        # Unary nodes cause errors.
        tables = _tskit.TableCollection(1.0)
        tables.nodes.add_row(flags=1)
        tables.nodes.add_row(flags=1, time=1)
        tables.edges.add_row(0, 1, 1, 0)
        tables.build_index()
        ts = _tskit.TreeSequence()
        ts.load_tables(tables)
        t1 = _tskit.Tree(ts, options=_tskit.SAMPLE_LISTS)
        t1.first()
        self.verify_kc_library_error(t1, t1)

    def verify_kc_library_error(self, t1, t2):
        with pytest.raises(_tskit.LibraryError):
            t1.get_kc_distance(t2, 0)

    def test_kc_distance(self):
        ts1 = self.get_example_tree_sequence(10, random_seed=123456)
        t1 = _tskit.Tree(ts1, options=_tskit.SAMPLE_LISTS)
        t1.first()
        ts2 = self.get_example_tree_sequence(10, random_seed=1234)
        t2 = _tskit.Tree(ts2, options=_tskit.SAMPLE_LISTS)
        t2.first()
        for lambda_ in [-1, 0, 1, 1000, -1e300]:
            x1 = t1.get_kc_distance(t2, lambda_)
            x2 = t2.get_kc_distance(t1, lambda_)
            assert x1 == x2

    def test_copy(self):
        for ts in self.get_example_tree_sequences():
            t1 = _tskit.Tree(ts)
            t2 = t1.copy()
            assert t1.get_index() == t2.get_index()
            assert t1 is not t2
            while t1.next():
                t2 = t1.copy()
                assert t1.get_index() == t2.get_index()

    def test_map_mutations_null(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        n = ts.get_num_samples()
        genotypes = np.zeros(n, dtype=np.int8)
        ancestral_state, transitions = tree.map_mutations(genotypes)
        assert ancestral_state == 0
        assert len(transitions) == 0

        genotypes = np.arange(n, dtype=np.int8)
        ancestral_state, transitions = tree.map_mutations(genotypes)
        assert ancestral_state == 0
        for j in range(n - 1):
            assert transitions[j][0] == j + 1
            assert transitions[j][1] == -1
            assert transitions[j][2] == j + 1

    def test_map_mutations(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        tree.next()
        n = ts.get_num_samples()
        genotypes = np.zeros(n, dtype=np.int8)
        ancestral_state, transitions = tree.map_mutations(genotypes)
        assert ancestral_state == 0
        assert len(transitions) == 0

    def test_map_mutations_errors(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        n = ts.get_num_samples()
        genotypes = np.zeros(n, dtype=np.int8)
        with pytest.raises(TypeError):
            tree.map_mutations()
        for bad_size in [0, 1, n - 1, n + 1]:
            with pytest.raises(ValueError):
                tree.map_mutations(np.zeros(bad_size, dtype=np.int8))
        for bad_type in [None, {}, set()]:
            with pytest.raises(TypeError):
                tree.map_mutations([bad_type] * n)
        for bad_type in [np.uint8, np.uint64, np.float32]:
            with pytest.raises(TypeError):
                tree.map_mutations(np.zeros(bad_size, dtype=bad_type))
        genotypes = np.zeros(n, dtype=np.int8)
        tree.map_mutations(genotypes)
        for bad_value in [64, 65, 127, -2]:
            genotypes[0] = bad_value
            with pytest.raises(_tskit.LibraryError):
                tree.map_mutations(genotypes)


class TestTableMetadataSchema(MetadataTestMixin):
    def test_metadata_schema_attribute(self):
        tables = _tskit.TableCollection(1.0)
        for table in self.metadata_tables:
            table = getattr(tables, f"{table}s")
            # Check default value
            assert table.metadata_schema == ""
            # Set and read back
            example = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋"
            table.metadata_schema = example
            assert table.metadata_schema == example
            # Can't del, or set to None
            with pytest.raises(AttributeError):
                del table.metadata_schema
            with pytest.raises(TypeError):
                table.metadata_schema = None
            # Del or None had no effect
            assert table.metadata_schema == example
            # Clear and read back
            table.metadata_schema = ""
            assert table.metadata_schema == ""


class TestMetadataSchemaNamedTuple(MetadataTestMixin):
    def test_named_tuple_init(self):
        # Test init errors
        with pytest.raises(TypeError):
            metadata_schemas = _tskit.MetadataSchemas()
        with pytest.raises(TypeError):
            metadata_schemas = _tskit.MetadataSchemas([])
        with pytest.raises(TypeError):
            metadata_schemas = _tskit.MetadataSchemas(["test_schema"])
        # Set and read back
        metadata_schemas = _tskit.MetadataSchemas(
            f"{table}_test_schema" for table in self.metadata_tables
        )
        assert metadata_schemas == tuple(
            f"{table}_test_schema" for table in self.metadata_tables
        )
        for i, table in enumerate(self.metadata_tables):
            # Read back via attr, index
            assert getattr(metadata_schemas, table) == f"{table}_test_schema"
            assert metadata_schemas[i] == f"{table}_test_schema"
            # Check read-only
            with pytest.raises(AttributeError):
                setattr(metadata_schemas, table, "")
            with pytest.raises(TypeError):
                metadata_schemas[i] = ""
        # Equality
        metadata_schemas2 = _tskit.MetadataSchemas(
            f"{table}_test_schema" for table in self.metadata_tables
        )
        assert metadata_schemas == metadata_schemas2
        metadata_schemas3 = _tskit.MetadataSchemas(
            f"{table}_test_schema_diff" for table in self.metadata_tables
        )
        assert metadata_schemas != metadata_schemas3


class TestModuleFunctions:
    """
    Tests for the module level functions.
    """

    def test_kastore_version(self):
        version = _tskit.get_kastore_version()
        assert version == (2, 0, 0)

    def test_tskit_version(self):
        version = _tskit.get_tskit_version()
        assert version == (0, 99, 9)


def test_uninitialised():
    # These methods work from an instance that has a NULL ref so don't check
    skip_list = [
        "TableCollection_load",
        "TreeSequence_load",
        "TreeSequence_load_tables",
    ]
    for cls_name, cls in inspect.getmembers(_tskit):
        if (
            type(cls) == type
            and not issubclass(cls, Exception)
            and not issubclass(cls, tuple)
        ):
            methods = []
            attributes = []
            for name, value in inspect.getmembers(cls):
                if not name.startswith("__") and f"{cls_name}_{name}" not in skip_list:
                    if inspect.isdatadescriptor(value):
                        attributes.append(name)
                    else:
                        methods.append(name)
            uninitialised = cls.__new__(cls)
            for attr in attributes:
                with pytest.raises((SystemError, ValueError)):
                    getattr(uninitialised, attr)
                with pytest.raises((SystemError, ValueError, AttributeError)):
                    setattr(uninitialised, attr, None)
            for method_name in methods:
                method = getattr(uninitialised, method_name)
                with pytest.raises((SystemError, ValueError)):
                    method()
