# MIT License
#
# Copyright (c) 2018-2025 Tskit Developers
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
import gc
import inspect
import itertools
import os
import random
import tempfile

import msprime
import numpy as np
import pytest

import _tskit
import tskit


NON_UTF8_STRING = "\ud861\udd37"


def get_tracked_sample_counts(ts, st, tracked_samples):
    """
    Returns a list giving the number of samples in the specified list
    that are in the subtree rooted at each node.
    """
    nu = [0 for j in range(ts.get_num_nodes())]
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
    nu = [0 for j in range(tree_sequence.get_num_nodes())]
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

    def test_skip_tables(self, tmp_path):
        tc = _tskit.TableCollection(1)
        self.get_example_tree_sequence().dump_tables(tc)
        with open(tmp_path / "tmp.trees", "wb") as f:
            tc.dump(f)

        for good_bool in [1, True]:
            with open(tmp_path / "tmp.trees", "rb") as f:
                tc_skip = _tskit.TableCollection()
                tc_skip.load(f, skip_tables=good_bool)
            assert not tc.equals(tc_skip)
            assert tc.equals(tc_skip, ignore_tables=True)

        for bad_bool in ["x", 0.5, {}]:
            with open(tmp_path / "tmp.trees", "rb") as f:
                tc_skip = _tskit.TableCollection()
                with pytest.raises(TypeError):
                    tc_skip.load(f, skip_tables=bad_bool)

    def test_skip_reference_sequence(self, tmp_path):
        tc = _tskit.TableCollection(1)
        self.get_example_tree_sequence().dump_tables(tc)
        tc.reference_sequence.data = "ACGT"
        with open(tmp_path / "tmp.trees", "wb") as f:
            tc.dump(f)

        for good_bool in [1, True]:
            with open(tmp_path / "tmp.trees", "rb") as f:
                tc_skip = _tskit.TableCollection()
                tc_skip.load(f, skip_reference_sequence=good_bool)
            assert not tc.equals(tc_skip)
            assert tc.equals(tc_skip, ignore_reference_sequence=True)

        for bad_bool in ["x", 0.5, {}]:
            with open(tmp_path / "tmp.trees", "rb") as f:
                tc_skip = _tskit.TableCollection()
                with pytest.raises(TypeError):
                    tc_skip.load(f, skip_reference_sequence=bad_bool)

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

    def test_file_format_eof_error(self, tmp_path):
        with open(tmp_path / "tmp.trees", "wb") as f:
            f.write(b"")
        with open(tmp_path / "tmp.trees", "rb") as f:
            tc2 = _tskit.TableCollection()
            with pytest.raises(EOFError):
                tc2.load(f)

    def test_file_format_kas_error(self, tmp_path):
        tc1 = _tskit.TableCollection(1)
        self.get_example_tree_sequence().dump_tables(tc1)
        with open(tmp_path / "tmp.trees", "wb") as f:
            tc1.dump(f)
        with open(tmp_path / "tmp.trees", "rb") as f:
            f.seek(1)
            tc2 = _tskit.TableCollection()
            with pytest.raises(_tskit.FileFormatError):
                tc2.load(f)

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
        tc = ts.dump_tables()._ll_tables
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
        for value in [-1, 1e6, 1e-22, 1000, 2**32, -10000]:
            tables.sequence_length = value
            assert tables.sequence_length == value

    def test_set_time_units_errors(self):
        tables = _tskit.TableCollection(1)
        with pytest.raises(AttributeError):
            del tables.time_units
        for bad_value in [b"no bytes", 59, 43.4, None, []]:
            with pytest.raises(TypeError):
                tables.time_units = bad_value

    def test_set_time_units(self):
        tables = _tskit.TableCollection(1)
        assert tables.time_units == tskit.TIME_UNITS_UNKNOWN
        for value in ["foo", "", "💩", "null char \0 in string"]:
            tables.time_units = value
            assert tables.time_units == value

    def test_set_metadata_errors(self):
        tables = _tskit.TableCollection(1)
        with pytest.raises(AttributeError):
            del tables.metadata
        for bad_value in ["no bytes", 59, 43.4, None, []]:
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
        tc = ts.dump_tables()._ll_tables
        with pytest.raises(TypeError):
            tc.simplify()
        with pytest.raises(ValueError):
            tc.simplify("asdf")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], keep_unary="sdf")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], keep_unary_in_individuals="abc")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], keep_input_roots="sdf")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], filter_populations="x")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], filter_nodes="x")
        with pytest.raises(TypeError):
            tc.simplify([0, 1], update_sample_flags="x")
        with pytest.raises(_tskit.LibraryError):
            tc.simplify([0, -1])

    @pytest.mark.parametrize("value", [True, False])
    @pytest.mark.parametrize(
        "flag",
        [
            "filter_sites",
            "filter_populations",
            "filter_individuals",
            "filter_nodes",
            "update_sample_flags",
            "reduce_to_site_topology",
            "keep_unary",
            "keep_unary_in_individuals",
            "keep_input_roots",
        ],
    )
    def test_simplify_flags(self, flag, value):
        tables = _tskit.TableCollection(1)
        tables.simplify([], **{flag: value})

    def test_link_ancestors_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
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
        tc = ts.dump_tables()._ll_tables
        edges = tc.link_ancestors([0, 1], [3])
        assert isinstance(edges, _tskit.EdgeTable)
        del edges
        assert tc.edges.num_rows == 2

    def test_subset_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
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
        tc = ts.dump_tables()._ll_tables
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

    def test_equals_bad_args(self):
        ts = msprime.simulate(10, random_seed=1242)
        tc = ts.dump_tables()._ll_tables
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
        with pytest.raises(TypeError):
            tc.equals(tc, ignore_tables=bad_bool)
        with pytest.raises(TypeError):
            tc.equals(tc, ignore_reference_sequence=bad_bool)

    def test_asdict(self):
        for ts in self.get_example_tree_sequences():
            tc = _tskit.TableCollection(sequence_length=ts.get_sequence_length())
            ts.dump_tables(tc)
            d = tc.asdict()
            # Method is tested extensively elsewhere, just basic sanity check here
            assert isinstance(d, dict)
            assert len(d) > 0

    def test_fromdict(self):
        for ts in self.get_example_tree_sequences():
            tc1 = _tskit.TableCollection(sequence_length=ts.get_sequence_length())
            ts.dump_tables(tc1)
            d = tc1.asdict()
            tc2 = _tskit.TableCollection(sequence_length=0)
            tc2.fromdict(d)
            assert tc1.equals(tc2)

    def test_asdict_bad_args(self):
        ts = msprime.simulate(10, random_seed=1242)
        tc = ts.dump_tables()._ll_tables
        for bad_type in [None, 0.1, "str"]:
            with pytest.raises(TypeError):
                tc.asdict(force_offset_64=bad_type)

    def test_fromdict_bad_args(self):
        tc = _tskit.TableCollection(0)
        for bad_type in [None, 0.1, "str"]:
            with pytest.raises(TypeError):
                tc.fromdict(bad_type)

    def test_sort_individuals(self):
        tc = _tskit.TableCollection(1)
        tc.sort_individuals()

    def test_delete_older_bad_args(self):
        tc = _tskit.TableCollection(1)
        self.get_example_tree_sequence().dump_tables(tc)
        with pytest.raises(TypeError):
            tc.delete_older()
        with pytest.raises(TypeError):
            tc.delete_older("1234")


class TestIbd:
    def test_uninitialised(self):
        result = _tskit.IdentitySegments.__new__(_tskit.IdentitySegments)
        with pytest.raises(SystemError):
            result.get(0, 1)
        with pytest.raises(SystemError):
            result.print_state()
        with pytest.raises(SystemError):
            result.num_segments
        with pytest.raises(SystemError):
            result.total_span
        with pytest.raises(SystemError):
            result.get_keys()

    def test_get_keys(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        pairs = [[0, 1], [0, 2], [1, 2]]
        result = tc.ibd_segments_within([0, 1, 2], store_pairs=True)
        np.testing.assert_array_equal(result.get_keys(), pairs)

    def test_store_pairs(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        # By default we can't get any information about pairs.
        result = tc.ibd_segments_within()
        with pytest.raises(_tskit.IdentityPairsNotStoredError):
            result.get_keys()
        with pytest.raises(_tskit.IdentityPairsNotStoredError):
            result.num_pairs
        with pytest.raises(_tskit.IdentityPairsNotStoredError):
            result.get(0, 1)

        num_pairs = 45
        result = tc.ibd_segments_within(store_pairs=True)
        assert len(result.get_keys()) == num_pairs
        assert result.num_pairs == num_pairs

        seglist = result.get(0, 1)
        assert seglist.num_segments == 1
        assert seglist.total_span == 1
        with pytest.raises(_tskit.IdentitySegmentsNotStoredError):
            seglist.node
        with pytest.raises(_tskit.IdentitySegmentsNotStoredError):
            seglist.left
        with pytest.raises(_tskit.IdentitySegmentsNotStoredError):
            seglist.right

    def test_within_all_pairs(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        num_pairs = ts.num_samples * (ts.num_samples - 1) / 2
        result = tc.ibd_segments_within(store_pairs=True)
        assert result.num_pairs == num_pairs
        pairs = np.array(list(itertools.combinations(range(ts.num_samples), 2)))
        np.testing.assert_array_equal(result.get_keys(), pairs)

    def test_between_all_pairs(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        result = tc.ibd_segments_between([5, 5], range(10), store_pairs=True)
        assert result.num_pairs == 25
        pairs = np.array(list(itertools.product(range(5), range(5, 10))))
        np.testing.assert_array_equal(result.get_keys(), pairs)

    def test_within_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        for bad_samples in ["sdf", {}]:
            with pytest.raises(ValueError):
                tc.ibd_segments_within(bad_samples)
        # input array must be 1D
        with pytest.raises(ValueError):
            tc.ibd_segments_within([[[1], [1]]])
        for bad_float in ["sdf", None, {}]:
            with pytest.raises(TypeError):
                tc.ibd_segments_within(min_span=bad_float)
            with pytest.raises(TypeError):
                tc.ibd_segments_within(max_time=bad_float)
        with pytest.raises(_tskit.LibraryError):
            tc.ibd_segments_within(max_time=-1)
        with pytest.raises(_tskit.LibraryError):
            tc.ibd_segments_within(min_span=-1)

    def test_between_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        with pytest.raises(TypeError):
            tc.ibd_segments_between()
        with pytest.raises(TypeError):
            tc.ibd_segments_between([1])

        with pytest.raises(ValueError):
            tc.ibd_segments_between("sdf", [1, 2])
        with pytest.raises(ValueError):
            tc.ibd_segments_between([1, 2], "sdf")
        # The sample_set parsing code is tested elsewhere, so just test
        # something basic.
        with pytest.raises(ValueError, match="Sum of sample_set_sizes"):
            tc.ibd_segments_between([1, 1], [1])
        for bad_float in ["sdf", None, {}]:
            with pytest.raises(TypeError):
                tc.ibd_segments_between([1, 1], [0, 1], min_span=bad_float)
            with pytest.raises(TypeError):
                tc.ibd_segments_between([1, 1], [0, 1], max_time=bad_float)
        with pytest.raises(_tskit.LibraryError):
            tc.ibd_segments_between([1, 1], [0, 1], min_span=-1)
        with pytest.raises(_tskit.LibraryError):
            tc.ibd_segments_between([1, 1], [0, 1], max_time=-1)
        with pytest.raises(_tskit.LibraryError, match="Duplicate sample"):
            tc.ibd_segments_between([1, 1], [0, 0])

    def test_get_output(self):
        ts = msprime.simulate(5, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        pairs = [(0, 1), (2, 3)]
        result = tc.ibd_segments_within([0, 1, 2, 3], store_segments=True)
        assert isinstance(result, _tskit.IdentitySegments)
        for pair in pairs:
            value = result.get(*pair)
            assert isinstance(value, _tskit.IdentitySegmentList)
            assert value.num_segments == 1
            assert isinstance(value.left, np.ndarray)
            assert isinstance(value.right, np.ndarray)
            assert isinstance(value.node, np.ndarray)
            assert list(value.left) == [0]
            assert list(value.right) == [1]
            assert len(value.node) == 1
            assert value.num_segments == 1
            assert value.total_span == 1

    def test_get_bad_args(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        result = tc.ibd_segments_within([0, 1, 2], store_segments=True)
        with pytest.raises(TypeError):
            result.get()
        with pytest.raises(TypeError):
            result.get("0", 1)
        with pytest.raises(_tskit.LibraryError, match="Both nodes"):
            result.get(0, 0)
        with pytest.raises(_tskit.LibraryError, match="Node out of bounds"):
            result.get(-1, 0)
        with pytest.raises(_tskit.LibraryError, match="Node out of bounds"):
            result.get(0, 100)
        with pytest.raises(KeyError):
            result.get(0, 3)

    def test_print_state(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        result = tc.ibd_segments_within()
        with pytest.raises(TypeError):
            result.print_state()

        with tempfile.TemporaryFile("w+") as f:
            result.print_state(f)
            f.seek(0)
            output = f.read()
        assert len(output) > 0
        assert "IBD" in output

    def test_direct_instantiation(self):
        # Nobody should do this, but just in case
        result = _tskit.IdentitySegments()
        assert result.num_segments == 0
        assert result.total_span == 0
        with tempfile.TemporaryFile("w+") as f:
            result.print_state(f)
            f.seek(0)
            output = f.read()
        assert len(output) > 0
        assert "IBD" in output


class TestIdentitySegmentList:
    def test_direct_instantiation(self):
        # Nobody should do this, but just in case
        seglist = _tskit.IdentitySegmentList()
        attrs = ["num_segments", "total_span", "left", "right", "node"]
        for attr in attrs:
            with pytest.raises(SystemError, match="not initialised"):
                getattr(seglist, attr)

    def test_memory_management_within(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        result = tc.ibd_segments_within(store_segments=True)
        del ts, tc
        lst = result.get(0, 1)
        assert lst.num_segments == 1
        del result
        gc.collect()
        assert lst.num_segments == 1
        # Do some allocs to see if we're still working properly
        x = sum(list(range(1000)))
        assert x > 0
        assert lst.num_segments == 1

    def test_memory_management_between(self):
        ts = msprime.simulate(10, random_seed=1)
        tc = ts.dump_tables()._ll_tables
        result = tc.ibd_segments_between([2, 2], range(4), store_segments=True)
        del ts, tc
        lst = result.get(0, 2)
        assert lst.num_segments == 1
        del result
        gc.collect()
        assert lst.num_segments == 1
        # Do some allocs to see if we're still working properly
        x = sum(list(range(1000)))
        assert x > 0
        assert lst.num_segments == 1


class TestTableMethods:
    """
    Tests for the low-level table methods.
    """

    @pytest.mark.parametrize("table_name", tskit.TABLE_NAMES)
    def test_table_extend(self, table_name, ts_fixture):
        table = getattr(ts_fixture.dump_tables(), table_name)
        assert len(table) >= 5
        ll_table = table.ll_table
        table_copy = table.copy()

        ll_table.extend(table_copy.ll_table, row_indexes=[])
        assert table == table_copy

        ll_table.clear()
        ll_table.extend(table_copy.ll_table, row_indexes=range(len(table_copy)))
        assert table == table_copy

    @pytest.mark.parametrize("table_name", tskit.TABLE_NAMES)
    @pytest.mark.parametrize(
        ["row_indexes", "expected_rows"],
        [
            ([0], [0]),
            ([4] * 1000, [4] * 1000),
            ([4, 1, 3, 0, 0], [4, 1, 3, 0, 0]),
            (np.array([0, 1, 4], dtype=np.uint8), [0, 1, 4]),
            (np.array([3, 3, 3], dtype=np.uint16), [3, 3, 3]),
            (np.array([4, 2, 1], dtype=np.int8), [4, 2, 1]),
            (np.array([4, 2], dtype=np.int16), [4, 2]),
            (np.array([0, 1], dtype=np.int32), [0, 1]),
            (range(2, -1, -1), [2, 1, 0]),
        ],
    )
    def test_table_extend_types(
        self, ts_fixture, table_name, row_indexes, expected_rows
    ):
        table = getattr(ts_fixture.dump_tables(), table_name)
        assert len(table) >= 5
        ll_table = table.ll_table
        table_copy = table.copy()

        ll_table.extend(table_copy.ll_table, row_indexes=row_indexes)
        assert len(table) == len(table_copy) + len(expected_rows)
        for i, expected_row in enumerate(expected_rows):
            assert table[len(table_copy) + i] == table_copy[expected_row]

    @pytest.mark.parametrize("table_name", tskit.TABLE_NAMES)
    def test_table_keep_rows_errors(self, table_name, ts_fixture):
        table = getattr(ts_fixture.dump_tables(), table_name)
        n = len(table)
        ll_table = table.ll_table
        with pytest.raises(ValueError, match="must be of length"):
            ll_table.keep_rows(np.ones(n - 1, dtype=bool))
        with pytest.raises(ValueError, match="must be of length"):
            ll_table.keep_rows(np.ones(n + 1, dtype=bool))
        with pytest.raises(TypeError, match="Cannot cast"):
            ll_table.keep_rows(np.ones(n, dtype=int))

    @pytest.mark.parametrize("table_name", tskit.TABLE_NAMES)
    def test_table_keep_rows_all(self, table_name, ts_fixture):
        table = getattr(ts_fixture.dump_tables(), table_name)
        n = len(table)
        ll_table = table.ll_table
        a = ll_table.keep_rows(np.ones(n, dtype=bool))
        assert ll_table.num_rows == n
        assert a.shape == (n,)
        assert a.dtype == np.int32
        assert np.all(a == np.arange(n))

    @pytest.mark.parametrize("table_name", tskit.TABLE_NAMES)
    def test_table_keep_rows_none(self, table_name, ts_fixture):
        table = getattr(ts_fixture.dump_tables(), table_name)
        n = len(table)
        ll_table = table.ll_table
        a = ll_table.keep_rows(np.zeros(n, dtype=bool))
        assert ll_table.num_rows == 0
        assert a.shape == (n,)
        assert a.dtype == np.int32
        assert np.all(a == -1)

    def test_mutation_table_keep_rows_ref_error(self):
        table = _tskit.MutationTable()
        table.add_row(site=0, node=0, derived_state="A", parent=2)
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_MUTATION_OUT_OF_BOUNDS"):
            table.keep_rows([True])

    def test_individual_table_keep_rows_ref_error(self):
        table = _tskit.IndividualTable()
        table.add_row(parents=[2])
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INDIVIDUAL_OUT_OF_BOUNDS"
        ):
            table.keep_rows([True])

    @pytest.mark.parametrize(
        ["table_name", "column_name"],
        [
            (t, c)
            for t in tskit.TABLE_NAMES
            for c in getattr(tskit, f"{t[:-1].capitalize()}Table").column_names
            if c[-7:] != "_offset"
        ],
    )
    def test_table_update(self, ts_fixture, table_name, column_name):
        table = getattr(ts_fixture.dump_tables(), table_name)
        copy = table.copy()
        ll_table = table.ll_table

        # Find the first row where this column differs to get a value to swap in
        other_row_index = -1
        for i, row in enumerate(table):
            if not np.array_equal(
                getattr(table[0], column_name), getattr(row, column_name)
            ):
                other_row_index = i
        assert other_row_index != -1

        # No-op update should not create a change
        args = ll_table.get_row(0)
        ll_table.update_row(0, *args)
        table.assert_equals(copy)

        # Modify the column under test in the first row
        new_args = list(ll_table.get_row(0))
        arg_index = list(inspect.signature(table.add_row).parameters.keys()).index(
            column_name
        )
        new_args[arg_index] = ll_table.get_row(other_row_index)[arg_index]
        ll_table.update_row(0, *new_args)
        for a, b in zip(ll_table.get_row(0), new_args):
            np.array_equal(a, b)

    def test_update_defaults(self):
        t = tskit.IndividualTable()
        assert t.add_row(flags=1, location=[1, 2], parents=[3, 4], metadata=b"FOO") == 0
        t.ll_table.update_row(0)
        assert t.flags[0] == 0
        assert len(t.location) == 0
        assert t.location_offset[0] == 0
        assert len(t.parents) == 0
        assert t.parents_offset[0] == 0
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

        t = tskit.NodeTable()
        assert (
            t.add_row(flags=1, time=2, population=3, individual=4, metadata=b"FOO") == 0
        )
        t.ll_table.update_row(0)
        assert t.time[0] == 0
        assert t.flags[0] == 0
        assert t.population[0] == tskit.NULL
        assert t.individual[0] == tskit.NULL
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

        t = tskit.EdgeTable()
        assert t.add_row(1, 2, 3, 4, metadata=b"FOO") == 0
        t.ll_table.update_row(0, 1, 2, 3, 4)
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

        t = tskit.MigrationTable()
        assert t.add_row(1, 2, 3, 4, 5, 6, b"FOO") == 0
        t.ll_table.update_row(0, 1, 2, 3, 4, 5, 6)
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

        t = tskit.MutationTable()
        assert t.add_row(1, 2, "A", 3, b"FOO", 4) == 0
        t.ll_table.update_row(0, 1, 2, "A", 3)
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0
        assert tskit.is_unknown_time(t.time[0])

        t = tskit.PopulationTable()
        assert t.add_row(b"FOO") == 0
        t.ll_table.update_row(0)
        assert len(t.metadata) == 0
        assert t.metadata_offset[0] == 0

    def test_update_bad_data(self):
        t = tskit.IndividualTable()
        t.add_row()
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, flags="x")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, metadata=123)
        with pytest.raises(ValueError):
            t.ll_table.update_row(0, location="1234")
        with pytest.raises(ValueError):
            t.ll_table.update_row(0, parents="forty-two")

        t = tskit.NodeTable()
        t.add_row()
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, flags="x")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, time="x")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, individual="x")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, population="x")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, metadata=123)

        t = tskit.EdgeTable()
        t.add_row(1, 2, 3, 4)
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, left="x", right=0, parent=0, child=0)
        with pytest.raises(TypeError):
            t.ll_table.update_row(
                0,
            )
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, 0, 0, 0, metadata=123)

        t = tskit.SiteTable()
        t.add_row(0, "A")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, "x", "A")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, 0)
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, "A", metadata=[0, 1, 2])

        t = tskit.MutationTable()
        t.add_row(0, 0, "A")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, "0", 0, "A")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, "0", "A")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, 0, "A", parent=None)
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, 0, "A", metadata=[0])
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, 0, "A", time="A")

        t = tskit.MigrationTable()
        with pytest.raises(TypeError):
            t.add_row(left="x", right=0, node=0, source=0, dest=0, time=0)
        with pytest.raises(TypeError):
            t.ll_table.update_row(
                0,
            )
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, 0, 0, 0, 0, 0, metadata=123)

        t = tskit.ProvenanceTable()
        t.add_row("a", "b")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, 0, "b")
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, "a", 0)

        t = tskit.PopulationTable()
        t.add_row()
        with pytest.raises(TypeError):
            t.ll_table.update_row(0, metadata=[0])


class TestTableMethodsErrors:
    """
    Tests for the error handling of errors in the low-level tables.
    """

    def yield_tables(self, ts):
        for table in ts.dump_tables().table_name_map.values():
            yield table.ll_table

    @pytest.mark.parametrize(
        "table_name",
        tskit.TABLE_NAMES,
    )
    def test_table_extend_bad_args(self, ts_fixture, table_name):
        table = getattr(ts_fixture.dump_tables(), table_name)
        ll_table = table.ll_table
        ll_table_copy = table.copy().ll_table

        with pytest.raises(
            _tskit.LibraryError,
            match="Tables can only be extended using rows from a different table",
        ):
            ll_table.extend(ll_table, row_indexes=[])
        with pytest.raises(TypeError):
            ll_table.extend(None, row_indexes=[])
        with pytest.raises(ValueError):
            ll_table.extend(ll_table_copy, row_indexes=5)
        with pytest.raises(TypeError):
            ll_table.extend(ll_table_copy, row_indexes=[None])
        with pytest.raises(ValueError, match="object too deep"):
            ll_table.extend(ll_table_copy, row_indexes=[[0, 1], [2, 3]])
        with pytest.raises(ValueError, match="object too deep"):
            ll_table.extend(ll_table_copy, row_indexes=[[0, 1]])
        with pytest.raises(_tskit.LibraryError, match="out of bounds"):
            ll_table.extend(ll_table_copy, row_indexes=[-1])
        with pytest.raises(_tskit.LibraryError, match="out of bounds"):
            ll_table.extend(ll_table_copy, row_indexes=[1000])
        with pytest.raises(_tskit.LibraryError, match="out of bounds"):
            ll_table.extend(ll_table_copy, row_indexes=range(10000000, 10000001))

        # Uncastable types
        for dtype in [np.uint32, np.int64, np.uint64, np.float32, np.float64]:
            with pytest.raises(TypeError, match="Cannot cast"):
                ll_table.extend(ll_table_copy, row_indexes=np.array([0], dtype=dtype))

    @pytest.mark.parametrize("table_name", tskit.TABLE_NAMES)
    def test_update_bad_row_index(self, ts_fixture, table_name):
        table = getattr(ts_fixture.dump_tables(), table_name)
        ll_table = table.ll_table
        row_data = ll_table.get_row(0)
        with pytest.raises(_tskit.LibraryError, match="out of bounds"):
            ll_table.update_row(-1, *row_data)
        with pytest.raises(ValueError, match="tskit ids must be"):
            ll_table.update_row(-42, *row_data)
        with pytest.raises(TypeError):
            ll_table.update_row([], *row_data)
        with pytest.raises(TypeError):
            ll_table.update_row("abc", *row_data)
        with pytest.raises(_tskit.LibraryError, match="out of bounds"):
            ll_table.update_row(10000, *row_data)
        with pytest.raises(OverflowError, match="Value too large for tskit id type"):
            ll_table.update_row(2**62, *row_data)

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
        tc = msprime.simulate(10, random_seed=42).dump_tables()._ll_tables
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
        tc = msprime.simulate(10, random_seed=42).dump_tables()._ll_tables
        tc.drop_index()
        assert tc.indexes == {}

    def test_bad_indexes(self):
        tc = msprime.simulate(10, random_seed=42).dump_tables()._ll_tables
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

        tc = (
            msprime.simulate(10, recombination_rate=10, random_seed=42)
            .dump_tables()
            ._ll_tables
        )
        modify_indexes = tc.indexes
        shape = modify_indexes["edge_insertion_order"].shape
        modify_indexes["edge_insertion_order"] = np.zeros(shape, dtype=np.int32)
        modify_indexes["edge_removal_order"] = np.zeros(shape, dtype=np.int32)
        tc.indexes = modify_indexes
        ts = _tskit.TreeSequence()
        with pytest.raises(
            _tskit.LibraryError,
            match="TSK_ERR_TABLES_BAD_INDEXES",
        ):
            ts.load_tables(tc, build_indexes=False)

        modify_indexes["edge_insertion_order"] = np.full(shape, 2**30, dtype=np.int32)
        modify_indexes["edge_removal_order"] = np.full(shape, 2**30, dtype=np.int32)
        tc.indexes = modify_indexes
        ts = _tskit.TreeSequence()
        with pytest.raises(_tskit.LibraryError, match="Edge out of bounds"):
            ts.load_tables(tc, build_indexes=False)


class TestTreeSequence(LowLevelTestCase, MetadataTestMixin):
    """
    Tests for the low-level interface for the TreeSequence.
    """

    ARRAY_NAMES = [
        "individuals_flags",
        "individuals_location",
        "individuals_location_offset",
        "individuals_parents",
        "individuals_parents_offset",
        "individuals_metadata",
        "individuals_metadata_offset",
        "nodes_time",
        "nodes_flags",
        "nodes_population",
        "nodes_individual",
        "nodes_metadata",
        "nodes_metadata_offset",
        "edges_left",
        "edges_right",
        "edges_parent",
        "edges_child",
        "edges_metadata",
        "edges_metadata_offset",
        "sites_position",
        "sites_metadata",
        "sites_metadata_offset",
        "mutations_site",
        "mutations_node",
        "mutations_parent",
        "mutations_time",
        "mutations_metadata",
        "mutations_metadata_offset",
        "sites_ancestral_state",
        "sites_ancestral_state_offset",
        "mutations_derived_state",
        "mutations_derived_state_offset",
        "provenances_record",
        "provenances_record_offset",
        "provenances_timestamp",
        "provenances_timestamp_offset",
        "migrations_left",
        "migrations_right",
        "migrations_node",
        "migrations_source",
        "migrations_dest",
        "migrations_time",
        "migrations_metadata",
        "migrations_metadata_offset",
        "populations_metadata",
        "populations_metadata_offset",
        "indexes_edge_insertion_order",
        "indexes_edge_removal_order",
    ]

    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(prefix="msp_ll_ts_")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)

    def test_skip_tables(self, tmp_path):
        ts = self.get_example_tree_sequence()
        with open(tmp_path / "tmp.trees", "wb") as f:
            ts.dump(f)
        tc = _tskit.TableCollection(1)
        ts.dump_tables(tc)

        for good_bool in [1, True]:
            with open(tmp_path / "tmp.trees", "rb") as f:
                ts_skip = _tskit.TreeSequence()
                ts_skip.load(f, skip_tables=good_bool)
            tc_skip = _tskit.TableCollection()
            ts_skip.dump_tables(tc_skip)
            assert not tc.equals(tc_skip)
            assert tc.equals(tc_skip, ignore_tables=True)

        for bad_bool in ["x", 0.5, {}]:
            with open(tmp_path / "tmp.trees", "rb") as f:
                ts_skip = _tskit.TreeSequence()
                with pytest.raises(TypeError):
                    ts_skip.load(f, skip_tables=bad_bool)

    def test_skip_reference_sequence(self, tmp_path):
        tc = _tskit.TableCollection(1)
        self.get_example_tree_sequence().dump_tables(tc)
        tc.reference_sequence.data = "ACGT"
        ts = _tskit.TreeSequence()
        ts.load_tables(tc, build_indexes=True)
        with open(tmp_path / "tmp.trees", "wb") as f:
            ts.dump(f)

        for good_bool in [1, True]:
            with open(tmp_path / "tmp.trees", "rb") as f:
                ts_skip = _tskit.TreeSequence()
                ts_skip.load(f, skip_reference_sequence=good_bool)
            tc_skip = _tskit.TableCollection()
            ts_skip.dump_tables(tc_skip)
            assert not tc.equals(tc_skip)
            assert tc.equals(tc_skip, ignore_reference_sequence=True)

        for bad_bool in ["x", 0.5, {}]:
            with open(tmp_path / "tmp.trees", "rb") as f:
                ts_skip = _tskit.TreeSequence()
                with pytest.raises(TypeError):
                    ts_skip.load(f, skip_reference_sequence=bad_bool)

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

    def test_get_mutation_interface(self):
        for ts in self.get_example_tree_sequences():
            mutations = [ts.get_mutation(j) for j in range(ts.get_num_mutations())]
            assert len(mutations) == ts.get_num_mutations()
            # Check the form of the mutations
            for packed in mutations:
                (
                    site,
                    node,
                    derived_state,
                    parent,
                    metadata,
                    time,
                    edge,
                    inherited_state,
                ) = packed
                assert isinstance(site, int)
                assert isinstance(node, int)
                assert isinstance(derived_state, str)
                assert isinstance(parent, int)
                assert isinstance(metadata, bytes)
                assert isinstance(time, float)
                assert isinstance(edge, int)
                assert isinstance(inherited_state, str)

    def test_get_edge_interface(self):
        for ts in self.get_example_tree_sequences():
            num_edges = ts.get_num_edges()
            # We don't accept Python negative indexes here.
            with pytest.raises(IndexError):
                ts.get_edge(-1)
            for j in [0, 10, 10**6]:
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
            for j in [0, 10, 10**6]:
                with pytest.raises(IndexError):
                    ts.get_node(num_nodes + j)
            for x in [None, "", {}, []]:
                with pytest.raises(TypeError):
                    ts.get_node(x)

    def test_get_migration_interface(self):
        ts = self.get_example_migration_tree_sequence()
        for bad_type in ["", None, {}]:
            with pytest.raises(TypeError):
                ts.get_migration(bad_type)
        num_records = ts.get_num_migrations()
        # We don't accept Python negative indexes here.
        with pytest.raises(IndexError):
            ts.get_migration(-1)
        for j in [0, 10, 10**6]:
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

    def test_time_units(self):
        tables = _tskit.TableCollection(1)
        tables.build_index()
        ts = _tskit.TreeSequence()
        ts.load_tables(tables)
        assert ts.get_time_units() == tskit.TIME_UNITS_UNKNOWN
        for value in ["foo", "", "💩", "null char \0 in string"]:
            tables.time_units = value
            ts = _tskit.TreeSequence()
            ts.load_tables(tables)
            assert ts.get_time_units() == value

    def test_extend_haplotypes(self):
        ts = self.get_example_tree_sequence(6)
        ets2 = ts.extend_haplotypes(2)
        ets4 = ts.extend_haplotypes(4)
        assert ets2.get_num_nodes() == ts.get_num_nodes()
        assert ets4.get_num_nodes() == ts.get_num_nodes()

    def test_extend_haplotypes_bad_args(self):
        ts1 = self.get_example_tree_sequence(10)
        with pytest.raises(TypeError):
            ts1.extend_haplotypes()
        with pytest.raises(TypeError, match="an integer"):
            ts1.extend_haplotypes("sdf")
        with pytest.raises(_tskit.LibraryError, match="positive"):
            ts1.extend_haplotypes(0)
        with pytest.raises(_tskit.LibraryError, match="positive"):
            ts1.extend_haplotypes(-1)
        tsm = self.get_example_migration_tree_sequence()
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_MIGRATIONS_NOT_SUPPORTED"
        ):
            tsm.extend_haplotypes(1)

    @pytest.mark.parametrize(
        "stat_method_name",
        [
            "D_matrix",
            "D2_matrix",
            "r2_matrix",
            "D_prime_matrix",
            "r_matrix",
            "Dz_matrix",
            "pi2_matrix",
            "D2_unbiased_matrix",
            "Dz_unbiased_matrix",
            "pi2_unbiased_matrix",
        ],
    )
    def test_ld_matrix(self, stat_method_name):
        ts = self.get_example_tree_sequence(10)
        stat_method = getattr(ts, stat_method_name)

        ss = ts.get_samples()  # sample sets
        ss_sizes = np.array([len(ss)], dtype=np.uint32)
        row_sites = np.arange(ts.get_num_sites(), dtype=np.int32)
        col_sites = row_sites
        row_pos = ts.get_breakpoints()[:-1]
        col_pos = row_pos
        row_pos_list = list(map(float, ts.get_breakpoints()[:-1]))
        col_pos_list = row_pos_list
        row_sites_list = list(range(ts.get_num_sites()))
        col_sites_list = row_sites_list

        # happy path
        a = stat_method(ss_sizes, ss, row_sites, col_sites, None, None, "site")
        assert a.shape == (10, 10, 1)
        a = stat_method(
            ss_sizes, ss, row_sites_list, col_sites_list, None, None, "site"
        )
        assert a.shape == (10, 10, 1)
        a = stat_method(ss_sizes, ss, None, None, None, None, "site")
        assert a.shape == (10, 10, 1)

        a = stat_method(ss_sizes, ss, None, None, row_pos, col_pos, "branch")
        assert a.shape == (2, 2, 1)
        a = stat_method(ss_sizes, ss, None, None, row_pos_list, col_pos_list, "branch")
        assert a.shape == (2, 2, 1)
        a = stat_method(ss_sizes, ss, None, None, None, None, "branch")
        assert a.shape == (2, 2, 1)

        # CPython API errors
        with pytest.raises(ValueError, match="Sum of sample_set_sizes"):
            bad_ss = np.array([], dtype=np.int32)
            stat_method(ss_sizes, bad_ss, row_sites, col_sites, None, None, "site")
        with pytest.raises(TypeError, match="cast array data"):
            bad_ss = np.array(ts.get_samples(), dtype=np.uint32)
            stat_method(ss_sizes, bad_ss, row_sites, col_sites, None, None, "site")
        with pytest.raises(ValueError, match="Unrecognised stats mode"):
            stat_method(ss_sizes, ss, row_sites, col_sites, None, None, "bla")
        with pytest.raises(TypeError, match="at most"):
            stat_method(ss_sizes, ss, row_sites, col_sites, None, None, "site", "abc")
        with pytest.raises(ValueError, match="invalid literal"):
            bad_sites = ["abadsite", 0, 3, 2]
            stat_method(ss_sizes, ss, bad_sites, col_sites, None, None, "site")
        with pytest.raises(TypeError):
            bad_sites = [None, 0, 3, 2]
            stat_method(ss_sizes, ss, bad_sites, col_sites, None, None, "site")
        with pytest.raises(TypeError):
            bad_sites = [{}, 0, 3, 2]
            stat_method(ss_sizes, ss, bad_sites, col_sites, None, None, "site")
        with pytest.raises(TypeError, match="Cannot cast array data"):
            bad_sites = np.array([0, 1, 2], dtype=np.uint32)
            stat_method(ss_sizes, ss, bad_sites, col_sites, None, None, "site")
        with pytest.raises(ValueError, match="invalid literal"):
            bad_sites = ["abadsite", 0, 3, 2]
            stat_method(ss_sizes, ss, row_sites, bad_sites, None, None, "site")
        with pytest.raises(TypeError):
            bad_sites = [None, 0, 3, 2]
            stat_method(ss_sizes, ss, row_sites, bad_sites, None, None, "site")
        with pytest.raises(TypeError):
            bad_sites = [{}, 0, 3, 2]
            stat_method(ss_sizes, ss, row_sites, bad_sites, None, None, "site")
        with pytest.raises(TypeError, match="Cannot cast array data"):
            bad_sites = np.array([0, 1, 2], dtype=np.uint32)
            stat_method(ss_sizes, ss, row_sites, bad_sites, None, None, "site")
        with pytest.raises(ValueError):
            bad_pos = ["abadpos", 0.1, 0.2, 2.0]
            stat_method(ss_sizes, ss, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(TypeError):
            bad_pos = [{}, 0.1, 0.2, 2.0]
            stat_method(ss_sizes, ss, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(ValueError):
            bad_pos = ["abadpos", 0, 3, 2]
            stat_method(ss_sizes, ss, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(TypeError):
            bad_pos = [{}, 0, 3, 2]
            stat_method(ss_sizes, ss, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(ValueError, match="Cannot specify sites in branch mode"):
            stat_method(ss_sizes, ss, row_sites, col_sites, None, None, "branch")
        with pytest.raises(ValueError, match="Cannot specify positions in site mode"):
            stat_method(ss_sizes, ss, None, None, row_pos, col_pos, "site")
        # C API errors
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_UNSORTED_SITES"):
            bad_sites = np.array([1, 0, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, bad_sites, col_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_UNSORTED_SITES"):
            bad_sites = np.array([1, 0, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, row_sites, bad_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_DUPLICATE_SITES"):
            bad_sites = np.array([1, 1, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, bad_sites, col_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_DUPLICATE_SITES"):
            bad_sites = np.array([1, 1, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, row_sites, bad_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_SITE_OUT_OF_BOUNDS"):
            bad_sites = np.array([-1, 0, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, bad_sites, col_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_SITE_OUT_OF_BOUNDS"):
            bad_sites = np.array([-1, 0, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, row_sites, bad_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_UNSORTED_POSITIONS"):
            bad_pos = np.array([0.7, 0, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_UNSORTED_POSITIONS"):
            bad_pos = np.array([0.7, 0, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(
            tskit.LibraryError, match="TSK_ERR_STAT_DUPLICATE_POSITIONS"
        ):
            bad_pos = np.array([0.7, 0.7, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(
            tskit.LibraryError, match="TSK_ERR_STAT_DUPLICATE_POSITIONS"
        ):
            bad_pos = np.array([0.7, 0.7, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_POSITION_OUT_OF_BOUNDS"):
            bad_pos = np.array([-0.1, 0.7, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_POSITION_OUT_OF_BOUNDS"):
            bad_pos = np.array([-0.1, 0.7, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INSUFFICIENT_SAMPLE_SETS"
        ):
            bad_ss = np.array([], dtype=np.int32)
            bad_ss_sizes = np.array([], dtype=np.uint32)
            stat_method(bad_ss_sizes, bad_ss, row_sites, col_sites, None, None, "site")
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INSUFFICIENT_SAMPLE_SETS"
        ):
            bad_ss = np.array([], dtype=np.int32)
            bad_ss_sizes = np.array([], dtype=np.uint32)
            stat_method(bad_ss_sizes, bad_ss, None, None, row_pos, col_pos, "branch")
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_EMPTY_SAMPLE_SET"):
            bad_ss = np.array([], dtype=np.int32)
            bad_ss_sizes = np.array([0], dtype=np.uint32)
            stat_method(bad_ss_sizes, bad_ss, row_sites, col_sites, None, None, "site")
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_EMPTY_SAMPLE_SET"):
            bad_ss = np.array([], dtype=np.int32)
            bad_ss_sizes = np.array([0], dtype=np.uint32)
            stat_method(bad_ss_sizes, bad_ss, None, None, row_pos, col_pos, "branch")
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_NODE_OUT_OF_BOUNDS"):
            bad_ss = np.array([1000], dtype=np.int32)
            bad_ss_sizes = np.array([1], dtype=np.uint32)
            stat_method(bad_ss_sizes, bad_ss, row_sites, col_sites, None, None, "site")
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_NODE_OUT_OF_BOUNDS"):
            bad_ss = np.array([1000], dtype=np.int32)
            bad_ss_sizes = np.array([1], dtype=np.uint32)
            stat_method(bad_ss_sizes, bad_ss, None, None, row_pos, col_pos, "branch")
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_DUPLICATE_SAMPLE"):
            bad_ss = np.array([2, 2], dtype=np.int32)
            bad_ss_sizes = np.array([2], dtype=np.uint32)
            stat_method(bad_ss_sizes, bad_ss, row_sites, col_sites, None, None, "site")
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_DUPLICATE_SAMPLE"):
            bad_ss = np.array([2, 2], dtype=np.int32)
            bad_ss_sizes = np.array([2], dtype=np.uint32)
            stat_method(bad_ss_sizes, bad_ss, None, None, row_pos, col_pos, "branch")
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_UNSUPPORTED_STAT_MODE"):
            stat_method(ss_sizes, ss, col_sites, row_sites, None, None, "node")

    @pytest.mark.parametrize(
        "stat_method_name",
        [
            "D2_ij_matrix",
            "r2_ij_matrix",
            "D2_ij_unbiased_matrix",
        ],
    )
    def test_ld_matrix_multipop(self, stat_method_name):
        ts = self.get_example_tree_sequence(10)
        stat_method = getattr(ts, stat_method_name)

        num_samples = len(ts.get_samples())
        ss = np.hstack([ts.get_samples(), ts.get_samples()])  # sample sets
        ss_sizes = np.array([num_samples, num_samples], dtype=np.uint32)
        indexes = [(0, 0), (0, 1)]
        row_sites = np.arange(ts.get_num_sites(), dtype=np.int32)
        col_sites = row_sites
        row_pos = ts.get_breakpoints()[:-1]
        col_pos = row_pos
        row_pos_list = list(map(float, ts.get_breakpoints()[:-1]))
        col_pos_list = row_pos_list
        row_sites_list = list(range(ts.get_num_sites()))
        col_sites_list = row_sites_list

        # happy path
        a = stat_method(ss_sizes, ss, indexes, row_sites, col_sites, None, None, "site")
        assert a.shape == (10, 10, 2)
        a = stat_method(
            ss_sizes, ss, indexes, row_sites_list, col_sites_list, None, None, "site"
        )
        assert a.shape == (10, 10, 2)
        a = stat_method(ss_sizes, ss, indexes, None, None, None, None, "site")
        assert a.shape == (10, 10, 2)

        a = stat_method(ss_sizes, ss, indexes, None, None, row_pos, col_pos, "branch")
        assert a.shape == (2, 2, 2)
        a = stat_method(
            ss_sizes, ss, indexes, None, None, row_pos_list, col_pos_list, "branch"
        )
        assert a.shape == (2, 2, 2)
        a = stat_method(ss_sizes, ss, indexes, None, None, None, None, "branch")
        assert a.shape == (2, 2, 2)

        # CPython API errors
        with pytest.raises(ValueError, match="Sum of sample_set_sizes"):
            bad_ss = np.array([], dtype=np.int32)
            stat_method(
                ss_sizes, bad_ss, indexes, row_sites, col_sites, None, None, "site"
            )
        with pytest.raises(TypeError, match="cast array data"):
            bad_ss = np.array(ts.get_samples(), dtype=np.uint32)
            stat_method(
                ss_sizes, bad_ss, indexes, row_sites, col_sites, None, None, "site"
            )
        with pytest.raises(ValueError, match="Unrecognised stats mode"):
            stat_method(ss_sizes, ss, indexes, row_sites, col_sites, None, None, "bla")
        with pytest.raises(TypeError, match="at most"):
            stat_method(
                ss_sizes, ss, indexes, row_sites, col_sites, None, None, "site", "abc"
            )
        with pytest.raises(ValueError, match="invalid literal"):
            bad_sites = ["abadsite", 0, 3, 2]
            stat_method(ss_sizes, ss, indexes, bad_sites, col_sites, None, None, "site")
        with pytest.raises(TypeError):
            bad_sites = [None, 0, 3, 2]
            stat_method(ss_sizes, ss, indexes, bad_sites, col_sites, None, None, "site")
        with pytest.raises(TypeError):
            bad_sites = [{}, 0, 3, 2]
            stat_method(ss_sizes, ss, indexes, bad_sites, col_sites, None, None, "site")
        with pytest.raises(TypeError, match="Cannot cast array data"):
            bad_sites = np.array([0, 1, 2], dtype=np.uint32)
            stat_method(ss_sizes, ss, indexes, bad_sites, col_sites, None, None, "site")
        with pytest.raises(ValueError, match="invalid literal"):
            bad_sites = ["abadsite", 0, 3, 2]
            stat_method(ss_sizes, ss, indexes, row_sites, bad_sites, None, None, "site")
        with pytest.raises(TypeError):
            bad_sites = [None, 0, 3, 2]
            stat_method(ss_sizes, ss, indexes, row_sites, bad_sites, None, None, "site")
        with pytest.raises(TypeError):
            bad_sites = [{}, 0, 3, 2]
            stat_method(ss_sizes, ss, indexes, row_sites, bad_sites, None, None, "site")
        with pytest.raises(TypeError, match="Cannot cast array data"):
            bad_sites = np.array([0, 1, 2], dtype=np.uint32)
            stat_method(ss_sizes, ss, indexes, row_sites, bad_sites, None, None, "site")
        with pytest.raises(ValueError):
            bad_pos = ["abadpos", 0.1, 0.2, 2.0]
            stat_method(ss_sizes, ss, indexes, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(TypeError):
            bad_pos = [{}, 0.1, 0.2, 2.0]
            stat_method(ss_sizes, ss, indexes, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(ValueError):
            bad_pos = ["abadpos", 0, 3, 2]
            stat_method(ss_sizes, ss, indexes, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(TypeError):
            bad_pos = [{}, 0, 3, 2]
            stat_method(ss_sizes, ss, indexes, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(ValueError, match="Cannot specify sites in branch mode"):
            stat_method(
                ss_sizes, ss, indexes, row_sites, col_sites, None, None, "branch"
            )
        with pytest.raises(ValueError, match="Cannot specify positions in site mode"):
            stat_method(ss_sizes, ss, indexes, None, None, row_pos, col_pos, "site")
        # C API errors
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_UNSORTED_SITES"):
            bad_sites = np.array([1, 0, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, indexes, bad_sites, col_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_UNSORTED_SITES"):
            bad_sites = np.array([1, 0, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, indexes, row_sites, bad_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_DUPLICATE_SITES"):
            bad_sites = np.array([1, 1, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, indexes, bad_sites, col_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_DUPLICATE_SITES"):
            bad_sites = np.array([1, 1, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, indexes, row_sites, bad_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_SITE_OUT_OF_BOUNDS"):
            bad_sites = np.array([-1, 0, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, indexes, bad_sites, col_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_SITE_OUT_OF_BOUNDS"):
            bad_sites = np.array([-1, 0, 2], dtype=np.int32)
            stat_method(ss_sizes, ss, indexes, row_sites, bad_sites, None, None, "site")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_UNSORTED_POSITIONS"):
            bad_pos = np.array([0.7, 0, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, indexes, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_STAT_UNSORTED_POSITIONS"):
            bad_pos = np.array([0.7, 0, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, indexes, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(
            tskit.LibraryError, match="TSK_ERR_STAT_DUPLICATE_POSITIONS"
        ):
            bad_pos = np.array([0.7, 0.7, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, indexes, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(
            tskit.LibraryError, match="TSK_ERR_STAT_DUPLICATE_POSITIONS"
        ):
            bad_pos = np.array([0.7, 0.7, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, indexes, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_POSITION_OUT_OF_BOUNDS"):
            bad_pos = np.array([-0.1, 0.7, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, indexes, None, None, bad_pos, col_pos, "branch")
        with pytest.raises(tskit.LibraryError, match="TSK_ERR_POSITION_OUT_OF_BOUNDS"):
            bad_pos = np.array([-0.1, 0.7, 0.8], dtype=np.float64)
            stat_method(ss_sizes, ss, indexes, None, None, row_pos, bad_pos, "branch")
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INSUFFICIENT_SAMPLE_SETS"
        ):
            bad_ss = np.array([], dtype=np.int32)
            bad_ss_sizes = np.array([], dtype=np.uint32)
            stat_method(
                bad_ss_sizes, bad_ss, indexes, row_sites, col_sites, None, None, "site"
            )
        with pytest.raises(
            _tskit.LibraryError, match="TSK_ERR_INSUFFICIENT_SAMPLE_SETS"
        ):
            bad_ss = np.array([], dtype=np.int32)
            bad_ss_sizes = np.array([], dtype=np.uint32)
            stat_method(
                bad_ss_sizes, bad_ss, indexes, None, None, row_pos, col_pos, "branch"
            )
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_SAMPLE_SET_INDEX"):
            bad_ss = np.array([], dtype=np.int32)
            bad_ss_sizes = np.array([0], dtype=np.uint32)
            stat_method(
                bad_ss_sizes, bad_ss, indexes, row_sites, col_sites, None, None, "site"
            )
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_SAMPLE_SET_INDEX"):
            bad_ss = np.array([], dtype=np.int32)
            bad_ss_sizes = np.array([0], dtype=np.uint32)
            stat_method(
                bad_ss_sizes, bad_ss, indexes, None, None, row_pos, col_pos, "branch"
            )
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_NODE_OUT_OF_BOUNDS"):
            bad_ss = np.array([1000, 1000], dtype=np.int32)
            bad_ss_sizes = np.array([1, 1], dtype=np.uint32)
            stat_method(
                bad_ss_sizes, bad_ss, indexes, row_sites, col_sites, None, None, "site"
            )
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_NODE_OUT_OF_BOUNDS"):
            bad_ss = np.array([1000, 1000], dtype=np.int32)
            bad_ss_sizes = np.array([1, 1], dtype=np.uint32)
            stat_method(
                bad_ss_sizes, bad_ss, indexes, None, None, row_pos, col_pos, "branch"
            )
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_DUPLICATE_SAMPLE"):
            bad_ss = np.array([1, 1, 2, 3], dtype=np.int32)
            bad_ss_sizes = np.array([2, 2], dtype=np.uint32)
            stat_method(
                bad_ss_sizes, bad_ss, indexes, row_sites, col_sites, None, None, "site"
            )
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_DUPLICATE_SAMPLE"):
            bad_ss = np.array([1, 1, 2, 3], dtype=np.int32)
            bad_ss_sizes = np.array([2, 2], dtype=np.uint32)
            stat_method(
                bad_ss_sizes, bad_ss, indexes, None, None, row_pos, col_pos, "branch"
            )
        with pytest.raises(ValueError, match="indexes must be a"):
            bad_indexes = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
            stat_method(
                ss_sizes, ss, bad_indexes, row_sites, col_sites, None, None, "site"
            )
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_UNSUPPORTED_STAT_MODE"):
            stat_method(ss_sizes, ss, indexes, col_sites, row_sites, None, None, "node")

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

    def test_divergence_matrix(self):
        n = 10
        ts = self.get_example_tree_sequence(n, random_seed=12)
        windows = [0, ts.get_sequence_length()]
        ids = np.arange(n, dtype=np.int32)
        sizes = np.ones(n, dtype=np.uint64)
        D = ts.divergence_matrix(windows, sizes, ids)
        assert D.shape == (1, n, n)
        D = ts.divergence_matrix(windows, sample_set_sizes=[1, 1], sample_sets=[0, 1])
        assert D.shape == (1, 2, 2)
        D = ts.divergence_matrix(
            windows, sample_set_sizes=[1, 1], sample_sets=[0, 1], span_normalise=True
        )
        assert D.shape == (1, 2, 2)

        for bad_node in [-1, -2, 1000]:
            with pytest.raises(_tskit.LibraryError, match="TSK_ERR_NODE_OUT_OF_BOUNDS"):
                ts.divergence_matrix(windows, [1, 1], [0, bad_node])
        with pytest.raises(ValueError, match="Sum of sample_set_sizes"):
            ts.divergence_matrix(windows, [1, 2], [0, 1])
        with pytest.raises((ValueError, OverflowError), match="Overflow|out of bounds"):
            ts.divergence_matrix(windows, [-1, 2], [0])

        with pytest.raises(TypeError, match="str"):
            ts.divergence_matrix(windows, sizes, ids, span_normalise="xdf")
        with pytest.raises(TypeError):
            ts.divergence_matrix(windoze=[0, 1])
        with pytest.raises(ValueError, match="at least 2"):
            ts.divergence_matrix(
                [0],
                sizes,
                ids,
            )
        with pytest.raises(_tskit.LibraryError, match="BAD_WINDOWS"):
            ts.divergence_matrix([-1, 0, 1], sizes, ids)
        with pytest.raises(ValueError, match="Unrecognised stats mode"):
            ts.divergence_matrix([0, 1], sizes, ids, mode="sdf")
        with pytest.raises(_tskit.LibraryError, match="UNSUPPORTED_STAT_MODE"):
            ts.divergence_matrix([0, 1], sizes, ids, mode="node")

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
        data_tables = [t for t in tskit.TABLE_NAMES if t != "provenances"]
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

        # Check for attributes that are not cleared
        assert tables.sequence_length == ts_fixture.tables.sequence_length
        assert tables.time_units == ts_fixture.tables.time_units

    def test_discrete_genome(self):
        tables = _tskit.TableCollection(1)
        tables.build_index()
        ts = _tskit.TreeSequence()
        ts.load_tables(tables)
        assert ts.get_discrete_genome() == 1

    def test_discrete_time(self):
        tables = _tskit.TableCollection(1)
        tables.build_index()
        ts = _tskit.TreeSequence()
        ts.load_tables(tables)
        assert ts.get_discrete_time() == 1

    def test_min_time(self):
        ts = self.get_example_tree_sequence(5)
        assert isinstance(ts.get_min_time(), float)

    def test_max_time(self):
        ts = self.get_example_tree_sequence(5)
        assert isinstance(ts.get_max_time(), float)

    def test_split_edges_return_type(self):
        ts = self.get_example_tree_sequence()
        split = ts.split_edges(time=0, flags=0, population=0, metadata=b"")
        assert isinstance(split, _tskit.TreeSequence)

    def test_split_edges_bad_types(self):
        ts = self.get_example_tree_sequence()

        def f(time=0, flags=0, population=0, metadata=b""):
            return ts.split_edges(
                time=time,
                flags=flags,
                population=population,
                metadata=metadata,
            )

        with pytest.raises(TypeError):
            f(time="0")
        with pytest.raises(TypeError):
            f(flags="0")
        with pytest.raises(TypeError):
            f(metadata="0")

    def test_split_edges_bad_population(self):
        ts = self.get_example_tree_sequence()
        with pytest.raises(_tskit.LibraryError, match="POPULATION_OUT_OF_BOUNDS"):
            ts.split_edges(
                time=0,
                flags=0,
                population=ts.get_num_populations(),
                metadata=b"",
            )

    @pytest.mark.parametrize("name", ARRAY_NAMES)
    def test_array_read_only(self, name, ts_fixture):
        ts_fixture = ts_fixture.ll_tree_sequence
        with pytest.raises(AttributeError, match="not writable"):
            setattr(ts_fixture, name, None)
        with pytest.raises(AttributeError, match="not writable"):
            delattr(ts_fixture, name)

        a = getattr(ts_fixture, name)
        with pytest.raises(ValueError, match="assignment destination"):
            a[:] = 0
        with pytest.raises(ValueError, match="assignment destination"):
            a[0] = 0
        with pytest.raises(ValueError, match="cannot set WRITEABLE"):
            a.setflags(write=True)

    @pytest.mark.parametrize("name", ARRAY_NAMES)
    def test_array_properties(self, name, ts_fixture):
        ts_fixture = ts_fixture.ll_tree_sequence
        a = getattr(ts_fixture, name)
        assert not a.flags.writeable
        assert a.flags.aligned
        assert a.flags.c_contiguous
        assert not a.flags.owndata
        assert a.base == ts_fixture
        b = getattr(ts_fixture, name)
        assert a is not b
        assert np.all(a == b)
        # This checks that the underlying pointer to memory is the same in
        # both arrays.
        assert a.__array_interface__ == b.__array_interface__

    @pytest.mark.parametrize("name", ARRAY_NAMES)
    def test_array_lifetime(self, name, ts_fixture):
        ts_fixture = ts_fixture.ll_tree_sequence
        a1 = getattr(ts_fixture, name)
        a2 = a1.copy()
        assert a1 is not a2
        del ts_fixture
        # Do some memory operations
        a3 = np.ones(10**6)
        assert np.all(a1 == a2)
        del a1
        # Just do something to touch memory
        a2[:] = 0
        assert a3 is not a2

    @pytest.mark.parametrize("name", ("individuals_nodes", "mutations_edge"))
    def test_generated_columns(self, ts_fixture, name):
        name = f"get_{name}"
        ts_fixture = ts_fixture.ll_tree_sequence

        # Properties
        a = getattr(ts_fixture, name)()
        assert a.flags.aligned
        assert a.flags.c_contiguous
        assert a.flags.owndata
        b = getattr(ts_fixture, name)()
        assert a is not b
        assert np.all(a == b)

        # Lifetime
        a1 = getattr(ts_fixture, name)()
        a2 = a1.copy()
        assert a1 is not a2
        del ts_fixture
        # Do some memory operations
        a3 = np.ones(10**6)
        assert np.all(a1 == a2)
        del a1
        # Just do something to touch memory
        a2[:] = 0
        assert a3 is not a2

    @pytest.mark.skipif(not _tskit.HAS_NUMPY_2, reason="Requires NumPy 2.0+")
    @pytest.mark.parametrize(
        "string_array",
        [
            "sites_ancestral_state_string",
            "mutations_derived_state_string",
            "mutations_inherited_state_string",
            "provenances_timestamp_string",
            "provenances_record_string",
        ],
    )
    @pytest.mark.parametrize(
        "str_lengths",
        ["none", "all-0", "all-1", "all-2", "mixed", "very_long", "unicode"],
    )
    def test_string_arrays(self, ts_fixture, str_lengths, string_array):
        if str_lengths == "none":
            ts = tskit.TableCollection(1.0).tree_sequence()
        else:
            if str_lengths == "all-1":
                ts = ts_fixture
                if string_array == "sites_ancestral_state_string":
                    assert ts.num_sites > 0
                    assert {len(site.ancestral_state) for site in ts.sites()} == {1}
                elif string_array == "mutations_derived_state_string":
                    assert ts.num_mutations > 0
                    assert {len(mut.derived_state) for mut in ts.mutations()} == {1}
                elif string_array == "mutations_inherited_state_string":
                    assert ts.num_mutations > 0
                    assert {len(mut.inherited_state) for mut in ts.mutations()} == {1}
                elif string_array == "provenances_timestamp_string":
                    assert ts.num_provenances > 0
                    assert len(ts.provenance(3).timestamp) == 1
                elif string_array == "provenances_record_string":
                    assert ts.num_provenances > 0
                    assert len(ts.provenance(3).record) == 1
            else:
                tables = ts_fixture.dump_tables()

                str_map = {
                    "all-0": lambda i, item: "",
                    "all-2": lambda i, item: chr(ord("A") + (i % 26)) * 2,
                    "mixed": lambda i, item: chr(ord("A") + (i % 26)) * (i % 20),
                    "very_long": lambda i, item: "A" * 100_000_000 if i == 1 else "T",
                    "unicode": lambda i, item: "🧬" * (i + 1),
                }

                if string_array == "sites_ancestral_state_string":
                    sites = tables.sites.copy()
                    tables.sites.clear()
                    get_ancestral_state = str_map[str_lengths]
                    for i, site in enumerate(sites):
                        tables.sites.append(
                            site.replace(ancestral_state=get_ancestral_state(i, site))
                        )
                elif string_array == "mutations_derived_state_string":
                    mutations = tables.mutations.copy()
                    tables.mutations.clear()
                    get_derived_state = str_map[str_lengths]
                    for i, mutation in enumerate(mutations):
                        tables.mutations.append(
                            mutation.replace(
                                derived_state=get_derived_state(i, mutation)
                            )
                        )
                elif string_array == "mutations_inherited_state_string":
                    # For inherited state, we modify sites and mutations to create
                    # varied lengths
                    sites = tables.sites.copy()
                    tables.sites.clear()
                    get_ancestral_state = str_map[str_lengths]
                    for i, site in enumerate(sites):
                        tables.sites.append(
                            site.replace(ancestral_state=get_ancestral_state(i, site))
                        )
                    mutations = tables.mutations.copy()
                    tables.mutations.clear()
                    get_derived_state = str_map[str_lengths]
                    for i, mutation in enumerate(mutations):
                        tables.mutations.append(
                            mutation.replace(
                                derived_state=get_derived_state(i, mutation)
                            )
                        )
                elif string_array == "provenances_timestamp_string":
                    provenances = tables.provenances.copy()
                    tables.provenances.clear()
                    get_timestamp = str_map[str_lengths]
                    for i, provenance in enumerate(provenances):
                        tables.provenances.append(
                            provenance.replace(timestamp=get_timestamp(i, provenance))
                        )
                elif string_array == "provenances_record_string":
                    provenances = tables.provenances.copy()
                    tables.provenances.clear()
                    get_record = str_map[str_lengths]
                    for i, provenance in enumerate(provenances):
                        tables.provenances.append(
                            provenance.replace(record=get_record(i, provenance))
                        )

                ts = tables.tree_sequence()
        ll_ts = ts.ll_tree_sequence

        a = getattr(ll_ts, string_array)

        # Contents
        if str_lengths == "none":
            assert a.size == 0
        else:
            if string_array == "sites_ancestral_state_string":
                for site in ts.sites():
                    assert a[site.id] == site.ancestral_state
            elif string_array == "mutations_derived_state_string":
                for mutation in ts.mutations():
                    assert a[mutation.id] == mutation.derived_state
            elif string_array == "mutations_inherited_state_string":
                for mutation in ts.mutations():
                    assert a[mutation.id] == mutation.inherited_state
            elif string_array == "provenances_timestamp_string":
                for provenance in ts.provenances():
                    assert a[provenance.id] == provenance.timestamp
            elif string_array == "provenances_record_string":
                for provenance in ts.provenances():
                    assert a[provenance.id] == provenance.record

        # Read only
        with pytest.raises(AttributeError, match="not writable"):
            setattr(ll_ts, string_array, None)
        with pytest.raises(AttributeError, match="not writable"):
            delattr(ll_ts, string_array)

        with pytest.raises(ValueError, match="assignment destination"):
            a[:] = 0
        with pytest.raises(ValueError, match="assignment destination"):
            a[0] = 0

        # Properties
        assert a.dtype == np.dtypes.StringDType()
        assert a.flags.aligned
        assert a.flags.c_contiguous
        b = getattr(ll_ts, string_array)
        assert a is not b
        assert np.all(a == b)

        # Lifetime
        a1 = getattr(ll_ts, string_array)
        a2 = a1.copy()
        assert a1 is not a2
        del ll_ts
        # Do some memory operations
        a3 = np.ones(10**6)
        assert np.all(a1 == a2)
        del a1
        # Just do something to touch memory
        a2[:] = 0
        assert a3 is not a2


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

    def test_time_window_errors(self):
        ts, f, params = self.get_example()
        if "time_windows" in params:
            del params["time_windows"]

            for bad_time_windows in [[], [0]]:
                with pytest.raises(ValueError, match="must have at least 2"):
                    f(
                        time_windows=bad_time_windows,
                        **params,
                    )
            bad_time_windows = [
                [-1, np.inf],
                [0, 0, np.inf],
                [0, 10, 5, np.inf],
                [0, np.inf, np.inf],
            ]
            for bad_time_window in bad_time_windows:
                with pytest.raises(
                    _tskit.LibraryError, match="TSK_ERR_BAD_TIME_WINDOWS"
                ):
                    f(time_windows=bad_time_window, **params)

    def test_polarisation(self):
        ts, f, params = self.get_example()
        with pytest.raises(TypeError):
            f(polarised="sdf", **params)
        x1 = f(polarised=False, **params)
        x2 = f(polarised=True, **params)
        # Basic check just to run both code paths
        assert x1.shape == x2.shape

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
        ts, method, params = self.get_example()
        out_u = method(**params, polarised=True)
        out_p = method(**params, polarised=False)
        assert np.all(out_u.shape == out_p.shape)


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

    def get_example(self):
        # temporary duplicate from OneWaySampleStatsMixin to include time windows
        ts, method = self.get_method()
        params = {
            "sample_set_sizes": [ts.get_num_samples()],
            "sample_sets": ts.get_samples(),
            "windows": [0, ts.get_sequence_length()],
            "time_windows": [0, np.inf],
        }
        return ts, method, params

    @pytest.mark.parametrize("mode", ["site", "branch"])
    def test_basic_example(self, mode):
        ts = self.get_example_tree_sequence()
        n = ts.get_num_samples()
        result = ts.allele_frequency_spectrum(
            [n],
            ts.get_samples(),
            [0, ts.get_sequence_length()],
            time_windows=[0, np.inf],
            mode=mode,
        )
        assert result.shape == (1, 1, n + 1)
        result = ts.allele_frequency_spectrum(
            [n],
            ts.get_samples(),
            [0, ts.get_sequence_length()],
            time_windows=[0, np.inf],
            mode=mode,
            polarised=True,
        )
        assert result.shape == (1, 1, n + 1)

    @pytest.mark.parametrize("mode", ["site", "branch"])
    def test_output_dims(self, mode):
        ts = self.get_example_tree_sequence()
        samples = ts.get_samples()
        L = ts.get_sequence_length()
        n = len(samples)
        time_windows = [0, np.inf]

        for s in [[n], [n - 2, 2], [n - 4, 2, 2], [1] * n]:
            s = np.array(s, dtype=np.uint32)
            windows = [0, L]
            for windows in [[0, L], [0, L / 2, L], np.linspace(0, L, num=10)]:
                jafs = ts.allele_frequency_spectrum(
                    s,
                    samples,
                    windows,
                    mode=mode,
                    time_windows=time_windows,
                    polarised=True,
                )
                assert jafs.shape == tuple(
                    [len(windows) - 1] + [len(time_windows) - 1] + list(s + 1)
                )
                jafs = ts.allele_frequency_spectrum(
                    s,
                    samples,
                    windows,
                    mode=mode,
                    time_windows=time_windows,
                    polarised=False,
                )
                assert jafs.shape == tuple(
                    [len(windows) - 1] + [len(time_windows) - 1] + list(s + 1)
                )

    def test_node_mode_not_supported(self):
        ts = self.get_example_tree_sequence()
        with pytest.raises(_tskit.LibraryError):
            ts.allele_frequency_spectrum(
                [ts.get_num_samples()],
                ts.get_samples(),
                [0, ts.get_sequence_length()],
                mode="node",
                time_windows=[0, np.inf],
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


class TwoWayWeightedStatsMixin(StatsInterfaceMixin):
    """
    Tests for the weighted two way sample stats.
    """

    def get_example(self):
        ts, method = self.get_method()
        params = {
            "weights": np.zeros((ts.get_num_samples(), 2)) + 0.5,
            "indexes": [[0, 1]],
            "windows": [0, ts.get_sequence_length()],
        }
        return ts, method, params

    def test_basic_example(self):
        ts, method = self.get_method()
        div = method(
            np.zeros((ts.get_num_samples(), 1)) + 0.5,
            [[0, 1]],
            windows=[0, ts.get_sequence_length()],
        )
        assert div.shape == (1, 1)

    def test_bad_weights(self):
        ts, f, params = self.get_example()
        del params["weights"]
        n = ts.get_num_samples()

        for bad_weight_type in [None, [None, None]]:
            with pytest.raises(ValueError, match="object of too small depth"):
                f(weights=bad_weight_type, **params)

        for bad_weight_shape in [(n - 1, 1), (n + 1, 1), (0, 3)]:
            with pytest.raises(ValueError, match="First dimension must be num_samples"):
                f(weights=np.ones(bad_weight_shape), **params)

    def test_output_dims(self):
        ts, method, params = self.get_example()
        weights = params.pop("weights")
        params["windows"] = [0, ts.get_sequence_length()]

        for mode in ["site", "branch"]:
            out = method(weights[:, [0]], mode=mode, **params)
            assert out.shape == (1, 1)
            out = method(weights, mode=mode, **params)
            assert out.shape == (1, 1)
            out = method(weights[:, [0, 0, 0]], mode=mode, **params)
            assert out.shape == (1, 1)
        mode = "node"
        N = ts.get_num_nodes()
        out = method(weights[:, [0]], mode=mode, **params)
        assert out.shape == (1, N, 1)
        out = method(weights, mode=mode, **params)
        assert out.shape == (1, N, 1)
        out = method(weights[:, [0, 0, 0]], mode=mode, **params)
        assert out.shape == (1, N, 1)

    def test_set_index_errors(self):
        ts, method, params = self.get_example()
        del params["indexes"]

        def f(indexes):
            method(indexes=indexes, **params)

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


class TestGeneticRelatedness(LowLevelTestCase, TwoWaySampleStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.genetic_relatedness

    def test_options(self):
        ts, _, params = self.get_example()
        x = ts.genetic_relatedness(**params)
        new_params = params.copy()
        new_params["centre"] = False
        y = ts.genetic_relatedness(**new_params)
        assert x.shape == y.shape
        new_params["polarised"] = False
        y = ts.genetic_relatedness(**new_params)
        assert x.shape == y.shape
        del new_params["centre"]
        y = ts.genetic_relatedness(**new_params)
        assert x.shape == y.shape

        del new_params["indexes"]
        with pytest.raises(ValueError, match="object of too small depth"):
            ts.genetic_relatedness(**new_params, indexes="foo")


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


class TestWeightedGeneticRelatedness(LowLevelTestCase, TwoWayWeightedStatsMixin):
    def get_method(self):
        ts = self.get_example_tree_sequence()
        return ts, ts.genetic_relatedness_weighted

    def test_options(self):
        ts, _, params = self.get_example()
        x = ts.genetic_relatedness_weighted(**params)

        new_params = params.copy()
        new_params["centre"] = False
        y = ts.genetic_relatedness_weighted(**new_params)
        assert x.shape == y.shape
        new_params["polarised"] = False
        y = ts.genetic_relatedness_weighted(**new_params)
        assert x.shape == y.shape
        del new_params["centre"]
        y = ts.genetic_relatedness_weighted(**new_params)
        assert x.shape == y.shape

        del new_params["weights"]
        with pytest.raises(ValueError, match="First dimension"):
            ts.genetic_relatedness_weighted(
                **new_params, weights=np.ones((ts.get_num_samples() + 2, 1))
            )


class TestGeneticRelatednessVector(LowLevelTestCase):
    def get_example(self, num_weights=2):
        ts = self.get_example_tree_sequence()
        num_samples = ts.get_num_samples()
        params = {
            "weights": np.linspace(0, 1, num_weights * num_samples).reshape(
                (num_samples, num_weights)
            ),
            "windows": [0, ts.get_sequence_length()],
            "nodes": list(ts.get_samples()),
        }
        return ts, params

    @pytest.mark.parametrize("mode", ["branch"])
    @pytest.mark.parametrize("num_weights", [1, 3])
    def test_basic_example(self, mode, num_weights):
        ts, params = self.get_example(num_weights)
        ns = ts.get_num_samples()
        params["mode"] = mode
        for a, b in ([True, True], [True, False], [False, True]):
            params["span_normalise"] = a
            params["centre"] = b
            result = ts.genetic_relatedness_vector(**params)
            assert result.shape == (1, ns, num_weights)

    @pytest.mark.parametrize("mode", ["branch"])
    def test_good_nodes(self, mode):
        num_weights = 2
        ts, params = self.get_example(num_weights)
        params["mode"] = mode
        for nodes in [
            list(ts.get_samples())[:3],
            list(ts.get_samples())[:1],
            [0, ts.get_num_nodes() - 1],
        ]:
            params["nodes"] = nodes
            result = ts.genetic_relatedness_vector(**params)
            assert result.shape == (1, len(nodes), num_weights)

    def test_bad_nodes(self):
        ts, params = self.get_example()
        params["mode"] = "branch"
        for nodes in ["abc", [[1, 2]]]:
            params["nodes"] = nodes
            with pytest.raises(ValueError, match="desired array"):
                ts.genetic_relatedness_vector(**params)
        for nodes in [[-1, 3], [3, 2 * ts.get_num_nodes()]]:
            params["nodes"] = nodes
            with pytest.raises(_tskit.LibraryError, match="TSK_ERR_NODE_OUT_OF_BOUNDS"):
                ts.genetic_relatedness_vector(**params)

    def test_bad_args(self):
        ts, params = self.get_example()
        for mode in ("", "abc"):
            with pytest.raises(ValueError, match="stats mode"):
                ts.genetic_relatedness_vector(
                    params["weights"], params["windows"], mode, True, True
                )
        for mode in (None, []):
            with pytest.raises(TypeError):
                ts.genetic_relatedness_vector(
                    params["weights"], params["windows"], mode, True, True
                )
        with pytest.raises(TypeError):
            ts.genetic_relatedness_vector(
                params["weights"], params["windows"], "branch", "yes", True
            )
        with pytest.raises(TypeError):
            ts.genetic_relatedness_vector(
                params["weights"], params["windows"], "branch", True, "no"
            )

    @pytest.mark.parametrize("mode", ["site", "node"])
    def test_modes_not_supported(self, mode):
        ts, params = self.get_example()
        params["mode"] = mode
        with pytest.raises(_tskit.LibraryError):
            ts.genetic_relatedness_vector(**params)

    @pytest.mark.parametrize("mode", ["branch"])
    def test_bad_weights(self, mode):
        ts, params = self.get_example()
        del params["weights"]
        ns = ts.get_num_samples()
        for bad_weight_type in [None, [None, None]]:
            with pytest.raises(ValueError, match="object of too small depth"):
                ts.genetic_relatedness_vector(
                    weights=bad_weight_type, mode=mode, **params
                )
        for bad_weight_shape in [(ns - 1, 1), (ns + 1, 1), (0, 3)]:
            with pytest.raises(ValueError, match="First dimension must be num_samples"):
                ts.genetic_relatedness_vector(
                    weights=np.ones(bad_weight_shape), mode=mode, **params
                )

    def test_window_errors(self):
        ts, params = self.get_example()
        del params["windows"]
        for bad_array in ["asdf", None, [[[[]], [[]]]], np.zeros((10, 3, 4))]:
            with pytest.raises(ValueError):
                ts.genetic_relatedness_vector(
                    windows=bad_array, mode="branch", **params
                )

        for bad_windows in [[], [0]]:
            with pytest.raises(ValueError):
                ts.genetic_relatedness_vector(
                    windows=bad_windows, mode="branch", **params
                )
        L = ts.get_sequence_length()
        bad_windows = [
            [L, 0],
            [-1, L],
            [0, L + 0.1],
            [0, 0.1, 0.1, L],
            [0, -1, L],
        ]
        for bad_window in bad_windows:
            with pytest.raises(_tskit.LibraryError):
                ts.genetic_relatedness_vector(
                    windows=bad_window, mode="branch", **params
                )


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
                raise exception("test")  # noqa: B023

            with pytest.raises(exception):
                ts.general_stat(W, f, 1, ts.get_breakpoints())

        # Wrong output dimensions
        for bad_array in [[1, 1], range(10)]:
            with pytest.raises(ValueError):
                ts.general_stat(
                    W, lambda x: bad_array, 1, ts.get_breakpoints()  # noqa:B023
                )
        with pytest.raises(ValueError):
            ts.general_stat(W, lambda x: [1], 2, ts.get_breakpoints())

        # Bad arrays returned from f
        for bad_array in [["sdf"], 0, "w4", None]:
            with pytest.raises(ValueError):
                ts.general_stat(
                    W, lambda x: bad_array, 1, ts.get_breakpoints()  # noqa:B023
                )


class TestVariant(LowLevelTestCase):
    """
    Tests for the Variant class.
    """

    def test_uninitialised_tree_sequence(self):
        ts = _tskit.TreeSequence()
        with pytest.raises(ValueError):
            _tskit.Variant(ts)

    def test_constructor(self):
        with pytest.raises(TypeError):
            _tskit.Variant()
        with pytest.raises(TypeError):
            _tskit.Variant(None)
        ts = self.get_example_tree_sequence()
        with pytest.raises(ValueError):
            _tskit.Variant(ts, samples={})
        with pytest.raises(TypeError):
            _tskit.Variant(ts, isolated_as_missing=None)
        with pytest.raises(_tskit.LibraryError):
            _tskit.Variant(ts, samples=[-1, 2])
        with pytest.raises(TypeError):
            _tskit.Variant(ts, alleles=1234)

    def test_bad_decode(self):
        ts = self.get_example_tree_sequence()
        variant = _tskit.Variant(ts)
        with pytest.raises(tskit.LibraryError, match="Site out of bounds"):
            variant.decode(-1)
        with pytest.raises(TypeError):
            variant.decode("42")
        with pytest.raises(TypeError):
            variant.decode({})
        with pytest.raises(TypeError):
            variant.decode()

    def test_alleles(self):
        ts = self.get_example_tree_sequence()
        for bad_type in [["a", "b"], "sdf", 234]:
            with pytest.raises(TypeError):
                _tskit.Variant(ts, samples=[1, 2], alleles=bad_type)
        with pytest.raises(ValueError):
            _tskit.Variant(ts, samples=[1, 2], alleles=tuple())

        for bad_allele_type in [None, 0, b"x", []]:
            with pytest.raises(TypeError):
                _tskit.Variant(ts, samples=[1, 2], alleles=(bad_allele_type,))

    def test_samples(self):
        ts = self.get_example_tree_sequence()
        v = _tskit.Variant(ts, samples=None, alleles=None)
        assert np.array_equal(
            v.samples, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        )
        v = _tskit.Variant(ts, samples=[4, 2], alleles=None)
        assert np.array_equal(v.samples, np.array([4, 2], dtype=np.int32))
        v = _tskit.Variant(ts, samples=[], alleles=None)
        assert np.array_equal(v.samples, np.array([], dtype=np.int32))
        with pytest.raises(AttributeError):
            v.samples = [1]

    def test_isolated_as_missing(self):
        ts = self.get_example_tree_sequence()
        v = _tskit.Variant(ts)
        assert v.isolated_as_missing
        v = _tskit.Variant(ts, isolated_as_missing=True)
        assert v.isolated_as_missing
        v = _tskit.Variant(ts, isolated_as_missing=False)
        assert not v.isolated_as_missing

    def test_undecoded(self):
        tables = _tskit.TableCollection(1)
        tables.build_index()
        ts = _tskit.TreeSequence(0)
        ts.load_tables(tables)
        variant = _tskit.Variant(ts)
        assert variant.site_id == tskit.NULL
        assert np.array_equal(variant.genotypes, [])
        assert variant.alleles == ()
        assert np.array_equal(variant.samples, ())
        assert variant.isolated_as_missing

    def test_properties_unwritable(self):
        ts = self.get_example_tree_sequence()
        variant = _tskit.Variant(ts)
        with pytest.raises(AttributeError):
            variant.site_id = 1
        with pytest.raises(AttributeError):
            variant.genotypes = [1]
        with pytest.raises(AttributeError):
            variant.samples = [1]
        with pytest.raises(AttributeError):
            variant.isolated_as_missing = False
        with pytest.raises(AttributeError):
            variant.alleles = "A"

    def test_missing_data(self):
        tables = _tskit.TableCollection(1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.sites.add_row(0.1, "A")
        tables.build_index()
        ts = _tskit.TreeSequence(0)
        ts.load_tables(tables)
        variant = _tskit.Variant(ts)
        variant.decode(0)
        assert variant.site_id == 0
        assert np.array_equal(variant.genotypes, [-1, -1])
        assert variant.alleles == ("A", None)

    def test_variants_lifecycle(self):
        ts = self.get_example_tree_sequence(random_seed=42)
        variant = _tskit.Variant(ts)
        variant.decode(0)
        genotypes = variant.genotypes
        expected = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1]
        assert np.array_equal(genotypes, expected)
        del variant
        assert np.array_equal(genotypes, expected)
        variant = _tskit.Variant(ts)
        del ts
        variant.decode(0)
        del variant
        assert np.array_equal(genotypes, expected)

    @pytest.mark.parametrize("isolated_as_missing", [True, False])
    @pytest.mark.parametrize("samples", [None, [], (0,)])
    @pytest.mark.parametrize("alleles", [None, ("1", "0")])
    def test_copy(self, isolated_as_missing, samples, alleles):
        ts = self.get_example_tree_sequence(random_seed=42)
        variant = _tskit.Variant(
            ts,
            isolated_as_missing=isolated_as_missing,
            samples=samples,
            alleles=alleles,
        )

        # Test taking a copy before decode
        variant2 = variant.restricted_copy()
        assert variant.site_id == variant2.site_id
        assert variant.alleles == variant2.alleles
        assert np.array_equal(variant.genotypes, variant2.genotypes)
        assert np.array_equal(variant.samples, variant2.samples)
        assert variant.isolated_as_missing == variant2.isolated_as_missing

        variant.decode(0)
        # Everything below should work even if the Python ts is free'd
        del ts
        variant2 = variant.restricted_copy()
        assert variant.site_id == variant2.site_id
        assert variant.alleles == variant2.alleles
        assert np.array_equal(variant.genotypes, variant2.genotypes)
        assert np.array_equal(variant.samples, variant2.samples)
        assert variant.isolated_as_missing == variant2.isolated_as_missing

        # Take a copy for comparison, then move the variant to check the copy
        # doesn't move too
        genotypes = variant.genotypes
        genotypes_copy = np.array(variant.genotypes)
        alleles = variant.alleles
        site_id = variant.site_id
        variant.decode(1)
        with pytest.raises(
            tskit.LibraryError, match="Can't decode a copy of a variant"
        ):
            variant2.decode(1)
        assert site_id == variant2.site_id
        assert alleles == variant2.alleles
        # Other properties shouldn't have changed
        assert np.array_equal(variant.samples, variant2.samples)
        assert variant.isolated_as_missing == variant2.isolated_as_missing

        # Variant should be equal to the copy we took earlier
        assert np.array_equal(genotypes_copy, variant2.genotypes)
        # But not equal to the un-copies genotypes anymore as they
        # have decoded a new site as a side effect of reusing the
        # array when decoding
        assert len(variant.samples) == 0 or not np.array_equal(
            genotypes, variant2.genotypes
        )

        # Check the lifecycle of copies and copies of copies
        del variant
        variant3 = variant2.restricted_copy()
        del variant2
        assert np.array_equal(genotypes_copy, variant3.genotypes)
        genotypes3 = variant3.genotypes
        del variant3
        assert np.array_equal(genotypes_copy, genotypes3)


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

        n = ts.get_num_sites()
        assert n > 2

        with pytest.raises(ValueError):
            calc.get_r2_array(0, max_distance=-1)
        with pytest.raises(ValueError):
            calc.get_r2_array(0, direction=1000)

        for bad_max_sites in [-2, -3]:
            with pytest.raises(ValueError, match="cannot be negative"):
                calc.get_r2_array(0, max_sites=bad_max_sites)
        for bad_start_pos in [-1, n, n + 1]:
            with pytest.raises(_tskit.LibraryError):
                calc.get_r2_array(bad_start_pos)

    def test_r2_array_properties(self):
        ts = self.get_example_tree_sequence()
        calc = _tskit.LdCalculator(ts)
        n = ts.get_num_sites()
        a = calc.get_r2_array(0)
        assert a.shape == (n - 1,)
        assert a.dtype == np.float64
        assert not a.flags.writeable
        assert a.flags.aligned
        assert a.flags.c_contiguous
        assert a.flags.owndata

    def test_r2_array_lifetime(self):
        ts = self.get_example_tree_sequence()
        calc = _tskit.LdCalculator(ts)
        n = ts.get_num_sites()

        a1 = calc.get_r2_array(0)
        assert a1.shape[0] == n - 1
        a2 = a1.copy()
        assert a1 is not a2
        del calc
        # Do some memory operations
        a3 = np.ones(10**6)
        assert np.all(a1 == a2)
        del ts
        assert np.all(a1 == a2)
        del a1
        # Just do something to touch memory
        a2[:] = 0
        assert a3 is not a2


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
        norm = np.ones(m)
        ls_hmm = _tskit.LsHmm(ts, np.zeros(m), np.zeros(m))
        for bad_size in [0, m - 1, m + 1, m + 2]:
            bad_array = np.zeros(bad_size, dtype=np.int8)
            with pytest.raises(ValueError, match="haplotype array"):
                ls_hmm.forward_matrix(bad_array, fm)
            with pytest.raises(ValueError, match="haplotype array"):
                ls_hmm.backward_matrix(bad_array, norm, fm)
            with pytest.raises(ValueError, match="haplotype array"):
                ls_hmm.viterbi_matrix(bad_array, vm)
        for bad_array in [[0.002], [[], []], None]:
            with pytest.raises(ValueError):
                ls_hmm.forward_matrix(bad_array, fm)
            with pytest.raises(ValueError):
                ls_hmm.viterbi_matrix(bad_array, vm)
            with pytest.raises(ValueError):
                ls_hmm.backward_matrix(bad_array, norm, fm)

    def test_norm_input(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        cm = _tskit.CompressedMatrix(ts)
        h = np.zeros(m, dtype=np.int32)
        ls_hmm = _tskit.LsHmm(ts, np.zeros(m), np.zeros(m))
        for bad_size in [0, m - 1, m + 1, m + 2]:
            bad_array = np.zeros(bad_size)
            with pytest.raises(ValueError, match="forward_norm array"):
                ls_hmm.backward_matrix(h, bad_array, cm)

        for bad_array in [[0.002], [[], []], None]:
            with pytest.raises(ValueError):
                ls_hmm.backward_matrix(h, bad_array, cm)

    def test_output_type_errors(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        h = np.zeros(m, dtype=np.int8)
        norm = np.ones(m)
        ls_hmm = _tskit.LsHmm(ts, np.zeros(m), np.zeros(m))
        for bad_type in [ls_hmm, None, m, []]:
            with pytest.raises(TypeError):
                ls_hmm.forward_matrix(h, bad_type)
            with pytest.raises(TypeError):
                ls_hmm.viterbi_matrix(h, bad_type)
            with pytest.raises(TypeError):
                ls_hmm.backward_matrix(h, norm, bad_type)

        other_ts = self.get_example_tree_sequence()
        output = _tskit.CompressedMatrix(other_ts)
        with pytest.raises(_tskit.LibraryError):
            ls_hmm.forward_matrix(h, output)
        with pytest.raises(_tskit.LibraryError):
            ls_hmm.backward_matrix(h, norm, output)
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
                assert value >= 0
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

    def test_backward_matrix(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        fm = _tskit.CompressedMatrix(ts)
        bm = _tskit.CompressedMatrix(ts)
        h = np.zeros(m, dtype=np.int32)
        ls_hmm = _tskit.LsHmm(ts, np.zeros(m) + 0.1, np.zeros(m) + 0.1)
        ls_hmm.forward_matrix(h, fm)
        ls_hmm.backward_matrix(h, fm.normalisation_factor, bm)
        self.verify_compressed_matrix(ts, bm)

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

    ARRAY_NAMES = [
        "parent",
        "left_child",
        "right_child",
        "left_sib",
        "right_sib",
        "num_children",
        "edge",
    ]

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
                    ancestral_state,
                    mutations,
                    index,
                    metadata,
                ) in tree_sites:
                    assert st.get_left() <= position < st.get_right()
                    assert isinstance(ancestral_state, str)
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
                            edge,
                            inherited_state,
                        ) = ts.get_mutation(mut_id)
                        assert site == index
                        assert mutation_id == mut_id
                        assert st.get_parent(node) != _tskit.NULL
                        assert metadata == b""
                        assert edge != _tskit.NULL
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

    def test_seek_errors(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        for bad_type in ["", "x", {}]:
            with pytest.raises(TypeError):
                tree.seek(bad_type)
        for bad_pos in [-1, 1e6]:
            with pytest.raises(_tskit.LibraryError):
                tree.seek(bad_pos)

    def seek_skip_errors(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        for bad_type in ["", "x", {}]:
            with pytest.raises(TypeError):
                tree.seek(0, bad_type)
            with pytest.raises(TypeError):
                tree.seek_index(0, bad_type)

    def test_seek_index_errors(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        for bad_type in ["", "x", {}]:
            with pytest.raises(TypeError):
                tree.seek_index(bad_type)
        for bad_index in [-1, 10**6]:
            with pytest.raises(_tskit.LibraryError):
                tree.seek_index(bad_index)

    @pytest.mark.parametrize("skip", [True, False])
    def test_seek_zero(self, skip):
        ts = self.get_example_tree_sequence()
        tree1 = _tskit.Tree(ts)
        tree1.seek_index(0, skip)
        assert tree1.get_left() == 0
        tree2 = _tskit.Tree(ts)
        tree2.seek(0, skip)
        assert tree2.get_left() == 0

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
            # An uninitialised tree should always be zero.
            samples = ts.get_samples()
            assert st.get_left_child(st.get_virtual_root()) == samples[0]
            assert st.get_right_child(st.get_virtual_root()) == samples[-1]
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
        for bad_sample in [10**6, -1e6]:
            with pytest.raises(ValueError):
                _tskit.Tree(
                    ts,
                    options=options,
                    tracked_samples=[bad_sample],
                )
            with pytest.raises(ValueError):
                _tskit.Tree(
                    ts,
                    options=options,
                    tracked_samples=[1, bad_sample],
                )
            with pytest.raises(ValueError):
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
            st = _tskit.Tree(ts)
            # Without initialisation we should be 0 samples for every node
            # that is not a sample.
            for j in range(ts.get_num_nodes()):
                count = 1 if j < ts.get_num_samples() else 0
                assert st.get_num_samples(j) == count
                assert st.get_num_tracked_samples(j) == 0
            while st.next():
                nu = get_sample_counts(ts, st)
                nu_prime = [st.get_num_samples(j) for j in range(ts.get_num_nodes())]
                assert nu == nu_prime
                # For tracked samples, this should be all zeros.
                nu = [st.get_num_tracked_samples(j) for j in range(ts.get_num_nodes())]
                assert nu == list(0 for _ in nu)

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
                    nu = get_tracked_sample_counts(ts, st, subset)
                    nu_prime = [
                        st.get_num_tracked_samples(j) for j in range(ts.get_num_nodes())
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
            for v in [num_nodes + 1, 10**6, _tskit.NULL]:
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
        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        assert st.next()
        with pytest.raises(ValueError):
            st.get_newick(root=0, precision=-1)
        with pytest.raises(ValueError):
            st.get_newick(root=0, precision=18)
        with pytest.raises(ValueError):
            st.get_newick(root=0, precision=100)

    def test_newick_legacy_ms(self):
        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        assert st.next()
        root = st.get_left_child(st.get_virtual_root())
        ns = st.get_newick(root)
        assert "n0" in ns
        assert ns == st.get_newick(root, legacy_ms_labels=False)
        assert ns != st.get_newick(root, legacy_ms_labels=True)

    def test_cleared_tree(self):
        ts = self.get_example_tree_sequence()
        samples = ts.get_samples()

        def check_tree(tree):
            assert tree.get_index() == -1
            assert tree.get_left_child(tree.get_virtual_root()) == samples[0]
            assert tree.get_num_edges() == 0
            assert tree.get_mrca(0, 1) == _tskit.NULL
            for u in range(ts.get_num_nodes()):
                assert tree.get_parent(u) == _tskit.NULL
                assert tree.get_left_child(u) == _tskit.NULL
                assert tree.get_right_child(u) == _tskit.NULL
                assert tree.get_num_children(u) == 0
                assert tree.get_edge(u) == _tskit.NULL

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
        assert st.get_newick(0) == "n0;"
        for bad_type in [None, "", [], {}]:
            with pytest.raises(TypeError):
                st.get_newick(0, precision=bad_type)
            with pytest.raises(TypeError):
                st.get_newick(0, buffer_size=bad_type)
            with pytest.raises(TypeError):
                st.get_newick(0, legacy_ms_labels=bad_type)

    def test_newick_buffer_size(self):
        ts = self.get_example_tree_sequence()
        st = _tskit.Tree(ts)
        assert st.next
        u = st.get_left_child(st.get_virtual_root())
        newick = st.get_newick(u)
        assert newick.endswith(";")
        with pytest.raises(ValueError):
            st.get_newick(u, buffer_size=-1)
        with pytest.raises(_tskit.LibraryError):
            st.get_newick(u, buffer_size=1)
        newick2 = st.get_newick(u, len(newick))
        assert newick2 == newick
        with pytest.raises(_tskit.LibraryError):
            st.get_newick(u, buffer_size=len(newick) - 1)

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
                for j in range(ts.get_num_nodes()):
                    if (
                        t.get_parent(j) == _tskit.NULL
                        and t.get_left_child(j) == _tskit.NULL
                    ):
                        assert t.get_left_sample(j) == _tskit.NULL
                        assert t.get_right_sample(j) == _tskit.NULL
                # The roots should have all samples.
                u = t.get_left_child(t.get_virtual_root())
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

    def test_b2_errors(self):
        ts1 = self.get_example_tree_sequence(10)
        t1 = _tskit.Tree(ts1)
        t1.first()
        with pytest.raises(TypeError):
            t1.get_b2_index()
        with pytest.raises(TypeError):
            t1.get_b2_index("asdf")

    def test_b2(self):
        ts1 = self.get_example_tree_sequence(10)
        t1 = _tskit.Tree(ts1)
        t1.first()
        assert t1.get_b2_index(10) > 0

    def test_num_lineages_errors(self):
        ts1 = self.get_example_tree_sequence(10)
        t1 = _tskit.Tree(ts1)
        t1.first()
        with pytest.raises(TypeError):
            t1.get_num_lineages()
        with pytest.raises(TypeError):
            t1.get_num_lineages("asdf")
        with pytest.raises(_tskit.LibraryError, match="TIME_NONFINITE"):
            t1.get_num_lineages(np.inf)

    def test_num_lineages(self):
        ts1 = self.get_example_tree_sequence(10)
        t1 = _tskit.Tree(ts1)
        t1.first()
        assert t1.get_num_lineages(0) == 10

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
        with pytest.raises(
            _tskit.NoSampleListsError, match="requires that sample lists are stored"
        ):
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
        assert len(transitions) == n - 1
        for j in range(n - 1):
            x = n - j - 1
            assert transitions[j][0] == x
            assert transitions[j][1] == -1
            assert transitions[j][2] == x

    def test_map_mutations(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        tree.next()
        n = ts.get_num_samples()
        genotypes = np.zeros(n, dtype=np.int8)
        ancestral_state, transitions = tree.map_mutations(genotypes)
        assert ancestral_state == 0
        assert len(transitions) == 0

    def test_map_mutations_fixed_ancestral_state(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        tree.next()
        n = ts.get_num_samples()
        genotypes = np.ones(n, dtype=np.int8)
        ancestral_state, transitions = tree.map_mutations(genotypes, 0)
        assert ancestral_state == 0
        assert len(transitions) == 1

    def test_map_mutations_errors(self):
        ts = self.get_example_tree_sequence()
        tree = _tskit.Tree(ts)
        n = ts.get_num_samples()
        with pytest.raises(TypeError):
            tree.map_mutations()
        for bad_size in [0, 1, n - 1, n + 1]:
            with pytest.raises(ValueError):
                tree.map_mutations(np.zeros(bad_size, dtype=np.int8))
        for bad_type in [None, {}, set()]:
            with pytest.raises(TypeError):
                tree.map_mutations([bad_type] * n)
        for bad_type in [np.uint32, np.uint64, np.float32]:
            with pytest.raises(TypeError):
                tree.map_mutations(np.zeros(n, dtype=bad_type))
        genotypes = np.zeros(n, dtype=np.int8)
        tree.map_mutations(genotypes)
        for bad_value in [64, 65, 127, -2]:
            genotypes[0] = bad_value
            with pytest.raises(_tskit.LibraryError):
                tree.map_mutations(genotypes)

        genotypes = np.zeros(n, dtype=np.int8)
        tree.map_mutations(genotypes)
        for bad_type in ["d", []]:
            with pytest.raises(TypeError):
                tree.map_mutations(genotypes, bad_type)
        for bad_state in [-2, -1, 127, 255]:
            with pytest.raises(_tskit.LibraryError, match="Bad ancestral"):
                tree.map_mutations(genotypes, bad_state)

    @pytest.mark.parametrize("array", ARRAY_NAMES)
    def test_array_read_only(self, array):
        name = array + "_array"
        ts1 = self.get_example_tree_sequence(10)
        t1 = _tskit.Tree(ts1)
        t1.first()
        with pytest.raises(AttributeError, match="not writable"):
            setattr(t1, name, None)
        with pytest.raises(AttributeError, match="not writable"):
            delattr(t1, name)

        a = getattr(t1, name)
        with pytest.raises(ValueError, match="assignment destination"):
            a[:] = 0
        with pytest.raises(ValueError, match="assignment destination"):
            a[0] = 0
        with pytest.raises(ValueError, match="cannot set WRITEABLE"):
            a.setflags(write=True)

    @pytest.mark.parametrize("array", ARRAY_NAMES)
    def test_array_properties(self, array):
        ts1 = self.get_example_tree_sequence(10)
        t1 = _tskit.Tree(ts1)
        a = getattr(t1, array + "_array")
        t1.first()
        a = getattr(t1, array + "_array")
        assert a.dtype == np.int32
        assert a.shape == (ts1.get_num_nodes() + 1,)
        assert a.base == t1
        assert not a.flags.writeable
        assert a.flags.aligned
        assert a.flags.c_contiguous
        assert not a.flags.owndata
        b = getattr(t1, array + "_array")
        assert a is not b
        assert np.all(a == b)
        a_copy = a.copy()
        # This checks that the underlying pointer to memory is the same in
        # both arrays.
        assert a.__array_interface__ == b.__array_interface__
        t1.next()
        # NB! Because we are pointing to the underlying memory, the arrays
        # will change as we iterate along the trees! This is a gotcha, but
        # it's just something we have to document as it's a consequence of the
        # zero copy semantics.
        b = getattr(t1, array + "_array")
        assert np.all(a == b)
        assert np.any(a_copy != b)

    @pytest.mark.parametrize("array", ARRAY_NAMES)
    def test_array_lifetime(self, array):
        ts1 = self.get_example_tree_sequence(10)
        t1 = _tskit.Tree(ts1)
        t1.first()
        a1 = getattr(t1, array + "_array")
        a2 = a1.copy()
        assert a1 is not a2
        del t1
        # Do some memory operations
        a3 = np.ones(10**6)
        assert np.all(a1 == a2)
        del ts1
        assert np.all(a1 == a2)
        del a1
        # Just do something to touch memory
        a2[:] = 0
        assert a3 is not a2

    @pytest.mark.parametrize("ordering", ["preorder", "postorder"])
    def test_traversal_arrays(self, ordering):
        ts = self.get_example_tree_sequence(10)
        tree = _tskit.Tree(ts)
        tree.first()
        method = getattr(tree, "get_" + ordering)
        for bad_type in [None, {}]:
            with pytest.raises(TypeError):
                method(bad_type)
        for bad_node in [-2, 10**6]:
            with pytest.raises(_tskit.LibraryError, match="out of bounds"):
                method(bad_node)
        a = method(tree.get_virtual_root())
        assert a.dtype == np.int32
        assert not a.flags.writeable
        assert a.flags.aligned
        assert a.flags.c_contiguous
        assert a.flags.owndata


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


class TestReferenceSequenceInputErrors:
    @pytest.mark.parametrize("bad_type", [1234, b"bytes", None, {}])
    @pytest.mark.parametrize("attr", ["data", "url", "metadata_schema"])
    def test_string_bad_type(self, attr, bad_type):
        refseq = _tskit.TableCollection().reference_sequence
        with pytest.raises(TypeError, match=f"{attr} must be a string"):
            setattr(refseq, attr, bad_type)

    @pytest.mark.parametrize("bad_type", [1234, "unicode", None, {}])
    def test_metadata_bad_type(self, bad_type):
        refseq = _tskit.TableCollection().reference_sequence
        with pytest.raises(TypeError):
            refseq.metadata = bad_type

    @pytest.mark.parametrize("attr", ["data", "url", "metadata_schema"])
    def test_unicode_error(self, attr):
        refseq = _tskit.TableCollection().reference_sequence
        with pytest.raises(UnicodeEncodeError):
            setattr(refseq, attr, NON_UTF8_STRING)

    @pytest.mark.parametrize("attr", ["data", "url", "metadata", "metadata_schema"])
    def test_del_attr(self, attr):
        refseq = _tskit.TableCollection().reference_sequence
        with pytest.raises(AttributeError, match=f"Cannot del {attr}"):
            delattr(refseq, attr)


class TestReferenceSequenceUpdates:
    @pytest.mark.parametrize("value", ["abc", "🎄🌳🌴🌲🎋"])
    @pytest.mark.parametrize("attr", ["data", "url", "metadata_schema"])
    def test_set_string(self, attr, value):
        refseq = _tskit.TableCollection().reference_sequence
        assert refseq.is_null()
        setattr(refseq, attr, value)
        assert getattr(refseq, attr) == value
        assert not refseq.is_null()

    @pytest.mark.parametrize("attr", ["data", "url", "metadata_schema"])
    def test_set_string_null_none(self, attr):
        refseq = _tskit.TableCollection().reference_sequence
        assert refseq.is_null()
        setattr(refseq, attr, "a")
        assert not refseq.is_null()
        setattr(refseq, attr, "")
        assert refseq.is_null()

    @pytest.mark.parametrize("value", [b"x", b"{}", b"abc\0defg"])
    def test_set_metadata(self, value):
        refseq = _tskit.TableCollection().reference_sequence
        assert refseq.is_null()
        refseq.metadata = value
        assert not refseq.is_null()
        refseq.metadata = b""
        assert refseq.is_null()


class TestReferenceSequenceTableCollection:
    def test_references(self):
        tables = _tskit.TableCollection()
        refseq = tables.reference_sequence
        assert refseq is not tables.reference_sequence

    def test_state(self):
        tables = _tskit.TableCollection()
        refseq = tables.reference_sequence
        assert refseq.is_null()
        assert not tables.has_reference_sequence()
        # Setting any non empty string changes the state to "non-null"
        refseq.data = "x"
        assert tables.has_reference_sequence()
        assert not refseq.is_null()

    @pytest.mark.parametrize("ref_data", ["abc", "A" * 10, "🎄🌳🌴🌲🎋"])
    def test_data(self, ref_data):
        tables = _tskit.TableCollection()
        refseq = tables.reference_sequence
        assert refseq.data == ""
        refseq.data = ref_data
        assert refseq.data == ref_data
        assert tables.reference_sequence.data == ref_data

    @pytest.mark.parametrize("url", ["", "abc", "A" * 10, "🎄🌳🌴🌲🎋"])
    def test_url(self, url):
        tables = _tskit.TableCollection()
        refseq = tables.reference_sequence
        assert refseq.url == ""
        refseq.url = url
        assert refseq.url == url
        assert tables.reference_sequence.url == url

    def test_metadata_default_none(self):
        tables = _tskit.TableCollection()
        assert tables.reference_sequence.metadata_schema == ""
        assert tables.reference_sequence.metadata == b""

    # we don't actually check the form here, just pass in and out strings
    @pytest.mark.parametrize("schema", ["", "{}", "abcdefg"])
    def test_metadata_schema(self, schema):
        tables = _tskit.TableCollection()
        tables.reference_sequence.metadata_schema = schema
        assert tables.has_reference_sequence
        assert tables.reference_sequence.metadata_schema == schema

    @pytest.mark.parametrize("metadata", [b"", b"{}", b"abcdefg"])
    def test_metadata(self, metadata):
        tables = _tskit.TableCollection()
        tables.reference_sequence.metadata = metadata
        assert tables.has_reference_sequence
        assert tables.reference_sequence.metadata == metadata


class TestReferenceSequenceTreeSequence:
    def test_references(self):
        tc = _tskit.TableCollection()
        tc.sequence_length = 1
        ts = _tskit.TreeSequence()
        ts.load_tables(tc, build_indexes=True)
        refseq = ts.reference_sequence
        assert refseq is not ts.reference_sequence
        assert refseq is not tc

    def test_state(self):
        tc = _tskit.TableCollection()
        tc.sequence_length = 1
        ts = _tskit.TreeSequence()
        ts.load_tables(tc, build_indexes=True)
        assert not ts.has_reference_sequence()

    def test_write(self):
        tc = _tskit.TableCollection()
        tc.sequence_length = 1
        ts = _tskit.TreeSequence()
        ts.load_tables(tc, build_indexes=True)
        refseq = ts.reference_sequence
        with pytest.raises(AttributeError, match="read-only"):
            refseq.data = "asdf"
        with pytest.raises(AttributeError, match="read-only"):
            refseq.url = "asdf"
        with pytest.raises(AttributeError, match="read-only"):
            refseq.metadata_schema = "asdf"
        with pytest.raises(AttributeError, match="read-only"):
            refseq.metadata = "asdf"

    @pytest.mark.parametrize("ref_data", ["", "ACTG" * 10, "🎄🌳🌴🌲🎋"])
    def test_data(self, ref_data):
        tc = _tskit.TableCollection()
        tc.sequence_length = 1
        tc.reference_sequence.data = ref_data
        ts = _tskit.TreeSequence()
        ts.load_tables(tc, build_indexes=True)
        assert ts.reference_sequence.data == ref_data

    @pytest.mark.parametrize("url", ["", "ACTG" * 10, "🎄🌳🌴🌲🎋"])
    def test_url(self, url):
        tc = _tskit.TableCollection()
        tc.sequence_length = 1
        tc.reference_sequence.url = url
        ts = _tskit.TreeSequence()
        ts.load_tables(tc, build_indexes=True)
        assert ts.reference_sequence.url == url

    # we don't actually check the form here, just pass in and out strings
    @pytest.mark.parametrize("schema", ["", "{}", "abcdefg"])
    def test_metadata_schema(self, schema):
        tc = _tskit.TableCollection()
        tc.sequence_length = 1
        tc.reference_sequence.metadata_schema = schema
        ts = _tskit.TreeSequence()
        ts.load_tables(tc, build_indexes=True)
        assert ts.has_reference_sequence
        assert ts.reference_sequence.metadata_schema == schema

    @pytest.mark.parametrize("metadata", [b"", b"{}", b"abcdefg"])
    def test_metadata(self, metadata):
        tc = _tskit.TableCollection()
        tc.sequence_length = 1
        tc.reference_sequence.metadata = metadata
        ts = _tskit.TreeSequence()
        ts.load_tables(tc, build_indexes=True)
        assert ts.has_reference_sequence
        assert ts.reference_sequence.metadata == metadata


class TestModuleFunctions:
    """
    Tests for the module level functions.
    """

    def test_kastore_version(self):
        version = _tskit.get_kastore_version()
        assert version == (2, 1, 1)

    def test_tskit_version(self):
        version = _tskit.get_tskit_version()
        assert version == (1, 2, 0)

    def test_tskit_version_file(self):
        maj, min_, patch = _tskit.get_tskit_version()
        with open(f"{tskit.__path__[0]}/../../c/VERSION.txt") as f:
            assert f.read() == f"{maj}.{min_}.{patch}"


def test_uninitialised():
    # These methods work from an instance that has a NULL ref so don't check
    skip_list = [
        "TableCollection_load",
        "TreeSequence_load",
        "TreeSequence_load_tables",
    ]
    for cls_name, cls in inspect.getmembers(_tskit):
        if (
            isinstance(cls, type)
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


def test_constants():
    assert _tskit.TIME_UNITS_UNKNOWN == "unknown"
    assert _tskit.TIME_UNITS_UNCALIBRATED == "uncalibrated"


class TestPairCoalescenceCountsErrors:
    def example_ts(self, sample_size=10):
        ts = msprime.sim_ancestry(
            sample_size,
            sequence_length=1e4,
            recombination_rate=1e-8,
            random_seed=1,
            population_size=1e4,
        )
        return ts.ll_tree_sequence

    @staticmethod
    def pair_coalescence_counts(
        ts,
        sample_sets=None,
        sample_set_sizes=None,
        indexes=None,
        windows=None,
        node_bin_map=None,
        span_normalise=False,
        pair_normalise=False,
    ):
        n = ts.get_num_samples()
        N = ts.get_num_nodes()
        if sample_sets is None:
            sample_sets = np.arange(n, dtype=np.int32)
        if sample_set_sizes is None:
            sample_set_sizes = [n // 2, n - n // 2]
        if indexes is None:
            pairs = itertools.combinations_with_replacement(
                range(len(sample_set_sizes)), 2
            )
            indexes = [(i, j) for i, j in pairs]
        if windows is None:
            windows = np.array([0, 0.5, 1.0]) * ts.get_sequence_length()
        if node_bin_map is None:
            node_bin_map = np.arange(N, dtype=np.int32)
        return ts.pair_coalescence_counts(
            sample_sets=sample_sets,
            sample_set_sizes=sample_set_sizes,
            windows=windows,
            indexes=indexes,
            node_bin_map=node_bin_map,
            span_normalise=span_normalise,
            pair_normalise=pair_normalise,
        )

    def test_output_dims(self):
        ts = self.example_ts()
        coal = self.pair_coalescence_counts(ts)
        dim = (2, 3, ts.get_num_nodes())
        assert coal.shape == dim
        coal = self.pair_coalescence_counts(ts, span_normalise=True)
        assert coal.shape == dim
        coal = self.pair_coalescence_counts(ts, pair_normalise=True)
        assert coal.shape == dim

    def test_node_shuffle(self):
        rng = np.random.default_rng(1024)
        ts = self.example_ts()
        coal = self.pair_coalescence_counts(ts)
        node_bin_map = np.arange(ts.get_num_nodes(), dtype=np.int32)
        rng.shuffle(node_bin_map)
        coal_shuffle = self.pair_coalescence_counts(ts, node_bin_map=node_bin_map)
        np.testing.assert_allclose(coal_shuffle[..., node_bin_map], coal)

    @pytest.mark.parametrize("bad_node", [-1, -2, 1000])
    def test_c_tsk_err_node_out_of_bounds(self, bad_node):
        ts = self.example_ts()
        ids = np.arange(ts.get_num_samples(), dtype=np.int32)
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_NODE_OUT_OF_BOUNDS"):
            self.pair_coalescence_counts(
                ts, sample_sets=np.append(ids[:-1], bad_node).astype(np.int32)
            )

    def test_c_tsk_err_bad_windows(self):
        ts = self.example_ts()
        L = ts.get_sequence_length()
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_WINDOWS"):
            self.pair_coalescence_counts(ts, windows=[1.0, L])

    def test_c_tsk_err_bad_node_bin_map(self):
        ts = self.example_ts()
        node_bin_map = np.arange(ts.get_num_nodes(), dtype=np.int32)
        node_bin_map[0] = -10
        with pytest.raises(_tskit.LibraryError, match="BAD_NODE_BIN_MAP"):
            self.pair_coalescence_counts(ts, node_bin_map=node_bin_map)

    @pytest.mark.parametrize("bad_index", [-1, 10])
    def test_c_tsk_err_bad_sample_set_index(self, bad_index):
        ts = self.example_ts()
        with pytest.raises(_tskit.LibraryError, match="BAD_SAMPLE_SET_INDEX"):
            self.pair_coalescence_counts(ts, indexes=[(0, bad_index)])

    @pytest.mark.parametrize("bad_ss_size", [-1, 1000])
    def test_cpy_bad_sample_sets(self, bad_ss_size):
        ts = self.example_ts()
        with pytest.raises(
            (ValueError, OverflowError),
            match="Sum of sample_set_sizes|Overflow|out of bounds",
        ):
            self.pair_coalescence_counts(
                ts, sample_set_sizes=[bad_ss_size, ts.get_num_samples()]
            )

    def test_cpy_bad_parse_inputs(self):
        ts = self.example_ts()
        with pytest.raises(TypeError, match="str"):
            self.pair_coalescence_counts(ts, span_normalise="foo")

    def test_cpy_bad_windows(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="at least 2"):
            self.pair_coalescence_counts(ts, windows=[0.0])

    @pytest.mark.parametrize("indexes", [[(0, 0, 0)], np.zeros((0, 2), dtype=np.int32)])
    def test_cpy_bad_indexes(self, indexes):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="k x 2 array"):
            self.pair_coalescence_counts(ts, indexes=indexes)
        with pytest.raises(ValueError, match="too small depth"):
            self.pair_coalescence_counts(ts, indexes=np.ravel(indexes))

    def test_cpy_bad_node_bin_map(self):
        ts = self.example_ts()
        num_nodes = ts.get_num_nodes()
        node_bin_map = np.full(num_nodes, tskit.NULL, dtype=np.int32)
        with pytest.raises(ValueError, match="null values for all nodes"):
            self.pair_coalescence_counts(ts, node_bin_map=node_bin_map)
        with pytest.raises(ValueError, match="a value per node"):
            self.pair_coalescence_counts(ts, node_bin_map=node_bin_map[:-1])
        with pytest.raises(TypeError, match="cast array data"):
            self.pair_coalescence_counts(ts, node_bin_map=np.zeros(num_nodes))


class TestPairCoalescenceQuantilesErrors:
    def example_ts(self, sample_size=10):
        ts = msprime.sim_ancestry(
            sample_size,
            sequence_length=1e4,
            recombination_rate=1e-8,
            random_seed=1,
            population_size=1e4,
        )
        return ts.ll_tree_sequence

    @staticmethod
    def pair_coalescence_quantiles(
        ts,
        quantiles=None,
        sample_sets=None,
        sample_set_sizes=None,
        indexes=None,
        windows=None,
        node_bin_map=None,
    ):
        n = ts.get_num_samples()
        if quantiles is None:
            quantiles = np.linspace(0, 1, 4)
        if sample_sets is None:
            sample_sets = np.arange(n, dtype=np.int32)
        if sample_set_sizes is None:
            sample_set_sizes = [n // 2, n - n // 2]
        if indexes is None:
            pairs = itertools.combinations_with_replacement(
                range(len(sample_set_sizes)), 2
            )
            indexes = [(i, j) for i, j in pairs]
        if windows is None:
            windows = np.array([0, 0.5, 1.0]) * ts.get_sequence_length()
        if node_bin_map is None:
            _, node_bin_map = np.unique(ts.nodes_time, return_inverse=True)
            node_bin_map = node_bin_map.astype(np.int32)
        return ts.pair_coalescence_quantiles(
            sample_sets=sample_sets,
            sample_set_sizes=sample_set_sizes,
            windows=windows,
            indexes=indexes,
            node_bin_map=node_bin_map,
            quantiles=quantiles,
        )

    def test_output_dims(self):
        ts = self.example_ts()
        coal = self.pair_coalescence_quantiles(ts)
        dim = (2, 3, 4)
        assert coal.shape == dim

    @pytest.mark.parametrize("quantiles", [[1.0, 0.0], [-1.0], [2.0]])
    def test_c_tsk_err_bad_quantiles(self, quantiles):
        ts = self.example_ts()
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_QUANTILES"):
            self.pair_coalescence_quantiles(ts, quantiles=quantiles)

    def test_c_tsk_err_unsorted_times(self):
        ts = self.example_ts()
        _, node_bin_map = np.unique(ts.nodes_time, return_inverse=True)
        node_bin_map = node_bin_map[::-1]
        node_bin_map = node_bin_map.astype(np.int32)
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_UNSORTED_TIMES"):
            self.pair_coalescence_quantiles(ts, node_bin_map=node_bin_map)

    def test_c_tsk_err_bad_windows(self):
        ts = self.example_ts()
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_WINDOWS"):
            self.pair_coalescence_quantiles(ts, windows=[1.0, ts.get_sequence_length()])

    @pytest.mark.parametrize("bad_ss_size", [-1, 1000])
    def test_cpy_bad_sample_sets(self, bad_ss_size):
        ts = self.example_ts()
        with pytest.raises(
            (ValueError, OverflowError),
            match="Sum of sample_set_sizes|Overflow|out of bounds",
        ):
            self.pair_coalescence_quantiles(
                ts, sample_set_sizes=[bad_ss_size, ts.get_num_samples()]
            )

    def test_cpy_bad_windows(self):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="at least 2"):
            self.pair_coalescence_quantiles(ts, windows=[0.0])

    @pytest.mark.parametrize("indexes", [[(0, 0, 0)], np.zeros((0, 2), dtype=np.int32)])
    def test_cpy_bad_indexes(self, indexes):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="k x 2 array"):
            self.pair_coalescence_quantiles(ts, indexes=indexes)
        with pytest.raises(ValueError, match="too small depth"):
            self.pair_coalescence_quantiles(ts, indexes=np.ravel(indexes))

    def test_cpy_bad_node_bin_map(self):
        ts = self.example_ts()
        num_nodes = ts.get_num_nodes()
        node_bin_map = np.full(num_nodes, tskit.NULL, dtype=np.int32)
        with pytest.raises(ValueError, match="null values for all nodes"):
            self.pair_coalescence_quantiles(ts, node_bin_map=node_bin_map)
        with pytest.raises(ValueError, match="a value per node"):
            self.pair_coalescence_quantiles(ts, node_bin_map=node_bin_map[:-1])
        with pytest.raises(TypeError, match="cast array data"):
            self.pair_coalescence_quantiles(ts, node_bin_map=np.zeros(num_nodes))

    def test_cpy_bad_quantiles(self):
        ts = self.example_ts()
        quantiles = np.zeros(0)
        with pytest.raises(ValueError, match="at least one quantile"):
            self.pair_coalescence_quantiles(ts, quantiles=quantiles)
        quantiles = np.zeros((3, 3))
        with pytest.raises(ValueError, match="object too deep"):
            self.pair_coalescence_quantiles(ts, quantiles=quantiles)

    def test_cpy_bad_inputs(self):
        ts = self.example_ts()
        with pytest.raises(TypeError, match="at most 6 keyword"):
            ts.pair_coalescence_quantiles(
                sample_sets=None,
                sample_set_sizes=None,
                windows=None,
                quantiles=None,
                indexes=None,
                node_bin_map=None,
                foo="bar",
            )


class TestPairCoalescenceRatesErrors:
    def example_ts(self, sample_size=10):
        ts = msprime.sim_ancestry(
            sample_size,
            sequence_length=1e4,
            recombination_rate=1e-8,
            random_seed=1,
            population_size=1e4,
        )
        return ts.ll_tree_sequence

    @staticmethod
    def pair_coalescence_rates(
        ts,
        time_windows=None,
        sample_sets=None,
        sample_set_sizes=None,
        indexes=None,
        windows=None,
        node_bin_map=None,
    ):
        n = ts.get_num_samples()
        if time_windows is None:
            time_windows = np.array([0.0, np.mean(ts.nodes_time), np.inf])
        if sample_sets is None:
            sample_sets = np.arange(n, dtype=np.int32)
        if sample_set_sizes is None:
            sample_set_sizes = [n // 2, n - n // 2]
        if indexes is None:
            pairs = itertools.combinations_with_replacement(
                range(len(sample_set_sizes)), 2
            )
            indexes = [(i, j) for i, j in pairs]
        if windows is None:
            windows = np.array([0, 0.5, 1.0]) * ts.get_sequence_length()
        if node_bin_map is None:
            node_bin_map = np.digitize(ts.nodes_time, time_windows) - 1
            node_bin_map[node_bin_map == time_windows.size - 1] = tskit.NULL
            node_bin_map = node_bin_map.astype(np.int32)
        return ts.pair_coalescence_rates(
            sample_sets=sample_sets,
            sample_set_sizes=sample_set_sizes,
            windows=windows,
            indexes=indexes,
            node_bin_map=node_bin_map,
            time_windows=time_windows,
        )

    def test_output_dims(self):
        ts = self.example_ts()
        coal = self.pair_coalescence_rates(ts)
        dim = (2, 3, 2)
        assert coal.shape == dim

    def test_c_tsk_err_bad_time_windows(self):
        ts = self.example_ts()
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_TIME_WINDOWS"):
            self.pair_coalescence_rates(ts, time_windows=np.array([np.inf, 0.0]))
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_TIME_WINDOWS_END"):
            self.pair_coalescence_rates(ts, time_windows=np.array([0.0, 10.0]))

    def test_c_tsk_err_bad_node_time_window(self):
        ts = self.example_ts()
        node_bin_map = np.zeros(ts.nodes_time.size, dtype=np.int32)
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_NODE_TIME_WINDOW"):
            self.pair_coalescence_rates(ts, node_bin_map=node_bin_map)

    def test_c_tsk_err_bad_sample_pair_times(self):
        ts = self.example_ts()
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_SAMPLE_PAIR_TIMES"):
            self.pair_coalescence_rates(ts, time_windows=np.array([-1.0, np.inf]))

    def test_c_tsk_err_bad_windows(self):
        ts = self.example_ts()
        with pytest.raises(_tskit.LibraryError, match="TSK_ERR_BAD_WINDOWS"):
            self.pair_coalescence_rates(ts, windows=[1.0, ts.get_sequence_length()])

    @pytest.mark.parametrize("bad_ss_size", [-1, 1000])
    def test_cpy_bad_sample_sets(self, bad_ss_size):
        ts = self.example_ts()
        with pytest.raises(
            (ValueError, OverflowError),
            match="Sum of sample_set_sizes|Overflow|out of bounds",
        ):
            self.pair_coalescence_rates(
                ts, sample_set_sizes=[bad_ss_size, ts.get_num_samples()]
            )

    @pytest.mark.parametrize("indexes", [[(0, 0, 0)], np.zeros((0, 2), dtype=np.int32)])
    def test_cpy_bad_indexes(self, indexes):
        ts = self.example_ts()
        with pytest.raises(ValueError, match="k x 2 array"):
            self.pair_coalescence_rates(ts, indexes=indexes)
        with pytest.raises(ValueError, match="too small depth"):
            self.pair_coalescence_rates(ts, indexes=np.ravel(indexes))

    def test_cpy_bad_node_bin_map(self):
        ts = self.example_ts()
        num_nodes = ts.get_num_nodes()
        node_bin_map = np.full(num_nodes, tskit.NULL, dtype=np.int32)
        with pytest.raises(ValueError, match="null values for all nodes"):
            self.pair_coalescence_rates(ts, node_bin_map=node_bin_map)
        with pytest.raises(ValueError, match="a value per node"):
            self.pair_coalescence_rates(ts, node_bin_map=node_bin_map[:-1])
        with pytest.raises(TypeError, match="cast array data"):
            self.pair_coalescence_rates(ts, node_bin_map=np.zeros(num_nodes))

    def test_cpy_bad_time_windows(self):
        ts = self.example_ts()
        time_windows = np.zeros(1)
        node_bin_map = np.zeros(ts.nodes_time.size, dtype=np.int32)
        with pytest.raises(ValueError, match="at least two breakpoints"):
            self.pair_coalescence_rates(
                ts, time_windows=time_windows, node_bin_map=node_bin_map
            )
        time_windows = np.zeros((3, 3))
        with pytest.raises(ValueError, match="object too deep"):
            self.pair_coalescence_rates(
                ts, time_windows=time_windows, node_bin_map=node_bin_map
            )

    def test_cpy_bad_inputs(self):
        ts = self.example_ts()
        with pytest.raises(TypeError, match="at most 6 keyword"):
            ts.pair_coalescence_rates(
                sample_sets=None,
                sample_set_sizes=None,
                windows=None,
                quantiles=None,
                indexes=None,
                node_bin_map=None,
                foo="bar",
            )
