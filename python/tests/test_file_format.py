# MIT License
#
# Copyright (c) 2018-2025 Tskit Developers
# Copyright (c) 2016-2018 University of Oxford
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
Test cases for tskit's file format.
"""
import os
import tempfile
import unittest
import uuid as _uuid

import kastore
import msprime
import numpy as np
import pytest
import tszip as tszip

import tests.tsutil as tsutil
import tskit
import tskit.exceptions as exceptions

CURRENT_FILE_MAJOR = 12
CURRENT_FILE_MINOR = 7

test_data_dir = os.path.join(os.path.dirname(__file__), "data")


def single_locus_no_mutation_example():
    return msprime.simulate(10, random_seed=10)


def single_locus_with_mutation_example():
    return msprime.simulate(10, mutation_rate=10, random_seed=11)


def multi_locus_with_mutation_example():
    return msprime.simulate(
        10, recombination_rate=1, length=10, mutation_rate=10, random_seed=2
    )


def recurrent_mutation_example():
    ts = msprime.simulate(10, recombination_rate=1, length=10, random_seed=2)
    return tsutil.insert_branch_mutations(ts)


def general_mutation_example():
    ts = msprime.simulate(10, recombination_rate=1, length=10, random_seed=2)
    tables = ts.dump_tables()
    tables.sites.add_row(position=0, ancestral_state="A", metadata=b"{}")
    tables.sites.add_row(position=1, ancestral_state="C", metadata=b"{'id':1}")
    tables.mutations.add_row(site=0, node=0, derived_state="T")
    tables.mutations.add_row(site=1, node=0, derived_state="G")
    return tables.tree_sequence()


def multichar_mutation_example():
    ts = msprime.simulate(10, recombination_rate=1, length=10, random_seed=2)
    return tsutil.insert_multichar_mutations(ts)


def migration_example():
    n = 10
    t = 1
    population_configurations = [
        msprime.PopulationConfiguration(n // 2),
        msprime.PopulationConfiguration(n // 2),
        msprime.PopulationConfiguration(0),
    ]
    demographic_events = [
        msprime.MassMigration(time=t, source=0, destination=2),
        msprime.MassMigration(time=t, source=1, destination=2),
    ]
    ts = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events,
        random_seed=1,
        mutation_rate=1,
        record_migrations=True,
    )
    tables = ts.dump_tables()
    for j in range(n):
        tables.individuals.add_row(flags=j, location=(j, j), parents=(j - 1, j - 1))
    return tables.tree_sequence()


def bottleneck_example():
    return msprime.simulate(
        random_seed=1,
        sample_size=100,
        recombination_rate=0.1,
        length=10,
        demographic_events=[
            msprime.SimpleBottleneck(time=0.01, population=0, proportion=0.75)
        ],
    )


def historical_sample_example():
    return msprime.simulate(
        recombination_rate=0.1,
        length=10,
        random_seed=1,
        samples=[(0, j) for j in range(10)],
    )


def no_provenance_example():
    ts = msprime.simulate(10, random_seed=1)
    tables = ts.dump_tables()
    tables.provenances.clear()
    return tables.tree_sequence()


def provenance_timestamp_only_example():
    ts = msprime.simulate(10, random_seed=1)
    tables = ts.dump_tables()
    provenances = tskit.ProvenanceTable()
    provenances.add_row(timestamp="12345", record="")
    return tables.tree_sequence()


def node_metadata_example():
    ts = msprime.simulate(
        sample_size=100, recombination_rate=0.1, length=10, random_seed=1
    )
    tables = ts.dump_tables()
    metadatas = [f"n_{u}" for u in range(ts.num_nodes)]
    packed, offset = tskit.pack_strings(metadatas)
    tables.nodes.set_columns(
        metadata=packed,
        metadata_offset=offset,
        flags=tables.nodes.flags,
        time=tables.nodes.time,
    )
    return tables.tree_sequence()


def site_metadata_example():
    ts = msprime.simulate(10, length=10, random_seed=2)
    tables = ts.dump_tables()
    for j in range(10):
        tables.sites.add_row(j, ancestral_state="a", metadata=b"1234")
    return tables.tree_sequence()


def mutation_metadata_example():
    ts = msprime.simulate(10, length=10, random_seed=2)
    tables = ts.dump_tables()
    tables.sites.add_row(0, ancestral_state="a")
    for j in range(10):
        tables.mutations.add_row(site=0, node=j, derived_state="t", metadata=b"1234")
    return tables.tree_sequence()


def migration_metadata_example():
    ts = migration_example()
    tables = ts.dump_tables()
    metadatas = [f"n_{u}" for u in range(ts.num_migrations)]
    packed, offset = tskit.pack_strings(metadatas)
    tables.migrations.set_columns(
        metadata=packed,
        metadata_offset=offset,
        left=tables.migrations.left,
        right=tables.migrations.right,
        source=tables.migrations.source,
        dest=tables.migrations.dest,
        node=tables.migrations.node,
        time=tables.migrations.time,
    )
    return tables.tree_sequence()


def edge_metadata_example():
    ts = msprime.simulate(
        sample_size=100, recombination_rate=0.1, length=10, random_seed=1
    )
    tables = ts.dump_tables()
    metadatas = [f"edge_{u}" for u in range(ts.num_edges)]
    packed, offset = tskit.pack_strings(metadatas)
    tables.edges.set_columns(
        metadata=packed,
        metadata_offset=offset,
        left=tables.edges.left,
        right=tables.edges.right,
        child=tables.edges.child,
        parent=tables.edges.parent,
    )
    return tables.tree_sequence()


class TestFileFormat(unittest.TestCase):
    """
    Superclass of file format tests.
    """

    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(prefix="msp_file_test_")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)


class TestLoadLegacyExamples(TestFileFormat):
    """
    Tests using the saved legacy file examples to ensure we can load them.
    """

    def verify_tree_sequence(self, ts):
        # Just some quick checks to make sure the tree sequence makes sense.
        assert ts.sample_size > 0
        assert ts.num_edges > 0
        assert ts.num_sites > 0
        assert ts.num_mutations > 0
        assert ts.sequence_length > 0
        for t in ts.trees():
            left, right = t.interval
            assert right > left
            for site in t.sites():
                assert left <= site.position < right
                for mut in site.mutations:
                    assert mut.site == site.id

    def verify_0_3_3(self, ts):
        for table in tskit.TABLE_NAMES:
            t = getattr(ts.tables, table)
            assert t.num_rows > 0
            if hasattr(t, "metadata_schema"):
                assert t.metadata_schema == tskit.MetadataSchema({"codec": "json"})
                assert t[2].metadata == f"n_{table}_2"
        assert ts.tables.has_index()

    def test_format_too_old_raised_for_hdf5(self):
        files = [
            "msprime-0.3.0_v2.0.hdf5",
            "msprime-0.4.0_v3.1.hdf5",
            "msprime-0.5.0_v10.0.hdf5",
        ]
        for filename in files:
            path = os.path.join(test_data_dir, "hdf5-formats", filename)

            with pytest.raises(
                exceptions.FileFormatError,
                match="appears to be in HDF5 format",
            ):
                tskit.load(path)
            with pytest.raises(
                exceptions.FileFormatError,
                match="appears to be in HDF5 format",
            ):
                tskit.TableCollection.load(path)

    def test_tskit_v_0_3_3(self):
        path = os.path.join(test_data_dir, "old-formats", "tskit-0.3.3.trees")
        ts = tskit.load(path)
        self.verify_tree_sequence(ts)


class TestErrors(TestFileFormat):
    """
    Test various API errors.
    """

    def test_tszip_file(self):
        ts = msprime.simulate(5)
        tszip.compress(ts, self.temp_file)
        with pytest.raises(tskit.FileFormatError, match="appears to be in zip format"):
            tskit.load(self.temp_file)
        with pytest.raises(tskit.FileFormatError, match="appears to be in zip format"):
            tskit.TableCollection.load(self.temp_file)


class TestDumpFormat(TestFileFormat):
    """
    Tests on the on-disk file format.
    """

    def verify_keys(self, ts):
        keys = [
            "edges/child",
            "edges/left",
            "edges/metadata",
            "edges/metadata_offset",
            "edges/metadata_schema",
            "edges/parent",
            "edges/right",
            "format/name",
            "format/version",
            "indexes/edge_insertion_order",
            "indexes/edge_removal_order",
            "individuals/flags",
            "individuals/location",
            "individuals/location_offset",
            "individuals/metadata",
            "individuals/metadata_offset",
            "individuals/metadata_schema",
            "individuals/parents",
            "individuals/parents_offset",
            "metadata",
            "metadata_schema",
            "migrations/dest",
            "migrations/left",
            "migrations/metadata",
            "migrations/metadata_offset",
            "migrations/metadata_schema",
            "migrations/node",
            "migrations/right",
            "migrations/source",
            "migrations/time",
            "mutations/derived_state",
            "mutations/derived_state_offset",
            "mutations/metadata",
            "mutations/metadata_offset",
            "mutations/metadata_schema",
            "mutations/node",
            "mutations/parent",
            "mutations/site",
            "mutations/time",
            "nodes/flags",
            "nodes/individual",
            "nodes/metadata",
            "nodes/metadata_offset",
            "nodes/metadata_schema",
            "nodes/population",
            "nodes/time",
            "populations/metadata",
            "populations/metadata_offset",
            "populations/metadata_schema",
            "provenances/record",
            "provenances/record_offset",
            "provenances/timestamp",
            "provenances/timestamp_offset",
            "sequence_length",
            "sites/ancestral_state",
            "sites/ancestral_state_offset",
            "sites/metadata",
            "sites/metadata_offset",
            "sites/metadata_schema",
            "sites/position",
            "time_units",
            "uuid",
        ]
        ts.dump(self.temp_file)
        store = kastore.load(self.temp_file)
        assert sorted(list(store.keys())) == keys

    def verify_uuid(self, ts, uuid):
        assert len(uuid) == 36
        # Check that the UUID is well-formed.
        parsed = _uuid.UUID("{" + uuid + "}")
        assert str(parsed) == uuid
        assert uuid == ts.file_uuid

    def verify_dump_format(self, ts):
        ts.dump(self.temp_file)
        assert os.path.exists(self.temp_file)
        assert os.path.getsize(self.temp_file) > 0
        self.verify_keys(ts)

        store = kastore.load(self.temp_file)
        # Check the basic root attributes
        format_name = store["format/name"]
        assert np.array_equal(
            np.array(bytearray(b"tskit.trees"), dtype=np.int8), format_name
        )
        format_version = store["format/version"]
        assert format_version[0] == CURRENT_FILE_MAJOR
        assert format_version[1] == CURRENT_FILE_MINOR
        assert ts.sequence_length == store["sequence_length"][0]
        assert repr(ts.metadata_schema) == "".join(store["metadata_schema"].astype("U"))

        # Load another copy from file so we can check the uuid.
        other_ts = tskit.load(self.temp_file)
        self.verify_uuid(other_ts, store["uuid"].tobytes().decode())

        tables = ts.tables

        assert np.array_equal(tables.metadata, b"".join(store["metadata"]))
        assert np.array_equal(tables.individuals.flags, store["individuals/flags"])
        assert np.array_equal(
            tables.individuals.location, store["individuals/location"]
        )
        assert np.array_equal(
            tables.individuals.location_offset, store["individuals/location_offset"]
        )
        assert np.array_equal(tables.individuals.parents, store["individuals/parents"])
        assert np.array_equal(
            tables.individuals.parents_offset, store["individuals/parents_offset"]
        )
        assert np.array_equal(
            tables.individuals.metadata, store["individuals/metadata"]
        )
        assert np.array_equal(
            tables.individuals.metadata_offset, store["individuals/metadata_offset"]
        )
        assert repr(tables.individuals.metadata_schema) == "".join(
            store["individuals/metadata_schema"].astype("U")
        )

        assert np.array_equal(tables.nodes.flags, store["nodes/flags"])
        assert np.array_equal(tables.nodes.time, store["nodes/time"])
        assert np.array_equal(tables.nodes.population, store["nodes/population"])
        assert np.array_equal(tables.nodes.individual, store["nodes/individual"])
        assert np.array_equal(tables.nodes.metadata, store["nodes/metadata"])
        assert np.array_equal(
            tables.nodes.metadata_offset, store["nodes/metadata_offset"]
        )
        assert repr(tables.nodes.metadata_schema) == "".join(
            store["nodes/metadata_schema"].astype("U")
        )

        assert np.array_equal(tables.edges.left, store["edges/left"])
        assert np.array_equal(tables.edges.right, store["edges/right"])
        assert np.array_equal(tables.edges.parent, store["edges/parent"])
        assert np.array_equal(tables.edges.child, store["edges/child"])
        assert np.array_equal(tables.edges.metadata, store["edges/metadata"])
        assert np.array_equal(
            tables.edges.metadata_offset, store["edges/metadata_offset"]
        )
        assert repr(tables.edges.metadata_schema) == "".join(
            store["edges/metadata_schema"].astype("U")
        )

        left = tables.edges.left
        right = tables.edges.right
        parent = tables.edges.parent
        child = tables.edges.child
        time = tables.nodes.time
        in_order = sorted(
            range(ts.num_edges),
            key=lambda j: (left[j], time[parent[j]], parent[j], child[j]),
        )
        out_order = sorted(
            range(ts.num_edges),
            key=lambda j: (right[j], -time[parent[j]], -parent[j], -child[j]),
        )
        assert np.array_equal(
            np.array(in_order, dtype=np.int32),
            store["indexes/edge_insertion_order"],
        )
        assert np.array_equal(
            np.array(out_order, dtype=np.int32), store["indexes/edge_removal_order"]
        )

        assert np.array_equal(tables.migrations.left, store["migrations/left"])
        assert np.array_equal(tables.migrations.right, store["migrations/right"])
        assert np.array_equal(tables.migrations.node, store["migrations/node"])
        assert np.array_equal(tables.migrations.source, store["migrations/source"])
        assert np.array_equal(tables.migrations.dest, store["migrations/dest"])
        assert np.array_equal(tables.migrations.time, store["migrations/time"])
        assert np.array_equal(tables.migrations.metadata, store["migrations/metadata"])
        assert np.array_equal(
            tables.migrations.metadata_offset, store["migrations/metadata_offset"]
        )
        assert repr(tables.migrations.metadata_schema) == "".join(
            store["migrations/metadata_schema"].astype("U")
        )

        assert np.array_equal(tables.sites.position, store["sites/position"])
        assert np.array_equal(
            tables.sites.ancestral_state, store["sites/ancestral_state"]
        )
        assert np.array_equal(
            tables.sites.ancestral_state_offset,
            store["sites/ancestral_state_offset"],
        )
        assert np.array_equal(tables.sites.metadata, store["sites/metadata"])
        assert np.array_equal(
            tables.sites.metadata_offset, store["sites/metadata_offset"]
        )
        assert repr(tables.sites.metadata_schema) == "".join(
            store["sites/metadata_schema"].astype("U")
        )

        assert np.array_equal(tables.mutations.site, store["mutations/site"])
        assert np.array_equal(tables.mutations.node, store["mutations/node"])
        # Default mutation time is a NaN value so we want to check for
        # bit equality, not numeric equality
        assert tables.mutations.time.tobytes() == store["mutations/time"].tobytes()
        assert np.array_equal(tables.mutations.parent, store["mutations/parent"])
        assert np.array_equal(
            tables.mutations.derived_state, store["mutations/derived_state"]
        )
        assert np.array_equal(
            tables.mutations.derived_state_offset,
            store["mutations/derived_state_offset"],
        )
        assert np.array_equal(tables.mutations.metadata, store["mutations/metadata"])
        assert np.array_equal(
            tables.mutations.metadata_offset, store["mutations/metadata_offset"]
        )
        assert repr(tables.mutations.metadata_schema) == "".join(
            store["mutations/metadata_schema"].astype("U")
        )

        assert np.array_equal(
            tables.populations.metadata, store["populations/metadata"]
        )
        assert np.array_equal(
            tables.populations.metadata_offset, store["populations/metadata_offset"]
        )
        assert repr(tables.populations.metadata_schema) == "".join(
            store["populations/metadata_schema"].astype("U")
        )

        assert np.array_equal(tables.provenances.record, store["provenances/record"])
        assert np.array_equal(
            tables.provenances.record_offset, store["provenances/record_offset"]
        )
        assert np.array_equal(
            tables.provenances.timestamp, store["provenances/timestamp"]
        )
        assert np.array_equal(
            tables.provenances.timestamp_offset,
            store["provenances/timestamp_offset"],
        )

        store.close()

    def test_single_locus_no_mutation(self):
        self.verify_dump_format(single_locus_no_mutation_example())

    def test_single_locus_with_mutation(self):
        self.verify_dump_format(single_locus_with_mutation_example())

    def test_multi_locus_with_mutation(self):
        self.verify_dump_format(multi_locus_with_mutation_example())

    def test_migration_example(self):
        self.verify_dump_format(migration_example())

    def test_bottleneck_example(self):
        self.verify_dump_format(bottleneck_example())

    def test_historical_sample_example(self):
        self.verify_dump_format(historical_sample_example())

    def test_node_metadata_example(self):
        self.verify_dump_format(node_metadata_example())

    def test_edge_metadata_example(self):
        self.verify_dump_format(edge_metadata_example())

    def test_site_metadata_example(self):
        self.verify_dump_format(site_metadata_example())

    def test_mutation_metadata_example(self):
        self.verify_dump_format(mutation_metadata_example())

    def test_migration_metadata_example(self):
        self.verify_dump_format(migration_metadata_example())

    def test_general_mutation_example(self):
        self.verify_dump_format(general_mutation_example())

    def test_multichar_mutation_example(self):
        self.verify_dump_format(multichar_mutation_example())


class TestUuid(TestFileFormat):
    """
    Basic tests for the UUID generation.
    """

    def test_different_files_same_ts(self):
        ts = msprime.simulate(10)
        uuids = []
        for _ in range(10):
            ts.dump(self.temp_file)
            with kastore.load(self.temp_file) as store:
                uuids.append(store["uuid"].tobytes().decode())
        assert len(uuids) == len(set(uuids))


class TestOptionalColumns(TestFileFormat):
    """
    Checks that optional columns in the file format are correctly handled.
    """

    def test_empty_edge_metadata(self):
        ts1 = migration_example()
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        assert ts1.tables == ts2.tables
        assert len(ts1.tables.edges.metadata) == 0

        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        del all_data["edges/metadata"]
        del all_data["edges/metadata_offset"]
        kastore.dump(all_data, self.temp_file)
        ts3 = tskit.load(self.temp_file)
        assert ts1.tables == ts3.tables

    def test_empty_migration_metadata(self):
        ts1 = migration_example()
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        assert ts1.tables == ts2.tables
        assert len(ts1.tables.migrations.metadata) == 0

        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        del all_data["migrations/metadata"]
        del all_data["migrations/metadata_offset"]
        kastore.dump(all_data, self.temp_file)
        ts3 = tskit.load(self.temp_file)
        assert ts1.tables == ts3.tables

    def test_empty_mutation_time(self):
        ts1 = migration_example()
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        assert ts1.tables == ts2.tables
        assert len(ts1.tables.mutations.metadata) == 0
        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        del all_data["mutations/time"]
        kastore.dump(all_data, self.temp_file)
        ts3 = tskit.load(self.temp_file)
        # Null out the time column
        t1 = ts1.dump_tables()
        t1.mutations.time = np.full_like(t1.mutations.time, tskit.UNKNOWN_TIME)
        t1.assert_equals(ts3.tables)

    def test_empty_individual_parents(self):
        ts1 = migration_example()
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        assert ts1.tables == ts2.tables
        assert len(ts1.tables.individuals.parents) > 0
        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        del all_data["individuals/parents"]
        del all_data["individuals/parents_offset"]
        kastore.dump(all_data, self.temp_file)
        ts3 = tskit.load(self.temp_file)
        tables = ts1.dump_tables()
        tables.individuals.packset_parents(
            [
                [],
            ]
            * tables.individuals.num_rows
        )
        tables.assert_equals(ts3.tables)


class TestReferenceSequence:
    def test_fixture_has_reference_sequence(self, ts_fixture):
        assert ts_fixture.has_reference_sequence()

    def test_round_trip(self, ts_fixture, tmp_path):
        ts1 = ts_fixture
        temp_file = tmp_path / "tmp.trees"
        ts1.dump(temp_file)
        ts2 = tskit.load(temp_file)
        ts1.tables.assert_equals(ts2.tables)

    def test_no_reference_sequence(self, ts_fixture, tmp_path):
        ts1 = ts_fixture
        temp_file = tmp_path / "tmp.trees"
        ts1.dump(temp_file)
        with kastore.load(temp_file) as store:
            all_data = dict(store)
        del all_data["reference_sequence/metadata_schema"]
        del all_data["reference_sequence/metadata"]
        del all_data["reference_sequence/data"]
        del all_data["reference_sequence/url"]
        for key in all_data.keys():
            assert not key.startswith("reference_sequence")
        kastore.dump(all_data, temp_file)
        ts2 = tskit.load(temp_file)
        assert not ts2.has_reference_sequence()
        tables = ts2.dump_tables()
        tables.reference_sequence = ts1.reference_sequence
        tables.assert_equals(ts1.tables)

    @pytest.mark.parametrize("attr", ["data", "url"])
    def test_missing_attr(self, ts_fixture, tmp_path, attr):
        ts1 = ts_fixture
        temp_file = tmp_path / "tmp.trees"
        ts1.dump(temp_file)
        with kastore.load(temp_file) as store:
            all_data = dict(store)
        del all_data[f"reference_sequence/{attr}"]
        kastore.dump(all_data, temp_file)
        ts2 = tskit.load(temp_file)
        assert ts2.has_reference_sequence
        assert getattr(ts2.reference_sequence, attr) == ""

    def test_missing_metadata(self, ts_fixture, tmp_path):
        ts1 = ts_fixture
        temp_file = tmp_path / "tmp.trees"
        ts1.dump(temp_file)
        with kastore.load(temp_file) as store:
            all_data = dict(store)
        del all_data["reference_sequence/metadata"]
        kastore.dump(all_data, temp_file)
        ts2 = tskit.load(temp_file)
        assert ts2.has_reference_sequence
        assert ts2.reference_sequence.metadata_bytes == b""

    def test_missing_metadata_schema(self, ts_fixture, tmp_path):
        ts1 = ts_fixture
        temp_file = tmp_path / "tmp.trees"
        ts1.dump(temp_file)
        with kastore.load(temp_file) as store:
            all_data = dict(store)
        del all_data["reference_sequence/metadata_schema"]
        kastore.dump(all_data, temp_file)
        ts2 = tskit.load(temp_file)
        assert ts2.has_reference_sequence
        assert repr(ts2.reference_sequence.metadata_schema) == ""


class TestFileFormatErrors(TestFileFormat):
    """
    Tests for errors in the HDF5 format.
    """

    def verify_missing_fields(self, ts):
        ts.dump(self.temp_file)
        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        for key in all_data.keys():
            # We skip these keys as they are optional
            if "metadata_schema" not in key and key not in [
                "metadata",
                "time_units",
                "mutations/time",
            ]:
                data = dict(all_data)
                del data[key]
                kastore.dump(data, self.temp_file)
                with pytest.raises(
                    (exceptions.FileFormatError, exceptions.LibraryError)
                ):
                    tskit.load(self.temp_file)

    def test_missing_fields(self):
        self.verify_missing_fields(migration_example())

    def verify_equal_length_columns(self, ts, table):
        ts.dump(self.temp_file)
        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        table_cols = [
            colname for colname in all_data.keys() if colname.startswith(table)
        ]
        # Remove all the 'offset' columns
        for col in list(table_cols):
            if col.endswith("_offset"):
                main_col = col[: col.index("_offset")]
                table_cols.remove(main_col)
                table_cols.remove(col)
            if "metadata_schema" in col:
                table_cols.remove(col)
        # Remaining columns should all be the same length
        for col in table_cols:
            for bad_val in [[], all_data[col][:-1]]:
                data = dict(all_data)
                data[col] = bad_val
                kastore.dump(data, self.temp_file)
                with pytest.raises(exceptions.FileFormatError):
                    tskit.load(self.temp_file)

    def test_equal_length_columns(self):
        ts = migration_example()
        for table in ["nodes", "edges", "migrations", "sites", "mutations"]:
            self.verify_equal_length_columns(ts, table)

    def verify_offset_columns(self, ts):
        ts.dump(self.temp_file)
        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        offset_col_pairs = []
        for col in all_data.keys():
            if col.endswith("_offset"):
                main_col = col[: col.index("_offset")]
                offset_col_pairs.append((main_col, col))
        for col, offset_col in offset_col_pairs:
            num_rows = len(all_data[offset_col]) - 1
            data = dict(all_data)
            # Check bad lengths of the offset col
            for bad_col_length in [[], range(2 * num_rows)]:
                data[offset_col] = bad_col_length
                kastore.dump(data, self.temp_file)
                with pytest.raises(exceptions.FileFormatError):
                    tskit.load(self.temp_file)

            # Check for a bad offset
            data = dict(all_data)
            original_offset = data[offset_col]
            original_col = data[col]
            data[offset_col] = np.zeros_like(original_offset)
            data[col] = np.zeros(10, dtype=original_col.dtype)
            kastore.dump(data, self.temp_file)
            with pytest.raises(exceptions.LibraryError):
                tskit.load(self.temp_file)

    def test_offset_columns(self):
        ts = migration_example()
        self.verify_offset_columns(ts)

    def test_index_columns(self):
        ts = migration_example()
        ts.dump(self.temp_file)
        with kastore.load(self.temp_file) as store:
            all_data = dict(store)

        edge_removal_order = "indexes/edge_removal_order"
        edge_insertion_order = "indexes/edge_insertion_order"

        data = dict(all_data)
        del data[edge_removal_order]
        del data[edge_insertion_order]
        kastore.dump(data, self.temp_file)
        with pytest.raises(exceptions.LibraryError):
            tskit.load(self.temp_file)

        data = dict(all_data)
        del data[edge_removal_order]
        kastore.dump(data, self.temp_file)
        with pytest.raises(exceptions.LibraryError):
            tskit.load(self.temp_file)

        data = dict(all_data)
        del data[edge_insertion_order]
        kastore.dump(data, self.temp_file)
        with pytest.raises(exceptions.LibraryError):
            tskit.load(self.temp_file)

        data = dict(all_data)
        data[edge_insertion_order] = data[edge_insertion_order][:1]
        kastore.dump(data, self.temp_file)
        with pytest.raises(exceptions.FileFormatError):
            tskit.load(self.temp_file)

        data = dict(all_data)
        data[edge_removal_order] = data[edge_removal_order][:1]
        kastore.dump(data, self.temp_file)
        with pytest.raises(exceptions.FileFormatError):
            tskit.load(self.temp_file)

    def test_load_empty_kastore(self):
        kastore.dump({}, self.temp_file)
        with pytest.raises(exceptions.LibraryError):
            tskit.load(self.temp_file)

    def test_old_version_load_error(self):
        ts = msprime.simulate(10, random_seed=1)
        for bad_version in [(0, 1), (0, 8), (2, 0), (CURRENT_FILE_MAJOR - 1, 0)]:
            ts.dump(self.temp_file)
            with kastore.load(self.temp_file) as store:
                data = dict(store)
            data["format/version"] = np.array(bad_version, dtype=np.uint32)
            kastore.dump(data, self.temp_file)
            with pytest.raises(tskit.VersionTooOldError):
                tskit.load(self.temp_file)

    def test_new_version_load_error(self):
        ts = msprime.simulate(10, random_seed=1)
        for bad_version in [(CURRENT_FILE_MAJOR + j, 0) for j in range(1, 5)]:
            ts.dump(self.temp_file)
            with kastore.load(self.temp_file) as store:
                data = dict(store)
            data["format/version"] = np.array(bad_version, dtype=np.uint32)
            kastore.dump(data, self.temp_file)
            with pytest.raises(tskit.VersionTooNewError):
                tskit.load(self.temp_file)

    def test_format_name_error(self):
        ts = msprime.simulate(10)
        for bad_name in ["tskit.tree", "tskit.treesAndOther", "", "x" * 100]:
            ts.dump(self.temp_file)
            with kastore.load(self.temp_file) as store:
                data = dict(store)
            data["format/name"] = np.array(bytearray(bad_name.encode()), dtype=np.int8)
            kastore.dump(data, self.temp_file)
            with pytest.raises(exceptions.FileFormatError):
                tskit.load(self.temp_file)

    def test_load_bad_formats(self):
        # try loading a bunch of files in various formats.
        # First, check the empty file.
        with pytest.raises(EOFError):
            tskit.load(self.temp_file)
        # Now some ascii text
        with open(self.temp_file, "wb") as f:
            f.write(b"Some ASCII text")
        with pytest.raises(exceptions.FileFormatError):
            tskit.load(self.temp_file)
        # Now write 8k of random bytes
        with open(self.temp_file, "wb") as f:
            f.write(os.urandom(8192))
        with pytest.raises(exceptions.FileFormatError):
            tskit.load(self.temp_file)

    def test_load_bad_formats_fileobj(self):
        def load():
            with open(self.temp_file, "rb") as f:
                tskit.load(f)

        with pytest.raises(EOFError):
            load()
        with open(self.temp_file, "wb") as f:
            f.write(b"Some ASCII text")
        with pytest.raises(exceptions.FileFormatError):
            load()


def assert_tables_empty(tables):
    for table in tables.table_name_map.values():
        assert len(table) == 0


class TestSkipTables:
    """
    Test `skip_tables` flag to TreeSequence.load() and TableCollection.load().
    """

    def test_ts_read_path_interface(self, tmp_path, ts_fixture):
        # Check the fixture has metadata and a schema
        assert ts_fixture.metadata_schema is not None
        assert len(ts_fixture.metadata) > 0
        save_path = tmp_path / "tmp.trees"
        ts_fixture.dump(save_path)
        ts_no_tables = tskit.load(save_path, skip_tables=True)
        assert not ts_no_tables.equals(ts_fixture)
        assert ts_no_tables.equals(ts_fixture, ignore_tables=True)
        assert_tables_empty(ts_no_tables.tables)

    def test_ts_read_one_stream(self, tmp_path, ts_fixture):
        save_path = tmp_path / "tmp.trees"
        ts_fixture.dump(save_path)
        with open(save_path, "rb") as f:
            ts_no_tables = tskit.load(f, skip_tables=True)
        assert not ts_no_tables.equals(ts_fixture)
        assert ts_no_tables.equals(ts_fixture, ignore_tables=True)
        assert_tables_empty(ts_no_tables.tables)

    def test_ts_twofile_stream_noskip(self, tmp_path, ts_fixture):
        save_path = tmp_path / "tmp.trees"
        with open(save_path, "wb") as f:
            ts_fixture.dump(f)
            ts_fixture.dump(f)
        with open(save_path, "rb") as f:
            ts1 = tskit.load(f)
            ts2 = tskit.load(f)
        assert ts_fixture.equals(ts1)
        assert ts_fixture.equals(ts2)

    def test_ts_twofile_stream_fails(self, tmp_path, ts_fixture):
        # We can't skip_tables while reading from a stream
        save_path = tmp_path / "tmp.trees"
        with open(save_path, "wb") as f:
            ts_fixture.dump(f)
            ts_fixture.dump(f)
        with open(save_path, "rb") as f:
            tskit.load(f, skip_tables=True)
            with pytest.raises(exceptions.FileFormatError):
                tskit.load(f)

    def test_table_collection_load_path(self, tmp_path, ts_fixture):
        save_path = tmp_path / "tmp.trees"
        ts_fixture.dump(save_path)
        tables_skipped = tskit.TableCollection.load(save_path, skip_tables=True)
        tables = ts_fixture.tables
        assert not tables_skipped.equals(tables)
        assert tables_skipped.equals(tables, ignore_tables=True)
        assert_tables_empty(tables_skipped)

    def test_table_collection_load_stream(self, tmp_path, ts_fixture):
        save_path = tmp_path / "tmp.trees"
        ts_fixture.dump(save_path)
        with open(save_path, "rb") as f:
            tables_skipped = tskit.TableCollection.load(f, skip_tables=True)
        tables = ts_fixture.tables
        assert not tables_skipped.equals(tables)
        assert tables_skipped.equals(tables, ignore_tables=True)
        assert_tables_empty(tables_skipped)


class TestSkipReferenceSequence:
    """
    Test `skip_reference_sequence` flag to TreeSequence.load() and
    TableCollection.load().
    """

    def test_ts_load_path(self, tmp_path, ts_fixture):
        assert ts_fixture.has_reference_sequence()
        save_path = tmp_path / "tmp.trees"
        ts_fixture.dump(save_path)
        ts_no_refseq = tskit.load(save_path, skip_reference_sequence=True)
        assert not ts_no_refseq.equals(ts_fixture)
        assert ts_no_refseq.equals(ts_fixture, ignore_reference_sequence=True)
        assert not ts_no_refseq.has_reference_sequence()

    def test_ts_load_stream(self, tmp_path, ts_fixture):
        save_path = tmp_path / "tmp.trees"
        ts_fixture.dump(save_path)
        with open(save_path, "rb") as f:
            ts_no_refseq = tskit.load(f, skip_reference_sequence=True)
        assert not ts_no_refseq.equals(ts_fixture)
        assert ts_no_refseq.equals(ts_fixture, ignore_reference_sequence=True)
        assert not ts_no_refseq.has_reference_sequence()

    def test_ts_twofile_stream_fails(self, tmp_path, ts_fixture):
        # We can't skip_reference_sequence while reading from a stream
        save_path = tmp_path / "tmp.trees"
        with open(save_path, "wb") as f:
            ts_fixture.dump(f)
            ts_fixture.dump(f)
        with open(save_path, "rb") as f:
            tskit.load(f, skip_reference_sequence=True)
            with pytest.raises(exceptions.FileFormatError):
                tskit.load(f)

    def test_table_collection_load_path(self, tmp_path, ts_fixture):
        save_path = tmp_path / "tmp.trees"
        ts_fixture.dump(save_path)
        tables_no_refseq = tskit.TableCollection.load(
            save_path, skip_reference_sequence=True
        )
        tables = ts_fixture.tables
        assert not tables_no_refseq.equals(tables)
        assert tables_no_refseq.equals(tables, ignore_reference_sequence=True)
        assert not tables_no_refseq.has_reference_sequence()

    def test_table_collection_load_stream(self, tmp_path, ts_fixture):
        save_path = tmp_path / "tmp.trees"
        ts_fixture.dump(save_path)
        with open(save_path, "rb") as f:
            tables_no_refseq = tskit.TableCollection.load(
                f, skip_reference_sequence=True
            )
        tables = ts_fixture.tables
        assert not tables_no_refseq.equals(tables)
        assert tables_no_refseq.equals(tables, ignore_reference_sequence=True)
        assert not tables_no_refseq.has_reference_sequence()
