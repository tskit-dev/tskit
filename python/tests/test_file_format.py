# MIT License
#
# Copyright (c) 2018-2020 Tskit Developers
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
import json
import os
import tempfile
import unittest
import uuid as _uuid

import h5py
import kastore
import msprime
import numpy as np

import tests.tsutil as tsutil
import tskit
import tskit.exceptions as exceptions
from tskit import UNKNOWN_TIME


CURRENT_FILE_MAJOR = 12
CURRENT_FILE_MINOR = 3

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
    return ts


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
        self.assertGreater(ts.sample_size, 0)
        self.assertGreater(ts.num_edges, 0)
        self.assertGreater(ts.num_sites, 0)
        self.assertGreater(ts.num_mutations, 0)
        self.assertGreater(ts.sequence_length, 0)
        for t in ts.trees():
            left, right = t.interval
            self.assertGreater(right, left)
            for site in t.sites():
                self.assertTrue(left <= site.position < right)
                for mut in site.mutations:
                    self.assertEqual(mut.site, site.id)

    def test_format_too_old_raised_for_hdf5(self):
        files = [
            "msprime-0.3.0_v2.0.hdf5",
            "msprime-0.4.0_v3.1.hdf5",
            "msprime-0.5.0_v10.0.hdf5",
        ]
        for filename in files:
            path = os.path.join(test_data_dir, "hdf5-formats", filename)
            self.assertRaises(exceptions.VersionTooOldError, tskit.load, path)

    def test_msprime_v_0_5_0(self):
        path = os.path.join(test_data_dir, "hdf5-formats", "msprime-0.5.0_v10.0.hdf5")
        ts = tskit.load_legacy(path)
        self.verify_tree_sequence(ts)

    def test_msprime_v_0_4_0(self):
        path = os.path.join(test_data_dir, "hdf5-formats", "msprime-0.4.0_v3.1.hdf5")
        ts = tskit.load_legacy(path)
        self.verify_tree_sequence(ts)

    def test_msprime_v_0_3_0(self):
        path = os.path.join(test_data_dir, "hdf5-formats", "msprime-0.3.0_v2.0.hdf5")
        ts = tskit.load_legacy(path)
        self.verify_tree_sequence(ts)


class TestRoundTrip(TestFileFormat):
    """
    Tests if we can round trip convert a tree sequence in memory
    through a V2 file format and a V3 format.
    """

    def verify_tree_sequences_equal(self, ts, tsp, simplify=True):
        self.assertEqual(ts.sequence_length, tsp.sequence_length)
        t1 = ts.tables
        # We need to sort and squash the edges in the new format because it
        # has gone through an edgesets representation. Simplest way to do this
        # is to call simplify.
        if simplify:
            t2 = tsp.simplify().tables
        else:
            t2 = tsp.tables
        self.assertEqual(t1.nodes, t2.nodes)
        self.assertEqual(t1.edges, t2.edges)
        self.assertEqual(t1.sites, t2.sites)
        self.assertEqual(t1.mutations, t2.mutations)

    def verify_round_trip(self, ts, version):
        tskit.dump_legacy(ts, self.temp_file, version=version)
        tsp = tskit.load_legacy(self.temp_file)
        simplify = version < 10
        self.verify_tree_sequences_equal(ts, tsp, simplify=simplify)
        tsp.dump(self.temp_file)
        tsp = tskit.load(self.temp_file)
        self.verify_tree_sequences_equal(ts, tsp, simplify=simplify)
        for provenance in tsp.provenances():
            tskit.validate_provenance(json.loads(provenance.record))

    def verify_round_trip_no_legacy(self, ts):
        ts.dump(self.temp_file)
        tsp = tskit.load(self.temp_file)
        self.verify_tree_sequences_equal(ts, tsp, simplify=False)
        for provenance in tsp.provenances():
            tskit.validate_provenance(json.loads(provenance.record))

    def verify_malformed_json_v2(self, ts, group_name, attr, bad_json):
        tskit.dump_legacy(ts, self.temp_file, 2)
        # Write some bad JSON to the provenance string.
        root = h5py.File(self.temp_file, "r+")
        group = root[group_name]
        group.attrs[attr] = bad_json
        root.close()
        tsp = tskit.load_legacy(self.temp_file)
        self.verify_tree_sequences_equal(ts, tsp)

    def test_malformed_json_v2(self):
        ts = multi_locus_with_mutation_example()
        for group_name in ["trees", "mutations"]:
            for attr in ["environment", "parameters"]:
                for bad_json in ["", "{", "{},"]:
                    self.verify_malformed_json_v2(ts, group_name, attr, bad_json)

    def test_single_locus_no_mutation(self):
        self.verify_round_trip(single_locus_no_mutation_example(), 2)
        self.verify_round_trip(single_locus_no_mutation_example(), 3)
        self.verify_round_trip(single_locus_no_mutation_example(), 10)

    def test_single_locus_with_mutation(self):
        self.verify_round_trip(single_locus_with_mutation_example(), 2)
        self.verify_round_trip(single_locus_with_mutation_example(), 3)
        self.verify_round_trip(single_locus_with_mutation_example(), 10)

    def test_multi_locus_with_mutation(self):
        self.verify_round_trip(multi_locus_with_mutation_example(), 2)
        self.verify_round_trip(multi_locus_with_mutation_example(), 3)
        self.verify_round_trip(multi_locus_with_mutation_example(), 10)

    def test_migration_example(self):
        self.verify_round_trip(migration_example(), 2)
        self.verify_round_trip(migration_example(), 3)
        self.verify_round_trip(migration_example(), 10)

    def test_bottleneck_example(self):
        self.verify_round_trip(migration_example(), 3)
        self.verify_round_trip(migration_example(), 10)

    def test_no_provenance(self):
        self.verify_round_trip(no_provenance_example(), 10)

    def test_provenance_timestamp_only(self):
        self.verify_round_trip(provenance_timestamp_only_example(), 10)

    def test_recurrent_mutation_example(self):
        ts = recurrent_mutation_example()
        for version in [2, 3]:
            self.assertRaises(
                ValueError, tskit.dump_legacy, ts, self.temp_file, version
            )
        self.verify_round_trip(ts, 10)

    def test_general_mutation_example(self):
        ts = general_mutation_example()
        for version in [2, 3]:
            self.assertRaises(
                ValueError, tskit.dump_legacy, ts, self.temp_file, version
            )
        self.verify_round_trip(ts, 10)

    def test_node_metadata_example(self):
        self.verify_round_trip(node_metadata_example(), 10)

    def test_site_metadata_example(self):
        self.verify_round_trip(site_metadata_example(), 10)

    def test_mutation_metadata_example(self):
        self.verify_round_trip(mutation_metadata_example(), 10)

    def test_migration_metadata_example(self):
        self.verify_round_trip(migration_metadata_example(), 10)

    def test_edge_metadata_example(self):
        # metadata for edges was introduced
        self.verify_round_trip_no_legacy(edge_metadata_example())

    def test_multichar_mutation_example(self):
        self.verify_round_trip(multichar_mutation_example(), 10)

    def test_empty_file(self):
        tables = tskit.TableCollection(sequence_length=3)
        self.verify_round_trip(tables.tree_sequence(), 10)

    def test_zero_edges(self):
        tables = tskit.TableCollection(sequence_length=3)
        tables.nodes.add_row(time=0)
        self.verify_round_trip(tables.tree_sequence(), 10)

    def test_v2_no_samples(self):
        ts = multi_locus_with_mutation_example()
        tskit.dump_legacy(ts, self.temp_file, version=2)
        root = h5py.File(self.temp_file, "r+")
        del root["samples"]
        root.close()
        tsp = tskit.load_legacy(self.temp_file)
        self.verify_tree_sequences_equal(ts, tsp)

    def test_duplicate_mutation_positions_single_value(self):
        ts = multi_locus_with_mutation_example()
        for version in [2, 3]:
            tskit.dump_legacy(ts, self.temp_file, version=version)
            root = h5py.File(self.temp_file, "r+")
            root["mutations/position"][:] = 0
            root.close()
            self.assertRaises(
                tskit.DuplicatePositionsError, tskit.load_legacy, self.temp_file
            )
            tsp = tskit.load_legacy(self.temp_file, remove_duplicate_positions=True)
            self.assertEqual(tsp.num_sites, 1)
            sites = list(tsp.sites())
            self.assertEqual(sites[0].position, 0)

    def test_duplicate_mutation_positions(self):
        ts = multi_locus_with_mutation_example()
        for version in [2, 3]:
            tskit.dump_legacy(ts, self.temp_file, version=version)
            root = h5py.File(self.temp_file, "r+")
            position = np.array(root["mutations/position"])
            position[0] = position[1]
            root["mutations/position"][:] = position
            root.close()
            self.assertRaises(
                tskit.DuplicatePositionsError, tskit.load_legacy, self.temp_file
            )
            tsp = tskit.load_legacy(self.temp_file, remove_duplicate_positions=True)
            self.assertEqual(tsp.num_sites, position.shape[0] - 1)
            position_after = list(s.position for s in tsp.sites())
            self.assertEqual(list(position[1:]), position_after)


class TestErrors(TestFileFormat):
    """
    Test various API errors.
    """

    def test_v2_non_binary_records(self):
        demographic_events = [
            msprime.SimpleBottleneck(time=0.01, population=0, proportion=1)
        ]
        ts = msprime.simulate(
            sample_size=10, demographic_events=demographic_events, random_seed=1
        )
        self.assertRaises(ValueError, tskit.dump_legacy, ts, self.temp_file, 2)

    def test_unsupported_version(self):
        ts = msprime.simulate(10)
        self.assertRaises(ValueError, tskit.dump_legacy, ts, self.temp_file, version=4)
        # Cannot read current files.
        ts.dump(self.temp_file)
        # Catch Exception here because h5py throws different exceptions on py2 and py3
        self.assertRaises(Exception, tskit.load_legacy, self.temp_file)

    def test_no_version_number(self):
        root = h5py.File(self.temp_file, "w")
        root.attrs["x"] = 0
        root.close()
        self.assertRaises(ValueError, tskit.load_legacy, self.temp_file)

    def test_unknown_legacy_version(self):
        root = h5py.File(self.temp_file, "w")
        root.attrs["format_version"] = (1024, 0)  # Arbitrary unknown version
        root.close()
        self.assertRaises(ValueError, tskit.load_legacy, self.temp_file)


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
            "uuid",
        ]
        ts.dump(self.temp_file)
        store = kastore.load(self.temp_file)
        self.assertEqual(sorted(list(store.keys())), keys)

    def verify_uuid(self, ts, uuid):
        self.assertEqual(len(uuid), 36)
        # Check that the UUID is well-formed.
        parsed = _uuid.UUID("{" + uuid + "}")
        self.assertEqual(str(parsed), uuid)
        self.assertEqual(uuid, ts.file_uuid)

    def verify_dump_format(self, ts):
        ts.dump(self.temp_file)
        self.assertTrue(os.path.exists(self.temp_file))
        self.assertGreater(os.path.getsize(self.temp_file), 0)
        self.verify_keys(ts)

        store = kastore.load(self.temp_file)
        # Check the basic root attributes
        format_name = store["format/name"]
        self.assertTrue(
            np.array_equal(
                np.array(bytearray(b"tskit.trees"), dtype=np.int8), format_name
            )
        )
        format_version = store["format/version"]
        self.assertEqual(format_version[0], CURRENT_FILE_MAJOR)
        self.assertEqual(format_version[1], CURRENT_FILE_MINOR)
        self.assertEqual(ts.sequence_length, store["sequence_length"][0])
        self.assertEqual(
            str(ts.metadata_schema), "".join(store["metadata_schema"].astype("U"))
        )

        # Load another copy from file so we can check the uuid.
        other_ts = tskit.load(self.temp_file)
        self.verify_uuid(other_ts, store["uuid"].tobytes().decode())

        tables = ts.tables

        self.assertTrue(np.array_equal(tables.metadata, b"".join(store["metadata"])))
        self.assertTrue(
            np.array_equal(tables.individuals.flags, store["individuals/flags"])
        )
        self.assertTrue(
            np.array_equal(tables.individuals.location, store["individuals/location"])
        )
        self.assertTrue(
            np.array_equal(
                tables.individuals.location_offset, store["individuals/location_offset"]
            )
        )
        self.assertTrue(
            np.array_equal(tables.individuals.metadata, store["individuals/metadata"])
        )
        self.assertTrue(
            np.array_equal(
                tables.individuals.metadata_offset, store["individuals/metadata_offset"]
            )
        )
        self.assertEqual(
            str(tables.individuals.metadata_schema),
            "".join(store["individuals/metadata_schema"].astype("U")),
        )

        self.assertTrue(np.array_equal(tables.nodes.flags, store["nodes/flags"]))
        self.assertTrue(np.array_equal(tables.nodes.time, store["nodes/time"]))
        self.assertTrue(
            np.array_equal(tables.nodes.population, store["nodes/population"])
        )
        self.assertTrue(
            np.array_equal(tables.nodes.individual, store["nodes/individual"])
        )
        self.assertTrue(np.array_equal(tables.nodes.metadata, store["nodes/metadata"]))
        self.assertTrue(
            np.array_equal(tables.nodes.metadata_offset, store["nodes/metadata_offset"])
        )
        self.assertEqual(
            str(tables.nodes.metadata_schema),
            "".join(store["nodes/metadata_schema"].astype("U")),
        )

        self.assertTrue(np.array_equal(tables.edges.left, store["edges/left"]))
        self.assertTrue(np.array_equal(tables.edges.right, store["edges/right"]))
        self.assertTrue(np.array_equal(tables.edges.parent, store["edges/parent"]))
        self.assertTrue(np.array_equal(tables.edges.child, store["edges/child"]))
        self.assertTrue(np.array_equal(tables.edges.metadata, store["edges/metadata"]))
        self.assertTrue(
            np.array_equal(tables.edges.metadata_offset, store["edges/metadata_offset"])
        )
        self.assertEqual(
            str(tables.edges.metadata_schema),
            "".join(store["edges/metadata_schema"].astype("U")),
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
        self.assertTrue(
            np.array_equal(
                np.array(in_order, dtype=np.int32),
                store["indexes/edge_insertion_order"],
            )
        )
        self.assertTrue(
            np.array_equal(
                np.array(out_order, dtype=np.int32), store["indexes/edge_removal_order"]
            )
        )

        self.assertTrue(
            np.array_equal(tables.migrations.left, store["migrations/left"])
        )
        self.assertTrue(
            np.array_equal(tables.migrations.right, store["migrations/right"])
        )
        self.assertTrue(
            np.array_equal(tables.migrations.node, store["migrations/node"])
        )
        self.assertTrue(
            np.array_equal(tables.migrations.source, store["migrations/source"])
        )
        self.assertTrue(
            np.array_equal(tables.migrations.dest, store["migrations/dest"])
        )
        self.assertTrue(
            np.array_equal(tables.migrations.time, store["migrations/time"])
        )
        self.assertTrue(
            np.array_equal(tables.migrations.metadata, store["migrations/metadata"])
        )
        self.assertTrue(
            np.array_equal(
                tables.migrations.metadata_offset, store["migrations/metadata_offset"]
            )
        )
        self.assertEqual(
            str(tables.migrations.metadata_schema),
            "".join(store["migrations/metadata_schema"].astype("U")),
        )

        self.assertTrue(np.array_equal(tables.sites.position, store["sites/position"]))
        self.assertTrue(
            np.array_equal(tables.sites.ancestral_state, store["sites/ancestral_state"])
        )
        self.assertTrue(
            np.array_equal(
                tables.sites.ancestral_state_offset,
                store["sites/ancestral_state_offset"],
            )
        )
        self.assertTrue(np.array_equal(tables.sites.metadata, store["sites/metadata"]))
        self.assertTrue(
            np.array_equal(tables.sites.metadata_offset, store["sites/metadata_offset"])
        )
        self.assertEqual(
            str(tables.sites.metadata_schema),
            "".join(store["sites/metadata_schema"].astype("U")),
        )

        self.assertTrue(np.array_equal(tables.mutations.site, store["mutations/site"]))
        self.assertTrue(np.array_equal(tables.mutations.node, store["mutations/node"]))
        # Default mutation time is a NaN value so we want to check for
        # bit equality, not numeric equality
        self.assertEqual(
            tables.mutations.time.tobytes(), store["mutations/time"].tobytes()
        )
        self.assertTrue(
            np.array_equal(tables.mutations.parent, store["mutations/parent"])
        )
        self.assertTrue(
            np.array_equal(
                tables.mutations.derived_state, store["mutations/derived_state"]
            )
        )
        self.assertTrue(
            np.array_equal(
                tables.mutations.derived_state_offset,
                store["mutations/derived_state_offset"],
            )
        )
        self.assertTrue(
            np.array_equal(tables.mutations.metadata, store["mutations/metadata"])
        )
        self.assertTrue(
            np.array_equal(
                tables.mutations.metadata_offset, store["mutations/metadata_offset"]
            )
        )
        self.assertEqual(
            str(tables.mutations.metadata_schema),
            "".join(store["mutations/metadata_schema"].astype("U")),
        )

        self.assertTrue(
            np.array_equal(tables.populations.metadata, store["populations/metadata"])
        )
        self.assertTrue(
            np.array_equal(
                tables.populations.metadata_offset, store["populations/metadata_offset"]
            )
        )
        self.assertEqual(
            str(tables.populations.metadata_schema),
            "".join(store["populations/metadata_schema"].astype("U")),
        )

        self.assertTrue(
            np.array_equal(tables.provenances.record, store["provenances/record"])
        )
        self.assertTrue(
            np.array_equal(
                tables.provenances.record_offset, store["provenances/record_offset"]
            )
        )
        self.assertTrue(
            np.array_equal(tables.provenances.timestamp, store["provenances/timestamp"])
        )
        self.assertTrue(
            np.array_equal(
                tables.provenances.timestamp_offset,
                store["provenances/timestamp_offset"],
            )
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
        self.assertEqual(len(uuids), len(set(uuids)))


class TestOptionalColumns(TestFileFormat):
    """
    Checks that optional columns in the file format are correctly handled.
    """

    def test_empty_edge_metadata(self):
        ts1 = migration_example()
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        self.assertEqual(ts1.tables, ts2.tables)
        self.assertEqual(len(ts1.tables.edges.metadata), 0)

        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        del all_data["edges/metadata"]
        del all_data["edges/metadata_offset"]
        kastore.dump(all_data, self.temp_file)
        ts3 = tskit.load(self.temp_file)
        self.assertEqual(ts1.tables, ts3.tables)

    def test_empty_migration_metadata(self):
        ts1 = migration_example()
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        self.assertEqual(ts1.tables, ts2.tables)
        self.assertEqual(len(ts1.tables.migrations.metadata), 0)

        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        del all_data["migrations/metadata"]
        del all_data["migrations/metadata_offset"]
        kastore.dump(all_data, self.temp_file)
        ts3 = tskit.load(self.temp_file)
        self.assertEqual(ts1.tables, ts3.tables)

    def test_empty_mutation_time(self):
        ts1 = migration_example()
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        self.assertEqual(ts1.tables, ts2.tables)
        self.assertEqual(len(ts1.tables.mutations.metadata), 0)
        with kastore.load(self.temp_file) as store:
            all_data = dict(store)
        del all_data["mutations/time"]
        kastore.dump(all_data, self.temp_file)
        ts3 = tskit.load(self.temp_file)
        self.assertEqual(ts1.tables, ts3.tables)


class TestMixedUnknownMutation(TestFileFormat):
    def test_unknown_mutation(self):
        ts1 = migration_example()
        tables = ts1.tables
        tables.compute_mutation_times()
        mutations = tables.mutations.asdict()
        mutations["time"][0] = UNKNOWN_TIME
        tables.mutations.set_columns(**mutations)
        tables.tree_sequence().dump(self.temp_file)
        tables2 = tskit.load(self.temp_file).tables
        self.assertEqual(tables, tables2)


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
                "mutations/time",
            ]:
                data = dict(all_data)
                del data[key]
                kastore.dump(data, self.temp_file)
                with self.assertRaises(
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
                with self.assertRaises(exceptions.FileFormatError):
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
                with self.assertRaises(exceptions.FileFormatError):
                    tskit.load(self.temp_file)

            # Check for a bad offset
            data = dict(all_data)
            original_offset = data[offset_col]
            original_col = data[col]
            data[offset_col] = np.zeros_like(original_offset)
            data[col] = np.zeros(10, dtype=original_col.dtype)
            kastore.dump(data, self.temp_file)
            with self.assertRaises(exceptions.LibraryError):
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
        with self.assertRaises(exceptions.LibraryError):
            tskit.load(self.temp_file)

        data = dict(all_data)
        del data[edge_removal_order]
        kastore.dump(data, self.temp_file)
        with self.assertRaises(exceptions.LibraryError):
            tskit.load(self.temp_file)

        data = dict(all_data)
        del data[edge_insertion_order]
        kastore.dump(data, self.temp_file)
        with self.assertRaises(exceptions.LibraryError):
            tskit.load(self.temp_file)

        data = dict(all_data)
        data[edge_insertion_order] = data[edge_insertion_order][:1]
        kastore.dump(data, self.temp_file)
        with self.assertRaises(exceptions.FileFormatError):
            tskit.load(self.temp_file)

        data = dict(all_data)
        data[edge_removal_order] = data[edge_removal_order][:1]
        kastore.dump(data, self.temp_file)
        with self.assertRaises(exceptions.FileFormatError):
            tskit.load(self.temp_file)

    def test_load_empty_kastore(self):
        kastore.dump({}, self.temp_file)
        self.assertRaises(exceptions.LibraryError, tskit.load, self.temp_file)

    def test_load_non_tskit_hdf5(self):
        with h5py.File(self.temp_file, "w") as root:
            root["x"] = np.zeros(10)
        self.assertRaises(exceptions.FileFormatError, tskit.load, self.temp_file)

    def test_old_version_load_error(self):
        ts = msprime.simulate(10, random_seed=1)
        for bad_version in [(0, 1), (0, 8), (2, 0), (CURRENT_FILE_MAJOR - 1, 0)]:
            ts.dump(self.temp_file)
            with kastore.load(self.temp_file) as store:
                data = dict(store)
            data["format/version"] = np.array(bad_version, dtype=np.uint32)
            kastore.dump(data, self.temp_file)
            self.assertRaises(tskit.VersionTooOldError, tskit.load, self.temp_file)

    def test_new_version_load_error(self):
        ts = msprime.simulate(10, random_seed=1)
        for bad_version in [(CURRENT_FILE_MAJOR + j, 0) for j in range(1, 5)]:
            ts.dump(self.temp_file)
            with kastore.load(self.temp_file) as store:
                data = dict(store)
            data["format/version"] = np.array(bad_version, dtype=np.uint32)
            kastore.dump(data, self.temp_file)
            self.assertRaises(tskit.VersionTooNewError, tskit.load, self.temp_file)

    def test_format_name_error(self):
        ts = msprime.simulate(10)
        for bad_name in ["tskit.tree", "tskit.treesAndOther", "", "x" * 100]:
            ts.dump(self.temp_file)
            with kastore.load(self.temp_file) as store:
                data = dict(store)
            data["format/name"] = np.array(bytearray(bad_name.encode()), dtype=np.int8)
            kastore.dump(data, self.temp_file)
            self.assertRaises(exceptions.FileFormatError, tskit.load, self.temp_file)

    def test_load_bad_formats(self):
        # try loading a bunch of files in various formats.
        # First, check the empty file.
        self.assertRaises(EOFError, tskit.load, self.temp_file)
        # Now some ascii text
        with open(self.temp_file, "wb") as f:
            f.write(b"Some ASCII text")
        self.assertRaises(exceptions.FileFormatError, tskit.load, self.temp_file)
        # Now write 8k of random bytes
        with open(self.temp_file, "wb") as f:
            f.write(os.urandom(8192))
        self.assertRaises(exceptions.FileFormatError, tskit.load, self.temp_file)
